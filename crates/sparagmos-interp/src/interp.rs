//! The MIR interpreter: a straightforward, obviously-correct executor
//! intended as the correctness oracle for the JIT lane.
//!
//! Semantics (aligned with rustc MIR / Rust 1.100.0):
//! * `BinaryOp` arithmetic is *checked*: overflow, division by zero and
//!   invalid shifts are guest panics ([`InterpError::Panic`]).
//! * `CheckedBinaryOp` never panics on overflow; it yields
//!   `(wrapped_value, overflowed)` — exactly what a following MIR `Assert`
//!   (also implemented) turns into a panic, as rustc emits it.
//! * Integer width comes from the destination local's declared [`Ty`];
//!   without a declared width, values use the full `i128`/`u128` range.
//! * Places are resolved against the whole frame stack: a reference
//!   value names a frame by its unique id, so reads and writes through
//!   references work across calls in both directions while the target
//!   frame is live.
//! * Execution is bounded by a step limit and a call-depth limit so a
//!   looping program can never hang the oracle (or a test run).
//! * Host-panic freedom: every failure a guest program can cause is
//!   an [`InterpError`], never a host `panic!`. The `expect` /
//!   `unreachable!` sites in this crate are internal invariants only:
//!   the frame stack is non-empty by construction while the run loop
//!   executes, builtin arities are checked before argument slots are
//!   read, and the arithmetic helpers split the closed `BinOp`
//!   enumeration exhaustively. Input-shaped failures (unsupported
//!   kinds, type mismatches, uninitialized reads, resource limits)
//!   all return errors. Regression coverage lives in
//!   `tests/robustness_tests.rs`.

use crate::error::{InterpError, InterpResult};
use crate::f128;
use crate::ir::*;
use crate::value::{PanicPayload, RefProj, RefTarget, Value};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

/// Interpreter limits / configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterpConfig {
    /// Maximum number of statements + terminators executed in one run.
    pub max_steps: usize,
    /// Maximum number of simultaneously live call frames.
    pub max_call_depth: usize,
    /// The guest program's command-line arguments after argv[0]
    /// (the run-flags a test prescribes), as handed to the driver
    /// through its `--guest-arg` options. The `env::args` /
    /// `env::args_os` runtime support reads them from here;
    /// argv[0] itself is the interpreter's policy, not part of
    /// this list.
    pub guest_args: Vec<String>,
    /// The guest program's argv[0]: the path the program was
    /// invoked as. The driver sets it to the source path it was
    /// handed (the stand-in for the compiled binary's path: a
    /// natively built test binary is invoked by its own path, and
    /// the source path is the corresponding name in this model).
    /// Empty when no invocation path is known; `env::args` then
    /// yields just the guest arguments.
    pub program_path: String,
    /// The session's overflow-checks setting, as the driver
    /// resolved it from the compilation's flags. Overflow
    /// assertions (see `TerminatorKind::Assert::overflow`) fire
    /// only when this is set; the arithmetic operations themselves
    /// wrap either way, exactly as in MIR. Defaults to `true`
    /// (the debug-profile default).
    pub overflow_checks: bool,
    /// Tool interception for this run (see [`ToolInterception`]);
    /// `None` keeps the process model's standing behaviour, under
    /// which no child process ever starts.
    pub tool_interception: Option<ToolInterception>,
}

impl Default for InterpConfig {
    fn default() -> Self {
        InterpConfig {
            max_steps: 1_000_000,
            max_call_depth: 1024,
            guest_args: Vec::new(),
            program_path: String::new(),
            overflow_checks: true,
            tool_interception: None,
        }
    }
}

/// A request to run a configured tool program, handed from the
/// interpreter to the host-side tool runner (see
/// [`InterpConfig::tool_interception`]). The fields mirror the
/// interpreter's command-table entry for the invocation: the
/// program and its arguments as raw byte strings, the environment
/// entries the guest set on the command (a `None` value is a
/// removal), whether the inherited environment was cleared, and
/// the child's working directory if the guest set one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRunRequest {
    pub program: Vec<u8>,
    pub args: Vec<Vec<u8>>,
    pub argv0: Option<Vec<u8>>,
    pub envs: Vec<(Vec<u8>, Option<Vec<u8>>)>,
    pub env_clear: bool,
    pub cwd: Option<Vec<u8>>,
}

/// The outcome of one tool invocation, from which the fabricated
/// `std::process::Output` / `ExitStatus` values are built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRunOutcome {
    /// The child's exit code; negative when the child ended on a
    /// signal (the fabricated wait status encodes it as such).
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// The host-side tool runner: invoked by the interpreter when the
/// guest runs one of the configured tool programs through
/// `Command::output` / `Command::status`. An `Err` return becomes
/// the run's unsupported error, with the returned text as the
/// reason.
#[derive(Clone)]
pub struct ToolRunner(
    pub std::sync::Arc<
        dyn Fn(&ToolRunRequest) -> Result<ToolRunOutcome, String> + Send + Sync,
    >,
);

impl std::fmt::Debug for ToolRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ToolRunner(..)")
    }
}

impl PartialEq for ToolRunner {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ToolRunner {}

/// Tool interception configured for a run: the exact program byte
/// strings the interpreter routes to `runner` instead of answering
/// with the process model's named unsupported error. This is how a
/// test recipe's compiler invocations reach the driver's own
/// conversion machinery: the program named here is the compiler,
/// and running it means converting the guest's input with the
/// driver, not starting a process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolInterception {
    pub programs: Vec<Vec<u8>>,
    pub runner: ToolRunner,
}

/// One call frame: a function's locals plus where to resume.
pub(crate) struct Frame {
    /// Unique id within the run; references name their target frame
    /// by this id, so a frame keeps its identity when other frames
    /// are pushed and popped around it.
    pub(crate) id: u64,
    pub(crate) func: String,
    pub(crate) locals: Vec<Value>,
    block: usize,
    stmt: usize,
    /// Where (in the caller) the return value goes; `None` for the entry frame.
    return_dest: Option<Place>,
    /// Block to resume in the caller after return.
    return_block: Option<BasicBlock>,
    /// The panic in flight for this frame while its cleanup blocks
    /// run: stashed by the unwinder when it jumps the frame to its
    /// cleanup block, and taken back by the `Resume` terminator
    /// that ends the cleanup chain. `None` at every other time.
    unwind_payload: Option<(String, Option<PanicPayload>)>,
    /// This frame is executing its cleanup blocks (the unwinder
    /// jumped it there): a fresh panic raised directly in this
    /// frame is a panic in a cleanup and aborts the run.
    in_cleanup: bool,
}

/// Drop glue suspended while a destructor body runs. A `Drop`
/// terminator whose glue calls a destructor pushes the destructor's
/// frame and one of these; when that frame returns, the interpreter
/// resumes the remaining glue work instead of delivering a return
/// value to a caller place.
struct DropContinuation {
    /// Frame id of the destructor call this work waits on.
    callee_frame: u64,
    /// The glue session the remaining work belongs to.
    session: GlueSession,
    /// Glue sessions suspended below `session` (dyn glue runs the
    /// concrete type's glue as a nested session; see
    /// [`GlueCompletion::Resume`]), innermost last.
    sessions: Vec<GlueSession>,
}

/// The outcome of running a suspended glue session as an
/// unwinding panic's cleanup.
enum GlueCleanupFlow {
    /// The session ran to completion synchronously; unwinding
    /// continues with the returned panic.
    Completed(InterpError),
    /// The session suspended on a field destructor's frame:
    /// executing that frame resumes the session, whose
    /// completion re-raises the stashed panic into the unwinder.
    Suspended,
}

/// One run of drop glue: a work stack of (path from `root`, plan)
/// items against one dropped place, with the ADT glue table the
/// plans index into.
struct GlueSession {
    /// The dropped place the work applies to.
    root: RefTarget,
    /// Remaining work items, a stack: the last item runs next.
    work: Vec<(Vec<RefProj>, crate::ir::DropPlan)>,
    /// ADT glue table of the originating `Drop` terminator.
    adts: Vec<crate::ir::DropPlan>,
    /// What happens when the work runs out.
    completion: GlueCompletion,
}

/// How a glue session ends when its work runs out.
enum GlueCompletion {
    /// The session is a whole `Drop` terminator (or `mem::drop`
    /// call): the dropped place's own slot is cleared and the owner
    /// frame jumps to this block.
    Jump(BasicBlock),
    /// The session is the nested glue of a `dyn` value's concrete
    /// type: the session below it on the stack resumes.
    Resume,
    /// The session runs as the cleanup of a panic unwinding out of
    /// its destructor call: when the work runs out, the stashed
    /// panic (in `Interpreter::glue_unwind_panic`) continues
    /// unwinding instead of control returning to the owner frame.
    UnwindCleanup,
}

/// A place fully resolved against the frame stack: either a concrete
/// slot (frame position + local + constant projection) or, when the
/// resolution passed through a constant aggregate, the value itself.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Resolved {
    Place {
        frame_pos: usize,
        local: usize,
        projection: Vec<RefProj>,
    },
    /// A constant aggregate value and the projection into it
    /// the resolution accumulated, mirroring [`RefTarget::Value`]:
    /// the projection applies when the target is read, so an
    /// empty projection denotes the value itself and a projection
    /// ending in an index step denotes that one element — the
    /// shape an array-to-element pointer cast through a constant
    /// produces, whose sibling elements the sequence access
    /// (`seq_element_target`) reaches by adjusting the trailing
    /// step, as for the other targets.
    Value(Value, Vec<RefProj>),
    /// A place inside a `static` item's storage in the
    /// interpreter's statics store.
    Static {
        name: String,
        projection: Vec<RefProj>,
    },
    /// A place inside a heap allocation's storage in the
    /// interpreter's allocation store (a `Box`'s pointee).
    Alloc {
        id: u64,
        projection: Vec<RefProj>,
    },
}

/// One spawned thread of the cooperative thread model: the
/// callable captured at `spawn` time, and the outcome once it has
/// run. A spawned thread never runs concurrently with the thread
/// that spawned it; it runs to completion when the running thread
/// reaches a point where it would block (see
/// [`Interpreter::drain_threads`]).
pub(crate) struct ThreadSlot {
    pub callable: Option<Value>,
    /// The thread's outcome: its return value, or the panic
    /// payload it ended with (kept whole, so `join`'s error can
    /// be the `Box<dyn Any>` the guest downcasts).
    pub outcome: Option<Result<Value, PanicPayload>>,
}

/// The shared state of one `std::sync::mpsc` channel, keyed by the
/// id carried inside the `Sender` / `Receiver` handle values.
/// `senders` / `receivers` count live handles: handle creation and
/// `Clone` increment them, and dropping a handle place decrements
/// them (see the `Drop` terminator's note in `drive_frames`), which
/// is what makes disconnection observable exactly when the last
/// handle of a side goes away.
pub(crate) struct ChannelState {
    pub queue: std::collections::VecDeque<Value>,
    pub senders: usize,
    pub receivers: usize,
    pub bound: Option<usize>,
}

/// A modelled `std::process::Command` under construction (the
/// process model in `builtins.rs`): the builder state the sys
/// `Command` methods record, keyed by the id inside the handle
/// value the `Command::new` builtin returns. No child process is
/// ever started; at spawn time the program is resolved
/// deterministically and the outcome is either the `io::Error`
/// the resolution failure produces natively or a named
/// unsupported error (see `process_spawn_error` in `builtins.rs`).
/// One captured `core::fmt::Formatter` (see [`Value::Formatter`]):
/// the text a guest `fmt` implementation body has written through
/// the `Formatter` builtins, and the options of the placeholder
/// that dispatched to the body — the flag word, width and
/// precision mirror `core::fmt::FormattingOptions`, and `indent`
/// is the pretty-print (`{:#}`) indentation level the value sits
/// at, which the builder and finisher builtins align to.
#[derive(Clone, Debug)]
pub(crate) struct FmtCapture {
    pub buffer: String,
    pub flags: u32,
    pub width: u16,
    pub precision: u16,
    pub indent: usize,
    /// The rendered type whose `fmt` body this capture serves
    /// (the pointee tag the dispatch was invoked with). The
    /// finisher and builder builtins read the type's recorded
    /// Debug shape for the field type tags their `&dyn Debug`
    /// arguments have already lost.
    pub self_tag: String,
}

/// Which of core's `fmt` builders a [`FmtBuilderState`] models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FmtBuilderKind {
    List,
    Set,
    Map,
    Struct,
    Tuple,
}

/// One `core::fmt` builder under construction (see
/// [`Value::FmtBuilder`]): the formatter it writes into, its
/// struct/tuple name, and the two pieces of progress core's
/// builders track — whether any field was written, and whether
/// the writes so far succeeded (a failed write short-circuits the
/// rest, as core's `result` field does).
#[derive(Clone, Debug)]
pub(crate) struct FmtBuilderState {
    pub formatter: u64,
    pub kind: FmtBuilderKind,
    pub name: Option<String>,
    pub has_fields: bool,
    pub result_ok: bool,
}

#[derive(Clone)]
pub(crate) struct CommandState {
    /// The program path as passed to `Command::new` (raw bytes:
    /// an `OsStr` is not required to be UTF-8).
    pub program: Vec<u8>,
    /// The arguments after argv[0].
    pub args: Vec<Vec<u8>>,
    /// An explicit argv[0] (`CommandExt::arg0`); `None` means
    /// argv[0] is the program, as std defaults it.
    pub argv0: Option<Vec<u8>>,
    /// Environment entries set on the command, in order; a
    /// `None` value is a removal (`env_remove`).
    pub envs: Vec<(Vec<u8>, Option<Vec<u8>>)>,
    /// Whether the inherited environment was cleared.
    pub env_clear: bool,
    /// The working directory for the child, if set.
    pub cwd: Option<Vec<u8>>,
    /// Whether any program/argument byte string contained a NUL
    /// byte (std rejects such commands at spawn with an
    /// `InvalidInput` error).
    pub saw_nul: bool,
}

/// The channel id carried by an mpsc handle value, if the value
/// has the handle shape (a one-field aggregate holding the id).
pub(crate) fn channel_handle_id(value: &Value) -> Option<u64> {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
            Value::Uint(id) => Some(*id as u64),
            _ => None,
        },
        _ => None,
    }
}

/// The allocation a `Box` value points at, if the value has the
/// shape the construction builtins produce (see
/// [`Interpreter::new_box`]): the allocation id and the
/// projection the box's pointer itself carries (empty for a
/// freshly built box; a pointer extracted from a box and stored
/// on can carry one). Only a box's own field 0 is ever an
/// allocation pointer in this model, so this shape test is
/// exact: a one-field aggregate whose field is an allocation
/// reference is a built box, and projecting its field 0 denotes
/// the pointee, not the pointer slot.
/// The `mem::Alignment` value for a byte alignment, in the
/// shape the conversion produces for a `usize` -> `Alignment`
/// transmute: the variant whose discriminant is the alignment
/// (the variants are the powers of two in order), wrapped as
/// the struct's single field. An alignment with no variant
/// (not a power of two, or beyond the enum's range) keeps the
/// plain integer shape.
pub(crate) fn alignment_value(align: u128) -> Value {
    if align >= 1 && align.is_power_of_two() {
        let exponent = align.trailing_zeros();
        if exponent <= 26 {
            return Value::Enum {
                variant: 0,
                fields: vec![Value::Enum {
                    variant: exponent,
                    fields: Vec::new(),
                }],
            };
        }
    }
    Value::Uint(align)
}

pub(crate) fn box_alloc_pointer(value: &Value) -> Option<(u64, Vec<RefProj>)> {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
            Value::Ref(RefTarget::Alloc { id, projection }) => {
                Some((*id, projection.clone()))
            }
            _ => None,
        },
        _ => None,
    }
}

/// The allocation id a reference-counted handle value names,
/// if the value has the handle shape: an aggregate (the `Rc` /
/// `Arc` / `Weak` struct) whose first field is a one-field
/// `NonNull` aggregate holding a reference to the allocation's
/// storage. Any further fields (the zero-sized phantom and
/// allocator fields of `Rc` / `Arc`) do not name the allocation.
pub(crate) fn rc_handle_id(value: &Value) -> Option<u64> {
    let fields = match value {
        Value::Enum { fields, .. } if !fields.is_empty() => fields,
        _ => return None,
    };
    let inner = match &fields[0] {
        Value::Enum { fields, .. } if fields.len() == 1 => fields,
        _ => return None,
    };
    match &inner[0] {
        Value::Ref(RefTarget::Alloc { id, projection }) if projection.is_empty() => Some(*id),
        _ => None,
    }
}

/// The allocation a hand-built `RawWaker` aggregate's data
/// pointer denotes, when the waker was made from an `Rc` / `Arc`
/// (the shape `ArcWake::into_waker` and its kin produce): the
/// aggregate's fields are (data, vtable), and the data pointer
/// is the one `Rc::into_raw` / `Arc::into_raw` returned -- a
/// reference to the value field (field 2) of the handle's
/// `RcInner` / `ArcInner` allocation. Releasing such a waker is
/// releasing that handle, which is what its vtable's drop
/// function does. `None` for any other shape.
pub(crate) fn raw_waker_alloc_id(value: &Value) -> Option<u64> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    let [data, vtable] = fields.as_slice() else {
        return None;
    };
    if !matches!(vtable, Value::Ref(_)) {
        return None;
    }
    match data {
        Value::Ref(RefTarget::Alloc { id, projection })
            if projection.as_slice() == [RefProj::Field(2)] =>
        {
            Some(*id)
        }
        _ => None,
    }
}

/// The slice length an unsizing cast reads from a pointee
/// value: the element count of the pointee's own array tail. A
/// custom DST value (an ADT whose trailing field is the unsized
/// part) carries the tail's length, recursively; an empty array
/// aggregate is the unit value, of length zero; and a scalar
/// pointee counts as one element — the one-element array view
/// of an allocation a reinterpreted pointer denotes
/// (`Box::into_boxed_slice` reinterprets a `Box<T>` allocation
/// as a `[T; 1]`). Anything else has no slice length.
fn unsized_tail_len(pointee: &Value) -> Option<u128> {
    match pointee {
        Value::Tuple(items) => Some(items.len() as u128),
        Value::Bytes(b) => Some(b.len() as u128),
        Value::Str(s) => Some(s.len() as u128),
        Value::FatRef { len, .. } => Some(*len),
        Value::Unit => Some(0),
        Value::Enum { fields, .. } => fields.last().and_then(unsized_tail_len),
        Value::Bool(_)
        | Value::Int(_)
        | Value::Uint(_)
        | Value::Float(_)
        | Value::Float128(_) => Some(1),
        _ => None,
    }
}

/// Whether a pointee's unsized length (see
/// [`unsized_tail_len`]) rests on the pointee's own array tail
/// rather than on the one-element view of a scalar: the
/// distinction the ADT field search in `unsize_to_slice` ranks
/// candidates by.
fn unsized_tail_is_array(pointee: &Value) -> bool {
    match pointee {
        Value::Tuple(_)
        | Value::Bytes(_)
        | Value::Str(_)
        | Value::FatRef { .. }
        | Value::Unit => true,
        Value::Enum { fields, .. } => {
            fields.last().is_some_and(unsized_tail_is_array)
        }
        _ => false,
    }
}

/// The count leaf of one reference count of an `RcInner` /
/// `ArcInner` aggregate: field `field` (0 = strong, 1 = weak) is
/// a `Cell<usize>` / `AtomicUsize` wrapper around an
/// `UnsafeCell<usize>` holding the count.
fn count_leaf(inner: &Value, field: usize) -> Option<u128> {
    fn unwrap1(v: &Value) -> Option<&Value> {
        match v {
            Value::Enum { fields, .. } if fields.len() == 1 => Some(&fields[0]),
            _ => None,
        }
    }
    let fields = match inner {
        Value::Enum { fields, .. } => fields,
        _ => return None,
    };
    let leaf = unwrap1(unwrap1(fields.get(field)?)?)?;
    match leaf {
        Value::Uint(n) => Some(*n),
        _ => None,
    }
}

/// The mutable count leaf of one reference count (see
/// [`count_leaf`]).
fn count_leaf_mut(inner: &mut Value, field: usize) -> Option<&mut u128> {
    fn unwrap1_mut(v: &mut Value) -> Option<&mut Value> {
        match v {
            Value::Enum { fields, .. } if fields.len() == 1 => fields.first_mut(),
            _ => None,
        }
    }
    let fields = match inner {
        Value::Enum { fields, .. } => fields,
        _ => return None,
    };
    let leaf = unwrap1_mut(unwrap1_mut(fields.get_mut(field)?)?)?;
    match leaf {
        Value::Uint(n) => Some(n),
        _ => None,
    }
}

/// The interpreter state for one program.
pub struct Interpreter<'a> {
    program: &'a Program,
    config: InterpConfig,
    steps: usize,
    next_frame_id: u64,
    /// Buffers of abstract strings created by
    /// `String::with_capacity` under a guest allocator, keyed by
    /// the (frame, local) place the string was written to: the
    /// allocation pointer and capacity the guest allocator
    /// handed out, which the abstract string value does not
    /// carry. The matching deallocation runs when that place's
    /// drop glue finishes. A string that moves to another place
    /// loses the association (its buffer then leaks in the
    /// model, as abstract-string buffers did before).
    string_buffers: std::cell::RefCell<
        std::collections::HashMap<(u64, usize), (Value, u128)>,
    >,
    /// The buffer a `String::with_capacity` call just allocated
    /// through the guest allocator, waiting for the call's
    /// destination place to be recorded in `string_buffers`.
    pub(crate) pending_string_buffer:
        std::cell::RefCell<Option<(Value, u128)>>,
    /// Pointee layouts of boxes the allocation builtins created,
    /// by allocation id: a dropped box's storage is freed through
    /// the guest deallocator with the pointee's layout, which the
    /// box value itself does not carry.
    box_layouts: std::cell::RefCell<std::collections::HashMap<u64, (u128, u128)>>,
    /// Whether the lazy stdout buffer allocation has been
    /// modelled (see `note_stdout_use`): std's stdout handle
    /// allocates its line buffer through the global allocator on
    /// first use, which a guest allocator observes.
    stdout_buffer_modelled: std::cell::Cell<bool>,
    /// Everything the guest wrote to its stdout (via the runtime
    /// print support) during the most recent run.
    output: String,
    /// Everything the guest wrote to its stderr (via the runtime
    /// print support) during the most recent run.
    error_output: String,
    /// Drop glue suspended on destructor calls (see
    /// [`DropContinuation`]); a stack, innermost last.
    drop_continuations: Vec<DropContinuation>,
    /// The panic stashed while a glue session runs as that
    /// panic's cleanup (see `GlueCompletion::UnwindCleanup`):
    /// taken back when the session's work runs out.
    glue_unwind_panic: Option<(String, Option<PanicPayload>)>,
    /// The guest's panic hook, installed by `std::panic::set_hook`
    /// (the boxed `dyn Fn` value as the guest handed it over);
    /// `None` is the default hook, which this runtime does not
    /// call (its observable output is the driver's own panic
    /// report). Interior-mutable: most execution paths hold only
    /// `&self`.
    panic_hook: std::cell::RefCell<Option<Value>>,
    /// The caller-location stamp of the panic currently being
    /// raised, stashed by the panic entry-point builtins (they
    /// receive it as a trailing argument) for the hook's
    /// `PanicHookInfo`.
    last_panic_stamp: std::cell::RefCell<Option<Value>>,
    /// Whether the panic hook is currently running (a panic
    /// raised by the hook itself does not re-enter it).
    in_panic_hook: bool,
    /// The drive depth at which the in-flight panic already
    /// fired the hook, so outer drive regions the panic
    /// propagates through do not fire it again. Cleared when a
    /// panic is consumed (caught, or turned into a thread
    /// outcome).
    hook_fired_depth: Option<usize>,
    /// Set by the `Resume` terminator right before it returns the
    /// in-flight panic: the unwinder (see [`Interpreter::drive`])
    /// reads and clears it to tell a completed cleanup (the panic
    /// continues in the caller) from a fresh panic raised directly
    /// in a cleanup frame (which aborts).
    resume_signaled: bool,
    /// The debugger session's control state, when this run is
    /// driven by the debugger interface ([`crate::debug`]): the
    /// drive loop consults it at every source-line change. `None`
    /// for every ordinary run, which then never pays for it beyond
    /// the one check.
    pub(crate) debug_ctl: Option<crate::debug::DebugControl>,
    /// The program-level statics store: one storage per `static`
    /// item, keyed by the static's program key (see
    /// [`crate::ir::Constant::StaticRef`]). A static's storage is
    /// seeded lazily from the initializer constant its first
    /// reference carries. Interior-mutable because most execution
    /// paths hold only `&self`.
    statics: std::cell::RefCell<HashMap<String, Value>>,
    /// Statics whose seeding is already in flight further up the
    /// stack (an initializer that mentions its own static through
    /// a reference): a nested seeding request for one of these
    /// must not start over — the outer seeding fills the storage,
    /// and the nested reference simply names it.
    statics_seeding: std::cell::RefCell<HashSet<String>>,
    /// Spawned threads of the cooperative thread model (see
    /// [`ThreadSlot`]), keyed by thread id. The entry thread is
    /// id 1, matching `std`'s numbering of the main thread.
    threads: std::cell::RefCell<HashMap<u64, ThreadSlot>>,
    /// The id the next spawned thread gets.
    next_thread: std::cell::RefCell<u64>,
    /// Thread-local destructors registered through std's
    /// destructor-registration entry point (see the
    /// `TlsRegisterDtor` builtin), per thread id, in registration
    /// order: the storage reference and the destructor callable
    /// the registration carried. They run when the thread's code
    /// finishes (see [`Interpreter::run_thread`]).
    tls_dtors: std::cell::RefCell<HashMap<u64, Vec<(Value, Value)>>>,
    /// The thread whose code is currently executing (the entry
    /// thread outside any spawned thread's run).
    current_thread: std::cell::Cell<u64>,
    /// mpsc channel states (see [`ChannelState`]), keyed by the id
    /// carried inside the handle values.
    channels: std::cell::RefCell<HashMap<u64, ChannelState>>,
    /// The id the next created channel gets.
    next_channel: std::cell::RefCell<u64>,
    /// Modelled process commands under construction (see
    /// [`CommandState`]), keyed by the id inside the handle
    /// values. Interior-mutable for the same reason as the
    /// channel table.
    commands: std::cell::RefCell<HashMap<u64, CommandState>>,
    /// The id the next created command gets.
    next_command: std::cell::RefCell<u64>,
    /// Captured `core::fmt::Formatter` states (see
    /// [`FmtCapture`]), keyed by the id inside the formatter
    /// handle values.
    formatters: std::cell::RefCell<HashMap<u64, FmtCapture>>,
    /// The id the next captured formatter gets.
    next_formatter: std::cell::RefCell<u64>,
    /// `core::fmt` builder states (see [`FmtBuilderState`]),
    /// keyed by the id inside the builder handle values.
    fmt_builders: std::cell::RefCell<HashMap<u64, FmtBuilderState>>,
    /// The id the next created builder gets.
    next_fmt_builder: std::cell::RefCell<u64>,
    /// The program-level heap allocation store: one storage per
    /// live `Box` allocation, keyed by the allocation id the box
    /// value carries as its single field (see
    /// [`RefTarget::Alloc`]). A box's pointee lives here, so the
    /// box value itself is just the pointer: copying it copies the
    /// id, and every copy denotes the same storage. An entry is
    /// removed when the owning box's drop glue completes (a box
    /// consumed by `Box::into_raw` is never dropped, so its entry
    /// outlives the call, as a leaked allocation does). Interior-
    /// mutable because most execution paths hold only `&self`.
    allocs: std::cell::RefCell<HashMap<u64, Value>>,
    /// Allocations created by the raw allocator builtins
    /// (`__rust_alloc` and friends): their content is an
    /// element tuple (or a zeroed byte block), and a pointer
    /// to the block denotes its first element. Every other
    /// allocation holds one whole pointee value (a box's
    /// contents, a materialised constant), where the same
    /// pointer denotes the value itself.
    elem_blocks: std::cell::RefCell<std::collections::HashSet<u64>>,
    /// The id the next created allocation gets.
    next_alloc: std::cell::Cell<u64>,
    /// Pre-decoded bodies for the run loop (see [`crate::decode`]),
    /// decoded lazily per body and keyed by body address.
    decoded: std::cell::RefCell<
        std::collections::HashMap<usize, std::rc::Rc<crate::decode::DecodedBody>>,
    >,
}

/// The synthetic address of a pointer value (see
/// [`crate::ir::Rvalue::PtrToInt`]): an FNV-1a hash of the reference
/// target, mapped into a non-zero, 4096-aligned range. The same
/// target always yields the same address within a run, so equality,
/// null and alignment checks on transmuted pointers behave as they
/// do natively; distinct targets collide only on a hash collision.
pub(crate) fn synthetic_address(value: &Value) -> Option<u128> {
    // An address-only pointer's address is the stored integer
    // itself (address 0 is the null address).
    if let Value::Ref(RefTarget::Address(addr)) = value {
        return Some(*addr as u128);
    }
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
    if matches!(value, Value::Ref(RefTarget::Null)) {
        // The null pointer's address is 0, as natively: `is_null`
        // and address comparisons depend on it.
        return Some(0);
    }
    let key = match value {
        Value::Ref(target) => format!("ref:{target:?}"),
        Value::FatRef { base, len } => format!("fat:{base:?}:{len}"),
        Value::FnPtr(name) => format!("fn:{name}"),
        // A non-capturing closure value is a function pointer (it
        // coerces to one); its address identity is its body.
        Value::Closure { def, env } if env.is_empty() => format!("closure:{def}"),
        // A dyn fat pointer's data address is its base pointer's
        // identity together with the concrete type (two dyn values
        // over the same base but different concrete types are
        // different values' pointers in this model).
        Value::DynRef { base, concrete, .. } => format!("dyn:{base:?}:{concrete}"),
        _ => return None,
    };
    let h = fnv1a(key.as_bytes());
    Some(0x1_0000_0000u128 + u128::from(h % 0x10_0000) * 0x1000)
}

/// The Self type of an impl-qualified method name
/// (`<i32 as std::default::Default>::default` gives `i32`):
/// the text between the leading `<` and the ` as ` separator at
/// the same bracket depth. `None` for any other name shape.
pub(crate) fn impl_qualified_self_type(name: &str) -> Option<String> {
    let inner = name.strip_prefix('<')?;
    let bytes = inner.as_bytes();
    let mut depth = 1usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    return None;
                }
            }
            b' ' if depth == 1 && inner[i..].starts_with(" as ") => {
                return Some(inner[..i].trim().to_string());
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// The value-semantics trait method an impl-path spelling of
/// a call denotes: the name ends in `>::METHOD` and names the
/// trait either qualified (`... as std::clone::Clone>::clone`)
/// or in an impl header (`<impl std::clone::Clone for i32>`).
/// `None` for any other shape.
/// Whether a value is a caller-location stamp: the `Location`
/// aggregate — (file, line, column) — or a reference to it, as
/// the driver-appended stamp of a `#[track_caller]` call
/// evaluates.
fn is_caller_location_value(v: &Value) -> bool {
    let inner = match v {
        Value::Ref(RefTarget::Value(inner, _)) => &**inner,
        other => other,
    };
    let Value::Enum { fields, .. } = inner else {
        return false;
    };
    if fields.len() < 3 {
        return false;
    }
    let file_is_str = match &fields[0] {
        Value::Str(_) => true,
        Value::Enum { fields: inner, .. } => {
            matches!(inner.first(), Some(Value::Str(_)))
        }
        _ => false,
    };
    file_is_str
        && matches!(fields[1], Value::Int(_) | Value::Uint(_))
        && matches!(fields[2], Value::Int(_) | Value::Uint(_))
}

fn impl_path_trait_method(name: &str) -> Option<crate::builtins::Builtin> {
    use crate::builtins::Builtin;
    const METHODS: &[(&str, &str, Builtin)] = &[
        ("PartialEq", "eq", Builtin::PartialEqEq),
        ("PartialEq", "ne", Builtin::PartialEqNe),
        ("PartialOrd", "partial_cmp", Builtin::PartialCmp),
        ("PartialOrd", "lt", Builtin::PartialOrdRel(crate::builtins::RelOp::Lt)),
        ("PartialOrd", "le", Builtin::PartialOrdRel(crate::builtins::RelOp::Le)),
        ("PartialOrd", "gt", Builtin::PartialOrdRel(crate::builtins::RelOp::Gt)),
        ("PartialOrd", "ge", Builtin::PartialOrdRel(crate::builtins::RelOp::Ge)),
        ("Ord", "cmp", Builtin::OrdCmp),
        ("Clone", "clone", Builtin::CloneClone),
    ];
    for (trait_seg, method, kind) in METHODS {
        if !name.ends_with(&format!(">::{method}")) {
            continue;
        }
        let qualified = format!("::{trait_seg}>::{method}");
        let impl_header = format!("::{trait_seg} for ");
        if name.contains(&qualified) || name.contains(&impl_header) {
            return Some(*kind);
        }
    }
    None
}

/// The primitive arithmetic trait method an impl-qualified
/// spelling of a call denotes (`<i32 as std::ops::Add>::add`,
/// `<i32 as std::ops::Add<i32>>::add`): the operator traits on
/// the primitive types, whose core impls are the operations
/// themselves. The Self type must be a primitive the value
/// model computes on — for any other Self type the impl body
/// is the only honest source, so the name does not qualify.
/// The plain trait path (`std::ops::Add::add`) does not
/// qualify either: it names no Self type, and the values carry
/// no width to compute with. `None` for any other shape.
fn impl_path_primitive_op(name: &str) -> Option<crate::builtins::Builtin> {
    use crate::builtins::Builtin;
    use crate::ir::{BinOp, UnOp};
    const BINOPS: &[(&str, &str, BinOp)] = &[
        ("Add", "add", BinOp::Add),
        ("Sub", "sub", BinOp::Sub),
        ("Mul", "mul", BinOp::Mul),
        ("Div", "div", BinOp::Div),
        ("Rem", "rem", BinOp::Rem),
        ("BitAnd", "bitand", BinOp::BitAnd),
        ("BitOr", "bitor", BinOp::BitOr),
        ("BitXor", "bitxor", BinOp::BitXor),
        ("Shl", "shl", BinOp::Shl),
        ("Shr", "shr", BinOp::Shr),
    ];
    const UNOPS: &[(&str, &str, UnOp)] = &[
        ("Neg", "neg", UnOp::Neg),
        ("Not", "not", UnOp::Not),
    ];
    const PRIMITIVE_SELF: &[&str] = &[
        "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64",
        "u128", "usize", "f32", "f64", "bool",
    ];
    let self_ty = impl_qualified_self_type(name)?;
    if !PRIMITIVE_SELF.contains(&self_ty.as_str()) {
        return None;
    }
    // The trait segment: the last path segment before the
    // closing `>::`, generic arguments aside.
    let close = name.rfind(">::")?;
    let segment = name[..close].rsplit("::").next()?;
    let trait_name = segment.split('<').next()?;
    for (trait_seg, method, op) in BINOPS {
        if trait_name == *trait_seg && name.ends_with(&format!("::{method}")) {
            return Some(Builtin::PrimitiveBinOp(*op));
        }
    }
    for (trait_seg, method, op) in UNOPS {
        if trait_name == *trait_seg && name.ends_with(&format!("::{method}")) {
            return Some(Builtin::PrimitiveUnOp(*op));
        }
    }
    None
}

impl<'a> Interpreter<'a> {
    pub fn new(program: &'a Program) -> Self {
        Interpreter::with_config(program, InterpConfig::default())
    }

    /// The guest program's command-line arguments after argv[0],
    /// as configured for this run (see [`InterpConfig::guest_args`]).
    pub fn guest_args(&self) -> &[String] {
        &self.config.guest_args
    }

    /// The guest program's argv[0] (see [`InterpConfig::program_path`]).
    pub fn program_path(&self) -> &str {
        &self.config.program_path
    }

    /// Whether the guest was compiled with overflow checks
    /// (see the `overflow_checks` field of `InterpConfig`).
    pub fn overflow_checks(&self) -> bool {
        self.config.overflow_checks
    }

    /// The tool interception configured for this run, if any
    /// (see [`InterpConfig::tool_interception`]).
    pub fn tool_interception(&self) -> Option<&ToolInterception> {
        self.config.tool_interception.as_ref()
    }

    pub fn with_config(program: &'a Program, config: InterpConfig) -> Self {
        Interpreter {
            program,
            config,
            steps: 0,
            next_frame_id: 0,
            string_buffers: std::cell::RefCell::new(
                std::collections::HashMap::new(),
            ),
            pending_string_buffer: std::cell::RefCell::new(None),
            box_layouts: std::cell::RefCell::new(std::collections::HashMap::new()),
            stdout_buffer_modelled: std::cell::Cell::new(false),
            output: String::new(),
            error_output: String::new(),
            drop_continuations: Vec::new(),
            glue_unwind_panic: None,
            resume_signaled: false,
            panic_hook: std::cell::RefCell::new(None),
            last_panic_stamp: std::cell::RefCell::new(None),
            in_panic_hook: false,
            hook_fired_depth: None,
            statics: std::cell::RefCell::new(HashMap::new()),
            statics_seeding: std::cell::RefCell::new(HashSet::new()),
            threads: std::cell::RefCell::new(HashMap::new()),
            next_thread: std::cell::RefCell::new(2),
            tls_dtors: std::cell::RefCell::new(HashMap::new()),
            current_thread: std::cell::Cell::new(1),
            channels: std::cell::RefCell::new(HashMap::new()),
            next_channel: std::cell::RefCell::new(1),
            commands: std::cell::RefCell::new(HashMap::new()),
            next_command: std::cell::RefCell::new(1),
            formatters: std::cell::RefCell::new(HashMap::new()),
            next_formatter: std::cell::RefCell::new(1),
            fmt_builders: std::cell::RefCell::new(HashMap::new()),
            next_fmt_builder: std::cell::RefCell::new(1),
            allocs: std::cell::RefCell::new(HashMap::new()),
            elem_blocks: std::cell::RefCell::new(std::collections::HashSet::new()),
            next_alloc: std::cell::Cell::new(1),
            decoded: std::cell::RefCell::new(std::collections::HashMap::new()),
            debug_ctl: None,
        }
    }

    /// The pre-decoded body cache (see [`crate::decode`]).
    pub(crate) fn decoded_cache(
        &self,
    ) -> &std::cell::RefCell<
        std::collections::HashMap<usize, std::rc::Rc<crate::decode::DecodedBody>>,
    > {
        &self.decoded
    }

    /// Steps consumed by the most recent (or current) run.
    pub fn steps_used(&self) -> usize {
        self.steps
    }

    /// The guest stdout captured during the most recent run.
    /// The program being executed (the builtin dispatch reads
    /// program-level tables such as the ADT layout table through
    /// this accessor).
    pub(crate) fn program(&self) -> &Program {
        self.program
    }

    pub fn output(&self) -> &str {
        &self.output
    }

    /// The guest stderr captured during the most recent run.
    pub fn error_output(&self) -> &str {
        &self.error_output
    }

    pub(crate) fn push_error_output(&mut self, text: &str) {
        self.error_output.push_str(text);
    }

    pub(crate) fn push_output(&mut self, text: &str) {
        self.output.push_str(text);
    }

    /// The current length of the guest's captured stdout, as a
    /// mark for [`Interpreter::drain_output_since`].
    pub(crate) fn output_len(&self) -> usize {
        self.output.len()
    }

    /// The current length of the guest's captured stderr, as a
    /// mark for [`Interpreter::drain_error_output_since`].
    pub(crate) fn error_output_len(&self) -> usize {
        self.error_output.len()
    }

    /// Remove and return everything written to the guest's stdout
    /// since `mark` (see [`Interpreter::output_len`]). The libtest
    /// harness model captures each test's output this way.
    pub(crate) fn drain_output_since(&mut self, mark: usize) -> String {
        if mark >= self.output.len() {
            return String::new();
        }
        self.output.split_off(mark)
    }

    /// Remove and return everything written to the guest's stderr
    /// since `mark` (see [`Interpreter::error_output_len`]).
    pub(crate) fn drain_error_output_since(&mut self, mark: usize) -> String {
        if mark >= self.error_output.len() {
            return String::new();
        }
        self.error_output.split_off(mark)
    }

    fn tick(&mut self) -> InterpResult<()> {
        self.steps += 1;
        if self.steps > self.config.max_steps {
            return Err(InterpError::StepLimitExceeded {
                limit: self.config.max_steps,
            });
        }
        Ok(())
    }

    /// Consume one step of the execution budget. Builtin loops
    /// that drive modelled machinery without executing guest
    /// statements (the iterator consumers) tick once per element,
    /// so consuming an unbounded iterator still meets the step
    /// limit instead of looping forever.
    pub(crate) fn tick_step(&mut self) -> InterpResult<()> {
        self.tick()
    }

    fn fresh_frame_id(&mut self) -> u64 {
        let id = self.next_frame_id;
        self.next_frame_id += 1;
        id
    }

    /// Run `entry` with `args` and return the value of its return place.
    pub fn run(&mut self, entry: &str, args: Vec<Value>) -> InterpResult<Value> {
        self.steps = 0;
        self.next_frame_id = 0;
        self.output.clear();
        self.error_output.clear();
        let body = self
            .program
            .get(entry)
            .ok_or_else(|| self.missing_body_error(entry))?;
        if args.len() != body.arg_count {
            return Err(InterpError::type_mismatch(
                format!("call to `{entry}`"),
                format!("{} argument(s)", body.arg_count),
                format!("{} argument(s)", args.len()),
            ));
        }
        let mut locals = vec![Value::Uninit; body.local_count()];
        for (i, arg) in args.into_iter().enumerate() {
            locals[i + 1] = arg;
        }
        let entry_id = self.fresh_frame_id();
        let mut frames: Vec<Frame> = vec![Frame {
            id: entry_id,
            func: entry.to_string(),
            locals,
            block: 0,
            stmt: 0,
            return_dest: None,
            return_block: None,
            unwind_payload: None,
            in_cleanup: false,
        }];
        let result = self.drive_frames(&mut frames, 0);
        // Program exit runs the entry thread's registered TLS
        // destructors (the registrations wrap
        // `__cxa_thread_atexit`, whose entries run at exit),
        // whatever the entry's outcome was.
        match self.run_tls_dtors(&mut frames, 1) {
            Ok(()) => result,
            Err(e) => Err(e),
        }
    }

    /// Drive the frame stack until the frame at depth `base`
    /// returns, yielding its return value. The caller's stack is
    /// always restored to the depth this drive started from: on
    /// success the loop stops exactly there; on an error — above
    /// all a guest panic, which a thread's join must observe as
    /// the thread's outcome rather than as a failure of the whole
    /// run — the frames above the base are dead (a panic has
    /// already unwound them, running their cleanups; see
    /// [`Interpreter::drive`]) and are discarded.
    fn drive_frames(&mut self, frames_out: &mut Vec<Frame>, base: usize) -> InterpResult<Value> {
        let mut frames: Vec<Frame> = std::mem::take(frames_out);
        let result = self.drive(&mut frames, base);
        frames.truncate(base);
        *frames_out = frames;
        result
    }

    /// Drive the frame stack with unwinding: run the raw loop
    /// ([`Interpreter::drive_loop`]); when it reports a guest
    /// panic, unwind the stack frame by frame down to `stop_depth`
    /// before reporting the panic. Each frame the panic passes
    /// through runs the cleanup its current block prescribes (see
    /// [`crate::ir::Body::unwind_action`]): a cleanup block chain
    /// (the `Drop` terminators of the locals live at the panic
    /// point, in MIR's own order) runs with the panic stashed in
    /// the frame, and its closing `Resume` pops the frame and
    /// continues the unwind in the caller at its call site. A
    /// frame with no cleanup is simply discarded. A panic that may
    /// not unwind — one raised directly in a cleanup frame, or one
    /// whose block prescribes termination — aborts the run
    /// ([`InterpError::Abort`]), as a native double panic does.
    /// Non-panic errors never unwind: they are oracle failures,
    /// not guest control flow.
    fn drive(
        &mut self,
        frames: &mut Vec<Frame>,
        stop_depth: usize,
    ) -> InterpResult<Value> {
        let program: &'a Program = self.program;
        loop {
            match self.drive_loop(frames, stop_depth) {
                Ok(value) => return Ok(value),
                Err(e) if !matches!(e, InterpError::Panic { .. }) => {
                    return Err(e);
                }
                Err(InterpError::Panic { message, payload }) => {
                    let resumed = std::mem::take(&mut self.resume_signaled);
                    if !resumed {
                        self.fire_panic_hook(frames, stop_depth, &message, &payload);
                    }
                    let mut panic = InterpError::Panic { message, payload };
                    // Whether the panic was raised directly in the
                    // top frame (as opposed to resuming into it from
                    // a callee this unwinder already popped).
                    let mut direct = true;
                    // Unwind one frame at a time until the panic
                    // leaves this drive region or a cleanup block
                    // takes over execution.
                    loop {
                        if frames.len() <= stop_depth {
                            return Err(panic);
                        }
                        let (func, block, in_cleanup) = {
                            let frame = frames.last().expect("frame");
                            (frame.func.clone(), frame.block, frame.in_cleanup)
                        };
                        if direct && in_cleanup && !resumed {
                            // A fresh panic raised by the cleanup
                            // frame itself (not one resuming into
                            // it from a callee): panicking while
                            // unwinding aborts, natively.
                            let InterpError::Panic { message, .. } = &panic else {
                                unreachable!("panic is a Panic")
                            };
                            return Err(InterpError::Abort {
                                message: format!(
                                    "panic `{message}` raised in a cleanup of `{func}`: \
                                     panicking while unwinding aborts the run"
                                ),
                            });
                        }
                        let body = program.get(&func);
                        let action = body
                            .map(|b| b.unwind_action(block))
                            .unwrap_or(crate::ir::UnwindAction::Continue);
                        match action {
                            crate::ir::UnwindAction::Cleanup(target) => {
                                let Some(body) = body else {
                                    unreachable!("cleanup action implies a body")
                                };
                                let InterpError::Panic { message, payload } = panic
                                else {
                                    unreachable!("panic is a Panic")
                                };
                                let frame = frames.last_mut().expect("frame");
                                frame.unwind_payload = Some((message, payload));
                                frame.in_cleanup = true;
                                Self::jump(frames, body, target)?;
                                break;
                            }
                            crate::ir::UnwindAction::Continue => {
                                // A destructor frame unwound out of
                                // with glue still suspended under it:
                                // the remaining glue is the cleanup
                                // of the native `drop_in_place` whose
                                // destructor panicked, so it runs
                                // before the panic moves on (the
                                // destructor frame itself is popped
                                // first; the glue works on the owner
                                // frame below it).
                                let cont_pos = frames.last().and_then(|top| {
                                    self.drop_continuations
                                        .iter()
                                        .rposition(|c| c.callee_frame == top.id)
                                });
                                if let Some(pos) = cont_pos {
                                    let cont = self.drop_continuations.remove(pos);
                                    self.pop_unwind_frame(frames);
                                    match self.run_glue_cleanup(frames, cont, panic)? {
                                        GlueCleanupFlow::Completed(p) => {
                                            panic = p;
                                        }
                                        GlueCleanupFlow::Suspended => break,
                                    }
                                } else {
                                    self.pop_unwind_frame(frames);
                                }
                                // The caller frame is now on top and
                                // the panic resumes into it: that is
                                // a resumption, not a fresh panic in
                                // the caller, even when the caller is
                                // itself running cleanups, so the
                                // direct-panic guard no longer
                                // applies; the caller's own call-site
                                // action decides.
                                direct = false;
                                continue;
                            }
                            crate::ir::UnwindAction::Terminate
                            | crate::ir::UnwindAction::Unreachable => {
                                let InterpError::Panic { message, .. } = &panic else {
                                    unreachable!("panic is a Panic")
                                };
                                return Err(InterpError::Abort {
                                    message: format!(
                                        "panic `{message}` may not unwind past `{func}`: \
                                         the run aborts"
                                    ),
                                });
                            }
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Pop one frame during unwinding, discarding any suspended
    /// drop glue that belonged to it. The unwinder runs a popped
    /// destructor frame's glue as the panic's cleanup before
    /// calling this (see `drive`); what remains here is glue whose
    /// session can no longer run (its destructors never resume:
    /// the frame they would have resumed into is gone).
    fn pop_unwind_frame(&mut self, frames: &mut Vec<Frame>) {
        let Some(frame) = frames.pop() else {
            return;
        };
        self.drop_continuations
            .retain(|cont| cont.callee_frame != frame.id);
    }

    /// The run loop: execute until the frame stack shrinks to
    /// `stop_depth`, then return the value the frame that was on
    /// top returned. The top-level [`Interpreter::run`] drives with
    /// `stop_depth == 0` (the entry frame's return ends the run);
    /// runtime support that calls back into guest code (the
    /// `catch_unwind` intrinsic) drives a nested region with the
    /// current depth as its stop, so the nested call's return value
    /// comes back here instead of being delivered to a caller
    /// place. Frames pushed for a nested region carry no return
    /// destination (see `start_detached_call`); stack discipline
    /// guarantees such a frame returns exactly when the stack is
    /// back at `stop_depth + 1`.
    fn drive_loop(
        &mut self,
        mut frames: &mut Vec<Frame>,
        stop_depth: usize,
    ) -> InterpResult<Value> {
        // The current frame's body, borrowed from the program for
        // the whole step: bodies never change during a run, so the
        // loop resolves the top frame's body once per frame switch
        // (keyed by the frame's unique id; a frame's function never
        // changes over its life) and borrows the current statement
        // and terminator from it, instead of cloning the body — and
        // with it every statement, type and operand it holds — on
        // every step.
        let program: &'a Program = self.program;
        let mut cached: Option<(u64, &Body, std::rc::Rc<crate::decode::DecodedBody>)> =
            None;
        let mut func_name = String::new();
        loop {
            let (block_idx, stmt_idx) = {
                let frame = frames
                    .last()
                    .expect("frame stack is non-empty while running");
                if cached.as_ref().map(|(id, ..)| *id) != Some(frame.id) {
                    func_name = frame.func.clone();
                    let resolved = program.get(&func_name).ok_or_else(|| {
                        InterpError::UnknownFunction {
                            name: func_name.clone(),
                        }
                    })?;
                    let decoded = self.decoded_body(resolved);
                    cached = Some((frame.id, resolved, decoded));
                }
                (frame.block, frame.stmt)
            };
            let body = cached
                .as_ref()
                .expect("current frame body resolved")
                .1;
            let block = body
                .blocks
                .get(block_idx)
                .ok_or_else(|| InterpError::UnknownBlock {
                    func: func_name.clone(),
                    block: block_idx,
                })?;
            // The decoded mirror of `block` (same block and
            // statement layout by construction).
            let dblock = &cached
                .as_ref()
                .expect("current frame body resolved")
                .2
                .blocks[block_idx];

            if self.debug_ctl.is_some() {
                crate::debug::drive_hook(self, frames, body, block_idx, stmt_idx);
            }
            if stmt_idx < block.statements.len() {
                let dstmt = &dblock.stmts[stmt_idx];
                // Advance the program counter before executing, so a call
                // (not possible in a statement) or an error leaves sane state.
                frames.last_mut().expect("frame").stmt += 1;
                self.tick()?;
                self.exec_dstmt(frames, body, dstmt)?;
                continue;
            }

            // Terminator.
            let terminator = &block.terminator;
            self.tick()?;
            match terminator {
                TerminatorKind::Goto { target } => {
                    Self::jump(frames, body, *target)?;
                }
                TerminatorKind::SwitchInt {
                    discr,
                    targets,
                    otherwise,
                } => {
                    let value = self.eval_operand(frames, body, discr)?;
                    let bits = value.switch_bits().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "SwitchInt discriminant",
                            "integer or bool",
                            value.kind_name(),
                        )
                    })?;
                    let target = match &dblock.switch {
                        // Sorted table with all-unique patterns:
                        // binary search finds the same target the
                        // linear scan would.
                        Some(table) => table
                            .binary_search_by_key(&bits, |(pattern, _)| *pattern)
                            .map(|idx| table[idx].1)
                            .unwrap_or(*otherwise),
                        None => targets
                            .iter()
                            .find(|(pattern, _)| *pattern == bits)
                            .map(|(_, bb)| *bb)
                            .unwrap_or(*otherwise),
                    };
                    Self::jump(frames, body, target)?;
                }
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    target,
                } => {
                    let mut arg_values = Vec::with_capacity(args.len());
                    for arg in args {
                        arg_values.push(self.eval_operand(frames, body, arg)?);
                    }
                    // A call naming a trait method by its plain
                    // trait path, in a body the converter kept in
                    // generic form, has no body under that name:
                    // when it is not a vtable dispatch, the
                    // builtin of the same name may serve it (see
                    // `builtin_name_fallback`). The vtable check
                    // comes first, exactly as in `start_call`.
                    let fallback = if self.virtual_call_target(func, &arg_values).is_none() {
                        self.builtin_name_fallback(func)
                    } else {
                        None
                    };
                    if let Some((canonical, kind, generic_args)) = fallback {
                        self.complete_builtin_name_fallback(
                            frames,
                            body,
                            &canonical,
                            kind,
                            &generic_args,
                            func,
                            arg_values,
                            destination,
                            *target,
                        )?;
                    } else {
                        self.start_call(frames, func, arg_values, destination, *target)?;
                    }
                }
                TerminatorKind::CallIndirect {
                    func,
                    args,
                    destination,
                    target,
                } => {
                    // The callee value comes from a place: resolve it
                    // together with the environment reference a
                    // closure body expects (see `eval_callable`). A
                    // body that takes its environment by value (an
                    // `FnOnce` closure's own body) gets the value
                    // itself instead (see `consumed_env_arg`).
                    let (callee, env_ref) = self.eval_callable(&frames, body, func)?;
                    let env_ref = match self.consumed_env_arg(frames, body, func, &callee)? {
                        Some(by_value) => by_value,
                        None => env_ref,
                    };
                    let mut arg_values = Vec::with_capacity(args.len());
                    for arg in args {
                        arg_values.push(self.eval_operand(frames, body, arg)?);
                    }
                    match callee {
                        Value::FnPtr(name) => {
                            // A function value naming a trait
                            // method by its plain trait path has
                            // no body under that name; the builtin
                            // of the same name may serve it (see
                            // `builtin_name_fallback`).
                            let fallback =
                                if self.virtual_call_target(&name, &arg_values).is_none() {
                                    self.builtin_name_fallback(&name)
                                } else {
                                    None
                                };
                            if let Some((canonical, kind, generic_args)) = fallback {
                                self.complete_builtin_name_fallback(
                                    frames,
                                    body,
                                    &canonical,
                                    kind,
                                    &generic_args,
                                    &name,
                                    arg_values,
                                    destination,
                                    *target,
                                )?;
                            } else {
                                self.start_call(
                                    frames,
                                    &name,
                                    arg_values,
                                    destination,
                                    *target,
                                )?;
                            }
                        }
                        Value::Closure { def, .. } => {
                            let mut full = vec![env_ref];
                            full.extend(arg_values);
                            self.start_call(frames, &def, full, destination, *target)?;
                        }
                        other => {
                            return Err(InterpError::type_mismatch(
                                "indirect call",
                                "callable value",
                                other.kind_name(),
                            ));
                        }
                    }
                }
                TerminatorKind::CallTrait {
                    kind,
                    receiver,
                    args,
                    stamp,
                    destination,
                    target,
                } => {
                    // The caller-location stamp the conversion
                    // carried on this edge (a track-caller
                    // closure called directly or through a
                    // generic `Fn` bound), evaluated in the
                    // caller frame like any other argument.
                    let edge_stamp = match &stamp {
                        Some(op) => {
                            Some(self.eval_operand(frames, body, op)?)
                        }
                        None => None,
                    };
                    let args_value = self.eval_operand(frames, body, &args)?;
                    let arg_values = match args_value {
                        Value::Tuple(items) => items,
                        Value::Unit => Vec::new(),
                        other => {
                            return Err(InterpError::type_mismatch(
                                "Fn trait call arguments",
                                "argument tuple",
                                other.kind_name(),
                            ));
                        }
                    };
                    // The callable value, and the environment
                    // argument a closure body expects as its first
                    // argument. For `call` / `call_mut` the receiver
                    // already is the reference the body takes. For
                    // `call_once` the receiver is the value itself:
                    // a body that takes its environment by value (an
                    // `FnOnce` closure's own body — no shim exists in
                    // the program) receives the value, consuming the
                    // receiver place; any other body receives a
                    // reference to the value in place, which is what
                    // the real `call_once` shim passes when it
                    // consumes an `Fn` / `FnMut` closure (aliasing
                    // the moved-from place is equivalent because that
                    // place is dead after the call).
                    // The concrete type behind a dyn receiver, when
                    // the receiver is one: the dispatch for an ADT
                    // concrete value (a hand-written Fn impl) keys
                    // on it, and `dyn_callable` erases it.
                    let mut dyn_concrete: Option<String> = None;
                    let (callable, env_ref) = match kind {
                        FnCallKind::CallOnce => {
                            let (value, env_ref) = self.eval_callable(&frames, body, receiver)?;
                            let (value, env_ref) =
                                if matches!(value, Value::DynRef { .. }) {
                                    if let Value::DynRef { concrete, .. } = &value {
                                        dyn_concrete = Some(concrete.clone());
                                    }
                                    self.dyn_callable(&frames, &value)?
                                } else {
                                    (value, env_ref)
                                };
                            // A by-value receiver may still be a
                            // transparent one-field wrapper around
                            // the callable (an `FnOnce` wrapper
                            // struct): unwrap it, but only when the
                            // wrapped value is itself callable.
                            let (value, unwrapped) = unwrap_callable_wrapper(value);
                            let env_ref = if unwrapped {
                                Value::Ref(RefTarget::value(value.clone()))
                            } else {
                                env_ref
                            };
                            let env_ref = match self.consumed_env_arg(
                                frames,
                                body,
                                receiver,
                                &value,
                            )? {
                                Some(by_value) => by_value,
                                None => env_ref,
                            };
                            (value, env_ref)
                        }
                        FnCallKind::Call | FnCallKind::CallMut => {
                            let recv = self.eval_operand(frames, body, receiver)?;
                            match recv {
                                Value::Ref(_) | Value::FatRef { .. } => {
                                    let resolved = self.deref_value(&frames, &recv)?;
                                    let mut value =
                                        self.read_resolved(&frames, &resolved)?;
                                    // Blanket `Fn` impls make
                                    // references and one-field
                                    // smart-pointer wrappers (an
                                    // `&F`, a `Box<dyn Fn>`) callable
                                    // themselves: follow the deref
                                    // chain until the callable value
                                    // itself shows.
                                    let mut unwrapped = false;
                                    for _ in 0..8 {
                                        let next = match &value {
                                            Value::Ref(_) | Value::FatRef { .. } => {
                                                let Ok(t) =
                                                    self.deref_value(&frames, &value)
                                                else {
                                                    break;
                                                };
                                                let Ok(v) =
                                                    self.read_resolved(&frames, &t)
                                                else {
                                                    break;
                                                };
                                                v
                                            }
                                            Value::Enum { fields, .. }
                                                if fields.len() == 1 =>
                                            {
                                                unwrapped = true;
                                                fields[0].clone()
                                            }
                                            Value::Tuple(items)
                                                if items.len() == 1 =>
                                            {
                                                unwrapped = true;
                                                items[0].clone()
                                            }
                                            _ => break,
                                        };
                                        value = next;
                                    }
                                    if matches!(value, Value::DynRef { .. }) {
                                        if let Value::DynRef { concrete, .. } = &value {
                                            dyn_concrete = Some(concrete.clone());
                                        }
                                        self.dyn_callable(&frames, &value)?
                                    } else if unwrapped {
                                        // The callable was reached by
                                        // unwrapping aggregate values
                                        // (smart pointers), so no place
                                        // reference names it: its
                                        // environment reference is a
                                        // value reference to the
                                        // callable itself, which is
                                        // what capture reads resolve
                                        // through.
                                        let env =
                                            Value::Ref(RefTarget::value(value.clone()));
                                        (value, env)
                                    } else {
                                        (value, recv)
                                    }
                                }
                                Value::DynRef { .. } => {
                                    if let Value::DynRef { concrete, .. } = &recv {
                                        dyn_concrete = Some(concrete.clone());
                                    }
                                    self.dyn_callable(&frames, &recv)?
                                }
                                value => {
                                    // A by-value receiver may be a
                                    // transparent one-field wrapper
                                    // around the callable: unwrap it
                                    // when the wrapped value is
                                    // callable, as in the `CallOnce`
                                    // arm.
                                    let (value, unwrapped) =
                                        unwrap_callable_wrapper(value);
                                    let env = if unwrapped {
                                        Value::Ref(RefTarget::value(
                                            value.clone(),
                                        ))
                                    } else {
                                        self.callable_env_ref(
                                            &frames,
                                            body,
                                            receiver,
                                            &value,
                                        )?
                                    };
                                    (value, env)
                                }
                            }
                        }
                    };
                    // A `dyn` receiver's call passes through
                    // the `Fn` vtable shim, which presents its
                    // own location (the trait method's definition
                    // in core) to a track-caller callee: with no
                    // edge stamp, that shim location is the fixed
                    // stamp for this entry.
                    let shim_stamp = if edge_stamp.is_none()
                        && dyn_concrete.is_some()
                    {
                        let idx = match kind {
                            FnCallKind::Call => 0,
                            FnCallKind::CallMut => 1,
                            FnCallKind::CallOnce => 2,
                        };
                        match &self.program.fn_shim_locations[idx] {
                            Some(c) => Some(self.eval_constant(c)?),
                            None => None,
                        }
                    } else {
                        None
                    };
                    match callable {
                        Value::Closure { def, .. } => {
                            let mut full = vec![env_ref];
                            full.extend(arg_values);
                            self.start_call_stamped(
                                frames,
                                &def,
                                full,
                                edge_stamp,
                                shim_stamp,
                                &destination,
                                *target,
                            )?;
                        }
                        Value::FnPtr(name) => {
                            // As in the indirect-call arm: a
                            // trait method named by its plain
                            // trait path may be served by the
                            // builtin of the same name.
                            let fallback =
                                if self.virtual_call_target(&name, &arg_values).is_none() {
                                    self.builtin_name_fallback(&name)
                                } else {
                                    None
                                };
                            if let Some((canonical, kind, generic_args)) = fallback {
                                self.complete_builtin_name_fallback(
                                    frames,
                                    body,
                                    &canonical,
                                    kind,
                                    &generic_args,
                                    &name,
                                    arg_values,
                                    destination,
                                    *target,
                                )?;
                            } else {
                                self.start_call_stamped(
                                    frames,
                                    &name,
                                    arg_values,
                                    edge_stamp,
                                    shim_stamp,
                                    &destination,
                                    *target,
                                )?;
                            }
                        }
                        other => {
                            // A dyn receiver whose concrete value
                            // is an ADT value: a hand-written Fn
                            // impl on that type, when the program
                            // has its body, is the callable.
                            if let Some(concrete) = dyn_concrete.clone() {
                                if let Some(result) = self.dyn_adt_fn_call(
                                    frames,
                                    *kind,
                                    &concrete,
                                    other.clone(),
                                    env_ref,
                                    arg_values,
                                    destination,
                                    *target,
                                ) {
                                    result?;
                                } else {
                                    return Err(InterpError::type_mismatch(
                                        "Fn trait call",
                                        "callable value",
                                        other.kind_name(),
                                    ));
                                }
                            } else {
                                return Err(InterpError::type_mismatch(
                                    "Fn trait call",
                                    "callable value",
                                    other.kind_name(),
                                ));
                            }
                        }
                    }
                }
                TerminatorKind::CallBuiltin {
                    name,
                    generic_args,
                    args,
                    destination,
                    target,
                    drop_glue,
                    cmp_discriminants,
                    result_unsize,
                } => {
                    if let Some(glue) = drop_glue {
                        // `std::mem::drop` with the dropped type's glue
                        // attached by conversion: run it against the
                        // argument (a place, or a constant's value),
                        // exactly like a `Drop`
                        // terminator — a destructor call suspends the
                        // remaining glue on a `DropContinuation` and
                        // resumes it when the destructor returns. The
                        // call's own (unit) result is written first;
                        // the destination is a different local from
                        // the dropped argument place.
                        let Some(bb) = target else {
                            return Err(InterpError::MissingContinuation {
                                func: func_name.clone(),
                            });
                        };
                        let Some(operand) = args.first() else {
                            return Err(InterpError::type_mismatch(
                                "std::mem::drop",
                                "one argument",
                                "no arguments",
                            ));
                        };
                        // The glue runs against the argument place,
                        // or — for a constant argument, whose value
                        // conversion accounted for when it attached
                        // the glue — against the constant's own value.
                        let root = match operand {
                            Operand::Copy(place) | Operand::Move(place) => {
                                let place = place.clone();
                                match self.resolve_place(&frames, body, &place)? {
                                    Resolved::Place {
                                        frame_pos,
                                        local,
                                        projection,
                                    } => RefTarget::place(
                                        frames[frame_pos].id,
                                        local,
                                        projection,
                                    ),
                                    Resolved::Static { name, projection } => {
                                        RefTarget::Static { name, projection }
                                    }
                                    Resolved::Alloc { id, projection } => {
                                        RefTarget::Alloc { id, projection }
                                    }
                                    Resolved::Value(..) => {
                                        return Err(InterpError::type_mismatch(
                                            "std::mem::drop",
                                            "a storage place",
                                            "a constant aggregate",
                                        ));
                                    }
                                }
                            }
                            Operand::Constant(_) => {
                                let value =
                                    self.eval_operand(&mut frames, body, operand)?;
                                RefTarget::value(value)
                            }
                        };
                        self.write_place(frames, body, destination, Value::Unit)?;
                        let work = vec![(Vec::new(), glue.root.clone())];
                        self.run_drop_glue(frames, root, work, &glue.adts, *bb)?;
                    } else {
                        // A `mem::drop` of a string whose glue is
                        // trivial runs no glue session: the recorded
                        // buffer's deallocation is observed here,
                        // against the argument place.
                        if name == "std::mem::drop" && args.len() == 1 {
                            if let Operand::Copy(place)
                            | Operand::Move(place) = &args[0]
                            {
                                if let Ok(Resolved::Place {
                                    frame_pos,
                                    local,
                                    projection,
                                }) = self.resolve_place(frames, body, place)
                                {
                                    if projection.is_empty() {
                                        let root = RefTarget::place(
                                            frames[frame_pos].id,
                                            local,
                                            projection,
                                        );
                                        self.note_string_buffer_free_inner(
                                            frames, &root, true,
                                        );
                                    }
                                }
                            }
                        }
                        let mut result = self.exec_builtin(
                            frames,
                            body,
                            name,
                            generic_args,
                            args,
                            cmp_discriminants.as_deref(),
                        )?;
                        // A `From` / `Into` conversion into a
                        // `Box<dyn Trait>` returns the plain box
                        // aggregate from the modelled conversion;
                        // conversion baked the unsizing the
                        // destination type implies, so the
                        // destination holds the same `DynRef` a
                        // coercion cast would have produced.
                        if let Some(meta) = result_unsize {
                            result = self.eval_unsize(frames, result, meta)?;
                        }
                        match target {
                            Some(bb) => {
                                self.write_place(frames, body, destination, result)?;
                                if name == "std::string::String::with_capacity" {
                                    self.record_string_buffer(frames, body, destination);
                                }
                                Self::jump(frames, body, *bb)?;
                            }
                            None => {
                                return Err(InterpError::MissingContinuation {
                                    func: func_name.clone(),
                                });
                            }
                        }
                    }
                }
                TerminatorKind::Drop { place, target, glue } => {
                    // Run the dropped type's glue (conversion built
                    // it; see `mir_convert`). The glue work starts
                    // here and either finishes in `run_drop_glue`
                    // (which jumps to `target`) or suspends on a
                    // destructor call and resumes when that call
                    // returns.
                    let resolved = self.resolve_place(&frames, body, place)?;
                    let root = match resolved {
                        Resolved::Place {
                            frame_pos,
                            local,
                            projection,
                        } => RefTarget::place(frames[frame_pos].id, local, projection),
                        Resolved::Static { name, projection } => RefTarget::Static {
                            name,
                            projection,
                        },
                        Resolved::Alloc { id, projection } => RefTarget::Alloc {
                            id,
                            projection,
                        },
                        Resolved::Value(base, projection) => {
                            // The place is the deref of a value
                            // reference (a `Box` / `ManuallyDrop`
                            // deref in this model): the glue runs
                            // against that value itself, as it does
                            // for a `mem::drop` constant argument.
                            RefTarget::Value(Box::new(base), projection)
                        }
                    };
                    let work = vec![(Vec::new(), glue.root.clone())];
                    self.run_drop_glue(frames, root, work, &glue.adts, *target)?;
                }
                TerminatorKind::Resume => {
                    // The cleanup chain of the top frame is done:
                    // pop the frame and report the panic stashed in
                    // it when the unwinder entered the cleanup. The
                    // unwinder in `drive` continues the panic in
                    // the caller at its call site.
                    let frame = frames.pop().expect("frame");
                    // A destructor frame whose own cleanup chain
                    // just ran: its suspended glue session is the
                    // panic's next cleanup — the fields of the
                    // dropped value, as in a native `drop_in_place`
                    // whose destructor panicked.
                    let cont_pos = self
                        .drop_continuations
                        .iter()
                        .rposition(|cont| cont.callee_frame == frame.id);
                    match cont_pos {
                        Some(pos) => {
                            let cont = self.drop_continuations.remove(pos);
                            match frame.unwind_payload {
                                Some((message, payload)) => {
                                    match self.run_glue_cleanup(
                                        frames,
                                        cont,
                                        InterpError::Panic { message, payload },
                                    )? {
                                        GlueCleanupFlow::Completed(p) => return Err(p),
                                        // The cleanup suspended on a
                                        // field destructor's frame:
                                        // the run loop continues into
                                        // it, and the session's
                                        // completion re-raises the
                                        // stashed panic.
                                        GlueCleanupFlow::Suspended => {}
                                    }
                                }
                                None => {
                                    return Err(InterpError::UnsupportedTerminator {
                                        kind: "Resume".to_string(),
                                    });
                                }
                            }
                        }
                        None => {
                            self.drop_continuations
                                .retain(|cont| cont.callee_frame != frame.id);
                            match frame.unwind_payload {
                                Some((message, payload)) => {
                                    self.resume_signaled = true;
                                    return Err(InterpError::Panic { message, payload });
                                }
                                None => {
                                    return Err(InterpError::UnsupportedTerminator {
                                        kind: "Resume".to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
                TerminatorKind::Abort => {
                    // A panic escaped a cleanup or crossed a
                    // nounwind boundary in the guest source: the
                    // run aborts, uncatchably.
                    let original = frames
                        .last()
                        .and_then(|f| f.unwind_payload.as_ref().map(|(m, _)| m.clone()));
                    return Err(InterpError::Abort {
                        message: match original {
                            Some(message) => format!(
                                "panicked while processing panic `{message}`: aborting"
                            ),
                            None => "abort terminator executed".to_string(),
                        },
                    });
                }
                TerminatorKind::Return => {
                    if let Some(value) = self.exec_return(frames, stop_depth)? {
                        return Ok(value);
                    }
                }
                TerminatorKind::Unreachable => {
                    return Err(InterpError::panic("entered unreachable code"));
                }
                TerminatorKind::Assert {
                    cond,
                    expected,
                    target,
                    message,
                    overflow,
                } => {
                    // An overflow assertion in a session compiled
                    // without overflow checks is dropped, as native
                    // codegen drops it: the checked operation's
                    // wrapped result is simply used, and the
                    // condition is not evaluated at all.
                    if *overflow && !self.config.overflow_checks {
                        Self::jump(frames, body, *target)?;
                    } else {
                        let value = self.eval_operand(frames, body, cond)?;
                        let actual = value.as_bool().ok_or_else(|| {
                            InterpError::type_mismatch("Assert condition", "bool", value.kind_name())
                        })?;
                        if actual != *expected {
                            return Err(InterpError::panic(message.clone()));
                        }
                        Self::jump(frames, body, *target)?;
                    }
                }
                TerminatorKind::InlineAsm { regs, steps, target } => {
                    self.exec_inline_asm(frames, body, &regs, &steps)?;
                    match target {
                        Some(target) => Self::jump(frames, body, *target)?,
                        None => {
                            if let Some(value) = self.exec_return(frames, stop_depth)? {
                                return Ok(value);
                            }
                        }
                    }
                }
                TerminatorKind::Unsupported(kind) => {
                    return Err(InterpError::UnsupportedTerminator { kind: kind.clone() });
                }
            }
        }
    }

    /// Execute one evaluable inline-asm block: seed a 64-bit slot
    /// per operand from its input value, run the register program
    /// with x86-64 partial-register semantics, and write back the
    /// operands the program wrote. Inout operands always write
    /// back; a pure output the program never wrote keeps its
    /// destination untouched — its value is unspecified, and
    /// leaving the place alone keeps a later read an explicit
    /// uninitialized error rather than a fabricated zero.
    fn exec_inline_asm(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        regs: &[AsmReg],
        steps: &[AsmStep],
    ) -> InterpResult<()> {
        let mut slots: Vec<u64> = Vec::with_capacity(regs.len());
        for reg in regs {
            let bits = match &reg.init {
                Some(op) => {
                    let value = self.eval_operand(frames, body, op)?;
                    asm_input_bits(&value, reg)?
                }
                None => 0,
            };
            slots.push(bits);
        }
        let mut written = vec![false; regs.len()];
        for step in steps {
            exec_asm_step(&mut slots, &mut written, regs, step);
        }
        for (i, reg) in regs.iter().enumerate() {
            let Some(dest) = &reg.dest else { continue };
            if !written[i] && reg.init.is_none() {
                continue;
            }
            let value = asm_output_value(slots[i], reg);
            self.write_place(frames, body, dest, value)?;
        }
        Ok(())
    }

    /// Execute a return from the top frame: deliver the return
    /// place's value to the caller, or finish the run when the
    /// frame stack reaches the stop depth. `Ok(Some(value))` ends
    /// the run; `Ok(None)` means execution continues in the caller
    /// (or in a resumed drop-glue session).
    fn exec_return(
        &mut self,
        frames: &mut Vec<Frame>,
        stop_depth: usize,
    ) -> InterpResult<Option<Value>> {
        // A returning destructor frame resumes its
        // suspended drop glue instead of delivering a
        // value to a caller place.
        let returning = frames.last().expect("frame").id;
        if let Some(pos) = self
            .drop_continuations
            .iter()
            .rposition(|c| c.callee_frame == returning)
        {
            let cont = self.drop_continuations.remove(pos);
            frames.pop();
            self.run_glue_session(frames, cont.session, cont.sessions)?;
            return Ok(None);
        }
        let frame = frames.pop().expect("frame");
        let mut value = frame.locals.first().cloned().unwrap_or(Value::Uninit);
        if value == Value::Uninit {
            // rustc MIR never writes the return place of
            // a zero-sized return type: the final block
            // is a bare `return;`, and the type's single
            // value is implied. An unwritten `_0`
            // therefore reads as that value exactly when
            // the declared return type is provably
            // zero-sized. For every other type an
            // unwritten `_0` is a genuine uninitialized
            // read and stays an error.
            match self.zst_local_value(&frame, 0) {
                Some(v) => value = v,
                None => {
                    return Err(InterpError::UninitializedRead {
                        func: frame.func.clone(),
                        place: "_0".to_string(),
                    });
                }
            }
        }
        if frames.len() == stop_depth {
            return Ok(Some(value));
        }
        let dest = frame
            .return_dest
            .clone()
            .expect("non-entry frame has a destination");
        let resume = frame.return_block;
        let caller_name = frames.last().expect("caller").func.clone();
        let caller_body = self.program.get(&caller_name).ok_or_else(|| {
            InterpError::UnknownFunction {
                name: caller_name.clone(),
            }
        })?;
        self.write_place(frames, caller_body, &dest, value)?;
        match resume {
            Some(bb) => Self::jump(frames, caller_body, bb)?,
            None => {
                return Err(InterpError::MissingContinuation { func: caller_name });
            }
        }
        Ok(None)
    }

    /// The error for naming a function that has no body in the
    /// program: the recorded conversion failure when the function
    /// exists but could not be converted, a plain unknown-function
    /// error otherwise.
    fn missing_body_error(&self, name: &str) -> InterpError {
        match self.program.unconverted_reason(name) {
            Some(reason) => InterpError::UnconvertedFunction {
                name: name.to_string(),
                reason: reason.to_string(),
            },
            None => InterpError::UnknownFunction {
                name: name.to_string(),
            },
        }
    }

    /// The builtin a body-less call to `func` may be served by
    /// at run time, with the name's canonical form: `func` names
    /// a trait method by its plain trait path (a function value
    /// such as `.map(Clone::clone)`, or a call inside a body the
    /// converter kept in generic form), the program holds no body
    /// under that name — the driver collects none for a
    /// builtin-owned definition, and conversion had no impl
    /// instance to resolve the name to — and the builtin of the
    /// same name is one of the value-semantics models (see
    /// `builtins::runtime_name_fallback`). `None` when a body
    /// exists (a collected body wins, as everywhere) or no such
    /// model applies.
    fn builtin_name_fallback(
        &self,
        func: &str,
    ) -> Option<(String, crate::builtins::Builtin, Vec<String>)> {
        if self.program.get(func).is_some() {
            return None;
        }
        let canonical = crate::builtins::canonical_name(func);
        if let Some(kind) = crate::builtins::lookup_qualified(&canonical) {
            if crate::builtins::runtime_name_fallback(kind) {
                return self.fallback_generic_args(kind, &canonical)
                    .map(|args| (canonical, kind, args));
            }
            return None;
        }
        // A body-less foreign callee named by a function value
        // (an extern declaration reified as a function pointer)
        // is served by the foreign-helper models: they compute
        // from the argument values alone, like the
        // value-semantics models above.
        if let Some(kind) = crate::builtins::foreign_helper_lookup(&canonical) {
            return Some((canonical, kind, Vec::new()));
        }
        // A name the converter kept in generic form carries the
        // turbofish spelling of its type arguments
        // (`std::vec::Vec::<T>::new`); the lookup's impl-path
        // rules are written against the instantiated spelling
        // (`std::vec::Vec<T>::new`). Retry with the turbofish
        // markers folded away.
        if canonical.contains("::<") {
            let folded = canonical.replace("::<", "<");
            if let Some(kind) = crate::builtins::lookup_qualified(&folded) {
                if crate::builtins::runtime_name_fallback(kind) {
                    return self
                        .fallback_generic_args(kind, &folded)
                        .map(|args| (folded, kind, args));
                }
            }
        }
        // The same trait methods spelled by an impl path: a
        // qualified name (`<i32 as std::clone::Clone>::clone`)
        // or an impl-block path (`std::clone::impls::<impl
        // std::clone::Clone for i32>::clone`). The lookup's
        // suffix rules are written against the trait-path
        // spelling only; the value-semantics models answer for
        // every spelling alike.
        if let Some(kind) = impl_path_trait_method(&canonical) {
            return Some((canonical, kind, Vec::new()));
        }
        // The same for the primitive arithmetic traits
        // (`<i32 as std::ops::Add>::add`): their core impls are
        // the operations themselves at the Self type the name
        // carries.
        if let Some(kind) = impl_path_primitive_op(&canonical) {
            return Some((canonical, kind, Vec::new()));
        }
        // An impl-qualified `Default::default` name (`<i32 as
        // std::default::Default>::default`). The global lookup
        // deliberately has no suffix rule for this shape —
        // conversion must keep collecting real impl bodies where
        // it can — but when no body was collected, the Self type
        // the name itself carries is everything the model needs.
        if canonical.ends_with(">::default")
            && canonical.contains("::Default>::default")
        {
            if let Some(self_ty) = impl_qualified_self_type(&canonical) {
                return Some((
                    canonical,
                    crate::builtins::Builtin::DefaultDefault,
                    vec![self_ty],
                ));
            }
        }
        // The same shape for the specialization entry point
        // every Vec collect delegates to
        // (`<std::vec::Vec<T> as std::vec::spec_from_iter_nested::
        // SpecFromIterNested>::from_iter`): the name carries the
        // Vec target, and the nested-collect model computes from
        // the iterator value alone (an empty iterator tag: the
        // generic transition reads the state value's shape).
        if canonical.ends_with(">::from_iter")
            && canonical.contains("::spec_from_iter_nested::SpecFromIterNested")
        {
            if let Some(target) = impl_qualified_self_type(&canonical) {
                if target.starts_with("std::vec::Vec<")
                    || target.starts_with("alloc::vec::Vec<")
                {
                    return Some((
                        canonical,
                        crate::builtins::Builtin::VecFromIterNested,
                        vec![String::new()],
                    ));
                }
            }
        }
        // An impl-qualified `FromIterator::from_iter` name
        // (`<std::vec::Vec<T> as std::iter::FromIterator<T>>::
        // from_iter`): the caller stayed in generic form, so no
        // impl instance was ever resolved and the plain trait-path
        // lookup never saw the name. The name itself carries the
        // target collection, and the collect model computes from
        // the iterator value alone (the iterator type tag only
        // selects special state machines; the generic transition
        // reads the state value), so serve it with the trait
        // path's builtin for the targets the model has a shape
        // for.
        if canonical.ends_with(">::from_iter")
            && (canonical.contains("::FromIterator<")
                || canonical.contains("::FromIterator>::"))
        {
            if let Some(target) = impl_qualified_self_type(&canonical) {
                if target.starts_with("std::vec::Vec<")
                    || target.starts_with("alloc::vec::Vec<")
                    || target == "std::string::String"
                    || target == "alloc::string::String"
                {
                    return Some((
                        canonical,
                        crate::builtins::Builtin::FromIteratorFromIter,
                        vec![target, String::new(), String::new()],
                    ));
                }
            }
        }
        None
    }

    /// The generic arguments a run-time name fallback executes
    /// with. The allowlisted models ignore them — all but
    /// `Default::default`, whose model needs the Self type: an
    /// impl-qualified name (`<i32 as std::default::Default>::
    /// default`) carries it, the plain trait path does not, and
    /// there the fallback does not apply (`None`).
    fn fallback_generic_args(
        &self,
        kind: crate::builtins::Builtin,
        canonical: &str,
    ) -> Option<Vec<String>> {
        if kind != crate::builtins::Builtin::DefaultDefault {
            return Some(Vec::new());
        }
        impl_qualified_self_type(canonical).map(|t| vec![t])
    }

    /// Complete a call served by the run-time name fallback:
    /// execute the builtin on the evaluated arguments and write
    /// its result to the call's destination, exactly as a
    /// `CallBuiltin` terminator completes.
    fn complete_builtin_name_fallback(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        canonical: &str,
        kind: crate::builtins::Builtin,
        generic_args: &[String],
        func: &str,
        arg_values: Vec<Value>,
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> InterpResult<()> {
        // A call the conversion kept as a plain call can still
        // carry the driver-appended caller-location stamp (a
        // `#[track_caller]` trait method's call); a builtin that
        // does not consume the location is served the declared
        // arguments only, as in the `CallBuiltin` conversion.
        let mut arg_values = arg_values;
        if !crate::builtins::consumes_caller_location(kind)
            && arg_values.len() > 1
            && arg_values.last().is_some_and(is_caller_location_value)
        {
            arg_values.pop();
        }
        let result = self.exec_builtin_values(
            frames,
            body,
            &[],
            kind,
            canonical,
            generic_args,
            arg_values,
            None,
        )?;
        match target {
            Some(bb) => {
                self.write_place(frames, body, destination, result)?;
                Self::jump(frames, body, bb)
            }
            None => Err(InterpError::MissingContinuation {
                func: func.to_string(),
            }),
        }
    }

    /// Resolve a call that has no body under its own name against
    /// a `dyn` receiver: the vtable lookup, with the fat pointer's
    /// (concrete type, trait) pair standing in for the vtable.
    ///
    /// Returns `None` when the call is not a dynamic dispatch (the
    /// first argument is not a dyn fat pointer, or the callee name
    /// is not a trait method of the pointer's bounds and no impl
    /// body matches). Returns `Some(Ok((impl_name, receiver)))` to
    /// dispatch: the receiver is the fat pointer's data pointer.
    /// Returns `Some(Err(_))` when the call is a dispatch through
    /// the pointer but the concrete implementation body is not in
    /// the program — the explicit boundary of the dyn model, hit at
    /// the call, never at the value's declaration.
    fn virtual_call_target(
        &self,
        func: &str,
        args: &[Value],
    ) -> Option<Result<(String, Value), InterpError>> {
        // The receiver of a trait call is the trait object, but a
        // method whose self type is a wrapper reaches here with
        // the object still inside it: `Future::poll` takes
        // `Pin<&mut Self>`, so the argument is the one-field `Pin`
        // aggregate around the `DynRef`. See through one-field
        // aggregates (a bounded walk: the wrappers are one-field
        // newtypes) to the object; the layers are remembered so
        // the receiver handed to the impl body can be rebuilt in
        // the callee's own self-type shape below.
        let mut layers: Vec<u32> = Vec::new();
        let mut cursor = args.first()?;
        while layers.len() <= 4 {
            match cursor {
                Value::Enum { variant, fields } if fields.len() == 1 => {
                    layers.push(*variant);
                    cursor = &fields[0];
                }
                _ => break,
            }
        }
        let Value::DynRef {
            base,
            concrete,
            traits,
            ..
        } = cursor
        else {
            return None;
        };
        // The receiver the impl body is handed: with no wrapper
        // layers, the fat pointer's data pointer as it stands.
        // With layers, the callee's self type is those same
        // wrappers around the *concrete* data pointer, so the
        // box-shaped base is stripped down to the pointer it
        // carries and the wrappers are rebuilt around it.
        let receiver = {
            let mut inner = (**base).clone();
            if !layers.is_empty() {
                while let Value::Enum { fields, .. } = &inner {
                    if fields.len() == 1 {
                        inner = fields[0].clone();
                    } else {
                        break;
                    }
                }
                for variant in layers.iter().rev() {
                    inner = Value::Enum {
                        variant: *variant,
                        fields: vec![inner],
                    };
                }
            }
            inner
        };
        let (trait_path, method) = func.rsplit_once("::")?;
        if trait_path.is_empty() || method.is_empty() {
            return None;
        }
        let candidate = format!("<{concrete} as {trait_path}>::{method}");
        if self.program.get(&candidate).is_some()
            || self.program.unconverted_reason(&candidate).is_some()
        {
            return Some(Ok((candidate, receiver)));
        }
        // The impl key may carry the trait's generic arguments
        // (`<S as std::ops::FnMut<(i32,)>>::call_mut`) or spell the
        // trait through the other of `std::` / `core::`: scan the
        // program's keys for the unique matching impl body.
        if let Some(found) = self.scan_virtual_impl(concrete, trait_path, method) {
            return Some(Ok((found, receiver)));
        }
        // A coroutine's `Future::poll` (and likewise a
        // generator's `Coroutine::resume` and an async
        // generator's `AsyncIterator::poll_next`) is the
        // coroutine body itself: the compiler synthesizes the
        // impl from the coroutine transform, so the vtable's
        // method instance *is* the coroutine definition, and its
        // body is in the program under the coroutine's own name
        // - the name the concrete tag the value carries denotes
        // (a tag only a coroutine definition produces: it ends
        // in a `{closure#N}` segment no user type can be named
        // with), though the tag's spelling and the body key's
        // differ; see `coroutine_body_key`.
        if concrete.contains("::{closure#")
            && ((method == "poll"
                && matches!(trait_path, "std::future::Future" | "core::future::Future"))
                || (method == "resume"
                    && matches!(
                        trait_path,
                        "std::ops::Coroutine" | "core::ops::Coroutine"
                    ))
                || (method == "poll_next"
                    && matches!(
                        trait_path,
                        "std::async_iter::AsyncIterator"
                            | "core::async_iter::AsyncIterator"
                    )))
        {
            if let Some(key) = self.coroutine_body_key(concrete) {
                return Some(Ok((key, receiver)));
            }
        }
        if traits.iter().any(|t| t == trait_path) {
            return Some(Err(InterpError::UnsupportedDynamicDispatch {
                method: func.to_string(),
                concrete: concrete.clone(),
            }));
        }
        None
    }

    /// The program key of the coroutine body a `dyn` value's
    /// concrete tag denotes. The tag renders the coroutine
    /// *type*: the definition path with the `{closure#N}` segment
    /// last and the full generic-argument list appended - the
    /// enclosing definition's arguments first, then the coroutine
    /// signature's own (resume) arguments. A body key instead
    /// spells an *instance*: the enclosing definition's arguments
    /// ride on the definition's own segments
    /// (`gen::<i32>::{closure#0}`), and a definition with no
    /// arguments keeps the plain path (`me::{closure#0}`). The
    /// matching key is therefore the one whose plain parent path
    /// and closure segment are the tag's and whose argument list
    /// is a prefix of the tag's; the longest such list wins (an
    /// argument-carrying instance beats the uninstantiated
    /// spelling), and a tie is ambiguous. `None` when no
    /// converted body matches.
    fn coroutine_body_key(&self, concrete: &str) -> Option<String> {
        if self.program.get(concrete).is_some() {
            return Some(concrete.to_string());
        }
        let (tag_parent, tag_seg, tag_args) = parse_coroutine_tag(concrete)?;
        let mut best: Option<(usize, String)> = None;
        for key in self.program.functions.keys() {
            let Some((key_parent, key_seg, key_args)) = parse_coroutine_key(key) else {
                continue;
            };
            if key_seg != tag_seg || key_parent != tag_parent {
                continue;
            }
            if key_args.len() > tag_args.len()
                || !key_args.iter().zip(tag_args.iter()).all(|(k, t)| k == t)
            {
                continue;
            }
            match &best {
                Some((len, _)) if *len > key_args.len() => {}
                Some((len, _)) if *len == key_args.len() => return None,
                _ => best = Some((key_args.len(), key.clone())),
            }
        }
        best.map(|(_, key)| key)
    }

    /// Push a call frame for `func` with already-evaluated argument
    /// values (placed at locals 1..), returning into `destination` /
    /// `target` of the current top frame when the callee returns.
    /// Shared by direct calls, indirect calls and `Fn`-trait calls.
    fn start_call(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> InterpResult<()> {
        self.start_call_stamped(
            frames,
            func,
            arg_values,
            None,
            None,
            destination,
            target,
        )
    }

    /// The definition-location stamp value recorded for the
    /// track-caller body `func`, if the driver recorded one.
    fn def_stamp_value(&self, func: &str) -> Option<Value> {
        let c = self.program.track_caller_defs.get(func)?;
        self.eval_constant(c).ok()
    }

    /// [`Interpreter::start_call`] with the caller-location
    /// stamps a call edge can carry beyond its argument list:
    /// `edge_stamp` is a call-site stamp (the stamp a
    /// `CallTrait` terminator carries for a track-caller
    /// closure), which forwards through track-caller callers
    /// like the driver-appended argument of a direct call;
    /// `fixed_stamp` names the shim the call passed through (an
    /// `Fn` vtable shim's location) and is bound as-is, never
    /// forwarded — the shim presents its own location, not the
    /// location it received.
    ///
    /// A marked callee ends up with exactly one stamp, from the
    /// first of these sources that has one: the appended
    /// argument of the call, the edge stamp, the fixed stamp,
    /// and finally the callee's definition location (the stamp
    /// of last resort for entries through a shim that fabricates
    /// one: a function-pointer call, a vtable dispatch to an
    /// impl-only `#[track_caller]` method).
    #[allow(clippy::too_many_arguments)]
    fn start_call_stamped(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
        edge_stamp: Option<Value>,
        fixed_stamp: Option<Value>,
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> InterpResult<()> {
        // A `#[track_caller]` callee (marked at conversion) may
        // arrive with one extra argument: the caller-location
        // stamp the driver appended to the call. Pop it before
        // anything else inspects the arguments; it is bound to
        // the callee's synthetic location local below. The pop
        // is keyed on the body registered under `func`, which
        // for a trait-method call is one instance of the method:
        // the popped stamp is threaded through the vtable
        // dispatch below, so it reaches the impl body that
        // actually runs (and is simply dropped when that body
        // turns out unmarked, whose location the model does not
        // track).
        let mut arg_values = arg_values;
        let mut stamp = match self.program.get(func) {
            Some(callee)
                if callee.caller_location_local.is_some()
                    && arg_values.len() == callee.arg_count + 1 =>
            {
                arg_values.pop()
            }
            _ => None,
        };
        let mut forwardable = stamp.is_some();
        if stamp.is_none() {
            if let Some(e) = edge_stamp {
                stamp = Some(e);
                forwardable = true;
            } else if let Some(f) = fixed_stamp {
                stamp = Some(f);
            }
        }
        // A call whose receiver is a `dyn` fat pointer is a vtable
        // dispatch even when a body exists under the callee's own
        // name: that body is one monomorphic instance of the trait
        // method (static demand registers the first demanded
        // instance of a trait method under the plain trait name),
        // not a dispatch — calling it directly would run that one
        // concrete type's instance for every receiver. The vtable
        // lookup therefore goes first; it returns `None` for calls
        // that are not dispatches through the receiver (a free
        // function taking a `&dyn` argument is the common shape),
        // which keep the direct lookup below.
        let callee = match self.virtual_call_target(func, &arg_values) {
            Some(vt) => match vt {
                // A vtable call: dispatch to the concrete type's
                // impl body, with the fat pointer's data pointer as
                // the receiver (the box aggregate itself for a
                // by-value `Box<dyn>` receiver).
                Ok((impl_func, receiver)) => {
                    // Shape the receiver the way the impl body's
                    // `self` type expects: a reference receiver gets
                    // a reference to the concrete value (through the
                    // box aggregate for a `Box<dyn>`), a by-value
                    // receiver gets the data pointer as it is.
                    let receiver = match self
                        .program
                        .get(&impl_func)
                        .and_then(|b| b.local_tys.get(1))
                    {
                        Some(Ty::Ref(..)) => ref_shaped_receiver(receiver),
                        _ => receiver,
                    };
                    let mut args = arg_values;
                    args[0] = receiver;
                    let (edge, fixed) = if forwardable {
                        (stamp, None)
                    } else {
                        (None, stamp)
                    };
                    return self.start_call_stamped(
                        frames,
                        &impl_func,
                        args,
                        edge,
                        fixed,
                        destination,
                        target,
                    );
                }
                Err(e) => return Err(e),
            },
            None => match self.program.get(func) {
                Some(c) => c,
                None => return Err(self.missing_body_error(func)),
            },
        };
        if frames.len() >= self.config.max_call_depth {
            return Err(InterpError::CallDepthExceeded {
                limit: self.config.max_call_depth,
            });
        }
        if arg_values.len() != callee.arg_count {
            return Err(InterpError::type_mismatch(
                format!("call to `{func}`"),
                format!("{} argument(s)", callee.arg_count),
                format!("{} argument(s)", arg_values.len()),
            ));
        }
        // No stamp on the edge at all: the call reached this
        // body through a shim that fabricates the callee's
        // definition location (a function-pointer call, a
        // vtable dispatch to an impl-only `#[track_caller]`
        // method), so bind the definition stamp the driver
        // recorded for the body.
        if callee.caller_location_local.is_some() && stamp.is_none() {
            stamp = self.def_stamp_value(func);
        }
        let mut locals = vec![Value::Uninit; callee.local_count()];
        for (i, v) in arg_values.into_iter().enumerate() {
            locals[i + 1] = v;
        }
        if let (Some(idx), Some(mut stamp)) = (callee.caller_location_local, stamp) {
            // When the calling frame is itself a track-caller
            // frame, a call-site stamp forwards — it is replaced
            // by the caller's own received location — matching
            // the compiler's propagation (a call written in a
            // plain function inlined into a track-caller body is
            // the one case this runtime forwarding cannot tell
            // apart; the stamp then still names a location in
            // the call chain, one frame further out). A shim
            // stamp (a fixed stamp, or a definition stamp bound
            // above) does not forward: the shim is a frame of
            // its own, and it presents its own location.
            if forwardable {
                if let Some(caller) = frames.last() {
                    if let Some(caller_idx) = self
                        .program
                        .get(&caller.func)
                        .and_then(|b| b.caller_location_local)
                    {
                        stamp = caller.locals[caller_idx].clone();
                    }
                }
            }
            locals[idx] = stamp;
        }
        let frame_id = self.fresh_frame_id();
        frames.push(Frame {
            id: frame_id,
            func: func.to_string(),
            locals,
            block: 0,
            stmt: 0,
            return_dest: Some(destination.clone()),
            return_block: target,
            unwind_payload: None,
            in_cleanup: false,
        });
        Ok(())
    }

    /// Push a frame for `func` with already-evaluated arguments and
    /// no return destination: the frame belongs to a nested
    /// [`Interpreter::drive`] region whose stop depth is the stack
    /// depth before the push, and its return value is that drive's
    /// result. Used by runtime support that calls back into guest
    /// code (the `catch_unwind` intrinsic's try/catch functions).
    fn start_detached_call(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
    ) -> InterpResult<()> {
        // The vtable lookup goes first, exactly as in `start_call`.
        let body = match self.virtual_call_target(func, &arg_values) {
            Some(vt) => match vt {
                // A vtable call, shaped exactly as in `start_call`.
                Ok((impl_func, receiver)) => {
                    let receiver = match self
                        .program
                        .get(&impl_func)
                        .and_then(|b| b.local_tys.get(1))
                    {
                        Some(Ty::Ref(..)) => ref_shaped_receiver(receiver),
                        _ => receiver,
                    };
                    let mut args = arg_values;
                    args[0] = receiver;
                    return self.start_detached_call(frames, &impl_func, args);
                }
                Err(e) => return Err(e),
            },
            None => match self.program.get(func) {
                Some(b) => b,
                None => return Err(self.missing_body_error(func)),
            },
        };
        if arg_values.len() != body.arg_count {
            return Err(InterpError::type_mismatch(
                format!("call to {func}"),
                format!("{} argument(s)", body.arg_count),
                format!("{} argument(s)", arg_values.len()),
            ));
        }
        let mut locals = vec![Value::Uninit; body.local_count()];
        for (i, arg) in arg_values.into_iter().enumerate() {
            locals[i + 1] = arg;
        }
        // A detached entry is a shim entry (runtime support
        // calling back into guest code): a track-caller body
        // receives its definition stamp, as in `start_call`.
        if let Some(idx) = body.caller_location_local {
            if let Some(stamp) = self.def_stamp_value(func) {
                locals[idx] = stamp;
            }
        }
        let id = self.fresh_frame_id();
        frames.push(Frame {
            id,
            func: func.to_string(),
            locals,
            block: 0,
            stmt: 0,
            return_dest: None,
            return_block: None,
            unwind_payload: None,
            in_cleanup: false,
        });
        Ok(())
    }

    /// Call `func` with evaluated arguments and run the nested
    /// drive region until it returns, yielding its return value.
    pub(crate) fn call_to_completion(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
    ) -> InterpResult<Value> {
        let depth = frames.len();
        self.start_detached_call(frames, func, arg_values)?;
        self.drive(frames, depth)
    }

    /// Call `func` synchronously from within the interpreter
    /// (a builtin that must apply a callable it holds as a value,
    /// such as an iterator adapter's closure): push a frame for it
    /// on the caller's stack, drive the stack until that frame
    /// returns, and yield its return value. The caller's frames
    /// below are left exactly as they were. A guest panic inside
    /// the callee propagates as the call's error, as a nested call
    /// from guest code would. On the current drive structure this
    /// is [`Self::call_to_completion`]: a detached frame driven by
    /// a nested [`Self::drive`] region, unwinding included.
    pub(crate) fn call_sync(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
    ) -> InterpResult<Value> {
        // A callable value naming a trait method by its plain
        // trait path — `.map(Clone::clone)` handed to an adapter
        // the builtins drive — has no body under that name; the
        // builtin of the same name serves it (see
        // `builtin_name_fallback`). The body handed to the
        // dispatch is the caller's: the allowlisted models never
        // read it (only the caller-location model does, and it is
        // not allowlisted).
        if let Some((canonical, kind, generic_args)) = self.builtin_name_fallback(func) {
            let program: &Program = self.program;
            if let Some(body) = frames.last().and_then(|f| program.get(&f.func)) {
                return self.exec_builtin_values(
                    frames,
                    body,
                    &[],
                    kind,
                    &canonical,
                    &generic_args,
                    arg_values,
                    None,
                );
            }
        }
        self.call_to_completion(frames, func, arg_values)
    }

    /// Apply a callable value (a closure or a function pointer)
    /// to `args` synchronously, via [`Self::call_sync`]. For a
    /// closure whose body takes its environment by reference,
    /// `env_ref` is the reference the body receives as its first
    /// argument: a caller that knows where the closure value lives
    /// (an iterator adapter's state field, say) passes a reference
    /// to that storage so `FnMut` capture writes persist in place;
    /// with `None` the environment is a reference to the value
    /// itself, through which capture reads resolve and capture
    /// writes are the explicit constant-aggregate error. A body
    /// that takes its environment by value (an `FnOnce` closure's
    /// own body) receives the closure value itself, as with a
    /// direct `FnOnce::call_once`.
    pub(crate) fn call_callable_sync(
        &mut self,
        frames: &mut Vec<Frame>,
        callable: &Value,
        env_ref: Option<Value>,
        args: Vec<Value>,
    ) -> InterpResult<Value> {
        if frames.len() >= self.config.max_call_depth {
            return Err(InterpError::CallDepthExceeded {
                limit: self.config.max_call_depth,
            });
        }
        match callable {
            Value::FnPtr(name) => self.call_sync(frames, name, args),
            Value::Closure { def, .. } => {
                let mut full = Vec::with_capacity(args.len() + 1);
                if self.closure_body_takes_env_by_value(def) {
                    full.push(callable.clone());
                } else {
                    full.push(env_ref.unwrap_or_else(|| {
                        Value::Ref(RefTarget::value(callable.clone()))
                    }));
                }
                full.extend(args);
                self.call_sync(frames, def, full)
            }
            other => Err(InterpError::type_mismatch(
                "synchronous call",
                "callable value",
                other.kind_name(),
            )),
        }
    }

    /// Run the guest's panic hook (if one is installed) for a
    /// freshly raised panic, before unwinding begins. The hook
    /// receives a modelled `PanicHookInfo`: field 0 is the
    /// payload as a `dyn Any` reference, field 1 the panic's
    /// location (the stamp the raising builtin received, or a
    /// zero location for a panic raised without one), field 2
    /// `can_unwind`. A panic raised by the hook itself is
    /// swallowed: the original panic proceeds (natively a
    /// panicking hook aborts the process; the model's runs are
    /// judged by the panic that reaches the guest's catch
    /// points, which is the original either way).
    fn fire_panic_hook(
        &mut self,
        frames: &mut Vec<Frame>,
        stop_depth: usize,
        message: &str,
        payload: &Option<PanicPayload>,
    ) {
        if self.in_panic_hook {
            return;
        }
        if matches!(self.hook_fired_depth, Some(d) if d > stop_depth) {
            return;
        }
        let hook = self.panic_hook.borrow().clone();
        let Some(hook) = hook else { return };
        if is_default_hook(&hook) {
            return;
        }
        self.hook_fired_depth = Some(stop_depth);
        let payload = payload.clone().unwrap_or_else(|| PanicPayload {
            ty: "&str".to_string(),
            value: Box::new(Value::Str(message.to_string())),
        });
        let location = self
            .last_panic_stamp
            .borrow_mut()
            .take()
            .unwrap_or_else(|| {
                Value::Ref(RefTarget::value(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Str(String::new()), Value::Uint(0), Value::Uint(0)],
                }))
            });
        let info = Value::Enum {
            variant: 0,
            fields: vec![
                Value::DynRef {
                    base: Box::new(Value::Enum {
                        variant: 0,
                        fields: vec![(*payload.value).clone()],
                    }),
                    concrete: payload.ty.clone(),
                    traits: vec![
                        "std::any::Any".to_string(),
                        "std::marker::Send".to_string(),
                    ],
                    drop_glue: None,
                },
                location,
                Value::Bool(true),
            ],
        };
        let info_ref = Value::Ref(RefTarget::value(info));
        let Ok((callable, _)) = self.dyn_callable(frames, &hook) else {
            return;
        };
        self.in_panic_hook = true;
        let _ = self.call_callable_to_completion(frames, &callable, vec![info_ref]);
        self.in_panic_hook = false;
    }

    /// Install the guest's panic hook (see
    /// [`Interpreter::panic_hook`]); the default-hook marker
    /// restores the default (no hook called).
    pub(crate) fn set_panic_hook(&self, hook: Value) {
        if is_default_hook(&hook) {
            *self.panic_hook.borrow_mut() = None;
        } else {
            *self.panic_hook.borrow_mut() = Some(hook);
        }
    }

    /// Take the guest's panic hook, leaving the default in its
    /// place; taking the default yields the default-hook marker
    /// value (see [`default_hook_value`]).
    pub(crate) fn take_panic_hook(&self) -> Value {
        self.panic_hook
            .borrow_mut()
            .take()
            .unwrap_or_else(default_hook_value)
    }

    /// Record the caller-location stamp of a panic being raised
    /// (see [`Interpreter::last_panic_stamp`]).
    pub(crate) fn note_panic_stamp(&self, stamp: Value) {
        *self.last_panic_stamp.borrow_mut() = Some(stamp);
    }

    /// Forget the recorded panic stamp (the libtest harness model
    /// clears it before each test so a stale stamp is never read
    /// for a later test).
    pub(crate) fn clear_panic_stamp(&self) {
        *self.last_panic_stamp.borrow_mut() = None;
    }

    /// The (file, line, column) of the panic currently recorded,
    /// read out of its caller-location stamp: a `Location` value
    /// whose fields are the file string (possibly wrapped as the
    /// one-field enum a `&str` constant decodes to), the line and
    /// the column.
    pub(crate) fn panic_location(&self) -> Option<(String, u64, u64)> {
        let stamp = self.last_panic_stamp.borrow().clone()?;
        fn fields_of(v: &Value) -> Option<&Vec<Value>> {
            match v {
                Value::Ref(RefTarget::Value(inner, projection)) if projection.is_empty() => {
                    fields_of(inner)
                }
                Value::Enum { fields, .. } | Value::Tuple(fields) => Some(fields),
                _ => None,
            }
        }
        let fields = fields_of(&stamp)?;
        if fields.len() < 3 {
            return None;
        }
        let file = match &fields[0] {
            Value::Str(s) => s.clone(),
            inner => match fields_of(inner) {
                Some(f) if f.len() == 1 => match &f[0] {
                    Value::Str(s) => s.clone(),
                    _ => return None,
                },
                _ => return None,
            },
        };
        let line = match &fields[1] {
            Value::Uint(n) => *n as u64,
            _ => return None,
        };
        let col = match &fields[2] {
            Value::Uint(n) => *n as u64,
            _ => return None,
        };
        Some((file, line, col))
    }

    /// Mark the panic about to be raised as a resumed one (the
    /// `resume_unwind` builtin): the unwinder reads and clears
    /// the flag, and the hook does not fire for it.
    pub(crate) fn note_resumed_panic(&mut self) {
        self.resume_signaled = true;
    }

    /// Record that the in-flight panic was consumed by a
    /// builtin (the libtest runner catching a test's panic), so
    /// a later panic is not mistaken for its propagation.
    pub(crate) fn note_panic_consumed(&mut self) {
        self.hook_fired_depth = None;
    }

    /// Execute the `catch_unwind` intrinsic's protocol: run
    /// `try_fn(data)`; when it returns, report `false` (no panic).
    /// When it panics, the nested drive has already unwound the
    /// panicking frames to the depth at entry, running each
    /// frame's cleanup blocks (the destructors of its live
    /// locals) on the way, exactly as native unwinding does; the
    /// stack is then at the entry depth. Run
    /// `catch_fn(data, payload)` with the panic's raw payload and
    /// report `true`. A panic raised by the catch function itself,
    /// an abort (a panic that escaped a cleanup), and any
    /// non-panic error, propagate unchanged. State outside the
    /// frame stack (the statics store, accumulated guest output)
    /// is not rolled back: writes made before the panic persist,
    /// as they do natively.
    pub(crate) fn exec_catch_unwind(
        &mut self,
        frames: &mut Vec<Frame>,
        try_fn: &str,
        data: Value,
        catch_fn: &str,
    ) -> InterpResult<bool> {
        let depth = frames.len();
        match self.call_to_completion(frames, try_fn, vec![data.clone()]) {
            Ok(_) => Ok(false),
            Err(InterpError::Panic { message, payload }) => {
                self.hook_fired_depth = None;
                self.truncate_frames(frames, depth);
                // A panic site that did not record a payload
                // carries a core literal-message panic: the payload
                // is the message itself, as a `&str`.
                let payload = payload.unwrap_or_else(|| PanicPayload {
                    ty: "&str".to_string(),
                    value: Box::new(Value::Str(message)),
                });
                self.call_to_completion(
                    frames,
                    catch_fn,
                    vec![data, Value::PanicPayload(payload)],
                )?;
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }

    /// Discard every frame above `depth`, together with any
    /// suspended drop-glue continuations belonging to the
    /// discarded frames (their destructors never resume: the
    /// frames they would have resumed into are gone).
    fn truncate_frames(&mut self, frames: &mut Vec<Frame>, depth: usize) {
        if frames.len() <= depth {
            return;
        }
        let removed: Vec<u64> = frames.drain(depth..).map(|f| f.id).collect();
        self.drop_continuations
            .retain(|cont| !removed.contains(&cont.callee_frame));
    }

    /// Evaluate a callable operand (a closure or function value being
    /// called by value) together with the environment reference a
    /// closure body expects as its first argument. The body's first
    /// parameter is a reference to the closure value in rustc MIR;
    /// when the operand is a place, the reference points at that
    /// place, so `FnMut` mutations of by-value captures persist
    /// across calls, exactly as through the real shim. A `Move`
    /// operand's place is deliberately not cleared here: the
    /// environment reference aliases it for the duration of the
    /// call, and the place is dead afterwards by MIR's move
    /// semantics. (A body that takes its environment by value is
    /// the exception: `consumed_env_arg` passes the value itself
    /// and does clear the place.)

    /// The callable value and environment reference behind a
    /// `dyn Fn`-family fat pointer: the concrete value it denotes
    /// (a closure value, for the payloads a program can contain),
    /// referenced through the fat pointer's own base when that
    /// base is a reference (a `&dyn Fn`), and as a value otherwise
    /// (an owned `Box<dyn Fn>` — the closure value inside the box
    /// aggregate; its captures are place references, so calls
    /// through the copy still reach them).
    /// The unique program key of an impl of `trait_path` for the
    /// concrete type `concrete` with method `method`, under any
    /// generic-argument spelling of the trait (or of `Self`) and
    /// under either the `std::` or the `core::` spelling of the
    /// trait path. `None` when there is not exactly one candidate
    /// among the converted and unconverted bodies.
    pub(crate) fn scan_virtual_impl(
        &self,
        concrete: &str,
        trait_path: &str,
        method: &str,
    ) -> Option<String> {
        let mut paths = vec![trait_path.to_string()];
        if let Some(rest) = trait_path.strip_prefix("std::") {
            paths.push(format!("core::{rest}"));
        } else if let Some(rest) = trait_path.strip_prefix("core::") {
            paths.push(format!("std::{rest}"));
        }
        let mut found: Option<String> = None;
        for key in self
            .program
            .functions
            .keys()
            .chain(self.program.unconverted.keys())
        {
            let base = key.split('#').next().unwrap_or(key.as_str());
            let Some((self_part, trait_base, key_method)) = parse_impl_key(base) else {
                continue;
            };
            if key_method != method
                || !paths.iter().any(|p| p == &trait_base)
                || !impl_self_matches(&self_part, concrete)
            {
                continue;
            }
            if found.as_deref() == Some(key.as_str()) {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(key.clone());
        }
        found
    }

    /// Dispatch an `Fn`-trait call whose dyn concrete value is an
    /// ADT value (a hand-written `Fn` / `FnMut` / `FnOnce` impl on
    /// a struct, reached through a `&dyn FnMut` and the like): scan
    /// for the concrete type's impl body for the call's kind and
    /// start it with `(receiver, argument tuple)`, the receiver
    /// shaped exactly as `start_call` shapes a vtable receiver —
    /// a reference when the impl body takes one (unwrapping a
    /// one-field aggregate data pointer, keeping an existing
    /// reference, otherwise referencing the value), the value
    /// itself when it takes the receiver by value. `None` when no
    /// impl body matches, leaving the caller to report the value
    /// as not callable.
    #[allow(clippy::too_many_arguments)]
    fn dyn_adt_fn_call(
        &mut self,
        frames: &mut Vec<Frame>,
        kind: FnCallKind,
        concrete: &str,
        callable: Value,
        env_ref: Value,
        arg_values: Vec<Value>,
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> Option<InterpResult<()>> {
        let (trait_path, method) = match kind {
            FnCallKind::Call => ("std::ops::Fn", "call"),
            FnCallKind::CallMut => ("std::ops::FnMut", "call_mut"),
            FnCallKind::CallOnce => ("std::ops::FnOnce", "call_once"),
        };
        let impl_name = self.scan_virtual_impl(concrete, trait_path, method)?;
        let wants_ref = matches!(
            self.program
                .get(&impl_name)
                .and_then(|b| b.local_tys.get(1)),
            Some(Ty::Ref(..))
        );
        let receiver = if wants_ref {
            match env_ref {
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => env_ref,
                _ => ref_shaped_receiver(callable),
            }
        } else {
            callable
        };
        Some(self.start_call(
            frames,
            &impl_name,
            vec![receiver, Value::Tuple(arg_values)],
            destination,
            target,
        ))
    }

    fn dyn_callable(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<(Value, Value)> {
        let Value::DynRef { base, .. } = value else {
            unreachable!("dyn_callable on a non-dyn value")
        };
        let resolved = self.deref_value(frames, value)?;
        let callable = self.read_resolved(frames, &resolved)?;
        let env = match &**base {
            Value::Ref(_) | Value::FatRef { .. } => (**base).clone(),
            _ => Value::Ref(RefTarget::value(callable.clone())),
        };
        Ok((callable, env))
    }

    fn eval_callable(
        &self,
        frames: &[Frame],
        body: &Body,
        operand: &Operand,
    ) -> InterpResult<(Value, Value)> {
        let value = match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                let resolved = self.resolve_place(frames, body, place)?;
                self.read_resolved(frames, &resolved)?
            }
            Operand::Constant(c) => self.eval_constant(c)?,
        };
        let env_ref = self.callable_env_ref(frames, body, operand, &value)?;
        Ok((value, env_ref))
    }

    /// The environment reference for a callable value obtained from
    /// `operand`: a reference to the operand's place when it has one,
    /// otherwise a reference to the value itself (constant form).
    fn callable_env_ref(
        &self,
        frames: &[Frame],
        body: &Body,
        operand: &Operand,
        value: &Value,
    ) -> InterpResult<Value> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                match self.resolve_place(frames, body, place)? {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => Ok(Value::Ref(RefTarget::place(
                        frames[frame_pos].id,
                        local,
                        projection,
                    ))),
                    Resolved::Value(..) => {
                        Ok(Value::Ref(RefTarget::value(value.clone())))
                    }
                    Resolved::Static { name, projection } => {
                        Ok(Value::Ref(RefTarget::Static { name, projection }))
                    }
                    Resolved::Alloc { id, projection } => {
                        Ok(Value::Ref(RefTarget::Alloc { id, projection }))
                    }
                }
            }
            Operand::Constant(_) => Ok(Value::Ref(RefTarget::value(value.clone()))),
        }
    }

    /// Whether the converted body of the closure `def` takes its
    /// environment by value: its first parameter is the closure
    /// value itself, not a reference to it. That is the body shape
    /// produced for a closure whose own kind is `FnOnce` — the
    /// closure body is resolved with the closure's own kind, so no
    /// `FnOnce` adapter shim exists in the program, and the body
    /// reads its captures with plain field projections on the value
    /// (and may move them out of it).
    fn closure_body_takes_env_by_value(&self, def: &str) -> bool {
        match self.program.get(def) {
            Some(body) => matches!(body.local_ty(Local(1)), Some(Ty::Closure(..))),
            None => false,
        }
    }

    /// The environment argument for a call that consumes its
    /// callable (`FnOnce::call_once`, or an indirect call through a
    /// callable place) when the callee is the closure `value`:
    /// `Some` holding the closure value itself when the closure's
    /// body takes its environment by value. A `Move` operand's
    /// place is then left uninitialized — the value has moved into
    /// the callee — so the closure is really consumed: a later read
    /// of the place is an uninitialized read and a later drop of it
    /// runs no glue a second time, exactly as with any other move.
    /// `None` when the body takes a reference to the environment
    /// instead; the caller passes the environment reference it
    /// already built, which is also what the `call_once` shim does
    /// when it consumes an `Fn` / `FnMut` closure.
    fn consumed_env_arg(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        operand: &Operand,
        value: &Value,
    ) -> InterpResult<Option<Value>> {
        match value {
            Value::Closure { def, .. } if self.closure_body_takes_env_by_value(def) => {}
            _ => return Ok(None),
        }
        if let Operand::Move(place) = operand {
            if place.is_direct_local() {
                let frame = frames.last_mut().expect("frame");
                if let Some(slot) = frame.locals.get_mut(place.local.0) {
                    *slot = Value::Uninit;
                }
            } else if let Ok(target) = self.resolve_place(frames, body, place) {
                if matches!(target, Resolved::Place { .. }) {
                    self.write_resolved(frames, &target, Value::Uninit)?;
                }
            }
        }
        Ok(Some(value.clone()))
    }

    /// Read the value at `root` + `path` for drop glue: `Ok(None)`
    /// when any part of the way is uninitialized (a moved-out or
    /// never-written part has no glue to run, matching rustc's drop
    /// flags); other read failures are errors.
    fn read_at(
        &self,
        frames: &[Frame],
        root: &RefTarget,
        path: &[RefProj],
    ) -> InterpResult<Option<Value>> {
        let Some(value) = self.read_at_raw(frames, root, path)? else {
            return Ok(None);
        };
        // A path that ends on a box's field 0 denotes the pointee
        // (drop plans address a box's contents through its field
        // 0), and the value stored there is the box's allocation
        // pointer: follow it to the pointee itself.
        self.follow_alloc(value)
    }

    /// Follow a value that is a box's allocation pointer to the
    /// pointee it denotes; any other value passes through. A
    /// freed allocation reads as nothing (no drop glue is left
    /// to read in it), as a null pointer does in `read_at`.
    fn follow_alloc(&self, value: Value) -> InterpResult<Option<Value>> {
        match value {
            Value::Ref(RefTarget::Alloc { id, projection }) => {
                let Ok(base) = self.alloc_base(id) else {
                    return Ok(None);
                };
                if base == Value::Uninit {
                    return Ok(None);
                }
                match self.project_value_t(base, &projection, "<drop>", "drop glue") {
                    Ok(v) if v == Value::Uninit => Ok(None),
                    Ok(v) => Ok(Some(v)),
                    Err(InterpError::UninitializedRead { .. }) => Ok(None),
                    Err(e) => Err(e),
                }
            }
            other => Ok(Some(other)),
        }
    }

    /// The value at `root` + `path` for drop glue, without
    /// following a box's allocation pointer the path ends on
    /// (see `read_at`): a path that crosses a box's field 0
    /// mid-way still applies its remaining steps inside the
    /// allocation, because the crossed step denotes the pointee,
    /// not the pointer slot.
    fn read_at_raw(
        &self,
        frames: &[Frame],
        root: &RefTarget,
        path: &[RefProj],
    ) -> InterpResult<Option<Value>> {
        let (base, prefix): (Value, &[RefProj]) = match root {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                let frame_pos = self.frame_pos(frames, *frame)?;
                let frame = &frames[frame_pos];
                let base = frame.locals.get(*local).cloned().ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: frame.func.clone(),
                        local: *local,
                    }
                })?;
                // A never-written place of an evidently zero-sized
                // type still denotes a value for drop glue: optimized
                // MIR deletes the (no-op) write of a ZST aggregate,
                // yet the place's destructor must run — it can have
                // observable effects, such as writing a static. The
                // synthesized value is only a gate for glue: field
                // glue beneath a fabricated ADT reads as never-written
                // and skips, exactly as drop flags dictate.
                //
                // That gate serves a root that *is* the local. A root
                // addressing a sub-place of the local (a non-empty
                // projection — the place a `DerefMut` result names,
                // dropped before the assignment through it) needs the
                // value at that address instead: the fabricated ADT
                // has no fields to project through, so the read would
                // report nothing and the sub-place's destructor would
                // be skipped even though MIR emitted its drop. A
                // single-valued type determines that value exactly
                // (conversion records it in `zst_locals`), so read
                // through it; a type with many values keeps the gate,
                // and the projection finds nothing, as before.
                let base = if base == Value::Uninit {
                    if projection.is_empty() {
                        self.zst_glue_value(frame, *local).unwrap_or(base)
                    } else {
                        self.zst_local_value(frame, *local)
                            .or_else(|| self.zst_glue_value(frame, *local))
                            .unwrap_or(base)
                    }
                } else {
                    base
                };
                (base, projection)
            }
            RefTarget::Value(v, projection) => ((**v).clone(), projection),
            RefTarget::Static { name, projection } => (self.static_base(name)?, projection),
            RefTarget::Alloc { id, projection } => {
                // A freed allocation holds nothing to read glue
                // from, as with a null pointer below.
                let Ok(base) = self.alloc_base(*id) else {
                    return Ok(None);
                };
                (base, projection)
            }
            // A null pointer denotes nothing; there is no drop
            // glue to read through it.
            RefTarget::Null => return Ok(None),
            // An address-only pointer denotes nothing readable;
            // there is no drop glue to read through it.
            RefTarget::Address(_) => return Ok(None),
        };
        let mut value = base;
        let steps: Vec<RefProj> = prefix.iter().chain(path.iter()).cloned().collect();
        let mut step_idx = 0;
        while step_idx < steps.len() {
            if value == Value::Uninit {
                return Ok(None);
            }
            // A previous step crossed a box's field 0: the value
            // here is the box's allocation pointer and this step
            // applies inside the allocation.
            match self.follow_alloc(value)? {
                Some(v) => value = v,
                None => return Ok(None),
            }
            // A coroutine downcast only names the state whose saved
            // storage the following field step addresses, so the two
            // steps are projected together: projected on its own,
            // the field step would read the upvar storage instead.
            let take = if matches!(steps[step_idx], RefProj::Downcast(_))
                && matches!(value, Value::Coroutine { .. })
                && matches!(steps.get(step_idx + 1), Some(RefProj::Field(_)))
            {
                2
            } else {
                1
            };
            match self.project_value_t(
                value,
                &steps[step_idx..step_idx + take],
                "<drop>",
                "drop glue",
            ) {
                Ok(v) => value = v,
                Err(InterpError::UninitializedRead { .. }) => return Ok(None),
                Err(e) => return Err(e),
            }
            step_idx += take;
        }
        if value == Value::Uninit {
            Ok(None)
        } else {
            Ok(Some(value))
        }
    }

    /// A reference to `root` + `path`, for a destructor call.
    /// The path is walked one step at a time: a step that lands
    /// on a box's allocation pointer crosses into the allocation
    /// (the step denotes the pointee, as in `read_at`), so the
    /// destructor's `&mut self` names the pointee's storage, not
    /// the pointer slot inside a copy of the box. A step whose
    /// landing value cannot be read (a never-written place the
    /// glue gate let through) keeps the symbolic projection, as
    /// before.
    fn extend_ref(
        &self,
        frames: &[Frame],
        root: &RefTarget,
        path: &[RefProj],
    ) -> InterpResult<RefTarget> {
        let mut cur = root.clone();
        for step in path {
            let landed = self
                .read_at_raw(frames, &cur, std::slice::from_ref(step))
                .ok()
                .flatten();
            if let Some(Value::Ref(RefTarget::Alloc { id, projection })) = landed {
                cur = RefTarget::Alloc { id, projection };
                continue;
            }
            cur = match cur {
                RefTarget::Place {
                    frame,
                    local,
                    mut projection,
                } => {
                    projection.push(step.clone());
                    RefTarget::place(frame, local, projection)
                }
                RefTarget::Value(v, projection) => {
                    let mut projection = projection.clone();
                    projection.push(step.clone());
                    RefTarget::Value(v.clone(), projection)
                }
                RefTarget::Static {
                    name,
                    mut projection,
                } => {
                    projection.push(step.clone());
                    RefTarget::Static { name, projection }
                }
                RefTarget::Alloc {
                    id,
                    mut projection,
                } => {
                    projection.push(step.clone());
                    RefTarget::Alloc { id, projection }
                }
                RefTarget::Null => {
                    return Err(InterpError::type_mismatch(
                        "drop glue reference",
                        "a valid reference",
                        "null pointer",
                    ))
                }
                RefTarget::Address(_) => {
                    return Err(InterpError::type_mismatch(
                        "drop glue reference",
                        "a valid reference",
                        "address-only pointer",
                    ))
                }
            };
        }
        Ok(cur)
    }

    /// The (data pointer, drop callable) a genuine `RawWaker`
    /// aggregate value carries: field 0 is the data pointer the
    /// vtable functions act on, field 1 a reference to the
    /// vtable value, whose four function pointers are (clone,
    /// wake, wake_by_ref, drop). The drop entry is returned as
    /// the value it was stored as: a plain function item is a
    /// `FnPtr`, while a non-capturing closure literal in the
    /// vtable constant stays a `Closure` value (its coercion to
    /// a function pointer is a type-level fact the value model
    /// does not rewrite). `None` for any other shape.
    fn raw_waker_drop_fn(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Option<(Value, Value)>> {
        let Value::Enum { fields, .. } = value else {
            return Ok(None);
        };
        let [data, vtable] = fields.as_slice() else {
            return Ok(None);
        };
        if !matches!(data, Value::Ref(_)) || !matches!(vtable, Value::Ref(_)) {
            return Ok(None);
        }
        let resolved = self.deref_value(frames, vtable)?;
        let table = self.read_resolved(frames, &resolved)?;
        let Value::Enum { fields, .. } = &table else {
            return Ok(None);
        };
        if fields.len() != 4 {
            return Ok(None);
        }
        if !matches!(fields[3], Value::FnPtr(_) | Value::Closure { .. }) {
            return Ok(None);
        }
        Ok(Some((data.clone(), fields[3].clone())))
    }

    /// Execute drop glue: process `work` until it is empty or a
    /// destructor call has to run. A destructor call pushes its
    /// frame plus a [`DropContinuation`] and returns; the `Return`
    /// terminator resumes the remaining work when the destructor
    /// finishes. When the work is done, the dropped place's own slot
    /// is cleared (for a direct local of the owner frame) and the
    /// owner frame jumps to `target`.
    /// Run a suspended drop-glue session as the cleanup of the
    /// panic unwinding out of its destructor call: the remaining
    /// glue (the fields the destructor itself did not reach) is
    /// the cleanup a native `drop_in_place` performs when the
    /// destructor panics, and the original panic then continues
    /// unwinding.
    fn run_glue_cleanup(
        &mut self,
        frames: &mut Vec<Frame>,
        cont: DropContinuation,
        panic: InterpError,
    ) -> InterpResult<GlueCleanupFlow> {
        let InterpError::Panic { message, payload } = panic else {
            return Ok(GlueCleanupFlow::Completed(panic));
        };
        self.glue_unwind_panic = Some((message, payload));
        let mut session = cont.session;
        session.completion = GlueCompletion::UnwindCleanup;
        match self.run_glue_session(frames, session, cont.sessions) {
            Ok(()) => Ok(GlueCleanupFlow::Suspended),
            Err(e @ InterpError::Panic { .. }) => Ok(GlueCleanupFlow::Completed(e)),
            Err(e) => Err(e),
        }
    }

    fn run_drop_glue(
        &mut self,
        frames: &mut Vec<Frame>,
        root: RefTarget,
        work: Vec<(Vec<RefProj>, crate::ir::DropPlan)>,
        adts: &[crate::ir::DropPlan],
        target: BasicBlock,
    ) -> InterpResult<()> {
        let session = GlueSession {
            root,
            work,
            adts: adts.to_vec(),
            completion: GlueCompletion::Jump(target),
        };
        self.run_glue_session(frames, session, Vec::new())
    }

    /// Run one glue session (and, below it, any sessions suspended
    /// under it) until the outermost session completes or a
    /// destructor call suspends the run.
    fn run_glue_session(
        &mut self,
        frames: &mut Vec<Frame>,
        mut active: GlueSession,
        mut suspended: Vec<GlueSession>,
    ) -> InterpResult<()> {
        use crate::ir::DropPlan as P;
        'sessions: loop {
            while let Some((path, plan)) = active.work.pop() {
                self.tick()?;
                match plan {
                    P::Trivial => {}
                    P::Untyped => {
                        // The type of this part is unrecorded, so judge
                        // by the value's own shape: kinds that never
                        // carry glue pass, tuples recurse (a tuple has
                        // no destructor of its own), and an ADT value
                        // might hide a destructor — an explicit error.
                        match self.read_at(frames, &active.root, &path)? {
                            None => {}
                            Some(Value::Tuple(items)) => {
                                for i in (0..items.len()).rev() {
                                    let mut sub = path.clone();
                                    sub.push(RefProj::Index(i));
                                    active.work.push((sub, P::Untyped));
                                }
                            }
                            // An enum with no fields has no contents
                            // whose glue could hide here (its own
                            // destructor, if any, is a separate plan
                            // node): nothing to judge, like an empty
                            // tuple. Typed zero-sized constants make
                            // this shape common — a fieldless variant
                            // written by an aggregate and never
                            // projected lands here with an untyped plan.
                            Some(Value::Enum { fields, .. }) if fields.is_empty() => {}
                            Some(Value::Enum { .. }) | Some(Value::Coroutine { .. }) => {
                                    return Err(InterpError::UnsupportedTerminator {
                                    kind: "Drop".to_string(),
                                });
                            }
                            Some(_) => {}
                        }
                    }
                    P::Unknown => {
                        return Err(InterpError::UnsupportedTerminator {
                            kind: "Drop".to_string(),
                        });
                    }
                    P::Seq(plans) => {
                        for plan in plans.into_iter().rev() {
                            active.work.push((path.clone(), plan));
                        }
                    }
                    P::At { path: extra, plan } => {
                        let mut full = path;
                        full.extend(extra);
                        active.work.push((full, *plan));
                    }
                    P::Adt { index } => {
                        let plan = active.adts.get(index).cloned().ok_or_else(|| {
                            InterpError::type_mismatch(
                                "drop glue",
                                "an ADT plan in the glue table",
                                "a glue index out of range",
                            )
                        })?;
                        active.work.push((path, plan));
                    }
                    P::Fields { fields, complete } => {
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        let is_coroutine = matches!(value, Value::Coroutine { .. });
                        // For a coroutine in a saved state, the plan
                        // (selected by the `Variants` arm for exactly
                        // this state) covers that state's saved
                        // storage, and the field paths below must
                        // route through its downcast; in the
                        // unresumed state it covers the upvar
                        // storage the bare field paths address.
                        let mut coroutine_state: Option<u32> = None;
                        let items = match value {
                            Value::Tuple(items) => items,
                            Value::Enum { fields, .. } => fields,
                            // A closure's fields are its captures.
                            Value::Closure { env, .. } => env,
                            Value::Coroutine {
                                variant,
                                fields,
                                saved,
                            } => {
                                // A path addressed through a
                                // downcast to the unresumed state
                                // names the upvar storage explicitly
                                // (the drop of a suspended coroutine
                                // schedules its captures' glue under
                                // that address; see the `Variants`
                                // arm); any other path is read in
                                // the value's own state.
                                let effective = match path.last() {
                                    Some(RefProj::Downcast(0)) => 0,
                                    _ => variant,
                                };
                                coroutine_state = Some(effective);
                                if effective == 0 {
                                    fields
                                } else {
                                    saved
                                        .get(effective as usize)
                                        .cloned()
                                        .unwrap_or_default()
                                }
                            }
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "drop glue",
                                    "a tuple or ADT value",
                                    other.kind_name(),
                                ));
                            }
                        };
                        // A coroutine judged through a downcast to
                        // the unresumed state is walked by its upvar
                        // plan, but a slot-mapped coroutine's flat
                        // field vector also holds the later states'
                        // cells, written through bare slot fields
                        // while it ran: those cells belong to the
                        // current state's plan, not the upvars', so
                        // a longer vector is not a mismatch — the
                        // walk below only visits the plan's own
                        // indices.
                        let upvar_walk = is_coroutine
                            && matches!(path.last(), Some(RefProj::Downcast(0)));
                        if items.len() != fields.len()
                            && !(is_coroutine && items.len() < fields.len())
                            && !(upvar_walk && items.len() > fields.len())
                        {
                            // With `complete`, conversion proved the
                            // field count, so a mismatch means the value
                            // is malformed; without it, the value holds
                            // fields whose glue conversion could not
                            // account for. Either way the drop cannot be
                            // carried out faithfully: explicit, never
                            // silent.
                            let _ = complete;
                            return Err(InterpError::UnsupportedTerminator {
                                kind: "Drop".to_string(),
                            });
                        }
                        for (i, field_plan) in fields.into_iter().enumerate().rev() {
                            if i >= items.len() {
                                // Only reachable for a coroutine value
                                // whose storage is shorter than the plan:
                                // the field was never written, so it
                                // holds nothing to drop.
                                continue;
                            }
                            if field_plan != P::Trivial {
                                let mut sub = path.clone();
                                if let Some(state) = coroutine_state {
                                    if state != 0 {
                                        sub.push(RefProj::Downcast(state));
                                    }
                                }
                                sub.push(RefProj::Field(i));
                                active.work.push((sub, field_plan));
                            }
                        }
                    }
                    P::Each { elem } => {
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        let (count, is_enum) = match &value {
                            Value::Tuple(items) => (items.len(), false),
                            Value::Enum { fields, .. } => (fields.len(), true),
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "drop glue",
                                    "an array, tuple or Vec value",
                                    other.kind_name(),
                                ));
                            }
                        };
                        for i in (0..count).rev() {
                            let mut sub = path.clone();
                            sub.push(if is_enum {
                                RefProj::Field(i)
                            } else {
                                RefProj::Index(i)
                            });
                            active.work.push((sub, (*elem).clone()));
                        }
                    }
                    P::Variants { variants } => {
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        let (variant, stored_fields) = match &value {
                            Value::Enum { variant, .. } => (*variant, None),
                            Value::Coroutine {
                                variant,
                                fields,
                                saved,
                            } => {
                                let stored = if *variant == 0 {
                                    fields.len()
                                } else {
                                    saved.get(*variant as usize).map_or(0, Vec::len)
                                };
                                (*variant, Some(stored))
                            }
                            Value::Tuple(_) => (0, None),
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "drop glue",
                                    "an ADT value",
                                    other.kind_name(),
                                ));
                            }
                        };
                        let plan = match variants.get(variant as usize) {
                            Some(plan) => plan.clone(),
                            None => match stored_fields {
                                // A coroutine state the conversion
                                // collected no field evidence for: a
                                // state whose fields are saved or
                                // resumed is projected by the body and
                                // so evidenced, so an unevidenced
                                // state holds only what the value
                                // itself shows - judge each stored
                                // field by its own shape.
                                Some(count) => P::Fields {
                                    fields: vec![P::Untyped; count],
                                    complete: false,
                                },
                                None => P::Unknown,
                            },
                        };
                        active.work.push((path.clone(), plan));
                        // A suspended coroutine's captures live in
                        // the upvar storage, which the selected
                        // state's plan does not cover (it addresses
                        // the state's saved storage): schedule the
                        // unresumed state's plan against the upvar
                        // storage too, addressed through a downcast
                        // to the unresumed state. The work stack is
                        // LIFO, so pushing it after the state's own
                        // plan runs it first: the captures are the
                        // shared prefix of every state's field list
                        // and drop before the state's own locals,
                        // the compiler's layout order. Captures the
                        // state machine already moved out read back
                        // as uninitialised, so for a completed or
                        // panicked coroutine this is inert.
                        if let Value::Coroutine { variant, .. } = &value {
                            if *variant != 0 {
                                if let Some(upvar_plan) = variants.first() {
                                    if *upvar_plan != P::Trivial {
                                        let mut upath = path;
                                        upath.push(RefProj::Downcast(0));
                                        active.work.push((upath, upvar_plan.clone()));
                                    }
                                }
                            }
                        }
                    }
                    P::RefCellRelease { mutable } => {
                        // A moved-out / never-written guard has no
                        // borrow to release, exactly like drop flags in
                        // real MIR.
                        let Some(guard) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        self.release_refcell_borrow(frames, &guard, mutable)?;
                    }
                    P::ChannelHandleDrop { sender } => {
                        // A moved-out / never-written handle has no
                        // count to decrement, exactly like drop flags
                        // in real MIR.
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        let Some(id) = channel_handle_id(&value) else {
                            return Err(InterpError::type_mismatch(
                                "channel handle drop glue",
                                "an mpsc channel handle",
                                value.kind_name(),
                            ));
                        };
                        self.channel_handle_dropped(
                            id,
                            if sender { "sender" } else { "receiver" },
                        );
                    }
                    P::RcHandleDrop { value } => {
                        // A moved-out / never-written handle has no
                        // count to decrement, exactly like drop
                        // flags in real MIR.
                        let Some(handle) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        let id = match rc_handle_id(&handle)
                            .or_else(|| raw_waker_alloc_id(&handle))
                        {
                            Some(id) => id,
                            None => {
                                // A hand-built waker whose data
                                // pointer does not denote an
                                // Rc/Arc allocation field (see
                                // `raw_waker_alloc_id`): its
                                // release is the vtable's drop
                                // function, run on the data
                                // pointer, exactly as the real
                                // `Waker` drop does. The call
                                // suspends this session like a
                                // destructor call does.
                                if let Some((data, callable)) =
                                    self.raw_waker_drop_fn(frames, &handle)?
                                {
                                    // A function item is called
                                    // with the data pointer alone; a
                                    // closure value takes its
                                    // environment argument first,
                                    // exactly as an indirect closure
                                    // call passes it.
                                    let (drop_fn, call_args) = match &callable {
                                        Value::FnPtr(name) => {
                                            (name.clone(), vec![data])
                                        }
                                        Value::Closure { def, .. } => {
                                            let env = if self
                                                .closure_body_takes_env_by_value(def)
                                            {
                                                callable.clone()
                                            } else {
                                                Value::Ref(RefTarget::value(
                                                    callable.clone(),
                                                ))
                                            };
                                            (def.clone(), vec![env, data])
                                        }
                                        _ => unreachable!(
                                            "raw_waker_drop_fn yields callables only"
                                        ),
                                    };
                                    let callee = self
                                        .program
                                        .get(&drop_fn)
                                        .ok_or_else(|| InterpError::UnknownFunction {
                                            name: drop_fn.clone(),
                                        })?;
                                    if frames.len() >= self.config.max_call_depth {
                                        return Err(InterpError::CallDepthExceeded {
                                            limit: self.config.max_call_depth,
                                        });
                                    }
                                    let (arg_count, local_count) =
                                        (callee.arg_count, callee.local_count());
                                    if arg_count != call_args.len()
                                        || local_count < 1 + call_args.len()
                                    {
                                        return Err(InterpError::type_mismatch(
                                            format!("waker vtable drop `{drop_fn}`"),
                                            format!("{} argument(s)", call_args.len()),
                                            format!("{arg_count} argument(s)"),
                                        ));
                                    }
                                    let mut locals = vec![Value::Uninit; local_count];
                                    for (i, arg) in call_args.into_iter().enumerate() {
                                        locals[1 + i] = arg;
                                    }
                                    let frame_id = self.fresh_frame_id();
                                    self.drop_continuations.push(DropContinuation {
                                        callee_frame: frame_id,
                                        session: active,
                                        sessions: suspended,
                                    });
                                    frames.push(Frame {
                                        id: frame_id,
                                        func: drop_fn,
                                        locals,
                                        block: 0,
                                        stmt: 0,
                                        return_dest: None,
                                        return_block: None,
                                        unwind_payload: None,
                                        in_cleanup: false,
                                    });
                                    return Ok(());
                                }
                                return Err(InterpError::type_mismatch(
                                    "reference-counted handle drop glue",
                                    "an Rc/Arc handle",
                                    handle.kind_name(),
                                ));
                            }
                        };
                        if !self.rc_dec_strong(id)? {
                            continue;
                        }
                        // The last strong handle is gone: the
                        // pointee's glue runs against the stored
                        // value in a nested session rooted at the
                        // allocation, and the weak bundle the
                        // strong handles held is released in a
                        // second nested session after it — so the
                        // storage, and its counts, stay observable
                        // to the destructor exactly as long as they
                        // are natively (a `Weak::upgrade` inside the
                        // destructor still reads strong = 0 and
                        // returns `None`, never a dangling error).
                        let release = GlueSession {
                            root: RefTarget::Alloc {
                                id,
                                projection: Vec::new(),
                            },
                            work: vec![(Vec::new(), P::RcWeakRelease)],
                            adts: active.adts.clone(),
                            completion: GlueCompletion::Resume,
                        };
                        let nested = GlueSession {
                            root: RefTarget::Alloc {
                                id,
                                projection: Vec::new(),
                            },
                            work: vec![(vec![RefProj::Field(2)], *value)],
                            adts: active.adts.clone(),
                            completion: GlueCompletion::Resume,
                        };
                        suspended.push(std::mem::replace(&mut active, release));
                        suspended.push(std::mem::replace(&mut active, nested));
                        continue 'sessions;
                    }
                    P::WakerDrop => {
                        // A moved-out / never-written waker has
                        // nothing to release, as with rustc's drop
                        // flags.
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        // The modelled `From` conversion's waker:
                        // a one-field aggregate carrying the
                        // `Arc` / `Rc` handle itself. Its release
                        // is the handle's, at the field's path.
                        if let Value::Enum { fields, .. } = &value {
                            if fields.len() == 1 && rc_handle_id(&fields[0]).is_some() {
                                let mut sub = path.clone();
                                sub.push(RefProj::Field(0));
                                active.work.push((
                                    sub,
                                    P::RcHandleDrop {
                                        value: Box::new(P::Dynamic),
                                    },
                                ));
                                continue;
                            }
                        }
                        // A real `RawWaker` (the `Waker::from_raw`
                        // shape): the waker value's field is the
                        // raw waker, whose fields are the data
                        // pointer and the vtable reference. Dropping
                        // the waker calls the vtable's drop entry
                        // with the data pointer, as `Waker`'s own
                        // `Drop` impl does; the call runs as a
                        // nested frame with this session as its
                        // continuation, exactly as a `CallDtor`
                        // call does.
                        let raw = match &value {
                            Value::Enum { fields, .. } if fields.len() == 1 => {
                                fields[0].clone()
                            }
                            _ => value.clone(),
                        };
                        let Value::Enum {
                            fields: raw_fields,
                            ..
                        } = &raw
                        else {
                            return Err(InterpError::type_mismatch(
                                "waker drop glue",
                                "a RawWaker value or a modelled waker handle",
                                value.kind_name(),
                            ));
                        };
                        if raw_fields.len() != 2 {
                            return Err(InterpError::type_mismatch(
                                "waker drop glue",
                                "a RawWaker value (data pointer and vtable)",
                                value.kind_name(),
                            ));
                        }
                        let data = raw_fields[0].clone();
                        let resolved = self.deref_value(frames, &raw_fields[1])?;
                        let vtable = self.read_resolved(frames, &resolved)?;
                        let Value::Enum {
                            fields: vtable_fields,
                            ..
                        } = &vtable
                        else {
                            return Err(InterpError::type_mismatch(
                                "waker drop glue",
                                "a RawWakerVTable value",
                                vtable.kind_name(),
                            ));
                        };
                        let Some(Value::FnPtr(drop_name)) = vtable_fields.get(3) else {
                            return Err(InterpError::type_mismatch(
                                "waker drop glue",
                                "a RawWakerVTable with a function-pointer drop entry",
                                vtable.kind_name(),
                            ));
                        };
                        let drop_name = drop_name.clone();
                        let callee = self.program.get(&drop_name).ok_or_else(|| {
                            InterpError::UnknownFunction {
                                name: drop_name.clone(),
                            }
                        })?;
                        if frames.len() >= self.config.max_call_depth {
                            return Err(InterpError::CallDepthExceeded {
                                limit: self.config.max_call_depth,
                            });
                        }
                        let (arg_count, local_count) =
                            (callee.arg_count, callee.local_count());
                        if arg_count != 1 || local_count < 2 {
                            return Err(InterpError::type_mismatch(
                                format!("waker vtable drop entry `{drop_name}`"),
                                "one argument",
                                format!("{arg_count} argument(s)"),
                            ));
                        }
                        let mut locals = vec![Value::Uninit; local_count];
                        locals[1] = data;
                        let frame_id = self.fresh_frame_id();
                        self.drop_continuations.push(DropContinuation {
                            callee_frame: frame_id,
                            session: active,
                            sessions: suspended,
                        });
                        frames.push(Frame {
                            id: frame_id,
                            func: drop_name,
                            locals,
                            block: 0,
                            stmt: 0,
                            return_dest: None,
                            return_block: None,
                            unwind_payload: None,
                            in_cleanup: false,
                        });
                        return Ok(());
                    }
                    P::RcWeakRelease => {
                        // The weak reference being released: the
                        // handle at the work path, or — for the
                        // bundle release the interpreter synthesizes
                        // after a pointee's glue — the allocation
                        // the session itself is rooted at.
                        let rooted = if path.is_empty() {
                            match &active.root {
                                RefTarget::Alloc { id, projection }
                                    if projection.is_empty() =>
                                {
                                    Some(*id)
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };
                        let id = match rooted {
                            Some(id) => id,
                            None => {
                                let Some(handle) =
                                    self.read_at(frames, &active.root, &path)?
                                else {
                                    continue;
                                };
                                // A handle naming no allocation is
                                // a dangling `Weak::new()`: its count
                                // lives in a static fake allocation
                                // that is never freed, so releasing
                                // it is unobservable - a no-op.
                                let Some(id) = rc_handle_id(&handle) else {
                                    continue;
                                };
                                id
                            }
                        };
                        self.rc_dec_weak(id)?;
                    }
                    P::Dynamic => {
                        // A moved-out / never-written place holds
                        // nothing to drop, as with rustc's drop flags.
                        // The value is judged like an unaccounted
                        // `dyn` payload: the one-field aggregate
                        // wrappings the interpreter's own models
                        // build are stripped before the shape
                        // judgment, whose destructor-hiding shapes
                        // stay explicit errors.
                        if let Some(value) = self.read_at(frames, &active.root, &path)? {
                            drop_unaccounted_dyn(&value)?;
                        }
                    }
                    P::Dyn => {
                        // The dropped value is a trait object: run the
                        // concrete type's glue, which the value
                        // carries from its unsizing cast, against the
                        // pointee as a nested session.
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        let Value::DynRef {
                            base,
                            drop_glue,
                            ..
                        } = value
                        else {
                            // Not a trait-object value at all: judge
                            // it by its shape, as the `Dynamic` plan
                            // does.
                            drop_unaccounted_dyn(&value)?;
                            continue;
                        };
                        let Some(glue) = drop_glue.map(|g| *g) else {
                            // The value carries no accounted glue:
                            // it was fabricated by the interpreter
                            // itself (a caught panic's payload box, a
                            // joined thread's message box) rather than
                            // formed by an unsizing cast. Judge its
                            // contents by shape; shapes that can hide
                            // a destructor are explicit errors, never
                            // a silent skip.
                            drop_unaccounted_dyn(&base)?;
                            continue;
                        };
                        if glue.root == P::Trivial {
                            continue;
                        }
                        // The pointee the glue applies to: inside the
                        // one-field aggregate for a smart pointer
                        // built by the `new` builtins, or behind the
                        // data pointer itself for a pointer-shaped
                        // base.
                        let rc_base = rc_handle_id(&base);
                        let pointee_root = match &*base {
                            // A reference-counted handle base:
                            // the pointee is the allocation's
                            // value field in the store.
                            Value::Enum { .. } if rc_base.is_some() => RefTarget::Alloc {
                                id: rc_base.expect("checked"),
                                projection: vec![RefProj::Field(2)],
                            },
                            // A built box's single field is its
                            // allocation pointer: the pointee lives
                            // in the allocation store, behind that
                            // very reference.
                            Value::Enum { fields, .. }
                                if fields.len() == 1
                                    && matches!(fields[0], Value::Ref(_)) =>
                            {
                                match &fields[0] {
                                    Value::Ref(target) => target.clone(),
                                    _ => unreachable!("field shape checked"),
                                }
                            }
                            Value::Enum { fields, .. } if fields.len() == 1 => {
                                RefTarget::value(fields[0].clone())
                            }
                            Value::Ref(target) => target.clone(),
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "drop glue",
                                    "a box aggregate or reference dyn data pointer",
                                    other.kind_name(),
                                ));
                            }
                        };
                        let nested = GlueSession {
                            root: pointee_root,
                            work: vec![(Vec::new(), glue.root)],
                            adts: glue.adts,
                            completion: GlueCompletion::Resume,
                        };
                        suspended.push(std::mem::replace(&mut active, nested));
                        continue 'sessions;
                    }
                    P::CallDtor { func } => {
                        if self.read_at(frames, &active.root, &path)?.is_none() {
                            // Moved out of / never written: no glue, as
                            // with rustc's drop flags.
                            continue;
                        }
                        let callee = self.program.get(&func).ok_or_else(|| {
                            InterpError::UnknownFunction { name: func.clone() }
                        })?;
                        if frames.len() >= self.config.max_call_depth {
                            return Err(InterpError::CallDepthExceeded {
                                limit: self.config.max_call_depth,
                            });
                        }
                        let (arg_count, local_count) =
                            (callee.arg_count, callee.local_count());
                        if arg_count != 1 || local_count < 2 {
                            return Err(InterpError::type_mismatch(
                                format!("destructor `{func}`"),
                                "one argument",
                                format!("{arg_count} argument(s)"),
                            ));
                        }
                        let target_ref = self.extend_ref(frames, &active.root, &path)?;
                        let mut locals = vec![Value::Uninit; local_count];
                        // A `pin_drop` destructor takes
                        // `Pin<&mut Self>`, and `Pin` is a one-field
                        // aggregate in this value model (the shape
                        // converted `Pin::new` bodies build): pass
                        // the wrapper, not the bare pointer, or the
                        // body's own `Deref` for `Pin` resolves
                        // through the pointee and unwraps the wrong
                        // field.
                        locals[1] = if func.ends_with(">::pin_drop") {
                            Value::Enum {
                                variant: 0,
                                fields: vec![Value::Ref(target_ref)],
                            }
                        } else {
                            Value::Ref(target_ref)
                        };
                        let frame_id = self.fresh_frame_id();
                        self.drop_continuations.push(DropContinuation {
                            callee_frame: frame_id,
                            session: active,
                            sessions: suspended,
                        });
                        frames.push(Frame {
                            id: frame_id,
                            func,
                            locals,
                            block: 0,
                            stmt: 0,
                            return_dest: None,
                            return_block: None,
                            unwind_payload: None,
                            in_cleanup: false,
                        });
                        return Ok(());
                    }
                }
            }
            // The active session's work is done.
            let finishing_target = match active.completion {
                GlueCompletion::Resume => match suspended.pop() {
                    Some(session) => {
                        active = session;
                        continue 'sessions;
                    }
                    None => {
                        return Err(InterpError::type_mismatch(
                            "drop glue",
                            "an enclosing glue session to resume",
                            "none",
                        ));
                    }
                },
                GlueCompletion::Jump(target) => Some(target),
                GlueCompletion::UnwindCleanup => None,
            };
            // A finishing session releases the dropped place
            // exactly as a completed drop does: the place is dead,
            // as in MIR -- a direct local of the owner frame is
            // cleared, and a dropped box's allocation is freed with
            // it: the pointee's glue has run, and the storage the
            // box pointed at is gone, as a real free makes it. (A
            // box nested inside a larger dropped value is reached
            // as a work item of that value's session, not as a
            // session root; its allocation entry is left in the
            // store -- ids are never reused, so nothing can observe
            // the difference within a run.)
            if let Ok(Some(value)) = self.read_at(frames, &active.root, &[]) {
                if let Some((id, _)) = box_alloc_pointer(&value) {
                    self.note_box_free(frames, &value);
                    self.free_alloc(id);
                }
            }
            // The place is cleared only when it is a local of the
            // frame executing the drop. A place that resolved
            // through a pointer into another frame (the
            // `drop_in_place` shim dropping the caller's local
            // through its `*mut` argument) is not killed: the
            // pointer operation is invisible to the owner body's
            // drop elaboration, which keeps treating the local as
            // live -- it may still be moved, reassigned or
            // forgotten afterwards. Only the pointee's glue ran,
            // as in real memory, where `drop_in_place` leaves the
            // bytes in place.
            self.note_string_buffer_free(frames, &active.root);
            if let RefTarget::Place {
                frame,
                local,
                projection,
            } = &active.root
            {
                if projection.is_empty()
                    && frames.last().map(|f| f.id) == Some(*frame)
                {
                    let frame_pos = self.frame_pos(frames, *frame)?;
                    if let Some(slot) = frames[frame_pos].locals.get_mut(*local) {
                        *slot = Value::Uninit;
                    }
                }
            }
            match finishing_target {
                Some(target) => {
                    let owner = frames.last().expect("owner frame");
                    let owner_body = self
                        .program
                        .get(&owner.func)
                        .cloned()
                        .ok_or_else(|| InterpError::UnknownFunction {
                            name: owner.func.clone(),
                        })?;
                    return Self::jump(frames, &owner_body, target);
                }
                None => {
                    // The session ran as a panic's cleanup: the
                    // stashed panic continues unwinding from here,
                    // as a resume of the original unwinding (the
                    // panic hook does not fire again for it).
                    let Some((message, payload)) = self.glue_unwind_panic.take() else {
                        return Err(InterpError::type_mismatch(
                            "drop glue cleanup",
                            "a stashed panic to resume",
                            "none",
                        ));
                    };
                    self.resume_signaled = true;
                    return Err(InterpError::Panic { message, payload });
                }
            }
        }
    }

    fn jump(frames: &mut Vec<Frame>, body: &Body, target: BasicBlock) -> InterpResult<()> {
        if target.0 >= body.blocks.len() {
            return Err(InterpError::UnknownBlock {
                func: body.name.clone(),
                block: target.0,
            });
        }
        let frame = frames.last_mut().expect("frame");
        frame.block = target.0;
        frame.stmt = 0;
        Ok(())
    }

    /// The error for touching a static that has no storage: its
    /// initializer constant was never evaluated, so there is no
    /// seeded value to read or write. (Every reference to a static
    /// carries its initializer and seeds the store when evaluated,
    /// so this is unreachable for driver-produced programs.)
    /// The id of the thread whose code is currently executing.
    pub(crate) fn current_thread_id(&self) -> u64 {
        self.current_thread.get()
    }

    /// Register a spawned thread: the callable is captured, not
    /// run. Returns the new thread's id.
    pub(crate) fn spawn_thread(&self, callable: Value) -> u64 {
        let id = {
            let mut next = self.next_thread.borrow_mut();
            let id = *next;
            *next += 1;
            id
        };
        self.threads.borrow_mut().insert(
            id,
            ThreadSlot {
                callable: Some(callable),
                outcome: None,
            },
        );
        id
    }

    /// Whether the thread has been spawned but not yet run.
    pub(crate) fn thread_pending(&self, id: u64) -> bool {
        self.threads
            .borrow()
            .get(&id)
            .is_some_and(|slot| slot.callable.is_some())
    }

    /// The recorded outcome of a thread that has run.
    pub(crate) fn thread_outcome(&self, id: u64) -> Option<Result<Value, PanicPayload>> {
        self.threads
            .borrow()
            .get(&id)
            .and_then(|slot| slot.outcome.clone())
    }

    /// Run one pending thread to completion, recording its
    /// outcome. A guest panic in the thread is its outcome (the
    /// `Err` a joiner observes), not a failure of the run; any
    /// other error (an unsupported feature inside the thread)
    /// aborts the run, as it would on the entry thread.
    pub(crate) fn run_thread(&mut self, frames: &mut Vec<Frame>, id: u64) -> InterpResult<()> {
        let callable = self
            .threads
            .borrow_mut()
            .get_mut(&id)
            .and_then(|slot| slot.callable.take());
        let Some(callable) = callable else {
            return Ok(());
        };
        let previous = self.current_thread.get();
        self.current_thread.set(id);
        let result = self.call_callable_to_completion(frames, &callable, Vec::new());
        // The thread's code has finished: run the thread-local
        // destructors it registered, as the platform does at
        // thread exit (a panicking thread's storages are still
        // destroyed; a hard interpreter error aborts the run
        // before this point's semantics matter). The current
        // thread stays `id` while they run — a destructor reads
        // its own thread's storages. A destructor that itself
        // panics aborts the run, as a panic in a native TLS
        // destructor aborts the process.
        let result = match &result {
            Ok(_) | Err(InterpError::Panic { .. }) => match self.run_tls_dtors(frames, id) {
                Ok(()) => result,
                Err(e) => Err(e),
            },
            Err(_) => result,
        };
        self.current_thread.set(previous);
        let outcome = match result {
            Ok(value) => Ok(value),
            Err(InterpError::Panic { message, payload }) => {
                self.hook_fired_depth = None;
                Err(payload.unwrap_or_else(|| PanicPayload {
                    ty: "&str".to_string(),
                    value: Box::new(Value::Str(message)),
                }))
            }
            Err(e) => return Err(e),
        };
        if let Some(slot) = self.threads.borrow_mut().get_mut(&id) {
            slot.outcome = Some(outcome);
        }
        Ok(())
    }

    /// Run every pending thread to completion, in spawn order.
    /// This is the cooperative scheduling point: the running
    /// thread has reached a place where it would block, so the
    /// threads it might be waiting for get their turn. Returns
    /// whether any thread ran.
    pub(crate) fn drain_threads(&mut self, frames: &mut Vec<Frame>) -> InterpResult<bool> {
        let mut ran = false;
        loop {
            let next = self
                .threads
                .borrow()
                .iter()
                .filter(|(_, slot)| slot.callable.is_some())
                .map(|(id, _)| *id)
                .min();
            let Some(id) = next else {
                return Ok(ran);
            };
            self.run_thread(frames, id)?;
            ran = true;
        }
    }

    /// Record a thread-local destructor registration for the
    /// currently executing thread (see the `TlsRegisterDtor`
    /// builtin): the storage reference and the destructor
    /// callable, in registration order.
    pub(crate) fn register_tls_dtor(&self, data: Value, dtor: Value) {
        self.tls_dtors
            .borrow_mut()
            .entry(self.current_thread_id())
            .or_default()
            .push((data, dtor));
    }

    /// Run thread `id`'s registered thread-local destructors in
    /// reverse registration order (the platform's order).
    /// Registrations a destructor itself makes (initialising
    /// another thread-local on the way out) are run as well, as
    /// the platform runs them; the round bound keeps a
    /// pathological registration loop from spinning forever. A
    /// registration whose destructor body the program does not
    /// hold is skipped: the registration call sites materialise
    /// the destructor as a function constant, which the driver's
    /// demand pass collects, so a missing body means the
    /// destructor belongs to a part of std the model never
    /// collected and its storage is one the model never wrote.
    fn run_tls_dtors(&mut self, frames: &mut Vec<Frame>, id: u64) -> InterpResult<()> {
        for _ in 0..1024 {
            let batch: Vec<(Value, Value)> = self
                .tls_dtors
                .borrow_mut()
                .remove(&id)
                .unwrap_or_default();
            if batch.is_empty() {
                return Ok(());
            }
            for (data, dtor) in batch.into_iter().rev() {
                if let Value::FnPtr(name) = &dtor {
                    if self.program.get(name).is_none() {
                        continue;
                    }
                }
                self.call_callable_to_completion(frames, &dtor, vec![data])?;
            }
        }
        Ok(())
    }

    /// Join a thread: run it if it is still pending, then return
    /// its recorded outcome.
    pub(crate) fn join_thread(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
    ) -> InterpResult<Result<Value, PanicPayload>> {
        if self.thread_pending(id) {
            self.run_thread(frames, id)?;
        }
        self.thread_outcome(id).ok_or_else(|| InterpError::UnsupportedTerminator {
            kind: format!("join of unknown thread {id}"),
        })
    }

    /// Call a callable value (a closure or a function pointer)
    /// with `args`, driving the frame stack until the call
    /// returns, and return its return value. Used by builtins
    /// that must execute guest code synchronously (thread join,
    /// `Once::call_once`, scoped threads).
    pub(crate) fn call_callable_to_completion(
        &mut self,
        frames: &mut Vec<Frame>,
        callable: &Value,
        args: Vec<Value>,
    ) -> InterpResult<Value> {
        let base = frames.len();
        match callable {
            Value::Closure { def, .. } => {
                let env = if self.closure_body_takes_env_by_value(def) {
                    callable.clone()
                } else {
                    Value::Ref(RefTarget::value(callable.clone()))
                };
                let mut full = vec![env];
                full.extend(args);
                self.start_call(frames, def, full, &Place::local(Local::RETURN), None)?;
            }
            Value::FnPtr(name) => {
                self.start_call(frames, name, args, &Place::local(Local::RETURN), None)?;
            }
            other => {
                return Err(InterpError::type_mismatch(
                    "callable value",
                    "a closure or function pointer",
                    other.kind_name(),
                ));
            }
        }
        self.drive_frames(frames, base)
    }

    /// Record the guest-allocated buffer of a
    /// `String::with_capacity` result against the place the
    /// call wrote it to (see `string_buffers`).
    fn record_string_buffer(
        &self,
        frames: &Vec<Frame>,
        body: &Body,
        destination: &Place,
    ) {
        let pending = self.pending_string_buffer.borrow_mut().take();
        let Some((ptr, cap)) = pending else {
            return;
        };
        if let Ok(Resolved::Place {
            frame_pos,
            local,
            projection,
        }) = self.resolve_place(frames, body, destination)
        {
            if projection.is_empty() {
                let frame_id = frames[frame_pos].id;
                self.string_buffers
                    .borrow_mut()
                    .insert((frame_id, local), (ptr, cap));
            }
        }
    }

    /// Run the guest deallocator for the recorded buffer of a
    /// string whose place's drop glue just finished (see
    /// `string_buffers`).
    fn note_string_buffer_free(&mut self, frames: &mut Vec<Frame>, root: &RefTarget) {
        self.note_string_buffer_free_inner(frames, root, false);
    }

    /// `note_string_buffer_free`, optionally with the
    /// single-entry fallback for strings whose recorded place
    /// was lost through an untracked move. The fallback is only
    /// sound at a `mem::drop` site, where the dropped value is
    /// consumed by the call itself.
    fn note_string_buffer_free_inner(
        &mut self,
        frames: &mut Vec<Frame>,
        root: &RefTarget,
        allow_fallback: bool,
    ) {
        let RefTarget::Place {
            frame,
            local,
            projection,
        } = root
        else {
            return;
        };
        if !projection.is_empty() {
            return;
        }
        let entry = self
            .string_buffers
            .borrow_mut()
            .remove(&(*frame, *local));
        let entry = entry.or_else(|| {
            if !allow_fallback {
                return None;
            }
            // The string reached this place through a move the
            // assignment transfer did not track (for example
            // straight into a call argument): when the frame
            // holds exactly one recorded buffer, it can only be
            // this string's.
            let mut buffers = self.string_buffers.borrow_mut();
            let in_frame = buffers
                .keys()
                .filter(|(f, _)| *f == *frame)
                .copied()
                .collect::<Vec<_>>();
            if in_frame.len() == 1 {
                buffers.remove(&in_frame[0])
            } else {
                None
            }
        });
        let Some((ptr, cap)) = entry else {
            return;
        };
        let Some(shim) = self.program().alloc_shims.get("__rust_dealloc") else {
            return;
        };
        let shim = shim.clone();
        let _ = self.call_callable_to_completion(
            frames,
            &Value::FnPtr(shim),
            vec![ptr, Value::Uint(cap), alignment_value(1)],
        );
    }

    /// Run the guest deallocator for a dropped box, when the
    /// program defines one: natively the box's storage is freed
    /// by `__rust_dealloc` with the pointee's layout, which the
    /// guest allocator observes. The layout is the one recorded
    /// when the allocation builtins created the box (see
    /// `box_layouts`); a box with no recorded layout (built by
    /// another path) frees only in the model, as before. The
    /// model's own store entry is released by the caller (the
    /// guest's deallocation usually frees it itself through the
    /// system path; a guest allocator that pools the block
    /// instead is beyond what the store can represent).
    pub(crate) fn note_box_free(
        &mut self,
        frames: &mut Vec<Frame>,
        box_value: &Value,
    ) {
        let Some(shim) = self.program().alloc_shims.get("__rust_dealloc") else {
            return;
        };
        let shim = shim.clone();
        let Some((id, _)) = box_alloc_pointer(box_value) else {
            return;
        };
        let Some((size, align)) = self.box_layouts.borrow_mut().remove(&id) else {
            return;
        };
        let Value::Enum { fields, .. } = box_value else {
            return;
        };
        let Some(ptr) = fields.first().cloned() else {
            return;
        };
        let _ = self.call_callable_to_completion(
            frames,
            &Value::FnPtr(shim),
            vec![ptr, Value::Uint(size), alignment_value(align)],
        );
    }

    /// Record the pointee layout of a box the allocation
    /// builtins just created, for `note_box_free`.
    pub(crate) fn record_box_layout(&self, box_value: &Value, size: u128, align: u128) {
        if let Some((id, _)) = box_alloc_pointer(box_value) {
            self.box_layouts.borrow_mut().insert(id, (size, align));
        }
    }

    /// Allocate through the program's guest allocator shim
    /// (`#[global_allocator]`), when the program defines one:
    /// the allocation the native entry point
    /// (`alloc::alloc::__rust_alloc`) would bind to the shim.
    /// Returns the shim's pointer, or `None` when the program
    /// has no guest allocator (the caller then allocates in the
    /// model directly, as before).
    pub(crate) fn call_guest_alloc(
        &mut self,
        frames: &mut Vec<Frame>,
        size: u128,
        align: u128,
    ) -> InterpResult<Option<Value>> {
        let Some(shim) = self.program().alloc_shims.get("__rust_alloc") else {
            return Ok(None);
        };
        let shim = shim.clone();
        let align_arg = alignment_value(align);
        let ptr = self.call_callable_to_completion(
            frames,
            &Value::FnPtr(shim),
            vec![Value::Uint(size), align_arg],
        )?;
        Ok(Some(ptr))
    }

    /// Model std's lazy stdout buffer allocation on the first
    /// use of the print support: the stdout handle's line buffer
    /// (1024 bytes) is allocated through the global allocator
    /// the first time the program prints, and never freed. A
    /// guest allocator observes exactly that one allocation;
    /// with no guest allocator there is nothing to model.
    pub(crate) fn note_stdout_use(
        &mut self,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<()> {
        if self.stdout_buffer_modelled.get() {
            return Ok(());
        }
        self.stdout_buffer_modelled.set(true);
        let _ = self.call_guest_alloc(frames, 1024, 8)?;
        Ok(())
    }

    /// Call the callable denoted by `operand` (a closure or
    /// function value in the current frame) with `args`, driving
    /// the frame stack until the call returns, and return its
    /// return value. The environment reference is built from the
    /// operand exactly as for an indirect call, so a closure
    /// place's captures are mutated in place and the mutations
    /// persist across repeated calls — the `FnMut` semantics the
    /// iterator consumer builtins (`fold`, `for_each`) rely on.
    pub(crate) fn call_callable_operand_to_completion(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        operand: &Operand,
        args: Vec<Value>,
    ) -> InterpResult<Value> {
        let base = frames.len();
        let (callee, env_ref) = self.eval_callable(&frames, body, operand)?;
        let env_ref = match self.consumed_env_arg(frames, body, operand, &callee)? {
            Some(by_value) => by_value,
            None => env_ref,
        };
        match callee {
            Value::Closure { def, .. } => {
                let mut full = vec![env_ref];
                full.extend(args);
                self.start_call(frames, &def, full, &Place::local(Local::RETURN), None)?;
            }
            Value::FnPtr(name) => {
                self.start_call(frames, &name, args, &Place::local(Local::RETURN), None)?;
            }
            other => {
                return Err(InterpError::type_mismatch(
                    "callable operand",
                    "a closure or function pointer",
                    other.kind_name(),
                ));
            }
        }
        self.drive_frames(frames, base)
    }

    /// Create a channel state (one sender, one receiver) and
    /// return its id.
    /// Register a freshly built process command and return the
    /// id its handle value carries.
    pub(crate) fn new_command(&self, state: CommandState) -> u64 {
        let mut next = self.next_command.borrow_mut();
        let id = *next;
        *next += 1;
        drop(next);
        self.commands.borrow_mut().insert(id, state);
        id
    }

    /// Register a captured formatter state and return the id
    /// its handle value carries.
    pub(crate) fn new_formatter(&self, state: FmtCapture) -> u64 {
        let mut next = self.next_formatter.borrow_mut();
        let id = *next;
        *next += 1;
        drop(next);
        self.formatters.borrow_mut().insert(id, state);
        id
    }

    /// Run `f` against the captured formatter with the given
    /// id; `None` if the id names no live formatter.
    pub(crate) fn with_formatter<R>(
        &self,
        id: u64,
        f: impl FnOnce(&mut FmtCapture) -> R,
    ) -> Option<R> {
        let mut formatters = self.formatters.borrow_mut();
        formatters.get_mut(&id).map(f)
    }

    /// A snapshot of the captured formatter with the given id.
    pub(crate) fn formatter_snapshot(&self, id: u64) -> Option<FmtCapture> {
        self.formatters.borrow().get(&id).cloned()
    }

    /// Remove the captured formatter with the given id and
    /// return its final state (the dispatch that created it
    /// reads the captured text out this way).
    pub(crate) fn take_formatter(&self, id: u64) -> Option<FmtCapture> {
        self.formatters.borrow_mut().remove(&id)
    }

    /// Register a builder state and return the id its handle
    /// value carries.
    pub(crate) fn new_fmt_builder(&self, state: FmtBuilderState) -> u64 {
        let mut next = self.next_fmt_builder.borrow_mut();
        let id = *next;
        *next += 1;
        drop(next);
        self.fmt_builders.borrow_mut().insert(id, state);
        id
    }

    /// Run `f` against the builder with the given id; `None`
    /// if the id names no live builder.
    pub(crate) fn with_fmt_builder<R>(
        &self,
        id: u64,
        f: impl FnOnce(&mut FmtBuilderState) -> R,
    ) -> Option<R> {
        let mut builders = self.fmt_builders.borrow_mut();
        builders.get_mut(&id).map(f)
    }

    /// A snapshot of the builder with the given id.
    pub(crate) fn fmt_builder_snapshot(&self, id: u64) -> Option<FmtBuilderState> {
        self.fmt_builders.borrow().get(&id).cloned()
    }

    /// Run `f` against the state of the command with the given
    /// id; `None` if the id names no live command.
    pub(crate) fn with_command<R>(
        &self,
        id: u64,
        f: impl FnOnce(&mut CommandState) -> R,
    ) -> Option<R> {
        let mut commands = self.commands.borrow_mut();
        commands.get_mut(&id).map(f)
    }

    /// A snapshot of the state of the command with the given id.
    pub(crate) fn command_snapshot(&self, id: u64) -> Option<CommandState> {
        self.commands.borrow().get(&id).cloned()
    }

    pub(crate) fn new_channel(&self, bound: Option<usize>) -> u64 {
        let id = {
            let mut next = self.next_channel.borrow_mut();
            let id = *next;
            *next += 1;
            id
        };
        self.channels.borrow_mut().insert(
            id,
            ChannelState {
                queue: std::collections::VecDeque::new(),
                senders: 1,
                receivers: 1,
                bound,
            },
        );
        id
    }

    /// Access one channel's state, if it exists.
    pub(crate) fn with_channel<R>(
        &self,
        id: u64,
        f: impl FnOnce(&mut ChannelState) -> R,
    ) -> Option<R> {
        self.channels.borrow_mut().get_mut(&id).map(f)
    }

    /// Record a channel handle's drop (see [`ChannelState`]):
    /// `kind` is `"sender"` or `"receiver"`.
    pub(crate) fn channel_handle_dropped(&self, id: u64, kind: &str) {
        if let Some(state) = self.channels.borrow_mut().get_mut(&id) {
            match kind {
                "sender" => state.senders = state.senders.saturating_sub(1),
                _ => state.receivers = state.receivers.saturating_sub(1),
            }
        }
    }

    /// Record a channel handle's clone (see [`ChannelState`]).
    pub(crate) fn channel_handle_cloned(&self, id: u64, kind: &str) {
        if let Some(state) = self.channels.borrow_mut().get_mut(&id) {
            match kind {
                "sender" => state.senders += 1,
                _ => state.receivers += 1,
            }
        }
    }

    /// The handle value naming reference-counted allocation `id`,
    /// in the struct's real field shape: `ptr` (a `NonNull`
    /// aggregate holding the allocation reference), then the
    /// zero-sized `phantom: PhantomData` and `alloc: A` fields -
    /// the default allocator is the unit struct `Global`.
    /// Converted std bodies read all three: `make_mut` reads the
    /// allocator field to build the replacement allocation.
    pub(crate) fn rc_handle(&self, id: u64) -> Value {
        Value::Enum {
            variant: 0,
            fields: vec![
                Value::Enum {
                    variant: 0,
                    fields: vec![Value::Ref(RefTarget::Alloc {
                        id,
                        projection: Vec::new(),
                    })],
                },
                Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                },
                Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                },
            ],
        }
    }

    /// Create the allocation a fresh `Rc::new` / `Arc::new`
    /// produces — an `RcInner` / `ArcInner` aggregate
    /// `{ strong, weak, value }` with both counts at 1 (the weak
    /// count includes the bundle the strong handles collectively
    /// hold, as natively) — and return the handle naming it.
    pub(crate) fn rc_new_handle(&self, pointee: Value) -> Value {
        fn count_cell(count: u128) -> Value {
            // A `Cell<usize>` (for `Rc`) or `AtomicUsize` (for
            // `Arc`) aggregate: a one-field wrapper around an
            // `UnsafeCell<usize>` holding the count.
            Value::Enum {
                variant: 0,
                fields: vec![Value::Enum {
                    variant: 0,
                    fields: vec![Value::Uint(count)],
                }],
            }
        }
        let inner = Value::Enum {
            variant: 0,
            fields: vec![count_cell(1), count_cell(1), pointee],
        };
        let id = self.alloc_insert(inner);
        self.rc_handle(id)
    }

    /// Adjust one reference count (field 0 = strong, 1 = weak) of
    /// allocation `id` by `delta`; returns the new count.
    fn rc_adjust_count(&self, id: u64, field: usize, delta: i128) -> InterpResult<u128> {
        let mut store = self.allocs.borrow_mut();
        let inner = store.get_mut(&id).ok_or_else(|| self.missing_alloc(id))?;
        let kind = inner.kind_name();
        let leaf = count_leaf_mut(inner, field).ok_or_else(|| {
            InterpError::type_mismatch(
                "reference-counted allocation",
                "an RcInner/ArcInner aggregate",
                kind,
            )
        })?;
        let next = *leaf as i128 + delta;
        if next < 0 {
            return Err(InterpError::panic(
                "reference count underflow (a handle was dropped twice)".to_string(),
            ));
        }
        *leaf = next as u128;
        Ok(next as u128)
    }

    /// Record a reference-counted handle's clone: another live
    /// strong handle on the same allocation.
    pub(crate) fn rc_inc_strong(&self, id: u64) -> InterpResult<()> {
        self.rc_adjust_count(id, 0, 1)?;
        Ok(())
    }

    /// Record a strong handle's drop; returns true when the
    /// strong count reached zero and the pointee's glue must run.
    pub(crate) fn rc_dec_strong(&self, id: u64) -> InterpResult<bool> {
        Ok(self.rc_adjust_count(id, 0, -1)? == 0)
    }

    /// Record a weak handle's creation (a `Weak` clone; a
    /// downgrade adjusts the count through the stored aggregate
    /// itself, as the converted body does).
    pub(crate) fn rc_inc_weak(&self, id: u64) -> InterpResult<()> {
        self.rc_adjust_count(id, 1, 1)?;
        Ok(())
    }

    /// Record a weak reference's release; frees the allocation's
    /// storage when both counts are zero.
    pub(crate) fn rc_dec_weak(&self, id: u64) -> InterpResult<()> {
        let weak = self.rc_adjust_count(id, 1, -1)?;
        if weak == 0 {
            let inner = self.alloc_base(id)?;
            if count_leaf(&inner, 0) == Some(0) {
                self.allocs.borrow_mut().remove(&id);
            }
        }
        Ok(())
    }

    /// The statics-store key of thread `tid`'s storage for the
    /// thread-local static `name`. TLS storage lives in the
    /// statics store under a key no real static can carry (real
    /// keys are item paths, optionally suffixed with a `DefId`
    /// disambiguator), so reads and writes through a TLS reference
    /// route through the ordinary static machinery while the
    /// storage itself stays private to the owning thread: the
    /// reference names that thread's storage permanently, exactly
    /// as a native pointer names that thread's TLS block.
    fn tls_store_key(tid: u64, name: &str) -> String {
        format!("\u{1}tls\u{1}{tid}\u{1}{name}")
    }

    fn missing_static(&self, name: &str) -> InterpError {
        InterpError::UnsupportedTerminator {
            kind: format!(
                "static `{name}` has no storage: its initializer constant was never evaluated"
            ),
        }
    }

    /// Seed the storage of the static `name` from its initializer
    /// constant `seed`, unless it is already seeded. A static
    /// whose initializer mentions itself (or a cycle of statics)
    /// through a reference does not recurse: the nested reference
    /// evaluates to a pointer at the storage the outer seeding is
    /// about to fill, exactly as the nested reference in a native
    /// static initializer names the static's address.
    fn ensure_static(&self, name: &str, seed: &Constant) -> InterpResult<()> {
        if self.statics.borrow().contains_key(name) {
            return Ok(());
        }
        if !self.statics_seeding.borrow_mut().insert(name.to_string()) {
            return Ok(());
        }
        let value = self.eval_constant(seed);
        self.statics_seeding.borrow_mut().remove(name);
        let value = value?;
        self.statics.borrow_mut().insert(name.to_string(), value);
        Ok(())
    }

    /// The current value of the static `name`'s storage.
    fn static_base(&self, name: &str) -> InterpResult<Value> {
        self.statics
            .borrow()
            .get(name)
            .cloned()
            .ok_or_else(|| self.missing_static(name))
    }

    /// The error for naming an allocation the store no longer
    /// holds (its owning box was dropped, or the id never named
    /// an allocation of this run).
    fn missing_alloc(&self, id: u64) -> InterpError {
        InterpError::type_mismatch(
            "dereference",
            "reference into a live allocation",
            format!("reference into freed allocation #{id}"),
        )
    }

    /// Allocate `pointee` in the allocation store and return the
    /// `Box` value denoting it: the one-field aggregate whose
    /// single field is the allocation pointer (see
    /// [`RefTarget::Alloc`]). This is the only shape the `Box`
    /// construction builtins produce; a box built from its MIR
    /// aggregates instead carries the `Unique`/`NonNull` wrapper
    /// chain above a pointer value.
    /// Create a heap allocation holding `value` in the
    /// allocation store and return the id that names it. The
    /// storage lives until the model frees it: for a box, when
    /// the owning box's drop glue completes; for a
    /// reference-counted allocation, when its strong and weak
    /// counts both reach zero.
    pub(crate) fn alloc_insert(&self, value: Value) -> u64 {
        let id = self.next_alloc.get();
        self.next_alloc.set(id + 1);
        self.allocs.borrow_mut().insert(id, value);
        id
    }

    pub(crate) fn new_box(&self, pointee: Value) -> Value {
        let id = self.alloc_insert(pointee);
        Value::Enum {
            variant: 0,
            fields: vec![Value::Ref(RefTarget::Alloc {
                id,
                projection: Vec::new(),
            })],
        }
    }

    /// The current contents of allocation `id` (a clone; the
    /// store is the shared storage, so writes go through
    /// `write_resolved`, never through this copy).
    pub(crate) fn alloc_base(&self, id: u64) -> InterpResult<Value> {
        self.allocs
            .borrow()
            .get(&id)
            .cloned()
            .ok_or_else(|| self.missing_alloc(id))
    }

    /// Remove allocation `id` from the store (its owning box's
    /// drop glue has completed). Removing an id that is not
    /// present is not an error: glue also runs for places whose
    /// box was fabricated rather than allocated.
    /// Mark an allocation as a raw element block (see
    /// `elem_blocks`).
    pub(crate) fn mark_elem_block(&self, id: u64) {
        self.elem_blocks.borrow_mut().insert(id);
    }

    /// Whether an allocation is a raw element block.
    /// The projection a raw element block's access uses: a
    /// lone byte offset (the denomination a differently sized
    /// pointer cast left on the pointer, see `PtrCastUnits`)
    /// denotes the block's byte slot of that offset — the
    /// block's slots are byte positions for a byte-denominated
    /// pointer exactly as they are element positions for an
    /// element-denominated one, and the two denominations are
    /// kept apart by the projection form itself. Any other
    /// projection passes through unchanged.
    pub(crate) fn elem_block_projection(
        &self,
        id: u64,
        projection: &[RefProj],
    ) -> Vec<RefProj> {
        if self.is_elem_block(id) {
            if let Some(RefProj::ByteOffset(d)) = projection.first() {
                if let Ok(i) = usize::try_from(*d) {
                    let mut out = vec![RefProj::Index(i)];
                    out.extend_from_slice(&projection[1..]);
                    return out;
                }
            }
        }
        projection.to_vec()
    }

    pub(crate) fn is_elem_block(&self, id: u64) -> bool {
        self.elem_blocks.borrow().contains(&id)
    }

    pub(crate) fn free_alloc(&self, id: u64) {
        self.allocs.borrow_mut().remove(&id);
        self.elem_blocks.borrow_mut().remove(&id);
    }

    /// Set the discriminant of the ADT value `value` to `variant`,
    /// as a `SetDiscriminant` statement does: a coroutine keeps both
    /// of its storages (the transform writes the entered state's
    /// fields through its downcast before the tag moves, and values
    /// saved for other states must survive the change), an enum's
    /// old payload is invalidated.
    fn set_discriminant_value(
        value: &mut Value,
        variant: u32,
        place: &Place,
    ) -> InterpResult<()> {
        match value {
            Value::Coroutine { variant: current, .. } => {
                *current = variant;
                // The state storage is shared between states: the
                // fields stay in place. The transform sets the tag
                // of the state it is entering only after (or before)
                // storing that state's fields through downcast
                // projections, and values saved for later states
                // must survive this tag change.
                Ok(())
            }
            Value::Enum {
                variant: current,
                fields,
            } => {
                if *current != variant {
                    *current = variant;
                    // The old payload is invalidated, as in MIR.
                    fields.clear();
                }
                // A SetDiscriminant naming the variant the place
                // already holds is a tag-only write: in the
                // writes-first order (the order the coroutine
                // transform also uses) the new variant's fields were
                // already stored through its downcast, which
                // switched the tag when it rematerialized the
                // payload, and those stores must survive this
                // statement.
                Ok(())
            }
            other => Err(InterpError::type_mismatch(
                format!("SetDiscriminant on {place}"),
                "ADT value",
                other.kind_name(),
            )),
        }
    }

    pub(crate) fn exec_statement(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        statement: &Statement,
    ) -> InterpResult<()> {
        match &statement.kind {
            StatementKind::Assign(place, rvalue) => {
                // A recorded string buffer follows a plain move of
                // the string to its new place (see
                // `string_buffers`).
                let transfer = match rvalue {
                    Rvalue::Use(Operand::Move(src) | Operand::Copy(src)) => {
                        let key_of = |p: &Place| match self
                            .resolve_place(frames, body, p)
                        {
                            Ok(Resolved::Place {
                                frame_pos,
                                local,
                                projection,
                            }) if projection.is_empty() => {
                                Some((frames[frame_pos].id, local))
                            }
                            _ => None,
                        };
                        match (key_of(src), key_of(place)) {
                            (Some(from), Some(to)) if from != to => Some((from, to)),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                let value = self.eval_rvalue(frames, body, rvalue, body.local_ty(place.local))?;
                let written = self.write_place(frames, body, place, value);
                if written.is_ok() {
                    if let Some((from, to)) = transfer {
                        let entry = self.string_buffers.borrow_mut().remove(&from);
                        if let Some(entry) = entry {
                            self.string_buffers.borrow_mut().insert(to, entry);
                        }
                    }
                }
                written
            }
            StatementKind::ShapedStore { place, value, shape } => {
                let value = self.eval_operand(frames, body, value)?;
                let target = self.resolve_place(frames, body, place)?;
                self.shaped_store(frames, &target, value, shape)
            }
            StatementKind::SetDiscriminant { place, variant } => {
                let target = self.resolve_place(frames, body, place)?;
                match target {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => {
                        let frame = &mut frames[frame_pos];
                        let func = frame.func.clone();
                        let slot = frame.locals.get_mut(local).ok_or_else(|| {
                            InterpError::UnknownLocal {
                                func: func.clone(),
                                local,
                            }
                        })?;
                        let place_desc = describe_resolved(local, &projection);
                        let value = navigate_mut(slot, &projection, &func, &place_desc)?;
                        Self::set_discriminant_value(value, *variant, place)
                    }
                    Resolved::Static { name, projection } => {
                        let mut store = self.statics.borrow_mut();
                        let slot = store
                            .get_mut(&name)
                            .ok_or_else(|| self.missing_static(&name))?;
                        let place_desc = format!("static {name}");
                        let value = navigate_mut(slot, &projection, "<static>", &place_desc)?;
                        Self::set_discriminant_value(value, *variant, place)
                    }
                    Resolved::Alloc { id, projection } => {
                        let mut store = self.allocs.borrow_mut();
                        let slot = store
                            .get_mut(&id)
                            .ok_or_else(|| self.missing_alloc(id))?;
                        let place_desc = format!("allocation #{id}");
                        let value =
                            navigate_mut(slot, &projection, "<allocation>", &place_desc)?;
                        Self::set_discriminant_value(value, *variant, place)
                    }
                    Resolved::Value(..) => Err(InterpError::type_mismatch(
                        format!("SetDiscriminant on {place}"),
                        "mutable place",
                        "constant aggregate",
                    )),
                }
            }
            // StorageLive / StorageDead are liveness hints in rustc MIR;
            // they do not change values. Nop is a no-op by definition.
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {
                Ok(())
            }
            StatementKind::Assume(op) => {
                let value = self.eval_operand(frames, body, op)?;
                match value {
                    Value::Bool(true) => Ok(()),
                    Value::Bool(false) => Err(InterpError::Panic {
                        message: "intrinsic `assume` called with a false condition                                   (undefined behaviour in the guest)"
                            .to_string(),
                        payload: None,
                    }),
                    other => Err(InterpError::type_mismatch(
                        "assume operand",
                        "bool",
                        other.kind_name(),
                    )),
                }
            }
            StatementKind::CopyNonOverlapping { src, dst, count } => {
                let src_value = self.eval_operand(frames, body, src)?;
                let dst_value = self.eval_operand(frames, body, dst)?;
                let count_value = self.eval_operand(frames, body, count)?;
                // The copy's element type is the pointee of the
                // pointer operands, when they are bare locals with a
                // declared pointer type. The intrinsic is generic
                // over the element, and the copy needs the type to
                // tell "one element that is itself an aggregate"
                // (an array element, a struct element) from "an
                // aggregate of elements".
                let elem = operand_pointee_ty(body, src)
                    .or_else(|| operand_pointee_ty(body, dst));
                self.exec_copy_nonoverlapping(
                    frames,
                    src_value,
                    dst_value,
                    count_value,
                    elem.as_ref(),
                )
            }
            StatementKind::Unsupported(kind) => Err(InterpError::UnsupportedStatement {
                kind: kind.clone(),
            }),
        }
    }

    /// The base value a bulk byte operation reads at `target`:
    /// the resolved value, or -- for a pointer displaced into a
    /// scalar's storage -- the byte the pointer denotes. The
    /// operation's element is a byte by construction, so the
    /// displaced position is a byte position here; a displaced
    /// read anywhere else (a wider element load) stays the
    /// explicit projection error `project_value` raises.
    fn read_bulk_base(&self, frames: &[Frame], target: &Resolved) -> InterpResult<Value> {
        match self.read_resolved(frames, target) {
            Ok(v) => Ok(v),
            Err(e) => match self.displaced_scalar_byte(frames, target) {
                Ok(Some(byte)) => Ok(byte),
                _ => Err(e),
            },
        }
    }

    /// The byte at the displacement of a target whose projection
    /// ends in a byte-offset step into a scalar's storage, when
    /// the base under the displacement is a scalar; `None` for
    /// every other shape.
    fn displaced_scalar_byte(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let (base_target, disp) = match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => match projection.split_last() {
                Some((&RefProj::ByteOffset(d), head)) if d >= 0 => (
                    Resolved::Place {
                        frame_pos: *frame_pos,
                        local: *local,
                        projection: head.to_vec(),
                    },
                    d as usize,
                ),
                _ => return Ok(None),
            },
            Resolved::Static { name, projection } => match projection.split_last() {
                Some((&RefProj::ByteOffset(d), head)) if d >= 0 => (
                    Resolved::Static {
                        name: name.clone(),
                        projection: head.to_vec(),
                    },
                    d as usize,
                ),
                _ => return Ok(None),
            },
            Resolved::Alloc { id, projection } => match projection.split_last() {
                Some((&RefProj::ByteOffset(d), head)) if d >= 0 => (
                    Resolved::Alloc {
                        id: *id,
                        projection: head.to_vec(),
                    },
                    d as usize,
                ),
                _ => return Ok(None),
            },
            Resolved::Value(base, projection) => match projection.split_last() {
                Some((&RefProj::ByteOffset(d), head)) if d >= 0 => (
                    Resolved::Value(base.clone(), head.to_vec()),
                    d as usize,
                ),
                _ => return Ok(None),
            },
        };
        let base = self.read_resolved(frames, &base_target)?;
        Ok(scalar_storage_byte(&base, disp).map(|byte| Value::Uint(u128::from(byte))))
    }

    /// The element copy behind `copy_nonoverlapping`, shared by the
    /// `StatementKind::Intrinsic` form (a direct
    /// `core::intrinsics::copy_nonoverlapping` call in MIR) and the
    /// `std::ptr::copy_nonoverlapping` / `std::ptr::copy` builtin
    /// (the same operation arriving as an ordinary call). The source
    /// elements are snapshotted before any write, so the copy is
    /// exact even if the ranges overlapped (which the
    /// non-overlapping contract forbids, and which makes the shared
    /// implementation a correct `memmove` for `ptr::copy`).
    ///
    /// `elem` is the copy's element type when the call site knows
    /// it (both call paths take it from the pointer operands'
    /// declared types); with `None` the copy reads the element
    /// boundary off the values' shapes alone, as it did before the
    /// type was threaded through.
    pub(crate) fn exec_copy_nonoverlapping(
        &self,
        frames: &mut Vec<Frame>,
        src_value: Value,
        dst_value: Value,
        count_value: Value,
        elem: Option<&Ty>,
    ) -> InterpResult<()> {
                let n = match count_value {
                    Value::Uint(n) => usize::try_from(n).map_err(|_| {
                        InterpError::type_mismatch(
                            "copy_nonoverlapping count",
                            "count that fits in usize",
                            format!("{n}"),
                        )
                    })?,
                    Value::Int(n) if n >= 0 => n as usize,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "copy_nonoverlapping count",
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                };
                if n == 0 {
                    // A zero-count copy touches no memory: the
                    // pointers are never dereferenced, so even a
                    // null or dangling pointer is a legal operand.
                    return Ok(());
                }
                let src_target = self.deref_value(frames, &src_value)?;
                let dst_target = self.deref_value(frames, &dst_value)?;
                // Snapshot the source elements before writing: the
                // intrinsic's contract is that the ranges do not
                // overlap, and the snapshot keeps the copy exact
                // element-for-element either way. A pointer in this
                // model denotes either an aggregate (an array place,
                // a fat pointer's base) whose elements are indexed, or
                // a single element, which a count of 1 copies directly
                // and larger counts continue at its siblings (the
                // sequence access in `read_seq_at` / `write_seq_at`).
                // A byte-granular copy whose other end is one
                // structured scalar value — the byte copy an
                // unaligned vector load/store lowers to, filling
                // or draining a vector local — moves the bytes
                // through the value's leaf structure: the
                // indexed access cannot address a vector value's
                // bytes (it would walk the storage array's lanes
                // as elements). Only whole structured values are
                // rerouted; every other shape keeps the element
                // path below.
                if n > 1 {
                    let dst_tree = self
                        .read_resolved(frames, &dst_target)
                        .ok()
                        .filter(|v| is_scalar_tree(v));
                    let src_tree = if dst_tree.is_none() {
                        self.read_resolved(frames, &src_target)
                            .ok()
                            .filter(|v| is_scalar_tree(v))
                    } else {
                        None
                    };
                    if dst_tree.is_some() || src_tree.is_some() {
                        let bytes = match &src_tree {
                            Some(tree) => tree_to_bytes(tree, n),
                            None => {
                                // The source's units as bytes: a
                                // unit can be a one-field wrapper
                                // over the byte (a control byte's
                                // tag in a raw block), which the
                                // byte view unwraps.
                                fn seq_byte(value: &Value) -> Option<u8> {
                                    match value {
                                        Value::Uint(b) if *b <= 0xFF => Some(*b as u8),
                                        Value::Int(b) if (0..=255).contains(b) => {
                                            Some(*b as u8)
                                        }
                                        Value::Enum { fields, .. } if fields.len() == 1 => {
                                            seq_byte(&fields[0])
                                        }
                                        _ => None,
                                    }
                                }
                                let mut bytes = Vec::with_capacity(n);
                                let mut all_bytes = true;
                                for i in 0..n {
                                    match self.read_seq_at(frames, &src_target, i) {
                                        Ok(v) => match seq_byte(&v) {
                                            Some(b) => bytes.push(b),
                                            None => {
                                                all_bytes = false;
                                                break;
                                            }
                                        },
                                        _ => {
                                            all_bytes = false;
                                            break;
                                        }
                                    }
                                }
                                if all_bytes {
                                    Some(bytes)
                                } else {
                                    None
                                }
                            }
                        };
                        if let Some(bytes) = bytes {
                            if let Some(tree) = &dst_tree {
                                if let Some(patched) = byte_patch_tree(tree, &bytes) {
                                    self.write_resolved(frames, &dst_target, patched)?;
                                    return Ok(());
                                }
                            }
                            if let Some(tree) = &src_tree {
                                // The copy moves exactly the source
                                // value's bytes (`tree_to_bytes`
                                // only succeeds at the value's own
                                // length): write the value itself as
                                // one slot. Writing its bytes one by
                                // one would shred the structured
                                // value into byte slots that a later
                                // whole-value read cannot reassemble
                                // (a hash table's resize copying its
                                // buckets byte-granularly).
                                let tree = tree.clone();
                                self.write_resolved(frames, &dst_target, tree)?;
                                return Ok(());
                            }
                        }
                    }
                }
                let aggregate = |v: &Value| {
                    matches!(
                        v,
                        Value::Tuple(_) | Value::FatRef { .. } | Value::Bytes(_) | Value::Str(_)
                    )
                };
                let src_base = self.read_bulk_base(frames, &src_target)?;
                // The elements are values of the element type. When
                // that type is known and the value found at the
                // source IS one element -- possibly inside a
                // one-field wrapper, since a pointer into a
                // transparent wrapper names the wrapper place, whose
                // only field is the pointee at that address -- the
                // first element is that value, not the first lane
                // of it. (This is the shape `Simd::load`'s copy
                // has: the element is an array, and the source is
                // the array itself.) Without a known element type
                // the shape rule applies: a non-aggregate source is
                // one element, an aggregate source is indexed.
                let src_single = match elem {
                    Some(ty) => {
                        value_fits_ty(crate::builtins::unwrap_wrapper(&src_base), ty)
                    }
                    None => n == 1 && !aggregate(&src_base),
                };
                let elems: Vec<Value> = if src_single {
                    let mut elems = Vec::with_capacity(n);
                    elems.push(crate::builtins::unwrap_wrapper(&src_base).clone());
                    for i in 1..n {
                        elems.push(self.read_seq_at(frames, &src_target, i)?);
                    }
                    elems
                } else {
                    let mut elems = Vec::with_capacity(n);
                    for i in 0..n {
                        // A unit of the source range that was
                        // never written copies as uninit: a
                        // byte-granular copy over a raw block
                        // spans model slots that carry no value
                        // (the block's element occupies one slot;
                        // its byte size counts the rest).
                        let unit = match self.read_seq_at(frames, &src_target, i) {
                            Ok(v) => v,
                            Err(InterpError::UninitializedRead { .. }) => Value::Uninit,
                            Err(e) => return Err(e),
                        };
                        elems.push(unit);
                    }
                    elems
                };
                // The destination, mirroring the source: with a
                // known element type it denotes one element when
                // the value found there is not an aggregate of
                // elements (a wrapper place being filled, a scalar
                // place, an element-shaped aggregate itself). An
                // unreadable (not yet written) destination is a
                // buffer being filled: the indexed path is the one
                // that can create its elements.
                let dst_current = self.read_resolved(frames, &dst_target).ok();
                let dst_single = match elem {
                    Some(ty) => match &dst_current {
                        Some(cur) => {
                            !aggregate(cur)
                                || value_fits_ty(
                                    crate::builtins::unwrap_wrapper(cur),
                                    ty,
                                )
                        }
                        None => false,
                    },
                    None => {
                        let dst_aggregate = dst_current
                            .as_ref()
                            .map(|v| aggregate(v))
                            .unwrap_or(true);
                        n == 1 && !dst_aggregate
                    }
                };
                if dst_single {
                    // The destination place's declared type, when
                    // the target is a whole local of some frame: a
                    // place whose declared type IS the element type
                    // holds the element itself, not a wrapper
                    // around it, so its value is replaced outright;
                    // any other whole place the pointer names is a
                    // wrapper whose payload slot the element fills
                    // (the write-side mirror of `unwrap_wrapper`,
                    // as atomic stores do).
                    let dst_place_ty = match &dst_target {
                        Resolved::Place {
                            frame_pos,
                            local,
                            projection,
                        } if projection.is_empty() => frames
                            .get(*frame_pos)
                            .and_then(|f| self.program.get(&f.func))
                            .and_then(|b| b.local_ty(Local(*local)))
                            .cloned(),
                        _ => None,
                    };
                    let mut elems = elems.into_iter();
                    let first = elems.next().expect("count is nonzero");
                    let stored = match &dst_current {
                        Some(cur) if dst_place_ty.as_ref() != elem => {
                            crate::builtins::rewrap_like(cur, first)
                        }
                        _ => first,
                    };
                    self.write_resolved(frames, &dst_target, stored)?;
                    for (i, value) in elems.enumerate() {
                        self.write_seq_at(frames, &dst_target, i + 1, value)?;
                    }
                } else {
                    for (i, value) in elems.into_iter().enumerate() {
                        self.write_seq_at(frames, &dst_target, i, value)?;
                    }
                }
                Ok(())
    }

    // ----- place resolution --------------------------------------------------
    //
    // A place is resolved against the frame stack. `Field`, `Index`,
    // `ConstantIndex` and `Downcast` steps accumulate as constant
    // `RefProj` steps on a (frame, local) pair; a `Deref` step reads
    // the reference value found there and continues at its target.
    // `Index` locals are read in the *executing* frame: a place is
    // syntactic to the function that contains it, so its index locals
    // always live there, even when earlier steps crossed into another
    // frame through a reference.

    pub(crate) fn frame_pos(&self, frames: &[Frame], id: u64) -> InterpResult<usize> {
        frames.iter().position(|f| f.id == id).ok_or_else(|| {
            InterpError::type_mismatch(
                "dereference",
                "reference into a live frame",
                format!("reference into finished frame #{id}"),
            )
        })
    }

    /// Classify a `BoxDerefTransmute` cast operand: when the
    /// operand is a place naming a `Box` or one of its pointer
    /// wrappers (`Unique` / `NonNull`), return the resolved operand
    /// place, the resolved place of the pointee, and the number of
    /// single-field wrapper layers an aggregate-built box value
    /// carries above its pointer (the box itself carries 3 — Box,
    /// Unique, NonNull — and each field already projected away
    /// removes one). Returns `None` for any other operand, whose
    /// cast keeps the strict wrapper-chain reading.
    fn box_cast_places(
        &self,
        frames: &[Frame],
        body: &Body,
        operand: &Operand,
    ) -> Option<(Resolved, Resolved, usize)> {
        let place = match operand {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Constant(_) => return None,
        };
        let Ty::Adt(name) = body.local_ty(place.local)? else {
            return None;
        };
        let segment = name.rsplit("::").next().unwrap_or(name.as_str());
        let is_box = matches!(name.as_str(), "std::boxed::Box" | "alloc::boxed::Box");
        let (depth, pointee_field) = match (segment, place.projection.as_slice()) {
            ("Box", []) if is_box => (3, true),
            ("Box", [ProjectionElem::Field(0)]) if is_box => (2, false),
            ("Unique", []) => (2, false),
            ("NonNull", []) => (1, false),
            _ => return None,
        };
        let operand_target = self.resolve_place(frames, body, place).ok()?;
        // A built box (see `Interpreter::new_box`) holds its
        // pointee in the allocation store, not in the aggregate:
        // when the operand value is the box aggregate itself, or
        // the allocation pointer stored in its field 0 (or in a
        // one-field pointer wrapper above it), the pointee place
        // is the allocation itself. Naming the allocation keeps
        // the extracted pointer valid after this frame is gone
        // and makes every copy of the box extract the same
        // pointer — the two properties of a real heap pointer
        // this cast exists to expose.
        if let Ok(value) = self.read_resolved(frames, &operand_target) {
            let alloc = box_alloc_pointer(&value).or_else(|| match &value {
                Value::Ref(RefTarget::Alloc { id, projection }) => {
                    Some((*id, projection.clone()))
                }
                _ => None,
            });
            if let Some((id, projection)) = alloc {
                return Some((
                    operand_target,
                    Resolved::Alloc { id, projection },
                    depth,
                ));
            }
        }
        let Resolved::Place {
            frame_pos,
            local,
            projection,
        } = operand_target.clone()
        else {
            return None;
        };
        let mut pointee_projection = projection.clone();
        if pointee_field {
            pointee_projection.push(RefProj::Field(0));
        }
        let pointee_target = Resolved::Place {
            frame_pos,
            local,
            projection: pointee_projection,
        };
        Some((operand_target, pointee_target, depth))
    }

    /// The reference value naming a resolved place (the value a
    /// `BoxDerefTransmute` cast yields in the builtin box model).
    fn pointee_ref(frames: &[Frame], target: &Resolved) -> Value {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => Value::Ref(RefTarget::place(
                frames[*frame_pos].id,
                *local,
                projection.clone(),
            )),
            Resolved::Value(base, projection) => {
                Value::Ref(RefTarget::Value(
                    Box::new(base.clone()),
                    projection.clone(),
                ))
            }
            Resolved::Static { name, projection } => Value::Ref(RefTarget::Static {
                name: name.clone(),
                projection: projection.clone(),
            }),
            Resolved::Alloc { id, projection } => Value::Ref(RefTarget::Alloc {
                id: *id,
                projection: projection.clone(),
            }),
        }
    }

    pub(crate) fn resolve_place(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> InterpResult<Resolved> {
        self.resolve_place_mode(frames, body, place, true)
    }




    fn resolve_place_mode(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
        deref_elem0: bool,
    ) -> InterpResult<Resolved> {
        let mut cur = Resolved::Place {
            frame_pos: frames.len() - 1,
            local: place.local.0,
            projection: Vec::new(),
        };
        // The length a `Deref` step through a fat pointer carries:
        // dereferencing a sub-slice reference reaches the base
        // place, whose own length is longer, so the fat length is
        // kept alongside and applied to the immediately following
        // index step (the only step it can bound).
        let mut fat_cap: Option<u128> = None;
        let projection_len = place.projection.len();
        for (step_idx, elem) in place.projection.iter().enumerate() {
            let is_last = step_idx + 1 == projection_len;
            cur = match elem {
                ProjectionElem::Field(idx) => {
                    fat_cap = None;
                    self.project_step(frames, cur, RefProj::Field(*idx), body, place)?
                }
                ProjectionElem::UnionField { idx, table } => {
                    fat_cap = None;
                    self.project_step(
                        frames,
                        cur,
                        RefProj::UnionField {
                            idx: *idx,
                            table: *table,
                        },
                        body,
                        place,
                    )?
                }
                ProjectionElem::Index(local) => {
                    let idx = self.read_index_local(frames, body, *local, place)?;
                    let via_fat = fat_cap.is_some_and(|cap| {
                        !self.deref_denotes_storage(frames, &cur, cap)
                    });
                    if let Some(cap) = fat_cap {
                        if idx as u128 >= cap {
                            return Err(index_oob(cap as usize, idx));
                        }
                    }
                    fat_cap = None;
                    match via_fat.then(|| compose_fat_index(&cur, idx)).flatten() {
                        Some(composed) => composed,
                        None => {
                            self.project_step(frames, cur, RefProj::Index(idx), body, place)?
                        }
                    }
                }
                ProjectionElem::ConstantIndex { offset, from_end } => {
                    // The length the index is bounded by: the fat
                    // length when the place was reached through a
                    // fat pointer (the value resolved at an
                    // element position has no length of its own
                    // to ask), otherwise the resolved value's own.
                    let len = match fat_cap {
                        Some(cap) => cap,
                        None => self.read_resolved(frames, &cur)?.len().ok_or_else(|| {
                            InterpError::type_mismatch(
                                format!("constant index into {place}"),
                                "array, slice or string value",
                                "value without a length",
                            )
                        })?,
                    };
                    let via_fat = fat_cap.is_some_and(|cap| {
                        !self.deref_denotes_storage(frames, &cur, cap)
                    });
                    fat_cap = None;
                    let idx = if *from_end {
                        len.checked_sub(*offset as u128).ok_or_else(|| {
                            index_oob(len as usize, *offset as usize)
                        })?
                    } else if *offset as u128 >= len {
                        return Err(index_oob(len as usize, *offset as usize));
                    } else {
                        *offset as u128
                    };
                    let idx = usize::try_from(idx).map_err(|_| {
                        InterpError::panic(format!(
                            "index out of bounds: the len is {len} but the index is {idx}"
                        ))
                    })?;
                    match via_fat.then(|| compose_fat_index(&cur, idx)).flatten() {
                        Some(composed) => composed,
                        None => {
                            match via_fat.then(|| compose_fat_index(&cur, idx)).flatten() {
                        Some(composed) => composed,
                        None => {
                            self.project_step(frames, cur, RefProj::Index(idx), body, place)?
                        }
                    }
                        }
                    }
                }
                ProjectionElem::Subslice { from, to, from_end } => {
                    // The range is resolved against the length the
                    // base place currently has: the fat length when
                    // the base was reached through a fat deref,
                    // otherwise the base value's own length. The
                    // resolved step carries an absolute offset and
                    // element count, so later steps (and reference
                    // targets formed from this place) need no
                    // knowledge of the MIR range shape.
                    let len = match fat_cap {
                        Some(cap) => cap,
                        None => self.read_resolved(frames, &cur)?.len().ok_or_else(|| {
                            InterpError::type_mismatch(
                                format!("subslice of {place}"),
                                "array, slice or string value",
                                "value without a length",
                            )
                        })?,
                    };
                    fat_cap = None;
                    let (offset, sub_len) = subslice_offset_len(*from, *to, *from_end, len)?;
                    self.project_step(
                        frames,
                        cur,
                        RefProj::Subslice {
                            offset,
                            len: sub_len,
                        },
                        body,
                        place,
                    )?
                }
                ProjectionElem::Downcast(variant) => {
                    fat_cap = None;
                    self.project_step(frames, cur, RefProj::Downcast(*variant), body, place)?
                }
                ProjectionElem::Deref => {
                    let value = self.read_resolved(frames, &cur)?;
                    fat_cap = match &value {
                        Value::FatRef { len, .. } => Some(*len),
                        _ => None,
                    };
                    let resolved = self.deref_value(frames, &value)?;
                    // A raw allocation block holds its elements
                    // as a tuple: a trailing dereference of its
                    // base pointer denotes the first element,
                    // not the block as a whole. The block as a
                    // whole is only reached by a projection that
                    // continues past the dereference (or by a
                    // write of a tuple value, which replaces
                    // it) — the read and write sides agree on
                    // this. Fat pointers are exempt: dereferencing
                    // one denotes the whole slice, whose length
                    // the caller pairs with the base.
                    if deref_elem0 && is_last && matches!(value, Value::Ref(_)) {
                        if let Resolved::Alloc { id, projection } = &resolved {
                            if projection.is_empty()
                                && self.is_elem_block(*id)
                                && matches!(
                                    self.alloc_base(*id).ok().as_ref(),
                                    Some(Value::Tuple(_) | Value::Bytes(_))
                                )
                            {
                                return Ok(Resolved::Alloc {
                                    id: *id,
                                    projection: vec![RefProj::Index(0)],
                                });
                            }
                        }
                    }
                    resolved
                }
            };
        }
        Ok(cur)
    }

    /// The length carried by the fat pointer a place's final
    /// `Deref` step traverses (`(*p)` where `p` holds a fat
    /// reference), when there is one. Resolving that place reaches
    /// the base place and loses the fat length, so consumers that
    /// need the pointee's own length (`Len`, reference formation)
    /// recover it here from the pointer value itself.
    /// Whether the position a fat-pointer dereference resolved
    /// to holds the slice's own storage: an aggregate value whose
    /// element count is the fat length (a `Vec` value's fields are
    /// its elements; an unsized array value is its element tuple).
    /// Indexing such a position goes INTO the aggregate, so a
    /// following index step is appended as usual. When the
    /// position instead holds one element of the container the
    /// slice lives in (a raw-pointer-derived subslice base), the
    /// value there is a single element — a scalar, or an aggregate
    /// whose own count is the element's, not the slice's — and the
    /// index step composes with the base position instead (see
    /// `compose_fat_index`). A position that cannot be read does
    /// not denote storage.
    fn deref_denotes_storage(
        &self,
        frames: &[Frame],
        cur: &Resolved,
        fat_len: u128,
    ) -> bool {
        match self.read_resolved(frames, cur) {
            Ok(value) => value.len() == Some(fat_len),
            Err(_) => false,
        }
    }

    pub(crate) fn fat_deref_len(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> Option<u128> {
        if !matches!(place.projection.last(), Some(ProjectionElem::Deref)) {
            return None;
        }
        let prefix = Place {
            local: place.local,
            projection: place.projection[..place.projection.len() - 1].to_vec(),
        };
        match self.read_place(frames, body, &prefix) {
            Ok(Value::FatRef { len, .. }) => Some(len),
            _ => None,
        }
    }

    fn project_step(
        &self,
        frames: &[Frame],
        cur: Resolved,
        step: RefProj,
        body: &Body,
        place: &Place,
    ) -> InterpResult<Resolved> {
        match cur {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(step);
                Ok(Resolved::Place {
                    frame_pos,
                    local,
                    projection,
                })
            }
            Resolved::Value(base, projection) => {
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| body.name.clone());
                let mut projection = projection.clone();
                projection.push(step);
                let projected =
                    self.project_value_t(base, &projection, &func, &place.to_string())?;
                Ok(Resolved::Value(projected, Vec::new()))
            }
            Resolved::Static {
                name,
                mut projection,
            } => {
                projection.push(step);
                Ok(Resolved::Static { name, projection })
            }
            Resolved::Alloc {
                id,
                mut projection,
            } => {
                projection.push(step);
                Ok(Resolved::Alloc { id, projection })
            }
        }
    }

    /// Read an `Index` projection's local in the executing frame.
    fn read_index_local(
        &self,
        frames: &[Frame],
        body: &Body,
        local: Local,
        place: &Place,
    ) -> InterpResult<usize> {
        let frame = frames.last().expect("frame");
        let value = frame
            .locals
            .get(local.0)
            .cloned()
            .ok_or_else(|| InterpError::UnknownLocal {
                func: body.name.clone(),
                local: local.0,
            })?;
        match value {
            Value::Uint(v) => usize::try_from(v).map_err(|_| {
                InterpError::panic(format!("index {v} does not fit in usize"))
            }),
            Value::Int(v) if v >= 0 => usize::try_from(v).map_err(|_| {
                InterpError::panic(format!("index {v} does not fit in usize"))
            }),
            Value::Int(v) => Err(InterpError::panic(format!(
                "index out of bounds: negative index {v} used for {place}"
            ))),
            Value::Uninit => Err(InterpError::UninitializedRead {
                func: body.name.clone(),
                place: place.to_string(),
            }),
            other => Err(InterpError::type_mismatch(
                format!("index of {place}"),
                "integer index",
                other.kind_name(),
            )),
        }
    }

    /// Follow a reference value to what it points at.
    pub(crate) fn deref_value(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Resolved> {
        match value {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                projection,
            }) => Ok(Resolved::Place {
                frame_pos: self.frame_pos(frames, *frame)?,
                local: *local,
                projection: projection.clone(),
            }),
            Value::Ref(RefTarget::Value(v, projection)) => {
                Ok(Resolved::Value((**v).clone(), projection.clone()))
            }
            Value::Ref(RefTarget::Static { name, projection }) => Ok(Resolved::Static {
                name: name.clone(),
                projection: projection.clone(),
            }),
            Value::Ref(RefTarget::Alloc { id, projection }) => Ok(Resolved::Alloc {
                id: *id,
                projection: projection.clone(),
            }),
            Value::Ref(RefTarget::Null) => Err(InterpError::type_mismatch(
                "dereference",
                "reference",
                "null pointer",
            )),
            Value::Ref(RefTarget::Address(_)) => Err(InterpError::type_mismatch(
                "dereference",
                "reference",
                "address-only pointer (no pointee in this model)",
            )),
            // A byte-array reference local holds the byte data
            // itself (see `normalize_for_ty`), and a `&str` value
            // that was materialised as bare string data likewise:
            // dereferencing yields the data value.
            Value::Bytes(_) | Value::Str(_) => {
                Ok(Resolved::Value(value.clone(), Vec::new()))
            }
            // Dereferencing a fat pointer whose base is a reference
            // reaches through to the base place. A self-contained
            // fat pointer (a slice constant: the base is the element
            // aggregate itself) derefs to the slice value itself, so
            // `Len` and index bounds see the fat pointer's length.
            Value::FatRef { base, .. } => match &**base {
                Value::Tuple(_) => Ok(Resolved::Value(value.clone(), Vec::new())),
                // An address-only base has no pointee to reach: the
                // fat pointer is self-contained (a dangling slice
                // constant, a `from_raw_parts` result), and
                // dereferencing it yields the slice value itself —
                // the same treatment as the aggregate-base case.
                Value::Ref(RefTarget::Address(_)) => {
                    Ok(Resolved::Value(value.clone(), Vec::new()))
                }
                _ => self.deref_value(frames, base),
            },
            // Dereferencing a dyn fat pointer reaches the value it
            // was formed over: through the base pointer for a
            // reference or slice base, and to the held pointee for
            // a box base (the box aggregate itself — exactly what
            // `Deref::deref` for `Box` yields).
            Value::DynRef { base, .. } => {
                // A reference-counted base names its pointee in the
                // allocation store, not inside the handle value.
                if let Some(id) = rc_handle_id(base) {
                    return Ok(Resolved::Alloc {
                        id,
                        projection: vec![RefProj::Field(2)],
                    });
                }
                match &**base {
                    // A built box base: its single field is the
                    // allocation pointer, and the pointee lives in
                    // the allocation store behind it.
                    Value::Enum { fields, .. }
                        if fields.len() == 1
                            && matches!(fields[0], Value::Ref(RefTarget::Alloc { .. })) =>
                    {
                        self.deref_value(frames, &fields[0])
                    }
                    Value::Enum { fields, .. } if fields.len() == 1 => {
                        Ok(Resolved::Value(fields[0].clone(), Vec::new()))
                    }
                    _ => self.deref_value(frames, base),
                }
            },
            // Dereferencing a built box value itself (or a
            // one-field pointer wrapper holding an allocation
            // pointer, such as a `NonNull` extracted from a built
            // box) reaches the allocation's storage: the field is
            // the pointer, and the pointee lives behind it.
            Value::Enum { fields, .. }
                if fields.len() == 1
                    && matches!(fields[0], Value::Ref(RefTarget::Alloc { .. })) =>
            {
                match &fields[0] {
                    Value::Ref(RefTarget::Alloc { id, projection }) => {
                        Ok(Resolved::Alloc {
                            id: *id,
                            projection: projection.clone(),
                        })
                    }
                    _ => unreachable!("field shape checked by the guard"),
                }
            }
            other => Err(InterpError::type_mismatch(
                "dereference",
                "reference",
                other.kind_name(),
            )),
        }
    }

    /// The pointer a transparent-wrapper cast produces (see
    /// [`Rvalue::PtrField`]): the operand pointer with a field-0
    /// projection appended, so it denotes the wrapper's only
    /// field. A pointer to a constant aggregate is projected
    /// eagerly (such pointers are read-only in this model anyway,
    /// so nothing observable is lost); null and address-only
    /// pointers denote no field and pass through unchanged.
    pub(crate) fn ptr_field_value(
        &self,
        frames: &[Frame],
        body: &Body,
        value: Value,
        depth: u32,
    ) -> InterpResult<Value> {
        match value {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                mut projection,
            }) => {
                projection.extend((0..depth).map(|_| RefProj::Field(0)));
                Ok(Value::Ref(RefTarget::Place {
                    frame,
                    local,
                    projection,
                }))
            }
            Value::Ref(RefTarget::Static {
                name,
                mut projection,
            }) => {
                projection.extend((0..depth).map(|_| RefProj::Field(0)));
                Ok(Value::Ref(RefTarget::Static { name, projection }))
            }
            Value::Ref(RefTarget::Alloc {
                id,
                mut projection,
            }) => {
                projection.extend((0..depth).map(|_| RefProj::Field(0)));
                Ok(Value::Ref(RefTarget::Alloc { id, projection }))
            }
            Value::Ref(RefTarget::Value(v, projection)) => {
                let mut projection = projection.clone();
                projection.extend((0..depth).map(|_| RefProj::Field(0)));
                Ok(Value::Ref(RefTarget::Value(v.clone(), projection)))
            }
            Value::Ref(RefTarget::Null) | Value::Ref(RefTarget::Address(_)) => Ok(value),
            Value::FatRef { base, len } => match *base {
                Value::Ref(_) => {
                    let base = self.ptr_field_value(frames, body, *base, depth)?;
                    Ok(Value::FatRef {
                        base: Box::new(base),
                        len,
                    })
                }
                ref other => Err(InterpError::type_mismatch(
                    "transparent wrapper pointer cast",
                    "fat pointer over a reference",
                    other.kind_name(),
                )),
            },
            other => Err(InterpError::type_mismatch(
                "transparent wrapper pointer cast",
                "pointer to an UnsafeCell/MaybeUninit wrapper",
                other.kind_name(),
            )),
        }
    }

    /// The pointer an array / slice / `str` to element cast
    /// produces (see [`Rvalue::PtrFirstElem`]): the operand pointer
    /// retargeted at the aggregate's first element. A thin pointer
    /// denotes the aggregate itself (its type says so), so an
    /// index-0 projection is appended. A fat pointer's base can
    /// denote either the slice's own storage (a `Vec` value, an
    /// unsized array value: the aggregate whose elements are the
    /// slice's) or its first element's position in the container
    /// the slice lives in (a raw-pointer-derived subslice base).
    /// The storage case takes the appended index 0 like the thin
    /// case; the element case is already the first-element
    /// pointer, and appending index 0 there would denote the
    /// element's own first element instead. A base position is the
    /// element case exactly when its projection already ends at
    /// an element index and the value there is not the slice's
    /// storage (see [`Self::deref_denotes_storage`]); a position
    /// denoting a whole scalar or aggregate (a one-element slice
    /// over a lone scalar, say) is storage whatever its shape. The projection is carried
    /// on the target like every other retargeting in this model
    /// (a value target's projection applies when the reference
    /// resolves); a fat pointer loses its metadata, as the cast's
    /// result is a thin pointer; null and address-only pointers
    /// denote no element and pass through unchanged.
    pub(crate) fn ptr_first_elem_value(
        &self,
        frames: &[Frame],
        body: &Body,
        value: Value,
    ) -> InterpResult<Value> {
        match value {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                mut projection,
            }) => {
                projection.push(RefProj::Index(0));
                Ok(Value::Ref(RefTarget::Place {
                    frame,
                    local,
                    projection,
                }))
            }
            Value::Ref(RefTarget::Static {
                name,
                mut projection,
            }) => {
                projection.push(RefProj::Index(0));
                Ok(Value::Ref(RefTarget::Static { name, projection }))
            }
            Value::Ref(RefTarget::Alloc {
                id,
                mut projection,
            }) => {
                projection.push(RefProj::Index(0));
                Ok(Value::Ref(RefTarget::Alloc { id, projection }))
            }
            Value::Ref(RefTarget::Value(v, mut projection)) => {
                projection.push(RefProj::Index(0));
                Ok(Value::Ref(RefTarget::Value(v, projection)))
            }
            Value::Ref(RefTarget::Null) | Value::Ref(RefTarget::Address(_)) => Ok(value),
            Value::FatRef { base, len } => match *base {
                Value::Ref(_) => {
                    let resolved = self.deref_value(frames, &base)?;
                    let ends_at_element = match &resolved {
                        Resolved::Place { projection, .. }
                        | Resolved::Static { projection, .. }
                        | Resolved::Alloc { projection, .. }
                        | Resolved::Value(_, projection) => {
                            matches!(projection.last(), Some(RefProj::Index(_)))
                        }
                    };
                    if ends_at_element
                        && !self.deref_denotes_storage(frames, &resolved, len)
                    {
                        Ok(*base)
                    } else {
                        self.ptr_first_elem_value(frames, body, *base)
                    }
                }
                Value::Tuple(_) | Value::Bytes(_) | Value::Str(_) => Ok(Value::Ref(
                    RefTarget::Value(base, vec![RefProj::Index(0)]),
                )),
                ref other => Err(InterpError::type_mismatch(
                    "first-element pointer cast",
                    "fat pointer over a reference or an aggregate value",
                    other.kind_name(),
                )),
            },
            other => Err(InterpError::type_mismatch(
                "first-element pointer cast",
                "pointer to an array, slice or string",
                other.kind_name(),
            )),
        }
    }

    /// The pointer an element-to-array cast produces (see
    /// [`Rvalue::PtrArrayBase`]): the operand pointer with a
    /// trailing index-0 projection removed, so it denotes the
    /// array as a whole again. Any other shape passes through
    /// unchanged: a pointer to an element past the first has no
    /// array-base representation in this model.
    /// The element values of an array value: a tuple of
    /// elements, or the bytes of a byte array (each byte is its
    /// own element value). Anything else is not an array value.
    fn array_elem_values(value: Value) -> InterpResult<Vec<Value>> {
        match value {
            Value::Tuple(elems) => Ok(elems),
            Value::Bytes(bytes) => Ok(bytes
                .into_iter()
                .map(|b| Value::Uint(b as u128))
                .collect()),
            other => Err(InterpError::type_mismatch(
                "array element wrapper transmute",
                "an array value",
                other.kind_name(),
            )),
        }
    }

    /// `[T; N]` -> `[MaybeUninit<T>; N]` (or the `UnsafeCell`
    /// twin): each element takes the wrapper's value
    /// representation, an ADT value holding the payload at
    /// field 0 -- the shape `MaybeUninit::new` constructions
    /// are normalized to.
    fn wrap_array_elems(value: Value) -> InterpResult<Value> {
        Ok(Value::Tuple(
            Self::array_elem_values(value)?
                .into_iter()
                .map(|elem| Value::Enum {
                    variant: 0,
                    fields: vec![elem],
                })
                .collect(),
        ))
    }

    /// The reverse cast: each element's wrapper representation
    /// is unwrapped to the payload it holds. The payload sits
    /// at field 0 in the normalized representation, or in the
    /// union's value field (index 1) in a raw aggregate that
    /// still carries the uninitialized field before it.
    fn unwrap_array_elems(value: Value) -> InterpResult<Value> {
        let elems = Self::array_elem_values(value)?
            .into_iter()
            .map(|elem| match elem {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    Ok(fields.into_iter().next().expect("one field"))
                }
                Value::Enum { fields, .. } if fields.len() == 2 => {
                    Ok(fields.into_iter().nth(1).expect("two fields"))
                }
                other => Err(InterpError::type_mismatch(
                    "array element wrapper transmute",
                    "a wrapper element value",
                    other.kind_name(),
                )),
            })
            .collect::<InterpResult<Vec<_>>>()?;
        Ok(Value::Tuple(elems))
    }

    pub(crate) fn ptr_array_base_value(&self, value: Value) -> InterpResult<Value> {
        match value {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                mut projection,
            }) => {
                if projection.last() == Some(&RefProj::Index(0)) {
                    projection.pop();
                }
                Ok(Value::Ref(RefTarget::Place {
                    frame,
                    local,
                    projection,
                }))
            }
            Value::Ref(RefTarget::Static {
                name,
                mut projection,
            }) => {
                if projection.last() == Some(&RefProj::Index(0)) {
                    projection.pop();
                }
                Ok(Value::Ref(RefTarget::Static { name, projection }))
            }
            Value::Ref(RefTarget::Alloc {
                id,
                mut projection,
            }) => {
                if projection.last() == Some(&RefProj::Index(0)) {
                    projection.pop();
                }
                Ok(Value::Ref(RefTarget::Alloc { id, projection }))
            }
            Value::Ref(RefTarget::Value(v, mut projection)) => {
                if projection.last() == Some(&RefProj::Index(0)) {
                    projection.pop();
                }
                Ok(Value::Ref(RefTarget::Value(v, projection)))
            }
            other => Ok(other),
        }
    }

    /// The borrowing form of [`read_resolved`]: a place read lends
    /// the projected value out of the frame's storage instead of
    /// cloning it, for consumers that only inspect the value (or
    /// clone a small part of it). Reads out of the statics and
    /// allocation stores cannot lend (the stores sit behind
    /// `RefCell`s), and a detached `Resolved::Value` is itself a
    /// copy, so those arms yield an owned value — exactly what
    /// `read_resolved` returns. Errors are identical.
    pub(crate) fn read_resolved_ref<'x>(
        &self,
        frames: &'x [Frame],
        target: &Resolved,
    ) -> InterpResult<Cow<'x, Value>> {
        match target {
            Resolved::Value(base, projection) => {
                let value = self.value_target_value(base, projection)?;
                if value == Value::Uninit {
                    return Err(InterpError::UninitializedRead {
                        func: "<constant>".to_string(),
                        place: "constant aggregate".to_string(),
                    });
                }
                Ok(Cow::Owned(value))
            }
            Resolved::Static { .. } | Resolved::Alloc { .. } => {
                Ok(Cow::Owned(self.read_resolved(frames, target)?))
            }
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                // Union field reads need the typed projector and
                // the owned path's rescue rules; they take the
                // owned read unchanged (the pre-GK behaviour).
                if projection
                    .iter()
                    .any(|step| matches!(step, RefProj::UnionField { .. }))
                {
                    return Ok(Cow::Owned(self.read_resolved(frames, target)?));
                }
                let frame = &frames[*frame_pos];
                let base = frame.locals.get(*local).ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: frame.func.clone(),
                        local: *local,
                    }
                })?;
                let place_desc = if projection.is_empty() {
                    String::new()
                } else {
                    describe_resolved(*local, projection)
                };
                if *base == Value::Uninit {
                    if let Some(zst) = self.zst_local_value(frame, *local) {
                        let value = project_value(zst, projection, &frame.func, &place_desc)?;
                        if value == Value::Uninit {
                            return Err(InterpError::UninitializedRead {
                                func: frame.func.clone(),
                                place: if projection.is_empty() {
                                    describe_resolved(*local, projection)
                                } else {
                                    place_desc
                                },
                            });
                        }
                        return Ok(Cow::Owned(value));
                    }
                }
                let value = match project_value_ref(base, projection, &frame.func, &place_desc) {
                    Ok(v) if *v != Value::Uninit => v,
                    Ok(_) | Err(InterpError::UninitializedRead { .. }) => {
                        // Mirror the owned read: a single-valued
                        // field denotes its value when nothing was
                        // ever written there.
                        match self.zst_field_value(frame, *local, projection) {
                            Some(v) => return Ok(Cow::Owned(v)),
                            None => {
                                return Err(InterpError::UninitializedRead {
                                    func: frame.func.clone(),
                                    place: if projection.is_empty() {
                                        describe_resolved(*local, projection)
                                    } else {
                                        place_desc
                                    },
                                });
                            }
                        }
                    }
                    Err(e) => return Err(e),
                };
                Ok(value)
            }
        }
    }

    /// The size in bytes of an IR type whose layout is fixed by
    /// the type alone or by the program's ADT layout table:
    /// scalars, arrays of such types, pointers (fat when the
    /// pointee is unsized), and the ADT instances the driver
    /// recorded a layout for. Anything else — multi-element
    /// tuples (their field layout is the compiler's choice),
    /// slices, `str`, closures — yields `None`.
    fn ty_size(&self, ty: &Ty) -> Option<u64> {
        Some(match ty {
            Ty::Unit => 0,
            Ty::Bool => 1,
            Ty::Int(t) => u64::from(t.bits() / 8),
            Ty::Uint(t) => u64::from(t.bits() / 8),
            Ty::Float(t) => u64::from(t.bits() / 8),
            Ty::Adt(name) => self.program().adt_layouts.get(name).map(|&(size, _)| size)?,
            Ty::Ref(pointee) => {
                if matches!(&**pointee, Ty::Slice(_) | Ty::Str | Ty::Dynamic(..)) {
                    16
                } else {
                    8
                }
            }
            Ty::Array(elem, n) => n.checked_mul(self.ty_size(elem)?)?,
            Ty::Tuple(ts) if ts.len() <= 1 => match ts.first() {
                Some(t) => self.ty_size(t)?,
                None => 0,
            },
            Ty::FnPtr => 8,
            Ty::FnDef(_) => 0,
            _ => return None,
        })
    }

    /// The element size of a pointer offset whose size conversion
    /// could not bake: derived from the executing body's declared
    /// type of the pointer operand, when the operand is a plain
    /// local of pointer type.
    fn offset_elem_size_of(&self, body: &Body, op: &Operand) -> Option<u64> {
        let place = match op {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Constant(_) => return None,
        };
        if !place.projection.is_empty() {
            return None;
        }
        match body.local_tys.get(place.local.0)? {
            Ty::Ref(pointee) => self.ty_size(pointee),
            _ => None,
        }
    }

    /// The value of a pointer offset (see [`Rvalue::PtrOffset`]):
    /// `base` advanced by `count` elements of `elem_size` bytes
    /// each (the size may be unknown, in which case only the
    /// Absorb a pointer offset into an element index for a
    /// raw allocation block (see the `Alloc` arm of
    /// `offset_target`). Answers `None` when the block's
    /// contents are not element-shaped, the element size is
    /// unknown, or the resulting position is not
    /// element-aligned; the caller then carries the byte
    /// displacement as before.
    fn alloc_offset_absorb(
        &self,
        id: u64,
        projection: &[RefProj],
        elem_size: Option<u64>,
        disp: Option<i128>,
    ) -> Option<InterpResult<Vec<RefProj>>> {
        let size = elem_size?;
        if size == 0 {
            return None;
        }
        if !self.is_elem_block(id) {
            return None;
        }
        let content = self.alloc_base(id).ok()?;
        match &content {
            Value::Tuple(_) => {}
            // A zeroed block holds bytes; only byte elements
            // index it faithfully.
            Value::Bytes(_) if size == 1 => {}
            _ => return None,
        }
        let disp = disp?;
        let (base, carried) = match projection.last() {
            Some(RefProj::ByteOffset(d)) => {
                (&projection[..projection.len() - 1], *d)
            }
            _ => (projection, 0),
        };
        let total = carried.checked_add(disp)?;
        // A pointer already denominated in bytes (its index was
        // rewritten by a differently sized pointer cast, see
        // `PtrCastUnits`) with nothing before the byte offset
        // stays in bytes: converting the total back into the
        // current element size's index would silently change
        // the position whenever the two denominations meet —
        // the byte-denominated bucket pointers of a raw table
        // versus its element-denominated ones. Byte offsets
        // resolve against the block's byte slots directly.
        if base.is_empty() && matches!(projection.last(), Some(RefProj::ByteOffset(_))) {
            return Some(Ok(vec![RefProj::ByteOffset(total)]));
        }
        let size_i = i128::from(size);
        if total % size_i != 0 {
            return None;
        }
        let steps = total / size_i;
        let idx = match base {
            [] => steps,
            [RefProj::Index(j)] => i128::try_from(*j).ok()?.checked_add(steps)?,
            _ => return None,
        };
        if idx < 0 {
            return Some(Err(InterpError::UnsupportedRvalue {
                kind: "pointer offset: negative element index into an allocation"
                    .to_string(),
            }));
        }
        let idx = match usize::try_from(idx) {
            Ok(i) => i,
            Err(_) => {
                return Some(Err(InterpError::UnsupportedRvalue {
                    kind: "pointer offset: element index out of range".to_string(),
                }))
            }
        };
        Some(Ok(vec![RefProj::Index(idx)]))
    }

    /// shapes that need no size — a zero count — evaluate).
    /// `wrapping` selects the `arith_offset` intrinsic's semantics
    /// over `BinaryOp::Offset`'s: the two compute the same denoted
    /// position wherever both are defined, and differ only for a
    /// null base, which wrapping turns into the bare address and
    /// the checked form rejects as the undefined behaviour it is.
    pub(crate) fn ptr_offset_value(
        &self,
        frames: &[Frame],
        base: Value,
        count: i128,
        elem_size: Option<u64>,
        wrapping: bool,
    ) -> InterpResult<Value> {
        if count == 0 {
            return Ok(base);
        }
        // The byte displacement, when the element size is known.
        // A displacement that overflows an i128 is not a position
        // in any allocation and has no honest rendering.
        let disp: Option<i128> = match elem_size {
            Some(size) => Some(count.checked_mul(i128::from(size)).ok_or_else(|| {
                InterpError::UnsupportedRvalue {
                    kind: "pointer offset: element displacement overflows".to_string(),
                }
            })?),
            None => None,
        };
        match base {
            Value::Ref(target) => Ok(Value::Ref(
                self.offset_target(target, frames, count, elem_size, disp, wrapping)?,
            )),
            Value::FatRef { base, len } => match *base {
                Value::Ref(_) => {
                    let base = self.ptr_offset_value(frames, *base, count, elem_size, wrapping)?;
                    Ok(Value::FatRef {
                        base: Box::new(base),
                        len,
                    })
                }
                Value::Tuple(elems) => {
                    // A self-contained slice constant: the offset
                    // data pointer denotes the suffix starting at
                    // the count, with the length metadata unchanged
                    // (exactly what offsetting a fat pointer does).
                    let n = usize::try_from(count).map_err(|_| {
                        InterpError::UnsupportedRvalue {
                            kind: "pointer offset: negative offset of a slice constant"
                                .to_string(),
                        }
                    })?;
                    if n > elems.len() {
                        return Err(InterpError::UnsupportedRvalue {
                            kind: "pointer offset: offset past the end of a slice constant"
                                .to_string(),
                        });
                    }
                    Ok(Value::FatRef {
                        base: Box::new(Value::Tuple(elems[n..].to_vec())),
                        len,
                    })
                }
                ref other => Err(InterpError::type_mismatch(
                    "pointer offset",
                    "fat pointer over a reference or a slice constant",
                    other.kind_name(),
                )),
            },
            Value::DynRef {
                base,
                concrete,
                traits,
                drop_glue,
            } => match *base {
                Value::Ref(_) => {
                    let base = self.ptr_offset_value(frames, *base, count, elem_size, wrapping)?;
                    Ok(Value::DynRef {
                        base: Box::new(base),
                        concrete,
                        traits,
                        drop_glue,
                    })
                }
                ref other => Err(InterpError::type_mismatch(
                    "pointer offset",
                    "dyn pointer over a reference",
                    other.kind_name(),
                )),
            },
            other => Err(InterpError::type_mismatch(
                "pointer offset",
                "pointer value",
                other.kind_name(),
            )),
        }
    }

    /// A reference target advanced by a pointer offset (see
    /// [`Self::ptr_offset_value`]).
    fn offset_target(
        &self,
        target: RefTarget,
        frames: &[Frame],
        count: i128,
        elem_size: Option<u64>,
        disp: Option<i128>,
        wrapping: bool,
    ) -> InterpResult<RefTarget> {
        match target {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                let base = self
                    .frame_pos(frames, frame)
                    .ok()
                    .and_then(|pos| frames[pos].locals.get(local));
                let projection = self.offset_projection(
                    frames,
                    Some((frame, local)),
                    projection,
                    disp,
                    elem_size,
                    base,
                )?;
                Ok(RefTarget::Place {
                    frame,
                    local,
                    projection,
                })
            }
            RefTarget::Static { name, projection } => {
                // A static's declared type is not part of the IR
                // program, so no array absorption is attempted for
                // statics: the displacement is carried unless the
                // static value's own shape absorbs it.
                let base = self.static_base(&name).ok();
                let projection = self.offset_projection(
                    frames,
                    None,
                    projection,
                    disp,
                    elem_size,
                    base.as_ref(),
                )?;
                Ok(RefTarget::Static { name, projection })
            }
            RefTarget::Alloc { id, projection } => {
                // A raw block from the allocator builtins holds
                // its elements as a tuple (or, for a zeroed byte
                // block, as bytes): an offset in element units
                // is absorbed into an element index, the form
                // the block's contents are read and written
                // through. Any other pointee shape is not a
                // declared local type either, so no absorption
                // is attempted and the displacement is carried,
                // as for a static.
                if let Some(absorbed) =
                    self.alloc_offset_absorb(id, &projection, elem_size, disp)
                {
                    return absorbed.map(|projection| RefTarget::Alloc { id, projection });
                }
                let base = self.alloc_base(id).ok();
                let projection = self.offset_projection(
                    frames,
                    None,
                    projection,
                    disp,
                    elem_size,
                    base.as_ref(),
                )?;
                Ok(RefTarget::Alloc { id, projection })
            }
            RefTarget::Value(v, projection) => {
                if projection.is_empty() {
                    // An unprojected value target denotes the
                    // aggregate itself: the offset is applied
                    // eagerly to the constant value.
                    Ok(RefTarget::value(
                        self.offset_value_target(*v, count, elem_size)?,
                    ))
                } else if let Some((&RefProj::Index(i), head)) = projection.split_last() {
                    // A value target projected to an element of
                    // the constant aggregate (the shape a
                    // first-element cast of a constant array or
                    // string reference produces): the offset is
                    // counted in elements of the type the index
                    // denotes, so it absorbs into the index
                    // directly, whatever the element size.
                    let idx = i128::try_from(i).map_err(|_| InterpError::UnsupportedRvalue {
                        kind: "pointer offset: element index out of range".to_string(),
                    })? + count;
                    if idx < 0 {
                        return Err(InterpError::UnsupportedRvalue {
                            kind: "pointer offset: negative offset of a constant array"
                                .to_string(),
                        });
                    }
                    let mut projection = head.to_vec();
                    projection.push(RefProj::Index(usize::try_from(idx).map_err(|_| {
                        InterpError::UnsupportedRvalue {
                            kind: "pointer offset: element index out of range".to_string(),
                        }
                    })?));
                    Ok(RefTarget::Value(v, projection))
                } else {
                    // A projected value target denotes a position
                    // inside the value: carry the displacement on
                    // the projection, as for statics and
                    // allocations, whose pointees are likewise not
                    // declared locals.
                    let projection = self.offset_projection(
                        frames,
                        None,
                        projection,
                        disp,
                        elem_size,
                        Some(&v),
                    )?;
                    Ok(RefTarget::Value(v, projection))
                }
            }
            RefTarget::Null => {
                if wrapping {
                    let d = disp.ok_or_else(|| InterpError::UnsupportedRvalue {
                        kind: "pointer offset: element size is not known".to_string(),
                    })?;
                    Ok(RefTarget::Address(d as u64))
                } else {
                    Err(InterpError::UnsupportedRvalue {
                        kind: "pointer offset: offset of a null pointer".to_string(),
                    })
                }
            }
            RefTarget::Address(addr) => {
                // An address-only pointer has no allocation bounds
                // in this model for the checked form to check; the
                // address arithmetic is the observable behaviour.
                let d = disp.ok_or_else(|| InterpError::UnsupportedRvalue {
                    kind: "pointer offset: element size is not known".to_string(),
                })?;
                Ok(RefTarget::Address(addr.wrapping_add(d as u64)))
            }
        }
    }

    /// A constant aggregate advanced by a pointer offset. Such a
    /// target under an offset denotes its first element (the shape
    /// a cast of a constant array reference to an element pointer
    /// produces), so the offset denotes the count-th element,
    /// projected eagerly — constant aggregates are read-only in
    /// this model, so nothing observable is lost (the same
    /// reasoning as [`Self::ptr_field_value`]'s eager projection).
    /// A one-past-the-end position has no value to denote and
    /// stays an explicit error.
    fn offset_value_target(
        &self,
        value: Value,
        count: i128,
        elem_size: Option<u64>,
    ) -> InterpResult<Value> {
        match value {
            Value::Tuple(elems) => {
                let idx = usize::try_from(count).map_err(|_| InterpError::UnsupportedRvalue {
                    kind: "pointer offset: negative offset of a constant array".to_string(),
                })?;
                elems.get(idx).cloned().ok_or_else(|| InterpError::UnsupportedRvalue {
                    kind: "pointer offset: offset at or past the end of a constant array"
                        .to_string(),
                })
            }
            Value::Bytes(bytes) => {
                // A pointer into constant byte storage: only a
                // byte-sized element type denotes a single byte
                // of it.
                if elem_size != Some(1) {
                    return Err(InterpError::UnsupportedRvalue {
                        kind: "pointer offset: offset of a byte-string pointer with a                                non-byte element type"
                            .to_string(),
                    });
                }
                let idx = usize::try_from(count).map_err(|_| InterpError::UnsupportedRvalue {
                    kind: "pointer offset: negative offset of a byte string".to_string(),
                })?;
                bytes
                    .get(idx)
                    .map(|b| Value::Uint(u128::from(*b)))
                    .ok_or_else(|| InterpError::UnsupportedRvalue {
                        kind: "pointer offset: offset past the end of a byte string"
                            .to_string(),
                    })
            }
            other => Err(InterpError::UnsupportedRvalue {
                kind: format!(
                    "pointer offset: offset of a pointer to a constant {} value is not modelled",
                    other.kind_name()
                ),
            }),
        }
    }

    /// Advance a place/static target's projection by a byte
    /// displacement (`disp`, in bytes; `None` when the offset's
    /// element size is unknown). When the target denotes an element
    /// of an array place (or the array place itself) and the total
    /// displacement lands exactly on an element boundary, it is
    /// absorbed into the index projection, so reads and writes
    /// through the result resolve structurally. The same absorption
    /// applies over any element aggregate the target's value
    /// denotes (see [`Self::absorb_displacement`]). Otherwise the
    /// displacement is carried as a trailing
    /// [`RefProj::ByteOffset`] step: the position is faithfully
    /// denoted (and further offsets accumulate into it), but
    /// reading or writing through it is an explicit error at the
    /// use site. A total displacement of zero never appears in the
    /// result: an offset that cancels a carried one restores the
    /// structural position.
    fn offset_projection(
        &self,
        frames: &[Frame],
        place: Option<(u64, usize)>,
        mut projection: Vec<RefProj>,
        disp: Option<i128>,
        elem_size: Option<u64>,
        base: Option<&Value>,
    ) -> InterpResult<Vec<RefProj>> {
        // A displacement already carried on the target accumulates
        // into this one: the combination is what decides
        // absorption.
        let mut carried: i128 = 0;
        if let Some(RefProj::ByteOffset(prev)) = projection.last() {
            carried = *prev;
            projection.pop();
        }
        let total = match disp {
            Some(d) => carried.checked_add(d).ok_or_else(|| InterpError::UnsupportedRvalue {
                kind: "pointer offset: displacement overflows".to_string(),
            })?,
            None => {
                return Err(InterpError::UnsupportedRvalue {
                    kind: "pointer offset: element size is not known".to_string(),
                })
            }
        };
        if total == 0 {
            return Ok(projection);
        }
        let array_pos = self.array_position(frames, place, &projection);
        if let Some((cur, elem)) = array_pos {
            if elem > 0 {
                if let Some(byte_pos) = cur.checked_mul(elem).and_then(|b| b.checked_add(total)) {
                    if byte_pos >= 0 && byte_pos % elem == 0 {
                        let idx = byte_pos / elem;
                        // The result keeps the input's form: an
                        // index projection denotes the element
                        // itself, while an empty projection denotes
                        // the array place as a whole — different
                        // positions in this model, even at index 0.
                        return Ok(if idx == 0 && projection.is_empty() {
                            Vec::new()
                        } else {
                            vec![RefProj::Index(usize::try_from(idx).map_err(|_| {
                                InterpError::UnsupportedRvalue {
                                    kind: "pointer offset: element index out of range"
                                        .to_string(),
                                }
                            })?)]
                        });
                    }
                }
            }
        }
        // When the declared type cannot judge the position (the
        // target is not an array place), the value's own shape can:
        // a displacement over an element aggregate (a tuple, or an
        // ADT value whose fields are its elements, as a `Vec` value
        // is) in whole units of the offset's element size denotes
        // the element at that index, exactly as the array case
        // above. The array verdict governs whenever it applies, so
        // a byte displacement into a typed array is never
        // reinterpreted this way.
        if array_pos.is_none() {
            if let Some(base) = base {
                if let Some(absorbed) =
                    self.absorb_displacement(base, &projection, total, elem_size)
                {
                    return Ok(absorbed);
                }
            }
        }
        projection.push(RefProj::ByteOffset(total));
        Ok(projection)
    }

    /// Absorb a byte displacement into an element index over the
    /// element aggregate `base` that `projection` denotes a
    /// position in (see the call in
    /// [`Self::offset_projection`]). The projection must navigate
    /// structurally (fields, downcasts, sub-slices) to an element
    /// container — a tuple, a byte string, or an ADT value whose
    /// fields are its elements — and end either at the container
    /// itself (element 0) or at one index step into it. The
    /// displacement must move the position by a whole number of
    /// the offset's element size and land within the container
    /// (one past the end included: such a pointer may be formed,
    /// and reading through it stays an explicit out-of-bounds
    /// error). Anything else answers `None`, and the caller
    /// carries the displacement as before.
    fn absorb_displacement(
        &self,
        base: &Value,
        projection: &[RefProj],
        total: i128,
        elem_size: Option<u64>,
    ) -> Option<Vec<RefProj>> {
        let size = i128::from(elem_size?);
        if size <= 0 {
            return None;
        }
        // Navigate to the container the position sits in. A
        // sub-slice step descends into the sub-aggregate it
        // denotes (materialised on the side; the walk only reads
        // shapes and counts from it), so a position inside a
        // sub-slice of an element container — the back half of a
        // `split_at_mut` on a range slice, say — absorbs exactly
        // like a position in the container itself.
        let mut temps: Vec<Box<Value>> = Vec::new();
        let mut cur: &Value = base;
        let mut prefix = projection.len();
        let mut index: i128 = 0;
        for (j, step) in projection.iter().enumerate() {
            match step {
                RefProj::Field(idx) => {
                    cur = match cur {
                        Value::Tuple(items) => items.get(*idx)?,
                        Value::Enum { fields, .. } => fields.get(*idx)?,
                        Value::Coroutine { fields, .. } => fields.get(*idx)?,
                        Value::Closure { env, .. } => env.get(*idx)?,
                        _ => return None,
                    };
                }
                RefProj::Downcast(_) => {
                    if !matches!(cur, Value::Enum { .. } | Value::Coroutine { .. }) {
                        return None;
                    }
                }
                RefProj::Subslice { offset, len } => {
                    let sub = subslice_value(cur, *offset, *len)?;
                    temps.push(Box::new(sub));
                    cur = temps.last().expect("just pushed");
                }
                RefProj::Index(idx) => {
                    if j + 1 != projection.len() {
                        return None;
                    }
                    prefix = j;
                    index = i128::try_from(*idx).ok()?;
                }
                _ => return None,
            }
        }
        // The container's element count, unwrapping the shapes a
        // pointer position can sit in: a fat reference denotes its
        // base's elements with the carried length, and a
        // single-field ADT whose only field is an element
        // aggregate is indexed through that field (the
        // `#[repr(simd)]` pun `project_value` also honours).
        let mut container: &Value = cur;
        loop {
            match container {
                Value::FatRef { base, .. } => container = base,
                Value::Enum { fields, .. }
                    if fields.len() == 1
                        && matches!(fields[0], Value::Tuple(_) | Value::Bytes(_))
                        && prefix < projection.len() =>
                {
                    container = &fields[0];
                }
                _ => break,
            }
        }
        let count: i128 = match container {
            Value::Tuple(items) => i128::try_from(items.len()).ok()?,
            Value::Enum { fields, .. } => i128::try_from(fields.len()).ok()?,
            Value::Bytes(b) => {
                if size != 1 {
                    return None;
                }
                i128::try_from(b.len()).ok()?
            }
            Value::Str(text) => {
                if size != 1 {
                    return None;
                }
                i128::try_from(text.len()).ok()?
            }
            _ => return None,
        };
        let byte_pos = index.checked_mul(size)?.checked_add(total)?;
        if byte_pos < 0 || byte_pos % size != 0 {
            return None;
        }
        let new_index = byte_pos / size;
        if new_index > count {
            return None;
        }
        let new_index = usize::try_from(new_index).ok()?;
        let mut out: Vec<RefProj> = projection[..prefix].to_vec();
        // Keep the input's form when the position stays at the
        // container base itself (see `offset_projection`).
        if prefix == projection.len() && new_index == 0 {
            return Some(out);
        }
        out.push(RefProj::Index(new_index));
        Some(out)
    }

    /// The array position a place target denotes, when it is
    /// directly determinable from the target local's declared
    /// type: the local is an array and the projection is empty
    /// (the array base, element 0) or a single index into it.
    /// Returns the current element index and the element size.
    fn array_position(
        &self,
        frames: &[Frame],
        place: Option<(u64, usize)>,
        projection: &[RefProj],
    ) -> Option<(i128, i128)> {
        let (frame_id, local) = place?;
        let frame = &frames[self.frame_pos(frames, frame_id).ok()?];
        let body = self.program().functions.get(&frame.func)?;
        let elem = match body.local_tys.get(local)? {
            Ty::Array(elem, _) => self.ty_size(elem)?,
            _ => return None,
        };
        let cur = match projection {
            [] => 0,
            [RefProj::Index(i)] => i128::try_from(*i).ok()?,
            _ => return None,
        };
        Some((cur, i128::from(elem)))
    }

    /// The value a `Resolved::Value` target denotes: its base
    /// projected by the target's pending projection (the base
    /// itself when the projection is empty).
    pub(crate) fn value_target_value(
        &self,
        base: &Value,
        projection: &[RefProj],
    ) -> InterpResult<Value> {
        if projection.is_empty() {
            Ok(base.clone())
        } else {
            self.project_value_t(base.clone(), projection, "<constant>", "constant aggregate")
        }
    }

    pub(crate) fn read_resolved(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> InterpResult<Value> {
        match target {
            Resolved::Value(base, projection) => {
                let value = self.value_target_value(base, projection)?;
                if value == Value::Uninit {
                    return Err(InterpError::UninitializedRead {
                        func: "<constant>".to_string(),
                        place: "constant aggregate".to_string(),
                    });
                }
                Ok(value)
            }
            Resolved::Static { name, projection } => {
                let base = self.static_base(name)?;
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| "<static>".to_string());
                let place_desc = format!("static {name}");
                let value = self.project_value_t(base, projection, &func, &place_desc)?;
                if value == Value::Uninit {
                    return Err(InterpError::UninitializedRead {
                        func,
                        place: place_desc,
                    });
                }
                Ok(value)
            }
            Resolved::Alloc { id, projection } => {
                let projection = &self.elem_block_projection(*id, projection);
                let base = self.alloc_base(*id)?;
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| "<allocation>".to_string());
                let place_desc = format!("allocation #{id}");
                // An elem block's stored tuple holds only the
                // elements written so far (it grows on write; see
                // the allocation arm of `write_resolved`): a read
                // whose first index step lands past the tuple's
                // current end addresses a slot that was never
                // written. That is an uninitialized read, not a
                // guest bounds violation -- the block's extent is
                // its byte size, which the tuple's length does
                // not measure, so the projector's bounds panic
                // does not apply to it.
                let past_unwritten = self.is_elem_block(*id)
                    && match (&base, projection.first()) {
                        (Value::Tuple(items), Some(RefProj::Index(i))) => {
                            *i >= items.len()
                        }
                        _ => false,
                    };
                let value = match self.project_value_t(base, projection, &func, &place_desc) {
                    Ok(v) => v,
                    Err(e) => {
                        if past_unwritten {
                            return Err(InterpError::UninitializedRead {
                                func,
                                place: place_desc,
                            });
                        }
                        return Err(e);
                    }
                };
                if value == Value::Uninit {
                    return Err(InterpError::UninitializedRead {
                        func,
                        place: place_desc,
                    });
                }
                Ok(value)
            }
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let frame = &frames[*frame_pos];
                let base = frame.locals.get(*local).cloned().ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: frame.func.clone(),
                        local: *local,
                    }
                })?;
                // A zero-sized local's place may never be written
                // by rustc MIR at all (a unit struct constructor's
                // body is a bare `return`): an unwritten read of
                // such a local denotes the type's single value.
                let base = if base == Value::Uninit {
                    self.zst_local_value(frame, *local).unwrap_or(base)
                } else {
                    base
                };
                // The description is only read inside projection
                // steps and by the error below, so with an empty
                // projection it is built only if that error fires.
                let place_desc = if projection.is_empty() {
                    String::new()
                } else {
                    describe_resolved(*local, projection)
                };
                // A field whose type has a single value denotes
                // that value when nothing was ever written there:
                // the store of a zero-sized field is elided like a
                // ZST local's (the closure slot of a `catch_unwind`
                // Data union is read back by `do_call` without any
                // body having stored it). The stored value wins
                // whenever the read succeeds — a union's storage
                // slot can be shared with a field of a different
                // type — so the precomputed value (conversion's
                // `Body::zst_fields` table) only rescues a read
                // that otherwise fails as uninitialized.
                let value = match self.project_value_t(base, projection, &frame.func, &place_desc) {
                    Ok(v) if v != Value::Uninit => v,
                    Ok(_) | Err(InterpError::UninitializedRead { .. }) => {
                        match self.zst_field_value(frame, *local, projection) {
                            Some(v) => v,
                            None => {
                                return Err(InterpError::UninitializedRead {
                                    func: frame.func.clone(),
                                    place: if projection.is_empty() {
                                        describe_resolved(*local, projection)
                                    } else {
                                        place_desc
                                    },
                                });
                            }
                        }
                    }
                    Err(e) => return Err(e),
                };
                Ok(value)
            }
        }
    }

    /// The value of a never-written zero-sized local in `frame`
    /// (see the note in `read_resolved`); `None` for any other
    /// local. Conversion records the canonical value of every
    /// provably zero-sized local in the body (`Body::zst_locals`,
    /// computed from the driver's ADT shape tables); for
    /// hand-built bodies the shapes derivable from the IR type
    /// alone are synthesized by [`zst_value_of_ty`].
    fn zst_local_value(&self, frame: &Frame, local: usize) -> Option<Value> {
        let body = self.program.get(&frame.func)?;
        if let Some(Some(v)) = body.zst_locals.get(local) {
            return Some(v.clone());
        }
        zst_value_of_ty(body.local_ty(Local(local))?)
    }

    /// The canonical value of a single-valued field path of a
    /// local (see the note in `read_resolved`); `None` when the
    /// projection is not a pure field path or conversion recorded
    /// no single-valued value for it.
    pub(crate) fn zst_field_value(
        &self,
        frame: &Frame,
        local: usize,
        projection: &[RefProj],
    ) -> Option<Value> {
        if projection.is_empty() {
            return None;
        }
        let mut path = Vec::with_capacity(projection.len());
        for step in projection {
            match step {
                RefProj::Field(idx) => path.push(*idx),
                // Union fields sit in the same per-parent index
                // space the conversion-side walk records.
                RefProj::UnionField { idx, .. } => path.push(*idx),
                _ => return None,
            }
        }
        let body = self.program.get(&frame.func)?;
        body.zst_fields.get(local)?.get(&path).cloned()
    }

    /// The value of a never-written local of an evidently
    /// zero-sized type, for drop glue only (see the note in
    /// `read_at`); `None` for any other local. `read_resolved`
    /// deliberately does not do this: a genuine read of a
    /// never-written place is an uninitialized read, but drop glue
    /// must still run the destructor of a ZST place whose write
    /// optimized MIR elided.
    fn zst_glue_value(&self, frame: &Frame, local: usize) -> Option<Value> {
        let body = self.program.get(&frame.func)?;
        let ty = body.local_ty(Local(local))?;
        Self::zst_value_of(ty)
    }

    /// The unique value of an evidently zero-sized type, when the
    /// type alone determines one. ADT values are fabricated as
    /// variant 0 with no fields: exact for unit-like and fieldless
    /// structs; for payload-carrying types the fabrication is only
    /// a drop-glue gate — field glue beneath it reads as
    /// never-written and is skipped.
    fn zst_value_of(ty: &Ty) -> Option<Value> {
        match ty {
            Ty::Unit => Some(Value::Unit),
            Ty::Tuple(tys) => tys
                .iter()
                .map(Self::zst_value_of)
                .collect::<Option<Vec<_>>>()
                .map(Value::Tuple),
            Ty::Array(elem, n) => {
                if *n > 1_000_000 {
                    return None;
                }
                let elem = Self::zst_value_of(elem)?;
                Some(Value::Tuple(vec![elem; *n as usize]))
            }
            Ty::Closure(def, 0) => Some(Value::Closure {
                def: def.clone(),
                env: Vec::new(),
            }),
            Ty::FnDef(name) => Some(Value::FnPtr(name.clone())),
            Ty::Adt(_) => Some(Value::Enum {
                variant: 0,
                fields: Vec::new(),
            }),
            _ => None,
        }
    }

    pub(crate) fn write_resolved(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: Value,
    ) -> InterpResult<()> {
        match target {
            Resolved::Value(..) => Err(InterpError::type_mismatch(
                "assignment",
                "mutable place",
                "constant aggregate",
            )),
            Resolved::Static { name, projection } => {
                let mut store = self.statics.borrow_mut();
                let slot = store
                    .get_mut(name)
                    .ok_or_else(|| self.missing_static(name))?;
                if projection.is_empty() {
                    *slot = value;
                    return Ok(());
                }
                let place_desc = format!("static {name}");
                assign_projected(slot, projection, value, "<static>", &place_desc)
            }
            Resolved::Alloc { id, projection } => {
                let projection = &self.elem_block_projection(*id, projection);
                let mut store = self.allocs.borrow_mut();
                let slot = store
                    .get_mut(id)
                    .ok_or_else(|| self.missing_alloc(*id))?;
                if projection.is_empty() {
                    // A raw block's base pointer denotes its
                    // first element: writing a non-tuple value
                    // through it writes element 0, not the
                    // block as a whole. A tuple value is the
                    // block's own array value and replaces it.
                    if self.is_elem_block(*id)
                        && matches!(&*slot, Value::Tuple(_))
                        && !matches!(value, Value::Tuple(_))
                    {
                        let place_desc = format!("allocation #{id}");
                        if let Value::Tuple(items) = &mut *slot {
                            if items.is_empty() {
                                items.push(Value::Uninit);
                            }
                        }
                        let slot = navigate_mut(
                            slot,
                            &[RefProj::Index(0)],
                            "<allocation>",
                            &place_desc,
                        )?;
                        *slot = value;
                        return Ok(());
                    }
                    *slot = value;
                    return Ok(());
                }
                // A raw block's element tuple grows to the
                // written index: the allocation's byte size is
                // not a value-level element count, so unwritten
                // elements materialize as uninitialized slots
                // (reading one stays an uninitialized-read
                // error, never a fabricated value).
                if self.is_elem_block(*id) {
                    if let Some(RefProj::Index(i)) = projection.first() {
                        let i = *i;
                        // A block no write has materialized yet
                        // holds no value at all: the first element
                        // write materializes the element tuple,
                        // with the slots it skips over
                        // uninitialized, exactly as the grow
                        // below treats a short tuple.
                        if *slot == Value::Uninit {
                            *slot = Value::Tuple(vec![Value::Uninit; i + 1]);
                        }
                        if let Value::Tuple(items) = &mut *slot {
                            if i >= items.len() {
                                items.resize(i + 1, Value::Uninit);
                            }
                        }
                    }
                }
                let place_desc = format!("allocation #{id}");
                assign_projected(slot, projection, value, "<allocation>", &place_desc)
            }
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                if projection.is_empty() {
                    // Direct local of the target frame: normalize
                    // against that frame's declared type.
                    let func = frames[*frame_pos].func.clone();
                    let program: &'a Program = self.program;
                    let target_body = program.get(&func).ok_or_else(|| {
                        InterpError::UnknownFunction { name: func.clone() }
                    })?;
                    let frame = &mut frames[*frame_pos];
                    let slot = frame.locals.get_mut(*local).ok_or_else(|| {
                        InterpError::UnknownLocal {
                            func: func.clone(),
                            local: *local,
                        }
                    })?;
                    *slot = Self::normalize_for_ty(value, target_body.local_ty(Local(*local)))?;
                    return Ok(());
                }
                let frame = &mut frames[*frame_pos];
                let func = frame.func.clone();
                let place_desc = describe_resolved(*local, projection);
                let slot = frame.locals.get_mut(*local).ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: func.clone(),
                        local: *local,
                    }
                })?;
                assign_projected(slot, projection, value, &func, &place_desc)
            }
        }
    }

    /// Mutate the value a resolved target denotes in place,
    /// running `f` on the stored slot. Callers must preserve the
    /// value's top-level form (an `Enum` stays the same `Enum`, a
    /// `Str` stays a `Str`): write normalization
    /// ([`Self::normalize_for_ty`]) is the identity for those
    /// forms, so skipping the read-modify-write round trip through
    /// [`Self::write_resolved`] — which clones every element of an
    /// aggregate on the way out and back in — is unobservable.
    pub(crate) fn modify_resolved<R>(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        f: impl FnOnce(&mut Value) -> InterpResult<R>,
    ) -> InterpResult<R> {
        match target {
            Resolved::Value(..) => Err(InterpError::type_mismatch(
                "assignment",
                "mutable place",
                "constant aggregate",
            )),
            Resolved::Static { name, projection } => {
                let mut store = self.statics.borrow_mut();
                let slot = store
                    .get_mut(name)
                    .ok_or_else(|| self.missing_static(name))?;
                let place_desc = format!("static {name}");
                let slot = navigate_mut(slot, projection, "<static>", &place_desc)?;
                f(slot)
            }
            Resolved::Alloc { id, projection } => {
                let mut store = self.allocs.borrow_mut();
                let slot = store
                    .get_mut(id)
                    .ok_or_else(|| self.missing_alloc(*id))?;
                let place_desc = format!("allocation #{id}");
                let slot = navigate_mut(slot, projection, "<allocation>", &place_desc)?;
                f(slot)
            }
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let frame = &mut frames[*frame_pos];
                let func = frame.func.clone();
                let place_desc = describe_resolved(*local, projection);
                let slot = frame.locals.get_mut(*local).ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: func.clone(),
                        local: *local,
                    }
                })?;
                let slot = navigate_mut(slot, projection, &func, &place_desc)?;
                f(slot)
            }
        }
    }

    /// Read element `index` of the aggregate a resolved target
    /// denotes (the element access `copy_nonoverlapping` performs).
    pub(crate) fn read_resolved_at(
        &self,
        frames: &[Frame],
        target: &Resolved,
        index: usize,
    ) -> InterpResult<Value> {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.read_resolved(
                    frames,
                    &Resolved::Place {
                        frame_pos: *frame_pos,
                        local: *local,
                        projection,
                    },
                )
            }
            Resolved::Static { name, projection } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.read_resolved(
                    frames,
                    &Resolved::Static {
                        name: name.clone(),
                        projection,
                    },
                )
            }
            Resolved::Alloc { id, projection } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.read_resolved(
                    frames,
                    &Resolved::Alloc {
                        id: *id,
                        projection,
                    },
                )
            }
            Resolved::Value(base, projection) => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                project_value(base.clone(), &projection, "<constant>", "copy_nonoverlapping source")
            }
        }
    }

    /// Write element `index` of the aggregate a resolved target
    /// denotes. Writing through a constant aggregate is the error
    /// `write_resolved` already produces for it.
    pub(crate) fn write_resolved_at(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        index: usize,
        value: Value,
    ) -> InterpResult<()> {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.write_resolved(
                    frames,
                    &Resolved::Place {
                        frame_pos: *frame_pos,
                        local: *local,
                        projection,
                    },
                    value,
                )
            }
            Resolved::Static { name, projection } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.write_resolved(
                    frames,
                    &Resolved::Static {
                        name: name.clone(),
                        projection,
                    },
                    value,
                )
            }
            Resolved::Alloc { id, projection } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.write_resolved(
                    frames,
                    &Resolved::Alloc {
                        id: *id,
                        projection,
                    },
                    value,
                )
            }
            Resolved::Value(..) => self.write_resolved(frames, target, value),
        }
    }

    /// Read element `index` of the sequence a bulk-memory
    /// operation (copy / fill / compare) starting at `target`
    /// reads. Such an operation's pointer denotes the sequence's
    /// first element, which two target shapes express: an
    /// aggregate (an array place, a fat pointer's base), whose
    /// elements the plain indexed access
    /// ([`Self::read_resolved_at`]) walks, or a single element —
    /// a trailing index projection, the shape an array-to-element
    /// pointer cast produces — whose siblings continue the
    /// sequence, so the index adds to the trailing one. Which
    /// shape a target is follows from the value it denotes: an
    /// aggregate value (a nested array's row included) takes the
    /// plain access; anything else is one element of the sequence.
    pub(crate) fn read_seq_at(
        &self,
        frames: &[Frame],
        target: &Resolved,
        index: usize,
    ) -> InterpResult<Value> {
        match self.seq_element_target(frames, target, index)? {
            Some(target) => self.read_resolved(frames, &target),
            None => {
                if let Some(scalar) = self.seq_scalar(frames, target)? {
                    return Ok(Value::Uint(scalar_byte(&scalar, index)? as u128));
                }
                self.read_resolved_at(frames, target, index)
            }
        }
    }

    /// Write element `index` of the sequence a bulk-memory
    /// operation starting at `target` writes (see
    /// [`Self::read_seq_at`]).
    pub(crate) fn write_seq_at(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        index: usize,
        value: Value,
    ) -> InterpResult<()> {
        match self.seq_element_target(frames, target, index)? {
            Some(target) => self.write_resolved(frames, &target, value),
            None => {
                if let Some(scalar) = self.seq_scalar(frames, target)? {
                    let byte = match &value {
                        Value::Uint(v) if *v <= 0xFF => *v as u8,
                        Value::Int(v) if (-128..=255).contains(v) => *v as u8,
                        _ => {
                            return self.write_resolved_at(frames, target, index, value);
                        }
                    };
                    let patched = scalar_with_byte(&scalar, index, byte)?;
                    return self.write_resolved(frames, target, patched);
                }
                self.write_resolved_at(frames, target, index, value)
            }
        }
    }

    /// The integer a sequence target denotes as a whole, when a
    /// bulk-memory operation's pointer names a scalar place: the
    /// address of an integer local cast to a byte pointer (the
    /// shape `ptr::copy_nonoverlapping` into a `u32` in
    /// `core::hash::sip`'s integer loads takes) makes the
    /// operation's sequence the scalar's own bytes, which the
    /// indexed access cannot express. A target ending in an index
    /// step denotes one element of an aggregate instead and keeps
    /// the element treatment, as does any non-integer value.
    fn seq_scalar(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let projection = match target {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. }
            | Resolved::Value(_, projection) => projection,
        };
        if matches!(projection.last(), Some(RefProj::Index(_))) {
            return Ok(None);
        }
        // A byte-displaced target denotes a byte inside the
        // scalar's storage, not the scalar as a whole: the
        // indexed byte view (read / write _resolved_at) owns
        // those reads, and the whole-value read of a displaced
        // position is an unsupported projection, not a scalar.
        if projection
            .iter()
            .any(|s| matches!(s, RefProj::ByteOffset(d) if *d != 0))
        {
            return Ok(None);
        }
        match self.read_resolved(frames, target) {
            Ok(v @ (Value::Int(_) | Value::Uint(_))) => Ok(Some(v)),
            Ok(_) => Ok(None),
            Err(InterpError::UninitializedRead { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Read an integer of `bits` width at `target` by composing
    /// the bytes of the byte sequence the target addresses, when
    /// the target denotes one byte of such a sequence: the shape
    /// a type-punned pointer has (a `*const u16` into a `[u8]`
    /// buffer, the access `unaligned_volatile_load` performs).
    /// The interpreter's values are not byte-addressed, so a
    /// wider-than-byte access through such a pointer can only be
    /// honoured by composing the stored bytes (little-endian, the
    /// only target layout the model uses). `None` when the target
    /// does not denote a byte sequence position with enough bytes
    /// left — the plain element read is then the access's meaning.
    pub(crate) fn read_punned_int(
        &self,
        frames: &[Frame],
        target: &Resolved,
        bits: u32,
        signed: bool,
    ) -> InterpResult<Option<Value>> {
        let n = (bits / 8) as usize;
        if n <= 1 {
            return Ok(None);
        }
        let mut raw: u128 = 0;
        for k in 0..n {
            let byte = match self.read_seq_at(frames, target, k) {
                Ok(Value::Uint(b)) if b <= 0xFF => b,
                Ok(Value::Int(b)) if (0..=0xFF).contains(&b) => b as u128,
                _ => return Ok(None),
            };
            raw |= byte << (8 * k);
        }
        if signed {
            let shift = 128 - bits;
            Ok(Some(Value::Int(((raw << shift) as i128) >> shift)))
        } else {
            let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
            Ok(Some(Value::Uint(raw & mask)))
        }
    }

    /// The store matching [`Self::read_punned_int`]: decompose
    /// `value`'s low `bits` into bytes over the byte sequence at
    /// `target`. Returns `false` (and writes nothing) when the
    /// target does not denote a byte sequence position with
    /// enough bytes left.
    pub(crate) fn write_punned_int(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        bits: u32,
        value: &Value,
    ) -> InterpResult<bool> {
        let n = (bits / 8) as usize;
        if n <= 1 {
            return Ok(false);
        }
        // Probe first: every byte of the access must land on a
        // stored byte, so a non-byte target writes nothing.
        for k in 0..n {
            match self.read_seq_at(frames, target, k) {
                Ok(Value::Uint(b)) if b <= 0xFF => {}
                Ok(Value::Int(b)) if (0..=0xFF).contains(&b) => {}
                _ => return Ok(false),
            }
        }
        let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
        let raw = match value {
            Value::Uint(u) => *u & mask,
            Value::Int(i) => (*i as u128) & mask,
            _ => return Ok(false),
        };
        for k in 0..n {
            let byte = (raw >> (8 * k)) & 0xFF;
            self.write_seq_at(frames, target, k, Value::Uint(byte))?;
        }
        Ok(true)
    }

    /// The directly-resolved target of element `index` of the
    /// sequence starting at `target`, when `target` denotes a
    /// single element of it (see [`Self::read_seq_at`]); `None`
    /// when the plain indexed access applies. An unreadable
    /// element (a buffer being filled element by element) still
    /// denotes its position, so the adjustment applies to it as
    /// well; any other read failure is the failure the plain
    /// access would report, so it propagates unchanged. A value
    /// target carries its projection like the other targets, so
    /// an element pointer into a constant aggregate adjusts the
    /// same way; a target whose projection has no trailing index
    /// step takes the plain access.
    fn seq_element_target(
        &self,
        frames: &[Frame],
        target: &Resolved,
        index: usize,
    ) -> InterpResult<Option<Resolved>> {
        // A byte-denominated allocation target (see
        // `elem_block_projection`) reads its elements at the
        // byte slot, as the plain read/write paths do.
        let byte_translated = match target {
            Resolved::Alloc { id, projection } => Some(Resolved::Alloc {
                id: *id,
                projection: self.elem_block_projection(*id, projection),
            }),
            _ => None,
        };
        let target = byte_translated.as_ref().unwrap_or(target);
        // A raw element block target carrying a single index
        // step denotes a position in the block's flat sequence:
        // the sequence continues at the following slots, never
        // inside the value stored at this slot (which may be a
        // structured element a byte-granular copy moves whole).
        if let Resolved::Alloc { id, projection } = target {
            if self.is_elem_block(*id) {
                if let [RefProj::Index(a)] = projection.as_slice() {
                    return Ok(Some(Resolved::Alloc {
                        id: *id,
                        projection: vec![RefProj::Index(a + index)],
                    }));
                }
            }
        }
        // A raw element block target carrying two index steps —
        // an offset absorbed into an element index, followed by
        // the first-element step of an array-to-element cast —
        // denotes the flat sequence starting at the sum of the
        // two indices, unless the value stored at the first
        // index is itself an aggregate the second index ranges
        // over (a nested array value in the block, including
        // the single-array-field pun `project_value` reads
        // through): that shape keeps the trailing-index
        // reading below.
        if let Resolved::Alloc { id, projection } = target {
            if self.is_elem_block(*id) {
                if let [RefProj::Index(a), RefProj::Index(b)] = projection.as_slice() {
                    let (a, b) = (*a, *b);
                    let head = Resolved::Alloc {
                        id: *id,
                        projection: vec![RefProj::Index(a)],
                    };
                    match self.read_resolved(frames, &head) {
                        Ok(v) => {
                            // The nested reading is only
                            // available while the trailing index
                            // ranges inside the stored value: a
                            // byte-granular sequence running past
                            // the value's own length is the flat
                            // continuation (a raw table's resize
                            // copying a structured bucket value
                            // byte by byte reads past the bucket's
                            // field count immediately).
                            let inner_len = match &v {
                                Value::Tuple(items) => Some(items.len()),
                                Value::Bytes(bytes) => Some(bytes.len()),
                                Value::Str(text) => Some(text.len()),
                                Value::Enum { fields, .. }
                                    if fields.len() == 1 =>
                                {
                                    match &fields[0] {
                                        Value::Tuple(items) => Some(items.len()),
                                        Value::Bytes(bytes) => Some(bytes.len()),
                                        _ => None,
                                    }
                                }
                                _ => None,
                            };
                            let nested = match &v {
                                Value::Tuple(_)
                                | Value::Bytes(_)
                                | Value::Str(_)
                                | Value::FatRef { .. } => {
                                    inner_len.is_none_or(|l| b + index < l)
                                }
                                Value::Enum { fields, .. } => {
                                    fields.len() == 1
                                        && matches!(
                                            fields[0],
                                            Value::Tuple(_) | Value::Bytes(_)
                                        )
                                        && inner_len.is_none_or(|l| b + index < l)
                                }
                                _ => false,
                            };
                            if !nested {
                                return Ok(Some(Resolved::Alloc {
                                    id: *id,
                                    projection: vec![RefProj::Index(a + b + index)],
                                }));
                            }
                        }
                        Err(InterpError::UninitializedRead { .. }) => {
                            return Ok(Some(Resolved::Alloc {
                                id: *id,
                                projection: vec![RefProj::Index(a + b + index)],
                            }));
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        let projection = match target {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. }
            | Resolved::Value(_, projection) => projection,
        };
        let Some((&RefProj::Index(start), head)) = projection.split_last() else {
            return Ok(None);
        };
        match self.read_resolved(frames, target) {
            Ok(v)
                if matches!(
                    v,
                    Value::Tuple(_)
                        | Value::FatRef { .. }
                        | Value::Bytes(_)
                        | Value::Str(_)
                ) =>
            {
                return Ok(None);
            }
            Ok(_) | Err(InterpError::UninitializedRead { .. }) => {}
            Err(e) => return Err(e),
        }
        let mut projection = head.to_vec();
        projection.push(RefProj::Index(start + index));
        Ok(Some(match target {
            Resolved::Place {
                frame_pos, local, ..
            } => Resolved::Place {
                frame_pos: *frame_pos,
                local: *local,
                projection,
            },
            Resolved::Static { name, .. } => Resolved::Static {
                name: name.clone(),
                projection,
            },
            Resolved::Alloc { id, .. } => Resolved::Alloc {
                id: *id,
                projection,
            },
            Resolved::Value(base, _) => Resolved::Value(base.clone(), projection),
        }))
    }

    /// The live place a `&RefCell` / `&mut RefCell` argument points
    /// at. Borrowing needs a place: the flag lives inside the cell's
    /// storage and must be written back, which a constant aggregate
    /// (`RefTarget::Value`) cannot do — that case is an explicit
    /// unsupported error, never a silent no-op.
    pub(crate) fn refcell_place(&self, value: &Value, name: &str) -> InterpResult<RefTarget> {
        match value {
            Value::Ref(
                target @ (RefTarget::Place { .. }
                | RefTarget::Static { .. }
                | RefTarget::Alloc { .. }),
            ) => Ok(target.clone()),
            Value::Ref(_) => Err(InterpError::UnsupportedTerminator {
                kind: format!(
                    "runtime support: {name} on a RefCell with no live place \
                     (constant aggregate) cannot track borrows"
                ),
            }),
            other => Err(InterpError::type_mismatch(
                name,
                "reference to RefCell",
                other.kind_name(),
            )),
        }
    }

    /// Read the borrow flag of the `Cell` aggregate `flag_cell`
    /// points at: 0 = unused, -1 = mutably borrowed, n > 0 = n
    /// shared borrows.
    pub(crate) fn refcell_flag(
        &self,
        frames: &[Frame],
        flag_cell: &RefTarget,
    ) -> InterpResult<i128> {
        let resolved = self.deref_value(frames, &Value::Ref(flag_cell.clone()))?;
        let cell = self.read_resolved(frames, &resolved)?;
        match cell {
            Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
                Value::Enum { fields: inner, .. } if inner.len() == 1 => match inner[0] {
                    Value::Int(flag) => Ok(flag),
                    // A flag materialised from a constant whose
                    // signed context was lost arrives as an
                    // unsigned word; the flag values (0, -1,
                    // small counts) reinterpret exactly.
                    Value::Uint(flag) => Ok(flag as i128),
                    ref other => Err(InterpError::type_mismatch(
                        "RefCell borrow flag",
                        "isize flag",
                        other.kind_name(),
                    )),
                },
                other => Err(InterpError::type_mismatch(
                    "RefCell borrow flag",
                    "UnsafeCell aggregate",
                    other.kind_name(),
                )),
            },
            ref other => Err(InterpError::type_mismatch(
                "RefCell borrow flag",
                "Cell aggregate",
                other.kind_name(),
            )),
        }
    }

    /// Write the borrow flag of the `Cell` aggregate `flag_cell`
    /// points at, rebuilding the `Cell { UnsafeCell { flag } }`
    /// aggregate in place.
    pub(crate) fn refcell_set_flag(
        &mut self,
        frames: &mut Vec<Frame>,
        flag_cell: &RefTarget,
        flag: i128,
    ) -> InterpResult<()> {
        let resolved = self.deref_value(frames, &Value::Ref(flag_cell.clone()))?;
        let cell = Value::Enum {
            variant: 0,
            fields: vec![Value::Enum {
                variant: 0,
                fields: vec![Value::Int(flag)],
            }],
        };
        self.write_resolved(frames, &resolved, cell)
    }

    /// Release the borrow held by a `Ref` / `RefMut` guard value:
    /// field 1 is the reference to the cell's borrow-flag `Cell`.
    /// A shared release decrements the count; a mutable release
    /// resets the flag to unused (0).
    pub(crate) fn release_refcell_borrow(
        &mut self,
        frames: &mut Vec<Frame>,
        guard: &Value,
        mutable: bool,
    ) -> InterpResult<()> {
        let flag_ref = match guard {
            Value::Enum { fields, .. } if fields.len() == 2 => &fields[1],
            other => {
                return Err(InterpError::type_mismatch(
                    "RefCell guard drop",
                    "Ref/RefMut guard aggregate",
                    other.kind_name(),
                ))
            }
        };
        let flag_target = match flag_ref {
            Value::Ref(target) => target.clone(),
            other => {
                return Err(InterpError::type_mismatch(
                    "RefCell guard drop",
                    "reference to the borrow-flag Cell",
                    other.kind_name(),
                ))
            }
        };
        let flag = self.refcell_flag(frames, &flag_target)?;
        let next = if mutable { 0 } else { flag - 1 };
        self.refcell_set_flag(frames, &flag_target, next)
    }

    pub(crate) fn read_place(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> InterpResult<Value> {
        let target = self.resolve_place(frames, body, place)?;
        // Read through the borrowing form: a projected read lends
        // the denoted value out of the frame's storage, so only
        // the projected part is cloned into the owned result,
        // where the owned read cloned the whole base aggregate
        // (every element of a vector, for a single indexed read).
        // Results and errors are identical (see
        // `read_resolved_ref`).
        Ok(self.read_resolved_ref(frames, &target)?.into_owned())
    }

    pub(crate) fn write_place(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        place: &Place,
        value: Value,
    ) -> InterpResult<()> {
        let target = self.resolve_place_mode(frames, body, place, false)?;
        self.write_resolved(frames, &target, value)
    }

    /// Validate / wrap a value being stored into a typed direct local.
    pub(crate) fn normalize_for_ty(mut value: Value, ty: Option<&Ty>) -> InterpResult<Value> {
        // Storing the uninitialized marker itself is not a type
        // mismatch: MIR materialises uninitialised locals with an
        // assignment of the `Uninit` constant (the storage a
        // `MaybeUninit` or a union is filled through a pointer
        // afterwards), and moving a not-yet-initialised local into
        // another local propagates the same marker. The place simply
        // holds no value yet; any later read of it still raises
        // `UninitializedRead`, so nothing is fabricated here.
        if value == Value::Uninit {
            return Ok(value);
        }
        // A `MaybeUninit` place denotes the payload's own storage:
        // the union adds no representation of its own, and std's
        // byte-wise machinery reads and writes the payload through
        // the wrapper type while the storage is shared with the
        // payload-typed place (`ptr::swap_chunk` copies a chunk of
        // one element's storage into a `MaybeUninit<[u8; N]>`
        // local and back out into the other element). A read of
        // that storage yields the payload in storage form, so the
        // store accepts the value as it stands; wrapping it into
        // the field-0 form here would put a wrapper value where
        // the payload's own readers look.
        if matches!(
            ty,
            Some(Ty::Adt(name))
                if name.starts_with("std::mem::MaybeUninit")
                    || name.starts_with("core::mem::MaybeUninit")
        ) {
            return Ok(value);
        }
        // `core::ptr::Unaligned<T>` is a one-field `repr(C,
        // packed)` wrapper whose only field IS the value it
        // wraps: `ptr::read_unaligned` moves the pointee through
        // locals of this type, and a pointee read in this model
        // yields the field value itself (the allocation holds
        // the inner value; the wrapper has no address of its
        // own). Storing the bare value into an `Unaligned` place
        // therefore wraps it in the one-field aggregate, the
        // exact inverse of the wrapper-unwrap for scalar targets
        // below.
        if let Some(Ty::Adt(name)) = ty {
            if (name == "std::ptr::Unaligned" || name == "core::ptr::Unaligned")
                && !matches!(value, Value::Enum { .. })
            {
                return Ok(Value::Enum {
                    variant: 0,
                    fields: vec![value],
                });
            }
        }
        // A scalar place whose value arrives as a one-field wrapper
        // aggregate: reading through a raw pointer into a transparent
        // wrapper (`UnsafeCell<T>`, the `AtomicUsize`-family structs,
        // any one-field struct) yields the wrapper's aggregate value,
        // because the pointer names the wrapper place itself and the
        // model has no separate address for the field. The wrapper's
        // only field IS the scalar at that address (the layout fact
        // the pointer arithmetic in the source relies on), so the
        // store unwraps it, recursively for nested wrappers.
        //
        // The unwrap is limited to unsigned-integer, boolean and
        // float targets whose innermost leaf is a value of the same
        // kind: those are the shapes the atomic/pointer-metadata
        // assignments actually produce. A signed-integer target does
        // not unwrap: a signed field's width lives only in its
        // declared type, so reading a wrapper aggregate as a signed
        // scalar of some other width is exactly the reinterpretation
        // a cast to an unrelated pointee type asks for, and the
        // model — which has no byte memory — keeps it an explicit
        // type mismatch instead of fabricating a value (the negative
        // case `wrapper_cast_to_an_unrelated_pointee_is_not_
        // reinterpreted` pins this).
        if matches!(
            ty,
            Some(Ty::Bool) | Some(Ty::Uint(_)) | Some(Ty::Float(_))
        ) {
            let mut inner = &value;
            while let Value::Enum {
                variant: 0,
                fields,
            } = inner
            {
                if fields.len() != 1 {
                    break;
                }
                inner = &fields[0];
            }
            let same_kind = matches!(
                (ty, inner),
                (Some(Ty::Uint(_)), Value::Uint(_))
                    | (Some(Ty::Bool), Value::Bool(_))
                    | (Some(Ty::Float(_)), Value::Float(_))
            );
            if same_kind && !matches!(inner, Value::Enum { .. }) {
                value = inner.clone();
            }
        }
        // An array place whose value arrives as a one-field
        // wrapper aggregate: reading a one-field struct over an
        // array (a `#[repr(simd)]` vector, above all) through a
        // pointer cast to the array type -- `Simd::into_array`
        // dereferences a `*const [T; N]` at the vector's own
        // address -- yields the wrapper's aggregate value,
        // because the pointer names the wrapper place itself.
        // The wrapper's only field IS the array at that address,
        // so the store unwraps it, recursively for nested
        // wrappers: the aggregate counterpart of the scalar
        // unwrap above. The field must be exactly the declared
        // array (a tuple of the declared length, or the byte
        // form of a `u8` array); any other field shape keeps
        // the type mismatch.
        if let Some(Ty::Array(elem, n)) = ty {
            let mut inner = &value;
            while let Value::Enum {
                variant: 0,
                fields,
            } = inner
            {
                if fields.len() != 1 {
                    break;
                }
                let field = &fields[0];
                let is_the_array = match field {
                    Value::Tuple(items) => items.len() as u64 == *n,
                    Value::Bytes(b) => {
                        **elem == Ty::Uint(UintTy::U8) && b.len() as u64 == *n
                    }
                    _ => false,
                };
                if is_the_array {
                    value = field.clone();
                    break;
                }
                inner = field;
            }
        }
        match (ty, &value) {
            (Some(Ty::Bool), Value::Bool(_))
            | (Some(Ty::Unit), Value::Unit)
            | (Some(Ty::Tuple(_)), Value::Tuple(_))
            | (None, _) => Ok(value),
            (Some(Ty::Int(it)), Value::Int(v)) => Ok(Value::Int(wrap_signed(*v, it.bits()))),
            (Some(Ty::Uint(ut)), Value::Uint(v)) => Ok(Value::Uint(*v & ut.max())),
            // An `f32` place holds the `f32` rounding of the stored
            // (exactly widened) value; an `f64` place holds it as-is.
            (Some(Ty::Float(FloatTy::F16)), Value::Float(v)) => {
                Ok(Value::Float(round_to_f16(*v)))
            }
            (Some(Ty::Float(FloatTy::F32)), Value::Float(v)) => {
                Ok(Value::Float((*v as f32) as f64))
            }
            (Some(Ty::Float(FloatTy::F64)), Value::Float(_)) => Ok(value),
            (Some(Ty::Float(FloatTy::F128)), Value::Float128(_)) => Ok(value),
            // An integer payload whose kind tag differs from the
            // declared type: the bits are reinterpreted at the
            // declared width. This arises for scalar leaves of
            // structured constants, whose signedness is not stored
            // in the constant itself; MIR places and their values
            // always agree on width, so the declared type is the
            // exact interpretation.
            (Some(Ty::Int(it)), Value::Uint(v)) => {
                Ok(Value::Int(wrap_signed(*v as i128, it.bits())))
            }
            (Some(Ty::Uint(ut)), Value::Int(v)) => Ok(Value::Uint((*v as u128) & ut.max())),
            // Callable locals: a function-pointer place holds a
            // function item value or a (non-capturing) closure value;
            // a function-item place holds the item value; a closure
            // place holds the closure value.
            (Some(Ty::FnPtr), Value::FnPtr(_))
            | (Some(Ty::FnPtr), Value::Closure { .. })
            | (Some(Ty::FnDef(_)), Value::FnPtr(_))
            | (Some(Ty::Closure(..)), Value::Closure { .. }) => Ok(value),
            // ADT locals hold the ADT value forms: aggregates, string
            // data (`String`), and the fmt value family.
            (Some(Ty::Adt(_)), Value::Enum { .. })
            // A pointer-shaped ADT local (`Unique`, `NonNull`, a
            // `Box` field slot) can hold a built box's allocation
            // pointer directly: MIR moves the pointer between
            // these wrapper locals verbatim (a box's field 0 read
            // into a `Unique` local is the first step of every
            // `Box::into_raw`), and the pointer is the value each
            // of them denotes in this model. The target may be a
            // detached value as well as an allocation — the data
            // pointer read out of a string the model holds as
            // bare data names the string value itself.
            | (Some(Ty::Adt(_)), Value::Ref(_))
            | (Some(Ty::Adt(_)), Value::Str(_))
            // A smart-pointer ADT over a `dyn` pointee (`Box<dyn
            // Trait>`, `Rc<dyn Trait>`) holds the fat pointer.
            | (Some(Ty::Adt(_)), Value::DynRef { .. })
            | (Some(Ty::Adt(_)), Value::FmtArg { .. })
            | (Some(Ty::Adt(_)), Value::FmtCount(_))
            | (Some(Ty::Adt(_)), Value::FmtArguments(_))
            // The captured-`Formatter` and builder handles stand
            // for `core::fmt` ADT values (`Formatter` itself when
            // a body moves it, and the `DebugList` family the
            // builder constructors return by value).
            | (Some(Ty::Adt(_)), Value::Formatter(_))
            | (Some(Ty::Adt(_)), Value::FmtBuilder(_)) => Ok(value),
            // Reference locals hold resolved references, fat pointers,
            // and — for `&str` and byte-array references — the string
            // or byte data itself (a string constant *is* its value;
            // a reference to a string place is a `Ref`).
            (Some(Ty::Ref(_)), Value::Ref(_))
            | (Some(Ty::Ref(_)), Value::FatRef { .. })
            | (Some(Ty::Ref(_)), Value::DynRef { .. })
            | (Some(Ty::Ref(_)), Value::Str(_))
            | (Some(Ty::Ref(_)), Value::Bytes(_))
            // A pointer to a function stored under a cast (a
            // function item referenced through a data-pointer type,
            // e.g. `&(foo as *const u8)` as a constant): the value
            // is the callable, which is a pointer like any other
            // for storage, address and equality purposes; reading
            // the pointee through it stays an explicit error at the
            // read site, as the model has no code bytes.
            | (Some(Ty::Ref(_)), Value::FnPtr(_))
            | (Some(Ty::Ref(_)), Value::Closure { .. }) => Ok(value),
            // A niche-encoded enum over a pointer payload stored
            // into a pointer place: the enum's representation IS
            // the payload pointer -- the ABI fact the
            // transmuted-fn-pointer return relies on (a callee
            // whose declared return type is `Option<&T>` produces
            // the payload-variant aggregate, and a caller whose
            // signature was transmuted to return `&T` stores the
            // result into a reference place). The one field is
            // the pointer itself, the same reading a one-field
            // wrapper's store unwraps to its scalar above. A
            // fieldless (niche) variant value carries no pointer
            // to read, so it stays a mismatch rather than
            // fabricating one.
            // A pointer-shaped ADT value (`NonNull`, `Unique`)
            // written to a raw-pointer or reference place: MIR
            // moves these transparent wrappers into pointer
            // places verbatim (the `Allocator::deallocate` body
            // passes its `NonNull<u8>` straight to
            // `__rust_dealloc`), and the pointer the wrapper
            // holds is the value the place denotes.
            (Some(Ty::Ref(_)), Value::Enum { fields, .. })
                if fields.len() == 1
                    && matches!(
                        fields[0],
                        Value::Ref(_)
                            | Value::FatRef { .. }
                            | Value::DynRef { .. }
                    ) =>
            {
                Ok(fields[0].clone())
            }
            (Some(Ty::Str), Value::Str(_)) => Ok(value),
            (Some(Ty::Array(_, n)), Value::Tuple(items)) if items.len() as u64 == *n => Ok(value),
            (Some(Ty::Array(elem, n)), Value::Bytes(b))
                if **elem == Ty::Uint(UintTy::U8) && b.len() as u64 == *n =>
            {
                Ok(value)
            }
            // A byte read through a `*const u8` / `*const i8`
            // whose pointee is a byte-string value: the pointer
            // denotes the storage's first byte (any displacement
            // has already been projected away by the read).
            (Some(Ty::Uint(UintTy::U8)), Value::Bytes(b)) if !b.is_empty() => {
                Ok(Value::Uint(u128::from(b[0])))
            }
            (Some(Ty::Int(IntTy::I8)), Value::Bytes(b)) if !b.is_empty() => {
                Ok(Value::Int(i128::from(b[0] as i8)))
            }
            (Some(Ty::Slice(_)), Value::FatRef { .. }) => Ok(value),
            // Coroutine locals hold the state object itself. (A
            // coroutine value also reaches places typed as the ADT
            // wrappers around it - `Box`, `Pin` - as a field of the
            // wrapper's aggregate value, which is checked as the
            // wrapper, not here.)
            (Some(Ty::Coroutine(_)), Value::Coroutine { .. }) => Ok(value),
            // The async-drop transform types a coroutine body's
            // resume place as the ADT `std::future::ResumeTy`,
            // the wrapper the coroutine environment travels in;
            // the value stored there is the coroutine object
            // itself, exactly as in a coroutine-typed place.
            (Some(Ty::Adt(name)), Value::Coroutine { .. })
                if name == "std::future::ResumeTy" =>
            {
                Ok(value)
            }
            (Some(expected), other) => Err(InterpError::type_mismatch(
                "assignment",
                format!("{expected:?}"),
                other.kind_name(),
            )),
        }
    }

    /// Apply an unsizing decided at conversion time
    /// ([`UnsizeMeta`]) to a value: the shared machinery of the
    /// `Rvalue::Unsize` coercion cast and of a `From` / `Into`
    /// conversion into a `Box<dyn Trait>`, whose result goes
    /// through the same unsizing (see
    /// `TerminatorKind::CallBuiltin::result_unsize`).
    fn eval_unsize(&self, frames: &[Frame], v: Value, meta: &UnsizeMeta) -> InterpResult<Value> {
        match meta {
            UnsizeMeta::Slice => self.unsize_to_slice(frames, v),
            UnsizeMeta::Dyn {
                concrete,
                traits,
                glue,
            } => match v {
                // A supertrait upcast: the data pointer, the
                // concrete type and its drop glue carry over;
                // only the bounds the pointer answers to
                // change.
                Value::DynRef {
                    base,
                    concrete: existing,
                    drop_glue,
                    ..
                } => Ok(Value::DynRef {
                    base,
                    concrete: existing,
                    traits: traits.clone(),
                    drop_glue,
                }),
                Value::Ref(_) | Value::FatRef { .. } | Value::Enum { .. } => {
                    let concrete = concrete.clone().ok_or_else(|| {
                        InterpError::UnsupportedRvalue {
                            kind: "Unsize to dyn (no concrete type recorded)".to_string(),
                        }
                    })?;
                    Ok(Value::DynRef {
                        base: Box::new(v),
                        concrete,
                        traits: traits.clone(),
                        drop_glue: glue.clone().map(Box::new),
                    })
                }
                other => Err(InterpError::type_mismatch(
                    "unsizing cast to dyn",
                    "reference, fat pointer or box value",
                    other.kind_name(),
                )),
            },
        }
    }

    /// Array-to-slice unsizing of a pointer-shaped value: a
    /// reference to an array becomes a fat reference over it, and
    /// a pointer-carrying ADT value (a `Box`, the `Unique` /
    /// `NonNull` aggregate chain a MIR-built box is made of, an
    /// `Rc` / `Arc` / `Weak` handle, or a user wrapper around
    /// one) is rewrapped around the unsized pointee, so
    /// `Box<[T; N]>` becomes the same shape a directly built
    /// `Box<[T]>` has.
    fn unsize_to_slice(&self, frames: &[Frame], v: Value) -> InterpResult<Value> {
        match v {
            Value::Ref(_) => {
                let target = self.deref_value(frames, &v)?;
                let pointee = self.read_resolved(frames, &target)?;
                let len = unsized_tail_len(&pointee).ok_or_else(|| {
                    InterpError::type_mismatch(
                        "unsizing cast",
                        "array pointee",
                        pointee.kind_name(),
                    )
                })?;
                Ok(Value::FatRef {
                    base: Box::new(v),
                    len,
                })
            }
            // Unsizing a fat pointer into a slice pointer again
            // (a slice-typed field reached through the ADT search
            // below) leaves it as it stands.
            Value::FatRef { .. } => Ok(v),
            Value::Bytes(b) => {
                let len = b.len() as u128;
                Ok(Value::FatRef {
                    base: Box::new(Value::Bytes(b)),
                    len,
                })
            }
            Value::Tuple(items) => {
                let len = items.len() as u128;
                Ok(Value::FatRef {
                    base: Box::new(Value::Tuple(items)),
                    len,
                })
            }
            // An empty array aggregate is the unit value in this
            // model: unsizing it yields an empty slice.
            Value::Unit => Ok(Value::FatRef {
                base: Box::new(Value::Tuple(Vec::new())),
                len: 0,
            }),
            // A reference-counted handle is itself the pointer:
            // its pointee lives in the allocation store, whose
            // stored aggregate carries the length every consumer
            // derives (the `Deref` builtin, the drop glue and the
            // count adjustments all navigate the store through
            // the handle's allocation id), so unsizing the handle
            // into a slice handle changes nothing about the
            // value. Wrapping the handle's inner pointer into a
            // fat reference instead would leave the handle naming
            // a view no consumer recognises as a handle.
            Value::Enum { .. } if rc_handle_id(&v).is_some() => Ok(v),
            Value::Enum { variant, mut fields } if fields.len() == 1 => {
                let pointee = fields.pop().expect("one field");
                // A built box's field is its allocation pointer:
                // the array already lives in the allocation store
                // under that pointer, and unsizing is only a view
                // of it — every consumer derives the slice length
                // from the pointee value in the store. The box
                // keeps its shape (rewrapping the pointer into a
                // fat reference here would leave the box holding
                // a snapshot view instead of the pointer, and a
                // pointer extracted from it later would name this
                // place rather than the allocation).
                if matches!(pointee, Value::Ref(RefTarget::Alloc { .. })) {
                    return Ok(Value::Enum {
                        variant,
                        fields: vec![pointee],
                    });
                }
                let fat = self.unsize_to_slice(frames, pointee)?;
                Ok(Value::Enum {
                    variant,
                    fields: vec![fat],
                })
            }
            Value::Enum { variant, fields } => {
                // A multi-field ADT: the coercion propagates
                // through the one pointer-shaped field (rustc's
                // `CoerceUnsized` changes exactly one field's
                // type); the remaining fields — phantom markers,
                // counters, plain data — pass through untouched.
                // Candidates are judged from the values alone: a
                // candidate whose unsizing rests on the pointee's
                // own array tail outranks one resting on the
                // one-element view of a scalar, and among equals
                // a changed value outranks an unchanged one, with
                // the later field winning (the position a DST
                // tail occupies). When no candidate changes its
                // value — every one is an allocation-backed
                // handle, whose unsizing is the identity in this
                // model — the first candidate stands and the ADT
                // keeps its shape.
                let mut best: Option<(usize, Value)> = None;
                let mut best_strong = false;
                for (i, field) in fields.iter().enumerate() {
                    let Some(Ok(unsized_field)) =
                        self.unsize_pointer_shape(frames, field.clone())
                    else {
                        continue;
                    };
                    let strong = match field {
                        Value::Ref(_) => {
                            let target = self.deref_value(frames, field)?;
                            let pointee = self.read_resolved(frames, &target)?;
                            unsized_tail_is_array(&pointee)
                        }
                        _ => true,
                    };
                    let changed = unsized_field != *field;
                    let replace = match &best {
                        None => true,
                        Some(_) if strong && !best_strong => true,
                        Some(_) if strong == best_strong && changed => true,
                        _ => false,
                    };
                    if replace {
                        best = Some((i, unsized_field));
                        best_strong = strong;
                    }
                }
                let Some((idx, replacement)) = best else {
                    return Err(InterpError::type_mismatch(
                        "unsizing cast",
                        "reference to an array",
                        "ADT value",
                    ));
                };
                let mut fields = fields;
                fields[idx] = replacement;
                Ok(Value::Enum {
                    variant,
                    fields,
                })
            }
            other => Err(InterpError::type_mismatch(
                "unsizing cast",
                "reference to an array",
                other.kind_name(),
            )),
        }
    }

    /// Unsizing applied to a value in pointer position inside an
    /// ADT (the multi-field arm of [`Self::unsize_to_slice`]):
    /// references and fat pointers directly, wrapper chains and
    /// nested ADTs through the general unsizing, and
    /// reference-counted handles as the identity. Returns `None`
    /// when the value is not a pointer shape an unsizing can
    /// apply to — plain data, callables, trait-object pointers
    /// (their unsizing is the `Dyn` metadata's business) — so
    /// the caller can look at the ADT's other fields. A pointer
    /// shape whose own unsizing fails is likewise not a
    /// candidate: the field the coercion targets is the one
    /// whose unsizing succeeds.
    fn unsize_pointer_shape(
        &self,
        frames: &[Frame],
        v: Value,
    ) -> Option<InterpResult<Value>> {
        match v {
            Value::Ref(_) | Value::FatRef { .. } => Some(self.unsize_to_slice(frames, v)),
            Value::Enum { .. } => {
                if rc_handle_id(&v).is_some() {
                    return Some(Ok(v));
                }
                match self.unsize_to_slice(frames, v) {
                    Ok(u) => Some(Ok(u)),
                    Err(_) => None,
                }
            }
            _ => None,
        }
    }

    /// Leave the value at a `Field` / `Downcast`-only projected
    /// place over a local of the executing frame uninitialized:
    /// the clearing a `Move` out of that place performs. The
    /// resolved target of such a place is the executing frame's
    /// local under the same projection (the classification's
    /// steps are purely syntactic), so the write goes straight
    /// through `navigate_mut` — the same write
    /// `write_resolved`'s projected `Place` arm performs, minus
    /// the resolve round trip.
    pub(crate) fn clear_local_proj(
        &self,
        frames: &mut Vec<Frame>,
        local: usize,
        proj: &[RefProj],
        desc: &str,
    ) -> InterpResult<()> {
        let frame = frames.last_mut().expect("frame");
        let func = frame.func.clone();
        let slot = frame.locals.get_mut(local).ok_or_else(|| InterpError::UnknownLocal {
            func: func.clone(),
            local,
        })?;
        let slot = navigate_mut(slot, proj, &func, desc)?;
        *slot = Value::Uninit;
        Ok(())
    }

    pub(crate) fn eval_operand(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        operand: &Operand,
    ) -> InterpResult<Value> {
        match operand {
            Operand::Copy(place) => {
                // Fast path, mirroring `read_local` in the decode
                // layer: a projection-less local holding a value is
                // cloned straight out of the frame's slot, skipping
                // place resolution and the resolved read. Anything
                // else — a projected place, an out-of-range index,
                // an uninitialized slot (whose zero-sized fallback
                // and exact error the general read produces) —
                // delegates to the general read unchanged.
                if place.is_direct_local() {
                    let frame = frames.last().expect("frame");
                    if let Some(value) = frame.locals.get(place.local.0) {
                        if *value != Value::Uninit {
                            return Ok(value.clone());
                        }
                    }
                }
                // A `Field` / `Downcast`-only place over a local
                // takes the decode layer's projected fast read,
                // which mirrors the general read step for step;
                // any other shape keeps the general read.
                if let Some((proj, desc)) = crate::decode::classify_local_proj(place) {
                    return self.read_local_proj(
                        frames,
                        body,
                        place.local.0 as u32,
                        &proj,
                        &desc,
                        place,
                    );
                }
                self.read_place(frames, body, place)
            }
            Operand::Move(place) => {
                // The `Copy` fast path, plus the clearing of the
                // vacated slot the general path performs below for
                // a direct local.
                if place.is_direct_local() {
                    let idx = place.local.0;
                    let frame = frames.last().expect("frame");
                    if let Some(value) = frame.locals.get(idx) {
                        if *value != Value::Uninit {
                            let value = value.clone();
                            frames.last_mut().expect("frame").locals[idx] = Value::Uninit;
                            return Ok(value);
                        }
                    }
                }
                let fast = crate::decode::classify_local_proj(place);
                let value = match &fast {
                    Some((proj, desc)) => self.read_local_proj(
                        frames,
                        body,
                        place.local.0 as u32,
                        proj,
                        desc,
                        place,
                    )?,
                    None => self.read_place(frames, body, place)?,
                };
                if place.is_direct_local() {
                    let frame = frames.last_mut().expect("frame");
                    if let Some(slot) = frame.locals.get_mut(place.local.0) {
                        *slot = Value::Uninit;
                    }
                } else if let Some((proj, desc)) = &fast {
                    self.clear_local_proj(frames, place.local.0, proj, desc)?;
                } else if let Ok(target) = self.resolve_place(frames, body, place) {
                    // A move out of a projected place leaves that
                    // part uninitialized too, so a later `Drop` of
                    // the whole value does not run the moved-out
                    // part's glue a second time (rustc's drop flags
                    // track exactly this). A heap allocation's
                    // storage clears the same way: a boxed
                    // coroutine's prologue moves its parameters out
                    // of the environment (the allocation's upvar
                    // storage), and leaving the stale copies in
                    // place would drop them a second time when the
                    // future's own glue walks the environment.
                    if matches!(
                        target,
                        Resolved::Place { .. } | Resolved::Static { .. } | Resolved::Alloc { .. }
                    ) {
                        self.write_resolved(frames, &target, Value::Uninit)?;
                    }
                }
                Ok(value)
            }
            Operand::Constant(c) => self.eval_constant(c),
        }
    }

    /// Evaluate a constant to a value. Structured constants recurse:
    /// a reference constant becomes a reference to the constant value
    /// it points at, a slice constant becomes a fat reference over
    /// its element tuple.
    pub(crate) fn eval_constant(&self, c: &Constant) -> InterpResult<Value> {
        Ok(match c {
            Constant::Unit => Value::Unit,
            Constant::Bool(b) => Value::Bool(*b),
            Constant::Int(v) => Value::Int(*v),
            Constant::Uint(v) => Value::Uint(*v),
            Constant::Float(v) => Value::Float(*v),
            Constant::Float128(bits) => Value::Float128(*bits),
            Constant::VTablePtr(id) => Value::Ref(RefTarget::Address(*id)),
            Constant::FatPtr { addr, meta } => Value::FatRef {
                base: Box::new(Value::Ref(RefTarget::Address(*addr))),
                len: u128::from(*meta),
            },
            Constant::FnDef(name) => Value::FnPtr(name.clone()),
            Constant::Closure(def) => Value::Closure {
                def: def.clone(),
                env: Vec::new(),
            },
            Constant::Str(s) => Value::Str(s.clone()),
            Constant::Bytes(b) => Value::Bytes(b.clone()),
            Constant::Tuple(fields) => Value::Tuple(
                fields
                    .iter()
                    .map(|f| self.eval_constant(f))
                    .collect::<InterpResult<Vec<_>>>()?,
            ),
            Constant::Adt { variant, fields } => Value::Enum {
                variant: *variant,
                fields: fields
                    .iter()
                    .map(|f| self.eval_constant(f))
                    .collect::<InterpResult<Vec<_>>>()?,
            },
            Constant::Ref(inner) => {
                Value::Ref(RefTarget::value(self.eval_constant(inner)?))
            }
            Constant::IntPtr(addr) => Value::Ref(RefTarget::Address(*addr)),
            Constant::NullPtr => Value::Ref(RefTarget::Null),
            Constant::Uninit => Value::Uninit,
            Constant::StaticRef {
                name,
                value,
                projection,
                len,
            } => {
                self.ensure_static(name, value)?;
                let base = Value::Ref(RefTarget::Static {
                    name: name.clone(),
                    projection: projection.clone(),
                });
                // A fat static reference is the static reference
                // plus its element count, in the same shape the
                // runtime unsizing cast builds: dereferencing it
                // reaches the static's storage, so writes through
                // a mutable slice static land in the one storage
                // every later read observes.
                match len {
                    Some(len) => Value::FatRef {
                        base: Box::new(base),
                        len: *len,
                    },
                    None => base,
                }
            }
            Constant::SliceVal(elems) => {
                let elems = elems
                    .iter()
                    .map(|e| self.eval_constant(e))
                    .collect::<InterpResult<Vec<_>>>()?;
                let len = elems.len() as u128;
                Value::FatRef {
                    base: Box::new(Value::Tuple(elems)),
                    len,
                }
            }
        })
    }

    /// Project `value` through `projection`, resolving union
    /// field steps against the program's union tables: the plain
    /// projector ([`project_value`]) is type-free, so runs of
    /// ordinary steps go through it and each union step is
    /// applied here, where the scalar shapes are in reach.
    fn project_value_t(
        &self,
        value: Value,
        projection: &[RefProj],
        func: &str,
        place: &str,
    ) -> InterpResult<Value> {
        let mut value = value;
        let mut i = 0;
        while i < projection.len() {
            if let RefProj::UnionField { idx, table } = projection[i] {
                value = self.union_field_read(value, idx, table, func, place)?;
                i += 1;
            } else {
                let start = i;
                while i < projection.len()
                    && !matches!(projection[i], RefProj::UnionField { .. })
                {
                    i += 1;
                }
                value = project_value(value, &projection[start..i], func, place)?;
            }
        }
        Ok(value)
    }

    /// Read field `idx` of a union `value`. The value holds one
    /// field's value — the last one written; every other slot is
    /// uninitialized (see the union arm of aggregate evaluation
    /// and of `navigate_mut`). Reading the held field returns it
    /// as it stands. Reading any other field reinterprets the
    /// held field's bits as this field's type, little-endian: a
    /// narrower field reads the low bytes, and a wider field
    /// reads zeros above the held field's width (the bytes a
    /// native read would find there are unspecified, so the
    /// model fixes them at zero rather than fabricating data).
    /// A union holding no live field is an uninitialized read;
    /// a field whose type has no scalar shape, and a value
    /// holding several live fields (which the write side never
    /// produces), are explicit errors.
    fn union_field_read(
        &self,
        value: Value,
        idx: usize,
        table: usize,
        func: &str,
        place: &str,
    ) -> InterpResult<Value> {
        // A wholly uninitialized base holds no live field: that
        // is the limiting case of the uninitialized read below,
        // not a type mismatch (a partially initialized place can
        // reach here with its union slot never materialized).
        if value == Value::Uninit {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        }
        let Value::Enum { ref fields, .. } = value else {
            return Err(InterpError::type_mismatch(
                format!("union field read on {place}"),
                "union (ADT) value",
                value.kind_name(),
            ));
        };
        if let Some(held) = fields.get(idx) {
            if *held != Value::Uninit {
                return Ok(held.clone());
            }
        }
        let mut live = fields
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != Value::Uninit);
        let Some((src_idx, stored)) = live.next() else {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        };
        if live.next().is_some() {
            return Err(InterpError::type_mismatch(
                format!("union field read on {place}"),
                "a union value holding one field",
                "a union value holding several fields",
            ));
        }
        let stored = stored.clone();
        let shapes = self.program.union_tables.get(table).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("union field read on {place}"),
                "a union instance with a scalar-shape table",
                "a union instance without one",
            )
        })?;
        let src = shapes.get(src_idx).and_then(|o| o.as_ref());
        let dst = shapes.get(idx).and_then(|o| o.as_ref());
        let (Some(src), Some(dst)) = (src, dst) else {
            return Err(InterpError::type_mismatch(
                format!("union field read on {place}"),
                "scalar-shaped union fields on both sides of the read",
                "a union field whose type has no scalar shape",
            ));
        };
        let bits = self.flatten_scalar(&stored, src)?;
        self.construct_scalar(bits, dst)
    }

    /// A scalar `Transmute` reinterpretation: flatten the
    /// operand value to its bit pattern through the source
    /// shape, then construct the target shape's value from the
    /// same bits.
    /// The lane decomposition of a vector shape: the shape of
    /// one lane and the lane count, unwrapping one-payload-field
    /// ADTs to the array they store. A shape that is not (an ADT
    /// over) an array decomposes to itself as a single lane.
    fn shape_lanes(shape: &ScalarShape) -> (ScalarShape, u64) {
        match shape {
            ScalarShape::AdtField { inner, .. } => Self::shape_lanes(inner),
            ScalarShape::Array(elem, n) => ((**elem).clone(), *n),
            other => (other.clone(), 1),
        }
    }

    /// Flatten a stored element to a lane's bit pattern: the
    /// plain scalar flattening, seeing through one-field wrapper
    /// aggregates (a raw block's elements are the lane types'
    /// own values — a control byte is the wrapper struct around
    /// the byte, not the bare integer).
    fn flatten_lane(&self, value: &Value, lane: &ScalarShape) -> InterpResult<u128> {
        if let Ok(bits) = self.flatten_scalar(value, lane) {
            return Ok(bits);
        }
        if let Value::Enum { fields, .. } = value {
            if fields.len() == 1 {
                return self.flatten_lane(&fields[0], lane);
            }
        }
        self.flatten_scalar(value, lane)
    }

    /// The bit pattern of the vector value a raw element block
    /// holds at `target`: lane `i` is the element at sequence
    /// offset `i`, flattened through the lane shape.
    fn shaped_bits_from_elements(
        &self,
        frames: &[Frame],
        target: &Resolved,
        shape: &ScalarShape,
    ) -> InterpResult<u128> {
        let (lane, count) = Self::shape_lanes(shape);
        let lane_bits = lane.bits();
        // A target whose stored units are bytes (a byte string,
        // a raw block of control bytes) holds the vector as a
        // byte stream: the lanes are packed from consecutive
        // bytes, not read one byte per lane. The first unit
        // decides: a byte-sized scalar (or a one-field wrapper
        // over one, like a control-byte tag) means byte units.
        let first = self.read_seq_at(frames, target, 0)?;
        let byte_like = |v: &Value| -> bool {
            match v {
                Value::Uint(b) => *b <= 0xFF,
                Value::Int(b) => (-128..=255).contains(b),
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    matches!(&fields[0], Value::Uint(b) if *b <= 0xFF)
                        || matches!(&fields[0], Value::Int(b) if (-128..=255).contains(b))
                }
                _ => false,
            }
        };
        if lane_bits > 8 && byte_like(&first) {
            let byte_lane = ScalarShape::Int {
                bits: 8,
                signed: false,
            };
            let nbytes = (shape.bits() / 8) as usize;
            let mut out = 0u128;
            for k in 0..nbytes {
                let unit = if k == 0 {
                    first.clone()
                } else {
                    self.read_seq_at(frames, target, k)?
                };
                let b = self.flatten_lane(&unit, &byte_lane)?;
                out |= (b & 0xFF) << (8 * (k as u32));
            }
            return Ok(out);
        }
        let mut out = 0u128;
        for i in 0..count {
            let element = if i == 0 {
                first.clone()
            } else {
                self.read_seq_at(frames, target, i as usize)?
            };
            let bits = self.flatten_lane(&element, &lane)?;
            out |= bits << ((i as u32) * lane_bits);
        }
        Ok(out)
    }

    /// A shaped whole-pointee load (see [`Rvalue::ShapedLoad`]):
    /// a pointee stored whole loads as itself, canonicalised
    /// through its shape; a pointee in a raw element block whose
    /// stored elements are not the value (a byte buffer read as
    /// a vector) is assembled from the consecutive elements.
    fn shaped_load(
        &self,
        frames: &[Frame],
        target: &Resolved,
        shape: &ScalarShape,
    ) -> InterpResult<Value> {
        let elem_block = matches!(target, Resolved::Alloc { id, .. } if self.is_elem_block(*id));
        match self.read_resolved(frames, target) {
            Ok(v) => {
                if let Ok(bits) = self.flatten_scalar(&v, shape) {
                    return self.construct_scalar(bits, shape);
                }
                // A sequence-like value base (a byte string or an
                // element tuple behind a constant pointer) holds
                // the vector as its consecutive elements, like a
                // raw block does.
                let seq_base = matches!(
                    target,
                    Resolved::Value(base, _)
                        if matches!(base, Value::Bytes(_) | Value::Tuple(_))
                );
                if elem_block || seq_base {
                    let bits = self.shaped_bits_from_elements(frames, target, shape)?;
                    return self.construct_scalar(bits, shape);
                }
                Ok(v)
            }
            Err(e) => {
                if elem_block && matches!(e, InterpError::UninitializedRead { .. }) {
                    let bits = self.shaped_bits_from_elements(frames, target, shape)?;
                    return self.construct_scalar(bits, shape);
                }
                // A byte string behind field steps: the steps
                // below the dereference select the payload field
                // of a wrapper (a vector type's storage field is
                // field 0 at offset 0), so a chain of field-0
                // steps over a byte-string base denotes the
                // shape's bits at the start of the string — the
                // plain projection cannot enter a byte string,
                // but the shaped load names exactly those bits.
                if let Resolved::Value(base, projection) = target {
                    if let Value::Bytes(bytes) = base {
                        if projection
                            .iter()
                            .all(|p| matches!(p, RefProj::Field(0)))
                            && !projection.is_empty()
                        {
                            let nbytes = (shape.bits() / 8) as usize;
                            if bytes.len() >= nbytes {
                                let mut bits = 0u128;
                                for (k, b) in bytes[..nbytes].iter().enumerate() {
                                    bits |= u128::from(*b) << (8 * (k as u32));
                                }
                                return self.construct_scalar(bits, shape);
                            }
                        }
                    }
                }
                Err(e)
            }
        }
    }

    /// A shaped whole-pointee store (see
    /// [`StatementKind::ShapedStore`]): into a raw element block
    /// the value goes lane by lane over the consecutive elements
    /// at the target, each new lane keeping the wrapper shape of
    /// the element it replaces; anywhere else the store is the
    /// plain whole-value one.
    fn shaped_store(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: Value,
        shape: &ScalarShape,
    ) -> InterpResult<()> {
        let elem_block = matches!(target, Resolved::Alloc { id, .. } if self.is_elem_block(*id));
        if elem_block {
            if let Ok(bits) = self.flatten_scalar(&value, shape) {
                let (lane, count) = Self::shape_lanes(shape);
                let lane_bits = lane.bits();
                let mask = bit_mask(lane_bits);
                for i in 0..count {
                    let lane_value = self.construct_scalar(
                        (bits >> ((i as u32) * lane_bits)) & mask,
                        &lane,
                    )?;
                    let stored = match self.read_seq_at(frames, target, i as usize) {
                        Ok(current) => crate::builtins::rewrap_like(&current, lane_value),
                        Err(_) => lane_value,
                    };
                    self.write_seq_at(frames, target, i as usize, stored)?;
                }
                return Ok(());
            }
        }
        self.write_resolved(frames, target, value)
    }

    fn reinterpret_value(
        &self,
        value: &Value,
        src: &ScalarShape,
        dst: &ScalarShape,
    ) -> InterpResult<Value> {
        let bits = self.flatten_scalar(value, src)?;
        self.construct_scalar(bits, dst)
    }

    /// The bit pattern of `value` read through `shape`
    /// (little-endian: earlier array elements and lower fields
    /// occupy the lower bits).
    fn flatten_scalar(
        &self,
        value: &Value,
        shape: &ScalarShape,
    ) -> InterpResult<u128> {
        match (shape, value) {
            (ScalarShape::Int { bits, .. }, Value::Int(i)) => {
                Ok((*i as u128) & bit_mask(*bits))
            }
            (ScalarShape::Int { bits, .. }, Value::Uint(u)) => {
                Ok(*u & bit_mask(*bits))
            }
            (ScalarShape::Float { bits }, Value::Float(f)) => Ok(match bits {
                16 => u128::from(f64_to_f16_bits(*f)),
                32 => u128::from((*f as f32).to_bits()),
                _ => u128::from(f.to_bits()),
            }),
            (ScalarShape::Bool, Value::Bool(b)) => Ok(u128::from(*b)),
            (ScalarShape::Char, Value::Uint(u)) => Ok(*u & bit_mask(32)),
            (ScalarShape::Array(elem, n), Value::Tuple(items))
                if items.len() as u64 == *n =>
            {
                let eb = elem.bits();
                let mut out = 0u128;
                for (i, item) in items.iter().enumerate() {
                    out |= self.flatten_scalar(item, elem)? << (i as u32 * eb);
                }
                Ok(out)
            }
            // A byte-array constant's value is its bytes.
            (ScalarShape::Array(elem, n), Value::Bytes(b))
                if elem.bits() == 8 && b.len() as u64 == *n =>
            {
                let mut out = 0u128;
                for (i, byte) in b.iter().enumerate() {
                    out |= u128::from(*byte) << (8 * i);
                }
                Ok(out)
            }
            // A pointer payload stored bare (the model keeps a
            // wrapper's payload pointer as the value itself in
            // some slots — the same convention `project_value`
            // encodes for field 0 of a pointer value): flatten
            // the pointer through the payload shape directly.
            (
                ScalarShape::AdtField { index, inner, .. },
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. },
            ) if *index == 0 => self.flatten_scalar(value, inner),
            (
                ScalarShape::AdtField { index, inner, .. },
                Value::Enum { fields, .. },
            ) => match fields.get(*index as usize) {
                Some(field) => self.flatten_scalar(field, inner),
                None => Err(InterpError::type_mismatch(
                    "transmute",
                    "an ADT value holding the payload field",
                    "an ADT value without it",
                )),
            },
            // The storage form of the same payload: a write
            // through a pointer cast to the payload field's type
            // (the `Simd::load` copy through a `*mut [T; N]`
            // into the union's slot) stores the field's own
            // value where the aggregate would sit. Its bits are
            // the payload's bits, so it flattens as the payload.
            (ScalarShape::AdtField { inner, .. }, value) => {
                self.flatten_scalar(value, inner)
            },
            (
                ScalarShape::NicheEnum {
                    inner,
                    untagged_variant,
                    niche_variants,
                    niche_start,
                },
                Value::Enum { variant, fields },
            ) => {
                if (niche_variants.0..=niche_variants.1).contains(variant) {
                    Ok(*niche_start + u128::from(*variant - niche_variants.0))
                } else if variant == untagged_variant {
                    match fields.first() {
                        Some(field) => self.flatten_scalar(field, inner),
                        None => Err(InterpError::type_mismatch(
                            "transmute",
                            "an enum value holding the payload field",
                            "an enum value without it",
                        )),
                    }
                } else {
                    Err(InterpError::type_mismatch(
                        "transmute",
                        "a variant of the niche-encoded enum",
                        format!("variant {variant}"),
                    ))
                }
            }
            (ScalarShape::Ptr, _) => match synthetic_address(value) {
                Some(addr) => Ok(addr & bit_mask(64)),
                None => Err(InterpError::type_mismatch(
                    "transmute",
                    "a pointer value",
                    value.kind_name(),
                )),
            },
            (
                ScalarShape::FieldlessEnum {
                    bits,
                    discriminants,
                },
                Value::Enum { variant, .. },
            ) => match discriminants.get(*variant as usize) {
                Some(d) => Ok((*d as u128) & bit_mask(*bits)),
                None => Err(InterpError::type_mismatch(
                    "transmute",
                    "a variant of the fieldless enum",
                    format!("variant {variant}"),
                )),
            },
            (_, other) => Err(InterpError::type_mismatch(
                "transmute",
                "a value of the source shape",
                other.kind_name(),
            )),
        }
    }

    /// The value of `shape` whose bit pattern is `bits`
    /// (the low `shape.bits()` bits; higher bits are ignored).
    fn construct_scalar(
        &self,
        bits: u128,
        shape: &ScalarShape,
    ) -> InterpResult<Value> {
        match shape {
            ScalarShape::Int { bits: w, signed } => {
                if *signed {
                    Ok(Value::Int(wrap_signed(bits as i128, *w)))
                } else {
                    Ok(Value::Uint(wrap_unsigned(bits, *w)))
                }
            }
            ScalarShape::Float { bits: w } => Ok(Value::Float(match w {
                16 => crate::mir_convert::f16_to_f64(bits as u16),
                32 => f64::from(f32::from_bits(bits as u32)),
                _ => f64::from_bits(bits as u64),
            })),
            ScalarShape::Ptr => {
                Ok(Value::Ref(RefTarget::Address(bits as u64)))
            }
            ScalarShape::Bool => Ok(Value::Bool(bits != 0)),
            ScalarShape::Char => match char::from_u32(bits as u32) {
                Some(c) => Ok(Value::Uint(u128::from(c as u32))),
                None => Err(InterpError::type_mismatch(
                    "transmute to char",
                    "a valid Unicode code point",
                    format!("bit pattern {bits:#x}"),
                )),
            },
            ScalarShape::Array(elem, n) => {
                let eb = elem.bits();
                let mut items = Vec::with_capacity(*n as usize);
                for i in 0..*n {
                    let part = (bits >> (i as u32 * eb)) & bit_mask(eb);
                    items.push(self.construct_scalar(part, elem)?);
                }
                Ok(Value::Tuple(items))
            }
            ScalarShape::AdtField {
                index,
                count,
                fill,
                inner,
            } => {
                let mut fields = vec![Value::Uninit; *count];
                for (i, c) in fill {
                    fields[*i as usize] = self.eval_constant(c)?;
                }
                fields[*index as usize] = self.construct_scalar(bits, inner)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields,
                })
            }
            ScalarShape::NicheEnum {
                inner,
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                let span = u128::from(niche_variants.1 - niche_variants.0);
                if bits >= *niche_start && bits - *niche_start <= span {
                    Ok(Value::Enum {
                        variant: niche_variants.0 + (bits - *niche_start) as u32,
                        fields: Vec::new(),
                    })
                } else {
                    let payload = self.construct_scalar(bits, inner)?;
                    Ok(Value::Enum {
                        variant: *untagged_variant,
                        fields: vec![payload],
                    })
                }
            }
            ScalarShape::FieldlessEnum {
                bits: width,
                discriminants,
            } => {
                let pattern = bits & bit_mask(*width);
                match discriminants
                    .iter()
                    .position(|d| (*d as u128) & bit_mask(*width) == pattern)
                {
                    Some(variant) => Ok(Value::Enum {
                        variant: variant as u32,
                        fields: Vec::new(),
                    }),
                    None => Err(InterpError::type_mismatch(
                        "transmute to a fieldless enum",
                        "a discriminant value of the target enum",
                        format!("bit pattern {pattern:#x}"),
                    )),
                }
            }
        }
    }

    /// The metadata of a pointer value (`PtrMetadata` rvalue):
    /// a slice or string pointer's length, a dyn pointer's vtable
    /// (the fat value itself in this model, only ever passed along
    /// or compared opaquely), and the unit value for a sized
    /// pointee. A reference to a constant slice value (a static
    /// slice, materialised as a reference to the fat value) and a
    /// reference naming a place whose contents are a fat value (a
    /// slice reference formed as `&*box`, stored as a place
    /// reference) both carry that fat value's metadata.
    fn ptr_metadata(
        &self,
        frames: &[Frame],
        v: &Value,
        dest_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match v {
            Value::FatRef { len, .. } => Ok(Value::Uint(*len)),
            Value::Str(s) => Ok(Value::Uint(s.len() as u128)),
            Value::DynRef { .. } => Ok(v.clone()),
            Value::Ref(target) => {
                // A byte-displaced thin pointer's metadata is the
                // unit value: the pointer itself is thin whatever
                // it points into, and the displaced position is not
                // readable in this model, so the pointee read below
                // does not apply to it.
                let projection = match target {
                    RefTarget::Place { projection, .. }
                    | RefTarget::Static { projection, .. }
                    | RefTarget::Alloc { projection, .. } => Some(projection),
                    _ => None,
                };
                if let Some(projection) = projection {
                    if let Some(RefProj::ByteOffset(d)) = projection.last() {
                        if *d != 0 {
                            return Ok(Value::Unit);
                        }
                    }
                }
                let target = self.deref_value(frames, v)?;
                let pointee = self.read_resolved(frames, &target)?;
                self.metadata_of_pointee(frames, &pointee, dest_ty)
            }
            // A built box (a one-field aggregate whose field is
            // the allocation pointer) carries its pointee's
            // metadata, as the box pointer does.
            Value::Enum { fields, .. } if fields.len() == 1 => {
                let target = self.deref_value(frames, v)?;
                let pointee = self.read_resolved(frames, &target)?;
                self.metadata_of_pointee(frames, &pointee, dest_ty)
            }
            other => Err(InterpError::type_mismatch(
                "PtrMetadata",
                "fat pointer",
                other.kind_name(),
            )),
        }
    }

    /// The metadata denoted by a pointee value (see
    /// [`Self::ptr_metadata`]). A value with a dynamically sized
    /// tail — a custom DST struct value, or the array value a
    /// slice pointer names — carries the tail's metadata: the
    /// last field's metadata, recursively, or the element count.
    /// The recursion is gated on the destination being
    /// length-typed: MIR types a sized pointee's metadata as
    /// `()`, so a `usize` destination is exactly the case where
    /// the metadata is a length; every other shape's metadata is
    /// the unit value.
    fn metadata_of_pointee(
        &self,
        frames: &[Frame],
        pointee: &Value,
        dest_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match pointee {
            Value::FatRef { len, .. } => Ok(Value::Uint(*len)),
            Value::Str(s) => Ok(Value::Uint(s.len() as u128)),
            Value::DynRef { .. } => Ok(pointee.clone()),
            Value::Ref(_) => self.ptr_metadata(frames, pointee, dest_ty),
            Value::Tuple(items) => {
                if matches!(dest_ty, Some(Ty::Uint(_))) {
                    Ok(Value::Uint(items.len() as u128))
                } else {
                    Ok(Value::Unit)
                }
            }
            Value::Enum { fields, .. } => {
                if matches!(dest_ty, Some(Ty::Uint(_))) {
                    match fields.last() {
                        Some(last) => {
                            let last = last.clone();
                            self.metadata_of_pointee(frames, &last, dest_ty)
                        }
                        None => Ok(Value::Unit),
                    }
                } else {
                    Ok(Value::Unit)
                }
            }
            _ => Ok(Value::Unit),
        }
    }

    // ----- rvalues -----------------------------------------------------------

    /// The wrapper ADT value holding the address-only pointer
    /// for `addr` in its payload field, with the unique values
    /// in its zero-sized fields: the result of the
    /// integer/enum-to-pointer-wrapper transmutes.
    fn ptr_wrapped_value(
        &self,
        addr: u64,
        payload_index: u32,
        field_count: usize,
        fill: &[(u32, Constant)],
    ) -> InterpResult<Value> {
        let mut fields = vec![Value::Uninit; field_count];
        fields[payload_index as usize] = Value::Ref(RefTarget::Address(addr));
        for (index, constant) in fill {
            fields[*index as usize] = self.eval_constant(constant)?;
        }
        Ok(Value::Enum {
            variant: 0,
            fields,
        })
    }

    pub(crate) fn eval_rvalue(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        rvalue: &Rvalue,
        dest_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match rvalue {
            Rvalue::Use(op) => self.eval_operand(frames, body, op),
            Rvalue::DataPtr(op) => {
                let value = self.eval_operand(frames, body, op)?;
                match value {
                    Value::FatRef { base, .. } => Ok((*base).clone()),
                    other => Ok(other),
                }
            }
            Rvalue::RawPtrAggregate { data, meta } => {
                let base = self.eval_operand(frames, body, data)?;
                match meta {
                    None => Ok(base),
                    Some(meta_op) => {
                        let meta_value = self.eval_operand(frames, body, meta_op)?;
                        if meta_value == Value::Unit {
                            // Unit metadata is no metadata: the
                            // aggregate is the data pointer itself.
                            return Ok(base);
                        }
                        let len = match meta_value {
                            Value::Uint(v) => v,
                            Value::Int(v) => v as u128,
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "raw pointer metadata",
                                    "integer length",
                                    other.kind_name(),
                                ))
                            }
                        };
                        Ok(Value::FatRef {
                            base: Box::new(base),
                            len,
                        })
                    }
                }
            }
            Rvalue::BinaryOp(op, pair) => {
                let lhs = self.eval_operand(frames, body, &pair.0)?;
                let rhs = self.eval_operand(frames, body, &pair.1)?;
                eval_binary(*op, &lhs, &rhs, dest_ty, false).map(|(v, _)| v)
            }
            Rvalue::CheckedBinaryOp(op, pair) => {
                let lhs = self.eval_operand(frames, body, &pair.0)?;
                let rhs = self.eval_operand(frames, body, &pair.1)?;
                let (v, overflowed) = eval_binary(*op, &lhs, &rhs, dest_ty, true)?;
                Ok(Value::Tuple(vec![v, Value::Bool(overflowed)]))
            }
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => {
                let v = self.eval_operand(frames, body, operand)?;
                self.ptr_metadata(frames, &v, dest_ty)
            }
            Rvalue::UnaryOp(op, operand) => {
                let v = self.eval_operand(frames, body, operand)?;
                eval_unary(*op, &v, dest_ty)
            }
            Rvalue::AggregateTuple(operands) => {
                let mut items = Vec::with_capacity(operands.len());
                for op in operands {
                    items.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Tuple(items))            }
            Rvalue::AggregateAdt {
                variant,
                operands,
                field_index,
                units_from,
            } => {
                // A union aggregate's single operand initializes the
                // active field; the slots before it stay
                // uninitialized, exactly as a field write through a
                // projection leaves the slots it skipped over (see
                // `navigate_mut`'s coroutine arm). Reading the active
                // field then finds the stored value, and reading any
                // other field is an uninitialized read, as it should
                // be: the model does not reinterpret a union's bytes.
                let mut fields = match field_index {
                    Some(active) => vec![Value::Uninit; *active as usize],
                    None => Vec::with_capacity(operands.len()),
                };
                for op in operands {
                    fields.push(self.eval_operand(frames, body, op)?);
                }
                if let Some((from_size, payload_pos)) = units_from {
                    if let Some(field) = fields.get_mut(*payload_pos) {
                        let taken = std::mem::replace(field, Value::Unit);
                        *field = ptr_cast_units_value(taken, *from_size);
                    }
                }
                // A `MaybeUninit::new(v)` construction (the union's
                // `value` field, index 1): in this model a
                // `MaybeUninit<T>` value holds its payload at field
                // 0 -- the slot every pointer access to the union
                // denotes (`as_ptr` / `as_mut_ptr` enter field 0, and
                // the thread-local seed expansion builds the same
                // shape) -- not at the declared field index with
                // uninitialized padding before it. The construction
                // therefore unwraps the `ManuallyDrop<T>` aggregate
                // the operand arrives as and stores the payload at
                // field 0. Without this, a value written through a
                // `*mut MaybeUninit<T>` (e.g. `ptr::replace` in std's
                // thread-local lazy initialization) would sit in the
                // place where no reader of the payload can reach it.
                if *field_index == Some(1) && operands.len() == 1 {
                    let is_maybe_uninit = matches!(
                        dest_ty,
                        Some(Ty::Adt(name))
                            if name.starts_with("std::mem::MaybeUninit")
                                || name.starts_with("core::mem::MaybeUninit")
                    );
                    if is_maybe_uninit {
                        let operand = fields.pop().expect("one operand pushed");
                        let payload = match operand {
                            Value::Enum { fields: inner, .. } if inner.len() == 1 => {
                                inner.into_iter().next().expect("one field")
                            }
                            other => other,
                        };
                        return Ok(Value::Enum {
                            variant: 0,
                            fields: vec![payload],
                        });
                    }
                }
                Ok(Value::Enum {
                    variant: *variant,
                    fields,
                })
            }
            Rvalue::ClosureAggregate { def, operands } => {
                let mut env = Vec::with_capacity(operands.len());
                for op in operands {
                    env.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Closure {
                    def: def.clone(),
                    env,
                })
            }
            Rvalue::CoroutineAggregate(operands) => {
                let mut fields = Vec::with_capacity(operands.len());
                for op in operands {
                    fields.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Coroutine {
                    variant: 0,
                    fields,
                    saved: Vec::new(),
                })
            }
            Rvalue::Cast(operand, ty) => {
                let v = self.eval_operand(frames, body, operand)?;
                eval_cast(&v, ty)
            }
            Rvalue::Unsize(operand, meta) => {
                let v = self.eval_operand(frames, body, operand)?;
                self.eval_unsize(frames, v, meta)
            }
            Rvalue::PtrToInt(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                match synthetic_address(&value) {
                    Some(addr) => Ok(Value::Uint(addr)),
                    None => Err(InterpError::TypeMismatch {
                        context: "pointer-to-integer transmute".to_string(),
                        expected: "a pointer value".to_string(),
                        found: value.kind_name().to_string(),
                    }),
                }
            }
            Rvalue::PtrField(operand, depth) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.ptr_field_value(frames, body, value, *depth)
            }
            Rvalue::PtrFirstElem(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.ptr_first_elem_value(frames, body, value)
            }
            Rvalue::PtrCastUnits(operand, from_size) => {
                let value = self.eval_operand(frames, body, operand)?;
                Ok(ptr_cast_units_value(value, *from_size))
            }
            Rvalue::PtrArrayBase(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.ptr_array_base_value(value)
            }
            Rvalue::WrapArrayElems(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                Self::wrap_array_elems(value)
            }
            Rvalue::UnwrapArrayElems(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                Self::unwrap_array_elems(value)
            }
            Rvalue::IntToPtr(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                match value {
                    Value::Int(i) => Ok(Value::Ref(RefTarget::Address(i as u64))),
                    Value::Uint(u) => Ok(Value::Ref(RefTarget::Address(u as u64))),
                    other => Err(InterpError::type_mismatch(
                        "integer-to-pointer transmute",
                        "an integer value",
                        other.kind_name(),
                    )),
                }
            }
            Rvalue::IntToPtrWrapped {
                operand,
                payload_index,
                field_count,
                fill,
            } => {
                let value = self.eval_operand(frames, body, operand)?;
                let addr = match value {
                    Value::Int(i) => i as u64,
                    Value::Uint(u) => u as u64,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "integer-to-pointer-wrapper transmute",
                            "an integer value",
                            other.kind_name(),
                        ))
                    }
                };
                self.ptr_wrapped_value(addr, *payload_index, *field_count, fill)
            }
            Rvalue::ScalarToPtrWrapped {
                operand,
                from,
                payload_index,
                field_count,
                fill,
            } => {
                let value = self.eval_operand(frames, body, operand)?;
                let addr = self.flatten_scalar(&value, from)? as u64;
                self.ptr_wrapped_value(addr, *payload_index, *field_count, fill)
            }
            Rvalue::EnumToPtrWrapped {
                operand,
                discriminants,
                payload_index,
                field_count,
                fill,
            } => {
                let value = self.eval_operand(frames, body, operand)?;
                let addr = match value {
                    Value::Enum { variant, .. } => {
                        let discr = if discriminants.is_empty() {
                            variant as i128
                        } else {
                            *discriminants.get(variant as usize).ok_or_else(
                                || {
                                    InterpError::type_mismatch(
                                        "enum-to-pointer-wrapper transmute",
                                        "variant covered by the discriminant table",
                                        format!("variant {variant}"),
                                    )
                                },
                            )?
                        };
                        discr as u64
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "enum-to-pointer-wrapper transmute",
                            "an enum value",
                            other.kind_name(),
                        ))
                    }
                };
                self.ptr_wrapped_value(addr, *payload_index, *field_count, fill)
            }
            Rvalue::EnumToInt(operand, discriminants, ty) => {
                let value = self.eval_operand(frames, body, operand)?;
                match value {
                    Value::Enum { variant, .. } => {
                        let discr = if discriminants.is_empty() {
                            variant as i128
                        } else {
                            *discriminants.get(variant as usize).ok_or_else(|| {
                                InterpError::type_mismatch(
                                    "enum-to-integer transmute",
                                    "variant covered by the discriminant table",
                                    format!("variant {variant}"),
                                )
                            })?
                        };
                        eval_cast(&Value::Int(discr), ty)
                    }
                    other => Err(InterpError::type_mismatch(
                        "enum-to-integer transmute",
                        "an enum value",
                        other.kind_name(),
                    )),
                }
            }
            Rvalue::IntToEnum(operand, discriminants) => {
                let value = self.eval_operand(frames, body, operand)?;
                let bits = match value {
                    Value::Int(i) => i as u128,
                    Value::Uint(u) => u,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "integer-to-enum transmute",
                            "an integer value",
                            other.kind_name(),
                        ))
                    }
                };
                match discriminants.iter().position(|d| *d as u128 == bits) {
                    Some(variant) => Ok(Value::Enum {
                        variant: variant as u32,
                        fields: Vec::new(),
                    }),
                    None => Err(InterpError::type_mismatch(
                        "integer-to-enum transmute",
                        "a discriminant value of the target enum",
                        format!("integer {bits}"),
                    )),
                }
            }
            Rvalue::IntToEnumWrapped {
                operand,
                discriminants,
                payload_index,
                field_count,
                fill,
            } => {
                let value = self.eval_operand(frames, body, operand)?;
                let bits = match value {
                    Value::Int(i) => i as u128,
                    Value::Uint(u) => u,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "integer-to-enum-wrapper transmute",
                            "an integer value",
                            other.kind_name(),
                        ))
                    }
                };
                let variant = discriminants
                    .iter()
                    .position(|d| *d as u128 == bits)
                    .ok_or_else(|| {
                        InterpError::type_mismatch(
                            "integer-to-enum-wrapper transmute",
                            "a discriminant value of the wrapped enum",
                            format!("integer {bits}"),
                        )
                    })?;
                let mut fields = vec![Value::Uninit; *field_count];
                fields[*payload_index as usize] = Value::Enum {
                    variant: variant as u32,
                    fields: Vec::new(),
                };
                for (index, constant) in fill {
                    fields[*index as usize] = self.eval_constant(constant)?;
                }
                Ok(Value::Enum {
                    variant: 0,
                    fields,
                })
            }
            Rvalue::Reinterpret(operand, src, dst) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.reinterpret_value(&value, src, dst)
            }
            Rvalue::ShapedLoad(place, shape) => {
                let target = self.resolve_place(frames, body, place)?;
                self.shaped_load(frames, &target, shape)
            }
            Rvalue::PtrOffset(ptr_op, count_op, baked_size, count_width) => {
                let base = self.eval_operand(frames, body, ptr_op)?;
                let count_value = self.eval_operand(frames, body, count_op)?;
                let count = match count_value {
                    Value::Int(v) => v,
                    Value::Uint(v) => {
                        let v = i128::try_from(v).map_err(|_| {
                            InterpError::UnsupportedRvalue {
                                kind: "pointer offset: count does not fit in an isize"
                                    .to_string(),
                            }
                        })?;
                        if *count_width > 0 && *count_width < 128 {
                            // The count arrived in an unsigned-typed
                            // operand (the `byte_*` family passes its
                            // `usize` count straight through): its
                            // bits are the `isize` count.
                            let w = *count_width;
                            let masked = v & ((1i128 << w) - 1);
                            if (masked >> (w - 1)) & 1 == 1 {
                                masked - (1i128 << w)
                            } else {
                                masked
                            }
                        } else {
                            v
                        }
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "pointer offset count",
                            "integer count",
                            other.kind_name(),
                        ))
                    }
                };
                let elem_size = baked_size.or_else(|| self.offset_elem_size_of(body, ptr_op));
                self.ptr_offset_value(frames, base, count, elem_size, false)
            }
            Rvalue::BoxDerefTransmute(operand) => {
                // The cast yields a pointer to the box's pointee.
                // Two box shapes exist in this value model. A box
                // built from its MIR aggregates is the wrapper chain
                // `Unique` around `NonNull` around the pointer, and
                // unwrapping the chain reaches the pointer value
                // itself. A box made by the `Box::new` builtin instead
                // holds its pointee directly as its single field, so
                // the chain bottoms out at the pointee value (or at a
                // reference that *is* the pointee, when the pointee
                // is itself a reference); the result is then a
                // reference to the pointee's place, which is what
                // reading or writing through the extracted pointer
                // must reach. `box_cast_places` tells the two apart:
                // it reports how many wrapper layers an
                // aggregate-built value carries above its pointer.
                let places = self.box_cast_places(frames, body, operand);
                let original = self.eval_operand(frames, body, operand)?;
                let mut v = original.clone();
                let mut layers = 0usize;
                loop {
                    match v {
                        Value::DynRef { .. } => {
                            break Ok(v)
                        }
                        Value::Ref(_) | Value::FatRef { .. } => {
                            match &places {
                                Some((_, pointee, depth)) if layers < *depth => {
                                    break Ok(Self::pointee_ref(
                                        frames, pointee,
                                    ));
                                }
                                _ => break Ok(v),
                            }
                        }
                        Value::Enum { mut fields, .. } if fields.len() == 1 => {
                            v = fields.pop().expect("one field");
                            layers += 1;
                        }
                        other => {
                            // Builtin-built boxes whose operand the
                            // classification covers: it names the
                            // pointee place. A move out of the operand
                            // place cleared the storage the pointer
                            // must name, so the value is restored
                            // first. (Lane BH)
                            if let Some((source, pointee, _)) = &places {
                                if matches!(operand, Operand::Move(_)) {
                                    self.write_resolved(
                                        frames,
                                        source,
                                        original.clone(),
                                    )?;
                                }
                                break Ok(Self::pointee_ref(frames, pointee));
                            }
                            // A box built by the `Box::new` /
                            // `Box::new_uninit` builtins stores its
                            // pointee as the field value itself, not
                            // as a pointer chain. When the cast's
                            // operand place projects into the box, the
                            // pointer it stands for is a reference to
                            // that place; a resolved constant value is
                            // referenced as a value. (Lane BO)
                            if let Operand::Copy(place) | Operand::Move(place) = operand {
                                match self.resolve_place(frames, body, place)? {
                                    Resolved::Place {
                                        frame_pos,
                                        local,
                                        projection,
                                    } if !projection.is_empty() => {
                                        break Ok(Value::Ref(RefTarget::place(
                                            frames[frame_pos].id,
                                            local,
                                            projection,
                                        )));
                                    }
                                    Resolved::Value(base, projection) => {
                                        break Ok(Value::Ref(RefTarget::Value(
                                            Box::new(base),
                                            projection,
                                        )));
                                    }
                                    Resolved::Static { name, projection }
                                        if !projection.is_empty() =>
                                    {
                                        break Ok(Value::Ref(RefTarget::Static {
                                            name,
                                            projection,
                                        }));
                                    }
                                    Resolved::Alloc { id, projection }
                                        if !projection.is_empty() =>
                                    {
                                        break Ok(Value::Ref(RefTarget::Alloc {
                                            id,
                                            projection,
                                        }));
                                    }
                                    Resolved::Place { .. }
                                    | Resolved::Static { .. }
                                    | Resolved::Alloc { .. } => {}
                                }
                            }
                            break Err(InterpError::type_mismatch(
                                "BoxDerefTransmute cast",
                                "pointer wrapper (Unique/NonNull around a raw pointer)",
                                other.kind_name(),
                            ));
                        }
                    }
                }
            }
            Rvalue::Ref(place) => {
                // A borrow of exactly the place a dyn fat pointer
                // points at (`&*r`, `&*box`) is the fat pointer
                // itself: the result is again a `&dyn`, with the
                // same data pointer and the same vtable identity.
                if place.projection.len() == 1
                    && matches!(place.projection[0], ProjectionElem::Deref)
                {
                    let head = Place {
                        local: place.local,
                        projection: Vec::new(),
                    };
                    if let Ok(value @ Value::DynRef { .. }) =
                        self.read_place(frames, body, &head)
                    {
                        return Ok(value);
                    }
                }
                let target = self.resolve_place(frames, body, place)?;
                // A pointer to an unsized pointee (a slice or
                // `str`) is a fat pointer: it must carry the
                // pointee's length as its metadata, or `PtrMetadata`
                // and the slice consumers downstream see a thin
                // pointer whose metadata was dropped when the place
                // was resolved. The length is the element (or byte)
                // count of the value the place currently holds.
                // Where that value cannot be read (an uninitialised
                // or address-only pointee), the thin pointer is
                // formed as before and the metadata consumers report
                // the gap at their own site.
                let fat_len = match dest_ty {
                    Some(Ty::Ref(inner))
                        if matches!(&**inner, Ty::Slice(_) | Ty::Str) =>
                    {
                        self.fat_deref_len(frames, body, place).or_else(|| {
                            self.read_resolved(frames, &target)
                                .ok()
                                .and_then(|v| v.len())
                        })
                    }
                    _ => None,
                };
                let thin = match target {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => Value::Ref(RefTarget::place(
                        frames[frame_pos].id,
                        local,
                        projection,
                    )),
                    Resolved::Value(base, projection) => {
                        Value::Ref(RefTarget::Value(Box::new(base), projection))
                    }
                    Resolved::Static { name, projection } => {
                        Value::Ref(RefTarget::Static { name, projection })
                    }
                    Resolved::Alloc { id, projection } => {
                        Value::Ref(RefTarget::Alloc { id, projection })
                    }
                };
                match fat_len {
                    Some(len) => Ok(Value::FatRef {
                        base: Box::new(thin),
                        len,
                    }),
                    None => Ok(thin),
                }
            }
            Rvalue::Len(place) => {
                let value = self.read_place(frames, body, place)?;
                let len = self.fat_deref_len(frames, body, place);
                let len = match len {
                    Some(len) => Some(len),
                    None => value.len(),
                };
                let len = len.ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("Len of {place}"),
                        "array, slice or string value",
                        value.kind_name(),
                    )
                })?;
                Ok(Value::Uint(len))
            }
            Rvalue::Discriminant(place, discriminants) => {
                let value = self.read_place(frames, body, place)?;
                match value {
                    Value::Enum { variant, .. } | Value::Coroutine { variant, .. } => {
                        if discriminants.is_empty() {
                            // No table was recorded for this enum:
                            // its discriminants are the sequential
                            // defaults, so the index is the value.
                            Ok(Value::Int(variant as i128))
                        } else {
                            let discr = discriminants
                                .get(variant as usize)
                                .ok_or_else(|| {
                                    InterpError::type_mismatch(
                                        format!("Discriminant of {place}"),
                                        "variant covered by the discriminant table",
                                        format!("variant {variant}"),
                                    )
                                })?;
                            Ok(Value::Int(*discr))
                        }
                    }
                    // A non-ADT value (a scalar, a tuple, a
                    // reference, ...) has no variants: its
                    // discriminant is 0. This is the shape
                    // `mem::discriminant(&10)` produces — the
                    // rvalue is well-defined for every type, and
                    // only enum-like values carry a real tag.
                    _ => Ok(Value::Int(0)),
                }
            }
            Rvalue::CopyForDeref(place) => {
                let value = self.read_place(frames, body, place)?;
                match &value {
                    Value::Ref(_) | Value::FatRef { .. } => {
                        let target = self.deref_value(frames, &value)?;
                        self.read_resolved(frames, &target)
                    }
                    _ => Ok(value),
                }
            }
            Rvalue::Reborrow(place) => {
                // A bitwise copy of the pointer value held at the
                // place, reinterpreted as the target pointer type.
                // The value model tracks neither pointer mutability
                // nor provenance, and a CoerceShared pair has equal
                // layout by contract, so the copy is the value itself
                // (for a wrapper ADT around a pointer, the aggregate
                // itself, exactly as a bitwise copy prescribes).
                self.read_place(frames, body, place)
            }
            Rvalue::ThreadLocalRef(name) => {
                // A reference to the executing thread's storage for
                // this thread-local static, seeded on first access
                // from the initializer the driver recorded. The
                // storage's own machinery (std's `LocalKey` and
                // storage bodies, which are ordinary collected code)
                // performs the user-visible initialization; the seed
                // is only the storage struct's initial value.
                // Destructors registered for a storage run when
                // the owning thread finishes (see `run_thread`);
                // the entries themselves stay in the store under
                // the thread's unique id.
                let Some(seed) = self.program.thread_locals.get(name) else {
                    return Err(InterpError::UnsupportedRvalue {
                        kind: format!(
                            "ThreadLocalRef {name}: no thread-local definition \
                             was collected for this static (its initializer \
                             could not be decoded)"
                        ),
                    });
                };
                let key = Self::tls_store_key(self.current_thread_id(), name);
                self.ensure_static(&key, seed)?;
                Ok(Value::Ref(RefTarget::Static {
                    name: key,
                    projection: Vec::new(),
                }))
            }
            Rvalue::Unsupported(kind) => Err(InterpError::UnsupportedRvalue {
                kind: kind.clone(),
            }),
        }
    }
}

/// Human-readable form of a resolved place, for error messages.
pub(crate) fn describe_resolved(local: usize, projection: &[RefProj]) -> String {
    use std::fmt::Write as _;
    let mut out = format!("_{local}");
    for proj in projection {
        match proj {
            RefProj::Field(i) => {
                let _ = write!(out, ".{i}");
            }
            RefProj::UnionField { idx, .. } => {
                let _ = write!(out, ".{idx}");
            }
            RefProj::Index(i) => {
                let _ = write!(out, "[{i}]");
            }
            RefProj::Downcast(v) => {
                let _ = write!(out, " as variant {v}");
            }
            RefProj::Subslice { offset, len } => {
                let _ = write!(out, "[{offset}..{}]", offset + len);
            }
            RefProj::ByteOffset(d) => {
                let _ = write!(out, " +byte {d}");
            }
        }
    }
    out
}

/// The guest panic rustc raises for an out-of-bounds index.
/// The (offset, element count) a MIR `Subslice` projection
/// selects from a base of `len` elements, with core's own panic
/// messages for an out-of-range selection. With `from_end`, `to`
/// counts back from the end of the base (the range is
/// `from..len - to`); otherwise `to` is the range's end.
fn subslice_offset_len(
    from: u64,
    to: u64,
    from_end: bool,
    len: u128,
) -> InterpResult<(usize, usize)> {
    let from = from as u128;
    let end = if from_end {
        len.checked_sub(to as u128).ok_or_else(|| {
            InterpError::panic(format!(
                "range end index {to} out of range for slice of length {len}"
            ))
        })?
    } else {
        to as u128
    };
    if from > end {
        return Err(InterpError::panic(format!(
            "slice index starts at {from} but ends at {end}"
        )));
    }
    if end > len {
        return Err(InterpError::panic(format!(
            "range end index {end} out of range for slice of length {len}"
        )));
    }
    let offset = usize::try_from(from).map_err(|_| {
        InterpError::panic(format!("slice index {from} does not fit in usize"))
    })?;
    let count = usize::try_from(end - from).map_err(|_| {
        InterpError::panic(format!(
            "slice length {} does not fit in usize",
            end - from
        ))
    })?;
    Ok((offset, count))
}

/// Check an already-resolved (offset, count) sub-slice selection
/// against a materialised value's element count, with the same
/// panic messages as [`subslice_offset_len`].
fn check_subslice(value_len: usize, offset: usize, count: usize) -> InterpResult<()> {
    let end = offset + count;
    if end > value_len {
        return Err(InterpError::panic(format!(
            "range end index {end} out of range for slice of length {value_len}"
        )));
    }
    Ok(())
}

/// Compose an index step that immediately follows a fat-pointer
/// dereference with the resolved target's trailing element index
/// (the caller has already established that the dereference does
/// not denote the slice's own storage — see
/// `Interpreter::deref_denotes_storage`).
/// A fat pointer's base denotes the slice's first element, so
/// when that base is itself an element position (the trailing
/// step is an index into the container the slice lives in), the
/// slice-relative index denotes the element `base + idx` of that
/// container, not an element of the first element: the trailing
/// step is advanced instead of a nested step being appended.
/// Any other target shape answers `None` and the caller appends
/// the step as usual.
fn compose_fat_index(cur: &Resolved, idx: usize) -> Option<Resolved> {
    fn advanced(projection: &[RefProj], idx: usize) -> Option<Vec<RefProj>> {
        match projection.last() {
            Some(RefProj::Index(k)) => {
                let mut out = projection.to_vec();
                *out.last_mut().expect("nonempty projection") =
                    RefProj::Index(k.checked_add(idx)?);
                Some(out)
            }
            _ => None,
        }
    }
    match cur {
        Resolved::Place {
            frame_pos,
            local,
            projection,
        } => Some(Resolved::Place {
            frame_pos: *frame_pos,
            local: *local,
            projection: advanced(projection, idx)?,
        }),
        Resolved::Static { name, projection } => Some(Resolved::Static {
            name: name.clone(),
            projection: advanced(projection, idx)?,
        }),
        Resolved::Alloc { id, projection } => Some(Resolved::Alloc {
            id: *id,
            projection: advanced(projection, idx)?,
        }),
        Resolved::Value(..) => None,
    }
}

/// The sub-aggregate a `Subslice` projection step denotes over
/// an element container value: the elements `offset..offset+len`
/// of a tuple, an ADT value whose fields are its elements, a byte
/// string or a string. Anything else (or an out-of-range window)
/// answers `None`.
fn subslice_value(value: &Value, offset: usize, len: usize) -> Option<Value> {
    let end = offset.checked_add(len)?;
    match value {
        Value::Tuple(items) => Some(Value::Tuple(items.get(offset..end)?.to_vec())),
        Value::Enum { variant, fields } => Some(Value::Enum {
            variant: *variant,
            fields: fields.get(offset..end)?.to_vec(),
        }),
        Value::Bytes(b) => Some(Value::Bytes(b.get(offset..end)?.to_vec())),
        Value::Str(text) => Some(Value::Str(text.get(offset..end)?.to_string())),
        _ => None,
    }
}

fn index_oob(len: usize, index: usize) -> InterpError {
    InterpError::panic(format!(
        "index out of bounds: the len is {len} but the index is {index}"
    ))
}

/// Byte `index` of an integer value's little-endian
/// representation (see `Interpreter::seq_scalar`). The model's
/// word is 128 bits; a byte past it is outside every integer
/// type the model holds, so it is an explicit error rather
/// than a fabricated byte.
/// The scalar leaves of a structured scalar value (a vector
/// value's tree of integer/float/bool leaves inside its
/// aggregate wrappers), in storage order; `None` when any part
/// of the tree is not a scalar leaf or a plain aggregate over
/// them. Byte-granular bulk operations (the byte copy an
/// unaligned vector load lowers to) address such a value's
/// bytes, which the indexed access cannot express: the leaves'
/// widths are not recorded in the values themselves.
fn scalar_tree_leaves(value: &Value) -> Option<Vec<&Value>> {
    fn walk<'v>(value: &'v Value, out: &mut Vec<&'v Value>) -> Option<()> {
        match value {
            Value::Int(_) | Value::Uint(_) | Value::Float(_) | Value::Bool(_) => {
                out.push(value);
            }
            Value::Tuple(items) => {
                for item in items {
                    walk(item, out)?;
                }
            }
            Value::Enum { fields, .. } => {
                for field in fields {
                    walk(field, out)?;
                }
            }
            _ => return None,
        }
        Some(())
    }
    let mut out = Vec::new();
    walk(value, &mut out)?;
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Whether `value` is a structured scalar tree (an aggregate
/// over scalar leaves) rather than a plain scalar or a
/// sequence value the indexed access already handles.
fn is_scalar_tree(value: &Value) -> bool {
    matches!(value, Value::Tuple(_) | Value::Enum { .. })
        && scalar_tree_leaves(value).is_some()
}

/// The leaf width a byte view of a scalar tree uses: every
/// leaf must share one kind, and the total bit count must
/// divide evenly into a width the model can store. Vector
/// values — the trees this serves — are uniform by
/// construction (a `Simd<T, N>` storage array, a wrapper over
/// one scalar); a mixed tree has no honest byte view here and
/// keeps the plain access's outcome.
fn scalar_tree_leaf_width(leaves: &[&Value], total_bits: u32) -> Option<u32> {
    let count = leaves.len() as u32;
    if count == 0 || total_bits % count != 0 {
        return None;
    }
    let width = total_bits / count;
    let kind = leaves[0];
    if !leaves.iter().all(|leaf| {
        matches!(
            (kind, leaf),
            (Value::Int(_), Value::Int(_))
                | (Value::Uint(_), Value::Uint(_))
                | (Value::Float(_), Value::Float(_))
                | (Value::Bool(_), Value::Bool(_))
        )
    }) {
        return None;
    }
    match kind {
        Value::Int(_) | Value::Uint(_) if matches!(width, 8 | 16 | 32 | 64 | 128) => Some(width),
        Value::Float(_) if matches!(width, 32 | 64) => Some(width),
        Value::Bool(_) if width == 8 => Some(width),
        _ => None,
    }
}

/// Rebuild a scalar tree with its leaf bits taken from
/// `bytes` (little-endian, in storage order), preserving the
/// tree's shape and leaf kinds. `None` when the tree has no
/// uniform byte view (see [`scalar_tree_leaf_width`]).
fn byte_patch_tree(current: &Value, bytes: &[u8]) -> Option<Value> {
    let leaves = scalar_tree_leaves(current)?;
    let width = scalar_tree_leaf_width(&leaves, (bytes.len() as u32).checked_mul(8)?)?;
    let mut pos = 0usize;
    fn build(template: &Value, bytes: &[u8], width: u32, pos: &mut usize) -> Option<Value> {
        let nbytes = (width / 8) as usize;
        let take = |pos: &mut usize| -> Option<u128> {
            let mut bits = 0u128;
            for i in 0..nbytes {
                bits |= u128::from(*bytes.get(*pos + i)?) << (8 * i);
            }
            *pos += nbytes;
            Some(bits)
        };
        match template {
            Value::Int(_) => {
                let bits = take(pos)?;
                Some(Value::Int(wrap_signed(bits as i128, width)))
            }
            Value::Uint(_) => {
                let bits = take(pos)?;
                Some(Value::Uint(wrap_unsigned(bits, width)))
            }
            Value::Float(_) => {
                let bits = take(pos)?;
                Some(Value::Float(match width {
                    32 => f64::from(f32::from_bits(bits as u32)),
                    _ => f64::from_bits(bits as u64),
                }))
            }
            Value::Bool(_) => {
                let bits = take(pos)?;
                Some(Value::Bool(bits != 0))
            }
            Value::Tuple(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(build(item, bytes, width, pos)?);
                }
                Some(Value::Tuple(out))
            }
            Value::Enum { variant, fields } => {
                let mut out = Vec::with_capacity(fields.len());
                for field in fields {
                    out.push(build(field, bytes, width, pos)?);
                }
                Some(Value::Enum {
                    variant: *variant,
                    fields: out,
                })
            }
            _ => None,
        }
    }
    build(current, bytes, width, &mut pos)
}

/// Serialize a scalar tree to its little-endian bytes when it
/// has a uniform byte view of exactly `total` bytes (the
/// mirror of [`byte_patch_tree`], for a vector store's source).
fn tree_to_bytes(value: &Value, total: usize) -> Option<Vec<u8>> {
    let leaves = scalar_tree_leaves(value)?;
    let width = scalar_tree_leaf_width(&leaves, (total as u32).checked_mul(8)?)?;
    let nbytes = (width / 8) as usize;
    let mut out = Vec::with_capacity(total);
    for leaf in leaves {
        let bits: u128 = match leaf {
            Value::Int(i) => (*i as u128) & bit_mask(width),
            Value::Uint(u) => *u & bit_mask(width),
            Value::Float(f) => match width {
                32 => u128::from((*f as f32).to_bits()),
                _ => u128::from(f.to_bits()),
            },
            Value::Bool(b) => u128::from(*b),
            _ => return None,
        };
        for i in 0..nbytes {
            out.push((bits >> (8 * i)) as u8);
        }
    }
    if out.len() == total {
        Some(out)
    } else {
        None
    }
}

fn scalar_byte(value: &Value, index: usize) -> InterpResult<u8> {
    if index >= 16 {
        return Err(InterpError::UnsupportedProjection {
            kind: format!(
                "byte {index} of a scalar value is outside the model's word"
            ),
        });
    }
    let bits = match value {
        Value::Int(v) => *v as u128,
        Value::Uint(v) => *v,
        other => {
            return Err(InterpError::type_mismatch(
                "scalar byte access",
                "integer value",
                other.kind_name(),
            ));
        }
    };
    Ok((bits >> (index * 8)) as u8)
}

/// The integer value with byte `index` of its little-endian
/// representation replaced by `byte` (see
/// [`scalar_byte`]). The signed form keeps its two's-complement
/// word: patching a byte of a negative value patches the
/// representation the native write would touch.
fn scalar_with_byte(value: &Value, index: usize, byte: u8) -> InterpResult<Value> {
    if index >= 16 {
        return Err(InterpError::UnsupportedProjection {
            kind: format!(
                "byte {index} of a scalar value is outside the model's word"
            ),
        });
    }
    let shift = index * 8;
    let mask = !(0xFFu128 << shift);
    Ok(match value {
        Value::Int(v) => {
            Value::Int(((*v as u128 & mask) | ((byte as u128) << shift)) as i128)
        }
        Value::Uint(v) => Value::Uint((*v & mask) | ((byte as u128) << shift)),
        other => {
            return Err(InterpError::type_mismatch(
                "scalar byte access",
                "integer value",
                other.kind_name(),
            ));
        }
    })
}

/// Apply constant projection steps to a value (read direction).
/// The canonical value of a type that is zero-sized by its IR
/// shape alone (see `Interpreter::zst_local_value`): unit, tuples
/// of zero-sized types, arrays of zero-sized elements and empty
/// arrays, zero-capture closures, function items. ADT values need
/// the driver's shape tables and are only synthesized at
/// conversion time.
fn zst_value_of_ty(ty: &Ty) -> Option<Value> {
    match ty {
        Ty::Unit => Some(Value::Unit),
        Ty::Tuple(ts) => {
            let mut vals = Vec::with_capacity(ts.len());
            for t in ts {
                vals.push(zst_value_of_ty(t)?);
            }
            Some(Value::Tuple(vals))
        }
        Ty::Array(_, 0) => Some(Value::Tuple(Vec::new())),
        Ty::Array(elem, n) if *n <= 4096 => {
            let v = zst_value_of_ty(elem)?;
            Some(Value::Tuple(vec![v; *n as usize]))
        }
        Ty::Closure(def, 0) => Some(Value::Closure {
            def: def.clone(),
            env: Vec::new(),
        }),
        Ty::FnDef(name) => Some(Value::FnPtr(name.clone())),
        _ => None,
    }
}

/// Whether a projection contains an index step immediately
/// followed by a step that indexes further (another index, or
/// a sub-slice): the only shape [`compose_projection`] rewrites.
fn projection_needs_composition(projection: &[RefProj]) -> bool {
    projection.windows(2).any(|w| {
        matches!(w[0], RefProj::Index(_))
            && matches!(w[1], RefProj::Index(_) | RefProj::Subslice { .. })
    })
}

/// Compose the element-position index runs of a projection.
///
/// An index step that lands on one element of a sequence,
/// followed by steps that index further, denotes a position
/// in the sequence rather than a read of the element: the
/// first step came from a data pointer into the sequence (a
/// raw-parts slice's base, an `as_ptr` result), and the
/// following steps address the sequence's elements from that
/// position, so the run composes into a single index (or, with
/// a trailing sub-slice step, a sub-slice) from the sequence's
/// start. When the element is itself a sequence value the
/// steps nest as written (a row of a matrix, one string of a
/// table) - the same judgment `seq_element_target` makes for
/// sequence reads through an element pointer. Steps the walk
/// cannot evaluate pass through unchanged; the caller's own
/// application then reproduces their result or error.
fn compose_projection(
    value: &Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> Vec<RefProj> {
    let mut out: Vec<RefProj> = Vec::with_capacity(projection.len());
    let mut cur = value.clone();
    let mut i = 0;
    while i < projection.len() {
        if let RefProj::Index(a) = projection[i] {
            let mut total = a;
            let mut j = i + 1;
            while let Some(RefProj::Index(b)) = projection.get(j) {
                total += b;
                j += 1;
            }
            let subslice = match projection.get(j) {
                Some(RefProj::Subslice { offset, len }) => Some((*offset, *len)),
                _ => None,
            };
            if j > i + 1 || subslice.is_some() {
                let element = project_value(cur.clone(), &[RefProj::Index(a)], func, place);
                if let Ok(element) = element {
                    if !matches!(
                        element,
                        Value::Tuple(_)
                            | Value::FatRef { .. }
                            | Value::Bytes(_)
                            | Value::Str(_)
                    ) {
                        let merged = match subslice {
                            Some((offset, len)) => RefProj::Subslice {
                                offset: total + offset,
                                len,
                            },
                            None => RefProj::Index(total),
                        };
                        if let Ok(next) = project_value(
                            cur.clone(),
                            std::slice::from_ref(&merged),
                            func,
                            place,
                        ) {
                            cur = next;
                            out.push(merged);
                            i = if subslice.is_some() { j + 1 } else { j };
                            continue;
                        }
                    }
                }
            }
        }
        match project_value(cur.clone(), std::slice::from_ref(&projection[i]), func, place) {
            Ok(next) => {
                cur = next;
                out.push(projection[i].clone());
                i += 1;
            }
            Err(_) => {
                out.extend_from_slice(&projection[i..]);
                return out;
            }
        }
    }
    out
}

/// The run-time half of [`Rvalue::PtrCastUnits`]: rewrite a
/// trailing element index in the pointer's projection as the
/// byte offset it denotes in the source pointee's units
/// (`index * from_size`). Every other shape — no trailing
/// index, a byte offset already, a zero index (the same
/// position in any unit), an index whose byte position does
/// not fit — passes through unchanged: the position is the
/// same, only its denomination was ambiguous.
pub(crate) fn ptr_cast_units_value(value: Value, from_size: u64) -> Value {
    fn rescale(projection: &mut Vec<RefProj>, from_size: u64) {
        if let Some(RefProj::Index(k)) = projection.last() {
            let k = *k;
            if k > 0 {
                if let Some(bytes) = (k as u128).checked_mul(u128::from(from_size)) {
                    if let Ok(d) = i128::try_from(bytes) {
                        *projection.last_mut().expect("last checked") =
                            RefProj::ByteOffset(d);
                    }
                }
            }
        }
    }
    match value {
        Value::Ref(target) => Value::Ref(match target {
            // Only an allocation target's index is denominated
            // in the pointer's own pointee units: a raw block
            // has no declared element type, so the projection
            // the pointer carries is the only denomination in
            // play. A place, static or value target's index
            // belongs to the declared aggregate it projects
            // into (an array local's element index), which a
            // pointer cast does not redenominate.
            RefTarget::Alloc { id, mut projection } => {
                rescale(&mut projection, from_size);
                RefTarget::Alloc { id, projection }
            }
            other => other,
        }),
        Value::FatRef { base, len } => Value::FatRef {
            base: Box::new(ptr_cast_units_value(*base, from_size)),
            len,
        },
        Value::DynRef { base, concrete, traits, drop_glue } => Value::DynRef {
            base: Box::new(ptr_cast_units_value(*base, from_size)),
            concrete,
            traits,
            drop_glue,
        },
        other => other,
    }
}

pub(crate) fn project_value(
    value: Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<Value> {
    let composed;
    let projection = if projection_needs_composition(projection) {
        composed = compose_projection(&value, projection, func, place);
        &composed[..]
    } else {
        projection
    };
    let mut value = value;
    let mut step_idx = 0;
    // The state a coroutine downcast named, for the field step
    // that follows it (see the `Downcast` arm below).
    let mut downcast_state: Option<u32> = None;
    while step_idx < projection.len() {
        let step = &projection[step_idx];
        step_idx += 1;
        let pos = step_idx - 1;
        if value == Value::Uninit {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        }
        let step_state = downcast_state.take();
        value = match step {
            RefProj::Field(idx) => match value {
                Value::Tuple(ref items) => items.get(*idx).cloned().ok_or_else(|| {
                    InterpError::UninitializedRead {
                        func: func.to_string(),
                        place: place.to_string(),
                    }
                })?,
                Value::Enum { ref fields, .. } => {
                    fields.get(*idx).cloned().ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?
                }
                Value::Coroutine {
                    ref fields,
                    ref saved,
                    ..
                } => {
                    // A field step through a downcast reads that
                    // state's saved storage; without one it reads
                    // the upvar storage (see `Value::Coroutine`).
                    // State 0 is the unresumed state, whose fields
                    // ARE the upvars, so its downcast addresses the
                    // upvar storage too.
                    let storage = match step_state {
                        Some(0) | None => Some(fields),
                        Some(state) => saved.get(state as usize),
                    };
                    storage
                        .and_then(|fields| fields.get(*idx))
                        .cloned()
                        .ok_or_else(|| InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        })?
                }
                Value::Closure { ref env, .. } => {
                    env.get(*idx).cloned().ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?
                }
                // Field 0 of a string value is the string
                // itself: the byte-string wrappers std layers
                // over the same bytes (`String` over its buffer,
                // `OsString` over its `Buf`, `PathBuf` over its
                // `OsString`, `Path` over its `OsStr`, `OsStr`
                // over its byte slice) are all one-field chains
                // whose every layer denotes the same byte string
                // in this model, and the `AsRef` / `Deref` glue
                // that walks them (e.g. the projections inside
                // `env::split_paths`) reads the same value at
                // every step. Consumers that need the buffer in
                // vector form reach it through the modelled
                // `Vec` API, not through these projections.
                Value::Str(_) if *idx == 0 => value,
                // Field 0 of a fat pointer value is the pointer
                // field of its smart-pointer wrapper (a `Box`'s
                // `Unique`): in this model the `DynRef` *is* that
                // pointer, metadata included.
                Value::DynRef { .. } if *idx == 0 => value,
                // The same wrapper chain for a thin pointer: a
                // `Box`'s field 0 is its `Unique`, whose field 0
                // is the `NonNull`, whose field 0 is the raw
                // pointer — and in this model the box's one field
                // is the pointer value itself, so each of those
                // steps yields it unchanged. (A reference to a
                // place never sits at a field step of a value
                // projection: place references are resolved before
                // value projection begins, so a `Ref` here is a
                // pointer value extracted from a wrapper.)
                Value::Ref(_) if *idx == 0 => value,
                // The same wrapper chain for a byte string: a
                // `Path`'s field 0 is its `OsStr`, whose field 0
                // is the byte slice, and in this model the value
                // at the base of that chain is the string itself
                // (the constructors reach it through raw pointer
                // casts: `Path::new` casts `&str` to `&Path`).
                // Each wrapper's only field sits at offset 0 with
                // exactly the string's representation, so the
                // step yields the value unchanged.
                Value::Bytes(_) if *idx == 0 => value,
                // A one-field wrapper over a scalar whose storage
                // the model holds as the bare scalar (a control
                // byte reached through a byte-string constant:
                // the tag's storage is the byte itself): field 0
                // of the wrapper is the scalar it wraps.
                Value::Int(_) | Value::Uint(_) | Value::Float(_) | Value::Bool(_)
                    if *idx == 0 =>
                {
                    value
                }
                // String data reached through its own wrapper
                // descent: a `String` the model holds as bare
                // string data has no materialised `Vec` /
                // `RawVec` / `Unique` / `NonNull` structure, yet
                // converted code (the inlined `Borrow::borrow`
                // for `Cow`, `String::as_ptr`, ...) walks exactly
                // that one-field chain to the data. Field 0 of
                // the chain is the string data itself, yielded
                // unchanged like the byte-string wrapper chain
                // above: the consumers that need a pointer form
                // the reference at their own site (reference
                // formation reads the target's value to recover
                // the fat length, so the read must stay the
                // data). `Field(1)` at any step is the vector's
                // length slot, the string's length.
                Value::Str(_) if *idx == 0 => value,
                Value::Str(ref text) if *idx == 1 => {
                    Value::Uint(text.len() as u128)
                }
                // The two fields of a modelled `fmt::Arguments`
                // (see the struct's encoding in core::fmt): field
                // 0 is the `template: NonNull<u8>`, field 1 the
                // `args: NonNull<rt::Argument>`. The model keeps
                // neither as storage, so the projection answers
                // with the field's own value: a one-field `NonNull`
                // wrapper around a pointer to the modelled bytes
                // / argument array, or — for the `from_str` form,
                // whose `args` field is not a pointer at all — the
                // tagged length word `(len << 1) | 1` as an
                // address-only pointer, which is exactly what
                // `Arguments::as_str` transmutes it back to.
                Value::FmtArguments(ref args) => {
                    use crate::value::FmtArguments;
                    let ptr = match (idx, args) {
                        (0, FmtArguments::Plain(text)) => {
                            Value::Ref(RefTarget::Value(
                                Box::new(Value::Str(text.clone())),
                                Vec::new(),
                            ))
                        }
                        (0, FmtArguments::Template { template, .. }) => {
                            Value::Ref(RefTarget::Value(
                                Box::new(Value::Bytes(template.clone())),
                                Vec::new(),
                            ))
                        }
                        (1, FmtArguments::Plain(text)) => {
                            Value::Ref(RefTarget::Address(
                                ((text.len() as u64) << 1) | 1,
                            ))
                        }
                        (1, FmtArguments::Template { args, .. }) => {
                            Value::Ref(RefTarget::Value(
                                Box::new(Value::Tuple(args.clone())),
                                Vec::new(),
                            ))
                        }
                        _ => {
                            return Err(InterpError::type_mismatch(
                                format!("field projection on {place}"),
                                "field 0 or 1 of fmt::Arguments",
                                format!("field {idx}"),
                            ));
                        }
                    };
                    Value::Enum {
                        variant: 0,
                        fields: vec![ptr],
                    }
                }
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("field projection on {place}"),
                        "tuple or ADT value",
                        other.kind_name(),
                    ));
                }
            },
            RefProj::UnionField { .. } => {
                // Union field reads need the program's scalar
                // shapes, which this type-free projector does not
                // carry: every interpreter read path routes union
                // steps through `project_value_t` instead. Landing
                // here means a caller applied a stored projection
                // directly; that is an explicit error, not a slot
                // read.
                return Err(InterpError::type_mismatch(
                    format!("union field projection on {place}"),
                    "a union field read through the typed projection path",
                    "the untyped value projector",
                ));
            }
            RefProj::Index(idx) => match value {
                Value::FatRef {
                    ref base,
                    len,
                } => {
                    if *idx as u128 >= len {
                        return Err(index_oob(len as usize, *idx));
                    }
                    project_value((**base).clone(), &[RefProj::Index(*idx)], func, place)?
                }
                Value::Tuple(ref items) => items
                    .get(*idx)
                    .cloned()
                    .ok_or_else(|| index_oob(items.len(), *idx))?,
                Value::Bytes(ref b) => b
                    .get(*idx)
                    .map(|byte| Value::Uint(*byte as u128))
                    .ok_or_else(|| index_oob(b.len(), *idx))?,
                Value::Str(ref s) => s
                    .as_bytes()
                    .get(*idx)
                    .map(|byte| Value::Uint(*byte as u128))
                    .ok_or_else(|| index_oob(s.len(), *idx))?,
                // A single-field aggregate whose only field is
                // an array is layout-identical to the array
                // itself, and guest code indexes through that
                // pun (a pointer cast between a `#[repr(simd)]`
                // struct and `[T; N]`, as `Simd::as_array`
                // does): the index step continues into the
                // field's elements.
                Value::Enum { ref fields, .. }
                    if fields.len() == 1
                        && matches!(
                            fields[0],
                            Value::Tuple(_) | Value::Bytes(_)
                        ) =>
                {
                    project_value(
                        fields[0].clone(),
                        &[RefProj::Index(*idx)],
                        func,
                        place,
                    )?
                }
                // A `Vec` value is the aggregate of its elements
                // (see the `Vec::from` / `Deref` builtin model):
                // indexing the vector's own place reads the
                // element field. No other ADT value can be the
                // base of an index projection in valid MIR.
                Value::Enum { ref fields, .. } => fields
                    .get(*idx)
                    .cloned()
                    .ok_or_else(|| index_oob(fields.len(), *idx))?,
                ref other => {
                    // A scalar's storage is byte-addressable
                    // through a raw byte pointer (see
                    // `scalar_storage_byte`): the index step
                    // reads the byte at that offset of the
                    // storage, the only element it has.
                    if let Some(len) = scalar_storage_len(other) {
                        if *idx >= len {
                            return Err(index_oob(len, *idx));
                        }
                        Value::Uint(u128::from(
                            scalar_storage_byte(other, *idx)
                                .expect("index inside the storage"),
                        ))
                    } else {
                        return Err(InterpError::type_mismatch(
                            format!("index projection on {place}"),
                            "array, byte or string value",
                            other.kind_name(),
                        ));
                    }
                }
            },
            RefProj::Subslice { offset, len } => {
                let (start, end) = (*offset, *offset + *len);
                match value {
                    Value::Tuple(ref items) => {
                        check_subslice(items.len(), start, *len)?;
                        Value::Tuple(items[start..end].to_vec())
                    }
                    Value::Bytes(ref b) => {
                        check_subslice(b.len(), start, *len)?;
                        Value::Bytes(b[start..end].to_vec())
                    }
                    Value::Str(ref s) => match s.get(start..end) {
                        Some(sub) => Value::Str(sub.to_string()),
                        None => {
                            return Err(InterpError::panic(format!(
                                "byte range {start}..{end} is not a valid slice of the string at {place}"
                            )))
                        }
                    },
                    // A `Vec` value is the aggregate of its
                    // elements: its sub-slice is the aggregate of
                    // the sub-range's elements.
                    Value::Enum { variant, ref fields } => {
                        check_subslice(fields.len(), start, *len)?;
                        Value::Enum {
                            variant,
                            fields: fields[start..end].to_vec(),
                        }
                    }
                    // A sub-slice of a fat reference value (a
                    // constant slice's representation): a zero
                    // offset only adjusts the carried length; a
                    // materialised base is projected directly.
                    Value::FatRef { ref base, len: fat_len } => {
                        if start == 0 {
                            if end as u128 > fat_len {
                                return Err(InterpError::panic(format!(
                                    "range end index {end} out of range for slice of length {fat_len}"
                                )));
                            }
                            Value::FatRef {
                                base: base.clone(),
                                len: end as u128,
                            }
                        } else {
                            match &**base {
                                Value::Tuple(_) | Value::Bytes(_) | Value::Str(_) => {
                                    project_value(
                                        (**base).clone(),
                                        &[RefProj::Subslice { offset: start, len: *len }],
                                        func,
                                        place,
                                    )?
                                }
                                other => {
                                    return Err(InterpError::type_mismatch(
                                        format!("subslice projection on {place}"),
                                        "materialised slice value",
                                        other.kind_name(),
                                    ))
                                }
                            }
                        }
                    }
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("subslice projection on {place}"),
                            "array, slice or string value",
                            other.kind_name(),
                        ));
                    }
                }
            }
            RefProj::Downcast(variant) => match value {
                // A coroutine's downcast names the state whose saved
                // storage the following field step uses; the access
                // is valid whatever the current tag is (the body
                // writes a state's fields before setting the tag to
                // that state). The tag itself does not move here:
                // the state is remembered for the next step only.
                Value::Coroutine { .. } => {
                    downcast_state = Some(*variant);
                    value
                }
                Value::Enum {
                    variant: current, ..
                } => {
                    if current != *variant {
                        return Err(InterpError::type_mismatch(
                            format!("downcast of {place}"),
                            format!("variant {variant}"),
                            format!("variant {current}"),
                        ));
                    }
                    value
                }
                Value::Tuple(_) if *variant == 0 => value,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("downcast of {place}"),
                        "ADT value",
                        other.kind_name(),
                    ));
                }
            },
            RefProj::ByteOffset(disp) => {
                if *disp == 0 {
                    value
                } else if *disp >= 0 {
                    let Some(len) = scalar_storage_len(&value) else {
                        return Err(InterpError::UnsupportedProjection {
                            kind: format!(
                                "read through a pointer displaced {disp} bytes from a value boundary"
                            ),
                        });
                    };
                    let Some(&RefProj::Index(k)) = projection.get(step_idx) else {
                        return Err(InterpError::UnsupportedProjection {
                            kind: format!(
                                "read through a pointer displaced {disp} bytes from a value boundary"
                            ),
                        });
                    };
                    // A byte read through a pointer displaced into
                    // a scalar's storage: the index step that
                    // follows selects the byte at the combined
                    // offset, the read side of the byte view
                    // `scalar_storage_byte` describes.
                    let at = *disp as usize + k;
                    step_idx += 1;
                    match scalar_storage_byte(&value, at) {
                        Some(byte) => Value::Uint(u128::from(byte)),
                        None => return Err(index_oob(len, at)),
                    }
                } else {
                    // The position does not coincide with a value
                    // boundary, and the model has no byte-level
                    // view of a value's storage to reinterpret.
                    return Err(InterpError::UnsupportedProjection {
                        kind: format!(
                            "read through a pointer displaced {disp} bytes from a value boundary"
                        ),
                    });
                }
            }
        };
    }
    Ok(value)
}

/// The number of bytes in a scalar value's storage, for the
/// byte-level view of that storage a raw byte pointer has (see
/// [`scalar_storage_byte`]). Integers ride in a 128-bit slot, so
/// their storage is the full 16 bytes; the guest only ever
/// addresses the bytes of the value's own declared width, which
/// are the low ones.
fn scalar_storage_len(value: &Value) -> Option<usize> {
    match value {
        Value::Bool(_) => Some(1),
        Value::Float(_) => Some(8),
        Value::Int(_) | Value::Uint(_) | Value::Float128(_) => Some(16),
        _ => None,
    }
}

/// Byte `idx` of a scalar value's storage, little-endian: the
/// byte a `*const u8` cast of the value's address reads at that
/// offset. Bulk byte copies move data in and out of scalar
/// buffers this way (SipHash's `Hasher::write` assembles its
/// next word in a `u64` local), so an index step applied to a
/// scalar value denotes this byte, the only element a scalar's
/// storage has. `None` for a non-scalar value or an index past
/// the storage.
fn scalar_storage_byte(value: &Value, idx: usize) -> Option<u8> {
    let len = scalar_storage_len(value)?;
    if idx >= len {
        return None;
    }
    let bits: u128 = match value {
        Value::Bool(b) => u128::from(*b),
        Value::Int(i) => *i as u128,
        Value::Uint(u) => *u,
        Value::Float(f) => u128::from(f.to_bits()),
        Value::Float128(bits) => *bits,
        _ => return None,
    };
    Some((bits >> (8 * idx)) as u8)
}

/// The byte a written value carries when it is a byte value: a
/// `u8` (or an `i8`, or a `bool`) written through a byte pointer.
fn value_as_storage_byte(value: &Value) -> Option<u8> {
    match value {
        Value::Uint(u) if *u <= 0xff => Some(*u as u8),
        Value::Int(i) if (-128..=127).contains(i) => Some(*i as u8),
        Value::Bool(b) => Some(*b as u8),
        _ => None,
    }
}

/// Splice `byte` into byte `idx` of a scalar value's storage
/// (see [`scalar_storage_byte`]), in place. An unwritten slot is
/// zero storage for this purpose: a buffer being filled byte by
/// byte is written whole before it is read, so the bytes the
/// fill has not reached yet are never observed. Returns whether
/// the splice applied (the slot held a scalar, or no value, and
/// the index is inside the storage).
fn splice_scalar_storage_byte(slot: &mut Value, idx: usize, byte: u8) -> bool {
    let len = match &*slot {
        Value::Uninit => 16,
        other => match scalar_storage_len(other) {
            Some(len) => len,
            None => return false,
        },
    };
    if idx >= len {
        return false;
    }
    let bits: u128 = match &*slot {
        Value::Bool(b) => u128::from(*b),
        Value::Int(i) => *i as u128,
        Value::Uint(u) => *u,
        Value::Float(f) => u128::from(f.to_bits()),
        Value::Float128(bits) => *bits,
        _ => 0,
    };
    let bits = (bits & !(0xffu128 << (8 * idx))) | (u128::from(byte) << (8 * idx));
    *slot = match &*slot {
        Value::Bool(_) => Value::Bool(bits != 0),
        Value::Int(_) => Value::Int(bits as i128),
        Value::Float(_) => Value::Float(f64::from_bits(bits as u64)),
        Value::Float128(_) => Value::Float128(bits),
        _ => Value::Uint(bits),
    };
    true
}

/// Assign `value` to the slot `projection` selects inside the
/// stored value `slot` (the final step of a `write_resolved`
/// once the target's base slot is in hand). One shape needs a
/// read-modify-write instead of a slot assignment: a byte
/// written through a trailing index step into a scalar's
/// storage (see [`scalar_storage_byte`]) has no byte slot to
/// hand out, so it is spliced into the scalar in place. Every
/// other shape navigates to the selected slot and assigns, with
/// exactly the errors that navigation produced before.
fn assign_projected(
    slot: &mut Value,
    projection: &[RefProj],
    value: Value,
    func: &str,
    place: &str,
) -> InterpResult<()> {
    // The byte-splice pre-passes below read the projection's
    // last step as a byte position inside the value the head
    // denotes; an element-position index run (see
    // `compose_projection`) must be composed first, or the
    // splice mistakes the run's last index for a byte offset
    // into the element at the run's head.
    let composed;
    let projection = if projection_needs_composition(projection) {
        composed = compose_projection(slot, projection, func, place);
        &composed[..]
    } else {
        projection
    };
    if let Some((&RefProj::ByteOffset(disp), head)) = projection.split_last() {
        // A byte written through a pointer displaced into a
        // scalar's storage, with no index step after the
        // displacement (a one-byte bulk write, or the direct
        // write of a single element): the byte value being
        // written is what makes the position a byte position.
        if disp >= 0 {
            if let Some(byte) = value_as_storage_byte(&value) {
                if let Ok(head_slot) = navigate_mut(slot, head, func, place) {
                    if splice_scalar_storage_byte(head_slot, disp as usize, byte) {
                        return Ok(());
                    }
                }
            }
        }
    }
    if let Some((&RefProj::Index(idx), head)) = projection.split_last() {
        // A byte pointer displaced into the storage first
        // (`p.add(k)`) puts a byte-offset step immediately before
        // the index step; the byte position is the sum.
        let (idx, head) = match head.split_last() {
            Some((&RefProj::ByteOffset(disp), head)) if disp >= 0 => {
                (idx + disp as usize, head)
            }
            _ => (idx, head),
        };
        if let Some(byte) = value_as_storage_byte(&value) {
            if let Ok(head_slot) = navigate_mut(slot, head, func, place) {
                if splice_scalar_storage_byte(head_slot, idx, byte) {
                    return Ok(());
                }
            }
        }
    }
    let target = navigate_mut(slot, projection, func, place)?;
    *target = value;
    Ok(())
}

/// Project through `projection` by reference: the borrowing form
/// of [`project_value`]. Steps that only navigate (field, index
/// into an aggregate, downcast) lend the sub-value out of the
/// input; steps that construct a fresh value (byte/string indexing,
/// sub-slices) materialise it and finish the remaining steps with
/// the owning projection. Errors are exactly those of
/// `project_value`.
pub(crate) fn project_value_ref<'a>(
    value: &'a Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<Cow<'a, Value>> {
    let composed;
    let projection = if projection_needs_composition(projection) {
        composed = compose_projection(value, projection, func, place);
        &composed[..]
    } else {
        projection
    };
    let mut cur: &'a Value = value;
    let mut rest = projection;
    // The state a coroutine downcast named, for the field step
    // that follows it (see the `Downcast` arm below).
    let mut downcast_state: Option<u32> = None;
    while let Some((step, tail)) = rest.split_first() {
        if *cur == Value::Uninit {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        }
        rest = tail;
        let step_state = downcast_state.take();
        match step {
            RefProj::Field(idx) => {
                cur = match cur {
                    Value::Tuple(items) => items.get(*idx).ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?,
                    Value::Enum { fields, .. } => fields.get(*idx).ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?,
                    Value::Coroutine { fields, saved, .. } => {
                        // As in `project_value`: a downcast routes
                        // the field step to that state's saved
                        // storage, otherwise (and for state 0, the
                        // unresumed state) the upvar storage.
                        let storage = match step_state {
                            Some(0) | None => Some(fields),
                            Some(state) => saved.get(state as usize),
                        };
                        storage
                            .and_then(|fields| fields.get(*idx))
                            .ok_or_else(|| InterpError::UninitializedRead {
                                func: func.to_string(),
                                place: place.to_string(),
                            })?
                    }
                    Value::Closure { env, .. } => env.get(*idx).ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?,
                    Value::DynRef { .. } if *idx == 0 => cur,
                    // The byte-string wrapper chain, as in
                    // `project_value`: field 0 of the transparent
                    // `Path`/`OsStr`/`CStr` wrappers over a string
                    // value is the string itself.
                    Value::Str(_) | Value::Bytes(_) if *idx == 0 => cur,
                    Value::FmtArguments(_) => {
                        // The modelled `fmt::Arguments` fields are
                        // synthesized values, not storage (see the
                        // matching arm in `project_value`), so the
                        // projection materializes an owned field and
                        // any remaining steps project into it.
                        let owned = project_value(
                            cur.clone(),
                            std::slice::from_ref(step),
                            func,
                            place,
                        )?;
                        if tail.is_empty() {
                            return Ok(Cow::Owned(owned));
                        }
                        return Ok(Cow::Owned(
                            project_value_ref(&owned, tail, func, place)?.into_owned(),
                        ));
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("field projection on {place}"),
                            "tuple or ADT value",
                            other.kind_name(),
                        ))
                    }
                };
            }
            RefProj::Index(idx) => match cur {
                Value::FatRef { base, len } => {
                    if *idx as u128 >= *len {
                        return Err(index_oob(*len as usize, *idx));
                    }
                    match project_value_ref(base, std::slice::from_ref(step), func, place)? {
                        Cow::Borrowed(v) => cur = v,
                        Cow::Owned(v) => {
                            return Ok(Cow::Owned(project_value(v, rest, func, place)?));
                        }
                    }
                }
                Value::Tuple(items) => {
                    cur = items.get(*idx).ok_or_else(|| index_oob(items.len(), *idx))?;
                }
                Value::Enum { fields, .. }
                    if fields.len() == 1
                        && matches!(
                            fields[0],
                            Value::Tuple(_) | Value::Bytes(_)
                        ) =>
                {
                    // The single-array-field pun, as in
                    // `project_value`: the index step continues
                    // into the field's elements.
                    match project_value_ref(
                        &fields[0],
                        std::slice::from_ref(step),
                        func,
                        place,
                    )? {
                        Cow::Borrowed(v) => cur = v,
                        Cow::Owned(v) => {
                            return Ok(Cow::Owned(project_value(
                                v, rest, func, place,
                            )?));
                        }
                    }
                }
                Value::Enum { fields, .. } => {
                    cur = fields.get(*idx).ok_or_else(|| index_oob(fields.len(), *idx))?;
                }
                Value::Bytes(b) => {
                    let byte = b.get(*idx).ok_or_else(|| index_oob(b.len(), *idx))?;
                    let v = Value::Uint(*byte as u128);
                    return Ok(Cow::Owned(project_value(v, rest, func, place)?));
                }
                Value::Str(text) => {
                    let byte = text
                        .as_bytes()
                        .get(*idx)
                        .ok_or_else(|| index_oob(text.len(), *idx))?;
                    let v = Value::Uint(*byte as u128);
                    return Ok(Cow::Owned(project_value(v, rest, func, place)?));
                }
                Value::Bool(_)
                | Value::Int(_)
                | Value::Uint(_)
                | Value::Float(_)
                | Value::Float128(_) => {
                    // A scalar's storage byte, as in
                    // `project_value` (see `scalar_storage_byte`).
                    let len = scalar_storage_len(cur).expect("scalar storage");
                    if *idx >= len {
                        return Err(index_oob(len, *idx));
                    }
                    let byte =
                        scalar_storage_byte(cur, *idx).expect("index inside the storage");
                    let v = Value::Uint(u128::from(byte));
                    return Ok(Cow::Owned(project_value(v, rest, func, place)?));
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("index projection on {place}"),
                        "array, byte or string value",
                        other.kind_name(),
                    ))
                }
            },
            RefProj::Subslice { offset, len } => {
                let (start, end) = (*offset, *offset + *len);
                let v = match cur {
                    Value::Tuple(items) => {
                        check_subslice(items.len(), start, *len)?;
                        Value::Tuple(items[start..end].to_vec())
                    }
                    Value::Bytes(b) => {
                        check_subslice(b.len(), start, *len)?;
                        Value::Bytes(b[start..end].to_vec())
                    }
                    Value::Str(text) => match text.get(start..end) {
                        Some(sub) => Value::Str(sub.to_string()),
                        None => {
                            return Err(InterpError::panic(format!(
                                "byte range {start}..{end} is not a valid slice of the string at {place}"
                            )))
                        }
                    },
                    Value::Enum { variant, fields } => {
                        check_subslice(fields.len(), start, *len)?;
                        Value::Enum {
                            variant: *variant,
                            fields: fields[start..end].to_vec(),
                        }
                    }
                    Value::FatRef {
                        base,
                        len: fat_len,
                    } => {
                        if start == 0 {
                            if end as u128 > *fat_len {
                                return Err(InterpError::panic(format!(
                                    "range end index {end} out of range for slice of length {fat_len}"
                                )));
                            }
                            Value::FatRef {
                                base: base.clone(),
                                len: end as u128,
                            }
                        } else {
                            match &**base {
                                Value::Tuple(_) | Value::Bytes(_) | Value::Str(_) => {
                                    project_value_ref(
                                        base,
                                        std::slice::from_ref(step),
                                        func,
                                        place,
                                    )?
                                    .into_owned()
                                }
                                other => {
                                    return Err(InterpError::type_mismatch(
                                        format!("subslice projection on {place}"),
                                        "materialised slice value",
                                        other.kind_name(),
                                    ))
                                }
                            }
                        }
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("subslice projection on {place}"),
                            "array, slice or string value",
                            other.kind_name(),
                        ))
                    }
                };
                return Ok(Cow::Owned(project_value(v, rest, func, place)?));
            }
            RefProj::Downcast(variant) => match cur {
                Value::Coroutine { .. } => {
                    downcast_state = Some(*variant);
                }
                Value::Enum {
                    variant: current, ..
                } => {
                    if current != variant {
                        return Err(InterpError::type_mismatch(
                            format!("downcast of {place}"),
                            format!("variant {variant}"),
                            format!("variant {current}"),
                        ));
                    }
                }
                Value::Tuple(_) if *variant == 0 => {}
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("downcast of {place}"),
                        "ADT value",
                        other.kind_name(),
                    ))
                }
            },
            RefProj::UnionField { .. } => {
                // Same contract as `project_value`: union field
                // reads need the program's scalar shapes and route
                // through the typed projector (`project_value_t`).
                return Err(InterpError::type_mismatch(
                    format!("union field projection on {place}"),
                    "a union field read through the typed projection path",
                    "the untyped value projector",
                ));
            }
            RefProj::ByteOffset(disp) => {
                if *disp != 0 {
                    // A byte read through a pointer displaced into
                    // a scalar's storage, as in `project_value`:
                    // the index step that follows selects the byte
                    // at the combined offset.
                    if *disp >= 0 {
                        if let (Some(len), Some(&RefProj::Index(k))) =
                            (scalar_storage_len(cur), rest.first())
                        {
                            let at = *disp as usize + k;
                            let byte = scalar_storage_byte(cur, at)
                                .ok_or_else(|| index_oob(len, at))?;
                            let v = Value::Uint(u128::from(byte));
                            return Ok(Cow::Owned(project_value(v, &rest[1..], func, place)?));
                        }
                    }
                    // The position does not coincide with a value
                    // boundary, and the model has no byte-level
                    // view of a value's storage to reinterpret.
                    return Err(InterpError::UnsupportedProjection {
                        kind: format!(
                            "read through a pointer displaced {disp} bytes from a value boundary"
                        ),
                    });
                }
            }
        }
    }
    Ok(Cow::Borrowed(cur))
}

/// Navigate a projection mutably inside a stored value and return the
/// final slot (write direction).
pub(crate) fn navigate_mut<'x>(
    slot: &'x mut Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<&'x mut Value> {
    if projection.is_empty() {
        return Ok(slot);
    }
    let composed;
    let projection = if projection_needs_composition(projection) {
        composed = compose_projection(slot, projection, func, place);
        &composed[..]
    } else {
        projection
    };
    if *slot == Value::Uninit {
        // Partial initialization: MIR fills an aggregate place
        // field by field — a `catch_unwind` Data union written
        // through a raw pointer by the callee, a struct local
        // stored field-wise — so a field, union-field or downcast
        // write into a never-written slot materializes the
        // aggregate with every other slot still uninitialized, the
        // same sparse shape a union aggregate value has. The slots
        // the write does not touch keep raising
        // `UninitializedRead` when read, so no value is fabricated.
        // Any other step into an uninitialized place (an index
        // into an array that was never built) has no honest shape
        // to materialize and stays an error.
        match projection.first() {
            Some(RefProj::Field(idx)) | Some(RefProj::UnionField { idx, .. }) => {
                *slot = Value::Enum {
                    variant: 0,
                    fields: vec![Value::Uninit; *idx + 1],
                };
            }
            Some(RefProj::Downcast(variant)) => {
                *slot = Value::Enum {
                    variant: *variant,
                    fields: Vec::new(),
                };
            }
            _ => {
                return Err(InterpError::UninitializedRead {
                    func: func.to_string(),
                    place: place.to_string(),
                });
            }
        }
    }
    let (step, rest) = projection.split_first().expect("non-empty");
    match step {
        RefProj::UnionField { idx, .. } => {
            // A union holds one field's value: writing field
            // `idx` makes it the held field, so every other slot
            // is cleared (a later read of another field then
            // reinterprets this field's bits; see
            // `union_field_read`). Writing *inside* the field
            // (`u.a.0 = ...`) navigates into the held value as
            // usual, but a field that holds no value has nothing
            // to write inside of: the bytes a native write would
            // preserve there belong to a field this model does
            // not hold, so that stays an uninitialized error
            // rather than a fabricated partial value.
            let Value::Enum { fields, .. } = slot else {
                return Err(InterpError::type_mismatch(
                    format!("union field write on {place}"),
                    "union (ADT) value",
                    slot.kind_name(),
                ));
            };
            if fields.len() <= *idx {
                fields.resize(*idx + 1, Value::Uninit);
            }
            if !rest.is_empty() && fields[*idx] == Value::Uninit {
                return Err(InterpError::UninitializedRead {
                    func: func.to_string(),
                    place: place.to_string(),
                });
            }
            for (i, field) in fields.iter_mut().enumerate() {
                if i != *idx {
                    *field = Value::Uninit;
                }
            }
            navigate_mut(&mut fields[*idx], rest, func, place)
        }
        RefProj::ByteOffset(disp) => {
            if *disp == 0 {
                navigate_mut(slot, rest, func, place)
            } else {
                Err(InterpError::UnsupportedProjection {
                    kind: format!(
                        "write through a pointer displaced {disp} bytes from a value boundary"
                    ),
                })
            }
        }
        RefProj::Field(idx) => {
            if *idx == 0 && matches!(slot, Value::Str(_) | Value::Bytes(_)) {
                // The byte-string wrapper chain, as in
                // `project_value`: a write to field 0 of the
                // transparent `Path`/`OsStr` wrappers over a
                // string value lands on the string itself.
                return navigate_mut(slot, rest, func, place);
            }
            if let Value::Coroutine { fields, .. } = slot {
                // The state storage grows to the written slot: a
                // state field stored for the first time extends the
                // storage, and slots skipped over are uninitialized,
                // like a field that has not been stored yet.
                if fields.len() <= *idx {
                    fields.resize(*idx + 1, Value::Uninit);
                }
                return navigate_mut(&mut fields[*idx], rest, func, place);
            }
            // The field vector of the value in the slot may be
            // sparse or invalidated: a union value covers only
            // the fields written so far (see the union arm of
            // aggregate evaluation), and an enum place being
            // (re)initialized in place has no live payload slots
            // for the variant being written (SetDiscriminant
            // invalidated the old payload; a downcast write to a
            // different variant rematerialized it empty). Grow
            // the vector with Uninit padding up to the written
            // index; slots never written keep reading as
            // uninitialized.
            if let Value::Enum { fields, .. } = slot {
                if fields.len() <= *idx {
                    fields.resize(*idx + 1, Value::Uninit);
                }
                return navigate_mut(&mut fields[*idx], rest, func, place);
            }
            let items = match slot {
                Value::Tuple(items) => items,
                Value::Closure { env, .. } => env,
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("field write on {place}"),
                        "tuple or ADT value",
                        other.kind_name(),
                    ));
                }
            };
            let item = items.get_mut(*idx).ok_or_else(|| InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            })?;
            navigate_mut(item, rest, func, place)
        }
        RefProj::Index(idx) => match slot {
            Value::Tuple(items) => {
                if *idx >= items.len() {
                    return Err(index_oob(items.len(), *idx));
                }
                navigate_mut(&mut items[*idx], rest, func, place)
            }
            // A `Vec` value is the aggregate of its elements:
            // indexing the vector's own place writes the element
            // field (the read direction in `project_value`).
            Value::Enum { fields, .. } => {
                if fields.len() == 1
                    && matches!(
                        fields[0],
                        Value::Tuple(_) | Value::Bytes(_)
                    )
                {
                    // The single-array-field pun, as in
                    // `project_value`: the write continues into
                    // the field's elements.
                    let mut full = Vec::with_capacity(rest.len() + 1);
                    full.push(RefProj::Index(*idx));
                    full.extend_from_slice(rest);
                    navigate_mut(&mut fields[0], &full, func, place)
                } else if *idx >= fields.len() {
                    return Err(index_oob(fields.len(), *idx));
                } else {
                    navigate_mut(&mut fields[*idx], rest, func, place)
                }
            }
            Value::Bytes(b) => {
                if *idx >= b.len() {
                    return Err(index_oob(b.len(), *idx));
                }
                // Byte payloads come from constants and have no
                // `Value` slot to hand out; array locals built by
                // aggregates are tuples and take the arm above.
                Err(InterpError::type_mismatch(
                    format!("index write on {place}"),
                    "mutable array slot",
                    "constant bytes",
                ))
            }
            other => Err(InterpError::type_mismatch(
                format!("index write on {place}"),
                "array value",
                other.kind_name(),
            )),
        },
        RefProj::Subslice { offset, len } => {
            // A sub-slice has no single slot to hand out: the
            // navigation continues into one of its elements (an
            // `Index` step composes the offset into the base) or
            // into a nested sub-slice (the offsets compose). A
            // write to the whole sub-slice would have to splice
            // the base's element sequence in place; no MIR
            // assignment targets an unsized place that way, so it
            // stays an explicit error rather than a silent
            // partial write.
            match rest.first() {
                Some(RefProj::Index(j)) => {
                    let j = *j;
                    if j >= *len {
                        return Err(index_oob(*len, j));
                    }
                    let target = *offset + j;
                    let rest = &rest[1..];
                    match slot {
                        Value::Tuple(items) => {
                            if target >= items.len() {
                                return Err(index_oob(items.len(), target));
                            }
                            navigate_mut(&mut items[target], rest, func, place)
                        }
                        Value::Enum { fields, .. } => {
                            if target >= fields.len() {
                                return Err(index_oob(fields.len(), target));
                            }
                            navigate_mut(&mut fields[target], rest, func, place)
                        }
                        other => Err(InterpError::type_mismatch(
                            format!("subslice element write on {place}"),
                            "array value",
                            other.kind_name(),
                        )),
                    }
                }
                Some(RefProj::Subslice {
                    offset: inner_offset,
                    len: inner_len,
                }) => {
                    if inner_offset + inner_len > *len {
                        return Err(InterpError::panic(format!(
                            "range end index {} out of range for slice of length {len}",
                            inner_offset + inner_len
                        )));
                    }
                    let mut composed = vec![RefProj::Subslice {
                        offset: offset + inner_offset,
                        len: *inner_len,
                    }];
                    composed.extend_from_slice(&rest[1..]);
                    navigate_mut(slot, &composed, func, place)
                }
                _ => Err(InterpError::type_mismatch(
                    format!("write through sub-slice on {place}"),
                    "an element or nested sub-slice of the sub-slice",
                    "the whole sub-slice",
                )),
            }
        }
        RefProj::Downcast(variant) => match slot {
            Value::Coroutine { .. }
                if matches!(rest.first(), Some(RefProj::Field(_))) =>
            {
                // A coroutine's field writes go through the downcast
                // of the state being entered, whatever the current
                // tag is, and land in that state's saved storage -
                // not in the upvar storage a field step without a
                // downcast addresses (see `Value::Coroutine`); the
                // unresumed state's downcast is the one exception,
                // handled below. The state storage grows to the
                // written slot, like the upvar storage in the
                // `Field` arm above.
                let (idx, tail) = match rest.split_first() {
                    Some((RefProj::Field(idx), tail)) => (*idx, tail),
                    _ => unreachable!(),
                };
                let Value::Coroutine { fields, saved, .. } = slot else {
                    unreachable!()
                };
                // State 0 is the unresumed state: its fields are
                // the upvars, so the write lands in the upvar
                // storage, matching the read paths.
                let state = if *variant == 0 {
                    &mut *fields
                } else {
                    if saved.len() <= *variant as usize {
                        saved.resize(*variant as usize + 1, Vec::new());
                    }
                    &mut saved[*variant as usize]
                };
                if state.len() <= idx {
                    state.resize(idx + 1, Value::Uninit);
                }
                navigate_mut(&mut state[idx], tail, func, place)
            }
            // A downcast with no field step after it (a whole-state
            // place) denotes the coroutine value itself.
            Value::Coroutine { .. } => navigate_mut(slot, rest, func, place),
            Value::Enum {
                variant: current,
                fields,
            } => {
                if current != variant {
                    // A mutable navigation through the downcast of
                    // a variant other than the one the tag names is
                    // an in-place reinitialization: the old variant's
                    // payload is invalidated and the new variant's
                    // storage is rematerialized empty (the Field arm
                    // pads it with Uninit slots as fields are
                    // written). The tag switches here so that the
                    // closing SetDiscriminant, when the MIR emits
                    // one after the stores, names the held variant
                    // and leaves the stored fields in place.
                    *current = *variant;
                    fields.clear();
                }
                navigate_mut(slot, rest, func, place)
            }
            Value::Tuple(_) if *variant == 0 => navigate_mut(slot, rest, func, place),
            other => Err(InterpError::type_mismatch(
                format!("downcast write on {place}"),
                "ADT value",
                other.kind_name(),
            )),
        },
    }
}

// ----- arithmetic helpers ----------------------------------------------------

fn wrap_signed(v: i128, bits: u32) -> i128 {
    if bits >= 128 {
        return v;
    }
    let shift = 128 - bits;
    (v << shift) >> shift
}

/// The mask of the low `bits` bits (all bits for 128).
fn bit_mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

fn wrap_unsigned(v: u128, bits: u32) -> u128 {
    if bits >= 128 {
        v
    } else {
        v & ((1u128 << bits) - 1)
    }
}

/// The value `v` read under a binary operation's declared
/// operand type: an integer stored under the other kind tag is
/// reinterpreted at the declared width, exactly as an assignment
/// to a place of that type reinterprets it (see
/// `normalize_for_ty`). A checked operation's destination is the
/// `(T, bool)` tuple; the operand type lives in its first
/// element.
fn reinterpret_int(v: &Value, dest_ty: Option<&Ty>) -> Value {
    let operand_ty = match dest_ty {
        Some(Ty::Tuple(ts)) => ts.first(),
        other => other,
    };
    match (v, operand_ty) {
        (Value::Uint(u), Some(Ty::Int(it))) => Value::Int(wrap_signed(*u as i128, it.bits())),
        (Value::Int(i), Some(Ty::Uint(ut))) => Value::Uint((*i as u128) & ut.max()),
        _ => v.clone(),
    }
}

fn signed_bits(dest_ty: Option<&Ty>) -> u32 {
    match dest_ty {
        Some(Ty::Int(it)) => it.bits(),
        // A checked operation's destination is the `(T, bool)` tuple;
        // the operand width lives in its first element.
        Some(Ty::Tuple(ts)) => signed_bits(ts.first()),
        _ => 128,
    }
}

fn unsigned_bits(dest_ty: Option<&Ty>) -> u32 {
    match dest_ty {
        Some(Ty::Uint(ut)) => ut.bits(),
        Some(Ty::Tuple(ts)) => unsigned_bits(ts.first()),
        _ => 128,
    }
}

/// Evaluate a binary operation.
/// Returns `(result, overflowed)`. Overflow is reported only through
/// the checked form's flag: MIR's plain `BinaryOp` wraps on overflow
/// (the overflow panic of a checked build is the following `Assert`
/// terminator's business, driven by that flag), so the plain form
/// yields the wrapped result and never panics for overflow. Division
/// by zero panics in both forms — the operation itself requires it,
/// independent of overflow checks. Comparison and bitwise-on-bool
/// operations never overflow.
/// The `core::cmp::Ordering` value of a three-way comparison:
/// the enum in declaration order (Less, Equal, Greater), the
/// shape `Builtin::OrdCmp`'s `ordering_value` also produces.
fn ordering_of(ord: std::cmp::Ordering) -> Value {
    Value::Enum {
        variant: match ord {
            std::cmp::Ordering::Less => 0,
            std::cmp::Ordering::Equal => 1,
            std::cmp::Ordering::Greater => 2,
        },
        fields: Vec::new(),
    }
}

pub(crate) fn eval_binary(
    op: BinOp,
    lhs: &Value,
    rhs: &Value,
    dest_ty: Option<&Ty>,
    checked_form: bool,
) -> InterpResult<(Value, bool)> {
    use BinOp::*;
    // Boolean operations.
    if let (Value::Bool(a), Value::Bool(b)) = (lhs, rhs) {
        let v = match op {
            BitAnd => Value::Bool(*a && *b),
            BitOr => Value::Bool(*a || *b),
            BitXor => Value::Bool(*a ^ *b),
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Cmp => ordering_of(a.cmp(b)),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "bool",
                ))
            }
        };
        return Ok((v, false));
    }
    // String equality (the only binary operations MIR performs on
    // string data directly; ordering goes through library calls).
    if let (Value::Str(a), Value::Str(b)) = (lhs, rhs) {
        let v = match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "string",
                ))
            }
        };
        return Ok((v, false));
    }
    // Shifts: in MIR the amount of a `Shl` / `Shr` may have ANY
    // integer type, independent of the value's type, so the
    // same-kind pair dispatch below cannot see mixed pairs. Integer
    // shift pairs are evaluated by `eval_shift`, which reads the
    // amount as a `u32` and keeps the value's own type and
    // signedness.
    if matches!(op, Shl | Shr)
        && matches!(lhs, Value::Int(_) | Value::Uint(_))
        && matches!(rhs, Value::Int(_) | Value::Uint(_))
    {
        return eval_shift(op, lhs, rhs, dest_ty, checked_form);
    }

    // An integer operand whose kind tag differs from the
    // operation's declared operand type: reinterpret its bits at
    // the declared width before the strict kind dispatch below. A
    // value stored inside an aggregate keeps the kind it was
    // built with - a constant's bits are stored unsigned - and a
    // direct place operand reads it without an intervening typed
    // place that would reinterpret it, so the declared type is
    // the only exact reading (the same rule `normalize_for_ty`
    // applies at every place write).
    let lhs = &reinterpret_int(lhs, dest_ty);
    let rhs = &reinterpret_int(rhs, dest_ty);
    // Signed arithmetic.
    if let (Value::Int(a), Value::Int(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let bits = signed_bits(dest_ty);
        let (min, max) = if bits >= 128 {
            (i128::MIN, i128::MAX)
        } else {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        };
        let cmp = |f: &dyn Fn() -> bool| Ok((Value::Bool(f()), false));
        let wide: Option<i128> = match op {
            Cmp => return Ok((ordering_of(a.cmp(&b)), false)),
            Eq => return cmp(&|| a == b),
            Ne => return cmp(&|| a != b),
            Lt => return cmp(&|| a < b),
            Le => return cmp(&|| a <= b),
            Gt => return cmp(&|| a > b),
            Ge => return cmp(&|| a >= b),
            Add => a
                .checked_add(b)
                .or(Some(wrap_signed(a.wrapping_add(b), bits))),
            Sub => a
                .checked_sub(b)
                .or(Some(wrap_signed(a.wrapping_sub(b), bits))),
            Mul => a
                .checked_mul(b)
                .or(Some(wrap_signed(a.wrapping_mul(b), bits))),
            Div | Rem => {
                if b == 0 {
                    return Err(InterpError::panic(if op == Div {
                        "attempt to divide by zero"
                    } else {
                        "attempt to calculate the remainder with a divisor of zero"
                    }));
                }
                if a == min && b == -1 {
                    // Overflow: the value wraps in both forms (Div
                    // yields `min`, Rem yields 0); the checked form
                    // also reports it in the flag below.
                    Some(if op == Div { a } else { 0 })
                } else if op == Div {
                    Some(a / b)
                } else {
                    Some(a % b)
                }
            }
            BitAnd => Some(a & b),
            BitOr => Some(a | b),
            BitXor => Some(a ^ b),
            Shl | Shr => unreachable!("shifts are evaluated by `eval_shift`"),
        };
        // Invariant: comparisons returned early above and every other
        // arm of the match yields `Some`, so `wide` is always present.
        let raw = wide.expect("all signed ops produce a value");
        // Determine overflow by recomputing in the declared width: a result
        // is overflowing if the mathematical result leaves [min, max].
        // Division overflow (MIN / -1) is covered because raw == MIN while
        // the true quotient is MAX+1; detect it explicitly.
        let overflowed = match op {
            Add => a.checked_add(b).map_or(true, |r| r < min || r > max),
            Sub => a.checked_sub(b).map_or(true, |r| r < min || r > max),
            Mul => a.checked_mul(b).map_or(true, |r| r < min || r > max),
            Div | Rem => a == min && b == -1,
            Shl => unreachable!("shifts are evaluated by `eval_shift`"),
            _ => false,
        };
        let result = Value::Int(wrap_signed(raw, bits));
        return Ok((result, overflowed));
    }
    // Unsigned arithmetic.
    if let (Value::Uint(a), Value::Uint(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let bits = unsigned_bits(dest_ty);
        let max = if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        let cmp = |f: &dyn Fn() -> bool| Ok((Value::Bool(f()), false));
        match op {
            Cmp => return Ok((ordering_of(a.cmp(&b)), false)),
            Eq => return cmp(&|| a == b),
            Ne => return cmp(&|| a != b),
            Lt => return cmp(&|| a < b),
            Le => return cmp(&|| a <= b),
            Gt => return cmp(&|| a > b),
            Ge => return cmp(&|| a >= b),
            _ => {}
        }
        if (op == Div || op == Rem) && b == 0 {
            return Err(InterpError::panic(if op == Div {
                "attempt to divide by zero"
            } else {
                "attempt to calculate the remainder with a divisor of zero"
            }));
        }
        // Comparisons returned above and shifts are evaluated by
        // `eval_shift` before this dispatch; the operators that
        // can reach this match are exactly the eight covered below,
        // which is what makes the closing `unreachable!` an internal
        // invariant rather than an input-dependent failure.
        let (raw, overflowed) = match op {
            Add => {
                let r = a.wrapping_add(b);
                (r, a.checked_add(b).map_or(true, |v| v > max))
            }
            Sub => (a.wrapping_sub(b), a < b),
            Mul => {
                let r = a.wrapping_mul(b);
                (r, a.checked_mul(b).map_or(true, |v| v > max))
            }
            Div => (a / b, false),
            Rem => (a % b, false),
            BitAnd => (a & b, false),
            BitOr => (a | b, false),
            BitXor => (a ^ b, false),
            _ => unreachable!("comparisons and shifts handled above"),
        };
        let result = Value::Uint(wrap_unsigned(raw, bits));
        return Ok((result, overflowed));
    }
    // `f128` arithmetic: exact software binary128 (see
    // `crate::f128`); never rounded through `f64`.
    if let (Value::Float128(a), Value::Float128(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let v = match op {
            Eq => Value::Bool(f128::eq(a, b)),
            Ne => Value::Bool(!f128::eq(a, b)),
            Lt => Value::Bool(f128::partial_cmp(a, b) == Some(std::cmp::Ordering::Less)),
            Le => Value::Bool(matches!(
                f128::partial_cmp(a, b),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            )),
            Gt => Value::Bool(f128::partial_cmp(a, b) == Some(std::cmp::Ordering::Greater)),
            Ge => Value::Bool(matches!(
                f128::partial_cmp(a, b),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            )),
            Add => Value::Float128(f128::add(a, b)),
            Sub => Value::Float128(f128::sub(a, b)),
            Mul => Value::Float128(f128::mul(a, b)),
            Div => Value::Float128(f128::div(a, b)),
            Rem => Value::Float128(f128::rem(a, b)),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "f128",
                ))
            }
        };
        return Ok((v, false));
    }
    // Floating-point arithmetic, computed in `f64` (see `Value::Float`
    // for why that is exact at `f32` destinations too). Division by
    // zero and NaN follow IEEE 754, as in Rust: they are values, not
    // panics. Float operations never set the overflow flag.
    if let (Value::Float(a), Value::Float(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let round = |r: f64| match float_ty_of(dest_ty) {
            Some(FloatTy::F16) => round_to_f16(r),
            Some(FloatTy::F32) => (r as f32) as f64,
            _ => r,
        };
        let v = match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            Add => Value::Float(round(a + b)),
            Sub => Value::Float(round(a - b)),
            Mul => Value::Float(round(a * b)),
            Div => Value::Float(round(a / b)),
            Rem => Value::Float(round(a % b)),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "float",
                ))
            }
        };
        return Ok((v, false));
    }
    // Pointer operations: MIR compares raw pointers and function
    // pointers directly. Equality is address identity in the
    // synthetic-address model (the same identity `PtrToInt`
    // exposes); a fat pointer's equality also compares the
    // metadata word (its length). Ordering is defined on the
    // synthetic addresses, which is deterministic within a run.
    if matches!(op, Eq | Ne | Lt | Le | Gt | Ge | Cmp) {
        let key = |v: &Value| -> Option<(u128, Option<u128>)> {
            match v {
                // The null pointer's address is 0, matching an
                // address-only pointer holding address 0 (the
                // synthetic range keeps `PtrToInt` of null itself
                // non-zero, a model choice the comparison relation
                // does not inherit).
                Value::Ref(RefTarget::Null) => Some((0, None)),
                Value::FatRef { len, .. } => {
                    synthetic_address(v).map(|a| (a, Some(*len)))
                }
                Value::Ref(_) | Value::FnPtr(_) | Value::DynRef { .. } => {
                    synthetic_address(v).map(|a| (a, None))
                }
                // A non-capturing closure value is a function
                // pointer; it compares by its body identity.
                Value::Closure { .. } => synthetic_address(v).map(|a| (a, None)),
                _ => None,
            }
        };
        if let (Some(a), Some(b)) = (key(lhs), key(rhs)) {
            let ord = a.cmp(&b);
            if op == Cmp {
                return Ok((ordering_of(ord), false));
            }
            let v = match op {
                Eq => a == b,
                Ne => a != b,
                Lt => ord == std::cmp::Ordering::Less,
                Le => ord != std::cmp::Ordering::Greater,
                Gt => ord == std::cmp::Ordering::Greater,
                Ge => ord != std::cmp::Ordering::Less,
                _ => unreachable!(),
            };
            return Ok((Value::Bool(v), false));
        }
    }
        Err(InterpError::type_mismatch(
        format!("binary operation {op:?}"),
        "matching integer/bool operands",
        format!("{} and {}", lhs.kind_name(), rhs.kind_name()),
    ))
}

/// Evaluate a `Shl` / `Shr` operation.
///
/// The value (left operand) keeps its own type and signedness: the
/// declared width comes from the destination type exactly as in the
/// same-kind paths of `eval_binary`, a signed value shifts
/// arithmetically and an unsigned value logically. The amount
/// (right operand) may have any integer type; only its numeric
/// value matters. It is read as a `u32`, as rustc interprets shift
/// amounts: a negative amount or one above `u32::MAX` does not fit,
/// and `u32::MAX` stands in for it, which is at or above every
/// type width and so reports an overflowing amount either way.
///
/// An amount at or above the value's width is the only overflow a
/// shift has: both forms yield the wrapped result (the amount
/// taken modulo the width, the `wrapping_shl` / `wrapping_shr`
/// reading); the checked form also sets the overflow flag, and
/// the `Assert` that follows it in a checked build is what panics
/// (with Rust 1.100.0's native message, synthesized at conversion
/// time). Bits shifted out of the declared width are simply lost,
/// as natively: they set neither the overflow flag nor a panic,
/// in either form.
fn eval_shift(
    op: BinOp,
    lhs: &Value,
    rhs: &Value,
    dest_ty: Option<&Ty>,
    checked_form: bool,
) -> InterpResult<(Value, bool)> {
    let shift = match rhs {
        Value::Int(b) => u32::try_from(*b).unwrap_or(u32::MAX),
        Value::Uint(b) => u32::try_from(*b).unwrap_or(u32::MAX),
        _ => unreachable!("shift amounts are integer values"),
    };
    match lhs {
        Value::Int(a) => {
            let a = *a;
            let bits = signed_bits(dest_ty);
            if shift >= bits {
                let wrapped = if op == BinOp::Shl {
                    wrap_signed(a.wrapping_shl(shift % bits), bits)
                } else {
                    wrap_signed(a.wrapping_shr(shift % bits), bits)
                };
                return Ok((Value::Int(wrapped), checked_form));
            }
            let raw = if op == BinOp::Shl {
                wrap_signed(a << shift, bits)
            } else {
                a >> shift
            };
            Ok((Value::Int(wrap_signed(raw, bits)), false))
        }
        Value::Uint(a) => {
            let a = *a;
            let bits = unsigned_bits(dest_ty);
            if shift >= bits {
                let wrapped = if op == BinOp::Shl {
                    wrap_unsigned(a.wrapping_shl(shift % bits), bits)
                } else {
                    wrap_unsigned(a.wrapping_shr(shift % bits), bits)
                };
                return Ok((Value::Uint(wrapped), checked_form));
            }
            let raw = if op == BinOp::Shl {
                a << shift
            } else {
                a >> shift
            };
            Ok((Value::Uint(wrap_unsigned(raw, bits)), false))
        }
        _ => unreachable!("shift values are integer values"),
    }
}

/// The float width a binary operation's destination declares, if any.
/// A checked operation's destination is the `(T, bool)` tuple; the
/// operand width lives in its first element.
/// Round an `f64` value to `f16` precision (round-to-nearest-even,
/// as IEEE 754 requires) and widen it back: the stored form of an
/// `f16` place's value. Computing in `f64` first is exact for
/// `f16` operands by the same double-rounding argument as `f32`
/// (53 mantissa bits >= 2*11 + 2).
pub(crate) fn round_to_f16(v: f64) -> f64 {
    crate::mir_convert::f16_to_f64(f64_to_f16_bits(v))
}

/// The `f16` bit pattern nearest to `v` (round-to-nearest-even).
fn f64_to_f16_bits(v: f64) -> u16 {
    let bits = v.to_bits();
    let sign = (((bits >> 63) & 1) as u16) << 15;
    let exp = ((bits >> 52) & 0x7ff) as i64;
    let mant = bits & ((1u64 << 52) - 1);
    if exp == 0x7ff {
        // Infinity / NaN (a NaN keeps a nonzero payload marker).
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 1023;
    if e > 15 {
        return sign | 0x7c00;
    }
    if e >= -14 {
        // Normal f16: keep the top 10 mantissa bits.
        let mut kept = (mant >> 42) as u32;
        let dropped = mant & ((1u64 << 42) - 1);
        let half = 1u64 << 41;
        if dropped > half || (dropped == half && kept & 1 == 1) {
            kept += 1;
        }
        let mut e = e;
        if kept == 0x400 {
            kept = 0;
            e += 1;
            if e > 15 {
                return sign | 0x7c00;
            }
        }
        return sign | (((e + 15) as u16) << 10) | kept as u16;
    }
    if e < -25 {
        // Below the tie point of the smallest subnormal: zero
        // (the sign is preserved).
        return sign;
    }
    // Subnormal: the f16 subnormal unit is 2^-24 and the value is
    // m * 2^(e-52) with m carrying the implicit leading bit.
    let m = mant | (1u64 << 52);
    let shift = (28 - e) as u32;
    let mut kept = (m >> shift) as u32;
    let dropped = m & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    if dropped > half || (dropped == half && kept & 1 == 1) {
        kept += 1;
    }
    // `kept` may reach 0x400, which is exactly the smallest
    // normal's encoding, so it is used as-is.
    sign | kept as u16
}

fn float_ty_of(dest_ty: Option<&Ty>) -> Option<FloatTy> {
    match dest_ty {
        Some(Ty::Float(ft)) => Some(*ft),
        Some(Ty::Tuple(ts)) => float_ty_of(ts.first()),
        _ => None,
    }
}

pub(crate) fn eval_unary(op: UnOp, v: &Value, dest_ty: Option<&Ty>) -> InterpResult<Value> {
    match (op, v) {
        (UnOp::Neg, Value::Int(a)) => {
            // Plain `Neg` wraps, like the binary operators: the
            // overflow panic of a checked build is the following
            // `Assert`'s business (an `OverflowNeg` message), not
            // this operation's.
            let bits = signed_bits(dest_ty);
            Ok(Value::Int(wrap_signed(a.wrapping_neg(), bits)))
        }
        (UnOp::Neg, Value::Float(a)) => Ok(Value::Float(-a)),
        (UnOp::Neg, Value::Float128(a)) => Ok(Value::Float128(f128::neg(*a))),
        (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
        (UnOp::Not, Value::Int(a)) => Ok(Value::Int(wrap_signed(!a, signed_bits(dest_ty)))),
        (UnOp::Not, Value::Uint(a)) => Ok(Value::Uint(wrap_unsigned(!a, unsigned_bits(dest_ty)))),
        (UnOp::PtrMetadata, _) => Err(InterpError::type_mismatch(
            "PtrMetadata",
            "fat pointer",
            v.kind_name(),
        )),
        _ => Err(InterpError::type_mismatch(
            format!("unary operation {op:?}"),
            "integer or bool",
            v.kind_name(),
        )),
    }
}

/// Rust float-to-int cast semantics: saturate at the target type's
/// range, NaN becomes 0. The host `as` cast saturates at `i128`;
/// clamping to the declared width finishes the job.
pub(crate) fn float_to_int(v: f64, bits: u32) -> i128 {
    let wide = v as i128;
    if bits >= 128 {
        wide
    } else {
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        wide.clamp(min, max)
    }
}

/// Rust float-to-uint cast semantics: saturate at the target type's
/// range (negative values become 0), NaN becomes 0.
pub(crate) fn float_to_uint(v: f64, bits: u32) -> u128 {
    let wide = v as u128;
    if bits >= 128 {
        wide
    } else {
        wide.min((1u128 << bits) - 1)
    }
}

/// A stand-in numeric address for a pointer value. Places in this
/// value model have no real addresses, but an address is still
/// observable through pointer-to-integer transmutes — most
/// importantly the alignment precondition checks rustc emits
/// around raw-pointer dereferences, which compute
/// `addr & (align - 1) == 0`. The synthetic address is a pure
/// function of the pointer's target: the same target always yields
/// the same address, distinct targets yield distinct addresses
/// (up to the mixing hash), and every address is a multiple of
/// 4096, so the alignment check holds for any alignment a place
/// in this model can carry. A reference to a constant aggregate
/// has no place identity; it shares one reserved address.
fn synthetic_addr(v: &Value) -> u128 {
    fn mix(h: &mut u128, x: u128) {
        // FNV-1a over the value's limbs.
        for limb in [x as u64, (x >> 64) as u64] {
            *h ^= limb as u128;
            *h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    if matches!(v, Value::Ref(RefTarget::Null)) {
        return 0;
    }
    let mut h: u128 = 0xcbf2_9ce4_8422_2325;
    match v {
        Value::Ref(RefTarget::Place {
            frame,
            local,
            projection,
        }) => {
            mix(&mut h, *frame as u128);
            mix(&mut h, *local as u128);
            for step in projection {
                let (tag, n) = match step {
                    RefProj::Field(i) => (1u128, *i as u128),
                    RefProj::UnionField { idx, table } => {
                        (5u128, ((*table as u128) << 64) | (*idx as u128))
                    }
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
                    RefProj::ByteOffset(d) => (6u128, *d as u128),
                };
                mix(&mut h, tag);
                mix(&mut h, n);
            }
        }
        Value::Ref(RefTarget::Value(_, projection)) => {
            mix(&mut h, 0x5eed_5eed_5eed_5eed);
            for step in projection {
                let (tag, n) = match step {
                    RefProj::Field(i) => (1u128, *i as u128),
                    RefProj::UnionField { idx, table } => {
                        (5u128, ((*table as u128) << 64) | (*idx as u128))
                    }
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
                    RefProj::ByteOffset(d) => (6u128, *d as u128),
                };
                mix(&mut h, tag);
                mix(&mut h, n);
            }
        }
        Value::Ref(RefTarget::Static { name, projection }) => {
            mix(&mut h, 0x57a7_1c5_7a7_1c5);
            for b in name.bytes() {
                mix(&mut h, b as u128);
            }
            for step in projection {
                let (tag, n) = match step {
                    RefProj::Field(i) => (1u128, *i as u128),
                    RefProj::UnionField { idx, table } => {
                        (5u128, ((*table as u128) << 64) | (*idx as u128))
                    }
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
                    RefProj::ByteOffset(d) => (6u128, *d as u128),
                };
                mix(&mut h, tag);
                mix(&mut h, n);
            }
        }
        Value::Ref(RefTarget::Alloc { id, projection }) => {
            // An allocation's address identity is its id: every
            // copy of the box that owns it, and every pointer
            // extracted from any of them, yields this same
            // address, while distinct allocations differ.
            mix(&mut h, 0xa110_ca7e_a110_ca7e);
            mix(&mut h, *id as u128);
            for step in projection {
                let (tag, n) = match step {
                    RefProj::Field(i) => (1u128, *i as u128),
                    RefProj::UnionField { idx, table } => {
                        (5u128, ((*table as u128) << 64) | (*idx as u128))
                    }
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
                    RefProj::ByteOffset(d) => (6u128, *d as u128),
                };
                mix(&mut h, tag);
                mix(&mut h, n);
            }
        }
        Value::FatRef { base, .. } => {
            return synthetic_addr(base);
        }
        Value::FnPtr(name) => {
            for b in name.bytes() {
                mix(&mut h, b as u128);
            }
        }
        _ => {}
    }
    const BASE: u128 = 0x1000_0000_0000;
    BASE + (h % (1 << 40)) * 4096
}

pub(crate) fn eval_cast(v: &Value, ty: &Ty) -> InterpResult<Value> {
    match ty {
        Ty::Int(it) => {
            // A float source saturates; it does not wrap like an
            // integer source.
            if let Value::Float(a) = v {
                return Ok(Value::Int(float_to_int(*a, it.bits())));
            }
            if let Value::Float128(a) = v {
                return Ok(Value::Int(f128::to_int(*a, it.bits())));
            }
            // A pointer source (a pointer-to-integer transmute)
            // yields the pointer's synthetic address.
            if matches!(v, Value::Ref(_) | Value::FatRef { .. } | Value::FnPtr(_)) {
                return Ok(Value::Int(wrap_signed(
                    synthetic_addr(v) as i128,
                    it.bits(),
                )));
            }
            let raw = match v {
                Value::Int(a) => *a,
                Value::Uint(a) => *a as i128,
                Value::Bool(b) => *b as i128,
                other => {
                    return Err(InterpError::type_mismatch(
                        "cast",
                        "numeric source",
                        other.kind_name(),
                    ))
                }
            };
            Ok(Value::Int(wrap_signed(raw, it.bits())))
        }
        Ty::Uint(ut) => {
            if let Value::Float(a) = v {
                return Ok(Value::Uint(float_to_uint(*a, ut.bits())));
            }
            if let Value::Float128(a) = v {
                return Ok(Value::Uint(f128::to_uint(*a, ut.bits())));
            }
            // A pointer source (a pointer-to-integer transmute)
            // yields the pointer's synthetic address.
            if matches!(v, Value::Ref(_) | Value::FatRef { .. } | Value::FnPtr(_)) {
                return Ok(Value::Uint(wrap_unsigned(synthetic_addr(v), ut.bits())));
            }
            let raw = match v {
                Value::Int(a) => *a as u128,
                Value::Uint(a) => *a,
                Value::Bool(b) => *b as u128,
                other => {
                    return Err(InterpError::type_mismatch(
                        "cast",
                        "numeric source",
                        other.kind_name(),
                    ))
                }
            };
            Ok(Value::Uint(wrap_unsigned(raw, ut.bits())))
        }
        Ty::Float(ft) => {
            // `f128` sources and targets convert through the exact
            // binary128 module, never through an `f64` rounding of
            // the value (an `f64` *storage* widening of an `f16` /
            // `f32` source is exact, so widening from `Value::Float`
            // loses nothing).
            match (v, ft) {
                (Value::Float128(a), FloatTy::F128) => return Ok(Value::Float128(*a)),
                (Value::Float128(a), FloatTy::F64) => {
                    return Ok(Value::Float(f64::from_bits(f128::to_f64_bits(*a))))
                }
                (Value::Float128(a), FloatTy::F32) => {
                    return Ok(Value::Float(
                        f32::from_bits(f128::to_f32_bits(*a)) as f64,
                    ))
                }
                (Value::Float128(a), FloatTy::F16) => {
                    return Ok(Value::Float(f128::f16_bits_to_f64(f128::to_f16_bits(
                        *a,
                    ))))
                }
                (Value::Float(a), FloatTy::F128) => {
                    return Ok(Value::Float128(f128::from_f64(a.to_bits())))
                }
                (Value::Int(a), FloatTy::F128) => {
                    return Ok(Value::Float128(f128::from_int(*a)))
                }
                (Value::Uint(a), FloatTy::F128) => {
                    return Ok(Value::Float128(f128::from_uint(*a)))
                }
                _ => {}
            }
            // Convert at the target width directly: going through
            // `f64` first would double-round wide integer sources.
            let out = match (v, ft) {
                (Value::Int(a), FloatTy::F16) => round_to_f16(*a as f64),
                (Value::Int(a), FloatTy::F32) => (*a as f32) as f64,
                (Value::Int(a), FloatTy::F64) => *a as f64,
                (Value::Uint(a), FloatTy::F16) => round_to_f16(*a as f64),
                (Value::Uint(a), FloatTy::F32) => (*a as f32) as f64,
                (Value::Uint(a), FloatTy::F64) => *a as f64,
                (Value::Float(a), FloatTy::F16) => round_to_f16(*a),
                (Value::Float(a), FloatTy::F32) => (*a as f32) as f64,
                (Value::Float(a), FloatTy::F64) => *a,
                (other, _) => {
                    return Err(InterpError::type_mismatch(
                        "cast",
                        "numeric source",
                        other.kind_name(),
                    ))
                }
            };
            Ok(Value::Float(out))
        }
        Ty::Bool => match v {
            Value::Bool(b) => Ok(Value::Bool(*b)),
            Value::Int(a) => Ok(Value::Bool(*a != 0)),
            Value::Uint(a) => Ok(Value::Bool(*a != 0)),
            other => Err(InterpError::type_mismatch(
                "cast",
                "numeric source",
                other.kind_name(),
            )),
        },
        Ty::Ref(_) => match v {
            // A fat pointer cast to a thin pointer type keeps
            // only its data pointer: the metadata half (a slice
            // length, a string length, a vtable) has no
            // counterpart in the target type.
            Value::FatRef { base, .. } => Ok((**base).clone()),
            Value::DynRef { base, .. } => Ok((**base).clone()),
            Value::Ref(_) => Ok(v.clone()),
            other => Err(InterpError::type_mismatch(
                "cast",
                "pointer source",
                other.kind_name(),
            )),
        },
        _ => Err(InterpError::UnsupportedRvalue {
            kind: format!("Cast to {ty:?}"),
        }),
    }
}

/// Convenience one-shot entry point with the default configuration.
/// Judge the drop of one value of dynamic type (the
/// [`crate::ir::DropPlan::Dynamic`] plan): the value's concrete
/// contents are all the glue can know. Scalars, strings and
/// references own nothing; a `DynRef` recurses into the payload it
/// carries; tuples and closure environments recurse into their
/// elements (neither has a destructor of its own). An ADT or
/// coroutine value might hide a destructor in its concrete type,
/// which this judgment cannot name: an explicit error, never a
/// silent skip — the same rule `DropPlan::Untyped` applies.
/// Judge the drop of a `dyn` value that carries no
/// conversion-accounted glue (see the `DropPlan::Dyn` arm): strip
/// the box shapes the interpreter's own builtins build (a one-field
/// aggregate holding the payload, possibly behind a glue-less
/// `DynRef`) and judge the payload itself with
/// [`drop_dyn_value`], whose shapes that can hide a destructor are
/// explicit errors.
fn drop_unaccounted_dyn(value: &Value) -> InterpResult<()> {
    match value {
        Value::DynRef { base, .. } => drop_unaccounted_dyn(base),
        // One-field aggregates nest: a modelled smart-pointer or
        // handle value wraps its payload in one aggregate per
        // layer (a  around its counted handle around the
        // wake object, whose own one-field structs wrap theirs).
        // Strip the chain; the payload itself is still judged by
        // , whose shapes that can hide a
        // destructor stay explicit errors.
        Value::Enum { fields, .. } if fields.len() == 1 => {
            drop_unaccounted_dyn(&fields[0])
        }
        other => drop_dyn_value(other),
    }
}

fn drop_dyn_value(value: &Value) -> InterpResult<()> {
    match value {
        Value::Uninit
        | Value::Unit
        | Value::Bool(_)
        | Value::Int(_)
        | Value::Uint(_)
        | Value::Float(_)
        | Value::Float128(_)
        | Value::Str(_)
        | Value::Bytes(_)
        | Value::Ref(_)
        | Value::FatRef { .. }
        | Value::FnPtr(_)
        | Value::FmtArg { .. }
        | Value::FmtCount(_)
        | Value::FmtArguments(_)
        | Value::Formatter(_)
        | Value::FmtBuilder(_) => Ok(()),
        Value::DynRef { base, .. } => drop_dyn_value(base),
        Value::Tuple(items) => items.iter().try_for_each(drop_dyn_value),
        Value::Closure { env, .. } => env.iter().try_for_each(drop_dyn_value),
        Value::Enum { fields, .. } if fields.is_empty() => Ok(()),
        Value::Enum { .. } | Value::Coroutine { .. } | Value::PanicPayload(_) => {
            Err(InterpError::UnsupportedTerminator {
                kind: "Drop".to_string(),
            })
        }
    }
}

/// Parse an impl-instance key `<Self as Trait>::method` into
/// `(Self section, trait path without generic arguments, method)`,
/// mirroring `mir_convert`'s key parsing. `None` for other shapes.
/// Split a rendered generic-argument list's contents at its
/// top-level commas (commas inside nested `<...>`, `(...)`,
/// `[...]`, `{...}` groups do not split). Empty input yields an
/// empty list.
fn split_generic_args(inner: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = inner.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'-' if bytes.get(i + 1) == Some(&b'>') => {
                i += 1;
            }
            b'<' | b'(' | b'[' | b'{' => depth += 1,
            b'>' | b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(inner[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    let tail = inner[start..].trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }
    parts
}

/// The `(plain path, arguments)` of a rendered instance path:
/// every top-level `::<...>` group is removed from the path and
/// its arguments collected in order. A path with no group yields
/// itself and an empty list; an unbalanced group yields `None`.
fn strip_generic_groups(path: &str) -> Option<(String, Vec<String>)> {
    let mut plain = String::new();
    let mut args = Vec::new();
    let bytes = path.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b':' && path[i..].starts_with("::<") {
            let mut depth = 0i32;
            let mut j = i + 2;
            let inner_start = j + 1;
            let mut inner_end = None;
            while j < bytes.len() {
                match bytes[j] {
                    b'-' if bytes.get(j + 1) == Some(&b'>') => {
                        j += 1;
                    }
                    b'<' => depth += 1,
                    b'>' => {
                        depth -= 1;
                        if depth == 0 {
                            inner_end = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            let inner_end = inner_end?;
            args.extend(split_generic_args(&path[inner_start..inner_end]));
            i = inner_end + 1;
        } else {
            let ch_len = path[i..].chars().next()?.len_utf8();
            plain.push_str(&path[i..i + ch_len]);
            i += ch_len;
        }
    }
    Some((plain, args))
}

/// The parts of a coroutine *type* tag as a `dyn` value carries
/// it: the enclosing definition's plain path, the closing
/// `{closure#N}` segment, and the tag's generic arguments (the
/// enclosing definition's arguments first, then the coroutine
/// signature's own). `None` when the tag is not of this shape.
fn parse_coroutine_tag(tag: &str) -> Option<(String, String, Vec<String>)> {
    let idx = tag.rfind("::{closure#")?;
    let parent = &tag[..idx];
    let rest = &tag[idx + 2..];
    let seg_end = rest.find('}')?;
    let seg = &rest[..=seg_end];
    let tail = &rest[seg_end + 1..];
    let args = if tail.is_empty() {
        Vec::new()
    } else {
        let inner = tail.strip_prefix('<')?.strip_suffix('>')?;
        split_generic_args(inner)
    };
    Some((parent.to_string(), seg.to_string(), args))
}

/// The parts of a coroutine body *key*: the enclosing
/// definition's plain path and its instance arguments (see
/// [`strip_generic_groups`]), plus the `{closure#N}` segment.
/// `None` when the key is not a coroutine body key of this
/// shape.
fn parse_coroutine_key(key: &str) -> Option<(String, String, Vec<String>)> {
    let idx = key.rfind("::{closure#")?;
    let rest = &key[idx + 2..];
    let seg_end = rest.find('}')?;
    if seg_end + 1 != rest.len() {
        return None;
    }
    let seg = &rest[..=seg_end];
    let (parent, args) = strip_generic_groups(&key[..idx])?;
    Some((parent, seg.to_string(), args))
}

fn parse_impl_key(base: &str) -> Option<(String, String, String)> {
    let (head, method) = base.rsplit_once(">::")?;
    if method.is_empty() || method.contains("::") {
        return None;
    }
    let inner = head.strip_prefix('<')?;
    let (self_part, trait_part) = inner.split_once(" as ")?;
    if self_part.is_empty() || trait_part.is_empty() {
        return None;
    }
    let trait_base = trait_part.split('<').next().unwrap_or(trait_part);
    Some((
        self_part.to_string(),
        trait_base.to_string(),
        method.to_string(),
    ))
}

/// Whether an impl key's `Self` section denotes `name`: the name
/// itself, or the name with a generic argument list appended; the
/// crate prefix may be omitted on a side that carries none.
fn impl_self_matches(self_part: &str, name: &str) -> bool {
    let self_base = self_part.split('<').next().unwrap_or(self_part);
    // The queried name may itself carry a generic argument list
    // (a fully rendered concrete type); only its base takes part
    // in the comparison, as the key side's list is stripped above.
    let name = name.split('<').next().unwrap_or(name);
    if self_part != self_base && !self_part.ends_with('>') {
        return false;
    }
    if self_base == name {
        return true;
    }
    if self_base.contains("::") && name.contains("::") {
        return false;
    }
    self_base.rsplit("::").next().unwrap_or(self_base)
        == name.rsplit("::").next().unwrap_or(name)
}

/// Whether `v` holds a callable value anywhere inside it (a
/// closure, a function pointer, or a dyn callable).
fn value_holds_callable(v: &Value) -> bool {
    match v {
        Value::Closure { .. } | Value::FnPtr(_) | Value::DynRef { .. } => true,
        Value::Tuple(items) => items.iter().any(value_holds_callable),
        Value::Enum { fields, .. } => fields.iter().any(value_holds_callable),
        _ => false,
    }
}

/// Shape a dyn call's data pointer into the reference an impl
/// body's reference-typed `self` expects. A reference, fat or dyn
/// pointer already denotes the pointee and passes through as it
/// is. A one-field smart-pointer wrapper (a built `Box`
/// aggregate) denotes the pointee through its only field: when
/// that field is itself a pointer — an allocation pointer for a
/// built box, or any reference-shaped value — the field *is* the
/// reference to the pointee, and wrapping it in a value reference
/// instead would make the callee's deref yield the pointer itself,
/// with every further field projection then failing on a
/// reference where an aggregate belongs. A plain field value, and
/// any other bare value, is referenced as a value.
/// The value standing for the default panic hook (what
/// `take_hook` returns when no guest hook is installed): a
/// `dyn Fn` marker the runtime recognises instead of calling.
fn default_hook_value() -> Value {
    Value::DynRef {
        base: Box::new(Value::Unit),
        concrete: "std::panic::default_hook".to_string(),
        traits: vec![
            "std::ops::Fn".to_string(),
            "std::marker::Send".to_string(),
            "std::marker::Sync".to_string(),
        ],
        drop_glue: None,
    }
}

/// Whether a hook value is the default-hook marker.
fn is_default_hook(value: &Value) -> bool {
    matches!(value, Value::DynRef { concrete, .. } if concrete == "std::panic::default_hook")
}

fn ref_shaped_receiver(value: Value) -> Value {
    // A reference-counted handle (the data pointer of an
    // `Rc<dyn Trait>` / `Arc<dyn Trait>`) names its pointee in
    // the allocation store, not inside the handle value: the
    // receiver is a reference to the stored value field, live
    // for every handle on the allocation.
    if let Some(id) = rc_handle_id(&value) {
        return Value::Ref(RefTarget::Alloc {
            id,
            projection: vec![RefProj::Field(2)],
        });
    }
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => {
            let field = fields.into_iter().next().expect("one field");
            match field {
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => field,
                field => Value::Ref(RefTarget::value(field)),
            }
        }
        other @ (Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }) => other,
        other => Value::Ref(RefTarget::value(other)),
    }
}

/// Unwrap transparent one-field wrappers (`Enum` / `Tuple` values
/// with exactly one field) around a callable value, returning the
/// inner value and whether any unwrapping happened. A wrapper is
/// only entered when its sole field holds a callable, so an
/// ordinary one-field ADT value that is not a callable wrapper is
/// left untouched.
fn unwrap_callable_wrapper(v: Value) -> (Value, bool) {
    let mut value = v;
    let mut unwrapped = false;
    for _ in 0..8 {
        let next = match &value {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                if value_holds_callable(&fields[0]) {
                    fields[0].clone()
                } else {
                    break;
                }
            }
            Value::Tuple(items) if items.len() == 1 => {
                if value_holds_callable(&items[0]) {
                    items[0].clone()
                } else {
                    break;
                }
            }
            _ => break,
        };
        value = next;
        unwrapped = true;
    }
    (value, unwrapped)
}

pub fn interpret(program: &Program, entry: &str, args: Vec<Value>) -> InterpResult<Value> {
    Interpreter::new(program).run(entry, args)
}

/// Extend a `RefCell` place target with a field path inside the
/// cell. Only place targets can be extended; guard construction
/// starts from `Interpreter::refcell_place`, which has already
/// rejected the rest.
pub(crate) fn refcell_sub_target(cell: &RefTarget, path: &[RefProj]) -> RefTarget {
    match cell {
        RefTarget::Place {
            frame,
            local,
            projection,
        } => {
            let mut projection = projection.clone();
            projection.extend_from_slice(path);
            RefTarget::Place {
                frame: *frame,
                local: *local,
                projection,
            }
        }
        RefTarget::Static { name, projection } => {
            let mut projection = projection.clone();
            projection.extend_from_slice(path);
            RefTarget::Static {
                name: name.clone(),
                projection,
            }
        }
        RefTarget::Alloc { id, projection } => {
            let mut projection = projection.clone();
            projection.extend_from_slice(path);
            RefTarget::Alloc {
                id: *id,
                projection,
            }
        }
        // Unreachable through `refcell_place`; keep the target as-is
        // rather than panicking in the model.
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Inline-asm register program execution (the evaluable class of
// sparagmos-mir's `inline_asm_program`): slots are 64-bit, reads
// and writes follow the x86-64 partial-register rules.
// ---------------------------------------------------------------------------

/// The bit mask of a `width`-bit operand.
fn asm_width_mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Sign-extend the low `width` bits of `bits` to 64 bits.
fn asm_sign_extend(bits: u64, width: u32) -> u64 {
    if width >= 64 {
        bits
    } else {
        let shift = 64 - width;
        ((bits << shift) as i64 >> shift) as u64
    }
}

/// The bits an asm input value enters its slot with: the value
/// zero-extended from the operand width. (The bits above the
/// width are unspecified on hardware — a wider register form
/// read there observes whatever the caller left; the model
/// fixes them to zero.)
fn asm_input_bits(value: &Value, reg: &AsmReg) -> InterpResult<u64> {
    let bits = match value {
        Value::Int(v) => *v as u64,
        Value::Uint(v) => *v as u64,
        Value::Bool(b) => u64::from(*b),
        other => {
            return Err(InterpError::type_mismatch(
                "inline asm operand",
                "integer or bool",
                other.kind_name(),
            ));
        }
    };
    Ok(bits & asm_width_mask(reg.width))
}

/// The value written back to an asm output place: the slot bits
/// interpreted at the operand type.
fn asm_output_value(bits: u64, reg: &AsmReg) -> Value {
    let masked = bits & asm_width_mask(reg.width);
    match reg.kind {
        AsmRegKind::Signed => Value::Int(asm_sign_extend(masked, reg.width) as i128),
        AsmRegKind::Unsigned => Value::Uint(u128::from(masked)),
        AsmRegKind::Bool => Value::Bool(masked != 0),
    }
}

/// Read a register slot at an instruction operand: the modifier
/// picks the sub-register (`h` is bits 8..16); no modifier reads
/// the whole register — the full 64-bit register for the `reg`
/// class (a placeholder without a modifier names it, whatever
/// the operand width), the byte for the `reg_byte` class.
fn asm_read(slots: &[u64], regs: &[AsmReg], idx: usize, modifier: Option<char>) -> u64 {
    let slot = slots[idx];
    match modifier {
        Some('h') => (slot >> 8) & 0xff,
        Some('l') => slot & 0xff,
        Some('x') => slot & 0xffff,
        Some('e') => slot & 0xffff_ffff,
        _ if regs[idx].width == 8 => slot & 0xff,
        _ => slot,
    }
}

/// Write a register slot at an instruction operand, with x86-64
/// partial-register semantics: a 32-bit write zero-extends,
/// 8/16-bit writes preserve the rest of the slot, an `h` write
/// lands in bits 8..16, and a modifier-less write replaces the
/// whole register (the byte for the `reg_byte` class).
fn asm_write(
    slots: &mut [u64],
    written: &mut [bool],
    regs: &[AsmReg],
    idx: usize,
    modifier: Option<char>,
    value: u64,
) {
    written[idx] = true;
    let slot = &mut slots[idx];
    match modifier {
        Some('h') => *slot = (*slot & !0xff00) | ((value & 0xff) << 8),
        Some('l') => *slot = (*slot & !0xff) | (value & 0xff),
        Some('x') => *slot = (*slot & !0xffff) | (value & 0xffff),
        Some('e') => *slot = value & 0xffff_ffff,
        _ if regs[idx].width == 8 => *slot = (*slot & !0xff) | (value & 0xff),
        _ => *slot = value,
    }
}

/// The width an instruction operates at: the destination
/// modifier's width, else the full register (64 bits, or 8 for
/// the `reg_byte` class) — a modifier-less placeholder names the
/// whole register, so e.g. a shift count is masked at 64 bits,
/// not at the operand width.
fn asm_instr_width(regs: &[AsmReg], idx: usize, modifier: Option<char>) -> u32 {
    match modifier {
        Some('e') => 32,
        Some('x') => 16,
        Some('l') | Some('h') => 8,
        _ if regs[idx].width == 8 => 8,
        _ => 64,
    }
}

/// The value of an instruction source operand.
fn asm_src_value(slots: &[u64], regs: &[AsmReg], src: &sparagmos_mir::AsmSrc) -> u64 {
    match src {
        sparagmos_mir::AsmSrc::Reg(idx, modifier) => asm_read(slots, regs, *idx, *modifier),
        sparagmos_mir::AsmSrc::Imm(bits) => *bits,
    }
}

/// Run one classified asm instruction against the slots.
fn exec_asm_step(
    slots: &mut [u64],
    written: &mut [bool],
    regs: &[AsmReg],
    step: &AsmStep,
) {
    use sparagmos_mir::{AsmBinOp, AsmUnOp};
    match step {
        AsmStep::Nop | AsmStep::Ret => {}
        AsmStep::Mov { dst, dst_mod, src } => {
            let value = asm_src_value(slots, regs, src);
            asm_write(slots, written, regs, *dst, *dst_mod, value);
        }
        AsmStep::Bin {
            op,
            dst,
            dst_mod,
            src,
        } => {
            let width = asm_instr_width(regs, *dst, *dst_mod);
            let mask = asm_width_mask(width);
            let a = asm_read(slots, regs, *dst, *dst_mod) & mask;
            let b = asm_src_value(slots, regs, src) & mask;
            let count = (b & if width >= 64 { 63 } else { 31 }) as u32;
            let result = match op {
                AsmBinOp::Add => a.wrapping_add(b),
                AsmBinOp::Sub => a.wrapping_sub(b),
                AsmBinOp::And => a & b,
                AsmBinOp::Or => a | b,
                AsmBinOp::Xor => a ^ b,
                AsmBinOp::Shl => a << count,
                AsmBinOp::Shr => a >> count,
                AsmBinOp::Sar => (asm_sign_extend(a, width) as i64 >> count) as u64,
            };
            asm_write(slots, written, regs, *dst, *dst_mod, result & mask);
        }
        AsmStep::Un { op, dst, dst_mod } => {
            let width = asm_instr_width(regs, *dst, *dst_mod);
            let mask = asm_width_mask(width);
            let a = asm_read(slots, regs, *dst, *dst_mod) & mask;
            let result = match op {
                AsmUnOp::Neg => 0u64.wrapping_sub(a),
                AsmUnOp::Not => !a,
                AsmUnOp::Inc => a.wrapping_add(1),
                AsmUnOp::Dec => a.wrapping_sub(1),
            };
            asm_write(slots, written, regs, *dst, *dst_mod, result & mask);
        }
    }
}

/// The pointee type of a bare-local pointer operand: a reference
/// or raw-pointer local with no projection, whose declared type
/// says what it points at. Anything else (a constant, a projected
/// place, a non-pointer local) has no statically known pointee
/// here.
pub(crate) fn operand_pointee_ty(body: &Body, op: &Operand) -> Option<Ty> {
    let (Operand::Copy(place) | Operand::Move(place)) = op else {
        return None;
    };
    if !place.projection.is_empty() {
        return None;
    }
    match body.local_ty(place.local) {
        Some(Ty::Ref(inner)) => Some((**inner).clone()),
        _ => None,
    }
}

/// Whether `value` has the shape of a value of type `ty`, for the
/// shapes the copy machinery must tell apart: scalars directly,
/// tuples and arrays element by element, ADT values as variant
/// aggregates, references as resolved pointers. Anything
/// unrecognised does not fit, so callers fall back to treating
/// the value by its own shape.
fn value_fits_ty(value: &Value, ty: &Ty) -> bool {
    match (value, ty) {
        (Value::Unit, Ty::Unit) => true,
        (Value::Bool(_), Ty::Bool) => true,
        (Value::Int(_), Ty::Int(_)) => true,
        (Value::Uint(_), Ty::Uint(_)) => true,
        (Value::Float(_), Ty::Float(_)) => true,
        (Value::Float128(_), Ty::Float(_)) => true,
        (Value::Tuple(items), Ty::Tuple(tys)) => {
            items.len() == tys.len()
                && items
                    .iter()
                    .zip(tys.iter())
                    .all(|(v, t)| value_fits_ty(v, t))
        }
        (Value::Tuple(items), Ty::Array(elem, n)) => {
            items.len() as u64 == *n
                && items.iter().all(|v| value_fits_ty(v, elem))
        }
        (Value::Bytes(b), Ty::Array(elem, n)) => {
            matches!(&**elem, Ty::Uint(UintTy::U8)) && b.len() as u64 == *n
        }
        (Value::Enum { .. }, Ty::Adt(_)) => true,
        (Value::Str(_), Ty::Str) => true,
        (Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }, Ty::Ref(_)) => {
            true
        }
        (Value::FnPtr(_), Ty::FnPtr) => true,
        (Value::FnPtr(_), Ty::FnDef(_)) => true,
        (Value::Closure { .. }, Ty::Closure(..)) => true,
        (Value::Coroutine { .. }, Ty::Coroutine(_)) => true,
        _ => false,
    }
}
