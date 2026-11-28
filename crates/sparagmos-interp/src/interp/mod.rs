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
//!   looping program can never hang the oracle (or a test run). Each
//!   thread's frames are also charged against a stack budget (see
//!   [`InterpConfig::main_stack_size`]): a guest that would overflow
//!   its native stack observes the native outcome instead -- the
//!   runtime's overflow report on its stderr and an abort.
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

use crate::ir::*;
use crate::value::{PanicPayload, RefProj, RefTarget, Value};

use std::collections::{HashMap, HashSet};

mod address;
mod alloc;
mod arith;
mod asm;
mod byteview;
mod call;
mod cast;
mod drop_glue;
mod exec;
mod host;
mod materialize;
mod memory;
mod panics;
mod places;
mod project;
mod signals;
mod state;
mod threads;
mod values;

pub use host::{
    ProcessChildOutcome, ProcessHost, ProcessHostError, ProcessHostOps, ProcessSpawnRequest,
    ProcessSpawned, ProcessStdio, StdoutSink, ToolInterception, ToolRunOutcome, ToolRunRequest,
    ToolRunner,
};
pub use state::FmtBuilderKind;
pub(crate) use signals::{SignalAction, SIGPIPE};

pub(crate) use address::*;
pub(crate) use arith::*;
pub(crate) use asm::*;
pub(crate) use byteview::*;
pub(crate) use call::*;
pub(crate) use cast::*;
pub(crate) use drop_glue::*;
pub(crate) use host::*;
pub(crate) use memory::*;
pub(crate) use panics::*;
pub(crate) use places::*;
pub(crate) use project::*;
pub(crate) use state::*;
pub(crate) use values::*;

/// The default scheduler quantum, in interpreter steps (see
/// [`InterpConfig::thread_quantum`]). The size is set by the
/// spin-heavy end of the thread suite: a guest spin iteration
/// is a handful of steps, so the quantum is the number of spin
/// iterations a thread burns before its partner runs. Measured
/// on a strict two-thread rendezvous (2,000 ping-pong rounds on
/// one atomic, debug build): 8,192 steps took 139 s, 1,024 took
/// 17.7 s, and 256 took 4.4 s — the wasted spin shrinks with the
/// quantum while the switch cost (moving the segment's frame
/// vector, scanning its locals for the frames it references)
/// stays invisible at this size, so 256 it is. A slice still
/// holds dozens of spin iterations, far more than a channel
/// operation's step cost, so lock-step protocols are not
/// chopped mid-operation.
pub const DEFAULT_THREAD_QUANTUM: usize = 256;

/// How the guest program's runtime initialisation sets
/// up `SIGPIPE` before `main` runs: the `sigpipe` value
/// the code the compiler generates hands to `lang_start`
/// (see `compiler/rustc_session/src/config/sigpipe.rs`,
/// whose constants std's `reset_sigpipe` duplicates).
/// The driver resolves it from the compilation the way
/// the compiler does (see `InterpConfig::sigpipe`); the
/// interpreter seeds the disposition it names into the
/// signal model at construction, since the entry the
/// interpreter runs is `main` itself and the runtime
/// initialisation is not guest code it executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigpipeInit {
    /// Neither `-Zon-broken-pipe` nor `#[unix_sigpipe]`:
    /// the runtime installs `SIG_IGN`. A child the guest
    /// spawns has `SIGPIPE` reset to `SIG_DFL` before
    /// exec (std's spawn-time rule for the default
    /// setup).
    Default,
    /// `-Zon-broken-pipe=inherit` or
    /// `#[unix_sigpipe = "inherit"]`: the runtime
    /// installs nothing; the guest keeps the disposition
    /// it started with, and children inherit the guest's.
    Inherit,
    /// `-Zon-broken-pipe=error` or
    /// `#[unix_sigpipe = "sig_ign"]`: the runtime
    /// installs `SIG_IGN`, and children inherit the
    /// guest's disposition.
    Ignore,
    /// `-Zon-broken-pipe=kill` or
    /// `#[unix_sigpipe = "sig_dfl"]`: the runtime
    /// installs `SIG_DFL`, and children inherit the
    /// guest's disposition.
    SigDfl,
}

impl SigpipeInit {
    /// The handler the runtime initialisation installs
    /// for `SIGPIPE` (`SIG_IGN` = 1, `SIG_DFL` = 0), or
    /// `None` when it installs nothing.
    pub(crate) fn rt_init_handler(self) -> Option<u64> {
        match self {
            SigpipeInit::Default | SigpipeInit::Ignore => Some(1),
            SigpipeInit::SigDfl => Some(0),
            SigpipeInit::Inherit => None,
        }
    }

    /// Whether the compilation named a `SIGPIPE` mode
    /// explicitly (std's `on_broken_pipe_used`): a
    /// spawned child then inherits the guest's
    /// disposition instead of being reset to `SIG_DFL`.
    pub(crate) fn on_broken_pipe_used(self) -> bool {
        self != SigpipeInit::Default
    }
}

/// Interpreter limits / configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterpConfig {
    /// Maximum number of statements + terminators executed in one run.
    ///
    /// The default budget is a safety bound against guests that
    /// never terminate, not a model of any limit the native run
    /// has: compiletest bounds a test only by wall clock, so a
    /// finite guest must never meet this bound first. It is
    /// sized from measured demand (issue #399): the hungriest
    /// finite test-suite programs need ~2.8e8 steps
    /// (`tests/ui/range/step_overflowing.rs`; the plain-loop
    /// programs need ~1e6-7e6), so 1e9 leaves ~3.6x headroom
    /// over the largest measured need, while a guest that
    /// truly never terminates still dies with
    /// `StepLimitExceeded` -- in minutes for a plain loop,
    /// rather than running forever. In the measurement runner
    /// the per-record wall-clock timeout stays the operative
    /// bound for guests whose per-step cost is high.
    pub max_steps: usize,
    /// The scheduler quantum: the number of steps one thread
    /// segment may execute before it yields to another runnable
    /// thread (see [`Interpreter::run_thread`]). Yielding is the
    /// cooperative scheduler's preemption: a guest that busy-waits
    /// (a spin on an atomic, a `try_recv` polling loop) cannot
    /// monopolise the interpreter, exactly as an OS scheduler
    /// preempts it. The step budget ([`InterpConfig::max_steps`])
    /// is global and unaffected: a yield consumes no budget reset,
    /// it only changes which thread executes next.
    pub thread_quantum: usize,
    /// Maximum number of simultaneously live call frames.
    pub max_call_depth: usize,
    /// The entry (main) thread's stack budget, in bytes. A frame
    /// push that would take the thread's charged stack usage (the
    /// sum of its live frames' materialised-local sizes; see
    /// `body_stack_charge`) past this
    /// budget ends the run the way a native stack overflow does:
    /// the runtime's `has overflowed its stack` report on the
    /// guest's stderr and an abort (see
    /// [`Interpreter::charge_stack`]). The default is the native
    /// main thread's usual stack size (the common `RLIMIT_STACK`
    /// value a natively built test binary runs with).
    pub main_stack_size: usize,
    /// A spawned thread's default stack budget, in bytes: std's
    /// default thread stack size, which a
    /// `std::thread::Builder::stack_size` setting overrides for
    /// the thread it spawns (see [`InterpConfig::main_stack_size`]
    /// for the accounting).
    pub thread_stack_size: usize,
    /// The guest program's command-line arguments after argv[0]
    /// (the run-flags a test prescribes), as handed to the driver
    /// through its `--guest-arg` options. The arguments are raw
    /// bytes: a guest argument need not be valid UTF-8 (a
    /// self-spawned child can carry one, see issue #515), and
    /// the `env::args` / `env::args_os` runtime support reads
    /// them from here; argv[0] itself is the interpreter's
    /// policy, not part of this list.
    pub guest_args: Vec<Vec<u8>>,
    /// The guest program's argv[0]: the path the program was
    /// invoked as. The driver sets it to the source path it was
    /// handed (the stand-in for the compiled binary's path: a
    /// natively built test binary is invoked by its own path, and
    /// the source path is the corresponding name in this model),
    /// or to the invocation name it was given separately when it
    /// reconstructs a run under another name (an artifact run:
    /// the name the recipe invoked the built binary under).
    /// Empty when no invocation path is known; `env::args` then
    /// yields just the guest arguments.
    pub program_path: String,
    /// The main compilation's overflow-checks setting, as the
    /// driver resolved it from the compilation's flags. Overflow
    /// assertions (see `TerminatorKind::Assert::overflow`) fire
    /// per the *instantiating* crate's setting, resolved at run
    /// time by `overflow_checks_at`; this value is the setting
    /// of the main crate and the fallback for frames whose
    /// crate has no recorded setting. The arithmetic operations
    /// themselves wrap either way, exactly as in MIR. Defaults
    /// to `true` (the debug-profile default).
    pub overflow_checks: bool,
    /// The session's panic strategy, as the driver resolved it
    /// from the compilation's flags: `true` when the guest was
    /// compiled with `-C panic=abort`. Under the abort strategy
    /// a panic prints its report and the run aborts at the
    /// panic site: nothing unwinds, so the nounwind-boundary
    /// sequence (a `panic_cannot_unwind` report and the
    /// non-unwinding line) never appears. Defaults to `false`
    /// (the unwind strategy).
    pub panic_abort: bool,
    /// Whether the entry shim frame (the
    /// `<fn() as FnOnce<()>>::call_once` frame `lang_start`
    /// calls the entry function pointer through) appears
    /// below the entry frame in a short backtrace, as the
    /// driver resolved it from the compilation's flags.
    /// Natively the shim is a real frame in an unoptimised
    /// build; an optimising build inlines it into
    /// `__rust_begin_short_backtrace`, where it leaves no
    /// frame of its own unless debug info lets the
    /// backtrace printer reconstruct the inlined frame.
    /// The frame is therefore visible exactly when the
    /// build is unoptimised or carries debug info.
    /// Defaults to `true` (the unoptimised shape).
    pub entry_shim_visible: bool,
    /// The guest's `SIGPIPE` setup, as the driver
    /// resolved it from the compilation's flags (see
    /// [`SigpipeInit`]). The disposition it names is
    /// seeded into the signal model before the entry
    /// runs, and the child-process rule reads the mode
    /// at spawn time. Defaults to
    /// [`SigpipeInit::Default`].
    pub sigpipe: SigpipeInit,
    /// Tool interception for this run (see [`ToolInterception`]);
    /// `None` keeps the process model's standing behaviour, under
    /// which no child process ever starts.
    pub tool_interception: Option<ToolInterception>,
    /// The host side of real child processes for this run (see
    /// [`ProcessHost`]); `None` keeps the process model's
    /// standing behaviour, under which no child process ever
    /// starts. Tool interception takes precedence for the
    /// programs it claims.
    pub process_host: Option<ProcessHost>,
    /// The write-through target for the guest's standard
    /// output (see [`StdoutSink`]); `None` keeps the standing
    /// behaviour, under which the guest's output is only
    /// captured. The driver installs it for a run whose own
    /// stdout is a real stream the guest's writes must reach
    /// as they happen, errors included: a child process the
    /// process model re-runs through the driver, whose parent
    /// may already have closed the pipe (the guest then
    /// observes the broken pipe, as it would natively).
    pub stdout_sink: Option<StdoutSink>,
}

impl Default for InterpConfig {
    fn default() -> Self {
        InterpConfig {
            max_steps: 1_000_000_000,
            thread_quantum: DEFAULT_THREAD_QUANTUM,
            max_call_depth: 8192,
            main_stack_size: 8 * 1024 * 1024,
            thread_stack_size: 2 * 1024 * 1024,
            guest_args: Vec::new(),
            program_path: String::new(),
            overflow_checks: true,
            panic_abort: false,
            entry_shim_visible: true,
            sigpipe: SigpipeInit::Default,
            tool_interception: None,
            process_host: None,
            stdout_sink: None,
        }
    }
}

/// The panic entry path a freshly raised panic arrived by: it
/// decides the panic-machinery frames a rendered backtrace
/// starts with, which differ by entry in std's own traces (a
/// literal `panic!` reaches `begin_panic::<&str>` through
/// `panic_display`, a formatted one goes through `panic_fmt`).
/// Recorded by the panic entry points alongside the
/// caller-location stamp.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PanicOrigin {
    /// `core::panicking::panic` with a `&str` payload (a
    /// literal `panic!`, `todo!`, an explicit `panic("...")`):
    /// the payload's capture sits in std's `begin_panic::<&str>`,
    /// which heads the trace alone.
    PanicFn,
    /// A formatted `panic_fmt` / `panic_nounwind_fmt`: the
    /// trace heads with `__rustc::rust_begin_unwind` and
    /// `core::panicking::panic_fmt`.
    PanicFmt,
    /// `panic_any::<T>`: the trace heads with
    /// `std::panicking::begin_panic::<T>` and
    /// `std::panic::panic_any::<T>`.
    BeginPanic(String),
    PanicAny(String),
    /// `core::option::expect_failed`: the trace heads with
    /// `__rustc::rust_begin_unwind`, `core::panicking::panic_fmt`,
    /// `core::panicking::panic_display::<&str>` and itself.
    ExpectFailed,
    /// `core::option::unwrap_failed`: the trace heads with
    /// `__rustc::rust_begin_unwind`, `core::panicking::panic_fmt`,
    /// `core::panicking::panic` and itself.
    OptionUnwrap,
    /// `core::result::unwrap_failed`: the trace heads with
    /// `__rustc::rust_begin_unwind`, `core::panicking::panic_fmt`
    /// and itself.
    ResultUnwrap,
    /// `assert_failed`: the trace heads with
    /// `__rustc::rust_begin_unwind`, `core::panicking::panic_fmt`,
    /// `core::panicking::assert_failed_inner` and the
    /// `assert_failed::<T, U>` frame named here (empty when the
    /// call's generic arguments are unknown).
    AssertFailed(String),
    /// An `Assert` terminator (a bounds check, an arithmetic
    /// overflow, a misalignment check): the trace heads with
    /// `__rustc::rust_begin_unwind`, `core::panicking::panic_fmt`
    /// and the check's own entry, named here.
    Assert(&'static str),
}

/// The report context of the panic whose hook is running:
/// everything the default panic report would print for it.
/// An `update_hook` chain that bottoms out in the default
/// hook prints that report from this context (see
/// `Interpreter::invoke_panic_hook`).
#[derive(Clone)]
pub(crate) struct HookReportCtx {
    pub(crate) message: String,
    pub(crate) in_flight: usize,
    pub(crate) origin: Option<PanicOrigin>,
    pub(crate) no_backtrace: bool,
}

/// A resolved backtrace style for one panic report, mirroring
/// std's `BacktraceStyle` plus the force-no-backtrace case,
/// which prints neither a trace nor the note.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReportBacktrace {
    /// Force-no-backtrace: nothing follows the report head.
    None,
    /// No trace; the once-per-process note may follow.
    Off,
    Short,
    Full,
}

/// One call frame: a function's locals plus where to resume.
pub(crate) struct Frame {
    /// Unique id within the run; references name their target frame
    /// by this id, so a frame keeps its identity when other frames
    /// are pushed and popped around it.
    pub(crate) id: u64,
    pub(crate) func: String,
    pub(crate) locals: Vec<Value>,
    /// The frame's stack charge: the total size of the body's
    /// materialised locals, accounted against the owning
    /// thread's stack budget (see [`body_stack_charge`] and
    /// [`Interpreter::charge_stack`]).
    pub(crate) stack_bytes: u64,
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
    /// The `FnOnce::call_once` shim's environment drop owed for
    /// this frame's call: set on a closure-body frame when the
    /// caller started a `call_once` whose body takes its
    /// environment by reference and does not drop the environment
    /// itself — the real shim owns the receiver value and drops
    /// it when the call returns, and during unwinding out of the
    /// call. The root names the receiver place (or value) in the
    /// caller and the glue is the receiver `Self` type's, built
    /// by conversion onto the `CallTrait` terminator. `None` for
    /// every other frame.
    call_once_env_drop: Option<(RefTarget, crate::ir::DropGlue)>,
    /// The architectural register state of the asm block whose
    /// `call` step entered this frame (see
    /// `Interpreter::pending_asm_incoming`): when this frame's
    /// function is a naked body, its register file starts from
    /// this state, as a native `call` preserves every register
    /// the callee does not overwrite — including registers the
    /// argument seedings do not supply, which the calling block
    /// may have established deliberately (an explicit-register
    /// `in` operand the callee's body reads). The second array
    /// records which registers that state actually established,
    /// so reads of never-established registers keep their usual
    /// treatment (notably the incoming-`rsp` rejection). `None`
    /// for frames entered any other way: their naked blocks
    /// start from a zeroed file defined only by the seedings.
    incoming_asm: Option<([u64; 48], [bool; 48])>,
}

impl Frame {
    /// The frame's current position as a (block, statement)
    /// index pair into its body: the statement about to execute
    /// (or, for a caller frame, the call terminator it is
    /// suspended at, as a statement index past the block's
    /// statements). The debugger interface reads it to render a
    /// selected caller frame's source line.
    pub(crate) fn current_block_stmt(&self) -> (usize, usize) {
        (self.block, self.stmt)
    }
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

/// Where the `dyn` identity of a borrow whose own type is a
/// trait object comes from, when the borrowed place is not
/// itself a fat pointer (see
/// [`Interpreter::dyn_borrow_source`]).
pub(crate) enum DynBorrowSource {
    /// The value stored at the place is a fat pointer: the
    /// borrow is that fat pointer itself (a reborrow of it).
    Stored(Value),
    /// The place is the unsized tail of the pointee a fat
    /// pointer denotes: the borrow is a fat pointer over the
    /// thin borrow, with the crossed pointer's concrete type
    /// and drop glue, answering to the borrow's own bounds.
    Crossed {
        concrete: String,
        traits: Vec<String>,
        drop_glue: Option<Box<crate::ir::DropGlue>>,
    },
}

/// The interpreter state for one program.
pub struct Interpreter<'a> {
    program: &'a Program,
    config: InterpConfig,
    steps: usize,
    /// Coverage counting state: for each record of the program's
    /// coverage mapping table, the physical counter values the
    /// run has accumulated so far (indexed by counter id), and
    /// the body-name -> record-index link the counting statement
    /// resolves through. Both are empty when the program carries
    /// no coverage table.
    coverage_counts: std::cell::RefCell<Vec<Vec<u64>>>,
    coverage_by_body: std::collections::HashMap<String, usize>,
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
    /// Stack frames of the arithmetic address model (see
    /// [`Self::arithmetic_price`]): the frame's function name
    /// (the body its places resolve against) and the synthetic
    /// stack address its slot area starts at, recorded when
    /// the frame is pushed ([`Self::register_frame_base`]). A
    /// frame keeps the first base it was assigned, so a
    /// coroutine frame that suspends and resumes — and any
    /// pointer into a frame priced after the frame popped —
    /// keeps its addresses, like the storage it models.
    frame_info:
        std::cell::RefCell<std::collections::HashMap<u64, (String, u128)>>,
    /// Per-body stack slot layouts for the arithmetic address
    /// model, by function name (see [`SlotLayout`]): `None`
    /// for a body whose layout cannot be computed — its
    /// places keep synthetic-hash pricing.
    body_slots: std::cell::RefCell<
        std::collections::HashMap<String, Option<std::rc::Rc<SlotLayout>>>,
    >,
    /// Synthetic addresses of static storages, assigned from
    /// the statics region in first-pricing order.
    static_prices: std::cell::RefCell<std::collections::HashMap<String, u128>>,
    static_cursor: std::cell::Cell<u128>,
    /// Synthetic addresses of allocation-store allocations,
    /// assigned from the heap region in allocation-id order:
    /// `heap_prices` memoises by id, and `heap_cursor` — the
    /// next unpriced id and its address — advances as layouts
    /// become known.
    heap_prices: std::cell::RefCell<std::collections::HashMap<u64, u128>>,
    heap_cursor: std::cell::Cell<(u64, u128)>,
    /// Synthetic addresses of the data buffers of `Vec`
    /// values held in frame locals, assigned from the buffer
    /// region in first-pricing order (see
    /// `Interpreter::buffer_price`): `buffer_prices` memoises
    /// by (frame, local) of the holding local, and
    /// `buffer_cursor` is the region's next free address.
    buffer_prices:
        std::cell::RefCell<std::collections::HashMap<(u64, usize), u128>>,
    buffer_cursor: std::cell::Cell<u128>,
    /// Allocation layouts recorded by the raw allocator
    /// builtins (`__rust_alloc` and the libc family): (size,
    /// align) by allocation id. Typed creations are covered
    /// by `alloc_tys` / `box_layouts` instead.
    alloc_layouts:
        std::cell::RefCell<std::collections::HashMap<u64, (u64, u64)>>,
    /// Fat pointer values flattened through a fat pointer
    /// shape, by their 128-bit pattern: the pattern's words do
    /// not denote the value on their own (the data word is a
    /// synthetic address and a `dyn` metadata word is a vtable
    /// identity, not a number), so construction looks the
    /// pattern up here first and returns the very value that
    /// flattened to it. Only a pattern no value registered is
    /// fabricated from its words, and only where the words
    /// carry the value (a length metadata).
    fat_values: std::cell::RefCell<std::collections::HashMap<u128, Value>>,
    /// The concrete type each vtable word in circulation
    /// names, registered when a `dyn` value is flattened to
    /// its bit pattern (the word is all a transmuted
    /// metadata half carries afterwards): the reverse of
    /// `vtable_word`, which the `vtable_size` /
    /// `vtable_align` intrinsics consult.
    vtable_words: std::cell::RefCell<std::collections::HashMap<u128, String>>,
    /// Whether the lazy stdout buffer allocation has been
    /// modelled (see `note_stdout_use`): std's stdout handle
    /// allocates its line buffer through the global allocator on
    /// first use, which a guest allocator observes.
    stdout_buffer_modelled: std::cell::Cell<bool>,
    /// Everything the guest wrote to its stdout (via the runtime
    /// print support) during the most recent run.
    output: String,
    /// The guest stdout's line buffer, on its way to a
    /// write-through sink: std's `Stdout` is line-buffered
    /// (a `LineWriter`), so bytes the guest writes reach the
    /// stream only through the last newline, at a flush, or
    /// when the buffer fills — and a stream error surfaces
    /// at that point, not at the write that merely buffered
    /// the bytes. Without a sink this stays empty.
    stdout_line_buffer: Vec<u8>,
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
    /// The architectural register state an asm `call` step hands
    /// to the frame its call creates: set by `exec_asm_call` for
    /// the duration of the callee's `call_sync` when the callee
    /// is a naked body, and taken by `start_detached_call` into
    /// the new frame (see `Frame::incoming_asm`). `None` at
    /// every other time.
    pending_asm_incoming: Option<([u64; 48], [bool; 48])>,
    /// The architectural register state a naked body leaves
    /// behind for the asm `call` step that entered it: set by
    /// `exec_inline_asm` when a naked block finishes, and taken
    /// by `exec_asm_call` when the `call_sync` that ran the body
    /// returns, so the calling block continues with the state
    /// the callee actually established (see `exec_asm_call`).
    /// The second array marks the registers the body's steps
    /// changed relative to the seeded file they started from;
    /// registers the body never touched are not the callee's
    /// to impose on the caller. `None` at every other time.
    pending_asm_outgoing: Option<([u64; 48], [bool; 48])>,
    /// The guest's panic hook, installed by `std::panic::set_hook`
    /// (the boxed `dyn Fn` value as the guest handed it over);
    /// `None` is the default hook, whose report
    /// [`Interpreter::fire_panic_hook`] prints itself.
    /// Interior-mutable: most execution paths hold only `&self`.
    panic_hook: std::cell::RefCell<Option<Value>>,
    /// The guest's allocation-error hook, installed by
    /// `std::alloc::set_alloc_error_hook` (the `fn(Layout)`
    /// value as the guest handed it over); `None` is std's
    /// default hook. Interior-mutable, like `panic_hook`:
    /// most execution paths hold only `&self`.
    alloc_error_hook: std::cell::RefCell<Option<Value>>,
    /// The report context of the panic whose hook is
    /// currently running (see [`HookReportCtx`]); stashed by
    /// `call_panic_hook` for the length of the invocation.
    hook_report_ctx: std::cell::RefCell<Option<HookReportCtx>>,
    /// The backtrace style installed by
    /// `std::panic::set_backtrace_style`; recorded for the
    /// (currently unconverted) backtrace machinery.
    backtrace_style: std::cell::RefCell<Option<Value>>,
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
    /// The panics currently in flight on each thread (raised and
    /// not yet consumed: caught, turned into a thread outcome,
    /// or taken by the libtest harness model), keyed by thread
    /// id. std's per-thread panic count: a panic raised while
    /// another is in flight on the same thread forces a full
    /// backtrace in its report, and the count is what
    /// `std::panicking()` reads (see `Interpreter::panicking`).
    panics_in_flight: std::cell::RefCell<HashMap<u64, usize>>,
    /// Whether the process's first Off-style panic report has
    /// printed the `RUST_BACKTRACE=1` note (std's FIRST_PANIC):
    /// spent only by a report that prints the note.
    panic_note_spent: std::cell::Cell<bool>,
    /// Whether std's default allocation-error hook has already
    /// reported a failure in this process (its
    /// `PREV_ALLOC_FAILURE` static): a second failure prints
    /// the message with the recursion-guard line and no
    /// backtrace part. Spent only by the allocation-error
    /// model (see `builtins::alloc_error`).
    alloc_failure_reported: std::cell::Cell<bool>,
    /// Whether the default panic reports are suppressed because
    /// an enclosing model produces them itself: the libtest
    /// harness model sets this while it runs tests, since it
    /// synthesizes the captured reports (see
    /// `set_panic_report_suppressed`).
    panic_report_suppressed: std::cell::Cell<bool>,
    /// Whether `std::panic::always_abort` has run: every later
    /// panic aborts without the hook running.
    always_abort: std::cell::Cell<bool>,
    /// Set when a panic raised inside the panic hook has printed
    /// the processing-panic abort lines: the unwinder turns the
    /// in-flight panic into an abort when it sees this (see
    /// [`Interpreter::fire_panic_hook`]).
    hook_abort: std::cell::Cell<bool>,
    /// Whether the panic currently being raised forbids a
    /// backtrace (`panic_nounwind_fmt`'s flag): recorded by the
    /// entry point alongside the location stamp.
    last_panic_no_backtrace: std::cell::Cell<bool>,
    /// The entry path of the panic currently being raised (see
    /// [`PanicOrigin`]).
    last_panic_origin: std::cell::RefCell<Option<PanicOrigin>>,
    /// The frame-stack index the current thread's own stack
    /// starts at: frames below it belong to the thread that
    /// spawned this one (threads run cooperatively on one
    /// stack), so a rendered backtrace stops there. Set by
    /// `run_thread`; the entry thread's base is 0.
    thread_stack_base: std::cell::Cell<usize>,
    /// The frame id of the entry frame `run_with_report`
    /// pushed for the program entry, while that frame is live:
    /// a rendered backtrace whose stack bottom is this frame
    /// continues with the `lang_start` shim frame std's own
    /// traces show below the entry (see `render_backtrace`).
    /// `None` outside the entry's execution.
    entry_frame_id: std::cell::Cell<Option<u64>>,
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
    /// Memoised per-body stack charges (see
    /// [`Interpreter::frame_stack_bytes`]), keyed by body name --
    /// the key the program's body table itself resolves by.
    frame_stack_sizes: std::cell::RefCell<HashMap<String, u64>>,
    /// The ids of the threads that can still be activated:
    /// a spawned thread whose callable has not been taken yet,
    /// or a started thread currently parked (see
    /// [`ThreadSlot`]). Maintained incrementally where a
    /// slot's `callable` / `parked` is taken, restored or
    /// installed (all in the thread model), so scheduling
    /// rounds, drains and the per-statement yield decision
    /// iterate this set — whose size is the number of *live*
    /// threads — instead of scanning the whole thread table,
    /// whose size is the number of threads *ever spawned* (the
    /// table keeps finished threads' outcomes for `join`). A
    /// stress program that spawns tens of thousands of threads
    /// otherwise pays a full-table scan for every scheduling
    /// decision (issue #399).
    pending_threads: std::cell::RefCell<std::collections::BTreeSet<u64>>,
    /// The ids of the threads whose park token is currently
    /// set: `Thread::unpark` inserts the target's id,
    /// `std::thread::park` / `park_timeout` remove the
    /// executing thread's id when they consume the token (see
    /// the `ThreadPark` builtin). One token per thread, as in
    /// std: repeated unparks before a park coalesce.
    park_notified: std::cell::RefCell<HashSet<u64>>,
    /// The id the next spawned thread gets.
    next_thread: std::cell::RefCell<u64>,
    /// Thread-local destructors registered through std's
    /// destructor-registration entry point (see the
    /// `TlsRegisterDtor` builtin), per thread id, in registration
    /// order: the storage reference and the destructor callable
    /// the registration carried. They run when the thread's code
    /// finishes (see [`Interpreter::run_thread`]).
    tls_dtors: std::cell::RefCell<HashMap<u64, Vec<(Value, Value)>>>,
    /// Signal dispositions installed through the `sigaction`
    /// model (see the `LibcSigaction` builtin), keyed by
    /// signal number.
    signal_actions: std::cell::RefCell<HashMap<i64, SignalAction>>,
    /// Exit handlers registered through the `atexit` model
    /// (see the `LibcAtexit` builtin), in registration
    /// order. They run when the entry's exit runs them (see
    /// `run_exit_handlers`).
    exit_handlers: std::cell::RefCell<Vec<Value>>,
    /// The thread whose code is currently executing (the entry
    /// thread outside any spawned thread's run).
    current_thread: std::cell::Cell<u64>,
    /// The number of detached synchronous drives (see
    /// `detached_drive` in `call`) currently enclosing the
    /// executing code, reset to zero at a spawned thread's root
    /// drive. A suspension is only sound at zero: inside a
    /// detached drive, the host-side continuation waiting for
    /// the callee's return value cannot be saved and resumed.
    sync_nesting: std::cell::Cell<u32>,
    /// A counter bumped on every observable channel-state
    /// change (a queue push or pop, a handle count change):
    /// the cooperative scheduler's progress signal (see
    /// `schedule_round` in `threads`).
    channel_epoch: std::cell::Cell<u64>,
    /// The depth of the innermost active
    /// [`Interpreter::drive`] region (0 outside any drive).
    drive_depth: usize,
    /// The thread segment the scheduler is currently driving, if
    /// any: runtime support that drives guest code outside a
    /// segment (the termination report, a thread's TLS
    /// destructors) runs with no segment marked, and its drives
    /// never yield.
    segment: Option<SegmentMark>,
    /// How many scheduler drains (`drain_threads`) are currently
    /// on the stack. While a drain runs, some thread is blocked
    /// waiting for state another thread may produce, so a driven
    /// segment may yield back to the drain even when no other
    /// thread is runnable: the waiter itself is who the segment
    /// yields to. (An unbounded producer drained by a `recv` on
    /// the entry thread is the shape that needs it: the producer
    /// is the only spawned thread, and without the yield it
    /// would fill the channel until the step budget ends the
    /// run.)
    drain_depth: usize,
    /// mpsc channel states (see [`ChannelState`]), keyed by the id
    /// carried inside the handle values.
    channels: std::cell::RefCell<HashMap<u64, ChannelState>>,
    /// `Wake` dispatch of modelled wakers (see [`WakeDispatch`]),
    /// keyed by the handle allocation's id.
    pub(crate) wake_dispatch: std::cell::RefCell<HashMap<u64, WakeDispatch>>,
    /// The id the next created channel gets.
    next_channel: std::cell::RefCell<u64>,
    /// Open host files of the `std::fs::File` value model (see
    /// `file_value` in `builtins`): the table holds the real
    /// `std::fs::File` each guest handle delegates to, keyed by
    /// the id in the handle value's descriptor slot. Ids start
    /// at 3 (the standard streams' descriptor numbers) and are
    /// never reused, so a stale handle can never alias a newer
    /// file. Interior-mutable for the same reason as the channel
    /// table; each entry is itself a cell because reads and
    /// writes need `&mut` access to one file while the table is
    /// borrowed.
    files: std::cell::RefCell<
        HashMap<u64, std::cell::RefCell<std::fs::File>>,
    >,
    /// Open host sockets of the `std::net` value model (see
    /// `socket_value` in `builtins::net`): the table holds the
    /// real host socket each guest handle delegates to, keyed
    /// by the id in the handle value's descriptor slot. Ids
    /// come from the file counter (`next_file`): files and
    /// sockets share the guest's one descriptor space. Each
    /// entry is itself a cell for the same reason as the file
    /// entries.
    sockets: std::cell::RefCell<
        HashMap<u64, std::cell::RefCell<SocketEntry>>,
    >,
    /// The id the next opened file gets.
    next_file: std::cell::RefCell<u64>,
    /// The guest's errno cell (see `errno_cell_ref` in
    /// `interp::state`): the allocation id of the four-byte
    /// block whose address the errno-location calls hand
    /// out, or `None` until the first use allocates it.
    /// Interior-mutable for the same reason as the file
    /// table.
    errno_cell: std::cell::RefCell<Option<u64>>,
    /// Descriptor reroutings the guest's `dup2` / `close`
    /// calls record (see [`DescRoute`] and the `desc_*`
    /// accessors in `interp::state`): the route of a
    /// descriptor number the guest has rerouted, keyed by
    /// the guest-visible number. A number with no entry
    /// keeps its default route (the standard streams for
    /// 0/1/2, the same-numbered table entry above them).
    /// Interior-mutable for the same reason as the file
    /// table.
    desc_routes: std::cell::RefCell<HashMap<u64, DescRoute>>,
    /// The file-table id of the `/dev/null` entry the
    /// standard-input model hands out while descriptor 0
    /// is closed (see `null_file_id` in `interp::state`):
    /// std reads a closed stdin as end-of-file, which a
    /// null device answers exactly. `None` until first
    /// use. Interior-mutable for the same reason as the
    /// file table.
    null_file: std::cell::RefCell<Option<u64>>,
    /// The allocation id of the main-program handle the
    /// `dlopen` model hands out, allocated on first use
    /// (see `dlopen_main_handle_id` in `interp::state`).
    dlopen_main_handle: std::cell::RefCell<Option<u64>>,
    /// Modelled process commands under construction (see
    /// [`CommandState`]), keyed by the id inside the handle
    /// values. Interior-mutable for the same reason as the
    /// channel table.
    commands: std::cell::RefCell<HashMap<u64, CommandState>>,
    /// The id the next created command gets.
    next_command: std::cell::RefCell<u64>,
    /// Live child processes started through the process host
    /// (see [`ChildState`]), keyed by the id inside the `Child`
    /// handle values. Interior-mutable for the same reason as
    /// the command table.
    children: std::cell::RefCell<HashMap<u64, ChildState>>,
    /// The id the next spawned child gets.
    next_child: std::cell::RefCell<u64>,
    /// Live `std::sync::Barrier` states (see
    /// [`BarrierState`]), keyed by the id inside the barrier
    /// handle values.
    barriers: std::cell::RefCell<HashMap<u64, BarrierState>>,
    /// The id the next created barrier gets.
    next_barrier: std::cell::RefCell<u64>,
    /// Captured `core::fmt::Formatter` states (see
    /// [`FmtCapture`]), keyed by the id inside the formatter
    /// handle values.
    formatters: std::cell::RefCell<HashMap<u64, FmtCapture>>,
    /// The id the next captured formatter gets.
    next_formatter: std::cell::RefCell<u64>,
    /// The one-shot stream arm for captured formatters (see
    /// [`FmtCapture::stream`]): while set, the next capture a
    /// guest `fmt` body dispatch creates streams, and creating it
    /// clears the arm. The detached `Debug` rendering a
    /// `Termination` report performs arms it, so the payload's
    /// own `Debug` body writes through a streaming capture, as
    /// natively; structural composition suppresses the arm while
    /// it composes (its subvalue bodies buffer into the composed
    /// text instead), so no other capture can take it.
    pub(crate) fmt_stream_arm: bool,
    /// `core::fmt` builder states (see [`FmtBuilderState`]),
    /// keyed by the id inside the builder handle values.
    fmt_builders: std::cell::RefCell<HashMap<u64, FmtBuilderState>>,
    /// The id the next created builder gets.
    next_fmt_builder: std::cell::RefCell<u64>,
    /// Live C variable argument lists (see [`VaListState`]),
    /// keyed by the id inside the va-list handle values.
    va_lists: std::cell::RefCell<HashMap<u64, VaListState>>,
    /// The id the next created va list gets.
    next_va_list: std::cell::RefCell<u64>,
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
    /// Allocations whose value a completed drop-glue session has
    /// dropped through a pointer (see the session-completion
    /// marking in `drop_glue`): the storage's value stays in
    /// `allocs` -- a native drop leaves the bytes in place until
    /// the free, and the owning box's destructor body still
    /// reads the pointee's metadata (its `size_of_val`) on its
    /// way to that free -- but drop glue reaching the same
    /// allocation afterwards must find nothing left to drop a
    /// second time, so glue reads (and only glue reads) treat a
    /// recorded allocation as empty. Entries are removed when
    /// the allocation is freed.
    glue_dropped_allocs: std::cell::RefCell<std::collections::HashSet<u64>>,
    /// The reverse registry for synthetic addresses that denote
    /// an allocation in the store: a synthetic address (see
    /// [`synthetic_address`]) -> the allocation id it denotes.
    /// Synthetic addresses are one-way hashes of their target, so
    /// a value the guest can only name by address (the payload of
    /// an `io::Error`'s packed `Custom` representation: the error
    /// word stores `address + tag`, and every later access names
    /// the payload through that integer) needs this map to find
    /// its way back to the stored value. Entries are added when
    /// such an address is minted and removed when the owning
    /// value's drop glue has run, so a live entry always denotes
    /// a live allocation.
    address_owners: std::cell::RefCell<HashMap<u64, u64>>,
    /// Allocations created by the raw allocator builtins
    /// (`__rust_alloc` and friends): their content is an
    /// element tuple (or a zeroed byte block), and a pointer
    /// to the block denotes its first element. Every other
    /// allocation holds one whole pointee value (a box's
    /// contents, a materialised constant), where the same
    /// pointer denotes the value itself.
    elem_blocks: std::cell::RefCell<std::collections::HashSet<u64>>,
    /// Targets of thin pointers that have been exposed as
    /// integers (a `PtrToInt` rvalue, or a scalar transmute
    /// flattening through a pointer shape), keyed by the
    /// synthetic address the exposure produced. Materialising a
    /// pointer from an integer (an `IntToPtr` rvalue, the
    /// wrapped forms, scalar construction through a pointer
    /// shape) restores the recorded target for an address this
    /// table holds, so a pointer -> integer -> pointer round
    /// trip recovers the original target -- a rebuilt `Box`
    /// dereferences to its pointee again -- instead of an
    /// address-only pointer that denotes no readable place. An
    /// address the table does not hold materialises as
    /// [`RefTarget::Address`], as before. Only thin pointers
    /// ([`Value::Ref`]) are recorded: a fat pointer's metadata
    /// has no integer form to round-trip through. Entries are
    /// never removed; allocation ids are never reused, so a
    /// stale entry can never alias a later allocation.
    exposed_targets: std::cell::RefCell<std::collections::HashMap<u64, RefTarget>>,
    /// The function-pointer values whose synthetic address
    /// has been exposed as an integer (see
    /// [`Self::note_exposed_target`]): a function pointer's
    /// integer form denotes the function itself, not a place,
    /// so constructing a function-pointer shape from that
    /// address
    /// recovers the callable value -- the same round-trip
    /// contract `exposed_targets` gives data pointers --
    /// instead of an address-only data reference sitting in
    /// a function-pointer place. Only function identities
    /// (a `Value::FnPtr`, or a non-capturing closure value,
    /// which coerces to a function pointer) are recorded.
    exposed_fn_ptrs:
        std::cell::RefCell<std::collections::HashMap<u64, Value>>,
    /// Seeds of the functions' code images, keyed by body
    /// name (see [`Interpreter::fn_code_deref`]): the hash of
    /// a body's canonical form, from which the deterministic
    /// byte stream a code read serves is generated. Computed
    /// lazily and cached: canonicalising a large body is not
    /// free, and a function's image never changes within a
    /// run.
    code_seeds:
        std::cell::RefCell<std::collections::HashMap<String, u64>>,
    /// The declared type recorded for an allocation at its
    /// creation point (see [`Interpreter::record_alloc_ty`]):
    /// for an ordinary allocation, the pointee type it was
    /// created to hold; for a raw element block, the element
    /// type. A position inside an allocation has no declared
    /// type of its own in this model (a projected place's type
    /// chain stops at the pointer it came through), so this
    /// record is what the `MaybeUninit` entry gate and the
    /// byte-offset denomination consult for allocation
    /// positions, exactly as a home local's declared type and
    /// the program's static-type table serve the same
    /// judgments for place and static positions. Allocations
    /// whose creation point knows no type (the raw allocator
    /// builtins hand out storage for a size, not a type) have
    /// no entry, and their positions keep the value-shape
    /// evidence they are judged by without one.
    alloc_tys: std::cell::RefCell<HashMap<u64, Ty>>,
    /// The id the next created allocation gets.
    next_alloc: std::cell::Cell<u64>,
    /// Pre-decoded bodies for the run loop (see [`crate::decode`]),
    /// decoded lazily per body and keyed by body address.
    decoded: std::cell::RefCell<
        std::collections::HashMap<usize, std::rc::Rc<crate::decode::DecodedBody>>,
    >,
    /// The final state of the owned iterator a consumer builtin
    /// (`collect`, `from_iter`, `fold`, ...) has just driven to
    /// exhaustion, handed to the `CallBuiltin` execution so it
    /// can run the drop glue conversion attached to the call
    /// against it (the callee's real body drops the iterator
    /// when it is done with it). Set by the consumer builtins on
    /// their success path only; cleared before every builtin
    /// call and taken by the caller afterwards, so a state can
    /// never leak into an unrelated call.
    consumed_iter_final: std::cell::RefCell<Option<Value>>,
    /// Resolved ADT instance keys by bare ADT name (see
    /// [`Interpreter::adt_instance_key`]), memoised: the
    /// resolution scans the program's instance tables.
    adt_instance_keys: std::cell::RefCell<HashMap<String, Option<String>>>,
    /// The typed views materialised from type-erased constant
    /// blocks, keyed by the view's rendering and mapping to
    /// the block the view was materialised from (see
    /// [`Interpreter::note_mat_origin`]): address resolution
    /// denotes a pointer to a view by its block, so a
    /// materialising cast preserves the pointer's identity.
    mat_origins: std::cell::RefCell<HashMap<String, Value>>,
}

/// How an assignment's rvalue treats an operand that carries
/// a pointer into the assigning frame (see
/// [`body_stack_charge`]'s address-escape accounting).
#[derive(Clone, Copy, PartialEq, Eq)]
enum PointerFlow {
    /// The rvalue's result is a pointer derived from the
    /// operand (a move, cast, offset or pointer projection):
    /// the flow travels on to the destination local.
    Propagate,
    /// The operand's bits are observed as data, or the pointer
    /// is stored into a constructed value: every frame local
    /// the operand may address materialises.
    Escape,
    /// A modelled operation consumes the pointer in place (a
    /// memcpy's endpoints): the flow neither travels nor
    /// escapes.
    Modelled,
}

/// Note one operand read for the address-escape accounting:
/// when the operand is a plain place naming a local that
/// carries a pointer into the frame (`flow` maps such a local
/// to the frame locals it may address), apply `mode`. A
/// projected place reads through (or into) the root's value,
/// not the pointer itself, and is not a pointer read. Returns
/// whether `flow` or `escaped` changed.
fn note_pointer_operand(
    flow: &mut std::collections::HashMap<usize, Vec<usize>>,
    escaped: &mut Vec<bool>,
    dest: Option<&Place>,
    operand: &Operand,
    mode: PointerFlow,
) -> bool {
    let place = match operand {
        Operand::Copy(place) | Operand::Move(place) => place,
        Operand::Constant(_) => return false,
    };
    if !place.projection.is_empty() {
        return false;
    }
    let Some(targets) = flow.get(&place.local.0).cloned() else {
        return false;
    };
    let mut escape_targets = |escaped: &mut Vec<bool>| {
        let mut changed = false;
        for t in &targets {
            if *t < escaped.len() && !escaped[*t] {
                escaped[*t] = true;
                changed = true;
            }
        }
        changed
    };
    match mode {
        PointerFlow::Modelled => false,
        PointerFlow::Escape => escape_targets(escaped),
        PointerFlow::Propagate => {
            match dest {
                // A plain destination other than the return
                // place keeps the pointer inside the frame's
                // plain locals: the flow travels.
                Some(dest) if dest.projection.is_empty() && dest.local.0 != 0 => {
                    let entry = flow.entry(dest.local.0).or_default();
                    let mut changed = false;
                    for t in targets {
                        if !entry.contains(&t) {
                            entry.push(t);
                            changed = true;
                        }
                    }
                    changed
                }
                // Stored into a projected place, returned in
                // the return place, or consumed with no
                // destination: the address leaves the frame's
                // plain locals.
                _ => escape_targets(escaped),
            }
        }
    }
}

/// The stack charge of one frame of `body`: the total size of
/// the locals that materialise as stack storage in the native
/// frame. A local materialises when it is an argument (the
/// caller marshals it into the frame's argument area), when a
/// call or drop terminator reads it (its value must be
/// marshalled into, or preserved across, the callee -- a
/// modelled builtin callee counts the same: `mem::drop` of a
/// large local really does pass it by value), or when its
/// address escapes the body's modelled pointer flow.
///
/// Every other local is a value in flight between a definition
/// and its uses within the frame: the interpreter holds those
/// as values, and the native compiler likewise keeps them in
/// registers or eliminates them. The measured shape that pins
/// this down is a `vec![zeroed::<[u8; 32_768]>()]` built on a
/// 16 KiB thread stack: the construction pipelines the array
/// through three declared locals of 32 KiB each (the
/// `MaybeUninit` the memset writes into and its two move
/// destinations), yet the native frame is a few hundred bytes,
/// because the value is written straight into the box
/// allocation -- charging every declared local charged that
/// frame 98,529 bytes and reported an overflow the native run
/// never has.
///
/// Address escape: a `Ref` / `Reborrow` rvalue gives a local
/// an address. The address stays inside the frame's modelled
/// flow while it only travels between plain locals through
/// pointer-preserving rvalues (moves, casts, offsets, pointer
/// projections) and is read only by modelled operations -- a
/// `CallBuiltin` terminator's arguments (a `write_bytes`
/// memset writes through it in place, into the value the
/// frame already holds) and `CopyNonOverlapping` (a memcpy
/// between two such pointers). The rooted local materialises
/// as soon as the address leaves that flow: passed to a guest
/// call, stored into a projected place or a constructed
/// aggregate, returned in the return place, converted to an
/// integer, or read as data by anything else. `size_of`
/// resolves each materialised local's size (the
/// conversion-recorded size when there is one, else the
/// declared type's size, else 0).
fn body_stack_charge(body: &Body, size_of: &mut dyn FnMut(usize, &Ty) -> u64) -> u64 {
    let n = body.local_tys.len();
    let mut materialised = vec![false; n];
    let mut mark = |materialised: &mut Vec<bool>, local: usize| {
        if local < n {
            materialised[local] = true;
        }
    };
    // Arguments: locals 1..=arg_count (local 0 is the return
    // place, whose storage is the caller's).
    for local in 1..=body.arg_count {
        mark(&mut materialised, local);
    }
    let operand_root = |operand: &Operand| -> Option<usize> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => Some(place.local.0),
            Operand::Constant(_) => None,
        }
    };
    // Terminator reads: a value a terminator consumes must be
    // marshalled into (or preserved across) the callee.
    for block in &body.blocks {
        match &block.terminator {
            TerminatorKind::Call { args, .. }
            | TerminatorKind::CallBuiltin { args, .. } => {
                for arg in args {
                    if let Some(root) = operand_root(arg) {
                        mark(&mut materialised, root);
                    }
                }
            }
            TerminatorKind::CallIndirect { func, args, .. } => {
                if let Some(root) = operand_root(func) {
                    mark(&mut materialised, root);
                }
                for arg in args {
                    if let Some(root) = operand_root(arg) {
                        mark(&mut materialised, root);
                    }
                }
            }
            TerminatorKind::CallTrait {
                receiver, args, ..
            } => {
                if let Some(root) = operand_root(receiver) {
                    mark(&mut materialised, root);
                }
                if let Some(root) = operand_root(args) {
                    mark(&mut materialised, root);
                }
            }
            TerminatorKind::Drop { place, .. } => {
                mark(&mut materialised, place.local.0)
            }
            _ => {}
        }
    }
    // Address escape: seeds and propagation, to a fixpoint
    // (the flow only grows).
    let mut flow: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    let mut escaped = vec![false; n];
    loop {
        let mut changed = false;
        for block in &body.blocks {
            for stmt in &block.statements {
                match &stmt.kind {
                    StatementKind::Assign(dest, rvalue) => {
                        let dest = Some(dest);
                        match rvalue {
                            Rvalue::Ref(place) | Rvalue::Reborrow(place) => {
                                // The seed: a pointer to `place`'s
                                // root, or -- through a deref --
                                // whatever the dereferenced pointer
                                // itself may address.
                                let targets: Vec<usize> = if place
                                    .projection
                                    .first()
                                    == Some(&ProjectionElem::Deref)
                                {
                                    flow.get(&place.local.0).cloned().unwrap_or_default()
                                } else {
                                    vec![place.local.0]
                                };
                                if targets.is_empty() {
                                    continue;
                                }
                                match dest {
                                    Some(d)
                                        if d.projection.is_empty() && d.local.0 != 0 =>
                                    {
                                        let entry =
                                            flow.entry(d.local.0).or_default();
                                        for t in targets {
                                            if !entry.contains(&t) {
                                                entry.push(t);
                                                changed = true;
                                            }
                                        }
                                    }
                                    _ => {
                                        for t in targets {
                                            if t < n && !escaped[t] {
                                                escaped[t] = true;
                                                changed = true;
                                            }
                                        }
                                    }
                                }
                            }
                            Rvalue::Use(op)
                            | Rvalue::Cast(op, _)
                            | Rvalue::Unsize(op, _)
                            | Rvalue::BoxDerefTransmute(op)
                            | Rvalue::DataPtr(op)
                            | Rvalue::PtrField(op, _)
                            | Rvalue::PtrFieldFirstElem(op, _)
                            | Rvalue::PtrCastUnits(op, _)
                            | Rvalue::PtrFirstElem(op)
                            | Rvalue::PtrArrayBase(op)
                            | Rvalue::PtrFieldBase(op, _)
                            | Rvalue::WrapArrayElems(op)
                            | Rvalue::UnwrapArrayElems(op)
                            | Rvalue::UnwrapSliceElems(op)
                            | Rvalue::WrapSliceElems(op)
                            | Rvalue::BytesToStr(op)
                            | Rvalue::FnPtrToPtr(op)
                            | Rvalue::SliceEnumToInt(op, ..)
                            | Rvalue::SliceIntToEnum(op, ..) => {
                                changed |= note_pointer_operand(
                                    &mut flow,
                                    &mut escaped,
                                    dest,
                                    op,
                                    PointerFlow::Propagate,
                                );
                            }
                            Rvalue::PtrOffset(base, count, ..) => {
                                changed |= note_pointer_operand(
                                    &mut flow,
                                    &mut escaped,
                                    dest,
                                    base,
                                    PointerFlow::Propagate,
                                );
                                changed |= note_pointer_operand(
                                    &mut flow,
                                    &mut escaped,
                                    dest,
                                    count,
                                    PointerFlow::Escape,
                                );
                            }
                            Rvalue::RawPtrAggregate { data, meta } => {
                                changed |= note_pointer_operand(
                                    &mut flow,
                                    &mut escaped,
                                    dest,
                                    data,
                                    PointerFlow::Propagate,
                                );
                                if let Some(meta) = meta {
                                    changed |= note_pointer_operand(
                                        &mut flow,
                                        &mut escaped,
                                        dest,
                                        meta,
                                        PointerFlow::Escape,
                                    );
                                }
                            }
                            Rvalue::AggregateTuple(ops)
                            | Rvalue::CoroutineAggregate(ops) => {
                                for op in ops {
                                    changed |= note_pointer_operand(
                                        &mut flow,
                                        &mut escaped,
                                        dest,
                                        op,
                                        PointerFlow::Escape,
                                    );
                                }
                            }
                            Rvalue::AggregateAdt { operands, .. }
                            | Rvalue::ClosureAggregate { operands, .. } => {
                                for op in operands {
                                    changed |= note_pointer_operand(
                                        &mut flow,
                                        &mut escaped,
                                        dest,
                                        op,
                                        PointerFlow::Escape,
                                    );
                                }
                            }
                            Rvalue::UnaryOp(_, op)
                            | Rvalue::PtrToInt(op)
                            | Rvalue::IntToPtr(op)
                            | Rvalue::Materialize(op, _)
                            | Rvalue::EnumToInt(op, ..)
                            | Rvalue::IntToEnum(op, ..)
                            | Rvalue::Reinterpret(op, ..) => {
                                changed |= note_pointer_operand(
                                    &mut flow,
                                    &mut escaped,
                                    dest,
                                    op,
                                    PointerFlow::Escape,
                                );
                            }
                            Rvalue::IntToPtrWrapped { operand, .. }
                            | Rvalue::ScalarToPtrWrapped { operand, .. }
                            | Rvalue::EnumToPtrWrapped { operand, .. }
                            | Rvalue::IntToEnumWrapped { operand, .. } => {
                                changed |= note_pointer_operand(
                                    &mut flow,
                                    &mut escaped,
                                    dest,
                                    operand,
                                    PointerFlow::Escape,
                                );
                            }
                            Rvalue::ShapedLoad(place, ..)
                            | Rvalue::Len(place)
                            | Rvalue::Discriminant(place, _)
                            | Rvalue::CopyForDeref(place) => {
                                // A place read: through a
                                // projection it reads the root's
                                // value, not the pointer itself; a
                                // plain place naming a pointer
                                // local observes the pointer as
                                // data of some unmodelled kind.
                                if place.projection.is_empty() {
                                    if let Some(targets) =
                                        flow.get(&place.local.0).cloned()
                                    {
                                        for t in targets {
                                            if t < n && !escaped[t] {
                                                escaped[t] = true;
                                                changed = true;
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    StatementKind::CopyNonOverlapping { src, dst, count } => {
                        note_pointer_operand(
                            &mut flow,
                            &mut escaped,
                            None,
                            src,
                            PointerFlow::Modelled,
                        );
                        note_pointer_operand(
                            &mut flow,
                            &mut escaped,
                            None,
                            dst,
                            PointerFlow::Modelled,
                        );
                        changed |= note_pointer_operand(
                            &mut flow,
                            &mut escaped,
                            None,
                            count,
                            PointerFlow::Escape,
                        );
                    }
                    StatementKind::Assume(op) => {
                        changed |= note_pointer_operand(
                            &mut flow,
                            &mut escaped,
                            None,
                            op,
                            PointerFlow::Escape,
                        );
                    }
                    StatementKind::ShapedStore { value, .. } => {
                        changed |= note_pointer_operand(
                            &mut flow,
                            &mut escaped,
                            None,
                            value,
                            PointerFlow::Escape,
                        );
                    }
                    _ => {}
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Terminator uses of a flow pointer: a guest call (or an
    // assertion reading the value) observes the address; a
    // modelled builtin call does not.
    let mut escape_operand = |flow: &std::collections::HashMap<usize, Vec<usize>>,
                              escaped: &mut Vec<bool>,
                              operand: &Operand| {
        if let Operand::Copy(place) | Operand::Move(place) = operand {
            if place.projection.is_empty() {
                if let Some(targets) = flow.get(&place.local.0) {
                    for t in targets.clone() {
                        if t < n {
                            escaped[t] = true;
                        }
                    }
                }
            }
        }
    };
    for block in &body.blocks {
        match &block.terminator {
            TerminatorKind::Call { args, .. } => {
                for arg in args {
                    escape_operand(&flow, &mut escaped, arg);
                }
            }
            TerminatorKind::CallIndirect { func, args, .. } => {
                escape_operand(&flow, &mut escaped, func);
                for arg in args {
                    escape_operand(&flow, &mut escaped, arg);
                }
            }
            TerminatorKind::CallTrait {
                receiver, args, ..
            } => {
                escape_operand(&flow, &mut escaped, receiver);
                escape_operand(&flow, &mut escaped, args);
            }
            TerminatorKind::SwitchInt { discr, .. } => {
                escape_operand(&flow, &mut escaped, discr);
            }
            TerminatorKind::Assert {
                cond, operands, ..
            } => {
                escape_operand(&flow, &mut escaped, cond);
                for op in operands {
                    escape_operand(&flow, &mut escaped, op);
                }
            }
            _ => {}
        }
    }
    for (local, is_escaped) in escaped.iter().enumerate() {
        if *is_escaped {
            mark(&mut materialised, local);
        }
    }
    body.local_tys
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            if materialised[i] {
                size_of(i, ty)
            } else {
                0
            }
        })
        .sum()
}

impl<'a> Interpreter<'a> {
    pub fn new(program: &'a Program) -> Self {
        Interpreter::with_config(program, InterpConfig::default())
    }

    /// The coverage counter values this run has accumulated:
    /// one vector of physical counter values per record of the
    /// program's coverage mapping table (see
    /// [`CapturedRun::coverage_counts`]).
    pub fn coverage_counts(&self) -> Vec<Vec<u64>> {
        self.coverage_counts.borrow().clone()
    }

    /// The guest program's command-line arguments after argv[0],
    /// as configured for this run (see [`InterpConfig::guest_args`]).
    pub fn guest_args(&self) -> &[Vec<u8>] {
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

    /// Whether the entry shim frame appears below the
    /// entry frame in a short backtrace (see the
    /// `entry_shim_visible` field of `InterpConfig`).
    pub fn entry_shim_visible(&self) -> bool {
        self.config.entry_shim_visible
    }

    /// The overflow-checks setting in force for an operation
    /// evaluated on behalf of the call stack `frames`.
    ///
    /// Overflow checks are a per-crate codegen setting: a
    /// generic body marked `#[rustc_inherit_overflow_checks]`
    /// (a core trait implementation, an iterator step) is
    /// compiled into the crate that instantiates it, under
    /// *that* crate's setting. The interpreter holds one body
    /// per instance, shared by every instantiating crate, so
    /// the setting is resolved from the stack instead: walking
    /// from the innermost frame out, the first frame whose
    /// body name carries a recorded crate as a `::` segment
    /// decides (body names are rooted at the main crate, so an
    /// auxiliary crate's bodies appear as
    /// `main::<aux>::<fn>`). Frames of the shared libraries
    /// (std / core / alloc) name no recorded crate and are
    /// stepped over, so a library operation stepped from two
    /// guest crates with different settings follows each
    /// caller in turn. When no frame names a recorded crate,
    /// the session's setting (the main compilation's) applies.
    pub fn overflow_checks_at(&self, frames: &[Frame]) -> bool {
        if self.program.crate_overflow_checks.is_empty() {
            return self.config.overflow_checks;
        }
        for frame in frames.iter().rev() {
            for segment in frame.func.split("::") {
                if let Some((_, on)) = self
                    .program
                    .crate_overflow_checks
                    .iter()
                    .find(|(name, _)| name == segment)
                {
                    return *on;
                }
            }
        }
        self.config.overflow_checks
    }

    /// The tool interception configured for this run, if any
    /// (see [`InterpConfig::tool_interception`]).
    pub fn tool_interception(&self) -> Option<&ToolInterception> {
        self.config.tool_interception.as_ref()
    }

    /// The process host configured for this run, if any
    /// (see [`InterpConfig::process_host`]).
    pub fn process_host(&self) -> Option<&ProcessHost> {
        self.config.process_host.as_ref()
    }

    pub fn with_config(program: &'a Program, config: InterpConfig) -> Self {
        // The runtime initialisation's `SIGPIPE`
        // disposition is part of the process state the
        // entry starts with (see `SigpipeInit`).
        let sigpipe_actions = match config.sigpipe.rt_init_handler() {
            Some(handler) => HashMap::from([(
                SIGPIPE,
                SignalAction::runtime_init(handler),
            )]),
            None => HashMap::new(),
        };
        Interpreter {
            program,
            config,
            steps: 0,
            coverage_counts: std::cell::RefCell::new(
                program
                    .coverage
                    .functions
                    .iter()
                    .map(|f| vec![0; f.num_counters as usize])
                    .collect(),
            ),
            coverage_by_body: program
                .coverage
                .functions
                .iter()
                .enumerate()
                .flat_map(|(i, f)| {
                    f.body_names.iter().map(move |name| (name.clone(), i))
                })
                .collect(),
            next_frame_id: 0,
            string_buffers: std::cell::RefCell::new(
                std::collections::HashMap::new(),
            ),
            pending_string_buffer: std::cell::RefCell::new(None),
            box_layouts: std::cell::RefCell::new(std::collections::HashMap::new()),
            frame_info: std::cell::RefCell::new(std::collections::HashMap::new()),
            body_slots: std::cell::RefCell::new(std::collections::HashMap::new()),
            static_prices: std::cell::RefCell::new(std::collections::HashMap::new()),
            static_cursor: std::cell::Cell::new(STATICS_BASE),
            heap_prices: std::cell::RefCell::new(std::collections::HashMap::new()),
            heap_cursor: std::cell::Cell::new((0, 0)),
            buffer_prices: std::cell::RefCell::new(std::collections::HashMap::new()),
            buffer_cursor: std::cell::Cell::new(BUFFER_BASE),
            alloc_layouts: std::cell::RefCell::new(std::collections::HashMap::new()),
            fat_values: std::cell::RefCell::new(std::collections::HashMap::new()),
            vtable_words: std::cell::RefCell::new(std::collections::HashMap::new()),
            stdout_buffer_modelled: std::cell::Cell::new(false),
            output: String::new(),
            stdout_line_buffer: Vec::new(),
            error_output: String::new(),
            drop_continuations: Vec::new(),
            glue_unwind_panic: None,
            pending_asm_incoming: None,
            pending_asm_outgoing: None,
            resume_signaled: false,
            panic_hook: std::cell::RefCell::new(None),
            alloc_error_hook: std::cell::RefCell::new(None),
            hook_report_ctx: std::cell::RefCell::new(None),
            backtrace_style: std::cell::RefCell::new(None),
            last_panic_stamp: std::cell::RefCell::new(None),
            in_panic_hook: false,
            hook_fired_depth: None,
            panics_in_flight: std::cell::RefCell::new(HashMap::new()),
            panic_note_spent: std::cell::Cell::new(false),
            alloc_failure_reported: std::cell::Cell::new(false),
            panic_report_suppressed: std::cell::Cell::new(false),
            always_abort: std::cell::Cell::new(false),
            hook_abort: std::cell::Cell::new(false),
            last_panic_no_backtrace: std::cell::Cell::new(false),
            last_panic_origin: std::cell::RefCell::new(None),
            thread_stack_base: std::cell::Cell::new(0),
            entry_frame_id: std::cell::Cell::new(None),
            statics: std::cell::RefCell::new(HashMap::new()),
            statics_seeding: std::cell::RefCell::new(HashSet::new()),
            threads: std::cell::RefCell::new(HashMap::new()),
            frame_stack_sizes: std::cell::RefCell::new(HashMap::new()),
            pending_threads: std::cell::RefCell::new(
                std::collections::BTreeSet::new(),
            ),
            park_notified: std::cell::RefCell::new(HashSet::new()),
            next_thread: std::cell::RefCell::new(2),
            tls_dtors: std::cell::RefCell::new(HashMap::new()),
            signal_actions: std::cell::RefCell::new(sigpipe_actions),
            exit_handlers: std::cell::RefCell::new(Vec::new()),
            current_thread: std::cell::Cell::new(1),
            sync_nesting: std::cell::Cell::new(0),
            channel_epoch: std::cell::Cell::new(0),
            drive_depth: 0,
            segment: None,
            drain_depth: 0,
            channels: std::cell::RefCell::new(HashMap::new()),
            wake_dispatch: std::cell::RefCell::new(HashMap::new()),
            next_channel: std::cell::RefCell::new(1),
            files: std::cell::RefCell::new(HashMap::new()),
            sockets: std::cell::RefCell::new(HashMap::new()),
            next_file: std::cell::RefCell::new(3),
            errno_cell: std::cell::RefCell::new(None),
            desc_routes: std::cell::RefCell::new(
                std::collections::HashMap::new(),
            ),
            null_file: std::cell::RefCell::new(None),
            dlopen_main_handle: std::cell::RefCell::new(None),
            commands: std::cell::RefCell::new(HashMap::new()),
            next_command: std::cell::RefCell::new(1),
            children: std::cell::RefCell::new(HashMap::new()),
            next_child: std::cell::RefCell::new(1),
            barriers: std::cell::RefCell::new(HashMap::new()),
            next_barrier: std::cell::RefCell::new(1),
            formatters: std::cell::RefCell::new(HashMap::new()),
            next_formatter: std::cell::RefCell::new(1),
            fmt_stream_arm: false,
            fmt_builders: std::cell::RefCell::new(HashMap::new()),
            next_fmt_builder: std::cell::RefCell::new(1),
            va_lists: std::cell::RefCell::new(HashMap::new()),
            next_va_list: std::cell::RefCell::new(1),
            allocs: std::cell::RefCell::new(HashMap::new()),
            glue_dropped_allocs: std::cell::RefCell::new(
                std::collections::HashSet::new(),
            ),
            consumed_iter_final: std::cell::RefCell::new(None),
            address_owners: std::cell::RefCell::new(HashMap::new()),
            elem_blocks: std::cell::RefCell::new(std::collections::HashSet::new()),
            exposed_targets: std::cell::RefCell::new(
                std::collections::HashMap::new(),
            ),
            exposed_fn_ptrs: std::cell::RefCell::new(
                std::collections::HashMap::new(),
            ),
            code_seeds: std::cell::RefCell::new(
                std::collections::HashMap::new(),
            ),
            alloc_tys: std::cell::RefCell::new(HashMap::new()),
            next_alloc: std::cell::Cell::new(1),
            decoded: std::cell::RefCell::new(std::collections::HashMap::new()),
            adt_instance_keys: std::cell::RefCell::new(HashMap::new()),
            mat_origins: std::cell::RefCell::new(HashMap::new()),
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

    /// The stack charge of one frame of `body`, in bytes: the
    /// total size of the locals the native frame reserves room
    /// for -- the ones that materialise as stack storage (see
    /// [`body_stack_charge`]). A local's size is the one
    /// conversion recorded for it (see `ir::Body::local_sizes`)
    /// when it has one, else the size the declared type resolves
    /// to here; a local neither can size contributes nothing
    /// (the accounting undercharges rather than inventing a
    /// size). Memoised per body name.
    pub(crate) fn frame_stack_bytes(&self, body: &Body) -> u64 {
        if let Some(size) = self.frame_stack_sizes.borrow().get(&body.name) {
            return *size;
        }
        let size = body_stack_charge(body, &mut |i, ty| {
            body.local_sizes
                .get(i)
                .copied()
                .flatten()
                .or_else(|| self.ty_size_full(ty))
                .unwrap_or(0)
        });
        self.frame_stack_sizes
            .borrow_mut()
            .insert(body.name.clone(), size);
        size
    }

    /// The stack budget of the thread currently executing: the
    /// entry thread's is [`InterpConfig::main_stack_size`]; a
    /// spawned thread's is the `stack_size` its
    /// `std::thread::Builder` set, or
    /// [`InterpConfig::thread_stack_size`] by default.
    fn current_thread_stack_limit(&self) -> u64 {
        let id = self.current_thread.get();
        if id != 1 {
            if let Some(slot) = self.threads.borrow().get(&id) {
                if let Some(limit) = slot.stack_limit {
                    return limit;
                }
            }
            return self.config.thread_stack_size as u64;
        }
        self.config.main_stack_size as u64
    }

    /// Charge one frame against the current thread's stack
    /// budget; `charge` is the frame's charge (see
    /// [`Interpreter::frame_stack_bytes`], computed by the
    /// caller first because the body it comes from borrows the
    /// interpreter immutably). The usage is the sum of the
    /// charges of the thread's live frames -- its segment of
    /// the shared frame stack, from `thread_stack_base` up. A
    /// push that would exceed the budget is the guest's stack
    /// overflow: the runtime's report goes to the guest's
    /// stderr in std's shape and the run aborts, exactly as
    /// the native process dies at the guard page (std's stack
    /// probes make a single oversized frame overflow at its
    /// entry the same way). Returns the charge, for the caller
    /// to store in the frame it pushes.
    pub(crate) fn charge_stack(
        &mut self,
        frames: &[Frame],
        charge: u64,
    ) -> InterpResult<u64> {
        let base = self.thread_stack_base.get().min(frames.len());
        let used: u64 = frames[base..].iter().map(|f| f.stack_bytes).sum();
        if used + charge > self.current_thread_stack_limit() {
            let name = self
                .thread_name(self.current_thread.get())
                .unwrap_or_else(|| "<unknown>".to_string());
            self.push_error_output(&format!(
                "\nthread '{name}' has overflowed its stack\n\
                 fatal runtime error: stack overflow, aborting\n"
            ));
            return Err(InterpError::Abort {
                message: "stack overflow".to_string(),
            });
        }
        Ok(charge)
    }

    pub(crate) fn push_output(&mut self, text: &str) {
        self.output.push_str(text);
    }

    /// Write bytes to the guest's standard output: the
    /// captured stream records them, as always. A
    /// write-through sink (see [`InterpConfig::stdout_sink`])
    /// receives them under std's line-buffering discipline
    /// (see `stdout_line_buffer`): through the last newline
    /// the write carried, or when the buffer fills — and a
    /// stream error there is the outcome, the error a native
    /// run's write would have produced (a broken pipe above
    /// all).
    pub(crate) fn write_stdout(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        // The descriptor routing decides where the
        // guest's standard output currently goes: a
        // reroute onto a file (a `dup2` onto descriptor 1)
        // sends the bytes to that file and nowhere else,
        // and a closed descriptor 1 discards them -- std's
        // own answer to a write on an invalid descriptor
        // is success with the bytes dropped (see
        // `write_stdout_partial`).
        match self.desc_route(1) {
            DescRoute::File(id) => {
                use std::io::Write as _;
                return self
                    .with_file(id, |file| file.write_all(bytes))
                    .unwrap_or_else(|| {
                        Err(std::io::Error::from_raw_os_error(9))
                    });
            }
            DescRoute::Closed => return Ok(()),
            DescRoute::Stdout => {}
            _ => {
                return Err(std::io::Error::from_raw_os_error(9));
            }
        }
        self.push_output(&String::from_utf8_lossy(bytes));
        if self.config.stdout_sink.is_none() {
            return Ok(());
        }
        self.stdout_line_buffer.extend_from_slice(bytes);
        if bytes.contains(&b'\n') {
            let last = self
                .stdout_line_buffer
                .iter()
                .rposition(|b| *b == b'\n')
                .expect("a newline was just appended");
            self.stdout_sink_write_through(last + 1)?;
        } else if self.stdout_line_buffer.len() >= 1024 {
            // std's buffer capacity: a full buffer goes out
            // whole (a write larger than the buffer bypasses
            // it the same way, after what was buffered).
            let len = self.stdout_line_buffer.len();
            self.stdout_sink_write_through(len)?;
        }
        Ok(())
    }

    /// The single-`write` form of [`Self::write_stdout`]:
    /// the count is the bytes the guest handed over, as a
    /// native `write` on a line-buffered stream reports the
    /// count it accepted into the buffer.
    pub(crate) fn write_stdout_partial(
        &mut self,
        bytes: &[u8],
    ) -> std::io::Result<usize> {
        // A reroute of descriptor 1 onto a file accepts
        // what the file accepts (a single `write` may come
        // back short); the captured stream and a closed
        // descriptor 1 accept everything handed over (std
        // reports a write on an invalid descriptor as
        // fully written, dropping the bytes).
        if let DescRoute::File(id) = self.desc_route(1) {
            use std::io::Write as _;
            return self
                .with_file(id, |file| file.write(bytes))
                .unwrap_or_else(|| {
                    Err(std::io::Error::from_raw_os_error(9))
                });
        }
        self.write_stdout(bytes)?;
        Ok(bytes.len())
    }

    /// Write bytes to the guest's standard error, under
    /// the descriptor routing as `write_stdout` does for
    /// descriptor 1: the captured error stream records
    /// them, a reroute onto a file sends them there, and
    /// a closed descriptor 2 drops them with success, as
    /// std answers a write on an invalid descriptor. The
    /// count is the bytes accepted.
    pub(crate) fn write_stderr_bytes(
        &mut self,
        bytes: &[u8],
    ) -> std::io::Result<usize> {
        match self.desc_route(2) {
            DescRoute::Stderr => {
                self.push_error_output(
                    &String::from_utf8_lossy(bytes),
                );
                Ok(bytes.len())
            }
            DescRoute::File(id) => {
                use std::io::Write as _;
                self.with_file(id, |file| file.write(bytes))
                    .unwrap_or_else(|| {
                        Err(std::io::Error::from_raw_os_error(9))
                    })
            }
            DescRoute::Closed => Ok(bytes.len()),
            _ => Err(std::io::Error::from_raw_os_error(9)),
        }
    }

    /// Send the first `n` buffered stdout bytes to the
    /// write-through sink, keeping the rest buffered.
    fn stdout_sink_write_through(&mut self, n: usize) -> std::io::Result<()> {
        let sink = self
            .config
            .stdout_sink
            .as_ref()
            .expect("a sink is installed")
            .clone();
        let mut file = &*sink.0;
        std::io::Write::write_all(&mut file, &self.stdout_line_buffer[..n])?;
        self.stdout_line_buffer.drain(..n);
        Ok(())
    }

    /// Flush the guest's standard output: the line buffer
    /// goes out whole and the sink is flushed, with the
    /// sink's error as the outcome. Without a sink the
    /// captured stream is unbuffered and there is nothing
    /// to do.
    pub(crate) fn flush_stdout(&mut self) -> std::io::Result<()> {
        if self.desc_route(1) != DescRoute::Stdout {
            // A rerouted descriptor 1 is written unbuffered
            // (see `write_stdout`), and a closed one holds
            // nothing: there is no line buffer to settle.
            return Ok(());
        }
        if self.config.stdout_sink.is_none() {
            return Ok(());
        }
        let len = self.stdout_line_buffer.len();
        if len > 0 {
            self.stdout_sink_write_through(len)?;
        }
        let sink = self
            .config
            .stdout_sink
            .as_ref()
            .expect("a sink is installed")
            .clone();
        let mut file = &*sink.0;
        std::io::Write::flush(&mut file)
    }

    /// End-of-run stdout settlement: whatever the line
    /// buffer still holds goes to the sink, as std flushes
    /// the stream when a program ends. An error here has no
    /// guest left to observe it and is dropped.
    pub fn finish_stdout(&mut self) {
        let _ = self.flush_stdout();
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

    /// Whether the segment currently being driven must yield to
    /// the scheduler at this statement boundary. True only when
    /// the segment's quantum has expired, the drive is the
    /// segment's own drive loop (not a nested drive a builtin
    /// started, whose interpreter-side continuation parking would
    /// lose), no guest panic is mid-unwind inside the segment
    /// (the unwinder's state lives in the interpreter's own call
    /// stack and cannot be parked), no debugger session is
    /// attached (a session observes one thread's positions at a
    /// time), and at least one other thread can run against the
    /// current frame stack. A single-threaded program therefore
    /// never yields, and its verdicts are unchanged. The scans
    /// run only once the quantum has actually expired, so the
    /// per-statement cost is one subtraction.
    fn yield_due(&self, frames: &[Frame], stop_depth: usize) -> bool {
        let Some(mark) = self.segment else {
            return false;
        };
        if self.drive_depth != mark.depth {
            return false;
        }
        if self.steps.wrapping_sub(mark.steps0) < self.config.thread_quantum {
            return false;
        }
        if self.debug_ctl.is_some() || self.glue_unwind_panic.is_some() {
            return false;
        }
        if frames[stop_depth..]
            .iter()
            .any(|f| f.in_cleanup || f.unwind_payload.is_some())
        {
            return false;
        }
        // A drain on the stack means a waiter: the segment may
        // yield back to it even if no other thread is runnable.
        self.drain_depth > 0 || self.other_thread_runnable(frames, mark.thread)
    }

    /// Whether a thread other than `except` could run against
    /// the current frame stack: it has not started yet (its
    /// callable is still captured), or it is parked and every
    /// frame it was suspended above is still on the stack (see
    /// [`ParkedThread`]).
    fn other_thread_runnable(&self, frames: &[Frame], except: u64) -> bool {
        // Only live threads can be runnable, so walk the
        // pending set (see the `pending_threads` field) rather
        // than the whole thread table.
        let candidates: Vec<u64> = self
            .pending_threads
            .borrow()
            .iter()
            .copied()
            .filter(|id| *id != except)
            .collect();
        let threads = self.threads.borrow();
        candidates.iter().any(|id| {
            let Some(slot) = threads.get(id) else {
                return false;
            };
            slot.callable.is_some()
                || slot.parked.as_ref().is_some_and(|parked| {
                    parked
                        .below
                        .iter()
                        .all(|fid| frames.iter().any(|f| f.id == *fid))
                })
        })
    }

    fn fresh_frame_id(&mut self) -> u64 {
        let id = self.next_frame_id;
        self.next_frame_id += 1;
        id
    }
}

pub fn interpret(program: &Program, entry: &str, args: Vec<Value>) -> InterpResult<Value> {
    Interpreter::new(program).run(entry, args)
}
