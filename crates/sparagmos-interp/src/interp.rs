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
use crate::ir::*;
use crate::value::{PanicPayload, RefProj, RefTarget, Value};
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
}

impl Default for InterpConfig {
    fn default() -> Self {
        InterpConfig {
            max_steps: 1_000_000,
            max_call_depth: 1024,
            guest_args: Vec::new(),
            program_path: String::new(),
        }
    }
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
    Value(Value),
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
    pub outcome: Option<Result<Value, String>>,
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
    /// Everything the guest wrote to its stdout (via the runtime
    /// print support) during the most recent run.
    output: String,
    /// Everything the guest wrote to its stderr (via the runtime
    /// print support) during the most recent run.
    error_output: String,
    /// Drop glue suspended on destructor calls (see
    /// [`DropContinuation`]); a stack, innermost last.
    drop_continuations: Vec<DropContinuation>,
    /// Set by the `Resume` terminator right before it returns the
    /// in-flight panic: the unwinder (see [`Interpreter::drive`])
    /// reads and clears it to tell a completed cleanup (the panic
    /// continues in the caller) from a fresh panic raised directly
    /// in a cleanup frame (which aborts).
    resume_signaled: bool,
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
    /// The thread whose code is currently executing (the entry
    /// thread outside any spawned thread's run).
    current_thread: std::cell::Cell<u64>,
    /// mpsc channel states (see [`ChannelState`]), keyed by the id
    /// carried inside the handle values.
    channels: std::cell::RefCell<HashMap<u64, ChannelState>>,
    /// The id the next created channel gets.
    next_channel: std::cell::RefCell<u64>,
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
    /// The id the next created allocation gets.
    next_alloc: std::cell::Cell<u64>,
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

    pub fn with_config(program: &'a Program, config: InterpConfig) -> Self {
        Interpreter {
            program,
            config,
            steps: 0,
            next_frame_id: 0,
            output: String::new(),
            error_output: String::new(),
            drop_continuations: Vec::new(),
            resume_signaled: false,
            statics: std::cell::RefCell::new(HashMap::new()),
            statics_seeding: std::cell::RefCell::new(HashSet::new()),
            threads: std::cell::RefCell::new(HashMap::new()),
            next_thread: std::cell::RefCell::new(2),
            current_thread: std::cell::Cell::new(1),
            channels: std::cell::RefCell::new(HashMap::new()),
            next_channel: std::cell::RefCell::new(1),
            allocs: std::cell::RefCell::new(HashMap::new()),
            next_alloc: std::cell::Cell::new(1),
        }
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

    fn tick(&mut self) -> InterpResult<()> {
        self.steps += 1;
        if self.steps > self.config.max_steps {
            return Err(InterpError::StepLimitExceeded {
                limit: self.config.max_steps,
            });
        }
        Ok(())
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
        self.drive_frames(&mut frames, 0)
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
        loop {
            match self.drive_loop(frames, stop_depth) {
                Ok(value) => return Ok(value),
                Err(InterpError::Panic { message, payload }) => {
                    let resumed = std::mem::take(&mut self.resume_signaled);
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
                        let body = self.program.get(&func).cloned();
                        let action = body
                            .as_ref()
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
                                Self::jump(frames, &body, target)?;
                                break;
                            }
                            crate::ir::UnwindAction::Continue => {
                                self.pop_unwind_frame(frames);
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
    /// drop glue that belonged to it (its destructors never resume:
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
        loop {
            // Fetch the current block's next unit of work, cloning the small
            // pieces we need so the frame stack can be mutated freely after.
            let (func_name, block_idx, stmt_idx) = {
                let frame = frames
                    .last()
                    .expect("frame stack is non-empty while running");
                (frame.func.clone(), frame.block, frame.stmt)
            };
            // Clone the body for this iteration: the oracle favours clarity
            // over speed, and owning the body lets the frame stack and the
            // step counter be mutated without borrow conflicts.
            let body = self.program.get(&func_name).cloned().ok_or_else(|| {
                InterpError::UnknownFunction {
                    name: func_name.clone(),
                }
            })?;
            let body = &body;
            let block = body
                .blocks
                .get(block_idx)
                .ok_or_else(|| InterpError::UnknownBlock {
                    func: func_name.clone(),
                    block: block_idx,
                })?;

            if stmt_idx < block.statements.len() {
                let statement = block.statements[stmt_idx].clone();
                // Advance the program counter before executing, so a call
                // (not possible in a statement) or an error leaves sane state.
                frames.last_mut().expect("frame").stmt += 1;
                self.tick()?;
                self.exec_statement(frames, body, &statement)?;
                continue;
            }

            // Terminator.
            let terminator = block.terminator.clone();
            self.tick()?;
            match terminator {
                TerminatorKind::Goto { target } => {
                    Self::jump(frames, body, target)?;
                }
                TerminatorKind::SwitchInt {
                    discr,
                    targets,
                    otherwise,
                } => {
                    let value = self.eval_operand(frames, body, &discr)?;
                    let bits = value.switch_bits().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "SwitchInt discriminant",
                            "integer or bool",
                            value.kind_name(),
                        )
                    })?;
                    let target = targets
                        .iter()
                        .find(|(pattern, _)| *pattern == bits)
                        .map(|(_, bb)| *bb)
                        .unwrap_or(otherwise);
                    Self::jump(frames, body, target)?;
                }
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    target,
                } => {
                    let mut arg_values = Vec::with_capacity(args.len());
                    for arg in &args {
                        arg_values.push(self.eval_operand(frames, body, arg)?);
                    }
                    self.start_call(frames, &func, arg_values, &destination, target)?;
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
                    let (callee, env_ref) = self.eval_callable(&frames, body, &func)?;
                    let env_ref = match self.consumed_env_arg(frames, body, &func, &callee)? {
                        Some(by_value) => by_value,
                        None => env_ref,
                    };
                    let mut arg_values = Vec::with_capacity(args.len());
                    for arg in &args {
                        arg_values.push(self.eval_operand(frames, body, arg)?);
                    }
                    match callee {
                        Value::FnPtr(name) => {
                            self.start_call(
                                frames,
                                &name,
                                arg_values,
                                &destination,
                                target,
                            )?;
                        }
                        Value::Closure { def, .. } => {
                            let mut full = vec![env_ref];
                            full.extend(arg_values);
                            self.start_call(frames, &def, full, &destination, target)?;
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
                    destination,
                    target,
                } => {
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
                            let (value, env_ref) = self.eval_callable(&frames, body, &receiver)?;
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
                                Value::Ref(RefTarget::Value(Box::new(value.clone())))
                            } else {
                                env_ref
                            };
                            let env_ref = match self.consumed_env_arg(
                                frames,
                                body,
                                &receiver,
                                &value,
                            )? {
                                Some(by_value) => by_value,
                                None => env_ref,
                            };
                            (value, env_ref)
                        }
                        FnCallKind::Call | FnCallKind::CallMut => {
                            let recv = self.eval_operand(frames, body, &receiver)?;
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
                                        let env = Value::Ref(RefTarget::Value(
                                            Box::new(value.clone()),
                                        ));
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
                                        Value::Ref(RefTarget::Value(Box::new(
                                            value.clone(),
                                        )))
                                    } else {
                                        self.callable_env_ref(
                                            &frames,
                                            body,
                                            &receiver,
                                            &value,
                                        )?
                                    };
                                    (value, env)
                                }
                            }
                        }
                    };
                    match callable {
                        Value::Closure { def, .. } => {
                            let mut full = vec![env_ref];
                            full.extend(arg_values);
                            self.start_call(frames, &def, full, &destination, target)?;
                        }
                        Value::FnPtr(name) => {
                            self.start_call(
                                frames,
                                &name,
                                arg_values,
                                &destination,
                                target,
                            )?;
                        }
                        other => {
                            // A dyn receiver whose concrete value
                            // is an ADT value: a hand-written Fn
                            // impl on that type, when the program
                            // has its body, is the callable.
                            if let Some(concrete) = dyn_concrete.clone() {
                                if let Some(result) = self.dyn_adt_fn_call(
                                    frames,
                                    kind,
                                    &concrete,
                                    other.clone(),
                                    env_ref,
                                    arg_values,
                                    &destination,
                                    target,
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
                                    Resolved::Value(_) => {
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
                                RefTarget::Value(Box::new(value))
                            }
                        };
                        self.write_place(frames, body, &destination, Value::Unit)?;
                        let work = vec![(Vec::new(), glue.root.clone())];
                        self.run_drop_glue(frames, root, work, &glue.adts, bb)?;
                    } else {
                        let mut result = self.exec_builtin(
                            frames,
                            body,
                            &name,
                            &generic_args,
                            &args,
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
                            result = self.eval_unsize(frames, result, &meta)?;
                        }
                        match target {
                            Some(bb) => {
                                self.write_place(frames, body, &destination, result)?;
                                Self::jump(frames, body, bb)?;
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
                    let resolved = self.resolve_place(&frames, body, &place)?;
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
                        Resolved::Value(value) => {
                            // The place is the deref of a value
                            // reference (a `Box` / `ManuallyDrop`
                            // deref in this model): the glue runs
                            // against that value itself, as it does
                            // for a `mem::drop` constant argument.
                            RefTarget::Value(Box::new(value))
                        }
                    };
                    let work = vec![(Vec::new(), glue.root.clone())];
                    self.run_drop_glue(frames, root, work, &glue.adts, target)?;
                }
                TerminatorKind::Resume => {
                    // The cleanup chain of the top frame is done:
                    // pop the frame and report the panic stashed in
                    // it when the unwinder entered the cleanup. The
                    // unwinder in `drive` continues the panic in
                    // the caller at its call site.
                    let frame = frames.pop().expect("frame");
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
                        continue;
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
                        return Ok(value);
                    }
                    let dest = frame
                        .return_dest
                        .clone()
                        .expect("non-entry frame has a destination");
                    let resume = frame.return_block;
                    let caller_name = frames.last().expect("caller").func.clone();
                    let caller_body = self.program.get(&caller_name).cloned().ok_or_else(|| {
                        InterpError::UnknownFunction {
                            name: caller_name.clone(),
                        }
                    })?;
                    self.write_place(frames, &caller_body, &dest, value)?;
                    match resume {
                        Some(bb) => Self::jump(frames, &caller_body, bb)?,
                        None => {
                            return Err(InterpError::MissingContinuation { func: caller_name });
                        }
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
                } => {
                    let value = self.eval_operand(frames, body, &cond)?;
                    let actual = value.as_bool().ok_or_else(|| {
                        InterpError::type_mismatch("Assert condition", "bool", value.kind_name())
                    })?;
                    if actual != expected {
                        return Err(InterpError::panic(message));
                    }
                    Self::jump(frames, body, target)?;
                }
                TerminatorKind::Unsupported(kind) => {
                    return Err(InterpError::UnsupportedTerminator { kind });
                }
            }
        }
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
        let Value::DynRef {
            base,
            concrete,
            traits,
            ..
        } = args.first()?
        else {
            return None;
        };
        let (trait_path, method) = func.rsplit_once("::")?;
        if trait_path.is_empty() || method.is_empty() {
            return None;
        }
        let candidate = format!("<{concrete} as {trait_path}>::{method}");
        if self.program.get(&candidate).is_some()
            || self.program.unconverted_reason(&candidate).is_some()
        {
            return Some(Ok((candidate, (**base).clone())));
        }
        // The impl key may carry the trait's generic arguments
        // (`<S as std::ops::FnMut<(i32,)>>::call_mut`) or spell the
        // trait through the other of `std::` / `core::`: scan the
        // program's keys for the unique matching impl body.
        if let Some(found) = self.scan_virtual_impl(concrete, trait_path, method) {
            return Some(Ok((found, (**base).clone())));
        }
        if traits.iter().any(|t| t == trait_path) {
            return Some(Err(InterpError::UnsupportedDynamicDispatch {
                method: func.to_string(),
                concrete: concrete.clone(),
            }));
        }
        None
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
        // A `#[track_caller]` callee (marked at conversion) may
        // arrive with one extra argument: the caller-location
        // stamp the driver appended to the call. Pop it before
        // anything else inspects the arguments; it is bound to
        // the callee's synthetic location local below (and simply
        // dropped when the call turns out to be a vtable
        // dispatch to an unmarked impl body, whose location the
        // model does not track).
        let mut arg_values = arg_values;
        let stamp = match self.program.get(func) {
            Some(callee)
                if callee.caller_location_local.is_some()
                    && arg_values.len() == callee.arg_count + 1 =>
            {
                arg_values.pop()
            }
            _ => None,
        };
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
                    return self.start_call(frames, &impl_func, args, destination, target);
                }
                Err(e) => return Err(e),
            },
            None => match self.program.get(func) {
                Some(c) => c.clone(),
                None => return Err(self.missing_body_error(func)),
            },
        };
        if frames.len() >= self.config.max_call_depth {
            return Err(InterpError::CallDepthExceeded {
                limit: self.config.max_call_depth,
            });
        }
        // When the calling frame is itself a track-caller frame,
        // the location forwards — the stamp is replaced by the
        // caller's own received location — matching the
        // compiler's propagation (a call written in a plain
        // function inlined into a track-caller body is the one
        // case this runtime forwarding cannot tell apart; the
        // stamp then still names a location in the call chain,
        // one frame further out).
        if arg_values.len() != callee.arg_count {
            return Err(InterpError::type_mismatch(
                format!("call to `{func}`"),
                format!("{} argument(s)", callee.arg_count),
                format!("{} argument(s)", arg_values.len()),
            ));
        }
        let mut locals = vec![Value::Uninit; callee.local_count()];
        for (i, v) in arg_values.into_iter().enumerate() {
            locals[i + 1] = v;
        }
        if let (Some(idx), Some(mut stamp)) = (callee.caller_location_local, stamp) {
            if let Some(caller) = frames.last() {
                if let Some(caller_idx) = self
                    .program
                    .get(&caller.func)
                    .and_then(|b| b.caller_location_local)
                {
                    stamp = caller.locals[caller_idx].clone();
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
                Some(b) => b.clone(),
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
    fn scan_virtual_impl(
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
            _ => Value::Ref(RefTarget::Value(Box::new(callable.clone()))),
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
                    Resolved::Value(_) => {
                        Ok(Value::Ref(RefTarget::Value(Box::new(value.clone()))))
                    }
                    Resolved::Static { name, projection } => {
                        Ok(Value::Ref(RefTarget::Static { name, projection }))
                    }
                    Resolved::Alloc { id, projection } => {
                        Ok(Value::Ref(RefTarget::Alloc { id, projection }))
                    }
                }
            }
            Operand::Constant(_) => Ok(Value::Ref(RefTarget::Value(Box::new(value.clone())))),
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
                match project_value(base, &projection, "<drop>", "drop glue") {
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
                let base = if base == Value::Uninit {
                    self.zst_glue_value(frame, *local).unwrap_or(base)
                } else {
                    base
                };
                (base, projection)
            }
            RefTarget::Value(v) => ((**v).clone(), &[]),
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
        for step in prefix.iter().chain(path.iter()) {
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
            match project_value(value, std::slice::from_ref(step), "<drop>", "drop glue") {
                Ok(v) => value = v,
                Err(InterpError::UninitializedRead { .. }) => return Ok(None),
                Err(e) => return Err(e),
            }
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
                RefTarget::Value(v) => {
                    let projected =
                        project_value(*v, std::slice::from_ref(step), "<drop>", "drop glue")?;
                    RefTarget::Value(Box::new(projected))
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

    /// Execute drop glue: process `work` until it is empty or a
    /// destructor call has to run. A destructor call pushes its
    /// frame plus a [`DropContinuation`] and returns; the `Return`
    /// terminator resumes the remaining work when the destructor
    /// finishes. When the work is done, the dropped place's own slot
    /// is cleared (for a direct local of the owner frame) and the
    /// owner frame jumps to `target`.
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
                        let items = match value {
                            Value::Tuple(items) => items,
                            Value::Enum { fields, .. } => fields,
                            // A closure's fields are its captures.
                            Value::Closure { env, .. } => env,
                            Value::Coroutine { fields, .. } => fields,
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "drop glue",
                                    "a tuple or ADT value",
                                    other.kind_name(),
                                ));
                            }
                        };
                        if items.len() != fields.len()
                            && !(is_coroutine && items.len() < fields.len())
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
                        let variant = match value {
                            Value::Enum { variant, .. } => variant,
                            Value::Coroutine { variant, .. } => variant,
                            Value::Tuple(_) => 0,
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "drop glue",
                                    "an ADT value",
                                    other.kind_name(),
                                ));
                            }
                        };
                        let plan = variants
                            .get(variant as usize)
                            .cloned()
                            .unwrap_or(P::Unknown);
                        active.work.push((path, plan));
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
                        let Some(id) = rc_handle_id(&handle) else {
                            return Err(InterpError::type_mismatch(
                                "reference-counted handle drop glue",
                                "an Rc/Arc handle",
                                handle.kind_name(),
                            ));
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
                        if let Some(value) = self.read_at(frames, &active.root, &path)? {
                            drop_dyn_value(&value)?;
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
                                RefTarget::Value(Box::new(fields[0].clone()))
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
                        let callee = self.program.get(&func).cloned().ok_or_else(|| {
                            InterpError::UnknownFunction { name: func.clone() }
                        })?;
                        if frames.len() >= self.config.max_call_depth {
                            return Err(InterpError::CallDepthExceeded {
                                limit: self.config.max_call_depth,
                            });
                        }
                        if callee.arg_count != 1 || callee.local_count() < 2 {
                            return Err(InterpError::type_mismatch(
                                format!("destructor `{func}`"),
                                "one argument",
                                format!("{} argument(s)", callee.arg_count),
                            ));
                        }
                        let target_ref = self.extend_ref(frames, &active.root, &path)?;
                        let mut locals = vec![Value::Uninit; callee.local_count()];
                        locals[1] = Value::Ref(target_ref);
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
            match active.completion {
                GlueCompletion::Jump(target) => {
                    // Glue finished: the dropped place is dead, as in
                    // MIR. A direct local of the owner frame is
                    // cleared.
                    // A dropped box's allocation is freed with it:
                    // the pointee's glue has run, and the storage
                    // the box pointed at is gone, as a real free
                    // makes it. (A box nested inside a larger
                    // dropped value is reached as a work item of
                    // that value's session, not as a session root;
                    // its allocation entry is left in the store —
                    // ids are never reused, so nothing can observe
                    // the difference within a run.)
                    if let Ok(Some(value)) = self.read_at(frames, &active.root, &[]) {
                        if let Some((id, _)) = box_alloc_pointer(&value) {
                            self.free_alloc(id);
                        }
                    }
                    if let RefTarget::Place {
                        frame,
                        local,
                        projection,
                    } = &active.root
                    {
                        if projection.is_empty() {
                            let frame_pos = self.frame_pos(frames, *frame)?;
                            if let Some(slot) = frames[frame_pos].locals.get_mut(*local) {
                                *slot = Value::Uninit;
                            }
                        }
                    }
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
                GlueCompletion::Resume => match suspended.pop() {
                    Some(session) => {
                        active = session;
                    }
                    None => {
                        return Err(InterpError::type_mismatch(
                            "drop glue",
                            "an enclosing glue session to resume",
                            "none",
                        ));
                    }
                },
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
    pub(crate) fn thread_outcome(&self, id: u64) -> Option<Result<Value, String>> {
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
        self.current_thread.set(previous);
        let outcome = match result {
            Ok(value) => Ok(value),
            Err(InterpError::Panic { message, .. }) => Err(message),
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

    /// Join a thread: run it if it is still pending, then return
    /// its recorded outcome.
    pub(crate) fn join_thread(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
    ) -> InterpResult<Result<Value, String>> {
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
                    Value::Ref(RefTarget::Value(Box::new(callable.clone())))
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

    /// Create a channel state (one sender, one receiver) and
    /// return its id.
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
    fn alloc_base(&self, id: u64) -> InterpResult<Value> {
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
    fn free_alloc(&self, id: u64) {
        self.allocs.borrow_mut().remove(&id);
    }

    /// Set the discriminant of the ADT value `value` to `variant`,
    /// as a `SetDiscriminant` statement does: a coroutine keeps its
    /// state fields (they are the shared storage the transform
    /// writes before the tag moves), an enum's old payload is
    /// invalidated.
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

    fn exec_statement(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        statement: &Statement,
    ) -> InterpResult<()> {
        match &statement.kind {
            StatementKind::Assign(place, rvalue) => {
                let dest_ty = body.local_ty(place.local).cloned();
                let value = self.eval_rvalue(frames, body, rvalue, dest_ty.as_ref())?;
                self.write_place(frames, body, place, value)
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
                    Resolved::Value(_) => Err(InterpError::type_mismatch(
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
                self.exec_copy_nonoverlapping(frames, src_value, dst_value, count_value)
            }
            StatementKind::Unsupported(kind) => Err(InterpError::UnsupportedStatement {
                kind: kind.clone(),
            }),
        }
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
    pub(crate) fn exec_copy_nonoverlapping(
        &self,
        frames: &mut Vec<Frame>,
        src_value: Value,
        dst_value: Value,
        count_value: Value,
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
                // a single element, which only a count of 1 can copy.
                let aggregate = |v: &Value| {
                    matches!(
                        v,
                        Value::Tuple(_) | Value::FatRef { .. } | Value::Bytes(_) | Value::Str(_)
                    )
                };
                let src_base = self.read_resolved(frames, &src_target)?;
                let elems: Vec<Value> = if n == 1 && !aggregate(&src_base) {
                    vec![src_base]
                } else {
                    let mut elems = Vec::with_capacity(n);
                    for i in 0..n {
                        elems.push(self.read_resolved_at(frames, &src_target, i)?);
                    }
                    elems
                };
                // An unreadable (not yet written) destination is a
                // buffer being filled: the indexed path is the one
                // that can create its elements.
                let dst_aggregate = self
                    .read_resolved(frames, &dst_target)
                    .map(|v| aggregate(&v))
                    .unwrap_or(true);
                if n == 1 && !dst_aggregate {
                    let value = elems.into_iter().next().expect("one element");
                    self.write_resolved(frames, &dst_target, value)?;
                } else {
                    for (i, value) in elems.into_iter().enumerate() {
                        self.write_resolved_at(frames, &dst_target, i, value)?;
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
            Resolved::Value(value) => {
                Value::Ref(RefTarget::Value(Box::new(value.clone())))
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
        for elem in &place.projection {
            cur = match elem {
                ProjectionElem::Field(idx) => {
                    fat_cap = None;
                    self.project_step(frames, cur, RefProj::Field(*idx), body, place)?
                }
                ProjectionElem::Index(local) => {
                    let idx = self.read_index_local(frames, body, *local, place)?;
                    if let Some(cap) = fat_cap {
                        if idx as u128 >= cap {
                            return Err(index_oob(cap as usize, idx));
                        }
                    }
                    fat_cap = None;
                    self.project_step(frames, cur, RefProj::Index(idx), body, place)?
                }
                ProjectionElem::ConstantIndex { offset, from_end } => {
                    let len = self
                        .read_resolved(frames, &cur)?
                        .len()
                        .ok_or_else(|| {
                            InterpError::type_mismatch(
                                format!("constant index into {place}"),
                                "array, slice or string value",
                                "value without a length",
                            )
                        })?;
                    let len = fat_cap.unwrap_or(len);
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
                    self.project_step(frames, cur, RefProj::Index(idx), body, place)?
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
                    let base_len = self.read_resolved(frames, &cur)?.len().ok_or_else(|| {
                        InterpError::type_mismatch(
                            format!("subslice of {place}"),
                            "array, slice or string value",
                            "value without a length",
                        )
                    })?;
                    let len = fat_cap.unwrap_or(base_len);
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
                    self.deref_value(frames, &value)?
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
    fn fat_deref_len(
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
            Resolved::Value(value) => {
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| body.name.clone());
                let projected = project_value(value, &[step], &func, &place.to_string())?;
                Ok(Resolved::Value(projected))
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
            Value::Ref(RefTarget::Value(v)) => Ok(Resolved::Value((**v).clone())),
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
            Value::Bytes(_) | Value::Str(_) => Ok(Resolved::Value(value.clone())),
            // Dereferencing a fat pointer whose base is a reference
            // reaches through to the base place. A self-contained
            // fat pointer (a slice constant: the base is the element
            // aggregate itself) derefs to the slice value itself, so
            // `Len` and index bounds see the fat pointer's length.
            Value::FatRef { base, .. } => match &**base {
                Value::Tuple(_) => Ok(Resolved::Value(value.clone())),
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
                        Ok(Resolved::Value(fields[0].clone()))
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
    ) -> InterpResult<Value> {
        match value {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                mut projection,
            }) => {
                projection.push(RefProj::Field(0));
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
                projection.push(RefProj::Field(0));
                Ok(Value::Ref(RefTarget::Static { name, projection }))
            }
            Value::Ref(RefTarget::Alloc {
                id,
                mut projection,
            }) => {
                projection.push(RefProj::Field(0));
                Ok(Value::Ref(RefTarget::Alloc { id, projection }))
            }
            Value::Ref(RefTarget::Value(v)) => {
                let inner = project_value(
                    *v,
                    &[RefProj::Field(0)],
                    &body.name,
                    "transparent wrapper pointer",
                )?;
                Ok(Value::Ref(RefTarget::Value(Box::new(inner))))
            }
            Value::Ref(RefTarget::Null) | Value::Ref(RefTarget::Address(_)) => Ok(value),
            Value::FatRef { base, len } => match *base {
                Value::Ref(_) => {
                    let base = self.ptr_field_value(frames, body, *base)?;
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

    pub(crate) fn read_resolved(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> InterpResult<Value> {
        match target {
            Resolved::Value(value) => {
                if *value == Value::Uninit {
                    return Err(InterpError::UninitializedRead {
                        func: "<constant>".to_string(),
                        place: "constant aggregate".to_string(),
                    });
                }
                Ok(value.clone())
            }
            Resolved::Static { name, projection } => {
                let base = self.static_base(name)?;
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| "<static>".to_string());
                let place_desc = format!("static {name}");
                let value = project_value(base, projection, &func, &place_desc)?;
                if value == Value::Uninit {
                    return Err(InterpError::UninitializedRead {
                        func,
                        place: place_desc,
                    });
                }
                Ok(value)
            }
            Resolved::Alloc { id, projection } => {
                let base = self.alloc_base(*id)?;
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| "<allocation>".to_string());
                let place_desc = format!("allocation #{id}");
                let value = project_value(base, projection, &func, &place_desc)?;
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
                let place_desc = describe_resolved(*local, projection);
                let value = project_value(base, projection, &frame.func, &place_desc)?;
                if value == Value::Uninit {
                    return Err(InterpError::UninitializedRead {
                        func: frame.func.clone(),
                        place: place_desc,
                    });
                }
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
            Resolved::Value(_) => Err(InterpError::type_mismatch(
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
                let slot = navigate_mut(slot, projection, "<static>", &place_desc)?;
                *slot = value;
                Ok(())
            }
            Resolved::Alloc { id, projection } => {
                let mut store = self.allocs.borrow_mut();
                let slot = store
                    .get_mut(id)
                    .ok_or_else(|| self.missing_alloc(*id))?;
                if projection.is_empty() {
                    *slot = value;
                    return Ok(());
                }
                let place_desc = format!("allocation #{id}");
                let slot = navigate_mut(slot, projection, "<allocation>", &place_desc)?;
                *slot = value;
                Ok(())
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
                    let target_body = self.program.get(&func).cloned().ok_or_else(|| {
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
                let slot = navigate_mut(slot, projection, &func, &place_desc)?;
                *slot = value;
                Ok(())
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
            Resolved::Value(value) => project_value(
                value.clone(),
                &[RefProj::Index(index)],
                "<constant>",
                "copy_nonoverlapping source",
            ),
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
            Resolved::Value(_) => self.write_resolved(frames, target, value),
        }
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
        self.read_resolved(frames, &target)
    }

    pub(crate) fn write_place(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        place: &Place,
        value: Value,
    ) -> InterpResult<()> {
        let target = self.resolve_place(frames, body, place)?;
        self.write_resolved(frames, &target, value)
    }

    /// Validate / wrap a value being stored into a typed direct local.
    fn normalize_for_ty(mut value: Value, ty: Option<&Ty>) -> InterpResult<Value> {
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
            // of them denotes in this model.
            | (Some(Ty::Adt(_)), Value::Ref(RefTarget::Alloc { .. }))
            | (Some(Ty::Adt(_)), Value::Str(_))
            // A smart-pointer ADT over a `dyn` pointee (`Box<dyn
            // Trait>`, `Rc<dyn Trait>`) holds the fat pointer.
            | (Some(Ty::Adt(_)), Value::DynRef { .. })
            | (Some(Ty::Adt(_)), Value::FmtArg { .. })
            | (Some(Ty::Adt(_)), Value::FmtCount(_))
            | (Some(Ty::Adt(_)), Value::FmtArguments(_)) => Ok(value),
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
            (Some(Ty::Str), Value::Str(_)) => Ok(value),
            (Some(Ty::Array(_, n)), Value::Tuple(items)) if items.len() as u64 == *n => Ok(value),
            (Some(Ty::Array(elem, n)), Value::Bytes(b))
                if **elem == Ty::Uint(UintTy::U8) && b.len() as u64 == *n =>
            {
                Ok(value)
            }
            (Some(Ty::Slice(_)), Value::FatRef { .. }) => Ok(value),
            // Coroutine locals hold the state object itself. (A
            // coroutine value also reaches places typed as the ADT
            // wrappers around it - `Box`, `Pin` - as a field of the
            // wrapper's aggregate value, which is checked as the
            // wrapper, not here.)
            (Some(Ty::Coroutine(_)), Value::Coroutine { .. }) => Ok(value),
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
    /// a one-field wrapper value (a `Box`, or the `Unique` /
    /// `NonNull` aggregate chain a MIR-built box is made of) is
    /// rewrapped around the unsized pointee, so `Box<[T; N]>`
    /// becomes the same shape a directly built `Box<[T]>` has.
    fn unsize_to_slice(&self, frames: &[Frame], v: Value) -> InterpResult<Value> {
        match v {
            Value::Ref(_) => {
                let target = self.deref_value(frames, &v)?;
                let pointee = self.read_resolved(frames, &target)?;
                let len = pointee.len().ok_or_else(|| {
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
            other => Err(InterpError::type_mismatch(
                "unsizing cast",
                "reference to an array",
                other.kind_name(),
            )),
        }
    }

    pub(crate) fn eval_operand(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        operand: &Operand,
    ) -> InterpResult<Value> {
        match operand {
            Operand::Copy(place) => self.read_place(frames, body, place),
            Operand::Move(place) => {
                let value = self.read_place(frames, body, place)?;
                if place.is_direct_local() {
                    let frame = frames.last_mut().expect("frame");
                    if let Some(slot) = frame.locals.get_mut(place.local.0) {
                        *slot = Value::Uninit;
                    }
                } else if let Ok(target) = self.resolve_place(frames, body, place) {
                    // A move out of a projected place leaves that
                    // part uninitialized too, so a later `Drop` of
                    // the whole value does not run the moved-out
                    // part's glue a second time (rustc's drop flags
                    // track exactly this).
                    if matches!(target, Resolved::Place { .. } | Resolved::Static { .. }) {
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
    fn eval_constant(&self, c: &Constant) -> InterpResult<Value> {
        Ok(match c {
            Constant::Unit => Value::Unit,
            Constant::Bool(b) => Value::Bool(*b),
            Constant::Int(v) => Value::Int(*v),
            Constant::Uint(v) => Value::Uint(*v),
            Constant::Float(v) => Value::Float(*v),
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
                Value::Ref(RefTarget::Value(Box::new(self.eval_constant(inner)?)))
            }
            Constant::IntPtr(addr) => Value::Ref(RefTarget::Address(*addr)),
            Constant::NullPtr => Value::Ref(RefTarget::Null),
            Constant::Uninit => Value::Uninit,
            Constant::StaticRef {
                name,
                value,
                projection,
            } => {
                self.ensure_static(name, value)?;
                Value::Ref(RefTarget::Static {
                    name: name.clone(),
                    projection: projection.clone(),
                })
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

    /// A scalar `Transmute` reinterpretation: flatten the
    /// operand value to its bit pattern through the source
    /// shape, then construct the target shape's value from the
    /// same bits.
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
        }
    }

    // ----- rvalues -----------------------------------------------------------

    fn eval_rvalue(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        rvalue: &Rvalue,
        dest_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match rvalue {
            Rvalue::Use(op) => self.eval_operand(frames, body, op),
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
                match &v {
                    Value::FatRef { len, .. } => Ok(Value::Uint(*len)),
                    Value::Str(s) => Ok(Value::Uint(s.len() as u128)),
                    // The metadata of a dyn pointer is its vtable,
                    // which this model represents by the fat value
                    // itself (it is only ever passed along or
                    // compared opaquely).
                    Value::DynRef { .. } => Ok(v.clone()),
                    // A reference to a constant slice value (a static
                    // slice, materialised as a reference to the fat
                    // value) carries the slice's length.
                    Value::Ref(RefTarget::Value(inner)) => match &**inner {
                        Value::FatRef { len, .. } => Ok(Value::Uint(*len)),
                        Value::Str(s) => Ok(Value::Uint(s.len() as u128)),
                        // Any other reference points at a sized
                        // value: its metadata is the unit value.
                        _ => Ok(Value::Unit),
                    },
                    // A reference value naming a place whose
                    // contents are a fat value (a slice reference
                    // formed as `&*box`, stored as a place reference)
                    // carries that fat value's metadata; any other
                    // reference points at a sized value, whose
                    // metadata is the unit value.
                    Value::Ref(_) => {
                        let target = self.deref_value(frames, &v)?;
                        match self.read_resolved(frames, &target)? {
                            Value::FatRef { len, .. } => Ok(Value::Uint(len)),
                            Value::Str(text) => Ok(Value::Uint(text.len() as u128)),
                            _ => Ok(Value::Unit),
                        }
                    }
                    other => Err(InterpError::type_mismatch(
                        "PtrMetadata",
                        "fat pointer",
                        other.kind_name(),
                    )),
                }
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
                Ok(Value::Tuple(items))
            }
            Rvalue::AggregateAdt {
                variant,
                operands,
                field_index,
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
            Rvalue::PtrField(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.ptr_field_value(frames, body, value)
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
            Rvalue::Reinterpret(operand, src, dst) => {
                let value = self.eval_operand(frames, body, operand)?;
                self.reinterpret_value(&value, src, dst)
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
                                    Resolved::Value(value) => {
                                        break Ok(Value::Ref(RefTarget::Value(Box::new(
                                            value,
                                        ))));
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
                    Resolved::Value(value) => Value::Ref(RefTarget::Value(Box::new(value))),
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
            Rvalue::ThreadLocalRef(name) => Err(InterpError::UnsupportedRvalue {
                kind: format!(
                    "ThreadLocalRef {name}: thread-local statics are not modelled \
                     (the driver collects no static items, so there is no storage \
                     or initializer to bind)"
                ),
            }),
            Rvalue::Unsupported(kind) => Err(InterpError::UnsupportedRvalue {
                kind: kind.clone(),
            }),
        }
    }
}

/// Human-readable form of a resolved place, for error messages.
pub(crate) fn describe_resolved(local: usize, projection: &[RefProj]) -> String {
    let mut out = format!("_{local}");
    for proj in projection {
        match proj {
            RefProj::Field(i) => out.push_str(&format!(".{i}")),
            RefProj::Index(i) => out.push_str(&format!("[{i}]")),
            RefProj::Downcast(v) => out.push_str(&format!(" as variant {v}")),
            RefProj::Subslice { offset, len } => {
                out.push_str(&format!("[{offset}..{}]", offset + len))
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

fn index_oob(len: usize, index: usize) -> InterpError {
    InterpError::panic(format!(
        "index out of bounds: the len is {len} but the index is {index}"
    ))
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

pub(crate) fn project_value(
    value: Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<Value> {
    let mut value = value;
    for step in projection {
        if value == Value::Uninit {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        }
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
                Value::Coroutine { ref fields, .. } => {
                    fields.get(*idx).cloned().ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
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
                // Field 0 of a fat pointer value is the pointer
                // field of its smart-pointer wrapper (a `Box`'s
                // `Unique`): in this model the `DynRef` *is* that
                // pointer, metadata included.
                Value::DynRef { .. } if *idx == 0 => value,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("field projection on {place}"),
                        "tuple or ADT value",
                        other.kind_name(),
                    ));
                }
            },
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
                    return Err(InterpError::type_mismatch(
                        format!("index projection on {place}"),
                        "array, byte or string value",
                        other.kind_name(),
                    ));
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
                // A coroutine's downcast names the state whose field
                // layout the access uses; the storage is shared
                // between states, so the access is valid whatever
                // the current tag is (the body writes a state's
                // fields before setting the tag to that state).
                Value::Coroutine { .. } => value,
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
        };
    }
    Ok(value)
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
    if *slot == Value::Uninit {
        return Err(InterpError::UninitializedRead {
            func: func.to_string(),
            place: place.to_string(),
        });
    }
    let (step, rest) = projection.split_first().expect("non-empty");
    match step {
        RefProj::Field(idx) => {
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
                if *idx >= fields.len() {
                    return Err(index_oob(fields.len(), *idx));
                }
                navigate_mut(&mut fields[*idx], rest, func, place)
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
            // See the read direction: a coroutine's field writes go
            // through the downcast of the state being entered,
            // whatever the current tag is.
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
/// Returns `(result, overflowed)`. In non-checked mode an overflow (or a
/// division by zero / invalid shift) is a guest panic instead of a flag.
/// Comparison and bitwise-on-bool operations never overflow.
fn eval_binary(
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
                    // Overflow: checked form wraps (Div yields `min`,
                    // Rem yields 0), plain form panics via the flag below.
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
        if overflowed && !checked_form {
            return Err(InterpError::panic(format!("arithmetic overflow in {op:?}")));
        }
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
        if overflowed && !checked_form {
            return Err(InterpError::panic(format!("arithmetic overflow in {op:?}")));
        }
        return Ok((result, overflowed));
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
    if matches!(op, Eq | Ne | Lt | Le | Gt | Ge) {
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

/// The guest panic message for a shift whose amount meets or
/// exceeds the value's width: Rust 1.100.0's native wording, which
/// names only the direction (the amount is not part of the text).
fn shift_overflow_message(op: BinOp) -> &'static str {
    if op == BinOp::Shl {
        "attempt to shift left with overflow"
    } else {
        "attempt to shift right with overflow"
    }
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
/// shift has: the checked form yields the wrapped result (the
/// amount taken modulo the width, the `wrapping_shl` /
/// `wrapping_shr` reading) with the overflow flag set, and the
/// plain form is a guest panic with Rust 1.100.0's native message
/// (`attempt to shift left with overflow` / `attempt to shift
/// right with overflow`). Bits shifted out of the declared width
/// are simply lost, as natively: they set neither the overflow
/// flag nor a panic, in either form.
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
                if checked_form {
                    let wrapped = if op == BinOp::Shl {
                        wrap_signed(a.wrapping_shl(shift % bits), bits)
                    } else {
                        wrap_signed(a.wrapping_shr(shift % bits), bits)
                    };
                    return Ok((Value::Int(wrapped), true));
                }
                return Err(InterpError::panic(shift_overflow_message(op)));
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
                if checked_form {
                    let wrapped = if op == BinOp::Shl {
                        wrap_unsigned(a.wrapping_shl(shift % bits), bits)
                    } else {
                        wrap_unsigned(a.wrapping_shr(shift % bits), bits)
                    };
                    return Ok((Value::Uint(wrapped), true));
                }
                return Err(InterpError::panic(shift_overflow_message(op)));
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
fn round_to_f16(v: f64) -> f64 {
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

fn eval_unary(op: UnOp, v: &Value, dest_ty: Option<&Ty>) -> InterpResult<Value> {
    match (op, v) {
        (UnOp::Neg, Value::Int(a)) => {
            let bits = signed_bits(dest_ty);
            match a.checked_neg() {
                Some(r)
                    if bits >= 128
                        || (r >= -(1i128 << (bits - 1)) && r <= (1i128 << (bits - 1)) - 1) =>
                {
                    Ok(Value::Int(wrap_signed(r, bits)))
                }
                _ => Err(InterpError::panic("arithmetic overflow in Neg")),
            }
        }
        (UnOp::Neg, Value::Float(a)) => Ok(Value::Float(-a)),
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
fn float_to_int(v: f64, bits: u32) -> i128 {
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
fn float_to_uint(v: f64, bits: u32) -> u128 {
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
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
                };
                mix(&mut h, tag);
                mix(&mut h, n);
            }
        }
        Value::Ref(RefTarget::Value(_)) => {
            mix(&mut h, 0x5eed_5eed_5eed_5eed);
        }
        Value::Ref(RefTarget::Static { name, projection }) => {
            mix(&mut h, 0x57a7_1c5_7a7_1c5);
            for b in name.bytes() {
                mix(&mut h, b as u128);
            }
            for step in projection {
                let (tag, n) = match step {
                    RefProj::Field(i) => (1u128, *i as u128),
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
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
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
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

fn eval_cast(v: &Value, ty: &Ty) -> InterpResult<Value> {
    match ty {
        Ty::Int(it) => {
            // A float source saturates; it does not wrap like an
            // integer source.
            if let Value::Float(a) = v {
                return Ok(Value::Int(float_to_int(*a, it.bits())));
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
        | Value::Str(_)
        | Value::Bytes(_)
        | Value::Ref(_)
        | Value::FatRef { .. }
        | Value::FnPtr(_)
        | Value::FmtArg { .. }
        | Value::FmtCount(_)
        | Value::FmtArguments(_) => Ok(()),
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
                field => Value::Ref(RefTarget::Value(Box::new(field))),
            }
        }
        other @ (Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }) => other,
        other => Value::Ref(RefTarget::Value(Box::new(other))),
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
