//! Modelled host state: commands, channels, files, formatters.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{PanicPayload, RefProj, RefTarget, Value};

/// One spawned thread of the cooperative thread model: the
/// callable captured at `spawn` time, the frames of a run
/// suspended at a quantum boundary, and the outcome once it has
/// finished. A spawned thread never runs concurrently with the
/// thread that spawned it; it runs when the running thread
/// reaches a scheduling point (a point where it would block, a
/// join, a scope exit — see [`Interpreter::drain_threads`]) or
/// when the running thread's quantum expires, and it runs for at
/// most one quantum at a time (see [`Interpreter::run_thread`]),
/// so a busy-waiting thread cannot starve the threads it waits
/// for, exactly as under an OS scheduler.
pub(crate) struct ThreadSlot {
    /// The name the thread's `std::thread::Builder` set (read
    /// back by `std::thread::Thread::name`); `None` for an
    /// unnamed thread.
    pub name: Option<String>,
    pub callable: Option<Value>,
    /// The frames of a run suspended at a quantum boundary, or
    /// blocked on a channel operation no other thread could
    /// satisfy (see [`InterpError::Suspended`]), if this thread
    /// has started but not finished.
    pub parked: Option<ParkedThread>,
    /// The thread's outcome: its return value, or the panic
    /// payload it ended with (kept whole, so `join`'s error can
    /// be the `Box<dyn Any>` the guest downcasts).
    pub outcome: Option<Result<Value, PanicPayload>>,
}
/// A thread segment suspended at a quantum boundary, or
/// blocked on a channel operation: its call
/// frames (the bottom frame is the thread's entry frame), and
/// the ids of the frames below the segment that its frames
/// actually reference. The segment may resume only on a stack
/// that still holds every one of those frames: its places name
/// frames by id, and a referenced frame that has since returned
/// is storage the segment may still point into (a scoped
/// thread's borrowed locals), so resuming without it would read
/// dead storage. Frames the segment was merely driven *above*
/// — a transient runtime region that has since returned —
/// do not pin it: a `'static` spawned thread references no frame
/// below itself and can resume on any stack. Every frame the
/// segment itself owns travels with it, and channel /
/// allocation / static targets are id-keyed, so referenced
/// frames below are the only external dependency.
pub(crate) struct ParkedThread {
    pub frames: Vec<Frame>,
    pub below: Vec<u64>,
}

/// What one [`Interpreter::run_thread`] call did.
pub(crate) enum ThreadRun {
    /// The thread finished; its outcome is recorded.
    Finished,
    /// The thread's quantum expired; its frames are parked and
    /// it will resume at a later scheduling point.
    Yielded,
    /// The thread blocked on a channel operation no other
    /// thread could satisfy at the time; its frames are parked
    /// (see [`InterpError::Suspended`]) and it will resume at a
    /// later scheduling point, where the operation it blocked
    /// in re-executes and re-checks its condition.
    Blocked,
    /// The thread is parked above frames that are not on the
    /// current stack, so it cannot resume in this context.
    NotRunnable,
}

/// The scheduler's record of the thread segment currently being
/// driven (see the `segment` field of [`Interpreter`]): the
/// drive depth at which its loop runs, the thread it belongs
/// to, and the step count at which the current quantum began.
/// Set by the entry drive, by `run_thread`, and by scheduled
/// regions (`drive_scheduled`) for the duration of one drive;
/// `None` outside any drive.
#[derive(Clone, Copy)]
pub(crate) struct SegmentMark {
    /// The drive depth of the segment's own drive loop. A yield
    /// is honoured only at exactly this depth: inside a nested
    /// drive (a builtin executing a callable — a `Once` closure,
    /// a waker wake, a comparator) the interpreter's own call
    /// stack carries a continuation that parking would lose, so
    /// nested drives always run to completion and the yield is
    /// taken when control returns to the segment's own loop.
    pub depth: usize,
    /// The thread the segment belongs to (1 is the entry thread).
    pub thread: u64,
    /// The interpreter's step count when this quantum began.
    pub steps0: usize,
}


/// The shared state of one `std::sync::mpsc` channel, keyed by the
/// id carried inside the `Sender` / `Receiver` handle values.
/// `senders` / `receivers` count live handles: handle creation and
/// `Clone` increment them, and dropping a handle place decrements
/// them (see the `Drop` terminator's note in `drive_frames`), which
/// is what makes disconnection observable exactly when the last
/// handle of a side goes away.
/// The `Wake` / `LocalWake` impl methods a modelled waker
/// dispatches to, recorded by the modelled
/// `From<Arc<T: Wake>>` / `From<Rc<T: LocalWake>>` conversion
/// against the handle's allocation id (see
/// `Interpreter::note_waker_dispatch` in `builtins`): the names
/// the driver names the impl methods' bodies by. The waker
/// value itself carries only the handle, whose pointee type -
/// and with it the impl a wake runs - the value does not record.
#[derive(Clone)]
pub(crate) struct WakeDispatch {
    pub(crate) wake: String,
    pub(crate) wake_by_ref: String,
}

pub(crate) struct ChannelState {
    pub queue: std::collections::VecDeque<Value>,
    pub senders: usize,
    pub receivers: usize,
    pub bound: Option<usize>,
}

/// A modelled `std::process::Command` under construction (the
/// process model in `builtins.rs`): the builder state the sys
/// `Command` methods record, keyed by the id inside the handle
/// value the `Command::new` builtin returns. At spawn time the
/// program is first resolved deterministically: a resolution
/// failure is the `io::Error` it produces natively (see
/// `process_spawn_error` in `builtins.rs`); a program that
/// would start goes to the run's tool interception or process
/// host when one is configured, and is a named unsupported
/// error in a run with neither.
/// One C variable argument list's state (see [`Value::VaList`]):
/// the tail arguments of one variadic call, in order, and the
/// cursor the next `va_arg` reads. The arguments are the values
/// the caller evaluated, stored in the interpreter's usual shapes
/// (integers sign-extended / zero-extended to 128 bits, `f32`
/// widened to `f64`), which is exactly the promoted shape C's
/// default argument promotions hand to the callee.
#[derive(Clone, Debug)]
pub(crate) struct VaListState {
    pub args: Vec<Value>,
    pub cursor: usize,
}

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
    /// Whether writes to this capture stream straight into the
    /// interpreter's captured stderr instead of accumulating in
    /// `buffer`: the shape of a `Termination` report rendering,
    /// whose formatter natively writes to stderr as the payload's
    /// `Debug` body runs, interleaved with the body's own stderr
    /// writes. Only the report rendering's own payload body
    /// capture streams (see `Interpreter::fmt_stream_arm`); a
    /// streaming capture's `buffer` stays empty.
    pub stream: bool,
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

/// One stream's `Stdio` configuration, as the guest set it
/// on a command (see [`CommandState::stdout_stdio`]): the
/// routing an invocation's outcome bytes get for that
/// stream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StdioConfig {
    Inherit,
    Null,
    Piped,
    /// The file-descriptor form (`Stdio::from(..)` over a
    /// file handle -- a `File`, a child pipe handle): the
    /// stream is wired to the file-table entry the handle
    /// carried, whose id this is.
    File(u64),
    /// A configuration the value model does not decode: an
    /// invocation carrying one is reported unsupported
    /// rather than run with a guessed routing.
    Unmodelled,
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
    /// The `Stdio` configuration set for the child's stdout,
    /// when the guest set one; `None` keeps the entry point's
    /// default (inherit for `status`, piped for `output`).
    pub stdout_stdio: Option<StdioConfig>,
    /// The `Stdio` configuration set for the child's stderr;
    /// see `stdout_stdio`.
    pub stderr_stdio: Option<StdioConfig>,
    /// The `Stdio` configuration set for the child's stdin;
    /// see `stdout_stdio`. Recorded for the process host's
    /// wiring; the tool model never feeds a child's stdin and
    /// ignores it.
    pub stdin_stdio: Option<StdioConfig>,
}

/// One live child process started through the process host
/// (see [`ProcessHost`]): the host's child id and the child's
/// process id, the command's stdio configuration (the finished
/// child's stream bytes are routed by it), the file-table ids
/// of the child's pipe ends (see [`ProcessSpawned`]), the
/// stream bytes the interpreter has collected from the ends
/// the guest did not take over, and the exit code once the end
/// has been observed, so repeated `wait` / `try_wait` calls
/// answer with the same status, as std's cached status does.
pub(crate) struct ChildState {
    pub host_id: u64,
    pub pid: u32,
    pub stdout_stdio: Option<StdioConfig>,
    pub stderr_stdio: Option<StdioConfig>,
    /// The file-table id of the child's stdin write end, when
    /// the wiring piped it. `wait` / `wait_with_output` close
    /// it (std drops the handle the `Child` still holds before
    /// waiting) unless the guest took the handle out of the
    /// `Child`, in which case the guest owns its lifetime.
    pub stdin_file: Option<u64>,
    /// The file-table id of the child's stdout read end.
    pub stdout_file: Option<u64>,
    /// The file-table id of the child's stderr read end.
    pub stderr_file: Option<u64>,
    pub finished: Option<i32>,
    /// Whether the finished child's inherited streams have
    /// been relayed into the guest's own streams (they join
    /// them once, at the first observation of the end whose
    /// collected bytes are complete).
    pub relayed: bool,
    /// The child's stdout bytes collected so far from the read
    /// end (see `collect_child_streams` in `builtins.rs`):
    /// relayed at the end for an inherited stream, handed to
    /// `wait_with_output` for a piped one the guest did not
    /// take out of the `Child`.
    pub stdout: Vec<u8>,
    /// The child's stderr bytes collected so far; see
    /// `stdout`.
    pub stderr: Vec<u8>,
    /// Stream collections in flight: a drain thread per stream
    /// reading a cloned handle of its read end, started when a
    /// blocking call needs the child's end observed without
    /// starving a child that fills its pipes. Finished drains
    /// move their bytes into `stdout` / `stderr`.
    pub drains: Vec<(bool, std::thread::JoinHandle<Vec<u8>>)>,
}

/// One host socket behind a guest `std::net` handle value
/// (see the socket model in `builtins::net`): the real host
/// socket the handle delegates to, plus the guest-visible
/// state the host object cannot carry for the model. The
/// host socket itself is always nonblocking: a guest-visible
/// blocking operation is emulated by the cooperative
/// scheduler (the mpsc waits' pattern), and `nonblocking`
/// records the mode the guest selected, which decides
/// whether a would-block outcome suspends or surfaces as
/// the `WouldBlock` error. The stream timeouts are the
/// guest's `set_read_timeout` / `set_write_timeout` values:
/// on a nonblocking host socket they cannot fire by
/// themselves, so a would-block outcome while one is set
/// is the timeout outcome std reports (`WouldBlock`).
pub(crate) enum SocketEntry {
    Listener {
        socket: std::net::TcpListener,
        nonblocking: bool,
    },
    Stream {
        socket: std::net::TcpStream,
        nonblocking: bool,
        read_timeout: Option<std::time::Duration>,
        write_timeout: Option<std::time::Duration>,
    },
    Udp {
        socket: std::net::UdpSocket,
        nonblocking: bool,
    },
}

impl SocketEntry {
    /// The guest-selected nonblocking mode of the entry.
    pub(crate) fn nonblocking(&self) -> bool {
        match self {
            SocketEntry::Listener { nonblocking, .. }
            | SocketEntry::Stream { nonblocking, .. }
            | SocketEntry::Udp { nonblocking, .. } => *nonblocking,
        }
    }

    /// Record the guest-selected nonblocking mode.
    pub(crate) fn set_nonblocking(&mut self, value: bool) {
        match self {
            SocketEntry::Listener { nonblocking, .. }
            | SocketEntry::Stream { nonblocking, .. }
            | SocketEntry::Udp { nonblocking, .. } => *nonblocking = value,
        }
    }
}

/// One `std::sync::Barrier`'s shared state (see the
/// `Barrier` builtins in `builtins.rs`): how many parties
/// wait at the barrier, how many have arrived in the
/// current generation, and the generation counter a waiter
/// watches to learn its generation has been released.
pub(crate) struct BarrierState {
    pub parties: usize,
    pub arrived: usize,
    pub generation: u64,
}

/// The file-table id a `std::fs::File` handle value carries:
/// the integer at the bottom of the wrapper chain the file
/// model builds (`File` -> sys `File` -> `FileDesc` ->
/// `OwnedFd` -> descriptor), each layer a one-field aggregate.
/// A value of any other shape is not a modelled handle.
pub(crate) fn file_handle_id(value: &Value) -> Option<u64> {
    let mut cur = value;
    for _ in 0..8 {
        match cur {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = &fields[0];
            }
            Value::Int(id) => return Some(*id as u64),
            Value::Uint(id) => return Some(*id as u64),
            _ => return None,
        }
    }
    None
}

pub(crate) fn channel_handle_id(value: &Value) -> Option<u64> {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
            Value::Uint(id) => Some(*id as u64),
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
/// Whether the value has the `Rc` / `Arc` handle shape, as
/// opposed to a `Box` (or a `Weak`): the reference-counted
/// handles carry the phantom and allocator fields next to the
/// pointer field, so their aggregate has at least two fields,
/// where a `Box` value is the one-field pointer wrapper alone.
/// The distinction matters where a trait-object value's base is
/// classified by shape only: a `Box<dyn Trait>` and an
/// `Rc<dyn Trait>` have the same first-field shape, but field
/// steps through the trait object resolve differently (the
/// box's field 0 is the fat pointer itself, the handle's fields
/// are the handle's own).
pub(crate) fn is_rc_arc_handle(value: &Value) -> bool {
    if rc_handle_id(value).is_none() {
        return false;
    }
    matches!(value, Value::Enum { fields, .. } if fields.len() >= 2)
}

pub(crate) fn rc_handle_id(value: &Value) -> Option<u64> {
    // A handle in trait-object form denotes the same allocation
    // as the handle it wraps: the unsizing coercion changed the
    // pointee's view, not the handle's identity, so the count
    // machinery (clone, drop, the count readers) must see
    // through the `DynRef` to the handle aggregate.
    let mut value = value;
    while let Value::DynRef { base, .. } = value {
        value = base;
    }
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

impl<'a> Interpreter<'a> {
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

    /// Register a freshly spawned child process and return the
    /// id its `Child` handle value carries.
    pub(crate) fn new_child(&self, state: ChildState) -> u64 {
        let mut next = self.next_child.borrow_mut();
        let id = *next;
        *next += 1;
        drop(next);
        self.children.borrow_mut().insert(id, state);
        id
    }

    /// Run `f` against the state of the child with the given
    /// id; `None` if the id names no live child.
    pub(crate) fn with_child<R>(
        &self,
        id: u64,
        f: impl FnOnce(&mut ChildState) -> R,
    ) -> Option<R> {
        let mut children = self.children.borrow_mut();
        children.get_mut(&id).map(f)
    }

    /// Forget the child with the given id (its outcome was
    /// collected, or the guest dropped the handle).
    pub(crate) fn remove_child(&self, id: u64) -> Option<ChildState> {
        self.children.borrow_mut().remove(&id)
    }

    /// Register a C variable argument list holding `args` (the
    /// evaluated tail arguments of one variadic call, in order)
    /// and return the handle value that names it.
    pub(crate) fn new_va_list(&self, args: Vec<Value>) -> Value {
        let mut next = self.next_va_list.borrow_mut();
        let id = *next;
        *next += 1;
        drop(next);
        self.va_lists
            .borrow_mut()
            .insert(id, VaListState { args, cursor: 0 });
        Value::VaList(id)
    }

    /// Duplicate the va list with the given id (C `va_copy`):
    /// a new list over the same arguments, starting at the
    /// source's current cursor and advancing independently.
    pub(crate) fn va_list_duplicate(&self, id: u64) -> InterpResult<Value> {
        let state = self
            .va_lists
            .borrow()
            .get(&id)
            .cloned()
            .ok_or_else(|| InterpError::UnsupportedTerminator {
                kind: format!("va list handle {id} is not live"),
            })?;
        Ok(self.new_va_list_at(state.args, state.cursor))
    }

    /// Register a va list over `args` with its cursor already at
    /// `cursor` (the [`Self::va_list_duplicate`] back end).
    fn new_va_list_at(&self, args: Vec<Value>, cursor: usize) -> Value {
        let mut next = self.next_va_list.borrow_mut();
        let id = *next;
        *next += 1;
        drop(next);
        self.va_lists
            .borrow_mut()
            .insert(id, VaListState { args, cursor });
        Value::VaList(id)
    }

    /// Read the next argument of the va list with the given id
    /// (`va_arg`): the argument at the cursor, which advances by
    /// one. Reading past the last argument is undefined behaviour
    /// in C; here it is a named failure.
    pub(crate) fn va_list_next(&self, id: u64) -> InterpResult<Value> {
        let mut lists = self.va_lists.borrow_mut();
        let state = lists
            .get_mut(&id)
            .ok_or_else(|| InterpError::UnsupportedTerminator {
                kind: format!("va list handle {id} is not live"),
            })?;
        let value = state.args.get(state.cursor).cloned().ok_or_else(|| {
            InterpError::UnsupportedTerminator {
                kind: "va_arg read past the end of the argument list".to_string(),
            }
        })?;
        state.cursor += 1;
        Ok(value)
    }

    /// Resolve a value denoting a va list — the handle itself,
    /// or a reference to a place holding one — to the list's
    /// table id.
    pub(crate) fn va_list_handle(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<u64> {
        match value {
            Value::VaList(id) => Ok(*id),
            Value::Ref(_) => {
                let target = self.deref_value(frames, value)?;
                match self.read_resolved(frames, &target)? {
                    Value::VaList(id) => Ok(id),
                    other => Err(InterpError::type_mismatch(
                        "va list argument",
                        "a va list value",
                        other.kind_name(),
                    )),
                }
            }
            other => Err(InterpError::type_mismatch(
                "va list argument",
                "a va list value",
                other.kind_name(),
            )),
        }
    }

    /// Whether `ty` is the C variable argument list type
    /// (`std::ffi::VaList`, spelled through `core` inside core's
    /// own bodies).
    fn is_va_list_ty(ty: &Ty) -> bool {
        matches!(ty, Ty::Adt(name) if name == "std::ffi::VaList" || name == "core::ffi::VaList")
    }

    /// Pack the argument values of a call to the C-variadic body
    /// `callee` (registered under `func`) into the shape the body
    /// declares: a variadic body's last parameter is its
    /// `VaList`, and the caller supplies the fixed arguments
    /// followed by every tail argument as a separate value, so
    /// the tail becomes one va-list value. A call that already
    /// passes exactly the declared arguments with a va list in
    /// the last slot forwards that list unchanged (the
    /// `vprintf`-style idiom), as does any call to a body whose
    /// last parameter is not a va list; anything else is left
    /// for the caller's arity check to judge.
    pub(crate) fn pack_variadic_args(
        &self,
        func: &str,
        callee: &Body,
        args: Vec<Value>,
    ) -> Vec<Value> {
        let _ = func;
        if callee.arg_count == 0 || args.len() < callee.arg_count - 1 {
            return args;
        }
        let Some(last_ty) = callee.local_tys.get(callee.arg_count) else {
            return args;
        };
        if !Self::is_va_list_ty(last_ty) {
            return args;
        }
        if args.len() == callee.arg_count {
            if matches!(args.last(), Some(Value::VaList(_))) {
                return args;
            }
        }
        let fixed = callee.arg_count - 1;
        let mut packed: Vec<Value> = args[..fixed].to_vec();
        packed.push(self.new_va_list(args[fixed..].to_vec()));
        packed
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
        self.channel_epoch.set(self.channel_epoch.get() + 1);
        id
    }

    /// Access one channel's state, if it exists.
    pub(crate) fn with_channel<R>(
        &self,
        id: u64,
        f: impl FnOnce(&mut ChannelState) -> R,
    ) -> Option<R> {
        self.channels.borrow_mut().get_mut(&id).map(|st| {
            // The scheduler's progress signal: an access that
            // changes the observable state (the queue length or
            // a handle count) moves the epoch.
            let before = (st.queue.len(), st.senders, st.receivers);
            let result = f(st);
            if before != (st.queue.len(), st.senders, st.receivers) {
                self.channel_epoch.set(self.channel_epoch.get() + 1);
            }
            result
        })
    }

    /// Record a channel handle's drop (see [`ChannelState`]):
    /// `kind` is `"sender"` or `"receiver"`.
    /// Register a freshly opened host file in the file table
    /// and return the id the guest handle value carries.
    pub(crate) fn register_file(&self, file: std::fs::File) -> u64 {
        let id = *self.next_file.borrow();
        *self.next_file.borrow_mut() = id + 1;
        self.files
            .borrow_mut()
            .insert(id, std::cell::RefCell::new(file));
        id
    }

    /// Run `f` against the host file behind table id `id`;
    /// `None` when the id names no open file (a handle closed
    /// already, or one the model never issued).
    pub(crate) fn with_file<R>(
        &self,
        id: u64,
        f: impl FnOnce(&mut std::fs::File) -> R,
    ) -> Option<R> {
        let files = self.files.borrow();
        let entry = files.get(&id)?;
        let result = f(&mut entry.borrow_mut());
        Some(result)
    }

    /// Register a host socket in the socket table and return
    /// the id the guest handle value carries. Socket ids come
    /// from the same counter as file ids: both surface to the
    /// guest as descriptor numbers (a socket handle's
    /// `as_raw_fd` is its table id), and the descriptor space
    /// is one space, so no two live handles of either kind
    /// ever share a number.
    pub(crate) fn register_socket(&self, entry: SocketEntry) -> u64 {
        let id = *self.next_file.borrow();
        *self.next_file.borrow_mut() = id + 1;
        self.sockets
            .borrow_mut()
            .insert(id, std::cell::RefCell::new(entry));
        id
    }

    /// Run `f` against the socket behind table id `id`;
    /// `None` when the id names no open socket (a handle
    /// closed already, or one the model never issued).
    pub(crate) fn with_socket<R>(
        &self,
        id: u64,
        f: impl FnOnce(&mut SocketEntry) -> R,
    ) -> Option<R> {
        let sockets = self.sockets.borrow();
        let entry = sockets.get(&id)?;
        let result = f(&mut entry.borrow_mut());
        Some(result)
    }

    /// The reference to the guest's errno cell, allocating
    /// the cell on first use: a four-byte scalar block,
    /// zeroed, whose address the errno-location calls
    /// (`std::sys::io::error::unix::errno_location`, the
    /// libc crate's `__errno_location`) hand the guest and
    /// whose contents the failing descriptor operations
    /// record their host errno in, for
    /// `std::io::Error::last_os_error` to read back (see
    /// `builtins::libc_io`). One cell per process, not per
    /// thread: the interpreter's threads run cooperatively,
    /// so the call-then-check sequence through which a
    /// program observes errno cannot interleave with
    /// another thread's failing call, which is the only
    /// pattern a per-thread cell would change the outcome
    /// of.
    pub(crate) fn errno_cell_ref(&self) -> Value {
        if let Some(id) = *self.errno_cell.borrow() {
            return Value::Ref(RefTarget::Alloc {
                id,
                projection: Vec::new(),
            });
        }
        let id = self.alloc_insert(Value::Int(0));
        self.record_alloc_layout(id, 4, 4);
        *self.errno_cell.borrow_mut() = Some(id);
        Value::Ref(RefTarget::Alloc {
            id,
            projection: Vec::new(),
        })
    }

    /// Close the host socket behind table id `id` (the
    /// `SocketHandleDrop` plan's effect): removing the entry
    /// drops the host socket, closing the descriptor. An id
    /// that names no open socket is already closed, so there
    /// is nothing to do.
    pub(crate) fn socket_handle_closed(&self, id: u64) {
        self.sockets.borrow_mut().remove(&id);
    }

    /// Close the host file behind table id `id` (the
    /// `FileHandleDrop` plan's effect): removing the entry drops
    /// the host `File`, closing the descriptor. An id that names
    /// no open file is already closed, so there is nothing to do.
    /// The file-table id of the guest's standard input,
    /// registering a duplicate of this process's stdin on
    /// first use: the guest's `std::io::stdin` reads go
    /// through the ordinary file model, and in a self-child
    /// the descriptor is the pipe the parent wired, exactly
    /// as a native child's stdin is the parent's.
    pub(crate) fn stdin_file_id(&self) -> u64 {
        if !self.files.borrow().contains_key(&STDIN_FILE_ID) {
            use std::os::fd::AsFd as _;
            if let Ok(owned) =
                std::io::stdin().as_fd().try_clone_to_owned()
            {
                self.files.borrow_mut().insert(
                    STDIN_FILE_ID,
                    std::cell::RefCell::new(std::fs::File::from(
                        owned,
                    )),
                );
            }
        }
        STDIN_FILE_ID
    }

    /// Register a barrier for `parties` parties and return
    /// its table id.
    pub(crate) fn new_barrier(&self, parties: usize) -> u64 {
        let id = {
            let mut next = self.next_barrier.borrow_mut();
            let id = *next;
            *next += 1;
            id
        };
        self.barriers.borrow_mut().insert(
            id,
            BarrierState {
                parties,
                arrived: 0,
                generation: 0,
            },
        );
        id
    }

    /// Record the current thread's arrival at the barrier
    /// `id`: `(generation, leader)` -- the generation the
    /// arrival belongs to, and whether this arrival completes
    /// it (the last party releases the generation: the count
    /// resets and the generation counter advances). `None`
    /// for a stale id.
    pub(crate) fn barrier_arrive(
        &self,
        id: u64,
    ) -> Option<(u64, bool)> {
        let mut barriers = self.barriers.borrow_mut();
        let state = barriers.get_mut(&id)?;
        let generation = state.generation;
        state.arrived += 1;
        if state.arrived >= state.parties {
            state.arrived = 0;
            state.generation += 1;
            Some((generation, true))
        } else {
            Some((generation, false))
        }
    }

    /// The barrier `id`'s current generation counter.
    pub(crate) fn barrier_generation(&self, id: u64) -> Option<u64> {
        self.barriers.borrow().get(&id).map(|state| state.generation)
    }

    pub(crate) fn file_handle_closed(&self, id: u64) {
        self.files.borrow_mut().remove(&id);
    }

    pub(crate) fn channel_handle_dropped(&self, id: u64, kind: &str) {
        if let Some(state) = self.channels.borrow_mut().get_mut(&id) {
            let before = (state.senders, state.receivers);
            match kind {
                "sender" => state.senders = state.senders.saturating_sub(1),
                _ => state.receivers = state.receivers.saturating_sub(1),
            }
            if before != (state.senders, state.receivers) {
                self.channel_epoch.set(self.channel_epoch.get() + 1);
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
            self.channel_epoch.set(self.channel_epoch.get() + 1);
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
    /// The reference-counted allocation a value denotes, when
    /// the value has a handle's shape AND the allocation it
    /// names really holds an `RcInner` / `ArcInner` aggregate:
    /// a `Box` handle has the same value shape as an rc handle,
    /// so shape alone cannot license count discipline -- the
    /// store's contents are the deciding evidence.
    pub(crate) fn rc_inner_id(&self, value: &Value) -> Option<u64> {
        let id = rc_handle_id(value)?;
        let store = self.allocs.borrow();
        let inner = store.get(&id)?;
        fn count_cell(v: &Value) -> bool {
            fn unwrap1(v: &Value) -> Option<&Value> {
                match v {
                    Value::Enum { fields, .. } if fields.len() == 1 => fields.first(),
                    _ => None,
                }
            }
            matches!(unwrap1(v).and_then(unwrap1), Some(Value::Uint(_)))
        }
        match inner {
            Value::Enum { fields, .. } if fields.len() >= 3 => {
                (count_cell(&fields[0]) && count_cell(&fields[1])).then_some(id)
            }
            _ => None,
        }
    }

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
}
