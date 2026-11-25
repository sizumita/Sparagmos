//! Guest threads and the cooperative scheduler.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{PanicPayload, RefTarget, Value};

impl<'a> Interpreter<'a> {
    pub(crate) fn jump(frames: &mut Vec<Frame>, body: &Body, target: BasicBlock) -> InterpResult<()> {
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
                name: None,
                callable: Some(callable),
                parked: None,
                outcome: None,
            },
        );
        id
    }

    /// Record the name a `std::thread::Builder` gave the thread
    /// `id` (see `Builtin::ThreadBuilderSpawn`).
    pub(crate) fn set_thread_name(&self, id: u64, name: Option<String>) {
        if let Some(slot) = self.threads.borrow_mut().get_mut(&id) {
            slot.name = name;
        }
    }

    /// The name of the thread `id`: the main thread (id 1) is
    /// `main`; a spawned thread carries the name registered for
    /// it at spawn time, or none.
    pub(crate) fn thread_name(&self, id: u64) -> Option<String> {
        if id == 1 {
            return Some("main".to_string());
        }
        self.threads
            .borrow()
            .get(&id)
            .and_then(|slot| slot.name.clone())
    }

    /// Whether the thread has been spawned and has no recorded
    /// outcome yet: it has not started, or it is parked mid-run.
    pub(crate) fn thread_unfinished(&self, id: u64) -> bool {
        self.threads
            .borrow()
            .get(&id)
            .is_some_and(|slot| slot.outcome.is_none())
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

    /// The ids of the frames in `below` that a suspended
    /// segment's frames reference: a `RefTarget::Place` anywhere
    /// in their locals, transitively through aggregates,
    /// reference targets, closure environments and formatting
    /// arguments. Computed once per suspension; the scan walks
    /// the segment's own values only, which a quantum of guest
    /// work already dwarfs.
    fn referenced_below_frames(segment: &[Frame], below: &[Frame]) -> Vec<u64> {
        fn collect(value: &Value, out: &mut HashSet<u64>) {
            match value {
                Value::Ref(RefTarget::Place { frame, .. }) => {
                    out.insert(*frame);
                }
                Value::Ref(RefTarget::Value(inner, _)) => collect(inner, out),
                Value::Tuple(items) | Value::SparseVec { elems: items, .. } => {
                    for item in items {
                        collect(item, out);
                    }
                }
                Value::Enum { fields, .. } => {
                    for field in fields {
                        collect(field, out);
                    }
                }
                Value::FatRef { base, .. } | Value::DynRef { base, .. } => {
                    collect(base, out)
                }
                Value::FmtArg { target, .. } => collect(target, out),
                Value::Closure { env, .. } => {
                    for item in env {
                        collect(item, out);
                    }
                }
                Value::ErasedBlock { relocs, .. } => {
                    for (_, v) in relocs {
                        collect(v, out);
                    }
                }
                Value::FmtArguments(crate::value::FmtArguments::Template {
                    args,
                    ..
                }) => {
                    for arg in args {
                        collect(arg, out);
                    }
                }
                _ => {}
            }
        }
        let mut referenced = HashSet::new();
        for frame in segment {
            for local in &frame.locals {
                collect(local, &mut referenced);
            }
        }
        below
            .iter()
            .map(|f| f.id)
            .filter(|id| referenced.contains(id))
            .collect()
    }

    /// Run one thread for at most one quantum: start it if it
    /// has never run, resume it if it is parked, and drive until
    /// it finishes or its quantum expires. A finished thread's
    /// outcome is recorded (a guest panic in the thread is its
    /// outcome — the `Err` a joiner observes — not a failure of
    /// the run; any other error, an unsupported feature inside
    /// the thread, aborts the run, as it would on the entry
    /// thread). A thread whose quantum expires is parked: its
    /// frames move into its slot together with the ids of the
    /// frames below that they reference, and a later call resumes
    /// it on a stack that still holds those frames.
    pub(crate) fn run_thread(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
    ) -> InterpResult<ThreadRun> {
        enum Start {
            Fresh(Value),
            Resume(Vec<Frame>),
        }
        let base = frames.len();
        let start = {
            let mut threads = self.threads.borrow_mut();
            let Some(slot) = threads.get_mut(&id) else {
                return Ok(ThreadRun::Finished);
            };
            if let Some(parked) = slot.parked.take() {
                if parked
                    .below
                    .iter()
                    .all(|fid| frames.iter().any(|f| f.id == *fid))
                {
                    Start::Resume(parked.frames)
                } else {
                    slot.parked = Some(parked);
                    return Ok(ThreadRun::NotRunnable);
                }
            } else if let Some(callable) = slot.callable.take() {
                Start::Fresh(callable)
            } else {
                return Ok(ThreadRun::Finished);
            }
        };
        let previous = self.current_thread.get();
        self.current_thread.set(id);
        let previous_segment = self.segment.replace(SegmentMark {
            depth: self.drive_depth + 1,
            thread: id,
            steps0: self.steps,
        });
        let result = match start {
            Start::Fresh(callable) => {
                self.call_callable_to_completion(frames, &callable, Vec::new())
            }
            Start::Resume(parked_frames) => {
                frames.extend(parked_frames);
                self.drive_frames(frames, base)
            }
        };
        self.segment = previous_segment;
        if matches!(result, Err(InterpError::Yield)) {
            // The quantum expired mid-flight: park the segment's
            // frames (the drive left them above `base`) and the
            // ids of the frames below them. The thread has no
            // outcome yet, and its TLS destructors have not run:
            // they belong to its completion, below.
            let parked_frames = frames.split_off(base);
            let below = Self::referenced_below_frames(&parked_frames, frames);
            self.current_thread.set(previous);
            if let Some(slot) = self.threads.borrow_mut().get_mut(&id) {
                slot.parked = Some(ParkedThread {
                    frames: parked_frames,
                    below,
                });
            }
            return Ok(ThreadRun::Yielded);
        }
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
        Ok(ThreadRun::Finished)
    }

    /// One scheduling pass: every pending thread that can run
    /// against the current frame stack gets up to one quantum,
    /// in spawn order. This is the cooperative scheduling point:
    /// the running thread has reached a place where it would
    /// block, so the threads it might be waiting for get their
    /// turn. Callers that wait on a condition (the channel
    /// operations) loop over their condition and drain again, so
    /// one pass per drain is enough — a thread that yields is
    /// resumed by the next pass if the condition still does not
    /// hold. Returns whether any thread ran (a thread that ran
    /// out its quantum counts: it made progress).
    pub(crate) fn drain_threads(&mut self, frames: &mut Vec<Frame>) -> InterpResult<bool> {
        self.drain_depth += 1;
        let result = self.drain_threads_pass(frames);
        self.drain_depth -= 1;
        result
    }

    /// One scheduling pass (see [`Self::drain_threads`]).
    fn drain_threads_pass(&mut self, frames: &mut Vec<Frame>) -> InterpResult<bool> {
        let mut ran = false;
        let mut visited: HashSet<u64> = HashSet::new();
        loop {
            let next = self
                .threads
                .borrow()
                .iter()
                .filter(|(id, slot)| {
                    !visited.contains(id)
                        && (slot.callable.is_some() || slot.parked.is_some())
                })
                .map(|(id, _)| *id)
                .min();
            let Some(id) = next else {
                return Ok(ran);
            };
            visited.insert(id);
            match self.run_thread(frames, id)? {
                ThreadRun::Finished | ThreadRun::Yielded => ran = true,
                ThreadRun::NotRunnable => {}
            }
        }
    }

    /// Run scheduling passes until no pending thread remains or
    /// a pass makes no progress: the scope-exit drain, where
    /// every thread spawned inside the scope must finish before
    /// the scope returns. A pass with no progress means every
    /// remaining thread is parked above frames this stack no
    /// longer holds; those stay parked (their join, in a context
    /// that still holds the frames, can still resume them).
    /// Termination is bounded by the global step budget: every
    /// pass either finishes a thread or spends at least one
    /// quantum of steps on it.
    pub(crate) fn drain_threads_to_completion(
        &mut self,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<()> {
        loop {
            let ran = self.drain_threads(frames)?;
            let any_pending = self
                .threads
                .borrow()
                .values()
                .any(|slot| slot.callable.is_some() || slot.parked.is_some());
            if !any_pending || !ran {
                return Ok(());
            }
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
    pub(crate) fn run_tls_dtors(&mut self, frames: &mut Vec<Frame>, id: u64) -> InterpResult<()> {
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

    /// Join a thread: run it until it finishes, then return its
    /// recorded outcome. The joiner blocks natively, so the
    /// joined thread is resumed quantum after quantum; between
    /// quanta every other runnable thread gets a scheduling pass,
    /// as an OS scheduler would run them while the joiner waits —
    /// a joined thread that busy-waits on a third thread's work
    /// therefore still observes it. Each resumption spends at
    /// least one quantum of the global step budget, so the loop
    /// is bounded by it.
    pub(crate) fn join_thread(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
    ) -> InterpResult<Result<Value, PanicPayload>> {
        while self.thread_unfinished(id) {
            match self.run_thread(frames, id)? {
                ThreadRun::Finished => break,
                ThreadRun::Yielded => {
                    self.drain_threads(frames)?;
                }
                ThreadRun::NotRunnable => {
                    return Err(InterpError::UnsupportedTerminator {
                        kind: format!(
                            "join of thread {id} suspended above frames that are \
                             no longer on the stack (a suspended scoped thread \
                             whose borrowed frames have returned)"
                        ),
                    });
                }
            }
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
        let base = self.begin_callable(frames, callable, args)?;
        self.drive_frames(frames, base)
    }

    /// Like [`Self::call_callable_to_completion`], but the call
    /// runs as a scheduled region (see
    /// [`Self::drive_scheduled`]): for callables whose bodies
    /// are arbitrary guest programs that may themselves spawn
    /// threads and spin — test bodies, thread-scope closures.
    pub(crate) fn call_callable_scheduled(
        &mut self,
        frames: &mut Vec<Frame>,
        callable: &Value,
        args: Vec<Value>,
    ) -> InterpResult<Value> {
        let base = self.begin_callable(frames, callable, args)?;
        self.drive_scheduled(frames, base)
    }

    /// Push the frame for a callable value (a closure with its
    /// captured environment as its first argument, or a function
    /// pointer), yielding the stack depth the call's drive stops
    /// at.
    fn begin_callable(
        &mut self,
        frames: &mut Vec<Frame>,
        callable: &Value,
        args: Vec<Value>,
    ) -> InterpResult<usize> {
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
        Ok(base)
    }

    /// Drive the segment above `base` as a *scheduled region*:
    /// the region belongs to the current thread, so a scheduler
    /// yield raised inside it is served here — the runnable
    /// threads get a scheduling pass and the region resumes from
    /// exactly the stack the yield left — instead of escaping
    /// to the region's caller, whose Rust-side continuation (a
    /// builtin mid-computation, the harness mid-test) could not
    /// be parked. Regions that drive arbitrary guest code for an
    /// unbounded computation (the libtest harness's test
    /// bodies, a thread scope's closure, a `catch_unwind` try
    /// function) drive through a scheduled region; bounded
    /// synchronous machinery (comparators, `Once` closures,
    /// formatter bodies, waker wakes) keeps the raw drive, so a
    /// yield can only ever be raised — and served — at a
    /// boundary whose driver loops on it.
    fn drive_scheduled(
        &mut self,
        frames: &mut Vec<Frame>,
        base: usize,
    ) -> InterpResult<Value> {
        loop {
            let prev_segment = self.segment.replace(SegmentMark {
                depth: self.drive_depth + 1,
                thread: self.current_thread_id(),
                steps0: self.steps,
            });
            let result = self.drive_frames(frames, base);
            self.segment = prev_segment;
            match result {
                Err(InterpError::Yield) => {
                    self.drain_threads(frames)?;
                    continue;
                }
                other => return other,
            }
        }
    }
}
