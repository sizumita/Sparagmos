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
}
