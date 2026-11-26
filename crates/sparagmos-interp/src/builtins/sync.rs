//! Synchronisation primitives: mpsc channels, barriers and thread spawning.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    channel_handle_id,
    Frame, Interpreter, Resolved,
};
use crate::value::{RefProj, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// Register a thread spawn: the callable must be a closure or
    /// a function pointer (a `Box<dyn FnOnce>` has no body the
    /// model can run and stays an explicit error). Returns the
    /// join handle value.
    pub(super) fn spawn_callable(&mut self, callable: Value) -> InterpResult<Value> {
        match callable {
            Value::Closure { .. } | Value::FnPtr(_) => {
                let id = self.spawn_thread(callable);
                Ok(join_handle_value(id))
            }
            other => Err(unsupported(format!(
                "thread spawn of a {} value is not runtime-supported",
                other.kind_name()
            ))),
        }
    }

    /// The channel id of the handle an mpsc method's `&self`
    /// argument points at.
    pub(super) fn handle_id_of_ref(
        &self,
        frames: &[Frame],
        arg: &Value,
        name: &str,
    ) -> InterpResult<u64> {
        let target = self.deref_value(frames, arg)?;
        let value = self.read_resolved(frames, &target)?;
        channel_handle_id(&value).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "an mpsc channel handle",
                value.kind_name(),
            )
        })
    }

    /// Why a blocking channel operation cannot proceed: either
    /// every pending thread has already run (or none exists), or —
    /// when the block happens inside a spawned thread's run — the
    /// thread that must act next is the very thread that spawned
    /// this one and is itself blocked waiting for it: a cyclic
    /// wait no run-to-completion schedule can serve.
    fn block_forever_message(&self, name: &str, situation: &str) -> String {
        if self.current_thread_id() != 1 {
            format!(
                "{name} would block forever: {situation}; the thread that \
                 must act next is the blocked spawner of the running thread \
                 (a cyclic wait between threads is not servable by the \
                 single-threaded model)"
            )
        } else {
            format!(
                "{name} would block forever: {situation} (single-threaded model)"
            )
        }
    }

    /// The outcome of a blocking operation that cannot proceed
    /// after every scheduling round the cooperative scheduler
    /// can offer: where the running thread can be suspended
    /// (see [`Interpreter::suspend_permitted`]), suspend it —
    /// a later activation re-executes the operation and
    /// re-checks its condition, so the wait stays honest and
    /// no result is fabricated. Where it cannot (the entry
    /// thread, or code inside a detached synchronous drive),
    /// the would-block-forever error stands.
    pub(super) fn block_or_suspend(&self, name: &str, situation: &str) -> InterpError {
        if self.suspend_permitted() {
            InterpError::Suspended
        } else {
            unsupported(self.block_forever_message(name, situation))
        }
    }


    /// The wait at the heart of the contended sys lock
    /// acquisitions: run scheduling rounds until `attempt`
    /// takes the lock. A round that makes no progress leaves
    /// the wait unservable from here — one final attempt (the
    /// holder may have released the lock on its way into a
    /// block, which the round does not count as progress),
    /// then the outcome of [`Self::block_or_suspend`]: a
    /// spawned thread parks, and its re-activation re-runs the
    /// whole builtin, re-checking the word from scratch.
    pub(super) fn wait_for_sys_lock(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        situation: &str,
        mut attempt: impl FnMut(&mut Self, &mut Vec<Frame>) -> InterpResult<bool>,
    ) -> InterpResult<Value> {
        loop {
            if attempt(self, frames)? {
                return Ok(Value::Unit);
            }
            self.tick_step()?;
            let (_, progressed) = self.schedule_round(frames)?;
            if !progressed {
                if attempt(self, frames)? {
                    return Ok(Value::Unit);
                }
                return Err(self.block_or_suspend(name, situation));
            }
        }
    }

    /// `std::sys::sync::mutex::futex::Mutex::lock_contended`:
    /// the fast path's compare-exchange found the mutex held,
    /// so wait for the holder to release it and take it. The
    /// body's spin is the scheduler's business in this model —
    /// waiting means running the other threads until the word
    /// changes (see [`Self::wait_for_sys_lock`]).
    pub(super) fn sys_mutex_lock_contended(
        &mut self,
        frames: &mut Vec<Frame>,
        arg: &Value,
        name: &str,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, arg)?;
        self.wait_for_sys_lock(
            frames,
            name,
            "the mutex is locked and no pending thread can release it",
            move |this, frames| this.sys_mutex_contended_attempt(frames, &target, name),
        )
    }

    /// One acquisition attempt of the sys mutex's contended
    /// path: take the lock if the word is unlocked, otherwise
    /// mark it contended — the holder's unlock then takes its
    /// wake path — and report that the caller must wait. The
    /// read, the mark and the take are one builtin step: no
    /// other thread runs between them, so they compose exactly
    /// like the body's compare-exchange and swap.
    pub(super) fn sys_mutex_contended_attempt(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        name: &str,
    ) -> InterpResult<bool> {
        self.modify_resolved(frames, target, |value| {
            if sys_lock_word(value).is_none() {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a sys mutex value",
                    value.kind_name(),
                ));
            }
            let slot = sys_lock_word_mut(value).expect("the word shape was just read");
            let state = *slot as u32;
            if state == SYS_MUTEX_UNLOCKED {
                *slot = u128::from(SYS_MUTEX_LOCKED);
                Ok(true)
            } else {
                if state == SYS_MUTEX_LOCKED {
                    *slot = u128::from(SYS_MUTEX_CONTENDED);
                }
                Ok(false)
            }
        })
    }

    /// `std::sys::sync::rwlock::futex::RwLock::read_contended`:
    /// wait until the rwlock can be read-locked and take a
    /// read lock.
    pub(super) fn sys_rwlock_read_contended(
        &mut self,
        frames: &mut Vec<Frame>,
        arg: &Value,
        name: &str,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, arg)?;
        // The body's `has_slept`: any attempt past the first
        // in this activation runs after a wait (a scheduling
        // round), which is when the downgrade handoff case of
        // `is_read_lockable_after_wakeup` applies. A resumed
        // activation starts un-slept again, which only costs
        // the stricter check once.
        let mut attempts = 0usize;
        self.wait_for_sys_lock(
            frames,
            name,
            "the rwlock cannot be read-locked and no pending thread can change that",
            move |this, frames| {
                attempts += 1;
                this.sys_rwlock_read_attempt(frames, &target, attempts > 1, name)
            },
        )
    }

    /// One acquisition attempt of the sys rwlock's contended
    /// read path (see [`Self::sys_rwlock_read_contended`]):
    /// take a read lock when the state allows it, otherwise
    /// set the readers-waiting bit and report that the caller
    /// must wait.
    pub(super) fn sys_rwlock_read_attempt(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        has_slept: bool,
        name: &str,
    ) -> InterpResult<bool> {
        self.modify_resolved(frames, target, |value| {
            if sys_rwlock_state(value).is_none() {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a sys rwlock value",
                    value.kind_name(),
                ));
            }
            let slot = sys_rwlock_state_mut(value).expect("the word shape was just read");
            let state = *slot as u32;
            if (has_slept && sys_rwlock_is_read_lockable_after_wakeup(state))
                || sys_rwlock_is_read_lockable(state)
            {
                *slot = u128::from(state + SYS_RWLOCK_READ_LOCKED);
                Ok(true)
            } else if state & SYS_RWLOCK_MASK == SYS_RWLOCK_MAX_READERS {
                // The body's assertion: the reader count is
                // full.
                Err(InterpError::Panic {
                    message: "too many active read locks on RwLock".to_string(),
                    payload: None,
                })
            } else {
                if state & SYS_RWLOCK_READERS_WAITING == 0 {
                    *slot = u128::from(state | SYS_RWLOCK_READERS_WAITING);
                }
                Ok(false)
            }
        })
    }

    /// `std::sys::sync::rwlock::futex::RwLock::write_contended`:
    /// wait until the rwlock is unlocked and take the write
    /// lock.
    pub(super) fn sys_rwlock_write_contended(
        &mut self,
        frames: &mut Vec<Frame>,
        arg: &Value,
        name: &str,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, arg)?;
        // The body's `other_writers_waiting`: once this call
        // has waited, the writers-waiting bit stays on across
        // its own acquisition, so the unlock that follows
        // routes the wake to the next writer.
        let mut other_writers_waiting = 0u32;
        self.wait_for_sys_lock(
            frames,
            name,
            "the rwlock is locked and no pending thread can release it",
            move |this, frames| {
                this.sys_rwlock_write_attempt(
                    frames,
                    &target,
                    &mut other_writers_waiting,
                    name,
                )
            },
        )
    }

    /// One acquisition attempt of the sys rwlock's contended
    /// write path (see [`Self::sys_rwlock_write_contended`]):
    /// take the write lock when the state is unlocked,
    /// otherwise set the writers-waiting bit and report that
    /// the caller must wait.
    pub(super) fn sys_rwlock_write_attempt(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        other_writers_waiting: &mut u32,
        name: &str,
    ) -> InterpResult<bool> {
        self.modify_resolved(frames, target, |value| {
            if sys_rwlock_state(value).is_none() {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a sys rwlock value",
                    value.kind_name(),
                ));
            }
            let slot = sys_rwlock_state_mut(value).expect("the word shape was just read");
            let state = *slot as u32;
            if sys_rwlock_is_unlocked(state) {
                *slot = u128::from(
                    state | SYS_RWLOCK_WRITE_LOCKED | *other_writers_waiting,
                );
                Ok(true)
            } else {
                if state & SYS_RWLOCK_WRITERS_WAITING == 0 {
                    *slot = u128::from(state | SYS_RWLOCK_WRITERS_WAITING);
                }
                *other_writers_waiting = SYS_RWLOCK_WRITERS_WAITING;
                Ok(false)
            }
        })
    }

    /// `std::sys::sync::rwlock::futex::RwLock::wake_writer_or_readers`:
    /// the caller's unlock left the lock unlocked with the
    /// waiting bits of `state` set; route the next
    /// acquisition, exactly as the body does: a lone waiting
    /// writer is handed the lock through the writer-notify
    /// word, waiting readers keep priority over a second
    /// writer only when no writer could be woken, and readers
    /// alone are woken by clearing their bit — in this model
    /// the cleared word, observed at the waiters' next
    /// activation, is the wake.
    pub(super) fn sys_rwlock_wake_writer_or_readers(
        &self,
        frames: &mut Vec<Frame>,
        arg: &Value,
        state: u32,
        name: &str,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, arg)?;
        let mut state = state;
        // If only writers are waiting, wake one of them.
        if state == SYS_RWLOCK_WRITERS_WAITING {
            match self.sys_rwlock_state_cas(frames, &target, state, 0, name)? {
                Ok(()) => {
                    self.sys_rwlock_wake_writer(frames, &target, name)?;
                    return Ok(Value::Unit);
                }
                Err(current) => state = current,
            }
        }
        // If both writers and readers are waiting, leave the
        // readers waiting and wake one writer.
        if state == SYS_RWLOCK_READERS_WAITING + SYS_RWLOCK_WRITERS_WAITING {
            if self
                .sys_rwlock_state_cas(
                    frames,
                    &target,
                    state,
                    SYS_RWLOCK_READERS_WAITING,
                    name,
                )?
                .is_err()
            {
                // The lock got locked in the meantime; its new
                // holder routes the next wake.
                return Ok(Value::Unit);
            }
            if self.sys_rwlock_wake_writer(frames, &target, name)? {
                return Ok(Value::Unit);
            }
            state = SYS_RWLOCK_READERS_WAITING;
        }
        // If readers are waiting, wake them all.
        if state == SYS_RWLOCK_READERS_WAITING {
            let _ = self.sys_rwlock_state_cas(frames, &target, state, 0, name)?;
        }
        Ok(Value::Unit)
    }

    /// The compare-exchange on the sys rwlock's state word
    /// that the wake routing is built from: `Ok(())` when the
    /// word held `expected` and now holds `new`, `Err(word)`
    /// with the word as it stands otherwise. One builtin
    /// step, so no other thread runs between the check and
    /// the set.
    pub(super) fn sys_rwlock_state_cas(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        expected: u32,
        new: u32,
        name: &str,
    ) -> InterpResult<Result<(), u32>> {
        self.modify_resolved(frames, target, |value| {
            if sys_rwlock_state(value).is_none() {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a sys rwlock value",
                    value.kind_name(),
                ));
            }
            let slot = sys_rwlock_state_mut(value).expect("the word shape was just read");
            let state = *slot as u32;
            if state == expected {
                *slot = u128::from(new);
                Ok(Ok(()))
            } else {
                Ok(Err(state))
            }
        })
    }

    /// Wake one writer waiting on the sys rwlock: bump the
    /// writer-notify word, whose change is what a waiting
    /// writer's next activation observes. Returns the futex
    /// wake's result: whether a writer was actually blocked
    /// on the notify word. This model keeps no per-lock wait
    /// queue, so it cannot know — and, like the platforms std
    /// notes always report `false`, it reports that none was
    /// woken: the caller's fallback then wakes the readers as
    /// well, and every waiter re-checks the state word at its
    /// next activation, so the extra wake is unobservable.
    pub(super) fn sys_rwlock_wake_writer(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        name: &str,
    ) -> InterpResult<bool> {
        self.modify_resolved(frames, target, |value| {
            if sys_rwlock_notify_word(value).is_none() {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a sys rwlock value",
                    value.kind_name(),
                ));
            }
            let slot =
                sys_rwlock_notify_word_mut(value).expect("the word shape was just read");
            *slot = u128::from((*slot as u32).wrapping_add(1));
            Ok(false)
        })
    }


    /// `std::sys::sync::futex::unix::futex_wait`: wait until
    /// the futex word differs from `expected` — the platform
    /// primitive the sys locks, `Once`, the condvar and the
    /// thread parker sleep on. Returns at once (`true`) when
    /// the word has already changed; otherwise the wait runs
    /// scheduling rounds like the lock waits (see
    /// [`Self::wait_for_sys_lock`]). A timed wait whose word
    /// no pending thread can change reports the timeout
    /// (`false`): natively the thread would sleep on until
    /// its deadline with the same outcome, and no thread
    /// could have woken it in between.
    pub(super) fn futex_wait(
        &mut self,
        frames: &mut Vec<Frame>,
        arg: &Value,
        expected: u32,
        has_timeout: bool,
        name: &str,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, arg)?;
        loop {
            if self.futex_word_differs(frames, &target, expected, name)? {
                return Ok(Value::Bool(true));
            }
            self.tick_step()?;
            let (_, progressed) = self.schedule_round(frames)?;
            if !progressed {
                if self.futex_word_differs(frames, &target, expected, name)? {
                    return Ok(Value::Bool(true));
                }
                if has_timeout {
                    return Ok(Value::Bool(false));
                }
                return Err(self.block_or_suspend(
                    name,
                    "the futex word is unchanged and no pending thread can change it",
                ));
            }
        }
    }

    /// Whether the futex word `target` denotes currently
    /// differs from `expected` (the check `futex_wait` makes
    /// before sleeping and after every wake).
    pub(super) fn futex_word_differs(
        &self,
        frames: &[Frame],
        target: &Resolved,
        expected: u32,
        name: &str,
    ) -> InterpResult<bool> {
        let value = self.read_resolved(frames, target)?;
        let word = sys_lock_word(&value).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "a futex word",
                value.kind_name(),
            )
        })?;
        Ok(word != expected)
    }

    /// `Sender::send` / `SyncSender::send`: enqueue the value,
    /// running pending threads while a bounded channel is full
    /// (the sender would block; the threads it waits for run
    /// first). Disconnection hands the value back inside a
    /// `SendError`.
    pub(super) fn mpsc_send(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
        value: Value,
        name: &str,
    ) -> InterpResult<Value> {
        enum Step {
            Sent,
            Disconnected,
            Full,
            Unknown,
        }
        loop {
            let step = match self.with_channel(id, |st| {
                if st.receivers == 0 {
                    Step::Disconnected
                } else if st.bound.is_none_or(|b| st.queue.len() < b) {
                    st.queue.push_back(value.clone());
                    Step::Sent
                } else {
                    Step::Full
                }
            }) {
                Some(step) => step,
                None => Step::Unknown,
            };
            match step {
                Step::Sent => return Ok(ok_result(Value::Unit)),
                Step::Disconnected => {
                    return Ok(err_result(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    }))
                }
                Step::Full => {
                    // Give the other threads a scheduling round;
                    // a receiver among them may make room. When
                    // a round changes nothing, the send cannot
                    // proceed (see `block_or_suspend`).
                    self.tick_step()?;
                    let (_, progressed) = self.schedule_round(frames)?;
                    if !progressed {
                        return Err(self.block_or_suspend(
                            name,
                            "the channel is full and no pending thread can receive",
                        ));
                    }
                }
                Step::Unknown => {
                    return Err(unsupported(format!("{name} on an unknown channel")))
                }
            }
        }
    }

    /// `Receiver::recv`: dequeue the oldest value, running
    /// pending threads while the channel is empty but senders
    /// live. `Err(())` is disconnection (queue empty and the last
    /// sender gone).
    pub(super) fn mpsc_recv(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
        name: &str,
    ) -> InterpResult<Result<Value, ()>> {
        loop {
            let step = self.with_channel(id, |st| (st.queue.pop_front(), st.senders));
            let Some((popped, senders)) = step else {
                return Err(unsupported(format!("{name} on an unknown channel")));
            };
            if let Some(value) = popped {
                return Ok(Ok(value));
            }
            if senders == 0 {
                return Ok(Err(()));
            }
            // Give the other threads a scheduling round; a
            // sender among them may produce (or drop the last
            // sender handle). When a round changes nothing, the
            // receive cannot proceed (see `block_or_suspend`).
            self.tick_step()?;
            let (_, progressed) = self.schedule_round(frames)?;
            if !progressed {
                return Err(self.block_or_suspend(
                    name,
                    "the channel is empty and no pending thread can send",
                ));
            }
        }
    }

    /// `Receiver::recv_timeout`: like `mpsc_recv`, except that
    /// when no pending thread can produce a value the outcome is
    /// a timeout (`Err(0)`) rather than an error — the model
    /// cannot wait, and a native run whose producers have all
    /// finished observes exactly that. Disconnection is `Err(1)`.
    pub(super) fn mpsc_recv_or(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
        name: &str,
    ) -> InterpResult<Result<Value, u32>> {
        loop {
            let step = self.with_channel(id, |st| (st.queue.pop_front(), st.senders));
            let Some((popped, senders)) = step else {
                return Err(unsupported(format!("{name} on an unknown channel")));
            };
            if let Some(value) = popped {
                return Ok(Ok(value));
            }
            if senders == 0 {
                return Ok(Err(1));
            }
            // Like `mpsc_recv`, but a round that changes
            // nothing is the timeout outcome, not an error.
            self.tick_step()?;
            let (_, progressed) = self.schedule_round(frames)?;
            if !progressed {
                return Ok(Err(0));
            }
        }
    }

    /// `Deref` / `DerefMut` for a lock guard whose first field
    /// is a reference to the lock itself (`MutexGuard`,
    /// `RwLockWriteGuard`, in the shape the converted lock
    /// bodies build them): the guarded value is the lock
    /// aggregate's data field -- field 2, the `UnsafeCell` --
    /// and the reference denotes the cell's held value (its
    /// field 0), a live place, so reads and writes through the
    /// result reach the storage every guard of the lock shares.
    pub(super) fn lock_guard_deref(
        &self,
        frames: &[Frame],
        arg: &Value,
        name: &str,
    ) -> InterpResult<Value> {
        let guard = self.resolve_value(frames, arg)?;
        let fields = match guard {
            Value::Enum { fields, .. } if fields.len() >= 2 => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    name,
                    "a lock guard aggregate",
                    other.kind_name(),
                ))
            }
        };
        let lock_ref = fields.into_iter().next().expect("len checked");
        let target = match lock_ref {
            Value::Ref(target) => target,
            other => {
                return Err(InterpError::type_mismatch(
                    name,
                    "a reference to the lock",
                    other.kind_name(),
                ))
            }
        };
        let data = target
            .projected(RefProj::Field(2))
            .and_then(|t| t.projected(RefProj::Field(0)))
            .ok_or_else(|| {
                unsupported(format!(
                    "{name}: the guard's lock reference denotes no live place"
                ))
            })?;
        Ok(Value::Ref(data))
    }

    /// `Deref` for `RwLockReadGuard`: unlike the other lock
    /// guards, its first field is not the lock reference but a
    /// `NonNull` aggregate holding the data pointer the guard's
    /// constructor took from the lock's `UnsafeCell`. That
    /// pointer denotes the cell's place (see
    /// `project_pointee_cell`), so the pointee is the cell's
    /// field 0.
    pub(super) fn read_guard_deref(
        &self,
        frames: &[Frame],
        arg: &Value,
        name: &str,
    ) -> InterpResult<Value> {
        // The pointer denotes the lock's `UnsafeCell` place
        // (std forms it by casting a pointer to the cell, and
        // a cast does not change the denoted place in this
        // model): the pointee is the cell's field 0.
        let pointer = self.guard_data_pointer(frames, arg, name)?;
        project_pointee_cell(pointer)
    }

    /// `Deref` / `DerefMut` for a mapped lock guard
    /// (`MappedMutexGuard`, `MappedRwLockReadGuard`,
    /// `MappedRwLockWriteGuard`): like a read guard, its first
    /// field is the `NonNull` data pointer its constructor
    /// stored, but that pointer already denotes the mapped
    /// component itself -- `map` built it from the reference
    /// the mapping closure returned, which names the exact
    /// sub-place inside the lock's data -- so the pointee is
    /// the pointer itself, with no `UnsafeCell` projection.
    pub(super) fn mapped_guard_deref(
        &self,
        frames: &[Frame],
        arg: &Value,
        name: &str,
    ) -> InterpResult<Value> {
        self.guard_data_pointer(frames, arg, name)
    }

    /// The data pointer a guard aggregate stores in its first
    /// field (a read guard's or a mapped guard's): the field
    /// is a `NonNull`, so peel the one-field wrappers around
    /// the pointer value the constructor left there.
    fn guard_data_pointer(
        &self,
        frames: &[Frame],
        arg: &Value,
        name: &str,
    ) -> InterpResult<Value> {
        let guard = self.resolve_value(frames, arg)?;
        let fields = match guard {
            Value::Enum { fields, .. } if !fields.is_empty() => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    name,
                    "a guard aggregate",
                    other.kind_name(),
                ))
            }
        };
        let mut pointer = fields.into_iter().next().expect("len checked");
        loop {
            match pointer {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    pointer = fields.into_iter().next().expect("len checked");
                }
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => {
                    return Ok(pointer);
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        name,
                        "a data pointer",
                        other.kind_name(),
                    ))
                }
            }
        }
    }
}

/// Projects a lock guard's data pointer through the
/// `UnsafeCell` it denotes: std forms the pointer by casting
/// `UnsafeCell::get`'s result, and a pointer cast does not
/// change the denoted place in this model, so the raw pointer
/// a read guard stores denotes the lock's cell place and the
/// guarded value is that place's field 0. A fat or dyn
/// pointer's base is projected the same way.
pub(super) fn project_pointee_cell(pointer: Value) -> InterpResult<Value> {
    Ok(match pointer {
        Value::Ref(target) => Value::Ref(target.projected(RefProj::Field(0)).ok_or_else(
            || unsupported("a lock guard data pointer with no live place"),
        )?),
        Value::FatRef { base, len } => Value::FatRef {
            base: Box::new(project_pointee_cell(*base)?),
            len,
        },
        Value::DynRef {
            base,
            concrete,
            traits,
            drop_glue,
        } => Value::DynRef {
            base: Box::new(project_pointee_cell(*base)?),
            concrete,
            traits,
            drop_glue,
        },
        other => other,
    })
}
/// A `std::sync::Barrier` handle value denoting
/// barrier-table id `id`: the id in the handle's only slot.
/// The guest treats the value opaquely -- it is stored,
/// shared through `Arc`, and handed back to `Barrier::wait`
/// -- and the barrier's state lives in the table.
pub(super) fn barrier_value(id: u64) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Uint(u128::from(id))],
    }
}

/// The barrier-table id a `Barrier` handle value carries.
/// `None` for any other shape.
pub(super) fn barrier_handle_id(value: &Value) -> Option<u64> {
    let fields = match value {
        Value::Enum { fields, .. } | Value::Tuple(fields) => fields,
        _ => return None,
    };
    match fields.first() {
        Some(Value::Uint(id)) => Some(*id as u64),
        Some(Value::Int(id)) => Some(*id as u64),
        _ => None,
    }
}

/// A `std::sync::BarrierWaitResult` value: its one field
/// is the leader flag `Barrier::wait` reports.
pub(super) fn barrier_wait_result_value(leader: bool) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Bool(leader)],
    }
}

/// The child-table id a `Child` value carries: the first
/// slot of its handle field. `None` for any other shape.
pub(super) fn child_handle_id(value: &Value) -> Option<u64> {
    let fields = match value {
        Value::Enum { fields, .. } | Value::Tuple(fields) => fields,
        _ => return None,
    };
    let handle = fields.first()?;
    let handle_fields = match handle {
        Value::Enum { fields, .. } | Value::Tuple(fields) => fields,
        _ => return None,
    };
    match handle_fields.first() {
        Some(Value::Uint(id)) => Some(*id as u64),
        Some(Value::Int(id)) => Some(*id as u64),
        _ => None,
    }
}

// ----- The futex-backed sys locks (std's `sys::sync` layer) -----

/// The sys mutex's futex word states (std's
/// `sys::sync::mutex::futex`): unlocked, locked, and locked
/// with waiters.
const SYS_MUTEX_UNLOCKED: u32 = 0;
const SYS_MUTEX_LOCKED: u32 = 1;
const SYS_MUTEX_CONTENDED: u32 = 2;

/// The sys rwlock's state word layout (std's
/// `sys::sync::rwlock::futex`): the low 30 bits count the
/// readers (all bits set is the write lock), bit 30 marks
/// readers waiting and bit 31 marks writers waiting. The
/// second word, `writer_notify`, is a sequence counter
/// writers wait on.
const SYS_RWLOCK_READ_LOCKED: u32 = 1;
const SYS_RWLOCK_MASK: u32 = (1 << 30) - 1;
const SYS_RWLOCK_WRITE_LOCKED: u32 = SYS_RWLOCK_MASK;
const SYS_RWLOCK_MAX_READERS: u32 = SYS_RWLOCK_MASK - 1;
const SYS_RWLOCK_READERS_WAITING: u32 = 1 << 30;
const SYS_RWLOCK_WRITERS_WAITING: u32 = 1 << 31;

/// Whether the sys rwlock state is unlocked (std's
/// `is_unlocked`).
pub(super) fn sys_rwlock_is_unlocked(state: u32) -> bool {
    state & SYS_RWLOCK_MASK == 0
}

/// Whether a reader may take the sys rwlock in `state`
/// (std's `is_read_lockable`).
pub(super) fn sys_rwlock_is_read_lockable(state: u32) -> bool {
    state & SYS_RWLOCK_MASK < SYS_RWLOCK_MAX_READERS
        && state & SYS_RWLOCK_READERS_WAITING == 0
        && state & SYS_RWLOCK_WRITERS_WAITING == 0
}

/// Whether a woken reader may take the sys rwlock in
/// `state` (std's `is_read_lockable_after_wakeup`: the
/// downgrade handoff case, where the lock is already
/// read-locked and writers may be waiting).
pub(super) fn sys_rwlock_is_read_lockable_after_wakeup(state: u32) -> bool {
    state & SYS_RWLOCK_MASK < SYS_RWLOCK_MAX_READERS
        && state & SYS_RWLOCK_READERS_WAITING == 0
        && state & SYS_RWLOCK_MASK != SYS_RWLOCK_WRITE_LOCKED
        && !sys_rwlock_is_unlocked(state)
}

/// The futex word at the bottom of a sys lock field's
/// wrapper chain: a sys lock stores each futex as an atomic
/// cell around the word, and the conversion's value for the
/// chain is single-field aggregates down to the word
/// itself. `None` for any other shape.
pub(super) fn sys_lock_word(value: &Value) -> Option<u32> {
    let mut cur = value;
    loop {
        match cur {
            Value::Uint(n) => return Some(*n as u32),
            Value::Enum { fields, .. } if fields.len() == 1 => cur = &fields[0],
            Value::Tuple(fields) if fields.len() == 1 => cur = &fields[0],
            _ => return None,
        }
    }
}

/// The mutable futex word at the bottom of a sys lock
/// field's wrapper chain (see [`sys_lock_word`]).
pub(super) fn sys_lock_word_mut(value: &mut Value) -> Option<&mut u128> {
    match value {
        Value::Uint(n) => Some(n),
        Value::Enum { fields, .. } if fields.len() == 1 => {
            sys_lock_word_mut(fields.get_mut(0)?)
        }
        Value::Tuple(fields) if fields.len() == 1 => {
            sys_lock_word_mut(fields.get_mut(0)?)
        }
        _ => None,
    }
}

/// Field `index` of a sys lock value (the sys rwlock's two
/// futex fields, in declaration order).
pub(super) fn sys_lock_field(value: &Value, index: usize) -> Option<&Value> {
    match value {
        Value::Enum { fields, .. } | Value::Tuple(fields) => fields.get(index),
        _ => None,
    }
}

/// Mutable field `index` of a sys lock value (see
/// [`sys_lock_field`]).
pub(super) fn sys_lock_field_mut(value: &mut Value, index: usize) -> Option<&mut Value> {
    match value {
        Value::Enum { fields, .. } | Value::Tuple(fields) => fields.get_mut(index),
        _ => None,
    }
}

/// The sys rwlock's `state` word (its first field).
pub(super) fn sys_rwlock_state(value: &Value) -> Option<u32> {
    sys_lock_word(sys_lock_field(value, 0)?)
}

/// The mutable sys rwlock `state` word (see
/// [`sys_rwlock_state`]).
pub(super) fn sys_rwlock_state_mut(value: &mut Value) -> Option<&mut u128> {
    sys_lock_word_mut(sys_lock_field_mut(value, 0)?)
}

/// The sys rwlock's `writer_notify` word (its second
/// field).
pub(super) fn sys_rwlock_notify_word(value: &Value) -> Option<u32> {
    sys_lock_word(sys_lock_field(value, 1)?)
}

/// The mutable sys rwlock `writer_notify` word (see
/// [`sys_rwlock_notify_word`]).
pub(super) fn sys_rwlock_notify_word_mut(value: &mut Value) -> Option<&mut u128> {
    sys_lock_word_mut(sys_lock_field_mut(value, 1)?)
}

/// Registration fragment of the builtin name registry: the futex-backed synchronisation surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn sync_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        // The thread surface (cooperative model, see `interp.rs`).
        // Registering these names also keeps the driver from
        // collecting the std thread machinery, whose internals
        // (spawn hooks, OS threads, packet transmutes) the value
        // model does not serve; the builtins model the observable
        // semantics instead.
        "std::thread::spawn" | "std::thread::spawn_unchecked" => Builtin::ThreadSpawn,
        "std::sync::Barrier::new" => Builtin::BarrierNew,
        "std::sync::Barrier::wait" => Builtin::BarrierWait,
        // The futex-backed sys locks (std's `sys::sync`
        // layer): the fast paths are plain atomic operations
        // the converted bodies serve, but the contended paths
        // sleep on a futex and have no converted body; the
        // builtins model the waiting on the cooperative
        // scheduler instead (see `builtins::sync`).
        "std::sys::sync::mutex::futex::Mutex::lock_contended" => {
            Builtin::SysMutexLockContended
        }
        "std::sys::sync::mutex::futex::Mutex::wake" => Builtin::SysMutexWake,
        "std::sys::sync::rwlock::futex::RwLock::read_contended" => {
            Builtin::SysRwLockReadContended
        }
        "std::sys::sync::rwlock::futex::RwLock::write_contended" => {
            Builtin::SysRwLockWriteContended
        }
        "std::sys::sync::rwlock::futex::RwLock::wake_writer_or_readers" => {
            Builtin::SysRwLockWakeWriterOrReaders
        }
        // The futex primitives themselves: bodies that ARE
        // converted (the rwlock's `downgrade`, `Once`, the
        // condvar, the thread parker) call them directly.
        "std::sys::sync::futex::unix::futex_wait" => Builtin::FutexWait,
        "std::sys::sync::futex::unix::futex_wake" => Builtin::FutexWake,
        "std::sys::sync::futex::unix::futex_wake_all" => Builtin::FutexWakeAll,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the thread surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn thread_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        _ if canonical.starts_with("std::thread::Builder")
            && (canonical.ends_with("::spawn")
                || canonical.ends_with("::spawn_unchecked")) =>
        {
            Builtin::ThreadBuilderSpawn
        }
        _ if canonical.starts_with("std::thread::JoinHandle")
            && canonical.ends_with("::join") =>
        {
            Builtin::ThreadJoin
        }
        // Optimizing MIR inlines `JoinHandle::join` into a
        // direct call to the inner type's join; it has the same
        // contract (the thread's result or panic payload) and
        // the same modelled receiver (see `join_handle_value`).
        _ if canonical.starts_with("std::thread::lifecycle::JoinInner")
            && canonical.ends_with("::join") =>
        {
            Builtin::ThreadJoin
        }
        _ if canonical.starts_with("std::thread::ScopedJoinHandle")
            && canonical.ends_with("::join") =>
        {
            Builtin::ScopedThreadJoin
        }
        _ if canonical.contains("::Scope::") && canonical.ends_with("::spawn") => {
            Builtin::ThreadScopedSpawn
        }
        "std::sys::thread_local::destructors::linux_like::register" => {
            Builtin::TlsRegisterDtor
        }
        "std::thread::local::panic_access_error" => Builtin::PanicAccessError,
        "std::thread::Builder::new" => Builtin::ThreadBuilderNew,
        "std::thread::Builder::name" => Builtin::ThreadBuilderName,
        "std::thread::Builder::stack_size" => Builtin::ThreadBuilderStackSize,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the current-thread surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn thread_api_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::thread::scope" => Builtin::ThreadScope,
        "std::thread::current" => Builtin::ThreadCurrent,
        "std::thread::Thread::id" => Builtin::ThreadIdOf,
        "std::thread::Thread::name" => Builtin::ThreadName,
        "std::thread::sleep" | "std::thread::sleep_ms" => Builtin::ThreadSleep,
        "std::thread::yield_now" => Builtin::ThreadYield,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the mpsc channel surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn mpsc_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        // The mpsc surface: the channel state lives in the
        // interpreter (see `ChannelState` in `interp.rs`).
        "std::sync::mpsc::channel" => Builtin::MpscChannel,
        "std::sync::mpsc::sync_channel" => Builtin::MpscSyncChannel,
        _ if canonical.contains("mpsc::SyncSender") && canonical.ends_with("::send") => {
            Builtin::MpscSend
        }
        _ if canonical.contains("mpsc::Sender") && canonical.ends_with("::send") => {
            Builtin::MpscSend
        }
        _ if canonical.contains("mpsc::SyncSender")
            && canonical.ends_with("::try_send") =>
        {
            Builtin::MpscTrySend
        }
        _ if canonical.contains("mpsc::Receiver") && canonical.ends_with("::recv") => {
            Builtin::MpscRecv
        }
        _ if canonical.contains("mpsc::Receiver")
            && canonical.ends_with("::try_recv") =>
        {
            Builtin::MpscTryRecv
        }
        _ if canonical.contains("mpsc::Receiver")
            && canonical.ends_with("::recv_timeout") =>
        {
            Builtin::MpscRecvTimeout
        }
        _ if canonical.contains("mpsc::Receiver") && canonical.ends_with("::iter") => {
            Builtin::MpscIter
        }
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the `Rc` and `Arc` constructors.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn rc_arc_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        _ if canonical.starts_with("std::rc::Rc") && canonical.ends_with(">::new") => {
            Builtin::RcNew
        }
        _ if canonical.starts_with("std::sync::Arc") && canonical.ends_with(">::new") => {
            Builtin::ArcNew
        }
        _ if canonical.starts_with("std::rc::Rc") && canonical.ends_with(">::from_raw") => {
            Builtin::RcFromRaw
        }
        _ if canonical.starts_with("std::sync::Arc") && canonical.ends_with(">::from_raw") => {
            Builtin::ArcFromRaw
        }
        _ if (canonical.starts_with("std::rc::Weak")
            || canonical.starts_with("std::sync::Weak"))
            && canonical.ends_with(">::from_raw") =>
        {
            Builtin::WeakFromRaw
        }
        _ => return None,
    })
}
