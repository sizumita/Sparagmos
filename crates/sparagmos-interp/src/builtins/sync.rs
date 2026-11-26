//! Synchronisation primitives: mpsc channels, barriers and thread spawning.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    channel_handle_id,
    Frame, Interpreter,
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
        let guard = self.resolve_value(frames, arg)?;
        let fields = match guard {
            Value::Enum { fields, .. } if !fields.is_empty() => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    name,
                    "a read guard aggregate",
                    other.kind_name(),
                ))
            }
        };
        let mut pointer = fields.into_iter().next().expect("len checked");
        // Peel the `NonNull` wrapper (and any further one-field
        // wrapper the constructor left around the pointer).
        loop {
            match pointer {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    pointer = fields.into_iter().next().expect("len checked");
                }
                // The pointer denotes the lock's `UnsafeCell`
                // place (std forms it by casting a pointer to
                // the cell, and a cast does not change the
                // denoted place in this model): the pointee is
                // the cell's field 0.
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => {
                    return project_pointee_cell(pointer);
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
