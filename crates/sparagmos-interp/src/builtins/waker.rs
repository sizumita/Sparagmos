//! Waker execution for the async builtins.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    rc_handle_id,
    Frame, Interpreter, WakeDispatch,
};
use crate::value::{RefTarget, Value};


impl<'a> Interpreter<'a> {
    /// Record the `Wake` / `LocalWake` impl methods a modelled
    /// waker built from the handle allocation `id` dispatches to:
    /// the impl methods of the pointee type `pointee` under the
    /// trait `trait_path`, named exactly as the driver names
    /// their bodies. Recorded by the modelled `From` conversion
    /// (the one place the pointee type is known; the waker value
    /// itself carries only the handle) and read by
    /// [`Self::exec_waker_wake`]. The driver demands the named
    /// bodies (see `demand_wake_instances` in the driver), so a
    /// wake on a live modelled waker always finds them: the
    /// handle the waker holds keeps the allocation - and with it
    /// this entry's meaning - alive for exactly the waker's own
    /// lifetime.
    pub(crate) fn note_waker_dispatch(&self, id: u64, pointee: &str, trait_path: &str) {
        self.wake_dispatch.borrow_mut().insert(
            id,
            WakeDispatch {
                wake: format!("<{pointee} as {trait_path}>::wake"),
                wake_by_ref: format!("<{pointee} as {trait_path}>::wake_by_ref"),
            },
        );
    }

    /// Execute a `Waker::wake` / `wake_by_ref` call (or the
    /// `LocalWaker` counterparts) on the waker `value`. The two
    /// shapes are judged exactly as the `Clone` model
    /// ([`Self::clone_waker_value`]) and the `WakerDrop` glue
    /// judge them:
    ///
    /// - The modelled `From<Arc<T: Wake>>` /
    ///   `From<Rc<T: LocalWake>>` waker is a one-field aggregate
    ///   carrying the handle itself: the wake runs the handle's
    ///   `Wake` / `LocalWake` impl method - `wake_by_ref` with a
    ///   reference to the stored handle, `wake` with the handle
    ///   itself, which the method consumes, exactly as the std
    ///   vtable entry hands the reconstructed handle to the same
    ///   method. A modelled waker whose allocation has no
    ///   recorded dispatch is an explicit error.
    /// - A `from_raw` waker holds a real `RawWaker` (as the
    ///   waker value itself or as its one field): the call runs
    ///   the vtable's wake (entry 1) / wake_by_ref (entry 2) with
    ///   the data pointer, exactly as std's own `Waker::wake`
    ///   body does.
    ///
    /// Any other shape is an explicit error.
    pub(super) fn exec_waker_wake(
        &mut self,
        frames: &mut Vec<Frame>,
        value: Value,
        by_ref: bool,
        name: &str,
    ) -> InterpResult<Value> {
        if let Value::Enum { fields, .. } = &value {
            if fields.len() == 1 {
                if let Some(id) = rc_handle_id(&fields[0]) {
                    let dispatch =
                        self.wake_dispatch.borrow().get(&id).cloned().ok_or_else(|| {
                            InterpError::type_mismatch(
                                format!("builtin {name}"),
                                "a waker whose Wake dispatch the modelled                                  From conversion recorded",
                                "a modelled waker with no recorded dispatch",
                            )
                        })?;
                    let handle = fields[0].clone();
                    return if by_ref {
                        let arg = Value::Ref(RefTarget::value(handle));
                        self.call_sync(frames, &dispatch.wake_by_ref, vec![arg])
                    } else {
                        self.call_sync(frames, &dispatch.wake, vec![handle])
                    };
                }
            }
        }
        let raw = match &value {
            Value::Enum { fields, .. } if fields.len() == 1 => fields[0].clone(),
            Value::Enum { .. } => value.clone(),
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a Waker value",
                    other.kind_name(),
                ));
            }
        };
        let Value::Enum { fields: raw_fields, .. } = &raw else {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWaker value",
                raw.kind_name(),
            ));
        };
        if raw_fields.len() != 2 {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWaker value (data pointer and vtable)",
                raw.kind_name(),
            ));
        }
        let data = raw_fields[0].clone();
        let resolved = self.deref_value(frames, &raw_fields[1])?;
        let vtable = self.read_resolved(frames, &resolved)?;
        let Value::Enum { fields: vtable_fields, .. } = &vtable else {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWakerVTable value",
                vtable.kind_name(),
            ));
        };
        if vtable_fields.len() != 4 {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWakerVTable value (clone, wake, wake_by_ref, drop)",
                vtable.kind_name(),
            ));
        }
        let entry = vtable_fields[if by_ref { 2 } else { 1 }].clone();
        if !matches!(entry, Value::FnPtr(_) | Value::Closure { .. }) {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWakerVTable with a function-pointer wake entry",
                entry.kind_name(),
            ));
        }
        self.call_callable_sync(frames, &entry, None, vec![data])
    }

    /// Clone a `Waker` / `LocalWaker` value the way std's manual
    /// `Clone` impls do. The modelled `From<Arc<T: Wake>>` /
    /// `From<Rc<T: LocalWake>>` waker is a one-field aggregate
    /// carrying the handle itself: the clone is another live
    /// handle on the same allocation, so the strong count moves
    /// and the value copies. A `from_raw` waker holds a real
    /// `RawWaker` (as the waker value itself or as its one
    /// non-zero-sized field): the clone calls the vtable's clone
    /// entry with the data pointer and wraps the `RawWaker` it
    /// returns in the same shape. Any other shape is an explicit
    /// error.
    pub(super) fn clone_waker_value(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        name: &str,
    ) -> InterpResult<Value> {
        if let Value::Enum { fields, .. } = value {
            if fields.len() == 1 {
                if let Some(id) = rc_handle_id(&fields[0]) {
                    self.rc_inc_strong(id)?;
                    return Ok(value.clone());
                }
            }
        }
        let (variant, wrapped, raw) = match value {
            Value::Enum { variant, fields } if fields.len() == 1 => {
                (*variant, true, fields[0].clone())
            }
            Value::Enum { .. } => (0, false, value.clone()),
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a Waker value",
                    other.kind_name(),
                ));
            }
        };
        let Value::Enum { fields: raw_fields, .. } = &raw else {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWaker value",
                raw.kind_name(),
            ));
        };
        if raw_fields.len() != 2 {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWaker value (data pointer and vtable)",
                raw.kind_name(),
            ));
        }
        let data = raw_fields[0].clone();
        let resolved = self.deref_value(frames, &raw_fields[1])?;
        let vtable = self.read_resolved(frames, &resolved)?;
        let Value::Enum { fields: vtable_fields, .. } = &vtable else {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWakerVTable value",
                vtable.kind_name(),
            ));
        };
        let Some(Value::FnPtr(clone_name)) = vtable_fields.first() else {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWakerVTable with a function-pointer clone entry",
                vtable.kind_name(),
            ));
        };
        let clone_name = clone_name.clone();
        let new_raw = self.call_sync(frames, &clone_name, vec![data])?;
        Ok(if wrapped {
            Value::Enum { variant, fields: vec![new_raw] }
        } else {
            new_raw
        })
    }
}
