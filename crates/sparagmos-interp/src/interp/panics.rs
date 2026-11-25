//! Guest panic handling.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{PanicPayload, RefTarget, Value};

impl<'a> Interpreter<'a> {
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
    pub(crate) fn fire_panic_hook(
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
    pub(crate) fn set_backtrace_style(&self, style: Value) {
        *self.backtrace_style.borrow_mut() = Some(style);
    }

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
}

/// Format an assertion message from its operand values, with
/// core's exact text (`core::panicking`'s messages for the
/// bounds-check and misaligned-dereference assertions).
/// Returns None when a value does not render as an integer,
/// leaving the caller its baked fallback text.
pub(crate) fn format_assert_message(
    kind: crate::ir::AssertFormat,
    vals: &[Value],
) -> Option<String> {
    fn num(v: &Value) -> Option<String> {
        match v {
            Value::Int(i) => Some(i.to_string()),
            Value::Uint(u) => Some(u.to_string()),
            _ => None,
        }
    }
    match kind {
        crate::ir::AssertFormat::BoundsCheck => Some(format!(
            "index out of bounds: the len is {} but the index is {}",
            num(vals.first()?)?,
            num(vals.get(1)?)?,
        )),
        crate::ir::AssertFormat::MisalignedPointerDeref => Some(format!(
            "misaligned pointer dereference: address must be a multiple of {} but is {}",
            num(vals.first()?)?,
            num(vals.get(1)?)?,
        )),
    }
}
