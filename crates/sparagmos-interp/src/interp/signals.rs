//! The signal model: dispositions installed through
//! `sigaction`, synchronous delivery on `raise`, and the
//! `atexit` exit-handler list.
//!
//! The interpreter keeps no host signal state of its own: a
//! guest's signal dispositions are interpreter state, and
//! delivery is synchronous — `raise` runs the installed
//! handler to completion on the current frame stack before
//! returning, which is the observable behaviour of a signal
//! arriving while the guest runs (the handler observes the
//! guest's state, and its effects are visible to the code
//! that raised). The alternate stack a handler was installed
//! for (`SA_ONSTACK`) has no representation: execution runs
//! on the frame stack either way, and nothing a guest can
//! observe names the difference. Blocking masks
//! (`sigprocmask`, the action's `sa_mask`) and the
//! `sa_restorer` trampoline are likewise unmodelled: the
//! recorded action keeps the installed structure verbatim
//! for `oldact` write-back, but delivery consults only the
//! handler and the `SA_SIGINFO` flag.

use super::*;
use crate::error::{InterpError, InterpResult};
use crate::value::Value;

/// A signal disposition installed through the `sigaction`
/// model: the handler field exactly as the guest wrote it
/// (an exposed function address or a function value, or the
/// `SIG_DFL` / `SIG_IGN` sentinel integers), the `sa_flags`
/// word, and the action structure exactly as read, which a
/// later call's `oldact` write-back hands back verbatim.
#[derive(Clone)]
pub(crate) struct SignalAction {
    pub(crate) handler: Value,
    pub(crate) flags: i64,
    pub(crate) act: Value,
}

/// The `SA_SIGINFO` flag: the handler takes the
/// three-argument (signum, siginfo, context) form.
const SA_SIGINFO: i64 = 0x4;

/// The default disposition of a signal with no installed
/// handler, from the Linux table: most signals terminate
/// the process, a few are ignored, the job-control signals
/// stop it, and `SIGCONT`'s resume is a no-op for a model
/// that never stops.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DefaultDisposition {
    Ignore,
    Terminate,
    Stop,
}

fn default_disposition(signum: i64) -> DefaultDisposition {
    match signum {
        // SIGCHLD (17), SIGCONT (18), SIGURG (23) and
        // SIGWINCH (28) never interrupt the guest.
        17 | 18 | 23 | 28 => DefaultDisposition::Ignore,
        // SIGSTOP (19), SIGTSTP (20), SIGTTIN (21),
        // SIGTTOU (22).
        19 | 20 | 21 | 22 => DefaultDisposition::Stop,
        _ => DefaultDisposition::Terminate,
    }
}

impl<'a> Interpreter<'a> {
    /// Install `new` as the disposition of `signum` — or,
    /// with `None`, only query it — returning the previously
    /// installed disposition, if any.
    pub(crate) fn signal_install(
        &self,
        signum: i64,
        new: Option<SignalAction>,
    ) -> Option<SignalAction> {
        let mut table = self.signal_actions.borrow_mut();
        match new {
            Some(action) => table.insert(signum, action),
            None => table.get(&signum).cloned(),
        }
    }

    /// The callable a value denotes as a signal or exit
    /// handler: a function value itself, or an integer an
    /// earlier cast exposed a function value at (see the
    /// `exposed_fn_ptrs` field). `None` for anything else.
    pub(crate) fn exposed_callable(&self, value: &Value) -> Option<Value> {
        let bits: u64 = match value {
            Value::FnPtr(_) => return Some(value.clone()),
            Value::Int(bits) => *bits as u64,
            Value::Uint(bits) => *bits as u64,
            _ => return None,
        };
        self.exposed_fn_ptrs.borrow().get(&bits).cloned()
    }

    /// Record an `atexit` registration: the handler
    /// callable, in registration order.
    pub(crate) fn register_exit_handler(&self, handler: Value) {
        self.exit_handlers.borrow_mut().push(handler);
    }

    /// Run the registered `atexit` handlers in reverse
    /// registration order (the platform's order), as the
    /// entry's exit runs them natively. Registrations a
    /// handler itself makes run as well, as the platform
    /// runs them; the round bound keeps a pathological
    /// registration loop from spinning forever. A
    /// registration whose handler body the program does not
    /// hold is skipped, as for TLS destructors: the
    /// registration materialised the handler as a function
    /// constant, which the driver's demand pass collects,
    /// so a missing body means the handler belongs to a
    /// part of std the model never collected.
    pub(crate) fn run_exit_handlers(
        &mut self,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<()> {
        for _ in 0..1024 {
            let batch: Vec<Value> =
                std::mem::take(&mut *self.exit_handlers.borrow_mut());
            if batch.is_empty() {
                return Ok(());
            }
            for handler in batch.into_iter().rev() {
                if let Value::FnPtr(name) = &handler {
                    if self.program.get(name).is_none() {
                        continue;
                    }
                }
                self.call_callable_to_completion(frames, &handler, Vec::new())?;
            }
        }
        Ok(())
    }

    /// Deliver signal `signum` to the guest, synchronously:
    /// run the installed handler to completion, apply the
    /// default disposition, or do nothing for an ignored
    /// signal. This is `raise`'s semantics, and the only
    /// delivery the model performs — no host signal can
    /// arrive.
    pub(crate) fn deliver_signal(
        &mut self,
        frames: &mut Vec<Frame>,
        signum: i64,
    ) -> InterpResult<()> {
        let action = self.signal_actions.borrow().get(&signum).cloned();
        let Some(action) = action else {
            return self.deliver_default(signum);
        };
        if let Value::FnPtr(_) = action.handler {
            let handler = action.handler.clone();
            return self.call_signal_handler(frames, &handler, &action, signum);
        }
        let bits: u128 = match &action.handler {
            Value::Int(bits) => *bits as u128,
            Value::Uint(bits) => *bits,
            other => {
                return Err(InterpError::type_mismatch(
                    "signal handler",
                    "a function pointer or a SIG_DFL / SIG_IGN sentinel",
                    other.kind_name(),
                ));
            }
        };
        match bits {
            // SIG_DFL: the default disposition applies.
            0 => self.deliver_default(signum),
            // SIG_IGN: delivery is a no-op.
            1 => Ok(()),
            addr => match self.exposed_callable(&action.handler) {
                Some(callable) => {
                    self.call_signal_handler(frames, &callable, &action, signum)
                }
                None => Err(InterpError::UnsupportedTerminator {
                    kind: format!(
                        "runtime support: the signal handler at address \
                         {addr:#x} was not exposed from a function value"
                    ),
                }),
            },
        }
    }

    /// Run an installed signal handler to completion. With
    /// `SA_SIGINFO` the handler takes the signal number, a
    /// `siginfo_t` pointer and a context pointer: the
    /// siginfo block is a zeroed structure carrying the
    /// fields a `raise` delivery fills (the signal number,
    /// and the `SI_TKILL` code `raise` reports), and the
    /// context pointer names a zeroed block of the same
    /// size — the model keeps no register state a handler
    /// could read from it. Without the flag the handler
    /// takes the signal number alone.
    fn call_signal_handler(
        &mut self,
        frames: &mut Vec<Frame>,
        handler: &Value,
        action: &SignalAction,
        signum: i64,
    ) -> InterpResult<()> {
        if action.flags & SA_SIGINFO != 0 {
            let mut info = vec![0u8; 128];
            info[0..4].copy_from_slice(&(signum as i32).to_le_bytes());
            info[8..12].copy_from_slice(&(-6i32).to_le_bytes());
            let info_id = self.alloc_insert(Value::Bytes(info));
            let info_ptr = Value::Ref(RefTarget::Alloc {
                id: info_id,
                projection: Vec::new(),
            });
            let context_id = self.alloc_insert(Value::Bytes(vec![0u8; 128]));
            let context_ptr = Value::Ref(RefTarget::Alloc {
                id: context_id,
                projection: Vec::new(),
            });
            self.call_callable_to_completion(
                frames,
                handler,
                vec![Value::Int(signum as i128), info_ptr, context_ptr],
            )?;
        } else {
            self.call_callable_to_completion(
                frames,
                handler,
                vec![Value::Int(signum as i128)],
            )?;
        }
        Ok(())
    }

    /// Apply the default disposition of a signal with no
    /// installed handler (or an explicit `SIG_DFL`).
    fn deliver_default(&mut self, signum: i64) -> InterpResult<()> {
        match default_disposition(signum) {
            DefaultDisposition::Ignore => Ok(()),
            DefaultDisposition::Stop => Err(InterpError::UnsupportedTerminator {
                kind: format!(
                    "runtime support: delivery of signal {signum} stops \
                     the guest, which the model does not perform"
                ),
            }),
            DefaultDisposition::Terminate => Err(InterpError::Abort {
                message: format!("guest terminated by signal {signum}"),
            }),
        }
    }
}
