//! The `sigaction` / `signal` / `raise` / `atexit` builtin bodies:
//! the argument decoding around the interpreter's signal
//! model (the state and the delivery live with the
//! interpreter, in `interp`'s signal module).
//!
//! The `sigaction` structure is read through the value
//! model in the field order the host's libc lays it out —
//! `sa_sigaction`, `sa_mask`, `sa_flags`, `sa_restorer` on
//! Linux. Delivery consults the handler field and the
//! flags word only; the mask and the restorer ride along
//! in the recorded structure, which a later call's
//! `oldact` write-back hands back verbatim. A structure
//! that reaches the builtin in byte form (a write path
//! composed it as bytes) is read at the same layout's byte
//! offsets instead.

use crate::error::{InterpError, InterpResult};
use crate::interp::{Frame, Interpreter, SignalAction};
use crate::value::{RefTarget, Value};

use super::*;

/// The byte offset of `sa_flags` in the host's
/// `struct sigaction` (after the 8-byte handler and the
/// 128-byte `sa_mask`).
const SA_FLAGS_OFFSET: usize = 136;

/// The `SA_RESTART` flag word: the one flag the
/// platform's `signal` installs its handlers with.
const SA_RESTART: i64 = 0x1000_0000;

impl<'a> Interpreter<'a> {
    /// `libc::sigaction(signum, act, oldact) -> c_int`:
    /// install the action `act` names for `signum` (a null
    /// `act` only queries), write the previously installed
    /// action through `oldact` when it is non-null, and
    /// answer 0. Signal numbers outside `1..=64`, and the
    /// two uncatchable signals (`SIGKILL`, `SIGSTOP`),
    /// answer -1, as the host call fails for them.
    pub(super) fn exec_libc_sigaction(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let signum = signal_number_arg(name, &vals[0])?;
        if !(1..=64).contains(&signum) || signum == 9 || signum == 19 {
            return Ok(Value::Int(-1));
        }
        let new_action = if is_null_pointer(&vals[1]) {
            None
        } else {
            let target = self.deref_value(frames, &vals[1])?;
            let act = self.read_resolved(frames, &target)?;
            Some(parse_sigaction(name, &act)?)
        };
        let previous = self.signal_install(signum, new_action.clone());
        if !is_null_pointer(&vals[2]) {
            let target = self.deref_value(frames, &vals[2])?;
            let previous_value = match &previous {
                Some(previous) if previous.runtime_init => {
                    // The previous disposition is the one
                    // the runtime initialisation installed:
                    // it has no guest structure of its own,
                    // so the answer takes the shape this
                    // call names — the installed action's
                    // zeroed twin for an install, the
                    // answer buffer's zeroed twin for a
                    // pure query — with the handler leaf
                    // set to the installed handler.
                    let shape = match &new_action {
                        Some(action) => zeroed_twin(&action.act),
                        None => {
                            let current =
                                self.read_resolved(frames, &target)?;
                            zeroed_twin(&current)
                        }
                    };
                    with_handler(shape, &previous.handler)
                }
                Some(previous) => previous.act.clone(),
                // No action was ever installed: the previous
                // disposition is the default, whose structure
                // is all zero (the default handler, an empty
                // mask, no flags, no restorer).
                None => match &new_action {
                    // An install names the structure's shape
                    // in its own action: the default is that
                    // structure zeroed.
                    Some(action) => zeroed_twin(&action.act),
                    // A pure query names no shape of its own;
                    // the buffer the guest hands over for the
                    // answer carries the structure's shape,
                    // and zeroing its current contents leaf by
                    // leaf yields exactly the default
                    // structure.
                    None => {
                        let current = self.read_resolved(frames, &target)?;
                        zeroed_twin(&current)
                    }
                },
            };
            self.write_resolved(frames, &target, previous_value)?;
        }
        Ok(Value::Int(0))
    }

    /// `libc::signal(signum, handler) -> sighandler_t`:
    /// install `handler` as the disposition of `signum`
    /// and answer the handler it replaces — the previously
    /// installed action's handler field, or `SIG_DFL` when
    /// no action was ever installed. The platform's
    /// `signal` is the BSD form: the recorded action
    /// carries the `SA_RESTART` flag, an empty mask and no
    /// restorer, and the handler stays installed across
    /// delivery (delivery in this model never uninstalls,
    /// so the persistence needs no extra state). Signal
    /// numbers outside `1..=64`, and the two uncatchable
    /// signals (`SIGKILL`, `SIGSTOP`), answer `SIG_ERR`,
    /// as the host call fails for them.
    pub(super) fn exec_libc_signal(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let _ = frames;
        let signum = signal_number_arg(name, &vals[0])?;
        if !(1..=64).contains(&signum) || signum == 9 || signum == 19 {
            return Ok(Value::Uint(u64::MAX as u128));
        }
        // A null function pointer names the default
        // disposition: record it in the sentinel-integer
        // form delivery consults.
        let handler = match &vals[1] {
            Value::Ref(RefTarget::Null) => Value::Uint(0),
            other => other.clone(),
        };
        let action = SignalAction {
            handler: handler.clone(),
            flags: SA_RESTART,
            act: signal_action_structure(&handler),
            runtime_init: false,
        };
        let previous = self.signal_install(signum, Some(action));
        Ok(match previous {
            Some(previous) => previous.handler,
            None => Value::Uint(0),
        })
    }

    /// `libc::raise(sig) -> c_int`: deliver the signal to
    /// the guest itself, synchronously (see the
    /// interpreter's `deliver_signal`), and answer 0. A
    /// signal number outside `1..=64` answers -1.
    pub(super) fn exec_libc_raise(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let signum = signal_number_arg(name, &vals[0])?;
        if !(1..=64).contains(&signum) {
            return Ok(Value::Int(-1));
        }
        self.deliver_signal(frames, signum)?;
        Ok(Value::Int(0))
    }

    /// `libc::atexit(func) -> c_int`: record the handler
    /// for the entry's exit (see `run_exit_handlers`) and
    /// answer 0.
    pub(super) fn exec_libc_atexit(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let _ = frames;
        match self.exposed_callable(&vals[0]) {
            Some(callable) => {
                self.register_exit_handler(callable);
                Ok(Value::Int(0))
            }
            None => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a function pointer handler",
                vals[0].kind_name(),
            )),
        }
    }
}

/// A `sigaction` structure value with its handler
/// leaf set to `handler`: the first field of the
/// aggregate form, the first eight bytes of the byte
/// form. The leaf keeps its own scalar kind; a shape
/// with no handler leaf is returned unchanged.
fn with_handler(value: Value, handler: &Value) -> Value {
    let bits: u128 = match handler {
        Value::Int(bits) => *bits as u128,
        Value::Uint(bits) => *bits,
        _ => return value,
    };
    let leaf = |old: &Value| match old {
        Value::Int(_) => Value::Int(bits as i128),
        _ => Value::Uint(bits),
    };
    match value {
        Value::Bytes(mut bytes) => {
            if bytes.len() >= 8 {
                bytes[0..8]
                    .copy_from_slice(&(bits as u64).to_le_bytes());
            }
            Value::Bytes(bytes)
        }
        Value::Tuple(mut fields) => {
            if let Some(first) = fields.first_mut() {
                *first = leaf(first);
            }
            Value::Tuple(fields)
        }
        Value::Enum {
            variant,
            mut fields,
        } => {
            if let Some(first) = fields.first_mut() {
                *first = leaf(first);
            }
            Value::Enum { variant, fields }
        }
        other => other,
    }
}

/// A signal-number argument (or a `c_int` structure field)
/// as an `i64`.
fn signal_number_arg(name: &str, value: &Value) -> InterpResult<i64> {
    match value {
        Value::Int(v) => Ok(*v as i64),
        Value::Uint(v) => Ok(*v as i64),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "an integer signal number",
            other.kind_name(),
        )),
    }
}

/// Whether a pointer argument is the null pointer.
fn is_null_pointer(value: &Value) -> bool {
    matches!(value, Value::Ref(RefTarget::Null))
        || matches!(value, Value::Int(0) | Value::Uint(0))
}

/// Read a `sigaction` structure value into the recorded
/// action: the handler field, the `sa_flags` word, and the
/// structure itself for `oldact` write-back.
fn parse_sigaction(name: &str, act: &Value) -> InterpResult<SignalAction> {
    if let Value::Bytes(bytes) = act {
        // The structure in byte form: the handler is the
        // 8-byte word at offset 0 and `sa_flags` the 4-byte
        // word at its layout offset.
        if bytes.len() < SA_FLAGS_OFFSET + 4 {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a sigaction structure",
                "a shorter byte string",
            ));
        }
        let handler = u64::from_le_bytes(
            bytes[0..8].try_into().expect("eight bytes checked"),
        );
        let flags = i32::from_le_bytes(
            bytes[SA_FLAGS_OFFSET..SA_FLAGS_OFFSET + 4]
                .try_into()
                .expect("four bytes checked"),
        );
        return Ok(SignalAction {
            handler: Value::Uint(handler as u128),
            flags: flags as i64,
            act: act.clone(),
            runtime_init: false,
        });
    }
    let fields = value_elements(act)?;
    if fields.len() < 3 {
        return Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "a sigaction structure (handler, mask, flags, restorer)",
            format!("an aggregate of {} field(s)", fields.len()),
        ));
    }
    let flags = signal_number_arg(name, &fields[2])?;
    Ok(SignalAction {
        handler: fields[0].clone(),
        flags,
        act: act.clone(),
        runtime_init: false,
    })
}

/// The `sigaction` structure value a `signal` install
/// records: the handler, an empty `sigset_t` mask (the
/// sixteen zero words of the set, in their wrapper
/// structure), the `SA_RESTART` flags word `signal`
/// installs with, and no restorer. A later `sigaction`
/// call's `oldact` write-back hands this back verbatim,
/// as it hands back a structure the guest itself
/// installed.
fn signal_action_structure(handler: &Value) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            handler.clone(),
            Value::Enum {
                variant: 0,
                fields: vec![Value::Tuple(vec![Value::Uint(0); 16])],
            },
            Value::Int(SA_RESTART as i128),
            Value::Ref(RefTarget::Null),
        ],
    }
}

/// The all-zero twin of a structure value: every scalar
/// leaf zeroed, the aggregate shape (and an enum's
/// variant) kept. A pointer leaf zeroes to the null
/// pointer.
fn zeroed_twin(value: &Value) -> Value {
    match value {
        Value::Bool(_) => Value::Bool(false),
        Value::Int(_) => Value::Int(0),
        Value::Uint(_) => Value::Uint(0),
        Value::Float(_) => Value::Float(0.0),
        Value::Float128(_) => Value::Float128(0),
        Value::Tuple(items) => {
            Value::Tuple(items.iter().map(zeroed_twin).collect())
        }
        Value::Enum { variant, fields } => Value::Enum {
            variant: *variant,
            fields: fields.iter().map(zeroed_twin).collect(),
        },
        Value::SparseVec { elems, len, cap } => Value::SparseVec {
            elems: elems.iter().map(zeroed_twin).collect(),
            len: *len,
            cap: *cap,
        },
        Value::Bytes(bytes) => Value::Bytes(vec![0; bytes.len()]),
        Value::Str(_) => Value::Str(String::new()),
        Value::Ref(_) | Value::FnPtr(_) | Value::FatRef { .. } => {
            Value::Ref(RefTarget::Null)
        }
        other => other.clone(),
    }
}
