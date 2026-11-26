//! Time values: Instant/SystemTime/Duration construction and arithmetic.



use crate::error::{InterpError, InterpResult};
use crate::interp::Interpreter;
use crate::value::Value;

use super::*;

/// A point in time as the model stores it: seconds and
/// nanoseconds (0 <= nsec < 1e9) since the clock's epoch (the
/// monotonic clock's unspecified epoch for `Instant`, the Unix
/// epoch for `SystemTime`).
type TimePair = (i128, i128);

/// The host monotonic clock as a [`TimePair`]: the elapsed time
/// since this process's first reading, so the values are
/// non-negative and strictly follow the host clock's order (the
/// epoch itself is unobservable for `Instant`, as natively).
pub(super) fn monotonic_now() -> TimePair {
    static BASE: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let base = BASE.get_or_init(std::time::Instant::now);
    let d = std::time::Instant::now().duration_since(*base);
    (d.as_secs() as i128, d.subsec_nanos() as i128)
}

/// The host wall clock as a [`TimePair`] (seconds since the Unix
/// epoch; negative for earlier times, as `SystemTime` allows).
pub(super) fn wall_now() -> TimePair {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => (d.as_secs() as i128, d.subsec_nanos() as i128),
        Err(e) => {
            let d = e.duration();
            (-(d.as_secs() as i128), -(d.subsec_nanos() as i128))
        }
    }
}

/// The stored value of an `Instant` / `SystemTime`: the nesting
/// the constant decoder produces for `UNIX_EPOCH`
/// (`{ t: { t: { sec, { nsec } } } }`), so modelled values and
/// decoded constants have the same shape and compare field-wise.
pub(super) fn instant_value(pair: TimePair) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Int(pair.0),
                    Value::Enum {
                        variant: 0,
                        fields: vec![Value::Int(pair.1)],
                    },
                ],
            }],
        }],
    }
}

/// Read a [`TimePair`] out of a stored time value, unwrapping
/// the single-field wrapper layers (and the nanoseconds
/// wrapper), so both modelled values and constants decoded from
/// allocations are accepted.
pub(super) fn time_pair(value: &Value) -> Option<TimePair> {
    fn as_int(value: &Value) -> Option<i128> {
        match value {
            Value::Int(v) => Some(*v),
            Value::Uint(v) => Some(*v as i128),
            Value::Enum { fields, .. } | Value::Tuple(fields) if fields.len() == 1 => {
                as_int(&fields[0])
            }
            _ => None,
        }
    }
    match value {
        Value::Enum { fields, .. } | Value::Tuple(fields) => {
            if fields.len() == 1 {
                return time_pair(&fields[0]);
            }
            if fields.len() == 2 {
                return Some((as_int(&fields[0])?, as_int(&fields[1])?));
            }
            None
        }
        _ => None,
    }
}

/// Read a `Duration` value as a (secs, nanos) pair in the same
/// form as a [`TimePair`]. The fields are unsigned (`u64` secs,
/// `u32` nanos, the nanos possibly behind its wrapper struct in
/// a decoded constant), so an integer field is read as the bit
/// pattern of the unsigned value.
pub(super) fn duration_pair(value: &Value) -> Option<TimePair> {
    fn as_unsigned(value: &Value) -> Option<i128> {
        match value {
            Value::Uint(v) => Some(*v as i128),
            Value::Int(v) => Some(*v as u64 as i128),
            Value::Enum { fields, .. } | Value::Tuple(fields) if fields.len() == 1 => {
                as_unsigned(&fields[0])
            }
            _ => None,
        }
    }
    match value {
        Value::Enum { fields, .. } | Value::Tuple(fields) if fields.len() == 2 => {
            Some((as_unsigned(&fields[0])?, as_unsigned(&fields[1])?))
        }
        _ => None,
    }
}

/// `base + dur` on time pairs, normalized; `None` on overflow of
/// the seconds range. For `Instant` (system = false) the result
/// must stay in the monotonic clock's representable range, which
/// this model takes as the i64 seconds range; `SystemTime` and
/// `Duration` arithmetic shares the same bound here.
pub(super) fn checked_time_add(base: TimePair, dur: TimePair, _system: bool) -> Option<TimePair> {
    let mut secs = base.0.checked_add(dur.0)?;
    let mut nanos = base.1 + dur.1;
    if nanos >= 1_000_000_000 {
        nanos -= 1_000_000_000;
        secs = secs.checked_add(1)?;
    }
    if secs > i64::MAX as i128 {
        return None;
    }
    Some((secs, nanos))
}

/// `base - dur` on time pairs, normalized; `None` when the
/// result would be negative (for a duration-since computation)
/// or underflow the seconds range.
pub(super) fn checked_time_sub(base: TimePair, dur: TimePair, _system: bool) -> Option<TimePair> {
    let mut secs = base.0.checked_sub(dur.0)?;
    let mut nanos = base.1 - dur.1;
    if nanos < 0 {
        nanos += 1_000_000_000;
        secs = secs.checked_sub(1)?;
    }
    if secs < 0 {
        return None;
    }
    Some((secs, nanos))
}

impl<'a> Interpreter<'a> {
    /// `Add::add` / `Sub::sub` for the modelled time types,
    /// completed by the interpreter when the demanded instance
    /// has no collected body (the `std::time` instances are not
    /// available from the compiler). Instances with bodies — a
    /// guest impl, a generic instantiation's shim — always take
    /// the normal call path and never reach here, so the
    /// collected-body priority is untouched. Returns the result
    /// value, or an error for a Self type outside the model.
    pub(crate) fn time_add_sub(
        &self,
        name: &str,
        self_ty: &str,
        sub: bool,
        vals: &[Value],
    ) -> InterpResult<Value> {
        if vals.len() != 2 {
            return Err(InterpError::type_mismatch(
                name,
                "2 argument(s)",
                format!("{} argument(s)", vals.len()),
            ));
        }
        match self_ty {
                "std::time::Instant" | "std::time::SystemTime" => {
                    let system = self_ty == "std::time::SystemTime";
                    let base = time_pair(&vals[0]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            name,
                            "a time value",
                            vals[0].kind_name(),
                        )
                    })?;
                    let dur = duration_pair(&vals[1]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            name,
                            "a Duration value",
                            vals[1].kind_name(),
                        )
                    })?;
                    let pair = if sub {
                        checked_time_sub(base, dur, system)
                    } else {
                        checked_time_add(base, dur, system)
                    };
                    match pair {
                        Some(pair) => Ok(instant_value(pair)),
                        None => Err(InterpError::panic_str_payload(
                            match (system, sub) {
                                (false, false) => {
                                    "overflow when adding duration to instant"
                                }
                                (false, true) => {
                                    "overflow when subtracting duration from instant"
                                }
                                (true, false) => {
                                    "overflow when adding duration to system time"
                                }
                                (true, true) => {
                                    "overflow when subtracting duration from system time"
                                }
                            },
                        )),
                    }
                }
                "std::time::Duration" => {
                    let a = duration_pair(&vals[0]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            name,
                            "a Duration value",
                            vals[0].kind_name(),
                        )
                    })?;
                    let b = duration_pair(&vals[1]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            name,
                            "a Duration value",
                            vals[1].kind_name(),
                        )
                    })?;
                    if sub {
                        match checked_time_sub(a, b, true) {
                            Some(pair) => {
                                self.duration_value(pair.0 as u64, pair.1 as u64)
                            }
                            None => Err(InterpError::panic_str_payload(
                                "overflow when subtracting duration",
                            )),
                        }
                    } else {
                        match checked_time_add(a, b, true) {
                            Some(pair) => {
                                self.duration_value(pair.0 as u64, pair.1 as u64)
                            }
                            None => Err(InterpError::panic_str_payload(
                                "overflow when adding duration",
                            )),
                        }
                    }
                }
                _ => Err(unsupported(format!(
                    "{} for {self_ty} is not runtime-supported",
                    if sub { "Sub::sub" } else { "Add::add" }
                ))),
            }
    }

    pub(super) fn duration_value(&self, secs: u64, nanos: u64) -> InterpResult<Value> {
        let (secs, nanos) = if nanos >= 1_000_000_000 {
            let carry = nanos / 1_000_000_000;
            let secs = secs
                .checked_add(carry)
                .ok_or_else(|| InterpError::panic("overflow in Duration::new"))?;
            (secs, nanos % 1_000_000_000)
        } else {
            (secs, nanos)
        };
        Ok(Value::Enum {
            variant: 0,
            fields: vec![Value::Uint(u128::from(secs)), Value::Uint(u128::from(nanos))],
        })
    }
}

/// Registration fragment of the builtin name registry: the `Duration` constructors.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn time_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::time::Duration::new" => Builtin::DurationNew,
        "std::time::Duration::from_secs" => Builtin::DurationFromSecs,
        "std::time::Duration::from_millis" => Builtin::DurationFromMillis,
        "std::time::Duration::from_micros" => Builtin::DurationFromMicros,
        "std::time::Duration::from_nanos" => Builtin::DurationFromNanos,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the clock readings.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn time_now_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::time::Instant::now" => Builtin::InstantNow,
        "std::time::SystemTime::now" => Builtin::SystemTimeNow,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the clock arithmetic surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn time_arith_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::time::Instant::checked_add" => Builtin::TimeCheckedAdd(false),
        "std::time::SystemTime::checked_add" => Builtin::TimeCheckedAdd(true),
        "std::time::Instant::checked_sub" => Builtin::TimeCheckedSub(false),
        "std::time::SystemTime::checked_sub" => Builtin::TimeCheckedSub(true),
        "std::time::Instant::duration_since" => Builtin::InstantDurationSince,
        "std::time::Instant::checked_duration_since" => {
            Builtin::InstantCheckedDurationSince
        }
        "std::time::Instant::saturating_duration_since" => {
            Builtin::InstantSaturatingDurationSince
        }
        "std::time::Instant::elapsed" => Builtin::InstantElapsed,
        "std::time::SystemTime::duration_since" => Builtin::SystemTimeDurationSince,
        "std::time::SystemTime::elapsed" => Builtin::SystemTimeElapsed,
        _ => return None,
    })
}
