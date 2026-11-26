//! Range and wrapper value helpers for the range builtins.

use std::cmp::Ordering;


use crate::error::{InterpError, InterpResult};
use crate::value::Value;


/// Peel one-field wrapper aggregates off a value: reading through
/// a raw pointer into a transparent wrapper (`UnsafeCell<T>`, the
/// `AtomicUsize`-family structs, any one-field struct) yields the
/// wrapper's aggregate, whose only field is the scalar the pointer
/// denotes. Anything else is returned unchanged.
pub(crate) fn unwrap_wrapper(value: &Value) -> &Value {
    let mut current = value;
    while let Value::Enum {
        variant: 0,
        fields,
    } = current
    {
        if fields.len() != 1 {
            break;
        }
        current = &fields[0];
    }
    current
}

/// Whether `value` already carries the wrapper-chain form of
/// `current`: the same number of one-field `Enum` layers over a
/// core of the same kind. A store into a wrapper-shaped slot
/// uses it to tell a whole element in the slot's own
/// representation (stored as it stands) from the wrapped core a
/// pointer into the wrapper denotes (rebuilt by
/// [`rewrap_like`]).
pub(crate) fn same_wrapper_form(current: &Value, value: &Value) -> bool {
    fn form(v: &Value) -> (usize, &Value) {
        let mut depth = 0;
        let mut core = v;
        while let Value::Enum {
            variant: 0,
            fields,
        } = core
        {
            if fields.len() != 1 {
                break;
            }
            depth += 1;
            core = &fields[0];
        }
        (depth, core)
    }
    let (d1, c1) = form(current);
    let (d2, c2) = form(value);
    d1 == d2 && c1.kind_name() == c2.kind_name()
}

/// Rebuild the wrapper chain `current` carries around a new core
/// value: the mirror of [`unwrap_wrapper`] for writes, so a cell
/// place keeps its aggregate shape with the new scalar inside.
pub(crate) fn rewrap_like(current: &Value, core: Value) -> Value {
    match current {
        Value::Enum {
            variant: 0,
            fields,
        } if fields.len() == 1 => Value::Enum {
            variant: 0,
            fields: vec![rewrap_like(&fields[0], core)],
        },
        _ => core,
    }
}

pub(super) fn compare_values(a: &Value, b: &Value) -> InterpResult<Ordering> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(x.cmp(y)),
        (Value::Uint(x), Value::Uint(y)) => Ok(x.cmp(y)),
        _ => Err(InterpError::type_mismatch(
            "iterator bounds",
            "matching integer bounds",
            format!("{} and {}", a.kind_name(), b.kind_name()),
        )),
    }
}

fn increment(value: &Value) -> InterpResult<Value> {
    match value {
        Value::Int(v) => Ok(Value::Int(v + 1)),
        Value::Uint(v) => Ok(Value::Uint(v + 1)),
        other => Err(InterpError::type_mismatch(
            "iterator bound increment",
            "integer",
            other.kind_name(),
        )),
    }
}

fn decrement(value: &Value) -> InterpResult<Value> {
    match value {
        Value::Int(v) => Ok(Value::Int(v - 1)),
        Value::Uint(v) => Ok(Value::Uint(v - 1)),
        other => Err(InterpError::type_mismatch(
            "iterator bound decrement",
            "integer",
            other.kind_name(),
        )),
    }
}

/// One forward step of a range bound in the element type's
/// own domain: for `char`, the step skips the surrogate gap
/// [0xD800, 0xE000) — std's `Step` for `char` never lands in
/// it — and every other element type steps by one.
pub(super) fn range_increment(value: &Value, elem: &str) -> InterpResult<Value> {
    if elem == "char" {
        return Ok(match increment(value)? {
            Value::Uint(0xD800) => Value::Uint(0xE000),
            Value::Int(0xD800) => Value::Int(0xE000),
            stepped => stepped,
        });
    }
    increment(value)
}

/// One backward step of a range bound: the mirror of
/// [`range_increment`].
pub(super) fn range_decrement(value: &Value, elem: &str) -> InterpResult<Value> {
    if elem == "char" {
        return Ok(match decrement(value)? {
            Value::Uint(0xDFFF) => Value::Uint(0xD7FF),
            Value::Int(0xDFFF) => Value::Int(0xD7FF),
            stepped => stepped,
        });
    }
    decrement(value)
}

/// The code point `n` valid chars after `start` in a `char`
/// range: the raw advance plus the surrogate gap's width
/// when the advance reaches the gap from below it. `start`
/// is a valid char, so it never lies in the gap itself.
pub(super) fn char_advance(start: u128, n: u128) -> u128 {
    let advanced = start + n;
    if start < 0xD800 && advanced >= 0xD800 {
        advanced + 0x800
    } else {
        advanced
    }
}

/// Whether a `From` conversion between a legacy `std::ops::`
/// range type and its `std::range::` counterpart (in either
/// direction) exists in core for this pair of type tags with
/// the same generic argument: `Range`, `RangeFrom` and
/// `RangeToInclusive` convert both ways by copying their
/// fields in order. (`RangeInclusive` is absent on purpose:
/// its legacy-to-new conversion reads the source's iteration
/// state and can panic, so it is not a field copy.)
pub(super) fn legacy_new_range_conversion(source: &str, target: &str) -> bool {
    fn kind_arg<'a>(ty: &'a str, module: &str, kind: &str) -> Option<&'a str> {
        ty.strip_prefix(&format!("{module}::{kind}<"))
            .and_then(|rest| rest.strip_suffix('>'))
    }
    for kind in ["Range", "RangeFrom", "RangeToInclusive"] {
        for (a, b) in [(source, target), (target, source)] {
            if let (Some(x), Some(y)) = (
                kind_arg(a, "std::ops", kind),
                kind_arg(b, "std::range", kind),
            ) {
                return x == y;
            }
        }
    }
    false
}

/// The element type tag of a `Range<elem>` / `RangeInclusive<elem>` /
/// `RangeFrom<elem>` / `RangeFromIter<elem>` type tag.
pub(super) fn range_elem_ty(self_ty: &str) -> &str {
    self_ty
        .strip_prefix("std::ops::RangeInclusive<")
        .or_else(|| self_ty.strip_prefix("std::range::RangeFromIter<"))
        .or_else(|| self_ty.strip_prefix("std::ops::RangeFrom<"))
        .or_else(|| self_ty.strip_prefix("std::range::RangeFrom<"))
        .or_else(|| self_ty.strip_prefix("std::ops::Range<"))
        .and_then(|rest| rest.strip_suffix('>'))
        .unwrap_or("")
}

/// The width in bits and signedness of an integer element type
/// whose `Step` arithmetic the range machines serve.
fn range_int_width(elem: &str) -> Option<(u32, bool)> {
    Some(match elem {
        "u8" => (8, false),
        "u16" => (16, false),
        "u32" => (32, false),
        "u64" => (64, false),
        "u128" => (128, false),
        "usize" => (64, false),
        "i8" => (8, true),
        "i16" => (16, true),
        "i32" => (32, true),
        "i64" => (64, true),
        "i128" => (128, true),
        "isize" => (64, true),
        _ => return None,
    })
}

/// Whether the range machines step this element type at all:
/// the integers and `char`. Any other `Step` type is a user
/// implementation whose forward step is code, not arithmetic.
pub(super) fn range_elem_step_served(elem: &str) -> bool {
    elem == "char" || range_int_width(elem).is_some()
}

/// The bit pattern of an integer bound in its element width
/// (see `range_count` on the signed materialisation of wide
/// unsigned constants: the pattern is what denotes the value,
/// whichever of the two integer forms carries it).
fn range_bits(value: &Value, width: u32) -> Option<u128> {
    let mask = if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    match value {
        Value::Uint(v) => Some(v & mask),
        Value::Int(v) => Some((*v as u128) & mask),
        _ => None,
    }
}

/// The value of a stepped bit pattern, in the representation
/// the input carried: an input in signed form (a wide unsigned
/// constant's materialisation, or a signed element's value)
/// reads the pattern back sign-extended from the width.
fn range_unbits(like: &Value, bits: u128, width: u32) -> Value {
    match like {
        Value::Int(_) => {
            let v = if width >= 128 {
                bits as i128
            } else if bits & (1u128 << (width - 1)) != 0 {
                bits as i128 - (1i128 << width)
            } else {
                bits as i128
            };
            Value::Int(v)
        }
        _ => Value::Uint(bits),
    }
}

/// `Step::forward_checked(value, 1)` for a range bound: the
/// successor, or `None` where the step leaves the element
/// type's domain (an integer at its maximum, `char` at
/// `char::MAX`). An element type the range machines do not
/// step also yields `None`; callers that must tell the two
/// apart consult [`range_elem_step_served`] first.
pub(super) fn range_step_forward_checked(
    value: &Value,
    elem: &str,
) -> InterpResult<Option<Value>> {
    if elem == "char" {
        let code = match value {
            Value::Uint(v) => *v,
            Value::Int(v) => *v as u128,
            other => {
                return Err(InterpError::type_mismatch(
                    "range bound step",
                    "integer",
                    other.kind_name(),
                ))
            }
        };
        if code >= 0x10FFFF {
            return Ok(None);
        }
        return Ok(Some(range_increment(value, elem)?));
    }
    let Some((width, signed)) = range_int_width(elem) else {
        return Ok(None);
    };
    let Some(bits) = range_bits(value, width) else {
        return Err(InterpError::type_mismatch(
            "range bound step",
            "integer",
            value.kind_name(),
        ));
    };
    let max = if signed {
        (1u128 << (width - 1)) - 1
    } else if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    if bits == max {
        return Ok(None);
    }
    let mask = if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    Ok(Some(range_unbits(
        value,
        bits.wrapping_add(1) & mask,
        width,
    )))
}

/// The maximum value of an unsigned integer element type,
/// as a count: the clamp core's specialised `StepBy` step
/// applies to a step wider than the element type.
pub(super) fn range_elem_max(elem: &str) -> Option<u128> {
    let (width, signed) = range_int_width(elem)?;
    if signed {
        return None;
    }
    Some(if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    })
}

/// `value + delta` wrapping in the element type's width, in
/// the representation `value` carries: the arithmetic of
/// core's specialised `StepBy` step over an integer range,
/// whose stored bound is stepped past its last yield without
/// ever being read again. `None` for an element type the
/// range machines do not step.
pub(super) fn range_wrapping_add(value: &Value, elem: &str, delta: u128) -> Option<Value> {
    let (width, _) = range_int_width(elem)?;
    let bits = range_bits(value, width)?;
    let mask = if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    Some(range_unbits(
        value,
        bits.wrapping_add(delta) & mask,
        width,
    ))
}

/// `Step::forward(value, 1)` for a range bound, as core's
/// integer and `char` implementations perform it: the checked
/// step's result when it exists; at the maximum, the addition
/// overflow panic when the session runs with overflow checks
/// (the `#[rustc_inherit_overflow_checks]` on the step makes
/// the caller's setting decide) and the wrap to the type's
/// minimum when it does not. `char` has no wrapping form: its
/// `Step::forward` is the trait default, which panics at the
/// maximum in every build.
pub(super) fn range_step_forward(
    value: &Value,
    elem: &str,
    overflow_checks: bool,
) -> InterpResult<Value> {
    if let Some(next) = range_step_forward_checked(value, elem)? {
        return Ok(next);
    }
    if elem == "char" {
        return Err(InterpError::panic_str_payload(
            "overflow in `Step::forward`",
        ));
    }
    let Some((width, signed)) = range_int_width(elem) else {
        return Err(InterpError::type_mismatch(
            "range bound step",
            "an integer element type",
            elem,
        ));
    };
    if overflow_checks {
        return Err(InterpError::panic_str_payload(
            "attempt to add with overflow",
        ));
    }
    let min = if signed { 1u128 << (width - 1) } else { 0 };
    Ok(range_unbits(value, min, width))
}

/// The element count of a range, as `size_hint` reports it:
/// `end - start` (plus one for an inclusive range), zero when
/// the range is empty, with the surrogate gap [0xD800, 0xE000)
/// removed for `char` ranges (std's char range iterators skip
/// the gap, and their `size_hint` accounts for it). `None` when
/// the count exceeds `usize::MAX` — std then reports
/// `(usize::MAX, None)`.
///
/// The bounds are read in the element type's own arithmetic:
/// for an unsigned element type a bound is its bit pattern (a
/// constant with the high bit set, `u64::MAX`, materialises as
/// a signed value in this model — see `uint_arg_bits` — and an
/// unassigned temporary's fields keep that form), for a signed
/// element type its value.
pub(super) fn range_count(
    start: &Value,
    end: &Value,
    inclusive: bool,
    elem: &str,
) -> InterpResult<Option<u128>> {
    fn bits(value: &Value) -> InterpResult<u128> {
        match value {
            Value::Uint(v) => Ok(*v),
            Value::Int(v) => Ok(*v as u128),
            other => Err(InterpError::type_mismatch(
                "range bounds",
                "integer bounds",
                other.kind_name(),
            )),
        }
    }
    fn signed(value: &Value) -> InterpResult<i128> {
        match value {
            Value::Int(v) => Ok(*v),
            Value::Uint(v) => i128::try_from(*v).map_err(|_| {
                InterpError::type_mismatch(
                    "range bounds",
                    "signed integer bounds",
                    "an out-of-range unsigned integer",
                )
            }),
            other => Err(InterpError::type_mismatch(
                "range bounds",
                "integer bounds",
                other.kind_name(),
            )),
        }
    }
    let span: Option<u128> = match elem {
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "char" => {
            let (a, b) = (bits(start)?, bits(end)?);
            if b < a {
                Some(0)
            } else {
                (b - a).checked_add(u128::from(inclusive))
            }
        }
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => {
            let (a, b) = (signed(start)?, signed(end)?);
            if b < a {
                Some(0)
            } else {
                b.checked_sub(a)
                    .and_then(|d| d.checked_add(i128::from(inclusive)))
                    .map(|v| v as u128)
            }
        }
        _ => {
            return Err(InterpError::type_mismatch(
                "range bounds",
                "an integer element type",
                elem,
            ))
        }
    };
    let Some(mut span) = span else {
        return Ok(None);
    };
    if elem == "char" {
        let (a, b) = (bits(start)?, bits(end)?);
        let hi = if inclusive { b + 1 } else { b };
        let g_lo = a.max(0xD800);
        let g_hi = hi.min(0xE000);
        if g_hi > g_lo {
            span -= g_hi - g_lo;
        }
    }
    if span > u128::from(u64::MAX) {
        Ok(None)
    } else {
        Ok(Some(span))
    }
}


/// A `u64` argument read as its bit pattern: a constant with
/// the high bit set (e.g. `u64::MAX`) materialises as a signed
/// value in this model, and a time constructor's argument is the
/// unsigned value with that pattern.
pub(super) fn uint_arg_bits(value: &Value, name: &str) -> InterpResult<u64> {
    match value {
        Value::Uint(n) => Ok(*n as u64),
        Value::Int(n) => Ok(*n as u64),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "an unsigned integer",
            other.kind_name(),
        )),
    }
}

/// A `u64`/`u32` argument of a time constructor.
fn uint_arg(value: &Value, name: &str) -> InterpResult<u64> {
    match value {
        Value::Uint(n) => Ok(*n as u64),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "an unsigned integer",
            other.kind_name(),
        )),
    }
}
