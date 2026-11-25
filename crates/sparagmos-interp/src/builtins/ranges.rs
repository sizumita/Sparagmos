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

/// The element type tag of a `Range<elem>` / `RangeInclusive<elem>`
/// type tag.
pub(super) fn range_elem_ty(self_ty: &str) -> &str {
    self_ty
        .strip_prefix("std::ops::RangeInclusive<")
        .or_else(|| self_ty.strip_prefix("std::ops::Range<"))
        .and_then(|rest| rest.strip_suffix('>'))
        .unwrap_or("")
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
