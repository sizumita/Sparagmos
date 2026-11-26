//! Scalar arithmetic: binary/unary operations and float helpers.

use super::*;
use crate::error::{InterpError, InterpResult};
use crate::f128;
use crate::value::{RefProj, RefTarget, Value};

// ----- arithmetic helpers ----------------------------------------------------

pub(crate) fn wrap_signed(v: i128, bits: u32) -> i128 {
    if bits >= 128 {
        return v;
    }
    let shift = 128 - bits;
    (v << shift) >> shift
}

/// The mask of the low `bits` bits (all bits for 128).
pub(crate) fn bit_mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

pub(crate) fn wrap_unsigned(v: u128, bits: u32) -> u128 {
    if bits >= 128 {
        v
    } else {
        v & ((1u128 << bits) - 1)
    }
}

/// The value `v` read under a binary operation's declared
/// operand type: an integer stored under the other kind tag is
/// reinterpreted at the declared width, exactly as an assignment
/// to a place of that type reinterprets it (see
/// `normalize_for_ty`). A checked operation's destination is the
/// `(T, bool)` tuple; the operand type lives in its first
/// element.
fn reinterpret_int(v: &Value, dest_ty: Option<&Ty>) -> Value {
    let operand_ty = match dest_ty {
        Some(Ty::Tuple(ts)) => ts.first(),
        other => other,
    };
    match (v, operand_ty) {
        (Value::Uint(u), Some(Ty::Int(it))) => Value::Int(wrap_signed(*u as i128, it.bits())),
        (Value::Int(i), Some(Ty::Uint(ut))) => Value::Uint((*i as u128) & ut.max()),
        _ => v.clone(),
    }
}

fn signed_bits(dest_ty: Option<&Ty>) -> u32 {
    match dest_ty {
        Some(Ty::Int(it)) => it.bits(),
        // A checked operation's destination is the `(T, bool)` tuple;
        // the operand width lives in its first element.
        Some(Ty::Tuple(ts)) => signed_bits(ts.first()),
        _ => 128,
    }
}

fn unsigned_bits(dest_ty: Option<&Ty>) -> u32 {
    match dest_ty {
        Some(Ty::Uint(ut)) => ut.bits(),
        Some(Ty::Tuple(ts)) => unsigned_bits(ts.first()),
        _ => 128,
    }
}

/// Evaluate a binary operation.
/// Returns `(result, overflowed)`. Overflow is reported only through
/// the checked form's flag: MIR's plain `BinaryOp` wraps on overflow
/// (the overflow panic of a checked build is the following `Assert`
/// terminator's business, driven by that flag), so the plain form
/// yields the wrapped result and never panics for overflow. Division
/// by zero panics in both forms — the operation itself requires it,
/// independent of overflow checks. Comparison and bitwise-on-bool
/// operations never overflow.
/// The `core::cmp::Ordering` value of a three-way comparison:
/// the enum in declaration order (Less, Equal, Greater), the
/// shape `Builtin::OrdCmp`'s `ordering_value` also produces.
fn ordering_of(ord: std::cmp::Ordering) -> Value {
    Value::Enum {
        variant: match ord {
            std::cmp::Ordering::Less => 0,
            std::cmp::Ordering::Equal => 1,
            std::cmp::Ordering::Greater => 2,
        },
        fields: Vec::new(),
    }
}

/// The position order of two pointers into the same storage,
/// or `None` when the two targets do not denote positions of
/// one storage in a comparable way. The targets must share a
/// storage root (the same place, static, allocation, or equal
/// constant base value) and a common projection prefix; the
/// first step on which they differ must then be positional on
/// both sides -- an element index or sub-slice offset (element
/// units), or a byte displacement (byte units) -- which decides
/// the order, as one element's storage wholly precedes the
/// next's whatever the remaining steps denote. A side whose
/// projection ends at the common prefix denotes the shared
/// position itself, position zero of the sequence the other
/// side steps into. Steps that are not positional (a field, a
/// downcast) and mixed units leave the order to the caller.
fn structural_ptr_order(lhs: &Value, rhs: &Value) -> Option<std::cmp::Ordering> {
    fn target_of(v: &Value) -> Option<&RefTarget> {
        match v {
            Value::Ref(t) => Some(t),
            Value::FatRef { base, .. } => match &**base {
                Value::Ref(t) => Some(t),
                _ => None,
            },
            _ => None,
        }
    }
    let (pa, pb): (&[RefProj], &[RefProj]) = match (target_of(lhs)?, target_of(rhs)?) {
        (
            RefTarget::Place {
                frame: fa,
                local: la,
                projection: pa,
            },
            RefTarget::Place {
                frame: fb,
                local: lb,
                projection: pb,
            },
        ) if fa == fb && la == lb => (pa, pb),
        (
            RefTarget::Static {
                name: na,
                projection: pa,
            },
            RefTarget::Static {
                name: nb,
                projection: pb,
            },
        ) if na == nb => (pa, pb),
        (
            RefTarget::Alloc {
                id: ia,
                projection: pa,
            },
            RefTarget::Alloc {
                id: ib,
                projection: pb,
            },
        ) if ia == ib => (pa, pb),
        (RefTarget::Value(va, pa), RefTarget::Value(vb, pb)) if va == vb => (pa, pb),
        _ => return None,
    };
    let mut n = 0;
    while n < pa.len() && n < pb.len() && pa[n] == pb[n] {
        n += 1;
    }
    let (ra, rb) = (&pa[n..], &pb[n..]);
    if ra.is_empty() && rb.is_empty() {
        return Some(std::cmp::Ordering::Equal);
    }
    // The position a remainder denotes within the sequence the
    // common prefix ends at: `None` for an empty remainder (the
    // shared position itself, zero in the other side's units),
    // else the first step's displacement and its unit (element
    // steps and byte steps are not comparable to each other).
    fn position(rem: &[RefProj]) -> Option<Option<(i128, bool)>> {
        match rem.first() {
            None => Some(None),
            Some(RefProj::Index(i)) => {
                i128::try_from(*i).ok().map(|p| Some((p, false)))
            }
            Some(RefProj::Subslice { offset, .. }) => {
                i128::try_from(*offset).ok().map(|p| Some((p, false)))
            }
            Some(RefProj::ByteOffset(d)) => Some(Some((*d, true))),
            _ => None,
        }
    }
    let (pos_a, pos_b) = match (position(ra)?, position(rb)?) {
        (Some(a), Some(b)) => (a, b),
        (Some(a), None) => (a, (0, a.1)),
        (None, Some(b)) => ((0, b.1), b),
        (None, None) => return None,
    };
    if pos_a.1 != pos_b.1 {
        return None;
    }
    Some(pos_a.0.cmp(&pos_b.0))
}

pub(crate) fn eval_binary(
    op: BinOp,
    lhs: &Value,
    rhs: &Value,
    dest_ty: Option<&Ty>,
    checked_form: bool,
    addr_of: &dyn Fn(&Value) -> Option<u128>,
) -> InterpResult<(Value, bool)> {
    use BinOp::*;
    // Boolean operations.
    if let (Value::Bool(a), Value::Bool(b)) = (lhs, rhs) {
        let v = match op {
            BitAnd => Value::Bool(*a && *b),
            BitOr => Value::Bool(*a || *b),
            BitXor => Value::Bool(*a ^ *b),
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Cmp => ordering_of(a.cmp(b)),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "bool",
                ))
            }
        };
        return Ok((v, false));
    }
    // String equality (the only binary operations MIR performs on
    // string data directly; ordering goes through library calls).
    if let (Value::Str(a), Value::Str(b)) = (lhs, rhs) {
        let v = match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "string",
                ))
            }
        };
        return Ok((v, false));
    }
    // Shifts: in MIR the amount of a `Shl` / `Shr` may have ANY
    // integer type, independent of the value's type, so the
    // same-kind pair dispatch below cannot see mixed pairs. Integer
    // shift pairs are evaluated by `eval_shift`, which reads the
    // amount as a `u32` and keeps the value's own type and
    // signedness.
    if matches!(op, Shl | Shr)
        && matches!(lhs, Value::Int(_) | Value::Uint(_))
        && matches!(rhs, Value::Int(_) | Value::Uint(_))
    {
        return eval_shift(op, lhs, rhs, dest_ty, checked_form);
    }

    // An integer operand whose kind tag differs from the
    // operation's declared operand type: reinterpret its bits at
    // the declared width before the strict kind dispatch below. A
    // value stored inside an aggregate keeps the kind it was
    // built with - a constant's bits are stored unsigned - and a
    // direct place operand reads it without an intervening typed
    // place that would reinterpret it, so the declared type is
    // the only exact reading (the same rule `normalize_for_ty`
    // applies at every place write).
    let lhs = &reinterpret_int(lhs, dest_ty);
    let rhs = &reinterpret_int(rhs, dest_ty);
    // Signed arithmetic.
    if let (Value::Int(a), Value::Int(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let bits = signed_bits(dest_ty);
        let (min, max) = if bits >= 128 {
            (i128::MIN, i128::MAX)
        } else {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        };
        let cmp = |f: &dyn Fn() -> bool| Ok((Value::Bool(f()), false));
        let wide: Option<i128> = match op {
            Cmp => return Ok((ordering_of(a.cmp(&b)), false)),
            Eq => return cmp(&|| a == b),
            Ne => return cmp(&|| a != b),
            Lt => return cmp(&|| a < b),
            Le => return cmp(&|| a <= b),
            Gt => return cmp(&|| a > b),
            Ge => return cmp(&|| a >= b),
            Add => a
                .checked_add(b)
                .or(Some(wrap_signed(a.wrapping_add(b), bits))),
            Sub => a
                .checked_sub(b)
                .or(Some(wrap_signed(a.wrapping_sub(b), bits))),
            Mul => a
                .checked_mul(b)
                .or(Some(wrap_signed(a.wrapping_mul(b), bits))),
            Div | Rem => {
                if b == 0 {
                    return Err(InterpError::panic(if op == Div {
                        "attempt to divide by zero"
                    } else {
                        "attempt to calculate the remainder with a divisor of zero"
                    }));
                }
                if a == min && b == -1 {
                    // Overflow: the value wraps in both forms (Div
                    // yields `min`, Rem yields 0); the checked form
                    // also reports it in the flag below.
                    Some(if op == Div { a } else { 0 })
                } else if op == Div {
                    Some(a / b)
                } else {
                    Some(a % b)
                }
            }
            BitAnd => Some(a & b),
            BitOr => Some(a | b),
            BitXor => Some(a ^ b),
            Shl | Shr => unreachable!("shifts are evaluated by `eval_shift`"),
        };
        // Invariant: comparisons returned early above and every other
        // arm of the match yields `Some`, so `wide` is always present.
        let raw = wide.expect("all signed ops produce a value");
        // Determine overflow by recomputing in the declared width: a result
        // is overflowing if the mathematical result leaves [min, max].
        // Division overflow (MIN / -1) is covered because raw == MIN while
        // the true quotient is MAX+1; detect it explicitly.
        let overflowed = match op {
            Add => a.checked_add(b).map_or(true, |r| r < min || r > max),
            Sub => a.checked_sub(b).map_or(true, |r| r < min || r > max),
            Mul => a.checked_mul(b).map_or(true, |r| r < min || r > max),
            Div | Rem => a == min && b == -1,
            Shl => unreachable!("shifts are evaluated by `eval_shift`"),
            _ => false,
        };
        let result = Value::Int(wrap_signed(raw, bits));
        return Ok((result, overflowed));
    }
    // Unsigned arithmetic.
    if let (Value::Uint(a), Value::Uint(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let bits = unsigned_bits(dest_ty);
        let max = if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        let cmp = |f: &dyn Fn() -> bool| Ok((Value::Bool(f()), false));
        match op {
            Cmp => return Ok((ordering_of(a.cmp(&b)), false)),
            Eq => return cmp(&|| a == b),
            Ne => return cmp(&|| a != b),
            Lt => return cmp(&|| a < b),
            Le => return cmp(&|| a <= b),
            Gt => return cmp(&|| a > b),
            Ge => return cmp(&|| a >= b),
            _ => {}
        }
        if (op == Div || op == Rem) && b == 0 {
            return Err(InterpError::panic(if op == Div {
                "attempt to divide by zero"
            } else {
                "attempt to calculate the remainder with a divisor of zero"
            }));
        }
        // Comparisons returned above and shifts are evaluated by
        // `eval_shift` before this dispatch; the operators that
        // can reach this match are exactly the eight covered below,
        // which is what makes the closing `unreachable!` an internal
        // invariant rather than an input-dependent failure.
        let (raw, overflowed) = match op {
            Add => {
                let r = a.wrapping_add(b);
                (r, a.checked_add(b).map_or(true, |v| v > max))
            }
            Sub => (a.wrapping_sub(b), a < b),
            Mul => {
                let r = a.wrapping_mul(b);
                (r, a.checked_mul(b).map_or(true, |v| v > max))
            }
            Div => (a / b, false),
            Rem => (a % b, false),
            BitAnd => (a & b, false),
            BitOr => (a | b, false),
            BitXor => (a ^ b, false),
            _ => unreachable!("comparisons and shifts handled above"),
        };
        let result = Value::Uint(wrap_unsigned(raw, bits));
        return Ok((result, overflowed));
    }
    // `f128` arithmetic: exact software binary128 (see
    // `crate::f128`); never rounded through `f64`.
    if let (Value::Float128(a), Value::Float128(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let v = match op {
            Eq => Value::Bool(f128::eq(a, b)),
            Ne => Value::Bool(!f128::eq(a, b)),
            Lt => Value::Bool(f128::partial_cmp(a, b) == Some(std::cmp::Ordering::Less)),
            Le => Value::Bool(matches!(
                f128::partial_cmp(a, b),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            )),
            Gt => Value::Bool(f128::partial_cmp(a, b) == Some(std::cmp::Ordering::Greater)),
            Ge => Value::Bool(matches!(
                f128::partial_cmp(a, b),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            )),
            Add => Value::Float128(f128::add(a, b)),
            Sub => Value::Float128(f128::sub(a, b)),
            Mul => Value::Float128(f128::mul(a, b)),
            Div => Value::Float128(f128::div(a, b)),
            Rem => Value::Float128(f128::rem(a, b)),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "f128",
                ))
            }
        };
        return Ok((v, false));
    }
    // Floating-point arithmetic, computed in `f64` (see `Value::Float`
    // for why that is exact at `f32` destinations too). Division by
    // zero and NaN follow IEEE 754, as in Rust: they are values, not
    // panics. Float operations never set the overflow flag.
    if let (Value::Float(a), Value::Float(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let round = |r: f64| match float_ty_of(dest_ty) {
            Some(FloatTy::F16) => round_to_f16(r),
            Some(FloatTy::F32) => (r as f32) as f64,
            _ => r,
        };
        let v = match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            Add => Value::Float(round(a + b)),
            Sub => Value::Float(round(a - b)),
            Mul => Value::Float(round(a * b)),
            Div => Value::Float(round(a / b)),
            Rem => Value::Float(round(a % b)),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "float",
                ))
            }
        };
        return Ok((v, false));
    }
    // Pointer operations: MIR compares raw pointers and function
    // pointers directly. Equality is address identity in the
    // synthetic-address model (the same identity `PtrToInt`
    // exposes), as resolved by `addr_of` -- the caller supplies
    // the size-aware resolution ([`Interpreter::pointer_address`])
    // so that element positions in a zero-sized sequence compare
    // equal, as they do natively; a fat pointer's equality also
    // compares the metadata word (its length, or a dyn pointer's
    // vtable identity). Ordering is
    // defined on the resolved addresses, which is deterministic
    // within a run -- except between two pointers into the same
    // storage, whose order is their position order (see
    // `positional_order` and `structural_ptr_order`): the
    // synthetic address hashes the whole target, projections
    // included, so it orders two positions of one buffer
    // arbitrarily, while the guest's pointer preconditions
    // (`offset_from_unsigned`'s `self >= origin`, an iterator's
    // `end >= ptr`) are stated on the real order. A position
    // order is consulted only between distinct resolved
    // addresses: address-identical pointers have no position
    // order between them.
    if matches!(op, Eq | Ne | Lt | Le | Gt | Ge | Cmp) {
        let key = |v: &Value| -> Option<(u128, Option<u128>)> {
            match v {
                // The null pointer's address is 0, matching an
                // address-only pointer holding address 0 (the
                // synthetic range keeps `PtrToInt` of null itself
                // non-zero, a model choice the comparison relation
                // does not inherit).
                Value::Ref(RefTarget::Null) => Some((0, None)),
                Value::FatRef { len, .. } => addr_of(v).map(|a| (a, Some(*len))),
                // A dyn fat pointer's metadata word is its vtable,
                // whose identity in this model is the (concrete
                // type, trait set) pair the value carries. The data
                // address alone does not distinguish two dyn
                // pointers over one address -- the null address
                // included -- with different vtables, which native
                // wide-pointer equality keeps distinct, so the
                // comparison key carries the vtable word — the
                // very word the pointer flattens to (see
                // `vtable_word`), so the comparison order and
                // the flattened representation's order agree —
                // alongside the address (the address itself, for
                // null checks and `PtrToInt`, stays the base's).
                Value::DynRef {
                    concrete, traits, ..
                } => addr_of(v).map(|a| {
                    (
                        a,
                        Some(vtable_word(concrete, traits)),
                    )
                }),
                Value::Ref(_) | Value::FnPtr(_) => addr_of(v).map(|a| (a, None)),
                // A non-capturing closure value is a function
                // pointer; it compares by its body identity.
                Value::Closure { .. } => addr_of(v).map(|a| (a, None)),
                _ => None,
            }
        };
        let ka = key(lhs);
        let kb = key(rhs);
        let positional = if ka.is_some() && ka == kb {
            // Address-identical pointers have no position order
            // between them (see the comment above).
            None
        } else {
            match (lhs, rhs) {
                (Value::Ref(a), Value::Ref(b)) => positional_order(a, b),
                (
                    Value::FatRef { base: a, len: la },
                    Value::FatRef { base: b, len: lb },
                ) if la == lb => match (a.as_ref(), b.as_ref()) {
                    (Value::Ref(x), Value::Ref(y)) => positional_order(x, y),
                    _ => None,
                },
                _ => None,
            }
        };
        if let Some(ord) = positional {
            if op == Cmp {
                return Ok((ordering_of(ord), false));
            }
            let v = match op {
                Eq => ord == std::cmp::Ordering::Equal,
                Ne => ord != std::cmp::Ordering::Equal,
                Lt => ord == std::cmp::Ordering::Less,
                Le => ord != std::cmp::Ordering::Greater,
                Gt => ord == std::cmp::Ordering::Greater,
                Ge => ord != std::cmp::Ordering::Less,
                _ => unreachable!(),
            };
            return Ok((Value::Bool(v), false));
        }
        let structural = if matches!(op, Eq | Ne) {
            None
        } else if ka.is_some() && ka == kb {
            // Address-identical pointers have no position order
            // between them: the size-aware resolution maps every
            // element position of a zero-sized sequence to the
            // shared base address, so those positions compare
            // equal, as the address comparison below reports.
            // The structural order decides only between distinct
            // addresses into one storage.
            None
        } else {
            // A fat pointer's order is (address, metadata):
            // between two fat pointers at one address whose
            // metadata words differ (two slices of one buffer
            // starting at the same position with different
            // lengths), the position order is Equal and must
            // not pre-empt the metadata comparison below.
            let fat_meta_decides = matches!(
                (lhs, rhs),
                (
                    Value::FatRef { .. } | Value::DynRef { .. },
                    Value::FatRef { .. } | Value::DynRef { .. }
                )
            ) && match (ka, kb) {
                (Some(a), Some(b)) => a.0 == b.0 && a.1 != b.1,
                _ => false,
            };
            if fat_meta_decides {
                None
            } else {
                structural_ptr_order(lhs, rhs).filter(|ord| {
                    // For `Ord::cmp`, an equal position reached by
                    // different projections keeps the address order
                    // below, so `cmp` stays consistent with the
                    // identity equality; the relational operators
                    // take the position order as it stands.
                    op != Cmp || *ord != std::cmp::Ordering::Equal
                })
            }
        };
        if let Some(ord) = structural {
            if op == Cmp {
                return Ok((ordering_of(ord), false));
            }
            let v = match op {
                Lt => ord == std::cmp::Ordering::Less,
                Le => ord != std::cmp::Ordering::Greater,
                Gt => ord == std::cmp::Ordering::Greater,
                Ge => ord != std::cmp::Ordering::Less,
                _ => unreachable!(),
            };
            return Ok((Value::Bool(v), false));
        }
        if let (Some(a), Some(b)) = (ka, kb) {
            let ord = a.cmp(&b);
            if op == Cmp {
                return Ok((ordering_of(ord), false));
            }
            let v = match op {
                Eq => a == b,
                Ne => a != b,
                Lt => ord == std::cmp::Ordering::Less,
                Le => ord != std::cmp::Ordering::Greater,
                Gt => ord == std::cmp::Ordering::Greater,
                Ge => ord != std::cmp::Ordering::Less,
                _ => unreachable!(),
            };
            return Ok((Value::Bool(v), false));
        }
    }
        Err(InterpError::type_mismatch(
        format!("binary operation {op:?}"),
        "matching integer/bool operands",
        format!("{} and {}", lhs.kind_name(), rhs.kind_name()),
    ))
}

/// Evaluate a `Shl` / `Shr` operation.
///
/// The value (left operand) keeps its own type and signedness: the
/// declared width comes from the destination type exactly as in the
/// same-kind paths of `eval_binary`, a signed value shifts
/// arithmetically and an unsigned value logically. The amount
/// (right operand) may have any integer type; only its numeric
/// value matters. It is read as a `u32`, as rustc interprets shift
/// amounts: a negative amount or one above `u32::MAX` does not fit,
/// and `u32::MAX` stands in for it, which is at or above every
/// type width and so reports an overflowing amount either way.
///
/// An amount at or above the value's width is the only overflow a
/// shift has: both forms yield the wrapped result (the amount
/// taken modulo the width, the `wrapping_shl` / `wrapping_shr`
/// reading); the checked form also sets the overflow flag, and
/// the `Assert` that follows it in a checked build is what panics
/// (with Rust 1.100.0's native message, synthesized at conversion
/// time). Bits shifted out of the declared width are simply lost,
/// as natively: they set neither the overflow flag nor a panic,
/// in either form.
fn eval_shift(
    op: BinOp,
    lhs: &Value,
    rhs: &Value,
    dest_ty: Option<&Ty>,
    checked_form: bool,
) -> InterpResult<(Value, bool)> {
    let shift = match rhs {
        Value::Int(b) => u32::try_from(*b).unwrap_or(u32::MAX),
        Value::Uint(b) => u32::try_from(*b).unwrap_or(u32::MAX),
        _ => unreachable!("shift amounts are integer values"),
    };
    match lhs {
        Value::Int(a) => {
            let a = *a;
            let bits = signed_bits(dest_ty);
            if shift >= bits {
                let wrapped = if op == BinOp::Shl {
                    wrap_signed(a.wrapping_shl(shift % bits), bits)
                } else {
                    wrap_signed(a.wrapping_shr(shift % bits), bits)
                };
                return Ok((Value::Int(wrapped), checked_form));
            }
            let raw = if op == BinOp::Shl {
                wrap_signed(a << shift, bits)
            } else {
                a >> shift
            };
            Ok((Value::Int(wrap_signed(raw, bits)), false))
        }
        Value::Uint(a) => {
            let a = *a;
            let bits = unsigned_bits(dest_ty);
            if shift >= bits {
                let wrapped = if op == BinOp::Shl {
                    wrap_unsigned(a.wrapping_shl(shift % bits), bits)
                } else {
                    wrap_unsigned(a.wrapping_shr(shift % bits), bits)
                };
                return Ok((Value::Uint(wrapped), checked_form));
            }
            let raw = if op == BinOp::Shl {
                a << shift
            } else {
                a >> shift
            };
            Ok((Value::Uint(wrap_unsigned(raw, bits)), false))
        }
        _ => unreachable!("shift values are integer values"),
    }
}

/// The float width a binary operation's destination declares, if any.
/// A checked operation's destination is the `(T, bool)` tuple; the
/// operand width lives in its first element.
/// Round an `f64` value to `f16` precision (round-to-nearest-even,
/// as IEEE 754 requires) and widen it back: the stored form of an
/// `f16` place's value. Computing in `f64` first is exact for
/// `f16` operands by the same double-rounding argument as `f32`
/// (53 mantissa bits >= 2*11 + 2).
pub(crate) fn round_to_f16(v: f64) -> f64 {
    crate::mir_convert::f16_to_f64(f64_to_f16_bits(v))
}

/// The `f16` bit pattern nearest to `v` (round-to-nearest-even).
pub(crate) fn f64_to_f16_bits(v: f64) -> u16 {
    let bits = v.to_bits();
    let sign = (((bits >> 63) & 1) as u16) << 15;
    let exp = ((bits >> 52) & 0x7ff) as i64;
    let mant = bits & ((1u64 << 52) - 1);
    if exp == 0x7ff {
        // Infinity / NaN. A NaN keeps its sign and the top-aligned
        // payload of its `f64` storage form (an `f16` NaN is stored
        // as its exact widening, so this recovers the pattern bit
        // for bit, signaling NaNs included); a payload entirely
        // below the `f16` window keeps the quiet marker so the
        // result stays a NaN.
        if mant == 0 {
            return sign | 0x7c00;
        }
        let payload = (mant >> 42) as u16;
        return sign | 0x7c00 | if payload == 0 { 0x0200 } else { payload };
    }
    let e = exp - 1023;
    if e > 15 {
        return sign | 0x7c00;
    }
    if e >= -14 {
        // Normal f16: keep the top 10 mantissa bits.
        let mut kept = (mant >> 42) as u32;
        let dropped = mant & ((1u64 << 42) - 1);
        let half = 1u64 << 41;
        if dropped > half || (dropped == half && kept & 1 == 1) {
            kept += 1;
        }
        let mut e = e;
        if kept == 0x400 {
            kept = 0;
            e += 1;
            if e > 15 {
                return sign | 0x7c00;
            }
        }
        return sign | (((e + 15) as u16) << 10) | kept as u16;
    }
    if e < -25 {
        // Below the tie point of the smallest subnormal: zero
        // (the sign is preserved).
        return sign;
    }
    // Subnormal: the f16 subnormal unit is 2^-24 and the value is
    // m * 2^(e-52) with m carrying the implicit leading bit.
    let m = mant | (1u64 << 52);
    let shift = (28 - e) as u32;
    let mut kept = (m >> shift) as u32;
    let dropped = m & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    if dropped > half || (dropped == half && kept & 1 == 1) {
        kept += 1;
    }
    // `kept` may reach 0x400, which is exactly the smallest
    // normal's encoding, so it is used as-is.
    sign | kept as u16
}

fn float_ty_of(dest_ty: Option<&Ty>) -> Option<FloatTy> {
    match dest_ty {
        Some(Ty::Float(ft)) => Some(*ft),
        Some(Ty::Tuple(ts)) => float_ty_of(ts.first()),
        _ => None,
    }
}

pub(crate) fn eval_unary(op: UnOp, v: &Value, dest_ty: Option<&Ty>) -> InterpResult<Value> {
    match (op, v) {
        (UnOp::Neg, Value::Int(a)) => {
            // Plain `Neg` wraps, like the binary operators: the
            // overflow panic of a checked build is the following
            // `Assert`'s business (an `OverflowNeg` message), not
            // this operation's.
            let bits = signed_bits(dest_ty);
            Ok(Value::Int(wrap_signed(a.wrapping_neg(), bits)))
        }
        (UnOp::Neg, Value::Float(a)) => Ok(Value::Float(-a)),
        (UnOp::Neg, Value::Float128(a)) => Ok(Value::Float128(f128::neg(*a))),
        (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
        (UnOp::Not, Value::Int(a)) => Ok(Value::Int(wrap_signed(!a, signed_bits(dest_ty)))),
        (UnOp::Not, Value::Uint(a)) => Ok(Value::Uint(wrap_unsigned(!a, unsigned_bits(dest_ty)))),
        (UnOp::PtrMetadata, _) => Err(InterpError::type_mismatch(
            "PtrMetadata",
            "fat pointer",
            v.kind_name(),
        )),
        _ => Err(InterpError::type_mismatch(
            format!("unary operation {op:?}"),
            "integer or bool",
            v.kind_name(),
        )),
    }
}

/// Rust float-to-int cast semantics: saturate at the target type's
/// range, NaN becomes 0. The host `as` cast saturates at `i128`;
/// clamping to the declared width finishes the job.
pub(crate) fn float_to_int(v: f64, bits: u32) -> i128 {
    let wide = v as i128;
    if bits >= 128 {
        wide
    } else {
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        wide.clamp(min, max)
    }
}

/// Rust float-to-uint cast semantics: saturate at the target type's
/// range (negative values become 0), NaN becomes 0.
pub(crate) fn float_to_uint(v: f64, bits: u32) -> u128 {
    let wide = v as u128;
    if bits >= 128 {
        wide
    } else {
        wide.min((1u128 << bits) - 1)
    }
}

/// A stand-in numeric address for a pointer value. Places in this
/// value model have no real addresses, but an address is still
/// observable through pointer-to-integer transmutes — most
/// importantly the alignment precondition checks rustc emits
/// around raw-pointer dereferences, which compute
/// `addr & (align - 1) == 0`. The synthetic address is a pure
/// function of the pointer's target: the same target always yields
/// the same address, distinct targets yield distinct addresses
/// (up to the mixing hash), and every address is a multiple of
/// 65536, so the alignment check holds for any alignment up to
/// 2^16 — the largest a place in this model can carry in practice
/// (`#[repr(align(0x10000))]`). A reference to a constant aggregate
/// has no place identity; it shares one reserved address.
pub(crate) fn synthetic_addr(v: &Value) -> u128 {
    fn mix(h: &mut u128, x: u128) {
        // FNV-1a over the value's limbs.
        for limb in [x as u64, (x >> 64) as u64] {
            *h ^= limb as u128;
            *h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    if matches!(v, Value::Ref(RefTarget::Null)) {
        return 0;
    }
    let mut h: u128 = 0xcbf2_9ce4_8422_2325;
    match v {
        Value::Ref(RefTarget::Place {
            frame,
            local,
            projection,
        }) => {
            mix(&mut h, *frame as u128);
            mix(&mut h, *local as u128);
            for step in projection {
                let (tag, n) = match step {
                    RefProj::Field(i) => (1u128, *i as u128),
                    RefProj::UnionField { idx, table } => {
                        (5u128, ((*table as u128) << 64) | (*idx as u128))
                    }
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
                    RefProj::ByteOffset(d) => (6u128, *d as u128),
                    RefProj::VecLen => (7u128, 0u128),
                    RefProj::View(ty) => {
                        // The recorded denomination is part
                        // of the target's identity, as the
                        // other steps are.
                        let mut th: u128 = 0xcbf2_9ce4_8422_2325;
                        for b in format!("{ty:?}").bytes() {
                            th ^= u128::from(b);
                            th = th.wrapping_mul(0x100_0000_01b3);
                        }
                        (8u128, th)
                    }
                };
                mix(&mut h, tag);
                mix(&mut h, n);
            }
        }
        Value::Ref(RefTarget::Value(_, projection)) => {
            mix(&mut h, 0x5eed_5eed_5eed_5eed);
            for step in projection {
                let (tag, n) = match step {
                    RefProj::Field(i) => (1u128, *i as u128),
                    RefProj::UnionField { idx, table } => {
                        (5u128, ((*table as u128) << 64) | (*idx as u128))
                    }
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
                    RefProj::ByteOffset(d) => (6u128, *d as u128),
                    RefProj::VecLen => (7u128, 0u128),
                    RefProj::View(ty) => {
                        // The recorded denomination is part
                        // of the target's identity, as the
                        // other steps are.
                        let mut th: u128 = 0xcbf2_9ce4_8422_2325;
                        for b in format!("{ty:?}").bytes() {
                            th ^= u128::from(b);
                            th = th.wrapping_mul(0x100_0000_01b3);
                        }
                        (8u128, th)
                    }
                };
                mix(&mut h, tag);
                mix(&mut h, n);
            }
        }
        Value::Ref(RefTarget::Static { name, projection }) => {
            mix(&mut h, 0x57a7_1c5_7a7_1c5);
            for b in name.bytes() {
                mix(&mut h, b as u128);
            }
            for step in projection {
                let (tag, n) = match step {
                    RefProj::Field(i) => (1u128, *i as u128),
                    RefProj::UnionField { idx, table } => {
                        (5u128, ((*table as u128) << 64) | (*idx as u128))
                    }
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
                    RefProj::ByteOffset(d) => (6u128, *d as u128),
                    RefProj::VecLen => (7u128, 0u128),
                    RefProj::View(ty) => {
                        // The recorded denomination is part
                        // of the target's identity, as the
                        // other steps are.
                        let mut th: u128 = 0xcbf2_9ce4_8422_2325;
                        for b in format!("{ty:?}").bytes() {
                            th ^= u128::from(b);
                            th = th.wrapping_mul(0x100_0000_01b3);
                        }
                        (8u128, th)
                    }
                };
                mix(&mut h, tag);
                mix(&mut h, n);
            }
        }
        Value::Ref(RefTarget::Alloc { id, projection }) => {
            // An allocation's address identity is its id: every
            // copy of the box that owns it, and every pointer
            // extracted from any of them, yields this same
            // address, while distinct allocations differ.
            mix(&mut h, 0xa110_ca7e_a110_ca7e);
            mix(&mut h, *id as u128);
            for step in projection {
                let (tag, n) = match step {
                    RefProj::Field(i) => (1u128, *i as u128),
                    RefProj::UnionField { idx, table } => {
                        (5u128, ((*table as u128) << 64) | (*idx as u128))
                    }
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                    RefProj::Subslice { offset, len } => {
                        (4u128, ((*offset as u128) << 64) | (*len as u128))
                    }
                    RefProj::ByteOffset(d) => (6u128, *d as u128),
                    RefProj::VecLen => (7u128, 0u128),
                    RefProj::View(ty) => {
                        // The recorded denomination is part
                        // of the target's identity, as the
                        // other steps are.
                        let mut th: u128 = 0xcbf2_9ce4_8422_2325;
                        for b in format!("{ty:?}").bytes() {
                            th ^= u128::from(b);
                            th = th.wrapping_mul(0x100_0000_01b3);
                        }
                        (8u128, th)
                    }
                };
                mix(&mut h, tag);
                mix(&mut h, n);
            }
        }
        Value::FatRef { base, .. } => {
            return synthetic_addr(base);
        }
        Value::FnPtr(name) => {
            for b in name.bytes() {
                mix(&mut h, b as u128);
            }
        }
        _ => {}
    }
    const BASE: u128 = 0x1000_0000_0000;
    BASE + (h % (1 << 40)) * 65536
}
