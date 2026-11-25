use sparagmos_mir as mir;

use crate::ir::{FloatKind as IrFloatKind, IntKind as IrIntKind};

use super::adt::AdtLayouts;
use super::types::JitError;

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// The argument word layout: where each MIR argument's words sit
/// in the flat word sequence a call passes and a callee's prologue
/// reads. This is the single contract both sides compute — the
/// caller in `lower_call`, the callee in `FnLower::run` — from the
/// callee's parameter types (via [`flatten`]), so the two always
/// agree.
///
/// * Arguments are laid out in order, one flat word per scalar leaf.
/// * A zero-word argument (unit, never) still occupies one dummy
///   word, whose value nobody reads.
/// * A multi-word argument is never split across the
///   register/stack boundary: if it would start among the
///   register words but end past them, pad words are inserted
///   before it, so the argument lies entirely on the stack.
/// * Words 0..6 travel in the System V argument registers, words
///   6.. on the stack in order (word 6 at the lowest address).
///
/// Returns `(start word of each argument, total word count)`; the
/// total is the function's IR `param_count`, dummies and pads
/// included.
pub(super) fn arg_word_layout(counts: &[usize]) -> (Vec<usize>, usize) {
    let mut starts = Vec::with_capacity(counts.len());
    let mut word = 0usize;
    for &count in counts {
        let width = count.max(1);
        if width > 1 && word < 6 && word + width > 6 {
            word = 6; // pad: keep the argument off the register/stack seam
        }
        starts.push(word);
        word += width;
    }
    (starts, word)
}

/// Bit width of an integer type on this 64-bit target.
pub(super) fn int_bits(it: mir::IntTy) -> u32 {
    use mir::IntTy::*;
    match it {
        I8 => 8,
        I16 => 16,
        I32 => 32,
        I64 => 64,
        I128 => 128,
        Isize => 64,
    }
}

pub(super) fn uint_bits(ut: mir::UintTy) -> u32 {
    use mir::UintTy::*;
    match ut {
        U8 => 8,
        U16 => 16,
        U32 => 32,
        U64 => 64,
        U128 => 128,
        Usize => 64,
    }
}

/// The integer shape of a scalar type, if it has one the JIT supports.
/// Render a MIR place the way the interpreter's IR renders it
/// (`_1`, `_1.0`, `_1.*[_2]`, `_1[0..3]`, ...): the string the
/// interpreter's negative-index panic names. `OpaqueCast` /
/// `Subtype` steps render as nothing (the interpreter's IR has no
/// such steps).
pub(super) fn place_display(place: &mir::Place) -> String {
    let mut out = format!("_{}", place.local.index());
    for elem in &place.projection {
        match elem {
            mir::PlaceElem::Field(i, _) => out.push_str(&format!(".{i}")),
            mir::PlaceElem::Deref => out.push_str(".*"),
            mir::PlaceElem::Index(l) => out.push_str(&format!("[_{}]", l.index())),
            mir::PlaceElem::ConstantIndex {
                offset, from_end, ..
            } => {
                if *from_end {
                    out.push_str(&format!("[len - {offset}]"));
                } else {
                    out.push_str(&format!("[{offset}]"));
                }
            }
            mir::PlaceElem::Downcast(_, v) => out.push_str(&format!(" as variant {v}")),
            mir::PlaceElem::Subslice { from, to, from_end } => {
                if *from_end {
                    out.push_str(&format!("[{from}..len - {to}]"));
                } else {
                    out.push_str(&format!("[{from}..{to}]"));
                }
            }
            mir::PlaceElem::OpaqueCast(_) | mir::PlaceElem::Subtype(_) => {}
        }
    }
    out
}

/// `from * per` as a leaf offset, clamped: a range that does not
/// fit its base is caught by the place's sub-slice check before
/// the address is ever materialised, so an overflowing product
/// only needs a value that is never used.
pub(super) fn subslice_offset(from: u64, per: usize) -> usize {
    let v = (from as u128) * (per as u128);
    if v > usize::MAX as u128 {
        usize::MAX
    } else {
        v as usize
    }
}

/// The signedness of a 128-bit integer type (`i128` / `u128`),
/// whose values occupy a word pair in the slot model; `None` for
/// every other type.
pub(super) fn int128_signed(ty: &mir::Ty) -> Option<bool> {
    match ty {
        mir::Ty::Int(it) if int_bits(*it) == 128 => Some(true),
        mir::Ty::Uint(ut) if uint_bits(*ut) == 128 => Some(false),
        _ => None,
    }
}

/// Whether a type's derived `PartialEq` is leaf-wise word
/// equality: every leaf is a raw word whose semantic equality is
/// word equality. Floats are excluded (their equality is IEEE,
/// not bitwise), as are references and pointers inside the value
/// (a derived comparison would follow them), strings and slices
/// (variable-length data, not fixed words), and ADTs (a variant's
/// inactive field slots hold no defined value to compare).
pub(super) fn word_transparent(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::Bool | mir::Ty::Char | mir::Ty::Int(_) | mir::Ty::Uint(_) => true,
        mir::Ty::Array(elem, mir::ArrayLen::Known(_)) => word_transparent(elem),
        mir::Ty::Tuple(fields) => fields.iter().all(|f| word_transparent(f)),
        _ => false,
    }
}

pub(super) fn int_kind(ty: &mir::Ty) -> Option<IrIntKind> {
    match ty {
        mir::Ty::Int(it) => {
            let b = int_bits(*it);
            (b <= 64).then(|| IrIntKind::new(b as u8, true))
        }
        mir::Ty::Uint(ut) => {
            let b = uint_bits(*ut);
            (b <= 64).then(|| IrIntKind::new(b as u8, false))
        }
        mir::Ty::Bool => Some(IrIntKind::new(1, false)),
        mir::Ty::Char => Some(IrIntKind::new(32, false)),
        _ => None,
    }
}

/// Leaf bound for one array type. Every leaf is an 8-byte stack
/// slot in the compiled frame, so an unbounded array type would let
/// a guest program demand an unbounded frame; past this many scalar
/// elements the type is outside the JIT's range. (The interpreter
/// bounds `Repeat` element counts the same way, at 1,000,000; the
/// JIT's bound is in leaves because its frames are physical.)
pub(super) const MAX_ARRAY_LEAVES: usize = 4096;

/// Number of scalar leaves a value of `ty` occupies, or the reason
/// the type is outside the JIT's range.
pub(super) fn flatten(ty: &mir::Ty, fname: &str, layouts: &AdtLayouts) -> Result<usize, JitError> {
    match ty {
        mir::Ty::Bool | mir::Ty::Char => Ok(1),
        mir::Ty::Int(it) => Ok(if int_bits(*it) <= 64 { 1 } else { 2 }),
        mir::Ty::Uint(ut) => Ok(if uint_bits(*ut) <= 64 { 1 } else { 2 }),
        mir::Ty::Tuple(ts) => {
            let mut n = 0;
            for t in ts {
                n += flatten(t, fname, layouts)?;
            }
            Ok(n)
        }
        // A never-typed value is never produced; it occupies no leaf.
        mir::Ty::Never => Ok(0),
        mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _) => {
            // A thin reference / raw pointer is one word: the
            // pointee slot's address. A fat reference to a slice or
            // `str` is two words: the address, then the length — the
            // shape of the interpreter's `FatRef`. A trait object's
            // metadata is a vtable pointer, not a length, and stays
            // unsupported.
            match &**pointee {
                mir::Ty::Str | mir::Ty::Slice(_) => Ok(2),
                mir::Ty::Dynamic(..) => {
                    Err(unsup(fname, "fat reference / pointer type"))
                }
                _ => Ok(1),
            }
        }
        mir::Ty::Float(mir::FloatTy::F32) | mir::Ty::Float(mir::FloatTy::F64) => Ok(1),
        mir::Ty::Float(mir::FloatTy::F16) => {
            Err(unsup(fname, "16-bit floating-point type"))
        }
        mir::Ty::Float(mir::FloatTy::F128) => {
            Err(unsup(fname, "128-bit floating-point type"))
        }
        mir::Ty::Array(elem, len) => {
            // An array value is its elements' leaves in order, the
            // same shape the interpreter gives it (arrays ride as
            // tuples of their elements there).
            let mir::ArrayLen::Known(n) = len else {
                return Err(unsup(fname, "array type with unknown length"));
            };
            let per = flatten(elem, fname, layouts)?;
            let total = (*n as usize).checked_mul(per).unwrap_or(usize::MAX);
            if total > MAX_ARRAY_LEAVES {
                return Err(unsup(
                    fname,
                    "array type with more scalar elements than the JIT frame holds",
                ));
            }
            Ok(total)
        }
        mir::Ty::Adt(..) => {
            // An ADT value is its tag leaf (for a multi-variant
            // ADT) plus its payload area, under the layout the
            // program scan computed for this instantiation (see
            // `AdtLayouts`). An ADT the scan says nothing about
            // keeps the tag-only shape.
            layouts
                .adt_total(ty)
                .map_err(|reason| unsup(fname, &reason))
        }
        other => Err(unsup(fname, &format!("type {}", other.kind_name()))),
    }
}

/// The backend float format of a scalar type, if it is one the JIT
/// supports.
pub(super) fn float_kind(ty: &mir::Ty) -> Option<IrFloatKind> {
    match ty {
        mir::Ty::Float(mir::FloatTy::F32) => Some(IrFloatKind::F32),
        mir::Ty::Float(mir::FloatTy::F64) => Some(IrFloatKind::F64),
        _ => None,
    }
}

pub(super) fn unsup(fname: &str, what: &str) -> JitError {
    JitError::Unsupported(format!("function `{fname}`: {what}"))
}

/// Reduce a raw `u128` target value of a `SwitchInt` to the canonical
/// `i64` for the discriminant type (target values are stored unsigned
/// in MIR; signed types sign-extend from their width).
pub(super) fn normalize_u128(v: u128, ty: &mir::Ty) -> i64 {
    match int_kind(ty) {
        Some(k) => {
            let bits = k.bits as u32;
            if bits >= 64 {
                v as i64
            } else if k.signed {
                let shift = 64 - bits;
                (((v as u64) << shift) as i64) >> shift
            } else {
                (v as u64 & ((1u64 << bits) - 1)) as i64
            }
        }
        None => v as i64,
    }
}

/// The shared type of a pair of integer literals, for a comparison
/// neither of whose operands has a place type. Mirrors the
/// interpreter's rule: both literals must have the same byte size;
/// if either has the width's sign bit set the pair is read as the
/// signed integer of that width, otherwise as the unsigned one.
pub(super) fn literal_pair_ty(lhs: &mir::Operand, rhs: &mir::Operand) -> Option<mir::Ty> {
    fn scalar_int(op: &mir::Operand) -> Option<(u128, u8)> {
        let mir::Operand::Constant(c) = op else {
            return None;
        };
        let mir::ConstKind::Value(mir::ConstValue::Scalar(mir::Scalar::Int { data, size })) =
            &c.literal
        else {
            return None;
        };
        Some((*data, *size))
    }
    let (ldata, lsize) = scalar_int(lhs)?;
    let (rdata, rsize) = scalar_int(rhs)?;
    if lsize != rsize {
        return None;
    }
    let bits = u32::from(lsize) * 8;
    let sign_bit = |data: u128| bits > 0 && (data >> (bits - 1)) & 1 == 1;
    if sign_bit(ldata) || sign_bit(rdata) {
        int_ty_of_bits(bits).map(mir::Ty::Int)
    } else {
        uint_ty_of_bits(bits).map(mir::Ty::Uint)
    }
}

pub(super) fn int_ty_of_bits(bits: u32) -> Option<mir::IntTy> {
    use mir::IntTy::*;
    Some(match bits {
        8 => I8,
        16 => I16,
        32 => I32,
        64 => I64,
        128 => I128,
        _ => return None,
    })
}

pub(super) fn uint_ty_of_bits(bits: u32) -> Option<mir::UintTy> {
    use mir::UintTy::*;
    Some(match bits {
        8 => U8,
        16 => U16,
        32 => U32,
        64 => U64,
        128 => U128,
        _ => return None,
    })
}

/// The shared type of a pair of float literals, for a comparison
/// neither of whose operands has a place type (the float form of
/// [`literal_pair_ty`]).
pub(super) fn literal_float_pair_ty(lhs: &mir::Operand, rhs: &mir::Operand) -> Option<mir::Ty> {
    fn scalar_float(op: &mir::Operand) -> Option<mir::FloatTy> {
        let mir::Operand::Constant(c) = op else {
            return None;
        };
        let mir::ConstKind::Value(mir::ConstValue::Scalar(mir::Scalar::Float { ty, .. })) =
            &c.literal
        else {
            return None;
        };
        Some(*ty)
    }
    let l = scalar_float(lhs)?;
    let r = scalar_float(rhs)?;
    if l == r && matches!(l, mir::FloatTy::F32 | mir::FloatTy::F64) {
        Some(mir::Ty::Float(l))
    } else {
        None
    }
}

/// The integer type of a lone integer literal, typed by the same
/// heuristic as [`literal_pair_ty`]: a set sign bit at the literal's
/// own width reads as the signed type of that width, otherwise the
/// unsigned one. Used for the source of an `IntToFloat` cast whose
/// operand is a constant.
pub(super) fn literal_int_ty(op: &mir::Operand) -> Option<mir::Ty> {
    let mir::Operand::Constant(c) = op else {
        return None;
    };
    let mir::ConstKind::Value(mir::ConstValue::Scalar(mir::Scalar::Int { data, size })) =
        &c.literal
    else {
        return None;
    };
    let bits = u32::from(*size) * 8;
    if bits > 0 && bits < 128 && (data >> (bits - 1)) & 1 == 1 {
        int_ty_of_bits(bits).map(mir::Ty::Int)
    } else {
        uint_ty_of_bits(bits).map(mir::Ty::Uint)
    }
}

/// The panic message text for an `Assert`, mirroring the
/// interpreter's rule: a custom message verbatim; for the
/// compiler-generated kinds, the exact message core panics with;
/// kinds whose message embeds operand values keep the stable
/// `assertion failed: <kind>` text.
pub(super) fn assert_message_text(msg: &mir::AssertMessage) -> String {
    use mir::AssertMessage as A;
    match msg {
        A::Custom(s) => s.clone(),
        A::Overflow(op, _, _) => match op {
            mir::BinOp::Add | mir::BinOp::AddWithOverflow => {
                "attempt to add with overflow".to_string()
            }
            mir::BinOp::Sub | mir::BinOp::SubWithOverflow => {
                "attempt to subtract with overflow".to_string()
            }
            mir::BinOp::Mul | mir::BinOp::MulWithOverflow => {
                "attempt to multiply with overflow".to_string()
            }
            mir::BinOp::Div => "attempt to divide with overflow".to_string(),
            mir::BinOp::Rem => {
                "attempt to calculate the remainder with overflow".to_string()
            }
            mir::BinOp::Shl | mir::BinOp::ShlUnchecked => {
                "attempt to shift left with overflow".to_string()
            }
            mir::BinOp::Shr | mir::BinOp::ShrUnchecked => {
                "attempt to shift right with overflow".to_string()
            }
            other => format!("assertion failed: {other:?}"),
        },
        A::OverflowNeg(_) => "attempt to negate with overflow".to_string(),
        A::DivisionByZero(_) => "attempt to divide by zero".to_string(),
        A::RemainderByZero(_) => {
            "attempt to calculate the remainder with a divisor of zero".to_string()
        }
        other => format!("assertion failed: {}", other.kind_name()),
    }
}
