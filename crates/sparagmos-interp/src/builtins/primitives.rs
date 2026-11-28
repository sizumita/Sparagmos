//! Primitive binary and unary operation execution.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    eval_binary, eval_unary, synthetic_address, Frame, Interpreter,
};
use crate::ir::{BinOp, Ty, UnOp};
use crate::value::Value;

use super::*;

impl<'a> Interpreter<'a> {
    /// Execute a primitive arithmetic trait method served by
    /// the run-time name fallback: the name carries the Self
    /// type, whose tag selects the operation's width and
    /// signedness, and the computation is the interpreter's own
    /// binary evaluation (division by zero panics there, as the
    /// operation itself requires). In an overflow-checked build
    /// the core impls panic on overflow — the same panic the
    /// `Assert` after a checked MIR operation raises — so the
    /// checked form's flag is consulted here; otherwise the
    /// wrapped result stands. The flag is the instantiating
    /// crate's, resolved from the caller stack (see
    /// `Interpreter::overflow_checks_at`).
    pub(super) fn exec_primitive_binop(
        &self,
        frames: &[Frame],
        name: &str,
        op: BinOp,
        lhs: &Value,
        rhs: &Value,
    ) -> InterpResult<Value> {
        let ty = primitive_self_ty(name)?;
        let (result, overflowed) =
            eval_binary(op, lhs, rhs, Some(&ty), true, &synthetic_address)?;
        if overflowed && self.overflow_checks_at(frames) {
            return Err(InterpError::panic_str_payload(
                primitive_overflow_message(op),
            ));
        }
        Ok(result)
    }

    /// The unary counterpart: `Neg` on the Self type's minimum
    /// is the one overflowing case (the checked form of the
    /// unary evaluation wraps it, leaving the panic to the
    /// following `Assert` in MIR; here the call is the whole
    /// operation, so the panic is raised directly).
    pub(super) fn exec_primitive_unop(
        &self,
        frames: &[Frame],
        name: &str,
        op: UnOp,
        value: &Value,
    ) -> InterpResult<Value> {
        let ty = primitive_self_ty(name)?;
        if op == UnOp::Neg && self.overflow_checks_at(frames) {
            if let (Value::Int(a), Ty::Int(it)) = (value, &ty) {
                let bits = it.bits();
                let min = if bits >= 128 {
                    i128::MIN
                } else {
                    -(1i128 << (bits - 1))
                };
                if *a == min {
                    return Err(InterpError::panic_str_payload(
                        "attempt to negate with overflow",
                    ));
                }
            }
        }
        eval_unary(op, value, Some(&ty))
    }
}

/// Registration fragment of the builtin name registry: the bit-operation intrinsics.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn primitives_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        // The `hint` spellings name the public const fn whose
        // whole body is the intrinsic call. Registering them is
        // load-bearing for the instances whose argument type the
        // runtime cannot represent in a converted body (an `f16`
        // value, a reference to an extern type): collection skips
        // those instances, so without a registry entry the call
        // site is left with neither a body nor a builtin.
        "std::intrinsics::black_box" | "core::intrinsics::black_box" => Builtin::BlackBox,
        "std::hint::black_box" | "core::hint::black_box" => Builtin::BlackBox,
        "std::intrinsics::ctpop" | "core::intrinsics::ctpop" => Builtin::IntBitOp(BitOp::CtPop),
        "std::intrinsics::ctlz" | "core::intrinsics::ctlz" => Builtin::IntBitOp(BitOp::Ctlz),
        "std::intrinsics::cttz" | "core::intrinsics::cttz" => Builtin::IntBitOp(BitOp::Cttz),
        // The `_nonzero` variants promise a non-zero operand (a
        // zero operand is undefined behaviour in the guest); on
        // every input where they are defined they agree with the
        // plain operation, which is what the model computes.
        "std::intrinsics::ctlz_nonzero" | "core::intrinsics::ctlz_nonzero" => {
            Builtin::IntBitOp(BitOp::Ctlz)
        }
        "std::intrinsics::cttz_nonzero" | "core::intrinsics::cttz_nonzero" => {
            Builtin::IntBitOp(BitOp::Cttz)
        }
        "std::intrinsics::bitreverse" | "core::intrinsics::bitreverse" => {
            Builtin::IntBitOp(BitOp::BitReverse)
        }
        "std::intrinsics::bswap" | "core::intrinsics::bswap" => Builtin::IntBitOp(BitOp::Bswap),
        "std::intrinsics::saturating_add" | "core::intrinsics::saturating_add" => {
            Builtin::SaturatingArith(SatOp::Add)
        }
        "std::intrinsics::saturating_sub" | "core::intrinsics::saturating_sub" => {
            Builtin::SaturatingArith(SatOp::Sub)
        }
        "std::intrinsics::exact_div" | "core::intrinsics::exact_div" => Builtin::ExactDiv,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the slice-sort helper surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn slice_sort_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "core::slice::sort::stable::drift::sqrt_approx" => Builtin::SqrtApprox,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the validity-assert intrinsics.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn assert_intrinsics_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::intrinsics::assert_inhabited" | "core::intrinsics::assert_inhabited" => {
            Builtin::AssertValidity
        }
        "std::intrinsics::assert_zero_valid" | "core::intrinsics::assert_zero_valid" => {
            Builtin::AssertValidity
        }
        "std::intrinsics::assert_mem_uninitialized_valid"
        | "core::intrinsics::assert_mem_uninitialized_valid" => Builtin::AssertValidity,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the
/// overflow-checks configuration intrinsic.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn overflow_checks_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::intrinsics::overflow_checks" | "core::intrinsics::overflow_checks" => {
            Builtin::OverflowChecks
        }
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the integer-power overflow helper.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn num_pow_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "core::num::imp::overflow_panic::pow" => Builtin::IntPow,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the
/// inherent integer methods whose overflow behaviour core
/// gates on the instantiating crate's setting.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// The `pow` methods dispatch through the same model as the
/// `overflow_panic::pow` helper their checked path is built
/// from; the funnel shifts keep their own variant. The
/// collected bodies of all of these have the
/// `intrinsics::overflow_checks()` predicate folded to a
/// constant of core's own build, so the registry's model is
/// the only faithful execution.
///
/// [`lookup`]: super::lookup
pub(crate) fn int_inherent_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "core::num::<impl u8>::funnel_shl"
        | "core::num::<impl u16>::funnel_shl"
        | "core::num::<impl u32>::funnel_shl"
        | "core::num::<impl u64>::funnel_shl"
        | "core::num::<impl u128>::funnel_shl"
        | "core::num::<impl usize>::funnel_shl" => Builtin::FunnelShift { left: true },
        "core::num::<impl u8>::funnel_shr"
        | "core::num::<impl u16>::funnel_shr"
        | "core::num::<impl u32>::funnel_shr"
        | "core::num::<impl u64>::funnel_shr"
        | "core::num::<impl u128>::funnel_shr"
        | "core::num::<impl usize>::funnel_shr" => Builtin::FunnelShift { left: false },
        "core::num::<impl i8>::pow"
        | "core::num::<impl i16>::pow"
        | "core::num::<impl i32>::pow"
        | "core::num::<impl i64>::pow"
        | "core::num::<impl i128>::pow"
        | "core::num::<impl isize>::pow"
        | "core::num::<impl u8>::pow"
        | "core::num::<impl u16>::pow"
        | "core::num::<impl u32>::pow"
        | "core::num::<impl u64>::pow"
        | "core::num::<impl u128>::pow"
        | "core::num::<impl usize>::pow" => Builtin::IntPow,
        _ => return None,
    })
}
