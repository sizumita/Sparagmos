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
