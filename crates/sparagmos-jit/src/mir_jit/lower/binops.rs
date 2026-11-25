
use sparagmos_mir as mir;

use crate::ir::{
    BinOp as IrBinOp, BinOp128 as IrBinOp128, CheckedOp,
    FloatBinOp as IrFloatBinOp, FloatKind as IrFloatKind, Instr,
    IntKind as IrIntKind, Operand as IrOperand, ShiftOp128 as IrShiftOp128,
    Terminator as IrTerminator, UnOp as IrUnOp,
};
use crate::builtins::{self};

use super::*;

impl<'a> FnLower<'a> {
    /// The base operation of a plain or `*WithOverflow` MIR operator,
    /// for the checked path.
    fn checked_base(op: mir::BinOp) -> Option<CheckedOp> {
        use mir::BinOp as M;
        match op {
            M::Add | M::AddWithOverflow => Some(CheckedOp::Add),
            M::Sub | M::SubWithOverflow => Some(CheckedOp::Sub),
            M::Mul | M::MulWithOverflow => Some(CheckedOp::Mul),
            _ => None,
        }
    }

    pub(super) fn lower_checked_binop_to(
        &mut self,
        dst: u32,
        dst_ty: &mir::Ty,
        op: mir::BinOp,
        lhs: &mir::Operand,
        rhs: &mir::Operand,
    ) -> Result<(), JitError> {
        // The 128-bit form: the (value, bool) result is three
        // words — the value's pair, then the flag.
        if let mir::Ty::Tuple(fields) = dst_ty {
            if fields.len() == 2 {
                if let Some(signed) = int128_signed(&fields[0]) {
                    if matches!(op, mir::BinOp::Shl | mir::BinOp::Shr) {
                        return self.lower_checked_shift128_to(
                            dst,
                            op == mir::BinOp::Shl,
                            lhs,
                            rhs,
                            &fields[0],
                        );
                    }
                    let Some(base) = Self::checked_base(op) else {
                        return Err(self.err(&format!("checked operation {op:?}")));
                    };
                    let val_ty = fields[0].clone();
                    let l = self.operand_leaves(lhs, &val_ty, 2)?;
                    let r = self.operand_leaves(rhs, &val_ty, 2)?;
                    self.emit(Instr::CheckedBinOp128 {
                        dst_val: dst,
                        dst_val2: dst + 1,
                        dst_flag: dst + 2,
                        op: base,
                        signed,
                        lhs_lo: IrOperand::Value(l[0]),
                        lhs_hi: IrOperand::Value(l[1]),
                        rhs_lo: IrOperand::Value(r[0]),
                        rhs_hi: IrOperand::Value(r[1]),
                    });
                    return Ok(());
                }
            }
        }
        if matches!(op, mir::BinOp::Shl | mir::BinOp::Shr) {
            return self.lower_checked_shift_to(dst, dst_ty, op == mir::BinOp::Shl, lhs, rhs);
        }
        let Some(base) = Self::checked_base(op) else {
            return Err(self.err(&format!("checked operation {op:?}")));
        };
        let mir::Ty::Tuple(fields) = dst_ty else {
            return Err(self.err("checked operation result is not a (value, bool) tuple"));
        };
        if fields.len() != 2 {
            return Err(self.err("checked operation result is not a (value, bool) tuple"));
        }
        let val_ty = fields[0].clone();
        let Some(kind) = int_kind(&val_ty) else {
            return Err(self.err("checked operation on a non-integer type"));
        };
        let lv = self.operand_value(lhs, &val_ty)?;
        let rv = self.operand_value(rhs, &val_ty)?;
        self.emit(Instr::CheckedBinOp {
            dst_val: dst,
            dst_flag: dst + 1,
            op: base,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(rv),
            kind,
        });
        Ok(())
    }

    pub(super) fn lower_binop_to(
        &mut self,
        dst: u32,
        dst_ty: &mir::Ty,
        op: mir::BinOp,
        lhs: &mir::Operand,
        rhs: &mir::Operand,
    ) -> Result<(), JitError> {
        use mir::BinOp as M;
        // A plain BinaryOp with a *WithOverflow operator also yields
        // the (value, bool) pair.
        if Self::checked_base(op).is_some()
            && matches!(op, M::AddWithOverflow | M::SubWithOverflow | M::MulWithOverflow)
        {
            return self.lower_checked_binop_to(dst, dst_ty, op, lhs, rhs);
        }
        let is_compare = matches!(op, M::Eq | M::Ne | M::Lt | M::Le | M::Gt | M::Ge);
        // Arithmetic operands share the destination's type; comparison
        // operands share each other's type (the destination is bool,
        // which says nothing about them). A comparison of two literals
        // — the shape of rustc's expanded division checks, `Eq(4, 0)`
        // — has no place type anywhere, so the pair is typed from the
        // literals themselves (see literal_pair_ty); typing it by the
        // bool destination would collapse every nonzero literal to 1.
        let op_ty = if is_compare {
            self.operand_place_ty(lhs)
                .or_else(|| self.operand_place_ty(rhs))
                .or_else(|| literal_pair_ty(lhs, rhs))
                .or_else(|| literal_float_pair_ty(lhs, rhs))
                .unwrap_or_else(|| dst_ty.clone())
        } else {
            dst_ty.clone()
        };
        // A 128-bit integer operation works on the operands' word
        // pair: the destination is the pair itself for arithmetic
        // and a single bool word for a comparison.
        if let Some(signed) = int128_signed(&op_ty) {
            return self.lower_binop128_to(dst, op, lhs, rhs, &op_ty, signed);
        }
        if flatten(dst_ty, &self.func.name, self.layouts)? != 1 {
            return Err(self.err("binary operation on an aggregate value"));
        }
        if let Some(fk) = float_kind(&op_ty) {
            return self.lower_float_binop(dst, op, lhs, rhs, fk, &op_ty);
        }
        let Some(kind) = int_kind(&op_ty) else {
            return Err(self.err(&format!(
                "binary operation {op:?} on type {}",
                op_ty.kind_name()
            )));
        };
        let lv = self.operand_value(lhs, &op_ty)?;
        let rv = self.operand_value(rhs, &op_ty)?;
        let signed = kind.signed;
        match op {
            M::Add | M::AddUnchecked | M::Sub | M::SubUnchecked | M::Mul | M::MulUnchecked => {
                // A plain arithmetic operator wraps on overflow
                // (the overflow panic belongs to the `Assert` after
                // a checked operation): compile it as the checked
                // operation with the flag unused.
                let base = match op {
                    M::Add | M::AddUnchecked => CheckedOp::Add,
                    M::Sub | M::SubUnchecked => CheckedOp::Sub,
                    _ => CheckedOp::Mul,
                };
                self.lower_plain_arith(dst, base, lv, rv, kind)?;
            }
            M::BitAnd | M::BitOr | M::BitXor => {
                let ir_op = match op {
                    M::BitAnd => IrBinOp::And,
                    M::BitOr => IrBinOp::Or,
                    _ => IrBinOp::Xor,
                };
                self.emit(Instr::BinOp {
                    dst,
                    op: ir_op,
                    lhs: IrOperand::Value(lv),
                    rhs: IrOperand::Value(rv),
                });
                self.normalize(dst, dst_ty);
            }
            M::Div | M::Rem => {
                self.lower_div_rem(dst, op == M::Rem, lv, rv, kind, dst_ty)?;
            }
            M::Shl | M::ShlUnchecked | M::Shr | M::ShrUnchecked => {
                self.lower_shift(
                    dst,
                    matches!(op, M::Shl | M::ShlUnchecked),
                    lv,
                    rv,
                    kind,
                    dst_ty,
                )?;
            }
            M::Eq | M::Ne | M::Lt | M::Le | M::Gt | M::Ge => {
                let ir_op = match (op, signed) {
                    (M::Eq, _) => IrBinOp::Eq,
                    (M::Ne, _) => IrBinOp::Ne,
                    (M::Lt, true) => IrBinOp::Lt,
                    (M::Lt, false) => IrBinOp::LtU,
                    (M::Le, true) => IrBinOp::Le,
                    (M::Le, false) => IrBinOp::LeU,
                    (M::Gt, true) => IrBinOp::Gt,
                    (M::Gt, false) => IrBinOp::GtU,
                    (M::Ge, true) => IrBinOp::Ge,
                    (M::Ge, false) => IrBinOp::GeU,
                    _ => unreachable!(),
                };
                self.emit(Instr::BinOp {
                    dst,
                    op: ir_op,
                    lhs: IrOperand::Value(lv),
                    rhs: IrOperand::Value(rv),
                });
            }
            M::Offset => return Err(self.err("Offset binary operation")),
            M::Cmp => return Err(self.err("Cmp binary operation")),
            M::AddWithOverflow | M::SubWithOverflow | M::MulWithOverflow => unreachable!(),
        }
        Ok(())
    }

    /// A binary operation on 128-bit integers: both operands are
    /// word pairs (low word first, the layout every 128-bit value
    /// has in the slot model), computed on by the backend's pair
    /// operations. The destination is the pair for arithmetic and
    /// shifts, a single bool word for a comparison.
    fn lower_binop128_to(
        &mut self,
        dst: u32,
        op: mir::BinOp,
        lhs: &mir::Operand,
        rhs: &mir::Operand,
        op_ty: &mir::Ty,
        signed: bool,
    ) -> Result<(), JitError> {
        use mir::BinOp as M;
        match op {
            M::Shl | M::ShlUnchecked | M::Shr | M::ShrUnchecked => {
                let l = self.operand_leaves(lhs, op_ty, 2)?;
                let amount = self.operand_value(rhs, op_ty)?;
                // The plain operators take the amount modulo the
                // type width, as the interpreter's plain form does
                // (the overshift panic belongs to the `Assert`
                // after a checked shift).
                let masked = self.fresh();
                self.emit(Instr::BinOp {
                    dst: masked,
                    op: IrBinOp::And,
                    lhs: IrOperand::Value(amount),
                    rhs: IrOperand::Const(127),
                });
                let shift_op = if matches!(op, M::Shl | M::ShlUnchecked) {
                    IrShiftOp128::Shl
                } else if signed {
                    IrShiftOp128::Sar
                } else {
                    IrShiftOp128::Shr
                };
                self.emit(Instr::Shift128 {
                    dst,
                    dst2: dst + 1,
                    op: shift_op,
                    lo: IrOperand::Value(l[0]),
                    hi: IrOperand::Value(l[1]),
                    amount: IrOperand::Value(masked),
                });
                Ok(())
            }
            M::Div | M::Rem => {
                let l = self.operand_leaves(lhs, op_ty, 2)?;
                let r = self.operand_leaves(rhs, op_ty, 2)?;
                // A zero divisor panics with the same messages as
                // the 64-bit path; the pair is zero exactly when
                // both of its words are.
                let msg = self.messages.intern(if op == M::Rem {
                    "attempt to calculate the remainder with a divisor of zero"
                } else {
                    "attempt to divide by zero"
                });
                let pblk = self.panic_block(msg);
                let lo_nz = self.fresh();
                self.emit(Instr::BinOp {
                    dst: lo_nz,
                    op: IrBinOp::Ne,
                    lhs: IrOperand::Value(r[0]),
                    rhs: IrOperand::Const(0),
                });
                let hi_nz = self.fresh();
                self.emit(Instr::BinOp {
                    dst: hi_nz,
                    op: IrBinOp::Ne,
                    lhs: IrOperand::Value(r[1]),
                    rhs: IrOperand::Const(0),
                });
                let nonzero = self.fresh();
                self.emit(Instr::BinOp {
                    dst: nonzero,
                    op: IrBinOp::Or,
                    lhs: IrOperand::Value(lo_nz),
                    rhs: IrOperand::Value(hi_nz),
                });
                let cont = self.new_block();
                self.seal_and_continue(
                    IrTerminator::Branch {
                        cond: IrOperand::Value(nonzero),
                        then_blk: cont,
                        else_blk: pblk,
                    },
                    cont,
                );
                let ir_op = match (op, signed) {
                    (M::Div, true) => IrBinOp128::Div,
                    (M::Div, false) => IrBinOp128::DivU,
                    (M::Rem, true) => IrBinOp128::Rem,
                    _ => IrBinOp128::RemU,
                };
                self.emit(Instr::BinOp128 {
                    dst,
                    dst2: dst + 1,
                    op: ir_op,
                    lhs_lo: IrOperand::Value(l[0]),
                    lhs_hi: IrOperand::Value(l[1]),
                    rhs_lo: IrOperand::Value(r[0]),
                    rhs_hi: IrOperand::Value(r[1]),
                });
                Ok(())
            }
            M::Eq | M::Ne | M::Lt | M::Le | M::Gt | M::Ge => {
                let l = self.operand_leaves(lhs, op_ty, 2)?;
                let r = self.operand_leaves(rhs, op_ty, 2)?;
                let result = self.emit_cmp128(op, signed, &l, &r);
                self.emit(Instr::Copy {
                    dst,
                    src: IrOperand::Value(result),
                });
                Ok(())
            }
            M::Add
            | M::AddUnchecked
            | M::Sub
            | M::SubUnchecked
            | M::Mul
            | M::MulUnchecked
            | M::BitAnd
            | M::BitOr
            | M::BitXor => {
                let l = self.operand_leaves(lhs, op_ty, 2)?;
                let r = self.operand_leaves(rhs, op_ty, 2)?;
                let ir_op = match op {
                    M::Add | M::AddUnchecked => IrBinOp128::Add,
                    M::Sub | M::SubUnchecked => IrBinOp128::Sub,
                    M::Mul | M::MulUnchecked => IrBinOp128::Mul,
                    M::BitAnd => IrBinOp128::And,
                    M::BitOr => IrBinOp128::Or,
                    _ => IrBinOp128::Xor,
                };
                self.emit(Instr::BinOp128 {
                    dst,
                    dst2: dst + 1,
                    op: ir_op,
                    lhs_lo: IrOperand::Value(l[0]),
                    lhs_hi: IrOperand::Value(l[1]),
                    rhs_lo: IrOperand::Value(r[0]),
                    rhs_hi: IrOperand::Value(r[1]),
                });
                Ok(())
            }
            M::Offset => Err(self.err("Offset binary operation")),
            M::Cmp => Err(self.err("Cmp binary operation")),
            M::AddWithOverflow | M::SubWithOverflow | M::MulWithOverflow => {
                unreachable!()
            }
        }
    }

    /// Emit `a op b` (both already in slots) into a fresh slot
    /// and return it.
    fn emit_binop_fresh(&mut self, op: IrBinOp, a: u32, b: u32) -> u32 {
        let d = self.fresh();
        self.emit(Instr::BinOp {
            dst: d,
            op,
            lhs: IrOperand::Value(a),
            rhs: IrOperand::Value(b),
        });
        d
    }

    /// Emit a 128-bit comparison of the word pairs `l` and `r`
    /// (slot pairs, low word first) and return the slot holding
    /// the 0/1 result. The high words decide, compared with the
    /// type's signedness; equal high words fall to the low words,
    /// compared unsigned, as the pair's numeric order dictates.
    pub(super) fn emit_cmp128(
        &mut self,
        op: mir::BinOp,
        signed: bool,
        l: &[u32],
        r: &[u32],
    ) -> u32 {
        use mir::BinOp as M;
        match op {
            M::Eq | M::Ne => {
                let lo_eq = self.emit_binop_fresh(IrBinOp::Eq, l[0], r[0]);
                let hi_eq = self.emit_binop_fresh(IrBinOp::Eq, l[1], r[1]);
                let both = self.emit_binop_fresh(IrBinOp::And, lo_eq, hi_eq);
                if op == M::Eq {
                    both
                } else {
                    let d = self.fresh();
                    self.emit(Instr::BinOp {
                        dst: d,
                        op: IrBinOp::Xor,
                        lhs: IrOperand::Value(both),
                        rhs: IrOperand::Const(1),
                    });
                    d
                }
            }
            M::Lt | M::Le | M::Gt | M::Ge => {
                // `Gt` / `Ge` compare the swapped pair; `Le` / `Ge`
                // add the equality case to the strict order.
                let (a, b) = if matches!(op, M::Gt | M::Ge) {
                    (r, l)
                } else {
                    (l, r)
                };
                let hi_op = if signed { IrBinOp::Lt } else { IrBinOp::LtU };
                let hi_lt = self.emit_binop_fresh(hi_op, a[1], b[1]);
                let hi_eq = self.emit_binop_fresh(IrBinOp::Eq, a[1], b[1]);
                let lo_lt = self.emit_binop_fresh(IrBinOp::LtU, a[0], b[0]);
                let tie = self.emit_binop_fresh(IrBinOp::And, hi_eq, lo_lt);
                let lt = self.emit_binop_fresh(IrBinOp::Or, hi_lt, tie);
                if matches!(op, M::Lt | M::Gt) {
                    lt
                } else {
                    let lo_eq = self.emit_binop_fresh(IrBinOp::Eq, a[0], b[0]);
                    let eq = self.emit_binop_fresh(IrBinOp::And, hi_eq, lo_eq);
                    self.emit_binop_fresh(IrBinOp::Or, lt, eq)
                }
            }
            _ => unreachable!("emit_cmp128 on a non-comparison operator"),
        }
    }

    /// A plain `BinaryOp` on floats: arithmetic and comparisons map
    /// directly onto the backend's float operations (the hardware's
    /// IEEE semantics are Rust's, including the NaN comparison
    /// rules). Bitwise and shift operators have no float form.
    fn lower_float_binop(
        &mut self,
        dst: u32,
        op: mir::BinOp,
        lhs: &mir::Operand,
        rhs: &mir::Operand,
        kind: IrFloatKind,
        op_ty: &mir::Ty,
    ) -> Result<(), JitError> {
        use mir::BinOp as M;
        let fop = match op {
            M::Add | M::AddUnchecked => IrFloatBinOp::Add,
            M::Sub | M::SubUnchecked => IrFloatBinOp::Sub,
            M::Mul | M::MulUnchecked => IrFloatBinOp::Mul,
            M::Div => IrFloatBinOp::Div,
            M::Rem => IrFloatBinOp::Rem,
            M::Eq => IrFloatBinOp::Eq,
            M::Ne => IrFloatBinOp::Ne,
            M::Lt => IrFloatBinOp::Lt,
            M::Le => IrFloatBinOp::Le,
            M::Gt => IrFloatBinOp::Gt,
            M::Ge => IrFloatBinOp::Ge,
            other => {
                return Err(self.err(&format!(
                    "binary operation {other:?} on a floating-point type"
                )));
            }
        };
        let lv = self.operand_value(lhs, op_ty)?;
        let rv = self.operand_value(rhs, op_ty)?;
        self.emit(Instr::FloatBinOp {
            dst,
            op: fop,
            kind,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(rv),
        });
        Ok(())
    }

    /// Plain `Add` / `Sub` / `Mul`: the checked backend operation
    /// with its overflow flag unused — a plain operator's result
    /// wraps, and the checked result is already normalised to the
    /// type by the backend, so no further fixup is needed.
    fn lower_plain_arith(
        &mut self,
        dst: u32,
        base: CheckedOp,
        lv: u32,
        rv: u32,
        kind: IrIntKind,
    ) -> Result<(), JitError> {
        let flag = self.fresh();
        self.emit(Instr::CheckedBinOp {
            dst_val: dst,
            dst_flag: flag,
            op: base,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(rv),
            kind,
        });
        Ok(())
    }

    /// Plain division / remainder with the guards the semantics
    /// (and the hardware) need: a zero divisor panics with the
    /// divide-by-zero message, and the signed 64-bit `MIN / -1`
    /// case — which wraps, to `MIN` for division and `0` for
    /// remainder, but traps the hardware instruction — is computed
    /// through a rewritten divisor (see below). Narrower signed
    /// kinds cannot trap: their values divide exactly in 64 bits
    /// and the result is normalised to the type.
    fn lower_div_rem(
        &mut self,
        dst: u32,
        is_rem: bool,
        lv: u32,
        rv: u32,
        kind: IrIntKind,
        dst_ty: &mir::Ty,
    ) -> Result<(), JitError> {
        let msg = self.messages.intern(if is_rem {
            "attempt to calculate the remainder with a divisor of zero"
        } else {
            "attempt to divide by zero"
        });
        let pblk = self.panic_block(msg);
        let nonzero = self.fresh();
        self.emit(Instr::BinOp {
            dst: nonzero,
            op: IrBinOp::Ne,
            lhs: IrOperand::Value(rv),
            rhs: IrOperand::Const(0),
        });
        let cont = self.new_block();
        self.seal_and_continue(
            IrTerminator::Branch {
                cond: IrOperand::Value(nonzero),
                then_blk: cont,
                else_blk: pblk,
            },
            cont,
        );
        // A divisor of -1 in the signed 64-bit case: rewrite it
        // to 1 (whose quotient is the dividend itself and whose
        // remainder is 0) so the instruction cannot trap, and
        // negate a division's result back below — wrapping
        // negation, which maps `MIN` to itself and every other
        // dividend to its true quotient by -1.
        let mut divisor = rv;
        let mut neg1: Option<u32> = None;
        if kind.signed && kind.bits == 64 {
            let is_neg1 = self.fresh();
            self.emit(Instr::BinOp {
                dst: is_neg1,
                op: IrBinOp::Eq,
                lhs: IrOperand::Value(rv),
                rhs: IrOperand::Const(-1),
            });
            let twice = self.fresh();
            self.emit(Instr::BinOp {
                dst: twice,
                op: IrBinOp::Add,
                lhs: IrOperand::Value(is_neg1),
                rhs: IrOperand::Value(is_neg1),
            });
            let adjusted = self.fresh();
            self.emit(Instr::BinOp {
                dst: adjusted,
                op: IrBinOp::Add,
                lhs: IrOperand::Value(rv),
                rhs: IrOperand::Value(twice),
            });
            divisor = adjusted;
            neg1 = Some(is_neg1);
        }
        let ir_op = match (is_rem, kind.signed) {
            (false, true) => IrBinOp::Div,
            (false, false) => IrBinOp::DivU,
            (true, true) => IrBinOp::Rem,
            (true, false) => IrBinOp::RemU,
        };
        self.emit(Instr::BinOp {
            dst,
            op: ir_op,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(divisor),
        });
        if let (Some(is_neg1), false) = (neg1, is_rem) {
            let twice = self.fresh();
            self.emit(Instr::BinOp {
                dst: twice,
                op: IrBinOp::Add,
                lhs: IrOperand::Value(is_neg1),
                rhs: IrOperand::Value(is_neg1),
            });
            let factor = self.fresh();
            self.emit(Instr::BinOp {
                dst: factor,
                op: IrBinOp::Sub,
                lhs: IrOperand::Const(1),
                rhs: IrOperand::Value(twice),
            });
            let wrapped = self.fresh();
            self.emit(Instr::BinOp {
                dst: wrapped,
                op: IrBinOp::Mul,
                lhs: IrOperand::Value(dst),
                rhs: IrOperand::Value(factor),
            });
            self.emit(Instr::Copy {
                dst,
                src: IrOperand::Value(wrapped),
            });
        }
        self.normalize(dst, dst_ty);
        Ok(())
    }

    /// Plain shift: the amount is taken modulo the type width, as
    /// the interpreter's plain form does (the overshift panic
    /// belongs to the `Assert` after a checked shift). Bits shifted
    /// out of the width are simply lost, as natively.
    fn lower_shift(
        &mut self,
        dst: u32,
        is_shl: bool,
        lv: u32,
        rv: u32,
        kind: IrIntKind,
        dst_ty: &mir::Ty,
    ) -> Result<(), JitError> {
        let bits = kind.bits as u32;
        let masked = self.fresh();
        self.emit(Instr::BinOp {
            dst: masked,
            op: IrBinOp::And,
            lhs: IrOperand::Value(rv),
            rhs: IrOperand::Const((bits - 1) as i64),
        });
        let shift_op = if is_shl {
            IrBinOp::Shl
        } else if kind.signed {
            IrBinOp::Sar
        } else {
            IrBinOp::Shr
        };
        self.emit(Instr::BinOp {
            dst,
            op: shift_op,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(masked),
        });
        self.normalize(dst, dst_ty);
        Ok(())
    }

    /// Checked shift (`CheckedBinaryOp` with `Shl` / `Shr`): the
    /// value wraps (the amount is taken modulo the width) and the
    /// flag reports an overshift only; bits shifted out are lost
    /// and do not set the flag, as natively.
    fn lower_checked_shift_to(
        &mut self,
        dst: u32,
        dst_ty: &mir::Ty,
        is_shl: bool,
        lhs: &mir::Operand,
        rhs: &mir::Operand,
    ) -> Result<(), JitError> {
        let mir::Ty::Tuple(fields) = dst_ty else {
            return Err(self.err("checked operation result is not a (value, bool) tuple"));
        };
        if fields.len() != 2 {
            return Err(self.err("checked operation result is not a (value, bool) tuple"));
        }
        let val_ty = fields[0].clone();
        let Some(kind) = int_kind(&val_ty) else {
            return Err(self.err("checked operation on a non-integer type"));
        };
        let bits = kind.bits as u32;
        let lv = self.operand_value(lhs, &val_ty)?;
        let rv = self.operand_value(rhs, &val_ty)?;
        let over = self.fresh();
        self.emit(Instr::BinOp {
            dst: over,
            op: IrBinOp::GeU,
            lhs: IrOperand::Value(rv),
            rhs: IrOperand::Const(bits as i64),
        });
        let masked = self.fresh();
        self.emit(Instr::BinOp {
            dst: masked,
            op: IrBinOp::And,
            lhs: IrOperand::Value(rv),
            rhs: IrOperand::Const((bits - 1) as i64),
        });
        let shift_op = if is_shl {
            IrBinOp::Shl
        } else if kind.signed {
            IrBinOp::Sar
        } else {
            IrBinOp::Shr
        };
        self.emit(Instr::BinOp {
            dst,
            op: shift_op,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(masked),
        });
        self.normalize(dst, &val_ty);
        self.emit(Instr::Copy {
            dst: dst + 1,
            src: IrOperand::Value(over),
        });
        Ok(())
    }

    /// Checked shift on a 128-bit value (`CheckedBinaryOp` with
    /// `Shl` / `Shr`): the value wraps (the amount taken modulo
    /// 128) into the pair at `dst`, and the flag at `dst + 2`
    /// reports an overshift only — the 128-bit form of
    /// [`Self::lower_checked_shift_to`].
    fn lower_checked_shift128_to(
        &mut self,
        dst: u32,
        is_shl: bool,
        lhs: &mir::Operand,
        rhs: &mir::Operand,
        val_ty: &mir::Ty,
    ) -> Result<(), JitError> {
        let signed = int128_signed(val_ty).expect("128-bit value type");
        let l = self.operand_leaves(lhs, val_ty, 2)?;
        let amount = self.operand_value(rhs, val_ty)?;
        let over = self.fresh();
        self.emit(Instr::BinOp {
            dst: over,
            op: IrBinOp::GeU,
            lhs: IrOperand::Value(amount),
            rhs: IrOperand::Const(128),
        });
        let masked = self.fresh();
        self.emit(Instr::BinOp {
            dst: masked,
            op: IrBinOp::And,
            lhs: IrOperand::Value(amount),
            rhs: IrOperand::Const(127),
        });
        let shift_op = if is_shl {
            IrShiftOp128::Shl
        } else if signed {
            IrShiftOp128::Sar
        } else {
            IrShiftOp128::Shr
        };
        self.emit(Instr::Shift128 {
            dst,
            dst2: dst + 1,
            op: shift_op,
            lo: IrOperand::Value(l[0]),
            hi: IrOperand::Value(l[1]),
            amount: IrOperand::Value(masked),
        });
        self.emit(Instr::Copy {
            dst: dst + 2,
            src: IrOperand::Value(over),
        });
        Ok(())
    }

    pub(super) fn lower_unop_to(
        &mut self,
        dst: u32,
        dst_ty: &mir::Ty,
        op: mir::UnOp,
        operand: &mir::Operand,
    ) -> Result<(), JitError> {
        // A 128-bit operand is a word pair: negation is the
        // pair's subtraction from zero, `Not` flips each word.
        if int128_signed(dst_ty).is_some() {
            match op {
                mir::UnOp::Neg => {
                    let l = self.operand_leaves(operand, dst_ty, 2)?;
                    self.emit(Instr::BinOp128 {
                        dst,
                        dst2: dst + 1,
                        op: IrBinOp128::Sub,
                        lhs_lo: IrOperand::Const(0),
                        lhs_hi: IrOperand::Const(0),
                        rhs_lo: IrOperand::Value(l[0]),
                        rhs_hi: IrOperand::Value(l[1]),
                    });
                    return Ok(());
                }
                mir::UnOp::Not => {
                    let l = self.operand_leaves(operand, dst_ty, 2)?;
                    for k in 0..2u32 {
                        self.emit(Instr::UnOp {
                            dst: dst + k,
                            op: IrUnOp::Not,
                            src: IrOperand::Value(l[k as usize]),
                        });
                    }
                    return Ok(());
                }
                mir::UnOp::PtrMetadata => {}
            }
        }
        match op {
            mir::UnOp::Neg => {
                if let Some(fk) = float_kind(dst_ty) {
                    // Float negation is an exact sign-bit flip (it
                    // applies to NaNs and zeroes alike), so it lowers
                    // to an integer xor on the value's bit pattern:
                    // the sign bit for f64, bit 31 for f32 (whose
                    // canonical form keeps the upper half zero).
                    let v = self.operand_value(operand, dst_ty)?;
                    let mask = match fk {
                        IrFloatKind::F64 => i64::MIN,
                        IrFloatKind::F32 => 0x8000_0000,
                    };
                    self.emit(Instr::BinOp {
                        dst,
                        op: IrBinOp::Xor,
                        lhs: IrOperand::Value(v),
                        rhs: IrOperand::Const(mask),
                    });
                    return Ok(());
                }
                let mir::Ty::Int(it) = dst_ty else {
                    return Err(self.err("Neg on a non-signed-integer type"));
                };
                let v = self.operand_value(operand, dst_ty)?;
                // Plain negation wraps: the type's minimum maps
                // to itself (the overflow panic belongs to the
                // `Assert` after a checked negation).
                self.emit(Instr::UnOp {
                    dst,
                    op: IrUnOp::Neg,
                    src: IrOperand::Value(v),
                });
                self.normalize(dst, dst_ty);
                Ok(())
            }
            mir::UnOp::Not => {
                let v = self.operand_value(operand, dst_ty)?;
                if matches!(dst_ty, mir::Ty::Bool) {
                    // Logical not on a canonical 0/1 value.
                    self.emit(Instr::BinOp {
                        dst,
                        op: IrBinOp::Xor,
                        lhs: IrOperand::Value(v),
                        rhs: IrOperand::Const(1),
                    });
                } else {
                    self.emit(Instr::UnOp {
                        dst,
                        op: IrUnOp::Not,
                        src: IrOperand::Value(v),
                    });
                    self.normalize(dst, dst_ty);
                }
                Ok(())
            }
            mir::UnOp::PtrMetadata => {
                // The metadata of a fat pointer value is its length
                // word — the interpreter's `PtrMetadata` on a
                // `FatRef`. A thin operand has no metadata and is
                // rejected by `operand_pair`.
                let (_, len) = self.operand_pair(operand)?;
                self.emit(Instr::Copy {
                    dst,
                    src: IrOperand::Value(len),
                });
                Ok(())
            }
        }
    }

    pub(super) fn lower_cast_to(
        &mut self,
        dst: u32,
        kind: &mir::CastKind,
        operand: &mir::Operand,
        target: &mir::Ty,
    ) -> Result<(), JitError> {
        match kind {
            mir::CastKind::IntToInt => {
                // The source type comes from the operand's place,
                // or from the literal itself for a constant operand.
                let src_ty = self
                    .operand_place_ty(operand)
                    .or_else(|| literal_int_ty(operand))
                    .unwrap_or_else(|| target.clone());
                // The 128-bit forms move the value's word pair: a
                // same-width cast is the pair itself, a narrower
                // target is the pair's low word normalised to it,
                // and a wider (128-bit) target is the source's
                // canonical word plus its extension word — the
                // sign extension for a signed source, zero for an
                // unsigned one.
                if int128_signed(target).is_some() {
                    if int128_signed(&src_ty).is_some() {
                        let l = self.operand_leaves(operand, &src_ty, 2)?;
                        for k in 0..2u32 {
                            self.emit(Instr::Copy {
                                dst: dst + k,
                                src: IrOperand::Value(l[k as usize]),
                            });
                        }
                        return Ok(());
                    }
                    let Some(src_kind) = int_kind(&src_ty) else {
                        return Err(self.err(&format!(
                            "cast from type {}",
                            src_ty.kind_name()
                        )));
                    };
                    let v = self.operand_value(operand, &src_ty)?;
                    self.emit(Instr::Copy {
                        dst,
                        src: IrOperand::Value(v),
                    });
                    self.normalize(dst, &src_ty);
                    if src_kind.signed {
                        self.emit(Instr::BinOp {
                            dst: dst + 1,
                            op: IrBinOp::Sar,
                            lhs: IrOperand::Value(dst),
                            rhs: IrOperand::Const(63),
                        });
                    } else {
                        self.emit(Instr::Const {
                            dst: dst + 1,
                            imm: 0,
                        });
                    }
                    return Ok(());
                }
                if int128_signed(&src_ty).is_some() {
                    if int_kind(target).is_none() {
                        return Err(self.err(&format!(
                            "cast to type {}",
                            target.kind_name()
                        )));
                    }
                    let l = self.operand_leaves(operand, &src_ty, 2)?;
                    self.emit(Instr::Copy {
                        dst,
                        src: IrOperand::Value(l[0]),
                    });
                    self.normalize(dst, target);
                    return Ok(());
                }
                if int_kind(target).is_none() {
                    return Err(self.err(&format!(
                        "cast to type {}",
                        target.kind_name()
                    )));
                }
                // The source value is canonical for its own type, so
                // casting is exactly normalising that 64-bit value to
                // the target type.
                let v = self.operand_value(operand, &src_ty)?;
                self.emit(Instr::Copy {
                    dst,
                    src: IrOperand::Value(v),
                });
                self.normalize(dst, target);
                Ok(())
            }
            mir::CastKind::FloatToInt => {
                let Some(src_ty) = self.operand_float_ty(operand) else {
                    return Err(self.err("FloatToInt cast with an unknown source type"));
                };
                let fk = float_kind(&src_ty).expect("operand_float_ty yields a float kind");
                if let Some(signed) = int128_signed(target) {
                    let v = self.operand_value(operand, &src_ty)?;
                    self.emit(Instr::FloatToInt128 {
                        dst,
                        dst2: dst + 1,
                        kind: fk,
                        src: IrOperand::Value(v),
                        signed,
                    });
                    return Ok(());
                }
                let Some(int) = int_kind(target) else {
                    return Err(self.err(&format!(
                        "FloatToInt cast to type {}",
                        target.kind_name()
                    )));
                };
                let v = self.operand_value(operand, &src_ty)?;
                self.emit(Instr::FloatToInt {
                    dst,
                    kind: fk,
                    src: IrOperand::Value(v),
                    int,
                });
                Ok(())
            }
            mir::CastKind::IntToFloat => {
                let Some(fk) = float_kind(target) else {
                    return Err(self.err(&format!(
                        "IntToFloat cast to type {}",
                        target.kind_name()
                    )));
                };
                // The source type comes from the operand's place, or
                // from the literal itself for a constant operand.
                let Some(src_ty) = self
                    .operand_place_ty(operand)
                    .or_else(|| literal_int_ty(operand))
                else {
                    return Err(self.err("IntToFloat cast with an unknown source type"));
                };
                if let Some(signed) = int128_signed(&src_ty) {
                    let l = self.operand_leaves(operand, &src_ty, 2)?;
                    self.emit(Instr::Int128ToFloat {
                        dst,
                        kind: fk,
                        lo: IrOperand::Value(l[0]),
                        hi: IrOperand::Value(l[1]),
                        signed,
                    });
                    return Ok(());
                }
                let Some(int) = int_kind(&src_ty) else {
                    return Err(self.err("IntToFloat cast from a non-integer type"));
                };
                let v = self.operand_value(operand, &src_ty)?;
                self.emit(Instr::IntToFloat {
                    dst,
                    kind: fk,
                    src: IrOperand::Value(v),
                    int,
                });
                Ok(())
            }
            mir::CastKind::FloatToFloat => {
                let Some(src_ty) = self.operand_float_ty(operand) else {
                    return Err(self.err("FloatToFloat cast with an unknown source type"));
                };
                let src_kind = float_kind(&src_ty).expect("operand_float_ty yields a float kind");
                let Some(fk) = float_kind(target) else {
                    return Err(self.err(&format!(
                        "FloatToFloat cast to type {}",
                        target.kind_name()
                    )));
                };
                let v = self.operand_value(operand, &src_ty)?;
                if src_kind == fk {
                    self.emit(Instr::Copy {
                        dst,
                        src: IrOperand::Value(v),
                    });
                } else {
                    self.emit(Instr::FloatToFloat {
                        dst,
                        kind: fk,
                        src: IrOperand::Value(v),
                        src_kind,
                    });
                }
                Ok(())
            }
            // Pointer casts that keep the address word unchanged:
            // the interpreter converts exactly these to a plain
            // `Use`, and a JIT pointer *is* the address word, so
            // the cast is a copy. Unsize coercions would have to
            // build a fat pointer and stay unsupported, as do the
            // provenance casts, which the interpreter also rejects.
            mir::CastKind::PtrToPtr
            | mir::CastKind::PointerCoercion(
                mir::PointerCoercion::MutToConstPointer
                | mir::PointerCoercion::ArrayToPointer,
            ) => {
                let src_ty = self
                    .operand_place_ty(operand)
                    .unwrap_or_else(|| target.clone());
                // A cast from a fat pointer to a thin one (a
                // `*const str` read as a `*const u8`) keeps the
                // data word and drops the metadata; every other
                // shape here is one word on both sides.
                let v = if flatten(&src_ty, &self.func.name, self.layouts)? == 2
                    && flatten(target, &self.func.name, self.layouts)? == 1
                {
                    self.operand_leaves(operand, &src_ty, 2)?[0]
                } else {
                    self.operand_value(operand, &src_ty)?
                };
                self.emit(Instr::Copy {
                    dst,
                    src: IrOperand::Value(v),
                });
                Ok(())
            }
            mir::CastKind::Transmute => {
                // A transmute reinterprets the value's own
                // words: between two types of the same leaf
                // count, the canonical words are already the
                // bit pattern both sides read (a pointer
                // wrapper and the raw pointer, an integer and
                // the float of the same width, a 128-bit integer
                // and its word pair). A transmute across leaf
                // counts would need byte-level repacking, which
                // the word model does not do.
                let src_ty = self
                    .operand_place_ty(operand)
                    .unwrap_or_else(|| target.clone());
                let src_count = flatten(&src_ty, &self.func.name, self.layouts)?;
                let dst_count = flatten(target, &self.func.name, self.layouts)?;
                if src_count != dst_count || src_count == 0 {
                    return Err(self.err(&format!(
                        "transmute between {src_count}-leaf and {dst_count}-leaf types"
                    )));
                }
                let leaves = self.operand_leaves(operand, &src_ty, src_count)?;
                for (k, leaf) in leaves.iter().enumerate() {
                    self.emit(Instr::Copy {
                        dst: dst + k as u32,
                        src: IrOperand::Value(*leaf),
                    });
                }
                Ok(())
            }
            other => Err(self.err(&format!("cast {other:?}"))),
        }
    }

    /// Store an `AggregateKind::RawPtr` value's words at
    /// `base`: the data pointer, then the metadata word for an
    /// unsized pointee (a slice / `str` length). `count` is the
    /// destination's leaf count (1 thin, 2 fat); a two-operand
    /// aggregate with a one-word destination carries unit
    /// metadata, which is no word (the interpreter's conversion
    /// decides the same shapes the same way). A `dyn` pointee's
    /// metadata is a vtable pointer and stays unsupported.
    pub(super) fn store_raw_ptr_aggregate(
        &mut self,
        data_ty: &mir::Ty,
        ops: &[mir::Operand],
        base: u32,
        count: usize,
    ) -> Result<(), JitError> {
        if matches!(data_ty, mir::Ty::Dynamic(..)) {
            return Err(self.err("raw pointer aggregate with dyn metadata"));
        }
        // The data operand is a thin pointer to the pointee's
        // first element (to the element type for a slice, to
        // `u8` for `str`), whatever the aggregate's pointee.
        let data_ctx = mir::Ty::RawPtr(
            Box::new(match data_ty {
                mir::Ty::Slice(elem) => (**elem).clone(),
                mir::Ty::Str => mir::Ty::Uint(mir::UintTy::U8),
                other => other.clone(),
            }),
            mir::Mutability::Not,
        );
        match (ops.len(), count) {
            (1, 1) | (2, 1) => self.store_operand_leaves(&ops[0], &data_ctx, base),
            (2, 2) => {
                self.store_operand_leaves(&ops[0], &data_ctx, base)?;
                self.store_operand_leaves(
                    &ops[1],
                    &mir::Ty::Uint(mir::UintTy::Usize),
                    base + 1,
                )
            }
            _ => Err(self.err("raw pointer aggregate shape mismatch")),
        }
    }

    pub(super) fn lower_aggregate(
        &mut self,
        place: &mir::Place,
        kind: &mir::AggregateKind,
        ops: &[mir::Operand],
    ) -> Result<(), JitError> {
        // A structural `fmt::Arguments` aggregate (core's own
        // constructors build the value from its two fields
        // instead of calling `Arguments::new`): the builtin
        // layer's representation of the type is the one-word
        // store handle, so the two field words go to the host
        // function that interprets the structural encoding and
        // produces the handle.
        if let mir::AggregateKind::Adt { def, .. } = kind {
            if builtins::canonical_name(&self.callee_name(*def))
                == "std::fmt::Arguments"
            {
                let (dst, count, _ty) = self.place_leaves(place)?;
                if count != 1 || ops.len() != 2 {
                    return Err(
                        self.err("structural fmt::Arguments aggregate shape mismatch")
                    );
                }
                let word_ty = mir::Ty::RawPtr(
                    Box::new(mir::Ty::Uint(mir::UintTy::U8)),
                    mir::Mutability::Not,
                );
                let f0 = self.operand_value(&ops[0], &word_ty)?;
                let f1 = self.operand_value(&ops[1], &word_ty)?;
                self.emit_host_call(
                    builtins::HOST_ARGUMENTS_FROM_STRUCT,
                    dst,
                    None,
                    vec![IrOperand::Value(f0), IrOperand::Value(f1)],
                );
                self.emit(Instr::CheckPanic);
                return Ok(());
            }
        }
        match kind {
            mir::AggregateKind::Tuple => {}
            mir::AggregateKind::Adt {
                def,
                variant_index,
                field_index,
                ..
            } => {
                if field_index.is_some() {
                    // A union aggregate writes one active field;
                    // unions are outside the layout model.
                    return Err(self.err("ADT aggregate with an active-field index (union)"));
                }
                let (dst, count, dst_ty) = self.place_leaves(place)?;
                let layout = self.adt_layout(&dst_ty)?;
                if count != layout.total {
                    return Err(self.err("ADT aggregate destination shape mismatch"));
                }
                let slots = layout
                    .variants
                    .get(*variant_index as usize)
                    .ok_or_else(|| self.err("ADT aggregate variant outside the layout"))?
                    .clone();
                if slots.len() != ops.len() {
                    return Err(self.err("ADT aggregate field count mismatch"));
                }
                if layout.tag {
                    // The value's first leaf is the variant's
                    // discriminant value from the table.
                    let tag = self.discriminant_value(*def, *variant_index)?;
                    self.emit(Instr::Const { dst, imm: tag });
                }
                for (op, slot) in ops.iter().zip(slots.iter()) {
                    self.store_operand_leaves(op, &slot.ty, dst + slot.offset as u32)?;
                }
                return Ok(());
            }
            mir::AggregateKind::Array(_) => {
                let (dst, _count, dst_ty) = self.place_leaves(place)?;
                let (elem, n) = match &dst_ty {
                    mir::Ty::Array(elem, mir::ArrayLen::Known(n)) => {
                        ((**elem).clone(), *n as usize)
                    }
                    _ => {
                        return Err(
                            self.err("array aggregate with a non-array place type")
                        )
                    }
                };
                if ops.len() != n {
                    return Err(self.err("array aggregate arity mismatch"));
                }
                let per = flatten(&elem, &self.func.name, self.layouts)?;
                for (i, op) in ops.iter().enumerate() {
                    self.store_operand_leaves(op, &elem, dst + (i * per) as u32)?;
                }
                return Ok(());
            }
            mir::AggregateKind::RawPtr { data_ty, .. } => {
                let (dst, count, _dst_ty) = self.place_leaves(place)?;
                self.store_raw_ptr_aggregate(data_ty, ops, dst, count)?;
                return Ok(());
            }
            other => {
                return Err(self.err(&format!("aggregate {}", other.kind_name())));
            }
        }
        // Only the Tuple kind reaches here (the other kinds
        // return from the match above): each field is stored at
        // its leaf offset under the destination tuple's shape —
        // a field may itself be multi-leaf (a nested tuple, a
        // fat pointer, a wide call result).
        let (dst, _count, dst_ty) = self.place_leaves(place)?;
        let mir::Ty::Tuple(tys) = dst_ty else {
            return Err(self.err("tuple aggregate destination is not a tuple"));
        };
        if tys.len() != ops.len() {
            return Err(self.err("tuple aggregate arity mismatch"));
        }
        let mut off = 0usize;
        for (op, ty) in ops.iter().zip(tys.iter()) {
            self.store_operand_leaves(op, ty, dst + off as u32)?;
            off += flatten(ty, &self.func.name, self.layouts)?;
        }
        Ok(())
    }

    // ----- terminators -------------------------------------------------------

}
