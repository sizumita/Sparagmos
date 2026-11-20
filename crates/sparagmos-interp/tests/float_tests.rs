//! Float support tests: hand-built `sparagmos-mir` bodies using
//! `f32` / `f64` locals, constants, arithmetic and casts, executed
//! by the interpreter and compared against host-computed values.
//!
//! Semantics pinned here (Rust 1.100.0):
//! * `f32` operations round each result to `f32` (the interpreter
//!   computes in `f64`, which is exact for + - * / % by the
//!   double-rounding theorem).
//! * Float division by zero yields an infinity and `0.0 / 0.0`
//!   yields NaN — values, not guest panics.
//! * Float-to-int casts saturate at the target range; NaN becomes 0.
//! * Int-to-float casts convert at the target width directly.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::Value;
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir (mirroring mir_convert_tests) -----

fn f64_ty() -> mir::Ty {
    mir::Ty::Float(mir::FloatTy::F64)
}
fn f32_ty() -> mir::Ty {
    mir::Ty::Float(mir::FloatTy::F32)
}
fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}
fn u32_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U32)
}
fn bool_ty() -> mir::Ty {
    mir::Ty::Bool
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}

fn move_(idx: u32) -> mir::Operand {
    mir::Operand::Move(place(idx))
}

fn const_f64(v: f64) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Float {
            bits: v.to_bits() as u128,
            ty: mir::FloatTy::F64,
        },
    )))
}

fn const_f32(v: f32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Float {
            bits: v.to_bits() as u128,
            ty: mir::FloatTy::F32,
        },
    )))
}

fn const_i32(v: i32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u32 as u128,
            size: 4,
        },
    )))
}

fn block(stmts: Vec<mir::Statement>, kind: mir::TerminatorKind) -> mir::BasicBlockData {
    let mut bb = mir::BasicBlockData::new(mir::Terminator::new(kind));
    bb.statements = stmts;
    bb
}

fn assign(place: mir::Place, rv: mir::Rvalue) -> mir::Statement {
    mir::Statement::assign(place, rv)
}

fn binop(op: mir::BinOp, lhs: mir::Operand, rhs: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::BinaryOp(op, Box::new((lhs, rhs)))
}

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

/// A no-argument function whose single block is `stmts` then `return`.
fn run(name: &str, tys: &[mir::Ty], stmts: Vec<mir::Statement>) -> Value {
    let body = mir::Body::new(decls(tys), 0, vec![block(stmts, ret())]);
    let mut program = MirProgram::new();
    program.add(name, body);
    interpret_mir(&program, name, Vec::new()).unwrap_or_else(|e| panic!("{name} must run: {e}"))
}

// ----- arithmetic ------------------------------------------------------------

#[test]
fn f64_add() {
    let v = run(
        "f64_add",
        &[f64_ty()],
        vec![assign(
            place(0),
            binop(mir::BinOp::Add, const_f64(1.5), const_f64(2.25)),
        )],
    );
    assert_eq!(v, Value::Float(3.75));
}

#[test]
fn f32_div_rounds_to_f32() {
    let v = run(
        "f32_div",
        &[f32_ty()],
        vec![assign(
            place(0),
            binop(mir::BinOp::Div, const_f32(1.0), const_f32(3.0)),
        )],
    );
    assert_eq!(v, Value::Float((1.0f32 / 3.0f32) as f64));
}

#[test]
fn f64_local_copy_and_mul() {
    let v = run(
        "f64_mul",
        &[f64_ty(), f64_ty()],
        vec![
            assign(place(1), mir::Rvalue::Use(const_f64(2.5))),
            assign(place(0), binop(mir::BinOp::Mul, move_(1), const_f64(2.0))),
        ],
    );
    assert_eq!(v, Value::Float(5.0));
}

#[test]
fn f64_neg() {
    let v = run(
        "f64_neg",
        &[f64_ty()],
        vec![assign(
            place(0),
            mir::Rvalue::UnaryOp(mir::UnOp::Neg, const_f64(1.5)),
        )],
    );
    assert_eq!(v, Value::Float(-1.5));
}

#[test]
fn f64_div_by_zero_is_infinity_not_a_panic() {
    let v = run(
        "f64_div_zero",
        &[f64_ty()],
        vec![assign(
            place(0),
            binop(mir::BinOp::Div, const_f64(1.0), const_f64(0.0)),
        )],
    );
    match v {
        Value::Float(x) => assert!(x.is_infinite() && x.is_sign_positive(), "got {x}"),
        other => panic!("expected a float infinity, got {other}"),
    }
}

#[test]
fn f64_zero_over_zero_is_nan_and_nan_compares_unequal() {
    let v = run(
        "f64_nan",
        &[f64_ty()],
        vec![assign(
            place(0),
            binop(mir::BinOp::Div, const_f64(0.0), const_f64(0.0)),
        )],
    );
    match v {
        Value::Float(x) => assert!(x.is_nan(), "got {x}"),
        other => panic!("expected a float NaN, got {other}"),
    }
    let eq = run(
        "f64_nan_eq",
        &[bool_ty()],
        vec![assign(
            place(0),
            binop(mir::BinOp::Eq, const_f64(f64::NAN), const_f64(f64::NAN)),
        )],
    );
    assert_eq!(eq, Value::Bool(false));
}

#[test]
fn f64_comparison() {
    let v = run(
        "f64_gt",
        &[bool_ty()],
        vec![assign(
            place(0),
            binop(mir::BinOp::Gt, const_f64(2.5), const_f64(2.25)),
        )],
    );
    assert_eq!(v, Value::Bool(true));
}

// ----- casts -----------------------------------------------------------------

#[test]
fn float_to_int_cast_saturates() {
    let hi = run(
        "f2i_hi",
        &[i32_ty(), f64_ty()],
        vec![
            assign(place(1), mir::Rvalue::Use(const_f64(3.0e9))),
            assign(
                place(0),
                mir::Rvalue::Cast(mir::CastKind::FloatToInt, move_(1), i32_ty()),
            ),
        ],
    );
    assert_eq!(hi, Value::Int(i32::MAX as i128));
    let lo = run(
        "f2i_lo",
        &[i32_ty(), f64_ty()],
        vec![
            assign(place(1), mir::Rvalue::Use(const_f64(-3.0e9))),
            assign(
                place(0),
                mir::Rvalue::Cast(mir::CastKind::FloatToInt, move_(1), i32_ty()),
            ),
        ],
    );
    assert_eq!(lo, Value::Int(i32::MIN as i128));
}

#[test]
fn float_nan_to_uint_cast_is_zero() {
    let v = run(
        "nan2u",
        &[u32_ty(), f64_ty()],
        vec![
            assign(place(1), mir::Rvalue::Use(const_f64(f64::NAN))),
            assign(
                place(0),
                mir::Rvalue::Cast(mir::CastKind::FloatToInt, move_(1), u32_ty()),
            ),
        ],
    );
    assert_eq!(v, Value::Uint(0));
}

#[test]
fn int_to_float_cast() {
    let v = run(
        "i2f",
        &[f64_ty(), i32_ty()],
        vec![
            assign(place(1), mir::Rvalue::Use(const_i32(42))),
            assign(
                place(0),
                mir::Rvalue::Cast(mir::CastKind::IntToFloat, move_(1), f64_ty()),
            ),
        ],
    );
    assert_eq!(v, Value::Float(42.0));
}

#[test]
fn float_to_float_cast_narrows() {
    let v = run(
        "f2f",
        &[f32_ty(), f64_ty()],
        vec![
            assign(place(1), mir::Rvalue::Use(const_f64(0.1))),
            assign(
                place(0),
                mir::Rvalue::Cast(mir::CastKind::FloatToFloat, move_(1), f32_ty()),
            ),
        ],
    );
    assert_eq!(v, Value::Float(0.1f32 as f64));
}
