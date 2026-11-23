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

fn const_f16_bits(bits: u16) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Float {
            bits: bits as u128,
            ty: mir::FloatTy::F16,
        },
    )))
}

#[test]
fn f16_add_rounds_to_f16() {
    // 1.0 + 2^-11 is exactly halfway between 1.0 and the next f16
    // (spacing 2^-10): round-to-nearest-even lands on 1.0.
    let v = run(
        "f16_add_tie",
        &[mir::Ty::Float(mir::FloatTy::F16)],
        vec![assign(
            place(0),
            binop(
                mir::BinOp::Add,
                const_f16_bits(0x3c00),
                const_f16_bits(0x1000),
            ),
        )],
    );
    assert_eq!(v, Value::Float(1.0));
    // 1.0 + 2^-10 is exactly representable.
    let v = run(
        "f16_add_exact",
        &[mir::Ty::Float(mir::FloatTy::F16)],
        vec![assign(
            place(0),
            binop(
                mir::BinOp::Add,
                const_f16_bits(0x3c00),
                const_f16_bits(0x1400),
            ),
        )],
    );
    assert_eq!(v, Value::Float(1.0009765625));
}

#[test]
fn f16_constant_widens_exactly() {
    // 0x3e00 = 1.5, 0xc000 = -2.0, 0x0001 = 2^-24 (smallest
    // subnormal), 0x7c00 = +inf: every f16 value is exact in f64.
    for (bits, want) in [
        (0x3e00u16, 1.5f64),
        (0xc000, -2.0),
        (0x0001, 2f64.powi(-24)),
        (0x3555, 0.333251953125),
        (0x7c00, f64::INFINITY),
    ] {
        let v = run(
            "f16_const",
            &[mir::Ty::Float(mir::FloatTy::F16)],
            vec![assign(
                place(0),
                mir::Rvalue::Use(const_f16_bits(bits)),
            )],
        );
        assert_eq!(v, Value::Float(want), "f16 bits {bits:#06x}");
    }
    let v = run(
        "f16_nan",
        &[mir::Ty::Float(mir::FloatTy::F16)],
        vec![assign(
            place(0),
            mir::Rvalue::Use(const_f16_bits(0x7e00)),
        )],
    );
    assert!(matches!(v, Value::Float(x) if x.is_nan()));
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
// ----- projected cast sources (PR #419 Codex follow-up) -----

fn f16_ty() -> mir::Ty {
    mir::Ty::Float(mir::FloatTy::F16)
}

fn u8_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U8)
}

/// Bit-exact float comparison: `Value`'s derived equality is
/// `f64` equality, under which a NaN never equals itself, so
/// NaN-carrying cast results are judged by their bit pattern.
fn assert_float_bits(v: Value, bits: u64) {
    match v {
        Value::Float(f) => assert_eq!(f.to_bits(), bits),
        other => panic!("expected a Float value, got {other:?}"),
    }
}

fn const_u8(v: u8) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
            size: 1,
        },
    )))
}

#[test]
fn cast_tuple_field_f64_to_f16_quiets_snan() {
    // The cast reads a tuple field: the source is a genuine `f64`
    // narrowing, so the signaling NaN is quieted with its payload
    // carried across (native `f64` sNaN -> `f16` = 0x7e01), not
    // transported bit-exactly (0x7c01).
    let tuple_ty = mir::Ty::Tuple(vec![f64_ty(), u8_ty()]);
    let v = run(
        "cast_tuple_field_f64_f16",
        &[f16_ty(), tuple_ty],
        vec![
            assign(
                place(1),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Tuple),
                    vec![
                        const_f64(f64::from_bits(0x7ff0_0400_0000_0000)),
                        const_u8(7),
                    ],
                ),
            ),
            assign(
                place(0),
                mir::Rvalue::Cast(
                    mir::CastKind::FloatToFloat,
                    mir::Operand::Copy(place(1).field(0, f64_ty())),
                    f16_ty(),
                ),
            ),
        ],
    );
    assert_float_bits(v, 0x7ff8_0400_0000_0000);
}

#[test]
fn cast_array_elem_f64_to_f16_quiets_snan() {
    // Same narrowing through an array element read by constant
    // index.
    let array_ty = mir::Ty::Array(Box::new(f64_ty()), mir::ArrayLen::Known(2));
    let elem = mir::Place {
        local: mir::Local(1),
        projection: vec![mir::PlaceElem::ConstantIndex {
            offset: 0,
            min_length: 2,
            from_end: false,
        }],
    };
    let v = run(
        "cast_array_elem_f64_f16",
        &[f16_ty(), array_ty],
        vec![
            assign(
                place(1),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Array(f64_ty())),
                    vec![
                        const_f64(f64::from_bits(0x7ff0_0400_0000_0000)),
                        const_f64(0.0),
                    ],
                ),
            ),
            assign(
                place(0),
                mir::Rvalue::Cast(
                    mir::CastKind::FloatToFloat,
                    mir::Operand::Copy(elem),
                    f16_ty(),
                ),
            ),
        ],
    );
    assert_float_bits(v, 0x7ff8_0400_0000_0000);
}

#[test]
fn cast_tuple_field_f32_to_f16_quiets_snan() {
    // An `f32` signaling NaN (payload in the top window) narrows
    // to the quiet `f16` pattern 0x7e01 natively; through a tuple
    // field the cast must agree.
    let tuple_ty = mir::Ty::Tuple(vec![f32_ty(), u8_ty()]);
    let v = run(
        "cast_tuple_field_f32_f16",
        &[f16_ty(), tuple_ty],
        vec![
            assign(
                place(1),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Tuple),
                    vec![const_f32(f32::from_bits(0x7f80_2000)), const_u8(7)],
                ),
            ),
            assign(
                place(0),
                mir::Rvalue::Cast(
                    mir::CastKind::FloatToFloat,
                    mir::Operand::Copy(place(1).field(0, f32_ty())),
                    f16_ty(),
                ),
            ),
        ],
    );
    assert_float_bits(v, 0x7ff8_0400_0000_0000);
}

#[test]
fn cast_tuple_field_f16_to_f64_quiets_snan() {
    // Widening an `f16` signaling NaN to `f64` sets the quiet bit
    // (native materialized cast), even when the source is read
    // out of a tuple field.
    let tuple_ty = mir::Ty::Tuple(vec![f16_ty(), u8_ty()]);
    let v = run(
        "cast_tuple_field_f16_f64",
        &[f64_ty(), tuple_ty],
        vec![
            assign(
                place(1),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Tuple),
                    vec![const_f16_bits(0x7c01), const_u8(7)],
                ),
            ),
            assign(
                place(0),
                mir::Rvalue::Cast(
                    mir::CastKind::FloatToFloat,
                    mir::Operand::Copy(place(1).field(0, f16_ty())),
                    f64_ty(),
                ),
            ),
        ],
    );
    assert_float_bits(v, 0x7ff8_0400_0000_0000);
}

#[test]
fn cast_deref_f16_to_f32_preserves_snan() {
    // Widening an `f16` signaling NaN to `f32` preserves it
    // (native: 0x7f802000); through a dereference the cast must
    // still know the source is `f16`, not host-convert the stored
    // `f64` form (which would quiet it to 0x7fc02000).
    let ref_ty = mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(f16_ty()),
        mir::Mutability::Not,
    );
    let v = run(
        "cast_deref_f16_f32",
        &[f32_ty(), f16_ty(), ref_ty],
        vec![
            assign(place(1), mir::Rvalue::Use(const_f16_bits(0x7c01))),
            assign(
                place(2),
                mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
            ),
            assign(
                place(0),
                mir::Rvalue::Cast(
                    mir::CastKind::FloatToFloat,
                    mir::Operand::Copy(place(2).deref()),
                    f32_ty(),
                ),
            ),
        ],
    );
    assert_float_bits(v, 0x7ff0_0400_0000_0000);
}

#[test]
fn cast_array_elem_f32_to_f16_quiets_snan() {
    // The same `f32` narrowing through an array element read by
    // constant index.
    let array_ty = mir::Ty::Array(Box::new(f32_ty()), mir::ArrayLen::Known(2));
    let elem = mir::Place {
        local: mir::Local(1),
        projection: vec![mir::PlaceElem::ConstantIndex {
            offset: 0,
            min_length: 2,
            from_end: false,
        }],
    };
    let v = run(
        "cast_array_elem_f32_f16",
        &[f16_ty(), array_ty],
        vec![
            assign(
                place(1),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Array(f32_ty())),
                    vec![const_f32(f32::from_bits(0x7f80_2000)), const_f32(0.0)],
                ),
            ),
            assign(
                place(0),
                mir::Rvalue::Cast(
                    mir::CastKind::FloatToFloat,
                    mir::Operand::Copy(elem),
                    f16_ty(),
                ),
            ),
        ],
    );
    assert_float_bits(v, 0x7ff8_0400_0000_0000);
}

// ----- constant cast sources (PR #419 Codex follow-up, round 2) -----

#[test]
fn cast_const_f64_to_f16_quiets_snan() {
    // A cast that consumes an `f64` constant directly is a genuine
    // narrowing: the signaling NaN is quieted with its payload
    // carried across (native `f64` sNaN -> `f16` = 0x7e01), exactly
    // as when the identical value is read from a local. It must not
    // be transported bit-exactly (0x7c01) just because the operand
    // was never materialized in a local.
    let v = run(
        "cast_const_f64_f16",
        &[f16_ty()],
        vec![assign(
            place(0),
            mir::Rvalue::Cast(
                mir::CastKind::FloatToFloat,
                const_f64(f64::from_bits(0x7ff0_0400_0000_0000)),
                f16_ty(),
            ),
        )],
    );
    assert_float_bits(v, 0x7ff8_0400_0000_0000);
}

#[test]
fn cast_const_f32_to_f16_quiets_snan() {
    // The same narrowing from an `f32` constant consumed directly.
    let v = run(
        "cast_const_f32_f16",
        &[f16_ty()],
        vec![assign(
            place(0),
            mir::Rvalue::Cast(
                mir::CastKind::FloatToFloat,
                const_f32(f32::from_bits(0x7f80_2000)),
                f16_ty(),
            ),
        )],
    );
    assert_float_bits(v, 0x7ff8_0400_0000_0000);
}

#[test]
fn cast_const_f16_to_f32_preserves_snan() {
    // Widening an `f16` signaling NaN constant to `f32` preserves
    // it (native: 0x7f802000); the cast must know the constant's
    // source is `f16`, not host-convert the stored `f64` form
    // (which would quiet it to 0x7fc02000).
    let v = run(
        "cast_const_f16_f32",
        &[f32_ty()],
        vec![assign(
            place(0),
            mir::Rvalue::Cast(
                mir::CastKind::FloatToFloat,
                const_f16_bits(0x7c01),
                f32_ty(),
            ),
        )],
    );
    assert_float_bits(v, 0x7ff0_0400_0000_0000);
}

#[test]
fn cast_const_f16_to_f64_quiets_snan() {
    // Widening an `f16` signaling NaN constant to `f64` sets the
    // quiet bit (native materialized cast), as it does for a local
    // or projected source.
    let v = run(
        "cast_const_f16_f64",
        &[f64_ty()],
        vec![assign(
            place(0),
            mir::Rvalue::Cast(
                mir::CastKind::FloatToFloat,
                const_f16_bits(0x7c01),
                f64_ty(),
            ),
        )],
    );
    assert_float_bits(v, 0x7ff8_0400_0000_0000);
}
