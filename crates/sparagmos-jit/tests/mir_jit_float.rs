//! MIR front-end float tests: hand-built `sparagmos-mir` bodies
//! with `f32` / `f64` locals, executed on the JIT.
//!
//! The interpreter has no float semantics yet (its conversion
//! rejects float constants), so it cannot be the oracle here. The
//! expected values are computed with Rust's own float operators and
//! `as` casts in this test process — the native semantics the JIT
//! must reproduce — and compared as bit patterns. NaN results are
//! compared by class (payloads may differ between code paths).

use std::collections::HashMap;

use sparagmos_jit::{JitProgram, MirFunctionInput, MirProgramInput};
use sparagmos_mir as mir;

// ----- small builders --------------------------------------------------------

fn def(index: u32) -> mir::DefId {
    mir::DefId { krate: 0, index }
}

fn bb(stmts: Vec<mir::StatementKind>, term: mir::TerminatorKind) -> mir::BasicBlockData {
    mir::BasicBlockData {
        statements: stmts.into_iter().map(mir::Statement::new).collect(),
        terminator: Some(mir::Terminator::new(term)),
        is_cleanup: false,
    }
}

fn assign(place: mir::Place, rv: mir::Rvalue) -> mir::StatementKind {
    mir::StatementKind::Assign(Box::new((place, rv)))
}

fn pl(n: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(n))
}

fn fld(n: u32, field: u32, ty: mir::Ty) -> mir::Place {
    pl(n).field(field, ty)
}

fn cp(n: u32) -> mir::Operand {
    mir::Operand::Copy(pl(n))
}

fn cpf(n: u32, field: u32, ty: mir::Ty) -> mir::Operand {
    mir::Operand::Copy(fld(n, field, ty))
}

fn mvf(n: u32, field: u32, ty: mir::Ty) -> mir::Operand {
    mir::Operand::Move(fld(n, field, ty))
}

fn f64c(v: f64) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Float {
            bits: v.to_bits() as u128,
            ty: mir::FloatTy::F64,
        },
    )))
}

fn f32c(v: f32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Float {
            bits: v.to_bits() as u128,
            ty: mir::FloatTy::F32,
        },
    )))
}

fn u64c(v: u64) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
            size: 8,
        },
    )))
}

fn fndef(d: mir::DefId) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(d, vec![])),
        user_ty: None,
    })
}

fn locals(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn f64t() -> mir::Ty {
    mir::Ty::Float(mir::FloatTy::F64)
}
fn f32t() -> mir::Ty {
    mir::Ty::Float(mir::FloatTy::F32)
}
fn u64t() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U64)
}

fn build(specs: Vec<(u32, &str, usize, Vec<mir::Ty>, Vec<mir::BasicBlockData>)>) -> JitProgram {
    let names: HashMap<mir::DefId, String> = specs
        .iter()
        .map(|(d, name, ..)| (def(*d), name.to_string()))
        .collect();
    let input = MirProgramInput {
        functions: specs
            .into_iter()
            .map(|(d, name, argc, tys, blocks)| MirFunctionInput {
                name: name.to_string(),
                def_id: Some(def(d)),
                body: mir::Body::new(locals(&tys), argc, blocks),
            })
            .collect(),
        names,
    };
    JitProgram::compile(&input).expect("JIT compile failed")
}

/// A two-argument float binary operation as a whole function:
/// `fn f(a, b) = a <op> b` with all locals of type `ty`.
fn binop_fn<'a>(
    name: &'a str,
    ty: mir::Ty,
    op: mir::BinOp,
) -> (u32, &'a str, usize, Vec<mir::Ty>, Vec<mir::BasicBlockData>) {
    (
        0,
        name,
        2,
        vec![ty.clone(), ty.clone(), ty],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(op, Box::new((cp(1), cp(2)))),
            )],
            mir::TerminatorKind::Return,
        )],
    )
}

fn call_bits(jit: &JitProgram, name: &str, args: &[i64]) -> u64 {
    jit.call(name, args).expect("JIT call failed") as u64
}

/// Compare a JIT-produced f64 bit pattern with the natively
/// computed value: bit-exact, except that NaN matches NaN.
fn assert_f64(got_bits: u64, want: f64, ctx: &str) {
    let got = f64::from_bits(got_bits);
    assert!(
        got_bits == want.to_bits() || (got.is_nan() && want.is_nan()),
        "{ctx}: got {got} ({got_bits:#x}), want {want} ({:#x})",
        want.to_bits()
    );
}

fn assert_f32(got_bits: u64, want: f32, ctx: &str) {
    assert_eq!(got_bits >> 32, 0, "{ctx}: f32 result not canonical: {got_bits:#x}");
    let got = f32::from_bits(got_bits as u32);
    assert!(
        got_bits as u32 == want.to_bits() || (got.is_nan() && want.is_nan()),
        "{ctx}: got {got}, want {want}"
    );
}

// ----- f64 arithmetic ----------------------------------------------------------

#[test]
fn f64_arithmetic_matches_native() {
    let cases: &[(f64, f64)] = &[
        (1.5, 2.25),
        (0.1, 0.2),
        (-7.25, 0.5),
        (1e308, 1e308),
        (1.0, 0.0),
        (0.0, 0.0),
        (-0.0, 0.0),
        (f64::INFINITY, 1.0),
        (f64::INFINITY, f64::INFINITY),
        (f64::NAN, 1.0),
        (5.5, 2.0),
        (-5.5, 2.0),
        (1e300, 3.0),
    ];
    for (name, op, want) in [
        ("add", mir::BinOp::Add, (|a: f64, b: f64| a + b) as fn(f64, f64) -> f64),
        ("sub", mir::BinOp::Sub, (|a, b| a - b) as fn(f64, f64) -> f64),
        ("mul", mir::BinOp::Mul, (|a, b| a * b) as fn(f64, f64) -> f64),
        ("div", mir::BinOp::Div, (|a, b| a / b) as fn(f64, f64) -> f64),
        ("rem", mir::BinOp::Rem, (|a, b| a % b) as fn(f64, f64) -> f64),
    ] {
        let jit = build(vec![binop_fn(name, f64t(), op)]);
        for &(a, b) in cases {
            let got = call_bits(&jit, name, &[a.to_bits() as i64, b.to_bits() as i64]);
            assert_f64(got, want(a, b), &format!("{name}({a}, {b})"));
        }
    }
}

// ----- f32 arithmetic ----------------------------------------------------------

#[test]
fn f32_arithmetic_matches_native() {
    let cases: &[(f32, f32)] = &[
        (1.5, 2.25),
        (0.1, 0.2),
        (-7.25, 0.5),
        (3.4e38, 3.4e38), // overflows to infinity in f32
        (1.0, 0.0),
        (0.0, 0.0),
        (f32::INFINITY, 1.0),
        (f32::NAN, 1.0),
        (7.5, 2.0),
        (-7.5, 2.0),
    ];
    for (name, op, want) in [
        ("add", mir::BinOp::Add, (|a: f32, b: f32| a + b) as fn(f32, f32) -> f32),
        ("sub", mir::BinOp::Sub, (|a, b| a - b) as fn(f32, f32) -> f32),
        ("mul", mir::BinOp::Mul, (|a, b| a * b) as fn(f32, f32) -> f32),
        ("div", mir::BinOp::Div, (|a, b| a / b) as fn(f32, f32) -> f32),
        ("rem", mir::BinOp::Rem, (|a, b| a % b) as fn(f32, f32) -> f32),
    ] {
        let jit = build(vec![binop_fn(name, f32t(), op)]);
        for &(a, b) in cases {
            let got = call_bits(&jit, name, &[a.to_bits() as i64, b.to_bits() as i64]);
            assert_f32(got, want(a, b), &format!("{name}({a}, {b})"));
        }
    }
}

// ----- comparisons ---------------------------------------------------------------

fn cmp_program(ty: mir::Ty, op: mir::BinOp) -> JitProgram {
    build(vec![(
        0,
        "cmp",
        2,
        vec![mir::Ty::Bool, ty.clone(), ty],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(op, Box::new((cp(1), cp(2)))),
            )],
            mir::TerminatorKind::Return,
        )],
    )])
}

#[test]
fn f64_comparisons_match_native() {
    let cases: &[(f64, f64)] = &[
        (1.0, 2.0),
        (2.0, 1.0),
        (1.5, 1.5),
        (0.0, -0.0),
        (-0.0, 0.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::NAN, f64::NAN),
        (f64::INFINITY, 1e308),
        (f64::NEG_INFINITY, -1e308),
        (-1.5, -2.5),
    ];
    for (name, op, want) in [
        ("eq", mir::BinOp::Eq, (|a: f64, b: f64| a == b) as fn(f64, f64) -> bool),
        ("ne", mir::BinOp::Ne, (|a, b| a != b) as fn(f64, f64) -> bool),
        ("lt", mir::BinOp::Lt, (|a, b| a < b) as fn(f64, f64) -> bool),
        ("le", mir::BinOp::Le, (|a, b| a <= b) as fn(f64, f64) -> bool),
        ("gt", mir::BinOp::Gt, (|a, b| a > b) as fn(f64, f64) -> bool),
        ("ge", mir::BinOp::Ge, (|a, b| a >= b) as fn(f64, f64) -> bool),
    ] {
        let jit = cmp_program(f64t(), op);
        for &(a, b) in cases {
            let got = call_bits(&jit, "cmp", &[a.to_bits() as i64, b.to_bits() as i64]);
            assert_eq!(got, want(a, b) as u64, "{name}({a}, {b})");
        }
    }
}

#[test]
fn f32_comparisons_match_native() {
    let cases: &[(f32, f32)] = &[
        (1.0, 2.0),
        (2.0, 1.0),
        (1.5, 1.5),
        (0.0, -0.0),
        (f32::NAN, 1.0),
        (1.0, f32::NAN),
        (f32::INFINITY, 3.0e38),
    ];
    for (op, want) in [
        (mir::BinOp::Eq, (|a: f32, b: f32| a == b) as fn(f32, f32) -> bool),
        (mir::BinOp::Ne, (|a, b| a != b) as fn(f32, f32) -> bool),
        (mir::BinOp::Lt, (|a, b| a < b) as fn(f32, f32) -> bool),
        (mir::BinOp::Le, (|a, b| a <= b) as fn(f32, f32) -> bool),
        (mir::BinOp::Gt, (|a, b| a > b) as fn(f32, f32) -> bool),
        (mir::BinOp::Ge, (|a, b| a >= b) as fn(f32, f32) -> bool),
    ] {
        let jit = cmp_program(f32t(), op);
        for &(a, b) in cases {
            let got = call_bits(&jit, "cmp", &[a.to_bits() as i64, b.to_bits() as i64]);
            assert_eq!(got, want(a, b) as u64, "{op:?}({a}, {b})");
        }
    }
}

// ----- negation ------------------------------------------------------------------

#[test]
fn float_negation_matches_native() {
    // f64 cases, including the sign flip of NaN and of zeroes.
    let jit = build(vec![(
        0,
        "neg64",
        1,
        vec![f64t(), f64t()],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::UnaryOp(mir::UnOp::Neg, cp(1)))],
            mir::TerminatorKind::Return,
        )],
    )]);
    for a in [1.5f64, -2.5, 0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let got = call_bits(&jit, "neg64", &[a.to_bits() as i64]);
        // Negation is exact (a pure sign flip), so compare bit-for-bit
        // even for NaN.
        assert_eq!(got, (-a).to_bits(), "neg({a})");
    }
    let jit = build(vec![(
        0,
        "neg32",
        1,
        vec![f32t(), f32t()],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::UnaryOp(mir::UnOp::Neg, cp(1)))],
            mir::TerminatorKind::Return,
        )],
    )]);
    for a in [1.5f32, -2.5, 0.0, -0.0, f32::INFINITY, f32::NAN] {
        let got = call_bits(&jit, "neg32", &[a.to_bits() as i64]);
        assert_eq!(got, (-a).to_bits() as u64, "neg({a})");
    }
}

// ----- casts ---------------------------------------------------------------------

fn cast_fn<'a>(
    name: &'a str,
    kind: mir::CastKind,
    src_ty: mir::Ty,
    dst_ty: mir::Ty,
) -> (u32, &'a str, usize, Vec<mir::Ty>, Vec<mir::BasicBlockData>) {
    (
        0,
        name,
        1,
        vec![dst_ty.clone(), src_ty],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Cast(kind, cp(1), dst_ty))],
            mir::TerminatorKind::Return,
        )],
    )
}

#[test]
fn float_to_int_casts_saturate_like_native() {
    let i64t = mir::Ty::Int(mir::IntTy::I64);
    let i32t = mir::Ty::Int(mir::IntTy::I32);
    let u64t = u64t();
    let u8t = mir::Ty::Uint(mir::UintTy::U8);
    // f64 -> i64
    let jit = build(vec![cast_fn("c", mir::CastKind::FloatToInt, f64t(), i64t.clone())]);
    for a in [
        3.99f64, -3.99, 0.5, -0.5, 0.0, -0.0, f64::NAN, 1e300, -1e300,
        9223372036854775808.0, -9223372036854775808.0, 9223372036854774784.0,
        f64::INFINITY, f64::NEG_INFINITY,
    ] {
        let got = call_bits(&jit, "c", &[a.to_bits() as i64]) as i64;
        assert_eq!(got, a as i64, "({a}) as i64");
    }
    // f64 -> u64
    let jit = build(vec![cast_fn("c", mir::CastKind::FloatToInt, f64t(), u64t.clone())]);
    for a in [
        3.99f64, -3.99, -0.5, 0.0, f64::NAN, 1e300, 1.8e19,
        18446744073709551616.0, 18446744073709549568.0, 9223372036854775808.0,
        f64::INFINITY,
    ] {
        let got = call_bits(&jit, "c", &[a.to_bits() as i64]);
        assert_eq!(got, a as u64, "({a}) as u64");
    }
    // f64 -> i32 (narrow saturation)
    let jit = build(vec![cast_fn("c", mir::CastKind::FloatToInt, f64t(), i32t)]);
    for a in [3.99f64, -3.99, 3.0e9, -3.0e9, 2147483647.0, -2147483648.0, f64::NAN, 1e300] {
        let got = call_bits(&jit, "c", &[a.to_bits() as i64]) as i64;
        assert_eq!(got, (a as i32) as i64, "({a}) as i32");
    }
    // f64 -> u8 (narrow unsigned clamp)
    let jit = build(vec![cast_fn("c", mir::CastKind::FloatToInt, f64t(), u8t)]);
    for a in [3.99f64, -3.99, 300.0, 255.9, 256.0, f64::NAN] {
        let got = call_bits(&jit, "c", &[a.to_bits() as i64]);
        assert_eq!(got, (a as u8) as u64, "({a}) as u8");
    }
    // f32 -> i64
    let jit = build(vec![cast_fn("c", mir::CastKind::FloatToInt, f32t(), i64t)]);
    for a in [3.99f32, -3.99, f32::NAN, 1e30f32, -1e30, 2147483648.0] {
        let got = call_bits(&jit, "c", &[a.to_bits() as i64]) as i64;
        assert_eq!(got, a as i64, "({a}) as i64");
    }
    // f32 -> u64
    let jit = build(vec![cast_fn("c", mir::CastKind::FloatToInt, f32t(), u64t)]);
    for a in [3.99f32, -1.5, f32::NAN, 1.8e19f32] {
        let got = call_bits(&jit, "c", &[a.to_bits() as i64]);
        assert_eq!(got, a as u64, "({a}) as u64");
    }
}

#[test]
fn int_to_float_casts_match_native() {
    let i64t = mir::Ty::Int(mir::IntTy::I64);
    let i32t = mir::Ty::Int(mir::IntTy::I32);
    // i64 -> f64 (rounding at 2^53)
    let jit = build(vec![cast_fn("c", mir::CastKind::IntToFloat, i64t.clone(), f64t())]);
    for a in [0i64, 1, -1, 42, (1 << 53) + 1, -(1 << 53) - 1, i64::MAX, i64::MIN] {
        let got = call_bits(&jit, "c", &[a]);
        assert_eq!(got, (a as f64).to_bits(), "({a}) as f64");
    }
    // u64 -> f64 (the top-bit path)
    let jit = build(vec![cast_fn("c", mir::CastKind::IntToFloat, u64t(), f64t())]);
    for a in [0u64, 1, 42, u64::MAX, u64::MAX - 1, 1 << 63, (1 << 53) + 1] {
        let got = call_bits(&jit, "c", &[a as i64]);
        assert_eq!(got, (a as f64).to_bits(), "({a}) as f64");
    }
    // i64 -> f32
    let jit = build(vec![cast_fn("c", mir::CastKind::IntToFloat, i64t.clone(), f32t())]);
    for a in [0i64, 1, -1, (1 << 24) + 1, -(1 << 24) - 1, i64::MAX, i64::MIN] {
        let got = call_bits(&jit, "c", &[a]);
        assert_f32(got, a as f32, &format!("({a}) as f32"));
    }
    // u64 -> f32 (the halving path for large values)
    let jit = build(vec![cast_fn("c", mir::CastKind::IntToFloat, u64t(), f32t())]);
    for a in [0u64, 1, 42, u64::MAX, u64::MAX - 1, u64::MAX - 12345, 1 << 63, (1 << 24) + 1] {
        let got = call_bits(&jit, "c", &[a as i64]);
        assert_f32(got, a as f32, &format!("({a}) as f32"));
    }
    // i32 -> f32 and u8 -> f64 (narrow sources)
    let jit = build(vec![cast_fn("c", mir::CastKind::IntToFloat, i32t, f32t())]);
    for a in [0i32, -1, 16777217, i32::MAX, i32::MIN] {
        let got = call_bits(&jit, "c", &[a as i64]);
        assert_f32(got, a as f32, &format!("({a}) as f32"));
    }
    let jit = build(vec![cast_fn(
        "c",
        mir::CastKind::IntToFloat,
        mir::Ty::Uint(mir::UintTy::U8),
        f64t(),
    )]);
    for a in [0u8, 200, 255] {
        let got = call_bits(&jit, "c", &[a as i64]);
        assert_eq!(got, (a as f64).to_bits(), "({a}) as f64");
    }
}

#[test]
fn float_to_float_casts_match_native() {
    // f64 -> f32: rounding, overflow to infinity, underflow.
    let jit = build(vec![cast_fn("c", mir::CastKind::FloatToFloat, f64t(), f32t())]);
    for a in [
        0.1f64, 1.5, -2.75, 1e300, -1e300, 1e-300, 3.4028235e38, 3.4028236e38,
        0.0, -0.0, f64::INFINITY, f64::NAN, 16777217.0,
    ] {
        let got = call_bits(&jit, "c", &[a.to_bits() as i64]);
        assert_f32(got, a as f32, &format!("({a}) as f32"));
    }
    // f32 -> f64: exact.
    let jit = build(vec![cast_fn("c", mir::CastKind::FloatToFloat, f32t(), f64t())]);
    for a in [0.1f32, 1.5, -2.75, 3.4e38, 1e-40, 0.0, f32::INFINITY, f32::NAN] {
        let got = call_bits(&jit, "c", &[a.to_bits() as i64]);
        assert_f64(got, a as f64, &format!("({a}) as f64"));
    }
}

// ----- constants, mixed operands, control flow -----------------------------------

#[test]
fn float_constants_and_mixed_operands() {
    // fn k(x: f64) -> f64 { x * 2.5 + 0.25 }
    let jit = build(vec![(
        0,
        "k",
        1,
        vec![f64t(), f64t(), f64t()],
        vec![bb(
            vec![
                assign(
                    pl(2),
                    mir::Rvalue::BinaryOp(mir::BinOp::Mul, Box::new((cp(1), f64c(2.5)))),
                ),
                assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(2), f64c(0.25)))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    )]);
    for x in [0.0f64, 1.0, -3.3, 1e10] {
        let got = call_bits(&jit, "k", &[x.to_bits() as i64]);
        assert_f64(got, x * 2.5 + 0.25, &format!("k({x})"));
    }

    // fn lt_const(x: f64) -> bool { x < 2.5 } (place vs constant)
    let jit = build(vec![(
        0,
        "lt_const",
        1,
        vec![mir::Ty::Bool, f64t()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(mir::BinOp::Lt, Box::new((cp(1), f64c(2.5)))),
            )],
            mir::TerminatorKind::Return,
        )],
    )]);
    for (x, want) in [(2.4f64, 1u64), (2.5, 0), (3.0, 0), (f64::NAN, 0)] {
        assert_eq!(call_bits(&jit, "lt_const", &[x.to_bits() as i64]), want, "lt_const({x})");
    }

    // fn lit() -> bool { 1.5 < 2.5 } (constant vs constant)
    let jit = build(vec![(
        0,
        "lit",
        0,
        vec![mir::Ty::Bool],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(mir::BinOp::Lt, Box::new((f64c(1.5), f64c(2.5)))),
            )],
            mir::TerminatorKind::Return,
        )],
    )]);
    assert_eq!(call_bits(&jit, "lit", &[]), 1);

    // fn eq_nan() -> bool { 0.0/0.0 == 0.0/0.0 } is false, via constants.
    let jit = build(vec![(
        0,
        "nanlit",
        0,
        vec![mir::Ty::Bool, f64t()],
        vec![bb(
            vec![
                assign(
                    pl(1),
                    mir::Rvalue::BinaryOp(mir::BinOp::Div, Box::new((f64c(0.0), f64c(0.0)))),
                ),
                assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(mir::BinOp::Eq, Box::new((cp(1), cp(1)))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    )]);
    assert_eq!(call_bits(&jit, "nanlit", &[]), 0);
}

#[test]
fn float_loop_and_call() {
    // fn sum_to(n: u64) -> f64 { let mut s = 0.0; let mut i = 0; while i < n { s += i as f64; i += 1 } s }
    let pair = mir::Ty::Tuple(vec![u64t(), mir::Ty::Bool]);
    let sum_to = (
        0,
        "sum_to",
        1,
        vec![f64t(), u64t(), u64t(), f64t(), mir::Ty::Bool, pair],
        vec![
            bb(
                vec![
                    assign(pl(2), mir::Rvalue::Use(u64c(0))),
                    assign(pl(0), mir::Rvalue::Use(f64c(0.0))),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            bb(
                vec![assign(
                    pl(4),
                    mir::Rvalue::BinaryOp(mir::BinOp::Lt, Box::new((cp(2), cp(1)))),
                )],
                mir::TerminatorKind::SwitchInt {
                    discr: cp(4),
                    targets: mir::SwitchTargets::new(
                        vec![0],
                        vec![mir::BasicBlock(5)],
                        mir::BasicBlock(2),
                    ),
                },
            ),
            bb(
                vec![
                    assign(
                        pl(3),
                        mir::Rvalue::Cast(mir::CastKind::IntToFloat, cp(2), f64t()),
                    ),
                    assign(
                        pl(0),
                        mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(0), cp(3)))),
                    ),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(3),
                },
            ),
            bb(
                vec![assign(
                    pl(5),
                    mir::Rvalue::CheckedBinaryOp(
                        mir::BinOp::Add,
                        Box::new((cp(2), u64c(1))),
                    ),
                )],
                mir::TerminatorKind::Assert {
                    cond: mir::Operand::Copy(fld(5, 1, mir::Ty::Bool)),
                    expected: false,
                    msg: Box::new(mir::AssertMessage::Overflow(
                        mir::BinOp::Add,
                        Box::new(cp(1)),
                        Box::new(cp(1)),
                    )),
                    target: mir::BasicBlock(4),
                    unwind: mir::UnwindAction::Continue,
                },
            ),
            bb(
                vec![assign(pl(2), mir::Rvalue::Use(mvf(5, 0, u64t())))],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    let jit = build(vec![sum_to]);
    for n in [0u64, 1, 10, 100, 1000] {
        // The expectation accumulates from +0.0 in order, exactly
        // like the MIR program. (Iterator::sum is not used: its
        // float identity is -0.0, which differs for n == 0.)
        let mut want = 0.0f64;
        for i in 0..n {
            want += i as f64;
        }
        let got = call_bits(&jit, "sum_to", &[n as i64]);
        assert_f64(got, want, &format!("sum_to({n})"));
    }

    // fn main() -> f64 { addf(1.25, 2.5) } — a call with float
    // arguments and a float result.
    let addf = binop_fn("addf", f64t(), mir::BinOp::Add);
    let main = (
        1,
        "main",
        0,
        vec![f64t(), f64t()],
        vec![
            bb(
                vec![],
                mir::TerminatorKind::Call {
                    func: fndef(def(0)),
                    args: vec![f64c(1.25), f64c(2.5)],
                    destination: pl(1),
                    target: Some(mir::BasicBlock(1)),
                    unwind: mir::UnwindAction::Continue,
                    call_source: mir::CallSource::Normal,
                },
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(1)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let jit = build(vec![addf, main]);
    assert_f64(call_bits(&jit, "main", &[]), 3.75, "main()");
}

#[test]
fn float_tuple_fields() {
    // fn f(a: f64) -> f64 { let t = (a, a * 2.0); t.0 + t.1 }
    let pair_ty = mir::Ty::Tuple(vec![f64t(), f64t()]);
    let jit = build(vec![(
        0,
        "f",
        1,
        vec![f64t(), f64t(), pair_ty, f64t()],
        vec![bb(
            vec![
                assign(
                    pl(3),
                    mir::Rvalue::BinaryOp(mir::BinOp::Mul, Box::new((cp(1), f64c(2.0)))),
                ),
                assign(
                    pl(2),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![cp(1), cp(3)],
                    ),
                ),
                assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((cpf(2, 0, f64t()), cpf(2, 1, f64t()))),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    )]);
    for a in [0.5f64, -1.25, 100.0] {
        let got = call_bits(&jit, "f", &[a.to_bits() as i64]);
        assert_f64(got, a + a * 2.0, &format!("f({a})"));
    }
    // f32 constants in a tuple-free shape: fn g(x: f32) -> f32 { x + 0.5 }
    let jit = build(vec![(
        0,
        "g",
        1,
        vec![f32t(), f32t(), f32t()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(1), f32c(0.5)))),
            )],
            mir::TerminatorKind::Return,
        )],
    )]);
    for a in [0.25f32, -1.5, 1e20] {
        let got = call_bits(&jit, "g", &[a.to_bits() as i64]);
        assert_f32(got, a + 0.5, &format!("g({a})"));
    }
}
