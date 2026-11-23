//! Primitive float method builtins: the `f32` / `f64` inherent
//! methods whose bodies live in core/std are executed by
//! `builtins::exec_float_method` at the method's own width.
//!
//! Semantics pinned here (Rust 1.100.0), all checked against
//! host-computed values:
//! * `f32` methods compute on the `f32` value (results round to
//!   `f32`, then widen back exactly).
//! * `min` / `max` ignore a NaN operand instead of propagating it.
//! * `round` rounds half away from zero; `signum` of NaN is NaN.
//! * `abs` clears the sign bit, including for NaN and -0.0.

use std::collections::HashMap;

use sparagmos_interp::builtins::{lookup, Builtin, FloatOp};
use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::Value;
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir (mirroring float_tests) -----------

fn f64_ty() -> mir::Ty {
    mir::Ty::Float(mir::FloatTy::F64)
}
fn f32_ty() -> mir::Ty {
    mir::Ty::Float(mir::FloatTy::F32)
}
fn u64_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U64)
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

fn const_int(data: u128, size: u8) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int { data, size },
    )))
}

fn const_i32(v: i32) -> mir::Operand {
    const_int(v as u32 as u128, 4)
}

fn block(stmts: Vec<mir::Statement>, kind: mir::TerminatorKind) -> mir::BasicBlockData {
    let mut bb = mir::BasicBlockData::new(mir::Terminator::new(kind));
    bb.statements = stmts;
    bb
}

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

fn fndef(def: mir::DefId, generic_args: Vec<mir::Ty>) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def, generic_args)),
        user_ty: None,
        ty: None,
    })
}

fn call(
    func: mir::Operand,
    args: Vec<mir::Operand>,
    dest: u32,
    next: u32,
) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func,
        args,
        destination: place(dest),
        target: Some(mir::BasicBlock(next)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

const METHOD_DEF: mir::DefId = mir::DefId::new(0, 900);

/// A `main` whose only action is `_0 = <name>(args...)` then return.
fn run_method(name: &str, ret_ty: mir::Ty, args: Vec<mir::Operand>) -> Value {
    let body = mir::Body::new(
        decls(&[ret_ty]),
        0,
        vec![
            block(vec![], call(fndef(METHOD_DEF, vec![]), args, 0, 1)),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    prog.fn_names = HashMap::from([(METHOD_DEF, name.to_string())]);
    interpret_mir(&prog, "main", vec![])
        .unwrap_or_else(|e| panic!("{name} must run: {e}"))
}

// ----- the registry -----------------------------------------------------------

#[test]
fn registry_covers_float_methods_at_both_widths_and_prefixes() {
    use FloatOp as O;
    let cases: &[(&str, bool, FloatOp)] = &[
        ("is_nan", false, O::IsNan),
        ("is_infinite", true, O::IsInfinite),
        ("is_finite", false, O::IsFinite),
        ("is_sign_positive", true, O::IsSignPositive),
        ("is_sign_negative", false, O::IsSignNegative),
        ("abs", true, O::Abs),
        ("sqrt", false, O::Sqrt),
        ("floor", true, O::Floor),
        ("ceil", false, O::Ceil),
        ("round", true, O::Round),
        ("trunc", false, O::Trunc),
        ("fract", true, O::Fract),
        ("recip", false, O::Recip),
        ("signum", true, O::Signum),
        ("to_degrees", false, O::ToDegrees),
        ("to_radians", true, O::ToRadians),
        ("powi", false, O::Powi),
        ("powf", true, O::Powf),
        ("min", false, O::Min),
        ("max", true, O::Max),
        ("to_bits", false, O::ToBits),
        ("from_bits", true, O::FromBits),
        ("exp", false, O::Exp),
        ("exp2", true, O::Exp2),
        ("exp_m1", false, O::ExpM1),
        ("ln", true, O::Ln),
        ("ln_1p", false, O::Ln1p),
        ("log", true, O::Log),
        ("log2", false, O::Log2),
        ("log10", true, O::Log10),
        ("sin", false, O::Sin),
        ("cos", true, O::Cos),
        ("tan", false, O::Tan),
        ("asin", true, O::Asin),
        ("acos", false, O::Acos),
        ("atan", true, O::Atan),
        ("atan2", false, O::Atan2),
        ("sin_cos", true, O::SinCos),
        ("sinh", false, O::Sinh),
        ("cosh", true, O::Cosh),
        ("tanh", false, O::Tanh),
        ("asinh", true, O::Asinh),
        ("acosh", false, O::Acosh),
        ("atanh", true, O::Atanh),
        ("cbrt", false, O::Cbrt),
        ("hypot", true, O::Hypot),
        ("mul_add", false, O::MulAdd),
        ("div_euclid", true, O::DivEuclid),
    ];
    for (method, _flip, op) in cases {
        for prefix in ["core", "std"] {
            for (ty_name, f32) in [("f32", true), ("f64", false)] {
                let name = format!("{prefix}::{ty_name}::<impl {ty_name}>::{method}");
                assert_eq!(
                    lookup(&name),
                    Some(Builtin::FloatMethod { f32, op: *op }),
                    "{name}"
                );
            }
        }
    }
}

#[test]
fn registry_leaves_unmodelled_float_methods_unsupported() {
    // `maximum` / `minimum` are still unstable library features in
    // Rust 1.100.0 and are not modelled: they must stay explicit
    // unsupported calls, not silent matches.
    assert_eq!(lookup("core::f64::<impl f64>::maximum"), None);
    assert_eq!(lookup("std::f32::<impl f32>::minimum"), None);
    // A non-float inherent impl is untouched by the float dispatch.
    assert_eq!(lookup("core::i32::<impl i32>::abs"), None);
}

// ----- predicates ---------------------------------------------------------------

#[test]
fn f64_is_nan() {
    let name = "core::f64::<impl f64>::is_nan";
    assert_eq!(
        run_method(name, bool_ty(), vec![const_f64(f64::NAN)]),
        Value::Bool(true)
    );
    assert_eq!(
        run_method(name, bool_ty(), vec![const_f64(1.0)]),
        Value::Bool(false)
    );
}

#[test]
fn f32_is_nan_and_sign_of_zero() {
    assert_eq!(
        run_method(
            "core::f32::<impl f32>::is_nan",
            bool_ty(),
            vec![const_f32(f32::NAN)]
        ),
        Value::Bool(true)
    );
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::is_sign_negative",
            bool_ty(),
            vec![const_f64(-0.0)]
        ),
        Value::Bool(true)
    );
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::is_sign_positive",
            bool_ty(),
            vec![const_f64(-0.0)]
        ),
        Value::Bool(false)
    );
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::is_infinite",
            bool_ty(),
            vec![const_f64(f64::INFINITY)]
        ),
        Value::Bool(true)
    );
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::is_finite",
            bool_ty(),
            vec![const_f64(f64::INFINITY)]
        ),
        Value::Bool(false)
    );
}

// ----- unary float methods -------------------------------------------------------

#[test]
fn f64_abs_clears_sign_including_nan_and_neg_zero() {
    let name = "core::f64::<impl f64>::abs";
    assert_eq!(
        run_method(name, f64_ty(), vec![const_f64(-3.5)]),
        Value::Float(3.5)
    );
    let neg_zero = run_method(name, f64_ty(), vec![const_f64(-0.0)]);
    assert_eq!(neg_zero, Value::Float(0.0));
    match run_method(name, f64_ty(), vec![const_f64(f64::NAN)]) {
        Value::Float(v) => {
            assert!(v.is_nan());
            assert!(v.is_sign_positive(), "abs(NaN) clears the sign bit");
        }
        other => panic!("expected a float, got {other:?}"),
    }
}

#[test]
fn f64_sqrt_floor_ceil_round_trunc_fract() {
    assert_eq!(
        run_method("std::f64::<impl f64>::sqrt", f64_ty(), vec![const_f64(2.0)]),
        Value::Float(2.0f64.sqrt())
    );
    assert_eq!(
        run_method("std::f64::<impl f64>::floor", f64_ty(), vec![const_f64(-2.5)]),
        Value::Float(-3.0)
    );
    assert_eq!(
        run_method("std::f64::<impl f64>::ceil", f64_ty(), vec![const_f64(-2.5)]),
        Value::Float(-2.0)
    );
    // round: half away from zero.
    assert_eq!(
        run_method("std::f64::<impl f64>::round", f64_ty(), vec![const_f64(2.5)]),
        Value::Float(3.0)
    );
    assert_eq!(
        run_method("std::f64::<impl f64>::round", f64_ty(), vec![const_f64(-2.5)]),
        Value::Float(-3.0)
    );
    assert_eq!(
        run_method("std::f64::<impl f64>::trunc", f64_ty(), vec![const_f64(-2.5)]),
        Value::Float(-2.0)
    );
    assert_eq!(
        run_method("std::f64::<impl f64>::fract", f64_ty(), vec![const_f64(2.75)]),
        Value::Float(0.75)
    );
}

#[test]
fn f64_recip_signum_degrees_radians() {
    assert_eq!(
        run_method("core::f64::<impl f64>::recip", f64_ty(), vec![const_f64(4.0)]),
        Value::Float(0.25)
    );
    assert_eq!(
        run_method("core::f64::<impl f64>::signum", f64_ty(), vec![const_f64(-0.5)]),
        Value::Float(-1.0)
    );
    match run_method(
        "core::f64::<impl f64>::signum",
        f64_ty(),
        vec![const_f64(f64::NAN)],
    ) {
        Value::Float(v) => assert!(v.is_nan(), "signum(NaN) is NaN"),
        other => panic!("expected a float, got {other:?}"),
    }
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::to_degrees",
            f64_ty(),
            vec![const_f64(std::f64::consts::PI)]
        ),
        Value::Float(std::f64::consts::PI.to_degrees())
    );
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::to_radians",
            f64_ty(),
            vec![const_f64(180.0)]
        ),
        Value::Float(180.0f64.to_radians())
    );
}

#[test]
fn f32_methods_compute_at_f32_width() {
    // sqrt(2) differs between f32 and f64; the f32 method must give
    // the f32-rounded value, widened exactly.
    assert_eq!(
        run_method("std::f32::<impl f32>::sqrt", f32_ty(), vec![const_f32(2.0)]),
        Value::Float(2.0f32.sqrt() as f64)
    );
    assert_eq!(
        run_method("std::f32::<impl f32>::floor", f32_ty(), vec![const_f32(2.5)]),
        Value::Float(2.0)
    );
    assert_eq!(
        run_method("core::f32::<impl f32>::abs", f32_ty(), vec![const_f32(-1.25)]),
        Value::Float(1.25)
    );
}

// ----- binary float methods -------------------------------------------------------

#[test]
fn f64_powi_and_powf() {
    assert_eq!(
        run_method(
            "std::f64::<impl f64>::powi",
            f64_ty(),
            vec![const_f64(1.5), const_i32(3)]
        ),
        Value::Float(3.375)
    );
    assert_eq!(
        run_method(
            "std::f64::<impl f64>::powi",
            f64_ty(),
            vec![const_f64(2.0), const_i32(-2)]
        ),
        Value::Float(0.25)
    );
    assert_eq!(
        run_method(
            "std::f64::<impl f64>::powf",
            f64_ty(),
            vec![const_f64(2.0), const_f64(0.5)]
        ),
        Value::Float(2.0f64.powf(0.5))
    );
    assert_eq!(
        run_method(
            "std::f32::<impl f32>::powf",
            f32_ty(),
            vec![const_f32(2.0), const_f32(0.5)]
        ),
        Value::Float(2.0f32.powf(0.5) as f64)
    );
}

#[test]
fn f64_min_max_ignore_nan() {
    // Rust semantics: min/max return the non-NaN operand.
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::min",
            f64_ty(),
            vec![const_f64(f64::NAN), const_f64(1.0)]
        ),
        Value::Float(1.0)
    );
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::max",
            f64_ty(),
            vec![const_f64(1.0), const_f64(f64::NAN)]
        ),
        Value::Float(1.0)
    );
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::min",
            f64_ty(),
            vec![const_f64(2.0), const_f64(1.0)]
        ),
        Value::Float(1.0)
    );
    assert_eq!(
        run_method(
            "core::f32::<impl f32>::max",
            f32_ty(),
            vec![const_f32(2.0), const_f32(1.0)]
        ),
        Value::Float(2.0)
    );
}

// ----- bit casts -------------------------------------------------------------------

#[test]
fn to_bits_and_from_bits_round_trip() {
    assert_eq!(
        run_method("core::f64::<impl f64>::to_bits", u64_ty(), vec![const_f64(1.0)]),
        Value::Uint(0x3FF0_0000_0000_0000)
    );
    assert_eq!(
        run_method("core::f32::<impl f32>::to_bits", u32_ty(), vec![const_f32(1.0)]),
        Value::Uint(0x3F80_0000)
    );
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::from_bits",
            f64_ty(),
            vec![const_int(0x3FF0_0000_0000_0000, 8)]
        ),
        Value::Float(1.0)
    );
    assert_eq!(
        run_method(
            "core::f32::<impl f32>::from_bits",
            f32_ty(),
            vec![const_int(0x3F80_0000, 4)]
        ),
        Value::Float(1.0)
    );
}

// ----- unmodelled methods stay errors ------------------------------------------------

#[test]
fn unmodelled_float_method_call_is_an_error_not_a_silent_value() {
    let body = mir::Body::new(
        decls(&[f64_ty()]),
        0,
        vec![
            block(
                vec![],
                call(
                    fndef(METHOD_DEF, vec![]),
                    vec![const_f64(1.0)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    prog.fn_names = HashMap::from([(
        METHOD_DEF,
        "std::f64::<impl f64>::maximum".to_string(),
    )]);
    assert!(interpret_mir(&prog, "main", vec![]).is_err());
}

// ----- the transcendental / libm family ------------------------------------

#[test]
fn f64_transcendental_unary_methods() {
    // Every value is compared against the host method, which is the
    // definition core/std themselves use on this platform.
    let cases: &[(&str, f64)] = &[
        ("exp", 1.25),
        ("exp2", 9.5),
        ("exp_m1", 0.75),
        ("ln", 2.5),
        ("ln_1p", 0.5),
        ("log2", 1024.0),
        ("log10", 0.001),
        ("sin", 0.7),
        ("cos", 0.7),
        ("tan", 0.7),
        ("asin", 0.3),
        ("acos", 0.3),
        ("atan", 1.5),
        ("sinh", 0.9),
        ("cosh", 0.9),
        ("tanh", 0.9),
        ("asinh", 1.2),
        ("acosh", 1.5),
        ("atanh", 0.4),
        ("cbrt", -8.0),
    ];
    for (method, x) in cases {
        let name = format!("core::f64::<impl f64>::{method}");
        let expected = match *method {
            "exp" => x.exp(),
            "exp2" => x.exp2(),
            "exp_m1" => x.exp_m1(),
            "ln" => x.ln(),
            "ln_1p" => x.ln_1p(),
            "log2" => x.log2(),
            "log10" => x.log10(),
            "sin" => x.sin(),
            "cos" => x.cos(),
            "tan" => x.tan(),
            "asin" => x.asin(),
            "acos" => x.acos(),
            "atan" => x.atan(),
            "sinh" => x.sinh(),
            "cosh" => x.cosh(),
            "tanh" => x.tanh(),
            "asinh" => x.asinh(),
            "acosh" => x.acosh(),
            "atanh" => x.atanh(),
            "cbrt" => x.cbrt(),
            _ => unreachable!(),
        };
        assert_eq!(
            run_method(&name, f64_ty(), vec![const_f64(*x)]),
            Value::Float(expected),
            "{method}({x})"
        );
    }
    // Exact anchors and NaN behaviour.
    assert_eq!(
        run_method("core::f64::<impl f64>::exp", f64_ty(), vec![const_f64(0.0)]),
        Value::Float(1.0)
    );
    assert_eq!(
        run_method("core::f64::<impl f64>::cbrt", f64_ty(), vec![const_f64(-8.0)]),
        Value::Float(-2.0)
    );
    match run_method(
        "core::f64::<impl f64>::ln",
        f64_ty(),
        vec![const_f64(-1.0)],
    ) {
        Value::Float(v) => assert!(v.is_nan(), "ln(-1) is NaN"),
        other => panic!("expected a float, got {other:?}"),
    }
    match run_method(
        "core::f64::<impl f64>::sin",
        f64_ty(),
        vec![const_f64(f64::INFINITY)],
    ) {
        Value::Float(v) => assert!(v.is_nan(), "sin(inf) is NaN"),
        other => panic!("expected a float, got {other:?}"),
    }
}

#[test]
fn f32_transcendental_methods_compute_at_f32_width() {
    assert_eq!(
        run_method("std::f32::<impl f32>::exp", f32_ty(), vec![const_f32(1.25)]),
        Value::Float(1.25f32.exp() as f64)
    );
    assert_eq!(
        run_method("std::f32::<impl f32>::sin", f32_ty(), vec![const_f32(0.7)]),
        Value::Float(0.7f32.sin() as f64)
    );
    assert_eq!(
        run_method("core::f32::<impl f32>::ln", f32_ty(), vec![const_f32(2.5)]),
        Value::Float(2.5f32.ln() as f64)
    );
    assert_eq!(
        run_method(
            "core::f32::<impl f32>::hypot",
            f32_ty(),
            vec![const_f32(3.0), const_f32(4.0)]
        ),
        Value::Float(5.0)
    );
}

#[test]
fn f64_log_atan2_hypot_div_euclid() {
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::log",
            f64_ty(),
            vec![const_f64(1000.0), const_f64(10.0)]
        ),
        Value::Float(1000.0f64.log(10.0))
    );
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::atan2",
            f64_ty(),
            vec![const_f64(1.0), const_f64(2.0)]
        ),
        Value::Float(1.0f64.atan2(2.0))
    );
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::hypot",
            f64_ty(),
            vec![const_f64(3.0), const_f64(4.0)]
        ),
        Value::Float(5.0)
    );
    // div_euclid rounds the quotient towards negative infinity,
    // unlike the truncating `/` operator.
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::div_euclid",
            f64_ty(),
            vec![const_f64(-7.5), const_f64(2.0)]
        ),
        Value::Float(-4.0)
    );
    // The adjustment keys on the (dividend-signed) remainder:
    // 7.5 % -2.0 is +1.5, so the truncated quotient stands.
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::div_euclid",
            f64_ty(),
            vec![const_f64(7.5), const_f64(-2.0)]
        ),
        Value::Float(-3.0)
    );
}

#[test]
fn f64_mul_add_is_the_fused_operation() {
    for (a, b, c) in [
        (0.1f64, 0.2f64, 0.3f64),
        (1.5, -2.25, 0.75),
        (1.0e300, 1.0e-300, -1.0),
    ] {
        assert_eq!(
            run_method(
                "core::f64::<impl f64>::mul_add",
                f64_ty(),
                vec![const_f64(a), const_f64(b), const_f64(c)]
            ),
            Value::Float(a.mul_add(b, c)),
            "mul_add({a}, {b}, {c})"
        );
    }
    assert_eq!(
        run_method(
            "std::f32::<impl f32>::mul_add",
            f32_ty(),
            vec![const_f32(1.5), const_f32(2.0), const_f32(0.5)]
        ),
        Value::Float(1.5f32.mul_add(2.0, 0.5) as f64)
    );
}

#[test]
fn f64_sin_cos_returns_the_pair() {
    let pair_ty = mir::Ty::Tuple(vec![f64_ty(), f64_ty()]);
    let (s, c) = 0.7f64.sin_cos();
    assert_eq!(
        run_method(
            "core::f64::<impl f64>::sin_cos",
            pair_ty,
            vec![const_f64(0.7)]
        ),
        Value::Tuple(vec![Value::Float(s), Value::Float(c)])
    );
    let pair32_ty = mir::Ty::Tuple(vec![f32_ty(), f32_ty()]);
    let (s, c) = 0.7f32.sin_cos();
    assert_eq!(
        run_method(
            "std::f32::<impl f32>::sin_cos",
            pair32_ty,
            vec![const_f32(0.7)]
        ),
        Value::Tuple(vec![
            Value::Float(s as f64),
            Value::Float(c as f64)
        ])
    );
}
