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
    // `sin_cos` returns a tuple, `exp` and friends are not modelled:
    // they must stay explicit unsupported calls, not silent matches.
    assert_eq!(lookup("core::f64::<impl f64>::sin_cos"), None);
    assert_eq!(lookup("std::f64::<impl f64>::exp"), None);
    assert_eq!(lookup("std::f32::<impl f32>::ln"), None);
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
        "std::f64::<impl f64>::exp".to_string(),
    )]);
    assert!(interpret_mir(&prog, "main", vec![]).is_err());
}
