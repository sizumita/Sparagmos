//! Per-body conversion isolation: one function whose body cannot
//! be converted must not fail the whole program. It is recorded as
//! unconverted with its reason; calling it fails with that reason,
//! and the entry function's own failure stays a whole-run
//! conversion failure.
//!
//! Background: in a stage-1 subset measurement, an unreachable
//! nested function holding a `dyn Trait` local failed the
//! conversion of the entire program, including a perfectly
//! convertible `main` (the shape of
//! `tests/ui/traits/object/nested-trait-method-call.rs`).

use sparagmos_interp::mir_convert::{
    convert_program, interpret_mir, interpret_mir_captured, ConvertError, MirProgram, MirRunError,
};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir (mirroring mir_convert_tests) -----

fn u32_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U32)
}

fn unit_ty() -> mir::Ty {
    mir::Ty::Tuple(Vec::new())
}

/// A coroutine witness type: outside the interpreter's type
/// model, so any body holding one as a local fails conversion.
/// (The original fixture used a `dyn Trait` local — the shape
/// that motivated per-body isolation — but `dyn` and coroutines
/// are modelled now, so the fixture uses a type that is still
/// unrepresentable.)
fn dyn_ty() -> mir::Ty {
    mir::Ty::CoroutineWitness(mir::DefId::new(0, 7), Vec::new())
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}

fn const_u32(v: u32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
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

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

/// A body that cannot convert: its second local is a coroutine.
fn broken_body() -> mir::Body {
    mir::Body::new(
        decls(&[unit_ty(), dyn_ty()]),
        0,
        vec![block(vec![], ret())],
    )
}

/// `fn main() -> u32 { _0 = 42; return }`
fn main_body() -> mir::Body {
    mir::Body::new(
        decls(&[u32_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_u32(42)))],
            ret(),
        )],
    )
}

const BROKEN_DEF: mir::DefId = mir::DefId::new(0, 42);

fn fndef_operand(def: mir::DefId) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def, Vec::new())),
        user_ty: None,
        ty: None,
    })
}

/// `fn main() { _1 = broken(); return }`
fn main_calling_broken_body() -> mir::Body {
    mir::Body::new(
        decls(&[unit_ty(), unit_ty()]),
        0,
        vec![
            block(
                vec![],
                mir::TerminatorKind::Call {
                    func: fndef_operand(BROKEN_DEF),
                    args: vec![],
                    destination: place(1),
                    target: Some(mir::BasicBlock(1)),
                    unwind: mir::UnwindAction::Continue,
                    call_source: mir::CallSource::Normal,
                },
            ),
            block(vec![], ret()),
        ],
    )
}

// ----- the isolation contract ------------------------------------------------

#[test]
fn uncalled_unconvertible_body_does_not_fail_the_program() {
    let mut prog = MirProgram::new();
    prog.add("main", main_body());
    prog.add("broken", broken_body());

    let converted = convert_program(&prog).expect("conversion is isolated per body");
    assert!(converted.get("main").is_some(), "main converts");
    assert!(
        converted.get("broken").is_none(),
        "broken has no converted body"
    );
    let reason = converted
        .unconverted_reason("broken")
        .expect("the failure is recorded with its reason");
    assert!(
        reason.contains("CoroutineWitness"),
        "the recorded reason names the unconvertible type: {reason}"
    );

    assert_eq!(
        interpret_mir(&prog, "main", vec![]).expect("main runs"),
        Value::Uint(42)
    );
}

#[test]
fn calling_an_unconvertible_body_fails_with_its_reason() {
    let mut prog = MirProgram::new();
    prog.add("main", main_calling_broken_body());
    prog.add_with_def_id("broken", BROKEN_DEF, broken_body());

    let err = interpret_mir(&prog, "main", vec![]).expect_err("the call must fail");
    match err {
        MirRunError::Interp(InterpError::UnconvertedFunction { name, reason }) => {
            assert_eq!(name, "broken");
            assert!(
                reason.contains("CoroutineWitness"),
                "the call error carries the conversion reason: {reason}"
            );
        }
        other => panic!("expected UnconvertedFunction for `broken`, got {other}"),
    }
}

#[test]
fn entry_conversion_failure_stays_a_whole_run_failure() {
    let mut prog = MirProgram::new();
    prog.add("main", broken_body());
    prog.add("helper", main_body());

    let err = interpret_mir(&prog, "main", vec![]).expect_err("entry failure is total");
    assert!(
        matches!(
            err,
            MirRunError::Convert(ConvertError::UnsupportedType { .. })
        ),
        "expected Convert(UnsupportedType), got {err}"
    );

    let captured = interpret_mir_captured(&prog, "main", vec![]);
    assert!(
        matches!(captured.result, Err(MirRunError::Convert(_))),
        "captured run reports the conversion failure: {:?}",
        captured.result
    );
    assert!(
        captured.stdout.is_empty(),
        "no guest output before conversion"
    );
}

#[test]
fn duplicate_names_are_rejected_even_when_one_body_is_broken() {
    let mut prog = MirProgram::new();
    prog.add("dup", broken_body());
    prog.add("dup", main_body());
    let err = convert_program(&prog).expect_err("duplicate names must fail");
    assert!(matches!(err, ConvertError::DuplicateFunction { .. }));
}
