//! Robustness regression tests: constructs the interpreter does not
//! support must fail with an explicit, reasoned error — never with a
//! host panic, an abort, or a silent skip.
//!
//! Background: in the first metric-2(b) measurement (2026-11-21),
//! unsupported guest constructs (explicit tail calls, coroutines,
//! async closures) surfaced as process failures rather than clean
//! `unsupported` results. At this layer the contract is that
//! conversion and execution return [`ConvertError`] /
//! [`InterpError`] values for every input-shaped failure; these tests
//! pin that behaviour for the construct families involved.

use sparagmos_interp::mir_convert::{
    convert_program, interpret_mir, ConvertError, MirProgram, MirRunError,
};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir (mirroring mir_convert_tests) -----

fn u8_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U8)
}

fn unit_ty() -> mir::Ty {
    mir::Ty::Tuple(Vec::new())
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}

fn const_u8(v: u8) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
            size: 1,
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

fn program_with(name: &str, body: mir::Body) -> MirProgram {
    let mut program = MirProgram::new();
    program.add(name, body);
    program
}

/// Run a single-function program and return its error, asserting the
/// run failed (a pass here would mean the construct was silently
/// accepted or executed).
fn run_err(name: &str, body: mir::Body) -> MirRunError {
    match interpret_mir(&program_with(name, body), name, Vec::new()) {
        Ok(value) => panic!("{name} must not execute successfully, got {value}"),
        Err(e) => e,
    }
}

// ----- unsupported terminators: explicit errors, not host panics -----------

#[test]
fn tail_call_is_an_explicit_error() {
    // `become f(..)` — the terminator rustc emits for explicit tail
    // calls. The callee operand is never resolved: conversion marks
    // the terminator unsupported before execution begins.
    let body = mir::Body::new(
        decls(&[unit_ty()]),
        0,
        vec![block(
            Vec::new(),
            mir::TerminatorKind::TailCall {
                func: const_u8(0),
                args: Vec::new(),
            },
        )],
    );
    let err = run_err("tail_call", body);
    assert_eq!(
        err,
        MirRunError::Interp(InterpError::UnsupportedTerminator {
            kind: "TailCall".to_string(),
        }),
        "tail call must surface as an unsupported-terminator error: {err}"
    );
}

#[test]
fn yield_is_an_explicit_error() {
    // The terminator at the heart of every coroutine/async body.
    let body = mir::Body::new(
        decls(&[unit_ty(), u8_ty()]),
        0,
        vec![block(
            Vec::new(),
            mir::TerminatorKind::Yield {
                value: const_u8(1),
                resume: mir::BasicBlock(0),
                resume_arg: place(1),
                drop: None,
            },
        )],
    );
    let err = run_err("yielder", body);
    assert_eq!(
        err,
        MirRunError::Interp(InterpError::UnsupportedTerminator {
            kind: "Yield".to_string(),
        }),
        "yield must surface as an unsupported-terminator error: {err}"
    );
}

#[test]
fn resume_and_coroutine_drop_are_explicit_errors() {
    for (name, kind, expected) in [
        ("resumer", mir::TerminatorKind::Resume, "Resume"),
        (
            "coroutine_drop",
            mir::TerminatorKind::CoroutineDrop,
            "CoroutineDrop",
        ),
    ] {
        let body = mir::Body::new(decls(&[unit_ty()]), 0, vec![block(Vec::new(), kind)]);
        let err = run_err(name, body);
        assert_eq!(
            err,
            MirRunError::Interp(InterpError::UnsupportedTerminator {
                kind: expected.to_string(),
            }),
            "{expected} must surface as an unsupported-terminator error: {err}"
        );
    }
}

// ----- unsupported local types: conversion errors with reasons -------------

fn conversion_err(body: mir::Body) -> ConvertError {
    match convert_program(&program_with("typed", body)) {
        Ok(_) => panic!("program with an unsupported local type must not convert"),
        Err(e) => e,
    }
}

fn assert_unsupported_type(body: mir::Body, kind_word: &str) {
    match conversion_err(body) {
        ConvertError::UnsupportedType { ty, .. } => {
            assert!(
                ty.contains(kind_word),
                "error must name the unsupported type kind {kind_word}: {ty}"
            );
        }
        other => panic!("expected UnsupportedType naming {kind_word}, got {other:?}"),
    }
}

fn body_with_local(ty: mir::Ty) -> mir::Body {
    mir::Body::new(
        decls(&[unit_ty(), ty]),
        0,
        vec![block(Vec::new(), mir::TerminatorKind::Return)],
    )
}

#[test]
fn coroutine_typed_local_fails_conversion_with_reason() {
    assert_unsupported_type(
        body_with_local(mir::Ty::Coroutine(mir::DefId::new(0, 1), Vec::new())),
        "Coroutine",
    );
}

#[test]
fn closure_typed_local_fails_conversion_with_reason() {
    assert_unsupported_type(
        body_with_local(mir::Ty::Closure(mir::DefId::new(0, 2), Vec::new())),
        "Closure",
    );
}

#[test]
fn dynamic_typed_local_fails_conversion_with_reason() {
    assert_unsupported_type(
        body_with_local(mir::Ty::Dynamic(
            vec!["std::fmt::Debug".to_string()],
            Box::new(unit_ty()),
            mir::Region::Erased,
        )),
        "Dynamic",
    );
}

#[test]
fn fnptr_typed_local_fails_conversion_with_reason() {
    assert_unsupported_type(
        body_with_local(mir::Ty::FnPtr(Box::new(mir::FnSig::new(
            Vec::new(),
            unit_ty(),
        )))),
        "FnPtr",
    );
}

// ----- Repeat aggregates: resource bound instead of an abort ---------------

#[test]
fn huge_repeat_is_an_explicit_error_not_an_abort() {
    // `[0u8; 1_000_001]`: materialising this aggregate is a resource
    // decision, not a semantics question. The conversion layer must
    // refuse it with a reason instead of attempting the allocation.
    let array_ty = mir::Ty::Array(Box::new(u8_ty()), mir::ArrayLen::Known(1_000_001));
    let body = mir::Body::new(
        decls(&[array_ty]),
        0,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Repeat(const_u8(0), 1_000_001),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    let err = run_err("huge_repeat", body);
    match err {
        MirRunError::Interp(InterpError::UnsupportedRvalue { kind }) => {
            assert!(
                kind.contains("Repeat of 1000001 elements"),
                "error must name the refused repeat count: {kind}"
            );
        }
        other => panic!("expected UnsupportedRvalue for the huge repeat, got {other:?}"),
    }
}

#[test]
fn small_repeat_still_executes() {
    // The bound must not refuse ordinary aggregates: `[7u8; 4]`.
    let array_ty = mir::Ty::Array(Box::new(u8_ty()), mir::ArrayLen::Known(4));
    let body = mir::Body::new(
        decls(&[array_ty]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Repeat(const_u8(7), 4))],
            mir::TerminatorKind::Return,
        )],
    );
    let mut program = MirProgram::new();
    program.add("small_repeat", body);
    let value = interpret_mir(&program, "small_repeat", Vec::new())
        .expect("a small repeat must execute");
    assert_eq!(
        value,
        Value::Tuple(vec![
            Value::Uint(7),
            Value::Uint(7),
            Value::Uint(7),
            Value::Uint(7)
        ])
    );
}
