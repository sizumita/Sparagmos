//! Joint-fixpoint tests: conversion, cleanliness and the
//! builtin-served set are iterated together in `convert_all`
//! (see `mir_convert::convert_all`). These hand-built programs
//! pin the shapes the old two-phase conversion withheld:
//!
//! * an impl body that becomes clean only once a trait call
//!   inside it resolves (a chain of impl bodies), which then
//!   serves as a resolution target itself;
//! * a body withheld only because it called a body that failed
//!   conversion, where the builtin model serves that callee;
//! * a trait call whose collected body is the trait item's
//!   (unconverted) default implementation, superseded by the
//!   clean impl body for the call's `Self` type;
//!
//! and the invariant that bounds all of it: a call naming an
//! unconverted body with no builtin model and no impl body
//! still fails with the recorded `UnconvertedFunction` error —
//! nothing is ever silently substituted.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram, MirRunError};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- items ---------------------------------------------------------------

const A_TY: mir::DefId = mir::DefId::new(0, 41);
const B_TY: mir::DefId = mir::DefId::new(0, 42);
const S_TY: mir::DefId = mir::DefId::new(0, 43);
const OUTER: mir::DefId = mir::DefId::new(0, 44);
const INNER: mir::DefId = mir::DefId::new(0, 45);
const MEASURE: mir::DefId = mir::DefId::new(0, 46);
const SIZE_OF: mir::DefId = mir::DefId::new(0, 47);
const PARTIAL_EQ: mir::DefId = mir::DefId::new(0, 48);
const UNMODELLED: mir::DefId = mir::DefId::new(0, 49);

// ----- small builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn usize_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::Usize)
}

fn u32_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U32)
}

fn a_ty() -> mir::Ty {
    mir::Ty::Adt(A_TY, Vec::new())
}

fn b_ty() -> mir::Ty {
    mir::Ty::Adt(B_TY, Vec::new())
}

fn s_ty() -> mir::Ty {
    mir::Ty::Adt(S_TY, Vec::new())
}

fn ref_ty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Not)
}

/// A local type the conversion rejects outright (a coroutine
/// witness: the value model has no representation for it): a
/// body holding one fails conversion as a whole and lands in
/// the unconverted table.
fn unconvertible_ty() -> mir::Ty {
    mir::Ty::CoroutineWitness(mir::DefId::new(0, 7), Vec::new())
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}

fn copy(idx: u32) -> mir::Operand {
    mir::Operand::Copy(place(idx))
}

fn move_(idx: u32) -> mir::Operand {
    mir::Operand::Move(place(idx))
}

fn const_i32(v: i32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u32 as u128,
            size: 4,
        },
    )))
}

fn const_bool(v: bool) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Bool(v)))
}

fn fndef_operand(def: mir::DefId, generic_args: Vec<mir::Ty>) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def, generic_args)),
        user_ty: None,
        ty: None,
    })
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

fn adt_aggregate(def: mir::DefId, ops: Vec<mir::Operand>) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def,
            variant_index: 0,
            generic_args: Vec::new(),
            user_ty: None,
            field_index: None,
        }),
        ops,
    )
}

fn ref_of(idx: u32) -> mir::Rvalue {
    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(idx))
}

/// `fn main() -> R { _1 = call f(); _0 = _1 }` for a nullary call.
fn main_calling(func: mir::Operand, ret_ty: mir::Ty) -> mir::Body {
    mir::Body::new(
        decls(&[ret_ty.clone(), ret_ty]),
        0,
        vec![
            block(Vec::new(), call(func, Vec::new(), 1, 1)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    )
}

// ----- the chain: an impl body that becomes clean one round in --------------

#[test]
fn impl_chain_resolves_to_full_depth() {
    // `<B as Inner>::inner` is clean from the start. `<A as
    // Outer>::outer` calls the `Inner` trait item: with resolution
    // off that call is unsupported (no builtin models the probe
    // trait), so `outer` is unclean in the first program and the
    // old two-phase conversion withheld it as a target — `main`'s
    // call to the `Outer` trait item stayed unsupported even
    // though `outer`'s second-phase form, its own call resolved
    // to `inner`, is perfectly clean. The fixpoint round that
    // recomputes cleanliness on the merged program promotes
    // `outer`, and `main` resolves to it.
    let inner_impl = mir::Body::new(
        decls(&[i32_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_i32(42)))],
            ret(),
        )],
    );
    let outer_impl = mir::Body::new(
        decls(&[i32_ty(), i32_ty()]),
        0,
        vec![
            block(
                Vec::new(),
                call(fndef_operand(INNER, vec![b_ty()]), Vec::new(), 1, 1),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.fn_names.insert(A_TY, "probe::A".to_string());
    prog.fn_names.insert(B_TY, "probe::B".to_string());
    prog.fn_names
        .insert(OUTER, "probe::Outer::outer".to_string());
    prog.fn_names
        .insert(INNER, "probe::Inner::inner".to_string());
    prog.add(
        "main",
        main_calling(fndef_operand(OUTER, vec![a_ty()]), i32_ty()),
    );
    prog.add("<probe::A as probe::Outer>::outer", outer_impl);
    prog.add("<probe::B as probe::Inner>::inner", inner_impl);

    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Int(42));
}

// ----- an unconverted callee the builtin model serves ------------------------

/// `std::mem::size_of` as a collected body that fails conversion
/// (it holds a coroutine-witness local). Calls naming it can only fail
/// with the recorded reason — unless the builtin model of
/// `size_of` serves them, as it serves callees the driver never
/// collected.
fn unconverted_size_of() -> mir::Body {
    mir::Body::new(
        decls(&[usize_ty(), unconvertible_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_usize(999)))],
            ret(),
        )],
    )
}

fn const_usize(v: u64) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
            size: 8,
        },
    )))
}

fn size_of_program() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names
        .insert(SIZE_OF, "std::mem::size_of".to_string());
    prog.add_with_def_id("std::mem::size_of", SIZE_OF, unconverted_size_of());
    prog
}

#[test]
fn call_to_unconverted_body_with_a_builtin_model_is_served_by_the_builtin() {
    // `main` calls the collected-but-unconverted `size_of`
    // directly. The builtin computes the function's own
    // semantics: 4 for `u32`, not the unconverted body's 999 and
    // not an error.
    let mut prog = size_of_program();
    prog.add(
        "main",
        main_calling(fndef_operand(SIZE_OF, vec![u32_ty()]), usize_ty()),
    );

    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Uint(4));
}

#[test]
fn impl_body_calling_an_unconverted_builtin_served_body_becomes_a_target() {
    // `<A as Measure>::measure` calls the unconverted `size_of`:
    // in the first program its direct call edge names an
    // unconverted body, so the impl body is unclean and withheld,
    // and `main`'s trait call stays unsupported (no builtin
    // models the probe trait). Once the `size_of` call is
    // builtin-served the impl body is clean, the next round
    // promotes it to a resolution target, and the trait call
    // runs it.
    let measure_impl = mir::Body::new(
        decls(&[usize_ty(), usize_ty()]),
        0,
        vec![
            block(
                Vec::new(),
                call(fndef_operand(SIZE_OF, vec![u32_ty()]), Vec::new(), 1, 1),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let mut prog = size_of_program();
    prog.fn_names.insert(A_TY, "probe::A".to_string());
    prog.fn_names
        .insert(MEASURE, "probe::Measure::measure".to_string());
    prog.add(
        "main",
        main_calling(fndef_operand(MEASURE, vec![a_ty()]), usize_ty()),
    );
    prog.add("<probe::A as probe::Measure>::measure", measure_impl);

    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Uint(4));
}

// ----- an unconverted default body superseded by the impl body ---------------

#[test]
fn trait_call_with_an_unconverted_default_body_resolves_to_the_impl() {
    // The collected body for the `PartialEq::eq` call is the
    // trait item's default implementation, and it does not
    // convert. Native execution of `S == S` dispatches to `S`'s
    // impl, not to the default body, so the clean impl body is
    // the faithful target: it returns `false` unconditionally,
    // while the builtin's structural comparison of two equal
    // unit values would give `true` and the default body would
    // raise its conversion failure. `false` proves the impl ran.
    let default_body = mir::Body::new(
        decls(&[
            mir::Ty::Bool,
            ref_ty(s_ty()),
            ref_ty(s_ty()),
            unconvertible_ty(),
        ]),
        2,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_bool(true)))],
            ret(),
        )],
    );
    let eq_impl = mir::Body::new(
        decls(&[mir::Ty::Bool, ref_ty(s_ty()), ref_ty(s_ty())]),
        2,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_bool(false)))],
            ret(),
        )],
    );
    // `fn main() -> bool { _1 = S; _2 = S; _3 = &_1; _4 = &_2;
    // _5 = PartialEq::eq(move _3, move _4); _0 = _5 }`
    let main = mir::Body::new(
        decls(&[
            mir::Ty::Bool,
            s_ty(),
            s_ty(),
            ref_ty(s_ty()),
            ref_ty(s_ty()),
            mir::Ty::Bool,
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(S_TY, Vec::new())),
                    assign(place(2), adt_aggregate(S_TY, Vec::new())),
                    assign(place(3), ref_of(1)),
                    assign(place(4), ref_of(2)),
                ],
                call(
                    fndef_operand(PARTIAL_EQ, vec![s_ty(), s_ty()]),
                    vec![move_(3), move_(4)],
                    5,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(5)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.fn_names.insert(S_TY, "probe::S".to_string());
    prog.fn_names
        .insert(PARTIAL_EQ, "std::cmp::PartialEq::eq".to_string());
    prog.add("main", main);
    prog.add_with_def_id("std::cmp::PartialEq::eq", PARTIAL_EQ, default_body);
    prog.add("<probe::S as std::cmp::PartialEq>::eq", eq_impl);

    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Bool(false));
}

// ----- the invariant: no silent substitution ---------------------------------

#[test]
fn call_to_unconverted_body_without_a_builtin_model_stays_an_error() {
    // `probe::unmodelled` does not convert and has neither a
    // builtin model nor an impl body: the call keeps naming the
    // unconverted body and fails with its recorded reason. The
    // fixpoint must never paper over this with a substitute.
    let body = mir::Body::new(
        decls(&[i32_ty(), unconvertible_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_i32(7)))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.fn_names
        .insert(UNMODELLED, "probe::unmodelled".to_string());
    prog.add_with_def_id("probe::unmodelled", UNMODELLED, body);
    prog.add(
        "main",
        main_calling(fndef_operand(UNMODELLED, Vec::new()), i32_ty()),
    );

    match interpret_mir(&prog, "main", Vec::new()) {
        Err(MirRunError::Interp(InterpError::UnconvertedFunction { name, .. })) => {
            assert_eq!(name, "probe::unmodelled");
        }
        other => panic!("expected the recorded unconverted-body error, got {other:?}"),
    }
}

// ----- demand-driven rounds: work skipped, failures unchanged ----------------

use sparagmos_interp::mir_convert::{conversions_performed, convert_program};

const HELPER: mir::DefId = mir::DefId::new(0, 50);
const MID: mir::DefId = mir::DefId::new(0, 51);

/// A three-body program in which every body converts in phase 1
/// and no resolution answer ever changes: `main` calls `caller`,
/// `caller` calls `helper`, and every call is a direct call to a
/// converted body.
fn all_clean_program() -> MirProgram {
    let helper = mir::Body::new(
        decls(&[i32_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_i32(7)))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.fn_names
        .insert(HELPER, "probe::helper".to_string());
    prog.fn_names.insert(MID, "probe::caller".to_string());
    prog.add_with_def_id("probe::helper", HELPER, helper);
    prog.add_with_def_id(
        "probe::caller",
        MID,
        main_calling(fndef_operand(HELPER, Vec::new()), i32_ty()),
    );
    prog.add(
        "main",
        main_calling(fndef_operand(MID, Vec::new()), i32_ty()),
    );
    prog
}

#[test]
fn run_mode_rounds_skip_bodies_whose_answers_never_change() {
    // Convert mode still converts `probe::caller` in its eager
    // round (six conversions: three bodies in phase 1, three in
    // the round), while the run entry point's demand-driven rounds
    // re-convert nothing: no body's resolution answers change, so
    // the worklist is empty and only the phase-1 conversions
    // happen. The programs the two paths produce are the same.
    let prog = all_clean_program();

    let before = conversions_performed();
    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    let run_conversions = conversions_performed() - before;
    assert_eq!(value, Value::Int(7));
    assert_eq!(
        run_conversions, 3,
        "run mode must convert each body once (phase 1 only)"
    );

    let before = conversions_performed();
    let converted = convert_program(&prog).expect("program converts");
    let convert_conversions = conversions_performed() - before;
    assert_eq!(converted.functions.len(), 3);
    assert!(converted.unconverted.is_empty());
    assert_eq!(
        convert_conversions, 6,
        "convert mode's eager round re-converts every body once"
    );
}

#[test]
fn demanded_unconverted_body_fails_the_same_through_the_rounds() {
    // `main` -> `probe::mid` -> `probe::unmodelled`, where the
    // last body does not convert and has no builtin model. The
    // demand-driven rounds do re-convert `mid` (its unconverted
    // answers change when `unmodelled` lands in the unconverted
    // set), and the run still fails with the recorded
    // `UnconvertedFunction` error for the demanded body — the
    // pre-change failure class, unsuperseded. Convert mode's
    // report keeps the body in the unconverted table.
    let unmodelled = mir::Body::new(
        decls(&[i32_ty(), unconvertible_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_i32(7)))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.fn_names
        .insert(UNMODELLED, "probe::unmodelled".to_string());
    prog.fn_names.insert(MID, "probe::mid".to_string());
    prog.add_with_def_id("probe::unmodelled", UNMODELLED, unmodelled);
    prog.add_with_def_id(
        "probe::mid",
        MID,
        main_calling(fndef_operand(UNMODELLED, Vec::new()), i32_ty()),
    );
    prog.add(
        "main",
        main_calling(fndef_operand(MID, Vec::new()), i32_ty()),
    );

    match interpret_mir(&prog, "main", Vec::new()) {
        Err(MirRunError::Interp(InterpError::UnconvertedFunction { name, .. })) => {
            assert_eq!(name, "probe::unmodelled");
        }
        other => panic!("expected the recorded unconverted-body error, got {other:?}"),
    }

    let converted = convert_program(&prog).expect("program converts");
    assert!(converted.unconverted.contains_key("probe::unmodelled"));
    assert!(converted.functions.contains_key("probe::mid"));
    assert!(converted.functions.contains_key("main"));
}
