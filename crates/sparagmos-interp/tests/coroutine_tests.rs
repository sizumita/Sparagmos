//! Coroutine state-object tests (Lane BZ): hand-built `sparagmos-mir`
//! bodies shaped after the driver's output for `gen` blocks and
//! `async` functions on Rust 1.100.0.
//!
//! The shape that matters: Stable MIR never exposes a `Yield`
//! terminator. rustc's coroutine transform has already run, so a
//! coroutine body is an ordinary function taking `&mut` to the state
//! object (in real programs the collected `Iterator::next` /
//! `Future::poll` body *is* this transformed body), which
//!   - dispatches on `Discriminant((*_1))` at entry,
//!   - reads the current state's fields through downcast projections
//!     `((*_1) as state).field`,
//!   - saves the next state's fields through the same projections,
//!   - sets the tag with `SetDiscriminant`, and
//!   - returns a `CoroutineState` / `Poll` aggregate.
//! The bodies below are called directly by `main` with that `&mut`;
//! the shim layer (`Option` wrapping in `next`, `Context` in `poll`)
//! is not part of the coroutine mechanism and is not modelled.
//!
//! State numbering follows the coroutine layout: 0 = unresumed, one
//! state per suspension point, then the returned state, then the
//! panicked state. Resuming a returned coroutine lands in the entry
//! switch's `otherwise` branch, which in real bodies calls
//! `core::panicking::panic` - modelled here exactly that way.

use std::collections::HashMap;

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram, MirRunError};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir ------------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn isize_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::Isize)
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}

/// `((*_arg) as variant).field` - a state field of the coroutine
/// behind the `&mut` in local 1 of a coroutine body.
fn state_field(variant: u32, field: u32, ty: mir::Ty) -> mir::Place {
    place(1).deref().downcast(None, variant).field(field, ty)
}

/// `(*_arg)` - the state object place for `SetDiscriminant`.
fn state_obj() -> mir::Place {
    place(1).deref()
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

fn const_isize(v: i128) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
            size: 8,
        },
    )))
}

fn const_unit() -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::ZeroSized))
}

fn const_str(s: &str) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Str(
        s.to_string(),
    )))
}

fn fndef_operand(def: mir::DefId, generic_args: Vec<mir::Ty>) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def, generic_args)),
        user_ty: None,
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

fn set_disc(place: mir::Place, variant: u32) -> mir::Statement {
    mir::Statement::new(mir::StatementKind::SetDiscriminant {
        place: Box::new(place),
        variant_index: variant,
    })
}

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

fn goto(bb: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Goto {
        target: mir::BasicBlock(bb),
    }
}

fn call(func: mir::Operand, args: Vec<mir::Operand>, dest: u32, next: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func,
        args,
        destination: place(dest),
        target: Some(mir::BasicBlock(next)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

fn call_diverging(func: mir::Operand, args: Vec<mir::Operand>, dest: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func,
        args,
        destination: place(dest),
        target: None,
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

fn switch(
    discr: mir::Operand,
    values: Vec<u128>,
    targets: Vec<u32>,
    otherwise: u32,
) -> mir::TerminatorKind {
    mir::TerminatorKind::SwitchInt {
        discr,
        targets: mir::SwitchTargets::new(
            values,
            targets.into_iter().map(mir::BasicBlock).collect(),
            mir::BasicBlock(otherwise),
        ),
    }
}

fn drop(place_: mir::Place, next: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Drop {
        place: place_,
        target: mir::BasicBlock(next),
        unwind: mir::UnwindAction::Continue,
        replace: false,
    }
}

fn ref_mut(idx: u32) -> mir::Rvalue {
    mir::Rvalue::Ref(
        mir::Region::Erased,
        mir::BorrowKind::Mut {
            allow_two_phase_borrow: false,
        },
        place(idx),
    )
}

fn ref_mut_place(p: mir::Place) -> mir::Rvalue {
    mir::Rvalue::Ref(
        mir::Region::Erased,
        mir::BorrowKind::Mut {
            allow_two_phase_borrow: false,
        },
        p,
    )
}

fn ref_ty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Mut)
}

fn aggregate_adt(
    def: mir::DefId,
    variant: u32,
    generic_args: Vec<mir::Ty>,
    ops: Vec<mir::Operand>,
) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def,
            variant_index: variant,
            generic_args,
            user_ty: None,
            field_index: None,
        }),
        ops,
    )
}

fn aggregate_coroutine(def: mir::DefId, generic_args: Vec<mir::Ty>, ops: Vec<mir::Operand>) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Coroutine {
            def,
            generic_args,
        }),
        ops,
    )
}

fn cast_int(op: mir::Operand, ty: mir::Ty) -> mir::Rvalue {
    mir::Rvalue::Cast(mir::CastKind::IntToInt, op, ty)
}

fn add(lhs: mir::Operand, rhs: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((lhs, rhs)))
}

fn mul(lhs: mir::Operand, rhs: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::BinaryOp(mir::BinOp::Mul, Box::new((lhs, rhs)))
}

// ----- the probe crate's items ------------------------------------------------

const GEN_DEF: mir::DefId = mir::DefId::new(0, 300);
const STATE_TY: mir::DefId = mir::DefId::new(0, 302);
const FUT_DEF: mir::DefId = mir::DefId::new(0, 310);
const POLL_TY: mir::DefId = mir::DefId::new(0, 312);
const OUTER_DEF: mir::DefId = mir::DefId::new(0, 330);
const INNER_DEF: mir::DefId = mir::DefId::new(0, 320);
const OUTER2_DEF: mir::DefId = mir::DefId::new(0, 340);
const D_TY: mir::DefId = mir::DefId::new(0, 350);
const DROP_GEN_DEF: mir::DefId = mir::DefId::new(0, 360);
const PANIC_DEF: mir::DefId = mir::DefId::new(0, 399);

const GEN_NAME: &str = "probe::gen_body";
const FUT_NAME: &str = "probe::fut_poll";
const OUTER_NAME: &str = "probe::outer_poll";
const INNER_NAME: &str = "probe::inner_poll";
const OUTER2_NAME: &str = "probe::outer2_poll";
const DROP_GEN_NAME: &str = "probe::drop_gen_body";
const D_DROP_NAME: &str = "<probe::D as std::ops::Drop>::drop";

fn probe_names() -> HashMap<mir::DefId, String> {
    HashMap::from([
        (GEN_DEF, GEN_NAME.to_string()),
        (FUT_DEF, FUT_NAME.to_string()),
        (OUTER_DEF, OUTER_NAME.to_string()),
        (INNER_DEF, INNER_NAME.to_string()),
        (OUTER2_DEF, OUTER2_NAME.to_string()),
        (DROP_GEN_DEF, DROP_GEN_NAME.to_string()),
        (D_TY, "probe::D".to_string()),
        (STATE_TY, "probe::CoroutineState".to_string()),
        (POLL_TY, "core::task::Poll".to_string()),
        (PANIC_DEF, "core::panicking::panic".to_string()),
    ])
}

/// The  a body is called by, for the bodies  calls
/// through  constants:  is what
/// lets the converter resolve those calls (a plain  records no
/// ). The destructor body is found by name, not by call, so
/// it needs no entry here.
fn body_def(name: &str) -> Option<mir::DefId> {
    match name {
        GEN_NAME => Some(GEN_DEF),
        FUT_NAME => Some(FUT_DEF),
        OUTER_NAME => Some(OUTER_DEF),
        INNER_NAME => Some(INNER_DEF),
        OUTER2_NAME => Some(OUTER2_DEF),
        DROP_GEN_NAME => Some(DROP_GEN_DEF),
        _ => None,
    }
}

fn run_program(main: mir::Body, extra: &[(&str, mir::Body)]) -> Result<Value, MirRunError> {
    let mut prog = MirProgram::new();
    prog.add("probe::main", main);
    for (name, body) in extra {
        match body_def(name) {
            Some(def) => prog.add_with_def_id(*name, def, body.clone()),
            None => prog.add(*name, body.clone()),
        };
    }
    prog.fn_names = probe_names();
    interpret_mir(&prog, "probe::main", vec![])
}

fn expect_panic(result: Result<Value, MirRunError>) -> String {
    match result {
        Err(MirRunError::Interp(InterpError::Panic { message, .. })) => message,
        other => panic!("expected a guest panic, got {other:?}"),
    }
}

/// The generic arguments of a coroutine type, in the driver's order:
/// (upvars, resume arg, yield type, return type, witness).
fn coroutine_args(upvars: Vec<mir::Ty>, yield_ty: mir::Ty) -> Vec<mir::Ty> {
    vec![
        mir::Ty::Tuple(upvars),
        mir::Ty::Tuple(vec![]),
        yield_ty,
        mir::Ty::unit(),
        mir::Ty::Tuple(vec![]),
    ]
}

fn state_ty() -> mir::Ty {
    mir::Ty::Adt(STATE_TY, vec![i32_ty(), mir::Ty::unit()])
}

fn poll_ty() -> mir::Ty {
    mir::Ty::Adt(POLL_TY, vec![i32_ty()])
}

/// The panic block every coroutine body ends with: the entry switch's
/// `otherwise` arm in the transformed body (resume of a returned or
/// panicked coroutine). `dest` is a unit scratch local.
fn panic_block(message: &'static str, dest: u32) -> mir::BasicBlockData {
    block(
        vec![],
        call_diverging(
            fndef_operand(PANIC_DEF, vec![]),
            vec![const_str(message)],
            dest,
        ),
    )
}

// ----- a gen block: yield a sequence, then return ------------------------------

fn gen_ty() -> mir::Ty {
    mir::Ty::Coroutine(GEN_DEF, coroutine_args(vec![i32_ty()], i32_ty()))
}

/// `gen` body for `gen { let mut n = start; yield n; n += 1; yield n; n += 1; yield n; }`:
/// the running value lives in the shared state slot 0 - the unresumed
/// state's upvar field first, then each suspension state's saved field.
/// States: 0 unresumed, 1..=3 suspended, 4 returned, 5 panicked.
/// Locals: _0 state result, _1 &mut state object, _2 discriminant,
/// _3 current value, _4 next value.
fn gen_body() -> mir::Body {
    let yielded = |v: mir::Operand| aggregate_adt(STATE_TY, 0, vec![i32_ty(), mir::Ty::unit()], vec![v]);
    mir::Body::new(
        decls(&[
            state_ty(),
            ref_ty(gen_ty()),
            isize_ty(),
            i32_ty(),
            i32_ty(),
            mir::Ty::unit(),
        ]),
        1,
        vec![
            // bb0: dispatch on the state discriminant.
            block(
                vec![assign(place(2), mir::Rvalue::Discriminant(state_obj()))],
                switch(copy(2), vec![0, 1, 2, 3], vec![1, 2, 3, 4], 5),
            ),
            // bb1 (state 0): read the upvar, save value+1 as state 1's
            // field, tag state 1, yield the upvar.
            block(
                vec![
                    assign(place(3), mir::Rvalue::Use(mir::Operand::Copy(state_field(0, 0, i32_ty())))),
                    assign(place(4), add(copy(3), const_i32(1))),
                    assign(state_field(1, 0, i32_ty()), mir::Rvalue::Use(copy(4))),
                    set_disc(state_obj(), 1),
                    assign(place(0), yielded(copy(3))),
                ],
                ret(),
            ),
            // bb2 (state 1): same step, into state 2.
            block(
                vec![
                    assign(place(3), mir::Rvalue::Use(mir::Operand::Copy(state_field(1, 0, i32_ty())))),
                    assign(place(4), add(copy(3), const_i32(1))),
                    assign(state_field(2, 0, i32_ty()), mir::Rvalue::Use(copy(4))),
                    set_disc(state_obj(), 2),
                    assign(place(0), yielded(copy(3))),
                ],
                ret(),
            ),
            // bb3 (state 2): yield the saved value, tag state 3.
            block(
                vec![
                    assign(place(3), mir::Rvalue::Use(mir::Operand::Copy(state_field(2, 0, i32_ty())))),
                    set_disc(state_obj(), 3),
                    assign(place(0), yielded(copy(3))),
                ],
                ret(),
            ),
            // bb4 (state 3): tag returned (4), return Complete.
            block(
                vec![
                    set_disc(state_obj(), 4),
                    assign(
                        place(0),
                        aggregate_adt(STATE_TY, 1, vec![i32_ty(), mir::Ty::unit()], vec![const_unit()]),
                    ),
                ],
                ret(),
            ),
            // bb5: resumed after completion.
            panic_block("generator resumed after completion", 5),
        ],
    )
}

#[test]
fn gen_yields_a_sequence_then_completes() {
    // fn main() -> i32 {
    //   _1 = gen(1); sum = 0;
    //   resume x3, adding each yielded value; resume once more:
    //   Complete -> return sum + 100; a Yielded there would read the
    //   Complete value's field through the wrong downcast and fail.
    // }
    let resume = |dest_block: u32| {
        block(vec![assign(place(2), ref_mut(1))], call(fndef_operand(GEN_DEF, coroutine_args(vec![i32_ty()], i32_ty())), vec![move_(2)], 3, dest_block))
    };
    let check_yielded = |add_block: u32, fail_block: u32| {
        block(
            vec![assign(place(5), mir::Rvalue::Discriminant(place(3)))],
            switch(copy(5), vec![0], vec![add_block], fail_block),
        )
    };
    let add_yielded = |next: u32| {
        block(
            vec![
                assign(place(6), mir::Rvalue::Use(mir::Operand::Copy(place(3).downcast(None, 0).field(0, i32_ty())))),
                assign(place(4), add(copy(4), copy(6))),
            ],
            goto(next),
        )
    };
    let main = mir::Body::new(
        decls(&[i32_ty(), gen_ty(), ref_ty(gen_ty()), state_ty(), i32_ty(), isize_ty(), i32_ty()]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), aggregate_coroutine(GEN_DEF, coroutine_args(vec![i32_ty()], i32_ty()), vec![const_i32(1)])),
                    assign(place(4), mir::Rvalue::Use(const_i32(0))),
                ],
                goto(1),
            ),
            resume(2),
            check_yielded(3, 13),
            add_yielded(4),
            resume(5),
            check_yielded(6, 13),
            add_yielded(7),
            resume(8),
            check_yielded(9, 13),
            add_yielded(10),
            resume(11),
            // Fourth resume: Complete (discriminant 1) is the only
            // acceptable outcome; a Yielded falls into bb13 and fails.
            block(
                vec![assign(place(5), mir::Rvalue::Discriminant(place(3)))],
                switch(copy(5), vec![1], vec![12], 13),
            ),
            block(vec![assign(place(0), add(copy(4), const_i32(100)))], ret()),
            block(vec![assign(place(0), mir::Rvalue::Use(const_i32(-1)))], ret()),
        ],
    );
    let value = run_program(main, &[(GEN_NAME, gen_body())]).expect("run");
    // Yields 1 + 2 + 3, plus 100 for completing: the saved state field
    // carried the running value across all three suspensions.
    assert_eq!(value, Value::Int(106));
}

#[test]
fn resuming_a_completed_gen_panics() {
    // fn main() -> i32 { _1 = gen(1); resume x5; return 0 }
    // The fifth resume dispatches on the returned state (4) and hits
    // the body's panic branch.
    let resume = |next: u32| {
        block(vec![assign(place(2), ref_mut(1))], call(fndef_operand(GEN_DEF, coroutine_args(vec![i32_ty()], i32_ty())), vec![move_(2)], 3, next))
    };
    let main = mir::Body::new(
        decls(&[i32_ty(), gen_ty(), ref_ty(gen_ty()), state_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), aggregate_coroutine(GEN_DEF, coroutine_args(vec![i32_ty()], i32_ty()), vec![const_i32(1)]))],
                goto(1),
            ),
            resume(2),
            resume(3),
            resume(4),
            resume(5),
            resume(6),
            block(vec![assign(place(0), mir::Rvalue::Use(const_i32(0)))], ret()),
        ],
    );
    let message = expect_panic(run_program(main, &[(GEN_NAME, gen_body())]));
    assert_eq!(message, "generator resumed after completion");
}

// ----- an async fn that never yields -------------------------------------------

fn fut_ty() -> mir::Ty {
    mir::Ty::Coroutine(FUT_DEF, coroutine_args(vec![], i32_ty()))
}

/// Poll body of `async fn fut() -> i32 { 42 }`: no suspension point,
/// so the layout is 0 unresumed, 1 returned, 2 panicked. The first
/// poll runs straight to `Ready`.
/// Locals: _0 poll result, _1 &mut state object, _2 discriminant,
/// _3 unit scratch for the panic call.
fn fut_poll() -> mir::Body {
    mir::Body::new(
        decls(&[poll_ty(), ref_ty(fut_ty()), isize_ty(), mir::Ty::unit()]),
        1,
        vec![
            block(
                vec![assign(place(2), mir::Rvalue::Discriminant(state_obj()))],
                switch(copy(2), vec![0], vec![1], 2),
            ),
            block(
                vec![
                    set_disc(state_obj(), 1),
                    assign(place(0), aggregate_adt(POLL_TY, 0, vec![i32_ty()], vec![const_i32(42)])),
                ],
                ret(),
            ),
            panic_block("async fn resumed after completion", 3),
        ],
    )
}

#[test]
fn async_fn_that_never_yields_is_ready_on_first_poll() {
    // fn main() -> isize {
    //   _1 = fut(); poll it once;
    //   return ready_value * 100 + poll_discriminant * 10 + fut_tag
    // }
    let main = mir::Body::new(
        decls(&[
            isize_ty(),
            fut_ty(),
            ref_ty(fut_ty()),
            poll_ty(),
            isize_ty(),
            i32_ty(),
            isize_ty(),
            isize_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), aggregate_coroutine(FUT_DEF, coroutine_args(vec![], i32_ty()), vec![])),
                    assign(place(2), ref_mut(1)),
                ],
                call(fndef_operand(FUT_DEF, coroutine_args(vec![], i32_ty())), vec![move_(2)], 3, 1),
            ),
            block(
                vec![
                    assign(place(4), mir::Rvalue::Discriminant(place(3))),
                    assign(place(5), mir::Rvalue::Use(mir::Operand::Copy(place(3).downcast(None, 0).field(0, i32_ty())))),
                    assign(place(6), mir::Rvalue::Discriminant(place(1))),
                    assign(place(7), cast_int(copy(5), isize_ty())),
                    assign(place(7), mul(copy(7), const_isize(100))),
                    assign(place(4), mul(copy(4), const_isize(10))),
                    assign(place(7), add(copy(7), copy(4))),
                    assign(place(0), add(copy(7), copy(6))),
                ],
                ret(),
            ),
        ],
    );
    let value = run_program(main, &[(FUT_NAME, fut_poll())]).expect("run");
    // Ready(42), poll discriminant 0, future tagged returned (1).
    assert_eq!(value, Value::Int(4201));
}

// ----- awaiting a ready future ---------------------------------------------------

fn outer_ty() -> mir::Ty {
    mir::Ty::Coroutine(OUTER_DEF, coroutine_args(vec![fut_ty()], i32_ty()))
}

/// Poll body of `async fn outer(inner: Fut) -> i32 { inner.await + 1 }`
/// where the inner future is ready on its first poll: the await runs
/// to completion inside the outer future's first poll.
/// Locals: _0 poll result, _1 &mut state object, _2 discriminant,
/// _3 &mut inner field, _4 inner poll result, _5 inner value,
/// _6 unit scratch.
fn outer_poll() -> mir::Body {
    mir::Body::new(
        decls(&[
            poll_ty(),
            ref_ty(outer_ty()),
            isize_ty(),
            ref_ty(fut_ty()),
            poll_ty(),
            i32_ty(),
            mir::Ty::unit(),
        ]),
        1,
        vec![
            block(
                vec![assign(place(2), mir::Rvalue::Discriminant(state_obj()))],
                switch(copy(2), vec![0], vec![1], 3),
            ),
            block(
                vec![assign(place(3), ref_mut_place(state_field(0, 0, fut_ty())))],
                call(fndef_operand(FUT_DEF, coroutine_args(vec![], i32_ty())), vec![move_(3)], 4, 2),
            ),
            block(
                vec![
                    assign(place(5), mir::Rvalue::Use(mir::Operand::Copy(place(4).downcast(None, 0).field(0, i32_ty())))),
                    set_disc(state_obj(), 1),
                    assign(place(5), add(copy(5), const_i32(1))),
                    assign(place(0), aggregate_adt(POLL_TY, 0, vec![i32_ty()], vec![copy(5)])),
                ],
                ret(),
            ),
            panic_block("async fn resumed after completion", 6),
        ],
    )
}

#[test]
fn awaiting_a_ready_future_completes_in_one_poll() {
    // fn main() -> isize {
    //   _1 = fut(); _2 = outer(_1); poll outer once;
    //   return value * 10 + poll_discriminant
    // }
    let main = mir::Body::new(
        decls(&[
            isize_ty(),
            fut_ty(),
            outer_ty(),
            ref_ty(outer_ty()),
            poll_ty(),
            isize_ty(),
            i32_ty(),
            isize_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), aggregate_coroutine(FUT_DEF, coroutine_args(vec![], i32_ty()), vec![])),
                    assign(place(2), aggregate_coroutine(OUTER_DEF, coroutine_args(vec![fut_ty()], i32_ty()), vec![move_(1)])),
                    assign(place(3), ref_mut(2)),
                ],
                call(fndef_operand(OUTER_DEF, coroutine_args(vec![fut_ty()], i32_ty())), vec![move_(3)], 4, 1),
            ),
            block(
                vec![
                    assign(place(5), mir::Rvalue::Discriminant(place(4))),
                    assign(place(6), mir::Rvalue::Use(mir::Operand::Copy(place(4).downcast(None, 0).field(0, i32_ty())))),
                    assign(place(7), cast_int(copy(6), isize_ty())),
                    assign(place(7), mul(copy(7), const_isize(10))),
                    assign(place(0), add(copy(7), copy(5))),
                ],
                ret(),
            ),
        ],
    );
    let value = run_program(main, &[(FUT_NAME, fut_poll()), (OUTER_NAME, outer_poll())]).expect("run");
    // Ready(42 + 1), poll discriminant 0.
    assert_eq!(value, Value::Int(430));
}

// ----- awaiting a future that returns Pending ------------------------------------

fn inner_ty() -> mir::Ty {
    mir::Ty::Coroutine(INNER_DEF, coroutine_args(vec![], i32_ty()))
}

/// Poll body of a future that is `Pending` on its first poll and
/// `Ready(10)` on its second. Layout: 0 unresumed, 1 suspended,
/// 2 returned, 3 panicked.
/// Locals: _0 poll result, _1 &mut state object, _2 discriminant,
/// _3 unit scratch.
fn inner_poll() -> mir::Body {
    mir::Body::new(
        decls(&[poll_ty(), ref_ty(inner_ty()), isize_ty(), mir::Ty::unit()]),
        1,
        vec![
            block(
                vec![assign(place(2), mir::Rvalue::Discriminant(state_obj()))],
                switch(copy(2), vec![0, 1], vec![1, 2], 3),
            ),
            block(
                vec![
                    set_disc(state_obj(), 1),
                    assign(place(0), aggregate_adt(POLL_TY, 1, vec![i32_ty()], vec![])),
                ],
                ret(),
            ),
            block(
                vec![
                    set_disc(state_obj(), 2),
                    assign(place(0), aggregate_adt(POLL_TY, 0, vec![i32_ty()], vec![const_i32(10)])),
                ],
                ret(),
            ),
            panic_block("async fn resumed after completion", 3),
        ],
    )
}

fn outer2_ty() -> mir::Ty {
    mir::Ty::Coroutine(OUTER2_DEF, coroutine_args(vec![inner_ty()], i32_ty()))
}

/// Poll body of `async fn outer2(inner: Inner) -> i32 { inner.await }`:
/// both the unresumed state and the suspended state resume by polling
/// the inner future stored in state field 0 - the inner state object
/// persists in the shared storage across the outer suspension.
/// Layout: 0 unresumed, 1 suspended, 2 returned, 3 panicked.
/// Locals: _0 poll result, _1 &mut state object, _2 discriminant,
/// _3 &mut inner field, _4 inner poll result, _5 inner discriminant,
/// _6 inner value, _7 unit scratch.
fn outer2_poll() -> mir::Body {
    mir::Body::new(
        decls(&[
            poll_ty(),
            ref_ty(outer2_ty()),
            isize_ty(),
            ref_ty(inner_ty()),
            poll_ty(),
            isize_ty(),
            i32_ty(),
            mir::Ty::unit(),
        ]),
        1,
        vec![
            block(
                vec![assign(place(2), mir::Rvalue::Discriminant(state_obj()))],
                switch(copy(2), vec![0, 1], vec![1, 1], 4),
            ),
            block(
                vec![assign(place(3), ref_mut_place(state_field(0, 0, inner_ty())))],
                call(fndef_operand(INNER_DEF, coroutine_args(vec![], i32_ty())), vec![move_(3)], 4, 2),
            ),
            block(
                vec![assign(place(5), mir::Rvalue::Discriminant(place(4)))],
                switch(copy(5), vec![0], vec![3], 5),
            ),
            // Inner Ready: take the value, tag returned, return Ready.
            block(
                vec![
                    assign(place(6), mir::Rvalue::Use(mir::Operand::Copy(place(4).downcast(None, 0).field(0, i32_ty())))),
                    set_disc(state_obj(), 2),
                    assign(place(0), aggregate_adt(POLL_TY, 0, vec![i32_ty()], vec![copy(6)])),
                ],
                ret(),
            ),
            panic_block("async fn resumed after completion", 7),
            // Inner Pending: tag suspended (1), return Pending. The
            // inner future stays in the state field for the next poll.
            block(
                vec![
                    set_disc(state_obj(), 1),
                    assign(place(0), aggregate_adt(POLL_TY, 1, vec![i32_ty()], vec![])),
                ],
                ret(),
            ),
        ],
    )
}

#[test]
fn awaiting_a_pending_future_suspends_and_resumes() {
    // fn main() -> isize {
    //   _1 = inner(); _2 = outer2(_1);
    //   p1 = poll outer2;            // Pending
    //   p2 = poll outer2;            // Ready(10)
    //   return outer_tag * 1000 + discr(p1) * 100 + discr(p2) * 10 + value(p2)
    // }
    let main = mir::Body::new(
        decls(&[
            isize_ty(),
            inner_ty(),
            outer2_ty(),
            ref_ty(outer2_ty()),
            poll_ty(),
            poll_ty(),
            isize_ty(),
            isize_ty(),
            i32_ty(),
            isize_ty(),
            isize_ty(),
            isize_ty(),
            isize_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), aggregate_coroutine(INNER_DEF, coroutine_args(vec![], i32_ty()), vec![])),
                    assign(place(2), aggregate_coroutine(OUTER2_DEF, coroutine_args(vec![inner_ty()], i32_ty()), vec![move_(1)])),
                    assign(place(3), ref_mut(2)),
                ],
                call(fndef_operand(OUTER2_DEF, coroutine_args(vec![inner_ty()], i32_ty())), vec![move_(3)], 4, 1),
            ),
            block(
                vec![
                    assign(place(6), mir::Rvalue::Discriminant(place(4))),
                    assign(place(3), ref_mut(2)),
                ],
                call(fndef_operand(OUTER2_DEF, coroutine_args(vec![inner_ty()], i32_ty())), vec![move_(3)], 5, 2),
            ),
            block(
                vec![
                    assign(place(7), mir::Rvalue::Discriminant(place(5))),
                    assign(place(8), mir::Rvalue::Use(mir::Operand::Copy(place(5).downcast(None, 0).field(0, i32_ty())))),
                    assign(place(9), mir::Rvalue::Discriminant(place(2))),
                    assign(place(10), cast_int(copy(8), isize_ty())),
                    assign(place(11), mul(copy(9), const_isize(1000))),
                    assign(place(12), mul(copy(6), const_isize(100))),
                    assign(place(11), add(copy(11), copy(12))),
                    assign(place(12), mul(copy(7), const_isize(10))),
                    assign(place(11), add(copy(11), copy(12))),
                    assign(place(0), add(copy(11), copy(10))),
                ],
                ret(),
            ),
        ],
    );
    let value = run_program(main, &[(INNER_NAME, inner_poll()), (OUTER2_NAME, outer2_poll())]).expect("run");
    // Outer tagged returned (2), first poll Pending (1), second poll
    // Ready (0) with value 10: the inner future's own state survived
    // inside the suspended outer future.
    assert_eq!(value, Value::Int(2110));
}

// ----- dropping a suspended coroutine runs the saved fields' glue -----------------

fn d_ty() -> mir::Ty {
    mir::Ty::Adt(D_TY, vec![])
}

/// `D`'s destructor: `*self.sink = *self.sink * 10 + self.id`, the
/// observable channel of `drop_glue_tests`.
fn bump_dtor() -> mir::Body {
    mir::Body::new(
        decls(&[
            mir::Ty::unit(),
            ref_ty(d_ty()),
            ref_ty(i32_ty()),
            i32_ty(),
            i32_ty(),
            i32_ty(),
        ]),
        1,
        vec![block(
            vec![
                assign(place(2), mir::Rvalue::Use(mir::Operand::Copy(place(1).deref().field(0, ref_ty(i32_ty()))))),
                assign(place(3), mir::Rvalue::Use(mir::Operand::Copy(place(1).deref().field(1, i32_ty())))),
                assign(place(4), mir::Rvalue::Use(mir::Operand::Copy(place(2).deref()))),
                assign(place(5), mul(copy(4), const_i32(10))),
                assign(place(5), add(copy(5), copy(3))),
                assign(place(2).deref(), mir::Rvalue::Use(copy(5))),
            ],
            ret(),
        )],
    )
}

fn drop_gen_ty() -> mir::Ty {
    mir::Ty::Coroutine(DROP_GEN_DEF, coroutine_args(vec![d_ty()], i32_ty()))
}

/// Body of a gen whose upvar is a droppable `D`: the first resume
/// moves the upvar into the suspension state's field (the shared
/// storage slot, addressed through the state-1 downcast) and yields.
/// Locals: _0 state result, _1 &mut state object, _2 discriminant,
/// _3 unit scratch.
fn drop_gen_body() -> mir::Body {
    mir::Body::new(
        decls(&[state_ty(), ref_ty(drop_gen_ty()), isize_ty(), mir::Ty::unit()]),
        1,
        vec![
            block(
                vec![assign(place(2), mir::Rvalue::Discriminant(state_obj()))],
                switch(copy(2), vec![0], vec![1], 2),
            ),
            block(
                vec![
                    assign(state_field(1, 0, d_ty()), mir::Rvalue::Use(mir::Operand::Move(state_field(0, 0, d_ty())))),
                    set_disc(state_obj(), 1),
                    assign(place(0), aggregate_adt(STATE_TY, 0, vec![i32_ty(), mir::Ty::unit()], vec![const_i32(0)])),
                ],
                ret(),
            ),
            block(
                vec![
                    set_disc(state_obj(), 2),
                    assign(place(0), aggregate_adt(STATE_TY, 1, vec![i32_ty(), mir::Ty::unit()], vec![const_unit()])),
                ],
                ret(),
            ),
        ],
    )
}

#[test]
fn dropping_a_suspended_coroutine_runs_saved_field_glue() {
    // fn main() -> i32 {
    //   counter = 0; d = D { &mut counter, 7 }; g = gen(d);
    //   resume g once (suspends with D saved in the state field);
    //   drop(g); return counter;
    // }
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            drop_gen_ty(),
            ref_ty(drop_gen_ty()),
            state_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_mut(1)),
                    assign(place(3), aggregate_adt(D_TY, 0, vec![], vec![move_(2), const_i32(7)])),
                    assign(place(4), aggregate_coroutine(DROP_GEN_DEF, coroutine_args(vec![d_ty()], i32_ty()), vec![move_(3)])),
                    assign(place(5), ref_mut(4)),
                ],
                call(fndef_operand(DROP_GEN_DEF, coroutine_args(vec![d_ty()], i32_ty())), vec![move_(5)], 6, 1),
            ),
            block(vec![], drop(place(4), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_program(
        main,
        &[(DROP_GEN_NAME, drop_gen_body()), (D_DROP_NAME, bump_dtor())],
    )
    .expect("run");
    // The suspended state's field held D; dropping the coroutine ran
    // D's destructor exactly once: 0 * 10 + 7.
    assert_eq!(value, Value::Int(7));
}
