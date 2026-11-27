//! `FnOnce` by-value environment tests: hand-built `sparagmos-mir`
//! bodies shaped after real `rustc 1.100.0` output for closures whose
//! own kind is `FnOnce`, converted and executed by the interpreter.
//!
//! The shape that distinguishes these bodies from the `Fn` / `FnMut`
//! ones in `callable_tests.rs`: the driver resolves a closure body
//! with the closure's own kind, so no `FnOnce` adapter shim exists in
//! the program. An `FnOnce` closure's body therefore takes the
//! closure value *itself* as its first argument (not a reference to
//! it) and reads its captures with plain field projections on that
//! value (`_1.0`), moving them out when the closure consumes them.
//! `FnOnce::call_once` passes the closure by value, so the call
//! consumes the caller's place: afterwards the place is
//! uninitialized, exactly as after any other move.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram, MirRunError};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir (mirroring callable_tests) ---------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn fnptr_ty(inputs: Vec<mir::Ty>, output: mir::Ty) -> mir::Ty {
    mir::Ty::FnPtr(Box::new(mir::FnSig::new(inputs, output)))
}

/// The closure type as the driver produces it: generic arguments are
/// (kind byte, signature, upvars tuple).
fn closure_ty(def: mir::DefId, params: Vec<mir::Ty>, upvars: Vec<mir::Ty>) -> mir::Ty {
    mir::Ty::Closure(
        def,
        vec![
            mir::Ty::Int(mir::IntTy::I8),
            fnptr_ty(vec![mir::Ty::Tuple(params)], i32_ty()),
            mir::Ty::Tuple(upvars),
        ],
    )
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

fn fndef_operand(def: mir::DefId, generic_args: Vec<mir::Ty>) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def, generic_args)),
        user_ty: None,
        ty: None,
        generic_type_ids: None,
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

fn add(lhs: mir::Operand, rhs: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((lhs, rhs)))
}

fn closure_aggregate(
    def: mir::DefId,
    clo_ty: &mir::Ty,
    operands: Vec<mir::Operand>,
) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Closure {
            def,
            generic_args: match clo_ty {
                mir::Ty::Closure(_, args) => args.clone(),
                _ => unreachable!(),
            },
        }),
        operands,
    )
}

const CLOSURE: mir::DefId = mir::DefId { krate: 0, index: 1 };
const FN_CALL_ONCE: mir::DefId = mir::DefId { krate: 0, index: 6 };
const APPLY: mir::DefId = mir::DefId { krate: 0, index: 7 };

fn program_with_std_call_names() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names
        .insert(FN_CALL_ONCE, "std::ops::FnOnce::call_once".to_string());
    prog
}

fn add_closure(prog: &mut MirProgram, body: mir::Body) {
    prog.fn_names
        .insert(CLOSURE, "main::{closure#0}".to_string());
    prog.add_with_def_id("main::{closure#0}", CLOSURE, body);
}

// ----- the tests --------------------------------------------------------------

#[test]
fn fnonce_moves_aggregate_capture_out_of_env() {
    // The closure captures one aggregate (a tuple standing in for a
    // Vec-like value) and returns it by moving it out of the
    // environment: `{closure#0}(_1: {closure}) -> (i32, i32)` with
    // `_0 = move _1.0`.
    let pair_ty = mir::Ty::Tuple(vec![i32_ty(), i32_ty()]);
    let clo_ty = closure_ty(CLOSURE, vec![], vec![pair_ty.clone()]);
    let body = mir::Body::new(
        decls(&[pair_ty.clone(), clo_ty.clone()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Use(mir::Operand::Move(place(1).field(0, pair_ty.clone()))),
            )],
            ret(),
        )],
    );
    // main: _1 = (40, 2); _2 = Closure(move _1); _3 = ();
    //       _4 = call_once(move _2, move _3); _0 = _4.0 + _4.1
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            pair_ty.clone(),
            clo_ty.clone(),
            mir::Ty::Tuple(vec![]),
            pair_ty.clone(),
            i32_ty(),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![const_i32(40), const_i32(2)],
                        ),
                    ),
                    assign(
                        place(2),
                        closure_aggregate(CLOSURE, &clo_ty, vec![move_(1)]),
                    ),
                    assign(
                        place(3),
                        mir::Rvalue::Aggregate(Box::new(mir::AggregateKind::Tuple), vec![]),
                    ),
                ],
                call(
                    fndef_operand(FN_CALL_ONCE, vec![clo_ty.clone(), mir::Ty::Tuple(vec![])]),
                    vec![move_(2), move_(3)],
                    4,
                    1,
                ),
            ),
            block(
                vec![
                    assign(
                        place(5),
                        mir::Rvalue::Use(mir::Operand::Copy(place(4).field(0, i32_ty()))),
                    ),
                    assign(
                        place(6),
                        mir::Rvalue::Use(mir::Operand::Copy(place(4).field(1, i32_ty()))),
                    ),
                    assign(place(0), add(copy(5), copy(6))),
                ],
                ret(),
            ),
        ],
    );
    let mut prog = program_with_std_call_names();
    add_closure(&mut prog, body);
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}

#[test]
fn fnonce_by_value_env_with_argument() {
    // `{closure#0}(_1: {closure}, _2: i32) -> i32` with
    // `_0 = _1.0 + _2`: the capture (40) is read out of the by-value
    // environment, the argument (2) is the closure's parameter.
    let clo_ty = closure_ty(CLOSURE, vec![i32_ty()], vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone(), i32_ty()]),
        2,
        vec![block(
            vec![assign(
                place(0),
                add(mir::Operand::Copy(place(1).field(0, i32_ty())), copy(2)),
            )],
            ret(),
        )],
    );
    // main: _1 = Closure(const 40); _2 = (2,);
    //       _3 = call_once(move _1, move _2); _0 = copy _3
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            clo_ty.clone(),
            mir::Ty::Tuple(vec![i32_ty()]),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        closure_aggregate(CLOSURE, &clo_ty, vec![const_i32(40)]),
                    ),
                    assign(
                        place(2),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![const_i32(2)],
                        ),
                    ),
                ],
                call(
                    fndef_operand(
                        FN_CALL_ONCE,
                        vec![clo_ty.clone(), mir::Ty::Tuple(vec![i32_ty()])],
                    ),
                    vec![move_(1), move_(2)],
                    3,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(3)))], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    add_closure(&mut prog, body);
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}

#[test]
fn fnonce_called_through_generic_parameter() {
    // `apply(_1: {closure}) -> i32` — the shape a monomorphised
    // `fn apply<F: FnOnce() -> i32>(f: F) -> i32 { f() }` has: the
    // parameter place holds the closure value and `f()` is
    // `call_once(move _1, ())` inside `apply`.
    let clo_ty = closure_ty(CLOSURE, vec![], vec![i32_ty()]);
    // {closure#0}(_1: {closure}) -> i32: _0 = _1.0 + 1
    let body = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                add(
                    mir::Operand::Copy(place(1).field(0, i32_ty())),
                    const_i32(1),
                ),
            )],
            ret(),
        )],
    );
    // apply: _2 = (); _3 = call_once(move _1, move _2); _0 = copy _3
    let apply = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone(), mir::Ty::Tuple(vec![]), i32_ty()]),
        1,
        vec![
            block(
                vec![assign(
                    place(2),
                    mir::Rvalue::Aggregate(Box::new(mir::AggregateKind::Tuple), vec![]),
                )],
                call(
                    fndef_operand(FN_CALL_ONCE, vec![clo_ty.clone(), mir::Ty::Tuple(vec![])]),
                    vec![move_(1), move_(2)],
                    3,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(3)))], ret()),
        ],
    );
    // main: _1 = Closure(const 41); _0 = apply(move _1)
    let main = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone()]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    closure_aggregate(CLOSURE, &clo_ty, vec![const_i32(41)]),
                )],
                call(
                    fndef_operand(APPLY, vec![clo_ty.clone()]),
                    vec![move_(1)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    add_closure(&mut prog, body);
    prog.add_with_def_id("apply", APPLY, apply);
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}

#[test]
fn fnonce_move_closure_passed_to_higher_order_fn() {
    // `hof(_1: {closure}, _2: i32) -> i32` calls its FnOnce parameter
    // once with its own argument: _3 = (copy _2);
    // _4 = call_once(move _1, move _3); _0 = copy _4.
    // The closure adds its by-value capture (35) to the argument (7).
    let clo_ty = closure_ty(CLOSURE, vec![i32_ty()], vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone(), i32_ty()]),
        2,
        vec![block(
            vec![assign(
                place(0),
                add(mir::Operand::Copy(place(1).field(0, i32_ty())), copy(2)),
            )],
            ret(),
        )],
    );
    let hof = mir::Body::new(
        decls(&[
            i32_ty(),
            clo_ty.clone(),
            i32_ty(),
            mir::Ty::Tuple(vec![i32_ty()]),
            i32_ty(),
        ]),
        2,
        vec![
            block(
                vec![assign(
                    place(3),
                    mir::Rvalue::Aggregate(Box::new(mir::AggregateKind::Tuple), vec![copy(2)]),
                )],
                call(
                    fndef_operand(
                        FN_CALL_ONCE,
                        vec![clo_ty.clone(), mir::Ty::Tuple(vec![i32_ty()])],
                    ),
                    vec![move_(1), move_(3)],
                    4,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(4)))], ret()),
        ],
    );
    // main: _1 = Closure(const 35); _0 = hof(move _1, 7)
    let main = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone()]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    closure_aggregate(CLOSURE, &clo_ty, vec![const_i32(35)]),
                )],
                call(
                    fndef_operand(APPLY, vec![clo_ty.clone(), i32_ty()]),
                    vec![move_(1), const_i32(7)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    add_closure(&mut prog, body);
    prog.add_with_def_id("hof", APPLY, hof);
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}

#[test]
fn fnonce_consumed_closure_cannot_be_read_again() {
    // After `call_once(move _1, ..)` the closure place is moved-from:
    // reading it again is an uninitialized read, the same error any
    // other moved place produces. (Real MIR never contains this —
    // borrowck forbids it — but the interpreter must not silently
    // resurrect the value if the shape appears.)
    let clo_ty = closure_ty(CLOSURE, vec![], vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Use(mir::Operand::Move(place(1).field(0, i32_ty()))),
            )],
            ret(),
        )],
    );
    // main: _1 = Closure(const 41); _2 = ();
    //       _3 = call_once(move _1, move _2); _0 = copy _1 (use after move)
    let main = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone(), mir::Ty::Tuple(vec![]), i32_ty()]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        closure_aggregate(CLOSURE, &clo_ty, vec![const_i32(41)]),
                    ),
                    assign(
                        place(2),
                        mir::Rvalue::Aggregate(Box::new(mir::AggregateKind::Tuple), vec![]),
                    ),
                ],
                call(
                    fndef_operand(FN_CALL_ONCE, vec![clo_ty.clone(), mir::Ty::Tuple(vec![])]),
                    vec![move_(1), move_(2)],
                    3,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    add_closure(&mut prog, body);
    prog.add("main", main);
    let err = interpret_mir(&prog, "main", vec![]).expect_err("use after move must fail");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UninitializedRead { .. })
        ),
        "expected an uninitialized read of the consumed closure, got: {err}"
    );
}
