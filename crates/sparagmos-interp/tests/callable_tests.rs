//! Callable-value tests: hand-built `sparagmos-mir` bodies shaped
//! after real `rustc 1.100.0 --emit=mir` output for closures,
//! function pointers and function items, converted and executed by
//! the interpreter and compared against independently known values.
//!
//! Shapes reproduced from real MIR:
//! * a closure value is an `AggregateKind::Closure` over its capture
//!   operands; the closure body takes a reference to the closure
//!   value as its first argument and reads captures through
//!   `(*env).<field>` (with a further deref for by-reference
//!   captures);
//! * calling a closure is a call to `std::ops::Fn::call` /
//!   `FnMut::call_mut` / `FnOnce::call_once` with (receiver, args
//!   tuple) arguments;
//! * a function item becomes a function pointer through
//!   `Cast(PointerCoercion(ReifyFnPointer), const FnDef, FnPtr)` and
//!   is called through the place holding it.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram, MirRunError};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn ref_ty(inner: mir::Ty, mutable: bool) -> mir::Ty {
    mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(inner),
        if mutable {
            mir::Mutability::Mut
        } else {
            mir::Mutability::Not
        },
    )
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

fn add(lhs: mir::Operand, rhs: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((lhs, rhs)))
}

// ----- shared bodies ---------------------------------------------------------

const TWICE: mir::DefId = mir::DefId { krate: 0, index: 0 };
const CLOSURE: mir::DefId = mir::DefId { krate: 0, index: 1 };
const FN_CALL: mir::DefId = mir::DefId { krate: 0, index: 4 };
const FN_CALL_MUT: mir::DefId = mir::DefId { krate: 0, index: 5 };
const FN_CALL_ONCE: mir::DefId = mir::DefId { krate: 0, index: 6 };
const APPLY: mir::DefId = mir::DefId { krate: 0, index: 7 };

/// `fn twice(_1: i32) -> i32 { _0 = _1 * 2; return }`
fn twice_body() -> mir::Body {
    mir::Body::new(
        decls(&[i32_ty(), i32_ty()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::BinaryOp(mir::BinOp::Mul, Box::new((copy(1), const_i32(2)))),
            )],
            ret(),
        )],
    )
}

fn program_with_std_call_names() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names
        .insert(FN_CALL, "std::ops::Fn::call".to_string());
    prog.fn_names
        .insert(FN_CALL_MUT, "std::ops::FnMut::call_mut".to_string());
    prog.fn_names
        .insert(FN_CALL_ONCE, "std::ops::FnOnce::call_once".to_string());
    prog
}

// ----- function pointers -----------------------------------------------------

#[test]
fn fn_pointer_reified_and_called_through_place() {
    // main: _1 = Cast(ReifyFnPointer, const twice, fn(i32) -> i32);
    //       _2 = call copy _1 (21); _0 = copy _2; return
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            fnptr_ty(vec![i32_ty()], i32_ty()),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    mir::Rvalue::Cast(
                        mir::CastKind::PointerCoercion(mir::PointerCoercion::ReifyFnPointer),
                        fndef_operand(TWICE, vec![]),
                        fnptr_ty(vec![i32_ty()], i32_ty()),
                    ),
                )],
                call(copy(1), vec![const_i32(21)], 2, 1),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    prog.add_with_def_id("twice", TWICE, twice_body());
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}

#[test]
fn fn_item_value_called_through_fn_trait() {
    // main: _1 = const twice (fn item value);
    //       _2 = &_1; _4 = Fn::call(move _2, (21,)) via tuple _3; _0 = copy _4
    let item_ty = mir::Ty::FnDef(TWICE, vec![]);
    let tuple_ty = mir::Ty::Tuple(vec![i32_ty()]);
    let main = mir::Body::new(
        decls(&[i32_ty(), item_ty.clone(), ref_ty(item_ty, false), tuple_ty, i32_ty()]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(fndef_operand(TWICE, vec![]))),
                    assign(
                        place(2),
                        mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
                    ),
                    assign(
                        place(3),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![const_i32(21)],
                        ),
                    ),
                ],
                call(
                    fndef_operand(
                        FN_CALL,
                        vec![mir::Ty::FnDef(TWICE, vec![]), mir::Ty::Tuple(vec![i32_ty()])],
                    ),
                    vec![move_(2), move_(3)],
                    4,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(4)))], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    prog.add_with_def_id("twice", TWICE, twice_body());
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}

#[test]
fn higher_order_function_takes_fn_pointer() {
    // apply(_1: fn(i32) -> i32, _2: i32) -> i32 { _3 = _1(_2); _0 = _3 + 1 }
    let apply = mir::Body::new(
        decls(&[
            i32_ty(),
            fnptr_ty(vec![i32_ty()], i32_ty()),
            i32_ty(),
            i32_ty(),
        ]),
        2,
        vec![
            block(vec![], call(copy(1), vec![copy(2)], 3, 1)),
            block(
                vec![assign(place(0), add(copy(3), const_i32(1)))],
                ret(),
            ),
        ],
    );
    // main: _1 = reify twice; _0 = apply(move _1, 14); return
    let main = mir::Body::new(
        decls(&[i32_ty(), fnptr_ty(vec![i32_ty()], i32_ty())]),
        0,
        vec![block(
            vec![assign(
                place(1),
                mir::Rvalue::Cast(
                    mir::CastKind::PointerCoercion(mir::PointerCoercion::ReifyFnPointer),
                    fndef_operand(TWICE, vec![]),
                    fnptr_ty(vec![i32_ty()], i32_ty()),
                ),
            )],
            call(fndef_operand(APPLY, vec![]), vec![move_(1), const_i32(14)], 0, 1),
        )],
    );
    // The direct call's destination is _0 and its continuation is a
    // bare return block.
    let main = {
        let mut blocks = main.basic_blocks.clone();
        blocks.push(block(vec![], ret()));
        mir::Body::new(main.local_decls.clone(), 0, blocks)
    };
    let mut prog = program_with_std_call_names();
    prog.add_with_def_id("twice", TWICE, twice_body());
    prog.add_with_def_id("apply", APPLY, apply);
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(29));
}

// ----- closures --------------------------------------------------------------

/// `{closure#0}(_1: &{closure}, _2: i32) -> i32` for
/// `|x| x + base` with `base: i32` captured by reference:
/// `_0 = _2 + *((*_1).0)`.
fn add_base_closure_body() -> mir::Body {
    let env_ty = ref_ty(closure_ty(CLOSURE, vec![i32_ty()], vec![ref_ty(i32_ty(), false)]), false);
    let captured = place(1)
        .deref()
        .field(0, ref_ty(i32_ty(), false))
        .deref();
    mir::Body::new(
        decls(&[i32_ty(), env_ty, i32_ty()]),
        2,
        vec![block(
            vec![assign(place(0), add(copy(2), mir::Operand::Copy(captured)))],
            ret(),
        )],
    )
}

#[test]
fn closure_capture_by_ref_called_via_fn_call() {
    // main: _1 = 40; _3 = &_1; _2 = Closure(move _3); _4 = &_2;
    //       _5 = (2,); _6 = Fn::call(move _4, move _5); _0 = copy _6
    let clo_ty = closure_ty(CLOSURE, vec![i32_ty()], vec![ref_ty(i32_ty(), false)]);
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            clo_ty.clone(),
            ref_ty(i32_ty(), false),
            ref_ty(clo_ty.clone(), false),
            mir::Ty::Tuple(vec![i32_ty()]),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(40))),
                    assign(
                        place(3),
                        mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
                    ),
                    assign(
                        place(2),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Closure {
                                def: CLOSURE,
                                generic_args: match &clo_ty {
                                    mir::Ty::Closure(_, args) => args.clone(),
                                    _ => unreachable!(),
                                },
                            }),
                            vec![move_(3)],
                        ),
                    ),
                    assign(
                        place(4),
                        mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(2)),
                    ),
                    assign(
                        place(5),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![const_i32(2)],
                        ),
                    ),
                ],
                call(
                    fndef_operand(FN_CALL, vec![clo_ty.clone(), mir::Ty::Tuple(vec![i32_ty()])]),
                    vec![move_(4), move_(5)],
                    6,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(6)))], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    prog.fn_names
        .insert(CLOSURE, "main::{closure#0}".to_string());
    prog.add_with_def_id("main::{closure#0}", CLOSURE, add_base_closure_body());
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}

/// `{closure#0}(_1: &mut {closure}) -> i32` for `|| { n += 1; n }`
/// with `n: i32` captured by unique borrow:
/// `_2 = *((*_1).0) + 1; *((*_1).0) = _2; _0 = _2`.
fn inc_closure_body() -> mir::Body {
    let env_ty = ref_ty(closure_ty(CLOSURE, vec![], vec![ref_ty(i32_ty(), true)]), true);
    let captured = || {
        place(1)
            .deref()
            .field(0, ref_ty(i32_ty(), true))
            .deref()
    };
    mir::Body::new(
        decls(&[i32_ty(), env_ty, i32_ty()]),
        1,
        vec![block(
            vec![
                assign(place(2), add(mir::Operand::Copy(captured()), const_i32(1))),
                assign(captured(), mir::Rvalue::Use(copy(2))),
                assign(place(0), mir::Rvalue::Use(copy(2))),
            ],
            ret(),
        )],
    )
}

#[test]
fn fnmut_closure_mutation_persists_across_calls() {
    // main: _1 = 0 (n); _2 = &mut _1; _3 = Closure(move _2);
    //       call_mut twice (results _6, _9); _0 = copy _1 (n == 2)
    let clo_ty = closure_ty(CLOSURE, vec![], vec![ref_ty(i32_ty(), true)]);
    let unit_tuple = mir::Ty::Tuple(vec![]);
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty(), true),
            clo_ty.clone(),
            ref_ty(clo_ty.clone(), true),
            unit_tuple.clone(),
            i32_ty(),
            ref_ty(clo_ty.clone(), true),
            unit_tuple.clone(),
            i32_ty(),
        ]),
        0,
        vec![
        block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(0))),
                assign(
                    place(2),
                    mir::Rvalue::Ref(
                        mir::Region::Erased,
                        mir::BorrowKind::mutable(),
                        place(1),
                    ),
                ),
                assign(
                    place(3),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Closure {
                            def: CLOSURE,
                            generic_args: match &clo_ty {
                                mir::Ty::Closure(_, args) => args.clone(),
                                _ => unreachable!(),
                            },
                        }),
                        vec![move_(2)],
                    ),
                ),
                assign(
                    place(4),
                    mir::Rvalue::Ref(
                        mir::Region::Erased,
                        mir::BorrowKind::mutable(),
                        place(3),
                    ),
                ),
                assign(
                    place(5),
                    mir::Rvalue::Aggregate(Box::new(mir::AggregateKind::Tuple), vec![]),
                ),
            ],
            call(
                fndef_operand(FN_CALL_MUT, vec![clo_ty.clone(), mir::Ty::Tuple(vec![])]),
                vec![move_(4), move_(5)],
                6,
                1,
            ),
        ),
        block(
            vec![
                assign(
                    place(7),
                    mir::Rvalue::Ref(
                        mir::Region::Erased,
                        mir::BorrowKind::mutable(),
                        place(3),
                    ),
                ),
                assign(
                    place(8),
                    mir::Rvalue::Aggregate(Box::new(mir::AggregateKind::Tuple), vec![]),
                ),
            ],
            call(
                fndef_operand(FN_CALL_MUT, vec![clo_ty.clone(), mir::Ty::Tuple(vec![])]),
                vec![move_(7), move_(8)],
                9,
                2,
            ),
        ),
        block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    prog.fn_names
        .insert(CLOSURE, "main::{closure#0}".to_string());
    prog.add_with_def_id("main::{closure#0}", CLOSURE, inc_closure_body());
    prog.add("main", body);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(2));
}

/// `{closure#0}(_1: &{closure}) -> i32` for `move || s + 1` with
/// `s: i32` captured by value: `_0 = ((*_1).0) + 1`.
fn by_value_closure_body() -> mir::Body {
    let env_ty = ref_ty(closure_ty(CLOSURE, vec![], vec![i32_ty()]), false);
    let captured = place(1).deref().field(0, i32_ty());
    mir::Body::new(
        decls(&[i32_ty(), env_ty]),
        1,
        vec![block(
            vec![assign(
                place(0),
                add(mir::Operand::Copy(captured), const_i32(1)),
            )],
            ret(),
        )],
    )
}

#[test]
fn fnonce_closure_with_by_value_capture() {
    // main: _1 = Closure(const 41); _2 = (); _3 = call_once(move _1, move _2);
    //       _0 = copy _3
    let clo_ty = closure_ty(CLOSURE, vec![], vec![i32_ty()]);
    let main = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone(), mir::Ty::Tuple(vec![]), i32_ty()]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Closure {
                                def: CLOSURE,
                                generic_args: match &clo_ty {
                                    mir::Ty::Closure(_, args) => args.clone(),
                                    _ => unreachable!(),
                                },
                            }),
                            vec![const_i32(41)],
                        ),
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
            block(vec![assign(place(0), mir::Rvalue::Use(copy(3)))], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    prog.fn_names
        .insert(CLOSURE, "main::{closure#0}".to_string());
    prog.add_with_def_id("main::{closure#0}", CLOSURE, by_value_closure_body());
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}

#[test]
fn closure_passed_as_argument_and_called_inside() {
    // hof(_1: {closure}) -> i32: _2 = &_1; _3 = (7,);
    //   _4 = Fn::call(move _2, move _3); _0 = copy _4
    let clo_ty = closure_ty(CLOSURE, vec![i32_ty()], vec![i32_ty()]);
    let hof = mir::Body::new(
        decls(&[
            i32_ty(),
            clo_ty.clone(),
            ref_ty(clo_ty.clone(), false),
            mir::Ty::Tuple(vec![i32_ty()]),
            i32_ty(),
        ]),
        1,
        vec![
            block(
                vec![
                    assign(
                        place(2),
                        mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
                    ),
                    assign(
                        place(3),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![const_i32(7)],
                        ),
                    ),
                ],
                call(
                    fndef_operand(FN_CALL, vec![clo_ty.clone(), mir::Ty::Tuple(vec![i32_ty()])]),
                    vec![move_(2), move_(3)],
                    4,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(4)))], ret()),
        ],
    );
    // {closure#0}(_1: &{closure}, _2: i32) -> i32: _0 = ((*_1).0) + _2
    // (capture 35 by value, argument 7)
    let env_ty = ref_ty(clo_ty.clone(), false);
    let closure_body = mir::Body::new(
        decls(&[i32_ty(), env_ty, i32_ty()]),
        2,
        vec![block(
            vec![assign(
                place(0),
                add(
                    mir::Operand::Copy(place(1).deref().field(0, i32_ty())),
                    copy(2),
                ),
            )],
            ret(),
        )],
    );
    // main: _1 = Closure(const 35); _0 = hof(move _1) (direct call)
    let main = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone()]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Closure {
                            def: CLOSURE,
                            generic_args: match &clo_ty {
                                mir::Ty::Closure(_, args) => args.clone(),
                                _ => unreachable!(),
                            },
                        }),
                        vec![const_i32(35)],
                    ),
                )],
                call(fndef_operand(APPLY, vec![]), vec![move_(1)], 0, 1),
            ),
            block(vec![], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    prog.fn_names
        .insert(CLOSURE, "main::{closure#0}".to_string());
    prog.add_with_def_id("main::{closure#0}", CLOSURE, closure_body);
    prog.add_with_def_id("hof", APPLY, hof);
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}

#[test]
fn closure_without_body_in_program_fails_with_its_name() {
    // The closure value is built and called, but no body for it is in
    // the program: the failure must name the closure, not fall back
    // to a generic unsupported.
    let clo_ty = closure_ty(CLOSURE, vec![], vec![i32_ty()]);
    let main = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone(), mir::Ty::Tuple(vec![]), i32_ty()]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Closure {
                                def: CLOSURE,
                                generic_args: match &clo_ty {
                                    mir::Ty::Closure(_, args) => args.clone(),
                                    _ => unreachable!(),
                                },
                            }),
                            vec![const_i32(41)],
                        ),
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
            block(vec![assign(place(0), mir::Rvalue::Use(copy(3)))], ret()),
        ],
    );
    let mut prog = program_with_std_call_names();
    prog.fn_names
        .insert(CLOSURE, "main::{closure#0}".to_string());
    prog.add("main", main);
    let err = interpret_mir(&prog, "main", vec![]).expect_err("missing closure body must fail");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnknownFunction { ref name })
                if name == "main::{closure#0}"
        ),
        "expected UnknownFunction naming the closure, got {err}"
    );
}

#[test]
fn zero_capture_closure_place_is_never_written() {
    // Real rustc MIR for a zero-capture closure: the closure local
    // is never assigned (the value is zero-sized); the caller only
    // ever takes a reference to the place and calls through it.
    // main: _2 = &_1 (no write to _1 anywhere); _3 = (41,);
    //       _4 = Fn::call(move _2, move _3); _0 = copy _4
    let clo_ty = closure_ty(CLOSURE, vec![i32_ty()], vec![]);
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            clo_ty.clone(),
            ref_ty(clo_ty.clone(), false),
            mir::Ty::Tuple(vec![i32_ty()]),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(2),
                        mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
                    ),
                    assign(
                        place(3),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![const_i32(41)],
                        ),
                    ),
                ],
                call(
                    fndef_operand(FN_CALL, vec![clo_ty.clone(), mir::Ty::Tuple(vec![i32_ty()])]),
                    vec![move_(2), move_(3)],
                    4,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(4)))], ret()),
        ],
    );
    // {closure#0}(_1: &{closure}, _2: i32) -> i32 { _0 = _2 + 1 }
    let env_ty = ref_ty(clo_ty.clone(), false);
    let closure_body = mir::Body::new(
        decls(&[i32_ty(), env_ty, i32_ty()]),
        2,
        vec![block(
            vec![assign(place(0), add(copy(2), const_i32(1)))],
            ret(),
        )],
    );
    let mut prog = program_with_std_call_names();
    prog.fn_names
        .insert(CLOSURE, "main::{closure#0}".to_string());
    prog.add_with_def_id("main::{closure#0}", CLOSURE, closure_body);
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}
