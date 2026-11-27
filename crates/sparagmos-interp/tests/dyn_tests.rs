//! `dyn Trait` value-model tests: hand-built `sparagmos-mir`
//! bodies shaped after real `rustc 1.100.0 --emit=mir` output for
//! trait objects, converted and executed by the interpreter.
//!
//! The model: a dyn value is a fat pointer (`Value::DynRef`)
//! carrying the data pointer plus its vtable identity — the
//! concrete type's name and the trait bounds. Shapes reproduced
//! from real MIR:
//! * `&t as &dyn Trait` is `Cast(PointerCoercion(Unsize), &t,
//!   &dyn Trait)`; a dyn-to-dyn upcast is the same cast between
//!   two dyn types;
//! * a method call on a dyn value names the *trait* method
//!   (`probe::Greeter::greet`) with the dyn receiver as its first
//!   argument; the call dispatches to the concrete type's impl
//!   body (`<probe::Bob as probe::Greeter>::greet`);
//! * a `Box<dyn Trait>` method receiver is built by projecting
//!   the box's pointer field (`_b.0`), a `BoxDerefTransmute` cast
//!   to `*const dyn Trait`, and a borrow of the dereferenced raw
//!   pointer (`&*_p`).

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram, MirRunError};
use sparagmos_interp::{InterpError, RefTarget, Value};
use sparagmos_mir as mir;

// ----- items ---------------------------------------------------------------

const BOB_TY: mir::DefId = mir::DefId::new(0, 1);
const BOX_TY: mir::DefId = mir::DefId::new(0, 3);
const UNIQUE_TY: mir::DefId = mir::DefId::new(0, 6);
const GREET: mir::DefId = mir::DefId::new(0, 10);
const BASE_VAL: mir::DefId = mir::DefId::new(0, 11);
const PASSTHRU: mir::DefId = mir::DefId::new(0, 12);

const GREETER: &str = "probe::Greeter";
const BOB: &str = "probe::Bob";

// ----- small builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn unit_ty() -> mir::Ty {
    mir::Ty::Tuple(Vec::new())
}

fn bob_ty() -> mir::Ty {
    mir::Ty::Adt(BOB_TY, Vec::new())
}

fn dyn_ty(trait_name: &str) -> mir::Ty {
    mir::Ty::Dynamic(
        vec![trait_name.to_string()],
        Box::new(unit_ty()),
        mir::Region::Erased,
    )
}

fn ref_ty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Not)
}

fn box_ty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Adt(BOX_TY, vec![inner])
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

fn unsize(from: mir::Operand, target: mir::Ty) -> mir::Rvalue {
    mir::Rvalue::Cast(
        mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
        from,
        target,
    )
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

/// `<probe::Bob as probe::Greeter>::greet`: `_0 = (*_1).0`.
fn greet_impl_body() -> mir::Body {
    mir::Body::new(
        decls(&[i32_ty(), ref_ty(bob_ty())]),
        1,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Use(mir::Operand::Copy(place(1).deref().field(0, i32_ty()))),
            )],
            ret(),
        )],
    )
}

/// `<probe::Bob as probe::Base>::base_val`: `_0 = (*_1).0`.
fn base_val_impl_body() -> mir::Body {
    greet_impl_body()
}

fn base_program() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names.insert(BOB_TY, BOB.to_string());
    prog.fn_names.insert(BOX_TY, "std::boxed::Box".to_string());
    prog.fn_names.insert(UNIQUE_TY, "std::ptr::Unique".to_string());
    prog.fn_names.insert(GREET, "probe::Greeter::greet".to_string());
    prog.fn_names.insert(BASE_VAL, "probe::Base::base_val".to_string());
    prog
}

// ----- the tests -------------------------------------------------------------

/// Declaring a `&dyn Trait` local, passing it to another function
/// and returning it — without ever calling through it — converts
/// and executes; the returned value is the fat pointer with the
/// concrete type and bounds it was formed with.
#[test]
fn dyn_reference_declared_passed_and_returned() {
    // fn passthrough(_1: &dyn Greeter) -> &dyn Greeter { _0 = _1 }
    let passthrough = mir::Body::new(
        decls(&[ref_ty(dyn_ty(GREETER)), ref_ty(dyn_ty(GREETER))]),
        1,
        vec![block(vec![assign(place(0), mir::Rvalue::Use(move_(1)))], ret())],
    );
    // fn main() -> &dyn Greeter {
    //   _1 = Bob { 42 }; _2 = &_1; _3 = _2 as &dyn Greeter;
    //   _4 = passthrough(_3); _0 = _4
    // }
    let main = mir::Body::new(
        decls(&[
            ref_ty(dyn_ty(GREETER)),
            bob_ty(),
            ref_ty(bob_ty()),
            ref_ty(dyn_ty(GREETER)),
            ref_ty(dyn_ty(GREETER)),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(BOB_TY, vec![const_i32(42)])),
                    assign(place(2), ref_of(1)),
                    assign(place(3), unsize(copy(2), ref_ty(dyn_ty(GREETER)))),
                ],
                call(fndef_operand(PASSTHRU, Vec::new()), vec![move_(3)], 4, 1),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(move_(4)))], ret()),
        ],
    );
    let mut prog = base_program();
    prog.add("main", main);
    prog.add_with_def_id("probe::passthrough", PASSTHRU, passthrough);

    match interpret_mir(&prog, "main", Vec::new()).expect("main runs") {
        Value::DynRef {
            concrete, traits, ..
        } => {
            assert_eq!(concrete, BOB);
            assert_eq!(traits, vec![GREETER.to_string()]);
        }
        other => panic!("expected a dyn fat pointer, got {other:?}"),
    }
}

/// The sized-to-unsized coercion of a concrete value, then a
/// method call through the resulting `&dyn`: the call dispatches
/// to the concrete type's impl body and returns its result.
#[test]
fn unsize_coercion_then_virtual_dispatch() {
    // fn main() -> i32 {
    //   _1 = Bob { 42 }; _2 = &_1; _3 = _2 as &dyn Greeter;
    //   _0 = Greeter::greet(_3)
    // }
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            bob_ty(),
            ref_ty(bob_ty()),
            ref_ty(dyn_ty(GREETER)),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(BOB_TY, vec![const_i32(42)])),
                    assign(place(2), ref_of(1)),
                    assign(place(3), unsize(copy(2), ref_ty(dyn_ty(GREETER)))),
                ],
                call(
                    fndef_operand(GREET, vec![dyn_ty(GREETER)]),
                    vec![move_(3)],
                    0,
                    1,
                ),
            ),
            block(Vec::new(), ret()),
        ],
    );
    let mut prog = base_program();
    prog.add("main", main);
    prog.add("<probe::Bob as probe::Greeter>::greet", greet_impl_body());

    assert_eq!(
        interpret_mir(&prog, "main", Vec::new()).expect("main runs"),
        Value::Int(42)
    );
}

/// Dispatch whose concrete impl body is not in the program fails
/// at the call — with the dedicated error naming the method and
/// the concrete type — not at the dyn local's declaration.
#[test]
fn dispatch_without_impl_body_is_explicit_at_the_call() {
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            bob_ty(),
            ref_ty(bob_ty()),
            ref_ty(dyn_ty(GREETER)),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(BOB_TY, vec![const_i32(42)])),
                    assign(place(2), ref_of(1)),
                    assign(place(3), unsize(copy(2), ref_ty(dyn_ty(GREETER)))),
                ],
                call(
                    fndef_operand(GREET, vec![dyn_ty(GREETER)]),
                    vec![move_(3)],
                    0,
                    1,
                ),
            ),
            block(Vec::new(), ret()),
        ],
    );
    let mut prog = base_program();
    prog.add("main", main);
    // No `<probe::Bob as probe::Greeter>::greet` body.

    match interpret_mir(&prog, "main", Vec::new()) {
        Err(MirRunError::Interp(InterpError::UnsupportedDynamicDispatch {
            method,
            concrete,
        })) => {
            assert_eq!(method, "probe::Greeter::greet");
            assert_eq!(concrete, BOB);
        }
        other => panic!("expected UnsupportedDynamicDispatch, got {other:?}"),
    }
}

/// `Box<dyn Trait>`: the coercion of the box aggregate, then the
/// receiver construction real MIR uses (pointer-field projection,
/// `BoxDerefTransmute`, borrow of the deref) and the dispatch.
#[test]
fn box_dyn_receiver_chain_dispatches() {
    // fn main() -> i32 {
    //   _1 = Bob { 42 }
    //   _2 = Box { _1 }                    (the box aggregate)
    //   _3 = _2 as Box<dyn Greeter>        (Unsize)
    //   _4 = BoxDerefTransmute(_3.0)       (*const dyn Greeter)
    //   _5 = &*_4                          (&dyn Greeter)
    //   _0 = Greeter::greet(_5)
    // }
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            bob_ty(),
            box_ty(bob_ty()),
            box_ty(dyn_ty(GREETER)),
            mir::Ty::RawPtr(Box::new(dyn_ty(GREETER)), mir::Mutability::Not),
            ref_ty(dyn_ty(GREETER)),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(BOB_TY, vec![const_i32(42)])),
                    assign(place(2), adt_aggregate(BOX_TY, vec![move_(1)])),
                    assign(place(3), unsize(move_(2), box_ty(dyn_ty(GREETER)))),
                    assign(
                        place(4),
                        mir::Rvalue::Cast(
                            mir::CastKind::BoxDerefTransmute,
                            mir::Operand::Copy(
                                place(3).field(0, mir::Ty::Adt(UNIQUE_TY, vec![dyn_ty(GREETER)])),
                            ),
                            mir::Ty::RawPtr(Box::new(dyn_ty(GREETER)), mir::Mutability::Not),
                        ),
                    ),
                    assign(
                        place(5),
                        mir::Rvalue::Ref(
                            mir::Region::Erased,
                            mir::BorrowKind::Shared,
                            place(4).deref(),
                        ),
                    ),
                ],
                call(
                    fndef_operand(GREET, vec![dyn_ty(GREETER)]),
                    vec![move_(5)],
                    0,
                    1,
                ),
            ),
            block(Vec::new(), ret()),
        ],
    );
    let mut prog = base_program();
    prog.add("main", main);
    prog.add("<probe::Bob as probe::Greeter>::greet", greet_impl_body());

    assert_eq!(
        interpret_mir(&prog, "main", Vec::new()).expect("main runs"),
        Value::Int(42)
    );
}

/// A dyn-to-dyn upcast (`&dyn Derived` to `&dyn Base`) keeps the
/// concrete type; a supertrait method call through the upcast
/// value dispatches to the concrete type's `Base` impl.
#[test]
fn upcast_keeps_concrete_and_supertrait_dispatches() {
    // fn main() -> i32 {
    //   _1 = Bob { 42 }; _2 = &_1;
    //   _3 = _2 as &dyn Derived; _4 = _3 as &dyn Base;
    //   _0 = Base::base_val(_4)
    // }
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            bob_ty(),
            ref_ty(bob_ty()),
            ref_ty(dyn_ty("probe::Derived")),
            ref_ty(dyn_ty("probe::Base")),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(BOB_TY, vec![const_i32(42)])),
                    assign(place(2), ref_of(1)),
                    assign(place(3), unsize(copy(2), ref_ty(dyn_ty("probe::Derived")))),
                    assign(place(4), unsize(copy(3), ref_ty(dyn_ty("probe::Base")))),
                ],
                call(
                    fndef_operand(BASE_VAL, vec![dyn_ty("probe::Base")]),
                    vec![move_(4)],
                    0,
                    1,
                ),
            ),
            block(Vec::new(), ret()),
        ],
    );
    let mut prog = base_program();
    prog.add("main", main);
    prog.add("<probe::Bob as probe::Base>::base_val", base_val_impl_body());

    assert_eq!(
        interpret_mir(&prog, "main", Vec::new()).expect("main runs"),
        Value::Int(42)
    );
}

// ----- trait-method calls resolved to collected impl bodies ------------------
//
// Issue #65 (tests/ui/dst/dst-index-success.rs): a call naming the
// trait item (`std::ops::Index::index`, generic arguments
// `[Self, usize]`) whose program also contains the concrete impl
// body must execute the impl body. Before the fix, conversion fell
// through to the name-based builtin model of the trait method,
// which treated the receiver as a slice: for the unit-struct
// receivers below it panicked with a bogus "index out of bounds:
// the len is 0 but the index is 0" — a silent wrong result for a
// shape the runtime fully supports.

const S_TY: mir::DefId = mir::DefId::new(0, 21);
const T_TY: mir::DefId = mir::DefId::new(0, 22);
const INDEX: mir::DefId = mir::DefId::new(0, 23);
const DEBUG: &str = "std::fmt::Debug";

fn s_ty() -> mir::Ty {
    mir::Ty::Adt(S_TY, Vec::new())
}

fn t_ty() -> mir::Ty {
    mir::Ty::Adt(T_TY, Vec::new())
}

fn usize_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::Usize)
}

fn const_usize(v: u64) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
            size: 8,
        },
    )))
}

fn const_str(v: &str) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Str(
        v.to_string(),
    )))
}

fn index_program() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names.insert(S_TY, "probe::S".to_string());
    prog.fn_names.insert(T_TY, "probe::T".to_string());
    prog.fn_names
        .insert(INDEX, "std::ops::Index::index".to_string());
    prog
}

/// The string a `&str` value denotes, unwrapping the fat-pointer
/// and reference layers the model wraps it in.
fn str_content(v: &Value) -> Option<String> {
    match v {
        Value::Str(s) => Some(s.clone()),
        Value::FatRef { base, .. } => str_content(base),
        Value::Ref(RefTarget::Value(inner, _)) => str_content(inner),
        _ => None,
    }
}

/// `<probe::S as std::ops::Index<usize>>::index`: `_0 = "hello"`.
fn s_index_impl_body() -> mir::Body {
    mir::Body::new(
        decls(&[ref_ty(mir::Ty::Str), ref_ty(s_ty()), usize_ty()]),
        2,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_str("hello")))],
            ret(),
        )],
    )
}

/// `<probe::T as std::ops::Index<usize>>::index`:
/// `_3 = 42; _4 = &_3; _0 = _4 as &dyn Debug` (the suite file's
/// referent is a static; a local models the same Unsize shape).
fn t_index_impl_body() -> mir::Body {
    mir::Body::new(
        decls(&[
            ref_ty(dyn_ty(DEBUG)),
            ref_ty(t_ty()),
            usize_ty(),
            usize_ty(),
            ref_ty(usize_ty()),
        ]),
        2,
        vec![block(
            vec![
                assign(place(3), mir::Rvalue::Use(const_usize(42))),
                assign(place(4), ref_of(3)),
                assign(place(0), unsize(copy(4), ref_ty(dyn_ty(DEBUG)))),
            ],
            ret(),
        )],
    )
}

/// `fn main() -> &str { _1 = S; _2 = &_1; _3 = Index::index(move _2, 0); _0 = _3 }`
fn s_index_main() -> mir::Body {
    mir::Body::new(
        decls(&[
            ref_ty(mir::Ty::Str),
            s_ty(),
            ref_ty(s_ty()),
            ref_ty(mir::Ty::Str),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(S_TY, Vec::new())),
                    assign(place(2), ref_of(1)),
                ],
                call(
                    fndef_operand(INDEX, vec![s_ty(), usize_ty()]),
                    vec![move_(2), const_usize(0)],
                    3,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(move_(3)))], ret()),
        ],
    )
}

#[test]
fn index_impl_with_str_output_runs_the_impl_body() {
    let mut prog = index_program();
    prog.add("main", s_index_main());
    prog.add(
        "<probe::S as std::ops::Index<usize>>::index",
        s_index_impl_body(),
    );

    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(str_content(&value).as_deref(), Some("hello"));
}

#[test]
fn index_impl_with_dyn_output_runs_the_impl_body() {
    // fn main() -> &dyn Debug {
    //   _1 = T; _2 = &_1; _3 = Index::index(move _2, 0); _0 = _3
    // }
    let main = mir::Body::new(
        decls(&[
            ref_ty(dyn_ty(DEBUG)),
            t_ty(),
            ref_ty(t_ty()),
            ref_ty(dyn_ty(DEBUG)),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(T_TY, Vec::new())),
                    assign(place(2), ref_of(1)),
                ],
                call(
                    fndef_operand(INDEX, vec![t_ty(), usize_ty()]),
                    vec![move_(2), const_usize(0)],
                    3,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(move_(3)))], ret()),
        ],
    );
    let mut prog = index_program();
    prog.add("main", main);
    prog.add(
        "<probe::T as std::ops::Index<usize>>::index",
        t_index_impl_body(),
    );

    match interpret_mir(&prog, "main", Vec::new()).expect("main runs") {
        Value::DynRef {
            concrete, traits, ..
        } => {
            assert_eq!(concrete, "usize");
            assert_eq!(traits, vec![DEBUG.to_string()]);
        }
        other => panic!("expected a dyn fat pointer, got {other:?}"),
    }
}

#[test]
fn index_call_without_a_collected_impl_body_keeps_the_builtin_model() {
    // The same call as `index_impl_with_str_output_runs_the_impl_body`
    // but with no impl body in the program: the name-based builtin
    // model of `Index::index` applies, as it does for core's
    // slice/array/Vec impls whose bodies are not collected. On
    // this unit-struct receiver the builtin reports the receiver
    // as a length-0 indexable — the pinned boundary is that the
    // builtin, not an impl body, ran.
    let mut prog = index_program();
    prog.add("main", s_index_main());

    match interpret_mir(&prog, "main", Vec::new()) {
        Err(MirRunError::Interp(InterpError::Panic { message, .. })) => {
            assert_eq!(
                message,
                "index out of bounds: the len is 0 but the index is 0"
            );
        }
        other => panic!("expected the builtin's bounds panic, got {other:?}"),
    }
}

const PARTIAL_EQ: mir::DefId = mir::DefId::new(0, 24);

fn const_bool(v: bool) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Bool(v)))
}

fn partial_eq_program() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names.insert(S_TY, "probe::S".to_string());
    prog.fn_names
        .insert(PARTIAL_EQ, "std::cmp::PartialEq::eq".to_string());
    prog
}

/// `fn main() -> bool { _1 = S; _2 = S; _3 = &_1; _4 = &_2;
/// _5 = PartialEq::eq(move _3, move _4); _0 = _5 }`
fn partial_eq_main() -> mir::Body {
    mir::Body::new(
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
    )
}

#[test]
fn partial_eq_call_runs_a_converted_custom_impl_body() {
    // Resolution is not Index-specific: a converted custom
    // PartialEq impl body wins over the builtin's structural
    // comparison. The impl returns false unconditionally, so two
    // equal unit values compare unequal — proof the body ran.
    let eq_impl = mir::Body::new(
        decls(&[mir::Ty::Bool, ref_ty(s_ty()), ref_ty(s_ty())]),
        2,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_bool(false)))],
            ret(),
        )],
    );
    let mut prog = partial_eq_program();
    prog.add("main", partial_eq_main());
    prog.add("<probe::S as std::cmp::PartialEq>::eq", eq_impl);

    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Bool(false));
}

#[test]
fn impl_resolution_skips_impl_bodies_that_do_not_convert() {
    // A derived-PartialEq shape: the impl body is in the program
    // but cannot convert (here it holds a coroutine local; the
    // real case is a packed struct's derived eq, whose body the
    // conversion rejects for its pointer casts). The call must
    // keep the builtin's structural comparison — resolving to the
    // unconvertible body would trade a correct answer for an
    // unconverted-body error.
    let eq_impl = mir::Body::new(
        decls(&[
            mir::Ty::Bool,
            ref_ty(s_ty()),
            ref_ty(s_ty()),
            mir::Ty::Coroutine(mir::DefId::new(0, 7), Vec::new()),
        ]),
        2,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_bool(true)))],
            ret(),
        )],
    );
    let mut prog = partial_eq_program();
    prog.add("main", partial_eq_main());
    prog.add("<probe::S as std::cmp::PartialEq>::eq", eq_impl);

    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Bool(true));
}

const HELPER: mir::DefId = mir::DefId::new(0, 25);

#[test]
fn impl_resolution_skips_impl_bodies_calling_unconvertible_helpers() {
    // The transitive shape of the packed-struct regression: the
    // impl body itself converts, but it calls a helper whose body
    // holds an unsupported rvalue — a tuple-to-integer
    // `Transmute` cast, standing in for the `check_align` helper
    // of `tests/ui/derives/deriving-with-repr-packed-3.rs`
    // (whose `PointerExposeProvenance` cast became executable in
    // Lane BV and whose reverse cast in Lane FB, so an aggregate
    // transmute — still unconvertible, as the value model has
    // no flat byte memory — plays the role now). Resolving
    // the trait call to the impl body would run the helper and
    // fail on the unsupported cast; the builtin's structural
    // comparison is exact, so it must run instead. The impl
    // returns false unconditionally, so two equal unit values
    // comparing equal proves the builtin ran.
    let helper = mir::Body::new(
        decls(&[
            mir::Ty::Uint(mir::UintTy::Usize),
            mir::Ty::Tuple(vec![
                mir::Ty::Uint(mir::UintTy::U8),
                mir::Ty::Uint(mir::UintTy::U8),
            ]),
        ]),
        0,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Cast(
                    mir::CastKind::Transmute,
                    copy(1),
                    mir::Ty::Uint(mir::UintTy::Usize),
                ),
            )],
            ret(),
        )],
    );
    let eq_impl = mir::Body::new(
        decls(&[
            mir::Ty::Bool,
            ref_ty(s_ty()),
            ref_ty(s_ty()),
            mir::Ty::Uint(mir::UintTy::Usize),
        ]),
        2,
        vec![
            block(Vec::new(), call(fndef_operand(HELPER, Vec::new()), Vec::new(), 3, 1)),
            block(
                vec![assign(place(0), mir::Rvalue::Use(const_bool(false)))],
                ret(),
            ),
        ],
    );
    let mut prog = partial_eq_program();
    prog.fn_names.insert(HELPER, "probe::helper".to_string());
    prog.add("main", partial_eq_main());
    prog.add("<probe::S as std::cmp::PartialEq>::eq", eq_impl);
    prog.add_with_def_id("probe::helper", HELPER, helper);

    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Bool(true));
}

// ----- data-pointer extraction over an inline box base ----------------------

const PAIR_TY: mir::DefId = mir::DefId::new(0, 26);

fn pair_ty() -> mir::Ty {
    mir::Ty::Adt(PAIR_TY, Vec::new())
}

/// Casting a trait object whose base is the one-field box
/// aggregate that holds its payload inline (the shape the
/// panic-cleanup and thread-join models fabricate for a caught
/// payload) to the concrete pointer type must yield a data
/// pointer denoting the payload as a whole: reading the
/// payload's fields through it reads the payload, not a
/// position inside it.
#[test]
fn data_ptr_cast_over_inline_box_base_denotes_whole_payload() {
    // fn main() -> i32 {
    //   _1 = Pair { 11, 22 }
    //   _2 = Box { _1 }                       (the box aggregate)
    //   _3 = _2 as Box<dyn Any>               (Unsize)
    //   _4 = BoxDerefTransmute(_3.0)          (*const dyn Any)
    //   _5 = _4 as *const Pair                (PtrToPtr: data pointer)
    //   _0 = (*_5).1 + (*_5).0
    // }
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            pair_ty(),
            box_ty(pair_ty()),
            box_ty(dyn_ty("std::any::Any")),
            mir::Ty::RawPtr(Box::new(dyn_ty("std::any::Any")), mir::Mutability::Not),
            mir::Ty::RawPtr(Box::new(pair_ty()), mir::Mutability::Not),
        ]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    adt_aggregate(PAIR_TY, vec![const_i32(11), const_i32(22)]),
                ),
                assign(place(2), adt_aggregate(BOX_TY, vec![move_(1)])),
                assign(place(3), unsize(move_(2), box_ty(dyn_ty("std::any::Any")))),
                assign(
                    place(4),
                    mir::Rvalue::Cast(
                        mir::CastKind::BoxDerefTransmute,
                        mir::Operand::Copy(
                            place(3)
                                .field(0, mir::Ty::Adt(UNIQUE_TY, vec![dyn_ty("std::any::Any")])),
                        ),
                        mir::Ty::RawPtr(Box::new(dyn_ty("std::any::Any")), mir::Mutability::Not),
                    ),
                ),
                assign(
                    place(5),
                    mir::Rvalue::Cast(
                        mir::CastKind::PtrToPtr,
                        move_(4),
                        mir::Ty::RawPtr(Box::new(pair_ty()), mir::Mutability::Not),
                    ),
                ),
                assign(
                    place(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((
                            mir::Operand::Copy(place(5).deref().field(1, i32_ty())),
                            mir::Operand::Copy(place(5).deref().field(0, i32_ty())),
                        )),
                    ),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = base_program();
    prog.fn_names.insert(PAIR_TY, "probe::Pair".to_string());
    prog.add("main", main);

    assert_eq!(
        interpret_mir(&prog, "main", Vec::new()).expect("main runs"),
        Value::Int(33)
    );
}
