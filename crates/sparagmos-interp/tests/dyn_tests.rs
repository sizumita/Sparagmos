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
use sparagmos_interp::{InterpError, Value};
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
