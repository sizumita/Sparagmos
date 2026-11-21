//! Call-dispatch leftovers (Lane EK): the dispatch shapes that
//! fell between the `Fn`-trait, impl-resolution and drop-glue
//! machinery.
//!
//! * a `&mut dyn FnMut` receiver whose concrete value is an ADT
//!   with a hand-written `FnMut` impl (the impl key carries the
//!   trait's generic argument list);
//! * an `Fn`-trait call whose collected impl body is keyed by its
//!   generic definition name (`<YCombinator<F, A, R> as ...>`);
//! * a `Fun<F>` wrapper received as a zero-sized constant under a
//!   `FnOnce::call_once` call: the constant only materialises
//!   under the call's concrete `Self` type;
//! * a `Drop::drop` call on a `Box` shell, converted to the box's
//!   drop glue, with the following `Drop` terminator running no
//!   glue a second time;
//! * one-field callable wrappers unwrapped for `CallOnce` and
//!   by-value `CallMut` receivers;
//! * a virtual call whose impl key carries trait generic
//!   arguments.
//!
//! The programs are hand-built `sparagmos-mir` bodies shaped
//! after the driver's output for the corresponding Rust sources
//! (the `unboxed-closures-*`, `manual-fnonce-impl-wrapper` and
//! box-drop test families).

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::Value;
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn unit_ty() -> mir::Ty {
    mir::Ty::Tuple(Vec::new())
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

fn tuple1_i32() -> mir::Ty {
    mir::Ty::Tuple(vec![i32_ty()])
}

fn fnptr_ty(inputs: Vec<mir::Ty>, output: mir::Ty) -> mir::Ty {
    mir::Ty::FnPtr(Box::new(mir::FnSig::new(inputs, output)))
}

/// The closure type as the driver produces it: generic arguments
/// are (kind byte, signature, upvars tuple).
fn closure_ty(def: mir::DefId, upvars: Vec<mir::Ty>) -> mir::Ty {
    mir::Ty::Closure(
        def,
        vec![
            mir::Ty::Int(mir::IntTy::I8),
            fnptr_ty(vec![tuple1_i32()], i32_ty()),
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

fn const_zst() -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::ZeroSized))
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

fn adt_aggregate(def: mir::DefId, generic_args: Vec<mir::Ty>, ops: Vec<mir::Operand>) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def,
            variant_index: 0,
            generic_args,
            user_ty: None,
            field_index: None,
        }),
        ops,
    )
}

fn tuple_aggregate(ops: Vec<mir::Operand>) -> mir::Rvalue {
    mir::Rvalue::Aggregate(Box::new(mir::AggregateKind::Tuple), ops)
}

fn ref_of(idx: u32, mutable: bool) -> mir::Rvalue {
    mir::Rvalue::Ref(
        mir::Region::Erased,
        if mutable {
            mir::BorrowKind::Mut {
                allow_two_phase_borrow: false,
            }
        } else {
            mir::BorrowKind::Shared
        },
        place(idx),
    )
}

fn unsize(from: mir::Operand, target: mir::Ty) -> mir::Rvalue {
    mir::Rvalue::Cast(
        mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
        from,
        target,
    )
}

fn dyn_ty(trait_name: &str) -> mir::Ty {
    mir::Ty::Dynamic(
        vec![trait_name.to_string()],
        Box::new(unit_ty()),
        mir::Region::Erased,
    )
}

// ----- shared items -----------------------------------------------------------

const FN_CALL: mir::DefId = mir::DefId::new(0, 4);
const FN_CALL_MUT: mir::DefId = mir::DefId::new(0, 5);
const FN_CALL_ONCE: mir::DefId = mir::DefId::new(0, 6);

fn fn_trait_names(prog: &mut MirProgram) {
    prog.fn_names
        .insert(FN_CALL, "std::ops::Fn::call".to_string());
    prog.fn_names
        .insert(FN_CALL_MUT, "std::ops::FnMut::call_mut".to_string());
    prog.fn_names
        .insert(FN_CALL_ONCE, "std::ops::FnOnce::call_once".to_string());
}

// ----- manual FnMut impl through &mut dyn FnMut -------------------------------

const S_TY: mir::DefId = mir::DefId::new(0, 30);

#[test]
fn dyn_fnmut_dispatches_to_adt_impl() {
    // struct S(i32); impl FnMut<(i32,)> for S {
    //   fn call_mut(&mut self, (x,)) -> i32 { self.0 + x } }
    // fn main() -> i32 {
    //   _1 = S { 40 }; _2 = &mut _1; _3 = _2 as &mut dyn FnMut;
    //   _5 = FnMut::call_mut(move _3, (2,)); return _5 }
    let s_ty = mir::Ty::Adt(S_TY, vec![]);
    let impl_body = mir::Body::new(
        decls(&[i32_ty(), ref_ty(s_ty.clone(), true), tuple1_i32()]),
        2,
        vec![block(
            vec![assign(
                place(0),
                add(
                    mir::Operand::Copy(place(1).deref().field(0, i32_ty())),
                    mir::Operand::Copy(place(2).field(0, i32_ty())),
                ),
            )],
            ret(),
        )],
    );
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            s_ty.clone(),
            ref_ty(s_ty.clone(), true),
            ref_ty(dyn_ty("std::ops::FnMut"), true),
            tuple1_i32(),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(S_TY, vec![], vec![const_i32(40)])),
                    assign(place(2), ref_of(1, true)),
                    assign(
                        place(3),
                        unsize(copy(2), ref_ty(dyn_ty("std::ops::FnMut"), true)),
                    ),
                    assign(place(4), tuple_aggregate(vec![const_i32(2)])),
                ],
                call(
                    fndef_operand(
                        FN_CALL_MUT,
                        vec![dyn_ty("std::ops::FnMut"), tuple1_i32()],
                    ),
                    vec![move_(3), move_(4)],
                    5,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(5)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    fn_trait_names(&mut prog);
    prog.fn_names.insert(S_TY, "probe::S".to_string());
    prog.add("main", main);
    prog.add(
        "<probe::S as std::ops::FnMut<(i32,)>>::call_mut",
        impl_body,
    );
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).expect("dyn FnMut call runs"),
        Value::Int(42)
    );
}

// ----- YCombinator: impl keyed by its generic definition name -----------------

const Y_TY: mir::DefId = mir::DefId::new(0, 32);

#[test]
fn fn_call_resolves_impl_keyed_by_generic_definition() {
    // struct YCombinator<F, A, R>;
    // impl Fn<(A,)> for YCombinator<F, A, R> { fn call(&self, (x,)) = x + 1 }
    // fn main() -> i32 { _1 = YCombinator {}; _2 = &_1;
    //   _4 = Fn::call(move _2, (41,)); return _4 }
    let y_ty = mir::Ty::Adt(
        Y_TY,
        vec![i32_ty(), i32_ty(), i32_ty()],
    );
    let impl_body = mir::Body::new(
        decls(&[i32_ty(), ref_ty(y_ty.clone(), false), tuple1_i32()]),
        2,
        vec![block(
            vec![assign(
                place(0),
                add(
                    mir::Operand::Copy(place(2).field(0, i32_ty())),
                    const_i32(1),
                ),
            )],
            ret(),
        )],
    );
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            y_ty.clone(),
            ref_ty(y_ty.clone(), false),
            tuple1_i32(),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        adt_aggregate(Y_TY, vec![i32_ty(), i32_ty(), i32_ty()], vec![]),
                    ),
                    assign(place(2), ref_of(1, false)),
                    assign(place(3), tuple_aggregate(vec![const_i32(41)])),
                ],
                call(
                    fndef_operand(FN_CALL, vec![y_ty.clone(), tuple1_i32()]),
                    vec![move_(2), move_(3)],
                    4,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(4)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    fn_trait_names(&mut prog);
    prog.fn_names.insert(Y_TY, "probe::YCombinator".to_string());
    prog.add("main", main);
    prog.add(
        "<YCombinator<F, A, R> as std::ops::Fn<(A,)>>::call",
        impl_body,
    );
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).expect("YCombinator call runs"),
        Value::Int(42)
    );
}

// ----- Fun<F>: zero-sized constant receiver under call_once -------------------

const FUN_TY: mir::DefId = mir::DefId::new(0, 34);
const FUN_CLOSURE: mir::DefId = mir::DefId::new(0, 35);

#[test]
fn fun_wrapper_zst_constant_receiver_materialises() {
    // struct Fun<F>(F); impl FnOnce<(i32,)> for Fun<F> {
    //   fn call_once(self, args) -> i32 { self.0.call(args) } }
    // The closure is a zero-capture closure returning x + 1.
    // fn main() -> i32 { _2 = FnOnce::call_once(const Fun ZST, (41,)) }
    let clo_ty = closure_ty(FUN_CLOSURE, vec![]);
    let fun_ty = mir::Ty::Adt(FUN_TY, vec![clo_ty.clone()]);

    // {closure}(_1: &{closure}, _2: i32) -> i32 { _0 = _2 + 1 }
    let closure_body = mir::Body::new(
        decls(&[i32_ty(), ref_ty(clo_ty.clone(), false), i32_ty()]),
        2,
        vec![block(
            vec![assign(place(0), add(copy(2), const_i32(1)))],
            ret(),
        )],
    );
    // <Fun<F> as FnOnce>::call_once(_1: Fun<{closure}>, _2: (i32,)):
    //   _3 = &_1.0; _4 = Fn::call(move _3, copy _2); _0 = copy _4
    let impl_body = mir::Body::new(
        decls(&[
            i32_ty(),
            fun_ty.clone(),
            tuple1_i32(),
            ref_ty(clo_ty.clone(), false),
            i32_ty(),
        ]),
        2,
        vec![
            block(
                vec![assign(
                    place(3),
                    mir::Rvalue::Ref(
                        mir::Region::Erased,
                        mir::BorrowKind::Shared,
                        place(1).field(0, clo_ty.clone()),
                    ),
                )],
                call(
                    fndef_operand(FN_CALL, vec![clo_ty.clone(), tuple1_i32()]),
                    vec![move_(3), copy(2)],
                    4,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(4)))], ret()),
        ],
    );
    let main = mir::Body::new(
        decls(&[i32_ty(), tuple1_i32(), i32_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), tuple_aggregate(vec![const_i32(41)]))],
                call(
                    fndef_operand(FN_CALL_ONCE, vec![fun_ty.clone(), tuple1_i32()]),
                    vec![const_zst(), move_(1)],
                    2,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    fn_trait_names(&mut prog);
    prog.fn_names.insert(FUN_TY, "probe::Fun".to_string());
    prog.fn_names
        .insert(FUN_CLOSURE, "probe::{closure#0}".to_string());
    prog.adt_shapes.insert(
        FUN_TY,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![vec![mir::Ty::Param(0, "F".to_string())]],
        },
    );
    prog.add("main", main);
    prog.add(
        "<probe::Fun<F> as std::ops::FnOnce<(i32,)>>::call_once",
        impl_body,
    );
    prog.add("probe::{closure#0}", closure_body);
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).expect("Fun wrapper call runs"),
        Value::Int(42)
    );
}

// ----- Box shell Drop::drop -----------------------------------------------------

const D_TY: mir::DefId = mir::DefId::new(0, 40);
const BOX_TY: mir::DefId = mir::DefId::new(0, 41);
const DROP_FN: mir::DefId = mir::DefId::new(0, 42);

fn d_ty() -> mir::Ty {
    mir::Ty::Adt(D_TY, vec![])
}

fn box_d_ty() -> mir::Ty {
    mir::Ty::Adt(BOX_TY, vec![d_ty()])
}

/// `<probe::D as std::ops::Drop>::drop`: `*self.sink = *self.sink * 10 + self.id`.
fn bump_dtor() -> mir::Body {
    mir::Body::new(
        decls(&[
            unit_ty(),
            ref_ty(d_ty(), true),
            ref_ty(i32_ty(), true),
            i32_ty(),
            i32_ty(),
            i32_ty(),
        ]),
        1,
        vec![block(
            vec![
                assign(
                    place(2),
                    mir::Rvalue::Use(mir::Operand::Copy(
                        place(1).deref().field(0, ref_ty(i32_ty(), true)),
                    )),
                ),
                assign(
                    place(3),
                    mir::Rvalue::Use(mir::Operand::Copy(
                        place(1).deref().field(1, i32_ty()),
                    )),
                ),
                assign(
                    place(4),
                    mir::Rvalue::Use(mir::Operand::Copy(place(2).deref())),
                ),
                assign(
                    place(5),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Mul,
                        Box::new((copy(4), const_i32(10))),
                    ),
                ),
                assign(
                    place(5),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((copy(5), copy(3))),
                    ),
                ),
                assign(place(2).deref(), mir::Rvalue::Use(copy(5))),
            ],
            ret(),
        )],
    )
}

#[test]
fn box_shell_drop_drop_call_runs_glue_once() {
    // fn main() -> i32 {
    //   _1 = 0; _2 = &mut _1; _3 = D { _2, 7 }; _4 = Box { _3 };
    //   _5 = move _4.0;              (the content moves out)
    //   drop(_5);                    (D's dtor runs: counter = 7)
    //   _6 = &mut _4; Drop::drop(move _6);   (the shell: no glue left)
    //   drop(_4);                    (already cleared: no second run)
    //   return _1 }
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty(), true),
            d_ty(),
            box_d_ty(),
            d_ty(),
            ref_ty(box_d_ty(), true),
            unit_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_of(1, true)),
                    assign(
                        place(3),
                        adt_aggregate(D_TY, vec![], vec![move_(2), const_i32(7)]),
                    ),
                    assign(
                        place(4),
                        adt_aggregate(BOX_TY, vec![d_ty()], vec![move_(3)]),
                    ),
                    assign(
                        place(5),
                        mir::Rvalue::Use(mir::Operand::Move(place(4).field(0, d_ty()))),
                    ),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(
                vec![],
                mir::TerminatorKind::Drop {
                    place: place(5),
                    target: mir::BasicBlock(2),
                    unwind: mir::UnwindAction::Continue,
                    replace: false,
                },
            ),
            block(
                vec![assign(place(6), ref_of(4, true))],
                call(
                    fndef_operand(DROP_FN, vec![box_d_ty()]),
                    vec![move_(6)],
                    7,
                    3,
                ),
            ),
            block(
                vec![],
                mir::TerminatorKind::Drop {
                    place: place(4),
                    target: mir::BasicBlock(4),
                    unwind: mir::UnwindAction::Continue,
                    replace: false,
                },
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.fn_names.insert(D_TY, "probe::D".to_string());
    prog.fn_names
        .insert(BOX_TY, "std::boxed::Box".to_string());
    prog.fn_names
        .insert(DROP_FN, "std::ops::Drop::drop".to_string());
    prog.add("probe::main", main);
    prog.add("<probe::D as std::ops::Drop>::drop", bump_dtor());
    assert_eq!(
        interpret_mir(&prog, "probe::main", vec![]).expect("box shell drop runs"),
        Value::Int(7)
    );
}

// ----- one-field callable wrappers: CallOnce and by-value CallMut --------------

const W_TY: mir::DefId = mir::DefId::new(0, 50);
const W_CLOSURE: mir::DefId = mir::DefId::new(0, 51);
const W_ONCE_CLOSURE: mir::DefId = mir::DefId::new(0, 52);

fn w_ty(clo: mir::DefId) -> mir::Ty {
    mir::Ty::Adt(W_TY, vec![closure_ty(clo, vec![])])
}

fn w_aggregate(clo: mir::DefId, ops: Vec<mir::Operand>) -> mir::Rvalue {
    adt_aggregate(W_TY, vec![closure_ty(clo, vec![])], ops)
}

fn closure_aggregate(def: mir::DefId) -> mir::Rvalue {
    let ty = closure_ty(def, vec![]);
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Closure {
            def,
            generic_args: match &ty {
                mir::Ty::Closure(_, args) => args.clone(),
                _ => unreachable!(),
            },
        }),
        vec![],
    )
}

#[test]
fn call_once_unwraps_one_field_wrapper() {
    // {closure}(_1: {closure} by value, _2: i32) -> i32 { _0 = _2 + 1 }
    // fn main() -> i32 { _1 = W { closure }; _3 = FnOnce::call_once(move _1, (41,)) }
    let clo_ty = closure_ty(W_ONCE_CLOSURE, vec![]);
    let closure_body = mir::Body::new(
        decls(&[i32_ty(), clo_ty.clone(), i32_ty()]),
        2,
        vec![block(
            vec![assign(place(0), add(copy(2), const_i32(1)))],
            ret(),
        )],
    );
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            w_ty(W_ONCE_CLOSURE),
            closure_ty(W_ONCE_CLOSURE, vec![]),
            tuple1_i32(),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(2), closure_aggregate(W_ONCE_CLOSURE)),
                    assign(place(1), w_aggregate(W_ONCE_CLOSURE, vec![move_(2)])),
                    assign(place(3), tuple_aggregate(vec![const_i32(41)])),
                ],
                call(
                    fndef_operand(
                        FN_CALL_ONCE,
                        vec![w_ty(W_ONCE_CLOSURE), tuple1_i32()],
                    ),
                    vec![move_(1), move_(3)],
                    4,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(4)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    fn_trait_names(&mut prog);
    prog.fn_names.insert(W_TY, "probe::W".to_string());
    prog.fn_names
        .insert(W_ONCE_CLOSURE, "probe::{closure#1}".to_string());
    prog.add("main", main);
    prog.add("probe::{closure#1}", closure_body);
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).expect("wrapped call_once runs"),
        Value::Int(42)
    );
}

#[test]
fn by_value_call_mut_unwraps_one_field_wrapper() {
    // {closure}(_1: &{closure}, _2: i32) -> i32 { _0 = _2 + 1 }
    // fn main() -> i32 { _1 = W { closure };
    //   _3 = FnMut::call_mut(move _1, (41,))  (by-value receiver shape) }
    let clo_ty = closure_ty(W_CLOSURE, vec![]);
    let closure_body = mir::Body::new(
        decls(&[i32_ty(), ref_ty(clo_ty.clone(), false), i32_ty()]),
        2,
        vec![block(
            vec![assign(place(0), add(copy(2), const_i32(1)))],
            ret(),
        )],
    );
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            w_ty(W_CLOSURE),
            closure_ty(W_CLOSURE, vec![]),
            tuple1_i32(),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(2), closure_aggregate(W_CLOSURE)),
                    assign(place(1), w_aggregate(W_CLOSURE, vec![move_(2)])),
                    assign(place(3), tuple_aggregate(vec![const_i32(41)])),
                ],
                call(
                    fndef_operand(FN_CALL_MUT, vec![w_ty(W_CLOSURE), tuple1_i32()]),
                    vec![move_(1), move_(3)],
                    4,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(4)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    fn_trait_names(&mut prog);
    prog.fn_names.insert(W_TY, "probe::W".to_string());
    prog.fn_names
        .insert(W_CLOSURE, "probe::{closure#0}".to_string());
    prog.add("main", main);
    prog.add("probe::{closure#0}", closure_body);
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).expect("wrapped call_mut runs"),
        Value::Int(42)
    );
}

// ----- virtual call with trait generic arguments in the impl key ---------------

const BOB_TY: mir::DefId = mir::DefId::new(0, 60);
const PICK: mir::DefId = mir::DefId::new(0, 61);

#[test]
fn virtual_call_scans_impl_key_with_trait_generics() {
    // trait Pick<Idx> { fn pick(&self) -> i32 }
    // impl Pick<usize> for Bob { fn pick(&self) = self.0 }
    // fn main() -> i32 { _1 = Bob { 42 }; _2 = &_1;
    //   _3 = _2 as &dyn Pick; _0 = Pick::pick(move _3) }
    let bob_ty = mir::Ty::Adt(BOB_TY, vec![]);
    let impl_body = mir::Body::new(
        decls(&[i32_ty(), ref_ty(bob_ty.clone(), false)]),
        1,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Use(mir::Operand::Copy(place(1).deref().field(0, i32_ty()))),
            )],
            ret(),
        )],
    );
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            bob_ty.clone(),
            ref_ty(bob_ty.clone(), false),
            ref_ty(dyn_ty("probe::Pick"), false),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(BOB_TY, vec![], vec![const_i32(42)])),
                    assign(place(2), ref_of(1, false)),
                    assign(place(3), unsize(copy(2), ref_ty(dyn_ty("probe::Pick"), false))),
                ],
                call(
                    fndef_operand(PICK, vec![dyn_ty("probe::Pick")]),
                    vec![move_(3)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.fn_names.insert(BOB_TY, "probe::Bob".to_string());
    prog.fn_names.insert(PICK, "probe::Pick::pick".to_string());
    prog.add("main", main);
    prog.add("<probe::Bob as probe::Pick<usize>>::pick", impl_body);
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).expect("virtual pick runs"),
        Value::Int(42)
    );
}
