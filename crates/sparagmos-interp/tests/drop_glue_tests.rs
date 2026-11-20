//! Drop glue tests (Lane AR): `Drop` terminators whose types have
//! real glue — converted `Drop` impl bodies, field glue, container
//! contents — and the cases that must stay explicit errors.
//!
//! The programs are hand-built `sparagmos-mir` bodies shaped after
//! the driver's output for the corresponding Rust sources. The
//! observable effect channel is a counter local in `main` that the
//! destructors bump through a `&mut i32` stored in the dropped
//! values: `*sink = *sink * 10 + id`, so the final counter encodes
//! exactly which destructors ran, in which order. `main` returns
//! the counter.

use std::collections::HashMap;

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::{InterpError, MirRunError, Value};
use sparagmos_mir as mir;

// ----- shared builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
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

fn const_str(s: &str) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Str(
        s.to_string(),
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

fn drop(place_: mir::Place, next: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Drop {
        place: place_,
        target: mir::BasicBlock(next),
        unwind: mir::UnwindAction::Continue,
        replace: false,
    }
}

fn ref_rvalue(idx: u32) -> mir::Rvalue {
    mir::Rvalue::Ref(
        mir::Region::Erased,
        mir::BorrowKind::Mut {
            allow_two_phase_borrow: false,
        },
        place(idx),
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

// ----- the probe crate's items ------------------------------------------------

const D_TY: mir::DefId = mir::DefId::new(0, 200);
const OUTER_TY: mir::DefId = mir::DefId::new(0, 202);
const S_TY: mir::DefId = mir::DefId::new(0, 204);
const P_TY: mir::DefId = mir::DefId::new(0, 206);
const E_TY: mir::DefId = mir::DefId::new(0, 208);
const VEC_TY: mir::DefId = mir::DefId::new(0, 210);
const BOX_TY: mir::DefId = mir::DefId::new(0, 212);
const STRING_TY: mir::DefId = mir::DefId::new(0, 214);
const G_TY: mir::DefId = mir::DefId::new(0, 216);
const W_TY: mir::DefId = mir::DefId::new(0, 218);
const RESULT_TY: mir::DefId = mir::DefId::new(0, 220);
const CLO_TY: mir::DefId = mir::DefId::new(0, 222);

const D_DROP_NAME: &str = "<probe::D as std::ops::Drop>::drop";
const OUTER_DROP_NAME: &str = "<probe::Outer as std::ops::Drop>::drop";

fn d_ty() -> mir::Ty {
    mir::Ty::Adt(D_TY, vec![])
}

fn probe_names() -> HashMap<mir::DefId, String> {
    HashMap::from([
        (D_TY, "probe::D".to_string()),
        (OUTER_TY, "probe::Outer".to_string()),
        (S_TY, "probe::S".to_string()),
        (P_TY, "probe::P".to_string()),
        (E_TY, "probe::E".to_string()),
        (VEC_TY, "std::vec::Vec".to_string()),
        (BOX_TY, "std::boxed::Box".to_string()),
        (STRING_TY, "std::string::String".to_string()),
        (G_TY, "probe::G".to_string()),
        (W_TY, "othercrate::W".to_string()),
        (RESULT_TY, "std::result::Result".to_string()),
        (CLO_TY, "probe::{closure#0}".to_string()),
    ])
}

/// The destructor of a type whose field `sink_field` is a
/// `&mut i32` and whose field `id_field` is an `i32`:
/// `*self.sink = *self.sink * 10 + self.id`.
fn bump_dtor(self_ty: mir::Ty, sink_field: u32, id_field: u32) -> mir::Body {
    mir::Body::new(
        decls(&[
            mir::Ty::unit(),
            ref_ty(self_ty),
            ref_ty(i32_ty()),
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
                        place(1).deref().field(sink_field, ref_ty(i32_ty())),
                    )),
                ),
                assign(
                    place(3),
                    mir::Rvalue::Use(mir::Operand::Copy(
                        place(1).deref().field(id_field, i32_ty()),
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

/// A program whose `main` starts with a counter in `_1` and a
/// `&mut` to it in `_2`; `build` fills the remaining locals/blocks.
fn run_main(main: mir::Body, extra: &[(&str, mir::Body)]) -> Result<Value, MirRunError> {
    let mut prog = MirProgram::new();
    prog.add("probe::main", main);
    for (name, body) in extra {
        prog.add(*name, body.clone());
    }
    prog.fn_names = probe_names();
    interpret_mir(&prog, "probe::main", vec![])
}

// ----- destructor bodies run ----------------------------------------------------

#[test]
fn destructor_body_runs_at_scope_end() {
    // struct D { sink: &mut i32, id: i32 } with a Drop impl.
    // fn main() -> i32 { _1 = 0; _2 = &mut _1; _3 = D { _2, 7 };
    //                    drop(_3); return _1; }
    let main = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), ref_ty(i32_ty()), d_ty()]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_rvalue(1)),
                    assign(
                        place(3),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(2), const_i32(7)]),
                    ),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(3), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run");
    assert_eq!(value, Value::Int(7));
}

#[test]
fn nested_destructors_run_outer_first() {
    // struct Outer { inner: D, sink: &mut i32, id: i32 } with a Drop
    // impl; D's own impl bumps by its id (1), Outer's by its id (2).
    // Native order: Outer's body first, then the field's glue: the
    // counter becomes 0*10+2 = 2, then 2*10+1 = 21.
    let outer_ty = mir::Ty::Adt(OUTER_TY, vec![]);
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            ref_ty(i32_ty()),
            outer_ty.clone(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_rvalue(1)),
                    assign(
                        place(3),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(2), const_i32(1)]),
                    ),
                    assign(place(4), ref_rvalue(1)),
                    assign(
                        place(5),
                        aggregate_adt(
                            OUTER_TY,
                            0,
                            vec![],
                            vec![move_(3), move_(4), const_i32(2)],
                        ),
                    ),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(5), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(
        main,
        &[
            (D_DROP_NAME, bump_dtor(d_ty(), 0, 1)),
            (OUTER_DROP_NAME, bump_dtor(outer_ty, 1, 2)),
        ],
    )
    .expect("run");
    assert_eq!(value, Value::Int(21));
}

// ----- trivial glue ----------------------------------------------------------------

#[test]
fn struct_with_string_field_and_no_impl_drops() {
    // struct S { s: String, n: i32 } (no Drop impl anywhere).
    // fn main() -> i32 { _1 = S { "hi", 3 }; _2 = _1.n; drop(_1);
    //                    return _2; }
    let s_ty = mir::Ty::Adt(S_TY, vec![]);
    let main = mir::Body::new(
        decls(&[i32_ty(), s_ty, i32_ty()]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        aggregate_adt(
                            S_TY,
                            0,
                            vec![],
                            vec![const_str("hi"), const_i32(3)],
                        ),
                    ),
                    assign(
                        place(2),
                        mir::Rvalue::Use(mir::Operand::Copy(place(1).field(1, i32_ty()))),
                    ),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(1), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    );
    let value = run_main(main, &[]).expect("run");
    assert_eq!(value, Value::Int(3));
}

// ----- container contents ------------------------------------------------------------

#[test]
fn vec_elements_drop_in_order() {
    // A Vec<D> value (in this model an ADT value whose fields are
    // the elements) with ids 1 and 2: the counter becomes 12.
    let vec_ty = mir::Ty::Adt(VEC_TY, vec![d_ty()]);
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            vec_ty,
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_rvalue(1)),
                    assign(
                        place(3),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(2), const_i32(1)]),
                    ),
                    assign(place(4), ref_rvalue(1)),
                    assign(
                        place(5),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(4), const_i32(2)]),
                    ),
                    assign(
                        place(6),
                        aggregate_adt(VEC_TY, 0, vec![d_ty()], vec![move_(3), move_(5)]),
                    ),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(6), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run");
    assert_eq!(value, Value::Int(12));
}

#[test]
fn array_elements_drop_in_order() {
    // [D; 2] with ids 1 and 2: the counter becomes 12.
    let array_ty = mir::Ty::Array(Box::new(d_ty()), mir::ArrayLen::Known(2));
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            array_ty,
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_rvalue(1)),
                    assign(
                        place(3),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(2), const_i32(1)]),
                    ),
                    assign(place(4), ref_rvalue(1)),
                    assign(
                        place(5),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(4), const_i32(2)]),
                    ),
                    assign(
                        place(6),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Array(d_ty())),
                            vec![move_(3), move_(5)],
                        ),
                    ),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(6), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run");
    assert_eq!(value, Value::Int(12));
}

#[test]
fn box_pointee_drops() {
    // Box<D> (an ADT value holding the pointee as its one field):
    // the pointee's destructor runs.
    let box_ty = mir::Ty::Adt(BOX_TY, vec![d_ty()]);
    let main = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), ref_ty(i32_ty()), d_ty(), box_ty]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_rvalue(1)),
                    assign(
                        place(3),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(2), const_i32(5)]),
                    ),
                    assign(
                        place(4),
                        aggregate_adt(BOX_TY, 0, vec![d_ty()], vec![move_(3)]),
                    ),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(4), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run");
    assert_eq!(value, Value::Int(5));
}

// ----- enum variants -------------------------------------------------------------------

#[test]
fn enum_variant_glue_follows_the_live_variant() {
    // enum E { A(D), B(i32) }: dropping an `A` runs D's destructor;
    // dropping a `B` runs nothing.
    let e_ty = mir::Ty::Adt(E_TY, vec![]);
    let build = |variant: u32, payload: Vec<mir::Operand>, with_d: bool| {
        let mut tys = vec![i32_ty(), i32_ty(), ref_ty(i32_ty())];
        let mut stmts = vec![
            assign(place(1), mir::Rvalue::Use(const_i32(0))),
            assign(place(2), ref_rvalue(1)),
        ];
        if with_d {
            tys.push(d_ty());
            stmts.push(assign(place(3), aggregate_adt(D_TY, 0, vec![], payload)));
            tys.push(e_ty.clone());
            stmts.push(assign(
                place(4),
                aggregate_adt(E_TY, variant, vec![], vec![move_(3)]),
            ));
            mir::Body::new(
                decls(&tys),
                0,
                vec![
                    block(
                        stmts,
                        mir::TerminatorKind::Goto {
                            target: mir::BasicBlock(1),
                        },
                    ),
                    block(vec![], drop(place(4), 2)),
                    block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
                ],
            )
        } else {
            tys.push(e_ty.clone());
            stmts.push(assign(
                place(3),
                aggregate_adt(E_TY, variant, vec![], payload),
            ));
            mir::Body::new(
                decls(&tys),
                0,
                vec![
                    block(
                        stmts,
                        mir::TerminatorKind::Goto {
                            target: mir::BasicBlock(1),
                        },
                    ),
                    block(vec![], drop(place(3), 2)),
                    block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
                ],
            )
        }
    };
    // A(D { .., 4 }): the payload's payload operands are the D
    // aggregate's operands (sink ref in _2, id literal).
    let a_main = build(0, vec![move_(2), const_i32(4)], true);
    let value = run_main(a_main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run A");
    assert_eq!(value, Value::Int(4));
    // B(9): no destructor anywhere.
    let b_main = build(1, vec![const_i32(9)], false);
    let value = run_main(b_main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run B");
    assert_eq!(value, Value::Int(0));
}

// ----- partial moves ----------------------------------------------------------------------

#[test]
fn moved_out_field_is_not_dropped_twice() {
    // struct P { d: D, x: i32 } (no impl of its own).
    // fn main() -> i32 { _1 = 0; _2 = &mut _1; _3 = D { _2, 3 };
    //   _4 = P { _3, 1 }; _5 = move _4.d; drop(_4); drop(_5);
    //   return _1; }
    // D's destructor must run exactly once (for _5): counter = 3,
    // not 33.
    let p_ty = mir::Ty::Adt(P_TY, vec![]);
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            p_ty,
            d_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_rvalue(1)),
                    assign(
                        place(3),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(2), const_i32(3)]),
                    ),
                    assign(
                        place(4),
                        aggregate_adt(P_TY, 0, vec![], vec![move_(3), const_i32(1)]),
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
            block(vec![], drop(place(4), 2)),
            block(vec![], drop(place(5), 3)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run");
    assert_eq!(value, Value::Int(3));
}

// ----- cases that stay explicit errors -------------------------------------------------------

#[test]
fn generic_dtor_without_a_body_stays_unsupported() {
    // struct G<T> { v: T } with `impl<T> Drop for G<T>`: the name
    // table records the impl (with its parameter spelling), but a
    // generic impl gets no monomorphic body in the program, so the
    // drop cannot be executed and must not be skipped either.
    let g_ty = mir::Ty::Adt(G_TY, vec![i32_ty()]);
    let main = mir::Body::new(
        decls(&[mir::Ty::unit(), g_ty]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    aggregate_adt(G_TY, 0, vec![i32_ty()], vec![const_i32(1)]),
                )],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(1), 2)),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("probe::main", main);
    let mut names = probe_names();
    names.insert(
        mir::DefId::new(0, 217),
        "<probe::G<T> as std::ops::Drop>::drop".to_string(),
    );
    prog.fn_names = names;
    let err = interpret_mir(&prog, "probe::main", vec![]).expect_err("must stay unsupported");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedTerminator { ref kind }) if kind == "Drop"
        ),
        "expected unsupported Drop, got {err}"
    );
}

#[test]
fn foreign_type_that_might_hide_an_impl_stays_unsupported() {
    // struct W { x: i32 } from another crate: the name table cannot
    // rule out a `Drop` impl for it, so even a scalar-only shape
    // stays an explicit error rather than a silent skip.
    let w_ty = mir::Ty::Adt(W_TY, vec![]);
    let main = mir::Body::new(
        decls(&[mir::Ty::unit(), w_ty]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    aggregate_adt(W_TY, 0, vec![], vec![const_i32(1)]),
                )],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(1), 2)),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("probe::main", main);
    prog.fn_names = probe_names();
    let err = interpret_mir(&prog, "probe::main", vec![]).expect_err("must stay unsupported");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedTerminator { ref kind }) if kind == "Drop"
        ),
        "expected unsupported Drop, got {err}"
    );
}

// ----- generic destructors, several instantiations (Lane BS) -----------------

/// Like `bump_dtor`, but the counter is scaled by `scale` before the
/// id is added, so two instantiations of one generic destructor are
/// observably different bodies.
fn bump_dtor_scaled(self_ty: mir::Ty, sink_field: u32, id_field: u32, scale: i32) -> mir::Body {
    mir::Body::new(
        decls(&[
            mir::Ty::unit(),
            ref_ty(self_ty),
            ref_ty(i32_ty()),
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
                        place(1).deref().field(sink_field, ref_ty(i32_ty())),
                    )),
                ),
                assign(
                    place(3),
                    mir::Rvalue::Use(mir::Operand::Copy(
                        place(1).deref().field(id_field, i32_ty()),
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
                        Box::new((copy(4), const_i32(scale))),
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

fn const_u8(v: u8) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
            size: 1,
        },
    )))
}

#[test]
fn generic_dtor_two_instantiations_pick_their_own_bodies() {
    // struct G<T> { sink: &mut i32, id: i32, marker: T } with a
    // generic Drop impl. The program holds two monomorphizations of
    // the destructor, `G<i32>` (counter *10 + id) and `G<u8>`
    // (counter *100 + id); dropping each value must run its own
    // instantiation: (0*10+3) = 3, then (3*100+4) = 304. Before the
    // glue builder keyed destructors by their full instantiated
    // name, two candidate bodies made the lookup give up and the
    // drops were refused outright.
    let u8_ty = mir::Ty::Uint(mir::UintTy::U8);
    let g_i32 = mir::Ty::Adt(G_TY, vec![i32_ty()]);
    let g_u8 = mir::Ty::Adt(G_TY, vec![u8_ty.clone()]);
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty()),
            g_i32.clone(),
            ref_ty(i32_ty()),
            g_u8.clone(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_rvalue(1)),
                    assign(
                        place(3),
                        aggregate_adt(
                            G_TY,
                            0,
                            vec![i32_ty()],
                            vec![move_(2), const_i32(3), const_i32(0)],
                        ),
                    ),
                    assign(place(4), ref_rvalue(1)),
                    assign(
                        place(5),
                        aggregate_adt(
                            G_TY,
                            0,
                            vec![u8_ty],
                            vec![move_(4), const_i32(4), const_u8(0)],
                        ),
                    ),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(3), 2)),
            block(vec![], drop(place(5), 3)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(
        main,
        &[
            (
                "<probe::G<i32> as std::ops::Drop>::drop",
                bump_dtor_scaled(g_i32, 0, 1, 10),
            ),
            (
                "<probe::G<u8> as std::ops::Drop>::drop",
                bump_dtor_scaled(g_u8, 0, 1, 100),
            ),
        ],
    )
    .expect("run");
    assert_eq!(value, Value::Int(304));
}

// ----- Result glue (Lane BS) ---------------------------------------------------

#[test]
fn result_payload_dtor_runs_for_ok_and_err() {
    // Result<D, D>: the modelled glue follows the live variant and
    // drops the payload at field 0, exactly like `Option`.
    let result_ty = mir::Ty::Adt(RESULT_TY, vec![d_ty(), d_ty()]);
    let build = |variant: u32, id: i32| {
        mir::Body::new(
            decls(&[i32_ty(), i32_ty(), ref_ty(i32_ty()), d_ty(), result_ty.clone()]),
            0,
            vec![
                block(
                    vec![
                        assign(place(1), mir::Rvalue::Use(const_i32(0))),
                        assign(place(2), ref_rvalue(1)),
                        assign(
                            place(3),
                            aggregate_adt(D_TY, 0, vec![], vec![move_(2), const_i32(id)]),
                        ),
                        assign(
                            place(4),
                            aggregate_adt(
                                RESULT_TY,
                                variant,
                                vec![d_ty(), d_ty()],
                                vec![move_(3)],
                            ),
                        ),
                    ],
                    mir::TerminatorKind::Goto {
                        target: mir::BasicBlock(1),
                    },
                ),
                block(vec![], drop(place(4), 2)),
                block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
            ],
        )
    };
    let value = run_main(build(0, 5), &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run Ok");
    assert_eq!(value, Value::Int(5));
    let value = run_main(build(1, 6), &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run Err");
    assert_eq!(value, Value::Int(6));
}

// ----- closure glue (Lane BS) ----------------------------------------------------

fn closure_ty(upvars: Vec<mir::Ty>) -> mir::Ty {
    mir::Ty::Closure(
        CLO_TY,
        vec![
            mir::Ty::Int(mir::IntTy::I8),
            mir::Ty::FnPtr(Box::new(mir::FnSig::new(
                vec![mir::Ty::Tuple(vec![])],
                mir::Ty::unit(),
            ))),
            mir::Ty::Tuple(upvars),
        ],
    )
}

fn closure_rvalue(clo_ty: &mir::Ty, ops: Vec<mir::Operand>) -> mir::Rvalue {
    let generic_args = match clo_ty {
        mir::Ty::Closure(_, args) => args.clone(),
        _ => unreachable!(),
    };
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Closure {
            def: CLO_TY,
            generic_args,
        }),
        ops,
    )
}

#[test]
fn closure_capture_dtor_runs() {
    // A closure capturing a `D` by move: dropping the closure drops
    // the capture. Before the glue builder planned `Ty::Closure`,
    // this drop was refused.
    let clo_ty = closure_ty(vec![d_ty()]);
    let main = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), ref_ty(i32_ty()), d_ty(), clo_ty.clone()]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_rvalue(1)),
                    assign(
                        place(3),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(2), const_i32(9)]),
                    ),
                    assign(place(4), closure_rvalue(&clo_ty, vec![move_(3)])),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(4), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run");
    assert_eq!(value, Value::Int(9));
}

#[test]
fn closure_partially_moved_capture_is_skipped() {
    // A closure capturing two `D`s; the first capture is moved out
    // (and never dropped), so dropping the closure must run only
    // the second capture's destructor: the counter ends at 10, not
    // 9*10+10 or 10*10+9.
    let clo_ty = closure_ty(vec![d_ty(), d_ty()]);
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            clo_ty.clone(),
            d_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_rvalue(1)),
                    assign(
                        place(3),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(2), const_i32(9)]),
                    ),
                    assign(place(4), ref_rvalue(1)),
                    assign(
                        place(5),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(4), const_i32(10)]),
                    ),
                    assign(
                        place(6),
                        closure_rvalue(&clo_ty, vec![move_(3), move_(5)]),
                    ),
                    // Move capture 0 out of the closure; `_7` is
                    // never dropped in this body.
                    assign(
                        place(7),
                        mir::Rvalue::Use(mir::Operand::Move(place(6).field(0, d_ty()))),
                    ),
                ],
                mir::TerminatorKind::Goto {
                    target: mir::BasicBlock(1),
                },
            ),
            block(vec![], drop(place(6), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run");
    assert_eq!(value, Value::Int(10));
}
