//! Drop glue tests for explicit `std::mem::drop` calls (Issue #61):
//! a `mem::drop` call must run the same glue as a `Drop` terminator
//! for the same type, so destructor side effects are identical on
//! both paths, and a call whose glue cannot be accounted for stays
//! an explicit error instead of silently discarding the value.
//!
//! The programs are hand-built `sparagmos-mir` bodies shaped after
//! the driver's output for the corresponding Rust sources, following
//! the `drop_glue_tests` fixture style. The observable effect
//! channel is a counter local in `main` that the destructors bump
//! through a `&mut i32` stored in the dropped values:
//! `*sink = *sink * 10 + id`, so the final counter encodes exactly
//! which destructors ran, in which order. `main` returns the counter.

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

fn fndef(def: mir::DefId, generic_args: Vec<mir::Ty>) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def, generic_args)),
        user_ty: None,
        ty: None,
        generic_type_ids: None,
    })
}

/// `std::mem::drop::<ty>(move _arg) -> _dest; goto next`, the shape
/// rustc emits for an explicit `drop(x)` call.
fn mem_drop(ty: mir::Ty, arg: u32, dest: u32, next: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func: fndef(MEM_DROP_FN, vec![ty]),
        args: vec![move_(arg)],
        destination: place(dest),
        target: Some(mir::BasicBlock(next)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

// ----- the probe crate's items ------------------------------------------------

const D_TY: mir::DefId = mir::DefId::new(0, 200);
const OUTER_TY: mir::DefId = mir::DefId::new(0, 202);
const S_TY: mir::DefId = mir::DefId::new(0, 204);
const VEC_TY: mir::DefId = mir::DefId::new(0, 210);
const G_TY: mir::DefId = mir::DefId::new(0, 216);
const W_TY: mir::DefId = mir::DefId::new(0, 218);
const MEM_DROP_FN: mir::DefId = mir::DefId::new(0, 240);

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
        (VEC_TY, "std::vec::Vec".to_string()),
        (G_TY, "probe::G".to_string()),
        (W_TY, "othercrate::W".to_string()),
        (MEM_DROP_FN, "std::mem::drop".to_string()),
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

fn run_main(main: mir::Body, extra: &[(&str, mir::Body)]) -> Result<Value, MirRunError> {
    let mut prog = MirProgram::new();
    prog.add("probe::main", main);
    for (name, body) in extra {
        prog.add(*name, body.clone());
    }
    prog.fn_names = probe_names();
    interpret_mir(&prog, "probe::main", vec![])
}

// ----- the destructor runs through mem::drop ---------------------------------

#[test]
fn explicit_mem_drop_runs_destructor() {
    // struct D { sink: &mut i32, id: i32 } with a Drop impl.
    // fn main() -> i32 { _1 = 0; _2 = &mut _1; _3 = D { _2, 7 };
    //                    std::mem::drop(_3); return _1; }
    // The scope-end version of this program (drop_glue_tests)
    // returns 7; the explicit call must match it.
    let main = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), ref_ty(i32_ty()), d_ty(), mir::Ty::unit()]),
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
                mem_drop(d_ty(), 3, 4, 1),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run");
    assert_eq!(value, Value::Int(7));
}

#[test]
fn explicit_and_scope_end_drop_run_identical_glue() {
    // One value dropped explicitly (id 3), another at scope end
    // (id 4): the counter becomes 3, then 3*10+4 = 34 — both paths
    // run the same destructor, in program order.
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            mir::Ty::unit(),
            ref_ty(i32_ty()),
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
                    assign(place(5), ref_rvalue(1)),
                    assign(
                        place(6),
                        aggregate_adt(D_TY, 0, vec![], vec![move_(5), const_i32(4)]),
                    ),
                ],
                mem_drop(d_ty(), 3, 4, 1),
            ),
            block(vec![], drop(place(6), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run");
    assert_eq!(value, Value::Int(34));
}

// ----- containers and nesting through mem::drop --------------------------------

#[test]
fn mem_drop_of_vec_runs_element_destructors_in_order() {
    // A Vec<D> (in this model an ADT value whose fields are the
    // elements) with ids 1 and 2, dropped explicitly: the counter
    // becomes 12, exactly as for the scope-end drop.
    let vec_ty = mir::Ty::Adt(VEC_TY, vec![d_ty()]);
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            vec_ty.clone(),
            mir::Ty::unit(),
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
                mem_drop(vec_ty, 6, 7, 1),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let value = run_main(main, &[(D_DROP_NAME, bump_dtor(d_ty(), 0, 1))]).expect("run");
    assert_eq!(value, Value::Int(12));
}

#[test]
fn mem_drop_of_struct_with_impl_runs_outer_then_fields() {
    // struct Outer { inner: D, sink: &mut i32, id: i32 } with a Drop
    // impl, dropped explicitly. Native order: Outer's body first
    // (id 2), then the field's glue (id 1): the counter becomes 21,
    // exactly as for the scope-end drop.
    let outer_ty = mir::Ty::Adt(OUTER_TY, vec![]);
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty()),
            d_ty(),
            ref_ty(i32_ty()),
            outer_ty.clone(),
            mir::Ty::unit(),
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
                mem_drop(outer_ty.clone(), 5, 6, 1),
            ),
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

// ----- glue-free values still drop through the call ------------------------------

#[test]
fn mem_drop_of_plain_data_still_works() {
    // struct S { s: String, n: i32 } (no Drop impl anywhere) and a
    // plain i32, both dropped explicitly: nothing observable runs,
    // execution continues, and the value read before the drops is
    // returned.
    let s_ty = mir::Ty::Adt(S_TY, vec![]);
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            s_ty,
            i32_ty(),
            i32_ty(),
            mir::Ty::unit(),
            mir::Ty::unit(),
        ]),
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
                    assign(place(3), mir::Rvalue::Use(const_i32(9))),
                ],
                mem_drop(mir::Ty::Adt(S_TY, vec![]), 1, 4, 1),
            ),
            block(vec![], mem_drop(i32_ty(), 3, 5, 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    );
    let value = run_main(main, &[]).expect("run");
    assert_eq!(value, Value::Int(3));
}

// ----- cases that stay explicit errors --------------------------------------------

#[test]
fn mem_drop_with_generic_dtor_without_a_body_stays_unsupported() {
    // struct G<T> { v: T } with `impl<T> Drop for G<T>`: the name
    // table records the impl, but a generic impl gets no monomorphic
    // body in the program, so the explicit drop cannot be executed
    // and must not be skipped either.
    let g_ty = mir::Ty::Adt(G_TY, vec![i32_ty()]);
    let main = mir::Body::new(
        decls(&[mir::Ty::unit(), g_ty.clone(), mir::Ty::unit()]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    aggregate_adt(G_TY, 0, vec![i32_ty()], vec![const_i32(1)]),
                )],
                mem_drop(g_ty, 1, 2, 1),
            ),
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
            MirRunError::Interp(InterpError::UnsupportedTerminator { ref kind })
                if kind == "Call to std::mem::drop (drop glue cannot be accounted for)"
        ),
        "expected unsupported mem::drop, got {err}"
    );
}

#[test]
fn mem_drop_of_foreign_type_that_might_hide_an_impl_stays_unsupported() {
    // struct W { x: i32 } from another crate: the name table cannot
    // rule out a `Drop` impl for it, so the explicit drop stays an
    // explicit error rather than a silent skip.
    let w_ty = mir::Ty::Adt(W_TY, vec![]);
    let main = mir::Body::new(
        decls(&[mir::Ty::unit(), w_ty.clone(), mir::Ty::unit()]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    aggregate_adt(W_TY, 0, vec![], vec![const_i32(1)]),
                )],
                mem_drop(w_ty, 1, 2, 1),
            ),
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
            MirRunError::Interp(InterpError::UnsupportedTerminator { ref kind })
                if kind == "Call to std::mem::drop (drop glue cannot be accounted for)"
        ),
        "expected unsupported mem::drop, got {err}"
    );
}
