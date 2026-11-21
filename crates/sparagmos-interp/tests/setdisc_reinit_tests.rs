//! SetDiscriminant in-place reinitialization tests (Lane CR):
//! hand-built `sparagmos-mir` bodies in the shape rustc emits when an
//! enum place is reused for a different variant:
//!
//!   _1 = E::A(7);
//!   SetDiscriminant(_1, 1);
//!   ((_1 as B).0) = 30;
//!   ((_1 as B).1) = 12;
//!   _2 = Discriminant(_1);
//!   switchInt(copy _2) -> [1: bb1, otherwise: bb2]
//!
//! After `SetDiscriminant` invalidates the old payload, the field
//! writes that follow target slots that do not exist yet in the
//! abstract `Value::Enum`: the mutable place navigation must
//! (re)materialize them as uninitialized slots instead of rejecting
//! the write. The same shape through a `&mut` (a `RefTarget::Place`
//! navigation) and the writes-before-`SetDiscriminant` order are
//! covered too, as is the invariant that a field of the new variant
//! that was never written still reads as uninitialized.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram, MirRunError};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir ------------------------------------

const E_DEF: mir::DefId = mir::DefId::new(0, 77);

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn e_ty() -> mir::Ty {
    mir::Ty::Adt(E_DEF, vec![])
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

fn const_i32(v: i32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u32 as u128,
            size: 4,
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

fn set_disc(place: mir::Place, variant: u32) -> mir::Statement {
    mir::Statement::new(mir::StatementKind::SetDiscriminant {
        place: Box::new(place),
        variant_index: variant,
    })
}

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

fn adt_aggregate(variant: u32, ops: Vec<mir::Operand>) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def: E_DEF,
            variant_index: variant,
            generic_args: vec![],
            user_ty: None,
            field_index: None,
        }),
        ops,
    )
}

/// `(_base as variant).field` where `_base` is local `idx`.
fn variant_field(idx: u32, variant: u32, f: u32) -> mir::Place {
    place(idx).downcast(None, variant).field(f, i32_ty())
}

/// `((*_idx) as variant).field` — the enum behind the reference in
/// local `idx`.
fn deref_variant_field(idx: u32, variant: u32, f: u32) -> mir::Place {
    place(idx)
        .deref()
        .downcast(None, variant)
        .field(f, i32_ty())
}

fn binop(op: mir::BinOp, lhs: mir::Operand, rhs: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::BinaryOp(op, Box::new((lhs, rhs)))
}

fn prog_with(body: mir::Body) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.adt_discriminants.insert(E_DEF, vec![0, 1]);
    prog.add("main", body);
    prog
}

// ----- the standard shape: SetDiscriminant first, then field writes --------

/// `fn main() -> i32 { _1 = E::A(7); SetDiscriminant(_1, 1);
///   ((_1 as B).0) = 30; ((_1 as B).1) = 12;
///   _2 = Discriminant(_1);
///   switchInt(copy _2) -> [1: bb1, otherwise: bb2];
///   bb1: _0 = Add(copy ((_1 as B).0), copy ((_1 as B).1)); return;
///   bb2: _0 = -1; return }` = 42.
#[test]
fn setdisc_then_field_writes_frame_local() {
    let body = mir::Body::new(
        decls(&[i32_ty(), e_ty(), i32_ty()]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), adt_aggregate(0, vec![const_i32(7)])),
                    set_disc(place(1), 1),
                    assign(variant_field(1, 1, 0), mir::Rvalue::Use(const_i32(30))),
                    assign(variant_field(1, 1, 1), mir::Rvalue::Use(const_i32(12))),
                    assign(place(2), mir::Rvalue::Discriminant(place(1))),
                ],
                mir::TerminatorKind::SwitchInt {
                    discr: copy(2),
                    targets: mir::SwitchTargets::new(
                        vec![1],
                        vec![mir::BasicBlock(1)],
                        mir::BasicBlock(2),
                    ),
                },
            ),
            block(
                vec![assign(
                    place(0),
                    binop(
                        mir::BinOp::Add,
                        mir::Operand::Copy(variant_field(1, 1, 0)),
                        mir::Operand::Copy(variant_field(1, 1, 1)),
                    ),
                )],
                ret(),
            ),
            block(
                vec![assign(place(0), mir::Rvalue::Use(const_i32(-1)))],
                ret(),
            ),
        ],
    );
    let prog = prog_with(body);
    let v = interpret_mir(&prog, "main", vec![]).expect("SetDiscriminant reinit must run");
    assert_eq!(v, Value::Int(42));
}

/// The same shape through a `&mut`: the place navigation crosses a
/// `RefTarget::Place` into the caller's frame local before reaching
/// the enum, and the writes must land in that storage.
/// `fn main() -> i32 { _1 = E::A(7); _2 = &mut _1;
///   SetDiscriminant((*_2), 1); (((*_2) as B).0) = 20;
///   (((*_2) as B).1) = 22;
///   _0 = Add(copy ((_1 as B).0), copy ((_1 as B).1)); return }` = 42.
#[test]
fn setdisc_then_field_writes_through_mut_reference() {
    let ref_ty = mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(e_ty()),
        mir::Mutability::Mut,
    );
    let body = mir::Body::new(
        decls(&[i32_ty(), e_ty(), ref_ty]),
        0,
        vec![block(
            vec![
                assign(place(1), adt_aggregate(0, vec![const_i32(7)])),
                assign(
                    place(2),
                    mir::Rvalue::Ref(
                        mir::Region::Erased,
                        mir::BorrowKind::Mut {
                            allow_two_phase_borrow: false,
                        },
                        place(1),
                    ),
                ),
                set_disc(place(2).deref(), 1),
                assign(
                    deref_variant_field(2, 1, 0),
                    mir::Rvalue::Use(const_i32(20)),
                ),
                assign(
                    deref_variant_field(2, 1, 1),
                    mir::Rvalue::Use(const_i32(22)),
                ),
                assign(
                    place(0),
                    binop(
                        mir::BinOp::Add,
                        mir::Operand::Copy(variant_field(1, 1, 0)),
                        mir::Operand::Copy(variant_field(1, 1, 1)),
                    ),
                ),
            ],
            ret(),
        )],
    );
    let prog = prog_with(body);
    let v = interpret_mir(&prog, "main", vec![])
        .expect("SetDiscriminant reinit through &mut must run");
    assert_eq!(v, Value::Int(42));
}

/// Reinitializing back to a variant with fewer fields: the slots of
/// the wider variant must not leak into the narrower one.
/// `fn main() -> i32 { _1 = E::B(1, 2); SetDiscriminant(_1, 0);
///   ((_1 as A).0) = 42; _0 = copy ((_1 as A).0); return }` = 42.
#[test]
fn setdisc_reinit_back_to_first_variant() {
    let body = mir::Body::new(
        decls(&[i32_ty(), e_ty()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    adt_aggregate(1, vec![const_i32(1), const_i32(2)]),
                ),
                set_disc(place(1), 0),
                assign(variant_field(1, 0, 0), mir::Rvalue::Use(const_i32(42))),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(variant_field(1, 0, 0))),
                ),
            ],
            ret(),
        )],
    );
    let prog = prog_with(body);
    let v = interpret_mir(&prog, "main", vec![]).expect("reinit back to A must run");
    assert_eq!(v, Value::Int(42));
}

/// The writes-before-`SetDiscriminant` order (the order the
/// coroutine transform also uses): fields are stored through the
/// downcast of the variant being entered while the tag still names
/// the old variant, and the closing `SetDiscriminant` — now naming
/// the variant the place already holds — must not discard them.
/// `fn main() -> i32 { _1 = E::A(7); ((_1 as B).0) = 11;
///   ((_1 as B).1) = 31; SetDiscriminant(_1, 1);
///   _0 = Add(copy ((_1 as B).0), copy ((_1 as B).1)); return }` = 42.
#[test]
fn field_writes_before_setdisc_survive() {
    let body = mir::Body::new(
        decls(&[i32_ty(), e_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), adt_aggregate(0, vec![const_i32(7)])),
                assign(variant_field(1, 1, 0), mir::Rvalue::Use(const_i32(11))),
                assign(variant_field(1, 1, 1), mir::Rvalue::Use(const_i32(31))),
                set_disc(place(1), 1),
                assign(
                    place(0),
                    binop(
                        mir::BinOp::Add,
                        mir::Operand::Copy(variant_field(1, 1, 0)),
                        mir::Operand::Copy(variant_field(1, 1, 1)),
                    ),
                ),
            ],
            ret(),
        )],
    );
    let prog = prog_with(body);
    let v = interpret_mir(&prog, "main", vec![])
        .expect("writes-before-SetDiscriminant must run");
    assert_eq!(v, Value::Int(42));
}

/// A field of the new variant that was never written is not the old
/// variant's payload and not a default: reading it is an
/// uninitialized read.
/// `fn main() -> i32 { _1 = E::A(7); SetDiscriminant(_1, 1);
///   ((_1 as B).0) = 5; _0 = copy ((_1 as B).1); return }`
#[test]
fn read_of_unwritten_field_after_setdisc_is_uninit() {
    let body = mir::Body::new(
        decls(&[i32_ty(), e_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), adt_aggregate(0, vec![const_i32(7)])),
                set_disc(place(1), 1),
                assign(variant_field(1, 1, 0), mir::Rvalue::Use(const_i32(5))),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(variant_field(1, 1, 1))),
                ),
            ],
            ret(),
        )],
    );
    let prog = prog_with(body);
    let err = interpret_mir(&prog, "main", vec![])
        .expect_err("reading a never-written field of the new variant must fail");
    match &err {
        MirRunError::Interp(InterpError::UninitializedRead { place, .. }) => {
            assert!(
                place.ends_with(".1"),
                "the uninitialized read must be field 1 (the write of field 0 must have succeeded), got place {place}"
            );
        }
        other => panic!("expected UninitializedRead, got {other}"),
    }
}
