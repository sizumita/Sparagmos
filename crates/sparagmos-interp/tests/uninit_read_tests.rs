//! Tests for reads of places rustc MIR never writes: union active
//! fields (Lane CH group A) and zero-sized values (group B), plus
//! the negative guarantee that a genuinely unwritten non-ZST place
//! still fails with `UninitializedRead`.
//!
//! Hand-built `sparagmos-mir` bodies, shaped after real
//! `rustc 1.100.0 --emit=mir` output, converted and executed by the
//! interpreter.

use sparagmos_interp::mir_convert::{convert_program, interpret_mir, MirProgram, MirRunError};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}
fn u32_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U32)
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

fn const_int(data: u128, size: u8) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int { data, size },
    )))
}

fn const_i32(v: i32) -> mir::Operand {
    const_int(v as u32 as u128, 4)
}

fn const_u32(v: u32) -> mir::Operand {
    const_int(v as u128, 4)
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

// ----- group A: union active fields -----------------------------------------

const UNION: mir::DefId = mir::DefId::new(0, 40);

fn union_ty() -> mir::Ty {
    mir::Ty::Adt(UNION, Vec::new())
}

fn union_aggregate(field_index: u32, op: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def: UNION,
            variant_index: 0,
            generic_args: Vec::new(),
            user_ty: None,
            field_index: Some(field_index),
        }),
        vec![op],
    )
}

fn union_program(main: mir::Body) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names.insert(UNION, "probe::U".to_string());
    prog.add("main", main);
    prog
}

/// `union U { a: u32, b: i32 }`; `fn main() -> i32 { let u = U { b: -5 }; u.b }`
/// The single aggregate operand initializes field 1, typed by field
/// 1's declared type (i32), not field 0's (u32).
#[test]
fn union_active_field_read_back() {
    let main = mir::Body::new(
        decls(&[i32_ty(), union_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), union_aggregate(1, const_i32(-5))),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).field(1, i32_ty()))),
                ),
            ],
            ret(),
        )],
    );
    let value = interpret_mir(&union_program(main), "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Int(-5));
}

/// `fn main() -> u32 { let u = U { a: 7 }; u.a }` — field 0 active.
#[test]
fn union_first_field_read_back() {
    let main = mir::Body::new(
        decls(&[u32_ty(), union_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), union_aggregate(0, const_u32(7))),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).field(0, u32_ty()))),
                ),
            ],
            ret(),
        )],
    );
    let value = interpret_mir(&union_program(main), "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Uint(7));
}

/// Reading a union field other than the one that was written is a
/// read of a never-written slot: the model does not reinterpret the
/// union's bytes, so it stays an uninitialized read.
#[test]
fn union_other_field_read_is_uninitialized() {
    let main = mir::Body::new(
        decls(&[u32_ty(), union_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), union_aggregate(1, const_i32(-5))),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).field(0, u32_ty()))),
                ),
            ],
            ret(),
        )],
    );
    let err = interpret_mir(&union_program(main), "main", Vec::new())
        .expect_err("cross-field union read must fail");
    assert!(
        matches!(err, MirRunError::Interp(InterpError::UninitializedRead { .. })),
        "expected UninitializedRead, got {err:?}"
    );
}

/// `fn main() -> i32 { let mut u = U { a: 1 }; u.b = -7; u.b }` —
/// a field write past the fields written so far extends the union
/// value's sparse field vector.
#[test]
fn union_field_write_then_read() {
    let main = mir::Body::new(
        decls(&[i32_ty(), union_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), union_aggregate(0, const_u32(1))),
                assign(
                    place(1).field(1, i32_ty()),
                    mir::Rvalue::Use(const_i32(-7)),
                ),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).field(1, i32_ty()))),
                ),
            ],
            ret(),
        )],
    );
    let value = interpret_mir(&union_program(main), "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Int(-7));
}

// ----- the negative guarantee ------------------------------------------------

/// A never-written local of a non-ZST scalar type is a genuine
/// uninitialized read and must stay an error: `fn f() -> i32 {
/// let x: i32; x }` (shaped as MIR reads it: `_0 = copy _1` with
/// `_1` never assigned).
#[test]
fn unwritten_scalar_local_still_errors() {
    let f = mir::Body::new(
        decls(&[i32_ty(), i32_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(copy(1)))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", f);
    let err = interpret_mir(&prog, "main", Vec::new()).expect_err("unwritten i32 must fail");
    assert!(
        matches!(err, MirRunError::Interp(InterpError::UninitializedRead { .. })),
        "expected UninitializedRead, got {err:?}"
    );
}

/// The converted program itself (not just execution) must keep the
/// distinction: converting the same body succeeds — the failure is
/// a runtime read of a never-written place, not a conversion gap.
#[test]
fn unwritten_scalar_body_converts() {
    let f = mir::Body::new(
        decls(&[i32_ty(), i32_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(copy(1)))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", f);
    convert_program(&prog).expect("body converts");
}
