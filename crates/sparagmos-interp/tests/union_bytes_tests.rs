//! Tests for cross-field union writes: a partial (or narrower)
//! write through one field of a union composing with the bytes
//! an earlier write through another field left behind.
//!
//! The value model holds one field's value per union, so before
//! byte storage these writes failed with an uninitialized read
//! raised by the write itself (`navigate_mut_grow`'s union arm
//! refuses a partial write into a field the union does not
//! currently hold). The composition now decodes the written
//! field's value from the held field, applies the write inside
//! it, and stores the union back — in its aggregate form when
//! the written field's span covers the held field's, and in its
//! byte form (`Value::UnionBytes`) when the held field is
//! wider, so the bytes past the written field survive.
//!
//! Hand-built `sparagmos-mir` bodies, shaped after real
//! `rustc 1.100.0 --emit=mir` output, converted and executed by
//! the interpreter. Every test here fails on the interpreter
//! without the byte-storage composition.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram, MirRunError};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir -----------------------------------

fn u8_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U8)
}
fn u16_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U16)
}
fn u32_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U32)
}
fn u64_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U64)
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

fn union_aggregate(def: mir::DefId, field_index: u32, op: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def,
            variant_index: 0,
            generic_args: Vec::new(),
            user_ty: None,
            field_index: Some(field_index),
        }),
        vec![op],
    )
}

fn union_shape(field_tys: Vec<mir::Ty>) -> mir::AdtShape {
    mir::AdtShape {
        kind: mir::AdtShapeKind::Union,
        variants: vec![field_tys],
    }
}

fn program_with_union(
    def: mir::DefId,
    name: &str,
    field_tys: Vec<mir::Ty>,
    main: mir::Body,
) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names.insert(def, name.to_string());
    prog.adt_shapes.insert(def, union_shape(field_tys));
    prog.add("main", main);
    prog
}

// ----- equal widths: the aggregate form switches fields ---------------------

const U_PAIR16: mir::DefId = mir::DefId::new(0, 60);

fn pair16_ty() -> mir::Ty {
    mir::Ty::Tuple(vec![u8_ty(), u8_ty()])
}

fn union_pair16_ty() -> mir::Ty {
    mir::Ty::Adt(U_PAIR16, Vec::new())
}

/// `union U { a: (u8, u8), b: u16 }`
/// `fn main() -> u16 { let mut u = U { b: 0xDEDE }; u.a.1 = 0xBE; u.b }`
/// The partial write into field `a` composes with the bytes
/// field `b` holds: the low byte survives, the high byte is the
/// written one.
#[test]
fn cross_field_partial_write_equal_widths() {
    let main = mir::Body::new(
        decls(&[u16_ty(), union_pair16_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), union_aggregate(U_PAIR16, 1, const_int(0xDEDE, 2))),
                assign(
                    place(1).field(0, pair16_ty()).field(1, u8_ty()),
                    mir::Rvalue::Use(const_int(0xBE, 1)),
                ),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).field(1, u16_ty()))),
                ),
            ],
            ret(),
        )],
    );
    let prog = program_with_union(U_PAIR16, "probe::U", vec![pair16_ty(), u16_ty()], main);
    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Uint(0xBEDE));
}

// ----- wider held field: the byte form keeps the tail -----------------------

const U_WIDE: mir::DefId = mir::DefId::new(0, 61);

fn union_wide_ty() -> mir::Ty {
    mir::Ty::Adt(U_WIDE, Vec::new())
}

fn wide_program(tail: Vec<mir::Statement>, ret_ty: mir::Ty) -> MirProgram {
    let mut stmts = vec![
        assign(
            place(1),
            union_aggregate(U_WIDE, 1, const_int(0x1122_3344_5566_7788, 8)),
        ),
        assign(
            place(1).field(0, pair16_ty()).field(0, u8_ty()),
            mir::Rvalue::Use(const_int(0x99, 1)),
        ),
    ];
    stmts.extend(tail);
    let main = mir::Body::new(
        decls(&[ret_ty, union_wide_ty()]),
        0,
        vec![block(stmts, ret())],
    );
    program_with_union(U_WIDE, "probe::U", vec![pair16_ty(), u64_ty()], main)
}

/// `union U { a: (u8, u8), b: u64 }`
/// `fn main() -> u64 { let mut u = U { b: 0x1122334455667788 }; u.a.0 = 0x99; u.b }`
/// Field `b` is wider than the written field: the union's
/// storage continues as bytes, and the bytes past field `a`
/// survive the write.
#[test]
fn cross_field_partial_write_wider_held_field() {
    let prog = wide_program(
        vec![assign(
            place(0),
            mir::Rvalue::Use(mir::Operand::Copy(place(1).field(1, u64_ty()))),
        )],
        u64_ty(),
    );
    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Uint(0x1122_3344_5566_7799));
}

/// The untouched byte of the written field itself also reads
/// back through the byte form: `u.a.1` is the byte `b`'s write
/// left at offset 1.
/// `fn main() -> u8 { let mut u = U { b: 0x1122334455667788 }; u.a.0 = 0x99; u.a.1 }`
#[test]
fn byte_form_field_read_back() {
    let prog = wide_program(
        vec![assign(
            place(0),
            mir::Rvalue::Use(mir::Operand::Copy(
                place(1).field(0, pair16_ty()).field(1, u8_ty()),
            )),
        )],
        u8_ty(),
    );
    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Uint(0x77));
}

// ----- a whole-field write into a narrower field ----------------------------

const U_NARROW: mir::DefId = mir::DefId::new(0, 62);

/// `union U { a: u8, b: u64 }`
/// `fn main() -> u64 { let mut u = U { b: 0xDEADBEEF_DEADBEEF }; u.a = 0x11; u.b }`
/// A whole-field write into the narrower field still composes:
/// only the byte field `a` covers changes.
#[test]
fn cross_field_whole_write_narrower_field() {
    let main = mir::Body::new(
        decls(&[u64_ty(), mir::Ty::Adt(U_NARROW, Vec::new())]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    union_aggregate(U_NARROW, 1, const_int(0xDEAD_BEEF_DEAD_BEEF, 8)),
                ),
                assign(
                    place(1).field(0, u8_ty()),
                    mir::Rvalue::Use(const_int(0x11, 1)),
                ),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).field(1, u64_ty()))),
                ),
            ],
            ret(),
        )],
    );
    let prog = program_with_union(U_NARROW, "probe::U", vec![u8_ty(), u64_ty()], main);
    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Uint(0xDEAD_BEEF_DEAD_BE11));
}

// ----- uninitialised bytes stay uninitialised in the byte form --------------

const U_PARTS: mir::DefId = mir::DefId::new(0, 63);

fn pair64_ty() -> mir::Ty {
    mir::Ty::Tuple(vec![u32_ty(), u32_ty()])
}

fn pair128_ty() -> mir::Ty {
    mir::Ty::Tuple(vec![u64_ty(), u64_ty()])
}

fn union_parts_ty() -> mir::Ty {
    mir::Ty::Adt(U_PARTS, Vec::new())
}

/// `union U { a: (u32, u32), b: u64, c: u16, d: (u64, u64) }`
/// where the union starts out holding field `a`'s pair and a
/// whole-field write into the narrower field `c` then switches
/// the union to its byte form.
fn parts_program(tail: Vec<mir::Statement>, ret_ty: mir::Ty) -> MirProgram {
    // _2 is field `a`'s pair value, built as one tuple.
    let main = mir::Body::new(
        decls(&[ret_ty, union_parts_ty(), pair64_ty()]),
        0,
        vec![block(
            {
                let mut stmts = vec![
                    assign(
                        place(2),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![const_int(0x1111_1111, 4), const_int(0x2222_2222, 4)],
                        ),
                    ),
                    assign(place(1), union_aggregate(U_PARTS, 0, copy(2))),
                    assign(
                        place(1).field(2, u16_ty()),
                        mir::Rvalue::Use(const_int(0x1234, 2)),
                    ),
                ];
                stmts.extend(tail);
                stmts
            },
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.fn_names.insert(U_PARTS, "probe::U".to_string());
    prog.adt_shapes.insert(
        U_PARTS,
        union_shape(vec![pair64_ty(), u64_ty(), u16_ty(), pair128_ty()]),
    );
    prog.add("main", main);
    prog
}

/// Reading field `b` after the switch: the bytes field `c`
/// covered hold its write, and the rest of field `a`'s image
/// survives.
/// `fn main() -> u64 { ...; u.b }` = 0x22222222_11111234.
#[test]
fn byte_form_held_field_tail_survives() {
    let prog = parts_program(
        vec![assign(
            place(0),
            mir::Rvalue::Use(mir::Operand::Copy(place(1).field(1, u64_ty()))),
        )],
        u64_ty(),
    );
    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Uint(0x2222_2222_1111_1234));
}

/// The composition reaches inside the held field's own value
/// too: field `a`'s first element now holds field `c`'s bytes
/// in its low half and its own in the high half.
/// `fn main() -> u32 { ...; u.a.0 }` = 0x11111234.
#[test]
fn byte_form_read_back_inside_held_field() {
    let prog = parts_program(
        vec![assign(
            place(0),
            mir::Rvalue::Use(mir::Operand::Copy(
                place(1).field(0, pair64_ty()).field(0, u32_ty()),
            )),
        )],
        u32_ty(),
    );
    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Uint(0x1111_1234));
}

/// A part no write ever covered stays uninitialised in the
/// byte form: field `d`'s second element spans bytes past
/// every write, so reading it raises the uninitialised-read
/// error at its own point of use.
/// `fn main() -> u64 { ...; u.d.1 }` fails with
/// `UninitializedRead`.
#[test]
fn byte_form_unwritten_part_still_uninitialized() {
    let prog = parts_program(
        vec![assign(
            place(0),
            mir::Rvalue::Use(mir::Operand::Copy(
                place(1).field(3, pair128_ty()).field(1, u64_ty()),
            )),
        )],
        u64_ty(),
    );
    let err = interpret_mir(&prog, "main", Vec::new()).expect_err("unwritten part must fail");
    assert!(
        matches!(err, MirRunError::Interp(InterpError::UninitializedRead { .. })),
        "expected UninitializedRead, got {err:?}"
    );
}
