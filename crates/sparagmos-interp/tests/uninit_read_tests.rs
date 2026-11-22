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

// ----- group B: zero-sized values are synthesized ----------------------------

const ZST_STRUCT: mir::DefId = mir::DefId::new(0, 50);
const SINGLE_ENUM: mir::DefId = mir::DefId::new(0, 51);
const EMPTY_ENUM: mir::DefId = mir::DefId::new(0, 52);
const OPT_EMPTY: mir::DefId = mir::DefId::new(0, 53);
const BLACK_BOX: mir::DefId = mir::DefId::new(0, 54);

fn adt_ty(def: mir::DefId) -> mir::Ty {
    mir::Ty::Adt(def, Vec::new())
}

fn shape(kind: mir::AdtShapeKind, variants: Vec<Vec<mir::Ty>>) -> mir::AdtShape {
    mir::AdtShape { kind, variants }
}

fn zst_program(main: mir::Body, shapes: &[(mir::DefId, mir::AdtShape)]) -> MirProgram {
    let mut prog = MirProgram::new();
    for (def, s) in shapes {
        prog.adt_shapes.insert(*def, s.clone());
    }
    prog.add("main", main);
    prog
}

/// `struct Z; fn main() -> Z { return }` — a unit struct
/// constructor's body is a bare `return`: the return place is
/// never written, and its single value is synthesized from the
/// shape table.
#[test]
fn zst_struct_return_synthesizes_value() {
    let shapes = [(
        ZST_STRUCT,
        shape(mir::AdtShapeKind::Struct, vec![vec![]]),
    )];
    let main = mir::Body::new(
        decls(&[adt_ty(ZST_STRUCT)]),
        0,
        vec![block(vec![], ret())],
    );
    let value = interpret_mir(&zst_program(main, &shapes), "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Enum { variant: 0, fields: vec![] });
}

/// `enum E { A(()) } fn main() -> E { let e: E; e }` — an enum
/// with a single inhabited variant whose fields are all
/// zero-sized has exactly one value; a never-written local of
/// that type reads as it.
#[test]
fn single_variant_enum_unwritten_read_synthesizes() {
    let shapes = [(
        SINGLE_ENUM,
        shape(
            mir::AdtShapeKind::Enum,
            vec![vec![mir::Ty::Tuple(Vec::new())]],
        ),
    )];
    let main = mir::Body::new(
        decls(&[adt_ty(SINGLE_ENUM), adt_ty(SINGLE_ENUM)]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(copy(1)))],
            ret(),
        )],
    );
    let value = interpret_mir(&zst_program(main, &shapes), "main", Vec::new()).expect("main runs");
    assert_eq!(
        value,
        Value::Enum { variant: 0, fields: vec![Value::Unit] }
    );
}

/// `enum Empty {} fn main() -> Option<Empty> { let o; o }` —
/// the `Some` variant's field type is uninhabited, so it can
/// never be constructed and the enum's single value is `None`
/// (variant 1), even though two variants are declared.
#[test]
fn option_of_uninhabited_synthesizes_none() {
    let shapes = [
        (EMPTY_ENUM, shape(mir::AdtShapeKind::Enum, vec![])),
        (
            OPT_EMPTY,
            shape(
                mir::AdtShapeKind::Enum,
                vec![vec![adt_ty(EMPTY_ENUM)], vec![]],
            ),
        ),
    ];
    let main = mir::Body::new(
        decls(&[adt_ty(OPT_EMPTY), adt_ty(OPT_EMPTY)]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(copy(1)))],
            ret(),
        )],
    );
    let value = interpret_mir(&zst_program(main, &shapes), "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Enum { variant: 1, fields: vec![] });
}

/// A zero-sized constant of an ADT type — the typed
/// `ConstValue::Adt` the driver now emits for a unit struct
/// constant instead of an untyped unit — keeps its ADT value
/// through `black_box` (the identity builtin), instead of
/// collapsing to `Value::Unit`.
#[test]
fn zst_const_through_black_box_keeps_adt_value() {
    let shapes = [(
        ZST_STRUCT,
        shape(mir::AdtShapeKind::Struct, vec![vec![]]),
    )];
    let mut prog = zst_program(
        mir::Body::new(
            decls(&[adt_ty(ZST_STRUCT)]),
            0,
            vec![
                block(
                    vec![],
                    mir::TerminatorKind::Call {
                        func: mir::Operand::constant(mir::Constant {
                            literal: mir::ConstKind::Ty(mir::Ty::FnDef(BLACK_BOX, Vec::new())),
                            user_ty: None,
                            ty: None,
                        }),
                        args: vec![mir::Operand::constant(mir::Constant::from_value(
                            mir::ConstValue::Adt { variant: 0, fields: Vec::new() },
                        ))],
                        destination: place(0),
                        target: Some(mir::BasicBlock(1)),
                        unwind: mir::UnwindAction::Continue,
                        call_source: mir::CallSource::Normal,
                    },
                ),
                block(vec![], ret()),
            ],
        ),
        &shapes,
    );
    prog.fn_names
        .insert(BLACK_BOX, "std::intrinsics::black_box".to_string());
    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Enum { variant: 0, fields: vec![] });
}

// ----- group C: Repeat evaluates its operand once ----------------------------

/// `fn main() -> [[u32; 2]; 3] { let a = [7u32; 2]; [a; 3] }` —
/// MIR writes the outer repeat's operand as a move of `_1`;
/// a repeat evaluates its operand once and replicates the
/// value, so the conversion reads it as a copy. Evaluating the
/// move per element would fail on the second read (the source
/// having been moved out of).
#[test]
fn repeat_of_moved_array_replicates() {
    let inner = mir::Ty::Array(Box::new(u32_ty()), mir::ArrayLen::Known(2));
    let outer = mir::Ty::Array(Box::new(inner.clone()), mir::ArrayLen::Known(3));
    let main = mir::Body::new(
        decls(&[outer.clone(), inner.clone(), outer]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Repeat(const_u32(7), 2)),
                assign(
                    place(2),
                    mir::Rvalue::Repeat(mir::Operand::Move(place(1)), 3),
                ),
                assign(place(0), mir::Rvalue::Use(copy(2))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", main);
    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    let row = Value::Tuple(vec![Value::Uint(7), Value::Uint(7)]);
    assert_eq!(value, Value::Tuple(vec![row.clone(), row.clone(), row]));
}

// ----- typed ZST constants under drop glue -----------------------------------
//
// A field written only by a constant operand has no recorded type
// anywhere (the aggregate scan records field types for place
// operands only), so its drop plan is `Untyped` and the glue
// judges the value by shape. A fieldless enum value — which the
// typed zero-sized constants of group B produce where an untyped
// unit used to sit — has no contents and must pass; an enum value
// with a payload must stay an explicit error.

const FIELDLESS: mir::DefId = mir::DefId::new(0, 55);
const RAW_T: mir::DefId = mir::DefId::new(0, 56);
const PAYLOAD_ENUM: mir::DefId = mir::DefId::new(0, 57);
const RAW_P: mir::DefId = mir::DefId::new(0, 58);

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

fn adt_const(variant: u32, fields: Vec<mir::ConstValue>) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Adt {
        variant,
        fields,
    }))
}

fn drop_term(idx: u32, next: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Drop {
        place: place(idx),
        target: mir::BasicBlock(next),
        unwind: mir::UnwindAction::Continue,
        replace: false,
    }
}

/// `enum TyS { Nil } struct RawT { struct_: TyS, hash: usize }
/// fn main() -> i32 { let t = RawT { struct_: TyS::Nil, hash: 0 };
/// drop(t); 7 }` — the fieldless field's value is the typed
/// constant `Adt{0, []}`; its untyped drop plan must not reject
/// it (the shape this regressed on in the rustc suite:
/// codegen/alias-uninit-value).
#[test]
fn drop_of_fieldless_enum_field_from_const() {
    let main = mir::Body::new(
        decls(&[i32_ty(), adt_ty(RAW_T)]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    adt_aggregate(RAW_T, vec![adt_const(0, vec![]), const_int(0, 8)]),
                )],
                drop_term(1, 1),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(const_i32(7)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.fn_names.insert(FIELDLESS, "probe::TyS".to_string());
    prog.fn_names.insert(RAW_T, "probe::RawT".to_string());
    prog.adt_shapes.insert(
        FIELDLESS,
        shape(mir::AdtShapeKind::Enum, vec![vec![]]),
    );
    prog.adt_shapes.insert(
        RAW_T,
        shape(
            mir::AdtShapeKind::Struct,
            vec![vec![adt_ty(FIELDLESS), mir::Ty::Uint(mir::UintTy::Usize)]],
        ),
    );
    prog.add("probe::main", main);
    let value = interpret_mir(&prog, "probe::main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Int(7));
}

/// The conservative half of the rule stays: an enum value WITH a
/// payload under an untyped plan might hide a destructor in its
/// contents, so the drop is still an explicit error.
#[test]
fn drop_of_payload_enum_field_from_const_still_errors() {
    let main = mir::Body::new(
        decls(&[i32_ty(), adt_ty(RAW_P)]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    adt_aggregate(
                        RAW_P,
                        vec![adt_const(
                            0,
                            vec![mir::ConstValue::Str("x".to_string())],
                        )],
                    ),
                )],
                drop_term(1, 1),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(const_i32(7)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.fn_names
        .insert(PAYLOAD_ENUM, "probe::E".to_string());
    prog.fn_names.insert(RAW_P, "probe::RawP".to_string());
    prog.adt_shapes.insert(
        PAYLOAD_ENUM,
        shape(
            mir::AdtShapeKind::Enum,
            vec![vec![mir::Ty::Str]],
        ),
    );
    prog.adt_shapes.insert(
        RAW_P,
        shape(
            mir::AdtShapeKind::Struct,
            vec![vec![adt_ty(PAYLOAD_ENUM)]],
        ),
    );
    prog.add("probe::main", main);
    let err = interpret_mir(&prog, "probe::main", Vec::new()).expect_err("payload drop must fail");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedTerminator { ref kind }) if kind == "Drop"
        ),
        "expected unsupported Drop, got {err:?}"
    );
}

// ----- group D: partial initialization and single-valued fields (Lane GD) ---

const PAIR: mir::DefId = mir::DefId::new(0, 60);
const HOST: mir::DefId = mir::DefId::new(0, 61);
const ZWRAP: mir::DefId = mir::DefId::new(0, 62);
const ZLEAF: mir::DefId = mir::DefId::new(0, 63);

fn const_uninit() -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Uninit))
}

fn gd_program(main: mir::Body, shapes: &[(mir::DefId, mir::AdtShape)]) -> MirProgram {
    let mut prog = MirProgram::new();
    for (def, s) in shapes {
        prog.adt_shapes.insert(*def, s.clone());
    }
    prog.add("main", main);
    prog
}

fn pair_program(main: mir::Body) -> MirProgram {
    gd_program(
        main,
        &[(
            PAIR,
            shape(
                mir::AdtShapeKind::Struct,
                vec![vec![u32_ty(), u32_ty()]],
            ),
        )],
    )
}

/// `struct Pair { a: u32, b: u32 }`
/// `fn main() -> u32 { let p: Pair = <uninit>; p.b = 7; p.b }` —
/// the shape of std's `catch_unwind` Data storage at opt-level 3:
/// the local is materialized by an assignment of the uninitialized
/// constant (which is not a type mismatch: the place simply holds
/// no value yet) and then filled field by field. The field write
/// into the never-written slot materializes the aggregate; the
/// written field reads back.
#[test]
fn uninit_local_field_write_then_read() {
    let main = mir::Body::new(
        decls(&[u32_ty(), adt_ty(PAIR)]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_uninit())),
                assign(
                    place(1).field(1, u32_ty()),
                    mir::Rvalue::Use(const_u32(7)),
                ),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).field(1, u32_ty()))),
                ),
            ],
            ret(),
        )],
    );
    let value = interpret_mir(&pair_program(main), "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Uint(7));
}

/// The sibling field the partial initialization never wrote stays
/// uninitialized: materializing the aggregate fabricates nothing,
/// so reading field 0 of the same local still fails.
#[test]
fn uninit_local_unwritten_field_still_errors() {
    let main = mir::Body::new(
        decls(&[u32_ty(), adt_ty(PAIR)]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_uninit())),
                assign(
                    place(1).field(1, u32_ty()),
                    mir::Rvalue::Use(const_u32(7)),
                ),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).field(0, u32_ty()))),
                ),
            ],
            ret(),
        )],
    );
    let err = interpret_mir(&pair_program(main), "main", Vec::new())
        .expect_err("unwritten field must fail");
    assert!(
        matches!(err, MirRunError::Interp(InterpError::UninitializedRead { .. })),
        "expected UninitializedRead, got {err:?}"
    );
}

/// `struct ZLeaf {} struct ZWrap { z: ZLeaf } struct Host { w: ZWrap, x: u32 }`
/// `fn main() -> ZLeaf { let h: Host; h.w.z }` — a field whose type
/// has a single value denotes that value even when no body ever
/// wrote it: MIR elides the store of a zero-sized field (the
/// closure slot of a `catch_unwind` Data union is read back by
/// `do_call` exactly this way), so the read is not an
/// uninitialized read. The local as a whole is not single-valued
/// (its `x` field varies), which keeps this on the field-path
/// table rather than the whole-local synthesis.
#[test]
fn single_valued_field_path_read_from_unwritten_local() {
    let shapes = [
        (
            HOST,
            shape(
                mir::AdtShapeKind::Struct,
                vec![vec![adt_ty(ZWRAP), u32_ty()]],
            ),
        ),
        (
            ZWRAP,
            shape(mir::AdtShapeKind::Struct, vec![vec![adt_ty(ZLEAF)]]),
        ),
        (ZLEAF, shape(mir::AdtShapeKind::Struct, vec![vec![]])),
    ];
    let main = mir::Body::new(
        decls(&[adt_ty(ZLEAF), adt_ty(HOST)]),
        0,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Use(mir::Operand::Copy(
                    place(1).field(0, adt_ty(ZWRAP)).field(0, adt_ty(ZLEAF)),
                )),
            )],
            ret(),
        )],
    );
    let value = interpret_mir(&gd_program(main, &shapes), "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Enum { variant: 0, fields: vec![] });
}

/// `struct ZHost { z: ZLeaf, x: u32 }`
/// `fn main() -> ZLeaf { let h: ZHost = <uninit>; h.x = 7; h.z }` —
/// the same single-valued field as
/// `single_valued_field_path_read_from_unwritten_local`, but the
/// base local is initialized (materialized by the field write)
/// with only the zero-sized field slot unwritten. A statement
/// level read of the field runs through the decoded fast path
/// (`read_local_proj`), which must rescue the read with the
/// recorded single value exactly as the general path does,
/// instead of reporting an uninitialized read.
#[test]
fn single_valued_field_read_from_materialized_local() {
    const ZHOST: mir::DefId = mir::DefId::new(0, 64);
    let shapes = [
        (
            ZHOST,
            shape(
                mir::AdtShapeKind::Struct,
                vec![vec![adt_ty(ZLEAF), u32_ty()]],
            ),
        ),
        (ZLEAF, shape(mir::AdtShapeKind::Struct, vec![vec![]])),
    ];
    let main = mir::Body::new(
        decls(&[adt_ty(ZLEAF), adt_ty(ZHOST)]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_uninit())),
                assign(
                    place(1).field(1, u32_ty()),
                    mir::Rvalue::Use(const_u32(7)),
                ),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).field(0, adt_ty(ZLEAF)))),
                ),
            ],
            ret(),
        )],
    );
    let value = interpret_mir(&gd_program(main, &shapes), "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Enum { variant: 0, fields: vec![] });
}
