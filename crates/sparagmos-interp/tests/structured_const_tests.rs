//! Structured constant tests (Lane R): tuple / ADT / reference /
//! slice constants materialised from by-reference allocations,
//! converted through `mir_convert` and executed.
//!
//! The bodies are shaped after the real shapes the driver produces
//! for promoted and static constants: `assert_eq!` builds a promoted
//! `&i32`, panic paths carry a `&Location`-shaped struct constant,
//! and name tables are `&[&str]` constants.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::Value;
use sparagmos_mir as mir;

// ----- shared builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}
fn u32_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U32)
}
fn usize_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::Usize)
}

fn ref_ty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Not)
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}

fn proj(idx: u32, projection: Vec<mir::PlaceElem>) -> mir::Place {
    mir::Place {
        local: mir::Local(idx),
        projection,
    }
}

fn copy(idx: u32) -> mir::Operand {
    mir::Operand::Copy(place(idx))
}

fn copy_proj(idx: u32, projection: Vec<mir::PlaceElem>) -> mir::Operand {
    mir::Operand::Copy(proj(idx, projection))
}

fn scalar(v: u128, size: u8) -> mir::ConstValue {
    mir::ConstValue::Scalar(mir::Scalar::Int { data: v, size })
}

fn const_op(v: mir::ConstValue) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(v))
}

fn block(stmts: Vec<mir::Statement>, kind: mir::TerminatorKind) -> mir::BasicBlockData {
    let mut bb = mir::BasicBlockData::new(mir::Terminator::new(kind));
    bb.statements = stmts;
    bb
}

fn assign(place: mir::Place, rv: mir::Rvalue) -> mir::Statement {
    mir::Statement::assign(place, rv)
}

fn use_op(op: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::Use(op)
}

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

fn run(body: mir::Body) -> Value {
    let mut prog = MirProgram::new();
    prog.add("main", body);
    interpret_mir(&prog, "main", vec![]).expect("run should succeed")
}

// ----- a promoted `&i32` (the `assert_eq!` shape) ----------------------------

#[test]
fn promoted_ref_to_scalar_derefs() {
    // fn main() -> i32 { _1 = const &10_i32; _2 = copy (*_1); _0 = copy _2; return }
    let body = mir::Body::new(
        decls(&[i32_ty(), ref_ty(i32_ty()), i32_ty()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    use_op(const_op(mir::ConstValue::Ref(Box::new(scalar(10, 4))))),
                ),
                assign(
                    place(2),
                    use_op(copy_proj(1, vec![mir::PlaceElem::Deref])),
                ),
                assign(place(0), use_op(copy(2))),
            ],
            ret(),
        )],
    );
    assert_eq!(run(body), Value::Int(10));
}

// ----- a `&Struct` constant (the panic `&Location` shape) --------------------

fn loc_ty() -> mir::Ty {
    mir::Ty::Adt(mir::DefId::new(0, 900), vec![])
}

fn loc_const() -> mir::ConstValue {
    mir::ConstValue::Ref(Box::new(mir::ConstValue::Adt {
        variant: 0,
        fields: vec![
            mir::ConstValue::Str("src/main.rs".to_string()),
            scalar(12, 4),
            scalar(34, 4),
        ],
    }))
}

#[test]
fn ref_to_struct_constant_fields_read_through_deref() {
    // fn main() -> u32 { _1 = const &LOC; _2 = copy ((*_1).1); _0 = copy _2; return }
    let body = mir::Body::new(
        decls(&[u32_ty(), ref_ty(loc_ty()), u32_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), use_op(const_op(loc_const()))),
                assign(
                    place(2),
                    use_op(copy_proj(
                        1,
                        vec![mir::PlaceElem::Deref, mir::PlaceElem::Field(1, u32_ty())],
                    )),
                ),
                assign(place(0), use_op(copy(2))),
            ],
            ret(),
        )],
    );
    assert_eq!(run(body), Value::Uint(12));
}

#[test]
fn ref_to_struct_constant_str_field_len() {
    // fn main() -> usize { _1 = const &LOC; _2 = copy ((*_1).0); _0 = Len(_2); return }
    let body = mir::Body::new(
        decls(&[usize_ty(), ref_ty(loc_ty()), ref_ty(mir::Ty::Str)]),
        0,
        vec![block(
            vec![
                assign(place(1), use_op(const_op(loc_const()))),
                assign(
                    place(2),
                    use_op(copy_proj(
                        1,
                        vec![
                            mir::PlaceElem::Deref,
                            mir::PlaceElem::Field(0, ref_ty(mir::Ty::Str)),
                        ],
                    )),
                ),
                assign(place(0), mir::Rvalue::Len(place(2))),
            ],
            ret(),
        )],
    );
    assert_eq!(run(body), Value::Uint("src/main.rs".len() as u128));
}

// ----- a `&[&str]` constant ---------------------------------------------------

fn str_slice_ty() -> mir::Ty {
    ref_ty(mir::Ty::Slice(Box::new(ref_ty(mir::Ty::Str))))
}

#[test]
fn slice_of_str_constants_len_and_index() {
    // fn main() -> usize {
    //   _1 = const &["a", "bb", "ccc"]; _0 = Len((*_1)); return
    // }
    let names = mir::ConstValue::SliceVal(vec![
        mir::ConstValue::Str("a".to_string()),
        mir::ConstValue::Str("bb".to_string()),
        mir::ConstValue::Str("ccc".to_string()),
    ]);
    let body = mir::Body::new(
        decls(&[usize_ty(), str_slice_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), use_op(const_op(names))),
                assign(
                    place(0),
                    mir::Rvalue::Len(proj(1, vec![mir::PlaceElem::Deref])),
                ),
            ],
            ret(),
        )],
    );
    assert_eq!(run(body), Value::Uint(3));
}

#[test]
fn slice_of_str_constants_index_reads_element() {
    // fn main() -> usize {
    //   _1 = const &["a", "bb", "ccc"]; _2 = copy ((*_1)[2]); _0 = Len(_2); return
    // }
    let names = mir::ConstValue::SliceVal(vec![
        mir::ConstValue::Str("a".to_string()),
        mir::ConstValue::Str("bb".to_string()),
        mir::ConstValue::Str("ccc".to_string()),
    ]);
    let body = mir::Body::new(
        decls(&[usize_ty(), str_slice_ty(), ref_ty(mir::Ty::Str)]),
        0,
        vec![block(
            vec![
                assign(place(1), use_op(const_op(names))),
                assign(
                    place(2),
                    use_op(copy_proj(
                        1,
                        vec![
                            mir::PlaceElem::Deref,
                            mir::PlaceElem::ConstantIndex {
                                offset: 2,
                                min_length: 3,
                                from_end: false,
                            },
                        ],
                    )),
                ),
                assign(place(0), mir::Rvalue::Len(place(2))),
            ],
            ret(),
        )],
    );
    assert_eq!(run(body), Value::Uint(3));
}

// ----- by-value structured constants ------------------------------------------

#[test]
fn tuple_constant_field_read() {
    // fn main() -> i32 { _1 = const (7, 9); _2 = copy (_1.1); _0 = copy _2; return }
    let pair_ty = mir::Ty::Tuple(vec![i32_ty(), i32_ty()]);
    let body = mir::Body::new(
        decls(&[i32_ty(), pair_ty, i32_ty()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    use_op(const_op(mir::ConstValue::Tuple(vec![
                        scalar(7, 4),
                        scalar(9, 4),
                    ]))),
                ),
                assign(
                    place(2),
                    use_op(copy_proj(1, vec![mir::PlaceElem::Field(1, i32_ty())])),
                ),
                assign(place(0), use_op(copy(2))),
            ],
            ret(),
        )],
    );
    assert_eq!(run(body), Value::Int(9));
}

#[test]
fn adt_constant_variant_field_read() {
    // fn main() -> i32 {
    //   _1 = const Some(42); _2 = copy (((_1 as Some).0)); _0 = copy _2; return
    // }
    let opt_ty = mir::Ty::Adt(mir::DefId::new(0, 901), vec![]);
    let body = mir::Body::new(
        decls(&[i32_ty(), opt_ty, i32_ty()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    use_op(const_op(mir::ConstValue::Adt {
                        variant: 1,
                        fields: vec![scalar(42, 4)],
                    })),
                ),
                assign(
                    place(2),
                    use_op(copy_proj(
                        1,
                        vec![
                            mir::PlaceElem::Downcast(None, 1),
                            mir::PlaceElem::Field(0, i32_ty()),
                        ],
                    )),
                ),
                assign(place(0), use_op(copy(2))),
            ],
            ret(),
        )],
    );
    assert_eq!(run(body), Value::Int(42));
}

#[test]
fn nested_ref_constant_derefs_twice() {
    // fn main() -> i32 { _1 = const &&5_i32; _2 = copy (**_1); _0 = copy _2; return }
    let body = mir::Body::new(
        decls(&[i32_ty(), ref_ty(ref_ty(i32_ty())), i32_ty()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    use_op(const_op(mir::ConstValue::Ref(Box::new(
                        mir::ConstValue::Ref(Box::new(scalar(5, 4))),
                    )))),
                ),
                assign(
                    place(2),
                    use_op(copy_proj(
                        1,
                        vec![mir::PlaceElem::Deref, mir::PlaceElem::Deref],
                    )),
                ),
                assign(place(0), use_op(copy(2))),
            ],
            ret(),
        )],
    );
    assert_eq!(run(body), Value::Int(5));
}
