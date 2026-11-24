//! Partially materialised aggregates and the transparent
//! wrapper descent: a struct stored field by field holds a
//! prefix of its fields, whose first field can itself be an
//! aggregate -- the same stored shape a one-field wrapper
//! around a complete payload has. Field writes and reads past
//! the stored prefix must judge the two apart by the declared
//! types: a sibling field write extends the struct's own
//! field vector (it does not descend into the first field),
//! and a never-stored zero-sized field reads as its canonical
//! value (it does not read a sibling's contents through the
//! descent).
use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::Value;
use sparagmos_mir as mir;

const S_DEF: mir::DefId = mir::DefId::new(0, 78);
const S2_DEF: mir::DefId = mir::DefId::new(0, 79);
const M_DEF: mir::DefId = mir::DefId::new(0, 80);

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}
fn tuple2_ty() -> mir::Ty {
    mir::Ty::Tuple(vec![i32_ty(), i32_ty()])
}
fn s_ty() -> mir::Ty {
    mir::Ty::Adt(S_DEF, vec![])
}
fn s2_ty() -> mir::Ty {
    mir::Ty::Adt(S2_DEF, vec![])
}
fn m_ty() -> mir::Ty {
    mir::Ty::Adt(M_DEF, vec![])
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}
fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}
fn copy(p: mir::Place) -> mir::Operand {
    mir::Operand::Copy(p)
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
fn assign(p: mir::Place, rv: mir::Rvalue) -> mir::Statement {
    mir::Statement::assign(p, rv)
}
fn tuple_agg(a: i32, b: i32) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Tuple),
        vec![const_i32(a), const_i32(b)],
    )
}

/// _1.0 = (10, 20); _1.1 = 30; return _1.0.1
fn f1_body() -> mir::Body {
    mir::Body::new(
        decls(&[i32_ty(), s_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1).field(0, tuple2_ty()), tuple_agg(10, 20)),
                assign(place(1).field(1, i32_ty()), mir::Rvalue::Use(const_i32(30))),
                assign(
                    place(0),
                    mir::Rvalue::Use(copy(place(1).field(0, tuple2_ty()).field(1, i32_ty()))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    )
}

fn s_shape() -> mir::AdtShape {
    mir::AdtShape {
        kind: mir::AdtShapeKind::Struct,
        variants: vec![vec![tuple2_ty(), i32_ty()]],
    }
}

#[test]
fn partial_struct_field_writes_keep_siblings() {
    let mut prog = MirProgram::new();
    prog.adt_discriminants.insert(S_DEF, vec![0]);
    prog.adt_shapes.insert(S_DEF, s_shape());
    prog.add("main", f1_body());
    let v = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(v, Value::Int(20), "first.1 must stay 20");
}

#[test]
fn partial_struct_field_writes_keep_siblings_unrecorded_shape() {
    // The program records no shape for the struct at all: with
    // a named ADT whose field table entry is missing there is
    // no evidence the stored form is a wrapper, so the write
    // still extends the struct's own field vector.
    let mut prog = MirProgram::new();
    prog.adt_discriminants.insert(S_DEF, vec![0]);
    prog.add("main", f1_body());
    let v = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(v, Value::Int(20), "first.1 must stay 20");
}

/// _1.0 = (10, 20); return (_1.1, _1.0.1) -- _1.1 is a
/// zero-sized field whose store is elided.
fn f3_body() -> mir::Body {
    mir::Body::new(
        decls(&[mir::Ty::Tuple(vec![m_ty(), i32_ty()]), s2_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1).field(0, tuple2_ty()), tuple_agg(10, 20)),
                assign(
                    place(0),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![
                            copy(place(1).field(1, m_ty())),
                            copy(place(1).field(0, tuple2_ty()).field(1, i32_ty())),
                        ],
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    )
}

#[test]
fn elided_zst_field_read_is_canonical() {
    let mut prog = MirProgram::new();
    prog.adt_discriminants.insert(S2_DEF, vec![0]);
    prog.adt_discriminants.insert(M_DEF, vec![0]);
    prog.adt_shapes.insert(
        S2_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![vec![tuple2_ty(), m_ty()]],
        },
    );
    prog.adt_shapes.insert(
        M_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![vec![]],
        },
    );
    prog.add("main", f3_body());
    let v = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(
        v,
        Value::Tuple(vec![
            Value::Enum {
                variant: 0,
                fields: vec![]
            },
            Value::Int(20)
        ]),
        "the elided field must read as its canonical value"
    );
}
