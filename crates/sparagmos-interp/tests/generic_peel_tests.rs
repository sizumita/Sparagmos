//! Regression tests for the `MaybeUninit` entry judgment when
//! the pointee is a generic ADT (Codex review of PR #355).
//!
//! The interpreter's converted pointee type names the ADT
//! definition alone (`convert_ty` erases the generic arguments),
//! while the recorded debug shapes are per monomorphic instance.
//! When a program contains both `G<u32>` and `G<E>` for
//! `struct G<T>(T)` with `struct E(u32)`, judging a stored
//! `G<E>` value against an arbitrarily picked instance's shape
//! can misread the genuine value as a `MaybeUninit` wrapper:
//! under `G<u32>`'s shape the stored field (an `E`) fails the
//! field-fit refusal, yet the payload check passes against that
//! same shape, so the dereference peels the struct and returns
//! its inner `E`. The judgment must consider every recorded
//! instance before treating a field mismatch as wrapper
//! evidence.
//!
//! The programs are hand-built `sparagmos-mir` bodies in the
//! `unsafecell_tests` fixture style. The dereference runs
//! through a raw pointer to an array element, a position whose
//! home type the entry gate does not claim, so the stored-form
//! judgment itself decides. Because the pre-fix pick depends on
//! the shape table's iteration order, the test rebuilds the
//! program many times: the fixed judgment is order-independent,
//! so every rebuild must read the genuine value.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::Value;
use sparagmos_mir as mir;

// ----- shared builders over sparagmos_mir -----------------------------------

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}

fn mov(idx: u32) -> mir::Operand {
    mir::Operand::Move(place(idx))
}

fn const_u32(v: u32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
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

fn adt_aggregate(
    def: mir::DefId,
    generic_args: Vec<mir::Ty>,
    fields: Vec<mir::Operand>,
) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def,
            variant_index: 0,
            generic_args,
            user_ty: None,
            field_index: None,
        }),
        fields,
    )
}

// ----- the generic fixture ----------------------------------------------------

const E_DEF: mir::DefId = mir::DefId::new(0, 400);
const G_DEF: mir::DefId = mir::DefId::new(0, 401);

fn u32_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U32)
}

fn e_ty() -> mir::Ty {
    mir::Ty::Adt(E_DEF, Vec::new())
}

fn g_ty(arg: mir::Ty) -> mir::Ty {
    mir::Ty::Adt(G_DEF, vec![arg])
}

fn field(ty: &str) -> mir::AdtDebugField {
    mir::AdtDebugField {
        name: None,
        ty: ty.to_string(),
    }
}

fn struct_shape(variant_name: &str, field_ty: &str) -> mir::AdtDebugShape {
    mir::AdtDebugShape {
        kind: mir::AdtShapeKind::Struct,
        variants: vec![mir::AdtDebugVariant {
            name: variant_name.to_string(),
            fields: vec![field(field_ty)],
        }],
    }
}

/// A main that stores a genuine `G<E>(E(42))` in a one-element
/// array, takes a raw pointer to the element, and returns the
/// value read back through `(*p).0` -- the whole `E`. A
/// wrongful peel makes that read land on the bare `u32` inside
/// the `E` instead, which the assignment's type check rejects.
/// Locals: _0 ret `E`, _1 `[G<E>; 1]`, _2 `*const G<E>`,
/// _3 `G<E>`, _4 `E`.
fn main_body() -> mir::Body {
    let ge = g_ty(e_ty());
    mir::Body::new(
        decls(&[
            e_ty(),
            mir::Ty::Array(Box::new(ge.clone()), mir::ArrayLen::Known(1)),
            mir::Ty::RawPtr(Box::new(ge.clone()), mir::Mutability::Not),
            ge,
            e_ty(),
        ]),
        0,
        vec![block(
            vec![
                assign(
                    place(4),
                    adt_aggregate(E_DEF, Vec::new(), vec![const_u32(42)]),
                ),
                assign(place(3), adt_aggregate(G_DEF, vec![e_ty()], vec![mov(4)])),
                assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(g_ty(e_ty()))),
                        vec![mov(3)],
                    ),
                ),
                assign(
                    place(2),
                    mir::Rvalue::RawPtr(
                        mir::Mutability::Not,
                        mir::Place {
                            local: mir::Local(1),
                            projection: vec![mir::PlaceElem::ConstantIndex {
                                offset: 0,
                                min_length: 1,
                                from_end: false,
                            }],
                        },
                    ),
                ),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(mir::Place {
                        local: mir::Local(2),
                        projection: vec![
                            mir::PlaceElem::Deref,
                            mir::PlaceElem::Field(0, e_ty()),
                        ],
                    })),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    )
}

/// The program with the debug shapes of `E` and of the given
/// `G` instances recorded under their rendered instance names.
fn program(g_instances: &[(&str, &str)]) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names.insert(E_DEF, "test::E".to_string());
    prog.fn_names.insert(G_DEF, "test::G".to_string());
    prog.adt_debug_shapes
        .insert("test::E".to_string(), struct_shape("E", "u32"));
    for (key, field_ty) in g_instances {
        prog.adt_debug_shapes
            .insert(key.to_string(), struct_shape("G", field_ty));
    }
    prog.add("main", main_body());
    prog
}

fn expected_e() -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Uint(42)],
    }
}

// ----- the tests ---------------------------------------------------------------

#[test]
fn genuine_generic_value_is_not_peeled_with_both_instances_recorded() {
    // Both `G<u32>` and `G<E>` recorded: whichever instance an
    // arbitrary pick lands on, the stored `G<E>` value is a
    // genuine value of one of them and must not be peeled.
    // Rebuilt many times because the pre-fix judgment depended
    // on the shape table's iteration order.
    for _ in 0..100 {
        let prog = program(&[("test::G<u32>", "u32"), ("test::G<test::E>", "test::E")]);
        let v = interpret_mir(&prog, "main", vec![]).expect("program must run");
        assert_eq!(v, expected_e());
    }
}

#[test]
fn genuine_generic_value_is_not_peeled_with_matching_instance_only() {
    let prog = program(&[("test::G<test::E>", "test::E")]);
    let v = interpret_mir(&prog, "main", vec![]).expect("program must run");
    assert_eq!(v, expected_e());
}
