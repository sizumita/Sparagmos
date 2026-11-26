//! Regression test for Issue #480: the union-table walk in
//! `Converter::compute_union_tables` expanded ADT field types
//! without bound on a *generative* recursive type -- one whose
//! recursive occurrence carries larger arguments than the
//! instance it was reached from, like the
//! `Expanding<T> { b: *const Expanding<(T, T)> }` of
//! tests/debuginfo/recursive-struct.rs. Every expansion minted
//! a new, twice-as-large type, so the driver grew at
//! ~190 MB/s until it was OOM-killed. With the walk's
//! non-shrinking-re-entry guard the program below converts
//! and runs; before the guard this test never came back.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::Value;
use sparagmos_mir as mir;

const EXPANDING: mir::DefId = mir::DefId::new(0, 90);
const PHANTOM: mir::DefId = mir::DefId::new(0, 91);

fn param_ty() -> mir::Ty {
    mir::Ty::Param(0, "T".to_string())
}

fn expanding_ty(arg: mir::Ty) -> mir::Ty {
    mir::Ty::Adt(EXPANDING, vec![arg])
}

/// `fn main() -> i32 { let _e: Expanding<()>; 7 }`: the local's
/// type is never used by a statement -- conversion alone walks
/// it (the union-table computation runs over every local
/// declaration's type), which is exactly what ballooned.
#[test]
fn generative_recursive_type_does_not_balloon_union_walk() {
    let mut prog = MirProgram::new();
    prog.fn_names
        .insert(EXPANDING, "probe::Expanding".to_string());
    prog.fn_names
        .insert(PHANTOM, "core::marker::PhantomData".to_string());
    // `struct Expanding<T> { a: PhantomData<T>, b: *const Expanding<(T, T)> }`
    prog.adt_shapes.insert(
        EXPANDING,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![vec![
                mir::Ty::Adt(PHANTOM, vec![param_ty()]),
                mir::Ty::RawPtr(
                    Box::new(expanding_ty(mir::Ty::Tuple(vec![
                        param_ty(),
                        param_ty(),
                    ]))),
                    mir::Mutability::Not,
                ),
            ]],
        },
    );
    prog.adt_shapes.insert(
        PHANTOM,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![Vec::new()],
        },
    );
    let main = mir::Body::new(
        vec![
            mir::LocalDecl::new(mir::Ty::Int(mir::IntTy::I32)),
            mir::LocalDecl::new(expanding_ty(mir::Ty::Tuple(Vec::new()))),
        ],
        0,
        vec![{
            let mut bb = mir::BasicBlockData::new(mir::Terminator::new(
                mir::TerminatorKind::Return,
            ));
            bb.statements = vec![mir::Statement::assign(
                mir::Place::from_local(mir::Local(0)),
                mir::Rvalue::Use(mir::Operand::constant(
                    mir::Constant::from_value(mir::ConstValue::Scalar(
                        mir::Scalar::Int { data: 7, size: 4 },
                    )),
                )),
            )];
            bb
        }],
    );
    prog.add("main", main);
    let value = interpret_mir(&prog, "main", Vec::new()).expect("main runs");
    assert_eq!(value, Value::Int(7));
}
