//! Element-typed `copy_nonoverlapping` tests (Lane ED): the copy's
//! element is itself an aggregate -- an array, as in
//! `core::simd::Simd::load`'s internal
//! `copy_nonoverlapping::<[T; N]>` into a `MaybeUninit` slot. The
//! element boundary cannot be read off the values' shapes alone
//! there (the source IS one element, an array value; the
//! destination is a wrapper aggregate holding an uninit marker),
//! so the copy takes the element type from the pointer operands'
//! declared types, on both the intrinsic statement path (core's
//! own bodies) and the `std::ptr::copy_nonoverlapping` builtin
//! call path (user code).
//!
//! The programs are hand-built `sparagmos-mir` bodies shaped
//! after the driver's output for the corresponding Rust sources,
//! following the `mir_convert_tests` / `unsafecell_tests` fixture
//! style.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::Value;
use sparagmos_mir as mir;

// ----- shared builders over sparagmos_mir -----------------------------------

fn f32_ty() -> mir::Ty {
    mir::Ty::Float(mir::FloatTy::F32)
}

fn array4_ty() -> mir::Ty {
    mir::Ty::Array(Box::new(f32_ty()), mir::ArrayLen::Known(4))
}

fn array2_ty() -> mir::Ty {
    mir::Ty::Array(Box::new(f32_ty()), mir::ArrayLen::Known(2))
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

fn const_f32(v: f32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Float {
            bits: v.to_bits() as u128,
            ty: mir::FloatTy::F32,
        },
    )))
}

fn const_usize(v: usize) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int {
            data: v as u128,
            size: 8,
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

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

fn raw_ty(inner: mir::Ty, mutable: bool) -> mir::Ty {
    mir::Ty::RawPtr(
        Box::new(inner),
        if mutable {
            mir::Mutability::Mut
        } else {
            mir::Mutability::Not
        },
    )
}

fn add(lhs: mir::Operand, rhs: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((lhs, rhs)))
}

fn intrinsic_stmt(intr: mir::NonDivergingIntrinsic) -> mir::Statement {
    mir::Statement::new(mir::StatementKind::Intrinsic(Box::new(intr)))
}

/// A place indexing `local`'s field 0 (a wrapper payload) at
/// constant index `offset`.
fn payload_elem(local: u32, payload_ty: mir::Ty, offset: u64, min_length: u64) -> mir::Place {
    mir::Place {
        local: mir::Local(local),
        projection: vec![
            mir::PlaceElem::Field(0, payload_ty),
            mir::PlaceElem::ConstantIndex {
                offset,
                min_length,
                from_end: false,
            },
        ],
    }
}

/// A place indexing `local` at two constant indices.
fn idx2(local: u32, outer: u64, inner: u64) -> mir::Place {
    mir::Place {
        local: mir::Local(local),
        projection: vec![
            mir::PlaceElem::ConstantIndex {
                offset: outer,
                min_length: 2,
                from_end: false,
            },
            mir::PlaceElem::ConstantIndex {
                offset: inner,
                min_length: 2,
                from_end: false,
            },
        ],
    }
}

// ----- the MaybeUninit fixture ------------------------------------------------

const MU_DEF: mir::DefId = mir::DefId::new(0, 400);
const COPY_DEF: mir::DefId = mir::DefId::new(0, 401);

fn mu_ty() -> mir::Ty {
    mir::Ty::Adt(MU_DEF, vec![array4_ty()])
}

/// `MaybeUninit::<[f32; 4]>::uninit()`: the one-field aggregate
/// over the zero-sized constant, as rustc emits it.
fn mu_uninit() -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def: MU_DEF,
            variant_index: 0,
            generic_args: vec![array4_ty()],
            user_ty: None,
            field_index: None,
        }),
        vec![mir::Operand::constant(mir::Constant::from_value(
            mir::ConstValue::ZeroSized,
        ))],
    )
}

fn array4_aggregate(a: f32, b: f32, c: f32, d: f32) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Array(f32_ty())),
        vec![const_f32(a), const_f32(b), const_f32(c), const_f32(d)],
    )
}

/// The statements before the copy, shared by the statement-form
/// and call-form tests: build the source array, the uninit slot,
/// and the two pointers. Locals: _1 array, _2 slot, _3 src ptr,
/// _4 dst ptr.
fn load_like_setup() -> Vec<mir::Statement> {
    vec![
        assign(place(1), array4_aggregate(1.0, 2.0, 3.0, 4.0)),
        assign(place(2), mu_uninit()),
        assign(
            place(3),
            mir::Rvalue::RawPtr(mir::Mutability::Not, place(1)),
        ),
        assign(
            place(4),
            mir::Rvalue::RawPtr(mir::Mutability::Mut, place(2)),
        ),
    ]
}

/// Read the slot's payload back and sum it into _0 (via _5).
fn load_like_readback() -> Vec<mir::Statement> {
    vec![
        assign(
            place(5),
            add(
                mir::Operand::Copy(payload_elem(2, array4_ty(), 0, 4)),
                mir::Operand::Copy(payload_elem(2, array4_ty(), 1, 4)),
            ),
        ),
        assign(
            place(0),
            add(
                copy(5),
                mir::Operand::Copy(payload_elem(2, array4_ty(), 2, 4)),
            ),
        ),
        assign(
            place(0),
            add(
                copy(0),
                mir::Operand::Copy(payload_elem(2, array4_ty(), 3, 4)),
            ),
        ),
    ]
}

fn load_like_decls() -> Vec<mir::LocalDecl> {
    decls(&[
        f32_ty(),
        array4_ty(),
        mu_ty(),
        raw_ty(array4_ty(), false),
        raw_ty(array4_ty(), true),
        f32_ty(),
        mir::Ty::Tuple(Vec::new()),
    ])
}

// ----- the tests ---------------------------------------------------------------

/// The `Simd::load` shape, intrinsic statement form: one element
/// that is an array, copied into a `MaybeUninit` place. Before the
/// element type was threaded through, the copy read element 0 of
/// the array (a scalar) and the write into the wrapper place was
/// a type mismatch.
#[test]
fn intrinsic_copy_with_array_element_fills_maybeuninit_place() {
    let mut stmts = load_like_setup();
    stmts.push(intrinsic_stmt(
        mir::NonDivergingIntrinsic::CopyNonOverlapping {
            src: copy(3),
            dst: copy(4),
            count: const_usize(1),
        },
    ));
    stmts.extend(load_like_readback());
    let body = mir::Body::new(load_like_decls(), 0, vec![block(stmts, ret())]);
    let mut prog = MirProgram::new();
    prog.fn_names.insert(MU_DEF, "std::mem::MaybeUninit".to_string());
    prog.add("load_like", body);
    assert_eq!(
        interpret_mir(&prog, "load_like", vec![]).expect("load_like runs"),
        Value::Float(10.0)
    );
}

/// The same shape arriving as an ordinary call to
/// `std::ptr::copy_nonoverlapping` -- the form user code
/// produces, dispatched to the interpreter builtin.
#[test]
fn builtin_copy_with_array_element_fills_maybeuninit_place() {
    let call = mir::TerminatorKind::Call {
        func: mir::Operand::constant(mir::Constant {
            literal: mir::ConstKind::Ty(mir::Ty::FnDef(COPY_DEF, vec![array4_ty()])),
            user_ty: None,
            ty: None,
        }),
        args: vec![copy(3), copy(4), const_usize(1)],
        destination: place(6),
        target: Some(mir::BasicBlock(1)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    };
    let body = mir::Body::new(
        load_like_decls(),
        0,
        vec![
            block(load_like_setup(), call),
            block(load_like_readback(), ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.fn_names.insert(MU_DEF, "std::mem::MaybeUninit".to_string());
    prog.fn_names
        .insert(COPY_DEF, "std::ptr::copy_nonoverlapping".to_string());
    prog.add("load_like_call", body);
    assert_eq!(
        interpret_mir(&prog, "load_like_call", vec![]).expect("load_like_call runs"),
        Value::Float(10.0)
    );
}

/// More than one aggregate element: a slice of arrays copied
/// element by element (the destination is an aggregate of
/// elements, so the indexed path runs with the element type
/// known).
#[test]
fn intrinsic_copy_with_array_elements_count_two() {
    // _1 = [[1, 2], [3, 4]]; _2 = [[0, 0], [0, 0]];
    // _3 = &raw const _1 as *const [f32; 2]; _4 = &raw mut _2;
    // copy_nonoverlapping(_3, _4, 2);
    // return _2[1][0] + _2[0][1]
    let arr2x2 = mir::Ty::Array(Box::new(array2_ty()), mir::ArrayLen::Known(2));
    let pair = |a: f32, b: f32| {
        mir::Rvalue::Aggregate(
            Box::new(mir::AggregateKind::Array(f32_ty())),
            vec![const_f32(a), const_f32(b)],
        )
    };
    let body = mir::Body::new(
        decls(&[
            f32_ty(),
            arr2x2.clone(),
            arr2x2,
            raw_ty(array2_ty(), false),
            raw_ty(array2_ty(), true),
            array2_ty(),
            array2_ty(),
            array2_ty(),
            array2_ty(),
        ]),
        0,
        vec![block(
            vec![
                assign(place(5), pair(1.0, 2.0)),
                assign(place(6), pair(3.0, 4.0)),
                assign(place(7), pair(0.0, 0.0)),
                assign(place(8), pair(0.0, 0.0)),
                assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(array2_ty())),
                        vec![copy(5), copy(6)],
                    ),
                ),
                assign(
                    place(2),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(array2_ty())),
                        vec![copy(7), copy(8)],
                    ),
                ),
                assign(
                    place(3),
                    mir::Rvalue::RawPtr(mir::Mutability::Not, place(1)),
                ),
                assign(
                    place(4),
                    mir::Rvalue::RawPtr(mir::Mutability::Mut, place(2)),
                ),
                intrinsic_stmt(mir::NonDivergingIntrinsic::CopyNonOverlapping {
                    src: copy(3),
                    dst: copy(4),
                    count: const_usize(2),
                }),
                assign(
                    place(0),
                    add(
                        mir::Operand::Copy(idx2(2, 1, 0)),
                        mir::Operand::Copy(idx2(2, 0, 1)),
                    ),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("copy_pairs", body);
    assert_eq!(
        interpret_mir(&prog, "copy_pairs", vec![]).expect("copy_pairs runs"),
        Value::Float(5.0)
    );
}
