//! Regression tests for the "type mismatch in interp conversion"
//! bucket: shapes where a value's representation in this model
//! legitimately differs from the MIR place's declared type, and the
//! conversion / interpreter must bridge them exactly.
//!
//! * A closure body the driver collected through an `FnOnce::
//!   call_once` instance is recorded under the instance's `DefId`
//!   with the trait method's name and takes only its environment
//!   argument (the closure's own body, keyed by the instance). A
//!   call naming the trait method must still dispatch through the
//!   `Fn` trait machinery to the closure value's own body, and the
//!   closure value must resolve to that recorded body.
//! * A raw pointer formed over a slice place is a fat pointer: the
//!   `PtrMetadata` of the place's deref is the slice length, not the
//!   unit value of a thin pointer.
//! * A scalar read through a transparent wrapper (`UnsafeCell<T>`,
//!   the atomic structs) yields the wrapper's one-field aggregate;
//!   storing it into the scalar place unwraps the wrapper chain.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::Value;
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn usize_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::Usize)
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn place(idx: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(idx))
}

fn deref_place(idx: u32) -> mir::Place {
    mir::Place {
        local: mir::Local(idx),
        projection: vec![mir::PlaceElem::Deref],
    }
}

fn move_(idx: u32) -> mir::Operand {
    mir::Operand::Move(place(idx))
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

fn const_usize(v: u128) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int { data: v, size: 8 },
    )))
}

fn fndef_operand(def: mir::DefId, generic_args: Vec<mir::Ty>) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def, generic_args)),
        user_ty: None,
    })
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

fn call(
    func: mir::Operand,
    args: Vec<mir::Operand>,
    dest: u32,
    next: u32,
) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func,
        args,
        destination: place(dest),
        target: Some(mir::BasicBlock(next)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

fn adt_aggregate(
    def: mir::DefId,
    variant_index: u32,
    fields: Vec<mir::Operand>,
) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def,
            variant_index,
            generic_args: vec![],
            user_ty: None,
            field_index: None,
        }),
        fields,
    )
}

/// The closure type as the driver produces it: generic arguments are
/// (kind byte, signature, upvars tuple).
fn closure_ty(def: mir::DefId) -> mir::Ty {
    mir::Ty::Closure(
        def,
        vec![
            mir::Ty::Int(mir::IntTy::I8),
            mir::Ty::FnPtr(Box::new(mir::FnSig::new(
                vec![mir::Ty::Tuple(vec![])],
                i32_ty(),
            ))),
            mir::Ty::Tuple(vec![]),
        ],
    )
}

// ----- FnOnce dispatch to a body recorded under the call_once instance -----

const CLOSURE: mir::DefId = mir::DefId { krate: 0, index: 10 };
const CALL_ONCE_INSTANCE: mir::DefId = mir::DefId { krate: 0, index: 11 };
const CALL_ONCE_FN: mir::DefId = mir::DefId { krate: 0, index: 12 };

#[test]
fn fn_once_call_reaches_closure_body_recorded_under_the_instance() {
    // The closure's own body, as the driver records it when the only
    // demand was through the `FnOnce::call_once` instance: named for
    // the trait method, keyed by the instance's `DefId`, taking only
    // the (by-value) environment argument.
    let closure_body = mir::Body::new(
        decls(&[i32_ty(), closure_ty(CLOSURE)]),
        1,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_i32(42)))],
            ret(),
        )],
    );
    // main: _1 = closure aggregate; _2 = ();
    //       _3 = call_once(move _1, move _2); _0 = copy _3; return
    let main = mir::Body::new(
        decls(&[
            i32_ty(),
            closure_ty(CLOSURE),
            mir::Ty::Tuple(vec![]),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Closure {
                                def: CLOSURE,
                                generic_args: vec![],
                            }),
                            vec![],
                        ),
                    ),
                    assign(
                        place(2),
                        mir::Rvalue::Aggregate(Box::new(mir::AggregateKind::Tuple), vec![]),
                    ),
                ],
                call(
                    fndef_operand(CALL_ONCE_FN, vec![closure_ty(CLOSURE)]),
                    vec![move_(1), move_(2)],
                    3,
                    1,
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(3)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.fn_names.insert(
        CALL_ONCE_FN,
        "std::ops::FnOnce::call_once".to_string(),
    );
    prog.add_with_def_id(
        "std::ops::FnOnce::call_once",
        CALL_ONCE_INSTANCE,
        closure_body,
    );
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Int(42));
}

// ----- PtrMetadata of a raw pointer to a slice place ------------------------

#[test]
fn ptr_metadata_of_raw_pointer_to_slice_place_is_the_length() {
    // fn main(_1: &[i32]) -> usize { _2 = &raw * _1; _0 = PtrMetadata(move _2) }
    let slice_ref_ty = mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(mir::Ty::Slice(Box::new(i32_ty()))),
        mir::Mutability::Not,
    );
    let raw_ptr_ty = mir::Ty::RawPtr(
        Box::new(mir::Ty::Slice(Box::new(i32_ty()))),
        mir::Mutability::Not,
    );
    let main = mir::Body::new(
        decls(&[usize_ty(), slice_ref_ty, raw_ptr_ty]),
        1,
        vec![block(
            vec![
                assign(
                    place(2),
                    mir::Rvalue::RawPtr(mir::Mutability::Not, deref_place(1)),
                ),
                assign(
                    place(0),
                    mir::Rvalue::UnaryOp(mir::UnOp::PtrMetadata, move_(2)),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", main);
    let arg = Value::FatRef {
        base: Box::new(Value::Tuple(vec![
            Value::Int(10),
            Value::Int(20),
            Value::Int(30),
        ])),
        len: 3,
    };
    let result = interpret_mir(&prog, "main", vec![arg]).expect("run must succeed");
    assert_eq!(result, Value::Uint(3));
}

// ----- scalar store unwraps one-field wrapper aggregates --------------------

const OUTER: mir::DefId = mir::DefId { krate: 0, index: 20 };
const INNER: mir::DefId = mir::DefId { krate: 0, index: 21 };

#[test]
fn scalar_store_unwraps_one_field_wrapper_aggregates() {
    // _2 = Inner(7); _1 = Outer(move _2); _0 = move _1; return
    // The whole-value move of the wrapper aggregate into the `usize`
    // place is the shape a read through a raw pointer into a
    // transparent wrapper produces (an atomic load, a `Cell` read).
    let main = mir::Body::new(
        decls(&[
            usize_ty(),
            mir::Ty::Adt(OUTER, vec![]),
            mir::Ty::Adt(INNER, vec![]),
        ]),
        0,
        vec![block(
            vec![
                assign(place(2), adt_aggregate(INNER, 0, vec![const_usize(7)])),
                assign(place(1), adt_aggregate(OUTER, 0, vec![move_(2)])),
                assign(place(0), mir::Rvalue::Use(move_(1))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.fn_names.insert(OUTER, "test::Outer".to_string());
    prog.fn_names.insert(INNER, "test::Inner".to_string());
    prog.add("main", main);
    let result = interpret_mir(&prog, "main", vec![]).expect("run must succeed");
    assert_eq!(result, Value::Uint(7));
}
