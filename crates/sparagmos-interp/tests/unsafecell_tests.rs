//! Transparent-wrapper pointer tests (Lane DP): a pointer cast
//! from `UnsafeCell<T>` / `MaybeUninit<T>` to `T` itself - what
//! `UnsafeCell::get` and `MaybeUninit::as_mut_ptr` return - denotes
//! the wrapper's only field, so reading and writing through it
//! reach the wrapped value, not the wrapper aggregate.
//!
//! The programs are hand-built `sparagmos-mir` bodies shaped after
//! the driver's output for the corresponding Rust sources (the
//! `get` body is the real 1.100.0 shape: a raw borrow of the
//! referent followed by a `PtrToPtr` cast), following the
//! `mir_convert_tests` fixture style.

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::{InterpError, MirRunError, Value};
use sparagmos_mir as mir;

// ----- shared builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn i64_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I64)
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

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

fn fndef_operand(def: mir::DefId) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def, Vec::new())),
        user_ty: None,
        ty: None,
        generic_type_ids: None,
    })
}

fn call(target_fn: mir::Operand, arg: mir::Operand, dest: u32, next: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func: target_fn,
        args: vec![arg],
        destination: place(dest),
        target: Some(mir::BasicBlock(next)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

// ----- the wrapper fixtures ---------------------------------------------------

const UC_DEF: mir::DefId = mir::DefId::new(0, 300);
const UC_GET_DEF: mir::DefId = mir::DefId::new(0, 301);
const MU_DEF: mir::DefId = mir::DefId::new(0, 302);
const MU_PTR_DEF: mir::DefId = mir::DefId::new(0, 303);

fn wrapper_ty(def: mir::DefId) -> mir::Ty {
    mir::Ty::Adt(def, vec![i32_ty()])
}

fn ref_ty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Not)
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

/// The `UnsafeCell::get` / `MaybeUninit::as_mut_ptr` body shape:
/// `_2 = &raw const (*_1); _0 = copy _2 as *mut T (PtrToPtr); return`
/// with the cast target a pointer to `target_pointee`.
fn get_body(def: mir::DefId, target_pointee: mir::Ty) -> mir::Body {
    let w = wrapper_ty(def);
    mir::Body::new(
        decls(&[
            raw_ty(target_pointee.clone(), true),
            ref_ty(w.clone()),
            raw_ty(w, false),
        ]),
        1,
        vec![block(
            vec![
                assign(
                    place(2),
                    mir::Rvalue::RawPtr(mir::Mutability::Not, deref_place(1)),
                ),
                assign(
                    place(0),
                    mir::Rvalue::Cast(
                        mir::CastKind::PtrToPtr,
                        copy(2),
                        raw_ty(target_pointee, true),
                    ),
                ),
            ],
            ret(),
        )],
    )
}

/// A main that builds the wrapper aggregate holding `init`, calls
/// the get-shaped body on it, optionally writes `written` through
/// the returned pointer, and returns the value read back through
/// the pointer.
/// Locals: _0 ret i32, _1 wrapper, _2 &wrapper, _3 *mut i32, _4 ().
fn main_body(def: mir::DefId, get_def: mir::DefId, init: i32, written: Option<i32>) -> mir::Body {
    let w = wrapper_ty(def);
    let mut tail = Vec::new();
    if let Some(v) = written {
        tail.push(assign(deref_place(3), mir::Rvalue::Use(const_i32(v))));
    }
    tail.push(assign(place(0), mir::Rvalue::Use(mir::Operand::Copy(deref_place(3)))));
    mir::Body::new(
        decls(&[
            i32_ty(),
            w.clone(),
            ref_ty(w),
            raw_ty(i32_ty(), true),
            mir::Ty::Tuple(Vec::new()),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Adt {
                                def,
                                variant_index: 0,
                                generic_args: vec![i32_ty()],
                                user_ty: None,
                                field_index: None,
                            }),
                            vec![const_i32(init)],
                        ),
                    ),
                    assign(
                        place(2),
                        mir::Rvalue::Ref(
                            mir::Region::Erased,
                            mir::BorrowKind::Shared,
                            place(1),
                        ),
                    ),
                ],
                call(fndef_operand(get_def), copy(2), 3, 1),
            ),
            block(tail, ret()),
        ],
    )
}

fn wrapper_prog(
    def: mir::DefId,
    adt_name: &str,
    get_def: mir::DefId,
    get_name: &str,
    target_pointee: mir::Ty,
    init: i32,
    written: Option<i32>,
) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names.insert(def, adt_name.to_string());
    prog.add_with_def_id(get_name, get_def, get_body(def, target_pointee.clone()));
    prog.add("main", main_body(def, get_def, init, written));
    prog
}

// ----- the tests ---------------------------------------------------------------

#[test]
fn unsafecell_get_write_then_read_reaches_the_wrapped_value() {
    let prog = wrapper_prog(
        UC_DEF,
        "std::cell::UnsafeCell",
        UC_GET_DEF,
        "std::cell::UnsafeCell::<T>::get",
        i32_ty(),
        5,
        Some(7),
    );
    let v = interpret_mir(&prog, "main", vec![]).expect("program must run");
    assert_eq!(v, Value::Int(7));
}

#[test]
fn unsafecell_get_read_reaches_the_wrapped_value() {
    let prog = wrapper_prog(
        UC_DEF,
        "std::cell::UnsafeCell",
        UC_GET_DEF,
        "std::cell::UnsafeCell::<T>::get",
        i32_ty(),
        42,
        None,
    );
    let v = interpret_mir(&prog, "main", vec![]).expect("program must run");
    assert_eq!(v, Value::Int(42));
}

#[test]
fn maybeuninit_as_mut_ptr_write_then_read_reaches_the_wrapped_value() {
    let prog = wrapper_prog(
        MU_DEF,
        "std::mem::MaybeUninit",
        MU_PTR_DEF,
        "std::mem::MaybeUninit::<T>::as_mut_ptr",
        i32_ty(),
        0,
        Some(11),
    );
    let v = interpret_mir(&prog, "main", vec![]).expect("program must run");
    assert_eq!(v, Value::Int(11));
}

/// A cast from the wrapper to an *unrelated* pointee type is not
/// the transparent shape: the model cannot reinterpret the
/// aggregate's bytes, so the read stays an explicit type mismatch
/// rather than a fabricated value.
#[test]
fn wrapper_cast_to_an_unrelated_pointee_is_not_reinterpreted() {
    let prog = wrapper_prog(
        UC_DEF,
        "std::cell::UnsafeCell",
        UC_GET_DEF,
        "std::cell::UnsafeCell::<T>::get",
        i64_ty(),
        5,
        None,
    );
    let err = interpret_mir(&prog, "main", vec![]).expect_err("must not reinterpret");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::TypeMismatch { .. })
        ),
        "expected a type mismatch, got {err}"
    );
}
