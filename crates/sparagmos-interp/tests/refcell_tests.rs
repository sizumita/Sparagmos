//! RefCell model tests (Lane CC): the borrow-flag discipline of
//! `std::cell::RefCell` in the interpreter — the borrow /
//! borrow_mut builtins building two-field guard values, `Ref` /
//! `RefMut` dereference through the guard's value reference, and
//! the guard drop glue (`DropPlan::RefCellRelease`) releasing the
//! flag: at scope end, through explicit `std::mem::drop`, and
//! across call frames. Panic messages are Rust 1.100.0's real
//! wording, verified natively.
//!
//! The programs are hand-built `sparagmos-mir` bodies shaped after
//! the driver's output for the corresponding Rust sources,
//! following the `builtins_tests` / `drop_glue_tests` fixture
//! style.

use std::collections::HashMap;

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::{InterpError, MirRunError, Value};
use sparagmos_mir as mir;

// ----- shared builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}

fn bool_ty() -> mir::Ty {
    mir::Ty::Bool
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

fn move_(idx: u32) -> mir::Operand {
    mir::Operand::Move(place(idx))
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

fn goto(next: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Goto {
        target: mir::BasicBlock(next),
    }
}

fn drop(place_: mir::Place, next: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Drop {
        place: place_,
        target: mir::BasicBlock(next),
        unwind: mir::UnwindAction::Continue,
        replace: false,
    }
}

fn fndef(def: mir::DefId, generic_args: Vec<mir::Ty>) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(def, generic_args)),
        user_ty: None,
    })
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

/// `std::mem::drop::<ty>(move _arg) -> _dest; goto next`.
fn mem_drop(ty: mir::Ty, arg: u32, dest: u32, next: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func: fndef(MEM_DROP_FN, vec![ty]),
        args: vec![move_(arg)],
        destination: place(dest),
        target: Some(mir::BasicBlock(next)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

fn ref_rvalue(idx: u32, mutable: bool) -> mir::Rvalue {
    let kind = if mutable {
        mir::BorrowKind::Mut {
            allow_two_phase_borrow: false,
        }
    } else {
        mir::BorrowKind::Shared
    };
    mir::Rvalue::Ref(mir::Region::Erased, kind, place(idx))
}

fn ref_ty(inner: mir::Ty, mutable: bool) -> mir::Ty {
    mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(inner),
        if mutable {
            mir::Mutability::Mut
        } else {
            mir::Mutability::Not
        },
    )
}

fn adt_aggregate(
    def: mir::DefId,
    variant_index: u32,
    generic_args: Vec<mir::Ty>,
    fields: Vec<mir::Operand>,
) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def,
            variant_index,
            generic_args,
            user_ty: None,
            field_index: None,
        }),
        fields,
    )
}

fn binop(op: mir::BinOp, lhs: mir::Operand, rhs: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::BinaryOp(op, Box::new((lhs, rhs)))
}

// ----- the items the tests name ----------------------------------------------

const REFCELL_TY: mir::DefId = mir::DefId::new(0, 300);
const REF_TY: mir::DefId = mir::DefId::new(0, 301);
const REFMUT_TY: mir::DefId = mir::DefId::new(0, 302);
const REFCELL_NEW: mir::DefId = mir::DefId::new(0, 303);
const REFCELL_BORROW: mir::DefId = mir::DefId::new(0, 304);
const REFCELL_BORROW_MUT: mir::DefId = mir::DefId::new(0, 305);
const REFCELL_TRY_BORROW: mir::DefId = mir::DefId::new(0, 306);
const REFCELL_TRY_BORROW_MUT: mir::DefId = mir::DefId::new(0, 307);
const REFCELL_GET_MUT: mir::DefId = mir::DefId::new(0, 308);
const REFCELL_REPLACE: mir::DefId = mir::DefId::new(0, 309);
const REFCELL_AS_PTR: mir::DefId = mir::DefId::new(0, 310);
const DEREF: mir::DefId = mir::DefId::new(0, 311);
const DEREF_MUT: mir::DefId = mir::DefId::new(0, 312);
const RESULT_TY: mir::DefId = mir::DefId::new(0, 313);
const RESULT_IS_ERR: mir::DefId = mir::DefId::new(0, 314);
const RESULT_UNWRAP: mir::DefId = mir::DefId::new(0, 315);
const MEM_DROP_FN: mir::DefId = mir::DefId::new(0, 316);
const TOUCH_FN: mir::DefId = mir::DefId::new(0, 317);
const BORROW_ERROR_TY: mir::DefId = mir::DefId::new(0, 318);
const BORROW_MUT_ERROR_TY: mir::DefId = mir::DefId::new(0, 319);
const D_TY: mir::DefId = mir::DefId::new(0, 320);

const D_DROP_NAME: &str = "<probe::D as std::ops::Drop>::drop";

fn std_names() -> HashMap<mir::DefId, String> {
    HashMap::from([
        (REFCELL_TY, "std::cell::RefCell".to_string()),
        (REF_TY, "std::cell::Ref".to_string()),
        (REFMUT_TY, "std::cell::RefMut".to_string()),
        (REFCELL_NEW, "std::cell::RefCell::<T>::new".to_string()),
        (REFCELL_BORROW, "std::cell::RefCell::<T>::borrow".to_string()),
        (
            REFCELL_BORROW_MUT,
            "std::cell::RefCell::<T>::borrow_mut".to_string(),
        ),
        (
            REFCELL_TRY_BORROW,
            "std::cell::RefCell::<T>::try_borrow".to_string(),
        ),
        (
            REFCELL_TRY_BORROW_MUT,
            "std::cell::RefCell::<T>::try_borrow_mut".to_string(),
        ),
        (REFCELL_GET_MUT, "std::cell::RefCell::<T>::get_mut".to_string()),
        (REFCELL_REPLACE, "std::cell::RefCell::<T>::replace".to_string()),
        (REFCELL_AS_PTR, "std::cell::RefCell::<T>::as_ptr".to_string()),
        (DEREF, "std::ops::Deref::deref".to_string()),
        (DEREF_MUT, "std::ops::DerefMut::deref_mut".to_string()),
        (RESULT_TY, "std::result::Result".to_string()),
        (
            RESULT_IS_ERR,
            "std::result::Result::<T, E>::is_err".to_string(),
        ),
        (
            RESULT_UNWRAP,
            "std::result::Result::<T, E>::unwrap".to_string(),
        ),
        (MEM_DROP_FN, "std::mem::drop".to_string()),
        (TOUCH_FN, "probe::touch".to_string()),
        (BORROW_ERROR_TY, "std::cell::BorrowError".to_string()),
        (BORROW_MUT_ERROR_TY, "std::cell::BorrowMutError".to_string()),
        (D_TY, "probe::D".to_string()),
    ])
}

fn refcell_i32() -> mir::Ty {
    mir::Ty::Adt(REFCELL_TY, vec![i32_ty()])
}

fn ref_i32() -> mir::Ty {
    mir::Ty::Adt(REF_TY, vec![i32_ty()])
}

fn refmut_i32() -> mir::Ty {
    mir::Ty::Adt(REFMUT_TY, vec![i32_ty()])
}

fn borrow_error_ty() -> mir::Ty {
    mir::Ty::Adt(BORROW_ERROR_TY, vec![])
}

fn borrow_mut_error_ty() -> mir::Ty {
    mir::Ty::Adt(BORROW_MUT_ERROR_TY, vec![])
}

fn result_ref() -> mir::Ty {
    mir::Ty::Adt(RESULT_TY, vec![ref_i32(), borrow_error_ty()])
}

fn result_refmut() -> mir::Ty {
    mir::Ty::Adt(RESULT_TY, vec![refmut_i32(), borrow_mut_error_ty()])
}

fn d_ty() -> mir::Ty {
    mir::Ty::Adt(D_TY, vec![])
}

fn program(main: mir::Body, extra: &[(&str, mir::Body)]) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.add("main", main);
    for (name, body) in extra {
        prog.add(*name, body.clone());
    }
    prog.fn_names = std_names();
    prog
}

fn run(main: mir::Body) -> Result<Value, MirRunError> {
    interpret_mir(&program(main, &[]), "main", vec![])
}

fn expect_panic(result: Result<Value, MirRunError>) -> String {
    match result {
        Err(MirRunError::Interp(InterpError::Panic { message })) => message,
        other => panic!("expected a guest panic, got {other:?}"),
    }
}

// ----- borrow / deref ----------------------------------------------------------

#[test]
fn borrow_reads_value_through_deref() {
    // fn main() -> i32 {
    //   _1 = RefCell::new(41);
    //   _3 = RefCell::borrow(&_1);
    //   _5 = Deref::deref(&_3);
    //   _0 = *_5;
    //   drop(_3);
    //   return _0;
    // }
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            refcell_i32(),
            ref_ty(refcell_i32(), false),
            ref_i32(),
            ref_ty(ref_i32(), false),
            ref_ty(i32_ty(), false),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(41)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![assign(place(4), ref_rvalue(3, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(4)], 5, 3),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(deref_place(5))),
                )],
                goto(4),
            ),
            block(vec![], drop(place(3), 5)),
            block(vec![], ret()),
        ],
    );
    let value = run(body).expect("run");
    assert_eq!(value, Value::Int(41));
}

#[test]
fn shared_borrows_coexist_then_borrow_mut_after_drops() {
    // let c = RefCell::new(5);
    // let a = c.borrow(); let b = c.borrow();
    // let sum = *a + *b;              // 10
    // drop(a); drop(b);
    // *c.borrow_mut() = 7;            // released, so this succeeds
    // return sum * 100 + *c.borrow(); // 1007
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                    // _0
            refcell_i32(),               // _1 cell
            ref_ty(refcell_i32(), false), // _2
            ref_i32(),                   // _3 guard a
            ref_ty(refcell_i32(), false), // _4
            ref_i32(),                   // _5 guard b
            ref_ty(ref_i32(), false),    // _6
            ref_ty(i32_ty(), false),     // _7
            i32_ty(),                    // _8 read a
            ref_ty(ref_i32(), false),    // _9
            ref_ty(i32_ty(), false),     // _10
            i32_ty(),                    // _11 read b
            i32_ty(),                    // _12 sum
            ref_ty(refcell_i32(), false), // _13
            refmut_i32(),                // _14 guard mut
            ref_ty(refmut_i32(), true),  // _15
            ref_ty(i32_ty(), true),      // _16
            ref_ty(refcell_i32(), false), // _17
            ref_i32(),                   // _18 guard c
            ref_ty(ref_i32(), false),    // _19
            ref_ty(i32_ty(), false),     // _20
            i32_ty(),                    // _21 final read
            i32_ty(),                    // _22 tmp
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(5)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(4)], 5, 3),
            ),
            block(
                vec![assign(place(6), ref_rvalue(3, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(6)], 7, 4),
            ),
            block(
                vec![
                    assign(place(8), mir::Rvalue::Use(mir::Operand::Copy(deref_place(7)))),
                    assign(place(9), ref_rvalue(5, false)),
                ],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(9)], 10, 5),
            ),
            block(
                vec![
                    assign(place(11), mir::Rvalue::Use(mir::Operand::Copy(deref_place(10)))),
                    assign(place(12), binop(mir::BinOp::Add, copy(8), copy(11))),
                ],
                drop(place(3), 6),
            ),
            block(vec![], drop(place(5), 7)),
            block(
                vec![assign(place(13), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW_MUT, vec![i32_ty()]), vec![move_(13)], 14, 8),
            ),
            block(
                vec![assign(place(15), ref_rvalue(14, true))],
                call(fndef(DEREF_MUT, vec![refmut_i32()]), vec![move_(15)], 16, 9),
            ),
            block(
                vec![assign(deref_place(16), mir::Rvalue::Use(const_i32(7)))],
                drop(place(14), 10),
            ),
            block(
                vec![assign(place(17), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(17)], 18, 11),
            ),
            block(
                vec![assign(place(19), ref_rvalue(18, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(19)], 20, 12),
            ),
            block(
                vec![
                    assign(place(21), mir::Rvalue::Use(mir::Operand::Copy(deref_place(20)))),
                    assign(place(22), binop(mir::BinOp::Mul, copy(12), const_i32(100))),
                    assign(place(0), binop(mir::BinOp::Add, copy(22), copy(21))),
                ],
                drop(place(18), 13),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(body).expect("run");
    assert_eq!(value, Value::Int(1007));
}

#[test]
fn borrow_mut_while_shared_borrow_panics() {
    // let c = RefCell::new(1);
    // let _a = c.borrow();
    // let _b = c.borrow_mut(); // panics: RefCell already borrowed
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            refcell_i32(),
            ref_ty(refcell_i32(), false),
            ref_i32(),
            ref_ty(refcell_i32(), false),
            refmut_i32(),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(1)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW_MUT, vec![i32_ty()]), vec![move_(4)], 5, 3),
            ),
            block(vec![], ret()),
        ],
    );
    let message = expect_panic(run(body));
    assert_eq!(message, "RefCell already borrowed");
}

#[test]
fn borrow_while_borrow_mut_panics() {
    // let c = RefCell::new(1);
    // let _m = c.borrow_mut();
    // let _a = c.borrow(); // panics: RefCell already mutably borrowed
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            refcell_i32(),
            ref_ty(refcell_i32(), false),
            refmut_i32(),
            ref_ty(refcell_i32(), false),
            ref_i32(),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(1)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW_MUT, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(4)], 5, 3),
            ),
            block(vec![], ret()),
        ],
    );
    let message = expect_panic(run(body));
    assert_eq!(message, "RefCell already mutably borrowed");
}

#[test]
fn borrow_mut_while_borrow_mut_panics() {
    // let c = RefCell::new(1);
    // let _a = c.borrow_mut();
    // let _b = c.borrow_mut(); // panics: RefCell already borrowed
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            refcell_i32(),
            ref_ty(refcell_i32(), false),
            refmut_i32(),
            ref_ty(refcell_i32(), false),
            refmut_i32(),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(1)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW_MUT, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW_MUT, vec![i32_ty()]), vec![move_(4)], 5, 3),
            ),
            block(vec![], ret()),
        ],
    );
    let message = expect_panic(run(body));
    assert_eq!(message, "RefCell already borrowed");
}

// ----- guard release paths -----------------------------------------------------

#[test]
fn mem_drop_of_guard_releases_the_borrow() {
    // let c = RefCell::new(3);
    // let m = c.borrow_mut();
    // std::mem::drop(m);            // explicit drop releases the flag
    // *c.borrow_mut() = 8;          // succeeds only if released
    // return *c.borrow();           // 8
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                     // _0
            refcell_i32(),                // _1 cell
            ref_ty(refcell_i32(), false), // _2
            refmut_i32(),                 // _3 guard m
            mir::Ty::unit(),              // _4 mem::drop dest
            ref_ty(refcell_i32(), false), // _5
            refmut_i32(),                 // _6 guard m2
            ref_ty(refmut_i32(), true),   // _7
            ref_ty(i32_ty(), true),       // _8
            ref_ty(refcell_i32(), false), // _9
            ref_i32(),                    // _10 guard
            ref_ty(ref_i32(), false),     // _11
            ref_ty(i32_ty(), false),      // _12
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(3)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW_MUT, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(vec![], mem_drop(refmut_i32(), 3, 4, 3)),
            block(
                vec![assign(place(5), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW_MUT, vec![i32_ty()]), vec![move_(5)], 6, 4),
            ),
            block(
                vec![assign(place(7), ref_rvalue(6, true))],
                call(fndef(DEREF_MUT, vec![refmut_i32()]), vec![move_(7)], 8, 5),
            ),
            block(
                vec![assign(deref_place(8), mir::Rvalue::Use(const_i32(8)))],
                drop(place(6), 6),
            ),
            block(
                vec![assign(place(9), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(9)], 10, 7),
            ),
            block(
                vec![assign(place(11), ref_rvalue(10, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(11)], 12, 8),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(deref_place(12))),
                )],
                drop(place(10), 9),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(body).expect("run");
    assert_eq!(value, Value::Int(8));
}

#[test]
fn guard_dropped_at_callee_frame_end_releases_the_caller_cell() {
    // fn touch(c: &RefCell<i32>) -> i32 {
    //   let g = c.borrow();   // guard lives in THIS frame
    //   *g                    // its frame-end Drop must release
    // }                       // the borrow in the caller's cell
    // fn main() -> i32 {
    //   let c = RefCell::new(6);
    //   let seen = touch(&c);       // 6
    //   *c.borrow_mut() = 60;       // panics unless touch released
    //   *c.borrow() + seen          // 66
    // }
    let touch = mir::Body::new(
        decls(&[
            i32_ty(),
            ref_ty(refcell_i32(), false),
            ref_i32(),
            ref_ty(ref_i32(), false),
            ref_ty(i32_ty(), false),
        ]),
        1,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![copy(1)], 2, 1),
            ),
            block(
                vec![assign(place(3), ref_rvalue(2, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(3)], 4, 2),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(deref_place(4))),
                )],
                drop(place(2), 3),
            ),
            block(vec![], ret()),
        ],
    );
    let main = mir::Body::new(
        decls(&[
            i32_ty(),                     // _0
            refcell_i32(),                // _1 cell
            ref_ty(refcell_i32(), false), // _2
            i32_ty(),                     // _3 seen
            ref_ty(refcell_i32(), false), // _4
            refmut_i32(),                 // _5 guard mut
            ref_ty(refmut_i32(), true),   // _6
            ref_ty(i32_ty(), true),       // _7
            ref_ty(refcell_i32(), false), // _8
            ref_i32(),                    // _9 guard
            ref_ty(ref_i32(), false),     // _10
            ref_ty(i32_ty(), false),      // _11
            i32_ty(),                     // _12 read
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(6)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(TOUCH_FN, vec![]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW_MUT, vec![i32_ty()]), vec![move_(4)], 5, 3),
            ),
            block(
                vec![assign(place(6), ref_rvalue(5, true))],
                call(fndef(DEREF_MUT, vec![refmut_i32()]), vec![move_(6)], 7, 4),
            ),
            block(
                vec![assign(deref_place(7), mir::Rvalue::Use(const_i32(60)))],
                drop(place(5), 5),
            ),
            block(
                vec![assign(place(8), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(8)], 9, 6),
            ),
            block(
                vec![assign(place(10), ref_rvalue(9, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(10)], 11, 7),
            ),
            block(
                vec![
                    assign(place(12), mir::Rvalue::Use(mir::Operand::Copy(deref_place(11)))),
                    assign(place(0), binop(mir::BinOp::Add, copy(12), copy(3))),
                ],
                drop(place(9), 8),
            ),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("main", main);
    prog.add_with_def_id("probe::touch", TOUCH_FN, touch);
    prog.fn_names = std_names();
    let value = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(66));
}

// ----- try_borrow / try_borrow_mut ----------------------------------------------

#[test]
fn try_borrow_err_while_mutably_borrowed() {
    // let c = RefCell::new(11);
    // let _m = c.borrow_mut();
    // return c.try_borrow().is_err();   // true
    let body = mir::Body::new(
        decls(&[
            bool_ty(),
            refcell_i32(),
            ref_ty(refcell_i32(), false),
            refmut_i32(),
            ref_ty(refcell_i32(), false),
            result_ref(),
            ref_ty(result_ref(), false),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(11)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW_MUT, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, false))],
                call(fndef(REFCELL_TRY_BORROW, vec![i32_ty()]), vec![move_(4)], 5, 3),
            ),
            block(
                vec![assign(place(6), ref_rvalue(5, false))],
                call(
                    fndef(RESULT_IS_ERR, vec![ref_i32(), borrow_error_ty()]),
                    vec![move_(6)],
                    0,
                    4,
                ),
            ),
            block(vec![], drop(place(5), 5)),
            block(vec![], drop(place(3), 6)),
            block(vec![], ret()),
        ],
    );
    let value = run(body).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn try_borrow_ok_unwraps_and_reads() {
    // let c = RefCell::new(11);
    // let g = c.try_borrow().unwrap();
    // return *g;   // 11
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            refcell_i32(),
            ref_ty(refcell_i32(), false),
            result_ref(),
            ref_i32(),
            ref_ty(ref_i32(), false),
            ref_ty(i32_ty(), false),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(11)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_TRY_BORROW, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![],
                call(
                    fndef(RESULT_UNWRAP, vec![ref_i32(), borrow_error_ty()]),
                    vec![move_(3)],
                    4,
                    3,
                ),
            ),
            block(
                vec![assign(place(5), ref_rvalue(4, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(5)], 6, 4),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(deref_place(6))),
                )],
                drop(place(4), 5),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(body).expect("run");
    assert_eq!(value, Value::Int(11));
}

#[test]
fn try_borrow_mut_err_while_shared_borrow_held() {
    // let c = RefCell::new(2);
    // let _g = c.borrow();
    // return c.try_borrow_mut().is_err();   // true
    let body = mir::Body::new(
        decls(&[
            bool_ty(),
            refcell_i32(),
            ref_ty(refcell_i32(), false),
            ref_i32(),
            ref_ty(refcell_i32(), false),
            result_refmut(),
            ref_ty(result_refmut(), false),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(2)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, false))],
                call(fndef(REFCELL_TRY_BORROW_MUT, vec![i32_ty()]), vec![move_(4)], 5, 3),
            ),
            block(
                vec![assign(place(6), ref_rvalue(5, false))],
                call(
                    fndef(RESULT_IS_ERR, vec![refmut_i32(), borrow_mut_error_ty()]),
                    vec![move_(6)],
                    0,
                    4,
                ),
            ),
            block(vec![], drop(place(5), 5)),
            block(vec![], drop(place(3), 6)),
            block(vec![], ret()),
        ],
    );
    let value = run(body).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn try_borrow_mut_ok_writes_through_unwrapped_guard() {
    // let c = RefCell::new(2);
    // *c.try_borrow_mut().unwrap() = 5;
    // return *c.borrow();   // 5
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                     // _0
            refcell_i32(),                // _1
            ref_ty(refcell_i32(), false), // _2
            result_refmut(),              // _3
            refmut_i32(),                 // _4
            ref_ty(refmut_i32(), true),   // _5
            ref_ty(i32_ty(), true),       // _6
            ref_ty(refcell_i32(), false), // _7
            ref_i32(),                    // _8
            ref_ty(ref_i32(), false),     // _9
            ref_ty(i32_ty(), false),      // _10
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(2)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_TRY_BORROW_MUT, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![],
                call(
                    fndef(RESULT_UNWRAP, vec![refmut_i32(), borrow_mut_error_ty()]),
                    vec![move_(3)],
                    4,
                    3,
                ),
            ),
            block(
                vec![assign(place(5), ref_rvalue(4, true))],
                call(fndef(DEREF_MUT, vec![refmut_i32()]), vec![move_(5)], 6, 4),
            ),
            block(
                vec![assign(deref_place(6), mir::Rvalue::Use(const_i32(5)))],
                drop(place(4), 5),
            ),
            block(
                vec![assign(place(7), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(7)], 8, 6),
            ),
            block(
                vec![assign(place(9), ref_rvalue(8, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(9)], 10, 7),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(deref_place(10))),
                )],
                drop(place(8), 8),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(body).expect("run");
    assert_eq!(value, Value::Int(5));
}

// ----- get_mut / replace / as_ptr ------------------------------------------------

#[test]
fn get_mut_writes_through_the_unique_reference() {
    // let mut c = RefCell::new(4);
    // *c.get_mut() = 9;
    // return *c.borrow();   // 9
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            refcell_i32(),
            ref_ty(refcell_i32(), true),
            ref_ty(i32_ty(), true),
            ref_ty(refcell_i32(), false),
            ref_i32(),
            ref_ty(ref_i32(), false),
            ref_ty(i32_ty(), false),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(4)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, true))],
                call(fndef(REFCELL_GET_MUT, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![
                    assign(deref_place(3), mir::Rvalue::Use(const_i32(9))),
                    assign(place(4), ref_rvalue(1, false)),
                ],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(4)], 5, 3),
            ),
            block(
                vec![assign(place(6), ref_rvalue(5, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(6)], 7, 4),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(deref_place(7))),
                )],
                drop(place(5), 5),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(body).expect("run");
    assert_eq!(value, Value::Int(9));
}

#[test]
fn replace_swaps_and_returns_the_old_value() {
    // let c = RefCell::new(3);
    // let old = c.replace(10);
    // return old * 100 + *c.borrow();   // 310
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                     // _0
            refcell_i32(),                // _1
            ref_ty(refcell_i32(), false), // _2
            i32_ty(),                     // _3 old
            ref_ty(refcell_i32(), false), // _4
            ref_i32(),                    // _5
            ref_ty(ref_i32(), false),     // _6
            ref_ty(i32_ty(), false),      // _7
            i32_ty(),                     // _8 read
            i32_ty(),                     // _9 tmp
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(3)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(
                    fndef(REFCELL_REPLACE, vec![i32_ty()]),
                    vec![move_(2), const_i32(10)],
                    3,
                    2,
                ),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, false))],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(4)], 5, 3),
            ),
            block(
                vec![assign(place(6), ref_rvalue(5, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(6)], 7, 4),
            ),
            block(
                vec![
                    assign(place(8), mir::Rvalue::Use(mir::Operand::Copy(deref_place(7)))),
                    assign(place(9), binop(mir::BinOp::Mul, copy(3), const_i32(100))),
                    assign(place(0), binop(mir::BinOp::Add, copy(9), copy(8))),
                ],
                drop(place(5), 5),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(body).expect("run");
    assert_eq!(value, Value::Int(310));
}

#[test]
fn as_ptr_writes_through_the_raw_pointer() {
    // let c = RefCell::new(4);
    // unsafe { *c.as_ptr() = 5; }
    // return *c.borrow();   // 5
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            refcell_i32(),
            ref_ty(refcell_i32(), false),
            mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Mut),
            ref_ty(refcell_i32(), false),
            ref_i32(),
            ref_ty(ref_i32(), false),
            ref_ty(i32_ty(), false),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_NEW, vec![i32_ty()]), vec![const_i32(4)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(REFCELL_AS_PTR, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(
                vec![
                    assign(deref_place(3), mir::Rvalue::Use(const_i32(5))),
                    assign(place(4), ref_rvalue(1, false)),
                ],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![move_(4)], 5, 3),
            ),
            block(
                vec![assign(place(6), ref_rvalue(5, false))],
                call(fndef(DEREF, vec![ref_i32()]), vec![move_(6)], 7, 4),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(deref_place(7))),
                )],
                drop(place(5), 5),
            ),
            block(vec![], ret()),
        ],
    );
    let value = run(body).expect("run");
    assert_eq!(value, Value::Int(5));
}

// ----- RefCell<D>: the wrapped value's destructor runs ----------------------------

/// `<D as Drop>::drop`: counter = counter * 10 + id, through the
/// sink reference the D value carries (the `drop_glue_tests`
/// fixture, transcribed exactly).
fn bump_dtor(self_ty: mir::Ty, sink_field: u32, id_field: u32) -> mir::Body {
    mir::Body::new(
        decls(&[
            mir::Ty::unit(),
            ref_ty(self_ty, false),
            ref_ty(i32_ty(), false),
            i32_ty(),
            i32_ty(),
            i32_ty(),
        ]),
        1,
        vec![block(
            vec![
                assign(
                    place(2),
                    mir::Rvalue::Use(mir::Operand::Copy(
                        place(1).deref().field(sink_field, ref_ty(i32_ty(), false)),
                    )),
                ),
                assign(
                    place(3),
                    mir::Rvalue::Use(mir::Operand::Copy(
                        place(1).deref().field(id_field, i32_ty()),
                    )),
                ),
                assign(
                    place(4),
                    mir::Rvalue::Use(mir::Operand::Copy(place(2).deref())),
                ),
                assign(
                    place(5),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Mul,
                        Box::new((copy(4), const_i32(10))),
                    ),
                ),
                assign(
                    place(5),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((copy(5), copy(3))),
                    ),
                ),
                assign(place(2).deref(), mir::Rvalue::Use(copy(5))),
            ],
            ret(),
        )],
    )
}

#[test]
fn refcell_drop_runs_the_wrapped_values_destructor() {
    // let mut counter = 0;
    // let cell = RefCell::new(D { sink: &mut counter, id: 7 });
    // drop(cell);            // must run D::drop through [F1, F0]
    // return counter;        // 0 * 10 + 7 = 7
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                               // _0 return
            i32_ty(),                               // _1 counter
            ref_ty(i32_ty(), true),                 // _2 &mut counter
            d_ty(),                                 // _3 the D value
            mir::Ty::Adt(REFCELL_TY, vec![d_ty()]), // _4 the cell
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), ref_rvalue(1, true)),
                    assign(
                        place(3),
                        adt_aggregate(D_TY, 0, vec![], vec![move_(2), const_i32(7)]),
                    ),
                ],
                call(fndef(REFCELL_NEW, vec![d_ty()]), vec![move_(3)], 4, 1),
            ),
            block(vec![], drop(place(4), 2)),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let mut prog = program(body, &[]);
    prog.add(D_DROP_NAME, bump_dtor(d_ty(), 0, 1));
    let value = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(7));
}

#[test]
fn borrow_of_a_constant_aggregate_cell_is_an_explicit_error() {
    // The cell exists only as a constant aggregate value (no live
    // place): `static`-style. The borrow flag has nowhere to live,
    // so the borrow fails with a named Unsupported error instead
    // of inventing state.
    let cell_const = mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Ref(
        Box::new(mir::ConstValue::Adt {
            variant: 0,
            fields: vec![
                mir::ConstValue::Adt {
                    variant: 0,
                    fields: vec![mir::ConstValue::Adt {
                        variant: 0,
                        fields: vec![mir::ConstValue::Scalar(mir::Scalar::Int {
                            data: 0,
                            size: 8,
                        })],
                    }],
                },
                mir::ConstValue::Adt {
                    variant: 0,
                    fields: vec![mir::ConstValue::Scalar(mir::Scalar::Int { data: (-3i32) as u32 as u128, size: 4 })],
                },
            ],
        }),
    )));
    let body = mir::Body::new(
        decls(&[i32_ty(), refcell_i32(), ref_i32()]),
        0,
        vec![
            block(
                vec![],
                call(fndef(REFCELL_BORROW, vec![i32_ty()]), vec![cell_const], 2, 1),
            ),
            block(vec![], ret()),
        ],
    );
    match run(body) {
        Err(MirRunError::Interp(err)) => {
            let text = err.to_string();
            assert!(
                text.contains("constant aggregate"),
                "unexpected error: {text}"
            );
        }
        other => panic!("expected the explicit constant-aggregate error, got {other:?}"),
    }
}
