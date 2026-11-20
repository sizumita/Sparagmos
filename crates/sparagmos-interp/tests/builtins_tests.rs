//! Runtime-support tests (Lane V): the builtin registry for calls
//! whose bodies exist only in std/core — the panic path, semantic
//! comparisons, modelled containers (`Vec`, `Box`, `Cell`), `Option` /
//! `Result` methods, slice iterators, indexing and `std::mem`.
//!
//! The bodies are hand-built `sparagmos-mir`, shaped after the real
//! `rustc 1.100.0` MIR for the corresponding source constructs (the
//! shapes were checked against `rustc --emit mir` output), converted
//! and executed through `mir_convert`.

use std::collections::HashMap;

use sparagmos_interp::builtins::{canonical_name, lookup, BitOp, Builtin, RelOp, SatOp, TrimSide};
use sparagmos_interp::mir_convert::{interpret_mir, MirProgram};
use sparagmos_interp::{InterpError, MirRunError, Value};
use sparagmos_mir as mir;

// ----- shared builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}
fn usize_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::Usize)
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

fn const_int(data: u128, size: u8) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int { data, size },
    )))
}

fn const_i32(v: i32) -> mir::Operand {
    const_int(v as u32 as u128, 4)
}

fn const_usize(v: u64) -> mir::Operand {
    const_int(v as u128, 8)
}

fn const_str(s: &str) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Str(
        s.to_string(),
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

fn goto(bb: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Goto {
        target: mir::BasicBlock(bb),
    }
}

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
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

fn call_diverging(func: mir::Operand, args: Vec<mir::Operand>, dest: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func,
        args,
        destination: place(dest),
        target: None,
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

// DefIds of the std items the tests name (krate 0, arbitrary indices).
const ARGUMENTS_TY: mir::DefId = mir::DefId::new(0, 200);
const FROM_STR: mir::DefId = mir::DefId::new(0, 201);
const PANIC_FMT: mir::DefId = mir::DefId::new(0, 202);
const PANICKING_PANIC: mir::DefId = mir::DefId::new(0, 203);
const ASSERT_FAILED: mir::DefId = mir::DefId::new(0, 204);
const ASSERT_KIND_TY: mir::DefId = mir::DefId::new(0, 205);
const PARTIAL_EQ: mir::DefId = mir::DefId::new(0, 206);
const PARTIAL_NE: mir::DefId = mir::DefId::new(0, 207);
const ORD_CMP: mir::DefId = mir::DefId::new(0, 208);
const PARTIAL_LE: mir::DefId = mir::DefId::new(0, 209);
const VEC_TY: mir::DefId = mir::DefId::new(0, 210);
const VEC_NEW: mir::DefId = mir::DefId::new(0, 211);
const VEC_PUSH: mir::DefId = mir::DefId::new(0, 212);
const VEC_WITH_CAPACITY: mir::DefId = mir::DefId::new(0, 239);
const VEC_RESERVE: mir::DefId = mir::DefId::new(0, 240);
const VEC_CAPACITY: mir::DefId = mir::DefId::new(0, 241);
const VEC_FROM_ELEM: mir::DefId = mir::DefId::new(0, 242);
const BOX_NEW_UNINIT: mir::DefId = mir::DefId::new(0, 243);
const BOX_ASSUME_INIT_INTO_VEC: mir::DefId = mir::DefId::new(0, 244);
const VEC_LEN: mir::DefId = mir::DefId::new(0, 213);
const INDEX: mir::DefId = mir::DefId::new(0, 214);
const INDEX_MUT: mir::DefId = mir::DefId::new(0, 215);
const BOX_TY: mir::DefId = mir::DefId::new(0, 216);
const BOX_NEW: mir::DefId = mir::DefId::new(0, 217);
const DEREF: mir::DefId = mir::DefId::new(0, 218);
const TO_STRING: mir::DefId = mir::DefId::new(0, 219);
const CLONE: mir::DefId = mir::DefId::new(0, 220);
const DEFAULT: mir::DefId = mir::DefId::new(0, 221);
const STRING_TY: mir::DefId = mir::DefId::new(0, 222);
const CELL_TY: mir::DefId = mir::DefId::new(0, 223);
const CELL_NEW: mir::DefId = mir::DefId::new(0, 224);
const CELL_GET: mir::DefId = mir::DefId::new(0, 225);
const CELL_SET: mir::DefId = mir::DefId::new(0, 226);
const OPTION_TY: mir::DefId = mir::DefId::new(0, 227);
const OPTION_UNWRAP: mir::DefId = mir::DefId::new(0, 228);
const RESULT_TY: mir::DefId = mir::DefId::new(0, 229);
const RESULT_UNWRAP: mir::DefId = mir::DefId::new(0, 230);
const RESULT_IS_ERR: mir::DefId = mir::DefId::new(0, 231);
const SLICE_ITER: mir::DefId = mir::DefId::new(0, 232);
const ITER_TY: mir::DefId = mir::DefId::new(0, 233);
const INTO_ITER: mir::DefId = mir::DefId::new(0, 234);
const ITER_NEXT: mir::DefId = mir::DefId::new(0, 235);
const SIZE_OF: mir::DefId = mir::DefId::new(0, 236);
const MEM_SWAP: mir::DefId = mir::DefId::new(0, 237);
const MEM_DROP: mir::DefId = mir::DefId::new(0, 238);

fn std_names() -> HashMap<mir::DefId, String> {
    HashMap::from([
        (ARGUMENTS_TY, "std::fmt::Arguments".to_string()),
        (FROM_STR, "std::fmt::Arguments::<'a>::from_str".to_string()),
        (PANIC_FMT, "std::rt::panic_fmt".to_string()),
        (PANICKING_PANIC, "core::panicking::panic".to_string()),
        (ASSERT_FAILED, "core::panicking::assert_failed".to_string()),
        (ASSERT_KIND_TY, "core::panicking::AssertKind".to_string()),
        (PARTIAL_EQ, "std::cmp::PartialEq::eq".to_string()),
        (PARTIAL_NE, "std::cmp::PartialEq::ne".to_string()),
        (ORD_CMP, "std::cmp::Ord::cmp".to_string()),
        (PARTIAL_LE, "std::cmp::PartialOrd::le".to_string()),
        (VEC_TY, "std::vec::Vec".to_string()),
        (VEC_NEW, "std::vec::Vec::<T>::new".to_string()),
        (VEC_PUSH, "std::vec::Vec::<T, A>::push".to_string()),
        (VEC_WITH_CAPACITY, "std::vec::Vec::<T>::with_capacity".to_string()),
        (VEC_RESERVE, "std::vec::Vec::<T, A>::reserve".to_string()),
        (VEC_CAPACITY, "std::vec::Vec::<T, A>::capacity".to_string()),
        (VEC_FROM_ELEM, "std::vec::from_elem".to_string()),
        (BOX_NEW_UNINIT, "std::boxed::Box::<T>::new_uninit".to_string()),
        (
            BOX_ASSUME_INIT_INTO_VEC,
            "std::boxed::box_assume_init_into_vec_unsafe".to_string(),
        ),
        (VEC_LEN, "std::vec::Vec::<T, A>::len".to_string()),
        (INDEX, "std::ops::Index::index".to_string()),
        (INDEX_MUT, "std::ops::IndexMut::index_mut".to_string()),
        (BOX_TY, "std::boxed::Box".to_string()),
        (BOX_NEW, "std::boxed::Box::<T>::new".to_string()),
        (DEREF, "std::ops::Deref::deref".to_string()),
        (TO_STRING, "std::string::ToString::to_string".to_string()),
        (CLONE, "std::clone::Clone::clone".to_string()),
        (DEFAULT, "std::default::Default::default".to_string()),
        (STRING_TY, "std::string::String".to_string()),
        (CELL_TY, "std::cell::Cell".to_string()),
        (CELL_NEW, "std::cell::Cell::<T>::new".to_string()),
        (CELL_GET, "std::cell::Cell::<T>::get".to_string()),
        (CELL_SET, "std::cell::Cell::<T>::set".to_string()),
        (OPTION_TY, "std::option::Option".to_string()),
        (OPTION_UNWRAP, "std::option::Option::<T>::unwrap".to_string()),
        (RESULT_TY, "std::result::Result".to_string()),
        (RESULT_UNWRAP, "std::result::Result::<T, E>::unwrap".to_string()),
        (RESULT_IS_ERR, "std::result::Result::<T, E>::is_err".to_string()),
        (SLICE_ITER, "core::slice::<impl [T]>::iter".to_string()),
        (ITER_TY, "std::slice::Iter".to_string()),
        (INTO_ITER, "std::iter::IntoIterator::into_iter".to_string()),
        (ITER_NEXT, "std::iter::Iterator::next".to_string()),
        (SIZE_OF, "std::mem::size_of".to_string()),
        (MEM_SWAP, "std::mem::swap".to_string()),
        (MEM_DROP, "std::mem::drop".to_string()),
    ])
}

fn arguments_ty() -> mir::Ty {
    mir::Ty::Adt(ARGUMENTS_TY, vec![])
}

fn vec_i32_ty() -> mir::Ty {
    mir::Ty::Adt(VEC_TY, vec![i32_ty()])
}

fn program(body: mir::Body) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.add("main", body);
    prog.fn_names = std_names();
    prog
}

fn expect_panic(result: Result<Value, MirRunError>) -> String {
    match result {
        Err(MirRunError::Interp(InterpError::Panic { message })) => message,
        other => panic!("expected a guest panic, got {other:?}"),
    }
}

// ----- the registry -----------------------------------------------------------

#[test]
fn registry_covers_the_modelled_callees() {
    let cases: &[(&str, Builtin)] = &[
        ("core::panicking::panic", Builtin::PanicStr),
        ("std::rt::panic_fmt", Builtin::PanicFmt),
        ("core::panicking::panic_fmt", Builtin::PanicFmt),
        ("core::panicking::assert_failed", Builtin::AssertFailed),
        ("std::cmp::PartialEq::eq", Builtin::PartialEqEq),
        ("std::cmp::PartialEq::ne", Builtin::PartialEqNe),
        ("std::cmp::PartialOrd::le", Builtin::PartialOrdRel(RelOp::Le)),
        ("std::cmp::Ord::cmp", Builtin::OrdCmp),
        ("std::clone::Clone::clone", Builtin::CloneClone),
        ("std::default::Default::default", Builtin::DefaultDefault),
        ("std::string::ToString::to_string", Builtin::ToStringToString),
        ("std::boxed::Box::<T>::new", Builtin::BoxNew),
        ("std::vec::Vec::<T>::new", Builtin::VecNew),
        ("std::vec::Vec::<T, A>::push", Builtin::VecPush),
        ("std::vec::Vec::<T, A>::len", Builtin::VecLen),
        ("core::slice::<impl [T]>::iter", Builtin::SliceIter),
        ("std::ops::Index::index", Builtin::IndexIndex),
        ("std::ops::IndexMut::index_mut", Builtin::IndexMut),
        ("std::ops::DerefMut::deref_mut", Builtin::DerefMut),
        ("std::cell::Cell::<T>::new", Builtin::CellNew),
        ("std::cell::Cell::<T>::get", Builtin::CellGet),
        ("std::cell::Cell::<T>::set", Builtin::CellSet),
        ("std::option::Option::<T>::unwrap", Builtin::OptionUnwrap),
        ("core::option::Option::<T>::unwrap", Builtin::OptionUnwrap),
        ("std::result::Result::<T, E>::unwrap", Builtin::ResultUnwrap),
        ("std::result::Result::<T, E>::is_err", Builtin::ResultIsErr),
        ("std::mem::drop", Builtin::MemDrop),
        ("std::mem::swap", Builtin::MemSwap),
        ("std::mem::size_of", Builtin::MemSizeOf),
        ("std::mem::ManuallyDrop::<T>::new", Builtin::ManuallyDropNew),
        ("std::string::String::new", Builtin::StringNew),
        ("std::string::String::len", Builtin::StringLen),
        ("std::string::String::is_empty", Builtin::StringIsEmpty),
        ("std::string::String::capacity", Builtin::StringCapacity),
        ("std::string::String::as_str", Builtin::StringAsStr),
        ("std::string::String::as_bytes", Builtin::StringAsBytes),
        ("std::string::String::into_bytes", Builtin::StringIntoBytes),
        ("std::string::String::push_str", Builtin::StringPushStr),
        ("std::string::String::push", Builtin::StringPush),
        ("core::str::<impl str>::len", Builtin::StrLen),
        ("core::str::<impl str>::is_empty", Builtin::StrIsEmpty),
        ("core::str::<impl str>::as_bytes", Builtin::StrAsBytes),
        ("core::str::<impl str>::chars", Builtin::StrChars),
        ("core::str::<impl str>::bytes", Builtin::StrBytes),
        ("core::str::<impl str>::trim", Builtin::StrTrim(TrimSide::Both)),
        (
            "core::str::<impl str>::trim_start",
            Builtin::StrTrim(TrimSide::Start),
        ),
        (
            "core::str::<impl str>::trim_end",
            Builtin::StrTrim(TrimSide::End),
        ),
        ("std::borrow::ToOwned::to_owned", Builtin::ToOwned),
        ("std::convert::Into::into", Builtin::IntoInto),
    ];
    for (name, expected) in cases {
        assert_eq!(
            lookup(&canonical_name(name)),
            Some(*expected),
            "lookup for {name}"
        );
    }
    // Callees that stay unsupported: filesystem, unwinding through
    // closures, channels — no name-only registrations.
    for name in [
        "std::fs::File::create",
        "std::panic::catch_unwind",
        "std::sync::mpsc::channel",
        "std::io::stdout",
        "std::env::args",
    ] {
        assert_eq!(lookup(&canonical_name(name)), None, "{name} stays unsupported");
    }
}

// ----- the panic path -----------------------------------------------------------

#[test]
fn panic_fmt_raises_a_guest_panic_with_the_message() {
    // fn main() {
    //   _2 = Arguments::from_str(const "kaboom");
    //   _1 = panic_fmt(move _2) -> !;
    // }
    let body = mir::Body::new(
        decls(&[mir::Ty::unit(), mir::Ty::unit(), arguments_ty()]),
        0,
        vec![
            block(
                vec![],
                call(fndef(FROM_STR, vec![]), vec![const_str("kaboom")], 2, 1),
            ),
            block(vec![], call_diverging(fndef(PANIC_FMT, vec![]), vec![move_(2)], 1)),
        ],
    );
    let message = expect_panic(interpret_mir(&program(body), "main", vec![]));
    assert_eq!(message, "kaboom");
}

#[test]
fn panicking_panic_raises_a_guest_panic_with_the_str() {
    // fn main() { _1 = core::panicking::panic(const "explicit blowup") -> !; }
    let body = mir::Body::new(
        decls(&[mir::Ty::unit(), mir::Ty::unit()]),
        0,
        vec![block(
            vec![],
            call_diverging(
                fndef(PANICKING_PANIC, vec![]),
                vec![const_str("explicit blowup")],
                1,
            ),
        )],
    );
    let message = expect_panic(interpret_mir(&program(body), "main", vec![]));
    assert_eq!(message, "explicit blowup");
}

#[test]
fn assert_failed_raises_a_guest_panic_naming_the_comparison() {
    // fn main() {
    //   _1 = 3; _2 = 4;
    //   _3 = AssertKind::Eq; _4 = &_1; _5 = &_2; _6 = Option::None;
    //   _7 = assert_failed(move _3, copy _4, copy _5, move _6) -> !;
    // }
    let option_args_ty = mir::Ty::Adt(OPTION_TY, vec![arguments_ty()]);
    let body = mir::Body::new(
        decls(&[
            mir::Ty::unit(),
            i32_ty(),
            i32_ty(),
            mir::Ty::Adt(ASSERT_KIND_TY, vec![]),
            ref_ty(i32_ty(), false),
            ref_ty(i32_ty(), false),
            option_args_ty.clone(),
            mir::Ty::unit(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(3))),
                    assign(place(2), mir::Rvalue::Use(const_i32(4))),
                    assign(
                        place(3),
                        adt_aggregate(ASSERT_KIND_TY, 0, vec![], vec![]),
                    ),
                    assign(place(4), ref_rvalue(1, false)),
                    assign(place(5), ref_rvalue(2, false)),
                    assign(
                        place(6),
                        adt_aggregate(OPTION_TY, 0, vec![arguments_ty()], vec![]),
                    ),
                ],
                call_diverging(
                    fndef(ASSERT_FAILED, vec![i32_ty(), i32_ty()]),
                    vec![move_(3), copy(4), copy(5), move_(6)],
                    7,
                ),
            ),
        ],
    );
    let message = expect_panic(interpret_mir(&program(body), "main", vec![]));
    assert!(
        message.contains("assertion `left == right` failed"),
        "message: {message}"
    );
    assert!(message.contains("left: 3"), "message: {message}");
    assert!(message.contains("right: 4"), "message: {message}");
}

// ----- semantic comparisons -----------------------------------------------------

#[test]
fn partial_eq_compares_strings_and_ints_by_value() {
    // fn main() -> bool {
    //   _1 = "hello"; _2 = "hello"; _3 = &_1; _4 = &_2;
    //   _0 = PartialEq::eq(move _3, move _4); return _0;
    // }
    let body = mir::Body::new(
        decls(&[
            bool_ty(),
            mir::Ty::Str,
            mir::Ty::Str,
            ref_ty(mir::Ty::Str, false),
            ref_ty(mir::Ty::Str, false),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_str("hello"))),
                    assign(place(2), mir::Rvalue::Use(const_str("hello"))),
                    assign(place(3), ref_rvalue(1, false)),
                    assign(place(4), ref_rvalue(2, false)),
                ],
                call(
                    fndef(PARTIAL_EQ, vec![mir::Ty::Str]),
                    vec![move_(3), move_(4)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Bool(true));

    // ne on different ints.
    let body = mir::Body::new(
        decls(&[
            bool_ty(),
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty(), false),
            ref_ty(i32_ty(), false),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(5))),
                    assign(place(2), mir::Rvalue::Use(const_i32(6))),
                    assign(place(3), ref_rvalue(1, false)),
                    assign(place(4), ref_rvalue(2, false)),
                ],
                call(
                    fndef(PARTIAL_NE, vec![i32_ty()]),
                    vec![move_(3), move_(4)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Bool(true));
}

#[test]
fn ord_cmp_and_partial_ord_on_strings() {
    // Ord::cmp("abc", "abd") == Ordering::Less (variant 0).
    let body = mir::Body::new(
        decls(&[
            mir::Ty::Adt(mir::DefId::new(0, 240), vec![]),
            mir::Ty::Str,
            mir::Ty::Str,
            ref_ty(mir::Ty::Str, false),
            ref_ty(mir::Ty::Str, false),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_str("abc"))),
                    assign(place(2), mir::Rvalue::Use(const_str("abd"))),
                    assign(place(3), ref_rvalue(1, false)),
                    assign(place(4), ref_rvalue(2, false)),
                ],
                call(
                    fndef(ORD_CMP, vec![mir::Ty::Str]),
                    vec![move_(3), move_(4)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let mut names = std_names();
    names.insert(mir::DefId::new(0, 240), "std::cmp::Ordering".to_string());
    let mut prog = MirProgram::new();
    prog.add("main", body);
    prog.fn_names = names;
    let value = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(
        value,
        Value::Enum {
            variant: 0,
            fields: vec![]
        }
    );
}

// ----- Vec ----------------------------------------------------------------------

#[test]
fn vec_new_push_len_and_index() {
    // fn main() -> i32 {
    //   _1 = Vec::new();
    //   Vec::push(&_1, 10); Vec::push(&_1, 20);
    //   _5 = Vec::len(&_1);
    //   _7 = Index::index(&_1, 0); _9 = Index::index(&_1, 1);
    //   return *_7 + *_9 + (_5 as i32);   // 10 + 20 + 2 = 32
    // }
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                            // _0
            vec_i32_ty(),                        // _1 vec
            ref_ty(vec_i32_ty(), true),          // _2 &mut vec
            mir::Ty::unit(),                     // _3 push result
            ref_ty(vec_i32_ty(), true),          // _4 &mut vec
            mir::Ty::unit(),                     // _5 push result
            ref_ty(vec_i32_ty(), false),         // _6 &vec
            usize_ty(),                          // _7 len
            ref_ty(vec_i32_ty(), false),         // _8 &vec
            ref_ty(i32_ty(), false),             // _9 &elem
            i32_ty(),                            // _10 elem 0
            ref_ty(vec_i32_ty(), false),         // _11 &vec
            ref_ty(i32_ty(), false),             // _12 &elem
            i32_ty(),                            // _13 elem 1
            i32_ty(),                            // _14 len as i32
            i32_ty(),                            // _15 sum
        ]),
        0,
        vec![
            block(vec![], call(fndef(VEC_NEW, vec![i32_ty()]), vec![], 1, 1)),
            block(
                vec![assign(place(2), ref_rvalue(1, true))],
                call(
                    fndef(VEC_PUSH, vec![i32_ty()]),
                    vec![move_(2), const_i32(10)],
                    3,
                    2,
                ),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, true))],
                call(
                    fndef(VEC_PUSH, vec![i32_ty()]),
                    vec![move_(4), const_i32(20)],
                    5,
                    3,
                ),
            ),
            block(
                vec![assign(place(6), ref_rvalue(1, false))],
                call(fndef(VEC_LEN, vec![i32_ty()]), vec![move_(6)], 7, 4),
            ),
            block(
                vec![assign(place(8), ref_rvalue(1, false))],
                call(
                    fndef(INDEX, vec![vec_i32_ty(), usize_ty()]),
                    vec![move_(8), const_usize(0)],
                    9,
                    5,
                ),
            ),
            block(
                vec![
                    assign(place(10), mir::Rvalue::Use(mir::Operand::Copy(deref_place(9)))),
                    assign(place(11), ref_rvalue(1, false)),
                ],
                call(
                    fndef(INDEX, vec![vec_i32_ty(), usize_ty()]),
                    vec![move_(11), const_usize(1)],
                    12,
                    6,
                ),
            ),
            block(
                vec![
                    assign(place(13), mir::Rvalue::Use(mir::Operand::Copy(deref_place(12)))),
                    assign(
                        place(14),
                        mir::Rvalue::Cast(
                            mir::CastKind::IntToInt,
                            copy(7),
                            i32_ty(),
                        ),
                    ),
                    assign(
                        place(15),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((copy(10), copy(13))),
                        ),
                    ),
                    assign(
                        place(0),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((copy(15), copy(14))),
                        ),
                    ),
                ],
                ret(),
            ),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(32));
}

#[test]
fn vec_index_out_of_bounds_is_a_guest_panic() {
    // fn main() { _1 = Vec::new(); _3 = Index::index(&_1, 5); }
    let body = mir::Body::new(
        decls(&[
            mir::Ty::unit(),
            vec_i32_ty(),
            ref_ty(vec_i32_ty(), false),
            ref_ty(i32_ty(), false),
        ]),
        0,
        vec![
            block(vec![], call(fndef(VEC_NEW, vec![i32_ty()]), vec![], 1, 1)),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(
                    fndef(INDEX, vec![vec_i32_ty(), usize_ty()]),
                    vec![move_(2), const_usize(5)],
                    3,
                    2,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let message = expect_panic(interpret_mir(&program(body), "main", vec![]));
    assert_eq!(
        message,
        "index out of bounds: the len is 0 but the index is 5"
    );
}

#[test]
fn vec_equality_is_element_wise() {
    // Two vectors with the same elements compare equal via the
    // derived `PartialEq` of `Vec`.
    fn build(second: i32) -> mir::Body {
        mir::Body::new(
            decls(&[
                bool_ty(),
                vec_i32_ty(),
                vec_i32_ty(),
                ref_ty(vec_i32_ty(), true),
                mir::Ty::unit(),
                ref_ty(vec_i32_ty(), true),
                mir::Ty::unit(),
                ref_ty(vec_i32_ty(), true),
                mir::Ty::unit(),
                ref_ty(vec_i32_ty(), true),
                mir::Ty::unit(),
                ref_ty(vec_i32_ty(), false),
                ref_ty(vec_i32_ty(), false),
            ]),
            0,
            vec![
                block(vec![], call(fndef(VEC_NEW, vec![i32_ty()]), vec![], 1, 1)),
                block(vec![], call(fndef(VEC_NEW, vec![i32_ty()]), vec![], 2, 2)),
                block(
                    vec![assign(place(3), ref_rvalue(1, true))],
                    call(fndef(VEC_PUSH, vec![i32_ty()]), vec![move_(3), const_i32(10)], 4, 3),
                ),
                block(
                    vec![assign(place(5), ref_rvalue(1, true))],
                    call(fndef(VEC_PUSH, vec![i32_ty()]), vec![move_(5), const_i32(20)], 6, 4),
                ),
                block(
                    vec![assign(place(7), ref_rvalue(2, true))],
                    call(fndef(VEC_PUSH, vec![i32_ty()]), vec![move_(7), const_i32(10)], 8, 5),
                ),
                block(
                    vec![assign(place(9), ref_rvalue(2, true))],
                    call(
                        fndef(VEC_PUSH, vec![i32_ty()]),
                        vec![move_(9), const_i32(second)],
                        10,
                        6,
                    ),
                ),
                block(
                    vec![
                        assign(place(11), ref_rvalue(1, false)),
                        assign(place(12), ref_rvalue(2, false)),
                    ],
                    call(
                        fndef(PARTIAL_EQ, vec![vec_i32_ty()]),
                        vec![move_(11), move_(12)],
                        0,
                        7,
                    ),
                ),
                block(vec![], ret()),
            ],
        )
    }
    let value = interpret_mir(&program(build(20)), "main", vec![]).expect("run");
    assert_eq!(value, Value::Bool(true));
    let value = interpret_mir(&program(build(99)), "main", vec![]).expect("run");
    assert_eq!(value, Value::Bool(false));
}

// ----- Box ------------------------------------------------------------------------

#[test]
fn box_new_then_deref_reads_the_pointee() {
    // fn main() -> i32 {
    //   _1 = Box::new(41);
    //   _2 = Deref::deref(&_1);
    //   return *_2;
    // }
    let box_i32 = mir::Ty::Adt(BOX_TY, vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            box_i32.clone(),
            ref_ty(i32_ty(), false),
            ref_ty(box_i32.clone(), false),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(BOX_NEW, vec![i32_ty()]), vec![const_i32(41)], 1, 1),
            ),
            block(
                vec![assign(place(3), ref_rvalue(1, false))],
                call(fndef(DEREF, vec![box_i32]), vec![move_(3)], 2, 2),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(deref_place(2))),
                )],
                ret(),
            ),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(41));
}

// ----- ToString / Clone / Default -------------------------------------------------

#[test]
fn to_string_for_str_and_int() {
    // "abc".to_string()
    let body = mir::Body::new(
        decls(&[
            mir::Ty::Adt(STRING_TY, vec![]),
            ref_ty(mir::Ty::Str, false),
            mir::Ty::Str,
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(2), mir::Rvalue::Use(const_str("abc"))),
                    assign(place(1), ref_rvalue(2, false)),
                ],
                call(
                    fndef(TO_STRING, vec![mir::Ty::Str]),
                    vec![move_(1)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Str("abc".to_string()));

    // 42.to_string()
    let body = mir::Body::new(
        decls(&[
            mir::Ty::Adt(STRING_TY, vec![]),
            ref_ty(i32_ty(), false),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(2), mir::Rvalue::Use(const_i32(42))),
                    assign(place(1), ref_rvalue(2, false)),
                ],
                call(fndef(TO_STRING, vec![i32_ty()]), vec![move_(1)], 0, 1),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Str("42".to_string()));
}

#[test]
fn clone_copies_the_pointee_and_default_builds_zero_values() {
    // fn main() -> String { _1 = "data"; _2 = &_1; _0 = Clone::clone(move _2); }
    let string_ty = mir::Ty::Adt(STRING_TY, vec![]);
    let body = mir::Body::new(
        decls(&[string_ty.clone(), string_ty.clone(), ref_ty(string_ty.clone(), false)]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_str("data"))),
                    assign(place(2), ref_rvalue(1, false)),
                ],
                call(fndef(CLONE, vec![string_ty]), vec![move_(2)], 0, 1),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Str("data".to_string()));

    // Default::default() for a few types.
    let check = |self_ty: mir::Ty, dest_ty: mir::Ty, expected: Value| {
        let body = mir::Body::new(
            decls(&[dest_ty]),
            0,
            vec![
                block(vec![], call(fndef(DEFAULT, vec![self_ty]), vec![], 0, 1)),
                block(vec![], ret()),
            ],
        );
        let value = interpret_mir(&program(body), "main", vec![]).expect("run");
        assert_eq!(value, expected);
    };
    check(i32_ty(), i32_ty(), Value::Int(0));
    check(bool_ty(), bool_ty(), Value::Bool(false));
    check(
        mir::Ty::Adt(STRING_TY, vec![]),
        mir::Ty::Adt(STRING_TY, vec![]),
        Value::Str(String::new()),
    );
    check(
        mir::Ty::Adt(OPTION_TY, vec![i32_ty()]),
        mir::Ty::Adt(OPTION_TY, vec![i32_ty()]),
        Value::Enum {
            variant: 0,
            fields: vec![],
        },
    );
    check(vec_i32_ty(), vec_i32_ty(), Value::Enum {
        variant: 0,
        fields: vec![],
    });
    check(
        mir::Ty::Tuple(vec![i32_ty(), bool_ty()]),
        mir::Ty::Tuple(vec![i32_ty(), bool_ty()]),
        Value::Tuple(vec![Value::Int(0), Value::Bool(false)]),
    );
}

// ----- Cell -----------------------------------------------------------------------

#[test]
fn cell_new_set_get_roundtrip() {
    // fn main() -> i32 {
    //   _1 = Cell::new(5);
    //   Cell::set(&_1, 9);
    //   return Cell::get(&_1);
    // }
    let cell_ty = mir::Ty::Adt(CELL_TY, vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            cell_ty.clone(),
            ref_ty(cell_ty.clone(), false),
            mir::Ty::unit(),
            ref_ty(cell_ty.clone(), false),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(CELL_NEW, vec![i32_ty()]), vec![const_i32(5)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(
                    fndef(CELL_SET, vec![i32_ty()]),
                    vec![move_(2), const_i32(9)],
                    3,
                    2,
                ),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, false))],
                call(fndef(CELL_GET, vec![i32_ty()]), vec![move_(4)], 0, 3),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(9));
}

// ----- Option / Result --------------------------------------------------------------

#[test]
fn option_unwrap_some_and_none() {
    // Some(42).unwrap() — the payload goes through a typed local, as
    // in real MIR, so the constant takes the `i32` type.
    let option_ty = mir::Ty::Adt(OPTION_TY, vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[i32_ty(), option_ty.clone(), i32_ty()]),
        0,
        vec![
            block(
                vec![
                    assign(place(2), mir::Rvalue::Use(const_i32(42))),
                    assign(
                        place(1),
                        adt_aggregate(OPTION_TY, 1, vec![i32_ty()], vec![move_(2)]),
                    ),
                ],
                call(
                    fndef(OPTION_UNWRAP, vec![i32_ty()]),
                    vec![move_(1)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(42));

    // None.unwrap() panics.
    let body = mir::Body::new(
        decls(&[i32_ty(), option_ty]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    adt_aggregate(OPTION_TY, 0, vec![i32_ty()], vec![]),
                )],
                call(
                    fndef(OPTION_UNWRAP, vec![i32_ty()]),
                    vec![move_(1)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let message = expect_panic(interpret_mir(&program(body), "main", vec![]));
    assert_eq!(message, "called `Option::unwrap()` on a `None` value");
}

#[test]
fn result_unwrap_err_panics_and_is_err_reads_the_variant() {
    // Err("bad").unwrap() panics; Ok(1).is_err() is false.
    let result_ty = mir::Ty::Adt(RESULT_TY, vec![i32_ty(), mir::Ty::Adt(STRING_TY, vec![])]);
    let body = mir::Body::new(
        decls(&[i32_ty(), result_ty.clone()]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    adt_aggregate(
                        RESULT_TY,
                        1,
                        vec![i32_ty(), mir::Ty::Adt(STRING_TY, vec![])],
                        vec![const_str("bad")],
                    ),
                )],
                call(
                    fndef(RESULT_UNWRAP, vec![i32_ty(), mir::Ty::Adt(STRING_TY, vec![])]),
                    vec![move_(1)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let message = expect_panic(interpret_mir(&program(body), "main", vec![]));
    assert!(
        message.contains("called `Result::unwrap()` on an `Err` value"),
        "message: {message}"
    );

    let body = mir::Body::new(
        decls(&[bool_ty(), result_ty.clone(), ref_ty(result_ty, false)]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        adt_aggregate(
                            RESULT_TY,
                            0,
                            vec![i32_ty(), mir::Ty::Adt(STRING_TY, vec![])],
                            vec![const_i32(1)],
                        ),
                    ),
                    assign(place(2), ref_rvalue(1, false)),
                ],
                call(
                    fndef(RESULT_IS_ERR, vec![i32_ty(), mir::Ty::Adt(STRING_TY, vec![])]),
                    vec![move_(2)],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Bool(false));
}

// ----- slice iterators ---------------------------------------------------------------

#[test]
fn slice_iter_next_loop_sums() {
    // fn main() -> i32 {
    //   _1 = [1, 2, 3];
    //   _3 = core::slice::iter(&_1 as &[i32]);
    //   _4 = IntoIterator::into_iter(move _3);
    //   loop { _6 = Iterator::next(&_4); match discriminant { 0 => break, _ => sum += payload } }
    //   return sum;   // 6
    // }
    let iter_ty = mir::Ty::Adt(ITER_TY, vec![i32_ty()]);
    let option_ty = mir::Ty::Adt(OPTION_TY, vec![i32_ty()]);
    let array_ty = mir::Ty::Array(Box::new(i32_ty()), mir::ArrayLen::Known(3));
    let slice_ref_ty = ref_ty(mir::Ty::Slice(Box::new(i32_ty())), false);
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                       // _0 return
            array_ty.clone(),               // _1 array
            ref_ty(array_ty.clone(), false), // _2 &[i32; 3]
            slice_ref_ty.clone(),           // _3 &[i32]
            iter_ty.clone(),                // _4 Iter
            iter_ty.clone(),                // _5 iterator state
            ref_ty(iter_ty.clone(), true),  // _6 &mut iter
            option_ty,                      // _7 Option<i32>
            mir::Ty::Int(mir::IntTy::Isize), // _8 discriminant
            i32_ty(),                       // _9 payload
            i32_ty(),                       // _10 sum
        ]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Array(i32_ty())),
                            vec![const_i32(1), const_i32(2), const_i32(3)],
                        ),
                    ),
                    assign(place(2), ref_rvalue(1, false)),
                    assign(
                        place(3),
                        mir::Rvalue::Cast(
                            mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
                            copy(2),
                            slice_ref_ty,
                        ),
                    ),
                ],
                call(
                    fndef(SLICE_ITER, vec![mir::Ty::Slice(Box::new(i32_ty()))]),
                    vec![move_(3)],
                    4,
                    1,
                ),
            ),
            block(
                vec![assign(place(10), mir::Rvalue::Use(const_i32(0)))],
                call(fndef(INTO_ITER, vec![iter_ty.clone()]), vec![move_(4)], 5, 2),
            ),
            block(
                vec![assign(place(6), ref_rvalue(5, true))],
                call(fndef(ITER_NEXT, vec![iter_ty.clone()]), vec![move_(6)], 7, 3),
            ),
            block(
                vec![assign(place(8), mir::Rvalue::Discriminant(place(7)))],
                mir::TerminatorKind::SwitchInt {
                    discr: copy(8),
                    targets: mir::SwitchTargets::new(
                        vec![0],
                        vec![mir::BasicBlock(5)],
                        mir::BasicBlock(4),
                    ),
                },
            ),
            block(
                vec![
                    assign(
                        place(9),
                        mir::Rvalue::Use(mir::Operand::Copy(
                            place(7).downcast(None, 1).field(0, i32_ty()),
                        )),
                    ),
                    assign(
                        place(10),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((copy(10), copy(9))),
                        ),
                    ),
                ],
                goto(2),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(10)))], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(6));
}

// ----- std::mem -----------------------------------------------------------------------

#[test]
fn size_of_swap_and_drop() {
    // size_of::<i32>() == 4 and size_of::<&str>() == 16.
    let body = mir::Body::new(
        decls(&[usize_ty()]),
        0,
        vec![
            block(vec![], call(fndef(SIZE_OF, vec![i32_ty()]), vec![], 0, 1)),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Uint(4));

    let body = mir::Body::new(
        decls(&[usize_ty()]),
        0,
        vec![
            block(
                vec![],
                call(
                    fndef(SIZE_OF, vec![ref_ty(mir::Ty::Str, false)]),
                    vec![],
                    0,
                    1,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Uint(16));

    // swap(&mut a, &mut b) exchanges the locals; drop consumes a value.
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty(), true),
            ref_ty(i32_ty(), true),
            mir::Ty::unit(),
            mir::Ty::unit(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(1))),
                    assign(place(2), mir::Rvalue::Use(const_i32(2))),
                    assign(place(3), ref_rvalue(1, true)),
                    assign(place(4), ref_rvalue(2, true)),
                ],
                call(
                    fndef(MEM_SWAP, vec![i32_ty()]),
                    vec![move_(3), move_(4)],
                    5,
                    1,
                ),
            ),
            block(
                vec![],
                call(fndef(MEM_DROP, vec![i32_ty()]), vec![move_(2)], 6, 2),
            ),
            block(
                vec![assign(place(0), mir::Rvalue::Use(copy(1)))],
                ret(),
            ),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(2));
}

// ----- IndexMut ------------------------------------------------------------------------

#[test]
fn index_mut_writes_an_array_element() {
    // fn main() -> i32 {
    //   _1 = [1, 2, 3];
    //   _3 = IndexMut::index_mut(&mut _1, 1);
    //   *_3 = 99;
    //   return _1[1];
    // }
    let array_ty = mir::Ty::Array(Box::new(i32_ty()), mir::ArrayLen::Known(3));
    let const_indexed = mir::Place {
        local: mir::Local(1),
        projection: vec![mir::PlaceElem::ConstantIndex {
            offset: 1,
            min_length: 3,
            from_end: false,
        }],
    };
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            array_ty.clone(),
            ref_ty(array_ty, true),
            ref_ty(i32_ty(), true),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(
                        place(1),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Array(i32_ty())),
                            vec![const_i32(1), const_i32(2), const_i32(3)],
                        ),
                    ),
                    assign(place(2), ref_rvalue(1, true)),
                ],
                call(
                    fndef(INDEX_MUT, vec![
                        mir::Ty::Array(Box::new(i32_ty()), mir::ArrayLen::Known(3)),
                        usize_ty(),
                    ]),
                    vec![move_(2), const_usize(1)],
                    3,
                    1,
                ),
            ),
            block(
                vec![
                    assign(deref_place(3), mir::Rvalue::Use(const_i32(99))),
                    assign(
                        place(0),
                        mir::Rvalue::Use(mir::Operand::Copy(const_indexed)),
                    ),
                ],
                ret(),
            ),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(99));
}

// ----- Round 2: intrinsics, ops traits, Rc/Arc, zeroed, null (Lane BI) --------
//
// The second builtin census round: callees the combined subset
// measurement (both8) reported as "call to function with no
// converted body", modelled where the value model gives them
// faithful, bounded semantics.

const BLACK_BOX: mir::DefId = mir::DefId::new(0, 300);
const STRING_NEW: mir::DefId = mir::DefId::new(0, 301);
const CTPOP: mir::DefId = mir::DefId::new(0, 303);
const CTLZ: mir::DefId = mir::DefId::new(0, 304);
const CTTZ: mir::DefId = mir::DefId::new(0, 305);
const BITREVERSE: mir::DefId = mir::DefId::new(0, 306);
const BSWAP: mir::DefId = mir::DefId::new(0, 307);
const SAT_ADD: mir::DefId = mir::DefId::new(0, 308);
const SAT_SUB: mir::DefId = mir::DefId::new(0, 309);
const ATOMIC_LOAD: mir::DefId = mir::DefId::new(0, 310);
const ATOMIC_STORE: mir::DefId = mir::DefId::new(0, 311);
const RC_TY: mir::DefId = mir::DefId::new(0, 318);
const RC_NEW: mir::DefId = mir::DefId::new(0, 319);
const ARC_TY: mir::DefId = mir::DefId::new(0, 320);
const ARC_NEW: mir::DefId = mir::DefId::new(0, 321);
const MEM_ZEROED: mir::DefId = mir::DefId::new(0, 322);
const PTR_NULL: mir::DefId = mir::DefId::new(0, 323);

fn names_bi() -> HashMap<mir::DefId, String> {
    let mut names = std_names();
    names.extend([
        (BLACK_BOX, "std::intrinsics::black_box".to_string()),
        (STRING_NEW, "std::string::String::new".to_string()),
        (CTPOP, "std::intrinsics::ctpop".to_string()),
        (CTLZ, "std::intrinsics::ctlz".to_string()),
        (CTTZ, "std::intrinsics::cttz".to_string()),
        (BITREVERSE, "std::intrinsics::bitreverse".to_string()),
        (BSWAP, "std::intrinsics::bswap".to_string()),
        (SAT_ADD, "std::intrinsics::saturating_add".to_string()),
        (SAT_SUB, "std::intrinsics::saturating_sub".to_string()),
        (ATOMIC_LOAD, "std::intrinsics::atomic_load".to_string()),
        (ATOMIC_STORE, "std::intrinsics::atomic_store".to_string()),
        (RC_TY, "std::rc::Rc".to_string()),
        (RC_NEW, "std::rc::Rc::<T>::new".to_string()),
        (ARC_TY, "std::sync::Arc".to_string()),
        (ARC_NEW, "std::sync::Arc::<T>::new".to_string()),
        (MEM_ZEROED, "std::mem::zeroed".to_string()),
        (PTR_NULL, "std::ptr::null".to_string()),
    ]);
    names
}

fn program_bi(body: mir::Body) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.add("main", body);
    prog.fn_names = names_bi();
    prog
}

/// `fn main() -> R { return call f::<G>(args) }` with the call's
/// destination as the return place.
fn unary_call(
    def: mir::DefId,
    generic: mir::Ty,
    arg: mir::Operand,
    ret_ty: mir::Ty,
) -> MirProgram {
    let body = mir::Body::new(
        decls(&[ret_ty]),
        0,
        vec![
            block(vec![], call(fndef(def, vec![generic]), vec![arg], 0, 1)),
            block(vec![], ret()),
        ],
    );
    program_bi(body)
}

fn binary_call(
    def: mir::DefId,
    generic: mir::Ty,
    a: mir::Operand,
    b: mir::Operand,
    ret_ty: mir::Ty,
) -> MirProgram {
    let body = mir::Body::new(
        decls(&[ret_ty]),
        0,
        vec![
            block(vec![], call(fndef(def, vec![generic]), vec![a, b], 0, 1)),
            block(vec![], ret()),
        ],
    );
    program_bi(body)
}

fn u8_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U8)
}
fn u16_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U16)
}
fn u32_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U32)
}
fn i8_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I8)
}

#[test]
fn registry_round2_covers_the_new_callees() {
    let cases: &[(&str, Builtin)] = &[
        ("std::intrinsics::black_box", Builtin::BlackBox),
        ("std::string::String::new", Builtin::StringNew),
        ("std::intrinsics::ctpop", Builtin::IntBitOp(BitOp::CtPop)),
        ("std::intrinsics::ctlz", Builtin::IntBitOp(BitOp::Ctlz)),
        ("std::intrinsics::cttz", Builtin::IntBitOp(BitOp::Cttz)),
        ("std::intrinsics::bitreverse", Builtin::IntBitOp(BitOp::BitReverse)),
        ("std::intrinsics::bswap", Builtin::IntBitOp(BitOp::Bswap)),
        ("std::intrinsics::saturating_add", Builtin::SaturatingArith(SatOp::Add)),
        ("std::intrinsics::saturating_sub", Builtin::SaturatingArith(SatOp::Sub)),
        ("std::intrinsics::atomic_load", Builtin::AtomicLoad),
        ("std::intrinsics::atomic_store", Builtin::AtomicStore),
        ("std::mem::zeroed", Builtin::MemZeroed),
        ("std::ptr::null", Builtin::PtrNull),
        ("std::ptr::null_mut", Builtin::PtrNull),
        ("std::rc::Rc::<T>::new", Builtin::RcNew),
        ("std::sync::Arc::<T>::new", Builtin::ArcNew),
    ];
    for (name, expected) in cases {
        assert_eq!(
            lookup(&canonical_name(name)),
            Some(*expected),
            "lookup for {name}"
        );
    }
    // Deliberately unmodelled: allocation machinery (its callers
    // write through the returned raw pointer, which the value
    // model cannot serve), closure-taking callees (a builtin
    // cannot invoke a guest closure), contract thunks, channels
    // and thread-locals. The `std::ops` arithmetic traits are
    // likewise not registered: their core default bodies are
    // collected and executed as ordinary functions (which also
    // covers reference and `Wrapping` Self types), and a
    // name-only builtin would shadow custom impls.
    for name in [
        "std::ops::Add::add",
        "std::ops::Mul::mul",
        "std::ops::AddAssign::add_assign",
        "std::ops::Neg::neg",
        "std::ops::Not::not",
        "std::alloc::Global::alloc_impl_runtime",
        "std::intrinsics::contract_check_requires",
        "std::iter::Iterator::map",
        "std::iter::Iterator::fold",
        "std::option::Option::<T>::map",
        "std::sync::mpmc::channel",
        "std::thread::LocalKey::<T>::with",
        "std::intrinsics::caller_location",
        "std::intrinsics::va_arg",
    ] {
        assert_eq!(lookup(&canonical_name(name)), None, "{name} stays unsupported");
    }
}

#[test]
fn black_box_passes_the_value_through() {
    let prog = unary_call(BLACK_BOX, i32_ty(), const_i32(42), i32_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Int(42));
    let prog = unary_call(BLACK_BOX, bool_ty(), mir::Operand::constant(
        mir::Constant::from_value(mir::ConstValue::Bool(true)),
    ), bool_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Bool(true));
}

#[test]
fn string_new_builds_an_empty_string() {
    let string_ty = mir::Ty::Adt(STRING_TY, vec![]);
    let body = mir::Body::new(
        decls(&[string_ty]),
        0,
        vec![
            block(vec![], call(fndef(STRING_NEW, vec![]), vec![], 0, 1)),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program_bi(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Str(String::new()));
}

#[test]
fn bit_intrinsics_respect_the_type_width() {
    // ctpop::<u8>(0b1011_0011) == 5
    let prog = unary_call(CTPOP, u8_ty(), const_int(0b1011_0011, 1), u8_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Uint(5));
    // ctpop::<i8>(-1) == 8: the count is over the i8 pattern, not
    // the sign-extended storage.
    let prog = unary_call(CTPOP, i8_ty(), const_int(0xFF, 1), i8_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Int(8));
    // ctlz::<u32>(1) == 31
    let prog = unary_call(CTLZ, u32_ty(), const_int(1, 4), u32_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Uint(31));
    // cttz::<u16>(0) == 16 (the width, as in Rust's definition)
    let prog = unary_call(CTTZ, u16_ty(), const_int(0, 2), u16_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Uint(16));
    // cttz::<u32>(0x8000) == 15
    let prog = unary_call(CTTZ, u32_ty(), const_int(0x8000, 4), u32_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Uint(15));
    // bswap::<u32>(0x0102_0304) == 0x0403_0201
    let prog = unary_call(BSWAP, u32_ty(), const_int(0x0102_0304, 4), u32_ty());
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).expect("run"),
        Value::Uint(0x0403_0201)
    );
    // bitreverse::<u8>(0b1011_0000) == 0b0000_1101
    let prog = unary_call(BITREVERSE, u8_ty(), const_int(0b1011_0000, 1), u8_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Uint(0b0000_1101));
}

#[test]
fn saturating_intrinsics_clamp_to_the_type_range() {
    // saturating_add::<u8>(200, 100) == 255
    let prog = binary_call(SAT_ADD, u8_ty(), const_int(200, 1), const_int(100, 1), u8_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Uint(255));
    // saturating_sub::<i8>(-100, 50) == -128
    let prog = binary_call(
        SAT_SUB,
        i8_ty(),
        const_int((-100i8) as u8 as u128, 1),
        const_int(50, 1),
        i8_ty(),
    );
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Int(-128));
    // saturating_add::<i32>(5, 3) == 8 (no clamping in range)
    let prog = binary_call(SAT_ADD, i32_ty(), const_i32(5), const_i32(3), i32_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Int(8));
    // saturating_sub::<u32>(3, 5) == 0
    let prog = binary_call(SAT_SUB, u32_ty(), const_int(3, 4), const_int(5, 4), u32_ty());
    assert_eq!(interpret_mir(&prog, "main", vec![]).expect("run"), Value::Uint(0));
}

#[test]
fn atomic_load_reads_the_pointee() {
    // fn main() -> i32 { let x = 7; return atomic_load(&x); }
    let body = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), ref_ty(i32_ty(), true)]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(7))),
                    assign(place(2), ref_rvalue(1, true)),
                ],
                call(fndef(ATOMIC_LOAD, vec![i32_ty()]), vec![move_(2)], 0, 1),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program_bi(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(7));
}

#[test]
fn atomic_store_writes_through_and_load_reads_back() {
    // fn main() -> i32 {
    //   let mut x = 7;
    //   atomic_store(&mut x, 41);
    //   return atomic_load(&x);
    // }
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_ty(i32_ty(), true),
            mir::Ty::unit(),
            ref_ty(i32_ty(), true),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(7))),
                    assign(place(2), ref_rvalue(1, true)),
                ],
                call(
                    fndef(ATOMIC_STORE, vec![i32_ty()]),
                    vec![move_(2), const_i32(41)],
                    3,
                    1,
                ),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, true))],
                call(fndef(ATOMIC_LOAD, vec![i32_ty()]), vec![move_(4)], 0, 2),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program_bi(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(41));
}

#[test]
fn rc_new_then_deref_reads_the_value() {
    // fn main() -> i32 {
    //   _1 = Rc::new(42);
    //   _2 = Deref::deref(&_1);
    //   return *_2;
    // }
    let rc_i32 = mir::Ty::Adt(RC_TY, vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            rc_i32.clone(),
            ref_ty(i32_ty(), false),
            ref_ty(rc_i32.clone(), false),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(RC_NEW, vec![i32_ty()]), vec![const_i32(42)], 1, 1),
            ),
            block(
                vec![assign(place(3), ref_rvalue(1, false))],
                call(fndef(DEREF, vec![rc_i32]), vec![move_(3)], 2, 2),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(deref_place(2))),
                )],
                ret(),
            ),
        ],
    );
    let value = interpret_mir(&program_bi(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(42));
}

#[test]
fn arc_new_then_deref_reads_the_value() {
    // fn main() -> i32 {
    //   _1 = Arc::new(7);
    //   _2 = Deref::deref(&_1);
    //   return *_2;
    // }
    let arc_i32 = mir::Ty::Adt(ARC_TY, vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            arc_i32.clone(),
            ref_ty(i32_ty(), false),
            ref_ty(arc_i32.clone(), false),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(ARC_NEW, vec![i32_ty()]), vec![const_i32(7)], 1, 1),
            ),
            block(
                vec![assign(place(3), ref_rvalue(1, false))],
                call(fndef(DEREF, vec![arc_i32]), vec![move_(3)], 2, 2),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(deref_place(2))),
                )],
                ret(),
            ),
        ],
    );
    let value = interpret_mir(&program_bi(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(7));
}

#[test]
fn mem_zeroed_builds_zero_values() {
    // zeroed::<i32>() == 0
    let body = mir::Body::new(
        decls(&[i32_ty()]),
        0,
        vec![
            block(vec![], call(fndef(MEM_ZEROED, vec![i32_ty()]), vec![], 0, 1)),
            block(vec![], ret()),
        ],
    );
    assert_eq!(
        interpret_mir(&program_bi(body), "main", vec![]).expect("run"),
        Value::Int(0)
    );
    // zeroed::<(u8, bool)>() == (0, false)
    let pair_ty = mir::Ty::Tuple(vec![u8_ty(), bool_ty()]);
    let body = mir::Body::new(
        decls(&[pair_ty.clone()]),
        0,
        vec![
            block(vec![], call(fndef(MEM_ZEROED, vec![pair_ty]), vec![], 0, 1)),
            block(vec![], ret()),
        ],
    );
    assert_eq!(
        interpret_mir(&program_bi(body), "main", vec![]).expect("run"),
        Value::Tuple(vec![Value::Uint(0), Value::Bool(false)])
    );
    // zeroed::<f64>() == 0.0
    let f64_ty = mir::Ty::Float(mir::FloatTy::F64);
    let body = mir::Body::new(
        decls(&[f64_ty.clone()]),
        0,
        vec![
            block(vec![], call(fndef(MEM_ZEROED, vec![f64_ty]), vec![], 0, 1)),
            block(vec![], ret()),
        ],
    );
    assert_eq!(
        interpret_mir(&program_bi(body), "main", vec![]).expect("run"),
        Value::Float(0.0)
    );
}

#[test]
fn ptr_null_compares_unequal_to_a_real_reference_and_equal_to_null() {
    // fn main() -> bool {
    //   _1 = 5;
    //   _2 = ptr::null::<i32>();
    //   return PartialEq::eq(&_2, &_1-as-pointer);
    // }
    // The compared values are the pointer itself and a reference
    // to `_1` used as a pointer; equality is structural, and the
    // null word 0 is not a reference.
    let ptr_ty = mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Not);
    let body = mir::Body::new(
        decls(&[
            bool_ty(),
            i32_ty(),
            ptr_ty.clone(),
            ref_ty(ptr_ty.clone(), false),
            ref_ty(ptr_ty.clone(), false),
            ptr_ty.clone(),
        ]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_i32(5)))],
                call(fndef(PTR_NULL, vec![i32_ty()]), vec![], 2, 1),
            ),
            block(
                vec![
                    // _5 = a raw pointer to _1, via a reference
                    // place read (the value a `&_1 as *const i32`
                    // holds in this model is the reference itself).
                    assign(place(5), ref_rvalue(1, false)),
                    assign(place(3), ref_rvalue(2, false)),
                    assign(place(4), ref_rvalue(5, false)),
                ],
                call(
                    fndef(PARTIAL_EQ, vec![ptr_ty.clone()]),
                    vec![move_(3), move_(4)],
                    0,
                    2,
                ),
            ),
            block(vec![], ret()),
        ],
    );
    // _5 is declared as a raw pointer but assigned a reference
    // rvalue; in this value model the two share a representation.
    let value = interpret_mir(&program_bi(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Bool(false));

    // null == null
    let body = mir::Body::new(
        decls(&[
            bool_ty(),
            ptr_ty.clone(),
            ptr_ty.clone(),
            ref_ty(ptr_ty.clone(), false),
            ref_ty(ptr_ty.clone(), false),
        ]),
        0,
        vec![
            block(vec![], call(fndef(PTR_NULL, vec![i32_ty()]), vec![], 1, 1)),
            block(vec![], call(fndef(PTR_NULL, vec![i32_ty()]), vec![], 2, 2)),
            block(
                vec![
                    assign(place(3), ref_rvalue(1, false)),
                    assign(place(4), ref_rvalue(2, false)),
                ],
                call(fndef(PARTIAL_EQ, vec![ptr_ty]), vec![move_(3), move_(4)], 0, 3),
            ),
            block(vec![], ret()),
        ],
    );
    let value = interpret_mir(&program_bi(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Bool(true));
}

// ----- the Vec allocation entry points (Lane BO) ----------------------------

#[test]
fn vec_with_capacity_reserve_then_push_and_len() {
    // fn main() -> i32 {
    //   _1 = Vec::with_capacity(4);
    //   Vec::reserve(&mut _1, 8);
    //   Vec::push(&mut _1, 10); Vec::push(&mut _1, 20);
    //   return Vec::len(&_1) as i32;   // 2
    // }
    // Capacity is not represented in the value model: with_capacity
    // yields the empty vector and reserve is a no-op, so the
    // observable behavior (length, elements) is the native one.
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                    // _0
            vec_i32_ty(),                // _1 vec
            ref_ty(vec_i32_ty(), true),  // _2 &mut vec (reserve)
            mir::Ty::unit(),             // _3 reserve result
            ref_ty(vec_i32_ty(), true),  // _4 &mut vec (push)
            mir::Ty::unit(),             // _5 push result
            ref_ty(vec_i32_ty(), true),  // _6 &mut vec (push)
            mir::Ty::unit(),             // _7 push result
            ref_ty(vec_i32_ty(), false), // _8 &vec
            usize_ty(),                  // _9 len
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(VEC_WITH_CAPACITY, vec![i32_ty()]), vec![const_usize(4)], 1, 1),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, true))],
                call(
                    fndef(VEC_RESERVE, vec![i32_ty()]),
                    vec![move_(2), const_usize(8)],
                    3,
                    2,
                ),
            ),
            block(
                vec![assign(place(4), ref_rvalue(1, true))],
                call(
                    fndef(VEC_PUSH, vec![i32_ty()]),
                    vec![move_(4), const_i32(10)],
                    5,
                    3,
                ),
            ),
            block(
                vec![assign(place(6), ref_rvalue(1, true))],
                call(
                    fndef(VEC_PUSH, vec![i32_ty()]),
                    vec![move_(6), const_i32(20)],
                    7,
                    4,
                ),
            ),
            block(
                vec![assign(place(8), ref_rvalue(1, false))],
                call(fndef(VEC_LEN, vec![i32_ty()]), vec![move_(8)], 9, 5),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Cast(mir::CastKind::IntToInt, copy(9), i32_ty()),
                )],
                ret(),
            ),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(2));
}

#[test]
fn vec_from_elem_repeats_the_element() {
    // fn main() -> i32 {
    //   _1 = std::vec::from_elem(7, 3);
    //   return *Index::index(&_1, 2) + (Vec::len(&_1) as i32);  // 7 + 3
    // }
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                    // _0
            vec_i32_ty(),                // _1 vec
            ref_ty(vec_i32_ty(), false), // _2 &vec
            ref_ty(i32_ty(), false),     // _3 &elem
            i32_ty(),                    // _4 elem
            ref_ty(vec_i32_ty(), false), // _5 &vec
            usize_ty(),                  // _6 len
            i32_ty(),                    // _7 len as i32
        ]),
        0,
        vec![
            block(
                vec![],
                call(
                    fndef(VEC_FROM_ELEM, vec![i32_ty()]),
                    vec![const_i32(7), const_usize(3)],
                    1,
                    1,
                ),
            ),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(
                    fndef(INDEX, vec![vec_i32_ty(), usize_ty()]),
                    vec![move_(2), const_usize(2)],
                    3,
                    2,
                ),
            ),
            block(
                vec![
                    assign(place(4), mir::Rvalue::Use(mir::Operand::Copy(deref_place(3)))),
                    assign(place(5), ref_rvalue(1, false)),
                ],
                call(fndef(VEC_LEN, vec![i32_ty()]), vec![move_(5)], 6, 3),
            ),
            block(
                vec![
                    assign(
                        place(7),
                        mir::Rvalue::Cast(mir::CastKind::IntToInt, copy(6), i32_ty()),
                    ),
                    assign(
                        place(0),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((copy(4), copy(7))),
                        ),
                    ),
                ],
                ret(),
            ),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(10));
}

#[test]
fn vec_capacity_is_an_explicit_error() {
    // fn main() { _1 = Vec::new(); _3 = Vec::capacity(&_1); }
    // Capacity has no representation in the value model, so the
    // call fails with an explicit error naming the function instead
    // of projecting the abstract value as if it had the std layout.
    let body = mir::Body::new(
        decls(&[
            mir::Ty::unit(),             // _0
            vec_i32_ty(),                // _1 vec
            ref_ty(vec_i32_ty(), false), // _2 &vec
            usize_ty(),                  // _3 capacity
        ]),
        0,
        vec![
            block(vec![], call(fndef(VEC_NEW, vec![i32_ty()]), vec![], 1, 1)),
            block(
                vec![assign(place(2), ref_rvalue(1, false))],
                call(fndef(VEC_CAPACITY, vec![i32_ty()]), vec![move_(2)], 3, 2),
            ),
            block(vec![], ret()),
        ],
    );
    let err = interpret_mir(&program(body), "main", vec![])
        .expect_err("Vec::capacity must fail explicitly");
    match err {
        MirRunError::Interp(InterpError::UnsupportedTerminator { kind }) => {
            assert!(
                kind.contains("Vec::capacity is not runtime-supported"),
                "error names the function: {kind}"
            );
        }
        other => panic!("expected an explicit unsupported error, got {other:?}"),
    }
}

fn field_place(idx: u32, fields: &[(u32, mir::Ty)]) -> mir::Place {
    mir::Place {
        local: mir::Local(idx),
        projection: fields
            .iter()
            .map(|(i, ty)| mir::PlaceElem::Field(*i, ty.clone()))
            .collect(),
    }
}

fn deref_field_place(idx: u32, fields: &[(u32, mir::Ty)]) -> mir::Place {
    let mut projection = vec![mir::PlaceElem::Deref];
    projection.extend(
        fields
            .iter()
            .map(|(i, ty)| mir::PlaceElem::Field(*i, ty.clone())),
    );
    mir::Place {
        local: mir::Local(idx),
        projection,
    }
}

#[test]
fn box_new_uninit_write_then_assume_init_into_vec() {
    // The `vec![1, 2, 3]` expansion, shaped as rustc 1.100 emits it:
    // fn main() -> i32 {
    //   _1 = Box::<[i32; 3]>::new_uninit();
    //   _3 = BoxDerefTransmute(copy _1.0);
    //   ((*_3).1: ManuallyDrop<[i32; 3]>).0 = [1, 2, 3];
    //   _4 = box_assume_init_into_vec_unsafe(copy _1);
    //   return *Index::index(&_4, 0) + *Index::index(&_4, 1)
    //        + *Index::index(&_4, 2);   // 6
    // }
    let array_ty = mir::Ty::Array(Box::new(i32_ty()), mir::ArrayLen::Known(3));
    let box_ty = mir::Ty::Adt(BOX_TY, vec![array_ty.clone()]);
    let raw_ptr_ty = mir::Ty::RawPtr(Box::new(array_ty.clone()), mir::Mutability::Mut);
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                    // _0
            box_ty.clone(),              // _1 box
            box_ty.clone(),              // _2 box field copy source type (unused shape)
            raw_ptr_ty,                  // _3 raw ptr
            vec_i32_ty(),                // _4 vec
            ref_ty(vec_i32_ty(), false), // _5 &vec
            ref_ty(i32_ty(), false),     // _6 &elem
            i32_ty(),                    // _7 elem 0
            ref_ty(vec_i32_ty(), false), // _8 &vec
            ref_ty(i32_ty(), false),     // _9 &elem
            i32_ty(),                    // _10 elem 1
            ref_ty(vec_i32_ty(), false), // _11 &vec
            ref_ty(i32_ty(), false),     // _12 &elem
            i32_ty(),                    // _13 elem 2
            i32_ty(),                    // _14 partial sum
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef(BOX_NEW_UNINIT, vec![array_ty.clone()]), vec![], 1, 1),
            ),
            block(
                vec![
                    assign(
                        place(3),
                        mir::Rvalue::Cast(
                            mir::CastKind::BoxDerefTransmute,
                            mir::Operand::Copy(field_place(1, &[(0, box_ty.clone())])),
                            mir::Ty::RawPtr(Box::new(array_ty.clone()), mir::Mutability::Mut),
                        ),
                    ),
                    assign(
                        deref_field_place(3, &[(1, array_ty.clone()), (0, array_ty.clone())]),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Array(i32_ty())),
                            vec![const_i32(1), const_i32(2), const_i32(3)],
                        ),
                    ),
                ],
                call(
                    fndef(BOX_ASSUME_INIT_INTO_VEC, vec![i32_ty(), array_ty.clone()]),
                    vec![copy(1)],
                    4,
                    2,
                ),
            ),
            block(
                vec![assign(place(5), ref_rvalue(4, false))],
                call(
                    fndef(INDEX, vec![vec_i32_ty(), usize_ty()]),
                    vec![move_(5), const_usize(0)],
                    6,
                    3,
                ),
            ),
            block(
                vec![
                    assign(place(7), mir::Rvalue::Use(mir::Operand::Copy(deref_place(6)))),
                    assign(place(8), ref_rvalue(4, false)),
                ],
                call(
                    fndef(INDEX, vec![vec_i32_ty(), usize_ty()]),
                    vec![move_(8), const_usize(1)],
                    9,
                    4,
                ),
            ),
            block(
                vec![
                    assign(place(10), mir::Rvalue::Use(mir::Operand::Copy(deref_place(9)))),
                    assign(place(11), ref_rvalue(4, false)),
                ],
                call(
                    fndef(INDEX, vec![vec_i32_ty(), usize_ty()]),
                    vec![move_(11), const_usize(2)],
                    12,
                    5,
                ),
            ),
            block(
                vec![
                    assign(place(13), mir::Rvalue::Use(mir::Operand::Copy(deref_place(12)))),
                    assign(
                        place(14),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((copy(7), copy(10))),
                        ),
                    ),
                    assign(
                        place(0),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((copy(14), copy(13))),
                        ),
                    ),
                ],
                ret(),
            ),
        ],
    );
    let value = interpret_mir(&program(body), "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(6));
}
