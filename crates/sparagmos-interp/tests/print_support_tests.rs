//! Print-support tests (Lane N): ADT values, references, strings,
//! arrays, runtime-support calls and the fmt engine.
//!
//! Two levels are covered:
//! * hand-built `sparagmos-mir` bodies (shaped after the real
//!   `rustc 1.100.0` driver dumps of the corpus programs) converted
//!   and executed through `mir_convert`, including stdout capture;
//! * hand-built interpreter IR using `CallBuiltin` terminators
//!   directly, for the fmt template/options matrix.

use std::collections::HashMap;

use sparagmos_interp::builtins::{canonical_name, lookup};
use sparagmos_interp::mir_convert::{interpret_mir, interpret_mir_captured, MirProgram};
use sparagmos_interp::{
    BasicBlock, BasicBlockData, Body, Constant, InterpError, Interpreter, MirRunError, Operand,
    Place, Program, Rvalue, Statement, TerminatorKind, Ty, Value,
};
use sparagmos_mir as mir;

// ----- shared builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}
fn usize_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::Usize)
}
fn u8_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U8)
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

// DefIds of the std items the tests name (krate 0, arbitrary indices).
const ARGUMENTS_TY: mir::DefId = mir::DefId::new(0, 100);
const FROM_STR: mir::DefId = mir::DefId::new(0, 101);
const ARGUMENTS_NEW: mir::DefId = mir::DefId::new(0, 102);
const PRINT: mir::DefId = mir::DefId::new(0, 103);
const ARGUMENT_TY: mir::DefId = mir::DefId::new(0, 104);
const NEW_DISPLAY: mir::DefId = mir::DefId::new(0, 105);
const RANGE_INCL_TY: mir::DefId = mir::DefId::new(0, 106);
const RANGE_INCL_NEW: mir::DefId = mir::DefId::new(0, 107);
const INTO_ITER: mir::DefId = mir::DefId::new(0, 108);
const ITER_NEXT: mir::DefId = mir::DefId::new(0, 109);
const FROM_USIZE: mir::DefId = mir::DefId::new(0, 110);
const STRING_TY: mir::DefId = mir::DefId::new(0, 111);
const OPTION_TY: mir::DefId = mir::DefId::new(0, 113);

fn std_names() -> HashMap<mir::DefId, String> {
    HashMap::from([
        (ARGUMENTS_TY, "std::fmt::Arguments".to_string()),
        (FROM_STR, "std::fmt::Arguments::<'a>::from_str".to_string()),
        (ARGUMENTS_NEW, "std::fmt::Arguments::<'a>::new".to_string()),
        (PRINT, "std::io::_print".to_string()),
        (ARGUMENT_TY, "core::fmt::rt::Argument".to_string()),
        (
            NEW_DISPLAY,
            "core::fmt::rt::Argument::<'_>::new_display".to_string(),
        ),
        (RANGE_INCL_TY, "std::ops::RangeInclusive".to_string()),
        (
            RANGE_INCL_NEW,
            "std::ops::RangeInclusive::<Idx>::new".to_string(),
        ),
        (INTO_ITER, "std::iter::IntoIterator::into_iter".to_string()),
        (ITER_NEXT, "std::iter::Iterator::next".to_string()),
        (
            FROM_USIZE,
            "core::fmt::rt::Argument::from_usize".to_string(),
        ),
        (STRING_TY, "std::string::String".to_string()),
        (OPTION_TY, "std::option::Option".to_string()),
    ])
}

fn arguments_ty() -> mir::Ty {
    mir::Ty::Adt(ARGUMENTS_TY, vec![])
}

fn argument_ty() -> mir::Ty {
    mir::Ty::Adt(ARGUMENT_TY, vec![])
}

// ----- MIR level: from_str + _print (the hello shape) ------------------------

#[test]
fn hello_from_str_prints_to_captured_stdout() {
    // fn main() {
    //   _2 = Arguments::from_str(const "hello from corpus\n");
    //   _1 = _print(move _2);
    //   return;
    // }
    let body = mir::Body::new(
        decls(&[mir::Ty::unit(), mir::Ty::unit(), arguments_ty()]),
        0,
        vec![
            block(
                vec![],
                call(
                    fndef(FROM_STR, vec![]),
                    vec![const_str("hello from corpus\n")],
                    2,
                    1,
                ),
            ),
            block(vec![], call(fndef(PRINT, vec![]), vec![move_(2)], 1, 2)),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    prog.fn_names = std_names();
    let run = interpret_mir_captured(&prog, "main", vec![]);
    assert_eq!(run.result, Ok(Value::Unit));
    assert_eq!(run.stdout, "hello from corpus\n");
}

// ----- MIR level: println!("{}", 42) -----------------------------------------

#[test]
fn println_with_display_int() {
    // The arith-style shape: build a refs tuple's single reference,
    // wrap it in an rt::Argument, snapshot the argument array, build
    // Arguments from the template bytes, print.
    let template: Vec<u8> = vec![192, 1, 10, 0]; // "{}" placeholder, "\n", end
    let body = mir::Body::new(
        decls(&[
            mir::Ty::unit(),                                            // _0
            i32_ty(),                                                   // _1 value
            ref_ty(i32_ty(), false),                                    // _2 &value
            argument_ty(),                                              // _3 Argument
            mir::Ty::Array(Box::new(argument_ty()), mir::ArrayLen::Known(1)), // _4 args
            ref_ty(
                mir::Ty::Array(Box::new(u8_ty()), mir::ArrayLen::Known(4)),
                false,
            ), // _5 template
            ref_ty(
                mir::Ty::Array(Box::new(argument_ty()), mir::ArrayLen::Known(1)),
                false,
            ), // _6 &args
            arguments_ty(), // _7 Arguments
            mir::Ty::unit(), // _8 print dest
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(42))),
                    assign(place(2), ref_rvalue(1, false)),
                ],
                call(fndef(NEW_DISPLAY, vec![i32_ty()]), vec![move_(2)], 3, 1),
            ),
            block(
                vec![
                    assign(
                        place(4),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Array(argument_ty())),
                            vec![move_(3)],
                        ),
                    ),
                    assign(place(6), ref_rvalue(4, false)),
                    assign(
                        place(5),
                        mir::Rvalue::Use(mir::Operand::constant(mir::Constant::from_value(
                            mir::ConstValue::Bytes(template),
                        ))),
                    ),
                ],
                call(
                    fndef(ARGUMENTS_NEW, vec![]),
                    vec![move_(5), copy(6)],
                    7,
                    2,
                ),
            ),
            block(vec![], call(fndef(PRINT, vec![]), vec![move_(7)], 8, 3)),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    prog.fn_names = std_names();
    let run = interpret_mir_captured(&prog, "main", vec![]);
    assert_eq!(run.result, Ok(Value::Unit));
    assert_eq!(run.stdout, "42\n");
}

// ----- MIR level: RangeInclusive iteration through builtins ------------------

#[test]
fn range_inclusive_next_loop_sums() {
    // fn main() -> i32 {
    //   _1 = RangeInclusive::new(1, 3);
    //   _7 = 0;
    //   _2 = IntoIterator::into_iter(move _1);
    //   loop {
    //     _3 = &_2;
    //     _4 = Iterator::next(move _3);
    //     _5 = Discriminant(_4);
    //     switchInt(_5) [0 -> exit, otherwise -> body]
    //     body: _6 = (_4 as Some).0; _7 = _7 + _6;
    //   }
    //   return _7;  // 1 + 2 + 3 = 6
    // }
    let range_ty = mir::Ty::Adt(RANGE_INCL_TY, vec![i32_ty()]);
    let option_ty = mir::Ty::Adt(OPTION_TY, vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[
            i32_ty(),                                // _0 return
            range_ty.clone(),                        // _1
            range_ty.clone(),                        // _2 iterator
            ref_ty(range_ty.clone(), true),          // _3 &mut iter
            option_ty,                               // _4 Option<i32>
            mir::Ty::Int(mir::IntTy::Isize),         // _5 discriminant
            i32_ty(),                                // _6 payload
            i32_ty(),                                // _7 sum
        ]),
        0,
        vec![
            block(
                vec![],
                call(
                    fndef(RANGE_INCL_NEW, vec![i32_ty()]),
                    vec![const_i32(1), const_i32(3)],
                    1,
                    1,
                ),
            ),
            block(
                vec![assign(place(7), mir::Rvalue::Use(const_i32(0)))],
                call(fndef(INTO_ITER, vec![range_ty.clone()]), vec![move_(1)], 2, 2),
            ),
            block(
                vec![assign(place(3), ref_rvalue(2, true))],
                call(fndef(ITER_NEXT, vec![range_ty.clone()]), vec![move_(3)], 4, 3),
            ),
            block(
                vec![assign(
                    place(5),
                    mir::Rvalue::Discriminant(place(4)),
                )],
                mir::TerminatorKind::SwitchInt {
                    discr: copy(5),
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
                        place(6),
                        mir::Rvalue::Use(mir::Operand::Copy(
                            place(4).downcast(None, 1).field(0, i32_ty()),
                        )),
                    ),
                    assign(
                        place(7),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((copy(7), copy(6))),
                        ),
                    ),
                ],
                goto(2),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(7)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    prog.fn_names = std_names();
    let value = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(6));
}

// ----- MIR level: arrays, Index / ConstantIndex, OOB panic -------------------

fn array_body(index_value: u32) -> mir::Body {
    // fn main() -> i32 {
    //   _1 = [10, 20, 30]; _2 = index;
    //   _3 = _1[_2]; _0 = _3 + _1[0]; return _0;
    // }
    let const_indexed = mir::Place {
        local: mir::Local(1),
        projection: vec![mir::PlaceElem::ConstantIndex {
            offset: 0,
            min_length: 3,
            from_end: false,
        }],
    };
    mir::Body::new(
        decls(&[
            i32_ty(),
            mir::Ty::Array(Box::new(i32_ty()), mir::ArrayLen::Known(3)),
            usize_ty(),
            i32_ty(),
        ]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(i32_ty())),
                        vec![const_i32(10), const_i32(20), const_i32(30)],
                    ),
                ),
                assign(place(2), mir::Rvalue::Use(const_int(index_value as u128, 8))),
                assign(
                    place(3),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).index(mir::Local(2)))),
                ),
                assign(
                    place(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((copy(3), mir::Operand::Copy(const_indexed))),
                    ),
                ),
            ],
            ret(),
        )],
    )
}

#[test]
fn array_index_and_constant_index_read() {
    let mut prog = MirProgram::new();
    prog.add("main", array_body(1));
    let value = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(30)); // 20 + 10
}

#[test]
fn array_index_write_through_index_place() {
    // _1[_2] = 99 with _2 = 1, then return _1[_2] + _1[0] = 99 + 10.
    let mut body = array_body(1);
    body.basic_blocks[0].statements.insert(
        2,
        assign(
            place(1).index(mir::Local(2)),
            mir::Rvalue::Use(const_i32(99)),
        ),
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    let value = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(109));
}

#[test]
fn array_index_out_of_bounds_is_a_guest_panic() {
    let mut prog = MirProgram::new();
    prog.add("main", array_body(5));
    let err = interpret_mir(&prog, "main", vec![]).expect_err("OOB must panic");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::Panic { ref message })
            if message == "index out of bounds: the len is 3 but the index is 5"
        ),
        "expected the rustc bounds-check panic, got {err}"
    );
}

// ----- MIR level: writing through a &mut across a call ------------------------

#[test]
fn write_through_mut_ref_across_frames() {
    // fn bump(_1: &mut i32) { *_1 = *_1 + 1; }
    // fn main() -> i32 { _1 = 41; _2 = &mut _1; bump(move _2); return _1; }
    const BUMP: mir::DefId = mir::DefId::new(0, 50);
    let bump = mir::Body::new(
        decls(&[mir::Ty::unit(), ref_ty(i32_ty(), true), i32_ty()]),
        1,
        vec![block(
            vec![
                assign(
                    place(2),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((
                            mir::Operand::Copy(place(1).deref()),
                            const_i32(1),
                        )),
                    ),
                ),
                assign(place(1).deref(), mir::Rvalue::Use(copy(2))),
            ],
            ret(),
        )],
    );
    let main = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), ref_ty(i32_ty(), true), mir::Ty::unit()]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(41))),
                    assign(place(2), ref_rvalue(1, true)),
                ],
                call(fndef(BUMP, vec![]), vec![move_(2)], 3, 1),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add_with_def_id("bump", BUMP, bump);
    prog.add("main", main);
    let value = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(value, Value::Int(42));
}

// ----- MIR level: Drop of a String executes -----------------------------------

#[test]
fn string_drop_executes_as_a_jump() {
    // fn main() { _1 = "x"; drop(_1); return; }
    let body = mir::Body::new(
        decls(&[mir::Ty::unit(), mir::Ty::Adt(STRING_TY, vec![])]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_str("x")))],
                mir::TerminatorKind::Drop {
                    place: place(1),
                    target: mir::BasicBlock(1),
                    unwind: mir::UnwindAction::Continue,
                    replace: false,
                },
            ),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    prog.fn_names = std_names();
    let value = interpret_mir(&prog, "main", vec![]).expect("run");
    assert_eq!(value, Value::Unit);
}

// ----- name handling -----------------------------------------------------------

#[test]
fn canonical_name_strips_only_lifetime_groups() {
    assert_eq!(
        canonical_name("std::fmt::Arguments::<'a>::from_str"),
        "std::fmt::Arguments::from_str"
    );
    assert_eq!(
        canonical_name("core::fmt::rt::Argument::<'_>::new_display"),
        "core::fmt::rt::Argument::new_display"
    );
    assert_eq!(
        canonical_name("std::ops::RangeInclusive::<Idx>::new"),
        "std::ops::RangeInclusive::<Idx>::new"
    );
    assert_eq!(
        canonical_name("std::iter::Iterator::next"),
        "std::iter::Iterator::next"
    );
    for name in [
        "std::io::_print",
        "std::fmt::Arguments::from_str",
        "core::fmt::rt::Argument::new_display",
        "std::iter::Iterator::next",
    ] {
        assert!(lookup(name).is_some(), "{name} must be registered");
    }
    assert!(lookup("std::vec::Vec::push").is_none());
    assert!(lookup("std::fs::read").is_none());
}

// ----- IR level: the fmt engine -------------------------------------------------

enum Step {
    Stmt(Statement),
    Call {
        name: &'static str,
        generic_args: Vec<String>,
        args: Vec<Operand>,
        dest: usize,
    },
}

fn stmt(place: usize, rvalue: Rvalue) -> Step {
    Step::Stmt(Statement::assign(Place::from_index(place), rvalue))
}

fn builtin(name: &'static str, generic_args: &[&str], args: Vec<Operand>, dest: usize) -> Step {
    Step::Call {
        name,
        generic_args: generic_args.iter().map(|s| s.to_string()).collect(),
        args,
        dest,
    }
}

/// Assemble steps into a `main` body (each `Call` step closes a
/// block) and run it, returning the captured guest stdout.
fn run_steps(local_tys: Vec<Ty>, steps: Vec<Step>) -> String {
    let mut blocks: Vec<BasicBlockData> = Vec::new();
    let mut stmts: Vec<Statement> = Vec::new();
    for step in steps {
        match step {
            Step::Stmt(s) => stmts.push(s),
            Step::Call {
                name,
                generic_args,
                args,
                dest,
            } => {
                let target = BasicBlock(blocks.len() + 1);
                blocks.push(BasicBlockData::new(
                    std::mem::take(&mut stmts),
                    TerminatorKind::CallBuiltin {
                        name: name.to_string(),
                        generic_args,
                        args,
                        destination: Place::from_index(dest),
                        target: Some(target),
                    },
                ));
            }
        }
    }
    blocks.push(BasicBlockData::new(stmts, TerminatorKind::Return));
    let mut program = Program::new();
    program.add_function(Body::new("main", 0, local_tys, blocks));
    let mut interp = Interpreter::new(&program);
    let value = interp.run("main", vec![]).expect("run must succeed");
    assert_eq!(value, Value::Unit);
    interp.output().to_string()
}

fn adt(name: &str) -> Ty {
    Ty::Adt(name.to_string())
}

fn arg_array_ty(n: u64) -> Ty {
    Ty::Array(Box::new(adt("core::fmt::rt::Argument")), n)
}

fn template_ty(n: u64) -> Ty {
    Ty::Ref(Box::new(Ty::Array(
        Box::new(Ty::Uint(sparagmos_interp::ir::UintTy::U8)),
        n,
    )))
}

/// The standard print chain over locals laid out as
/// `[..., array, template, &array, arguments, unit]`: snapshot the
/// argument values in `arg_dests` into the array, build the
/// `Arguments`, print. Returns the steps; the caller declares the
/// local types.
fn print_chain(
    array: usize,
    template: usize,
    array_ref: usize,
    arguments: usize,
    unit: usize,
    arg_dests: &[usize],
    template_bytes: Vec<u8>,
) -> Vec<Step> {
    vec![
        stmt(
            array,
            Rvalue::AggregateTuple(
                arg_dests
                    .iter()
                    .map(|d| Operand::move_local(*d))
                    .collect(),
            ),
        ),
        stmt(
            template,
            Rvalue::Use(Operand::Constant(Constant::Bytes(template_bytes))),
        ),
        stmt(array_ref, Rvalue::Ref(Place::from_index(array))),
        builtin(
            "std::fmt::Arguments::new",
            &[],
            vec![
                Operand::move_local(template),
                Operand::copy_local(array_ref),
            ],
            arguments,
        ),
        builtin(
            "std::io::_print",
            &[],
            vec![Operand::move_local(arguments)],
            unit,
        ),
    ]
}

/// Locals for a one-argument print: _0 unit, _1 value, _2 &value,
/// _3 Argument, then the chain locals _4.._8.
fn one_arg_tys(value_ty: Ty, template_len: u64) -> Vec<Ty> {
    vec![
        Ty::Unit,
        value_ty.clone(),
        Ty::Ref(Box::new(value_ty)),
        adt("core::fmt::rt::Argument"),
        arg_array_ty(1),
        template_ty(template_len),
        Ty::Ref(Box::new(arg_array_ty(1))),
        adt("std::fmt::Arguments"),
        Ty::Unit,
    ]
}

fn print_one(
    value_ty: Ty,
    value_stmt: Rvalue,
    trait_fn: &'static str,
    tag: &str,
    template: Vec<u8>,
) -> String {
    let template_len = template.len() as u64;
    let mut steps = vec![
        stmt(1, value_stmt),
        stmt(2, Rvalue::Ref(Place::from_index(1))),
        builtin(trait_fn, &[tag], vec![Operand::move_local(2)], 3),
    ];
    steps.extend(print_chain(4, 5, 6, 7, 8, &[3], template));
    run_steps(one_arg_tys(value_ty, template_len), steps)
}

#[test]
fn fmt_default_placeholder_int() {
    let out = print_one(
        Ty::i32(),
        Rvalue::Use(Operand::const_int(42)),
        "core::fmt::rt::Argument::new_display",
        "i32",
        vec![192, 1, 10, 0],
    );
    assert_eq!(out, "42\n");
}

#[test]
fn fmt_width_align_and_zero_pad() {
    // "{:>6}" of -42: flags = fill ' ' | align Right; width 6.
    let flags: u32 = 0x20 | (1 << 29);
    let mut template = vec![0xC3u8];
    template.extend_from_slice(&flags.to_le_bytes());
    template.extend_from_slice(&6u16.to_le_bytes());
    template.push(0);
    let out = print_one(
        Ty::i32(),
        Rvalue::Use(Operand::const_int(-42)),
        "core::fmt::rt::Argument::new_display",
        "i32",
        template,
    );
    assert_eq!(out, "   -42");

    // "{:05}" of 42: flags = fill ' ' | align unknown | zero pad.
    let flags: u32 = 0x20 | (3 << 29) | (1 << 24);
    let mut template = vec![0xC3u8];
    template.extend_from_slice(&flags.to_le_bytes());
    template.extend_from_slice(&5u16.to_le_bytes());
    template.push(0);
    let out = print_one(
        Ty::i32(),
        Rvalue::Use(Operand::const_int(42)),
        "core::fmt::rt::Argument::new_display",
        "i32",
        template,
    );
    assert_eq!(out, "00042");
}

#[test]
fn fmt_precision_truncates_str() {
    // "{:.3}" of "abcdef": precision field only.
    let out = print_one(
        Ty::Ref(Box::new(Ty::Str)),
        Rvalue::Use(Operand::Constant(Constant::Str("abcdef".to_string()))),
        "core::fmt::rt::Argument::new_display",
        "&str",
        vec![0xC4, 3, 0, 0],
    );
    assert_eq!(out, "abc");
}

#[test]
fn fmt_char_display() {
    let out = print_one(
        Ty::u32(),
        Rvalue::Use(Operand::const_uint(97)),
        "core::fmt::rt::Argument::new_display",
        "char",
        vec![192, 0],
    );
    assert_eq!(out, "a");
}

#[test]
fn fmt_lower_hex_with_alternate_prefix() {
    // "{:#x}" through new_lower_hex: flags field carries ALTERNATE.
    let flags: u32 = 0x20 | (3 << 29) | (1 << 23);
    let mut template = vec![0xC1u8];
    template.extend_from_slice(&flags.to_le_bytes());
    template.push(0);
    let out = print_one(
        Ty::i32(),
        Rvalue::Use(Operand::const_int(255)),
        "core::fmt::rt::Argument::new_lower_hex",
        "i32",
        template,
    );
    assert_eq!(out, "0xff");
}

#[test]
fn fmt_debug_tuple() {
    let out = print_one(
        Ty::Tuple(vec![Ty::i32(), Ty::i32()]),
        Rvalue::AggregateTuple(vec![Operand::const_int(1), Operand::const_int(2)]),
        "core::fmt::rt::Argument::new_debug",
        "(i32, i32)",
        vec![192, 0],
    );
    assert_eq!(out, "(1, 2)");
}

#[test]
fn fmt_explicit_arg_index() {
    // "{1} {0}" over (10, 20) -> "20 10".
    let template = vec![0xC8, 1, 0, 1, 32, 0xC8, 0, 0, 0];
    let tys = vec![
        Ty::Unit,                                    // _0
        Ty::i32(),                                   // _1 = 10
        Ty::Ref(Box::new(Ty::i32())),                // _2
        adt("core::fmt::rt::Argument"),              // _3
        Ty::i32(),                                   // _4 = 20
        Ty::Ref(Box::new(Ty::i32())),                // _5
        adt("core::fmt::rt::Argument"),              // _6
        arg_array_ty(2),                             // _7
        template_ty(template.len() as u64),          // _8
        Ty::Ref(Box::new(arg_array_ty(2))),          // _9
        adt("std::fmt::Arguments"),                  // _10
        Ty::Unit,                                    // _11
    ];
    let mut steps = vec![
        stmt(1, Rvalue::Use(Operand::const_int(10))),
        stmt(2, Rvalue::Ref(Place::from_index(1))),
        builtin(
            "core::fmt::rt::Argument::new_display",
            &["i32"],
            vec![Operand::move_local(2)],
            3,
        ),
        stmt(4, Rvalue::Use(Operand::const_int(20))),
        stmt(5, Rvalue::Ref(Place::from_index(4))),
        builtin(
            "core::fmt::rt::Argument::new_display",
            &["i32"],
            vec![Operand::move_local(5)],
            6,
        ),
    ];
    steps.extend(print_chain(7, 8, 9, 10, 11, &[3, 6], template));
    assert_eq!(run_steps(tys, steps), "20 10");
}

#[test]
fn fmt_indirect_width_from_count_arg() {
    // "{:w$}" over (7, width=5) -> "    7": the placeholder's width
    // field is the index of a count argument.
    let template = vec![0xD2, 1, 0, 0];
    let tys = vec![
        Ty::Unit,                           // _0
        Ty::i32(),                          // _1 = 7
        Ty::Ref(Box::new(Ty::i32())),       // _2
        adt("core::fmt::rt::Argument"),     // _3 display arg
        adt("core::fmt::rt::Argument"),     // _4 count arg
        arg_array_ty(2),                    // _5
        template_ty(template.len() as u64), // _6
        Ty::Ref(Box::new(arg_array_ty(2))), // _7
        adt("std::fmt::Arguments"),         // _8
        Ty::Unit,                           // _9
    ];
    let mut steps = vec![
        stmt(1, Rvalue::Use(Operand::const_int(7))),
        stmt(2, Rvalue::Ref(Place::from_index(1))),
        builtin(
            "core::fmt::rt::Argument::new_display",
            &["i32"],
            vec![Operand::move_local(2)],
            3,
        ),
        builtin(
            "core::fmt::rt::Argument::from_usize",
            &[],
            vec![Operand::const_uint(5)],
            4,
        ),
    ];
    steps.extend(print_chain(5, 6, 7, 8, 9, &[3, 4], template));
    assert_eq!(run_steps(tys, steps), "    7");
}

#[test]
fn fmt_long_literal_piece() {
    // A 200-byte literal uses the 0x80 long-piece form.
    let literal = "a".repeat(200);
    let mut template = vec![0x80u8];
    template.extend_from_slice(&200u16.to_le_bytes());
    template.extend_from_slice(literal.as_bytes());
    template.push(0);
    let tys = vec![
        Ty::Unit,
        arg_array_ty(0),
        template_ty(template.len() as u64),
        Ty::Ref(Box::new(arg_array_ty(0))),
        adt("std::fmt::Arguments"),
        Ty::Unit,
    ];
    let steps = print_chain(1, 2, 3, 4, 5, &[], template);
    assert_eq!(run_steps(tys, steps), literal);
}

// ----- floats and array Debug (Lane S) ---------------------------------------

fn f64_ir_ty() -> Ty {
    Ty::Float(sparagmos_interp::ir::FloatTy::F64)
}

fn f32_ir_ty() -> Ty {
    Ty::Float(sparagmos_interp::ir::FloatTy::F32)
}

#[test]
fn fmt_display_f64() {
    let out = print_one(
        f64_ir_ty(),
        Rvalue::Use(Operand::Constant(Constant::Float(3.5))),
        "core::fmt::rt::Argument::new_display",
        "f64",
        vec![192, 0],
    );
    assert_eq!(out, "3.5");
}

#[test]
fn fmt_display_f32_narrows_before_formatting() {
    // The stored value is the exact f64 widening of 0.1f32; Display
    // must render the f32 value ("0.1"), not the widened digits.
    let out = print_one(
        f32_ir_ty(),
        Rvalue::Use(Operand::Constant(Constant::Float(0.1f32 as f64))),
        "core::fmt::rt::Argument::new_display",
        "f32",
        vec![192, 0],
    );
    assert_eq!(out, "0.1");
}

#[test]
fn fmt_debug_f64_keeps_point_zero() {
    let out = print_one(
        f64_ir_ty(),
        Rvalue::Use(Operand::Constant(Constant::Float(2.0))),
        "core::fmt::rt::Argument::new_debug",
        "f64",
        vec![192, 0],
    );
    assert_eq!(out, "2.0");
}

#[test]
fn fmt_float_precision() {
    // "{:.2}" of 3.14159: precision field only.
    let out = print_one(
        f64_ir_ty(),
        Rvalue::Use(Operand::Constant(Constant::Float(3.14159))),
        "core::fmt::rt::Argument::new_display",
        "f64",
        vec![0xC4, 2, 0, 0],
    );
    assert_eq!(out, "3.14");
}

#[test]
fn fmt_float_zero_pad_and_width() {
    // "{:08.2}" of 3.14159: flags + width + precision fields.
    let flags: u32 = 0x20 | (3 << 29) | (1 << 24);
    let mut template = vec![0xC7u8];
    template.extend_from_slice(&flags.to_le_bytes());
    template.extend_from_slice(&8u16.to_le_bytes());
    template.extend_from_slice(&2u16.to_le_bytes());
    template.push(0);
    let out = print_one(
        f64_ir_ty(),
        Rvalue::Use(Operand::Constant(Constant::Float(3.14159))),
        "core::fmt::rt::Argument::new_display",
        "f64",
        template,
    );
    assert_eq!(out, "00003.14");
}

#[test]
fn fmt_float_lower_exp() {
    let out = print_one(
        f64_ir_ty(),
        Rvalue::Use(Operand::Constant(Constant::Float(314.159))),
        "core::fmt::rt::Argument::new_lower_exp",
        "f64",
        vec![192, 0],
    );
    assert_eq!(out, "3.14159e2");
}

#[test]
fn fmt_debug_array_uses_brackets() {
    let out = print_one(
        Ty::Array(Box::new(Ty::i32()), 4),
        Rvalue::AggregateTuple(vec![
            Operand::const_int(1),
            Operand::const_int(3),
            Operand::const_int(4),
            Operand::const_int(5),
        ]),
        "core::fmt::rt::Argument::new_debug",
        "[i32; 4]",
        vec![192, 0],
    );
    assert_eq!(out, "[1, 3, 4, 5]");
}

#[test]
fn fmt_debug_array_pretty_uses_brackets() {
    // "{:#?}" of the array: ALTERNATE flag in the flags field.
    let flags: u32 = 0x20 | (3 << 29) | (1 << 23);
    let mut template = vec![0xC1u8];
    template.extend_from_slice(&flags.to_le_bytes());
    template.push(0);
    let out = print_one(
        Ty::Array(Box::new(Ty::i32()), 4),
        Rvalue::AggregateTuple(vec![
            Operand::const_int(1),
            Operand::const_int(3),
            Operand::const_int(4),
            Operand::const_int(5),
        ]),
        "core::fmt::rt::Argument::new_debug",
        "[i32; 4]",
        template,
    );
    assert_eq!(out, "[\n    1,\n    3,\n    4,\n    5,\n]");
}
