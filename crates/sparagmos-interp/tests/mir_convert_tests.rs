//! Conversion-layer tests: hand-built `sparagmos-mir` bodies (shaped
//! after real `rustc 1.100.0 --emit=mir` output) are converted and
//! executed by the interpreter, and the results are compared against
//! independently known values.

use sparagmos_interp::mir_convert::{
    convert_program, interpret_mir, ConvertError, MirProgram, MirRunError,
};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir -----------------------------------

fn i32_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I32)
}
fn u32_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U32)
}
fn u8_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U8)
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

fn field(idx: u32, field: u32, ty: mir::Ty) -> mir::Place {
    place(idx).field(field, ty)
}

fn copy(idx: u32) -> mir::Operand {
    mir::Operand::Copy(place(idx))
}

fn copy_deref(idx: u32) -> mir::Operand {
    mir::Operand::Copy(place(idx).deref())
}

fn move_(idx: u32) -> mir::Operand {
    mir::Operand::Move(place(idx))
}

/// An integer scalar constant with an explicit storage size in bytes
/// (the shape `sparagmos-mir` uses for `const 42_i32` etc.).
fn const_int(data: u128, size: u8) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int { data, size },
    )))
}

fn const_i32(v: i32) -> mir::Operand {
    const_int(v as u32 as u128, 4)
}

fn const_u32(v: u32) -> mir::Operand {
    const_int(v as u128, 4)
}

fn const_u8(v: u8) -> mir::Operand {
    const_int(v as u128, 1)
}

fn block(stmts: Vec<mir::Statement>, kind: mir::TerminatorKind) -> mir::BasicBlockData {
    let mut bb = mir::BasicBlockData::new(mir::Terminator::new(kind));
    bb.statements = stmts;
    bb
}

fn assign(place: mir::Place, rv: mir::Rvalue) -> mir::Statement {
    mir::Statement::assign(place, rv)
}

fn binop(op: mir::BinOp, lhs: mir::Operand, rhs: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::BinaryOp(op, Box::new((lhs, rhs)))
}

fn goto(bb: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::Goto {
        target: mir::BasicBlock(bb),
    }
}

fn switch_bool(discr: mir::Operand, on_false: u32, otherwise: u32) -> mir::TerminatorKind {
    mir::TerminatorKind::SwitchInt {
        discr,
        targets: mir::SwitchTargets::new(
            vec![0],
            vec![mir::BasicBlock(on_false)],
            mir::BasicBlock(otherwise),
        ),
    }
}

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

// ----- add: plain and rustc-checked shapes -----------------------------------

/// `fn add(_1: i32, _2: i32) -> i32 { _0 = Add(copy _1, copy _2); return }`
fn add_plain_body() -> mir::Body {
    mir::Body::new(
        decls(&[i32_ty(), i32_ty(), i32_ty()]),
        2,
        vec![block(
            vec![assign(place(0), binop(mir::BinOp::Add, copy(1), copy(2)))],
            ret(),
        )],
    )
}

/// The shape rustc 1.100.0 actually emits for `a + b` in debug builds:
/// `_3 = CheckedBinaryOp(Add, ..); assert(!move _3.1) -> bb1; bb1: _0 = move _3.0; return`
/// (the `!` in the printed MIR is `expected: false` on the Assert).
fn add_checked_body() -> mir::Body {
    let pair_ty = mir::Ty::Tuple(vec![i32_ty(), bool_ty()]);
    mir::Body::new(
        decls(&[i32_ty(), i32_ty(), i32_ty(), pair_ty]),
        2,
        vec![
            block(
                vec![assign(
                    place(3),
                    mir::Rvalue::CheckedBinaryOp(mir::BinOp::Add, Box::new((copy(1), copy(2)))),
                )],
                mir::TerminatorKind::Assert {
                    cond: mir::Operand::Move(field(3, 1, bool_ty())),
                    expected: false,
                    msg: Box::new(mir::AssertMessage::Overflow(
                        mir::BinOp::Add,
                        Box::new(copy(1)),
                        Box::new(copy(2)),
                    )),
                    target: mir::BasicBlock(1),
                    unwind: mir::UnwindAction::Continue,
                },
            ),
            block(
                vec![assign(place(0), mir::Rvalue::Use(move_field(3, 0)))],
                ret(),
            ),
        ],
    )
}

fn move_field(idx: u32, f: u32) -> mir::Operand {
    let ty = if f == 0 { i32_ty() } else { bool_ty() };
    mir::Operand::Move(field(idx, f, ty))
}

#[test]
fn plain_add_converts_and_runs() {
    let mut prog = MirProgram::new();
    prog.add("add", add_plain_body());
    let v = interpret_mir(&prog, "add", vec![Value::Int(40), Value::Int(2)]).expect("add must run");
    assert_eq!(v, Value::Int(42));
}

#[test]
fn checked_add_matches_rustc_shape() {
    let mut prog = MirProgram::new();
    prog.add("add", add_checked_body());
    let v = interpret_mir(&prog, "add", vec![Value::Int(40), Value::Int(2)])
        .expect("checked add must run");
    assert_eq!(v, Value::Int(42));
}

#[test]
fn checked_add_overflow_is_a_guest_panic() {
    let mut prog = MirProgram::new();
    prog.add("add", add_checked_body());
    let err = interpret_mir(
        &prog,
        "add",
        vec![Value::Int(i32::MAX as i128), Value::Int(1)],
    )
    .expect_err("overflowing checked add must panic via the Assert");
    assert!(
        matches!(err, MirRunError::Interp(InterpError::Panic { .. })),
        "expected guest panic, got {err}"
    );
}

// ----- loop: sum_to ----------------------------------------------------------

/// `fn sum_to(_1: u32) -> u32`: sum of `1..=_1`, MIR-style with a
/// bool temporary and a `switchInt` loop head.
fn sum_to_body() -> mir::Body {
    mir::Body::new(
        decls(&[u32_ty(), u32_ty(), u32_ty(), u32_ty(), bool_ty()]),
        1,
        vec![
            // bb0: _2 = 0; _3 = 1; goto bb1
            block(
                vec![
                    assign(place(2), mir::Rvalue::Use(const_u32(0))),
                    assign(place(3), mir::Rvalue::Use(const_u32(1))),
                ],
                goto(1),
            ),
            // bb1: _4 = Le(copy _3, copy _1); switchInt(copy _4) -> [0: bb3, otherwise: bb2]
            block(
                vec![assign(place(4), binop(mir::BinOp::Le, copy(3), copy(1)))],
                switch_bool(copy(4), 3, 2),
            ),
            // bb2: _2 = Add(copy _2, copy _3); _3 = Add(copy _3, 1); goto bb1
            block(
                vec![
                    assign(place(2), binop(mir::BinOp::Add, copy(2), copy(3))),
                    assign(place(3), binop(mir::BinOp::Add, copy(3), const_u32(1))),
                ],
                goto(1),
            ),
            // bb3: _0 = copy _2; return
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    )
}

#[test]
fn loop_sum_to_10() {
    let mut prog = MirProgram::new();
    prog.add("sum_to", sum_to_body());
    assert_eq!(
        interpret_mir(&prog, "sum_to", vec![Value::Uint(10)]).expect("sum_to runs"),
        Value::Uint(55)
    );
    assert_eq!(
        interpret_mir(&prog, "sum_to", vec![Value::Uint(0)]).expect("sum_to runs"),
        Value::Uint(0)
    );
    assert_eq!(
        interpret_mir(&prog, "sum_to", vec![Value::Uint(100)]).expect("sum_to runs"),
        Value::Uint(5050)
    );
}

// ----- recursion: fib through a FnDef call ------------------------------------

const FIB_DEF: mir::DefId = mir::DefId::new(0, 7);

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

/// `fn fib(_1: u32) -> u32` with two recursive calls resolved by DefId.
fn fib_body() -> mir::Body {
    mir::Body::new(
        decls(&[
            u32_ty(),
            u32_ty(),
            bool_ty(),
            u32_ty(),
            u32_ty(),
            u32_ty(),
            u32_ty(),
        ]),
        1,
        vec![
            // bb0: _2 = Lt(copy _1, 2); switchInt(copy _2) -> [0: bb2, otherwise: bb1]
            block(
                vec![assign(
                    place(2),
                    binop(mir::BinOp::Lt, copy(1), const_u32(2)),
                )],
                switch_bool(copy(2), 2, 1),
            ),
            // bb1: _0 = copy _1; return
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
            // bb2: _3 = Sub(copy _1, 1); _4 = call fib(copy _3) -> bb3
            block(
                vec![assign(
                    place(3),
                    binop(mir::BinOp::Sub, copy(1), const_u32(1)),
                )],
                call(fndef_operand(FIB_DEF), copy(3), 4, 3),
            ),
            // bb3: _5 = Sub(copy _1, 2); _6 = call fib(copy _5) -> bb4
            block(
                vec![assign(
                    place(5),
                    binop(mir::BinOp::Sub, copy(1), const_u32(2)),
                )],
                call(fndef_operand(FIB_DEF), copy(5), 6, 4),
            ),
            // bb4: _0 = Add(copy _4, copy _6); return
            block(
                vec![assign(place(0), binop(mir::BinOp::Add, copy(4), copy(6)))],
                ret(),
            ),
        ],
    )
}

#[test]
fn recursive_fib_via_fndef_calls() {
    let mut prog = MirProgram::new();
    prog.add_with_def_id("fib", FIB_DEF, fib_body());
    for (n, want) in [(0u128, 0u128), (1, 1), (10, 55), (15, 610)] {
        assert_eq!(
            interpret_mir(&prog, "fib", vec![Value::Uint(n)]).expect("fib runs"),
            Value::Uint(want),
            "fib({n})"
        );
    }
}

// ----- if/else: max ------------------------------------------------------------

fn max_body() -> mir::Body {
    mir::Body::new(
        decls(&[i32_ty(), i32_ty(), i32_ty(), bool_ty()]),
        2,
        vec![
            block(
                vec![assign(place(3), binop(mir::BinOp::Gt, copy(1), copy(2)))],
                switch_bool(copy(3), 2, 1),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    )
}

#[test]
fn if_else_max() {
    let mut prog = MirProgram::new();
    prog.add("max", max_body());
    assert_eq!(
        interpret_mir(&prog, "max", vec![Value::Int(7), Value::Int(3)]).expect("max runs"),
        Value::Int(7)
    );
    assert_eq!(
        interpret_mir(&prog, "max", vec![Value::Int(2), Value::Int(9)]).expect("max runs"),
        Value::Int(9)
    );
}

// ----- casts, negative constants, aggregates ------------------------------------

#[test]
fn int_to_int_cast_widens_and_truncates() {
    // fn widen(_1: u8) -> u64 { _0 = Cast(IntToInt, copy _1, u64); return }
    let widen = mir::Body::new(
        decls(&[mir::Ty::Uint(mir::UintTy::U64), u8_ty()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Cast(
                    mir::CastKind::IntToInt,
                    copy(1),
                    mir::Ty::Uint(mir::UintTy::U64),
                ),
            )],
            ret(),
        )],
    );
    // fn trunc(_1: i32) -> i8 { _0 = Cast(IntToInt, copy _1, i8); return }
    let trunc = mir::Body::new(
        decls(&[mir::Ty::Int(mir::IntTy::I8), i32_ty()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Cast(
                    mir::CastKind::IntToInt,
                    copy(1),
                    mir::Ty::Int(mir::IntTy::I8),
                ),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("widen", widen);
    prog.add("trunc", trunc);
    assert_eq!(
        interpret_mir(&prog, "widen", vec![Value::Uint(200)]).expect("widen runs"),
        Value::Uint(200)
    );
    // 300 wraps to 44 in i8, as a Rust `as` cast does.
    assert_eq!(
        interpret_mir(&prog, "trunc", vec![Value::Int(300)]).expect("trunc runs"),
        Value::Int(44)
    );
}

#[test]
fn negative_constant_sign_extends_from_dest_ty() {
    // fn neg5() -> i32 { _0 = const -5_i32; return }
    let body = mir::Body::new(
        decls(&[i32_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_i32(-5)))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("neg5", body);
    assert_eq!(
        interpret_mir(&prog, "neg5", vec![]).expect("neg5 runs"),
        Value::Int(-5)
    );
}

#[test]
fn tuple_and_array_aggregates_and_repeat() {
    let triple_ty = mir::Ty::Tuple(vec![u8_ty(), u8_ty(), u8_ty()]);
    // fn pair() -> (u8, u8) { _0 = (const 1, const 2); return }  (tuple aggregate)
    let pair = mir::Body::new(
        decls(&[mir::Ty::Tuple(vec![u8_ty(), u8_ty()])]),
        0,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Tuple),
                    vec![const_u8(1), const_u8(2)],
                ),
            )],
            ret(),
        )],
    );
    // fn sevens() -> (u8, u8, u8) { _0 = Repeat(const 7, 3); return }
    let sevens = mir::Body::new(
        decls(&[triple_ty]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Repeat(const_u8(7), 3))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("pair", pair);
    prog.add("sevens", sevens);
    assert_eq!(
        interpret_mir(&prog, "pair", vec![]).expect("pair runs"),
        Value::Tuple(vec![Value::Uint(1), Value::Uint(2)])
    );
    assert_eq!(
        interpret_mir(&prog, "sevens", vec![]).expect("sevens runs"),
        Value::Tuple(vec![Value::Uint(7), Value::Uint(7), Value::Uint(7)])
    );
}

// ----- explicit failures (nothing is silently skipped) ---------------------------

#[test]
fn f128_local_converts_and_executes() {
    // f128 is part of the type model: the value rides as its raw
    // binary128 bit pattern and arithmetic runs in the exact
    // software module (see f128_tests.rs for the arithmetic
    // itself). _1 = 1.5f128; return _1 + _1 == 3.0f128, bit-exact.
    let const_1_5 = mir::Operand::constant(mir::Constant::from_value(
        mir::ConstValue::Scalar(mir::Scalar::Float {
            bits: 0x3fff_8000_0000_0000_0000_0000_0000_0000,
            ty: mir::FloatTy::F128,
        }),
    ));
    let body = mir::Body::new(
        decls(&[
            mir::Ty::Float(mir::FloatTy::F128),
            mir::Ty::Float(mir::FloatTy::F128),
        ]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_1_5)),
                assign(place(0), binop(mir::BinOp::Add, copy(1), copy(1))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    let converted = convert_program(&prog).expect("conversion is per-body");
    assert!(converted.get("f").is_some(), "f128 body must convert");
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("f runs"),
        Value::Float128(0x4000_8000_0000_0000_0000_0000_0000_0000)
    );
}

#[test]
fn subslice_projection_converts_and_executes() {
    // _1 = [10, 20]; _1[0..2][1] = 7; _0 = copy (_1[1])
    // The Subslice step converts, and a write through it composed
    // with a constant index lands on the base array element.
    let write_place = mir::Place {
        local: mir::Local(1),
        projection: vec![
            mir::PlaceElem::Subslice {
                from: 0,
                to: 2,
                from_end: false,
            },
            mir::PlaceElem::ConstantIndex {
                offset: 1,
                min_length: 2,
                from_end: false,
            },
        ],
    };
    let read_place = mir::Place {
        local: mir::Local(1),
        projection: vec![mir::PlaceElem::ConstantIndex {
            offset: 1,
            min_length: 2,
            from_end: false,
        }],
    };
    let array_ty = mir::Ty::Array(Box::new(u32_ty()), mir::ArrayLen::Known(2));
    let body = mir::Body::new(
        decls(&[u32_ty(), array_ty]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(u32_ty())),
                        vec![const_u32(10), const_u32(20)],
                    ),
                ),
                assign(write_place, mir::Rvalue::Use(const_u32(7))),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(read_place)),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("sub", body);
    let result = interpret_mir(&prog, "sub", vec![]).expect("subslice body must run");
    assert_eq!(result, Value::Uint(7));
}

#[test]
fn subslice_len_resolves_against_the_base() {
    // _1 = [10, 20, 30]; _0 = Len(_1[1..]) -> 2: the from_end
    // Subslice form resolves its length against the base place.
    let len_place = mir::Place {
        local: mir::Local(1),
        projection: vec![mir::PlaceElem::Subslice {
            from: 1,
            to: 0,
            from_end: true,
        }],
    };
    let array_ty = mir::Ty::Array(Box::new(u32_ty()), mir::ArrayLen::Known(3));
    let body = mir::Body::new(
        decls(&[mir::Ty::Uint(mir::UintTy::Usize), array_ty]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(u32_ty())),
                        vec![const_u32(10), const_u32(20), const_u32(30)],
                    ),
                ),
                assign(place(0), mir::Rvalue::Len(len_place)),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("sublen", body);
    let result = interpret_mir(&prog, "sublen", vec![]).expect("subslice Len must run");
    assert_eq!(result, Value::Uint(2));
}

#[test]
fn pattern_type_converts_and_executes_as_its_base_type() {
    // _0 = copy _1 with _1: Pat(u32, "1.."): a pattern type is a
    // refinement of its base type with the same representation,
    // so conversion keeps the base type and the value flows
    // through unchanged.
    let pat_ty = mir::Ty::Pat(Box::new(u32_ty()), "1..".to_string());
    let body = mir::Body::new(
        decls(&[u32_ty(), pat_ty]),
        1,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(copy(1)))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("patid", body);
    let result = interpret_mir(&prog, "patid", vec![Value::Uint(41)])
        .expect("pattern-typed body must run");
    assert_eq!(result, Value::Uint(41));
}
#[test]
fn drop_terminator_fails_at_execution_not_silently() {
    // The dropped place has an ADT type the value model does not know:
    // its drop glue might do anything, so the Drop stays an explicit
    // execution error. (Drops of plain data and of modelled std types
    // execute as plain jumps; see print_support_tests.)
    let body = mir::Body::new(
        decls(&[
            mir::Ty::unit(),
            mir::Ty::Adt(mir::DefId::new(0, 7), vec![]),
        ]),
        0,
        vec![block(
            vec![],
            mir::TerminatorKind::Drop {
                place: place(1),
                target: mir::BasicBlock(1),
                unwind: mir::UnwindAction::Continue,
                replace: false,
            },
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("dropper", body);
    let err = interpret_mir(&prog, "dropper", vec![]).expect_err("Drop must not execute");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedTerminator { ref kind }) if kind == "Drop"
        ),
        "expected UnsupportedTerminator(Drop), got {err}"
    );
}

#[test]
fn unresolved_callee_fails_at_execution_not_silently() {
    let body = mir::Body::new(
        decls(&[u32_ty(), u32_ty()]),
        1,
        vec![block(
            vec![],
            call(fndef_operand(mir::DefId::new(9, 9)), copy(1), 0, 1),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("caller", body);
    let err = interpret_mir(&prog, "caller", vec![Value::Uint(1)])
        .expect_err("call to unknown DefId must fail");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedTerminator { .. })
        ),
        "expected UnsupportedTerminator, got {err}"
    );
}

#[test]
fn closure_aggregate_into_wrong_place_type_fails_loudly() {
    // A Closure aggregate builds a closure value; storing it into a
    // tuple-typed place is a type mismatch, reported at execution.
    // (Positive closure coverage lives in `callable_tests.rs`.)
    let body = mir::Body::new(
        decls(&[mir::Ty::Tuple(vec![u32_ty()])]),
        0,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Closure {
                        def: mir::DefId::new(0, 1),
                        generic_args: vec![],
                    }),
                    vec![const_u32(1)],
                ),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("mk", body);
    let err = interpret_mir(&prog, "mk", vec![]).expect_err("Closure aggregate must fail");
    assert!(
        matches!(err, MirRunError::Interp(InterpError::TypeMismatch { .. })),
        "expected TypeMismatch, got {err}"
    );
}

#[test]
fn duplicate_function_names_are_rejected() {
    let mut prog = MirProgram::new();
    prog.add("dup", add_plain_body());
    prog.add("dup", add_plain_body());
    let err = convert_program(&prog).expect_err("duplicate names must fail");
    assert!(matches!(err, ConvertError::DuplicateFunction { .. }));
}

// ----- unit return place: real MIR never writes `_0` of a `()` fn -----------

/// `fn unit_main()` shaped like real rustc MIR for a unit-returning
/// function (a counting loop, then a bare `return`): `_0: ()` is never
/// assigned anywhere — rustc emits no write for the unit return place.
fn unit_main_body() -> mir::Body {
    mir::Body::new(
        decls(&[mir::Ty::unit(), i32_ty(), i32_ty(), bool_ty()]),
        0,
        vec![
            // bb0: _1 = 0 (sum); _2 = 0 (i); goto bb1
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(0))),
                    assign(place(2), mir::Rvalue::Use(const_i32(0))),
                ],
                goto(1),
            ),
            // bb1: _3 = Lt(copy _2, 100); switchInt(move _3) -> [0: bb3, otherwise: bb2]
            block(
                vec![assign(place(3), binop(mir::BinOp::Lt, copy(2), const_i32(100)))],
                switch_bool(move_(3), 3, 2),
            ),
            // bb2: _1 = Add(copy _1, copy _2); _2 = Add(copy _2, 1); goto bb1
            block(
                vec![
                    assign(place(1), binop(mir::BinOp::Add, copy(1), copy(2))),
                    assign(place(2), binop(mir::BinOp::Add, copy(2), const_i32(1))),
                ],
                goto(1),
            ),
            // bb3: return — `_0` was never written, exactly as in real MIR.
            block(vec![], ret()),
        ],
    )
}

#[test]
fn unit_returning_function_never_writes_return_place() {
    let mut prog = MirProgram::new();
    prog.add("unit_main", unit_main_body());
    let v = interpret_mir(&prog, "unit_main", vec![])
        .expect("a unit-returning function must run to its bare `return`");
    assert_eq!(v, Value::Unit);
}

#[test]
fn uninitialized_non_unit_return_place_is_still_an_error() {
    // `fn missing() -> i32 { return }` with `_0` never written: only the
    // unit return place may be read unwritten; any other type is a
    // genuine uninitialized read and must stay an error.
    let body = mir::Body::new(decls(&[i32_ty()]), 0, vec![block(vec![], ret())]);
    let mut prog = MirProgram::new();
    prog.add("missing", body);
    let err = interpret_mir(&prog, "missing", vec![])
        .expect_err("unwritten i32 return place must not read as a value");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UninitializedRead { .. })
        ),
        "expected UninitializedRead, got {err}"
    );
}
// ----- literal operands of binary operations --------------------------------

/// `fn lit_add() -> i32 { 40 + 2 }`: both operands are literals, so the
/// only typing context is the checked pair's destination type
/// `(i32, bool)` — the operand type is its first element.
/// Locals: _0: i32, _1: (i32, bool).
fn lit_add_body() -> mir::Body {
    let pair_ty = mir::Ty::Tuple(vec![i32_ty(), bool_ty()]);
    mir::Body::new(
        decls(&[i32_ty(), pair_ty]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    mir::Rvalue::CheckedBinaryOp(
                        mir::BinOp::Add,
                        Box::new((const_i32(40), const_i32(2))),
                    ),
                )],
                mir::TerminatorKind::Assert {
                    cond: mir::Operand::Move(field(1, 1, bool_ty())),
                    expected: false,
                    msg: Box::new(mir::AssertMessage::Overflow(
                        mir::BinOp::Add,
                        Box::new(const_i32(40)),
                        Box::new(const_i32(2)),
                    )),
                    target: mir::BasicBlock(1),
                    unwind: mir::UnwindAction::Continue,
                },
            ),
            block(
                vec![assign(place(0), mir::Rvalue::Use(move_field(1, 0)))],
                ret(),
            ),
        ],
    )
}

#[test]
fn literal_checked_add_uses_the_pair_element_type() {
    let mut prog = MirProgram::new();
    prog.add("lit_add", lit_add_body());
    let v = interpret_mir(&prog, "lit_add", vec![])
        .expect("literal 40 + 2 must convert with i32 operand typing and run");
    assert_eq!(v, Value::Int(42));
}

/// The block shape rustc 1.100.0 emits for `lhs % rhs` on literals: the
/// division-by-zero and overflow conditions are expanded into explicit
/// `Eq` comparisons against `0`, `-1` and the type minimum, guarded by
/// `Assert`s, before the plain `Rem` in the final block.
/// Locals: _0: i32 (return), _1: i32 (result), _2..=_5: bool checks.
fn lit_rem_body(lhs: i32, rhs: i32) -> mir::Body {
    fn assert_not(
        cond: mir::Operand,
        msg: mir::AssertMessage,
        target: u32,
    ) -> mir::TerminatorKind {
        mir::TerminatorKind::Assert {
            cond,
            expected: false,
            msg: Box::new(msg),
            target: mir::BasicBlock(target),
            unwind: mir::UnwindAction::Continue,
        }
    }
    mir::Body::new(
        decls(&[i32_ty(), i32_ty(), bool_ty(), bool_ty(), bool_ty(), bool_ty()]),
        0,
        vec![
            // bb0: _2 = Eq(rhs, 0); assert(!move _2, RemainderByZero) -> bb1
            block(
                vec![assign(place(2), binop(mir::BinOp::Eq, const_i32(rhs), const_i32(0)))],
                assert_not(
                    move_(2),
                    mir::AssertMessage::RemainderByZero(Box::new(const_i32(lhs))),
                    1,
                ),
            ),
            // bb1: _3 = Eq(rhs, -1); _4 = Eq(lhs, i32::MIN);
            //      _5 = BitAnd(move _3, move _4);
            //      assert(!move _5, Overflow(Rem)) -> bb2
            block(
                vec![
                    assign(place(3), binop(mir::BinOp::Eq, const_i32(rhs), const_i32(-1))),
                    assign(
                        place(4),
                        binop(mir::BinOp::Eq, const_i32(lhs), const_i32(i32::MIN)),
                    ),
                    assign(place(5), binop(mir::BinOp::BitAnd, move_(3), move_(4))),
                ],
                assert_not(
                    move_(5),
                    mir::AssertMessage::Overflow(
                        mir::BinOp::Rem,
                        Box::new(const_i32(lhs)),
                        Box::new(const_i32(rhs)),
                    ),
                    2,
                ),
            ),
            // bb2: _1 = Rem(lhs, rhs); _0 = move _1; return
            block(
                vec![
                    assign(place(1), binop(mir::BinOp::Rem, const_i32(lhs), const_i32(rhs))),
                    assign(place(0), mir::Rvalue::Use(move_(1))),
                ],
                ret(),
            ),
        ],
    )
}

#[test]
fn literal_rem_checks_do_not_misfire() {
    // `17 % 5`: neither the by-zero check (5 == 0) nor the overflow
    // check (5 == -1 && 17 == i32::MIN) holds, so execution must reach
    // the `Rem` and produce 2 — not a spurious overflow panic.
    let mut prog = MirProgram::new();
    prog.add("lit_rem", lit_rem_body(17, 5));
    let v = interpret_mir(&prog, "lit_rem", vec![]).expect("17 % 5 must run");
    assert_eq!(v, Value::Int(2));
}

#[test]
fn literal_rem_min_by_neg_one_still_panics() {
    // `i32::MIN % -1` is the one genuinely overflowing remainder: the
    // expanded checks must still catch it and panic in the guest.
    let mut prog = MirProgram::new();
    prog.add("lit_rem_ovf", lit_rem_body(i32::MIN, -1));
    let err = interpret_mir(&prog, "lit_rem_ovf", vec![])
        .expect_err("i32::MIN % -1 must panic via the overflow Assert");
    assert!(
        matches!(err, MirRunError::Interp(InterpError::Panic { .. })),
        "expected guest panic, got {err}"
    );
}

// ----- ADT aggregate operands: declared field types --------------------------

const POINT_DEF: mir::DefId = mir::DefId::new(0, 21);
const MESSAGE_DEF: mir::DefId = mir::DefId::new(0, 22);
const OPTION_DEF: mir::DefId = mir::DefId::new(0, 23);
const PAIR_DEF: mir::DefId = mir::DefId::new(0, 24);
const MAKE_DEF: mir::DefId = mir::DefId::new(0, 25);
const NONNULL_DEF: mir::DefId = mir::DefId::new(0, 101);
const UNIQUE_DEF: mir::DefId = mir::DefId::new(0, 102);
const BOX_DEF: mir::DefId = mir::DefId::new(0, 103);


fn adt_aggregate(
    def: mir::DefId,
    variant: u32,
    generic_args: Vec<mir::Ty>,
    ops: Vec<mir::Operand>,
) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def,
            variant_index: variant,
            generic_args,
            user_ty: None,
            field_index: None,
        }),
        ops,
    )
}

fn variant_field(idx: u32, variant: u32, f: u32, ty: mir::Ty) -> mir::Place {
    place(idx).downcast(None, variant).field(f, ty)
}

/// `struct Point { x: i32, y: i32 }`
/// `fn main() -> i32 { _1 = Point { x: 3, y: -4 };
///   _2 = Add(copy _1.0, 10); _3 = Mul(copy _1.1, 2);
///   _0 = Add(copy _2, copy _3); return }` = 13 + -8 = 5.
///
/// The aggregate operands are bare scalars with no type of their own:
/// typed by the no-context rule, the non-negative `3` would be stored
/// unsigned and the first read of `_1.0` would mismatch its declared
/// `i32`. The declared field types (carried by the `Field`
/// projections below) are what the operands must be typed by.
#[test]
fn struct_i32_fields_keep_their_signed_tag() {
    let point_ty = mir::Ty::Adt(POINT_DEF, vec![]);
    let body = mir::Body::new(
        decls(&[i32_ty(), point_ty, i32_ty(), i32_ty()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    adt_aggregate(POINT_DEF, 0, vec![], vec![const_i32(3), const_i32(-4)]),
                ),
                assign(
                    place(2),
                    binop(
                        mir::BinOp::Add,
                        mir::Operand::Copy(field(1, 0, i32_ty())),
                        const_i32(10),
                    ),
                ),
                assign(
                    place(3),
                    binop(
                        mir::BinOp::Mul,
                        mir::Operand::Copy(field(1, 1, i32_ty())),
                        const_i32(2),
                    ),
                ),
                assign(place(0), binop(mir::BinOp::Add, copy(2), copy(3))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    let v = interpret_mir(&prog, "main", vec![]).expect("struct i32 fields must run");
    assert_eq!(v, Value::Int(5));
}

/// `enum Message { Code(i32) }` (variant 2 here), built in one
/// function and projected in another:
/// `fn make() -> Message { _0 = Message::Code(5); return }`
/// `fn main() -> i32 { _1 = call make();
///   _0 = Mul(copy (_1 as Code).0, 10); return }` = 50.
///
/// The aggregate lives only in `make`, the `Downcast`/`Field`
/// projections only in `main`: the field's declared type has to be
/// recovered program-wide, not per body. (The payload is positive on
/// purpose: a negative literal would be tagged signed even by the
/// no-context fallback rule and could not catch the bug.)
#[test]
fn enum_i32_payload_keeps_its_signed_tag_across_bodies() {
    let message_ty = mir::Ty::Adt(MESSAGE_DEF, vec![]);
    let make_body = mir::Body::new(
        decls(&[message_ty.clone()]),
        0,
        vec![block(
            vec![assign(
                place(0),
                adt_aggregate(MESSAGE_DEF, 2, vec![], vec![const_i32(5)]),
            )],
            ret(),
        )],
    );
    let main_body = mir::Body::new(
        decls(&[i32_ty(), message_ty]),
        0,
        vec![
            block(
                vec![],
                mir::TerminatorKind::Call {
                    func: fndef_operand(MAKE_DEF),
                    args: vec![],
                    destination: place(1),
                    target: Some(mir::BasicBlock(1)),
                    unwind: mir::UnwindAction::Continue,
                    call_source: mir::CallSource::Normal,
                },
            ),
            block(
                vec![assign(
                    place(0),
                    binop(
                        mir::BinOp::Mul,
                        mir::Operand::Copy(variant_field(1, 2, 0, i32_ty())),
                        const_i32(10),
                    ),
                )],
                ret(),
            ),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add_with_def_id("make", MAKE_DEF, make_body);
    prog.add("main", main_body);
    let v = interpret_mir(&prog, "main", vec![]).expect("enum i32 payload must run");
    assert_eq!(v, Value::Int(50));
}

/// `fn main() -> i32 { _1 = Some(40);
///   _0 = Add(copy (_1 as Some).0, 2); return }` = 42.
///
/// The payload's declared type is the ADT instantiation's generic
/// argument (`Option<i32>`): a constant payload must be stored signed.
#[test]
fn option_i32_constant_payload_keeps_its_signed_tag() {
    let option_ty = mir::Ty::Adt(OPTION_DEF, vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[i32_ty(), option_ty]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    adt_aggregate(OPTION_DEF, 1, vec![i32_ty()], vec![const_i32(40)]),
                ),
                assign(
                    place(0),
                    binop(
                        mir::BinOp::Add,
                        mir::Operand::Copy(variant_field(1, 1, 0, i32_ty())),
                        const_i32(2),
                    ),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    let v = interpret_mir(&prog, "main", vec![]).expect("Option<i32> payload must run");
    assert_eq!(v, Value::Int(42));
}

/// `struct Pair<T> { a: T, b: T }` instantiated with both `i32` and
/// `u32` in one program: each instantiation's fields keep their own
/// declared tag (the recovery key includes the generic arguments).
#[test]
fn generic_struct_instantiations_keep_separate_field_types() {
    // `fn sum_i() -> i32 { _1 = Pair { a: 5, b: -2 };
    //   _0 = Add(copy _1.0, copy _1.1); return }` = 3
    let pair_i_ty = mir::Ty::Adt(PAIR_DEF, vec![i32_ty()]);
    let sum_i_body = mir::Body::new(
        decls(&[i32_ty(), pair_i_ty]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    adt_aggregate(PAIR_DEF, 0, vec![i32_ty()], vec![const_i32(5), const_i32(-2)]),
                ),
                assign(
                    place(0),
                    binop(
                        mir::BinOp::Add,
                        mir::Operand::Copy(field(1, 0, i32_ty())),
                        mir::Operand::Copy(field(1, 1, i32_ty())),
                    ),
                ),
            ],
            ret(),
        )],
    );
    // `fn sum_u() -> u32 { _1 = Pair { a: 5, b: 2 };
    //   _0 = Add(copy _1.0, copy _1.1); return }` = 7
    let pair_u_ty = mir::Ty::Adt(PAIR_DEF, vec![u32_ty()]);
    let sum_u_body = mir::Body::new(
        decls(&[u32_ty(), pair_u_ty]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    adt_aggregate(PAIR_DEF, 0, vec![u32_ty()], vec![const_u32(5), const_u32(2)]),
                ),
                assign(
                    place(0),
                    binop(
                        mir::BinOp::Add,
                        mir::Operand::Copy(field(1, 0, u32_ty())),
                        mir::Operand::Copy(field(1, 1, u32_ty())),
                    ),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("sum_i", sum_i_body);
    prog.add("sum_u", sum_u_body);
    assert_eq!(
        interpret_mir(&prog, "sum_i", vec![]).expect("Pair<i32> must run"),
        Value::Int(3)
    );
    assert_eq!(
        interpret_mir(&prog, "sum_u", vec![]).expect("Pair<u32> must run"),
        Value::Uint(7)
    );
}
// ----- SwitchInt keys and Discriminant values -------------------------------

/// `fn pick(_1: i32) -> i32`, switching on `_1` with branch keys in
/// the raw MIR storage form: the `-2` key is stored as the
/// zero-extended bit pattern 4294967294.
fn pick_body() -> mir::Body {
    let switch = mir::TerminatorKind::SwitchInt {
        discr: copy(1),
        targets: mir::SwitchTargets::new(
            vec![4294967294, 3],
            vec![mir::BasicBlock(1), mir::BasicBlock(2)],
            mir::BasicBlock(3),
        ),
    };
    mir::Body::new(
        decls(&[i32_ty(), i32_ty()]),
        1,
        vec![
            block(vec![], switch),
            block(
                vec![assign(place(0), mir::Rvalue::Use(const_i32(300)))],
                ret(),
            ),
            block(
                vec![assign(place(0), mir::Rvalue::Use(const_i32(100)))],
                ret(),
            ),
            block(
                vec![assign(place(0), mir::Rvalue::Use(const_i32(0)))],
                ret(),
            ),
        ],
    )
}

#[test]
fn switch_int_negative_key_hits_its_branch() {
    let mut prog = MirProgram::new();
    prog.add("pick", pick_body());
    for (input, expected) in [(-2, 300), (3, 100), (9, 0)] {
        let v = interpret_mir(&prog, "pick", vec![Value::Int(input)]).expect("pick runs");
        assert_eq!(v, Value::Int(expected), "pick({input})");
    }
}

/// `fn discr() -> i32 { _1 = const E::<variant>; _0 = Discriminant(_1); return }`
/// for an enum whose recorded discriminants are `[3, 7, -2]`.
fn discr_body(variant: u32) -> mir::Body {
    let adt = mir::Ty::Adt(mir::DefId::new(0, 42), vec![]);
    let konst = mir::Operand::Constant(Box::new(mir::Constant::from_value(
        mir::ConstValue::Adt {
            variant,
            fields: vec![],
        },
    )));
    mir::Body::new(
        decls(&[i32_ty(), adt]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(konst)),
                assign(place(0), mir::Rvalue::Discriminant(place(1))),
            ],
            ret(),
        )],
    )
}

#[test]
fn discriminant_rvalue_yields_the_recorded_value() {
    let mut prog = MirProgram::new();
    prog.adt_discriminants
        .insert(mir::DefId::new(0, 42), vec![3, 7, -2]);
    prog.add("discr_c", discr_body(2));
    prog.add("discr_a", discr_body(0));
    // Variant index 2 has discriminant -2; the index itself (2) or
    // the zero-extended pattern would both be wrong answers here.
    let v = interpret_mir(&prog, "discr_c", vec![]).expect("discr_c runs");
    assert_eq!(v, Value::Int(-2));
    let v = interpret_mir(&prog, "discr_a", vec![]).expect("discr_a runs");
    assert_eq!(v, Value::Int(3));
}

// ----- raw pointers: creation shares the reference representation ------------

#[test]
fn raw_ptr_creation_reads_and_writes_through() {
    // fn read_raw() -> i32 { _1 = 41; _2 = &raw const _1; return *_2 }
    let raw_ty = mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Not);
    let read_raw = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), raw_ty]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(41))),
                assign(
                    place(2),
                    mir::Rvalue::RawPtr(mir::Mutability::Not, place(1)),
                ),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(2).deref())),
                ),
            ],
            ret(),
        )],
    );
    // fn write_raw() -> i32 { _1 = 1; _2 = &raw mut _1; *_2 = 99; return _1 }
    let raw_mut_ty = mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Mut);
    let write_raw = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), raw_mut_ty]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(1))),
                assign(
                    place(2),
                    mir::Rvalue::RawPtr(mir::Mutability::Mut, place(1)),
                ),
                assign(place(2).deref(), mir::Rvalue::Use(const_i32(99))),
                assign(place(0), mir::Rvalue::Use(copy(1))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("read_raw", read_raw);
    prog.add("write_raw", write_raw);
    assert_eq!(
        interpret_mir(&prog, "read_raw", vec![]).expect("read_raw runs"),
        Value::Int(41)
    );
    assert_eq!(
        interpret_mir(&prog, "write_raw", vec![]).expect("write_raw runs"),
        Value::Int(99)
    );
}

// ----- CopyForDeref: read through a reference place, plain copy otherwise ----

#[test]
fn copy_for_deref_reads_through_reference_and_plain_place() {
    // fn cfd_ref() -> i32 { _1 = 7; _2 = &_1; return CopyForDeref(_2) }
    let ref_ty = mir::Ty::Ref(mir::Region::Erased, Box::new(i32_ty()), mir::Mutability::Not);
    let cfd_ref = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), ref_ty]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(7))),
                assign(
                    place(2),
                    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
                ),
                assign(place(0), mir::Rvalue::CopyForDeref(place(2))),
            ],
            ret(),
        )],
    );
    // fn cfd_plain() -> i32 { _1 = 9; return CopyForDeref(_1) }
    // (a non-reference place: CopyForDeref is a plain copy there)
    let cfd_plain = mir::Body::new(
        decls(&[i32_ty(), i32_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(9))),
                assign(place(0), mir::Rvalue::CopyForDeref(place(1))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("cfd_ref", cfd_ref);
    prog.add("cfd_plain", cfd_plain);
    assert_eq!(
        interpret_mir(&prog, "cfd_ref", vec![]).expect("cfd_ref runs"),
        Value::Int(7)
    );
    assert_eq!(
        interpret_mir(&prog, "cfd_plain", vec![]).expect("cfd_plain runs"),
        Value::Int(9)
    );
}

// ----- Unreachable: reaching it is a guest panic in the interpreter ----------

#[test]
fn unreachable_terminator_is_a_guest_panic() {
    // fn f() -> i32 { unreachable }
    let body = mir::Body::new(
        decls(&[i32_ty()]),
        0,
        vec![block(vec![], mir::TerminatorKind::Unreachable)],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    match interpret_mir(&prog, "f", vec![]) {
        Err(MirRunError::Interp(InterpError::Panic { message, .. })) => {
            assert_eq!(message, "entered unreachable code");
        }
        Err(e) => panic!("expected guest panic from Unreachable, got error {e}"),
        Ok(v) => panic!("expected guest panic from Unreachable, got value {v:?}"),
    }
}

fn adt_i32_ty(def: mir::DefId) -> mir::Ty {
    mir::Ty::Adt(def, vec![i32_ty()])
}

fn aggregate_adt(def: mir::DefId, operand: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::Aggregate(
        Box::new(mir::AggregateKind::Adt {
            def,
            variant_index: 0,
            generic_args: vec![i32_ty()],
            user_ty: None,
            field_index: None,
        }),
        vec![operand],
    )
}

/// The shape rustc 1.100.0 emits around a `Box` deref: the box value
/// is the chain Box(Unique(NonNull(*const i32))); extracting the
/// pointer goes through a field read of the box followed by a
/// `BoxDerefTransmute` cast, and the pointee is read or written
/// through the resulting raw pointer.
///
/// Locals: _0 ret i32, _1 pointee i32, _2 *const i32, _3 NonNull,
/// _4 Unique, _5 Box, _6 Unique (field copy), _7 *const i32 (cast).
/// `tail` runs after the cast with the pointer in _7.
fn box_deref_body(tail: Vec<mir::Statement>) -> mir::Body {
    let mut stmts = vec![
        assign(place(1), mir::Rvalue::Use(const_i32(41))),
        assign(
            place(2),
            mir::Rvalue::RawPtr(mir::Mutability::Mut, place(1)),
        ),
        assign(place(3), aggregate_adt(NONNULL_DEF, move_(2))),
        assign(place(4), aggregate_adt(UNIQUE_DEF, move_(3))),
        assign(place(5), aggregate_adt(BOX_DEF, move_(4))),
        assign(
            place(6),
            mir::Rvalue::Use(mir::Operand::Copy(field(5, 0, adt_i32_ty(UNIQUE_DEF)))),
        ),
        assign(
            place(7),
            mir::Rvalue::Cast(mir::CastKind::BoxDerefTransmute, copy(6), raw_ptr_i32_ty()),
        ),
    ];
    stmts.extend(tail);
    mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            raw_ptr_i32_ty(),
            adt_i32_ty(NONNULL_DEF),
            adt_i32_ty(UNIQUE_DEF),
            adt_i32_ty(BOX_DEF),
            adt_i32_ty(UNIQUE_DEF),
            raw_ptr_i32_ty(),
        ]),
        0,
        vec![block(stmts, ret())],
    )
}

#[test]
fn box_deref_transmute_reads_pointee() {
    let body = box_deref_body(vec![assign(
        place(0),
        mir::Rvalue::Use(mir::Operand::Copy(place(7).deref())),
    )]);
    let mut prog = MirProgram::new();
    prog.add("unbox_read", body);
    let v = interpret_mir(&prog, "unbox_read", vec![]).expect("unbox_read must run");
    assert_eq!(v, Value::Int(41));
}

#[test]
fn box_deref_transmute_rejects_non_wrapper() {
    // Casting a plain integer with BoxDerefTransmute is a type error,
    // not a silent pass-through.
    let body = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), raw_ptr_i32_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(5))),
                assign(
                    place(2),
                    mir::Rvalue::Cast(mir::CastKind::BoxDerefTransmute, copy(1), raw_ptr_i32_ty()),
                ),
                assign(place(0), mir::Rvalue::Use(const_i32(0))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("bad_cast", body);
    let err = interpret_mir(&prog, "bad_cast", vec![])
        .expect_err("BoxDerefTransmute of an integer must fail");
    assert!(
        matches!(err, MirRunError::Interp(InterpError::TypeMismatch { .. })),
        "expected a type mismatch, got {err}"
    );
}

#[test]
fn box_deref_transmute_writes_pointee() {
    let body = box_deref_body(vec![
        assign(place(7).deref(), mir::Rvalue::Use(const_i32(42))),
        assign(place(0), mir::Rvalue::Use(copy(1))),
    ]);
    let mut prog = MirProgram::new();
    prog.add("unbox_write", body);
    let v = interpret_mir(&prog, "unbox_write", vec![]).expect("unbox_write must run");
    assert_eq!(v, Value::Int(42));
}

const BOX_NEW_DEF: mir::DefId = mir::DefId::new(0, 104);

/// A program with the names the driver registers for the Box
/// family. The builtin-box cast classification below keys off
/// these names, exactly as it does on driver-produced programs.
fn box_prog(body: mir::Body, name: &str) -> MirProgram {
    let mut prog = MirProgram::new();
    prog.add(name, body);
    prog.fn_names.insert(BOX_DEF, "std::boxed::Box".to_string());
    prog.fn_names.insert(UNIQUE_DEF, "std::ptr::Unique".to_string());
    prog.fn_names
        .insert(NONNULL_DEF, "std::ptr::NonNull".to_string());
    prog.fn_names
        .insert(BOX_NEW_DEF, "std::boxed::Box::<T>::new".to_string());
    prog
}

fn box_cast(operand: mir::Operand, dest: u32, target: mir::Ty) -> mir::Statement {
    assign(
        place(dest),
        mir::Rvalue::Cast(mir::CastKind::BoxDerefTransmute, operand, target),
    )
}

/// The shape the interpreter actually receives for
/// `let b = Box::new(41); *b` when the box comes from the
/// `Box::new` builtin: the builtin allocates the pointee in the
/// allocation store and the box value's single field is the
/// allocation pointer, so there is no Unique/NonNull wrapper
/// chain in the value, while rustc's deref elaboration still
/// reads the box's field 0 (the `Unique`) and casts it with
/// `BoxDerefTransmute`. Regression test: this used to fail with
/// "type mismatch in BoxDerefTransmute cast: expected pointer
/// wrapper ..., found signed integer" on every real program,
/// because the aggregate-built chain above was the only shape
/// exercised.
///
/// Locals: _0 ret i32, _1 Box<i32>, _2 *const i32; `tail` runs
/// after the cast with the extracted pointer in _2.
fn builtin_box_body(tail: Vec<mir::Statement>) -> mir::Body {
    let mut stmts = vec![box_cast(
        mir::Operand::Copy(field(1, 0, adt_i32_ty(UNIQUE_DEF))),
        2,
        raw_ptr_i32_ty(),
    )];
    stmts.extend(tail);
    mir::Body::new(
        decls(&[i32_ty(), adt_i32_ty(BOX_DEF), raw_ptr_i32_ty()]),
        0,
        vec![
            block(
                vec![],
                call(fndef_operand(BOX_NEW_DEF), const_i32(41), 1, 1),
            ),
            block(stmts, ret()),
        ],
    )
}

#[test]
fn box_deref_transmute_builtin_box_field_reads_pointee() {
    let body = builtin_box_body(vec![assign(
        place(0),
        mir::Rvalue::Use(mir::Operand::Copy(place(2).deref())),
    )]);
    let prog = box_prog(body, "unbox_builtin_read");
    let v = interpret_mir(&prog, "unbox_builtin_read", vec![])
        .expect("builtin box deref must run");
    assert_eq!(v, Value::Int(41));
}

#[test]
fn box_deref_transmute_builtin_box_field_writes_pointee() {
    // Write through the extracted pointer, then extract the
    // pointer from the box a second time and read the pointee
    // back through it: both extractions must name the box's one
    // allocation, so the second read observes the first write.
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            adt_i32_ty(BOX_DEF),
            raw_ptr_i32_ty(),
            raw_ptr_i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![],
                call(fndef_operand(BOX_NEW_DEF), const_i32(41), 1, 1),
            ),
            block(
                vec![
                    box_cast(
                        mir::Operand::Copy(field(1, 0, adt_i32_ty(UNIQUE_DEF))),
                        2,
                        raw_ptr_i32_ty(),
                    ),
                    assign(place(2).deref(), mir::Rvalue::Use(const_i32(42))),
                    box_cast(
                        mir::Operand::Copy(field(1, 0, adt_i32_ty(UNIQUE_DEF))),
                        3,
                        raw_ptr_i32_ty(),
                    ),
                    assign(
                        place(0),
                        mir::Rvalue::Use(mir::Operand::Copy(place(3).deref())),
                    ),
                ],
                ret(),
            ),
        ],
    );
    let prog = box_prog(body, "unbox_builtin_write");
    let v = interpret_mir(&prog, "unbox_builtin_write", vec![])
        .expect("builtin box write must run");
    assert_eq!(v, Value::Int(42));
}

#[test]
fn box_deref_transmute_whole_builtin_box() {
    // The cast operand is the box place itself (no field read).
    let body = mir::Body::new(
        decls(&[i32_ty(), adt_i32_ty(BOX_DEF), raw_ptr_i32_ty()]),
        0,
        vec![
            block(
                vec![],
                call(fndef_operand(BOX_NEW_DEF), const_i32(41), 1, 1),
            ),
            block(
                vec![
                    box_cast(copy(1), 2, raw_ptr_i32_ty()),
                    assign(
                        place(0),
                        mir::Rvalue::Use(mir::Operand::Copy(place(2).deref())),
                    ),
                ],
                ret(),
            ),
        ],
    );
    let prog = box_prog(body, "unbox_builtin_whole");
    let v = interpret_mir(&prog, "unbox_builtin_whole", vec![])
        .expect("whole-box cast must run");
    assert_eq!(v, Value::Int(41));
}

#[test]
fn box_deref_transmute_builtin_box_move_operand() {
    // A Move operand clears the field's storage while it is
    // evaluated; the extracted pointer names that same storage,
    // so the value must be restored for the read to see it.
    let body = mir::Body::new(
        decls(&[i32_ty(), adt_i32_ty(BOX_DEF), raw_ptr_i32_ty()]),
        0,
        vec![
            block(
                vec![],
                call(fndef_operand(BOX_NEW_DEF), const_i32(41), 1, 1),
            ),
            block(
                vec![
                    box_cast(
                        mir::Operand::Move(field(1, 0, adt_i32_ty(UNIQUE_DEF))),
                        2,
                        raw_ptr_i32_ty(),
                    ),
                    assign(
                        place(0),
                        mir::Rvalue::Use(mir::Operand::Copy(place(2).deref())),
                    ),
                ],
                ret(),
            ),
        ],
    );
    let prog = box_prog(body, "unbox_builtin_move");
    let v = interpret_mir(&prog, "unbox_builtin_move", vec![])
        .expect("move-operand cast must run");
    assert_eq!(v, Value::Int(41));
}

#[test]
fn box_deref_transmute_builtin_box_of_reference() {
    // Box<&i32>: the builtin box's single layer holds the pointee
    // — a reference — itself. The cast must yield a pointer TO
    // that reference, not the reference: reading through the
    // result twice reaches the i32.
    let ref_i32 = mir::Ty::Ref(mir::Region::Erased, Box::new(i32_ty()), mir::Mutability::Not);
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            ref_i32.clone(),
            mir::Ty::Adt(BOX_DEF, vec![ref_i32.clone()]),
            mir::Ty::RawPtr(Box::new(ref_i32.clone()), mir::Mutability::Mut),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(33))),
                    assign(
                        place(2),
                        mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
                    ),
                ],
                call(fndef_operand(BOX_NEW_DEF), copy(2), 3, 1),
            ),
            block(
                vec![
                    box_cast(
                        mir::Operand::Copy(field(
                            3,
                            0,
                            mir::Ty::Adt(UNIQUE_DEF, vec![ref_i32.clone()]),
                        )),
                        4,
                        mir::Ty::RawPtr(Box::new(ref_i32.clone()), mir::Mutability::Mut),
                    ),
                    assign(
                        place(0),
                        mir::Rvalue::Use(mir::Operand::Copy(place(4).deref().deref())),
                    ),
                ],
                ret(),
            ),
        ],
    );
    let prog = box_prog(body, "unbox_builtin_ref");
    let v = interpret_mir(&prog, "unbox_builtin_ref", vec![])
        .expect("box-of-reference cast must run");
    assert_eq!(v, Value::Int(33));
}

#[test]
fn box_deref_transmute_builtin_box_of_struct() {
    // Box<Point>: the value found under the cast is an ADT value
    // (the "found ADT value" failure shape). Fields of the
    // pointee are read through the extracted pointer.
    let point_ty = mir::Ty::Adt(POINT_DEF, vec![]);
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            point_ty.clone(),
            mir::Ty::Adt(BOX_DEF, vec![point_ty.clone()]),
            mir::Ty::RawPtr(Box::new(point_ty.clone()), mir::Mutability::Mut),
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Adt {
                            def: POINT_DEF,
                            variant_index: 0,
                            generic_args: vec![],
                            user_ty: None,
                            field_index: None,
                        }),
                        vec![const_i32(3), const_i32(4)],
                    ),
                )],
                call(fndef_operand(BOX_NEW_DEF), copy(1), 2, 1),
            ),
            block(
                vec![
                    box_cast(
                        mir::Operand::Copy(field(
                            2,
                            0,
                            mir::Ty::Adt(UNIQUE_DEF, vec![point_ty.clone()]),
                        )),
                        3,
                        mir::Ty::RawPtr(Box::new(point_ty.clone()), mir::Mutability::Mut),
                    ),
                    assign(
                        place(4),
                        mir::Rvalue::Use(mir::Operand::Copy(place(3).deref().field(0, i32_ty()))),
                    ),
                    assign(
                        place(0),
                        mir::Rvalue::Use(mir::Operand::Copy(place(3).deref().field(1, i32_ty()))),
                    ),
                    assign(place(0), binop(mir::BinOp::Add, copy(0), copy(4))),
                ],
                ret(),
            ),
        ],
    );
    let prog = box_prog(body, "unbox_builtin_struct");
    let v = interpret_mir(&prog, "unbox_builtin_struct", vec![])
        .expect("box-of-struct cast must run");
    assert_eq!(v, Value::Int(7));
}

fn usize_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::Usize)
}

/// `addr_of` helper body: locals _0 usize ret, _1 i32, _2 i32,
/// Builds a body that takes the addresses of two i32
/// locals — by transmute of a reference (`expose` false)
/// or by expose-provenance of a raw pointer (`expose`
/// true) — and returns 1 as a usize when both addresses
/// are non-zero, aligned to the pointee (4), and distinct,
/// and 0 otherwise. The checks run in the guest as a chain
/// of switch blocks.
/// locals: 0 usize (return), 1/2 i32, 3/4 &i32,
/// 5/6 *const i32 (expose variant only), 7/8 usize
/// (addresses), 9/10 usize (addresses & 3), 11..15 bool
/// (distinct, a1 aligned, a2 aligned, a1 non-zero,
/// a2 non-zero).
fn addr_pair_body(expose: bool) -> mir::Body {
    let ref_i32 = mir::Ty::Ref(mir::Region::Erased, Box::new(i32_ty()), mir::Mutability::Not);
    let raw_i32 = mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Not);
    let mut tys = vec![
        usize_ty(),
        i32_ty(),
        i32_ty(),
        ref_i32.clone(),
        ref_i32,
        raw_i32.clone(),
        raw_i32,
        usize_ty(),
        usize_ty(),
        usize_ty(),
        usize_ty(),
    ];
    for _ in 0..5 {
        tys.push(mir::Ty::Bool);
    }
    let mut stmts = vec![
        assign(place(1), mir::Rvalue::Use(const_i32(5))),
        assign(place(2), mir::Rvalue::Use(const_i32(7))),
        assign(
            place(3),
            mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
        ),
        assign(
            place(4),
            mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(2)),
        ),
    ];
    if expose {
        stmts.push(assign(
            place(5),
            mir::Rvalue::Cast(mir::CastKind::PtrToPtr, copy(3), raw_i32_ty()),
        ));
        stmts.push(assign(
            place(6),
            mir::Rvalue::Cast(mir::CastKind::PtrToPtr, copy(4), raw_i32_ty()),
        ));
        stmts.push(assign(
            place(7),
            mir::Rvalue::Cast(mir::CastKind::PointerExposeProvenance, copy(5), usize_ty()),
        ));
        stmts.push(assign(
            place(8),
            mir::Rvalue::Cast(mir::CastKind::PointerExposeProvenance, copy(6), usize_ty()),
        ));
    } else {
        stmts.push(assign(
            place(7),
            mir::Rvalue::Cast(mir::CastKind::Transmute, copy(3), usize_ty()),
        ));
        stmts.push(assign(
            place(8),
            mir::Rvalue::Cast(mir::CastKind::Transmute, copy(4), usize_ty()),
        ));
    }
    stmts.push(assign(place(9), binop(mir::BinOp::BitAnd, copy(7), const_int(3, 8))));
    stmts.push(assign(place(10), binop(mir::BinOp::BitAnd, copy(8), const_int(3, 8))));
    stmts.push(assign(place(11), binop(mir::BinOp::Ne, copy(7), copy(8))));
    stmts.push(assign(place(12), binop(mir::BinOp::Eq, copy(9), const_int(0, 8))));
    stmts.push(assign(place(13), binop(mir::BinOp::Eq, copy(10), const_int(0, 8))));
    stmts.push(assign(place(14), binop(mir::BinOp::Ne, copy(7), const_int(0, 8))));
    stmts.push(assign(place(15), binop(mir::BinOp::Ne, copy(8), const_int(0, 8))));
    // blocks: 0 = statements + switch on local 11;
    // 1..4 = switches on locals 12..15; 5 = success
    // (return 1); 6 = failure (return 0).
    let mut blocks = vec![block(stmts, switch_bool(copy(11), 6, 1))];
    for (i, local) in [12u32, 13, 14, 15].iter().enumerate() {
        blocks.push(block(vec![], switch_bool(copy(*local), 6, (i + 2) as u32)));
    }
    blocks.push(block(
        vec![assign(place(0), mir::Rvalue::Use(const_int(1, 8)))],
        ret(),
    ));
    blocks.push(block(
        vec![assign(place(0), mir::Rvalue::Use(const_int(0, 8)))],
        ret(),
    ));
    mir::Body::new(decls(&tys), 0, blocks)
}

#[test]
fn transmute_pointer_to_usize_is_aligned_stable_and_distinct() {
    // The alignment precondition checks rustc emits around raw
    // pointer dereferences transmute the pointer to usize and
    // test `addr & (align - 1) == 0`; the address must be
    // aligned to the pointee type (4 for an i32 place), stable
    // per target, and distinct per place.
    let mut prog = MirProgram::new();
    prog.add("addr_pair", addr_pair_body(false));
    let v = interpret_mir(&prog, "addr_pair", vec![]).expect("addr_pair must run");
    assert_eq!(
        v,
        Value::Uint(1),
        "addresses are non-zero, i32-aligned, and distinct"
    );
}

#[test]
fn transmute_int_to_int_reinterprets_bits() {
    // Since Lane FB an integer-to-integer transmute of the same
    // width is modelled as a bit reinterpretation: -7i32 read as
    // u32 is its two's-complement pattern.
    let body = mir::Body::new(
        decls(&[u32_ty()]),
        0,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Cast(mir::CastKind::Transmute, const_i32(-7), u32_ty()),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("int_transmute", body);
    let v = interpret_mir(&prog, "int_transmute", vec![])
        .expect("same-width int transmute must run");
    assert_eq!(v, Value::Uint(4_294_967_289));
}

#[test]
fn closure_aggregate_fails_at_execution_not_silently() {
    // Destination is a tuple type so conversion itself succeeds; the
    // Closure aggregate kind is what is unsupported.
    let body = mir::Body::new(
        decls(&[mir::Ty::Tuple(vec![u32_ty()])]),
        0,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Closure {
                        def: mir::DefId::new(0, 1),
                        generic_args: vec![],
                    }),
                    vec![const_u32(1)],
                ),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("mk", body);
    let err = interpret_mir(&prog, "mk", vec![]).expect_err("Closure aggregate must fail");
    // Since Lane AJ, closure aggregates convert into callable values;
    // assigning one into a tuple-typed place fails loudly with a type
    // mismatch instead of an unsupported-rvalue error.
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::TypeMismatch { .. })
        ),
        "expected TypeMismatch, got {err}"
    );
}

fn raw_ptr_i32_ty() -> mir::Ty {
    mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Mut)
}

// ----- shifts: the amount may have any integer type --------------------------

fn i8_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I8)
}
fn i16_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I16)
}
fn i64_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I64)
}
fn u16_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U16)
}
fn u64_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U64)
}

fn const_i8(v: i8) -> mir::Operand {
    const_int(v as u8 as u128, 1)
}
fn const_u64(v: u64) -> mir::Operand {
    const_int(v as u128, 8)
}

/// `fn f(_1: V, _2: A) -> V { _0 = <op>(copy _1, copy _2); return }`
fn shift_place_body(op: mir::BinOp, value_ty: mir::Ty, amount_ty: mir::Ty) -> mir::Body {
    mir::Body::new(
        decls(&[value_ty.clone(), value_ty, amount_ty]),
        2,
        vec![block(
            vec![assign(place(0), binop(op, copy(1), copy(2)))],
            ret(),
        )],
    )
}

/// `fn f(_1: V, _2: A) -> (V, bool)`: the checked form, returning
/// the `(value, overflowed)` pair itself.
fn shift_checked_body(op: mir::BinOp, value_ty: mir::Ty, amount_ty: mir::Ty) -> mir::Body {
    let pair_ty = mir::Ty::Tuple(vec![value_ty.clone(), bool_ty()]);
    mir::Body::new(
        decls(&[pair_ty.clone(), value_ty, amount_ty, pair_ty]),
        2,
        vec![block(
            vec![
                assign(
                    place(3),
                    mir::Rvalue::CheckedBinaryOp(op, Box::new((copy(1), copy(2)))),
                ),
                assign(place(0), mir::Rvalue::Use(copy(3))),
            ],
            ret(),
        )],
    )
}

fn run_shift(
    op: mir::BinOp,
    value_ty: mir::Ty,
    amount_ty: mir::Ty,
    value: Value,
    amount: Value,
) -> Result<Value, MirRunError> {
    let mut prog = MirProgram::new();
    prog.add("f", shift_place_body(op, value_ty, amount_ty));
    interpret_mir(&prog, "f", vec![value, amount])
}

#[test]
fn mixed_type_shift_amounts_run() {
    use mir::BinOp::{Shl, Shr};
    // Signed value, unsigned amount.
    assert_eq!(
        run_shift(Shl, i32_ty(), u8_ty(), Value::Int(3), Value::Uint(2)).expect("i32 << u8"),
        Value::Int(12)
    );
    // Arithmetic right shift of a negative value, unsigned amount.
    assert_eq!(
        run_shift(Shr, i32_ty(), u8_ty(), Value::Int(-16), Value::Uint(2)).expect("i32 >> u8"),
        Value::Int(-4)
    );
    // Unsigned value, signed amount.
    assert_eq!(
        run_shift(Shl, u32_ty(), i64_ty(), Value::Uint(7), Value::Int(3)).expect("u32 << i64"),
        Value::Uint(56)
    );
    // Logical right shift: the sign bit does not replicate.
    assert_eq!(
        run_shift(Shr, u32_ty(), i64_ty(), Value::Uint(0x8000_0000), Value::Int(4))
            .expect("u32 >> i64"),
        Value::Uint(0x0800_0000)
    );
    // Narrow value, wider signed amount.
    assert_eq!(
        run_shift(Shl, u8_ty(), i16_ty(), Value::Uint(1), Value::Int(7)).expect("u8 << i16"),
        Value::Uint(128)
    );
    assert_eq!(
        run_shift(Shr, u8_ty(), i16_ty(), Value::Uint(255), Value::Int(7)).expect("u8 >> i16"),
        Value::Uint(1)
    );
    // Wider value, narrower amount kinds in both directions.
    assert_eq!(
        run_shift(Shl, i64_ty(), u32_ty(), Value::Int(5), Value::Uint(4)).expect("i64 << u32"),
        Value::Int(80)
    );
    assert_eq!(
        run_shift(Shl, i32_ty(), u64_ty(), Value::Int(3), Value::Uint(2)).expect("i32 << u64"),
        Value::Int(12)
    );
}

#[test]
fn same_type_shifts_still_run() {
    use mir::BinOp::{Shl, Shr};
    assert_eq!(
        run_shift(Shl, i32_ty(), i32_ty(), Value::Int(3), Value::Int(2)).expect("i32 << i32"),
        Value::Int(12)
    );
    assert_eq!(
        run_shift(Shr, i32_ty(), i32_ty(), Value::Int(-16), Value::Int(2)).expect("i32 >> i32"),
        Value::Int(-4)
    );
    assert_eq!(
        run_shift(Shl, u32_ty(), u32_ty(), Value::Uint(7), Value::Uint(3)).expect("u32 << u32"),
        Value::Uint(56)
    );
}

#[test]
fn shift_amount_boundaries() {
    use mir::BinOp::{Shl, Shr};
    // Amount 0 is the identity, whatever its type.
    assert_eq!(
        run_shift(Shl, i32_ty(), u64_ty(), Value::Int(42), Value::Uint(0)).expect("<< 0"),
        Value::Int(42)
    );
    assert_eq!(
        run_shift(Shr, i32_ty(), u8_ty(), Value::Int(42), Value::Uint(0)).expect(">> 0"),
        Value::Int(42)
    );
    // Amount bits-1: the largest legal shift.
    assert_eq!(
        run_shift(Shr, i32_ty(), u16_ty(), Value::Int(-8), Value::Uint(31)).expect(">> 31"),
        Value::Int(-1)
    );
    assert_eq!(
        run_shift(Shr, u32_ty(), i8_ty(), Value::Uint(0x8000_0000), Value::Int(31))
            .expect("u32 >> 31i8"),
        Value::Uint(1)
    );
    assert_eq!(
        run_shift(Shl, u32_ty(), u8_ty(), Value::Uint(1), Value::Uint(30)).expect("<< 30"),
        Value::Uint(1 << 30)
    );
    // Bits shifted out of the width are simply lost, as natively:
    // only the amount can overflow a shift. Shifting into the sign
    // bit, and shifting significant bits out, both yield the
    // wrapped value (42u8 << 4 == 160, 10i8 << 4 == -96 natively).
    assert_eq!(
        run_shift(Shl, i32_ty(), u8_ty(), Value::Int(1), Value::Uint(31)).expect("1i32 << 31"),
        Value::Int(i32::MIN as i128)
    );
    assert_eq!(
        run_shift(Shl, u8_ty(), u8_ty(), Value::Uint(42), Value::Uint(4)).expect("42u8 << 4"),
        Value::Uint(160)
    );
    assert_eq!(
        run_shift(Shl, i8_ty(), u8_ty(), Value::Int(10), Value::Uint(4)).expect("10i8 << 4"),
        Value::Int(-96)
    );
    // Amount == bits is an overflowing shift. These bodies are
    // the plain form, which wraps the amount modulo the width
    // (amount % bits == 0 here, so each result is the value
    // itself); the panic of a checked build belongs to the
    // `Assert` over the checked form's flag, not to this form.
    assert_eq!(
        run_shift(Shl, i32_ty(), u8_ty(), Value::Int(1), Value::Uint(32)).expect("1i32 << 32"),
        Value::Int(1)
    );
    assert_eq!(
        run_shift(Shr, u8_ty(), i16_ty(), Value::Uint(1), Value::Int(8)).expect("1u8 >> 8"),
        Value::Uint(1)
    );
    assert_eq!(
        run_shift(Shl, u32_ty(), u32_ty(), Value::Uint(1), Value::Uint(32)).expect("1u32 << 32"),
        Value::Uint(1)
    );
}

#[test]
fn negative_and_huge_shift_amounts_wrap() {
    use mir::BinOp::{Shl, Shr};
    // A negative amount does not fit the u32 reading; u32::MAX
    // stands in for it, as in the same-type path. In the plain
    // form the stand-in amount wraps modulo the width:
    // u32::MAX % 32 == 31, so 1i32 << -1 == 1 << 31 and
    // 16u32 >> -1 == 16 >> 31 == 0.
    assert_eq!(
        run_shift(Shl, i32_ty(), i8_ty(), Value::Int(1), Value::Int(-1)).expect("1i32 << -1i8"),
        Value::Int(i32::MIN as i128)
    );
    assert_eq!(
        run_shift(Shr, u32_ty(), i64_ty(), Value::Uint(16), Value::Int(-1)).expect("16u32 >> -1i64"),
        Value::Uint(0)
    );
    assert_eq!(
        run_shift(Shl, i32_ty(), i32_ty(), Value::Int(1), Value::Int(-1)).expect("1i32 << -1i32"),
        Value::Int(i32::MIN as i128)
    );
    // An amount above u32::MAX reads the same way.
    assert_eq!(
        run_shift(Shl, i32_ty(), u64_ty(), Value::Int(1), Value::Uint(5_000_000_000))
            .expect("1i32 << 5000000000u64"),
        Value::Int(i32::MIN as i128)
    );
}

#[test]
fn checked_mixed_type_shifts() {
    use mir::BinOp::{Shl, Shr};
    let run = |op: mir::BinOp, value_ty: mir::Ty, amount_ty: mir::Ty, value: Value, amount: Value| {
        let mut prog = MirProgram::new();
        prog.add("f", shift_checked_body(op, value_ty, amount_ty));
        interpret_mir(&prog, "f", vec![value, amount]).expect("checked shift must run")
    };
    // In range: flag clear.
    assert_eq!(
        run(Shl, i32_ty(), u8_ty(), Value::Int(3), Value::Uint(4)),
        Value::Tuple(vec![Value::Int(48), Value::Bool(false)])
    );
    assert_eq!(
        run(Shr, i32_ty(), u8_ty(), Value::Int(-16), Value::Uint(2)),
        Value::Tuple(vec![Value::Int(-4), Value::Bool(false)])
    );
    // Overshift: the wrapped result shifts by amount % width, flag set.
    assert_eq!(
        run(Shl, i32_ty(), u8_ty(), Value::Int(1), Value::Uint(32)),
        Value::Tuple(vec![Value::Int(1), Value::Bool(true)])
    );
    assert_eq!(
        run(Shr, i32_ty(), u8_ty(), Value::Int(-8), Value::Uint(33)),
        Value::Tuple(vec![Value::Int(-4), Value::Bool(true)])
    );
    assert_eq!(
        run(Shr, u32_ty(), i16_ty(), Value::Uint(0x8000_0000), Value::Int(33)),
        Value::Tuple(vec![Value::Uint(0x4000_0000), Value::Bool(true)])
    );
    // Bits shifted out of a Shl do NOT set the flag in the checked
    // form either: only the amount overflows a shift (natively,
    // 3u8.checked_shl(7) is Some(128) and overflowing_shl reports
    // (128, false)).
    assert_eq!(
        run(Shl, u8_ty(), i64_ty(), Value::Uint(3), Value::Int(7)),
        Value::Tuple(vec![Value::Uint(128), Value::Bool(false)])
    );
}

#[test]
fn constant_shift_amounts_keep_their_own_type() {
    // `fn f(_1: i32) -> i32 { _0 = Shl(copy _1, <const amount>) }`
    let body_with_amount = |amount: mir::Operand| {
        mir::Body::new(
            decls(&[i32_ty(), i32_ty()]),
            1,
            vec![block(
                vec![assign(place(0), binop(mir::BinOp::Shl, copy(1), amount))],
                ret(),
            )],
        )
    };
    let mut prog = MirProgram::new();
    prog.add("small", body_with_amount(const_u8(2)));
    assert_eq!(
        interpret_mir(&prog, "small", vec![Value::Int(3)]).expect("const u8 amount"),
        Value::Int(12)
    );
    // A negative constant amount (const -1_i8) is an overflowing
    // shift; read under the value's i32 type it would have been
    // 255. The plain form wraps: the u32 reading of the amount is
    // u32::MAX (the stand-in for an amount that does not fit),
    // and u32::MAX % 32 == 31.
    let mut prog = MirProgram::new();
    prog.add("neg", body_with_amount(const_i8(-1)));
    assert_eq!(
        interpret_mir(&prog, "neg", vec![Value::Int(1)]).expect("const -1i8 amount wraps"),
        Value::Int(i32::MIN as i128)
    );
    // A constant amount wider than the value (const 4294967296_u64)
    // keeps its value: read in the i32 context it truncated to 0
    // and the shift silently succeeded (as 1 << 0) before; the
    // u32 reading stands in u32::MAX for it, exactly as for a
    // negative amount.
    let mut prog = MirProgram::new();
    prog.add("huge", body_with_amount(const_u64(1 << 32)));
    assert_eq!(
        interpret_mir(&prog, "huge", vec![Value::Int(1)]).expect("const 2^32 amount wraps"),
        Value::Int(i32::MIN as i128)
    );
    // A constant value shifted by a differently-typed place amount
    // keeps the value's type: `fn f(_1: u8) -> i32 { _0 = Shl(const 3_i32, copy _1) }`.
    let body = mir::Body::new(
        decls(&[i32_ty(), u8_ty()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                binop(mir::BinOp::Shl, const_i32(3), copy(1)),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![Value::Uint(2)]).expect("const value, u8 amount"),
        Value::Int(12)
    );
}

// ----- Transmute: pointer-to-integer only (Lane BO) ---------------------------

#[test]
fn transmute_pointer_to_int_yields_aligned_nonzero_address() {
    // The compiler's UB precondition checks transmute a raw pointer
    // to `usize` to test it for null and misalignment. The model
    // answers with the pointer's arithmetic address: non-zero and
    // aligned to the pointee type (4 for an i32 place).
    // fn main() -> i32 {
    //   _1 = 41; _2 = &raw mut _1;
    //   _3 = transmute::<*mut i32, usize>(_2);
    //   _4 = _3 & 3; _5 = _3 == 0;
    //   return if _5 { 1000 } else { _4 as i32 };   // 0
    // }
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            i32_ty(),
            raw_ptr_i32_ty(),
            usize_ty(),
            usize_ty(),
            mir::Ty::Bool,
            i32_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(41))),
                    assign(
                        place(2),
                        mir::Rvalue::RawPtr(mir::Mutability::Mut, place(1)),
                    ),
                    assign(
                        place(3),
                        mir::Rvalue::Cast(mir::CastKind::Transmute, copy(2), usize_ty()),
                    ),
                    assign(place(4), binop(mir::BinOp::BitAnd, copy(3), const_int(3, 8))),
                    assign(place(5), binop(mir::BinOp::Eq, copy(3), const_int(0, 8))),
                ],
                switch_bool(copy(5), 1, 2),
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Cast(mir::CastKind::IntToInt, copy(4), i32_ty()),
                )],
                ret(),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(const_i32(1000)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("ptr_addr", body);
    let v = interpret_mir(&prog, "ptr_addr", vec![]).expect("ptr_addr must run");
    assert_eq!(v, Value::Int(0));
}

#[test]
fn transmute_between_non_pointers_stays_unsupported() {
    // A width-changing transmute (integer to integer of another
    // width — a shape rustc itself rejects, exercised here as a
    // synthetic body) cannot be honoured; it stays an explicit
    // conversion error naming the cast. Same-width scalar
    // transmutes are modelled since Lane FB (see
    // `transmute_int_to_int_reinterprets_bits`), and integer to
    // pointer is modelled separately since Lane DN, as an
    // address-only reference value.
    for target in [i64_ty()] {
        let body = mir::Body::new(
            decls(&[i32_ty(), i32_ty(), target.clone()]),
            0,
            vec![block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(5))),
                    assign(
                        place(2),
                        mir::Rvalue::Cast(mir::CastKind::Transmute, copy(1), target),
                    ),
                    assign(place(0), mir::Rvalue::Use(const_i32(0))),
                ],
                ret(),
            )],
        );
        let mut prog = MirProgram::new();
        prog.add("bad_transmute", body);
        let err = interpret_mir(&prog, "bad_transmute", vec![])
            .expect_err("non-pointer transmute must fail");
        assert!(
            format!("{err}").contains("Cast Transmute"),
            "error names the cast: {err}"
        );
    }
}

// ----- PointerExposeProvenance (Lane BV) -----

/// `expose_addr_of` helper body: locals _0 usize ret, _1 i32,
/// _2 i32, _3 &i32, _4 *const i32, _5 usize; returns the
/// PointerExposeProvenance cast of a raw pointer to local `which`
#[test]
fn expose_provenance_pointer_to_usize_is_aligned_stable_and_distinct() {
    // `ptr as usize` (PointerExposeProvenance) yields the
    // pointer's address, like the pointer-to-integer
    // transmute: aligned to the pointee type, stable per
    // target, distinct per place. Alignment assertions in
    // guest code (`addr % align == 0`, e.g. std::mem::align_of
    // checks on raw pointers) read it.
    let mut prog = MirProgram::new();
    prog.add("expose_pair", addr_pair_body(true));
    let v = interpret_mir(&prog, "expose_pair", vec![]).expect("expose_pair must run");
    assert_eq!(
        v,
        Value::Uint(1),
        "addresses are non-zero, i32-aligned, and distinct"
    );
}

#[test]
fn expose_provenance_non_pointer_stays_unsupported() {
    // Only a pointer source is modelled; exposing an integer
    // keeps failing explicitly.
    let body = mir::Body::new(
        decls(&[usize_ty(), i32_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(7))),
                assign(
                    place(0),
                    mir::Rvalue::Cast(
                        mir::CastKind::PointerExposeProvenance,
                        copy(1),
                        usize_ty(),
                    ),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("bad_expose", body);
    let err = interpret_mir(&prog, "bad_expose", vec![])
        .expect_err("non-pointer expose must stay unsupported");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedRvalue { .. })
        ),
        "expected UnsupportedRvalue, got {err}"
    );
}

// ----- Reborrow: a bitwise copy of the pointer value at the place (Lane CO) --

#[test]
fn reborrow_copies_the_pointer_value() {
    // fn read_shared() -> i32 { _1 = 41; _2 = &mut _1; _3 = Reborrow(*const i32, _2); return *_3 }
    let raw_ty = mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Not);
    let ref_mut_ty = mir::Ty::Ref(mir::Region::Erased, Box::new(i32_ty()), mir::Mutability::Mut);
    let read_shared = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), ref_mut_ty.clone(), raw_ty]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(41))),
                assign(
                    place(2),
                    mir::Rvalue::Ref(
                        mir::Region::Erased,
                        mir::BorrowKind::Mut {
                            allow_two_phase_borrow: false,
                        },
                        place(1),
                    ),
                ),
                assign(
                    place(3),
                    mir::Rvalue::Reborrow(
                        mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Not),
                        mir::Mutability::Not,
                        place(2),
                    ),
                ),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(3).deref())),
                ),
            ],
            ret(),
        )],
    );
    // fn write_mut() -> i32 { _1 = 1; _2 = &mut _1; _3 = Reborrow(*mut i32, _2); *_3 = 99; return _1 }
    let raw_mut_ty = mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Mut);
    let write_mut = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), ref_mut_ty, raw_mut_ty]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(1))),
                assign(
                    place(2),
                    mir::Rvalue::Ref(
                        mir::Region::Erased,
                        mir::BorrowKind::Mut {
                            allow_two_phase_borrow: false,
                        },
                        place(1),
                    ),
                ),
                assign(
                    place(3),
                    mir::Rvalue::Reborrow(
                        mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Mut),
                        mir::Mutability::Mut,
                        place(2),
                    ),
                ),
                assign(place(3).deref(), mir::Rvalue::Use(const_i32(99))),
                assign(place(0), mir::Rvalue::Use(copy(1))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("read_shared", read_shared);
    prog.add("write_mut", write_mut);
    assert_eq!(
        interpret_mir(&prog, "read_shared", vec![]).expect("read_shared runs"),
        Value::Int(41)
    );
    assert_eq!(
        interpret_mir(&prog, "write_mut", vec![]).expect("write_mut runs"),
        Value::Int(99)
    );
}

// ----- ThreadLocalRef: per-thread storage seeded from the recorded
// ----- initializer; a static with no recorded definition is a named
// ----- explicit error, never a fabricated value -----

#[test]
fn thread_local_ref_without_a_definition_is_a_named_execution_error() {
    // fn f() -> i32 { _1 = ThreadLocalRef(some static); return *_1 }
    // The program records no thread-local definition for the static,
    // so evaluating the reference fails, naming the gap.
    let ref_ty = mir::Ty::Ref(mir::Region::Erased, Box::new(i32_ty()), mir::Mutability::Not);
    let body = mir::Body::new(
        decls(&[i32_ty(), ref_ty]),
        0,
        vec![block(
            vec![assign(
                place(1),
                mir::Rvalue::ThreadLocalRef(mir::DefId::new(0, 3)),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    match interpret_mir(&prog, "f", vec![]) {
        Err(MirRunError::Interp(InterpError::UnsupportedRvalue { kind })) => {
            assert!(
                kind.contains("ThreadLocalRef")
                    && kind.contains("no thread-local definition was collected"),
                "error names the gap: {kind}"
            );
        }
        Err(e) => panic!("expected UnsupportedRvalue for ThreadLocalRef, got {e}"),
        Ok(v) => panic!("expected UnsupportedRvalue for ThreadLocalRef, got {v:?}"),
    }
}

#[test]
fn thread_local_ref_reads_and_writes_the_seeded_thread_storage() {
    // fn read_seed() -> i32 { _1 = ThreadLocalRef(def); return *_1 }
    // fn write_then_read() -> i32 {
    //     _1 = ThreadLocalRef(def); *_1 = 99; return *_1
    // }
    // The recorded initializer (41) seeds the executing thread's
    // storage for the static: the first body reads the seed, the
    // second overwrites its own storage and reads the new value.
    let ref_ty = mir::Ty::Ref(mir::Region::Erased, Box::new(i32_ty()), mir::Mutability::Not);
    let def = mir::DefId::new(0, 3);
    let read_seed = mir::Body::new(
        decls(&[i32_ty(), ref_ty.clone()]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::ThreadLocalRef(def)),
                assign(place(0), mir::Rvalue::Use(copy_deref(1))),
            ],
            ret(),
        )],
    );
    let write_then_read = mir::Body::new(
        decls(&[i32_ty(), ref_ty]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::ThreadLocalRef(def)),
                assign(place(1).deref(), mir::Rvalue::Use(const_i32(99))),
                assign(place(0), mir::Rvalue::Use(copy_deref(1))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.thread_locals.insert(
        def,
        mir::ThreadLocalDef {
            ty: i32_ty(),
            init: mir::ConstValue::Scalar(mir::Scalar::Int { data: 41, size: 4 }),
        },
    );
    prog.add("read_seed", read_seed);
    prog.add("write_then_read", write_then_read);
    assert_eq!(
        interpret_mir(&prog, "read_seed", vec![]).expect("read_seed runs"),
        Value::Int(41)
    );
    assert_eq!(
        interpret_mir(&prog, "write_then_read", vec![]).expect("write_then_read runs"),
        Value::Int(99)
    );
}

// ----- StatementKind::Intrinsic: Assume and CopyNonOverlapping (Lane CO) -----

fn intrinsic_stmt(intr: mir::NonDivergingIntrinsic) -> mir::Statement {
    mir::Statement::new(mir::StatementKind::Intrinsic(Box::new(intr)))
}

#[test]
fn intrinsic_assume_true_is_a_noop_and_false_is_a_guest_panic() {
    let mk = |assumed: bool| {
        mir::Body::new(
            decls(&[i32_ty()]),
            0,
            vec![block(
                vec![
                    intrinsic_stmt(mir::NonDivergingIntrinsic::Assume(
                        mir::Operand::constant(mir::Constant::bool(assumed)),
                    )),
                    assign(place(0), mir::Rvalue::Use(const_i32(5))),
                ],
                ret(),
            )],
        )
    };
    let mut prog = MirProgram::new();
    prog.add("assume_true", mk(true));
    prog.add("assume_false", mk(false));
    assert_eq!(
        interpret_mir(&prog, "assume_true", vec![]).expect("assume(true) runs"),
        Value::Int(5)
    );
    match interpret_mir(&prog, "assume_false", vec![]) {
        Err(MirRunError::Interp(InterpError::Panic { message, .. })) => {
            assert!(message.contains("assume"), "panic names assume: {message}");
        }
        Err(e) => panic!("expected guest panic from assume(false), got {e}"),
        Ok(v) => panic!("expected guest panic from assume(false), got {v:?}"),
    }
}

#[test]
fn intrinsic_copy_nonoverlapping_copies_elements() {
    // fn copy_arr() -> i32 {
    //   _1 = [10, 20, 30]; _2 = [0, 0, 0];
    //   _3 = &raw const _1; _4 = &raw mut _2;
    //   copy_nonoverlapping(_3, _4, 3);
    //   return _2[0] + _2[2]
    // }
    let array_ty = mir::Ty::Array(Box::new(i32_ty()), mir::ArrayLen::Known(3));
    let elem = |local: u32, offset: u64| mir::Place {
        local: mir::Local(local),
        projection: vec![mir::PlaceElem::ConstantIndex {
            offset,
            min_length: 3,
            from_end: false,
        }],
    };
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            array_ty.clone(),
            array_ty,
            mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Not),
            mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Mut),
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
                assign(
                    place(2),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(i32_ty())),
                        vec![const_i32(0), const_i32(0), const_i32(0)],
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
                    count: const_int(3, 8),
                }),
                assign(
                    place(0),
                    binop(
                        mir::BinOp::Add,
                        mir::Operand::Copy(elem(2, 0)),
                        mir::Operand::Copy(elem(2, 2)),
                    ),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("copy_arr", body);
    assert_eq!(
        interpret_mir(&prog, "copy_arr", vec![]).expect("copy_arr runs"),
        Value::Int(40)
    );
}

// ----- InlineAsm: the no-op class steps over, the rest is a named error -----

fn asm_term(template: &str, options: &str, destination: Option<u32>) -> mir::TerminatorKind {
    mir::TerminatorKind::InlineAsm {
        template: template.to_string(),
        operands: Vec::new(),
        options: options.to_string(),
        destination: destination.map(mir::BasicBlock),
        targets: destination.map(mir::BasicBlock).into_iter().collect(),
        unwind: mir::UnwindAction::Continue,
    }
}

#[test]
fn inline_asm_noop_templates_step_to_the_destination() {
    for (template, options) in [
        ("", ""),
        ("nop", ""),
        ("[String(\"nop\")]", ""),
        ("/* placeholder */", "NOMEM"),
    ] {
        // fn f() -> i32 { _1 = 7; asm!(...); return _1 }
        let body = mir::Body::new(
            decls(&[i32_ty(), i32_ty()]),
            0,
            vec![
                block(
                    vec![assign(place(1), mir::Rvalue::Use(const_i32(7)))],
                    asm_term(template, options, Some(1)),
                ),
                block(
                    vec![assign(place(0), mir::Rvalue::Use(copy(1)))],
                    ret(),
                ),
            ],
        );
        let mut prog = MirProgram::new();
        prog.add("f", body);
        assert_eq!(
            interpret_mir(&prog, "f", vec![]).expect("no-op asm runs"),
            Value::Int(7),
            "template {template:?} options {options:?}"
        );
    }
}

#[test]
fn inline_asm_with_real_instructions_is_a_named_error() {
    // fn f() -> i32 { asm!("cpuid"); return 0 }
    let body = mir::Body::new(
        decls(&[i32_ty()]),
        0,
        vec![
            block(vec![], asm_term("cpuid", "", Some(1))),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    match interpret_mir(&prog, "f", vec![]) {
        Err(MirRunError::Interp(InterpError::UnsupportedTerminator { kind })) => {
            assert!(kind.contains("InlineAsm"), "error names InlineAsm: {kind}");
        }
        Err(e) => panic!("expected UnsupportedTerminator for InlineAsm, got {e}"),
        Ok(v) => panic!("expected UnsupportedTerminator for InlineAsm, got {v:?}"),
    }
}

// ----- InlineAsm: naked functions receive arguments in ABI registers -----

fn naked_identity_body() -> mir::Body {
    // fn f(x: u64) -> u64 { naked_asm!("mov rax, rdi", "ret") }
    mir::Body::new(
        decls(&[u64_ty(), u64_ty()]),
        1,
        vec![block(
            vec![],
            asm_prog_term("mov rax, rdi\nret", vec![], None),
        )],
    )
}

#[test]
fn naked_fn_with_parameters_and_no_recorded_abi_is_a_named_error() {
    // A hand-built program records no ABI for `f`: its parameter
    // cannot be seeded into the register the ABI would pass it
    // in, so the block is a named error rather than a program
    // that reads a zeroed register as if it were the argument.
    let mut prog = MirProgram::new();
    prog.add("f", naked_identity_body());
    match interpret_mir(&prog, "f", vec![Value::Uint(42)]) {
        Err(MirRunError::Interp(InterpError::UnsupportedTerminator { kind })) => {
            assert!(
                kind.contains("incoming ABI registers cannot be seeded"),
                "error names the unseedable parameters: {kind}"
            );
        }
        Err(e) => panic!("expected UnsupportedTerminator for the naked body, got {e}"),
        Ok(v) => panic!("expected UnsupportedTerminator for the naked body, got {v:?}"),
    }
}

#[test]
fn naked_fn_parameters_are_seeded_from_the_frame_arguments() {
    // With the function's ABI recorded (`extern "C"`, the
    // SysV64 family on this target), the argument local is
    // seeded into rdi before the steps run and the naked body
    // returns its argument.
    const F_DEF: mir::DefId = mir::DefId::new(0, 77);
    let mut prog = MirProgram::new();
    prog.add_with_def_id("f", F_DEF, naked_identity_body());
    prog.fn_abis.insert(F_DEF, mir::Abi::C);
    assert_eq!(
        interpret_mir(&prog, "f", vec![Value::Uint(42)]).expect("naked identity runs"),
        Value::Uint(42)
    );
}

#[test]
fn naked_fn_parameters_are_seeded_with_a_sysv64_recorded_abi() {
    // The driver records `extern "sysv64"` in its raw form
    // (`SysV64 { .. }`): the same SysV64 family as `extern "C"`,
    // so the argument local is seeded into rdi exactly as above.
    const F_DEF: mir::DefId = mir::DefId::new(0, 78);
    let mut prog = MirProgram::new();
    prog.add_with_def_id("f", F_DEF, naked_identity_body());
    prog.fn_abis.insert(
        F_DEF,
        mir::Abi::Other("SysV64 { unwind: false }".to_string()),
    );
    assert_eq!(
        interpret_mir(&prog, "f", vec![Value::Uint(42)]).expect("naked identity runs"),
        Value::Uint(42)
    );
}

#[test]
fn naked_fn_signed_return_is_read_at_the_return_shape() {
    // fn f() -> i64 { naked_asm!("mov rax, -1", "ret") }: the
    // return place is assigned from the register file directly,
    // with no `write_place` normalization in between, so the
    // bits are read as the declared type's two's-complement
    // value: -1, not the zero-extended 2^64 - 1.
    let body = mir::Body::new(
        decls(&[i64_ty()]),
        0,
        vec![block(
            vec![],
            asm_prog_term("mov rax, -1\nret", vec![], None),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("naked body runs"),
        Value::Int(-1)
    );
}

#[test]
fn naked_fn_incoming_rsp_read_is_a_named_error() {
    // fn f() -> u64 { naked_asm!("mov rax, rsp", "and rax, 15",
    // "ret") }: the incoming stack pointer is not modelled, so
    // the read is a named error rather than a computation with
    // the fabricated zero the register file starts with.
    let body = mir::Body::new(
        decls(&[u64_ty()]),
        0,
        vec![block(
            vec![],
            asm_prog_term("mov rax, rsp\nand rax, 15\nret", vec![], None),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    match interpret_mir(&prog, "f", vec![]) {
        Err(MirRunError::Interp(InterpError::UnsupportedTerminator { kind })) => {
            assert!(
                kind.contains("stack pointer"),
                "error names the incoming stack pointer: {kind}"
            );
        }
        Err(e) => panic!("expected UnsupportedTerminator for the rsp read, got {e}"),
        Ok(v) => panic!("expected UnsupportedTerminator for the rsp read, got {v:?}"),
    }
}

// ----- InlineAsm: the evaluable register class executes (Lane GI) -----

fn asm_op(
    kind: mir::AsmOperandKind,
    width: u32,
    in_value: Option<mir::Operand>,
    out_place: Option<mir::Place>,
) -> mir::InlineAsmOperand {
    mir::InlineAsmOperand {
        kind,
        reg_class: true,
        width,
        const_bits: None,
        in_value,
        out_place,
        reg_name: None,
        static_name: None,
        label_index: None,
        call_sig: None,
    }
}

fn asm_prog_term(
    template: &str,
    operands: Vec<mir::InlineAsmOperand>,
    destination: Option<u32>,
) -> mir::TerminatorKind {
    mir::TerminatorKind::InlineAsm {
        template: template.to_string(),
        operands,
        options: String::new(),
        destination: destination.map(mir::BasicBlock),
        targets: destination.map(mir::BasicBlock).into_iter().collect(),
        unwind: mir::UnwindAction::Continue,
    }
}

fn run_asm_body(name: &str, body: mir::Body) -> Value {
    let mut prog = MirProgram::new();
    prog.add(name, body);
    interpret_mir(&prog, name, vec![]).expect("asm body runs")
}

#[test]
fn inline_asm_mov_and_add_execute() {
    // _1: u32 = 40 (in), _2: u32 (out): mov {0:e}, {1:e}; add {0:e}, 2
    let body = mir::Body::new(
        decls(&[u32_ty(), u32_ty(), u32_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_u32(40)))],
                asm_prog_term(
                    "mov {0:e}, {1:e}\nadd {0:e}, 2",
                    vec![
                        asm_op(mir::AsmOperandKind::Out, 32, None, Some(place(2))),
                        asm_op(mir::AsmOperandKind::In, 32, Some(copy(1)), None),
                    ],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("f", body), Value::Uint(42));
}

#[test]
fn inline_asm_inout_sub_sign_extends() {
    // _1: i32 = 10 (inout), _2: i32 = 25 (in): sub {0}, {1} -> -15
    let body = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), i32_ty()]),
        0,
        vec![
            block(
                vec![
                    assign(place(1), mir::Rvalue::Use(const_i32(10))),
                    assign(place(2), mir::Rvalue::Use(const_i32(25))),
                ],
                asm_prog_term(
                    "sub {}, {}",
                    vec![
                        asm_op(
                            mir::AsmOperandKind::InOut,
                            32,
                            Some(copy(1)),
                            Some(place(1)),
                        ),
                        asm_op(mir::AsmOperandKind::In, 32, Some(copy(2)), None),
                    ],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("f", body), Value::Int(-15));
}

#[test]
fn inline_asm_shift_ops() {
    // i32: sar {0:e} -16 by 2 -> -4 (32-bit arithmetic shift).
    // The modifier-less form names the whole 64-bit register, so
    // sar {} on the zero-extended -16 gives 0x3FFFFFFC — the
    // native result for the same source.
    let sar_body = mir::Body::new(
        decls(&[i32_ty(), i32_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_i32(-16)))],
                asm_prog_term(
                    "sar {0:e}, 2",
                    vec![asm_op(
                        mir::AsmOperandKind::InOut,
                        32,
                        Some(copy(1)),
                        Some(place(1)),
                    )],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("sar", sar_body), Value::Int(-4));
    let sar_plain_body = mir::Body::new(
        decls(&[i32_ty(), i32_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_i32(-16)))],
                asm_prog_term(
                    "sar {}, 2",
                    vec![asm_op(
                        mir::AsmOperandKind::InOut,
                        32,
                        Some(copy(1)),
                        Some(place(1)),
                    )],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    assert_eq!(
        run_asm_body("sar_plain", sar_plain_body),
        Value::Int(1073741820)
    );
    let shl_body = mir::Body::new(
        decls(&[u32_ty(), u32_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_u32(3)))],
                asm_prog_term(
                    "shl {}, 4",
                    vec![asm_op(
                        mir::AsmOperandKind::InOut,
                        32,
                        Some(copy(1)),
                        Some(place(1)),
                    )],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("shl", shl_body), Value::Uint(48));
}

#[test]
fn inline_asm_subregister_modifiers() {
    // u16 0x1234: mov {0:l}, 0xAB -> 0x12AB; mov {0:h}, 0xCD -> 0xCDAB
    let inout16 = || {
        asm_op(
            mir::AsmOperandKind::InOut,
            16,
            Some(copy(1)),
            Some(place(1)),
        )
    };
    let body = mir::Body::new(
        decls(&[u16_ty(), u16_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_int(0x1234, 2)))],
                asm_prog_term("mov {0:l}, 0xAB", vec![inout16()], Some(1)),
            ),
            block(
                vec![],
                asm_prog_term("mov {0:h}, 0xCD", vec![inout16()], Some(2)),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("f", body), Value::Uint(0xCDAB));
}

#[test]
fn inline_asm_32bit_write_zero_extends() {
    // u64 u64::MAX: mov {0:e}, 1 -> 1 (the 32-bit write clears the
    // high half, as on x86-64).
    let body = mir::Body::new(
        decls(&[u64_ty(), u64_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_u64(u64::MAX)))],
                asm_prog_term(
                    "mov {0:e}, 1",
                    vec![asm_op(
                        mir::AsmOperandKind::InOut,
                        64,
                        Some(copy(1)),
                        Some(place(1)),
                    )],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("f", body), Value::Uint(1));
}

#[test]
fn inline_asm_const_operand_and_unary_ops() {
    // out u32 := const 7 via mov; u8 0x0F not -> 0xF0.
    let mut konst = asm_op(mir::AsmOperandKind::Const, 32, None, None);
    konst.const_bits = Some(7);
    let body = mir::Body::new(
        decls(&[u32_ty(), u32_ty()]),
        0,
        vec![
            block(
                vec![],
                asm_prog_term(
                    "mov {}, {1}",
                    vec![
                        asm_op(mir::AsmOperandKind::Out, 32, None, Some(place(1))),
                        konst,
                    ],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("konst", body), Value::Uint(7));
    let not_body = mir::Body::new(
        decls(&[u8_ty(), u8_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_int(0x0F, 1)))],
                asm_prog_term(
                    "not {}",
                    vec![asm_op(
                        mir::AsmOperandKind::InOut,
                        8,
                        Some(copy(1)),
                        Some(place(1)),
                    )],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(1)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("not", not_body), Value::Uint(0xF0));
}

#[test]
fn inline_asm_naked_ret_returns() {
    // _0 = 9; then a destination-less `ret` block ends the function.
    let body = mir::Body::new(
        decls(&[i32_ty()]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Use(const_i32(9)))],
            asm_prog_term("ret", vec![], None),
        )],
    );
    assert_eq!(run_asm_body("naked", body), Value::Int(9));
}

#[test]
fn inline_asm_memory_operand_is_a_named_error() {
    // mov {0}, [{1}] touches memory: outside the class, named error.
    let body = mir::Body::new(
        decls(&[u32_ty(), u32_ty(), u32_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_u32(1)))],
                asm_prog_term(
                    "mov {}, [{}]",
                    vec![
                        asm_op(mir::AsmOperandKind::Out, 32, None, Some(place(2))),
                        asm_op(mir::AsmOperandKind::In, 32, Some(copy(1)), None),
                    ],
                    Some(1),
                ),
            ),
            block(vec![], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    match interpret_mir(&prog, "f", vec![]) {
        Err(MirRunError::Interp(InterpError::UnsupportedTerminator { kind })) => {
            assert!(kind.contains("InlineAsm"), "error names InlineAsm: {kind}");
        }
        Err(e) => panic!("expected UnsupportedTerminator for InlineAsm, got {e}"),
        Ok(v) => panic!("expected UnsupportedTerminator for InlineAsm, got {v:?}"),
    }
}

fn asm_op_named(
    kind: mir::AsmOperandKind,
    width: u32,
    in_value: Option<mir::Operand>,
    out_place: Option<mir::Place>,
    reg_name: &str,
) -> mir::InlineAsmOperand {
    mir::InlineAsmOperand {
        kind,
        reg_class: false,
        width,
        const_bits: None,
        in_value,
        out_place,
        reg_name: Some(reg_name.to_string()),
        static_name: None,
        label_index: None,
        call_sig: None,
    }
}

#[test]
fn inline_asm_lateout_aliased_with_input_observes_the_seed() {
    // asm!("mov rcx, rax", in("rax") 42u64, lateout("rax") y,
    //      out("rcx") _): the template never writes rax, but the
    // lateout still observes the register's incoming content —
    // natively y == 42. (The driver reads a `lateout` as a plain
    // Out operand; the register name is the operand's identity,
    // at its 16-bit base form in the raw representation.)
    // _1: u64 = 42 (the input's value), _2: u64 (y), _3: u64
    // (the discarded rcx output).
    let body = mir::Body::new(
        decls(&[u64_ty(), u64_ty(), u64_ty(), u64_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_u64(42)))],
                asm_prog_term(
                    "mov rcx, rax",
                    vec![
                        asm_op_named(mir::AsmOperandKind::In, 64, Some(copy(1)), None, "ax"),
                        asm_op_named(mir::AsmOperandKind::Out, 64, None, Some(place(2)), "ax"),
                        asm_op_named(mir::AsmOperandKind::Out, 64, None, Some(place(3)), "cx"),
                    ],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("f", body), Value::Uint(42));
}

#[test]
fn inline_asm_lea_wraps_at_the_base_address_size() {
    // lea {0}, [ecx + 1] with rcx = 0xffff_ffff: the effective
    // address wraps at the base form's 32-bit address size
    // before it is zero-extended into the 64-bit destination —
    // natively 0, not 2^32.
    // _1: u64 = 0xffff_ffff (bound to rcx), _2: u64 (the result).
    let body32 = mir::Body::new(
        decls(&[u64_ty(), u64_ty(), u64_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_u64(0xffff_ffff)))],
                asm_prog_term(
                    "lea {0}, [ecx + 1]",
                    vec![
                        asm_op(mir::AsmOperandKind::Out, 64, None, Some(place(2))),
                        asm_op_named(mir::AsmOperandKind::In, 64, Some(copy(1)), None, "cx"),
                    ],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("lea32", body32), Value::Uint(0));

    // The 64-bit base form keeps the full sum: [rcx + 1] at the
    // same incoming value yields 2^32.
    let body64 = mir::Body::new(
        decls(&[u64_ty(), u64_ty(), u64_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_u64(0xffff_ffff)))],
                asm_prog_term(
                    "lea {0}, [rcx + 1]",
                    vec![
                        asm_op(mir::AsmOperandKind::Out, 64, None, Some(place(2))),
                        asm_op_named(mir::AsmOperandKind::In, 64, Some(copy(1)), None, "cx"),
                    ],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("lea64", body64), Value::Uint(0x1_0000_0000));

    // A placeholder base at the `:e` modifier is a 32-bit
    // address form too: [{1:e} + 1] wraps the same way.
    let body_ph = mir::Body::new(
        decls(&[u64_ty(), u64_ty(), u64_ty()]),
        0,
        vec![
            block(
                vec![assign(place(1), mir::Rvalue::Use(const_u64(0xffff_ffff)))],
                asm_prog_term(
                    "lea {0}, [{1:e} + 1]",
                    vec![
                        asm_op(mir::AsmOperandKind::Out, 64, None, Some(place(2))),
                        asm_op(mir::AsmOperandKind::In, 64, Some(copy(1)), None),
                    ],
                    Some(1),
                ),
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(copy(2)))], ret()),
        ],
    );
    assert_eq!(run_asm_body("lea_ph", body_ph), Value::Uint(0));
}

#[test]
fn naked_fn_signed_return_is_sign_extended_from_the_return_width() {
    // A naked body returns whatever it left in rax, read at the
    // function's return shape: for a signed return type the
    // register bits are the value's two's-complement form at
    // that width, so `mov rax, -1` in an i64 body is -1 — not
    // the zero-extended 2^64 - 1 the raw u64 bits widen to when
    // they are cast straight to i128.
    for (template, ty, expected) in [
        ("mov rax, -1\nret", i64_ty(), -1i128),
        ("mov eax, -1\nret", i32_ty(), -1),
        ("mov ax, -1\nret", i16_ty(), -1),
        ("mov al, -1\nret", i8_ty(), -1),
        ("mov eax, -128\nret", i32_ty(), -128),
        ("mov eax, 127\nret", i32_ty(), 127),
    ] {
        let body = mir::Body::new(
            decls(&[ty]),
            0,
            vec![block(vec![], asm_prog_term(template, vec![], None))],
        );
        assert_eq!(
            run_asm_body("neg", body),
            Value::Int(expected),
            "template {template:?}"
        );
    }
}

#[test]
fn naked_fn_read_of_the_incoming_stack_pointer_is_a_named_error() {
    // The incoming `rsp` is fixed by the caller's frame layout;
    // the register model does not represent it (the register
    // file starts at zero and no seed supplies it — arguments
    // never arrive in `rsp`), so a naked body that reads it
    // before establishing its own value is a named error rather
    // than a program that computes with a fabricated zero stack
    // pointer. Natively `mov rax, rsp; and rax, 15` yields 8
    // (the return address occupies the top of the stack).
    for template in [
        "mov rax, rsp\nand rax, 15\nret",
        "sub rsp, 8\nmov rax, rsp\nret",
        "lea rax, [rsp + 8]\nret",
    ] {
        let body = mir::Body::new(
            decls(&[u64_ty()]),
            0,
            vec![block(vec![], asm_prog_term(template, vec![], None))],
        );
        let mut prog = MirProgram::new();
        prog.add("f", body);
        match interpret_mir(&prog, "f", vec![]) {
            Err(MirRunError::Interp(InterpError::UnsupportedTerminator { kind })) => {
                assert!(
                    kind.contains("stack pointer"),
                    "error names the incoming stack pointer: {kind}"
                );
            }
            Err(e) => panic!("expected UnsupportedTerminator for the rsp read, got {e}"),
            Ok(v) => panic!("expected UnsupportedTerminator for the rsp read, got {v:?}"),
        }
    }
}

#[test]
fn naked_fn_read_of_a_self_written_stack_pointer_runs() {
    // A naked body that establishes its own `rsp` value first
    // reads that value back: only the *incoming* stack pointer
    // is unmodelled.
    const F_DEF: mir::DefId = mir::DefId::new(0, 78);
    let body = mir::Body::new(
        decls(&[u64_ty(), u64_ty()]),
        1,
        vec![block(
            vec![],
            asm_prog_term(
                "mov rax, rdi\nmov rsp, rax\nmov rax, rsp\nret",
                vec![],
                None,
            ),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add_with_def_id("f", F_DEF, body);
    prog.fn_abis.insert(F_DEF, mir::Abi::C);
    assert_eq!(
        interpret_mir(&prog, "f", vec![Value::Uint(42)]).expect("self-written rsp runs"),
        Value::Uint(42)
    );
}

// ----- transmute: the honestly representable shapes (Lane DN) ----------------

fn ref_i32_ty() -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(i32_ty()), mir::Mutability::Not)
}

fn raw_i32_ty() -> mir::Ty {
    mir::Ty::RawPtr(Box::new(i32_ty()), mir::Mutability::Not)
}

fn transmute(op: mir::Operand, target: mir::Ty) -> mir::Rvalue {
    mir::Rvalue::Cast(mir::CastKind::Transmute, op, target)
}

#[test]
fn transmute_ref_to_raw_ptr_reads_through() {
    // fn f() -> i32 { _1 = 41; _2 = &_1; _3 = transmute(_2) as *const i32; return *_3 }
    let body = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), ref_i32_ty(), raw_i32_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(41))),
                assign(
                    place(2),
                    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
                ),
                assign(place(3), transmute(copy(2), raw_i32_ty())),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(3).deref())),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("ref-to-pointer transmute runs"),
        Value::Int(41)
    );
}

#[test]
fn transmute_ptr_to_ptr_keeps_the_synthetic_address() {
    // fn f() -> bool { _1 = 41; _2 = &raw const _1; _3 = transmute(_2) as *const u8;
    //                  _4 = transmute(_3) as usize; _5 = transmute(_2) as usize;
    //                  return _4 == _5 }
    let raw_u8_ty = mir::Ty::RawPtr(Box::new(u8_ty()), mir::Mutability::Not);
    let body = mir::Body::new(
        decls(&[
            bool_ty(),
            i32_ty(),
            raw_i32_ty(),
            raw_u8_ty.clone(),
            usize_ty(),
            usize_ty(),
        ]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(41))),
                assign(
                    place(2),
                    mir::Rvalue::RawPtr(mir::Mutability::Not, place(1)),
                ),
                assign(place(3), transmute(copy(2), raw_u8_ty)),
                assign(place(4), transmute(copy(3), usize_ty())),
                assign(place(5), transmute(copy(2), usize_ty())),
                assign(place(0), binop(mir::BinOp::Eq, copy(4), copy(5))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("pointer-to-pointer transmute runs"),
        Value::Bool(true)
    );
}

#[test]
fn transmute_int_to_pointer_round_trips_through_usize() {
    // fn f(_1: usize) -> usize { _2 = transmute(_1) as *const i32;
    //                            _0 = transmute(_2) as usize; return }
    let body = mir::Body::new(
        decls(&[usize_ty(), usize_ty(), raw_i32_ty()]),
        1,
        vec![block(
            vec![
                assign(place(2), transmute(copy(1), raw_i32_ty())),
                assign(place(0), transmute(copy(2), usize_ty())),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    for input in [0u128, 4096, 12288] {
        let v = interpret_mir(&prog, "f", vec![Value::Uint(input)])
            .expect("integer-to-pointer transmute runs");
        assert_eq!(v, Value::Uint(input), "round trip of {input}");
    }
}

#[test]
fn transmute_int_to_pointer_deref_is_an_explicit_error() {
    // fn f() -> i32 { _1 = transmute(4096usize) as *const i32; return *_1 }
    // The fabricated address names no place: reading through it
    // fails loudly instead of inventing a pointee.
    let body = mir::Body::new(
        decls(&[i32_ty(), raw_i32_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), transmute(const_int(4096, 8), raw_i32_ty())),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(1).deref())),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    let err = interpret_mir(&prog, "f", vec![])
        .expect_err("dereferencing an integer-made pointer must fail");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::TypeMismatch { .. })
        ),
        "expected TypeMismatch, got {err}"
    );
}

const SMALL_DEF: mir::DefId = mir::DefId::new(0, 77);

fn small_prog() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.adt_discriminants.insert(SMALL_DEF, vec![1, 5, 9]);
    prog.adt_shapes.insert(
        SMALL_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Enum,
            variants: vec![vec![], vec![], vec![]],
        },
    );
    prog
}

fn small_ty() -> mir::Ty {
    mir::Ty::Adt(SMALL_DEF, vec![])
}

fn const_small(variant: u32) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Adt {
        variant,
        fields: vec![],
    }))
}

#[test]
fn transmute_enum_to_int_yields_the_discriminant() {
    // fn f() -> u8 { _1 = Small::B; _0 = transmute(_1) as u8; return }
    let body = mir::Body::new(
        decls(&[u8_ty(), small_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_small(1))),
                assign(place(0), transmute(copy(1), u8_ty())),
            ],
            ret(),
        )],
    );
    let mut prog = small_prog();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("enum-to-int transmute runs"),
        Value::Uint(5)
    );
}

#[test]
fn transmute_int_to_enum_round_trips() {
    // fn f() -> u8 { _1 = transmute(9u8) as Small; _0 = transmute(_1) as u8; return }
    let body = mir::Body::new(
        decls(&[u8_ty(), small_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), transmute(const_u8(9), small_ty())),
                assign(place(0), transmute(copy(1), u8_ty())),
            ],
            ret(),
        )],
    );
    let mut prog = small_prog();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("int-to-enum transmute runs"),
        Value::Uint(9)
    );
}

#[test]
fn transmute_int_to_enum_rejects_a_non_discriminant() {
    // fn f() -> Small { _0 = transmute(7u8) as Small; return }
    // 7 is not a discriminant of Small: an explicit error, never a
    // fabricated variant.
    let body = mir::Body::new(
        decls(&[small_ty()]),
        0,
        vec![block(
            vec![assign(place(0), transmute(const_u8(7), small_ty()))],
            ret(),
        )],
    );
    let mut prog = small_prog();
    prog.add("f", body);
    let err = interpret_mir(&prog, "f", vec![])
        .expect_err("a non-discriminant integer must not become an enum value");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::TypeMismatch { .. })
        ),
        "expected TypeMismatch, got {err}"
    );
}

const DN_NONNULL_DEF: mir::DefId = mir::DefId::new(0, 78);

fn nonnull_ty() -> mir::Ty {
    mir::Ty::Adt(DN_NONNULL_DEF, vec![i32_ty()])
}

fn wrapper_prog() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.adt_shapes.insert(
        DN_NONNULL_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![vec![mir::Ty::RawPtr(
                Box::new(mir::Ty::Param(0, "T".to_string())),
                mir::Mutability::Not,
            )]],
        },
    );
    prog
}

#[test]
fn transmute_pointer_wrapper_round_trips() {
    // fn f() -> i32 { _1 = 41; _2 = &raw const _1;
    //                 _3 = transmute(_2) as NonNull<i32>;
    //                 _4 = transmute(_3) as *const i32; return *_4 }
    let body = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), raw_i32_ty(), nonnull_ty(), raw_i32_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(41))),
                assign(
                    place(2),
                    mir::Rvalue::RawPtr(mir::Mutability::Not, place(1)),
                ),
                assign(place(3), transmute(copy(2), nonnull_ty())),
                assign(place(4), transmute(copy(3), raw_i32_ty())),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(4).deref())),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = wrapper_prog();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("wrapper transmutes run"),
        Value::Int(41)
    );
}

const PTR_ZST_DEF: mir::DefId = mir::DefId::new(0, 79);

#[test]
fn transmute_wrapper_with_a_zst_field_round_trips() {
    // The `PtrAndZst` shape: a payload pointer field plus a unit
    // field. Wrapping fills the unit field with its unique value.
    // fn f() -> i32 { _1 = 41; _2 = &raw const _1;
    //                 _3 = transmute(_2) as PtrAndZst<i32>;
    //                 _4 = transmute(_3) as *const i32; return *_4 }
    let pz_ty = mir::Ty::Adt(PTR_ZST_DEF, vec![i32_ty()]);
    let body = mir::Body::new(
        decls(&[i32_ty(), i32_ty(), raw_i32_ty(), pz_ty.clone(), raw_i32_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(41))),
                assign(
                    place(2),
                    mir::Rvalue::RawPtr(mir::Mutability::Not, place(1)),
                ),
                assign(place(3), transmute(copy(2), pz_ty)),
                assign(place(4), transmute(copy(3), raw_i32_ty())),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(4).deref())),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = wrapper_prog();
    prog.adt_shapes.insert(
        PTR_ZST_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![vec![
                mir::Ty::RawPtr(
                    Box::new(mir::Ty::Param(0, "T".to_string())),
                    mir::Mutability::Not,
                ),
                mir::Ty::Tuple(vec![]),
            ]],
        },
    );
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("wrapper-with-ZST transmutes run"),
        Value::Int(41)
    );
}

// ----- transmute: one-field ADT over an array <-> integer -------------------
//
// The `Simd<T, N>` storage shape: a one-field struct whose
// only field is an array whose length is a const parameter
// the type representation does not carry. Core's integer
// comparison of small arrays, inlined, transmutes the
// wrapper straight to the integer of the array's total
// width (and builds the wrapper back from an integer the
// same way). The specialised ADT <-> integer arms read
// neither an array field nor a non-integer payload, so the
// pair goes through the bit-stream reinterpretation, whose
// wrapper derivation takes the element count from the
// integer's width.

const SIMD_LIKE_DEF: mir::DefId = mir::DefId::new(0, 130);

fn simd_like_ty() -> mir::Ty {
    mir::Ty::Adt(SIMD_LIKE_DEF, vec![u8_ty()])
}

/// A program holding the `SimdLike<T, const N: usize>`
/// record: one variant whose only field is an array over
/// the type parameter, its length the const parameter,
/// carried as a named parameter.
fn simd_like_prog() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.adt_shapes.insert(
        SIMD_LIKE_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![vec![mir::Ty::Array(
                Box::new(mir::Ty::Param(0, "T".to_string())),
                mir::ArrayLen::Param("N".to_string()),
            )]],
        },
    );
    prog
}

#[test]
fn transmute_array_wrapper_to_int_reinterprets_bits() {
    // fn f() -> u16 { _1 = [1u8, 5]; _2 = SimdLike(_1);
    //                 _0 = transmute(_2) as u16; return }
    // The array's little-endian bit stream is the integer:
    // 1 | 5 << 8 = 1281.
    let arr_ty = mir::Ty::Array(Box::new(u8_ty()), mir::ArrayLen::Known(2));
    let body = mir::Body::new(
        decls(&[u16_ty(), arr_ty, simd_like_ty()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(u8_ty())),
                        vec![const_u8(1), const_u8(5)],
                    ),
                ),
                assign(
                    place(2),
                    adt_aggregate(
                        SIMD_LIKE_DEF,
                        0,
                        vec![u8_ty()],
                        vec![mir::Operand::Copy(place(1))],
                    ),
                ),
                assign(place(0), transmute(copy(2), u16_ty())),
            ],
            ret(),
        )],
    );
    let mut prog = simd_like_prog();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![])
            .expect("array-wrapper-to-int transmute runs"),
        Value::Uint(1281)
    );
}

#[test]
fn transmute_int_to_array_wrapper_reinterprets_bits() {
    // fn f() -> u8 { _2 = 1281u16; _1 = transmute(_2) as SimdLike;
    //                _0 = copy _1.0[1]; return }
    // The wrapper built from the integer holds the array
    // [1, 5] (little-endian), so element 1 is 5.
    let field_inst_ty = mir::Ty::Array(
        Box::new(u8_ty()),
        mir::ArrayLen::Param("N".to_string()),
    );
    let elem1 = mir::Place {
        local: mir::Local(1),
        projection: vec![
            mir::PlaceElem::Field(0, field_inst_ty),
            mir::PlaceElem::ConstantIndex {
                offset: 1,
                min_length: 2,
                from_end: false,
            },
        ],
    };
    let body = mir::Body::new(
        decls(&[u8_ty(), simd_like_ty(), u16_ty()]),
        0,
        vec![block(
            vec![
                assign(place(2), mir::Rvalue::Use(const_int(1281, 2))),
                assign(place(1), transmute(copy(2), simd_like_ty())),
                assign(place(0), mir::Rvalue::Use(mir::Operand::Copy(elem1))),
            ],
            ret(),
        )],
    );
    let mut prog = simd_like_prog();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![])
            .expect("int-to-array-wrapper transmute runs"),
        Value::Uint(5)
    );
}

#[test]
fn transmute_zst_to_zst_yields_the_unique_value() {
    // fn f() -> [i32; 0] { _1 = (); _0 = transmute(_1); return }
    let arr_ty = mir::Ty::Array(Box::new(i32_ty()), mir::ArrayLen::Known(0));
    let body = mir::Body::new(
        decls(&[arr_ty.clone(), mir::Ty::Tuple(vec![])]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![],
                    ),
                ),
                assign(place(0), transmute(copy(1), arr_ty.clone())),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("ZST-to-ZST transmute runs"),
        Value::Tuple(vec![])
    );
}

#[test]
fn transmute_fat_pointer_stays_unsupported() {
    // fn f(_1: &str) -> *const u8 { _0 = transmute(_1); return }
    // A fat pointer's metadata half has no thin target to land on.
    let str_ref_ty = mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(mir::Ty::Str),
        mir::Mutability::Not,
    );
    let raw_u8_ty = mir::Ty::RawPtr(Box::new(u8_ty()), mir::Mutability::Not);
    let body = mir::Body::new(
        decls(&[raw_u8_ty.clone(), str_ref_ty]),
        1,
        vec![block(
            vec![assign(place(0), transmute(copy(1), raw_u8_ty))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    let err = interpret_mir(&prog, "f", vec![Value::Str("hi".to_string())])
        .expect_err("fat pointer transmute must stay unsupported");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedRvalue { .. })
        ),
        "expected UnsupportedRvalue, got {err}"
    );
}

#[test]
fn transmute_enum_with_payload_stays_unsupported() {
    // An enum with a payload variant has no discriminant-only
    // reinterpretation: its bytes are not an integer.
    const PAYLOAD_DEF: mir::DefId = mir::DefId::new(0, 80);
    let payload_ty = mir::Ty::Adt(PAYLOAD_DEF, vec![]);
    let konst = mir::Operand::constant(mir::Constant::from_value(
        mir::ConstValue::Adt {
            variant: 0,
            fields: vec![],
        },
    ));
    let body = mir::Body::new(
        decls(&[i32_ty(), payload_ty]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(konst)),
                assign(place(0), transmute(copy(1), i32_ty())),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.adt_shapes.insert(
        PAYLOAD_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Enum,
            variants: vec![vec![], vec![i32_ty()]],
        },
    );
    prog.add("f", body);
    let err = interpret_mir(&prog, "f", vec![])
        .expect_err("payload enum transmute must stay unsupported");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedRvalue { .. })
        ),
        "expected UnsupportedRvalue, got {err}"
    );
}

// A `ZeroSized` constant operand transmuted into a zero-sized
// array type: the target type's unique value.
#[test]
fn transmute_zero_sized_constant_into_zst_array() {
    let target = mir::Ty::Array(
        Box::new(mir::Ty::Array(Box::new(u32_ty()), mir::ArrayLen::Known(0))),
        mir::ArrayLen::Known(2),
    );
    let body = mir::Body::new(
        decls(&[target.clone()]),
        0,
        vec![block(
            vec![assign(
                place(0),
                transmute(
                    mir::Operand::constant(mir::Constant::from_value(
                        mir::ConstValue::ZeroSized,
                    )),
                    target,
                ),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![])
            .expect("zero-sized constant transmute runs"),
        Value::Tuple(vec![Value::Tuple(vec![]), Value::Tuple(vec![])])
    );
}


// A constant array operand transmuted into an array of a
// wider element, beyond the 128-bit whole-shape budget:
// the constant carries no place type, so the source reading
// is the type recorded for it, and the bit-stream
// reinterpretation regroups its narrower leaves.
#[test]
fn transmute_constant_array_regroups_beyond_128_bits() {
    // fn f() -> u32 { _1 = transmute(const [u16; 16]) as [u32; 8];
    //                 _0 = copy _1[3]; return }
    // Lane 3 is the little-endian regrouping of source
    // elements 6 (low half) and 7 (high half).
    let src_ty = mir::Ty::Array(Box::new(u16_ty()), mir::ArrayLen::Known(16));
    let dst_ty = mir::Ty::Array(Box::new(u32_ty()), mir::ArrayLen::Known(8));
    let fields: Vec<mir::ConstValue> = (0..16u128)
        .map(|i| {
            mir::ConstValue::Scalar(mir::Scalar::Int {
                data: 0x101 * i + 1,
                size: 2,
            })
        })
        .collect();
    let mut konst = mir::Constant::from_value(mir::ConstValue::Tuple(fields));
    konst.ty = Some(src_ty);
    let elem3 = mir::Place {
        local: mir::Local(1),
        projection: vec![mir::PlaceElem::ConstantIndex {
            offset: 3,
            min_length: 8,
            from_end: false,
        }],
    };
    let body = mir::Body::new(
        decls(&[u32_ty(), dst_ty.clone()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    transmute(mir::Operand::Constant(Box::new(konst)), dst_ty),
                ),
                assign(place(0), mir::Rvalue::Use(mir::Operand::Copy(elem3))),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    let lo = 0x101u32 * 6 + 1;
    let hi = 0x101u32 * 7 + 1;
    assert_eq!(
        interpret_mir(&prog, "f", vec![])
            .expect("constant array regroup transmute runs"),
        Value::Uint(u128::from(lo | (hi << 16)))
    );
}

const WRAP1_DEF: mir::DefId = mir::DefId::new(0, 81);

fn wrap1_ty() -> mir::Ty {
    mir::Ty::Adt(WRAP1_DEF, vec![])
}

// Integer -> enum with a single variant carrying a single
// integer payload: that variant holding the integer.
#[test]
fn transmute_int_into_single_variant_payload_enum() {
    let body = mir::Body::new(
        decls(&[wrap1_ty()]),
        0,
        vec![block(
            vec![assign(place(0), transmute(const_u32(2), wrap1_ty()))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.adt_shapes.insert(
        WRAP1_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Enum,
            variants: vec![vec![u32_ty()]],
        },
    );
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![])
            .expect("int-to-single-variant-enum transmute runs"),
        Value::Enum {
            variant: 0,
            fields: vec![Value::Uint(2)],
        }
    );
}

const BAR_DEF: mir::DefId = mir::DefId::new(0, 82);

fn bar_ty() -> mir::Ty {
    mir::Ty::Adt(BAR_DEF, vec![])
}

// Single-field struct wrapping an integer -> fieldless enum:
// the field is the discriminant (`Bar { a: 5 }` -> `Small`
// variant with discriminant 5, read back as 5).
#[test]
fn transmute_single_field_struct_into_fieldless_enum() {
    let body = mir::Body::new(
        decls(&[u8_ty(), bar_ty(), small_ty()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    adt_aggregate(BAR_DEF, 0, vec![], vec![const_u32(5)]),
                ),
                assign(place(2), transmute(copy(1), small_ty())),
                assign(place(0), transmute(copy(2), u8_ty())),
            ],
            ret(),
        )],
    );
    let mut prog = small_prog();
    prog.adt_shapes.insert(
        BAR_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![vec![u32_ty()]],
        },
    );
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![])
            .expect("struct-to-enum transmute runs"),
        Value::Uint(5)
    );
}

// Single-field struct -> integer of the field's type: the
// field itself (`Bar { a: 5 }` transmuted to u32 is 5).
#[test]
fn transmute_single_field_struct_into_int() {
    let body = mir::Body::new(
        decls(&[u32_ty(), bar_ty()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    adt_aggregate(BAR_DEF, 0, vec![], vec![const_u32(5)]),
                ),
                assign(place(0), transmute(copy(1), u32_ty())),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.adt_shapes.insert(
        BAR_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![vec![u32_ty()]],
        },
    );
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![])
            .expect("struct-to-int transmute runs"),
        Value::Uint(5)
    );
}

const CELL_DEF: mir::DefId = mir::DefId::new(0, 83);

fn cell_prog() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.fn_names
        .insert(CELL_DEF, "std::cell::UnsafeCell".to_string());
    prog.adt_shapes.insert(
        CELL_DEF,
        mir::AdtShape {
            kind: mir::AdtShapeKind::Struct,
            variants: vec![vec![u32_ty()]],
        },
    );
    prog
}

fn ref_cell_ty() -> mir::Ty {
    mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(mir::Ty::Adt(CELL_DEF, vec![u32_ty()])),
        mir::Mutability::Not,
    )
}

fn ref_u32_ty() -> mir::Ty {
    mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(u32_ty()),
        mir::Mutability::Not,
    )
}

// A transmute crossing an `UnsafeCell` wrapper boundary in the
// entering direction (&UnsafeCell<u32> -> &u32) must NOT collapse
// to the identity: the pointer would keep denoting the wrapper
// aggregate. It composes the cast conversion's field-entering
// treatment (Lane DP's `PtrField`), so the transmuted pointer
// denotes the wrapper's inner field and reads through it.
// fn f() -> u32 { _1 = UnsafeCell(41); _2 = &_1;
//                 _3 = transmute(copy _2) as &u32; return *copy _3 }
#[test]
fn transmute_across_cell_wrapper_enters_the_field() {
    let cell_ty = mir::Ty::Adt(CELL_DEF, vec![u32_ty()]);
    let body = mir::Body::new(
        decls(&[u32_ty(), cell_ty.clone(), ref_cell_ty(), ref_u32_ty()]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    adt_aggregate(CELL_DEF, 0, vec![u32_ty()], vec![const_u32(41)]),
                ),
                assign(
                    place(2),
                    mir::Rvalue::Ref(
                        mir::Region::Erased,
                        mir::BorrowKind::Shared,
                        place(1),
                    ),
                ),
                assign(place(3), transmute(copy(2), ref_u32_ty())),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(place(3).deref())),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = cell_prog();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![])
            .expect("cell-entering transmute runs"),
        Value::Uint(41)
    );
}

// The exiting direction (&u32 -> &UnsafeCell<u32>) still has no
// honest treatment: the pointer would have to come to denote a
// wrapper aggregate around the field it points into, and there
// is no field-exiting counterpart to `PtrField`. It stays an
// explicit unsupported instead of silently pointing at the field.
#[test]
fn transmute_out_of_cell_wrapper_stays_unsupported() {
    let body = mir::Body::new(
        decls(&[ref_cell_ty(), ref_u32_ty()]),
        0,
        vec![block(
            vec![assign(place(0), transmute(copy(1), ref_cell_ty()))],
            ret(),
        )],
    );
    let mut prog = cell_prog();
    prog.add("f", body);
    let err = interpret_mir(&prog, "f", vec![])
        .expect_err("cell-exiting transmute must stay unsupported");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedRvalue { .. })
        ),
        "expected UnsupportedRvalue, got {err}"
    );
}

// ----- slice element reinterpretation (Issue #513) --------------------------

fn usize_const(v: u64) -> mir::Operand {
    const_int(v as u128, 8)
}

fn ref_of(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Not)
}

#[test]
fn transmute_slice_of_fieldless_enum_to_ints_reads_discriminants() {
    // fn f() -> i32 {
    //   _1 = [Small::B, Small::C, Small::A];
    //   _2 = &_1;
    //   _3 = _2 as &[Small];
    //   _4 = transmute(_3) as &[i32];
    //   return (*_4)[1];   // the Small::C discriminant: 9
    // }
    // The pointer-identity reading of the transmute would hand
    // back the enum elements themselves, and the read of an
    // i32 element out of them is the type mismatch the
    // enum-univariant-repr record fails with.
    let small_arr_ty = mir::Ty::Array(Box::new(small_ty()), mir::ArrayLen::Known(3));
    let ref_slice_small = ref_of(mir::Ty::Slice(Box::new(small_ty())));
    let ref_slice_i32 = ref_of(mir::Ty::Slice(Box::new(i32_ty())));
    let body = mir::Body::new(
        decls(&[
            i32_ty(),
            small_arr_ty.clone(),
            ref_of(small_arr_ty),
            ref_slice_small.clone(),
            ref_slice_i32.clone(),
            usize_ty(),
        ]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(small_ty())),
                        vec![const_small(1), const_small(2), const_small(0)],
                    ),
                ),
                assign(
                    place(2),
                    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
                ),
                assign(
                    place(3),
                    mir::Rvalue::Cast(
                        mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
                        copy(2),
                        ref_slice_small,
                    ),
                ),
                assign(place(4), transmute(copy(3), ref_slice_i32)),
                assign(place(5), mir::Rvalue::Use(usize_const(1))),
                assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Copy(
                        place(4).deref().index(mir::Local(5)),
                    )),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = small_prog();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("slice enum-to-int transmute runs"),
        Value::Int(9)
    );
}

#[test]
fn transmute_slice_of_ints_to_fieldless_enum_reads_variants() {
    // fn f() -> u8 {
    //   _1 = [5i32, 1i32];
    //   _2 = &_1;
    //   _3 = _2 as &[i32];
    //   _4 = transmute(_3) as &[Small];
    //   _5 = (*_4)[0];
    //   return transmute(_5) as u8;   // back to the discriminant: 5
    // }
    let i32_arr_ty = mir::Ty::Array(Box::new(i32_ty()), mir::ArrayLen::Known(2));
    let ref_slice_i32 = ref_of(mir::Ty::Slice(Box::new(i32_ty())));
    let ref_slice_small = ref_of(mir::Ty::Slice(Box::new(small_ty())));
    let body = mir::Body::new(
        decls(&[
            u8_ty(),
            i32_arr_ty.clone(),
            ref_of(i32_arr_ty),
            ref_slice_i32.clone(),
            ref_slice_small.clone(),
            small_ty(),
            usize_ty(),
        ]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(i32_ty())),
                        vec![const_i32(5), const_i32(1)],
                    ),
                ),
                assign(
                    place(2),
                    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(1)),
                ),
                assign(
                    place(3),
                    mir::Rvalue::Cast(
                        mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
                        copy(2),
                        ref_slice_i32,
                    ),
                ),
                assign(place(4), transmute(copy(3), ref_slice_small)),
                assign(place(6), mir::Rvalue::Use(usize_const(0))),
                assign(
                    place(5),
                    mir::Rvalue::Use(mir::Operand::Copy(
                        place(4).deref().index(mir::Local(6)),
                    )),
                ),
                assign(place(0), transmute(copy(5), u8_ty())),
            ],
            ret(),
        )],
    );
    let mut prog = small_prog();
    prog.add("f", body);
    assert_eq!(
        interpret_mir(&prog, "f", vec![]).expect("slice int-to-enum transmute runs"),
        Value::Uint(5)
    );
}

// ----- function pointer to raw pointer casts (FnPtrToPtr) -------------------

const CALLEE_DEF: mir::DefId = mir::DefId::new(0, 21);
const OTHER_DEF: mir::DefId = mir::DefId::new(0, 22);

fn fnptr_u32_ty() -> mir::Ty {
    mir::Ty::FnPtr(Box::new(mir::FnSig::new(vec![u32_ty()], u32_ty())))
}

fn raw_unit_ty() -> mir::Ty {
    mir::Ty::RawPtr(Box::new(mir::Ty::unit()), mir::Mutability::Not)
}

fn raw_u8_ty() -> mir::Ty {
    mir::Ty::RawPtr(Box::new(u8_ty()), mir::Mutability::Not)
}

/// callee body: _0 = Add(copy _1, 41_u32); return
fn callee_body() -> mir::Body {
    mir::Body::new(
        decls(&[u32_ty(), u32_ty()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                binop(mir::BinOp::Add, copy(1), const_u32(41)),
            )],
            ret(),
        )],
    )
}

/// other body: _0 = Add(copy _1, 1_u32); return
fn other_body() -> mir::Body {
    mir::Body::new(
        decls(&[u32_ty(), u32_ty()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                binop(mir::BinOp::Add, copy(1), const_u32(1)),
            )],
            ret(),
        )],
    )
}

fn fnptr_prog() -> MirProgram {
    let mut prog = MirProgram::new();
    prog.add_with_def_id("callee", CALLEE_DEF, callee_body());
    prog.add_with_def_id("other", OTHER_DEF, other_body());
    prog
}

fn reify(def: mir::DefId, dest: u32) -> mir::Statement {
    assign(
        place(dest),
        mir::Rvalue::Cast(
            mir::CastKind::PointerCoercion(mir::PointerCoercion::ReifyFnPointer),
            fndef_operand(def),
            fnptr_u32_ty(),
        ),
    )
}

#[test]
fn fnptr_to_ptr_address_matches_fnptr_address_and_differs_per_function() {
    // check returns 1 when the address of (callee as *const ())
    // equals the address the callee fn pointer itself exposes,
    // is non-zero, and differs from (other as *const ()) address.
    // locals: 0 u32 (return), 1/2 fnptr, 3/4 raw ptr, 5/6/7 usize,
    // 8/9/10 bool
    let body = mir::Body::new(
        decls(&[
            u32_ty(),
            fnptr_u32_ty(),
            fnptr_u32_ty(),
            raw_unit_ty(),
            raw_unit_ty(),
            usize_ty(),
            usize_ty(),
            usize_ty(),
            bool_ty(),
            bool_ty(),
            bool_ty(),
        ]),
        0,
        vec![
            block(
                vec![
                    reify(CALLEE_DEF, 1),
                    reify(OTHER_DEF, 2),
                    assign(
                        place(3),
                        mir::Rvalue::Cast(mir::CastKind::FnPtrToPtr, copy(1), raw_unit_ty()),
                    ),
                    assign(
                        place(4),
                        mir::Rvalue::Cast(mir::CastKind::FnPtrToPtr, copy(2), raw_unit_ty()),
                    ),
                    assign(
                        place(5),
                        mir::Rvalue::Cast(
                            mir::CastKind::PointerExposeProvenance,
                            copy(3),
                            usize_ty(),
                        ),
                    ),
                    assign(
                        place(6),
                        mir::Rvalue::Cast(
                            mir::CastKind::PointerExposeProvenance,
                            copy(1),
                            usize_ty(),
                        ),
                    ),
                    assign(
                        place(7),
                        mir::Rvalue::Cast(
                            mir::CastKind::PointerExposeProvenance,
                            copy(4),
                            usize_ty(),
                        ),
                    ),
                    assign(place(8), binop(mir::BinOp::Eq, copy(5), copy(6))),
                    assign(place(9), binop(mir::BinOp::Ne, copy(5), copy(7))),
                    assign(place(10), binop(mir::BinOp::Ne, copy(5), const_int(0, 8))),
                ],
                switch_bool(copy(8), 3, 1),
            ),
            block(vec![], switch_bool(copy(9), 3, 2)),
            block(vec![], switch_bool(copy(10), 3, 4)),
            block(
                vec![assign(place(0), mir::Rvalue::Use(const_u32(0)))],
                ret(),
            ),
            block(
                vec![assign(place(0), mir::Rvalue::Use(const_u32(1)))],
                ret(),
            ),
        ],
    );
    let mut prog = fnptr_prog();
    prog.add("check", body);
    let v = interpret_mir(&prog, "check", vec![]).expect("fnptr cast check must run");
    assert_eq!(v, Value::Uint(1));
}

#[test]
fn fnptr_to_ptr_transmuted_back_calls_the_function() {
    // roundtrip: let p = callee as *const (); let f = transmute(p);
    // return f(1) -- the cast result transmutes back to the callable
    // value and the call returns 42.
    // locals: 0 u32 (return), 1 fnptr, 2 raw ptr, 3 fnptr
    let body = mir::Body::new(
        decls(&[u32_ty(), fnptr_u32_ty(), raw_unit_ty(), fnptr_u32_ty()]),
        0,
        vec![
            block(
                vec![
                    reify(CALLEE_DEF, 1),
                    assign(
                        place(2),
                        mir::Rvalue::Cast(mir::CastKind::FnPtrToPtr, copy(1), raw_unit_ty()),
                    ),
                    assign(
                        place(3),
                        mir::Rvalue::Cast(mir::CastKind::Transmute, copy(2), fnptr_u32_ty()),
                    ),
                ],
                call(copy(3), const_u32(1), 0, 1),
            ),
            block(vec![], ret()),
        ],
    );
    let mut prog = fnptr_prog();
    prog.add("roundtrip", body);
    let v = interpret_mir(&prog, "roundtrip", vec![]).expect("fnptr roundtrip must run");
    assert_eq!(v, Value::Uint(42));
}

#[test]
fn fnptr_as_unsafe_fnptr_calls_the_function() {
    // via_unsafe: let f = callee as unsafe fn(u32) -> u32
    // (PointerCoercion UnsafeFnPointer); return f(1) == 42.
    // locals: 0 u32 (return), 1 fnptr, 2 fnptr (unsafe view)
    let body = mir::Body::new(
        decls(&[u32_ty(), fnptr_u32_ty(), fnptr_u32_ty()]),
        0,
        vec![
            block(
                vec![
                    reify(CALLEE_DEF, 1),
                    assign(
                        place(2),
                        mir::Rvalue::Cast(
                            mir::CastKind::PointerCoercion(
                                mir::PointerCoercion::UnsafeFnPointer,
                            ),
                            copy(1),
                            fnptr_u32_ty(),
                        ),
                    ),
                ],
                call(copy(2), const_u32(1), 0, 1),
            ),
            block(vec![], ret()),
        ],
    );
    let mut prog = fnptr_prog();
    prog.add("via_unsafe", body);
    let v = interpret_mir(&prog, "via_unsafe", vec![]).expect("unsafe fnptr view must run");
    assert_eq!(v, Value::Uint(42));
}

#[test]
fn fnptr_to_ptr_direct_deref_reads_the_code_image() {
    // code_byte: the byte read directly through (callee as *const u8)
    // equals the byte read through (callee as usize as *const u8), the
    // code-image path the integer round trip already served.
    // locals: 0 bool (return), 1 fnptr, 2 raw u8 ptr, 3 usize,
    // 4 raw u8 ptr, 5 u8, 6 u8
    let body = mir::Body::new(
        decls(&[
            bool_ty(),
            fnptr_u32_ty(),
            raw_u8_ty(),
            usize_ty(),
            raw_u8_ty(),
            u8_ty(),
            u8_ty(),
        ]),
        0,
        vec![block(
            vec![
                reify(CALLEE_DEF, 1),
                assign(
                    place(2),
                    mir::Rvalue::Cast(mir::CastKind::FnPtrToPtr, copy(1), raw_u8_ty()),
                ),
                assign(
                    place(3),
                    mir::Rvalue::Cast(
                        mir::CastKind::PointerExposeProvenance,
                        copy(1),
                        usize_ty(),
                    ),
                ),
                assign(
                    place(4),
                    mir::Rvalue::Cast(
                        mir::CastKind::PointerWithExposedProvenance,
                        copy(3),
                        raw_u8_ty(),
                    ),
                ),
                assign(place(5), mir::Rvalue::Use(copy_deref(2))),
                assign(place(6), mir::Rvalue::Use(copy_deref(4))),
                assign(place(0), binop(mir::BinOp::Eq, copy(5), copy(6))),
            ],
            ret(),
        )],
    );
    let mut prog = fnptr_prog();
    prog.add("code_byte", body);
    let v = interpret_mir(&prog, "code_byte", vec![]).expect("fnptr code deref must run");
    assert_eq!(v, Value::Bool(true));
}
