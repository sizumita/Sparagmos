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
fn f16_local_is_a_conversion_error() {
    // f32 / f64 locals convert and execute (see float_tests.rs);
    // f16 stays outside the type model and must fail conversion
    // with an explicit reason. Conversion is isolated per body:
    // the failure is recorded for `f`, and running `f` as the
    // entry fails with that conversion error.
    let body = mir::Body::new(
        decls(&[mir::Ty::Float(mir::FloatTy::F16)]),
        0,
        vec![block(vec![], ret())],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    let converted = convert_program(&prog).expect("conversion is per-body");
    assert!(converted.get("f").is_none(), "f16 body must not convert");
    let reason = converted
        .unconverted_reason("f")
        .expect("the failure must be recorded");
    assert!(
        reason.contains("unsupported type"),
        "the recorded reason names the failure: {reason}"
    );
    let err = interpret_mir(&prog, "f", vec![]).expect_err("entry conversion must fail");
    assert!(
        matches!(
            err,
            MirRunError::Convert(ConvertError::UnsupportedType { .. })
        ),
        "expected Convert(UnsupportedType), got {err}"
    );
}

#[test]
fn subslice_projection_is_a_conversion_error() {
    // _0 = copy (_1[0..1]) with a Subslice projection.
    let indexed = mir::Place {
        local: mir::Local(1),
        projection: vec![mir::PlaceElem::Subslice {
            from: 0,
            to: 1,
            from_end: false,
        }],
    };
    let body = mir::Body::new(
        decls(&[
            u32_ty(),
            mir::Ty::Array(Box::new(u32_ty()), mir::ArrayLen::Known(2)),
        ]),
        0,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Use(mir::Operand::Copy(indexed)),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("idx", body);
    // The conversion must fail explicitly, never guess: the body
    // is recorded as unconverted, and running it as the entry
    // fails with the conversion error.
    let converted = convert_program(&prog).expect("conversion is per-body");
    assert!(
        converted.get("idx").is_none(),
        "subslice body must not convert"
    );
    assert!(
        converted.unconverted_reason("idx").is_some(),
        "the failure must be recorded"
    );
    let err = interpret_mir(&prog, "idx", vec![]).expect_err("entry conversion must fail");
    assert!(
        matches!(
            err,
            MirRunError::Convert(
                ConvertError::UnsupportedProjection { .. } | ConvertError::UnsupportedType { .. }
            )
        ),
        "expected a projection/type conversion error, got {err}"
    );
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
        Err(MirRunError::Interp(InterpError::Panic { message })) => {
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
/// `Box::new` builtin: the builtin stores the pointee directly in
/// the box value, so there is no Unique/NonNull wrapper chain in
/// the value, while rustc's deref elaboration still reads the
/// box's field 0 (the `Unique`) and casts it with
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
    // Write through the extracted pointer, then read the pointee
    // back out of the box's field 0: the pointer must name the
    // box's own storage.
    let body = builtin_box_body(vec![
        assign(place(2).deref(), mir::Rvalue::Use(const_i32(42))),
        assign(
            place(0),
            mir::Rvalue::Use(mir::Operand::Copy(field(1, 0, adt_i32_ty(UNIQUE_DEF)))),
        ),
    ]);
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
/// _3 &i32, _4 usize; returns the transmute of a reference to
/// local `which` (1 or 2).
fn addr_of_body(which: u32) -> mir::Body {
    let ref_i32 = mir::Ty::Ref(mir::Region::Erased, Box::new(i32_ty()), mir::Mutability::Not);
    mir::Body::new(
        decls(&[usize_ty(), i32_ty(), i32_ty(), ref_i32, usize_ty()]),
        0,
        vec![block(
            vec![
                assign(place(1), mir::Rvalue::Use(const_i32(5))),
                assign(place(2), mir::Rvalue::Use(const_i32(7))),
                assign(
                    place(3),
                    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place(which)),
                ),
                assign(
                    place(4),
                    mir::Rvalue::Cast(mir::CastKind::Transmute, copy(3), usize_ty()),
                ),
                assign(place(0), mir::Rvalue::Use(copy(4))),
            ],
            ret(),
        )],
    )
}

#[test]
fn transmute_pointer_to_usize_is_aligned_stable_and_distinct() {
    // The alignment precondition checks rustc emits around raw
    // pointer dereferences transmute the pointer to usize and
    // test `addr & (align - 1) == 0`; the synthetic address must
    // be aligned, stable per target, and distinct per place.
    let mut prog = MirProgram::new();
    prog.add("addr1", addr_of_body(1));
    let a1 = interpret_mir(&prog, "addr1", vec![]).expect("addr1 must run");
    let mut prog = MirProgram::new();
    prog.add("addr2", addr_of_body(2));
    let a2 = interpret_mir(&prog, "addr2", vec![]).expect("addr2 must run");
    let (Value::Uint(a1), Value::Uint(a2)) = (a1, a2) else {
        panic!("expected usize results");
    };
    assert_ne!(a1, 0, "a place address is never null");
    assert_eq!(a1 % 4096, 0, "address must satisfy alignment checks");
    assert_eq!(a2 % 4096, 0, "address must satisfy alignment checks");
    assert_ne!(a1, a2, "distinct places get distinct addresses");
}

#[test]
fn transmute_non_pointer_stays_unsupported() {
    // Only the pointer-to-integer transmute is modelled; an
    // integer-to-integer transmute keeps failing explicitly.
    let body = mir::Body::new(
        decls(&[i32_ty()]),
        0,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Cast(mir::CastKind::Transmute, const_i32(7), i32_ty()),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("bad_transmute", body);
    let err = interpret_mir(&prog, "bad_transmute", vec![])
        .expect_err("non-pointer transmute must stay unsupported");
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedRvalue { .. })
        ),
        "expected UnsupportedRvalue, got {err}"
    );
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

fn expect_shift_panic(
    op: mir::BinOp,
    value_ty: mir::Ty,
    amount_ty: mir::Ty,
    value: Value,
    amount: Value,
    message: &str,
) {
    match run_shift(op, value_ty, amount_ty, value, amount) {
        Err(MirRunError::Interp(InterpError::Panic { message: got })) => {
            assert_eq!(got, message)
        }
        Err(e) => panic!("expected guest panic `{message}`, got error {e}"),
        Ok(v) => panic!("expected guest panic `{message}`, got value {v:?}"),
    }
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
    // Shifting the sign bit into place with Shl loses bits on the
    // way back, which this interpreter's plain operators report as
    // an arithmetic overflow (as for same-type amounts).
    expect_shift_panic(
        Shl,
        i32_ty(),
        u8_ty(),
        Value::Int(1),
        Value::Uint(31),
        "arithmetic overflow in Shl",
    );
    // Amount == bits is an overflowing shift.
    expect_shift_panic(
        Shl,
        i32_ty(),
        u8_ty(),
        Value::Int(1),
        Value::Uint(32),
        "attempt to shift by 32 bits, which overflows the type width 32",
    );
    expect_shift_panic(
        Shr,
        u8_ty(),
        i16_ty(),
        Value::Uint(1),
        Value::Int(8),
        "attempt to shift by 8 bits, which overflows the type width 8",
    );
    expect_shift_panic(
        Shl,
        u32_ty(),
        u32_ty(),
        Value::Uint(1),
        Value::Uint(32),
        "attempt to shift by 32 bits, which overflows the type width 32",
    );
}

#[test]
fn negative_and_huge_shift_amounts_panic() {
    use mir::BinOp::{Shl, Shr};
    // A negative amount does not fit the u32 reading; u32::MAX
    // stands in for it in the message, as in the same-type path.
    expect_shift_panic(
        Shl,
        i32_ty(),
        i8_ty(),
        Value::Int(1),
        Value::Int(-1),
        "attempt to shift by 4294967295 bits, which overflows the type width 32",
    );
    expect_shift_panic(
        Shr,
        u32_ty(),
        i64_ty(),
        Value::Uint(16),
        Value::Int(-1),
        "attempt to shift by 4294967295 bits, which overflows the type width 32",
    );
    expect_shift_panic(
        Shl,
        i32_ty(),
        i32_ty(),
        Value::Int(1),
        Value::Int(-1),
        "attempt to shift by 4294967295 bits, which overflows the type width 32",
    );
    // An amount above u32::MAX overflows the same way.
    expect_shift_panic(
        Shl,
        i32_ty(),
        u64_ty(),
        Value::Int(1),
        Value::Uint(5_000_000_000),
        "attempt to shift by 4294967295 bits, which overflows the type width 32",
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
    // Bits shifted out of a Shl set the flag in the checked form.
    assert_eq!(
        run(Shl, u8_ty(), i64_ty(), Value::Uint(3), Value::Int(7)),
        Value::Tuple(vec![Value::Uint(128), Value::Bool(true)])
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
    // shift; read under the value's i32 type it would have been 255.
    let mut prog = MirProgram::new();
    prog.add("neg", body_with_amount(const_i8(-1)));
    match interpret_mir(&prog, "neg", vec![Value::Int(1)]) {
        Err(MirRunError::Interp(InterpError::Panic { message })) => assert_eq!(
            message,
            "attempt to shift by 4294967295 bits, which overflows the type width 32"
        ),
        other => panic!("expected overflowing-shift panic, got {other:?}"),
    }
    // A constant amount wider than the value (const 4294967296_u64)
    // keeps its value: read in the i32 context it truncated to 0
    // and the shift silently succeeded before.
    let mut prog = MirProgram::new();
    prog.add("huge", body_with_amount(const_u64(1 << 32)));
    match interpret_mir(&prog, "huge", vec![Value::Int(1)]) {
        Err(MirRunError::Interp(InterpError::Panic { message })) => assert_eq!(
            message,
            "attempt to shift by 4294967295 bits, which overflows the type width 32"
        ),
        other => panic!("expected overflowing-shift panic, got {other:?}"),
    }
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
