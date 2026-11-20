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
fn float_local_is_a_conversion_error() {
    let body = mir::Body::new(
        decls(&[mir::Ty::Float(mir::FloatTy::F64)]),
        0,
        vec![block(vec![], ret())],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    let err = convert_program(&prog).expect_err("float local must not convert");
    assert!(
        matches!(err, ConvertError::UnsupportedType { .. }),
        "expected UnsupportedType, got {err}"
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
    // The conversion must fail explicitly, never guess.
    let err = convert_program(&prog).expect_err("must not convert");
    assert!(
        matches!(
            err,
            ConvertError::UnsupportedProjection { .. } | ConvertError::UnsupportedType { .. }
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
    assert!(
        matches!(
            err,
            MirRunError::Interp(InterpError::UnsupportedRvalue { .. })
        ),
        "expected UnsupportedRvalue, got {err}"
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
