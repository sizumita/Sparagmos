//! Integration tests for the MIR interpreter, using small hand-built
//! programs whose expected results are obvious by inspection.

use sparagmos_interp::ir::*;
use sparagmos_interp::{interpret, InterpConfig, InterpError, Interpreter, Value};

fn bb(statements: Vec<Statement>, terminator: TerminatorKind) -> BasicBlockData {
    BasicBlockData::new(statements, terminator)
}

fn assign(local: usize, rvalue: Rvalue) -> Statement {
    Statement::assign(Place::from_index(local), rvalue)
}

fn use_local(local: usize) -> Rvalue {
    Rvalue::Use(Operand::copy_local(local))
}

fn bin(op: BinOp, lhs: Operand, rhs: Operand) -> Rvalue {
    Rvalue::binary(op, lhs, rhs)
}

fn goto(target: usize) -> TerminatorKind {
    TerminatorKind::Goto {
        target: BasicBlock(target),
    }
}

fn ret() -> TerminatorKind {
    TerminatorKind::Return
}

fn program_of(body: Body) -> Program {
    let mut p = Program::new();
    p.add_function(body);
    p
}

// ---------------------------------------------------------------------------

#[test]
fn addition() {
    // fn add(a: i64, b: i64) -> i64 { _0 = Add(_1, _2); return }
    let body = Body::new(
        "add",
        2,
        vec![Ty::i64(), Ty::i64(), Ty::i64()],
        vec![bb(
            vec![assign(
                0,
                bin(BinOp::Add, Operand::copy_local(1), Operand::copy_local(2)),
            )],
            ret(),
        )],
    );
    let p = program_of(body);
    assert_eq!(
        interpret(&p, "add", vec![Value::Int(40), Value::Int(2)]).unwrap(),
        Value::Int(42)
    );
    assert_eq!(
        interpret(&p, "add", vec![Value::Int(-5), Value::Int(5)]).unwrap(),
        Value::Int(0)
    );
}

#[test]
fn arithmetic_ops() {
    // fn calc(a: i64, b: i64) -> i64 { (((a * b) + (a / b)) - (a % b)) }
    // locals: 0 ret, 1 a, 2 b, 3 tmp_mul, 4 tmp_div, 5 tmp_rem
    let body = Body::new(
        "calc",
        2,
        vec![Ty::i64(); 6],
        vec![bb(
            vec![
                assign(
                    3,
                    bin(BinOp::Mul, Operand::copy_local(1), Operand::copy_local(2)),
                ),
                assign(
                    4,
                    bin(BinOp::Div, Operand::copy_local(1), Operand::copy_local(2)),
                ),
                assign(
                    5,
                    bin(BinOp::Rem, Operand::copy_local(1), Operand::copy_local(2)),
                ),
                assign(
                    3,
                    bin(BinOp::Add, Operand::copy_local(3), Operand::copy_local(4)),
                ),
                assign(
                    0,
                    bin(BinOp::Sub, Operand::copy_local(3), Operand::copy_local(5)),
                ),
            ],
            ret(),
        )],
    );
    let p = program_of(body);
    // (20*3) + (20/3=6) - (20%3=2) = 64
    assert_eq!(
        interpret(&p, "calc", vec![Value::Int(20), Value::Int(3)]).unwrap(),
        Value::Int(64)
    );
}

#[test]
fn unary_ops() {
    // fn neg_not(a: i64) -> i64 { _0 = Neg(_1) } and bitwise not via second fn
    let neg = Body::new(
        "neg",
        1,
        vec![Ty::i64(), Ty::i64()],
        vec![bb(
            vec![assign(
                0,
                Rvalue::UnaryOp(UnOp::Neg, Operand::copy_local(1)),
            )],
            ret(),
        )],
    );
    let not = Body::new(
        "not",
        1,
        vec![Ty::i64(), Ty::i64()],
        vec![bb(
            vec![assign(
                0,
                Rvalue::UnaryOp(UnOp::Not, Operand::copy_local(1)),
            )],
            ret(),
        )],
    );
    let mut p = Program::new();
    p.add_function(neg);
    p.add_function(not);
    assert_eq!(
        interpret(&p, "neg", vec![Value::Int(7)]).unwrap(),
        Value::Int(-7)
    );
    assert_eq!(
        interpret(&p, "not", vec![Value::Int(0)]).unwrap(),
        Value::Int(-1)
    );
}

#[test]
fn loop_sum_1_to_100() {
    // fn sum(n: i64) -> i64
    // locals: 0 ret, 1 n, 2 i, 3 sum, 4 cond(bool)
    // bb0: i = 1; sum = 0; goto bb1
    // bb1: cond = Le(i, n); switchInt(cond) [1 => bb2, otherwise bb3]
    // bb2: sum = Add(sum, i); i = Add(i, 1); goto bb1
    // bb3: _0 = sum; return
    let body = Body::new(
        "sum",
        1,
        vec![Ty::i64(), Ty::i64(), Ty::i64(), Ty::i64(), Ty::Bool],
        vec![
            bb(
                vec![
                    assign(2, Rvalue::Use(Operand::const_int(1))),
                    assign(3, Rvalue::Use(Operand::const_int(0))),
                ],
                goto(1),
            ),
            bb(
                vec![assign(
                    4,
                    bin(BinOp::Le, Operand::copy_local(2), Operand::copy_local(1)),
                )],
                TerminatorKind::SwitchInt {
                    discr: Operand::copy_local(4),
                    targets: vec![(1, BasicBlock(2))],
                    otherwise: BasicBlock(3),
                },
            ),
            bb(
                vec![
                    assign(
                        3,
                        bin(BinOp::Add, Operand::copy_local(3), Operand::copy_local(2)),
                    ),
                    assign(
                        2,
                        bin(BinOp::Add, Operand::copy_local(2), Operand::const_int(1)),
                    ),
                ],
                goto(1),
            ),
            bb(vec![assign(0, use_local(3))], ret()),
        ],
    );
    let p = program_of(body);
    assert_eq!(
        interpret(&p, "sum", vec![Value::Int(100)]).unwrap(),
        Value::Int(5050)
    );
    assert_eq!(
        interpret(&p, "sum", vec![Value::Int(1)]).unwrap(),
        Value::Int(1)
    );
    assert_eq!(
        interpret(&p, "sum", vec![Value::Int(0)]).unwrap(),
        Value::Int(0)
    );
}

#[test]
fn fib_recursive() {
    // fn fib(n: i64) -> i64
    // locals: 0 ret, 1 n, 2 cond, 3 tmp(n-1), 4 a, 5 tmp(n-2), 6 b
    // bb0: cond = Le(n, 1); switchInt(cond) [1 => bb1, otherwise bb2]
    // bb1: _0 = n; return
    // bb2: tmp = Sub(n, 1); call fib(tmp) -> a, goto bb3
    // bb3: tmp2 = Sub(n, 2); call fib(tmp2) -> b, goto bb4
    // bb4: _0 = Add(a, b); return
    let tys = vec![
        Ty::i64(),
        Ty::i64(),
        Ty::Bool,
        Ty::i64(),
        Ty::i64(),
        Ty::i64(),
        Ty::i64(),
    ];
    let body = Body::new(
        "fib",
        1,
        tys,
        vec![
            bb(
                vec![assign(
                    2,
                    bin(BinOp::Le, Operand::copy_local(1), Operand::const_int(1)),
                )],
                TerminatorKind::SwitchInt {
                    discr: Operand::copy_local(2),
                    targets: vec![(1, BasicBlock(1))],
                    otherwise: BasicBlock(2),
                },
            ),
            bb(vec![assign(0, use_local(1))], ret()),
            bb(
                vec![assign(
                    3,
                    bin(BinOp::Sub, Operand::copy_local(1), Operand::const_int(1)),
                )],
                TerminatorKind::Call {
                    func: "fib".into(),
                    args: vec![Operand::copy_local(3)],
                    destination: Place::from_index(4),
                    target: Some(BasicBlock(3)),
                },
            ),
            bb(
                vec![assign(
                    5,
                    bin(BinOp::Sub, Operand::copy_local(1), Operand::const_int(2)),
                )],
                TerminatorKind::Call {
                    func: "fib".into(),
                    args: vec![Operand::copy_local(5)],
                    destination: Place::from_index(6),
                    target: Some(BasicBlock(4)),
                },
            ),
            bb(
                vec![assign(
                    0,
                    bin(BinOp::Add, Operand::copy_local(4), Operand::copy_local(6)),
                )],
                ret(),
            ),
        ],
    );
    let p = program_of(body);
    assert_eq!(
        interpret(&p, "fib", vec![Value::Int(0)]).unwrap(),
        Value::Int(0)
    );
    assert_eq!(
        interpret(&p, "fib", vec![Value::Int(1)]).unwrap(),
        Value::Int(1)
    );
    assert_eq!(
        interpret(&p, "fib", vec![Value::Int(10)]).unwrap(),
        Value::Int(55)
    );
    assert_eq!(
        interpret(&p, "fib", vec![Value::Int(15)]).unwrap(),
        Value::Int(610)
    );
}

#[test]
fn switch_int_branches() {
    // fn classify(x: i64) -> i64
    // switchInt(x) [0 => bb1, 1 => bb2, 42 => bb3, otherwise bb4]
    let mk = |v: i128| bb(vec![assign(0, Rvalue::Use(Operand::const_int(v)))], ret());
    let body = Body::new(
        "classify",
        1,
        vec![Ty::i64(), Ty::i64()],
        vec![
            bb(
                vec![],
                TerminatorKind::SwitchInt {
                    discr: Operand::copy_local(1),
                    targets: vec![(0, BasicBlock(1)), (1, BasicBlock(2)), (42, BasicBlock(3))],
                    otherwise: BasicBlock(4),
                },
            ),
            mk(100),
            mk(101),
            mk(142),
            mk(999),
        ],
    );
    let p = program_of(body);
    for (input, expected) in [(0, 100), (1, 101), (42, 142), (7, 999), (-1, 999)] {
        assert_eq!(
            interpret(&p, "classify", vec![Value::Int(input)]).unwrap(),
            Value::Int(expected),
            "classify({input})"
        );
    }
}

#[test]
fn checked_binary_op_reports_overflow_as_tuple() {
    // fn checked_add(a: i64, b: i64) -> (i64, bool)
    let body = Body::new(
        "checked_add",
        2,
        vec![Ty::Tuple(vec![Ty::i64(), Ty::Bool]), Ty::i64(), Ty::i64()],
        vec![bb(
            vec![assign(
                0,
                Rvalue::checked_binary(BinOp::Add, Operand::copy_local(1), Operand::copy_local(2)),
            )],
            ret(),
        )],
    );
    let p = program_of(body);
    assert_eq!(
        interpret(&p, "checked_add", vec![Value::Int(2), Value::Int(3)]).unwrap(),
        Value::Tuple(vec![Value::Int(5), Value::Bool(false)])
    );
    // i64::MAX + 1 wraps to i64::MIN and reports overflow = true.
    assert_eq!(
        interpret(
            &p,
            "checked_add",
            vec![Value::Int(i64::MAX as i128), Value::Int(1)]
        )
        .unwrap(),
        Value::Tuple(vec![Value::Int(i64::MIN as i128), Value::Bool(true)])
    );
}

#[test]
fn checked_result_fields_can_be_read() {
    // fn add_wrapped(a: i64, b: i64) -> i64  (value part of a checked add)
    // locals: 0 ret, 1 a, 2 b, 3 pair: (i64, bool)
    let pair_ty = Ty::Tuple(vec![Ty::i64(), Ty::Bool]);
    let body = Body::new(
        "add_wrapped",
        2,
        vec![Ty::i64(), Ty::i64(), Ty::i64(), pair_ty],
        vec![bb(
            vec![
                assign(
                    3,
                    Rvalue::checked_binary(
                        BinOp::Add,
                        Operand::copy_local(1),
                        Operand::copy_local(2),
                    ),
                ),
                Statement::assign(
                    Place::from_index(0),
                    Rvalue::Use(Operand::Copy(Place::from_index(3).field(0))),
                ),
            ],
            ret(),
        )],
    );
    let p = program_of(body);
    assert_eq!(
        interpret(
            &p,
            "add_wrapped",
            vec![Value::Int(i64::MAX as i128), Value::Int(1)]
        )
        .unwrap(),
        Value::Int(i64::MIN as i128)
    );
}

#[test]
fn plain_binary_op_overflow_panics() {
    // i64 overflow in a plain BinaryOp is a guest panic (checked semantics).
    let body = Body::new(
        "add",
        2,
        vec![Ty::i64(), Ty::i64(), Ty::i64()],
        vec![bb(
            vec![assign(
                0,
                bin(BinOp::Add, Operand::copy_local(1), Operand::copy_local(2)),
            )],
            ret(),
        )],
    );
    let p = program_of(body);
    let err = interpret(&p, "add", vec![Value::Int(i64::MAX as i128), Value::Int(1)]).unwrap_err();
    assert!(matches!(err, InterpError::Panic { .. }), "got: {err}");
}

#[test]
fn division_by_zero_panics() {
    let body = Body::new(
        "div",
        2,
        vec![Ty::i64(), Ty::i64(), Ty::i64()],
        vec![bb(
            vec![assign(
                0,
                bin(BinOp::Div, Operand::copy_local(1), Operand::copy_local(2)),
            )],
            ret(),
        )],
    );
    let p = program_of(body);
    let err = interpret(&p, "div", vec![Value::Int(1), Value::Int(0)]).unwrap_err();
    assert!(matches!(err, InterpError::Panic { .. }), "got: {err}");
}

#[test]
fn assert_terminator_panics_on_failure() {
    // fn require_positive(x: i64) -> i64 with Assert(x > 0) then identity.
    // locals: 0 ret, 1 x, 2 cond
    let body = Body::new(
        "require_positive",
        1,
        vec![Ty::i64(), Ty::i64(), Ty::Bool],
        vec![
            bb(
                vec![assign(
                    2,
                    bin(BinOp::Gt, Operand::copy_local(1), Operand::const_int(0)),
                )],
                TerminatorKind::Assert {
                    cond: Operand::copy_local(2),
                    expected: true,
                    target: BasicBlock(1),
                    message: "x must be positive".into(),
                },
            ),
            bb(vec![assign(0, use_local(1))], ret()),
        ],
    );
    let p = program_of(body);
    assert_eq!(
        interpret(&p, "require_positive", vec![Value::Int(3)]).unwrap(),
        Value::Int(3)
    );
    let err = interpret(&p, "require_positive", vec![Value::Int(-1)]).unwrap_err();
    assert!(matches!(err, InterpError::Panic { .. }), "got: {err}");
}

#[test]
fn unsigned_arithmetic() {
    // fn uadd(a: u32, b: u32) -> u32
    let body = Body::new(
        "uadd",
        2,
        vec![Ty::u32(), Ty::u32(), Ty::u32()],
        vec![bb(
            vec![assign(
                0,
                bin(BinOp::Add, Operand::copy_local(1), Operand::copy_local(2)),
            )],
            ret(),
        )],
    );
    let p = program_of(body);
    assert_eq!(
        interpret(&p, "uadd", vec![Value::Uint(20), Value::Uint(22)]).unwrap(),
        Value::Uint(42)
    );
    // u32 overflow in a plain op panics rather than wrapping silently.
    let err = interpret(
        &p,
        "uadd",
        vec![Value::Uint(u32::MAX as u128), Value::Uint(1)],
    )
    .unwrap_err();
    assert!(matches!(err, InterpError::Panic { .. }), "got: {err}");
}

#[test]
fn step_limit_stops_infinite_loop() {
    // fn forever() { bb0: goto bb0 }
    let body = Body::new("forever", 0, vec![Ty::Unit], vec![bb(vec![], goto(0))]);
    let p = program_of(body);
    let mut interp = Interpreter::with_config(
        &p,
        InterpConfig {
            max_steps: 1_000,
            max_call_depth: 16,
            guest_args: Vec::new(),
        },
    );
    let err = interp.run("forever", vec![]).unwrap_err();
    assert!(
        matches!(err, InterpError::StepLimitExceeded { limit: 1_000 }),
        "got: {err}"
    );
}

#[test]
fn call_depth_limit_stops_unbounded_recursion() {
    // fn f() { call f() -> _0, goto bb1 } bb1: return
    let body = Body::new(
        "f",
        0,
        vec![Ty::i64()],
        vec![
            bb(
                vec![],
                TerminatorKind::Call {
                    func: "f".into(),
                    args: vec![],
                    destination: Place::from_index(0),
                    target: Some(BasicBlock(1)),
                },
            ),
            bb(vec![], ret()),
        ],
    );
    let p = program_of(body);
    let mut interp = Interpreter::with_config(
        &p,
        InterpConfig {
            max_steps: 1_000_000,
            max_call_depth: 8,
            guest_args: Vec::new(),
        },
    );
    let err = interp.run("f", vec![]).unwrap_err();
    assert!(
        matches!(err, InterpError::CallDepthExceeded { .. }),
        "got: {err}"
    );
}

#[test]
fn unsupported_statement_is_an_explicit_error() {
    let body = Body::new(
        "bad",
        0,
        vec![Ty::i64()],
        vec![bb(
            vec![Statement {
                kind: StatementKind::Unsupported("SetDiscriminant".into()),
            }],
            ret(),
        )],
    );
    let p = program_of(body);
    let err = interpret(&p, "bad", vec![]).unwrap_err();
    assert!(
        matches!(err, InterpError::UnsupportedStatement { .. }),
        "got: {err}"
    );
}

#[test]
fn unsupported_rvalue_is_an_explicit_error() {
    let body = Body::new(
        "bad",
        0,
        vec![Ty::i64()],
        vec![bb(
            vec![assign(0, Rvalue::Unsupported("frobnicate".to_string()))],
            ret(),
        )],
    );
    let p = program_of(body);
    let err = interpret(&p, "bad", vec![]).unwrap_err();
    assert!(
        matches!(err, InterpError::UnsupportedRvalue { .. }),
        "got: {err}"
    );
}

#[test]
fn unknown_function_is_an_explicit_error() {
    let body = Body::new(
        "caller",
        0,
        vec![Ty::i64()],
        vec![
            bb(
                vec![],
                TerminatorKind::Call {
                    func: "missing".into(),
                    args: vec![],
                    destination: Place::from_index(0),
                    target: Some(BasicBlock(1)),
                },
            ),
            bb(vec![], ret()),
        ],
    );
    let p = program_of(body);
    let err = interpret(&p, "caller", vec![]).unwrap_err();
    assert!(
        matches!(err, InterpError::UnknownFunction { .. }),
        "got: {err}"
    );
    // And an unknown entry point, too.
    let err = interpret(&p, "nope", vec![]).unwrap_err();
    assert!(
        matches!(err, InterpError::UnknownFunction { .. }),
        "got: {err}"
    );
}

#[test]
fn uninitialized_read_is_an_explicit_error() {
    // _0 = Copy(_1) where _1 was never written and is not an argument.
    let body = Body::new(
        "bad",
        0,
        vec![Ty::i64(), Ty::i64()],
        vec![bb(vec![assign(0, use_local(1))], ret())],
    );
    let p = program_of(body);
    let err = interpret(&p, "bad", vec![]).unwrap_err();
    assert!(
        matches!(err, InterpError::UninitializedRead { .. }),
        "got: {err}"
    );
}

#[test]
fn cross_function_call() {
    // fn double(x) = x + x ; fn main(x) = double(x) + 1
    let double = Body::new(
        "double",
        1,
        vec![Ty::i64(), Ty::i64()],
        vec![bb(
            vec![assign(
                0,
                bin(BinOp::Add, Operand::copy_local(1), Operand::copy_local(1)),
            )],
            ret(),
        )],
    );
    let main = Body::new(
        "main",
        1,
        vec![Ty::i64(), Ty::i64(), Ty::i64()],
        vec![
            bb(
                vec![],
                TerminatorKind::Call {
                    func: "double".into(),
                    args: vec![Operand::copy_local(1)],
                    destination: Place::from_index(2),
                    target: Some(BasicBlock(1)),
                },
            ),
            bb(
                vec![assign(
                    0,
                    bin(BinOp::Add, Operand::copy_local(2), Operand::const_int(1)),
                )],
                ret(),
            ),
        ],
    );
    let mut p = Program::new();
    p.add_function(double);
    p.add_function(main);
    assert_eq!(
        interpret(&p, "main", vec![Value::Int(20)]).unwrap(),
        Value::Int(41)
    );
}

// ---------------------------------------------------------------------------
// Shifts: the amount may have a different integer type than the value.

#[test]
fn mixed_type_shift_amounts() {
    // fn shl(a: i32, b: u32) -> i32 { _0 = Shl(_1, _2); return }
    let shl = Body::new(
        "shl",
        2,
        vec![Ty::i32(), Ty::i32(), Ty::u32()],
        vec![bb(
            vec![assign(
                0,
                bin(BinOp::Shl, Operand::copy_local(1), Operand::copy_local(2)),
            )],
            ret(),
        )],
    );
    let p = program_of(shl);
    assert_eq!(
        interpret(&p, "shl", vec![Value::Int(3), Value::Uint(2)]).unwrap(),
        Value::Int(12)
    );
    // fn shr(a: i32, b: u32) -> i32: arithmetic shift of a negative value.
    let shr = Body::new(
        "shr",
        2,
        vec![Ty::i32(), Ty::i32(), Ty::u32()],
        vec![bb(
            vec![assign(
                0,
                bin(BinOp::Shr, Operand::copy_local(1), Operand::copy_local(2)),
            )],
            ret(),
        )],
    );
    let p = program_of(shr);
    assert_eq!(
        interpret(&p, "shr", vec![Value::Int(-16), Value::Uint(2)]).unwrap(),
        Value::Int(-4)
    );
}

#[test]
fn mixed_type_shift_amount_overflow_panics() {
    // fn shl(a: i32, b: u32) -> i32 with an amount at the type width.
    let shl = Body::new(
        "shl",
        2,
        vec![Ty::i32(), Ty::i32(), Ty::u32()],
        vec![bb(
            vec![assign(
                0,
                bin(BinOp::Shl, Operand::copy_local(1), Operand::copy_local(2)),
            )],
            ret(),
        )],
    );
    let p = program_of(shl);
    let err = interpret(&p, "shl", vec![Value::Int(1), Value::Uint(32)]).unwrap_err();
    match err {
        InterpError::Panic { message } => assert_eq!(
            message,
            "attempt to shift left with overflow"
        ),
        other => panic!("expected shift-overflow panic, got {other}"),
    }
}
