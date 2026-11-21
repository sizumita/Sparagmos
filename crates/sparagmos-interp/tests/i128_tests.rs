//! 128-bit integer semantics (Lane BL): `i128` / `u128` in the
//! interpreter — boundary arithmetic, overflow panics, division and
//! remainder edge cases, shifts, casts (including float saturation),
//! comparisons, and the constant-decoding paths that carry 128-bit
//! values (promoted by-reference array constants and context-free
//! scalars).
//!
//! Bodies are hand-built `sparagmos-mir` in the shapes rustc 1.100.0
//! emits, converted and executed through `mir_convert`, and compared
//! against host `as` / operator semantics (the test binary itself is
//! compiled by real Rust, so host expressions are the oracle).

use sparagmos_interp::mir_convert::{interpret_mir, MirProgram, MirRunError};
use sparagmos_interp::{InterpError, Value};
use sparagmos_mir as mir;

// ----- small builders over sparagmos_mir -----------------------------------

fn i128_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I128)
}
fn u128_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U128)
}
fn i64_ty() -> mir::Ty {
    mir::Ty::Int(mir::IntTy::I64)
}
fn u8_ty() -> mir::Ty {
    mir::Ty::Uint(mir::UintTy::U8)
}
fn f32_ty() -> mir::Ty {
    mir::Ty::Float(mir::FloatTy::F32)
}
fn f64_ty() -> mir::Ty {
    mir::Ty::Float(mir::FloatTy::F64)
}
fn bool_ty() -> mir::Ty {
    mir::Ty::Bool
}
fn array_ty(elem: mir::Ty, len: u64) -> mir::Ty {
    mir::Ty::Array(Box::new(elem), mir::ArrayLen::Known(len))
}
fn ref_ty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Not)
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

fn field(idx: u32, f: u32, ty: mir::Ty) -> mir::Place {
    place(idx).field(f, ty)
}

/// An integer scalar constant with an explicit storage size in bytes
/// (the shape `sparagmos-mir` uses for integer literals).
fn const_int(data: u128, size: u8) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int { data, size },
    )))
}

fn const_i128(v: i128) -> mir::Operand {
    const_int(v as u128, 16)
}

fn const_u128(v: u128) -> mir::Operand {
    const_int(v, 16)
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

fn ret() -> mir::TerminatorKind {
    mir::TerminatorKind::Return
}

/// `fn f(_1: T, _2: T) -> T { _0 = <op>(copy _1, copy _2); return }`
fn plain_binop_body(ty: mir::Ty, op: mir::BinOp) -> mir::Body {
    mir::Body::new(
        decls(&[ty.clone(), ty.clone(), ty]),
        2,
        vec![block(
            vec![assign(place(0), binop(op, copy(1), copy(2)))],
            ret(),
        )],
    )
}

fn run_binop(ty: mir::Ty, op: mir::BinOp, a: Value, b: Value) -> Result<Value, MirRunError> {
    let mut prog = MirProgram::new();
    prog.add("f", plain_binop_body(ty, op));
    interpret_mir(&prog, "f", vec![a, b])
}

fn expect_panic(res: Result<Value, MirRunError>) -> String {
    match res {
        Err(MirRunError::Interp(InterpError::Panic { message, .. })) => message,
        other => panic!("expected a guest panic, got {other:?}"),
    }
}

// ----- boundary arithmetic (plain form: overflow is a guest panic) ----------

#[test]
fn i128_add_sub_mul_boundaries() {
    use mir::BinOp::*;
    let run = |op: mir::BinOp, a: i128, b: i128| run_binop(i128_ty(), op, Value::Int(a), Value::Int(b));
    assert_eq!(run(Add, i128::MAX, 0).unwrap(), Value::Int(i128::MAX));
    assert_eq!(run(Add, i128::MAX - 1, 1).unwrap(), Value::Int(i128::MAX));
    assert_eq!(run(Sub, i128::MIN, -1).unwrap(), Value::Int(i128::MIN + 1));
    assert_eq!(run(Mul, 1i128 << 64, 1i128 << 62).unwrap(), Value::Int(1i128 << 126));
    assert_eq!(run(Mul, -3, 5).unwrap(), Value::Int(-15));
    // Plain-form overflow panics, with the interpreter's overflow message.
    let msg = expect_panic(run(Add, i128::MAX, 1));
    assert!(msg.contains("overflow"), "unexpected message: {msg}");
    expect_panic(run(Sub, i128::MIN, 1));
    expect_panic(run(Mul, 1i128 << 64, 1i128 << 64));
    // 2^127 is one past the largest i128: it overflows too.
    expect_panic(run(Mul, 1i128 << 64, 1i128 << 63));
    expect_panic(run(Add, i128::MIN, -1));
}

#[test]
fn u128_add_sub_mul_boundaries() {
    use mir::BinOp::*;
    let run = |op: mir::BinOp, a: u128, b: u128| {
        run_binop(u128_ty(), op, Value::Uint(a), Value::Uint(b))
    };
    assert_eq!(run(Add, u128::MAX, 0).unwrap(), Value::Uint(u128::MAX));
    assert_eq!(run(Add, u128::MAX - 1, 1).unwrap(), Value::Uint(u128::MAX));
    assert_eq!(run(Mul, 1u128 << 127, 1).unwrap(), Value::Uint(1u128 << 127));
    assert_eq!(run(Sub, 0, 0).unwrap(), Value::Uint(0));
    expect_panic(run(Add, u128::MAX, 1));
    expect_panic(run(Sub, 0, 1));
    expect_panic(run(Mul, 1u128 << 127, 2));
}

#[test]
fn i128_div_rem_edges() {
    use mir::BinOp::*;
    let run = |op: mir::BinOp, a: i128, b: i128| run_binop(i128_ty(), op, Value::Int(a), Value::Int(b));
    assert_eq!(run(Div, i128::MIN, 3).unwrap(), Value::Int(i128::MIN / 3));
    assert_eq!(run(Rem, i128::MIN, 3).unwrap(), Value::Int(i128::MIN % 3));
    assert_eq!(run(Div, -7, 2).unwrap(), Value::Int(-3));
    assert_eq!(run(Rem, -7, 2).unwrap(), Value::Int(-1));
    // Division by zero: the exact native panic messages.
    assert_eq!(expect_panic(run(Div, 5, 0)), "attempt to divide by zero");
    assert_eq!(
        expect_panic(run(Rem, 5, 0)),
        "attempt to calculate the remainder with a divisor of zero"
    );
    // MIN / -1 and MIN % -1 overflow.
    expect_panic(run(Div, i128::MIN, -1));
    expect_panic(run(Rem, i128::MIN, -1));
}

#[test]
fn u128_div_rem_edges() {
    use mir::BinOp::*;
    let run = |op: mir::BinOp, a: u128, b: u128| {
        run_binop(u128_ty(), op, Value::Uint(a), Value::Uint(b))
    };
    assert_eq!(
        run(Div, u128::MAX, (1u128 << 64) - 1).unwrap(),
        Value::Uint(u128::MAX / ((1u128 << 64) - 1))
    );
    assert_eq!(
        run(Rem, u128::MAX, 1u128 << 100).unwrap(),
        Value::Uint(u128::MAX % (1u128 << 100))
    );
    assert_eq!(expect_panic(run(Div, 5, 0)), "attempt to divide by zero");
    assert_eq!(
        expect_panic(run(Rem, 5, 0)),
        "attempt to calculate the remainder with a divisor of zero"
    );
}

// ----- checked form (rustc's debug shape: pair + Assert) ---------------------

/// The shape rustc 1.100.0 emits for `a <op> b` in debug builds:
/// `_3 = CheckedBinaryOp(op, ..); assert(!move _3.1); _0 = move _3.0; return`
fn checked_binop_body(ty: mir::Ty, op: mir::BinOp) -> mir::Body {
    let pair_ty = mir::Ty::Tuple(vec![ty.clone(), bool_ty()]);
    mir::Body::new(
        decls(&[ty.clone(), ty.clone(), ty, pair_ty]),
        2,
        vec![
            block(
                vec![assign(
                    place(3),
                    mir::Rvalue::CheckedBinaryOp(op, Box::new((copy(1), copy(2)))),
                )],
                mir::TerminatorKind::Assert {
                    cond: mir::Operand::Move(field(3, 1, bool_ty())),
                    expected: false,
                    msg: Box::new(mir::AssertMessage::Overflow(
                        op,
                        Box::new(copy(1)),
                        Box::new(copy(2)),
                    )),
                    target: mir::BasicBlock(1),
                    unwind: mir::UnwindAction::Continue,
                },
            ),
            block(
                vec![assign(
                    place(0),
                    mir::Rvalue::Use(mir::Operand::Move(field(3, 0, i128_ty()))),
                )],
                ret(),
            ),
        ],
    )
}

#[test]
fn i128_checked_ops_panic_via_assert_on_overflow() {
    use mir::BinOp::*;
    let mut prog = MirProgram::new();
    prog.add("add", checked_binop_body(i128_ty(), Add));
    assert_eq!(
        interpret_mir(&prog, "add", vec![Value::Int(40), Value::Int(2)]).unwrap(),
        Value::Int(42)
    );
    let err = interpret_mir(&prog, "add", vec![Value::Int(i128::MAX), Value::Int(1)])
        .expect_err("checked add at MAX must trip the Assert");
    assert!(
        matches!(err, MirRunError::Interp(InterpError::Panic { .. })),
        "expected guest panic, got {err}"
    );

    let mut prog = MirProgram::new();
    prog.add("div", checked_binop_body(i128_ty(), Div));
    let err = interpret_mir(&prog, "div", vec![Value::Int(i128::MIN), Value::Int(-1)])
        .expect_err("checked MIN / -1 must trip the Assert");
    assert!(matches!(
        err,
        MirRunError::Interp(InterpError::Panic { .. })
    ));
}

/// The `(value, overflowed)` pair itself, without the Assert: checked
/// shifts report an out-of-range amount through the flag.
fn checked_pair_body(ty: mir::Ty, op: mir::BinOp) -> mir::Body {
    let pair_ty = mir::Ty::Tuple(vec![ty.clone(), bool_ty()]);
    mir::Body::new(
        decls(&[pair_ty, ty.clone(), ty]),
        2,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::CheckedBinaryOp(op, Box::new((copy(1), copy(2)))),
            )],
            ret(),
        )],
    )
}

#[test]
fn i128_checked_shift_reports_amount_overflow_in_flag() {
    use mir::BinOp::*;
    let mut prog = MirProgram::new();
    prog.add("shl", checked_pair_body(i128_ty(), Shl));
    // 1 << 127 fits exactly.
    assert_eq!(
        interpret_mir(&prog, "shl", vec![Value::Int(1), Value::Int(127)]).unwrap(),
        Value::Tuple(vec![Value::Int(i128::MIN), Value::Bool(false)])
    );
    // Amount 128 is out of range: flag set (the wrapped value is the
    // amount modulo the width, per CheckedBinaryOp semantics).
    assert_eq!(
        interpret_mir(&prog, "shl", vec![Value::Int(1), Value::Int(128)]).unwrap(),
        Value::Tuple(vec![Value::Int(1), Value::Bool(true)])
    );
    // Bits shifted out of the value are lost, NOT an overflow: the
    // flag stays clear and the value wraps (3 << 127 == i128::MIN).
    assert_eq!(
        interpret_mir(&prog, "shl", vec![Value::Int(3), Value::Int(127)]).unwrap(),
        Value::Tuple(vec![Value::Int(i128::MIN), Value::Bool(false)])
    );
}

// ----- shifts (same-type amounts; mixed-type amounts are Lane BB) ------------

#[test]
fn i128_shifts() {
    use mir::BinOp::*;
    let run = |op: mir::BinOp, a: i128, b: i128| run_binop(i128_ty(), op, Value::Int(a), Value::Int(b));
    assert_eq!(run(Shl, 1, 127).unwrap(), Value::Int(i128::MIN));
    assert_eq!(run(Shl, 1, 0).unwrap(), Value::Int(1));
    assert_eq!(run(Shr, -16, 2).unwrap(), Value::Int(-4));
    assert_eq!(run(Shr, i128::MIN, 127).unwrap(), Value::Int(-1));
    // An out-of-range amount in the plain form is a guest panic.
    let msg = expect_panic(run(Shl, 1, 128));
    assert!(msg.contains("shift"), "unexpected message: {msg}");
    expect_panic(run(Shr, 1, 200));
}

#[test]
fn u128_shifts() {
    use mir::BinOp::*;
    let run = |op: mir::BinOp, a: u128, b: u128| {
        run_binop(u128_ty(), op, Value::Uint(a), Value::Uint(b))
    };
    assert_eq!(run(Shl, 1, 127).unwrap(), Value::Uint(1u128 << 127));
    assert_eq!(run(Shr, u128::MAX, 64).unwrap(), Value::Uint((1u128 << 64) - 1));
    expect_panic(run(Shl, 1, 128));
}

// ----- comparisons and bitwise ops -------------------------------------------

#[test]
fn i128_u128_comparisons_and_bitwise() {
    use mir::BinOp::*;
    // A comparison body returns bool: `fn f(_1: T, _2: T) -> bool`.
    let cmp_body = |ty: mir::Ty, op: mir::BinOp| {
        mir::Body::new(
            decls(&[bool_ty(), ty.clone(), ty]),
            2,
            vec![block(
                vec![assign(place(0), binop(op, copy(1), copy(2)))],
                ret(),
            )],
        )
    };
    let cmp = |op: mir::BinOp, a: i128, b: i128| {
        let mut prog = MirProgram::new();
        prog.add("f", cmp_body(i128_ty(), op));
        interpret_mir(&prog, "f", vec![Value::Int(a), Value::Int(b)]).unwrap()
    };
    assert_eq!(cmp(Lt, i128::MIN, i128::MAX), Value::Bool(true));
    assert_eq!(cmp(Gt, -1, i128::MIN), Value::Bool(true));
    assert_eq!(cmp(Eq, i128::MIN, i128::MIN), Value::Bool(true));
    assert_eq!(cmp(Ne, i128::MIN, -1), Value::Bool(true));
    assert_eq!(cmp(Le, i128::MAX, i128::MAX), Value::Bool(true));
    assert_eq!(cmp(Ge, i128::MIN, 0), Value::Bool(false));

    let ucmp = |op: mir::BinOp, a: u128, b: u128| {
        let mut prog = MirProgram::new();
        prog.add("f", cmp_body(u128_ty(), op));
        interpret_mir(&prog, "f", vec![Value::Uint(a), Value::Uint(b)]).unwrap()
    };
    assert_eq!(ucmp(Gt, u128::MAX, 1u128 << 127), Value::Bool(true));
    assert_eq!(ucmp(Lt, 0, u128::MAX), Value::Bool(true));

    let bit = |op: mir::BinOp, a: i128, b: i128| {
        run_binop(i128_ty(), op, Value::Int(a), Value::Int(b)).unwrap()
    };
    assert_eq!(bit(BitAnd, -1, i128::MIN), Value::Int(i128::MIN));
    assert_eq!(bit(BitOr, 0, -1), Value::Int(-1));
    assert_eq!(bit(BitXor, -1, -1), Value::Int(0));
}

#[test]
fn i128_unary_ops() {
    // fn f(_1: i128) -> i128 { _0 = Neg(copy _1); return }
    let neg = mir::Body::new(
        decls(&[i128_ty(), i128_ty()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::UnaryOp(mir::UnOp::Neg, copy(1)),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("neg", neg);
    assert_eq!(
        interpret_mir(&prog, "neg", vec![Value::Int(5)]).unwrap(),
        Value::Int(-5)
    );
    // Negating MIN overflows.
    let err = interpret_mir(&prog, "neg", vec![Value::Int(i128::MIN)])
        .expect_err("-(i128::MIN) must panic");
    assert!(matches!(
        err,
        MirRunError::Interp(InterpError::Panic { .. })
    ));

    // fn f(_1: i128) -> i128 { _0 = Not(copy _1); return }
    let not = mir::Body::new(
        decls(&[i128_ty(), i128_ty()]),
        1,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::UnaryOp(mir::UnOp::Not, copy(1)),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("not", not);
    assert_eq!(
        interpret_mir(&prog, "not", vec![Value::Int(0)]).unwrap(),
        Value::Int(-1)
    );
}

// ----- casts -------------------------------------------------------------------

/// `fn f(_1: S) -> T { _0 = Cast(kind, copy _1, T); return }`
fn cast_body(src: mir::Ty, kind: mir::CastKind, target: mir::Ty) -> mir::Body {
    mir::Body::new(
        decls(&[target.clone(), src]),
        1,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Cast(kind, copy(1), target))],
            ret(),
        )],
    )
}

fn run_cast(src: mir::Ty, kind: mir::CastKind, target: mir::Ty, v: Value) -> Value {
    let mut prog = MirProgram::new();
    prog.add("f", cast_body(src, kind, target));
    interpret_mir(&prog, "f", vec![v]).expect("cast must run")
}

#[test]
fn i128_int_to_int_casts() {
    use mir::CastKind::IntToInt;
    // Truncation to narrower widths.
    assert_eq!(
        run_cast(i128_ty(), IntToInt, mir::Ty::Int(mir::IntTy::I8), Value::Int(300)),
        Value::Int(44)
    );
    assert_eq!(
        run_cast(i128_ty(), IntToInt, u8_ty(), Value::Int(-1)),
        Value::Uint(255)
    );
    assert_eq!(
        run_cast(i128_ty(), IntToInt, i64_ty(), Value::Int((1i128 << 64) + 5)),
        Value::Int(5)
    );
    // Signed <-> unsigned at the full width.
    assert_eq!(
        run_cast(i128_ty(), IntToInt, u128_ty(), Value::Int(-1)),
        Value::Uint(u128::MAX)
    );
    assert_eq!(
        run_cast(u128_ty(), IntToInt, i128_ty(), Value::Uint(u128::MAX)),
        Value::Int(-1)
    );
    assert_eq!(
        run_cast(u128_ty(), IntToInt, i128_ty(), Value::Uint(1u128 << 127)),
        Value::Int(i128::MIN)
    );
    // Widening sign-extends / zero-extends.
    assert_eq!(
        run_cast(i64_ty(), IntToInt, i128_ty(), Value::Int(-1)),
        Value::Int(-1)
    );
    assert_eq!(
        run_cast(u8_ty(), IntToInt, u128_ty(), Value::Uint(200)),
        Value::Uint(200)
    );
    assert_eq!(
        run_cast(i128_ty(), IntToInt, mir::Ty::Int(mir::IntTy::Isize), Value::Int(-3)),
        Value::Int(-3)
    );
}

#[test]
fn float_to_i128_u128_casts_saturate() {
    use mir::CastKind::FloatToInt;
    // Host `as` casts are the oracle for every case.
    for v in [3.99f64, -3.99, 1e30, -1e30, 1e39, -1e39, 0.0] {
        assert_eq!(
            run_cast(f64_ty(), FloatToInt, i128_ty(), Value::Float(v)),
            Value::Int(v as i128),
            "f64 {v} as i128"
        );
        assert_eq!(
            run_cast(f64_ty(), FloatToInt, u128_ty(), Value::Float(v)),
            Value::Uint(v as u128),
            "f64 {v} as u128"
        );
    }
    // NaN becomes 0.
    assert_eq!(
        run_cast(f64_ty(), FloatToInt, i128_ty(), Value::Float(f64::NAN)),
        Value::Int(0)
    );
    assert_eq!(
        run_cast(f64_ty(), FloatToInt, u128_ty(), Value::Float(f64::NAN)),
        Value::Uint(0)
    );
    // f32 sources saturate identically.
    assert_eq!(
        run_cast(f32_ty(), FloatToInt, i128_ty(), Value::Float(1e30f32 as f64)),
        Value::Int(1e30f32 as i128)
    );
}

#[test]
fn i128_u128_to_float_casts() {
    use mir::CastKind::IntToFloat;
    assert_eq!(
        run_cast(i128_ty(), IntToFloat, f64_ty(), Value::Int(i128::MAX)),
        Value::Float(i128::MAX as f64)
    );
    assert_eq!(
        run_cast(u128_ty(), IntToFloat, f64_ty(), Value::Uint(u128::MAX)),
        Value::Float(u128::MAX as f64)
    );
    // u128::MAX as f32 overflows the f32 range and rounds to infinity,
    // exactly as the host cast does.
    assert_eq!(
        run_cast(u128_ty(), IntToFloat, f32_ty(), Value::Uint(u128::MAX)),
        Value::Float((u128::MAX as f32) as f64)
    );
    // 2^24 + 1 rounds to 2^24 in f32 (no double rounding via f64).
    assert_eq!(
        run_cast(i128_ty(), IntToFloat, f32_ty(), Value::Int(16_777_217)),
        Value::Float(16_777_216.0)
    );
}

// ----- aggregates: literals, repeats, switch -----------------------------------

#[test]
fn i128_array_literal_and_repeat() {
    // fn main() -> [i128; 2] { _0 = [const MIN, const -7]; return }
    let lit = mir::Body::new(
        decls(&[array_ty(i128_ty(), 2)]),
        0,
        vec![block(
            vec![assign(
                place(0),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Array(i128_ty())),
                    vec![const_i128(i128::MIN), const_i128(-7)],
                ),
            )],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", lit);
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).unwrap(),
        Value::Tuple(vec![Value::Int(i128::MIN), Value::Int(-7)])
    );

    // fn main() -> [i128; 1] { _0 = [const MIN; 1]; return }
    let rep = mir::Body::new(
        decls(&[array_ty(i128_ty(), 1)]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Repeat(const_i128(i128::MIN), 1))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", rep);
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).unwrap(),
        Value::Tuple(vec![Value::Int(i128::MIN)])
    );

    // fn main() -> [u128; 2] { _0 = [const MAX; 2]; return }
    let rep_u = mir::Body::new(
        decls(&[array_ty(u128_ty(), 2)]),
        0,
        vec![block(
            vec![assign(place(0), mir::Rvalue::Repeat(const_u128(u128::MAX), 2))],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", rep_u);
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).unwrap(),
        Value::Tuple(vec![Value::Uint(u128::MAX), Value::Uint(u128::MAX)])
    );
}

#[test]
fn switch_int_on_i128_extremes() {
    // fn f(_1: i128) -> i128: match _1 { MIN => 1, 0 => 2, MAX => 3, _ => 4 }
    let body = mir::Body::new(
        decls(&[i128_ty(), i128_ty()]),
        1,
        vec![
            block(
                vec![],
                mir::TerminatorKind::SwitchInt {
                    discr: copy(1),
                    targets: mir::SwitchTargets::new(
                        vec![i128::MIN as u128, 0, i128::MAX as u128],
                        vec![mir::BasicBlock(1), mir::BasicBlock(2), mir::BasicBlock(3)],
                        mir::BasicBlock(4),
                    ),
                },
            ),
            block(vec![assign(place(0), mir::Rvalue::Use(const_i128(1)))], ret()),
            block(vec![assign(place(0), mir::Rvalue::Use(const_i128(2)))], ret()),
            block(vec![assign(place(0), mir::Rvalue::Use(const_i128(3)))], ret()),
            block(vec![assign(place(0), mir::Rvalue::Use(const_i128(4)))], ret()),
        ],
    );
    let mut prog = MirProgram::new();
    prog.add("f", body);
    for (input, want) in [
        (i128::MIN, 1),
        (0, 2),
        (i128::MAX, 3),
        (-1, 4),
        (42, 4),
    ] {
        assert_eq!(
            interpret_mir(&prog, "f", vec![Value::Int(input)]).unwrap(),
            Value::Int(want),
            "switch on {input}"
        );
    }
}

// ----- constant decoding: the 128-bit sign paths --------------------------------

fn scalar(v: u128, size: u8) -> mir::ConstValue {
    mir::ConstValue::Scalar(mir::Scalar::Int { data: v, size })
}

fn const_op(v: mir::ConstValue) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(v))
}

fn use_op(op: mir::Operand) -> mir::Rvalue {
    mir::Rvalue::Use(op)
}

/// A promoted array constant, in the shape the driver produces for
/// `[5i128, -7i128]` in temporary position: a by-reference tuple of
/// raw scalar elements. The elements must keep their signedness —
/// before the fix, the negative element decoded as its unsigned
/// 2^128-magnitude pattern.
#[test]
fn promoted_array_constant_i128_elements_stay_signed() {
    // fn main() -> [i128; 2] { _1 = const &[5, -7]; _0 = copy *_1; return }
    let promoted = mir::ConstValue::Ref(Box::new(mir::ConstValue::Tuple(vec![
        scalar(5, 16),
        scalar((-7i128) as u128, 16),
    ])));
    let body = mir::Body::new(
        decls(&[array_ty(i128_ty(), 2), ref_ty(array_ty(i128_ty(), 2))]),
        0,
        vec![block(
            vec![
                assign(place(1), use_op(const_op(promoted))),
                assign(
                    place(0),
                    use_op(mir::Operand::Copy(mir::Place {
                        local: mir::Local(1),
                        projection: vec![mir::PlaceElem::Deref],
                    })),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).unwrap(),
        Value::Tuple(vec![Value::Int(5), Value::Int(-7)])
    );
}

/// The same promoted shape for `u128`: the element with its top bit
/// set must stay unsigned (`u128::MAX`, not -1).
#[test]
fn promoted_array_constant_u128_elements_stay_unsigned() {
    let promoted = mir::ConstValue::Ref(Box::new(mir::ConstValue::Tuple(vec![
        scalar(u128::MAX, 16),
        scalar(1, 16),
    ])));
    let body = mir::Body::new(
        decls(&[array_ty(u128_ty(), 2), ref_ty(array_ty(u128_ty(), 2))]),
        0,
        vec![block(
            vec![
                assign(place(1), use_op(const_op(promoted))),
                assign(
                    place(0),
                    use_op(mir::Operand::Copy(mir::Place {
                        local: mir::Local(1),
                        projection: vec![mir::PlaceElem::Deref],
                    })),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).unwrap(),
        Value::Tuple(vec![Value::Uint(u128::MAX), Value::Uint(1)])
    );
}

/// A context-free 16-byte scalar (here: a field of an ADT constant,
/// whose fields convert without a type context) with its sign bit
/// set is a negative `i128`, at 128 bits just as at every narrower
/// width.
#[test]
fn context_free_i128_scalar_decodes_signed() {
    // fn main() -> i128 { _1 = const Adt{0, [MIN-pattern]}; _0 = copy (_1.0); return }
    let adt_ty = mir::Ty::Adt(mir::DefId::new(0, 901), vec![]);
    let body = mir::Body::new(
        decls(&[i128_ty(), adt_ty]),
        0,
        vec![block(
            vec![
                assign(
                    place(1),
                    use_op(const_op(mir::ConstValue::Adt {
                        variant: 0,
                        fields: vec![scalar(i128::MIN as u128, 16)],
                    })),
                ),
                assign(
                    place(0),
                    use_op(mir::Operand::Copy(field(1, 0, i128_ty()))),
                ),
            ],
            ret(),
        )],
    );
    let mut prog = MirProgram::new();
    prog.add("main", body);
    assert_eq!(
        interpret_mir(&prog, "main", vec![]).unwrap(),
        Value::Int(i128::MIN)
    );
}
