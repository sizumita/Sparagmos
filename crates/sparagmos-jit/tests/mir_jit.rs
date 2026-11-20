//! MIR front-end tests: hand-built `sparagmos-mir` bodies in the
//! shapes the driver produces from the silent corpus programs
//! (checked arithmetic + `Assert`, recursion, loops, guarded
//! division), executed on the JIT and on the interpreter — the
//! semantic oracle — with identical inputs. Results, including
//! guest-panic messages, must agree exactly.

use std::collections::HashMap;

use sparagmos_interp::{InterpError, MirRunError, Value};
use sparagmos_jit::{JitCallError, JitError, JitProgram, MirFunctionInput, MirProgramInput};
use sparagmos_mir as mir;

// ----- small builders --------------------------------------------------------

fn def(index: u32) -> mir::DefId {
    mir::DefId { krate: 0, index }
}

fn bb(stmts: Vec<mir::StatementKind>, term: mir::TerminatorKind) -> mir::BasicBlockData {
    mir::BasicBlockData {
        statements: stmts.into_iter().map(mir::Statement::new).collect(),
        terminator: Some(mir::Terminator::new(term)),
        is_cleanup: false,
    }
}

fn assign(place: mir::Place, rv: mir::Rvalue) -> mir::StatementKind {
    mir::StatementKind::Assign(Box::new((place, rv)))
}

fn pl(n: u32) -> mir::Place {
    mir::Place::from_local(mir::Local(n))
}

fn fld(n: u32, field: u32, ty: mir::Ty) -> mir::Place {
    pl(n).field(field, ty)
}

fn cp(n: u32) -> mir::Operand {
    mir::Operand::Copy(pl(n))
}

fn cpf(n: u32, field: u32, ty: mir::Ty) -> mir::Operand {
    mir::Operand::Copy(fld(n, field, ty))
}

fn mvf(n: u32, field: u32, ty: mir::Ty) -> mir::Operand {
    mir::Operand::Move(fld(n, field, ty))
}

fn iconst(data: u128, size: u8) -> mir::Operand {
    mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Scalar(
        mir::Scalar::Int { data, size },
    )))
}

fn i32c(v: i32) -> mir::Operand {
    iconst(v as u32 as u128, 4)
}
fn i64c(v: i64) -> mir::Operand {
    iconst(v as u64 as u128, 8)
}
fn u64c(v: u64) -> mir::Operand {
    iconst(v as u128, 8)
}

fn fndef(d: mir::DefId) -> mir::Operand {
    mir::Operand::constant(mir::Constant {
        literal: mir::ConstKind::Ty(mir::Ty::FnDef(d, vec![])),
        user_ty: None,
    })
}

fn assert_overflow(
    flag_place: mir::Place,
    op: mir::BinOp,
    target: u32,
) -> mir::TerminatorKind {
    mir::TerminatorKind::Assert {
        cond: mir::Operand::Copy(flag_place),
        expected: false,
        msg: Box::new(mir::AssertMessage::Overflow(
            op,
            Box::new(cp(1)),
            Box::new(cp(1)),
        )),
        target: mir::BasicBlock(target),
        unwind: mir::UnwindAction::Continue,
    }
}

fn call_term(
    func: mir::Operand,
    args: Vec<mir::Operand>,
    dest: u32,
    target: u32,
) -> mir::TerminatorKind {
    mir::TerminatorKind::Call {
        func,
        args,
        destination: pl(dest),
        target: Some(mir::BasicBlock(target)),
        unwind: mir::UnwindAction::Continue,
        call_source: mir::CallSource::Normal,
    }
}

fn locals(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn i(ty: mir::IntTy) -> mir::Ty {
    mir::Ty::Int(ty)
}
fn u(ty: mir::UintTy) -> mir::Ty {
    mir::Ty::Uint(ty)
}

// ----- dual-engine harness ---------------------------------------------------

struct FnSpec {
    name: &'static str,
    def: u32,
    body: mir::Body,
}

fn spec(name: &'static str, def: u32, arg_count: usize, tys: &[mir::Ty], blocks: Vec<mir::BasicBlockData>) -> FnSpec {
    FnSpec {
        name,
        def,
        body: mir::Body::new(locals(tys), arg_count, blocks),
    }
}

fn build(specs: &[FnSpec]) -> (JitProgram, sparagmos_interp::MirProgram) {
    let names: HashMap<mir::DefId, String> = specs
        .iter()
        .map(|s| (def(s.def), s.name.to_string()))
        .collect();
    let jit_in = MirProgramInput {
        functions: specs
            .iter()
            .map(|s| MirFunctionInput {
                name: s.name.to_string(),
                def_id: Some(def(s.def)),
                body: s.body.clone(),
            })
            .collect(),
        names: names.clone(),
    };
    let interp_in = sparagmos_interp::MirProgram {
        functions: specs
            .iter()
            .map(|s| sparagmos_interp::MirFunction {
                name: s.name.to_string(),
                def_id: Some(def(s.def)),
                body: s.body.clone(),
            })
            .collect(),
        fn_names: names,
    };
    let jit = JitProgram::compile(&jit_in).expect("JIT compile failed");
    (jit, interp_in)
}

fn jit_call(jit: &JitProgram, name: &str, args: &[i64]) -> Result<i64, String> {
    jit.call(name, args).map_err(|e| match e {
        JitCallError::Panic(rec) => rec.message,
        other => other.to_string(),
    })
}

fn interp_call(
    prog: &sparagmos_interp::MirProgram,
    name: &str,
    args: Vec<Value>,
) -> Result<i64, String> {
    match sparagmos_interp::interpret_mir(prog, name, args) {
        Ok(v) => Ok(match v {
            Value::Int(x) => x as i64,
            Value::Uint(x) => x as i64,
            Value::Bool(b) => b as i64,
            Value::Unit => 0,
            other => panic!("unexpected interp result {other:?}"),
        }),
        Err(MirRunError::Interp(InterpError::Panic { message })) => Err(message),
        Err(other) => panic!("unexpected interp error: {other}"),
    }
}

/// Both engines must produce `expected` (a value, or a guest panic
/// with exactly this message).
fn check(
    jit: &JitProgram,
    prog: &sparagmos_interp::MirProgram,
    name: &str,
    jit_args: &[i64],
    interp_args: Vec<Value>,
    expected: Result<i64, &str>,
) {
    let expected = expected.map_err(|s| s.to_string());
    assert_eq!(jit_call(jit, name, jit_args), expected, "JIT result for {name}");
    assert_eq!(
        interp_call(prog, name, interp_args),
        expected,
        "interp result for {name}"
    );
}

fn int(v: i128) -> Value {
    Value::Int(v)
}
fn uint(v: u128) -> Value {
    Value::Uint(v)
}

// ----- the silent_panic shape: checked add + Assert, called from main --------

fn boom_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    let pair = mir::Ty::Tuple(vec![i(mir::IntTy::I32), mir::Ty::Bool]);
    let boom = spec(
        "boom",
        0,
        1,
        &[i(mir::IntTy::I32), i(mir::IntTy::I32), pair.clone()],
        vec![
            bb(
                vec![assign(
                    pl(2),
                    mir::Rvalue::CheckedBinaryOp(
                        mir::BinOp::Add,
                        Box::new((cp(1), cp(1))),
                    ),
                )],
                assert_overflow(fld(2, 1, mir::Ty::Bool), mir::BinOp::Add, 1),
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(mvf(2, 0, i(mir::IntTy::I32))))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let main = spec(
        "main",
        1,
        0,
        &[mir::Ty::Tuple(vec![]), i(mir::IntTy::I32)],
        vec![
            bb(vec![], call_term(fndef(def(0)), vec![i32c(2_000_000_000)], 1, 1)),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    build(&[boom, main])
}

#[test]
fn checked_add_ok_and_overflow_panic() {
    let (jit, prog) = boom_program();
    check(&jit, &prog, "boom", &[21], vec![int(21)], Ok(42));
    check(
        &jit,
        &prog,
        "boom",
        &[2_000_000_000],
        vec![int(2_000_000_000)],
        Err("assertion failed: Overflow"),
    );
    // The panic propagates out of the caller, as in the interpreter.
    check(
        &jit,
        &prog,
        "main",
        &[],
        vec![],
        Err("assertion failed: Overflow"),
    );
}

// ----- recursion: fib with checked sub/add and SwitchInt ---------------------

fn fib_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    let u64t = u(mir::UintTy::U64);
    let pair = mir::Ty::Tuple(vec![u64t.clone(), mir::Ty::Bool]);
    let fib = spec(
        "fib",
        0,
        1,
        &[
            u64t.clone(),
            u64t.clone(),
            mir::Ty::Bool,
            pair.clone(),
            u64t.clone(),
            u64t.clone(),
            pair.clone(),
            u64t.clone(),
            u64t.clone(),
            pair.clone(),
        ],
        vec![
            // bb0: _2 = _1 < 2; switchInt(_2) -> [0: bb2] otherwise bb1
            bb(
                vec![assign(
                    pl(2),
                    mir::Rvalue::BinaryOp(mir::BinOp::Lt, Box::new((cp(1), u64c(2)))),
                )],
                mir::TerminatorKind::SwitchInt {
                    discr: cp(2),
                    targets: mir::SwitchTargets::new(vec![0], vec![mir::BasicBlock(2)], mir::BasicBlock(1)),
                },
            ),
            // bb1: return _1
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(1)))],
                mir::TerminatorKind::Return,
            ),
            // bb2: _3 = checked_sub(_1, 1); assert !_3.1 -> bb3
            bb(
                vec![assign(
                    pl(3),
                    mir::Rvalue::CheckedBinaryOp(mir::BinOp::Sub, Box::new((cp(1), u64c(1)))),
                )],
                assert_overflow(fld(3, 1, mir::Ty::Bool), mir::BinOp::Sub, 3),
            ),
            // bb3: _4 = _3.0; _5 = call fib(_4) -> bb4
            bb(
                vec![assign(pl(4), mir::Rvalue::Use(mvf(3, 0, u64t.clone())))],
                call_term(fndef(def(0)), vec![cp(4)], 5, 4),
            ),
            // bb4: _6 = checked_sub(_1, 2); assert -> bb5
            bb(
                vec![assign(
                    pl(6),
                    mir::Rvalue::CheckedBinaryOp(mir::BinOp::Sub, Box::new((cp(1), u64c(2)))),
                )],
                assert_overflow(fld(6, 1, mir::Ty::Bool), mir::BinOp::Sub, 5),
            ),
            // bb5: _7 = _6.0; _8 = call fib(_7) -> bb6
            bb(
                vec![assign(pl(7), mir::Rvalue::Use(mvf(6, 0, u64t.clone())))],
                call_term(fndef(def(0)), vec![cp(7)], 8, 6),
            ),
            // bb6: _9 = checked_add(_5, _8); assert -> bb7
            bb(
                vec![assign(
                    pl(9),
                    mir::Rvalue::CheckedBinaryOp(mir::BinOp::Add, Box::new((cp(5), cp(8)))),
                )],
                assert_overflow(fld(9, 1, mir::Ty::Bool), mir::BinOp::Add, 7),
            ),
            // bb7: return _9.0
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(mvf(9, 0, u64t.clone())))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let main = spec(
        "main",
        1,
        0,
        &[mir::Ty::Tuple(vec![]), u64t.clone()],
        vec![
            bb(vec![], call_term(fndef(def(0)), vec![u64c(10)], 1, 1)),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    build(&[fib, main])
}

#[test]
fn recursive_fib() {
    let (jit, prog) = fib_program();
    for (n, want) in [(0u64, 0i64), (1, 1), (2, 1), (10, 55), (15, 610), (20, 6765)] {
        check(&jit, &prog, "fib", &[n as i64], vec![uint(n as u128)], Ok(want));
    }
    check(&jit, &prog, "main", &[], vec![], Ok(0));
}

// ----- the silent_loop shape -------------------------------------------------

#[test]
fn loop_sum() {
    let u64t = u(mir::UintTy::U64);
    let pair = mir::Ty::Tuple(vec![u64t.clone(), mir::Ty::Bool]);
    let sum = spec(
        "sum",
        0,
        0,
        &[
            u64t.clone(),
            u64t.clone(),
            u64t.clone(),
            mir::Ty::Bool,
            pair.clone(),
            pair.clone(),
        ],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Use(u64c(0))),
                    assign(pl(2), mir::Rvalue::Use(u64c(0))),
                ],
                mir::TerminatorKind::Goto { target: mir::BasicBlock(1) },
            ),
            bb(
                vec![assign(
                    pl(3),
                    mir::Rvalue::BinaryOp(mir::BinOp::Lt, Box::new((cp(1), u64c(100)))),
                )],
                mir::TerminatorKind::SwitchInt {
                    discr: cp(3),
                    targets: mir::SwitchTargets::new(vec![0], vec![mir::BasicBlock(5)], mir::BasicBlock(2)),
                },
            ),
            bb(
                vec![assign(
                    pl(4),
                    mir::Rvalue::CheckedBinaryOp(mir::BinOp::Add, Box::new((cp(2), cp(1)))),
                )],
                assert_overflow(fld(4, 1, mir::Ty::Bool), mir::BinOp::Add, 3),
            ),
            bb(
                vec![
                    assign(pl(2), mir::Rvalue::Use(mvf(4, 0, u64t.clone()))),
                    assign(
                        pl(5),
                        mir::Rvalue::CheckedBinaryOp(mir::BinOp::Add, Box::new((cp(1), u64c(1)))),
                    ),
                ],
                assert_overflow(fld(5, 1, mir::Ty::Bool), mir::BinOp::Add, 4),
            ),
            bb(
                vec![assign(pl(1), mir::Rvalue::Use(mvf(5, 0, u64t.clone())))],
                mir::TerminatorKind::Goto { target: mir::BasicBlock(1) },
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(2)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[sum]);
    check(&jit, &prog, "sum", &[], vec![], Ok(4950));
}

// ----- the silent_arith shape: casts and Assert-guarded division -------------

#[test]
fn casts_and_guarded_division() {
    let i32t = i(mir::IntTy::I32);
    let i64t = i(mir::IntTy::I64);
    // fn calc(a: i32, b: i64) -> i64 { let x = a as i64 * b; (x / b) + (x % b) }
    let calc = spec(
        "calc",
        0,
        2,
        &[
            i64t.clone(),
            i32t.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
            mir::Ty::Bool,
            i64t.clone(),
            i64t.clone(),
        ],
        vec![
            bb(
                vec![
                    assign(
                        pl(3),
                        mir::Rvalue::Cast(mir::CastKind::IntToInt, cp(1), i64t.clone()),
                    ),
                    assign(
                        pl(4),
                        mir::Rvalue::BinaryOp(mir::BinOp::Mul, Box::new((cp(3), cp(2)))),
                    ),
                    assign(
                        pl(5),
                        mir::Rvalue::BinaryOp(mir::BinOp::Eq, Box::new((cp(2), i64c(0)))),
                    ),
                ],
                mir::TerminatorKind::Assert {
                    cond: cp(5),
                    expected: false,
                    msg: Box::new(mir::AssertMessage::DivisionByZero(Box::new(cp(2)))),
                    target: mir::BasicBlock(1),
                    unwind: mir::UnwindAction::Continue,
                },
            ),
            bb(
                vec![
                    assign(
                        pl(6),
                        mir::Rvalue::BinaryOp(mir::BinOp::Div, Box::new((cp(4), cp(2)))),
                    ),
                    assign(
                        pl(5),
                        mir::Rvalue::BinaryOp(mir::BinOp::Eq, Box::new((cp(2), i64c(0)))),
                    ),
                ],
                mir::TerminatorKind::Assert {
                    cond: cp(5),
                    expected: false,
                    msg: Box::new(mir::AssertMessage::RemainderByZero(Box::new(cp(2)))),
                    target: mir::BasicBlock(2),
                    unwind: mir::UnwindAction::Continue,
                },
            ),
            bb(
                vec![
                    assign(
                        pl(7),
                        mir::Rvalue::BinaryOp(mir::BinOp::Rem, Box::new((cp(4), cp(2)))),
                    ),
                    assign(
                        pl(0),
                        mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(6), cp(7)))),
                    ),
                ],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[calc]);
    check(&jit, &prog, "calc", &[7, 3], vec![int(7), int(3)], Ok(7));
    check(&jit, &prog, "calc", &[-7, 2], vec![int(-7), int(2)], Ok(-7));
    check(
        &jit,
        &prog,
        "calc",
        &[5, 0],
        vec![int(5), int(0)],
        Err("assertion failed: DivisionByZero"),
    );
}

// ----- literal-operand guards (the real silent_arith shape) ------------------
//
// rustc expands the division check over the *literal* divisor:
// `_t = Eq(const 4, const 0)` — neither operand has a place, so the
// comparison's type comes from the literals. (Typing it by the bool
// destination collapses 4 to 1 and the guard misfires.)

#[test]
fn literal_division_guard() {
    let i32t = i(mir::IntTy::I32);
    let f = spec(
        "lit_guard",
        0,
        0,
        &[
            i32t.clone(),
            mir::Ty::Tuple(vec![i32t.clone(), mir::Ty::Bool]),
            i32t.clone(),
            mir::Ty::Bool,
        ],
        vec![
            bb(
                vec![
                    assign(
                        pl(1),
                        mir::Rvalue::CheckedBinaryOp(
                            mir::BinOp::Sub,
                            Box::new((i32c(40), i32c(10))),
                        ),
                    ),
                ],
                assert_overflow(fld(1, 1, mir::Ty::Bool), mir::BinOp::Sub, 1),
            ),
            bb(
                vec![
                    assign(pl(2), mir::Rvalue::Use(mvf(1, 0, i32t.clone()))),
                    assign(
                        pl(3),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Eq,
                            Box::new((i32c(4), i32c(0))),
                        ),
                    ),
                ],
                mir::TerminatorKind::Assert {
                    cond: mir::Operand::Move(pl(3)),
                    expected: false,
                    msg: Box::new(mir::AssertMessage::DivisionByZero(Box::new(i32c(4)))),
                    target: mir::BasicBlock(2),
                    unwind: mir::UnwindAction::Continue,
                },
            ),
            bb(
                vec![assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(mir::BinOp::Div, Box::new((cp(2), i32c(4)))),
                )],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[f]);
    check(&jit, &prog, "lit_guard", &[], vec![], Ok(7));
}

// ----- unguarded division: the lowering's own guards -------------------------

fn div_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    let i64t = i(mir::IntTy::I64);
    let mk = |name: &'static str, def: u32, op: mir::BinOp| {
        spec(
            name,
            def,
            2,
            &[i64t.clone(), i64t.clone(), i64t.clone()],
            vec![bb(
                vec![assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(op, Box::new((cp(1), cp(2)))),
                )],
                mir::TerminatorKind::Return,
            )],
        )
    };
    build(&[mk("rawdiv", 0, mir::BinOp::Div), mk("rawrem", 1, mir::BinOp::Rem)])
}

#[test]
fn unguarded_division_panics() {
    let (jit, prog) = div_program();
    check(&jit, &prog, "rawdiv", &[10, 3], vec![int(10), int(3)], Ok(3));
    check(&jit, &prog, "rawrem", &[10, 3], vec![int(10), int(3)], Ok(1));
    check(
        &jit,
        &prog,
        "rawdiv",
        &[1, 0],
        vec![int(1), int(0)],
        Err("attempt to divide by zero"),
    );
    check(
        &jit,
        &prog,
        "rawrem",
        &[1, 0],
        vec![int(1), int(0)],
        Err("attempt to calculate the remainder with a divisor of zero"),
    );
    check(
        &jit,
        &prog,
        "rawdiv",
        &[i64::MIN, -1],
        vec![int(i64::MIN as i128), int(-1)],
        Err("arithmetic overflow in Div"),
    );
    check(
        &jit,
        &prog,
        "rawrem",
        &[i64::MIN, -1],
        vec![int(i64::MIN as i128), int(-1)],
        Err("arithmetic overflow in Rem"),
    );
}

// ----- plain arithmetic: the interpreter's overflow panics -------------------

#[test]
fn plain_arithmetic_overflow_panics() {
    let i32t = i(mir::IntTy::I32);
    let add_i32 = spec(
        "add_i32",
        0,
        2,
        &[i32t.clone(), i32t.clone(), i32t.clone()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(1), cp(2)))),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    let u8t = u(mir::UintTy::U8);
    let add_u8 = spec(
        "add_u8",
        1,
        2,
        &[u8t.clone(), u8t.clone(), u8t.clone()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(1), cp(2)))),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    let i16t = i(mir::IntTy::I16);
    let mul_i16 = spec(
        "mul_i16",
        2,
        2,
        &[i16t.clone(), i16t.clone(), i16t.clone()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(mir::BinOp::Mul, Box::new((cp(1), cp(2)))),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[add_i32, add_u8, mul_i16]);
    check(&jit, &prog, "add_i32", &[40, 2], vec![int(40), int(2)], Ok(42));
    check(
        &jit,
        &prog,
        "add_i32",
        &[2_000_000_000, 2_000_000_000],
        vec![int(2_000_000_000), int(2_000_000_000)],
        Err("arithmetic overflow in Add"),
    );
    check(&jit, &prog, "add_u8", &[100, 55], vec![uint(100), uint(55)], Ok(155));
    check(
        &jit,
        &prog,
        "add_u8",
        &[200, 200],
        vec![uint(200), uint(200)],
        Err("arithmetic overflow in Add"),
    );
    check(&jit, &prog, "mul_i16", &[12, 11], vec![int(12), int(11)], Ok(132));
    check(
        &jit,
        &prog,
        "mul_i16",
        &[1000, 100],
        vec![int(1000), int(100)],
        Err("arithmetic overflow in Mul"),
    );
}

// ----- checked subtraction on u64 (the flag + Assert path) -------------------

#[test]
fn checked_sub_u64() {
    let u64t = u(mir::UintTy::U64);
    let pair = mir::Ty::Tuple(vec![u64t.clone(), mir::Ty::Bool]);
    let f = spec(
        "sub_u64",
        0,
        2,
        &[u64t.clone(), u64t.clone(), u64t.clone(), pair],
        vec![
            bb(
                vec![assign(
                    pl(3),
                    mir::Rvalue::CheckedBinaryOp(mir::BinOp::Sub, Box::new((cp(1), cp(2)))),
                )],
                assert_overflow(fld(3, 1, mir::Ty::Bool), mir::BinOp::Sub, 1),
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(mvf(3, 0, u64t.clone())))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[f]);
    check(&jit, &prog, "sub_u64", &[10, 3], vec![uint(10), uint(3)], Ok(7));
    check(
        &jit,
        &prog,
        "sub_u64",
        &[0, 1],
        vec![uint(0), uint(1)],
        Err("assertion failed: Overflow"),
    );
    check(
        &jit,
        &prog,
        "sub_u64",
        &[u64::MAX as i64, u64::MAX as i64],
        vec![uint(u64::MAX as u128), uint(u64::MAX as u128)],
        Ok(0),
    );
}

// ----- shifts ------------------------------------------------------------------

fn shift_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    let u32t = u(mir::UintTy::U32);
    let i32t = i(mir::IntTy::I32);
    let shl = spec(
        "shl_u32",
        0,
        2,
        &[u32t.clone(), u32t.clone(), u32t.clone()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(mir::BinOp::Shl, Box::new((cp(1), cp(2)))),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    let shr = spec(
        "shr_i32",
        1,
        2,
        &[i32t.clone(), i32t.clone(), i32t.clone()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(mir::BinOp::Shr, Box::new((cp(1), cp(2)))),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // Checked shift, flag and value read back out of the pair.
    let pair = mir::Ty::Tuple(vec![u32t.clone(), mir::Ty::Bool]);
    let cshl_flag = spec(
        "cshl_flag",
        2,
        2,
        &[u(mir::UintTy::U64), u32t.clone(), u32t.clone(), pair.clone(), u(mir::UintTy::U64)],
        vec![bb(
            vec![
                assign(
                    pl(3),
                    mir::Rvalue::CheckedBinaryOp(mir::BinOp::Shl, Box::new((cp(1), cp(2)))),
                ),
                assign(
                    pl(0),
                    mir::Rvalue::Cast(
                        mir::CastKind::IntToInt,
                        cpf(3, 1, mir::Ty::Bool),
                        u(mir::UintTy::U64),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let cshl_val = spec(
        "cshl_val",
        3,
        2,
        &[u32t.clone(), u32t.clone(), u32t.clone(), pair],
        vec![bb(
            vec![
                assign(
                    pl(3),
                    mir::Rvalue::CheckedBinaryOp(mir::BinOp::Shl, Box::new((cp(1), cp(2)))),
                ),
                assign(pl(0), mir::Rvalue::Use(mvf(3, 0, u32t.clone()))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    build(&[shl, shr, cshl_flag, cshl_val])
}

#[test]
fn shifts_plain_and_checked() {
    let (jit, prog) = shift_program();
    check(&jit, &prog, "shl_u32", &[1, 10], vec![uint(1), uint(10)], Ok(1024));
    check(&jit, &prog, "shl_u32", &[3, 4], vec![uint(3), uint(4)], Ok(48));
    check(
        &jit,
        &prog,
        "shl_u32",
        &[1, 32],
        vec![uint(1), uint(32)],
        Err("attempt to shift by 32 bits, which overflows the type width 32"),
    );
    check(
        &jit,
        &prog,
        "shl_u32",
        &[0x1_0000, 16],
        vec![uint(0x1_0000), uint(16)],
        Err("arithmetic overflow in Shl"),
    );
    check(&jit, &prog, "shr_i32", &[-16, 2], vec![int(-16), int(2)], Ok(-4));
    check(
        &jit,
        &prog,
        "shr_i32",
        &[16, 33],
        vec![int(16), int(33)],
        Err("attempt to shift by 33 bits, which overflows the type width 32"),
    );
    // Checked form: no panic, the flag and the wrapped value instead.
    check(&jit, &prog, "cshl_flag", &[1, 32], vec![uint(1), uint(32)], Ok(1));
    check(&jit, &prog, "cshl_flag", &[1, 4], vec![uint(1), uint(4)], Ok(0));
    check(&jit, &prog, "cshl_val", &[1, 32], vec![uint(1), uint(32)], Ok(1));
    check(
        &jit,
        &prog,
        "cshl_val",
        &[0x8000, 16],
        vec![uint(0x8000), uint(16)],
        Ok(0x8000_0000),
    );
}

// ----- negation ------------------------------------------------------------------

#[test]
fn negation() {
    let i32t = i(mir::IntTy::I32);
    let neg = spec(
        "neg_i32",
        0,
        1,
        &[i32t.clone(), i32t.clone()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::UnaryOp(mir::UnOp::Neg, cp(1)),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[neg]);
    check(&jit, &prog, "neg_i32", &[-5], vec![int(-5)], Ok(5));
    check(
        &jit,
        &prog,
        "neg_i32",
        &[i32::MIN as i64],
        vec![int(i32::MIN as i128)],
        Err("arithmetic overflow in Neg"),
    );
}

// ----- SwitchInt on a char --------------------------------------------------------

#[test]
fn switch_on_char() {
    let f = spec(
        "is_a",
        0,
        1,
        &[u(mir::UintTy::U64), mir::Ty::Char, u(mir::UintTy::U64)],
        vec![
            bb(
                vec![],
                mir::TerminatorKind::SwitchInt {
                    discr: cp(1),
                    targets: mir::SwitchTargets::new(
                        vec![97],
                        vec![mir::BasicBlock(1)],
                        mir::BasicBlock(2),
                    ),
                },
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(u64c(1)))],
                mir::TerminatorKind::Return,
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(u64c(0)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[f]);
    check(&jit, &prog, "is_a", &[97], vec![uint(97)], Ok(1));
    check(&jit, &prog, "is_a", &[98], vec![uint(98)], Ok(0));
}

// ----- unsupported constructs are reported, never substituted ---------------------

#[test]
fn unsupported_constructs_report_reasons() {
    // A floating-point local.
    let float_fn = MirFunctionInput {
        name: "floaty".to_string(),
        def_id: Some(def(0)),
        body: mir::Body::new(
            locals(&[mir::Ty::Float(mir::FloatTy::F64), mir::Ty::Float(mir::FloatTy::F64)]),
            1,
            vec![bb(vec![], mir::TerminatorKind::Return)],
        ),
    };
    let err = JitProgram::compile(&MirProgramInput {
        functions: vec![float_fn],
        names: HashMap::new(),
    })
    .expect_err("float must be unsupported");
    match err {
        JitError::Unsupported(what) => assert!(what.contains("floating-point"), "{what}"),
        other => panic!("expected Unsupported, got {other}"),
    }

    // A 128-bit integer local.
    let wide_fn = MirFunctionInput {
        name: "wide".to_string(),
        def_id: Some(def(0)),
        body: mir::Body::new(
            locals(&[i(mir::IntTy::I128), i(mir::IntTy::I128)]),
            1,
            vec![bb(vec![], mir::TerminatorKind::Return)],
        ),
    };
    let err = JitProgram::compile(&MirProgramInput {
        functions: vec![wide_fn],
        names: HashMap::new(),
    })
    .expect_err("i128 must be unsupported");
    match err {
        JitError::Unsupported(what) => assert!(what.contains("128-bit"), "{what}"),
        other => panic!("expected Unsupported, got {other}"),
    }

    // A call to a function that is not part of the program.
    let caller = spec(
        "caller",
        0,
        0,
        &[mir::Ty::Tuple(vec![]), i(mir::IntTy::I32)],
        vec![
            bb(vec![], call_term(fndef(def(99)), vec![i32c(1)], 1, 1)),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    let mut names = HashMap::new();
    names.insert(def(99), "somewhere::else::missing".to_string());
    let err = JitProgram::compile(&MirProgramInput {
        functions: vec![MirFunctionInput {
            name: caller.name.to_string(),
            def_id: Some(def(0)),
            body: caller.body.clone(),
        }],
        names,
    })
    .expect_err("unknown callee must be unsupported");
    match err {
        JitError::Unsupported(what) => {
            assert!(what.contains("somewhere::else::missing"), "{what}")
        }
        other => panic!("expected Unsupported, got {other}"),
    }

    // A multi-word (tuple) argument.
    let tuple_arg = MirFunctionInput {
        name: "tup".to_string(),
        def_id: Some(def(0)),
        body: mir::Body::new(
            locals(&[
                i(mir::IntTy::I32),
                mir::Ty::Tuple(vec![i(mir::IntTy::I32), i(mir::IntTy::I32)]),
            ]),
            1,
            vec![bb(vec![], mir::TerminatorKind::Return)],
        ),
    };
    let err = JitProgram::compile(&MirProgramInput {
        functions: vec![tuple_arg],
        names: HashMap::new(),
    })
    .expect_err("tuple argument must be unsupported");
    match err {
        JitError::Unsupported(what) => assert!(what.contains("multi-word"), "{what}"),
        other => panic!("expected Unsupported, got {other}"),
    }
}

#[test]
fn call_api_errors() {
    let (jit, _prog) = boom_program();
    match jit.call("nope", &[]) {
        Err(JitCallError::UnknownFunction(n)) => assert_eq!(n, "nope"),
        other => panic!("expected UnknownFunction, got {other:?}"),
    }
    match jit.call("boom", &[]) {
        Err(JitCallError::WrongArgCount { expected: 1, got: 0, .. }) => {}
        other => panic!("expected WrongArgCount, got {other:?}"),
    }
    assert_eq!(jit.function_names(), vec!["boom", "main"]);
}

// ----- storage / annotation statements: present, no effect on values --------

fn annotated_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    // fn annotated(a: i32) -> i32 { _2 = a + 1; return _2 }, with the
    // block carrying the execution-no-op statements rustc emits around
    // real code: storage markers, a fake read, a place mention, a
    // user-type ascription, a coverage counter, a const-eval counter
    // and a plain nop. None of them may disturb the computed value.
    let i32t = i(mir::IntTy::I32);
    let annotated = spec(
        "annotated",
        0,
        1,
        &[i32t.clone(), i32t.clone(), i32t.clone()],
        vec![bb(
            vec![
                mir::StatementKind::StorageLive(mir::Local(2)),
                assign(
                    pl(2),
                    mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(1), i32c(1)))),
                ),
                mir::StatementKind::FakeRead(Box::new((mir::FakeReadCause::ForLet, pl(2)))),
                mir::StatementKind::PlaceMention(Box::new(pl(2))),
                mir::StatementKind::AscribeUserType(
                    Box::new((
                        pl(2),
                        mir::UserTypeProjection {
                            user_ty_index: 0,
                            projections: vec![],
                        },
                    )),
                    mir::Variance::Covariant,
                ),
                mir::StatementKind::Coverage,
                mir::StatementKind::ConstEvalCounter,
                mir::StatementKind::Nop,
                assign(pl(0), mir::Rvalue::Use(cp(2))),
                mir::StatementKind::StorageDead(mir::Local(2)),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    build(&[annotated])
}

#[test]
fn storage_and_annotation_statements_do_not_affect_execution() {
    let (jit, prog) = annotated_program();
    check(&jit, &prog, "annotated", &[41], vec![int(41)], Ok(42));
    check(&jit, &prog, "annotated", &[-1], vec![int(-1)], Ok(0));
}

// ----- Aggregate (tuple): fields read back -----------------------------------

fn tuple_aggregate_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    // fn combine(a: i32, b: i32) -> i32 {
    //   _3 = (a, b); _4 = _3.0 * 10; return _4 + _3.1
    // }
    let i32t = i(mir::IntTy::I32);
    let pair = mir::Ty::Tuple(vec![i32t.clone(), i32t.clone()]);
    let combine = spec(
        "combine",
        0,
        2,
        &[i32t.clone(), i32t.clone(), i32t.clone(), pair, i32t.clone()],
        vec![bb(
            vec![
                assign(
                    pl(3),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![cp(1), cp(2)],
                    ),
                ),
                assign(
                    pl(4),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Mul,
                        Box::new((cpf(3, 0, i32t.clone()), i32c(10))),
                    ),
                ),
                assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((cp(4), cpf(3, 1, i32t.clone()))),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    build(&[combine])
}

#[test]
fn tuple_aggregate_fields_read_back() {
    let (jit, prog) = tuple_aggregate_program();
    check(&jit, &prog, "combine", &[3, 4], vec![int(3), int(4)], Ok(34));
    check(&jit, &prog, "combine", &[5, 7], vec![int(5), int(7)], Ok(57));
}

// ----- Drop of a scalar: no code runs, execution continues at target --------

fn drop_scalar_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    // fn drop_scalar(a: i32) -> i32 { _2 = a; drop(_2) -> bb1; bb1: return a + 1 }
    let i32t = i(mir::IntTy::I32);
    let f = spec(
        "drop_scalar",
        0,
        1,
        &[i32t.clone(), i32t.clone(), i32t.clone()],
        vec![
            bb(
                vec![assign(pl(2), mir::Rvalue::Use(cp(1)))],
                mir::TerminatorKind::Drop {
                    place: pl(2),
                    target: mir::BasicBlock(1),
                    unwind: mir::UnwindAction::Continue,
                    replace: false,
                },
            ),
            bb(
                vec![assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(1), i32c(1)))),
                )],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    build(&[f])
}

#[test]
fn drop_of_scalar_jumps_to_target() {
    let (jit, prog) = drop_scalar_program();
    check(&jit, &prog, "drop_scalar", &[41], vec![int(41)], Ok(42));
}

// ----- Unreachable: a block that is present but never taken ------------------

fn unreachable_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    // fn keep(a: i32) -> i32 { if a < 0 { unreachable } else { a } }:
    // the Unreachable block is the taken-when-negative arm of the
    // switch; the checks below only exercise the non-negative arm, so
    // execution must return `a` untouched on both engines. The
    // behaviour when execution *does* reach the block is asserted for
    // the interpreter in mir_convert_tests (guest panic); on the JIT
    // the block compiles to a trap (see Issue #39).
    let i32t = i(mir::IntTy::I32);
    let keep = spec(
        "keep",
        0,
        1,
        &[i32t.clone(), i32t.clone(), mir::Ty::Bool],
        vec![
            bb(
                vec![assign(
                    pl(2),
                    mir::Rvalue::BinaryOp(mir::BinOp::Lt, Box::new((cp(1), i32c(0)))),
                )],
                mir::TerminatorKind::SwitchInt {
                    discr: cp(2),
                    targets: mir::SwitchTargets::new(
                        vec![0],
                        vec![mir::BasicBlock(1)],
                        mir::BasicBlock(2),
                    ),
                },
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(1)))],
                mir::TerminatorKind::Return,
            ),
            bb(vec![], mir::TerminatorKind::Unreachable),
        ],
    );
    build(&[keep])
}

#[test]
fn unreachable_block_not_taken_leaves_execution_intact() {
    let (jit, prog) = unreachable_program();
    check(&jit, &prog, "keep", &[5], vec![int(5)], Ok(5));
    check(&jit, &prog, "keep", &[0], vec![int(0)], Ok(0));
}
