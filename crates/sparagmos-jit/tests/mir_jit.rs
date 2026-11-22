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
        ty: None,
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
    build_with_table(specs, mir::AdtDiscriminants::new())
}

/// Like `build`, with a discriminant table shared by both engines.
fn build_with_table(
    specs: &[FnSpec],
    table: mir::AdtDiscriminants,
) -> (JitProgram, sparagmos_interp::MirProgram) {
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
        adt_discriminants: table.clone(),
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
        adt_discriminants: table,
        adt_layouts: Default::default(),
        coroutine_layouts: Default::default(),
        adt_shapes: Default::default(),
        adt_debug_shapes: Default::default(),
        thread_locals: Default::default(),
        adt_instance_shapes: Default::default(),
        adt_niches: Default::default(),
        source_files: Default::default(),
        track_caller_defs: Default::default(),
        fn_shim_locations: Default::default(),
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
        Err(MirRunError::Interp(InterpError::Panic { message, .. })) => Err(message),
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
        Err("attempt to add with overflow"),
    );
    // The panic propagates out of the caller, as in the interpreter.
    check(
        &jit,
        &prog,
        "main",
        &[],
        vec![],
        Err("attempt to add with overflow"),
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
        Err("attempt to divide by zero"),
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
fn plain_division_wraps_and_zero_panics() {
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
    // The plain forms wrap: MIN / -1 is MIN and MIN % -1 is 0 in
    // both engines (the panic belongs to the Assert after a
    // *checked* division, which this body does not contain).
    check(
        &jit,
        &prog,
        "rawdiv",
        &[i64::MIN, -1],
        vec![int(i64::MIN as i128), int(-1)],
        Ok(i64::MIN),
    );
    check(
        &jit,
        &prog,
        "rawrem",
        &[i64::MIN, -1],
        vec![int(i64::MIN as i128), int(-1)],
        Ok(0),
    );
}

// ----- plain arithmetic wraps (the panic is the checked form's Assert) --------

#[test]
fn plain_arithmetic_wraps() {
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
        Ok(-294_967_296),
    );
    check(&jit, &prog, "add_u8", &[100, 55], vec![uint(100), uint(55)], Ok(155));
    check(
        &jit,
        &prog,
        "add_u8",
        &[200, 200],
        vec![uint(200), uint(200)],
        Ok(144),
    );
    check(&jit, &prog, "mul_i16", &[12, 11], vec![int(12), int(11)], Ok(132));
    check(
        &jit,
        &prog,
        "mul_i16",
        &[1000, 100],
        vec![int(1000), int(100)],
        Ok(-31_072),
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
        Err("attempt to subtract with overflow"),
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
    // The plain form takes the amount modulo the width: shifting
    // by 32 is shifting by 0.
    check(
        &jit,
        &prog,
        "shl_u32",
        &[1, 32],
        vec![uint(1), uint(32)],
        Ok(1),
    );
    // Bits shifted out are simply lost, as natively.
    check(
        &jit,
        &prog,
        "shl_u32",
        &[0x1_0000, 16],
        vec![uint(0x1_0000), uint(16)],
        Ok(0),
    );
    check(
        &jit,
        &prog,
        "shl_u32",
        &[42, 28],
        vec![uint(42), uint(28)],
        Ok(0xA000_0000),
    );
    check(&jit, &prog, "shr_i32", &[-16, 2], vec![int(-16), int(2)], Ok(-4));
    check(
        &jit,
        &prog,
        "shr_i32",
        &[16, 33],
        vec![int(16), int(33)],
        Ok(8),
    );
    // Checked form: no panic, the flag and the wrapped value instead.
    // The flag reports the overshift only: bits shifted out leave
    // it clear.
    check(&jit, &prog, "cshl_flag", &[1, 32], vec![uint(1), uint(32)], Ok(1));
    check(&jit, &prog, "cshl_flag", &[1, 4], vec![uint(1), uint(4)], Ok(0));
    check(
        &jit,
        &prog,
        "cshl_flag",
        &[0x1_0000, 16],
        vec![uint(0x1_0000), uint(16)],
        Ok(0),
    );
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
        Ok(i32::MIN as i64),
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
    // A 16-bit floating-point local (f32 / f64 are supported; the
    // SSE-less formats are not).
    let float_fn = MirFunctionInput {
        name: "floaty".to_string(),
        def_id: Some(def(0)),
        body: mir::Body::new(
            locals(&[mir::Ty::Float(mir::FloatTy::F16), mir::Ty::Float(mir::FloatTy::F16)]),
            1,
            vec![bb(vec![], mir::TerminatorKind::Return)],
        ),
    };
    let err = JitProgram::compile(&MirProgramInput {
        functions: vec![float_fn],
        names: HashMap::new(),
        adt_discriminants: Default::default(),
    })
    .expect_err("f16 must be unsupported");
    match err {
        JitError::Unsupported(what) => assert!(what.contains("floating-point"), "{what}"),
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
        adt_discriminants: Default::default(),
    })
    .expect_err("unknown callee must be unsupported");
    match err {
        JitError::Unsupported(what) => {
            assert!(what.contains("somewhere::else::missing"), "{what}")
        }
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
    // switch. The non-negative checks exercise the return arm; the
    // negative checks reach the block, which both engines surface as
    // the guest panic `entered unreachable code`.
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

#[test]
fn unreachable_block_reached_is_a_guest_panic() {
    // Reaching the Unreachable block must return the interpreter's
    // guest panic from `JitProgram::call`, not trap: this used to
    // lower to int3 and kill the host process with SIGTRAP
    // (Issue #56), so this test could not even run to completion.
    let (jit, prog) = unreachable_program();
    check(
        &jit,
        &prog,
        "keep",
        &[-1],
        vec![int(-1)],
        Err("entered unreachable code"),
    );
    check(
        &jit,
        &prog,
        "keep",
        &[-42],
        vec![int(-42)],
        Err("entered unreachable code"),
    );
}

fn unreachable_callee_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    // fn dive() -> i32 { unreachable }
    // fn main() -> i32 { dive() }
    // The panic raised in the callee must propagate out of the
    // caller through the panic flag, as an `Assert` panic does.
    let i32t = i(mir::IntTy::I32);
    let dive = spec(
        "dive",
        0,
        0,
        &[i32t.clone()],
        vec![bb(vec![], mir::TerminatorKind::Unreachable)],
    );
    let main = spec(
        "main",
        1,
        0,
        &[mir::Ty::Tuple(vec![]), i32t],
        vec![
            bb(vec![], call_term(fndef(def(0)), vec![], 1, 1)),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    build(&[dive, main])
}

#[test]
fn unreachable_in_callee_propagates_as_guest_panic() {
    let (jit, prog) = unreachable_callee_program();
    check(
        &jit,
        &prog,
        "dive",
        &[],
        vec![],
        Err("entered unreachable code"),
    );
    check(
        &jit,
        &prog,
        "main",
        &[],
        vec![],
        Err("entered unreachable code"),
    );
}

// ----- Repeat: arrays built by repeating one element -------------------------

fn arr_i32(n: u64) -> mir::Ty {
    mir::Ty::Array(Box::new(i(mir::IntTy::I32)), mir::ArrayLen::Known(n))
}

/// `_n[offset]`, a constant index projection in the shape rustc
/// emits for a literal subscript.
fn cidx(n: u32, offset: u64, len: u64) -> mir::Place {
    mir::Place {
        local: mir::Local(n),
        projection: vec![mir::PlaceElem::ConstantIndex {
            offset,
            min_length: len,
            from_end: false,
        }],
    }
}

/// `_n[offset from the end]`, the `from_end` form of the same
/// projection.
fn cidx_end(n: u32, offset: u64, len: u64) -> mir::Place {
    mir::Place {
        local: mir::Local(n),
        projection: vec![mir::PlaceElem::ConstantIndex {
            offset,
            min_length: len,
            from_end: true,
        }],
    }
}

fn repeat_sum_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    // fn repeat_sum() -> i32 {
    //   _1 = Repeat(const 7, 4);
    //   _0 = _1[0] + _1[1] + _1[2] + _1[1 from the end];
    //   return
    // }
    let i32t = i(mir::IntTy::I32);
    let sum_from = |place: mir::Place| {
        assign(
            pl(0),
            mir::Rvalue::BinaryOp(
                mir::BinOp::Add,
                Box::new((cp(0), mir::Operand::Copy(place))),
            ),
        )
    };
    let f = spec(
        "repeat_sum",
        0,
        0,
        &[i32t, arr_i32(4)],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Repeat(i32c(7), 4)),
                assign(pl(0), mir::Rvalue::Use(mir::Operand::Copy(cidx(1, 0, 4)))),
                sum_from(cidx(1, 1, 4)),
                sum_from(cidx(1, 2, 4)),
                sum_from(cidx_end(1, 1, 4)),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    build(&[f])
}

#[test]
fn repeat_const_elements_sum() {
    let (jit, prog) = repeat_sum_program();
    check(&jit, &prog, "repeat_sum", &[], vec![], Ok(28));
}

fn repeat_mix_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    // fn repeat_mix(a: i32) -> i32 {
    //   _3 = a + a;
    //   _2 = Repeat(copy _3, 3);
    //   _2[1] = 100;
    //   _0 = _2[0] + _2[1] + _2[2];
    //   return
    // }
    let i32t = i(mir::IntTy::I32);
    let f = spec(
        "repeat_mix",
        0,
        1,
        &[i32t.clone(), i32t.clone(), arr_i32(3), i32t],
        vec![bb(
            vec![
                assign(
                    pl(3),
                    mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(1), cp(1)))),
                ),
                assign(pl(2), mir::Rvalue::Repeat(cp(3), 3)),
                assign(cidx(2, 1, 3), mir::Rvalue::Use(i32c(100))),
                assign(pl(0), mir::Rvalue::Use(mir::Operand::Copy(cidx(2, 0, 3)))),
                assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((cp(0), mir::Operand::Copy(cidx(2, 1, 3)))),
                    ),
                ),
                assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((cp(0), mir::Operand::Copy(cidx(2, 2, 3)))),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    build(&[f])
}

#[test]
fn repeat_place_element_and_element_write() {
    let (jit, prog) = repeat_mix_program();
    check(&jit, &prog, "repeat_mix", &[5], vec![int(5)], Ok(120));
    check(&jit, &prog, "repeat_mix", &[0], vec![int(0)], Ok(100));
}

fn repeat_pairs_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    // fn repeat_pairs() -> i32 {
    //   _2 = (3, 4);
    //   _1 = Repeat(copy _2, 2);
    //   _0 = _1[1].0 + _1[0].1;
    //   return
    // }
    let i32t = i(mir::IntTy::I32);
    let pair_ty = mir::Ty::Tuple(vec![i32t.clone(), i32t.clone()]);
    let arr_ty = mir::Ty::Array(Box::new(pair_ty.clone()), mir::ArrayLen::Known(2));
    let at = |offset: u64, field: u32| mir::Place {
        local: mir::Local(1),
        projection: vec![
            mir::PlaceElem::ConstantIndex {
                offset,
                min_length: 2,
                from_end: false,
            },
            mir::PlaceElem::Field(field, i32t.clone()),
        ],
    };
    let f = spec(
        "repeat_pairs",
        0,
        0,
        &[i32t.clone(), arr_ty, pair_ty],
        vec![bb(
            vec![
                assign(
                    pl(2),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![i32c(3), i32c(4)],
                    ),
                ),
                assign(pl(1), mir::Rvalue::Repeat(cp(2), 2)),
                assign(pl(0), mir::Rvalue::Use(mir::Operand::Copy(at(1, 0)))),
                assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((cp(0), mir::Operand::Copy(at(0, 1)))),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    build(&[f])
}

#[test]
fn repeat_tuple_elements() {
    let (jit, prog) = repeat_pairs_program();
    check(&jit, &prog, "repeat_pairs", &[], vec![], Ok(7));
}

#[test]
fn repeat_zero_elements() {
    // fn repeat_zero() -> i32 { _1 = Repeat(const 9, 0); _0 = 5; return }
    let i32t = i(mir::IntTy::I32);
    let f = spec(
        "repeat_zero",
        0,
        0,
        &[i32t, arr_i32(0)],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Repeat(i32c(9), 0)),
                assign(pl(0), mir::Rvalue::Use(i32c(5))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[f]);
    check(&jit, &prog, "repeat_zero", &[], vec![], Ok(5));
}

#[test]
fn repeat_beyond_frame_bound_is_unsupported() {
    let i32t = i(mir::IntTy::I32);
    let f = MirFunctionInput {
        name: "big_repeat".to_string(),
        def_id: Some(def(0)),
        body: mir::Body::new(
            locals(&[i32t, arr_i32(5000)]),
            0,
            vec![bb(
                vec![assign(pl(1), mir::Rvalue::Repeat(i32c(1), 5000))],
                mir::TerminatorKind::Return,
            )],
        ),
    };
    let err = JitProgram::compile(&MirProgramInput {
        functions: vec![f],
        names: HashMap::new(),
        adt_discriminants: Default::default(),
    })
    .expect_err("an array past the frame bound must be unsupported");
    match err {
        JitError::Unsupported(what) => {
            assert!(what.contains("scalar elements"), "{what}")
        }
        other => panic!("expected Unsupported, got {other}"),
    }
}

#[test]
fn dynamic_index_into_a_local_array() {
    let i32t = i(mir::IntTy::I32);
    // fn idx(i: usize) -> i32 { _2 = Repeat(3, 2); _2[1] = 7; return _2[_1] }
    let idx = spec(
        "idx",
        0,
        1,
        &[i32t.clone(), u(mir::UintTy::Usize), arr_i32(2)],
        vec![bb(
            vec![
                assign(pl(2), mir::Rvalue::Repeat(i32c(3), 2)),
                assign(cidx(2, 1, 2), mir::Rvalue::Use(i32c(7))),
                assign(
                    pl(0),
                    mir::Rvalue::Use(mir::Operand::Copy(pl(2).index(mir::Local(1)))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn set(i: usize, v: i32) -> i32 { _3 = Repeat(1, 2); _3[_1] = _2; return _3[_1] }
    let set = spec(
        "set",
        1,
        2,
        &[i32t.clone(), u(mir::UintTy::Usize), i32t.clone(), arr_i32(2)],
        vec![bb(
            vec![
                assign(pl(3), mir::Rvalue::Repeat(i32c(1), 2)),
                assign(pl(3).index(mir::Local(1)), mir::Rvalue::Use(cp(2))),
                assign(
                    pl(0),
                    mir::Rvalue::Use(mir::Operand::Copy(pl(3).index(mir::Local(1)))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[idx, set]);
    check(&jit, &prog, "idx", &[0], vec![uint(0)], Ok(3));
    check(&jit, &prog, "idx", &[1], vec![uint(1)], Ok(7));
    check(
        &jit,
        &prog,
        "idx",
        &[2],
        vec![uint(2)],
        Err("index out of bounds: the len is 2 but the index is 2"),
    );
    check(&jit, &prog, "set", &[1, 42], vec![uint(1), int(42)], Ok(42));
    check(
        &jit,
        &prog,
        "set",
        &[3, 42],
        vec![uint(3), int(42)],
        Err("index out of bounds: the len is 2 but the index is 3"),
    );
}

// ----- ADT discriminants: tag-only ADT values --------------------------------

/// The enum type used by the discriminant tests: `DefId(0, 42)`,
/// with recorded discriminants `[3, 7, -2]` where a table is given.
fn enum_ty() -> mir::Ty {
    mir::Ty::Adt(def(42), vec![])
}

fn discr_table() -> mir::AdtDiscriminants {
    let mut table = mir::AdtDiscriminants::new();
    table.insert(def(42), vec![3, 7, -2]);
    table
}

/// `_n = <fieldless variant of enum 42>` as an ADT aggregate.
fn adt_agg(n: u32, variant_index: u32) -> mir::StatementKind {
    assign(
        pl(n),
        mir::Rvalue::Aggregate(
            Box::new(mir::AggregateKind::Adt {
                def: def(42),
                variant_index,
                generic_args: vec![],
                user_ty: None,
                field_index: None,
            }),
            vec![],
        ),
    )
}

fn set_discr(n: u32, variant_index: u32) -> mir::StatementKind {
    mir::StatementKind::SetDiscriminant {
        place: Box::new(pl(n)),
        variant_index,
    }
}

fn set_discr_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    // fn set_c() -> i32 { _1 = E::A; SetDiscriminant(_1, 2); _0 = Discriminant(_1); return }
    // fn set_b() -> i32 { _1 = E::A; SetDiscriminant(_1, 1); _0 = Discriminant(_1); return }
    // The table maps the variants to 3, 7, -2: the answers are the
    // recorded values, not the variant indices (2 and 1).
    let i32t = i(mir::IntTy::I32);
    let mk = |name: &'static str, fn_def: u32, variant: u32| {
        spec(
            name,
            fn_def,
            0,
            &[i32t.clone(), enum_ty()],
            vec![bb(
                vec![
                    adt_agg(1, 0),
                    set_discr(1, variant),
                    assign(pl(0), mir::Rvalue::Discriminant(pl(1))),
                ],
                mir::TerminatorKind::Return,
            )],
        )
    };
    build_with_table(&[mk("set_c", 0, 2), mk("set_b", 1, 1)], discr_table())
}

#[test]
fn set_discriminant_then_read() {
    let (jit, prog) = set_discr_program();
    check(&jit, &prog, "set_c", &[], vec![], Ok(-2));
    check(&jit, &prog, "set_b", &[], vec![], Ok(7));
}

fn discr_switch_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    // fn pick_b() -> i32 {
    //   _1 = E::A; SetDiscriminant(_1, 1); _2 = Discriminant(_1);
    //   SwitchInt(_2) { 7 => bb1, otherwise => bb2 }
    // }
    // fn pick_a() -> i32 { same with SetDiscriminant(_1, 0) }
    // bb1: _0 = 111; return    bb2: _0 = 222; return
    let i32t = i(mir::IntTy::I32);
    let mk = |name: &'static str, fn_def: u32, variant: u32| {
        spec(
            name,
            fn_def,
            0,
            &[i32t.clone(), enum_ty(), i32t.clone()],
            vec![
                bb(
                    vec![
                        adt_agg(1, 0),
                        set_discr(1, variant),
                        assign(pl(2), mir::Rvalue::Discriminant(pl(1))),
                    ],
                    mir::TerminatorKind::SwitchInt {
                        discr: cp(2),
                        targets: mir::SwitchTargets::new(
                            vec![7],
                            vec![mir::BasicBlock(1)],
                            mir::BasicBlock(2),
                        ),
                    },
                ),
                bb(
                    vec![assign(pl(0), mir::Rvalue::Use(i32c(111)))],
                    mir::TerminatorKind::Return,
                ),
                bb(
                    vec![assign(pl(0), mir::Rvalue::Use(i32c(222)))],
                    mir::TerminatorKind::Return,
                ),
            ],
        )
    };
    build_with_table(&[mk("pick_b", 0, 1), mk("pick_a", 1, 0)], discr_table())
}

#[test]
fn set_discriminant_drives_switch() {
    let (jit, prog) = discr_switch_program();
    // Variant 1's discriminant is 7: the keyed branch.
    check(&jit, &prog, "pick_b", &[], vec![], Ok(111));
    // Variant 0's discriminant is 3: falls to otherwise.
    check(&jit, &prog, "pick_a", &[], vec![], Ok(222));
}

#[test]
fn discriminant_of_aggregate_and_constant() {
    // fn discr_agg() -> i32 { _1 = E::C; _0 = Discriminant(_1); return }
    // fn discr_const() -> i32 { _1 = const E::B; _0 = Discriminant(_1); return }
    let i32t = i(mir::IntTy::I32);
    let konst = mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Adt {
        variant: 1,
        fields: vec![],
    }));
    let agg = spec(
        "discr_agg",
        0,
        0,
        &[i32t.clone(), enum_ty()],
        vec![bb(
            vec![
                adt_agg(1, 2),
                assign(pl(0), mir::Rvalue::Discriminant(pl(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let con = spec(
        "discr_const",
        1,
        0,
        &[i32t, enum_ty()],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(konst)),
                assign(pl(0), mir::Rvalue::Discriminant(pl(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build_with_table(&[agg, con], discr_table());
    check(&jit, &prog, "discr_agg", &[], vec![], Ok(-2));
    check(&jit, &prog, "discr_const", &[], vec![], Ok(7));
}

#[test]
fn sequential_enum_needs_no_table() {
    // An enum with no table entry has the sequential discriminants,
    // so the variant index itself is the value on both engines.
    // fn seq() -> i32 { _1 = F::A; SetDiscriminant(_1, 2); _0 = Discriminant(_1); return }
    let i32t = i(mir::IntTy::I32);
    let f = spec(
        "seq",
        0,
        0,
        &[i32t, mir::Ty::Adt(def(77), vec![])],
        vec![bb(
            vec![
                assign(
                    pl(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Adt {
                            def: def(77),
                            variant_index: 0,
                            generic_args: vec![],
                            user_ty: None,
                            field_index: None,
                        }),
                        vec![],
                    ),
                ),
                set_discr(1, 2),
                assign(pl(0), mir::Rvalue::Discriminant(pl(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[f]);
    check(&jit, &prog, "seq", &[], vec![], Ok(2));
}

// ----- References and raw pointers: the pointer is the slot address ----------

fn refty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Not)
}
fn refmty(inner: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(mir::Region::Erased, Box::new(inner), mir::Mutability::Mut)
}
fn rawty(inner: mir::Ty, mutable: bool) -> mir::Ty {
    mir::Ty::RawPtr(
        Box::new(inner),
        if mutable {
            mir::Mutability::Mut
        } else {
            mir::Mutability::Not
        },
    )
}
fn dpl(n: u32) -> mir::Place {
    pl(n).deref()
}
fn cpd(n: u32) -> mir::Operand {
    mir::Operand::Copy(dpl(n))
}
fn shared(place: mir::Place) -> mir::Rvalue {
    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::Shared, place)
}
fn mutref(place: mir::Place) -> mir::Rvalue {
    mir::Rvalue::Ref(mir::Region::Erased, mir::BorrowKind::mutable(), place)
}

/// `(*_n)[offset]`, a constant index through a dereference.
fn dcidx(n: u32, offset: u64, len: u64) -> mir::Place {
    mir::Place {
        local: mir::Local(n),
        projection: vec![
            mir::PlaceElem::Deref,
            mir::PlaceElem::ConstantIndex {
                offset,
                min_length: len,
                from_end: false,
            },
        ],
    }
}

#[test]
fn raw_ptr_creation_reads_and_writes_through() {
    let i32t = i(mir::IntTy::I32);
    // fn read_raw() -> i32 { _1 = 41; _2 = &raw const _1; return *_2 }
    let read_raw = spec(
        "read_raw",
        0,
        0,
        &[i32t.clone(), i32t.clone(), rawty(i32t.clone(), false)],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(41))),
                assign(pl(2), mir::Rvalue::RawPtr(mir::Mutability::Not, pl(1))),
                assign(pl(0), mir::Rvalue::Use(cpd(2))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn write_raw() -> i32 { _1 = 1; _2 = &raw mut _1; *_2 = 99; return _1 }
    let write_raw = spec(
        "write_raw",
        1,
        0,
        &[i32t.clone(), i32t.clone(), rawty(i32t.clone(), true)],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(1))),
                assign(pl(2), mir::Rvalue::RawPtr(mir::Mutability::Mut, pl(1))),
                assign(dpl(2), mir::Rvalue::Use(i32c(99))),
                assign(pl(0), mir::Rvalue::Use(cp(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[read_raw, write_raw]);
    check(&jit, &prog, "read_raw", &[], vec![], Ok(41));
    check(&jit, &prog, "write_raw", &[], vec![], Ok(99));
}

#[test]
fn mut_ref_read_modify_write_reaches_the_place() {
    let i32t = i(mir::IntTy::I32);
    // fn bump() -> i32 { _1 = 10; _2 = &mut _1; *_2 = *_2 + 5; return _1 }
    let bump = spec(
        "bump",
        0,
        0,
        &[i32t.clone(), i32t.clone(), refmty(i32t.clone())],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(10))),
                assign(pl(2), mutref(pl(1))),
                assign(
                    dpl(2),
                    mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cpd(2), i32c(5)))),
                ),
                assign(pl(0), mir::Rvalue::Use(cp(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[bump]);
    check(&jit, &prog, "bump", &[], vec![], Ok(15));
}

#[test]
fn ref_reassignment_reads_the_new_pointee() {
    let i32t = i(mir::IntTy::I32);
    // fn pick() -> i32 { _1 = 3; _2 = 4; _3 = &_1; _3 = &_2; return *_3 }
    let pick = spec(
        "pick",
        0,
        0,
        &[i32t.clone(), i32t.clone(), i32t.clone(), refty(i32t.clone())],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(3))),
                assign(pl(2), mir::Rvalue::Use(i32c(4))),
                assign(pl(3), shared(pl(1))),
                assign(pl(3), shared(pl(2))),
                assign(pl(0), mir::Rvalue::Use(cpd(3))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[pick]);
    check(&jit, &prog, "pick", &[], vec![], Ok(4));
}

#[test]
fn ref_of_a_dereference_is_the_pointer_itself() {
    let i32t = i(mir::IntTy::I32);
    // fn f() -> i32 { _1 = 5; _2 = &mut _1; _3 = &mut *_2; *_3 = 6; return _1 }
    let f = spec(
        "ref_of_deref",
        0,
        0,
        &[
            i32t.clone(),
            i32t.clone(),
            refmty(i32t.clone()),
            refmty(i32t.clone()),
        ],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(5))),
                assign(pl(2), mutref(pl(1))),
                assign(pl(3), mutref(dpl(2))),
                assign(dpl(3), mir::Rvalue::Use(i32c(6))),
                assign(pl(0), mir::Rvalue::Use(cp(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[f]);
    check(&jit, &prog, "ref_of_deref", &[], vec![], Ok(6));
}

#[test]
fn ref_arguments_read_and_write_across_frames() {
    let i32t = i(mir::IntTy::I32);
    let unit = mir::Ty::Tuple(vec![]);
    // fn incr(_1: &mut i32) { *_1 = *_1 + 1; return }
    let incr = spec(
        "incr",
        0,
        1,
        &[unit.clone(), refmty(i32t.clone())],
        vec![bb(
            vec![assign(
                dpl(1),
                mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cpd(1), i32c(1)))),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn get(_1: &i32) -> i32 { return *_1 }
    let get = spec(
        "get",
        1,
        1,
        &[i32t.clone(), refty(i32t.clone())],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Use(cpd(1)))],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main_write() -> i32 { _1 = 41; _3 = &mut _1; incr(_3); return _1 }
    let main_write = spec(
        "main_write",
        2,
        0,
        &[i32t.clone(), i32t.clone(), unit.clone(), refmty(i32t.clone())],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Use(i32c(41))),
                    assign(pl(3), mutref(pl(1))),
                ],
                call_term(fndef(def(0)), vec![cp(3)], 2, 1),
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(1)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    // fn main_read() -> i32 { _1 = 7; _2 = &_1; return get(_2) }
    let main_read = spec(
        "main_read",
        3,
        0,
        &[i32t.clone(), i32t.clone(), refty(i32t.clone())],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Use(i32c(7))),
                    assign(pl(2), shared(pl(1))),
                ],
                call_term(fndef(def(1)), vec![cp(2)], 0, 1),
            ),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    let (jit, prog) = build(&[incr, get, main_write, main_read]);
    check(&jit, &prog, "main_write", &[], vec![], Ok(42));
    check(&jit, &prog, "main_read", &[], vec![], Ok(7));
}

#[test]
fn ref_to_a_tuple_field_writes_the_field() {
    let i32t = i(mir::IntTy::I32);
    let pair = mir::Ty::Tuple(vec![i32t.clone(), i32t.clone()]);
    // fn f() -> i32 { _1 = (10, 20); _2 = &mut _1.1; *_2 = 25; return _1.1 }
    let f = spec(
        "field_ref",
        0,
        0,
        &[i32t.clone(), pair, refmty(i32t.clone())],
        vec![bb(
            vec![
                assign(
                    pl(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![i32c(10), i32c(20)],
                    ),
                ),
                assign(pl(2), mutref(fld(1, 1, i32t.clone()))),
                assign(dpl(2), mir::Rvalue::Use(i32c(25))),
                assign(pl(0), mir::Rvalue::Use(cpf(1, 1, i32t.clone()))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[f]);
    check(&jit, &prog, "field_ref", &[], vec![], Ok(25));
}

#[test]
fn pointer_casts_keep_the_address() {
    let i32t = i(mir::IntTy::I32);
    // fn via_ptr_to_ptr() -> i32 {
    //   _1 = 33; _2 = &raw mut _1; _3 = _2 as *const i32; return *_3 }
    let f1 = spec(
        "via_ptr_to_ptr",
        0,
        0,
        &[
            i32t.clone(),
            i32t.clone(),
            rawty(i32t.clone(), true),
            rawty(i32t.clone(), false),
        ],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(33))),
                assign(pl(2), mir::Rvalue::RawPtr(mir::Mutability::Mut, pl(1))),
                assign(
                    pl(3),
                    mir::Rvalue::Cast(mir::CastKind::PtrToPtr, cp(2), rawty(i32t.clone(), false)),
                ),
                assign(pl(0), mir::Rvalue::Use(cpd(3))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn via_coercion() -> i32 {
    //   _1 = 34; _2 = &mut _1; _3 = _2 as *const i32; return *_3 }
    let f2 = spec(
        "via_coercion",
        1,
        0,
        &[
            i32t.clone(),
            i32t.clone(),
            refmty(i32t.clone()),
            rawty(i32t.clone(), false),
        ],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(34))),
                assign(pl(2), mutref(pl(1))),
                assign(
                    pl(3),
                    mir::Rvalue::Cast(
                        mir::CastKind::PointerCoercion(
                            mir::PointerCoercion::MutToConstPointer,
                        ),
                        cp(2),
                        rawty(i32t.clone(), false),
                    ),
                ),
                assign(pl(0), mir::Rvalue::Use(cpd(3))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[f1, f2]);
    check(&jit, &prog, "via_ptr_to_ptr", &[], vec![], Ok(33));
    check(&jit, &prog, "via_coercion", &[], vec![], Ok(34));
}

// ----- Shapes that stay unsupported -------------------------------------------

#[test]
fn copy_for_deref_reads_the_pointee() {
    let i32t = i(mir::IntTy::I32);
    // fn f() -> i32 { _1 = 21; _2 = &_1; _0 = CopyForDeref(_2); return }
    let f = spec(
        "cfd",
        0,
        0,
        &[i32t.clone(), i32t.clone(), refty(i32t.clone())],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(21))),
                assign(pl(2), shared(pl(1))),
                assign(pl(0), mir::Rvalue::CopyForDeref(pl(2))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[f]);
    check(&jit, &prog, "cfd", &[], vec![], Ok(21));
}

#[test]
fn discriminant_through_a_dereference() {
    let i32t = i(mir::IntTy::I32);
    // fn read_d() -> i64 { _1 = E::B; _2 = &_1; _0 = Discriminant(*_2); return }
    // fn write_d() -> i64 { _1 = E::A; _2 = &mut _1; SetDiscriminant(*_2, 2); _0 = Discriminant(_1); return }
    let i64t = i(mir::IntTy::I64);
    let read_d = spec(
        "read_d",
        0,
        0,
        &[i64t.clone(), enum_ty(), refty(enum_ty())],
        vec![bb(
            vec![
                adt_agg(1, 1),
                assign(pl(2), shared(pl(1))),
                assign(pl(0), mir::Rvalue::Discriminant(dpl(2))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let write_d = spec(
        "write_d",
        1,
        0,
        &[i64t.clone(), enum_ty(), refmty(enum_ty())],
        vec![bb(
            vec![
                adt_agg(1, 0),
                assign(pl(2), mutref(pl(1))),
                mir::StatementKind::SetDiscriminant {
                    place: Box::new(dpl(2)),
                    variant_index: 2,
                },
                assign(pl(0), mir::Rvalue::Discriminant(pl(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let _ = i32t;
    let (jit, prog) = build_with_table(&[read_d, write_d], discr_table());
    check(&jit, &prog, "read_d", &[], vec![], Ok(7));
    check(&jit, &prog, "write_d", &[], vec![], Ok(-2));
}

fn jit_compile(specs: &[FnSpec]) -> Result<JitProgram, JitError> {
    let names: HashMap<mir::DefId, String> = specs
        .iter()
        .map(|s| (def(s.def), s.name.to_string()))
        .collect();
    JitProgram::compile(&MirProgramInput {
        functions: specs
            .iter()
            .map(|s| MirFunctionInput {
                name: s.name.to_string(),
                def_id: Some(def(s.def)),
                body: s.body.clone(),
            })
            .collect(),
        names,
        adt_discriminants: Default::default(),
    })
}

fn slicety(elem: mir::Ty) -> mir::Ty {
    mir::Ty::Slice(Box::new(elem))
}

fn fatrefty(elem: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(slicety(elem)),
        mir::Mutability::Not,
    )
}

fn fatrefmty(elem: mir::Ty) -> mir::Ty {
    mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(slicety(elem)),
        mir::Mutability::Mut,
    )
}

fn fatval(elems: &[i128], len: u128) -> Value {
    Value::FatRef {
        base: Box::new(Value::Tuple(elems.iter().map(|v| int(*v)).collect())),
        len,
    }
}

#[test]
fn fat_reference_argument_index_len_and_metadata() {
    let i32t = i(mir::IntTy::I32);
    let usizet = u(mir::UintTy::Usize);
    // fn get(s: &[i32], i: usize) -> i32 { return (*s)[_2] }
    let get = spec(
        "get",
        0,
        2,
        &[i32t.clone(), fatrefty(i32t.clone()), usizet.clone()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::Use(mir::Operand::Copy(dpl(1).index(mir::Local(2)))),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn slen(s: &[i32]) -> usize { return Len(*_1) }
    let slen = spec(
        "slen",
        1,
        1,
        &[usizet.clone(), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Len(dpl(1)))],
            mir::TerminatorKind::Return,
        )],
    );
    // fn meta(s: &[i32]) -> usize { return PtrMetadata(_1) }
    let meta = spec(
        "meta",
        2,
        1,
        &[usizet.clone(), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::UnaryOp(mir::UnOp::PtrMetadata, cp(1)),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[get, slen, meta]);
    // The JIT side is called with a real backing store laid out
    // the way a caller frame's slots lie: element 0 at the base
    // address, later elements at descending addresses (the slot
    // model's layout). The interpreter side gets the same slice
    // as a fat value.
    let store: [i64; 3] = [30, 20, 10];
    let addr = store.as_ptr().wrapping_add(2) as i64;
    check(
        &jit,
        &prog,
        "get",
        &[addr, 3, 1],
        vec![fatval(&[10, 20, 30], 3), uint(1)],
        Ok(20),
    );
    check(
        &jit,
        &prog,
        "get",
        &[addr, 3, 5],
        vec![fatval(&[10, 20, 30], 3), uint(5)],
        Err("index out of bounds: the len is 3 but the index is 5"),
    );
    check(&jit, &prog, "slen", &[addr, 3], vec![fatval(&[10, 20, 30], 3)], Ok(3));
    check(&jit, &prog, "meta", &[addr, 3], vec![fatval(&[10, 20, 30], 3)], Ok(3));
}

#[test]
fn unsize_coercion_builds_a_slice_that_calls_use() {
    let i32t = i(mir::IntTy::I32);
    // fn sum2(s: &[i32]) -> i32 { return (*s)[0] + (*s)[1] }
    let sum2 = spec(
        "sum2",
        0,
        1,
        &[i32t.clone(), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::BinaryOp(
                    mir::BinOp::Add,
                    Box::new((
                        mir::Operand::Copy(dcidx(1, 0, 3)),
                        mir::Operand::Copy(dcidx(1, 1, 3)),
                    )),
                ),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main() -> i32 {
    //   _1 = [5, 6, 7]; _4 = &_1; _2 = _4 as &[i32]; return sum2(_2)
    // }
    let main = spec(
        "main_unsize",
        1,
        0,
        &[
            i32t.clone(),
            arr_i32(3),
            fatrefty(i32t.clone()),
            i32t.clone(),
            refty(arr_i32(3)),
        ],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Repeat(i32c(5), 3)),
                    assign(cidx(1, 1, 3), mir::Rvalue::Use(i32c(6))),
                    assign(cidx(1, 2, 3), mir::Rvalue::Use(i32c(7))),
                    assign(pl(4), shared(pl(1))),
                    assign(
                        pl(2),
                        mir::Rvalue::Cast(
                            mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
                            cp(4),
                            fatrefty(i32t.clone()),
                        ),
                    ),
                ],
                call_term(fndef(def(0)), vec![cp(2)], 3, 1),
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(3)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[sum2, main]);
    check(&jit, &prog, "main_unsize", &[], vec![], Ok(11));
}

#[test]
fn slice_write_and_reborrow_in_program() {
    let i32t = i(mir::IntTy::I32);
    let usizet = u(mir::UintTy::Usize);
    let unit = mir::Ty::Tuple(vec![]);
    // fn set(s: &mut [i32], i: usize, v: i32) { (*s)[i] = v }
    let set = spec(
        "set",
        0,
        3,
        &[unit.clone(), fatrefmty(i32t.clone()), usizet, i32t.clone()],
        vec![bb(
            vec![assign(
                dpl(1).index(mir::Local(2)),
                mir::Rvalue::Use(cp(3)),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main() -> i32 {
    //   _1 = [1, 2, 3]; _5 = &mut _1; _2 = _5 as &mut [i32];
    //   set(_2, 1, 42); return _1[1]
    // }
    let main = spec(
        "main_set",
        1,
        0,
        &[
            i32t.clone(),
            arr_i32(3),
            fatrefmty(i32t.clone()),
            unit.clone(),
            i32t.clone(),
            refmty(arr_i32(3)),
        ],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Repeat(i32c(1), 3)),
                    assign(cidx(1, 1, 3), mir::Rvalue::Use(i32c(2))),
                    assign(cidx(1, 2, 3), mir::Rvalue::Use(i32c(3))),
                    assign(pl(5), mutref(pl(1))),
                    assign(
                        pl(2),
                        mir::Rvalue::Cast(
                            mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
                            cp(5),
                            fatrefmty(i32t.clone()),
                        ),
                    ),
                ],
                call_term(fndef(def(0)), vec![cp(2), u64c(1), i32c(42)], 3, 1),
            ),
            bb(
                vec![assign(
                    pl(0),
                    mir::Rvalue::Use(mir::Operand::Copy(cidx(1, 1, 3))),
                )],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    // fn first_of(s: &[i32]) -> i32 { _2 = &*_1; return (*_2)[0] }
    let first_of = spec(
        "first_of",
        2,
        1,
        &[i32t.clone(), fatrefty(i32t.clone()), fatrefty(i32t.clone())],
        vec![bb(
            vec![
                assign(pl(2), shared(dpl(1))),
                assign(
                    pl(0),
                    mir::Rvalue::Use(mir::Operand::Copy(dcidx(2, 0, 3))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main_first() -> i32 { _1 = [8, 9, 10]; _4 = &_1; _2 = _4 as &[i32]; return first_of(_2) }
    let main_first = spec(
        "main_first",
        3,
        0,
        &[
            i32t.clone(),
            arr_i32(3),
            fatrefty(i32t.clone()),
            i32t.clone(),
            refty(arr_i32(3)),
        ],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Repeat(i32c(8), 3)),
                    assign(pl(4), shared(pl(1))),
                    assign(
                        pl(2),
                        mir::Rvalue::Cast(
                            mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
                            cp(4),
                            fatrefty(i32t.clone()),
                        ),
                    ),
                ],
                call_term(fndef(def(2)), vec![cp(2)], 3, 1),
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(3)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[set, main, first_of, main_first]);
    check(&jit, &prog, "main_set", &[], vec![], Ok(42));
    check(&jit, &prog, "main_first", &[], vec![], Ok(8));
}

#[test]
fn str_pair_metadata() {
    let usizet = u(mir::UintTy::Usize);
    let strref = mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(mir::Ty::Str),
        mir::Mutability::Not,
    );
    // fn slen(s: &str) -> usize { return PtrMetadata(_1) }
    let slen = spec(
        "slen_str",
        0,
        1,
        &[usizet, strref],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::UnaryOp(mir::UnOp::PtrMetadata, cp(1)),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[slen]);
    let fat = Value::FatRef {
        base: Box::new(Value::Str("hello".to_string())),
        len: 5,
    };
    check(&jit, &prog, "slen_str", &[0, 5], vec![fat], Ok(5));
}

#[test]
fn trait_object_reference_is_unsupported() {
    let i32t = i(mir::IntTy::I32);
    let dynref = mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(mir::Ty::Dynamic(
            vec!["core::marker::Trait".to_string()],
            Box::new(mir::Ty::Tuple(vec![])),
            mir::Region::Erased,
        )),
        mir::Mutability::Not,
    );
    let f = spec(
        "dynref",
        0,
        1,
        &[i32t.clone(), dynref],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Use(i32c(0)))],
            mir::TerminatorKind::Return,
        )],
    );
    match jit_compile(&[f]) {
        Err(JitError::Unsupported(msg)) => assert!(msg.contains("fat"), "{msg}"),
        Err(other) => panic!("expected Unsupported, got {other}"),
        Ok(_) => panic!("expected Unsupported, but the program compiled"),
    }
}

#[test]
fn projections_through_a_dereference() {
    let i32t = i(mir::IntTy::I32);
    let pair = mir::Ty::Tuple(vec![i32t.clone(), i32t.clone()]);
    // fn read_field() -> i32 { _1 = (10, 20); _2 = &_1; return (*_2).1 }
    let read_field = spec(
        "read_field",
        0,
        0,
        &[i32t.clone(), pair.clone(), refty(pair.clone())],
        vec![bb(
            vec![
                assign(
                    pl(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![i32c(10), i32c(20)],
                    ),
                ),
                assign(pl(2), shared(pl(1))),
                assign(
                    pl(0),
                    mir::Rvalue::Use(mir::Operand::Copy(dpl(2).field(1, i32t.clone()))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn write_field() -> i32 { _1 = (10, 20); _2 = &mut _1; (*_2).0 = 99; return _1.0 }
    let write_field = spec(
        "write_field",
        1,
        0,
        &[i32t.clone(), pair.clone(), refmty(pair.clone())],
        vec![bb(
            vec![
                assign(
                    pl(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![i32c(10), i32c(20)],
                    ),
                ),
                assign(pl(2), mutref(pl(1))),
                assign(dpl(2).field(0, i32t.clone()), mir::Rvalue::Use(i32c(99))),
                assign(pl(0), mir::Rvalue::Use(cpf(1, 0, i32t.clone()))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn read_elem() -> i32 { _1 = [1, 2, 42]; _2 = &_1; return (*_2)[2] }
    let read_elem = spec(
        "read_elem",
        2,
        0,
        &[i32t.clone(), arr_i32(3), refty(arr_i32(3))],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Repeat(i32c(1), 3)),
                assign(cidx(1, 2, 3), mir::Rvalue::Use(i32c(42))),
                assign(pl(2), shared(pl(1))),
                assign(
                    pl(0),
                    mir::Rvalue::Use(mir::Operand::Copy(dcidx(2, 2, 3))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn ref_field() -> i32 { _1 = (1, 2); _2 = &mut _1; _3 = &mut (*_2).1; *_3 = 5; return _1.1 }
    let ref_field = spec(
        "ref_field",
        3,
        0,
        &[
            i32t.clone(),
            pair.clone(),
            refmty(pair),
            refmty(i32t.clone()),
        ],
        vec![bb(
            vec![
                assign(
                    pl(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![i32c(1), i32c(2)],
                    ),
                ),
                assign(pl(2), mutref(pl(1))),
                assign(pl(3), mutref(dpl(2).field(1, i32t.clone()))),
                assign(dpl(3), mir::Rvalue::Use(i32c(5))),
                assign(pl(0), mir::Rvalue::Use(cpf(1, 1, i32t.clone()))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[read_field, write_field, read_elem, ref_field]);
    check(&jit, &prog, "read_field", &[], vec![], Ok(20));
    check(&jit, &prog, "write_field", &[], vec![], Ok(99));
    check(&jit, &prog, "read_elem", &[], vec![], Ok(42));
    check(&jit, &prog, "ref_field", &[], vec![], Ok(5));
}

#[test]
fn double_dereference_reads_and_writes() {
    let i32t = i(mir::IntTy::I32);
    // fn read2() -> i32 { _1 = 9; _2 = &_1; _3 = &_2; return **_3 }
    let read2 = spec(
        "read2",
        0,
        0,
        &[
            i32t.clone(),
            i32t.clone(),
            refty(i32t.clone()),
            refty(refty(i32t.clone())),
        ],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(9))),
                assign(pl(2), shared(pl(1))),
                assign(pl(3), shared(pl(2))),
                assign(pl(0), mir::Rvalue::Use(mir::Operand::Copy(dpl(3).deref()))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn write2() -> i32 { _1 = 9; _2 = &mut _1; _3 = &mut _2; **_3 = 4; return _1 }
    let write2 = spec(
        "write2",
        1,
        0,
        &[
            i32t.clone(),
            i32t.clone(),
            refmty(i32t.clone()),
            refmty(refmty(i32t.clone())),
        ],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(9))),
                assign(pl(2), mutref(pl(1))),
                assign(pl(3), mutref(pl(2))),
                assign(dpl(3).deref(), mir::Rvalue::Use(i32c(4))),
                assign(pl(0), mir::Rvalue::Use(cp(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn read2_arg(pp: &&i32) -> i32 { return **pp } — the double
    // dereference of a value that arrived as an argument.
    let read2_arg = spec(
        "read2_arg",
        2,
        1,
        &[i32t.clone(), refty(refty(i32t.clone()))],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::Use(mir::Operand::Copy(dpl(1).deref())),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main2() -> i32 { _1 = 9; _2 = &_1; _3 = &_2; return read2_arg(_3) }
    let main2 = spec(
        "main2",
        3,
        0,
        &[
            i32t.clone(),
            i32t.clone(),
            refty(i32t.clone()),
            refty(refty(i32t.clone())),
        ],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Use(i32c(9))),
                    assign(pl(2), shared(pl(1))),
                    assign(pl(3), shared(pl(2))),
                ],
                call_term(fndef(def(2)), vec![cp(3)], 0, 1),
            ),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    let (jit, prog) = build(&[read2, write2, read2_arg, main2]);
    check(&jit, &prog, "read2", &[], vec![], Ok(9));
    check(&jit, &prog, "write2", &[], vec![], Ok(4));
    check(&jit, &prog, "main2", &[], vec![], Ok(9));
}

// ----- two-word return values ------------------------------------------------
//
// A function's result of two words (a fat pointer, a two-scalar
// pair) travels in RAX/RDX. The pair helpers below are `check`
// for word pairs: the interpreter's result is the corresponding
// two-field tuple value.

fn jit_call_pair(jit: &JitProgram, name: &str, args: &[i64]) -> Result<(i64, i64), String> {
    jit.call_pair(name, args).map_err(|e| match e {
        JitCallError::Panic(rec) => rec.message,
        other => other.to_string(),
    })
}

fn value_word(v: &Value) -> i64 {
    match v {
        Value::Int(x) => *x as i64,
        Value::Uint(x) => *x as i64,
        Value::Bool(b) => *b as i64,
        Value::Unit => 0,
        other => panic!("unexpected interp word {other:?}"),
    }
}

fn interp_call_pair(
    prog: &sparagmos_interp::MirProgram,
    name: &str,
    args: Vec<Value>,
) -> Result<(i64, i64), String> {
    match sparagmos_interp::interpret_mir(prog, name, args) {
        Ok(Value::Tuple(fields)) if fields.len() == 2 => {
            Ok((value_word(&fields[0]), value_word(&fields[1])))
        }
        Ok(other) => panic!("unexpected interp result {other:?}"),
        Err(MirRunError::Interp(InterpError::Panic { message, .. })) => Err(message),
        Err(other) => panic!("unexpected interp error: {other}"),
    }
}

/// Both engines must produce `expected` (a word pair, or a guest
/// panic with exactly this message — in which case neither engine
/// produces any words).
fn check_pair(
    jit: &JitProgram,
    prog: &sparagmos_interp::MirProgram,
    name: &str,
    jit_args: &[i64],
    interp_args: Vec<Value>,
    expected: Result<(i64, i64), &str>,
) {
    let expected = expected.map_err(|s| s.to_string());
    assert_eq!(
        jit_call_pair(jit, name, jit_args),
        expected,
        "JIT result for {name}"
    );
    assert_eq!(
        interp_call_pair(prog, name, interp_args),
        expected,
        "interp result for {name}"
    );
}

/// fn make_pair(x: i32) -> (i32, i32) { (x, x + 1) }
fn make_pair_spec() -> FnSpec {
    let i32t = i(mir::IntTy::I32);
    let pair = mir::Ty::Tuple(vec![i32t.clone(), i32t.clone()]);
    spec(
        "make_pair",
        0,
        1,
        &[pair, i32t.clone(), i32t.clone()],
        vec![bb(
            vec![
                assign(
                    pl(2),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((cp(1), i32c(1))),
                    ),
                ),
                assign(
                    pl(0),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![cp(1), cp(2)],
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    )
}

#[test]
fn pair_return_words_and_consumer() {
    let i32t = i(mir::IntTy::I32);
    let pair = mir::Ty::Tuple(vec![i32t.clone(), i32t.clone()]);
    // fn main() -> i32 { let p = make_pair(10); p.0 * 100 + p.1 }
    let main = spec(
        "main_pair",
        1,
        0,
        &[i32t.clone(), pair.clone(), i32t.clone()],
        vec![
            bb(vec![], call_term(fndef(def(0)), vec![i32c(10)], 1, 1)),
            bb(
                vec![
                    assign(
                        pl(2),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Mul,
                            Box::new((cpf(1, 0, i32t.clone()), i32c(100))),
                        ),
                    ),
                    assign(
                        pl(0),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((cp(2), cpf(1, 1, i32t.clone()))),
                        ),
                    ),
                ],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[make_pair_spec(), main]);
    check_pair(&jit, &prog, "make_pair", &[41], vec![int(41)], Ok((41, 42)));
    check_pair(&jit, &prog, "make_pair", &[-1], vec![int(-1)], Ok((-1, 0)));
    check(&jit, &prog, "main_pair", &[], vec![], Ok(1011));
}

#[test]
fn fat_return_built_by_unsize_in_callee() {
    let i32t = i(mir::IntTy::I32);
    let usizet = u(mir::UintTy::Usize);
    // fn wrap(a: &[i32; 3]) -> &[i32] { a as &[i32] }
    let wrap = spec(
        "wrap",
        0,
        1,
        &[fatrefty(i32t.clone()), refty(arr_i32(3))],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::Cast(
                    mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
                    cp(1),
                    fatrefty(i32t.clone()),
                ),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main() -> i32 {
    //   _1 = [5, 6, 7]; _2 = wrap(&_1); (*_2)[0] + (*_2)[2]
    // }
    let main = spec(
        "main_wrap",
        1,
        0,
        &[
            i32t.clone(),
            arr_i32(3),
            fatrefty(i32t.clone()),
            refty(arr_i32(3)),
        ],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Repeat(i32c(5), 3)),
                    assign(cidx(1, 1, 3), mir::Rvalue::Use(i32c(6))),
                    assign(cidx(1, 2, 3), mir::Rvalue::Use(i32c(7))),
                    assign(pl(3), shared(pl(1))),
                ],
                call_term(fndef(def(0)), vec![cp(3)], 2, 1),
            ),
            bb(
                vec![assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((
                            mir::Operand::Copy(dcidx(2, 0, 3)),
                            mir::Operand::Copy(dcidx(2, 2, 3)),
                        )),
                    ),
                )],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    // fn main_len() -> usize { _1 = [5, 6, 7]; Len(*wrap(&_1)) }
    let main_len = spec(
        "main_wrap_len",
        2,
        0,
        &[
            usizet.clone(),
            arr_i32(3),
            fatrefty(i32t.clone()),
            refty(arr_i32(3)),
        ],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Repeat(i32c(5), 3)),
                    assign(pl(3), shared(pl(1))),
                ],
                call_term(fndef(def(0)), vec![cp(3)], 2, 1),
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Len(dpl(2)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[wrap, main, main_len]);
    check(&jit, &prog, "main_wrap", &[], vec![], Ok(5 + 7));
    check(&jit, &prog, "main_wrap_len", &[], vec![], Ok(3));
}

#[test]
fn fat_return_identity_pair_words() {
    let i32t = i(mir::IntTy::I32);
    // fn id(s: &[i32]) -> &[i32] { s }
    let id = spec(
        "id_slice",
        0,
        1,
        &[fatrefty(i32t.clone()), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Use(cp(1)))],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[id]);
    // The JIT address word is exactly the address passed in (the
    // pair travels through unchanged), and the length word is the
    // length; the interpreter's result is the same fat value, whose
    // comparable word is its length.
    let store: [i64; 3] = [30, 20, 10];
    let addr = store.as_ptr().wrapping_add(2) as i64;
    assert_eq!(jit_call_pair(&jit, "id_slice", &[addr, 3]), Ok((addr, 3)));
    match sparagmos_interp::interpret_mir(&prog, "id_slice", vec![fatval(&[10, 20, 30], 3)]) {
        Ok(Value::FatRef { len, .. }) => assert_eq!(len, 3),
        other => panic!("unexpected interp result {other:?}"),
    }
}

#[test]
fn str_return_through_nested_calls() {
    let usizet = u(mir::UintTy::Usize);
    let strref = mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(mir::Ty::Str),
        mir::Mutability::Not,
    );
    // fn id_str(s: &str) -> &str { s }
    let id_str = spec(
        "id_str",
        0,
        1,
        &[strref.clone(), strref.clone()],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Use(cp(1)))],
            mir::TerminatorKind::Return,
        )],
    );
    // fn use_str(s: &str) -> usize { PtrMetadata(s) }
    let use_str = spec(
        "use_str",
        1,
        1,
        &[usizet.clone(), strref.clone()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::UnaryOp(mir::UnOp::PtrMetadata, cp(1)),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main(s: &str) -> usize { use_str(id_str(s)) }
    let main = spec(
        "main_str",
        2,
        1,
        &[usizet.clone(), strref.clone(), strref.clone()],
        vec![
            bb(vec![], call_term(fndef(def(0)), vec![cp(1)], 2, 1)),
            bb(vec![], call_term(fndef(def(1)), vec![cp(2)], 0, 2)),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    let (jit, prog) = build(&[id_str, use_str, main]);
    let fat = Value::FatRef {
        base: Box::new(Value::Str("hello".to_string())),
        len: 5,
    };
    check(&jit, &prog, "main_str", &[0, 5], vec![fat], Ok(5));
}

#[test]
fn nested_calls_returning_fat_refs() {
    let i32t = i(mir::IntTy::I32);
    // fn inner(a: &[i32; 3]) -> &[i32] { a as &[i32] }
    let inner = spec(
        "inner",
        0,
        1,
        &[fatrefty(i32t.clone()), refty(arr_i32(3))],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::Cast(
                    mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
                    cp(1),
                    fatrefty(i32t.clone()),
                ),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn middle(a: &[i32; 3]) -> &[i32] { inner(a) }
    let middle = spec(
        "middle",
        1,
        1,
        &[
            fatrefty(i32t.clone()),
            refty(arr_i32(3)),
            fatrefty(i32t.clone()),
        ],
        vec![
            bb(vec![], call_term(fndef(def(0)), vec![cp(1)], 2, 1)),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(mir::Operand::Move(pl(2))))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    // fn main() -> i32 { _1 = [5, 6, 7]; (*middle(&_1))[1] }
    let main = spec(
        "main_nested",
        2,
        0,
        &[
            i32t.clone(),
            arr_i32(3),
            fatrefty(i32t.clone()),
            refty(arr_i32(3)),
        ],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Repeat(i32c(5), 3)),
                    assign(cidx(1, 1, 3), mir::Rvalue::Use(i32c(6))),
                    assign(cidx(1, 2, 3), mir::Rvalue::Use(i32c(7))),
                    assign(pl(3), shared(pl(1))),
                ],
                call_term(fndef(def(1)), vec![cp(3)], 2, 1),
            ),
            bb(
                vec![assign(
                    pl(0),
                    mir::Rvalue::Use(mir::Operand::Copy(dcidx(2, 1, 3))),
                )],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[inner, middle, main]);
    check(&jit, &prog, "main_nested", &[], vec![], Ok(6));
}

#[test]
fn pair_return_panics_propagate_without_a_result() {
    let i32t = i(mir::IntTy::I32);
    let pair = mir::Ty::Tuple(vec![i32t.clone(), i32t.clone()]);
    let checked = mir::Ty::Tuple(vec![i32t.clone(), mir::Ty::Bool]);
    // fn boom_pair(x: i32) -> (i32, i32) { (x + x, 7) }  (checked add)
    let boom_pair = spec(
        "boom_pair",
        0,
        1,
        &[pair.clone(), i32t.clone(), checked, i32t.clone()],
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
                vec![
                    assign(pl(3), mir::Rvalue::Use(mvf(2, 0, i32t.clone()))),
                    assign(
                        pl(0),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![cp(3), i32c(7)],
                        ),
                    ),
                ],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    // fn main() -> i32 { boom_pair(2_000_000_000).0 }
    let main = spec(
        "main_boom_pair",
        1,
        0,
        &[i32t.clone(), pair.clone()],
        vec![
            bb(
                vec![],
                call_term(fndef(def(0)), vec![i32c(2_000_000_000)], 1, 1),
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cpf(1, 0, i32t.clone())))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[boom_pair, main]);
    check_pair(&jit, &prog, "boom_pair", &[21], vec![int(21)], Ok((42, 7)));
    // On the panicking path neither engine produces result words:
    // the pair call itself fails with the panic, and the caller
    // propagates it instead of reading its destination.
    check_pair(
        &jit,
        &prog,
        "boom_pair",
        &[2_000_000_000],
        vec![int(2_000_000_000)],
        Err("attempt to add with overflow"),
    );
    check(
        &jit,
        &prog,
        "main_boom_pair",
        &[],
        vec![],
        Err("attempt to add with overflow"),
    );
}

#[test]
fn pair_return_into_a_memory_destination() {
    let i32t = i(mir::IntTy::I32);
    let pair = mir::Ty::Tuple(vec![i32t.clone(), i32t.clone()]);
    // fn main() -> i32 {
    //   _1 = (0, 0); _2 = &mut _1; *_2 = make_pair(5); _1.0 + _1.1
    // }
    let main = spec(
        "main_mem_dest",
        1,
        0,
        &[i32t.clone(), pair.clone(), refmty(pair.clone()), i32t.clone()],
        vec![
            bb(
                vec![
                    assign(
                        pl(1),
                        mir::Rvalue::Aggregate(
                            Box::new(mir::AggregateKind::Tuple),
                            vec![i32c(0), i32c(0)],
                        ),
                    ),
                    assign(pl(2), mutref(pl(1))),
                ],
                mir::TerminatorKind::Call {
                    func: fndef(def(0)),
                    args: vec![i32c(5)],
                    destination: dpl(2),
                    target: Some(mir::BasicBlock(1)),
                    unwind: mir::UnwindAction::Continue,
                    call_source: mir::CallSource::Normal,
                },
            ),
            bb(
                vec![
                    assign(
                        pl(3),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((cpf(1, 0, i32t.clone()), cpf(1, 1, i32t.clone()))),
                        ),
                    ),
                    assign(pl(0), mir::Rvalue::Use(cp(3))),
                ],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[make_pair_spec(), main]);
    check(&jit, &prog, "main_mem_dest", &[], vec![], Ok(11));
}

#[test]
fn three_word_return_via_sret() {
    let i32t = i(mir::IntTy::I32);
    let triple = mir::Ty::Tuple(vec![i32t.clone(), i32t.clone(), i32t.clone()]);
    // fn triple() -> (i32, i32, i32) { (10, 20, 30) }
    let f = spec(
        "triple",
        0,
        0,
        &[triple.clone()],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::Aggregate(
                    Box::new(mir::AggregateKind::Tuple),
                    vec![i32c(10), i32c(20), i32c(30)],
                ),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main_sret() -> i32 { let t = triple(); t.0 + t.1 + t.2 }
    let main = spec(
        "main_sret",
        1,
        0,
        &[i32t.clone(), triple, i32t.clone()],
        vec![
            bb(vec![], call_term(fndef(def(0)), vec![], 1, 1)),
            bb(
                vec![
                    assign(
                        pl(2),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((cpf(1, 0, i32t.clone()), cpf(1, 1, i32t.clone()))),
                        ),
                    ),
                    assign(
                        pl(0),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((cp(2), cpf(1, 2, i32t.clone()))),
                        ),
                    ),
                ],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[f, main]);
    check(&jit, &prog, "main_sret", &[], vec![], Ok(60));
    // The public call API supplies the sret word itself: `call`
    // yields the result's first word, `call_pair` its first two.
    match jit.call("triple", &[]) {
        Ok(v) => assert_eq!(v, 10),
        other => panic!("expected Ok(10), got {other:?}"),
    }
    match jit.call_pair("triple", &[]) {
        Ok(v) => assert_eq!(v, (10, 20)),
        other => panic!("expected Ok((10, 20)), got {other:?}"),
    }
}

#[test]
fn three_word_argument_and_seam_layout() {
    let i32t = i(mir::IntTy::I32);
    let triple = mir::Ty::Tuple(vec![i32t.clone(), i32t.clone(), i32t.clone()]);
    // fn sum3(t: (i32, i32, i32)) -> i32 { t.0 + t.1 + t.2 }
    let sum3 = spec(
        "sum3",
        0,
        1,
        &[i32t.clone(), triple.clone(), i32t.clone()],
        vec![bb(
            vec![
                assign(
                    pl(2),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((cpf(1, 0, i32t.clone()), cpf(1, 1, i32t.clone()))),
                    ),
                ),
                assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((cp(2), cpf(1, 2, i32t.clone()))),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn seam3(a, b, c, d: i32, t: (i32, i32, i32)) -> i32
    //   { a + b + c + d + t.0 + t.1 + t.2 }
    // The triple starts at word 4 and would straddle the
    // register/stack boundary, so both sides pad it to word 6.
    let mut stmts = vec![assign(
        pl(6),
        mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(1), cp(2)))),
    )];
    for k in 3..=4u32 {
        stmts.push(assign(
            pl(6),
            mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(6), cp(k)))),
        ));
    }
    for f in 0..2u32 {
        stmts.push(assign(
            pl(6),
            mir::Rvalue::BinaryOp(
                mir::BinOp::Add,
                Box::new((cp(6), cpf(5, f, i32t.clone()))),
            ),
        ));
    }
    stmts.push(assign(
        pl(0),
        mir::Rvalue::BinaryOp(
            mir::BinOp::Add,
            Box::new((cp(6), cpf(5, 2, i32t.clone()))),
        ),
    ));
    let seam3 = spec(
        "seam3",
        1,
        5,
        &[
            i32t.clone(),
            i32t.clone(),
            i32t.clone(),
            i32t.clone(),
            i32t.clone(),
            triple.clone(),
            i32t.clone(),
        ],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    // fn main_sum3() -> i32 { sum3((7, 8, 9)) }
    let main_sum3 = spec(
        "main_sum3",
        2,
        0,
        &[i32t.clone(), triple.clone()],
        vec![
            bb(
                vec![assign(
                    pl(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![i32c(7), i32c(8), i32c(9)],
                    ),
                )],
                call_term(fndef(def(0)), vec![cp(1)], 0, 1),
            ),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    // fn main_seam3() -> i32 { seam3(1, 2, 3, 4, (5, 6, 7)) }
    let main_seam3 = spec(
        "main_seam3",
        3,
        0,
        &[i32t.clone(), triple],
        vec![
            bb(
                vec![assign(
                    pl(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![i32c(5), i32c(6), i32c(7)],
                    ),
                )],
                call_term(
                    fndef(def(1)),
                    vec![i32c(1), i32c(2), i32c(3), i32c(4), cp(1)],
                    0,
                    1,
                ),
            ),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    let (jit, prog) = build(&[sum3, seam3, main_sum3, main_seam3]);
    check(&jit, &prog, "main_sum3", &[], vec![], Ok(24));
    check(&jit, &prog, "main_seam3", &[], vec![], Ok(28));
}

#[test]
fn array_aggregate_and_index() {
    let i32t = i(mir::IntTy::I32);
    let arr3 = mir::Ty::Array(Box::new(i32t.clone()), mir::ArrayLen::Known(3));
    let idx = |k: u64| mir::Place {
        local: mir::Local(1),
        projection: vec![mir::PlaceElem::ConstantIndex {
            offset: k,
            min_length: 3,
            from_end: false,
        }],
    };
    // fn main_arr() -> i32 { let a = [5, 6, 7]; a[0] + a[1] + a[2] }
    let main = spec(
        "main_arr",
        0,
        0,
        &[i32t.clone(), arr3, i32t.clone()],
        vec![bb(
            vec![
                assign(
                    pl(1),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Array(i32t.clone())),
                        vec![i32c(5), i32c(6), i32c(7)],
                    ),
                ),
                assign(
                    pl(2),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((
                            mir::Operand::Copy(idx(0)),
                            mir::Operand::Copy(idx(1)),
                        )),
                    ),
                ),
                assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((cp(2), mir::Operand::Copy(idx(2)))),
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[main]);
    check(&jit, &prog, "main_arr", &[], vec![], Ok(18));
}

#[test]
fn string_constant_as_str_argument() {
    let usizet = u(mir::UintTy::Usize);
    let strref = mir::Ty::Ref(
        mir::Region::Erased,
        Box::new(mir::Ty::Str),
        mir::Mutability::Not,
    );
    // fn slen_const(s: &str) -> usize { return PtrMetadata(_1) }
    let slen = spec(
        "slen_const",
        0,
        1,
        &[usizet.clone(), strref],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::UnaryOp(mir::UnOp::PtrMetadata, cp(1)),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main_strc() -> usize { slen_const("hello") }
    let hello = mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Str(
        "hello".to_string(),
    )));
    let main = spec(
        "main_strc",
        1,
        0,
        &[usizet],
        vec![
            bb(vec![], call_term(fndef(def(0)), vec![hello], 0, 1)),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    let (jit, prog) = build(&[slen, main]);
    check(&jit, &prog, "main_strc", &[], vec![], Ok(5));
}

#[test]
fn reference_constants_deref() {
    let i32t = i(mir::IntTy::I32);
    // fn main_refc() -> i32 { let r = &42; *r }
    let rc = mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Ref(
        Box::new(mir::ConstValue::Scalar(mir::Scalar::Int {
            data: 42,
            size: 4,
        })),
    )));
    let main = spec(
        "main_refc",
        0,
        0,
        &[i32t.clone(), refty(i32t.clone())],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(rc)),
                assign(pl(0), mir::Rvalue::Use(cpd(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[main]);
    check(&jit, &prog, "main_refc", &[], vec![], Ok(42));

    // fn main_fref() -> i32 { let r = &1.5f64; *r as i32 }
    let f64t = mir::Ty::Float(mir::FloatTy::F64);
    let fc = mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::Ref(
        Box::new(mir::ConstValue::Scalar(mir::Scalar::Float {
            bits: 1.5f64.to_bits() as u128,
            ty: mir::FloatTy::F64,
        })),
    )));
    let main_f = spec(
        "main_fref",
        0,
        0,
        &[i32t.clone(), refty(f64t.clone()), f64t],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(fc)),
                assign(pl(2), mir::Rvalue::Use(cpd(1))),
                assign(
                    pl(0),
                    mir::Rvalue::Cast(mir::CastKind::FloatToInt, cp(2), i32t.clone()),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[main_f]);
    check(&jit, &prog, "main_fref", &[], vec![], Ok(1));
}

// ----- Calls with more than six argument words (Lane CM) ---------------------

/// fn wsum(a, b, c, d, e, f, g, h: i64) -> i64
///   { a + 2*b + 3*c + ... + 8*h }  (distinct weights: any word
///   misalignment changes the result), plus a `main` that calls it
///   with constants.
fn many_scalar_args_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    let i64t = i(mir::IntTy::I64);
    // locals: 0 ret, 1..=8 args, 9 acc, 10 tmp
    let mut stmts = vec![assign(pl(9), mir::Rvalue::Use(cp(1)))];
    for k in 2..=8u32 {
        stmts.push(assign(
            pl(10),
            mir::Rvalue::BinaryOp(
                mir::BinOp::Mul,
                Box::new((cp(k), i64c(k as i64))),
            ),
        ));
        stmts.push(assign(
            pl(9),
            mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(9), cp(10)))),
        ));
    }
    stmts.push(assign(pl(0), mir::Rvalue::Use(cp(9))));
    let tys: Vec<mir::Ty> = (0..11).map(|_| i64t.clone()).collect();
    let wsum = spec("wsum", 0, 8, &tys, vec![bb(stmts, mir::TerminatorKind::Return)]);
    // fn main() -> i64 { wsum(1, 2, 3, 4, 5, 6, 7, 8) }
    let main = spec(
        "main_wsum",
        1,
        0,
        &[i64t.clone(), i64t.clone()],
        vec![
            bb(
                vec![],
                call_term(fndef(def(0)), (1..=8).map(i64c).collect(), 1, 1),
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(1)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    build(&[wsum, main])
}

#[test]
fn many_scalar_arguments() {
    let (jit, prog) = many_scalar_args_program();
    // Through a JIT-compiled caller (words 7 and 8 travel on the stack).
    check(&jit, &prog, "main_wsum", &[], vec![], Ok(204));
    // Through the Rust-side entry (the trampoline path).
    check(
        &jit,
        &prog,
        "wsum",
        &[1, 2, 3, 4, 5, 6, 7, 8],
        (1..=8).map(int).collect(),
        Ok(204),
    );
    check(
        &jit,
        &prog,
        "wsum",
        &[8, 7, 6, 5, 4, 3, 2, 1],
        [8, 7, 6, 5, 4, 3, 2, 1].iter().map(|v| int(*v)).collect(),
        Ok(120),
    );
}

/// fn pick6(a..f: i32, s: &[i32], i: usize) -> i32 { (*s)[i] + a }
/// (six scalars fill the registers; the fat reference is words 6-7
/// and the index word 8, both on the stack), and
/// fn pick_straddle(a..e: i32, s: &[i32], i: usize) -> i32
///   { (*s)[i] + a }
/// (five scalars: the fat reference would start at the last
/// register word, so a pad word precedes it and the pair lies
/// entirely on the stack — words 6-7, the index at word 8).
fn fat_ref_stack_args_program() -> (JitProgram, sparagmos_interp::MirProgram) {
    let i32t = i(mir::IntTy::I32);
    let usizet = u(mir::UintTy::Usize);
    let pick = |name: &'static str, def_id: u32, scalars: usize| {
        // locals: 0 ret, 1..=scalars i32, scalars+1 fat, scalars+2 usize,
        // scalars+3 tmp
        let fat_local = scalars as u32 + 1;
        let idx_local = fat_local + 1;
        let tmp_local = idx_local + 1;
        let mut tys: Vec<mir::Ty> = vec![i32t.clone()];
        tys.extend((0..scalars).map(|_| i32t.clone()));
        tys.push(fatrefty(i32t.clone()));
        tys.push(usizet.clone());
        tys.push(i32t.clone());
        spec(
            name,
            def_id,
            scalars + 2,
            &tys,
            vec![bb(
                vec![
                    assign(
                        pl(tmp_local),
                        mir::Rvalue::Use(mir::Operand::Copy(
                            dpl(fat_local).index(mir::Local(idx_local)),
                        )),
                    ),
                    assign(
                        pl(0),
                        mir::Rvalue::BinaryOp(
                            mir::BinOp::Add,
                            Box::new((cp(tmp_local), cp(1))),
                        ),
                    ),
                ],
                mir::TerminatorKind::Return,
            )],
        )
    };
    // fn main_pick() -> i32 {
    //   _1 = [5, 6, 7]; _5 = &_1; _2 = _5 as &[i32];
    //   pick6(10, 0, 0, 0, 0, 0, _2, 1)
    // }
    let main = spec(
        "main_pick",
        2,
        0,
        &[
            i32t.clone(),
            arr_i32(3),
            fatrefty(i32t.clone()),
            i32t.clone(),
            i32t.clone(),
            refty(arr_i32(3)),
        ],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Repeat(i32c(5), 3)),
                    assign(cidx(1, 1, 3), mir::Rvalue::Use(i32c(6))),
                    assign(cidx(1, 2, 3), mir::Rvalue::Use(i32c(7))),
                    assign(pl(5), shared(pl(1))),
                    assign(
                        pl(2),
                        mir::Rvalue::Cast(
                            mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
                            cp(5),
                            fatrefty(i32t.clone()),
                        ),
                    ),
                ],
                call_term(
                    fndef(def(0)),
                    vec![
                        i32c(10),
                        i32c(0),
                        i32c(0),
                        i32c(0),
                        i32c(0),
                        i32c(0),
                        cp(2),
                        u64c(1),
                    ],
                    3,
                    1,
                ),
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(3)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    build(&[pick("pick6", 0, 6), pick("pick_straddle", 1, 5), main])
}

#[test]
fn fat_reference_beyond_the_registers() {
    let (jit, prog) = fat_ref_stack_args_program();
    // The backing store, laid out the way a caller frame's slots
    // lie (element 0 at the highest address), as in the fat
    // reference tests above.
    let store: [i64; 3] = [30, 20, 10];
    let addr = store.as_ptr().wrapping_add(2) as i64;
    // JIT-compiled caller passing the fat reference on the stack.
    check(&jit, &prog, "main_pick", &[], vec![], Ok(16));
    // Rust-side entry: six scalars, then (addr, len), then the index.
    check(
        &jit,
        &prog,
        "pick6",
        &[10, 0, 0, 0, 0, 0, addr, 3, 1],
        vec![
            int(10),
            int(0),
            int(0),
            int(0),
            int(0),
            int(0),
            fatval(&[10, 20, 30], 3),
            uint(1),
        ],
        Ok(30),
    );
    // The straddling shape: five scalars, a pad word (0), the pair,
    // the index. The interpreter sees only the MIR-level arguments.
    check(
        &jit,
        &prog,
        "pick_straddle",
        &[10, 0, 0, 0, 0, 0, addr, 3, 2],
        vec![
            int(10),
            int(0),
            int(0),
            int(0),
            int(0),
            fatval(&[10, 20, 30], 3),
            uint(2),
        ],
        Ok(40),
    );
    // An out-of-bounds index through the straddled pair panics
    // identically on both engines (the length word arrived intact).
    check(
        &jit,
        &prog,
        "pick_straddle",
        &[10, 0, 0, 0, 0, 0, addr, 3, 5],
        vec![
            int(10),
            int(0),
            int(0),
            int(0),
            int(0),
            fatval(&[10, 20, 30], 3),
            uint(5),
        ],
        Err("index out of bounds: the len is 3 but the index is 5"),
    );
}

#[test]
fn many_arguments_returning_a_pair() {
    let i64t = i(mir::IntTy::I64);
    let pair_ty = mir::Ty::Tuple(vec![i64t.clone(), i64t.clone()]);
    // fn mkpair(a..g: i64) -> (i64, i64) { (a + g, b * f) }
    // locals: 0 ret, 1..=7 args, 8 t0, 9 t1
    let mkpair = spec(
        "mkpair",
        0,
        7,
        &[
            pair_ty.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
        ],
        vec![bb(
            vec![
                assign(
                    pl(8),
                    mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(1), cp(7)))),
                ),
                assign(
                    pl(9),
                    mir::Rvalue::BinaryOp(mir::BinOp::Mul, Box::new((cp(2), cp(6)))),
                ),
                assign(
                    pl(0),
                    mir::Rvalue::Aggregate(
                        Box::new(mir::AggregateKind::Tuple),
                        vec![cp(8), cp(9)],
                    ),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main() -> i64 { let p = mkpair(1, 2, 3, 4, 5, 6, 7); p.0 + p.1 }
    let main = spec(
        "main_mkpair",
        1,
        0,
        &[i64t.clone(), pair_ty.clone()],
        vec![
            bb(
                vec![],
                call_term(fndef(def(0)), (1..=7).map(i64c).collect(), 1, 1),
            ),
            bb(
                vec![assign(
                    pl(0),
                    mir::Rvalue::BinaryOp(
                        mir::BinOp::Add,
                        Box::new((cpf(1, 0, i64t.clone()), cpf(1, 1, i64t.clone()))),
                    ),
                )],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[mkpair, main]);
    check(&jit, &prog, "main_mkpair", &[], vec![], Ok(20));
    // The pair itself, through the Rust-side pair entry.
    assert_eq!(
        jit.call_pair("mkpair", &[1, 2, 3, 4, 5, 6, 7]).unwrap(),
        (8, 12)
    );
}

#[test]
fn unit_argument_occupies_a_dummy_word() {
    let i64t = i(mir::IntTy::I64);
    let unit = mir::Ty::Tuple(vec![]);
    // fn with_unit(a: i64, u: (), b..f: i64) -> i64 { a + b + c + d + e + f }
    // The unit parameter's dummy word is word 1; f is word 6, on the
    // stack. locals: 0 ret, 1 a, 2 u, 3..=7 b..f, 8 acc
    let mut stmts = vec![assign(pl(8), mir::Rvalue::Use(cp(1)))];
    for k in 3..=7u32 {
        stmts.push(assign(
            pl(8),
            mir::Rvalue::BinaryOp(mir::BinOp::Add, Box::new((cp(8), cp(k)))),
        ));
    }
    stmts.push(assign(pl(0), mir::Rvalue::Use(cp(8))));
    let with_unit = spec(
        "with_unit",
        0,
        7,
        &[
            i64t.clone(),
            i64t.clone(),
            unit.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
            i64t.clone(),
        ],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    // fn main() -> i64 { _1 = (); with_unit(10, _1, 1, 2, 3, 4, 5) }
    // locals: 0 ret, 1 unit placeholder, 2 dest
    let main = spec(
        "main_unit",
        1,
        0,
        &[i64t.clone(), unit.clone(), i64t.clone()],
        vec![
            bb(
                vec![assign(
                    pl(1),
                    mir::Rvalue::Aggregate(Box::new(mir::AggregateKind::Tuple), vec![]),
                )],
                call_term(
                    fndef(def(0)),
                    vec![i64c(10), cp(1), i64c(1), i64c(2), i64c(3), i64c(4), i64c(5)],
                    2,
                    1,
                ),
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(2)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[with_unit, main]);
    check(&jit, &prog, "main_unit", &[], vec![], Ok(25));
    // Rust-side entry: the dummy word is passed explicitly.
    check(
        &jit,
        &prog,
        "with_unit",
        &[10, 0, 1, 2, 3, 4, 5],
        vec![
            int(10),
            Value::Unit,
            int(1),
            int(2),
            int(3),
            int(4),
            int(5),
        ],
        Ok(25),
    );
}

// ----- Reborrow / Intrinsic / no-op InlineAsm (Lane CO) -----

#[test]
fn intrinsic_assume_true_and_inline_asm_noop_do_not_affect_execution() {
    let i32t = i(mir::IntTy::I32);
    // fn f() -> i32 { _1 = 7; assume(true); asm!("nop"); return _1 }
    let f = spec(
        "f",
        0,
        0,
        &[i32t.clone(), i32t.clone()],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Use(i32c(7))),
                    mir::StatementKind::Intrinsic(Box::new(mir::NonDivergingIntrinsic::Assume(
                        mir::Operand::constant(mir::Constant::bool(true)),
                    ))),
                ],
                mir::TerminatorKind::InlineAsm {
                    template: "nop".to_string(),
                    operands: Vec::new(),
                    options: String::new(),
                    destination: Some(mir::BasicBlock(1)),
                    unwind: mir::UnwindAction::Continue,
                },
            ),
            bb(
                vec![assign(pl(0), mir::Rvalue::Use(cp(1)))],
                mir::TerminatorKind::Return,
            ),
        ],
    );
    let (jit, prog) = build(&[f]);
    check(&jit, &prog, "f", &[], vec![], Ok(7));
}

#[test]
fn reborrow_copies_the_pointer_value() {
    let i32t = i(mir::IntTy::I32);
    // fn read_shared() -> i32 { _1 = 41; _2 = &mut _1; _3 = Reborrow(*const i32, _2); return *_3 }
    let read_shared = spec(
        "read_shared",
        0,
        0,
        &[
            i32t.clone(),
            i32t.clone(),
            refmty(i32t.clone()),
            rawty(i32t.clone(), false),
        ],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(41))),
                assign(pl(2), mutref(pl(1))),
                assign(
                    pl(3),
                    mir::Rvalue::Reborrow(
                        mir::Ty::RawPtr(Box::new(i32t.clone()), mir::Mutability::Not),
                        mir::Mutability::Not,
                        pl(2),
                    ),
                ),
                assign(pl(0), mir::Rvalue::Use(cpd(3))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn write_mut() -> i32 { _1 = 1; _2 = &mut _1; _3 = Reborrow(*mut i32, _2); *_3 = 99; return _1 }
    let write_mut = spec(
        "write_mut",
        1,
        0,
        &[
            i32t.clone(),
            i32t.clone(),
            refmty(i32t.clone()),
            rawty(i32t.clone(), true),
        ],
        vec![bb(
            vec![
                assign(pl(1), mir::Rvalue::Use(i32c(1))),
                assign(pl(2), mutref(pl(1))),
                assign(
                    pl(3),
                    mir::Rvalue::Reborrow(
                        mir::Ty::RawPtr(Box::new(i32t.clone()), mir::Mutability::Mut),
                        mir::Mutability::Mut,
                        pl(2),
                    ),
                ),
                assign(dpl(3), mir::Rvalue::Use(i32c(99))),
                assign(pl(0), mir::Rvalue::Use(cp(1))),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[read_shared, write_mut]);
    check(&jit, &prog, "read_shared", &[], vec![], Ok(41));
    check(&jit, &prog, "write_mut", &[], vec![], Ok(99));
}

// ----- signed integer indices -------------------------------------------------

#[test]
fn signed_index_into_a_local_array() {
    let i32t = i(mir::IntTy::I32);
    // fn idx(i: i32) -> i32 { _2 = Repeat(3, 2); _2[1] = 7; return _2[_1] }
    let idx = spec(
        "idx",
        0,
        1,
        &[i32t.clone(), i32t.clone(), arr_i32(2)],
        vec![bb(
            vec![
                assign(pl(2), mir::Rvalue::Repeat(i32c(3), 2)),
                assign(cidx(2, 1, 2), mir::Rvalue::Use(i32c(7))),
                assign(
                    pl(0),
                    mir::Rvalue::Use(mir::Operand::Copy(pl(2).index(mir::Local(1)))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn set(i: i32, v: i32) -> i32 { _3 = Repeat(1, 2); _3[_1] = _2; return _3[_1] }
    let set = spec(
        "set",
        1,
        2,
        &[i32t.clone(), i32t.clone(), i32t.clone(), arr_i32(2)],
        vec![bb(
            vec![
                assign(pl(3), mir::Rvalue::Repeat(i32c(1), 2)),
                assign(pl(3).index(mir::Local(1)), mir::Rvalue::Use(cp(2))),
                assign(
                    pl(0),
                    mir::Rvalue::Use(mir::Operand::Copy(pl(3).index(mir::Local(1)))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[idx, set]);
    check(&jit, &prog, "idx", &[0], vec![int(0)], Ok(3));
    check(&jit, &prog, "idx", &[1], vec![int(1)], Ok(7));
    check(
        &jit,
        &prog,
        "idx",
        &[2],
        vec![int(2)],
        Err("index out of bounds: the len is 2 but the index is 2"),
    );
    check(
        &jit,
        &prog,
        "idx",
        &[-1],
        vec![int(-1)],
        Err("index out of bounds: negative index -1 used for _2[_1]"),
    );
    check(&jit, &prog, "set", &[1, 42], vec![int(1), int(42)], Ok(42));
    check(
        &jit,
        &prog,
        "set",
        &[-2, 42],
        vec![int(-2), int(42)],
        Err("index out of bounds: negative index -2 used for _3[_1]"),
    );
}

#[test]
fn signed_index_into_a_slice() {
    let i32t = i(mir::IntTy::I32);
    // fn get(s: &[i32], i: i64) -> i32 { return (*s)[_2] }
    let get = spec(
        "get",
        0,
        2,
        &[i32t.clone(), fatrefty(i32t.clone()), i(mir::IntTy::I64)],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::Use(mir::Operand::Copy(dpl(1).index(mir::Local(2)))),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    let (jit, prog) = build(&[get]);
    let store: [i64; 3] = [30, 20, 10];
    let addr = store.as_ptr().wrapping_add(2) as i64;
    check(
        &jit,
        &prog,
        "get",
        &[addr, 3, 1],
        vec![fatval(&[10, 20, 30], 3), int(1)],
        Ok(20),
    );
    check(
        &jit,
        &prog,
        "get",
        &[addr, 3, 5],
        vec![fatval(&[10, 20, 30], 3), int(5)],
        Err("index out of bounds: the len is 3 but the index is 5"),
    );
    check(
        &jit,
        &prog,
        "get",
        &[addr, 3, -2],
        vec![fatval(&[10, 20, 30], 3), int(-2)],
        Err("index out of bounds: negative index -2 used for _1.*[_2]"),
    );
}

// ----- sub-slice projections ----------------------------------------------------

/// `_n[from..to]`, or `_n[from..len - to]` when `from_end`: a
/// sub-slice projection in the shape rustc emits for a range
/// subscript of an array place.
fn sub(n: u32, from: u64, to: u64, from_end: bool) -> mir::Place {
    mir::Place {
        local: mir::Local(n),
        projection: vec![mir::PlaceElem::Subslice { from, to, from_end }],
    }
}

/// `(*_n)[from..to]` (or the `from_end` form): a sub-slice of a
/// slice reached through a dereference.
fn dsub(n: u32, from: u64, to: u64, from_end: bool) -> mir::Place {
    mir::Place {
        local: mir::Local(n),
        projection: vec![
            mir::PlaceElem::Deref,
            mir::PlaceElem::Subslice { from, to, from_end },
        ],
    }
}

/// Append a constant index projection to a place.
fn then_cidx(mut p: mir::Place, offset: u64, min_length: u64) -> mir::Place {
    p.projection.push(mir::PlaceElem::ConstantIndex {
        offset,
        min_length,
        from_end: false,
    });
    p
}

/// Append a dynamic index projection to a place.
fn then_idx(mut p: mir::Place, idx_local: u32) -> mir::Place {
    p.projection.push(mir::PlaceElem::Index(mir::Local(idx_local)));
    p
}

/// The statements that fill local `_1` (an `[i32; 5]`) with
/// `[10, 20, 30, 40, 50]`.
fn arr5_fill() -> Vec<mir::StatementKind> {
    let mut stmts = vec![assign(pl(1), mir::Rvalue::Repeat(i32c(0), 5))];
    for (k, v) in [10, 20, 30, 40, 50].iter().enumerate() {
        stmts.push(assign(
            cidx(1, k as u64, 5),
            mir::Rvalue::Use(i32c(*v)),
        ));
    }
    stmts
}

#[test]
fn subslice_of_a_local_array() {
    let i32t = i(mir::IntTy::I32);
    let usizet = u(mir::UintTy::Usize);
    // fn sublen() -> usize { _1 = [10,20,30,40,50]; return Len(_1[1..3]) }
    let mut stmts = arr5_fill();
    stmts.push(assign(pl(0), mir::Rvalue::Len(sub(1, 1, 3, false))));
    let sublen = spec(
        "sublen",
        0,
        0,
        &[usizet.clone(), arr_i32(5)],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    // fn subget() -> i32 { return _1[1..4][2] }
    let mut stmts = arr5_fill();
    stmts.push(assign(
        pl(0),
        mir::Rvalue::Use(mir::Operand::Copy(then_cidx(sub(1, 1, 4, false), 2, 3))),
    ));
    let subget = spec(
        "subget",
        1,
        0,
        &[i32t.clone(), arr_i32(5)],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    // fn subdyn(i: usize) -> i32 { _2 = [10, 20, 30, 40, 50]; return _2[1..4][_1] }
    let mut stmts = vec![assign(pl(2), mir::Rvalue::Repeat(i32c(0), 5))];
    for (k, v) in [10, 20, 30, 40, 50].iter().enumerate() {
        stmts.push(assign(cidx(2, k as u64, 5), mir::Rvalue::Use(i32c(*v))));
    }
    stmts.push(assign(
        pl(0),
        mir::Rvalue::Use(mir::Operand::Copy(then_idx(sub(2, 1, 4, false), 1))),
    ));
    let subdyn = spec(
        "subdyn",
        2,
        1,
        &[i32t.clone(), usizet.clone(), arr_i32(5)],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    // fn subend() -> usize { return Len(_1[1..len - 1]) }
    let mut stmts = arr5_fill();
    stmts.push(assign(pl(0), mir::Rvalue::Len(sub(1, 1, 1, true))));
    let subend = spec(
        "subend",
        3,
        0,
        &[usizet.clone(), arr_i32(5)],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    // fn subsub() -> i32 { return _1[1..4][1..3][1] }
    let mut inner = sub(1, 1, 4, false);
    inner.projection.push(mir::PlaceElem::Subslice {
        from: 1,
        to: 3,
        from_end: false,
    });
    let mut stmts = arr5_fill();
    stmts.push(assign(
        pl(0),
        mir::Rvalue::Use(mir::Operand::Copy(then_cidx(inner, 1, 2))),
    ));
    let subsub = spec(
        "subsub",
        4,
        0,
        &[i32t.clone(), arr_i32(5)],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    // fn subref() -> i32 { _2 = &_1[1..4]; return (*_2)[2] }
    let mut stmts = arr5_fill();
    stmts.push(assign(pl(2), shared(sub(1, 1, 4, false))));
    stmts.push(assign(
        pl(0),
        mir::Rvalue::Use(mir::Operand::Copy(then_cidx(dpl(2), 2, 3))),
    ));
    let subref = spec(
        "subref",
        5,
        0,
        &[i32t.clone(), arr_i32(5), fatrefty(i32t.clone())],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    // fn subreflen() -> usize { _2 = &_1[1..4]; return Len(*_2) }
    let mut stmts = arr5_fill();
    stmts.push(assign(pl(2), shared(sub(1, 1, 4, false))));
    stmts.push(assign(pl(0), mir::Rvalue::Len(dpl(2))));
    let subreflen = spec(
        "subreflen",
        6,
        0,
        &[usizet.clone(), arr_i32(5), fatrefty(i32t.clone())],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    let (jit, prog) = build(&[sublen, subget, subdyn, subend, subsub, subref, subreflen]);
    check(&jit, &prog, "sublen", &[], vec![], Ok(2));
    check(&jit, &prog, "subget", &[], vec![], Ok(40));
    check(&jit, &prog, "subdyn", &[0], vec![uint(0)], Ok(20));
    check(&jit, &prog, "subdyn", &[2], vec![uint(2)], Ok(40));
    check(
        &jit,
        &prog,
        "subdyn",
        &[3],
        vec![uint(3)],
        Err("index out of bounds: the len is 3 but the index is 3"),
    );
    check(&jit, &prog, "subend", &[], vec![], Ok(3));
    check(&jit, &prog, "subsub", &[], vec![], Ok(40));
    check(&jit, &prog, "subref", &[], vec![], Ok(40));
    check(&jit, &prog, "subreflen", &[], vec![], Ok(3));
}

#[test]
fn subslice_of_a_local_array_out_of_range() {
    let usizet = u(mir::UintTy::Usize);
    // fn bad_end() -> usize { return Len(_1[3..9]) }
    let mut stmts = arr5_fill();
    stmts.push(assign(pl(0), mir::Rvalue::Len(sub(1, 3, 9, false))));
    let bad_end = spec(
        "bad_end",
        0,
        0,
        &[usizet.clone(), arr_i32(5)],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    // fn bad_inverted() -> usize { return Len(_1[4..2]) }
    let mut stmts = arr5_fill();
    stmts.push(assign(pl(0), mir::Rvalue::Len(sub(1, 4, 2, false))));
    let bad_inverted = spec(
        "bad_inverted",
        1,
        0,
        &[usizet.clone(), arr_i32(5)],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    // fn bad_from_end() -> usize { return Len(_1[0..len - 9]) }
    let mut stmts = arr5_fill();
    stmts.push(assign(pl(0), mir::Rvalue::Len(sub(1, 0, 9, true))));
    let bad_from_end = spec(
        "bad_from_end",
        2,
        0,
        &[usizet.clone(), arr_i32(5)],
        vec![bb(stmts, mir::TerminatorKind::Return)],
    );
    let (jit, prog) = build(&[bad_end, bad_inverted, bad_from_end]);
    check(
        &jit,
        &prog,
        "bad_end",
        &[],
        vec![],
        Err("range end index 9 out of range for slice of length 5"),
    );
    check(
        &jit,
        &prog,
        "bad_inverted",
        &[],
        vec![],
        Err("slice index starts at 4 but ends at 2"),
    );
    check(
        &jit,
        &prog,
        "bad_from_end",
        &[],
        vec![],
        Err("range end index 9 out of range for slice of length 5"),
    );
}

#[test]
fn subslice_of_a_slice_argument() {
    let i32t = i(mir::IntTy::I32);
    let usizet = u(mir::UintTy::Usize);
    // fn slen2(s: &[i32]) -> usize { return Len((*s)[1..3]) }
    let slen2 = spec(
        "slen2",
        0,
        1,
        &[usizet.clone(), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Len(dsub(1, 1, 3, false)))],
            mir::TerminatorKind::Return,
        )],
    );
    // fn stail(s: &[i32]) -> usize { return Len((*s)[1..]) }
    let stail = spec(
        "stail",
        1,
        1,
        &[usizet.clone(), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Len(dsub(1, 1, 0, true)))],
            mir::TerminatorKind::Return,
        )],
    );
    // fn send(s: &[i32]) -> i32 { return (*s)[1..len - 1][1] }
    let send = spec(
        "send",
        2,
        1,
        &[i32t.clone(), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(
                pl(0),
                mir::Rvalue::Use(mir::Operand::Copy(then_cidx(dsub(1, 1, 1, true), 1, 2))),
            )],
            mir::TerminatorKind::Return,
        )],
    );
    // fn sset(s: &mut [i32], v: i32) -> i32 { (*s)[1..3][0] = _2; return (*s)[1] }
    let sset = spec(
        "sset",
        3,
        2,
        &[i32t.clone(), fatrefmty(i32t.clone()), i32t.clone()],
        vec![bb(
            vec![
                assign(then_cidx(dsub(1, 1, 3, false), 0, 2), mir::Rvalue::Use(cp(2))),
                assign(
                    pl(0),
                    mir::Rvalue::Use(mir::Operand::Copy(then_cidx(dpl(1), 1, 4))),
                ),
            ],
            mir::TerminatorKind::Return,
        )],
    );
    // fn sbad_end(s: &[i32]) -> usize { return Len((*s)[2..9]) }
    let sbad_end = spec(
        "sbad_end",
        4,
        1,
        &[usizet.clone(), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Len(dsub(1, 2, 9, false)))],
            mir::TerminatorKind::Return,
        )],
    );
    // fn sbad_from_end(s: &[i32]) -> usize { return Len((*s)[0..len - 9]) }
    let sbad_from_end = spec(
        "sbad_from_end",
        5,
        1,
        &[usizet.clone(), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Len(dsub(1, 0, 9, true)))],
            mir::TerminatorKind::Return,
        )],
    );
    // fn sbad_inverted(s: &[i32]) -> usize { return Len((*s)[3..2]) }
    let sbad_inverted = spec(
        "sbad_inverted",
        6,
        1,
        &[usizet.clone(), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Len(dsub(1, 3, 2, false)))],
            mir::TerminatorKind::Return,
        )],
    );
    // fn sbad_inverted_end(s: &[i32]) -> usize { return Len((*s)[3..len - 2]) }
    let sbad_inverted_end = spec(
        "sbad_inverted_end",
        7,
        1,
        &[usizet.clone(), fatrefty(i32t.clone())],
        vec![bb(
            vec![assign(pl(0), mir::Rvalue::Len(dsub(1, 3, 2, true)))],
            mir::TerminatorKind::Return,
        )],
    );
    // fn main_sset() -> i32 { _1 = [10, 20, 30, 40]; _2 = &mut _1; return sset(_2, 99) }
    let main_sset = spec(
        "main_sset",
        8,
        0,
        &[i32t.clone(), arr_i32(4), fatrefmty(i32t.clone())],
        vec![
            bb(
                vec![
                    assign(pl(1), mir::Rvalue::Repeat(i32c(0), 4)),
                    assign(cidx(1, 0, 4), mir::Rvalue::Use(i32c(10))),
                    assign(cidx(1, 1, 4), mir::Rvalue::Use(i32c(20))),
                    assign(cidx(1, 2, 4), mir::Rvalue::Use(i32c(30))),
                    assign(cidx(1, 3, 4), mir::Rvalue::Use(i32c(40))),
                    assign(pl(2), mutref(pl(1))),
                ],
                mir::TerminatorKind::Call {
                    func: fndef(def(3)),
                    args: vec![mir::Operand::Move(pl(2)), i32c(99)],
                    destination: pl(0),
                    target: Some(mir::BasicBlock(1)),
                    unwind: mir::UnwindAction::Continue,
                    call_source: mir::CallSource::Normal,
                },
            ),
            bb(vec![], mir::TerminatorKind::Return),
        ],
    );
    let (jit, prog) = build(&[
        slen2,
        stail,
        send,
        sset,
        sbad_end,
        sbad_from_end,
        sbad_inverted,
        sbad_inverted_end,
        main_sset,
    ]);
    let store: [i64; 4] = [40, 30, 20, 10];
    let addr = store.as_ptr().wrapping_add(3) as i64;
    let arg = |jit: &JitProgram, prog: &sparagmos_interp::MirProgram, name: &str, expected| {
        check(jit, prog, name, &[addr, 4], vec![fatval(&[10, 20, 30, 40], 4)], expected);
    };
    arg(&jit, &prog, "slen2", Ok(2));
    arg(&jit, &prog, "stail", Ok(3));
    arg(&jit, &prog, "send", Ok(30));
    check(&jit, &prog, "main_sset", &[], vec![], Ok(99));
    arg(
        &jit,
        &prog,
        "sbad_end",
        Err("range end index 9 out of range for slice of length 4"),
    );
    arg(
        &jit,
        &prog,
        "sbad_from_end",
        Err("range end index 9 out of range for slice of length 4"),
    );
    arg(
        &jit,
        &prog,
        "sbad_inverted",
        Err("slice index starts at 3 but ends at 2"),
    );
    arg(
        &jit,
        &prog,
        "sbad_inverted_end",
        Err("slice index starts at 3 but ends at 2"),
    );
}
