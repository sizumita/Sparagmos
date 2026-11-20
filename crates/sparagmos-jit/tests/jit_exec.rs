//! Integration tests: JIT-compile IR and actually execute the generated
//! machine code, checking results against native Rust computation.

use sparagmos_jit::emitter::{Emitter, Reg};
use sparagmos_jit::ir::{BinOp as B, FloatBinOp as F, FunctionBuilder, Operand, Terminator};
use sparagmos_jit::{compile, FloatKind, Instr, IntKind, JitMemory};

#[test]
fn jit_add_2_3_is_5() {
    let mut b = FunctionBuilder::new("add", 2);
    let (a, c) = (b.param(0), b.param(1));
    let r = b.binop(B::Add, a, c);
    let f = b.finish(Terminator::Return { value: r.into(), second: None });
    let cf = compile(&f).unwrap();
    // SAFETY: cf was compiled from the IR above: fn(i64, i64) -> i64.
    assert_eq!(unsafe { cf.call2(2, 3) }, 5);
}

#[test]
fn jit_factorial_loop() {
    // fn fact(n): acc=1; i=1; loop if i<=n { acc*=i; i+=1 } return acc
    let mut b = FunctionBuilder::new("fact", 1);
    let n = b.param(0);
    let acc = b.const_(1);
    let i = b.const_(1);
    b.end_block(Terminator::Jump(1));
    let cond = b.binop(B::Le, i, n);
    b.end_block(Terminator::Branch {
        cond: cond.into(),
        then_blk: 2,
        else_blk: 3,
    });
    b.push_raw(Instr::BinOp {
        dst: acc,
        op: B::Mul,
        lhs: acc.into(),
        rhs: i.into(),
    });
    b.push_raw(Instr::BinOp {
        dst: i,
        op: B::Add,
        lhs: i.into(),
        rhs: Operand::Const(1),
    });
    b.end_block(Terminator::Jump(1));
    let f = b.finish(Terminator::Return { value: acc.into(), second: None });
    let cf = compile(&f).unwrap();
    assert_eq!(unsafe { cf.call1(0) }, 1);
    assert_eq!(unsafe { cf.call1(5) }, 120);
    assert_eq!(unsafe { cf.call1(10) }, 3_628_800);
}

#[test]
fn jit_raw_emitter_max() {
    // Hand-emitted: fn max(a: i64 /*rdi*/, b: i64 /*rsi*/) -> i64
    //   cmp rdi, rsi ; jge done ; mov rdi, rsi ; done: mov rax, rdi ; ret
    let mut e = Emitter::new();
    e.cmp_rr(Reg::Rdi, Reg::Rsi);
    let jge_at = e.jcc_rel32(sparagmos_jit::Cond::GreaterOrEqual);
    e.mov_rr(Reg::Rdi, Reg::Rsi);
    let done = e.position();
    e.mov_rr(Reg::Rax, Reg::Rdi);
    e.ret();
    e.buf.patch_rel32(jge_at, done);

    let mut mem = JitMemory::new(e.buf.as_slice().len()).unwrap();
    mem.write_code(e.buf.as_slice()).unwrap();
    mem.make_executable().unwrap();
    // SAFETY: bytes above implement exactly fn(i64, i64) -> i64 (SysV).
    let f: unsafe extern "C" fn(i64, i64) -> i64 = unsafe { mem.function_ptr() };
    assert_eq!(unsafe { f(7, 3) }, 7);
    assert_eq!(unsafe { f(2, 9) }, 9);
    assert_eq!(unsafe { f(-5, -1) }, -1);
}

#[test]
fn jit_gcd_loop() {
    // Euclid: fn gcd(a,b): loop if b!=0 { t=b; b=a%b; a=t } return a
    let mut b = FunctionBuilder::new("gcd", 2);
    let a = b.param(0);
    let bb = b.param(1);
    let t = b.fresh();
    b.end_block(Terminator::Jump(1));
    let cond = b.binop(B::Ne, bb, 0i64);
    b.end_block(Terminator::Branch {
        cond: cond.into(),
        then_blk: 2,
        else_blk: 3,
    });
    b.push_raw(Instr::Copy {
        dst: t,
        src: bb.into(),
    });
    b.push_raw(Instr::BinOp {
        dst: bb,
        op: B::Rem,
        lhs: a.into(),
        rhs: bb.into(),
    });
    b.push_raw(Instr::Copy {
        dst: a,
        src: t.into(),
    });
    b.end_block(Terminator::Jump(1));
    let f = b.finish(Terminator::Return { value: a.into(), second: None });
    let cf = compile(&f).unwrap();
    assert_eq!(unsafe { cf.call2(48, 36) }, 12);
    assert_eq!(unsafe { cf.call2(17, 5) }, 1);
    assert_eq!(unsafe { cf.call2(100, 10) }, 10);
}

// ----- floating point (arguments and results travel as bit patterns) --------

#[test]
fn jit_f64_arithmetic() {
    // fn f(a, b) = (a + b) * (a - b) / b  on f64 bit patterns
    let mut b = FunctionBuilder::new("f", 2);
    let (a, c) = (b.param(0), b.param(1));
    let s = b.float_binop(F::Add, FloatKind::F64, a, c);
    let d = b.float_binop(F::Sub, FloatKind::F64, a, c);
    let m = b.float_binop(F::Mul, FloatKind::F64, s, d);
    let q = b.float_binop(F::Div, FloatKind::F64, m, c);
    let f = b.finish(Terminator::Return { value: q.into(), second: None });
    let cf = compile(&f).unwrap();
    let run = |x: f64, y: f64| f64::from_bits(unsafe { cf.call2(x.to_bits() as i64, y.to_bits() as i64) } as u64);
    for (x, y) in [(3.5f64, 1.5f64), (0.1, 0.2), (-7.25, 0.5), (1e308, 1e308)] {
        assert_eq!(run(x, y).to_bits(), ((x + y) * (x - y) / y).to_bits(), "({x}, {y})");
    }
}

#[test]
fn jit_f32_arithmetic_and_rem() {
    // fn f(a, b) = (a % b) + a * b  on f32 bit patterns (zero-extended)
    let mut b = FunctionBuilder::new("f", 2);
    let (a, c) = (b.param(0), b.param(1));
    let r = b.float_binop(F::Rem, FloatKind::F32, a, c);
    let m = b.float_binop(F::Mul, FloatKind::F32, a, c);
    let s = b.float_binop(F::Add, FloatKind::F32, r, m);
    let f = b.finish(Terminator::Return { value: s.into(), second: None });
    let cf = compile(&f).unwrap();
    let run = |x: f32, y: f32| {
        let bits = unsafe { cf.call2(x.to_bits() as i64, y.to_bits() as i64) } as u64;
        assert_eq!(bits >> 32, 0, "f32 result must be canonical (upper half zero)");
        f32::from_bits(bits as u32)
    };
    for (x, y) in [(7.5f32, 2.0f32), (-7.5, 2.0), (0.1, 0.3), (5.0, 0.0)] {
        let want = (x % y) + x * y;
        assert_eq!(run(x, y).to_bits(), want.to_bits(), "({x}, {y})");
    }
}

#[test]
fn jit_f64_rem() {
    let mut b = FunctionBuilder::new("f", 2);
    let (a, c) = (b.param(0), b.param(1));
    let r = b.float_binop(F::Rem, FloatKind::F64, a, c);
    let f = b.finish(Terminator::Return { value: r.into(), second: None });
    let cf = compile(&f).unwrap();
    let run = |x: f64, y: f64| f64::from_bits(unsafe { cf.call2(x.to_bits() as i64, y.to_bits() as i64) } as u64);
    for (x, y) in [
        (7.5f64, 2.0f64),
        (-7.5, 2.0),
        (7.5, -2.0),
        (1e300, 3.0),
        (5.0, f64::INFINITY),
        (f64::INFINITY, 2.0),
    ] {
        assert_eq!(run(x, y).to_bits(), (x % y).to_bits(), "({x}, {y})");
    }
}

#[test]
fn jit_f64_comparisons() {
    for (op, want_fn) in [
        (F::Eq, f64::eq as fn(&f64, &f64) -> bool),
        (F::Ne, f64::ne),
        (F::Lt, f64::lt),
        (F::Le, f64::le),
        (F::Gt, f64::gt),
        (F::Ge, f64::ge),
    ] {
        let mut b = FunctionBuilder::new("cmp", 2);
        let (a, c) = (b.param(0), b.param(1));
        let r = b.float_binop(op, FloatKind::F64, a, c);
        let f = b.finish(Terminator::Return { value: r.into(), second: None });
        let cf = compile(&f).unwrap();
        for (x, y) in [
            (1.0f64, 2.0f64),
            (2.0, 1.0),
            (1.5, 1.5),
            (0.0, -0.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::NAN, f64::NAN),
            (f64::INFINITY, 1e308),
            (f64::NEG_INFINITY, -1e308),
        ] {
            let got = unsafe { cf.call2(x.to_bits() as i64, y.to_bits() as i64) };
            assert_eq!(got, want_fn(&x, &y) as i64, "{op:?} ({x}, {y})");
        }
    }
}

#[test]
fn jit_float_casts() {
    // f64 -> i32 (saturating) via a raw instruction.
    let mut b = FunctionBuilder::new("f2i", 1);
    let a = b.param(0);
    let dst = b.fresh();
    b.push_raw(Instr::FloatToInt {
        dst,
        kind: FloatKind::F64,
        src: a.into(),
        int: IntKind::new(32, true),
    });
    let f = b.finish(Terminator::Return { value: dst.into(), second: None });
    let cf = compile(&f).unwrap();
    let run = |x: f64| unsafe { cf.call1(x.to_bits() as i64) };
    assert_eq!(run(3.99), 3);
    assert_eq!(run(-3.99), -3);
    assert_eq!(run(f64::NAN), 0);
    assert_eq!(run(3.0e9), i32::MAX as i64);
    assert_eq!(run(-3.0e9), i32::MIN as i64);
    assert_eq!(run(2147483647.0), i32::MAX as i64);
    assert_eq!(run(-2147483648.0), i32::MIN as i64);

    // u64 -> f64 via a raw instruction.
    let mut b = FunctionBuilder::new("u2f", 1);
    let a = b.param(0);
    let dst = b.fresh();
    b.push_raw(Instr::IntToFloat {
        dst,
        kind: FloatKind::F64,
        src: a.into(),
        int: IntKind::new(64, false),
    });
    let f = b.finish(Terminator::Return { value: dst.into(), second: None });
    let cf = compile(&f).unwrap();
    for x in [0u64, 1, 42, u64::MAX, u64::MAX - 1, 1 << 63, (1 << 53) + 1] {
        let got = unsafe { cf.call1(x as i64) } as u64;
        assert_eq!(got, (x as f64).to_bits(), "u64 {x} as f64");
    }

    // f64 -> f32 narrowing via a raw instruction.
    let mut b = FunctionBuilder::new("narrow", 1);
    let a = b.param(0);
    let dst = b.fresh();
    b.push_raw(Instr::FloatToFloat {
        dst,
        kind: FloatKind::F32,
        src: a.into(),
        src_kind: FloatKind::F64,
    });
    let f = b.finish(Terminator::Return { value: dst.into(), second: None });
    let cf = compile(&f).unwrap();
    for x in [0.1f64, 1.5, -2.75, 1e300, 1e-300] {
        let got = unsafe { cf.call1(x.to_bits() as i64) } as u64;
        assert_eq!(got, (x as f32).to_bits() as u64, "f64 {x} as f32");
    }
}
