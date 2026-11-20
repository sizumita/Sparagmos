//! Integration tests: JIT-compile IR and actually execute the generated
//! machine code, checking results against native Rust computation.

use sparagmos_jit::emitter::{Emitter, Reg};
use sparagmos_jit::ir::{BinOp as B, FunctionBuilder, Operand, Terminator};
use sparagmos_jit::{compile, Instr, JitMemory};

#[test]
fn jit_add_2_3_is_5() {
    let mut b = FunctionBuilder::new("add", 2);
    let (a, c) = (b.param(0), b.param(1));
    let r = b.binop(B::Add, a, c);
    let f = b.finish(Terminator::Return(r.into()));
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
    let f = b.finish(Terminator::Return(acc.into()));
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
    let f = b.finish(Terminator::Return(a.into()));
    let cf = compile(&f).unwrap();
    assert_eq!(unsafe { cf.call2(48, 36) }, 12);
    assert_eq!(unsafe { cf.call2(17, 5) }, 1);
    assert_eq!(unsafe { cf.call2(100, 10) }, 10);
}
