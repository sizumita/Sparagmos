//! First compiler: lowers the crate-local IR ([`crate::ir`]) to x86_64
//! machine code following the System V AMD64 calling convention.
//!
//! Strategy (deliberately simple — a baseline JIT):
//! * Every IR value gets a fixed stack slot at `[rbp - 8*(slot+1)]`.
//! * The prologue saves `rbp`, sets up the frame, and copies the first six
//!   integer arguments (rdi, rsi, rdx, rcx, r8, r9) into the parameter
//!   values' slots. More than six parameters is an error for now.
//! * Each instruction loads its operands into rax (lhs) / rcx (rhs),
//!   computes in registers, and stores the result back to its slot.
//!   Correctness first; register allocation is future work.
//! * Block terminators become jumps; branch fixups are patched once all
//!   block offsets are known.
//!
//! When Lane A's MIR types (targeting Rust 1.100.0 MIR) are fixed, the
//! intended connection is a MIR -> IR lowering feeding this same backend,
//! or a second backend sharing `emitter` / `memory` unchanged.

use crate::emitter::{Cond, Emitter, Reg};
use crate::ir::{BinOp, Function, Instr, Operand, Terminator, UnOp, Value};
use crate::memory::JitMemory;

#[derive(Debug)]
pub enum CompileError {
    TooManyParams(usize),
    UnknownBlock(u32),
    ValueOutOfRange(Value),
    Memory(std::io::Error),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::TooManyParams(n) => {
                write!(f, "too many parameters: {n} (max 6 register args for now)")
            }
            CompileError::UnknownBlock(b) => write!(f, "jump to unknown block {b}"),
            CompileError::ValueOutOfRange(v) => write!(f, "value {v} out of range"),
            CompileError::Memory(e) => write!(f, "jit memory error: {e}"),
        }
    }
}
impl std::error::Error for CompileError {}
impl From<std::io::Error> for CompileError {
    fn from(e: std::io::Error) -> Self {
        CompileError::Memory(e)
    }
}

/// A compiled function living in executable memory.
pub struct CompiledFunction {
    mem: JitMemory,
    #[allow(dead_code)]
    name: String,
}

impl CompiledFunction {
    pub fn as_ptr(&self) -> *const u8 {
        self.mem.as_ptr()
    }
    /// # Safety
    /// Same contract as [`JitMemory::function_ptr`]: `F` must match the
    /// compiled function's real signature and this value must outlive calls.
    pub unsafe fn function_ptr<F: Copy>(&self) -> F {
        unsafe { self.mem.function_ptr() }
    }
    /// Call as `fn(i64) -> i64`.
    /// # Safety
    /// The compiled function must really take one i64 and return one i64.
    pub unsafe fn call1(&self, a: i64) -> i64 {
        let f: unsafe extern "C" fn(i64) -> i64 = unsafe { self.function_ptr() };
        unsafe { f(a) }
    }
    /// Call as `fn(i64, i64) -> i64`.
    /// # Safety
    /// The compiled function must really take two i64s and return one i64.
    pub unsafe fn call2(&self, a: i64, b: i64) -> i64 {
        let f: unsafe extern "C" fn(i64, i64) -> i64 = unsafe { self.function_ptr() };
        unsafe { f(a, b) }
    }
    /// Call as `fn() -> i64`.
    /// # Safety
    /// The compiled function must really take no arguments.
    pub unsafe fn call0(&self) -> i64 {
        let f: unsafe extern "C" fn() -> i64 = unsafe { self.function_ptr() };
        unsafe { f() }
    }
}

struct Compiler<'a> {
    func: &'a Function,
    e: Emitter,
    /// rel32 fixups: (field offset, target block id)
    fixups: Vec<(usize, u32)>,
    block_offsets: Vec<usize>,
}

impl<'a> Compiler<'a> {
    fn slot_disp(&self, v: Value) -> Result<i32, CompileError> {
        if (v as usize) >= self.func.value_count {
            return Err(CompileError::ValueOutOfRange(v));
        }
        Ok(-8 * (v as i32 + 1))
    }

    /// Load an operand into register `r`. Uses rax as scratch when r is rcx
    /// and the operand is a memory value — avoided by loading via r itself
    /// directly (mov r, [rbp+disp] works for any register).
    fn load_operand(&mut self, r: Reg, op: Operand) -> Result<(), CompileError> {
        match op {
            Operand::Const(imm) => self.e.mov_imm(r, imm),
            Operand::Value(v) => {
                let d = self.slot_disp(v)?;
                self.e.mov_load(r, Reg::Rbp, d);
            }
        }
        Ok(())
    }

    fn store_result(&mut self, dst: Value) -> Result<(), CompileError> {
        let d = self.slot_disp(dst)?;
        self.e.mov_store(Reg::Rbp, d, Reg::Rax);
        Ok(())
    }

    fn compile_instr(&mut self, instr: &Instr) -> Result<(), CompileError> {
        match *instr {
            Instr::Const { dst, imm } => {
                self.e.mov_imm(Reg::Rax, imm);
                self.store_result(dst)?;
            }
            Instr::Copy { dst, src } => {
                self.load_operand(Reg::Rax, src)?;
                self.store_result(dst)?;
            }
            Instr::UnOp { dst, op, src } => {
                self.load_operand(Reg::Rax, src)?;
                match op {
                    UnOp::Neg => self.e.neg(Reg::Rax),
                    UnOp::Not => self.e.not(Reg::Rax),
                }
                self.store_result(dst)?;
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                self.load_operand(Reg::Rax, lhs)?;
                self.load_operand(Reg::Rcx, rhs)?;
                match op {
                    BinOp::Add => self.e.add_rr(Reg::Rax, Reg::Rcx),
                    BinOp::Sub => self.e.sub_rr(Reg::Rax, Reg::Rcx),
                    BinOp::Mul => self.e.imul_rr(Reg::Rax, Reg::Rcx),
                    BinOp::Div => {
                        self.e.cqo();
                        self.e.idiv(Reg::Rcx);
                    }
                    BinOp::Rem => {
                        self.e.cqo();
                        self.e.idiv(Reg::Rcx);
                        self.e.mov_rr(Reg::Rax, Reg::Rdx);
                    }
                    BinOp::And => self.e.and_rr(Reg::Rax, Reg::Rcx),
                    BinOp::Or => self.e.or_rr(Reg::Rax, Reg::Rcx),
                    BinOp::Xor => self.e.xor_rr(Reg::Rax, Reg::Rcx),
                    BinOp::Shl => self.e.shl_cl(Reg::Rax),
                    BinOp::Shr => self.e.shr_cl(Reg::Rax),
                    BinOp::Sar => self.e.sar_cl(Reg::Rax),
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        let cc = match op {
                            BinOp::Eq => Cond::Equal,
                            BinOp::Ne => Cond::NotEqual,
                            BinOp::Lt => Cond::Less,
                            BinOp::Le => Cond::LessOrEqual,
                            BinOp::Gt => Cond::Greater,
                            _ => Cond::GreaterOrEqual,
                        };
                        self.e.cmp_rr(Reg::Rax, Reg::Rcx);
                        self.e.setcc_al(cc);
                        self.e.movzx_rax_al();
                    }
                }
                self.store_result(dst)?;
            }
        }
        Ok(())
    }

    fn compile_terminator(&mut self, term: &Terminator) -> Result<(), CompileError> {
        match *term {
            Terminator::Return(op) => {
                self.load_operand(Reg::Rax, op)?;
                self.e.leave();
                self.e.ret();
            }
            Terminator::Jump(target) => {
                if (target as usize) >= self.func.blocks.len() {
                    return Err(CompileError::UnknownBlock(target));
                }
                let at = self.e.jmp_rel32();
                self.fixups.push((at, target));
            }
            Terminator::Branch {
                cond,
                then_blk,
                else_blk,
            } => {
                for b in [then_blk, else_blk] {
                    if (b as usize) >= self.func.blocks.len() {
                        return Err(CompileError::UnknownBlock(b));
                    }
                }
                self.load_operand(Reg::Rax, cond)?;
                self.e.test_rr(Reg::Rax, Reg::Rax);
                let at = self.e.jcc_rel32(Cond::NotEqual);
                self.fixups.push((at, then_blk));
                let at2 = self.e.jmp_rel32();
                self.fixups.push((at2, else_blk));
            }
            Terminator::Unreachable => {
                self.e.int3();
            }
        }
        Ok(())
    }

    fn compile(mut self) -> Result<Vec<u8>, CompileError> {
        let func = self.func;
        if func.param_count > 6 {
            return Err(CompileError::TooManyParams(func.param_count));
        }
        // Prologue: push rbp; mov rbp, rsp; sub rsp, frame (16-byte aligned
        // frame size keeps rsp 16-aligned after the saved rbp + return addr
        // accounting: slots rounded up to an even count).
        let slots = func.value_count.max(1);
        let frame = ((slots * 8) + 15) & !15;
        self.e.push(Reg::Rbp);
        self.e.mov_rr(Reg::Rbp, Reg::Rsp);
        if frame > 0 {
            self.e.sub_ri(Reg::Rsp, frame as i32);
        }
        // Copy register arguments into their values' stack slots.
        for i in 0..func.param_count {
            let d = self.slot_disp(i as Value)?;
            self.e.mov_store(Reg::Rbp, d, Reg::SYSV_ARG_REGS[i]);
        }

        for (idx, block) in func.blocks.iter().enumerate() {
            self.block_offsets[idx] = self.e.position();
            for instr in &block.instrs {
                self.compile_instr(instr)?;
            }
            self.compile_terminator(&block.terminator)?;
        }
        // Patch branch fixups now that every block offset is known.
        let offsets = self.block_offsets.clone();
        for (at, target) in &self.fixups {
            let t = offsets[*target as usize];
            self.e.buf.patch_rel32(*at, t);
        }
        Ok(self.e.buf.into_vec())
    }
}

/// Compile `func` to raw machine-code bytes (not yet executable).
pub fn compile_to_bytes(func: &Function) -> Result<Vec<u8>, CompileError> {
    let c = Compiler {
        func,
        e: Emitter::new(),
        fixups: Vec::new(),
        block_offsets: vec![0; func.blocks.len()],
    };
    c.compile()
}

/// Compile `func` into executable memory, ready to call.
pub fn compile(func: &Function) -> Result<CompiledFunction, CompileError> {
    let bytes = compile_to_bytes(func)?;
    let mut mem = JitMemory::new(bytes.len())?;
    mem.write_code(&bytes)?;
    mem.make_executable()?;
    Ok(CompiledFunction {
        mem,
        name: func.name.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp as B, FunctionBuilder, Terminator};

    #[test]
    fn add_function() {
        // fn add(a, b) = a + b
        let mut b = FunctionBuilder::new("add", 2);
        let (a, c) = (b.param(0), b.param(1));
        let r = b.binop(B::Add, a, c);
        let f = b.finish(Terminator::Return(r.into()));
        let cf = compile(&f).unwrap();
        assert_eq!(unsafe { cf.call2(2, 3) }, 5);
        assert_eq!(unsafe { cf.call2(-10, 4) }, -6);
        assert_eq!(unsafe { cf.call2(1_000_000, 2_000_000) }, 3_000_000);
    }

    #[test]
    fn arithmetic_ops() {
        // fn f(a,b) = (a * b) - (a / b) + (a % b)   with b != 0
        let mut b = FunctionBuilder::new("f", 2);
        let (a, c) = (b.param(0), b.param(1));
        let m = b.binop(B::Mul, a, c);
        let d = b.binop(B::Div, a, c);
        let r = b.binop(B::Rem, a, c);
        let t = b.binop(B::Sub, m, d);
        let out = b.binop(B::Add, t, r);
        let f = b.finish(Terminator::Return(out.into()));
        let cf = compile(&f).unwrap();
        // (17*5) - (17/5) + (17%5) = 85 - 3 + 2 = 84
        assert_eq!(unsafe { cf.call2(17, 5) }, 84);
        // (-7*3) - (-7/3) + (-7%3) = -21 - (-2) + (-1) = -20
        assert_eq!(unsafe { cf.call2(-7, 3) }, -20);
    }

    #[test]
    fn comparisons() {
        // fn lt(a,b) = a < b  (as 0/1)
        let mut b = FunctionBuilder::new("lt", 2);
        let (a, c) = (b.param(0), b.param(1));
        let r = b.binop(B::Lt, a, c);
        let f = b.finish(Terminator::Return(r.into()));
        let cf = compile(&f).unwrap();
        assert_eq!(unsafe { cf.call2(3, 4) }, 1);
        assert_eq!(unsafe { cf.call2(4, 3) }, 0);
        assert_eq!(unsafe { cf.call2(-1, 0) }, 1);
    }

    #[test]
    fn sum_loop() {
        // fn sum(n): s=0; i=1; loop: if i<=n { s+=i; i+=1; goto loop } return s
        let mut b = FunctionBuilder::new("sum", 1);
        let n = b.param(0);
        let s0 = b.const_(0);
        let i0 = b.const_(1);
        // We need mutable loop variables; with SSA values each update defines
        // a new value, and block structure carries them explicitly.
        // block0: goto block1 with (s0, i0) — modelled by fixing slots:
        // Instead use dedicated values that we overwrite via Copy/BinOp into
        // the *same* destination ids (allowed: dst is just a slot).
        // s = value id of s0, i = value id of i0.
        let (s, i) = (s0, i0);
        let loop_blk = 1;
        b.end_block(Terminator::Jump(loop_blk));
        // block1 (loop head): cond = i <= n ; branch body / exit
        // Note: instructions for block1 are emitted after end_block.
        let cond = b.binop(B::Le, i, n);
        let body_blk = 2;
        let exit_blk = 3;
        b.end_block(Terminator::Branch {
            cond: cond.into(),
            then_blk: body_blk,
            else_blk: exit_blk,
        });
        // block2 (body): s = s + i ; i = i + 1 ; goto loop
        // Emit raw instrs writing back into the same dst slots s and i.
        // FunctionBuilder::binop allocates fresh dsts, so push manually.
        b.push_raw(Instr::BinOp {
            dst: s,
            op: B::Add,
            lhs: s.into(),
            rhs: i.into(),
        });
        b.push_raw(Instr::BinOp {
            dst: i,
            op: B::Add,
            lhs: i.into(),
            rhs: Operand::Const(1),
        });
        b.end_block(Terminator::Jump(loop_blk));
        // block3 (exit): return s
        let f = b.finish(Terminator::Return(s.into()));
        let cf = compile(&f).unwrap();
        assert_eq!(unsafe { cf.call1(10) }, 55);
        assert_eq!(unsafe { cf.call1(100) }, 5050);
        assert_eq!(unsafe { cf.call1(0) }, 0);
    }

    #[test]
    fn fibonacci_iterative() {
        // fn fib(n): a=0,b=1,i=0; loop if i<n { t=a+b; a=b; b=t; i+=1 } return a
        let mut b = FunctionBuilder::new("fib", 1);
        let n = b.param(0);
        let a = b.const_(0);
        let bb = b.const_(1);
        let i = b.const_(0);
        let t = b.fresh();
        let loop_blk = 1;
        b.end_block(Terminator::Jump(loop_blk));
        let cond = b.binop(B::Lt, i, n);
        b.end_block(Terminator::Branch {
            cond: cond.into(),
            then_blk: 2,
            else_blk: 3,
        });
        b.push_raw(Instr::BinOp {
            dst: t,
            op: B::Add,
            lhs: a.into(),
            rhs: bb.into(),
        });
        b.push_raw(Instr::Copy {
            dst: a,
            src: bb.into(),
        });
        b.push_raw(Instr::Copy {
            dst: bb,
            src: t.into(),
        });
        b.push_raw(Instr::BinOp {
            dst: i,
            op: B::Add,
            lhs: i.into(),
            rhs: Operand::Const(1),
        });
        b.end_block(Terminator::Jump(loop_blk));
        let f = b.finish(Terminator::Return(a.into()));
        let cf = compile(&f).unwrap();
        assert_eq!(unsafe { cf.call1(0) }, 0);
        assert_eq!(unsafe { cf.call1(1) }, 1);
        assert_eq!(unsafe { cf.call1(10) }, 55);
        assert_eq!(unsafe { cf.call1(15) }, 610);
        assert_eq!(unsafe { cf.call1(20) }, 6765);
    }
}
