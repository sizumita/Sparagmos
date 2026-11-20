//! Baseline compiler: lowers the backend IR ([`crate::ir`]) to x86_64
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
//! Two entry points share this code generator:
//! * [`compile`] / [`compile_to_bytes`] compile a single self-contained
//!   function. The program-only instructions ([`Instr::Call`],
//!   [`Instr::Panic`], [`Instr::CheckPanic`]) are rejected there.
//! * [`crate::program::compile_program`] compiles a whole table of
//!   functions into one executable image, supplying a [`ProgramCtx`]
//!   (panic flag address, panic runtime entry, message table) so calls
//!   between the functions and guest panics work.

use crate::emitter::{Cond, Emitter, Reg};
use crate::ir::{BinOp, CheckedOp, Function, Instr, Operand, Terminator, UnOp, Value};
use crate::memory::JitMemory;

#[derive(Debug)]
pub enum CompileError {
    TooManyParams(usize),
    TooManyArgs(usize),
    UnknownBlock(u32),
    /// A call fixup referenced a function outside the program table.
    UnknownFunction(u32),
    ValueOutOfRange(Value),
    /// A program-only instruction was used in a single-function compile.
    ProgramOnly(&'static str),
    /// A panic instruction referenced a message outside the table.
    BadMessageIndex(u32),
    Memory(std::io::Error),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::TooManyParams(n) => {
                write!(f, "too many parameters: {n} (max 6 register args for now)")
            }
            CompileError::TooManyArgs(n) => {
                write!(f, "too many call arguments: {n} (max 6 register args for now)")
            }
            CompileError::UnknownBlock(b) => write!(f, "jump to unknown block {b}"),
            CompileError::UnknownFunction(i) => write!(f, "call to unknown function index {i}"),
            CompileError::ValueOutOfRange(v) => write!(f, "value {v} out of range"),
            CompileError::ProgramOnly(what) => {
                write!(f, "{what} is only available in a program compile")
            }
            CompileError::BadMessageIndex(i) => write!(f, "panic message index {i} out of range"),
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

/// Context a program compile hands to the code generator: where the
/// program's panic state lives and how to reach the runtime's panic
/// entry point. The addresses are embedded into the generated code as
/// immediates, so everything they point at must outlive the compiled
/// program (the owner is [`crate::program::CompiledProgram`]).
#[derive(Debug, Clone)]
pub struct ProgramCtx {
    /// Address of the program's panic context: two `u64` words,
    /// `[panicked: u64, message_index: u64]`.
    pub flag_addr: usize,
    /// Address of `sparagmos_panic` (the runtime's panic entry point).
    pub panic_fn: usize,
    /// Panic message table: `(ptr, len)` of each message's UTF-8 bytes.
    pub messages: Vec<(usize, usize)>,
    /// Address of the program layer's shift-message formatter:
    /// `extern "C" fn(amount: u64, bits: u64, buf: *mut u8) -> usize`,
    /// writing a length-prefixed message into `shift_buf` and
    /// returning the message length.
    pub shift_fmt_fn: usize,
    /// Address of the scratch buffer the formatter writes into:
    /// 8 length bytes (little-endian u64) followed by message bytes.
    pub shift_buf: usize,
}

/// Call fixups produced when compiling one function of a program:
/// `(rel32 field offset, callee function index)` pairs.
pub(crate) type CallFixups = Vec<(usize, u32)>;

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
    ctx: Option<&'a ProgramCtx>,
    /// rel32 fixups: (field offset, target block id)
    fixups: Vec<(usize, u32)>,
    /// call fixups: (field offset in this function's code, callee index)
    call_fixups: Vec<(usize, u32)>,
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

    /// Store register `r` into value `dst`'s slot.
    fn store_reg(&mut self, dst: Value, r: Reg) -> Result<(), CompileError> {
        let d = self.slot_disp(dst)?;
        self.e.mov_store(Reg::Rbp, d, r);
        Ok(())
    }

    fn store_result(&mut self, dst: Value) -> Result<(), CompileError> {
        self.store_reg(dst, Reg::Rax)
    }

    fn compile_instr(&mut self, instr: &Instr) -> Result<(), CompileError> {
        match instr {
            Instr::Const { dst, imm } => {
                self.e.mov_imm(Reg::Rax, *imm);
                self.store_result(*dst)?;
            }
            Instr::Copy { dst, src } => {
                self.load_operand(Reg::Rax, *src)?;
                self.store_result(*dst)?;
            }
            Instr::UnOp { dst, op, src } => {
                self.load_operand(Reg::Rax, *src)?;
                match op {
                    UnOp::Neg => self.e.neg(Reg::Rax),
                    UnOp::Not => self.e.not(Reg::Rax),
                }
                self.store_result(*dst)?;
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                self.load_operand(Reg::Rax, *lhs)?;
                self.load_operand(Reg::Rcx, *rhs)?;
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
                    BinOp::DivU => {
                        // div divides rdx:rax; the dividend here is the
                        // zero-extended operand in rax, so rdx must be 0.
                        self.e.xor_rr(Reg::Rdx, Reg::Rdx);
                        self.e.div(Reg::Rcx);
                    }
                    BinOp::RemU => {
                        self.e.xor_rr(Reg::Rdx, Reg::Rdx);
                        self.e.div(Reg::Rcx);
                        self.e.mov_rr(Reg::Rax, Reg::Rdx);
                    }
                    BinOp::And => self.e.and_rr(Reg::Rax, Reg::Rcx),
                    BinOp::Or => self.e.or_rr(Reg::Rax, Reg::Rcx),
                    BinOp::Xor => self.e.xor_rr(Reg::Rax, Reg::Rcx),
                    BinOp::Shl => self.e.shl_cl(Reg::Rax),
                    BinOp::Shr => self.e.shr_cl(Reg::Rax),
                    BinOp::Sar => self.e.sar_cl(Reg::Rax),
                    BinOp::Eq
                    | BinOp::Ne
                    | BinOp::Lt
                    | BinOp::Le
                    | BinOp::Gt
                    | BinOp::Ge
                    | BinOp::LtU
                    | BinOp::LeU
                    | BinOp::GtU
                    | BinOp::GeU => {
                        let cc = match op {
                            BinOp::Eq => Cond::Equal,
                            BinOp::Ne => Cond::NotEqual,
                            BinOp::Lt => Cond::Less,
                            BinOp::Le => Cond::LessOrEqual,
                            BinOp::Gt => Cond::Greater,
                            BinOp::Ge => Cond::GreaterOrEqual,
                            BinOp::LtU => Cond::Below,
                            BinOp::LeU => Cond::BelowOrEqual,
                            BinOp::GtU => Cond::Above,
                            _ => Cond::AboveOrEqual,
                        };
                        self.e.cmp_rr(Reg::Rax, Reg::Rcx);
                        self.e.setcc_al(cc);
                        self.e.movzx_rax_al();
                    }
                }
                self.store_result(*dst)?;
            }
            Instr::CheckedBinOp {
                dst_val,
                dst_flag,
                op,
                lhs,
                rhs,
                kind,
            } => {
                self.load_operand(Reg::Rax, *lhs)?;
                self.load_operand(Reg::Rcx, *rhs)?;
                let bits = kind.bits as u32;
                if bits == 64 && !kind.signed && matches!(op, CheckedOp::Mul) {
                    // Unsigned 64x64: the low half alone cannot reveal
                    // overflow, so use the widening multiply and test
                    // the high half in rdx afterwards.
                    self.e.mul(Reg::Rcx);
                    self.store_result(*dst_val)?;
                    self.e.cmp_ri(Reg::Rdx, 0);
                    self.e.setcc_al(Cond::NotEqual);
                    self.e.movzx_rax_al();
                    self.store_result(*dst_flag)?;
                } else if bits == 64 {
                    match op {
                        CheckedOp::Add => self.e.add_rr(Reg::Rax, Reg::Rcx),
                        CheckedOp::Sub => self.e.sub_rr(Reg::Rax, Reg::Rcx),
                        CheckedOp::Mul => self.e.imul_rr(Reg::Rax, Reg::Rcx),
                    }
                    // The wrapped result is final at this width; the
                    // flags from the operation carry the overflow bit
                    // (OF for signed, CF for unsigned add/sub). Storing
                    // the result first is safe: `mov` does not touch
                    // flags, and setcc only rewrites al afterwards.
                    self.store_result(*dst_val)?;
                    let cc = if kind.signed {
                        Cond::Overflow
                    } else {
                        Cond::Below
                    };
                    self.e.setcc_al(cc);
                    self.e.movzx_rax_al();
                    self.store_result(*dst_flag)?;
                } else {
                    // Narrower than 64 bits: the operands are canonical
                    // (sign-/zero-extended), so the 64-bit operation is
                    // exact. Overflow is "normalising the exact result
                    // to the type's width changes it".
                    match op {
                        CheckedOp::Add => self.e.add_rr(Reg::Rax, Reg::Rcx),
                        CheckedOp::Sub => self.e.sub_rr(Reg::Rax, Reg::Rcx),
                        CheckedOp::Mul => self.e.imul_rr(Reg::Rax, Reg::Rcx),
                    }
                    self.e.mov_rr(Reg::Rcx, Reg::Rax);
                    if kind.signed {
                        let shift = 64 - bits;
                        self.e.shl_imm(Reg::Rcx, shift as u8);
                        self.e.sar_imm(Reg::Rcx, shift as u8);
                    } else {
                        let mask = ((1u128 << bits) - 1) as i64;
                        self.e.mov_ri64(Reg::Rdx, mask);
                        self.e.and_rr(Reg::Rcx, Reg::Rdx);
                    }
                    self.e.cmp_rr(Reg::Rax, Reg::Rcx);
                    self.e.setcc_al(Cond::NotEqual);
                    self.e.movzx_rax_al();
                    // rax now holds the flag, rcx the wrapped value.
                    self.store_reg(*dst_val, Reg::Rcx)?;
                    self.store_result(*dst_flag)?;
                }
            }
            Instr::Call { dst, func, args } => {
                if self.ctx.is_none() {
                    return Err(CompileError::ProgramOnly("Call"));
                }
                if args.len() > 6 {
                    return Err(CompileError::TooManyArgs(args.len()));
                }
                for (i, a) in args.iter().enumerate() {
                    self.load_operand(Reg::SYSV_ARG_REGS[i], *a)?;
                }
                let at = self.e.call_rel32();
                self.call_fixups.push((at, *func));
                self.store_result(*dst)?;
            }
            Instr::Panic { msg } => {
                let ctx = self
                    .ctx
                    .ok_or(CompileError::ProgramOnly("Panic"))?;
                let (ptr, len) = ctx
                    .messages
                    .get(*msg as usize)
                    .copied()
                    .ok_or(CompileError::BadMessageIndex(*msg))?;
                // sparagmos_panic(msg_ptr, msg_len): records the panic
                // in the runtime. Then record it in the program's own
                // panic context (flag + message index) and return from
                // this function; callers propagate via CheckPanic.
                self.e.mov_ri64(Reg::Rdi, ptr as i64);
                self.e.mov_ri64(Reg::Rsi, len as i64);
                self.e.mov_ri64(Reg::Rax, ctx.panic_fn as i64);
                self.e.call_r(Reg::Rax);
                self.e.mov_ri64(Reg::Rax, ctx.flag_addr as i64);
                self.e.mov_ri64(Reg::Rcx, 1);
                self.e.mov_store(Reg::Rax, 0, Reg::Rcx);
                self.e.mov_ri64(Reg::Rcx, *msg as i64);
                self.e.mov_store(Reg::Rax, 8, Reg::Rcx);
                self.e.leave();
                self.e.ret();
            }
            Instr::CheckPanic => {
                let ctx = self
                    .ctx
                    .ok_or(CompileError::ProgramOnly("CheckPanic"))?;
                self.e.mov_ri64(Reg::Rax, ctx.flag_addr as i64);
                self.e.mov_load(Reg::Rcx, Reg::Rax, 0);
                self.e.test_rr(Reg::Rcx, Reg::Rcx);
                let at = self.e.jcc_rel32(Cond::Equal);
                self.e.leave();
                self.e.ret();
                let target = self.e.position();
                self.e.buf.patch_rel32(at, target);
            }
            Instr::PanicShiftOverflow { amount, bits } => {
                let ctx = self
                    .ctx
                    .ok_or(CompileError::ProgramOnly("PanicShiftOverflow"))?;
                // Format the message into the program's scratch
                // buffer, then record it through the runtime's panic
                // path like a static message. The message index is
                // the u64::MAX sentinel: the program layer reads the
                // text back from the buffer instead of the table.
                self.load_operand(Reg::Rdi, *amount)?;
                self.e.mov_ri64(Reg::Rsi, *bits as i64);
                self.e.mov_ri64(Reg::Rdx, ctx.shift_buf as i64);
                self.e.mov_ri64(Reg::Rax, ctx.shift_fmt_fn as i64);
                self.e.call_r(Reg::Rax);
                self.e.mov_rr(Reg::Rsi, Reg::Rax);
                self.e.mov_ri64(Reg::Rdi, (ctx.shift_buf + 8) as i64);
                self.e.mov_ri64(Reg::Rax, ctx.panic_fn as i64);
                self.e.call_r(Reg::Rax);
                self.e.mov_ri64(Reg::Rax, ctx.flag_addr as i64);
                self.e.mov_ri64(Reg::Rcx, 1);
                self.e.mov_store(Reg::Rax, 0, Reg::Rcx);
                self.e.mov_ri64(Reg::Rcx, -1);
                self.e.mov_store(Reg::Rax, 8, Reg::Rcx);
                self.e.leave();
                self.e.ret();
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

    fn compile(mut self) -> Result<(Vec<u8>, CallFixups), CompileError> {
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
        Ok((self.e.buf.into_vec(), self.call_fixups))
    }
}

/// Compile `func` to raw machine-code bytes (not yet executable).
/// Program-only instructions are rejected with
/// [`CompileError::ProgramOnly`].
pub fn compile_to_bytes(func: &Function) -> Result<Vec<u8>, CompileError> {
    let c = Compiler {
        func,
        e: Emitter::new(),
        ctx: None,
        fixups: Vec::new(),
        call_fixups: Vec::new(),
        block_offsets: vec![0; func.blocks.len()],
    };
    let (bytes, call_fixups) = c.compile()?;
    debug_assert!(call_fixups.is_empty());
    Ok(bytes)
}

/// Compile one function of a program with the program context.
/// Returns the code bytes and the call fixups within them:
/// `(rel32 field offset, callee function index)`.
pub(crate) fn compile_with_ctx(
    func: &Function,
    ctx: &ProgramCtx,
) -> Result<(Vec<u8>, CallFixups), CompileError> {
    let c = Compiler {
        func,
        e: Emitter::new(),
        ctx: Some(ctx),
        fixups: Vec::new(),
        call_fixups: Vec::new(),
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
