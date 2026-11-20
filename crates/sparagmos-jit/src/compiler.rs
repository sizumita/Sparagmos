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

use crate::emitter::{Cond, Emitter, Reg, Xmm};
use crate::ir::{
    BinOp, CheckedOp, FloatBinOp, FloatKind, Function, Instr, Operand, Terminator, UnOp, Value,
};
use crate::memory::JitMemory;

/// Rust's `%` on `f64` (the exact `fmod` remainder). SSE has no
/// remainder instruction, and the exact result cannot be computed
/// from a truncated quotient without extended precision, so — like
/// the libcall a native compiler emits — the generated code calls
/// this helper, which lives in the JIT crate itself and is therefore
/// at a stable address in any process that runs generated code.
/// Arguments and result follow the System V float convention
/// (xmm0, xmm1 -> xmm0), which is exactly where the compiler's
/// float sequences keep them.
extern "C" fn float_rem_f64(a: f64, b: f64) -> f64 {
    a % b
}

/// The `f32` form of [`float_rem_f64`].
extern "C" fn float_rem_f32(a: f32, b: f32) -> f32 {
    a % b
}

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

    /// Load a floating-point operand (a slot's bit pattern, or an
    /// inline pattern for `Operand::Const`) into SSE register `x`.
    fn load_float_operand(
        &mut self,
        x: Xmm,
        kind: FloatKind,
        op: Operand,
    ) -> Result<(), CompileError> {
        match op {
            Operand::Value(v) => {
                let d = self.slot_disp(v)?;
                match kind {
                    FloatKind::F64 => self.e.movsd_load(x, Reg::Rbp, d),
                    FloatKind::F32 => self.e.movss_load(x, Reg::Rbp, d),
                }
            }
            Operand::Const(bits) => {
                self.e.mov_ri64(Reg::Rcx, bits);
                match kind {
                    FloatKind::F64 => self.e.movq_xmm(x, Reg::Rcx),
                    FloatKind::F32 => self.e.movd_xmm(x, Reg::Rcx),
                }
            }
        }
        Ok(())
    }

    /// Store the float result in xmm0 into `dst`'s slot. An `f32`
    /// result goes through a general register so the slot's upper
    /// half becomes zero — the canonical form for `f32` (a `movss`
    /// store would leave stale bits in the upper half).
    fn store_float_result(&mut self, dst: Value, kind: FloatKind) -> Result<(), CompileError> {
        let d = self.slot_disp(dst)?;
        match kind {
            FloatKind::F64 => self.e.movsd_store(Reg::Rbp, d, Xmm::Xmm0),
            FloatKind::F32 => {
                self.e.movd_gpr(Reg::Rax, Xmm::Xmm0);
                self.e.mov_store(Reg::Rbp, d, Reg::Rax);
            }
        }
        Ok(())
    }

    /// Patch every collected rel32 field to the current position.
    fn patch_all_here(&mut self, fields: &[usize]) {
        let target = self.e.position();
        for at in fields {
            self.e.buf.patch_rel32(*at, target);
        }
    }

    /// Emit the saturating `f64` -> `i64` conversion of Rust's `as`
    /// cast: the value in xmm0 becomes the result in rax.
    /// `cvttsd2si` alone truncates but yields the indefinite value
    /// (i64::MIN) for NaN and out-of-range inputs; the fixup path
    /// distinguishes those cases: NaN maps to 0, an input of at
    /// least 2^63 saturates to i64::MAX, and anything else that
    /// produced the indefinite value is a genuine i64::MIN (an input
    /// of at most -2^63, which saturates to or equals the minimum).
    fn emit_f64_to_i64(&mut self) {
        self.e.cvttsd2si(Reg::Rax, Xmm::Xmm0);
        self.e.mov_ri64(Reg::Rcx, i64::MIN);
        self.e.cmp_rr(Reg::Rax, Reg::Rcx);
        let done1 = self.e.jcc_rel32(Cond::NotEqual);
        // rax == i64::MIN: the sentinel, or a genuine minimum.
        self.e.ucomisd(Xmm::Xmm0, Xmm::Xmm0);
        let not_nan = self.e.jcc_rel32(Cond::NotParity);
        self.e.xor_rr(Reg::Rax, Reg::Rax); // NaN -> 0
        let done2 = self.e.jmp_rel32();
        let not_nan_at = self.e.position();
        self.e.buf.patch_rel32(not_nan, not_nan_at);
        self.e.mov_ri64(Reg::Rcx, 0x43E0_0000_0000_0000); // 2^63
        self.e.movq_xmm(Xmm::Xmm1, Reg::Rcx);
        self.e.ucomisd(Xmm::Xmm0, Xmm::Xmm1);
        let done3 = self.e.jcc_rel32(Cond::Below);
        self.e.mov_ri64(Reg::Rax, i64::MAX);
        self.patch_all_here(&[done1, done2, done3]);
    }

    /// Emit the saturating `f64` -> `u64` conversion of Rust's `as`
    /// cast: the value in xmm0 becomes the result in rax. Inputs of
    /// at least 2^63 take the subtract-and-set-the-top-bit path (the
    /// hardware conversion is signed); negative inputs clamp to 0
    /// and NaN maps to 0.
    fn emit_f64_to_u64(&mut self) {
        self.e.ucomisd(Xmm::Xmm0, Xmm::Xmm0);
        let not_nan = self.e.jcc_rel32(Cond::NotParity);
        self.e.xor_rr(Reg::Rax, Reg::Rax); // NaN -> 0
        let done1 = self.e.jmp_rel32();
        let not_nan_at = self.e.position();
        self.e.buf.patch_rel32(not_nan, not_nan_at);
        // At least 2^64 saturates to u64::MAX.
        self.e.mov_ri64(Reg::Rcx, 0x43F0_0000_0000_0000); // 2^64
        self.e.movq_xmm(Xmm::Xmm1, Reg::Rcx);
        self.e.ucomisd(Xmm::Xmm0, Xmm::Xmm1);
        let not_huge = self.e.jcc_rel32(Cond::Below);
        self.e.mov_ri64(Reg::Rax, -1);
        let done2 = self.e.jmp_rel32();
        let not_huge_at = self.e.position();
        self.e.buf.patch_rel32(not_huge, not_huge_at);
        // Compare against 2^63 to pick the path.
        self.e.mov_ri64(Reg::Rcx, 0x43E0_0000_0000_0000); // 2^63
        self.e.movq_xmm(Xmm::Xmm1, Reg::Rcx);
        self.e.ucomisd(Xmm::Xmm0, Xmm::Xmm1);
        let small = self.e.jcc_rel32(Cond::Below);
        // Big path: (x - 2^63) fits the signed conversion; setting
        // the top bit adds 2^63 back exactly.
        self.e.subsd(Xmm::Xmm0, Xmm::Xmm1);
        self.e.cvttsd2si(Reg::Rax, Xmm::Xmm0);
        self.e.mov_ri64(Reg::Rcx, i64::MIN);
        self.e.or_rr(Reg::Rax, Reg::Rcx);
        let done3 = self.e.jmp_rel32();
        let small_at = self.e.position();
        self.e.buf.patch_rel32(small, small_at);
        // Small path: the signed conversion is exact; a negative
        // result is a negative input, which clamps to 0.
        self.e.cvttsd2si(Reg::Rax, Xmm::Xmm0);
        self.e.test_rr(Reg::Rax, Reg::Rax);
        let done4 = self.e.jcc_rel32(Cond::NotSign);
        self.e.xor_rr(Reg::Rax, Reg::Rax);
        self.patch_all_here(&[done1, done2, done3, done4]);
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
            Instr::FloatBinOp {
                dst,
                op,
                kind,
                lhs,
                rhs,
            } => {
                use FloatBinOp as F;
                self.load_float_operand(Xmm::Xmm0, *kind, *lhs)?;
                self.load_float_operand(Xmm::Xmm1, *kind, *rhs)?;
                match op {
                    F::Add | F::Sub | F::Mul | F::Div => {
                        match (kind, op) {
                            (FloatKind::F64, F::Add) => self.e.addsd(Xmm::Xmm0, Xmm::Xmm1),
                            (FloatKind::F64, F::Sub) => self.e.subsd(Xmm::Xmm0, Xmm::Xmm1),
                            (FloatKind::F64, F::Mul) => self.e.mulsd(Xmm::Xmm0, Xmm::Xmm1),
                            (FloatKind::F64, F::Div) => self.e.divsd(Xmm::Xmm0, Xmm::Xmm1),
                            (FloatKind::F32, F::Add) => self.e.addss(Xmm::Xmm0, Xmm::Xmm1),
                            (FloatKind::F32, F::Sub) => self.e.subss(Xmm::Xmm0, Xmm::Xmm1),
                            (FloatKind::F32, F::Mul) => self.e.mulss(Xmm::Xmm0, Xmm::Xmm1),
                            (FloatKind::F32, F::Div) => self.e.divss(Xmm::Xmm0, Xmm::Xmm1),
                            _ => unreachable!(),
                        }
                        self.store_float_result(*dst, *kind)?;
                    }
                    F::Rem => {
                        // The operands are already in the System V
                        // float argument registers (xmm0, xmm1) and
                        // the helper returns into xmm0. The frame
                        // keeps rsp 16-byte aligned at every point
                        // (see the prologue), so the call is ABI-correct.
                        let helper = match kind {
                            FloatKind::F64 => float_rem_f64 as usize,
                            FloatKind::F32 => float_rem_f32 as usize,
                        };
                        self.e.mov_ri64(Reg::Rax, helper as i64);
                        self.e.call_r(Reg::Rax);
                        self.store_float_result(*dst, *kind)?;
                    }
                    F::Eq | F::Ne | F::Lt | F::Le | F::Gt | F::Ge => {
                        match kind {
                            FloatKind::F64 => self.e.ucomisd(Xmm::Xmm0, Xmm::Xmm1),
                            FloatKind::F32 => self.e.ucomiss(Xmm::Xmm0, Xmm::Xmm1),
                        }
                        // ucomi sets flags like an unsigned compare,
                        // with PF set exactly when an operand is NaN.
                        // Each Rust comparison is a setcc on those
                        // flags combined with a parity test, so NaN
                        // behaves as Rust requires (false everywhere
                        // except Ne).
                        let (cc, parity, or) = match op {
                            F::Eq => (Cond::Equal, Cond::NotParity, false),
                            F::Ne => (Cond::NotEqual, Cond::Parity, true),
                            F::Lt => (Cond::Below, Cond::NotParity, false),
                            F::Le => (Cond::BelowOrEqual, Cond::NotParity, false),
                            F::Gt => (Cond::Above, Cond::NotParity, false),
                            F::Ge => (Cond::AboveOrEqual, Cond::NotParity, false),
                            _ => unreachable!(),
                        };
                        self.e.setcc_reg(Reg::Rax, cc);
                        self.e.movzx_rr(Reg::Rax, Reg::Rax);
                        self.e.setcc_reg(Reg::Rcx, parity);
                        self.e.movzx_rr(Reg::Rcx, Reg::Rcx);
                        if or {
                            self.e.or_rr(Reg::Rax, Reg::Rcx);
                        } else {
                            self.e.and_rr(Reg::Rax, Reg::Rcx);
                        }
                        self.store_result(*dst)?;
                    }
                }
            }
            Instr::FloatToInt {
                dst,
                kind,
                src,
                int,
            } => {
                self.load_float_operand(Xmm::Xmm0, *kind, *src)?;
                // Widening f32 to f64 is exact and preserves every
                // boundary case, so one f64 path serves both formats.
                if *kind == FloatKind::F32 {
                    self.e.cvtss2sd(Xmm::Xmm0, Xmm::Xmm0);
                }
                if int.signed {
                    self.emit_f64_to_i64();
                } else {
                    self.emit_f64_to_u64();
                }
                // A narrower target saturates at its own bounds,
                // which is a clamp of the 64-bit saturated result.
                if int.bits < 64 {
                    if int.signed {
                        let max = (1i64 << (int.bits - 1)) - 1;
                        let min = -(1i64 << (int.bits - 1));
                        self.e.mov_ri64(Reg::Rcx, max);
                        self.e.cmp_rr(Reg::Rax, Reg::Rcx);
                        let ok1 = self.e.jcc_rel32(Cond::LessOrEqual);
                        self.e.mov_ri64(Reg::Rax, max);
                        let at1 = self.e.position();
                        self.e.buf.patch_rel32(ok1, at1);
                        self.e.mov_ri64(Reg::Rcx, min);
                        self.e.cmp_rr(Reg::Rax, Reg::Rcx);
                        let ok2 = self.e.jcc_rel32(Cond::GreaterOrEqual);
                        self.e.mov_ri64(Reg::Rax, min);
                        let at2 = self.e.position();
                        self.e.buf.patch_rel32(ok2, at2);
                    } else {
                        let max = ((1u64 << int.bits) - 1) as i64;
                        self.e.mov_ri64(Reg::Rcx, max);
                        self.e.cmp_rr(Reg::Rax, Reg::Rcx);
                        let ok = self.e.jcc_rel32(Cond::BelowOrEqual);
                        self.e.mov_ri64(Reg::Rax, max);
                        let at = self.e.position();
                        self.e.buf.patch_rel32(ok, at);
                    }
                }
                self.store_result(*dst)?;
            }
            Instr::IntToFloat {
                dst,
                kind,
                src,
                int,
            } => {
                self.load_operand(Reg::Rax, *src)?;
                if !int.signed && int.bits == 64 {
                    // The one case where rax's bits are not the
                    // value read as a signed integer.
                    match kind {
                        FloatKind::F64 => {
                            // Convert as signed (giving x - 2^64 for
                            // large x), then add 2^64 back when the
                            // top bit was set. Both steps are exact,
                            // so the result is the correctly rounded
                            // conversion of x.
                            self.e.cvtsi2sd(Xmm::Xmm0, Reg::Rax);
                            self.e.test_rr(Reg::Rax, Reg::Rax);
                            let done = self.e.jcc_rel32(Cond::NotSign);
                            self.e.mov_ri64(Reg::Rcx, 0x43F0_0000_0000_0000); // 2^64
                            self.e.movq_xmm(Xmm::Xmm1, Reg::Rcx);
                            self.e.addsd(Xmm::Xmm0, Xmm::Xmm1);
                            let at = self.e.position();
                            self.e.buf.patch_rel32(done, at);
                        }
                        FloatKind::F32 => {
                            // The compiler-rt algorithm: for x with
                            // the top bit set, convert (x >> 1) |
                            // (x & 1) (exact as a signed value) and
                            // double the result — a single rounding,
                            // unlike going through f64.
                            self.e.test_rr(Reg::Rax, Reg::Rax);
                            let small = self.e.jcc_rel32(Cond::NotSign);
                            self.e.mov_rr(Reg::Rcx, Reg::Rax);
                            self.e.shr_imm(Reg::Rcx, 1);
                            self.e.and_ri(Reg::Rax, 1);
                            self.e.or_rr(Reg::Rcx, Reg::Rax);
                            self.e.cvtsi2ss(Xmm::Xmm0, Reg::Rcx);
                            self.e.addss(Xmm::Xmm0, Xmm::Xmm0);
                            let done = self.e.jmp_rel32();
                            let small_at = self.e.position();
                            self.e.buf.patch_rel32(small, small_at);
                            self.e.cvtsi2ss(Xmm::Xmm0, Reg::Rax);
                            let at = self.e.position();
                            self.e.buf.patch_rel32(done, at);
                        }
                    }
                } else {
                    // Canonical form means rax already holds the
                    // exact value as an i64 (sign-extended signed
                    // types, zero-extended narrow unsigned types).
                    match kind {
                        FloatKind::F64 => self.e.cvtsi2sd(Xmm::Xmm0, Reg::Rax),
                        FloatKind::F32 => self.e.cvtsi2ss(Xmm::Xmm0, Reg::Rax),
                    }
                }
                self.store_float_result(*dst, *kind)?;
            }
            Instr::FloatToFloat {
                dst,
                kind,
                src,
                src_kind,
            } => {
                self.load_float_operand(Xmm::Xmm0, *src_kind, *src)?;
                match (src_kind, kind) {
                    (FloatKind::F32, FloatKind::F64) => {
                        self.e.cvtss2sd(Xmm::Xmm0, Xmm::Xmm0)
                    }
                    (FloatKind::F64, FloatKind::F32) => {
                        self.e.cvtsd2ss(Xmm::Xmm0, Xmm::Xmm0)
                    }
                    _ => {}
                }
                self.store_float_result(*dst, *kind)?;
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
