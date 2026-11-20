//! Hand-written x86_64 machine-code emitter.
//!
//! Every byte here is encoded directly from the Intel SDM encoding rules
//! (REX prefix, ModRM, SIB, displacement, immediate). No external code
//! generator is involved anywhere in this crate.

/// A general-purpose x86_64 register, identified by its 4-bit encoding
/// (bit 3 is the REX.R / REX.B extension bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Reg {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl Reg {
    /// Low 3 bits used inside ModRM / opcode fields.
    fn low3(self) -> u8 {
        (self as u8) & 0b111
    }
    /// Extension bit (registers r8..r15).
    fn ext(self) -> bool {
        (self as u8) & 0b1000 != 0
    }
    /// The first six integer argument registers of the System V AMD64 ABI,
    /// in order.
    pub const SYSV_ARG_REGS: [Reg; 6] = [Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9];
}

/// Condition codes for `jcc` / `setcc` (the low nibble of the opcode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Cond {
    Overflow = 0x0,
    NotOverflow = 0x1,
    Below = 0x2,        // unsigned <
    AboveOrEqual = 0x3, // unsigned >=
    Equal = 0x4,
    NotEqual = 0x5,
    BelowOrEqual = 0x6, // unsigned <=
    Above = 0x7,        // unsigned >
    Sign = 0x8,
    NotSign = 0x9,
    Parity = 0xA,
    NotParity = 0xB,
    Less = 0xC,           // signed <
    GreaterOrEqual = 0xD, // signed >=
    LessOrEqual = 0xE,    // signed <=
    Greater = 0xF,        // signed >
}

/// A growable buffer of machine code with label / fixup support.
#[derive(Debug, Default)]
pub struct CodeBuffer {
    bytes: Vec<u8>,
}

impl CodeBuffer {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
        }
    }
    /// Current offset (== length so far); usable as a label target.
    pub fn position(&self) -> usize {
        self.bytes.len()
    }
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
    pub fn into_vec(self) -> Vec<u8> {
        self.bytes
    }
    pub fn emit_u8(&mut self, b: u8) {
        self.bytes.push(b);
    }
    pub fn emit_bytes(&mut self, bs: &[u8]) {
        self.bytes.extend_from_slice(bs);
    }
    pub fn emit_u32_le(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    pub fn emit_u64_le(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    /// Overwrite the 4 bytes at `at` with a little-endian u32 (for fixups).
    pub fn patch_u32(&mut self, at: usize, v: u32) {
        self.bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    /// Patch a rel32 field located at `at` so it reaches `target`
    /// (rel32 is relative to the end of the 4-byte field).
    pub fn patch_rel32(&mut self, at: usize, target: usize) {
        let rel = target as i64 - (at as i64 + 4);
        debug_assert!(rel >= i32::MIN as i64 && rel <= i32::MAX as i64);
        self.patch_u32(at, rel as i32 as u32);
    }
}

/// The instruction emitter: wraps a [`CodeBuffer`] with typed encoders.
#[derive(Debug, Default)]
pub struct Emitter {
    pub buf: CodeBuffer,
}

impl Emitter {
    pub fn new() -> Self {
        Self {
            buf: CodeBuffer::new(),
        }
    }
    pub fn position(&self) -> usize {
        self.buf.position()
    }

    // ---- encoding primitives -------------------------------------------------

    /// REX prefix. `w` = 64-bit operand size, `r`/`x`/`b` extend the
    /// ModRM reg field, the SIB index field and the ModRM r/m (or opcode)
    /// field respectively.
    fn rex(&mut self, w: bool, r: bool, x: bool, b: bool) {
        let mut v = 0x40u8;
        if w {
            v |= 0x08;
        }
        if r {
            v |= 0x04;
        }
        if x {
            v |= 0x02;
        }
        if b {
            v |= 0x01;
        }
        self.buf.emit_u8(v);
    }
    fn rex_w_rb(&mut self, reg: Reg, rm: Reg) {
        self.rex(true, reg.ext(), false, rm.ext());
    }
    fn modrm(&mut self, mode: u8, reg: u8, rm: u8) {
        self.buf
            .emit_u8(((mode & 3) << 6) | ((reg & 7) << 3) | (rm & 7));
    }
    /// ModRM for register-direct addressing (mod = 0b11).
    fn modrm_rr(&mut self, reg: Reg, rm: Reg) {
        self.modrm(0b11, reg.low3(), rm.low3());
    }
    /// ModRM (+ SIB + displacement) for `[base + disp]` addressing.
    /// `reg_field` is the raw 3-bit reg/opcode-extension value.
    fn modrm_mem(&mut self, reg_field: u8, base: Reg, disp: i32) {
        let rm = base.low3();
        // rsp/r12 as base need a SIB byte.
        let needs_sib = rm == 0b100;
        let mode = if disp == 0 && base != Reg::Rbp && base != Reg::R13 {
            0b00
        } else if (-128..=127).contains(&disp) {
            0b01
        } else {
            0b10
        };
        self.modrm(mode, reg_field, rm);
        if needs_sib {
            // SIB: scale=1, index=none (0b100), base=rm.
            self.buf.emit_u8(0x24);
        }
        match mode {
            0b01 => self.buf.emit_u8(disp as i8 as u8),
            0b10 => self.buf.emit_u32_le(disp as u32),
            _ => {}
        }
    }

    // ---- data movement -------------------------------------------------------

    /// `mov dst, src` (64-bit, register to register): REX.W 89 /r
    pub fn mov_rr(&mut self, dst: Reg, src: Reg) {
        self.rex_w_rb(src, dst);
        self.buf.emit_u8(0x89);
        self.modrm_rr(src, dst);
    }

    /// `mov dst, imm64`: REX.W B8+rd iq
    pub fn mov_ri64(&mut self, dst: Reg, imm: i64) {
        self.rex(true, false, false, dst.ext());
        self.buf.emit_u8(0xB8 + dst.low3());
        self.buf.emit_u64_le(imm as u64);
    }

    /// `mov dst, imm32` (sign-extended to 64-bit): REX.W C7 /0 id
    pub fn mov_ri32(&mut self, dst: Reg, imm: i32) {
        self.rex(true, false, false, dst.ext());
        self.buf.emit_u8(0xC7);
        self.modrm(0b11, 0, dst.low3());
        self.buf.emit_u32_le(imm as u32);
    }

    /// Load the most compact `mov` for a constant into `dst`.
    pub fn mov_imm(&mut self, dst: Reg, imm: i64) {
        if imm >= i32::MIN as i64 && imm <= i32::MAX as i64 {
            self.mov_ri32(dst, imm as i32);
        } else {
            self.mov_ri64(dst, imm);
        }
    }

    /// `mov dst, [base + disp]`: REX.W 8B /r
    pub fn mov_load(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.rex(true, dst.ext(), false, base.ext());
        self.buf.emit_u8(0x8B);
        self.modrm_mem(dst.low3(), base, disp);
    }

    /// `mov [base + disp], src`: REX.W 89 /r
    pub fn mov_store(&mut self, base: Reg, disp: i32, src: Reg) {
        self.rex(true, src.ext(), false, base.ext());
        self.buf.emit_u8(0x89);
        self.modrm_mem(src.low3(), base, disp);
    }

    /// `lea dst, [base + disp]`: REX.W 8D /r
    pub fn lea(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.rex(true, dst.ext(), false, base.ext());
        self.buf.emit_u8(0x8D);
        self.modrm_mem(dst.low3(), base, disp);
    }

    /// `push reg`: (REX.B) 50+rd
    pub fn push(&mut self, r: Reg) {
        if r.ext() {
            self.rex(false, false, false, true);
        }
        self.buf.emit_u8(0x50 + r.low3());
    }

    /// `pop reg`: (REX.B) 58+rd
    pub fn pop(&mut self, r: Reg) {
        if r.ext() {
            self.rex(false, false, false, true);
        }
        self.buf.emit_u8(0x58 + r.low3());
    }

    // ---- arithmetic ------------------------------------------------------------

    /// Shared encoder for the reg/reg ALU family (dst op= src).
    fn alu_rr(&mut self, opcode: u8, dst: Reg, src: Reg) {
        self.rex_w_rb(src, dst);
        self.buf.emit_u8(opcode);
        self.modrm_rr(src, dst);
    }
    pub fn add_rr(&mut self, dst: Reg, src: Reg) {
        self.alu_rr(0x01, dst, src);
    }
    pub fn sub_rr(&mut self, dst: Reg, src: Reg) {
        self.alu_rr(0x29, dst, src);
    }
    pub fn and_rr(&mut self, dst: Reg, src: Reg) {
        self.alu_rr(0x21, dst, src);
    }
    pub fn or_rr(&mut self, dst: Reg, src: Reg) {
        self.alu_rr(0x09, dst, src);
    }
    pub fn xor_rr(&mut self, dst: Reg, src: Reg) {
        self.alu_rr(0x31, dst, src);
    }
    /// `cmp dst, src` (computes dst - src, sets flags only): 39 /r
    pub fn cmp_rr(&mut self, dst: Reg, src: Reg) {
        self.alu_rr(0x39, dst, src);
    }
    /// `test dst, src`: 85 /r
    pub fn test_rr(&mut self, dst: Reg, src: Reg) {
        self.alu_rr(0x85, dst, src);
    }

    /// Shared encoder for ALU reg, imm (81 /digit id, or 83 /digit ib when
    /// the immediate fits in a sign-extended byte).
    fn alu_ri(&mut self, digit: u8, dst: Reg, imm: i32) {
        self.rex(true, false, false, dst.ext());
        if (-128..=127).contains(&imm) {
            self.buf.emit_u8(0x83);
            self.modrm(0b11, digit, dst.low3());
            self.buf.emit_u8(imm as i8 as u8);
        } else {
            self.buf.emit_u8(0x81);
            self.modrm(0b11, digit, dst.low3());
            self.buf.emit_u32_le(imm as u32);
        }
    }
    pub fn add_ri(&mut self, dst: Reg, imm: i32) {
        self.alu_ri(0, dst, imm);
    }
    pub fn or_ri(&mut self, dst: Reg, imm: i32) {
        self.alu_ri(1, dst, imm);
    }
    pub fn and_ri(&mut self, dst: Reg, imm: i32) {
        self.alu_ri(4, dst, imm);
    }
    pub fn sub_ri(&mut self, dst: Reg, imm: i32) {
        self.alu_ri(5, dst, imm);
    }
    pub fn xor_ri(&mut self, dst: Reg, imm: i32) {
        self.alu_ri(6, dst, imm);
    }
    pub fn cmp_ri(&mut self, dst: Reg, imm: i32) {
        self.alu_ri(7, dst, imm);
    }

    /// `imul dst, src` (two-operand): REX.W 0F AF /r
    pub fn imul_rr(&mut self, dst: Reg, src: Reg) {
        self.rex_w_rb(dst, src);
        self.buf.emit_bytes(&[0x0F, 0xAF]);
        self.modrm_rr(dst, src);
    }
    /// `imul dst, src, imm`: REX.W 6B /r ib (or 69 /r id)
    pub fn imul_ri(&mut self, dst: Reg, src: Reg, imm: i32) {
        self.rex_w_rb(dst, src);
        if (-128..=127).contains(&imm) {
            self.buf.emit_u8(0x6B);
            self.modrm_rr(dst, src);
            self.buf.emit_u8(imm as i8 as u8);
        } else {
            self.buf.emit_u8(0x69);
            self.modrm_rr(dst, src);
            self.buf.emit_u32_le(imm as u32);
        }
    }

    /// Shift by `%cl`: D3 /digit, digit 4=shl, 5=shr (logical), 7=sar.
    fn shift_cl(&mut self, digit: u8, dst: Reg) {
        self.rex(true, false, false, dst.ext());
        self.buf.emit_u8(0xD3);
        self.modrm(0b11, digit, dst.low3());
    }
    pub fn shl_cl(&mut self, dst: Reg) {
        self.shift_cl(4, dst);
    }
    pub fn shr_cl(&mut self, dst: Reg) {
        self.shift_cl(5, dst);
    }
    pub fn sar_cl(&mut self, dst: Reg) {
        self.shift_cl(7, dst);
    }
    /// Shift by a constant 1..=63: REX.W C1 /digit ib
    fn shift_imm(&mut self, digit: u8, dst: Reg, imm: u8) {
        self.rex(true, false, false, dst.ext());
        self.buf.emit_u8(0xC1);
        self.modrm(0b11, digit, dst.low3());
        self.buf.emit_u8(imm);
    }
    pub fn shl_imm(&mut self, dst: Reg, imm: u8) {
        self.shift_imm(4, dst, imm);
    }
    pub fn shr_imm(&mut self, dst: Reg, imm: u8) {
        self.shift_imm(5, dst, imm);
    }
    pub fn sar_imm(&mut self, dst: Reg, imm: u8) {
        self.shift_imm(7, dst, imm);
    }

    /// `neg dst`: REX.W F7 /3
    pub fn neg(&mut self, dst: Reg) {
        self.rex(true, false, false, dst.ext());
        self.buf.emit_u8(0xF7);
        self.modrm(0b11, 3, dst.low3());
    }
    /// `not dst`: REX.W F7 /2
    pub fn not(&mut self, dst: Reg) {
        self.rex(true, false, false, dst.ext());
        self.buf.emit_u8(0xF7);
        self.modrm(0b11, 2, dst.low3());
    }

    /// `cqo` (sign-extend rax into rdx:rax): REX.W 99
    pub fn cqo(&mut self) {
        self.buf.emit_bytes(&[0x48, 0x99]);
    }
    /// `idiv divisor`: REX.W F7 /7 — quotient to rax, remainder to rdx.
    pub fn idiv(&mut self, divisor: Reg) {
        self.rex(true, false, false, divisor.ext());
        self.buf.emit_u8(0xF7);
        self.modrm(0b11, 7, divisor.low3());
    }
    /// `div divisor` (unsigned): REX.W F7 /6
    pub fn div(&mut self, divisor: Reg) {
        self.rex(true, false, false, divisor.ext());
        self.buf.emit_u8(0xF7);
        self.modrm(0b11, 6, divisor.low3());
    }
    /// `mul divisor` (unsigned): REX.W F7 /4 — product to rdx:rax.
    pub fn mul(&mut self, divisor: Reg) {
        self.rex(true, false, false, divisor.ext());
        self.buf.emit_u8(0xF7);
        self.modrm(0b11, 4, divisor.low3());
    }

    /// `setcc al` for the given condition: 0F 9x C0
    pub fn setcc_al(&mut self, cc: Cond) {
        self.buf.emit_bytes(&[0x0F, 0x90 + cc as u8, 0xC0]);
    }
    /// `movzx rax, al`: REX.W 0F B6 C0
    pub fn movzx_rax_al(&mut self) {
        self.buf.emit_bytes(&[0x48, 0x0F, 0xB6, 0xC0]);
    }

    // ---- control flow ------------------------------------------------------------
    // Each branch encoder returns the offset of its rel32 field so callers can
    // patch it later with `CodeBuffer::patch_rel32`.

    /// `jmp rel32`: E9 cd — returns rel32 field offset.
    pub fn jmp_rel32(&mut self) -> usize {
        self.buf.emit_u8(0xE9);
        let at = self.position();
        self.buf.emit_u32_le(0);
        at
    }
    /// `jcc rel32`: 0F 8x cd — returns rel32 field offset.
    pub fn jcc_rel32(&mut self, cc: Cond) -> usize {
        self.buf.emit_bytes(&[0x0F, 0x80 + cc as u8]);
        let at = self.position();
        self.buf.emit_u32_le(0);
        at
    }
    /// `call rel32`: E8 cd — returns rel32 field offset.
    pub fn call_rel32(&mut self) -> usize {
        self.buf.emit_u8(0xE8);
        let at = self.position();
        self.buf.emit_u32_le(0);
        at
    }
    /// `call reg` (indirect): FF /2
    pub fn call_r(&mut self, r: Reg) {
        if r.ext() {
            self.rex(false, false, false, true);
        }
        self.buf.emit_u8(0xFF);
        self.modrm(0b11, 2, r.low3());
    }
    /// `jmp reg` (indirect): FF /4
    pub fn jmp_r(&mut self, r: Reg) {
        if r.ext() {
            self.rex(false, false, false, true);
        }
        self.buf.emit_u8(0xFF);
        self.modrm(0b11, 4, r.low3());
    }
    pub fn ret(&mut self) {
        self.buf.emit_u8(0xC3);
    }
    /// `leave`: C9 (mov rsp, rbp; pop rbp)
    pub fn leave(&mut self) {
        self.buf.emit_u8(0xC9);
    }
    pub fn nop(&mut self) {
        self.buf.emit_u8(0x90);
    }
    pub fn int3(&mut self) {
        self.buf.emit_u8(0xCC);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(f: impl Fn(&mut Emitter)) -> Vec<u8> {
        let mut e = Emitter::new();
        f(&mut e);
        e.buf.into_vec()
    }

    #[test]
    fn mov_reg_reg_bytes() {
        // mov rax, rbx = 48 89 D8
        assert_eq!(enc(|e| e.mov_rr(Reg::Rax, Reg::Rbx)), [0x48, 0x89, 0xD8]);
        // mov rdi, rax = 48 89 C7
        assert_eq!(enc(|e| e.mov_rr(Reg::Rdi, Reg::Rax)), [0x48, 0x89, 0xC7]);
        // mov r8, r9 = 4D 89 C8
        assert_eq!(enc(|e| e.mov_rr(Reg::R8, Reg::R9)), [0x4D, 0x89, 0xC8]);
        // mov rbp, rsp = 48 89 E5
        assert_eq!(enc(|e| e.mov_rr(Reg::Rbp, Reg::Rsp)), [0x48, 0x89, 0xE5]);
    }

    #[test]
    fn mov_imm_bytes() {
        // mov rax, 0x1122334455667788 = 48 B8 88 77 ...
        assert_eq!(
            enc(|e| e.mov_ri64(Reg::Rax, 0x1122_3344_5566_7788)),
            [0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
        );
        // mov rcx, -1 (imm32 form) = 48 C7 C1 FF FF FF FF
        assert_eq!(
            enc(|e| e.mov_ri32(Reg::Rcx, -1)),
            [0x48, 0xC7, 0xC1, 0xFF, 0xFF, 0xFF, 0xFF]
        );
        // mov r10, 5 (imm32 form) = 49 C7 C2 05 00 00 00
        assert_eq!(
            enc(|e| e.mov_imm(Reg::R10, 5)),
            [0x49, 0xC7, 0xC2, 0x05, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn alu_bytes() {
        // add rax, rcx = 48 01 C8
        assert_eq!(enc(|e| e.add_rr(Reg::Rax, Reg::Rcx)), [0x48, 0x01, 0xC8]);
        // sub rdx, rsi = 48 29 F2
        assert_eq!(enc(|e| e.sub_rr(Reg::Rdx, Reg::Rsi)), [0x48, 0x29, 0xF2]);
        // imul rax, rcx = 48 0F AF C1
        assert_eq!(
            enc(|e| e.imul_rr(Reg::Rax, Reg::Rcx)),
            [0x48, 0x0F, 0xAF, 0xC1]
        );
        // cmp rax, rcx = 48 39 C8
        assert_eq!(enc(|e| e.cmp_rr(Reg::Rax, Reg::Rcx)), [0x48, 0x39, 0xC8]);
        // add rsp, 32 (imm8 form) = 48 83 C4 20
        assert_eq!(enc(|e| e.add_ri(Reg::Rsp, 32)), [0x48, 0x83, 0xC4, 0x20]);
        // sub rsp, 1000 (imm32 form) = 48 81 EC E8 03 00 00
        assert_eq!(
            enc(|e| e.sub_ri(Reg::Rsp, 1000)),
            [0x48, 0x81, 0xEC, 0xE8, 0x03, 0x00, 0x00]
        );
        // xor eax-style zeroing idiom at 64-bit: xor rax, rax = 48 31 C0
        assert_eq!(enc(|e| e.xor_rr(Reg::Rax, Reg::Rax)), [0x48, 0x31, 0xC0]);
    }

    #[test]
    fn stack_and_misc_bytes() {
        assert_eq!(enc(|e| e.push(Reg::Rbp)), [0x55]);
        assert_eq!(enc(|e| e.push(Reg::R12)), [0x41, 0x54]);
        assert_eq!(enc(|e| e.pop(Reg::Rbx)), [0x5B]);
        assert_eq!(enc(|e| e.ret()), [0xC3]);
        assert_eq!(enc(|e| e.leave()), [0xC9]);
        assert_eq!(enc(|e| e.cqo()), [0x48, 0x99]);
        // idiv rcx = 48 F7 F9
        assert_eq!(enc(|e| e.idiv(Reg::Rcx)), [0x48, 0xF7, 0xF9]);
        // call rax = FF D0 ; call r10 = 41 FF D2
        assert_eq!(enc(|e| e.call_r(Reg::Rax)), [0xFF, 0xD0]);
        assert_eq!(enc(|e| e.call_r(Reg::R10)), [0x41, 0xFF, 0xD2]);
    }

    #[test]
    fn mem_bytes() {
        // mov rax, [rbp-8] = 48 8B 45 F8
        assert_eq!(
            enc(|e| e.mov_load(Reg::Rax, Reg::Rbp, -8)),
            [0x48, 0x8B, 0x45, 0xF8]
        );
        // mov [rbp-16], rcx = 48 89 4D F0
        assert_eq!(
            enc(|e| e.mov_store(Reg::Rbp, -16, Reg::Rcx)),
            [0x48, 0x89, 0x4D, 0xF0]
        );
        // mov rax, [rsp] needs SIB: 48 8B 04 24
        assert_eq!(
            enc(|e| e.mov_load(Reg::Rax, Reg::Rsp, 0)),
            [0x48, 0x8B, 0x04, 0x24]
        );
        // lea rdx, [rbp+256] = 48 8D 95 00 01 00 00
        assert_eq!(
            enc(|e| e.lea(Reg::Rdx, Reg::Rbp, 256)),
            [0x48, 0x8D, 0x95, 0x00, 0x01, 0x00, 0x00]
        );
    }

    #[test]
    fn branch_patch() {
        let mut e = Emitter::new();
        let at = e.jmp_rel32();
        e.nop();
        e.nop();
        let target = e.position();
        e.ret();
        e.buf.patch_rel32(at, target);
        // jmp +2 (skips the two nops), nop, nop, ret
        assert_eq!(
            e.buf.as_slice(),
            [0xE9, 0x02, 0x00, 0x00, 0x00, 0x90, 0x90, 0xC3]
        );
    }
}
