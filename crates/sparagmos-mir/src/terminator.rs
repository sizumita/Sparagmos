//! `Terminator` / `TerminatorKind`, unwind and call support types (rustc MIR, 1.100.0).
use crate::operand::Operand;
use crate::place::Place;
use crate::rvalue::BinOp;
use crate::statement::SourceInfo;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BasicBlock(pub u32);
impl BasicBlock {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
impl From<u32> for BasicBlock {
    fn from(v: u32) -> Self {
        Self(v)
    }
}
impl From<usize> for BasicBlock {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchTargets {
    pub values: Vec<u128>,
    pub targets: Vec<BasicBlock>,
    pub otherwise: BasicBlock,
}
impl SwitchTargets {
    pub fn new(values: Vec<u128>, targets: Vec<BasicBlock>, otherwise: BasicBlock) -> Self {
        assert_eq!(
            values.len(),
            targets.len(),
            "SwitchTargets values/targets length mismatch"
        );
        Self {
            values,
            targets,
            otherwise,
        }
    }
    pub fn all_targets(&self) -> Vec<BasicBlock> {
        let mut v = self.targets.clone();
        v.push(self.otherwise);
        v
    }
    pub fn target_for_value(&self, value: u128) -> BasicBlock {
        self.values
            .iter()
            .position(|&v| v == value)
            .map(|i| self.targets[i])
            .unwrap_or(self.otherwise)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnwindTerminateReason {
    Generic,
    InCleanup,
}
#[derive(Clone, Debug, PartialEq)]
pub enum UnwindAction {
    Continue,
    Unreachable,
    Terminate(UnwindTerminateReason),
    Cleanup(BasicBlock),
}
impl UnwindAction {
    pub fn cleanup_block(&self) -> Option<BasicBlock> {
        match self {
            Self::Cleanup(b) => Some(*b),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TerminateKind {
    Resume,
    Abort,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallSource {
    Normal,
    OverloadedOperator,
    DivergingOperator,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoroutineKind {
    Async,
    Gen,
    Coroutine,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssertMessage {
    BoundsCheck {
        len: Box<Operand>,
        index: Box<Operand>,
    },
    Overflow(BinOp, Box<Operand>, Box<Operand>),
    OverflowNeg(Box<Operand>),
    DivisionByZero(Box<Operand>),
    RemainderByZero(Box<Operand>),
    ResumedAfterReturn(CoroutineKind),
    ResumedAfterPanic(CoroutineKind),
    ResumedAfterDrop(CoroutineKind),
    MisalignedPointerDereference {
        required: Box<Operand>,
        found: Box<Operand>,
    },
    NullPointerDereference,
    InvalidEnumConstruction(Box<Operand>),
    Custom(String),
}
impl AssertMessage {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::BoundsCheck { .. } => "BoundsCheck",
            Self::Overflow(..) => "Overflow",
            Self::OverflowNeg(_) => "OverflowNeg",
            Self::DivisionByZero(_) => "DivisionByZero",
            Self::RemainderByZero(_) => "RemainderByZero",
            Self::ResumedAfterReturn(_) => "ResumedAfterReturn",
            Self::ResumedAfterPanic(_) => "ResumedAfterPanic",
            Self::ResumedAfterDrop(_) => "ResumedAfterDrop",
            Self::MisalignedPointerDereference { .. } => "MisalignedPointerDereference",
            Self::NullPointerDereference => "NullPointerDereference",
            Self::InvalidEnumConstruction(_) => "InvalidEnumConstruction",
            Self::Custom(_) => "Custom",
        }
    }
}

/// The role of one operand of an [`TerminatorKind::InlineAsm`], as
/// recovered by the driver from Stable MIR (whose public operand
/// struct keeps only the value/place pair; the kind and the
/// register come from its raw representation string).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsmOperandKind {
    In,
    Out,
    InOut,
    Const,
    Sym,
}

/// One operand of an [`TerminatorKind::InlineAsm`], with the facts
/// the executors need about it.
#[derive(Clone, Debug, PartialEq)]
pub struct InlineAsmOperand {
    pub kind: AsmOperandKind,
    /// True when the operand is bound to a register *class*
    /// (`reg`, `reg_byte`, ...); false for an explicit register
    /// (`"eax"`). Only class operands are ever classified as
    /// evaluable: an explicit register could alias another operand
    /// or a register the template names literally, and the virtual
    /// register file does not model either.
    pub reg_class: bool,
    /// The width in bits of the operand's type, as determined by
    /// the driver; 0 when the type is not a plain integer, `bool`
    /// or `char` type. Such an operand keeps the whole terminator
    /// unsupported unless the template has no real instructions.
    pub width: u32,
    /// For [`AsmOperandKind::Const`]: the constant's bits, masked
    /// to `width`. `None` for every other kind, and for constants
    /// the driver could not read as an integer literal.
    pub const_bits: Option<u64>,
    pub in_value: Option<Operand>,
    pub out_place: Option<Place>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TerminatorKind {
    Goto {
        target: BasicBlock,
    },
    SwitchInt {
        discr: Operand,
        targets: SwitchTargets,
    },
    Resume,
    Terminate(TerminateKind),
    Return,
    Unreachable,
    Drop {
        place: Place,
        target: BasicBlock,
        unwind: UnwindAction,
        replace: bool,
    },
    Call {
        func: Operand,
        args: Vec<Operand>,
        destination: Place,
        target: Option<BasicBlock>,
        unwind: UnwindAction,
        call_source: CallSource,
    },
    Assert {
        cond: Operand,
        expected: bool,
        msg: Box<AssertMessage>,
        target: BasicBlock,
        unwind: UnwindAction,
    },
    Yield {
        value: Operand,
        resume: BasicBlock,
        resume_arg: Place,
        drop: Option<BasicBlock>,
    },
    CoroutineDrop,
    FalseEdge {
        real_target: BasicBlock,
        imaginary_target: BasicBlock,
    },
    FalseUnwind {
        real_target: BasicBlock,
        unwind: UnwindAction,
    },
    InlineAsm {
        template: String,
        operands: Vec<InlineAsmOperand>,
        options: String,
        destination: Option<BasicBlock>,
        unwind: UnwindAction,
    },
    TailCall {
        func: Operand,
        args: Vec<Operand>,
    },
}
/// Whether an `InlineAsm` terminator is a **no-op**: executing it
/// has no observable effect, so an executor may step over it.
///
/// The template, with block comments (`/* ... */`), line comments
/// (`// ...`) and operand placeholders (`{...}`) removed, must consist
/// only of empty pieces and `nop` instructions (separated by `;` or
/// newlines), and the options must not contain `NORETURN` (a
/// noreturn block never reaches its destination, so stepping over it
/// would invent a continuation the program does not have). Anything
/// else — real instructions, port I/O, fences, `cpuid` — has effects
/// this predicate does not model, and the caller keeps it an explicit
/// error. Whether the block may be *converted* at all additionally
/// depends on its operands (an operand with an out place is written
/// by the assembly, which no executor here models); that check is
/// the driver's, since only it sees the operand places.
pub fn inline_asm_is_noop(template: &str, options: &str) -> bool {
    if options.contains("NORETURN") {
        return false;
    }
    let template: &str = &inline_asm_template_text(template);
    let mut stripped = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("/*") {
        stripped.push_str(&rest[..start]);
        match rest[start + 2..].find("*/") {
            Some(end) => rest = &rest[start + 2 + end + 2..],
            None => {
                rest = "";
                break;
            }
        }
    }
    stripped.push_str(rest);
    let mut body = String::with_capacity(stripped.len());
    for line in stripped.lines() {
        body.push_str(line.split("//").next().unwrap_or(""));
        body.push('\n');
    }
    let mut depth = 0usize;
    let mut visible = String::with_capacity(body.len());
    for c in body.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => visible.push(c),
            _ => {}
        }
    }
    visible
        .split([';', '\n'])
        .all(|piece| piece.trim().is_empty() || piece.trim().eq_ignore_ascii_case("nop"))
}

/// The literal instruction text of an asm template. Stable MIR
/// renders the template as the Debug form of a
/// `Vec<InlineAsmTemplatePiece>` (`[String("nop")]`,
/// `[String("mov "), Placeholder { .. }, String("")]`): the
/// `String("...")` pieces are the literal text and the placeholders
/// are operand references, which contribute no instruction text.
/// A template that is not in that form (hand-built programs) is
/// taken as the literal text itself.
fn inline_asm_template_text(template: &str) -> String {
    let t = template.trim();
    if !t.starts_with('[') {
        return template.to_string();
    }
    let mut out = String::new();
    let mut rest = t;
    while let Some(i) = rest.find("String(\"") {
        let start = i + "String(\"".len();
        let mut chars = rest[start..].chars();
        let mut consumed = 0usize;
        loop {
            match chars.next() {
                None => break,
                Some('\\') => {
                    if let Some(c) = chars.next() {
                        out.push(match c {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            other => other,
                        });
                        consumed += 1 + c.len_utf8();
                    }
                }
                Some('"') => {
                    consumed += 1;
                    break;
                }
                Some(c) => {
                    out.push(c);
                    consumed += c.len_utf8();
                }
            }
        }
        rest = &rest[start + consumed..];
    }
    out
}

/// One source operand of a classified asm instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsmSrc {
    /// A template placeholder: an index into the terminator's
    /// operands, with the placeholder's size modifier (`e` = low
    /// 32 bits, `x` = low 16, `l` = low 8, `h` = bits 8..16), when
    /// the template spelled one.
    Reg(usize, Option<char>),
    /// An integer immediate, as raw bits.
    Imm(u64),
}

/// The binary operations of a classified asm program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsmBinOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Sar,
}

/// The unary operations of a classified asm program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsmUnOp {
    Neg,
    Not,
    Inc,
    Dec,
}

/// One instruction of a classified asm program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsmStep {
    /// A `nop` instruction: no effect. Classification drops these
    /// from the program; the variant exists so the parser can
    /// report one.
    Nop,
    Mov {
        dst: usize,
        dst_mod: Option<char>,
        src: AsmSrc,
    },
    Bin {
        op: AsmBinOp,
        dst: usize,
        dst_mod: Option<char>,
        src: AsmSrc,
    },
    Un {
        op: AsmUnOp,
        dst: usize,
        dst_mod: Option<char>,
    },
    Ret,
}

/// An inline-asm template classified as evaluable by
/// [`inline_asm_program`]: a straight-line sequence of register
/// operations over the terminator's operands. An empty `steps`
/// list is the step-over class (the template held no real
/// instructions: empty, comment-only or `nop`-only).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsmProgram {
    pub steps: Vec<AsmStep>,
    /// The terminator has no destination and the program's last
    /// step is `Ret` (the naked-function shape): after the steps
    /// and the output write-back, execution returns from the
    /// enclosing function.
    pub returns: bool,
}

/// One piece of an asm template: literal instruction text, or an
/// operand placeholder.
#[derive(Clone, Debug, PartialEq)]
enum AsmPiece {
    Text(String),
    Placeholder(usize, Option<char>),
}

/// Split a template into literal text and placeholders. Accepts
/// both the Stable MIR Debug form (`[String("..."), Placeholder {
/// operand_idx: N, modifier: Some('e'), .. }, ...]`) and the
/// plain-text form hand-built programs use, where placeholders are
/// written `{N}`, `{N:modifier}` or `{}` (the next unnamed index).
fn asm_pieces(template: &str) -> Vec<AsmPiece> {
    let t = template.trim();
    if t.starts_with('[') {
        asm_pieces_debug_form(t)
    } else {
        asm_pieces_plain_form(template)
    }
}

fn asm_pieces_debug_form(t: &str) -> Vec<AsmPiece> {
    let bytes = t.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if t[i..].starts_with("String(\"") {
            let start = i + "String(\"".len();
            let mut text = String::new();
            let mut j = start;
            let chars: Vec<(usize, char)> = t[start..].char_indices().collect();
            let mut k = 0usize;
            while k < chars.len() {
                let (off, c) = chars[k];
                j = start + off + c.len_utf8();
                match c {
                    '\\' => {
                        if let Some(&(_, esc)) = chars.get(k + 1) {
                            text.push(match esc {
                                'n' => '\n',
                                't' => '\t',
                                'r' => '\r',
                                other => other,
                            });
                            j = start + chars[k + 1].0 + esc.len_utf8();
                            k += 2;
                        } else {
                            k += 1;
                        }
                    }
                    '"' => {
                        break;
                    }
                    _ => {
                        text.push(c);
                        k += 1;
                    }
                }
            }
            out.push(AsmPiece::Text(text));
            i = j;
        } else if t[i..].starts_with("Placeholder") {
            let Some(open) = t[i..].find('{') else {
                break;
            };
            let Some(close) = t[i + open..].find('}') else {
                break;
            };
            let inner = &t[i + open + 1..i + open + close];
            let idx = inner.split("operand_idx:").nth(1).and_then(|rest| {
                rest.trim()
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse::<usize>()
                    .ok()
            });
            let modifier = inner.split("modifier:").nth(1).and_then(|rest| {
                let rest = rest.trim();
                if rest.starts_with("None") {
                    None
                } else if rest.starts_with("Some('") {
                    rest.chars().nth(6)
                } else {
                    None
                }
            });
            if let Some(idx) = idx {
                out.push(AsmPiece::Placeholder(idx, modifier));
            }
            i = i + open + close + 1;
        } else {
            i += t[i..].chars().next().map_or(1, |c| c.len_utf8());
        }
    }
    out
}

fn asm_pieces_plain_form(template: &str) -> Vec<AsmPiece> {
    let mut out = Vec::new();
    let mut text = String::new();
    let mut unnamed = 0usize;
    let chars: Vec<char> = template.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '{' {
            if chars.get(i + 1) == Some(&'{') {
                text.push('{');
                i += 2;
                continue;
            }
            // Try `{N}`, `{N:mod}` or `{}`.
            let mut j = i + 1;
            let mut digits = String::new();
            while j < chars.len() && chars[j].is_ascii_digit() {
                digits.push(chars[j]);
                j += 1;
            }
            let mut modifier = None;
            let mut ok = true;
            if chars.get(j) == Some(&':') {
                j += 1;
                let start = j;
                while j < chars.len() && chars[j] != '}' {
                    j += 1;
                }
                let mod_str: String = chars[start..j].iter().collect();
                if mod_str.chars().count() == 1 {
                    modifier = mod_str.chars().next();
                } else {
                    ok = false;
                }
            }
            if ok && chars.get(j) == Some(&'}') {
                if !text.is_empty() {
                    out.push(AsmPiece::Text(std::mem::take(&mut text)));
                }
                let idx = if digits.is_empty() {
                    let v = unnamed;
                    unnamed += 1;
                    v
                } else {
                    digits.parse::<usize>().unwrap_or(usize::MAX)
                };
                out.push(AsmPiece::Placeholder(idx, modifier));
                i = j + 1;
                continue;
            }
            text.push('{');
            i += 1;
        } else if chars[i] == '}' && chars.get(i + 1) == Some(&'}') {
            text.push('}');
            i += 2;
        } else {
            text.push(chars[i]);
            i += 1;
        }
    }
    if !text.is_empty() {
        out.push(AsmPiece::Text(text));
    }
    out
}

/// The pieces of a template with comments removed. A placeholder
/// inside a comment is substituted text in the rendered assembly
/// and has no effect there, so it is dropped with the comment.
fn asm_stream(template: &str) -> Vec<AsmPiece> {
    let mut out: Vec<AsmPiece> = Vec::new();
    let mut block_comment = false;
    let mut line_comment = false;
    for piece in asm_pieces(template) {
        match piece {
            AsmPiece::Placeholder(..) => {
                if !block_comment && !line_comment {
                    out.push(piece);
                }
            }
            AsmPiece::Text(text) => {
                let chars: Vec<char> = text.chars().collect();
                let mut cur = String::new();
                let mut i = 0usize;
                while i < chars.len() {
                    if block_comment {
                        if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                            block_comment = false;
                            i += 2;
                        } else {
                            i += 1;
                        }
                        continue;
                    }
                    if line_comment {
                        if chars[i] == '\n' {
                            line_comment = false;
                            cur.push('\n');
                        }
                        i += 1;
                        continue;
                    }
                    if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                        block_comment = true;
                        i += 2;
                        continue;
                    }
                    if chars[i] == '/' && chars.get(i + 1) == Some(&'/') {
                        line_comment = true;
                        i += 2;
                        continue;
                    }
                    cur.push(chars[i]);
                    i += 1;
                }
                if !cur.is_empty() {
                    out.push(AsmPiece::Text(cur));
                }
            }
        }
    }
    out
}

/// Split the comment-free stream into instructions (separated by
/// `;` or newlines). Each instruction is rendered as text with
/// placeholders replaced by `\u{1}<idx>[:<mod>]\u{2}` sentinels.
fn asm_instructions(template: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for piece in asm_stream(template) {
        match piece {
            AsmPiece::Text(text) => {
                for c in text.chars() {
                    if c == ';' || c == '\n' {
                        out.push(std::mem::take(&mut cur));
                    } else {
                        cur.push(c);
                    }
                }
            }
            AsmPiece::Placeholder(idx, modifier) => {
                cur.push('\u{1}');
                cur.push_str(&idx.to_string());
                if let Some(m) = modifier {
                    cur.push(':');
                    cur.push(m);
                }
                cur.push('\u{2}');
            }
        }
    }
    out.push(cur);
    out.into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse an integer immediate: an optional sign, then a decimal,
/// `0x`, `0b` or `0o` literal. Returns the raw bits.
fn asm_immediate(text: &str) -> Option<u64> {
    let (neg, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let (radix, digits) = if let Some(d) = digits.strip_prefix("0x") {
        (16, d)
    } else if let Some(d) = digits.strip_prefix("0b") {
        (2, d)
    } else if let Some(d) = digits.strip_prefix("0o") {
        (8, d)
    } else {
        (10, digits)
    };
    if digits.is_empty() {
        return None;
    }
    let v = i128::from_str_radix(digits, radix).ok()?;
    let v = if neg { -v } else { v };
    Some(v as u64)
}

/// Parse one operand token of an instruction: a placeholder
/// sentinel or an immediate. Anything else (a literal register
/// name, a memory reference, a label) is outside the class.
fn asm_src(token: &str, operands: &[InlineAsmOperand]) -> Option<AsmSrc> {
    let token = token.trim();
    if let Some(inner) = token
        .strip_prefix('\u{1}')
        .and_then(|t| t.strip_suffix('\u{2}'))
    {
        let (idx, modifier) = match inner.split_once(':') {
            Some((n, m)) => {
                let mut cs = m.chars();
                let c = cs.next()?;
                if cs.next().is_some() {
                    return None;
                }
                (n.parse::<usize>().ok()?, Some(c))
            }
            None => (inner.parse::<usize>().ok()?, None),
        };
        if idx >= operands.len() {
            return None;
        }
        if let Some(m) = modifier {
            if !matches!(m, 'e' | 'x' | 'l' | 'h') {
                return None;
            }
        }
        return Some(AsmSrc::Reg(idx, modifier));
    }
    if token.contains('\u{1}') {
        // A placeholder fused with literal text (e.g. `[{}]` or
        // `x{}`) is a memory reference or a sized register name:
        // outside the class.
        return None;
    }
    asm_immediate(token).map(AsmSrc::Imm)
}

fn asm_writable(operands: &[InlineAsmOperand], src: &AsmSrc) -> bool {
    match src {
        AsmSrc::Reg(idx, _) => matches!(
            operands[*idx].kind,
            AsmOperandKind::Out | AsmOperandKind::InOut
        ),
        AsmSrc::Imm(_) => false,
    }
}

/// Parse one instruction of the stream. `None` for anything
/// outside the supported class.
fn asm_step(text: &str, operands: &[InlineAsmOperand]) -> Option<AsmStep> {
    let mut segments = text.split(',');
    let first = segments.next()?.trim();
    let mut words = first.splitn(2, char::is_whitespace);
    let mnemonic = words.next()?.to_ascii_lowercase();
    let first_operand = words.next().map(str::trim).filter(|w| !w.is_empty());
    let mut args: Vec<&str> = Vec::new();
    if let Some(op) = first_operand {
        args.push(op);
    }
    for seg in segments {
        args.push(seg.trim());
    }
    let dst_src = |args: &[&str]| -> Option<(AsmSrc, AsmSrc)> {
        if args.len() != 2 {
            return None;
        }
        Some((asm_src(args[0], operands)?, asm_src(args[1], operands)?))
    };
    match mnemonic.as_str() {
        "nop" => {
            if args.is_empty() {
                Some(AsmStep::Nop)
            } else {
                None
            }
        }
        "ret" => {
            if args.is_empty() {
                Some(AsmStep::Ret)
            } else {
                None
            }
        }
        "mov" => {
            let (dst, src) = dst_src(&args)?;
            let AsmSrc::Reg(didx, dmod) = dst else {
                return None;
            };
            if !asm_writable(operands, &dst) {
                return None;
            }
            Some(AsmStep::Mov {
                dst: didx,
                dst_mod: dmod,
                src,
            })
        }
        "add" | "sub" | "and" | "or" | "xor" | "shl" | "shr" | "sar" => {
            let (dst, src) = dst_src(&args)?;
            let AsmSrc::Reg(didx, dmod) = dst else {
                return None;
            };
            if !asm_writable(operands, &dst) {
                return None;
            }
            let op = match mnemonic.as_str() {
                "add" => AsmBinOp::Add,
                "sub" => AsmBinOp::Sub,
                "and" => AsmBinOp::And,
                "or" => AsmBinOp::Or,
                "xor" => AsmBinOp::Xor,
                "shl" => AsmBinOp::Shl,
                "shr" => AsmBinOp::Shr,
                _ => AsmBinOp::Sar,
            };
            Some(AsmStep::Bin {
                op,
                dst: didx,
                dst_mod: dmod,
                src,
            })
        }
        "neg" | "not" | "inc" | "dec" => {
            if args.len() != 1 {
                return None;
            }
            let dst = asm_src(args[0], operands)?;
            let AsmSrc::Reg(didx, dmod) = dst else {
                return None;
            };
            if !asm_writable(operands, &dst) {
                return None;
            }
            let op = match mnemonic.as_str() {
                "neg" => AsmUnOp::Neg,
                "not" => AsmUnOp::Not,
                "inc" => AsmUnOp::Inc,
                _ => AsmUnOp::Dec,
            };
            Some(AsmStep::Un {
                op,
                dst: didx,
                dst_mod: dmod,
            })
        }
        _ => None,
    }
}

/// Whether an `InlineAsm` terminator is in the evaluable class,
/// and if so its program. The class is deliberately narrow:
///
/// * the template, comments removed, is a straight-line sequence
///   of `mov` / `add` / `sub` / `and` / `or` / `xor` / `shl` /
///   `shr` / `sar` / `neg` / `not` / `inc` / `dec` / `nop`
///   instructions whose operands are only template placeholders
///   and integer immediates — no memory references, no literal
///   register names, no labels, no control flow; or the template
///   holds no real instructions at all (empty, comment-only or
///   `nop`-only: the step-over class, where the operands are
///   inert and never inspected);
/// * every operand of a non-empty program is a register-class
///   integer operand of width 8, 16, 32 or 64 (no explicit
///   registers, no `sym` operands, no floats or aggregates), so
///   each operand is one virtual register in the interpreter;
/// * a `ret` may appear only as the last instruction of a
///   destination-less terminator (the naked-function shape);
/// * a destination-less terminator with real instructions but no
///   `ret` (a `noreturn` block) is outside the class, as is AT&T
///   syntax.
///
/// Anything else keeps the named conversion error the caller
/// already produces for `InlineAsm`.
pub fn inline_asm_program(
    template: &str,
    options: &str,
    operands: &[InlineAsmOperand],
    has_destination: bool,
) -> Option<AsmProgram> {
    if options.contains("ATT_SYNTAX") {
        return None;
    }
    if has_destination && options.contains("NORETURN") {
        return None;
    }
    let instructions = asm_instructions(template);
    let mut steps: Vec<AsmStep> = Vec::new();
    let mut real = 0usize;
    for text in &instructions {
        match asm_step(text, operands)? {
            AsmStep::Nop => {}
            step => {
                real += 1;
                steps.push(step);
            }
        }
    }
    if real == 0 {
        // The step-over class: no instruction reads or writes any
        // operand, so the operands are not gated.
        return Some(AsmProgram {
            steps: Vec::new(),
            returns: false,
        });
    }
    // A non-empty program runs on the virtual register file: gate
    // every operand.
    for op in operands {
        if op.kind == AsmOperandKind::Sym {
            return None;
        }
        // Register operands must be register-class (an explicit
        // register is outside the class); a const operand has no
        // register and is gated on its literal bits below instead.
        if op.kind != AsmOperandKind::Const && !op.reg_class {
            return None;
        }
        if !matches!(op.width, 8 | 16 | 32 | 64) {
            return None;
        }
        if op.kind == AsmOperandKind::Const && op.const_bits.is_none() {
            return None;
        }
    }
    let rets = steps.iter().filter(|s| matches!(s, AsmStep::Ret)).count();
    if rets > 0 {
        if has_destination || rets > 1 || !matches!(steps.last(), Some(AsmStep::Ret)) {
            return None;
        }
        return Some(AsmProgram {
            steps,
            returns: true,
        });
    }
    if !has_destination {
        return None;
    }
    Some(AsmProgram {
        steps,
        returns: false,
    })
}

impl TerminatorKind {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Goto { .. } => "Goto",
            Self::SwitchInt { .. } => "SwitchInt",
            Self::Resume => "Resume",
            Self::Terminate(_) => "Terminate",
            Self::Return => "Return",
            Self::Unreachable => "Unreachable",
            Self::Drop { .. } => "Drop",
            Self::Call { .. } => "Call",
            Self::Assert { .. } => "Assert",
            Self::Yield { .. } => "Yield",
            Self::CoroutineDrop => "CoroutineDrop",
            Self::FalseEdge { .. } => "FalseEdge",
            Self::FalseUnwind { .. } => "FalseUnwind",
            Self::InlineAsm { .. } => "InlineAsm",
            Self::TailCall { .. } => "TailCall",
        }
    }
    /// All basic blocks this terminator can transfer control to (incl. unwind cleanup).
    pub fn successors(&self) -> Vec<BasicBlock> {
        match self {
            Self::Goto { target } => vec![*target],
            Self::SwitchInt { targets, .. } => targets.all_targets(),
            Self::Resume
            | Self::Terminate(_)
            | Self::Return
            | Self::Unreachable
            | Self::CoroutineDrop
            | Self::TailCall { .. } => vec![],
            Self::Drop { target, unwind, .. } => {
                let mut v = vec![*target];
                if let Some(c) = unwind.cleanup_block() {
                    v.push(c);
                }
                v
            }
            Self::Call { target, unwind, .. } => {
                let mut v = Vec::new();
                if let Some(t) = target {
                    v.push(*t);
                }
                if let Some(c) = unwind.cleanup_block() {
                    v.push(c);
                }
                v
            }
            Self::Assert { target, unwind, .. } => {
                let mut v = vec![*target];
                if let Some(c) = unwind.cleanup_block() {
                    v.push(c);
                }
                v
            }
            Self::Yield { resume, drop, .. } => {
                let mut v = vec![*resume];
                if let Some(d) = drop {
                    v.push(*d);
                }
                v
            }
            Self::FalseEdge {
                real_target,
                imaginary_target,
            } => vec![*real_target, *imaginary_target],
            Self::FalseUnwind {
                real_target,
                unwind,
            } => {
                let mut v = vec![*real_target];
                if let Some(c) = unwind.cleanup_block() {
                    v.push(c);
                }
                v
            }
            Self::InlineAsm {
                destination,
                unwind,
                ..
            } => {
                let mut v = Vec::new();
                if let Some(d) = destination {
                    v.push(*d);
                }
                if let Some(c) = unwind.cleanup_block() {
                    v.push(c);
                }
                v
            }
        }
    }
    pub fn samples() -> Vec<TerminatorKind> {
        use crate::operand::Constant;
        use crate::place::Local;
        let bb = |n: u32| BasicBlock(n);
        let op = || Operand::constant(Constant::bool(true));
        let pl = || Place::from_local(Local(1));
        vec![
            Self::Goto { target: bb(1) },
            Self::SwitchInt {
                discr: op(),
                targets: SwitchTargets::new(vec![0], vec![bb(1)], bb(2)),
            },
            Self::Resume,
            Self::Terminate(TerminateKind::Abort),
            Self::Return,
            Self::Unreachable,
            Self::Drop {
                place: pl(),
                target: bb(1),
                unwind: UnwindAction::Continue,
                replace: false,
            },
            Self::Call {
                func: op(),
                args: vec![],
                destination: pl(),
                target: Some(bb(1)),
                unwind: UnwindAction::Continue,
                call_source: CallSource::Normal,
            },
            Self::Assert {
                cond: op(),
                expected: true,
                msg: Box::new(AssertMessage::NullPointerDereference),
                target: bb(1),
                unwind: UnwindAction::Continue,
            },
            Self::Yield {
                value: op(),
                resume: bb(1),
                resume_arg: pl(),
                drop: None,
            },
            Self::CoroutineDrop,
            Self::FalseEdge {
                real_target: bb(1),
                imaginary_target: bb(2),
            },
            Self::FalseUnwind {
                real_target: bb(1),
                unwind: UnwindAction::Continue,
            },
            Self::InlineAsm {
                template: String::new(),
                operands: vec![],
                options: String::new(),
                destination: None,
                unwind: UnwindAction::Continue,
            },
            Self::TailCall {
                func: op(),
                args: vec![],
            },
        ]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Terminator {
    pub source_info: SourceInfo,
    pub kind: TerminatorKind,
}
impl Terminator {
    pub fn new(kind: TerminatorKind) -> Self {
        Self {
            source_info: SourceInfo::DUMMY,
            kind,
        }
    }
    pub fn goto(target: BasicBlock) -> Self {
        Self::new(TerminatorKind::Goto { target })
    }
    pub fn return_() -> Self {
        Self::new(TerminatorKind::Return)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage;
    #[test]
    fn terminator_samples_cover_table() {
        let names: Vec<&str> = TerminatorKind::samples()
            .iter()
            .map(|t| t.kind_name())
            .collect();
        assert_eq!(names.len(), coverage::TERMINATOR_KINDS.len());
        for n in coverage::TERMINATOR_KINDS {
            assert!(names.contains(n), "missing {n}");
        }
    }
    #[test]
    fn successors_include_cleanup() {
        let t = TerminatorKind::Call {
            func: Operand::constant(crate::operand::Constant::int(0)),
            args: vec![],
            destination: Place::from_local(crate::place::Local(0)),
            target: Some(BasicBlock(2)),
            unwind: UnwindAction::Cleanup(BasicBlock(9)),
            call_source: CallSource::Normal,
        };
        assert_eq!(t.successors(), vec![BasicBlock(2), BasicBlock(9)]);
        // Diverging call (e.g. to a `!` function, stable in 1.100.0): no return target.
        let d = TerminatorKind::Call {
            func: Operand::constant(crate::operand::Constant::int(0)),
            args: vec![],
            destination: Place::from_local(crate::place::Local(0)),
            target: None,
            unwind: UnwindAction::Continue,
            call_source: CallSource::Normal,
        };
        assert!(d.successors().is_empty());
    }

    #[test]
    fn inline_asm_noop_classification() {
        // Empty, comment-only and nop-only templates are no-ops.
        assert!(inline_asm_is_noop("", ""));
        assert!(inline_asm_is_noop("nop", ""));
        assert!(inline_asm_is_noop("nop; nop", "NOMEM"));
        assert!(inline_asm_is_noop("/* {0} */", ""));
        assert!(inline_asm_is_noop("// just a comment", ""));
        // Stable MIR's Debug-form template pieces.
        assert!(inline_asm_is_noop("[]", ""));
        assert!(inline_asm_is_noop("[String(\"nop\")]", ""));
        assert!(inline_asm_is_noop(
            "[String(\"/* \"), Placeholder { operand_idx: 0 }, String(\" */\")]",
            ""
        ));
        // Real instructions, fences and queries are not.
        assert!(!inline_asm_is_noop("mov eax, 1", ""));
        assert!(!inline_asm_is_noop("cpuid", ""));
        assert!(!inline_asm_is_noop("mfence", ""));
        assert!(!inline_asm_is_noop("[String(\"nop; cpuid\")]", ""));
        // A noreturn block never reaches its destination.
        assert!(!inline_asm_is_noop("", "NORETURN"));
    }

    fn asm_op(kind: AsmOperandKind, width: u32) -> InlineAsmOperand {
        InlineAsmOperand {
            kind,
            reg_class: true,
            width,
            const_bits: None,
            in_value: None,
            out_place: None,
        }
    }

    #[test]
    fn inline_asm_program_classification() {
        use AsmOperandKind::*;
        let out32 = asm_op(Out, 32);
        let in32 = asm_op(In, 32);
        // The Debug-form mov/add program from a real body.
        let tmpl = "[String(\"mov \"), Placeholder { operand_idx: 0, modifier: Some('e'), span: x }, String(\", \"), Placeholder { operand_idx: 1, modifier: Some('e'), span: y }, String(\"\\n\"), String(\"add \"), Placeholder { operand_idx: 0, modifier: Some('e'), span: z }, String(\", 2\")]";
        let prog = inline_asm_program(tmpl, "", &[out32.clone(), in32.clone()], true)
            .expect("mov/add program is in the class");
        assert_eq!(
            prog.steps,
            vec![
                AsmStep::Mov {
                    dst: 0,
                    dst_mod: Some('e'),
                    src: AsmSrc::Reg(1, Some('e')),
                },
                AsmStep::Bin {
                    op: AsmBinOp::Add,
                    dst: 0,
                    dst_mod: Some('e'),
                    src: AsmSrc::Imm(2),
                },
            ]
        );
        assert!(!prog.returns);
        // Plain-text form with {} placeholders and a hex immediate.
        let prog = inline_asm_program(
            "sub {}, 0x10",
            "",
            &[asm_op(InOut, 64)],
            true,
        )
        .expect("sub program is in the class");
        assert_eq!(
            prog.steps,
            vec![AsmStep::Bin {
                op: AsmBinOp::Sub,
                dst: 0,
                dst_mod: None,
                src: AsmSrc::Imm(16),
            }]
        );
        // Comment-only templates are the step-over class, and their
        // operands are not gated (explicit registers are inert).
        let mut explicit = asm_op(Out, 32);
        explicit.reg_class = false;
        let prog = inline_asm_program("/* {0} */", "", &[explicit], true)
            .expect("comment-only template steps over");
        assert!(prog.steps.is_empty());
        // A destination-less empty template (noreturn) also steps
        // over (to the unreachable continuation).
        assert!(inline_asm_program("", "NORETURN", &[], false).is_some());
        // The naked `ret` shape.
        let prog = inline_asm_program("ret", "", &[], false).expect("naked ret");
        assert!(prog.returns);
        assert_eq!(prog.steps, vec![AsmStep::Ret]);
        // Outside the class:
        // memory reference
        assert!(inline_asm_program("mov {}, [{}]", "", &[out32.clone(), in32.clone()], true).is_none());
        // literal register names
        assert!(inline_asm_program("mov eax, 1", "", &[], true).is_none());
        // control flow / calls / unknown mnemonics
        assert!(inline_asm_program("call {}", "", &[in32.clone()], true).is_none());
        assert!(inline_asm_program("jmp {}", "", &[in32.clone()], true).is_none());
        assert!(inline_asm_program("cpuid", "", &[], true).is_none());
        // ret with a destination, and a destination-less program
        // without ret (noreturn with real instructions)
        assert!(inline_asm_program("ret", "", &[], true).is_none());
        assert!(inline_asm_program("mov {}, 1", "NORETURN", &[out32.clone()], false).is_none());
        // a sym operand in a real program
        assert!(inline_asm_program("mov {}, {}", "", &[out32.clone(), asm_op(Sym, 0)], true).is_none());
        // a float-width operand (width 0 from the driver)
        assert!(inline_asm_program("mov {}, {}", "", &[out32.clone(), asm_op(In, 0)], true).is_none());
        // an unknown modifier
        assert!(inline_asm_program("mov {0:w}, {1:w}", "", &[out32.clone(), in32.clone()], true).is_none());
        // writing to an in-only operand
        assert!(inline_asm_program("mov {}, {}", "", &[in32.clone(), in32.clone()], true).is_none());
        // AT&T syntax
        assert!(inline_asm_program("mov {}, {}", "ATT_SYNTAX", &[out32, in32], true).is_none());
        // A const operand has no register: it is gated on its
        // literal bits, not on the register class.
        let mut konst = asm_op(Const, 32);
        konst.reg_class = false;
        konst.const_bits = Some(7);
        assert!(inline_asm_program(
            "mov {}, {1}",
            "",
            &[asm_op(Out, 32), konst.clone()],
            true,
        )
        .is_some());
        konst.const_bits = None;
        assert!(inline_asm_program("mov {}, {1}", "", &[asm_op(Out, 32), konst], true).is_none());
    }
}
