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
        operands: Vec<Operand>,
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
}
