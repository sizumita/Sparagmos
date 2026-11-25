//! `Body`, `BasicBlockData`, debug info.
use crate::operand::Constant;
use crate::place::{Local, LocalDecl, Place};
use crate::statement::{SourceInfo, Span, Statement};
use crate::terminator::{BasicBlock, Terminator};
use crate::ty::Ty;

#[derive(Clone, Debug, PartialEq)]
pub enum VarDebugInfoContents {
    Place(Place),
    Local(Local),
    Constant(Box<Constant>),
}
#[derive(Clone, Debug, PartialEq)]
pub struct VarDebugInfo {
    pub name: String,
    pub source_info: SourceInfo,
    pub value: VarDebugInfoContents,
}

/// The source position of one statement or terminator, resolved by
/// the driver from the Stable MIR span at conversion time (the span
/// itself is opaque outside the compiler, so the resolution happens
/// there): an index into the program-wide source-file table plus a
/// 1-based line number. The interpreter's debugger interface is the
/// consumer: breakpoints and stepping are line-based.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SourceLine {
    /// Index into the program's source-file table; `u32::MAX` when
    /// the position is unknown.
    pub file: u32,
    /// 1-based line number; 0 when unknown.
    pub line: u32,
    /// 1-based column number; 0 when unknown.
    pub col: u32,
}

impl SourceLine {
    pub const UNKNOWN: SourceLine = SourceLine { file: u32::MAX, line: 0, col: 0 };

    pub fn is_known(&self) -> bool {
        self.line != 0 && self.file != u32::MAX
    }
}

/// Source lines of one basic block, parallel to
/// [`BasicBlockData`]: one entry per statement, plus the
/// terminator's. Empty in bodies built by hand.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlockSourceLines {
    pub statements: Vec<SourceLine>,
    pub terminator: SourceLine,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasicBlockData {
    pub statements: Vec<Statement>,
    pub terminator: Option<Terminator>,
    pub is_cleanup: bool,
}
impl BasicBlockData {
    pub fn new(terminator: Terminator) -> Self {
        Self {
            statements: Vec::new(),
            terminator: Some(terminator),
            is_cleanup: false,
        }
    }
    pub fn cleanup(terminator: Terminator) -> Self {
        Self {
            statements: Vec::new(),
            terminator: Some(terminator),
            is_cleanup: true,
        }
    }
    pub fn successors(&self) -> Vec<BasicBlock> {
        self.terminator
            .as_ref()
            .map(|t| t.kind.successors())
            .unwrap_or_default()
    }
}

/// Our model of `rustc_middle::mir::Body`. Locals are indexed: `_0` is the
/// return place, `_1..=arg_count` are arguments, the rest are temporaries /
/// user variables.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    pub basic_blocks: Vec<BasicBlockData>,
    pub local_decls: Vec<LocalDecl>,
    pub arg_count: usize,
    pub var_debug_info: Vec<VarDebugInfo>,
    pub span: Span,
    pub required_consts: Vec<Constant>,
    /// Source lines of every block, parallel to `basic_blocks`
    /// (see [`BlockSourceLines`]). Filled by the driver from the
    /// Stable MIR spans; empty for hand-built bodies.
    pub source_lines: Vec<BlockSourceLines>,
    /// For each coverage statement in this body (in statement
    /// order), the mangled symbol of the instance the statement
    /// was inlined from, when it was inlined from another
    /// instance; `None` for the body's own statements. Filled by
    /// the driver's coverage capture from the internal MIR source
    /// scopes; empty when the body's coverage statements could
    /// not be aligned with the internal body (the interpreter
    /// then attributes every statement to the executing body,
    /// which is exact for bodies without inlined coverage).
    pub coverage_owners: Vec<Option<String>>,
}
impl Body {
    pub fn new(
        local_decls: Vec<LocalDecl>,
        arg_count: usize,
        basic_blocks: Vec<BasicBlockData>,
    ) -> Self {
        Self {
            basic_blocks,
            local_decls,
            arg_count,
            var_debug_info: Vec::new(),
            span: Span::DUMMY,
            required_consts: Vec::new(),
            source_lines: Vec::new(),
            coverage_owners: Vec::new(),
        }
    }
    pub fn return_ty(&self) -> Option<&Ty> {
        self.local_decls.first().map(|d| &d.ty)
    }
    pub fn local_ty(&self, local: Local) -> Option<&Ty> {
        self.local_decls.get(local.index()).map(|d| &d.ty)
    }
    pub fn arg_locals(&self) -> Vec<Local> {
        (1..=self.arg_count).map(|i| Local(i as u32)).collect()
    }
    pub fn block_count(&self) -> usize {
        self.basic_blocks.len()
    }
    pub fn statement_count(&self) -> usize {
        self.basic_blocks.iter().map(|b| b.statements.len()).sum()
    }
    /// True when the function can never return normally (return type is `!`).
    /// Since Rust 1.100.0 the never type is stable, so this is a first-class case:
    /// such bodies contain calls with `target: None` and no reachable `Return`.
    pub fn is_diverging(&self) -> bool {
        matches!(self.return_ty(), Some(Ty::Never))
    }
    /// Reachable blocks from `bb0` following terminator successors.
    pub fn reachable_blocks(&self) -> Vec<BasicBlock> {
        let mut seen = vec![false; self.basic_blocks.len()];
        let mut stack = vec![BasicBlock(0)];
        let mut out = Vec::new();
        while let Some(bb) = stack.pop() {
            if bb.index() >= seen.len() || seen[bb.index()] {
                continue;
            }
            seen[bb.index()] = true;
            out.push(bb);
            for s in self.basic_blocks[bb.index()].successors() {
                stack.push(s);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{IntTy, Ty};
    #[test]
    fn diverging_body_detected() {
        let body = Body::new(
            vec![LocalDecl::new(Ty::Never)],
            0,
            vec![BasicBlockData::new(Terminator::new(
                crate::terminator::TerminatorKind::Unreachable,
            ))],
        );
        assert!(body.is_diverging());
        assert_eq!(body.return_ty(), Some(&Ty::Never));
    }
    #[test]
    fn int_body_not_diverging() {
        let body = Body::new(vec![LocalDecl::new(Ty::Int(IntTy::I32))], 0, vec![]);
        assert!(!body.is_diverging());
    }
}
