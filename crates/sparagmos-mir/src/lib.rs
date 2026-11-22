//! Sparagmos MIR — our own model of rustc MIR.
//!
//! The reference is `rustc_middle::mir` as of **Rust 1.100.0** (released
//! 2026-11-12, verified on the dev host with `rustup run 1.100.0 rustc
//! --emit=mir`). Rust 1.100.0 stabilised the never type `!`, so [`Ty::Never`]
//! is a first-class return/local type here, and calls to diverging functions
//! are represented with `TerminatorKind::Call { target: None, .. }`.
//!
//! Other lanes should depend only on the types re-exported from this root.
//! Variant denominators for coverage live in [`coverage`].

pub mod body;
pub mod coverage;
pub mod operand;
pub mod place;
pub mod rvalue;
pub mod statement;
pub mod terminator;
pub mod ty;

pub use body::{
    BasicBlockData, BlockSourceLines, Body, SourceLine, VarDebugInfo, VarDebugInfoContents,
};
pub use operand::{ConstKind, ConstValue, Constant, Operand, Scalar};
pub use place::{Local, LocalDecl, Place, PlaceElem, ProjectionElem, RETURN_PLACE};
pub use rvalue::{
    AggregateKind, BinOp, BorrowKind, CastKind, FakeBorrowKind, NullaryOp, PointerCoercion, Rvalue,
    UnOp,
};
pub use statement::{
    FakeReadCause, NonDivergingIntrinsic, RetagKind, SourceInfo, Span, Statement, StatementKind,
    UserTypeProjection, Variance,
};
pub use terminator::{
    inline_asm_is_noop, inline_asm_program, AsmBinOp, AsmOperandKind, AsmProgram, AsmSrc, AsmStep,
    AsmUnOp, AssertMessage, BasicBlock, CallSource, CoroutineKind, InlineAsmOperand, SwitchTargets,
    TerminateKind, Terminator, TerminatorKind, UnwindAction, UnwindTerminateReason,
};
pub use ty::{
    Abi, AdtDebugField, AdtDebugShape, AdtDebugShapes, AdtDebugVariant, AdtDefId,
    AdtDiscriminants, AdtInstanceShapes, AdtLayouts, AdtNiches, CoroutineLayouts, CoroutineSlots, AdtShape, AdtShapeKind,
    AdtShapes, AliasKind, ArrayLen, DefId, DiscriminantValues, EnumNiche, FloatTy,
    ThreadLocalDef, ThreadLocals,
    FnSig, IntTy, Mutability, Region, Safety, Ty, UintTy,
};
