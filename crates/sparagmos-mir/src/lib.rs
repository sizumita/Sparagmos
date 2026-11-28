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
pub mod coverage_mapping;
pub mod operand;
pub mod place;
pub mod rvalue;
pub mod statement;
pub mod terminator;
pub mod ty;

pub use coverage_mapping::{
    CovExpression, CovFunction, CovMapping, CovMappingKind, CovTerm, CoverageMapping,
};
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
    arch_reg_index, arch_reg_named, inline_asm_is_noop, inline_asm_program, ArchReg, AsmAbi,
    AsmBinOp, AsmCallSig, AsmCond, AsmDst, AsmOperandKind, AsmProgram, AsmScalar, AsmSrc, AsmStep, AsmUnOp,
    AssertMessage, BasicBlock, CallSource, CoroutineKind, InlineAsmOperand, SwitchTargets,
    TerminateKind, Terminator, TerminatorKind, UnwindAction, UnwindTerminateReason,
};
pub use ty::{
    Abi, AdtDebugField, AdtDebugShape, AdtDebugShapes, AdtDebugVariant, AdtDefId,
    EnvDebugShape, EnvDebugShapes, EnvDebugVariant,
    AdtDiscriminants, AdtDstLayout, AdtDstLayouts, AdtFieldLayouts, AdtInstanceShapes, AdtLayouts, AdtNiches, AdtTagLayouts, CoroutineLayouts, CoroutineSlots, AdtShape, AdtShapeKind, TupleLayouts,
    AdtShapes, AliasKind, ArrayLen, DefId, DiscriminantValues, EnumNiche, EnumTag, FloatTy,
    ExtraStaticSeeds, StaticAligns, StaticSeedDef, StaticTys, ThreadLocalDef, ThreadLocals,
    FnSig, IntTy, Mutability, Region, Safety, Ty, UintTy,
};
