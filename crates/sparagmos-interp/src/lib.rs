//! # sparagmos-interp — MIR interpreter (correctness oracle)
//!
//! Sparagmos is a from-scratch JIT runtime for Rust MIR (target: executing
//! Rust 1.100.0, released 2026-11-12, at the 99% level). This crate is the
//! reference interpreter: it executes MIR slowly but obviously correctly,
//! so the JIT lane can diff its results against this oracle on the same
//! input instead of against expectations written by hand.
//!
//! The interpreter executes the minimal MIR-like IR defined in
//! [`ir`], modelled on rustc MIR as of Rust 1.100.0. Implemented:
//!
//! * values: signed/unsigned integers (width-aware), bool, unit, tuples,
//!   ADT aggregates (variant + fields), string/byte data, resolved
//!   references and fat pointers, and the `fmt` value family;
//! * frames with a locals array, basic-block execution; places resolve
//!   across frames so references read and write through;
//! * assignments, binary / unary / checked-binary operations
//!   (overflow of a plain operation is a guest panic, a checked operation
//!   yields `(value, overflowed)`), aggregates, scalar casts, unsizing
//!   casts, `Len` / `Discriminant` / `PtrMetadata`;
//! * terminators: `goto`, `switchInt`, intra-module `call`/`return`
//!   (recursion included), runtime-support calls ([`builtins`]: printing
//!   and formatting, fmt argument construction, modelled std iterators),
//!   `drop` (for types with provably unobservable drop glue), `assert`,
//!   `return`, `unreachable`;
//! * a step limit and a call-depth limit, so no input can hang the oracle;
//! * explicit errors for every unimplemented statement / terminator /
//!   rvalue / projection — nothing is silently skipped.
//!
//! ## Assumed Lane A (`sparagmos-mir`) API — and the differences
//!
//! Lane A was still in flight when this lane was written, so this crate
//! defines its own IR in [`ir`], shaped after the API Lane A is expected
//! to expose:
//!
//! | Assumed Lane A API | This crate ([`ir`]) | Difference to reconcile |
//! |---|---|---|
//! | `Body { local_decls, basic_blocks, arg_count }` | [`ir::Body`] `{ name, arg_count, local_tys, blocks }` | Lane A will likely key bodies by `DefId`, not `name`; an adapter can build the name map. `local_tys` stands in for `LocalDecl { ty, .. }`. |
//! | `BasicBlockData { statements, terminator }` | [`ir::BasicBlockData`] | Same shape. |
//! | `StatementKind::Assign(Place, Rvalue)` | [`ir::StatementKind::Assign`] | Same shape. |
//! | `TerminatorKind::{Goto, SwitchInt, Call, Return, Assert, Unreachable}` | [`ir::TerminatorKind`] | `Call.func` is a `String` here; rustc/Lane A use a constant `FnDef` operand. `SwitchInt` targets are `(u128, BasicBlock)` pairs here vs. `SwitchTargets` there. |
//! | `Place { local, projection }`, `Operand::{Copy, Move, Constant}` | [`ir::Place`], [`ir::Operand`] | Same shape; only `Field` projection is executable yet. |
//! | `Rvalue::{Use, BinaryOp, CheckedBinaryOp, UnaryOp, Aggregate, Cast}` | [`ir::Rvalue`] | `Aggregate` is narrowed to `AggregateTuple`; other aggregate kinds are `Rvalue::Unsupported`. |
//! | `Ty` (full rustc type model) | [`ir::Ty`] | Reduced to scalars + tuples; no ADTs/references/generics yet. |
//!
//! Lane A has since landed, and the reconciliation path is implemented
//! in [`mir_convert`]: `sparagmos_mir::Body -> sparagmos_interp::ir::Body`,
//! kept inside this crate as the interpreter's internal adapter (MIR
//! itself arrives only via the Stable MIR driver, which produces
//! `sparagmos-mir` bodies). The interpreter core (frames, arithmetic,
//! limits) is unchanged by that layer.

pub mod builtins;
mod decode;
pub mod error;
pub mod interp;
pub mod ir;
pub mod mir_convert;
pub mod value;

pub use error::{InterpError, InterpResult};
pub use interp::{interpret, InterpConfig, Interpreter};
pub use ir::{
    BasicBlock, BasicBlockData, BinOp, Body, Constant, DropGlue, DropPlan, Local, Operand,
    Place, Program,
    ProjectionElem, Rvalue, Statement, StatementKind, TerminatorKind, Ty, UnOp, UnwindAction,
};
pub use mir_convert::{
    convert_program, interpret_mir, interpret_mir_captured,
    interpret_mir_captured_with_config, program_key_of, CapturedRun, ConvertError, MirFunction,
    MirProgram, MirRunError,
};
pub use value::{FmtArguments, FmtTrait, RefProj, RefTarget, Value};
