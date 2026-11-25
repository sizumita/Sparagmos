//! Conversion from `sparagmos-mir` bodies into the interpreter's own IR,
//! plus a one-shot runner on top of it.
//!
//! This module is an internal adapter of the interpreter — it is *not* a
//! separate input path. On the project's main path, MIR reaches Sparagmos
//! exclusively through the Stable MIR driver (Lane I), which produces
//! [`sparagmos_mir::Body`] values; this layer only re-shapes those bodies
//! into the form [`crate::interp::Interpreter`] executes, inside this
//! crate. It is deliberately not a standalone public system.
//!
//! Conversion policy (the interpreter's honesty contract, extended):
//! * Kinds the interpreter's IR can mark as unsupported (rvalue /
//!   statement / terminator kinds) are converted into the corresponding
//!   `Unsupported` IR nodes, so execution fails loudly with context
//!   instead of conversion silently dropping them.
//! * Anything that cannot even be represented in the IR (local types,
//!   place projections, constant forms) fails conversion with a
//!   [`ConvertError`]. Nothing is ever silently skipped.
//!
//! Notable semantic choices:
//! * `mir` `BinaryOp` with a `*WithOverflow` operator and
//!   `CheckedBinaryOp` both become the IR's `CheckedBinaryOp` (they yield
//!   `(value, overflowed)` in rustc MIR as well).
//! * `*Unchecked` operators map to the plain IR operator: in valid MIR
//!   they only execute when no overflow occurs (overflow is UB there),
//!   so the oracle's checked behaviour is unobservable for them.
//! * Execution-only no-ops in rustc MIR (`FakeRead`, `PlaceMention`,
//!   `Retag`, `AscribeUserType`, `Coverage`, `ConstEvalCounter`) become
//!   IR `Nop`s, exactly as an executor is expected to treat them.
//! * `FalseEdge` / `FalseUnwind` go to their real target; the imaginary
//!   edge is never taken at run time.
//! * Unwind actions on `Call` / `Assert` / `Drop` / `InlineAsm` are
//!   preserved per block in the IR body (see [`ir::Body::unwind`]):
//!   a guest panic runs the cleanup blocks of each frame it unwinds
//!   through — the `Drop` terminators of the live locals, in MIR's
//!   own order — before the panic propagates, exactly as native
//!   unwinding does. A panic that escapes a cleanup aborts the run
//!   ([`InterpError::Abort`]).
//! * `AggregateKind::Array` becomes a tuple value (the IR has no distinct
//!   array value); `AggregateKind::Adt` becomes a variant + fields value.
//!   ADT aggregate operands are typed by the fields' declared types,
//!   recovered program-wide from `Field` projections (an aggregate
//!   rvalue itself carries no field types in this MIR form); without
//!   that, a non-negative literal in a signed field would be read as
//!   unsigned by the no-context scalar rule and the stored field would
//!   mismatch its declared type on first use.
//! * Calls to functions without converted bodies resolve through the
//!   driver's full name table: runtime-supported callees
//!   ([`crate::builtins`]) become `CallBuiltin`; anything else stays an
//!   unsupported call that names the function.
//! * Conversion is isolated per body: a function whose body cannot
//!   be converted does not fail the whole program. It is recorded
//!   as unconverted with its reason; calling it fails at execution
//!   time with that reason, while the entry function's own failure
//!   stays a whole-run conversion failure.
//! * Functions are keyed by name. Bodies that share one name —
//!   anonymous `const _` items, which the driver names by a single
//!   shared item path — are disambiguated by their `DefId`, which
//!   is also how references to them resolve (see [`program_keys`]).
//! * `Drop` converts when conversion can account for the dropped
//!   type's whole drop glue: plain data needs none, modelled std
//!   containers drop their contents, and a user type's glue is its
//!   converted `Drop` impl body (when the program has one) followed
//!   by its fields' glue. A type whose glue cannot be fully
//!   accounted for (a `Drop` impl without a converted body, a field
//!   whose type no projection anywhere records, a type from another
//!   crate that might hide an impl) keeps the terminator an explicit
//!   error; nothing is dropped silently. An explicit
//!   `std::mem::drop` call is treated the same way: the dropped
//!   type's glue is built at conversion and attached to the call, so
//!   both paths run identical glue.




mod adt_fields;
mod asm;
mod converter;
mod drop_glue;
mod dyn_unsize;
mod errors;
mod helpers;
mod operands;
mod places;
mod program;
mod resolve;
mod rvalues;
mod run;
mod statements;
mod transmute;
mod types;

pub use self::errors::{ConvertError, MirRunError};
pub use self::program::{program_key_of, MirFunction, MirProgram};
pub use self::resolve::conversions_performed;
pub use self::run::{
    convert_program, debug_mir, interpret_mir, interpret_mir_captured,
    interpret_mir_captured_with_config, interpret_mir_captured_with_report, CapturedRun,
};

pub(crate) use self::program::f16_to_f64;
