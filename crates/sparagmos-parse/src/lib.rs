//! # sparagmos-parse — textual MIR parser
//!
//! Parses the textual MIR emitted by `rustc --emit=mir` (reference toolchain:
//! **Rust 1.100.0**, released 2026-11-12) into this crate's own AST
//! ([`ast`]), with a thin conversion layer ([`convert`]) toward the assumed
//! `sparagmos-mir` model. The lexer and recursive-descent parser are fully
//! hand-written; no parser/lexer generator crates are used.
//!
//! ## Supported (tranche 1)
//!
//! - Whole files containing multiple `fn` bodies, `//` comments, the
//!   standard `// WARNING: ...` / `// HINT: ...` header lines.
//! - Function signatures: `fn name(_1: i32, _2: bool) -> i32 { ... }`,
//!   qualified paths (`a::b`), optional `const`/`async`/`unsafe` prefixes,
//!   generic parameter lists on the function name (skipped, balanced).
//! - Local declarations: `let mut _0: i32;` / `let _1: bool;`, including
//!   declarations nested inside `scope N { ... }` blocks.
//! - `debug name => _N;` bindings, including inside scopes (collected into
//!   [`ast::Body::debug_vars`]).
//! - Types: `bool`, `char`, `str`, all int/uint/float primitives, `()`,
//!   tuples, `&T` / `&mut T` (with lifetimes), `*const T` / `*mut T`,
//!   arrays `[T; len]`, slices `[T]`, never type `!`, and arbitrary named
//!   paths with generics preserved verbatim as [`ast::Ty::Named`].
//! - Basic blocks `bbN: { ... }` with statements and exactly one terminator.
//! - Assignment statements with rvalues:
//!   - constants: `const 1_i32`, `const -1_i32`, `const 2_u64`,
//!     `const true`/`false`, `const 1.5_f64`, char/string literals,
//!     `const ()`, named constants such as `const i32::MIN`;
//!   - `copy place` / `move place` (including bare places);
//!   - binary ops in call form: `Add`, `Sub`, `Mul`, `Div`, `Rem`,
//!     `BitXor`, `BitAnd`, `BitOr`, `Shl`, `Shr`, `Eq`, `Ne`, `Lt`, `Le`,
//!     `Gt`, `Ge`, `Offset`, and the `*WithOverflow` / `*Unchecked`
//!     variants;
//!   - unary ops: `Not`, `Neg`, `PtrMetadata`;
//!   - references: `&place`, `&mut place`, `&raw const/mut place`;
//!   - places with projections: field access `_3.0`, typed field
//!     projections `(_3.1: bool)`, deref `(*_1)`, index `_1[_2]`.
//!   - Other `Name(args...)` rvalues are preserved as
//!     [`ast::Rvalue::AggregateCall`] and parenthesised aggregates as
//!     [`ast::Rvalue::Raw`] — preserved, not silently dropped.
//! - Statements `StorageLive(_N);`, `StorageDead(_N);`, `nop;`.
//! - Terminators: `return;`, `unreachable;`, `goto -> bbN;`,
//!   `switchInt(operand) -> [0: bbA, otherwise: bbB];`,
//!   call terminators `_1 = func(args) -> [return: bbN, unwind continue];`
//!   (also `unwind: bbN` / `unwind unreachable` / `unwind terminate`),
//!   `assert(cond, "msg", ...) -> [success: bbN, unwind continue];`
//!   (including negated conditions `!move _2`), and
//!   `drop(place) -> [return: bbN, unwind continue];`.
//!
//! ## Not supported yet (each produces a line-numbered [`ParseError`])
//!
//! - Local declarations with an initializer (`let _0: i32 = ...;`).
//! - Named (non-numeric) field projections such as `_1.foo`.
//! - Top-level items other than functions (`static` / `const` items,
//!   promoted constants dumped as separate bodies).
//! - Inline assembly, `yield`/`await` (generator bodies), and
//!   `intrinsic` / `foreign` body shorthands.
//! - MIR dumps from `-Z dump-mir` per-pass files that use syntax beyond the
//!   `--emit=mir` final form (e.g. `Coverage` statements, `PlaceMention`,
//!   `AscribeUserType`, `Retag`, `FakeRead` statements).
//! - Anything unrecognised at all: the parser never silently skips unknown
//!   syntax inside a body; it errors with the 1-based line and column.
//!
//! ## Fixtures
//!
//! `tests/fixtures/*.mir` are real outputs of
//! `rustup run 1.100.0 rustc --edition 2021 --emit=mir` on the sibling
//! `*.rs` sources (see `tests/parse_fixtures.rs`); `binop.mir` additionally
//! used `-C overflow-checks=off` so plain (unchecked) binary ops appear.

pub mod ast;
pub mod convert;
pub mod error;
pub mod lexer;
pub mod parser;

pub use ast::MirFile;
pub use error::{ParseError, ParseResult};
pub use parser::parse_mir;
