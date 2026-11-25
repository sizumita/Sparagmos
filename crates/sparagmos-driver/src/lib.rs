//! Sparagmos rustc driver library: MIR acquisition via Stable MIR
//! (`rustc_public`) and conversion into `sparagmos-mir` types.
//!
//! This is the *only* MIR input path of Sparagmos (project rule: no
//! alternative/fallback input systems). The driver must be built with the
//! pinned Rust 1.100.0 toolchain and its `rustc-dev` component:
//!
//! ```text
//! RUSTC_BOOTSTRAP=1 cargo +1.100.0 build   # in this crate's directory
//! ```
//!
//! `RUSTC_BOOTSTRAP=1` is required because `#![feature(rustc_private)]`
//! is a feature gate and 1.100.0 is a release toolchain. The build script
//! captures the building rustc's sysroot, links the driver against that
//! sysroot's `librustc_driver` (rpath), and the binary passes the same
//! sysroot to the compiler it drives, so it always operates on the
//! 1.100.0 sysroot regardless of the caller's `PATH`.
//!
//! Direct `rustc_private` usage is intentionally minimal: everything goes
//! through `rustc_public` (Stable MIR). The internal crates declared are
//! the ones the `rustc_public::run_with_tcx!` macro expansion itself
//! requires (`rustc_driver`, `rustc_interface`, `rustc_middle`,
//! `rustc_span`). Two internal facilities are used directly, by design:
//! the debugger's statement lines are resolved through
//! `rustc_span::hygiene::walk_chain_collapsed` (via the
//! `rustc_public::rustc_internal` span conversions), because that
//! collapsed attribution is what the compiler's own debuginfo
//! records and the public span API cannot express it; and
//! `rustc_crate_store`, `rustc_session`, `rustc_structures` and
//! `rustc_target` name the
//! crate-graph types (dependency kinds, crate types, panic
//! strategies) the monomorphization verdicts in `mono_check` read
//! from the compiler's own queries, mirroring the dependency-format
//! validation code generation performs; and
//! `rustc_codegen_ssa::back::symbol_export` names the collector
//! root set's own computation, whose query form is gated on the
//! session code-generating — which this driver's metadata-only
//! session never does — so `mono_check` applies the provider's
//! ungated helper to the same export threshold; and
//! `rustc_attr_ir::lang_items::LangItem` names the `drop_glue` lang
//! item so drop-glue certification can check the item exists before
//! asking the compiler to resolve `drop_in_place` (a minicore
//! defines no `drop_glue` item, and asking emits a fatal diagnostic
//! no `catch_unwind` can contain); and the
//! `rustc_deny_explicit_impl` attribute and the lang items naming
//! core's compiler-implemented traits are read through the internal
//! context so the unnormalizable-projection screen can exempt those
//! traits, whose implementations no `impl` item ever records.

#![feature(rustc_private)]

extern crate rustc_attr_ir;
extern crate rustc_codegen_ssa;
extern crate rustc_crate_store;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public_bridge;
extern crate rustc_public;
extern crate rustc_session;
extern crate rustc_structures;
extern crate rustc_span;
extern crate rustc_target;

pub mod convert;
pub mod dump;
pub mod mono_check;
pub mod reach;
pub mod report;
