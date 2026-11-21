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
//! through `rustc_public` (Stable MIR). The only internal crates touched
//! are the ones the `rustc_public::run!` macro expansion itself requires
//! to be declared (`rustc_driver`, `rustc_interface`, `rustc_middle`,
//! `rustc_span`); no internal compiler types are used directly.

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public_bridge;
extern crate rustc_public;
extern crate rustc_span;

pub mod convert;
pub mod dump;
pub mod mono_check;
pub mod reach;
pub mod report;
