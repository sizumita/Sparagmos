//! sparagmos-runtime: runtime support for the Sparagmos MIR JIT/interpreter.
//!
//! Provides, with no external codegen/JIT dependency:
//! - [`layout`]: naive size/alignment computation for MIR types
//! - [`alloc`]: allocator layer (leaking bump [`alloc::Arena`] v1, swappable;
//!   [`alloc::SystemAllocator`] as the real-dealloc alternative)
//! - [`error`]: [`error::RuntimeError`] panic/error propagation shared by JIT + interpreter
//! - [`intrinsic`]: name -> implementation registry (unknown = explicit error)
//! - [`ffi`]: `extern "C"` entry points (`sparagmos_alloc`, ...) for JIT-generated code
pub mod alloc;
pub mod error;
pub mod ffi;
pub mod intrinsic;
pub mod layout;

pub use alloc::{Allocator, Arena, SystemAllocator};
pub use error::{PanicRecord, RuntimeError, RuntimeResult};
pub use intrinsic::{default_registry, IntrinsicRegistry, Value};
pub use layout::{layout_of, Layout, Ty};
