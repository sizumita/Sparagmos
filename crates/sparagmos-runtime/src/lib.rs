//! sparagmos-runtime: runtime support for the Sparagmos MIR JIT/interpreter.
pub mod alloc;
pub mod error;
pub mod ffi;
pub mod layout;
pub use alloc::{Allocator, Arena, SystemAllocator};
pub use error::{PanicRecord, RuntimeError, RuntimeResult};
pub use layout::{layout_of, Layout, Ty};
