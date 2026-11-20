//! sparagmos-runtime: runtime support for the Sparagmos MIR JIT/interpreter.
pub mod error;
pub mod layout;
pub use error::{PanicRecord, RuntimeError, RuntimeResult};
pub use layout::{layout_of, Layout, Ty};
