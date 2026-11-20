//! sparagmos-jit: Sparagmos's own x86_64 baseline JIT.
//!
//! This crate is the code-generation core of Sparagmos. It is written from
//! scratch: machine code is encoded byte by byte in [`emitter`], executable
//! pages are managed in [`memory`], and [`compiler`] lowers a small
//! function IR ([`ir`]) to native code under the System V AMD64 ABI.
//! No existing code generator (Cranelift / LLVM / Wasmtime / ...) is used.
//!
//! The `libc` crate is used only for OS primitives (mmap/mprotect/munmap)
//! in [`memory`] — an OS foundation, not a code generator. See that module
//! for the note.
//!
//! Status / next connection point: Lane A's MIR types (against Rust
//! 1.100.0's MIR) are still being finalised, so the compiler's current
//! input is the crate-local IR in [`ir`]. Once the MIR types are fixed,
//! the plan is a MIR -> IR lowering that feeds this backend unchanged
//! (emitter and memory layers are IR-agnostic).

pub mod compiler;
pub mod emitter;
pub mod ir;
pub mod memory;

pub use compiler::{compile, compile_to_bytes, CompileError, CompiledFunction};
pub use emitter::{CodeBuffer, Cond, Emitter, Reg};
pub use ir::{
    BasicBlock, BinOp, BlockId, Function, FunctionBuilder, Instr, Operand, Terminator, UnOp, Value,
};
pub use memory::JitMemory;
