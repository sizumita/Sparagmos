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
//! The main entry point is [`mir_jit::JitProgram`]: it lowers
//! `sparagmos-mir` bodies to the backend IR and compiles a whole
//! program (multiple functions, calls between them, guest panics)
//! through [`program`]. The single-function IR API ([`compile`],
//! [`CompiledFunction`]) remains for the backend's own use and tests.

pub mod builtins;
pub mod compiler;
pub mod emitter;
pub mod ir;
pub mod memory;
pub mod mir_jit;
pub mod program;

pub use compiler::{compile, compile_to_bytes, CompileError, CompiledFunction};
pub use emitter::{CodeBuffer, Cond, Emitter, Reg, Xmm};
pub use ir::{
    BasicBlock, BinOp, BlockId, FloatBinOp, FloatKind, Function, FunctionBuilder, Instr, IntKind,
    Operand, Terminator, UnOp, Value,
};
pub use builtins::{Builtin, BuiltinCtx, BuiltinTables, FloatOp, FmtDescriptor, FmtTrait, FmtTy};
pub use memory::JitMemory;
pub use mir_jit::{JitCallError, JitError, JitProgram, MirFunctionInput, MirProgramInput};
pub use program::CompiledProgram;
pub use sparagmos_runtime::error::std_default_panic_report;
