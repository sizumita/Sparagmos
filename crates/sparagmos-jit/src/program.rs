//! Program-level compilation: a table of IR functions compiled into
//! one executable image, with calls between them and a guest-panic
//! path.
//!
//! All functions are laid out in a single [`JitMemory`]; the `call
//! rel32` fixups each function produced are patched once every
//! function's offset is known. Functions use the plain System V
//! signature `fn(i64, ...) -> i64` (at most six arguments), so a
//! compiled program's entry points can be called from Rust directly.
//!
//! Guest panics: generated code for [`crate::ir::Instr::Panic`] calls
//! the runtime's `sparagmos_panic` entry point (which records the
//! panic in `sparagmos-runtime`, the shared contract), stores the
//! message index in the program's own panic context, and returns from
//! the panicking function immediately. Every generated call site is
//! followed by [`crate::ir::Instr::CheckPanic`], which returns early
//! in turn when the flag is set, so the panic unwinds the JIT call
//! stack frame by frame until [`CompiledProgram::call`] observes the
//! flag and reports a [`PanicRecord`]. There are no DWARF unwind
//! tables; the flag is the whole mechanism, which is exactly as much
//! as the interpreter's model (a guest panic aborts the run) needs.

use std::cell::Cell;

use sparagmos_runtime::error::{clear_last_panic, PanicRecord};

use crate::compiler::{compile_with_ctx, CompileError, ProgramCtx};
use crate::ir::Function;
use crate::memory::JitMemory;

/// The panic context generated code talks to. Its address is embedded
/// into the code as an immediate, so it is boxed (stable address for
/// the lifetime of the [`CompiledProgram`]) and word-sized for the
/// generated loads/stores.
#[repr(C)]
struct PanicCtx {
    panicked: Cell<u64>,
    message_index: Cell<u64>,
}

/// Message index the generated code stores for a panic whose text
/// was formatted at run time (an index-out-of-bounds message): the text
/// lives in the program's scratch buffer, not the message table.
const DYNAMIC_MESSAGE: u64 = u64::MAX;

/// The index-out-of-bounds message formatter the generated code
/// calls: same buffer contract as [`format_index_message`]. The
/// wording matches the interpreter's message for an index outside
/// an array or slice: both numbers are the runtime length and
/// index, read as `usize` values.
///
/// # Safety
/// `buf` must point to at least 128 writable bytes (the program's
/// scratch buffer); the message is well under that.
extern "C" fn format_index_message(len: u64, index: u64, buf: *mut u8) -> usize {
    let text = format!("index out of bounds: the len is {len} but the index is {index}");
    let bytes = text.as_bytes();
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.len().to_le_bytes().as_ptr(), buf, 8);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.add(8), bytes.len());
    }
    bytes.len()
}

/// A whole compiled program: executable image plus the state its
/// generated code refers to by absolute address.
pub struct CompiledProgram {
    mem: JitMemory,
    /// Byte offset of each function inside the image.
    offsets: Vec<usize>,
    param_counts: Vec<usize>,
    /// Kept alive (and address-stable) for the generated code.
    panic_ctx: Box<PanicCtx>,
    /// The panic message strings; generated code passes pointers to
    /// these bytes to `sparagmos_panic`, so they must not move.
    #[allow(dead_code)]
    messages: Vec<Box<str>>,
    /// Scratch buffer for run-time-formatted panic messages
    /// (address-stable; see [`format_index_message`]).
    scratch_buf: Box<[u8; 128]>,
}

impl CompiledProgram {
    /// Address of function `idx`'s entry point.
    pub fn function_ptr(&self, idx: usize) -> *const u8 {
        self.mem.as_ptr().wrapping_add(self.offsets[idx])
    }

    pub fn function_count(&self) -> usize {
        self.offsets.len()
    }

    /// Call function `idx` with `args` (at most six, exactly the
    /// function's parameter count). Returns the function's result, or
    /// the [`PanicRecord`] if the guest panicked anywhere in the call
    /// tree.
    ///
    /// # Panics
    /// Panics if `idx` is out of range or `args.len()` differs from
    /// the function's parameter count — both are caller bugs, not
    /// guest behaviour.
    pub fn call(&self, idx: usize, args: &[i64]) -> Result<i64, PanicRecord> {
        assert!(idx < self.offsets.len(), "function index {idx} out of range");
        assert_eq!(
            args.len(),
            self.param_counts[idx],
            "argument count mismatch for function {idx}"
        );
        self.panic_ctx.panicked.set(0);
        self.panic_ctx.message_index.set(0);
        clear_last_panic();
        let mut padded = [0i64; 6];
        padded[..args.len()].copy_from_slice(args);
        let f: unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 =
            unsafe { std::mem::transmute(self.function_ptr(idx)) };
        let result = unsafe { f(padded[0], padded[1], padded[2], padded[3], padded[4], padded[5]) };
        if self.panic_ctx.panicked.get() != 0 {
            let msg_idx = self.panic_ctx.message_index.get();
            let message = if msg_idx == DYNAMIC_MESSAGE {
                let len = u64::from_le_bytes(self.scratch_buf[0..8].try_into().unwrap()) as usize;
                String::from_utf8_lossy(&self.scratch_buf[8..8 + len]).into_owned()
            } else {
                self.messages
                    .get(msg_idx as usize)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "<unknown panic message>".to_string())
            };
            return Err(PanicRecord {
                message,
                location: None,
            });
        }
        Ok(result)
    }
}

/// Compile a table of IR functions plus their panic message table
/// into one executable program.
pub fn compile_program(
    funcs: &[Function],
    messages: Vec<String>,
) -> Result<CompiledProgram, CompileError> {
    // Everything the generated code refers to by absolute address is
    // created first and kept in the returned program: the panic
    // context (boxed) and the message strings (each boxed).
    let panic_ctx = Box::new(PanicCtx {
        panicked: Cell::new(0),
        message_index: Cell::new(0),
    });
    let messages: Vec<Box<str>> = messages.into_iter().map(|s| s.into_boxed_str()).collect();
    let scratch_buf: Box<[u8; 128]> = Box::new([0; 128]);
    let ctx = ProgramCtx {
        flag_addr: panic_ctx.as_ref() as *const PanicCtx as usize,
        panic_fn: sparagmos_runtime::ffi::sparagmos_panic as usize,
        messages: messages
            .iter()
            .map(|s| (s.as_ptr() as usize, s.len()))
            .collect(),
        index_fmt_fn: format_index_message as usize,
        scratch_buf: scratch_buf.as_ptr() as usize,
    };

    // Compile each function separately, then lay the bodies out in
    // one image and patch the inter-function call fixups.
    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(funcs.len());
    let mut fixups: Vec<(usize, usize, u32)> = Vec::new(); // (fn idx, field off, callee idx)
    for (i, f) in funcs.iter().enumerate() {
        let (bytes, calls) = compile_with_ctx(f, &ctx)?;
        for (at, callee) in calls {
            if (callee as usize) >= funcs.len() {
                return Err(CompileError::UnknownFunction(callee));
            }
            fixups.push((i, at, callee));
        }
        bodies.push(bytes);
    }
    let mut offsets = Vec::with_capacity(funcs.len());
    let mut image: Vec<u8> = Vec::new();
    for body in &bodies {
        while !image.len().is_multiple_of(16) {
            image.push(0xCC); // int3 padding between functions
        }
        offsets.push(image.len());
        image.extend_from_slice(body);
    }
    for (fn_idx, at, callee) in fixups {
        let field = offsets[fn_idx] + at;
        let target = offsets[callee as usize];
        let rel = target as i64 - (field as i64 + 4);
        debug_assert!(rel >= i32::MIN as i64 && rel <= i32::MAX as i64);
        image[field..field + 4].copy_from_slice(&(rel as i32).to_le_bytes());
    }

    let mut mem = JitMemory::new(image.len())?;
    mem.write_code(&image)?;
    mem.make_executable()?;
    Ok(CompiledProgram {
        mem,
        offsets,
        param_counts: funcs.iter().map(|f| f.param_count).collect(),
        panic_ctx,
        messages,
        scratch_buf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp as B, FunctionBuilder, Instr, Operand, Terminator};

    fn caller_and_callee() -> (Vec<Function>, Vec<String>) {
        // fn callee(x) = x * 3
        let mut b = FunctionBuilder::new("callee", 1);
        let x = b.param(0);
        let r = b.binop(B::Mul, x, Operand::Const(3));
        let callee = b.finish(Terminator::Return(r.into()));
        // fn caller(x) = callee(x + 1) + callee(10)
        let mut b = FunctionBuilder::new("caller", 1);
        let x = b.param(0);
        let xp1 = b.binop(B::Add, x, Operand::Const(1));
        let c1 = b.fresh();
        b.push_raw(Instr::Call {
            dst: c1,
            func: 0,
            args: vec![xp1.into()],
        });
        b.push_raw(Instr::CheckPanic);
        let c2 = b.fresh();
        b.push_raw(Instr::Call {
            dst: c2,
            func: 0,
            args: vec![Operand::Const(10)],
        });
        b.push_raw(Instr::CheckPanic);
        let sum = b.binop(B::Add, c1, c2);
        let caller = b.finish(Terminator::Return(sum.into()));
        (vec![callee, caller], vec![])
    }

    #[test]
    fn call_between_functions() {
        let (funcs, msgs) = caller_and_callee();
        let prog = compile_program(&funcs, msgs).unwrap();
        // caller(4) = callee(5) + callee(10) = 15 + 30
        assert_eq!(prog.call(1, &[4]), Ok(45));
        assert_eq!(prog.call(0, &[7]), Ok(21));
    }

    #[test]
    fn panic_propagates_through_calls() {
        // fn boom() = panic("kaboom"); fn mid() = boom(); fn top() = mid()
        let mut b = FunctionBuilder::new("boom", 0);
        b.push_raw(Instr::Panic { msg: 0 });
        let boom = b.finish(Terminator::Unreachable);
        let mk_caller = |name: &str, callee: u32| {
            let mut b = FunctionBuilder::new(name, 0);
            let d = b.fresh();
            b.push_raw(Instr::Call {
                dst: d,
                func: callee,
                args: vec![],
            });
            b.push_raw(Instr::CheckPanic);
            b.finish(Terminator::Return(d.into()))
        };
        let mid = mk_caller("mid", 0);
        let top = mk_caller("top", 1);
        let prog =
            compile_program(&[boom, mid, top], vec!["kaboom".to_string()]).unwrap();
        let err = prog.call(2, &[]).expect_err("panic must surface");
        assert_eq!(err.message, "kaboom");
        // The runtime's shared panic record was written too.
        assert_eq!(
            sparagmos_runtime::error::last_panic().map(|p| p.message),
            Some("kaboom".to_string())
        );
    }
}
