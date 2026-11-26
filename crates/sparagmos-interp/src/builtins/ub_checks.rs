//! The `core::ub_checks` runtime predicates (and the
//! raw-pointer alignment predicate they bottom out in), served
//! natively: conversion diverts calls to these helpers — and to
//! the `assert_unsafe_precondition!` shims of `ptr::copy` and
//! `ptr::copy_nonoverlapping`, whose bodies only compose them —
//! into a builtin node (see `ub_check_terminator` in
//! `mir_convert`) because the collected bodies spend dozens of
//! statements re-deriving, per call, a predicate over the
//! arguments' priced addresses — and the runtime checks the
//! converter resolves to `true` put the helpers inside the
//! precondition checks of the standard library's unsafe
//! operations, so hot code calls them per element copy.

use crate::error::{InterpError, InterpResult};
use crate::interp::{Frame, Interpreter, PanicOrigin};
use crate::ir::{Body, Operand};
use crate::value::{RefTarget, Value};

use super::*;

/// The library source prefix the pinned 1.100.0 build's panic
/// locations carry (the same stamp the other hardcoded library
/// locations in this crate use).
const RUSTC_SRC: &str = "/rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library";

/// The target's `usize` bound: the guest targets are 64-bit, so
/// the overflow the `is_nonoverlapping` predicate checks for is
/// a product past `u64::MAX`.
const USIZE_MAX: u128 = u64::MAX as u128;

/// The panic location of `*const T::is_aligned_to`'s
/// not-a-power-of-two panic in the pinned library sources.
const CONST_PTR_ALIGN_PANIC: (&str, u128, u128) = ("core/src/ptr/const_ptr.rs", 1358, 13);

/// The same panic in `*mut T::is_aligned_to`.
const MUT_PTR_ALIGN_PANIC: (&str, u128, u128) = ("core/src/ptr/mut_ptr.rs", 1627, 13);

impl<'a> Interpreter<'a> {
    /// Execute one diverted ub-check helper call (see
    /// [`ub_check_helper`]). `name` is the canonical callee name;
    /// `args` and `vals` are the call's operands and their
    /// evaluated values. Each predicate is computed exactly as
    /// the collected body computes it: pointer operands are
    /// priced with the same address model (and the same exposure
    /// side effect) as the pointer-to-integer casts the body
    /// performs, in the body's own order, and the failure cases
    /// raise the body's own panic with the body's own message
    /// and location.
    pub(super) fn exec_ub_check_helper(
        &mut self,
        frames: &[Frame],
        body: &Body,
        name: &str,
        helper: UbCheckHelper,
        args: &[Operand],
        vals: Vec<Value>,
    ) -> InterpResult<Value> {
        match helper {
            UbCheckHelper::Nonoverlapping => {
                // `maybe_is_nonoverlapping::runtime(src, dst,
                // size, count)`.
                let size = ub_check_usize(&vals, 2)?;
                let count = ub_check_usize(&vals, 3)?;
                let overlaps =
                    self.ub_check_nonoverlapping(frames, body, args, &vals, size, count)?;
                Ok(Value::Bool(!overlaps))
            }
            UbCheckHelper::IsAlignedTo => {
                // `*const T::is_aligned_to(align)` /
                // `*mut T::is_aligned_to(align)`.
                let align = ub_check_usize(&vals, 1)?;
                let stamp = if name.contains("mut_ptr") {
                    MUT_PTR_ALIGN_PANIC
                } else {
                    CONST_PTR_ALIGN_PANIC
                };
                let aligned =
                    self.ub_check_is_aligned(frames, body, args, &vals, 0, align, stamp)?;
                Ok(Value::Bool(aligned))
            }
            UbCheckHelper::SelectUnpredictable => {
                // `hint::select_unpredictable(cond, true_val,
                // false_val)`: the chosen argument, unmodified.
                match vals.first() {
                    Some(Value::Bool(true)) => Ok(vals[1].clone()),
                    Some(Value::Bool(false)) => Ok(vals[2].clone()),
                    other => Err(InterpError::type_mismatch(
                        "select_unpredictable",
                        "a boolean condition",
                        format!("{other:?}"),
                    )),
                }
            }
            UbCheckHelper::CopyNonoverlappingPrecondition => {
                // The shim of `ptr::copy_nonoverlapping` (args:
                // src, dst, size, align, count): both pointers
                // aligned and non-null — unless the copy is
                // zero-sized — and the ranges non-overlapping,
                // composed in the shim body's own order.
                let size = ub_check_usize(&vals, 2)?;
                let align = ub_check_usize(&vals, 3)?;
                let count = ub_check_usize(&vals, 4)?;
                let zero_size = count == 0 || size == 0;
                let message = "ptr::copy_nonoverlapping requires that both pointer arguments are aligned and non-null and the specified memory ranges do not overlap";
                if !self.ub_check_aligned_and_not_null(
                    frames,
                    body,
                    args,
                    &vals,
                    0,
                    align,
                    zero_size,
                    CONST_PTR_ALIGN_PANIC,
                )? || !self.ub_check_aligned_and_not_null(
                    frames,
                    body,
                    args,
                    &vals,
                    1,
                    align,
                    zero_size,
                    MUT_PTR_ALIGN_PANIC,
                )? {
                    return Err(self.ub_check_precondition_panic(message));
                }
                if self.ub_check_nonoverlapping(frames, body, args, &vals, size, count)? {
                    return Err(self.ub_check_precondition_panic(message));
                }
                Ok(Value::Unit)
            }
            UbCheckHelper::CopyPrecondition => {
                // The shim of `ptr::copy` (args: src, dst,
                // align, zero_size): both pointers aligned and
                // non-null, unless the copy is zero-sized.
                let align = ub_check_usize(&vals, 2)?;
                let zero_size = match vals.get(3) {
                    Some(Value::Bool(b)) => *b,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "ptr::copy precondition check",
                            "a boolean zero-size flag",
                            format!("{other:?}"),
                        ))
                    }
                };
                if !self.ub_check_aligned_and_not_null(
                    frames,
                    body,
                    args,
                    &vals,
                    0,
                    align,
                    zero_size,
                    CONST_PTR_ALIGN_PANIC,
                )? || !self.ub_check_aligned_and_not_null(
                    frames,
                    body,
                    args,
                    &vals,
                    1,
                    align,
                    zero_size,
                    MUT_PTR_ALIGN_PANIC,
                )? {
                    return Err(self.ub_check_precondition_panic(
                        "ptr::copy requires that both pointer arguments are aligned and non-null",
                    ));
                }
                Ok(Value::Unit)
            }
        }
    }

    /// `maybe_is_nonoverlapping`'s runtime predicate, as a
    /// positive overlap answer: the arguments at indices 0 and
    /// 1, priced in the body's own order (source cast,
    /// destination cast, then the product), overlap iff
    /// `src.addr().abs_diff(dst.addr()) < size * count`. The
    /// product overflowing a usize is the body's own nounwind
    /// panic.
    fn ub_check_nonoverlapping(
        &mut self,
        frames: &[Frame],
        body: &Body,
        args: &[Operand],
        vals: &[Value],
        size: u128,
        count: u128,
    ) -> InterpResult<bool> {
        let src = self.ub_check_pointer_addr(frames, body, args, vals, 0)?;
        let dst = self.ub_check_pointer_addr(frames, body, args, vals, 1)?;
        let total = size * count;
        if total > USIZE_MAX {
            self.note_panic_stamp(core_location("core/src/ub_checks.rs", 174, 17));
            self.note_panic_origin(PanicOrigin::PanicFmt);
            self.note_panic_no_backtrace(false);
            return Err(InterpError::panic_str_payload(
                "is_nonoverlapping: `size_of::<T>() * count` overflows a usize",
            ));
        }
        Ok(src.abs_diff(dst) < total)
    }

    /// `*const T::is_aligned_to` / `*mut T::is_aligned_to` for
    /// the pointer argument at `idx`: a plain panic when `align`
    /// is not a power of two — raised before the pointer is
    /// priced, as the body raises it before its `addr()` — else
    /// `ptr.addr() & (align - 1) == 0`. `stamp` is the panic
    /// location of the impl the argument's pointer type names.
    fn ub_check_is_aligned(
        &mut self,
        frames: &[Frame],
        body: &Body,
        args: &[Operand],
        vals: &[Value],
        idx: usize,
        align: u128,
        stamp: (&str, u128, u128),
    ) -> InterpResult<bool> {
        if !align.is_power_of_two() {
            self.note_panic_stamp(core_location(stamp.0, stamp.1, stamp.2));
            self.note_panic_origin(PanicOrigin::PanicFmt);
            self.note_panic_no_backtrace(false);
            return Err(InterpError::panic_str_payload(
                "is_aligned_to: align is not a power-of-two",
            ));
        }
        let addr = self.ub_check_pointer_addr(frames, body, args, vals, idx)?;
        Ok(addr & (align - 1) == 0)
    }

    /// `maybe_is_aligned_and_not_null`'s runtime predicate for
    /// the pointer argument at `idx`: the alignment check first
    /// (which prices the pointer), then — unless the operation
    /// is zero-sized — the null check, which compares the priced
    /// address against zero exactly as the body's own `addr()`
    /// comparison does.
    fn ub_check_aligned_and_not_null(
        &mut self,
        frames: &[Frame],
        body: &Body,
        args: &[Operand],
        vals: &[Value],
        idx: usize,
        align: u128,
        zero_size: bool,
        stamp: (&str, u128, u128),
    ) -> InterpResult<bool> {
        if !self.ub_check_is_aligned(frames, body, args, vals, idx, align, stamp)? {
            return Ok(false);
        }
        if zero_size {
            return Ok(true);
        }
        let addr = self.ub_check_pointer_addr(frames, body, args, vals, idx)?;
        Ok(addr != 0)
    }

    /// A precondition shim's failure: the shim's own nounwind
    /// panic. The payload is the full text the
    /// `assert_unsafe_precondition!` macro generates — the
    /// "unsafe precondition(s) violated: " prefix, the
    /// operation's message, and the fixed suffix — carried as a
    /// `&str`, exactly as the shim's
    /// `panic_nounwind_fmt(Arguments::from_str(msg))` carries
    /// it. The shim is `#[track_caller]`, so the panic's
    /// location is the call-site stamp the call itself carries,
    /// and `#[rustc_nounwind]`, so the panic aborts (conversion
    /// also marks the call block `Terminate` for these helpers).
    fn ub_check_precondition_panic(&mut self, message: &'static str) -> InterpError {
        self.note_panic_origin(PanicOrigin::PanicFmt);
        self.note_panic_no_backtrace(false);
        InterpError::panic_str_payload(format!(
            "unsafe precondition(s) violated: {message}\n\nThis indicates a bug in the program. This Undefined Behavior check is optional, and cannot be relied on for safety."
        ))
    }

    /// The `addr()` of one pointer argument of a diverted helper
    /// call: priced with the argument's pointee type, exactly as
    /// the pointer-to-integer cast the collected body performs,
    /// including the cast's exposure side effect.
    fn ub_check_pointer_addr(
        &self,
        frames: &[Frame],
        body: &Body,
        args: &[Operand],
        vals: &[Value],
        idx: usize,
    ) -> InterpResult<u128> {
        let value = &vals[idx];
        let leaf = crate::interp::operand_pointee_ty(body, &args[idx]);
        match self.pointer_address(frames, value, leaf.as_ref()) {
            Some(addr) => {
                self.note_exposed_target(addr as u64, value);
                Ok(addr)
            }
            None => Err(InterpError::type_mismatch(
                "ub-check helper",
                "a pointer with a priceable address",
                format!("{value:?}"),
            )),
        }
    }
}

/// One `usize` argument of a diverted helper call.
fn ub_check_usize(vals: &[Value], idx: usize) -> InterpResult<u128> {
    match vals.get(idx) {
        Some(Value::Uint(v)) => Ok(*v),
        other => Err(InterpError::type_mismatch(
            "ub-check helper",
            "a usize argument",
            format!("{other:?}"),
        )),
    }
}

/// A `core::panic::Location` value for one of the library
/// source positions the diverted helpers panic from, in the
/// stamp form the panic machinery carries.
fn core_location(file: &str, line: u128, col: u128) -> Value {
    Value::Ref(RefTarget::value(Value::Enum {
        variant: 0,
        fields: vec![
            Value::Str(format!("{RUSTC_SRC}/{file}")),
            Value::Uint(line),
            Value::Uint(col),
        ],
    }))
}
