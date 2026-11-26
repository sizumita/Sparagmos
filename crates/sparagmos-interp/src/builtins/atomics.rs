//! Atomic value equality and write-back for the atomic builtins.



use crate::error::InterpResult;
use crate::interp::{
    Frame, Interpreter, Resolved,
};
use crate::value::Value;

use super::*;

impl<'a> Interpreter<'a> {
    /// The equality an atomic compare-exchange performs: the
    /// pointee's current value against the expected one.
    /// Pointer-shaped values compare by identity — the
    /// size-aware synthetic address, the same identity
    /// `PtrToInt` exposes — and every other value by the
    /// semantic (derived) equality.
    pub(super) fn atomic_value_eq(
        &self,
        frames: &[Frame],
        current: &Value,
        expected: &Value,
    ) -> InterpResult<bool> {
        let current = unwrap_wrapper(current);
        let expected = unwrap_wrapper(expected);
        match (
            self.pointer_address(frames, current, None),
            self.pointer_address(frames, expected, None),
        ) {
            (Some(a), Some(b)) => Ok(a == b),
            _ => self.values_equal(frames, current, expected),
        }
    }
    /// Write an atomic cell's new scalar, preserving the wrapper
    /// aggregates the cell place holds (`UnsafeCell` inside the
    /// atomic struct): the pointer names the wrapper place, so a
    /// bare scalar write would replace the aggregate and later
    /// whole-place reads of the cell would see the wrong shape.
    /// When the current contents cannot be read (an uninitialised
    /// cell), the value is written as-is.
    pub(super) fn atomic_write(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: Value,
    ) -> InterpResult<()> {
        let value = match self.read_resolved(frames, target) {
            Ok(current) => rewrap_like(&current, value),
            Err(_) => value,
        };
        self.write_resolved(frames, target, value)
    }
}

/// Registration fragment of the builtin name registry: the atomic intrinsics.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn atomics_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::intrinsics::atomic_load" | "core::intrinsics::atomic_load" => Builtin::AtomicLoad,
        "std::intrinsics::atomic_store" | "core::intrinsics::atomic_store" => {
            Builtin::AtomicStore
        }
        "std::intrinsics::atomic_cxchg" | "core::intrinsics::atomic_cxchg" => {
            Builtin::AtomicCxchg
        }
        "std::intrinsics::atomic_cxchgweak" | "core::intrinsics::atomic_cxchgweak" => {
            Builtin::AtomicCxchg
        }
        "std::intrinsics::atomic_xchg" | "core::intrinsics::atomic_xchg" => Builtin::AtomicXchg,
        "std::intrinsics::atomic_xadd" | "core::intrinsics::atomic_xadd" => Builtin::AtomicXadd,
        "std::intrinsics::atomic_xsub" | "core::intrinsics::atomic_xsub" => Builtin::AtomicXsub,
        "std::intrinsics::atomic_and" | "core::intrinsics::atomic_and" => {
            Builtin::AtomicFetchOp(FetchOp::And)
        }
        "std::intrinsics::atomic_or" | "core::intrinsics::atomic_or" => {
            Builtin::AtomicFetchOp(FetchOp::Or)
        }
        "std::intrinsics::atomic_xor" | "core::intrinsics::atomic_xor" => {
            Builtin::AtomicFetchOp(FetchOp::Xor)
        }
        "std::intrinsics::atomic_nand" | "core::intrinsics::atomic_nand" => {
            Builtin::AtomicFetchOp(FetchOp::Nand)
        }
        "std::intrinsics::atomic_max" | "core::intrinsics::atomic_max" => {
            Builtin::AtomicFetchOp(FetchOp::Max)
        }
        "std::intrinsics::atomic_min" | "core::intrinsics::atomic_min" => {
            Builtin::AtomicFetchOp(FetchOp::Min)
        }
        "std::intrinsics::atomic_umax" | "core::intrinsics::atomic_umax" => {
            Builtin::AtomicFetchOp(FetchOp::UMax)
        }
        "std::intrinsics::atomic_umin" | "core::intrinsics::atomic_umin" => {
            Builtin::AtomicFetchOp(FetchOp::UMin)
        }
        "std::intrinsics::atomic_fence" | "core::intrinsics::atomic_fence" => {
            Builtin::AtomicFence
        }
        _ => return None,
    })
}
