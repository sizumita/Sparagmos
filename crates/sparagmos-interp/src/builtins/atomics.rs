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
