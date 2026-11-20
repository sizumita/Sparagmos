//! Runtime values manipulated by the interpreter.
//!
//! The representation is deliberately small but extensible: new variants
//! (references, aggregates, floats, ...) can be added without changing the
//! shape of the interpreter loop. The first cut is integer-centred, matching
//! the scope agreed for Lane C.

use std::fmt;

/// A value produced or consumed during interpretation.
///
/// Integers are stored widened (`i128` / `u128`); the *declared* width of a
/// local (see [`crate::ir::Ty`]) is applied by the interpreter when a value
/// is computed or stored, so an `i8` local never silently holds `200`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// A local that has not been written yet. Reading it is an error.
    Uninit,
    /// The unit value `()`.
    Unit,
    /// A boolean.
    Bool(bool),
    /// A signed integer, stored widened to `i128`.
    Int(i128),
    /// An unsigned integer, stored widened to `u128`.
    Uint(u128),
    /// A tuple / small aggregate, e.g. the result of a checked operation
    /// `(value, overflowed)` in rustc MIR.
    Tuple(Vec<Value>),
}

impl Value {
    /// Short human-readable kind name, used in error messages.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Value::Uninit => "uninitialized",
            Value::Unit => "unit",
            Value::Bool(_) => "bool",
            Value::Int(_) => "signed integer",
            Value::Uint(_) => "unsigned integer",
            Value::Tuple(_) => "tuple",
        }
    }

    /// View as a signed integer, if this value is one.
    pub fn as_int(&self) -> Option<i128> {
        match self {
            Value::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// View as an unsigned integer, if this value is one.
    pub fn as_uint(&self) -> Option<u128> {
        match self {
            Value::Uint(v) => Some(*v),
            _ => None,
        }
    }

    /// View as a boolean, if this value is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Bit pattern used by `SwitchInt` target matching: signed values are
    /// compared by their two's-complement representation, booleans as 0/1.
    pub fn switch_bits(&self) -> Option<u128> {
        match self {
            Value::Int(v) => Some(*v as u128),
            Value::Uint(v) => Some(*v),
            Value::Bool(b) => Some(*b as u128),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Uninit => write!(f, "<uninit>"),
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(v) => write!(f, "{v}"),
            Value::Uint(v) => write!(f, "{v}"),
            Value::Tuple(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, ")")
            }
        }
    }

    // NOTE: Display is implemented manually above; see `kind_name` for
    // diagnostics. This impl block intentionally has no other members.
}
