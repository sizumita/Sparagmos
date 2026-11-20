//! The interpreter's value model.
//!
//! Scalars are stored widened (`i128` / `u128`); the declared width
//! and signedness of a place are applied from its [`crate::ir::Ty`]
//! when the value is written (see `normalize_for_ty` in `interp`).
//!
//! Composite and reference values (added for the print/format path):
//!
//! - [`Value::Str`] is the string data of `str` / `String` values and
//!   of `&str` values. The driver materialises `&str` constants to
//!   `ConstValue::Str`, so a string constant *is* this value; taking a
//!   reference to a place holding one yields a [`Value::Ref`] as usual.
//! - [`Value::Bytes`] is raw byte data: byte-array constants such as
//!   the `&[u8; N]` pieces template that `core::fmt` passes to
//!   `Arguments::new`.
//! - [`Value::Ref`] is a reference (or raw pointer) resolved at
//!   creation time to the frame, local and constant projection it
//!   points at. Frames carry unique ids so a reference stays valid
//!   while its frame is on the stack, across calls in both
//!   directions. The stored projection never contains a `Deref` step:
//!   dereferences are followed eagerly when the reference is created.
//! - [`Value::FatRef`] is a fat pointer: a base value plus the slice
//!   length metadata that `PtrMetadata` reads. The base is the
//!   [`Value::Ref`] of an array place (from an unsizing cast) or the
//!   aggregate value itself (from a slice constant).
//! - [`Value::Enum`] is every ADT aggregate: structs use variant 0,
//!   enums carry their variant index. Field projections read
//!   `fields`, `Downcast` checks the variant, `Discriminant` reads it.
//! - [`Value::FmtArg`] is one `core::fmt::rt::Argument`: the reference
//!   handed to a `new_display` / `new_debug` / ... constructor, the
//!   formatting trait it selects, and the pointee type tag. Like the
//!   real `format_args!`, the pointee is read through the reference
//!   when the arguments are formatted, not when the argument is
//!   built. The formatter function pointer of the real type cannot
//!   exist here (its target is std code with no converted body), so
//!   the trait + type tag pair stands in for it; that pair is exactly
//!   what the formatter dispatch in core keys on.
//! - [`Value::FmtCount`] is an `rt::Argument` built by
//!   `Argument::from_usize`: a width/precision count, not a value.
//! - [`Value::FmtArguments`] is a built `std::fmt::Arguments` value:
//!   either a plain string (`from_str`) or a pieces template plus the
//!   snapshot of its argument array (`Arguments::new`).

use std::fmt;

/// One step of a resolved reference target. Indices are constants:
/// an `Index` projection through a local is evaluated against the
/// frame that syntactically contains the place when the reference is
/// created, which is the only frame it can refer to.
#[derive(Clone, Debug, PartialEq)]
pub enum RefProj {
    Field(usize),
    Index(usize),
    Downcast(u32),
}

/// Where a [`Value::Ref`] points.
#[derive(Clone, Debug, PartialEq)]
pub enum RefTarget {
    /// A place in a live frame, identified by the frame's unique id.
    Place {
        frame: u64,
        local: usize,
        projection: Vec<RefProj>,
    },
    /// An immutable constant aggregate (the base of a constant slice
    /// or array). Reads resolve against the stored value; writes
    /// through it are an error.
    Value(Box<Value>),
}

impl RefTarget {
    pub fn place(frame: u64, local: usize, projection: Vec<RefProj>) -> RefTarget {
        RefTarget::Place {
            frame,
            local,
            projection,
        }
    }
}

/// Which formatting trait an `rt::Argument` was built for; mirrors
/// the `new_*` constructor family in `core::fmt::rt`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FmtTrait {
    Display,
    Debug,
    /// `new_debug_noop`: formats as the empty string.
    DebugNoop,
    Octal,
    LowerHex,
    UpperHex,
    Pointer,
    Binary,
    LowerExp,
    UpperExp,
}

/// A built `std::fmt::Arguments` value.
#[derive(Clone, Debug, PartialEq)]
pub enum FmtArguments {
    /// `Arguments::from_str`: printed verbatim.
    Plain(String),
    /// `Arguments::new`: the encoded pieces template (see the parser
    /// in `builtins`) plus the argument values, snapshotted from the
    /// argument array when `new` runs, as the real call snapshots
    /// the array reference's elements.
    Template { template: Vec<u8>, args: Vec<Value> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// The slot was never written (or was moved out of).
    Uninit,
    Unit,
    Bool(bool),
    /// Signed integer, stored sign-extended to 128 bits.
    Int(i128),
    /// Unsigned integer (also `char`, stored as its code point).
    Uint(u128),
    /// A tuple, or an array (arrays ride as tuples of their elements;
    /// the length lives in the declared type).
    Tuple(Vec<Value>),
    /// String data: the value of `str`, `String` and `&str` places.
    Str(String),
    /// Raw bytes (byte-array constants, e.g. fmt pieces templates).
    Bytes(Vec<u8>),
    /// A reference or raw pointer.
    Ref(RefTarget),
    /// A fat pointer (slice reference): base + element count.
    FatRef { base: Box<Value>, len: u128 },
    /// An ADT value: variant index + field values.
    Enum { variant: u32, fields: Vec<Value> },
    /// A `core::fmt::rt::Argument` formatting a pointee.
    FmtArg {
        /// The reference passed to the `new_*` constructor.
        target: Box<Value>,
        trait_: FmtTrait,
        /// Rendered pointee type (the constructor's generic
        /// argument), e.g. `i32`, `&str`, `char`.
        ty: String,
    },
    /// An `rt::Argument` count (`Argument::from_usize`).
    FmtCount(u16),
    /// A built `std::fmt::Arguments`.
    FmtArguments(FmtArguments),
}

impl Value {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Value::Uninit => "uninitialized",
            Value::Unit => "unit",
            Value::Bool(_) => "bool",
            Value::Int(_) => "signed integer",
            Value::Uint(_) => "unsigned integer",
            Value::Tuple(_) => "tuple",
            Value::Str(_) => "string",
            Value::Bytes(_) => "bytes",
            Value::Ref(_) => "reference",
            Value::FatRef { .. } => "slice reference",
            Value::Enum { .. } => "ADT value",
            Value::FmtArg { .. } => "fmt argument",
            Value::FmtCount(_) => "fmt count",
            Value::FmtArguments(_) => "fmt arguments",
        }
    }

    /// View as a boolean, if this value is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The bit pattern used by `SwitchInt`: booleans as 0/1, integers
    /// as their stored pattern. Anything else cannot be switched on.
    pub fn switch_bits(&self) -> Option<u128> {
        match self {
            Value::Bool(b) => Some(*b as u128),
            Value::Int(v) => Some(*v as u128),
            Value::Uint(v) => Some(*v),
            _ => None,
        }
    }

    /// Number of elements of an aggregate/byte/string value, for
    /// `Len` and index resolution. `Str` counts bytes, as MIR `Len`
    /// on a `str` place does.
    pub fn len(&self) -> Option<u128> {
        match self {
            Value::Tuple(items) => Some(items.len() as u128),
            Value::Bytes(b) => Some(b.len() as u128),
            Value::Str(s) => Some(s.len() as u128),
            Value::Enum { fields, .. } => Some(fields.len() as u128),
            Value::FatRef { len, .. } => Some(*len),
            _ => None,
        }
    }
}

impl fmt::Display for RefTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                write!(f, "frame#{frame} local _{local}")?;
                for proj in projection {
                    match proj {
                        RefProj::Field(i) => write!(f, ".{i}")?,
                        RefProj::Index(i) => write!(f, "[{i}]")?,
                        RefProj::Downcast(v) => write!(f, " as variant {v}")?,
                    }
                }
                Ok(())
            }
            RefTarget::Value(_) => write!(f, "constant aggregate"),
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
                if items.len() == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")
            }
            Value::Str(s) => write!(f, "{s:?}"),
            Value::Bytes(b) => write!(f, "b\"{}\"", String::from_utf8_lossy(b)),
            Value::Ref(target) => write!(f, "&({target})"),
            Value::FatRef { len, .. } => write!(f, "&[_; {len}]"),
            Value::Enum { variant, fields } => {
                write!(f, "variant {variant} (")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{field}")?;
                }
                write!(f, ")")
            }
            Value::FmtArg { ty, trait_, .. } => {
                write!(f, "<fmt {trait_:?} argument of {ty}>")
            }
            Value::FmtCount(n) => write!(f, "<fmt count {n}>"),
            Value::FmtArguments(_) => write!(f, "<fmt::Arguments>"),
        }
    }
}
