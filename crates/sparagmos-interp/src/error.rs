//! Errors reported by the interpreter.
//!
//! Two families are distinguished on purpose:
//!
//! * [`InterpError::Panic`] models a *guest* panic (arithmetic overflow in a
//!   checked operation, division by zero, a failed `Assert`, reaching
//!   `unreachable`). This mirrors how rustc MIR represents panicking code
//!   and is what the JIT is expected to reproduce.
//! * Every other variant is an interpreter/oracle failure: the input MIR is
//!   malformed, unsupported, or exhausted a resource limit. Unsupported
//!   constructs always fail loudly here — the oracle must never silently
//!   skip a statement, or a JIT/oracle comparison would pass vacuously.

use std::fmt;

/// Result alias for interpreter operations.
pub type InterpResult<T> = Result<T, InterpError>;

/// An interpretation failure. See the module docs for the panic split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpError {
    /// A guest panic (overflow, divide by zero, failed assert, ...).
    Panic { message: String },
    /// The configured step budget was exhausted (probable infinite loop).
    StepLimitExceeded { limit: usize },
    /// The configured call-depth budget was exhausted.
    CallDepthExceeded { limit: usize },
    /// A call named a function that is not in the program.
    UnknownFunction { name: String },
    /// A call named a function whose body could not be converted;
    /// `reason` is the conversion failure, carried along so it
    /// surfaces with its cause at the call site instead of failing
    /// the whole program before execution starts.
    UnconvertedFunction { name: String, reason: String },
    /// A terminator jumped to a basic block that does not exist.
    UnknownBlock { func: String, block: usize },
    /// A place referred to a local outside the frame.
    UnknownLocal { func: String, local: usize },
    /// A read of a local (or tuple field) that was never written / was moved.
    UninitializedRead { func: String, place: String },
    /// A value of the wrong kind was used where another was required.
    TypeMismatch {
        context: String,
        expected: String,
        found: String,
    },
    /// A statement kind this interpreter does not implement yet.
    UnsupportedStatement { kind: String },
    /// A terminator kind this interpreter does not implement yet.
    UnsupportedTerminator { kind: String },
    /// A method call through a `dyn Trait` value whose concrete
    /// implementation body is not part of the program, so the
    /// vtable entry cannot be resolved to a converted body. The
    /// failure is at the dispatch (the call), never at the value's
    /// declaration.
    UnsupportedDynamicDispatch { method: String, concrete: String },
    /// An rvalue kind this interpreter does not implement yet.
    UnsupportedRvalue { kind: String },
    /// A place projection this interpreter does not implement yet.
    UnsupportedProjection { kind: String },
    /// A call returned into a caller that had no continuation block.
    MissingContinuation { func: String },
    /// Execution fell off the end of a function without a terminator.
    MissingTerminator { func: String, block: usize },
}

impl InterpError {
    /// Build a guest-panic error with the given message.
    pub fn panic(message: impl Into<String>) -> Self {
        InterpError::Panic {
            message: message.into(),
        }
    }

    /// Build a type-mismatch error.
    pub fn type_mismatch(
        context: impl Into<String>,
        expected: impl Into<String>,
        found: impl Into<String>,
    ) -> Self {
        InterpError::TypeMismatch {
            context: context.into(),
            expected: expected.into(),
            found: found.into(),
        }
    }
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpError::Panic { message } => write!(f, "guest panic: {message}"),
            InterpError::StepLimitExceeded { limit } => {
                write!(f, "step limit exceeded ({limit} steps)")
            }
            InterpError::CallDepthExceeded { limit } => {
                write!(f, "call depth limit exceeded ({limit} frames)")
            }
            InterpError::UnknownFunction { name } => write!(f, "unknown function `{name}`"),
            InterpError::UnconvertedFunction { name, reason } => {
                write!(f, "call to `{name}`, whose body could not be converted: {reason}")
            }
            InterpError::UnknownBlock { func, block } => {
                write!(f, "unknown basic block bb{block} in function `{func}`")
            }
            InterpError::UnknownLocal { func, local } => {
                write!(f, "unknown local _{local} in function `{func}`")
            }
            InterpError::UninitializedRead { func, place } => {
                write!(
                    f,
                    "read of uninitialized place {place} in function `{func}`"
                )
            }
            InterpError::TypeMismatch {
                context,
                expected,
                found,
            } => write!(
                f,
                "type mismatch in {context}: expected {expected}, found {found}"
            ),
            InterpError::UnsupportedStatement { kind } => {
                write!(f, "unsupported statement: {kind}")
            }
            InterpError::UnsupportedTerminator { kind } => {
                write!(f, "unsupported terminator: {kind}")
            }
            InterpError::UnsupportedDynamicDispatch { method, concrete } => {
                write!(
                    f,
                    "dynamic dispatch of `{method}` on `{concrete}` is unsupported: \
                     the concrete implementation body is not in the program"
                )
            }
            InterpError::UnsupportedRvalue { kind } => write!(f, "unsupported rvalue: {kind}"),
            InterpError::UnsupportedProjection { kind } => {
                write!(f, "unsupported projection: {kind}")
            }
            InterpError::MissingContinuation { func } => {
                write!(
                    f,
                    "call in `{func}` has no continuation block but the callee returned"
                )
            }
            InterpError::MissingTerminator { func, block } => {
                write!(f, "basic block bb{block} in `{func}` has no terminator")
            }
        }
    }
}

impl std::error::Error for InterpError {}
