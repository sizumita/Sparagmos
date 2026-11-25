//! Errors reported by the interpreter.
//!
//! Two families are distinguished on purpose:
//!
//! * [`InterpError::Panic`] models a *guest* panic (arithmetic overflow in a
//!   checked operation, division by zero, a failed `Assert`, reaching
//!   `unreachable`). This mirrors how rustc MIR represents panicking code
//!   and is what the JIT is expected to reproduce. A panic carries its
//!   payload (see [`crate::value::PanicPayload`]) from the panic site to
//!   the `catch_unwind` that observes it; sites that cannot know the
//!   payload's concrete type leave it unset, and the catch site then
//!   presents the message itself as a `&str` payload, which is what
//!   core's literal-message panics carry.
//! * [`InterpError::Abort`] models a panic that may not unwind
//!   (a panic in a cleanup, a `nounwind` crossing): native execution
//!   aborts the process, so no `catch_unwind` observes it.
//! * Every other variant is an interpreter/oracle failure: the input MIR is
//!   malformed, unsupported, or exhausted a resource limit. Unsupported
//!   constructs always fail loudly here — the oracle must never silently
//!   skip a statement, or a JIT/oracle comparison would pass vacuously.

use std::fmt;

use crate::value::PanicPayload;

/// Result alias for interpreter operations.
pub type InterpResult<T> = Result<T, InterpError>;

/// An interpretation failure. See the module docs for the panic split.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpError {
    /// A guest panic (overflow, divide by zero, failed assert, ...).
    /// `payload` is the panic's payload value when the panic site
    /// knows it (a `panic!` message, a `panic_any` argument);
    /// `None` for the interpreter's own modelled panics, whose
    /// payload the catch site synthesizes from the message.
    Panic {
        message: String,
        payload: Option<PanicPayload>,
    },
    /// The run aborted: a guest panic escaped a cleanup block or
    /// crossed a `nounwind` boundary (MIR `UnwindAction::Terminate`
    /// / `Unreachable`, or a `Terminate(Abort)` terminator). Native
    /// execution aborts the process here (a double panic); this
    /// model reports it as its own error so it stays distinct from
    /// a catchable guest panic: `catch_unwind` never observes it,
    /// and the driver maps it to the guest-crash exit (134, the
    /// SIGABRT death code) with the abort reason on stderr.
    Abort { message: String },
    /// The guest called `std::process::exit(code)`: the run ends
    /// successfully with that exit code. This is not a failure —
    /// it is the guest's normal termination path — so it travels
    /// as its own variant the driver maps to the process exit
    /// code, distinct from both a completed `main` (code 0) and
    /// every error outcome.
    ProcessExit { code: i32 },
    /// The running thread segment's scheduler quantum expired
    /// while another thread could run: the cooperative
    /// scheduler's preemption signal. Like
    /// [`InterpError::ProcessExit`] this is control flow, not a
    /// failure — it travels as an error so it can unwind the
    /// interpreter's own call stack out of the drive loop, and
    /// the segment's driver (a spawned thread's
    /// `Interpreter::run_thread`, the entry thread's scheduler
    /// loop in `Interpreter::run_with_report`) catches it, parks
    /// the segment's frames, and reschedules. It is never a run
    /// outcome; any other context observing it has found an
    /// interpreter bug, so its message says so.
    Yield,
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
    /// A guest `fmt` implementation body returned
    /// `Err(fmt::Error)`. The failure is the guest's own
    /// behavior, not an interpreter limitation: the formatting
    /// sinks native composes through panic on it (a `write_fmt`
    /// to a stream whose own writes did not fail), so a context
    /// modelling such a sink turns this into the panic native
    /// raises there. It travels as its own variant, distinct
    /// from the unsupported family, so those contexts can
    /// recognize it; anywhere else it ends the run like any
    /// other error.
    FmtReturnedError {
        /// The formatting trait whose implementation failed
        /// (`Debug`, `Display`, ...).
        trait_: String,
        /// The type whose implementation failed.
        ty: String,
    },
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
            payload: None,
        }
    }

    /// Build a guest-panic error whose payload is the message
    /// itself, as a `&str` — the payload of `panic!("literal")`
    /// and of core's literal-message panics.
    pub fn panic_str_payload(message: impl Into<String>) -> Self {
        let message = message.into();
        InterpError::Panic {
            payload: Some(PanicPayload {
                ty: "&str".to_string(),
                value: Box::new(crate::value::Value::Str(message.clone())),
            }),
            message,
        }
    }

    /// Build a guest-panic error whose payload is the message as a
    /// `String` — the payload of a formatted `panic!` (core
    /// formats the message into a fresh `String` and boxes that).
    pub fn panic_string_payload(message: impl Into<String>) -> Self {
        let message = message.into();
        InterpError::Panic {
            payload: Some(PanicPayload {
                ty: "std::string::String".to_string(),
                value: Box::new(crate::value::Value::Str(message.clone())),
            }),
            message,
        }
    }

    /// Build a guest-panic error with an explicit payload (the
    /// `panic_any` argument, a resumed payload).
    pub fn panic_with_payload(message: impl Into<String>, payload: PanicPayload) -> Self {
        InterpError::Panic {
            message: message.into(),
            payload: Some(payload),
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
            InterpError::Panic { message, .. } => write!(f, "guest panic: {message}"),
            InterpError::ProcessExit { code } => {
                write!(f, "guest exited with code {code}")
            }
            InterpError::Yield => {
                write!(f, "scheduler yield escaped its segment (interpreter bug)")
            }
            InterpError::Abort { message } => write!(f, "guest abort: {message}"),
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
            InterpError::FmtReturnedError { trait_, ty } => {
                write!(f, "the {trait_} implementation for {ty} returned an error")
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
