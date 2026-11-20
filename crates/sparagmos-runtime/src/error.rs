//! Panic / runtime-error representation shared by JIT and interpreter.
use std::fmt;
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    Panic {
        message: String,
        location: Option<String>,
    },
    Overflow {
        op: &'static str,
    },
    BoundsCheck {
        index: usize,
        len: usize,
    },
    DivideByZero,
    Unreachable,
    NullPointer,
    Unimplemented(String),
    Layout(String),
    AllocFailed {
        size: usize,
        align: usize,
    },
    UnknownIntrinsic(String),
    InvalidIntrinsicArgs {
        name: String,
        expected: String,
    },
}
impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Panic { message, location } => write!(
                f,
                "panic: {message}{}",
                location
                    .as_ref()
                    .map(|l| format!(" at {l}"))
                    .unwrap_or_default()
            ),
            Self::Overflow { op } => write!(f, "arithmetic overflow in {op}"),
            Self::BoundsCheck { index, len } => {
                write!(f, "index out of bounds: index {index} len {len}")
            }
            Self::DivideByZero => write!(f, "attempt to divide by zero"),
            Self::Unreachable => write!(f, "entered unreachable code"),
            Self::NullPointer => write!(f, "null pointer dereference"),
            Self::Unimplemented(s) => write!(f, "unimplemented: {s}"),
            Self::Layout(s) => write!(f, "layout error: {s}"),
            Self::AllocFailed { size, align } => {
                write!(f, "allocation failed: size={size} align={align}")
            }
            Self::UnknownIntrinsic(n) => write!(f, "unknown intrinsic: {n}"),
            Self::InvalidIntrinsicArgs { name, expected } => {
                write!(f, "invalid args for intrinsic {name}: expected {expected}")
            }
        }
    }
}
impl std::error::Error for RuntimeError {}
pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanicRecord {
    pub message: String,
    pub location: Option<String>,
}

static LAST_PANIC: Mutex<Option<PanicRecord>> = Mutex::new(None);
pub fn record_panic(message: impl Into<String>, location: Option<String>) {
    if let Ok(mut g) = LAST_PANIC.lock() {
        *g = Some(PanicRecord {
            message: message.into(),
            location,
        });
    }
}
pub fn last_panic() -> Option<PanicRecord> {
    LAST_PANIC.lock().ok().and_then(|g| g.clone())
}
pub fn take_last_panic() -> Option<PanicRecord> {
    LAST_PANIC.lock().ok().and_then(|mut g| g.take())
}
pub fn clear_last_panic() {
    if let Ok(mut g) = LAST_PANIC.lock() {
        *g = None;
    }
}
/// Convert a RuntimeError into a recorded panic and return it (propagation helper).
pub fn propagate<T>(r: RuntimeResult<T>) -> RuntimeResult<T> {
    if let Err(e) = &r {
        if let RuntimeError::Panic { message, location } = e {
            record_panic(message.clone(), location.clone());
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn panic_record_roundtrip() {
        clear_last_panic();
        record_panic("boom", Some("a.rs:1".into()));
        assert_eq!(last_panic().unwrap().message, "boom");
        assert_eq!(
            take_last_panic().unwrap().location.as_deref(),
            Some("a.rs:1")
        );
        assert!(last_panic().is_none());
    }
    #[test]
    fn display_variants() {
        assert!(RuntimeError::BoundsCheck { index: 5, len: 3 }
            .to_string()
            .contains('5'));
        assert!(RuntimeError::Overflow { op: "add" }
            .to_string()
            .contains("overflow"));
    }
    #[test]
    fn propagate_records() {
        clear_last_panic();
        let r: RuntimeResult<()> = Err(RuntimeError::Panic {
            message: "x".into(),
            location: None,
        });
        assert!(propagate(r).is_err());
        assert_eq!(last_panic().unwrap().message, "x");
        clear_last_panic();
    }
}
