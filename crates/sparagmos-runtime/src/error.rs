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

/// Render std's default panic report for a guest panic, in the
/// shape the interpreter's default-hook path prints (see
/// `sparagmos-interp`'s `print_default_panic_report`): a leading
/// newline, the head naming the thread (`main`, guest thread id
/// 1 — the JIT path runs only the entry thread), the location
/// when the record carries one, and the message, followed by
/// the backtrace part the environment prescribes. The JIT
/// records no guest frames, so a backtrace style (Short or
/// Full) renders no trace here and prints no note; the Off
/// style prints the `RUST_BACKTRACE=1` note, as std does.
/// The style resolution mirrors std's `default_hook`:
/// `RUST_LIB_BACKTRACE` first (only `0` / `1` / `full` count),
/// then `RUST_BACKTRACE` (unset or `0` is Off, `full` is Full,
/// anything else is Short), read from the process environment
/// the guest itself observes.
pub fn std_default_panic_report(message: &str, location: Option<&str>) -> String {
    let mut out = String::new();
    out.push('\n');
    match location {
        Some(loc) => out.push_str(&format!(
            "thread 'main' (1) panicked at {loc}:\n{message}\n"
        )),
        None => out.push_str(&format!("thread 'main' (1) panicked:\n{message}\n")),
    }
    let off = match std::env::var("RUST_LIB_BACKTRACE").ok().as_deref() {
        Some("0") => true,
        Some("1") | Some("full") => false,
        _ => matches!(
            std::env::var("RUST_BACKTRACE").ok().as_deref(),
            None | Some("0")
        ),
    };
    if off {
        out.push_str(
            "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
        );
    }
    out
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
pub(crate) static PANIC_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_panic_report_shape() {
        // The head is std's: leading newline, thread name and
        // guest id, the location when one is recorded, then the
        // message. (The note line depends on the process
        // backtrace environment, so it is not asserted here.)
        let with_loc = std_default_panic_report("boom", Some("guest.rs:3:7"));
        assert!(
            with_loc.starts_with("\nthread 'main' (1) panicked at guest.rs:3:7:\nboom\n"),
            "report: {with_loc:?}"
        );
        let without_loc = std_default_panic_report("boom", None);
        assert!(
            without_loc.starts_with("\nthread 'main' (1) panicked:\nboom\n"),
            "report: {without_loc:?}"
        );
        // The invented single-line form is not produced.
        assert!(!with_loc.contains("panicked: boom"));
        assert!(!without_loc.contains("panicked: boom"));
    }

    #[test]
    fn panic_record_roundtrip() {
        let _g = PANIC_TEST_LOCK.lock().unwrap();
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
        let _g = PANIC_TEST_LOCK.lock().unwrap();
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
