//! Backtrace capture values for the backtrace builtins.



use super::*;
use crate::interp::Frame;
use crate::value::Value;


/// A `std::backtrace::Backtrace` value: the capture status
/// (`std::backtrace::BacktraceStatus`'s discriminant:
/// unsupported 0, disabled 1, captured 2) and, for a captured
/// backtrace, its rendered trace -- the interpreter's own
/// call stack, innermost frame first, in std's short format.
/// Whether capture happens follows the environment std
/// itself consults: `RUST_LIB_BACKTRACE` when set (`0`
/// disables, anything else enables), else `RUST_BACKTRACE`
/// (unset or `0` disables); `full` in either selects the
/// full style. A forced capture always captures.
pub(super) fn backtrace_value(frames: &[Frame], force: bool) -> Value {
    let (status, full) = backtrace_capture_decision(force);
    let text = if status == 2 {
        let mut out = String::from("stack backtrace:\n");
        for (i, frame) in frames.iter().rev().enumerate() {
            out.push_str(&format!("   {i}: {}\n", frame.func));
        }
        if !full {
            out.push_str(
                "note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.",
            );
        }
        out
    } else {
        String::new()
    };
    Value::Enum {
        variant: 0,
        fields: vec![Value::Uint(u128::from(status)), Value::Str(text)],
    }
}

/// The `(status, full)` a capture with the given force flag
/// settles on (see `backtrace_value`).
fn backtrace_capture_decision(force: bool) -> (u32, bool) {
    if force {
        let full = std::env::var("RUST_BACKTRACE")
            .map(|v| v == "full")
            .unwrap_or(false);
        return (2, full);
    }
    if let Ok(lib) = std::env::var("RUST_LIB_BACKTRACE") {
        if lib == "0" {
            return (1, false);
        }
        let full = lib == "full"
            || std::env::var("RUST_BACKTRACE")
                .map(|v| v == "full")
                .unwrap_or(false);
        return (2, full);
    }
    match std::env::var("RUST_BACKTRACE") {
        Ok(v) if v == "0" => (1, false),
        Ok(v) => (2, v == "full"),
        Err(_) => (1, false),
    }
}

/// The capture status recorded in a `Backtrace` value (see
/// `backtrace_value`); `None` for any other shape.
pub(super) fn backtrace_status_of(value: &Value) -> Option<u32> {
    let fields = match value {
        Value::Enum { fields, .. } | Value::Tuple(fields) => fields,
        _ => return None,
    };
    match fields.first() {
        Some(Value::Uint(status)) => Some(*status as u32),
        Some(Value::Int(status)) => Some(*status as u32),
        _ => None,
    }
}

/// The text a `Backtrace` value's `Display` / `Debug`
/// produce: the disabled and unsupported markers, or the
/// rendered trace recorded at capture.
pub(super) fn backtrace_display(value: &Value) -> String {
    match backtrace_status_of(value) {
        Some(2) => match value {
            Value::Enum { fields, .. } | Value::Tuple(fields) => {
                match fields.get(1) {
                    Some(Value::Str(text)) => text.clone(),
                    _ => String::new(),
                }
            }
            _ => String::new(),
        },
        Some(1) => "disabled backtrace".to_string(),
        _ => "unsupported backtrace".to_string(),
    }
}

/// Registration fragment of the builtin name registry: the backtrace surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn backtrace_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::backtrace::Backtrace::capture" => Builtin::BacktraceCapture,
        "std::backtrace::Backtrace::force_capture" => {
            Builtin::BacktraceForceCapture
        }
        "std::backtrace::Backtrace::status" => Builtin::BacktraceStatus,
        _ => return None,
    })
}
