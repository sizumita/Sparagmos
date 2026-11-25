//! Guest panic report rendering.




use super::*;

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Escape a string for the harness's JSON output formats:
/// the quote, the backslash, and the control characters (the
/// short escapes for newline, carriage return, and tab, the
/// `\uXXXX` form for the rest), as libtest's own JSON writer
/// emits them.
pub(super) fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// A panic report as it appears in a passing should-panic
/// test's captured output: the `thread '...' panicked at ...`
/// head and the message, without the backtrace note the
/// failure account adds.
/// The full panic report of a failing test's captured output:
/// the report head plus the backtrace note line, as libtest
/// prints it ahead of a failure's own account (for example the
/// `panic did not contain expected string` note).
pub(super) fn panic_report_full(
    test: &TestDesc,
    thread_id: u64,
    location: Option<(String, u64, u64)>,
    message: &str,
) -> String {
    format!(
        "{}note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
        panic_report_head(test, thread_id, location, message)
    )
}

pub(super) fn panic_report_head(
    test: &TestDesc,
    thread_id: u64,
    location: Option<(String, u64, u64)>,
    message: &str,
) -> String {
    match location {
        Some((file, line, col)) => format!(
            "\nthread '{}' ({}) panicked at {}:{}:{}:\n{}\n",
            test.name, thread_id, file, line, col, message
        ),
        None => format!(
            "\nthread '{}' ({}) panicked:\n{}\n",
            test.name, thread_id, message
        ),
    }
}
