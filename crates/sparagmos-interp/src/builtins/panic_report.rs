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

/// Registration fragment of the builtin name registry: the panic hook surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn panic_hook_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::panic::set_hook" => Builtin::PanicSetHook,
        "std::panic::set_backtrace_style" => Builtin::PanicSetBacktraceStyle,
        "std::panic::always_abort" => Builtin::PanicAlwaysAbort,
        "std::panic::take_hook" => Builtin::PanicTakeHook,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the panic entry points.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn panic_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        // The panic path: these callees never return; the builtin
        // raises the guest panic the call stands for.
        "core::panicking::panic" => Builtin::PanicStr,
        "std::rt::panic_fmt" | "core::panicking::panic_fmt" => Builtin::PanicFmt,
        "core::panicking::assert_failed" => Builtin::AssertFailed,
        "core::panicking::panic_nounwind_fmt" => Builtin::PanicNounwindFmt,
        // `panic_nounwind(&str)` and `Option::expect_failed(&str)`
        // panic with exactly the given message, as `panic` does.
        "core::panicking::panic_nounwind" => Builtin::PanicStr,
        "std::option::expect_failed" | "core::option::expect_failed" => Builtin::PanicStr,
        "std::option::unwrap_failed" | "core::option::unwrap_failed" => Builtin::OptionUnwrapFailed,
        "std::result::unwrap_failed" | "core::result::unwrap_failed" => Builtin::ResultUnwrapFailed,
        "core::num::from_ascii_bytes_radix_panic" => Builtin::FromAsciiRadixPanic,
        "core::slice::index::slice_index_fail" => Builtin::SliceIndexFail,
        "core::slice::copy_from_slice_impl::len_mismatch_fail" => Builtin::LenMismatchFail,
        "core::num::imp::int_log10::panic_for_nonpositive_argument" => {
            Builtin::PanicForNonpositiveArgument
        }
        "core::slice::sort::shared::smallsort::panic_on_ord_violation" => {
            Builtin::PanicOnOrdViolation
        }
        "alloc::sync::panic_arc_overflow" => Builtin::PanicArcOverflow,
        "core::str::slice_error_fail" => Builtin::SliceErrorFail,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the panic-catching surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn catch_unwind_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        // The panic-catching path: the `catch_unwind` intrinsic
        // itself (its std wrapper's body is collected and runs;
        // only the intrinsic is modelled here), the payload
        // plumbing around it, and the `dyn Any` downcast methods,
        // whose collected bodies dispatch through `Any::type_id`,
        // a virtual call this model cannot resolve.
        "std::intrinsics::catch_unwind" | "core::intrinsics::catch_unwind" => {
            Builtin::CatchUnwind
        }
        "std::panicking::catch_unwind::cleanup" => Builtin::PanicCleanup,
        "std::panicking::panicking" | "std::thread::panicking" => Builtin::Panicking,
        "std::rt::panic_count::count_is_zero" => Builtin::PanicCountIsZero,
        "std::rt::panic_count::is_zero_slow_path" => Builtin::PanicCountIsZero,
        "std::panicking::begin_panic" | "std::rt::begin_panic" => Builtin::PanicAny,
        "std::panic::resume_unwind" | "std::panicking::resume_unwind" => {
            Builtin::ResumeUnwind
        }
        "<(dyn std::any::Any + 'static)>::downcast_ref"
        | "<(dyn std::any::Any + std::marker::Send + 'static)>::downcast_ref" => {
            Builtin::DynAnyDowncastRef
        }
        "<(dyn std::any::Any + 'static)>::downcast"
        | "<(dyn std::any::Any + std::marker::Send + 'static)>::downcast" => {
            Builtin::DynAnyDowncast(0)
        }
        "<(dyn std::any::Any + 'static)>::is"
        | "<(dyn std::any::Any + std::marker::Send + 'static)>::is" => Builtin::DynAnyIs,
        // By-value downcast is an inherent method on the `Box`
        // itself (in alloc's `convert` module), not on the dyn
        // type: `Box<dyn Any>::downcast` is the call guest code
        // actually makes.
        "std::boxed::convert::<impl std::boxed::Box<(dyn std::any::Any + 'static), A>>::downcast"
        | "std::boxed::convert::<impl std::boxed::Box<(dyn std::any::Any + std::marker::Send + 'static), A>>::downcast" => {
            Builtin::DynAnyDowncast(1)
        }
        _ => return None,
    })
}
