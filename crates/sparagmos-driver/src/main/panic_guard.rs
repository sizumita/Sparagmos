//! Containment of compiler panics and Stable MIR query failures, plus the compiler-stdout redirect.

use super::*;

/// The process panic hook, as returned by `std::panic::take_hook`.
type PanicHook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync + 'static>;

/// Format a panic as a concrete reason: the payload text when the
/// payload is a string, plus the panic location when known. The
/// compiler also panics with non-string marker payloads; those are
/// reported as such, with the location still identifying where the
/// conversion gave up.
fn panic_reason(info: &std::panic::PanicHookInfo<'_>) -> String {
    let message = if let Some(s) = info.payload().downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_string()
    };
    match info.location() {
        Some(loc) => format!(
            "{message} (panicked at {}:{}:{})",
            loc.file(),
            loc.line(),
            loc.column()
        ),
        None => message,
    }
}

/// The report text for a delayed bug the compiler created during
/// compilation and flushed at the end of the run.
pub(crate) const DELAYED_BUG_ABORT: &str = "the compiler aborted on a delayed bug it created during \
    compilation (see the internal compiler error above)";

/// Whether a panic that escaped the compiler run is the flush of
/// a delayed bug the compiler created during compilation: such a
/// bug is flushed at the end of the run by panicking with the
/// non-string `DelayedBugPanic` marker (rustc_errors/src/lib.rs:
/// 1499), and its own text is an internal compiler error the
/// compiler has already printed. The flush only panics when the
/// compilation would otherwise have finished, so the compilation
/// died on the compiler's own internal error: a compiler crash,
/// which the caller reports as one (Issue #66).
pub(crate) fn is_delayed_bug_flush(reason: &str) -> bool {
    reason.contains("rustc_errors/src/lib.rs:1499:")
}

/// When a panic that escaped the compiler run is the compiler
/// aborting the compilation on a diagnostic it had already
/// reported, the compilation-failure reason to report for it;
/// `None` for anything else. One shape: an error emitted under
/// `-Z treat-err-as-bug` panics with "aborting due to
/// `-Z treat-err-as-bug=…`" (rustc_errors'
/// `panic_if_treat_err_as_bug`). The flag orders exactly this
/// abort, so the outcome stays a compilation failure carrying the
/// reported diagnostic, not a crash.
pub(crate) fn treat_err_abort_reason(reason: &str) -> Option<String> {
    if reason.contains("treat-err-as-bug") {
        Some(reason.to_string())
    } else {
        None
    }
}

/// Reason text from a caught panic payload, when it is a string.
pub(crate) fn payload_reason(payload: &(dyn std::any::Any + Send)) -> Option<String> {
    if let Some(s) = payload.downcast_ref::<&str>() {
        Some((*s).to_string())
    } else if let Some(s) = payload.downcast_ref::<String>() {
        Some(s.clone())
    } else {
        None
    }
}

/// The skip reason for a caught panic whose payload is the
/// bare text "not implemented": several of rustc_public
/// 1.100.0's unimplemented conversions panic with exactly that
/// payload (`unimplemented!()` with no message), so the
/// payload alone cannot tell them apart — but the panic
/// hook's record of the same panic carries its location,
/// which names the site. The record must be this very panic:
/// its message part is the same bare text. (The toolchain is
/// pinned, so the locations are stable.)
fn unimplemented_site_reason(
    payload: &(dyn std::any::Any + Send),
) -> Option<&'static str> {
    if payload_reason(payload).as_deref() != Some("not implemented") {
        return None;
    }
    let captured = peek_captured_panic()?;
    let (message, location) = captured.split_once(" (panicked at ")?;
    if message != "not implemented" {
        return None;
    }
    if location.contains("rustc_public/src/unstable/convert/stable/mir.rs:775:") {
        Some(TAIL_CALL_REASON)
    } else if location
        .contains("rustc_public/src/unstable/convert/stable/ty.rs:471:")
    {
        Some(UNSAFE_BINDER_REASON)
    } else if location
        .contains("rustc_public/src/unstable/convert/stable/ty.rs:577:")
    {
        Some(CONST_EXPR_REASON)
    } else {
        None
    }
}

/// Run a Stable MIR query that may hit one of rustc_public
/// 1.100.0's known unimplemented conversions (inspecting the
/// type kind of an async closure item materializes it, unlike
/// a body fetch, whose types stay opaque until conversion —
/// but a body fetch materializes the whole body, terminators
/// and projections included). A panic from one of those
/// specific unimplemented conversions is contained and
/// reported as its named reason (`stable_mir_panic_reason`
/// for the payloads that identify themselves,
/// `unimplemented_site_reason` for the bare "not implemented"
/// ones); any other panic is resumed — this guard must never
/// paper over an unrelated compiler failure. The query also
/// runs as a body materialization where it fetches one, so the
/// diagnostics its eager constant evaluation emits are
/// discounted the same way (see
/// `sparagmos_driver::convert::with_body_materialization`).
pub(crate) fn guard_stable_mir_query<T>(
    key: &str,
    query: impl FnOnce() -> T,
) -> Result<T, String> {
    sparagmos_driver::convert::with_body_materialization(key, || {
        guard_stable_mir_query_contained(query)
    })
}

fn guard_stable_mir_query_contained<T>(
    query: impl FnOnce() -> T,
) -> Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(query)) {
        Ok(value) => Ok(value),
        Err(payload) => {
            let reason = stable_mir_panic_reason(&*payload)
                .map(str::to_string)
                .or_else(|| unimplemented_site_reason(&*payload).map(str::to_string));
            if let Some(reason) = reason {
                // Contained panic: clear the capture slot, as the
                // instance collection does — a later, genuinely
                // fatal panic must report its own message.
                let _ = take_captured_panic();
                Err(reason)
            } else {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

/// Run one body fetch through the Stable MIR guard,
/// additionally containing the trait solver's whole-body
/// normalization overflow (see
/// `convert::BODY_FETCH_NORMALIZE_OVERFLOW`): the guard
/// resumes every panic it does not know, and this class is
/// known only by its recorded location, so it is caught once
/// more here and classified from the panic-capture slot the
/// hook filled (cleared before the fetch, so the record read
/// back is this fetch's own). The fetch's named reasons pass
/// through unchanged, and any other resumed panic keeps
/// unwinding.
pub(crate) fn fetch_query_normalize_aware<T>(
    key: &str,
    simd_generated: bool,
    query: impl FnOnce() -> T,
) -> Result<T, String> {
    sparagmos_driver::convert::clear_captured_panic();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        guard_stable_mir_query(key, query)
    })) {
        Ok(result) => result,
        Err(payload) => {
            let captured = sparagmos_driver::convert::peek_captured_panic();
            let overflow = captured.as_ref().is_some_and(|text| {
                sparagmos_driver::convert::captured_panic_is_normalize_overflow(
                    text,
                )
            });
            // A MIR validation failure is contained only when
            // the fetch is a compiler-generated impl body over
            // a SIMD type (`simd_generated`): that is the one
            // class whose materialization trips the MCP#838
            // projection ban by construction. Any other
            // validation failure keeps unwinding.
            let simd_validator = simd_generated
                && captured.as_ref().is_some_and(|text| {
                    sparagmos_driver::convert::captured_panic_is_mir_validator_fail(
                        text,
                    )
                });
            if overflow {
                let _ = sparagmos_driver::convert::take_captured_panic();
                Err(sparagmos_driver::convert::BODY_FETCH_NORMALIZE_OVERFLOW
                    .to_string())
            } else if simd_validator {
                let _ = sparagmos_driver::convert::take_captured_panic();
                Err(sparagmos_driver::convert::BODY_FETCH_SIMD_VALIDATOR
                    .to_string())
            } else {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

/// Convert one body through the Stable MIR containment (see
/// `Converter::convert_body_contained`), clearing the panic
/// capture slot when a containment fired: the hook recorded
/// the contained panic's message, and a later, genuinely fatal
/// panic must report its own. A panic the inner containment
/// resumes because its payload is the bare "not implemented"
/// is caught once more here and identified by its recorded
/// location (`unimplemented_site_reason`).
pub(crate) fn convert_body_isolated(
    conv: &mut Converter,
    body: &rustc_public::mir::Body,
) -> Result<Body, Vec<String>> {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        conv.convert_body_contained(body)
    }));
    let result = match outcome {
        Ok(result) => result,
        Err(payload) => {
            if let Some(reason) = unimplemented_site_reason(&*payload) {
                let _ = take_captured_panic();
                return Err(vec![reason.to_string()]);
            }
            std::panic::resume_unwind(payload);
        }
    };
    if matches!(&result, Err(errs) if errs.iter().any(|e| is_contained_reason(e))) {
        let _ = take_captured_panic();
    }
    result
}

/// Fetch one demanded instance's body through the Stable MIR
/// containment (see [`guard_stable_mir_query`]): materializing a
/// body can hit one of rustc_public's known unimplemented
/// conversions (a tail-call terminator is the known case), and in
/// the demand passes that panic must become the definition's
/// named demand failure. An uncontained fetch would unwind to
/// the per-body scan guard, which abandons the scan silently and
/// leaves a demanded definition looking undemanded. `Ok(None)`
/// is an instance with no body available; `Err` is the named
/// reason.
pub(crate) fn fetch_instance_body(
    inst: &Instance,
) -> Result<Option<rustc_public::mir::Body>, String> {
    // An instance whose types mention a projection that
    // does not normalize: fetching its body would pose that
    // normalization and poison the compilation (see
    // `unnormalizable_projection_in_instance`). Refuse the
    // fetch by name instead; the caller records the named
    // demand failure.
    if let Some(reason) =
        sparagmos_driver::convert::unnormalizable_projection_in_instance(inst)
    {
        return Err(reason);
    }
    // A fetch whose whole-body normalization overflowed the
    // trait solver is reported under its own name (see
    // `convert::BODY_FETCH_NORMALIZE_OVERFLOW`): the callers
    // route it to the recursion-limit compilation failure,
    // the verdict rustc's collector reports for the same
    // overflow, instead of recording it as this definition's
    // demand failure.
    fetch_query_normalize_aware(&inst.name(), false, || inst.body())
}

/// Whether a conversion error is one of the contained Stable
/// MIR panics' named reasons (see `stable_mir_panic_reason`).
fn is_contained_reason(reason: &str) -> bool {
    reason == COROUTINE_CLOSURE_TYPE_REASON
        || reason == UNSAFE_BINDER_REASON
        || reason == INSTANCE_CONST_REASON
}

/// Install a panic hook that records the first panic's message into
/// the library's panic-capture slot (see
/// `sparagmos_driver::convert::record_captured_panic`) and prints
/// nothing. While the compiler runs, the
/// default panic dump must not leak into this process's stderr: run
/// mode reserves stderr for the contract messages, and the captured
/// message is reported as the conversion-failure reason instead.
/// Returns the previous hook, to be restored by `end_panic_capture`
/// as soon as the compiler run is over — panics outside the run
/// (e.g. in the interpreter) keep the normal reporting.
pub(crate) fn begin_panic_capture() -> PanicHook {
    clear_captured_panic();
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        record_captured_panic(panic_reason(info));
    }));
    previous
}

pub(crate) fn end_panic_capture(previous: PanicHook) {
    std::panic::set_hook(previous);
}

/// Guard that points the process's stdout file descriptor at stderr
/// for the duration of the compiler run. The compiler runs
/// in-process, so anything it prints to *its* stdout at compile
/// time (the `log_syntax!` macro is the known case) would otherwise
/// land in this process's stdout — which run mode reserves for the
/// guest program's output, and summary mode for the driver's own
/// report. Compiler-side output is diagnostic in nature, so it
/// belongs on stderr. Restore explicitly via `restore` once the run
/// is over; dropping the guard restores as a safety net.
#[cfg(unix)]
pub(crate) struct StdoutToStderr {
    saved: Option<std::os::unix::io::RawFd>,
}

#[cfg(unix)]
impl StdoutToStderr {
    pub(crate) fn new() -> Self {
        use std::io::Write;
        // Flush anything buffered before swapping the descriptor,
        // so it cannot surface on the wrong stream afterwards.
        let _ = std::io::stdout().flush();
        let saved = unsafe { libc::dup(libc::STDOUT_FILENO) };
        if saved < 0 {
            return Self { saved: None };
        }
        if unsafe { libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) } < 0 {
            unsafe { libc::close(saved) };
            return Self { saved: None };
        }
        Self { saved: Some(saved) }
    }

    /// Point stdout at the file at `path` for the duration of the
    /// run instead of at stderr: a caller that judges the
    /// compiler's own printed output (a `-Zunpretty` dump or a
    /// `--print` answer, both written to the compiler's stdout)
    /// reads the file back once the run is over. If the file
    /// cannot be created, fall back to the stderr redirect: the
    /// capture is then absent, which the caller sees as a missing
    /// file rather than as output the compiler never produced.
    pub(crate) fn to_file(path: &std::path::Path) -> Self {
        use std::io::Write;
        use std::os::unix::io::AsRawFd;
        let _ = std::io::stdout().flush();
        let saved = unsafe { libc::dup(libc::STDOUT_FILENO) };
        if saved < 0 {
            return Self { saved: None };
        }
        let file = std::fs::File::create(path);
        let target = match &file {
            Ok(f) => f.as_raw_fd(),
            Err(_) => libc::STDERR_FILENO,
        };
        if unsafe { libc::dup2(target, libc::STDOUT_FILENO) } < 0 {
            unsafe { libc::close(saved) };
            return Self { saved: None };
        }
        Self { saved: Some(saved) }
    }

    pub(crate) fn restore(&mut self) {
        use std::io::Write;
        if let Some(saved) = self.saved.take() {
            // Flush compile-time output still buffered in Rust's
            // stdout handle so it goes to stderr, as intended,
            // instead of leaking into the restored stdout.
            let _ = std::io::stdout().flush();
            unsafe {
                libc::dup2(saved, libc::STDOUT_FILENO);
                libc::close(saved);
            }
        }
    }
}

#[cfg(unix)]
impl Drop for StdoutToStderr {
    fn drop(&mut self) {
        self.restore();
    }
}

/// No-op counterpart on non-Unix platforms (the driver is developed
/// and measured on Linux; the descriptor swap has no portable
/// equivalent there).
#[cfg(not(unix))]
struct StdoutToStderr;

#[cfg(not(unix))]
impl StdoutToStderr {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn to_file(_path: &std::path::Path) -> Self {
        Self
    }

    pub(crate) fn restore(&mut self) {}
}
