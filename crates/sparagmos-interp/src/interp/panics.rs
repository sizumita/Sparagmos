//! Guest panic handling.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{PanicPayload, RefTarget, Value};

impl<'a> Interpreter<'a> {
    /// Run the panic hook for a freshly raised panic, before
    /// unwinding begins: with no custom hook installed, print
    /// std's default report (see
    /// [`Interpreter::print_default_panic_report`]); with one,
    /// call it with a modelled `PanicHookInfo` (field 0 is the
    /// payload as a `dyn Any` reference, field 1 the panic's
    /// location — the stamp the raising builtin received, or a
    /// zero location for a panic raised without one — field 2
    /// `can_unwind`). A panic raised by the hook itself does
    /// not re-enter the hook: std prints the processing-panic
    /// abort lines for it and aborts, which the `hook_abort`
    /// mark hands to the unwinder.
    pub(crate) fn fire_panic_hook(
        &mut self,
        frames: &mut Vec<Frame>,
        stop_depth: usize,
        message: &str,
        payload: &Option<PanicPayload>,
    ) {
        if self.in_panic_hook {
            // A panic raised while the hook runs: std prints
            // `panicked at {location}:` and the second panic's
            // message — the message only when the payload is a
            // static `&str` (a payload whose `fmt::Arguments`
            // came from `Arguments::from_str`: the direct
            // `core::panicking::panic` calls, as in `todo!()`);
            // a formatted or `panic_any` payload prints an
            // empty message — then the abort line.
            if !self.hook_abort.get() {
                let msg = match payload {
                    Some(p) if p.ty == "&str" => message,
                    _ => "",
                };
                let mut out = String::new();
                if let Some((file, line, col)) = self.panic_location_in(frames) {
                    out.push_str(&format!("panicked at {file}:{line}:{col}:\n{msg}\n"));
                } else {
                    out.push_str(&format!("panicked:\n{msg}\n"));
                }
                out.push_str("thread panicked while processing panic. aborting.\n");
                self.push_error_output(&out);
                self.hook_abort.set(true);
            }
            return;
        }
        if matches!(self.hook_fired_depth, Some(d) if d > stop_depth) {
            return;
        }
        self.hook_fired_depth = Some(stop_depth);
        // The panic is now in flight on this thread (std's
        // panic count): it stays counted until it is consumed
        // (caught, turned into a thread outcome, or taken by
        // the libtest harness model).
        let tid = self.current_thread_id();
        let in_flight = {
            let mut counts = self.panics_in_flight.borrow_mut();
            let n = counts.entry(tid).or_insert(0);
            *n += 1;
            *n
        };
        // The panic's report state (its entry path and its
        // force-no-backtrace flag) is consumed by this fire: a
        // later panic that records neither starts clean.
        let origin = self.last_panic_origin.borrow_mut().take();
        let no_backtrace = self.last_panic_no_backtrace.get();
        self.last_panic_no_backtrace.set(false);
        let hook = self.panic_hook.borrow().clone();
        let is_custom = hook.as_ref().is_some_and(|h| !is_default_hook(h));
        if !is_custom {
            if !self.panic_report_suppressed.get() {
                self.print_default_panic_report(
                    frames,
                    message,
                    in_flight,
                    &origin,
                    no_backtrace,
                );
            }
            return;
        }
        let location = self
            .last_panic_stamp
            .borrow_mut()
            .take()
            .unwrap_or_else(|| {
                Value::Ref(RefTarget::value(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Str(String::new()), Value::Uint(0), Value::Uint(0)],
                }))
            });
        self.call_panic_hook(frames, message, payload.clone(), location, true);
    }

    /// Install the guest's panic hook (see
    /// [`Interpreter::panic_hook`]); the default-hook marker
    /// restores the default (no hook called).
    pub(crate) fn set_backtrace_style(&self, style: Value) {
        *self.backtrace_style.borrow_mut() = Some(style);
    }

    pub(crate) fn set_panic_hook(&self, hook: Value) {
        if is_default_hook(&hook) {
            *self.panic_hook.borrow_mut() = None;
        } else {
            *self.panic_hook.borrow_mut() = Some(hook);
        }
    }

    /// Take the guest's panic hook, leaving the default in its
    /// place; taking the default yields the default-hook marker
    /// value (see [`default_hook_value`]).
    pub(crate) fn take_panic_hook(&self) -> Value {
        self.panic_hook
            .borrow_mut()
            .take()
            .unwrap_or_else(default_hook_value)
    }

    /// Record the caller-location stamp of a panic being raised
    /// (see [`Interpreter::last_panic_stamp`]).
    pub(crate) fn note_panic_stamp(&self, stamp: Value) {
        *self.last_panic_stamp.borrow_mut() = Some(stamp);
    }

    /// Record the caller-location stamp of a panic entry call,
    /// applying `#[track_caller]` forwarding: when the calling
    /// frame is itself a track-caller frame, the location the
    /// entry reports is the location that frame received from
    /// its own caller, not the baked call-site stamp (which
    /// names the call site inside the track-caller body). This
    /// is the same replacement the call path performs when it
    /// binds a track-caller callee's location local.
    pub(crate) fn note_forwarded_panic_stamp(
        &self,
        frames: &[Frame],
        stamp: Value,
    ) {
        if let Some(frame) = frames.last() {
            if let Some(local) = self
                .program
                .get(&frame.func)
                .and_then(|b| b.caller_location_local)
            {
                if let Some(received) = frame.locals.get(local) {
                    if Self::location_of_stamp(received).is_some() {
                        self.note_panic_stamp(received.clone());
                        return;
                    }
                }
            }
        }
        self.note_panic_stamp(stamp);
    }

    /// Forget the recorded panic stamp (the libtest harness model
    /// clears it before each test so a stale stamp is never read
    /// for a later test).
    pub(crate) fn clear_panic_stamp(&self) {
        *self.last_panic_stamp.borrow_mut() = None;
    }

    /// The (file, line, column) a location stamp value carries:
    /// a `Location` value whose fields are the file string
    /// (possibly wrapped as the one-field enum a `&str` constant
    /// decodes to), the line and the column.
    fn location_of_stamp(stamp: &Value) -> Option<(String, u64, u64)> {
        fn fields_of(v: &Value) -> Option<&Vec<Value>> {
            match v {
                Value::Ref(RefTarget::Value(inner, projection)) if projection.is_empty() => {
                    fields_of(inner)
                }
                Value::Enum { fields, .. } | Value::Tuple(fields) => Some(fields),
                _ => None,
            }
        }
        let fields = fields_of(stamp)?;
        if fields.len() < 3 {
            return None;
        }
        let file = match &fields[0] {
            Value::Str(s) => s.clone(),
            inner => match fields_of(inner) {
                Some(f) if f.len() == 1 => match &f[0] {
                    Value::Str(s) => s.clone(),
                    _ => return None,
                },
                _ => return None,
            },
        };
        let line = match &fields[1] {
            Value::Uint(n) => *n as u64,
            _ => return None,
        };
        let col = match &fields[2] {
            Value::Uint(n) => *n as u64,
            _ => return None,
        };
        Some((file, line, col))
    }

    /// The (file, line, column) of the panic currently recorded,
    /// read out of its caller-location stamp.
    pub(crate) fn panic_location(&self) -> Option<(String, u64, u64)> {
        let stamp = self.last_panic_stamp.borrow().clone()?;
        Self::location_of_stamp(&stamp)
    }

    /// The panic's location as a report prints it: the recorded
    /// stamp when the raising entry point left one, else the
    /// caller location of the innermost `#[track_caller]` frame
    /// on the stack — the propagation `Location::caller()`
    /// performs for a panic raised inside a track-caller body
    /// (an `unwrap_failed` inside `Option::unwrap`, whose frame
    /// carries the stamp as its caller-location local).
    pub(crate) fn panic_location_in(
        &self,
        frames: &[Frame],
    ) -> Option<(String, u64, u64)> {
        if let Some(loc) = self.panic_location() {
            return Some(loc);
        }
        for frame in frames.iter().rev() {
            let Some(body) = self.program.get(&frame.func) else {
                continue;
            };
            let Some(local) = body.caller_location_local else {
                continue;
            };
            if let Some(stamp) = frame.locals.get(local) {
                if let Some(loc) = Self::location_of_stamp(stamp) {
                    return Some(loc);
                }
            }
        }
        None
    }

    /// Whether a source file is one of the toolchain's library
    /// sources (core / std / alloc under the rustlib tree, in
    /// either the installed or the `/rustc/<hash>` remapped
    /// spelling).
    fn is_library_source(file: &str) -> bool {
        file.contains("/library/core/")
            || file.contains("/library/std/")
            || file.contains("/library/alloc/")
    }

    /// Record the caller location of the operation in the
    /// innermost frame as the panic's stamp: the stamp for a
    /// panic a modelled operation raises directly (an `Assert`
    /// terminator's check, a modelled `Index` bounds check),
    /// where std's `#[track_caller]` propagation places the
    /// location at the operation's own site.
    pub(crate) fn note_call_site_stamp(&self, frames: &[Frame]) {
        // The site std reports is the innermost frame in the
        // guest's own code: a check that fires inside a core or
        // std body (an arithmetic impl's overflow assert, say)
        // is inlined into its caller natively, so its reported
        // location is the call site in the first frame outside
        // the library sources.
        let mut fallback: Option<(String, u32, u32)> = None;
        for frame in frames.iter().rev() {
            let Some(body) = self.program.get(&frame.func) else {
                continue;
            };
            // A track-caller frame's received location is what
            // a panic raised in it reports (std's propagation),
            // ahead of the frame's own operation site.
            if let Some(local) = body.caller_location_local {
                if let Some(received) = frame.locals.get(local) {
                    if Self::location_of_stamp(received).is_some() {
                        self.note_panic_stamp(received.clone());
                        return;
                    }
                }
            }
            let Some(lines) = body.source_lines.get(frame.block) else {
                continue;
            };
            let sl = lines.terminator;
            if !sl.is_known() {
                continue;
            }
            let Some(file) = self.program.source_files.get(sl.file as usize) else {
                continue;
            };
            if fallback.is_none() {
                fallback = Some((file.clone(), sl.line, sl.col));
            }
            if !Self::is_library_source(file) {
                self.note_panic_stamp(Value::Ref(RefTarget::value(Value::Enum {
                    variant: 0,
                    fields: vec![
                        Value::Str(file.clone()),
                        Value::Uint(sl.line as u128),
                        Value::Uint(sl.col as u128),
                    ],
                })));
                return;
            }
        }
        if let Some((file, line, col)) = fallback {
            self.note_panic_stamp(Value::Ref(RefTarget::value(Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Str(file),
                    Value::Uint(line as u128),
                    Value::Uint(col as u128),
                ],
            })));
        }
    }

    /// The panic a modelled bounds check raises (the `Index` /
    /// `IndexMut` operations and their place form): the check
    /// is `core::panicking::panic_bounds_check` reached through
    /// the operation's `#[track_caller]` propagation, so the
    /// panic's location is the operation's call site and its
    /// backtrace entry path is that function's.
    pub(crate) fn bounds_check_panic(
        &self,
        frames: &[Frame],
        len: u128,
        idx: u128,
    ) -> InterpError {
        if self.panic_location().is_none() {
            self.note_call_site_stamp(frames);
        }
        self.note_panic_origin(PanicOrigin::Assert(
            "core::panicking::panic_bounds_check",
        ));
        self.note_panic_no_backtrace(false);
        InterpError::panic(format!(
            "index out of bounds: the len is {len} but the index is {idx}"
        ))
    }

    /// Record the entry path of a panic being raised (see
    /// [`Interpreter::last_panic_origin`]).
    pub(crate) fn note_panic_origin(&self, origin: PanicOrigin) {
        *self.last_panic_origin.borrow_mut() = Some(origin);
    }

    /// Record whether the panic being raised forbids a
    /// backtrace (see [`Interpreter::last_panic_no_backtrace`]).
    pub(crate) fn note_panic_no_backtrace(&self, no_backtrace: bool) {
        self.last_panic_no_backtrace.set(no_backtrace);
    }

    /// Record that `std::panic::always_abort` has run.
    pub(crate) fn note_always_abort(&self) {
        self.always_abort.set(true);
    }

    /// Whether `std::panic::always_abort` has run.
    pub(crate) fn always_abort(&self) -> bool {
        self.always_abort.get()
    }

    /// Whether the panic hook is currently running.
    pub(crate) fn in_panic_hook(&self) -> bool {
        self.in_panic_hook
    }

    /// Mark the panic hook as running (or done). The message
    /// formatting a panic entry point performs stands in for
    /// the formatting std performs inside the hook (the
    /// payload's lazy `fill`), so the entry points hold this
    /// mark while they format: a panic the formatting raises
    /// is then a panic while processing a panic, as natively.
    pub(crate) fn set_in_panic_hook(&mut self, running: bool) {
        self.in_panic_hook = running;
    }

    /// Suppress (or restore) the default panic reports while an
    /// enclosing model produces them itself (the libtest
    /// harness model, while it runs tests).
    pub(crate) fn set_panic_report_suppressed(&self, suppressed: bool) {
        self.panic_report_suppressed.set(suppressed);
    }

    /// Whether the default panic reports are suppressed (see
    /// [`Interpreter::panic_report_suppressed`]).
    pub(crate) fn panic_report_suppressed(&self) -> bool {
        self.panic_report_suppressed.get()
    }

    /// Record that the in-flight panic on a thread was consumed
    /// (see [`Interpreter::panics_in_flight`]): it stops counting
    /// towards the next panic's forced-full-backtrace rule.
    pub(crate) fn note_panic_finished(&self, tid: u64) {
        let mut counts = self.panics_in_flight.borrow_mut();
        if let Some(n) = counts.get_mut(&tid) {
            *n = n.saturating_sub(1);
        }
    }

    /// The backtrace style one panic report prints, resolved
    /// the way std's `default_hook` resolves it: a
    /// force-no-backtrace panic prints nothing further; a panic
    /// raised while another is in flight on the thread forces
    /// the full style; otherwise the style installed by
    /// `set_backtrace_style` wins over the environment
    /// (`RUST_LIB_BACKTRACE` first — only its `0` / `1` / `full`
    /// spellings count — then `RUST_BACKTRACE`: unset or `0` is
    /// off, `full` is full, anything else is short). The
    /// environment read is the process's own, which is the
    /// guest's: the driver runs the guest under the environment
    /// the guest program itself observes (a self-spawned child
    /// is a new driver process under the child's environment).
    fn report_backtrace(&self, in_flight: usize, no_backtrace: bool) -> ReportBacktrace {
        if no_backtrace {
            return ReportBacktrace::None;
        }
        if in_flight >= 2 {
            return ReportBacktrace::Full;
        }
        if let Some(style) = self.backtrace_style.borrow().as_ref() {
            let variant = match style {
                Value::Enum { variant, .. } => Some(*variant as u64),
                Value::Uint(n) => Some(*n as u64),
                Value::Int(n) => Some(*n as u64),
                _ => None,
            };
            // `std::panic::BacktraceStyle` declares its
            // variants Short, Full, Off in that order.
            return match variant {
                Some(0) => ReportBacktrace::Short,
                Some(1) => ReportBacktrace::Full,
                Some(_) => ReportBacktrace::Off,
                None => ReportBacktrace::Off,
            };
        }
        match std::env::var("RUST_LIB_BACKTRACE").ok().as_deref() {
            Some("0") => return ReportBacktrace::Off,
            Some("1") => return ReportBacktrace::Short,
            Some("full") => return ReportBacktrace::Full,
            _ => {}
        }
        match std::env::var("RUST_BACKTRACE").ok().as_deref() {
            None | Some("0") => ReportBacktrace::Off,
            Some("full") => ReportBacktrace::Full,
            Some(_) => ReportBacktrace::Short,
        }
    }

    /// Render the guest's stack as a backtrace in std's print
    /// shape for the style: the panic-machinery frames the
    /// panic's entry path implies, then the guest frames from
    /// the innermost out to the current thread's stack base.
    /// Frame source positions are not rendered (std prints one
    /// `at` line under each frame; the model's judging surface
    /// does not read them, and the snapshots that compare
    /// traces normalize them away).
    fn render_backtrace(
        &self,
        frames: &[Frame],
        style: ReportBacktrace,
        origin: &Option<PanicOrigin>,
    ) -> String {
        let mut names: Vec<String> = Vec::new();
        match origin {
            Some(PanicOrigin::BeginPanic(ty)) => {
                names.push(format!("std::panicking::begin_panic::<{ty}>"));
            }
            Some(PanicOrigin::PanicAny(ty)) => {
                names.push(format!("std::panicking::begin_panic::<{ty}>"));
                names.push(format!("std::panic::panic_any::<{ty}>"));
            }
            Some(PanicOrigin::PanicFmt) => {
                names.push("__rustc::rust_begin_unwind".to_string());
                names.push("core::panicking::panic_fmt".to_string());
            }
            Some(PanicOrigin::ExpectFailed) => {
                names.push("__rustc::rust_begin_unwind".to_string());
                names.push("core::panicking::panic_fmt".to_string());
                names.push("core::panicking::panic_display::<&str>".to_string());
                names.push("core::option::expect_failed".to_string());
            }
            Some(PanicOrigin::OptionUnwrap) => {
                names.push("__rustc::rust_begin_unwind".to_string());
                names.push("core::panicking::panic_fmt".to_string());
                names.push("core::panicking::panic".to_string());
                names.push("core::option::unwrap_failed".to_string());
            }
            Some(PanicOrigin::ResultUnwrap) => {
                names.push("__rustc::rust_begin_unwind".to_string());
                names.push("core::panicking::panic_fmt".to_string());
                names.push("core::result::unwrap_failed".to_string());
            }
            Some(PanicOrigin::AssertFailed(frame)) => {
                names.push("__rustc::rust_begin_unwind".to_string());
                names.push("core::panicking::panic_fmt".to_string());
                names.push("core::panicking::assert_failed_inner".to_string());
                if !frame.is_empty() {
                    names.push(frame.clone());
                }
            }
            Some(PanicOrigin::Assert(check)) => {
                names.push("__rustc::rust_begin_unwind".to_string());
                names.push("core::panicking::panic_fmt".to_string());
                names.push(check.to_string());
            }
            Some(PanicOrigin::PanicFn) | None => {
                // `core::panicking::panic` raises through
                // `panic_fmt` (measured at both opt levels);
                // the `begin_panic::<&str>` entry belongs to
                // the separate `std::panicking::begin_panic`
                // origin below.
                names.push("__rustc::rust_begin_unwind".to_string());
                names.push("core::panicking::panic_fmt".to_string());
                names.push("core::panicking::panic".to_string());
            }
        }
        let base = self.thread_stack_base.get().min(frames.len());
        for frame in frames[base..].iter().rev() {
            // std's own backtrace plumbing is not part of the
            // panicking thread's trace (its frames are the
            // capture machinery's, which the trace omits).
            if frame.func.starts_with("std::backtrace")
                || frame.func.starts_with("std::sys::backtrace")
            {
                continue;
            }
            names.push(frame.func.clone());
        }
        let mut out = String::from("stack backtrace:
");
        match style {
            ReportBacktrace::Short => {
                for (i, name) in names.iter().enumerate() {
                    out.push_str(&format!("   {i}: {name}
"));
                }
                out.push_str(
                    "note: Some details are omitted, run with `RUST_BACKTRACE=full` \
                     for a verbose backtrace.
",
                );
            }
            ReportBacktrace::Full => {
                for (i, name) in names.iter().enumerate() {
                    // The addresses are the model's own: no
                    // judging surface reads them (the verbose
                    // shape is what the tests recognize).
                    let addr = 0x5555_5555_4000u64 + (i as u64) * 0x10;
                    out.push_str(&format!("   {i}:     0x{addr:x} - {name}
"));
                }
            }
            _ => {}
        }
        out
    }

    /// The report head for one panic: the leading newline, the
    /// thread's name and id, the location (or none, for a panic
    /// raised without one) and the message.
    fn panic_report_head_text(
        &self,
        location: Option<(String, u64, u64)>,
        message: &str,
    ) -> String {
        let name = self
            .thread_name(self.current_thread_id())
            .unwrap_or_else(|| "<unnamed>".to_string());
        let tid = self.current_thread_id();
        match location {
            Some((file, line, col)) => format!(
                "
thread '{name}' ({tid}) panicked at {file}:{line}:{col}:
{message}
"
            ),
            None => format!("
thread '{name}' ({tid}) panicked:
{message}
"),
        }
    }

    /// The backtrace part of one default report, appended after
    /// its head: a rendered trace, or — when the style is off —
    /// the note, which only the process's first Off-style
    /// report prints (std's FIRST_PANIC).
    fn print_report_backtrace_part(
        &mut self,
        frames: &[Frame],
        in_flight: usize,
        origin: &Option<PanicOrigin>,
        no_backtrace: bool,
    ) {
        match self.report_backtrace(in_flight, no_backtrace) {
            ReportBacktrace::None => {}
            ReportBacktrace::Off => {
                if !self.panic_note_spent.get() {
                    self.panic_note_spent.set(true);
                    self.push_error_output(
                        "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
                    );
                }
            }
            style => {
                let text = self.render_backtrace(frames, style, origin);
                self.push_error_output(&text);
            }
        }
    }

    /// Print std's default panic report for a freshly raised
    /// panic to the guest's stderr: the head (thread name and
    /// id, location, message) and the backtrace part the
    /// resolved style prescribes.
    fn print_default_panic_report(
        &mut self,
        frames: &[Frame],
        message: &str,
        in_flight: usize,
        origin: &Option<PanicOrigin>,
        no_backtrace: bool,
    ) {
        let head = self.panic_report_head_text(self.panic_location_in(frames), message);
        self.push_error_output(&head);
        self.print_report_backtrace_part(frames, in_flight, origin, no_backtrace);
    }

    /// Fire the hook machinery for one of the runtime's own
    /// nounwind panics — `panic_cannot_unwind` at a nounwind
    /// boundary, `panic_in_cleanup` after a fresh panic in a
    /// cleanup — and print the line that follows it
    /// (`thread caused non-unwinding panic. aborting.`). The
    /// panic's location is a fixed std source position and its
    /// payload a `&str`, as in std, where `core::panicking`
    /// raises both itself. A custom hook replaces the default
    /// report for these too; the abort line follows either way.
    /// The caller turns the run into an abort.
    pub(crate) fn fire_nounwind_abort_panic(
        &mut self,
        frames: &mut Vec<Frame>,
        location: (String, u64, u64),
        message: &str,
        force_no_backtrace: bool,
    ) {
        // The panic that brought the run here is still in
        // flight, so this one counts as a further panic on the
        // thread (its report forces a full backtrace unless it
        // forbids one, exactly as std's count rule prescribes).
        let tid = self.current_thread_id();
        let in_flight = {
            let mut counts = self.panics_in_flight.borrow_mut();
            let n = counts.entry(tid).or_insert(0);
            *n += 1;
            *n
        };
        if self.panic_report_suppressed.get() {
            return;
        }
        let hook = self.panic_hook.borrow().clone();
        let is_custom = hook.as_ref().is_some_and(|h| !is_default_hook(h));
        if is_custom {
            let stamp = Value::Ref(RefTarget::value(Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Str(location.0.clone()),
                    Value::Uint(location.1 as u128),
                    Value::Uint(location.2 as u128),
                ],
            }));
            self.call_panic_hook(frames, message, None, stamp, false);
        } else {
            let head = self.panic_report_head_text(Some(location), message);
            self.push_error_output(&head);
            self.print_report_backtrace_part(
                frames,
                in_flight,
                &Some(PanicOrigin::PanicFn),
                force_no_backtrace,
            );
        }
        self.push_error_output("thread caused non-unwinding panic. aborting.
");
    }

    /// Call the guest's custom panic hook with a modelled
    /// `PanicHookInfo` (payload as a `dyn Any` reference, the
    /// location stamp, `can_unwind`). A panic raised by the
    /// hook itself is discarded here: `fire_panic_hook` has
    /// already printed the processing-panic abort lines for it
    /// and marked the run for abort.
    fn call_panic_hook(
        &mut self,
        frames: &mut Vec<Frame>,
        message: &str,
        payload: Option<PanicPayload>,
        location: Value,
        can_unwind: bool,
    ) {
        let hook = self.panic_hook.borrow().clone();
        let Some(hook) = hook else { return };
        let payload = payload.unwrap_or_else(|| PanicPayload {
            ty: "&str".to_string(),
            value: Box::new(Value::Str(message.to_string())),
        });
        let info = Value::Enum {
            variant: 0,
            fields: vec![
                Value::DynRef {
                    base: Box::new(Value::Enum {
                        variant: 0,
                        fields: vec![(*payload.value).clone()],
                    }),
                    concrete: payload.ty.clone(),
                    traits: vec![
                        "std::any::Any".to_string(),
                        "std::marker::Send".to_string(),
                    ],
                    drop_glue: None,
                },
                location,
                Value::Bool(can_unwind),
            ],
        };
        let info_ref = Value::Ref(RefTarget::value(info));
        let Ok((callable, _)) = self.dyn_callable(frames, &hook) else {
            return;
        };
        self.in_panic_hook = true;
        let result =
            self.call_callable_to_completion(frames, &callable, vec![info_ref]);
        self.in_panic_hook = false;
        // An abort raised inside the hook (a `resume_unwind`
        // the in-hook check aborts, an `always_abort` resume)
        // cannot travel back through this call — its result is
        // discarded by design, as a hook's own failure is — so
        // it is recorded on the run instead: the unwinder
        // converts the in-flight panic into the abort when it
        // sees the mark, which is the process abort std
        // performs at the raise.
        if matches!(result, Err(InterpError::Abort { .. })) {
            self.hook_abort.set(true);
        }
    }

    /// Mark the panic about to be raised as a resumed one (the
    /// `resume_unwind` builtin): the unwinder reads and clears
    /// the flag, and the hook does not fire for it.
    pub(crate) fn note_resumed_panic(&mut self) {
        self.resume_signaled = true;
    }

    /// Record that the in-flight panic was consumed by a
    /// builtin (the libtest runner catching a test's panic), so
    /// a later panic is not mistaken for its propagation.
    pub(crate) fn note_panic_consumed(&mut self) {
        self.hook_fired_depth = None;
        let tid = self.current_thread_id();
        self.note_panic_finished(tid);
    }

    /// Execute the `catch_unwind` intrinsic's protocol: run
    /// `try_fn(data)`; when it returns, report `false` (no panic).
    /// When it panics, the nested drive has already unwound the
    /// panicking frames to the depth at entry, running each
    /// frame's cleanup blocks (the destructors of its live
    /// locals) on the way, exactly as native unwinding does; the
    /// stack is then at the entry depth. Run
    /// `catch_fn(data, payload)` with the panic's raw payload and
    /// report `true`. A panic raised by the catch function itself,
    /// an abort (a panic that escaped a cleanup), and any
    /// non-panic error, propagate unchanged. State outside the
    /// frame stack (the statics store, accumulated guest output)
    /// is not rolled back: writes made before the panic persist,
    /// as they do natively.
    pub(crate) fn exec_catch_unwind(
        &mut self,
        frames: &mut Vec<Frame>,
        try_fn: &str,
        data: Value,
        catch_fn: &str,
    ) -> InterpResult<bool> {
        let depth = frames.len();
        match self.call_to_completion(frames, try_fn, vec![data.clone()]) {
            Ok(_) => Ok(false),
            Err(InterpError::Panic { message, payload }) => {
                self.hook_fired_depth = None;
                let tid = self.current_thread_id();
                self.note_panic_finished(tid);
                self.truncate_frames(frames, depth);
                // A panic site that did not record a payload
                // carries a core literal-message panic: the payload
                // is the message itself, as a `&str`.
                let payload = payload.unwrap_or_else(|| PanicPayload {
                    ty: "&str".to_string(),
                    value: Box::new(Value::Str(message)),
                });
                self.call_to_completion(
                    frames,
                    catch_fn,
                    vec![data, Value::PanicPayload(payload)],
                )?;
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }

    /// Discard every frame above `depth`, together with any
    /// suspended drop-glue continuations belonging to the
    /// discarded frames (their destructors never resume: the
    /// frames they would have resumed into are gone).
    fn truncate_frames(&mut self, frames: &mut Vec<Frame>, depth: usize) {
        if frames.len() <= depth {
            return;
        }
        let removed: Vec<u64> = frames.drain(depth..).map(|f| f.id).collect();
        self.drop_continuations
            .retain(|cont| !removed.contains(&cont.callee_frame));
    }
}

/// Format an assertion message from its operand values, with
/// core's exact text (`core::panicking`'s messages for the
/// bounds-check and misaligned-dereference assertions).
/// Returns None when a value does not render as an integer,
/// leaving the caller its baked fallback text.
pub(crate) fn format_assert_message(
    kind: crate::ir::AssertFormat,
    vals: &[Value],
) -> Option<String> {
    fn num(v: &Value) -> Option<String> {
        match v {
            Value::Int(i) => Some(i.to_string()),
            Value::Uint(u) => Some(u.to_string()),
            _ => None,
        }
    }
    match kind {
        crate::ir::AssertFormat::BoundsCheck => Some(format!(
            "index out of bounds: the len is {} but the index is {}",
            num(vals.first()?)?,
            num(vals.get(1)?)?,
        )),
        crate::ir::AssertFormat::MisalignedPointerDeref => Some(format!(
            "misaligned pointer dereference: address must be a multiple of {} but is {}",
            num(vals.first()?)?,
            num(vals.get(1)?)?,
        )),
    }
}
