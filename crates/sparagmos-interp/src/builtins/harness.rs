//! The libtest harness model: test descriptors and harness options.



use super::*;
use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter,
};
use crate::value::Value;


/// The `ExitCode` value the harness main returns: the byte
/// wrapped in the two single-field enums of its model shape.
pub(super) fn test_exit_code(byte: u128) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![Value::Uint(byte)],
        }],
    }
}

/// The libtest command line, read from the guest's arguments:
/// positional filters (a test is selected when it matches ANY
/// of them, as a substring, or by equality with `--exact`),
/// `--skip` exclusions, `--list`, `--nocapture`, the
/// ignored-test mode (`--ignored` / `--include-ignored`), and
/// the output format (`--quiet` is libtest's terse format;
/// `--format json` is the JSON Lines event stream, accepted
/// only together with `-Z unstable-options`). Flags the model
/// does not act on (thread counts, colors, log files) are
/// accepted and ignored: they cannot change a sequential
/// run's output.
/// How a run treats `#[ignore]`d tests (libtest's
/// `RunIgnored`): by default they are skipped and reported as
/// ignored; `--include-ignored` runs them with the rest;
/// `--ignored` runs only them, the rest counting as filtered
/// out. When both flags are given the later one wins, as in
/// libtest's own parser.
#[derive(Clone, Copy, PartialEq)]
enum RunIgnored {
    No,
    Yes,
    Only,
}

pub(super) struct HarnessOpts {
    filters: Vec<String>,
    exact: bool,
    skips: Vec<String>,
    pub(super) list: bool,
    pub(super) nocapture: bool,
    pub(super) terse: bool,
    pub(super) json: bool,
    pub(super) show_output: bool,
    run_ignored: RunIgnored,
}

impl HarnessOpts {
    pub(super) fn parse(args: &[String]) -> HarnessOpts {
        let mut opts = HarnessOpts {
            filters: Vec::new(),
            exact: false,
            skips: Vec::new(),
            list: false,
            nocapture: false,
            terse: false,
            json: false,
            show_output: false,
            run_ignored: RunIgnored::No,
        };
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            match arg.as_str() {
                "--list" => opts.list = true,
                "--nocapture" => opts.nocapture = true,
                "--exact" => opts.exact = true,
                "--quiet" => opts.terse = true,
                // A `-Z` flag's value is an unstable option
                // (the gating in the harness arm judges it),
                // never a positional filter.
                "-Z" => i += 1,
                "--include-ignored" => opts.run_ignored = RunIgnored::Yes,
                "--ignored" => opts.run_ignored = RunIgnored::Only,
                "--show-output" => opts.show_output = true,
                "--format" => {
                    i += 1;
                    match args.get(i).map(String::as_str) {
                        Some("terse") => opts.terse = true,
                        Some("json") => opts.json = true,
                        _ => {}
                    }
                }
                "--skip" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        opts.skips.push(v.clone());
                    }
                }
                _ => {
                    if let Some(v) = arg.strip_prefix("--format=") {
                        if v == "terse" {
                            opts.terse = true;
                        } else if v == "json" {
                            opts.json = true;
                        }
                    } else if let Some(v) = arg.strip_prefix("--skip=") {
                        opts.skips.push(v.to_string());
                    } else if !arg.starts_with('-') {
                        opts.filters.push(arg.clone());
                    }
                }
            }
            i += 1;
        }
        opts
    }

    /// Whether ignored tests run in this run (with the rest,
    /// or as the only tests).
    pub(super) fn runs_ignored(&self) -> bool {
        self.run_ignored != RunIgnored::No
    }

    pub(super) fn selected(&self, test: &TestDesc) -> bool {
        if self.run_ignored == RunIgnored::Only && !test.ignore {
            return false;
        }
        if self.skips.iter().any(|s| test.name.contains(s.as_str())) {
            return false;
        }
        if self.filters.is_empty() {
            return true;
        }
        self.filters.iter().any(|f| {
            if self.exact {
                test.name == *f
            } else {
                test.name.contains(f.as_str())
            }
        })
    }
}

/// One entry of the libtest harness's test list, read by
/// `test_main_env_args` (see `Interpreter::read_test_descs`).
pub(super) struct TestDesc {
    pub(super) name: String,
    pub(super) ignore: bool,
    pub(super) ignore_message: Option<String>,
    pub(super) should_panic: ShouldPanic,
    /// The test function's definition site (the descriptor's
    /// source file and the line/column of its name), which the
    /// harness names in its did-not-panic note.
    pub(super) source_file: String,
    pub(super) start_line: u64,
    pub(super) start_col: u64,
    pub(super) end_line: u64,
    pub(super) end_col: u64,
    pub(super) callable: Value,
    /// Whether the descriptor's `TestFn` is a bench variant
    /// (`StaticBenchFn`, `StaticBenchAsTestFn`, `DynBenchFn`,
    /// `DynBenchAsTestFn`): its callable takes a
    /// `&mut test::Bencher`. A `--test` binary's descriptors
    /// carry the as-test variants, which the harness runs as
    /// tests with a single-iteration bencher.
    pub(super) bench: bool,
}

/// The descriptor's should-panic mode: libtest's `ShouldPanic`
/// enum, whose message-carrying variant holds the expected
/// substring a panicking test's payload must contain.
#[derive(Clone, PartialEq)]
pub(super) enum ShouldPanic {
    No,
    Yes,
    YesWithMessage(String),
}

impl<'a> Interpreter<'a> {
    /// The tests of a `&[TestDescAndFn]` slice, read out of the
    /// value the harness main hands to `test_main_env_args`.
    /// The descriptor layout below was measured against the
    /// descriptors this toolchain's `--test` harness emits
    /// (twelve fields): 0 the `TestName`, 1 the ignore flag,
    /// 2 the ignore message (`Option`: variant 1 carries the
    /// string), 3 the source file, 4..7 the definition span
    /// (start line/column, end line/column, the start naming
    /// the function's name), 8 the `ShouldPanic` mode (variant
    /// 0 `No`, 1 `Yes`, 2 `YesWithMessage` with the expected
    /// substring as its payload). The second field of the
    /// element is the `TestFn` enum whose payload is the
    /// callable.
    pub(super) fn read_test_descs(
        &self,
        frames: &[Frame],
        slice: &Value,
    ) -> InterpResult<Vec<TestDesc>> {
        let target = self.deref_value(frames, slice)?;
        let array = self.read_resolved(frames, &target)?;
        let elements = match &array {
            Value::Tuple(items) | Value::Enum { fields: items, .. } => items.clone(),
            other => {
                return Err(InterpError::type_mismatch(
                    "test::test_main_env_args",
                    "a slice of test descriptors",
                    other.kind_name(),
                ))
            }
        };
        let mut out = Vec::with_capacity(elements.len());
        for element in &elements {
            let element = self.resolve_value(frames, element)?;
            let pair = match &element {
                Value::Tuple(items) | Value::Enum { fields: items, .. } if items.len() == 2 => {
                    items
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "test::test_main_env_args",
                        "a TestDescAndFn value",
                        other.kind_name(),
                    ))
                }
            };
            let desc_value = self.resolve_value(frames, &pair[0])?;
            let desc = match &desc_value {
                Value::Tuple(items) | Value::Enum { fields: items, .. } if items.len() >= 3 => {
                    items
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "test::test_main_env_args",
                        "a TestDesc value",
                        other.kind_name(),
                    ))
                }
            };
            let name = match &desc[0] {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    self.resolve_string(frames, &fields[0])?
                }
                other => self.resolve_string(frames, other)?,
            };
            let ignore = matches!(desc.get(1), Some(Value::Bool(true)));
            // A descriptor string field, read through the
            // one-field enum a string constant may decode to.
            let desc_string = |v: &Value| -> Option<String> {
                match v {
                    Value::Enum { fields, .. } if fields.len() == 1 => {
                        self.resolve_string(frames, &fields[0]).ok()
                    }
                    other => self.resolve_string(frames, other).ok(),
                }
            };
            let ignore_message = match desc.get(2) {
                Some(Value::Enum { variant: 1, fields }) if !fields.is_empty() => {
                    desc_string(&fields[0])
                }
                _ => None,
            };
            let source_file = desc.get(3).and_then(desc_string).unwrap_or_default();
            let desc_uint = |i: usize| -> u64 {
                match desc.get(i) {
                    Some(Value::Uint(n)) => *n as u64,
                    _ => 0,
                }
            };
            let should_panic = match desc.get(8) {
                Some(Value::Enum { variant: 1, .. }) => ShouldPanic::Yes,
                Some(Value::Enum { variant: 2, fields }) if !fields.is_empty() => {
                    match desc_string(&fields[0]) {
                        Some(msg) => ShouldPanic::YesWithMessage(msg),
                        None => ShouldPanic::Yes,
                    }
                }
                _ => ShouldPanic::No,
            };
            let (callable, bench) = match &pair[1] {
                Value::Enum {
                    variant,
                    fields,
                } if fields.len() == 1 => (
                    self.resolve_value(frames, &fields[0])?,
                    matches!(variant, 1 | 2 | 4 | 5),
                ),
                other => (self.resolve_value(frames, other)?, false),
            };
            out.push(TestDesc {
                name,
                ignore,
                ignore_message,
                should_panic,
                source_file,
                start_line: desc_uint(4),
                start_col: desc_uint(5),
                end_line: desc_uint(6),
                end_col: desc_uint(7),
                callable,
                bench,
            });
        }
        Ok(out)
    }
}

/// The exit code a `panic=abort` harness's spawned test
/// process reports success with (libtest's `TR_OK`): the
/// parent maps exactly this code to a pass, and any other
/// code to a `got unexpected return code` failure.
pub(super) const TEST_CHILD_OK: i32 = 50;

/// The `&mut test::Bencher` a bench-as-test descriptor's
/// callable is invoked with: `BenchMode::Single` (variant 1),
/// under which the converted `Bencher::iter` body runs its
/// closure exactly once; no summary yet, no bytes counted.
/// The aggregate is the struct's model shape (variant 0 with
/// the fields in declaration order).
pub(super) fn bencher_value() -> Value {
    Value::Ref(crate::value::RefTarget::value(Value::Enum {
        variant: 0,
        fields: vec![
            Value::Enum {
                variant: 1,
                fields: Vec::new(),
            },
            Value::Enum {
                variant: 0,
                fields: Vec::new(),
            },
            Value::Uint(0),
        ],
    }))
}

/// The panic report of a `panic=abort` harness's spawned test
/// process, as its default hook prints it to the child's
/// stderr: the test runs on the child's main thread, and the
/// backtrace note is the fresh process's to spend, so it is
/// always carried. `thread_id` is the model's main-thread id.
pub(super) fn abort_child_report(
    thread_id: u64,
    location: Option<(String, u64, u64)>,
    message: &str,
) -> String {
    match location {
        Some((file, line, col)) => format!(
            "\nthread 'main' ({thread_id}) panicked at {file}:{line}:{col}:\n{message}\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n"
        ),
        None => format!(
            "\nthread 'main' ({thread_id}) panicked:\n{message}\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n"
        ),
    }
}

/// The verdict of one test run the `panic=abort` way, read
/// by the parent from the spawned process's exit status
/// (libtest's `get_result_from_exit_code`): the success code
/// is a pass; an abort (a signal the parent reads as
/// SIGABRT) is a plain failure; any other exit code is a
/// failure with the code as its note.
enum AbortVerdict {
    Ok,
    Aborted,
    ExitCode(i32),
}

/// One spawned test process's run, as the parent accounts
/// for it: the verdict, the child's captured stdout and
/// stderr (the guest's own writes only), and the panic
/// hook's account — the failure-message line a
/// should-panic mismatch prints, followed by the default
/// panic report — which the child's stderr gains at the
/// panic, after the guest's own writes. In nocapture mode
/// nothing is captured and the account is printed live
/// instead of folded into a failure section.
struct AbortChildOutcome {
    verdict: AbortVerdict,
    cap_out: String,
    cap_err: String,
    hook_account: String,
    panicked: bool,
}

impl<'a> Interpreter<'a> {
    /// Run one test the `panic=abort` harness's way: the
    /// callable is invoked in this process (a bench
    /// descriptor with the fabricated single-iteration
    /// bencher), with the harness synthesizing the child's
    /// panic account, so the general default-report printing
    /// is suppressed for the call. A test body's own
    /// `process::exit` is the child's exit; an interpreter
    /// abort is the child's abort.
    fn run_abort_child(
        &mut self,
        frames: &mut Vec<Frame>,
        test: &TestDesc,
        nocapture: bool,
    ) -> InterpResult<AbortChildOutcome> {
        let out_mark = self.output_len();
        let err_mark = self.error_output_len();
        self.clear_panic_stamp();
        self.set_panic_report_suppressed(true);
        let args = if test.bench {
            vec![bencher_value()]
        } else {
            Vec::new()
        };
        let result = self.call_callable_scheduled(frames, &test.callable, args);
        self.set_panic_report_suppressed(false);
        let (cap_out, cap_err) = if nocapture {
            (String::new(), String::new())
        } else {
            (
                self.drain_output_since(out_mark),
                self.drain_error_output_since(err_mark),
            )
        };
        let outcome = match result {
            Ok(_) => match test.should_panic {
                ShouldPanic::No => AbortChildOutcome {
                    verdict: AbortVerdict::Ok,
                    cap_out,
                    cap_err,
                    hook_account: String::new(),
                    panicked: false,
                },
                _ => {
                    // The child prints the did-not-panic
                    // message to its stderr and aborts.
                    let location = if test.source_file.is_empty() {
                        String::new()
                    } else {
                        format!(
                            " at {}:{}:{}",
                            test.source_file, test.start_line, test.start_col
                        )
                    };
                    AbortChildOutcome {
                        verdict: AbortVerdict::Aborted,
                        cap_out,
                        cap_err,
                        hook_account: format!(
                            "test did not panic as expected{location}\n"
                        ),
                        panicked: false,
                    }
                }
            },
            Err(InterpError::Panic { message, payload }) => {
                self.note_panic_consumed();
                let text = payload
                    .as_ref()
                    .and_then(|p| self.resolve_string(frames, &p.value).ok())
                    .unwrap_or_else(|| message.clone());
                let is_string = match &payload {
                    None => true,
                    Some(p) => {
                        p.ty == "&str"
                            || p.ty == "std::string::String"
                            || self.resolve_string(frames, &p.value).is_ok()
                    }
                };
                let ty = payload.as_ref().map(|p| p.ty.clone()).unwrap_or_default();
                self.abort_panic_outcome(test, cap_out, cap_err, &message, &text, is_string, &ty)
            }
            Err(InterpError::Abort { message }) => {
                match abort_panic_message(&message) {
                    // A panic in a `panic=abort` body converts
                    // to the run's abort at the first nounwind
                    // frame, carrying the panic's message in
                    // the abort's account and dropping its
                    // payload: the verdict is the panic
                    // verdict, with the message standing for
                    // the payload (a string payload's message
                    // is its text).
                    Some(panic_message) => {
                        self.note_panic_consumed();
                        let text = panic_message.clone();
                        self.abort_panic_outcome(
                            test,
                            cap_out,
                            cap_err,
                            &panic_message,
                            &text,
                            true,
                            "",
                        )
                    }
                    None => AbortChildOutcome {
                        verdict: AbortVerdict::Aborted,
                        cap_out,
                        cap_err,
                        hook_account: String::new(),
                        panicked: false,
                    },
                }
            }
            Err(InterpError::ProcessExit { code }) => AbortChildOutcome {
                verdict: if code == TEST_CHILD_OK {
                    AbortVerdict::Ok
                } else {
                    AbortVerdict::ExitCode(code)
                },
                cap_out,
                cap_err,
                hook_account: String::new(),
                panicked: false,
            },
            Err(e) => return Err(e),
        };
        Ok(outcome)
    }

    /// The outcome of a test whose run ended in a panic:
    /// the child's hook verdict (libtest's `calc_result`)
    /// over the descriptor's should-panic mode. `text` is
    /// the payload's string form where it has one (for the
    /// expected-substring check), `is_string` whether the
    /// payload is a string at all, and `ty` its type name
    /// for the non-string account.
    #[allow(clippy::too_many_arguments)]
    fn abort_panic_outcome(
        &self,
        test: &TestDesc,
        cap_out: String,
        cap_err: String,
        message: &str,
        text: &str,
        is_string: bool,
        ty: &str,
    ) -> AbortChildOutcome {
        let report = abort_child_report(
            self.current_thread_id(),
            self.panic_location(),
            message,
        );
        match &test.should_panic {
            ShouldPanic::No => AbortChildOutcome {
                verdict: AbortVerdict::Aborted,
                cap_out,
                cap_err,
                hook_account: report,
                panicked: true,
            },
            ShouldPanic::Yes => AbortChildOutcome {
                verdict: AbortVerdict::Ok,
                cap_out,
                cap_err,
                hook_account: report,
                panicked: true,
            },
            ShouldPanic::YesWithMessage(expected) => {
                if is_string && text.contains(expected.as_str()) {
                    AbortChildOutcome {
                        verdict: AbortVerdict::Ok,
                        cap_out,
                        cap_err,
                        hook_account: report,
                        panicked: true,
                    }
                } else if !is_string {
                    AbortChildOutcome {
                        verdict: AbortVerdict::Aborted,
                        cap_out,
                        cap_err,
                        hook_account: format!(
                            "expected panic with string value,\n found non-string value: `{ty}`\n     expected substring: \"{expected}\"\n{report}"
                        ),
                        panicked: true,
                    }
                } else {
                    AbortChildOutcome {
                        verdict: AbortVerdict::Aborted,
                        cap_out,
                        cap_err,
                        hook_account: format!(
                            "panic did not contain expected string\n      panic message: \"{text}\"\n expected substring: \"{expected}\"\n{report}"
                        ),
                        panicked: true,
                    }
                }
            }
        }
    }

    /// The spawned-process entry of the `panic=abort`
    /// harness (see the dispatch arm): run the named test in
    /// this process, print the hook's account to the guest's
    /// stderr as the child's hook does, and leave the process
    /// the way the child does — the success exit code, an
    /// abort, or the test body's own exit. Never returns.
    pub(super) fn run_abort_secondary(
        &mut self,
        frames: &mut Vec<Frame>,
        test: &TestDesc,
    ) -> InterpResult<Value> {
        let outcome = self.run_abort_child(frames, test, true)?;
        if !outcome.hook_account.is_empty() {
            self.push_error_output(&outcome.hook_account);
        }
        match outcome.verdict {
            AbortVerdict::Ok => Err(InterpError::ProcessExit {
                code: TEST_CHILD_OK,
            }),
            AbortVerdict::Aborted => Err(InterpError::Abort {
                message: format!(
                    "test `{}` aborted in its spawned subprocess",
                    test.name
                ),
            }),
            AbortVerdict::ExitCode(code) => {
                Err(InterpError::ProcessExit { code })
            }
        }
    }

    /// The `panic=abort` harness's parent run: each selected
    /// test is run as a spawned child (see
    /// [`Self::run_abort_child`]) and accounted for from its
    /// exit status, in the default (pretty) format. The json
    /// and terse formats of this entry are not modelled; the
    /// run stops with an explicit unsupported error rather
    /// than emit a format it cannot state.
    pub(super) fn run_test_main_abort(
        &mut self,
        frames: &mut Vec<Frame>,
        tests: Vec<TestDesc>,
    ) -> InterpResult<Value> {
        // The `-Z` gate and the json refusal are libtest's
        // own, shared with the unwinding entry (see the
        // `TestMainEnvArgs` arm): a `-Z` option is refused
        // before anything else happens, and the json format
        // requires `-Z unstable-options` under a bootstrap.
        let raw_args: Vec<String> = self.guest_args().to_vec();
        let opts = HarnessOpts::parse(&raw_args);
        let bootstrap = std::env::var("RUSTC_BOOTSTRAP").is_ok();
        let mut unstable_options = false;
        let mut other_z = false;
        {
            let mut j = 0;
            while j < raw_args.len() {
                let a = &raw_args[j];
                if a == "-Zunstable-options" {
                    unstable_options = true;
                } else if a == "-Z" {
                    if raw_args.get(j + 1).map(String::as_str)
                        == Some("unstable-options")
                    {
                        unstable_options = true;
                        j += 1;
                    } else {
                        other_z = true;
                    }
                } else if a.starts_with("-Z") {
                    other_z = true;
                }
                j += 1;
            }
        }
        if other_z || (unstable_options && !bootstrap) {
            self.push_error_output(
                "error: the option `Z` is only accepted on the nightly compiler\n",
            );
            return Ok(test_exit_code(101));
        }
        if opts.json && !unstable_options {
            self.push_error_output(
                "error: The \"json\" format is only accepted on the nightly \
                 compiler with -Z unstable-options\n",
            );
            return Ok(test_exit_code(101));
        }
        if opts.json || opts.terse {
            return Err(unsupported(
                "test::test_main_env_args_abort with the json or terse output format",
            ));
        }
        let mut selected: Vec<&TestDesc> = Vec::new();
        let mut filtered_out = 0usize;
        for test in &tests {
            if opts.selected(test) {
                selected.push(test);
            } else {
                filtered_out += 1;
            }
        }
        if opts.list {
            for test in &selected {
                self.push_output(&format!("{}: test\n", test.name));
            }
            self.push_output(&format!(
                "\n{} tests, 0 benchmarks\n",
                selected.len()
            ));
            return Ok(test_exit_code(0));
        }
        // libtest's concurrency setup reads RUST_TEST_THREADS
        // before any test runs and panics on a value that is
        // not a positive integer (see the unwinding arm).
        if let Ok(v) = std::env::var("RUST_TEST_THREADS") {
            if !matches!(v.parse::<usize>(), Ok(n) if n > 0) {
                return Err(InterpError::panic(format!(
                    "RUST_TEST_THREADS is `{v}`, should be a positive integer."
                )));
            }
        }
        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut ignored = 0usize;
        let mut failure_names: Vec<String> = Vec::new();
        let mut failure_sections: Vec<String> = Vec::new();
        self.push_output(&format!(
            "\nrunning {} {}\n",
            selected.len(),
            if selected.len() == 1 { "test" } else { "tests" }
        ));
        for test in selected {
            let annotation = match test.should_panic {
                ShouldPanic::No => "",
                _ => " - should panic",
            };
            if test.ignore && !opts.runs_ignored() {
                ignored += 1;
                let note = match &test.ignore_message {
                    Some(msg) => format!(", {msg}"),
                    None => String::new(),
                };
                self.push_output(&format!(
                    "test {}{} ... ignored{}\n",
                    test.name, annotation, note
                ));
                continue;
            }
            if opts.nocapture {
                self.push_output(&format!(
                    "test {}{} ... ",
                    test.name, annotation
                ));
            }
            let outcome = self.run_abort_child(frames, test, opts.nocapture)?;
            // The failure section's body: the child's stdout,
            // the stderr delimiter (which first terminates a
            // partial line), the child's stderr, and the
            // hook's account. In nocapture mode nothing was
            // captured, so the body is the delimiter alone
            // and the account was printed live instead.
            let section_body = |outcome: &AbortChildOutcome| -> String {
                let mut body = outcome.cap_out.clone();
                if !body.is_empty() && !body.ends_with('\n') {
                    body.push('\n');
                }
                body.push_str(&format!("---- {} stderr ----\n", test.name));
                body.push_str(&outcome.cap_err);
                body.push_str(&outcome.hook_account);
                body
            };
            match outcome.verdict {
                AbortVerdict::Ok => {
                    passed += 1;
                    if opts.nocapture {
                        if outcome.panicked {
                            self.push_error_output(&outcome.hook_account);
                        }
                        self.push_output("ok\n");
                    } else {
                        self.push_output(&format!(
                            "test {}{} ... ok\n",
                            test.name, annotation
                        ));
                    }
                }
                AbortVerdict::Aborted => {
                    failed += 1;
                    failure_names.push(test.name.clone());
                    if opts.nocapture {
                        if !outcome.hook_account.is_empty() {
                            self.push_error_output(&outcome.hook_account);
                        }
                        failure_sections.push(format!(
                            "\n---- {} stdout ----\n---- {} stderr ----\n",
                            test.name, test.name
                        ));
                        self.push_output("FAILED\n");
                    } else {
                        failure_sections.push(format!(
                            "\n---- {} stdout ----\n{}",
                            test.name,
                            section_body(&outcome)
                        ));
                        self.push_output(&format!(
                            "test {}{} ... FAILED\n",
                            test.name, annotation
                        ));
                    }
                }
                AbortVerdict::ExitCode(code) => {
                    failed += 1;
                    failure_names.push(test.name.clone());
                    if opts.nocapture {
                        failure_sections.push(format!(
                            "\n---- {} stdout ----\n---- {} stderr ----\nnote: got unexpected return code {code}",
                            test.name, test.name
                        ));
                        self.push_output("FAILED\n");
                    } else {
                        let mut body = section_body(&outcome);
                        body.push_str(&format!(
                            "note: got unexpected return code {code}"
                        ));
                        failure_sections.push(format!(
                            "\n---- {} stdout ----\n{}",
                            test.name, body
                        ));
                        self.push_output(&format!(
                            "test {}{} ... FAILED\n",
                            test.name, annotation
                        ));
                    }
                }
            }
        }
        if !failure_names.is_empty() {
            self.push_output("\nfailures:\n");
            for section in &failure_sections {
                self.push_output(section);
            }
            if !failure_sections.is_empty() {
                self.push_output("\n");
            }
            self.push_output("\nfailures:\n");
            for name in &failure_names {
                self.push_output(&format!("    {name}\n"));
            }
        }
        let verdict = if failed == 0 { "ok" } else { "FAILED" };
        self.push_output(&format!(
            "\ntest result: {verdict}. {passed} passed; {failed} failed; \
             {ignored} ignored; 0 measured; {filtered_out} filtered out; \
             finished in 0.00s\n\n"
        ));
        let byte = if failed == 0 { 0 } else { 101 };
        Ok(test_exit_code(byte))
    }
}

/// The panic message an abort's account carries when the
/// abort is a panic that reached a nounwind boundary: the
/// drive states it as `panic \`<message>\` may not unwind
/// past ...` (or `... raised in a cleanup of ...` when the
/// boundary is a cleanup block, and `panic \`<message>\`
/// after \`panic::always_abort\`` for that arm). `None` for
/// an abort of any other origin — a signal, an allocation
/// failure, an explicit `abort()` — which the harness reads
/// as a plain failure with no panic account.
fn abort_panic_message(abort: &str) -> Option<String> {
    let rest = abort.strip_prefix("panic `")?;
    for marker in [
        "` may not unwind past `",
        "` raised in a cleanup of `",
        "` after `panic::always_abort`",
    ] {
        if let Some(idx) = rest.rfind(marker) {
            return Some(rest[..idx].to_string());
        }
    }
    None
}

/// Registration fragment of the builtin name registry: the test harness entry point.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn harness_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        // The harness calls either a crate-local wrapper
        // (`<crate>::test::test_main_env_args`) or, for an empty
        // test list, the `test` crate's function directly.
        _ if canonical == "test::test_main_env_args"
            || canonical.ends_with("::test::test_main_env_args") =>
        {
            Builtin::TestMainEnvArgs
        }
        _ if canonical == "test::test_main_env_args_abort"
            || canonical.ends_with("::test::test_main_env_args_abort") =>
        {
            Builtin::TestMainEnvArgsAbort
        }
        _ => return None,
    })
}
