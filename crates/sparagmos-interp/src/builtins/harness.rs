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
            let callable = match &pair[1] {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    self.resolve_value(frames, &fields[0])?
                }
                other => self.resolve_value(frames, other)?,
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
            });
        }
        Ok(out)
    }
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
        _ => return None,
    })
}
