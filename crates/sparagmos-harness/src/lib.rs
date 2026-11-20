//! sparagmos-harness: corpus runner and coverage measurement.
//!
//! Honesty contract (project rule):
//! - The *native* result (compiled and run with the reference rustc) is the
//!   expected result. It is always measured for real.
//! - A case is `Passed` only when a Sparagmos execution actually happened
//!   and its stdout / exit code match the native result exactly.
//! - If Sparagmos cannot execute a case yet (parser / interpreter / JIT are
//!   under parallel development), the case is `Unsupported` — never counted
//!   as passed. See [`SparagmosRunner`] for the plug-in point.

pub mod interp_runner;

pub use interp_runner::{default_driver_bin, InterpRunner, DRIVER_BIN_ENV};

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

/// Reference toolchain used for MIR generation and native execution.
/// Rust 1.100.0 (released 2026-11-12) is the target release for Sparagmos;
/// all MIR generation and native oracle runs pin this exact toolchain.
pub const TOOLCHAIN: &str = "1.100.0";

/// Edition used when compiling corpus programs natively / emitting MIR.
pub const EDITION: &str = "2021";

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Result of running a program (native or, later, via Sparagmos).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunResult {
    pub stdout: String,
    pub stderr: String,
    /// `None` if the process was killed by a signal.
    pub exit_code: Option<i32>,
}

impl RunResult {
    pub fn success(&self) -> bool {
        self.exit_code == Some(0)
    }
}

/// One corpus program (`tests/corpus/*.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusCase {
    /// File stem, e.g. `arith_int`.
    pub name: String,
    pub path: PathBuf,
}

impl CorpusCase {
    pub fn new(path: PathBuf) -> Self {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unnamed".to_string());
        Self { name, path }
    }
}

/// Why a Sparagmos execution did not happen / is not supported yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsupportedReason {
    /// No Sparagmos runner has been wired into the harness yet.
    NoRunner,
    /// The runner reported this feature / MIR construct as unsupported.
    Feature(String),
}

impl fmt::Display for UnsupportedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnsupportedReason::NoRunner => {
                write!(f, "no Sparagmos runner wired in yet (interp/JIT pending)")
            }
            UnsupportedReason::Feature(what) => write!(f, "unsupported: {what}"),
        }
    }
}

/// Plug-in point for the execution lanes.
///
/// On the project's main path, MIR for `case.path` is produced by the
/// Stable MIR driver (`sparagmos-driver`) as `sparagmos-mir` bodies
/// and executed by the interpreter / JIT. [`InterpRunner`] is that
/// runner: it drives the whole path through the driver binary's
/// `run` mode. Returning `Err` must stay cheap and honest: when in
/// doubt, report unsupported instead of guessing output.
pub trait SparagmosRunner {
    fn run(&self, case: &CorpusCase) -> Result<RunResult, UnsupportedReason>;
}

/// Default runner used until the interpreter/JIT lanes land.
/// It supports nothing, so every corpus case is reported `Unsupported`.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnsupportedRunner;

impl SparagmosRunner for UnsupportedRunner {
    fn run(&self, _case: &CorpusCase) -> Result<RunResult, UnsupportedReason> {
        Err(UnsupportedReason::NoRunner)
    }
}

/// Outcome of evaluating one corpus case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaseStatus {
    /// Sparagmos ran and matched native exactly.
    Passed,
    /// Sparagmos ran but diverged from native (details in the report entry).
    Failed { detail: String },
    /// Sparagmos could not run this case yet. NOT a pass.
    Unsupported { reason: UnsupportedReason },
    /// The native reference itself failed to build/run; harness problem,
    /// not a Sparagmos result.
    NativeError { detail: String },
}

impl CaseStatus {
    pub fn label(&self) -> &'static str {
        match self {
            CaseStatus::Passed => "PASS",
            CaseStatus::Failed { .. } => "FAIL",
            CaseStatus::Unsupported { .. } => "UNSUPPORTED",
            CaseStatus::NativeError { .. } => "NATIVE-ERROR",
        }
    }
}

impl fmt::Display for CaseStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CaseStatus::Passed => write!(f, "PASS"),
            CaseStatus::Failed { detail } => write!(f, "FAIL ({detail})"),
            CaseStatus::Unsupported { reason } => write!(f, "UNSUPPORTED ({reason})"),
            CaseStatus::NativeError { detail } => write!(f, "NATIVE-ERROR ({detail})"),
        }
    }
}

/// Full evaluation record for one case.
#[derive(Debug, Clone)]
pub struct CaseReport {
    pub case: CorpusCase,
    /// Native expected result, when the native build succeeded.
    pub native: Option<RunResult>,
    /// Sparagmos result, when a runner actually executed the case.
    pub sparagmos: Option<RunResult>,
    pub status: CaseStatus,
}

/// Aggregate counts. `passed + failed + unsupported + native_error == total`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Summary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub unsupported: usize,
    pub native_error: usize,
}

impl Summary {
    /// Pass rate over *all* cases (unsupported counts against it — honest).
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }
    /// Pass rate over only the cases Sparagmos actually attempted.
    pub fn attempted_pass_rate(&self) -> f64 {
        let attempted = self.passed + self.failed;
        if attempted == 0 {
            0.0
        } else {
            self.passed as f64 / attempted as f64
        }
    }
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "total={} passed={} failed={} unsupported={} native_error={} pass_rate={:.1}% (of attempted: {:.1}%)",
            self.total,
            self.passed,
            self.failed,
            self.unsupported,
            self.native_error,
            self.pass_rate() * 100.0,
            self.attempted_pass_rate() * 100.0
        )
    }
}

/// Whole-corpus report.
#[derive(Debug, Clone)]
pub struct CorpusReport {
    pub cases: Vec<CaseReport>,
    pub summary: Summary,
}

impl CorpusReport {
    pub fn from_cases(cases: Vec<CaseReport>) -> Self {
        let mut summary = Summary {
            total: cases.len(),
            ..Summary::default()
        };
        for c in &cases {
            match &c.status {
                CaseStatus::Passed => summary.passed += 1,
                CaseStatus::Failed { .. } => summary.failed += 1,
                CaseStatus::Unsupported { .. } => summary.unsupported += 1,
                CaseStatus::NativeError { .. } => summary.native_error += 1,
            }
        }
        Self { cases, summary }
    }

    /// Minimal JSON rendering (hand-rolled; no external deps by design).
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n  \"summary\": {");
        out.push_str(&format!(
            "\"total\": {}, \"passed\": {}, \"failed\": {}, \"unsupported\": {}, \"native_error\": {}, \"pass_rate\": {:.4}",
            self.summary.total,
            self.summary.passed,
            self.summary.failed,
            self.summary.unsupported,
            self.summary.native_error,
            self.summary.pass_rate()
        ));
        out.push_str("},\n  \"cases\": [\n");
        for (i, c) in self.cases.iter().enumerate() {
            let comma = if i + 1 < self.cases.len() { "," } else { "" };
            out.push_str(&format!(
                "    {{\"name\": {}, \"status\": {}, \"native_exit\": {}, \"sparagmos_exit\": {}}}{}\n",
                json_str(&c.case.name),
                json_str(c.status.label()),
                c.native
                    .as_ref()
                    .map(|r| r.exit_code.map(|e| e.to_string()).unwrap_or_else(|| "null".into()))
                    .unwrap_or_else(|| "null".into()),
                c.sparagmos
                    .as_ref()
                    .map(|r| r.exit_code.map(|e| e.to_string()).unwrap_or_else(|| "null".into()))
                    .unwrap_or_else(|| "null".into()),
                comma
            ));
        }
        out.push_str("  ]\n}\n");
        out
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Default corpus directory for a checkout: `<repo>/tests/corpus`,
/// resolved relative to this crate (`crates/sparagmos-harness`).
pub fn default_corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("corpus")
}

/// Discover all `*.rs` corpus programs in `dir`, sorted by file name.
pub fn discover_corpus(dir: &Path) -> std::io::Result<Vec<CorpusCase>> {
    let mut cases = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "rs").unwrap_or(false) {
            cases.push(CorpusCase::new(path));
        }
    }
    cases.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(cases)
}

fn unique_temp_dir(tag: &str) -> std::io::Result<PathBuf> {
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-harness-{}-{}-{}",
        std::process::id(),
        tag,
        seq
    ));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn run_output(mut cmd: Command) -> std::io::Result<RunResult> {
    let Output {
        status,
        stdout,
        stderr,
    } = cmd.output()?;
    Ok(RunResult {
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        exit_code: status.code(),
    })
}

/// Resolve the rustup binary: PATH first, then the standard
/// `~/.cargo/bin/rustup` location (non-login shells on the dev host do
/// not always have `~/.cargo/bin` on PATH).
fn rustup_bin() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        let candidate = Path::new(&home).join(".cargo").join("bin").join("rustup");
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("rustup")
}

fn rustc_command() -> Command {
    // `rustup run <toolchain> rustc ...`
    let mut cmd = Command::new(rustup_bin());
    cmd.arg("run").arg(TOOLCHAIN).arg("rustc");
    cmd
}

/// Compile `source` with the reference rustc and run the resulting binary.
/// This is the *expected-result oracle*: native behaviour.
pub fn compile_and_run_native(source: &Path) -> Result<RunResult, String> {
    let dir = unique_temp_dir("native").map_err(|e| e.to_string())?;
    let bin = dir.join("prog");
    let mut build = rustc_command();
    build
        .arg("--edition")
        .arg(EDITION)
        .arg("-C")
        .arg("opt-level=0")
        .arg("-o")
        .arg(&bin)
        .arg(source);
    let build_res = run_output(build).map_err(|e| format!("failed to spawn rustc: {e}"))?;
    if !build_res.success() {
        let detail = format!(
            "native rustc failed (exit {:?}): {}",
            build_res.exit_code,
            build_res.stderr.trim()
        );
        let _ = fs::remove_dir_all(&dir);
        return Err(detail);
    }
    let run = run_output(Command::new(&bin));
    let _ = fs::remove_dir_all(&dir);
    run.map_err(|e| format!("failed to run native binary: {e}"))
}

/// Emit textual MIR for `source` with the reference rustc
/// (`rustc --emit=mir`) and return the MIR text.
pub fn emit_mir(source: &Path) -> Result<String, String> {
    let dir = unique_temp_dir("mir").map_err(|e| e.to_string())?;
    let out = dir.join("out.mir");
    let mut cmd = rustc_command();
    cmd.arg("--edition")
        .arg(EDITION)
        .arg("--emit=mir")
        .arg("-o")
        .arg(&out)
        .arg(source);
    let res = run_output(cmd).map_err(|e| format!("failed to spawn rustc: {e}"))?;
    if !res.success() {
        let detail = format!(
            "rustc --emit=mir failed (exit {:?}): {}",
            res.exit_code,
            res.stderr.trim()
        );
        let _ = fs::remove_dir_all(&dir);
        return Err(detail);
    }
    let text = fs::read_to_string(&out).map_err(|e| format!("failed to read MIR output: {e}"))?;
    let _ = fs::remove_dir_all(&dir);
    Ok(text)
}

/// Group native outputs by case name (utility for later diff tooling).
pub fn group_by_status(report: &CorpusReport) -> BTreeMap<&'static str, Vec<String>> {
    let mut map: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    for c in &report.cases {
        map.entry(c.status.label())
            .or_default()
            .push(c.case.name.clone());
    }
    map
}

/// Evaluate one case: native oracle first, then the Sparagmos runner.
pub fn evaluate_case(case: &CorpusCase, runner: &dyn SparagmosRunner) -> CaseReport {
    let native = match compile_and_run_native(&case.path) {
        Ok(r) => r,
        Err(detail) => {
            return CaseReport {
                case: case.clone(),
                native: None,
                sparagmos: None,
                status: CaseStatus::NativeError { detail },
            };
        }
    };
    match runner.run(case) {
        Err(reason) => CaseReport {
            case: case.clone(),
            native: Some(native),
            sparagmos: None,
            status: CaseStatus::Unsupported { reason },
        },
        Ok(got) => {
            let status = if got.stdout == native.stdout && got.exit_code == native.exit_code {
                CaseStatus::Passed
            } else {
                CaseStatus::Failed {
                    detail: format!(
                        "stdout/exit mismatch: native exit={:?} stdout={:?}, sparagmos exit={:?} stdout={:?}",
                        native.exit_code, native.stdout, got.exit_code, got.stdout
                    ),
                }
            };
            CaseReport {
                case: case.clone(),
                native: Some(native),
                sparagmos: Some(got),
                status,
            }
        }
    }
}

/// Evaluate the whole corpus directory with the given runner.
pub fn evaluate_corpus(dir: &Path, runner: &dyn SparagmosRunner) -> std::io::Result<CorpusReport> {
    let cases = discover_corpus(dir)?;
    let reports = cases.iter().map(|c| evaluate_case(c, runner)).collect();
    Ok(CorpusReport::from_cases(reports))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoNativeRunner;
    impl SparagmosRunner for EchoNativeRunner {
        // Test double: replays the native oracle. Only used to exercise the
        // comparison logic — never used for real coverage numbers.
        fn run(&self, case: &CorpusCase) -> Result<RunResult, UnsupportedReason> {
            compile_and_run_native(&case.path).map_err(UnsupportedReason::Feature)
        }
    }

    struct WrongRunner;
    impl SparagmosRunner for WrongRunner {
        fn run(&self, _case: &CorpusCase) -> Result<RunResult, UnsupportedReason> {
            Ok(RunResult {
                stdout: "wrong\n".into(),
                stderr: String::new(),
                exit_code: Some(0),
            })
        }
    }

    fn corpus_dir() -> PathBuf {
        default_corpus_dir()
    }

    #[test]
    fn discovers_corpus_cases() {
        let cases = discover_corpus(&corpus_dir()).expect("corpus dir must exist");
        assert!(
            cases.len() >= 9,
            "expected at least 9 corpus cases, got {}",
            cases.len()
        );
        assert!(cases.iter().any(|c| c.name == "hello"), "hello.rs missing");
        // sorted by name
        let names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn native_oracle_runs_hello() {
        let res = compile_and_run_native(&corpus_dir().join("hello.rs"))
            .expect("native build/run of hello.rs must succeed");
        assert_eq!(res.exit_code, Some(0));
        assert!(res.stdout.contains("hello"), "stdout was: {:?}", res.stdout);
    }

    #[test]
    fn emit_mir_produces_main() {
        let mir = emit_mir(&corpus_dir().join("hello.rs")).expect("MIR emit must succeed");
        assert!(mir.contains("fn main"), "MIR should define main:\n{mir}");
    }

    #[test]
    fn unsupported_is_not_counted_as_passed() {
        let report = evaluate_corpus(&corpus_dir(), &UnsupportedRunner)
            .expect("corpus evaluation must succeed");
        assert_eq!(report.summary.passed, 0, "unsupported must never pass");
        assert_eq!(report.summary.unsupported, report.summary.total);
        assert_eq!(report.summary.native_error, 0);
        assert_eq!(report.summary.pass_rate(), 0.0);
    }

    #[test]
    fn matching_runner_passes_and_wrong_runner_fails() {
        let hello = CorpusCase::new(corpus_dir().join("hello.rs"));
        let ok = evaluate_case(&hello, &EchoNativeRunner);
        assert_eq!(ok.status, CaseStatus::Passed);
        let bad = evaluate_case(&hello, &WrongRunner);
        assert!(matches!(bad.status, CaseStatus::Failed { .. }));
    }

    #[test]
    fn summary_counts_add_up() {
        let s = Summary {
            total: 4,
            passed: 1,
            failed: 1,
            unsupported: 2,
            native_error: 0,
        };
        assert_eq!(
            s.passed + s.failed + s.unsupported + s.native_error,
            s.total
        );
        assert_eq!(s.pass_rate(), 0.25);
        assert_eq!(s.attempted_pass_rate(), 0.5);
    }
}
