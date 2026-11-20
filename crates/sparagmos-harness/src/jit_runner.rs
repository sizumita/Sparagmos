//! The harness's JIT runner: the same main path as the interpreter
//! runner up to the Stable MIR driver, then execution by
//! `sparagmos-jit` instead of the interpreter.
//!
//! Flow: the driver converts the case with Stable MIR and writes
//! its dump (`sparagmos-driver <file.rs> --dump <path>`, format v0 —
//! see [`crate::dump_parse`]); the runner reads the dump back into
//! `sparagmos-mir` bodies, compiles the whole program with the JIT,
//! and calls its `main`. Result mapping follows the harness
//! contract:
//!
//! * the guest terminates → exit 0 with empty stdout (a program
//!   whose output the JIT path could produce would have to print,
//!   and every printing path is outside the JIT's supported range,
//!   so such cases are unsupported at compile time, never faked);
//! * the guest panics → exit 101, the panic message on stderr;
//! * anything the driver, the dump reader, or the JIT cannot handle
//!   → [`UnsupportedReason::Feature`] with the concrete reason.
//!
//! The driver binary resolution matches [`crate::InterpRunner`]:
//! an explicit path, the `SPARAGMOS_DRIVER_BIN` environment
//! variable, then the default in-checkout location.

use std::path::{Path, PathBuf};
use std::process::Command;

use sparagmos_jit::{JitCallError, JitError, JitProgram};

use crate::dump_parse::parse_dump;
use crate::{
    default_driver_bin, CorpusCase, RunResult, SparagmosRunner, UnsupportedReason, DRIVER_BIN_ENV,
};

/// Corpus runner that executes cases through the Stable MIR driver
/// and the JIT (see the module docs).
#[derive(Debug, Clone, Default)]
pub struct JitRunner {
    driver_bin: Option<PathBuf>,
}

impl JitRunner {
    /// Runner resolving the driver binary from `SPARAGMOS_DRIVER_BIN`
    /// or the default in-checkout location.
    pub fn new() -> Self {
        Self::default()
    }

    /// Runner using an explicit driver binary path.
    pub fn with_driver_bin(path: impl Into<PathBuf>) -> Self {
        Self {
            driver_bin: Some(path.into()),
        }
    }

    fn resolve_driver_bin(&self) -> Result<PathBuf, UnsupportedReason> {
        if let Some(p) = &self.driver_bin {
            return found_or_reason(p);
        }
        if let Some(p) = std::env::var_os(DRIVER_BIN_ENV) {
            return found_or_reason(&PathBuf::from(p));
        }
        found_or_reason(&default_driver_bin())
    }
}

fn found_or_reason(p: &Path) -> Result<PathBuf, UnsupportedReason> {
    if p.is_file() {
        Ok(p.to_path_buf())
    } else {
        Err(UnsupportedReason::Feature(format!(
            "driver binary not found: {} (build crates/sparagmos-driver with the pinned \
             1.100.0 toolchain, or set {DRIVER_BIN_ENV})",
            p.display()
        )))
    }
}

fn is_main(name: &str) -> bool {
    name.rsplit("::").next() == Some("main")
}

impl SparagmosRunner for JitRunner {
    fn run(&self, case: &CorpusCase) -> Result<RunResult, UnsupportedReason> {
        let bin = self.resolve_driver_bin()?;
        let dir = crate::unique_temp_dir("jit").map_err(|e| {
            UnsupportedReason::Feature(format!("cannot create a scratch directory: {e}"))
        })?;
        let dump_path = dir.join("dump.txt");
        let result = self.run_with_dump(case, &bin, &dump_path);
        let _ = std::fs::remove_dir_all(&dir);
        result
    }
}

impl JitRunner {
    fn run_with_dump(
        &self,
        case: &CorpusCase,
        bin: &Path,
        dump_path: &Path,
    ) -> Result<RunResult, UnsupportedReason> {
        let output = Command::new(bin)
            .arg(&case.path)
            .arg("--dump")
            .arg(dump_path)
            .output()
            .map_err(|e| {
                UnsupportedReason::Feature(format!(
                    "failed to spawn driver {}: {e}",
                    bin.display()
                ))
            })?;
        if !output.status.success() {
            return Err(UnsupportedReason::Feature(format!(
                "driver failed (exit {:?}): {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        let text = std::fs::read_to_string(dump_path).map_err(|e| {
            UnsupportedReason::Feature(format!(
                "cannot read the driver dump {}: {e}",
                dump_path.display()
            ))
        })?;
        let parsed = parse_dump(&text)
            .map_err(|e| UnsupportedReason::Feature(e.to_string()))?;
        let Some(entry) = parsed
            .program
            .functions
            .iter()
            .map(|f| f.name.clone())
            .find(|n| is_main(n))
        else {
            if let Some((_, reasons)) = parsed.skipped.iter().find(|(n, _)| is_main(n)) {
                return Err(UnsupportedReason::Feature(format!(
                    "driver skipped the entry function: {reasons}"
                )));
            }
            return Err(UnsupportedReason::Feature(
                "no `main` function among the converted bodies".to_string(),
            ));
        };
        let prog = JitProgram::compile(&parsed.program).map_err(|e| match e {
            JitError::Unsupported(what) => UnsupportedReason::Feature(what),
            JitError::Backend(e) => {
                UnsupportedReason::Feature(format!("jit backend error: {e}"))
            }
        })?;
        match prog.call(&entry, &[]) {
            Ok(_) => Ok(RunResult {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(0),
            }),
            Err(JitCallError::Panic(rec)) => Ok(RunResult {
                stdout: String::new(),
                stderr: format!("panicked: {}", rec.message),
                exit_code: Some(101),
            }),
            Err(other) => Err(UnsupportedReason::Feature(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{evaluate_case, evaluate_corpus, CaseStatus};

    fn case(name: &str) -> CorpusCase {
        CorpusCase::new(crate::default_corpus_dir().join(name))
    }

    #[test]
    fn missing_driver_binary_is_unsupported_with_reason() {
        let runner = JitRunner::with_driver_bin("/nonexistent/sparagmos-driver");
        let err = runner
            .run(&case("silent_fib.rs"))
            .expect_err("no binary, no execution");
        match err {
            UnsupportedReason::Feature(what) => {
                assert!(what.contains("driver binary not found"), "reason: {what}");
            }
            other => panic!("expected Feature reason, got {other}"),
        }
    }

    /// End-to-end over the real driver, when one is resolvable: the
    /// silent corpus cases must pass on the JIT path, and no corpus
    /// case may *fail* (diverge) — cases outside the JIT's range are
    /// unsupported with a reason, which is the honest outcome.
    #[test]
    fn real_driver_measures_corpus_via_jit() {
        let runner = JitRunner::new();
        if runner.resolve_driver_bin().is_err() {
            let err = runner
                .run(&case("silent_fib.rs"))
                .expect_err("without a driver nothing can run");
            match err {
                UnsupportedReason::Feature(what) => {
                    assert!(what.contains("driver binary not found"), "reason: {what}")
                }
                other => panic!("expected Feature reason, got {other}"),
            }
            return;
        }
        for name in ["silent_arith.rs", "silent_loop.rs", "silent_fib.rs", "silent_panic.rs"] {
            let report = evaluate_case(&case(name), &runner);
            assert_eq!(
                report.status,
                CaseStatus::Passed,
                "{name} should pass on the JIT path: {report:?}"
            );
        }
        let report = evaluate_corpus(&crate::default_corpus_dir(), &runner)
            .expect("corpus evaluation runs");
        for c in &report.cases {
            eprintln!("jit corpus: {} -> {}", c.case.name, c.status);
            assert!(
                !matches!(c.status, CaseStatus::Failed { .. }),
                "{} diverged on the JIT path: {:?}",
                c.case.name,
                c.status
            );
        }
        eprintln!("jit corpus summary: {}", report.summary);
    }
}
