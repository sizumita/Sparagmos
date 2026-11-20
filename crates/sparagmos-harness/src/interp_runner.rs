//! The harness's Sparagmos runner on the project's main path.
//!
//! Main path: `.rs` --(Stable MIR driver)--> `sparagmos-mir` bodies
//! --(sparagmos-interp)--> execution result. The driver performs the
//! whole path in its own process (`sparagmos-driver run <file.rs>`:
//! it converts with Stable MIR and executes `main` in the
//! interpreter), so this runner is a thin subprocess adapter over the
//! driver's exit-code contract:
//!
//! * exit 0 → the guest program terminated normally;
//! * exit 101 → the guest program panicked (message on stderr);
//! * exit 2 → conversion or execution is unsupported (concrete reason
//!   on stderr) — mapped to [`UnsupportedReason::Feature`], never to a
//!   faked result;
//! * any other exit code → the driver itself failed; also reported as
//!   unsupported-with-reason, with the driver's stderr included.
//!
//! The driver binary is *not* built by the harness (it needs the
//! pinned 1.100.0 toolchain and its own workspace). Resolution order:
//! an explicitly configured path ([`InterpRunner::with_driver_bin`]),
//! the `SPARAGMOS_DRIVER_BIN` environment variable, then the default
//! location inside a checkout,
//! `crates/sparagmos-driver/target/debug/sparagmos-driver`. If no
//! binary is found, every case is honestly `Unsupported` with a
//! "driver binary not found" reason.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{CorpusCase, RunResult, SparagmosRunner, UnsupportedReason};

/// Environment variable naming an explicit `sparagmos-driver` binary.
pub const DRIVER_BIN_ENV: &str = "SPARAGMOS_DRIVER_BIN";

/// Corpus runner that executes cases through the Stable MIR driver
/// (see the module docs for the exit-code contract).
#[derive(Debug, Clone, Default)]
pub struct InterpRunner {
    driver_bin: Option<PathBuf>,
}

impl InterpRunner {
    /// Runner resolving the driver binary from `SPARAGMOS_DRIVER_BIN`
    /// or the default in-checkout location.
    pub fn new() -> Self {
        Self::default()
    }

    /// Runner using an explicit driver binary path (used by tests and
    /// by callers with a non-standard checkout layout).
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

/// Default driver location inside a checkout:
/// `<repo>/crates/sparagmos-driver/target/debug/sparagmos-driver`,
/// resolved relative to this crate (`crates/sparagmos-harness`).
pub fn default_driver_bin() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("crates")
        .join("sparagmos-driver")
        .join("target")
        .join("debug")
        .join("sparagmos-driver")
}

/// Extract the driver's own reason lines from its stderr (rustc
/// diagnostics and warnings share the stream; the driver prefixes its
/// run-mode reasons with `sparagmos-driver run:`).
fn driver_reason(stderr: &str) -> String {
    let lines: Vec<&str> = stderr
        .lines()
        .map(|l| l.trim())
        .filter(|l| l.contains("sparagmos-driver run:"))
        .collect();
    if lines.is_empty() {
        format!(
            "driver reported unsupported (exit 2) without a reason line: {}",
            stderr.trim()
        )
    } else {
        lines.join("; ")
    }
}

impl SparagmosRunner for InterpRunner {
    fn run(&self, case: &CorpusCase) -> Result<RunResult, UnsupportedReason> {
        let bin = self.resolve_driver_bin()?;
        let output = Command::new(&bin)
            .arg("run")
            .arg(&case.path)
            .output()
            .map_err(|e| {
                UnsupportedReason::Feature(format!("failed to spawn driver {}: {e}", bin.display()))
            })?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        match output.status.code() {
            Some(0) => Ok(RunResult {
                stdout,
                stderr,
                exit_code: Some(0),
            }),
            Some(101) => Ok(RunResult {
                stdout,
                stderr,
                exit_code: Some(101),
            }),
            Some(2) => Err(UnsupportedReason::Feature(driver_reason(&stderr))),
            other => Err(UnsupportedReason::Feature(format!(
                "driver failed (exit {other:?}): {}",
                stderr.trim()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{evaluate_case, CaseStatus};
    use std::path::PathBuf;

    fn case(name: &str) -> CorpusCase {
        CorpusCase::new(crate::default_corpus_dir().join(name))
    }

    #[test]
    fn missing_driver_binary_is_unsupported_with_reason() {
        let runner = InterpRunner::with_driver_bin("/nonexistent/sparagmos-driver");
        let err = runner
            .run(&case("arith_int.rs"))
            .expect_err("no binary, no execution");
        match err {
            UnsupportedReason::Feature(what) => {
                assert!(what.contains("driver binary not found"), "reason: {what}");
                assert!(what.contains("sparagmos-driver"), "reason: {what}");
            }
            other => panic!("expected Feature reason, got {other}"),
        }
    }

    /// Path to a fake `sparagmos-driver` fixture script (under
    /// `tests/fixtures/`) with a fixed stdout/stderr/exit behaviour, so
    /// the exit-code mapping can be tested without the real driver.
    ///
    /// The scripts are static fixtures rather than files written by
    /// the test process: a just-written script cannot always be
    /// executed immediately. While this process has the script open
    /// for writing, a child forked by a parallel test inherits the
    /// write descriptor, and the kernel refuses the `execve` with
    /// ETXTBSY ("Text file busy") until that child execs or exits.
    /// A fixture checked out with the crate is never opened for
    /// writing here, so the race cannot occur.
    #[cfg(unix)]
    fn fake_driver(name: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name);
        // The executable bit is recorded in git; re-assert it in case
        // the checkout lost it. chmod does not open the file, so this
        // cannot reintroduce the write/exec race described above.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn exit_zero_and_101_become_run_results() {
        let ok = InterpRunner::with_driver_bin(fake_driver("fake-driver-exit-0.sh"));
        let res = ok.run(&case("arith_int.rs")).expect("exit 0 is a result");
        assert_eq!(res.exit_code, Some(0));

        let panicky = InterpRunner::with_driver_bin(fake_driver("fake-driver-exit-101.sh"));
        let res = panicky
            .run(&case("arith_int.rs"))
            .expect("exit 101 is a result");
        assert_eq!(res.exit_code, Some(101));
        assert!(res.stderr.contains("boom"), "stderr: {:?}", res.stderr);
    }

    #[cfg(unix)]
    #[test]
    fn exit_two_is_unsupported_with_the_drivers_reason() {
        let runner = InterpRunner::with_driver_bin(fake_driver("fake-driver-exit-2.sh"));
        let err = runner
            .run(&case("arith_int.rs"))
            .expect_err("exit 2 is unsupported");
        match err {
            UnsupportedReason::Feature(what) => {
                assert!(what.contains("some variant"), "reason: {what}")
            }
            other => panic!("expected Feature reason, got {other}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn other_exit_codes_are_unsupported_with_stderr() {
        let runner = InterpRunner::with_driver_bin(fake_driver("fake-driver-exit-1.sh"));
        let err = runner
            .run(&case("arith_int.rs"))
            .expect_err("exit 1 is unsupported");
        match err {
            UnsupportedReason::Feature(what) => {
                assert!(what.contains("driver failed"), "reason: {what}");
                assert!(what.contains("internal explosion"), "reason: {what}");
            }
            other => panic!("expected Feature reason, got {other}"),
        }
    }

    /// End-to-end over the real driver, when one is resolvable. If no
    /// driver binary exists in this environment, the runner must say
    /// so honestly instead of passing anything.
    #[test]
    fn real_driver_measures_silent_cases() {
        let runner = InterpRunner::new();
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
        // A guest panic in both worlds: native and Sparagmos agree on
        // exit 101 and empty stdout.
        let panic_case = evaluate_case(&case("silent_panic.rs"), &runner);
        assert_eq!(
            panic_case.status,
            CaseStatus::Passed,
            "silent_panic should pass end-to-end: {panic_case:?}"
        );
        // println! output is modelled: hello passes end-to-end with
        // stdout and exit code matching the native run.
        let hello = evaluate_case(&case("hello.rs"), &runner);
        assert_eq!(
            hello.status,
            CaseStatus::Passed,
            "hello should pass end-to-end: {hello:?}"
        );
    }
}
