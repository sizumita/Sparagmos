//! Crate-type handling, end to end through the driver binary.
//!
//! Convert mode mirrors compiletest, which passes no `--crate-type`
//! for a test file: a forwarded `--crate-type` or a source-level
//! `#![crate_type]` declaration governs when present, and rustc's
//! default (bin) applies otherwise -- entry-point checks and
//! dead-code reachability roots included. Run mode keeps its
//! `--crate-type=lib` pin. Each test writes a tiny crate to a temp
//! dir and spawns the driver binary (built with the pinned 1.100.0
//! toolchain, see the crate README).

use std::process::Command;

fn run_driver(mode: Option<&str>, source: &str, extra: &[&str]) -> (i32, String, String) {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-crate-type-{}-{}",
        std::process::id(),
        source.len() * 131 + extra.len()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("case.rs");
    std::fs::write(&src, source).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    if let Some(mode) = mode {
        cmd.arg(mode);
    }
    cmd.arg(&src).arg("--edition").arg("2021");
    for a in extra {
        cmd.arg(a);
    }
    let out = cmd.output().expect("failed to run sparagmos-driver");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn convert_mode_main_is_a_dead_code_root() {
    // Under the old `--crate-type=lib` pin this was rejected with
    // "function `main` is never used": in a library crate `main` is
    // an ordinary private function. As a bin (compiletest's shape)
    // it is the entry point and a reachability root.
    let (code, stdout, stderr) = run_driver(
        None,
        "#![deny(dead_code)]\nfn main() {}\n",
        &[],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn convert_mode_without_main_is_a_compilation_failure() {
    // No `main`, no `#![no_main]`, no crate-type declaration: rustc's
    // bin default applies and the entry-point check fires (E0601),
    // as it does under compiletest.
    let (code, _stdout, stderr) = run_driver(None, "pub fn f() {}\n", &[]);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("compilation failed"),
        "stderr={stderr}"
    );
    assert!(stderr.contains("E0601"), "stderr={stderr}");
}

#[test]
fn convert_mode_declared_crate_type_governs() {
    // A source-level `#![crate_type]` declaration decides, as under
    // compiletest (which passes no flag of its own).
    let (code, stdout, stderr) = run_driver(
        None,
        "#![crate_type = \"rlib\"]\npub fn f() {}\n",
        &[],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn convert_mode_forwarded_crate_type_governs() {
    // A caller-forwarded `--crate-type` (a test's own compile-flags,
    // forwarded by the subset runner) decides.
    let (code, stdout, stderr) = run_driver(
        None,
        "pub fn f() {}\n",
        &["--rustc-arg", "--crate-type=rlib"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn run_mode_keeps_the_lib_pin() {
    // Run mode is deliberately unchanged: it still pins
    // `--crate-type=lib`, under which this program's `main` is dead
    // code and the front end rejects it (run mode reports a failed
    // compilation as exit 2). Execution semantics stay scoped to the
    // crate shape the interpreter was built against; the
    // compiletest-parity treatment applies to convert mode only.
    let (code, _stdout, stderr) = run_driver(
        Some("run"),
        "#![deny(dead_code)]\nfn main() { println!(\"hi\"); }\n",
        &[],
    );
    assert_eq!(code, 2, "stderr={stderr}");
    assert!(
        stderr.contains("function `main` is never used"),
        "stderr={stderr}"
    );
}
