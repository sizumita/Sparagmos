//! The stage-3 cargo shim (issue #359): tool interception and
//! the process host configured together, and the driver's
//! `rustc-shim` mode (convert one unit, record its verdict on
//! a side channel, delegate to the real compiler).

use super::{Command, Path, fixture, run_driver_extra, temp_dir};

const COEXIST_GUEST_SRC: &str = r#"
use std::process::Command;

fn main() {
    // The harness passes the tool program's path as the first
    // guest argument.
    let tool = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    // The named tool program is answered by the conversion
    // machinery, as in any tool-interception run.
    std::fs::write("t.rs", "pub fn f() -> u32 { 7 }\n").unwrap();
    let out = Command::new(&tool)
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("t.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "tool compile failed");

    // A real program the run neither named nor produced is
    // the process host's to serve: it really runs, and its
    // output and exit status are the host process's own.
    let out = Command::new("/bin/echo")
        .arg("host-served")
        .output()
        .unwrap();
    assert!(out.status.success(), "echo failed");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "host-served"
    );
    let st = Command::new("/bin/true").status().unwrap();
    assert!(st.success(), "true failed");

    println!("coexist ok");
}
"#;

#[test]
fn run_mode_tool_interception_and_process_host_coexist() {
    let dir = temp_dir("tool-host-coexist");
    let src = fixture(&dir, "coexist_guest.rs", COEXIST_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(
        &src,
        &[
            "--tool-program",
            driver,
            "--guest-process-host",
            "--guest-arg",
            driver,
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert!(stdout.contains("coexist ok"), "stdout={stdout}");
}

fn shim_command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
}

fn pinned_rustc() -> std::path::PathBuf {
    Path::new(env!("SPARAGMOS_BUILD_SYSROOT"))
        .join("bin")
        .join("rustc")
}

#[test]
fn rustc_shim_converts_and_delegates() {
    let dir = temp_dir("cargo-shim-unit");
    let src = fixture(&dir, "lib.rs", "pub fn f() -> u32 { 7 }\n");
    let verdicts = dir.join("verdicts.txt");
    let rlib = dir.join("liblib.rlib");
    let out = shim_command()
        .arg("rustc-shim")
        .arg(pinned_rustc())
        .arg("--edition=2021")
        .arg("--crate-name")
        .arg("lib")
        .arg("--crate-type=rlib")
        .arg("-o")
        .arg(&rlib)
        .arg(&src)
        .env("SPARAGMOS_CARGO_SHIM_VERDICTS", &verdicts)
        .current_dir(&dir)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr={stderr}");
    // The delegated artifact is a real rlib, not the empty
    // placeholder the conversion side fabricates.
    assert!(
        std::fs::metadata(&rlib).unwrap().len() > 0,
        "delegated rlib is empty"
    );
    let text = std::fs::read_to_string(&verdicts).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1, "verdicts: {text}");
    assert!(
        lines[0].starts_with("converted\tlib\t"),
        "verdict line: {}",
        lines[0]
    );
}

#[test]
fn rustc_shim_records_failed_conversion_and_delegates_the_failure() {
    let dir = temp_dir("cargo-shim-fail");
    let src = fixture(
        &dir,
        "bad.rs",
        "fn main() { let _x: i32 = \"s\"; }\n",
    );
    let verdicts = dir.join("verdicts.txt");
    let out = shim_command()
        .arg("rustc-shim")
        .arg(pinned_rustc())
        .arg("--edition=2021")
        .arg(&src)
        .env("SPARAGMOS_CARGO_SHIM_VERDICTS", &verdicts)
        .current_dir(&dir)
        .output()
        .unwrap();
    // The exit code is the real compiler's, not the
    // conversion's verdict.
    assert_eq!(out.status.code(), Some(1));
    let text = std::fs::read_to_string(&verdicts).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1, "verdicts: {text}");
    assert!(
        lines[0].starts_with("failed\t"),
        "verdict line: {}",
        lines[0]
    );
}

#[test]
fn rustc_shim_delegates_informational_invocations_without_a_verdict() {
    let dir = temp_dir("cargo-shim-info");
    let verdicts = dir.join("verdicts.txt");
    let out = shim_command()
        .arg("rustc-shim")
        .arg(pinned_rustc())
        .arg("-vV")
        .env("SPARAGMOS_CARGO_SHIM_VERDICTS", &verdicts)
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("rustc"), "stdout={stdout}");
    // A probe is not a compilation unit: no verdict line.
    assert!(!verdicts.exists(), "a verdict was recorded");
}
