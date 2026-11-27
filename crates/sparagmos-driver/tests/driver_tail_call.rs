//! Tail calls (Issue #384): a body using `become` carries a
//! `TerminatorKind::TailCall` terminator, whose internal-to-
//! stable conversion rustc_public 1.100.0 leaves
//! unimplemented, so the Stable MIR fetch itself panicked and
//! the whole body was skipped with the named tail-call reason.
//! The driver now materializes such a body through the
//! compiler's internal MIR, rewriting the tail call as the
//! call into the return place plus a return that its
//! documented semantics make it equivalent to (see
//! `convert::tail_call_erased_body`). The fixtures pin the
//! conversion side (a tail-call body converts completely, as a
//! local item and as a demanded generic instance) and the
//! execution side (the rewritten body computes the same
//! result the native tail call does).

use std::path::{Path, PathBuf};
use std::process::Command;

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits.
static SCRATCH_DIRS: std::sync::Mutex<Vec<std::path::PathBuf>> =
    std::sync::Mutex::new(Vec::new());

fn register_scratch_dir(dir: &std::path::Path) {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| unsafe {
        let _ = libc::atexit(remove_scratch_dirs);
    });
    if let Ok(mut dirs) = SCRATCH_DIRS.lock() {
        if !dirs.iter().any(|d| d == dir) {
            dirs.push(dir.to_path_buf());
        }
    }
}

extern "C" fn remove_scratch_dirs() {
    if let Ok(dirs) = SCRATCH_DIRS.try_lock() {
        for dir in dirs.iter() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn driver() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sparagmos-driver"))
}

fn write_fixture(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-tail-call-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    register_scratch_dir(&dir);
    let path = dir.join(format!("{name}.rs"));
    std::fs::write(&path, src).expect("write fixture");
    path
}

/// Convert `fixture` (convert mode: no subcommand), returning
/// (exit code, stdout, stderr). Mirrors the subset runner's
/// invocations, which carry RUSTC_BOOTSTRAP=1 for the embedded
/// nightly rustc.
fn convert(name: &str, src: &str) -> (Option<i32>, String, String) {
    let fixture = write_fixture(name, src);
    let out = Command::new(driver())
        .arg(&fixture)
        .arg("--edition")
        .arg("2021")
        .env("RUSTC_BOOTSTRAP", "1")
        .env_remove("SPARAGMOS_DUMP")
        .output()
        .expect("spawn driver");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Run `fixture` (run mode), returning (exit code, stdout,
/// stderr): the guest program's own output is the driver's
/// stdout.
fn run(name: &str, src: &str) -> (Option<i32>, String, String) {
    let fixture = write_fixture(name, src);
    let out = Command::new(driver())
        .arg("run")
        .arg(&fixture)
        .arg("--edition")
        .arg("2021")
        .env("RUSTC_BOOTSTRAP", "1")
        .env_remove("SPARAGMOS_DUMP")
        .output()
        .expect("spawn driver");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The target crate converted completely: every local body
/// converted, none skipped, and no body still reports the
/// tail-call reason.
fn assert_target_crate_complete(stdout: &str) {
    let line = stdout
        .lines()
        .find(|l| l.starts_with("target-crate converted:"))
        .expect("target-crate conversion summary");
    assert!(
        line.ends_with("skipped: 0"),
        "target crate conversion incomplete: {line}\n{stdout}"
    );
    assert!(
        !stdout.contains("body contains a tail call"),
        "a body still skips on a tail call:\n{stdout}"
    );
}

/// A directly tail-recursive function: the `become` is in the
/// local item's own body, fetched by the item loop.
const TAIL_RECURSIVE_RS: &str = "\
#![feature(explicit_tail_calls)]
#![allow(incomplete_features)]

pub fn count(n: u64) -> u64 {
    if n == 0 {
        0
    } else {
        become count(n - 1)
    }
}

fn main() {}
";

#[test]
fn tail_recursive_body_converts() {
    let (code, stdout, stderr) = convert("tail_recursive", TAIL_RECURSIVE_RS);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_target_crate_complete(&stdout);
}

/// A tail call to a *different* function, and a generic
/// tail-recursive function whose monomorphized instance is
/// demanded by `main`: the instance's body is fetched by the
/// demand passes, not the item loop, and its materialization
/// goes through the instantiation branch of the body builder.
const TAIL_GENERIC_RS: &str = "\
#![feature(explicit_tail_calls)]
#![allow(incomplete_features)]

pub fn last<T: Copy>(x: T, n: u32) -> T {
    if n == 0 {
        x
    } else {
        become last(x, n - 1)
    }
}

pub fn finish(_n: u64) -> u64 {
    0
}

pub fn countdown(n: u64) -> u64 {
    if n == 0 {
        become finish(n)
    } else {
        become countdown(n - 1)
    }
}

fn main() {
    let _ = last(1u32, 3);
    let _ = countdown(2);
}
";

#[test]
fn tail_call_to_other_and_generic_instance_convert() {
    let (code, stdout, stderr) = convert("tail_generic", TAIL_GENERIC_RS);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_target_crate_complete(&stdout);
}

/// The execution side: an accumulator-style tail recursion
/// rewritten as call-plus-return computes the native result
/// (sum 1..=10 = 55).
const TAIL_SUM_RS: &str = "\
#![feature(explicit_tail_calls)]
#![allow(incomplete_features)]

pub fn sum(n: u64, acc: u64) -> u64 {
    if n == 0 {
        acc
    } else {
        become sum(n - 1, acc + n)
    }
}

fn main() {
    println!(\"{}\", sum(10, 0));
}
";

#[test]
fn tail_recursive_body_executes_with_native_result() {
    let (code, stdout, stderr) = run("tail_sum", TAIL_SUM_RS);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_eq!(stdout.trim(), "55", "unexpected guest output: {stdout}");
}
