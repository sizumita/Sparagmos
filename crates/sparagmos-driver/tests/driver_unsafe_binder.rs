//! Unsafe binder types (Issue #510): a body that merely
//! mentions an unsafe binder type — a local's type, an
//! argument or return type — converts with the binder erased
//! to the type it wraps, because the binder is transparent
//! at the value level and only the wrap/unwrap constructs
//! themselves need the Stable MIR conversions that stay
//! unimplemented. Before this, the kind question for a binder
//! type panicked inside rustc_public (the `unimplemented!()`
//! at convert/stable/ty.rs:471) and the whole body was
//! skipped with the named unsafe-binder reason, so the
//! compile-check records whose only binder contact is a type
//! (tests/ui/unsafe-binders/simple.rs, unused-lifetimes.rs,
//! unused-lifetimes-2.rs and
//! debuginfo-escaping-bound-vars-issue-159576.rs) failed on
//! conversion completeness. The fixtures pin both sides of
//! the boundary: type-only contact converts completely, and
//! a body that actually wraps a value with `wrap_binder!`
//! still skips with the named reason instead of crashing.

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
        "sparagmos-driver-unsafe-binder-{}-{name}",
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
fn convert(name: &str, src: &str, rustc_args: &[&str]) -> (Option<i32>, String, String) {
    let fixture = write_fixture(name, src);
    let mut cmd = Command::new(driver());
    cmd.arg(&fixture).arg("--edition").arg("2021");
    cmd.env("RUSTC_BOOTSTRAP", "1");
    cmd.env_remove("SPARAGMOS_DUMP");
    for a in rustc_args {
        cmd.arg("--rustc-arg").arg(a);
    }
    let out = cmd.output().expect("spawn driver");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The target crate converted completely: every local body
/// converted, none skipped.
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
        !stdout.contains("body mentions an unsafe binder"),
        "a body still skips on an unsafe binder type:\n{stdout}"
    );
}

/// A local declared with an unsafe binder type and never
/// used (the shape of tests/ui/unsafe-binders/simple.rs):
/// the binder is the local's whole contact with the body.
const BINDER_LOCAL_RS: &str = "\
#![feature(unsafe_binders)]

fn main() {
    let x: unsafe<'a> &'a ();
}
";

#[test]
fn binder_typed_local_converts() {
    let (code, stdout, stderr) = convert("binder_local", BINDER_LOCAL_RS, &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_target_crate_complete(&stdout);
}

/// An argument of unsafe binder type whose body never wraps
/// or unwraps it (the shape of
/// tests/ui/unsafe-binders/unused-lifetimes.rs).
const BINDER_ARG_RS: &str = "\
#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

#[derive(Copy, Clone)]
pub struct S([usize; 8]);

pub fn by_value(_x: unsafe<'a> S) -> usize {
    0
}

fn main() {}
";

#[test]
fn binder_typed_argument_converts() {
    let (code, stdout, stderr) = convert("binder_arg", BINDER_ARG_RS, &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_target_crate_complete(&stdout);
}

/// A multi-variable binder over a reference argument (the
/// shape of tests/ui/unsafe-binders/unused-lifetimes-2.rs):
/// erasure unwraps the binder and keeps the reference.
const BINDER_MULTI_ARG_RS: &str = "\
#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

#[derive(Copy, Clone)]
pub struct S([usize; 8]);

pub fn by_value(_x: unsafe<'a, 'b> &'a S) -> usize {
    0
}

fn main() {}
";

#[test]
fn multi_variable_binder_argument_converts() {
    let (code, stdout, stderr) = convert("binder_multi_arg", BINDER_MULTI_ARG_RS, &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_target_crate_complete(&stdout);
}

/// An unsafe binder only in the return type, over a trait
/// object type mentioning the bound lifetimes (the shape of
/// tests/ui/unsafe-binders/debuginfo-escaping-bound-vars-issue-159576.rs).
const BINDER_RETURN_RS: &str = "\
#![feature(unsafe_binders)]
#![allow(incomplete_features)]

pub fn woof() -> unsafe<'a, 'b> &'b Box<dyn Fn(Box<dyn Fn() -> &'a isize>)> {
    todo!()
}
";

#[test]
fn binder_typed_return_converts() {
    let (code, stdout, stderr) =
        convert("binder_return", BINDER_RETURN_RS, &["--crate-type=lib"]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_target_crate_complete(&stdout);
}

/// The other side of the boundary: a body that actually
/// wraps a value with `wrap_binder!` needs the wrap
/// conversion the bridge leaves unimplemented, so it still
/// skips — with the named reason, never a crash (the shape
/// of tests/mir-opt/gvn_on_unsafe_binder.rs).
const WRAP_BINDER_RS: &str = "\
#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::unsafe_binder::wrap_binder;

pub fn propagate() -> unsafe<'a> i32 {
    unsafe {
        let x = 1;
        let binder: unsafe<'a> i32 = wrap_binder!(x);
        binder
    }
}
";

#[test]
fn wrap_binder_body_still_skips_with_named_reason() {
    let (code, stdout, stderr) =
        convert("wrap_binder", WRAP_BINDER_RS, &["--crate-type=lib"]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert!(
        stdout.contains("body mentions an unsafe binder"),
        "expected the named unsafe-binder skip reason, got:\n{stdout}"
    );
}
