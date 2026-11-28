//! Unsafe binders (Issue #510): a body that mentions an
//! unsafe binder — in a type, or through the wrap/unwrap
//! constructs themselves — converts with the binder erased
//! to the type it wraps, because the binder is transparent
//! at the value level: wrapping and unwrapping are identity
//! operations on the wrapped value. Before the type erasure,
//! the kind question for a binder type panicked inside
//! rustc_public (the `unimplemented!()` at
//! convert/stable/ty.rs:471) and the whole body was skipped
//! with the named unsafe-binder reason, so the compile-check
//! records whose only binder contact is a type
//! (tests/ui/unsafe-binders/simple.rs, unused-lifetimes.rs,
//! unused-lifetimes-2.rs and
//! debuginfo-escaping-bound-vars-issue-159576.rs) failed on
//! conversion completeness. The wrap/unwrap constructs
//! stayed skipped longer: their Stable MIR conversions are
//! `unimplemented!("FIXME(unsafe_binders):")`, so the fetch
//! itself panicked; they now convert through the internal
//! MIR materialization that erases them to the identity
//! (see `convert::unsafe_binder_erased_body`), which covers
//! the remaining records of this family:
//! tests/ui/unsafe-binders/cat-projection.rs (the
//! `unwrap_binder!` projection) and
//! tests/mir-opt/gvn_on_unsafe_binder.rs (the
//! `wrap_binder!` rvalue). The fixtures pin all of these
//! shapes converting completely.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits.
static SCRATCH_DIRS: std::sync::Mutex<Vec<std::path::PathBuf>> = std::sync::Mutex::new(Vec::new());

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
    let (code, stdout, stderr) = convert("binder_return", BINDER_RETURN_RS, &["--crate-type=lib"]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_target_crate_complete(&stdout);
}

/// A body that actually wraps a value with `wrap_binder!`
/// (the shape of tests/mir-opt/gvn_on_unsafe_binder.rs's
/// `propagate`): the wrap rvalue's Stable MIR conversion is
/// unimplemented, so the body converts through the internal
/// materialization that erases the wrap to a plain use.
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
fn wrap_binder_body_converts() {
    let (code, stdout, stderr) = convert("wrap_binder", WRAP_BINDER_RS, &["--crate-type=lib"]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_target_crate_complete(&stdout);
}

/// Wrapping a reference (the other body of
/// tests/mir-opt/gvn_on_unsafe_binder.rs): the wrapped
/// operand is a borrow, and the binder type is over the
/// reference.
const WRAP_BINDER_REF_RS: &str = "\
#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::unsafe_binder::wrap_binder;

fn test() {
    unsafe {
        let x = 1;
        let binder: unsafe<'a> &'a i32 = wrap_binder!(&x);
        let _ = binder;
    }
}

fn main() {}
";

#[test]
fn wrap_binder_reference_body_converts() {
    let (code, stdout, stderr) = convert("wrap_binder_ref", WRAP_BINDER_REF_RS, &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_target_crate_complete(&stdout);
}

/// A body that unwraps a binder argument with
/// `unwrap_binder!`, by value and by reference, inside
/// closures (the shape of
/// tests/ui/unsafe-binders/cat-projection.rs): the unwrap
/// is a place projection whose Stable MIR conversion is
/// unimplemented, so these bodies convert through the
/// internal materialization that drops the projection.
const UNWRAP_BINDER_RS: &str = "\
#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::unsafe_binder::unwrap_binder;

#[derive(Copy, Clone)]
pub struct S([usize; 8]);

pub fn by_value(x: unsafe<'a> S) -> usize {
    unsafe { (|| unwrap_binder!(x).0[0])() }
}

pub fn by_ref(x: unsafe<'a> &'a S) -> usize {
    unsafe { (|| unwrap_binder!(x).0[0])() }
}

fn main() {}
";

#[test]
fn unwrap_binder_projection_bodies_convert() {
    let (code, stdout, stderr) = convert("unwrap_binder", UNWRAP_BINDER_RS, &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert_target_crate_complete(&stdout);
}
