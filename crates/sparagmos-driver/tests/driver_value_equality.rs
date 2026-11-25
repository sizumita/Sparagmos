//! Run-mode regression tests for value equality and the
//! return boundary where a value's representation in the
//! model legitimately differs from the declared type of the
//! place it is compared against or stored into (the
//! "type mismatch in interp conversion" family, Issue #513).
//!
//! * A `dyn Trait` value behind an `Rc` / `Arc` handle: the
//!   handle aggregate is the pointer chain plus metadata, and
//!   the pointee the equality must compare lives in the shared
//!   allocation's `RcInner` / `ArcInner` aggregate. Comparing
//!   the handle itself against the sized side cannot decide.
//! * An `OsString` against a `str`: the string's bytes sit
//!   behind two wrapper newtypes (`OsString` is a `Buf`
//!   around the `Vec<u8>`), so the element-form string
//!   extraction must descend through the nested single-field
//!   wrappers to the element list.
//! * A `Vec<u8>` in its element form (the shape a place
//!   type-ascription assignment leaves) against a byte-array
//!   literal: the vector value's fields ARE the elements, so
//!   the field list itself is the byte sequence.
//! * A fieldless (C-like) enum returned through a transmuted
//!   function pointer into an integer-typed destination: the
//!   enum is represented as its discriminant integer, and the
//!   value must cross the return boundary as that integer.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits.
/// Tests spawn the driver per case and used to leave every
/// fixture directory behind (#537); the registry keeps the
/// per-test `temp_dir` shape and sweeps at exit instead.
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

fn fixture(dir: &Path, name: &str, src: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    path
}

fn run_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-valeq-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn assert_runs(tag: &str, src: &str, marker: &str) {
    let dir = temp_dir(tag);
    let file = fixture(&dir, "main.rs", src);
    let out = run_driver(&file);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains(marker),
        "driver run failed for {tag}: status {:?}, stdout: {stdout}, stderr: {stderr}",
        out.status
    );
}

#[test]
fn rc_arc_dyn_pointee_equality() {
    assert_runs(
        "rc-arc",
        r#"
use std::rc::Rc;
use std::sync::Arc;
trait Foo: PartialEq<i32> + std::fmt::Debug {
    fn a(&self) -> i32 { 10 }
}
impl Foo for i32 {
    fn a(&self) -> i32 { 100 }
}
fn main() {
    let b: Rc<dyn Foo> = Rc::new(1i32);
    assert_eq!(*b, 1);
    assert_ne!(*b, 2);
    assert_eq!(b.a(), 100);
    let c: Arc<dyn Foo> = Arc::new(7i32);
    assert_eq!(*c, 7);
    assert_ne!(*c, 8);
    println!("rc-arc-dyn-eq ok");
}
"#,
        "rc-arc-dyn-eq ok",
    );
}

#[test]
fn osstring_str_equality() {
    assert_runs(
        "osstring",
        r#"
use std::ffi::OsString;
fn main() {
    let os_str = OsString::from("Hello Rust!");
    assert_eq!(os_str, "Hello Rust!");
    assert_eq!("Hello Rust!", os_str);
    assert_ne!(os_str, "Hello Moon!");
    println!("osstring-eq ok");
}
"#,
        "osstring-eq ok",
    );
}

#[test]
fn element_form_vec_against_byte_array() {
    assert_runs(
        "ascribe",
        r#"
#![feature(type_ascription)]
fn main() {
    let mut v = Vec::new();
    type_ascribe!(v, Vec<u8>) = vec![1, 2, 3];
    assert_eq!(v, [1u8, 2, 3]);
    assert_ne!(v, [1u8, 2, 4]);
    println!("vec-bytes-eq ok");
}
"#,
        "vec-bytes-eq ok",
    );
}

#[test]
fn fieldless_enum_return_as_int() {
    assert_runs(
        "clike",
        r#"
#[repr(u32)]
enum Foo {
    A = 0,
    B = 23,
}
#[inline(never)]
extern "C" fn foo(_x: usize) -> Foo { Foo::B }
#[repr(i32)]
enum Bar {
    N = -5,
    P = 3,
}
#[inline(never)]
extern "C" fn bar(_x: usize) -> Bar { Bar::N }
fn main() {
    unsafe {
        let f: extern "C" fn(usize) -> u32 =
            std::mem::transmute(foo as extern "C" fn(usize) -> Foo);
        assert_eq!(f(0xDEADBEEF), 23);
        let g: extern "C" fn(usize) -> i32 =
            std::mem::transmute(bar as extern "C" fn(usize) -> Bar);
        assert_eq!(g(0), -5);
    }
    println!("clike-return ok");
}
"#,
        "clike-return ok",
    );
}
