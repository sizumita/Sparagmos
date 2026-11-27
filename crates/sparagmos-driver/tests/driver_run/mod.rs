//! Run-mode smoke tests: build small fixture programs in a temp dir,
//! execute them with `sparagmos-driver run`, and check the exit-code
//! contract (0 = normal termination, 101 = guest panic, 2 = could
//! not convert/execute, with the reason on stderr). The test only
//! spawns the binary; the driver itself must have been built with
//! the pinned 1.100.0 toolchain (see the crate README).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
mod alloc_error;
mod asm_intrinsics;
mod aux_crates;
mod cargo_shim;
mod closures_coroutines;
mod collections;
mod constants;
mod debug_mode;
mod drop_box;
mod dst_size_of_val;
mod dyn_vtable;
mod env_values;
mod ffi;
mod fs_paths;
mod generics;
mod io;
mod libtest;
mod mono;
mod net;
mod numerics;
mod panic_unwind;
mod pointers;
mod process;
mod run_basics;
mod rustdoc_path;
mod stack_overflow;
mod string_mutators;
mod strings;
mod summary_mode;
mod sync;
mod tool_interception;
mod transmute;
mod vec_slice;

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
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("sparagmos-driver-run-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

const TAIL_CALL_SRC: &str = r#"
#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]
fn countdown(n: u32) -> u32 {
    if n == 0 { 0 } else { become countdown(n - 1) }
}
fn main() { let _ = countdown(3); }
"#;

fn run_driver_with_aux(src: &Path, aux: &[(&str, &Path)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg("run").arg(src);
    for (name, path) in aux {
        cmd.arg("--aux").arg(format!("{}={}", name, path.display()));
    }
    cmd.output().expect("failed to run sparagmos-driver")
}

fn run_driver_with_aux_specs(
    src: &Path,
    edition: Option<&str>,
    specs: &[String],
) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg("run").arg(src);
    if let Some(ed) = edition {
        cmd.arg("--edition").arg(ed);
    }
    for spec in specs {
        cmd.arg("--aux").arg(spec);
    }
    cmd.output().expect("failed to run sparagmos-driver")
}

fn aux_spec(name: &str, path: &Path, attrs: &str) -> String {
    format!("{}={}{}", name, path.display(), attrs)
}

// ----- closures (Lane AU) ----------------------------------------------------

fn summarize_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg(src)
        .output()
        .expect("failed to run sparagmos-driver")
}

const CLOSURE_GENERIC_CONTEXT_SRC: &str = r#"
fn wrap<T: Copy>(v: T) -> T {
    let id = |x: T| x;
    id(v)
}
fn main() { println!("{}", wrap::<i32>(7)); }
"#;

const DYN_DEFAULT_SRC: &str = r#"
trait Foo {
    fn a(&self) -> i32 { 100 }
    fn b(&self) -> i32 { self.a() + 1 }
}
impl Foo for i32 {}
trait Greeter {
    fn name(&self) -> i32;
    fn greet(&self) -> i32 { self.name() + 1 }
}
struct S;
impl Greeter for S {
    fn name(&self) -> i32 { 41 }
}
fn main() {
    let f: &dyn Foo = &1i32;
    assert_eq!(f.a(), 100);
    assert_eq!(f.b(), 101);
    let g: Box<dyn Foo> = Box::new(2i32);
    assert_eq!(g.a(), 100);
    assert_eq!(g.b(), 101);
    let h: &dyn Greeter = &S;
    assert_eq!(h.greet(), 42);
    let h2: Box<dyn Greeter> = Box::new(S);
    assert_eq!(h2.greet(), 42);
    println!("dyn-default ok");
}
"#;

// ----- anonymous constants (Lane BW) -----

const ANON_CONST_GENERIC_SRC: &str = r#"
fn f<T>(x: T) -> usize {
    let a = [0u32; 1 + 2];
    a.len() + std::mem::size_of::<T>()
}
fn main() {
    println!("{}", f::<u8>(0));
}
"#;

// ----- external-demand instances (Lane BX) -----

const EXTERNAL_DEMAND_SRC: &str = r#"
struct RepeatN<T> { v: T, n: usize }
impl<T: Copy> Iterator for RepeatN<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.n == 0 { None } else { self.n -= 1; Some(self.v) }
    }
}

struct Total(i32);
impl std::iter::Sum<i32> for Total {
    fn sum<I: Iterator<Item = i32>>(mut iter: I) -> Self {
        let mut acc = 0i32;
        while let Some(x) = iter.next() {
            acc += x;
        }
        Total(acc)
    }
}

fn unused<T>(x: T) -> T { x }

fn main() {
    let t: Total = RepeatN { v: 7i32, n: 3 }.sum();
    println!("{}", t.0);
}
"#;

// ----- generic constructors (Lane CF) -----
//
// A generic tuple-struct / tuple-variant constructor has no
// monomorphic instance as an item. Its direct uses are aggregate
// rvalues in the callers, so nothing is left unconverted; used as
// a value, the instance its FnDef constant demands is converted
// (the synthetic one-block body that aggregates the fields) and
// the call through the value runs it.

const CTOR_VALUE_SRC: &str = r#"
struct Pair<T>(T, i32);
fn main() {
    let f = Pair;
    let p = f(3, 4);
    println!("{}", p.0 + p.1);
}
"#;

fn run_ok(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver(&src_path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "catch_unwind case {tag} must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "catch_unwind case {tag} stdout mismatch; stderr={stderr}"
    );
}

fn run_driver_extra(src: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .args(extra)
        .output()
        .expect("failed to run sparagmos-driver")
}

// ---------------------------------------------------------------------
// Async execution base (Lane JF, Issue #167): the modelled waker
// machinery - RawWaker vtable function-pointer calls, `Arc::from_raw`
// reconstruction, waker drop through the vtable, and `Future::poll`
// dispatch, static and through a `dyn` receiver.

fn run_ok_edition(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver_extra(&src_path, &["--edition", "2021"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "case {tag} must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "case {tag} stdout mismatch; stderr={stderr}"
    );
}

const DYN_CONST_DISPATCH_SRC: &str = r#"
trait Sub { fn x(&self) -> u32; }
struct NoSub;
impl Sub for NoSub { fn x(&self) -> u32 { 7 } }
static NO_SUB: NoSub = NoSub;
enum Kind { Global(&'static (dyn Sub + Send + Sync)), Other }
struct Dispatch { subscriber: Kind }
static mut GLOBAL_DISPATCH: Dispatch = Dispatch { subscriber: Kind::Global(&NO_SUB) };
fn get_global() -> &'static Dispatch { unsafe { &*std::ptr::addr_of!(GLOBAL_DISPATCH) } }
fn main() {
    match &get_global().subscriber {
        Kind::Global(s) => println!("x={}", s.x()),
        Kind::Other => println!("other"),
    }
}
"#;
