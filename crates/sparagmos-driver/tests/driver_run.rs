//! Run-mode smoke tests: build small fixture programs in a temp dir,
//! execute them with `sparagmos-driver run`, and check the exit-code
//! contract (0 = normal termination, 101 = guest panic, 2 = could
//! not convert/execute, with the reason on stderr). The test only
//! spawns the binary; the driver itself must have been built with
//! the pinned 1.100.0 toolchain (see the crate README).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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
    dir
}

const PANIC_SRC: &str = r#"
fn boom(x: i32) -> i32 { x + x }
fn main() { let _ = boom(2_000_000_000); }
"#;

const PRINTLN_SRC: &str = r#"
fn main() { println!("hi"); }
"#;

const SILENT_SRC: &str = r#"
fn main() {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < 100 { s += i; i += 1; }
    let _ = s;
}
"#;

#[test]
fn run_mode_guest_panic_exits_101() {
    let dir = temp_dir("panic");
    let src = fixture(&dir, "panic.rs", PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "guest panic must exit 101; stderr={stderr}"
    );
    assert!(
        stderr.contains("panicked"),
        "panic message goes to stderr: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn run_mode_println_prints_to_stdout() {
    let dir = temp_dir("println");
    let src = fixture(&dir, "println.rs", PRINTLN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "println! runs through the interpreter's runtime print support; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hi\n",
        "the guest's stdout is the driver's stdout"
    );
}

#[test]
fn run_mode_silent_program_exits_zero() {
    // A plain computation-only main converts and executes to the end.
    // This used to be blocked in the interpreter: real MIR never
    // writes the return place `_0` of a unit-returning function, and
    // `Return` read it as uninitialized (exit 2, reason naming `_0`).
    // The interpreter now reads an unwritten unit `_0` as `()`, so
    // the program terminates normally.
    let dir = temp_dir("silent");
    let src = fixture(&dir, "silent.rs", SILENT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "silent program must terminate normally; stderr={stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

// ----- by-reference constants (Lane R) ---------------------------------------

const PROMOTED_REF_SRC: &str = r#"
fn main() {
    let x = 10;
    let y = x;
    assert_eq!(y, 10);
    let z: u8 = 255;
    assert_eq!(z, 255);
}
"#;

const CONST_SHAPES_SRC: &str = r#"
struct S { a: i32, b: u8 }
const S1: S = S { a: -2, b: 7 };
const ARR: [i32; 3] = [10, 20, 30];
const R: &i32 = &ARR[1];
static A: usize = 42;
static B: usize = A;
const NAMES: [&str; 2] = ["x", "yy"];
const OPT: Option<&i32> = Some(&5);
fn main() {
    println!("{} {} {} {} {} {}", S1.a, S1.b, *R, B, NAMES[1],
             match OPT { Some(r) => *r, None => 0 });
}
"#;

#[test]
fn run_mode_promoted_ref_constant() {
    // `assert_eq!` against a literal builds a promoted `&i32`
    // constant; it must execute, not stop at the constant.
    let dir = temp_dir("promoted");
    let src = fixture(&dir, "promoted.rs", PROMOTED_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "promoted reference constants must run; stderr={stderr}"
    );
}

#[test]
fn run_mode_materialised_constant_shapes() {
    // Struct constants, a reference into the middle of an array
    // constant, statics read by value, a `[&str]` constant and a
    // niche-encoded `Option<&i32>` constant, all printed.
    let dir = temp_dir("const-shapes");
    let src = fixture(&dir, "const_shapes.rs", CONST_SHAPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "materialised constants must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "-2 7 20 42 yy 5\n",
        "decoded constant values must match the source program"
    );
}

const TAIL_CALL_SRC: &str = r#"
#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]
fn countdown(n: u32) -> u32 {
    if n == 0 { 0 } else { become countdown(n - 1) }
}
fn main() { let _ = countdown(3); }
"#;

const ASYNC_CLOSURE_SRC: &str = r#"
fn main() {
    let f = async move |x: i32| x + 1;
    let _ = &f;
}
"#;

const LOG_SYNTAX_SRC: &str = r#"
#![feature(log_syntax)]
fn main() {
    log_syntax!("compile-time marker");
    println!("guest marker");
}
"#;


#[test]
fn run_mode_tail_call_conversion_panic_exits_two() {
    // Converting a `become` (explicit tail call) makes rustc_public
    // panic (`unimplemented!()` in Stable MIR's terminator
    // conversion). The driver must catch that panic at the compiler
    // boundary and report it under the run-mode contract — exit 2
    // with the panic message as the reason — instead of dying with
    // the panic (exit 101 and a default panic dump on stderr).
    let dir = temp_dir("tailcall");
    let src = fixture(&dir, "tailcall.rs", TAIL_CALL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a conversion panic is a conversion failure (exit 2), not a process panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("Stable MIR conversion panicked"),
        "the reason names the conversion panic: {stderr}"
    );
    assert!(
        stderr.contains("not implemented"),
        "the reason carries the panic message: {stderr}"
    );
    assert!(
        !stderr.contains("RUST_BACKTRACE"),
        "the default panic dump is suppressed, the message is reported as the reason: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}


#[test]
fn run_mode_async_closure_conversion_panic_exits_two() {
    // Async closures hit a different `unimplemented!()` inside
    // Stable MIR (type conversion: "FIXME(async_closures)"). Same
    // contract as the tail-call case: exit 2, panic message as the
    // reason, no process panic.
    let dir = temp_dir("asyncclosure");
    let src = fixture(&dir, "asyncclosure.rs", ASYNC_CLOSURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a conversion panic is a conversion failure (exit 2), not a process panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("Stable MIR conversion panicked"),
        "the reason names the conversion panic: {stderr}"
    );
    assert!(
        stderr.contains("async_closures"),
        "the reason carries the panic message: {stderr}"
    );
    assert!(
        !stderr.contains("RUST_BACKTRACE"),
        "the default panic dump is suppressed, the message is reported as the reason: {stderr}"
    );
}


#[test]
fn run_mode_log_syntax_output_stays_off_stdout() {
    // `log_syntax!` prints to the *compiler's* stdout at compile
    // time. The compiler runs in-process, so without redirection
    // that output would land in the driver's stdout — which run
    // mode reserves for the guest program's output. It must appear
    // on stderr (compiler-side diagnostics) instead, leaving stdout
    // byte-identical to the guest's own output.
    let dir = temp_dir("logsyntax");
    let src = fixture(&dir, "logsyntax.rs", LOG_SYNTAX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program itself runs fine; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "guest marker\n",
        "stdout carries the guest's output alone"
    );
    assert!(
        stderr.contains("compile-time marker"),
        "the log_syntax output is redirected to stderr: {stderr}"
    );
}

// ----- auxiliary crates (--aux) ----------------------------------------------

const AUX_LIB_SRC: &str = r#"
fn secret(x: i32) -> i32 { x * 3 }
pub fn add(a: i32, b: i32) -> i32 { a + b }
pub fn triple(x: i32) -> i32 { secret(x) }
pub fn generic_id<T>(x: T) -> T { x }
"#;

const AUX_MAIN_SRC: &str = r#"
extern crate auxlib;
fn main() {
    println!("add={}", auxlib::add(2, 3));
    println!("triple={}", auxlib::triple(7));
}
"#;

const AUX_BASE_SRC: &str = r#"
pub fn base_val() -> i32 { 40 }
"#;

const AUX_MID_SRC: &str = r#"
extern crate base;
pub fn with_base() -> i32 { base::base_val() + 2 }
"#;

const AUX_NESTED_MAIN_SRC: &str = r#"
extern crate mid;
fn main() { println!("val={}", mid::with_base()); }
"#;

const AUX_BROKEN_SRC: &str = r#"
pub fn broken( -> i32 { 1 }
"#;

fn run_driver_with_aux(src: &Path, aux: &[(&str, &Path)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg("run").arg(src);
    for (name, path) in aux {
        cmd.arg("--aux").arg(format!("{}={}", name, path.display()));
    }
    cmd.output().expect("failed to run sparagmos-driver")
}

#[test]
fn run_mode_aux_crate_functions_execute() {
    // Functions of an auxiliary crate -- including its private
    // helpers -- are converted into the same program and execute in
    // the interpreter; the crate's generic functions are skipped
    // without affecting the run.
    let dir = temp_dir("aux-basic");
    let aux = fixture(&dir, "auxlib.rs", AUX_LIB_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "aux calls must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "add=5\ntriple=21\n",
        "aux function results reach the guest's stdout"
    );
}

#[test]
fn run_mode_nested_aux_crates_execute() {
    // Auxiliary crates are passed in dependency order: `mid` uses
    // `base`, and the main crate uses `mid`.
    let dir = temp_dir("aux-nested");
    let base = fixture(&dir, "base.rs", AUX_BASE_SRC);
    let mid = fixture(&dir, "mid.rs", AUX_MID_SRC);
    let src = fixture(&dir, "nestedmain.rs", AUX_NESTED_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("base", &base), ("mid", &mid)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested aux calls must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "val=42\n");
}

#[test]
fn run_mode_aux_compile_failure_exits_two() {
    let dir = temp_dir("aux-broken");
    let aux = fixture(&dir, "auxlib.rs", AUX_BROKEN_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an uncompilable auxiliary crate is a conversion failure; stderr={stderr}"
    );
    assert!(
        stderr.contains("auxiliary crate `auxlib` failed to compile"),
        "the reason names the auxiliary crate: {stderr}"
    );
}
