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
    // The driver is built with RUSTC_BOOTSTRAP=1; guests that use
    // feature gates (explicit tail calls, log_syntax) need it at
    // run time too, regardless of how the test harness was invoked.
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .env("RUSTC_BOOTSTRAP", "1")
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
