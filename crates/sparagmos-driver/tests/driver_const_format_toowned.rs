//! All-constant `format!` at opt-level >= 2 (Issue #462): build
//! small fixture programs in a temp dir, execute them with
//! `sparagmos-driver run --rustc-arg -Copt-level=2/3`, and check
//! the exit-code contract (0 = normal termination, 101 = guest
//! panic, 2 = could not convert/execute, with the reason on
//! stderr) plus the exact stdout.
//!
//! At opt-level >= 2 rustc inlines `alloc::fmt::format` into the
//! guest body. Its `args.as_str().map_or_else(|| format_inner(args),
//! ToOwned::to_owned)` shape means an all-constant `format!` takes
//! the `ToOwned::to_owned` branch, through an `FnOnce::call_once`
//! instance for the fn item `<fn(&str) -> <str as ToOwned>::Owned
//! {ToOwned::to_owned}>`. That instance has no MIR body at all, so
//! the call is served by the interpreter's run-time name fallback
//! (`Builtin::ToOwned`); without it the run stops at
//! "unknown function `std::borrow::ToOwned::to_owned`".
//! Runtime-argument formats take the `format_inner` branch instead
//! (served since #397) and are pinned here as the control case.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

fn fixture(dir: &Path, name: &str, src: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    path
}

fn run_driver_extra(src: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .args(extra)
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-const-format-toowned-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

/// Run one fixture at opt-level 2 and 3 and require the exact
/// native stdout at both.
fn run_ok_both_opts(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let path = fixture(&dir, "case.rs", src);
    for opt in ["-Copt-level=2", "-Copt-level=3"] {
        let out = run_driver_extra(&path, &["--rustc-arg", opt]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "all-constant format! must run at {opt}; stderr={stderr}"
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            expected_stdout,
            "formatted output must match native at {opt}; stderr={stderr}"
        );
    }
}

/// The issue's first repro: a constant integer argument. Every
/// argument is a compile-time constant, so the optimized
/// `fmt::format` body takes the `ToOwned::to_owned` branch.
#[test]
fn const_format_int_argument() {
    run_ok_both_opts(
        "int-arg",
        r#"
fn main() { let s = format!("{}", 42); println!("{s}"); }
"#,
        "42\n",
    );
}

/// The issue's second repro: a constant string argument.
#[test]
fn const_format_str_argument() {
    run_ok_both_opts(
        "str-arg",
        r#"
fn main() { let s = format!("{}", "lit"); println!("{s}"); }
"#,
        "lit\n",
    );
}

/// The issue's third repro: no arguments at all, a plain literal
/// template -- the purest `as_str()` shape.
#[test]
fn const_format_plain_literal() {
    run_ok_both_opts(
        "plain-literal",
        r#"
fn main() { let s = format!("hello"); println!("{s}"); }
"#,
        "hello\n",
    );
}

/// Several constant arguments substituted into one template, and
/// the resulting `String` used further (its length, a push, a
/// comparison), so the owned value must be a real string and not
/// merely print once.
#[test]
fn const_format_multiple_args_string_used() {
    run_ok_both_opts(
        "multi-arg",
        r#"
fn main() {
    let mut s = format!("{}-{}-{}", "ab", 7, 'z');
    s.push('!');
    assert_eq!(s, "ab-7-z!");
    println!("{} {}", s, s.len());
}
"#,
        "ab-7-z! 7\n",
    );
}

/// Constant scalar arguments whose formatter reaches the fmt
/// engine as a raw `Argument` over a value-backed target: the
/// denoted value alone cannot denominate them (a `char` and a
/// `u32` are the same value), so the formatter constant must
/// carry its impl instance name for the engine to recover the
/// `Self` type. Each scalar below renders through its own
/// type: the char as a character, the cast as a number.
#[test]
fn const_format_constant_scalar_args() {
    run_ok_both_opts(
        "scalar-args",
        r#"
fn main() {
    let s = format!("{} {} {} {}", 'z', true, 1.5f64, 'z' as u32);
    let t = format!("{:?} {:x}", 'z', 'z' as u32);
    println!("{s} | {t}");
}
"#,
        "z true 1.5 122 | 'z' 7a\n",
    );
}

/// Control: a runtime argument takes the `format_inner` branch
/// of the same inlined `fmt::format` body, not the `ToOwned`
/// branch; it must keep working at the same opt levels.
#[test]
fn runtime_format_argument_control() {
    run_ok_both_opts(
        "runtime-arg",
        r#"
fn main() {
    let n = std::hint::black_box(42);
    let s = format!("{}", n);
    println!("{s}");
}
"#,
        "42\n",
    );
}

/// Guard: the slice receiver of the same builtin still yields a
/// vector of the elements (the `str` shape must not swallow the
/// `[T]::to_owned` shape).
#[test]
fn slice_to_owned_still_yields_vec() {
    run_ok_both_opts(
        "slice-to-owned",
        r#"
fn main() {
    let a = [1, 2, 3];
    let v: Vec<i32> = a[..].to_owned();
    println!("{:?}", v);
}
"#,
        "[1, 2, 3]\n",
    );
}
