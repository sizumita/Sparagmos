//! Run-jit panic-report tests: a guest panic under
//! `sparagmos-driver run-jit` must exit 101 with the panic's
//! own message on stderr -- the message the native binary
//! reports -- for each panic entry shape: `panic_fmt` (a
//! literal or formatted `panic!`), `panicking::panic` (the
//! bare `panic!()`), `assert_failed` (the `assert_eq!` /
//! `assert_ne!` macros) and the bounds-check `Assert` whose
//! message is formatted from its operands at run time.
//! Before the JIT builtin layer modelled these callees, every
//! one of them surfaced as "call to function ... which is not
//! part of the program" and the guest's message was lost. The
//! pre-2021 `panic!` shapes are covered too: they lower to
//! `begin_panic`, whose payload reports by std's
//! `payload_as_str` rule (a string payload as itself, any other
//! payload as the opaque box).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits
/// (the registry shape of the other driver test files, #537).
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

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("sparagmos-driver-jit-panic-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn run_driver_jit(src: &Path) -> Output {
    run_driver_jit_edition(src, "2021")
}

fn run_driver_jit_edition(src: &Path, edition: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run-jit")
        .arg(src)
        .arg("--edition")
        .arg(edition)
        .output()
        .expect("failed to run sparagmos-driver")
}

/// Run one fixture in JIT mode and require the guest-panic
/// contract: exit 101 and `expected` in the panic message on
/// stderr.
fn expect_jit_panic(tag: &str, src: &str, expected: &str) {
    expect_jit_panic_edition(tag, src, expected, "2021");
}

fn expect_jit_panic_edition(tag: &str, src: &str, expected: &str, edition: &str) {
    let dir = temp_dir(tag);
    let file = fixture(&dir, "case.rs", src);
    let out = run_driver_jit_edition(&file, edition);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "{tag}: expected exit 101, stderr: {stderr}"
    );
    assert!(
        stderr.contains(expected),
        "{tag}: expected stderr to contain {expected:?}, got: {stderr}"
    );
}

#[test]
fn jit_panic_literal_message() {
    expect_jit_panic("literal", "fn main() { panic!(\"quux\"); }\n", "quux");
}

#[test]
fn jit_panic_bare_message() {
    expect_jit_panic("bare", "fn main() { panic!(); }\n", "explicit panic");
}

#[test]
fn jit_panic_formatted_message() {
    expect_jit_panic(
        "formatted",
        "fn main() { panic!(\"{} is {}\", \"x\", 5); }\n",
        "x is 5",
    );
}

#[test]
fn jit_panic_from_callee() {
    expect_jit_panic(
        "callee",
        "fn deep() { panic!(\"from below\"); }\nfn main() { deep(); }\n",
        "from below",
    );
}

#[test]
fn jit_assert_eq_with_message() {
    let dir = temp_dir("assert-eq-msg");
    let file = fixture(
        &dir,
        "case.rs",
        "fn main() { assert_eq!(1 + 1, 3, \"one plus one should be 3\"); }\n",
    );
    let out = run_driver_jit(&file);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr: {stderr}");
    assert!(
        stderr.contains("assertion `left == right` failed: one plus one should be 3"),
        "stderr: {stderr}"
    );
    assert!(stderr.contains("left: 2"), "stderr: {stderr}");
    assert!(stderr.contains("right: 3"), "stderr: {stderr}");
}

#[test]
fn jit_assert_ne_without_message() {
    expect_jit_panic(
        "assert-ne",
        "fn main() { assert_ne!(2, 2); }\n",
        "assertion `left != right` failed",
    );
}

#[test]
fn jit_panic_2015_literal_message() {
    // The pre-2021 `panic!` lowers to `begin_panic` with a
    // `&str` payload (a constant: no place type), whose message
    // is the payload itself.
    expect_jit_panic_edition(
        "panic-2015-literal",
        "fn main() { panic!(\"quux\"); }\n",
        "quux",
        "2015",
    );
}

#[test]
fn jit_panic_2015_bare_message() {
    expect_jit_panic_edition(
        "panic-2015-bare",
        "fn main() { panic!(); }\n",
        "explicit panic",
        "2015",
    );
}

#[test]
fn jit_panic_2015_string_payload() {
    // A `String` payload reports as its contents.
    expect_jit_panic_edition(
        "panic-2015-string",
        "fn main() { panic!(String::from(\"owned boom\")); }\n",
        "owned boom",
        "2015",
    );
}

#[test]
fn jit_panic_any_opaque_payload() {
    // A non-string payload reports as the opaque box.
    expect_jit_panic(
        "panic-any-opaque",
        "fn main() { std::panic::panic_any(42); }\n",
        "Box<dyn Any>",
    );
}

#[test]
fn jit_bounds_check_assert_message() {
    expect_jit_panic(
        "bounds-check",
        "const C: &'static [u8; 5] = b\"hello\";\n\
         #[allow(unconditional_panic)]\n\
         fn test() -> u8 { C[10] }\n\
         fn main() { test(); }\n",
        "index out of bounds: the len is 5 but the index is 10",
    );
}
