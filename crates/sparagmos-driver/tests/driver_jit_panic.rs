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

#[test]
fn jit_panic_report_is_std_default_shape() {
    // #540: a guest panic under run-jit prints std's default
    // panic report on stderr, not the JIT path's former invented
    // `thread 'main' panicked: <msg>` line. The report head
    // names the panic's location (the `panic!` call site, file
    // :1:13 in the fixture below) as std's does; with the
    // backtrace environment off the report ends with the
    // `RUST_BACKTRACE=1` note.
    let dir = temp_dir("report-shape");
    let file = fixture(&dir, "case.rs", "fn main() { panic!(\"quux\"); }\n");
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run-jit")
        .arg(&file)
        .arg("--edition")
        .arg("2021")
        .env("RUST_BACKTRACE", "0")
        .env_remove("RUST_LIB_BACKTRACE")
        .output()
        .expect("failed to run sparagmos-driver");
    assert_eq!(out.status.code(), Some(101), "expected exit 101");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        stderr,
        format!(
            "\nthread 'main' (1) panicked at {}:1:13:\nquux\n\
             note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
            file.display()
        ),
        "stderr must be exactly std's default panic report"
    );
    assert!(
        !stderr.contains("panicked: quux"),
        "the invented single-line form must not appear: {stderr}"
    );
}

#[test]
fn jit_panic_report_names_assert_location() {
    // #540: a bounds-check `Assert` panic under run-jit also
    // names its location: the indexing site inside the callee
    // (line 3, the `C[10]` expression), not the caller's.
    let dir = temp_dir("report-assert-location");
    let file = fixture(
        &dir,
        "case.rs",
        "const C: &'static [u8; 5] = b\"hello\";\n\
         #[allow(unconditional_panic)]\n\
         fn test() -> u8 { C[10] }\n\
         fn main() { test(); }\n",
    );
    let out = run_driver_jit(&file);
    assert_eq!(out.status.code(), Some(101), "expected exit 101");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(&format!(
            "thread 'main' (1) panicked at {}:3:19:\n\
             index out of bounds: the len is 5 but the index is 10",
            file.display()
        )),
        "stderr must name the assert site: {stderr}"
    );
}

#[test]
fn jit_instant_now_reaches_the_panic() {
    // #550 remainder: `Instant::now` was an unmodelled callee
    // reached before the panic, so the run reported the
    // unknown-callee text instead of the guest's message.
    expect_jit_panic(
        "instant-now",
        "fn main() { let _t = std::time::Instant::now(); panic!(\"after instant\"); }\n",
        "after instant",
    );
}

#[test]
fn jit_system_time_now_reaches_the_panic() {
    expect_jit_panic(
        "system-time-now",
        "fn main() { let _t = std::time::SystemTime::now(); panic!(\"after systime\"); }\n",
        "after systime",
    );
}

#[test]
fn jit_box_new_value_is_readable_before_the_panic() {
    // #550 remainder: `Box::new` was an unmodelled callee. The
    // box the builtin builds must read back through the
    // `BoxDerefTransmute` cast the dereference lowers to.
    expect_jit_panic(
        "box-new",
        "fn main() { let b = Box::new(41u64); assert_eq!(*b + 1, 42); panic!(\"after box\"); }\n",
        "after box",
    );
}

#[test]
fn jit_option_unwrap_none_reports_the_std_message() {
    // #550 remainder: `Option::unwrap` was an unmodelled
    // callee; on `None` the run must report std's own unwrap
    // message at the call site, not the unknown-callee text.
    expect_jit_panic(
        "unwrap-none",
        "fn main() { let x: Option<u64> = None; let _y = x.unwrap(); }\n",
        "called `Option::unwrap()` on a `None` value",
    );
}

#[test]
fn jit_option_unwrap_some_reaches_the_panic() {
    expect_jit_panic(
        "unwrap-some",
        "fn main() { let x: Option<u64> = Some(7); assert_eq!(x.unwrap(), 7); panic!(\"after unwrap\"); }\n",
        "after unwrap",
    );
}

#[test]
fn jit_size_of_in_generic_context_reaches_the_panic() {
    expect_jit_panic(
        "size-of-generic",
        "fn sz<T>() -> usize { std::mem::size_of::<T>() }\n\
         fn main() { assert_eq!(sz::<u32>(), 4); assert_eq!(sz::<(u8, u64)>(), 16); panic!(\"after sizeof\"); }\n",
        "after sizeof",
    );
}
