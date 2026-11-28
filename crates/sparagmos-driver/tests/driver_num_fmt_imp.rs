//! Integer `to_string` at opt-level >= 2: build small fixture
//! programs in a temp dir, execute them with
//! `sparagmos-driver run --rustc-arg -Copt-level=2/3`, and check
//! the exit-code contract (0 = normal termination, 101 = guest
//! panic, 2 = could not convert/execute, with the reason on
//! stderr) plus the exact stdout.
//!
//! At opt-level >= 2 rustc inlines the integer `Display` bodies
//! (and the `SpecToString` bodies built on them) into the guest,
//! which then calls `core::fmt::num::imp::<impl uN>::_fmt`
//! directly -- the unstable core-internal digit writer whose
//! MIR body is not encoded in the distributed core rlib. The
//! interpreter serves it as a builtin (`Builtin::FmtNumFmt`):
//! the digits are written right-aligned into the caller's
//! `&mut [MaybeUninit<u8>]` buffer and the result is the
//! buffer's written suffix as a `&str`. Without the model the
//! run stops at "Call to core::fmt::num::imp::<impl u32>::_fmt
//! (no converted body)".

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
        "sparagmos-driver-num-fmt-imp-{}-{tag}",
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
            "integer formatting must run at {opt}; stderr={stderr}"
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            expected_stdout,
            "formatted output must match native at {opt}; stderr={stderr}"
        );
    }
}

/// The original stop: a `u32` receiver's `to_string()`, whose
/// inlined body calls `<impl u32>::_fmt` directly.
#[test]
fn u32_to_string_at_opt_levels() {
    run_ok_both_opts(
        "u32",
        r#"
fn main() {
    assert_eq!(7u32.to_string(), "7");
    assert_eq!(0u32.to_string(), "0");
    assert_eq!(u32::MAX.to_string(), "4294967295");
    println!("{}", 7u32.to_string());
}
"#,
        "7\n",
    );
}

/// Every unsigned width's digit writer, at its boundary values
/// (the `u128` impl sits at the `core::fmt::num` level, the
/// others in its `imp` submodule; both spellings are served).
#[test]
fn unsigned_to_string_all_widths() {
    run_ok_both_opts(
        "widths",
        r#"
fn main() {
    assert_eq!(255u8.to_string(), "255");
    assert_eq!(65535u16.to_string(), "65535");
    assert_eq!(18446744073709551615u64.to_string(), "18446744073709551615");
    assert_eq!(
        340282366920938463463374607431768211455u128.to_string(),
        "340282366920938463463374607431768211455"
    );
    assert_eq!(12345usize.to_string(), "12345");
    println!("widths ok");
}
"#,
        "widths ok\n",
    );
}

/// Non-negative signed receivers: their `SpecToString` body
/// takes the no-sign branch and hands the `_fmt` result to
/// `String::push_str` over a capacity-built string.
#[test]
fn positive_signed_to_string_at_opt_levels() {
    run_ok_both_opts(
        "positive-signed",
        r#"
fn main() {
    assert_eq!(123i32.to_string(), "123");
    assert_eq!(0i8.to_string(), "0");
    assert_eq!(i64::MAX.to_string(), "9223372036854775807");
    println!("positive ok");
}
"#,
        "positive ok\n",
    );
}

/// Negative values through `Display` (the `format!` route):
/// the inlined `Display` body writes the digits with `_fmt`
/// and pads them with the sign.
#[test]
fn negative_display_via_format() {
    run_ok_both_opts(
        "negative-display",
        r#"
fn main() {
    assert_eq!(format!("{}", -123i32), "-123");
    assert_eq!(format!("{}", i64::MIN), "-9223372036854775808");
    assert_eq!(format!("{}", -1i8), "-1");
    println!("negative ok");
}
"#,
        "negative ok\n",
    );
}

/// The `_fmt` result consumed by the formatter's padding:
/// zero padding and width alignment over the digit string.
#[test]
fn padded_unsigned_format() {
    run_ok_both_opts(
        "padded",
        r#"
fn main() {
    assert_eq!(format!("{:05}", 42u32), "00042");
    assert_eq!(format!("{:>6}", 42u32), "    42");
    assert_eq!(format!("{:<6}|", 42u32), "42    |");
    println!("padded ok");
}
"#,
        "padded ok\n",
    );
}
