//! String raw-parts walks at body level: build small fixture
//! programs in a temp dir, execute them with
//! `sparagmos-driver run --rustc-arg -Copt-level=0..3`, and
//! check the exit-code contract (0 = normal termination,
//! 101 = guest panic, 2 = could not convert/execute, with the
//! reason on stderr) plus the exact stdout.
//!
//! At opt-level >= 2 the `push_str` behind a negative number's
//! `to_string()` (and behind `push_str` generally) is inlined
//! far enough that the guest reaches the collected
//! `Vec::append_elements` body, which walks the string's raw
//! parts: it reads the buffer pointer out of the
//! `RawVec`/`Vec` aggregate, offsets it, copies the new bytes
//! through it and stores the advanced length back. A string
//! the modelled mutators rewrote (`String::push` stores the
//! text back in the abstract string form) or that was born
//! abstract (the `format!` result) has no aggregate for that
//! walk: the field-chain read returns the string itself and
//! the run stops at "type mismatch in pointer offset:
//! expected pointer value, found string". `append_elements`
//! opens with `Vec::reserve` on the string's vector place,
//! and the interpreter materialises the storage the parts
//! denote -- a fresh allocation holding the bytes, in the
//! real aggregate form -- at that reserve, so the walk
//! proceeds on the form it expects.

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
        "sparagmos-driver-string-raw-parts-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

/// Run one fixture at every opt-level and require the exact
/// native stdout at each.
fn run_ok_all_opts(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let path = fixture(&dir, "case.rs", src);
    for opt in [
        "-Copt-level=0",
        "-Copt-level=1",
        "-Copt-level=2",
        "-Copt-level=3",
    ] {
        let out = run_driver_extra(&path, &["--rustc-arg", opt]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "string raw-parts walk must run at {opt}; stderr={stderr}"
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            expected_stdout,
            "output must match native at {opt}; stderr={stderr}"
        );
    }
}

/// The pinned stop: a negative signed value's `to_string()`.
/// The sign is pushed by the modelled `String::push` (which
/// rewrites the place to the abstract form) and the digits
/// follow through an inlined `push_str`, whose
/// `Vec::append_elements` body walks the raw parts.
#[test]
fn negative_to_string_at_all_opt_levels() {
    run_ok_all_opts(
        "negative-to-string",
        r#"
fn main() {
    assert_eq!((-42i32).to_string(), "-42");
    assert_eq!((-7i64).to_string(), "-7");
    assert_eq!(i64::MIN.to_string(), "-9223372036854775808");
    assert_eq!((-1i8).to_string(), "-1");
    println!("{}", (-42i32).to_string());
}
"#,
        "-42\n",
    );
}

/// The same walk reached through `format!`: the digits and
/// the sign are written into the formatter's string, whose
/// appends are the inlined `push_str` body.
#[test]
fn negative_format_at_all_opt_levels() {
    run_ok_all_opts(
        "negative-format",
        r#"
fn main() {
    assert_eq!(format!("{}", -7i64), "-7");
    assert_eq!(format!("{}", -123i32), "-123");
    assert_eq!(format!("{:05}", -42i32), "-0042");
    println!("{}", format!("{}", -7i64));
}
"#,
        "-7\n",
    );
}

/// The minimal guest the pin was reduced to:
/// `String::with_capacity` + `push('-')` + `push_str`. The
/// modelled `push` rewrites the capacity-built aggregate to
/// the abstract form; the following `push_str` walks the raw
/// parts of the place it denotes.
#[test]
fn push_then_push_str_minimal() {
    run_ok_all_opts(
        "push-then-push-str",
        r#"
fn main() {
    let mut s = String::with_capacity(8);
    s.push('-');
    s.push_str("123");
    assert_eq!(s, "-123");
    println!("{}", s);
}
"#,
        "-123\n",
    );
}

/// A string born in the abstract form (the `format!` result)
/// walked by a later `push_str`: no modelled mutator rewrote
/// it, the walk meets the abstract form directly.
#[test]
fn born_abstract_string_push_str() {
    run_ok_all_opts(
        "born-abstract",
        r#"
fn main() {
    let mut s = format!("ab");
    s.push_str("cd");
    assert_eq!(s, "abcd");
    let mut t = String::from("ab");
    t.push('x');
    t.push_str("cd");
    assert_eq!(t, "abxcd");
    println!("{} {}", s, t);
}
"#,
        "abcd abxcd\n",
    );
}

/// The walk with growth past the materialised capacity, and
/// a second modelled rewrite after the walk: the place must
/// stay coherent across the abstract/aggregate transitions.
#[test]
fn raw_parts_walk_growth_and_rewrite() {
    run_ok_all_opts(
        "growth-and-rewrite",
        r#"
fn main() {
    let mut s = String::from("ab");
    s.push('x');
    s.push_str("cdefghij");
    assert_eq!(s, "abxcdefghij");
    s.push('!');
    s.insert(0, '>');
    assert_eq!(s, ">abxcdefghij!");
    println!("{}", s);
}
"#,
        ">abxcdefghij!\n",
    );
}
