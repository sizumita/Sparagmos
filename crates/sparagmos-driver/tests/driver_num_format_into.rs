//! Integer `format_into` (the public `NumBuffer` API,
//! stable since Rust 1.98.0): build small fixture programs
//! in a temp dir, execute them with `sparagmos-driver run
//! --rustc-arg -Copt-level=...`, and check the exit-code
//! contract (0 = normal termination, 101 = guest panic,
//! 2 = could not convert/execute, with the reason on
//! stderr) plus the exact stdout.
//!
//! `format_into` is the public entry of the same core
//! digit writer `Builtin::FmtNumFmt` serves for `_fmt`:
//! `<iN>::format_into(self, &mut NumBuffer<T>) -> &str`
//! writes the value's decimal digits (with the sign, for
//! a negative signed value) right-aligned into the
//! buffer's byte array and returns the written suffix as
//! a `&str` into the buffer itself. Its MIR body is not
//! encoded in the distributed core rlib, so without the
//! model (`Builtin::FmtNumFormatInto`) a run stops at
//! "Call to core::fmt::num::imp::<impl i32>::format_into
//! (no converted body)" at every opt level. The signed
//! impls and the unsigned impls live in the `imp`
//! submodule, except the 128-bit pair, whose impls sit at
//! the `core::fmt::num` level.

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
        "sparagmos-driver-num-format-into-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

/// Run one fixture at opt-level 0, 2 and 3 and require the
/// exact native stdout at each.
fn run_ok_all_opts(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let path = fixture(&dir, "case.rs", src);
    for opt in ["-Copt-level=0", "-Copt-level=2", "-Copt-level=3"] {
        let out = run_driver_extra(&path, &["--rustc-arg", opt]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "format_into must run at {opt}; stderr={stderr}"
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            expected_stdout,
            "formatted output must match native at {opt}; stderr={stderr}"
        );
    }
}

/// The original stop: one buffer per integer type, at
/// zero, extremal and negative values. Every signed type
/// renders its sign inside the buffer; `i32::MIN` and
/// `i128::MIN` exactly fill their buffers.
#[test]
fn format_into_all_types() {
    run_ok_all_opts(
        "all-types",
        r#"
fn main() {
    let mut b32 = core::fmt::NumBuffer::new();
    println!("{}", 0i32.format_into(&mut b32));
    println!("{}", 32i32.format_into(&mut b32));
    println!("{}", i32::MIN.format_into(&mut b32));
    let mut b64 = core::fmt::NumBuffer::new();
    println!("{}", (-1972i64).format_into(&mut b64));
    let mut bu32 = core::fmt::NumBuffer::new();
    println!("{}", 4294967295u32.format_into(&mut bu32));
    let mut b128 = core::fmt::NumBuffer::new();
    println!("{}", u128::MAX.format_into(&mut b128));
    let mut bi128 = core::fmt::NumBuffer::new();
    println!("{}", i128::MIN.format_into(&mut bi128));
    let mut b8 = core::fmt::NumBuffer::new();
    let s: &str = 7u8.format_into(&mut b8);
    println!("{s}");
    let mut busize = core::fmt::NumBuffer::new();
    println!("{}", 123456usize.format_into(&mut busize));
    let mut bi16 = core::fmt::NumBuffer::new();
    println!("{}", (-300i16).format_into(&mut bi16));
}
"#,
        "0\n32\n-2147483648\n-1972\n4294967295\n\
         340282366920938463463374607431768211455\n\
         -170141183460469231731687303715884105728\n7\n123456\n-300\n",
    );
}

/// Runtime-computed values (through `black_box`, so no
/// constant folding) of every integer type, buffer reuse
/// with each result read before the next call overwrites
/// the buffer, the returned `&str` feeding the formatting
/// machinery and `push_str`, and buffers reached through
/// a helper function and through a struct field.
#[test]
fn format_into_runtime_values_and_consumers() {
    run_ok_all_opts(
        "runtime",
        r#"
use std::hint::black_box;

fn via_helper(buf: &mut core::fmt::NumBuffer<i64>, v: i64) -> String {
    let s = v.format_into(buf);
    s.to_string()
}

struct Holder {
    buf: core::fmt::NumBuffer<u32>,
    label: String,
}

fn main() {
    let mut bi8 = core::fmt::NumBuffer::new();
    println!("{}", black_box(-128i8).format_into(&mut bi8));
    let mut bu8 = core::fmt::NumBuffer::new();
    println!("{}", black_box(255u8).format_into(&mut bu8));
    let mut bi16 = core::fmt::NumBuffer::new();
    println!("{}", black_box(-32768i16).format_into(&mut bi16));
    let mut bu16 = core::fmt::NumBuffer::new();
    println!("{}", black_box(65535u16).format_into(&mut bu16));
    let mut bi32 = core::fmt::NumBuffer::new();
    println!("{}", black_box(-1i32).format_into(&mut bi32));
    let mut bu32 = core::fmt::NumBuffer::new();
    println!("{}", black_box(0u32).format_into(&mut bu32));
    let mut bi64 = core::fmt::NumBuffer::new();
    println!("{}", black_box(i64::MIN).format_into(&mut bi64));
    let mut bu64 = core::fmt::NumBuffer::new();
    println!("{}", black_box(u64::MAX).format_into(&mut bu64));
    let mut bisz = core::fmt::NumBuffer::new();
    println!("{}", black_box(-42isize).format_into(&mut bisz));
    let mut busz = core::fmt::NumBuffer::new();
    println!("{}", black_box(42usize).format_into(&mut busz));
    let mut bi128 = core::fmt::NumBuffer::new();
    println!("{}", black_box(0i128).format_into(&mut bi128));
    println!("{}", black_box(-1i128).format_into(&mut bi128));

    let mut rb = core::fmt::NumBuffer::new();
    for v in [5i32, -67, 890, -12345, 0] {
        let s = black_box(v).format_into(&mut rb);
        println!("{s}");
    }

    let mut fb = core::fmt::NumBuffer::new();
    let s = black_box(-1972i32).format_into(&mut fb);
    assert_eq!(s, "-1972");
    println!("[{:>8}]", s);
    let mut owned = String::from("n=");
    owned.push_str(s);
    println!("{owned}");

    let mut hb = core::fmt::NumBuffer::new();
    println!("{}", via_helper(&mut hb, black_box(-9_999_999_999i64)));
    println!("{}", via_helper(&mut hb, black_box(7i64)));

    let mut h = Holder { buf: core::fmt::NumBuffer::new(), label: "hits".to_string() };
    let s = black_box(2026u32).format_into(&mut h.buf);
    println!("{}:{}", h.label, s);
    let s2 = black_box(31u32).format_into(&mut h.buf);
    println!("{}:{}", h.label, s2);
}
"#,
        "-128\n255\n-32768\n65535\n-1\n0\n-9223372036854775808\n\
         18446744073709551615\n-42\n42\n0\n-1\n\
         5\n-67\n890\n-12345\n0\n[   -1972]\nn=-1972\n\
         -9999999999\n7\nhits:2026\nhits:31\n",
    );
}

/// The returned `&str` pushed into an empty `String` as
/// well as into one already holding text: the suffix
/// reference denotes the buffer's own storage, so the
/// copy reads the digits the call wrote.
#[test]
fn format_into_result_push_str() {
    run_ok_all_opts(
        "push-str",
        r#"
use std::hint::black_box;

fn main() {
    let mut fb = core::fmt::NumBuffer::new();
    let s = black_box(-1972i32).format_into(&mut fb);
    let mut owned = String::from("n=");
    owned.push_str(s);
    println!("{owned}");
    let mut acc = String::new();
    acc.push_str(s);
    println!("{acc}");
}
"#,
        "n=-1972\n-1972\n",
    );
}
