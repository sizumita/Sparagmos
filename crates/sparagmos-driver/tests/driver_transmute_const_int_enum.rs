//! Run-mode tests for integer-constant -> ADT transmutes
//! (Issue #513): when the input of a `transmute` is an integer
//! literal, the cast arrives with a constant operand that
//! carries no place type, and the integer -> ADT conversion
//! arms (a fieldless enum read through its discriminant table,
//! a one-field wrapper over one, a single-variant enum holding
//! the integer as its payload) key on the source being an
//! integer type. The converter now takes that type from the
//! constant's recorded type, exactly as a place holding the
//! integer would present it. On the base tree every case here
//! fails with `unsupported rvalue: Cast Transmute`, where the
//! same cast of a place holding the integer converted.

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

fn run_driver(src: &Path, opt: bool) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg("run").arg(src);
    if opt {
        cmd.arg("--rustc-arg").arg("-O");
    }
    cmd.output().expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-tcie-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn run_ok(tag: &str, src: &str, opt: bool, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver(&src_path, opt);
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

/// The failing record's crossing (tests/ui/mir/enum/
/// single_with_repr_ok.rs): a `u16` literal transmuted into a
/// single-variant `#[repr(u16)]` fieldless enum.
#[test]
fn int_literal_into_single_variant_repr_enum() {
    let src = r#"#[allow(dead_code)]
#[repr(u16)]
enum Single {
    A
}

fn main() {
    let val: Single = unsafe { std::mem::transmute::<u16, Single>(0) };
    println!("{}", val as u16);
}
"#;
    run_ok("single-repr-opt", src, true, "0\n");
    run_ok("single-repr", src, false, "0\n");
}

/// A literal selecting a later variant of a multi-variant
/// fieldless enum, through the same discriminant table.
#[test]
fn int_literal_into_multi_variant_fieldless_enum() {
    let src = r#"#[allow(dead_code)]
#[repr(u8)]
enum Multi {
    A = 0,
    B = 1,
    C = 2,
}

fn main() {
    let val: Multi = unsafe { std::mem::transmute::<u8, Multi>(2) };
    println!("{}", val as u8);
}
"#;
    run_ok("multi-opt", src, true, "2\n");
    run_ok("multi", src, false, "2\n");
}

/// A negative literal into an enum whose discriminant is
/// negative: the signed recorded type carries the sign.
#[test]
fn negative_int_literal_into_fieldless_enum() {
    let src = r#"#[allow(dead_code)]
#[repr(i8)]
enum Neg {
    M = -1,
    Z = 0,
}

fn main() {
    let val: Neg = unsafe { std::mem::transmute::<i8, Neg>(-1) };
    println!("{}", val as i8);
}
"#;
    run_ok("neg-opt", src, true, "-1\n");
    run_ok("neg", src, false, "-1\n");
}

/// A literal into a transparent one-field wrapper over a
/// fieldless enum: the discriminant integer wrapped back up.
#[test]
fn int_literal_into_wrapper_over_fieldless_enum() {
    let src = r#"#[allow(dead_code)]
#[repr(u8)]
enum Multi {
    A = 0,
    B = 1,
    C = 2,
}

#[repr(transparent)]
struct Wrap(Multi);

fn main() {
    let val: Wrap = unsafe { std::mem::transmute::<u8, Wrap>(1) };
    println!("{}", val.0 as u8);
}
"#;
    run_ok("wrap-opt", src, true, "1\n");
    run_ok("wrap", src, false, "1\n");
}

/// A literal into a single-variant enum carrying it as the
/// variant's integer payload.
#[test]
fn int_literal_into_single_variant_payload_enum() {
    let src = r#"#[allow(dead_code)]
enum Payload {
    V(u16),
}

fn main() {
    let val: Payload = unsafe { std::mem::transmute::<u16, Payload>(7) };
    match val {
        Payload::V(x) => println!("{x}"),
    }
}
"#;
    run_ok("payload-opt", src, true, "7\n");
    run_ok("payload", src, false, "7\n");
}
