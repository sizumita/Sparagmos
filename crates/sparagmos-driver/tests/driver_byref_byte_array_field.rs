//! By-reference constants naming a byte array that is one field
//! of a larger allocation (Issue #396): a reference constant
//! whose pointee is a `[u8; N]` field of a static or const item
//! of tuple/struct type. The byte-array shortcuts in the
//! driver's constant decoding read the backing allocation from
//! the field's offset to the allocation's end, so the decoded
//! `Bytes` value absorbed the following fields (a wrong value
//! with a wrong length), and trailing uninitialised padding
//! bytes after the field made the read fail outright (the
//! constant kept its `ByRef` form and conversion rejected it,
//! or the constant was recorded unreadable). The declared
//! length of the array now governs both reads. These fixtures
//! run end to end through `sparagmos-driver run` and pin the
//! exact stdout.

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

fn run_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-byref-byte-array-field-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn run_ok(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src = fixture(&dir, "prog.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "stdout must match the native run"
    );
}

const STATIC_TUPLE_FIELD_SRC: &str = r#"
static S: ([u8; 3], u8) = ([1, 2, 3], 99);
const R: &[u8; 3] = &S.0;
fn main() {
    println!("{:?} {}", R, R.len());
}
"#;

#[test]
fn run_mode_byref_byte_array_static_tuple_field() {
    run_ok("static-tuple-field", STATIC_TUPLE_FIELD_SRC, "[1, 2, 3] 3\n");
}

const STATIC_STRUCT_FIELD_SRC: &str = r#"
struct T {
    a: [u8; 3],
    b: u8,
}
static S: T = T { a: [4, 5, 6], b: 77 };
const R: &[u8; 3] = &S.a;
fn main() {
    println!("{:?} {}", R, R.len());
}
"#;

#[test]
fn run_mode_byref_byte_array_static_struct_field() {
    run_ok("static-struct-field", STATIC_STRUCT_FIELD_SRC, "[4, 5, 6] 3\n");
}

const CONST_TUPLE_FIELD_SRC: &str = r#"
const S: ([u8; 3], u8) = ([10, 20, 30], 99);
const R: &[u8; 3] = &S.0;
fn main() {
    println!("{:?} {}", R, R.len());
}
"#;

#[test]
fn run_mode_byref_byte_array_const_tuple_field() {
    run_ok("const-tuple-field", CONST_TUPLE_FIELD_SRC, "[10, 20, 30] 3\n");
}

const STATIC_PADDED_LAST_FIELD_SRC: &str = r#"
#[repr(C)]
struct T {
    b: u32,
    a: [u8; 3],
}
static S: T = T { b: 42, a: [4, 5, 6] };
const R: &[u8; 3] = &S.a;
fn main() {
    println!("{:?} {}", R, R.len());
}
"#;

#[test]
fn run_mode_byref_byte_array_static_padded_last_field() {
    run_ok(
        "static-padded-last-field",
        STATIC_PADDED_LAST_FIELD_SRC,
        "[4, 5, 6] 3\n",
    );
}

const CONST_MIDDLE_FIELD_SRC: &str = r#"
const S: (u32, [u8; 3], u16) = (42, [7, 8, 9], 5);
const R: &[u8; 3] = &S.1;
fn main() {
    println!("{:?} {}", R, R.len());
}
"#;

#[test]
fn run_mode_byref_byte_array_const_middle_field() {
    run_ok("const-middle-field", CONST_MIDDLE_FIELD_SRC, "[7, 8, 9] 3\n");
}

const WHOLE_ARRAY_SRC: &str = r#"
static A: [u8; 4] = [9, 8, 7, 6];
const R: &[u8; 4] = &A;
fn main() {
    println!("{:?} {}", R, R.len());
}
"#;

#[test]
fn run_mode_byref_byte_array_whole_static() {
    run_ok("whole-array", WHOLE_ARRAY_SRC, "[9, 8, 7, 6] 4\n");
}
