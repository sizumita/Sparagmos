//! By-reference constants naming arrays of zero-sized elements
//! past the element-wise bound (Issue #396): a reference constant
//! whose pointee is an array with more elements than the
//! element-wise decode bound, of a zero-sized element type. Every
//! element stores zero bytes, so the typed decode materialises
//! the array from one element decode replicated to the declared
//! count, charged to the shared expansion budget; before this,
//! such a constant fell back to the by-reference form and
//! conversion rejected it ("unsupported constant: by-reference
//! constant"). These fixtures run end to end through
//! `sparagmos-driver run` and pin the exact stdout.

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
        "sparagmos-driver-byref-zst-array-{}-{tag}",
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

const UNIT_ARRAY_SRC: &str = r#"
const A: &[(); 5000] = &[(); 5000];
fn main() {
    let mut n = 0;
    for _ in A { n += 1; }
    println!("{} {}", A.len(), n);
}
"#;

#[test]
fn run_mode_byref_unit_array_past_bound() {
    run_ok("unit-array", UNIT_ARRAY_SRC, "5000 5000\n");
}

const ZST_STRUCT_ARRAY_SRC: &str = r#"
#[derive(Clone, Copy)]
struct Z;
const BIG: &[Z; 5000] = &[Z; 5000];
fn main() {
    println!("{}", BIG.len());
}
"#;

#[test]
fn run_mode_byref_zst_struct_array_past_bound() {
    run_ok("zst-struct-array", ZST_STRUCT_ARRAY_SRC, "5000\n");
}

const NESTED_ZST_ARRAY_SRC: &str = r#"
const A: &[[(); 5000]; 2] = &[[(); 5000]; 2];
fn main() {
    println!("{} {}", A.len(), A[1].len());
}
"#;

#[test]
fn run_mode_byref_nested_zst_array_past_bound() {
    run_ok("nested-zst-array", NESTED_ZST_ARRAY_SRC, "2 5000\n");
}

const ZST_FIELD_ARRAY_SRC: &str = r#"
#[derive(Clone, Copy)]
struct Wrap(());
const A: &[Wrap; 4321] = &[Wrap(()); 4321];
fn main() {
    println!("{}", A.len());
}
"#;

#[test]
fn run_mode_byref_zst_field_array_past_bound() {
    run_ok("zst-field-array", ZST_FIELD_ARRAY_SRC, "4321\n");
}

const VALUE_POSITION_SRC: &str = r#"
const A: [(); 5000] = [(); 5000];
fn main() {
    println!("{}", A.len());
}
"#;

#[test]
fn run_mode_value_position_zst_array_past_bound() {
    run_ok("value-position", VALUE_POSITION_SRC, "5000\n");
}
