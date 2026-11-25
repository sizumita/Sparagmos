//! Run-mode tests for `Vec::as_ptr` / `Vec::as_mut_ptr` (Lane NM,
//! #277): reading a vector's buffer pointer walks the declared
//! `Vec -> RawVec -> RawVecInner -> Unique -> NonNull` field chain
//! over the element-list value the model holds a vector as. The
//! walk is answered by type-tracked place resolution (the buffer
//! chain denotes the first element position); on the base tree
//! every program here fails with `type mismatch in assignment:
//! expected Adt("std::ptr::NonNull"), found signed integer` (or,
//! for the empty vector, an uninitialized read of the chain).

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
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("sparagmos-driver-asptr-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn run_ok(tag: &str, src: &str) -> String {
    let dir = temp_dir(tag);
    let path = fixture(&dir, "prog.rs", src);
    let out = run_driver(&path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "program must exit 0; stderr={stderr}"
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn run_mode_vec_as_ptr_reads_elements() {
    let stdout = run_ok(
        "reads",
        r#"
fn main() {
    let v = vec![10, 20, 30];
    let p = v.as_ptr();
    unsafe {
        println!("{} {}", *p, *p.add(1));
    }
    println!("len {}", v.len());
}
"#,
    );
    assert_eq!(stdout, "10 20\nlen 3\n");
}

#[test]
fn run_mode_empty_vec_as_ptr() {
    let stdout = run_ok(
        "empty",
        r#"
fn main() {
    let v: Vec<i32> = Vec::new();
    let p = v.as_ptr();
    let _ = p;
    println!("done");
}
"#,
    );
    assert_eq!(stdout, "done\n");
}

#[test]
fn run_mode_vec_as_mut_ptr_write_and_from_raw_parts() {
    let stdout = run_ok(
        "mutptr",
        r#"
fn main() {
    let mut v = vec![1, 2, 3];
    let p = v.as_mut_ptr();
    unsafe { *p.add(1) = 99; }
    assert_eq!(v[1], 99);
    let q = v.as_ptr();
    unsafe { assert_eq!(*q.add(2), 3); }
    assert_eq!(q, v.as_ptr());
    let s = unsafe { std::slice::from_raw_parts(q, 3) };
    assert_eq!(s, &[1, 99, 3]);
    println!("ok");
}
"#,
    );
    assert_eq!(stdout, "ok\n");
}

#[test]
fn run_mode_vec_as_ptr_struct_field_box_and_string_elements() {
    let stdout = run_ok(
        "shapes",
        r#"
struct Holder { v: Vec<String> }
fn main() {
    let h = Holder { v: vec!["aa".to_string(), "bb".to_string()] };
    let p = h.v.as_ptr();
    unsafe { assert_eq!((*p).as_str(), "aa"); assert_eq!((*p.add(1)).as_str(), "bb"); }
    let b = Box::new(vec![7, 8]);
    let bp = b.as_ptr();
    unsafe { assert_eq!(*bp.add(1), 8); }
    println!("ok");
}
"#,
    );
    assert_eq!(stdout, "ok\n");
}

#[test]
fn run_mode_vec_as_ptr_copy_nested_and_zst() {
    let stdout = run_ok(
        "copy",
        r#"
fn main() {
    let src = vec![1, 2, 3];
    let mut dst = vec![0; 3];
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), 3); }
    assert_eq!(dst, vec![1, 2, 3]);
    let z: Vec<()> = vec![(), ()];
    assert_eq!(z.as_ptr(), z.as_ptr());
    let nested = vec![vec![9, 8], vec![7]];
    let ip = nested[1].as_ptr();
    unsafe { assert_eq!(*ip, 7); }
    println!("ok");
}
"#,
    );
    assert_eq!(stdout, "ok\n");
}

#[test]
fn run_mode_vec_as_ptr_after_huge_set_len() {
    let stdout = run_ok(
        "sparse",
        r#"
fn main() {
    let mut v = Vec::new();
    v.push(3);
    unsafe { v.set_len(100_000); }
    let p = v.as_ptr();
    unsafe { assert_eq!(*p, 3); }
    println!("ok");
}
"#,
    );
    assert_eq!(stdout, "ok\n");
}

#[test]
fn run_mode_vec_as_ptr_drop_count_unchanged() {
    let stdout = run_ok(
        "drops",
        r#"
use std::sync::atomic::{AtomicUsize, Ordering};
static DROPS: AtomicUsize = AtomicUsize::new(0);
struct D(i32);
impl Drop for D { fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); } }
fn main() {
    {
        let v = vec![D(1), D(2), D(3)];
        let p = v.as_ptr();
        unsafe { assert_eq!((*p).0, 1); assert_eq!((*p.add(2)).0, 3); }
    }
    assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    println!("ok");
}
"#,
    );
    assert_eq!(stdout, "ok\n");
}
