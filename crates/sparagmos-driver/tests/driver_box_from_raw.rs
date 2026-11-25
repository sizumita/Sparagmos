//! `Box::from_raw` round-trip regression tests (Lane MP, #254):
//! a box reconstructed from a raw pointer by the converted
//! `from_raw` / `from_raw_in` body is the real `Unique` /
//! `NonNull` wrapper layout, not the `Box::new` builtin's
//! one-field aggregate. Reading, writing, and dropping through
//! such a box must behave exactly as through a builtin box.

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
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-box-from-raw-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn assert_runs(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let path = fixture(&dir, "main.rs", src);
    let out = run_driver(&path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "round-trip program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "the guest's stdout must match the native output"
    );
}

const READ_SRC: &str = r#"
fn main() {
    let b = Box::new(42u32);
    let raw = Box::into_raw(b);
    let b2 = unsafe { Box::from_raw(raw) };
    println!("v={}", *b2);
}
"#;

const WRITE_BACK_SRC: &str = r#"
fn main() {
    let b = Box::new(1u32);
    let raw = Box::into_raw(b);
    let mut b2 = unsafe { Box::from_raw(raw) };
    *b2 = 99;
    let via_raw = unsafe { *raw };
    println!("via_raw={}", via_raw);
    drop(b2);
    println!("end");
}
"#;

const DROP_SRC: &str = r#"
struct D(u32);
impl Drop for D {
    fn drop(&mut self) { println!("drop {}", self.0); }
}
fn main() {
    let raw = Box::into_raw(Box::new(D(5)));
    let b2 = unsafe { Box::from_raw(raw) };
    println!("val {}", b2.0);
    drop(b2);
    println!("end");
}
"#;

const NESTED_SRC: &str = r#"
struct D(u32);
impl Drop for D {
    fn drop(&mut self) { println!("drop {}", self.0); }
}
struct S { b: Box<D>, x: u32 }
fn main() {
    let raw = Box::into_raw(Box::new(D(8)));
    let b2 = unsafe { Box::from_raw(raw) };
    let s = S { b: b2, x: 1 };
    println!("x={} v={}", s.x, s.b.0);
    drop(s);
    println!("end");
}
"#;

const DOUBLE_SRC: &str = r#"
fn main() {
    let b = Box::new(3u32);
    let r1 = Box::into_raw(b);
    let b2 = unsafe { Box::from_raw(r1) };
    let r2 = Box::into_raw(b2);
    let b3 = unsafe { Box::from_raw(r2) };
    println!("v={}", *b3);
    drop(b3);
    println!("end");
}
"#;

const DYN_SRC: &str = r#"
trait T { fn get(&self) -> u32; }
struct V(u32);
impl T for V { fn get(&self) -> u32 { self.0 } }
impl Drop for V { fn drop(&mut self) { println!("drop V {}", self.0); } }
fn main() {
    let b: Box<dyn T> = Box::new(V(6));
    let raw = Box::into_raw(b);
    let b2 = unsafe { Box::from_raw(raw) };
    println!("v={}", b2.get());
    drop(b2);
    println!("end");
}
"#;

const SLICE_SRC: &str = r#"
struct D(u32);
impl Drop for D {
    fn drop(&mut self) { println!("drop {}", self.0); }
}
fn main() {
    let v: Vec<D> = vec![D(1), D(2)];
    let bs: Box<[D]> = v.into_boxed_slice();
    let raw = Box::into_raw(bs);
    let b2 = unsafe { Box::from_raw(raw) };
    println!("len={} first={}", b2.len(), b2[0].0);
    drop(b2);
    println!("end");
}
"#;

const CUSTOM_ALLOC_SRC: &str = r#"
use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;

#[derive(Clone, Copy, Debug)]
struct CountAlloc(u32);
unsafe impl Allocator for CountAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, layout) }
    }
}

struct D(u32);
impl Drop for D {
    fn drop(&mut self) { println!("drop {}", self.0); }
}

fn main() {
    let raw = Box::into_raw(Box::new(D(9)));
    let b2 = unsafe { Box::from_raw_in(raw, CountAlloc(7)) };
    println!("v={} alloc={}", b2.0, Box::allocator(&b2).0);
    drop(b2);
    println!("end");
}
"#;

#[test]
fn from_raw_read_round_trip() {
    assert_runs("read", READ_SRC, "v=42\n");
}

#[test]
fn from_raw_write_reaches_the_allocation() {
    assert_runs("write-back", WRITE_BACK_SRC, "via_raw=99\nend\n");
}

#[test]
fn from_raw_box_drops_its_pointee() {
    assert_runs("drop", DROP_SRC, "val 5\ndrop 5\nend\n");
}

#[test]
fn from_raw_box_nested_in_a_struct_drops() {
    assert_runs("nested", NESTED_SRC, "x=1 v=8\ndrop 8\nend\n");
}

#[test]
fn from_raw_double_round_trip() {
    assert_runs("double", DOUBLE_SRC, "v=3\nend\n");
}

#[test]
fn from_raw_dyn_box_drops_its_pointee() {
    assert_runs("dyn", DYN_SRC, "v=6\ndrop V 6\nend\n");
}

#[test]
fn from_raw_slice_box_drops_its_elements() {
    assert_runs("slice", SLICE_SRC, "len=2 first=1\ndrop 1\ndrop 2\nend\n");
}

#[test]
fn from_raw_in_with_a_stateful_allocator() {
    assert_runs(
        "custom-alloc",
        CUSTOM_ALLOC_SRC,
        "v=9 alloc=7\ndrop 9\nend\n",
    );
}
