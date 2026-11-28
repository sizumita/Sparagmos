//! Run-mode tests for assignment stores whose value's
//! denomination comes from a constant / aggregate /
//! partially-initialised source (Issue #513): a place the
//! MIR denominates as a pointer holds the payload of the
//! niche-encoded aggregate stored there (and the null
//! pointer for its payload-less variant), and a byte write
//! into a union place is judged by the union's storage
//! size even when the program records several instances of
//! the union's definition, so the bare definition name the
//! IR types carry resolves through the instances that agree.
//! On the base tree every program here stops with a type
//! mismatch in assignment.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-acd-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn run_ok(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver(&src_path);
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

fn run_err(tag: &str, src: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver(&src_path);
    assert_ne!(
        out.status.code(),
        Some(0),
        "case {tag} must not run to completion; stdout={}",
        String::from_utf8_lossy(&out.stdout)
    );
}


#[test]
fn run_mode_niche_option_none_read_as_pointer() {
    // tests/ui/consts/issue-21721.rs: a static `Option` over
    // a reference, read through a `*const *const u8`, is the
    // null pointer -- the aggregate the storage holds is the
    // niche encoding of the pointer the place denotes.
    run_ok(
        "niche-none-static",
        r#"
fn main() {
    static NONE: Option<((), &'static u8)> = None;
    let ptr = unsafe { *(&NONE as *const _ as *const *const u8) };
    assert!(ptr.is_null());
    println!("ok");
}
"#,
        "ok\n",
    );
}

#[test]
fn run_mode_niche_option_some_read_as_pointer() {
    // The payload variant of the same read: the pointer the
    // place denotes is the reference the aggregate's payload
    // tuple holds beside its unit field.
    run_ok(
        "niche-some-static",
        r#"
static X: u8 = 5;
fn main() {
    static SOME: Option<((), &'static u8)> = Some(((), &X));
    let ptr = unsafe { *(&SOME as *const _ as *const *const u8) };
    assert!(!ptr.is_null());
    assert_eq!(unsafe { *ptr }, 5);
    println!("ok");
}
"#,
        "ok\n",
    );
}

#[test]
fn run_mode_niche_option_local_read_as_pointer() {
    run_ok(
        "niche-none-local",
        r#"
fn main() {
    let none: Option<((), &u8)> = None;
    let ptr = unsafe { *(&none as *const _ as *const *const u8) };
    assert!(ptr.is_null());
    let x = 9u8;
    let some: Option<((), &u8)> = Some(((), &x));
    let ptr = unsafe { *(&some as *const _ as *const *const u8) };
    assert_eq!(unsafe { *ptr }, 9);
    println!("ok");
}
"#,
        "ok\n",
    );
}

#[test]
fn run_mode_partial_init_union_two_instances() {
    // tests/ui/consts/load-preserves-partial-init.rs: one
    // program instantiates the union at two payload types, so
    // the bare definition name the IR types carry denotes two
    // recorded instances; the byte write into a bag's storage
    // is still the union's own (both instances record the
    // same layout, and the field the write's tag names first
    // is the zero-sized one every instance agrees on).
    run_ok(
        "bag-two-instances",
        r#"
pub union BagOfBits<T: Copy> {
    uninit: (),
    _storage: T,
}

pub const fn make_1u8_bag<T: Copy>() -> BagOfBits<T> {
    assert!(core::mem::size_of::<T>() >= 1);
    let mut bag = BagOfBits { uninit: () };
    unsafe { (&mut bag as *mut _ as *mut u8).write(1); };
    bag
}

pub fn check_bag<T: Copy>(bag: &BagOfBits<T>) {
    let val = unsafe { (bag as *const _ as *const u8).read() };
    assert_eq!(val, 1);
}

fn main() {
    check_bag(&make_1u8_bag::<[usize; 1]>());
    check_bag(&make_1u8_bag::<usize>());
    const CONST_ARRAY_BAG: BagOfBits<[usize; 1]> = make_1u8_bag();
    check_bag(&CONST_ARRAY_BAG);
    const CONST_USIZE_BAG: BagOfBits<usize> = make_1u8_bag();
    check_bag(&CONST_USIZE_BAG);
    println!("ok");
}
"#,
        "ok\n",
    );
}
