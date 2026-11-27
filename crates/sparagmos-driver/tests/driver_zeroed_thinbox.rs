//! Run-mode tests for three value-model mechanisms (Issue #513):
//! `mem::zeroed` / `MaybeUninit::zeroed` for enum types (the
//! zero bit pattern denotes the variant the recorded layout
//! assigns it, niche encodings included), `ThinBox`
//! dereference (the metadata header sits directly before the
//! value in the same allocation), and RawPtr aggregate
//! construction with dyn metadata (the shape
//! `std::any::try_as_dyn` builds). On the base tree every
//! program here fails: zeroed enums are "not
//! runtime-supported", ThinBox deref is "not
//! runtime-supported", and the RawPtr aggregate is rejected
//! in conversion.

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
        "sparagmos-driver-zt-{}-{tag}",
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

#[test]
fn run_mode_zeroed_enum_direct_tag() {
    run_ok(
        "zeroed-direct",
        r#"
#[derive(Debug, PartialEq)]
enum E {
    A(u32),
    B { x: u8 },
    C,
}

fn main() {
    let e: E = unsafe { std::mem::zeroed() };
    assert_eq!(e, E::A(0));
    println!("{e:?}");
}
"#,
        "A(0)\n",
    );
}

#[test]
fn run_mode_zeroed_enum_niche() {
    run_ok(
        "zeroed-niche",
        r#"
use std::num::NonZeroU32;

enum N {
    X(NonZeroU32),
    Y,
}

fn main() {
    // The zero pattern is the niche value of the `NonZeroU32`
    // field, which denotes the niche variant.
    let n: N = unsafe { std::mem::zeroed() };
    match n {
        N::X(_) => println!("X"),
        N::Y => println!("Y"),
    }
    let o: Option<NonZeroU32> = unsafe { std::mem::zeroed() };
    match o {
        Some(_) => println!("Some"),
        None => println!("None"),
    }
}
"#,
        "Y\nNone\n",
    );
}

#[test]
fn run_mode_maybe_uninit_zeroed_enum() {
    run_ok(
        "zeroed-maybeuninit",
        r#"
#[derive(Debug, PartialEq)]
enum E {
    A(u32),
    B,
}

fn main() {
    let e: E = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
    assert_eq!(e, E::A(0));
    println!("{e:?}");
}
"#,
        "A(0)\n",
    );
}

#[test]
fn run_mode_try_as_dyn() {
    run_ok(
        "try-as-dyn",
        r#"
#![feature(try_as_dyn)]

use std::fmt::{Debug, Write as _};

#[derive(Debug)]
struct A(u32);

fn main() {
    let a = A(7);
    let d = std::any::try_as_dyn::<A, dyn Debug>(&a).expect("A is Debug");
    println!("{d:?}");
    let mut s = String::from("x");
    let w = std::any::try_as_dyn_mut::<String, dyn std::fmt::Write>(&mut s)
        .expect("String is Write");
    let mut w = w;
    write!(w, "y").unwrap();
    println!("{s}");
}
"#,
        "A(7)\nxy\n",
    );
}

#[test]
fn run_mode_thin_box_deref() {
    run_ok(
        "thinbox",
        r#"
#![feature(thin_box)]

use std::boxed::ThinBox;
use std::error::Error;
use std::fmt;

#[derive(Debug)]
struct Foo;
impl fmt::Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "foo!")
    }
}
impl Error for Foo {}

fn main() {
    let b = ThinBox::<dyn Error>::new_unsize(Foo);
    println!("{}", b.to_string());
    let n = ThinBox::new(42i32);
    println!("{}", *n);
    let s = ThinBox::<[i32]>::new_unsize([1, 2, 3]);
    println!("{:?}", &*s);
}
"#,
        "foo!\n42\n[1, 2, 3]\n",
    );
}

#[test]
fn run_mode_thin_box_sized_zst() {
    run_ok(
        "thinbox-zst",
        r#"
#![feature(thin_box)]

use std::boxed::ThinBox;
use std::fmt;

#[derive(Debug)]
struct Foo;
impl fmt::Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "zst!")
    }
}

fn main() {
    // A sized zero-sized pointee: the allocation is empty
    // and std serves the box a dangling pointer; writing,
    // reading and dropping through it must still work.
    let mut b = ThinBox::new(Foo);
    println!("{}", b.to_string());
    *b = Foo;
    println!("done");
}
"#,
        "zst!\ndone\n",
    );
}
