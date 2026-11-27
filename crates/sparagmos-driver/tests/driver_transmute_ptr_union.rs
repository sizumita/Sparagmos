//! Run-mode tests for raw pointer <-> union transmutes
//! (Issue #394): a raw pointer transmuted into a union, and
//! union fields read back through the storage the pointer
//! field wrote. On the base tree the optimized fixtures
//! fail with `unsupported rvalue: Cast Transmute`: at higher
//! opt levels the pointer reaches the transmute as a pointer
//! constant with no place type (so no shape is selected for
//! either side), and a union with several scalar-shaped
//! fields gets no scalar shape at all (the multi-field shape
//! reads the per-field layout offsets, which unions have no
//! entry for). The transmute path now derives a union's
//! shape from its field types and the ADT's total size, and
//! a pointer constant carries the thin-pointer shape.

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
        "sparagmos-driver-tpu-{}-{tag}",
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

/// The failing record's crossing (tests/ui/union/
/// union-nonzero.rs): a null raw pointer transmuted into
/// `U2<&u8, ()>`, then held in an `Option` whose discriminant
/// must read as `Some` -- unions take no niche from the
/// pointer field.
#[test]
fn null_ptr_constant_into_union_with_zst_field() {
    let src = r#"use std::mem::transmute;
#[derive(Clone, Copy)]
union U2<A: Copy, B: Copy> { a: A, b: B }
fn main() {
    let zeroed: U2<&u8, ()> = unsafe { transmute(std::ptr::null::<u8>()) };
    if let None = Some(zeroed) {
        panic!("union took a niche from its pointer field");
    }
    println!("some");
}
"#;
    run_ok("null-zst-opt", src, true, "some\n");
    run_ok("null-zst", src, false, "some\n");
}

/// A null pointer constant into a union of two pointer
/// fields: no field-layout entry exists for a union, so the
/// shape comes from the field types and the total size.
#[test]
fn null_ptr_constant_into_two_pointer_field_union() {
    let src = r#"use std::mem::transmute;
#[derive(Clone, Copy)]
union U2<A: Copy, B: Copy> { a: A, b: B }
fn main() {
    let u: U2<&u8, &u8> = unsafe { transmute(std::ptr::null::<u8>()) };
    let a = unsafe { u.a };
    let b = unsafe { u.b };
    println!("{} {}", (a as *const u8).is_null(), (b as *const u8).is_null());
}
"#;
    run_ok("null-two-ptr-opt", src, true, "true true\n");
    run_ok("null-two-ptr", src, false, "true true\n");
}

/// Storage written through the pointer field, read back
/// through the integer field of the same union.
#[test]
fn ptr_into_union_read_through_integer_field() {
    let src = r#"use std::mem::transmute;
#[derive(Clone, Copy)]
union U { p: *const u8, n: usize }
fn main() {
    let u: U = unsafe { transmute(0x2ausize as *const u8) };
    println!("{}", unsafe { u.n });
}
"#;
    run_ok("ptr-int-opt", src, true, "42\n");
    run_ok("ptr-int", src, false, "42\n");
}

/// The payload field need not be field 0: a zero-sized field
/// ahead of the pointer field is a fill.
#[test]
fn ptr_into_union_with_leading_zst_field() {
    let src = r#"use std::mem::transmute;
#[derive(Clone, Copy)]
union V { z: (), p: *const u8 }
fn main() {
    let v: V = unsafe { transmute(std::ptr::null::<u8>()) };
    println!("{}", unsafe { v.p.is_null() });
}
"#;
    run_ok("ptr-zst-first-opt", src, true, "true\n");
    run_ok("ptr-zst-first", src, false, "true\n");
}

/// The reverse crossing over a multi-field union: the
/// pointer the union was built through comes back out.
#[test]
fn union_into_raw_ptr() {
    let src = r#"use std::mem::transmute;
#[derive(Clone, Copy)]
union U2<A: Copy, B: Copy> { a: A, b: B }
fn main() {
    let x = 7u8;
    let w: U2<&u8, &u8> = U2 { a: &x };
    let p: *const u8 = unsafe { transmute(w) };
    println!("{}", p == (&x as *const u8));
}
"#;
    run_ok("union-ptr-opt", src, true, "true\n");
    run_ok("union-ptr", src, false, "true\n");
}

/// A union built through its integer field, transmuted out
/// to a raw pointer: the union value holds the integer
/// field and leaves the pointer slot uninit, so the crossing
/// reads the live field's bits as the pointer's address --
/// the same reinterpretation a union field read performs.
#[test]
fn union_built_through_int_field_into_raw_ptr() {
    let src = r#"use std::mem::transmute;
#[derive(Clone, Copy)]
union U { p: *const u8, n: usize }
fn main() {
    let u = U { n: 123 };
    let p: *mut u8 = unsafe { transmute(u) };
    println!("{}", p as usize);
}
"#;
    run_ok("union-int-ptr-opt", src, true, "123\n");
    run_ok("union-int-ptr", src, false, "123\n");
}

/// A union built through its float field, transmuted out to
/// a raw pointer: the pointer slot is uninit in the union
/// value, and the crossing reinterprets the float field's
/// bit pattern as the address.
#[test]
fn union_built_through_float_field_into_raw_ptr() {
    let src = r#"use std::mem::transmute;
#[derive(Clone, Copy)]
union U { p: *const u8, f: f64 }
fn main() {
    let u = U { f: 1.0 };
    let p: *const u8 = unsafe { transmute(u) };
    println!("{}", p as usize == 0x3ff0000000000000usize);
}
"#;
    run_ok("union-float-ptr-opt", src, true, "true\n");
    run_ok("union-float-ptr", src, false, "true\n");
}
