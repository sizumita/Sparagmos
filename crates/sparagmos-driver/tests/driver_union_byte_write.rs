//! Run-mode tests for byte writes and reads through raw
//! pointers into union-typed places (Issue #513): a
//! `*mut u8` aimed at a union's storage denotes the union's
//! bytes, not one of its fields, so the write composes the
//! union's storage image and the value takes its byte form;
//! later field reads decode from those bytes and later byte
//! reads see them back. On the base tree every program here
//! stops: the byte view had no union arm, so a raw-pointer
//! byte write into a union place declined through every
//! write path and fell to a whole-value type mismatch.

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

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-ubw-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
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
fn run_mode_union_byte_write_at_base_read_back_through_fields() {
    run_ok(
        "base",
        r#"
union U {
    a: u8,
    b: usize,
}

fn main() {
    let mut u = U { b: 0 };
    unsafe {
        (&mut u as *mut U as *mut u8).write(1);
        println!("{}", u.a);
        println!("{}", u.b);
    }
}
"#,
        "1\n1\n",
    );
}

#[test]
fn run_mode_union_byte_write_displaced_read_back_through_wide_field() {
    run_ok(
        "displaced",
        r#"
union U {
    a: u8,
    storage: usize,
}

fn main() {
    let mut u = U { storage: 0 };
    unsafe {
        (&mut u as *mut U as *mut u8).add(3).write(0xAB);
        println!("{:x}", u.storage);
    }
}
"#,
        "ab000000\n",
    );
}

#[test]
fn run_mode_union_byte_write_preserves_unwritten_bytes() {
    run_ok(
        "preserve",
        r#"
union U {
    a: u8,
    storage: usize,
}

fn main() {
    let mut u = U { storage: 0x0102_0304_0506_0708 };
    unsafe {
        (&mut u as *mut U as *mut u8).write(0xFF);
        println!("{:x}", u.storage);
        // The written byte reads back through the pointer too.
        let p = &u as *const U as *const u8;
        println!("{}", p.read());
        println!("{}", p.add(7).read());
    }
}
"#,
        "1020304050607ff\n255\n1\n",
    );
}

#[test]
fn run_mode_union_byte_reads_of_field_written_union() {
    run_ok(
        "reads",
        r#"
pub union Bag {
    uninit: (),
    _storage: usize,
}

fn make_1u8_bag() -> Bag {
    let mut bag = Bag { uninit: () };
    unsafe { (&mut bag as *mut Bag as *mut u8).write(1); }
    bag
}

fn main() {
    let bag = make_1u8_bag();
    let val = unsafe { (&bag as *const Bag as *const u8).read() };
    println!("{val}");
    println!("{}", unsafe { bag._storage });
}
"#,
        "1\n1\n",
    );
}

#[test]
fn run_mode_union_byte_write_nested_in_struct() {
    run_ok(
        "nested",
        r#"
union U {
    a: u8,
    b: usize,
}

struct W {
    u: U,
    tail: u8,
}

fn main() {
    let mut w = W { u: U { b: 0 }, tail: 3 };
    unsafe {
        (&mut w.u as *mut U as *mut u8).write(5);
        println!("{}", w.u.b);
        println!("{}", w.tail);
    }
}
"#,
        "5\n3\n",
    );
}

#[test]
fn run_mode_union_byte_write_outside_storage_is_an_error() {
    run_err(
        "outside",
        r#"
union U {
    a: u8,
    b: usize,
}

fn main() {
    let mut u = U { b: 0 };
    unsafe {
        (&mut u as *mut U as *mut u8).add(8).write(1);
    }
    println!("wrote");
}
"#,
    );
}

#[test]
fn run_mode_struct_byte_write_unchanged() {
    run_ok(
        "struct-control",
        r#"
struct S {
    a: usize,
    b: u8,
}

fn main() {
    let mut s = S { a: 0, b: 0 };
    unsafe {
        let p = &mut s as *mut S as *mut u8;
        p.write(7);
        p.add(3).write(0xAB);
        println!("{:x}", s.a);
        println!("{}", s.b);
    }
}
"#,
        "ab000007\n0\n",
    );
}
