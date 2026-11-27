//! Run-mode tests for whole-value stores into union-typed
//! places nested inside other aggregates (Issue #513): a
//! `MaybeUninit` field reached through an `UnsafeCell` inside a
//! struct — the shape a `OnceLock` static's value cell has — is
//! written by a whole-value shaped store (the assignment inside
//! `MaybeUninit::write` through the raw pointer the cell hands
//! out) and read back through the payload descent
//! (`assume_init_ref`, a pointee deref). The store's shape covers
//! the union position exactly, so the store is a whole-value
//! store of that position and must keep the value's own form;
//! splicing its bytes through the byte view instead left the
//! slot in its byte form (`Value::UnionBytes`), which the
//! payload-descent reads cannot denominate, and every program
//! here stopped with a field-projection type mismatch on the
//! base. The byte view's exact-cover decline keyed only on the
//! position type's size, and the type walk loses the position
//! type at a union field (union layouts are not recorded in the
//! field layout tables), so the decline never fired for these
//! positions; it now also keys on the position size the layout
//! walk prices from the field rows.

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
        "sparagmos-driver-uws-{}-{tag}",
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

const STATIC_CELL_SRC: &str = r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
struct Cell(UnsafeCell<MaybeUninit<u32>>);
unsafe impl Sync for Cell {}
static CELL: Cell = Cell(UnsafeCell::new(MaybeUninit::uninit()));
fn main() {
    unsafe {
        (*CELL.0.get()).write(42);
        println!("{}", (*CELL.0.get()).assume_init_ref());
        println!("{}", (*CELL.0.get()).assume_init_ref());
    }
}
"#;

#[test]
fn run_mode_union_whole_store_static_nested() {
    run_ok("static-nested", STATIC_CELL_SRC, "42\n42\n");
}

const LOCAL_CELL_SRC: &str = r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
fn main() {
    let cell = UnsafeCell::new(MaybeUninit::<u32>::uninit());
    unsafe {
        (*cell.get()).write(42);
        println!("{}", (*cell.get()).assume_init_ref());
    }
}
"#;

#[test]
fn run_mode_union_whole_store_local_nested() {
    run_ok("local-nested", LOCAL_CELL_SRC, "42\n");
}

const NEIGHBOUR_SRC: &str = r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
struct Pair {
    tag: u32,
    cell: UnsafeCell<MaybeUninit<u32>>,
    tail: u16,
}
unsafe impl Sync for Pair {}
static PAIR: Pair = Pair { tag: 7, cell: UnsafeCell::new(MaybeUninit::uninit()), tail: 9 };
fn main() {
    unsafe {
        (*PAIR.cell.get()).write(42);
        println!("tag={} val={} tail={}", PAIR.tag, (*PAIR.cell.get()).assume_init_ref(), PAIR.tail);
    }
}
"#;

#[test]
fn run_mode_union_whole_store_neighbour_fields() {
    run_ok("neighbour", NEIGHBOUR_SRC, "tag=7 val=42 tail=9\n");
}

const DEREF_READ_SRC: &str = r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
struct Cell(UnsafeCell<MaybeUninit<u32>>);
unsafe impl Sync for Cell {}
static CELL: Cell = Cell(UnsafeCell::new(MaybeUninit::uninit()));
fn main() {
    unsafe {
        (*CELL.0.get()).write(42);
        let v = *(CELL.0.get() as *const u32);
        println!("{}", v);
    }
}
"#;

#[test]
fn run_mode_union_whole_store_deref_read() {
    run_ok("deref-read", DEREF_READ_SRC, "42\n");
}

const WIDE_SRC: &str = r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
struct Cell(UnsafeCell<MaybeUninit<u64>>);
unsafe impl Sync for Cell {}
static CELL: Cell = Cell(UnsafeCell::new(MaybeUninit::uninit()));
fn main() {
    unsafe {
        (*CELL.0.get()).write(0x1122_3344_5566_7788u64);
        println!("{:x}", (*CELL.0.get()).assume_init_ref());
    }
}
"#;

#[test]
fn run_mode_union_whole_store_wide_payload() {
    run_ok("wide", WIDE_SRC, "1122334455667788\n");
}

const OVERWRITE_SRC: &str = r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
struct Cell(UnsafeCell<MaybeUninit<u32>>);
unsafe impl Sync for Cell {}
static CELL: Cell = Cell(UnsafeCell::new(MaybeUninit::uninit()));
fn main() {
    unsafe {
        (*CELL.0.get()).write(42);
        (*CELL.0.get()).write(7);
        println!("{}", (*CELL.0.get()).assume_init_ref());
    }
}
"#;

#[test]
fn run_mode_union_whole_store_overwrite() {
    run_ok("overwrite", OVERWRITE_SRC, "7\n");
}

const DEEP_SRC: &str = r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
struct Inner { cell: UnsafeCell<MaybeUninit<u32>> }
struct Outer { inner: Inner }
unsafe impl Sync for Outer {}
static OUTER: Outer = Outer { inner: Inner { cell: UnsafeCell::new(MaybeUninit::uninit()) } };
fn main() {
    unsafe {
        (*OUTER.inner.cell.get()).write(42);
        println!("{}", (*OUTER.inner.cell.get()).assume_init_ref());
    }
}
"#;

#[test]
fn run_mode_union_whole_store_deep_nesting() {
    run_ok("deep", DEEP_SRC, "42\n");
}
