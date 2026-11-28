//! Cell/RefCell receiver-denomination regression tests
//! (Issue #513): the Cell/RefCell operations rejected the
//! denomination the conversion actually produced for their
//! receiver -- a `Cell` place denominated as the bare held value
//! (the storage a `Cell::from_mut` / `as_slice_of_cells`
//! reinterpretation denotes), and a `RefCell` receiver that
//! arrived through an unsizing coercion as a dyn reference over
//! a smart-pointer handle. Each fixture runs end to end through
//! `sparagmos-driver run` and pins the exact stdout; every one
//! of them fails on the unmodified driver with a type mismatch
//! naming the receiver (`Cell::set`: expected cell aggregate,
//! found signed integer; `RefCell::borrow`: expected reference
//! to RefCell, found dyn reference / ADT value).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-cell-refcell-denom-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn assert_run(tag: &str, src: &str, expected_stdout: &str) {
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

/// `Cell::set` on an element of a `&[Cell<i32>; 3]` produced by
/// `Cell::from_mut` + `as_array_of_cells`: the element place is
/// denominated as the bare `i32` the array storage holds, not as
/// a cell aggregate.
#[test]
fn cell_set_on_array_of_cells_from_mut() {
    assert_run(
        "array-of-cells-set",
        r#"
use std::cell::Cell;

fn main() {
    let mut array: [i32; 3] = [1, 2, 3];
    let cell_array: &Cell<[i32; 3]> = Cell::from_mut(&mut array);
    let array_cell: &[Cell<i32>; 3] = cell_array.as_array_of_cells();

    array_cell[0].set(99);
    println!("{:?}", array);
}
"#,
        "[99, 2, 3]\n",
    );
}

/// `Cell::get` / `Cell::replace` through the same reinterpreted
/// denomination (a slice of cells over a `&mut [i32]`).
#[test]
fn cell_get_replace_on_slice_of_cells() {
    assert_run(
        "slice-of-cells",
        r#"
use std::cell::Cell;

fn main() {
    let slice: &mut [i32] = &mut [1, 2, 3];
    let cell_slice: &Cell<[i32]> = Cell::from_mut(slice);
    let slice_cell: &[Cell<i32>] = cell_slice.as_slice_of_cells();

    println!("{}", slice_cell.len());
    println!("{}", slice_cell[1].get());
    println!("{}", slice_cell[1].replace(20));
    println!("{}", slice_cell[1].get());
}
"#,
        "3\n2\n2\n20\n",
    );
}

/// `RefCell::borrow` on an `Rc<RefCell<dyn Baz>>`: the receiver
/// reaches the builtin as a dyn reference over the Rc handle,
/// and the guard must keep the dyn denomination so the virtual
/// call through it dispatches.
#[test]
fn refcell_borrow_through_rc_dyn_coercion() {
    assert_run(
        "rc-refcell-dyn-borrow",
        r#"
use std::cell::RefCell;
use std::rc::Rc;

trait Baz {
    fn get(&self) -> i32;
}

impl Baz for i32 {
    fn get(&self) -> i32 {
        *self
    }
}

fn main() {
    let a: Rc<RefCell<i32>> = Rc::new(RefCell::new(42));
    let b: Rc<RefCell<dyn Baz>> = a.clone();
    println!("{}", b.borrow().get());
}
"#,
        "42\n",
    );
}

/// `RefCell::borrow_mut` through the same coerced receiver:
/// the write through the guard lands in the shared cell.
#[test]
fn refcell_borrow_mut_through_rc_dyn_coercion() {
    assert_run(
        "rc-refcell-dyn-borrow-mut",
        r#"
use std::cell::RefCell;
use std::rc::Rc;

trait Baz {
    fn get(&self) -> i32;
    fn set(&mut self, v: i32);
}

impl Baz for i32 {
    fn get(&self) -> i32 {
        *self
    }
    fn set(&mut self, v: i32) {
        *self = v;
    }
}

fn main() {
    let a: Rc<RefCell<i32>> = Rc::new(RefCell::new(1));
    let b: Rc<RefCell<dyn Baz>> = a.clone();
    b.borrow_mut().set(7);
    println!("{}", b.borrow().get());
    println!("{}", a.borrow());
}
"#,
        "7\n7\n",
    );
}
