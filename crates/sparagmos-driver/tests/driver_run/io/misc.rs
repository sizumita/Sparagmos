// A shared index write through a `Cell` element: the test
// lives with the io tests but exercises place semantics.

const VEC_CELL_INDEX_SRC: &str = r#"use std::cell::Cell;
fn main() {
    let mut v = vec![Cell::new(1u32), Cell::new(2u32)];
    v[0].set(5);
    v[1].set(v[0].get() + 1);
    assert_eq!(v[0].get(), 5);
    assert_eq!(v[1].get(), 6);
    println!("vec cell ok");
    // The vector is dropped at the end of `main`: the `Drop`
    // terminator for the cell element glue runs as well.
}
"#;

#[test]
fn run_mode_shared_index_write_through_cell() {
    // `v[i].set(..)` mutates through the *shared* element
    // reference `Index::index` returns: the reference must name
    // the vector's storage (a live place), not a snapshot of the
    // element, or the `Cell` write has no place to land.
    let dir = temp_dir("vec-cell-index");
    let src = fixture(&dir, "vec_cell.rs", VEC_CELL_INDEX_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "vec cell ok\n");
}


const DYN_FILE_READ_SRC: &str = r#"
use std::io::Read;
fn main() {
    let path = std::env::temp_dir().join("sparagmos-dyn-file-read.txt");
    std::fs::write(&path, b"dynread").unwrap();
    let f = std::fs::File::open(&path).unwrap();
    let mut r: Box<dyn Read> = Box::new(f);
    let mut v = Vec::new();
    let n = r.read_to_end(&mut v).unwrap();
    assert_eq!(n, 7);
    assert_eq!(v, b"dynread");
    let f = std::fs::File::open(&path).unwrap();
    let mut r: Box<dyn Read> = Box::new(f);
    let mut s = String::new();
    let m = r.read_to_string(&mut s).unwrap();
    assert_eq!(m, 7);
    assert_eq!(s, "dynread");
    println!("dyn file read ok");
}
"#;

#[test]
fn run_mode_dyn_read_on_file() {
    // `Read::read_to_end` / `read_to_string` on a
    // `Box<dyn Read>` over a `File`: the default trait bodies
    // run as converted code and their leaf `read` dispatches
    // through the vtable into the file-I/O model, whose
    // handle extraction must follow the reference layer the
    // recovered receiver carries.
    let dir = temp_dir("dyn-file-read");
    let src = fixture(&dir, "dyn_file_read.rs", DYN_FILE_READ_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "dyn file read ok\n");
}
