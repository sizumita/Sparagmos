// Range slice view and indexing tests.

const PLACE_SLICE_META_SRC: &str = r#"
static C: &[i32] = &[10, 20, 30];
fn main() {
    assert_eq!(C.len(), 3);
    assert_eq!(C[1], 20);
    let ys: Box<[i32]> = Box::new([1, 2, 3]);
    let s: &[i32] = &*ys;
    assert_eq!(s.len(), 3);
    assert_eq!(s[1], 2);
    println!("meta ok");
}
"#;

const RANGE_INDEX_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3, 4];
    assert_eq!(&a[1..3], &[2, 3]);
    assert_eq!(&a[..2], &[1, 2]);
    assert_eq!(&a[2..], &[3, 4]);
    assert_eq!(&a[..], &[1, 2, 3, 4]);
    assert_eq!(&a[1..=2], &[2, 3]);
    let st = "héllo";
    assert_eq!(&st[0..1], "h");
    assert_eq!(&st[1..3], "é");
    println!("ranges ok");
}
"#;

const RANGE_SLICE_VIEW_CELL_SRC: &str = r#"
use std::cell::Cell;

fn main() {
    let src = [Cell::new(1), Cell::new(2), Cell::new(3)];
    let view = &src[1..];
    view[0].set(20);
    src[2].set(30);
    println!("{} {} {} {}", src[0].get(), src[1].get(), src[2].get(), view[1].get());
}
"#;

#[test]
fn run_mode_range_slice_view_cell_writes_reach_source() {
    let dir = temp_dir("range-slice-view-cell");
    let src = fixture(&dir, "range_slice_view_cell.rs", RANGE_SLICE_VIEW_CELL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a shared range slice over a place must alias its storage; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1 20 30 30\n");
}

const RANGE_SLICE_BOX_FROM_MUTATING_CLONE_SRC: &str = r#"
use std::cell::Cell;

struct S { counter: Cell<i32> }

impl Clone for S {
    fn clone(&self) -> Self {
        self.counter.set(self.counter.get() + 1);
        S { counter: Cell::new(self.counter.get()) }
    }
}

fn main() {
    let src = [S { counter: Cell::new(10) }];
    let boxed: Box<[S]> = Box::from(&src[..]);
    println!("src={} boxed={}", src[0].counter.get(), boxed[0].counter.get());
}
"#;

#[test]
fn run_mode_box_from_range_slice_runs_mutating_clone_on_source() {
    let dir = temp_dir("box-from-range-slice-mutating-clone");
    let src = fixture(
        &dir,
        "box_from_range_slice_mutating_clone.rs",
        RANGE_SLICE_BOX_FROM_MUTATING_CLONE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box::from over a range slice must pass the source elements to Clone; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "src=11 boxed=11\n");
}

const RANGE_SLICE_VIEW_NESTED_SRC: &str = r#"
use std::cell::Cell;

fn main() {
    let src = [Cell::new(1), Cell::new(2), Cell::new(3), Cell::new(4)];
    let mid = &src[1..3];
    let tail = &mid[1..];
    tail[0].set(30);
    let head = &src[..2];
    head[0].set(10);
    println!("{} {} {} {}", src[0].get(), src[1].get(), src[2].get(), src[3].get());
}
"#;

#[test]
fn run_mode_range_slice_view_nested_ranges_alias_source() {
    let dir = temp_dir("range-slice-view-nested");
    let src = fixture(
        &dir,
        "range_slice_view_nested.rs",
        RANGE_SLICE_VIEW_NESTED_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a range slice of a range slice must alias the original storage; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "10 2 30 4\n");
}

const RANGE_SLICE_VIEW_RANGE_KINDS_SRC: &str = r#"
use std::cell::Cell;

fn main() {
    let src = [Cell::new(1), Cell::new(2), Cell::new(3), Cell::new(4)];
    let a = &src[2..];
    a[0].set(30);
    let b = &src[..=1];
    b[1].set(20);
    let c = &src[..];
    c[3].set(40);
    println!("{} {} {} {}", src[0].get(), src[1].get(), src[2].get(), src[3].get());
}
"#;

#[test]
fn run_mode_range_slice_view_range_kinds_alias_source() {
    let dir = temp_dir("range-slice-view-range-kinds");
    let src = fixture(
        &dir,
        "range_slice_view_range_kinds.rs",
        RANGE_SLICE_VIEW_RANGE_KINDS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "every range kind over a place must alias its storage; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1 20 30 40\n");
}

const RANGE_SLICE_VIEW_VEC_SRC: &str = r#"
use std::cell::Cell;

fn main() {
    let v = vec![Cell::new(1), Cell::new(2), Cell::new(3)];
    let view = &v[1..];
    view[1].set(30);
    println!("{} {} {}", v[0].get(), v[1].get(), v[2].get());
}
"#;

#[test]
fn run_mode_range_slice_view_vec_writes_reach_source() {
    let dir = temp_dir("range-slice-view-vec");
    let src = fixture(&dir, "range_slice_view_vec.rs", RANGE_SLICE_VIEW_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a shared range slice over a Vec must alias its storage; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1 2 30\n");
}

const RANGE_SLICE_VIEW_BOXED_SRC: &str = r#"
use std::cell::Cell;

fn main() {
    let b: Box<[Cell<i32>]> = Box::from([Cell::new(1), Cell::new(2), Cell::new(3)]);
    let view = &b[1..];
    view[0].set(20);
    println!("{} {} {}", b[0].get(), b[1].get(), b[2].get());
}
"#;

#[test]
fn run_mode_range_slice_view_boxed_slice_writes_reach_source() {
    let dir = temp_dir("range-slice-view-boxed");
    let src = fixture(
        &dir,
        "range_slice_view_boxed.rs",
        RANGE_SLICE_VIEW_BOXED_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a shared range slice over a boxed slice must alias its storage; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1 20 3\n");
}

#[test]
fn run_mode_range_indexing() {
    let dir = temp_dir("range-index");
    let src = fixture(&dir, "range_index.rs", RANGE_INDEX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "range indexing must run to completion; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ranges ok\n");
}

#[test]
fn run_mode_slice_metadata_through_places() {
    let dir = temp_dir("place-slice-meta");
    let src = fixture(&dir, "place_slice_meta.rs", PLACE_SLICE_META_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "PtrMetadata/Len through place references must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "meta ok\n");
}
