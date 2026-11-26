// Tests that do not belong to a single vec/slice family.

const VEC_REPEAT_SRC: &str = r#"
fn main() {
    let v = vec![7; 4];
    println!("len {} sum {}", v.len(), v[0] + v[3]);
    let w = vec![String::from("ab"); 2];
    println!("{} {}", w.len(), w[1]);
}
"#;

#[test]
fn run_mode_vec_repeat_form() {
    // The `vec![x; n]` repeat form goes through
    // `std::vec::from_elem`, another allocation entry point the
    // interpreter models.
    let dir = temp_dir("vecrepeat");
    let src = fixture(&dir, "vecrepeat.rs", VEC_REPEAT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec![x; n] must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 4 sum 14\n2 ab\n",
        "repeat-form Vec behaves exactly as natively"
    );
}

const VEC_CLONE_CUSTOM_CLONE_SRC: &str = r#"
use std::cell::Cell;
use std::rc::Rc;

#[derive(Debug)]
struct Counted {
    id: i32,
    clones: Rc<Cell<i32>>,
}
impl Clone for Counted {
    fn clone(&self) -> Self {
        self.clones.set(self.clones.get() + 1);
        Counted { id: self.id + 100, clones: self.clones.clone() }
    }
}

fn main() {
    let counter = Rc::new(Cell::new(0));
    let v = vec![
        Counted { id: 1, clones: counter.clone() },
        Counted { id: 2, clones: counter.clone() },
        Counted { id: 3, clones: counter.clone() },
    ];
    let w = v.clone();
    println!("clones: {}", counter.get());
    println!("w: {:?}", w.iter().map(|c| c.id).collect::<Vec<_>>());
    println!("v: {:?}", v.iter().map(|c| c.id).collect::<Vec<_>>());
    let tv = vec![(Counted { id: 7, clones: counter.clone() }, 1u32)];
    let tw = tv.clone();
    println!("tuple clones: {}", counter.get());
    println!("tw: {:?}", tw.iter().map(|c| c.0.id).collect::<Vec<_>>());
    let arr = [
        Counted { id: 10, clones: counter.clone() },
        Counted { id: 20, clones: counter.clone() },
    ];
    let av = arr.to_vec();
    println!("to_vec clones: {}", counter.get());
    println!("av: {:?}", av.iter().map(|c| c.id).collect::<Vec<_>>());
}
"#;

#[test]
fn run_mode_vec_clone_runs_custom_clone() {
    let dir = temp_dir("vec-clone-custom-clone");
    let src = fixture(&dir, "vec_clone_custom_clone.rs", VEC_CLONE_CUSTOM_CLONE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::clone must run the elements' own Clone impls; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "clones: 3\nw: [101, 102, 103]\nv: [1, 2, 3]\ntuple clones: 4\ntw: [107]\nto_vec clones: 6\nav: [110, 120]\n"
    );
}

const VEC_CLONE_COUNTED_HANDLES_SRC: &str = r#"
use std::cell::Cell;
use std::rc::Rc;
fn main() {
    let rc = Rc::new(Cell::new(10));
    let v = vec![rc.clone(), rc.clone()];
    println!("base: {}", Rc::strong_count(&rc));
    let w = v.clone();
    println!("after clone: {}", Rc::strong_count(&rc));
    w[0].set(99);
    println!("shared: {}", rc.get());
    drop(w);
    println!("after drop clone: {}", Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_vec_clone_counted_handles() {
    let dir = temp_dir("vec-clone-counted-handles");
    let src = fixture(
        &dir,
        "vec_clone_counted_handles.rs",
        VEC_CLONE_COUNTED_HANDLES_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::clone of counted handles must clone the handles, not copy or deep-clone them; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "base: 3\nafter clone: 5\nshared: 99\nafter drop clone: 3\n"
    );
}

const VEC_CELL_DROP_SRC: &str = r#"use std::cell::Cell;
struct D(i32);
impl Drop for D { fn drop(&mut self) { println!("drop D({})", self.0); } }
fn main() {
    let v = vec![Cell::new(D(1)), Cell::new(D(2))];
    drop(v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_cell_drop_runs_element_destructors() {
    // Dropping a `Vec<Cell<D>>` runs the element drop glue
    // through the cells: each element's destructor observes the
    // value the cell holds.
    let dir = temp_dir("vec-cell-drop");
    let src = fixture(&dir, "vec_cell_drop.rs", VEC_CELL_DROP_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "drop D(1)\ndrop D(2)\ndone\n");
}

const VEC_AS_REF_SLICE_SRC: &str = r#"fn main() {
    let v: Vec<i32> = vec![5, 3, 8, 1];
    let s: &[i32] = v.as_ref();
    println!("idx {} {} {}", s[0], s[2], s[3]);
    println!("first {:?}", s.first());
    println!("get {:?}", s.get(2));
    println!("debug {:?}", s);
    println!("sub {:?}", &s[1..3]);
    println!("contains {}", s.contains(&8));
    println!("eq {}", s == [5, 3, 8, 1].as_slice());
    println!("to_vec {:?}", s.to_vec());
    let mut w: Vec<i32> = Vec::with_capacity(16);
    w.push(10);
    w.push(20);
    let ws: &[i32] = w.as_ref();
    println!("sparse {} {} len {}", ws[0], ws[1], ws.len());
    let e: Vec<i32> = Vec::new();
    let es: &[i32] = e.as_ref();
    println!("empty {} {:?}", es.len(), es.first());
    let bv = Box::new(vec![7, 9]);
    let bs: &[i32] = bv.as_ref();
    println!("boxvec {} {}", bs[0], bs[1]);
    let rv = std::rc::Rc::new(vec![4, 6]);
    let rs: &[i32] = rv.as_ref();
    println!("rcvec {} {}", rs[0], rs[1]);
    let vs: Vec<String> = vec!["aa".to_string(), "bbb".to_string()];
    let ss: &[String] = vs.as_ref();
    println!("strings {} {} {}", ss[0], ss[1], ss[1].len());
    let vv: Vec<Vec<i32>> = vec![vec![1, 2], vec![3]];
    let sv: &[Vec<i32>] = vv.as_ref();
    println!("nested {} {}", sv[0][1], sv[1][0]);
}
"#;

#[test]
fn run_mode_vec_as_ref_slice_element_access() {
    // Issue #443: `Vec::as_ref` built its slice view as a fat
    // pointer over the vector aggregate itself, a bare ADT value
    // no dereference can follow: `len` and iteration (which read
    // the fat pointer itself) worked, but element access through
    // the view (`s[0]`, `first`, `get`, sub-slicing, `Debug`)
    // failed with "type mismatch in dereference: expected
    // reference, found ADT value". The view now denotes the
    // element sequence, like every other slice construction.
    let dir = temp_dir("vec-as-ref-slice");
    let src = fixture(&dir, "vec_as_ref_slice.rs", VEC_AS_REF_SLICE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        [
            "idx 5 8 1",
            "first Some(5)",
            "get Some(8)",
            "debug [5, 3, 8, 1]",
            "sub [3, 8]",
            "contains true",
            "eq true",
            "to_vec [5, 3, 8, 1]",
            "sparse 10 20 len 2",
            "empty 0 None",
            "boxvec 7 9",
            "rcvec 4 6",
            "strings aa bbb 3",
            "nested 2 3",
        ]
    );
}

const GENERIC_PATHS_SRC: &str = r#"
fn make<T>() -> Vec<T> { Vec::<T>::new() }
fn main() {
    let v: Vec<i32> = make();
    assert!(v.is_empty());
    let d = <i32 as std::default::Default>::default();
    assert_eq!(d, 0);
    println!("ok");
}
"#;

#[test]
fn run_mode_generic_context_vec_new_and_qualified_default() {
    let dir = temp_dir("generic-paths");
    let src = fixture(&dir, "generic_paths.rs", GENERIC_PATHS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const VEC_CELL_SRC: &str = r#"
use std::cell::Cell;
fn main() {
    let v = vec![Cell::new(0u32), Cell::new(0u32)];
    v[1].set(9);
    println!("{} {}", v[0].get(), v[1].get());
    std::mem::forget(v);
}
"#;

#[test]
fn run_mode_vec_cell_index_set() {
    // `v[1].set(..)` reaches the element through the `Index`
    // builtin; with a snapshot element the `Cell::set` write
    // failed with "expected mutable place, found constant
    // aggregate". When the indexed value lives in a place, the
    // builtin returns a reference to the element place.
    let dir = temp_dir("vec-cell");
    let src = fixture(&dir, "vec_cell.rs", VEC_CELL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell::set through a Vec index must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "0 9\n");
}

// ----- Lane KE: reborrowed &mut Vec slice positions ---------------

const MUT_VEC_REBORROW_REVERSE_SRC: &str = r#"
fn reverse_arg(v: &mut [usize]) {
    v.reverse();
}
fn bar(v: &mut [usize]) {
    v.reverse();
    reverse_arg(v);
    v.reverse();
}
fn main() {
    let mut the_vec = vec![0x0102usize, 0x0304, 0x0506, 0x0708];
    bar(&mut the_vec);
    assert_eq!(the_vec, [0x0708, 0x0506, 0x0304, 0x0102]);
    println!("ok");
}
"#;

#[test]
fn run_mode_mut_vec_reborrow_reverse() {
    // A `&mut Vec` reborrowed as `&mut [usize]` and reversed
    // through the receiver and through a helper argument: the
    // slice machinery (`split_at_mut`, `revswap`, `mem::swap` over
    // byte-chunk pointers) must address whole elements of the
    // backing Vec, in both directions. Multi-byte element values
    // pin the semantics: a byte-wise swap would corrupt them.
    run_ok("mut-vec-reborrow", MUT_VEC_REBORROW_REVERSE_SRC, "ok\n");
}

const DEFAULT_FIELD_VEC_SRC: &str = r#"
#![feature(default_field_values)]

pub struct Foo {
    pub baz: i32 = 42 + 3,
    pub bak: Vec<i32> = Vec::new(),
}

fn main() {
    let x = Foo { .. };
    assert_eq!(x.baz, 45);
    assert!(x.bak.is_empty());
    println!("default field values ok");
}
"#;

#[test]
fn run_mode_default_field_value_empty_vec_is_empty() {
    // A field default is materialised as a constant in the
    // type's raw layout; for a modelled container (`Vec`) the
    // constant must be translated into the model's
    // contents representation, or the raw tree's field count
    // reads back as the vector's length.
    let dir = temp_dir("default-field-vec");
    let src = fixture(&dir, "default_field_vec.rs", DEFAULT_FIELD_VEC_SRC);
    let out = run_driver_extra(&src, &["--edition", "2024"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a Vec::new() field default must be an empty vector; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "default field values ok\n");
}

const BOX_DEFAULT_SLICE_OF_ARRAYS_SRC: &str = r#"
fn main() {
    let b: Box<[[i32; 2]]> = Default::default();
    println!("slice {} {:?}", b.len(), b);
    let c: Box<[i32; 2]> = Default::default();
    println!("array {:?}", c);
    let d: Box<[[u8; 3]]> = Box::default();
    println!("via-box {} {:?}", d.len(), d);
}
"#;

#[test]
fn run_mode_box_default_slice_of_arrays() {
    // `Box<[[i32; 2]]>::default()` is an empty boxed slice, but
    // the pointee check in `default_value` judged slice-ness
    // with a raw `!contains(';')`, fell through to the array
    // default of the outer tag, and gave up as unsupported.
    // The `Box<[i32; 2]>` line pins the array direction: an
    // array pointee still defaults elementwise.
    run_ok(
        "box-default-slice-of-arrays",
        BOX_DEFAULT_SLICE_OF_ARRAYS_SRC,
        "slice 0 []\narray [0, 0]\nvia-box 0 []\n",
    );
}

const AS_REF_SLICE_OF_ARRAYS_SRC: &str = r#"
fn main() {
    let a: [[i32; 2]; 2] = [[1, 2], [3, 4]];
    let s: &[[i32; 2]] = a.as_ref();
    println!("asref {} {:?}", s.len(), s);
    let t: &[[i32; 2]] = std::borrow::Borrow::borrow(&a);
    println!("borrow {} {:?}", t.len(), t);
}
"#;

#[test]
fn run_mode_as_ref_slice_of_arrays() {
    // The array-to-slice view in `view_value_as` used the same
    // raw `contains(';')` pair to recognize its case; it now
    // goes through `is_slice_tag` on both sides like the other
    // consumers. (The pre-unification code reached the same
    // output here through the reference fallback, so this test
    // pins the behavior while the predicate is unified.)
    run_ok(
        "as-ref-slice-of-arrays",
        AS_REF_SLICE_OF_ARRAYS_SRC,
        "asref 2 [[1, 2], [3, 4]]\nborrow 2 [[1, 2], [3, 4]]\n",
    );
}

const OFFSET_FROM_WRAPPER_DENOMINATED_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = (0..64).collect();
    let base = v.as_mut_ptr();
    unsafe {
        let wrapped = base.add(40) as *mut std::mem::MaybeUninit<usize>;
        let plain = base.add(24) as *const usize;
        let d = (wrapped as *const usize).offset_from(plain);
        println!("dist {d}");
        let d2 = plain.offset_from(wrapped as *const usize);
        println!("dist2 {d2}");
    }
}
"#;

#[test]
fn run_mode_offset_from_wrapper_denominated_pointer() {
    // A distance between an element pointer denominated
    // through a wrapper entry (its projection carries the
    // entry step after the element step) and a plain element
    // pointer into the same buffer: the distance split's key
    // included the entry step, so the two never shared a key.
    // The split now also offers the reading whose key drops
    // the whole-value entry steps after the element step.
    run_ok(
        "offset-from-wrapper-denominated",
        OFFSET_FROM_WRAPPER_DENOMINATED_SRC,
        "dist 16\ndist2 -16\n",
    );
}
