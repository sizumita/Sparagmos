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

const ROTATE_SWAP_BRANCH_SRC: &str = r#"
fn main() {
    let mut v: Vec<(u64, u64)> = (0..42u64).map(|i| (i, i * 10)).collect();
    v.rotate_left(20);
    println!("t42={:?}", v);
    v.rotate_right(20);
    println!("back={:?}", v);
    let mut v: Vec<Box<str>> = (0..42).map(|i| format!("s{}", i).into_boxed_str()).collect();
    v.rotate_left(20);
    println!("bx={:?}", v);
    let mut v: Vec<String> = (0..50).map(|i| format!("s{}", i)).collect();
    v.rotate_left(20);
    println!("st={:?}", v);
    let mut v: Vec<(u16, u16)> = (0..200u16).map(|i| (i, i * 3)).collect();
    v.rotate_left(101);
    println!("tail={:?} {:?} {:?} {:?}", v[0], v[98], v[99], v[199]);
}
"#;

#[test]
fn run_mode_rotate_swap_branch() {
    // Both segments over the memmove threshold and a
    // segment sum of 24 or more: core's slice rotation
    // takes the swap branch (`ptr_rotate_swap`), whose
    // exchange runs through `ptr::swap_nonoverlapping`'s
    // byte form -- `MaybeUninit<[u8; N]>` chunk loads and
    // stores at byte positions inside element-shaped
    // storage, which stopped the run on a displaced read
    // (and, where the displacement absorbed into an index,
    // silently exchanged the wrong units). The interpreter
    // serves the branch over the element sequence (see
    // `Builtin::PtrRotate`). The last case's range byte
    // length is not a multiple of the chunk width, so the
    // exchange's tail chunks run as well.
    run_ok(
        "rotate-swap-branch",
        ROTATE_SWAP_BRANCH_SRC,
        "t42=[(20, 200), (21, 210), (22, 220), (23, 230), (24, 240), (25, 250), (26, 260), (27, 270), (28, 280), (29, 290), (30, 300), (31, 310), (32, 320), (33, 330), (34, 340), (35, 350), (36, 360), (37, 370), (38, 380), (39, 390), (40, 400), (41, 410), (0, 0), (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90), (10, 100), (11, 110), (12, 120), (13, 130), (14, 140), (15, 150), (16, 160), (17, 170), (18, 180), (19, 190)]\nback=[(0, 0), (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90), (10, 100), (11, 110), (12, 120), (13, 130), (14, 140), (15, 150), (16, 160), (17, 170), (18, 180), (19, 190), (20, 200), (21, 210), (22, 220), (23, 230), (24, 240), (25, 250), (26, 260), (27, 270), (28, 280), (29, 290), (30, 300), (31, 310), (32, 320), (33, 330), (34, 340), (35, 350), (36, 360), (37, 370), (38, 380), (39, 390), (40, 400), (41, 410)]\nbx=[\"s20\", \"s21\", \"s22\", \"s23\", \"s24\", \"s25\", \"s26\", \"s27\", \"s28\", \"s29\", \"s30\", \"s31\", \"s32\", \"s33\", \"s34\", \"s35\", \"s36\", \"s37\", \"s38\", \"s39\", \"s40\", \"s41\", \"s0\", \"s1\", \"s2\", \"s3\", \"s4\", \"s5\", \"s6\", \"s7\", \"s8\", \"s9\", \"s10\", \"s11\", \"s12\", \"s13\", \"s14\", \"s15\", \"s16\", \"s17\", \"s18\", \"s19\"]\nst=[\"s20\", \"s21\", \"s22\", \"s23\", \"s24\", \"s25\", \"s26\", \"s27\", \"s28\", \"s29\", \"s30\", \"s31\", \"s32\", \"s33\", \"s34\", \"s35\", \"s36\", \"s37\", \"s38\", \"s39\", \"s40\", \"s41\", \"s42\", \"s43\", \"s44\", \"s45\", \"s46\", \"s47\", \"s48\", \"s49\", \"s0\", \"s1\", \"s2\", \"s3\", \"s4\", \"s5\", \"s6\", \"s7\", \"s8\", \"s9\", \"s10\", \"s11\", \"s12\", \"s13\", \"s14\", \"s15\", \"s16\", \"s17\", \"s18\", \"s19\"]\ntail=(101, 303) (199, 597) (0, 0) (100, 300)\n",
    );
}

const ROTATE_GCD_BRANCH_SRC: &str = r#"
fn main() {
    let mut v: Vec<[u64; 5]> = (0..22u64).map(|i| [i, i + 100, i + 200, i + 300, i + 400]).collect();
    v.rotate_left(10);
    println!("a5={:?}", v);
    let mut v: Vec<[u64; 4]> = (0..23u64).map(|i| [i, i + 100, i + 200, i + 300]).collect();
    v.rotate_left(11);
    println!("a4={:?}", v);
    let mut v: Vec<String> = (0..23).map(|i| format!("s{}", i)).collect();
    v.rotate_left(9);
    println!("st={:?}", v);
    let mut a: [[u64; 5]; 22] = [[0; 5]; 22];
    for i in 0..22u64 { a[i as usize] = [i, i + 100, i + 200, i + 300, i + 400]; }
    a.rotate_left(10);
    println!("stk={:?} {:?} {:?}", a[0], a[11], a[12]);
}
"#;

#[test]
fn run_mode_rotate_gcd_branch() {
    // The GCD (juggling) branch of the rotation primitive
    // (`ptr_rotate_gcd`): taken when the segment sum is
    // below 24 or the element is wider than 32 bytes. Its
    // single-element moves are typed, but over a stack
    // array's storage the element read came back as the
    // array's first scalar and the run stopped; the branch
    // is served by the same modelled primitive as the swap
    // branch (see `Builtin::PtrRotate`).
    run_ok(
        "rotate-gcd-branch",
        ROTATE_GCD_BRANCH_SRC,
        "a5=[[10, 110, 210, 310, 410], [11, 111, 211, 311, 411], [12, 112, 212, 312, 412], [13, 113, 213, 313, 413], [14, 114, 214, 314, 414], [15, 115, 215, 315, 415], [16, 116, 216, 316, 416], [17, 117, 217, 317, 417], [18, 118, 218, 318, 418], [19, 119, 219, 319, 419], [20, 120, 220, 320, 420], [21, 121, 221, 321, 421], [0, 100, 200, 300, 400], [1, 101, 201, 301, 401], [2, 102, 202, 302, 402], [3, 103, 203, 303, 403], [4, 104, 204, 304, 404], [5, 105, 205, 305, 405], [6, 106, 206, 306, 406], [7, 107, 207, 307, 407], [8, 108, 208, 308, 408], [9, 109, 209, 309, 409]]\na4=[[11, 111, 211, 311], [12, 112, 212, 312], [13, 113, 213, 313], [14, 114, 214, 314], [15, 115, 215, 315], [16, 116, 216, 316], [17, 117, 217, 317], [18, 118, 218, 318], [19, 119, 219, 319], [20, 120, 220, 320], [21, 121, 221, 321], [22, 122, 222, 322], [0, 100, 200, 300], [1, 101, 201, 301], [2, 102, 202, 302], [3, 103, 203, 303], [4, 104, 204, 304], [5, 105, 205, 305], [6, 106, 206, 306], [7, 107, 207, 307], [8, 108, 208, 308], [9, 109, 209, 309], [10, 110, 210, 310]]\nst=[\"s9\", \"s10\", \"s11\", \"s12\", \"s13\", \"s14\", \"s15\", \"s16\", \"s17\", \"s18\", \"s19\", \"s20\", \"s21\", \"s22\", \"s0\", \"s1\", \"s2\", \"s3\", \"s4\", \"s5\", \"s6\", \"s7\", \"s8\"]\nstk=[10, 110, 210, 310, 410] [21, 121, 221, 321, 421] [0, 100, 200, 300, 400]\n",
    );
}

const ROTATE_FORMS_SRC: &str = r#"
fn main() {
    let mut v: Vec<(u64, u64)> = (0..60u64).map(|i| (i, i * 10)).collect();
    v[10..50].rotate_left(20);
    println!("sub={:?}", v);
    let mut v: Vec<u8> = (0..600u32).map(|i| (i % 251) as u8).collect();
    v.rotate_left(300);
    println!("u8={} {} {}", v[0], v[300], v.iter().map(|b| *b as u64).sum::<u64>());
    v.rotate_left(0);
    v.rotate_left(600);
    println!("noop={} {}", v[0], v[300]);
    let mut b: Box<[(u64, u64)]> = (0..34u64).map(|i| (i, i * 10)).collect();
    b.rotate_left(17);
    println!("bx={:?}", b);
    let mut w: Vec<(u64, u64)> = (0..34u64).map(|i| (i, i * 10)).collect();
    let s: &mut [(u64, u64)] = &mut w[..];
    s.rotate_left(17);
    println!("sl={:?}", w);
}
"#;

#[test]
fn run_mode_rotate_forms() {
    // The modelled rotation over the other storage forms
    // a slice can denote: a sub-slice range inside a larger
    // buffer (the elements outside the range must not
    // move), a byte vector's flat storage, a boxed slice,
    // a slice reborrow, and the empty rotations.
    run_ok(
        "rotate-forms",
        ROTATE_FORMS_SRC,
        "sub=[(0, 0), (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90), (30, 300), (31, 310), (32, 320), (33, 330), (34, 340), (35, 350), (36, 360), (37, 370), (38, 380), (39, 390), (40, 400), (41, 410), (42, 420), (43, 430), (44, 440), (45, 450), (46, 460), (47, 470), (48, 480), (49, 490), (10, 100), (11, 110), (12, 120), (13, 130), (14, 140), (15, 150), (16, 160), (17, 170), (18, 180), (19, 190), (20, 200), (21, 210), (22, 220), (23, 230), (24, 240), (25, 250), (26, 260), (27, 270), (28, 280), (29, 290), (50, 500), (51, 510), (52, 520), (53, 530), (54, 540), (55, 550), (56, 560), (57, 570), (58, 580), (59, 590)]\nu8=49 0 67503\nnoop=49 0\nbx=[(17, 170), (18, 180), (19, 190), (20, 200), (21, 210), (22, 220), (23, 230), (24, 240), (25, 250), (26, 260), (27, 270), (28, 280), (29, 290), (30, 300), (31, 310), (32, 320), (33, 330), (0, 0), (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90), (10, 100), (11, 110), (12, 120), (13, 130), (14, 140), (15, 150), (16, 160)]\nsl=[(17, 170), (18, 180), (19, 190), (20, 200), (21, 210), (22, 220), (23, 230), (24, 240), (25, 250), (26, 260), (27, 270), (28, 280), (29, 290), (30, 300), (31, 310), (32, 320), (33, 330), (0, 0), (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90), (10, 100), (11, 110), (12, 120), (13, 130), (14, 140), (15, 150), (16, 160)]\n",
    );
}
