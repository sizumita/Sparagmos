// Subslice projection and split tests.

// ----- subslice projections and pattern types (Lane FG) ---------------------

const SUBSLICE_SRC: &str = r#"
fn head_rest(s: &[i32]) -> i32 {
    match s {
        [first, rest @ ..] => first + rest.len() as i32,
        [] => -1,
    }
}
fn tail2(s: &[i32]) -> i32 {
    match s {
        [.., a, b] => a * 10 + b,
        _ => 0,
    }
}
fn mid_sum(s: &[i32]) -> i32 {
    match s {
        [_, m @ .., _] => {
            let mut acc = 0;
            let mut i = 0;
            while i < m.len() { acc += m[i]; i += 1; }
            acc
        }
        _ => 0,
    }
}
fn main() {
    let arr = [1, 2, 3, 4, 5];
    let s = &arr[1..3];
    println!("len {} sum {}", s.len(), s[0] + s[1]);
    let mut arr2 = [10, 20, 30, 40];
    arr2[1..3][0] = 99;
    println!("w {}", arr2[1]);
    let t = &arr[..2];
    println!("head {}", t.len());
    let u = &arr[3..];
    println!("tail {}", u[0] + u[1]);
    let v = [5, 1, 2, 3];
    println!("hr {}", head_rest(&v));
    println!("t2 {}", tail2(&v));
    println!("mid {}", mid_sum(&v));
    println!("hr2 {}", head_rest(&[9, 8]));
    let mut arr3 = [0, 1, 2, 3, 4, 5];
    arr3[2..5][1] = 77;
    println!("nested {}", arr3[3]);
    let w = &arr[1..4][1..];
    println!("subsub {} {}", w.len(), w[0]);
}
"#;

#[test]
fn run_mode_subslice_projections_read_write_and_patterns() {
    // Subslice places (from slice patterns and range indexing)
    // convert and execute: shared range reads, writes through
    // `IndexMut` ranges starting above zero, `@ ..` bindings, and
    // nested sub-slices all alias the base storage.
    let dir = temp_dir("subslice");
    let src = fixture(&dir, "subslice.rs", SUBSLICE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "subslice program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2 sum 5\nw 99\nhead 2\ntail 9\nhr 8\nt2 23\nmid 3\nhr2 10\nnested 77\nsubsub 2 3\n",
        "subslice stdout mismatch; stderr={stderr}"
    );
}

const PATTERN_TYPE_SRC: &str = r#"
#![feature(pattern_types, pattern_type_macro)]
use std::pat::pattern_type;
type Y = pattern_type!(u32 is 1..);
struct Thing(Y);
fn id_thing(t: Thing) -> Thing { t }
fn describe(z: Option<Y>) -> u32 {
    match z { Some(_) => 1, None => 0 }
}
fn main() {
    let a = id_thing(Thing(2));
    let _ = a.0;
    println!("some {}", describe(Some(42)));
    println!("none {}", describe(None));
    println!("size {}", std::mem::size_of::<Y>());
    let arr = [Thing(1), Thing(2)];
    println!("moved {}", describe(Some(arr[0].0)) + describe(Some(arr[1].0)));
}
"#;

#[test]
fn run_mode_pattern_type_values_execute_as_base_type() {
    // Pattern types (`Ty::Pat`) convert as their base type: the
    // values move through calls, arrays and enum construction
    // with the base representation. (The fixture needs the
    // internal feature; the driver test harness runs with
    // RUSTC_BOOTSTRAP=1, like compiletest.)
    let dir = temp_dir("pattern-type");
    let src = fixture(&dir, "pattern_type.rs", PATTERN_TYPE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pattern-type program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "some 1\nnone 0\nsize 4\nmoved 2\n",
        "pattern-type stdout mismatch; stderr={stderr}"
    );
}

const SUBSLICE_SPLIT_AT_MUT_SRC: &str = r#"
fn main() {
    let mut v = vec![10usize, 20, 30, 40, 50, 60];
    let sub = &mut v[1..5];
    let (a, b) = sub.split_at_mut(2);
    a[1] = 99;
    b[0] = 77;
    a.reverse();
    assert_eq!(v, [10, 99, 20, 77, 50, 60]);
    println!("ok");
}
"#;

#[test]
fn run_mode_subslice_split_at_mut_positions() {
    // `split_at_mut` on a range sub-slice: the front half's data
    // pointer is the sub-slice's first element (not that
    // element's own first element), indexing either half composes with the
    // sub-slice offset in the backing Vec, and the back half's data
    // pointer absorbs its displacement through the sub-slice step
    // instead of degrading to a displaced pointer.
    run_ok("subslice-split", SUBSLICE_SPLIT_AT_MUT_SRC, "ok\n");
}

const RAW_PARTS_NESTED_VEC_SRC: &str = r#"
fn main() {
    let v: Vec<Vec<usize>> = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9], vec![10]];
    let s: &[Vec<usize>] = &v;
    let (a, b) = s.split_at(3);
    println!("split {} {} {}", a[0][0], a[1][0], b[0][0]);
    let u = unsafe { std::slice::from_raw_parts(s.as_ptr().add(1), 2) };
    println!("raw {} {}", u[0][0], u[1][0]);
    let w = &u[1..];
    println!("win {}", w[0][0]);
    let mut mv = vec![vec![1], vec![2], vec![3]];
    let sm: &mut [Vec<usize>] = &mut mv;
    let (ma, _mb) = sm.split_at_mut(2);
    ma[0].push(99);
    println!("mut {}", mv[0][1]);
}
"#;

#[test]
fn run_mode_split_at_and_raw_parts_over_vec_of_vecs() {
    // The same formation denomination through the direct
    // shapes: `split_at` (core's body forms both halves with
    // `from_raw_parts` over the receiver's element pointer),
    // an offset `from_raw_parts`, a range window over the
    // formed slice, and a mutable split whose writes must
    // alias the container through the window denomination.
    // The outputs below are native rustc 1.100.0's.
    run_ok(
        "raw-parts-nested-vec",
        RAW_PARTS_NESTED_VEC_SRC,
        "split 1 4 10\nraw 4 6\nwin 6\nmut 99\n",
    );
}

const ARRAY_SUBSLICE_BINDING_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let a = [D(1), D(2), D(3)];
    let [_, _y @ ..] = a;
    println!("bound");
}
"#;

#[test]
fn run_mode_array_subslice_binding_moves_window_out() {
    // A by-value `@ ..` binding in an array pattern moves the
    // sub-slice window out of the array: the move leaves the
    // vacated window uninitialised, a whole-sub-slice write
    // that distributes the uninitialised marker over the
    // window's elements, and the array's remaining element
    // drops on its own afterwards. The run prints the native
    // drop order: the binding's elements first, then the
    // element left in the array.
    let dir = temp_dir("array-subslice-binding");
    let src = fixture(&dir, "array_subslice_binding.rs", ARRAY_SUBSLICE_BINDING_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the array sub-slice binding must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "bound\ndtor 2\ndtor 3\ndtor 1\n",
        "the moved window's elements drop with the binding, the rest with the array"
    );
}
