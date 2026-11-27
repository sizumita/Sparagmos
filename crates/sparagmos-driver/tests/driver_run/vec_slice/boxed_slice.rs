// Boxed slice construction and conversion tests.

const BOX_UNSIZE_SRC: &str = r#"
fn main() {
    let xs: Box<[()]> = Box::<[(); 0]>::new([]);
    assert_eq!(xs.len(), 0);
    let ys: Box<[i32]> = Box::new([1, 2, 3]);
    assert_eq!(ys.len(), 3);
    assert_eq!(&*ys, &[1, 2, 3]);
    println!("unsize ok");
}
"#;

const BOXED_SLICE_FROM_VEC_SRC: &str = r#"
#[derive(Debug)]
struct P {
    x: i32,
}

fn main() {
    // `From<Vec<T>> for Box<[T]>` is `Vec::into_boxed_slice`:
    // the pointee is the element sequence, so every slice
    // consumer of the box sees the elements.
    let b: Box<[i32]> = Box::from(vec![4, 5, 6]);
    println!("{:?}", b);
    println!("{}", b.len());
    println!("{}", b[0] + b[2]);
    for x in b.iter() {
        println!("{}", x);
    }
    // `Into` resolves to the same conversion.
    let c: Box<[i32]> = vec![7, 8].into();
    println!("{:?}", c);
    // Clone, comparison and pretty Debug see the elements too.
    let d = b.clone();
    println!("{} {}", b == d, b < c);
    println!("{:#?}", b);
    // ADT elements keep their derived Debug.
    let e: Box<[P]> = Box::from(vec![P { x: 1 }, P { x: 2 }]);
    println!("{:?}", e);
    // `into_vec` hands back a working vector.
    let mut v = b.into_vec();
    v.push(9);
    v.sort();
    println!("{:?}", v);
    // `From<Cow<'_, [T]>>` hands over the Cow's contents: the
    // owned vector's elements, or a clone of the borrowed
    // slice's elements.
    let cow: std::borrow::Cow<[i32]> = std::borrow::Cow::Owned(vec![11, 12]);
    let f: Box<[i32]> = Box::from(cow);
    println!("{:?}", f);
    let v2 = vec![10, 20, 30, 40];
    let cow2: std::borrow::Cow<[i32]> = std::borrow::Cow::Borrowed(&v2[1..3]);
    let g: Box<[i32]> = Box::from(cow2);
    println!("{:?}", g);
    println!("boxed from vec ok");
}
"#;

const BOXED_SLICE_FROM_BORROWED_COW_SRC: &str = r#"
use std::borrow::Cow;

fn main() {
    // `From<Cow<'_, [T]>> for Box<[T]>` clones the borrowed
    // slice's elements into the new allocation: mutating the
    // source afterwards must not show through the box.
    let mut src = [1, 2, 3];
    let b: Box<[i32]> = Box::from(Cow::Borrowed(&src[..]));
    src[0] = 99;
    println!("array: {} {:?} {}", b[0], b, b.len());

    let mut v = vec![10, 20, 30];
    let c: Box<[i32]> = Box::from(Cow::Borrowed(&v[..]));
    v[1] = 77;
    println!("vec: {} {:?}", c[1], c);

    let mut w = vec![5, 6, 7, 8];
    let d: Box<[i32]> = Box::from(Cow::Borrowed(&w[1..3]));
    w[2] = 66;
    println!("sub: {} {:?}", d[1], d);
}
"#;

const BOXED_SLICE_COLLECT_SRC: &str = r#"
fn main() {
    let b: Box<[i32]> = (0..5).collect();
    println!("{:?}", b);
    let s: Box<[String]> = ["a", "bb"].iter().map(|x| x.to_string()).collect();
    println!("{:?}", s);
    let e: Box<[i32]> = std::iter::empty().collect();
    println!("{:?} {}", e, e.len());
    println!("boxed collect ok");
}
"#;

#[test]
fn run_mode_boxed_slice_unsize() {
    let dir = temp_dir("box-unsize");
    let src = fixture(&dir, "box_unsize.rs", BOX_UNSIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T; N]> to Box<[T]> unsizing must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "unsize ok\n");
}

#[test]
fn run_mode_boxed_slice_from_vec() {
    let dir = temp_dir("boxed-from-vec");
    let src = fixture(&dir, "boxed_from_vec.rs", BOXED_SLICE_FROM_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T]> from Vec must convert and format like into_boxed_slice; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[4, 5, 6]\n3\n10\n4\n5\n6\n[7, 8]\ntrue true\n[\n    4,\n    5,\n    6,\n]\n[P { x: 1 }, P { x: 2 }]\n[4, 5, 6, 9]\n[11, 12]\n[20, 30]\nboxed from vec ok\n"
    );
}

#[test]
fn run_mode_boxed_slice_from_borrowed_cow_clones_elements() {
    let dir = temp_dir("boxed-from-borrowed-cow");
    let src = fixture(
        &dir,
        "boxed_from_borrowed_cow.rs",
        BOXED_SLICE_FROM_BORROWED_COW_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T]> from a borrowed Cow must clone the elements; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "array: 1 [1, 2, 3] 3\nvec: 20 [10, 20, 30]\nsub: 7 [6, 7]\n"
    );
}

const BOXED_SLICE_ARRAY_ELEMENTS_SRC: &str = r#"
fn main() {
    // The element type of a boxed slice may itself be an array:
    // the semicolon inside `[i32; 2]` sits one bracket level
    // deeper and does not make the pointee an array type.
    let b: Box<[[i32; 2]]> = vec![[1, 2], [3, 4]].into_iter().collect();
    println!("collect: {:?}", b);

    let v: Vec<[i32; 2]> = vec![[5, 6], [7, 8]];
    let c: Box<[[i32; 2]]> = Box::from(v);
    println!("fromvec: {:?}", c);

    let arr: [[i32; 2]; 2] = [[9, 10], [11, 12]];
    let d: Box<[[i32; 2]]> = Box::from(&arr[..]);
    println!("fromslice: {:?}", d);

    let e: Box<[[i32; 2]]> = Box::from(std::borrow::Cow::Borrowed(&arr[..]));
    println!("fromcow: {:?}", e);

    let f: Box<[[[i32; 1]; 2]]> = vec![[[1], [2]], [[3], [4]]].into_iter().collect();
    println!("deep: {:?}", f);
}
"#;

#[test]
fn run_mode_boxed_slice_array_elements() {
    let dir = temp_dir("boxed-array-elements");
    let src = fixture(
        &dir,
        "boxed_array_elements.rs",
        BOXED_SLICE_ARRAY_ELEMENTS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "boxed slices of array elements must convert and collect; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "collect: [[1, 2], [3, 4]]\nfromvec: [[5, 6], [7, 8]]\nfromslice: [[9, 10], [11, 12]]\nfromcow: [[9, 10], [11, 12]]\ndeep: [[[1], [2]], [[3], [4]]]\n"
    );
}

const BOXED_SLICE_FROM_BORROWED_COW_DEEP_SRC: &str = r#"
use std::borrow::Cow;

fn main() {
    // The borrowed elements themselves own allocations: boxing
    // clones every element (reallocating the inner boxes), so a
    // later write through the source's element -- or dropping it
    // by replacement -- is invisible through the box.
    let mut src: Vec<Box<i32>> = vec![Box::new(1), Box::new(2), Box::new(3)];
    let b: Box<[Box<i32>]> = Box::from(Cow::Borrowed(&src[..]));
    *src[0] = 99;
    src[1] = Box::new(77);
    println!("deepclone: {} {} {:?}", *b[0], *b[1], b);

    let mut sv: Vec<String> = vec!["aa".to_string(), "bb".to_string()];
    let s: Box<[String]> = Box::from(Cow::Borrowed(&sv[..]));
    sv[0].push_str("ZZ");
    println!("strings: {} {:?}", s[0], s);
}
"#;

#[test]
fn run_mode_boxed_slice_from_borrowed_cow_deep_clones_elements() {
    let dir = temp_dir("boxed-from-borrowed-cow-deep");
    let src = fixture(
        &dir,
        "boxed_from_borrowed_cow_deep.rs",
        BOXED_SLICE_FROM_BORROWED_COW_DEEP_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T]> from a borrowed Cow must deep-clone owning elements; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "deepclone: 1 2 [1, 2, 3]\nstrings: aa [\"aa\", \"bb\"]\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_SRC: &str = r#"
fn main() {
    // `From<&[T]> for Box<[T]>` clones the source slice's
    // elements into the new allocation: neither side's later
    // writes or drops are visible through the other.
    let mut v: Vec<i32> = vec![1, 2, 3];
    let b: Box<[i32]> = Box::from(&v[..]);
    println!("readback: {:?}", b);
    v[0] = 99;
    println!("after source write: {:?}", b);
    let mut b = b;
    b[1] = 77;
    println!("after box write: src={:?} box={:?}", v, b);
    v.clear();
    println!("after source clear: {:?}", b);

    // Owning elements are cloned, not shared.
    let mut src: Vec<Box<i32>> = vec![Box::new(1), Box::new(2)];
    let bb: Box<[Box<i32>]> = Box::from(&src[..]);
    *src[0] = 42;
    println!("deep after elem write: {} {:?}", *bb[0], bb);
    src[1] = Box::new(55);
    println!("deep after elem replace: {} {:?}", *bb[1], bb);

    let mut sv: Vec<String> = vec!["aa".to_string(), "bb".to_string()];
    let s: Box<[String]> = Box::from(&sv[..]);
    sv[0].push_str("ZZ");
    println!("strings: {} {:?}", s[0], s);

    // The `.into()` spelling of the same conversion.
    let w: Vec<i32> = vec![5, 6];
    let c: Box<[i32]> = (&w[..]).into();
    println!("into: {:?}", c);

    // The box outlives the source owner.
    let owned: Box<[i32]>;
    {
        let tmp: Vec<i32> = vec![10, 20, 30];
        owned = Box::from(&tmp[..]);
    }
    println!("after source drop: {:?}", owned);
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_clones_elements() {
    let dir = temp_dir("boxed-from-slice-ref");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref.rs",
        BOXED_SLICE_FROM_SLICE_REF_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T]> from &[T] must clone the elements, not alias the source; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "readback: [1, 2, 3]\nafter source write: [1, 2, 3]\nafter box write: src=[99, 2, 3] box=[1, 77, 3]\nafter source clear: [1, 77, 3]\ndeep after elem write: 1 [1, 2]\ndeep after elem replace: 2 [1, 2]\nstrings: aa [\"aa\", \"bb\"]\ninto: [5, 6]\nafter source drop: [10, 20, 30]\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_COUNTED_SRC: &str = r#"
use std::cell::Cell;
use std::rc::{Rc, Weak};
use std::sync::Arc;

fn main() {
    // Reference-counted elements clone as handles on the
    // same allocation, never as deep copies of the counted
    // allocation.
    let src: Vec<Rc<Cell<i32>>> = vec![Rc::new(Cell::new(1)), Rc::new(Cell::new(2))];
    let boxed: Box<[Rc<Cell<i32>>]> = Box::from(&src[..]);
    println!("rc ptr_eq: {}", Rc::ptr_eq(&src[0], &boxed[0]));
    boxed[0].set(42);
    println!("rc shared mutation: src[0]={}", src[0].get());
    println!("rc strong: {}", Rc::strong_count(&src[0]));
    drop(boxed);
    println!("rc strong after box drop: {}", Rc::strong_count(&src[0]));
    println!("rc still alive: {}", src[0].get());

    let asrc: Vec<Arc<Cell<i32>>> = vec![Arc::new(Cell::new(7))];
    let aboxed: Box<[Arc<Cell<i32>>]> = Box::from(&asrc[..]);
    println!("arc ptr_eq: {}", Arc::ptr_eq(&asrc[0], &aboxed[0]));
    aboxed[0].set(9);
    println!("arc shared mutation: asrc[0]={}", asrc[0].get());
    println!("arc strong: {}", Arc::strong_count(&asrc[0]));

    let wsrc: Vec<Weak<Cell<i32>>> = vec![Rc::downgrade(&src[1])];
    println!("weak_count before: {}", Rc::weak_count(&src[1]));
    let wboxed: Box<[Weak<Cell<i32>>]> = Box::from(&wsrc[..]);
    println!("weak_count after: {}", Rc::weak_count(&src[1]));
    println!("weak upgrade ok: {}", wboxed[0].upgrade().map(|r| r.get()) == Some(2));
    drop(wboxed);
    println!("weak_count after drop: {}", Rc::weak_count(&src[1]));

    let bsrc: Vec<Box<Rc<Cell<i32>>>> = vec![Box::new(Rc::new(Cell::new(5)))];
    let bboxed: Box<[Box<Rc<Cell<i32>>>]> = Box::from(&bsrc[..]);
    println!("box-rc ptr_eq: {}", Rc::ptr_eq(&bsrc[0], &bboxed[0]));
    bboxed[0].set(6);
    println!("box-rc shared mutation: bsrc[0]={}", bsrc[0].get());
    println!("box-rc strong: {}", Rc::strong_count(&bsrc[0]));
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_keeps_counted_identity() {
    let dir = temp_dir("boxed-from-slice-ref-counted");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_counted.rs",
        BOXED_SLICE_FROM_SLICE_REF_COUNTED_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[Rc<T>]> from &[Rc<T>] must share the counted allocations; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "rc ptr_eq: true\nrc shared mutation: src[0]=42\nrc strong: 2\nrc strong after box drop: 1\nrc still alive: 42\narc ptr_eq: true\narc shared mutation: asrc[0]=9\narc strong: 2\nweak_count before: 1\nweak_count after: 2\nweak upgrade ok: true\nweak_count after drop: 1\nbox-rc ptr_eq: true\nbox-rc shared mutation: bsrc[0]=6\nbox-rc strong: 2\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_NESTED_COUNTED_SRC: &str = r#"use std::borrow::Cow;
use std::cell::Cell;
use std::rc::Rc;

#[derive(Clone)]
struct Wrap {
    rc: Rc<Cell<i32>>,
    n: i32,
}

fn main() {
    // Tuple elements: the counted handle sits inside an
    // aggregate, in either field position.
    let src: Vec<(Rc<Cell<i32>>,)> = vec![(Rc::new(Cell::new(1)),), (Rc::new(Cell::new(2)),)];
    let boxed: Box<[(Rc<Cell<i32>>,)]> = Box::from(&src[..]);
    println!("tuple1 ptr_eq: {}", Rc::ptr_eq(&src[0].0, &boxed[0].0));
    boxed[0].0.set(42);
    println!("tuple1 shared mutation: src[0]={}", src[0].0.get());
    println!("tuple1 strong: {}", Rc::strong_count(&src[0].0));
    drop(boxed);
    println!("tuple1 strong after drop: {}", Rc::strong_count(&src[0].0));

    let psrc: Vec<(i32, Rc<Cell<i32>>)> = vec![(7, Rc::new(Cell::new(2))), (8, Rc::new(Cell::new(3)))];
    let pboxed: Box<[(i32, Rc<Cell<i32>>)]> = Box::from(&psrc[..]);
    println!("tuple2 ptr_eq: {}", Rc::ptr_eq(&psrc[0].1, &pboxed[0].1));
    println!("tuple2 plain field: {}", pboxed[0].0);
    pboxed[0].1.set(4);
    println!("tuple2 shared mutation: psrc[0]={}", psrc[0].1.get());

    // Option element holding a counted handle.
    let osrc: Vec<Option<Rc<Cell<i32>>>> = vec![Some(Rc::new(Cell::new(7))), None];
    let oboxed: Box<[Option<Rc<Cell<i32>>>]> = Box::from(&osrc[..]);
    let (s, b) = (osrc[0].as_ref().unwrap(), oboxed[0].as_ref().unwrap());
    println!("option ptr_eq: {}", Rc::ptr_eq(s, b));
    b.set(9);
    println!("option shared mutation: osrc[0]={}", s.get());
    println!("option strong: {}", Rc::strong_count(s));
    println!("option none: {}", oboxed[1].is_none());

    // Vec element: its elements share identity, its storage
    // does not alias the source's.
    let vsrc: Vec<Vec<Rc<Cell<i32>>>> = vec![vec![Rc::new(Cell::new(3))]];
    let mut vboxed: Box<[Vec<Rc<Cell<i32>>>]> = Box::from(&vsrc[..]);
    println!("vec ptr_eq: {}", Rc::ptr_eq(&vsrc[0][0], &vboxed[0][0]));
    vboxed[0][0].set(11);
    println!("vec shared mutation: vsrc[0][0]={}", vsrc[0][0].get());
    println!("vec strong: {}", Rc::strong_count(&vsrc[0][0]));
    vboxed[0].push(Rc::new(Cell::new(4)));
    println!("vec lens: src={} boxed={}", vsrc[0].len(), vboxed[0].len());

    // Struct element with a counted field.
    let wsrc: Vec<Wrap> = vec![Wrap { rc: Rc::new(Cell::new(5)), n: 1 }];
    let wboxed: Box<[Wrap]> = Box::from(&wsrc[..]);
    println!("struct ptr_eq: {}", Rc::ptr_eq(&wsrc[0].rc, &wboxed[0].rc));
    wboxed[0].rc.set(6);
    println!("struct shared mutation: wsrc[0]={}", wsrc[0].rc.get());
    println!("struct strong: {}", Rc::strong_count(&wsrc[0].rc));
    println!("struct plain field: {}", wboxed[0].n);

    // Array element, judged through a second conversion of
    // the same source: the two boxes share the counted
    // allocations iff the element clone preserves identity.
    let asrc: Vec<[Rc<Cell<i32>>; 2]> = vec![[Rc::new(Cell::new(8)), Rc::new(Cell::new(9))]];
    let b1: Box<[[Rc<Cell<i32>>; 2]]> = Box::from(&asrc[..]);
    let b2: Box<[[Rc<Cell<i32>>; 2]]> = Box::from(&asrc[..]);
    println!("array ptr_eq: {}", Rc::ptr_eq(&b1[0][1], &b2[0][1]));
    b1[0][1].set(10);
    println!("array shared mutation: b2[0][1]={}", b2[0][1].get());
    println!("array ptr_eq first: {}", Rc::ptr_eq(&b1[0][0], &b2[0][0]));

    // The borrowed-Cow conversion clones its elements the
    // same way.
    let csrc: Vec<(Rc<Cell<i32>>,)> = vec![(Rc::new(Cell::new(12)),), (Rc::new(Cell::new(13)),)];
    let cow: Cow<'_, [(Rc<Cell<i32>>,)]> = Cow::Borrowed(&csrc[..]);
    let cboxed: Box<[(Rc<Cell<i32>>,)]> = Box::from(cow);
    println!("cow ptr_eq: {}", Rc::ptr_eq(&csrc[0].0, &cboxed[0].0));
    println!("cow strong: {}", Rc::strong_count(&csrc[0].0));
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_keeps_nested_counted_identity() {
    let dir = temp_dir("boxed-from-slice-ref-nested-counted");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_nested_counted.rs",
        BOXED_SLICE_FROM_SLICE_REF_NESTED_COUNTED_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T]> from &[T] must preserve counted-handle identity inside aggregate elements; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "tuple1 ptr_eq: true\ntuple1 shared mutation: src[0]=42\ntuple1 strong: 2\ntuple1 strong after drop: 1\ntuple2 ptr_eq: true\ntuple2 plain field: 7\ntuple2 shared mutation: psrc[0]=4\noption ptr_eq: true\noption shared mutation: osrc[0]=9\noption strong: 2\noption none: true\nvec ptr_eq: true\nvec shared mutation: vsrc[0][0]=11\nvec strong: 2\nvec lens: src=1 boxed=2\nstruct ptr_eq: true\nstruct shared mutation: wsrc[0]=6\nstruct strong: 2\nstruct plain field: 1\narray ptr_eq: true\narray shared mutation: b2[0][1]=10\narray ptr_eq first: true\ncow ptr_eq: true\ncow strong: 2\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_BOXED_SLICE_SRC: &str = r#"use std::cell::Cell;
use std::rc::Rc;

#[derive(Debug)]
struct S {
    v: i32,
}

thread_local! {
    static CLONES: Cell<i32> = const { Cell::new(0) };
}

impl Clone for S {
    fn clone(&self) -> Self {
        CLONES.with(|c| c.set(c.get() + 1));
        S { v: self.v + 100 }
    }
}

fn main() {
    // A `Box<[T]>` element: the pointee's tag is a slice tag,
    // and its elements clone with their own types judged.
    let inner: Box<[Rc<Cell<i32>>]> =
        Box::from(&[Rc::new(Cell::new(1)), Rc::new(Cell::new(2))][..]);
    let src: Vec<Box<[Rc<Cell<i32>>]>> = vec![inner];
    let keep = src[0][0].clone();
    let boxed: Box<[Box<[Rc<Cell<i32>>]>]> = Box::from(&src[..]);
    println!("boxslice ptr_eq: {}", Rc::ptr_eq(&keep, &boxed[0][0]));
    boxed[0][0].set(42);
    println!("boxslice shared: keep={}", keep.get());
    println!("boxslice strong: {}", Rc::strong_count(&keep));
    println!("boxslice len: {}", boxed[0].len());
    println!("boxslice second: {}", boxed[0][1].get());

    // The same through a pointee built by into_boxed_slice
    // (the vector's value form under the slice tag).
    let vinner: Box<[Rc<Cell<i32>>]> =
        vec![Rc::new(Cell::new(3)), Rc::new(Cell::new(4))].into_boxed_slice();
    let vsrc: Vec<Box<[Rc<Cell<i32>>]>> = vec![vinner];
    let vkeep = vsrc[0][1].clone();
    let vboxed: Box<[Box<[Rc<Cell<i32>>]>]> = Box::from(&vsrc[..]);
    println!("vecform ptr_eq: {}", Rc::ptr_eq(&vkeep, &vboxed[0][1]));
    println!("vecform strong: {}", Rc::strong_count(&vkeep));

    // Custom Clone elements inside the slice pointee run
    // their Clone bodies, one per element per conversion.
    let sinner: Box<[S]> = Box::from(&[S { v: 1 }, S { v: 2 }][..]);
    let ssrc: Vec<Box<[S]>> = vec![sinner];
    let sboxed: Box<[Box<[S]>]> = Box::from(&ssrc[..]);
    println!("custom clones run: {}", CLONES.with(|c| c.get()));
    println!("custom values: {} {}", sboxed[0][0].v, sboxed[0][1].v);
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_boxed_slice_elements() {
    let dir = temp_dir("boxed-from-slice-ref-boxed-slice");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_boxed_slice.rs",
        BOXED_SLICE_FROM_SLICE_REF_BOXED_SLICE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[Box<[T]>]> from &[Box<[T]>] must clone the pointee's elements with their types judged; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "boxslice ptr_eq: true\nboxslice shared: keep=42\nboxslice strong: 3\nboxslice len: 2\nboxslice second: 2\nvecform ptr_eq: true\nvecform strong: 3\ncustom clones run: 4\ncustom values: 201 202\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_CLOSURE_SRC: &str = r#"use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    // A closure element capturing a Box<Cell<i32>> by value:
    // the boxed clone's cell is independent of the source's.
    let cell = Box::new(Cell::new(10));
    let f = move || {
        cell.set(cell.get() + 1);
        cell.get()
    };
    let src = [f];
    let boxed: Box<[_]> = Box::from(&src[..]);
    println!("boxcapture src call: {}", src[0]());
    println!("boxcapture boxed call: {}", boxed[0]());
    println!("boxcapture src call 2: {}", src[0]());

    // A closure element capturing an Rc<Cell<i32>> by value:
    // the conversion moves the strong count.
    let rc = Rc::new(Cell::new(7));
    let rc_in = rc.clone();
    let g = move || rc_in.get();
    let gsrc = [g];
    println!("rccapture strong before: {}", Rc::strong_count(&rc));
    let gboxed: Box<[_]> = Box::from(&gsrc[..]);
    println!("rccapture strong after: {}", Rc::strong_count(&rc));
    println!("rccapture boxed call: {}", gboxed[0]());
    drop(gboxed);
    println!("rccapture strong after drop: {}", Rc::strong_count(&rc));

    // A direct .clone() of a closure value clones its
    // environment the same way.
    let dcell = Box::new(Cell::new(10));
    let df = move || {
        dcell.set(dcell.get() + 1);
        dcell.get()
    };
    let df2 = df.clone();
    println!("direct src call: {}", df());
    println!("direct clone call: {}", df2());
    let drc = Rc::new(Cell::new(7));
    let drc_in = drc.clone();
    let dg = move || drc_in.get();
    println!("direct strong before: {}", Rc::strong_count(&drc));
    let dg2 = dg.clone();
    println!("direct strong after clone: {}", Rc::strong_count(&drc));
    drop(dg2);
    println!("direct strong after drop: {}", Rc::strong_count(&drc));
    println!("direct value: {}", dg());

    // A by-reference capture: the clone shares the referent.
    let shared = Cell::new(5);
    let r = &shared;
    let rf = move || r.get();
    let rsrc = [rf];
    let rboxed: Box<[_]> = Box::from(&rsrc[..]);
    shared.set(9);
    println!("refcapture boxed: {}", rboxed[0]());
    println!("refcapture src: {}", rsrc[0]());

    // An Arc capture: the count moves.
    let arc = Arc::new(Cell::new(3));
    let arc_in = arc.clone();
    let ag = move || arc_in.get();
    let agsrc = [ag];
    let agboxed: Box<[_]> = Box::from(&agsrc[..]);
    println!("arccapture strong: {}", Arc::strong_count(&arc));
    println!("arccapture boxed call: {}", agboxed[0]());

    // A closure nested in a tuple element.
    let tcell = Box::new(Cell::new(20));
    let k = move || {
        tcell.set(tcell.get() + 1);
        tcell.get()
    };
    let tsrc = [(k, 7)];
    let tboxed: Box<[_]> = Box::from(&tsrc[..]);
    println!("tuple src call: {}", (tsrc[0].0)());
    println!("tuple boxed call: {}", (tboxed[0].0)());
    println!("tuple plain: {}", tboxed[0].1);

    // A closure capturing another closure: the inner
    // environment clones recursively.
    let ncell = Rc::new(Cell::new(1));
    let ncell_in = ncell.clone();
    let inner = move || ncell_in.get();
    let outer = move || inner();
    let osrc = [outer];
    println!("nested strong before: {}", Rc::strong_count(&ncell));
    let oboxed: Box<[_]> = Box::from(&osrc[..]);
    println!("nested strong after: {}", Rc::strong_count(&ncell));
    println!("nested boxed call: {}", oboxed[0]());
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_closure_elements() {
    let dir = temp_dir("boxed-from-slice-ref-closure");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_closure.rs",
        BOXED_SLICE_FROM_SLICE_REF_CLOSURE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "cloning a closure element must clone its capture environment; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "boxcapture src call: 11\nboxcapture boxed call: 11\nboxcapture src call 2: 12\nrccapture strong before: 2\nrccapture strong after: 3\nrccapture boxed call: 7\nrccapture strong after drop: 2\ndirect src call: 11\ndirect clone call: 11\ndirect strong before: 2\ndirect strong after clone: 3\ndirect strong after drop: 2\ndirect value: 7\nrefcapture boxed: 9\nrefcapture src: 9\narccapture strong: 3\narccapture boxed call: 3\ntuple src call: 21\ntuple boxed call: 21\ntuple plain: 7\nnested strong before: 2\nnested strong after: 3\nnested boxed call: 1\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_CHANNEL_SRC: &str = r#"use std::sync::mpsc::{channel, sync_channel, TryRecvError};

fn state(r: &std::sync::mpsc::Receiver<i32>) -> &'static str {
    match r.try_recv() {
        Ok(_) => "got",
        Err(TryRecvError::Empty) => "empty",
        Err(TryRecvError::Disconnected) => "disconnected",
    }
}

fn main() {
    // A Sender element: the boxed clone is another live
    // handle on the same channel.
    let (tx, rx) = channel::<i32>();
    let src = vec![tx];
    let boxed: Box<[_]> = Box::from(&src[..]);
    drop(boxed);
    println!("sender after box drop: {}", state(&rx));
    src[0].send(42).unwrap();
    println!("sender recv: {}", rx.recv().unwrap());
    drop(src);
    println!("sender after src drop: {}", state(&rx));

    // A SyncSender element, the same discipline.
    let (stx, srx) = sync_channel::<i32>(4);
    let ssrc = vec![stx];
    let sboxed: Box<[_]> = Box::from(&ssrc[..]);
    drop(sboxed);
    println!("sync after box drop: {}", state(&srx));
    ssrc[0].send(7).unwrap();
    println!("sync recv: {}", srx.recv().unwrap());

    // A sender nested in a tuple element: the count moves
    // through the aggregate clone as well.
    let (ntx, nrx) = channel::<i32>();
    let nsrc = vec![(ntx,)];
    let nboxed: Box<[_]> = Box::from(&nsrc[..]);
    drop(nboxed);
    println!("nested after box drop: {}", state(&nrx));
    drop(nsrc);
    println!("nested after src drop: {}", state(&nrx));
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_keeps_channel_handles() {
    let dir = temp_dir("boxed-from-slice-ref-channel");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_channel.rs",
        BOXED_SLICE_FROM_SLICE_REF_CHANNEL_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[Sender<T>]> from &[Sender<T>] must clone the channel handles, not copy them; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "sender after box drop: empty\nsender recv: 42\nsender after src drop: disconnected\nsync after box drop: empty\nsync recv: 7\nnested after box drop: empty\nnested after src drop: disconnected\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_WAKER_SRC: &str = r#"use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{RawWaker, RawWakerVTable, Wake, Waker};

static DROPS: AtomicUsize = AtomicUsize::new(0);
static WAKES: AtomicUsize = AtomicUsize::new(0);

struct Counted;
impl Drop for Counted {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe fn clone_fn(data: *const ()) -> RawWaker {
    let arc = Arc::from_raw(data as *const Counted);
    let cloned = arc.clone();
    let raw = Arc::into_raw(cloned) as *const ();
    std::mem::forget(arc);
    RawWaker::new(raw, &VTABLE)
}
unsafe fn wake_fn(data: *const ()) {
    let arc = Arc::from_raw(data as *const Counted);
    drop(arc);
}
unsafe fn wake_by_ref_fn(data: *const ()) {
    let arc = Arc::from_raw(data as *const Counted);
    std::mem::forget(arc);
}
unsafe fn drop_fn(data: *const ()) {
    let arc = Arc::from_raw(data as *const Counted);
    drop(arc);
}

static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_fn, wake_fn, wake_by_ref_fn, drop_fn);

fn make_waker(probe: &Arc<Counted>) -> Waker {
    let raw = Arc::into_raw(probe.clone()) as *const ();
    unsafe { Waker::from_raw(RawWaker::new(raw, &VTABLE)) }
}

struct MyWake;
impl Wake for MyWake {
    fn wake(self: Arc<Self>) {
        WAKES.fetch_add(1, Ordering::SeqCst);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        WAKES.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    // A from_raw Waker element: the boxed clone must run
    // the vtable clone entry, moving the Arc count.
    let probe = Arc::new(Counted);
    let src = [make_waker(&probe)];
    println!("base count: {}", Arc::strong_count(&probe));
    let boxed: Box<[Waker]> = Box::from(&src[..]);
    println!("after box count: {}", Arc::strong_count(&probe));
    boxed[0].wake_by_ref();
    println!("wake_by_ref ok, count: {}", Arc::strong_count(&probe));
    drop(boxed);
    println!("after drop-box count: {}", Arc::strong_count(&probe));
    println!("pointee drops so far: {}", DROPS.load(Ordering::SeqCst));
    drop(src);
    println!("after drop-src count: {}", Arc::strong_count(&probe));

    // A from_raw Waker nested in a tuple element.
    let probe2 = Arc::new(Counted);
    let src2 = [(make_waker(&probe2), 7u32)];
    let boxed2: Box<[(Waker, u32)]> = Box::from(&src2[..]);
    println!("tuple: after box count: {}", Arc::strong_count(&probe2));
    drop(boxed2);
    println!("tuple: after drop-box count: {}", Arc::strong_count(&probe2));
    println!("tuple: pointee drops so far: {}", DROPS.load(Ordering::SeqCst));
    drop(src2);

    // A modelled From<Arc<T: Wake>> Waker element: the
    // boxed clone is another live handle on the same Arc.
    let arc = Arc::new(MyWake);
    let w: Waker = Waker::from(arc.clone());
    println!("modelled base count: {}", Arc::strong_count(&arc));
    let src3 = [w];
    let boxed3: Box<[Waker]> = Box::from(&src3[..]);
    println!("modelled after box count: {}", Arc::strong_count(&arc));
    boxed3[0].wake_by_ref();
    println!("modelled wakes: {}", WAKES.load(Ordering::SeqCst));
    drop(boxed3);
    println!("modelled after drop-box count: {}", Arc::strong_count(&arc));
    src3[0].wake_by_ref();
    println!("modelled wakes: {}", WAKES.load(Ordering::SeqCst));
    drop(src3);
    println!("modelled after drop-src count: {}", Arc::strong_count(&arc));
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_keeps_waker_handles() {
    let dir = temp_dir("boxed-from-slice-ref-waker");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_waker.rs",
        BOXED_SLICE_FROM_SLICE_REF_WAKER_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[Waker]> from &[Waker] must clone the wakers, not copy them; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "base count: 2\nafter box count: 3\nwake_by_ref ok, count: 3\nafter drop-box count: 2\npointee drops so far: 0\nafter drop-src count: 1\ntuple: after box count: 3\ntuple: after drop-box count: 2\ntuple: pointee drops so far: 0\nmodelled base count: 2\nmodelled after box count: 3\nmodelled wakes: 1\nmodelled after drop-box count: 2\nmodelled wakes: 2\nmodelled after drop-src count: 1\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_CUSTOM_CLONE_SRC: &str = r#"
#[derive(Debug)]
struct S(i32);
impl Clone for S {
    fn clone(&self) -> Self {
        S(self.0 + 100)
    }
}

fn main() {
    let src = [S(1), S(2), S(3)];
    let boxed: Box<[S]> = Box::from(&src[..]);
    println!("boxed: {:?}", boxed);
    let cow: std::borrow::Cow<[S]> = std::borrow::Cow::Borrowed(&src[..]);
    let boxed2: Box<[S]> = Box::from(cow);
    println!("cow boxed: {:?}", boxed2);
    // Nested: a tuple element carrying a custom-Clone field.
    let src3 = [(S(7), 1u32)];
    let boxed3: Box<[(S, u32)]> = Box::from(&src3[..]);
    println!("tuple boxed: {:?}", boxed3);
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_runs_custom_clone() {
    let dir = temp_dir("boxed-from-slice-ref-custom-clone");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_custom_clone.rs",
        BOXED_SLICE_FROM_SLICE_REF_CUSTOM_CLONE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T]> from &[T] must run the elements' own Clone impls; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "boxed: [S(101), S(102), S(103)]\ncow boxed: [S(101), S(102), S(103)]\ntuple boxed: [(S(107), 1)]\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_CUSTOM_CLONE_SOURCE_REF_SRC: &str = r#"
use std::cell::{Cell, RefCell};

struct S {
    counter: Cell<i32>,
}

impl Clone for S {
    fn clone(&self) -> Self {
        self.counter.set(self.counter.get() + 1);
        S { counter: Cell::new(self.counter.get()) }
    }
}

struct R {
    log: RefCell<Vec<i32>>,
}

impl Clone for R {
    fn clone(&self) -> Self {
        self.log.borrow_mut().push(1);
        R { log: RefCell::new(self.log.borrow().clone()) }
    }
}

fn main() {
    // The conversion's source slice is place-backed here, so
    // the elements' Clone bodies receive the source elements
    // themselves: the mutations they make through &self land
    // on the source, as a direct clone() call's would.
    let src = [S { counter: Cell::new(10) }];
    let s: &[S] = &src;
    let boxed: Box<[S]> = Box::from(s);
    println!("src={} boxed={}", src[0].counter.get(), boxed[0].counter.get());
    // Through aggregate recursion: a tuple element's field.
    let src2 = [(S { counter: Cell::new(20) }, 5u32)];
    let s2: &[(S, u32)] = &src2;
    let boxed2: Box<[(S, u32)]> = Box::from(s2);
    println!(
        "nested src={} boxed={}",
        src2[0].0.counter.get(),
        boxed2[0].0.counter.get()
    );
    // A RefCell mutation through &self.
    let src3 = [R { log: RefCell::new(vec![]) }];
    let s3: &[R] = &src3;
    let boxed3: Box<[R]> = Box::from(s3);
    println!(
        "refcell src len={} first={} boxed len={} first={}",
        src3[0].log.borrow().len(),
        src3[0].log.borrow()[0],
        boxed3[0].log.borrow().len(),
        boxed3[0].log.borrow()[0]
    );
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_custom_clone_gets_source_ref() {
    let dir = temp_dir("boxed-from-slice-ref-custom-clone-source-ref");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_custom_clone_source_ref.rs",
        BOXED_SLICE_FROM_SLICE_REF_CUSTOM_CLONE_SOURCE_REF_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a custom Clone that mutates through &self must run against the source element; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "src=11 boxed=11\nnested src=21 boxed=21\nrefcell src len=1 first=1 boxed len=1 first=1\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_CLONE_READS_FIELDS_LIVE_SRC: &str = r#"
use std::cell::Cell;

struct Bump {
    target: *const Cell<i32>,
}

impl Clone for Bump {
    fn clone(&self) -> Self {
        unsafe {
            if !self.target.is_null() {
                (*self.target).set(42);
            }
        }
        Bump { target: self.target }
    }
}

struct E {
    target: *const Cell<i32>,
    cell: Cell<i32>,
}

impl Clone for E {
    fn clone(&self) -> Self {
        unsafe {
            if !self.target.is_null() {
                (*self.target).set(43);
            }
        }
        E { target: self.target, cell: Cell::new(self.cell.get()) }
    }
}

#[derive(Clone)]
struct Pair {
    bump: Bump,
    cell: Cell<i32>,
}

fn main() {
    // 1. Tuple element: field 0's Clone mutates field 1 (a
    //    Cell) through a pointer to the tuple's own second
    //    field. The field-wise construction reads field 1
    //    after field 0's clone ran, so the clone holds 42.
    {
        let mut src = [(Bump { target: std::ptr::null() }, Cell::new(0))];
        src[0].0.target = &src[0].1 as *const Cell<i32>;
        let s: &[(Bump, Cell<i32>)] = &src;
        let boxed: Box<[(Bump, Cell<i32>)]> = Box::from(s);
        println!("tuple boxed={} src={}", boxed[0].1.get(), src[0].1.get());
    }
    // 2. Array element: item 0's Clone mutates item 1's cell.
    {
        let mut src = [[
            E { target: std::ptr::null(), cell: Cell::new(0) },
            E { target: std::ptr::null(), cell: Cell::new(0) },
        ]];
        src[0][0].target = &src[0][1].cell as *const Cell<i32>;
        let s: &[[E; 2]] = &src;
        let boxed: Box<[[E; 2]]> = Box::from(s);
        println!("array boxed={} src={}", boxed[0][1].cell.get(), src[0][1].cell.get());
    }
    // 3. Nested tuple: the mutation crosses one structural
    //    level (the outer field 0 is an inner tuple whose
    //    field 0 mutates the outer field 1).
    {
        let mut src = [((Bump { target: std::ptr::null() }, 7u32), Cell::new(0))];
        src[0].0 .0.target = &src[0].1 as *const Cell<i32>;
        let s: &[((Bump, u32), Cell<i32>)] = &src;
        let boxed: Box<[((Bump, u32), Cell<i32>)]> = Box::from(s);
        println!("nested boxed={} src={}", boxed[0].1.get(), src[0].1.get());
    }
    // 4. Cross-element: element 0's field clone mutates
    //    element 1's cell field in the source slice.
    {
        let mut src = [
            (Bump { target: std::ptr::null() }, Cell::new(0)),
            (Bump { target: std::ptr::null() }, Cell::new(0)),
        ];
        src[0].0.target = &src[1].1 as *const Cell<i32>;
        let s: &[(Bump, Cell<i32>)] = &src;
        let boxed: Box<[(Bump, Cell<i32>)]> = Box::from(s);
        println!("cross boxed={} src={}", boxed[1].1.get(), src[1].1.get());
    }
    // 5. Struct element: the first field's Clone mutates the
    //    second field.
    {
        let mut src = [Pair { bump: Bump { target: std::ptr::null() }, cell: Cell::new(0) }];
        src[0].bump.target = &src[0].cell as *const Cell<i32>;
        let s: &[Pair] = &src;
        let boxed: Box<[Pair]> = Box::from(s);
        println!("struct boxed={} src={}", boxed[0].cell.get(), src[0].cell.get());
    }
    // 6. Vec element: item 0's field clone mutates item 1's
    //    cell inside the vector.
    {
        let mut src = [vec![
            (Bump { target: std::ptr::null() }, Cell::new(0)),
            (Bump { target: std::ptr::null() }, Cell::new(0)),
        ]];
        src[0][0].0.target = &src[0][1].1 as *const Cell<i32>;
        let s: &[Vec<(Bump, Cell<i32>)>] = &src;
        let boxed: Box<[Vec<(Bump, Cell<i32>)>]> = Box::from(s);
        println!("vec boxed={} src={}", boxed[0][1].1.get(), src[0][1].1.get());
    }
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_clone_reads_fields_live() {
    let dir = temp_dir("boxed-from-slice-ref-clone-reads-fields-live");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_clone_reads_fields_live.rs",
        BOXED_SLICE_FROM_SLICE_REF_CLONE_READS_FIELDS_LIVE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a field cloned after a sibling's Clone rewrote it must observe the mutation; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "tuple boxed=42 src=42\narray boxed=43 src=43\nnested boxed=42 src=42\ncross boxed=42 src=42\nstruct boxed=42 src=42\nvec boxed=42 src=42\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_CLONE_NAN_ELEMENT_SRC: &str = r#"
use std::cell::Cell;

// An element whose Clone body updates its own inline Cell
// through `&self`, and which also carries an f64.
struct Elem {
    cell: Cell<i32>,
    x: f64,
}

impl Clone for Elem {
    fn clone(&self) -> Self {
        self.cell.set(42);
        Elem { cell: Cell::new(self.cell.get()), x: self.x }
    }
}

// Same, with the float nested one level down (inside a tuple
// field of the element).
struct Nested {
    cell: Cell<i32>,
    pair: (f64, u32),
}

impl Clone for Nested {
    fn clone(&self) -> Self {
        self.cell.set(43);
        Nested { cell: Cell::new(self.cell.get()), pair: self.pair }
    }
}

// Same, with an f32 payload.
struct Elem32 {
    cell: Cell<i32>,
    x: f32,
}

impl Clone for Elem32 {
    fn clone(&self) -> Self {
        self.cell.set(44);
        Elem32 { cell: Cell::new(self.cell.get()), x: self.x }
    }
}

fn main() {
    // 1. NaN directly in the cloned element: the element read
    //    back from its own storage must still denote the
    //    snapshot, so the Clone body runs on the live element
    //    and its Cell update lands in the source.
    {
        let src = [Elem { cell: Cell::new(0), x: f64::NAN }];
        let s: &[Elem] = &src;
        let boxed: Box<[Elem]> = Box::from(s);
        println!(
            "nan src={} boxed={} is_nan={}",
            src[0].cell.get(),
            boxed[0].cell.get(),
            boxed[0].x.is_nan()
        );
    }
    // 2. NaN nested inside a tuple field of the element.
    {
        let src = [Nested { cell: Cell::new(0), pair: (f64::NAN, 7) }];
        let s: &[Nested] = &src;
        let boxed: Box<[Nested]> = Box::from(s);
        println!(
            "nested src={} boxed={} is_nan={}",
            src[0].cell.get(),
            boxed[0].cell.get(),
            boxed[0].pair.0.is_nan()
        );
    }
    // 3. -0.0 control: sign-zero float, no NaN.
    {
        let src = [Elem { cell: Cell::new(0), x: -0.0 }];
        let s: &[Elem] = &src;
        let boxed: Box<[Elem]> = Box::from(s);
        println!(
            "negzero src={} boxed={} sign={}",
            src[0].cell.get(),
            boxed[0].cell.get(),
            boxed[0].x.is_sign_negative()
        );
    }
    // 4. Ordinary float control.
    {
        let src = [Elem { cell: Cell::new(0), x: 1.5 }];
        let s: &[Elem] = &src;
        let boxed: Box<[Elem]> = Box::from(s);
        println!(
            "plain src={} boxed={} x={}",
            src[0].cell.get(),
            boxed[0].cell.get(),
            boxed[0].x
        );
    }
    // 5. f32 NaN in the cloned element.
    {
        let src = [Elem32 { cell: Cell::new(0), x: f32::NAN }];
        let s: &[Elem32] = &src;
        let boxed: Box<[Elem32]> = Box::from(s);
        println!(
            "f32nan src={} boxed={} is_nan={}",
            src[0].cell.get(),
            boxed[0].cell.get(),
            boxed[0].x.is_nan()
        );
    }
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_clone_nan_element_keeps_live_ref() {
    let dir = temp_dir("boxed-from-slice-ref-clone-nan-element");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_clone_nan_element.rs",
        BOXED_SLICE_FROM_SLICE_REF_CLONE_NAN_ELEMENT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an element containing NaN must keep its live source reference, so its Clone body's Cell update lands in the source; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "nan src=42 boxed=42 is_nan=true\nnested src=43 boxed=43 is_nan=true\nnegzero src=42 boxed=42 sign=true\nplain src=42 boxed=42 x=1.5\nf32nan src=44 boxed=44 is_nan=true\n"
    );
}

#[test]
fn run_mode_boxed_slice_collect() {
    let dir = temp_dir("boxed-collect");
    let src = fixture(&dir, "boxed_collect.rs", BOXED_SLICE_COLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "collect into Box<[T]> must work; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[0, 1, 2, 3, 4]\n[\"a\", \"bb\"]\n[] 0\nboxed collect ok\n"
    );
}

const BOXED_SLICE_INTO_ASREF_SRC: &str = r#"
#![feature(box_into_boxed_slice)]
fn main() {
    // `Box::into_boxed_slice` is a guest body: the box it
    // leaves behind is the real aggregate form (the
    // Box / Unique / NonNull wrapper chain around the fat
    // pointer), not the one-field aggregate the `Box::new`
    // builtin builds. `AsRef` / `AsMut` must still extract
    // the stored pointer from the chain's bottom.
    let b: Box<[u8]> = Box::into_boxed_slice(Box::new(5u8));
    let s: &[u8] = b.as_ref();
    println!("len={} first={}", s.len(), s[0]);
    let mut b2: Box<[u8]> = Box::into_boxed_slice(Box::new(7u8));
    let m: &mut [u8] = b2.as_mut();
    m[0] = 9;
    let s2: &[u8] = b2.as_ref();
    println!("after={}", s2[0]);
    let c: Box<[[u8; 1]]> = Box::into_boxed_slice(Box::new([25u8]));
    let t: &[[u8; 1]] = c.as_ref();
    println!("nested={} {}", t.len(), t[0][0]);
    println!("boxed slice asref ok");
}
"#;

#[test]
fn run_mode_boxed_slice_into_boxed_slice_asref() {
    let dir = temp_dir("boxed-into-asref");
    let src = fixture(&dir, "boxed_into_asref.rs", BOXED_SLICE_INTO_ASREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "AsRef/AsMut on a guest-built into_boxed_slice box must extract the wrapper chain's fat pointer; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=1 first=5\nafter=9\nnested=1 25\nboxed slice asref ok\n"
    );
}

const INTO_BOXED_SLICE_SRC: &str = r#"
#![feature(box_into_boxed_slice)]
fn main() {
    let b = Box::into_boxed_slice(Box::new(5u8));
    assert_eq!(b, Box::new([5u8]) as Box<[u8]>);
    let b2 = Box::into_boxed_slice(Box::new([25u8]));
    assert_eq!(b2, Box::new([[25u8]]) as Box<[[u8; 1]]>);
    let b3 = Box::into_boxed_slice(Box::new(Box::new([5u8])));
    assert_eq!(b3, Box::new([Box::new([5u8])]) as Box<[Box<[u8; 1]>]>);
    println!("{:?}", b);
    let s: &[u8] = &b[..];
    assert_eq!(s, &[5u8][..]);
    println!("sum {}", b.iter().sum::<u8>());
}
"#;

#[test]
fn run_mode_into_boxed_slice_scalar_storage() {
    // `Box::into_boxed_slice` on a scalar box leaves the
    // scalar itself in the allocation, denominated as a
    // one-element slice only by the fat pointer over it:
    // equality against an array-built box, the Debug form,
    // the `[..]` subslice and iteration must all read the
    // one element, not the scalar as the whole value (the
    // rustc suite's `box/into-boxed-slice` shape).
    run_ok(
        "into-boxed-slice",
        INTO_BOXED_SLICE_SRC,
        "[5]\nsum 5\n",
    );
}
