// Vec and slice iterator tests (collect, size_hint, into_iter).

const REF_VEC_ITER_SRC: &str = r#"
fn main() {
    let v = vec![10, 20, 30];
    let mut sum = 0;
    for x in &v {
        sum += *x;
    }
    println!("{}", sum);
}
"#;

#[test]
fn run_mode_ref_vec_iteration() {
    // `for x in &v` goes through `IntoIterator::into_iter` for
    // `&Vec<T>`, which yields a `slice::Iter` over the elements;
    // its items are `&T`, dereferenced by the loop body.
    let dir = temp_dir("refveciter");
    let src = fixture(&dir, "refveciter.rs", REF_VEC_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating &Vec must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "60\n");
}

// ----- Vec collect / size_hint / Rev (Lane GE) -------------------------------

const COLLECT_SRC: &str = r#"
fn main() {
    let v: Vec<i32> = (0..3).collect();
    println!("n {} s {}", v.len(), v[0] + v[1] + v[2]);
    let w: Vec<i32> = (0..3).rev().collect();
    println!("rev {} {}", w[0], w[2]);
    let e: Vec<i32> = (5..5).collect();
    println!("empty {}", e.len());
    let mut x: Vec<i32> = Vec::new();
    x.extend(10..12);
    println!("ext {} {}", x.len(), x[0] + x[1]);
}
"#;

#[test]
fn run_mode_collect_builds_vec_from_iterators() {
    // `Iterator::collect::<Vec<_>>` reaches
    // `SpecFromIterNested::from_iter`, whose collected body has
    // `Vec::with_capacity` inlined and drives `RawVecInner`
    // directly; the entry point is modelled by stepping the
    // iterator (see `Builtin::VecFromIterNested`), and
    // `Vec::extend` reaches the modelled `extend_trusted`.
    let dir = temp_dir("collect");
    let src = fixture(&dir, "collect.rs", COLLECT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "n 3 s 3\nrev 2 0\nempty 0\next 2 21\n",
        "stdout={stdout}"
    );
}

const ARGS_COLLECT_SRC: &str = r#"
fn main() {
    let fwd: Vec<String> = std::env::args().collect();
    println!("fwd {}", fwd.len());
    println!("fwd1 {}", fwd[1]);
    let rev: Vec<String> = std::env::args().rev().collect();
    println!("rev0 {}", rev[0]);
    println!("rev-last {}", rev[rev.len() - 1] == fwd[0]);
}
"#;

#[test]
fn run_mode_env_args_collect_and_reverse() {
    // `args().collect()` calls the iterator's `size_hint` first
    // (the modelled `Iterator::size_hint` answers for the env
    // iterators, whose real bodies read the real `vec::IntoIter`
    // layout off the modelled state). `args().rev()` is the
    // `Rev` adapter: its `next` is the inner iterator's back
    // step.
    let dir = temp_dir("args-collect");
    let src = fixture(&dir, "argscollect.rs", ARGS_COLLECT_SRC);
    let out = run_driver_extra(&src, &["--guest-arg", "one", "--guest-arg", "two"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "fwd 3\nfwd1 one\nrev0 two\nrev-last true\n",
        "stdout={stdout}"
    );
}

const SIZE_HINT_SRC: &str = r#"
fn main() {
    let (l, u) = (0..10).size_hint();
    println!("range {l} {}", u.unwrap());
    let (l, u) = (2..=5).size_hint();
    println!("incl {l} {}", u.unwrap());
    let (l, u) = "aé€".chars().size_hint();
    println!("chars {l} {}", u.unwrap());
    let (l, u) = "aé€".bytes().size_hint();
    println!("bytes {l} {}", u.unwrap());
    let (l, u) = ('a'..'z').size_hint();
    println!("charrange {l} {}", u.unwrap());
    let (l, u) = std::env::args().size_hint();
    println!("args {l} {}", u.unwrap());
    let (l, u) = (0..=u64::MAX).size_hint();
    println!("big {l} {}", u.is_none());
}
"#;

#[test]
fn run_mode_size_hint_reports_std_values() {
    // The exact pairs the standard library reports (measured
    // against the pinned toolchain): exact counts for ranges and
    // the owned iterators, the byte-count bounds for `Chars`
    // (lower = remaining bytes / 4 rounded up, upper = remaining
    // bytes), the surrogate gap removed for char ranges, and
    // `(usize::MAX, None)` for a count that overflows `usize`.
    let dir = temp_dir("size-hint");
    let src = fixture(&dir, "sizehint.rs", SIZE_HINT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "range 10 10\nincl 4 4\nchars 2 6\nbytes 6 6\ncharrange 25 25\nargs 1 1\nbig 18446744073709551615 true\n",
        "stdout={stdout}"
    );
}

const STR_ITER_WRAPPER_SRC: &str = r#"
fn main() {
    let b = "aé€b".bytes();
    println!("len {}", b.len());
    let mut b2 = "aé€b".bytes();
    b2.next();
    b2.next_back();
    println!("len {}", b2.len());
    let mut it = "aé€b".chars();
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
    println!("{:?}", it.next_back());
    println!("{:?}", it.next());
    let cs: Vec<char> = "aé€b".chars().rev().collect();
    println!("{cs:?}");
}
"#;

#[test]
fn run_mode_str_iter_states_use_the_real_wrapper_shape() {
    // The `str::Chars` / `str::Bytes` states are the structs'
    // real shapes (Chars { slice::Iter }, Bytes(Copied<slice::
    // Iter>)): `ExactSizeIterator::len` for `Bytes` is a
    // collected core body that reads that layout off the state,
    // and it must agree with the modelled front/back steps at
    // every point.
    let dir = temp_dir("str-iter-wrapper");
    let src = fixture(&dir, "striterwrapper.rs", STR_ITER_WRAPPER_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "len 7\nlen 5\nSome('a')\nSome('b')\nSome('é')\nSome('€')\nNone\n['b', '€', 'é', 'a']\n",
        "stdout={stdout}"
    );
}

const SIZE_HINT_DEFAULT_SRC: &str = r#"
struct Thing<'a>(&'a str);
impl<'a> Iterator for Thing<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        None
    }
}
fn main() {
    let n = Thing("test").fuse().filter(|_| true).count();
    println!("count {n}");
    let (l, u) = Thing("test").size_hint();
    println!("hint {l} {}", u.is_none());
    let (l, u) = Thing("test").fuse().size_hint();
    println!("fused {l} {}", u.is_none());
}
"#;

#[test]
fn run_mode_size_hint_unmodelled_iterator_gets_trait_default() {
    // Regression (tests/ui/inference/issue-36053.rs): a user
    // iterator that does not override `size_hint`, reached
    // through adapters (`Fuse`, `Filter::count`). Registering
    // the `Iterator::size_hint` builtin removes the trait's
    // default body from collection, so the builtin itself
    // answers unmodelled iterators with that default,
    // `(0, None)` — the exact native answer here, and a
    // contract-valid one for every iterator.
    let dir = temp_dir("size-hint-default");
    let src = fixture(&dir, "sizehintdefault.rs", SIZE_HINT_DEFAULT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "count 0\nhint 0 true\nfused 0 true\n",
        "stdout={stdout}"
    );
}

const ZST_ITER_LEN_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn main() {
    let a = [(), (), (), (), ()];
    let mut it = a.iter();
    println!("len {}", it.len());
    println!("slice {}", it.as_slice().len());
    it.next();
    println!("len {}", it.len());
    it.next_back();
    println!("len {}", it.len());
    println!("slice {}", it.as_slice().len());
    let e: [(); 0] = [];
    println!("empty {}", e.iter().len());
    let v = vec![(), (), ()];
    println!("vec {}", v.iter().len());
}
"#;

#[test]
fn run_mode_zst_slice_iter_len_and_as_slice() {
    // Issue #346: for a zero-sized element, core's `len()`
    // body reads the remaining count out of the end
    // pointer's own bits, and `as_slice` builds on the same
    // reading; over the modelled snapshot state that word is
    // the snapshot's address, not a count. The modelled
    // `ExactSizeIterator::len` / `as_slice` read the unified
    // bounds the modelled steps maintain instead, before and
    // after steps from both ends.
    let dir = temp_dir("zst-iter-len");
    let src = fixture(&dir, "zstiterlen.rs", ZST_ITER_LEN_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "len 5\nslice 5\nlen 4\nlen 3\nslice 3\nempty 0\nvec 3\n",
        "stdout={stdout}"
    );
}

const ITER_MUT_LEN_SRC: &str = r#"
use std::iter::ExactSizeIterator;
#[derive(Debug)]
struct Z;
fn main() {
    let mut a = [Z, Z, Z, Z];
    let mut m = a.iter_mut();
    println!("mlen {}", m.len());
    m.next();
    println!("mlen {}", m.len());
    m.next_back();
    println!("mlen {}", m.len());
    let mut b = [1, 2, 3, 4];
    let mut n = b.iter_mut();
    n.next();
    println!("rest {:?}", n.as_slice());
    println!("nlen {}", n.len());
    let rest = n.into_slice();
    rest[0] = 20;
    println!("b {:?}", b);
}
"#;

#[test]
fn run_mode_slice_iter_mut_len_after_steps() {
    // Issue #346, the stepped-state half of the same defect:
    // an `IterMut` state built by core's own constructor
    // pairs a positioned front pointer with an end pointer
    // holding the terminal index, so the collected `len()`
    // body — which reads the end pointer's address — stays
    // at the original length as the front steps. The
    // modelled `len` reads the current bounds; `as_slice` /
    // `into_slice` build the remaining slice from the same
    // reading.
    let dir = temp_dir("iter-mut-len");
    let src = fixture(&dir, "itermutlen.rs", ITER_MUT_LEN_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "mlen 4\nmlen 3\nmlen 2\nrest [2, 3, 4]\nnlen 3\nb [1, 20, 3, 4]\n",
        "stdout={stdout}"
    );
}

const EXACT_SIZE_LEN_OTHERS_SRC: &str = r#"
use std::iter::ExactSizeIterator;
use std::collections::BTreeSet;
struct Count(u32, u32);
impl Iterator for Count {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.0 < self.1 { self.0 += 1; Some(self.0) } else { None }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = (self.1 - self.0) as usize;
        (r, Some(r))
    }
}
impl ExactSizeIterator for Count {}
fn main() {
    println!("range {}", (3..7).len());
    let v = vec![9, 8, 7, 6];
    let mut vi = v.into_iter();
    println!("into0 {}", vi.len());
    vi.next();
    println!("into1 {}", vi.len());
    let mut cu = Count(0, 4);
    println!("user0 {}", cu.len());
    cu.next();
    println!("user1 {}", cu.len());
    let mut set = BTreeSet::new();
    set.insert(5);
    set.insert(6);
    set.insert(7);
    let mut bi = set.iter();
    println!("btree0 {}", bi.len());
    bi.next();
    println!("btree1 {}", bi.len());
}
"#;

#[test]
fn run_mode_exact_size_len_modelled_family_and_bodies() {
    // The modelled `len` serves only the slice-state family
    // (here: the B-tree iterator sharing that state); every
    // other type keeps its collected body — `Range`'s and
    // `vec::IntoIter`'s own overrides, and the trait default
    // (delegating to `size_hint`) for a user iterator.
    // Registering the builtin must not divert any of them.
    let dir = temp_dir("exact-size-len-others");
    let src = fixture(&dir, "exactlenothers.rs", EXACT_SIZE_LEN_OTHERS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "range 4\ninto0 4\ninto1 3\nuser0 4\nuser1 3\nbtree0 3\nbtree1 2\n",
        "stdout={stdout}"
    );
}

const EXACT_SIZE_LEN_REF_WRAPPER_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn len_of<I: ExactSizeIterator>(it: I) -> usize { it.len() }
fn main() {
    let a = [10, 20, 30, 40];
    let mut it = a.iter();
    let r = &mut it;
    println!("mut0 {}", ExactSizeIterator::len(&r));
    r.next();
    r.next();
    println!("mut1 {}", ExactSizeIterator::len(&r));
    let mut it2 = a.iter();
    let mut r2 = &mut it2;
    let rr = &mut r2;
    println!("dbl0 {}", ExactSizeIterator::len(&rr));
    let mut it3 = a.iter();
    println!("gen0 {}", len_of(&mut it3));
    it3.next();
    println!("gen1 {}", len_of(&mut it3));
    let z = [(), (), (), (), ()];
    let mut zit = z.iter();
    let zr = &mut zit;
    println!("zst0 {}", ExactSizeIterator::len(&zr));
    zr.next();
    println!("zst1 {}", ExactSizeIterator::len(&zr));
    let mut b = [1, 2, 3];
    let mut mit = b.iter_mut();
    let mr = &mut mit;
    println!("itermut0 {}", ExactSizeIterator::len(&mr));
    mr.next();
    println!("itermut1 {}", ExactSizeIterator::len(&mr));
    let mut it4 = a.iter();
    let br = it4.by_ref();
    println!("byref0 {}", br.len());
    println!("byref1 {}", it4.len());
}
"#;

#[test]
fn run_mode_exact_size_len_reference_wrapper_forwards() {
    // A reference-wrapper `Self` (`&mut I`, the one reference
    // form implementing the trait) is not the modelled
    // builtin's to serve: core's blanket body for `&mut I` is
    // collected and forwards to the inner call, which the
    // builtin then serves with the iterator type itself.
    // Peeling the reference in the modelled predicate
    // preempted that body, and the builtin — dereferencing
    // its receiver only once — found a reference where the
    // iterator state belongs ("expected ADT value, found
    // reference"). The ZST case pins that the forwarded
    // inner call still reaches the modelled builtin.
    let dir = temp_dir("exact-size-len-ref-wrapper");
    let src = fixture(&dir, "exactlenref.rs", EXACT_SIZE_LEN_REF_WRAPPER_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "mut0 4\nmut1 2\ndbl0 4\ngen0 4\ngen1 3\nzst0 5\nzst1 4\nitermut0 3\nitermut1 2\nbyref0 4\nbyref1 4\n",
        "stdout={stdout}"
    );
}

const ARRAY_INTO_ITER_COLLECT_SRC: &str = r#"
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
fn main() {
    let v: Vec<i32> = [1, 2, 3].into_iter().collect();
    let s: String = ['a', 'b'].into_iter().collect();
    println!("v={v:?} s={s}");
    let b: Box<[i32]> = [4, 5, 6].into_iter().collect();
    let bs: Box<str> = ['h', 'i'].into_iter().collect();
    println!("b={b:?} bs={bs}");
    let r: Rc<[i32]> = [10, 20, 30].into_iter().collect();
    let r2 = r.clone();
    println!("r={} {} {}", r[2], r2.len(), Rc::strong_count(&r));
    let a: Arc<[String]> = ["x".to_string(), "yy".to_string()].into_iter().collect();
    println!("a={} {}", a.len(), a[1]);
    let mut d: VecDeque<i32> = [7, 8].into_iter().collect();
    d.push_back(9);
    let front = d.pop_front();
    println!("d={d:?} f={front:?}");
    let empty: Rc<[i32]> = [].into_iter().collect();
    let from_arr: Rc<[i32]> = Rc::from([3, 1]);
    let from_arc: Arc<[i32]> = Arc::from([2, 4]);
    println!("e={} fa={from_arr:?} fc={from_arc:?}", empty.len());
}
"#;

const FROM_BOXED_SLICE_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let boxed = vec![1, 2, 3].into_boxed_slice();
    let rc: Rc<[i32]> = boxed.into();
    let boxed2 = vec![4, 5].into_boxed_slice();
    let arc: Arc<[i32]> = boxed2.into();
    println!("{:?} {} {:?} {}", rc, rc.len(), arc, arc.len());
}
"#;

#[test]
fn run_mode_from_boxed_slice_to_counted_slice() {
    // `From<Box<[T]>>` for `Rc<[T]>` / `Arc<[T]>`: the counted
    // handle is built over the box's pointee (the element tuple),
    // the same take as for the boxed string.
    run_ok(
        "from-boxed-slice",
        FROM_BOXED_SLICE_SRC,
        "[1, 2, 3] 3 [4, 5] 2\n",
    );
}

const FROM_VEC_RAW_PTRS_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let v = vec![Box::into_raw(Box::new(42))];
    let rc: Rc<[*mut i32]> = v.into();
    let a = unsafe { *rc[0] };
    let v2 = vec![Box::into_raw(Box::new(7))];
    let arc: Arc<[*mut i32]> = v2.into();
    let b = unsafe { *arc[0] };
    println!("{a} {b} {} {}", rc.len(), arc.len());
}
"#;

#[test]
fn run_mode_from_vec_of_raw_ptrs_to_counted_slice() {
    // `From<Vec<T>>` for `Rc<[T]>` / `Arc<[T]>` where the
    // vector's single element is an allocation-backed raw
    // pointer: the vector's value has the same one-field shape
    // as a box aggregate, so the counted-slice conversion must
    // not unbox it -- the pointer stays the vector's element,
    // and its pointee is untouched. (The boxed-source take is
    // gated on the source type naming `Box`.)
    run_ok(
        "from-vec-raw-ptrs",
        FROM_VEC_RAW_PTRS_SRC,
        "42 7 1 1\n",
    );
}

const FROM_VEC_RAW_PTR_FROM_SRC: &str = r#"
use std::rc::Rc;

fn main() {
    let p = Box::into_raw(Box::new(11));
    let v = vec![p];
    let rc: Rc<[*mut i32]> = Rc::from(v);
    println!("{} {}", unsafe { *rc[0] }, rc.len());
}
"#;

#[test]
fn run_mode_from_vec_of_raw_ptr_to_counted_slice_via_from() {
    // The `Rc::from(vec)` spelling of the same conversion.
    run_ok(
        "from-vec-raw-ptr-from",
        FROM_VEC_RAW_PTR_FROM_SRC,
        "11 1\n",
    );
}

#[test]
fn run_mode_array_into_iter_collect_into_boxed_and_counted_slices() {
    // `array::IntoIter` is a modelled iterator, so its `collect`
    // reaches the runtime's collect model directly (unlike a
    // slice iterator, whose collected std body can build these
    // targets itself). The boxed and counted slice targets take
    // the shapes the `into_boxed_slice` / counted-slice `From`
    // models build, and the deque the `From`-conversion
    // aggregate; each collected value must then behave like the
    // native one (indexing, clone counts, mutation, drops).
    let dir = temp_dir("array_into_iter_collect");
    let src = fixture(&dir, "case.rs", ARRAY_INTO_ITER_COLLECT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "array IntoIter collect into boxed/counted/deque targets runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "v=[1, 2, 3] s=ab\nb=[4, 5, 6] bs=hi\nr=30 3 2\na=2 yy\nd=[8, 9] f=Some(7)\ne=0 fa=[3, 1] fc=[2, 4]\n"
    );
}

const ARRAY_INTO_ITER_COLLECT_PARTIAL_SRC: &str = r#"
use std::iter::FromIterator;
use std::rc::Rc;
fn main() {
    let b = Box::from_iter([4, 5].into_iter());
    println!("{:?}", b);
    let mut it = [1, 2, 3, 4].into_iter();
    let _ = it.next();
    let r: Rc<[i32]> = it.collect();
    println!("{:?}", r);
    let v = Vec::from_iter(['x', 'y'].into_iter());
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_array_into_iter_collect_partial_and_from_iter() {
    // The `FromIterator::from_iter` entry reaches the same
    // collect model, and a partially consumed array iterator
    // collects only its remaining elements.
    let dir = temp_dir("array_into_iter_collect_partial");
    let src = fixture(&dir, "case.rs", ARRAY_INTO_ITER_COLLECT_PARTIAL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "from_iter and partially consumed array IntoIter collect run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[4, 5]\n[2, 3, 4]\n['x', 'y']\n"
    );
}

// ----- IntoIterator::into_iter runtime support (Lane DT) ---------------------

const VEC_BY_VALUE_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3];
    let mut sum = 0;
    for x in v {
        sum += x;
    }
    println!("sum={sum}");
    let mut it = vec![10, 20].into_iter();
    let a = it.next().unwrap_or(-1);
    let b = it.next().unwrap_or(-1);
    let c = it.next().unwrap_or(-1);
    println!("seq={a},{b},{c}");
    let w = vec![String::from("a"), String::from("bb"), String::from("ccc")];
    let mut n = 0;
    for s in w {
        n += s.len();
        break;
    }
    println!("n={n}");
}
"#;

#[test]
fn run_mode_vec_by_value_iteration() {
    let dir = temp_dir("vec-by-value");
    let src = fixture(&dir, "vec_by_value.rs", VEC_BY_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec by-value iteration must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "sum=6\nseq=10,20,-1\nn=1\n",
        "owned elements are yielded in order, then None"
    );
}

const VEC_DROP_ORDER_SRC: &str = r#"
struct Frag(i32);
impl Drop for Frag {
    fn drop(&mut self) { println!("drop {}", self.0); }
}
fn main() {
    let v = vec![Frag(1), Frag(2), Frag(3)];
    for x in v {
        println!("got {}", x.0);
        break;
    }
    println!("end");
    let w = vec![Frag(4), Frag(5)];
    for x in w {
        println!("got {}", x.0);
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_into_iter_drop_order() {
    // Each element drops exactly once: yielded ones at the end of
    // their loop-body scope, the rest in order when the iterator
    // itself is dropped.
    let dir = temp_dir("vec-drop-order");
    let src = fixture(&dir, "vec_drop_order.rs", VEC_DROP_ORDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a partially consumed vec::IntoIter must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "got 1\ndrop 1\ndrop 2\ndrop 3\nend\ngot 4\ndrop 4\ngot 5\ndrop 5\ndone\n",
        "yielded elements drop once, unyielded elements drop once, in order"
    );
}

const BOX_ARRAY_INTO_ITER_SRC: &str = r#"
fn main() {
    let b = Box::new([1, 2, 3]);
    let mut it = b.into_iter();
    println!("hint={:?}", it.size_hint());
    println!("back={:?}", it.next_back());
    let rest: Vec<i32> = it.collect();
    println!("rest={rest:?}");
    let big: Box<[u8; 33]> = Box::new([7u8; 33]);
    println!("big={}", big.into_iter().count());
    let nested = Box::new(Box::new([1, 2]));
    println!("nested={}", nested.into_iter().sum::<i32>());
    let e: Box<[i32; 0]> = Box::new([]);
    println!("empty={}", e.into_iter().count());
    let z: Box<[(); 3]> = Box::new([(), (), ()]);
    println!("zst={}", z.into_iter().count());
}
"#;

/// A by-value `Box<[T; N]>` iterates as
/// `BoxedArrayIntoIter`: the modelled state is the
/// pointee array's elements plus the next index, so the
/// front and back steps, the exact size hint and the
/// consumers all read it like the array iterator's.
#[test]
fn run_mode_box_array_into_iter_consumes() {
    run_ok(
        "box-array-into-iter",
        BOX_ARRAY_INTO_ITER_SRC,
        "hint=(3, Some(3))\nback=Some(3)\nrest=[1, 2]\nbig=33\nnested=3\nempty=0\nzst=3\n",
    );
}

const REF_BOX_INTO_ITER_SRC: &str = r#"
fn main() {
    let boxed = vec![1, 2].into_boxed_slice();
    let mut s = 0;
    for x in &boxed { s += x; }
    println!("refsum={s}");
    let mut s2 = 0;
    for x in boxed.into_iter() { s2 += x; }
    println!("refsum2={s2}");
    let ba = Box::new([10, 20]);
    let mut s3 = 0;
    for x in &ba { s3 += x; }
    println!("refarr={s3}");
    let mut bm = vec![1, 2].into_boxed_slice();
    for x in &mut bm { *x += 10; }
    println!("mut={:?}", &bm[..]);
    let mut bam = Box::new([1, 2]);
    for x in &mut bam { *x *= 3; }
    println!("mutarr={:?}", &bam[..]);
    let mut zm: Box<[(); 2]> = Box::new([(), ()]);
    let mut cnt = 0;
    for _ in &mut zm { cnt += 1; }
    println!("zstmut={cnt}");
}
"#;

/// `&Box<[T]>` and `&Box<[T; N]>` iterate as `slice::Iter`
/// over the pointee's elements, and the `&mut` forms as
/// `slice::IterMut` whose element references denote
/// positions in the box's own allocation, so mutations
/// through them land in the box.
#[test]
fn run_mode_ref_box_into_iter() {
    run_ok(
        "ref-box-into-iter",
        REF_BOX_INTO_ITER_SRC,
        "refsum=3\nrefsum2=3\nrefarr=30\nmut=[11, 12]\nmutarr=[3, 6]\nzstmut=2\n",
    );
}

const BOX_INTO_ITER_DROP_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};
static DROPS: AtomicUsize = AtomicUsize::new(0);
struct D(i32);
impl Drop for D { fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); } }
fn main() {
    let bs: Box<[D]> = vec![D(4), D(5)].into_boxed_slice();
    let mut it = bs.into_iter();
    let f = it.next().unwrap();
    drop(f);
    drop(it);
    println!("slicedrops={}", DROPS.load(Ordering::SeqCst));
    DROPS.store(0, Ordering::SeqCst);
    let b = Box::new([D(1), D(2), D(3)]);
    let mut it2 = b.into_iter();
    let f2 = it2.next().unwrap();
    drop(f2);
    drop(it2);
    println!("arraydrops={}", DROPS.load(Ordering::SeqCst));
    let v: Vec<i32> = vec![1, 2, 3].into_boxed_slice().into_iter().collect();
    println!("v={v:?}");
}
"#;

/// In edition 2024 `boxed.into_iter()` resolves to the
/// by-value `Box<[T]>` impl (in earlier editions it is
/// the reference impl): a partially consumed owning
/// boxed iterator drops exactly the elements it still
/// owns, the yielded one once and the rest with the
/// iterator — the same discipline as `vec::IntoIter`,
/// and the same for `Box<[T; N]>` in any edition.
#[test]
fn run_mode_box_into_iter_drop_discipline_2024() {
    let dir = temp_dir("box-into-iter-drop-2024");
    let src = fixture(&dir, "case.rs", BOX_INTO_ITER_DROP_SRC);
    let out = run_driver_extra(&src, &["--edition", "2024"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "slicedrops=2\narraydrops=3\nv=[1, 2, 3]\n",
        "stdout={stdout}"
    );
}

// ----- repeat_with (RepeatWith) ---------------------------------------------

const REPEAT_WITH_REF_CAPTURE_SRC: &str = r#"
use std::iter::repeat_with;

fn main() {
    let mut state: u32 = 0;
    let mut s = String::new();
    for c in repeat_with(|| {
        state += 1;
        (b'a' + (state % 26) as u8) as char
    })
    .take(5)
    {
        s.push(c);
    }
    println!("{} {}", s, state);
}
"#;

#[test]
fn run_mode_repeat_with_ref_capture_take() {
    // The shape `tempfile`'s name generation uses: a
    // `RepeatWith` whose producer captures `&mut` state,
    // consumed through `Take` by a for loop. The step calls
    // the producer on every pull; the mutations land in the
    // captured place through the reference.
    let dir = temp_dir("repeatwithref");
    let src = fixture(&dir, "repeatwithref.rs", REPEAT_WITH_REF_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "repeat_with over a &mut capture must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "bcdef 5\n");
}

const REPEAT_WITH_VALUE_CAPTURE_SRC: &str = r#"
use std::iter::repeat_with;

fn main() {
    let mut n = 0u32;
    let mut it = repeat_with(move || {
        n += 1;
        n * 10
    });
    println!("{:?} {:?} {:?}", it.next(), it.next(), it.next());
}
"#;

#[test]
fn run_mode_repeat_with_by_value_capture_next() {
    // The producer is `FnMut` over a by-value capture: each
    // `next()` on the iterator in its place must observe the
    // previous call's writes to the capture, so the step is
    // taken in place against the closure field's own storage.
    let dir = temp_dir("repeatwithval");
    let src = fixture(&dir, "repeatwithval.rs", REPEAT_WITH_VALUE_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "repeat_with over a by-value capture must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Some(10) Some(20) Some(30)\n"
    );
}

const REPEAT_WITH_CONSUMERS_SRC: &str = r#"
use std::iter::repeat_with;

fn main() {
    let v: Vec<u32> = repeat_with(|| 7u32).take(3).collect();
    let seed = 7u32;
    let found = repeat_with(|| seed * 3)
        .find_map(|x| if x > 20 { Some(x + 1) } else { None });
    println!("{:?} {:?}", v, found);
}
"#;

#[test]
fn run_mode_repeat_with_collect_and_find_map() {
    // `collect` over `Take<RepeatWith<_>>` drains the state
    // functionally through the modelled step, and `find_map`
    // pulls the bare `RepeatWith` until the closure's item
    // matches; both must see the producer's values in order.
    let dir = temp_dir("repeatwithconsumers");
    let src = fixture(
        &dir,
        "repeatwithconsumers.rs",
        REPEAT_WITH_CONSUMERS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "collect/find_map over repeat_with must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[7, 7, 7] Some(22)\n"
    );
}
