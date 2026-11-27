// Sparse, uninitialised and assume-init slice tests.

const MAYBE_UNINIT_ARRAY_BUFFER_SRC: &str = r#"
use std::mem::MaybeUninit;
use std::rc::Rc;
struct S {
    id: u32,
    rc: Rc<()>,
}
impl Clone for S {
    fn clone(&self) -> Self {
        S { id: self.id, rc: self.rc.clone() }
    }
}
fn main() {
    let counter = Rc::new(());
    let mut buf: MaybeUninit<[S; 4]> = MaybeUninit::uninit();
    let ptr = buf.as_mut_ptr() as *mut S;
    unsafe {
        for i in 0..4 {
            ptr.add(i).write(S { id: i as u32, rc: counter.clone() });
        }
        let arr = buf.assume_init();
        println!("after assume_init: {}", Rc::strong_count(&counter));
        println!("ids: {} {} {} {}", arr[0].id, arr[1].id, arr[2].id, arr[3].id);
        let arr2 = arr.clone();
        println!("after clone: {}", Rc::strong_count(&counter));
        drop(arr2);
        drop(arr);
    }
    println!("final: {}", Rc::strong_count(&counter));
}
"#;

#[test]
fn run_mode_maybe_uninit_array_buffer_element_access() {
    // A whole-buffer `MaybeUninit<[S; 4]>`: the `uninit()`
    // construction must materialize the payload slot as the
    // array's uninitialized skeleton (the model's field-0
    // payload convention), so element writes through
    // `as_mut_ptr` land in array slots and `assume_init` reads
    // the filled array back.
    let dir = temp_dir("maybe-uninit-array-buffer");
    let src = fixture(&dir, "maybe_uninit_buffer.rs", MAYBE_UNINIT_ARRAY_BUFFER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after assume_init: 5\nids: 0 1 2 3\nafter clone: 9\nfinal: 1\n"
    );
}

const SPARSE_VEC_SRC: &str = r#"
fn main() {
    // A length past the value model's materialisation limit is
    // held sparsely: the length is exact, the stored prefix is
    // kept, and the tail denotes uninitialised slots.
    let mut vec: Vec<String> = Vec::with_capacity(5_000_000);
    unsafe { vec.set_len(5_000_000) }
    assert_eq!(vec.len(), 5_000_000);
    drop(vec);
    println!("sparse string drop ok");

    let mut v2: Vec<String> = Vec::new();
    v2.push("a".to_string());
    v2.push("bb".to_string());
    unsafe { v2.set_len(5_000_000) }
    assert_eq!(v2.len(), 5_000_000);
    assert_eq!(v2[0], "a");
    assert_eq!(v2[1], "bb");
    drop(v2);
    println!("sparse prefix drop ok");

    let mut v3: Vec<u8> = Vec::with_capacity(5_000_000);
    unsafe { v3.set_len(5_000_000) }
    let s = &v3[..];
    assert_eq!(s.len(), 5_000_000);
    assert_eq!(v3.len(), 5_000_000);
    drop(v3);
    println!("sparse u8 ok");

    // Shrinking back under the limit materialises the vector.
    let mut v4: Vec<u8> = Vec::with_capacity(5_000_000);
    unsafe { v4.set_len(5_000_000) }
    unsafe { v4.set_len(3) }
    assert_eq!(v4.len(), 3);
    v4.push(9);
    assert_eq!(v4[3], 9);
    println!("shrink ok");

    // Growing a materialised vector past the limit keeps its
    // stored elements.
    let mut v5: Vec<u8> = vec![1, 2, 3];
    unsafe { v5.set_len(5_000_000) }
    assert_eq!(v5.len(), 5_000_000);
    assert_eq!(v5[0], 1);
    assert_eq!(v5[2], 3);
    let s5 = &v5[..];
    assert_eq!(s5.len(), 5_000_000);
    drop(v5);
    println!("grow ok");
}
"#;

#[test]
fn run_mode_sparse_vec_set_len() {
    run_ok(
        "sparse-vec",
        SPARSE_VEC_SRC,
        "sparse string drop ok\nsparse prefix drop ok\nsparse u8 ok\nshrink ok\ngrow ok\n",
    );
}

const SPARSE_VEC_UNINIT_READ_SRC: &str = r#"
fn main() {
    let mut vec: Vec<u8> = Vec::with_capacity(5_000_000);
    unsafe { vec.set_len(5_000_000) }
    // A tail slot was never written: reading it is the same
    // uninitialised read a materialised `set_len` vector's
    // padding raises.
    println!("{}", vec[0]);
}
"#;

#[test]
fn run_mode_sparse_vec_uninit_read() {
    let dir = temp_dir("sparse-vec-uninit");
    let src = fixture(&dir, "case.rs", SPARSE_VEC_UNINIT_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("uninitialized"),
        "tail read must be an uninitialised read; stderr={stderr}"
    );
}

const ASSUME_INIT_SLICE_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let a: [MaybeUninit<i32>; 3] =
        [MaybeUninit::new(10), MaybeUninit::new(20), MaybeUninit::new(30)];
    let s = unsafe { &*(&a as *const [MaybeUninit<i32>] as *const [i32]) };
    // Every element of a slice whose element type changed across
    // a pointer cast is readable: by index, by iteration, and by
    // raw pointer offset.
    println!("len={} sum={}", s.len(), s.iter().sum::<i32>());
    println!("idx={} {} {}", s[0], s[1], s[2]);
    println!("last={:?} count={}", s.last(), s.iter().count());
    let q = s.as_ptr();
    unsafe {
        println!("raw={} {} {}", *q, *q.add(1), *q.add(2));
    }
    let v = s.to_vec();
    println!("vec={:?}", v);
}
"#;

#[test]
fn run_mode_assume_init_slice_reads_all_elements() {
    run_ok(
        "assume-init-slice",
        ASSUME_INIT_SLICE_SRC,
        "len=3 sum=60\nidx=10 20 30\nlast=Some(30) count=3\nraw=10 20 30\nvec=[10, 20, 30]\n",
    );
}

const ASSUME_INIT_SLICE_MUT_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let mut a: [MaybeUninit<(i32, i32)>; 4] = [
        MaybeUninit::new((1, 2)),
        MaybeUninit::new((3, 4)),
        MaybeUninit::new((5, 6)),
        MaybeUninit::new((7, 8)),
    ];
    let s = unsafe { &mut *(&mut a as *mut [MaybeUninit<(i32, i32)>] as *mut [(i32, i32)]) };
    // A write through the cast slice lands in the addressed
    // element, not in element zero's payload.
    s[1].0 = 99;
    let sub = &mut s[1..4];
    sub[2].1 = 66;
    let t = unsafe { &*(&a as *const [MaybeUninit<(i32, i32)>] as *const [(i32, i32)]) };
    println!("e0={:?} e1={:?} e2={:?} e3={:?}", t[0], t[1], t[2], t[3]);
}
"#;

#[test]
fn run_mode_assume_init_slice_mut_writes_addressed_elements() {
    run_ok(
        "assume-init-slice-mut",
        ASSUME_INIT_SLICE_MUT_SRC,
        "e0=(1, 2) e1=(99, 4) e2=(5, 6) e3=(7, 66)\n",
    );
}

const MAYBEUNINIT_SLICE_COPY_SRC: &str = r#"
use std::mem::MaybeUninit;

fn main() {
    let mut v: Vec<usize> = vec![0, 1000, 2000, 3000, 1, 1001, 2001, 3001];
    let mut storage: [MaybeUninit<usize>; 8] = [const { MaybeUninit::uninit() }; 8];
    unsafe {
        let buf = storage.as_mut_ptr().cast::<usize>();
        std::ptr::copy_nonoverlapping(v.as_ptr(), buf, 4);
        println!("first {}", *buf);
        println!("second {}", *buf.add(1));
    }
    println!("v0 {}", v[0]);
}
"#;

#[test]
fn run_mode_maybeuninit_slice_copy_wider_elements() {
    // A bulk copy into `[MaybeUninit<T>]` storage through the
    // cast pointer: the destination target ends in the
    // wrapper's field-0 entry step behind the first-element
    // index step, and the plain indexed write appended the
    // element index after the entry, navigating into the
    // scalar payload. The sequence access now composes with
    // the index step behind whole-value entry steps, keeping
    // the entry, so the elements land in the storage slots.
    run_ok(
        "maybeuninit-slice-copy",
        MAYBEUNINIT_SLICE_COPY_SRC,
        "first 0\nsecond 1000\nv0 0\n",
    );
}

const VISITABLE_SLICE_DEREF_SRC: &str = r##"
#![feature(rustc_private)]

extern crate rustc_type_ir;
extern crate rustc_type_ir_macros;

use rustc_type_ir::GenericTypeVisitable;
use rustc_type_ir_macros::GenericTypeVisitable;

// Necessary to pull in object code as the rest of the rustc crates are shipped only as rmeta
// files.
#[expect(unused_extern_crates)]
extern crate rustc_driver;

#[derive(GenericTypeVisitable)]
enum Enum {
    A,
    B(Incrementer),
    C { one: Incrementer, two: Vec<Incrementer> },
}

#[derive(GenericTypeVisitable)]
struct PartiallyRecursiveField<T> {
    #[generic_type_visitable(bounds(T: GenericTypeVisitable<__V>))]
    partially_rec: (Vec<Self>, T),
    other: Incrementer,
}

use std::sync::atomic::{AtomicU8, Ordering};

static COUNT: AtomicU8 = AtomicU8::new(0);

#[derive(Clone)]
struct Incrementer;

unsafe impl<V> GenericTypeVisitable<V> for Incrementer {
    fn generic_visit_with(&self, _visitor: &mut V) {
        COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    use Incrementer as Inc; // for brevity

    #[track_caller]
    fn check<T: GenericTypeVisitable<()>>(item: T, count: u8) {
        let mut v = ();
        item.generic_visit_with(&mut v);
        assert_eq!(COUNT.swap(0, Ordering::Relaxed), count);
    }

    macro_rules! prec {
        ([$($i:expr),* $(,)?], $o:expr) => {
            PartiallyRecursiveField { partially_rec: (vec![$($i),*], $o), other: Inc }
        }
    }
    let a = || Enum::A;
    check(prec!([], a()), 1 + 0);
    check(prec!([prec!([], a())], a()), 1 + (1 + 0) + 0);
    println!("visitable ok");
}
"##;

#[test]
fn run_mode_visitable_slice_deref_element_not_wrapper() {
    // The reduced shape of tests/ui-fulldeps/derive-generic-type-visitable.rs:
    // the derived visitor iterates the (Vec<Self>, T) field through the
    // slice iterator machinery, whose construction reborrows the fat
    // pointer (`&*fat`). That dereference resolves to the base of the
    // fat pointer, a position denominated at the first element, and
    // the value stored there is the element itself -- here a struct
    // in enum form whose first field is the inner (Vec, Enum) tuple,
    // a tuple of values that fit the element type. The MaybeUninit
    // entry judgment in place resolution mistook that element for a
    // wrapper around the whole slice and descended into its first
    // field, shifting every later step one level too deep (the read
    // of the inner enum field then reported an uninitialised place).
    // A slice payload is the whole sequence, so the entry must be
    // refused at an element position.
    let dir = temp_dir("visitable-slice-deref");
    let src = fixture(&dir, "visitable.rs", VISITABLE_SLICE_DEREF_SRC);
    let out = run_driver_extra(&src, &["--edition", "2024"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "visitable ok\n");
}
