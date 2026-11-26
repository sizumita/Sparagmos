use super::{fixture, run_driver, run_driver_extra, run_driver_with_aux, run_ok, summarize_driver, temp_dir};


const AUX_SMALLVEC_SRC: &str = r##"// A faithful subset of smallvec 1.15.1 (the `union` feature
// form): the same struct/union layout and the same
// capacity-word invariant as the real crate. `push` covers
// the inline path; the spilled state is built by `from_vec`,
// which assembles exactly the heap shape the real crate's
// growth path produces.
use std::alloc::{alloc, Layout};
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

pub trait Array {
    type Item;
    fn size() -> usize;
}

impl<T, const N: usize> Array for [T; N] {
    type Item = T;
    fn size() -> usize {
        N
    }
}

union SmallVecData<A: Array> {
    inline: ManuallyDrop<MaybeUninit<A>>,
    heap: (NonNull<A::Item>, usize),
}

impl<A: Array> SmallVecData<A> {
    unsafe fn inline(&self) -> NonNull<A::Item> {
        NonNull::new(self.inline.as_ptr() as *mut A::Item).unwrap()
    }

    unsafe fn inline_mut(&mut self) -> NonNull<A::Item> {
        NonNull::new(self.inline.as_mut_ptr() as *mut A::Item).unwrap()
    }

    unsafe fn heap(&self) -> (NonNull<A::Item>, usize) {
        self.heap
    }

    unsafe fn heap_mut(&mut self) -> (NonNull<A::Item>, *mut usize) {
        let h = &mut self.heap;
        (h.0, &mut h.1)
    }

    unsafe fn heap_len(&self) -> usize {
        self.heap.1
    }

    fn from_heap(ptr: NonNull<A::Item>, len: usize) -> SmallVecData<A> {
        SmallVecData { heap: (ptr, len) }
    }
}

pub struct SmallVec<A: Array> {
    capacity: usize,
    data: SmallVecData<A>,
    _marker: PhantomData<A::Item>,
}

impl<A: Array> SmallVec<A> {
    pub fn new() -> Self {
        SmallVec {
            capacity: 0,
            data: SmallVecData {
                inline: ManuallyDrop::new(MaybeUninit::uninit()),
            },
            _marker: PhantomData,
        }
    }

    // Assemble the spilled state directly over a fresh heap
    // allocation -- the shape the real crate's growth path
    // produces (heap capacity above the inline capacity,
    // the heap field holding the buffer pointer and length).
    pub fn spilled_with(values: &[A::Item]) -> Self
    where
        A::Item: Copy,
    {
        let cap = A::size() + 4;
        unsafe {
            let layout = Layout::array::<A::Item>(cap).unwrap();
            let ptr = alloc(layout) as *mut A::Item;
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                ptr,
                values.len(),
            );
            SmallVec {
                capacity: cap,
                data: SmallVecData::from_heap(
                    NonNull::new(ptr).unwrap(),
                    values.len(),
                ),
                _marker: PhantomData,
            }
        }
    }

    pub fn len(&self) -> usize {
        if self.spilled() {
            unsafe { self.data.heap_len() }
        } else {
            self.capacity
        }
    }

    pub fn capacity(&self) -> usize {
        if self.spilled() {
            self.capacity
        } else {
            A::size()
        }
    }

    pub fn spilled(&self) -> bool {
        self.capacity > A::size()
    }

    pub fn as_ptr(&self) -> *const A::Item {
        unsafe {
            if self.spilled() {
                self.data.heap().0.as_ptr()
            } else {
                self.data.inline().as_ptr()
            }
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut A::Item {
        unsafe {
            if self.spilled() {
                self.data.heap().0.as_ptr()
            } else {
                self.data.inline_mut().as_ptr()
            }
        }
    }

    unsafe fn set_len(&mut self, len: usize) {
        if self.spilled() {
            let (_, len_ptr) = self.data.heap_mut();
            *len_ptr = len;
        } else {
            self.capacity = len;
        }
    }

    pub fn push(&mut self, value: A::Item) {
        let len = self.len();
        assert!(len < self.capacity(), "inline capacity exhausted");
        unsafe {
            self.as_mut_ptr().add(len).write(value);
            self.set_len(len + 1);
        }
    }
}

impl<A: Array> Deref for SmallVec<A> {
    type Target = [A::Item];
    fn deref(&self) -> &[A::Item] {
        unsafe {
            if self.spilled() {
                let (ptr, len) = self.data.heap();
                std::slice::from_raw_parts(ptr.as_ptr(), len)
            } else {
                let ptr = self.data.inline();
                std::slice::from_raw_parts(ptr.as_ptr(), self.capacity)
            }
        }
    }
}

impl<A: Array> DerefMut for SmallVec<A> {
    fn deref_mut(&mut self) -> &mut [A::Item] {
        unsafe {
            if self.spilled() {
                let (ptr, len_ptr) = self.data.heap_mut();
                std::slice::from_raw_parts_mut(ptr.as_ptr(), *len_ptr)
            } else {
                let ptr = self.data.inline_mut();
                std::slice::from_raw_parts_mut(ptr.as_ptr(), self.capacity)
            }
        }
    }
}
"##;

const AUX_SMALLVEC_MAIN_SRC: &str = r##"extern crate smallvec;
use smallvec::SmallVec;

#[derive(Debug, PartialEq, Clone, Copy)]
struct Directive {
    level: u32,
    enabled: bool,
}

fn main() {
    // Inline deref + deref_mut.
    let mut v: SmallVec<[u32; 4]> = SmallVec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v[0] = 10;
    let m: &mut [u32] = &mut v;
    m[1] = 20;
    m[2] += 3;
    assert_eq!(v.len(), 3);
    let s: &[u32] = &v;
    assert_eq!(s, [10, 20, 6]);
    println!("mut ok: {:?}", s);

    // Signed elements.
    let mut w: SmallVec<[i32; 2]> = SmallVec::new();
    w.push(-5);
    w.push(-7);
    let ws: &[i32] = &w;
    assert_eq!(ws, [-5, -7]);
    println!("signed ok: {:?}", ws);

    // One inline struct element.
    let mut d: SmallVec<[Directive; 8]> = SmallVec::new();
    d.push(Directive { level: 3, enabled: true });
    let ds: &[Directive] = &d;
    assert_eq!(ds.len(), 1);
    assert_eq!(ds[0].level, 3);
    assert!(ds[0].enabled);
    println!("struct ok: {}", ds[0].level);

    // Spilled deref: 6 elements in a capacity-4 vector.
    let g: SmallVec<[u32; 4]> =
        SmallVec::spilled_with(&[100, 101, 102, 103, 104, 105]);
    assert!(g.spilled());
    let gs: &[u32] = &g;
    assert_eq!(gs.len(), 6);
    assert_eq!(gs[0], 100);
    assert_eq!(gs[5], 105);
    let mut sum = 0u32;
    for x in gs {
        sum += x;
    }
    assert_eq!(sum, 615);
    println!("spilled ok: {:?}", gs);

    // Spilled deref_mut.
    let mut g = g;
    g[5] = 200;
    let gm: &mut [u32] = &mut g;
    gm[0] = 50;
    let gs2: &[u32] = &g;
    assert_eq!(gs2[0], 50);
    assert_eq!(gs2[5], 200);
    println!("spilled mut ok: {:?}", gs2);

    // Spilled struct elements.
    let sd: SmallVec<[Directive; 8]> = SmallVec::spilled_with(&[
        Directive { level: 3, enabled: true },
        Directive { level: 1, enabled: false },
    ]);
    assert!(sd.spilled());
    let sds: &[Directive] = &sd;
    assert_eq!(sds.len(), 2);
    assert_eq!(sds[0].level, 3);
    assert!(sds[0].enabled);
    assert_eq!(sds[1].level, 1);
    assert!(!sds[1].enabled);
    println!("spilled struct ok: {} {}", sds[0].level, sds[1].level);
}
"##;

#[test]
fn run_mode_smallvec_deref_and_deref_mut_execute() {
    // `Deref::deref` / `DerefMut::deref_mut` for
    // `smallvec::SmallVec<[T; N]>` are runtime-supported: the
    // interpreter models the crate's layout (the capacity word
    // is the length while it fits the inline capacity, and the
    // spill marker -- capacity word above the inline capacity
    // -- with the heap field holding the buffer pointer and
    // length once spilled). The auxiliary crate replicates
    // smallvec 1.15's layout (the `union` data form); its
    // `push` covers the inline path and `spilled_with`
    // assembles exactly the heap shape the real growth path
    // produces.
    let dir = temp_dir("aux-smallvec-deref");
    let aux = fixture(&dir, "smallvec.rs", AUX_SMALLVEC_SRC);
    let src = fixture(&dir, "smallvecmain.rs", AUX_SMALLVEC_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("smallvec", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "SmallVec deref/deref_mut must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "mut ok: [10, 20, 6]\nsigned ok: [-5, -7]\nstruct ok: 3\nspilled ok: [100, 101, 102, 103, 104, 105]\nspilled mut ok: [50, 101, 102, 103, 104, 200]\nspilled struct ok: 3 1\n",
        "deref reads and deref_mut writes observe the modelled layout"
    );
}

const AUX_SMALLVEC_ENUM_SRC: &str = r##"// The non-`union` form of smallvec 1.15.1: the data word is
// an enum with an Inline variant (the buffer) and a Heap
// variant (pointer + length).
use std::alloc::{alloc, Layout};
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

pub trait Array {
    type Item;
    fn size() -> usize;
}

impl<T, const N: usize> Array for [T; N] {
    type Item = T;
    fn size() -> usize {
        N
    }
}

enum SmallVecData<A: Array> {
    Inline { elements: ManuallyDrop<MaybeUninit<A>> },
    Heap { ptr: NonNull<A::Item>, len: usize },
}

pub struct SmallVec<A: Array> {
    capacity: usize,
    data: SmallVecData<A>,
    _marker: PhantomData<A::Item>,
}

impl<A: Array> SmallVec<A> {
    pub fn new() -> Self {
        SmallVec {
            capacity: 0,
            data: SmallVecData::Inline {
                elements: ManuallyDrop::new(MaybeUninit::uninit()),
            },
            _marker: PhantomData,
        }
    }

    pub fn spilled_with(values: &[A::Item]) -> Self
    where
        A::Item: Copy,
    {
        let cap = A::size() + 4;
        unsafe {
            let layout = Layout::array::<A::Item>(cap).unwrap();
            let ptr = alloc(layout) as *mut A::Item;
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
            SmallVec {
                capacity: cap,
                data: SmallVecData::Heap {
                    ptr: NonNull::new(ptr).unwrap(),
                    len: values.len(),
                },
                _marker: PhantomData,
            }
        }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            SmallVecData::Heap { len, .. } => *len,
            SmallVecData::Inline { .. } => self.capacity,
        }
    }

    pub fn spilled(&self) -> bool {
        self.capacity > A::size()
    }

    pub fn push(&mut self, value: A::Item) {
        let len = self.len();
        assert!(len < A::size() && !self.spilled());
        unsafe {
            let base = match &mut self.data {
                SmallVecData::Inline { elements } => {
                    elements.as_mut_ptr() as *mut A::Item
                }
                SmallVecData::Heap { .. } => unreachable!(),
            };
            base.add(len).write(value);
            self.capacity = len + 1;
        }
    }
}

impl<A: Array> Deref for SmallVec<A> {
    type Target = [A::Item];
    fn deref(&self) -> &[A::Item] {
        unsafe {
            match &self.data {
                SmallVecData::Inline { elements } => std::slice::from_raw_parts(
                    elements.as_ptr() as *const A::Item,
                    self.capacity,
                ),
                SmallVecData::Heap { ptr, len } => {
                    std::slice::from_raw_parts(ptr.as_ptr(), *len)
                }
            }
        }
    }
}

impl<A: Array> DerefMut for SmallVec<A> {
    fn deref_mut(&mut self) -> &mut [A::Item] {
        unsafe {
            match &mut self.data {
                SmallVecData::Inline { elements } => {
                    let ptr = elements.as_mut_ptr() as *mut A::Item;
                    std::slice::from_raw_parts_mut(ptr, self.capacity)
                }
                SmallVecData::Heap { ptr, len } => {
                    std::slice::from_raw_parts_mut(ptr.as_ptr(), *len)
                }
            }
        }
    }
}
"##;

const AUX_SMALLVEC_ENUM_MAIN_SRC: &str = r##"extern crate smallvec;
use smallvec::SmallVec;

fn main() {
    let mut v: SmallVec<[u32; 4]> = SmallVec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v[0] = 10;
    let m: &mut [u32] = &mut v;
    m[1] = 20;
    let s: &[u32] = &v;
    assert_eq!(s, [10, 20, 3]);
    println!("enum inline ok: {:?}", s);
    let g: SmallVec<[u32; 4]> =
        SmallVec::spilled_with(&[100, 101, 102, 103, 104, 105]);
    assert!(g.spilled());
    let gs: &[u32] = &g;
    assert_eq!(gs[5], 105);
    println!("enum spilled ok: {:?}", gs);
}
"##;

#[test]
fn run_mode_smallvec_deref_enum_data_form_executes() {
    // The same runtime support over the non-`union` data
    // form: an enum whose Inline variant holds the buffer
    // and whose Heap variant holds the buffer pointer and
    // length as separate fields.
    let dir = temp_dir("aux-smallvec-deref-enum");
    let aux = fixture(&dir, "smallvec.rs", AUX_SMALLVEC_ENUM_SRC);
    let src = fixture(&dir, "smallvecenummain.rs", AUX_SMALLVEC_ENUM_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("smallvec", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "SmallVec (enum data form) deref/deref_mut must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "enum inline ok: [10, 20, 3]\nenum spilled ok: [100, 101, 102, 103, 104, 105]\n",
        "deref reads observe the modelled enum layout"
    );
}

// ----- Vec allocation paths (Lane BO) -----

const VEC_DROPPABLE_SRC: &str = r#"
struct D(i32);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let v = vec![D(3), D(4)];
    println!("len {}", v.len());
    drop(v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_macro_of_droppables() {
    // The `vec![...]` list form builds the vector through
    // `Box::new_uninit` + `box_assume_init_into_vec_unsafe`, whose
    // real bodies bottom out in the runtime allocator
    // (`Global::alloc_impl_runtime` has no body). The interpreter
    // models those entry points, so a Vec of droppable elements
    // runs end to end and the element destructors run in order,
    // byte for byte as natively.
    let dir = temp_dir("vecdroppable");
    let src = fixture(&dir, "vecdroppable.rs", VEC_DROPPABLE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of droppables must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2\ndrop 3\ndrop 4\ndone\n",
        "vec! elements drop in order, exactly as natively"
    );
}

const VEC_ZST_ENUM_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) {} }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
fn read_it(v: Vec<CS<Never, DropMe>>) {
    match &v[0] { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    match &v[1] { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    println!("done");
}
fn main() {
    read_it(vec![CS::Complete(DropMe), CS::Complete(DropMe)]);
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_enum_elements_read_by_index() {
    // A `Vec` of a zero-sized enum (one variant uninhabited, the
    // other's payload zero-sized) built by `vec!`: MIR elides the
    // element store entirely, so the elements reach execution as
    // uninitialized slots, and an index read of an element denotes
    // the element type's single value (the canonical-value rescue
    // follows the projection's index step). Both elements read
    // back as the `Complete` variant, exactly as natively.
    let dir = temp_dir("veczstenum");
    let src = fixture(&dir, "veczstenum.rs", VEC_ZST_ENUM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\nc\ndone\n",
        "zero-sized enum elements read by index denote their single value"
    );
}

const VEC_ZST_ENUM_GENERIC_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) {} }
enum CS<Y, R> { Yielded(Y), Complete(R) }
trait Tr { type Out; }
struct S;
impl Tr for S { type Out = DropMe; }
enum Never {}
fn read_it<T: Tr>(v: Vec<CS<Never, T::Out>>) {
    match &v[0] { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    println!("done");
}
fn main() {
    read_it::<S>(vec![CS::Complete(DropMe)]);
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_enum_elements_generic_projection() {
    // The same element read with the payload type arriving
    // through a trait projection (`T::Out`): the element type is
    // still single-valued after instantiation, so the index read
    // is rescued the same way.
    let dir = temp_dir("veczstenumgeneric");
    let src = fixture(&dir, "veczstenumgeneric.rs", VEC_ZST_ENUM_GENERIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\ndone\n",
        "projected zero-sized enum elements read by index denote their single value"
    );
}

const VEC_GROWTH_SRC: &str = r#"
struct D(i32);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let mut v: Vec<D> = Vec::with_capacity(4);
    v.reserve(8);
    v.push(D(1));
    v.push(D(2));
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_with_capacity_push_and_reserve() {
    // `with_capacity` and `reserve` are allocation entry points:
    // the abstract Vec model does not represent capacity, so both
    // are modelled (an empty vector / a no-op) and pushes onto the
    // reserved vector behave as natively, including the scope-end
    // drop of the elements.
    let dir = temp_dir("vecgrowth");
    let src = fixture(&dir, "vecgrowth.rs", VEC_GROWTH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "with_capacity/reserve/push must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2\ndrop 1\ndrop 2\n",
        "reserved Vec pushes and drops exactly as natively"
    );
}


const VEC_RESIZE_GROW_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    v.resize(6, 9);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
    let mut w: Vec<i32> = Vec::new();
    w.resize(4, 7);
    println!("{:?} len={} cap={}", w, w.len(), w.capacity());
}
"#;

#[test]
fn run_mode_vec_resize_grows_with_fill() {
    // `Vec::resize` growth runs through `extend_with`, whose
    // `SetLenOnDrop` guard holds a `&mut` to the vector's length
    // slot and writes the filled length back through it on drop:
    // the write must land on the vector itself (it used to fail
    // with "expected mutable place, found constant aggregate"
    // because the length-slot reference denoted a detached value).
    let dir = temp_dir("vecresizegrow");
    let src = fixture(&dir, "vecresizegrow.rs", VEC_RESIZE_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec resize growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3, 9, 9, 9] len=6 cap=6\n[7, 7, 7, 7] len=4 cap=4\n",
        "grown vectors hold the fill values with native len/cap"
    );
}

const VEC_RESIZE_REGROW_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];
    v.resize(2, 0);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
    v.resize(3, 42);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
}
"#;

#[test]
fn run_mode_vec_resize_regrows_within_capacity() {
    // Shrinking then growing back inside the reserved capacity
    // takes the same length-slot write path with no reservation
    // in between: the regrown slot is filled, the length and the
    // (unchanged) capacity answer natively.
    let dir = temp_dir("vecresizeregrow");
    let src = fixture(&dir, "vecresizeregrow.rs", VEC_RESIZE_REGROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "shrink-then-regrow must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2] len=2 cap=5\n[1, 2, 42] len=3 cap=5\n",
        "regrowth within capacity fills and reports natively"
    );
}

const VEC_RESIZE_PAYLOAD_SRC: &str = r#"
#[derive(Debug, Clone)]
struct P {
    a: i32,
    b: i32,
}
fn main() {
    let mut s = vec!["a".to_string()];
    s.resize(4, "x".to_string());
    println!("{:?} len={}", s, s.len());
    let mut p = vec![P { a: 1, b: 2 }];
    p.resize(3, P { a: 7, b: 8 });
    println!("{:?} len={}", p, p.len());
}
"#;

#[test]
fn run_mode_vec_resize_grows_clone_payloads() {
    // The fill value is cloned per new slot: heap (`String`) and
    // aggregate (struct) payloads grow through the same path and
    // read back exactly.
    let dir = temp_dir("vecresizepayload");
    let src = fixture(&dir, "vecresizepayload.rs", VEC_RESIZE_PAYLOAD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "resize of payload vectors must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[\"a\", \"x\", \"x\", \"x\"] len=4\n[P { a: 1, b: 2 }, P { a: 7, b: 8 }, P { a: 7, b: 8 }] len=3\n",
        "cloned fill payloads read back exactly"
    );
}

const VEC_RESIZE_SPARE_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(100);
    v.push(1);
    v.push(2);
    v.resize(50, 7);
    println!("len={} cap={} sum={}", v.len(), v.capacity(), v.iter().sum::<i32>());
    v.resize(10, 0);
    println!("len={} cap={}", v.len(), v.capacity());
    v.resize(60, 3);
    println!("len={} cap={} sum={}", v.len(), v.capacity(), v.iter().sum::<i32>());
}
"#;

#[test]
fn run_mode_vec_resize_uses_spare_capacity() {
    // Growth into a large spare capacity, a shrink, and a
    // regrowth past the previous length all stay inside the
    // original allocation: capacity never moves and the sums
    // cover exactly the declared elements.
    let dir = temp_dir("vecresizespare");
    let src = fixture(&dir, "vecresizespare.rs", VEC_RESIZE_SPARE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "resize within spare capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=50 cap=100 sum=339\nlen=10 cap=100\nlen=60 cap=100 sum=209\n",
        "spare-capacity growth/shrink/regrowth answers natively"
    );
}

const VEC_HUGE_CAPACITY_SRC: &str = r#"
fn main() {
    let n: usize = (1 << 22) + 5;
    let v: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(n);
    println!("len {} cap {}", v.len(), v.capacity());
    println!("done");
}
"#;

#[test]
fn run_mode_vec_with_capacity_past_materialisation_limit() {
    // A `with_capacity` request past the materialisation limit
    // stores no elements, but the requested capacity is the
    // vector's storage and stays in the value's account: the
    // reported capacity is the requested one, as natively —
    // not the zero stored elements.
    let dir = temp_dir("vechugecapacity");
    let src = fixture(&dir, "vechugecapacity.rs", VEC_HUGE_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "huge with_capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 0 cap 4194309\ndone\n",
        "the requested capacity past the limit is reported as natively"
    );
}

const VEC_HUGE_SET_LEN_POP_SRC: &str = r#"
fn main() {
    let n: usize = (1 << 22) + 5;
    let mut v: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(n);
    unsafe {
        v.set_len(n);
        match v.pop() {
            Some(b) => {
                std::mem::forget(b);
                println!("popped len {}", v.len());
            }
            None => println!("none"),
        }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_huge_set_len_pop() {
    // `Vec::pop` decrements the length and then asserts the
    // new length is below the capacity. On a vector grown by
    // `set_len` to the requested capacity, the accounted
    // capacity is the full request, so the assertion holds as
    // natively; a capacity collapsed onto the declared length
    // would make the decremented length equal it and trip the
    // assertion.
    let dir = temp_dir("vechugesetlenpop");
    let src = fixture(&dir, "vechugesetlenpop.rs", VEC_HUGE_SET_LEN_POP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pop on a huge set_len-grown vector must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "popped len 4194308\ndone\n",
        "pop returns the tail slot and the decremented length, as natively"
    );
}

const STRING_WITH_CAPACITY_SRC: &str = r#"
fn main() {
    let mut s = String::with_capacity(16);
    s.push_str("hi");
    println!("{} {}", s, s.len());
}
"#;

#[test]
fn run_mode_string_with_capacity_push_str_and_len() {
    // `String::with_capacity` is a direct `RawVecInner` caller:
    // its collected body inlines down to
    // `RawVecInner::with_capacity_in`, which has no converted
    // body, so the constructor is modelled as the empty string
    // (the requested capacity is not represented, as for
    // `Vec::with_capacity`).
    let dir = temp_dir("stringwithcapacity");
    let src = fixture(&dir, "stringwithcapacity.rs", STRING_WITH_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "String::with_capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hi 2\n");
}

const VEC_TRY_RESERVE_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    println!("small {}", v.try_reserve(10).is_ok());
    println!("huge {}", v.try_reserve(usize::MAX).is_ok());
    v.push(4);
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_try_reserve_verdicts() {
    // Allocation never fails in the abstract model, so the only
    // error `try_reserve` can produce is capacity overflow:
    // 3 + 10 elements fit easily (Ok), while `usize::MAX` more
    // elements of 4 bytes exceed the `isize::MAX`-byte limit
    // (Err), exactly as natively.
    let dir = temp_dir("vectryreserve");
    let src = fixture(&dir, "vectryreserve.rs", VEC_TRY_RESERVE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::try_reserve must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "small true\nhuge false\nlen 4\n"
    );
}

const VEC_SHRINK_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    v.shrink_to_fit();
    v.shrink_to(2);
    println!("len {}", v.len());
    v.push(4);
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_shrink_to_fit_and_shrink_to() {
    // Shrinking moves only the capacity, which the abstract Vec
    // model does not represent: length, elements and later
    // pushes are exactly as natively.
    let dir = temp_dir("vecshrink");
    let src = fixture(&dir, "vecshrink.rs", VEC_SHRINK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::shrink_to_fit/shrink_to must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len 3\nlen 4\n");
}

const VEC_REMOVE_ADT_SRC: &str = r#"
struct E(i32);
impl Drop for E {
    fn drop(&mut self) {
        println!("drop E({})", self.0);
    }
}
fn main() {
    // `Vec::remove` shifts the tail with one `ptr::copy` of ADT
    // elements: every shifted element must land whole, in order.
    let mut v: Vec<E> = (0..6).map(E).collect();
    let x = v.remove(1);
    println!("removed {:?}", x.0);
    std::mem::forget(x);
    println!("v={:?}", v.iter().map(|e| e.0).collect::<Vec<_>>());
    // A shorter tail (a two-element shift) and a two-field
    // element take the same copy.
    let mut w: Vec<E> = (0..4).map(E).collect();
    let y = w.remove(1);
    std::mem::forget(y);
    println!("w={:?}", w.iter().map(|e| e.0).collect::<Vec<_>>());
}
"#;

#[test]
fn run_mode_vec_remove_shifts_adt_elements_whole() {
    let dir = temp_dir("vec-remove-adt");
    let src = fixture(&dir, "vec_remove_adt.rs", VEC_REMOVE_ADT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::remove over ADT elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "removed 1\nv=[0, 2, 3, 4, 5]\nw=[0, 2, 3]\ndrop E(0)\ndrop E(2)\ndrop E(3)\ndrop E(0)\ndrop E(2)\ndrop E(3)\ndrop E(4)\ndrop E(5)\n"
    );
}

const ARRAY_PTR_COPY_ADT_SRC: &str = r#"
struct E(i32);
fn main() {
    // The element copy behind `Vec::insert`'s right shift and
    // `Vec::remove`'s left shift, directly over an array of ADT
    // elements: elements move whole, never byte-composed.
    let mut a: [E; 6] = [E(0), E(1), E(2), E(3), E(4), E(5)];
    unsafe {
        let p = a.as_mut_ptr();
        std::ptr::copy(p.add(2), p.add(1), 4);
    }
    assert_eq!((a[0].0, a[1].0, a[2].0, a[3].0, a[4].0, a[5].0), (0, 2, 3, 4, 5, 5));
    let mut b: [E; 6] = [E(0), E(1), E(2), E(3), E(4), E(5)];
    unsafe {
        let p = b.as_mut_ptr();
        std::ptr::copy(p.add(1), p.add(2), 4);
        std::ptr::write(p.add(1), E(99));
    }
    assert_eq!((b[0].0, b[1].0, b[2].0, b[3].0, b[4].0, b[5].0), (0, 99, 1, 2, 3, 4));
    println!("array-copy ok");
}
"#;

#[test]
fn run_mode_ptr_copy_over_adt_array_elements() {
    let dir = temp_dir("array-ptr-copy-adt");
    let src = fixture(&dir, "array_ptr_copy_adt.rs", ARRAY_PTR_COPY_ADT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ptr::copy over ADT elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "array-copy ok\n");
}

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

const HUGE_ZST_ARRAY_CONST_SRC: &str = r#"
fn main() {
    println!("{}", [(); usize::MAX].len());
}
"#;

#[test]
fn convert_mode_huge_zst_array_constant_converts() {
    // A constant of a huge zero-sized array type (`[(); usize::MAX]`,
    // the shape of rustc's own compile-time regression test 68010)
    // has an empty allocation; decoding it element by element would
    // materialise `usize::MAX` values and abort the host process.
    // The constant is not materialised past the decode bound, and
    // the program converts.
    let dir = temp_dir("mono-huge-zst-array-const");
    let src = fixture(&dir, "huge_zst_array_const.rs", HUGE_ZST_ARRAY_CONST_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a huge ZST array constant must not be materialised element \
         by element and the program must convert; stderr={stderr}"
    );
}

const TRANSMUTE_ARRAY_SRC: &str = r#"
fn inp() -> u32 { let mut s = 0u32; let mut i = 0; while i < 41 { s += 1; i += 1; } s }
fn main() {
    let n = inp() + 0x04030200 - 41;
    let a: [u8; 4] = unsafe { std::mem::transmute(n) };
    let back: u32 = unsafe { std::mem::transmute(a) };
    let wide: u64 = unsafe { std::mem::transmute([1u16, 2, 3, 4]) };
    let pair: u64 = unsafe { std::mem::transmute([n, n + 1]) };
    let halves: [u32; 2] = unsafe { std::mem::transmute(pair) };
    let fpair: u64 = unsafe { std::mem::transmute([1.5f32, 2.5f32]) };
    println!("{:?} {:x} {:x} {:x} {:?} {:x}", a, back, wide, pair, halves, fpair);
}
"#;

#[test]
fn run_mode_transmute_arrays_and_scalars() {
    run_ok(
        "transmute-arrays",
        TRANSMUTE_ARRAY_SRC,
        "[0, 2, 3, 4] 4030200 4000300020001 403020104030200 [67305984, 67305985] 402000003fc00000\n",
    );
}

const TRANSMUTE_LONG_SCALAR_ARRAY_SRC: &str = r#"
fn main() {
    let mut a: [u32; 40] = [0; 40];
    for i in 0..40 { a[i] = 0x3fc00000 + (i as u32) * 0x10000; }
    let f: [f32; 40] = unsafe { std::mem::transmute(a) };
    println!("{} {} {}", f[0], f[1], f[39]);
    let back: [u32; 40] = unsafe { std::mem::transmute(f) };
    println!("{:x} {:x}", back[0], back[39]);
}
"#;

#[test]
fn run_mode_transmute_long_scalar_arrays() {
    // The same over-budget length with element shapes
    // that do not line up structurally (integer ->
    // float): the evaluation reinterprets element by
    // element instead of packing the whole array into
    // one 128-bit pattern.
    run_ok(
        "transmute-long-scalar-arrays",
        TRANSMUTE_LONG_SCALAR_ARRAY_SRC,
        "1.5 1.5078125 1.8046875\n3fc00000 3fe70000\n",
    );
}

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

const SLICE_ITER_MUT_SCALAR_BACKED_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn main() {
    let mut x = 41i32;
    let mut it = std::slice::from_mut(&mut x).iter_mut();
    println!("len {}", ExactSizeIterator::len(&it));
    println!("view {}", it.as_slice().len());
    let rest = it.into_slice();
    println!("rest {}", rest.len());
    let mut w = 900u64;
    let itw = std::slice::from_mut(&mut w).iter_mut();
    println!("wlen {}", ExactSizeIterator::len(&itw));
    println!("wview {}", itw.as_slice().len());
}
"#;

#[test]
fn run_mode_slice_iter_mut_scalar_backed_len() {
    // PR #387, Codex round 2: over a scalar-backed slice
    // (`slice::from_mut(&mut x)`), the iterator's front
    // pointer denotes the scalar place while the end pointer
    // denotes the same place with a trailing
    // `ByteOffset(size_of::<T>())`. The index reading of the
    // unified bounds drops that offset and reports an empty
    // iterator, and the snapshot-address fallback cannot read
    // a storage-backed pair at all, so the modelled `len`
    // and slice views failed instead of returning the
    // one-element result. They now read the pair's distance
    // through the same storage-backed offset reading as
    // `offset_from` before falling back to the bounds.
    let dir = temp_dir("slice-iter-mut-scalar-backed");
    let src = fixture(&dir, "case.rs", SLICE_ITER_MUT_SCALAR_BACKED_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "len 1\nview 1\nrest 1\nwlen 1\nwview 1\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_ACCESS_SRC: &str = r#"
fn main() {
    let mut x: i32 = 41;
    {
        let mut it = std::slice::from_mut(&mut x).iter_mut();
        println!("n1 {:?}", it.next().map(|v| *v));
        println!("n2 {:?}", it.next());
        println!("nb {:?}", it.next_back());
    }
    println!("x {}", x);
    let mut y: u64 = 900;
    {
        let mut it = std::slice::from_mut(&mut y).iter_mut();
        if let Some(v) = it.next() { *v = 1234; }
        println!("after {:?}", it.next());
    }
    println!("y {}", y);
    let mut z: u64 = 77;
    {
        let s = std::slice::from_mut(&mut z);
        let mut it = s.iter();
        println!("hint {:?}", it.size_hint());
        println!("back {:?}", it.next_back().map(|v| *v));
        println!("front {:?}", it.next().map(|v| *v));
        println!("len {}", std::iter::ExactSizeIterator::len(&it));
    }
    let w: u64 = 900;
    let sr = std::slice::from_ref(&w);
    println!("idx {}", sr[0]);
    println!("ptr {}", unsafe { *sr.as_ptr() });
    let mut sum = 0u64;
    for v in sr { sum += *v; }
    println!("sum {}", sum);
    let mut it = sr.iter();
    println!("first {:?}", it.next().map(|v| *v));
    println!("second {:?}", it.next());
    let n: i32 = -5;
    println!("neg {}", std::slice::from_ref(&n)[0]);
    let mut m: u32 = 0x01020304;
    {
        let sm = std::slice::from_mut(&mut m);
        sm[0] = 0xAABBCCDD;
    }
    println!("m {:x}", m);
}
"#;

#[test]
fn run_mode_slice_scalar_backed_access() {
    // Issue #413: over a slice formed on a scalar's own
    // storage (`slice::from_ref` / `slice::from_mut` of a
    // scalar place), three readings disagreed with the shape.
    // The step machine's index-pair shortcut read the
    // iterator's front and end pointers by their trailing
    // index step only, dropping the end pointer's byte offset,
    // so `next()` reported the one element as consumed. An
    // element access appended a bare index step to the scalar
    // place, which the projectors read as a byte of the
    // storage (900 read back as 132). And the shared `Iter`
    // construction rejected the scalar the snapshot read found
    // where a container was expected. Expected output is the
    // native result of the same source.
    let dir = temp_dir("slice-scalar-backed-access");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_ACCESS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "n1 Some(41)\nn2 None\nnb None\nx 41\nafter None\ny 1234\nhint (1, Some(1))\nback Some(77)\nfront None\nlen 0\nidx 900\nptr 900\nsum 900\nfirst Some(900)\nsecond None\nneg -5\nm aabbccdd\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_BYTE_VIEW_SRC: &str = r#"
fn main() {
    let w: u64 = 900;
    let b0 = unsafe { *(&w as *const u64 as *const u8) };
    let b1 = unsafe { *(&w as *const u64 as *const u8).add(1) };
    println!("bytes {} {}", b0, b1);
    let bs = unsafe { std::slice::from_raw_parts(&w as *const u64 as *const u8, 8) };
    let mut itb = bs.iter();
    println!("b0 {:?}", itb.next());
    println!("b1 {:?}", itb.next());
    println!("blen {}", bs.len());
    println!("bidx {}", bs[1]);
}
"#;

#[test]
fn run_mode_slice_scalar_backed_byte_view() {
    // The byte views of a scalar's storage keep their
    // byte-wise reading (a byte pointer into the storage, and
    // a byte slice over it): only an element the width of the
    // storage denotes the scalar itself. Expected output is
    // the native result of the same source.
    let dir = temp_dir("slice-scalar-backed-byte-view");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_BYTE_VIEW_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "bytes 132 3\nb0 Some(132)\nb1 Some(3)\nblen 8\nbidx 3\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_PUNNED_SRC: &str = r#"
fn main() {
    let w: u64 = 0x1_0000_0302;
    println!("idx {}", unsafe { std::slice::from_raw_parts(&w as *const u64 as *const u32, 2) }[0]);
}
"#;

#[test]
fn run_mode_slice_scalar_backed_punned_view_declines() {
    // A punned element width over a scalar's storage (u32
    // elements over a u64) holds no elements as values: the
    // access declines with an explicit error instead of
    // reading a byte of the storage (the pre-#413 reading
    // printed 2 here, the low byte, where the native element
    // is 770).
    let dir = temp_dir("slice-scalar-backed-punned");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_PUNNED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "stderr={stderr}");
    assert!(
        stderr.contains("punned element width"),
        "stderr={stderr}"
    );
}

const SLICE_ITER_ENUMERATE_CONST_PTR_SRC: &str = r#"
trait Dimension {
    fn slice(&self) -> &[usize];
}
impl Dimension for () {
    fn slice(&self) -> &[usize] { &[] }
}
impl Dimension for usize {
    fn slice(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self, 1) }
    }
}
fn main() {
    let tup = ();
    let mut it = tup.slice().iter();
    for (i, _) in 3usize.slice().iter().enumerate() {
        if i == 0 { continue; }
        it.next();
    }
    println!("done");
}
"#;

#[test]
fn run_mode_slice_iter_enumerate_const_ptr_o3() {
    // The boundary of the scalar-backed first-element rule:
    // a slice formed over a pointer to a constant value (the
    // `from_raw_parts` over a promoted `&3` shape, at the
    // opt level the shape's regression test compiles at) is
    // not storage-backed — its element position is the
    // appended index the offset machinery advances, and
    // iterating it must keep working.
    let dir = temp_dir("slice-iter-enumerate-const-ptr");
    let src = fixture(&dir, "case.rs", SLICE_ITER_ENUMERATE_CONST_PTR_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "done\n", "stdout={stdout}");
}

const SLICE_SCALAR_BACKED_BOOL_SRC: &str = r#"
fn main() {
    let t = true;
    let s = std::slice::from_ref(&t);
    let mut it = s.iter();
    println!("first {:?}", it.next());
    println!("second {:?}", it.next());
    let f = false;
    let mut sum = 0;
    for v in std::slice::from_ref(&f) {
        if *v {
            sum += 1;
        }
    }
    println!("sum {sum}");
    println!("idx {}", s[0]);
}
"#;

#[test]
fn run_mode_slice_scalar_backed_bool_element() {
    // A one-byte element of the storage's own type is the
    // scalar itself, not its byte: `slice::from_ref(&true)`
    // iterated as a byte snapshot yields `Uint(1)`, whose
    // pointee cannot be read as the `bool` the iterator
    // promises. Expected output is the native result of the
    // same source.
    let dir = temp_dir("slice-scalar-backed-bool");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_BOOL_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "first Some(true)\nsecond None\nsum 0\nidx true\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_FLOAT_VIEW_SRC: &str = r#"
#![feature(f16)]
fn main() {
    let x: f32 = 1.0;
    let bs = unsafe { std::slice::from_raw_parts(&x as *const f32 as *const u8, 4) };
    let mut it = bs.iter();
    println!("b0 {:?}", it.next());
    println!("b1 {:?}", it.next());
    println!("b2 {:?}", it.next());
    println!("b3 {:?}", it.next());
    let y: f64 = 1.0;
    let bs8 = unsafe { std::slice::from_raw_parts(&y as *const f64 as *const u8, 8) };
    let v: Vec<u8> = bs8.iter().copied().collect();
    println!("f64 {v:?}");
    let h: f16 = 1.0;
    let bsh = unsafe { std::slice::from_raw_parts(&h as *const f16 as *const u8, 2) };
    let vh: Vec<u8> = bsh.iter().copied().collect();
    println!("f16 {vh:?}");
}
"#;

#[test]
fn run_mode_slice_scalar_backed_float_byte_view() {
    // A byte view over a float's storage reads the bytes of
    // the float's own width under its declared layout, not
    // the widened model form's: a four-byte view of `1.0f32`
    // iterates as [0, 0, 128, 63]. Expected output is the
    // native result of the same source.
    let dir = temp_dir("slice-scalar-backed-float-view");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_FLOAT_VIEW_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "b0 Some(0)\nb1 Some(0)\nb2 Some(128)\nb3 Some(63)\nf64 [0, 0, 0, 0, 0, 0, 240, 63]\nf16 [0, 60]\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_SIGNED_BYTE_SRC: &str = r#"
fn main() {
    let n = -1i8;
    let s = std::slice::from_ref(&n);
    let mut it = s.iter();
    println!("first {:?}", it.next());
    println!("second {:?}", it.next());
    let m = -128i8;
    for v in std::slice::from_ref(&m) {
        println!("elem {v}");
    }
    let x: u32 = 0x80FF_7F01;
    let bs = unsafe { std::slice::from_raw_parts(&x as *const u32 as *const i8, 4) };
    let v: Vec<i8> = bs.iter().copied().collect();
    println!("view {v:?}");
    let p = 127i8;
    let s2 = std::slice::from_ref(&p);
    println!("pos {:?}", s2.iter().next());
}
"#;

#[test]
fn run_mode_slice_scalar_backed_signed_byte_element() {
    // A byte element of a signed type denotes the byte's
    // signed value: `slice::from_ref(&-1i8)` snapshotted as
    // an unsigned byte yields `Uint(255)`, which a consumer
    // resolving the yielded reference directly (the
    // formatting path) prints as 255 instead of -1. The same
    // holds for a signed byte view over a wider storage.
    // Expected output is the native result of the same
    // source.
    let dir = temp_dir("slice-scalar-backed-signed-byte");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_SIGNED_BYTE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "first Some(-1)\nsecond None\nelem -128\nview [1, 127, -1, -128]\npos Some(127)\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_BYTE_VIEW_OF_BOOL_SRC: &str = r#"
fn main() {
    let b = true;
    let bytes = unsafe { std::slice::from_raw_parts(&b as *const bool as *const u8, 1) };
    let n: u8 = bytes[0];
    println!("n={n}");
    println!("r={:?}", &bytes[0]);
    let b2 = false;
    let bytes2 = unsafe { std::slice::from_raw_parts(&b2 as *const bool as *const u8, 1) };
    println!("n2={}", bytes2[0]);
    println!("r2={:?}", &bytes2[0]);
    let m = -1i8;
    let sbytes = unsafe { std::slice::from_raw_parts(&m as *const i8 as *const u8, 1) };
    println!("s={:?}", &sbytes[0]);
    let bb = unsafe { std::slice::from_raw_parts(&b as *const bool as *const bool, 1) };
    println!("same={} {:?}", bb[0], &bb[0]);
}
"#;

#[test]
fn run_mode_slice_scalar_backed_byte_view_of_bool() {
    // A byte view of a boolean's storage denotes the storage's
    // byte, not the boolean: the same-width shortcut in place
    // resolution left the element position on the boolean
    // itself, so a consumer resolving the element reference
    // directly (the formatting path) printed `true` instead of
    // `1`. The same held for a byte view of an `i8` (printed
    // `-1` instead of `255`). A same-kind view still denotes
    // the scalar itself. Expected output is the native result
    // of the same source.
    let dir = temp_dir("slice-scalar-backed-byte-view-of-bool");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_BYTE_VIEW_OF_BOOL_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "n=1\nr=1\nn2=0\nr2=0\ns=255\nsame=true true\n",
        "stdout={stdout}"
    );
}

const SLICE_SCALAR_BACKED_BYTE_VIEW_AS_PTR_SRC: &str = r#"
fn main() {
    let b = true;
    let bytes = unsafe { std::slice::from_raw_parts(&b as *const bool as *const u8, 1) };
    let p: *const u8 = bytes.as_ptr();
    let n: u8 = unsafe { *p };
    println!("n={n}");
    let r: &u8 = unsafe { &*bytes.as_ptr() };
    println!("r={r}");
    let b2 = false;
    let bytes2 = unsafe { std::slice::from_raw_parts(&b2 as *const bool as *const u8, 1) };
    let r2: &u8 = unsafe { &*bytes2.as_ptr() };
    println!("r2={r2}");
    let m = -1i8;
    let sbytes = unsafe { std::slice::from_raw_parts(&m as *const i8 as *const u8, 1) };
    let sr: &u8 = unsafe { &*sbytes.as_ptr() };
    println!("s={sr}");
    let same = unsafe { std::slice::from_raw_parts(&b as *const bool as *const bool, 1) };
    let br: &bool = unsafe { &*same.as_ptr() };
    println!("same={br}");
    let mut b3 = false;
    let mbytes = unsafe { std::slice::from_raw_parts_mut(&mut b3 as *mut bool as *mut u8, 1) };
    unsafe { *mbytes.as_mut_ptr() = 1u8; }
    println!("w={b3}");
}
"#;

#[test]
fn run_mode_slice_scalar_backed_byte_view_as_ptr() {
    // The first-element pointer a byte view of a scalar's
    // storage extracts (`as_ptr` / `as_mut_ptr`) denotes the
    // storage's byte, not the scalar itself: the scalar
    // shortcut in the pointer extraction returned the storage
    // pointer as it stands, so a consumer resolving the
    // reference the pointer forms directly printed the boolean
    // (`true` instead of `1`), and a byte write through
    // `as_mut_ptr` met the boolean place's type check and
    // failed. A same-kind view's pointer still denotes the
    // scalar itself. Expected output is the native result of
    // the same source.
    let dir = temp_dir("slice-scalar-backed-byte-view-as-ptr");
    let src = fixture(&dir, "case.rs", SLICE_SCALAR_BACKED_BYTE_VIEW_AS_PTR_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "n=1\nr=1\nr2=0\ns=255\nsame=true\nw=true\n",
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

// ----- large array fills (Issue #140) ------------------------------------------

const LARGE_FILL_SRC: &str = r#"
use std::mem::MaybeUninit;

fn main() {
    // `mem::zeroed` of a large scalar array materialises the
    // whole zero tree.
    let z: [u64; 5000] = unsafe { std::mem::zeroed() };
    assert_eq!(z[4999], 0);
    let nested: [[u16; 300]; 300] = unsafe { std::mem::zeroed() };
    assert_eq!(nested[299][299], 0);
    // `write_bytes` whose element type is the whole array (the
    // shape `mem::zeroed` takes under optimisation) fills it in
    // one operation.
    let mut un: MaybeUninit<[u8; 1048576]> = MaybeUninit::uninit();
    unsafe {
        std::ptr::write_bytes(un.as_mut_ptr(), 0u8, 1);
        let arr = un.assume_init_read();
        assert_eq!(arr[0], 0);
        assert_eq!(arr[1048575], 0);
        std::ptr::write_bytes(un.as_mut_ptr(), 0x5Au8, 1);
        let arr = un.assume_init_read();
        assert_eq!(arr[524288], 0x5A);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_large_array_zeroed_and_write_bytes_execute() {
    let dir = temp_dir("large-fill");
    let src = fixture(&dir, "large_fill.rs", LARGE_FILL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "large array zeroed / write_bytes fills must execute; stderr={stderr}"
    );
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

const WINDOWS_DISPLACED_SRC: &str = r#"
pub trait Iter {
    type Item<'a> where Self: 'a;
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
    fn for_each<F>(mut self, mut f: F)
    where Self: Sized, F: for<'a> FnMut(Self::Item<'a>)
    {
        while let Some(item) = self.next() {
            f(item);
        }
    }
}
pub struct Windows<T> {
    items: Vec<T>,
    start: usize,
    len: usize,
}
impl<T> Windows<T> {
    pub fn new(items: Vec<T>, len: usize) -> Self {
        Self { items, start: 0, len }
    }
}
impl<T> Iter for Windows<T> {
    type Item<'a> = &'a mut [T] where T: 'a;
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        let slice = self.items.get_mut(self.start..self.start + self.len)?;
        self.start += 1;
        Some(slice)
    }
}
fn main() {
    Windows::new(vec![1, 2, 3, 4, 5], 3)
        .for_each(|slice| {
            println!("{:?}", slice);
            assert_eq!(slice.len(), 3);
            let mut sum = 0;
            for x in slice.iter() {
                sum += x;
            }
            println!("sum={}", sum);
        });
    println!("ok");
}
"#;

#[test]
fn run_mode_windows_displaced_slice_reads_elements() {
    // A slice formed by displacing a container pointer by a whole
    // number of elements (the `windows` shape: a range index into
    // the middle of a Vec) denotes the elements from that position
    // on: its length, its iterated elements and its Debug rendering
    // all see the displaced window, not just the first element.
    let dir = temp_dir("windows-displaced");
    let src = fixture(&dir, "windows_displaced.rs", WINDOWS_DISPLACED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "displaced window slices must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3]\nsum=6\n[2, 3, 4]\nsum=9\n[3, 4, 5]\nsum=12\nok\n"
    );
}

const WINDOWS_NESTED_VEC_SRC: &str = r#"
fn main() {
    let v: Vec<Vec<usize>> = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
    for w in v.windows(2) {
        println!("w {} {} {} {}", w[0][0], w[1][0], w[0].len(), w[1].len());
    }
    let u: Vec<Vec<usize>> = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let w = u.windows(2).next().unwrap();
    println!("c {} {}", w[0][1], w[1][0]);
    let a: Vec<[u8; 2]> = vec![[1, 2], [3, 4], [5, 6]];
    let w = a.windows(2).next().unwrap();
    println!("a {}", w[0][1]);
}
"#;

#[test]
fn run_mode_windows_over_vec_of_vecs_index_reads() {
    // A window the `Windows` step yields is a snapshot over
    // the elements it materialised: indexing the window reads
    // the window's elements themselves. The step used to
    // denominate the snapshot's base as an element position
    // inside its own element tuple, so an index step through
    // the window composed onto that position and landed
    // inside the first element whenever the element was
    // itself a sequence: `w[0]` over a `Vec<Vec<usize>>`
    // handed back the inner vector's first element (and
    // `w[0].len()` died on the integer it found), for every
    // inner-length shape -- the ones whose length equals the
    // window size included. The outputs below are native
    // rustc 1.100.0's.
    run_ok(
        "windows-nested-vec",
        WINDOWS_NESTED_VEC_SRC,
        "w 1 4 3 2\nw 4 6 2 4\nc 2 3\na 2\n",
    );
}

const CHUNKS_NESTED_VEC_SRC: &str = r#"
fn main() {
    let v: Vec<Vec<usize>> = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9], vec![10]];
    for c in v.chunks(2) {
        println!("chunk {} {}", c[0][0], c[1][0]);
    }
    for c in v.rchunks(2) {
        println!("rchunk {} {}", c[0][0], c[1][0]);
    }
}
"#;

#[test]
fn run_mode_chunks_over_vec_of_vecs_index_reads() {
    // The chunk family's yielded sub-slices are snapshots
    // denominated exactly as the `Windows` step's windows
    // are (see `run_mode_windows_over_vec_of_vecs_index_reads`):
    // indexing a chunk of a `Vec<Vec<usize>>` reads the
    // chunk's elements, not the inside of its first element.
    run_ok(
        "chunks-nested-vec",
        CHUNKS_NESTED_VEC_SRC,
        "chunk 1 4\nchunk 6 10\nrchunk 6 10\nrchunk 1 4\n",
    );
}

const CHUNKS_WINDOWS_LEN_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn main() {
    let v = [1, 2, 3, 4, 5];
    println!("c_rem {}", v.chunks(2).len());
    println!("c_exact {}", v.chunks(5).len());
    println!("c_big {}", v.chunks(9).len());
    println!("c_one {}", v.chunks(1).len());
    println!("w_rem {}", v.windows(3).len());
    println!("w_exact {}", v.windows(5).len());
    println!("w_big {}", v.windows(6).len());
    println!("w_one {}", v.windows(1).len());
    let e: [i32; 0] = [];
    println!("c_empty {}", e.chunks(2).len());
    println!("w_empty {}", e.windows(1).len());
    println!("ce_empty {}", e.chunks_exact(2).len());
    println!("hint_c {:?}", v.chunks(2).size_hint());
    println!("hint_w {:?}", v.windows(3).size_hint());
    println!("hint_ce {:?}", v.chunks_exact(2).size_hint());
    let mut w = v.windows(2);
    let _ = w.next();
    println!("w_after1 {}", w.len());
    let _ = w.next();
    let _ = w.next();
    let _ = w.next();
    println!("w_after4 {}", w.len());
    let mut c = v.chunks(2);
    let rc = &mut c;
    println!("refmut_len {}", rc.len());
    let c2 = v.chunks(3);
    let r2 = &c2;
    println!("ref_len {}", r2.len());
    println!("ce_rem {:?}", v.chunks_exact(2).remainder());
    println!("rce_rem {:?}", v.rchunks_exact(2).remainder());
    let vec = vec![10, 20, 30, 40, 50, 60, 70];
    println!("vec_c {}", vec.chunks(3).len());
    println!("vec_w {}", vec.windows(4).len());
    let z = [(), (), (), ()];
    println!("zst_c {}", z.chunks(3).len());
    println!("zst_w {}", z.windows(2).len());
    let s = "hello".as_bytes();
    println!("bytes_c {}", s.chunks(2).len());
}
"#;

#[test]
fn run_mode_chunks_windows_len() {
    // `ExactSizeIterator::len` for the slice chunk/window
    // family: the trait default body asserts `size_hint`'s
    // upper bound equals its lower bound, and the modelled
    // `size_hint` answered the trait default `(0, None)` for
    // these types, so every `len()` call died in core's
    // `assert_eq!`. The modelled `size_hint` now computes
    // std's exact counts (ceiling division by the chunk size,
    // floor for the exact variants, `remaining - size + 1`
    // for windows), and the outputs below are native
    // rustc 1.100.0's.
    let dir = temp_dir("chunks-windows-len");
    let src = fixture(&dir, "chunks_windows_len.rs", CHUNKS_WINDOWS_LEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "chunks/windows len must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c_rem 3\nc_exact 1\nc_big 1\nc_one 5\nw_rem 3\nw_exact 1\nw_big 0\nw_one 5\nc_empty 0\nw_empty 0\nce_empty 0\nhint_c (3, Some(3))\nhint_w (3, Some(3))\nhint_ce (2, Some(2))\nw_after1 3\nw_after4 0\nrefmut_len 3\nref_len 2\nce_rem [5]\nrce_rem [1]\nvec_c 3\nvec_w 4\nzst_c 2\nzst_w 3\nbytes_c 3\n"
    );
}


const WINDOWS_COLLECT_NESTED_SRC: &str = r#"
fn main() {
    let v: Vec<Vec<usize>> = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let w: Vec<&[Vec<usize>]> = v.windows(2).collect();
    println!("len {}", w.len());
    println!("w0 {:?}", w[0]);
    println!("w1 {:?}", w[1]);
    println!("first {}", w[0][0][0]);
    println!("last {}", w[1][1][1]);
}
"#;

#[test]
fn run_mode_windows_collect_nested_vec_and_drop() {
    // Collecting `windows` into a `Vec` and dropping the
    // result stopped at an unsupported `Drop` terminator:
    // the glue conversion attached to the consuming
    // `collect` call planned the `Windows` iterator's fields
    // from its native two-field shape -- the second field is
    // a `NonZero<usize>`, whose inner type is an associated
    // type the evidence tables cannot resolve, so its plan
    // stayed a one-field `Untyped` plan and never collapsed
    // -- and then tried to run that plan against the modelled
    // state, a [slice, window size, next index] snapshot with
    // three fields. `NonZero` is glue-free (a primitive
    // integer in a niche wrapper) and the borrowing slice
    // iterators own nothing, so both plans are trivial now.
    // The collected windows also read through the window
    // slices the step machine manufactures: those denoted
    // their snapshot tuple at an element position, which the
    // slice reader mistook for the first element's own
    // elements when that element was itself a sequence of
    // the window's length (here a `Vec` of two numbers), so
    // `w[0]` printed the first inner vector instead of the
    // window. The manufactured slices now denote the
    // snapshot itself.
    run_ok(
        "windows-collect-nested",
        WINDOWS_COLLECT_NESTED_SRC,
        "len 2\nw0 [[1, 2], [3, 4]]\nw1 [[3, 4], [5, 6]]\nfirst 1\nlast 6\n",
    );
}

const WINDOWS_COLLECT_SCALAR_SRC: &str = r#"
fn main() {
    let v = vec![1usize, 2, 3, 4];
    let w: Vec<&[usize]> = v.windows(2).collect();
    println!("len {}", w.len());
    println!("w0 {:?}", w[0]);
    println!("w2 {:?}", w[2]);
    let nested: Vec<Vec<usize>> = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let c: Vec<&[Vec<usize>]> = nested.chunks(2).collect();
    println!("chunks {} {:?} {:?}", c.len(), c[0][0], c[1][0]);
}
"#;

#[test]
fn run_mode_windows_collect_scalar_and_chunks_collect_nested() {
    // The scalar-element shape of the same collect-and-drop
    // path (the `Drop` failure did not depend on the element
    // type), and the chunk family's manufactured slices over
    // nested vectors, which had the same element-position
    // denomination as the window slices.
    run_ok(
        "windows-collect-scalar",
        WINDOWS_COLLECT_SCALAR_SRC,
        "len 3\nw0 [1, 2]\nw2 [3, 4]\nchunks 2 [1, 2] [5, 6]\n",
    );
}

const CHUNKS_MUT_LEN_SRC: &str = r#"
use std::iter::ExactSizeIterator;
fn main() {
    let v = [1, 2, 3, 4, 5];
    println!("ce {}", v.chunks_exact(2).len());
    println!("rce {}", v.rchunks_exact(2).len());
    println!("rc {}", v.rchunks(2).len());
    let mut m = [1, 2, 3, 4, 5];
    println!("cm {}", m.chunks_mut(2).len());
    let mut m2 = [1, 2, 3, 4, 5];
    println!("cem {}", m2.chunks_exact_mut(2).len());
    let mut m3 = [1, 2, 3, 4, 5];
    println!("rcm {}", m3.rchunks_mut(2).len());
    let mut m4 = [1, 2, 3, 4, 5];
    println!("rcem {}", m4.rchunks_exact_mut(2).len());
}
"#;

#[test]
fn run_mode_chunks_mut_and_exact_len() {
    // The mutable and exact variants of the chunk family
    // store the remaining slice differently (a raw fat
    // pointer plus a marker field for the `Mut` shapes), but
    // `len` is the same count family: ceiling division for
    // the plain variants, floor division for the exact ones.
    let dir = temp_dir("chunks-mut-len");
    let src = fixture(&dir, "chunks_mut_len.rs", CHUNKS_MUT_LEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "chunks mut/exact len must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "ce 2\nrce 2\nrc 3\ncm 3\ncem 2\nrcm 3\nrcem 2\n"
    );
}

// Slice unsizing of pointer-carrying values: a reference-counted
// handle is itself the pointer (its pointee lives in the
// allocation store), a custom DST propagates the metadata
// through its trailing field, and a byte-slice constant in
// pointer position is the fat pointer, not the pointee data.

const UNSIZE_RC_SLICE_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let a: Rc<[i32; 3]> = Rc::new([1, 2, 3]);
    let b: Rc<[i32]> = a;
    println!("len={} {} {} {}", b.len(), b[0], b[1], b[2]);
    let c = b.clone();
    println!("strong={} clen={}", Rc::strong_count(&b), c.len());
}
"#;

#[test]
fn run_mode_unsize_rc_slice_handle() {
    run_ok("unsize-rc-slice", UNSIZE_RC_SLICE_SRC, "len=3 1 2 3\nstrong=2 clen=3\n");
}

const UNSIZE_BYTE_STATICS_SRC: &str = r#"
static BYTES: &[u8] = &[1, 2, 3];
const CB: &[u8] = b"ab";
fn take(s: &[u8]) { println!("take {} {}", s.len(), s[2]); }
fn main() {
    println!("st {} {}", BYTES.len(), BYTES[1]);
    take(BYTES);
    println!("cb {} {}", CB.len(), CB[0]);
}
"#;

#[test]
fn run_mode_unsize_byte_slice_statics_are_fat_pointers() {
    run_ok(
        "unsize-byte-statics",
        UNSIZE_BYTE_STATICS_SRC,
        "st 3 2\ntake 3 3\ncb 2 97\n",
    );
}

const UNSIZE_RAW_ONE_ELEM_SRC: &str = r#"
fn main() {
    let b = Box::new([5u8]);
    let raw = Box::into_raw(b);
    let raw2 = raw as *mut [u8];
    let s: &[u8] = unsafe { &*raw2 };
    println!("len={} v={}", s.len(), s[0]);
    let b2 = unsafe { Box::from_raw(raw2) };
    drop(b2);
    println!("done");
}
"#;

#[test]
fn run_mode_unsize_raw_pointer_slice_view() {
    run_ok("unsize-raw-view", UNSIZE_RAW_ONE_ELEM_SRC, "len=1 v=5\ndone\n");
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

const NESTED_VEC_SRC: &str = r#"
fn main() {
    let mut v: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4], vec![5, 6]];
    assert_eq!(v[0][1], 2);
    assert_eq!(v[1][0], 5);
    v[0].reverse();
    v[1].push(7);
    assert_eq!(v[0], vec![4, 3, 2, 1]);
    assert_eq!(v[1][2], 7);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_nested_vec_index_and_reverse() {
    // A `Vec` value is the aggregate of its elements, so an
    // index step into it is element access and a following index
    // step ranges inside it: composing the two steps into one
    // index over the outer vector (the flat reading a data
    // pointer's position takes) reads the sibling vector whole
    // instead of the element. Nested reads, writes through
    // `v[0].reverse()`'s swaps, and growth of an inner vector
    // must all observe the inner elements.
    let dir = temp_dir("nestedvec");
    let src = fixture(&dir, "nestedvec.rs", NESTED_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested vec index/reverse must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[[4, 3, 2, 1], [5, 6, 7]]\n",
        "the inner vectors reverse and grow independently, exactly as natively"
    );
}
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

const VEC_STRING_SORT_SRC: &str = r#"
fn main() {
    let mut v: Vec<String> = vec![
        "banana".to_string(),
        "apple".to_string(),
        "cherry".to_string(),
        "apple".to_string(),
    ];
    v.sort();
    for s in &v { println!("sorted: {s}"); }
    v.reverse();
    for s in &v { println!("reversed: {s}"); }
    v.sort_unstable();
    for s in &v { println!("unstable: {s}"); }
    println!("binsearch apple: {:?}", v.binary_search(&"apple".to_string()));
    v.dedup();
    for s in &v { println!("dedup: {s}"); }
    let mut refs: Vec<&str> = vec!["pear", "fig", "plum", "fig"];
    refs.sort();
    for s in &refs { println!("refs: {s}"); }
    let mut w: Vec<String> = vec!["a".to_string(), "bb".to_string(), "ccc".to_string()];
    w.sort_by_key(|s| s.len());
    for s in &w { println!("by len: {s}"); }
    println!("vec-string-sort-ok");
}
"#;

#[test]
fn run_mode_vec_string_sort() {
    run_ok(
        "vec-string-sort",
        VEC_STRING_SORT_SRC,
        "sorted: apple\nsorted: apple\nsorted: banana\nsorted: cherry\nreversed: cherry\nreversed: banana\nreversed: apple\nreversed: apple\nunstable: apple\nunstable: apple\nunstable: banana\nunstable: cherry\nbinsearch apple: Ok(1)\ndedup: apple\ndedup: banana\ndedup: cherry\nrefs: fig\nrefs: fig\nrefs: pear\nrefs: plum\nby len: a\nby len: bb\nby len: ccc\nvec-string-sort-ok\n",
    );
}

const VEC_POP_AFTER_SHRINK_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(16);
    v.push(1);
    v.push(2);
    v.shrink_to_fit();
    println!("shrunk len={}", v.len());
    match v.pop() {
        Some(x) => println!("popped {}", x),
        None => println!("none"),
    }
    println!("after len={}", v.len());
    match v.pop() {
        Some(x) => println!("popped {}", x),
        None => println!("none"),
    }
    println!("final len={} empty={}", v.len(), v.is_empty());
}
"#;

#[test]
fn run_mode_vec_pop_after_shrink() {
    // `pop`'s collected body reads the length slot, writes the
    // decremented length back, reads the capacity slot for its
    // `assert_unchecked(len < capacity)`, and reads the element
    // at the new length through the buffer pointer: the length
    // write must not destroy the element it is about to read,
    // and the capacity answer must stay the storage extent
    // across the write (see the vec length/storage rule in
    // `vec_with_len`).
    run_ok(
        "vec-pop-after-shrink",
        VEC_POP_AFTER_SHRINK_SRC,
        "shrunk len=2\npopped 2\nafter len=1\npopped 1\nfinal len=0 empty=true\n",
    );
}

const VEC_APPEND_TAIL_SRC: &str = r#"
fn main() {
    let mut a = vec![1, 2];
    let mut b = vec![3, 4, 5];
    a.append(&mut b);
    println!("a={:?} len={} cap={}", a, a.len(), a.capacity());
    println!("b={:?} len={}", b, b.len());
    println!("a[0]={} a[3]={} a[4]={}", a[0], a[3], a[4]);
}
"#;

#[test]
fn run_mode_vec_append_tail() {
    // `append`'s collected body copies the source elements
    // through the destination's buffer pointer at the old
    // length and only then writes the length slot: the raw
    // writes extend the storage without moving the declared
    // length, and the final length write lands on the extended
    // storage.
    run_ok(
        "vec-append-tail",
        VEC_APPEND_TAIL_SRC,
        "a=[1, 2, 3, 4, 5] len=5 cap=5\nb=[] len=0\na[0]=1 a[3]=4 a[4]=5\n",
    );
}

const VEC_SPLIT_OFF_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3, 4, 5, 6];
    v.truncate(3);
    println!("trunc v={:?} len={}", v, v.len());
    let mut w = vec![1, 2, 3, 4, 5];
    let tail = w.split_off(2);
    println!("w={:?} tail={:?}", w, tail);
}
"#;

#[test]
fn run_mode_vec_split_off() {
    // `split_off` builds its result in the real aggregate form
    // inside its body (an inlined `with_capacity_in`, `set_len`,
    // and a raw copy into the buffer allocation): the value
    // translates to the element form as it returns to the
    // caller (see `vec_from_real_form`).
    run_ok(
        "vec-split-off",
        VEC_SPLIT_OFF_SRC,
        "trunc v=[1, 2, 3] len=3\nw=[1, 2] tail=[3, 4, 5]\n",
    );
}

const VEC_DRAIN_COLLECT_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];
    let taken: Vec<i32> = v.drain(1..3).collect();
    println!("taken={:?}", taken);
    let mut u = vec![10, 20, 30];
    let all: Vec<i32> = u.drain(..).collect();
    println!("all={:?}", all);
}
"#;

#[test]
fn run_mode_vec_drain_collect() {
    // A drain's iteration steps its slice iterator through the
    // source vector's storage (see `drain_next_state`); the
    // collected elements are the drained range.
    run_ok(
        "vec-drain-collect",
        VEC_DRAIN_COLLECT_SRC,
        "taken=[2, 3]\nall=[10, 20, 30]\n",
    );
}

const VEC_INSERT_GROW_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    v.insert(3, 4);
    println!("{:?} len={} cap={}", v, v.len(), v.capacity());
    v.insert(1, 99);
    println!("{:?}", v);
    v.insert(0, 0);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_vec_insert_positions_with_growth() {
    // `Vec::insert` runs its collected body: the capacity
    // check and the regrowth walk the buffer wrapper chain
    // (`RawVec` / `RawVecInner`) through references handed
    // across calls, and the chain's bookkeeping slots are the
    // vector's own storage extent and declared length in this
    // model. End, middle and start inserts over a growing
    // vector land exactly as natively, capacity included.
    let dir = temp_dir("vecinsertgrow");
    let src = fixture(&dir, "vecinsertgrow.rs", VEC_INSERT_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert with growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 2, 3, 4] len=4 cap=6\n[1, 99, 2, 3, 4]\n[0, 1, 99, 2, 3, 4]\n"
    );
}

const VEC_INSERT_RESERVED_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(16);
    v.push(10);
    v.push(20);
    v.push(30);
    v.insert(1, 15);
    println!("{:?} cap={}", v, v.capacity());
    assert_eq!(v, vec![10, 15, 20, 30]);
    let mut s = vec![String::from("a"), String::from("c")];
    s.insert(1, String::from("b"));
    assert_eq!(s, vec![String::from("a"), String::from("b"), String::from("c")]);
    println!("ok");
}
"#;

#[test]
fn run_mode_vec_insert_within_capacity_and_equality() {
    // An insert inside the reserved capacity does not grow:
    // `with_capacity` materialises the requested storage
    // extent, so the capacity stays 16 and the elements shift
    // in place. Whole-vector equality after a mutation
    // compares the logical element sequence: the sparse form
    // a growth or a shrink leaves behind (declared length
    // below the stored elements) equals the plain aggregate
    // form of the same elements.
    let dir = temp_dir("vecinsertreserved");
    let src = fixture(&dir, "vecinsertreserved.rs", VEC_INSERT_RESERVED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert within capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[10, 15, 20, 30] cap=16\nok\n"
    );
}

const VEC_INSERT_NESTED_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    v.insert(1, vec![3]);
    println!("{:?}", v);
    let mut t = (vec![1, 3], 7);
    t.0.insert(1, 2);
    println!("{:?}", t);
}
"#;

#[test]
fn run_mode_vec_insert_nested_and_tuple_field() {
    // The inserted element may itself be a vector, and the
    // mutated vector may live inside another aggregate: the
    // chain re-entry judges the place the buffer pointer
    // denotes by the value it currently holds, not by the
    // declared type of the local that happens to contain it.
    let dir = temp_dir("vecinsertnested");
    let src = fixture(&dir, "vecinsertnested.rs", VEC_INSERT_NESTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert on nested/tuple-field vectors must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[[1, 2], [3], [4]]\n([1, 2, 3], 7)\n"
    );
}

const VEC_INSERT_SHRINK_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(16);
    v.push(1);
    v.push(2);
    v.insert(1, 9);
    println!("{:?} cap={}", v, v.capacity());
    v.shrink_to_fit();
    println!("{:?} cap={}", v, v.capacity());
    v.insert(3, 5);
    println!("{:?} cap={}", v, v.capacity());
}
"#;

#[test]
fn run_mode_vec_insert_after_shrink_to_fit() {
    // After a shrink the storage extent is the length itself,
    // so the next insert grows again from there: the capacity
    // slot writes of both paths (the shrink's truncation and
    // the regrowth's amortised capacity) rebuild the vector's
    // storage extent exactly.
    let dir = temp_dir("vecinsertshrink");
    let src = fixture(&dir, "vecinsertshrink.rs", VEC_INSERT_SHRINK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert after shrink_to_fit must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[1, 9, 2] cap=16\n[1, 9, 2] cap=3\n[1, 9, 2, 5] cap=6\n"
    );
}

const BOX_VEC_GROW_SRC: &str = r#"
struct D(&'static str);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let mut b: Box<Vec<D>> = Box::new(Vec::with_capacity(2));
    b.push(D("a"));
    b.push(D("b"));
    b.insert(1, D("x"));
    b.reserve(16);
    for d in b.iter() {
        println!("elem {}", d.0);
    }
    drop(b);
    println!("done");
}
"#;

#[test]
fn run_mode_box_vec_insert_and_reserve_with_growth() {
    // Growing a `Vec` that is the whole content of a `Box`
    // allocation through the collected `insert` / `reserve`
    // bodies: the vector's buffer pointer is derived from the
    // vector place inside the box's allocation, and the
    // reallocation must not free or move the box's storage --
    // the elements and their drop order land exactly as
    // natively and the box stays live throughout.
    let dir = temp_dir("boxvecgrow");
    let src = fixture(&dir, "boxvecgrow.rs", BOX_VEC_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<Vec> growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "elem a\nelem x\nelem b\ndrop a\ndrop x\ndrop b\ndone\n"
    );
}

const BOX_BOX_VEC_GROW_SRC: &str = r#"
fn main() {
    let mut b: Box<Box<Vec<i32>>> = Box::new(Box::new(Vec::with_capacity(2)));
    b.push(1);
    b.push(2);
    b.push(3);
    b.insert(1, 9);
    println!("{:?}", b);
}
"#;

#[test]
fn run_mode_box_box_vec_insert_with_growth() {
    // The same growth through two owning wrappers: the inner
    // vector's buffer pointer denotes a position inside the
    // inner box's stored value, never a wrapper allocation's
    // base.
    let dir = temp_dir("boxboxvecgrow");
    let src = fixture(&dir, "boxboxvecgrow.rs", BOX_BOX_VEC_GROW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<Box<Vec>> growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "[1, 9, 2, 3]\n");
}

const NESTED_VEC_POP_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    let x = v[0].pop();
    println!("{:?} {:?}", v, x);
}
"#;

#[test]
fn run_mode_nested_vec_pop_after_shrink() {
    // A collected method (`pop`) on a vector element reached
    // through an index projection: `pop` shrinks the element
    // in place, so the element takes the sparse form, and the
    // body's own reads of the element's elements (through the
    // buffer pointer the method forms) must range inside the
    // element, not compose into positions of the outer vector.
    run_ok(
        "nested-vec-pop",
        NESTED_VEC_POP_SRC,
        "[[1], [4]] Some(2)\n",
    );
}

const NESTED_VEC_INSERT_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    v.insert(1, vec![3]);
    println!("{:?}", v);
    v[0].insert(1, 9);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_nested_vec_insert_after_outer_insert() {
    // After the outer vector's own collected `insert` leaves
    // it in the sparse form, a collected `insert` on one of its
    // elements forms its mutable receiver through `IndexMut`
    // over the outer place: the sparse form is the same element
    // aggregate as the plain form and is addressable the same
    // way.
    run_ok(
        "nested-vec-insert",
        NESTED_VEC_INSERT_SRC,
        "[[1, 2], [3], [4]]\n[[1, 9, 2], [3], [4]]\n",
    );
}

const NESTED_VEC_SET_LEN_READS_SRC: &str = r#"
fn main() {
    let mut v = vec![vec![1, 2], vec![4]];
    unsafe { v[0].set_len(1); }
    println!("len={}", v[0].len());
    println!("e={}", v[0][0]);
    println!("{:?}", v);
}
"#;

#[test]
fn run_mode_nested_vec_reads_after_set_len() {
    // `set_len` rewrites the element's declared length in
    // place, leaving the sparse form behind; element reads
    // through the index projection then denote the element's
    // own elements.
    run_ok(
        "nested-vec-set-len",
        NESTED_VEC_SET_LEN_READS_SRC,
        "len=1\ne=1\n[[1], [4]]\n",
    );
}

const NESTED_VEC_PUSH_TRUNC_SRC: &str = r#"
fn main() {
    let mut w = vec![vec![vec![7, 8]], vec![vec![3]]];
    w.truncate(1);
    w[0][0].push(10);
    println!("{:?}", w);
}
"#;

#[test]
fn run_mode_nested_vec_push_after_outer_truncate() {
    // With the outer vector in the sparse form (after
    // `truncate`), a method on a doubly-indexed element forms
    // its receiver through the outer element position: the
    // steps nest into the element rather than composing into
    // the outer vector's positions.
    run_ok(
        "nested-vec-push-trunc",
        NESTED_VEC_PUSH_TRUNC_SRC,
        "[[[7, 8, 10]]]\n",
    );
}

// ----- slice-tag predicate unification (Issue #403) -----

const SIZE_OF_VAL_SLICE_OF_ARRAYS_SRC: &str = r#"
fn main() {
    let a: [[i32; 2]; 3] = [[1, 2], [3, 4], [5, 6]];
    let s: &[[i32; 2]] = &a;
    println!("full {}", std::mem::size_of_val(s));
    println!("part {}", std::mem::size_of_val(&a[..2]));
    let v: Vec<[u16; 4]> = vec![[1, 2, 3, 4], [5, 6, 7, 8]];
    println!("vec {}", std::mem::size_of_val(&v[..]));
    let deep: &[[[i32; 1]; 2]] = &[[[1], [2]], [[3], [4]]];
    println!("deep {}", std::mem::size_of_val(deep));
}
"#;

#[test]
fn run_mode_size_of_val_slice_of_arrays() {
    // The element type of a slice may itself be an array: the
    // semicolon inside `[i32; 2]` sits one bracket level deeper
    // and does not make `[[i32; 2]]` an array tag. `size_of_val`
    // judged slice-ness with a raw `!contains(';')` and refused
    // these as not runtime-supported (expected values verified
    // against a native 1.100.0 build of the same source).
    run_ok(
        "size-of-val-slice-of-arrays",
        SIZE_OF_VAL_SLICE_OF_ARRAYS_SRC,
        "full 24\npart 16\nvec 16\ndeep 16\n",
    );
}

const ALIGN_OF_VAL_SLICE_OF_ARRAYS_SRC: &str = r#"
fn main() {
    let a: [[i32; 2]; 3] = [[1, 2], [3, 4], [5, 6]];
    let s: &[[i32; 2]] = &a;
    println!("full {}", std::mem::align_of_val(s));
    let v: Vec<[u16; 4]> = vec![[1, 2, 3, 4]];
    println!("vec {}", std::mem::align_of_val(&v[..]));
    let w: &[[u8; 3]] = &[[1, 2, 3], [4, 5, 6]];
    println!("bytes {}", std::mem::align_of_val(w));
}
"#;

#[test]
fn run_mode_align_of_val_slice_of_arrays() {
    // The `align_of_val` twin of the size case above: same raw
    // predicate, same misjudgment of a slice of arrays.
    run_ok(
        "align-of-val-slice-of-arrays",
        ALIGN_OF_VAL_SLICE_OF_ARRAYS_SRC,
        "full 4\nvec 2\nbytes 1\n",
    );
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

const SORT_INTERLEAVED_SCRATCH_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..8usize {
        for p in 0..4usize {
            v.push(p * 1000 + i);
        }
    }
    v.sort();
    println!("sorted: {v:?}");
}
"#;

#[test]
fn run_mode_sort_interleaved_scratch_partition() {
    // Driftsort on interleaved runs takes the stable-quicksort
    // path: `stable_partition` copies elements one at a time
    // between the vector buffer and the stack scratch array
    // (`AlignedStorage`, a byte array field reinterpreted as
    // elements). The scratch base pointer is a bare field
    // projection, and `partition_one` offsets it by an element
    // count (`scratch_base.add(num_left)`). That offset used to
    // fall out of the declared-type array judgment (which
    // required a trailing index step) into the value-shape
    // absorption, which denominated the position in the
    // offset's element size over a byte container: element 1
    // landed at byte 1, the address was misaligned, and the
    // guest died in the `copy_nonoverlapping` precondition
    // check (exit 134). The bare field base now judges as
    // element 0 of the field's sequence, so the partition
    // copies land on element boundaries and the sort runs.
    run_ok(
        "sort-interleaved-scratch",
        SORT_INTERLEAVED_SCRATCH_SRC,
        "sorted: [0, 1, 2, 3, 4, 5, 6, 7, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007]\n",
    );
}

const SORT_INTERLEAVED_FORTY_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..10usize {
        for p in 0..4usize {
            v.push(p * 1000 + i);
        }
    }
    v.sort();
    let ok = v.windows(2).all(|w| w[0] <= w[1]);
    println!("len {} sorted {} first {} last {}", v.len(), ok, v[0], v[v.len()-1]);
}
"#;

#[test]
fn run_mode_sort_interleaved_forty_scratch_offset_from() {
    // At forty elements driftsort's merge measures the saved
    // run against the scratch base with `ptr_offset_from`:
    // one pointer denotes an element through the storage
    // field's index step, the other is the bare field base the
    // cast chain left behind. The distance split read the
    // trailing field step as the element index under an empty
    // prefix key, so the two never shared a key and the
    // intrinsic reported unrelated pointers. The split now
    // also offers the bare-base reading (the field path as the
    // key, index 0), so the pair measures.
    run_ok(
        "sort-interleaved-forty",
        SORT_INTERLEAVED_FORTY_SRC,
        "len 40 sorted true first 0 last 3009\n",
    );
}

const SORT_TWO_SORTED_RUNS_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..16usize {
        v.push(i * 2);
    }
    for i in 0..16usize {
        v.push(i * 2 + 1);
    }
    v.sort();
    let ok = v.windows(2).all(|w| w[0] <= w[1]);
    println!("len {} sorted {} first {} last {}", v.len(), ok, v[0], v[v.len()-1]);
}
"#;

#[test]
fn run_mode_sort_two_sorted_runs_merge_scratch_first_use() {
    // Two pre-sorted runs of sixteen: driftsort finds both
    // runs already sorted, so `merge` is the first user of a
    // pristine scratch. Three denomination gaps used to break
    // it in sequence: the stored scratch pointer is the bare
    // field base (its first-element step stripped by the cast
    // chain), so the merge's first comparison read the whole
    // storage array instead of element 0; the run copy into
    // the byte-denominated storage landed element `i` in byte
    // slot `i` instead of slot `i * size_of::<T>()`; and the
    // elements the copy moved can carry the `MaybeUninit`
    // union form, which a scalar comparison now reads through.
    // The dereference of a bare array base denotes element 0
    // for an element or scalar pointee, bulk sequence access
    // prices its stride from the recorded field element size,
    // and the comparison unwraps a one-sided wrapper form.
    run_ok(
        "sort-two-sorted-runs",
        SORT_TWO_SORTED_RUNS_SRC,
        "len 32 sorted true first 0 last 31\n",
    );
}

const SORT_INTERLEAVED_SIXTYFOUR_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..16usize {
        for p in 0..4usize {
            v.push(p * 1000 + i);
        }
    }
    v.sort();
    let ok = v.windows(2).all(|w| w[0] <= w[1]);
    println!("len {} sorted {} first {} last {}", v.len(), ok, v[0], v[v.len()-1]);
}
"#;

#[test]
fn run_mode_sort_interleaved_sixtyfour_scratch_merge() {
    // The sixty-four-element interleaved sort: the scratch
    // merge path at the size where the saved run no longer
    // fits the small-sort special cases (see
    // `run_mode_sort_two_sorted_runs_merge_scratch_first_use`
    // for the denomination gaps in this path).
    run_ok(
        "sort-interleaved-sixtyfour",
        SORT_INTERLEAVED_SIXTYFOUR_SRC,
        "len 64 sorted true first 0 last 3015\n",
    );
}

const SORT_INTERLEAVED_FOUR_HUNDRED_SRC: &str = r#"
fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..100usize {
        for p in 0..4usize {
            v.push(p * 1000 + i);
        }
    }
    v.sort();
    let ok = v.windows(2).all(|w| w[0] <= w[1]);
    println!("len {} sorted {} first {} last {}", v.len(), ok, v[0], v[v.len()-1]);
}
"#;

#[test]
fn run_mode_sort_interleaved_four_hundred_scratch_merge() {
    // The four-hundred-element interleaved sort: driftsort's
    // merge walks the scratch with pointers the cast chain
    // left entered into element 0's `MaybeUninit` payload
    // (`[Field(1), Index(0), Field(0)]` over the byte storage
    // field). Two denomination gaps used to break it in
    // sequence. The pointer offset over that entered position
    // fell out of the declared-type array judgment (which
    // required the index step to be trailing) into the
    // value-shape absorption, which priced the offset in the
    // offset's own element size over the byte-denominated
    // container: `add(1)` on the `*mut usize` landed on byte
    // slot 1 instead of slot 8, and the merge's comparisons
    // read the union wrapper's storage as a unit value. The
    // array judgment now splits a position into its index step
    // and the whole-value entry steps behind it, prices the
    // offset by the sequence's recorded element size, and
    // keeps the entry steps. And the verification's `windows`
    // read the sorted elements in the `MaybeUninit` union form
    // the bulk copies left behind: an index run over the
    // materialised window tuple refused to compose because
    // the head element was an enum, so the second index
    // ranged inside the one-field wrapper and reported a
    // position past its end. A one-field variant-0 wrapper
    // around a scalar payload is the scalar in wrapper form,
    // not an element aggregate, so the run now composes.
    run_ok(
        "sort-interleaved-four-hundred",
        SORT_INTERLEAVED_FOUR_HUNDRED_SRC,
        "len 400 sorted true first 0 last 3099\n",
    );
}

const MERGE_UP_BYTE_STORAGE_SRC: &str = r#"
use std::mem::MaybeUninit;

#[repr(C)]
struct AlignedStorage {
    _align: [usize; 0],
    storage: [MaybeUninit<u8>; 4096],
}

impl AlignedStorage {
    fn new() -> Self {
        Self { _align: [], storage: [const { MaybeUninit::uninit() }; 4096] }
    }
    fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<usize>] {
        let len = 4096 / std::mem::size_of::<usize>();
        unsafe { core::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast(), len) }
    }
}

struct MergeState {
    start: *mut usize,
    end: *mut usize,
    dst: *mut usize,
}

impl MergeState {
    unsafe fn merge_up(&mut self, mut right: *const usize, right_end: *const usize) {
        unsafe {
            let left = &mut self.start;
            let out = &mut self.dst;
            while *left != self.end && right as *const usize != right_end {
                let consume_left = !(*right < **left);
                let src = if consume_left { *left } else { right };
                std::ptr::copy_nonoverlapping(src, *out, 1);
                *left = left.add(consume_left as usize);
                right = right.add(!consume_left as usize);
                *out = out.add(1);
            }
        }
    }
}

fn do_merge(v: &mut [usize], scratch: &mut [MaybeUninit<usize>], mid: usize) {
    unsafe {
        let buf = scratch.as_mut_ptr().cast::<usize>();
        let v_base = v.as_mut_ptr();
        let v_mid = v_base.add(mid);
        let v_end = v_base.add(v.len());
        std::ptr::copy_nonoverlapping(v_base, buf, mid);
        let mut merge_state = MergeState { start: buf, end: buf.add(mid), dst: v_base };
        merge_state.merge_up(v_mid, v_end);
    }
}

fn main() {
    let mut v: Vec<usize> = Vec::new();
    for i in 0..16usize {
        for p in 0..4usize {
            v.push(p * 1000 + i);
        }
    }
    let mut stack_buf = AlignedStorage::new();
    let scratch = stack_buf.as_uninit_slice_mut();
    do_merge(&mut v, scratch, 32);
    let mut wsum: usize = 0;
    for (i, x) in v.iter().enumerate() {
        wsum += (i + 1) * x;
    }
    println!("len {} first {} last {} wsum {}", v.len(), v[0], v[63], wsum);
}
"#;

#[test]
fn run_mode_merge_up_over_byte_storage_scratch() {
    // The hand-written merge over a user-defined byte-storage
    // scratch (the shape std's driftsort scratch takes, in
    // user MIR): the scratch pointers are denominated through
    // a `[MaybeUninit<u8>]` field reinterpreted as
    // `[MaybeUninit<usize>]`, so the base pointer denotes
    // element 0 entered through the union payload and every
    // `add` must be priced in the storage's own byte slots
    // (see `run_mode_sort_interleaved_four_hundred_scratch_merge`).
    // The merge used to stop at the second element with a
    // unit value assigned to a `usize` place.
    run_ok(
        "merge-up-byte-storage",
        MERGE_UP_BYTE_STORAGE_SRC,
        "len 64 first 0 last 3015 wsum 3272656\n",
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


const VEC_RESIZE_OPT_SRC: &str = r#"
fn main() {
    // At opt-level 3 the reserve and the element writes are
    // inlined into one body: the vector value the guest built
    // is the real RawVec aggregate, whose buffer pointer
    // field still holds the dangling pointer after the
    // capacity grows -- the growth must install storage
    // behind it for the writes in extend_with to land (the
    // shape tests/ui/moves/bycopy_untupled.rs exercises in
    // its opt revision).
    let mut v: Vec<usize> = Vec::new();
    v.resize(2, 1);
    assert_eq!(v.len(), 2);
    assert_eq!(v[0], 1);
    assert_eq!(v[1], 1);
    println!("vec-resize-opt-ok");
}
"#;

#[test]
fn run_mode_vec_resize_opt_level_3() {
    let dir = temp_dir("vec-resize-opt");
    let src = fixture(&dir, "vec_resize_opt.rs", VEC_RESIZE_OPT_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec resize at opt-level 3 must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "vec-resize-opt-ok\n");
}
