// SmallVec tests (auxiliary crate forms).

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
