// Tests that do not belong to a single collection family.


const TRANSMUTE_DEEP_ZST_CHAIN_SRC: &str = r#"
#[repr(C)]
struct Inner2((), *mut u32);
#[repr(C)]
struct Outer2(Inner2, ());
fn main() {
    let mut x: u32 = 42;
    let p = &mut x as *mut u32;
    let outer: Outer2 = unsafe { std::mem::transmute(p) };
    let q = outer.0.1;
    unsafe { *q += 1; }
    println!("{}", x);
}
"#;

#[test]
fn run_mode_transmute_deep_zst_chain_is_explicit_unsupported() {
    // The chain enters a deeper layer at a non-zero field;
    // rebuilding it would need an aggregate nested inside
    // an operand, which the IR does not express. The
    // transmute stays an explicit unsupported instead of
    // building a value whose fields were never constructed.
    let dir = temp_dir("transmute-deep-zst-chain");
    let src = fixture(&dir, "case.rs", TRANSMUTE_DEEP_ZST_CHAIN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "deep non-zero chain transmute must be a named unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("Cast Transmute"),
        "stderr must name the transmute; stderr={stderr}"
    );
}

const NONNULL_CAST_DENOMINATION_SRC: &str = r#"
use std::ptr::NonNull;
struct Inner { ctrl: NonNull<u8>, mask: usize }
struct Bucket<T> { ptr: NonNull<T> }
impl Inner {
    fn data_end<T>(&self) -> NonNull<T> { self.ctrl.cast() }
    unsafe fn bucket<T>(&self, index: usize) -> Bucket<T> {
        unsafe { Bucket::from_base_index(self.data_end(), index) }
    }
}
impl<T> Bucket<T> {
    unsafe fn from_base_index(base: NonNull<T>, index: usize) -> Self {
        let ptr = unsafe { base.as_ptr().sub(index) };
        Bucket { ptr: unsafe { NonNull::new_unchecked(ptr) } }
    }
    unsafe fn write(&self, value: T) {
        unsafe { self.ptr.as_ptr().sub(1).write(value) }
    }
}
fn main() {
    unsafe {
        let layout = std::alloc::Layout::from_size_align(64, 8).unwrap();
        let base = std::alloc::alloc(layout);
        let inner = Inner { ctrl: NonNull::new_unchecked(base.add(32)), mask: 3 };
        let b: Bucket<(i32, i32)> = inner.bucket(0);
        b.write((7i32, 8i32));
        let back = base.add(24) as *const (i32, i32);
        let v = back.read();
        assert_eq!(v, (7, 8));
        std::alloc::dealloc(base, layout);
    }
    println!("denomination ok");
}
"#;

#[test]
fn run_mode_nonnull_cast_denominates_pointer_units() {
    let dir = temp_dir("nonnull-cast-units");
    let src = fixture(&dir, "nonnull_cast_units.rs", NONNULL_CAST_DENOMINATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "denomination ok\n");
}

const SWAP_STRUCT_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct TableLike { mask: usize, ctrl: u64, growth: usize, items: usize }
fn main() {
    let mut a = TableLike { mask: 3, ctrl: 11, growth: 5, items: 2 };
    let mut b = TableLike { mask: 7, ctrl: 22, growth: 1, items: 6 };
    std::mem::swap(&mut a, &mut b);
    assert_eq!(a, TableLike { mask: 7, ctrl: 22, growth: 1, items: 6 });
    assert_eq!(b, TableLike { mask: 3, ctrl: 11, growth: 5, items: 2 });
    println!("swap ok");
}
"#;

#[test]
fn run_mode_swap_nonoverlapping_whole_struct() {
    let dir = temp_dir("swap-struct");
    let src = fixture(&dir, "swap_struct.rs", SWAP_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "swap ok\n");
}

const TYPED_ARENA_EMPTY_SRC: &str = r#"
struct Chunk<T> { storage: std::ptr::NonNull<[std::mem::MaybeUninit<T>]>, entries: usize }
impl<T> Drop for Chunk<T> {
    fn drop(&mut self) { unsafe { drop(Box::from_raw(self.storage.as_mut())) } }
}
struct Arena<T> { chunks: std::cell::RefCell<Vec<Chunk<T>>>, _own: std::marker::PhantomData<T> }
impl<T> Default for Arena<T> {
    fn default() -> Self { Arena { chunks: Default::default(), _own: std::marker::PhantomData } }
}
impl<T> Drop for Arena<T> {
    fn drop(&mut self) {
        let mut b = self.chunks.borrow_mut();
        if let Some(mut last) = b.pop() { let _ = &mut last; }
    }
}
fn main() {
    let a = Arena::<u32>::default();
    drop(a);
    println!("arena-ok");
}
"#;

#[test]
fn run_mode_typed_arena_empty_drop_mir_opt3() {
    // The `rustc_arena::TypedArena` shape: a `RefCell<Vec<Chunk>>`
    // whose element's destructor rebuilds a `Box` from the chunk's
    // `NonNull` slice storage. At mir-opt-level 3 the construction
    // of the arena is inlined, so the chunk vector reaches its
    // drop in the real aggregate form ([RawVec, len]) with a
    // dangling address-only buffer pointer and length 0. The drop
    // glue must run the element plan over the vector's elements
    // (none), never over the aggregate's fields: running it over
    // the buffer descriptor called the chunk destructor at the
    // RawVec position, which read the dangling pointer as the
    // chunk's storage and stopped at its dereference
    // (tests/ui-fulldeps/dropck_tarena_sound_drop.rs).
    let dir = temp_dir("typed-arena-empty-drop");
    let src = fixture(&dir, "typedarena.rs", TYPED_ARENA_EMPTY_SRC);
    let out = run_driver_extra(
        &src,
        &["--edition", "2021", "--rustc-arg", "-Zmir-opt-level=3"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an empty real-form chunk vector must drop cleanly; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "arena-ok\n");
}