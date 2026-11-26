//! `size_of_val` / `align_of_val` of dynamically sized types:
//! the layout of an unsized-tail aggregate is its sized prefix
//! plus the tail's own dynamic layout (the fat pointer's
//! metadata: a slice length, or the `dyn` pointee's concrete
//! layout), composed through the driver-recorded DST
//! descriptors. Expected outputs were verified against a
//! native 1.100.0 build of the same sources.

use super::run_ok;

const SLICE_TAIL_PACKED_SRC: &str = r#"
use std::mem;
#[repr(packed, C)]
struct PackedSized { f: u8, d: [u32; 4] }
#[repr(packed, C)]
struct PackedUnsized { f: u8, d: [u32] }
fn main() { unsafe {
    let p = PackedSized { f: 0, d: [1, 2, 3, 4] };
    let r: &PackedUnsized = mem::transmute((&p, 4usize));
    println!("size {}", mem::size_of_val(r));
    println!("align {}", mem::align_of_val(r));
    let d = std::ptr::addr_of!((*r).d);
    println!("d0 {}", d.cast::<u32>().read_unaligned());
    let (base, n): (*const PackedSized, usize) = mem::transmute(r);
    println!("back {} {}", n, std::ptr::addr_of!((*base).f).read_unaligned());
} }
"#;

#[test]
fn run_mode_size_of_val_slice_tail_packed() {
    // The `(ptr, len)` transmute forms the fat pointer over
    // the sized value's own place (and the reverse transmute
    // decomposes it again), so the tail field stays reachable
    // through the fat pointer. The packed prefix contributes
    // no padding: size is 1 + 4*4, not rounded up.
    run_ok(
        "dst-slice-tail-packed",
        SLICE_TAIL_PACKED_SRC,
        "size 17\nalign 1\nd0 1\nback 4 0\n",
    );
}

const DYN_TAIL_SRC: &str = r#"
use std::mem::{self, ManuallyDrop};
#[repr(C, packed(2))]
struct Packed<T: ?Sized> { a: u8, b: ManuallyDrop<T> }
struct Foo<T: ?Sized> { a: i64, b: bool, t: T }
fn main() {
    let p: &Packed<dyn Send> = &Packed { a: 1, b: ManuallyDrop::new(9i16) };
    println!("packed {} {}", mem::size_of_val(p), mem::align_of_val(p));
    let f: &Foo<dyn Send> = &Foo { a: 1, b: true, t: 7i32 };
    println!("foo {} {}", mem::size_of_val(f), mem::align_of_val(f));
}
"#;

#[test]
fn run_mode_size_of_val_dyn_tail() {
    // A `dyn` tail takes its layout from the concrete type
    // the vtable denotes: an `i16` behind a packed(2)
    // `ManuallyDrop` (prefix padded to the packed tail
    // alignment), and an `i32` behind a plain prefix whose
    // end (offset 9) is not rounded up to the struct's
    // alignment before the tail is placed (offset 12).
    run_ok("dst-dyn-tail", DYN_TAIL_SRC, "packed 4 2\nfoo 16 8\n");
}

const NESTED_SLICE_TAIL_SRC: &str = r#"
use std::mem;
struct W<T: ?Sized> { x: u8, t: T }
fn main() {
    let inner: &W<[u8]> = &W { x: 1, t: [10u8, 20, 30] };
    println!("inner {} {}", mem::size_of_val(inner), mem::align_of_val(inner));
    let x: Box<W<W<[u8; 3]>>> = Box::new(W { x: 2, t: W { x: 1, t: [10u8, 20, 30] } });
    let y: Box<W<W<[u8]>>> = x;
    println!("outer {} {}", mem::size_of_val(&*y), mem::align_of_val(&*y));
}
"#;

#[test]
fn run_mode_size_of_val_nested_slice_tail() {
    // The fat metadata belongs to the innermost tail and
    // passes through the outer struct layer: the outer size
    // is its own prefix (1) plus the inner struct's dynamic
    // size (1 + 3).
    run_ok(
        "dst-nested-slice-tail",
        NESTED_SLICE_TAIL_SRC,
        "inner 4 1\nouter 5 1\n",
    );
}

const WRAPPER_TAIL_SRC: &str = r#"
use std::mem;
struct Slice(#[allow(dead_code)] [u32]);
#[repr(C)]
struct Sized { f: u8, d: [u32; 4] }
#[repr(C)]
struct Unsized { f: u8, d: Slice }
fn main() { unsafe {
    let p = Sized { f: 0, d: [1, 2, 3, 4] };
    let r: &Unsized = mem::transmute((&p, 4usize));
    println!("size {}", mem::size_of_val(r));
    println!("align {}", mem::align_of_val(r));
} }
"#;

#[test]
fn run_mode_size_of_val_wrapper_tail() {
    // The tail is itself a one-field DST wrapper: the
    // composition recurses through the wrapper's own
    // descriptor to the slice it wraps.
    run_ok("dst-wrapper-tail", WRAPPER_TAIL_SRC, "size 20\nalign 4\n");
}

const FN_ITEM_SRC: &str = r#"
use std::mem::size_of_val;
fn local() {}
fn main() {
    let a = std::mem::size_of::<u8>;
    println!("{} {} {}", size_of_val(&local), size_of_val(&a), size_of_val(&main));
}
"#;

#[test]
fn run_mode_size_of_val_fn_item() {
    // Function item types are zero-sized, including an
    // intrinsic-backed library item (`std::mem::size_of`)
    // that has no collected body in the program.
    run_ok("dst-fn-item", FN_ITEM_SRC, "0 0 0\n");
}
