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
const TUPLE_TAIL_SRC: &str = r#"
use std::mem;
#[repr(C)]
struct WT<T: ?Sized> { a: u16, t: T }
fn main() { unsafe {
    let p = WT { a: 7u16, t: (9u16, [1u8, 2, 3]) };
    let r: &WT<(u16, [u8])> = mem::transmute((&p, 3usize));
    println!("size {}", mem::size_of_val(r));
    println!("align {}", mem::align_of_val(r));
    println!("a {} t0 {}", r.a, r.t.0);
    println!("t1 {} {} {}", r.t.1[0], r.t.1[1], r.t.1[2]);
    let (base, n): (*const WT<(u16, [u8; 3])>, usize) = mem::transmute(r);
    println!("back {} {}", n, (*base).t.1[2]);
} }
"#;

#[test]
fn run_mode_size_of_val_tuple_tail() {
    // The unsized tail is a tuple whose last element is the
    // slice: the `(ptr, len)` transmute forms the fat
    // reference (a tuple tail has no unsizing coercion, so
    // the transmute is the only formation route), and the
    // composed size is padded to the composed alignment
    // (2 + (2 + 3) bytes of fields, size 8).
    run_ok(
        "dst-tuple-tail",
        TUPLE_TAIL_SRC,
        "size 8\nalign 2\na 7 t0 9\nt1 1 2 3\nback 3 3\n",
    );
}

const BARE_TUPLE_TAIL_SRC: &str = r#"
use std::mem;
fn main() { unsafe {
    let p = (5u8, [10u32, 20, 30]);
    let r: &(u8, [u32]) = mem::transmute((&p, 3usize));
    println!("size {}", mem::size_of_val(r));
    println!("align {}", mem::align_of_val(r));
    println!("v {} {} {} {}", r.0, r.1[0], r.1[1], r.1[2]);
    let (base, n): (*const (u8, [u32; 3]), usize) = mem::transmute(r);
    println!("back {} {}", n, (*base).1[0]);
} }
"#;

#[test]
fn run_mode_size_of_val_bare_tuple_tail() {
    // The pointee itself is the tuple: the same `(ptr, len)`
    // formation and composition, with the tuple's sized
    // prefix element placed before the slice tail.
    run_ok(
        "dst-bare-tuple-tail",
        BARE_TUPLE_TAIL_SRC,
        "size 16\nalign 4\nv 5 10 20 30\nback 3 10\n",
    );
}
const WRAPPER_SCALAR_READ_SRC: &str = r#"
use std::mem;
#[repr(C)]
struct Slice([u32]);
#[repr(C)]
struct Outer { f: u8, d: Slice }
#[repr(C)]
struct Sized { f: u8, d: [u32; 4] }
fn main() { unsafe {
    let s = Sized { f: 0, d: [1, 2, 3, 4] };
    let r: &Outer = mem::transmute((&s, 4usize));
    let d = std::ptr::addr_of!((*r).d);
    println!("e0 {}", d.cast::<u32>().read_unaligned());
    println!("e1 {}", d.cast::<u32>().add(1).read_unaligned());
    println!("e3 {}", d.cast::<u32>().add(3).read_unaligned());
    let sl = d as *const [u32];
    println!("idx {}", (*sl)[2]);
    let d0 = std::ptr::addr_of!((*r).d.0);
    println!("proj {}", d0.cast::<u32>().read_unaligned());
} }
"#;

#[test]
fn run_mode_dst_wrapper_scalar_read() {
    // A scalar read through a pointer cast from a one-field
    // DST wrapper (`*const Slice` to `*const u32`): the cast
    // denotes the wrapper's slice field's first element, so
    // the read descends to the scalar at offset 0 instead of
    // returning the tail field's whole array value. The same
    // holds for offset reads from that element pointer, for
    // a cast back to the slice type, and for the
    // wrapper-first projection form.
    run_ok(
        "dst-wrapper-scalar-read",
        WRAPPER_SCALAR_READ_SRC,
        "e0 1\ne1 2\ne3 4\nidx 3\nproj 1\n",
    );
}

const FIELD_OFFSET_SRC: &str = r#"
use std::mem;
#[repr(packed, C)]
struct PackedSized { f: u8, d: [u32; 4] }
#[repr(packed, C)]
struct PackedUnsized { f: u8, d: [u32] }
impl PackedSized {
    fn unsize(&self) -> &PackedUnsized {
        let len = 4usize;
        unsafe { mem::transmute((self, len)) }
    }
}
fn main() { unsafe {
    let p = PackedSized { f: 0, d: [1, 2, 3, 4] };
    let p = p.unsize() as *const PackedUnsized;
    assert_eq!(mem::size_of_val_raw(p), 1 + 4*4);
    let d = std::ptr::addr_of!((*p).d);
    assert_eq!(d.cast::<u32>().read_unaligned(), 1);
    println!("ok");
} }
"#;

#[test]
fn run_mode_size_of_val_raw_packed_field_offset() {
    // The record form of the packed field-offset case: the
    // fat pointer is formed inside a method from `(self, len)`
    // and consumed as a raw pointer by `size_of_val_raw`,
    // whose size must not count padding in front of `d`.
    run_ok("dst-field-offset", FIELD_OFFSET_SRC, "ok\n");
}

const FIELD_OFFSET_ICE_SRC: &str = r#"
use std::mem;
#[repr(packed(4))]
struct Slice(#[allow(dead_code)] [u32]);
#[repr(packed(2), C)]
struct PackedSized { f: u8, d: [u32; 4] }
#[repr(packed(2), C)]
struct PackedUnsized { f: u8, d: Slice }
impl PackedSized {
    fn unsize(&self) -> &PackedUnsized {
        let len = 4usize;
        unsafe { mem::transmute((self, len)) }
    }
}
fn main() { unsafe {
    let p = PackedSized { f: 0, d: [1, 2, 3, 4] };
    let p = p.unsize() as *const PackedUnsized;
    assert_eq!(mem::size_of_val_raw(p), 1 + 1 + 4*4);
    let d = std::ptr::addr_of!((*p).d);
    assert_eq!(d.cast::<u32>().read_unaligned(), 1);
    println!("ok");
} }
"#;

#[test]
fn run_mode_size_of_val_raw_packed_field_offset_ice() {
    // The -ice record form: `packed(2)` puts exactly one byte
    // of padding in front of the wrapper tail, and the tail
    // is a one-field slice wrapper, so the raw size is
    // 1 + 1 + 4*4 and the field read goes through the wrapper.
    run_ok("dst-field-offset-ice", FIELD_OFFSET_ICE_SRC, "ok\n");
}
