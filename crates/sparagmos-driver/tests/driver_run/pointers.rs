use super::{Command, fixture, run_driver, run_driver_extra, run_ok, run_ok_edition, temp_dir};


// ----- ADT layout table (Lane CG) -----

const LAYOUT_SRC: &str = r#"
#[repr(C)]
struct Padded { a: u8, b: u32, c: u16 }
#[repr(packed)]
struct Packed { a: u8, b: u32 }
enum E { A, B(u16), C(u64) }
struct Outer { x: u8, inner: Padded }
struct Pair<A, B> { a: A, b: B }
fn main() {
    println!("padded {} {}", std::mem::size_of::<Padded>(), std::mem::align_of::<Padded>());
    println!("packed {} {}", std::mem::size_of::<Packed>(), std::mem::align_of::<Packed>());
    println!("enum {} {}", std::mem::size_of::<E>(), std::mem::align_of::<E>());
    println!("outer {} {}", std::mem::size_of::<Outer>(), std::mem::align_of::<Outer>());
    println!("pair {} {}", std::mem::size_of::<Pair<u8, u64>>(), std::mem::align_of::<Pair<u8, u64>>());
    let v = Padded { a: 1, b: 2, c: 3 };
    println!("val {}", std::mem::size_of_val(&v));
}
"#;

#[test]
fn run_mode_size_of_align_of_user_adts() {
    // The values are the compiler's own layouts (verified against a
    // native 1.100.0 build of the same source): the driver records
    // them per monomorphic ADT instance and the mem builtins answer
    // from that table.
    let dir = temp_dir("adt-layout");
    let src = fixture(&dir, "layout.rs", LAYOUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_of/align_of of user ADTs run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "padded 12 4\npacked 5 1\nenum 16 8\nouter 16 4\npair 16 8\nval 12\n",
        "layouts match the native ones"
    );
}

const PTR_COPY_AND_SELECT_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = (0..16).collect();
    let p = v.as_mut_ptr();
    unsafe {
        std::ptr::copy(p, p.add(1), 8);
        let src: Vec<i32> = (100..108).collect();
        std::ptr::copy_nonoverlapping(src.as_ptr(), p.add(8), 8);
    }
    println!("v: {:?}", v);
    let x = std::hint::select_unpredictable(true, 10, 20);
    let y = std::hint::select_unpredictable(false, 10, 20);
    let s = std::hint::select_unpredictable(true, "left", "right");
    println!("select: {x} {y} {s}");
}
"#;

#[test]
fn run_mode_ptr_copy_and_select_unpredictable() {
    // The public raw-pointer copies and
    // `hint::select_unpredictable` (which conversion now also
    // serves natively, as one of the ub-check helper family):
    // the overlapping `ptr::copy` shift, a disjoint
    // `copy_nonoverlapping`, and the select over both a scalar
    // and a fat `&str` value must produce exactly the native
    // results.
    run_ok(
        "ptr-copy-and-select",
        PTR_COPY_AND_SELECT_SRC,
        "v: [0, 0, 1, 2, 3, 4, 5, 6, 100, 101, 102, 103, 104, 105, 106, 107]\nselect: 10 20 left\n",
    );
}

const PTR_EQ_SRC: &str = r#"
fn f() {}
fn g() {}
fn main() {
    let x = 5;
    let p = &x as *const i32;
    let q = &x as *const i32;
    assert!(p == q);
    assert!(!p.is_null());
    let fp: fn() = f;
    let fq: fn() = f;
    let fg: fn() = g;
    assert!(fp == fq);
    assert!(fp != fg);
    let a = &x;
    let b = &x;
    assert!(std::ptr::eq(a, b));
    println!("ptr eq ok");
}
"#;

#[test]
fn run_mode_pointer_and_fn_pointer_equality() {
    let dir = temp_dir("ptr-eq");
    let src = fixture(&dir, "ptr_eq.rs", PTR_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pointer equality must run to completion; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ptr eq ok\n");
}

// ----- transparent-wrapper pointers (Lane DP) --------------------------------

const UNSAFECELL_GET_SRC: &str = r#"
use std::cell::UnsafeCell;
fn main() {
    let cell = UnsafeCell::new(5i32);
    unsafe {
        let p = cell.get();
        *p = 7;
        println!("{}", *p);
    }
}
"#;

const MAYBEUNINIT_PTR_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let mut slot = MaybeUninit::<i32>::uninit();
    unsafe {
        slot.as_mut_ptr().write(11);
        println!("{}", *slot.as_ptr());
    }
}
"#;

const CELL_AS_PTR_SRC: &str = r#"
use std::cell::Cell;
fn main() {
    let c = Cell::new(4i32);
    unsafe { *c.as_ptr() = 6; }
    println!("{}", c.get());
}
"#;

#[test]
fn run_mode_unsafecell_get_reads_and_writes_the_wrapped_value() {
    // `UnsafeCell::get` returns a pointer to the wrapped value
    // itself (the wrapper is transparent). Writing and reading
    // through it must reach the `i32`, not the `UnsafeCell`
    // aggregate: this used to fail with a type mismatch naming
    // `Adt("std::cell::UnsafeCell")` on both directions.
    let dir = temp_dir("unsafecell-get");
    let src = fixture(&dir, "unsafecell_get.rs", UNSAFECELL_GET_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "UnsafeCell::get program must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

#[test]
fn run_mode_maybeuninit_pointer_reads_and_writes_the_wrapped_value() {
    let dir = temp_dir("maybeuninit-ptr");
    let src = fixture(&dir, "maybeuninit_ptr.rs", MAYBEUNINIT_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "MaybeUninit pointer program must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "11\n");
}

#[test]
fn run_mode_cell_as_ptr_writes_the_cell_value() {
    // `Cell::as_ptr` is `UnsafeCell::get` on the cell's inner
    // field: the builtin `Cell` value shape and the collected
    // `get` body must agree on the natural
    // `Cell { UnsafeCell { value } }` nesting.
    let dir = temp_dir("cell-as-ptr");
    let src = fixture(&dir, "cell_as_ptr.rs", CELL_AS_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell::as_ptr program must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "6\n");
}

const TRANSMUTE_FAT_SLICE_SRC: &str = r#"
struct W(*const [u8]);
fn main() {
    let s: &[u8] = &[10, 20, 30, 40];
    let w: [usize; 2] = unsafe { std::mem::transmute(s) };
    println!("{}", w[1]);
    let back: &[u8] = unsafe { std::mem::transmute(w) };
    println!("{} {}", back.len(), back[2]);
    let lit: &str = "world!";
    let t: (usize, usize) = unsafe { std::mem::transmute(lit) };
    println!("{}", t.1);
    let sback: &str = unsafe { std::mem::transmute(t) };
    println!("{}", sback);
    let s3: &[u8] = &[1, 2, 3];
    let wrap = W(s3 as *const [u8]);
    let words: [usize; 2] = unsafe { std::mem::transmute(wrap) };
    println!("{}", words[1]);
    let w2: W = unsafe { std::mem::transmute(words) };
    unsafe { println!("{} {}", (&*w2.0).len(), (*w2.0)[1]) };
}
"#;

const TRANSMUTE_FN_PTR_NICHE_SRC: &str = r#"
unsafe extern "C" fn hello() { println!("hello"); }
fn main() {
    let none = unsafe { std::mem::transmute::<usize, Option<unsafe extern "C" fn()>>(0) };
    println!("none={}", none.is_none());
    let some = unsafe { std::mem::transmute::<usize, Option<unsafe extern "C" fn()>>(1) };
    println!("some={}", some.is_some());
    let null = unsafe { std::mem::transmute::<*const usize, Option<unsafe extern "C" fn()>>(std::ptr::null()) };
    println!("null={}", null.is_none());
    let p: unsafe extern "C" fn() = hello;
    let bits = unsafe { std::mem::transmute::<unsafe extern "C" fn(), usize>(p) };
    let back = unsafe { std::mem::transmute::<usize, Option<unsafe extern "C" fn()>>(bits) };
    match back { Some(f) => unsafe { f() }, None => println!("none") }
    let optbits = unsafe { std::mem::transmute::<Option<unsafe extern "C" fn()>, usize>(Some(p)) };
    println!("round={}", optbits == bits);
}
"#;

#[test]
fn run_mode_transmute_fn_ptr_niche_payload() {
    // A function pointer has the pointer-width scalar leaf,
    // so a niche-encoded enum over one (`Option<fn()>`)
    // transmutes from an integer or a raw pointer: the null
    // pattern is the niche variant, any other pattern is the
    // payload variant. A real function pointer flattens to
    // its synthetic address and the reverse construction
    // recovers the very function value, which still calls.
    run_ok(
        "transmute-fn-ptr-niche",
        TRANSMUTE_FN_PTR_NICHE_SRC,
        "none=true\nsome=true\nnull=true\nhello\nround=true\n",
    );
}

const TRANSMUTE_NICHE_MULTI_FIELD_SRC: &str = r#"
#![feature(core_intrinsics)]
#[derive(Copy, Clone, Debug, PartialEq)]
enum Safety { Unsafe, Safe }
#[derive(Copy, Clone, Debug, PartialEq)]
enum Mut { Immutable, Mutable }
#[derive(Copy, Clone, Debug, PartialEq)]
enum Kind {
    Mod,
    Const,
    Static { safety: Safety, mutability: Mut, nested: bool },
    Fn,
    Type,
}
#[derive(Copy, Clone, Debug, PartialEq)]
enum Tail {
    A,
    B,
    Payload { x: u8, flag: bool },
}
#[derive(Copy, Clone, Debug, PartialEq)]
enum Head {
    Payload { x: u8, flag: bool },
    A,
    B,
}
fn inp() -> u32 { let mut s = 0u32; let mut i = 0; while i < 41 { s += 1; i += 1; } s }
fn erase<T: Copy, const N: usize>(v: T) -> std::mem::MaybeUninit<[u8; N]> {
    unsafe { std::intrinsics::transmute_unchecked(v) }
}
fn restore<T: Copy, const N: usize>(data: std::mem::MaybeUninit<[u8; N]>) -> T {
    unsafe { std::intrinsics::transmute_unchecked(data) }
}
fn rt<T: Copy + PartialEq + std::fmt::Debug, const N: usize>(v: T) {
    let back: T = restore(erase::<T, N>(v));
    println!("{v:?} == {back:?} : {}", v == back);
}
fn main() {
    let n = inp();
    let safety = if n == 41 { Safety::Safe } else { Safety::Unsafe };
    let mutable = if n % 2 == 1 { Mut::Mutable } else { Mut::Immutable };
    println!("sizes {} {} {}", std::mem::size_of::<Kind>(), std::mem::size_of::<Tail>(), std::mem::size_of::<Head>());
    rt::<Kind, 3>(Kind::Mod);
    rt::<Kind, 3>(Kind::Const);
    rt::<Kind, 3>(Kind::Static { safety, mutability: mutable, nested: n == 41 });
    rt::<Kind, 3>(Kind::Static { safety: Safety::Unsafe, mutability: Mut::Immutable, nested: false });
    rt::<Kind, 3>(Kind::Fn);
    rt::<Kind, 3>(Kind::Type);
    rt::<Tail, 2>(Tail::A);
    rt::<Tail, 2>(Tail::B);
    rt::<Tail, 2>(Tail::Payload { x: (n + 159) as u8, flag: true });
    rt::<Tail, 2>(Tail::Payload { x: 3, flag: false });
    rt::<Head, 2>(Head::Payload { x: (n + 36) as u8, flag: false });
    rt::<Head, 2>(Head::A);
    rt::<Head, 2>(Head::B);
}
"#;

#[test]
fn run_mode_transmute_niche_multi_field_payload() {
    // A niche-encoded enum whose payload variant carries
    // several fields — the storage shape of the compiler's
    // `DefKind`, which the query-result erasure
    // (`erase_val` / `restore_val`) transmutes through a
    // `MaybeUninit<[u8; N]>` in both directions. The payload
    // variant sits between fieldless variants, so its own
    // index is a dead slot inside the recorded niche range;
    // the payload-first and payload-last orders exercise the
    // range's other two placements.
    run_ok(
        "transmute-niche-multi-field",
        TRANSMUTE_NICHE_MULTI_FIELD_SRC,
        "sizes 3 2 2\nMod == Mod : true\nConst == Const : true\nStatic { safety: Safe, mutability: Mutable, nested: true } == Static { safety: Safe, mutability: Mutable, nested: true } : true\nStatic { safety: Unsafe, mutability: Immutable, nested: false } == Static { safety: Unsafe, mutability: Immutable, nested: false } : true\nFn == Fn : true\nType == Type : true\nA == A : true\nB == B : true\nPayload { x: 200, flag: true } == Payload { x: 200, flag: true } : true\nPayload { x: 3, flag: false } == Payload { x: 3, flag: false } : true\nPayload { x: 77, flag: false } == Payload { x: 77, flag: false } : true\nA == A : true\nB == B : true\n",
    );
}

const TRANSMUTE_FN_ADDR_AS_DATA_PTR_SRC: &str = r#"
fn hello() { println!("hello called"); }
fn main() {
    let bits = hello as usize;
    let arr: [*const u8; 1] = unsafe { std::mem::transmute(bits) };
    let p = arr[0];
    let q = p.wrapping_add(1);
    println!("diff={}", (q as usize).wrapping_sub(p as usize));
    let opt: Option<fn()> = unsafe { std::mem::transmute(bits) };
    match opt { Some(f) => f(), None => println!("none") }
    let f2: fn() = unsafe { std::mem::transmute(bits) };
    f2();
}
"#;

#[test]
fn run_mode_transmute_fn_addr_into_data_ptr_stays_data_ptr() {
    // The address a function pointer was exposed at is just
    // an integer: transmuted into a data-pointer destination
    // it constructs a data pointer (pointer arithmetic
    // applies), not the callable value. The callable reading
    // is specific to function-pointer destinations, where
    // the same bits still recover the function.
    run_ok(
        "transmute-fn-addr-data-ptr",
        TRANSMUTE_FN_ADDR_AS_DATA_PTR_SRC,
        "diff=1\nhello called\nhello called\n",
    );
}

#[test]
fn run_mode_transmute_fat_pointer_words() {
    // A fat pointer transmutes to its two words and back: the
    // metadata word is the length, and constructing the words
    // returns the very pointer that flattened to them (its
    // pointee still reads). A newtype wrapper over a fat
    // pointer and a tuple target take the same shape path.
    run_ok(
        "transmute-fat-slice",
        TRANSMUTE_FAT_SLICE_SRC,
        "4\n4 30\n6\nworld!\n3\n3 2\n",
    );
}

const TRANSMUTE_FAT_DYN_UNION_SRC: &str = r#"
use std::fmt::Debug;
union U {
    p: *const [u8],
    w: [usize; 2],
}
fn main() {
    let x = 42u32;
    let d: &dyn Debug = &x;
    let w: [usize; 2] = unsafe { std::mem::transmute(d) };
    let back: &dyn Debug = unsafe { std::mem::transmute(w) };
    println!("{:?}", back);
    let s: &[u8] = &[9, 8, 7];
    let u = U { p: s as *const [u8] };
    unsafe { println!("{}", u.w[1]) };
}
"#;

#[test]
fn run_mode_transmute_dyn_pointer_and_union_fat_field() {
    // A `dyn` fat pointer's metadata word is a vtable identity
    // only the fat-value table resolves: the round trip
    // through the words returns the original pointer, whose
    // dispatch still works. A union field holding a fat
    // pointer reads back through the other field's words.
    run_ok(
        "transmute-fat-dyn-union",
        TRANSMUTE_FAT_DYN_UNION_SRC,
        "42\n3\n",
    );
}

const TRANSMUTE_WRAPPER_SRC: &str = r#"
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::num::Wrapping;
struct Padded { x: u32, _m: PhantomData<u8> }
fn inp() -> u32 { let mut s = 0u32; let mut i = 0; while i < 41 { s += 1; i += 1; } s }
fn main() {
    let n = inp();
    let w: Wrapping<u32> = unsafe { std::mem::transmute(n) };
    let back: u32 = unsafe { std::mem::transmute(w) };
    let wf: Wrapping<f32> = unsafe { std::mem::transmute(0x3fc00000u32) };
    let p: Padded = unsafe { std::mem::transmute(n + 1) };
    let md: ManuallyDrop<u32> = unsafe { std::mem::transmute(n) };
    let mback: u32 = unsafe { std::mem::transmute(md) };
    println!("{} {} {} {} {}", w.0, back, wf.0, p.x, mback);
}
"#;

#[test]
fn run_mode_transmute_wrapper_adts() {
    run_ok(
        "transmute-wrappers",
        TRANSMUTE_WRAPPER_SRC,
        "41 41 1.5 42 41\n",
    );
}

const TRANSMUTE_RAW_TO_ZST_FIRST_WRAPPER_SRC: &str = r#"
#[repr(transparent)]
struct Inner(*mut u32);
#[repr(C)]
struct Outer((), Inner);
fn main() {
    let mut x: u32 = 42;
    let p = &mut x as *mut u32;
    let outer: Outer = unsafe { std::mem::transmute(p) };
    let q = outer.1.0;
    unsafe { *q += 1; }
    println!("{}", x);
    let outer2: Outer = unsafe { std::mem::transmute(q) };
    println!("{}", unsafe { *outer2.1.0 });
}
"#;

#[test]
fn run_mode_transmute_raw_pointer_to_wrapper_with_leading_zst() {
    // The pointer chain enters the target at field 1 (a
    // zero-sized field sits before the payload layer), so
    // the rebuilt value must hold the pointer at that index
    // with the zero-sized field's value beside it — not in a
    // one-field aggregate at field 0, where reading `.1.0`
    // finds a field that was never constructed.
    run_ok(
        "transmute-raw-to-zst-first-wrapper",
        TRANSMUTE_RAW_TO_ZST_FIRST_WRAPPER_SRC,
        "43\n43\n",
    );
}

const TRANSMUTE_CONSTANTS_SRC: &str = r#"
fn main() {
    let a: u32 = unsafe { std::mem::transmute(1.5f32) };
    let b: f32 = unsafe { std::mem::transmute(0x3fc00000u32) };
    let c: u64 = unsafe { std::mem::transmute(-2.5f64) };
    let d: u8 = unsafe { std::mem::transmute(true) };
    let e: [u8; 4] = unsafe { std::mem::transmute(0x04030201u32) };
    println!("{:x} {} {:x} {} {:?}", a, b, c, d, e);
}
"#;

#[test]
fn run_mode_transmute_constant_operands() {
    run_ok(
        "transmute-constants",
        TRANSMUTE_CONSTANTS_SRC,
        "3fc00000 1.5 c004000000000000 1 [1, 2, 3, 4]\n",
    );
}

const TRANSMUTE_PACKED_STRUCT_SRC: &str = r#"
use std::mem;
#[repr(packed)]
struct S5 { a: u8, b: u32 }
fn main() {
    unsafe {
        let s5 = S5 { a: 1, b: 0xff_00_00_ff };
        let transd: [u8; 5] = mem::transmute(s5);
        assert_eq!(transd, [1, 0xff, 0, 0, 0xff]);
        let q: S5 = mem::transmute([9u8, 8, 7, 6, 5]);
        let qa = q.a;
        let qb = q.b;
        println!("{} {}", qa, qb);
    }
}
"#;

#[test]
fn run_mode_transmute_packed_struct_bytes() {
    // A multi-field packed struct transmuted to its byte
    // array and back: the fields sit at their recorded
    // layout offsets with no padding in between.
    run_ok(
        "transmute-packed-struct",
        TRANSMUTE_PACKED_STRUCT_SRC,
        "9 84281096\n",
    );
}

const TRANSMUTE_WIDE_SRC: &str = r#"
fn main() {
    let big: u128 = unsafe { std::mem::transmute(-1i128) };
    let neg: i8 = unsafe { std::mem::transmute(0xffu8) };
    let s16: i16 = unsafe { std::mem::transmute(0xffffu16) };
    println!("{:x} {} {}", big, neg, s16);
}
"#;

#[test]
fn run_mode_transmute_wide_and_signed() {
    run_ok(
        "transmute-wide",
        TRANSMUTE_WIDE_SRC,
        "ffffffffffffffffffffffffffffffff -1 -1\n",
    );
}

const NONZERO_TRANSMUTE_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    let x: NonZero<u32> = unsafe { std::mem::transmute(42u32) };
    println!("nz {}", x.get());
    let y: u32 = unsafe { std::mem::transmute(x) };
    println!("back {}", y);
    let z: NonZero<i32> = unsafe { std::mem::transmute(-5i32) };
    println!("neg {}", z.get());
    let w: NonZero<u16> = unsafe { std::mem::transmute(7u16) };
    println!("w {}", w.get());
}
"#;

#[test]
fn run_mode_nonzero_transmute_roundtrip() {
    // `NonZero<T>`'s only field is declared as the projection
    // alias `<T as ZeroablePrimitive>::NonZeroInner`; the driver
    // records the compiler-normalized field type of each
    // monomorphic instance, so the transmutes (and `get`, itself
    // a transmute in core) execute.
    run_ok("nonzero-transmute", NONZERO_TRANSMUTE_SRC, "nz 42\nback 42\nneg -5\nw 7\n");
}

const NONZERO_FIELD_SRC: &str = r#"
use std::num::NonZero;
struct Wrap { tag: u8, val: NonZero<u32> }
fn main() {
    let v: NonZero<u32> = unsafe { std::mem::transmute(9u32) };
    let w = Wrap { tag: 3, val: v };
    println!("field {} {}", w.tag, w.val.get());
    let raw: u32 = unsafe { std::mem::transmute(w.val) };
    println!("raw {}", raw);
}
"#;

#[test]
fn run_mode_nonzero_field_in_struct() {
    run_ok("nonzero-field", NONZERO_FIELD_SRC, "field 3 9\nraw 9\n");
}

const NONZERO_LAYOUT_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    println!("{} {}", std::mem::size_of::<Option<NonZero<u32>>>(), std::mem::size_of::<u32>());
    println!("{}", std::mem::size_of::<Option<NonZero<u64>>>());
    println!("{}", std::mem::size_of::<NonZero<u32>>());
}
"#;

#[test]
fn run_mode_nonzero_layout_sizes() {
    run_ok("nonzero-layout", NONZERO_LAYOUT_SRC, "4 4\n8\n4\n");
}

const NONZERO_NEW_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    let x: NonZero<u32> = unsafe { std::mem::transmute(42u32) };
    println!("nz {}", x.get());
    let y: u32 = unsafe { std::mem::transmute(x) };
    println!("back {}", y);
    println!("new {}", NonZero::new(7).unwrap().get());
}
"#;

#[test]
fn run_mode_nonzero_new_constructor() {
    // `NonZero::new` itself is a transmute between the bare
    // integer and `Option<NonZero<T>>` in core: it converts
    // through the enum's niche encoding, which the driver
    // records from the compiler's layout and the interpreter
    // reinterprets through.
    run_ok("nonzero-new", NONZERO_NEW_SRC, "nz 42\nback 42\nnew 7\n");
}

const NONZERO_OPTION_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    println!("size {} {}", std::mem::size_of::<Option<NonZero<u32>>>(), std::mem::size_of::<u32>());
    let a: Option<NonZero<u32>> = NonZero::new(5);
    let bits: u32 = unsafe { std::mem::transmute(a) };
    println!("bits {}", bits);
    let b: Option<NonZero<u32>> = unsafe { std::mem::transmute(0u32) };
    println!("none {}", b.is_none());
}
"#;

#[test]
fn run_mode_option_nonzero_transmutes() {
    run_ok("option-nonzero", NONZERO_OPTION_SRC, "size 4 4\nbits 5\nnone true\n");
}

const NONZERO_FIELD_NEW_SRC: &str = r#"
use std::num::NonZero;
struct Wrap { tag: u8, val: NonZero<u32> }
fn main() {
    let w = Wrap { tag: 3, val: NonZero::new(9).unwrap() };
    println!("field {} {}", w.tag, w.val.get());
    let raw: u32 = unsafe { std::mem::transmute(w.val) };
    println!("raw {}", raw);
}
"#;

#[test]
fn run_mode_nonzero_field_constructed_by_new() {
    run_ok("nonzero-field-new", NONZERO_FIELD_NEW_SRC, "field 3 9\nraw 9\n");
}

const UNION_DROP_SRC: &str = r#"
use std::mem::ManuallyDrop;
struct S;
static mut CHECK: u8 = 0;
impl Drop for S {
    fn drop(&mut self) {
        unsafe { CHECK += 10; }
    }
}
union U {
    a: ManuallyDrop<S>,
}
impl Drop for U {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}
fn main() {
    unsafe {
        let mut u = U { a: ManuallyDrop::new(S) };
        u = U { a: ManuallyDrop::new(S) };
        assert_eq!(CHECK, 1);
        *u.a = S;
        assert_eq!(CHECK, 11);
    }
    println!("union ok");
}
"#;

#[test]
fn run_mode_union_drop_and_field_assignment() {
    let dir = temp_dir("union-drop");
    let src = fixture(&dir, "union_drop.rs", UNION_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a union with a Drop impl must drop on overwrite, and assigning through its field must drop the old field value; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "union ok\n");
}

const MEM_DROP_ZST_UNION_SRC: &str = r#"
use std::mem::ManuallyDrop;
struct S;
static mut CHECK: u8 = 0;
impl Drop for S {
    fn drop(&mut self) {
        unsafe { CHECK += 10; }
    }
}
union U {
    a: ManuallyDrop<S>,
}
impl Drop for U {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}
fn main() {
    let u = U { a: ManuallyDrop::new(S) };
    drop(u);
    unsafe { assert_eq!(CHECK, 1); }
    println!("mem-drop ok");
}
"#;

#[test]
fn run_mode_mem_drop_of_zst_union_runs_destructor() {
    let dir = temp_dir("mem-drop-zst-union");
    let src = fixture(&dir, "mem_drop_zst_union.rs", MEM_DROP_ZST_UNION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an explicit mem::drop of a zero-sized union value must run its destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "mem-drop ok\n");
}

const UNION_REINTERPRET_SRC: &str = r#"
union W {
    a: u32,
    b: f32,
}
union V {
    a: u32,
    b: [u8; 4],
    c: u16,
}
union T {
    a: (u8, u8),
    b: u16,
}
union U<A, B>
where
    A: Copy,
    B: Copy,
{
    a: A,
    b: B,
}
unsafe fn union_transmute<A, B>(a: A) -> B
where
    A: Copy,
    B: Copy,
{
    U { a }.b
}
fn main() {
    unsafe {
        let w = W { a: 0x3F80_0000 };
        assert_eq!(w.b, 1.0);
        let mut w2 = W { b: 2.5 };
        assert_eq!(w2.a, 0x4020_0000);
        w2.a = 0x3F80_0000;
        assert_eq!(w2.b, 1.0);
        let v = V { a: 0x0403_0201 };
        assert_eq!(v.b, [1, 2, 3, 4]);
        assert_eq!(v.c, 0x0201);
        let v2 = V { b: [9, 8, 7, 6] };
        assert_eq!(v2.a, 0x0607_0809);
        let mut m = V { a: 0 };
        m.c = 0xBEEF;
        assert_eq!(m.c, 0xBEEF);
        assert_eq!(m.b, [0xEF, 0xBE, 0, 0]);
        let t = T { a: (1, 1) };
        assert_eq!(t.b, (1 << 8) + 1);
        let g: u16 = union_transmute((1u8, 1u8));
        assert_eq!(g, (1 << 8) + 1);
        let f: f32 = union_transmute(0x3F80_0000u32);
        assert_eq!(f, 1.0);
        println!("union reinterpret ok");
    }
}
"#;

#[test]
fn run_mode_union_field_reads_reinterpret_bits() {
    // Reading a union through a different field than the one
    // written reinterprets the held field's bits (little-endian:
    // narrower fields read the low bytes), matching native runs
    // of this same program.
    let dir = temp_dir("union-reinterpret");
    let src = fixture(&dir, "union_reinterpret.rs", UNION_REINTERPRET_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "union field reads must reinterpret the held field's bits; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "union reinterpret ok\n"
    );
}

const UNION_WRAPPER_FIELDS_SRC: &str = r#"
use std::mem::ManuallyDrop;
union C {
    a: u32,
    b: char,
}
union B {
    a: u8,
    b: bool,
}
union M {
    a: ManuallyDrop<u32>,
    b: f32,
}
fn main() {
    unsafe {
        let c = C { a: 0x41 };
        assert_eq!(c.b, 'A');
        let b = B { a: 1 };
        assert_eq!(b.b, true);
        let m = M { b: 1.5 };
        let md = m.a;
        let m3 = M { a: md };
        assert_eq!(m3.b, 1.5);
        let m2 = M { a: ManuallyDrop::new(0x3F80_0000) };
        assert_eq!(m2.b, 1.0);
        println!("union wrapper fields ok");
    }
}
"#;

#[test]
fn run_mode_union_field_reads_through_wrapper_fields() {
    // `char` / `bool` fields and one-payload wrapper fields
    // (ManuallyDrop) reinterpret through the same shapes.
    let dir = temp_dir("union-wrapper-fields");
    let src = fixture(&dir, "union_wrapper_fields.rs", UNION_WRAPPER_FIELDS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "union reads through char/bool/ManuallyDrop fields must reinterpret; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "union wrapper fields ok\n"
    );
}

const UNION_MULTI_FIELD_STRUCT_SRC: &str = r#"
#[derive(Copy, Clone)]
struct P(u32, u32);
union X {
    a: P,
    b: u64,
}
fn main() {
    unsafe {
        let x = X { b: 0x0000_0005_0000_002a };
        let p = x.a;
        println!("{} {}", p.0, p.1);
    }
}
"#;

#[test]
fn run_mode_union_field_read_multi_field_struct() {
    // A multi-field struct field read from another field's
    // bits: the struct's fields sit at their recorded layout
    // offsets, so the read reinterprets exactly.
    let dir = temp_dir("union-multi-field-struct");
    let src = fixture(&dir, "union_multi_field_struct.rs", UNION_MULTI_FIELD_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a union read of a multi-field struct must reinterpret; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42 5\n");
}

const UNION_NO_SHAPE_SRC: &str = r#"
#[derive(Copy, Clone)]
struct Big([u64; 3]);
union X {
    a: Big,
    b: u64,
}
fn main() {
    unsafe {
        let x = X { b: 42 };
        let _p = x.a;
        println!("should not reach");
    }
}
"#;

#[test]
fn run_mode_union_field_read_without_scalar_shape_is_explicit() {
    // A field whose type has no scalar shape (a struct over
    // the 128-bit shape limit) cannot be reinterpreted from
    // another field's bits: the read is a named unsupported
    // error, not a fabricated value.
    let dir = temp_dir("union-no-shape");
    let src = fixture(&dir, "union_no_shape.rs", UNION_NO_SHAPE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a union read with no scalar shape must be a named unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("union field read"),
        "stderr must name the union field read; stderr={stderr}"
    );
}

const UNION_NESTED_WRITE_SRC: &str = r#"
union U {
    a: (u16, u16),
    b: u32,
}
fn main() {
    unsafe {
        let mut u = U { b: 0xDEAD_DEAD };
        u.a.0 = 0xBEEF;
        println!("{}", u.b);
    }
}
"#;

#[test]
fn run_mode_union_write_inside_unheld_field_composes() {
    // Writing inside a field the value does not currently hold
    // composes with the held field's bytes: the written part
    // changes, the held field's remaining bytes survive, and
    // reading the held field back shows both (0xDEAD_BEEF).
    let dir = temp_dir("union-nested-write");
    let src = fixture(&dir, "union_nested_write.rs", UNION_NESTED_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a write inside an unheld union field must compose; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "3735928559\n");
}

const PIN_REF_DEREF_SRC: &str = r#"
fn main() {
    let mut x = 5;
    let mut p = std::pin::Pin::new(&mut x);
    *p = 8;
    println!("{}", *p);
    println!("{}", x);
}
"#;

#[test]
fn run_mode_pin_ref_deref_reads_and_writes_through() {
    // Dereferencing a `Pin<&mut T>` goes through the wrapped
    // reference: reads and writes land on the pointee.
    let dir = temp_dir("pin-ref-deref");
    let src = fixture(&dir, "pin_ref_deref.rs", PIN_REF_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Pin<&mut T> deref must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "8\n8\n",
        "the write through the Pin is visible on the pointee"
    );
}

const ENV_ARGS_LEN_SRC: &str = r#"
fn main() {
    println!("len {}", std::env::args().len());
    println!("os-len {}", std::env::args_os().len());
    let mut it = std::env::args();
    println!("first {}", it.next().unwrap());
    println!("len-after {}", it.len());
    let mut os = std::env::args_os();
    println!("os-first {}", os.next().is_some());
    println!("os-len-after {}", os.len());
}
"#;

#[test]
fn run_mode_env_args_len_uses_the_real_wrapper_layout() {
    // `ExactSizeIterator::len` for `Args` / `ArgsOs` is not a
    // builtin: the collected std one-liners navigate the real
    // wrapper layout (`ArgsOs { inner: sys::args::common::Args {
    // inner: vec::IntoIter } }`) field by field and land in the
    // `size_hint` builtin on the innermost iterator. The modelled
    // value must carry all three wrapper layers or the second
    // field step reads the element snapshot as the state.
    let dir = temp_dir("env-args-len");
    let src = fixture(&dir, "args_len.rs", ENV_ARGS_LEN_SRC);
    let out = run_driver_extra(&src, &["--guest-arg", "one", "--guest-arg", "two"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 6, "stdout={stdout}");
    assert_eq!(lines[0], "len 3");
    assert_eq!(lines[1], "os-len 3");
    assert!(lines[2].starts_with("first "), "stdout={stdout}");
    assert!(lines[2].contains("args_len.rs"), "stdout={stdout}");
    assert_eq!(lines[3], "len-after 2");
    assert_eq!(lines[4], "os-first true");
    assert_eq!(lines[5], "os-len-after 2");
}

const TRANSMUTE_NONNULL_SRC: &str = r#"use std::ptr::NonNull;
fn main() {
    let p = NonNull::<u8>::dangling();
    let n: usize = unsafe { std::mem::transmute(p) };
    assert!(n != 0);
    let q: NonNull<u8> = unsafe { std::mem::transmute(n) };
    assert_eq!(q, p);
    println!("nonnull roundtrip ok");
}
"#;

#[test]
fn run_mode_transmute_nonnull_integer_roundtrip() {
    // `NonNull::dangling` transmutes a `mem::Alignment` into
    // the pointer wrapper (the discriminant is the address),
    // and user code transmutes the wrapper to `usize` and
    // back: the wrapper <-> integer transmutes carry the
    // payload pointer's synthetic address.
    let dir = temp_dir("transmute-nonnull");
    let src = fixture(&dir, "transmute_nonnull.rs", TRANSMUTE_NONNULL_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "nonnull roundtrip ok\n");
}

const OPTION_NONNULL_NICHE_SRC: &str = r#"use std::ptr::NonNull;
fn main() {
    let x = 7u8;
    let some: Option<NonNull<u8>> = NonNull::new(&x as *const u8 as *mut u8);
    // The niche encoding: the option *is* the pointer word,
    // zero for `None`. (The roundtrip is checked on the bits:
    // comparing an address-only pointer against a place
    // pointer is a separate, older gap.)
    let bits: usize = unsafe { std::mem::transmute(some) };
    assert!(bits != 0);
    let back: Option<NonNull<u8>> = unsafe { std::mem::transmute(bits) };
    assert!(back.is_some());
    let bits2: usize = unsafe { std::mem::transmute(back) };
    assert_eq!(bits2, bits);
    assert_eq!(unsafe { *some.unwrap().as_ptr() }, 7);
    let none: Option<NonNull<u8>> = None;
    let zero: usize = unsafe { std::mem::transmute(none) };
    assert_eq!(zero, 0);
    let back_none: Option<NonNull<u8>> = unsafe { std::mem::transmute(0usize) };
    assert!(back_none.is_none());
    // The same encoding through a raw pointer target type.
    let p: *const u32 = unsafe { std::mem::transmute(some) };
    assert!(!p.is_null());
    println!("option nonnull niche ok");
}
"#;

#[test]
fn run_mode_option_nonnull_niche_transmute() {
    // `Option<NonNull<T>>` uses the pointer's null niche as
    // its encoding, so a transmute between the option and an
    // integer (or a raw pointer) is the identity on the
    // address word, with zero reading back as `None`.
    let dir = temp_dir("option-nonnull-niche");
    let src = fixture(&dir, "option_nonnull_niche.rs", OPTION_NONNULL_NICHE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "option nonnull niche ok\n");
}
// ----- reference deref identity and DST metadata (Lane GB) -----------------

const COROUTINE_RESUME_ARGS_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let mut gen = #[coroutine] |x: i32| {
        let y = yield x + 1;
        y * 2
    };
    let mut pinned = Pin::new(&mut gen);
    match pinned.as_mut().resume(41) {
        CoroutineState::Yielded(v) => println!("yielded {v}"),
        CoroutineState::Complete(v) => println!("complete {v}"),
    }
    match pinned.as_mut().resume(10) {
        CoroutineState::Yielded(v) => println!("yielded {v}"),
        CoroutineState::Complete(v) => println!("complete {v}"),
    }
}
"#;

#[test]
fn run_mode_coroutine_resume_args_through_pin() {
    // `Pin::as_mut` dereferences the `&mut Coroutine` the pin
    // wraps (`DerefMut` for a mutable reference): the result is
    // the inner reference itself, and the coroutine state machine
    // advances across resumes with the resume argument bound.
    let dir = temp_dir("coroutine-resume-args");
    let src = fixture(&dir, "coroutine_resume_args.rs", COROUTINE_RESUME_ARGS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "yielded 42\ncomplete 20\n", "stdout={stdout}");
}

const DEREF_ON_REFERENCE_SRC: &str = r#"
use std::ops::{Deref, DerefMut};

fn bump(mut r: &mut i32) {
    *r.deref_mut() += 1;
}

fn read(r: &i32) -> i32 {
    *r.deref()
}

fn via_generic<P: DerefMut<Target = i32>>(mut p: P) {
    *p += 10;
}

fn main() {
    let mut x = 1;
    bump(&mut x);
    println!("x {x}");
    println!("read {}", read(&x));
    via_generic(&mut x);
    println!("x {x}");
}
"#;

#[test]
fn run_mode_deref_and_deref_mut_on_references() {
    // `<&mut T as DerefMut>::deref_mut` returns the inner
    // reference itself (writing through it lands in the original
    // place), and `<&T as Deref>::deref` likewise; generic code
    // over `P: DerefMut` instantiated with a reference reaches
    // the same impls.
    let dir = temp_dir("deref-on-reference");
    let src = fixture(&dir, "deref_on_reference.rs", DEREF_ON_REFERENCE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "x 2\nread 2\nx 12\n", "stdout={stdout}");
}

const DST_STRUCT_SRC: &str = r#"
struct Fat<T: ?Sized> {
    f1: isize,
    f2: &'static str,
    ptr: T,
}

fn foo(x: &Fat<[isize]>) {
    println!("f1 {} f2 {} len {} second {}", x.f1, x.f2, x.ptr.len(), x.ptr[1]);
}

fn main() {
    let f1 = Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    foo(&f1);
    let f3: &Fat<[isize]> = &f1;
    foo(f3);
    let b: Box<[isize]> = Box::new([4, 5, 6, 7]);
    println!("box len {}", b.len());
}
"#;

#[test]
fn run_mode_dst_struct_trailing_slice_metadata() {
    // Unsizing `&Fat<[isize; 3]>` to `&Fat<[isize]>` computes the
    // metadata of the struct value from its trailing field: the
    // slice length, not the unit metadata of a sized value.
    let dir = temp_dir("dst-struct");
    let src = fixture(&dir, "dst_struct.rs", DST_STRUCT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "f1 5 f2 some str len 3 second 2\nf1 5 f2 some str len 3 second 2\nbox len 4\n",
        "stdout={stdout}"
    );
}

const FAT_PTR_TRANSMUTE_SRC: &str = r#"
fn main() {
    let s = "hello";
    let nn: std::ptr::NonNull<str> = unsafe { std::mem::transmute(s as *const str) };
    let back: &str = unsafe { std::mem::transmute(nn) };
    println!("back {back}");
    let arr = [1u8, 2, 3];
    let nn2: std::ptr::NonNull<[u8]> = unsafe { std::mem::transmute(&arr[..] as *const [u8]) };
    let sl: &[u8] = unsafe { std::mem::transmute(nn2) };
    println!("len {} sum {}", sl.len(), sl[0] + sl[1] + sl[2]);
}
"#;

#[test]
fn run_mode_fat_pointer_wrapper_transmutes_round_trip() {
    let dir = temp_dir("fat-ptr-transmute");
    let src = fixture(&dir, "fatptr.rs", FAT_PTR_TRANSMUTE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "back hello\nlen 3 sum 6\n", "stdout={stdout}");
}

const ENUM_WRAP_TRANSMUTE_SRC: &str = r#"
#[repr(isize)]
#[derive(Clone, Copy)]
enum Align { A1 = 1, A2 = 2, A4 = 4, A8 = 8 }
#[repr(transparent)]
struct Wrap(Align);
fn main() {
    let n: usize = unsafe { std::mem::transmute(Wrap(Align::A8)) };
    println!("n {n}");
    let m: usize = unsafe { std::mem::transmute(Wrap(Align::A2)) };
    println!("m {m}");
}
"#;

#[test]
fn run_mode_transparent_wrapper_over_fieldless_enum_transmutes_to_int() {
    // The `mem::Alignment` shape: a one-field transparent struct
    // over a fieldless enum transmutes to the integer as the
    // field's discriminant. (RawVec's allocation path performs
    // exactly this transmute on the `Vec` collect path.)
    let dir = temp_dir("enum-wrap-transmute");
    let src = fixture(&dir, "alignwrap.rs", ENUM_WRAP_TRANSMUTE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "n 8\nm 2\n", "stdout={stdout}");
}

// ----- Optimized-build transmutes in inlined std shapes (Lane JJ) -----
// Expected outputs are the native results of the same sources.

const FN_PTR_ERASED_SIGNATURE_SRC: &str = r#"
fn f(_x: &u64, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Ok(())
}
fn main() {
    let p: fn(&u64, &mut std::fmt::Formatter<'_>) -> std::fmt::Result = f;
    let _q: unsafe fn(
        std::ptr::NonNull<()>,
        &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result = unsafe { std::mem::transmute(p) };
}
"#;

/// The transmute an inlined `fmt::Argument` construction performs
/// on the display function pointer: a function pointer viewed
/// through an erased signature (safe fn -> unsafe fn over an
/// erased argument). A function pointer's value is the function
/// identity, so the cast carries the value over.
#[test]
fn run_mode_o2_transmute_fn_ptr_erased_signature() {
    let dir = temp_dir("fnptr-erased");
    let src = fixture(&dir, "case.rs", FN_PTR_ERASED_SIGNATURE_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "fn-pointer erased-signature transmute must run; stderr={stderr}"
    );
}

const ARRAY_BY_VALUE_SUM_SRC: &str = r#"
fn main() {
    let a = [7u8; 64];
    let mut s: u64 = 0;
    for x in a {
        s += x as u64;
    }
    assert_eq!(s, 448);
}
"#;

/// At opt-level 2 the by-value array iterator is inlined, and
/// its construction transmutes the backing array
/// `[u8; 64]` -> `[MaybeUninit<u8>; 64]`. Each element takes
/// the wrapper's value representation, and the iterated sum
/// must come out exact.
#[test]
fn run_mode_o2_array_into_iter_sum() {
    let dir = temp_dir("array-into-iter");
    let src = fixture(&dir, "case.rs", ARRAY_BY_VALUE_SUM_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "by-value array iteration under -O must sum exactly; stderr={stderr}"
    );
}

const MAYBE_UNINIT_ARRAY_UNWRAP_SRC: &str = r#"
use std::mem::{transmute, MaybeUninit};
fn main() {
    let a: [MaybeUninit<u8>; 4] = [
        MaybeUninit::new(3),
        MaybeUninit::new(4),
        MaybeUninit::new(5),
        MaybeUninit::new(6),
    ];
    let b: [u8; 4] = unsafe { transmute(a) };
    let mut s: u32 = 0;
    let mut j = 0usize;
    while j < 4 {
        s += b[j] as u32;
        j += 1;
    }
    assert_eq!(s, 18);
}
"#;

/// The reverse element-wrapper crossing: an array of
/// `MaybeUninit` values transmuted back to the plain element
/// array unwraps each element to the payload it holds.
#[test]
fn run_mode_o2_transmute_maybe_uninit_array_unwrap() {
    let dir = temp_dir("mu-array-unwrap");
    let src = fixture(&dir, "case.rs", MAYBE_UNINIT_ARRAY_UNWRAP_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "MaybeUninit array unwrap transmute must run; stderr={stderr}"
    );
}

const UNSAFE_CELL_ARRAY_ROUND_TRIP_SRC: &str = r#"
use std::cell::UnsafeCell;
use std::mem::transmute;
fn main() {
    let a: [u8; 4] = [10, 20, 30, 40];
    let b: [UnsafeCell<u8>; 4] = unsafe { transmute(a) };
    let c: [u8; 4] = unsafe { transmute(b) };
    let mut s: u32 = 0;
    let mut j = 0usize;
    while j < 4 {
        s += c[j] as u32;
        j += 1;
    }
    assert_eq!(s, 100);
}
"#;

/// The `UnsafeCell` twin of the element-wrapper crossing, in
/// both directions in one program.
#[test]
fn run_mode_o2_transmute_unsafe_cell_array_round_trip() {
    let dir = temp_dir("uc-array-roundtrip");
    let src = fixture(&dir, "case.rs", UNSAFE_CELL_ARRAY_ROUND_TRIP_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "UnsafeCell array round-trip transmute must run; stderr={stderr}"
    );
}

const MANUALLY_DROP_UNION_SRC: &str = r#"
use std::mem::ManuallyDrop;

union MaybeItem<T: Iterator> {
    elem: ManuallyDrop<T::Item>,
    none: (),
}

fn main() {
    let v: Vec<u8> = vec![1, 2, 3];
    let mut i = v.iter();
    i.next();
    let mi = MaybeItem::<std::slice::Iter<'_, u8>> {
        elem: ManuallyDrop::new(i.next().unwrap()),
    };
    unsafe {
        assert_eq!(**mi.elem, 2);
    }
    println!("union ok");
}
"#;

#[test]
fn run_mode_manually_drop_in_union_field_deref() {
    // The union-generic shape: a `ManuallyDrop` read out of a
    // union field is dereferenced twice (`**mi.elem`) — the first
    // star is `ManuallyDrop`'s own `Deref`, the second the
    // reference it holds.
    let dir = temp_dir("manually-drop-union");
    let src = fixture(&dir, "manually_drop_union.rs", MANUALLY_DROP_UNION_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "union ok\n", "stdout={stdout}");
}

// ----- pointer offset arithmetic (Lane DV) --------------------------------
// `BinaryOp::Offset` (what `ptr::add` / `ptr::offset` compute in
// MIR) and the `arith_offset` intrinsic (what the `wrapping_*`
// family lowers to) move a pointer by whole elements. Element
// positions of an array place are represented exactly; positions
// between elements are carried as byte displacements, through
// which reads and writes stay explicit unsupported errors.

const PTR_ADD_SRC: &str = r#"
fn main() {
    let mut arr = [10i32, 20, 30, 40];
    unsafe {
        let p = arr.as_mut_ptr();
        *p.add(2) = 99;
        let q = p.add(1);
        assert_eq!(*q, 20);
        assert_eq!(*p.add(2), 99);
    }
    assert_eq!(arr[2], 99);
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_add_reads_and_writes_elements() {
    let dir = temp_dir("ptr-add");
    let src = fixture(&dir, "ptr_add.rs", PTR_ADD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ptr::add element access must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_OFFSET_SRC: &str = r#"
fn main() {
    let arr = [10i32, 20, 30, 40];
    unsafe {
        let p = arr.as_ptr();
        let q = p.offset(1);
        assert_eq!(*q, 20);
        let r = q.offset(-1);
        assert_eq!(*r, 10);
        let z = q.offset(0);
        assert_eq!(*z, 20);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_offset_checked_round_trip() {
    let dir = temp_dir("ptr-offset");
    let src = fixture(&dir, "ptr_offset.rs", PTR_OFFSET_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "checked ptr::offset round trip must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_WRAPPING_SRC: &str = r#"
fn main() {
    let v: i8 = 42;
    let p = &v as *const i8;
    unsafe {
        let q = p.wrapping_sub(1);
        let r = q.wrapping_add(1);
        assert_eq!(*r, 42);
        let s = p.wrapping_offset(0);
        assert_eq!(*s, 42);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_wrapping_offset_round_trip() {
    let dir = temp_dir("ptr-wrapping");
    let src = fixture(&dir, "ptr_wrapping.rs", PTR_WRAPPING_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "wrapping pointer offsets must round-trip; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_BYTE_SRC: &str = r#"
fn main() {
    let mut arr = [10i32, 20, 30, 40];
    unsafe {
        let p = arr.as_mut_ptr();
        let b = (p as *mut u8).byte_add(8);
        assert_eq!(*(b as *const i32), 30);
        let c = (p as *mut u8).byte_offset(4);
        *(c as *mut i32) = 21;
        let d = (p as *const u8).byte_add(1).byte_add(3);
        assert_eq!(*(d as *const i32), 21);
    }
    assert_eq!(arr[1], 21);
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_byte_add_and_byte_offset() {
    let dir = temp_dir("ptr-byte");
    let src = fixture(&dir, "ptr_byte.rs", PTR_BYTE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte_add/byte_offset on element boundaries must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_BYTE_DISPLACED_READ_SRC: &str = r#"
fn main() {
    let arr = [10i32, 20, 30, 40];
    unsafe {
        let b = (arr.as_ptr() as *const u8).byte_add(1);
        let x = core::ptr::read_unaligned(b as *const i32);
        assert_eq!(x, 0x1400_0000);
        let y = *((arr.as_ptr() as *const u8).byte_add(4));
        assert_eq!(y, 20);
        let z = *((arr.as_ptr() as *const u8).byte_add(5));
        assert_eq!(z, 0);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_byte_displaced_reads_match_native() {
    // A pointer displaced by a non-element number of bytes denotes
    // a position between elements: the load/store path reads and
    // writes it through the type-width byte view of the value's
    // storage (Lane LP, #228), composing the very bytes the native
    // program reads — an unaligned i32 load across an element
    // boundary and single bytes inside elements — never a silent
    // wrong value. Positions the byte view cannot judge (padding,
    // outside the storage) remain explicit unsupported errors.
    let dir = temp_dir("ptr-byte-displaced");
    let src = fixture(&dir, "ptr_byte_displaced.rs", PTR_BYTE_DISPLACED_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte-displaced reads must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
    // The same load through an ordinary dereference carries the
    // alignment assertion the compiler inserts under debug
    // assertions, and the displaced address (base + 1) fails it:
    // the run aborts as a guest crash, exactly as the native
    // program compiled with -Cdebug-assertions=yes does.
    let src = fixture(&dir, "ptr_byte_displaced_aligned.rs",
                      PTR_BYTE_DISPLACED_ALIGNED_LOAD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "aligned load at a displaced address must abort; stderr={stderr}"
    );
    assert!(
        stderr.contains("guest crashed"),
        "the abort is reported as a guest crash: {stderr}"
    );
}

const PTR_BYTE_DISPLACED_ALIGNED_LOAD_SRC: &str = r#"
fn main() {
    let arr = [10i32, 20, 30, 40];
    unsafe {
        let b = (arr.as_ptr() as *const u8).byte_add(1);
        let x = *(b as *const i32);
        println!("{x}");
    }
}
"#;

const ENUM_TAG_BYTE_READ_SRC: &str = r#"
enum E { A(u32), B(u16) }
enum SE { A = -2, B = 3 }
fn main() {
    let a = E::A(0x11223344);
    let b = E::B(0x1234);
    unsafe {
        let pa = &a as *const E as *const u8;
        assert_eq!(*pa, 0);
        assert_eq!(*pa.add(1), 0);
        assert_eq!(*pa.add(4), 0x44);
        let pb = &b as *const E as *const u8;
        assert_eq!(*pb, 1);
        assert_eq!(*pb.add(2), 0x34);
        // A padding byte (offset 2 of A) loads as the model's
        // indeterminate value: carried, never fabricated, and
        // never an error at the read itself.
        let _pad = *pa.add(2);
        let sa = SE::A;
        assert_eq!(*(&sa as *const SE as *const u8), 0xfe);
        let sb = SE::B;
        assert_eq!(*(&sb as *const SE as *const u8), 3);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_enum_tag_byte_reads_match_native() {
    // A byte pointer at an enum's storage reads the tag span as
    // the discriminant's own bytes (Lane MI, #247): the recorded
    // tag layout answers the positions the field rows do not
    // cover, and padding positions answer the indeterminate
    // value instead of the explicit error the byte view raised
    // before the tag layout was recorded.
    let dir = temp_dir("enum-tag-byte-read");
    let src = fixture(&dir, "enum_tag_byte_read.rs", ENUM_TAG_BYTE_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "enum tag byte reads must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const ENUM_TAG_BYTE_WRITE_SRC: &str = r#"
enum E { A(u32), B(u16) }
enum E2 { A(u32), B(u32) }
fn main() {
    let mut e = E::A(0x11223344);
    unsafe { *(&mut e as *mut E as *mut u8) = 1; }
    assert!(matches!(e, E::B(_)));
    let mut e2 = E::B(0x1234);
    unsafe { *(&mut e2 as *mut E as *mut u8) = 0; }
    assert!(matches!(e2, E::A(_)));
    // Shared payload storage: the bytes survive the flip.
    let mut s = E2::A(0x11223344);
    unsafe { *(&mut s as *mut E2 as *mut u8) = 1; }
    match s {
        E2::B(v) => assert_eq!(v, 0x11223344),
        E2::A(_) => panic!("tag write must flip the variant"),
    }
    // A two-byte store over the tag span.
    let mut w = E::A(0x11223344);
    unsafe { *(&mut w as *mut E as *mut u16) = 1; }
    assert!(matches!(w, E::B(_)));
    println!("ok");
}
"#;

#[test]
fn run_mode_enum_tag_byte_write_rewrites_discriminant() {
    // A byte write into the tag span rewrites the discriminant
    // (Lane MI, #247): when the spliced tag names another
    // variant the value is re-formed as that variant, carrying
    // the storage bytes the two variants share, so a branch on
    // the rewritten value takes the native arm.
    let dir = temp_dir("enum-tag-byte-write");
    let src = fixture(&dir, "enum_tag_byte_write.rs", ENUM_TAG_BYTE_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "enum tag byte writes must flip the variant; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_BYTE_VEC_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct Point { x: i32, y: i32 }
fn main() {
    // Byte displacement over a Vec's buffer is denominated in
    // the element type's size: byte_add(8) over a Vec<Point>
    // lands on element 1, not on element 8.
    let v: Vec<Point> = vec![
        Point { x: 1, y: 2 },
        Point { x: 3, y: 4 },
        Point { x: 5, y: 6 },
        Point { x: 7, y: 8 },
        Point { x: 9, y: 10 },
        Point { x: 11, y: 12 },
        Point { x: 13, y: 14 },
        Point { x: 15, y: 16 },
        Point { x: 17, y: 18 },
        Point { x: 19, y: 20 },
    ];
    let s: &[Point] = &v;
    let p = s.as_ptr();
    unsafe {
        assert_eq!(&*p.byte_add(8), &Point { x: 3, y: 4 });
        assert_eq!(&*p.byte_add(16), &Point { x: 5, y: 6 });
        assert_eq!(p.byte_add(16).offset_from(p), 2);
        assert_eq!(p.offset_from(p.byte_add(16)), -2);
        assert!(p.byte_add(8) == p.add(1));
    }
    // A write through a byte-displaced pointer lands in the
    // element the displacement denotes.
    let mut w: Vec<i32> = vec![10, 20, 30];
    let sw: &[i32] = &w;
    unsafe {
        let q = sw.as_ptr().byte_add(4) as *mut i32;
        *q = 99;
    }
    assert_eq!(w, vec![10, 99, 30]);
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_byte_add_over_vec_elements() {
    let dir = temp_dir("ptr-byte-vec");
    let src = fixture(&dir, "ptr_byte_vec.rs", PTR_BYTE_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte_add over Vec elements must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_BYTE_VEC_FIELD_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct Point { x: i32, y: i32 }
struct Holder { v: Vec<Point> }
fn main() {
    // The same denomination holds for a Vec held in a struct
    // field: the position navigates to the field's sequence,
    // then ranges over its elements.
    let h = Holder {
        v: vec![
            Point { x: 1, y: 100 },
            Point { x: 2, y: 200 },
            Point { x: 3, y: 300 },
            Point { x: 4, y: 400 },
            Point { x: 5, y: 500 },
            Point { x: 6, y: 600 },
            Point { x: 7, y: 700 },
            Point { x: 8, y: 800 },
            Point { x: 9, y: 900 },
            Point { x: 10, y: 1000 },
        ],
    };
    let s: &[Point] = &h.v;
    let p = s.as_ptr();
    unsafe {
        assert_eq!(&*p.byte_add(8), &Point { x: 2, y: 200 });
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_byte_add_over_vec_in_struct_field() {
    let dir = temp_dir("ptr-byte-vec-field");
    let src = fixture(&dir, "ptr_byte_vec_field.rs", PTR_BYTE_VEC_FIELD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte_add over a Vec in a struct field must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PTR_BYTE_VEC_DISPLACED_READ_SRC: &str = r#"
#[derive(Debug)]
struct Point { x: i32, y: i32 }
fn main() {
    let v: Vec<Point> = vec![Point { x: 1, y: 2 }, Point { x: 3, y: 4 }];
    let s: &[Point] = &v;
    unsafe {
        let b = s.as_ptr().byte_add(4);
        println!("{:?}", &*b);
    }
    println!("unreachable");
}
"#;

#[test]
fn run_mode_ptr_byte_displaced_read_over_vec_is_explicit_unsupported() {
    // A byte displacement that lands between a Vec's elements
    // denotes no element: reading through it is an explicit
    // unsupported error, never the neighbouring element's value.
    let dir = temp_dir("ptr-byte-vec-displaced");
    let src = fixture(
        &dir,
        "ptr_byte_vec_displaced.rs",
        PTR_BYTE_VEC_DISPLACED_READ_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "mid-element byte-displaced read over a Vec must be an explicit unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("displaced"),
        "the reason names the displacement; stderr={stderr}"
    );
}

const PTR_SLICE_SRC: &str = r#"
fn main() {
    let s: &[i32] = &[10, 20, 30];
    unsafe {
        let p = s.as_ptr();
        assert_eq!(*p.add(2), 30);
        assert_eq!(*p.offset(1), 20);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_ptr_offset_through_slice_pointer() {
    let dir = temp_dir("ptr-slice");
    let src = fixture(&dir, "ptr_slice.rs", PTR_SLICE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "offsets through a slice (fat) pointer must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const MEMORY_INTRINSICS_SRC: &str = r#"
fn main() {
    let mut buf = [0u8; 4];
    unsafe {
        std::ptr::write_bytes(buf.as_mut_ptr(), 0xABu8, 4);
        assert_eq!(buf, [0xAB; 4]);
        let mut v: u32 = 0;
        std::ptr::write_volatile(&mut v, 0x0101_0101);
        assert_eq!(std::ptr::read_volatile(&v), 0x0101_0101);
        let arr = [10u32, 20, 30, 40, 50];
        let base = arr.as_ptr();
        let third = base.add(2);
        assert_eq!(third.offset_from(base), 2);
    }
    println!("ok");
}
"#;

const ELEM_PTR_BULK_SRC: &str = r#"
fn main() {
    let mut dst = [0u8; 6];
    let src = [9u8, 8, 7];
    unsafe {
        core::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().add(2), 3);
        std::ptr::write_bytes(dst.as_mut_ptr().add(5), 0xEEu8, 1);
        std::ptr::write_bytes(dst.as_mut_ptr(), 0xAAu8, 2);
    }
    println!("{:?}", dst);
}
"#;

#[test]
fn run_mode_bulk_ops_through_element_pointers() {
    // Bulk memory operations take a pointer to the sequence's
    // first element: through an element pointer (the shape an
    // array-to-element cast produces) the copy / fill must land on
    // that element and its siblings, not index into the element
    // itself.
    let dir = temp_dir("elem-ptr-bulk");
    let src = fixture(&dir, "elem_ptr_bulk.rs", ELEM_PTR_BULK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "bulk ops through element pointers must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "[170, 170, 9, 8, 7, 238]\n");
}

const RAW_BLOCK_FILL_SRC: &str = r#"
fn main() {
    let s = String::from("ok");
    assert_eq!(s.len(), 2);
}
"#;

#[test]
fn run_mode_raw_block_bulk_fill_under_optimized_mir() {
    // Under optimized MIR, `String::from` / `to_vec` allocate a
    // raw block through `__rust_alloc` and fill it with
    // `copy_nonoverlapping` before any element exists. The fill's
    // destination probe reads element 0 of the still-empty block:
    // an unwritten slot of a raw block must read as uninitialized
    // (the write side then grows the block), never as a guest
    // "index out of bounds" panic -- the guest indexed nothing.
    let dir = temp_dir("raw-block-fill");
    let src = fixture(&dir, "raw_block_fill.rs", RAW_BLOCK_FILL_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("-Copt-level=3")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "bulk fill of a fresh raw block must execute; stderr={stderr}"
    );
}

#[test]
fn run_mode_write_bytes_volatile_and_offset_from_execute() {
    let dir = temp_dir("memory-intrinsics");
    let src = fixture(&dir, "memory_intrinsics.rs", MEMORY_INTRINSICS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "write_bytes, volatile accesses and ptr_offset_from must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const VOLATILE_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::{volatile_load, volatile_store};
#[derive(Debug, Eq, PartialEq)]
struct A(u32);
fn main() {
    unsafe {
        let mut x = A(0);
        volatile_store(&mut x as *mut A, A(1));
        assert_eq!(x, A(1));
        let v = volatile_load(&x as *const A);
        assert_eq!(v, A(1));
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_volatile_composite_access() {
    let dir = temp_dir("volatile");
    let src = fixture(&dir, "volatile.rs", VOLATILE_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

// ----- composite layouts and remaining Default shapes (Lane HQ) -----

const COMPOSITE_LAYOUT_SRC: &str = r#"
fn main() {
    let a: [u16; 3] = [1, 2, 3];
    println!("arr {} {}", std::mem::size_of_val(&a), std::mem::align_of_val(&a));
    let t: (u8, u32) = (1, 2);
    println!("tup {} {}", std::mem::size_of_val(&t), std::mem::align_of_val(&t));
    let n: [[u8; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
    println!("nest {} {}", std::mem::size_of_val(&n), std::mem::align_of_val(&n));
    let m: (u16, [u8; 5]) = (7, [1, 2, 3, 4, 5]);
    println!("mix {} {}", std::mem::size_of_val(&m), std::mem::align_of_val(&m));
}
"#;

#[test]
fn run_mode_size_of_val_composite_layouts() {
    // The layouts are computed from the element / field layouts
    // (verified against a native 1.100.0 build of the same source).
    run_ok(
        "composite-layout",
        COMPOSITE_LAYOUT_SRC,
        "arr 6 2\ntup 8 4\nnest 6 1\nmix 8 2\n",
    );
}

const SCALAR_BYTE_COPY_SRC: &str = r#"
fn main() {
    // A bulk byte copy through a `*mut u8` cast of a scalar's
    // address writes the scalar's storage byte by byte (the
    // pattern `core::hash::Hasher::write` uses to assemble its
    // next word in a `u64` buffer), and a copy back out reads
    // the same bytes.
    let mut buf: u64 = 0;
    let msg: &[u8] = &[1, 2, 3];
    unsafe {
        let p = &mut buf as *mut u64 as *mut u8;
        std::ptr::copy_nonoverlapping(msg.as_ptr(), p, 3);
    }
    assert_eq!(buf, 0x030201);
    let mut out = [0u8; 2];
    unsafe {
        let p = &buf as *const u64 as *const u8;
        std::ptr::copy_nonoverlapping(p.add(1), out.as_mut_ptr(), 2);
    }
    assert_eq!(out, [2, 3]);
    println!("ok {buf}");
}
"#;

#[test]
fn run_mode_byte_copy_through_scalar_storage() {
    let dir = temp_dir("scalar-byte-copy");
    let src = fixture(&dir, "scalar_byte_copy.rs", SCALAR_BYTE_COPY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok 197121\n");
}

const UNSIZE_CUSTOM_PTR_SRC: &str = r#"
#![feature(coerce_unsized, unsize)]
use std::marker::Unsize;
use std::ops::CoerceUnsized;
struct MyPtr<T: ?Sized> { p: *mut T, tag: u32 }
impl<T: ?Sized, U: ?Sized> CoerceUnsized<MyPtr<U>> for MyPtr<T> where T: Unsize<U> {}
fn main() {
    let mut arr = [10u8, 20, 30];
    let a: MyPtr<[u8; 3]> = MyPtr { p: &mut arr, tag: 7 };
    let b: MyPtr<[u8]> = a;
    println!("tag={}", b.tag);
    let s: &[u8] = unsafe { &*b.p };
    println!("len={} s1={}", s.len(), s[1]);
}
"#;

#[test]
fn run_mode_unsize_custom_pointer_multi_field() {
    run_ok("unsize-custom-ptr", UNSIZE_CUSTOM_PTR_SRC, "tag=7\nlen=3 s1=20\n");
}

// ---- Lane IW: one-payload wrapper transmutes (issue #158) ----
// A one-field struct (or a `MaybeUninit`) transmuted to
// exactly its payload type reads the payload field out, and
// the reverse transmute wraps the value back up -- the
// `assume_init` shape an inlined `mem::zeroed` performs
// under optimization.

const TRANSMUTE_PAYLOAD_WRAPPER_SRC: &str = r#"
use std::mem::{ManuallyDrop, MaybeUninit};
struct Wrap(u32);
fn main() {
    let mu = MaybeUninit::new([1u32, 2, 3, 4]);
    let arr: [u32; 4] = unsafe { std::mem::transmute(mu) };
    println!("arr {:?}", arr);
    let mu2: MaybeUninit<[u32; 4]> = unsafe { std::mem::transmute(arr) };
    let arr2 = unsafe { mu2.assume_init() };
    println!("arr2 {:?}", arr2);
    let w = Wrap(41);
    let x: u32 = unsafe { std::mem::transmute(w) };
    let w2: Wrap = unsafe { std::mem::transmute(x + 1) };
    println!("wrap {} {}", x, w2.0);
    let md = ManuallyDrop::new(vec![5, 6, 7]);
    let v: Vec<i32> = unsafe { std::mem::transmute(md) };
    println!("vec {:?}", v);
    let md2: ManuallyDrop<Vec<i32>> = unsafe { std::mem::transmute(v) };
    println!("md {}", md2.len());
}
"#;

#[test]
fn run_mode_transmute_one_payload_wrapper_round_trips() {
    run_ok(
        "transmute-payload-wrapper",
        TRANSMUTE_PAYLOAD_WRAPPER_SRC,
        "arr [1, 2, 3, 4]\narr2 [1, 2, 3, 4]\nwrap 41 42\nvec [5, 6, 7]\nmd 3\n",
    );
}

const FS_OS_ERROR_SRC: &str = r#"
use std::io::ErrorKind;
fn main() {
    let e = std::fs::metadata("/no/such/path-xyz-123").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::NotFound);
    assert_eq!(e.raw_os_error(), Some(2));
    println!("fs-os-error-ok");
}
"#;

#[test]
fn run_mode_fs_os_error_kind_and_raw() {
    let dir = temp_dir("fs-os-error");
    let src = fixture(&dir, "fs_os_error.rs", FS_OS_ERROR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "fs-os-error-ok\n");
}

// ----- Constant pointer `as usize` (Lane KC) -----

const CONST_NULL_PTR_AS_USIZE_SRC: &str = r#"
const P: *mut u8 = core::ptr::null_mut();
fn main() {
    let a = P as usize;
    assert_eq!(a, 0);
}
"#;

/// A null pointer constant cast to an integer: the operand is a
/// constant, so it carries no place type; the constant's own form
/// is the evidence for the `PtrToInt` reading, and the null
/// pointer's address is 0.
#[test]
fn run_mode_const_null_ptr_as_usize() {
    let dir = temp_dir("const-null-ptr-usize");
    let src = fixture(&dir, "case.rs", CONST_NULL_PTR_AS_USIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a null pointer constant must cast to 0; stderr={stderr}"
    );
}

const CONST_ADDR_PTR_AS_USIZE_SRC: &str = r#"
const P: *const u8 = 0x1000 as *const u8;
fn main() {
    let a = P as usize;
    assert_eq!(a, 0x1000);
}
"#;

/// An address-only pointer constant cast to an integer reads back
/// the address it was made from, exactly.
#[test]
fn run_mode_const_addr_ptr_as_usize() {
    let dir = temp_dir("const-addr-ptr-usize");
    let src = fixture(&dir, "case.rs", CONST_ADDR_PTR_AS_USIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an address pointer constant must cast to its address; stderr={stderr}"
    );
}

/// The same reading under optimization, where the constant is
/// folded into the cast's operand position.
#[test]
fn run_mode_const_addr_ptr_as_usize_o2() {
    let dir = temp_dir("const-addr-ptr-usize-o2");
    let src = fixture(&dir, "case.rs", CONST_ADDR_PTR_AS_USIZE_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=2"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an address pointer constant must cast to its address under -O; stderr={stderr}"
    );
}

const CONST_STATIC_PTR_AS_USIZE_SRC: &str = r#"
static X: u32 = 42;
const P: *const u32 = &X;
fn main() {
    let a = P as usize;
    assert!(a != 0);
    let b = &X as *const u32 as usize;
    assert_eq!(a, b);
    assert_eq!(unsafe { *P }, 42);
}
"#;

/// A static-reference pointer constant cast to an integer yields
/// the same synthetic address as the place cast of the same
/// static, and the pointer still reads the static's value.
#[test]
fn run_mode_const_static_ptr_as_usize() {
    let dir = temp_dir("const-static-ptr-usize");
    let src = fixture(&dir, "case.rs", CONST_STATIC_PTR_AS_USIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a static pointer constant must cast to the static's address; stderr={stderr}"
    );
}

const CONST_FN_PTR_AS_USIZE_SRC: &str = r#"
fn f() {}
const P: fn() = f;
fn main() {
    let a = P as usize;
    assert!(a != 0);
    P();
}
"#;

/// A function-item pointer constant cast to an integer yields the
/// callable's synthetic address (nonzero), and the constant still
/// calls.
#[test]
fn run_mode_const_fn_ptr_as_usize() {
    let dir = temp_dir("const-fn-ptr-usize");
    let src = fixture(&dir, "case.rs", CONST_FN_PTR_AS_USIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a fn pointer constant must cast to its address; stderr={stderr}"
    );
}

const CONST_REF_AS_USIZE_SRC: &str = r#"
const R: &u32 = &42;
fn main() {
    let a = R as *const u32 as usize;
    assert!(a != 0);
    let b = &*R as *const u32 as usize;
    assert_eq!(a, b);
    assert_eq!(*R, 42);
}
"#;

/// A reference constant cast through a raw pointer to an integer
/// yields the same synthetic address as the place-derived pointer
/// to the same value.
#[test]
fn run_mode_const_ref_as_usize() {
    let dir = temp_dir("const-ref-usize");
    let src = fixture(&dir, "case.rs", CONST_REF_AS_USIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a reference constant must cast to its value's address; stderr={stderr}"
    );
}

const DROP_IN_PLACE_VEC_FORGET_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("drop D({})", self.0); } }
fn main() {
    let mut v = vec![D(1), D(2)];
    unsafe { std::ptr::drop_in_place(&mut v) };
    std::mem::forget(v);
    println!("after");
}
"#;

/// `ptr::drop_in_place` on a `Vec` runs through the compiler's
/// drop-glue shim, whose field drops name the `RawVec` buffer:
/// the glue conversion must account for it (the buffer's own
/// glue is deallocation only), and the shim's drop must not
/// kill the caller's place -- `forget` still moves it after.
#[test]
fn run_mode_drop_in_place_vec_then_forget() {
    let dir = temp_dir("drop-in-place-vec-forget");
    let src = fixture(&dir, "case.rs", DROP_IN_PLACE_VEC_FORGET_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "drop D(1)\ndrop D(2)\nafter\n");
}

const DROP_IN_PLACE_NESTED_VEC_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("drop D({})", self.0); } }
fn main() {
    let mut v = vec![vec![D(1)], vec![D(2), D(3)]];
    unsafe { std::ptr::drop_in_place(&mut v) };
    std::mem::forget(v);
    println!("after");
}
"#;

/// The nested-`Vec` form of the drop-glue shim path: the outer
/// buffer's field drop and the inner vectors' element glue.
#[test]
fn run_mode_drop_in_place_nested_vec() {
    let dir = temp_dir("drop-in-place-nested-vec");
    let src = fixture(&dir, "case.rs", DROP_IN_PLACE_NESTED_VEC_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "drop D(1)\ndrop D(2)\ndrop D(3)\nafter\n");
}

const DROP_IN_PLACE_RC_FORGET_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("drop D({})", self.0); } }
fn main() {
    let mut r = std::rc::Rc::new(D(9));
    unsafe { std::ptr::drop_in_place(&mut r) };
    std::mem::forget(r);
    println!("after");
}
"#;

/// An `Rc` dropped in place through the shim: the pointee's
/// destructor runs once, and the caller's handle stays
/// movable (the drop through the pointer does not clear the
/// owner's slot).
#[test]
fn run_mode_drop_in_place_rc_then_forget() {
    let dir = temp_dir("drop-in-place-rc-forget");
    let src = fixture(&dir, "case.rs", DROP_IN_PLACE_RC_FORGET_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "drop D(9)\nafter\n");
}

const DROP_IN_PLACE_VEC_SCOPE_DROP_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("drop D({})", self.0); } }
fn main() {
    let mut v = vec![D(1), D(2)];
    unsafe { std::ptr::drop_in_place(&mut v) };
    println!("after");
}
"#;

/// After `drop_in_place`, the scope-end `Drop` terminator for
/// the same local is unconditional in MIR (drop elaboration
/// cannot see through the raw pointer), so the glue runs on
/// the value as it stands -- the same behaviour arrays and
/// struct fields already had through this path.
#[test]
fn run_mode_drop_in_place_vec_scope_drop_after() {
    let dir = temp_dir("drop-in-place-vec-scope");
    let src = fixture(&dir, "case.rs", DROP_IN_PLACE_VEC_SCOPE_DROP_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "drop D(1)\ndrop D(2)\nafter\ndrop D(1)\ndrop D(2)\n");
}

const PUNNED_ACCESS_SRC: &str = r#"
#![feature(core_intrinsics)]

use std::intrinsics::{unaligned_volatile_load, unaligned_volatile_store};

fn main() {
    unsafe {
        let mut arr: [u8; 3] = [1, 2, 3];
        // A u16 load through a pointer into byte storage composes
        // the two stored bytes.
        let ptr = arr[1..].as_mut_ptr() as *mut u16;
        unaligned_volatile_store(ptr, 0);
        assert_eq!(arr, [1, 0, 0]);
        unaligned_volatile_store(ptr, 0x0504);
        assert_eq!(arr, [1, 4, 5]);
        assert_eq!(unaligned_volatile_load(ptr as *const u16), 0x0504);
        println!("ok");
    }
}
"#;

#[test]
fn run_mode_unaligned_volatile_access_composes_bytes() {
    // An unaligned (type-punned) volatile access through a
    // pointer into byte storage reads and writes the composed
    // bytes of the wider integer, not just the first byte.
    let dir = temp_dir("punned-access");
    let src = fixture(&dir, "punned_access.rs", PUNNED_ACCESS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "unaligned volatile accesses must compose bytes; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const CONST_CAST_SRC: &str = r#"
fn main() {
    // A constant operand of a cast is read under its own
    // (source) type: u128::MAX is not -1i128.
    assert!(u128::MAX as f32 == f32::INFINITY);
    assert!(u64::MAX as f32 == 18446744073709551616.0);
    assert!((-1i8) as u32 == u32::MAX);
    assert!((-1i128) as f64 == -1.0);
    println!("ok");
}
"#;

#[test]
fn run_mode_constant_cast_operand_keeps_source_type() {
    // The operand of a cast on a constant must be converted
    // under the constant's own type, not the cast destination's:
    // typing it by the destination read `u128::MAX` as a signed
    // value and the int-to-float cast produced -1.0.
    let dir = temp_dir("const-cast");
    let src = fixture(&dir, "const_cast.rs", CONST_CAST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "constant cast operands keep their source type; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

// ----- zero-sized pointer addresses (Lane NA) -------------------------------

const ZST_VEC_PTR_EQ_SRC: &str = r#"
fn main() {
    let v: Vec<()> = vec![(), (), ()];
    let p = v.as_ptr();
    unsafe {
        println!("eq {}", p.add(1) == p);
        println!("eq2 {}", p.add(2) == p.add(0));
        println!("ne {}", p.add(1) != p);
        println!("lt {}", p.add(1) < p);
        println!("le {}", p.add(1) <= p);
        println!("gt {}", p.add(1) > p);
        println!("int {}", p.add(1) as usize == p as usize);
        println!("ptreq {}", std::ptr::eq(p.add(1), p));
    }
    println!("done");
}
"#;

const ZST_ARRAY_PTR_EQ_SRC: &str = r#"
struct Z;
fn main() {
    let a = [Z, Z, Z, Z];
    let p = a.as_ptr();
    unsafe {
        println!("arr {}", p.add(3) == p);
    }
    let s: &[Z] = &a;
    println!("slice {}", s.as_ptr() == p);
    let mut m: Vec<std::mem::MaybeUninit<()>> = Vec::new();
    m.push(std::mem::MaybeUninit::new(()));
    m.push(std::mem::MaybeUninit::uninit());
    let q = m.as_ptr();
    unsafe {
        println!("mu {}", q.add(1) == q);
    }
    println!("done");
}
"#;

const SIZED_PTR_POSITIONS_SRC: &str = r#"
fn main() {
    let v: Vec<i32> = vec![10, 20, 30];
    let p = v.as_ptr();
    unsafe {
        println!("eq0 {}", p.add(0) == p);
        println!("eq1 {}", p.add(1) == p);
        println!("ne12 {}", p.add(1) != p.add(2));
        println!("ordered {}", (p < p.add(1)) ^ (p.add(1) < p));
        println!("refl {}", p.add(1) <= p.add(1) && p.add(1) >= p.add(1));
    }
    println!("done");
}
"#;

#[test]
fn run_mode_zst_vec_pointer_positions_share_the_base_address() {
    // A zero-sized element occupies no bytes, so every element
    // position of a `Vec<()>` is the same address natively:
    // `p.add(1) == p`, no strict ordering between positions,
    // and the integer casts agree (Lane MV denominated the
    // offsets in elements, which had made the positions
    // distinct synthetic addresses).
    let dir = temp_dir("zst-vec-ptr-eq");
    let src = fixture(&dir, "zst_vec_ptr_eq.rs", ZST_VEC_PTR_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ZST pointer equality program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "eq true\neq2 true\nne false\nlt false\nle true\ngt false\nint true\nptreq true\ndone\n"
    );
}

#[test]
fn run_mode_zst_array_slice_and_maybeuninit_pointer_equality() {
    // The same address rule through an array root, a slice
    // fat pointer and a `Vec<MaybeUninit<()>>` buffer.
    let dir = temp_dir("zst-array-ptr-eq");
    let src = fixture(&dir, "zst_array_ptr_eq.rs", ZST_ARRAY_PTR_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ZST array pointer equality program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "arr true\nslice true\nmu true\ndone\n"
    );
}

#[test]
fn run_mode_sized_pointer_positions_stay_distinct() {
    // Sized elements keep their position distinction: only
    // zero-sized element positions collapse onto the base.
    // Distinct positions stay distinct and strictly ordered
    // one way or the other (the synthetic address order itself
    // is the hash order, as before this lane).
    let dir = temp_dir("sized-ptr-positions");
    let src = fixture(&dir, "sized_ptr_positions.rs", SIZED_PTR_POSITIONS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "sized pointer position program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "eq0 true\neq1 false\nne12 true\nordered true\nrefl true\ndone\n"
    );
}

const VEC_ZST_SHIFT_SRC: &str = r#"
struct D;
impl Drop for D {
    fn drop(&mut self) {
        println!("drop D");
    }
}
fn main() {
    let mut u: Vec<()> = vec![(), ()];
    u.insert(0, ());
    println!("ulen {}", u.len());
    u.remove(0);
    println!("ulen {}", u.len());
    let mut v = vec![D, D];
    v.insert(1, D);
    println!("len {}", v.len());
    v.remove(0);
    println!("len {}", v.len());
    drop(v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_macro_zst_elements_shift() {
    // The `vec![...]` list form over zero-sized elements: rustc
    // elides the element writes into the `Box::new_uninit`
    // buffer (a zero-sized write is a no-op), so the slots the
    // interpreter's `box_assume_init_into_vec_unsafe` model
    // reads still hold the uninitialized marker. Conversion
    // bakes the element type's one value into the call and the
    // model materializes the unwritten slots from it, so the
    // shift operations (`insert` / `remove`, which copy the
    // buffer's elements) read real elements and the destructors
    // run in order, byte for byte as natively.
    let dir = temp_dir("veczstshift");
    let src = fixture(&dir, "veczstshift.rs", VEC_ZST_SHIFT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of ZST elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "ulen 3\nulen 2\nlen 3\ndrop D\nlen 2\ndrop D\ndrop D\ndone\n",
        "vec! ZST elements shift and drop exactly as natively"
    );
}

const VEC_SET_LEN_MAYBE_UNINIT_SRC: &str = r#"
fn main() {
    // A slot the length declares but no write reached holds the
    // element type's uninitialized form: for a `MaybeUninit`
    // element that is the uninitialized `MaybeUninit` value
    // itself, which reads back and can be carried (here,
    // forgotten) without touching its payload.
    let mut v: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(2);
    unsafe {
        v.set_len(2);
        let b = std::ptr::read(v.as_ptr());
        std::mem::forget(b);
        let c = v[1];
        std::mem::forget(c);
    }
    println!("carried");

    // The slots are real storage: writing through the buffer
    // pointer and reading back observes the writes.
    let mut w: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(2);
    unsafe {
        w.set_len(2);
        w.as_mut_ptr().write(std::mem::MaybeUninit::new(41));
        w.as_mut_ptr().add(1).write(std::mem::MaybeUninit::new(42));
        println!("{} {}", w[0].assume_init(), w[1].assume_init());
    }

    // A written prefix survives a growth: the first element
    // keeps its value, the grown slot stays uninitialized.
    let mut p: Vec<std::mem::MaybeUninit<u8>> = Vec::with_capacity(2);
    p.push(std::mem::MaybeUninit::new(7));
    unsafe {
        p.set_len(2);
        println!("first {}", std::ptr::read(p.as_ptr()).assume_init());
        std::mem::forget(std::ptr::read(p.as_ptr().add(1)));
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_set_len_maybe_uninit_slots() {
    run_ok(
        "vec-set-len-maybe-uninit",
        VEC_SET_LEN_MAYBE_UNINIT_SRC,
        "carried\n41 42\nfirst 7\ndone\n",
    );
}

const VEC_SET_LEN_PLAIN_UNINIT_READ_SRC: &str = r#"
fn main() {
    let mut vec: Vec<u8> = Vec::with_capacity(2);
    unsafe { vec.set_len(2) }
    // A plain element type's grown slot has no value: reading
    // it is an uninitialised read, exactly as a sparse tail
    // slot's read is.
    println!("{}", vec[0]);
}
"#;

#[test]
fn run_mode_vec_set_len_plain_uninit_read() {
    let dir = temp_dir("vec-set-len-uninit");
    let src = fixture(&dir, "case.rs", VEC_SET_LEN_PLAIN_UNINIT_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("uninitialized"),
        "grown slot read must be an uninitialised read; stderr={stderr}"
    );
}
const VEC_INSERT_WRAPPER_ELEM_SRC: &str = r#"
struct D(&'static str);
impl Drop for D {
    fn drop(&mut self) { println!("drop {}", self.0); }
}
struct N(i32);
fn main() {
    let mut v: Vec<D> = Vec::with_capacity(1);
    v.push(D("a"));
    v.push(D("b"));
    v.push(D("c"));
    v.insert(1, D("x"));
    for d in v.iter() { println!("elem {}", d.0); }
    drop(v);
    let mut n: Vec<N> = Vec::with_capacity(8);
    n.push(N(1)); n.push(N(2)); n.push(N(3));
    n.insert(1, N(9));
    for e in n.iter() { println!("num {}", e.0); }
    let mut a = [D("a"), D("b"), D("c"), D("?")];
    unsafe {
        let p = a.as_mut_ptr();
        std::ptr::copy(p.add(1), p.add(2), 2);
    }
    for d in a.iter() { println!("arr {}", d.0); }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_insert_one_field_wrapper_elements() {
    // The shift inside `Vec::insert` is an element copy
    // (`ptr::copy` over the buffer), and its elements here are
    // one-field wrapper structs. The copy must move the
    // elements whole: a wrapper over a scalar is not a byte
    // tree to patch with the following elements' bytes, and a
    // wrapper over a fat pointer is not a wrapper place to
    // rewrap the incoming element inside. Shifted elements
    // read back exactly, fields and drops included, as does a
    // direct `ptr::copy` over an array of the same elements.
    let dir = temp_dir("vecinsertwrapper");
    let src = fixture(&dir, "vecinsertwrapper.rs", VEC_INSERT_WRAPPER_ELEM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::insert of one-field wrapper elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "elem a\nelem x\nelem b\nelem c\ndrop a\ndrop x\ndrop b\ndrop c\nnum 1\nnum 9\nnum 2\nnum 3\narr a\narr b\narr b\narr c\ndone\ndrop a\ndrop b\ndrop b\ndrop c\n"
    );
}

const SLICE_JOIN_SRC: &str = r#"
fn main() {
    // `slice::join` builds its `String` through
    // `join_generic_copy`: a `Vec<u8>` in its real aggregate
    // form, filled through raw-pointer copies, returned as a
    // `String`. Its core UB checks (`ptr::offset_from_unsigned`
    // preconditions) compare positions of one allocation, and
    // the result denotes its string through the buffer.
    let parts: &[&str] = &["a", "bb", "ccc"];
    let s = parts.join("-");
    println!("{}", s);
    println!("{:?}", s);
    println!("len={} lt={}", s.len(), s < "b".to_string());
    assert_eq!(s, "a-bb-ccc");
    let e: &[&str] = &[];
    println!("[{}]", e.join("-"));
    let one: &[&str] = &["x"];
    println!("{}", one.join("---"));
    let v = vec!["p".to_string(), "q".to_string()];
    println!("{}", v.join("::"));
    println!("{}", parts.concat());
    println!("{}", s.to_uppercase());
}
"#;

#[test]
fn run_mode_slice_join() {
    let dir = temp_dir("slice-join");
    let src = fixture(&dir, "slice_join.rs", SLICE_JOIN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "slice::join must run without a UB-check failure; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a-bb-ccc\n\"a-bb-ccc\"\nlen=8 lt=true\n[]\nx\np::q\nabbccc\nA-BB-CCC\n",
        "the joined string reads back through its real aggregate form"
    );
    assert!(
        !stderr.contains("offset_from_unsigned"),
        "no UB-check message fires: {stderr}"
    );
}

const WAKER_FROM_ARC_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll, Wake, Waker};

fn simple_waker() -> (Waker, mpsc::Receiver<()>) {
    struct SimpleWaker {
        tx: mpsc::Sender<()>,
    }
    impl Wake for SimpleWaker {
        fn wake(self: Arc<Self>) {
            self.tx.send(()).unwrap();
        }
    }
    let (tx, rx) = mpsc::channel();
    (Waker::from(Arc::new(SimpleWaker { tx })), rx)
}

fn main() {
    let (waker, _rx) = simple_waker();
    let mut cx = Context::from_waker(&waker);
    let fut = pin!(async { 42 });
    let mut fut = fut;
    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(v) => println!("ready {v}"),
        Poll::Pending => println!("pending"),
    }
    waker.wake_by_ref();
    println!("woken");
}
"#;

#[test]
fn run_mode_waker_from_arc_wake_by_ref_default() {
    // `Waker::from(Arc<T: Wake>)` with only `wake` defined:
    // `wake_by_ref` runs the trait's default body (clone the
    // handle, wake it), which the wake demand converts.
    run_ok_edition("waker-from-arc", WAKER_FROM_ARC_SRC, "ready 42\nwoken\n");
}

const WAKER_OVERRIDE_SRC: &str = r#"
use std::sync::{Arc, Weak};
use std::task::{Wake, Waker};

struct W2 {
    tag: u32,
}
impl Wake for W2 {
    fn wake(self: Arc<Self>) {
        println!("wake {}", self.tag);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        println!("wake_by_ref override {}", self.tag);
    }
}

fn main() {
    let arc = Arc::new(W2 { tag: 9 });
    let weak: Weak<W2> = Arc::downgrade(&arc);
    let waker = Waker::from(arc);
    println!("count {}", weak.strong_count());
    waker.wake_by_ref();
    println!("count {}", weak.strong_count());
    let waker2 = waker.clone();
    waker2.wake_by_ref();
    waker.wake();
    println!("count {}", weak.strong_count());
    waker2.wake_by_ref();
    drop(waker2);
    println!("count {}", weak.strong_count());
}
"#;

#[test]
fn run_mode_waker_wake_override_and_counts() {
    // An overridden `wake_by_ref` runs the override, and the
    // strong count the wake consumes / preserves is the native
    // one at every step.
    run_ok_edition(
        "waker-override",
        WAKER_OVERRIDE_SRC,
        "count 1\nwake_by_ref override 9\ncount 1\nwake_by_ref override 9\nwake 9\ncount 1\nwake_by_ref override 9\ncount 0\n",
    );
}

const WAKER_RAW_VTABLE_SRC: &str = r#"
use std::sync::Arc;
use std::task::{RawWaker, RawWakerVTable, Wake, Waker};

struct W3 {
    tag: u32,
}
impl Wake for W3 {
    fn wake(self: Arc<Self>) {
        println!("w3 wake {}", self.tag);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        println!("w3 wake_by_ref {}", self.tag);
    }
}

unsafe fn clone_fn(data: *const ()) -> RawWaker {
    let arc = unsafe { Arc::<W3>::from_raw(data as *const W3) };
    let cloned = arc.clone();
    std::mem::forget(arc);
    RawWaker::new(Arc::into_raw(cloned) as *const (), &VTABLE)
}
unsafe fn wake_fn(data: *const ()) {
    let arc = unsafe { Arc::<W3>::from_raw(data as *const W3) };
    arc.wake();
}
unsafe fn wake_by_ref_fn(data: *const ()) {
    let arc = unsafe { Arc::<W3>::from_raw(data as *const W3) };
    arc.wake_by_ref();
    std::mem::forget(arc);
}
unsafe fn drop_fn(data: *const ()) {
    let arc = unsafe { Arc::<W3>::from_raw(data as *const W3) };
    drop(arc);
}
static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_fn, wake_fn, wake_by_ref_fn, drop_fn);

fn main() {
    let arc = Arc::new(W3 { tag: 4 });
    let raw = RawWaker::new(Arc::into_raw(arc) as *const (), &VTABLE);
    let waker = unsafe { Waker::from_raw(raw) };
    waker.wake_by_ref();
    let waker2 = waker.clone();
    waker2.wake_by_ref();
    waker.wake();
    println!("mid");
    waker2.wake();
    println!("done");
}
"#;

#[test]
fn run_mode_waker_from_raw_vtable_wake() {
    // A hand-built `RawWaker`: wake and wake_by_ref run the
    // vtable's entries with the data pointer.
    run_ok_edition(
        "waker-raw-vtable",
        WAKER_RAW_VTABLE_SRC,
        "w3 wake_by_ref 4\nw3 wake_by_ref 4\nw3 wake 4\nmid\nw3 wake 4\ndone\n",
    );
}

const SLICE_JOIN_CLOSURE_CONST_SRC: &str = r#"
fn main() {
    let a: &[&[i32]] = &[&[1, 2], &[3]];
    let j = a.join(&0);
    println!("done {}", j.len());
}
"#;

#[test]
fn run_mode_slice_join_closure_constant_converts() {
    // The length fold inside slice Join::join reaches conversion as a
    // closure *type constant* whose environment captures the sum
    // closure (const-prop folded, all captures zero-sized).
    // Conversion must materialize that environment instead of
    // rejecting the constant; before the fix this program died in
    // conversion with "closure constant ... with a captured
    // environment". The joined length itself is Vec bookkeeping
    // territory and is not what this test pins.
    let dir = temp_dir("slice-join-closure-const");
    let src = fixture(&dir, "case.rs", SLICE_JOIN_CLOSURE_CONST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "join runs; stderr={stderr}");
    assert!(
        String::from_utf8_lossy(&out.stdout).starts_with("done "),
        "join completed: {stderr}"
    );
}

const ZIP_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3];
    let b = [10, 20, 30, 40];
    for (x, y) in a.iter().zip(b.iter()) {
        println!("{} {}", x, y);
    }
    let v: Vec<(&i32, &i32)> = a.iter().zip(b.iter()).collect();
    println!("collected {}", v.len());
    println!("count {}", a.iter().zip(b.iter()).count());
    for ((x, y), w) in a.iter().zip(b.iter()).zip([100, 200, 300].iter()) {
        println!("{} {} {}", x, y, w);
    }
    let s: i32 = a.iter().zip(b.iter()).map(|(x, y)| x + y).sum();
    println!("sum {}", s);
    for (x, y) in std::iter::zip([1, 2], [3, 4]) {
        println!("arr {} {}", x, y);
    }
}
"#;

#[test]
fn run_mode_zip_iteration() {
    let dir = temp_dir("zip-iteration");
    let src = fixture(&dir, "zip_iteration.rs", ZIP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Zip over slice iterators and by-value arrays must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1 10\n2 20\n3 30\ncollected 3\ncount 3\n1 10 100\n2 20 200\n3 30 300\nsum 66\narr 1 3\narr 2 4\n",
        "pairs are yielded front-aligned up to the shorter side"
    );
}

const ZIP_DOUBLE_ENDED_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3, 4];
    let b = [10, 20, 30];
    let mut z = a.iter().zip(b.iter());
    println!("hint {:?}", z.size_hint());
    println!("back {:?}", z.next_back());
    println!("next {:?}", z.next());
    println!("back {:?}", z.next_back());
    println!("back {:?}", z.next_back());
    println!("len {}", a.iter().zip(b.iter()).len());
    let mut z2 = std::iter::zip([1, 2, 3], [7, 8, 9, 10]);
    println!("mix {:?}", z2.next_back());
    println!("mix {:?}", z2.next());
    println!("mix {:?}", z2.next_back());
    println!("mix {:?}", z2.next());
}
"#;

#[test]
fn run_mode_zip_double_ended() {
    let dir = temp_dir("zip-double-ended");
    let src = fixture(&dir, "zip_double_ended.rs", ZIP_DOUBLE_ENDED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Zip next_back with unequal lengths must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hint (3, Some(3))\nback Some((3, 30))\nnext Some((1, 10))\nback Some((2, 20))\nback None\nlen 3\nmix Some((3, 9))\nmix Some((1, 7))\nmix Some((2, 8))\nmix None\n",
        "the back step drops the longer side's unpaired excess first"
    );
}

// ----- exposed-provenance pointer round trips ----------------------
//
// A pointer exposed as an integer (a cast or a scalar transmute)
// records its target under the synthetic address it produced;
// materialising a pointer from that integer restores the target,
// so a rebuilt `Box` dereferences to its pointee instead of an
// address-only pointer that denotes no readable place.

const EXPOSED_TRANSMUTE_BOX_SRC: &str = r#"
fn main() {
    let b = Box::new(7u32);
    let n: usize = unsafe { std::mem::transmute(b) };
    let b2: Box<u32> = unsafe { std::mem::transmute(n) };
    println!("{}", *b2);
}
"#;

#[test]
fn run_mode_exposed_provenance_transmute_box_round_trip() {
    run_ok("exposed-transmute-box", EXPOSED_TRANSMUTE_BOX_SRC, "7\n");
}

const EXPOSED_CAST_FROM_RAW_SRC: &str = r#"
fn main() {
    let b = Box::new(7u32);
    let raw = Box::into_raw(b);
    let n = raw as usize;
    let raw2 = n as *mut u32;
    println!("via raw: {}", unsafe { *raw2 });
    let b2 = unsafe { Box::from_raw(raw2) };
    println!("via box: {}", *b2);
}
"#;

#[test]
fn run_mode_exposed_provenance_cast_from_raw_round_trip() {
    run_ok(
        "exposed-cast-from-raw",
        EXPOSED_CAST_FROM_RAW_SRC,
        "via raw: 7\nvia box: 7\n",
    );
}

const EXPOSED_DROP_ONCE_SRC: &str = r#"
struct Counted(u32);
static mut DROPS: u32 = 0;
impl Drop for Counted {
    fn drop(&mut self) { unsafe { DROPS += 1; } }
}
fn main() {
    let b = Box::new(Counted(11));
    let n: usize = unsafe { std::mem::transmute(b) };
    let mut b2: Box<Counted> = unsafe { std::mem::transmute(n) };
    println!("v={}", b2.0);
    b2.0 = 12;
    println!("v={}", (*b2).0);
    drop(b2);
    println!("drops={}", unsafe { DROPS });
}
"#;
#[test]
fn run_mode_box_enum_tag_byte_view_matches_native() {
    // A byte pointer at a `Box` allocation's storage reads and
    // writes the enum's tag span, as it does for a place: the
    // allocation's recorded declared type lets the byte view
    // and the `MaybeUninit` entry gate judge the position as
    // the enum's own storage (Lane OC, #292). Before the
    // record existed, the byte read answered the payload's
    // first byte and the byte write spliced into the payload.
    let dir = temp_dir("box-enum-tag-byte-view");
    let src = fixture(&dir, "box_enum_tag_byte_view.rs", BOX_ENUM_TAG_BYTE_VIEW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "box enum tag byte view must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_ENUM_TAG_BYTE_VIEW_SRC: &str = r#"
enum E { A(u32), B(u16) }
fn main() {
    let b = Box::new(E::A(0x11223344));
    let tag = unsafe { *(&*b as *const E as *const u8) };
    assert_eq!(tag, 0);
    let mut b2 = Box::new(E::A(0x11223344));
    unsafe { *(&mut *b2 as *mut E as *mut u8) = 1; }
    assert!(matches!(*b2, E::B(_)));
    let raw = Box::into_raw(Box::new(E::A(0x11223344)));
    let tag = unsafe { *(raw as *const u8) };
    assert_eq!(tag, 0);
    unsafe { *(raw as *mut u8) = 1; }
    let b3 = unsafe { Box::from_raw(raw) };
    assert!(matches!(*b3, E::B(_)));
    println!("ok");
}
"#;

#[test]
fn run_mode_exposed_provenance_round_trip_drops_once() {
    run_ok(
        "exposed-drop-once",
        EXPOSED_DROP_ONCE_SRC,
        "v=11\nv=12\ndrops=1\n",
    );
}
#[test]
fn run_mode_box_byte_add_over_adt_elements_matches_native() {
    // A byte-denominated pointer step over a `Box<[E; N]>`
    // lands on the element it denotes natively: the
    // allocation's recorded array type denominates the
    // displacement by the element size (Lane OC, #292).
    // Before the record existed, the displaced read was an
    // explicit "displaced from a value boundary" error.
    let dir = temp_dir("box-byte-add-adt");
    let src = fixture(&dir, "box_byte_add_adt.rs", BOX_BYTE_ADD_ADT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "box byte_add over ADT elements must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_BYTE_ADD_ADT_SRC: &str = r#"
#[derive(Clone, Copy, PartialEq, Debug)]
enum E { A(u32), B(u16) }
fn main() {
    let b = Box::new([E::A(11), E::B(22), E::A(33)]);
    let mut p = b.as_ptr();
    unsafe {
        assert_eq!(*p, E::A(11));
        p = p.byte_add(std::mem::size_of::<E>());
        assert_eq!(*p, E::B(22));
        p = p.byte_add(std::mem::size_of::<E>());
        assert_eq!(*p, E::A(33));
        p = p.byte_sub(2 * std::mem::size_of::<E>());
        assert_eq!(*p, E::A(11));
    }
    // The same denomination over a boxed slice (a recorded
    // slice type) and for byte-denominated writes.
    let v = vec![E::A(11), E::B(22), E::A(33)];
    let bs = v.into_boxed_slice();
    let mut q = bs.as_ptr();
    unsafe {
        q = q.byte_add(std::mem::size_of::<E>());
        assert_eq!(*q, E::B(22));
    }
    let mut w = Box::new([0u32; 4]);
    let wp = w.as_mut_ptr();
    unsafe {
        *wp.byte_add(4) = 77;
        *wp.byte_add(12) = 99;
    }
    assert_eq!(*w, [0, 77, 0, 99]);
    println!("ok");
}
"#;

#[test]
fn run_mode_box_array_byte_ptr_cast_back_matches_native() {
    // A byte pointer advanced into a `Box<[u32; 3]>` is
    // absorbed as an index denominated by the allocation's
    // recorded element type (Lane OC, #292); casting it back
    // to `*const u32` must not redenominate that index by the
    // byte pointer's units -- the cast does not move the
    // address. Before the cast-side fix, the index was
    // rewritten as a one- or two-byte offset and the reads
    // answered the elements' shifted bytes.
    let dir = temp_dir("box-array-byte-ptr-cast-back");
    let src = fixture(&dir, "box_array_byte_ptr_cast_back.rs", BOX_ARRAY_BYTE_PTR_CAST_BACK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte pointer cast back over a boxed array must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "20\n30\n20\n30\n");
}

const BOX_ARRAY_BYTE_PTR_CAST_BACK_SRC: &str = r#"
fn main() {
    let b: Box<[u32; 3]> = Box::new([10, 20, 30]);
    let p = &*b as *const [u32; 3] as *const u8;
    unsafe {
        let r = p.add(4) as *const u32;
        println!("{}", *r);
        let r2 = p.add(8) as *const u32;
        println!("{}", *r2);
    }
    // The same denomination over a boxed slice (a recorded
    // slice type).
    let bs = vec![10u32, 20, 30].into_boxed_slice();
    let q = bs.as_ptr() as *const u8;
    unsafe {
        let r = q.add(4) as *const u32;
        println!("{}", *r);
        let r2 = q.add(8) as *const u32;
        println!("{}", *r2);
    }
}
"#;

#[test]
fn run_mode_box_transparent_wrapper_cast_read_matches_native() {
    // Reading a boxed transparent wrapper's payload through a
    // raw pointer cast to the payload type: the allocation's
    // recorded ADT type must not refuse the payload entry --
    // the stored form is a wrapper layer around the payload,
    // at the payload's own width and at byte width alike.
    // (The recorded type still judges byte positions of a
    // genuine enum's storage; see
    // `run_mode_box_enum_tag_byte_view_matches_native`.)
    let dir = temp_dir("box-transparent-wrapper-cast-read");
    let src = fixture(
        &dir,
        "box_transparent_wrapper_cast_read.rs",
        BOX_TRANSPARENT_WRAPPER_CAST_READ_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "transparent wrapper payload reads through casts must run with native values; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_TRANSPARENT_WRAPPER_CAST_READ_SRC: &str = r#"
#[repr(transparent)]
struct W<T>(T);

fn main() {
    let b: Box<W<u32>> = Box::new(W(42));
    let p = &*b as *const W<u32> as *const u32;
    let v = unsafe { *p };
    assert_eq!(v, 42);
    // The payload at byte width: a single-field struct's
    // storage is its field's, with no tag span in front.
    let b2: Box<W<u8>> = Box::new(W(0x5a));
    let bp = &*b2 as *const W<u8> as *const u8;
    let byte = unsafe { *bp };
    assert_eq!(byte, 0x5a);
    println!("ok");
}
"#;

const DEEP_RECURSION_SRC: &str = r#"
fn f(x: isize) -> isize {
    if x == 1 { return 1; } else { let y: isize = 1 + f(x - 1); return y; }
}

fn main() { println!("depth = {}", f(5000)); }
"#;

#[test]
fn run_mode_deep_recursion_completes() {
    run_ok("deep-recursion", DEEP_RECURSION_SRC, "depth = 5000\n");
}

const ERROR_SOURCE_CHAIN_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct Inner;
impl fmt::Display for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "inner") }
}
impl Error for Inner {}
#[derive(Debug)]
struct Outer { src: Inner }
impl fmt::Display for Outer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "outer") }
}
impl Error for Outer {
    fn source(&self) -> Option<&(dyn Error + 'static)> { Some(&self.src) }
}
#[derive(Debug)]
struct Plain;
impl fmt::Display for Plain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "plain") }
}
impl Error for Plain {}
fn main() {
    let outer = Outer { src: Inner };
    let e: &dyn Error = &outer;
    let s = e.source().expect("outer has source");
    println!("source display: {}", s);
    println!("source-of-source is none: {}", s.source().is_none());
    let plain = Plain;
    let e2: &dyn Error = &plain;
    println!("plain source is none: {}", e2.source().is_none());
    let boxed: Box<dyn Error> = Box::new(Outer { src: Inner });
    println!("boxed source: {}", boxed.source().map(|s| s.to_string()).unwrap_or_else(|| "none".into()));
    let pie = "x".parse::<u32>().unwrap_err();
    let e3: &dyn Error = &pie;
    println!("foreign source is none: {}", e3.source().is_none());
}
"#;

#[test]
fn run_mode_error_source_chain_and_default() {
    // The vtable demand puts every `Error` method of a coerced
    // concrete type in the program — an overriding `source`, and
    // the trait's default body (which answers `None`) for types
    // that do not override it, local and foreign alike.
    run_ok(
        "error-source-chain",
        ERROR_SOURCE_CHAIN_SRC,
        "source display: inner\nsource-of-source is none: true\nplain source is none: true\nboxed source: inner\nforeign source is none: true\n",
    );
}

const ERROR_DOWNCAST_SRC: &str = r#"
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
static DROPS: AtomicUsize = AtomicUsize::new(0);
#[derive(Debug)]
struct Counted { code: u32 }
impl fmt::Display for Counted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "counted {}", self.code) }
}
impl Error for Counted {}
impl Drop for Counted { fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); } }
#[derive(Debug)]
struct Other;
impl fmt::Display for Other {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "other") }
}
impl Error for Other {}
fn main() {
    let boxed: Box<dyn Error> = Box::new(Counted { code: 42 });
    println!("is Counted: {}", boxed.is::<Counted>());
    println!("is Other: {}", boxed.is::<Other>());
    println!("ref code: {}", boxed.downcast_ref::<Counted>().map(|r| r.code).unwrap_or(0));
    let b = boxed.downcast::<Counted>().unwrap();
    println!("box debug: {:?}", b);
    println!("code: {}", b.code);
    let e: Box<dyn Error> = b;
    println!("back via ref: {}", e.downcast_ref::<Counted>().unwrap().code);
    drop(e);
    println!("drops after ok path: {}", DROPS.load(Ordering::SeqCst));
    let boxed2: Box<dyn Error> = Box::new(Counted { code: 2 });
    let back = boxed2.downcast::<Other>().unwrap_err();
    println!("err display: {}", back);
    drop(back);
    println!("drops after err path: {}", DROPS.load(Ordering::SeqCst));
    let send_box: Box<dyn Error + Send> = Box::new(Counted { code: 3 });
    println!("send code: {}", send_box.downcast::<Counted>().unwrap().code);
    println!("drops final: {}", DROPS.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_error_downcast_by_value() {
    // By-value `Box<dyn Error>::downcast` runs std's collected
    // body: the `type_id` vtable dispatch decides, the fat-to-thin
    // pointer cast extracts the data pointer, and the rebuilt
    // box owns the payload exactly once on both the Ok and the
    // Err path (one drop each, counted).
    run_ok(
        "error-downcast",
        ERROR_DOWNCAST_SRC,
        "is Counted: true\nis Other: false\nref code: 42\nbox debug: Counted { code: 42 }\ncode: 42\nback via ref: 42\ndrops after ok path: 1\nerr display: counted 2\ndrops after err path: 2\nsend code: 3\ndrops final: 3\n",
    );
}

const JOIN_PAYLOAD_INTO_RAW_SRC: &str = r#"
use std::thread;
struct Pair { a: i64, b: i64 }
fn main() {
    let handle = thread::spawn(|| -> () {
        std::panic::panic_any(Pair { a: 33, b: 44 });
    });
    let err = handle.join().unwrap_err();
    let raw = Box::into_raw(err) as *mut Pair;
    unsafe {
        println!("fields: {} {}", (*raw).a, (*raw).b);
    }
}
"#;

#[test]
fn run_mode_join_payload_into_raw_cast_reads_whole_struct() {
    // The thread-join error payload has the same inline-box
    // trait-object shape as the catch_unwind cleanup product:
    // the data pointer extracted for a cast to the concrete
    // pointer type denotes the whole payload struct.
    run_ok(
        "join-payload-into-raw",
        JOIN_PAYLOAD_INTO_RAW_SRC,
        "fields: 33 44\n",
    );
}

const IPV4_FROM_CONVERSIONS_SRC: &str = r#"
use std::net::Ipv4Addr;
fn main() {
    println!("from array: {}", Ipv4Addr::from([1u8, 2, 3, 4]));
    println!("from u32: {}", Ipv4Addr::from(0x7f000001u32));
    println!("into addr: {}", <Ipv4Addr as From<[u8; 4]>>::from([10u8, 0, 0, 255]));
    let a = Ipv4Addr::new(127, 0, 0, 1);
    println!("to u32: {}", u32::from(a));
    println!("bits: {}", a.to_bits());
}
"#;

#[test]
fn run_mode_ipv4_addr_from_conversions() {
    // The `From` conversions between `Ipv4Addr`, `[u8; 4]`
    // and `u32` (big-endian) build / read the same octet
    // shape the formatter renders.
    run_ok(
        "ipv4-from",
        IPV4_FROM_CONVERSIONS_SRC,
        "from array: 1.2.3.4\nfrom u32: 127.0.0.1\ninto addr: 10.0.0.255\nto u32: 2130706433\nbits: 2130706433\n",
    );
}

const TRANSMUTE_TUPLE_FIELD_SRC: &str = r#"
#[repr(C, packed)]
struct S {
    a: (u8, u16, u8),
    b: u8,
}

fn main() {
    let bytes: [u8; 5] = [0x34, 0x12, 0x56, 0x78, 0x9a];
    let s: S = unsafe { std::mem::transmute(bytes) };
    let a = s.a;
    let b = s.b;
    println!("{} {} {} {}", a.0, a.1, a.2, b);
}
"#;

#[test]
fn run_mode_transmute_tuple_field_at_recorded_offsets() {
    // A tuple field of a packed struct transmuted from a
    // byte array: the tuple's storage order is the
    // compiler's choice (here the `u16` sits first, at
    // offset 0), so the field's fields are read at the
    // tuple's recorded layout offsets, not in declaration
    // order.
    run_ok(
        "transmute-tuple-field",
        TRANSMUTE_TUPLE_FIELD_SRC,
        "86 4660 120 154\n",
    );
}

const TRANSMUTE_TUPLE_LAYOUT_SRC: &str = r#"
fn main() {
    let b4: [u8; 4] = [0x34, 0x12, 0x56, 0x78];
    let t: (u8, u16, u8) = unsafe { std::mem::transmute(b4) };
    println!("{} {} {}", t.0, t.1, t.2);
    let back: [u8; 4] = unsafe { std::mem::transmute(t) };
    println!("{:?}", back);
    let v: u64 = 0x1122_3344_5566_7788;
    let p: (u8, u32) = unsafe { std::mem::transmute(v) };
    println!("{} {}", p.0, p.1);
    let v2: u64 = unsafe { std::mem::transmute(p) };
    println!("{:x}", v2);
    let nb: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let n: ((u8, u32), u8) = unsafe { std::mem::transmute(nb) };
    println!("{} {} {}", (n.0).0, (n.0).1, n.1);
}
"#;

#[test]
fn run_mode_transmute_tuple_at_recorded_offsets() {
    // Bare tuples transmuted to and from byte arrays and
    // integers at their recorded layout offsets: a
    // reordered tuple (`(u8, u16, u8)` stores its `u16`
    // first), a padded tuple (`(u8, u32)` spans eight
    // bytes, its pattern the full `u64` width in both
    // directions), and a nested tuple read field by field.
    run_ok(
        "transmute-tuple-layout",
        TRANSMUTE_TUPLE_LAYOUT_SRC,
        "86 4660 120\n[52, 18, 86, 120]\n136 287454020\n1122334400000088\n1 134678021 9\n",
    );
}

const TRANSMUTE_ENUM_TUPLE_PAYLOAD_SRC: &str = r#"
#[repr(u8)]
enum E {
    A((u8, u16, u8)),
    B(u32),
}

fn main() {
    let e = E::A((0x56, 0x1234, 0x78));
    let bytes: [u8; 8] = unsafe { std::mem::transmute(e) };
    let back: E = unsafe { std::mem::transmute(bytes) };
    match back {
        E::A(t) => println!("A {} {} {}", t.0, t.1, t.2),
        E::B(v) => println!("B {}", v),
    }
    let e2 = E::B(0xdeadbeef);
    let bytes2: [u8; 8] = unsafe { std::mem::transmute(e2) };
    let back2: E = unsafe { std::mem::transmute(bytes2) };
    match back2 {
        E::A(t) => println!("A {} {} {}", t.0, t.1, t.2),
        E::B(v) => println!("B {:x}", v),
    }
}
"#;

#[test]
fn run_mode_transmute_enum_tuple_payload() {
    // A direct-tag enum whose one variant carries a
    // tuple: the tuple payload is rebuilt at its recorded
    // layout offsets when the enum is reconstructed from
    // its bytes, for both variants.
    run_ok(
        "transmute-enum-tuple-payload",
        TRANSMUTE_ENUM_TUPLE_PAYLOAD_SRC,
        "A 86 4660 120\nB deadbeef\n",
    );
}

const TRANSMUTE_NICHE_WRAP_SRC: &str = r#"
#[repr(u8)]
#[derive(Clone, Copy)]
enum E { A = 253, B = 254 }
#[repr(C)]
#[derive(Clone, Copy)]
struct W { a: E, b: u8 }
#[derive(Clone, Copy)]
enum N { Payload(W), Empty1, Empty2 }
fn code(n: N) -> u32 {
    match n {
        N::Payload(w) => 1000 + (w.a as u32) + ((w.b as u32) << 8),
        N::Empty1 => 1,
        N::Empty2 => 2,
    }
}
fn main() {
    let a: N = unsafe { std::mem::transmute(0u16) };
    let b: N = unsafe { std::mem::transmute(255u16) };
    let c: N = unsafe { std::mem::transmute(253u16) };
    let d: N = unsafe { std::mem::transmute(0x5AFEu16) };
    println!("{} {} {} {}", code(a), code(b), code(c), code(d));
    let e2: u16 = unsafe { std::mem::transmute(N::Empty2) };
    let e1: u16 = unsafe { std::mem::transmute(N::Empty1) };
    let p: u16 = unsafe { std::mem::transmute(N::Payload(W { a: E::B, b: 0x5A })) };
    println!("{} {} {}", e2, e1, p);
}
"#;

#[test]
fn run_mode_transmute_niche_range_wraps_at_scalar_width() {
    // A niche range that starts at the niche scalar's
    // maximum (255 in a `u8` tag) wraps: the second niche
    // variant encodes as 0, in both directions. Decoding
    // measures the distance from the range's start at the
    // scalar's width, and encoding masks to that width
    // before shifting instead of spilling into the
    // adjacent payload byte.
    run_ok(
        "transmute-niche-wrap",
        TRANSMUTE_NICHE_WRAP_SRC,
        "2 1 1253 24294\n0 255 23294\n",
    );
}

const TRANSMUTE_NICHE_WRAP_SHIFTED_SRC: &str = r#"
#[repr(u8)]
#[derive(Clone, Copy)]
enum E { A = 253, B = 254 }
#[repr(C)]
#[derive(Clone, Copy)]
struct W { b: u8, a: E }
#[derive(Clone, Copy)]
enum N { Payload(W), Empty1, Empty2 }
fn code(n: N) -> u32 {
    match n {
        N::Payload(w) => 1000 + (w.a as u32) + (w.b as u32),
        N::Empty1 => 1,
        N::Empty2 => 2,
    }
}
fn main() {
    let a: N = unsafe { std::mem::transmute(0x0000u16) };
    let b: N = unsafe { std::mem::transmute(0xFF00u16) };
    let c: N = unsafe { std::mem::transmute(0xFE5Au16) };
    println!("{} {} {}", code(a), code(b), code(c));
    let e2: u16 = unsafe { std::mem::transmute(N::Empty2) };
    let p: u16 = unsafe { std::mem::transmute(N::Payload(W { b: 0x5A, a: E::B })) };
    println!("{} {}", e2, p);
}
"#;

#[test]
fn run_mode_transmute_niche_range_wraps_at_shifted_scalar() {
    // The same wrapping niche range with the niche scalar
    // in the payload's second byte: the wrapped value is
    // masked to the scalar's width before it is shifted
    // into its window, in both directions.
    run_ok(
        "transmute-niche-wrap-shifted",
        TRANSMUTE_NICHE_WRAP_SHIFTED_SRC,
        "2 1 1344\n0 65114\n",
    );
}
// ----- Function code bytes (fn pointer read as *const u8) -----

const FN_CODE_SRC: &str = r#"
fn alpha() -> u32 { 3 }
fn beta() -> u32 { 3 }
fn gamma() -> u32 { 4 }
fn delta() -> u32 { alpha() }

fn code_bytes(f: fn() -> u32) -> [u8; 8] {
    let base = f as usize;
    let mut out = [0u8; 8];
    for i in 0..8 {
        out[i] = unsafe { *((base + i) as *const u8) };
    }
    out
}

fn code_eq(f: fn() -> u32, g: fn() -> u32) -> bool {
    let a = code_bytes(f);
    let b = code_bytes(g);
    let mut i = 0;
    while i < 8 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

fn main() {
    // Identical bodies share a code image; a different
    // constant or a call body does not.
    assert!(code_eq(alpha, beta));
    assert!(!code_eq(alpha, gamma));
    assert!(!code_eq(alpha, delta));
    // The direct first-byte deref (the thin-lto-inlines
    // shape) and a wider scalar read through the same
    // pointer.
    let a = alpha as usize as *const u8;
    let b = beta as usize as *const u8;
    assert_eq!(unsafe { *a }, unsafe { *b });
    let w = alpha as usize as *const u32;
    assert_eq!(unsafe { *w }, unsafe { *(beta as usize as *const u32) });
    println!("fn-code-ok");
}
"#;

#[test]
fn run_mode_fn_pointer_code_bytes() {
    // A function pointer cast through an integer to a data
    // pointer dereferences to the function's code image: a
    // deterministic byte stream keyed by the body's canonical
    // form, so two functions with identical bodies read the
    // same bytes (the property tests/ui/lto/thin-lto-inlines
    // asserts of ThinLTO inlining) and different bodies read
    // different bytes.
    run_ok("fn-code", FN_CODE_SRC, "fn-code-ok\n");
}


const INT_PTR_FORM_SRC: &str = r#"
fn main() {
    let lsz: usize = 0x8090a0b0c0d0e0f0;
    let p: *const [u8; 1] = lsz as *const [u8; 1];
    // Forming a reference through a dereference of an
    // integer-derived pointer: the pointee has no storage in
    // the model, but the formed pointers are pure addresses --
    // they unsized-cast, thin-cast and compare as addresses
    // (the shape tests/ui/cast/cast-rfc0401.rs exercises).
    let u: *const [u8] = unsafe { &*p };
    assert_eq!(u as *const u8, p as *const u8);
    assert_eq!(u.len(), 1);
    assert_eq!(p as usize, lsz);
    println!("int-ptr-form-ok");
}
"#;

#[test]
fn run_mode_reference_formation_through_integer_pointer() {
    run_ok("int-ptr-form", INT_PTR_FORM_SRC, "int-ptr-form-ok\n");
}

const ADDROF_NULL_SRC: &str = r#"
struct FooBar { x: u32, y: u32 }
fn main() {
    // The address of a field of a null pointer's pointee: the
    // result is never read, it is the field's displaced null
    // address (the shape tests/ui/mir/null/addrof_null.rs
    // exercises).
    let p: *const FooBar = std::ptr::null();
    let q = unsafe { &raw const (*p).x };
    assert_eq!(q as usize, 0);
    let r = unsafe { &raw const (*p).y };
    assert_eq!(r as usize, 4);
    println!("addrof-null-ok");
}
"#;

#[test]
fn run_mode_address_of_field_through_null_pointer() {
    run_ok("addrof-null", ADDROF_NULL_SRC, "addrof-null-ok\n");
}

const ZST_NULL_OPS_SRC: &str = r#"
fn main() {
    // Zero-sized element operations through null pointers
    // move no bytes: the copies inside ptr::swap of a
    // zero-sized type, a zero-sized copy_nonoverlapping with
    // a nonzero count, and a volatile read of a zero-sized
    // pointee (the shapes tests/ui/precondition-checks/
    // zero-size-null.rs exercises under debug assertions).
    unsafe {
        let null: *mut () = std::ptr::null_mut();
        std::ptr::copy_nonoverlapping(null as *const (), null, 123);
        std::ptr::swap::<()>(null, null);
        let v = std::ptr::read_volatile::<()>(null as *const ());
        assert_eq!(v, ());
        std::ptr::write_volatile::<()>(null, ());
    }
    println!("zst-null-ops-ok");
}
"#;

#[test]
fn run_mode_zero_sized_pointer_ops_through_null() {
    let dir = temp_dir("zst-null-ops");
    let src = fixture(&dir, "zst_null_ops.rs", ZST_NULL_OPS_SRC);
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "-Zmir-opt-level=0",
            "--rustc-arg",
            "-Copt-level=0",
            "--rustc-arg",
            "-Cdebug-assertions=yes",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "zero-sized pointer ops through null must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "zst-null-ops-ok\n");
}


const FNPTR_CAST_SRC: &str = r#"
fn callee(x: u32) -> u32 { x + 41 }
fn other(x: u32) -> u32 { x + 1 }

fn main() {
    // Function item and function pointer to raw pointer
    // (FnPtrToPtr, the fptr section of
    // tests/ui/transmute/cast-rfc0401.rs / cast-rfc0401).
    let fp: fn(u32) -> u32 = callee;
    let p = fp as *const ();
    let q = callee as *const ();
    assert_eq!(p, q);
    assert_eq!(p as usize, fp as usize);
    assert!(!p.is_null());
    let r = fp as *mut u8;
    assert_eq!(r as usize, p as usize);
    let o = other as *const ();
    assert_ne!(p, o);
    // A safe fn pointer viewed as an unsafe fn pointer still
    // calls the same function.
    let ufp = fp as unsafe fn(u32) -> u32;
    assert_eq!(unsafe { ufp(1) }, 42);
    // The raw pointer transmutes back to a callable fn pointer.
    let back: fn(u32) -> u32 = unsafe { std::mem::transmute(p) };
    assert_eq!(back(1), 42);
    assert_eq!(back as usize, p as usize);
    // A direct read through the cast pointer reads the same
    // code byte as the integer round trip.
    let direct = unsafe { *(fp as *const u8) };
    let via_int = unsafe { *((fp as usize) as *const u8) };
    assert_eq!(direct, via_int);
    println!("fnptr-cast-ok");
}
"#;

#[test]
fn run_mode_fnptr_to_raw_ptr_cast() {
    run_ok("fnptr-cast", FNPTR_CAST_SRC, "fnptr-cast-ok\n");
}

// ----- cast-rfc0401, the whole record in one program -----

/// The upstream `tests/ui/cast/cast-rfc0401.rs` record verbatim
/// (directive line dropped). Its segments were verified in
/// isolation before — the slice regroup segment by
/// `driver_transmute_regroup.rs`, the fn-pointer segment by
/// `run_mode_fnptr_to_raw_ptr_cast` above — but the whole
/// record still failed at the regroup assertion on a tree
/// carrying only the fn-pointer cast support: without the
/// equality side's regroup rebuild, `assert_eq!(p, ...)`
/// compared the pointee's stored `[[u8; 3]; 2]` shape against
/// the `[[u8; 2]]` expectation. This pins the composition:
/// every cast section in sequence, in one `main`, must run
/// to completion with no output, as natively.
const CAST_RFC0401_SRC: &str = r##"
#![allow(dead_code, function_casts_as_integer)]

use std::vec;

enum Simple {
    A,
    B,
    C
}

enum Valued {
    H8=163,
    Z=0,
    X=256,
    H7=67,
}

enum ValuedSigned {
    M1=-1,
    P1=1
}

fn main()
{
    // coercion-cast
    let mut it = vec![137].into_iter();
    let itr: &mut vec::IntoIter<u32> = &mut it;
    assert_eq!((itr as &mut dyn Iterator<Item=u32>).next(), Some(137));
    assert_eq!((itr as &mut dyn Iterator<Item=u32>).next(), None);

    assert_eq!(Some(4u32) as Option<u32>, Some(4u32));
    assert_eq!((1u32,2u32) as (u32,u32), (1,2));

    // this isn't prim-int-cast. Check that it works.
    assert_eq!(false as bool, false);
    assert_eq!(true as bool, true);

    // numeric-cast
    let l: u64 = 0x8090a0b0c0d0e0f0;
    let lsz: usize = l as usize;
    assert_eq!(l as u32, 0xc0d0e0f0);

    // numeric-cast
    assert_eq!(l as u8, 0xf0);
    assert_eq!(l as i8,-0x10);
    assert_eq!(l as u32, 0xc0d0e0f0);
    assert_eq!(l as u32 as usize as u32, l as u32);
    assert_eq!(l as i32,-0x3f2f1f10);
    assert_eq!(l as i32 as isize as i32, l as i32);
    assert_eq!(l as i64,-0x7f6f5f4f3f2f1f10);

    assert_eq!(0 as f64, 0f64);
    assert_eq!(1 as f64, 1f64);

    assert_eq!(l as f64, 9264081114510712022f64);

    assert_eq!(l as i64 as f64, -9182662959198838444f64);
//  float overflow : needs fixing
//  assert_eq!(l as f32 as i64 as u64, 9264082620822882088u64);
//  assert_eq!(l as i64 as f32 as i64, 9182664080220408446i64);

    assert_eq!(4294967040f32 as u32, 0xffffff00u32);
    assert_eq!(1.844674407370955e19f64 as u64, 0xfffffffffffff800u64);

    assert_eq!(9.223372036854775e18f64 as i64, 0x7ffffffffffffc00i64);
    assert_eq!(-9.223372036854776e18f64 as i64, 0x8000000000000000u64 as i64);

    // addr-ptr-cast/ptr-addr-cast (thin ptr)
    let p: *const [u8; 1] = lsz as *const [u8; 1];
    assert_eq!(p as usize, lsz);

    // ptr-ptr-cast (thin ptr)
    let w: *const () = p as *const ();
    assert_eq!(w as usize, lsz);

    // ptr-ptr-cast (fat->thin)
    let u: *const [u8] = unsafe{&*p};
    assert_eq!(u as *const u8, p as *const u8);
    assert_eq!(u as *const u16, p as *const u16);

    // ptr-ptr-cast (Length vtables)
    let mut l : [u16; 2] = [0,1];
    let w: *mut [u8; 2] = &mut l as *mut [u16; 2] as *mut _;
    let w: *mut [u8] = unsafe {&mut *w};
    let w_u16 : *const [u16] = w as *const [u16];
    assert_eq!(unsafe{&*w_u16}, &l);

    let s: *mut str = w as *mut str;
    let l_via_str = unsafe{&*(s as *const [u16])};
    assert_eq!(&l, l_via_str);

    // ptr-ptr-cast (Length vtables, check length is preserved)
    let l: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    let p: *const [[u8; 3]] = &l;
    let p: &[[u8; 2]] = unsafe {&*(p as *const [[u8; 2]])};
    assert_eq!(p, [[3, 2], [6, 4]]);

    // enum-cast
    assert_eq!(Simple::A as u8, 0);
    assert_eq!(Simple::B as u8, 1);

    assert_eq!(Valued::H8 as i8, -93);
    assert_eq!(Valued::H7 as i8, 67);
    assert_eq!(Valued::Z as i8, 0);

    assert_eq!(Valued::H8 as u8, 163);
    assert_eq!(Valued::H7 as u8, 67);
    assert_eq!(Valued::Z as u8, 0);

    assert_eq!(Valued::H8 as u16, 163);
    assert_eq!(Valued::Z as u16, 0);
    assert_eq!(Valued::H8 as u16, 163);
    assert_eq!(Valued::Z as u16, 0);

    assert_eq!(ValuedSigned::M1 as u16, 65535);
    assert_eq!(ValuedSigned::M1 as i16, -1);
    assert_eq!(ValuedSigned::P1 as u16, 1);
    assert_eq!(ValuedSigned::P1 as i16, 1);

    // prim-int-cast
    assert_eq!(false as u16, 0);
    assert_eq!(true as u16, 1);
    assert_eq!(false as i64, 0);
    assert_eq!(true as i64, 1);
    assert_eq!('a' as u32, 0x61);
    assert_eq!('a' as u16, 0x61);
    assert_eq!('a' as u8, 0x61);
    assert_eq!('א' as u8, 0xd0);
    assert_eq!('א' as u16, 0x5d0);
    assert_eq!('א' as u32, 0x5d0);
    assert_eq!('🐵' as u8, 0x35);
    assert_eq!('🐵' as u16, 0xf435);
    assert_eq!('🐵' as u32, 0x1f435);
    assert_eq!('英' as i16, -0x7d0f);
    assert_eq!('英' as u16, 0x82f1);

    // u8-char-cast
    assert_eq!(0x61 as char, 'a');
    assert_eq!(0u8 as char, '\0');
    assert_eq!(0xd7 as char, '×');

    // array-ptr-cast
    let x = [1,2,3];
    let first : *const u32 = &x[0];

    assert_eq!(first, &x as *const _);
    assert_eq!(first, &x as *const u32);

    // fptr-addr-cast
    fn foo() {
        println!("foo!");
    }
    fn bar() {
        println!("bar!");
    }

    assert!(foo as usize != bar as usize);

    // Taking a few bits of a function's address is totally pointless and we detect that
    assert_eq!(foo as i16, foo as usize as i16);

    // fptr-ptr-cast

    assert_eq!(foo as *const u8 as usize, foo as usize);
    assert!(foo as *const u32 != first);
}
fn foo() { }
"##;

#[test]
fn run_mode_cast_rfc0401_full_record() {
    run_ok("cast-rfc0401-full", CAST_RFC0401_SRC, "");
}
