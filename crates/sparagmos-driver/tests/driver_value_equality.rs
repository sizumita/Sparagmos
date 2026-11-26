//! Run-mode regression tests for value equality and the
//! return boundary where a value's representation in the
//! model legitimately differs from the declared type of the
//! place it is compared against or stored into (the
//! "type mismatch in interp conversion" family, Issue #513).
//!
//! * A `dyn Trait` value behind an `Rc` / `Arc` handle: the
//!   handle aggregate is the pointer chain plus metadata, and
//!   the pointee the equality must compare lives in the shared
//!   allocation's `RcInner` / `ArcInner` aggregate. Comparing
//!   the handle itself against the sized side cannot decide.
//! * An `OsString` against a `str`: the string's bytes sit
//!   behind two wrapper newtypes (`OsString` is a `Buf`
//!   around the `Vec<u8>`), so the element-form string
//!   extraction must descend through the nested single-field
//!   wrappers to the element list.
//! * A `Vec<u8>` in its element form (the shape a place
//!   type-ascription assignment leaves) against a byte-array
//!   literal: the vector value's fields ARE the elements, so
//!   the field list itself is the byte sequence.
//! * A fieldless (C-like) enum returned through a transmuted
//!   function pointer into an integer-typed destination: the
//!   enum is represented as its discriminant integer, and the
//!   value must cross the return boundary as that integer.
//! * A `Vec` (or slice) compared against an array at
//!   `-Copt-level=3`, where core's slice equality lowers to
//!   the `compare_bytes` intrinsic over byte-denominated
//!   pointers: the byte range over a wider-element sequence
//!   must be read element by element at the element width,
//!   not element-per-byte (which walks into the buffer
//!   block's uninitialised spare-capacity slots, or past the
//!   end of a constant array).
//! * A raw pointer compared through its own `PartialEq` /
//!   `PartialOrd` / `Ord` impls (including the mixed
//!   `*const T == &T` form): the native impls compare
//!   addresses, so the comparison builtins must price both
//!   operands as addresses instead of resolving them to
//!   their pointees -- resolving also recurses without
//!   bound on a self-referential pointee (a stored pointer
//!   denoting the very aggregate that holds it).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits.
/// Tests spawn the driver per case and used to leave every
/// fixture directory behind (#537); the registry keeps the
/// per-test `temp_dir` shape and sweeps at exit instead.
static SCRATCH_DIRS: std::sync::Mutex<Vec<std::path::PathBuf>> =
    std::sync::Mutex::new(Vec::new());

fn register_scratch_dir(dir: &std::path::Path) {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| unsafe {
        let _ = libc::atexit(remove_scratch_dirs);
    });
    if let Ok(mut dirs) = SCRATCH_DIRS.lock() {
        if !dirs.iter().any(|d| d == dir) {
            dirs.push(dir.to_path_buf());
        }
    }
}

extern "C" fn remove_scratch_dirs() {
    if let Ok(dirs) = SCRATCH_DIRS.try_lock() {
        for dir in dirs.iter() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn fixture(dir: &Path, name: &str, src: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    path
}

fn run_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-valeq-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn assert_runs(tag: &str, src: &str, marker: &str) {
    let dir = temp_dir(tag);
    let file = fixture(&dir, "main.rs", src);
    let out = run_driver(&file);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains(marker),
        "driver run failed for {tag}: status {:?}, stdout: {stdout}, stderr: {stderr}",
        out.status
    );
}

fn run_driver_extra(src: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .args(extra)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn assert_runs_o3(tag: &str, src: &str, marker: &str) {
    let dir = temp_dir(tag);
    let file = fixture(&dir, "main.rs", src);
    let out = run_driver_extra(&file, &["--rustc-arg", "-Copt-level=3"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains(marker),
        "driver run failed for {tag}: status {:?}, stdout: {stdout}, stderr: {stderr}",
        out.status
    );
}

#[test]
fn rc_arc_dyn_pointee_equality() {
    assert_runs(
        "rc-arc",
        r#"
use std::rc::Rc;
use std::sync::Arc;
trait Foo: PartialEq<i32> + std::fmt::Debug {
    fn a(&self) -> i32 { 10 }
}
impl Foo for i32 {
    fn a(&self) -> i32 { 100 }
}
fn main() {
    let b: Rc<dyn Foo> = Rc::new(1i32);
    assert_eq!(*b, 1);
    assert_ne!(*b, 2);
    assert_eq!(b.a(), 100);
    let c: Arc<dyn Foo> = Arc::new(7i32);
    assert_eq!(*c, 7);
    assert_ne!(*c, 8);
    println!("rc-arc-dyn-eq ok");
}
"#,
        "rc-arc-dyn-eq ok",
    );
}

#[test]
fn osstring_str_equality() {
    assert_runs(
        "osstring",
        r#"
use std::ffi::OsString;
fn main() {
    let os_str = OsString::from("Hello Rust!");
    assert_eq!(os_str, "Hello Rust!");
    assert_eq!("Hello Rust!", os_str);
    assert_ne!(os_str, "Hello Moon!");
    println!("osstring-eq ok");
}
"#,
        "osstring-eq ok",
    );
}

#[test]
fn element_form_vec_against_byte_array() {
    assert_runs(
        "ascribe",
        r#"
#![feature(type_ascription)]
fn main() {
    let mut v = Vec::new();
    type_ascribe!(v, Vec<u8>) = vec![1, 2, 3];
    assert_eq!(v, [1u8, 2, 3]);
    assert_ne!(v, [1u8, 2, 4]);
    println!("vec-bytes-eq ok");
}
"#,
        "vec-bytes-eq ok",
    );
}

#[test]
fn fieldless_enum_return_as_int() {
    assert_runs(
        "clike",
        r#"
#[repr(u32)]
enum Foo {
    A = 0,
    B = 23,
}
#[inline(never)]
extern "C" fn foo(_x: usize) -> Foo { Foo::B }
#[repr(i32)]
enum Bar {
    N = -5,
    P = 3,
}
#[inline(never)]
extern "C" fn bar(_x: usize) -> Bar { Bar::N }
fn main() {
    unsafe {
        let f: extern "C" fn(usize) -> u32 =
            std::mem::transmute(foo as extern "C" fn(usize) -> Foo);
        assert_eq!(f(0xDEADBEEF), 23);
        let g: extern "C" fn(usize) -> i32 =
            std::mem::transmute(bar as extern "C" fn(usize) -> Bar);
        assert_eq!(g(0), -5);
    }
    println!("clike-return ok");
}
"#,
        "clike-return ok",
    );
}

#[test]
fn vec_array_equality_o3() {
    assert_runs_o3(
        "vec-array-o3",
        r#"
fn main() {
    // The reported shape: spare capacity in the buffer
    // block must not be read by the slice-equality path.
    let mut v: Vec<i32> = Vec::with_capacity(8);
    v.push(1);
    v.push(2);
    assert!(v == [1, 2]);
    assert!(v != [1, 3]);
    assert!(v != [1, 2, 3]);
    // An exactly-sized buffer has no spare slots; the
    // over-read past its end is the same stop.
    assert!(vec![1, 2] == [1, 2]);
    assert!(vec![1, 2] != [2, 1]);
    // Neighbouring comparison shapes over the same path.
    assert!(vec![1, 2] == vec![1, 2]);
    assert!(&[1, 2] == &[1, 2]);
    assert!(&[1, 2] != &[1, 3]);
    // A difference above the low byte must still register:
    // the range is the elements' bytes, not their low bytes.
    assert!(vec![256i32] != [0i32]);
    assert!(vec![256i32] == [256i32]);
    assert!(vec![-1i32, 2] == [-1, 2]);
    assert!(vec![-1i32] != [255i32]);
    println!("vec-array-eq-o3 ok");
}
"#,
        "vec-array-eq-o3 ok",
    );
}

#[test]
fn vec_array_equality_element_widths_o3() {
    assert_runs_o3(
        "vec-array-widths-o3",
        r#"
fn main() {
    assert!(vec![1000u16, 2] == [1000, 2]);
    assert!(vec![1000u16] != [1001u16]);
    assert!(vec![0x0102_0304_0506_0708u64, 9] == [0x0102_0304_0506_0708, 9]);
    assert!(vec![1.5f32, -2.25] == [1.5, -2.25]);
    assert!(vec![1.5f32] != [1.25f32]);
    assert!(vec![true, false] == [true, false]);
    assert!(vec![true] != [false]);
    assert!(vec![-1i8, 2] == [-1, 2]);
    assert!(vec!['a', 'b'] == ['a', 'b']);
    assert!(vec![0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10u128]
        == [0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10u128]);
    println!("vec-array-widths-o3 ok");
}
"#,
        "vec-array-widths-o3 ok",
    );
}

#[test]
fn slice_cmp_memcmp_order_o3() {
    assert_runs_o3(
        "slice-cmp-o3",
        r#"
use std::cmp::Ordering;
fn main() {
    // Slice ordering runs over the same intrinsic (core
    // maps its result back to the elements' numeric order,
    // so 256 orders after 255 despite its low byte being 0).
    assert_eq!(vec![2i32].as_slice().cmp(&[1i32]), Ordering::Greater);
    assert_eq!(vec![256i32].as_slice().cmp(&[255i32]), Ordering::Greater);
    assert_eq!(vec![255i32].as_slice().cmp(&[256i32]), Ordering::Less);
    assert_eq!(vec![1u64, 9].as_slice().cmp(&[1u64, 8]), Ordering::Greater);
    assert_eq!(vec![1u8, 2].as_slice().cmp(&[1u8, 3]), Ordering::Less);
    assert_eq!(vec![1i32, 2].as_slice().cmp(&[1i32, 2]), Ordering::Equal);
    println!("slice-cmp-o3 ok");
}
"#,
        "slice-cmp-o3 ok",
    );
}

#[test]
fn mixed_raw_pointer_ref_equality_compares_addresses() {
    assert_runs(
        "mixed-ptr-eq",
        r#"
struct P { x: u32 }
fn main() {
    let a = P { x: 5 };
    let b = P { x: 5 };
    let pa = &a as *const P;
    let pb = &b as *const P;
    // Same-type binop control (lowers to BinOp::Eq).
    assert!(pa != pb);
    assert!(pa == pa);
    // Mixed raw pointer / reference operands reach the
    // `PartialEq` trait call; the pointees are equal, so a
    // pointee comparison would wrongly report equality.
    assert!(pa != &b);
    assert!(pa == &a);
    assert!(!(pa == &b));
    // The explicit trait-call form of the same comparison.
    assert!(!std::cmp::PartialEq::eq(&pa, &pb));
    assert!(std::cmp::PartialEq::eq(&pa, &pa));
    assert!(std::cmp::PartialEq::ne(&pa, &pb));
    println!("mixed-ptr-eq ok");
}
"#,
        "mixed-ptr-eq ok",
    );
}

#[test]
fn self_referential_pointer_equality_terminates() {
    assert_runs(
        "selfref-ptr-eq",
        r#"
use std::cell::Cell;
struct Foo<'a> { p: Cell<Option<*const Foo<'a>>> }
fn main() {
    // The tls-dont-move-after-init shape: the stored
    // pointer denotes the very aggregate that holds it, so
    // a pointee comparison recurses without bound.
    let f = Foo { p: Cell::new(None) };
    f.p.set(Some(&f as *const Foo));
    let q = f.p.get().unwrap();
    assert!(q == &f);
    assert!(std::cmp::PartialEq::eq(&q, &(&f as *const Foo)));
    println!("selfref-ptr-eq ok");
}
"#,
        "selfref-ptr-eq ok",
    );
}

#[test]
fn raw_pointer_trait_ordering_compares_addresses() {
    assert_runs(
        "ptr-trait-ord",
        r#"
use std::cmp::Ordering;
struct P { x: u32 }
fn main() {
    let arr = [P { x: 30 }, P { x: 10 }, P { x: 20 }];
    let p0 = &arr[0] as *const P;
    let p1 = &arr[1] as *const P;
    // Address order: p0 < p1. Pointee-value order: p1 < p0.
    assert_eq!(std::cmp::Ord::cmp(&p0, &p1), Ordering::Less);
    assert_eq!(std::cmp::Ord::cmp(&p1, &p0), Ordering::Greater);
    assert_eq!(std::cmp::Ord::cmp(&p0, &p0), Ordering::Equal);
    assert_eq!(std::cmp::PartialOrd::partial_cmp(&p0, &p1), Some(Ordering::Less));
    assert!(!std::cmp::PartialOrd::lt(&p1, &p0));
    assert!(std::cmp::PartialOrd::gt(&p1, &p0));
    assert!(std::cmp::PartialOrd::le(&p0, &p0));
    println!("ptr-trait-ord ok");
}
"#,
        "ptr-trait-ord ok",
    );
}
