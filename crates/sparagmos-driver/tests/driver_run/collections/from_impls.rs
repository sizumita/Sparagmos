// From / Into conversion tests.


const RC_ARC_FROM_POINTEE_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;
#[derive(Debug, PartialEq)]
struct W(usize);
fn main() {
    let r: Rc<W> = Rc::from(W(3));
    println!("{:?} {}", r, r.0);
    let a: Arc<W> = Arc::from(W(4));
    println!("{}", a.0);
    let rv: Rc<Vec<i32>> = Rc::from(vec![1, 2]);
    println!("{:?} {}", rv, rv.len());
}
"#;

#[test]
fn run_mode_rc_arc_from_pointee_value() {
    // `Rc<T>: From<T>` / `Arc<T>: From<T>` is `Rc::new` /
    // `Arc::new` over the value itself -- including when
    // the value is a `Vec`, whose pointee is the vector,
    // not its elements (the counted-slice conversion must
    // not claim it).
    let dir = temp_dir("rc-arc-from-pointee");
    let src = fixture(&dir, "rcarcfrom.rs", RC_ARC_FROM_POINTEE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "W(3) 3\n4\n[1, 2] 2\n", "stdout={stdout}");
}

const PIN_FROM_BOX_SRC: &str = r#"
use std::pin::Pin;
fn main() {
    let p: Pin<Box<i64>> = Pin::from(Box::new(4i64));
    println!("{}", *p);
    let q: Pin<Box<i64>> = Box::new(5i64).into();
    println!("{}", *q);
}
"#;

#[test]
fn run_mode_pin_from_box() {
    // `Pin<P>: From<P>` is the one-field pin wrapper
    // around the pointer value, the shape `Pin::new`
    // builds and the `Deref` model unwraps.
    let dir = temp_dir("pin-from-box");
    let src = fixture(&dir, "pinfrom.rs", PIN_FROM_BOX_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "4\n5\n", "stdout={stdout}");
}

const NESTED_PIN_DEREF_SRC: &str = r#"
#![feature(arbitrary_self_types)]
#![feature(rustc_attrs)]
use std::pin::Pin;
trait Trait {
    fn by_pin2(self: Pin<Pin<&Self>>) -> i64;
    fn by_pin3(self: Pin<Pin<Pin<&Self>>>) -> i64;
    fn by_pin_box(self: Pin<Box<Self>>) -> i64;
    fn bump(self: Pin<Pin<&mut Self>>);
}
impl Trait for i64 {
    fn by_pin2(self: Pin<Pin<&Self>>) -> i64 { *self }
    fn by_pin3(self: Pin<Pin<Pin<&Self>>>) -> i64 { *self }
    fn by_pin_box(self: Pin<Box<Self>>) -> i64 { *self }
    fn bump(mut self: Pin<Pin<&mut Self>>) { *self += 10; }
}
fn main() {
    let v = 5i64;
    let p2 = Pin::new(Pin::new(&v)) as Pin<Pin<&dyn Trait>>;
    println!("{}", p2.by_pin2());
    let p3 = Pin::new(Pin::new(Pin::new(&v))) as Pin<Pin<Pin<&dyn Trait>>>;
    println!("{}", p3.by_pin3());
    let pb = Pin::new(Box::new(4i64)) as Pin<Box<dyn Trait>>;
    println!("{}", pb.by_pin_box());
    let mut w = 1i64;
    let pm = Pin::new(Pin::new(&mut w)) as Pin<Pin<&mut dyn Trait>>;
    pm.bump();
    println!("{}", w);
}
"#;

#[test]
fn run_mode_nested_pin_deref() {
    // A `Pin` around another `Pin` derefs through both
    // wrappers: `Pin<P>`'s target is `P`'s own target, so
    // the `Deref` / `DerefMut` models unwrap the wrapper
    // chain before the pointer rule applies. A single
    // unwrap left a reference to the inner wrapper, which
    // the method body's return assignment rejected.
    let dir = temp_dir("nested-pin-deref");
    let src = fixture(&dir, "nestedpin.rs", NESTED_PIN_DEREF_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "5\n5\n4\n11\n", "stdout={stdout}");
}

const NONNULL_FROM_MUT_SRC: &str = r#"
use std::ptr::NonNull;
fn main() {
    let mut a = [0u8; 5];
    let b: Option<NonNull<[u8]>> = Some(NonNull::from(&mut a));
    match b {
        Some(_) => println!("Got `Some`"),
        None => panic!("Unexpected `None`"),
    }
}
"#;

#[test]
fn run_mode_nonnull_from_mut_slice_ref() {
    // `NonNull<T>: From<&mut T>` is the one-field wrapper
    // around the reference's own pointer value; the
    // niche-encoded `Option` reads it as non-null.
    let dir = temp_dir("nonnull-from-mut");
    let src = fixture(&dir, "nonnullfrom.rs", NONNULL_FROM_MUT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "Got `Some`\n", "stdout={stdout}");
}

const GUEST_STRUCT_FROM_SRC: &str = r#"
use std::marker::PhantomData;
struct TupleSimple(u32);
impl From<u32> for TupleSimple { fn from(v: u32) -> Self { TupleSimple(v) } }
struct WithMarker<T> { t: T, pd: PhantomData<T> }
impl<T> From<T> for WithMarker<T> { fn from(t: T) -> Self { WithMarker { t, pd: PhantomData } } }
struct Ref3<'a, T>(&'a T);
impl<'a, T> From<&'a T> for Ref3<'a, T> { fn from(r: &'a T) -> Self { Ref3(r) } }
fn main() {
    let s: TupleSimple = 42u32.into();
    println!("{}", s.0);
    let w: WithMarker<u32> = 7u32.into();
    println!("{}", w.t);
    let a = 9u32;
    let r: Ref3<u32> = (&a).into();
    println!("{}", r.0);
}
"#;

#[test]
fn run_mode_guest_struct_from_wrap() {
    // A guest struct whose `From` impl is the struct
    // construction itself (`#[derive(From)]` and the
    // trivial manual impls): the value fills the one
    // real field; a `PhantomData` field takes the empty
    // aggregate. The generic cases matter: the recorded
    // shape renders a generic field's type with the
    // parameter's name, so the field is identified
    // structurally.
    let dir = temp_dir("guest-struct-from");
    let src = fixture(&dir, "guestfrom.rs", GUEST_STRUCT_FROM_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "42\n7\n9\n", "stdout={stdout}");
}
