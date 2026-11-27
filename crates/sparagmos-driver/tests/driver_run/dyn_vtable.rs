//! Run-mode regression tests for dynamic dispatch resolution:
//! a `dyn` call must reach the vtable instance's body however
//! that body entered the program and however its name is
//! spelled.
//!
//! * A trait *default* method that is also demanded through a
//!   static or generic call is converted by the instance-demand
//!   pass under the definition's plain name; the vtable demand
//!   pass must still register the same instance under its own
//!   name, which is the only spelling the interpreter's
//!   virtual-call lookup constructs.
//! * An instance key spells the method's own generic arguments
//!   as a trailing `::<...>` segment (`bar::<'_>` for a method
//!   with lifetime parameters); the impl-key scan must read
//!   the segment-free method name.
//! * A higher-ranked trait object's existential arguments
//!   carry the `dyn` binder's regions; the vtable demand must
//!   erase them before asking the compiler for the vtable
//!   (its predicate construction rejects escaping bound
//!   variables).
//! * An impl inside a method body for a foreign `Self` is
//!   recorded under the impl-block path form
//!   (`...::<impl Trait for Self>::method`); the scan must
//!   parse that form and match its `Self` section.
//! * A closure `Self` is spelled with the compiler's span
//!   form (`{closure@<file>:16:17: 16:23}`) in the impl
//!   instance key but with the definition path's indexed
//!   segment (`main::{closure#0}`) in the fat pointer's
//!   concrete tag; the impl scan must equate the two
//!   spellings (tests/ui/regions/issue-6157.rs).

use super::{fixture, run_driver, run_driver_with_aux, temp_dir, Output};

fn assert_ok(out: &Output) {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "run must succeed; stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("ok"),
        "guest prints ok; stdout={stdout} stderr={stderr}"
    );
}

const STATIC_AND_DYN_SRC: &str = r#"
trait Bar {
    fn m(&self) -> u32 { 41 }
    fn n(&self) -> u32;
}
struct Foo;
impl Bar for Foo {
    fn n(&self) -> u32 { 1 }
}
fn call_static(x: &Foo) -> u32 { x.m() + x.n() }
fn call_dyn(x: &dyn Bar) -> u32 { x.m() + x.n() }
fn main() {
    let f = Foo;
    assert_eq!(call_static(&f), 42);
    let d: Box<dyn Bar> = Box::new(Foo);
    assert_eq!(call_dyn(&*d), 42);
    println!("ok");
}
"#;

#[test]
fn dyn_default_method_also_called_statically() {
    let dir = temp_dir("dyn-static-dyn");
    let src = fixture(&dir, "main.rs", STATIC_AND_DYN_SRC);
    assert_ok(&run_driver(&src));
}

const GENERIC_AND_DYN_SRC: &str = r#"
trait Bar {
    fn m(&self) -> u32 { 41 }
    fn n(&self) -> u32;
}
struct Foo;
impl Bar for Foo {
    fn n(&self) -> u32 { 1 }
}
fn call_generic<T: Bar>(x: &T) -> u32 { x.m() + x.n() }
fn call_dyn(x: &dyn Bar) -> u32 { x.m() + x.n() }
fn main() {
    let f = Foo;
    assert_eq!(call_generic(&f), 42);
    let d: Box<dyn Bar> = Box::new(Foo);
    assert_eq!(call_dyn(&*d), 42);
    println!("ok");
}
"#;

#[test]
fn dyn_default_method_also_called_generically() {
    let dir = temp_dir("dyn-generic-dyn");
    let src = fixture(&dir, "main.rs", GENERIC_AND_DYN_SRC);
    assert_ok(&run_driver(&src));
}

const LIFETIME_METHOD_SRC: &str = r#"
trait Foo<'a> {
    fn bar<'b>(&self, x: &'b u8) -> u8 where 'a: 'b { *x + 7 }
}
pub struct FooBar;
impl Foo<'static> for FooBar {}
fn test(foobar: FooBar) -> Box<dyn Foo<'static>> {
    Box::new(foobar)
}
fn main() {
    assert_eq!(test(FooBar).bar(&4), 11);
    println!("ok");
}
"#;

#[test]
fn dyn_method_with_lifetime_parameters() {
    let dir = temp_dir("dyn-lifetime-method");
    let src = fixture(&dir, "main.rs", LIFETIME_METHOD_SRC);
    assert_ok(&run_driver(&src));
}

const HRTB_SRC: &str = r#"
trait FnLike<A, R> {
    fn call(&self, arg: A) -> R;
}
type FnObject<'b> = dyn for<'a> FnLike<&'a isize, &'a isize> + 'b;
struct Identity;
impl<'a, T> FnLike<&'a T, &'a T> for Identity {
    fn call(&self, arg: &'a T) -> &'a T { arg }
}
fn call_repeatedly(f: &FnObject) {
    let x = 3;
    let y = f.call(&x);
    assert_eq!(3, *y);
}
fn main() {
    call_repeatedly(&Identity);
    println!("ok");
}
"#;

#[test]
fn dyn_higher_ranked_trait_object() {
    let dir = temp_dir("dyn-hrtb");
    let src = fixture(&dir, "main.rs", HRTB_SRC);
    assert_ok(&run_driver(&src));
}

const FOREIGN_SELF_AUX_SRC: &str = r#"
pub struct Test;
"#;

const IMPL_IN_METHOD_SRC: &str = r#"
#![allow(non_local_definitions)]

pub trait Trait {
    fn foo(&self) -> u32 { 7 }
}

pub struct Foo;

impl Iterator for Foo {
    type Item = Box<dyn Trait>;
    fn next(&mut self) -> Option<Box<dyn Trait>> {
        extern crate foreign_self as issue_41053;
        impl crate::Trait for issue_41053::Test {}
        Some(Box::new(issue_41053::Test))
    }
}

fn main() {
    assert_eq!(Foo.next().unwrap().foo(), 7);
    println!("ok");
}
"#;

#[test]
fn dyn_impl_inside_method_body_foreign_self() {
    let dir = temp_dir("dyn-impl-in-method");
    let aux = fixture(&dir, "foreign_self.rs", FOREIGN_SELF_AUX_SRC);
    let src = fixture(&dir, "main.rs", IMPL_IN_METHOD_SRC);
    assert_ok(&run_driver_with_aux(&src, &[("foreign_self", &aux)]));
}

const CLOSURE_SELF_DYN_SRC: &str = r#"
pub trait OpInt { fn call(&mut self, _: isize, _: isize) -> isize; }

impl<F> OpInt for F where F: FnMut(isize, isize) -> isize {
    fn call(&mut self, a: isize, b: isize) -> isize { (*self)(a, b) }
}

fn apply<'a>(x: isize, op: &'a mut dyn OpInt) -> isize { op.call(x, x) }

fn muli(x: isize, y: isize) -> isize { x * y }

fn main() {
    let mut f = |x, y| muli(x, y);
    let r = {
        let g = &mut f;
        let h = g as &mut dyn OpInt;
        apply(3, h)
    };
    assert_eq!(r, 9);
    println!("ok");
}
"#;

#[test]
fn dyn_call_on_closure_self_via_blanket_fn_impl() {
    let dir = temp_dir("dyn-closure-self");
    let src = fixture(&dir, "main.rs", CLOSURE_SELF_DYN_SRC);
    assert_ok(&run_driver(&src));
}
