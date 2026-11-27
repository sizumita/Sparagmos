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
//! * An impl whose `Self` is a reference type keeps the
//!   reference's lifetime in the instance key
//!   (`<&'a isize as Trait>::get`), while the fat pointer's
//!   concrete tag drops it (`&isize`); the impl scan's
//!   canonical spelling must drop reference lifetimes too,
//!   without equating two different reference pointees
//!   (tests/ui/regions/regions-close-over-type-parameter-
//!   successfully.rs).
//! * A `final` trait method is not in the vtable: the
//!   compiler resolves a call through the trait object to
//!   the trait's own body with `Self` set to the trait
//!   object type, and that body is stored under the
//!   callee's plain name with a trait-object receiver. The
//!   virtual lookup must decline such a call so the direct
//!   path runs that body with the fat pointer as receiver
//!   (tests/ui/traits/final/dyn-compat.rs).
//! * A trait object formed inside a constant -- a static's
//!   initializer or a promoted constant such as the generated
//!   test harness's tests array -- is named by no body cast;
//!   decoding the constant's fat pointer must record the
//!   vtable's demand, or a trait default the impl does not
//!   override has no converted body to dispatch to
//!   (tests/ui/custom_test_frameworks/dynamic.rs).
//! * A closure `Self` is spelled with the compiler's span
//!   form (`{closure@<file>:16:17: 16:23}`) in the impl
//!   instance key but with the definition path's indexed
//!   segment (`main::{closure#0}`) in the fat pointer's
//!   concrete tag; the impl scan must equate the two
//!   spellings (tests/ui/regions/issue-6157.rs).
//! * A concrete type may implement one trait under several
//!   generic instantiations (`Bar<i32>` and `Bar<bool>` for
//!   one `S2`); neither the call's name nor the impl scan
//!   can choose between the instances, but the fat pointer's
//!   own bound names the instance the pointer was formed
//!   for, so the lookup tries the bound's spelling as a
//!   candidate key before scanning (tests/ui/traits/
//!   trait-upcasting/correct-supertrait-substitution.rs).
//! * A `Self` carrying a function pointer with a concrete
//!   region (`Foo<fn(&'static u8)>`) renders in the fat
//!   pointer's concrete tag with the region dropped
//!   (`Foo<fn(&u8)>`), while the impl key keeps it; the
//!   scan's last pass matches up to region erasure, with
//!   binder-carrying keys (`Foo<for<'a> fn(&'a u8)>`)
//!   excluded, and resolves only a unique candidate
//!   (tests/ui/codegen/subtyping-impacts-selection-1.rs).

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

const VTABLE_INTRINSIC_SRC: &str = r#"
#![feature(intrinsics)]
#[rustc_intrinsic]
pub unsafe fn vtable_size(_ptr: *const ()) -> usize {
    panic!("fallback body must not run");
}
#[rustc_intrinsic]
pub unsafe fn vtable_align(_ptr: *const ()) -> usize {
    panic!("fallback body must not run");
}
trait Trait {}
struct S(u64, u32);
impl Trait for S {}
impl Trait for () {}
fn main() {
    let x: &dyn Trait = &();
    let s: &dyn Trait = &S(1, 2);
    unsafe {
        let (_, vt0): (*const (), *const ()) = core::mem::transmute(x);
        assert_eq!(vtable_size(vt0), 0);
        let (_, vt1): (*const (), *const ()) = core::mem::transmute(s);
        assert_eq!(vtable_size(vt1), 16);
        assert_eq!(vtable_align(vt1), 8);
    }
    println!("ok");
}
"#;

#[test]
fn dyn_call_on_closure_self_via_blanket_fn_impl() {
    let dir = temp_dir("dyn-closure-self");
    let src = fixture(&dir, "main.rs", CLOSURE_SELF_DYN_SRC);
    assert_ok(&run_driver(&src));
}

const FN_ITEM_SELF_DYN_SRC: &str = r#"
use std::marker::PhantomData;

trait Trait {
    fn method(&self) -> u32;
}
impl<F, R> Trait for (F, PhantomData<R>)
where
    F: Fn() -> R,
{
    fn method(&self) -> u32 { 42 }
}

fn ret_i32() -> i32 { 3 }
fn ret_unit() {}
fn ret_opaque() -> impl Sized {}

fn main() {
    assert_eq!((&(ret_i32, PhantomData) as &dyn Trait).method(), 42);
    assert_eq!((&(ret_unit, PhantomData) as &dyn Trait).method(), 42);
    assert_eq!((&(ret_opaque, PhantomData) as &dyn Trait).method(), 42);
    println!("ok");
}
"#;

const CONST_FORMED_TABLE_SRC: &str = r#"
trait Testable {
    fn name(&self) -> u32;
    fn run(&self) -> bool { true }
    fn subtests(&self) -> u32 { 0 }
}
struct AllFoo(u32);
impl Testable for AllFoo {
    fn name(&self) -> u32 { self.0 }
    fn subtests(&self) -> u32 { 2 }
}
fn tests() -> &'static [&'static dyn Testable] { &[&AllFoo(7)] }
fn main() {
    let t = tests()[0];
    assert!(t.run());
    assert_eq!(t.name(), 7);
    assert_eq!(t.subtests(), 2);
    println!("ok");
}
"#;

#[test]
fn dyn_call_on_fn_item_self_via_blanket_impl() {
    let dir = temp_dir("dyn-fn-item-self");
    let src = fixture(&dir, "main.rs", FN_ITEM_SELF_DYN_SRC);
    assert_ok(&run_driver(&src));
}

#[test]
fn run_mode_guest_declared_vtable_intrinsics_override_fallback() {
    // A guest crate's own `#[rustc_intrinsic]` declaration
    // with a fallback body: the backend's model of the
    // intrinsic overrides the fallback (the body serves
    // backends without the intrinsic), so the call never
    // enters the panicking body. `vtable_size` /
    // `vtable_align` answer the size and alignment stored
    // in the vtable -- the concrete type's layout.
    let dir = temp_dir("vtable-intrinsic");
    let src = fixture(&dir, "vtintr.rs", VTABLE_INTRINSIC_SRC);
    assert_ok(&run_driver(&src));
}

const REF_SELF_DYN_SRC: &str = r#"
trait Get { fn get(&self) -> isize; }
impl<'a> Get for &'a isize {
    fn get(&self) -> isize { **self }
}
impl<'a> Get for &'a str {
    fn get(&self) -> isize { self.len() as isize }
}
fn make_object<'a, A: Get + 'a>(v: A) -> Box<dyn Get + 'a> {
    Box::new(v) as Box<dyn Get + 'a>
}
fn main() {
    let i: isize = 22;
    let obj = make_object(&i);
    assert_eq!(obj.get(), 22);
    let s: &str = "abcd";
    let obj2 = make_object(s);
    assert_eq!(obj2.get(), 4);
    println!("ok");
}
"#;

#[test]
fn dyn_call_on_reference_self_impl() {
    // The impl instance key spells the `Self` type with its
    // lifetime (`<&'a isize as Get>::get`); the fat pointer's
    // concrete tag drops it (`&isize`). The two reference
    // impls must still dispatch to their own bodies.
    let dir = temp_dir("dyn-ref-self");
    let src = fixture(&dir, "main.rs", REF_SELF_DYN_SRC);
    assert_ok(&run_driver(&src));
}

const FINAL_DYN_SRC: &str = r#"
#![feature(final_associated_functions)]
trait FinalGeneric {
    final fn generic<T>(&self, _value: T) {}
}
trait FinalSelfReturn {
    final fn self_return(&self) -> &Self { self }
}
trait Val {
    fn val(&self) -> i32;
    final fn doubled(&self) -> i32 { self.val() * 2 }
}
struct S;
impl FinalGeneric for S {}
impl FinalSelfReturn for S {}
impl Val for S {
    fn val(&self) -> i32 { 21 }
}
fn main() {
    let s = S;
    let obj_generic: &dyn FinalGeneric = &s;
    obj_generic.generic(1u8);
    let obj_return: &dyn FinalSelfReturn = &s;
    let _ = obj_return.self_return();
    let obj_val: &dyn Val = &s;
    assert_eq!(obj_val.doubled(), 42);
    println!("ok");
}
"#;

#[test]
fn dyn_call_final_trait_method() {
    // A `final` method has no vtable slot: the call through
    // the trait object names the trait's own body at
    // `Self = dyn Trait`, stored under the plain trait name
    // with a trait-object receiver. It must run directly --
    // including a nested virtual call on `self` from inside
    // the final body (`doubled` reaching `val`).
    let dir = temp_dir("dyn-final-method");
    let src = fixture(&dir, "main.rs", FINAL_DYN_SRC);
    assert_ok(&run_driver(&src));
}

const SUBTYPING_ERASED_RECEIVER_SRC: &str = r#"
#![allow(coherence_leak_check)]
struct Foo<T: 'static>(T);
fn useful<'a>(_: &'a u8) {}
trait GetInner {
    type Assoc;
    fn muahaha(&mut self) -> Self::Assoc;
}
impl GetInner for Foo<fn(&'static u8)> {
    type Assoc = String;
    fn muahaha(&mut self) -> String { String::from("I am a string") }
}
impl GetInner for Foo<for<'a> fn(&'a u8)> {
    type Assoc = [usize; 3];
    fn muahaha(&mut self) -> [usize; 3] { [100; 3] }
}
fn break_me(hr_fnptr: Box<Foo<for<'a> fn(&'a u8)>>) -> Box<dyn GetInner<Assoc = String>> {
    let lr_fnptr = hr_fnptr as Box<Foo<fn(&'static u8)>>;
    lr_fnptr as Box<dyn GetInner<Assoc = String>>
}
fn main() {
    drop(Box::new(Foo(useful as fn(&'static u8))) as Box<dyn GetInner<Assoc = String>>);
    drop(Box::new(Foo(useful as fn(&u8))) as Box<dyn GetInner<Assoc = [usize; 3]>>);
    let mut any = break_me(Box::new(Foo(useful)));
    let evil_string = any.muahaha();
    assert_eq!(evil_string, "I am a string");
    println!("ok");
}
"#;

#[test]
fn subtyping_erased_receiver_resolves_static_impl() {
    // tests/ui/codegen/subtyping-impacts-selection-1.rs: the
    // value reaching the call was reinterpreted to the
    // `'static` shape, but its fat pointer's concrete tag
    // drops the region (`Foo<fn(&u8)>`), so neither the
    // `'static` key nor the higher-ranked key matches it
    // under the canonical spelling. The scan's region-erasure
    // pass excludes the binder-carrying key and resolves the
    // unique binderless one.
    let dir = temp_dir("subtyping-erased-receiver");
    let src = fixture(&dir, "main.rs", SUBTYPING_ERASED_RECEIVER_SRC);
    assert_ok(&run_driver(&src));
}

const STATIC_IMPL_DIRECT_UNSIZE_SRC: &str = r#"
#![allow(coherence_leak_check)]
struct Foo<T: 'static>(T);
fn useful<'a>(_: &'a u8) {}
trait GetInner {
    type Assoc;
    fn muahaha(&mut self) -> Self::Assoc;
}
impl GetInner for Foo<fn(&'static u8)> {
    type Assoc = String;
    fn muahaha(&mut self) -> String { String::from("I am a string") }
}
impl GetInner for Foo<for<'a> fn(&'a u8)> {
    type Assoc = [usize; 3];
    fn muahaha(&mut self) -> [usize; 3] { [100; 3] }
}
fn main() {
    let mut any: Box<dyn GetInner<Assoc = String>> = Box::new(Foo(useful as fn(&'static u8)));
    let s = any.muahaha();
    assert_eq!(s, "I am a string");
    println!("ok");
}
"#;

#[test]
fn erased_receiver_direct_unsize_resolves_static_impl() {
    // The same erased tag as the subtyping record, formed by
    // a direct unsizing cast rather than a reinterpretation:
    // the call must still reach the `'static` impl, not the
    // higher-ranked sibling.
    let dir = temp_dir("erased-receiver-direct");
    let src = fixture(&dir, "main.rs", STATIC_IMPL_DIRECT_UNSIZE_SRC);
    assert_ok(&run_driver(&src));
}

const HRTB_IMPL_UNIQUE_SRC: &str = r#"
#![allow(coherence_leak_check)]
struct Foo<T: 'static>(T);
fn useful<'a>(_: &'a u8) {}
trait GetInner {
    type Assoc;
    fn muahaha(&mut self) -> Self::Assoc;
}
impl GetInner for Foo<for<'a> fn(&'a u8)> {
    type Assoc = [usize; 3];
    fn muahaha(&mut self) -> [usize; 3] { [100; 3] }
}
fn main() {
    let mut any: Box<dyn GetInner<Assoc = [usize; 3]>> = Box::new(Foo(useful as fn(&u8)));
    let v = any.muahaha();
    assert_eq!(v, [100; 3]);
    println!("ok");
}
"#;

#[test]
fn hrtb_impl_resolves_when_it_is_the_unique_impl() {
    // With no binderless sibling, the higher-ranked impl is
    // registered under the erased tag itself and the call
    // resolves to it exactly as before the region-erasure
    // pass existed.
    let dir = temp_dir("hrtb-impl-unique");
    let src = fixture(&dir, "main.rs", HRTB_IMPL_UNIQUE_SRC);
    assert_ok(&run_driver(&src));
}

const HRTB_IMPL_STATIC_CALL_SRC: &str = r#"
#![allow(coherence_leak_check)]
struct Foo<T: 'static>(T);
fn useful<'a>(_: &'a u8) {}
trait GetInner {
    type Assoc;
    fn muahaha(&mut self) -> Self::Assoc;
}
impl GetInner for Foo<fn(&'static u8)> {
    type Assoc = String;
    fn muahaha(&mut self) -> String { String::from("I am a string") }
}
impl GetInner for Foo<for<'a> fn(&'a u8)> {
    type Assoc = [usize; 3];
    fn muahaha(&mut self) -> [usize; 3] { [100; 3] }
}
fn main() {
    let mut f = Foo(useful as fn(&u8));
    let v = f.muahaha();
    assert_eq!(v, [100; 3]);
    println!("ok");
}
"#;

#[test]
fn hrtb_impl_static_call_resolves() {
    // A static call on the higher-ranked shape selects its
    // impl by the exact instance, as before.
    let dir = temp_dir("hrtb-impl-static-call");
    let src = fixture(&dir, "main.rs", HRTB_IMPL_STATIC_CALL_SRC);
    assert_ok(&run_driver(&src));
}

const REGION_ERASURE_DECLINE_SRC: &str = r#"
#![allow(coherence_leak_check)]
struct Foo<T: 'static>(T);
fn helper(_: &'static u8, _: &u8) {}
trait GetInner {
    fn muahaha(&mut self) -> u32;
}
impl GetInner for Foo<fn(&'static u8, &u8)> {
    fn muahaha(&mut self) -> u32 { 1 }
}
impl GetInner for Foo<fn(&u8, &'static u8)> {
    fn muahaha(&mut self) -> u32 { 2 }
}
fn main() {
    let mut f = Foo(helper as fn(&'static u8, &u8));
    let d: &mut dyn GetInner = &mut f;
    assert_eq!(d.muahaha(), 1);
    println!("ok");
}
"#;

#[test]
fn region_erasure_two_binderless_candidates_decline() {
    // Both impls are binderless and region-carrying, in
    // swapped positions of the same signature; both match
    // the erased tag `Foo<fn(&u8, &u8)>` up to region
    // erasure, so the scan must resolve nothing rather than
    // guess: the run fails with exit code 2, as it did
    // before the erasure pass existed.
    let dir = temp_dir("region-erasure-decline");
    let src = fixture(&dir, "main.rs", REGION_ERASURE_DECLINE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "ambiguous erasure candidates must stay unresolved; stderr={stderr}"
    );
}

const MULTI_INSTANCE_SUPERTRAIT_SRC: &str = r#"
trait Foo<T: Default + ToString>: Bar<i32> + Bar<T> {}
trait Bar<T: Default + ToString> {
    fn bar(&self) -> String {
        T::default().to_string()
    }
}

struct S1;
impl Bar<i32> for S1 {}
impl Foo<i32> for S1 {}

struct S2;
impl Bar<i32> for S2 {}
impl Bar<bool> for S2 {}
impl Foo<bool> for S2 {}

fn test1(x: &dyn Foo<i32>) {
    let s = x as &dyn Bar<i32>;
    assert_eq!("0", &s.bar().to_string());
}

fn test2(x: &dyn Foo<bool>) {
    let p = x as &dyn Bar<i32>;
    assert_eq!("0", &p.bar().to_string());
    let q = x as &dyn Bar<bool>;
    assert_eq!("false", &q.bar().to_string());
}

fn main() {
    let s1 = S1;
    test1(&s1);
    let s2 = S2;
    test2(&s2);
    println!("ok");
}
"#;

#[test]
fn dyn_call_picks_the_bound_instance_among_supertrait_upcasts() {
    let dir = temp_dir("dyn-bound-instance");
    let src = fixture(&dir, "main.rs", MULTI_INSTANCE_SUPERTRAIT_SRC);
    assert_ok(&run_driver(&src));
}

#[test]
fn dyn_default_method_on_const_formed_trait_object() {
    let dir = temp_dir("dyn-const-formed");
    let src = fixture(&dir, "main.rs", CONST_FORMED_TABLE_SRC);
    assert_ok(&run_driver(&src));
}
