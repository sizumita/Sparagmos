use super::{ANON_CONST_GENERIC_SRC, CLOSURE_GENERIC_CONTEXT_SRC, CTOR_VALUE_SRC, DYN_CONST_DISPATCH_SRC, DYN_DEFAULT_SRC, EXTERNAL_DEMAND_SRC, TAIL_CALL_SRC, fixture, summarize_driver, temp_dir};


#[test]
fn summary_mode_tail_call_body_is_skipped_with_named_reason() {
    // In summary (convert) mode the same containment reports the
    // tail-call body as skipped with its named reason, and the
    // rest of the crate still converts: no conversion panic is
    // reported anywhere.
    let dir = temp_dir("tailcall-summary");
    let src = fixture(&dir, "tailcall.rs", TAIL_CALL_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "summary mode succeeds with a named skip; stderr={stderr}"
    );
    let line = stdout
        .lines()
        .find(|l| l.contains("countdown"))
        .unwrap_or_else(|| panic!("summary lists countdown: {stdout}"));
    assert!(
        line.contains("body contains a tail call"),
        "the skip reason names the tail call: {line}"
    );
    assert!(
        !stdout.contains("panicked") && !stderr.contains("panicked"),
        "no conversion panic is reported: stdout={stdout} stderr={stderr}"
    );
}

const CLOSURE_FNONCE_SRC: &str = r#"
struct Bag(i32);
fn main() {
    let bag = Bag(41);
    let f = move || bag;
    let _b = f();
}
"#;

#[test]
fn summary_mode_generic_context_closure_is_converted_when_demanded() {
    // A closure inside a generic function has no instance as an
    // item, but the monomorphic instance of the enclosing function
    // (`wrap::<i32>` here) mentions it with concrete arguments, and
    // the closure collection converts the demanded instance.
    let dir = temp_dir("closure-generic");
    let src = fixture(&dir, "closure_generic.rs", CLOSURE_GENERIC_CONTEXT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the closure item");
    assert!(
        line.contains("[converted]"),
        "the demanded generic-context closure instance is converted: {line}"
    );
}

const CLOSURE_GENERIC_UNDEMANDED_SRC: &str = r#"
fn wrap<T: Copy>(v: T) -> T {
    let id = |x: T| x;
    id(v)
}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_undemanded_generic_context_closure_is_not_demanded() {
    // The same closure with no monomorphic caller anywhere is
    // accounted not demanded, with the reason naming the generic
    // context: nothing reaches it, so it is not part of the
    // program, but the record stays (it is not counted converted,
    // and it does not count as a conversion failure either).
    let dir = temp_dir("closure-generic-undemanded");
    let src = fixture(
        &dir,
        "closure_generic_undemanded.rs",
        CLOSURE_GENERIC_UNDEMANDED_SRC,
    );
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the closure item");
    assert!(
        line.contains("NOT DEMANDED") && line.contains("generic context"),
        "the undemanded closure is not demanded, with the generic-context reason: {line}"
    );
}

const COROUTINE_DEMANDED_SRC: &str = r#"
fn main() {
    let _a = async { 1 };
    let _b = async move { 2 };
}
"#;

#[test]
fn summary_mode_demanded_async_block_is_converted() {
    // A coroutine item (an async block) has no monomorphic
    // instance as an item, but the converted body that creates it
    // mentions it with concrete arguments, and the coroutine
    // collection converts the demanded instance.
    let dir = temp_dir("coroutine-demanded");
    let src = fixture(&dir, "coroutine_demanded.rs", COROUTINE_DEMANDED_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["{closure#0}", "{closure#1}"] {
        let line = stdout
            .lines()
            .find(|l| l.contains(name))
            .expect("summary lists the coroutine item");
        assert!(
            line.contains("[converted]"),
            "the demanded coroutine instance is converted: {line}"
        );
    }
}

const COROUTINE_UNDEMANDED_SRC: &str = r#"
fn wrap<T: Copy>(v: T) -> T {
    let _f = async move { v };
    v
}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_undemanded_coroutine_is_not_demanded() {
    // A coroutine inside a generic function that is never
    // instantiated has no concrete demand anywhere and is
    // accounted not demanded: an undemanded generic is never
    // counted converted, and its absence is not a conversion
    // failure either.
    let dir = temp_dir("coroutine-undemanded");
    let src = fixture(&dir, "coroutine_undemanded.rs", COROUTINE_UNDEMANDED_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the coroutine item");
    assert!(
        line.contains("NOT DEMANDED") && line.contains("no monomorphic instance"),
        "the undemanded coroutine is not demanded: {line}"
    );
}

const UNNORMALIZABLE_PROJECTION_FN_SRC: &str = r#"
trait Trait {
    type Type;
}
struct Extern;
fn f<'a>(_: <&'a Extern as Trait>::Type) where &'a Extern: Trait {}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_unnormalizable_projection_fn_is_not_demanded() {
    // A function whose argument type is a trait projection
    // no implementation covers: materializing its body would
    // ask the compiler to normalize the projection in a
    // monomorphized typing environment, which ICEs the whole
    // conversion (a delayed bug, uncontainable). Nothing
    // demands the function — rustc's monomorphization
    // collector never instantiates it either — so it is
    // accounted not demanded with the screen's reason, and
    // the rest of the crate still converts.
    let dir = temp_dir("projection-fn-undemanded");
    let src = fixture(
        &dir,
        "projection_fn_undemanded.rs",
        UNNORMALIZABLE_PROJECTION_FN_SRC,
    );
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("::f"))
        .expect("summary lists the projection-argument function");
    assert!(
        line.contains("NOT DEMANDED") && line.contains("does not normalize"),
        "the projection-argument function is not demanded: {line}"
    );
    let main_line = stdout
        .lines()
        .find(|l| l.contains("::main"))
        .expect("summary lists main");
    assert!(
        main_line.contains("[converted]"),
        "main still converts: {main_line}"
    );
}

const UNNORMALIZABLE_PROJECTION_CTOR_SRC: &str = r#"
#![feature(trivial_bounds)]
#![allow(trivial_bounds)]
pub trait Foo {
    type Assoc;
}
pub struct Bar;
pub struct Baz(<Bar as Foo>::Assoc)
where
    Bar: Foo;
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_unnormalizable_projection_ctor_is_converted() {
    // A tuple struct whose field is a projection no
    // implementation covers: the constructor's signature
    // carries the projection, so materializing its body
    // would ICE the conversion the same way. An undemanded
    // constructor is accounted converted (the accounting
    // undemanded generic constructors get), and the crate
    // still converts.
    let dir = temp_dir("projection-ctor");
    let src = fixture(
        &dir,
        "projection_ctor.rs",
        UNNORMALIZABLE_PROJECTION_CTOR_SRC,
    );
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("::Baz"))
        .expect("summary lists the projection-field constructor");
    assert!(
        line.contains("[converted]"),
        "the undemanded projection-field constructor is converted: {line}"
    );
}

const UNNORMALIZABLE_PROJECTION_CONST_SRC: &str = r#"
#![feature(const_trait_impl, generic_const_exprs)]
#![allow(incomplete_features)]
const trait Trait {
    type Assoc: [const] Trait;
    fn func() -> i32;
}
struct Type<const N: i32>;
fn unqualified<T: const Trait>() -> Type<{ T::Assoc::func() }> {
    Type
}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_unnormalizable_projection_const_is_not_demanded() {
    // An anonymous constant inside a generic function, whose
    // body resolves an associated call through a projection
    // over the function's own parameter: converting the
    // constant would pose that normalization during
    // resolution and ICE. The constant is a generic
    // definition nothing demands (monomorphized bodies that
    // use it carry its evaluated value), so it is accounted
    // not demanded with the screen's reason.
    let dir = temp_dir("projection-const-undemanded");
    let src = fixture(
        &dir,
        "projection_const_undemanded.rs",
        UNNORMALIZABLE_PROJECTION_CONST_SRC,
    );
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{constant#0}"))
        .expect("summary lists the anonymous constant");
    assert!(
        line.contains("NOT DEMANDED") && line.contains("does not normalize"),
        "the projection constant is not demanded: {line}"
    );
}

const PROJECTION_PARAM_SELF_CONST_SRC: &str = r#"
use std::mem::MaybeUninit;
pub fn fill<T: Iterator, const N: usize>(_: T) -> [MaybeUninit<T::Item>; N] {
    [const { MaybeUninit::uninit() }; N]
}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_projection_over_parameter_const_is_not_demanded() {
    // An anonymous constant inside a generic function whose
    // body calls an associated function through a projection
    // over the function's own parameter
    // (`MaybeUninit::<T::Item>::uninit`, the shape of core's
    // `array::try_from_fn`): converting the constant demands
    // the callee instance, and materializing that instance's
    // body poses a whole-body normalization of the projection
    // in an empty typing environment, which overflows inside
    // the compiler and poisons the whole conversion. The
    // instance screen refuses the demand up front (the
    // projection's self type is the parameter itself), the
    // demanded instance is accounted skipped with the
    // screen's reason, and the constant and the crate still
    // convert.
    let dir = temp_dir("projection-param-self-const");
    let src = fixture(
        &dir,
        "projection_param_self_const.rs",
        PROJECTION_PARAM_SELF_CONST_SRC,
    );
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "summary mode succeeds with the demand refused; stderr={stderr}"
    );
    let const_line = stdout
        .lines()
        .find(|l| l.contains("{constant#"))
        .expect("summary lists the anonymous constant");
    assert!(
        const_line.contains("[converted]"),
        "the anonymous constant converts: {const_line}"
    );
    let uninit_line = stdout
        .lines()
        .find(|l| l.contains("MaybeUninit"))
        .expect("summary lists the demanded callee");
    assert!(
        uninit_line.contains("SKIPPED") && uninit_line.contains("does not normalize"),
        "the projection callee is skipped with the screen's reason: {uninit_line}"
    );
    let target_line = stdout
        .lines()
        .find(|l| l.contains("target-crate converted:"))
        .expect("summary reports the target crate");
    assert!(
        target_line.contains("skipped: 0"),
        "the target crate converts with nothing skipped: {target_line}"
    );
}

const UNNORMALIZABLE_PROJECTION_AUTO_TRAIT_MISMATCH_SRC: &str = r#"
#![feature(trivial_bounds)]
#![allow(trivial_bounds)]
trait Trait {
    type Assoc;
}
trait Other {}
trait Carrier {
    type Item: Trait + ?Sized;
}
impl Carrier for dyn Other {
    type Item = dyn Trait<Assoc = u32>;
}
fn f(_: <<dyn Other + Send as Carrier>::Item as Trait>::Assoc) where dyn Other + Send: Carrier {}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_unnormalizable_projection_auto_trait_mismatch_is_not_demanded() {
    // The equal-auto-trait requirement cuts both ways:
    // with only the `dyn Other` implementation present,
    // the `dyn Other + Send` projection has no applicable
    // implementation (natively an E0277), so the function
    // carrying it is accounted not demanded with the
    // screen's reason while main still converts. The
    // false where-clause is a trivial bound, the form the
    // const-mismatch case above also takes.
    let dir = temp_dir("projection-auto-trait-mismatch-undemanded");
    let src = fixture(
        &dir,
        "projection_auto_trait_mismatch.rs",
        UNNORMALIZABLE_PROJECTION_AUTO_TRAIT_MISMATCH_SRC,
    );
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("::f"))
        .expect("summary lists the projection-argument function");
    assert!(
        line.contains("NOT DEMANDED") && line.contains("does not normalize"),
        "the auto-trait-mismatch projection function is not demanded: {line}"
    );
    let main_line = stdout
        .lines()
        .find(|l| l.contains("::main"))
        .expect("summary lists main");
    assert!(
        main_line.contains("[converted]"),
        "main still converts: {main_line}"
    );
}

const UNNORMALIZABLE_PROJECTION_CONST_MISMATCH_SRC: &str = r#"
#![feature(trivial_bounds)]
#![allow(trivial_bounds)]
trait Trait {
    type Assoc;
}
trait Carrier<const N: usize> {
    type Item: Trait + ?Sized;
}
impl Carrier<4> for () {
    type Item = dyn Trait<Assoc = u32>;
}
fn f(_: <<() as Carrier<3>>::Item as Trait>::Assoc) where (): Carrier<3> {}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_unnormalizable_projection_const_mismatch_is_not_demanded() {
    // The inconclusive treatment reaches only
    // unevaluated constants: two concrete constants that
    // differ (`Carrier<4>` implemented, `Carrier<3>`
    // projected) are still a proven mismatch, no
    // implementation assigns the item, and the function
    // carrying the projection is accounted not demanded
    // with the screen's reason while main still converts.
    // The false where-clause is a trivial bound, the
    // form the ctor case above also takes: without it
    // the definition itself fails WF checking.
    let dir = temp_dir("projection-const-mismatch-undemanded");
    let src = fixture(
        &dir,
        "projection_const_mismatch.rs",
        UNNORMALIZABLE_PROJECTION_CONST_MISMATCH_SRC,
    );
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("::f"))
        .expect("summary lists the projection-argument function");
    assert!(
        line.contains("NOT DEMANDED") && line.contains("does not normalize"),
        "the const-mismatch projection function is not demanded: {line}"
    );
    let main_line = stdout
        .lines()
        .find(|l| l.contains("::main"))
        .expect("summary lists main");
    assert!(
        main_line.contains("[converted]"),
        "main still converts: {main_line}"
    );
}

const COROUTINE_ONLY_CALLER_SRC: &str = r#"
fn helper<T: Copy>(v: T) -> T { v }
fn main() {
    let _f = async { helper::<i32>(3) };
}
"#;

#[test]
fn summary_mode_generic_fn_called_only_from_coroutine_body_is_converted() {
    // The only call to the generic function lives in the
    // coroutine's body. Converting the coroutine must feed the
    // instance demand too: every converted body is scanned by
    // every demand pass.
    let dir = temp_dir("coroutine-only-caller");
    let src = fixture(&dir, "coroutine_only_caller.rs", COROUTINE_ONLY_CALLER_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("helper"))
        .expect("summary lists the generic function item");
    assert!(
        line.contains("[converted]"),
        "the generic function demanded only by the coroutine body is converted: {line}"
    );
}

#[test]
fn summary_mode_fnonce_closure_body_is_converted() {
    // Conversion covers FnOnce closures too: the body is collected
    // even though the interpreter's by-value environment passing is
    // a separate concern.
    let dir = temp_dir("closure-fnonce");
    let src = fixture(&dir, "closure_fnonce.rs", CLOSURE_FNONCE_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the closure item");
    assert!(
        line.contains("[converted]"),
        "the FnOnce closure body is converted: {line}"
    );
}

const GENERIC_UNUSED_SRC: &str = r#"
fn used<T>(x: T) -> T { x }
fn unused<T: Clone>(x: T) -> T { x.clone() }
fn main() {
    let _ = used::<u32>(1);
}
"#;

#[test]
fn summary_mode_demanded_generic_instances_are_converted() {
    // The demanded definition is reported converted; a definition
    // nothing demands keeps its no-instance record as not
    // demanded: it is not silently counted as converted, and it
    // does not make the target crate's conversion incomplete.
    let dir = temp_dir("generic-unused");
    let src = fixture(&dir, "generic_unused.rs", GENERIC_UNUSED_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let used = stdout
        .lines()
        .find(|l| l.starts_with("fn generic_unused::used:"))
        .expect("summary lists the used generic definition");
    assert!(
        used.contains("[converted]"),
        "the demanded generic definition is converted: {used}"
    );
    let unused = stdout
        .lines()
        .find(|l| l.starts_with("fn generic_unused::unused:"))
        .expect("summary lists the unused generic definition");
    assert!(
        unused.contains("NOT DEMANDED (no monomorphic instance available (kind=Fn)"),
        "the undemanded generic definition is explicitly not demanded: {unused}"
    );
    assert!(
        stdout.contains("target-crate converted: 2, skipped: 0"),
        "the target crate is complete with the undemanded definition aside: {stdout}"
    );
    assert!(
        stdout.contains("target-crate not demanded: 1"),
        "the undemanded definition is counted separately: {stdout}"
    );
}

const GENERIC_CONST_SRC: &str = r#"
struct Foo<const N: usize>;
trait Bar { const VAL: usize; }
impl<const N: usize> Bar for Foo<N> { const VAL: usize = N; }
fn arr<const N: usize>() -> [u8; N] { [0; N] }
fn main() { let _ = <Foo<3> as Bar>::VAL; let _ = arr::<2>(); }
"#;

#[test]
fn summary_mode_generic_const_definition_is_not_demanded() {
    // A generic constant definition (here an associated constant
    // of a generic impl) has no instance as an item, and its own
    // body still mentions its const parameter, so converting the
    // definition body fails on `ConstantKind::Param`. That is the
    // generic definition itself, not a monomorphic body: the
    // monomorphized bodies that use the constant carry its
    // evaluated value, so the definition is accounted not
    // demanded — it does not make the target crate's conversion
    // incomplete, and it is not silently counted as converted.
    let dir = temp_dir("generic-const");
    let src = fixture(&dir, "generic_const.rs", GENERIC_CONST_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let val = stdout
        .lines()
        .find(|l| l.starts_with("fn <generic_const::Foo<N> as generic_const::Bar>::VAL:"))
        .expect("summary lists the generic associated constant");
    assert!(
        val.contains("NOT DEMANDED") && val.contains("generic constant definition"),
        "the generic constant definition is explicitly not demanded: {val}"
    );
    assert!(
        stdout.contains("target-crate converted: 4, skipped: 0"),
        "the target crate is complete with the generic constant aside: {stdout}"
    );
    assert!(
        stdout.contains("target-crate not demanded: 1"),
        "the generic constant is counted separately: {stdout}"
    );
}

const TAIL_CALL_GENERIC_SRC: &str = r#"
#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]
fn countdown<T: Copy>(n: u32, m: T) -> u32 {
    if n == 0 { 0 } else { become countdown(n - 1, m) }
}
fn main() { let _ = countdown::<u32>(3, 0); }
"#;

#[test]
fn summary_mode_demanded_generic_with_unconvertible_body_stays_skipped() {
    // Demand does not excuse a body that cannot be converted: the
    // generic definition is demanded concretely from `main`, its
    // instance fails conversion with the named tail-call reason,
    // and the definition stays skipped — the target crate stays
    // incomplete for that named reason alone.
    let dir = temp_dir("generic-tailcall");
    let src = fixture(&dir, "generic_tailcall.rs", TAIL_CALL_GENERIC_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("fn generic_tailcall::countdown:"))
        .expect("summary lists the generic definition");
    assert!(
        line.contains("SKIPPED") && line.contains("tail call"),
        "the demanded-but-unconvertible definition stays skipped with its reason: {line}"
    );
    assert!(
        stdout.contains("target-crate converted: 1, skipped: 1"),
        "the target crate stays incomplete for the demanded failure: {stdout}"
    );
}

const GENERIC_DROP_CREDIT_SRC: &str = r#"
struct Guard<T> { _v: T }
impl<T> Drop for Guard<T> { fn drop(&mut self) {} }
struct Holder { _g: Guard<u32> }
fn main() { let _h = Holder { _g: Guard { _v: 1 } }; }
"#;

#[test]
fn summary_mode_generic_drop_impl_is_credited_from_drop_demand() {
    // A generic `Drop` impl is demanded only through drop glue
    // (never a call edge): the drop demand converts the concrete
    // instance, and that conversion is credited to the impl
    // definition, whose own entry is reported converted instead
    // of standing as an unmet no-instance record. (That the
    // demanded destructor body actually runs is pinned by the
    // Lane BS run-mode tests below.)
    let dir = temp_dir("generic-drop-credit");
    let src = fixture(&dir, "generic_drop_credit.rs", GENERIC_DROP_CREDIT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("fn <generic_drop_credit::Guard<T> as std::ops::Drop>::drop:"))
        .expect("summary lists the generic Drop impl");
    assert!(
        line.contains("[converted]"),
        "the drop-demanded generic Drop impl is converted: {line}"
    );
    assert!(
        stdout.contains("target-crate converted: 2, skipped: 0"),
        "no target-crate function stays skipped: {stdout}"
    );
}

#[test]
fn summary_mode_dyn_trait_default_method_is_converted() {
    // The demanded vtable instance also settles the trait
    // method's own accounting: it no longer stands as a skipped
    // "dynamic dispatch target (virtual instance)".
    let dir = temp_dir("dyn-default-summary");
    let src = fixture(&dir, "dyn_default.rs", DYN_DEFAULT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["::Foo::a:", "::Foo::b:", "::Greeter::greet:"] {
        let line = stdout
            .lines()
            .find(|l| l.contains(name))
            .unwrap_or_else(|| panic!("summary lists {name}: {stdout}"));
        assert!(
            line.contains("[converted]"),
            "the trait default method {name} is converted: {line}"
        );
    }
    assert!(
        !stdout.contains("target-crate unsupported:"),
        "no target-crate function stays skipped: {stdout}"
    );
}

#[test]
fn summary_mode_generic_context_anon_consts_are_converted() {
    // An anonymous constant inside a generic function requires
    // monomorphization as an item, so it has no instance; its own
    // MIR body is converted instead, and the target crate reports
    // no skipped functions.
    let dir = temp_dir("anon-const-generic");
    let src = fixture(&dir, "anon_const_generic.rs", ANON_CONST_GENERIC_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let konst = stdout
        .lines()
        .find(|l| l.contains("::{constant#0}:"))
        .expect("summary lists the anonymous constant");
    assert!(
        konst.contains("[converted]"),
        "the generic-context anonymous constant is converted: {konst}"
    );
    assert!(
        stdout.contains("target-crate converted: 3, skipped: 0"),
        "no target-crate function is skipped: {stdout}"
    );
}

const TRIVIAL_CONST_SRC: &str = r#"
const N: usize = 3;
static S: [u32; 3] = [4; 1 + 2];
fn main() {
    let a = [0u32; N];
    println!("{}", a.len() + S.len());
}
"#;

#[test]
fn summary_mode_trivial_consts_are_converted_without_bodies() {
    // Trivial constants (a literal, a path) have no MIR body
    // anywhere by design; they are accounted as converted with no
    // body instead of being reported as missing bodies, while the
    // non-trivial constant of the static initializer keeps its
    // real body.
    let dir = temp_dir("trivial-consts");
    let src = fixture(&dir, "trivial_consts.rs", TRIVIAL_CONST_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let named = stdout
        .lines()
        .find(|l| l.starts_with("fn trivial_consts::N:"))
        .expect("summary lists the named constant");
    assert!(
        named.contains("blocks=0") && named.contains("[converted]"),
        "the trivial named constant is converted with no body: {named}"
    );
    let trivial = stdout
        .lines()
        .find(|l| l.contains("::S::{constant#0}:"))
        .expect("summary lists the static type-length constant");
    assert!(
        trivial.contains("blocks=0") && trivial.contains("[converted]"),
        "the trivial type-length constant is converted with no body: {trivial}"
    );
    let real = stdout
        .lines()
        .find(|l| l.contains("::S::{constant#1}:"))
        .expect("summary lists the static initializer constant");
    assert!(
        real.contains("[converted]") && !real.contains("blocks=0"),
        "the non-trivial constant keeps its converted body: {real}"
    );
    assert!(
        stdout.contains("target-crate skipped: 0")
            || stdout.contains("skipped: 0"),
        "no target-crate function is skipped: {stdout}"
    );
}

#[test]
fn summary_mode_externally_demanded_instances_are_converted() {
    // The definitions demanded through foreign bodies are
    // reported converted; a definition nothing demands keeps its
    // no-instance record.
    let dir = temp_dir("external-demand-summary");
    let src = fixture(&dir, "external_demand.rs", EXTERNAL_DEMAND_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let sum = stdout
        .lines()
        .find(|l| l.contains("::sum:") && l.contains("Sum<i32>"))
        .expect("summary lists the Sum impl method");
    assert!(
        sum.contains("[converted]"),
        "the externally demanded generic method is converted: {sum}"
    );
    let next = stdout
        .lines()
        .find(|l| l.contains("RepeatN") && l.contains(">::next:"))
        .expect("summary lists the Iterator impl method");
    assert!(
        next.contains("[converted]"),
        "the transitively demanded generic method is converted: {next}"
    );
    let unused = stdout
        .lines()
        .find(|l| l.starts_with("fn external_demand::unused:"))
        .expect("summary lists the unused generic definition");
    assert!(
        unused.contains("NOT DEMANDED (no monomorphic instance available (kind=Fn)"),
        "the undemanded generic definition is explicitly not demanded: {unused}"
    );
}

const CTOR_DIRECT_ONLY_SRC: &str = r#"
struct G<T>(T, i32);
fn main() {
    let g = G(10, 2);
    println!("{}", g.0 + g.1);
}
"#;

#[test]
fn summary_mode_generic_ctor_is_accounted_converted() {
    // Used directly (aggregate rvalues in main) or as a value
    // (demanded instance converted), a generic constructor is
    // reported converted — not as a missing monomorphic instance.
    let dir = temp_dir("ctor-direct-only");
    let src = fixture(&dir, "ctor_direct_only.rs", CTOR_DIRECT_ONLY_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let g = stdout
        .lines()
        .find(|l| l.starts_with("fn ctor_direct_only::G:"))
        .expect("summary lists the generic constructor");
    assert!(
        g.contains("[converted]"),
        "an undemanded generic constructor is converted without a body: {g}"
    );

    let dir = temp_dir("ctor-value-summary");
    let src = fixture(&dir, "ctor_value.rs", CTOR_VALUE_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let pair = stdout
        .lines()
        .find(|l| l.starts_with("fn ctor_value::Pair:"))
        .expect("summary lists the generic constructor");
    assert!(
        pair.contains("[converted]"),
        "a demanded generic constructor is converted: {pair}"
    );
    assert!(
        !stdout.contains("kind=Ctor"),
        "no constructor is left as a missing instance: {stdout}"
    );
}

#[test]
fn summary_mode_dyn_ref_constant_in_static_initializer_is_converted() {
    // The conversion-side accounting of the same shape: the
    // body that reads the dispatch static through `addr_of!`
    // converts (it was skipped with a `ConstantKind::Allocated`
    // reason for the raw-pointer constant).
    let dir = temp_dir("dyn-const-dispatch-summary");
    let src = fixture(&dir, "dyn_const_dispatch.rs", DYN_CONST_DISPATCH_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("::get_global:"))
        .unwrap_or_else(|| panic!("summary lists get_global: {stdout}"));
    assert!(
        line.contains("[converted]"),
        "get_global is converted: {line}"
    );
    assert!(
        !stdout.contains("target-crate unsupported:"),
        "no target-crate function stays skipped: {stdout}"
    );
}
