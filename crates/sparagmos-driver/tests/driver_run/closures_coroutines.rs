use super::{
    CLOSURE_GENERIC_CONTEXT_SRC, fixture, run_driver, run_driver_extra, summarize_driver,
    temp_dir,
};


const CLOSURE_CAPTURE_SRC: &str = r#"
fn main() {
    let base = 40;
    let add = |x: i32| x + base;
    println!("{}", add(2));
}
"#;

const CLOSURE_NO_CAPTURE_SRC: &str = r#"
fn main() {
    let twice = |x: i32| x * 2;
    println!("{}", twice(21));
}
"#;

const CLOSURE_FNMUT_SRC: &str = r#"
fn main() {
    let mut count = 0;
    let mut inc = || { count += 1; count };
    println!("{}", inc());
    println!("{}", inc());
}
"#;

const CLOSURE_IN_STRUCT_SRC: &str = r#"
struct Holder<F: Fn(i32) -> i32> { f: F }
fn main() {
    let k = 5;
    let h = Holder { f: |x: i32| x + k };
    println!("{}", (h.f)(7));
}
"#;

const CLOSURE_FNONCE_MOVE_OUT_SRC: &str = r#"
fn main() {
    let s = String::from("abc");
    let f = move || -> String { s };
    let out = f();
    println!("{}", out);
}
"#;

const CLOSURE_FNONCE_IN_STRUCT_SRC: &str = r#"
struct Once<F: FnOnce() -> String> { f: F }
fn main() {
    let s = String::from("xyz");
    let o = Once { f: move || s };
    let out = (o.f)();
    println!("{}", out);
}
"#;

const CLOSURE_RETURNED_SRC: &str = r#"
fn make_adder(n: i32) -> impl Fn(i32) -> i32 {
    move |x| x + n
}
fn main() {
    let add = make_adder(30);
    println!("{}", add(12));
}
"#;

const CLOSURE_NESTED_SRC: &str = r#"
fn main() {
    let outer = |x: i32| {
        let inner = |y: i32| y + 1;
        inner(x) * 2
    };
    println!("{}", outer(10));
}
"#;

#[test]
fn run_mode_closure_with_capture() {
    let dir = temp_dir("closure-capture");
    let src = fixture(&dir, "closure_capture.rs", CLOSURE_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure capturing a local must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_closure_without_capture() {
    let dir = temp_dir("closure-no-capture");
    let src = fixture(&dir, "closure_no_capture.rs", CLOSURE_NO_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a captureless closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_fnmut_closure_keeps_state() {
    let dir = temp_dir("closure-fnmut");
    let src = fixture(&dir, "closure_fnmut.rs", CLOSURE_FNMUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnMut closure must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1\n2\n",
        "the captured counter must persist across calls"
    );
}

#[test]
fn run_mode_closure_stored_in_struct() {
    let dir = temp_dir("closure-in-struct");
    let src = fixture(&dir, "closure_in_struct.rs", CLOSURE_IN_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure stored in a struct field must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "12\n");
}

#[test]
fn run_mode_fnonce_closure_moves_capture_out() {
    let dir = temp_dir("closure-fnonce-move-out");
    let src = fixture(&dir, "closure_fnonce_move_out.rs", CLOSURE_FNONCE_MOVE_OUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnOnce closure moving its capture out must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "abc\n");
}

#[test]
fn run_mode_fnonce_closure_in_struct_field() {
    let dir = temp_dir("closure-fnonce-in-struct");
    let src = fixture(
        &dir,
        "closure_fnonce_in_struct.rs",
        CLOSURE_FNONCE_IN_STRUCT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnOnce closure stored in a struct field must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "xyz\n");
}

#[test]
fn run_mode_closure_returned_from_function() {
    let dir = temp_dir("closure-returned");
    let src = fixture(&dir, "closure_returned.rs", CLOSURE_RETURNED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure returned from a function must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_nested_closures() {
    let dir = temp_dir("closure-nested");
    let src = fixture(&dir, "closure_nested.rs", CLOSURE_NESTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested closures must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "22\n");
}

#[test]
fn run_mode_generic_context_closure_runs() {
    let dir = temp_dir("closure-generic-run");
    let src = fixture(&dir, "closure_generic.rs", CLOSURE_GENERIC_CONTEXT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure inside a generic function must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

const CLOSURE_GENERIC_TWO_TYPES_SRC: &str = r#"
fn twice<T: Copy + std::ops::Add<Output = T>>(v: T) -> T {
    let f = |x: T| x + x;
    f(v)
}
fn main() {
    println!("{}", twice::<i32>(21));
    println!("{}", twice::<f64>(1.5));
}
"#;

#[test]
fn run_mode_generic_closure_two_instantiations() {
    // Two monomorphic instances of one closure definition coexist
    // in the program and each call reaches its own body.
    let dir = temp_dir("closure-generic-two-types");
    let src = fixture(
        &dir,
        "closure_generic_two_types.rs",
        CLOSURE_GENERIC_TWO_TYPES_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "both instantiations of the generic-context closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n3\n");
}

const GENERIC_HOF_FNONCE_SRC: &str = r#"
fn apply<F: FnOnce() -> i32>(f: F) -> i32 { f() }
fn main() {
    let base = 40;
    let r = apply(|| base + 2);
    println!("{r}");
}
"#;

#[test]
fn run_mode_generic_hof_fnonce_with_closure() {
    // A generic higher-order function applied to a closure: the
    // `FnOnce::call_once` call inside the generic body resolves to
    // the closure instance the caller monomorphized it with.
    let dir = temp_dir("generic-hof-fnonce");
    let src = fixture(&dir, "generic_hof_fnonce.rs", GENERIC_HOF_FNONCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic HOF applied to a closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const PACKED_DERIVE_SRC: &str = r#"
use std::mem;

#[derive(Copy, Clone)]
struct Aligned(usize);

#[inline(never)]
fn check_align(ptr: *const Aligned) {
    assert_eq!(ptr as usize % mem::align_of::<Aligned>(), 0);
}

impl PartialEq for Aligned {
    fn eq(&self, other: &Self) -> bool {
        check_align(self);
        check_align(other);
        self.0 == other.0
    }
}

#[repr(packed)]
#[derive(Copy, Clone, PartialEq)]
struct Packed(Aligned, Aligned);

#[derive(PartialEq)]
#[repr(C)]
struct Dealigned<T>(u8, T);

fn main() {
    let d1 = Dealigned(0, Packed(Aligned(1), Aligned(2)));
    let ck = d1 == d1;
    assert!(ck);
    println!("eq-ok");
}
"#;

// ----- coroutine/closure layouts (Lane HN, ex-DW) -----

const COROUTINE_CLOSURE_SIZE_SRC: &str = r#"
async fn f(x: u8) -> u8 { x + 1 }
fn main() {
    let fut = f(3);
    println!("fut {}", std::mem::size_of_val(&fut));
    let a = async { 1u8 };
    let b = async { 2u16 };
    println!("blocks {} {}", std::mem::size_of_val(&a), std::mem::size_of_val(&b));
    let arr = [1u64, 2u64];
    let big = async move { arr[0] };
    println!("big {}", std::mem::size_of_val(&big));
    let x = 5u32;
    let c1 = move || x;
    let c2 = move || x as u64;
    let y = 7u64;
    let c3 = move || y;
    println!("closures {} {} {}", std::mem::size_of_val(&c1), std::mem::size_of_val(&c2), std::mem::size_of_val(&c3));
    println!("tysize {}", std::mem::size_of_val(&c1) + std::mem::size_of::<u32>());
    println!("call {}", c1());
}
"#;

#[test]
fn run_mode_size_of_val_coroutine_and_closure() {
    // The values are the compiler's own layouts (verified against a
    // native 1.100.0 build of the same source): the driver records
    // closure and coroutine layouts per monomorphic type, keyed by
    // the definition name plus generic arguments, and the mem
    // builtins answer from that table. `big` (24) differs from the
    // two unit-capture blocks (1, 1), so distinct coroutine types in
    // one function do not share an entry; likewise the u64-capturing
    // closure (8) differs from the two u32-capturing ones (4, 4), so
    // distinct closure types do not share one either.
    let dir = temp_dir("coroutine-closure-size");
    let src = fixture(&dir, "sizes.rs", COROUTINE_CLOSURE_SIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_of_val of coroutine/closure values runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "fut 2\nblocks 1 1\nbig 24\nclosures 4 4 8\ntysize 8\ncall 5\n",
        "sizes match the native ones"
    );
}

#[test]
fn run_mode_derived_eq_on_packed_struct_uses_align_of() {
    let dir = temp_dir("packed-derive");
    let src = fixture(&dir, "packed_derive.rs", PACKED_DERIVE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived eq over a packed struct runs (align_of::<Aligned> resolves); stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "eq-ok\n");
}

const CLOSURE_CLONE_ZST_CAPTURE_SIDE_EFFECTS_SRC: &str = r#"use std::cell::Cell;
use std::rc::Rc;

struct Zap;

impl Clone for Zap {
    fn clone(&self) -> Self {
        println!("clone Zap");
        Zap
    }
}

impl Drop for Zap {
    fn drop(&mut self) {
        println!("drop Zap");
    }
}

fn main() {
    // A zero-sized capture with an observable Clone side effect
    // and an observable Drop: cloning the closure runs the
    // capture's Clone, and each closure's own drop runs the
    // capture's Drop exactly once (Issue #499).
    let rc = Rc::new(Cell::new(7));
    let rc_in = rc.clone();
    let zap = Zap;
    let f = move || {
        let _ = rc_in.get();
        let _ = &zap;
    };
    let g = f.clone();
    println!("count: {}", Rc::strong_count(&rc));
    drop(g);
    println!("after drop g: {}", Rc::strong_count(&rc));
    drop(f);
    println!("final: {}", Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_closure_clone_zst_capture_side_effects() {
    let dir = temp_dir("closure-clone-zst-capture-side-effects");
    let src = fixture(
        &dir,
        "closure_clone_zst_capture_side_effects.rs",
        CLOSURE_CLONE_ZST_CAPTURE_SIDE_EFFECTS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a zero-sized capture's Clone and Drop must run for a cloned closure; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "clone Zap\ncount: 3\ndrop Zap\nafter drop g: 2\ndrop Zap\nfinal: 1\n"
    );
}

const CLOSURE_CLONE_ZST_CAPTURE_FIRST_SRC: &str = r#"use std::cell::Cell;
use std::rc::Rc;

struct Zap;

impl Clone for Zap {
    fn clone(&self) -> Self {
        println!("clone Zap");
        Zap
    }
}

impl Drop for Zap {
    fn drop(&mut self) {
        println!("drop Zap");
    }
}

struct Boom(i32);

impl Clone for Boom {
    fn clone(&self) -> Self {
        panic!("boom clone");
    }
}

fn main() {
    // The zero-sized capture comes first, then a counted
    // handle, then a panicking capture: the prefix cloned
    // before the panic — the zero-sized capture and the handle
    // — drops during unwinding, in an environment whose field
    // plans must stay aligned across the recovered field
    // (Issue #499).
    let rc = Rc::new(Cell::new(7));
    let rc_in = rc.clone();
    let zap = Zap;
    let boom = Boom(3);
    let f = move || {
        let _ = &zap;
        let _ = rc_in.get();
        let _ = &boom;
    };
    println!("before: {}", Rc::strong_count(&rc));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = f.clone();
        println!("no panic");
    }));
    println!("caught: {}", r.is_err());
    println!("after: {}", Rc::strong_count(&rc));
    drop(f);
    println!("final: {}", Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_closure_clone_zst_capture_first() {
    let dir = temp_dir("closure-clone-zst-capture-first");
    let src = fixture(
        &dir,
        "closure_clone_zst_capture_first.rs",
        CLOSURE_CLONE_ZST_CAPTURE_FIRST_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a leading zero-sized capture must not misalign the cloned prefix's drop glue; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "before: 2\nclone Zap\ndrop Zap\ncaught: true\nafter: 2\ndrop Zap\nfinal: 1\n"
    );
}

/// The shape of tests/ui/recursion/infinite-instantiation-via-
/// nested-closures.rs, reduced: four methods pass the generic
/// parameter along a cycle, and two of the hops substitute a
/// fresh closure type that mentions the previous parameter
/// twice (as the enclosing substitution and as a capture), so
/// the demanded arguments' expansion doubles per cycle while
/// the nesting depth grows only linearly.
const SIZE_INCREASING_CLOSURES_SRC: &str = r#"
#![crate_type = "rlib"]
#[derive(Clone)]
struct A(B);

impl A {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let A(term) = self;
        term.matches(f);
    }
}

#[derive(Clone)]
enum B {
    Variant1,
    Variant2(C),
}

impl B {
    pub fn matches<F: Fn()>(&self, f: &F) {
        match self {
            B::Variant2(factor) => factor.matches(&|| ()),
            _ => unreachable!(""),
        }
    }
}

#[derive(Clone)]
struct C(D);

impl C {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let C(base) = self;
        base.matches(&|| {
            C(base.clone()).matches(f);
        });
    }
}

#[derive(Clone)]
struct D(Box<A>);

impl D {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let D(a) = self;
        a.matches(f);
    }
}

pub fn matches() {
    A(B::Variant1).matches(&|| ());
}
"#;

#[test]
fn convert_mode_size_increasing_closures_is_a_size_limit_failure() {
    // The closure-mediated shape of the same disease: the
    // demanded arguments' expansion doubles per cycle through
    // the closure arguments, so the size cap, not the depth
    // cap, is what stops the collection.
    let dir = temp_dir("mono-size-increasing-closures");
    let src = fixture(&dir, "size_increasing_closures.rs", SIZE_INCREASING_CLOSURES_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "closure-mediated size-increasing instantiation must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating")
            && stderr.contains("(instantiation size limit)"),
        "the failure must name the instantiation size limit; stderr={stderr}"
    );
}

// ----- Conversion type-size limit (Issue #481) -------------------------------

/// The source of the nested-closure exponential: each `dup`
/// call wraps the previous closure, and a closure type spells
/// its captured type twice, so the type tree of the final
/// binding doubles per call.
fn nested_dup_closures_src(depth: usize) -> String {
    let mut src = String::from(
        "fn dup(f: impl Fn(i32) -> i32) -> impl Fn(i32) -> i32 {\n    move |a| f(a * 2)\n}\n\nfn main() {\n    let f = |a| a;\n",
    );
    for _ in 0..depth {
        src.push_str("    let f = dup(f);\n");
    }
    src.push_str("    println!(\"{}\", f(1));\n}\n");
    src
}

#[test]
fn convert_mode_nested_closure_type_explosion_is_bounded() {
    // Thirty levels of nesting spell a type tree of ~2^30
    // nodes for the final binding. No single demand trips the
    // collection size cap before conversion starts
    // materializing that tree, so the conversion size limit is
    // what stops it: the run completes with `main` skipped for
    // that reason instead of growing until the runner's memory
    // bound kills the process (Issue #481).
    let dir = temp_dir("nested-closure-type-explosion");
    let src = fixture(&dir, "nested.rs", &nested_dup_closures_src(30));
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "the conversion must complete and report; status={:?}\nstdout={stdout}",
        out.status
    );
    assert!(
        stdout.contains("expanded type tree exceeds the conversion size limit"),
        "main must be skipped by the conversion size limit:\n{stdout}"
    );
}

#[test]
fn convert_mode_moderate_nested_closures_still_convert() {
    // Ten levels stay far below the limit: the program
    // converts exactly as it does without the bound.
    let dir = temp_dir("nested-closure-moderate");
    let src = fixture(&dir, "nested10.rs", &nested_dup_closures_src(10));
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "status={:?}\nstdout={stdout}",
        out.status
    );
    assert!(
        stdout.contains("::main: blocks=") && stdout.contains("[converted]"),
        "main must convert:\n{stdout}"
    );
    assert!(
        !stdout.contains("conversion size limit"),
        "no type may trip the size limit:\n{stdout}"
    );
}

// ----- Waker construction (Lane DQ) ------------------------------------------

const WAKER_FROM_SRC: &str = r#"
use std::sync::Arc;
use std::task::{Wake, Waker};

struct MyWaker;
impl Wake for MyWaker {
    fn wake(self: Arc<Self>) {}
}

fn main() {
    let waker = Waker::from(Arc::new(MyWaker));
    let waker2 = waker.clone();
    std::mem::forget(waker);
    std::mem::forget(waker2);
    println!("waker ok");
}
"#;

#[test]
fn run_mode_waker_from_arc_constructs_and_clones() {
    // `Waker::from(Arc<T: Wake>)` is interpreter runtime support:
    // the modelled waker carries the Arc value, so constructing
    // it and cloning it (a value copy, exactly how the modelled
    // `Arc` itself clones) run to completion. The wakers are
    // forgotten rather than dropped: dropping one runs the
    // shared-ownership glue the value model does not account
    // for - the same limit as dropping an `Arc` itself - and
    // handing the waker to `Context::from_waker` is blocked by
    // the reference-to-reference transmute in that core body,
    // not by the waker value.
    let dir = temp_dir("wakerfrom");
    let src = fixture(&dir, "wakerfrom.rs", WAKER_FROM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "waker construction and clone must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "waker ok\n",
        "the guest's stdout is the driver's stdout"
    );
}

// ----- coroutine drop glue and runtime support (Lane FF) ---------------------

const COROUTINE_SUSPENDED_DROP_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::Coroutine;
use std::pin::Pin;
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let mut g = #[coroutine] || {
        let d = D(1);
        yield;
        println!("resumed {}", d.0);
    };
    let _s = Pin::new(&mut g).resume(());
    println!("suspended");
    drop(g);
    println!("after drop");
}
"#;

#[test]
fn run_mode_suspended_coroutine_drop_runs_live_local_dtor() {
    // Dropping a coroutine suspended at a yield runs the
    // destructors of the locals live in that state, at the drop
    // point - the state the value is in selects the glue.
    let dir = temp_dir("coroutine-suspended-drop");
    let src = fixture(&dir, "coroutine_suspended_drop.rs", COROUTINE_SUSPENDED_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "suspended coroutine drop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "suspended\ndtor 1\nafter drop\n",
        "the live local's destructor runs when the suspended coroutine is dropped"
    );
}

const COROUTINE_COMPLETED_DROP_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::Coroutine;
use std::pin::Pin;
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let mut g = #[coroutine] || {
        let d = D(1);
        yield;
        println!("resumed {}", d.0);
    };
    let _s = Pin::new(&mut g).resume(());
    let _s2 = Pin::new(&mut g).resume(());
    println!("completed");
    drop(g);
    println!("after drop");
}
"#;

#[test]
fn run_mode_completed_coroutine_drop_runs_dtor_once() {
    // A completed coroutine's locals were already dropped by the
    // coroutine's own completion path; dropping the exhausted
    // value afterwards runs nothing a second time.
    let dir = temp_dir("coroutine-completed-drop");
    let src = fixture(&dir, "coroutine_completed_drop.rs", COROUTINE_COMPLETED_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "completed coroutine drop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "resumed 1\ndtor 1\ncompleted\nafter drop\n",
        "the destructor runs exactly once, inside the completion"
    );
}

const COROUTINE_UNRESUMED_DROP_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let d = D(9);
    let g = #[coroutine] move || { let dd = d; yield; println!("inner {}", dd.0); };
    println!("created");
    drop(g);
    println!("after drop");
}
"#;

#[test]
fn run_mode_unresumed_coroutine_drop_runs_capture_dtor() {
    // Dropping a coroutine that was never resumed drops the
    // values it captured, at the drop point.
    let dir = temp_dir("coroutine-unresumed-drop");
    let src = fixture(&dir, "coroutine_unresumed_drop.rs", COROUTINE_UNRESUMED_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "unresumed coroutine drop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "created\ndtor 9\nafter drop\n",
        "the captured value's destructor runs when the unresumed coroutine is dropped"
    );
}

const ASYNC_SIZE_OF_VAL_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

struct Nop;
impl Future for Nop {
    type Output = u8;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u8> { Poll::Ready(3) }
}

async fn one() -> u8 { Nop.await }

fn main() {
    println!("size {}", std::mem::size_of_val(&one()));
    println!("size {}", std::mem::size_of_val(&Nop));
}
"#;

#[test]
fn run_mode_size_of_val_on_async_future_uses_compiler_layout() {
    // `size_of_val` on a future answers with the compiler's own
    // state-machine layout for that coroutine instance (the same
    // values a native build prints), not with a model guess.
    let dir = temp_dir("async-size-of-val");
    let src = fixture(&dir, "async_size_of_val.rs", ASYNC_SIZE_OF_VAL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_of_val on a future must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "size 1\nsize 0\n",
        "the future's size is the compiler layout size"
    );
}

// ----- one closure instance demanded by two collectors (Lane HM) -----------

const SUM_CLOSURE_SRC: &str = r#"
fn main() {
    let v: [i128; 3] = [10, 20, 12];
    let s: i128 = v.iter().sum();
    assert_eq!(s, 42);
    println!("ok");
}
"#;

#[test]
fn run_mode_closure_demanded_by_two_collectors_converts_once() {
    // `<i128 as Sum<&i128>>::sum` creates a closure and hands it
    // to `Iterator::fold`. Two collectors demand that one closure
    // instance: the reachability collection resolves the closures
    // in a call's generic arguments itself (the callee is
    // builtin-owned, so its body is excluded), and the closure
    // demand pass finds the same closure mentioned in the
    // collected `sum` body. Both used to convert it, the program
    // held two bodies under one DefId, and conversion failed with
    // "duplicate function name
    // `<i128 as Sum<&'a i128>>::sum::{closure#0}`" (the
    // tests/ui/numbers-arithmetic/i128.rs shape). The demand
    // passes now skip an instance another collector already
    // converted, and the reachability merges drop duplicates by
    // DefId, so the closure enters the program exactly once.
    let dir = temp_dir("sum-closure");
    let src = fixture(&dir, "sum_closure.rs", SUM_CLOSURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the doubly-demanded closure must convert once and run; stderr={stderr}"
    );
    assert!(
        !stderr.contains("duplicate function name"),
        "no duplicate registration; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const ASSERT_INHABITED_SRC: &str = r#"
#![feature(core_intrinsics)]
fn main() {
    unsafe {
        std::intrinsics::assert_inhabited::<u32>();
        std::intrinsics::assert_inhabited::<[u8; 3]>();
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_assert_inhabited_is_a_no_op() {
    let dir = temp_dir("assert-inhabited");
    let src = fixture(&dir, "assert_inhabited.rs", ASSERT_INHABITED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "assert_inhabited must run as a no-op; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}


const ALLOC_CYCLE_SRC: &str = r#"
use std::alloc::{alloc, alloc_zeroed, dealloc, realloc, Layout};
fn main() {
    unsafe {
        let layout = Layout::array::<u32>(4).unwrap();
        let p = alloc(layout) as *mut u32;
        assert!(!p.is_null());
        for i in 0..4 { p.add(i).write(10 + i as u32); }
        for i in 0..4 { assert_eq!(*p.add(i), 10 + i as u32); }
        let layout8 = Layout::array::<u32>(8).unwrap();
        let p = realloc(p as *mut u8, layout, layout8.size()) as *mut u32;
        assert!(!p.is_null());
        for i in 0..4 { assert_eq!(*p.add(i), 10 + i as u32); }
        dealloc(p as *mut u8, layout8);
        let zlayout = Layout::array::<u8>(3).unwrap();
        let z = alloc_zeroed(zlayout);
        assert!(!z.is_null());
        assert_eq!(*z, 0);
        assert_eq!(*z.add(2), 0);
        dealloc(z, zlayout);
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_raw_allocator_cycle() {
    let dir = temp_dir("alloc-cycle");
    let src = fixture(&dir, "alloc_cycle.rs", ALLOC_CYCLE_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const POW_PARSE_SRC: &str = r#"
fn main() {
    assert_eq!(2i32.pow(10), 1024);
    assert_eq!((-3i64).pow(3), -27);
    assert_eq!(7u32.pow(0), 1);
    let x: f64 = "3.1415".parse().unwrap();
    assert!(x > 3.1414 && x < 3.1416);
    let n: i32 = "-42".parse().unwrap();
    assert_eq!(n, -42);
    assert!("abc".parse::<i32>().is_err());
    assert!("99999999999999999999".parse::<i32>().is_err());
    let b: bool = "true".parse().unwrap();
    assert!(b);
    println!("ok");
}
"#;

#[test]
fn run_mode_pow_and_from_str_parse() {
    let dir = temp_dir("pow-parse");
    let src = fixture(&dir, "pow_parse.rs", POW_PARSE_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const HAND_BUILT_WAKER_SRC: &str = r#"
use std::sync::Arc;
use std::task::{RawWaker, RawWakerVTable, Waker};

struct Wake;

unsafe fn clone_raw(data: *const ()) -> RawWaker {
    let arc = Arc::from_raw(data as *const Wake);
    let cloned = arc.clone();
    let _ = Arc::into_raw(arc);
    let _ = Arc::into_raw(cloned);
    RawWaker::new(data, &VTABLE)
}

unsafe fn wake_raw(data: *const ()) {
    let arc = Arc::from_raw(data as *const Wake);
    drop(arc);
}

unsafe fn wake_by_ref_raw(_data: *const ()) {}

unsafe fn drop_raw(data: *const ()) {
    let arc = Arc::from_raw(data as *const Wake);
    drop(arc);
}

static VTABLE: RawWakerVTable =
    RawWakerVTable::new(clone_raw, wake_raw, wake_by_ref_raw, drop_raw);

fn main() {
    // A waker built by hand from a raw `Arc` (the `arc_wake`
    // pattern): dropping it releases the handle the data
    // pointer denotes, so the weak handle observes the strong
    // count reaching zero.
    let arc = Arc::new(Wake);
    let weak = Arc::downgrade(&arc);
    let raw = Arc::into_raw(arc) as *const ();
    let waker = unsafe { Waker::from_raw(RawWaker::new(raw, &VTABLE)) };
    assert!(weak.upgrade().is_some());
    drop(waker);
    println!("released={}", weak.upgrade().is_none());
}
"#;

#[test]
fn run_mode_hand_built_waker_drop_releases_arc() {
    let dir = temp_dir("hand-built-waker");
    let src = fixture(&dir, "hand_built_waker.rs", HAND_BUILT_WAKER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "released=true\n");
}

const COROUTINE_DROP_ENV_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
#![allow(dropping_copy_types)]

use std::ops::Coroutine;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

static A: AtomicUsize = AtomicUsize::new(0);

struct B;

impl Drop for B {
    fn drop(&mut self) {
        A.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    let b = B;
    let mut foo = #[coroutine] || {
        yield;
        drop(b);
    };
    let n = A.load(Ordering::SeqCst);
    drop(Pin::new(&mut foo).resume(()));
    assert_eq!(A.load(Ordering::SeqCst), n);
    drop(foo);
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
    println!("coroutine drop env ok");
}
"#;

#[test]
fn run_mode_suspended_coroutine_drop_runs_capture_dtor() {
    // Dropping a suspended coroutine drops the captures it
    // still holds: the coroutine's storage is one field list
    // shared by every state, so the suspended state's glue is
    // the unresumed state's fields (plus its own saved locals),
    // not the empty plan an unevidenced variant would get.
    let dir = temp_dir("coroutine-drop-env");
    let src = fixture(&dir, "coroutine_drop_env.rs", COROUTINE_DROP_ENV_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a suspended coroutine's drop must run its captures' destructors; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "coroutine drop env ok\n");
}

const BOXED_FUTURE_PARAM_DROP_SRC: &str = r#"
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

static DROPS: AtomicUsize = AtomicUsize::new(0);

struct D;

impl Drop for D {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            Poll::Pending
        }
    }
}

async fn foo(x: D, _y: D) {
    let _ = &x;
    YieldOnce(false).await;
}

fn dummy_waker() -> Waker {
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VT)
    }
    fn noop(_: *const ()) {}
    static VT: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VT)) }
}

fn main() {
    let mut fut = Box::pin(foo(D, D));
    let w = dummy_waker();
    let mut cx = Context::from_waker(&w);
    let r = fut.as_mut().poll(&mut cx);
    assert!(matches!(r, Poll::Pending));
    assert_eq!(DROPS.load(Ordering::SeqCst), 0);
    drop(fut);
    assert_eq!(DROPS.load(Ordering::SeqCst), 2);
    println!("boxed future param drop ok");
}
"#;

#[test]
fn run_mode_boxed_suspended_future_drops_parameters_once() {
    // A boxed future's prologue moves its parameters out of the
    // environment (the allocation's upvar storage); the slots
    // clear as they are moved from, so when the suspended future
    // is dropped, its glue drops each parameter exactly once -
    // through the saved storage it now lives in, not again
    // through the environment.
    let dir = temp_dir("boxed-future-param-drop");
    let src = fixture(&dir, "boxed_future_param_drop.rs", BOXED_FUTURE_PARAM_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a boxed suspended future must drop its parameters exactly once; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "boxed future param drop ok\n"
    );
}

const ASYNC_DROP_BOX_ALLOCATOR_SRC: &str = r#"
#![feature(async_drop)]
#![allow(incomplete_features)]

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::future::{async_drop_in_place, AsyncDrop, Future};
use std::pin::{pin, Pin};
use std::ptr::NonNull;
use std::task::{Context, Poll, Waker};

struct Foo {
    handle: usize,
}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("Foo::drop() : {}", self.handle);
    }
}

impl AsyncDrop for Foo {
    async fn drop(self: Pin<&mut Self>) {
        println!("Foo::async drop() : {}", self.handle);
    }
}

unsafe impl Allocator for Foo {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        Global.deallocate(ptr, layout);
    }
}

struct HasDrop;
impl Drop for HasDrop {
    fn drop(&mut self) {}
}

async fn bar(ident: usize) {
    let b = Box::new_in(HasDrop, Foo { handle: ident });
    if true {
        let _x = *b;
    } else {
        let _y = b;
    }
}

fn block_on<F: Future>(fut: F) -> F::Output {
    let (waker, _rx) = simple_waker();
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(fut);
    let out = loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(out) => break out,
            Poll::Pending => std::thread::yield_now(),
        }
    };
    let drop_fut = unsafe { async_drop_in_place(fut.get_unchecked_mut()) };
    let mut drop_fut = pin!(drop_fut);
    loop {
        match drop_fut.as_mut().poll(&mut cx) {
            Poll::Ready(()) => break,
            Poll::Pending => std::thread::yield_now(),
        }
    }
    out
}

fn simple_waker() -> (Waker, std::sync::mpsc::Receiver<()>) {
    struct SimpleWaker {
        tx: std::sync::mpsc::Sender<()>,
    }
    impl std::task::Wake for SimpleWaker {
        fn wake(self: std::sync::Arc<Self>) {
            self.tx.send(()).unwrap();
        }
    }
    let (tx, rx) = std::sync::mpsc::channel();
    (Waker::from(std::sync::Arc::new(SimpleWaker { tx })), rx)
}

fn main() {
    block_on(bar(10));
    println!("Done");
}
"#;

#[test]
fn run_mode_async_drop_box_allocator_state_discriminant() {
    // The async-drop state machine of `bar` saves a `bool` in
    // two states at one layout offset and a `()` in a third
    // state at the same offset, where the `()` is that offset's
    // only occupant. The poll body stores the `()` for the
    // state it is entering and then dispatches on the `bool`
    // through the other state's view; with the offset's shared
    // slot for both, the unit store overwrote the bool and the
    // dispatch stopped with "type mismatch in SwitchInt
    // discriminant: expected integer or bool, found unit".
    // Provably zero-sized state fields now convert to private
    // slots, so the dispatch reads the bool. The run then
    // reaches the end-of-body drop sequence, whose first step
    // moves the pointee out of the box (`let _x = *b`): the
    // extracted raw pointer came from a `BoxDerefTransmute`
    // over the box's `Unique` field inside the coroutine
    // state, a place the cast's classification cannot name,
    // and the wrapper-chain descent stopped at the `Unique`
    // layer (it carries a `PhantomData` sibling), so the cast
    // fabricated a reference to the `Unique` field itself and
    // the move-out cleared the box's pointer word; the box's
    // drop glue then read that word as uninitialized. The
    // descent now also proceeds when the value's own field-0
    // chain verifiably bottoms out at a pointer, so the
    // extracted pointer names the allocation, the move-out
    // clears the pointee, and the whole run completes with
    // the native stdout.
    let dir = temp_dir("async-drop-box-allocator-discriminant");
    let src = fixture(&dir, "async_drop_box_allocator.rs", ASYNC_DROP_BOX_ALLOCATOR_SRC);
    let out = run_driver_extra(&src, &["--edition", "2021"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("SwitchInt discriminant"),
        "the state dispatch must read the saved bool, not the unit stored for another state; stderr={stderr}"
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "the run must complete through the async-drop sequence; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Foo::async drop() : 10\nDone\n",
        "the run must print the native stdout; stderr={stderr}"
    );
}
