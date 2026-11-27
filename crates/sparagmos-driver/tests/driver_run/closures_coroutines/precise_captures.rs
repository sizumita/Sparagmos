// Precise (deref) captures of async closures: the closure
// environment holds one reference to an aggregate, and the
// coroutine a call creates borrows the aggregate's fields
// disjointly through it. The created future's fields are the
// per-field references, formed at the `AsyncFn` call from the
// capture plan conversion records.

const PRECISE_CAPTURE_MUT_REF_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());
    loop {
        match unsafe { fut.as_mut().poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

#[derive(Debug)]
struct Drop(&'static str);
impl std::ops::Drop for Drop {
    fn drop(&mut self) { println!("{}", self.0); }
}
struct S { a: i32, b: Drop, c: Drop }

async fn async_main() {
    let s = &mut S { a: 1, b: Drop("fix me up"), c: Drop("untouched") };
    let mut c = async || {
        s.a = 2;
        let w = &mut s.b;
        w.0 = "fixed";
    };
    s.c.0 = "uncaptured";
    let fut = c();
    println!("after call");
    fut.await;
    println!("after await");
}

fn main() { block_on(async_main()); }
"#;

#[test]
fn run_mode_async_closure_precise_capture_through_mut_ref() {
    // The async closure captures `s.a` and `s.b` disjointly
    // through the `&mut S` it is given: its environment is the
    // single reference, and the future the call creates must
    // hold one reference per captured field, each formed
    // through the environment reference - a future built from
    // the environment verbatim has no second field for the
    // poll to read.
    let dir = temp_dir("async-precise-capture-mut-ref");
    let src = fixture(&dir, "precise_capture_mut_ref.rs", PRECISE_CAPTURE_MUT_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "precise capture through a &mut must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after call\nafter await\nfixed\nuncaptured\n",
        "both captured fields are written through the derived references"
    );
}

const PRECISE_CAPTURE_TUPLE_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());
    loop {
        match unsafe { fut.as_mut().poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

#[derive(Debug)]
struct Drop(&'static str);
impl std::ops::Drop for Drop {
    fn drop(&mut self) { println!("{}", self.0); }
}

async fn async_main() {
    let t = &mut (1, Drop("fix me up"), Drop("untouched"));
    let mut c = async || {
        t.0 = 2;
        let w = &mut t.1;
        w.0 = "fixed";
    };
    t.2.0 = "uncaptured";
    let fut = c();
    println!("after call");
    fut.await;
    println!("after await");
}

fn main() { block_on(async_main()); }
"#;

#[test]
fn run_mode_async_closure_precise_capture_through_tuple_ref() {
    // The same deref capture with a tuple pointee: the
    // environment reference's pointee fields are the tuple's
    // elements, matched in order.
    let dir = temp_dir("async-precise-capture-tuple");
    let src = fixture(&dir, "precise_capture_tuple.rs", PRECISE_CAPTURE_TUPLE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "precise capture through a &mut tuple must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after call\nafter await\nfixed\nuncaptured\n",
        "both captured elements are written through the derived references"
    );
}

const PINNED_SAVED_READBACK_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

struct YieldOnce(bool);
impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        if self.0 { Poll::Ready(()) } else { self.0 = true; Poll::Pending }
    }
}

async fn compute() -> i32 {
    let saved = 40;
    YieldOnce(false).await;
    saved + 2
}

fn main() {
    let mut boxed = Box::pin(compute());
    let ctx = &mut Context::from_waker(Waker::noop());
    match boxed.as_mut().poll(ctx) {
        Poll::Pending => println!("pending"),
        Poll::Ready(v) => println!("ready {v}"),
    }
    match boxed.as_mut().poll(ctx) {
        Poll::Pending => println!("pending"),
        Poll::Ready(v) => println!("ready {v}"),
    }
}
"#;

#[test]
fn run_mode_pinned_coroutine_saved_field_read_back_after_resume() {
    // A local written before the suspension point is read back
    // from the coroutine's saved storage after the resume, with
    // the future driven through `Pin::as_mut` on a box - the
    // poll path the coverage executor records use.
    let dir = temp_dir("pinned-saved-readback");
    let src = fixture(&dir, "pinned_saved_readback.rs", PINNED_SAVED_READBACK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "saved field read-back must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "pending\nready 42\n",
        "the saved local survives the suspension and is read back"
    );
}

const PRECISE_CAPTURE_ONCE_MOVE_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());
    loop {
        match unsafe { fut.as_mut().poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

#[derive(Debug)]
struct Drop(&'static str);
impl std::ops::Drop for Drop {
    fn drop(&mut self) { println!("{}", self.0); }
}
struct S { a: i32, b: Drop, c: Drop }

async fn call_once(f: impl AsyncFnOnce()) { f().await }

async fn async_main() {
    let s = &mut S { a: 1, b: Drop("fix me up"), c: Drop("untouched") };
    let mut c = async move || {
        s.a = 2;
        let w = &mut s.b;
        w.0 = "fixed";
    };
    let fut = call_once(c);
    println!("after call");
    fut.await;
    println!("after await");
}

fn main() { block_on(async_main()); }
"#;

#[test]
fn run_mode_async_closure_precise_capture_once_move() {
    // The same deref capture as above, but the closure is an
    // `async move` closure consumed by an `AsyncFnOnce` call
    // through a generic shim: the future the call creates
    // holds the environment verbatim (the single `&mut S`),
    // because the once future's poll body reads the
    // environment reference itself and projects the pointee's
    // fields through it. Expanding the environment into
    // per-field references at creation - the shape the
    // borrowed `AsyncFn` calls create - puts a per-field
    // reference where the poll body expects the environment
    // reference, and its field projections land one field off
    // (the write to `s.b` composes onto the `s.a` reference's
    // target and the run stops with a field-write type
    // mismatch). The reference the body's writes go through is
    // rooted at the enclosing coroutine's self place (the
    // struct lives in `async_main`'s saved storage, reached
    // through the `Pin` in the outer frame).
    let dir = temp_dir("async-precise-capture-once-move");
    let src = fixture(&dir, "precise_capture_once_move.rs", PRECISE_CAPTURE_ONCE_MOVE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "precise capture through a once call must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after call\nafter await\nfixed\nuntouched\n",
        "both captured fields are written through the environment reference"
    );
}
const PRECISE_CAPTURE_MOVE_MUT_DIRECT_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());
    loop {
        match unsafe { fut.as_mut().poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

#[derive(Debug)]
struct Drop(&'static str);
impl std::ops::Drop for Drop {
    fn drop(&mut self) { println!("{}", self.0); }
}
struct S { a: i32, b: Drop, c: Drop }

async fn async_main() {
    let s = &mut S { a: 1, b: Drop("fix me up"), c: Drop("untouched") };
    let mut c = async move || {
        s.a = 2;
        let w = &mut s.b;
        w.0 = "fixed";
    };
    let fut = c();
    println!("after call");
    fut.await;
    println!("after await");
}

fn main() { block_on(async_main()); }
"#;

#[test]
fn run_mode_async_closure_precise_capture_move_mut_direct_call() {
    // The `async move` counterpart of the deref capture above,
    // called directly (an `AsyncFnMut` call): the environment
    // is the single `&mut S`, and no body in the program
    // projects `S`'s fields - their accesses live in the
    // closure's own body - so the capture plan's field types
    // must come from the declared shape, not only from the
    // projection-recovered table. Without a plan the created
    // future holds the environment verbatim and its second
    // field reads uninitialised.
    let dir = temp_dir("async-precise-capture-move-mut-direct");
    let src = fixture(
        &dir,
        "precise_capture_move_mut_direct.rs",
        PRECISE_CAPTURE_MOVE_MUT_DIRECT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "precise capture of a moved &mut, called directly, must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after call\nafter await\nfixed\nuntouched\n",
        "both captured fields are written through the derived references"
    );
}

const PRECISE_CAPTURE_MOVE_MUT_SHIM_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());
    loop {
        match unsafe { fut.as_mut().poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

#[derive(Debug)]
struct Drop(&'static str);
impl std::ops::Drop for Drop {
    fn drop(&mut self) { println!("{}", self.0); }
}
struct S { a: i32, b: Drop, c: Drop }

async fn call_mut(f: &mut impl AsyncFnMut()) { f().await }

async fn async_main() {
    let s = &mut S { a: 1, b: Drop("fix me up"), c: Drop("untouched") };
    let mut c = async move || {
        s.a = 2;
        let w = &mut s.b;
        w.0 = "fixed";
    };
    let fut = call_mut(&mut c);
    println!("after call");
    fut.await;
    println!("after await");
}

fn main() { block_on(async_main()); }
"#;

#[test]
fn run_mode_async_closure_precise_capture_move_mut_shim_call() {
    // The same moved-`&mut` deref capture reached through a
    // generic `AsyncFnMut` shim instead of a direct call: the
    // plan computation does not depend on the call shape.
    let dir = temp_dir("async-precise-capture-move-mut-shim");
    let src = fixture(
        &dir,
        "precise_capture_move_mut_shim.rs",
        PRECISE_CAPTURE_MOVE_MUT_SHIM_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "precise capture of a moved &mut through a shim must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after call\nafter await\nfixed\nuntouched\n",
        "both captured fields are written through the derived references"
    );
}

const PRECISE_CAPTURE_MOVE_GENERIC_POINTEE_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());
    loop {
        match unsafe { fut.as_mut().poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

#[derive(Debug)]
struct Drop(&'static str);
impl std::ops::Drop for Drop {
    fn drop(&mut self) { println!("{}", self.0); }
}
struct W<T> { x: T, d: Drop, z: Drop }

async fn call_mut(f: &mut impl AsyncFnMut()) { f().await }

async fn async_main() {
    let w = &mut W { x: 3i32, d: Drop("dee"), z: Drop("zee") };
    let mut c = async move || {
        w.x = 4;
        let r = &mut w.d;
        r.0 = "dee fixed";
    };
    let fut = call_mut(&mut c);
    println!("after call");
    fut.await;
    println!("after await");
}

fn main() { block_on(async_main()); }
"#;

#[test]
fn run_mode_async_closure_precise_capture_move_generic_pointee() {
    // The pointee is a generic instantiation (`W<i32>`): the
    // shape fallback must substitute the instance's arguments
    // for the definition's parameters before the borrowed
    // types can match the declared fields.
    let dir = temp_dir("async-precise-capture-move-generic");
    let src = fixture(
        &dir,
        "precise_capture_move_generic.rs",
        PRECISE_CAPTURE_MOVE_GENERIC_POINTEE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "precise capture through a generic pointee must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after call\nafter await\ndee fixed\nzee\n",
        "both captured fields are written through the derived references"
    );
}

const PRECISE_CAPTURE_MOVE_TWO_REFS_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());
    loop {
        match unsafe { fut.as_mut().poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

#[derive(Debug)]
struct Drop(&'static str);
impl std::ops::Drop for Drop {
    fn drop(&mut self) { println!("{}", self.0); }
}
struct S { a: i32, b: Drop, c: Drop }
struct T { x: i32, y: Drop }

async fn async_main() {
    let s = &mut S { a: 1, b: Drop("fix me up"), c: Drop("untouched") };
    let t = &mut T { x: 10, y: Drop("tee") };
    let mut c = async move || {
        s.a = 2;
        let w = &mut s.b;
        w.0 = "fixed";
        t.x = 11;
        let u = &mut t.y;
        u.0 = "tee fixed";
    };
    let fut = c();
    println!("after call");
    fut.await;
    println!("after await");
}

fn main() { block_on(async_main()); }
"#;

#[test]
fn run_mode_async_closure_precise_capture_move_two_refs() {
    // Two reference captures in one closure, each fanning out
    // over its own pointee: the plan is per-capture, and the
    // shape fallback serves every capture's pointee, not only
    // the first.
    let dir = temp_dir("async-precise-capture-move-two-refs");
    let src = fixture(
        &dir,
        "precise_capture_move_two_refs.rs",
        PRECISE_CAPTURE_MOVE_TWO_REFS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "two fanned-out reference captures must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after call\nafter await\ntee fixed\nfixed\nuntouched\n",
        "all captured fields are written through the derived references"
    );
}
