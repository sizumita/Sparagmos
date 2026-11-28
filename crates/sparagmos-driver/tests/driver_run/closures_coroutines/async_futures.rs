// Async futures: size_of_val over futures, suspended boxed futures dropping parameters, and async drop with a boxed allocator.

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

const ASYNC_CLOSURE_UNPOLLED_DROP_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

struct DropMe(i32);
impl Drop for DropMe {
    fn drop(&mut self) {
        println!("{} was dropped", self.0);
    }
}

fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(ctx) {
            return v;
        }
    }
}

async fn call_once(f: impl AsyncFnOnce()) {
    println!("before call");
    let fut = Box::pin(f());
    println!("after call");
    drop(fut);
    println!("future dropped");
}

fn main() {
    block_on(async {
        let d = DropMe(42);
        let async_closure = async move || {
            let d = &d;
            println!("called");
        };

        call_once(async_closure).await;
        println!("after");
    });
}
"#;

#[test]
fn run_mode_unpolled_async_closure_future_drops_capture() {
    // The future an `async_call_once` creates is dropped without
    // ever being polled: its captures, moved into the future's
    // unresumed state, must drop there. The closure body is never
    // converted, so no projection evidence names the unresumed
    // state's fields; the drop plan comes from the future type's
    // own declared state-0 fields (the call arguments followed by
    // the captures). Mirrors tests/ui/async-await/async-closures/
    // drop.rs, whose expected stdout this reproduces.
    let dir = temp_dir("async-closure-unpolled-drop");
    let src = fixture(&dir, "async_closure_unpolled_drop.rs", ASYNC_CLOSURE_UNPOLLED_DROP_SRC);
    let out = run_driver_extra(&src, &["--edition", "2021"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the run must complete; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "before call\nafter call\n42 was dropped\nfuture dropped\nafter\n",
        "the unpolled future's capture must drop exactly once, in order; stderr={stderr}"
    );
}

const ASYNC_CLOSURE_VALUE_DROP_SRC: &str = r#"
struct DropMe(i32);
impl Drop for DropMe {
    fn drop(&mut self) {
        println!("{} was dropped", self.0);
    }
}

fn main() {
    let d = DropMe(43);
    let c = async move || {
        let r = &d;
        println!("called {}", r.0);
    };
    drop(c);
    println!("closure dropped");
}
"#;

#[test]
fn run_mode_dropped_async_closure_value_drops_capture() {
    // An explicit `std::mem::drop` of an async closure value: the
    // value's place type is the unit placeholder (the
    // coroutine-closure type's kind is unaskable through Stable
    // MIR), so the drop glue is planned from the body's creation
    // aggregate, as the `Drop` terminator conversion already does
    // for a placeholder-typed place.
    let dir = temp_dir("async-closure-value-drop");
    let src = fixture(&dir, "async_closure_value_drop.rs", ASYNC_CLOSURE_VALUE_DROP_SRC);
    let out = run_driver_extra(&src, &["--edition", "2021"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the run must complete; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "43 was dropped\nclosure dropped\n",
        "the dropped closure's capture must drop inside the drop; stderr={stderr}"
    );
}

const ASYNC_CLOSURE_POLLED_DROP_SRC: &str = r#"
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

struct DropMe(i32);
impl Drop for DropMe {
    fn drop(&mut self) {
        println!("{} was dropped", self.0);
    }
}

struct PendingOnce {
    done: bool,
}
impl Future for PendingOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        if self.done {
            Poll::Ready(())
        } else {
            self.done = true;
            Poll::Pending
        }
    }
}

use std::pin::Pin;

async fn call_once_susp(f: impl AsyncFnOnce()) {
    let mut fut = Box::pin(f());
    let ctx = &mut Context::from_waker(Waker::noop());
    match fut.as_mut().poll(ctx) {
        Poll::Pending => println!("pending"),
        Poll::Ready(()) => println!("ready"),
    }
    drop(fut);
    println!("future dropped");
}

async fn call_once_done(f: impl AsyncFnOnce()) {
    let mut fut = Box::pin(f());
    let ctx = &mut Context::from_waker(Waker::noop());
    loop {
        match fut.as_mut().poll(ctx) {
            Poll::Pending => println!("pending"),
            Poll::Ready(()) => break,
        }
    }
    drop(fut);
    println!("future dropped");
}

fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(ctx) {
            return v;
        }
    }
}

fn main() {
    println!("before susp");
    let d = DropMe(47);
    block_on(call_once_susp(async move || {
        let r = &d;
        println!("start {}", r.0);
        PendingOnce { done: false }.await;
        println!("resumed {}", r.0);
    }));
    println!("after susp");
    println!("before done");
    let e = DropMe(48);
    block_on(call_once_done(async move || {
        let r = &e;
        println!("ran {}", r.0);
    }));
    println!("after done");
}
"#;

#[test]
fn run_mode_polled_async_closure_future_drops_capture_once() {
    // The suspended and completed states of an `async_call_once`
    // future: a suspended future's capture drops when the future
    // is dropped; a completed future's capture already dropped
    // inside the final poll and must not drop again. Guards the
    // unresumed-state planning against double-drops in the states
    // that were already dropped correctly.
    let dir = temp_dir("async-closure-polled-drop");
    let src = fixture(&dir, "async_closure_polled_drop.rs", ASYNC_CLOSURE_POLLED_DROP_SRC);
    let out = run_driver_extra(&src, &["--edition", "2021"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the run must complete; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "before susp\nstart 47\npending\n47 was dropped\nfuture dropped\nafter susp\nbefore done\nran 48\n48 was dropped\nfuture dropped\nafter done\n",
        "each capture must drop exactly once; stderr={stderr}"
    );
}

const NESTED_ASYNC_GEN_SRC: &str = r#"
#![allow(incomplete_features)]
#![feature(async_iterator, gen_blocks, move_expr)]

use std::async_iter::AsyncIterator;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

fn poll_next<I: AsyncIterator>(iter: Pin<&mut I>) -> Poll<Option<I::Item>> {
    let cx = &mut Context::from_waker(Waker::noop());
    AsyncIterator::poll_next(iter, cx)
}

fn ready_next<I: AsyncIterator>(iter: Pin<&mut I>) -> Option<I::Item> {
    match poll_next(iter) {
        Poll::Ready(item) => item,
        Poll::Pending => panic!("async iterator unexpectedly returned pending"),
    }
}

fn main() {
    let y = Arc::new(String::from("nested"));
    let weak = Arc::downgrade(&y);
    let mut iter = Box::pin(async gen {
        let mut inner = Box::pin(async gen {
            let value = move(move(y.clone()));
            yield Arc::strong_count(&value);
        });
        yield ready_next(inner.as_mut()).unwrap();
    });
    assert_eq!(weak.strong_count(), 2);
    assert_eq!(ready_next(iter.as_mut()), Some(2));
    drop(iter);
    assert_eq!(weak.strong_count(), 1);
    println!("nested async gen ok");
}
"#;

#[test]
fn run_mode_nested_async_gen_boxed_in_gen_body() {
    // A `Box::pin`ed async gen block created inside another
    // async gen block's body: the pinned value is a local saved
    // as a field of the outer coroutine's state, and
    // `Pin::as_mut` on it reaches `DerefMut::deref_mut` for
    // `Box` with the box denominated through that state field's
    // projection. The deref stopped with "type mismatch in
    // DerefMut::deref_mut for Box: expected boxed value, found
    // coroutine" (tests/ui/move-expr/async-gen-blocks.rs).
    let dir = temp_dir("nested-async-gen");
    let src = fixture(&dir, "nested_async_gen.rs", NESTED_ASYNC_GEN_SRC);
    let out = run_driver_extra(&src, &["--edition", "2024"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested async gen must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "nested async gen ok\n"
    );
}
