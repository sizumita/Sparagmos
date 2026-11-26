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
