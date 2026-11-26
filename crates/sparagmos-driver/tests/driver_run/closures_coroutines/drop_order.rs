// Coroutine drop order: cancelled and unresumed coroutines drop locals and upvars in source order.

const COROUTINE_CANCEL_DROP_ORDER_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake};

struct Nop;
impl Wake for Nop { fn wake(self: Arc<Self>) {} }

struct D(&'static str);
impl Drop for D { fn drop(&mut self) { println!("drop {}", self.0); } }

struct Never;
impl Future for Never {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> { Poll::Pending }
}

async fn f(p: D) {
    let x = D("x");
    let y = D("y");
    Never.await;
    drop(p); drop(x); drop(y);
}

fn main() {
    let waker = std::task::Waker::from(Arc::new(Nop));
    let mut cx = Context::from_waker(&waker);
    let mut fut = Box::pin(f(D("p")));
    let _ = fut.as_mut().poll(&mut cx);
    drop(fut);
    println!("done");
}
"#;

#[test]
fn run_mode_suspended_coroutine_drops_locals_in_reverse_order() {
    // Dropping a suspended coroutine drops the locals live
    // across the suspension point in reverse declaration
    // order - the order the body's own drop elaboration
    // would use if the function returned there - with the
    // parameters (the state's first fields) last. Verified
    // against a native 1.100.0 build of the same source.
    run_ok_edition(
        "coroutine-cancel-drop-order",
        COROUTINE_CANCEL_DROP_ORDER_SRC,
        "drop y\ndrop x\ndrop p\ndone\n",
    );
}

const COROUTINE_UNRESUMED_DROP_ORDER_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake};

struct Nop;
impl Wake for Nop { fn wake(self: Arc<Self>) {} }

struct D(&'static str);
impl Drop for D { fn drop(&mut self) { println!("drop {}", self.0); } }

struct Never;
impl Future for Never {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> { Poll::Pending }
}

async fn g(x: D, y: D) {
    Never.await;
    drop(x); drop(y);
}

fn main() {
    let fut = Box::pin(g(D("x"), D("y")));
    drop(fut);
    println!("done");
}
"#;

#[test]
fn run_mode_unresumed_coroutine_drops_upvars_in_field_order() {
    // A coroutine dropped before its first poll holds only
    // its upvars, which are the value's own fields and drop
    // in field order, like a struct's fields - the reverse
    // order of the suspended states does not apply here.
    // Verified against a native 1.100.0 build of the same
    // source.
    run_ok_edition(
        "coroutine-unresumed-drop-order",
        COROUTINE_UNRESUMED_DROP_ORDER_SRC,
        "drop x\ndrop y\ndone\n",
    );
}
