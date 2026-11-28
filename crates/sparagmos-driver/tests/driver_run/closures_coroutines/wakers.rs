// Waker construction: Waker::from(Arc) and hand-built RawWaker vtables.

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

// ----- Closure-literal RawWakerVTable entries (issue-65419) ------------------
//
// The executor in rustc's issue-65419 tests builds its vtable
// from closure literals, so every entry reaches the interpreter
// as a `Closure` value rather than a `FnPtr`. Dropping (or
// cloning) such a waker must call the entry with its environment
// argument, exactly as it calls a function-pointer entry.

const CLOSURE_VTABLE_RESUME_AFTER_COMPLETION_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

static DROPS: AtomicUsize = AtomicUsize::new(0);

async fn foo() {}

fn block_on<F: Future>(mut future: F) -> F::Output {
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    static VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| unimplemented!("clone"),
        |_| unimplemented!("wake"),
        |_| unimplemented!("wake_by_ref"),
        |_| {
            DROPS.fetch_add(1, Ordering::SeqCst);
        },
    );
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut context = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(val) = future.as_mut().poll(&mut context) {
            break val;
        }
    }
}

fn main() {
    let mut future = Box::pin(foo());
    block_on(future.as_mut());
    // Polling the completed future again panics with
    // "`async fn` resumed after completion"; the panic is the
    // expected outcome, caught here so the run can report it.
    let second = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        block_on(future.as_mut());
    }));
    println!(
        "second poll panicked={} waker drops={}",
        second.is_err(),
        DROPS.load(Ordering::SeqCst)
    );
}
"#;

#[test]
fn run_mode_closure_vtable_waker_resume_after_completion() {
    let dir = temp_dir("closure-vtable-resume-completion");
    let src = fixture(
        &dir,
        "closure_vtable_resume_completion.rs",
        CLOSURE_VTABLE_RESUME_AFTER_COMPLETION_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "second poll panicked=true waker drops=2\n"
    );
}

const CLOSURE_VTABLE_RESUME_AFTER_PANIC_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

static DROPS: AtomicUsize = AtomicUsize::new(0);

async fn foo() {
    panic!();
}

fn block_on<F: Future>(mut future: F) -> F::Output {
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    static VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| unimplemented!("clone"),
        |_| unimplemented!("wake"),
        |_| unimplemented!("wake_by_ref"),
        |_| {
            DROPS.fetch_add(1, Ordering::SeqCst);
        },
    );
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut context = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(val) = future.as_mut().poll(&mut context) {
            break val;
        }
    }
}

fn main() {
    let mut future = Box::pin(foo());
    let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        block_on(future.as_mut());
    }));
    // Polling the panicked future again panics with
    // "`async fn` resumed after panicking".
    let second = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        block_on(future.as_mut());
    }));
    println!(
        "first panicked={} second panicked={} waker drops={}",
        first.is_err(),
        second.is_err(),
        DROPS.load(Ordering::SeqCst)
    );
}
"#;

#[test]
fn run_mode_closure_vtable_waker_resume_after_panic() {
    let dir = temp_dir("closure-vtable-resume-panic");
    let src = fixture(
        &dir,
        "closure_vtable_resume_panic.rs",
        CLOSURE_VTABLE_RESUME_AFTER_PANIC_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "first panicked=true second panicked=true waker drops=2\n"
    );
}

const CLOSURE_VTABLE_CLONE_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{RawWaker, RawWakerVTable, Waker};

static CLONES: AtomicUsize = AtomicUsize::new(0);
static DROPS: AtomicUsize = AtomicUsize::new(0);

static VTABLE: RawWakerVTable = RawWakerVTable::new(
    |data| {
        CLONES.fetch_add(1, Ordering::SeqCst);
        RawWaker::new(data, &VTABLE)
    },
    |_| unimplemented!("wake"),
    |_| unimplemented!("wake_by_ref"),
    |_| {
        DROPS.fetch_add(1, Ordering::SeqCst);
    },
);

fn main() {
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let cloned = waker.clone();
    drop(cloned);
    drop(waker);
    println!(
        "clones={} drops={}",
        CLONES.load(Ordering::SeqCst),
        DROPS.load(Ordering::SeqCst)
    );
}
"#;

#[test]
fn run_mode_closure_vtable_waker_clone_and_drop() {
    let dir = temp_dir("closure-vtable-clone");
    let src = fixture(&dir, "closure_vtable_clone.rs", CLOSURE_VTABLE_CLONE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "clones=1 drops=2\n");
}
