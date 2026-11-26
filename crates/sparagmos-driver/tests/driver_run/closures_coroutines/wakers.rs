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
