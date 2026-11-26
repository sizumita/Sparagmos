use super::{Command, fixture, run_driver, run_driver_extra, run_ok, temp_dir};


const PANIC_SRC: &str = r#"
fn boom(x: i32) -> i32 { x + x }
fn main() { let _ = boom(2_000_000_000); }
"#;

#[test]
fn run_mode_guest_panic_exits_101() {
    let dir = temp_dir("panic");
    let src = fixture(&dir, "panic.rs", PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "guest panic must exit 101; stderr={stderr}"
    );
    assert!(
        stderr.contains("panicked"),
        "panic message goes to stderr: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

const VEC_RESIZE_PANIC_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};
static CLONES: AtomicUsize = AtomicUsize::new(0);
#[derive(Debug)]
struct Bomb(i32);
impl Clone for Bomb {
    fn clone(&self) -> Self {
        let n = CLONES.fetch_add(1, Ordering::SeqCst);
        if n >= 1 {
            panic!("clone boom");
        }
        Bomb(self.0)
    }
}
impl Drop for Bomb {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let mut v = vec![Bomb(1)];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        v.resize(5, Bomb(9));
    }));
    println!("caught={} len={}", r.is_err(), v.len());
    println!(
        "vals: {}",
        v.iter().map(|b| b.0.to_string()).collect::<Vec<_>>().join(",")
    );
}
"#;

#[test]
fn run_mode_vec_resize_clone_panic_sets_partial_len() {
    // When a fill clone panics mid-growth, the `SetLenOnDrop`
    // guard still runs during the unwind and publishes the
    // partial length: the elements written so far stay in the
    // vector (and drop at scope end), the unwritten slots do
    // not appear.
    let dir = temp_dir("vecresizepanic");
    let src = fixture(&dir, "vecresizepanic.rs", VEC_RESIZE_PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a caught clone panic must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop 9\ncaught=true len=2\nvals: 1,9\ndrop 1\ndrop 9\n",
        "the guard publishes the partial length on unwind"
    );
}

const REFCELL_PANIC_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let c = RefCell::new(1);
    let _a = c.borrow();
    let _b = c.borrow_mut();
}
"#;

#[test]
fn run_mode_refcell_borrow_mut_conflict_panics() {
    let dir = temp_dir("refcell-panic");
    let src = fixture(&dir, "refcell_panic.rs", REFCELL_PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "borrow_mut while borrowed is a guest panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("RefCell already borrowed"),
        "the panic carries Rust 1.100.0's exact message: {stderr}"
    );
}

const REFCELL_MUT_PANIC_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let c = RefCell::new(1);
    let _m = c.borrow_mut();
    let _a = c.borrow();
}
"#;

#[test]
fn run_mode_refcell_borrow_while_mutable_panics() {
    let dir = temp_dir("refcell-mut-panic");
    let src = fixture(&dir, "refcell_mut_panic.rs", REFCELL_MUT_PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "borrow while mutably borrowed is a guest panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("RefCell already mutably borrowed"),
        "the panic carries Rust 1.100.0's exact message: {stderr}"
    );
}

const RANGE_PANIC_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3, 4];
    let _s = &a[2..9];
    println!("unreachable");
}
"#;

const BOXED_SLICE_FROM_SLICE_REF_CUSTOM_CLONE_PANIC_SRC: &str = r#"
struct P(i32);
impl Clone for P {
    fn clone(&self) -> Self {
        panic!("custom clone ran");
    }
}

fn main() {
    let src = [P(1)];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _b: Box<[P]> = Box::from(&src[..]);
        println!("no panic");
    }));
    println!("caught: {}", r.is_err());
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_custom_clone_panic_unwinds() {
    let dir = temp_dir("boxed-from-slice-ref-custom-clone-panic");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_custom_clone_panic.rs",
        BOXED_SLICE_FROM_SLICE_REF_CUSTOM_CLONE_PANIC_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a panicking custom Clone must unwind out of the conversion; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "caught: true\n");
}

const BOXED_SLICE_FROM_SLICE_REF_CLONE_PANIC_WAKER_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{RawWaker, RawWakerVTable, Waker};

static DROPS: AtomicUsize = AtomicUsize::new(0);

struct Counted;
impl Drop for Counted {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe fn clone_ok(data: *const ()) -> RawWaker {
    let arc = Arc::from_raw(data as *const Counted);
    let cloned = arc.clone();
    let raw = Arc::into_raw(cloned) as *const ();
    std::mem::forget(arc);
    RawWaker::new(raw, &VT_OK)
}
unsafe fn clone_panic(_data: *const ()) -> RawWaker {
    panic!("clone boom")
}
unsafe fn wake_fn(data: *const ()) {
    let arc = Arc::from_raw(data as *const Counted);
    drop(arc);
}
unsafe fn wake_by_ref_fn(data: *const ()) {
    let arc = Arc::from_raw(data as *const Counted);
    std::mem::forget(arc);
}
unsafe fn drop_fn(data: *const ()) {
    let arc = Arc::from_raw(data as *const Counted);
    drop(arc);
}

static VT_OK: RawWakerVTable = RawWakerVTable::new(clone_ok, wake_fn, wake_by_ref_fn, drop_fn);
static VT_PANIC: RawWakerVTable =
    RawWakerVTable::new(clone_panic, wake_fn, wake_by_ref_fn, drop_fn);

fn make(probe: &Arc<Counted>, vt: &'static RawWakerVTable) -> Waker {
    let raw = Arc::into_raw(probe.clone()) as *const ();
    unsafe { Waker::from_raw(RawWaker::new(raw, vt)) }
}

fn main() {
    let probe = Arc::new(Counted);
    let src = [make(&probe, &VT_OK), make(&probe, &VT_PANIC)];
    println!("base count: {}", Arc::strong_count(&probe));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _b: Box<[Waker]> = Box::from(&src[..]);
        println!("no panic");
    }));
    println!("caught: {}", r.is_err());
    println!("after count: {}", Arc::strong_count(&probe));
    drop(src);
    println!("after drop-src count: {}", Arc::strong_count(&probe));
    println!("pointee drops: {}", DROPS.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_clone_panic_drops_prefix() {
    let dir = temp_dir("boxed-from-slice-ref-clone-panic-waker");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_clone_panic_waker.rs",
        BOXED_SLICE_FROM_SLICE_REF_CLONE_PANIC_WAKER_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an unwinding element clone must drop the cloned prefix; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "base count: 3\ncaught: true\nafter count: 3\nafter drop-src count: 1\npointee drops: 0\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_CLONE_PANIC_AGGREGATE_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{RawWaker, RawWakerVTable, Waker};

static DROPS: AtomicUsize = AtomicUsize::new(0);

struct Counted;
impl Drop for Counted {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe fn clone_ok(data: *const ()) -> RawWaker {
    let arc = Arc::from_raw(data as *const Counted);
    let cloned = arc.clone();
    let raw = Arc::into_raw(cloned) as *const ();
    std::mem::forget(arc);
    RawWaker::new(raw, &VT_OK)
}
unsafe fn clone_panic(_data: *const ()) -> RawWaker {
    panic!("clone boom")
}
unsafe fn wake_fn(data: *const ()) {
    let arc = Arc::from_raw(data as *const Counted);
    drop(arc);
}
unsafe fn wake_by_ref_fn(data: *const ()) {
    let arc = Arc::from_raw(data as *const Counted);
    std::mem::forget(arc);
}
unsafe fn drop_fn(data: *const ()) {
    let arc = Arc::from_raw(data as *const Counted);
    drop(arc);
}

static VT_OK: RawWakerVTable = RawWakerVTable::new(clone_ok, wake_fn, wake_by_ref_fn, drop_fn);
static VT_PANIC: RawWakerVTable =
    RawWakerVTable::new(clone_panic, wake_fn, wake_by_ref_fn, drop_fn);

fn make(probe: &Arc<Counted>, vt: &'static RawWakerVTable) -> Waker {
    let raw = Arc::into_raw(probe.clone()) as *const ();
    unsafe { Waker::from_raw(RawWaker::new(raw, vt)) }
}

fn main() {
    // One tuple element whose SECOND field's clone panics:
    // the first field's successful clone must still be
    // dropped by the unwinding construction.
    let probe = Arc::new(Counted);
    let src = [(make(&probe, &VT_OK), make(&probe, &VT_PANIC))];
    println!("base count: {}", Arc::strong_count(&probe));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _b: Box<[(Waker, Waker)]> = Box::from(&src[..]);
        println!("no panic");
    }));
    println!("caught: {}", r.is_err());
    println!("after count: {}", Arc::strong_count(&probe));
    drop(src);
    println!("after drop-src count: {}", Arc::strong_count(&probe));
    println!("pointee drops: {}", DROPS.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_clone_panic_drops_aggregate_prefix() {
    let dir = temp_dir("boxed-from-slice-ref-clone-panic-aggregate");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_clone_panic_aggregate.rs",
        BOXED_SLICE_FROM_SLICE_REF_CLONE_PANIC_AGGREGATE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an unwinding field clone must drop the aggregate's cloned fields; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "base count: 3\ncaught: true\nafter count: 3\nafter drop-src count: 1\npointee drops: 0\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_CLONE_PANIC_COUNTED_SRC: &str = r#"
use std::cell::Cell;
use std::rc::Rc;

#[derive(Debug)]
struct S2 {
    rc: Rc<Cell<i32>>,
    boom: bool,
}

impl Clone for S2 {
    fn clone(&self) -> Self {
        if self.boom {
            panic!("s2 boom");
        }
        S2 {
            rc: self.rc.clone(),
            boom: self.boom,
        }
    }
}

fn main() {
    let rc = Rc::new(Cell::new(0));
    let src = [
        S2 { rc: rc.clone(), boom: false },
        S2 { rc: rc.clone(), boom: true },
    ];
    println!("base strong: {}", Rc::strong_count(&rc));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _b: Box<[S2]> = Box::from(&src[..]);
        println!("no panic");
    }));
    println!("caught: {}", r.is_err());
    println!("after strong: {}", Rc::strong_count(&rc));
    drop(src);
    println!("final strong: {}", Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_clone_panic_drops_counted_prefix() {
    let dir = temp_dir("boxed-from-slice-ref-clone-panic-counted");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_clone_panic_counted.rs",
        BOXED_SLICE_FROM_SLICE_REF_CLONE_PANIC_COUNTED_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an unwinding custom clone must drop the counted prefix; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "base strong: 3\ncaught: true\nafter strong: 3\nfinal strong: 1\n"
    );
}

const VEC_CLONE_PANIC_SRC: &str = r#"
use std::cell::Cell;
use std::rc::Rc;

struct Counted { clones: Rc<Cell<i32>> }
impl Clone for Counted {
    fn clone(&self) -> Self { self.clones.set(self.clones.get() + 1); Counted { clones: self.clones.clone() } }
}
impl Drop for Counted {
    fn drop(&mut self) { println!("drop counted"); }
}
struct Boom;
impl Clone for Boom { fn clone(&self) -> Self { panic!("boom clone"); } }

fn main() {
    let counter = Rc::new(Cell::new(0));
    let v = vec![
        (Counted { clones: counter.clone() }, Boom),
        (Counted { clones: counter.clone() }, Boom),
    ];
    println!("before strong: {}", Rc::strong_count(&counter));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _w = v.clone();
        println!("no panic");
    }));
    println!("caught: {}", r.is_err());
    println!("clones: {}", counter.get());
    println!("after strong: {}", Rc::strong_count(&counter));
    println!("source len: {}", v.len());
    drop(v);
    println!("final strong: {}", Rc::strong_count(&counter));
}
"#;

#[test]
fn run_mode_vec_clone_panic_drops_prefix_and_keeps_source() {
    let dir = temp_dir("vec-clone-panic");
    let src = fixture(&dir, "vec_clone_panic.rs", VEC_CLONE_PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a panicking element Clone must unwind out of Vec::clone, dropping the cloned prefix; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "before strong: 3\ndrop counted\ncaught: true\nclones: 1\nafter strong: 3\nsource len: 2\ndrop counted\ndrop counted\nfinal strong: 1\n"
    );
}

const CLOSURE_CLONE_PANIC_SRC: &str = r#"use std::cell::Cell;
use std::rc::Rc;

struct Boom(i32);

impl Clone for Boom {
    fn clone(&self) -> Self {
        panic!("boom clone");
    }
}

fn main() {
    // A direct .clone() of a closure whose second capture's
    // Clone panics: the capture cloned before it drops during
    // unwinding, restoring the strong count.
    let rc = Rc::new(Cell::new(7));
    let rc_in = rc.clone();
    let boom = Boom(3);
    let f = move || {
        let _ = rc_in.get();
        let _ = &boom;
    };
    println!("direct before: {}", Rc::strong_count(&rc));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = f.clone();
        println!("direct no panic");
    }));
    println!("direct caught: {}", r.is_err());
    println!("direct after: {}", Rc::strong_count(&rc));
    drop(f);
    println!("direct final: {}", Rc::strong_count(&rc));

    // The boxed-slice conversion of the same closure shape
    // drops its cloned prefix the same way.
    let rc2 = Rc::new(Cell::new(7));
    let rc2_in = rc2.clone();
    let boom2 = Boom(4);
    let f2 = move || {
        let _ = rc2_in.get();
        let _ = &boom2;
    };
    let src = [f2];
    println!("boxed before: {}", Rc::strong_count(&rc2));
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _b: Box<[_]> = Box::from(&src[..]);
        println!("boxed no panic");
    }));
    println!("boxed caught: {}", r2.is_err());
    println!("boxed after: {}", Rc::strong_count(&rc2));
}
"#;

#[test]
fn run_mode_closure_clone_panic_drops_cloned_captures() {
    let dir = temp_dir("closure-clone-panic");
    let src = fixture(
        &dir,
        "closure_clone_panic.rs",
        CLOSURE_CLONE_PANIC_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a panicking capture clone must drop the captures cloned before it; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "direct before: 2\ndirect caught: true\ndirect after: 2\ndirect final: 1\nboxed before: 2\nboxed caught: true\nboxed after: 2\n"
    );
}

const CLOSURE_CLONE_ZST_CAPTURE_PANIC_SRC: &str = r#"use std::cell::Cell;
use std::rc::Rc;

struct Boom;

impl Clone for Boom {
    fn clone(&self) -> Self {
        panic!("boom clone");
    }
}

fn main() {
    // A direct .clone() of a closure whose second capture is a
    // zero-sized unit struct whose Clone panics. The compiler
    // emits no debug-info entry for a zero-sized capture, so the
    // recorded environment shape used to drop the field and the
    // shape with it, and the clone fell back to a shallow copy:
    // no panic, and the copy aliased the Rc capture (Issue
    // #499). The capture cloned before the panicking one drops
    // during unwinding, restoring the strong count.
    let rc = Rc::new(Cell::new(7));
    let rc_in = rc.clone();
    let boom = Boom;
    let f = move || {
        let _ = rc_in.get();
        let _ = &boom;
    };
    println!("direct before: {}", Rc::strong_count(&rc));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = f.clone();
        println!("direct no panic");
    }));
    println!("direct caught: {}", r.is_err());
    println!("direct after: {}", Rc::strong_count(&rc));
    drop(f);
    println!("direct final: {}", Rc::strong_count(&rc));

    // The boxed-slice conversion of the same closure shape
    // dispatches the panicking capture clone the same way.
    let rc2 = Rc::new(Cell::new(7));
    let rc2_in = rc2.clone();
    let boom2 = Boom;
    let f2 = move || {
        let _ = rc2_in.get();
        let _ = &boom2;
    };
    let src = [f2];
    println!("boxed before: {}", Rc::strong_count(&rc2));
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _b: Box<[_]> = Box::from(&src[..]);
        println!("boxed no panic");
    }));
    println!("boxed caught: {}", r2.is_err());
    println!("boxed after: {}", Rc::strong_count(&rc2));
}
"#;

#[test]
fn run_mode_closure_clone_zst_capture_panics() {
    let dir = temp_dir("closure-clone-zst-capture-panic");
    let src = fixture(
        &dir,
        "closure_clone_zst_capture_panic.rs",
        CLOSURE_CLONE_ZST_CAPTURE_PANIC_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure capturing a zero-sized type must clone capture-by-capture, not shallow-copy; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "direct before: 2\ndirect caught: true\ndirect after: 2\ndirect final: 1\nboxed before: 2\nboxed caught: true\nboxed after: 2\n"
    );
}

const BOXED_SLICE_ELIDED_ELEM_CLONE_PANIC_SRC: &str = r#"struct D;

impl Clone for D {
    fn clone(&self) -> Self {
        println!("clone D");
        D
    }
}

impl Drop for D {
    fn drop(&mut self) {
        println!("drop D");
    }
}

struct Boom;

impl Clone for Boom {
    fn clone(&self) -> Self {
        panic!("boom clone");
    }
}

fn main() {
    // Array elements that are zero-sized tuples: the tuple's
    // drop plan reaches the clone through the array's elided
    // element plan, and the tuple's field glue must survive
    // the extraction — the field cloned before the panicking
    // one drops during unwinding.
    let src: [[(D, Boom); 1]; 1] = [[(D, Boom)]];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _b: Box<[[(D, Boom); 1]]> = Box::from(&src[..]);
        println!("array no panic");
    }));
    println!("array caught: {}", r.is_err());

    // Nested arrays: the inner array's own plan is the elided
    // element of the outer array, so the element extraction
    // must pass through the wrapper as well.
    let src2: [[[(D, Boom); 1]; 1]; 1] = [[[(D, Boom)]]];
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _b: Box<[[[(D, Boom); 1]; 1]]> = Box::from(&src2[..]);
        println!("nested no panic");
    }));
    println!("nested caught: {}", r2.is_err());
}
"#;

#[test]
fn run_mode_boxed_slice_elided_elem_clone_panic_drops_cloned_fields() {
    let dir = temp_dir("boxed-slice-elided-elem-clone-panic");
    let src = fixture(
        &dir,
        "boxed_slice_elided_elem_clone_panic.rs",
        BOXED_SLICE_ELIDED_ELEM_CLONE_PANIC_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a panicking clone behind an elided element plan must still drop the cloned prefix; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "clone D\ndrop D\narray caught: true\nclone D\ndrop D\nnested caught: true\ndrop D\ndrop D\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_CLONE_PANIC_AGGREGATE_DROP_ORDER_SRC: &str = r#"
struct G(i32);
impl Clone for G {
    fn clone(&self) -> Self { G(self.0) }
}
impl Drop for G {
    fn drop(&mut self) {
        println!("drop G{}", self.0);
    }
}

struct P;
impl Clone for P {
    fn clone(&self) -> Self { panic!("clone P panics") }
}

struct T {
    a: G,
    b: G,
    c: P,
}
impl Clone for T {
    fn clone(&self) -> Self {
        T { a: self.a.clone(), b: self.b.clone(), c: self.c.clone() }
    }
}

fn main() {
    // A tuple element whose third field's clone panics: the
    // two cloned fields are construction temporaries and drop
    // in reverse initialization order; the source tuple's own
    // fields later drop in declaration order.
    let src = [(G(0), G(1), P)];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _boxed: Box<[(G, G, P)]> = Box::from(&src[..]);
    }));
    println!("caught={}", r.is_err());
    // The same for a struct element's fields.
    let src2 = [T { a: G(0), b: G(1), c: P }];
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _boxed2: Box<[T]> = Box::from(&src2[..]);
    }));
    println!("caught2={}", r2.is_err());
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_clone_panic_drops_aggregate_fields_in_reverse() {
    let dir = temp_dir("boxed-from-slice-ref-clone-panic-aggregate-drop-order");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_clone_panic_aggregate_drop_order.rs",
        BOXED_SLICE_FROM_SLICE_REF_CLONE_PANIC_AGGREGATE_DROP_ORDER_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an unwinding aggregate field clone must drop the cloned fields in reverse order; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop G1\ndrop G0\ncaught=true\ndrop G1\ndrop G0\ncaught2=true\ndrop G0\ndrop G1\ndrop G0\ndrop G1\n"
    );
}

const BOXED_SLICE_FROM_SLICE_REF_CLONE_PANIC_ARRAY_DROP_ORDER_SRC: &str = r#"
struct D(i32);
impl Clone for D {
    fn clone(&self) -> Self {
        if self.0 == 3 {
            panic!("clone D3 panics");
        }
        D(self.0 + 100)
    }
}
impl Drop for D {
    fn drop(&mut self) {
        println!("drop D{}", self.0);
    }
}

fn main() {
    // An array element whose third item's clone panics: the
    // array clone's initialization guard drops the initialized
    // prefix in element order.
    {
        let src = [[D(1), D(2), D(3)]];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _boxed: Box<[[D; 3]]> = Box::from(&src[..]);
        }));
        println!("array caught={}", r.is_err());
    }
    // A tuple element whose third field's clone panics: the
    // cloned fields are construction temporaries and drop in
    // reverse initialization order.
    {
        let src = [(D(1), D(2), D(3))];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _boxed: Box<[(D, D, D)]> = Box::from(&src[..]);
        }));
        println!("tuple caught={}", r.is_err());
    }
    // Slice elements directly: the initialized prefix drops
    // in element order.
    {
        let src = [D(1), D(2), D(3)];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _boxed: Box<[D]> = Box::from(&src[..]);
        }));
        println!("slice caught={}", r.is_err());
    }
    // A Vec element whose third item's clone panics: element
    // order as well.
    {
        let src = [vec![D(1), D(2), D(3)]];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _boxed: Box<[Vec<D>]> = Box::from(&src[..]);
        }));
        println!("vec caught={}", r.is_err());
    }
}
"#;

#[test]
fn run_mode_boxed_slice_from_slice_ref_clone_panic_drops_array_prefix_in_element_order() {
    let dir = temp_dir("boxed-from-slice-ref-clone-panic-array-drop-order");
    let src = fixture(
        &dir,
        "boxed_from_slice_ref_clone_panic_array_drop_order.rs",
        BOXED_SLICE_FROM_SLICE_REF_CLONE_PANIC_ARRAY_DROP_ORDER_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an unwinding array element clone must drop the cloned prefix in element order while tuple fields drop in reverse; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop D101\ndrop D102\narray caught=true\ndrop D1\ndrop D2\ndrop D3\ndrop D102\ndrop D101\ntuple caught=true\ndrop D1\ndrop D2\ndrop D3\ndrop D101\ndrop D102\nslice caught=true\ndrop D1\ndrop D2\ndrop D3\ndrop D101\ndrop D102\nvec caught=true\ndrop D1\ndrop D2\ndrop D3\n"
    );
}

#[test]
fn run_mode_range_index_out_of_bounds_panics() {
    let dir = temp_dir("range-panic");
    let src = fixture(&dir, "range_panic.rs", RANGE_PANIC_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(101),
        "an out-of-bounds slice range is a guest panic (101)"
    );
}

const REFCELL_NESTED_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let cell = RefCell::new(1i32);
    *cell.borrow_mut() = *cell.borrow() + 1;
    println!("{}", cell.borrow());
}
"#;

#[test]
fn run_mode_refcell_nested_borrow_panics_like_native() {
    // `*cell.borrow_mut() = *cell.borrow() + 1` panics under real
    // Rust too (the shared borrow is taken while the mutable one
    // is still live): the guest panic is the correct outcome.
    let dir = temp_dir("refcell-nested");
    let src = fixture(&dir, "refcell_nested.rs", REFCELL_NESTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "nested borrow must panic like native; stderr={stderr}"
    );
    assert!(
        stderr.contains("RefCell already borrowed"),
        "the panic must be the borrow conflict; stderr={stderr}"
    );
}
// ----- catch_unwind (Lane DH) ------------------------------------------------

const CATCH_OK_SRC: &str = r#"
fn main() {
    let r = std::panic::catch_unwind(|| 40 + 2);
    match r {
        Ok(v) => println!("ok {v}"),
        Err(_) => println!("err"),
    }
}
"#;

const CATCH_LITERAL_SRC: &str = r#"
fn main() {
    let r = std::panic::catch_unwind(|| -> i32 { panic!("boom") });
    match r {
        Ok(_) => println!("ok"),
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&str>() {
                println!("payload str: {s}");
            } else {
                println!("payload not str");
            }
            println!("is String: {}", e.is::<String>());
        }
    }
}
"#;

const CATCH_FORMATTED_SRC: &str = r#"
fn main() {
    let mut x = 0usize;
    for i in 0..7 { x += i; }
    let r = std::panic::catch_unwind(|| -> i32 { panic!("fmt {}", x) });
    match r {
        Ok(_) => println!("ok"),
        Err(e) => {
            println!("is str: {}", e.is::<&str>());
            if let Some(s) = e.downcast_ref::<String>() {
                println!("payload String: {s}");
            } else {
                println!("payload not String");
            }
        }
    }
}
"#;

const CATCH_PANIC_ANY_SRC: &str = r#"
fn main() {
    let r = std::panic::catch_unwind(|| -> i32 { std::panic::panic_any(42i32) });
    match r {
        Ok(_) => println!("ok"),
        Err(e) => {
            if let Some(v) = e.downcast_ref::<i32>() {
                println!("payload i32: {v}");
            } else {
                println!("payload not i32");
            }
            match e.downcast::<i32>() {
                Ok(v) => println!("downcast value: {}", *v),
                Err(_) => println!("downcast failed"),
            }
        }
    }
}
"#;

const CATCH_RESUME_SRC: &str = r#"
fn main() {
    let r = std::panic::catch_unwind(|| -> i32 { panic!("again") });
    println!("caught: {}", r.is_err());
    if let Err(e) = r {
        std::panic::resume_unwind(e);
    }
    println!("unreachable");
}
"#;

const CATCH_ASSERT_UNWIND_SAFE_SRC: &str = r#"
use std::panic::{catch_unwind, AssertUnwindSafe};
fn main() {
    let mut state = 10;
    let r = catch_unwind(AssertUnwindSafe(|| {
        state = 99;
        panic!("inner");
    }));
    println!("caught={} state={}", r.is_err(), state);
}
"#;

const CATCH_NESTED_SRC: &str = r#"
use std::panic::catch_unwind;
fn main() {
    let r = catch_unwind(|| {
        let inner = catch_unwind(|| -> i32 { panic!("deep") });
        match inner { Ok(_) => 1, Err(_) => 2 }
    });
    match r {
        Ok(v) => println!("outer ok {v}"),
        Err(_) => println!("outer err"),
    }
    let r2 = catch_unwind(|| {
        let inner = catch_unwind(|| -> i32 { panic!("deep2") });
        if let Err(e) = inner { std::panic::resume_unwind(e); }
        0
    });
    match r2 {
        Ok(_) => println!("r2 ok"),
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&str>() {
                println!("r2 payload: {s}");
            } else {
                println!("r2 payload other");
            }
        }
    }
}
"#;

const CATCH_OVERFLOW_SRC: &str = r#"
fn main() {
    let mut x = 0i32;
    for i in 0..7 { x += i; }
    let r = std::panic::catch_unwind(|| { let a = i32::MAX; a + x });
    match r {
        Ok(v) => println!("ok {v}"),
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&str>() {
                println!("overflow payload: {s}");
            } else {
                println!("overflow payload other");
            }
        }
    }
}
"#;

const CATCH_STATE_SRC: &str = r#"
use std::panic::catch_unwind;
use std::sync::atomic::{AtomicI32, Ordering};
static MARK: AtomicI32 = AtomicI32::new(0);
fn main() {
    MARK.store(5, Ordering::SeqCst);
    let _ = catch_unwind(|| {
        MARK.store(7, Ordering::SeqCst);
        panic!("x");
    });
    println!("mark={}", MARK.load(Ordering::SeqCst));
    println!("after");
}
"#;

#[test]
fn run_mode_catch_unwind_ok_path() {
    run_ok("catch-ok", CATCH_OK_SRC, "ok 42\n");
}

#[test]
fn run_mode_catch_unwind_literal_payload_is_str() {
    run_ok(
        "catch-literal",
        CATCH_LITERAL_SRC,
        "payload str: boom\nis String: false\n",
    );
}

#[test]
fn run_mode_catch_unwind_formatted_payload_is_string() {
    run_ok(
        "catch-formatted",
        CATCH_FORMATTED_SRC,
        "is str: false\npayload String: fmt 21\n",
    );
}

#[test]
fn run_mode_catch_unwind_panic_any_payload() {
    run_ok(
        "catch-panic-any",
        CATCH_PANIC_ANY_SRC,
        "payload i32: 42\ndowncast value: 42\n",
    );
}

#[test]
fn run_mode_catch_unwind_assert_unwind_safe() {
    run_ok(
        "catch-aus",
        CATCH_ASSERT_UNWIND_SAFE_SRC,
        "caught=true state=99\n",
    );
}

#[test]
fn run_mode_catch_unwind_nested_and_resume_through() {
    run_ok(
        "catch-nested",
        CATCH_NESTED_SRC,
        "outer ok 2\nr2 payload: deep2\n",
    );
}

#[test]
fn run_mode_catch_unwind_overflow_payload_message() {
    run_ok(
        "catch-overflow",
        CATCH_OVERFLOW_SRC,
        "overflow payload: attempt to add with overflow\n",
    );
}

#[test]
fn run_mode_catch_unwind_state_survives_the_catch() {
    run_ok("catch-state", CATCH_STATE_SRC, "mark=7\nafter\n");
}

// ----- unwind drops (Lane EW) ------------------------------------------------
// A panic unwinding through a frame runs the cleanup blocks MIR
// built for it: the destructors of the locals live at the panic
// point, in reverse declaration order per frame, innermost frame
// first. A destructor that itself panics while unwinding aborts
// the run (a native double panic); a destructor that panics on the
// normal path is an ordinary, catchable panic.

const UNWIND_DROP_SINGLE_SRC: &str = r#"
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn boom() { let _m = M(7); panic!("boom"); }
fn main() {
    let r = std::panic::catch_unwind(boom);
    println!("caught {}", r.is_err());
}
"#;

const UNWIND_DROP_NESTED_SRC: &str = r#"
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn inner() { let _a = M(1); let _b = M(2); panic!("deep"); }
fn middle() { let _c = M(3); inner(); let _d = M(4); }
fn main() {
    let r = std::panic::catch_unwind(middle);
    println!("caught {}", r.is_err());
    let _e = M(5);
    println!("end");
}
"#;

const UNWIND_DROP_PAYLOAD_SRC: &str = r#"
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let r = std::panic::catch_unwind(|| -> i32 {
        let _m = M(3);
        panic!("boom")
    });
    match r {
        Ok(_) => println!("ok"),
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&str>() {
                println!("payload str: {s}");
            } else {
                println!("payload not str");
            }
        }
    }
}
"#;

const UNWIND_DROP_UNCAUGHT_SRC: &str = r#"
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn boom() { let _m = M(7); panic!("boom"); }
fn main() { let _x = M(9); boom(); }
"#;

const UNWIND_DROP_PANICS_SRC: &str = r#"
struct Bad;
impl Drop for Bad { fn drop(&mut self) { panic!("dtor panic"); } }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn boom() { let _b = Bad; let _m = M(1); panic!("orig"); }
fn main() {
    let r = std::panic::catch_unwind(boom);
    println!("caught {}", r.is_err());
}
"#;

const DROP_PANIC_NORMAL_SRC: &str = r#"
struct Bad;
impl Drop for Bad { fn drop(&mut self) { println!("bad dtor runs"); panic!("dtor panic"); } }
fn main() {
    let r = std::panic::catch_unwind(|| { let _b = Bad; });
    println!("caught {}", r.is_err());
}
"#;

#[test]
fn run_mode_unwind_drops_single_frame() {
    run_ok(
        "unwind-drop-single",
        UNWIND_DROP_SINGLE_SRC,
        "dtor 7\ncaught true\n",
    );
}

#[test]
fn run_mode_unwind_drops_nested_frames_in_order() {
    // Reverse declaration order within a frame (2 before 1), the
    // inner frame before its caller (3), a local declared after
    // the panicking call is never built (no dtor 4), and the
    // normal scope-end drop after the catch is unaffected (5,
    // after "end").
    run_ok(
        "unwind-drop-nested",
        UNWIND_DROP_NESTED_SRC,
        "dtor 2\ndtor 1\ndtor 3\ncaught true\nend\ndtor 5\n",
    );
}

#[test]
fn run_mode_unwind_drops_and_payload_downcast() {
    // The destructor runs during the unwind, before the catch
    // handler observes the payload; the payload itself (and its
    // downcast) is exactly as without drops.
    run_ok(
        "unwind-drop-payload",
        UNWIND_DROP_PAYLOAD_SRC,
        "dtor 3\npayload str: boom\n",
    );
}

#[test]
fn run_mode_uncaught_panic_runs_drops() {
    let dir = temp_dir("unwind-drop-uncaught");
    let src = fixture(&dir, "case.rs", UNWIND_DROP_UNCAUGHT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "uncaught panic must exit 101; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dtor 7\ndtor 9\n",
        "drops run innermost frame first on an uncaught panic; stderr={stderr}"
    );
}

#[test]
fn run_mode_unwind_drop_panics_aborts_uncaught_by_catch_unwind() {
    // Unwinding drops M(1) first (reverse order), then Bad's
    // destructor panics while unwinding: a double panic. The run
    // aborts (134, abort reason on stderr) and the surrounding
    // catch_unwind does NOT observe it ("caught" never prints),
    // exactly as a native abort is not a catchable panic.
    let dir = temp_dir("unwind-drop-panics");
    let src = fixture(&dir, "case.rs", UNWIND_DROP_PANICS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "a panic in an unwind destructor must abort with 134; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dtor 1\n",
        "only the first unwind drop runs before the abort; stderr={stderr}"
    );
    assert!(
        stderr.contains("aborting"),
        "the abort must be named on stderr; stderr={stderr}"
    );
}

#[test]
fn run_mode_drop_panic_on_normal_path_is_caught() {
    run_ok(
        "drop-panic-normal",
        DROP_PANIC_NORMAL_SRC,
        "bad dtor runs\ncaught true\n",
    );
}

#[test]
fn run_mode_resume_unwind_repanics_out_of_the_catch() {
    let dir = temp_dir("catch-resume");
    let src = fixture(&dir, "resume.rs", CATCH_RESUME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "a resumed payload escapes as a guest panic (101); stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "caught: true\n",
        "output before the resume is kept; the line after it never runs"
    );
    assert!(
        stderr.contains("again"),
        "the resumed panic reports the payload message; stderr={stderr}"
    );
}

const SUBSLICE_OOB_SRC: &str = r#"
fn main() {
    let v = [1, 2, 3];
    let s: &[i32] = &v;
    let a = 0usize;
    let b = 9usize;
    let _bad = &s[a..b];
    println!("unreachable");
}
"#;

#[test]
fn run_mode_subslice_out_of_range_panics() {
    // A dynamic range past the end of the slice is a guest panic
    // (core's range-bounds check), exit 101, like native.
    let dir = temp_dir("subslice-oob");
    let src = fixture(&dir, "subslice_oob.rs", SUBSLICE_OOB_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "out-of-range subslice must panic; stderr={stderr}"
    );
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("unreachable"),
        "the program must not continue past the panic"
    );
}

const ASSERT_UNWIND_SAFE_DROP_SRC: &str = r#"
use std::panic::AssertUnwindSafe;
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    { let _w = AssertUnwindSafe(D(3)); }
    println!("after");
    let w = AssertUnwindSafe(D(4));
    std::mem::drop(w);
    println!("end");
}
"#;

#[test]
fn run_mode_assert_unwind_safe_drop_runs_held_dtor() {
    // The one-field `AssertUnwindSafe` wrapper's glue is the held
    // value's glue, at scope end and through `mem::drop` alike.
    let dir = temp_dir("assert-unwind-safe-drop");
    let src = fixture(&dir, "assert_unwind_safe_drop.rs", ASSERT_UNWIND_SAFE_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "AssertUnwindSafe drops must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dtor 3\nafter\ndtor 4\nend\n",
        "the held value's destructor runs at each wrapper drop"
    );
}

const CATCH_UNWIND_RESUME_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::Coroutine;
use std::pin::Pin;
use std::panic;
fn main() {
    let mut foo = #[coroutine] || { yield; };
    let res = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        Pin::new(&mut foo).resume(())
    }));
    println!("err={}", res.is_err());
}
"#;

#[test]
fn run_mode_catch_unwind_resume_coroutine() {
    // The panic machinery's path through a coroutine resume:
    // `AssertUnwindSafe` derefs to the closure, the resume result
    // (`CoroutineState`) drops out of the returned `Result`.
    let dir = temp_dir("catch-unwind-resume");
    let src = fixture(&dir, "catch_unwind_resume.rs", CATCH_UNWIND_RESUME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "catch_unwind around a resume must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "err=false\n",
        "the resume completed without a panic"
    );
}

const CALLER_LOCATION_INLINED_SRC: &str = r#"macro_rules! caller_location_from_macro {
    () => (core::panic::Location::caller());
}

fn main() {
    let loc = core::panic::Location::caller();
    assert_eq!(loc.file(), file!());
    assert_eq!(loc.line(), 6);
    assert_eq!(loc.column(), 15);

    let loc2 = caller_location_from_macro!();
    assert_eq!(loc2.file(), file!());
    assert_eq!(loc2.line(), 11);
    assert_eq!(loc2.column(), 16);
}
"#;

#[test]
fn run_mode_caller_location_after_inlining() {
    // At high MIR optimization levels `Location::caller` is
    // inlined into the calling body, leaving a bare
    // `caller_location` intrinsic call: the stamp must come
    // from that call site's own span (the intrinsic is not
    // itself `#[track_caller]`, so the usual stamping gate
    // skips it).
    let dir = temp_dir("caller-location-inlined");
    let src = fixture(&dir, "caller_location.rs", CALLER_LOCATION_INLINED_SRC);
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-Zmir-opt-level=4"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "");
}

const CALLER_LOCATION_TRACK_CALLER_OPT_SRC: &str = r#"#[inline(never)]
#[track_caller]
fn codegen_caller_loc() -> &'static core::panic::Location<'static> {
    core::panic::Location::caller()
}

macro_rules! caller_location_from_macro {
    () => (codegen_caller_loc());
}

fn main() {
    let loc = codegen_caller_loc();
    assert_eq!(loc.line(), 12);
    assert_eq!(loc.column(), 15);

    let loc2 = caller_location_from_macro!();
    assert_eq!(loc2.line(), 16);
    assert_eq!(loc2.column(), 16);
}
"#;

#[test]
fn run_mode_caller_location_track_caller_at_optimized_mir() {
    // At mir-opt level 4 `Location::caller` is inlined into
    // the `#[track_caller]` body (a bare intrinsic call), and
    // the caller's call carries its function operand as a
    // promoted constant whose span is erased: the stamp for
    // the track-caller call must still name the call site,
    // taken from the call terminator's own span.
    let dir = temp_dir("caller-location-track-caller-opt");
    let src = fixture(
        &dir,
        "caller_location_opt.rs",
        CALLER_LOCATION_TRACK_CALLER_OPT_SRC,
    );
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-Zmir-opt-level=4"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
}

const CALLER_LOCATION_SRC: &str = r#"
#[track_caller]
fn here() -> u32 {
    std::panic::Location::caller().line()
}
fn main() {
    println!("tracked {}", here());
    println!("direct {}", std::panic::Location::caller().line());
}
"#;

#[test]
fn run_mode_caller_location_reports_the_call_site() {
    let dir = temp_dir("caller-location");
    let src = fixture(&dir, "loc.rs", CALLER_LOCATION_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // `here()` is called on line 7 of the fixture; the direct
    // `Location::caller()` is evaluated on line 8.
    assert_eq!(stdout, "tracked 7\ndirect 8\n", "stdout={stdout}");
}

const CALLER_LOCATION_FNPTR_SRC: &str = r#"
fn ptr_call(f: fn() -> u32) -> u32 { f() }
#[track_caller]
fn tracked() -> u32 { std::panic::Location::caller().line() }
trait Tr { fn m() -> u32; }
impl Tr for () {
    #[track_caller]
    fn m() -> u32 { std::panic::Location::caller().line() }
}
fn main() {
    println!("fn {}", ptr_call(tracked));
    println!("impl {}", ptr_call(<() as Tr>::m));
}
"#;

#[test]
fn run_mode_caller_location_through_fn_pointer_is_the_definition() {
    // A `#[track_caller]` function entered through a function
    // pointer observes its own definition location (the call
    // shim's location), for a free function (line 4) and for a
    // trait method item (the impl method, line 8) alike.
    run_ok(
        "caller-location-fnptr",
        CALLER_LOCATION_FNPTR_SRC,
        "fn 4\nimpl 8\n",
    );
}

const CALLER_LOCATION_FNPTR_OPT_SRC: &str = r#"
fn ptr_call(f: fn()) {
    f();
}

fn pass_to_ptr_call<T>(f: fn(T), x: T) {
    f(x);
}

#[track_caller]
fn tracked() {
    let expected_line = line!() - 1;
    let location = std::panic::Location::caller();
    assert_eq!(location.line(), expected_line);
}

#[track_caller]
fn tracked_unit(_: ()) {
    let expected_line = line!() - 1;
    let location = std::panic::Location::caller();
    assert_eq!(location.line(), expected_line);
}

trait Trait {
    fn trait_tracked();
    fn trait_tracked_unit(_: ());
}

impl Trait for () {
    #[track_caller]
    fn trait_tracked() {
        let expected_line = line!() - 1;
        let location = std::panic::Location::caller();
        assert_eq!(location.line(), expected_line);
    }

    #[track_caller]
    fn trait_tracked_unit(_: ()) {
        let expected_line = line!() - 1;
        let location = std::panic::Location::caller();
        assert_eq!(location.line(), expected_line);
    }
}

trait TrackedTrait {
    #[track_caller]
    fn trait_tracked_default() {
        let expected_line = line!() - 1;
        let location = std::panic::Location::caller();
        assert_eq!(location.line(), expected_line);
    }

    #[track_caller]
    fn trait_tracked_unit_default(_: ()) {
        let expected_line = line!() - 1;
        let location = std::panic::Location::caller();
        assert_eq!(location.line(), expected_line);
    }
}

impl TrackedTrait for () {}

trait BlanketTracked {
    #[track_caller]
    fn tracked_blanket();
    #[track_caller]
    fn tracked_blanket_unit(_: ());
}

impl BlanketTracked for () {
    fn tracked_blanket() {
        let expected_line = line!() - 1;
        let location = std::panic::Location::caller();
        assert_eq!(location.line(), expected_line);
    }

    fn tracked_blanket_unit(_: ()) {
        let expected_line = line!() - 1;
        let location = std::panic::Location::caller();
        assert_eq!(location.line(), expected_line);
    }
}

fn main() {
    ptr_call(tracked);
    ptr_call(<() as Trait>::trait_tracked);
    ptr_call(<() as TrackedTrait>::trait_tracked_default);
    ptr_call(<() as BlanketTracked>::tracked_blanket);
    pass_to_ptr_call(tracked_unit, ());
    pass_to_ptr_call(<() as Trait>::trait_tracked_unit, ());
    pass_to_ptr_call(<() as TrackedTrait>::trait_tracked_unit_default, ());
    pass_to_ptr_call(<() as BlanketTracked>::tracked_blanket_unit, ());
}
"#;

#[test]
fn run_mode_caller_location_through_fn_pointer_at_optimized_mir() {
    // At mir-opt level 4 `Location::caller` is inlined into
    // each `#[track_caller]` body, so the body's first
    // terminator carries the inlined intrinsic's scope: the
    // definition-location stamp recorded for a body entered
    // through a function pointer must still name the body's
    // definition, not the inlined call's site inside the body.
    // Covers a free function, a trait method, a trait default
    // method and a blanket-tracked impl method, with and
    // without an argument (the rustc
    // `rfc-2091-track-caller/tracked-fn-ptr{,-with-arg}`
    // shapes).
    let dir = temp_dir("caller-location-fnptr-opt");
    let src = fixture(
        &dir,
        "caller_location_fnptr_opt.rs",
        CALLER_LOCATION_FNPTR_OPT_SRC,
    );
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-Zmir-opt-level=4"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
}

const CALLER_LOCATION_CALL_CHAIN_OPT_SRC: &str = r#"
use std::panic::Location;

struct Foo;

impl Foo {
    #[inline(always)]
    #[track_caller]
    fn check_loc(&self, line: u32, col: u32) -> &Self {
        let loc = Location::caller();
        assert_eq!(loc.line(), line);
        assert_eq!(loc.column(), col);
        self
    }
}

fn main() {
    let foo = Foo;
    foo.
        check_loc(line!(), 9).check_loc(line!(), 31)
        .check_loc(line!(), 10);
}
"#;

#[test]
fn run_mode_caller_location_method_call_chain_at_optimized_mir() {
    // At mir-opt level 4 the function operand of a method call
    // can arrive as a promoted constant whose span was erased,
    // leaving the terminator's own span, which starts at the
    // receiver (`foo` on the line above the first segment): the
    // `#[track_caller]` stamp of each call in the chain must
    // still name that segment's method name (the rustc
    // `rfc-2091-track-caller/call-chain` shape).
    let dir = temp_dir("caller-location-call-chain-opt");
    let src = fixture(
        &dir,
        "caller_location_call_chain_opt.rs",
        CALLER_LOCATION_CALL_CHAIN_OPT_SRC,
    );
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-Zmir-opt-level=4"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
}

const CALLER_LOCATION_DYN_SRC: &str = r#"
trait Tr {
    #[track_caller]
    fn def_tracked(&self) -> (u32, u32) {
        let l = std::panic::Location::caller();
        (l.line(), l.column())
    }
    fn impl_only(&self) -> (u32, u32);
}
impl Tr for () {
    #[track_caller]
    fn impl_only(&self) -> (u32, u32) {
        let l = std::panic::Location::caller();
        (l.line(), l.column())
    }
}
fn main() {
    let t: &dyn Tr = &();
    let (l1, c1) = t.def_tracked();
    println!("def {l1} {c1}");
    let (l2, c2) = t.impl_only();
    println!("impl {l2} {c2}");
}
"#;

#[test]
fn run_mode_caller_location_through_dyn_trait_object() {
    // A trait definition's `#[track_caller]` method called on
    // a trait object observes the call site (line 19, the
    // method name at column 22); an impl-only `#[track_caller]`
    // method observes its definition (the impl method at
    // 12:5).
    run_ok(
        "caller-location-dyn",
        CALLER_LOCATION_DYN_SRC,
        "def 19 22\nimpl 12 5\n",
    );
}

const CALLER_LOCATION_CLOSURE_SRC: &str = r#"
#![feature(stmt_expr_attributes)]
#![feature(closure_track_caller)]
use std::panic::Location;
type Loc = &'static Location<'static>;
#[track_caller]
fn mono_fn<F: Fn() -> Loc>(v: &F) -> Loc { v() }
#[track_caller]
fn mono_once<F: FnOnce() -> Loc>(v: F) -> Loc { v() }
#[track_caller]
fn dyn_mut(v: &mut dyn FnMut() -> Loc) -> Loc { v() }
#[track_caller]
fn dyn_once(v: Box<dyn FnOnce() -> Loc>) -> Loc { v() }
fn main() {
    let mut c = #[track_caller] || Location::caller();
    let d = c();
    println!("direct {} {}", d.line(), d.column());
    let m = mono_fn(&c);
    println!("mono {} {}", m.line(), m.column());
    let dm = dyn_mut(&mut c);
    println!("dynmut-other {}", dm.file() != file!());
    let dn = dyn_once(Box::new(c));
    println!("dynonce-other {}", dn.file() != file!());
    let mo = mono_once(#[track_caller] || Location::caller());
    println!("monoonce {} {}", mo.line(), mo.column());
}
"#;

#[test]
fn run_mode_caller_location_through_closures() {
    // A track-caller closure called directly observes the call
    // site (16:13); through a `#[track_caller]` generic `Fn` /
    // `FnOnce` wrapper the location forwards to the wrapper's
    // call site (18:13 and 24:14); through a `dyn` receiver it
    // observes the `Fn` shim's location in core, not this file.
    run_ok(
        "caller-location-closure",
        CALLER_LOCATION_CLOSURE_SRC,
        "direct 16 13\nmono 18 13\ndynmut-other true\ndynonce-other true\nmonoonce 24 14\n",
    );
}

const RESULT_ERR_FMT_PANIC_DROP_EXITS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("fmt exploded");
    }
}
impl Drop for E {
    fn drop(&mut self) {
        eprintln!("dropped");
        std::process::exit(7);
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_PANIC_DROP_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("fmt exploded");
    }
}
impl Drop for E {
    fn drop(&mut self) {
        eprintln!("dropped");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_PANIC_DROP_PANICS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("fmt exploded");
    }
}
impl Drop for E {
    fn drop(&mut self) {
        panic!("drop exploded");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_STREAMING_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "first")?;
        eprintln!("side effect");
        write!(f, "last")
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_PARTIAL_PANIC_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "partial")?;
        panic!("fmt exploded");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_PARTIAL_EXIT_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "partial")?;
        std::process::exit(5);
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_DERIVED_DEBUG_SRC: &str = r#"
#[derive(Debug)]
struct E {
    x: i32,
    name: String,
}
fn main() -> Result<(), E> {
    Err(E {
        x: 42,
        name: "n".to_string(),
    })
}
"#;

const RESULT_ERR_INNER_FORMAT_DEBUG_SRC: &str = r#"
use std::fmt;
struct Inner;
impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "INNER")
    }
}
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = format!("{:?}", Inner);
        eprintln!("composed={s}");
        write!(f, "outer[{s}]")
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

#[test]
fn run_mode_termination_report_unwind_drop_and_streaming() {
    // A panic in the payload's `Debug` body unwinds through the
    // native report, dropping the error as it goes: the drop
    // runs as the panic's cleanup (its own guest outcome -- an
    // exit -- replaces the panic; a second panic aborts), while
    // an abort or a `process::exit` in the body keeps the
    // no-drop behavior. And the report's formatter writes
    // stream to stderr as they happen: they interleave with the
    // body's own stderr writes in native order, and text
    // written before a panic or an exit stands.
    let dir = temp_dir("termination-report-unwind");
    let src = fixture(
        &dir,
        "err_fmt_panic_drop_exits.rs",
        RESULT_ERR_FMT_PANIC_DROP_EXITS_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(7), "drop exits during unwind cleanup");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "Error: \nthread 'main' (1) panicked at {}:6:9:\nfmt exploded\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\ndropped\n",
            src.display()
        )
        .as_str()
    );
    let src = fixture(&dir, "err_fmt_panic_drop.rs", RESULT_ERR_FMT_PANIC_DROP_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "drop runs during unwind cleanup");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "Error: \nthread 'main' (1) panicked at {}:6:9:\nfmt exploded\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\ndropped\n",
            src.display()
        )
        .as_str()
    );
    let src = fixture(
        &dir,
        "err_fmt_panic_drop_panics.rs",
        RESULT_ERR_FMT_PANIC_DROP_PANICS_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(134), "second panic aborts");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "Error: \nthread 'main' (1) panicked at {0}:6:9:\nfmt exploded\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n\
             \nthread 'main' (1) panicked at {0}:11:9:\ndrop exploded\nstack backtrace:\n\
             \u{20}  0:     0x555555554000 - __rustc::rust_begin_unwind\n\
             \u{20}  1:     0x555555554010 - core::panicking::panic_fmt\n\
             \u{20}  2:     0x555555554020 - <err_fmt_panic_drop_panics::E as std::ops::Drop>::drop\n\
             \nthread 'main' (1) panicked at /rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/panicking.rs:233:5:\n\
             panic in a destructor during cleanup\n\
             thread caused non-unwinding panic. aborting.\n\
             sparagmos-driver run: guest crashed: abort\n",
            src.display()
        )
        .as_str()
    );
    let src = fixture(&dir, "err_streaming_debug.rs", RESULT_ERR_STREAMING_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "streaming Debug::fmt");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: firstside effect\nlast\n"
    );
    let src = fixture(
        &dir,
        "err_partial_panic_debug.rs",
        RESULT_ERR_PARTIAL_PANIC_DEBUG_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "partial text stands on panic");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "Error: partial\nthread 'main' (1) panicked at {}:7:9:\nfmt exploded\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
            src.display()
        )
        .as_str()
    );
    let src = fixture(
        &dir,
        "err_partial_exit_debug.rs",
        RESULT_ERR_PARTIAL_EXIT_DEBUG_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(5), "partial text stands on exit");
    assert_eq!(String::from_utf8_lossy(&out.stderr), "Error: partial");
    let src = fixture(&dir, "err_derived_debug.rs", RESULT_ERR_DERIVED_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "derived Debug payload");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: E { x: 42, name: \"n\" }\n"
    );
    let src = fixture(
        &dir,
        "err_inner_format_debug.rs",
        RESULT_ERR_INNER_FORMAT_DEBUG_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "inner format! composes");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: composed=INNER\nouter[INNER]\n"
    );
}

const RESULT_ERR_FMT_ERROR_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_ERROR_DROP_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}
impl Drop for E {
    fn drop(&mut self) {
        eprintln!("dropped");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_ERROR_PARTIAL_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "partial")?;
        Err(fmt::Error)
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_ERROR_DROP_EXITS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}
impl Drop for E {
    fn drop(&mut self) {
        eprintln!("dropped");
        std::process::exit(7);
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_FMT_ERROR_DROP_PANICS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}
impl Drop for E {
    fn drop(&mut self) {
        panic!("drop exploded");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

#[test]
fn run_mode_termination_report_fmt_error_panics() {
    // A payload `Debug` body that returns `Err(fmt::Error)`
    // does not make the native report fall back or complete:
    // the report's `write_fmt` to stderr panics on the
    // formatting error (core's message), the panic unwinds
    // through the report dropping the payload as its cleanup
    // (a destructor that exits replaces the panic; a second
    // panic in the drop aborts), text the body streamed before
    // returning the error stands, and the run exits 101.
    let dir = temp_dir("termination-report-fmt-error");
    let src = fixture(&dir, "err_fmt_error.rs", RESULT_ERR_FMT_ERROR_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "fmt error panics the report");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \nthread 'main' (1) panicked at /rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/io/write.rs:411:17:\n\
         a formatting trait implementation returned an error when the underlying stream did not\n\
         note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n"
    );
    let src = fixture(&dir, "err_fmt_error_drop.rs", RESULT_ERR_FMT_ERROR_DROP_SRC);
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(101),
        "drop runs during unwind cleanup"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \nthread 'main' (1) panicked at /rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/io/write.rs:411:17:\n\
         a formatting trait implementation returned an error when the underlying stream did not\n\
         note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\ndropped\n"
    );
    let src = fixture(
        &dir,
        "err_fmt_error_partial.rs",
        RESULT_ERR_FMT_ERROR_PARTIAL_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(101),
        "partial text stands on fmt error"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: partial\nthread 'main' (1) panicked at /rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/io/write.rs:411:17:\n\
         a formatting trait implementation returned an error when the underlying stream did not\n\
         note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n"
    );
    let src = fixture(
        &dir,
        "err_fmt_error_drop_exits.rs",
        RESULT_ERR_FMT_ERROR_DROP_EXITS_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(7),
        "drop exits during unwind cleanup"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \nthread 'main' (1) panicked at /rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/io/write.rs:411:17:\n\
         a formatting trait implementation returned an error when the underlying stream did not\n\
         note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\ndropped\n"
    );
    let src = fixture(
        &dir,
        "err_fmt_error_drop_panics.rs",
        RESULT_ERR_FMT_ERROR_DROP_PANICS_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(134), "second panic aborts");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "Error: \nthread 'main' (1) panicked at /rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/io/write.rs:411:17:\n\
             a formatting trait implementation returned an error when the underlying stream did not\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n\
             \nthread 'main' (1) panicked at {0}:11:9:\ndrop exploded\nstack backtrace:\n\
             \u{20}  0:     0x555555554000 - __rustc::rust_begin_unwind\n\
             \u{20}  1:     0x555555554010 - core::panicking::panic_fmt\n\
             \u{20}  2:     0x555555554020 - <err_fmt_error_drop_panics::E as std::ops::Drop>::drop\n\
             \nthread 'main' (1) panicked at /rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/panicking.rs:233:5:\n\
             panic in a destructor during cleanup\n\
             thread caused non-unwinding panic. aborting.\n\
             sparagmos-driver run: guest crashed: abort\n",
            src.display()
        )
        .as_str()
    );
}

const SHOULD_PANIC_MATCH_SRC: &str = r#"
#[test]
#[should_panic(expected = "hello")]
fn panics_with_hello() {
    panic!("hello world");
}
"#;

const SHOULD_PANIC_MISS_SRC: &str = r#"
#[test]
#[should_panic(expected = "bye")]
fn panics_with_hello() {
    panic!("hello world");
}
"#;

#[test]
fn run_mode_libtest_should_panic_expected_message() {
    let dir = temp_dir("libtest-should-panic");
    let src = fixture(&dir, "match.rs", SHOULD_PANIC_MATCH_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout={stdout}");
    let src = fixture(&dir, "miss.rs", SHOULD_PANIC_MISS_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(stdout.contains("FAILED"), "stdout={stdout}");
}

const PANIC_HOOK_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};
static FIRED: AtomicUsize = AtomicUsize::new(0);
fn main() {
    std::panic::set_hook(Box::new(|info| {
        FIRED.fetch_add(1, Ordering::SeqCst);
        println!("hook line {}", info.location().unwrap().line());
    }));
    let r = std::panic::catch_unwind(|| panic!("foobar"));
    assert!(r.is_err());
    println!("fired {}", FIRED.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_panic_hook_fires_with_location() {
    let dir = temp_dir("panic-hook");
    let src = fixture(&dir, "hook.rs", PANIC_HOOK_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The `panic!` is on line 9 of the fixture.
    assert_eq!(stdout, "hook line 9\nfired 1\n", "stdout={stdout}");
}

// ----- Location::file and the fat-pointer transmutes (Lane GE) --------------

const LOCATION_FILE_SRC: &str = r#"
fn main() {
    let loc = std::panic::Location::caller();
    println!("line {}", loc.line());
    println!("file-ok {}", loc.file() == file!());
}
"#;

#[test]
fn run_mode_location_file_reports_the_source_path() {
    // `Location::file` transmutes the stored `NonNull<str>` to a
    // `*const str`: a fat pointer read out of its transparent
    // wrapper. The conversion generalises wrapper transmutes to
    // fat payloads, so the whole accessor now executes.
    let dir = temp_dir("location-file");
    let src = fixture(&dir, "locfile.rs", LOCATION_FILE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // `Location::caller()` is evaluated on line 3 of the fixture.
    assert_eq!(stdout, "line 3\nfile-ok true\n", "stdout={stdout}");
}

const CATCH_UNWIND_O3_DATA_SRC: &str = r#"
#[inline(never)]
fn boom() { std::panic::resume_unwind(Box::new(7i32)); }
fn main() {
    let r = std::panic::catch_unwind(|| 40 + 2);
    let ok1 = matches!(r, Ok(42));
    let r2 = std::panic::catch_unwind(|| -> i32 { boom(); 0 });
    let ok2 = matches!(r2, Err(_));
    std::process::exit(if ok1 && ok2 { 0 } else { 3 });
}
"#;

/// At opt-level 3 `catch_unwind` is inlined into `main`, and its
/// `Data` union local is materialized as an assignment of the
/// uninitialized constant, filled field by field through a raw
/// pointer by `do_call` / `do_catch`, with the zero-sized closure
/// field never stored at all. The run must still complete: the
/// uninitialized assignment is not a type mismatch, the field
/// writes partially initialize the union, and the closure field's
/// single value is synthesized on read.
#[test]
fn run_mode_catch_unwind_o3_data_union() {
    let dir = temp_dir("catch-o3-data");
    let src = fixture(&dir, "case.rs", CATCH_UNWIND_O3_DATA_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-Copt-level=3"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "O3 catch_unwind Data flow must run to completion; stderr={stderr}"
    );
}

const POW_OVERFLOW_SRC: &str = r#"
fn main() { let _ = 2i32.pow(1024); }
"#;

#[test]
fn run_mode_pow_overflow_panics() {
    let dir = temp_dir("pow-overflow");
    let src = fixture(&dir, "pow_overflow.rs", POW_OVERFLOW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "overflowing pow must exit 101; stderr={stderr}"
    );
    assert!(
        stderr.contains("attempt to exponentiate with overflow"),
        "overflow message goes to stderr: {stderr}"
    );
}

/// `std::panic::set_backtrace_style` has no converted body
/// (it writes a std-internal atomic); the runtime records
/// the style. The rustdoc entry program installs its style
/// during startup.
const BACKTRACE_STYLE_SRC: &str = r#"#![feature(panic_backtrace_config)]
fn main() {
    std::panic::set_backtrace_style(std::panic::BacktraceStyle::Short);
    std::panic::set_backtrace_style(std::panic::BacktraceStyle::Full);
    println!("styled");
}
"#;

#[test]
fn run_mode_set_backtrace_style() {
    run_ok("backtrace-style", BACKTRACE_STYLE_SRC, "styled\n");
}
const PANIC_BOX_HOLDER_SRC: &str = r#"
use std::cell::Cell;
use std::ops::Deref;
use std::panic;
struct Holder {
    c: Cell<u32>,
}
fn main() {
    let r = panic::catch_unwind(|| {
        panic::panic_any(Holder { c: Cell::new(1u32) });
    });
    let err = match r {
        Ok(()) => panic!("no panic"),
        Err(e) => e,
    };
    let b = match err.downcast::<Holder>() {
        Ok(b) => b,
        Err(_) => panic!("downcast failed"),
    };
    let hr: &Holder = Deref::deref(&b);
    hr.c.set(5);
    println!("{}", b.c.get());
    std::mem::forget(b);
}
"#;

#[test]
fn run_mode_panic_box_holder_cell_set_through_deref() {
    // A panic payload's by-value downcast hands back its payload
    // in an aggregate-shaped box (the payload value as the box's
    // single field), not an allocation-backed one. The box lives
    // in a place, so the `Deref` builtin's result must denote
    // that place's field 0: with a snapshot of the payload, the
    // `Cell::set` write through the deref cannot reach the box's
    // storage.
    let dir = temp_dir("panic-box-holder");
    let src = fixture(&dir, "panic_box_holder.rs", PANIC_BOX_HOLDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell::set through a panic-payload Box deref must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "5\n");
}

const PANIC_BOX_STRING_SRC: &str = r#"
use std::ops::DerefMut;
use std::panic;
fn main() {
    let r = panic::catch_unwind(|| {
        panic::panic_any(String::from("a"));
    });
    let err = match r {
        Ok(()) => panic!("no panic"),
        Err(e) => e,
    };
    let mut b = match err.downcast::<String>() {
        Ok(b) => b,
        Err(_) => panic!("downcast failed"),
    };
    let s: &mut String = DerefMut::deref_mut(&mut b);
    s.push_str("z");
    println!("{}", b.as_str());
    std::mem::forget(b);
}
"#;

#[test]
fn run_mode_panic_box_string_push_through_deref_mut() {
    // The mutable counterpart: an explicit `DerefMut` call on an
    // aggregate-shaped box in a place must yield a reference to
    // that place's field 0, so the write through the result lands
    // in the box instead of failing on the snapshot the builtin
    // used to hand back ("expected mutable place, found constant
    // aggregate").
    let dir = temp_dir("panic-box-string");
    let src = fixture(&dir, "panic_box_string.rs", PANIC_BOX_STRING_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "push_str through a panic-payload Box deref_mut must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "az\n");
}

const LIBTEST_MISMATCH_SRC: &str = r#"
#[test]
#[should_panic(expected = "foo")]
fn bar_test() { panic!("bar"); }
"#;

#[test]
fn run_mode_libtest_should_panic_mismatch_section_has_report() {
    // A should-panic message mismatch fails with the panic
    // report ahead of the `did not contain` note in the
    // failure section, as libtest prints it.
    let dir = temp_dir("libtest-mismatch");
    let src = fixture(&dir, "mismatch.rs", LIBTEST_MISMATCH_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    let report = stdout.find("panicked at").expect("panic report in section");
    let note = stdout
        .find("note: panic did not contain expected string")
        .expect("mismatch note in section");
    assert!(report < note, "report precedes the note: stdout={stdout}");
    assert!(
        stdout.contains("panic message: \"bar\""),
        "stdout={stdout}"
    );
}

const PANIC_ANY_GENERIC_PAYLOAD_SRC: &str = r#"
use std::any::Any;
use std::cell::Cell;
use std::panic;
fn main() {
    let r = panic::catch_unwind(|| {
        panic::panic_any(Cell::new(7u32));
    });
    let err = match r {
        Ok(()) => panic!("no panic"),
        Err(e) => e,
    };
    println!("is: {}", err.is::<Cell<u32>>());
    match err.downcast_ref::<Cell<u32>>() {
        Some(c) => println!("ref: {}", c.get()),
        None => println!("ref FAILED"),
    }
    match err.downcast::<Cell<u32>>() {
        Ok(c) => println!("downcast: {}", c.get()),
        Err(_) => println!("downcast FAILED"),
    }
}
"#;

#[test]
fn run_mode_panic_any_generic_payload_downcast() {
    // A fabricated payload box (a caught panic's payload) has the
    // same box shape as a constructed one, so the full
    // is / downcast_ref / downcast surface works on a payload
    // whose type carries generic arguments.
    run_ok(
        "panic-any-generic",
        PANIC_ANY_GENERIC_PAYLOAD_SRC,
        "is: true\nref: 7\ndowncast: 7\n",
    );
}

const RESUME_UNWIND_GENERIC_PAYLOAD_SRC: &str = r#"
use std::any::Any;
use std::cell::Cell;
use std::panic;
fn main() {
    let r = panic::catch_unwind(|| {
        panic::panic_any(Cell::new(3u32));
    });
    let err = match r {
        Ok(()) => panic!("no panic"),
        Err(e) => e,
    };
    let r2 = panic::catch_unwind(panic::AssertUnwindSafe(move || {
        panic::resume_unwind(err);
    }));
    let err2 = match r2 {
        Ok(()) => panic!("no panic"),
        Err(e) => e,
    };
    match err2.downcast::<Cell<u32>>() {
        Ok(c) => println!("resumed: {}", c.get()),
        Err(_) => println!("resumed FAILED"),
    }
}
"#;

#[test]
fn run_mode_resume_unwind_generic_payload_downcast() {
    // A resumed panic re-raises the payload itself (the dyn box's
    // pointee), so the payload survives the round trip through
    // resume_unwind and a second catch with its type and value
    // intact.
    run_ok(
        "resume-generic",
        RESUME_UNWIND_GENERIC_PAYLOAD_SRC,
        "resumed: 3\n",
    );
}

const INDEX_LOCATION_SRC: &str = r#"
#[track_caller]
fn caller_line() -> u32 {
    std::panic::Location::caller().line()
}

struct A {
    prev_line: u32,
}

impl std::ops::Index<usize> for A {
    type Output = ();
    fn index(&self, _idx: usize) -> &() {
        assert_eq!(caller_line(), self.prev_line + 1);
        &()
    }
}

fn main() {
    let prev_line = caller_line();
    (A { prev_line })[0];
    println!("ok");
}
"#;

#[test]
fn run_mode_track_caller_location_forwards_through_index() {
    // `Index::index` is `#[track_caller]` through the trait:
    // a location taken inside an index impl names the indexing
    // expression in the caller, not the call inside the impl.
    // The call names the trait method, so the stamp must reach
    // the impl body the dispatch lands on.
    let dir = temp_dir("index-location");
    let src = fixture(&dir, "index_location.rs", INDEX_LOCATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "caller location must forward through Index::index; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const GEN_BLOCK_PANIC_SRC: &str = r#"
#![feature(gen_blocks)]

fn main() {
    let mut iter = gen {
        yield 42;
        panic!("foo");
    };
    assert_eq!(iter.next(), Some(42));
    let mut tmp = std::panic::AssertUnwindSafe(&mut iter);
    match std::panic::catch_unwind(move || tmp.next()) {
        Ok(_) => unreachable!(),
        Err(err) => assert_eq!(*err.downcast::<&'static str>().unwrap(), "foo"),
    }
    match std::panic::catch_unwind(move || iter.next()) {
        Ok(_) => unreachable!(),
        Err(err) => assert_eq!(
            *err.downcast::<&'static str>().unwrap(),
            "`gen fn` should just keep returning `None` after panicking",
        ),
    }
    println!("gen block panic ok");
}
"#;

#[test]
fn run_mode_gen_block_resume_after_panic_payload() {
    // Resuming a panicked `gen` coroutine panics with the
    // `panic_const_gen_fn_none_panic` payload, not with a
    // generic assertion-failure message: the resume guard's
    // assert carries a `ResumedAfterPanic` message whose text
    // is that exact string.
    let dir = temp_dir("gen-block-panic");
    let src = fixture(&dir, "gen_block_panic.rs", GEN_BLOCK_PANIC_SRC);
    let out = run_driver_extra(&src, &["--edition", "2024"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "resume after panic must carry the gen panic payload; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "gen block panic ok\n");
}
const ARRAY_CLONE_UNWIND_MULTIFIELD_SRC: &str = r#"
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
static N: AtomicUsize = AtomicUsize::new(0);
struct S {
    id: u32,
    rc: Rc<()>,
}
impl Clone for S {
    fn clone(&self) -> Self {
        let n = N.fetch_add(1, Ordering::SeqCst);
        if n == 2 {
            panic!("oops on third clone");
        }
        S { id: self.id, rc: self.rc.clone() }
    }
}
fn main() {
    let counter = Rc::new(());
    let arr = [
        S { id: 0, rc: counter.clone() },
        S { id: 1, rc: counter.clone() },
        S { id: 2, rc: counter.clone() },
        S { id: 3, rc: counter.clone() },
    ];
    println!("built: {}", Rc::strong_count(&counter));
    let child = std::panic::catch_unwind(move || {
        let _ = arr.clone();
    });
    println!("is_err: {}", child.is_err());
    println!("count after: {}", Rc::strong_count(&counter));
}
"#;
#[test]
fn run_mode_array_clone_unwind_drops_cloned_prefix() {
    // Cloning an array of multi-field elements through the
    // spec-clone guard: when an element's `Clone` panics, the
    // guard drops exactly the already-cloned prefix. The guard's
    // erased buffer pointer denotes element 0 entered through
    // its `MaybeUninit` wrapper; the drop's slice length is the
    // cloned count, which for a two-field element coincides with
    // the element's own field count -- the two must not be
    // confused when the drop's base place is composed.
    let dir = temp_dir("array-clone-unwind-multifield");
    let src = fixture(&dir, "array_clone_unwind.rs", ARRAY_CLONE_UNWIND_MULTIFIELD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "built: 5\nis_err: true\ncount after: 1\n"
    );
}


const NOUNWIND_FROM_STR_SRC: &str = r#"
#![allow(invalid_null_arguments)]
fn main() {
    let p: *const u8 = std::ptr::null();
    let s = unsafe { std::slice::from_raw_parts(p, 0) };
    println!("{}", s.len());
}
"#;

#[test]
fn run_mode_nounwind_from_str_panic_message() {
    // A UB-check failure inside a library body reports through
    // `panic_nounwind_fmt` with an `Arguments::from_str` value
    // the body built itself: the run must surface the check's
    // own message as the panic payload, not die in the fmt
    // coercion with a type mismatch.
    let dir = temp_dir("nounwind-from-str");
    let src = fixture(&dir, "case.rs", NOUNWIND_FROM_STR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "nounwind panic must exit 134 (guest crash); stderr={stderr}"
    );
    assert!(
        stderr.contains(
            "unsafe precondition(s) violated: slice::from_raw_parts requires the pointer to be aligned and non-null"
        ),
        "the check's message is the payload: {stderr}"
    );
    assert!(
        stderr.contains("This indicates a bug in the program. This Undefined Behavior check is optional"),
        "the full message renders: {stderr}"
    );
    assert!(
        !stderr.contains("expected byte data"),
        "no fmt coercion failure masks the message: {stderr}"
    );
}

const PANIC_PAYLOAD_INTO_RAW_SRC: &str = r#"
use std::panic;
#[derive(Clone, Copy)]
struct Pair { a: i64, b: i64 }
fn main() {
    let result = panic::catch_unwind(|| {
        panic::panic_any(Pair { a: 11, b: 22 });
    });
    let err = result.unwrap_err();
    let raw = Box::into_raw(err) as *mut Pair;
    unsafe {
        println!("fields: {} {}", (*raw).a, (*raw).b);
        let p = *raw;
        println!("whole: {} {}", p.a, p.b);
    }
}
"#;

#[test]
fn run_mode_panic_payload_into_raw_cast_reads_whole_struct() {
    // A caught panic payload travels as a trait object whose
    // base is the one-field box aggregate holding the payload
    // inline. Casting `Box::into_raw` of it to the concrete
    // pointer type must produce a data pointer denoting the
    // payload as a whole — field reads and a whole-struct read
    // through it read the payload, not its first field.
    run_ok(
        "panic-payload-into-raw",
        PANIC_PAYLOAD_INTO_RAW_SRC,
        "fields: 11 22\nwhole: 11 22\n",
    );
}

const STR_PATTERN_PRED_PANIC_DROP_SRC: &str = r#"
use std::rc::Rc;

struct Guard(Rc<i32>);
impl Drop for Guard {
    fn drop(&mut self) {
        println!("guard dropped, count now {}", Rc::strong_count(&self.0));
    }
}

fn main() {
    // A by-value predicate that panics is dropped by the
    // callee's cleanup as the call unwinds, for each of the
    // three searches.
    let rc = Rc::new(1);
    let rc2 = rc.clone();
    let pred = move |_: char| -> bool { let _ = &rc2; panic!("boom") };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| "ab".contains(pred)));
    println!("contains caught={} count={}", r.is_err(), Rc::strong_count(&rc));
    let rc3 = rc.clone();
    let pred2 = move |_: char| -> bool { let _ = &rc3; panic!("boom") };
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| "ab".starts_with(pred2)));
    println!("starts caught={} count={}", r2.is_err(), Rc::strong_count(&rc));
    let rc4 = rc.clone();
    let pred3 = move |_: char| -> bool { let _ = &rc4; panic!("boom") };
    let r3 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| "ab".ends_with(pred3)));
    println!("ends caught={} count={}", r3.is_err(), Rc::strong_count(&rc));
    // A capture with a real destructor runs it during the unwind.
    let rc5 = Rc::new(5);
    let g = Guard(rc5.clone());
    let pred4 = move |_: char| -> bool { let _ = &g; panic!("boom") };
    let r4 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| "ab".contains(pred4)));
    println!("guard caught={} count={}", r4.is_err(), Rc::strong_count(&rc5));
    // A borrowed predicate that panics is NOT dropped by the
    // call; its owner still holds it after the catch.
    let rc6 = Rc::new(6);
    let rc6c = rc6.clone();
    let mut pred5 = move |_: char| -> bool { let _ = &rc6c; panic!("boom") };
    let r5 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| "ab".contains(&mut pred5)));
    println!("borrowed caught={} count={}", r5.is_err(), Rc::strong_count(&rc6));
    drop(pred5);
    println!("borrowed after owner drop count={}", Rc::strong_count(&rc6));
}
"#;

#[test]
fn run_mode_str_pattern_predicate_drops_by_value_captures_on_unwind() {
    // A by-value predicate was moved into the search, so when
    // an invocation panics the caller's cleanup cannot drop
    // it -- the callee's cleanup must, exactly as on the
    // success path. Before the fix the search propagated the
    // panic before handing the predicate's final state back,
    // and the `CallBuiltin` dispatch only ran the attached
    // `consumed_glue` on success, so each panicking call
    // leaked the predicate's captures (the counts stayed
    // elevated). A borrowed predicate keeps its owner and is
    // not dropped by the call on either path.
    run_ok(
        "str-pattern-pred-panic-drop",
        STR_PATTERN_PRED_PANIC_DROP_SRC,
        "contains caught=true count=1\nstarts caught=true count=1\nends caught=true count=1\nguard dropped, count now 2\nguard caught=true count=1\nborrowed caught=true count=2\nborrowed after owner drop count=1\n",
    );
}

const STR_PATTERN_PRED_DOUBLE_PANIC_SRC: &str = r#"
use std::rc::Rc;

struct Guard(Rc<i32>);
impl Drop for Guard {
    fn drop(&mut self) {
        println!("guard dtor runs, count now {}", Rc::strong_count(&self.0));
        panic!("dtor panic");
    }
}

fn main() {
    let rc = Rc::new(1);
    let g = Guard(rc.clone());
    let pred = move |_: char| -> bool { let _ = &g; panic!("boom") };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| SEARCH));
    println!("caught={} count={}", r.is_err(), Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_str_pattern_predicate_cleanup_panic_aborts() {
    // An owned predicate panics, and dropping it (the search's
    // cleanup duty) panics again in a capture's destructor: a
    // double panic, which natively aborts. The run must abort
    // (134, the abort named on stderr) instead of letting the
    // original panic reach the guest's `catch_unwind` -- before
    // the fix the `CallBuiltin` dispatch discarded the cleanup
    // glue's result, so the run printed `caught=true` and
    // exited 0. Each of the three searches shares the dispatch,
    // so each is exercised.
    for (name, call) in [
        ("contains", "\"ab\".contains(pred)"),
        ("starts_with", "\"ab\".starts_with(pred)"),
        ("ends_with", "\"ab\".ends_with(pred)"),
    ] {
        let dir = temp_dir(&format!("str-pattern-pred-double-panic-{name}"));
        let src = fixture(
            &dir,
            "case.rs",
            &STR_PATTERN_PRED_DOUBLE_PANIC_SRC.replace("SEARCH", call),
        );
        let out = run_driver(&src);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(134),
            "a panic in the predicate cleanup must abort with 134 ({name}); stderr={stderr}"
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "guard dtor runs, count now 2\n",
            "only the cleanup destructor runs before the abort ({name}); stderr={stderr}"
        );
        assert!(
            stderr.contains("aborting"),
            "the abort must be named on stderr ({name}); stderr={stderr}"
        );
    }
}

const STR_PATTERN_BOXED_DYN_PRED_UNWIND_SRC: &str = r#"
use std::rc::Rc;

fn main() {
    let rc = Rc::new(1);
    let rc2 = rc.clone();
    let boxed: Box<dyn FnMut(char) -> bool> = Box::new(move |c: char| {
        let _ = &rc2;
        if c == 'b' {
            panic!("boom");
        }
        false
    });
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| "ab".contains(boxed)));
    println!("caught={} count={}", r.is_err(), Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_str_pattern_boxed_dyn_predicate_drops_on_unwind() {
    // The owned trait-object predicate is dropped on the unwind
    // path too: the panicking search still hands the object back
    // for the call's drop glue, so the capture's strong count
    // returns to 1 behind the guest's `catch_unwind`.
    run_ok(
        "str-pattern-boxed-dyn-pred-unwind",
        STR_PATTERN_BOXED_DYN_PRED_UNWIND_SRC,
        "caught=true count=1\n",
    );
}

const STR_PATTERN_BOXED_PRED_UNWIND_SRC: &str = r#"
use std::rc::Rc;

fn main() {
    let rc = Rc::new(1);
    let rc2 = rc.clone();
    let boxed = Box::new(move |c: char| {
        let _ = &rc2;
        if c == 'b' {
            panic!("boom");
        }
        false
    });
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| "ab".contains(boxed)));
    println!("caught={} count={}", r.is_err(), Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_str_pattern_boxed_concrete_predicate_drops_on_unwind() {
    // The hand-back also runs when the boxed predicate panics:
    // the box is dropped on the unwind path, so the capture's
    // strong count returns to 1 behind the guest's `catch_unwind`.
    run_ok(
        "str-pattern-boxed-pred-unwind",
        STR_PATTERN_BOXED_PRED_UNWIND_SRC,
        "caught=true count=1\n",
    );
}

const TWO_FAILURES_SRC: &str = r#"
#[test]
fn a_fail() { panic!("first"); }
#[test]
fn b_fail() { panic!("second"); }
"#;

#[test]
fn run_mode_libtest_backtrace_note_once() {
    // std's panic output carries the backtrace note on the
    // process's first panic only: of the two failure sections,
    // exactly one carries it, as in the native run.
    let dir = temp_dir("libtest-note-once");
    let src = fixture(&dir, "two_failures.rs", TWO_FAILURES_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert_eq!(
        stdout.matches("run with `RUST_BACKTRACE=1`").count(),
        1,
        "stdout={stdout}"
    );
}

const NOCAPTURE_SRC: &str = r#"
#[test]
fn a_fail() { panic!("boom nocapture"); }
#[test]
fn b_pass() {}
"#;

#[test]
fn run_mode_libtest_nocapture_prints_panic_report() {
    // In --nocapture mode the default panic hook prints the
    // failing test's report to stderr as the panic runs; the
    // harness model used to drop it (its failure account
    // assumed the report had already been printed).
    let dir = temp_dir("libtest-nocapture");
    let src = fixture(&dir, "nocapture.rs", NOCAPTURE_SRC);
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "--nocapture"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout} stderr={stderr}");
    assert!(
        stderr.contains("thread 'a_fail' (1) panicked at"),
        "stderr={stderr}"
    );
    assert!(stderr.contains("boom nocapture"), "stderr={stderr}");
    assert_eq!(
        stderr.matches("run with `RUST_BACKTRACE=1`").count(),
        1,
        "stderr={stderr}"
    );
}

const BACKTRACE_SRC: &str = r#"
fn main() {
    let bt = std::backtrace::Backtrace::capture();
    println!("{}", bt);
}
"#;

#[test]
fn run_mode_backtrace_capture_follows_the_environment() {
    // `Backtrace::capture` honours the backtrace settings
    // std consults: with capture enabled its `Display` is
    // the call stack's trace (the guest's `main` among the
    // frames); without, it is the disabled marker.
    let dir = temp_dir("backtrace");
    let src = fixture(&dir, "backtrace.rs", BACKTRACE_SRC);
    let enabled = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .env("RUST_BACKTRACE", "1")
        .output()
        .expect("failed to run sparagmos-driver");
    let stdout = String::from_utf8_lossy(&enabled.stdout);
    assert_eq!(enabled.status.code(), Some(0), "stdout={stdout}");
    assert!(stdout.contains("stack backtrace:"), "stdout={stdout}");
    assert!(stdout.contains("main"), "stdout={stdout}");

    let disabled = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .env_remove("RUST_BACKTRACE")
        .env_remove("RUST_LIB_BACKTRACE")
        .output()
        .expect("failed to run sparagmos-driver");
    let stdout = String::from_utf8_lossy(&disabled.stdout);
    assert_eq!(disabled.status.code(), Some(0), "stdout={stdout}");
    assert_eq!(stdout, "disabled backtrace\n");
}

const DOUBLE_FAIL_SRC: &str = r#"fn main() {
    struct Double;

    impl Drop for Double {
        fn drop(&mut self) {
            panic!("twice")
        }
    }

    let _d = Double;

    panic!("once");
}
"#;

#[test]
fn run_mode_double_fail_reports_destructor_during_cleanup() {
    // A destructor that panics while an earlier panic unwinds
    // through its owner (the `double-fail` child of
    // tests/ui/backtrace/backtrace.rs): the second panic's own
    // report carries the run's one backtrace (the panic-count
    // rule forces it), and the abort panic codegen raises for
    // the cleanup block is `panic_in_cleanup` — "panic in a
    // destructor during cleanup" at panicking.rs:233:5, with no
    // backtrace of its own — not the `panic_cannot_unwind`
    // report a nounwind boundary in ordinary code gets.
    let dir = temp_dir("double-fail");
    let src = fixture(&dir, "double_fail.rs", DOUBLE_FAIL_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(134), "double panic aborts");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "\nthread 'main' (1) panicked at {0}:12:5:\nonce\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n\
             \nthread 'main' (1) panicked at {0}:6:13:\ntwice\nstack backtrace:\n\
             \u{20}  0:     0x555555554000 - __rustc::rust_begin_unwind\n\
             \u{20}  1:     0x555555554010 - core::panicking::panic_fmt\n\
             \u{20}  2:     0x555555554020 - <double_fail::main::Double as std::ops::Drop>::drop\n\
             \u{20}  3:     0x555555554030 - double_fail::main\n\
             \nthread 'main' (1) panicked at /rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/panicking.rs:233:5:\n\
             panic in a destructor during cleanup\n\
             thread caused non-unwinding panic. aborting.\n\
             sparagmos-driver run: guest crashed: abort\n",
            src.display()
        )
        .as_str()
    );
}

const NOUNWIND_SRC: &str = r#"extern "C" fn boom() {
    panic!("in nounwind");
}
fn main() {
    boom();
}
"#;

#[test]
fn run_mode_nounwind_panic_reports_cannot_unwind() {
    // The other `Terminate` reason: a panic crossing a nounwind
    // boundary in ordinary code (no cleanup in progress) is
    // `panic_cannot_unwind` — "panic in a function that cannot
    // unwind" at panicking.rs:225:5, with a backtrace.
    let dir = temp_dir("nounwind");
    let src = fixture(&dir, "nounwind.rs", NOUNWIND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(134), "nounwind panic aborts: {stderr}");
    assert!(
        stderr.contains("library/core/src/panicking.rs:225:5"),
        "cannot-unwind location: {stderr}"
    );
    assert!(
        stderr.contains("panic in a function that cannot unwind"),
        "cannot-unwind message: {stderr}"
    );
    assert!(
        !stderr.contains("panic in a destructor during cleanup"),
        "not the cleanup report: {stderr}"
    );
}

const ASSERT_ARRAY_SRC: &str = r#"
fn main() { assert_eq!([1, 2, 3], [1, 2, 4]); }
"#;

#[test]
fn run_mode_assert_eq_array_operands_render_with_debug() {
    // core renders `assert_failed`'s operands with `{:?}`: an
    // array operand is the bracketed element list, not the
    // value model's tuple form.
    let dir = temp_dir("asserarray");
    let src = fixture(&dir, "asserarray.rs", ASSERT_ARRAY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "assert exits 101: {stderr}");
    assert!(
        stderr.contains(
            "assertion `left == right` failed\n  left: [1, 2, 3]\n right: [1, 2, 4]"
        ),
        "array operands render as Debug: {stderr}"
    );
}

const ASSERT_VEC_SRC: &str = r#"
fn main() { assert_eq!(vec![1, 2, 3], vec![1, 2, 4]); }
"#;

#[test]
fn run_mode_assert_eq_vec_operands_render_with_debug() {
    // A `Vec` operand is its element list under `{:?}`, not
    // the storage enum's internal form.
    let dir = temp_dir("assertvec");
    let src = fixture(&dir, "assertvec.rs", ASSERT_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "assert exits 101: {stderr}");
    assert!(
        stderr.contains(
            "assertion `left == right` failed\n  left: [1, 2, 3]\n right: [1, 2, 4]"
        ),
        "Vec operands render as Debug: {stderr}"
    );
    assert!(
        !stderr.contains("variant 0"),
        "no internal enum form in the message: {stderr}"
    );
}

const ASSERT_STRUCT_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct Point { x: i32, y: i32 }
fn main() { assert_eq!(Point { x: 1, y: 2 }, Point { x: 1, y: 3 }); }
"#;

#[test]
fn run_mode_assert_eq_struct_operands_render_with_debug() {
    // A derived-`Debug` struct operand renders through its
    // `Debug` impl: the type name and named fields.
    let dir = temp_dir("assertstruct");
    let src = fixture(&dir, "assertstruct.rs", ASSERT_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "assert exits 101: {stderr}");
    assert!(
        stderr.contains(
            "assertion `left == right` failed\n  left: Point { x: 1, y: 2 }\n right: Point { x: 1, y: 3 }"
        ),
        "struct operands render as Debug: {stderr}"
    );
}

const ASSERT_NE_MSG_SRC: &str = r#"
fn main() {
    assert_ne!(vec![1, 2], vec![1, 2]);
}
"#;

#[test]
fn run_mode_assert_ne_operands_render_with_debug() {
    // The `!=` comparison names its own failure line and the
    // operands still render with `{:?}`.
    let dir = temp_dir("assertne");
    let src = fixture(&dir, "assertne.rs", ASSERT_NE_MSG_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "assert exits 101: {stderr}");
    assert!(
        stderr.contains(
            "assertion `left != right` failed\n  left: [1, 2]\n right: [1, 2]"
        ),
        "assert_ne operands render as Debug: {stderr}"
    );
}

const ASSERT_CUSTOM_MSG_SRC: &str = r#"
fn main() { assert_eq!([1, 2, 3], [1, 2, 4], "custom note"); }
"#;

#[test]
fn run_mode_assert_eq_custom_message_keeps_debug_operands() {
    // With a custom message the failure line gains `: {msg}`
    // and the operands below it still render with `{:?}`.
    let dir = temp_dir("assertmsg");
    let src = fixture(&dir, "assertmsg.rs", ASSERT_CUSTOM_MSG_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "assert exits 101: {stderr}");
    assert!(
        stderr.contains(
            "assertion `left == right` failed: custom note\n  left: [1, 2, 3]\n right: [1, 2, 4]"
        ),
        "custom message with Debug operands: {stderr}"
    );
}

const GENERIC_ASSERT_SRC: &str = r##"
#![feature(generic_assert)]
use std::panic;

#[derive(Clone, Copy, PartialEq)]
struct NoDebug(i32);

#[derive(Clone, Copy, PartialEq)]
struct WithDebug(i32);
impl std::fmt::Debug for WithDebug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WD({})", self.0)
    }
}

fn grab<F: FnOnce() + panic::UnwindSafe>(f: F) -> String {
    let err = panic::catch_unwind(f).unwrap_err();
    if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&str>() {
        s.to_string()
    } else {
        panic!("payload not a string");
    }
}

fn main() {
    let elem = 1;
    let m1 = grab(|| { assert!(&elem == &3); });
    println!("M1|{m1}|");
    let nd = NoDebug(1);
    let m2 = grab(|| { assert!(nd == NoDebug(3)); });
    println!("M2|{m2}|");
    let wd = WithDebug(7);
    let m3 = grab(|| { assert!(wd == WithDebug(3)); });
    println!("M3|{m3}|");
}
"##;

#[test]
fn run_mode_generic_assert_captures_render_like_core() {
    // core's handwritten `Debug` for `asserting::Capture` is a
    // fixed function of the marker type argument: with
    // `TryCaptureWithDebug` it prints the captured element's own
    // `Debug`; with `TryCaptureWithoutDebug` it prints `N/A`.
    // The structural fmt engine must not render the wrapper's
    // recorded struct shape instead.
    let dir = temp_dir("genericassert");
    let src = fixture(&dir, "genericassert.rs", GENERIC_ASSERT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "caught assert panics terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "M1|Assertion failed: &elem == &3\nWith captures:\n  elem = 1\n|\n\
         M2|Assertion failed: nd == NoDebug(3)\nWith captures:\n  nd = N/A\n|\n\
         M3|Assertion failed: wd == WithDebug(3)\nWith captures:\n  wd = WD(7)\n|\n",
        "capture renderings match core's"
    );
}
