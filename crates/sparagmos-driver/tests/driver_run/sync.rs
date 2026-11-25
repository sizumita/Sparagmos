use super::{Command, Output, Path, fixture, run_driver, run_driver_extra, run_ok, temp_dir};


// ----- RefCell model (Lane CC) -----

const REFCELL_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let mut c = RefCell::new(40);
    {
        let a = c.borrow();
        let b = c.borrow();
        println!("{}", *a + *b);
    }
    *c.borrow_mut() += 2;
    let v = *c.borrow();
    println!("{}", v);
    let old = c.replace(7);
    println!("{}", old);
    {
        let _m = c.borrow_mut();
        println!("{}", c.try_borrow().is_err());
        println!("{}", c.try_borrow_mut().is_err());
    }
    println!("{}", c.try_borrow().is_ok());
    *c.get_mut() = 9;
    let w = *c.borrow();
    println!("{}", w);
}
"#;

#[test]
fn run_mode_refcell_borrow_discipline() {
    let dir = temp_dir("refcell");
    let src = fixture(&dir, "refcell.rs", REFCELL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "RefCell program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "80\n42\n42\ntrue\ntrue\ntrue\n9\n",
        "shared borrows coexist, guards release on drop, try_borrow \
         reports conflicts, replace/get_mut go through the cell"
    );
}

const REFCELL_DROP_SRC: &str = r#"
use std::cell::RefCell;
struct D(i32);
impl Drop for D {
    fn drop(&mut self) {
        println!("dropping {}", self.0);
    }
}
fn main() {
    let c = RefCell::new(D(5));
    println!("borrowed {}", c.borrow().0);
    drop(c);
    println!("done");
}
"#;

#[test]
fn run_mode_refcell_drop_runs_inner_destructor() {
    let dir = temp_dir("refcell-drop");
    let src = fixture(&dir, "refcell_drop.rs", REFCELL_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "RefCell<D> program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "borrowed 5\ndropping 5\ndone\n",
        "dropping the cell runs the wrapped value's destructor"
    );
}

const RANGE_SLICE_VIEW_REFCELL_SRC: &str = r#"
use std::cell::RefCell;

fn main() {
    let src = [RefCell::new(1), RefCell::new(2), RefCell::new(3)];
    let s: &[RefCell<i32>] = &src;
    let view = &s[1..];
    *view[0].borrow_mut() = 20;
    println!("{} {} {}", *src[0].borrow(), *src[1].borrow(), *src[2].borrow());
}
"#;

#[test]
fn run_mode_range_slice_view_refcell_borrow_mut_reaches_source() {
    let dir = temp_dir("range-slice-view-refcell");
    let src = fixture(
        &dir,
        "range_slice_view_refcell.rs",
        RANGE_SLICE_VIEW_REFCELL_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "borrow_mut through a range slice over a fat receiver must alias its storage; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1 20 3\n");
}

const RANGE_SLICE_VIEW_STATIC_ATOMIC_SRC: &str = r#"
use std::sync::atomic::{AtomicI32, Ordering};

static ARR: [AtomicI32; 3] = [AtomicI32::new(1), AtomicI32::new(2), AtomicI32::new(3)];

fn main() {
    let view = &ARR[1..];
    view[0].store(20, Ordering::SeqCst);
    ARR[2].store(30, Ordering::SeqCst);
    println!("{} {} {}", ARR[0].load(Ordering::SeqCst), ARR[1].load(Ordering::SeqCst), view[1].load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_range_slice_view_static_atomic_reaches_source() {
    let dir = temp_dir("range-slice-view-static-atomic");
    let src = fixture(
        &dir,
        "range_slice_view_static_atomic.rs",
        RANGE_SLICE_VIEW_STATIC_ATOMIC_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a shared range slice over a static must alias its storage; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1 20 30\n");
}

const REFCELL_COMPOUND_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let cell = RefCell::new(1u32);
    *cell.borrow_mut() += 1;
    let v = *cell.borrow();
    println!("{v}");
}
"#;

#[test]
fn run_mode_refcell_compound_assign_through_borrow_mut() {
    let dir = temp_dir("refcell-compound");
    let src = fixture(&dir, "refcell_compound.rs", REFCELL_COMPOUND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "compound assignment through a RefMut guard must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "2\n");
}

const TRANSMUTE_LONG_ATOMIC_ARRAY_SRC: &str = r#"
use std::sync::atomic::{AtomicPtr, Ordering};
static A: u8 = 11;
static B: u8 = 22;
fn main() {
    let mut raw: [*mut u8; 63] = [std::ptr::null_mut(); 63];
    raw[0] = std::ptr::addr_of!(A) as *mut u8;
    raw[62] = std::ptr::addr_of!(B) as *mut u8;
    let arr: [AtomicPtr<u8>; 63] = unsafe { std::mem::transmute(raw) };
    let p0 = arr[0].load(Ordering::SeqCst);
    let p62 = arr[62].load(Ordering::SeqCst);
    unsafe { println!("{} {} {}", *p0, *p62, arr[1].load(Ordering::SeqCst).is_null()); }
    arr[0].store(std::ptr::addr_of!(B) as *mut u8, Ordering::SeqCst);
    let back: [*mut u8; 63] = unsafe { std::mem::transmute(arr) };
    unsafe { println!("{} {} {}", *back[0], *back[62], back[1].is_null()); }
}
"#;

#[test]
fn run_mode_transmute_long_atomic_ptr_array() {
    // A long array whose total width exceeds the 128-bit
    // whole-array scalar-shape budget: the elements'
    // shapes carry the reinterpretation instead. The
    // 63-element pointer -> `Atomic` array is the
    // `thread_local` crate's bucket-table transmute.
    run_ok(
        "transmute-long-atomic-array",
        TRANSMUTE_LONG_ATOMIC_ARRAY_SRC,
        "11 22 true\n22 22 true\n",
    );
}

const THREAD_JOIN_OPT_SRC: &str = r#"fn main() {
    let h = std::thread::spawn(|| 40 + 2);
    let v = h.join().unwrap();
    assert_eq!(v, 42);
}
"#;

#[test]
fn run_mode_thread_join_at_optimized_mir() {
    // Optimizing MIR inlines `JoinHandle::join`: field 0 (the
    // `JoinInner`) is moved out of the handle and
    // `JoinInner::join` is called directly. The modelled spawn
    // result nests the thread-id handle inside the join handle
    // so the move yields a `JoinInner`-shaped value, and the
    // inner join is served by the same builtin as
    // `JoinHandle::join`.
    let dir = temp_dir("thread-join-opt");
    let src = fixture(&dir, "thread_join_opt.rs", THREAD_JOIN_OPT_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-O"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
}

const THREAD_LOCAL_EAGER_SRC: &str = r#"
use std::cell::Cell;

thread_local! {
    static COUNTER: Cell<u32> = const { Cell::new(7) };
}

fn main() {
    COUNTER.with(|c| {
        println!("first {}", c.get());
        c.set(42);
        println!("second {}", c.get());
    });
}
"#;

#[test]
fn run_mode_thread_local_eager_cell_get_set() {
    let dir = temp_dir("tls-eager");
    let src = fixture(&dir, "tls_eager.rs", THREAD_LOCAL_EAGER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "first 7\nsecond 42\n");
}

const THREAD_LOCAL_LAZY_SRC: &str = r#"
use std::cell::RefCell;

thread_local! {
    static LIST: RefCell<Vec<u32>> = RefCell::new(vec![1, 2]);
    static NAME: RefCell<String> = RefCell::new(String::from("hello"));
}

fn show(list: &[u32]) -> String {
    let mut s = String::new();
    let mut i = 0;
    while i < list.len() {
        if i > 0 { s.push_str(","); }
        s.push_str(&list[i].to_string());
        i += 1;
    }
    s
}

fn main() {
    LIST.with(|list| {
        list.borrow_mut().push(3);
        println!("list {}", show(&list.borrow()));
    });
    NAME.with(|name| {
        let s: String = name.borrow().clone();
        println!("name {s}");
    });
    let t = std::thread::spawn(|| {
        LIST.with(|list| {
            println!("in thread {}", show(&list.borrow()));
            list.borrow_mut().push(9);
            println!("thread own {}", show(&list.borrow()));
        });
    });
    t.join().unwrap();
    LIST.with(|list| {
        println!("back {}", show(&list.borrow()));
    });
}
"#;

#[test]
fn run_mode_thread_local_lazy_refcell_and_thread_separation() {
    // The lazy storage initializes on first access in each thread:
    // the spawned thread starts from the initializer value (not the
    // main thread's mutated one), its own mutation stays its own,
    // and the main thread's value is unchanged after the join.
    let dir = temp_dir("tls-lazy");
    let src = fixture(&dir, "tls_lazy.rs", THREAD_LOCAL_LAZY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "list 1,2,3\nname hello\nin thread 1,2\nthread own 1,2,9\nback 1,2,3\n"
    );
}

const THREAD_LOCAL_INIT_ONCE_SRC: &str = r#"
use std::cell::Cell;

static mut INIT_RUNS: u32 = 0;

thread_local! {
    static VAL: Cell<u32> = {
        unsafe { INIT_RUNS += 1; }
        Cell::new(5)
    };
}

fn main() {
    VAL.with(|v| println!("a {}", v.get()));
    VAL.with(|v| println!("b {}", v.get()));
    let t = std::thread::spawn(|| {
        VAL.with(|v| println!("t {}", v.get()));
        VAL.with(|v| println!("t2 {}", v.get()));
    });
    t.join().unwrap();
    println!("runs {}", unsafe { INIT_RUNS });
}
"#;

#[test]
fn run_mode_thread_local_initializer_runs_once_per_thread() {
    let dir = temp_dir("tls-init-once");
    let src = fixture(&dir, "tls_init_once.rs", THREAD_LOCAL_INIT_ONCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a 5\nb 5\nt 5\nt2 5\nruns 2\n"
    );
}

const ITER_FOLD_SRC: &str = r#"
fn main() {
    let summed = (0..5).fold(0, |acc, x| acc + x);
    let a = [1, 2, 3];
    let mut weight = 1;
    let folded = a.iter().fold(0, |acc, x| {
        weight *= 2;
        acc + x * weight
    });
    println!("summed={} folded={} weight={}", summed, folded, weight);
}
"#;

#[test]
fn run_mode_iterator_fold_threads_accumulator_and_captures() {
    // The fold closure is FnMut: its captured `weight` must keep
    // its mutations across items (1*2 + 2*4 + 3*8 = 34, w = 8).
    let dir = temp_dir("iter_fold");
    let src = fixture(&dir, "iter_fold.rs", ITER_FOLD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "fold runs; stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "summed=10 folded=34 weight=8\n"
    );
}

const FROM_INTO_WRAPPERS_SRC: &str = r#"
struct Outer { outer: std::cell::Cell<i32> }
fn main() {
    let o = Outer { outer: 1.into() };
    o.outer.set(0);
    println!("{}", o.outer.get());
    let c = std::cell::Cell::from(5);
    println!("{}", c.get());
    let rc: std::cell::RefCell<i32> = 7.into();
    println!("{}", *rc.borrow());
    let rf = std::cell::RefCell::from(9);
    println!("{}", *rf.borrow());
}
"#;

#[test]
fn run_mode_from_into_cell_and_refcell() {
    // The shape of tests/ui/contracts/
    // contracts-requires-is-not-inherited-when-nesting.rs: a
    // `Cell` field built with `1.into()`. `Into::into`'s instance
    // arguments are [Self, T] with T as the destination; reading
    // them in `From`'s order returned the bare integer here.
    let dir = temp_dir("from-into-wrappers");
    let src = fixture(&dir, "case.rs", FROM_INTO_WRAPPERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell/RefCell From and Into conversions must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "0\n5\n7\n9\n");
}

const RC_REFCELL_SRC: &str = r#"
use std::cell::RefCell;
use std::rc::Rc;
fn main() {
    let rc = Rc::new(RefCell::new(1u32));
    *rc.borrow_mut() = 7;
    let v = *rc.borrow();
    println!("{}", v);
    std::mem::forget(rc);
}
"#;

#[test]
fn run_mode_rc_refcell_borrow_mut_assignment() {
    // `RefCell::borrow_mut` is reached through the `Deref`
    // builtin for `Rc`; with a snapshot result the RefCell borrow
    // could not be tracked ("RefCell with no live place (constant
    // aggregate)"). The handle's pointee lives in a place in this
    // value model, so the builtin returns that place and the
    // borrow registers.
    let dir = temp_dir("rc-refcell");
    let src = fixture(&dir, "rc_refcell.rs", RC_REFCELL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "borrow_mut assignment through Rc<RefCell> must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

// ----- stdout-mismatch bucket: TLS destructors, Debug wrapper
// ----- forms, SIMD cast, libtest failure sections (Lane IM) -----

const TLS_DTOR_SRC: &str = r#"
use std::thread;

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        println!("test2");
    }
}

thread_local!(static FOO: Foo = Foo);

fn main() {
    thread::spawn(|| {
        FOO.with(|_| {});
        println!("test1");
    })
    .join()
    .unwrap();
}
"#;

#[test]
fn run_mode_tls_destructor_runs_at_spawned_thread_exit() {
    // The shape of tests/ui/threads-sendsync/issue-29488.rs: a
    // thread-local's destructor runs when the thread that
    // registered it finishes, so the spawned thread's `Foo` is
    // dropped (printing test2) after its own output (test1).
    run_ok("tls-dtor", TLS_DTOR_SRC, "test1\ntest2\n");
}

const UNSIZE_RC_WRAPPERS_SRC: &str = r#"
use std::cell::RefCell;
use std::rc::Rc;
struct Foo<T: ?Sized>(T);
fn main() {
    let rc: Rc<Foo<[u8]>> = Rc::new(Foo([7u8, 8, 9]));
    println!("foo {} {}", rc.0.len(), rc.0[2]);
    let a: Rc<RefCell<[isize; 3]>> = Rc::new(RefCell::new([10, 20, 30]));
    let b: Rc<RefCell<[isize]>> = a;
    println!("cell {} {}", b.borrow().len(), b.borrow()[1]);
    let s: std::sync::Arc<[i32]> = std::sync::Arc::new([4, 5]);
    let w = std::sync::Arc::downgrade(&s);
    println!("arc {} weak {}", s.len(), w.upgrade().map(|x| x.len()).unwrap_or(0));
}
"#;

#[test]
fn run_mode_unsize_rc_wrapped_and_refcell_slices() {
    run_ok(
        "unsize-rc-wrappers",
        UNSIZE_RC_WRAPPERS_SRC,
        "foo 3 9\ncell 3 20\narc 2 weak 2\n",
    );
}
const TLS_DTOR_BOTH_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPPED: AtomicUsize = AtomicUsize::new(0);

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        DROPPED.fetch_add(1, Ordering::SeqCst);
    }
}

thread_local!(static FOO: Foo = Foo);
thread_local!(static BAR: Foo = Foo);

fn main() {
    // A spawned thread's TLS destructors run at its exit, in
    // reverse registration order; try_with inside a destructor
    // observes the destruction.
    std::thread::spawn(|| {
        FOO.with(|_| ());
        BAR.with(|_| ());
    })
    .join()
    .unwrap();
    assert_eq!(DROPPED.load(Ordering::SeqCst), 2);
    // The entry thread's destructors run at program exit.
    FOO.with(|_| ());
    println!("ok");
}
"#;

#[test]
fn run_mode_tls_destructors_run_at_thread_exit() {
    // TLS destructors are registered through std's
    // `__cxa_thread_atexit` wrapper; the model records the
    // registrations per thread and runs them (as ordinary guest
    // code, so the destroy shim's state transitions happen) when
    // the owning thread exits.
    let dir = temp_dir("tls-dtor");
    let src = fixture(&dir, "tls_dtor.rs", TLS_DTOR_BOTH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "TLS destructors must run at thread exit; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}
// ----- modelled handle values under declared-type drop glue ---------
// Drop glue built from a declared type can name a handle at a
// position where a builtin model keeps a scalar stand-in: the
// thread model's `Thread` value is an id, although the declared
// type's field is an `Arc`. Dropping a struct that holds such a
// value runs the handle plan against the id; a scalar holds no
// count, so the drop completes (and the struct's own destructor
// still runs). A real handle aggregate in the same position is
// still released.

const THREAD_HOLDER_SRC: &str = r#"
struct ThreadHolder(std::thread::Thread);

struct Observed(std::thread::Thread);
impl Drop for Observed {
    fn drop(&mut self) {
        println!("observed");
    }
}

fn main() {
    let h = ThreadHolder(std::thread::current());
    drop(h);
    let o = Observed(std::thread::current());
    drop(o);
    let same = std::thread::current().id() == std::thread::current().id();
    println!("same {same}");
    println!("done");
}
"#;

#[test]
fn run_mode_thread_value_in_struct_field_drops() {
    let dir = temp_dir("thread-holder-drop");
    let src = fixture(&dir, "thread_holder.rs", THREAD_HOLDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a struct holding a modelled Thread value must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "observed\nsame true\ndone\n",
        "the struct destructor runs and the modelled handle drop is inert"
    );
}

// `std::thread::Thread::name` has no MIR body in the distributed
// std; the interpreter answers it from the names registered at
// spawn time (the `Builder` name field), with `main` for the
// main thread and `None` for an unnamed spawned thread.
const THREAD_NAME_SRC: &str = r#"
fn main() {
    println!("main-name={:?}", std::thread::current().name());
    let named = std::thread::Builder::new()
        .name("worker-1".to_string())
        .spawn(|| {
            println!("worker-name={:?}", std::thread::current().name());
        })
        .unwrap();
    named.join().unwrap();
    let plain = std::thread::spawn(|| {
        println!("plain-name={:?}", std::thread::current().name());
    });
    plain.join().unwrap();
}
"#;

#[test]
fn run_mode_thread_name() {
    run_ok(
        "thread-name",
        THREAD_NAME_SRC,
        "main-name=Some(\"main\")\n\
         worker-name=Some(\"worker-1\")\n\
         plain-name=None\n",
    );
}

const BARRIER_SRC: &str = r#"
use std::sync::{Arc, Barrier};
use std::thread;
fn main() {
    let barrier = Arc::new(Barrier::new(2));
    let other = barrier.clone();
    let t = thread::spawn(move || {
        let result = other.wait();
        println!("thread leader={}", result.is_leader());
    });
    let result = barrier.wait();
    println!("main leader={}", result.is_leader());
    t.join().unwrap();
    println!("joined");
}
"#;

#[test]
fn run_mode_barrier_releases_both_parties() {
    // A `Barrier` wait by the first party is the cooperative
    // scheduling point: the second party runs, completes the
    // generation as its leader, and both proceed.
    let dir = temp_dir("barrier");
    let src = fixture(&dir, "barrier.rs", BARRIER_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "barrier guest must exit 0; stdout={stdout} stderr={stderr}"
    );
    assert!(stdout.contains("thread leader=true"), "stdout={stdout}");
    assert!(stdout.contains("main leader=false"), "stdout={stdout}");
    assert!(stdout.contains("joined"), "stdout={stdout}");
}

/// Run the driver in `--test` harness mode: the fixture is a
/// test crate and the modelled libtest harness runs its tests.
fn run_driver_test_harness(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .arg("--rustc-arg")
        .arg("--test")
        .output()
        .expect("failed to run sparagmos-driver")
}

const SCHED_SPIN_TRY_RECV_SRC: &str = r#"use std::sync::mpsc::channel;
fn main() {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        for i in 0..5 {
            tx.send(i).unwrap();
        }
    });
    // The entry thread spins on `try_recv`: every call must
    // honestly report `Empty` until the producer has run, and
    // the producer only runs if the spinner yields the
    // scheduler (the shape of tests/ui/threads-sendsync's
    // issue-9396, which starved before time-slicing).
    let mut got = Vec::new();
    while got.len() < 5 {
        if let Ok(v) = rx.try_recv() {
            got.push(v);
        }
    }
    assert_eq!(got, vec![0, 1, 2, 3, 4]);
    println!("try_recv spin ok");
}
"#;

#[test]
fn run_mode_scheduler_spin_try_recv_interleaves() {
    let dir = temp_dir("sched-spin-try-recv");
    let src = fixture(&dir, "spin_try_recv.rs", SCHED_SPIN_TRY_RECV_SRC);
    let out = run_driver(&src);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "try_recv spin ok\n");
}

const SCHED_PRODUCER_CONSUMER_SRC: &str = r#"use std::sync::mpsc::channel;
fn main() {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let mut i = 0;
        loop {
            tx.send(i).unwrap();
            i += 1;
        }
    });
    // An unbounded producer drained by the entry thread's own
    // `recv`: the producer must yield back to the blocked
    // consumer instead of filling the channel until the step
    // budget ends the run (the shape of issue-8827).
    for expected in 0..20 {
        assert_eq!(rx.recv().unwrap(), expected);
    }
    println!("producer consumer ok");
}
"#;

#[test]
fn run_mode_scheduler_unbounded_producer_yields_to_consumer() {
    let dir = temp_dir("sched-producer-consumer");
    let src = fixture(&dir, "producer_consumer.rs", SCHED_PRODUCER_CONSUMER_SRC);
    let out = run_driver(&src);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "producer consumer ok\n"
    );
}

const SCHED_SPIN_BARRIER_SRC: &str = r#"use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
fn main() {
    let arrived = Arc::new(AtomicUsize::new(0));
    let a2 = arrived.clone();
    let h = std::thread::spawn(move || {
        a2.fetch_add(1, Ordering::SeqCst);
        while a2.load(Ordering::SeqCst) < 2 {}
        7
    });
    arrived.fetch_add(1, Ordering::SeqCst);
    while arrived.load(Ordering::SeqCst) < 2 {}
    assert_eq!(h.join().unwrap(), 7);
    println!("spin barrier ok");
}
"#;

#[test]
fn run_mode_scheduler_guest_spin_barrier_rendezvous() {
    let dir = temp_dir("sched-spin-barrier");
    let src = fixture(&dir, "spin_barrier.rs", SCHED_SPIN_BARRIER_SRC);
    let out = run_driver(&src);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "spin barrier ok\n");
}

const SCHED_JOIN_FAIRNESS_SRC: &str = r#"use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
fn main() {
    let flag = Arc::new(AtomicBool::new(false));
    let f2 = flag.clone();
    let spinner = std::thread::spawn(move || {
        while !f2.load(Ordering::SeqCst) {}
        9
    });
    let f3 = flag.clone();
    let setter = std::thread::spawn(move || {
        let mut x = 0u64;
        for i in 0..1000 {
            x += i;
        }
        f3.store(true, Ordering::SeqCst);
        x
    });
    // Joining the spinner first must not starve the setter:
    // the join's scheduling passes interleave both threads.
    assert_eq!(spinner.join().unwrap(), 9);
    assert_eq!(setter.join().unwrap(), 499500);
    println!("join fairness ok");
}
"#;

#[test]
fn run_mode_scheduler_join_does_not_starve_third_thread() {
    let dir = temp_dir("sched-join-fairness");
    let src = fixture(&dir, "join_fairness.rs", SCHED_JOIN_FAIRNESS_SRC);
    let out = run_driver(&src);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "join fairness ok\n");
}

const SCHED_HARNESS_SPIN_SRC: &str = r#"use std::sync::mpsc::channel;

fn spin_on_channel() {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        for i in 0..5 {
            tx.send(i).unwrap();
        }
    });
    let mut got = Vec::new();
    while got.len() < 5 {
        if let Ok(v) = rx.try_recv() {
            got.push(v);
        }
    }
    assert_eq!(got, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_body_threads_interleave() {
    spin_on_channel();
}
"#;

#[test]
fn run_mode_scheduler_harness_test_body_interleaves() {
    let dir = temp_dir("sched-harness-spin");
    let src = fixture(&dir, "harness_spin.rs", SCHED_HARNESS_SPIN_SRC);
    let out = run_driver_test_harness(&src);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("test result: ok. 1 passed"),
        "stdout={stdout}"
    );
}

// ----- mpsc iterators behind adapters (Issue #513) -----
//
// The mpsc iterator step (`Iter` / `IntoIter` blocking on
// `recv`, `TryIter` on `try_recv`) lives in
// `iterator_next_state`; the adapter arms (`Take`, `Map`,
// ...) step their inner iterators through `next_on_state`,
// which had no mpsc arms, so an mpsc iterator behind an
// adapter died with "Iterator::next for
// std::sync::mpsc::Iter<usize> is not runtime-supported" --
// the shape `tests/ui/threads-sendsync/mpsc_stress.rs`
// reaches through `rx.iter().take(N).collect()`.

const MPSC_ITER_TAKE_SRC: &str = r#"
use std::sync::mpsc::channel;
fn main() {
    let (tx, rx) = channel();
    for i in 0..5usize { tx.send(i).unwrap(); }
    let v: Vec<_> = rx.iter().take(3).collect();
    println!("take: {v:?}");
}
"#;

#[test]
fn run_mode_mpsc_iter_take_collect() {
    run_ok("mpsc-iter-take", MPSC_ITER_TAKE_SRC, "take: [0, 1, 2]\n");
}

const MPSC_ITER_TAKE_BLOCKING_SRC: &str = r#"
use std::sync::mpsc::channel;
use std::thread;
fn main() {
    let (tx, rx) = channel::<usize>();
    let th = thread::spawn(move || {
        for i in 0..4usize { tx.send(i).expect("send"); }
    });
    let mut v: Vec<_> = rx.iter().take(4).collect();
    v.sort();
    println!("take: {v:?}");
    th.join().unwrap();
}
"#;

#[test]
fn run_mode_mpsc_iter_take_blocking_producer() {
    // The blocking `recv` inside the adapter step runs
    // scheduling rounds: the spawned producer is driven
    // while the entry thread waits on the empty channel.
    run_ok(
        "mpsc-iter-take-blocking",
        MPSC_ITER_TAKE_BLOCKING_SRC,
        "take: [0, 1, 2, 3]\n",
    );
}

const MPSC_INTO_ITER_BLOCKING_SRC: &str = r#"
use std::sync::mpsc::channel;
use std::thread;
fn main() {
    let (tx, rx) = channel::<usize>();
    let _ = tx.clone(); // convert to shared; the clone drops at once
    let th = thread::spawn(move || {
        tx.send(17).expect("send");
        drop(tx);
    });
    let v: Vec<_> = rx.into_iter().collect();
    println!("into_iter: {v:?}");
    th.join().unwrap();
}
"#;

#[test]
fn run_mode_mpsc_into_iter_blocking_collect() {
    // The only live sender is the spawned thread's: the
    // collect must not report a would-block error before
    // that thread has run, and must end when it drops the
    // sender after sending.
    run_ok(
        "mpsc-into-iter-blocking",
        MPSC_INTO_ITER_BLOCKING_SRC,
        "into_iter: [17]\n",
    );
}

const MPSC_TRY_ITER_TAKE_SRC: &str = r#"
use std::sync::mpsc::channel;
fn main() {
    let (tx, rx) = channel::<usize>();
    tx.send(1).unwrap();
    tx.send(2).unwrap();
    tx.send(3).unwrap();
    let v: Vec<_> = rx.try_iter().take(2).collect();
    println!("try_take: {v:?}");
}
"#;

#[test]
fn run_mode_mpsc_try_iter_take_collect() {
    run_ok("mpsc-try-iter-take", MPSC_TRY_ITER_TAKE_SRC, "try_take: [1, 2]\n");
}

const MPSC_ITER_MAP_SRC: &str = r#"
use std::sync::mpsc::channel;
fn main() {
    let (tx, rx) = channel::<usize>();
    tx.send(2).unwrap();
    drop(tx);
    let v: Vec<_> = rx.iter().map(|x| x * 10).collect();
    println!("map: {v:?}");
}
"#;

#[test]
fn run_mode_mpsc_iter_map_collect() {
    run_ok("mpsc-iter-map", MPSC_ITER_MAP_SRC, "map: [20]\n");
}
