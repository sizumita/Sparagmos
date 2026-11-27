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

const THREAD_PARK_SRC: &str = r#"
use std::thread;
use std::time::Duration;
fn main() {
    let t = thread::current();
    let h = thread::spawn(move || {
        t.unpark();
    });
    thread::park();
    println!("unparked");
    thread::park_timeout(Duration::from_millis(10));
    println!("timeout done");
    h.join().unwrap();
}
"#;

#[test]
fn run_mode_thread_park_and_timeout() {
    // `std::thread::park` / `park_timeout` have no MIR body in
    // the pinned std, and `Thread::unpark`'s body cannot run
    // over the model's id-handle thread values: the three are
    // served by the interpreter's per-thread park token. The
    // first park consumes the spawned thread's unpark; the
    // timed park returns with no unparker left.
    run_ok("thread-park", THREAD_PARK_SRC, "unparked\ntimeout done\n");
}

const THREAD_PARK_PRESET_SRC: &str = r#"
use std::sync::mpsc;
use std::thread;
fn main() {
    thread::current().unpark();
    thread::park();
    println!("preset ok");
    let (tx, rx) = mpsc::channel();
    let h = thread::spawn(move || {
        tx.send(thread::current()).unwrap();
        thread::park();
        println!("child resumed");
        42
    });
    let child = rx.recv().unwrap();
    child.unpark();
    assert_eq!(h.join().unwrap(), 42);
    println!("all done");
}
"#;

#[test]
fn run_mode_thread_park_preset_token_and_spawned_parker() {
    // A token set before the park is consumed by it (no wait),
    // and a spawned thread parked inside its own run is
    // suspended and resumed by the main thread's unpark.
    run_ok(
        "thread-park-preset",
        THREAD_PARK_PRESET_SRC,
        "preset ok\nchild resumed\nall done\n",
    );
}

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

// ----- scheduler pending set (Issue #399) -----
//
// The scheduler's rounds, drains and yield decisions walk
// an incrementally maintained set of the threads that can
// still be activated, not the whole thread table (which
// keeps every finished thread's outcome for `join`). The
// rendezvous below only complete if the bookkeeping stays
// exact across the transitions: a live sender must be found
// by a `recv_timeout` retry loop after hundreds of threads
// have spawned and finished, and a parked barrier waiter
// must be resumed.

const PENDING_SET_SRC: &str = r#"
use std::sync::mpsc::channel;
use std::time::Duration;

fn main() {
    // Churn: many threads spawn, run and finish; their
    // slots stay in the thread table with their outcomes.
    for _ in 0..200 {
        let h = std::thread::spawn(|| 1 + 1);
        assert_eq!(h.join().unwrap(), 2);
    }
    // A recv_timeout loop that only the one live sender can
    // satisfy: every timeout asks the scheduler for a round,
    // and the round must find the sender among the dead
    // slots.
    let (tx, rx) = channel();
    let sender = std::thread::spawn(move || {
        tx.send(17).unwrap();
    });
    let mut got = None;
    for _ in 0..100000 {
        match rx.recv_timeout(Duration::from_micros(1)) {
            Ok(v) => {
                got = Some(v);
                break;
            }
            Err(_) => continue,
        }
    }
    assert_eq!(got, Some(17));
    sender.join().unwrap();
    // A parked-thread rendezvous after the churn: both
    // barrier parties must be resumed.
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let barrier2 = barrier.clone();
    let h = std::thread::spawn(move || {
        barrier2.wait();
        5
    });
    barrier.wait();
    assert_eq!(h.join().unwrap(), 5);
    println!("pending set ok");
}
"#;

#[test]
fn run_mode_scheduler_pending_set_after_thread_churn() {
    run_ok("sched-pending-set", PENDING_SET_SRC, "pending set ok\n");
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


// ----- Sync lock guards (Issue #541) -----
//
// `Mutex::lock` / `RwLock::read` / `RwLock::write` execute
// through the converted std bodies; what the runtime adds is
// the guards' `Deref` / `DerefMut` (the pointee is the lock
// aggregate's data field -- for a read guard, the `NonNull`
// data pointer the constructor stored, which denotes the
// lock's `UnsafeCell` place), the guards' `Drop` running the
// converted destructor body (which releases the lock), and
// `Display` / `Debug` forwarding to the guarded value.

const MUTEX_GUARD_SRC: &str = r#"
use std::sync::Mutex;

fn main() {
    let m = Mutex::new(41i32);
    {
        let mut g = m.lock().unwrap();
        *g += 1;
        println!("in-lock: {}", *g);
        println!("display-guard: {}", g);
        println!("debug-guard: {:?}", g);
    }
    let g2 = m.lock().unwrap();
    println!("after: {}", *g2);
}
"#;

#[test]
fn run_mode_mutex_guard_deref_mut_and_fmt() {
    let dir = temp_dir("mutex-guard");
    let src = fixture(&dir, "mutex_guard.rs", MUTEX_GUARD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "MutexGuard program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "in-lock: 42\ndisplay-guard: 42\ndebug-guard: 42\nafter: 42\n",
        "mutation through the guard lands in the lock, the guard \
         formats as its pointee, and dropping it releases the lock"
    );
}

const RWLOCK_GUARD_SRC: &str = r#"
use std::sync::RwLock;

fn main() {
    let l = RwLock::new(String::from("hello"));
    {
        let r = l.read().unwrap();
        println!("read: {}", *r);
        println!("read-display: {}", r);
    }
    {
        let mut w = l.write().unwrap();
        w.push_str(" world");
        println!("write-debug: {:?}", w);
    }
    let r1 = l.read().unwrap();
    let r2 = l.read().unwrap();
    println!("two readers: {} {}", *r1, *r2);
    println!("try_write while read: {}", l.try_write().is_err());
    drop(r1);
    drop(r2);
    println!("final: {}", *l.read().unwrap());
}
"#;

#[test]
fn run_mode_rwlock_guards_deref_and_try() {
    let dir = temp_dir("rwlock-guard");
    let src = fixture(&dir, "rwlock_guard.rs", RWLOCK_GUARD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "RwLock guard program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "read: hello\nread-display: hello\nwrite-debug: \"hello world\"\n\
         two readers: hello world hello world\ntry_write while read: true\n\
         final: hello world\n",
        "read/write guards deref to the shared value, readers \
         coexist, try_write reports the read hold, and the write \
         guard's mutation is visible after its drop"
    );
}

const GUARD_DROP_RELEASE_SRC: &str = r#"
use std::sync::{Arc, Mutex};

fn main() {
    let m = Mutex::new(0i32);
    for _ in 0..5 {
        let mut g = m.lock().unwrap();
        *g += 1;
    }
    println!("looped: {}", *m.lock().unwrap());
    let g = m.lock().unwrap();
    println!("try while held: {}", m.try_lock().is_err());
    drop(g);
    println!("try after drop: {}", m.try_lock().is_ok());

    let counter = Arc::new(Mutex::new(0i32));
    let c2 = Arc::clone(&counter);
    let h = std::thread::spawn(move || {
        for _ in 0..10 {
            *c2.lock().unwrap() += 1;
        }
    });
    h.join().unwrap();
    println!("count: {}", *counter.lock().unwrap());
}
"#;

#[test]
fn run_mode_mutex_guard_drop_releases_lock() {
    let dir = temp_dir("guard-drop-release");
    let src = fixture(&dir, "guard_drop_release.rs", GUARD_DROP_RELEASE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "guard drop program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "looped: 5\ntry while held: true\ntry after drop: true\ncount: 10\n",
        "a dropped guard releases the mutex (re-locking and \
         try_lock succeed), including across a spawned thread"
    );
}

const READ_GUARD_VEC_SRC: &str = r#"
use std::sync::RwLock;

fn main() {
    let v = RwLock::new(vec![5, 6, 7]);
    let r = v.read().unwrap();
    println!("len: {}", r.len());
    println!("first: {}", r[0]);
    println!("dbg: {:?}", *r);
    println!("guard dbg: {:?}", r);
}
"#;

#[test]
fn run_mode_rwlock_read_guard_vec_pointee() {
    let dir = temp_dir("read-guard-vec");
    let src = fixture(&dir, "read_guard_vec.rs", READ_GUARD_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "read guard Vec program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len: 3\nfirst: 5\ndbg: [5, 6, 7]\nguard dbg: [5, 6, 7]\n",
        "the read guard's stored data pointer denotes the \
         lock's UnsafeCell place, so the pointee is the cell's \
         contents (the whole Vec, not a prefix of it)"
    );
}

const GUARD_UNWIND_RELEASE_SRC: &str = r#"
use std::sync::RwLock;

fn main() {
    let l = RwLock::new(7i32);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _r = l.read().unwrap();
        panic!("x");
    }));
    assert!(r.is_err());
    println!("write after unwind: {}", *l.write().unwrap());
    let l2 = RwLock::new(3i32);
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _w = l2.write().unwrap();
        panic!("y");
    }));
    assert!(r2.is_err());
    let g = l2.read().unwrap_or_else(|e| e.into_inner());
    println!("read after write-unwind: {}", *g);
}
"#;

#[test]
fn run_mode_lock_guards_release_on_unwind() {
    let dir = temp_dir("guard-unwind-release");
    let src = fixture(&dir, "guard_unwind_release.rs", GUARD_UNWIND_RELEASE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "guard unwind program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "write after unwind: 7\nread after write-unwind: 3\n",
        "a guard dropped by unwinding still runs its destructor, \
         so the lock can be taken again afterwards"
    );
}

const POISON_ON_UNWIND_SRC: &str = r#"
use std::sync::{Mutex, RwLock};

fn main() {
    let m = Mutex::new(0);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = m.lock().unwrap();
        panic!("boom");
    }));
    assert!(r.is_err());
    println!("mutex poisoned: {}", m.is_poisoned());
    println!("mutex lock err: {}", m.lock().is_err());
    let l = RwLock::new(1);
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = l.write().unwrap();
        panic!("boom2");
    }));
    assert!(r2.is_err());
    println!("rwlock poisoned: {}", l.is_poisoned());
    println!("panicking outside: {}", std::thread::panicking());
}
"#;

#[test]
fn run_mode_lock_guards_poison_on_unwind() {
    let dir = temp_dir("guard-poison-unwind");
    let src = fixture(&dir, "guard_poison_unwind.rs", POISON_ON_UNWIND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "poison program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "mutex poisoned: true\nmutex lock err: true\n\
         rwlock poisoned: true\npanicking outside: false\n",
        "a guard dropped by unwinding observes `panicking()` and \
         poisons its lock; outside unwinding the query is false"
    );
}

const PANICKING_IN_DROP_SRC: &str = r#"
struct Probe;

impl Drop for Probe {
    fn drop(&mut self) {
        println!("drop panicking: {}", std::thread::panicking());
    }
}

fn main() {
    let _ = std::panic::catch_unwind(|| {
        let _p = Probe;
        panic!("x");
    });
    let _p2 = Probe;
    println!("done");
}
"#;

#[test]
fn run_mode_panicking_observed_in_drop_during_unwind() {
    let dir = temp_dir("panicking-in-drop");
    let src = fixture(&dir, "panicking_in_drop.rs", PANICKING_IN_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "panicking-in-drop program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop panicking: true\ndone\ndrop panicking: false\n",
        "a destructor run by unwinding observes `panicking() == true`; \
         the same destructor at normal scope exit observes false"
    );
}

const RESUMED_PANICKING_SRC: &str = r#"
struct Probe;

impl Drop for Probe {
    fn drop(&mut self) {
        println!("drop panicking: {}", std::thread::panicking());
    }
}

fn main() {
    let r = std::panic::catch_unwind(|| panic!("first"));
    let payload = r.unwrap_err();
    println!("caught panicking: {}", std::thread::panicking());
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _p = Probe;
        std::panic::resume_unwind(payload);
    }));
    println!("after panicking: {}", std::thread::panicking());
}
"#;

#[test]
fn run_mode_resumed_panic_panicking_in_drop() {
    let dir = temp_dir("resumed-panicking");
    let src = fixture(&dir, "resumed_panicking.rs", RESUMED_PANICKING_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "resumed-panic program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "caught panicking: false\ndrop panicking: true\nafter panicking: false\n",
        "a resumed panic counts as in flight again while it unwinds, \
         and stops counting once the outer catch consumes it"
    );
}

const TAIL_EXPR_LOCK_POISON_SRC: &str = r#"
use std::sync::Mutex;

struct PanicOnDrop;

impl Drop for PanicOnDrop {
    fn drop(&mut self) {
        panic!()
    }
}

fn f(m: &Mutex<i32>) -> i32 {
    let _x = PanicOnDrop;
    *m.lock().unwrap()
}

fn main() {
    let m = Mutex::new(0);
    let _ = std::panic::catch_unwind(|| f(&m));
    println!("poisoned: {}", m.is_poisoned());
}
"#;

#[test]
fn run_mode_tail_expr_temporary_guard_poisons_on_unwind() {
    let dir = temp_dir("tail-expr-lock-poison");
    let src = fixture(
        &dir,
        "tail_expr_lock_poison.rs",
        TAIL_EXPR_LOCK_POISON_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "tail-expr poison program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "poisoned: true\n",
        "the tail-expression guard temporary is still held when the \
         local's destructor panics, so its unwind drop poisons the lock"
    );
}

// ----- Mapped lock guards (PR #584 remainder) -----
//
// `MutexGuard::map` / `RwLockWriteGuard::map` /
// `RwLockReadGuard::map` (and the mapped guards' own `map`)
// build a guard whose first field is the `NonNull` pointer
// to the mapped component itself: `Deref` / `DerefMut`
// return that pointer's pointee, `Display` / `Debug` forward
// to it, and the converted `Drop` body releases the lock
// through the sys lock reference stored next to the pointer.

const MAPPED_MUTEX_GUARD_SRC: &str = r#"
#![feature(mapped_lock_guards)]
use std::sync::{MappedMutexGuard, Mutex, MutexGuard};

struct Pair { a: i32, b: String }

fn main() {
    let m = Mutex::new(Pair { a: 10, b: String::from("bee") });
    {
        let mut g = MutexGuard::map(m.lock().unwrap(), |p| &mut p.b);
        println!("mapped: {}", *g);
        g.push_str("!");
        println!("display: {}", g);
        println!("debug: {:?}", g);
    }
    println!("try after drop: {}", m.try_lock().is_ok());
    {
        let mut g = MutexGuard::map(m.lock().unwrap(), |p| &mut p.a);
        *g += 1;
        // A second mapping step, from the mapped guard itself.
        let mut g2 = MappedMutexGuard::map(g, |a| a);
        *g2 += 100;
        println!("chained: {}", *g2);
    }
    let orig = m.lock().unwrap();
    let failed = MutexGuard::filter_map(orig, |p| {
        if p.a > 1000 { Some(&mut p.b) } else { None }
    });
    println!("filter_map err: {}", failed.is_err());
    let orig = failed.unwrap_err();
    println!("orig back: {} {}", orig.a, orig.b);
    drop(orig);
    let g = m.lock().unwrap();
    println!("final: {} {}", g.a, g.b);
}
"#;

#[test]
fn run_mode_mapped_mutex_guard_deref_mut_fmt_and_drop() {
    run_ok(
        "mapped-mutex-guard",
        MAPPED_MUTEX_GUARD_SRC,
        "mapped: bee\ndisplay: bee!\ndebug: \"bee!\"\ntry after drop: true\n\
         chained: 111\nfilter_map err: true\norig back: 111 bee!\n\
         final: 111 bee!\n",
    );
}

const MAPPED_RWLOCK_GUARD_SRC: &str = r#"
#![feature(mapped_lock_guards)]
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

struct Pair { a: i32, b: i32 }

fn main() {
    let l = RwLock::new(Pair { a: 1, b: 2 });
    {
        let mut w = RwLockWriteGuard::map(l.write().unwrap(), |p| &mut p.b);
        println!("write mapped: {}", *w);
        *w += 20;
        println!("write display: {}", w);
        println!("try_read while mapped write: {}", l.try_read().is_err());
    }
    {
        let r = RwLockReadGuard::map(l.read().unwrap(), |p| &p.b);
        println!("read mapped: {}", *r);
        println!("read debug: {:?}", r);
        println!("try_write while mapped read: {}", l.try_write().is_err());
    }
    println!("try_write after drops: {}", l.try_write().is_ok());
    println!("final: {} {}", l.read().unwrap().a, l.read().unwrap().b);
}
"#;

#[test]
fn run_mode_mapped_rwlock_guards_deref_fmt_and_drop() {
    run_ok(
        "mapped-rwlock-guard",
        MAPPED_RWLOCK_GUARD_SRC,
        "write mapped: 2\nwrite display: 22\n\
         try_read while mapped write: true\nread mapped: 22\nread debug: 22\n\
         try_write while mapped read: true\ntry_write after drops: true\n\
         final: 1 22\n",
    );
}

const MAPPED_GUARD_SLICE_SRC: &str = r#"
#![feature(mapped_lock_guards)]
use std::sync::{Mutex, MutexGuard};

fn main() {
    let m = Mutex::new(vec![10i32, 20, 30]);
    {
        let mut g = MutexGuard::map(m.lock().unwrap(), |v| &mut v[..]);
        println!("slice len: {}", g.len());
        g[1] = 99;
        println!("slice: {:?}", g);
    }
    println!("final: {:?}", *m.lock().unwrap());
}
"#;

#[test]
fn run_mode_mapped_mutex_guard_slice_pointee() {
    run_ok(
        "mapped-mutex-guard-slice",
        MAPPED_GUARD_SLICE_SRC,
        "slice len: 3\nslice: [10, 99, 30]\nfinal: [10, 99, 30]\n",
    );
}

const MUTEX_DEBUG_SRC: &str = r#"
use std::sync::Mutex;
fn main() {
    let m = Mutex::new(5);
    println!("{:?}", m);
    println!("{:#?}", m);
    let g = m.lock().unwrap();
    println!("{:?}", m);
    drop(g);
    println!("{:?}", m);
}
"#;

#[test]
fn run_mode_mutex_debug_format() {
    let dir = temp_dir("mutex-debug");
    let src = fixture(&dir, "mutex_debug.rs", MUTEX_DEBUG_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Mutex Debug program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Mutex { data: 5, poisoned: false, .. }\n\
         Mutex {\n    data: 5,\n    poisoned: false,\n    ..\n}\n\
         Mutex { data: \"<locked>\", poisoned: false, .. }\n\
         Mutex { data: 5, poisoned: false, .. }\n",
        "std's handwritten Mutex Debug form: the guarded value, \
         the quoted <locked> placeholder while a guard is held, \
         the poison flag and the non-exhaustive tail, in both \
         the plain and pretty layouts"
    );
}

const RWLOCK_DEBUG_SRC: &str = r#"
use std::sync::RwLock;
fn main() {
    let r = RwLock::new(String::from("hi"));
    println!("{:?}", r);
    let w = r.write().unwrap();
    println!("{:?}", r);
    drop(w);
    let g = r.read().unwrap();
    println!("{:?}", r);
    drop(g);
    println!("{:#?}", r);
}
"#;

#[test]
fn run_mode_rwlock_debug_format() {
    let dir = temp_dir("rwlock-debug");
    let src = fixture(&dir, "rwlock_debug.rs", RWLOCK_DEBUG_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "RwLock Debug program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "RwLock { data: \"hi\", poisoned: false, .. }\n\
         RwLock { data: <locked>, poisoned: false, .. }\n\
         RwLock { data: \"hi\", poisoned: false, .. }\n\
         RwLock {\n    data: \"hi\",\n    poisoned: false,\n    ..\n}\n",
        "std's handwritten RwLock Debug form: a held read guard \
         does not block the read the Debug impl performs, a held \
         write guard shows the bare <locked> placeholder"
    );
}

// ----- Contended sys locks (futex-backed Mutex / RwLock) -----

const MUTEX_CONTENDED_SRC: &str = r#"use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let m = Arc::new(Mutex::new(0u32));
    let m1 = Arc::clone(&m);
    let h1 = thread::spawn(move || {
        let mut g = m1.lock().unwrap();
        *g += 1;
        // Hold the lock across a long computation: the
        // scheduler quantum expires many times before the
        // unlock, so the second thread's lock below is
        // genuinely contended and must wait for the holder.
        let mut s = 0u64;
        for i in 0..200_000u64 {
            s = s.wrapping_add(i);
        }
        std::hint::black_box(s);
        *g += 1;
    });
    let m2 = Arc::clone(&m);
    let h2 = thread::spawn(move || {
        let mut g = m2.lock().unwrap();
        *g += 10;
    });
    h1.join().unwrap();
    h2.join().unwrap();
    let v = *m.lock().unwrap();
    assert_eq!(v, 12);
    println!("mutex contention ok: {v}");
}
"#;

#[test]
fn run_mode_mutex_contended_lock_waits_for_holder() {
    let dir = temp_dir("mutex-contended");
    let src = fixture(&dir, "mutex_contended.rs", MUTEX_CONTENDED_SRC);
    let out = run_driver(&src);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "mutex contention ok: 12\n"
    );
}

const RWLOCK_CONTENDED_SRC: &str = r#"use std::sync::{Arc, RwLock};
use std::thread;

fn spin() {
    let mut s = 0u64;
    for i in 0..200_000u64 {
        s = s.wrapping_add(i);
    }
    std::hint::black_box(s);
}

fn main() {
    let l = Arc::new(RwLock::new(0u32));
    let l1 = Arc::clone(&l);
    let h1 = thread::spawn(move || {
        let mut g = l1.write().unwrap();
        *g += 1;
        spin();
        *g += 1;
    });
    let l3 = Arc::clone(&l);
    let h3 = thread::spawn(move || {
        let g = l3.read().unwrap();
        std::hint::black_box(*g);
    });
    let l2 = Arc::clone(&l);
    let h2 = thread::spawn(move || {
        let mut g = l2.write().unwrap();
        *g += 10;
    });
    h1.join().unwrap();
    h3.join().unwrap();
    h2.join().unwrap();
    let v = *l.read().unwrap();
    assert_eq!(v, 12);
    println!("rwlock contention ok: {v}");
}
"#;

#[test]
fn run_mode_rwlock_contended_locks_wait_for_holder() {
    let dir = temp_dir("rwlock-contended");
    let src = fixture(&dir, "rwlock_contended.rs", RWLOCK_CONTENDED_SRC);
    let out = run_driver(&src);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "rwlock contention ok: 12\n"
    );
}

const MUTEX_SELF_DEADLOCK_SRC: &str = r#"use std::sync::Mutex;
fn main() {
    let m = Mutex::new(1u32);
    let _first = m.lock().unwrap();
    // A second lock while the first guard is alive deadlocks
    // natively; the model must report the unservable wait
    // rather than fabricate an acquisition.
    let _second = m.lock().unwrap();
    println!("unreachable");
}
"#;

#[test]
fn run_mode_mutex_self_deadlock_is_reported_not_served() {
    let dir = temp_dir("mutex-self-deadlock");
    let src = fixture(&dir, "mutex_self_deadlock.rs", MUTEX_SELF_DEADLOCK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("would block forever"),
        "stderr={stderr}"
    );
}

const RWLOCK_DOWNGRADE_SRC: &str = r#"use std::sync::{Arc, RwLock};
use std::thread;

fn spin() {
    let mut s = 0u64;
    for i in 0..200_000u64 {
        s = s.wrapping_add(i);
    }
    std::hint::black_box(s);
}

fn main() {
    let l = Arc::new(RwLock::new(5u32));
    let l1 = Arc::clone(&l);
    let h1 = thread::spawn(move || {
        let mut g = l1.write().unwrap();
        *g += 1;
        spin();
        let g = std::sync::RwLockWriteGuard::downgrade(g);
        assert_eq!(*g, 6);
        spin();
    });
    let l3 = Arc::clone(&l);
    let h3 = thread::spawn(move || {
        let g = l3.read().unwrap();
        std::hint::black_box(*g);
    });
    h1.join().unwrap();
    h3.join().unwrap();
    println!("downgrade ok");
}
"#;

#[test]
fn run_mode_rwlock_downgrade_releases_waiting_reader() {
    let dir = temp_dir("rwlock-downgrade");
    let src = fixture(&dir, "rwlock_downgrade.rs", RWLOCK_DOWNGRADE_SRC);
    let out = run_driver(&src);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "downgrade ok\n");
}

const SPIN_LOOP_SRC: &str = r#"
fn main() {
    for _ in 0..5 {
        std::hint::spin_loop();
    }
    println!("done");
}
"#;

#[test]
fn run_mode_spin_loop_completes() {
    run_ok("spin-loop", SPIN_LOOP_SRC, "done\n");
}

const RC_TRY_UNWRAP_VEC_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let rc = Rc::new(vec![1, 2, 3]);
    match Rc::try_unwrap(rc) {
        Ok(v) => println!("ok {:?}", v),
        Err(_) => println!("err"),
    }
}
"#;

#[test]
fn run_mode_rc_try_unwrap_vec() {
    // `try_unwrap`'s converted body reads the pointee through
    // the handle stored inside a `ManuallyDrop`: the place
    // walk must take the handle's field steps as field
    // accesses, not as a vector buffer chain (the handle's
    // fields are all aggregates, which the suspected-vector
    // shape judgment once mistook for an element list).
    run_ok("rc-try-unwrap-vec", RC_TRY_UNWRAP_VEC_SRC, "ok [1, 2, 3]\n");
}

const RC_TRY_UNWRAP_STRING_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let rc = Rc::new(String::from("hello"));
    match Rc::try_unwrap(rc) {
        Ok(v) => println!("ok {}", v),
        Err(_) => println!("err"),
    }
}
"#;

#[test]
fn run_mode_rc_try_unwrap_string() {
    run_ok(
        "rc-try-unwrap-string",
        RC_TRY_UNWRAP_STRING_SRC,
        "ok hello\n",
    );
}

const ARC_TRY_UNWRAP_VEC_SRC: &str = r#"
use std::sync::Arc;
fn main() {
    let a = Arc::new(vec![4, 5, 6]);
    match Arc::try_unwrap(a) {
        Ok(v) => println!("ok {:?}", v),
        Err(_) => println!("err"),
    }
}
"#;

#[test]
fn run_mode_arc_try_unwrap_vec() {
    // The `Arc` handle has the same field shape as the `Rc`
    // handle (a wrapper around the allocation reference
    // beside two empty marker aggregates); the same place
    // walk serves it.
    run_ok("arc-try-unwrap-vec", ARC_TRY_UNWRAP_VEC_SRC, "ok [4, 5, 6]\n");
}

const RC_TRY_UNWRAP_ERR_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let rc = Rc::new(vec![9]);
    let keep = rc.clone();
    match Rc::try_unwrap(rc) {
        Ok(_) => println!("unexpected ok"),
        Err(rc) => {
            println!("err {:?}", *rc);
            drop(keep);
            match Rc::try_unwrap(rc) {
                Ok(v) => println!("ok {:?}", v),
                Err(_) => println!("unexpected err"),
            }
        }
    }
}
"#;

#[test]
fn run_mode_rc_try_unwrap_shared_returns_err() {
    run_ok(
        "rc-try-unwrap-err",
        RC_TRY_UNWRAP_ERR_SRC,
        "err [9]\nok [9]\n",
    );
}

const MD_VEC_INTO_ITER_OPTION_SRC: &str = r#"
fn main() {
    let v = vec![Some(1), None, Some(3), None];
    let sum: i32 = v.into_iter().flatten().sum();
    println!("sum {}", sum);
}
"#;

#[test]
fn run_mode_manually_drop_vec_into_iter_option_elements() {
    // The other side of the suspected-vector judgment: a
    // `Vec` of enum elements whose variants differ in shape
    // (`Some` beside `None`), reached through the
    // `ManuallyDrop` inside `Vec::into_iter`, is an element
    // list and keeps the buffer-chain treatment.
    run_ok(
        "md-vec-into-iter-option",
        MD_VEC_INTO_ITER_OPTION_SRC,
        "sum 4\n",
    );
}
