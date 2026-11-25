//! Exec conversion-failure regression tests (Issue #513): the
//! shapes the rustc suite's exec records failed on in bulk -
//! async closures handed to the generic `Fn` traits, EII
//! statics read through their declaration, and the `try_fold`
//! consumer family whose specialised bodies walk an iterator's
//! raw fields. Each fixture runs end to end through
//! `sparagmos-driver run` and pins the exact stdout; every one
//! of them fails on the unmodified driver (a conversion error,
//! an uninitialised read, or a type mismatch).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(dir: &Path, name: &str, src: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    path
}

fn run_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn run_driver_opt(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--rustc-arg")
        .arg("-O")
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn assert_run_opt(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src = fixture(&dir, "prog.rs", src);
    let out = run_driver_opt(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "stdout must match the native run"
    );
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-exec-conversion-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn assert_run(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src = fixture(&dir, "prog.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "stdout must match the native run"
    );
}

const POLL_PRELUDE: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

fn poll_once<F: Future>(fut: F) -> Poll<F::Output> {
    let mut fut = Box::pin(fut);
    let mut cx = Context::from_waker(Waker::noop());
    Pin::as_mut(&mut fut).poll(&mut cx)
}
"#;

// A capture-less async closure is a ZST constant: passed to a
// generic plain-`FnOnce` parameter it arrives as a constant
// operand, and the call must still create its coroutine. The
// unmodified driver typed the constant as a plain zero-sized
// value, so the call failed with "Fn trait call: expected
// callable value, found unit".
#[test]
fn async_closure_constant_through_generic_fn_once() {
    let src = r#"fn once<F: FnOnce() -> T, T>(f: F) -> T {
    f()
}

fn main() {
    let closure = async || {};
    let _fut = once(closure);
    println!("ok");
}
"#;
    assert_run("async-const-once", src, "ok\n");
}

// The same hand-off for an aggregate-created async closure
// with real captures, through a generic plain-`FnOnce` call:
// the call must be served by the async-call machinery (the
// closure's own body cannot be converted - it mentions the
// coroutine-closure type), not by the ordinary closure
// dispatch, which does not know the closure's body.
#[test]
fn async_closure_aggregate_through_generic_fn_once() {
    let src = r#"fn once<F: FnOnce() -> Fut, Fut>(f: F) -> Fut {
    f()
}

fn main() {
    let x = 5u32;
    let g = async move || x;
    let _fut = once(g);
    println!("ok");
}
"#;
    assert_run("async-agg-once", src, "ok\n");
}

// Direct call and generic-`FnOnce` call on the same ZST
// async closure value: the direct call goes through the
// `AsyncFn` machinery, the generic one through the plain `Fn`
// traits with a constant-typed receiver.
#[test]
fn async_closure_direct_and_method_calls() {
    let src = r#"fn once<F: FnOnce() -> T, T>(f: F) -> T {
    f()
}

fn main() {
    let closure = async || {};
    // direct call first
    let _fut = closure();
    // then via the generic FnOnce shim
    let _fut2 = once(closure);
    println!("ok");
}
"#;
    assert_run("async-direct", src, "ok\n");
}

// An EII static's declaration is a foreign item, but its value
// lives in the implementation static the compiler registers for
// it; reading the declaration must read the implementation. The
// unmodified driver decoded the declaration against an
// uninitialised foreign seed.
#[test]
fn eii_static_reads_its_implementation() {
    let src = r#"#![feature(extern_item_impls)]

#[eii(hello)]
static HELLO: u64;

#[hello]
static HELLO_IMPL: u64 = 5;

fn main() {
    println!("{HELLO_IMPL}");
    println!("{HELLO}");
}
"#;
    assert_run("eii-simple", src, "5\n5\n");
}

// A defaulted EII declaration: the value lives in the default
// static the compiler synthesises for the declaration.
#[test]
fn eii_static_default() {
    let src = r#"#![feature(extern_item_impls)]

#[eii(eii1)]
pub static DECL1: u64 = 5;

fn main() {
    println!("{DECL1}");
}
"#;
    assert_run("eii-default", src, "5\n");
}

// A defaulted EII declaration with an explicit implementation:
// the registry then holds two implementations (the default,
// registered under an item carrying the declaration's own
// name, and the explicit one), and the explicit implementation
// overrides the default. The unmodified driver required
// exactly one registered implementation and read the
// declaration as uninitialised storage.
#[test]
fn eii_static_explicit_impl_overrides_default() {
    let src = r#"#![feature(extern_item_impls)]

#[eii(eii1)]
pub static DECL1: u64 = 5;

#[eii1]
pub static EII1_IMPL: u64 = 10;

fn main() {
    println!("{DECL1}");
}
"#;
    assert_run("eii-default-explicit", src, "10\n");
}

// At -O a small vec is built as a `Box<MaybeUninit<[i8; 2]>>`:
// the guest writes the array through the union's declared
// members (`value: ManuallyDrop<T>`) into fresh allocation
// storage, and the later slice read dereferences the pointer
// the `assume_init` transmute produced. The write must
// partially initialise the fresh union storage instead of
// erroring on the never-written member, and the read must
// enter the raw two-field storage form the field-by-field
// write produced (the payload sits in the `value` member,
// not in the normalised one-field form), judging the payload
// against the slice pointee. The unmodified driver failed
// the write with an uninitialised read and, past it, the
// read with a transmute type mismatch.
#[test]
fn optimized_vec_element_read_through_slice() {
    let src = r#"fn qux(z: &[i8]) -> i8 { z[0] + z[1] }
fn main() { let z = vec![42i8, 43]; println!("{}", qux(&z)); }
"#;
    assert_run_opt("vec-opt-slice-read", src, "85\n");
}

// `Iterator::all` funnels into the specialised `try_fold` of
// `vec::IntoIter`, whose body walks the iterator's raw pointer
// fields; the modelled state has no such fields, so the run
// died on a read of the raw `end` field. `count` reaches the
// same consumer through `NeverShortCircuit`. Both must run the
// modelled loop.
#[test]
fn try_fold_consumers_all_and_count() {
    let src = r#"fn main() {
    let a = vec![1u32].into_iter().all(|x| x == 1);
    println!("all: {a}");
    let mut it = vec![1u32, 2].into_iter();
    println!("count: {}", it.count());
    let mut it2 = vec![1u32, 2].into_iter();
    println!("count2: {}", it2.by_ref().count());
    println!("done");
}
"#;
    assert_run(
        "try-fold-consumers",
        src,
        "all: true\ncount: 2\ncount2: 2\ndone\n",
    );
}

// `try_fold` itself, with the `Result` and `Option` branch
// shapes: an early `Err` returns unchanged, and exhausting the
// iterator wraps the accumulator.
#[test]
fn try_fold_result_and_option_shapes() {
    let src = r#"fn main() {
    let r: Result<i32, &'static str> = vec![1i32, 2, 3].into_iter().try_fold(0, |acc, x| {
        if x == 2 { Err("stop") } else { Ok(acc + x) }
    });
    println!("{r:?}");
    let s: Option<i32> = vec![1i32, 2].into_iter().try_fold(0, |acc, x| Some(acc + x));
    println!("{s:?}");
    println!("done");
}
"#;
    assert_run(
        "try-fold-shapes",
        src,
        "Err(\"stop\")\nSome(3)\ndone\n",
    );
}

// A generic `FnMut` call on a by-value closure parameter: the
// receiver is a `Move` operand, and dispatch must leave the
// parameter's value in place for the call itself.
#[test]
fn generic_fn_mut_param_stays_intact() {
    let src = r#"fn apply<F: FnMut(u32) -> u32>(mut f: F) -> u32 {
    f(3)
}

fn main() {
    let g = |x: u32| x + 1;
    let h = move |x: u32| g(x) * 2;
    println!("direct: {}", apply(h));
    println!("done");
}
"#;
    assert_run("fnmut-param", src, "direct: 8\ndone\n");
}

// Trait-object iterators: stepping a `Box<dyn Iterator>` (or a
// `&mut dyn Iterator`) steps the concrete iterator inside the
// dyn reference; the trait object's own provided methods reach
// `next` with the `dyn` tag while the receiver place resolves
// to the concrete state; and a boxed iterator is its own
// `IntoIter`. The unmodified driver rejected the dyn state in
// the iterator state machinery.
#[test]
fn dyn_iterator_steps_concrete_state() {
    let src = r#"fn main() {
    let mut boxed: Box<dyn Iterator<Item = i32>> = Box::new(vec![1, 2, 3].into_iter());
    println!("{:?}", boxed.next());
    println!("{:?}", boxed.max());
    let mut it: &mut dyn Iterator<Item = i32> = &mut vec![4, 5].into_iter();
    println!("{:?}", it.next());
    let mut de: Box<dyn DoubleEndedIterator<Item = i32>> =
        Box::new(vec![7, 8, 9].into_iter());
    println!("{:?}", de.next_back());
    println!("done");
}
"#;
    assert_run(
        "dyn-iter",
        src,
        "Some(1)\nSome(3)\nSome(4)\nSome(9)\ndone\n",
    );
}
