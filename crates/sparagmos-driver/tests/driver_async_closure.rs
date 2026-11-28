//! Async-closure run tests (Lane MJ, Issue #248): an async
//! closure's value is its capture environment, and the `AsyncFn`
//! family call creates the closure's coroutine with the state-0
//! fields in the compiler shim's order - the call arguments first,
//! then the captures, borrowed from the environment for
//! `async_call` / `async_call_mut` (a capture the closure itself
//! took by reference is reborrowed) and moved out of it for
//! `async_call_once`. These fixtures run end to end through
//! `sparagmos-driver run` and pin the exact stdout.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits.
/// Tests spawn the driver per case and used to leave every
/// fixture directory behind (#537); the registry keeps the
/// per-test `temp_dir` shape and sweeps at exit instead.
static SCRATCH_DIRS: std::sync::Mutex<Vec<std::path::PathBuf>> =
    std::sync::Mutex::new(Vec::new());

fn register_scratch_dir(dir: &std::path::Path) {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| unsafe {
        let _ = libc::atexit(remove_scratch_dirs);
    });
    if let Ok(mut dirs) = SCRATCH_DIRS.lock() {
        if !dirs.iter().any(|d| d == dir) {
            dirs.push(dir.to_path_buf());
        }
    }
}

extern "C" fn remove_scratch_dirs() {
    if let Ok(dirs) = SCRATCH_DIRS.try_lock() {
        for dir in dirs.iter() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

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
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-async-closure-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
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
        "async closure program must run to completion; stderr={stderr}"
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

#[test]
fn async_closure_call_and_poll() {
    let src = format!(
        r#"{POLL_PRELUDE}
fn main() {{
    let x = 41;
    let f = async move || {{
        println!("inside: {{}}", x);
        x + 1
    }};
    match poll_once(f()) {{
        Poll::Ready(v) => println!("ready: {{}}", v),
        Poll::Pending => println!("pending"),
    }}
}}
"#
    );
    assert_run("call", &src, "inside: 41\nready: 42\n");
}

#[test]
fn async_fn_item_through_async_fn_bounds() {
    // A function item satisfies an `AsyncFnOnce` / `AsyncFnMut` /
    // `AsyncFn` bound through the blanket impls, and its
    // `async_call_once` receiver is the function pointer value,
    // not an async closure: the shim's meaning is the function
    // call itself, whose return value is the future. (Measured on
    // tests/ui/async-await/async-closures/async-fn-once-for-async-fn.rs,
    // which stopped with "type mismatch in
    // AsyncFnOnce::async_call_once receiver: expected async
    // closure value, found function pointer".)
    let src = format!(
        r#"{POLL_PRELUDE}
async fn foo() {{}}

async fn needs_once(x: impl AsyncFnOnce()) {{ x().await; }}
async fn needs_mut(mut x: impl AsyncFnMut()) {{ x().await; }}
async fn needs_fn(x: impl AsyncFn()) {{ x().await; }}

fn main() {{
    match poll_once(async {{
        needs_once(foo).await;
        needs_mut(foo).await;
        needs_fn(foo).await;
        needs_once(async || {{}}).await;
        needs_once(|| async {{}}).await;
    }}) {{
        Poll::Ready(()) => println!("fn item ok"),
        Poll::Pending => println!("pending"),
    }}
}}
"#
    );
    assert_run("fn-item", &src, "fn item ok\n");
}

#[test]
fn async_closure_params_and_captures_field_order() {
    // The coroutine's state-0 fields are the call arguments first,
    // then the captures: two of each, of distinct types, so a
    // wrong order changes the output.
    let src = format!(
        r#"{POLL_PRELUDE}
fn main() {{
    let a: String = "capA".to_string();
    let b: u64 = 7;
    let f = async move |x: i32, y: bool| {{
        println!("inside: {{}} {{}} {{}} {{}}", x, y, a, b);
        x + b as i32
    }};
    match poll_once(f(5, true)) {{
        Poll::Ready(v) => println!("ready: {{}}", v),
        Poll::Pending => println!("pending"),
    }}
}}
"#
    );
    assert_run("order", &src, "inside: 5 true capA 7\nready: 12\n");
}

#[test]
fn async_closure_mut_repeated_calls_and_suspension() {
    // `async_call_mut` through the call sugar: the same closure is
    // called repeatedly, a by-`&mut` capture accumulates across
    // calls, and each future suspends once on the inner future.
    let src = format!(
        r#"{POLL_PRELUDE}
struct OncePending {{ done: bool }}
impl Future for OncePending {{
    type Output = i32;
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<i32> {{
        if self.done {{
            Poll::Ready(77)
        }} else {{
            self.done = true;
            Poll::Pending
        }}
    }}
}}

fn main() {{
    let mut total = 0i32;
    let mut f = async move |x: i32| {{
        let v = OncePending {{ done: false }}.await;
        total += x + v;
        total
    }};
    for x in [1, 2] {{
        let mut fut = Box::pin(f(x));
        let mut cx = Context::from_waker(Waker::noop());
        let first = Pin::as_mut(&mut fut).poll(&mut cx);
        println!("first pending: {{}}", first.is_pending());
        match Pin::as_mut(&mut fut).poll(&mut cx) {{
            Poll::Ready(v) => println!("ready: {{}}", v),
            Poll::Pending => println!("pending again"),
        }}
    }}
}}
"#
    );
    assert_run(
        "mut",
        &src,
        "first pending: true\nready: 78\nfirst pending: true\nready: 157\n",
    );
}

#[test]
fn async_closure_once_consumes_environment() {
    let src = format!(
        r#"{POLL_PRELUDE}
fn main() {{
    let s = String::from("owned");
    let g = async move || {{
        println!("g sees: {{}}", s);
        s.len()
    }};
    match poll_once(g()) {{
        Poll::Ready(v) => println!("g ready: {{}}", v),
        Poll::Pending => println!("g pending"),
    }}
}}
"#
    );
    assert_run("once", &src, "g sees: owned\ng ready: 5\n");
}

#[test]
fn async_closure_uncalled_drop_drops_captures() {
    // The closure value's drop is its captures' drop, in the
    // native order: an uncalled closure at scope end, and the
    // environment of a closure whose borrowed future was dropped.
    let src = r#"
struct Guard(&'static str);
impl Drop for Guard {
    fn drop(&mut self) {
        println!("dropped {}", self.0);
    }
}

fn main() {
    {
        let g = Guard("capture");
        let _f = async move || {
            let _ = &g;
            1
        };
        println!("scope end");
    }
    println!("after scope");
    {
        let g = Guard("once-capture");
        let f = async move || {
            let _ = &g;
            5
        };
        let fut = f();
        drop(fut);
        println!("future dropped");
    }
    println!("done");
}
"#;
    assert_run(
        "drop",
        src,
        "scope end\ndropped capture\nafter scope\nfuture dropped\ndropped once-capture\ndone\n",
    );
}

#[test]
fn async_closure_zero_capture_and_nested() {
    // A capture-less async closure is a zero-sized value whose
    // local is never assigned; so is an async closure whose only
    // capture is another zero-sized async closure. Both must
    // still create futures with the layout's field count.
    let src = format!(
        r#"{POLL_PRELUDE}
fn main() {{
    let g = async |s: &str| {{
        println!("g: {{}}", s);
        s.len()
    }};
    match poll_once(g("hello")) {{
        Poll::Ready(v) => println!("ready: {{}}", v),
        Poll::Pending => println!("pending"),
    }}
    let inner = async move |x: i32| x + 1;
    let outer = async move |x: i32| inner(x).await * 10;
    match poll_once(outer(4)) {{
        Poll::Ready(v) => println!("nested ready: {{}}", v),
        Poll::Pending => println!("nested pending"),
    }}
}}
"#
    );
    assert_run("zst", &src, "g: hello\nready: 5\nnested ready: 50\n");
}

#[test]
fn async_closure_capture_kinds() {
    // (a) a value captured by value, (b) a value captured by
    // shared reference, (c) a reference captured by value: the
    // coroutine's capture fields differ per kind even though (b)
    // and (c) have the same environment field type.
    let src = format!(
        r#"{POLL_PRELUDE}
fn main() {{
    let a = 10;
    let fa = async move || a + 1;
    let b = 20;
    let fb = async || b + 2;
    let c = 30;
    let cr = &c;
    let fc = async move || *cr + 3;
    match poll_once(fa()) {{
        Poll::Ready(v) => println!("a ready: {{}}", v),
        Poll::Pending => println!("a pending"),
    }}
    match poll_once(fb()) {{
        Poll::Ready(v) => println!("b ready: {{}}", v),
        Poll::Pending => println!("b pending"),
    }}
    match poll_once(fc()) {{
        Poll::Ready(v) => println!("c ready: {{}}", v),
        Poll::Pending => println!("c pending"),
    }}
}}
"#
    );
    assert_run("kinds", &src, "a ready: 11\nb ready: 22\nc ready: 33\n");
}

#[test]
fn async_closure_through_generic_bound() {
    let src = format!(
        r#"{POLL_PRELUDE}
async fn apply<F>(f: F, x: i32) -> i32
where
    F: AsyncFn(i32) -> i32,
{{
    f(x).await
}}

fn main() {{
    let base = 3;
    let f = async move |x: i32| x * base;
    match poll_once(apply(f, 14)) {{
        Poll::Ready(v) => println!("apply ready: {{}}", v),
        Poll::Pending => println!("apply pending"),
    }}
}}
"#
    );
    assert_run("generic", &src, "apply ready: 42\n");
}

#[test]
fn async_closure_inside_adt_wrappers() {
    // An async closure whose type is nested inside another
    // type (an `Option`, a struct field, a `Box`) converts and
    // runs like a bare one: the field-layout recording of the
    // wrapper ADT must not ask the closure type's kind, which
    // Stable MIR cannot answer — asking it used to reject the
    // whole body that merely mentioned such a wrapper.
    let src = format!(
        r#"{POLL_PRELUDE}
struct Holder<F> {{ f: F }}

fn main() {{
    let v = 3;
    let h = Holder {{ f: async move |x: i32| x * v }};
    match poll_once((h.f)(7)) {{
        Poll::Ready(n) => println!("field: {{}}", n),
        Poll::Pending => println!("field: pending"),
    }}
    let o = Some(async move |x: i32| x + v);
    match o {{
        Some(f) => match poll_once(f(1)) {{
            Poll::Ready(n) => println!("option: {{}}", n),
            Poll::Pending => println!("option: pending"),
        }},
        None => println!("option: none"),
    }}
    let b = Box::new(async move |x: i32| x + v);
    drop(b);
    println!("boxed: dropped");
}}
"#
    );
    assert_run("wrappers", &src, "field: 21\noption: 4\nboxed: dropped\n");
}

#[test]
fn async_closure_nested_option_in_struct() {
    // The wrapper nesting composes: an async closure inside
    // an `Option` inside a struct field is taken out and
    // called, exactly as in the bare-local shape.
    let src = format!(
        r#"{POLL_PRELUDE}
struct Holder<F> {{ f: Option<F> }}

fn main() {{
    let v = 9;
    let mut h = Holder {{ f: Some(async move |x: i32| x + v) }};
    let f = h.f.take().unwrap();
    match poll_once(f(1)) {{
        Poll::Ready(n) => println!("nested: {{}}", n),
        Poll::Pending => println!("nested: pending"),
    }}
}}
"#
    );
    assert_run("nested", &src, "nested: 10\n");
}

#[test]
fn async_closure_immediately_called_through_promoted_ref() {
    // An async closure that is called in the same expression
    // that creates it is not built by an aggregate in the
    // caller: MIR passes it as a promoted by-reference
    // constant (`const _promoted: &{async closure}`), whose
    // pointee type's kind cannot be asked through Stable MIR.
    // The constant must still decode to the closure value so
    // the call converts and runs (Issue #396).
    let src = format!(
        r#"{POLL_PRELUDE}
fn make(x: u8) -> impl Future<Output = u8> {{
    (async move |x: u8| -> u8 {{ x + 1 }})(x)
}}

fn main() {{
    match poll_once(make(41)) {{
        Poll::Ready(v) => println!("ready: {{}}", v),
        Poll::Pending => println!("pending"),
    }}
}}
"#
    );
    assert_run("promoted-ref", &src, "ready: 42\n");
}
