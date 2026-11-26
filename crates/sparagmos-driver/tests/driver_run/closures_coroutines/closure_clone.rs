// Closure cloning with zero-sized captures and their side effects.

const CLOSURE_CLONE_ZST_CAPTURE_SIDE_EFFECTS_SRC: &str = r#"use std::cell::Cell;
use std::rc::Rc;

struct Zap;

impl Clone for Zap {
    fn clone(&self) -> Self {
        println!("clone Zap");
        Zap
    }
}

impl Drop for Zap {
    fn drop(&mut self) {
        println!("drop Zap");
    }
}

fn main() {
    // A zero-sized capture with an observable Clone side effect
    // and an observable Drop: cloning the closure runs the
    // capture's Clone, and each closure's own drop runs the
    // capture's Drop exactly once (Issue #499).
    let rc = Rc::new(Cell::new(7));
    let rc_in = rc.clone();
    let zap = Zap;
    let f = move || {
        let _ = rc_in.get();
        let _ = &zap;
    };
    let g = f.clone();
    println!("count: {}", Rc::strong_count(&rc));
    drop(g);
    println!("after drop g: {}", Rc::strong_count(&rc));
    drop(f);
    println!("final: {}", Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_closure_clone_zst_capture_side_effects() {
    let dir = temp_dir("closure-clone-zst-capture-side-effects");
    let src = fixture(
        &dir,
        "closure_clone_zst_capture_side_effects.rs",
        CLOSURE_CLONE_ZST_CAPTURE_SIDE_EFFECTS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a zero-sized capture's Clone and Drop must run for a cloned closure; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "clone Zap\ncount: 3\ndrop Zap\nafter drop g: 2\ndrop Zap\nfinal: 1\n"
    );
}

const CLOSURE_CLONE_ZST_CAPTURE_FIRST_SRC: &str = r#"use std::cell::Cell;
use std::rc::Rc;

struct Zap;

impl Clone for Zap {
    fn clone(&self) -> Self {
        println!("clone Zap");
        Zap
    }
}

impl Drop for Zap {
    fn drop(&mut self) {
        println!("drop Zap");
    }
}

struct Boom(i32);

impl Clone for Boom {
    fn clone(&self) -> Self {
        panic!("boom clone");
    }
}

fn main() {
    // The zero-sized capture comes first, then a counted
    // handle, then a panicking capture: the prefix cloned
    // before the panic — the zero-sized capture and the handle
    // — drops during unwinding, in an environment whose field
    // plans must stay aligned across the recovered field
    // (Issue #499).
    let rc = Rc::new(Cell::new(7));
    let rc_in = rc.clone();
    let zap = Zap;
    let boom = Boom(3);
    let f = move || {
        let _ = &zap;
        let _ = rc_in.get();
        let _ = &boom;
    };
    println!("before: {}", Rc::strong_count(&rc));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = f.clone();
        println!("no panic");
    }));
    println!("caught: {}", r.is_err());
    println!("after: {}", Rc::strong_count(&rc));
    drop(f);
    println!("final: {}", Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_closure_clone_zst_capture_first() {
    let dir = temp_dir("closure-clone-zst-capture-first");
    let src = fixture(
        &dir,
        "closure_clone_zst_capture_first.rs",
        CLOSURE_CLONE_ZST_CAPTURE_FIRST_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a leading zero-sized capture must not misalign the cloned prefix's drop glue; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "before: 2\nclone Zap\ndrop Zap\ncaught: true\nafter: 2\ndrop Zap\nfinal: 1\n"
    );
}
