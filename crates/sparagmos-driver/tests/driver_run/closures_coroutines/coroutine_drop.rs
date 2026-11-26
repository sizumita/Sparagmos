// Drop glue for coroutines: suspended, completed and unresumed states drop their live values once.

// ----- coroutine drop glue and runtime support (Lane FF) ---------------------

const COROUTINE_SUSPENDED_DROP_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::Coroutine;
use std::pin::Pin;
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let mut g = #[coroutine] || {
        let d = D(1);
        yield;
        println!("resumed {}", d.0);
    };
    let _s = Pin::new(&mut g).resume(());
    println!("suspended");
    drop(g);
    println!("after drop");
}
"#;

#[test]
fn run_mode_suspended_coroutine_drop_runs_live_local_dtor() {
    // Dropping a coroutine suspended at a yield runs the
    // destructors of the locals live in that state, at the drop
    // point - the state the value is in selects the glue.
    let dir = temp_dir("coroutine-suspended-drop");
    let src = fixture(&dir, "coroutine_suspended_drop.rs", COROUTINE_SUSPENDED_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "suspended coroutine drop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "suspended\ndtor 1\nafter drop\n",
        "the live local's destructor runs when the suspended coroutine is dropped"
    );
}

const COROUTINE_COMPLETED_DROP_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::Coroutine;
use std::pin::Pin;
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let mut g = #[coroutine] || {
        let d = D(1);
        yield;
        println!("resumed {}", d.0);
    };
    let _s = Pin::new(&mut g).resume(());
    let _s2 = Pin::new(&mut g).resume(());
    println!("completed");
    drop(g);
    println!("after drop");
}
"#;

#[test]
fn run_mode_completed_coroutine_drop_runs_dtor_once() {
    // A completed coroutine's locals were already dropped by the
    // coroutine's own completion path; dropping the exhausted
    // value afterwards runs nothing a second time.
    let dir = temp_dir("coroutine-completed-drop");
    let src = fixture(&dir, "coroutine_completed_drop.rs", COROUTINE_COMPLETED_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "completed coroutine drop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "resumed 1\ndtor 1\ncompleted\nafter drop\n",
        "the destructor runs exactly once, inside the completion"
    );
}

const COROUTINE_UNRESUMED_DROP_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let d = D(9);
    let g = #[coroutine] move || { let dd = d; yield; println!("inner {}", dd.0); };
    println!("created");
    drop(g);
    println!("after drop");
}
"#;

#[test]
fn run_mode_unresumed_coroutine_drop_runs_capture_dtor() {
    // Dropping a coroutine that was never resumed drops the
    // values it captured, at the drop point.
    let dir = temp_dir("coroutine-unresumed-drop");
    let src = fixture(&dir, "coroutine_unresumed_drop.rs", COROUTINE_UNRESUMED_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "unresumed coroutine drop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "created\ndtor 9\nafter drop\n",
        "the captured value's destructor runs when the unresumed coroutine is dropped"
    );
}

const COROUTINE_DROP_ENV_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
#![allow(dropping_copy_types)]

use std::ops::Coroutine;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

static A: AtomicUsize = AtomicUsize::new(0);

struct B;

impl Drop for B {
    fn drop(&mut self) {
        A.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    let b = B;
    let mut foo = #[coroutine] || {
        yield;
        drop(b);
    };
    let n = A.load(Ordering::SeqCst);
    drop(Pin::new(&mut foo).resume(()));
    assert_eq!(A.load(Ordering::SeqCst), n);
    drop(foo);
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
    println!("coroutine drop env ok");
}
"#;

#[test]
fn run_mode_suspended_coroutine_drop_runs_capture_dtor() {
    // Dropping a suspended coroutine drops the captures it
    // still holds: the coroutine's storage is one field list
    // shared by every state, so the suspended state's glue is
    // the unresumed state's fields (plus its own saved locals),
    // not the empty plan an unevidenced variant would get.
    let dir = temp_dir("coroutine-drop-env");
    let src = fixture(&dir, "coroutine_drop_env.rs", COROUTINE_DROP_ENV_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a suspended coroutine's drop must run its captures' destructors; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "coroutine drop env ok\n");
}

// ----- coroutine drop glue: aggregate under construction across yields -----

const COROUTINE_TWO_YIELD_TUPLE_DROP_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::Coroutine;
use std::pin::Pin;
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let mut g = #[coroutine] || {
        (D(70), yield D(71), D(72), yield D(73))
    };
    let _ = Pin::new(&mut g).resume(());
    drop(g);
    println!("after drop");
}
"#;

#[test]
fn run_mode_coroutine_two_yield_tuple_drop_runs_placed_element_dtor() {
    // A tuple under construction across two suspension points:
    // the element placed before the first yield is live in the
    // first suspension state, but the transformed body projects
    // it only under the second state's index, so the first
    // state's drop plan - built from projection evidence alone -
    // never mentioned it and the element leaked when the
    // suspended coroutine was dropped. The plan now accounts
    // for a state's fields from the other states' plans for
    // the same storage slot, so the placed element's destructor
    // runs at the drop point.
    let dir = temp_dir("coroutine-two-yield-tuple-drop");
    let src = fixture(
        &dir,
        "coroutine_two_yield_tuple_drop.rs",
        COROUTINE_TWO_YIELD_TUPLE_DROP_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the two-yield tuple coroutine drop must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dtor 71\ndtor 70\nafter drop\n",
        "the yielded value drops at the resume statement, the placed element at the drop"
    );
}
