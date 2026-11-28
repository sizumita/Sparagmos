use super::{fixture, run_driver, run_driver_extra, run_driver_with_aux_specs, run_ok, temp_dir};


// ----- explicit mem::drop (Issue #61) ------------------------------------------

const MEM_DROP_SRC: &str = r#"
struct D(i32);
impl Drop for D {
    fn drop(&mut self) { println!("drop {}", self.0); }
}
fn main() {
    let a = D(1);
    let b = D(2);
    std::mem::drop(a);
    println!("middle");
}
"#;

const CONST_BOUND_MATCH_SRC: &str = r#"
struct Conn;
impl Drop for Conn {
    fn drop(&mut self) { println!("dropped"); }
}
struct Wrap(Conn);
impl ::std::ops::Deref for Wrap {
    type Target = Conn;
    fn deref(&self) -> &Conn { &self.0 }
}
fn inner() {
    let conn = &*match Some(Wrap(Conn)) {
        Some(val) => val,
        None => return,
    };
    let _ = conn;
    println!("inner end");
}
fn main() {
    inner();
    println!("main end");
}
"#;

#[test]
fn run_mode_const_bound_match_binding_is_initialized() {
    // The scrutinee is a constant aggregate, so optimized MIR
    // const-propagates the by-value match binding away: the
    // binding local has no assignment anywhere in the body and
    // its value survives only as a constant debug-info entry.
    // The driver restores the definition at conversion; without
    // it the local is met uninitialized and its scope-end Drop
    // fails with "unsupported terminator: Drop"
    // (tests/ui/mir/mir_early_return_scope.rs).
    let dir = temp_dir("constbound");
    let src = fixture(&dir, "constbound.rs", CONST_BOUND_MATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "const-bound match binding program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "inner end\ndropped\nmain end\n",
        "the binding's destructor runs at scope end, after the body"
    );
}

#[test]
fn run_mode_mem_drop_runs_destructors() {
    // An explicit std::mem::drop runs the same destructor as a
    // scope-end drop: the explicitly dropped value's destructor
    // runs at the call, and the remaining local drops afterwards,
    // at scope end — the native interleaving, byte for byte. (The
    // container cases are covered at the interpreter level in
    // memdrop_glue_tests; a Vec of droppable elements additionally
    // needs the runtime allocation path.)
    let dir = temp_dir("memdrop");
    let src = fixture(&dir, "memdrop.rs", MEM_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "mem::drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop 1\nmiddle\ndrop 2\n",
        "explicit and scope-end drops interleave exactly as natively"
    );
}

// ----- Box pointee move-out (Lane MQ) ----------------------------------------

const BOX_MOVE_OUT_ZST_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
fn main() {
    let b = Box::new(DropMe);
    let _d = *b;
    println!("done");
}
"#;

#[test]
fn run_mode_box_pointee_move_out_drops_once_zst() {
    // Moving the pointee out of a box leaves the box an empty
    // shell: the moved value's destructor runs once (at its own
    // scope end), and the box's own drop — which drop elaboration
    // emits as the `Drop::drop` method call — only releases the
    // allocation. The pointee is a ZST here, so the move itself
    // leaves no statement in the converted body at all.
    let dir = temp_dir("boxmovezst");
    let src = fixture(&dir, "boxmovezst.rs", BOX_MOVE_OUT_ZST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "box move-out program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "done\ndrop\n",
        "the moved-out pointee drops exactly once"
    );
}

const BOX_MOVE_OUT_SCALAR_SRC: &str = r#"
struct DropMe(u32);
impl Drop for DropMe { fn drop(&mut self) { println!("drop {}", self.0); } }
fn main() {
    let b = Box::new(DropMe(7));
    let _d = *b;
    println!("done");
}
"#;

#[test]
fn run_mode_box_pointee_move_out_drops_once_scalar() {
    // The non-ZST form of the move-out: the pointee load is a
    // shaped load converted from a move, which clears the
    // pointee's slot in the box's allocation, and the box's
    // method-call drop is the shell step only.
    let dir = temp_dir("boxmovescalar");
    let src = fixture(&dir, "boxmovescalar.rs", BOX_MOVE_OUT_SCALAR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "box move-out program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "done\ndrop 7\n",
        "the moved-out pointee drops exactly once, with its value intact"
    );
}

const BOX_MOVE_OUT_SCOPE_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
fn main() {
    let _d;
    {
        let b = Box::new(DropMe);
        _d = *b;
    }
    println!("mid");
}
"#;

#[test]
fn run_mode_box_pointee_move_out_scope_order() {
    // The emptied box drops at the inner scope's end, silently;
    // the moved-out value drops at the outer scope's end.
    let dir = temp_dir("boxmovescope");
    let src = fixture(&dir, "boxmovescope.rs", BOX_MOVE_OUT_SCOPE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "box move-out program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "mid\ndrop\n",
        "the empty box drops silently before the moved value drops"
    );
}

const BOX_PARTIAL_MOVE_OUT_SRC: &str = r#"
struct A(u32);
impl Drop for A { fn drop(&mut self) { println!("drop A {}", self.0); } }
struct B(u32);
impl Drop for B { fn drop(&mut self) { println!("drop B {}", self.0); } }
struct P { a: A, b: B }
fn main() {
    let bx = Box::new(P { a: A(1), b: B(2) });
    let _a = bx.a;
    println!("end");
}
"#;

#[test]
fn run_mode_box_pointee_partial_move_out_drops_rest_once() {
    // A field moved out of the pointee drops at its own scope
    // end; the field that stayed is dropped by the elaborated
    // field drop, and the box's method-call drop is the shell
    // step only — neither field's destructor runs twice.
    let dir = temp_dir("boxpartialmove");
    let src = fixture(&dir, "boxpartialmove.rs", BOX_PARTIAL_MOVE_OUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "partial move-out program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "end\ndrop A 1\ndrop B 2\n",
        "each field drops exactly once"
    );
}

const BOX_NO_MOVE_OUT_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
fn main() {
    let _b = Box::new(DropMe);
    println!("done");
}
"#;

#[test]
fn run_mode_box_drop_without_move_out_still_drops_pointee() {
    // The plain scope-end drop of a box (the `Drop` terminator
    // form, no move-out anywhere) still runs the pointee's
    // destructor exactly once: the shell-only glue applies to
    // the method-call form only.
    let dir = temp_dir("boxnomove");
    let src = fixture(&dir, "boxnomove.rs", BOX_NO_MOVE_OUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "plain box program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "done\ndrop\n",
        "an unmoved box still drops its pointee"
    );
}


// ----- Box<dyn Fn> drop timing (Lane EV) --------------------------------------

const BOX_DYN_FN_WHOLE_CAPTURE_SRC: &str = r#"
struct M(i32, i32);
impl M { fn get(&self) -> i32 { self.0 } }
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let m = M(7, 8);
    let f: Box<dyn Fn(i32) -> i32> = Box::new(move |x| x + m.get());
    println!("call {}", f(1));
    drop(f);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_fn_drop_runs_captured_dtor_at_drop() {
    // The closure owns the captured `M` (the body uses the whole
    // place through a method call, so edition-2021 disjoint capture
    // still captures all of `m`): dropping the box at `drop(f)`
    // runs the capture's destructor there, before the next
    // statement — the native interleaving, byte for byte.
    let dir = temp_dir("box-dyn-fn-drop");
    let src = fixture(&dir, "box_dyn_fn_drop.rs", BOX_DYN_FN_WHOLE_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<dyn Fn> drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "call 8\ndtor 7\nafter drop\n",
        "the captured value's destructor runs at the explicit drop"
    );
}

const BOX_DYN_FN_FIELD_CAPTURE_SRC: &str = r#"
struct M(i32, i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let m = M(7, 8);
    let f: Box<dyn Fn(i32) -> i32> = Box::new(move |x| x + m.0);
    println!("call {}", f(1));
    drop(f);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_fn_field_capture_dtor_stays_with_captured_place() {
    // The same source means different captures in different
    // editions, and the destructor follows the capture: in edition
    // 2021 the closure captures only the place `m.0` (an `i32`
    // copy), so `M` itself stays in `main` and its destructor runs
    // at scope end, after the explicit drop of the box — exactly
    // the native 2021 output. In edition 2015 the closure captures
    // all of `m`, so the destructor runs at `drop(f)`. Both
    // interleavings are pinned so a change in capture or drop
    // semantics cannot silently move the destructor.
    let dir = temp_dir("box-dyn-fn-field-capture");
    let src = fixture(
        &dir,
        "box_dyn_fn_field_capture.rs",
        BOX_DYN_FN_FIELD_CAPTURE_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "field-capture program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "call 8\nafter drop\ndtor 7\n",
        "edition 2021: only m.0 was captured, so M's destructor stays at scope end"
    );
    let out = run_driver_with_aux_specs(&src, Some("2015"), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "field-capture program must terminate normally in edition 2015; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "call 8\ndtor 7\nafter drop\n",
        "edition 2015: the whole M was captured, so its destructor runs at drop(f)"
    );
}

const BOX_DYN_TRAIT_PAYLOAD_SRC: &str = r#"
trait G { fn get(&self) -> i32; }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct S { m: M }
impl G for S { fn get(&self) -> i32 { self.m.0 } }
fn main() {
    let b: Box<dyn G> = Box::new(S { m: M(9) });
    println!("get {}", b.get());
    drop(b);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_trait_drop_runs_payload_dtor_at_drop() {
    // A `Box<dyn Trait>` whose payload is a plain ADT: the
    // payload's field destructor runs at the explicit drop, as
    // with the closure case.
    let dir = temp_dir("box-dyn-trait-drop");
    let src = fixture(&dir, "box_dyn_trait_drop.rs", BOX_DYN_TRAIT_PAYLOAD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<dyn Trait> drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "get 9\ndtor 9\nafter drop\n",
        "the payload's destructor runs at the explicit drop"
    );
}

const BOX_DYN_FNONCE_SRC: &str = r#"
struct M(i32, i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let m = M(7, 8);
    let f: Box<dyn FnOnce(i32) -> i32> = Box::new(move |x| { let mm = m; x + mm.0 });
    println!("made");
    drop(f);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_fnonce_uncalled_drop_runs_captured_dtor() {
    // An uncalled `Box<dyn FnOnce>` still owns its capture:
    // dropping the box runs the captured value's destructor at
    // the drop, even though the closure body (which would have
    // moved the capture out) never ran.
    let dir = temp_dir("box-dyn-fnonce-drop");
    let src = fixture(&dir, "box_dyn_fnonce_drop.rs", BOX_DYN_FNONCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<dyn FnOnce> drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "made\ndtor 7\nafter drop\n",
        "the uncalled FnOnce capture's destructor runs at the explicit drop"
    );
}

const BOX_DYN_FN_INNER_SCOPE_SRC: &str = r#"
struct M(i32, i32);
impl M { fn get(&self) -> i32 { self.0 } }
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let m = M(7, 8);
    let f: Box<dyn Fn(i32) -> i32> = Box::new(move |x| x + m.get());
    println!("call {}", f(1));
    { let _g = f; println!("in inner"); }
    println!("after inner");
}
"#;

#[test]
fn run_mode_box_dyn_fn_moved_into_inner_scope_drops_at_inner_end() {
    // Moving the box into an inner scope moves the capture's
    // destructor with it: it runs at the inner scope's end (the
    // `Drop` terminator for the moved-to local), not at the
    // outer function's end.
    let dir = temp_dir("box-dyn-fn-inner-scope");
    let src = fixture(&dir, "box_dyn_fn_inner_scope.rs", BOX_DYN_FN_INNER_SCOPE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "inner-scope program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "call 8\nin inner\ndtor 7\nafter inner\n",
        "the destructor runs at the inner scope's end"
    );
}

const SIZED_CLOSURE_DROP_SRC: &str = r#"
struct M(i32, i32);
impl M { fn get(&self) -> i32 { self.0 } }
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let m = M(7, 8);
    let c = move |x| x + m.get();
    println!("call {}", c(1));
    drop(c);
    println!("after drop c");
}
"#;

#[test]
fn run_mode_sized_closure_drop_runs_captured_dtor_at_drop() {
    // The sized counterpart of the `Box<dyn Fn>` case: an explicit
    // drop of the closure value itself runs its capture's
    // destructor at the drop.
    let dir = temp_dir("sized-closure-drop");
    let src = fixture(&dir, "sized_closure_drop.rs", SIZED_CLOSURE_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "sized closure drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "call 8\ndtor 7\nafter drop c\n",
        "the sized closure's captured destructor runs at the explicit drop"
    );
}

// ----- Drop demand pass (Lane BS) -----

const GENERIC_DROP_SRC: &str = r#"
struct G<T> { x: T }
impl<T> Drop for G<T> {
    fn drop(&mut self) { println!("dropped"); }
}
fn main() {
    let _g = G { x: 1i32 };
    println!("body");
}
"#;

#[test]
fn run_mode_generic_drop_impl_runs() {
    // A generic `Drop` impl has no monomorphic instance among the
    // crate's items and drop glue is not a call edge, so the driver
    // must demand the instantiation the program drops (Lane BS).
    // Before that pass existed this exited 2 with
    // "unsupported terminator: Drop".
    let dir = temp_dir("generic-drop");
    let src = fixture(&dir, "generic_drop.rs", GENERIC_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic Drop impl must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "body\ndropped\n",
        "the destructor's output must follow the body's"
    );
}

const TWO_INSTANCE_DROP_SRC: &str = r#"
struct G<T> { x: T }
impl<T> Drop for G<T> {
    fn drop(&mut self) { println!("dropped"); }
}
fn main() {
    let _a = G { x: 1i32 };
    let _b = G { x: true };
    println!("body");
}
"#;

#[test]
fn run_mode_two_instantiations_of_one_generic_drop_impl() {
    // Two monomorphizations of one generic `Drop` impl in a single
    // program: each drop must find its own instantiation's body.
    let dir = temp_dir("two-instance-drop");
    let src = fixture(&dir, "two_instance_drop.rs", TWO_INSTANCE_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "both instantiations must drop; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "body\ndropped\ndropped\n",
        "both destructors must run"
    );
}

const LIFETIME_DROP_SRC: &str = r#"
struct S<'a> { p: &'a str }
impl<'a> Drop for S<'a> {
    fn drop(&mut self) { println!("dropped {}", self.p); }
}
fn main() {
    let _s = S { p: "x" };
    println!("body");
}
"#;

#[test]
fn run_mode_lifetime_parameterized_drop_impl_runs() {
    // A lifetime-only `Drop` impl is converted by the item loop
    // under its declared spelling (`<S<'a> as ...>::drop`); the
    // drop-demand pass resolves the same instance with the erased
    // spelling (`<S<'_> as ...>::drop`). Registering both left the
    // interpreter's destructor lookup with two candidates, which it
    // treats as ambiguous, and the drop came out as
    // "unsupported terminator: Drop" (the Lane BS demand-pass
    // regression: drop-scope-exit, issue-2735-2 and seven more).
    // The demand pass now skips instances that are already
    // converted, by DefId.
    let dir = temp_dir("lifetime-drop");
    let src = fixture(&dir, "lifetime_drop.rs", LIFETIME_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "lifetime-parameterized Drop impl must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "body\ndropped x\n",
        "the destructor's output must follow the body's"
    );
}

const TRANSMUTE_BOX_SRC: &str = r#"
fn inp() -> u32 { let mut s = 0u32; let mut i = 0; while i < 41 { s += 1; i += 1; } s }
fn main() {
    let n = inp();
    let p: *mut u32 = unsafe { std::mem::transmute(Box::new(n)) };
    unsafe { *p += 1; }
    let q: *const u32 = unsafe { std::mem::transmute(Box::new(n + 2)) };
    println!("{} {}", unsafe { *p }, unsafe { *q });
}
"#;

#[test]
fn run_mode_transmute_box_to_raw_pointer() {
    run_ok("transmute-box", TRANSMUTE_BOX_SRC, "42 43\n");
}

const TRANSMUTE_RAW_TO_BOX_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};
static DROPS: AtomicUsize = AtomicUsize::new(0);
struct D(u32);
impl Drop for D {
    fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); }
}
fn main() {
    let p: *mut D = Box::into_raw(Box::new(D(41)));
    let mut b: Box<D> = unsafe { std::mem::transmute(p) };
    b.0 += 1;
    println!("{}", b.0);
    drop(b);
    println!("drops={}", DROPS.load(Ordering::SeqCst));
    let p2: *mut D = Box::into_raw(Box::new(D(7)));
    let b2: Box<D> = unsafe { std::mem::transmute(p2) };
    let p3: *mut D = unsafe { std::mem::transmute(b2) };
    let b3: Box<D> = unsafe { std::mem::transmute(p3) };
    println!("{}", b3.0);
    drop(b3);
    println!("drops={}", DROPS.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_transmute_raw_pointer_to_box() {
    run_ok(
        "transmute-raw-to-box",
        TRANSMUTE_RAW_TO_BOX_SRC,
        "42\ndrops=1\n7\ndrops=2\n",
    );
}

const RC_ARC_DROP_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;
static mut CHECK: u32 = 0;
struct Cat;
impl Drop for Cat {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}
fn main() {
    {
        let _r = Rc::new(Cat);
        let _a = Arc::new(Cat);
    }
    unsafe { assert_eq!(CHECK, 2); }
    println!("rc-arc ok");
}
"#;

#[test]
fn run_mode_rc_arc_drop_runs_pointee_destructor() {
    let dir = temp_dir("rc-arc-drop");
    let src = fixture(&dir, "rc_arc_drop.rs", RC_ARC_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping an Rc / Arc must run the pointee destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "rc-arc ok\n");
}

const BOX_DYN_DROP_SRC: &str = r#"
static mut CHECK: u32 = 0;
trait Dummy {
    fn probe(&self) -> u32;
}
struct Cat;
impl Dummy for Cat {
    fn probe(&self) -> u32 {
        7
    }
}
impl Drop for Cat {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}
struct Holder {
    d: Box<dyn Dummy>,
}
fn main() {
    {
        let b: Box<dyn Dummy> = Box::new(Cat);
        assert_eq!(b.probe(), 7);
        let _h = Holder { d: Box::new(Cat) };
    }
    unsafe { assert_eq!(CHECK, 2); }
    println!("box-dyn ok");
}
"#;

#[test]
fn run_mode_box_dyn_drop_runs_concrete_destructor() {
    let dir = temp_dir("box-dyn-drop");
    let src = fixture(&dir, "box_dyn_drop.rs", BOX_DYN_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a Box<dyn Trait> (directly or as a field) must run the concrete destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "box-dyn ok\n");
}

const DYN_EQ_BOX_SRC: &str = r#"
trait Foo: PartialEq<i32> + std::fmt::Debug {
    fn a(&self) -> i32 { 100 }
}
impl Foo for i32 {}
fn main() {
    let obj: Box<dyn Foo> = Box::new(1);
    assert_eq!(*obj, 1);
    assert!(*obj != 2);
    println!("dyn-eq-box ok");
}
"#;

#[test]
fn run_mode_box_dyn_partial_eq_compares_pointee() {
    let dir = temp_dir("dyn-eq-box");
    let src = fixture(&dir, "dyn_eq_box.rs", DYN_EQ_BOX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing through a Box<dyn trait> with a PartialEq supertrait must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-box ok\n");
}

const FROM_BOX_DYN_ERROR_DROP_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct MyErr { code: i32, other: i32 }
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "err {}", self.code) }
}
impl Error for MyErr {}
impl Drop for MyErr {
    fn drop(&mut self) { println!("dropping MyErr {}", self.code); }
}
fn main() {
    let b: Box<dyn Error> = Box::from(MyErr { code: 7, other: 1 });
    println!("made the box");
    drop(b);
    println!("end of main");
}
"#;

#[test]
fn run_mode_from_box_dyn_error_payload_drop_order() {
    let dir = temp_dir("from-box-dyn-error");
    let src = fixture(&dir, "from_box_dyn_error.rs", FROM_BOX_DYN_ERROR_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a From conversion into Box<dyn Error> must produce a droppable trait object; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "made the box\ndropping MyErr 7\nend of main\n"
    );
}

const INTO_BOX_DYN_ERROR_DROP_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct MyErr { code: i32 }
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "err {}", self.code) }
}
impl Error for MyErr {}
impl Drop for MyErr {
    fn drop(&mut self) { println!("dropping MyErr {}", self.code); }
}
fn main() {
    let b: Box<dyn Error> = MyErr { code: 9 }.into();
    println!("made the box");
    drop(b);
    println!("end of main");
}
"#;

#[test]
fn run_mode_into_box_dyn_error_payload_drop_order() {
    let dir = temp_dir("into-box-dyn-error");
    let src = fixture(&dir, "into_box_dyn_error.rs", INTO_BOX_DYN_ERROR_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an Into conversion into Box<dyn Error> must produce a droppable trait object; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "made the box\ndropping MyErr 9\nend of main\n"
    );
}

const FROM_BOX_DYN_ERROR_ZST_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct Unit;
impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "unit") }
}
impl Error for Unit {}
impl Drop for Unit {
    fn drop(&mut self) { println!("dropping Unit"); }
}
fn main() {
    let b: Box<dyn Error> = Box::from(Unit);
    println!("made the box");
    println!("end of main");
}
"#;

#[test]
fn run_mode_from_box_dyn_error_zst_payload_drop() {
    let dir = temp_dir("from-box-dyn-error-zst");
    let src = fixture(&dir, "from_box_dyn_error_zst.rs", FROM_BOX_DYN_ERROR_ZST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a From conversion of a zero-sized payload into Box<dyn Error> must stay droppable; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "made the box\nend of main\ndropping Unit\n"
    );
}

const BOX_DYN_ERROR_FROM_RAW_DROP_SRC: &str = r#"
use std::error::Error;
use std::fmt;
#[derive(Debug)]
struct MyErr { code: i32 }
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "err {}", self.code) }
}
impl Error for MyErr {}
impl Drop for MyErr {
    fn drop(&mut self) { println!("dropping MyErr {}", self.code); }
}
fn main() {
    let b: Box<dyn Error> = Box::new(MyErr { code: 13 });
    let raw = Box::into_raw(b);
    let b2 = unsafe { Box::from_raw(raw) };
    println!("rebuilt the box");
    drop(b2);
    println!("end of main");
}
"#;

#[test]
fn run_mode_box_dyn_error_from_raw_drop() {
    // A trait-object box rebuilt by `Box::from_raw` is an
    // aggregate chain around the fat pointer, not the
    // builtin-built surface: its drop plan must descend field
    // 0 of each wrapper layer to the `DynRef` at the chain's
    // bottom and run the payload glue from there.
    let dir = temp_dir("box-dyn-error-from-raw");
    let src = fixture(&dir, "box_dyn_error_from_raw.rs", BOX_DYN_ERROR_FROM_RAW_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a Box<dyn Error> rebuilt by from_raw must run the payload destructor; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "rebuilt the box\ndropping MyErr 13\nend of main\n"
    );
}

const FROM_BOX_DYN_FN_CAPTURE_SRC: &str = r#"
struct S { v: i32, w: i32 }
impl Drop for S {
    fn drop(&mut self) { println!("dropping S {} {}", self.v, self.w); }
}
fn main() {
    let s = S { v: 42, w: 1 };
    let f: Box<dyn Fn(i32) -> i32> = Box::from(move |x: i32| x + s.v + s.w);
    println!("f(1) = {}", f(1));
    println!("end of main");
}
"#;

#[test]
fn run_mode_from_box_dyn_fn_reads_captures_and_drops() {
    let dir = temp_dir("from-box-dyn-fn");
    let src = fixture(&dir, "from_box_dyn_fn.rs", FROM_BOX_DYN_FN_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "calling a Box<dyn Fn> built by From must read the captures through the dyn reference; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "f(1) = 44\nend of main\ndropping S 42 1\n"
    );
}

// ----- box allocation identity (Lane EP) ------------------------------------

// A built `Box`'s pointee lives in the interpreter's allocation
// store and the box value is the allocation pointer, so a
// reference to the pointee names the allocation — never a place
// in the frame that happened to hold a copy of the box. The
// fixtures below are the rustc suite programs whose run used to
// fail with "reference into finished frame #N" for exactly
// that reason (directives stripped; the feature attributes are
// accepted because the driver is built and run with
// RUSTC_BOOTSTRAP=1, like compiletest).

const FUNC_ARG_REF_PATTERN_SRC: &str = r#"// Test argument patterns where we create refs to the inside of
// boxes. Make sure that we don't free the box as we match the
// pattern.

#![feature(deref_patterns)]

fn getaddr(deref!(ref x): Box<usize>) -> *const usize {
    let addr: *const usize = &*x;
    addr
}

fn checkval(deref!(ref x): Box<usize>) -> usize {
    *x
}

pub fn main() {
    let obj: Box<_> = Box::new(1);
    let objptr: *const usize = &*obj;
    let xptr = getaddr(obj);
    assert_eq!(objptr, xptr);

    let obj = Box::new(22);
    assert_eq!(checkval(obj), 22);
}"#;

#[test]
fn run_mode_box_pointee_ref_pattern_arg() {
    // a by-value `Box` argument matched with a ref pattern: the reference the pattern forms names the box's allocation, so the pointer taken in the callee equals the one taken before the move, and the box's drop at the callee's end does not invalidate it
    let dir = temp_dir("func-arg-ref-pattern");
    let src = fixture(&dir, "func_arg_ref_pattern.rs", FUNC_ARG_REF_PATTERN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const UNSAFE_PTR_SRC: &str = r#"// Test that we can use raw ptrs when using `capture_disjoint_fields`.

#![allow(dead_code)]

#[derive(Debug)]
struct S {
    s: String,
    t: String,
}

struct T(*const S);

fn unsafe_imm() {
    let s = "".into();
    let t = "".into();
    let my_speed: Box<S> = Box::new(S { s, t });

    let p : *const S = Box::into_raw(my_speed);
    let t = T(p);

    let c = || unsafe {
        println!("{:?}", (*t.0).s);
    };

    c();
}

fn unsafe_mut() {
    let s = "".into();
    let t = "".into();
    let mut my_speed: Box<S> = Box::new(S { s, t });
    let p : *mut S = &mut *my_speed;

    let c = || {
        let x = unsafe { &mut (*p).s };
        *x = "s".into();
    };
    c();
}

fn main() {
    unsafe_mut();
    unsafe_imm();
}"#;

#[test]
fn run_mode_box_into_raw_pointee_outlives_frame() {
    // `Box::into_raw` leaks the allocation: closures created afterwards dereference the raw pointer into storage no frame owns any more, and must still read (and write) the pointee
    let dir = temp_dir("unsafe-ptr");
    let src = fixture(&dir, "unsafe_ptr.rs", UNSAFE_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const DEREF_MUT_CLOSURE_DROP_ORDER_SRC: &str = r#"//! Regression test for https://github.com/rust-lang/rust/issues/16774

#![feature(deref_patterns)]

use std::ops::{Deref, DerefMut};

struct X(Box<isize>);

static mut DESTRUCTOR_RAN: bool = false;

impl Drop for X {
    fn drop(&mut self) {
        unsafe {
            assert!(!DESTRUCTOR_RAN);
            DESTRUCTOR_RAN = true;
        }
    }
}

impl Deref for X {
    type Target = isize;

    fn deref(&self) -> &isize {
        let &X(ref x) = self;
        x
    }
}

impl DerefMut for X {
    fn deref_mut(&mut self) -> &mut isize {
        let &mut X(ref mut x) = self;
        x
    }
}

fn main() {
    {
        let mut test = X(Box::new(5));
        {
            let mut change = || { *test = 10 };
            change();
        }
        assert_eq!(*test, 10);
    }
    assert!(unsafe { DESTRUCTOR_RAN });
}"#;

#[test]
fn run_mode_box_deref_mut_write_through_pattern_copy() {
    // a user `DerefMut` impl whose pattern match copies the `X` value (and its `Box` field) into its own frame: the `&mut` it returns names the box's allocation, so the closure's write through it is observed by the owner's later read, and the drop order is preserved
    let dir = temp_dir("deref-mut-closure-drop-order");
    let src = fixture(&dir, "deref_mut_closure_drop_order.rs", DEREF_MUT_CLOSURE_DROP_ORDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const DST_DEREF_SRC: &str = r#"// Test that a custom deref with a fat pointer return type does not ICE


use std::ops::Deref;

pub struct Arr {
    ptr: Box<[usize]>
}

impl Deref for Arr {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        &*self.ptr
    }
}

pub fn foo(arr: &Arr) {
    assert_eq!(arr.len(), 3);
    let x: &[usize] = &**arr;
    assert_eq!(x[0], 1);
    assert_eq!(x[1], 2);
    assert_eq!(x[2], 3);
}

fn main() {
    let a = Arr { ptr: Box::new([1, 2, 3]) };
    foo(&a);
}"#;

#[test]
fn run_mode_box_dst_deref() {
    // a custom `Deref` returning `&*self.ptr` for a `Box<[usize]>`: the slice reference is formed from a copy of the box inside the `Deref` shim and must name the allocation, not the shim's frame
    let dir = temp_dir("dst-deref");
    let src = fixture(&dir, "dst_deref.rs", DST_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const DST_DEREF_MUT_SRC: &str = r#"// Test that a custom deref with a fat pointer return type does not ICE


use std::ops::{Deref, DerefMut};

pub struct Arr {
    ptr: Box<[usize]>
}

impl Deref for Arr {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        panic!();
    }
}

impl DerefMut for Arr {
    fn deref_mut(&mut self) -> &mut [usize] {
        &mut *self.ptr
    }
}

pub fn foo(arr: &mut Arr) {
    let x: &mut [usize] = &mut **arr;
    assert_eq!(x[0], 1);
    assert_eq!(x[1], 2);
    assert_eq!(x[2], 3);
}

fn main() {
    let mut a = Arr { ptr: Box::new([1, 2, 3]) };
    foo(&mut a);
}"#;

#[test]
fn run_mode_box_dst_deref_mut() {
    // the `DerefMut` counterpart: `&mut *self.ptr` for a `Box<[usize]>` names the allocation, so element reads through it in the caller observe the array
    let dir = temp_dir("dst-deref-mut");
    let src = fixture(&dir, "dst_deref_mut.rs", DST_DEREF_MUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const REGIONS_DEPENDENT_ADDR_OF_SRC: &str = r#"// Test lifetimes are linked properly when we create dependent region pointers.
// Issue #3148.
#![feature(deref_patterns)]

struct A {
    value: B
}

struct B {
    v1: isize,
    v2: [isize; 3],
    v3: Vec<isize> ,
    v4: C,
    v5: Box<C>,
    v6: Option<C>
}

#[derive(Copy, Clone)]
struct C {
    f: isize
}

fn get_v1(a: &A) -> &isize {
    // Region inferencer must deduce that &v < L2 < L1
    let foo = &a.value; // L1
    &foo.v1             // L2
}

fn get_v2(a: &A, i: usize) -> &isize {
    let foo = &a.value;
    &foo.v2[i]
}

fn get_v3(a: &A, i: usize) -> &isize {
    let foo = &a.value;
    &foo.v3[i]
}

fn get_v4(a: &A, _i: usize) -> &isize {
    let foo = &a.value;
    &foo.v4.f
}

fn get_v5(a: &A, _i: usize) -> &isize {
    let foo = &a.value;
    &foo.v5.f
}

fn get_v6_a(a: &A, _i: usize) -> &isize {
    match a.value.v6 {
        Some(ref v) => &v.f,
        None => panic!()
    }
}

fn get_v6_b(a: &A, _i: usize) -> &isize {
    match *a {
        A { value: B { v6: Some(ref v), .. } } => &v.f,
        _ => panic!()
    }
}

fn get_v6_c(a: &A, _i: usize) -> &isize {
    match a {
        &A { value: B { v6: Some(ref v), .. } } => &v.f,
        _ => panic!()
    }
}

fn get_v5_ref(a: &A, _i: usize) -> &isize {
    match &a.value {
        &B {v5: C {f: ref v}, ..} => v
    }
}

pub fn main() {
    let a = A {value: B {v1: 22,
                         v2: [23, 24, 25],
                         v3: vec![26, 27, 28],
                         v4: C { f: 29 },
                         v5: Box::new(C { f: 30 }),
                         v6: Some(C { f: 31 })}};

    let p = get_v1(&a);
    assert_eq!(*p, a.value.v1);

    let p = get_v2(&a, 1);
    assert_eq!(*p, a.value.v2[1]);

    let p = get_v3(&a, 1);
    assert_eq!(*p, a.value.v3[1]);

    let p = get_v4(&a, 1);
    assert_eq!(*p, a.value.v4.f);

    let p = get_v5(&a, 1);
    assert_eq!(*p, a.value.v5.f);

    let p = get_v6_a(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v6_b(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v6_c(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v5_ref(&a, 1);
    assert_eq!(*p, a.value.v5.f);
}"#;

#[test]
fn run_mode_box_dependent_addr_of() {
    // `&foo.v5.f` where `v5: Box<C>` is copied out of the borrowed struct into the callee: the returned reference names the box's allocation and stays valid after the callee returns
    let dir = temp_dir("regions-dependent-addr-of");
    let src = fixture(&dir, "regions_dependent_addr_of.rs", REGIONS_DEPENDENT_ADDR_OF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const ARBITRARY_SELF_TYPES_SRC: &str = r#"#![feature(arbitrary_self_types)]
#![allow(unused_allocation)]

struct SmartPtr<T: ?Sized>(T);

impl<T: ?Sized> std::ops::Receiver for SmartPtr<T> {
    type Target = T;
}

trait Trait {
    fn trait_method<'a>(self: &'a Box<SmartPtr<Self>>) -> &'a [i32];
}

impl Trait for Vec<i32> {
    fn trait_method<'a>(self: &'a Box<SmartPtr<Self>>) -> &'a [i32] {
        &(**self).0
    }
}

fn main() {
    let v = vec![1, 2, 3];

    assert_eq!(&[1, 2, 3], Box::new(SmartPtr(v)).trait_method());
}"#;

#[test]
fn run_mode_box_arbitrary_self_types_receiver() {
    // `&(**self).0` with `self: &Box<SmartPtr<Self>>`: the box is copied into the method's frame and the returned slice reference names the pointee chain inside the allocation
    let dir = temp_dir("arbitrary-self-types");
    let src = fixture(&dir, "arbitrary_self_types.rs", ARBITRARY_SELF_TYPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
}

const BOX_PTR_IDENTITY_SRC: &str = r#"
fn getaddr(b: Box<usize>) -> *const usize {
    &*b as *const usize
}
fn main() {
    let b = Box::new(3usize);
    let p1 = &*b as *const usize;
    let p2 = getaddr(b);
    assert_eq!(p1, p2);
    println!("ok");
}
"#;

#[test]
fn run_mode_box_pointer_identity_across_move() {
    // Pointer identity of a moved box: the address taken from
    // the box before the move and the address extracted from
    // the moved box inside the callee are the same address,
    // because both name the one allocation.
    let dir = temp_dir("box-ptr-identity");
    let src = fixture(&dir, "box_ptr_identity.rs", BOX_PTR_IDENTITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pointer identity across a move must hold; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_INTO_RAW_WRITE_SRC: &str = r#"
fn main() {
    let p = Box::into_raw(Box::new(7));
    println!("{}", unsafe { *p });
    unsafe { *p = 9; }
    println!("{}", unsafe { *p });
}
"#;

#[test]
fn run_mode_box_into_raw_write_and_read_back() {
    // A leaked allocation stays readable and writable through
    // the raw pointer after `into_raw`'s frame is gone.
    let dir = temp_dir("box-into-raw-write");
    let src = fixture(&dir, "box_into_raw_write.rs", BOX_INTO_RAW_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the leaked pointee must stay accessible; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n9\n");
}

const BOX_DYN_FIELD_PROJ_SRC: &str = r#"
trait G { fn get(&self) -> i32; }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct S { m: M }
impl G for S { fn get(&self) -> i32 { self.m.0 } }
fn main() {
    let b: Box<dyn G> = Box::new(S { m: M(9) });
    println!("get {}", b.get());
    drop(b);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_field_projection_reads_payload_field() {
    // The method body reads `self.m.0`: a multi-step field
    // projection through the box deref, where the pointee lives
    // behind the built box's allocation pointer. The receiver
    // must denote the allocation, not a value holding the
    // pointer, or the projection fails on a reference where an
    // ADT value belongs. The payload's destructor runs at the
    // `drop` point, after the read.
    let dir = temp_dir("box-dyn-field-proj");
    let src = fixture(&dir, "box_dyn_field_proj.rs", BOX_DYN_FIELD_PROJ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a Box<dyn Trait> method reading a nested payload field must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "get 9\ndtor 9\nafter drop\n"
    );
}

const BOX_SIZED_FIELD_RW_SRC: &str = r#"
struct M(i32);
struct S { m: M }
fn main() {
    let mut b = Box::new(S { m: M(9) });
    println!("read {}", b.m.0);
    b.m.0 += 5;
    println!("after write {}", b.m.0);
    (*b).m.0 = 100;
    println!("after deref write {}", b.m.0);
}
"#;

#[test]
fn run_mode_box_sized_field_read_write_through_deref() {
    // Field reads and writes through a sized Box's deref reach
    // the pointee in the allocation store: both the auto-deref
    // assignment and the explicit `(*b)` assignment land.
    let dir = temp_dir("box-sized-field-rw");
    let src = fixture(&dir, "box_sized_field_rw.rs", BOX_SIZED_FIELD_RW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "field reads/writes through a sized Box deref must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "read 9\nafter write 14\nafter deref write 100\n"
    );
}

const BOX_NEW_UNINIT_ASSUME_INIT_SRC: &str = r#"
use std::mem::MaybeUninit;
struct P { x: u32, y: u32 }
fn main() {
    let mut b: Box<MaybeUninit<(u32, u32)>> = Box::new_uninit();
    unsafe { b.as_mut_ptr().write((11, 22)); }
    let mut b = unsafe { b.assume_init() };
    println!("{} {}", b.0, b.1);
    b.0 = 99;
    println!("{} {}", b.0, b.1);
    let mut a: Box<MaybeUninit<[u32; 3]>> = Box::new_uninit();
    unsafe { a.as_mut_ptr().write([7, 8, 9]); }
    let a = unsafe { a.assume_init() };
    println!("{} {} {}", a[0], a[1], a[2]);
    let mut p: Box<MaybeUninit<P>> = Box::new_uninit();
    unsafe { p.as_mut_ptr().write(P { x: 5, y: 6 }); }
    let p = unsafe { p.assume_init() };
    println!("{} {}", p.x, p.y);
}
"#;

#[test]
fn run_mode_box_new_uninit_assume_init_reads_payload() {
    // `Box::new_uninit` + `write` + `assume_init`: the payload
    // was written through the wrapper-entering `as_mut_ptr`
    // pointer, and `assume_init`'s transmute retargets the box
    // without the entry step, so the reads and writes through
    // the retargeted box must still address the payload --
    // tuple fields, array elements and struct fields alike.
    let dir = temp_dir("box-new-uninit-assume-init");
    let src = fixture(
        &dir,
        "box_new_uninit_assume_init.rs",
        BOX_NEW_UNINIT_ASSUME_INIT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box::new_uninit + assume_init payload access must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "11 22\n99 22\n7 8 9\n5 6\n"
    );
}

const BOX_MAYBEUNINIT_ENUM_SRC: &str = r#"
#[derive(Clone, Copy, PartialEq, Debug)]
enum E { A(u32), B(u16) }
fn main() {
    let mut b = Box::new(std::mem::MaybeUninit::<E>::uninit());
    unsafe {
        b.as_mut_ptr().write(E::B(0x1234));
        assert_eq!(*b.assume_init(), E::B(0x1234));
    }
    let mut c = Box::new(std::mem::MaybeUninit::<E>::uninit());
    unsafe {
        c.as_mut_ptr().write(E::A(7));
        let c2 = c.assume_init();
        match *c2 {
            E::A(x) => assert_eq!(x, 7),
            E::B(_) => panic!("wrong variant"),
        }
        let p = Box::into_raw(c2);
        assert_eq!(*p, E::A(7));
        drop(Box::from_raw(p));
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_box_maybeuninit_enum_assume_init_reads_payload() {
    // `Box<MaybeUninit<E>>` where `E`'s first variant takes one
    // field: the wrapper's field count coincides with that
    // variant's declared count, so the entry judgment at the
    // retargeted dereference must rest on the stored field's
    // fit for the declared field type -- the payload (a complete
    // `E` value) cannot be a `u32`, so the stored enum is the
    // wrapper and the reads address the payload.
    let dir = temp_dir("box-maybeuninit-enum-assume-init");
    let src = fixture(&dir, "box_maybeuninit_enum.rs", BOX_MAYBEUNINIT_ENUM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<MaybeUninit<enum>> assume_init reads must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_MAYBEUNINIT_ENUM_CONTROLS_SRC: &str = r#"
#[derive(Clone, Copy, PartialEq, Debug)]
enum E { A(u32), B(u16) }
#[derive(Clone, Copy, PartialEq, Debug)]
enum G { A(E), B(u8) }
#[derive(Clone, Copy, PartialEq, Debug)]
enum H { Only(u32) }
fn main() {
    // A genuine `G::A(E::B(..))` value: the stored enum has the
    // wrapper's shape, but its field fits the declared field
    // type, so it stays the position's own value.
    let g = Box::new(G::A(E::B(0x1234)));
    assert_eq!(*g, G::A(E::B(0x1234)));
    // A one-field enum through `MaybeUninit` storage peels, and
    // the same value stored directly does not double-peel.
    let h = Box::new(H::Only(3));
    assert_eq!(*h, H::Only(3));
    let mut h2 = Box::new(std::mem::MaybeUninit::<H>::uninit());
    unsafe {
        h2.as_mut_ptr().write(H::Only(3));
        assert_eq!(*h2.assume_init(), H::Only(3));
    }
    // Reading the box as `MaybeUninit<E>` keeps the union value.
    let mut m = Box::new(std::mem::MaybeUninit::<E>::uninit());
    unsafe {
        m.as_mut_ptr().write(E::B(0x1234));
        let mu: std::mem::MaybeUninit<E> = *m;
        assert_eq!(mu.assume_init(), E::B(0x1234));
    }
    println!("ok");
}
"#;

#[test]
fn run_mode_box_maybeuninit_enum_entry_controls() {
    // The controls around the `MaybeUninit` entry judgment for
    // enum payloads: a nested enum value that merely looks like
    // a wrapper is not entered, a one-field enum payload is
    // entered, and a `MaybeUninit`-typed read is never entered.
    let dir = temp_dir("box-maybeuninit-enum-controls");
    let src = fixture(
        &dir,
        "box_maybeuninit_enum_controls.rs",
        BOX_MAYBEUNINIT_ENUM_CONTROLS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "MaybeUninit entry control cases must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const BOX_DYN_TUPLE_PAYLOAD_SRC: &str = r#"
trait G { fn get(&self) -> i32; }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct T(M, i32);
impl G for T { fn get(&self) -> i32 { self.0.0 + self.1 } }
fn main() {
    let b: Box<dyn G> = Box::new(T(M(30), 12));
    println!("get {}", b.get());
    drop(b);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_tuple_payload_field_projection() {
    // A tuple-struct payload behind a Box<dyn Trait>: the
    // method projects two fields of the tuple, one of them a
    // nested tuple field of a Drop type.
    let dir = temp_dir("box-dyn-tuple-payload");
    let src = fixture(&dir, "box_dyn_tuple_payload.rs", BOX_DYN_TUPLE_PAYLOAD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a Box<dyn Trait> method projecting tuple payload fields must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "get 42\ndtor 30\nafter drop\n"
    );
}

const BOX_DYN_DEEP_FIELD_SRC: &str = r#"
trait G { fn get(&self) -> i32; }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct Inner { m: M }
struct S { inner: Inner }
impl G for S { fn get(&self) -> i32 { self.inner.m.0 } }
fn main() {
    let b: Box<dyn G> = Box::new(S { inner: Inner { m: M(7) } });
    println!("get {}", b.get());
    drop(b);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_deep_field_projection() {
    // Three projection steps past the box deref
    // (`self.inner.m.0`) resolve through the allocation.
    let dir = temp_dir("box-dyn-deep-field");
    let src = fixture(&dir, "box_dyn_deep_field.rs", BOX_DYN_DEEP_FIELD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a Box<dyn Trait> method projecting three fields deep must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "get 7\ndtor 7\nafter drop\n"
    );
}

const BOX_DYN_MUT_WRITE_SRC: &str = r#"
trait G { fn get(&self) -> i32; fn bump(&mut self); }
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct S { m: M }
impl G for S {
    fn get(&self) -> i32 { self.m.0 }
    fn bump(&mut self) { self.m.0 += 1; }
}
fn main() {
    let mut b: Box<dyn G> = Box::new(S { m: M(9) });
    b.bump();
    println!("get {}", b.get());
    drop(b);
    println!("after drop");
}
"#;

#[test]
fn run_mode_box_dyn_mut_method_writes_through_allocation() {
    // A `&mut self` method writes through the same receiver:
    // the write lands in the allocation store, so the later
    // read sees it and the destructor observes the written
    // value.
    let dir = temp_dir("box-dyn-mut-write");
    let src = fixture(&dir, "box_dyn_mut_write.rs", BOX_DYN_MUT_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a &mut dyn method writing a payload field must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "get 10\ndtor 10\nafter drop\n"
    );
}

const RC_GET_MUT_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let mut r = Rc::new(41);
    *Rc::get_mut(&mut r).unwrap() += 1;
    println!("{}", *r);
}
"#;

#[test]
fn run_mode_rc_get_mut_write_is_observed_by_the_owner() {
    // `Rc::get_mut` runs its converted std body against the
    // allocation the handle names: the write through the returned
    // `&mut` lands in the shared storage and the owner reads it
    // back.
    let dir = temp_dir("rc-get-mut");
    let src = fixture(&dir, "rc_get_mut.rs", RC_GET_MUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc::get_mut on a unique Rc must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const RC_GET_MUT_SHARED_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let mut r = Rc::new(41);
    println!("unique: {}", Rc::get_mut(&mut r).is_some());
    let r2 = r.clone();
    println!("shared: {}", Rc::get_mut(&mut r).is_some());
    drop(r2);
    println!("unique again: {}", Rc::get_mut(&mut r).is_some());
}
"#;

#[test]
fn run_mode_rc_get_mut_is_none_while_shared() {
    // The strong/weak counts live in the allocation, so the
    // converted `get_mut` body observes clones and drops exactly
    // as natively: `None` while a second handle shares the value.
    let dir = temp_dir("rc-get-mut-shared");
    let src = fixture(&dir, "rc_get_mut_shared.rs", RC_GET_MUT_SHARED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc::get_mut must track sharing through clone/drop; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "unique: true\nshared: false\nunique again: true\n"
    );
}

const ARC_GET_MUT_SHARED_SRC: &str = r#"
use std::sync::Arc;
fn main() {
    let mut a = Arc::new(41);
    println!("unique: {}", Arc::get_mut(&mut a).is_some());
    let a2 = a.clone();
    println!("shared: {}", Arc::get_mut(&mut a).is_some());
    drop(a2);
    println!("unique again: {}", Arc::get_mut(&mut a).is_some());
}
"#;

#[test]
fn run_mode_arc_get_mut_is_none_while_shared() {
    let dir = temp_dir("arc-get-mut-shared");
    let src = fixture(&dir, "arc_get_mut_shared.rs", ARC_GET_MUT_SHARED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Arc::get_mut must track sharing through clone/drop; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "unique: true\nshared: false\nunique again: true\n"
    );
}

const ARC_MAKE_MUT_SRC: &str = r#"
use std::sync::Arc;
fn main() {
    let mut a = Arc::new(41);
    *Arc::make_mut(&mut a) += 1;
    println!("{}", *a);
}
"#;

#[test]
fn run_mode_arc_make_mut_on_unique_arc_writes_in_place() {
    // The unique path of the converted `make_mut` body: the
    // layout intrinsics resolve, the uniqueness check reads the
    // counts from the allocation, and the returned `&mut` writes
    // into the same storage the owner reads.
    let dir = temp_dir("arc-make-mut");
    let src = fixture(&dir, "arc_make_mut.rs", ARC_MAKE_MUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Arc::make_mut on a unique Arc must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const RC_MAKE_MUT_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let mut r = Rc::new(41);
    *Rc::make_mut(&mut r) += 1;
    println!("{}", *r);
}
"#;

#[test]
fn run_mode_rc_make_mut_on_unique_rc_writes_in_place() {
    let dir = temp_dir("rc-make-mut");
    let src = fixture(&dir, "rc_make_mut.rs", RC_MAKE_MUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc::make_mut on a unique Rc must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const RC_WEAK_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let r = Rc::new(41);
    let w = Rc::downgrade(&r);
    println!("{}", *w.upgrade().unwrap());
    println!("{}", Rc::strong_count(&r));
    println!("{}", Rc::weak_count(&r));
    drop(r);
    println!("{}", w.upgrade().is_none());
}
"#;

#[test]
fn run_mode_rc_weak_upgrade_and_counts() {
    // Downgrade bumps the weak count in the allocation; upgrade
    // observes the strong count, so it succeeds while a strong
    // handle lives and returns `None` once the last one drops -
    // the storage outlives the pointee on the weak bundle.
    let dir = temp_dir("rc-weak");
    let src = fixture(&dir, "rc_weak.rs", RC_WEAK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc::downgrade / Weak::upgrade must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "41\n1\n1\ntrue\n");
}

const RC_DROP_ONCE_SRC: &str = r#"
use std::rc::Rc;
static mut N: usize = 0;
struct D;
impl Drop for D {
    fn drop(&mut self) {
        unsafe { N += 1; }
    }
}
fn main() {
    let r = Rc::new(D);
    let r2 = r.clone();
    drop(r);
    println!("after drop r: {}", unsafe { N });
    drop(r2);
    println!("after drop r2: {}", unsafe { N });
}
"#;

#[test]
fn run_mode_rc_pointee_glue_runs_once_at_the_last_handle() {
    // Dropping one of two handles only decrements the count in
    // the allocation; the pointee's destructor runs when the
    // last strong handle drops, not once per handle copy.
    let dir = temp_dir("rc-drop-once");
    let src = fixture(&dir, "rc_drop_once.rs", RC_DROP_ONCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc drop glue must be count-driven; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "after drop r: 0\nafter drop r2: 1\n"
    );
}

const ARC_MAKE_MUT_COW_SRC: &str = r#"
use std::sync::Arc;
fn main() {
    let mut a = Arc::new(41);
    let a2 = a.clone();
    *Arc::make_mut(&mut a) += 1;
    println!("{} {}", *a, *a2);
}
"#;

#[test]
fn run_mode_arc_make_mut_shared_stops_at_the_allocator() {
    // The copy-on-write path of `make_mut` allocates the
    // replacement `ArcInner` through the allocator machinery
    // (`UniqueArcUninit`, `__rust_alloc`), which needs a
    // byte-addressable allocation model the interpreter does not
    // have yet. Until it does, the run must stop with an explicit
    // unsupported (exit 2) rather than answering wrongly.
    let dir = temp_dir("arc-make-mut-cow");
    let src = fixture(&dir, "arc_make_mut_cow.rs", ARC_MAKE_MUT_COW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "make_mut's shared path is unsupported until the allocator is modelled; stderr={stderr}"
    );
    assert!(
        stderr.contains("unsupported"),
        "the stop names itself as unsupported: {stderr}"
    );
}

const PIN_BOX_DROP_SRC: &str = r#"
struct D(u32);
impl Drop for D { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    { let _p = std::pin::Pin::new(Box::new(D(1))); }
    println!("scope done");
    let p = std::pin::Pin::new(Box::new(D(2)));
    std::mem::drop(p);
    println!("end");
}
"#;

#[test]
fn run_mode_pin_box_drop_and_mem_drop_run_pointee_dtor() {
    // `Pin<P>` has no destructor of its own: dropping it drops
    // the wrapped pointer, so a `Pin<Box<T>>` runs the pointee's
    // glue both at scope end and through `mem::drop`.
    let dir = temp_dir("pin-box-drop");
    let src = fixture(&dir, "pin_box_drop.rs", PIN_BOX_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Pin<Box<T>> drops must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dtor 1\nscope done\ndtor 2\nend\n",
        "the pointee destructor runs at each Pin drop"
    );
}

const BOXED_VIEWS_SRC: &str = r#"fn main() {
    let b: Box<str> = "hi".to_owned().into_boxed_str();
    println!("boxed {}", &*b);
    let v: Box<[i32]> = vec![10, 20, 30].into_boxed_slice();
    println!("slice {} {}", v.len(), v[1]);
    let mut s: Box<str> = "ab".to_owned().into_boxed_str();
    s.make_ascii_uppercase();
    println!("upper {}", &*s);
    let text = String::from("hello");
    let r: &str = text.as_ref();
    println!("asref {}", r);
    let bytes: &[u8] = text.as_ref();
    println!("bytes {}", bytes.len());
    let vec = vec![1u8, 2, 3];
    let vb: &[u8] = vec.as_ref();
    println!("vec {}", vb.len());
    let mut m = String::from("xy");
    let mm: &mut str = m.as_mut();
    mm.make_ascii_uppercase();
    println!("asmut {}", m);
}
"#;

const NESTED_BOX_MOVE_SRC: &str = r#"struct A {
    x: Box<isize>,
    y: isize,
}
struct C {
    x: Box<A>,
    y: isize,
}
fn main() {
    let a: Box<C> = Box::new(C { x: Box::new(A { x: Box::new(0), y: 1 }), y: 2 });
    let _x = a.x.x;
    let _y = a.y;
    println!("nested ok");
}
"#;

#[test]
fn run_mode_nested_box_partial_move_and_drop() {
    // Moving `a.x.x` out of nested boxes and dropping the
    // remainder runs the drop shims, whose places walk the
    // `Box` -> `Unique` -> `NonNull` -> pointer field chain:
    // in the value model the box's one field is the pointer
    // itself, so each field-0 step of that chain yields it
    // unchanged.
    let dir = temp_dir("nested-box-move");
    let src = fixture(&dir, "nested_box.rs", NESTED_BOX_MOVE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "nested ok\n");
}

#[test]
fn run_mode_boxed_conversions_and_ref_views() {
    // `String::into_boxed_str` and `Vec::into_boxed_slice` move
    // the `vec` field out of the value and rebuild a box from
    // raw parts in their collected bodies, which the string /
    // vector models cannot serve; the `AsRef` / `AsMut` bodies
    // for the std containers build their views the same way.
    // The builtins compute the views from the modelled data
    // directly.
    let dir = temp_dir("boxed-views");
    let src = fixture(&dir, "boxed_views.rs", BOXED_VIEWS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        [
            "boxed hi",
            "slice 3 20",
            "upper AB",
            "asref hello",
            "bytes 5",
            "vec 3",
            "asmut XY",
        ]
    );
}

const BOX_CLONE_SRC: &str = r#"
fn main() {
    let mut i = Box::new(1);
    let mut j = i.clone();
    *j = 3;
    println!("i={} j={}", *i, *j);
    let p1 = &*i as *const i32 as usize;
    let p2 = &*j as *const i32 as usize;
    println!("same_addr={}", p1 == p2);
    let mut a: Vec<Box<_>> = vec![Box::new(10)];
    let b = a.clone();
    *a[0] = 20;
    println!("a={} b={}", *a[0], *b[0]);
}
"#;

#[test]
fn run_mode_box_clone_owns_its_pointee() {
    // A cloned box (and the boxes inside a cloned vector) own fresh
    // storage: writing through one must not rewrite the other.
    run_ok(
        "box-clone",
        BOX_CLONE_SRC,
        "i=1 j=3\nsame_addr=false\na=20 b=10\n",
    );
}

const MANUALLY_DROP_DEREF_SRC: &str = r#"
use std::mem::ManuallyDrop;

struct Counter { n: u32 }
impl Counter {
    fn bump(&mut self) { self.n += 1; }
    fn get(&self) -> u32 { self.n }
}

fn main() {
    let mut md = ManuallyDrop::new(Counter { n: 41 });
    assert_eq!(md.get(), 41);
    assert_eq!(md.n, 41);
    md.bump();
    assert_eq!(md.get(), 42);
    *md = Counter { n: 7 };
    assert_eq!(md.n, 7);
    let mut v = ManuallyDrop::new(vec![1, 2, 3]);
    v.push(4);
    assert_eq!(v.len(), 4);
    assert_eq!(v[3], 4);
    let s = ManuallyDrop::new(String::from("hello"));
    assert_eq!(s.len(), 5);
    assert_eq!(&*s, "hello");
    let x = 2u8;
    let md2 = ManuallyDrop::new(&x);
    assert_eq!(**md2, 2);
    println!("ok");
}
"#;

#[test]
fn run_mode_manually_drop_deref_and_deref_mut() {
    // `ManuallyDrop<T>` derefs to its held value: method calls and
    // field reads go through `Deref`, mutations through `DerefMut`
    // land in the wrapper's own storage (a resolved-value snapshot
    // would silently drop them), and assignment through the
    // dereference replaces the held value.
    let dir = temp_dir("manually-drop-deref");
    let src = fixture(&dir, "manually_drop_deref.rs", MANUALLY_DROP_DEREF_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "ok\n", "stdout={stdout}");
}

// ----- drop glue for modelled std shapes (Issue #114) --------------------------

const DROP_MAYBE_UNINIT_SRC: &str = r#"
use std::mem::MaybeUninit;

fn main() {
    let mut v: Vec<MaybeUninit<[u32; 2]>> = Vec::new();
    v.push(MaybeUninit::new([1, 2]));
    v.push(MaybeUninit::uninit());
    let b: Box<MaybeUninit<String>> = Box::new(MaybeUninit::new(String::from("held")));
    drop(v);
    drop(b);
    println!("ok");
}
"#;

#[test]
fn run_mode_drop_maybe_uninit_containers() {
    run_ok("drop-maybe-uninit", DROP_MAYBE_UNINIT_SRC, "ok\n");
}

const DROP_CELL_SRC: &str = r#"
use std::cell::Cell;

struct Flag(u32);
impl Drop for Flag {
    fn drop(&mut self) { println!("flag {} dropped", self.0); }
}

fn main() {
    let c = Cell::new(Some(Flag(7)));
    drop(c);
    println!("done");
}
"#;

#[test]
fn run_mode_drop_cell_runs_contents_glue() {
    // The modelled `Cell` value nests its contents two aggregates
    // deep; dropping the cell must reach the contents' destructor.
    run_ok("drop-cell", DROP_CELL_SRC, "flag 7 dropped\ndone\n");
}

const DROP_WAKER_SRC: &str = r#"
use std::sync::Arc;
use std::task::{Wake, Waker};

struct FlagWaker {
    held: String,
}
impl Wake for FlagWaker {
    fn wake(self: Arc<Self>) {}
}

fn main() {
    let w = Waker::from(Arc::new(FlagWaker { held: String::from("x") }));
    drop(w);
    println!("done");
}
"#;

#[test]
fn run_mode_drop_waker_releases_wake_object() {
    // Dropping a Waker releases the counted handle behind it;
    // the erased wake object's own glue is judged by shape (the
    // Dynamic plan), which carries the held string's release.
    run_ok("drop-waker", DROP_WAKER_SRC, "done\n");
}

const DROP_LOCK_GUARDS_SRC: &str = r#"
use std::sync::Mutex;

fn main() {
    let m = Mutex::new(1u32);
    {
        let _g = m.lock().unwrap();
    }
    println!("guards dropped");
}
"#;

#[test]
fn run_mode_drop_lock_guards() {
    // Acquiring and dropping each guard kind: the guards' glue is
    // their fields' (a lock reference and a poison flag), since
    // the lock state itself is not modelled.
    run_ok("drop-lock-guards", DROP_LOCK_GUARDS_SRC, "guards dropped\n");
}

const DROP_TYPEID_SRC: &str = r#"
use std::any::TypeId;

fn main() {
    let ids = [TypeId::of::<u32>(), TypeId::of::<String>()];
    println!("{}", ids[0] == TypeId::of::<u32>());
    drop(ids);
    println!("done");
}
"#;

#[test]
fn run_mode_drop_type_id_values() {
    run_ok("drop-typeid", DROP_TYPEID_SRC, "true\ndone\n");
}

const DROP_FLAT_MAP_SRC: &str = r#"
fn main() {
    let it = Some(())
        .into_iter()
        .flat_map(|_| Some(()).into_iter().flat_map(|_| vec![1, 2]));
    drop(it);
    println!("ok");
}
"#;

#[test]
fn run_mode_drop_flat_map_chain() {
    run_ok("drop-flat-map", DROP_FLAT_MAP_SRC, "ok\n");
}

const DROP_PHANTOM_PINNED_SRC: &str = r#"
use std::marker::PhantomPinned;

struct Pinned {
    value: u32,
    _pin: PhantomPinned,
}
impl Drop for Pinned {
    fn drop(&mut self) { println!("pinned {} dropped", self.value); }
}

fn main() {
    let p = Pinned { value: 3, _pin: PhantomPinned };
    drop(p);
    println!("done");
}
"#;

#[test]
fn run_mode_drop_phantom_pinned_struct() {
    run_ok("drop-phantom-pinned", DROP_PHANTOM_PINNED_SRC, "pinned 3 dropped\ndone\n");
}

const ARRAY_AS_PTR_SRC: &str = r#"
fn main() {
    let arr = [10i32, 20, 30];
    let p = arr.as_ptr();
    let q = &arr as *const i32;
    let r = &arr[0] as *const i32;
    unsafe {
        println!("{}", *p);
        println!("{}", *q);
        println!("{}", p == r);
    }
}
"#;

#[test]
fn run_mode_array_as_ptr_reads_first_element() {
    // An element pointer (from `as_ptr` or an `as` cast) denotes the
    // array's first element, not the whole array aggregate: reading
    // through it used to read the tuple value of the whole array.
    // It also denotes exactly the place `&arr[0]` denotes.
    let dir = temp_dir("array-as-ptr");
    let src = fixture(&dir, "array_as_ptr.rs", ARRAY_AS_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "10\n10\ntrue\n");
}

const ELEM_PTR_WRITE_SRC: &str = r#"
fn main() {
    let mut a = [10i32, 20, 30];
    let p = a.as_mut_ptr();
    unsafe { *p = 99; }
    let mut b = [10i32, 20, 30];
    let q = &mut b as *mut i32;
    unsafe { *q = 7; }
    println!("{} {}", a[0], b[0]);
}
"#;

#[test]
fn run_mode_element_pointer_writes_hit_first_element() {
    // Writing through an element pointer writes the first element;
    // it used to fail as an assignment of a scalar to the whole
    // array place.
    let dir = temp_dir("elem-ptr-write");
    let src = fixture(&dir, "elem_ptr_write.rs", ELEM_PTR_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "99 7\n");
}

const STR_AS_PTR_SRC: &str = r#"
fn main() {
    let s = "hello";
    let owned = String::from("hi");
    let bytes = b"abc";
    unsafe {
        println!("{}", *s.as_ptr());
        println!("{}", *owned.as_ptr());
        println!("{}", *bytes.as_ptr());
    }
}
"#;

#[test]
fn run_mode_str_as_ptr_reads_first_byte() {
    // `str::as_ptr` (and `String::as_ptr` through it) denotes the
    // first byte; it used to read the whole string value, which the
    // formatter then printed verbatim - a silent wrong result.
    let dir = temp_dir("str-as-ptr");
    let src = fixture(&dir, "str_as_ptr.rs", STR_AS_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "104\n104\n97\n");
}

const ELEM_PTR_SHAPES_SRC: &str = r#"
struct S { a: [i32; 3] }
static ARR: [i32; 3] = [1, 2, 3];
fn main() {
    let s = S { a: [10, 20, 30] };
    let arr = [40i32, 50, 60];
    let slice: &[i32] = &arr[..];
    unsafe {
        println!("{}", *s.a.as_ptr());
        println!("{}", *ARR.as_ptr());
        println!("{}", *slice.as_ptr());
    }
}
"#;

#[test]
fn run_mode_element_pointers_for_field_static_and_slice() {
    // The first-element retargeting composes with the projection
    // the base pointer already carries (a struct field), with
    // statics, and with slices (whose pointer rides a fat base).
    let dir = temp_dir("elem-ptr-shapes");
    let src = fixture(&dir, "elem_ptr_shapes.rs", ELEM_PTR_SHAPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "10\n1\n40\n");
}

const ELEM_PTR_ROUNDTRIP_SRC: &str = r#"
fn main() {
    let arr = [10i32, 20, 30];
    let p = arr.as_ptr().cast::<[i32; 3]>();
    unsafe { println!("{}", (*p)[2]); }
}
"#;

#[test]
fn run_mode_element_pointer_cast_back_to_array() {
    // Casting an element pointer back to a whole-array pointer
    // denotes the array again (the index-0 projection the
    // first-element cast added is removed).
    let dir = temp_dir("elem-ptr-roundtrip");
    let src = fixture(&dir, "elem_ptr_roundtrip.rs", ELEM_PTR_ROUNDTRIP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "30\n");
}
const BOX_ASREF_IDENTITY_SRC: &str = r#"
#[repr(align(16))]
#[derive(Clone, Copy)]
struct Align16(i32);

fn main() {
    // The `AsRef` view of a box pointee is the pointee itself:
    // the result must be a reference to the box's storage, not
    // the pointee value.
    let b = Box::new(Align16(41));
    let r: &Align16 = b.as_ref();
    assert_eq!(r.0, 41);
    assert_eq!(std::mem::align_of_val(r), 16);
    assert_eq!(std::mem::size_of_val(r), 16);
    // An identity view of an aggregate receiver is a reference
    // to the receiver, not the receiver value.
    let v = vec![3u32, 4];
    let rv: &Vec<u32> = v.as_ref();
    assert_eq!(rv.len(), 2);
    assert_eq!(rv[0], 3);
    println!("box asref ok");
}
"#;

#[test]
fn run_mode_box_asref_identity_view_is_a_reference() {
    // Regression: the `AsRef` builtin (Lane HD) resolved a
    // `Box<T> -> &T` view to the bare pointee value, which the
    // destination's `Ref` type check rejects ("expected
    // Ref(Adt(..)), found ADT value") — this broke
    // tests/ui/structs-enums/align-struct.rs at both19. The
    // identity views (box pointee, aggregate self view) now
    // return reference-shaped results.
    let dir = temp_dir("box-asref");
    let src = fixture(&dir, "box_asref.rs", BOX_ASREF_IDENTITY_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "box asref ok\n");
}
const BOX_CELL_SRC: &str = r#"
use std::cell::Cell;
fn main() {
    let b = Box::new(Cell::new(1u32));
    b.set(5);
    println!("{}", b.get());
    std::mem::forget(b);
}
"#;

#[test]
fn run_mode_box_cell_set_through_deref() {
    // Method resolution reaches the `Cell` through the `Deref`
    // builtin for `Box`. The builtin used to hand back a snapshot
    // of the pointee value, so `Cell::set` (a write through the
    // reference) failed with "expected mutable place, found
    // constant aggregate". A box is uniquely owned, so the builtin
    // now returns a reference to the place the pointee lives in.
    // (The box is forgotten because dropping a `Box<Cell<u32>>` is
    // a separate drop-glue gap.)
    let dir = temp_dir("box-cell");
    let src = fixture(&dir, "box_cell.rs", BOX_CELL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell::set through a Box deref must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "5\n");
}

const DYN_ANY_DOWNCAST_WRITE_SRC: &str = r#"
use std::any::Any;
use std::cell::Cell;
use std::ops::{Deref, DerefMut};
fn main() {
    let b: Box<dyn Any> = Box::new(String::from("a"));
    let mut b = match b.downcast::<String>() {
        Ok(x) => x,
        Err(_) => panic!("downcast failed"),
    };
    let s: &mut String = DerefMut::deref_mut(&mut b);
    s.push_str("z");
    println!("{}", b.as_str());
    let b: Box<dyn Any> = Box::new(Cell::new(1u32));
    let b = match b.downcast::<Cell<u32>>() {
        Ok(x) => x,
        Err(_) => panic!("downcast failed"),
    };
    let r: &Cell<u32> = Deref::deref(&b);
    r.set(5);
    println!("{}", b.get());
}
"#;

#[test]
fn run_mode_dyn_any_downcast_constructed_box_writes_through_deref() {
    // A by-value downcast of a constructed `Box<dyn Any>` hands
    // back the box itself (the dyn value's base), not a box
    // wrapped around it: writes through the result's `Deref` /
    // `DerefMut` reach the payload, as they do for a box that was
    // never erased.
    run_ok("dyn-any-write", DYN_ANY_DOWNCAST_WRITE_SRC, "az\n5\n");
}


const PANIC_IN_DTOR_SRC: &str = r#"
use std::sync::atomic::{AtomicBool, Ordering};

static FIELD_DROPPED: AtomicBool = AtomicBool::new(false);

struct Field {
    _foo: isize,
}

impl Drop for Field {
    fn drop(&mut self) {
        FIELD_DROPPED.store(true, Ordering::SeqCst);
    }
}

struct Outer {
    _field: Field,
}

impl Drop for Outer {
    fn drop(&mut self) {
        panic!("outer destructor panics");
    }
}

fn main() {
    let handle = std::thread::spawn(|| {
        let _outer = Outer { _field: Field { _foo: 3 } };
    });
    assert!(handle.join().is_err());
    assert!(FIELD_DROPPED.load(Ordering::SeqCst));
    println!("ok");
}
"#;

#[test]
fn run_mode_panicking_destructor_still_drops_fields() {
    // When a destructor panics, the remaining drop glue of the
    // value (its fields) is the cleanup of the drop itself: the
    // field destructors still run while the panic unwinds, and
    // the panic then continues to the thread boundary.
    let dir = temp_dir("panic-in-dtor");
    let src = fixture(&dir, "panic_in_dtor.rs", PANIC_IN_DTOR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a panicking destructor must still drop the fields; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const SWAPPED_DROP_IMPL_SRC: &str = r#"
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNT: AtomicUsize = AtomicUsize::new(0);

struct S<U, V> {
    _u: U,
    size_of_u: usize,
    _v: V,
    size_of_v: usize,
}

impl<U, V> S<U, V> {
    fn new(u: U, v: V) -> Self {
        S {
            _u: u,
            size_of_u: mem::size_of::<U>(),
            _v: v,
            size_of_v: mem::size_of::<V>(),
        }
    }
}

// The impl declares its parameters in the opposite order
// from the self type: the destructor instance for
// `S<u8, u16>` must still substitute U = u8, V = u16.
impl<V, U> Drop for S<U, V> {
    fn drop(&mut self) {
        assert_eq!(mem::size_of::<U>(), self.size_of_u);
        assert_eq!(mem::size_of::<V>(), self.size_of_v);
        COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    assert_eq!(COUNT.load(Ordering::SeqCst), 0);
    {
        S::new(0u8, 1u16);
    }
    assert_eq!(COUNT.load(Ordering::SeqCst), 1);
    println!("swapped drop ok");
}
"#;

#[test]
fn run_mode_drop_impl_with_swapped_generic_params() {
    // A `Drop` impl whose generic parameters are declared in
    // a different order than the self type's must be
    // monomorphized by unifying the impl's self type with the
    // concrete type, not by copying the self type's arguments
    // positionally (which would run the destructor of the
    // swapped instantiation).
    let dir = temp_dir("swapped-drop-impl");
    let src = fixture(&dir, "swapped_drop_impl.rs", SWAPPED_DROP_IMPL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "swapped-parameter Drop impl must substitute correctly; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "swapped drop ok\n");
}

const BOX_DYN_SEND_DROP_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPPED: AtomicUsize = AtomicUsize::new(0);

struct Marker;

impl Drop for Marker {
    fn drop(&mut self) {
        DROPPED.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    {
        let _: Box<dyn Send> = Box::new(Marker);
    }
    assert_eq!(DROPPED.load(Ordering::SeqCst), 1);
    println!("dyn drop ok");
}
"#;

#[test]
fn run_mode_box_dyn_drop_runs_concrete_dtor() {
    // Dropping a `Box<dyn Trait>` runs the concrete type's
    // destructor through the vtable: the unsizing cast's
    // source pointee is a drop candidate like any dropped
    // type, so its destructor instance is demanded, and the
    // glue the cast carries is the concrete type's.
    let dir = temp_dir("box-dyn-send-drop");
    let src = fixture(&dir, "box_dyn_send_drop.rs", BOX_DYN_SEND_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<dyn Trait> drop must run the concrete destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn drop ok\n");
}

const PIN_DROP_SRC: &str = r#"
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[pin_v2]
struct Foo {
    dropped: Arc<AtomicBool>,
}

impl Drop for Foo {
    fn pin_drop(self: Pin<&mut Self>) {
        self.dropped.store(true, Ordering::Relaxed);
    }
}

fn main() {
    let dropped = Arc::new(AtomicBool::new(false));
    {
        let _f = Foo { dropped: dropped.clone() };
    }
    assert!(dropped.load(Ordering::Relaxed));
    println!("pin drop ok");
}
"#;

#[test]
fn run_mode_pin_drop_runs_as_the_destructor() {
    // Under pin ergonomics a type's destructor entry point is
    // `pin_drop(self: Pin<&mut Self>)` and there is no `drop`
    // method: the glue must call it, with the argument in the
    // one-field `Pin` aggregate shape the value model uses for
    // `Pin` everywhere else.
    let dir = temp_dir("pin-drop");
    let src = fixture(&dir, "pin_drop.rs", PIN_DROP_SRC);
    let out = run_driver_extra(&src, &["--edition", "2024"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pin_drop must run as the type's destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "pin drop ok\n");
}

const BOX_DYN_TAIL_STRUCT_SRC: &str = r#"
struct Wrapper<'a, T: ?Sized>(&'a mut i32, T);
impl<'a, T: ?Sized> Drop for Wrapper<'a, T> {
    fn drop(&mut self) { *self.0 = 432; }
}
fn main() {
    let mut x = 0;
    {
        let wrapper = Box::new(Wrapper(&mut x, [1u32, 2, 3]));
        let _: Box<Wrapper<dyn Send>> = wrapper;
    }
    println!("x is {}", x);
}
"#;

#[test]
fn run_mode_box_dyn_tail_struct_drop_runs_dtor() {
    // A `Box` over a struct whose unsized tail is a trait object:
    // the box value is the trait-object form the unsizing
    // coercion produced, and dropping it runs the struct's
    // destructor against the pointee in the allocation — the
    // write through the borrowed field reaches the caller's
    // local, exactly as the sized box's drop does.
    let dir = temp_dir("box-dyn-tail-struct-drop");
    let src = fixture(&dir, "box_dyn_tail_struct_drop.rs", BOX_DYN_TAIL_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<dyn-tail struct> drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "x is 432\n",
        "the dyn-tail struct's destructor runs at the box drop"
    );
}

const RC_DYN_TAIL_STRUCT_SRC: &str = r#"
use std::rc::Rc;
struct Wrapper<T: ?Sized>(i32, T);
impl<T: ?Sized> Drop for Wrapper<T> {
    fn drop(&mut self) { println!("dtor {}", self.0); }
}
fn main() {
    let r: Rc<Wrapper<dyn Send>> = Rc::new(Wrapper(5, [1u32, 2]));
    let r2 = r.clone();
    drop(r);
    println!("one left");
    drop(r2);
    println!("gone");
}
"#;

#[test]
fn run_mode_rc_dyn_tail_struct_drop_runs_dtor() {
    // The reference-counted form of the dyn-tail shape: the
    // handle value is the trait-object form, and the pointee's
    // destructor runs when the last strong handle drops.
    let dir = temp_dir("rc-dyn-tail-struct-drop");
    let src = fixture(&dir, "rc_dyn_tail_struct_drop.rs", RC_DYN_TAIL_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc<dyn-tail struct> drop program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "one left\ndtor 5\ngone\n",
        "the dyn-tail struct's destructor runs at the last handle drop"
    );
}

const BOX_POINTEE_MENTIONS_DYN_SRC: &str = r#"
struct M(i32);
impl Drop for M { fn drop(&mut self) { println!("dtor {}", self.0); } }
struct Holder(Vec<Box<dyn Fn()>>);
impl Drop for Holder {
    fn drop(&mut self) { println!("holder dtor"); }
}
fn main() {
    let m = M(9);
    let h = Box::new(Holder(vec![Box::new(move || { let _ = &m; })]));
    drop(h);
    println!("done");
}
"#;

#[test]
fn run_mode_box_pointee_mentions_dyn_plain_form_drop() {
    // A pointee that merely mentions a trait object (a `Vec` of
    // boxed closures in a field) is a plain box value, not the
    // trait-object form: dropping it runs the pointee's own
    // destructor and its fields' glue as usual.
    let dir = temp_dir("box-pointee-mentions-dyn-drop");
    let src = fixture(&dir, "box_pointee_mentions_dyn.rs", BOX_POINTEE_MENTIONS_DYN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "plain Box over a dyn-mentioning pointee must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "holder dtor\ndtor 9\ndone\n",
        "the plain box's pointee destructor and field glue run at the drop"
    );
}

const RC_DYN_TRAIT_COUNT_SRC: &str = r#"
use std::rc::Rc;
trait Tr { fn hi(&self) -> i32; }
struct S(i32);
impl Tr for S { fn hi(&self) -> i32 { self.0 } }
impl Drop for S { fn drop(&mut self) { println!("dtor {}", self.0); } }
fn main() {
    let r: Rc<dyn Tr> = Rc::new(S(5));
    let r2 = r.clone();
    println!("strong {}", Rc::strong_count(&r));
    drop(r);
    println!("one left");
    drop(r2);
    println!("gone");
}
"#;

#[test]
fn run_mode_rc_dyn_trait_clone_and_drop_counts() {
    // A cloned `Rc<dyn Trait>` handle is a second live handle on
    // the same allocation: the strong count moves with the clone
    // (the handle value is in trait-object form, and the count
    // machinery sees through it), and the pointee's destructor
    // runs only when the last handle drops.
    let dir = temp_dir("rc-dyn-trait-count");
    let src = fixture(&dir, "rc_dyn_trait_count.rs", RC_DYN_TRAIT_COUNT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc<dyn Trait> count program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "strong 2\none left\ndtor 5\ngone\n",
        "the count moves with the clone and the destructor runs at the last drop"
    );
}


// ----- Zero-sized Vec elements' destructors (Lane MH) -------------------------

const ZST_VEC_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
fn main() {
    let v = vec![DropMe, DropMe];
    drop(v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_elements_drops_them() {
    // A zero-sized element's store is elided in MIR, so the Vec's
    // value holds uninitialised slots for live elements; their
    // destructors still run, once each, as natively.
    let dir = temp_dir("zstvec");
    let src = fixture(&dir, "zstvec.rs", ZST_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of zero-sized elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop\ndrop\ndone\n",
        "both elided elements' destructors run at the Vec's drop"
    );
}

const ZST_ENUM_VEC_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum Never {}
enum CS { Yielded(Never), Complete(DropMe) }
fn main() {
    let v = vec![CS::Complete(DropMe)];
    drop(v);
    let mut v2: Vec<CS> = Vec::new();
    v2.push(CS::Complete(DropMe));
    drop(v2);
    let x = CS::Complete(DropMe);
    drop(x);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_enum_elements_drops_them() {
    // The element type here is zero-sized with a single
    // inhabitant, so its values are built as constants: no
    // aggregate statement records the variant's field, yet the
    // field's destructor runs — for vec! elements, for pushed
    // elements and for a plain local alike.
    let dir = temp_dir("zstenumvec");
    let src = fixture(&dir, "zstenumvec.rs", ZST_ENUM_VEC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "zero-sized enum element program must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop\ndrop\ndrop\ndone\n",
        "the constant-built elements' and local's destructors all run"
    );
}

const CLONED_SRC: &str = r#"
fn main() {
    let v = vec![1, 2, 3];
    for x in v.iter().cloned() {
        print!("{} ", x);
    }
    println!();
    let c: Vec<i32> = v.iter().cloned().collect();
    println!("collect={c:?}");
    println!("count={}", v.iter().cloned().count());
    let mut it = v.iter().cloned();
    println!("hint={:?}", it.size_hint());
    println!("next={:?}", it.next());
    println!("next_back={:?}", it.next_back());
    let words = vec![String::from("a"), String::from("bb")];
    for s in words.iter().cloned() {
        print!("{}:{} ", s, s.len());
    }
    println!();
    let w: Vec<String> = words.iter().cloned().collect();
    println!("{w:?}");
}
"#;

#[test]
fn run_mode_cloned_iteration() {
    let dir = temp_dir("cloned-iteration");
    let src = fixture(&dir, "cloned_iteration.rs", CLONED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cloned over slice iterators must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1 2 3 \ncollect=[1, 2, 3]\ncount=3\nhint=(3, Some(3))\nnext=Some(1)\nnext_back=Some(3)\na:1 bb:2 \n[\"a\", \"bb\"]\n",
        "stderr={stderr}"
    );
}

const CLONED_HANDLES_SRC: &str = r#"
use std::rc::Rc;
fn main() {
    let v: Vec<Rc<i32>> = vec![Rc::new(1), Rc::new(2)];
    println!("before: {} {}", Rc::strong_count(&v[0]), Rc::strong_count(&v[1]));
    let w: Vec<Rc<i32>> = v.iter().cloned().collect();
    println!("after: {} {}", Rc::strong_count(&v[0]), Rc::strong_count(&v[1]));
    println!("vals: {} {}", w[0], w[1]);
    drop(w);
    println!("dropped: {}", Rc::strong_count(&v[0]));
    let boxes: Vec<Box<i32>> = vec![Box::new(10), Box::new(20)];
    let mut clones: Vec<Box<i32>> = boxes.iter().cloned().collect();
    *clones[0] = 99;
    println!("orig {} clone {}", boxes[0], clones[0]);
}
"#;

#[test]
fn run_mode_cloned_handle_elements() {
    let dir = temp_dir("cloned-handles");
    let src = fixture(&dir, "cloned_handles.rs", CLONED_HANDLES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cloned over Rc/Box elements must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "before: 1 1\nafter: 2 2\nvals: 1 2\ndropped: 1\norig 10 clone 99\n",
        "stderr={stderr}"
    );
}
const VEC_ZST_ENUM_ITER_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) {} }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
fn main() {
    let v: Vec<CS<Never, DropMe>> = vec![CS::Complete(DropMe), CS::Complete(DropMe)];
    for x in &v {
        match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    }
    let s: &[CS<Never, DropMe>] = &v;
    for x in s {
        match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    }
    for x in &v[..] {
        match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_of_zero_sized_enum_elements_read_by_iteration() {
    // The elements of a `vec!` of a zero-sized enum reach an
    // iteration snapshot as uninitialized slots (their stores
    // are elided like the index-read case); the snapshot rescue
    // denotes the element type's single value, so borrowed
    // iteration, slice iteration and range-slice iteration all
    // read the `Complete` variant, exactly as natively.
    let dir = temp_dir("veczstenumiter");
    let src = fixture(&dir, "veczstenumiter.rs", VEC_ZST_ENUM_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\nc\nc\nc\nc\nc\ndone\n",
        "zero-sized enum elements read by iteration denote their single value"
    );
}

const VEC_ZST_ENUM_NESTED_ITER_SRC: &str = r#"
#[derive(Debug, PartialEq)]
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) {} }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
fn main() {
    let v: Vec<Vec<CS<Never, DropMe>>> = vec![vec![CS::Complete(DropMe)]];
    for inner in &v {
        for x in inner {
            match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") }
        }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_nested_vec_of_zero_sized_enum_elements_read_by_iteration() {
    // The inner `Vec` reaches the outer iteration's snapshot as
    // a value, detached from any place; the rescue follows the
    // recorded element path one index step deeper, so the
    // inner iteration reads the `Complete` variant too.
    let dir = temp_dir("veczstenumnestediter");
    let src = fixture(&dir, "veczstenumnestediter.rs", VEC_ZST_ENUM_NESTED_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a nested vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\ndone\n",
        "nested zero-sized enum elements read by iteration denote their single value"
    );
}

const VEC_ZST_ENUM_STRUCT_FIELD_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
struct S { v: Vec<CS<Never, DropMe>> }
fn main() {
    let outer: Vec<S> = vec![
        S { v: vec![CS::Complete(DropMe)] },
        S { v: vec![CS::Complete(DropMe), CS::Complete(DropMe)] },
    ];
    for s in &outer {
        for x in &s.v { match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") } }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_of_struct_with_zero_sized_enum_vec_field_read_by_iteration() {
    // The inner `Vec` sits in a field of the outer snapshot's
    // elements: the recorded path crosses a field step between
    // the two index steps, and the snapshot rescue follows it,
    // so the inner iteration reads the `Complete` variant
    // exactly as natively.
    let dir = temp_dir("veczstenumstructfielditer");
    let src = fixture(&dir, "veczstenumstructfielditer.rs", VEC_ZST_ENUM_STRUCT_FIELD_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a struct field's vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\nc\nc\ndone\ndrop\ndrop\ndrop\n",
        "zero-sized enum elements behind a struct field denote their single value"
    );
}

const VEC_ZST_ENUM_TUPLE_FIELD_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
fn main() {
    let outer: Vec<(Vec<CS<Never, DropMe>>, u32)> = vec![
        (vec![CS::Complete(DropMe)], 7),
        (vec![CS::Complete(DropMe), CS::Complete(DropMe)], 8),
    ];
    for t in &outer {
        println!("n={}", t.1);
        for x in &t.0 { match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") } }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_of_tuple_with_zero_sized_enum_vec_field_read_by_iteration() {
    // The same field crossing with a tuple element: the `Vec`
    // sits under the tuple's first field step.
    let dir = temp_dir("veczstenumtuplefielditer");
    let src = fixture(&dir, "veczstenumtuplefielditer.rs", VEC_ZST_ENUM_TUPLE_FIELD_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a tuple field's vec! of zero-sized enum elements must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "n=7\nc\nn=8\nc\nc\ndone\ndrop\ndrop\ndrop\n",
        "zero-sized enum elements behind a tuple field denote their single value"
    );
}

const SPARSE_TAIL_STRUCT_FIELD_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
struct S { v: Vec<CS<Never, DropMe>> }
fn main() {
    let mut inner: Vec<CS<Never, DropMe>> = Vec::new();
    unsafe { inner.set_len(2); }
    let outer: Vec<S> = vec![S { v: inner }];
    for s in &outer {
        for x in &s.v { match x { CS::Yielded(_) => println!("y"), CS::Complete(_) => println!("c") } }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_sparse_tail_in_struct_field_read_by_nested_iteration() {
    // The inner `Vec`'s declared length runs past its stored
    // elements (a `set_len` growth): the virtual tail denotes
    // the element type's single value, and the snapshot rescue
    // materialises it behind the struct field's step, so the
    // nested iteration reads the `Complete` variant exactly as
    // natively.
    let dir = temp_dir("sparsetailstructfielditer");
    let src = fixture(&dir, "sparsetailstructfielditer.rs", SPARSE_TAIL_STRUCT_FIELD_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating a sparse vec! behind a struct field must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\nc\ndone\ndrop\ndrop\n",
        "a sparse tail behind a struct field denotes the element type's single value"
    );
}


const ZIP_FOLD_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3, 4];
    let b = [10, 20, 30, 40];
    let f = a.into_iter().zip(b.into_iter()).fold(0, |acc, (x, y)| acc + x * y);
    println!("fold {}", f);
    let s: i32 = a.into_iter().zip(b.into_iter()).map(|(x, y)| x + y).sum();
    println!("sum {}", s);
    println!("count {}", a.into_iter().zip(b.into_iter()).count());
    let mut fe = 0;
    a.into_iter().zip(b.into_iter()).for_each(|(x, y)| fe += x * y);
    println!("for_each {}", fe);
    println!("last {:?}", a.into_iter().zip(b.into_iter()).last());
    println!("vec {}", vec![1, 2, 3].into_iter().zip(vec![10, 20, 30].into_iter()).fold(0, |acc, (x, y)| acc + x * y));
    println!("uneven {}", (0..3).zip(0..10).fold(0, |acc, (x, y)| acc + x + y));
    println!("uneven count {}", (0..3).zip(0..10).count());
}
"#;

#[test]
fn run_mode_zip_fold_specialized_consumers() {
    // `fold` over a `Zip` whose sides are `TrustedRandomAccess`
    // iterators (array / `vec` into-iterators) resolves to
    // core's specialized `ZipImpl::fold`, which reads the sides
    // through `__iterator_get_unchecked` and the state's stored
    // index/length counters — machinery the modelled state does
    // not maintain, so the collected body failed (an
    // `IndexRange` type mismatch) or read uninitialized places.
    // The driver detects the specialization in the resolved
    // instance's delegation graph and lets the `Iterator::fold`
    // builtin serve the call; `sum` / `count` / `for_each` /
    // `last` reach the same point through their collected
    // default bodies, which delegate to `fold`.
    let dir = temp_dir("zip-fold");
    let src = fixture(&dir, "zip_fold.rs", ZIP_FOLD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Zip fold/sum/count/for_each over owned iterators must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "fold 300\nsum 110\ncount 4\nfor_each 300\nlast Some((4, 40))\nvec 140\nuneven 6\nuneven count 3\n",
        "the specialized consumers produce the general fold's results"
    );
}

const ZIP_FOLD_MIXED_SRC: &str = r#"
fn main() {
    let mut z = [1, 2, 3, 4].into_iter().zip([10, 20, 30, 40].into_iter());
    println!("first {:?}", z.next());
    println!("rest {}", z.fold(0, |acc, (x, y)| acc + x + y));
    let mut z2 = [1, 2, 3, 4].into_iter().zip([10, 20, 30, 40].into_iter());
    let _ = z2.next();
    println!("count {}", z2.count());
    println!("max {:?}", (0..4).zip(10..14).max());
    println!("reduce {:?}", (0..4).zip(10..14).reduce(|p, q| if q.0 > p.0 { q } else { p }));
}
"#;

#[test]
fn run_mode_zip_fold_after_next() {
    // Mixed consumption: a modelled `next` step followed by a
    // specialized consumer. The builtin `next` leaves the Zip
    // state's specialization counters untouched, so a collected
    // specialized `fold` afterwards reads a stale length (and,
    // over `Range` sides, fabricated the one-past-the-end pair
    // in `max` / `reduce`). Serving `fold` through the builtin
    // keeps one state discipline for the whole consumption.
    let dir = temp_dir("zip-fold-mixed");
    let src = fixture(&dir, "zip_fold_mixed.rs", ZIP_FOLD_MIXED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Zip next-then-fold must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "first Some((1, 10))\nrest 99\ncount 3\nmax Some((3, 13))\nreduce Some((3, 13))\n",
        "consumers after a modelled next see exactly the remaining items"
    );
}

const ZST_STRUCT_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
struct S { d: DropMe }
fn main() {
    let v = vec![S { d: DropMe }, S { d: DropMe }];
    for s in v {
        let _ = s;
        println!("c");
    }
    println!("done");
}
"#;

#[test]
fn run_mode_by_value_iteration_drops_zst_struct_elements() {
    // The loop variable of a by-value iteration over zero-sized
    // struct elements is never written — the payload move out of
    // the iterator is a no-op the MIR pipeline deletes — yet each
    // element's field destructors run at the end of its iteration,
    // once each, as natively.
    let dir = temp_dir("zststructiter");
    let src = fixture(&dir, "zststructiter.rs", ZST_STRUCT_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "by-value iteration over zero-sized structs must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\ndrop\nc\ndrop\ndone\n",
        "each iterated element's field destructor runs"
    );
}

const ZST_ENUM_MATCH_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum Never {}
enum CS { Yielded(Never), Complete(DropMe) }
fn main() {
    let v = vec![CS::Complete(DropMe), CS::Complete(DropMe)];
    let mut i = 0;
    for x in v {
        match x {
            CS::Yielded(_) => println!("y"),
            CS::Complete(_) => println!("{i}"),
        }
        i += 1;
    }
    println!("done");
}
"#;

#[test]
fn run_mode_match_consumed_zst_enum_elements_drop() {
    // An enum element consumed by a match in a by-value iteration
    // still drops its payload when the scrutinee goes out of scope —
    // for the second element exactly as for the first, even though
    // the scrutinee local's write was elided as a no-op.
    let dir = temp_dir("zstenummatchiter");
    let src = fixture(&dir, "zstenummatchiter.rs", ZST_ENUM_MATCH_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "match-consumed zero-sized enum iteration must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "0\ndrop\n1\ndrop\ndone\n",
        "both matched elements' payloads drop"
    );
}

const ZST_NESTED_ITER_SRC: &str = r#"
struct DropMe;
impl Drop for DropMe { fn drop(&mut self) { println!("drop"); } }
enum CS<Y, R> { Yielded(Y), Complete(R) }
enum Never {}
struct S { v: Vec<CS<Never, DropMe>> }
fn main() {
    let outer: Vec<S> = vec![
        S { v: vec![CS::Complete(DropMe)] },
        S { v: vec![CS::Complete(DropMe), CS::Complete(DropMe)] },
    ];
    for s in outer {
        for x in s.v {
            match x {
                CS::Yielded(_) => println!("y"),
                CS::Complete(_) => println!("c"),
            }
        }
    }
    println!("done");
}
"#;

#[test]
fn run_mode_nested_by_value_iteration_drops_zst_payloads() {
    // Nested by-value iteration whose elements are structs holding
    // Vecs of zero-sized enums: every element's payload destructor
    // runs in iteration order, as natively.
    let dir = temp_dir("zstnestediter");
    let src = fixture(&dir, "zstnestediter.rs", ZST_NESTED_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested by-value iteration must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "c\ndrop\nc\ndrop\nc\ndrop\ndone\n",
        "every nested element's payload destructor runs in order"
    );
}

const SINGLE_ARRAY_FIELD_PUN_SRC: &str = r#"
struct S([u8; 4]);
struct W([u32; 2]);
fn main() {
    let a: [u8; 4] = [11, 22, 33, 44];
    let r: &S = unsafe { std::mem::transmute(&a) };
    println!("ref {} {} {} {}", r.0[0], r.0[1], r.0[2], r.0[3]);
    let f: [u8; 4] = r.0;
    println!("field {} {} {} {}", f[0], f[1], f[2], f[3]);
    let w: [u32; 2] = [0x11223344, 0x55667788];
    let p = &w as *const [u32; 2] as *const W;
    let w2 = unsafe { &*p };
    println!("ptr {:x} {:x}", w2.0[0], w2.0[1]);
    let mut b: [u8; 4] = [0, 0, 0, 0];
    let m: &mut S = unsafe { std::mem::transmute(&mut b) };
    m.0[2] = 77;
    println!("mut {} {} {} {}", b[0], b[1], b[2], b[3]);
    m.0 = [9, 8, 7, 6];
    println!("whole {} {} {} {}", b[0], b[1], b[2], b[3]);
    let bx: Box<[u8; 4]> = Box::new([5, 6, 7, 8]);
    let bs: Box<S> = unsafe { std::mem::transmute(bx) };
    println!("box {} {} {} {}", bs.0[0], bs.0[1], bs.0[2], bs.0[3]);
    let t = ([11u8, 22, 33, 44],);
    let tr: &[u8; 4] = unsafe { std::mem::transmute(&t) };
    println!("tup {} {} {} {}", tr[0], tr[1], tr[2], tr[3]);
    let br: &([u8; 4],) = unsafe { std::mem::transmute(&a) };
    println!("back {} {} {} {}", br.0[0], br.0[1], br.0[2], br.0[3]);
}
"#;

#[test]
fn run_mode_single_array_field_pun_reference_views() {
    // A struct (or one-element tuple) whose only field is an
    // array is layout-identical to the array, and a reference,
    // raw pointer or box transmuted or cast between the two
    // views denotes the same storage: the storage keeps the
    // form it was written with (the bare array), so reading
    // the wrapper's field through the other view must read
    // the array itself -- its elements at their indices, the
    // whole field as the whole array -- not the first element
    // and that element's bytes. Writes through the wrapper
    // view land in the array's elements.
    run_ok(
        "arrayfieldpun",
        SINGLE_ARRAY_FIELD_PUN_SRC,
        "ref 11 22 33 44\nfield 11 22 33 44\nptr 11223344 55667788\nmut 0 0 77 0\nwhole 9 8 7 6\nbox 5 6 7 8\ntup 11 22 33 44\nback 11 22 33 44\n",
    );
}

const ONE_ELEMENT_TUPLE_TRANSMUTE_SRC: &str = r#"
fn main() {
    let t = ([3u8, 1, 4, 1],);
    let a: [u8; 4] = unsafe { std::mem::transmute(t) };
    println!("out {} {} {} {}", a[0], a[1], a[2], a[3]);
    let b: ([u8; 4],) = unsafe { std::mem::transmute([9u8, 9, 9, 9]) };
    println!("in {} {} {} {}", b.0[0], b.0[1], b.0[2], b.0[3]);
    let s = (42u32,);
    let x: u32 = unsafe { std::mem::transmute(s) };
    println!("scalar {}", x);
    let s2: (u32,) = unsafe { std::mem::transmute(7u32) };
    println!("wrapped {}", s2.0);
}
"#;

#[test]
fn run_mode_one_element_tuple_transmute_crossings() {
    // A one-element tuple is a one-field wrapper, with the
    // same transmute crossings the one-field struct has: to
    // exactly the field's type it reads the field out, and
    // from exactly the field's type it wraps the value into
    // the tuple.
    run_ok(
        "onetupletransmute",
        ONE_ELEMENT_TUPLE_TRANSMUTE_SRC,
        "out 3 1 4 1\nin 9 9 9 9\nscalar 42\nwrapped 7\n",
    );
}

const SPARSE_NESTED_ARRAY_SRC: &str = r#"
use std::alloc::{alloc, Layout};

fn main() {
    unsafe {
        let layout = Layout::new::<[[u32; 2]; 2]>();
        let p = alloc(layout) as *mut [[u32; 2]; 2];
        (*p)[0] = [11, 22];
        println!("row {:?}", (*p)[0]);
        println!("elem {}", (*p)[0][1]);
        let q = p as *const [u32; 2];
        println!("via_row_ptr {}", (*q)[1]);
        (*p)[0][1] = 99;
        println!("written {:?}", (*p)[0]);
        println!("via_row_ptr_after {}", (*q)[1]);
    }
}
"#;

#[test]
fn run_mode_sparse_nested_array_prefix_is_not_a_tuple_wrapper() {
    // A nested array whose first row alone is materialised
    // stores as a one-item tuple holding the row -- the same
    // shape a one-element tuple wrapper around the whole array
    // has. The dereference of the array pointer must keep the
    // genuine reading (the tuple is the array's own sparse
    // prefix): reading the row reads the row, indexing through
    // it reaches the row's elements, and writes through the
    // array view land in the row the row view reads back.
    run_ok(
        "sparsenestedarray",
        SPARSE_NESTED_ARRAY_SRC,
        "row [11, 22]\nelem 22\nvia_row_ptr 22\nwritten [11, 99]\nvia_row_ptr_after 99\n",
    );
}

const ZST_RAW_SLICE_ITER_SRC: &str = r#"
use std::slice;

fn foo<T>(v: &[T]) -> Option<&[T]> {
    let mut it = v.iter();
    for _ in 0..5 {
        let _ = it.next();
    }
    Some(it.as_slice())
}

fn foo_mut<T>(v: &mut [T]) -> Option<&mut [T]> {
    let mut it = v.iter_mut();
    for _ in 0..5 {
        let _ = it.next();
    }
    Some(it.into_slice())
}

fn main() {
    let slice: &[()] = unsafe { slice::from_raw_parts(-5isize as *const (), 10) };
    println!("len {}", slice.len());
    println!("count {}", slice.iter().count());
    let mut it = slice.iter();
    println!("nth some {}", it.nth(5).is_some());
    println!("rest {}", it.count());
    println!("foo some {}", foo(slice).is_some());
    let slice: &mut [()] = unsafe { slice::from_raw_parts_mut(-5isize as *mut (), 10) };
    println!("mcount {}", slice.iter_mut().count());
    {
        let mut it = slice.iter_mut();
        println!("mnth some {}", it.nth(5).is_some());
        println!("mrest {}", it.count());
    }
    println!("foo_mut some {}", foo_mut(slice).is_some());
}
"#;

#[test]
fn run_mode_zst_raw_slice_iterators() {
    // A slice of zero-sized elements over a raw-parts pointer
    // at the edge of the address space: the iterator's end
    // pointer carries the remaining count in its own bits, a
    // representation the collected core bodies read through
    // pointer-to-integer puns the value model does not
    // reproduce. Iteration is served by the modelled `next`
    // state machine (whose address-pair step synthesises the
    // elements), and the consumers (`count`, `nth`) are served
    // by the same machine so the accounting cannot disagree.
    // This program used to spin forever in `slice_elements`
    // (the fat pointer over the raw address resolved to
    // itself).
    run_ok(
        "zst-raw-slice-iterators",
        ZST_RAW_SLICE_ITER_SRC,
        "len 10\ncount 10\nnth some true\nrest 4\nfoo some true\nmcount 10\nmnth some true\nmrest 4\nfoo_mut some true\n",
    );
}

const ZST_ARRAY_ITER_NTH_SRC: &str = r#"
fn main() {
    let a = [(), (), (), (), (), (), (), (), (), ()];
    println!("count {}", a.iter().count());
    let mut it = a.iter();
    println!("nth some {}", it.nth(5).is_some());
    println!("rest {}", it.count());
    let mut m = [(), (), (), ()];
    println!("mcount {}", m.iter_mut().count());
}
"#;

#[test]
fn run_mode_zst_array_iter_nth_count() {
    // `nth` on a zero-sized snapshot iterator: core's body
    // reads the end pointer's bits as the remaining count
    // through a `*mut usize` pun, which the value model cannot
    // serve; the modelled `nth` drives the `next` transition
    // instead.
    run_ok(
        "zst-array-iter-nth-count",
        ZST_ARRAY_ITER_NTH_SRC,
        "count 10\nnth some true\nrest 4\nmcount 4\n",
    );
}

const NTH_SKIPPED_DROP_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};

static DROPS: AtomicUsize = AtomicUsize::new(0);

struct Frag(u32);
impl Drop for Frag {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    let mut it = vec![Frag(1), Frag(2), Frag(3)].into_iter();
    let second = it.nth(1);
    println!("got {}", second.unwrap().0);
    println!("drops after nth {}", DROPS.load(Ordering::SeqCst));
    drop(it);
    println!("drops after drop(it) {}", DROPS.load(Ordering::SeqCst));
}
"#;

#[test]
fn run_mode_iterator_nth_drops_skipped_items() {
    // The items `nth` skips are consumed by the skip: the
    // modelled `nth` runs each skipped item's drop glue
    // (baked by conversion from the item type) before
    // stepping on, exactly as core's default body drops the
    // temporary each `advance_by` iteration yields. Before
    // the glue ran, the counter read 1 after `nth` (only the
    // returned item's temporary) and 2 at the end: the
    // skipped element's destructor never ran, and its slot
    // in the iterator was already marked consumed, so the
    // iterator's own drop could not recover it.
    run_ok(
        "iterator-nth-drops-skipped-items",
        NTH_SKIPPED_DROP_SRC,
        "got 2\ndrops after nth 2\ndrops after drop(it) 3\n",
    );
}

const NTH_RANGE_ADVANCE_SRC: &str = r#"
fn main() {
    let mut r = 0usize..2_000_000usize;
    println!("nth {:?}", r.nth(1_500_000));
    let mut ri = 0usize..=2_000_000usize;
    println!("incl nth {:?}", ri.nth(1_500_000));
}
"#;

#[test]
fn run_mode_iterator_nth_range_advances_directly() {
    // An arithmetic range's `nth` advances in one step, as
    // the specialised body in core does: driving the
    // modelled `next` transition once per skipped item
    // spends a step per item and a 1.5M skip exceeds the
    // default step limit before it can answer.
    run_ok(
        "iterator-nth-range-advances-directly",
        NTH_RANGE_ADVANCE_SRC,
        "nth Some(1500000)\nincl nth Some(1500000)\n",
    );
}

const LONG_WHILE_SRC: &str = r#"
#![allow(unused_variables)]
fn main() {
    let mut i: isize = 0;
    while i < 300_000 {
        i += 1;
        let x = 3;
    }
    println!("done {i}");
}
"#;

#[test]
fn run_mode_long_loop_within_step_budget() {
    // A plain counting loop spends ~7 steps per iteration in
    // the interpreter, so 300k iterations (~2.1M steps) cross
    // the old 1,000,000 default step budget even though the
    // program is trivially finite -- the suite's own
    // tests/ui/for-loop-while/long-while.rs needs 7,000,006
    // steps for its 1M iterations. The default budget is now
    // sized from measured demand (issue #399), and the run
    // completes.
    run_ok("step-budget-long-while", LONG_WHILE_SRC, "done 300000\n");
}

const NTH_CHAR_RANGE_GAP_SRC: &str = r#"
fn show(tag: &str, v: Option<char>) {
    match v {
        Some(c) => println!("{} Some({})", tag, c as u32),
        None => println!("{} None", tag),
    }
}

fn main() {
    // The Codex repro: skipping one item of '\u{d7ff}'..'\u{e001}'
    // must land on '\u{e000}', not on the surrogate 0xD800.
    show("codex", ('\u{d7ff}'..'\u{e001}').nth(1));
    show("nth0", ('\u{d7ff}'..'\u{e001}').nth(0));
    // Exactly the range length (2) and beyond it: no element.
    show("nth-len", ('\u{d7ff}'..'\u{e001}').nth(2));
    show("nth-over", ('\u{d7ff}'..'\u{e001}').nth(3));
    // A range whose raw span crosses the whole gap but holds
    // one element: '\u{d7ff}'..'\u{e000}'.
    show("gap1-nth0", ('\u{d7ff}'..'\u{e000}').nth(0));
    show("gap1-nth1", ('\u{d7ff}'..'\u{e000}').nth(1));
    // Crossing the gap with a wider skip.
    show("wide", ('\u{d7f0}'..'\u{e010}').nth(17));
    // Starting above the gap.
    show("above", ('\u{e000}'..'\u{e003}').nth(1));
    // Inclusive ranges across the gap.
    show("incl-nth1", ('\u{d7ff}'..='\u{e000}').nth(1));
    show("incl-nth-len", ('\u{d7ff}'..='\u{e000}').nth(2));
    show("incl-single", ('\u{d7ff}'..='\u{d7ff}').nth(0));
    show("incl-single-over", ('\u{d7ff}'..='\u{d7ff}').nth(1));
    // An inclusive range exhausted by a skip stays exhausted
    // from both ends.
    let mut ri = '\u{d7ff}'..='\u{e000}';
    show("incl-exh-nth", ri.nth(5));
    show("incl-exh-next", ri.next());
    show("incl-exh-back", ri.next_back());
    // A skip that lands just below the gap: the item `nth`
    // returns is '\u{d7ff}', and the following step crosses
    // the gap.
    let mut r = '\u{d7fe}'..'\u{e001}';
    show("land-below", r.nth(1));
    show("after-nth", r.next());
    show("after-nth2", r.next());
    // A skip past the end leaves an exclusive range exhausted.
    let mut ro = 'a'..'c';
    show("over-nth", ro.nth(2));
    show("over-next", ro.next());
    // Plain char ranges are unchanged.
    show("normal", ('a'..'f').nth(3));
    show("normal-incl", ('a'..='f').nth(5));
    show("normal-incl-over", ('a'..='f').nth(6));
    // Forward and backward per-item steps cross the gap too.
    let fwd: Vec<u32> = ('\u{d7ff}'..'\u{e001}').map(|c| c as u32).collect();
    println!("fwd {:?}", fwd);
    let back: Vec<u32> = ('\u{d7ff}'..'\u{e001}').rev().map(|c| c as u32).collect();
    println!("back {:?}", back);
    let incl: Vec<u32> = ('\u{d7fe}'..='\u{e000}').map(|c| c as u32).collect();
    println!("incl {:?}", incl);
    // A large char skip still advances in one step.
    show("big", ('\0'..='\u{10ffff}').nth(1_000_000));
}
"#;

#[test]
fn run_mode_iterator_nth_char_range_surrogate_gap() {
    // `nth` on a `Range<char>` / `RangeInclusive<char>`:
    // the one-step advance and the range's length must both
    // skip the surrogate gap [0xD800, 0xE000), as
    // `range_count`'s length already did — the raw
    // code-point arithmetic returned the invalid char
    // 0xD800 for ('\u{d7ff}'..'\u{e001}').nth(1) and
    // served elements past the end of gap-crossing ranges
    // whose raw span overstates their length. The per-item
    // steps share the same arithmetic, so the sequences
    // after a skip (and plain collect / rev) cross the gap
    // the same way.
    run_ok(
        "iterator-nth-char-range-surrogate-gap",
        NTH_CHAR_RANGE_GAP_SRC,
        "codex Some(57344)\nnth0 Some(55295)\nnth-len None\nnth-over None\ngap1-nth0 Some(55295)\ngap1-nth1 None\nwide Some(57345)\nabove Some(57345)\nincl-nth1 Some(57344)\nincl-nth-len None\nincl-single Some(55295)\nincl-single-over None\nincl-exh-nth None\nincl-exh-next None\nincl-exh-back None\nland-below Some(55295)\nafter-nth Some(57344)\nafter-nth2 None\nover-nth None\nover-next None\nnormal Some(100)\nnormal-incl Some(102)\nnormal-incl-over None\nfwd [55295, 57344]\nback [57344, 55295]\nincl [55294, 55295, 57344]\nbig Some(1002048)\n",
    );
}

const ITER_MUT_RAW_PARTS_SRC: &str = r#"
use std::slice;

fn main() {
    let mut v = vec![10, 20, 30, 40];
    let ptr = v.as_mut_ptr();
    let s: &mut [i32] = unsafe { slice::from_raw_parts_mut(ptr, 4) };
    let mut it = s.iter_mut();
    println!("first {:?}", it.next());
    let out = it.into_slice();
    println!("rest {:?}", out);
}
"#;

#[test]
fn run_mode_slice_iter_mut_raw_parts_next() {
    // `next` on the real field layout (ptr, end, marker) the
    // collected `<[T]>::iter_mut` body builds over a raw-parts
    // slice: the state machine's `IterMut` arm used to serve
    // only the [container, index, length] state `IntoIterator`
    // builds, and reported this shape as a type mismatch.
    run_ok(
        "slice-iter-mut-raw-parts-next",
        ITER_MUT_RAW_PARTS_SRC,
        "first Some(10)\nrest [20, 30, 40]\n",
    );
}

const COMPARE_BYTES_SIGNED_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::compare_bytes;

fn main() {
    let a: [i8; 4] = [1, -128, 3, 0];
    let b: [i8; 4] = [1, 127, 3, 0];
    let c: [i8; 4] = [1, -128, 3, 0];
    let d: [i8; 4] = [1, -128, 2, 0];
    unsafe {
        let gt = compare_bytes(a.as_ptr().cast(), b.as_ptr().cast(), 4);
        let eq = compare_bytes(a.as_ptr().cast(), c.as_ptr().cast(), 4);
        let lt = compare_bytes(b.as_ptr().cast(), a.as_ptr().cast(), 4);
        let tail = compare_bytes(a.as_ptr().cast(), d.as_ptr().cast(), 4);
        println!("gt={} eq={} lt={} tail={}", gt > 0, eq == 0, lt < 0, tail > 0);
    }
}
"#;

#[test]
fn run_mode_compare_bytes_signed_elements() {
    // `compare_bytes` over signed-element storage: an `i8`
    // element holding a byte above 127 keeps its bit pattern
    // (0x80 compares above 0x7f), instead of being rejected
    // as a type mismatch -- the same acceptance `read_c_str`
    // already gives the libc string scans.
    let dir = temp_dir("compare-bytes-signed");
    let src = fixture(&dir, "compare_bytes_signed.rs", COMPARE_BYTES_SIGNED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "compare_bytes over signed bytes must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "gt=true eq=true lt=true tail=true\n",
        "unsigned byte order over the signed elements' bit patterns"
    );
}

const ARC_HOLDER_SRC: &str = r#"
use std::sync::Arc;
struct Holder(Arc<u32>);
fn main() {
    let a = Arc::new(7u32);
    let h = Holder(a.clone());
    println!("during {}", Arc::strong_count(&a));
    drop(h);
    println!("after {}", Arc::strong_count(&a));
}
"#;

#[test]
fn run_mode_arc_handle_in_struct_field_still_releases() {
    let dir = temp_dir("arc-holder-drop");
    let src = fixture(&dir, "arc_holder.rs", ARC_HOLDER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping a struct holding a real Arc must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "during 2\nafter 1\n",
        "a real handle aggregate in a struct field is still released"
    );
}
const VEC_CAPACITY_GROWTH_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::new();
    for i in 0..5 {
        v.push(i);
    }
    println!("push5 cap {} spare {}", v.capacity(), v.spare_capacity_mut().len());
    let mut b: Vec<u8> = Vec::new();
    b.push(1);
    println!("u8 push1 cap {}", b.capacity());
    let mut w: Vec<i32> = Vec::with_capacity(1);
    w.push(1);
    w.push(2);
    println!("withcap1 push2 cap {}", w.capacity());
    let mut r = vec![1, 2, 3];
    r.reserve(1);
    println!("reserve cap {}", r.capacity());
    let mut e = vec![1, 2, 3];
    e.reserve_exact(1);
    println!("reserve_exact cap {}", e.capacity());
    let mut t = vec![1, 2, 3];
    println!("try_reserve ok {}", t.try_reserve(20).is_ok());
    println!("try_reserve cap {}", t.capacity());
}
"#;

#[test]
fn run_mode_vec_capacity_growth_matches_rawvec() {
    // Capacity is the storage extent in this model, and growth
    // follows RawVec's rules exactly: an append that outgrows
    // the storage doubles it (floored at the minimum non-zero
    // capacity for the element size -- 8 for bytes, 4 for
    // 4-byte elements), `reserve` grows amortized,
    // `reserve_exact` to the exact requirement, and a
    // successful `try_reserve` grows like `reserve`. The
    // answers are observable through `capacity()` and
    // `spare_capacity_mut().len()`, as natively.
    let dir = temp_dir("veccapacitygrowth");
    let src = fixture(&dir, "veccapacitygrowth.rs", VEC_CAPACITY_GROWTH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec capacity growth must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "push5 cap 8 spare 3\nu8 push1 cap 8\nwithcap1 push2 cap 4\nreserve cap 6\nreserve_exact cap 4\ntry_reserve ok true\ntry_reserve cap 23\n"
    );
}

const VEC_SPARE_WRITE_SRC: &str = r#"
fn main() {
    let mut v: Vec<i32> = Vec::with_capacity(8);
    v.push(10);
    v.push(20);
    let spare = v.spare_capacity_mut();
    println!("spare {}", spare.len());
    spare[0].write(30);
    spare[1].write(40);
    unsafe { v.set_len(4) };
    println!("v = {:?}", v);
    println!("idx {}", v[2]);
    println!("pop {:?}", v.pop());
}
"#;

#[test]
fn run_mode_vec_spare_capacity_write_then_set_len_readback() {
    // Elements written through the spare-capacity slice are
    // stored as `MaybeUninit` payload slots; once `set_len`
    // declares them, every read of them -- indexing, the
    // vector's own Debug list, `pop` -- sees the payload, as
    // natively. (Formatting a spare-written element goes
    // through the fmt engine's storage-wrapper unwrap.)
    let dir = temp_dir("vecsparewrite");
    let src = fixture(&dir, "vecsparewrite.rs", VEC_SPARE_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "spare-capacity write + set_len readback must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "spare 6\nv = [10, 20, 30, 40]\nidx 30\npop Some(40)\n"
    );
}

const VEC_ZST_CAPACITY_SRC: &str = r#"
fn main() {
    let mut v: Vec<()> = Vec::with_capacity(10);
    println!("cap {}", v.capacity());
    println!("spare {}", v.spare_capacity_mut().len());
    v.push(());
    println!("cap {}", v.capacity());
}
"#;

#[test]
fn run_mode_vec_zero_sized_element_capacity() {
    // A zero-sized element's allocation is the dangling
    // pointer: std's capacity for it is `usize::MAX` by
    // definition, not the storage extent the model holds.
    let dir = temp_dir("veczstcapacity");
    let src = fixture(&dir, "veczstcapacity.rs", VEC_ZST_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ZST Vec capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "cap 18446744073709551615\nspare 18446744073709551615\ncap 18446744073709551615\n"
    );
}

const VEC_SPARSE_ORDERING_SRC: &str = r#"
use std::cmp::Ordering;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Wrap(Vec<i32>);

fn main() {
    let mut v: Vec<i32> = Vec::new();
    v.push(1);
    println!("self le {} cmp {:?}", v <= v, v.cmp(&v));
    println!("partial {:?}", v.partial_cmp(&v));

    let mut a: Vec<i32> = Vec::new();
    a.push(1);
    a.push(3);
    let mut b: Vec<i32> = Vec::new();
    b.push(1);
    b.push(2);
    b.push(9);
    println!("a<b {} a>b {} a==b {}", a < b, a > b, a == b);
    println!("a cmp {:?}", a.cmp(&b));

    let lit = vec![1, 3];
    println!("a==lit {} a<=lit {} a>=lit {}", a == lit, a <= lit, a >= lit);
    let lit2 = vec![1, 4];
    println!("a<lit2 {} a cmp lit2 {:?}", a < lit2, a.cmp(&lit2));

    let mut short: Vec<i32> = Vec::new();
    short.push(1);
    println!("short<a {} short cmp a {:?}", short < a, short.cmp(&a));

    let w1 = Wrap(a.clone());
    let w2 = Wrap(vec![1, 3]);
    println!("wrap cmp {:?}", w1.cmp(&w2));
    let _ = Ordering::Equal;
}
"#;

#[test]
fn run_mode_vec_sparse_ordering_after_growth() {
    // A push that leaves spare capacity puts the vector in
    // its sparse form (declared length below the stored
    // elements); ordering -- `PartialOrd`, `cmp`,
    // `partial_cmp`, and derived `Ord` over a field holding
    // such a vector -- compares the declared elements only,
    // exactly as equality does, against sparse and plain
    // forms alike. The spare storage past the length is not
    // part of the value and never enters the comparison.
    let dir = temp_dir("vecsparseord");
    let src = fixture(&dir, "vecsparseord.rs", VEC_SPARSE_ORDERING_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "ordering of spare-capacity vectors must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "self le true cmp Equal\npartial Some(Equal)\na<b false a>b true a==b false\na cmp Greater\na==lit true a<=lit true a>=lit true\na<lit2 true a cmp lit2 Less\nshort<a true short cmp a Less\nwrap cmp Equal\n"
    );
}

const VEC_SPARSE_CONSUMERS_SRC: &str = r#"
use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let mut v = Vec::new();
    v.push(10);
    v.push(20);
    for x in v {
        println!("elem {x}");
    }

    let mut v = Vec::new();
    v.push(10);
    v.push(20);
    let r: Rc<[i32]> = v.into();
    println!("rc {:?} len {}", r, r.len());
    let mut w = Vec::new();
    w.push(7);
    let a: Arc<[i32]> = w.into();
    println!("arc {:?} len {}", a, a.len());

    let mut v = Vec::new();
    v.push(10);
    v.push(20);
    let mut d: VecDeque<i32> = v.into();
    println!("deque len {}", d.len());
    println!("front {:?} back {:?}", d.front(), d.back());
    println!("pop {:?}", d.pop_front());

    let mut bytes: Vec<u8> = Vec::new();
    bytes.push(65u8);
    write!(bytes, "BC").unwrap();
    println!("written {:?}", String::from_utf8(bytes).unwrap());

    let mut paths = Vec::new();
    paths.push(PathBuf::from("/a"));
    paths.push(PathBuf::from("/b"));
    let joined = std::env::join_paths(paths).unwrap();
    println!("joined {}", joined.to_string_lossy());
}
"#;

#[test]
fn run_mode_vec_sparse_consumers_after_growth() {
    // Consumers of a vector the growth path left in its
    // sparse form take its declared elements, exactly as
    // equality and ordering do: owning iteration, the
    // counted-slice conversions (`Rc<[T]>` / `Arc<[T]>`),
    // the `VecDeque` conversion, a formatted write into a
    // byte vector, and `join_paths` over a by-value vector.
    // The spare storage past the length is not part of the
    // value and never reaches any of them.
    let dir = temp_dir("vecsparseuse");
    let src = fixture(&dir, "vecsparseuse.rs", VEC_SPARSE_CONSUMERS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "consumers of spare-capacity vectors must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "elem 10\nelem 20\nrc [10, 20] len 2\narc [7] len 1\ndeque len 2\nfront Some(10) back Some(20)\npop Some(10)\nwritten \"ABC\"\njoined /a:/b\n"
    );
}

const OSSTRING_AS_OS_STR_SRC: &str = r#"
use std::ffi::OsString;
fn main() {
    let s = OsString::from("a/b");
    println!("len {}", s.as_os_str().len());
    let b = &s[..];
    println!("sub {}", b.to_str().unwrap());
    println!("enc {}", s.as_os_str().as_encoded_bytes().len());
    println!("done");
}
"#;

#[test]
fn run_mode_osstring_as_os_str_and_index() {
    run_ok(
        "osstring-as-os-str",
        OSSTRING_AS_OS_STR_SRC,
        "len 3\nsub a/b\nenc 3\ndone\n",
    );
}

const VEC_AS_PTR_WRAPPER_SRC: &str = r#"
struct Holder {
    v: Vec<u32>,
}
fn main() {
    let h = Holder { v: vec![10, 20, 30] };
    let p = h.v.as_ptr();
    unsafe {
        println!("{} {} {}", *p, *p.add(1), *p.add(2));
    }
    let v = vec![1u8, 2, 3];
    let q = v.as_ptr();
    unsafe {
        println!("first {}", *q);
    }
    println!("done");
}
"#;

#[test]
fn run_mode_vec_as_ptr_through_wrapper() {
    run_ok(
        "vec-as-ptr-wrapper",
        VEC_AS_PTR_WRAPPER_SRC,
        "10 20 30\nfirst 1\ndone\n",
    );
}

const VEC_DEDUP_LEN_WRITE_SRC: &str = r#"
fn main() {
    let mut v = vec![1u8, 1, 2, 2, 3];
    v.dedup();
    println!("{:?}", v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_dedup_len_slot_write() {
    run_ok("vec-dedup", VEC_DEDUP_LEN_WRITE_SRC, "[1, 2, 3]\ndone\n");
}

const VEC_TRUNCATE_LEN_WRITE_SRC: &str = r#"
fn main() {
    let mut v = vec![1u8, 1, 2, 2, 3];
    v.truncate(3);
    println!("{:?}", v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_truncate_len_slot_write() {
    run_ok(
        "vec-truncate",
        VEC_TRUNCATE_LEN_WRITE_SRC,
        "[1, 1, 2]\ndone\n",
    );
}


const FILE_MODEL_SRC: &str = r#"
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

fn main() {
    let p = "DATA_PATH";
    {
        let mut f = File::create(p).unwrap();
        f.write_all(b"hello").unwrap();
        writeln!(f, " {}", 42).unwrap();
        f.sync_all().unwrap();
    }
    let mut f = OpenOptions::new().read(true).write(true).open(p).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    println!("content={s:?}");
    f.seek(SeekFrom::Start(6)).unwrap();
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf).unwrap();
    println!("at6={:?}", &buf);
    let mut g = f.try_clone().unwrap();
    println!("clone pos={}", g.stream_position().unwrap());
    println!("len={}", f.metadata().unwrap().len());
    // Open/drop in a loop: the handle's drop plan must close
    // the host file every iteration, or the loop exhausts the
    // process's descriptor budget long before it finishes.
    let mut total = 0usize;
    for _ in 0..1500 {
        let mut h = File::open(p).unwrap();
        let mut b = [0u8; 1];
        h.read_exact(&mut b).unwrap();
        total += b[0] as usize;
    }
    println!("loop total={total}");
}
"#;

#[test]
fn run_mode_file_handle_value_model() {
    let dir = temp_dir("file-model");
    let data = dir.join("data.txt");
    let src = FILE_MODEL_SRC.replace("DATA_PATH", &data.to_string_lossy());
    let src_path = fixture(&dir, "file_model.rs", &src);
    let out = run_driver(&src_path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "file model case must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "content=\"hello 42\\n\"\nat6=[52, 50]\nclone pos=8\nlen=9\nloop total=156000\n",
        "file model stdout mismatch; stderr={stderr}"
    );
}


const BOX_NEW_IN_ZST_ALLOC_SRC: &str = r#"
use std::alloc::{AllocError, Allocator, Global, Layout};
use std::cell::Cell;
use std::ptr::NonNull;

struct DropCheckingAllocator<'a>(&'a Cell<bool>);

unsafe impl Allocator for DropCheckingAllocator<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(layout)
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        Global.deallocate(ptr, layout);
    }
}
impl Drop for DropCheckingAllocator<'_> {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

struct HasDrop;
impl Drop for HasDrop {
    fn drop(&mut self) {}
}

fn main() {
    // A zero-sized box value allocated through a custom
    // allocator: the allocator serves the zero-sized layout
    // with the dangling pointer, and writing the value
    // through it denotes no storage (the shape
    // tests/ui/drop/box-conditional-drop-allocator.rs
    // exercises).
    let dropped = Cell::new(false);
    {
        let b = Box::new_in(HasDrop, DropCheckingAllocator(&dropped));
        if true {
            drop(*b);
        } else {
            drop(b);
        }
    }
    assert!(dropped.get());
    println!("box-new-in-zst-ok");
}
"#;

#[test]
fn run_mode_box_new_in_zst_custom_allocator() {
    run_ok("box-new-in-zst", BOX_NEW_IN_ZST_ALLOC_SRC, "box-new-in-zst-ok\n");
}
const NEEDS_DROP_UNION_SRC: &str = r#"
use std::mem::{needs_drop, ManuallyDrop};

struct NeedDrop;
impl Drop for NeedDrop {
    fn drop(&mut self) {}
}

union UnionOk1<T> {
    empty: (),
    value: ManuallyDrop<T>,
}

union ActuallyDrop<T> {
    inner: ManuallyDrop<T>,
}
impl<T> Drop for ActuallyDrop<T> {
    fn drop(&mut self) {}
}

fn main() {
    // A union's fields never make it need drop; only its own
    // `Drop` impl does (the shape
    // tests/ui/union/union-manuallydrop-rpass.rs and
    // tests/ui/union/union-nodrop.rs exercise).
    assert!(!needs_drop::<UnionOk1<NeedDrop>>());
    assert!(needs_drop::<ActuallyDrop<u8>>());
    assert!(needs_drop::<ActuallyDrop<Box<u8>>>());
    println!("needs-drop-union-ok");
}
"#;

#[test]
fn run_mode_needs_drop_union_verdicts() {
    run_ok("needs-drop-union", NEEDS_DROP_UNION_SRC, "needs-drop-union-ok\n");
}

const NEEDS_DROP_GUEST_SRC: &str = r#"
use std::mem::{needs_drop, ManuallyDrop};

struct NeedDrop;
impl Drop for NeedDrop {
    fn drop(&mut self) {}
}

union NoDrop<T> {
    inner: ManuallyDrop<T>,
}

struct Foo {
    x: NoDrop<Box<u8>>,
}

struct Baz {
    x: NoDrop<Box<u8>>,
    y: Box<u8>,
}

enum Choice {
    Plain(u8),
    Owned(Box<u8>),
}

fn main() {
    assert!(!needs_drop::<Foo>());
    assert!(!needs_drop::<NoDrop<Box<u8>>>());
    assert!(needs_drop::<Baz>());
    assert!(needs_drop::<Choice>());
    assert!(needs_drop::<NeedDrop>());
    assert!(!needs_drop::<(u8, [u16; 2])>());
    assert!(needs_drop::<(u8, String)>());
    assert!(!needs_drop::<ManuallyDrop<String>>());
    println!("needs-drop-guest-ok");
}
"#;

#[test]
fn run_mode_needs_drop_guest_verdicts() {
    run_ok("needs-drop-guest", NEEDS_DROP_GUEST_SRC, "needs-drop-guest-ok\n");
}


const ADAPTER_DISCARD_DROP_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};

static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

struct Fragment;

impl Drop for Fragment {
    fn drop(&mut self) {
        DROP_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() {
    // The shape of tests/ui/drop/drop-count-assertion-16151.rs
    // (and its same-alloca sibling): every element is pulled
    // and discarded by the adapter's own step, so all three
    // drops must happen inside `collect`'s drain of the
    // `SkipWhile`, before the collected (empty) Vec drops.
    {
        let fragments = vec![Fragment, Fragment, Fragment];
        let _new: Vec<Fragment> = fragments
            .into_iter()
            .skip_while(|_fragment| true)
            .collect();
    }
    assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
    println!("adapter discard drop ok");
}
"#;

#[test]
fn run_mode_skip_while_discarded_items_drop() {
    run_ok(
        "adapter-discard-drop",
        ADAPTER_DISCARD_DROP_SRC,
        "adapter discard drop ok\n",
    );
}

const ADAPTER_STEP_DROP_COUNTS_SRC: &str = r#"
use std::sync::atomic::{AtomicUsize, Ordering};

static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

struct E(u32);

impl Drop for E {
    fn drop(&mut self) {
        DROP_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

fn count() -> usize {
    DROP_COUNT.load(Ordering::SeqCst)
}

fn main() {
    // A `Filter` step drops each rejected item as it pulls
    // it; the kept items drop as the loop binding does.
    let v = vec![E(0), E(1), E(2), E(3)];
    for _x in v.into_iter().filter(|e| e.0 % 2 == 0) {}
    println!("filter {}", count());
    // A `Skip` first step drops the skipped items, and
    // `count` drops the items it pulls and counts.
    let v = vec![E(0), E(1), E(2), E(3), E(4)];
    let k = v.into_iter().skip(2).count();
    println!("count {} {}", k, count());
    // Direct `next` on a `Filter` (the place-based step)
    // drops the rejected items it pulls past.
    let v = vec![E(0), E(1), E(2)];
    let mut it = v.into_iter().filter(|e| e.0 == 2);
    let x = it.next();
    println!("next {:?} {}", x.as_ref().map(|e| e.0), count());
    drop(x);
    drop(it);
    println!("done {}", count());
}
"#;

#[test]
fn run_mode_adapter_step_drop_counts() {
    run_ok(
        "adapter-step-drop-counts",
        ADAPTER_STEP_DROP_COUNTS_SRC,
        "filter 4\ncount 3 9\nnext Some(2) 11\ndone 12\n",
    );
}


const DYN_ANY_LIFETIME_SRC: &str = r#"
use std::any::Any;

struct Foo<'a>(&'a str);

fn good(s: &String) -> Foo<'_> {
    Foo(s)
}

trait AsStr<'a, 'b> {
    fn get(&'a self) -> &'b str;
}

impl<'a> AsStr<'a, 'a> for String {
    fn get(&'a self) -> &'a str {
        self
    }
}

fn main() {
    let s = String::from("foo");
    // A downcast target that differs from the stored type only
    // in a lifetime argument must not match: the stored value
    // is a fn pointer returning `Foo` borrowed from its
    // argument, not a `Foo<'static>`.
    let a: Box<dyn Any> = Box::new(good as fn(&String) -> Foo);
    assert!(a.downcast_ref::<fn(&String) -> Foo<'static>>().is_none());
    assert!(!a.is::<fn(&String) -> Foo<'static>>());
    // The identical type still matches.
    assert!(a.downcast_ref::<fn(&String) -> Foo<'_>>().is_some());
    assert!(a.is::<fn(&String) -> Foo<'_>>());
    // Same for a nested trait object whose second lifetime
    // argument is `'static` only in the downcast target.
    let b: Box<dyn Any> =
        Box::new(Box::new(s) as Box<dyn for<'a> AsStr<'a, 'a>>);
    assert!(b
        .downcast_ref::<Box<dyn for<'a> AsStr<'a, 'static>>>()
        .is_none());
    assert!(b
        .downcast_ref::<Box<dyn for<'a> AsStr<'a, 'a>>>()
        .is_some());
    println!("ok");
}
"#;

#[test]
fn run_mode_dyn_any_downcast_lifetime_only_difference_rejected() {
    // `Any` downcasts compare types by their `TypeId`
    // segments, which distinguish lifetime arguments the
    // converted type model erases: a target differing from
    // the stored type only in a lifetime must be rejected
    // (tests/ui/lifetimes/any-lifetime-escape-higher-rank.rs),
    // while the identical type still matches.
    run_ok(
        "dyn-any-lifetime",
        DYN_ANY_LIFETIME_SRC,
        "ok\n",
    );
}

// ----- packed field drops (packed-struct-drop-aligned) ----------------------

const PACKED_DROP_ALIGNED_SRC: &str = r#"
use std::cell::Cell;

struct Aligned<'a>(&'a Cell<usize>);
impl<'a> Drop for Aligned<'a> {
    fn drop(&mut self) {
        let p = self as *mut Self as usize;
        assert_eq!(p % std::mem::align_of::<Self>(), 0);
        self.0.set(self.0.get() + 1);
    }
}
struct NotCopy(u8);
#[repr(packed)]
struct Packed<'a>(NotCopy, Aligned<'a>);

fn main() {
    let drops = Cell::new(0);
    {
        let a = Aligned(&drops);
        let mut p = Packed(NotCopy(0), a);
        p.1 = Aligned(&drops);
        assert_eq!(drops.get(), 1);
    }
    assert_eq!(drops.get(), 2);
    println!("ok");
}
"#;

#[test]
fn run_mode_packed_field_drops_from_aligned_temporary() {
    // A `repr(packed)` struct's own alignment is its pack cap,
    // so a field whose type needs more alignment is dropped by
    // the compiler's glue from an aligned temporary it moves
    // the field into: the destructor must observe an aligned
    // self pointer both for the reassignment drop (which drop
    // elaboration moves into a temporary of its own) and for
    // the scope-exit drop of the whole local, whose field the
    // glue decomposes in place. Regression for
    // tests/ui/packed/packed-struct-drop-aligned.rs.
    run_ok("packed-drop-aligned", PACKED_DROP_ALIGNED_SRC, "ok\n");
}

const PACKED_DROP_ALIGNED_RECORD_SRC: &str = r#"#![feature(coroutines, stmt_expr_attributes)]
#![feature(coroutine_trait)]
#![allow(unused_assignments, unused_variables)]

use std::cell::Cell;
use std::mem;
use std::ops::Coroutine;
use std::pin::Pin;

struct Aligned<'a> {
    drop_count: &'a Cell<usize>,
}

#[inline(never)]
fn check_align(ptr: *const Aligned) {
    assert_eq!(ptr as usize % mem::align_of::<Aligned>(), 0);
}

impl<'a> Drop for Aligned<'a> {
    fn drop(&mut self) {
        check_align(self);
        self.drop_count.set(self.drop_count.get() + 1);
    }
}

#[repr(transparent)]
struct NotCopy(u8);

#[repr(packed)]
struct Packed<'a>(NotCopy, Aligned<'a>);

fn main() {
    let drop_count = &Cell::new(0);
    {
        let mut p = Packed(NotCopy(0), Aligned { drop_count });
        p.1 = Aligned { drop_count };
        assert_eq!(drop_count.get(), 1);
    }
    assert_eq!(drop_count.get(), 2);

    let drop_count = &Cell::new(0);
    let mut g = #[coroutine]
    || {
        let mut p = Packed(NotCopy(0), Aligned { drop_count });
        let _ = &p;
        p.1 = Aligned { drop_count };
        assert_eq!(drop_count.get(), 1);
        // Test that a coroutine drop function moves a value from a packed
        // struct to a separate local before dropping it. We move out the
        // first field to generate and open drop for the second field.
        drop(p.0);
        yield;
    };
    Pin::new(&mut g).resume(());
    assert_eq!(drop_count.get(), 1);
    drop(g);
    assert_eq!(drop_count.get(), 2);
}
"#;

#[test]
fn run_mode_packed_struct_drop_aligned_record() {
    // The full record: the plain-function half above plus the
    // coroutine half, whose saved packed field drops when the
    // coroutine itself is dropped after the yield.
    run_ok(
        "packed-drop-aligned-record",
        PACKED_DROP_ALIGNED_RECORD_SRC,
        "",
    );
}

// ----- Drop glue for std-internal types (census B-8) -----

const POLL_FN_DROP_SRC: &str = r#"
use std::future::{poll_fn, Future};
use std::pin::pin;
use std::task::{Context, Poll, Waker};

struct Guard(&'static str);
impl Drop for Guard {
    fn drop(&mut self) { println!("dropped {}", self.0); }
}

fn main() {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(&waker);
    {
        let g = Guard("capture");
        let fut = poll_fn(move |_| { let _ = &g; Poll::Ready(()) });
        let mut fut = pin!(fut);
        assert_eq!(fut.as_mut().poll(&mut cx), Poll::Ready(()));
        println!("polled");
    }
    println!("direct done");
    {
        let fut = async {
            let g = Guard("suspended");
            let mut first = true;
            poll_fn(move |_| {
                let _ = &g;
                if first { first = false; Poll::Pending } else { Poll::Ready(()) }
            }).await;
            println!("resumed");
        };
        let mut fut = pin!(fut);
        assert_eq!(fut.as_mut().poll(&mut cx), Poll::Pending);
        println!("suspended now");
    }
    println!("after");
}
"#;

#[test]
fn run_mode_poll_fn_future_drop_runs_capture_destructors() {
    // `std::future::PollFn` is a one-field adapter with no `Drop`
    // impl of its own, but the drop-glue builder's generic path
    // refused it as a foreign ADT whose impl it could not rule
    // out, so every `Drop` of a `poll_fn` future -- directly, as a
    // `Pin`'s pointee, or as a saved field of a suspended async
    // block -- stopped the conversion with "unsupported
    // terminator: Drop" (tests/ui/async-await/async-drop/
    // async-drop-initial.rs, where a `PollFn` is the
    // `test_idempotency` future held by the async-drop futures).
    run_ok(
        "poll-fn-drop",
        POLL_FN_DROP_SRC,
        "polled\ndropped capture\ndirect done\nsuspended now\ndropped suspended\nafter\n",
    );
}

const IO_ERROR_CUSTOM_DROP_SRC: &str = r#"
use std::fmt;
#[derive(Debug)]
struct MyErr;
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "my err") }
}
impl std::error::Error for MyErr {}
impl Drop for MyErr {
    fn drop(&mut self) { println!("MyErr dropped"); }
}
fn main() {
    {
        let e = std::io::Error::new(std::io::ErrorKind::Other, MyErr);
        println!("kind={:?}", e.kind());
    }
    println!("after drop");
}
"#;

#[test]
fn run_mode_io_error_custom_payload_drop_runs() {
    // Dropping an `io::Error` data value whose `Custom` variant
    // holds a `core::io::error::CustomOwner` stopped the
    // conversion with "unsupported terminator: Drop": the owner
    // is a foreign ADT whose `Drop` impl (a call through the
    // payload's stored outer-drop function pointer) the builder
    // could neither execute nor rule out. The modelled plan runs
    // the boxed trait object's glue in the owner's allocation,
    // exactly as the packed representation's drop does, so the
    // payload's destructor runs here.
    run_ok(
        "io-error-custom-drop",
        IO_ERROR_CUSTOM_DROP_SRC,
        "kind=Other\nMyErr dropped\nafter drop\n",
    );
}

const SPAWN_HOOK_SRC: &str = r#"
#![feature(thread_spawn_hook)]
fn main() {
    std::thread::add_spawn_hook(|_| || { println!("child hook"); });
    std::thread::add_spawn_hook(|_| || { println!("child hook2"); });
    println!("registered");
}
"#;

#[test]
fn run_mode_spawn_hook_registration_completes() {
    // Registering a thread spawn hook drops `SpawnHooks` values
    // (the thread-local list head taken out of its `Cell` and the
    // scope-end locals of std's hook functions); the drop-glue
    // builder refused that foreign std-internal type, so the
    // conversion stopped with "unsupported terminator: Drop"
    // (tests/ui/std/add-spawn-hook-reentrancy-159923.rs). Past
    // that conversion, the run then stopped in the drop of the
    // hook list's `Arc` chain: those handles are built by
    // `UniqueArc::into_arc` over a raw allocator block, so
    // their pointer carries the block's first-slot / first-field
    // steps and the inner aggregate sits behind the block's
    // slot and a wrapper field -- a shape the reference-count
    // machinery did not recognise. The machinery now finds the
    // inner through those steps, so registration and the chain
    // drop complete. (Running the hooks at thread spawn is a
    // separate mechanism: the full reentrancy record still
    // stops there, with the hook count at 0.)
    let dir = temp_dir("spawn-hook-registration");
    let src = fixture(&dir, "case.rs", SPAWN_HOOK_SRC);
    let out = run_driver_extra(&src, &["--edition", "2024"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "hook registration and the Arc chain drop must complete; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "registered\n",
        "hook registration stdout mismatch; stderr={stderr}"
    );
}

const UNIQUE_ARC_SRC: &str = r#"
#![feature(unique_rc_arc)]
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, UniqueArc};
static DROPS: AtomicU32 = AtomicU32::new(0);
struct Tracked(u32);
impl Drop for Tracked {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::Relaxed);
        println!("dropped {}", self.0);
    }
}
fn main() {
    let unique = UniqueArc::new(Tracked(7));
    let arc = UniqueArc::into_arc(unique);
    let arc2 = arc.clone();
    println!("strong={} val={}", Arc::strong_count(&arc), arc.0);
    drop(arc);
    println!("after first drop drops={}", DROPS.load(Ordering::Relaxed));
    drop(arc2);
    println!("after last drop drops={}", DROPS.load(Ordering::Relaxed));
}
"#;

#[test]
fn run_mode_unique_arc_into_arc_handle_counts_and_drops() {
    // `UniqueArc::into_arc` builds the `Arc` handle over a raw
    // allocator block (its real body goes through `Box::new_in`
    // and the raw allocation path), so the handle's pointer
    // carries the block's first-slot and wrapper first-field
    // steps and the `ArcInner` sits behind them. The
    // reference-count machinery now recognises that shape:
    // deref reads the pointee, the clone increments the strong
    // count, and the last drop runs the pointee's destructor
    // exactly once.
    run_ok(
        "unique-arc-into-arc",
        UNIQUE_ARC_SRC,
        "strong=2 val=7\nafter first drop drops=0\ndropped 7\nafter last drop drops=1\n",
    );
}
