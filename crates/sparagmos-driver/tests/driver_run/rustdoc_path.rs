use super::{Command, DYN_CONST_DISPATCH_SRC, fixture, run_driver, run_driver_extra, run_ok, summarize_driver, temp_dir};


// ----- rustc_private type defaults (rustdoc execution path) -----------------

// The rustdoc entry program's session configuration asks for
// `ErrorOutputType::default()` before anything else observable
// happens; the value is the `#[default]` variant with its field
// defaults (`HumanReadable` with a non-short, non-unicode kind
// and `ColorConfig::Auto`), served by the interpreter's modelled
// `Default::default`. The expected line is the native 1.100.0
// output of the same program.
const ERROR_OUTPUT_TYPE_DEFAULT_SRC: &str = r#"
#![feature(rustc_private)]
extern crate rustc_errors;
extern crate rustc_session;

use rustc_errors::emitter::ColorConfig;
use rustc_session::config::ErrorOutputType;

fn main() {
    let v = ErrorOutputType::default();
    match v {
        ErrorOutputType::HumanReadable { kind, color_config } => {
            println!(
                "human short={} unicode={} auto={}",
                kind.short,
                kind.unicode,
                matches!(color_config, ColorConfig::Auto)
            );
        }
        ErrorOutputType::Json { .. } => println!("json"),
    }
}
"#;

#[test]
fn run_mode_error_output_type_default() {
    let dir = temp_dir("error-output-type-default");
    let src = fixture(&dir, "case.rs", ERROR_OUTPUT_TYPE_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "human short=false unicode=false auto=true\n",
        "ErrorOutputType::default() must be the HumanReadable default; stderr={stderr}"
    );
}

// The rustdoc entry program's diagnostics context setup asks
// for `DiagCtxtFlags::default()` (the fourth stop on the
// rustdoc execution path); the value is the derived `Default`
// of `rustc_errors::DiagCtxtFlags` -- every field at its own
// default -- served by the interpreter's modelled
// `Default::default`. The guest reads every field of the
// default, then mutates three fields and reads them back; the
// expected lines are the native 1.100.0 output of the same
// program.
const DIAG_CTXT_FLAGS_DEFAULT_SRC: &str = r#"
#![feature(rustc_private)]
extern crate rustc_errors;

use std::num::NonZero;
use rustc_errors::DiagCtxtFlags;

fn show(f: DiagCtxtFlags) {
    let err = match f.treat_err_as_bug {
        None => "none",
        Some(n) => {
            if n.get() == 3 {
                "some3"
            } else {
                "some?"
            }
        }
    };
    println!(
        "warn={} err={} eager={} macro={} dedup={} track={}",
        f.can_emit_warnings,
        err,
        f.eagerly_emit_delayed_bugs,
        f.macro_backtrace,
        f.deduplicate_diagnostics,
        f.track_diagnostics
    );
}

fn main() {
    let mut f = DiagCtxtFlags::default();
    show(f);
    f.can_emit_warnings = true;
    f.treat_err_as_bug = Some(NonZero::new(3usize).unwrap());
    f.track_diagnostics = true;
    show(f);
}
"#;

#[test]
fn run_mode_diag_ctxt_flags_default() {
    let dir = temp_dir("diag-ctxt-flags-default");
    let src = fixture(&dir, "case.rs", DIAG_CTXT_FLAGS_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "warn=false err=none eager=false macro=false dedup=false track=false\n\
         warn=true err=some3 eager=false macro=false dedup=false track=true\n",
        "DiagCtxtFlags::default() must be the all-defaults flags value; stderr={stderr}"
    );
}

const EAGER_CONST_UNREACHABLE_FN_SRC: &str = r#"
struct Fail<T>(std::marker::PhantomData<T>);
impl<T> Fail<T> {
    const C: u32 = panic!("fail const evaluated");
}
fn bad() {
    if false {
        let _ = Fail::<i32>::C;
    }
}
fn main() {
    println!("ok");
}
"#;

#[test]
fn convert_mode_eager_const_in_unreachable_fn_body_converts() {
    // Issue #417: `bad` is a non-generic local function, so the
    // item loop materializes its body even though nothing calls
    // it, and materializing a body makes the Stable MIR builder
    // evaluate every constant operand in it. The failing
    // constant's evaluation report used to stay in the session's
    // error count and fail the whole conversion, even though
    // rustc never collects `bad`, never evaluates the constant,
    // and compiles the crate. A body materialization's
    // evaluation diagnostics are not a verdict; the
    // required-constant check over the collector's instances is.
    let dir = temp_dir("eager-const-unreachable-fn");
    let src = fixture(
        &dir,
        "eager_const_unreachable_fn.rs",
        EAGER_CONST_UNREACHABLE_FN_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing constant in an unreachable function's body must \
         not fail conversion; stderr={stderr}"
    );
}

#[test]
fn run_mode_eager_const_in_unreachable_fn_body_executes() {
    // The same program through run mode: conversion must not be
    // poisoned either, and the program runs and prints.
    let dir = temp_dir("eager-const-unreachable-fn-run");
    let src = fixture(
        &dir,
        "eager_const_unreachable_fn_run.rs",
        EAGER_CONST_UNREACHABLE_FN_SRC,
    );
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "run mode must execute the program; stderr={stderr}"
    );
    assert!(
        stdout.contains("ok"),
        "the program's output must be produced; stdout={stdout}"
    );
}

const EAGER_CONST_DEAD_GENERIC_CALL_SRC: &str = r#"
struct Fail<T>(std::marker::PhantomData<T>);
impl<T> Fail<T> {
    const C: u32 = panic!("fail const evaluated");
}
fn bad<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
fn unused() {
    bad::<i32>();
}
fn main() {
    println!("ok");
}
"#;

#[test]
fn convert_mode_eager_const_behind_dead_generic_call_converts() {
    // Issue #417's isolated shape: conversion demands
    // `bad::<i32>` because the unreachable `unused` calls it,
    // and materializing that body evaluates `Fail::<i32>::C`.
    // The mono collector roots nothing at `unused`, so rustc
    // compiles the crate; conversion must too.
    let dir = temp_dir("eager-const-dead-generic-call");
    let src = fixture(
        &dir,
        "eager_const_dead_generic_call.rs",
        EAGER_CONST_DEAD_GENERIC_CALL_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a failing constant in a body demanded only from dead code \
         must not fail conversion; stderr={stderr}"
    );
}

const EAGER_CONST_DEMANDED_SRC: &str = r#"
struct Fail<T>(std::marker::PhantomData<T>);
impl<T> Fail<T> {
    const C: u32 = panic!("fail const evaluated");
}
fn bad<T>() {
    if false {
        let _ = Fail::<T>::C;
    }
}
fn main() {
    bad::<i32>();
}
"#;

#[test]
fn convert_mode_eager_const_in_demanded_body_still_fails() {
    // Control: the same constant in a body the collector does
    // collect — `main` calls `bad::<i32>` directly — is a
    // required constant of a collected instance, so its failure
    // is a compilation error. The verdict comes from the
    // required-constant check; discounting the materialization's
    // own report must not turn this into a conversion.
    let dir = temp_dir("eager-const-demanded");
    let src = fixture(
        &dir,
        "eager_const_demanded.rs",
        EAGER_CONST_DEMANDED_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a failing required constant of a collected instance must \
         fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}

const MATERIALIZATION_LINT_ENTRY_SRC: &str = r#"
fn main() {
    let array = [std::env::args().len()];
    array[1];
}
"#;

#[test]
fn convert_mode_materialization_lint_in_entry_body_still_fails() {
    // Materializing `main`'s body runs the MIR lint passes over
    // it, and the known-panics lint reports the out-of-bounds
    // index at deny level. That diagnostic is a compilation
    // error under rustc (it lints the bodies it materializes),
    // so the discount for speculative materializations must not
    // swallow it: the verdict records it as a lint finding.
    let dir = temp_dir("materialization-lint-entry");
    let src = fixture(
        &dir,
        "materialization_lint_entry.rs",
        MATERIALIZATION_LINT_ENTRY_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a lint error in the entry body's materialization must \
         fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed"),
        "the failure must be the driver's compilation failure; \
         stderr={stderr}"
    );
}

const MATERIALIZATION_LINT_DEAD_FN_SRC: &str = r#"
fn bad() {
    let array = [std::env::args().len()];
    array[1];
}
fn main() {
    println!("ok");
}
"#;

#[test]
fn convert_mode_materialization_lint_in_dead_body_still_fails() {
    // The same lint in a function nothing calls: unlike a
    // failing constant (which rustc evaluates only for the
    // bodies it collects), the lint is reported for every local
    // body rustc builds, so this is a compilation error too —
    // the lint registry is not scoped to collected instances.
    let dir = temp_dir("materialization-lint-dead-fn");
    let src = fixture(
        &dir,
        "materialization_lint_dead_fn.rs",
        MATERIALIZATION_LINT_DEAD_FN_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a lint error in a dead function's materialization must \
         fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed"),
        "the failure must be the driver's compilation failure; \
         stderr={stderr}"
    );
}

const MATERIALIZATION_VALIDITY_ENTRY_SRC: &str = r#"
use std::cell::Cell;
use std::mem::ManuallyDrop;

#[repr(C)]
struct S {
    x: u32,
    y: E,
}

#[repr(u32)]
enum E {
    A,
    B(U)
}

union U {
    cell: ManuallyDrop<Cell<u32>>,
}

const C: S = {
    let mut s = S { x: 0, y: E::A };
    let p = &mut s.x as *mut u32;
    unsafe { *p.add(1) = 1 };
    s
};

fn main() {
    let _: &'static _ = &C;
}
"#;

#[test]
fn convert_mode_materialization_validity_in_entry_body_still_fails() {
    // Materializing `main`'s body evaluates its constant
    // operands, and validating `&C`'s value fails (interior
    // mutability in read-only memory, E0080). The constant is
    // an operand of a body the collector collects, so the
    // evaluation-class record becomes a finding even though
    // evaluating the constant itself succeeds — the required-
    // constant check alone does not validate.
    let dir = temp_dir("materialization-validity-entry");
    let src = fixture(
        &dir,
        "materialization_validity_entry.rs",
        MATERIALIZATION_VALIDITY_ENTRY_SRC,
    );
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a validity error in the entry body's materialization \
         must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("E0080"),
        "the failure must surface the evaluation error; stderr={stderr}"
    );
}
// ----- rustc_private type defaults (rustdoc execution path) -----------------

// The rustdoc entry program's emitter setup asks for
// `anstream::adapter::strip::StripBytes::default()` (the third
// stop on the rustdoc execution path). The distributed sysroot
// carries anstream as metadata only, so no `Default` body is
// ever collected and the interpreter's modelled
// `Default::default` serves the value: the ANSI-strip state
// machine in its initial state (`state: State::Ground`,
// wrapping a fresh `utf8parse::Parser { point: 0, state:
// Ground }`). The guest prints the value through the derived
// `Debug`, which the interpreter renders from the recorded
// shape; the expected line is the native 1.100.0 output of the
// same program. (Against the MIR-carrying artifact-tree build
// of anstream 0.6.21, the collected strip machinery also runs
// over this modelled value and renders it identically.)
const STRIP_BYTES_DEFAULT_SRC: &str = r#"
#![feature(rustc_private)]

extern crate anstream;

use anstream::adapter::StripBytes;

fn main() {
    let s = StripBytes::default();
    println!("{:?}", s);
}
"#;

#[test]
fn run_mode_strip_bytes_default() {
    let dir = temp_dir("strip-bytes-default");
    let src = fixture(&dir, "case.rs", STRIP_BYTES_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "StripBytes { state: Ground, utf8parser: Utf8Parser { utf8_parser: Parser { point: 0, state: Ground } } }\n",
        "StripBytes::default() must be the initial strip state; stderr={stderr}"
    );
}

// ----- body-less std surface on the rustdoc path (Issue #427) ---------------

// `std::io::Stderr::lock` and `std::io::Stdout::lock` have no MIR
// body in the distributed std; the interpreter serves them as
// builtins returning the handle itself (writes dispatch on the
// receiver type). Compiler programs lock the standard streams to
// emit diagnostics.
const STD_STREAM_LOCK_SRC: &str = r#"
use std::io::Write;

fn main() {
    let mut err_lock = std::io::stderr().lock();
    writeln!(err_lock, "to-stderr-lock").unwrap();
    drop(err_lock);
    let mut out_lock = std::io::stdout().lock();
    writeln!(out_lock, "to-stdout-lock").unwrap();
    drop(out_lock);
    println!("done");
}
"#;

#[test]
fn run_mode_std_stream_lock() {
    let dir = temp_dir("std-stream-lock");
    let src = fixture(&dir, "case.rs", STD_STREAM_LOCK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "to-stdout-lock\ndone\n",
        "stdout lock writes must reach guest stdout; stderr={stderr}"
    );
    assert_eq!(
        stderr, "to-stderr-lock\n",
        "stderr lock writes must reach guest stderr"
    );
}

// `core::slice::copy_from_slice_impl::len_mismatch_fail` has no
// MIR body in the distributed core; the interpreter raises the
// panic itself, with core's exact message as a `String` payload.
const COPY_FROM_SLICE_LEN_MISMATCH_SRC: &str = r#"
fn main() {
    let r = std::panic::catch_unwind(|| {
        let src = [1u8, 2, 3];
        let mut dst = [0u8; 2];
        dst.copy_from_slice(&src);
    });
    println!("caught={}", r.is_err());
    if let Err(e) = r {
        if let Some(s) = e.downcast_ref::<String>() {
            println!("payload={}", s);
        }
    }
}
"#;

#[test]
fn run_mode_copy_from_slice_len_mismatch() {
    run_ok(
        "copy-from-slice-len-mismatch",
        COPY_FROM_SLICE_LEN_MISMATCH_SRC,
        "caught=true\n\
         payload=copy_from_slice: source slice length (3) does not match \
         destination slice length (2)\n",
    );
}

const STRING_FROM_BOXED_STR_SRC: &str = r#"
fn main() {
    let b: Box<str> = Box::from("plain");
    let s = String::from(b);
    println!("[{}] len={}", s, s.len());
    let orig = String::from("héllo wörld");
    let b = orig.into_boxed_str();
    let mut s = String::from(b);
    s.push_str("!");
    println!("{}", s);
    let e: Box<str> = Box::from("");
    let es = String::from(e);
    println!("empty len={} is_empty={}", es.len(), es.is_empty());
    let b2: Box<str> = "abc".into();
    let s2: String = b2.into();
    println!("{}", s2);
}
"#;

#[test]
fn run_mode_string_from_boxed_str() {
    // `String: From<Box<str>>` (and the blanket `Into` twin):
    // the text moves out of the box into the string, which is
    // then an ordinary owned string (it can grow). The box's
    // allocation is released by the move.
    let dir = temp_dir("strfrombox");
    let src = fixture(&dir, "strfrombox.rs", STRING_FROM_BOXED_STR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "String::from(Box<str>) program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[plain] len=5\nhéllo wörld!\nempty len=0 is_empty=true\nabc\n",
        "String::from(Box<str>) stdout mismatch; stderr={stderr}"
    );
}

const COUNTED_STR_FROM_BOXED_STR_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;
fn main() {
    let b: Box<str> = Box::from("rc-test");
    let r: Rc<str> = Rc::from(b);
    let r2 = r.clone();
    println!("rc: {} {} strong={}", r, r2, Rc::strong_count(&r));
    let b: Box<str> = Box::from("arc-test");
    let a: Arc<str> = Arc::from(b);
    println!("arc: {} len={}", a, a.len());
    let r: Rc<str> = Rc::from("borrowed");
    println!("rc-ref: {}", r);
    let r2: Rc<str> = Rc::from(String::from("owned"));
    println!("rc-string: {}", r2);
    let a: Arc<str> = Arc::from("borrowed");
    println!("arc-ref: {}", a);
}
"#;

#[test]
fn run_mode_counted_str_from_boxed_str() {
    // `Rc<str>` / `Arc<str>` conversions: `From<Box<str>>`
    // moves the text out of the box into a fresh counted
    // handle; `From<&str>` and `From<String>` hand the string
    // itself over to the handle.
    let dir = temp_dir("countedstr");
    let src = fixture(&dir, "countedstr.rs", COUNTED_STR_FROM_BOXED_STR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Rc<str>/Arc<str> conversion program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "rc: rc-test rc-test strong=2\narc: arc-test len=8\nrc-ref: borrowed\nrc-string: owned\narc-ref: borrowed\n",
        "Rc<str>/Arc<str> conversion stdout mismatch; stderr={stderr}"
    );
}

#[test]
fn run_mode_dyn_ref_constant_in_static_initializer() {
    // A `&dyn Trait` field inside a static's initializer (the
    // `tracing_core::dispatcher::GLOBAL_DISPATCH` shape): the
    // constant is a fat pointer whose data word names a static
    // and whose vtable word names the concrete type. Decoding
    // it needs the constant form for that pair; without it the
    // whole initializer -- and every body whose constants touch
    // it, like `get_global` here -- stays unconverted.
    let dir = temp_dir("dyn-const-dispatch");
    let src = fixture(&dir, "dyn_const_dispatch.rs", DYN_CONST_DISPATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a dyn reference constant in a static initializer must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "x=7\n");
}

const DYN_CONST_SHAPES_SRC: &str = r#"
trait Sub { fn x(&self) -> u32; }
struct Val(u32);
impl Sub for Val { fn x(&self) -> u32 { self.0 } }
static VAL: Val = Val(42);
static DIRECT: &(dyn Sub + Send + Sync) = &VAL;
struct Holder(Val);
static HOLDER: Holder = Holder(Val(5));
static INTO_FIELD: &(dyn Sub + Send + Sync) = &HOLDER.0;
fn main() {
    let local: &dyn Sub = &Val(9);
    println!("local={}", local.x());
    println!("direct={}", DIRECT.x());
    println!("field={}", INTO_FIELD.x());
    let raw: *const dyn Sub = &VAL;
    println!("raw={}", unsafe { (*raw).x() });
    let a: &dyn Sub = &VAL;
    println!("eq={}", std::ptr::eq(a, DIRECT));
}
"#;

#[test]
fn run_mode_dyn_ref_constant_shapes() {
    // The other constant positions a `&dyn` fat pointer takes:
    // a promoted operand constant in a body (a snapshot base),
    // a static holding the pointer itself (a static-reference
    // base), a pointer into another static's field (a projected
    // static reference), a raw-pointer constant, and pointer
    // identity between two constants naming the same vtable.
    let dir = temp_dir("dyn-const-shapes");
    let src = fixture(&dir, "dyn_const_shapes.rs", DYN_CONST_SHAPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dyn reference constants in every position must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "local=9\ndirect=42\nfield=5\nraw=42\neq=true\n"
    );
}

#[test]
fn run_mode_null_dyn_ref_constant_is_null() {
    // A null `dyn` pointer formed in a const context: the
    // operand constant's allocation carries the vtable
    // provenance on its second word and no provenance on its
    // null data word. It decodes as a dyn reference over the
    // null pointer, and the value's address is the null
    // address: `is_null` observes it.
    let dir = temp_dir("dyn-ref-const-null");
    let src = fixture(
        &dir,
        "dyn_ref_const_null.rs",
        "trait Sub { fn x(&self) -> u32; }
         struct NoSub;
         impl Sub for NoSub { fn x(&self) -> u32 { 7 } }
         const N: *const dyn Sub =
             std::ptr::null::<NoSub>() as *const dyn Sub;
         fn main() {
             println!(\"null={}\", N.is_null());
             println!(\"addr={}\", N as *const () as usize);
         }",
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a null dyn reference constant must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "null=true\naddr=0\n");
}

#[test]
fn run_mode_dyn_ptr_equality_compares_vtable_identity() {
    // Two dyn fat pointers over the same data address (the null
    // address, or one integer address) but with different
    // vtables are different pointers natively: `ptr::eq` and
    // `==` compare the metadata word as well as the address.
    // The data address of such a pointer is the base's address
    // (null checks and address extraction keep observing it);
    // only the comparison distinguishes the vtables.
    let dir = temp_dir("dyn-ptr-eq-vtable");
    let src = fixture(
        &dir,
        "dyn_ptr_eq_vtable.rs",
        "trait T { fn v(&self) -> u32; }
         impl T for u8 { fn v(&self) -> u32 { 1 } }
         impl T for u16 { fn v(&self) -> u32 { 2 } }
         fn main() {
             let a = std::ptr::null::<u8>() as *const dyn T;
             let b = std::ptr::null::<u16>() as *const dyn T;
             let c = std::ptr::null::<u8>() as *const dyn T;
             println!(\"eq_ab={} eq_ac={}\", std::ptr::eq(a, b), std::ptr::eq(a, c));
             println!(\"direct_ab={} direct_ac={}\", a == b, a == c);
             let x = 0x1000usize as *const u8 as *const dyn T;
             let y = 0x1000usize as *const u16 as *const dyn T;
             let z = 0x1000usize as *const u8 as *const dyn T;
             println!(\"eq_xy={} eq_xz={}\", std::ptr::eq(x, y), std::ptr::eq(x, z));
             println!(\"null_a={} null_x={}\", a.is_null(), x.is_null());
             println!(\"addr_x={:#x}\", x as *const u8 as usize);
         }",
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dyn pointer equality must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "eq_ab=false eq_ac=true\ndirect_ab=false direct_ac=true\neq_xy=false eq_xz=true\nnull_a=true null_x=false\naddr_x=0x1000\n"
    );
}

const STR_PATTERN_PRED_MUT_SRC: &str = r#"
fn is_b(c: char) -> bool { c == 'b' }
fn main() {
    let mut n = 0;
    let a = "ab".contains(move |_| { n += 1; n == 2 });
    let mut m = 0;
    let b = "ab".contains(move |_| { m += 1; m == 3 });
    let mut k = 0;
    let c = "ab".starts_with(move |_| { k += 1; k == 1 });
    let d = "ab".contains(is_b);
    println!("{a} {b} {c} {d}");
}
"#;

#[test]
fn run_mode_str_pattern_predicate_mutates_by_value_capture() {
    // A str-pattern predicate is an `FnMut`: a `move` closure's
    // by-value captures must live in mutable storage that persists
    // across the per-character invocations, so the counter reaches
    // 2 on the second character (and never reaches 3). With no
    // environment reference the call machinery wraps a clone of the
    // closure in a value target, and the capture write fails as a
    // write to a constant aggregate.
    run_ok(
        "str-pattern-pred-mut",
        STR_PATTERN_PRED_MUT_SRC,
        "true false true true\n",
    );
}

const STR_PATTERN_PRED_BORROWED_SRC: &str = r#"
fn main() {
    let mut n = 0;
    let mut pred = move |_| { n += 1; n == 2 };
    let a = "b".contains(&mut pred);
    let b = "b".contains(&mut pred);
    let c = "b".contains(&mut pred);
    println!("{a} {b} {c}");
}
"#;

#[test]
fn run_mode_str_pattern_borrowed_predicate_keeps_its_environment() {
    // A borrowed predicate (`&mut pred`) keeps the caller's
    // environment: the counter lives in the caller's closure, so
    // it advances across separate `contains` calls (1, 2, 3) and
    // only the second call sees it reach 2. Resolving the
    // argument to a clone of the closure would give each call a
    // fresh counter and report `false false false`.
    run_ok(
        "str-pattern-pred-borrowed",
        STR_PATTERN_PRED_BORROWED_SRC,
        "false true false\n",
    );
}

const STR_PATTERN_PRED_ENDS_WITH_SRC: &str = r#"
fn main() {
    let mut e = 0;
    let a = "ab".ends_with(move |_| { e += 1; e == 1 });
    let mut v = 0;
    let pred = move |_| { v += 1; v == 2 };
    let b = "ab".contains(pred);
    println!("{a} {b}");
}
"#;

#[test]
fn run_mode_str_pattern_predicate_ends_with_and_moved_variable() {
    // The by-value predicate arrangement also covers `ends_with`
    // (which consults the predicate for the last character) and a
    // predicate held in a variable before being moved into the
    // call.
    run_ok(
        "str-pattern-pred-ends-with",
        STR_PATTERN_PRED_ENDS_WITH_SRC,
        "true true\n",
    );
}

const STR_PATTERN_CHAR_SLICE_SRC: &str = r#"
fn main() {
    let chars = ['a', 'b'];
    let slice = unsafe { std::slice::from_raw_parts(&chars[0], 2) };
    let a = "b".contains(slice);
    let b = "a".contains(slice);
    let c = "z".contains(slice);
    let d = "b".contains(&chars[..]);
    let e = "b".contains(&chars[0..2]);
    let f = "b".contains(&chars[1..2]);
    let g = "a".contains(&chars[1..2]);
    println!("{a} {b} {c} {d} {e} {f} {g}");
}
"#;

#[test]
fn run_mode_str_pattern_char_slice_from_element_base() {
    // A `&[char]` pattern whose fat base references an individual
    // array element (the `from_raw_parts` slice) must be read as
    // the whole set {'a', 'b'}: resolving the base first yields
    // the scalar 'a', which would mistake the set for its first
    // character and report `false` for the first query.
    run_ok(
        "str-pattern-char-slice",
        STR_PATTERN_CHAR_SLICE_SRC,
        "true true false true true true false\n",
    );
}
const STR_PATTERN_PRED_DROP_SRC: &str = r#"
use std::rc::Rc;

fn main() {
    let rc = Rc::new(1);
    let rc2 = rc.clone();
    let pred = move |_: char| {
        let _ = &rc2;
        false
    };
    let r0 = "".contains(pred);
    println!("empty: {r0} count={}", Rc::strong_count(&rc));
    let rc3 = rc.clone();
    let pred2 = move |_: char| {
        let _ = &rc3;
        false
    };
    let r1 = "ab".contains(pred2);
    println!("nomatch: {r1} count={}", Rc::strong_count(&rc));
    let rc4 = rc.clone();
    let mut pred3 = move |_: char| {
        let _ = &rc4;
        true
    };
    let r2 = "ab".contains(&mut pred3);
    println!("borrowed: {r2} count={}", Rc::strong_count(&rc));
    drop(pred3);
    println!("after owner drop: count={}", Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_str_pattern_predicate_drops_by_value_captures() {
    // A by-value predicate is consumed by the search: the
    // callee's real body drops it on the way out, so an `Rc`
    // its captures own is released when the call returns --
    // including for an empty haystack, where the predicate is
    // never invoked. A borrowed predicate is not dropped by
    // the call; its capture lives until the owner drops it.
    run_ok(
        "str-pattern-pred-drop",
        STR_PATTERN_PRED_DROP_SRC,
        "empty: false count=1\nnomatch: false count=1\nborrowed: true count=2\nafter owner drop: count=1\n",
    );
}

const STR_PATTERN_PRED_ZST_DROP_SRC: &str = r#"
struct Bad;
impl Drop for Bad {
    fn drop(&mut self) {
        println!("bad dtor runs");
    }
}
struct Bad2;
impl Drop for Bad2 {
    fn drop(&mut self) {
        println!("bad2 dtor runs");
    }
}
struct Plain;

fn main() {
    // single all-ZST capture
    let bad = Bad;
    let pred = move |_: char| {
        let _ = &bad;
        false
    };
    println!("single: {}", "ab".contains(pred));
    // tuple capture
    let t = (Bad,);
    let pred = move |_: char| {
        let _ = &t;
        false
    };
    println!("tuple: {}", "ab".contains(pred));
    // two distinct ZST captures
    let a = Bad;
    let b = Bad2;
    let pred = move |_: char| {
        let _ = &a;
        let _ = &b;
        true
    };
    println!("pair: {}", "ab".contains(pred));
    // starts_with / ends_with
    let bad = Bad;
    let pred = move |_: char| {
        let _ = &bad;
        true
    };
    println!("starts: {}", "ab".starts_with(pred));
    let bad = Bad;
    let pred = move |_: char| {
        let _ = &bad;
        true
    };
    println!("ends: {}", "ab".ends_with(pred));
    // empty haystack: never invoked, still dropped
    let bad = Bad;
    let pred = move |_: char| {
        let _ = &bad;
        true
    };
    println!("empty: {}", "".contains(pred));
    // panicking predicate: dropped by the callee's cleanup
    let bad = Bad;
    let pred = move |_: char| -> bool {
        let _ = &bad;
        panic!("boom")
    };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| "ab".contains(pred)));
    println!("panic caught: {}", r.is_err());
    // ZST captures without Drop: nothing to run
    let p = Plain;
    let pred = move |_: char| {
        let _ = &p;
        true
    };
    println!("plain: {}", "ab".contains(pred));
}
"#;

#[test]
fn run_mode_str_pattern_predicate_drops_all_zst_captures() {
    // A by-value predicate whose captures are all zero-sized
    // travels as a constant operand, whose type the operand
    // itself does not preserve: conversion built the call's
    // `consumed_glue` from the operand's place type alone, so
    // no glue was attached and the handed-back predicate was
    // never dropped -- its captures' destructors silently
    // never ran, on the success path and the unwind path
    // alike. (A single non-ZST capture made the predicate a
    // place operand, which is why mixed captures behaved.)
    // The glue now falls back to the pattern type the
    // callee's generic arguments name, as `mem_drop_glue`
    // does for the dropped type.
    run_ok(
        "str-pattern-pred-zst-drop",
        STR_PATTERN_PRED_ZST_DROP_SRC,
        "bad dtor runs\nsingle: false\nbad dtor runs\ntuple: false\nbad dtor runs\nbad2 dtor runs\npair: true\nbad dtor runs\nstarts: true\nbad dtor runs\nends: true\nbad dtor runs\nempty: false\nbad dtor runs\npanic caught: true\nplain: true\n",
    );
}

const STR_PATTERN_DYN_PRED_SRC: &str = r#"
fn main() {
    let mut n = 0;
    let mut pred = move |_| { n += 1; n == 2 };
    let a = "b".contains(&mut pred as &mut dyn FnMut(char) -> bool);
    let b = "b".contains(&mut pred as &mut dyn FnMut(char) -> bool);
    let is_b = |c: char| c == 'b';
    let c = "ab".contains(&is_b as &dyn Fn(char) -> bool);
    let d = "ab".starts_with(&is_b as &dyn Fn(char) -> bool);
    println!("{a} {b} {c} {d}");
}
"#;

#[test]
fn run_mode_str_pattern_dyn_predicate() {
    // A trait-object predicate (`&mut dyn FnMut` / `&dyn Fn`)
    // resolves through the dyn callable machinery: the callable
    // behind the object is invoked with the environment the
    // object's base denotes -- the caller's closure storage -- so
    // the counter advances across invocations and across separate
    // searches (1, then 2), exactly as for a concrete borrowed
    // predicate. Before the fix the object fell into the
    // character-slice fallback and the search failed with a type
    // mismatch ("found dyn reference").
    run_ok(
        "str-pattern-dyn-pred",
        STR_PATTERN_DYN_PRED_SRC,
        "false true true false\n",
    );
}

const STR_PATTERN_BOXED_DYN_PRED_SRC: &str = r#"
use std::rc::Rc;

fn main() {
    let mut n = 0;
    let boxed: Box<dyn FnMut(char) -> bool> = Box::new(move |_| {
        n += 1;
        n == 2
    });
    let a = "ab".contains(boxed);
    let rc = Rc::new(1);
    let rc2 = rc.clone();
    let boxed2: Box<dyn FnMut(char) -> bool> = Box::new(move |c: char| {
        let _ = &rc2;
        c == 'b'
    });
    let b = "ab".contains(boxed2);
    println!("{a} {b} count={}", Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_str_pattern_boxed_dyn_predicate() {
    // An owned `Box<dyn FnMut>` predicate is invoked through the
    // box's allocation (its `FnMut` state persists across the
    // per-character invocations, so the counter reaches 2), and
    // the object is dropped when the search returns: the second
    // predicate's `Rc` capture is released, returning the strong
    // count to 1.
    run_ok(
        "str-pattern-boxed-dyn-pred",
        STR_PATTERN_BOXED_DYN_PRED_SRC,
        "true true count=1\n",
    );
}

const STR_PATTERN_BOXED_PRED_SRC: &str = r#"
fn is_b(c: char) -> bool {
    c == 'b'
}

fn main() {
    let a = "".contains(Box::new(|c: char| c == 'b'));
    let b = "ab".contains(Box::new(|c: char| c == 'b'));
    let c = "ab".contains(Box::new(is_b));
    let d = "ab".starts_with(Box::new(|c: char| c == 'a'));
    let e = "ab".ends_with(Box::new(|c: char| c == 'a'));
    println!("{a} {b} {c} {d} {e}");
}
"#;

#[test]
fn run_mode_str_pattern_boxed_concrete_predicate() {
    // A concrete boxed predicate (`Box<closure>` / `Box<fn ptr>`)
    // is a valid `Pattern`: the box aggregate over the callable's
    // allocation is resolved to the callable before the
    // character-slice fallback. Before the fix the box was left as
    // an ADT value and the search failed with a type mismatch
    // ("found ADT value").
    run_ok(
        "str-pattern-boxed-pred",
        STR_PATTERN_BOXED_PRED_SRC,
        "false true true true false\n",
    );
}

const STR_PATTERN_BOXED_PRED_MUT_SRC: &str = r#"
fn main() {
    let mut n = 0;
    let boxed = Box::new(move |_| {
        n += 1;
        n == 2
    });
    let a = "ab".contains(boxed);
    println!("{a}");
}
"#;

#[test]
fn run_mode_str_pattern_boxed_concrete_predicate_mutates() {
    // The boxed predicate is invoked through the box's
    // allocation, so its `FnMut` state persists across the
    // per-character invocations and the counter reaches 2 on the
    // second character.
    run_ok(
        "str-pattern-boxed-pred-mut",
        STR_PATTERN_BOXED_PRED_MUT_SRC,
        "true\n",
    );
}

const STR_PATTERN_BOXED_PRED_DROP_SRC: &str = r#"
use std::rc::Rc;

fn main() {
    let rc = Rc::new(1);
    let rc2 = rc.clone();
    let boxed = Box::new(move |c: char| {
        let _ = &rc2;
        c == 'b'
    });
    let a = "".contains(boxed);
    let rc3 = rc.clone();
    let boxed2 = Box::new(move |c: char| {
        let _ = &rc3;
        c == 'b'
    });
    let b = "ab".contains(boxed2);
    println!("{a} {b} count={}", Rc::strong_count(&rc));
}
"#;

#[test]
fn run_mode_str_pattern_boxed_concrete_predicate_drops_captures() {
    // The owned boxed predicate is handed back as the final state
    // for the call's drop glue when the search returns, so its
    // captures are dropped -- on an empty haystack (never invoked)
    // and after a successful search alike -- and the `Rc` strong
    // count returns to 1.
    run_ok(
        "str-pattern-boxed-pred-drop",
        STR_PATTERN_BOXED_PRED_DROP_SRC,
        "false true count=1\n",
    );
}

const STR_PATTERN_PRED_REF_CHAIN_SRC: &str = r#"
fn main() {
    let mut n = 0;
    let mut pred = move |_| { n += 1; n == 2 };
    let mut borrowed = &mut pred;
    let a = "b".contains(&mut borrowed);
    let b = "b".contains(&mut borrowed);
    let c = "b".contains(&mut borrowed);
    println!("{a} {b} {c}");
}
"#;

#[test]
fn run_mode_str_pattern_predicate_reference_chain() {
    // The pattern may be a reference to a reference to the
    // closure (`&mut borrowed`, where `borrowed: &mut pred`):
    // the environment reference must denote the closure's own
    // storage -- the innermost reference of the chain -- or
    // the closure body's capture projections read the
    // intermediate reference as a capture and the counter's
    // increment fails as a type mismatch. The counter lives
    // in the caller's closure, so it advances across the
    // three calls and only the second sees it reach 2.
    run_ok(
        "str-pattern-pred-ref-chain",
        STR_PATTERN_PRED_REF_CHAIN_SRC,
        "false true false\n",
    );
}

const STR_PATTERN_PRED_REF_CHAIN_VARIANTS_SRC: &str = r#"
use std::rc::Rc;

fn main() {
    let mut n = 0;
    let mut pred = move |_| {
        n += 1;
        n == 2
    };
    let mut b1 = &mut pred;
    let mut b2 = &mut b1;
    let a = "b".contains(&mut b2);
    let b = "b".contains(&mut b2);
    println!("three-layer: {a} {b}");
    let pred2 = |c: char| c == 'b';
    let borrowed2 = &pred2;
    let c = "b".contains(&borrowed2);
    let d = "z".contains(&borrowed2);
    println!("shared-chain: {c} {d}");
    let rc = Rc::new(1);
    let rc2 = rc.clone();
    let mut pred3 = move |_: char| {
        let _ = &rc2;
        true
    };
    let mut borrowed3 = &mut pred3;
    let e = "ab".contains(&mut borrowed3);
    println!("chain-borrowed: {e} count={}", Rc::strong_count(&rc));
    drop(pred3);
    println!("after owner drop: count={}", Rc::strong_count(&rc));
    let mut k = 0;
    let mut pred4 = move |_| {
        k += 1;
        k == 1
    };
    let mut borrowed4 = &mut pred4;
    let f = "ab".starts_with(&mut borrowed4);
    let g = "ab".ends_with(&mut borrowed4);
    println!("chain-sw-ew: {f} {g}");
}
"#;

#[test]
fn run_mode_str_pattern_predicate_reference_chain_variants() {
    // A three-layer chain, a shared-reference chain over a
    // plain `Fn` closure, a chain through `starts_with` /
    // `ends_with`, and a predicate borrowed through a chain
    // keeping its owner: the call does not drop its captures,
    // so the `Rc` count only falls when the owner drops it.
    run_ok(
        "str-pattern-pred-ref-chain-variants",
        STR_PATTERN_PRED_REF_CHAIN_VARIANTS_SRC,
        "three-layer: false true\nshared-chain: true false\nchain-borrowed: true count=2\nafter owner drop: count=1\nchain-sw-ew: true false\n",
    );
}

const OS_STR_ELEMENT_BASE_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;

fn show(label: &str, s: &OsStr) {
    let owned = OsString::from(s);
    let b = owned.as_bytes();
    let sum: u32 = b.iter().map(|x| *x as u32).sum();
    println!("{label}: len={} sum={}", b.len(), sum);
}

fn main() {
    let arr: [u8; 3] = *b"abc";
    let full = unsafe { OsStr::from_encoded_bytes_unchecked(std::slice::from_raw_parts(&arr[0], 3)) };
    show("len3", full);
    let empty = unsafe { OsStr::from_encoded_bytes_unchecked(std::slice::from_raw_parts(&arr[0], 0)) };
    show("len0", empty);
    let one = unsafe { OsStr::from_encoded_bytes_unchecked(std::slice::from_raw_parts(&arr[0], 1)) };
    show("len1", one);
    show("whole", OsStr::from_bytes(&arr[..]));
}
"#;

#[test]
fn run_mode_os_str_from_element_base_keeps_length() {
    // An `OsStr` whose fat pointer references an individual
    // byte of an array keeps the length its metadata states
    // when it is cloned into an `OsString`: resolving the
    // pointer to the byte alone would report a one-byte
    // string for lengths 3 and 0 alike.
    run_ok(
        "os-str-element-base",
        OS_STR_ELEMENT_BASE_SRC,
        "len3: len=3 sum=294\nlen0: len=0 sum=0\nlen1: len=1 sum=97\nwhole: len=3 sum=294\n",
    );
}

// ---------------------------------------------------------------------
// Counted handles inside aggregate elements of a one-element `Vec`
// (Issue #473). A vector value is the aggregate of its elements, so
// a one-element vector whose element is a tuple or an array has the
// same stored shape as the single-array-field pun's wrapper (a
// one-field aggregate whose only field is a tuple). Indexing such a
// vector must select the element, never range inside it: the
// element's fields are not the vector's elements. Without the
// declared type's say, the projection descended into the element
// and the field steps that followed read the handle's own fields
// (an `Rc`'s `NonNull` field, a channel's id word) in its place.
// ---------------------------------------------------------------------

const VEC_ONE_TUPLE_RC_SRC: &str = r#"
use std::cell::Cell;
use std::rc::Rc;

fn main() {
    let src: Vec<(Rc<Cell<i32>>,)> = vec![(Rc::new(Cell::new(8)),)];
    println!("get={}", src[0].0.get());
}
"#;

#[test]
fn run_mode_vec_one_tuple_elem_rc_deref() {
    run_ok("vec-one-tuple-rc", VEC_ONE_TUPLE_RC_SRC, "get=8\n");
}

const VEC_ARRAY_RC_SRC: &str = r#"
use std::cell::Cell;
use std::rc::Rc;

fn main() {
    let a = Rc::new(Cell::new(1));
    let b = Rc::new(Cell::new(2));
    let v: Vec<[Rc<Cell<i32>>; 2]> = vec![[a, b]];
    println!("v01={}", v[0][1].get());
    let w = Rc::downgrade(&v[0][1]);
    println!("weak={}", w.upgrade().unwrap().get());
}
"#;

#[test]
fn run_mode_vec_array_elem_rc_deref_and_downgrade() {
    run_ok("vec-array-rc", VEC_ARRAY_RC_SRC, "v01=2\nweak=2\n");
}

const VEC_TUPLE_CHANNEL_SRC: &str = r#"
use std::sync::mpsc::channel;

fn main() {
    let (tx, rx) = channel::<i32>();
    let src = vec![(tx,)];
    src[0].0.send(9).unwrap();
    println!("recv {}", rx.recv().unwrap());
}
"#;

#[test]
fn run_mode_vec_tuple_elem_channel_send() {
    run_ok("vec-tuple-channel", VEC_TUPLE_CHANNEL_SRC, "recv 9\n");
}

const VEC_TUPLE_PAIR_RC_SRC: &str = r#"
use std::cell::Cell;
use std::rc::Rc;

fn main() {
    let a = Rc::new(Cell::new(1));
    let b = Rc::new(Cell::new(2));
    let src: Vec<(Rc<Cell<i32>>, Rc<Cell<i32>>)> = vec![(a, b)];
    println!("f0={}", src[0].0.get());
    println!("f1={}", src[0].1.get());
}
"#;

#[test]
fn run_mode_vec_tuple_pair_second_field_read() {
    run_ok("vec-tuple-pair-rc", VEC_TUPLE_PAIR_RC_SRC, "f0=1\nf1=2\n");
}

const VEC_ONE_TUPLE_RC_CLONE_SRC: &str = r#"
use std::cell::Cell;
use std::rc::Rc;

fn main() {
    let a = Rc::new(Cell::new(8));
    let probe = a.clone();
    let src: Vec<(Rc<Cell<i32>>,)> = vec![(a,)];
    println!("before={}", Rc::strong_count(&probe));
    let keep = src[0].0.clone();
    println!("after keep={}", Rc::strong_count(&probe));
    drop(keep);
    println!("after drop={}", Rc::strong_count(&probe));
    println!("probe val={}", probe.get());
}
"#;

#[test]
fn run_mode_vec_one_tuple_elem_rc_clone_counts() {
    run_ok(
        "vec-one-tuple-rc-clone",
        VEC_ONE_TUPLE_RC_CLONE_SRC,
        "before=2\nafter keep=3\nafter drop=2\nprobe val=8\n",
    );
}

const VEC_TUPLE_RC_OVERWRITE_SRC: &str = r#"
use std::cell::Cell;
use std::rc::Rc;

fn main() {
    let mut v: Vec<(Rc<Cell<i32>>,)> = vec![(Rc::new(Cell::new(1)),)];
    v[0].0 = Rc::new(Cell::new(42));
    println!("w1={}", v[0].0.get());
    v[0].0.set(43);
    println!("w1b={}", v[0].0.get());
}
"#;

#[test]
fn run_mode_vec_tuple_elem_overwrite_drops_old_handle() {
    // Assigning over the element's field drops the old handle:
    // the drop glue's read of the place must resolve the handle
    // as the ordinary read does.
    run_ok(
        "vec-tuple-rc-overwrite",
        VEC_TUPLE_RC_OVERWRITE_SRC,
        "w1=42\nw1b=43\n",
    );
}

const VEC_TUPLE_ELEM_WRITE_SRC: &str = r#"
fn main() {
    let mut v: Vec<(i32, i32)> = vec![(1, 2)];
    v.push((3, 4));
    println!("w2={} {}", v[1].0, v[1].1);
    let mut w: Vec<[i32; 2]> = vec![[5, 6]];
    w.push([7, 8]);
    println!("w2b={} {}", w[1][0], w[1][1]);
    w[0] = [9, 10];
    println!("w2c={} {}", w[0][0], w[0][1]);
}
"#;

#[test]
fn run_mode_vec_tuple_elem_push_and_replace() {
    // The write direction of the same shape: appends and
    // element writes on a one-element vector of tuples address
    // the element, not the inside of the element.
    run_ok(
        "vec-tuple-elem-write",
        VEC_TUPLE_ELEM_WRITE_SRC,
        "w2=3 4\nw2b=7 8\nw2c=9 10\n",
    );
}

const SELF_SPAWN_SRC: &str = r#"
use std::env;
use std::process::Command;
fn main() {
    let me = env::args_os().next().unwrap();
    match env::args().nth(1).as_deref() {
        Some("child-status") => std::process::exit(7),
        Some("child-output") => {
            println!("child out");
            eprintln!("child err");
        }
        Some("child-abort") => std::process::abort(),
        _ => {
            // A guest argument beyond argv[0] derefs as an
            // `OsStr`, like any other `OsString` value.
            let marker = env::args_os().nth(2).unwrap();
            assert_eq!(&*marker, "marker");
            let st = Command::new(&me).arg("child-status").status().unwrap();
            assert_eq!(st.code(), Some(7));
            let out = Command::new(&me).arg("child-output").output().unwrap();
            assert!(out.status.success());
            assert_eq!(out.stdout, b"child out\n");
            assert_eq!(out.stderr, b"child err\n");
            // A child that aborts dies on a signal: no exit
            // code, as `waitpid` reports it natively.
            let st = Command::new(&me).arg("child-abort").status().unwrap();
            assert_eq!(st.code(), None);
            println!("self spawn ok");
        }
    }
}
"#;

#[test]
fn run_mode_guest_spawning_itself_via_status_and_output() {
    // A guest that spawns its own program path (the
    // `needs-subprocess` shape throughout the rustc suite)
    // re-runs itself through the driver: `status` observes
    // the child's real exit code (or its death by abort),
    // and `output` captures the child's stdout and stderr.
    let dir = temp_dir("self-spawn");
    let src = fixture(&dir, "selfspawn.rs", SELF_SPAWN_SRC);
    let out = run_driver_extra(
        &src,
        &["--guest-arg", "parent", "--guest-arg", "marker"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "self-spawning guest must exit 0; stdout={stdout} stderr={stderr}"
    );
    assert_eq!(stdout, "self spawn ok\n");
}

// ---------------------------------------------------------------------------
// Imported `main` (issue #532): a crate-root re-export as the
// entry point. Run mode pins `--crate-type=lib`, under which
// rustc designates no entry point at all (its entry search only
// runs for an executable crate type); the driver replays the
// resolver's root-module resolution (root_main_target) and
// designates the re-export's target.
// ---------------------------------------------------------------------------

const IMPORTED_MAIN_INNER_SRC: &str = r#"
pub mod foo {
    pub fn bar() {
        println!("Hello world!");
    }
}
use foo::bar as main;
"#;

#[test]
fn run_mode_imported_main_from_inner_mod() {
    let dir = temp_dir("imported-main-inner");
    let src = fixture(&dir, "imported_main_from_inner_mod.rs", IMPORTED_MAIN_INNER_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr} stdout={stdout}");
    assert_eq!(stdout, "Hello world!\n", "stderr={stderr}");
}

// ---------------------------------------------------------------------------
// Aux env scrub (issue #532): the test's own rustc-env
// variables ride the driver process's environment into the
// main conversion, but an auxiliary build's environment is
// configured separately (compiletest): the driver scrubs
// the named variables from its auxiliary builds, whose own
// rustc-env directives are applied after the scrub.
// ---------------------------------------------------------------------------

const SCRUB_AUX_SRC: &str = r#"
pub fn baked() -> Option<&'static str> {
    option_env!("COMPILETEST_FOO")
}
"#;

const SCRUB_MAIN_SRC: &str = r#"
extern crate scrub_aux;

fn main() {
    // The main compilation sees the variable (the driver
    // process carries it); the auxiliary build must not.
    assert_eq!(env!("COMPILETEST_FOO"), "foo");
    assert_eq!(scrub_aux::baked(), None);
    println!("scrub ok");
}
"#;

#[test]
fn run_mode_aux_env_scrub() {
    let dir = temp_dir("aux-env-scrub");
    let aux = fixture(&dir, "scrub_aux.rs", SCRUB_AUX_SRC);
    let src = fixture(&dir, "scrub_main.rs", SCRUB_MAIN_SRC);
    let spec = format!("scrub_aux={}", aux.display());
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--aux")
        .arg(&spec)
        .arg("--aux-env-scrub")
        .arg("COMPILETEST_FOO")
        .env("COMPILETEST_FOO", "foo")
        .output()
        .expect("failed to run sparagmos-driver");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr} stdout={stdout}");
    assert_eq!(stdout, "scrub ok\n", "stderr={stderr}");
}
