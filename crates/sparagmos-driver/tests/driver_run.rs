//! Run-mode smoke tests: build small fixture programs in a temp dir,
//! execute them with `sparagmos-driver run`, and check the exit-code
//! contract (0 = normal termination, 101 = guest panic, 2 = could
//! not convert/execute, with the reason on stderr). The test only
//! spawns the binary; the driver itself must have been built with
//! the pinned 1.100.0 toolchain (see the crate README).

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
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("sparagmos-driver-run-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const PANIC_SRC: &str = r#"
fn boom(x: i32) -> i32 { x + x }
fn main() { let _ = boom(2_000_000_000); }
"#;

const PRINTLN_SRC: &str = r#"
fn main() { println!("hi"); }
"#;

const SILENT_SRC: &str = r#"
fn main() {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < 100 { s += i; i += 1; }
    let _ = s;
}
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

#[test]
fn run_mode_println_prints_to_stdout() {
    let dir = temp_dir("println");
    let src = fixture(&dir, "println.rs", PRINTLN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "println! runs through the interpreter's runtime print support; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hi\n",
        "the guest's stdout is the driver's stdout"
    );
}

#[test]
fn run_mode_silent_program_exits_zero() {
    // A plain computation-only main converts and executes to the end.
    // This used to be blocked in the interpreter: real MIR never
    // writes the return place `_0` of a unit-returning function, and
    // `Return` read it as uninitialized (exit 2, reason naming `_0`).
    // The interpreter now reads an unwritten unit `_0` as `()`, so
    // the program terminates normally.
    let dir = temp_dir("silent");
    let src = fixture(&dir, "silent.rs", SILENT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "silent program must terminate normally; stderr={stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

// ----- by-reference constants (Lane R) ---------------------------------------

const PROMOTED_REF_SRC: &str = r#"
fn main() {
    let x = 10;
    let y = x;
    assert_eq!(y, 10);
    let z: u8 = 255;
    assert_eq!(z, 255);
}
"#;

const CONST_SHAPES_SRC: &str = r#"
struct S { a: i32, b: u8 }
const S1: S = S { a: -2, b: 7 };
const ARR: [i32; 3] = [10, 20, 30];
const R: &i32 = &ARR[1];
static A: usize = 42;
static B: usize = A;
const NAMES: [&str; 2] = ["x", "yy"];
const OPT: Option<&i32> = Some(&5);
fn main() {
    println!("{} {} {} {} {} {}", S1.a, S1.b, *R, B, NAMES[1],
             match OPT { Some(r) => *r, None => 0 });
}
"#;

#[test]
fn run_mode_promoted_ref_constant() {
    // `assert_eq!` against a literal builds a promoted `&i32`
    // constant; it must execute, not stop at the constant.
    let dir = temp_dir("promoted");
    let src = fixture(&dir, "promoted.rs", PROMOTED_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "promoted reference constants must run; stderr={stderr}"
    );
}

#[test]
fn run_mode_materialised_constant_shapes() {
    // Struct constants, a reference into the middle of an array
    // constant, statics read by value, a `[&str]` constant and a
    // niche-encoded `Option<&i32>` constant, all printed.
    let dir = temp_dir("const-shapes");
    let src = fixture(&dir, "const_shapes.rs", CONST_SHAPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "materialised constants must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "-2 7 20 42 yy 5\n",
        "decoded constant values must match the source program"
    );
}

const TAIL_CALL_SRC: &str = r#"
#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]
fn countdown(n: u32) -> u32 {
    if n == 0 { 0 } else { become countdown(n - 1) }
}
fn main() { let _ = countdown(3); }
"#;

const ASYNC_CLOSURE_SRC: &str = r#"
fn main() {
    let f = async move |x: i32| x + 1;
    let _ = &f;
}
"#;

const LOG_SYNTAX_SRC: &str = r#"
#![feature(log_syntax)]
fn main() {
    log_syntax!("compile-time marker");
    println!("guest marker");
}
"#;


#[test]
fn run_mode_tail_call_conversion_panic_exits_two() {
    // Converting a `become` (explicit tail call) makes rustc_public
    // panic (`unimplemented!()` in Stable MIR's terminator
    // conversion). The driver must catch that panic at the compiler
    // boundary and report it under the run-mode contract — exit 2
    // with the panic message as the reason — instead of dying with
    // the panic (exit 101 and a default panic dump on stderr).
    let dir = temp_dir("tailcall");
    let src = fixture(&dir, "tailcall.rs", TAIL_CALL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a conversion panic is a conversion failure (exit 2), not a process panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("Stable MIR conversion panicked"),
        "the reason names the conversion panic: {stderr}"
    );
    assert!(
        stderr.contains("not implemented"),
        "the reason carries the panic message: {stderr}"
    );
    assert!(
        !stderr.contains("RUST_BACKTRACE"),
        "the default panic dump is suppressed, the message is reported as the reason: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "run mode keeps stdout clean: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}


#[test]
fn run_mode_async_closure_conversion_panic_exits_two() {
    // Async closures hit a different `unimplemented!()` inside
    // Stable MIR (type conversion: "FIXME(async_closures)"). Same
    // contract as the tail-call case: exit 2, panic message as the
    // reason, no process panic.
    let dir = temp_dir("asyncclosure");
    let src = fixture(&dir, "asyncclosure.rs", ASYNC_CLOSURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a conversion panic is a conversion failure (exit 2), not a process panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("Stable MIR conversion panicked"),
        "the reason names the conversion panic: {stderr}"
    );
    assert!(
        stderr.contains("async_closures"),
        "the reason carries the panic message: {stderr}"
    );
    assert!(
        !stderr.contains("RUST_BACKTRACE"),
        "the default panic dump is suppressed, the message is reported as the reason: {stderr}"
    );
}


#[test]
fn run_mode_log_syntax_output_stays_off_stdout() {
    // `log_syntax!` prints to the *compiler's* stdout at compile
    // time. The compiler runs in-process, so without redirection
    // that output would land in the driver's stdout — which run
    // mode reserves for the guest program's output. It must appear
    // on stderr (compiler-side diagnostics) instead, leaving stdout
    // byte-identical to the guest's own output.
    let dir = temp_dir("logsyntax");
    let src = fixture(&dir, "logsyntax.rs", LOG_SYNTAX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program itself runs fine; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "guest marker\n",
        "stdout carries the guest's output alone"
    );
    assert!(
        stderr.contains("compile-time marker"),
        "the log_syntax output is redirected to stderr: {stderr}"
    );
}

const EXPLICIT_DISCRIMINANT_SRC: &str = r#"
#[repr(i32)]
enum E { A = 3, B = 7, C = -2 }
#[repr(i32)]
enum P { X(i32) = -5, Y(u32) = 42, Z = 0 }
#[repr(u8)]
enum U { Lo = 3, Hi = 200 }
fn name(e: E) -> i32 { match e { E::A => 100, E::B => 200, E::C => 300 } }
fn p_val(p: P) -> i32 { match p { P::X(v) => v, P::Y(v) => v as i32 + 1, P::Z => -1 } }
fn u_val(u: U) -> u32 { match u { U::Lo => 1, U::Hi => 2 } }
const PC: P = P::Y(9);
fn main() {
    println!("{}", name(E::A));
    println!("{}", name(E::B));
    println!("{}", name(E::C));
    println!("{}", p_val(P::X(-7)));
    println!("{}", p_val(PC));
    println!("{}", u_val(U::Hi));
    println!("{}", E::C as i32);
}
"#;

#[test]
fn run_mode_explicit_discriminant_match() {
    // Enums with explicit discriminants: `Discriminant` must yield
    // the discriminant values (3, 7, -2, ...), not the variant
    // indices, or every match falls through. The expected output
    // is what the native build of the same program prints.
    let dir = temp_dir("explicit-discriminant");
    let src = fixture(&dir, "explicit_discriminant.rs", EXPLICIT_DISCRIMINANT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "explicit-discriminant enums must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "100\n200\n300\n-7\n10\n2\n-2\n",
        "matches on explicit discriminants must pick the native branches"
    );
}

const NEGATIVE_INT_MATCH_SRC: &str = r#"
fn f(x: i32) -> i32 { match x { -2 => 300, 3 => 100, _ => 0 } }
fn main() { println!("{}", f(-2)); println!("{}", f(3)); println!("{}", f(9)); }
"#;

#[test]
fn run_mode_negative_int_match() {
    // SwitchInt branch keys are stored as zero-extended bit
    // patterns; a signed match must still hit its negative arms.
    let dir = temp_dir("negative-int-match");
    let src = fixture(&dir, "negative_int_match.rs", NEGATIVE_INT_MATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "signed int match must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "300\n100\n0\n",
        "the -2 arm must be taken for -2"
    );
}

// ----- auxiliary crates (--aux) ----------------------------------------------

const AUX_BASE_SRC: &str = r#"
pub fn base_val() -> i32 { 40 }
"#;

const AUX_BROKEN_SRC: &str = r#"
pub fn broken( -> i32 { 1 }
"#;

const AUX_LIB_SRC: &str = r#"
fn secret(x: i32) -> i32 { x * 3 }
pub fn add(a: i32, b: i32) -> i32 { a + b }
pub fn triple(x: i32) -> i32 { secret(x) }
pub fn generic_id<T>(x: T) -> T { x }
"#;

const AUX_MAIN_SRC: &str = r#"
extern crate auxlib;
fn main() {
    println!("add={}", auxlib::add(2, 3));
    println!("triple={}", auxlib::triple(7));
}
"#;

const AUX_MID_SRC: &str = r#"
extern crate base;
pub fn with_base() -> i32 { base::base_val() + 2 }
"#;

const AUX_NESTED_MAIN_SRC: &str = r#"
extern crate mid;
fn main() { println!("val={}", mid::with_base()); }
"#;

fn run_driver_with_aux(src: &Path, aux: &[(&str, &Path)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg("run").arg(src);
    for (name, path) in aux {
        cmd.arg("--aux").arg(format!("{}={}", name, path.display()));
    }
    cmd.output().expect("failed to run sparagmos-driver")
}

#[test]
fn run_mode_aux_compile_failure_exits_two() {
    let dir = temp_dir("aux-broken");
    let aux = fixture(&dir, "auxlib.rs", AUX_BROKEN_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an uncompilable auxiliary crate is a conversion failure; stderr={stderr}"
    );
    assert!(
        stderr.contains("auxiliary crate `auxlib` failed to compile"),
        "the reason names the auxiliary crate: {stderr}"
    );
}

#[test]
fn run_mode_aux_crate_functions_execute() {
    // Functions of an auxiliary crate -- including its private
    // helpers -- are converted into the same program and execute in
    // the interpreter; the crate's generic functions are skipped
    // without affecting the run.
    let dir = temp_dir("aux-basic");
    let aux = fixture(&dir, "auxlib.rs", AUX_LIB_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "aux calls must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "add=5\ntriple=21\n",
        "aux function results reach the guest's stdout"
    );
}


const AUX_PM_SRC: &str = r#"
extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn answer(_input: TokenStream) -> TokenStream {
    "42".parse().unwrap()
}
"#;

const AUX_PM_MAIN_SRC: &str = r#"
extern crate pmaux;
fn main() { println!("answer={}", pmaux::answer!()); }
"#;

// `async` is an identifier only in edition 2015.
const AUX_ED2015_SRC: &str = r#"
pub fn async() -> i32 { 11 }
"#;

// `gen` is a reserved keyword only in edition 2024.
const AUX_ED2021_SRC: &str = r#"
pub fn gen() -> i32 { 7 }
"#;

const AUX_EDITIONS_MAIN_SRC: &str = r#"
extern crate auxa;
extern crate auxb;
fn main() {
    println!("a={}", auxa::r#async());
    println!("b={}", auxb::r#gen());
}
"#;

fn run_driver_with_aux_specs(
    src: &Path,
    edition: Option<&str>,
    specs: &[String],
) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg("run").arg(src);
    if let Some(ed) = edition {
        cmd.arg("--edition").arg(ed);
    }
    for spec in specs {
        cmd.arg("--aux").arg(spec);
    }
    cmd.output().expect("failed to run sparagmos-driver")
}

fn aux_spec(name: &str, path: &Path, attrs: &str) -> String {
    format!("{}={}{}", name, path.display(), attrs)
}

#[test]
fn run_mode_aux_proc_macro_crate_type_attribute() {
    // A proc-macro auxiliary crate whose source declares no crate
    // type (the shape compiletest's `//@ proc-macro:` produces):
    // the `;crate-type=proc-macro` attribute builds it as a proc
    // macro, and the main crate's macro invocation runs inside the
    // compiler before the guest executes.
    let dir = temp_dir("aux-proc-macro");
    let pm = fixture(&dir, "pmaux.rs", AUX_PM_SRC);
    let src = fixture(&dir, "pmmain.rs", AUX_PM_MAIN_SRC);
    let out = run_driver_with_aux_specs(
        &src,
        None,
        &[aux_spec("pmaux", &pm, ";crate-type=proc-macro")],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "proc-macro aux must build and expand; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "answer=42\n");
}

#[test]
fn run_mode_aux_proc_macro_without_attribute_fails() {
    // The same source without the attribute is built as an rlib,
    // which rustc rejects for `#[proc_macro]` items: a conversion
    // failure (exit 2), not a silent mis-build.
    let dir = temp_dir("aux-proc-macro-plain");
    let pm = fixture(&dir, "pmaux.rs", AUX_PM_SRC);
    let src = fixture(&dir, "pmmain.rs", AUX_PM_MAIN_SRC);
    let out = run_driver_with_aux_specs(&src, None, &[aux_spec("pmaux", &pm, "")]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("auxiliary crate `pmaux` failed to compile"),
        "the reason names the auxiliary crate: {stderr}"
    );
}

#[test]
fn run_mode_aux_per_crate_editions() {
    // Two auxiliary crates with different editions, both different
    // from the main crate's edition 2024: each `;edition=` wins for
    // its own crate (compiletest builds each auxiliary with the
    // edition the auxiliary file itself declares).
    let dir = temp_dir("aux-editions");
    let auxa = fixture(&dir, "auxa.rs", AUX_ED2015_SRC);
    let auxb = fixture(&dir, "auxb.rs", AUX_ED2021_SRC);
    let src = fixture(&dir, "edmain.rs", AUX_EDITIONS_MAIN_SRC);
    let out = run_driver_with_aux_specs(
        &src,
        Some("2024"),
        &[
            aux_spec("auxa", &auxa, ";edition=2015"),
            aux_spec("auxb", &auxb, ";edition=2021"),
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "per-aux editions must apply; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "a=11\nb=7\n");
}

#[test]
fn run_mode_aux_without_edition_attribute_uses_main_edition() {
    // Without `;edition=`, the auxiliary crate is built with the
    // main edition (2024), under which `gen` is reserved: the
    // build fails as a conversion failure.
    let dir = temp_dir("aux-edition-default");
    let auxb = fixture(&dir, "auxb.rs", AUX_ED2021_SRC);
    let src = fixture(&dir, "edmain.rs", AUX_EDITIONS_MAIN_SRC);
    let out = run_driver_with_aux_specs(
        &src,
        Some("2024"),
        &[aux_spec("auxb", &auxb, "")],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("auxiliary crate `auxb` failed to compile"),
        "the reason names the auxiliary crate: {stderr}"
    );
}

#[test]
fn run_mode_nested_aux_crates_execute() {
    // Auxiliary crates are passed in dependency order: `mid` uses
    // `base`, and the main crate uses `mid`.
    let dir = temp_dir("aux-nested");
    let base = fixture(&dir, "base.rs", AUX_BASE_SRC);
    let mid = fixture(&dir, "mid.rs", AUX_MID_SRC);
    let src = fixture(&dir, "nestedmain.rs", AUX_NESTED_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("base", &base), ("mid", &mid)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested aux calls must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "val=42\n");
}

// ----- closures (Lane AU) ----------------------------------------------------

fn summarize_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg(src)
        .output()
        .expect("failed to run sparagmos-driver")
}

const CLOSURE_CAPTURE_SRC: &str = r#"
fn main() {
    let base = 40;
    let add = |x: i32| x + base;
    println!("{}", add(2));
}
"#;

const CLOSURE_NO_CAPTURE_SRC: &str = r#"
fn main() {
    let twice = |x: i32| x * 2;
    println!("{}", twice(21));
}
"#;

const CLOSURE_FNMUT_SRC: &str = r#"
fn main() {
    let mut count = 0;
    let mut inc = || { count += 1; count };
    println!("{}", inc());
    println!("{}", inc());
}
"#;

const CLOSURE_IN_STRUCT_SRC: &str = r#"
struct Holder<F: Fn(i32) -> i32> { f: F }
fn main() {
    let k = 5;
    let h = Holder { f: |x: i32| x + k };
    println!("{}", (h.f)(7));
}
"#;

const CLOSURE_FNONCE_MOVE_OUT_SRC: &str = r#"
fn main() {
    let s = String::from("abc");
    let f = move || -> String { s };
    let out = f();
    println!("{}", out);
}
"#;

const CLOSURE_FNONCE_IN_STRUCT_SRC: &str = r#"
struct Once<F: FnOnce() -> String> { f: F }
fn main() {
    let s = String::from("xyz");
    let o = Once { f: move || s };
    let out = (o.f)();
    println!("{}", out);
}
"#;

const CLOSURE_RETURNED_SRC: &str = r#"
fn make_adder(n: i32) -> impl Fn(i32) -> i32 {
    move |x| x + n
}
fn main() {
    let add = make_adder(30);
    println!("{}", add(12));
}
"#;

const CLOSURE_NESTED_SRC: &str = r#"
fn main() {
    let outer = |x: i32| {
        let inner = |y: i32| y + 1;
        inner(x) * 2
    };
    println!("{}", outer(10));
}
"#;

#[test]
fn run_mode_closure_with_capture() {
    let dir = temp_dir("closure-capture");
    let src = fixture(&dir, "closure_capture.rs", CLOSURE_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure capturing a local must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_closure_without_capture() {
    let dir = temp_dir("closure-no-capture");
    let src = fixture(&dir, "closure_no_capture.rs", CLOSURE_NO_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a captureless closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_fnmut_closure_keeps_state() {
    let dir = temp_dir("closure-fnmut");
    let src = fixture(&dir, "closure_fnmut.rs", CLOSURE_FNMUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnMut closure must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1\n2\n",
        "the captured counter must persist across calls"
    );
}

#[test]
fn run_mode_closure_stored_in_struct() {
    let dir = temp_dir("closure-in-struct");
    let src = fixture(&dir, "closure_in_struct.rs", CLOSURE_IN_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure stored in a struct field must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "12\n");
}

#[test]
fn run_mode_fnonce_closure_moves_capture_out() {
    let dir = temp_dir("closure-fnonce-move-out");
    let src = fixture(&dir, "closure_fnonce_move_out.rs", CLOSURE_FNONCE_MOVE_OUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnOnce closure moving its capture out must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "abc\n");
}

#[test]
fn run_mode_fnonce_closure_in_struct_field() {
    let dir = temp_dir("closure-fnonce-in-struct");
    let src = fixture(
        &dir,
        "closure_fnonce_in_struct.rs",
        CLOSURE_FNONCE_IN_STRUCT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnOnce closure stored in a struct field must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "xyz\n");
}

#[test]
fn run_mode_closure_returned_from_function() {
    let dir = temp_dir("closure-returned");
    let src = fixture(&dir, "closure_returned.rs", CLOSURE_RETURNED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure returned from a function must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_nested_closures() {
    let dir = temp_dir("closure-nested");
    let src = fixture(&dir, "closure_nested.rs", CLOSURE_NESTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested closures must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "22\n");
}

const CLOSURE_GENERIC_CONTEXT_SRC: &str = r#"
fn wrap<T: Copy>(v: T) -> T {
    let id = |x: T| x;
    id(v)
}
fn main() { println!("{}", wrap::<i32>(7)); }
"#;

const CLOSURE_FNONCE_SRC: &str = r#"
struct Bag(i32);
fn main() {
    let bag = Bag(41);
    let f = move || bag;
    let _b = f();
}
"#;

#[test]
fn summary_mode_generic_context_closure_is_skipped_with_reason() {
    // A closure inside a generic function has no single monomorphic
    // body to collect (the same limitation as the generic function
    // itself). It stays skipped, with the reason naming the generic
    // context instead of the generic "no monomorphic instance".
    let dir = temp_dir("closure-generic");
    let src = fixture(&dir, "closure_generic.rs", CLOSURE_GENERIC_CONTEXT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the closure item");
    assert!(
        line.contains("SKIPPED") && line.contains("generic context"),
        "the closure is skipped with the generic-context reason: {line}"
    );
}

#[test]
fn summary_mode_fnonce_closure_body_is_converted() {
    // Conversion covers FnOnce closures too: the body is collected
    // even though the interpreter's by-value environment passing is
    // a separate concern.
    let dir = temp_dir("closure-fnonce");
    let src = fixture(&dir, "closure_fnonce.rs", CLOSURE_FNONCE_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the closure item");
    assert!(
        line.contains("[converted]"),
        "the FnOnce closure body is converted: {line}"
    );
}

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

// ----- generic function definitions (Lane BA) --------------------------------

const GENERIC_CALLS_SRC: &str = r#"
fn id<T>(x: T) -> T { x }
fn first<T>(v: &[T]) -> &T { &v[0] }
struct Wrap<T>(T);
impl<T> Wrap<T> {
    fn get(&self) -> &T { &self.0 }
}
fn main() {
    let a = id::<u32>(3);
    let b = id::<i64>(-4);
    let w = Wrap::<u32>(9);
    let arr = [10u32, 20, 30];
    let c = *first(&arr);
    println!("{}", a as i64 + b + *w.get() as i64 + c as i64);
}
"#;

#[test]
fn run_mode_generic_function_instances_execute() {
    // Generic definitions have no instance as items; the instances
    // their call sites demand are resolved and converted, so a
    // program calling generic functions runs end to end.
    let dir = temp_dir("generic-calls");
    let src = fixture(&dir, "generic_calls.rs", GENERIC_CALLS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic calls must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "18\n");
}

const GENERIC_UNUSED_SRC: &str = r#"
fn used<T>(x: T) -> T { x }
fn unused<T: Clone>(x: T) -> T { x.clone() }
fn main() {
    let _ = used::<u32>(1);
}
"#;

#[test]
fn summary_mode_demanded_generic_instances_are_converted() {
    // The demanded definition is reported converted; a definition
    // nothing demands keeps its no-instance record (it is not
    // silently counted as converted).
    let dir = temp_dir("generic-unused");
    let src = fixture(&dir, "generic_unused.rs", GENERIC_UNUSED_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let used = stdout
        .lines()
        .find(|l| l.starts_with("fn generic_unused::used:"))
        .expect("summary lists the used generic definition");
    assert!(
        used.contains("[converted]"),
        "the demanded generic definition is converted: {used}"
    );
    let unused = stdout
        .lines()
        .find(|l| l.starts_with("fn generic_unused::unused:"))
        .expect("summary lists the unused generic definition");
    assert!(
        unused.contains("SKIPPED (no monomorphic instance available (kind=Fn)"),
        "the undemanded generic definition stays explicitly skipped: {unused}"
    );
}

// ----- trait method keys (Issue #64) ------------------------------------------

const TRAIT_CALL_SRC: &str = r#"
trait Greeter { fn greet(&self) -> u32; }
struct S;
impl Greeter for S { fn greet(&self) -> u32 { 42 } }
fn main() {
    let s = S;
    println!("{} {}", s.greet(), <S as Greeter>::greet(&s));
}
"#;

#[test]
fn run_mode_trait_method_call_executes() {
    // A call to a trait-declared method names the trait
    // declaration in its FnDef target, while the item loop
    // converts the impl method under the impl's own key; the
    // driver registers the impl body under the trait key too, so
    // the call executes (Issue #64).
    let dir = temp_dir("trait-call");
    let src = fixture(&dir, "trait_call.rs", TRAIT_CALL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "trait method calls must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42 42\n");
}

const TRAIT_TWO_TYPES_SRC: &str = r#"
trait Get { fn get(&self) -> u32; }
struct Wrap<T>(T);
impl Get for Wrap<u32> { fn get(&self) -> u32 { self.0 + 1 } }
impl Get for Wrap<u16> { fn get(&self) -> u32 { self.0 as u32 + 100 } }
trait Tag { fn tag(&self) -> u32; }
struct Pair<T>(T, T);
impl<T> Tag for Pair<T> { fn tag(&self) -> u32 { 7 } }
fn main() {
    let a = Wrap(3u32).get();
    let b = Wrap(4u16).get();
    let c = Pair(1u32, 2u32).tag();
    let d = Pair(true, false).tag();
    println!("{} {} {} {}", a, b, c, d);
}
"#;

#[test]
fn run_mode_trait_method_two_type_arguments() {
    // The same trait method demanded at two different type
    // arguments resolves to the right impl body for each: two
    // concrete impls of one generic struct, and one generic impl
    // instantiated at two types.
    let dir = temp_dir("trait-two-types");
    let src = fixture(&dir, "trait_two_types.rs", TRAIT_TWO_TYPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "trait calls at two type arguments must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "4 104 7 7\n");
}

const TRAIT_DEFAULT_SRC: &str = r#"
trait D {
    fn val(&self) -> u32 { 10 }
    fn extra(&self) -> u32 { self.val() + 1 }
}
struct X;
impl D for X {}
struct Y;
impl D for Y { fn val(&self) -> u32 { 20 } }
fn main() {
    println!("{} {} {} {}", X.val(), X.extra(), Y.val(), Y.extra());
}
"#;

#[test]
fn run_mode_trait_default_method_body() {
    // A trait default method keeps its own body: X uses both
    // defaults, Y overrides `val`, and the default `extra` body
    // dispatches its inner `self.val()` call to Y's override.
    let dir = temp_dir("trait-default");
    let src = fixture(&dir, "trait_default.rs", TRAIT_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "trait default methods must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "10 11 20 21\n");
}
