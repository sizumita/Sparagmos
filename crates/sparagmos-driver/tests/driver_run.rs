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
fn run_mode_async_closure_exits_two_with_named_reason() {
    // Async closures hit an `unimplemented!()` inside Stable MIR
    // (type conversion: "FIXME(async_closures)"): a body that
    // mentions a coroutine-closure type cannot be converted at
    // all. The conversion contains that panic per body and skips
    // the body with a named reason, so run mode reports the
    // unconverted `main` — exit 2, the skip reason as the
    // message, no conversion panic and no process panic.
    let dir = temp_dir("asyncclosure");
    let src = fixture(&dir, "asyncclosure.rs", ASYNC_CLOSURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an unconvertible main is a conversion failure (exit 2), not a process panic; stderr={stderr}"
    );
    assert!(
        stderr.contains("`main` was not converted (body mentions a coroutine-closure"),
        "the reason names the coroutine-closure skip: {stderr}"
    );
    assert!(
        !stderr.contains("conversion panicked"),
        "the failure is a named skip, not a conversion panic: {stderr}"
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
fn run_mode_aux_namespaced_binding_name() {
    // Lane CI: an `--aux` spec name is an `--extern` binding
    // string, not a crate name: `my_api::utils` is a namespaced
    // binding (compiletest's open-ns tests, enabled by the test's
    // own `-Z namespaced-crates` compile-flag). The crate builds
    // under its file stem and binds under the namespaced name;
    // passing the binding to `--crate-name` made rustc reject
    // the build ("invalid character ':' in crate name").
    let dir = temp_dir("aux-namespaced");
    let aux = fixture(
        &dir,
        "open-ns-my_api_utils.rs",
        "pub fn val() -> i32 { 7 }\n",
    );
    let src = fixture(
        &dir,
        "nsmain2.rs",
        "//@ aux-crate:my_api::utils=open-ns-my_api_utils.rs\n\
         fn main() { println!(\"v={}\", my_api::utils::val()); }\n",
    );
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("-Z")
        .arg("--rustc-arg")
        .arg("namespaced-crates")
        .arg("--aux")
        .arg(aux_spec("my_api::utils", &aux, ""))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "namespaced aux binding must build and run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=7\n");
}

#[test]
fn run_mode_aux_commented_crate_name_is_not_a_declaration() {
    // Lane CI: `// #![crate_name = "my_api::core"]` at the top of
    // an auxiliary source is a comment, not a declaration. Reading
    // it as one made the driver search for an artifact under the
    // namespaced name and report "produced no linkable artifact"
    // for a build that had in fact succeeded under the file stem.
    let dir = temp_dir("aux-commented-name");
    let aux = fixture(
        &dir,
        "thing-core.rs",
        "// #![crate_name = \"my_api::core\"]\n\
         pub fn val() -> i32 { 9 }\n",
    );
    let src = fixture(
        &dir,
        "cmain.rs",
        "extern crate thing_core;\n\
         fn main() { println!(\"v={}\", thing_core::val()); }\n",
    );
    let out = run_driver_with_aux_specs(
        &src,
        None,
        &[aux_spec("thing_core", &aux, "")],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "commented crate_name must be ignored; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=9\n");
}

#[test]
fn run_mode_aux_build_name_binding() {
    // Lane CI round 2: a consumer can declare an auxiliary crate
    // by the name the crate's own source declares
    // (`#![crate_name = "buildname"]`), not only by the `--aux`
    // spec name (the svh crates and issue-6919 shape). Binding
    // only the spec name left the `extern crate buildname;` item
    // unresolved (E0463).
    let dir = temp_dir("aux-build-name");
    let aux = fixture(
        &dir,
        "iss-x.rs",
        "#![crate_name = \"buildname\"]\npub fn val() -> i32 { 33 }\n",
    );
    let src = fixture(
        &dir,
        "bmain.rs",
        "extern crate buildname;\n\
         fn main() { println!(\"v={}\", buildname::val()); }\n",
    );
    let out = run_driver_with_aux(&src, &[("iss_x", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "build-name declaration must bind; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=33\n");
}

#[test]
fn run_mode_aux_attribute_prefixed_extern_crate() {
    // Lane CI round 2: an auxiliary source can declare its
    // dependency with an attribute-prefixed item
    // (`#[macro_use] extern crate base;` — the no_link-crate
    // shape). The declaration scan skipped lines not starting
    // with `extern crate`, so the dependency was never bound
    // and the auxiliary build failed with E0463.
    let dir = temp_dir("aux-attr-extern");
    let base = fixture(&dir, "base.rs", "pub fn base_val() -> i32 { 40 }\n");
    let mid = fixture(
        &dir,
        "mid.rs",
        "#[macro_use] extern crate base;\n\
         pub fn with_base() -> i32 { base::base_val() + 2 }\n",
    );
    let src = fixture(
        &dir,
        "amain.rs",
        "extern crate mid;\n\
         fn main() { println!(\"val={}\", mid::with_base()); }\n",
    );
    let out = run_driver_with_aux(&src, &[("base", &base), ("mid", &mid)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "attribute-prefixed extern crate must declare the dependency; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "val=42\n");
}

#[test]
fn run_mode_aux_extra_filename_artifact() {
    // Lane CI round 2: an auxiliary source whose own
    // compile-flags carry `-C extra-filename=-1` (the
    // crateresolve shape) produces `lib<name>-1.rlib`; the
    // artifact search only recognised `lib<name>.rlib` and
    // reported the successful build as producing no linkable
    // artifact.
    let dir = temp_dir("aux-extra-filename");
    let aux = fixture(
        &dir,
        "xflag.rs",
        "//@ compile-flags: -C extra-filename=-1\n\
         pub fn val() -> i32 { 5 }\n",
    );
    let src = fixture(
        &dir,
        "xmain.rs",
        "extern crate xflag;\n\
         fn main() { println!(\"v={}\", xflag::val()); }\n",
    );
    let out = run_driver_with_aux(&src, &[("xflag", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "extra-filename artifact must be found; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=5\n");
}

#[test]
fn run_mode_aux_locator_extern_name() {
    // Lane CI round 2: a caller can declare an auxiliary crate
    // with a locator `--extern` flag (no path) in the forwarded
    // rustc arguments — tests/ui/resolve/extern-prelude.rs
    // passes `--extern Vec` for an auxiliary crate declared
    // `#![crate_name = "Vec"]` and refers to it through the
    // extern prelude with no `extern crate` item. The name
    // counts as a declaration, so the driver binds the built
    // crate under it.
    let dir = temp_dir("aux-locator");
    let aux = fixture(
        &dir,
        "vec-aux.rs",
        "#![crate_name = \"Vec\"]\npub fn new(x: i32) -> i32 { x + 1 }\n",
    );
    let src = fixture(
        &dir,
        "lmain.rs",
        "fn main() { println!(\"v={}\", Vec::new(41)); }\n",
    );
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("--extern")
        .arg("--rustc-arg")
        .arg("Vec")
        .arg("--aux")
        .arg(aux_spec("vec_aux", &aux, ""))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "locator --extern name must bind the built crate; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=42\n");
}

#[test]
fn run_mode_aux_modified_binding_stays_out_of_prelude() {
    // Lane CI round 2: an `aux-crate:` binding with modifiers
    // (`priv,noprelude:somedep`) is the binding itself — the
    // driver must not ALSO bind the crate under its unmodified
    // build name, or the crate lands in the extern prelude the
    // `noprelude` modifier exists to keep it out of, and
    // tests/ui/extern-flag/multiple-opts.rs stops failing the
    // way it must. The `extern crate somedep;` item inside the
    // module still resolves through the modified binding; the
    // prelude use in `main` cannot, so compilation fails
    // (run-mode exit 2).
    let dir = temp_dir("aux-modified-binding");
    let aux = fixture(
        &dir,
        "somedep.rs",
        "pub struct S;\npub fn somefun() {}\n",
    );
    let src = fixture(
        &dir,
        "momain.rs",
        "//@ aux-crate:priv,noprelude:somedep=somedep.rs\n\
         #![warn(exported_private_dependencies)]\n\
         pub mod m {\n\
             extern crate somedep;\n\
             pub struct PublicType { pub field: somedep::S }\n\
         }\n\
         fn main() { somedep::somefun(); }\n",
    );
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("-Z")
        .arg("--rustc-arg")
        .arg("unstable-options")
        .arg("--aux")
        .arg(aux_spec("priv,noprelude:somedep", &aux, ""))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a noprelude binding must keep the crate out of the prelude; stderr={stderr}"
    );
    assert!(
        stderr.contains("cannot find"),
        "the failure is the unresolved prelude use: {stderr}"
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
fn summary_mode_generic_context_closure_is_converted_when_demanded() {
    // A closure inside a generic function has no instance as an
    // item, but the monomorphic instance of the enclosing function
    // (`wrap::<i32>` here) mentions it with concrete arguments, and
    // the closure collection converts the demanded instance.
    let dir = temp_dir("closure-generic");
    let src = fixture(&dir, "closure_generic.rs", CLOSURE_GENERIC_CONTEXT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the closure item");
    assert!(
        line.contains("[converted]"),
        "the demanded generic-context closure instance is converted: {line}"
    );
}

const CLOSURE_GENERIC_UNDEMANDED_SRC: &str = r#"
fn wrap<T: Copy>(v: T) -> T {
    let id = |x: T| x;
    id(v)
}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_undemanded_generic_context_closure_stays_skipped() {
    // The same closure with no monomorphic caller anywhere stays
    // skipped, with the reason naming the generic context.
    let dir = temp_dir("closure-generic-undemanded");
    let src = fixture(
        &dir,
        "closure_generic_undemanded.rs",
        CLOSURE_GENERIC_UNDEMANDED_SRC,
    );
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the closure item");
    assert!(
        line.contains("SKIPPED") && line.contains("generic context"),
        "the undemanded closure stays skipped with the generic-context reason: {line}"
    );
}

const COROUTINE_DEMANDED_SRC: &str = r#"
fn main() {
    let _a = async { 1 };
    let _b = async move { 2 };
}
"#;

#[test]
fn summary_mode_demanded_async_block_is_converted() {
    // A coroutine item (an async block) has no monomorphic
    // instance as an item, but the converted body that creates it
    // mentions it with concrete arguments, and the coroutine
    // collection converts the demanded instance.
    let dir = temp_dir("coroutine-demanded");
    let src = fixture(&dir, "coroutine_demanded.rs", COROUTINE_DEMANDED_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["{closure#0}", "{closure#1}"] {
        let line = stdout
            .lines()
            .find(|l| l.contains(name))
            .expect("summary lists the coroutine item");
        assert!(
            line.contains("[converted]"),
            "the demanded coroutine instance is converted: {line}"
        );
    }
}

const COROUTINE_UNDEMANDED_SRC: &str = r#"
fn wrap<T: Copy>(v: T) -> T {
    let _f = async move { v };
    v
}
fn main() { println!("hi"); }
"#;

#[test]
fn summary_mode_undemanded_coroutine_stays_skipped() {
    // A coroutine inside a generic function that is never
    // instantiated has no concrete demand anywhere and stays
    // skipped: an undemanded generic is never counted converted.
    let dir = temp_dir("coroutine-undemanded");
    let src = fixture(&dir, "coroutine_undemanded.rs", COROUTINE_UNDEMANDED_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("{closure#0}"))
        .expect("summary lists the coroutine item");
    assert!(
        line.contains("SKIPPED") && line.contains("no monomorphic instance"),
        "the undemanded coroutine stays skipped: {line}"
    );
}

const COROUTINE_ONLY_CALLER_SRC: &str = r#"
fn helper<T: Copy>(v: T) -> T { v }
fn main() {
    let _f = async { helper::<i32>(3) };
}
"#;

#[test]
fn summary_mode_generic_fn_called_only_from_coroutine_body_is_converted() {
    // The only call to the generic function lives in the
    // coroutine's body. Converting the coroutine must feed the
    // instance demand too: every converted body is scanned by
    // every demand pass.
    let dir = temp_dir("coroutine-only-caller");
    let src = fixture(&dir, "coroutine_only_caller.rs", COROUTINE_ONLY_CALLER_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("helper"))
        .expect("summary lists the generic function item");
    assert!(
        line.contains("[converted]"),
        "the generic function demanded only by the coroutine body is converted: {line}"
    );
}

#[test]
fn run_mode_generic_context_closure_runs() {
    let dir = temp_dir("closure-generic-run");
    let src = fixture(&dir, "closure_generic.rs", CLOSURE_GENERIC_CONTEXT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure inside a generic function must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

const CLOSURE_GENERIC_TWO_TYPES_SRC: &str = r#"
fn twice<T: Copy + std::ops::Add<Output = T>>(v: T) -> T {
    let f = |x: T| x + x;
    f(v)
}
fn main() {
    println!("{}", twice::<i32>(21));
    println!("{}", twice::<f64>(1.5));
}
"#;

#[test]
fn run_mode_generic_closure_two_instantiations() {
    // Two monomorphic instances of one closure definition coexist
    // in the program and each call reaches its own body.
    let dir = temp_dir("closure-generic-two-types");
    let src = fixture(
        &dir,
        "closure_generic_two_types.rs",
        CLOSURE_GENERIC_TWO_TYPES_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "both instantiations of the generic-context closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n3\n");
}

const GENERIC_HOF_FNONCE_SRC: &str = r#"
fn apply<F: FnOnce() -> i32>(f: F) -> i32 { f() }
fn main() {
    let base = 40;
    let r = apply(|| base + 2);
    println!("{r}");
}
"#;

#[test]
fn run_mode_generic_hof_fnonce_with_closure() {
    // A generic higher-order function applied to a closure: the
    // `FnOnce::call_once` call inside the generic body resolves to
    // the closure instance the caller monomorphized it with.
    let dir = temp_dir("generic-hof-fnonce");
    let src = fixture(&dir, "generic_hof_fnonce.rs", GENERIC_HOF_FNONCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic HOF applied to a closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
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

// ----- the String / str surface on the abstract string value ------------
//
// `String` and `str` are the string data itself in the interpreter's
// value model, so the std bodies (which project the real struct's
// buffer/length/capacity fields) cannot run on them; the methods
// are runtime support instead. These fixtures pin the observable
// behaviour against native semantics.

const STRING_LEN_SRC: &str = r#"
fn main() {
    let s = String::from("héllo");
    println!("{}", s.len());
    println!("{}", s.is_empty());
    let e = String::new();
    println!("{}", e.is_empty());
    let r: &str = "héllo";
    println!("{}", r.len());
    println!("{}", r.is_empty());
}
"#;

#[test]
fn run_mode_string_len_is_byte_length() {
    let dir = temp_dir("string-len");
    let src = fixture(&dir, "string_len.rs", STRING_LEN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // `len` is a byte count on both String and str: "héllo" is
    // 6 bytes (é is two), not 5 chars.
    assert_eq!(String::from_utf8_lossy(&out.stdout), "6\nfalse\ntrue\n6\nfalse\n");
}

const STRING_INTO_BYTES_SRC: &str = r#"
fn main() {
    let s = String::from("abc");
    let mut b = s.into_bytes();
    println!("{}", b.len());
    println!("{}", b[1]);
    b.push(100);
    println!("{}", b.len());
    println!("{}", b[3]);
}
"#;

#[test]
fn run_mode_string_into_bytes_yields_a_real_vec() {
    let dir = temp_dir("string-into-bytes");
    let src = fixture(&dir, "string_into_bytes.rs", STRING_INTO_BYTES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The result is a genuine Vec<u8> value: it has a length, its
    // elements index, and it grows through Vec::push.
    assert_eq!(String::from_utf8_lossy(&out.stdout), "3\n98\n4\n100\n");
}

const STRING_PUSH_SRC: &str = r#"
fn main() {
    let mut s = String::new();
    s.push_str("ab");
    s.push('é');
    s.push_str("cd");
    println!("{}", s);
    println!("{}", s.len());
}
"#;

#[test]
fn run_mode_string_push_str_and_push_mutate_in_place() {
    let dir = temp_dir("string-push");
    let src = fixture(&dir, "string_push.rs", STRING_PUSH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "abécd\n6\n");
}

const STRING_VIEWS_SRC: &str = r#"
fn main() {
    let s = String::from("abc");
    let b = s.as_bytes();
    println!("{}", b.len());
    println!("{}", b[0]);
    let r: &str = s.as_str();
    println!("{}", r);
    let t: &str = "abc";
    println!("{}", t.as_bytes()[2]);
}
"#;

#[test]
fn run_mode_string_as_bytes_and_as_str() {
    let dir = temp_dir("string-views");
    let src = fixture(&dir, "string_views.rs", STRING_VIEWS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "3\n97\nabc\n99\n");
}

const STR_TRIM_CHARS_SRC: &str = r#"
fn main() {
    let s = String::from("  hi  ");
    println!("[{}]", s.trim());
    println!("[{}]", s.trim_start());
    println!("[{}]", s.trim_end());
    let mut n = 0;
    for _c in "aébc".chars() { n += 1; }
    println!("{}", n);
    let mut m = 0;
    for _b in "aé".bytes() { m += 1; }
    println!("{}", m);
    let mut it = "éx".chars();
    println!("{}", it.next().unwrap());
}
"#;

#[test]
fn run_mode_str_trim_and_char_byte_iterators() {
    let dir = temp_dir("str-trim-chars");
    let src = fixture(&dir, "str_trim_chars.rs", STR_TRIM_CHARS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // chars() counts/iterates chars (aébc = 4), bytes() iterates
    // the UTF-8 bytes (aé = 3).
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[hi]\n[hi  ]\n[  hi]\n4\n3\né\n"
    );
}

const STRING_CAPACITY_SRC: &str = r#"
fn main() {
    let s = String::from("abc");
    println!("{}", s.capacity());
}
"#;

#[test]
fn run_mode_string_capacity_is_an_explicit_unsupported() {
    // Allocation capacity is not tracked by the abstract string
    // model; the call must fail as unsupported naming the method,
    // not as a projection failure inside the collected std body.
    let dir = temp_dir("string-capacity");
    let src = fixture(&dir, "string_capacity.rs", STRING_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("std::string::String::capacity"),
        "the error names the method: {stderr}"
    );
}

const CONST_CLOSURE_FN_PTR_SRC: &str = r#"
const F: fn() -> i32 = || 42;
fn main() { println!("{}", F()); }
"#;

const STATIC_CLOSURE_FN_PTR_SRC: &str = r#"
static F: fn() -> i32 = || 42;
fn main() { println!("{}", F()); }
"#;

const DIRECT_CLOSURE_FN_PTR_SRC: &str = r#"
fn main() {
    let f: fn() -> i32 = || 42;
    println!("{}", f());
}
"#;

const CONST_CLOSURE_FN_PTR_ARGS_SRC: &str = r#"
const F: fn(i32) -> i32 = |x| x + 1;
fn main() { println!("{}", F(41)); }
"#;

const CONST_FN_ITEM_SRC: &str = r#"
fn foo() -> i32 { 42 }
const F: fn() -> i32 = foo;
fn main() { println!("{}", F()); }
"#;

fn run_expect_42(tag: &str, name: &str, src: &str) {
    let dir = temp_dir(tag);
    let path = fixture(&dir, name, src);
    let out = run_driver(&path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "callable constant must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "42\n",
        "output must match the native program"
    );
}

#[test]
fn run_mode_const_coerced_closure_fn_ptr() {
    run_expect_42("const-closure", "const_closure.rs", CONST_CLOSURE_FN_PTR_SRC);
}

#[test]
fn run_mode_static_coerced_closure_fn_ptr() {
    run_expect_42("static-closure", "static_closure.rs", STATIC_CLOSURE_FN_PTR_SRC);
}

#[test]
fn run_mode_direct_closure_fn_ptr_cast() {
    run_expect_42("direct-closure", "direct_closure.rs", DIRECT_CLOSURE_FN_PTR_SRC);
}

#[test]
fn run_mode_const_closure_fn_ptr_with_args() {
    run_expect_42("const-closure-args", "const_closure_args.rs", CONST_CLOSURE_FN_PTR_ARGS_SRC);
}

#[test]
fn run_mode_const_fn_item() {
    run_expect_42("const-fn-item", "const_fn_item.rs", CONST_FN_ITEM_SRC);
}

// ----- Vec allocation paths (Lane BO) -----

const VEC_DROPPABLE_SRC: &str = r#"
struct D(i32);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let v = vec![D(3), D(4)];
    println!("len {}", v.len());
    drop(v);
    println!("done");
}
"#;

#[test]
fn run_mode_vec_macro_of_droppables() {
    // The `vec![...]` list form builds the vector through
    // `Box::new_uninit` + `box_assume_init_into_vec_unsafe`, whose
    // real bodies bottom out in the runtime allocator
    // (`Global::alloc_impl_runtime` has no body). The interpreter
    // models those entry points, so a Vec of droppable elements
    // runs end to end and the element destructors run in order,
    // byte for byte as natively.
    let dir = temp_dir("vecdroppable");
    let src = fixture(&dir, "vecdroppable.rs", VEC_DROPPABLE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec! of droppables must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2\ndrop 3\ndrop 4\ndone\n",
        "vec! elements drop in order, exactly as natively"
    );
}

const VEC_GROWTH_SRC: &str = r#"
struct D(i32);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
    }
}
fn main() {
    let mut v: Vec<D> = Vec::with_capacity(4);
    v.reserve(8);
    v.push(D(1));
    v.push(D(2));
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_with_capacity_push_and_reserve() {
    // `with_capacity` and `reserve` are allocation entry points:
    // the abstract Vec model does not represent capacity, so both
    // are modelled (an empty vector / a no-op) and pushes onto the
    // reserved vector behave as natively, including the scope-end
    // drop of the elements.
    let dir = temp_dir("vecgrowth");
    let src = fixture(&dir, "vecgrowth.rs", VEC_GROWTH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "with_capacity/reserve/push must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2\ndrop 1\ndrop 2\n",
        "reserved Vec pushes and drops exactly as natively"
    );
}


const STRING_WITH_CAPACITY_SRC: &str = r#"
fn main() {
    let mut s = String::with_capacity(16);
    s.push_str("hi");
    println!("{} {}", s, s.len());
}
"#;

#[test]
fn run_mode_string_with_capacity_push_str_and_len() {
    // `String::with_capacity` is a direct `RawVecInner` caller:
    // its collected body inlines down to
    // `RawVecInner::with_capacity_in`, which has no converted
    // body, so the constructor is modelled as the empty string
    // (the requested capacity is not represented, as for
    // `Vec::with_capacity`).
    let dir = temp_dir("stringwithcapacity");
    let src = fixture(&dir, "stringwithcapacity.rs", STRING_WITH_CAPACITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "String::with_capacity must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hi 2\n");
}

const VEC_TRY_RESERVE_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    println!("small {}", v.try_reserve(10).is_ok());
    println!("huge {}", v.try_reserve(usize::MAX).is_ok());
    v.push(4);
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_try_reserve_verdicts() {
    // Allocation never fails in the abstract model, so the only
    // error `try_reserve` can produce is capacity overflow:
    // 3 + 10 elements fit easily (Ok), while `usize::MAX` more
    // elements of 4 bytes exceed the `isize::MAX`-byte limit
    // (Err), exactly as natively.
    let dir = temp_dir("vectryreserve");
    let src = fixture(&dir, "vectryreserve.rs", VEC_TRY_RESERVE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::try_reserve must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "small true\nhuge false\nlen 4\n"
    );
}

const VEC_SHRINK_SRC: &str = r#"
fn main() {
    let mut v = vec![1, 2, 3];
    v.shrink_to_fit();
    v.shrink_to(2);
    println!("len {}", v.len());
    v.push(4);
    println!("len {}", v.len());
}
"#;

#[test]
fn run_mode_vec_shrink_to_fit_and_shrink_to() {
    // Shrinking moves only the capacity, which the abstract Vec
    // model does not represent: length, elements and later
    // pushes are exactly as natively.
    let dir = temp_dir("vecshrink");
    let src = fixture(&dir, "vecshrink.rs", VEC_SHRINK_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Vec::shrink_to_fit/shrink_to must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "len 3\nlen 4\n");
}

const REF_VEC_ITER_SRC: &str = r#"
fn main() {
    let v = vec![10, 20, 30];
    let mut sum = 0;
    for x in &v {
        sum += *x;
    }
    println!("{}", sum);
}
"#;

#[test]
fn run_mode_ref_vec_iteration() {
    // `for x in &v` goes through `IntoIterator::into_iter` for
    // `&Vec<T>`, which yields a `slice::Iter` over the elements;
    // its items are `&T`, dereferenced by the loop body.
    let dir = temp_dir("refveciter");
    let src = fixture(&dir, "refveciter.rs", REF_VEC_ITER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "iterating &Vec must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "60\n");
}

const VEC_REPEAT_SRC: &str = r#"
fn main() {
    let v = vec![7; 4];
    println!("len {} sum {}", v.len(), v[0] + v[3]);
    let w = vec![String::from("ab"); 2];
    println!("{} {}", w.len(), w[1]);
}
"#;

#[test]
fn run_mode_vec_repeat_form() {
    // The `vec![x; n]` repeat form goes through
    // `std::vec::from_elem`, another allocation entry point the
    // interpreter models.
    let dir = temp_dir("vecrepeat");
    let src = fixture(&dir, "vecrepeat.rs", VEC_REPEAT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "vec![x; n] must terminate normally; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 4 sum 14\n2 ab\n",
        "repeat-form Vec behaves exactly as natively"
    );
}

// ----- trait-declared method keys (Lane BV, Issue #64) -----

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

const DYN_DEFAULT_SRC: &str = r#"
trait Foo {
    fn a(&self) -> i32 { 100 }
    fn b(&self) -> i32 { self.a() + 1 }
}
impl Foo for i32 {}
trait Greeter {
    fn name(&self) -> i32;
    fn greet(&self) -> i32 { self.name() + 1 }
}
struct S;
impl Greeter for S {
    fn name(&self) -> i32 { 41 }
}
fn main() {
    let f: &dyn Foo = &1i32;
    assert_eq!(f.a(), 100);
    assert_eq!(f.b(), 101);
    let g: Box<dyn Foo> = Box::new(2i32);
    assert_eq!(g.a(), 100);
    assert_eq!(g.b(), 101);
    let h: &dyn Greeter = &S;
    assert_eq!(h.greet(), 42);
    let h2: Box<dyn Greeter> = Box::new(S);
    assert_eq!(h2.greet(), 42);
    println!("dyn-default ok");
}
"#;

#[test]
fn run_mode_dyn_trait_default_method() {
    // A trait default method called through a `dyn` receiver: the
    // dispatch target is the trait's own body instantiated with
    // the concrete `Self`, demanded from the vtable the unsizing
    // cast forms (Lane EU). Both `&dyn` and `Box<dyn>` receivers,
    // a default calling another default, and a default calling a
    // required method the impl provides.
    let dir = temp_dir("dyn-default");
    let src = fixture(&dir, "dyn_default.rs", DYN_DEFAULT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dyn default methods must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-default ok\n");
}

#[test]
fn summary_mode_dyn_trait_default_method_is_converted() {
    // The demanded vtable instance also settles the trait
    // method's own accounting: it no longer stands as a skipped
    // "dynamic dispatch target (virtual instance)".
    let dir = temp_dir("dyn-default-summary");
    let src = fixture(&dir, "dyn_default.rs", DYN_DEFAULT_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    for name in ["::Foo::a:", "::Foo::b:", "::Greeter::greet:"] {
        let line = stdout
            .lines()
            .find(|l| l.contains(name))
            .unwrap_or_else(|| panic!("summary lists {name}: {stdout}"));
        assert!(
            line.contains("[converted]"),
            "the trait default method {name} is converted: {line}"
        );
    }
    assert!(
        !stdout.contains("target-crate unsupported:"),
        "no target-crate function stays skipped: {stdout}"
    );
}

const DYN_DEFAULT_OVERRIDE_SRC: &str = r#"
trait D {
    fn val(&self) -> u32 { 10 }
    fn extra(&self) -> u32 { self.val() + 1 }
}
struct X;
impl D for X {}
struct Y;
impl D for Y {
    fn val(&self) -> u32 { 20 }
}
fn main() {
    let x: &dyn D = &X;
    let y: &dyn D = &Y;
    println!("{} {} {} {}", x.val(), x.extra(), y.val(), y.extra());
    let bx: Box<dyn D> = Box::new(X);
    let by: Box<dyn D> = Box::new(Y);
    println!("{} {} {} {}", bx.val(), bx.extra(), by.val(), by.extra());
}
"#;

#[test]
fn run_mode_dyn_trait_default_method_override_wins() {
    // An impl that overrides the method is what a dyn call must
    // reach — never the default — and the default `extra` body
    // must dispatch its inner `self.val()` to the override for
    // `Y` and to the default for `X`. This also pins the dispatch
    // precedence: the statically demanded default instance of
    // `val` (from `extra`'s inner call) sits in the program under
    // the plain trait name, and a dyn call must not be served by
    // it directly.
    let dir = temp_dir("dyn-default-override");
    let src = fixture(&dir, "dyn_default_override.rs", DYN_DEFAULT_OVERRIDE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dyn default/override dispatch must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "10 11 20 21\n10 11 20 21\n"
    );
}

const DYN_DEFAULT_SUPER_SRC: &str = r#"
trait Super {
    fn s(&self) -> i32 { 1 }
    fn s2(&self) -> i32;
}
trait Sub: Super {
    fn b(&self) -> i32 { 2 }
}
struct S;
impl Super for S {
    fn s2(&self) -> i32 { 3 }
}
impl Sub for S {}
fn main() {
    let d: &dyn Sub = &S;
    println!("{} {} {}", d.b(), d.s(), d.s2());
    let u: &dyn Super = d;
    println!("{} {}", u.s(), u.s2());
}
"#;

#[test]
fn run_mode_dyn_trait_default_method_supertrait_upcast() {
    // Supertrait default methods are vtable entries of the
    // supertrait vtable (`TraitVPtr`): calling them through the
    // subtrait object and through an upcast `&dyn Super` both
    // reach the default body.
    let dir = temp_dir("dyn-default-super");
    let src = fixture(&dir, "dyn_default_super.rs", DYN_DEFAULT_SUPER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "supertrait dyn defaults must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "2 1 3\n1 3\n");
}

const DYN_DEFAULT_GENERIC_TRAIT_SRC: &str = r#"
trait G<T> {
    fn get(&self, t: T) -> T { t }
}
struct S;
impl G<i32> for S {}
fn main() {
    let d: &dyn G<i32> = &S;
    println!("{}", d.get(41) + 1);
}
"#;

#[test]
fn run_mode_dyn_trait_default_method_generic_trait() {
    // A trait with generic arguments: the instance name carries
    // them (`<S as G<i32>>::get`) while the dyn call name does
    // not, so the vtable lookup resolves by its unique scan.
    let dir = temp_dir("dyn-default-generic-trait");
    let src = fixture(&dir, "dyn_default_generic.rs", DYN_DEFAULT_GENERIC_TRAIT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "generic-trait dyn defaults must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

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

// ----- anonymous constants (Lane BW) -----

const ANON_CONST_GENERIC_SRC: &str = r#"
fn f<T>(x: T) -> usize {
    let a = [0u32; 1 + 2];
    a.len() + std::mem::size_of::<T>()
}
fn main() {
    println!("{}", f::<u8>(0));
}
"#;

#[test]
fn summary_mode_generic_context_anon_consts_are_converted() {
    // An anonymous constant inside a generic function requires
    // monomorphization as an item, so it has no instance; its own
    // MIR body is converted instead, and the target crate reports
    // no skipped functions.
    let dir = temp_dir("anon-const-generic");
    let src = fixture(&dir, "anon_const_generic.rs", ANON_CONST_GENERIC_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let konst = stdout
        .lines()
        .find(|l| l.contains("::{constant#0}:"))
        .expect("summary lists the anonymous constant");
    assert!(
        konst.contains("[converted]"),
        "the generic-context anonymous constant is converted: {konst}"
    );
    assert!(
        stdout.contains("target-crate converted: 3, skipped: 0"),
        "no target-crate function is skipped: {stdout}"
    );
}

#[test]
fn run_mode_generic_context_anon_const_executes() {
    let dir = temp_dir("anon-const-generic-run");
    let src = fixture(&dir, "anon_const_generic.rs", ANON_CONST_GENERIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "4\n");
}

const TRIVIAL_CONST_SRC: &str = r#"
const N: usize = 3;
static S: [u32; 3] = [4; 1 + 2];
fn main() {
    let a = [0u32; N];
    println!("{}", a.len() + S.len());
}
"#;

#[test]
fn summary_mode_trivial_consts_are_converted_without_bodies() {
    // Trivial constants (a literal, a path) have no MIR body
    // anywhere by design; they are accounted as converted with no
    // body instead of being reported as missing bodies, while the
    // non-trivial constant of the static initializer keeps its
    // real body.
    let dir = temp_dir("trivial-consts");
    let src = fixture(&dir, "trivial_consts.rs", TRIVIAL_CONST_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let named = stdout
        .lines()
        .find(|l| l.starts_with("fn trivial_consts::N:"))
        .expect("summary lists the named constant");
    assert!(
        named.contains("blocks=0") && named.contains("[converted]"),
        "the trivial named constant is converted with no body: {named}"
    );
    let trivial = stdout
        .lines()
        .find(|l| l.contains("::S::{constant#0}:"))
        .expect("summary lists the static type-length constant");
    assert!(
        trivial.contains("blocks=0") && trivial.contains("[converted]"),
        "the trivial type-length constant is converted with no body: {trivial}"
    );
    let real = stdout
        .lines()
        .find(|l| l.contains("::S::{constant#1}:"))
        .expect("summary lists the static initializer constant");
    assert!(
        real.contains("[converted]") && !real.contains("blocks=0"),
        "the non-trivial constant keeps its converted body: {real}"
    );
    assert!(
        stdout.contains("target-crate skipped: 0")
            || stdout.contains("skipped: 0"),
        "no target-crate function is skipped: {stdout}"
    );
}

// ----- external-demand instances (Lane BX) -----

const EXTERNAL_DEMAND_SRC: &str = r#"
struct RepeatN<T> { v: T, n: usize }
impl<T: Copy> Iterator for RepeatN<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.n == 0 { None } else { self.n -= 1; Some(self.v) }
    }
}

struct Total(i32);
impl std::iter::Sum<i32> for Total {
    fn sum<I: Iterator<Item = i32>>(mut iter: I) -> Self {
        let mut acc = 0i32;
        while let Some(x) = iter.next() {
            acc += x;
        }
        Total(acc)
    }
}

fn unused<T>(x: T) -> T { x }

fn main() {
    let t: Total = RepeatN { v: 7i32, n: 3 }.sum();
    println!("{}", t.0);
}
"#;

#[test]
fn run_mode_generic_instance_demanded_through_foreign_body_executes() {
    // Neither generic definition is named by a local body:
    // `Total::sum` (generic over the iterator type) is demanded
    // only from the collected body of `Iterator::sum`, and
    // `RepeatN::next` (a method of a generic impl) only from
    // `Total::sum`'s instance body. Both instances are resolved
    // and converted, so the program runs end to end.
    let dir = temp_dir("external-demand");
    let src = fixture(&dir, "external_demand.rs", EXTERNAL_DEMAND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "externally demanded generic instances must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "21\n");
}

#[test]
fn summary_mode_externally_demanded_instances_are_converted() {
    // The definitions demanded through foreign bodies are
    // reported converted; a definition nothing demands keeps its
    // no-instance record.
    let dir = temp_dir("external-demand-summary");
    let src = fixture(&dir, "external_demand.rs", EXTERNAL_DEMAND_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let sum = stdout
        .lines()
        .find(|l| l.contains("::sum:") && l.contains("Sum<i32>"))
        .expect("summary lists the Sum impl method");
    assert!(
        sum.contains("[converted]"),
        "the externally demanded generic method is converted: {sum}"
    );
    let next = stdout
        .lines()
        .find(|l| l.contains("RepeatN") && l.contains(">::next:"))
        .expect("summary lists the Iterator impl method");
    assert!(
        next.contains("[converted]"),
        "the transitively demanded generic method is converted: {next}"
    );
    let unused = stdout
        .lines()
        .find(|l| l.starts_with("fn external_demand::unused:"))
        .expect("summary lists the unused generic definition");
    assert!(
        unused.contains("SKIPPED (no monomorphic instance available (kind=Fn)"),
        "the undemanded generic definition stays explicitly skipped: {unused}"
    );
}

// ----- dst-index impl resolution (Lane BY) -----

const DST_INDEX_SRC: &str = r#"
use std::fmt::Debug;
use std::ops::Index;

struct S;

impl Index<usize> for S {
    type Output = str;

    fn index(&self, _: usize) -> &str {
        "hello"
    }
}

struct T;

impl Index<usize> for T {
    type Output = dyn Debug + 'static;

    fn index(&self, _: usize) -> &(dyn Debug + 'static) {
        static X: usize = 42;
        &X as &(dyn Debug + 'static)
    }
}

fn main() {
    assert_eq!(&S[0], "hello");
    let _ = &T[0];
    println!("dst index ok");
}
"#;

#[test]
fn run_mode_index_impls_with_dst_outputs_execute() {
    // The shape of tests/ui/dst/dst-index-success.rs (Issue #65):
    // the Index calls name the trait item, whose name-based builtin
    // model indexed the receiver struct itself and panicked with a
    // bogus out-of-bounds message; the collected impl bodies must
    // run instead, for both the `str` and the `dyn Debug` outputs.
    let dir = temp_dir("dstindex");
    let src = fixture(&dir, "dstindex.rs", DST_INDEX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dst index program must terminate normally; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dst index ok\n");
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

// ----- generic constructors (Lane CF) -----
//
// A generic tuple-struct / tuple-variant constructor has no
// monomorphic instance as an item. Its direct uses are aggregate
// rvalues in the callers, so nothing is left unconverted; used as
// a value, the instance its FnDef constant demands is converted
// (the synthetic one-block body that aggregates the fields) and
// the call through the value runs it.

const CTOR_VALUE_SRC: &str = r#"
struct Pair<T>(T, i32);
fn main() {
    let f = Pair;
    let p = f(3, 4);
    println!("{}", p.0 + p.1);
}
"#;

#[test]
fn run_mode_generic_ctor_as_value_executes() {
    let dir = temp_dir("ctor-value");
    let src = fixture(&dir, "ctor_value.rs", CTOR_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic constructor used as a value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

const CTOR_HOF_SRC: &str = r#"
struct Wrap<T>(T);
fn apply<F: Fn(i32) -> Wrap<i32>>(f: F) -> i32 { f(41).0 + 1 }
fn main() {
    println!("{}", apply(Wrap));
}
"#;

#[test]
fn run_mode_generic_ctor_passed_to_hof_executes() {
    let dir = temp_dir("ctor-hof");
    let src = fixture(&dir, "ctor_hof.rs", CTOR_HOF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic constructor passed as a callback must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

const CTOR_ENUM_VALUE_SRC: &str = r#"
enum E<T> { V(T, i32), W }
fn main() {
    let f = E::V;
    let e = f(5, 6);
    match e { E::V(a, b) => println!("{}", a + b), E::W => println!("w") }
}
"#;

#[test]
fn run_mode_generic_enum_variant_ctor_as_value_executes() {
    let dir = temp_dir("ctor-enum-value");
    let src = fixture(&dir, "ctor_enum_value.rs", CTOR_ENUM_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic variant constructor used as a value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "11\n");
}

const CTOR_TWO_INSTANCES_SRC: &str = r#"
struct G<T>(T, i32);
fn main() {
    let fi = G::<i32>;
    let fs = G::<&str>;
    let a = fi(1, 2);
    let b = fs("x", 3);
    println!("{} {} {}", a.0 + a.1, b.0, b.1);
}
"#;

#[test]
fn run_mode_generic_ctor_two_instances_as_values_execute() {
    let dir = temp_dir("ctor-two-instances");
    let src = fixture(&dir, "ctor_two_instances.rs", CTOR_TWO_INSTANCES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "two instances of one generic constructor must both execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "3 x 3\n");
}

const CTOR_DIRECT_ONLY_SRC: &str = r#"
struct G<T>(T, i32);
fn main() {
    let g = G(10, 2);
    println!("{}", g.0 + g.1);
}
"#;

#[test]
fn summary_mode_generic_ctor_is_accounted_converted() {
    // Used directly (aggregate rvalues in main) or as a value
    // (demanded instance converted), a generic constructor is
    // reported converted — not as a missing monomorphic instance.
    let dir = temp_dir("ctor-direct-only");
    let src = fixture(&dir, "ctor_direct_only.rs", CTOR_DIRECT_ONLY_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let g = stdout
        .lines()
        .find(|l| l.starts_with("fn ctor_direct_only::G:"))
        .expect("summary lists the generic constructor");
    assert!(
        g.contains("[converted]"),
        "an undemanded generic constructor is converted without a body: {g}"
    );

    let dir = temp_dir("ctor-value-summary");
    let src = fixture(&dir, "ctor_value.rs", CTOR_VALUE_SRC);
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let pair = stdout
        .lines()
        .find(|l| l.starts_with("fn ctor_value::Pair:"))
        .expect("summary lists the generic constructor");
    assert!(
        pair.contains("[converted]"),
        "a demanded generic constructor is converted: {pair}"
    );
    assert!(
        !stdout.contains("kind=Ctor"),
        "no constructor is left as a missing instance: {stdout}"
    );
}

// ----- monomorphization validity checks (Lane CA) ---------------------------

const SIMD_SCALAR_SRC: &str = r#"
#![feature(core_intrinsics)]
use std::intrinsics::simd::simd_add;
fn main() { unsafe { let _ = simd_add(0, 1); } }
"#;

const SIMD_VECTOR_SRC: &str = r#"
#![feature(core_intrinsics, repr_simd)]
use std::intrinsics::simd::simd_add;
#[repr(simd)]
#[derive(Copy, Clone)]
struct I32x4([i32; 4]);
fn main() { unsafe { let _ = simd_add(I32x4([1, 2, 3, 4]), I32x4([5, 6, 7, 8])); } }
"#;

const SIMD_TOO_MANY_LANES_SRC: &str = r#"
#![feature(portable_simd)]
use std::simd::Simd;
fn main() { let _x = Simd::<u16, 54321>::splat(0); }
"#;

#[test]
fn convert_mode_simd_intrinsic_with_scalar_args_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-scalar");
    let src = fixture(&dir, "simd_scalar.rs", SIMD_SCALAR_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an invalid SIMD monomorphization must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("simd_add")
            && stderr.contains("expected SIMD input type"),
        "the failure must name the intrinsic and the reason; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_intrinsic_with_vector_args_converts() {
    let dir = temp_dir("mono-simd-vector");
    let src = fixture(&dir, "simd_vector.rs", SIMD_VECTOR_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a valid SIMD monomorphization must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_simd_type_with_too_many_lanes_is_a_compile_failure() {
    let dir = temp_dir("mono-simd-lanes");
    let src = fixture(&dir, "simd_lanes.rs", SIMD_TOO_MANY_LANES_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a SIMD type over the lane limit must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("invalid number of lanes"),
        "the failure must name the reason; stderr={stderr}"
    );
}

// ----- ADT layout table (Lane CG) -----

const LAYOUT_SRC: &str = r#"
#[repr(C)]
struct Padded { a: u8, b: u32, c: u16 }
#[repr(packed)]
struct Packed { a: u8, b: u32 }
enum E { A, B(u16), C(u64) }
struct Outer { x: u8, inner: Padded }
struct Pair<A, B> { a: A, b: B }
fn main() {
    println!("padded {} {}", std::mem::size_of::<Padded>(), std::mem::align_of::<Padded>());
    println!("packed {} {}", std::mem::size_of::<Packed>(), std::mem::align_of::<Packed>());
    println!("enum {} {}", std::mem::size_of::<E>(), std::mem::align_of::<E>());
    println!("outer {} {}", std::mem::size_of::<Outer>(), std::mem::align_of::<Outer>());
    println!("pair {} {}", std::mem::size_of::<Pair<u8, u64>>(), std::mem::align_of::<Pair<u8, u64>>());
    let v = Padded { a: 1, b: 2, c: 3 };
    println!("val {}", std::mem::size_of_val(&v));
}
"#;

const PACKED_DERIVE_SRC: &str = r#"
use std::mem;

#[derive(Copy, Clone)]
struct Aligned(usize);

#[inline(never)]
fn check_align(ptr: *const Aligned) {
    assert_eq!(ptr as usize % mem::align_of::<Aligned>(), 0);
}

impl PartialEq for Aligned {
    fn eq(&self, other: &Self) -> bool {
        check_align(self);
        check_align(other);
        self.0 == other.0
    }
}

#[repr(packed)]
#[derive(Copy, Clone, PartialEq)]
struct Packed(Aligned, Aligned);

#[derive(PartialEq)]
#[repr(C)]
struct Dealigned<T>(u8, T);

fn main() {
    let d1 = Dealigned(0, Packed(Aligned(1), Aligned(2)));
    let ck = d1 == d1;
    assert!(ck);
    println!("eq-ok");
}
"#;

#[test]
fn run_mode_size_of_align_of_user_adts() {
    // The values are the compiler's own layouts (verified against a
    // native 1.100.0 build of the same source): the driver records
    // them per monomorphic ADT instance and the mem builtins answer
    // from that table.
    let dir = temp_dir("adt-layout");
    let src = fixture(&dir, "layout.rs", LAYOUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_of/align_of of user ADTs run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "padded 12 4\npacked 5 1\nenum 16 8\nouter 16 4\npair 16 8\nval 12\n",
        "layouts match the native ones"
    );
}

#[test]
fn run_mode_derived_eq_on_packed_struct_uses_align_of() {
    let dir = temp_dir("packed-derive");
    let src = fixture(&dir, "packed_derive.rs", PACKED_DERIVE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived eq over a packed struct runs (align_of::<Aligned> resolves); stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "eq-ok\n");
}

// ----- type-mismatch bucket fixes (Lane CD) -----

const BLANKET_FN_SRC: &str = r#"
fn a<F: Fn() -> i32>(f: F) -> i32 { f() }
fn b(f: &dyn Fn() -> i32) -> i32 { a(f) }
fn c<F: Fn() -> i32>(f: &F) -> i32 { a(f) }
fn main() {
    assert_eq!(b(&|| 22), 22);
    assert_eq!(c(&|| 23), 23);
    println!("blanket fn ok");
}
"#;

const BOX_UNSIZE_SRC: &str = r#"
fn main() {
    let xs: Box<[()]> = Box::<[(); 0]>::new([]);
    assert_eq!(xs.len(), 0);
    let ys: Box<[i32]> = Box::new([1, 2, 3]);
    assert_eq!(ys.len(), 3);
    assert_eq!(&*ys, &[1, 2, 3]);
    println!("unsize ok");
}
"#;

const BYTES_MATCH_SRC: &str = r#"
fn main() {
    match *b"abc" {
        [b'a', b'b', b'c'] => println!("bytes ok"),
        _ => panic!("no"),
    }
    match *b"abd" {
        [b'a', b'b', b'c'] => panic!("wrong arm"),
        _ => println!("bytes ok 2"),
    }
}
"#;

const FLOAT_EQ_SRC: &str = r#"
#[derive(PartialEq, Debug)]
struct P { x: f64, y: i32 }
#[derive(PartialEq, Debug)]
struct Pair(u32, u32);
fn main() {
    let a = P { x: 1.5, y: 2 };
    let b = P { x: 1.5, y: 2 };
    assert!(a == b);
    let nan = P { x: f64::NAN, y: 0 };
    assert!(nan != nan);
    assert!(Pair(3, 4) == Pair(3, 4));
    assert!(Pair(3, 4) != Pair(3, 5));
    let t1 = (1.25f32, 7u8);
    let t2 = (1.25f32, 7u8);
    assert!(t1 == t2);
    println!("float eq ok");
}
"#;

const PLACE_SLICE_META_SRC: &str = r#"
static C: &[i32] = &[10, 20, 30];
fn main() {
    assert_eq!(C.len(), 3);
    assert_eq!(C[1], 20);
    let ys: Box<[i32]> = Box::new([1, 2, 3]);
    let s: &[i32] = &*ys;
    assert_eq!(s.len(), 3);
    assert_eq!(s[1], 2);
    println!("meta ok");
}
"#;

const PTR_EQ_SRC: &str = r#"
fn f() {}
fn g() {}
fn main() {
    let x = 5;
    let p = &x as *const i32;
    let q = &x as *const i32;
    assert!(p == q);
    assert!(!p.is_null());
    let fp: fn() = f;
    let fq: fn() = f;
    let fg: fn() = g;
    assert!(fp == fq);
    assert!(fp != fg);
    let a = &x;
    let b = &x;
    assert!(std::ptr::eq(a, b));
    println!("ptr eq ok");
}
"#;

const RANGE_INDEX_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3, 4];
    assert_eq!(&a[1..3], &[2, 3]);
    assert_eq!(&a[..2], &[1, 2]);
    assert_eq!(&a[2..], &[3, 4]);
    assert_eq!(&a[..], &[1, 2, 3, 4]);
    assert_eq!(&a[1..=2], &[2, 3]);
    let st = "héllo";
    assert_eq!(&st[0..1], "h");
    assert_eq!(&st[1..3], "é");
    println!("ranges ok");
}
"#;

const RANGE_PANIC_SRC: &str = r#"
fn main() {
    let a = [1, 2, 3, 4];
    let _s = &a[2..9];
    println!("unreachable");
}
"#;

#[test]
fn run_mode_blanket_fn_impls_for_references() {
    let dir = temp_dir("blanket-fn");
    let src = fixture(&dir, "blanket_fn.rs", BLANKET_FN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "calling through &F / &dyn Fn blanket impls must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "blanket fn ok\n");
}

#[test]
fn run_mode_boxed_slice_unsize() {
    let dir = temp_dir("box-unsize");
    let src = fixture(&dir, "box_unsize.rs", BOX_UNSIZE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Box<[T; N]> to Box<[T]> unsizing must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "unsize ok\n");
}

#[test]
fn run_mode_byte_string_pattern_match() {
    let dir = temp_dir("bytes-match");
    let src = fixture(&dir, "bytes_match.rs", BYTES_MATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "byte-string pattern matching must work; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "bytes ok\nbytes ok 2\n"
    );
}

#[test]
fn run_mode_derived_eq_over_floats_and_tuple_structs() {
    let dir = temp_dir("float-eq");
    let src = fixture(&dir, "float_eq.rs", FLOAT_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived equality over float fields must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "float eq ok\n");
}

#[test]
fn run_mode_pointer_and_fn_pointer_equality() {
    let dir = temp_dir("ptr-eq");
    let src = fixture(&dir, "ptr_eq.rs", PTR_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pointer equality must run to completion; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ptr eq ok\n");
}

#[test]
fn run_mode_range_indexing() {
    let dir = temp_dir("range-index");
    let src = fixture(&dir, "range_index.rs", RANGE_INDEX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "range indexing must run to completion; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ranges ok\n");
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

#[test]
fn run_mode_slice_metadata_through_places() {
    let dir = temp_dir("place-slice-meta");
    let src = fixture(&dir, "place_slice_meta.rs", PLACE_SLICE_META_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "PtrMetadata/Len through place references must work; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "meta ok\n");
}

// ----- driver demand round 2 (Lane CP) -----

const EXT_CTOR_HOF_SRC: &str = r#"
fn apply<F: FnOnce(u32) -> Option<u32>>(f: F, v: u32) -> u32 {
    match f(v) { Some(x) => x, None => 0 }
}
fn main() {
    println!("{}", apply(Some, 40) + 2);
}
"#;

const EXT_CTOR_VALUE_SRC: &str = r#"
fn main() {
    let f = Some;
    let x = f(3);
    match x { Some(v) => println!("{}", v + 1), None => println!("none") }
}
"#;

const EXT_FN_VALUE_SRC: &str = r#"
fn main() {
    let f = std::num::Wrapping;
    let w = f(20u32) + std::num::Wrapping(22u32);
    println!("{}", w.0);
    let g = String::new;
    let mut s = g();
    s.push_str("hi");
    println!("{}", s.len());
}
"#;

#[test]
fn run_mode_foreign_ctor_as_value_executes() {
    let dir = temp_dir("ext-ctor-value");
    let src = fixture(&dir, "ext_ctor_value.rs", EXT_CTOR_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a foreign constructor used as a value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "4\n");
}

#[test]
fn run_mode_foreign_ctor_passed_to_hof_executes() {
    let dir = temp_dir("ext-ctor-hof");
    let src = fixture(&dir, "ext_ctor_hof.rs", EXT_CTOR_HOF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a foreign constructor passed to a higher-order function must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_foreign_fn_and_tuple_struct_ctor_as_values_execute() {
    let dir = temp_dir("ext-fn-value");
    let src = fixture(&dir, "ext_fn_value.rs", EXT_FN_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a foreign plain function and tuple-struct constructor used as values must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n2\n");
}

// ----- derived Ord on enums: discriminant values (Lane CT) -----

const DERIVE_DISCRIMINANT_SRC: &str = r#"
// The shape of tests/ui/derives/eq-ord/derive-partial-ord-discriminant:
// derived PartialOrd must order by discriminant value, not by
// variant declaration order, including negative discriminants and
// 64-bit extremes. Operands are constants, so they also arrive as
// promoted by-reference constants.
#[derive(PartialEq, PartialOrd)]
enum E1 {
    Pos2 = 2,
    Neg1 = -1,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(i8)]
enum E3 {
    Pos2 = 2,
    Neg1 = -1_i8,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(i64)]
enum Ei64 {
    Pos2 = 2,
    Neg1 = -1,
    NegMin = 1 << 63,
    PosMax = !(1 << 63),
    Pos1 = 1,
}

fn main() {
    assert!(E1::Pos2 > E1::Pos1);
    assert!(E1::Pos1 > E1::Neg1);
    assert!(E1::Pos2 > E1::Neg1);

    assert!(E3::Pos2 > E3::Pos1);
    assert!(E3::Pos1 > E3::Neg1);
    assert!(E3::Pos2 > E3::Neg1);

    assert!(Ei64::Pos2 > Ei64::Pos1);
    assert!(Ei64::Pos2 > Ei64::Neg1);
    assert!(Ei64::Pos1 > Ei64::Neg1);
    assert!(Ei64::Pos2 > Ei64::NegMin);
    assert!(Ei64::Pos1 > Ei64::NegMin);
    assert!(Ei64::Pos2 < Ei64::PosMax);
    assert!(Ei64::Pos1 < Ei64::PosMax);
}
"#;

const DERIVE_NEGATIVE_EQ_SRC: &str = r#"
// Derived PartialEq and Ordering comparisons where a negative
// discriminant is involved: the constant operand is a promoted
// by-reference constant whose stored tag is the width-truncated
// pattern of the discriminant.
#[derive(PartialEq)]
enum P {
    X = -1,
    Y = 7,
}

fn check() -> u32 {
    let mut n = 0;
    if P::X == P::X { n += 1; }
    if P::X != P::Y { n += 1; }
    let o = core::cmp::Ordering::Less;
    if o == core::cmp::Ordering::Less { n += 1; }
    if o < core::cmp::Ordering::Equal { n += 1; }
    if core::cmp::Ordering::Greater > o { n += 1; }
    n
}

fn main() {
    if check() == 5 { println!("neg eq ok"); } else { println!("neg eq BAD"); }
}
"#;

const DERIVE_PAYLOAD_ORD_SRC: &str = r#"
// Derived PartialOrd/Ord on an enum with payloads: the
// discriminant decides across variants, fields within a variant.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum E {
    A(u8),
    B,
    C(u16),
}

fn check() -> u32 {
    let a1 = E::A(1);
    let a2 = E::A(2);
    let b = E::B;
    let c = E::C(0);
    let mut n = 0;
    if a1 < a2 { n += 1; }
    if a2 < b { n += 1; }
    if b < c { n += 1; }
    if a1 < c { n += 1; }
    if a1 == E::A(1) { n += 1; }
    if a1 != a2 { n += 1; }
    if a1.cmp(&a2) == core::cmp::Ordering::Less { n += 1; }
    if b.cmp(&c) == core::cmp::Ordering::Less { n += 1; }
    if c.cmp(&c) == core::cmp::Ordering::Equal { n += 1; }
    n
}

fn main() {
    if check() == 9 { println!("payload ord ok"); } else { println!("payload ord BAD"); }
}
"#;

#[test]
fn run_mode_derived_eq_negative_discriminant_constants() {
    let dir = temp_dir("derive-neg-eq");
    let src = fixture(&dir, "derive_neg_eq.rs", DERIVE_NEGATIVE_EQ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "negative-discriminant constant comparisons must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "neg eq ok\n",
        "derived equality and Ordering comparisons must hold"
    );
}

#[test]
fn run_mode_derived_ord_payload_enum() {
    let dir = temp_dir("derive-payload-ord");
    let src = fixture(&dir, "derive_payload_ord.rs", DERIVE_PAYLOAD_ORD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived Ord on a payload enum must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "payload ord ok\n",
        "derived Ord must order by discriminant, then fields"
    );
}

#[test]
fn run_mode_derived_partial_ord_uses_discriminant_values() {
    let dir = temp_dir("derive-discr");
    let src = fixture(&dir, "derive_discr.rs", DERIVE_DISCRIMINANT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "derived PartialOrd on explicit discriminants must run and hold; stderr={stderr}"
    );
}

// ----- allocated constants: f16, null and fn pointers, uninit (Lane CK) -----

const F16_SRC: &str = r#"#![feature(f16)]
const A: f16 = 1.5;
fn main() {
    let b: f16 = A + 0.25;
    println!("{}", b as f32);
    println!("{}", (A * 2.0) as f64);
}
"#;

const NULLPTR_SRC: &str = r#"fn target() {}
const FN_AS_PTR: *const () = target as *const ();
const NULLP: *const u8 = core::ptr::null();
static WORD: u32 = 9;
const STATIC_PTR: *const u32 = &raw const WORD;
fn main() {
    println!("{:?}", NULLP.is_null());
    println!("{:?}", FN_AS_PTR.is_null());
    println!("{:?}", unsafe { *STATIC_PTR });
}
"#;

#[test]
fn run_mode_f16_constants_execute() {
    let dir = temp_dir("f16const");
    let src = fixture(&dir, "f16const.rs", F16_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "f16 program must run to completion; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "1.75\n3\n");
}

#[test]
fn run_mode_null_and_fn_pointer_constants_execute() {
    let dir = temp_dir("nullptr");
    let src = fixture(&dir, "nullptr.rs", NULLPTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "null/fn pointer constant program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "true\nfalse\n9\n"
    );
}

// ----- statics storage model (Lane CQ) -----

const DROP_15858_SRC: &str = r#"
#![allow(static_mut_refs)]
static mut DROP_RAN: bool = false;

trait Bar {
    fn do_something(&mut self);
}

struct BarImpl;

impl Bar for BarImpl {
    fn do_something(&mut self) {}
}

struct Foo<B: Bar>(B);

impl<B: Bar> Drop for Foo<B> {
    fn drop(&mut self) {
        unsafe {
            DROP_RAN = true;
        }
    }
}

fn main() {
    {
        let _x: Foo<BarImpl> = Foo(BarImpl);
    }
    unsafe {
        assert_eq!(DROP_RAN, true);
        println!("drop ran");
    }
}
"#;

const STATIC_INDEPENDENT_SRC: &str = r#"
static mut A: i32 = 1;
static mut B: i32 = 2;

fn main() {
    unsafe {
        A += 10;
        println!("a={} b={}", A, B);
    }
}
"#;

const STATIC_INTERIOR_PTR_SRC: &str = r#"
use std::cell::UnsafeCell;

#[repr(transparent)]
struct SyncUnsafeCellLike(UnsafeCell<u32>);
unsafe impl Sync for SyncUnsafeCellLike {}

#[repr(C)]
struct SyncPtr {
    foo: *mut u32,
}
unsafe impl Sync for SyncPtr {}

static mut STATIC: u32 = 42;
static PTR: SyncPtr = SyncPtr { foo: std::ptr::addr_of_mut!(STATIC) };

static WRAPPED: SyncUnsafeCellLike = SyncUnsafeCellLike(UnsafeCell::new(7));
static PTR2: SyncPtr =
    SyncPtr { foo: std::ptr::addr_of!(WRAPPED) as *mut u32 };

fn main() {
    unsafe {
        assert_eq!(*PTR.foo, 42);
        *PTR.foo = 0;
        assert_eq!(*PTR.foo, 0);
        assert_eq!(*PTR2.foo, 7);
        *PTR2.foo = 9;
        assert_eq!(*PTR2.foo, 9);
        println!("done");
    }
}
"#;

const STATIC_INTERIOR_REF_SRC: &str = r#"
struct S { a: usize }

static A: S = S { a: 3 };
static B: &'static usize = &A.a;
static D: [usize; 1] = [1];
static F: &'static usize = &D[0];

fn main() {
    assert_eq!(*B, A.a);
    assert_eq!(*F, D[0]);
    println!("b={} f={}", *B, *F);
}
"#;

const STATIC_INTERIOR_SRC: &str = r#"
use std::cell::RefCell;

static mut CELL: RefCell<i32> = RefCell::new(10);

fn main() {
    unsafe {
        *CELL.borrow_mut() = 5;
        let v = *CELL.borrow();
        *CELL.borrow_mut() = v + 2;
        println!("cell={}", *CELL.borrow());
    }
}
"#;

const STATIC_MUT_COUNTER_SRC: &str = r#"
static mut COUNT: i32 = 0;

fn bump() {
    unsafe { COUNT += 1; }
}

fn main() {
    bump();
    bump();
    bump();
    unsafe { println!("count={}", COUNT); }
}
"#;

#[test]
fn run_mode_generic_drop_trait_bound_15858() {
    let dir = temp_dir("drop-15858");
    let src = fixture(&dir, "drop_15858.rs", DROP_15858_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the 15858 destructor must run and write the static; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "drop ran\n",
        "main observes the static write the destructor made"
    );
}

#[test]
fn run_mode_static_interior_mutability() {
    let dir = temp_dir("static-interior");
    let src = fixture(&dir, "interior.rs", STATIC_INTERIOR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a RefCell living in a static must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "cell=7\n",
        "guard writes to a static are stored back and read by later borrows"
    );
}

#[test]
fn run_mode_static_interior_pointer_writes_through() {
    let dir = temp_dir("static-interior-ptr");
    let src = fixture(&dir, "interior_ptr.rs", STATIC_INTERIOR_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a raw pointer to a static (or into its wrapper) must write the store; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "done\n",
        "writes through interior pointers are observed by later reads"
    );
}

#[test]
fn run_mode_static_interior_reference_reads_the_field() {
    let dir = temp_dir("static-interior-ref");
    let src = fixture(&dir, "interior_ref.rs", STATIC_INTERIOR_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a reference into a static must read the field it names; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "b=3 f=1\n",
        "the store seed is the full initializer; the reference carries the projection"
    );
}

#[test]
fn run_mode_static_mut_counter_across_calls() {
    let dir = temp_dir("static-counter");
    let src = fixture(&dir, "counter.rs", STATIC_MUT_COUNTER_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a static mut counter must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "count=3\n",
        "writes to a static persist across calls"
    );
}

#[test]
fn run_mode_two_statics_of_same_type_stay_independent() {
    let dir = temp_dir("static-independent");
    let src = fixture(&dir, "independent.rs", STATIC_INDEPENDENT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "two statics must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a=11 b=2\n",
        "statics are keyed by identity, not by type or initializer"
    );
}

// ----- monomorphization validity, round 2 (Lane CS) -----

const BOUNDED_RECURSION_SRC: &str = r#"
fn fact(n: u64) -> u64 { if n == 0 { 1 } else { n * fact(n - 1) } }
fn down<T: Copy>(x: Option<T>) -> usize { match x { None => 0, Some(_) => down::<T>(x) } }
fn main() { let _ = fact(3); let _ = down::<i32>(Some(1)); }
"#;

const HUGE_PHANTOM_ONLY_SRC: &str = r#"
fn f<T>(_x: std::marker::PhantomData<T>) -> usize { std::mem::size_of::<usize>() }
fn main() { let _ = f::<[u8; usize::MAX]>(std::marker::PhantomData); }
"#;

const LARGE_BUT_LEGAL_SRC: &str = r#"
struct Big([u8; 1_000_000]);
fn main() { let _b = Big([7u8; 1_000_000]); }
"#;

const POINTER_TO_HUGE_SRC: &str = r#"
fn main() {
    let _p: *const [u8; usize::MAX] = core::ptr::null();
}
"#;

const TOO_BIG_LOCAL_SRC: &str = r#"
fn main() { let _a = [0u8; usize::MAX]; }
"#;

const TOO_BIG_SIZE_OF_SRC: &str = r#"
struct S { x: [u8; usize::MAX] }
fn f() -> usize { std::mem::size_of::<S>() }
fn main() { let _ = f(); }
"#;

const UNBOUNDED_INSTANTIATION_SRC: &str = r#"
fn g<T>() { g::<Option<T>>(); }
fn main() { g::<i32>(); }
"#;

const UNSIZED_NONFINAL_SRC: &str = r#"
#![feature(trivial_bounds)]
#![expect(trivial_bounds)]
struct TwoStrs(str, str)
where
    str: Sized;
fn main() {}
"#;

#[test]
fn convert_mode_bounded_generic_recursion_converts() {
    let dir = temp_dir("mono-bounded-recursion");
    let src = fixture(&dir, "bounded.rs", BOUNDED_RECURSION_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "plain recursion and same-instance generic recursion are \
         finite demand and must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_huge_type_named_only_through_phantom_data_converts() {
    let dir = temp_dir("mono-huge-phantom");
    let src = fixture(&dir, "huge_phantom.rs", HUGE_PHANTOM_ONLY_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a too-big type that is only named (PhantomData, an ordinary \
         generic argument) is never materialized and must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_large_but_legal_type_converts() {
    let dir = temp_dir("mono-large-legal");
    let src = fixture(&dir, "large_legal.rs", LARGE_BUT_LEGAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a large type within the target's bound must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_pointer_to_huge_type_converts() {
    // A pointer to a too-big type is itself an ordinary pointer;
    // merely pointing at the type does not materialize it, so
    // the program converts.
    let dir = temp_dir("mono-pointer-to-huge");
    let src = fixture(&dir, "pointer_to_huge.rs", POINTER_TO_HUGE_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a pointer to a huge type must convert; stderr={stderr}"
    );
}

#[test]
fn convert_mode_too_big_local_type_is_a_compile_failure() {
    let dir = temp_dir("mono-too-big-local");
    let src = fixture(&dir, "too_big_local.rs", TOO_BIG_LOCAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a local of a type too big for the target must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("too big for the target architecture"),
        "the failure must carry rustc's too-big wording; stderr={stderr}"
    );
}

#[test]
fn convert_mode_too_big_size_of_argument_is_a_compile_failure() {
    let dir = temp_dir("mono-too-big-size-of");
    let src = fixture(&dir, "too_big_size_of.rs", TOO_BIG_SIZE_OF_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "measuring a too-big type with size_of must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("too big for the target architecture"),
        "the failure must carry rustc's too-big wording; stderr={stderr}"
    );
}

#[test]
fn convert_mode_unbounded_self_instantiation_is_a_recursion_limit_failure() {
    let dir = temp_dir("mono-recursion-limit");
    let src = fixture(&dir, "unbounded.rs", UNBOUNDED_INSTANTIATION_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "unbounded self-instantiation must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating"),
        "the failure must name the recursion limit; stderr={stderr}"
    );
}

#[test]
fn convert_mode_unsized_field_in_non_final_position_converts() {
    // A struct with an unsized field in a non-final position is
    // accepted under trivial bounds but can never be
    // materialized; asking the compiler for its layout creates
    // a delayed internal compiler error instead of returning
    // one. The too-big scan must screen the type out and leave
    // the compilation alone.
    let dir = temp_dir("mono-unsized-nonfinal");
    let src = fixture(&dir, "unsized_nonfinal.rs", UNSIZED_NONFINAL_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a never-materializable unsized-field struct must not fail compilation; stderr={stderr}"
    );
}

#[test]
fn run_mode_unbounded_self_instantiation_is_unsupported() {
    let dir = temp_dir("mono-recursion-limit-run");
    let src = fixture(&dir, "unbounded_run.rs", UNBOUNDED_INSTANTIATION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "run mode reports a monomorphization failure as unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating"),
        "the failure must name the recursion limit; stderr={stderr}"
    );
}

// ----- transparent-wrapper pointers (Lane DP) --------------------------------

const UNSAFECELL_GET_SRC: &str = r#"
use std::cell::UnsafeCell;
fn main() {
    let cell = UnsafeCell::new(5i32);
    unsafe {
        let p = cell.get();
        *p = 7;
        println!("{}", *p);
    }
}
"#;

const MAYBEUNINIT_PTR_SRC: &str = r#"
use std::mem::MaybeUninit;
fn main() {
    let mut slot = MaybeUninit::<i32>::uninit();
    unsafe {
        slot.as_mut_ptr().write(11);
        println!("{}", *slot.as_ptr());
    }
}
"#;

const CELL_AS_PTR_SRC: &str = r#"
use std::cell::Cell;
fn main() {
    let c = Cell::new(4i32);
    unsafe { *c.as_ptr() = 6; }
    println!("{}", c.get());
}
"#;

const REFCELL_COMPOUND_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let cell = RefCell::new(1u32);
    *cell.borrow_mut() += 1;
    let v = *cell.borrow();
    println!("{v}");
}
"#;

const REFCELL_NESTED_SRC: &str = r#"
use std::cell::RefCell;
fn main() {
    let cell = RefCell::new(1i32);
    *cell.borrow_mut() = *cell.borrow() + 1;
    println!("{}", cell.borrow());
}
"#;

#[test]
fn run_mode_unsafecell_get_reads_and_writes_the_wrapped_value() {
    // `UnsafeCell::get` returns a pointer to the wrapped value
    // itself (the wrapper is transparent). Writing and reading
    // through it must reach the `i32`, not the `UnsafeCell`
    // aggregate: this used to fail with a type mismatch naming
    // `Adt("std::cell::UnsafeCell")` on both directions.
    let dir = temp_dir("unsafecell-get");
    let src = fixture(&dir, "unsafecell_get.rs", UNSAFECELL_GET_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "UnsafeCell::get program must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

#[test]
fn run_mode_maybeuninit_pointer_reads_and_writes_the_wrapped_value() {
    let dir = temp_dir("maybeuninit-ptr");
    let src = fixture(&dir, "maybeuninit_ptr.rs", MAYBEUNINIT_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "MaybeUninit pointer program must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "11\n");
}

#[test]
fn run_mode_cell_as_ptr_writes_the_cell_value() {
    // `Cell::as_ptr` is `UnsafeCell::get` on the cell's inner
    // field: the builtin `Cell` value shape and the collected
    // `get` body must agree on the natural
    // `Cell { UnsafeCell { value } }` nesting.
    let dir = temp_dir("cell-as-ptr");
    let src = fixture(&dir, "cell_as_ptr.rs", CELL_AS_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cell::as_ptr program must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "6\n");
}

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

fn run_ok(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver(&src_path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "catch_unwind case {tag} must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "catch_unwind case {tag} stdout mismatch; stderr={stderr}"
    );
}

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
    // aborts (101, abort reason on stderr) and the surrounding
    // catch_unwind does NOT observe it ("caught" never prints),
    // exactly as a native abort is not a catchable panic.
    let dir = temp_dir("unwind-drop-panics");
    let src = fixture(&dir, "case.rs", UNWIND_DROP_PANICS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "a panic in an unwind destructor must abort with 101; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "dtor 1\n",
        "only the first unwind drop runs before the abort; stderr={stderr}"
    );
    assert!(
        stderr.contains("aborted"),
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

// ----- Waker construction (Lane DQ) ------------------------------------------

const WAKER_FROM_SRC: &str = r#"
use std::sync::Arc;
use std::task::{Wake, Waker};

struct MyWaker;
impl Wake for MyWaker {
    fn wake(self: Arc<Self>) {}
}

fn main() {
    let waker = Waker::from(Arc::new(MyWaker));
    let waker2 = waker.clone();
    std::mem::forget(waker);
    std::mem::forget(waker2);
    println!("waker ok");
}
"#;

#[test]
fn run_mode_waker_from_arc_constructs_and_clones() {
    // `Waker::from(Arc<T: Wake>)` is interpreter runtime support:
    // the modelled waker carries the Arc value, so constructing
    // it and cloning it (a value copy, exactly how the modelled
    // `Arc` itself clones) run to completion. The wakers are
    // forgotten rather than dropped: dropping one runs the
    // shared-ownership glue the value model does not account
    // for - the same limit as dropping an `Arc` itself - and
    // handing the waker to `Context::from_waker` is blocked by
    // the reference-to-reference transmute in that core body,
    // not by the waker value.
    let dir = temp_dir("wakerfrom");
    let src = fixture(&dir, "wakerfrom.rs", WAKER_FROM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "waker construction and clone must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "waker ok\n",
        "the guest's stdout is the driver's stdout"
    );
}

const TRANSMUTE_FLOAT_INT_SRC: &str = r#"
fn inp32() -> f32 { let mut s = 0.0f32; let mut i = 0; while i < 7 { s += 0.5; i += 1; } s }
fn inp64() -> f64 { let mut s = 0.0f64; let mut i = 0; while i < 7 { s += 1.0; i += 1; } s }
fn main() {
    let f = inp32();
    let bits: u32 = unsafe { std::mem::transmute(f) };
    let back: f32 = unsafe { std::mem::transmute(bits) };
    let g = inp64();
    let gbits: u64 = unsafe { std::mem::transmute(g) };
    let sg: i64 = unsafe { std::mem::transmute(g) };
    println!("{:x} {} {:x} {}", bits, back, gbits, sg);
    println!("{:x} {} {}", f.to_bits(), f32::from_bits(bits), f64::from_bits(gbits));
}
"#;

#[test]
fn run_mode_transmute_float_int_bit_reinterpretation() {
    run_ok(
        "transmute-float-int",
        TRANSMUTE_FLOAT_INT_SRC,
        "40600000 3.5 401c000000000000 4619567317775286272\n40600000 3.5 7\n",
    );
}

const TRANSMUTE_BOOL_CHAR_SRC: &str = r#"
fn inpb() -> bool { let mut s = 0; let mut i = 0; while i < 41 { s += 1; i += 1; } s == 42 }
fn inpc() -> char { let mut s = 0u32; let mut i = 0; while i < 65 { s += 1; i += 1; } char::from_u32(s).unwrap() }
fn main() {
    let b = inpb();
    let n: u8 = unsafe { std::mem::transmute(b) };
    let b2: bool = unsafe { std::mem::transmute(1u8) };
    let c = inpc();
    let cn: u32 = unsafe { std::mem::transmute(c) };
    let c2: char = unsafe { std::mem::transmute(66u32) };
    println!("{} {} {} {}", n, b2, cn, c2);
}
"#;

#[test]
fn run_mode_transmute_bool_and_char() {
    run_ok("transmute-bool-char", TRANSMUTE_BOOL_CHAR_SRC, "0 true 65 B\n");
}

const TRANSMUTE_ARRAY_SRC: &str = r#"
fn inp() -> u32 { let mut s = 0u32; let mut i = 0; while i < 41 { s += 1; i += 1; } s }
fn main() {
    let n = inp() + 0x04030200 - 41;
    let a: [u8; 4] = unsafe { std::mem::transmute(n) };
    let back: u32 = unsafe { std::mem::transmute(a) };
    let wide: u64 = unsafe { std::mem::transmute([1u16, 2, 3, 4]) };
    let pair: u64 = unsafe { std::mem::transmute([n, n + 1]) };
    let halves: [u32; 2] = unsafe { std::mem::transmute(pair) };
    let fpair: u64 = unsafe { std::mem::transmute([1.5f32, 2.5f32]) };
    println!("{:?} {:x} {:x} {:x} {:?} {:x}", a, back, wide, pair, halves, fpair);
}
"#;

#[test]
fn run_mode_transmute_arrays_and_scalars() {
    run_ok(
        "transmute-arrays",
        TRANSMUTE_ARRAY_SRC,
        "[0, 2, 3, 4] 4030200 4000300020001 403020104030200 [67305984, 67305985] 402000003fc00000\n",
    );
}

const TRANSMUTE_WRAPPER_SRC: &str = r#"
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::num::Wrapping;
struct Padded { x: u32, _m: PhantomData<u8> }
fn inp() -> u32 { let mut s = 0u32; let mut i = 0; while i < 41 { s += 1; i += 1; } s }
fn main() {
    let n = inp();
    let w: Wrapping<u32> = unsafe { std::mem::transmute(n) };
    let back: u32 = unsafe { std::mem::transmute(w) };
    let wf: Wrapping<f32> = unsafe { std::mem::transmute(0x3fc00000u32) };
    let p: Padded = unsafe { std::mem::transmute(n + 1) };
    let md: ManuallyDrop<u32> = unsafe { std::mem::transmute(n) };
    let mback: u32 = unsafe { std::mem::transmute(md) };
    println!("{} {} {} {} {}", w.0, back, wf.0, p.x, mback);
}
"#;

#[test]
fn run_mode_transmute_wrapper_adts() {
    run_ok(
        "transmute-wrappers",
        TRANSMUTE_WRAPPER_SRC,
        "41 41 1.5 42 41\n",
    );
}

const TRANSMUTE_CONSTANTS_SRC: &str = r#"
fn main() {
    let a: u32 = unsafe { std::mem::transmute(1.5f32) };
    let b: f32 = unsafe { std::mem::transmute(0x3fc00000u32) };
    let c: u64 = unsafe { std::mem::transmute(-2.5f64) };
    let d: u8 = unsafe { std::mem::transmute(true) };
    let e: [u8; 4] = unsafe { std::mem::transmute(0x04030201u32) };
    println!("{:x} {} {:x} {} {:?}", a, b, c, d, e);
}
"#;

#[test]
fn run_mode_transmute_constant_operands() {
    run_ok(
        "transmute-constants",
        TRANSMUTE_CONSTANTS_SRC,
        "3fc00000 1.5 c004000000000000 1 [1, 2, 3, 4]\n",
    );
}

const TRANSMUTE_WIDE_SRC: &str = r#"
fn main() {
    let big: u128 = unsafe { std::mem::transmute(-1i128) };
    let neg: i8 = unsafe { std::mem::transmute(0xffu8) };
    let s16: i16 = unsafe { std::mem::transmute(0xffffu16) };
    println!("{:x} {} {}", big, neg, s16);
}
"#;

#[test]
fn run_mode_transmute_wide_and_signed() {
    run_ok(
        "transmute-wide",
        TRANSMUTE_WIDE_SRC,
        "ffffffffffffffffffffffffffffffff -1 -1\n",
    );
}

const TRANSMUTE_BAD_CHAR_SRC: &str = r#"
fn inp() -> u32 { let mut s = 0x10000u32; s *= 17; s }
fn main() {
    let c: char = unsafe { std::mem::transmute(inp()) };
    println!("{}", c);
}
"#;

#[test]
fn run_mode_transmute_invalid_char_is_explicit_unsupported() {
    // Transmuting an invalid bit pattern into `char` is UB in
    // real Rust; the model reports it as an explicit unsupported
    // error instead of inventing a character.
    let dir = temp_dir("transmute-bad-char");
    let src = fixture(&dir, "case.rs", TRANSMUTE_BAD_CHAR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "invalid char transmute must be a named unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("transmute to char"),
        "stderr must name the char transmute; stderr={stderr}"
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

const RC_ARC_DYN_SRC: &str = r#"
use std::rc::Rc;
use std::sync::Arc;
trait T {
    fn val(&self) -> i32;
    fn d(&self) -> i32 { 7 }
}
struct S(i32);
impl T for S {
    fn val(&self) -> i32 { self.0 }
}
fn main() {
    let r: Rc<dyn T> = Rc::new(S(42));
    let r2 = r.clone();
    let a: Arc<dyn T> = Arc::new(S(43));
    println!("{} {} {} {}", r.val(), r2.val(), r.d(), a.val());
}
"#;

#[test]
fn run_mode_rc_arc_dyn_trait_method_calls() {
    // Method calls through an `Rc<dyn Trait>` / `Arc<dyn Trait>`
    // receiver: the Deref builtin must yield the dyn fat pointer
    // itself (as it does for Box), and dispatch must reach both
    // the impl method (reading the pointee's state) and the
    // trait's default method.
    let dir = temp_dir("rc-arc-dyn");
    let src = fixture(&dir, "rc_arc_dyn.rs", RC_ARC_DYN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "calling trait methods through Rc<dyn Trait> / Arc<dyn Trait> must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42 42 7 43\n");
}

const DYN_DEFAULT_DISPATCH_SRC: &str = r#"
use std::rc::Rc;
trait T {
    fn val(&self) -> i32;
    fn extra(&self) -> i32 { self.val() + 1 }
    fn d(&self) -> i32 { 7 }
}
struct A;
impl T for A { fn val(&self) -> i32 { 10 } }
struct B;
impl T for B {
    fn val(&self) -> i32 { 20 }
    fn d(&self) -> i32 { 8 }
}
fn main() {
    let a: Rc<dyn T> = Rc::new(A);
    let b: Rc<dyn T> = Rc::new(B);
    println!("{} {} {} {}", a.extra(), a.d(), b.extra(), b.d());
    let r: &dyn T = &A;
    println!("{} {}", r.extra(), r.d());
}
"#;

#[test]
fn run_mode_dyn_default_method_dispatch() {
    // A trait method the impl does not override has only the
    // trait's default body: dispatch through a dyn receiver must
    // reach it (monomorphized at the concrete type), its inner
    // self calls must hit the concrete overrides, and an
    // overriding impl method must win over the default.
    let dir = temp_dir("dyn-default-dispatch");
    let src = fixture(&dir, "dyn_default.rs", DYN_DEFAULT_DISPATCH_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dispatching to trait default methods through dyn receivers must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "11 7 21 8\n11 7\n");
}

const DYN_EQ_REF_SRC: &str = r#"
trait Foo: PartialEq<i32> + std::fmt::Debug {
    fn a(&self) -> i32 { 100 }
}
impl Foo for i32 {}
fn main() {
    let x: i32 = 1;
    assert_eq!(x.a(), 100);
    let foo: &dyn Foo = &x;
    assert_eq!(*foo, 1);
    assert!(*foo != 2);
    println!("dyn-eq-ref ok");
}
"#;

#[test]
fn run_mode_dyn_ref_partial_eq_compares_pointee() {
    let dir = temp_dir("dyn-eq-ref");
    let src = fixture(&dir, "dyn_eq_ref.rs", DYN_EQ_REF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing through a &dyn trait object with a PartialEq supertrait must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-ref ok\n");
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

const DYN_EQ_ADT_SRC: &str = r#"
trait Foo: PartialEq<Bar> + std::fmt::Debug {}
#[derive(PartialEq, Debug)]
struct Bar(i32);
impl Foo for Bar {}
fn main() {
    let obj: &dyn Foo = &Bar(7);
    assert_eq!(*obj, Bar(7));
    assert!(*obj != Bar(8));
    println!("dyn-eq-adt ok");
}
"#;

#[test]
fn run_mode_dyn_ref_partial_eq_against_adt() {
    let dir = temp_dir("dyn-eq-adt");
    let src = fixture(&dir, "dyn_eq_adt.rs", DYN_EQ_ADT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing a trait object against an ADT value must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-adt ok\n");
}

const DYN_EQ_DYN_SRC: &str = r#"
trait T: std::fmt::Debug {
    fn val(&self) -> i32;
    fn type_id(&self) -> u32;
}
#[derive(Debug)]
struct A(i32);
#[derive(Debug)]
struct B(i32);
impl T for A {
    fn val(&self) -> i32 { self.0 }
    fn type_id(&self) -> u32 { 1 }
}
impl T for B {
    fn val(&self) -> i32 { self.0 }
    fn type_id(&self) -> u32 { 2 }
}
impl PartialEq for dyn T {
    fn eq(&self, other: &Self) -> bool {
        self.type_id() == other.type_id() && self.val() == other.val()
    }
}
fn main() {
    let a1: &dyn T = &A(1);
    let a2: &dyn T = &A(1);
    let a3: &dyn T = &A(2);
    let b1: &dyn T = &B(1);
    assert!(a1 == a2);
    assert!(a1 != a3);
    assert!(a1 != b1);
    println!("dyn-eq-dyn ok");
}
"#;

#[test]
fn run_mode_dyn_partial_eq_against_dyn() {
    let dir = temp_dir("dyn-eq-dyn");
    let src = fixture(&dir, "dyn_eq_dyn.rs", DYN_EQ_DYN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing two trait objects must run (unequal concrete types are unequal); stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-dyn ok\n");
}

const DYN_EQ_UPCAST_SRC: &str = r#"
trait Super: PartialEq<i32> + std::fmt::Debug {}
trait Sub: Super {}
impl Super for i32 {}
impl Sub for i32 {}
fn main() {
    let sub: &dyn Sub = &1;
    let sup: &dyn Super = sub;
    assert_eq!(*sup, 1);
    assert!(*sup != 2);
    let boxed: Box<dyn Sub> = Box::new(5);
    let boxed_super: Box<dyn Super> = boxed;
    assert_eq!(*boxed_super, 5);
    assert!(*boxed_super != 6);
    println!("dyn-eq-upcast ok");
}
"#;

#[test]
fn run_mode_upcast_dyn_partial_eq_compares_pointee() {
    let dir = temp_dir("dyn-eq-upcast");
    let src = fixture(&dir, "dyn_eq_upcast.rs", DYN_EQ_UPCAST_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "comparing through an upcast trait object must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "dyn-eq-upcast ok\n");
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

const UNION_DROP_SRC: &str = r#"
use std::mem::ManuallyDrop;
struct S;
static mut CHECK: u8 = 0;
impl Drop for S {
    fn drop(&mut self) {
        unsafe { CHECK += 10; }
    }
}
union U {
    a: ManuallyDrop<S>,
}
impl Drop for U {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}
fn main() {
    unsafe {
        let mut u = U { a: ManuallyDrop::new(S) };
        u = U { a: ManuallyDrop::new(S) };
        assert_eq!(CHECK, 1);
        *u.a = S;
        assert_eq!(CHECK, 11);
    }
    println!("union ok");
}
"#;

#[test]
fn run_mode_union_drop_and_field_assignment() {
    let dir = temp_dir("union-drop");
    let src = fixture(&dir, "union_drop.rs", UNION_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a union with a Drop impl must drop on overwrite, and assigning through its field must drop the old field value; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "union ok\n");
}

const MEM_DROP_ZST_UNION_SRC: &str = r#"
use std::mem::ManuallyDrop;
struct S;
static mut CHECK: u8 = 0;
impl Drop for S {
    fn drop(&mut self) {
        unsafe { CHECK += 10; }
    }
}
union U {
    a: ManuallyDrop<S>,
}
impl Drop for U {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}
fn main() {
    let u = U { a: ManuallyDrop::new(S) };
    drop(u);
    unsafe { assert_eq!(CHECK, 1); }
    println!("mem-drop ok");
}
"#;

#[test]
fn run_mode_mem_drop_of_zst_union_runs_destructor() {
    let dir = temp_dir("mem-drop-zst-union");
    let src = fixture(&dir, "mem_drop_zst_union.rs", MEM_DROP_ZST_UNION_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an explicit mem::drop of a zero-sized union value must run its destructor; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "mem-drop ok\n");
}

const UNION_REINTERPRET_SRC: &str = r#"
union W {
    a: u32,
    b: f32,
}
union V {
    a: u32,
    b: [u8; 4],
    c: u16,
}
union T {
    a: (u8, u8),
    b: u16,
}
union U<A, B>
where
    A: Copy,
    B: Copy,
{
    a: A,
    b: B,
}
unsafe fn union_transmute<A, B>(a: A) -> B
where
    A: Copy,
    B: Copy,
{
    U { a }.b
}
fn main() {
    unsafe {
        let w = W { a: 0x3F80_0000 };
        assert_eq!(w.b, 1.0);
        let mut w2 = W { b: 2.5 };
        assert_eq!(w2.a, 0x4020_0000);
        w2.a = 0x3F80_0000;
        assert_eq!(w2.b, 1.0);
        let v = V { a: 0x0403_0201 };
        assert_eq!(v.b, [1, 2, 3, 4]);
        assert_eq!(v.c, 0x0201);
        let v2 = V { b: [9, 8, 7, 6] };
        assert_eq!(v2.a, 0x0607_0809);
        let mut m = V { a: 0 };
        m.c = 0xBEEF;
        assert_eq!(m.c, 0xBEEF);
        assert_eq!(m.b, [0xEF, 0xBE, 0, 0]);
        let t = T { a: (1, 1) };
        assert_eq!(t.b, (1 << 8) + 1);
        let g: u16 = union_transmute((1u8, 1u8));
        assert_eq!(g, (1 << 8) + 1);
        let f: f32 = union_transmute(0x3F80_0000u32);
        assert_eq!(f, 1.0);
        println!("union reinterpret ok");
    }
}
"#;

#[test]
fn run_mode_union_field_reads_reinterpret_bits() {
    // Reading a union through a different field than the one
    // written reinterprets the held field's bits (little-endian:
    // narrower fields read the low bytes), matching native runs
    // of this same program.
    let dir = temp_dir("union-reinterpret");
    let src = fixture(&dir, "union_reinterpret.rs", UNION_REINTERPRET_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "union field reads must reinterpret the held field's bits; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "union reinterpret ok\n"
    );
}

const UNION_WRAPPER_FIELDS_SRC: &str = r#"
use std::mem::ManuallyDrop;
union C {
    a: u32,
    b: char,
}
union B {
    a: u8,
    b: bool,
}
union M {
    a: ManuallyDrop<u32>,
    b: f32,
}
fn main() {
    unsafe {
        let c = C { a: 0x41 };
        assert_eq!(c.b, 'A');
        let b = B { a: 1 };
        assert_eq!(b.b, true);
        let m = M { b: 1.5 };
        let md = m.a;
        let m3 = M { a: md };
        assert_eq!(m3.b, 1.5);
        let m2 = M { a: ManuallyDrop::new(0x3F80_0000) };
        assert_eq!(m2.b, 1.0);
        println!("union wrapper fields ok");
    }
}
"#;

#[test]
fn run_mode_union_field_reads_through_wrapper_fields() {
    // `char` / `bool` fields and one-payload wrapper fields
    // (ManuallyDrop) reinterpret through the same shapes.
    let dir = temp_dir("union-wrapper-fields");
    let src = fixture(&dir, "union_wrapper_fields.rs", UNION_WRAPPER_FIELDS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "union reads through char/bool/ManuallyDrop fields must reinterpret; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "union wrapper fields ok\n"
    );
}

const UNION_NO_SHAPE_SRC: &str = r#"
#[derive(Copy, Clone)]
struct P(u32, u32);
union X {
    a: P,
    b: u64,
}
fn main() {
    unsafe {
        let x = X { b: 42 };
        let _p = x.a;
        println!("should not reach");
    }
}
"#;

#[test]
fn run_mode_union_field_read_without_scalar_shape_is_explicit() {
    // A field whose type has no scalar shape (a two-payload
    // struct) cannot be reinterpreted from another field's
    // bits: the read is a named unsupported error, not a
    // fabricated value.
    let dir = temp_dir("union-no-shape");
    let src = fixture(&dir, "union_no_shape.rs", UNION_NO_SHAPE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a union read with no scalar shape must be a named unsupported; stderr={stderr}"
    );
    assert!(
        stderr.contains("union field read"),
        "stderr must name the union field read; stderr={stderr}"
    );
}

const UNION_NESTED_WRITE_SRC: &str = r#"
union U {
    a: (u16, u16),
    b: u32,
}
fn main() {
    unsafe {
        let mut u = U { b: 0xDEAD_DEAD };
        u.a.0 = 0xBEEF;
        println!("should not reach");
    }
}
"#;

#[test]
fn run_mode_union_write_inside_unheld_field_is_explicit() {
    // Writing inside a field the value does not currently hold
    // would have to preserve the held field's remaining bytes;
    // the model holds one field's value, so the write stays an
    // uninitialized error rather than fabricating the field.
    let dir = temp_dir("union-nested-write");
    let src = fixture(&dir, "union_nested_write.rs", UNION_NESTED_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a write inside an unheld union field must fail explicitly; stderr={stderr}"
    );
    assert!(
        stderr.contains("uninitialized"),
        "stderr must name the uninitialized field; stderr={stderr}"
    );
}

const WEAK_COW_DROP_SRC: &str = r#"
enum Void {}
fn main() {
    let _w1 = std::rc::Weak::<Void>::new();
    let _w2 = std::sync::Weak::<Void>::new();
    let c: std::borrow::Cow<str> = std::borrow::Cow::Borrowed("x");
    let _o: std::borrow::Cow<str> = std::borrow::Cow::Owned(String::from("y"));
    drop(c);
    println!("weak-cow ok");
}
"#;

#[test]
fn run_mode_weak_and_cow_str_drop() {
    let dir = temp_dir("weak-cow-drop");
    let src = fixture(&dir, "weak_cow_drop.rs", WEAK_COW_DROP_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "dropping Weak pointers and Cow<str> values must be accounted; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "weak-cow ok\n");
}
const TEST_CRATE_WITH_MAIN_SRC: &str = r#"
fn main() {
    println!("USER MAIN RAN");
}

#[test]
fn sample_test() {
    assert_eq!(1 + 1, 2);
}
"#;

#[test]
fn run_mode_test_crate_enters_harness_main_not_user_main() {
    // Lane ET: a `--test` crate has two bodies named
    // `<crate>::main` -- the user's and the generated harness
    // entry. Run mode used to hand the interpreter the bare
    // name, which names no stored body (bodies that share a
    // name are keyed by their `DefId`), and the run failed with
    // `unknown function <crate>::main`; picking the entry by
    // name alone would also have entered the user's `main`,
    // which natively is an ordinary function in a test crate.
    // The driver now enters the entry point rustc designated.
    // The harness's call into the libtest runner is served by
    // the `test_main_env_args` builtin, so the crate's tests
    // run; the user's `main` must not have run.
    let dir = temp_dir("test-crate-main");
    let src = fixture(&dir, "testcrate.rs", TEST_CRATE_WITH_MAIN_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("--test")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the harness entry must be reached and its tests run; stderr={stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("test sample_test ... ok"),
        "the libtest runner must report the test; stdout={stdout}"
    );
    assert!(
        !stderr.contains("unknown function"),
        "entry resolution must not lose the entry body; stderr={stderr}"
    );
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("USER MAIN RAN"),
        "the user's main is not the entry of a test crate and must not run"
    );
}

const TEST_CRATE_NO_TESTS_SRC: &str = r#"
fn main() {
    println!("USER MAIN RAN");
}
"#;

#[test]
fn run_mode_test_crate_without_tests_still_enters_harness() {
    // Lane ET: under `--test` with no `#[test]` items the
    // harness main is still the crate's entry (natively it runs
    // an empty test list and calls the libtest runner directly);
    // the user's `main` -- the only `main` in the source -- is
    // not entered.
    let dir = temp_dir("test-crate-no-tests");
    let src = fixture(&dir, "notests.rs", TEST_CRATE_NO_TESTS_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("--test")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the harness entry must be reached and run its empty test list; stderr={stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("running 0 tests")
            && stdout.contains("test result: ok. 0 passed; 0 failed"),
        "the libtest runner must report the empty run; stdout={stdout}"
    );
    assert!(
        !stderr.contains("unknown function"),
        "entry resolution must not lose the entry body; stderr={stderr}"
    );
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("USER MAIN RAN"),
        "the user's main is not the entry of a test crate and must not run"
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

// ----- subslice projections and pattern types (Lane FG) ---------------------

const SUBSLICE_SRC: &str = r#"
fn head_rest(s: &[i32]) -> i32 {
    match s {
        [first, rest @ ..] => first + rest.len() as i32,
        [] => -1,
    }
}
fn tail2(s: &[i32]) -> i32 {
    match s {
        [.., a, b] => a * 10 + b,
        _ => 0,
    }
}
fn mid_sum(s: &[i32]) -> i32 {
    match s {
        [_, m @ .., _] => {
            let mut acc = 0;
            let mut i = 0;
            while i < m.len() { acc += m[i]; i += 1; }
            acc
        }
        _ => 0,
    }
}
fn main() {
    let arr = [1, 2, 3, 4, 5];
    let s = &arr[1..3];
    println!("len {} sum {}", s.len(), s[0] + s[1]);
    let mut arr2 = [10, 20, 30, 40];
    arr2[1..3][0] = 99;
    println!("w {}", arr2[1]);
    let t = &arr[..2];
    println!("head {}", t.len());
    let u = &arr[3..];
    println!("tail {}", u[0] + u[1]);
    let v = [5, 1, 2, 3];
    println!("hr {}", head_rest(&v));
    println!("t2 {}", tail2(&v));
    println!("mid {}", mid_sum(&v));
    println!("hr2 {}", head_rest(&[9, 8]));
    let mut arr3 = [0, 1, 2, 3, 4, 5];
    arr3[2..5][1] = 77;
    println!("nested {}", arr3[3]);
    let w = &arr[1..4][1..];
    println!("subsub {} {}", w.len(), w[0]);
}
"#;

#[test]
fn run_mode_subslice_projections_read_write_and_patterns() {
    // Subslice places (from slice patterns and range indexing)
    // convert and execute: shared range reads, writes through
    // `IndexMut` ranges starting above zero, `@ ..` bindings, and
    // nested sub-slices all alias the base storage.
    let dir = temp_dir("subslice");
    let src = fixture(&dir, "subslice.rs", SUBSLICE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "subslice program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len 2 sum 5\nw 99\nhead 2\ntail 9\nhr 8\nt2 23\nmid 3\nhr2 10\nnested 77\nsubsub 2 3\n",
        "subslice stdout mismatch; stderr={stderr}"
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

const PATTERN_TYPE_SRC: &str = r#"
#![feature(pattern_types, pattern_type_macro)]
use std::pat::pattern_type;
type Y = pattern_type!(u32 is 1..);
struct Thing(Y);
fn id_thing(t: Thing) -> Thing { t }
fn describe(z: Option<Y>) -> u32 {
    match z { Some(_) => 1, None => 0 }
}
fn main() {
    let a = id_thing(Thing(2));
    let _ = a.0;
    println!("some {}", describe(Some(42)));
    println!("none {}", describe(None));
    println!("size {}", std::mem::size_of::<Y>());
    let arr = [Thing(1), Thing(2)];
    println!("moved {}", describe(Some(arr[0].0)) + describe(Some(arr[1].0)));
}
"#;

#[test]
fn run_mode_pattern_type_values_execute_as_base_type() {
    // Pattern types (`Ty::Pat`) convert as their base type: the
    // values move through calls, arrays and enum construction
    // with the base representation. (The fixture needs the
    // internal feature; the driver test harness runs with
    // RUSTC_BOOTSTRAP=1, like compiletest.)
    let dir = temp_dir("pattern-type");
    let src = fixture(&dir, "pattern_type.rs", PATTERN_TYPE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "pattern-type program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "some 1\nnone 0\nsize 4\nmoved 2\n",
        "pattern-type stdout mismatch; stderr={stderr}"
    );
}

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

const PIN_REF_DEREF_SRC: &str = r#"
fn main() {
    let mut x = 5;
    let mut p = std::pin::Pin::new(&mut x);
    *p = 8;
    println!("{}", *p);
    println!("{}", x);
}
"#;

#[test]
fn run_mode_pin_ref_deref_reads_and_writes_through() {
    // Dereferencing a `Pin<&mut T>` goes through the wrapped
    // reference: reads and writes land on the pointee.
    let dir = temp_dir("pin-ref-deref");
    let src = fixture(&dir, "pin_ref_deref.rs", PIN_REF_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Pin<&mut T> deref must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "8\n8\n",
        "the write through the Pin is visible on the pointee"
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

const ASYNC_SIZE_OF_VAL_SRC: &str = r#"
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

struct Nop;
impl Future for Nop {
    type Output = u8;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u8> { Poll::Ready(3) }
}

async fn one() -> u8 { Nop.await }

fn main() {
    println!("size {}", std::mem::size_of_val(&one()));
    println!("size {}", std::mem::size_of_val(&Nop));
}
"#;

#[test]
fn run_mode_size_of_val_on_async_future_uses_compiler_layout() {
    // `size_of_val` on a future answers with the compiler's own
    // state-machine layout for that coroutine instance (the same
    // values a native build prints), not with a model guess.
    let dir = temp_dir("async-size-of-val");
    let src = fixture(&dir, "async_size_of_val.rs", ASYNC_SIZE_OF_VAL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "size_of_val on a future must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "size 1\nsize 0\n",
        "the future's size is the compiler layout size"
    );
}

fn run_driver_extra(src: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .args(extra)
        .output()
        .expect("failed to run sparagmos-driver")
}

const ENV_ARGS_SRC: &str = r#"
fn main() {
    for a in std::env::args() {
        println!("arg:{a}");
    }
    let mut os = std::env::args_os();
    println!("os-first {}", os.next().is_some());
    println!("count {}", std::env::args().count());
}
"#;

#[test]
fn run_mode_env_args_reach_the_guest() {
    let dir = temp_dir("env-args");
    let src = fixture(&dir, "args.rs", ENV_ARGS_SRC);
    let out = run_driver_extra(&src, &["--guest-arg", "one", "--guest-arg", "two"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 5, "stdout={stdout}");
    // argv[0] is the guest's program path: the input source.
    assert!(lines[0].starts_with("arg:"), "stdout={stdout}");
    assert!(lines[0].ends_with("args.rs"), "stdout={stdout}");
    assert_eq!(lines[1], "arg:one");
    assert_eq!(lines[2], "arg:two");
    assert_eq!(lines[3], "os-first true");
    assert_eq!(lines[4], "count 3");
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

const PROCESS_EXIT_SRC: &str = r#"
fn main() {
    println!("before");
    std::process::exit(7);
}
"#;

const EXIT_CODE_SRC: &str = r#"
fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(3)
}
"#;

#[test]
fn run_mode_process_exit_and_exit_code() {
    let dir = temp_dir("exit-codes");
    let src = fixture(&dir, "exit.rs", PROCESS_EXIT_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(7), "process::exit(7)");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "before\n");
    let src = fixture(&dir, "exitcode.rs", EXIT_CODE_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(3), "ExitCode::from(3)");
}

const LIBTEST_SRC: &str = r#"
#[test]
fn alpha() { assert_eq!(1 + 1, 2); }
#[test]
#[ignore]
fn gamma() {}
#[test]
#[should_panic]
fn panics() { panic!("expected"); }
"#;

const LIBTEST_FAIL_SRC: &str = r#"
#[test]
fn good() { assert_eq!(2, 2); }
#[test]
fn bad() { assert_eq!(1, 2); }
"#;

#[test]
fn run_mode_libtest_harness_runs_tests() {
    let dir = temp_dir("libtest");
    let src = fixture(&dir, "harness.rs", LIBTEST_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr} stdout={stdout}");
    assert!(stdout.contains("running 3 tests"), "stdout={stdout}");
    assert!(stdout.contains("test alpha ... ok"), "stdout={stdout}");
    assert!(stdout.contains("test panics ... ok"), "stdout={stdout}");
    assert!(stdout.contains("test gamma ... ignored"), "stdout={stdout}");
    assert!(
        stdout.contains("test result: ok. 2 passed; 0 failed; 1 ignored"),
        "stdout={stdout}"
    );
    let src = fixture(&dir, "harness_fail.rs", LIBTEST_FAIL_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "--test"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(stdout.contains("test bad ... FAILED"), "stdout={stdout}");
    assert!(
        stdout.contains("test result: FAILED. 1 passed; 1 failed"),
        "stdout={stdout}"
    );
}

const TIME_SRC: &str = r#"
fn main() {
    let a = std::time::Instant::now();
    let b = std::time::Instant::now();
    assert!(b >= a);
    let s = std::time::SystemTime::now();
    let d = s.duration_since(std::time::UNIX_EPOCH).unwrap();
    assert!(d.as_secs() > 1_600_000_000);
    println!("time ok");
}
"#;

const TIME_OVERFLOW_SRC: &str = r#"
fn main() {
    let t = std::time::Instant::now();
    let _ = t + std::time::Duration::MAX;
}
"#;

#[test]
fn run_mode_time_now_and_overflow() {
    let dir = temp_dir("time");
    let src = fixture(&dir, "time.rs", TIME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "time ok\n");
    let src = fixture(&dir, "time_overflow.rs", TIME_OVERFLOW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("overflow when adding duration to instant"),
        "stderr={stderr}"
    );
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
// ----- reference deref identity and DST metadata (Lane GB) -----------------

const COROUTINE_RESUME_ARGS_SRC: &str = r#"
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let mut gen = #[coroutine] |x: i32| {
        let y = yield x + 1;
        y * 2
    };
    let mut pinned = Pin::new(&mut gen);
    match pinned.as_mut().resume(41) {
        CoroutineState::Yielded(v) => println!("yielded {v}"),
        CoroutineState::Complete(v) => println!("complete {v}"),
    }
    match pinned.as_mut().resume(10) {
        CoroutineState::Yielded(v) => println!("yielded {v}"),
        CoroutineState::Complete(v) => println!("complete {v}"),
    }
}
"#;

#[test]
fn run_mode_coroutine_resume_args_through_pin() {
    // `Pin::as_mut` dereferences the `&mut Coroutine` the pin
    // wraps (`DerefMut` for a mutable reference): the result is
    // the inner reference itself, and the coroutine state machine
    // advances across resumes with the resume argument bound.
    let dir = temp_dir("coroutine-resume-args");
    let src = fixture(&dir, "coroutine_resume_args.rs", COROUTINE_RESUME_ARGS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "yielded 42\ncomplete 20\n", "stdout={stdout}");
}

const DEREF_ON_REFERENCE_SRC: &str = r#"
use std::ops::{Deref, DerefMut};

fn bump(mut r: &mut i32) {
    *r.deref_mut() += 1;
}

fn read(r: &i32) -> i32 {
    *r.deref()
}

fn via_generic<P: DerefMut<Target = i32>>(mut p: P) {
    *p += 10;
}

fn main() {
    let mut x = 1;
    bump(&mut x);
    println!("x {x}");
    println!("read {}", read(&x));
    via_generic(&mut x);
    println!("x {x}");
}
"#;

#[test]
fn run_mode_deref_and_deref_mut_on_references() {
    // `<&mut T as DerefMut>::deref_mut` returns the inner
    // reference itself (writing through it lands in the original
    // place), and `<&T as Deref>::deref` likewise; generic code
    // over `P: DerefMut` instantiated with a reference reaches
    // the same impls.
    let dir = temp_dir("deref-on-reference");
    let src = fixture(&dir, "deref_on_reference.rs", DEREF_ON_REFERENCE_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "x 2\nread 2\nx 12\n", "stdout={stdout}");
}

const DST_STRUCT_SRC: &str = r#"
struct Fat<T: ?Sized> {
    f1: isize,
    f2: &'static str,
    ptr: T,
}

fn foo(x: &Fat<[isize]>) {
    println!("f1 {} f2 {} len {} second {}", x.f1, x.f2, x.ptr.len(), x.ptr[1]);
}

fn main() {
    let f1 = Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    foo(&f1);
    let f3: &Fat<[isize]> = &f1;
    foo(f3);
    let b: Box<[isize]> = Box::new([4, 5, 6, 7]);
    println!("box len {}", b.len());
}
"#;

#[test]
fn run_mode_dst_struct_trailing_slice_metadata() {
    // Unsizing `&Fat<[isize; 3]>` to `&Fat<[isize]>` computes the
    // metadata of the struct value from its trailing field: the
    // slice length, not the unit metadata of a sized value.
    let dir = temp_dir("dst-struct");
    let src = fixture(&dir, "dst_struct.rs", DST_STRUCT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "f1 5 f2 some str len 3 second 2\nf1 5 f2 some str len 3 second 2\nbox len 4\n",
        "stdout={stdout}"
    );
}
const DEBUG_ADT_SRC: &str = r#"
#[derive(Debug)]
struct Named { x: i32, name: String }
#[derive(Debug)]
struct Empty {}
#[derive(Debug)]
struct Tup(i32, bool);
#[derive(Debug)]
struct Tup1(i32);
#[derive(Debug)]
struct Unit;
#[derive(Debug)]
enum E { A, B(i32), C(i32, u8), D { f: i32 }, G {} }
fn main() {
    println!("{:?}", Named { x: 3, name: "hi".to_string() });
    println!("{:?}", Empty {});
    println!("{:?}", Tup(1, true));
    println!("{:?}", Tup1(9));
    println!("{:?}", Unit);
    println!("{:?}", E::A);
    println!("{:?}", E::B(5));
    println!("{:?}", E::C(5, 6));
    println!("{:?}", E::D { f: 7 });
    println!("{:?}", E::G {});
}
"#;

#[test]
fn run_mode_debug_format_derived_adts() {
    let dir = temp_dir("debug-adt");
    let src = fixture(&dir, "debug_adt.rs", DEBUG_ADT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "Named { x: 3, name: \"hi\" }\n\
         Empty\n\
         Tup(1, true)\n\
         Tup1(9)\n\
         Unit\n\
         A\n\
         B(5)\n\
         C(5, 6)\n\
         D { f: 7 }\n\
         G\n",
        "stdout={stdout}"
    );
}

const DEBUG_ADT_PRETTY_SRC: &str = r#"
#[derive(Debug)]
struct Named { x: i32, name: String }
#[derive(Debug)]
struct Tup(i32, bool);
#[derive(Debug)]
enum E { D { f: i32 } }
#[derive(Debug)]
struct Outer { inner: Named, list: Vec<Tup> }
fn main() {
    println!("{:#?}", Named { x: 3, name: "hi".to_string() });
    println!("{:#?}", Tup(1, true));
    println!("{:#?}", E::D { f: 7 });
    println!("{:#?}", Outer { inner: Named { x: 1, name: "z".into() }, list: vec![Tup(3, true)] });
    println!("{:#?}", Vec::<i32>::new());
}
"#;

#[test]
fn run_mode_debug_format_pretty_adts() {
    let dir = temp_dir("debug-adt-pretty");
    let src = fixture(&dir, "debug_adt_pretty.rs", DEBUG_ADT_PRETTY_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "Named {\n    x: 3,\n    name: \"hi\",\n}\n\
         Tup(\n    1,\n    true,\n)\n\
         D {\n    f: 7,\n}\n\
         Outer {\n    inner: Named {\n        x: 1,\n        name: \"z\",\n    },\n    list: [\n        Tup(\n            3,\n            true,\n        ),\n    ],\n}\n\
         []\n",
        "stdout={stdout}"
    );
}

const DEBUG_ADT_CONTAINERS_SRC: &str = r#"
#[derive(Debug)]
struct Named { x: i32, name: String }
#[derive(Debug)]
struct Tup(i32, bool);
#[derive(Debug)]
struct Outer { inner: Named, list: Vec<Tup>, opt: Option<i32> }
#[derive(Debug)]
struct Pair<T> { a: T, b: T }
fn main() {
    println!("{:?}", Some(Named { x: 1, name: "a".into() }));
    println!("{:?}", Ok::<i32, String>(3));
    println!("{:?}", Err::<i32, String>("bad".into()));
    println!("{:?}", vec![Tup(1, false), Tup(2, true)]);
    println!("{:?}", [Named { x: 2, name: "b".into() }]);
    println!("{:?}", (Tup(4, true), 8));
    println!("{:?}", Outer { inner: Named { x: 1, name: "z".into() }, list: vec![Tup(3, true)], opt: None });
    let v = vec![Named { x: 1, name: "p".into() }, Named { x: 2, name: "q".into() }];
    println!("{:?}", &v[..]);
    println!("{:?}", Pair { a: 1, b: 2 });
    println!("{:?}", Pair { a: "s", b: "t" });
    println!("{:?}", format!("{:?}", Named { x: 8, name: "f".into() }));
}
"#;

#[test]
fn run_mode_debug_format_adt_in_containers() {
    let dir = temp_dir("debug-adt-containers");
    let src = fixture(&dir, "debug_adt_containers.rs", DEBUG_ADT_CONTAINERS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "Some(Named { x: 1, name: \"a\" })\n\
         Ok(3)\n\
         Err(\"bad\")\n\
         [Tup(1, false), Tup(2, true)]\n\
         [Named { x: 2, name: \"b\" }]\n\
         (Tup(4, true), 8)\n\
         Outer { inner: Named { x: 1, name: \"z\" }, list: [Tup(3, true)], opt: None }\n\
         [Named { x: 1, name: \"p\" }, Named { x: 2, name: \"q\" }]\n\
         Pair { a: 1, b: 2 }\n\
         Pair { a: \"s\", b: \"t\" }\n\
         \"Named { x: 8, name: \\\"f\\\" }\"\n",
        "stdout={stdout}"
    );
}

const DEBUG_ADT_REFS_SRC: &str = r#"
use std::fmt::Debug;
#[derive(Debug)]
struct Named { x: i32 }
#[derive(Debug)]
struct Tup(i32, bool);
#[derive(Debug)]
enum E { V(i32), W }
fn main() {
    let n = Named { x: 3 };
    println!("{:?}", &n);
    println!("{:?}", Box::new(Tup(7, false)));
    let d: &dyn Debug = &Named { x: 5 };
    println!("{:?}", d);
    let d2: Box<dyn Debug> = Box::new(E::V(3));
    println!("{:?}", d2);
    let d3: &dyn Debug = &E::W;
    println!("{:?}", d3);
}
"#;

#[test]
fn run_mode_debug_format_refs_box_and_dyn() {
    let dir = temp_dir("debug-adt-refs");
    let src = fixture(&dir, "debug_adt_refs.rs", DEBUG_ADT_REFS_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "Named { x: 3 }\n\
         Tup(7, false)\n\
         Named { x: 5 }\n\
         V(3)\n\
         W\n",
        "stdout={stdout}"
    );
}

const DEBUG_RANGES_SRC: &str = r#"
fn main() {
    println!("{:?}", std::ops::Range { start: 1, end: 4 });
    println!("{:?}", 1..=4);
    println!("{:?}", 1..);
    println!("{:?}", ..4);
    println!("{:?}", ..=4);
    println!("{:?}", ..);
    println!("{:?}", 'a'..='z');
    println!("{:#?}", 1..4);
}
"#;

#[test]
fn run_mode_debug_format_ranges() {
    let dir = temp_dir("debug-ranges");
    let src = fixture(&dir, "debug_ranges.rs", DEBUG_RANGES_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout,
        "1..4\n1..=4\n1..\n..4\n..=4\n..\n'a'..='z'\n1..4\n",
        "stdout={stdout}"
    );
}
