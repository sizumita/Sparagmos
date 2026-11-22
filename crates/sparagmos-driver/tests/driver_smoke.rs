//! End-to-end smoke test: builds a tiny crate in a temp dir, runs the
//! driver binary on it, and checks that Stable MIR acquisition and the
//! conversion summary work. Requires the driver to be built with the
//! pinned 1.100.0 toolchain (see crate README); the test itself only
//! spawns the binary.

use std::process::Command;

const FIXTURE: &str = r#"
pub fn add(a: i32, b: i32) -> i32 { a + b }
pub fn fib(n: u32) -> u32 { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } }
fn main() { println!("{}", add(2, 3) + fib(10) as i32); }
"#;

#[test]
fn driver_converts_simple_functions() {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-smoke-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("smoke.rs");
    std::fs::write(&src, FIXTURE).unwrap();
    let dump = dir.join("smoke.dump");

    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg(&src)
        .arg("--dump")
        .arg(&dump)
        .output()
        .expect("failed to run sparagmos-driver");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "driver failed: status={:?}\nstdout={stdout}\nstderr={stderr}",
        out.status
    );
    assert!(stdout.contains("add"), "summary should mention add:\n{stdout}");
    assert!(stdout.contains("fib"), "summary should mention fib:\n{stdout}");
    assert!(
        stdout.contains("converted:"),
        "summary should report counts:\n{stdout}"
    );
    let dump_text = std::fs::read_to_string(&dump).expect("dump file written");
    assert!(dump_text.starts_with("sparagmos-mir-dump v0"));
    assert!(dump_text.contains("=== fn"), "dump lists functions");
}

const BOX_FIXTURE: &str = r#"
fn main() {
    let b = Box::new(41);
    println!("{}", *b + 1);
}
"#;

/// A `Box` program: deref/drop elaboration in `main` extracts the raw
/// pointer out of the box with a `BoxDerefTransmute` cast. That cast
/// used to make the driver skip `main` entirely; it must now convert.
#[test]
fn driver_converts_box_deref_transmute() {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-box-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("box.rs");
    std::fs::write(&src, BOX_FIXTURE).unwrap();
    let dump = dir.join("box.dump");

    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg(&src)
        .arg("--dump")
        .arg(&dump)
        .output()
        .expect("failed to run sparagmos-driver");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "driver failed: status={:?}\nstdout={stdout}\nstderr={stderr}",
        out.status
    );
    assert!(
        stdout.contains("fn box::main:") && stdout.contains("[converted]"),
        "main should be reported as converted:\n{stdout}"
    );
    // Since Lane AG the report also lists callees the collector
    // visited; builtin-covered callees appear as SKIPPED with that
    // reason, which is not a conversion failure. What this test
    // guards is that main itself converts despite the cast.
    assert!(
        !stdout.contains("fn box::main: SKIPPED"),
        "main must not be skipped:\n{stdout}"
    );
    let dump_text = std::fs::read_to_string(&dump).expect("dump file written");
    assert!(
        dump_text.contains("=== fn box::main"),
        "dump lists main as a converted function:\n{dump_text}"
    );
    assert!(
        dump_text.contains("BoxDerefTransmute"),
        "dump shows the converted cast:\n{dump_text}"
    );
}

const CFG_FIXTURE: &str = r#"
#[cfg(flag_on)]
pub fn gated() -> u32 { 1 }
pub fn plain() -> u32 { 2 }
fn main() {}
"#;

#[test]
fn driver_forwards_rustc_args() {
    // A caller-supplied `--cfg` must reach the embedded rustc: the
    // cfg-gated function is only converted when the flag is forwarded
    // (this is how the subset runner honours `//@ compile-flags`).
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-rustc-arg-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("cfg.rs");
    std::fs::write(&src, CFG_FIXTURE).unwrap();
    let bin = env!("CARGO_BIN_EXE_sparagmos-driver");

    let off = Command::new(bin).arg(&src).output().unwrap();
    let off_stdout = String::from_utf8_lossy(&off.stdout);
    assert!(off.status.success(), "driver failed: {off_stdout}");
    assert!(
        !off_stdout.contains("gated"),
        "gated fn must be cfg'd out without the flag:\n{off_stdout}"
    );

    let on = Command::new(bin)
        .arg(&src)
        .arg("--rustc-arg")
        .arg("--cfg")
        .arg("--rustc-arg")
        .arg("flag_on")
        .output()
        .unwrap();
    let on_stdout = String::from_utf8_lossy(&on.stdout);
    assert!(on.status.success(), "driver failed: {on_stdout}");
    assert!(
        on_stdout.contains("gated"),
        "gated fn must be converted with the flag:\n{on_stdout}"
    );
}

// ---------------------------------------------------------------------
// Convert-mode exit-code contract (Issue #66): 0 = converted,
// 1 = the compilation failed, 101 = the compiler crashed. A crash
// is reported only for a genuine internal failure of the compiler
// (an internal compiler error, or a panic that escaped the
// compiler run); a designed abort on a reported diagnostic and a
// clean rejection stay compilation failures.

fn convert_fixture(name: &str, fixture: &str, rustc_args: &[&str]) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-contract-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join(format!("{name}.rs"));
    std::fs::write(&src, fixture).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg(&src).arg("--edition").arg("2021");
    // Mirror the subset runner, whose invocations carry the
    // bootstrap env the embedded nightly rustc needs for -Z flags
    // and in-source feature gates.
    cmd.env("RUSTC_BOOTSTRAP", "1");
    for a in rustc_args {
        cmd.arg("--rustc-arg").arg(a);
    }
    let out = cmd.output().expect("failed to run sparagmos-driver");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The rustc test suite's known-bug #100041 shape: the compiler
/// creates a delayed bug during compilation and flushes it by
/// panicking at the end of the run, after printing it as an
/// internal compiler error.
const DELAYED_BUG_FIXTURE: &str = r#"
pub trait WellUnformed {
    type RequestNormalize;
}

impl<T: ?Sized> WellUnformed for T {
    type RequestNormalize = ();
}

pub fn latent(_: &[<[[()]] as WellUnformed>::RequestNormalize; 0]) {}

pub fn bang() {
    latent(&[]);
}

fn main() {}
"#;

#[test]
fn convert_mode_delayed_bug_is_a_compiler_crash() {
    let (code, stderr) = convert_fixture("delayed", DELAYED_BUG_FIXTURE, &[]);
    assert_eq!(code, Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("internal compiler error"),
        "the compiler's own ICE text is printed:\n{stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compiler crashed:")
            && stderr.contains("delayed bug"),
        "the crash report names the delayed bug:\n{stderr}"
    );
}

/// The known-bug #105238 shape: instantiating the generic const
/// expression panics inside the compiler and the panic escapes the
/// compiler run.
const CONVERSION_PANIC_FIXTURE: &str = r#"
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

trait Ret {
    type R;
}

struct Cond<const PRED: bool, U, V>(std::marker::PhantomData<U>, std::marker::PhantomData<V>);

impl<U, V> Ret for Cond<true, U, V> {
    type R = U;
}

impl<U, V> Ret for Cond<false, U, V> {
    type R = V;
}

struct RobinHashTable<const MAX_LENGTH: usize, CellIdx = Cond<{ MAX_LENGTH < 65535 }, u16, u32>>
where
    CellIdx: Ret,
{
    _idx: CellIdx::R,
}

fn main() {
    use std::mem::size_of;
    println!("{}", size_of::<RobinHashTable<1024>>());
    println!("{}", size_of::<RobinHashTable<65536>>());
}
"#;

#[test]
fn convert_mode_escaped_conversion_panic_is_a_compiler_crash() {
    let (code, stderr) = convert_fixture("panic", CONVERSION_PANIC_FIXTURE, &[]);
    assert_eq!(code, Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("sparagmos-driver: Stable MIR conversion panicked:"),
        "the escaped panic keeps its report line:\n{stderr}"
    );
}

const CONST_EVAL_FIXTURE: &str = r#"
const FOO: i32 = 1 / 0;
fn main() { println!("{}", FOO); }
"#;

#[test]
fn convert_mode_treat_err_abort_is_a_compilation_failure() {
    // `-Z treat-err-as-bug` orders the compiler to abort on the
    // reported diagnostic; that designed abort is a compilation
    // failure, not a crash, even though the diagnostic is a
    // const-eval error the flag promotes to a bug.
    let (code, stderr) = convert_fixture(
        "treaterr",
        CONST_EVAL_FIXTURE,
        &["-Ztreat-err-as-bug=1"],
    );
    assert_eq!(code, Some(1), "stderr={stderr}");
    assert!(
        stderr.contains("sparagmos-driver: compilation failed:")
            && stderr.contains("treat-err-as-bug"),
        "the abort takes the compilation-failure report:\n{stderr}"
    );
    assert!(
        !stderr.contains("compiler crashed"),
        "a designed abort is not a crash:\n{stderr}"
    );
}

#[test]
fn convert_mode_clean_rejection_is_a_compilation_failure() {
    let (code, stderr) = convert_fixture(
        "cleanfail",
        "fn main() { let x: i32 = \"s\"; }\n",
        &[],
    );
    assert_eq!(code, Some(1), "stderr={stderr}");
    assert!(
        stderr.contains("sparagmos-driver: compilation failed:"),
        "a clean rejection takes the compilation-failure report:\n{stderr}"
    );
    assert!(
        !stderr.contains("compiler crashed"),
        "a clean rejection is not a crash:\n{stderr}"
    );
}
