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
