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
