//! End-to-end check of the reachability collection's two
//! orders (see `sparagmos_driver::reach::CollectionOrder`):
//! convert mode walks first-in-first-out from every local body,
//! while run mode walks execution-first from the entry point,
//! so a foreign body the entry calls is collected before one
//! only an uncalled local function demands. The dump lists the
//! program's bodies in collection order, which is where the
//! difference is observable.

use std::process::Command;

const FIXTURE: &str = r#"
fn unused() -> u8 {
    let a = std::cmp::min(3u8, 4u8);
    let b = std::num::Wrapping(10u8) + std::num::Wrapping(1u8);
    a + b.0
}
fn main() {
    let c = std::cmp::max(5u16, 6u16);
    let d = std::mem::size_of::<u64>();
    println!("{} {}", c, d);
}
"#;

fn run_driver(dir: &std::path::Path, mode: Option<&str>) -> String {
    let src = dir.join("fixture.rs");
    std::fs::write(&src, FIXTURE).unwrap();
    let dump = dir.join("fixture.dump");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    if let Some(mode) = mode {
        cmd.arg(mode);
    }
    let out = cmd
        .arg(&src)
        .arg("--dump")
        .arg(&dump)
        .output()
        .expect("failed to run sparagmos-driver");
    assert!(
        out.status.success(),
        "driver failed: status={:?}\nstdout={}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read_to_string(&dump).expect("dump file written")
}

fn fn_position(dump: &str, name: &str) -> usize {
    let needle = format!("=== fn {name} (def=");
    dump
        .find(&needle)
        .unwrap_or_else(|| panic!("{name} not in dump:\n{dump}"))
}

#[test]
fn convert_mode_collects_in_seed_order() {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-order-convert-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let dump = run_driver(&dir, None);
    // `unused` is defined (and so seeded) before `main`, so the
    // FIFO walk records its demand (`std::cmp::min`) first.
    assert!(
        fn_position(&dump, "std::cmp::min") < fn_position(&dump, "std::cmp::max"),
        "convert mode must keep the historical seed order:\n{dump}"
    );
}

#[test]
fn run_mode_collects_execution_first() {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-order-run-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let dump = run_driver(&dir, Some("run"));
    // The entry's own call (`std::cmp::max`) is collected
    // before the uncalled local function's demand, whatever
    // the seed order is.
    assert!(
        fn_position(&dump, "std::cmp::max") < fn_position(&dump, "std::cmp::min"),
        "run mode must collect the entry's closure first:\n{dump}"
    );
}
