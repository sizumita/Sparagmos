//! Collected library bodies: functions from std / core reachable
//! from the target program are collected into the program — the
//! dump shows their converted bodies, body-less targets are
//! reported with reasons, and layout-agnostic bodies execute.

use std::path::{Path, PathBuf};
use std::process::Command;

fn driver() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sparagmos-driver"))
}

fn write_fixture(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-bodies-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let path = dir.join(format!("{name}.rs"));
    std::fs::write(&path, src).expect("write fixture");
    path
}

fn run_driver(args: &[&str], fixture: &Path) -> (String, std::process::ExitStatus) {
    let out = Command::new(driver())
        .args(args)
        .arg(fixture)
        .env_remove("SPARAGMOS_DUMP")
        .output()
        .expect("spawn driver");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status,
    )
}

const UNWRAP_RS: &str = "\
fn main() {
    let a = Some(40u8).unwrap();
    let b = None::<u8>.unwrap_or(7);
    println!(\"{} {}\", a, b);
}
";

const VEC_RS: &str = "\
fn main() {
    let mut v: Vec<u8> = Vec::new();
    v.push(1);
    println!(\"{}\", v.len());
}
";

#[test]
fn dump_contains_collected_core_bodies() {
    let fixture = write_fixture("unwrap", UNWRAP_RS);
    let dump_path = fixture.with_extension("dump.txt");
    let dump_arg = dump_path.to_string_lossy().into_owned();
    let (stdout, status) = run_driver(&["--dump", &dump_arg], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    let dump = std::fs::read_to_string(&dump_path).expect("read dump file");
    // The dump contains a body section for the collected
    // `Option::unwrap` (a core function, not part of the target
    // crate), under its plain definition name.
    assert!(
        dump.contains("=== fn std::option::Option::<T>::unwrap"),
        "collected unwrap body missing from dump:\n{dump}"
    );
}

#[test]
fn run_executes_collected_unwrap_bodies() {
    let fixture = write_fixture("unwrap_run", UNWRAP_RS);
    let (stdout, status) = run_driver(&["run"], &fixture);
    assert!(status.success(), "run failed: {stdout}");
    assert_eq!(stdout, "40 7\n");
}

const ARGS_RS: &str = "\
fn main() {
    let n = std::env::args().count();
    println!(\"{}\", n);
}
";

#[test]
fn report_records_bodyless_targets_with_reasons() {
    // Fixture note: this used to use Vec::push reaching the
    // allocator shims, but since Lane V those Vec operations are
    // interpreter builtins and are no longer collected. env::args
    // is a non-generic function of another crate, so no MIR body
    // is shipped for it; Iterator::count is collected instead.
    let fixture = write_fixture("args", ARGS_RS);
    let (stdout, status) = run_driver(&[], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains("fn std::env::args: SKIPPED (no MIR body available"),
        "body-less target not reported with its reason:\n{stdout}"
    );
    assert!(
        stdout.contains("fn std::iter::Iterator::count: blocks="),
        "collected Iterator::count body missing:\n{stdout}"
    );
}
