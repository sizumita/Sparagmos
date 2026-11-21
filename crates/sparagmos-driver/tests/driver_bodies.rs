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

const ASYNC_CLOSURE_MAIN_RS: &str = "\
fn main() {
    let f = async || 42;
    let _fut = f();
    println!(\"defined\");
}
";

const ASYNC_CLOSURE_HELPER_RS: &str = "\
fn make() {
    let f = async || 42;
    let _fut = f();
}
fn main() {
    println!(\"main only\");
}
";

fn run_driver_stderr(args: &[&str], fixture: &Path) -> (String, String, std::process::ExitStatus) {
    let out = Command::new(driver())
        .args(args)
        .arg(fixture)
        .env_remove("SPARAGMOS_DUMP")
        .output()
        .expect("spawn driver");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status,
    )
}

#[test]
fn async_closure_in_main_is_skipped_with_named_reason() {
    // Stable MIR 1.100.0 cannot convert a body that mentions a
    // coroutine-closure (async closure) type at all (rustc_public
    // leaves ty::CoroutineClosure unimplemented), so the body is
    // skipped with that named reason instead of the conversion
    // panic aborting the whole crate conversion.
    let fixture = write_fixture("async_closure_main", ASYNC_CLOSURE_MAIN_RS);
    let (stdout, status) = run_driver(&[], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains(
            "fn async_closure_main::main: SKIPPED (body mentions a coroutine-closure"
        ),
        "main not skipped with the coroutine-closure reason:\n{stdout}"
    );
}

#[test]
fn async_closure_in_uncalled_helper_does_not_block_main() {
    // Only the function whose body mentions the async closure is
    // skipped; the rest of the crate converts and runs.
    let fixture = write_fixture("async_closure_helper", ASYNC_CLOSURE_HELPER_RS);
    let (stdout, status) = run_driver(&[], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains(
            "fn async_closure_helper::make: SKIPPED (body mentions a coroutine-closure"
        ),
        "helper not skipped with the coroutine-closure reason:\n{stdout}"
    );
    assert!(
        stdout.contains("fn async_closure_helper::main: blocks="),
        "main not converted:\n{stdout}"
    );
    let (stdout, status) = run_driver(&["run"], &fixture);
    assert!(status.success(), "run failed: {stdout}");
    assert_eq!(stdout, "main only\n");
}

#[test]
fn run_main_with_async_closure_fails_with_named_reason() {
    // Run mode on a program whose main mentions an async closure
    // fails explicitly (exit 2), naming the skip reason — never a
    // conversion panic.
    let fixture = write_fixture("async_closure_main_run", ASYNC_CLOSURE_MAIN_RS);
    let (_stdout, stderr, status) = run_driver_stderr(&["run"], &fixture);
    assert_eq!(status.code(), Some(2), "run status: {status:?}, stderr: {stderr}");
    assert!(
        stderr.contains("`main` was not converted (body mentions a coroutine-closure"),
        "run failure does not name the coroutine-closure reason:\n{stderr}"
    );
    assert!(
        !stderr.contains("conversion panicked"),
        "run failure is a conversion panic:\n{stderr}"
    );
}
