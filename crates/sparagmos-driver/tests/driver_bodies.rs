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
    let _stdin = std::io::stdin();
    let n = std::env::args().count();
    println!(\"{}\", n);
}
";

#[test]
fn report_records_bodyless_targets_with_reasons() {
    // Fixture note: this used to use Vec::push reaching the
    // allocator shims, but since Lane V those Vec operations are
    // interpreter builtins and are no longer collected; env::args
    // joined them later (Lane FC), so std::io::stdin — a
    // non-generic function of another crate, for which no MIR body
    // is shipped — stands as the body-less witness here.
    // Iterator::count is likewise an interpreter builtin since
    // Lane EE and is reported with the builtin reason; the local
    // main body stands in as the collected-body witness.
    let fixture = write_fixture("args", ARGS_RS);
    let (stdout, status) = run_driver(&[], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains("fn std::io::stdin: SKIPPED (no MIR body available"),
        "body-less target not reported with its reason:\n{stdout}"
    );
    assert!(
        stdout.contains(
            "fn std::iter::Iterator::count: SKIPPED (implemented as an interpreter builtin"
        ),
        "builtin-owned target not reported with its reason:\n{stdout}"
    );
    assert!(
        stdout.contains("fn args::main: blocks="),
        "collected main body missing:\n{stdout}"
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

const BINARY_SEARCH_RS: &str = "\
fn main() {
    let v: &[i32] = &[1, 2, 3, 4, 5, 6, 9];
    match v.binary_search(&5) {
        Ok(i) => println!(\"found at {i}\"),
        Err(i) => println!(\"insert at {i}\"),
    }
}
";

#[test]
fn foreign_closure_inside_generic_std_function_is_collected() {
    // `slice::binary_search` is a generic core function whose body
    // creates a closure (`|p| p.cmp(x)`) and hands it to
    // `binary_search_by`. The closure is defined in core, so no
    // item loop ever defers it and the reachability collection
    // (which follows FnDef targets only) never demands it; the
    // closure demand pass collects it from the mention in the
    // collected `binary_search` body.
    let fixture = write_fixture("binary_search", BINARY_SEARCH_RS);
    let (stdout, status) = run_driver(&[], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains(
            "fn core::slice::<impl [T]>::binary_search::{closure#0}: blocks="
        ),
        "foreign closure not reported as converted:\n{stdout}"
    );
    let dump_path = fixture.with_extension("dump.txt");
    let dump_arg = dump_path.to_string_lossy().into_owned();
    let (stdout, status) = run_driver(&["--dump", &dump_arg], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    let dump = std::fs::read_to_string(&dump_path).expect("read dump file");
    assert!(
        dump.contains("=== fn core::slice::<impl [T]>::binary_search::{closure#0}"),
        "foreign closure body missing from dump:\n{dump}"
    );
}

const HASHMAP_RS: &str = "\
use std::collections::HashMap;
fn main() {
    let mut m = HashMap::new();
    m.insert(1, 2);
    println!(\"{:?}\", m.get(&1));
}
";

#[test]
fn foreign_closures_inside_hashbrown_and_random_state_are_collected() {
    // The closures hashbrown's table code and
    // `RandomState::new`'s `KEYS.with(...)` create are defined in
    // their own crates; before foreign closure collection they
    // were demanded by no pass and calls to them failed as
    // unknown functions.
    let fixture = write_fixture("hashmap", HASHMAP_RS);
    let (stdout, status) = run_driver(&[], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    for name in [
        "fn std::hash::RandomState::new::{closure#0}: blocks=",
        "fn hashbrown::map::make_hasher::{closure#0}: blocks=",
        "fn hashbrown::raw::RawTableInner::rehash_in_place::{closure#0}: blocks=",
    ] {
        assert!(
            stdout.contains(name),
            "foreign closure {name} not reported as converted:\n{stdout}"
        );
    }
}

const HASHMAP_INDEX_RS: &str = "\
use std::collections::HashMap;
fn main() {
    let mut m: HashMap<u32, u32> = HashMap::new();
    m.insert(1, 2);
    println!(\"{}\", m[&1]);
}
";

#[test]
fn closure_mentioned_only_in_a_constant_is_collected() {
    // `RandomState::new`'s thread-local keys live in the `KEYS`
    // constant, whose stored function pointer names the
    // initializer closure. No converted body's types mention
    // that closure — the decoded constant value is the only
    // mention — so only the constant-content demand collects its
    // body; without it, programs using `HashMap` failed at run
    // time with an unknown function when `LocalKey` invoked the
    // stored pointer.
    let fixture = write_fixture("hashmap_const_closure", HASHMAP_INDEX_RS);
    let dump_path = fixture.with_extension("dump.txt");
    let dump_arg = dump_path.to_string_lossy().into_owned();
    let (stdout, status) = run_driver(&["--dump", &dump_arg], &fixture);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains(
            "fn std::hash::RandomState::new::KEYS::{constant#0}::{closure#1}: blocks="
        ),
        "constant-mentioned closure not reported converted:\n{stdout}"
    );
    let dump = std::fs::read_to_string(&dump_path).expect("read dump file");
    assert!(
        dump.contains(
            "=== fn std::hash::RandomState::new::KEYS::{constant#0}::{closure#1}"
        ),
        "constant-mentioned closure body missing from dump:\n{dump}"
    );
}
