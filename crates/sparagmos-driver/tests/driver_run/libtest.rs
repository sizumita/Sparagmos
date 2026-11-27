use super::{Command, fixture, run_driver_extra, temp_dir};

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
    assert!(
        stdout.contains("test panics - should panic ... ok"),
        "stdout={stdout}"
    );
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

const LISTING_SRC: &str = r#"
#[test]
fn beta() {}
#[test]
fn alpha() {}
"#;

#[test]
fn run_mode_libtest_listing_and_nightly_only_options() {
    let dir = temp_dir("libtest-listing");
    let src = fixture(&dir, "listing.rs", LISTING_SRC);
    // `--list` prints the sorted test names and succeeds.
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "--list"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout={stdout}");
    assert!(stdout.contains("alpha: test"), "stdout={stdout}");
    assert!(stdout.contains("beta: test"), "stdout={stdout}");
    assert!(stdout.contains("2 tests, 0 benchmarks"), "stdout={stdout}");
    // `--format json` without `-Z unstable-options` is the nightly
    // error, exit 101.
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "--format",
            "--guest-arg",
            "json",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("The \"json\" format is only accepted on the nightly compiler"),
        "stderr={stderr}"
    );
    // Any `-Z` option is rejected the same way.
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "-Zunstable-options"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("the option `Z` is only accepted on the nightly compiler"),
        "stderr={stderr}"
    );}
const LIBTEST_SP_SRC: &str = r#"
#[test]
#[should_panic(expected = "hello")]
fn wrong_msg() { panic!("bye"); }
#[test]
#[should_panic]
fn no_panic() {}
#[test]
#[ignore = "later"]
fn skipped() {}
"#;

#[test]
fn run_mode_libtest_harness_output_fidelity() {
    let dir = temp_dir("libtest-fidelity");
    let src = fixture(&dir, "sp.rs", LIBTEST_SP_SRC);
    // `--list`: one line per test plus the count footer, and
    // nothing runs.
    let out = run_driver_extra(&src, &["--rustc-arg", "--test", "--guest-arg", "--list"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout={stdout}");
    assert_eq!(
        stdout, "no_panic: test\nskipped: test\nwrong_msg: test\n\n3 tests, 0 benchmarks\n",
        "stdout={stdout}"
    );
    // Should-panic verdicts: a panic whose message misses the
    // expected substring and a test that does not panic both
    // fail, with the harness's notes as their failure sections;
    // the ignored test's line carries its ignore message.
    let out = run_driver_extra(&src, &["--rustc-arg", "--test", "--guest-arg", "--nocapture"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(
        stdout.starts_with("\nrunning 3 tests\n"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("test no_panic - should panic ... FAILED"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("note: test did not panic as expected at "),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "note: panic did not contain expected string\n      panic message: \"bye\"\n expected substring: \"hello\""
        ),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("test skipped ... ignored, later"),
        "stdout={stdout}"
    );
    // Multiple positional filters select the union of matches.
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "wrong_msg",
            "--guest-arg",
            "no_panic",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("running 2 tests"), "stdout={stdout}");
    assert!(stdout.contains("1 filtered out"), "stdout={stdout}");

}

const LIBTEST_SP_FAIL_SRC: &str = r#"
#[test]
#[should_panic(expected = "hello")]
fn wrong_msg() { panic!("bye"); }
#[test]
#[should_panic(expected = "message")]
fn non_string() { std::panic::panic_any(123); }
"#;

#[test]
fn run_mode_libtest_nocapture_failure_sections_carry_notes_only() {
    let dir = temp_dir("libtest-nocapture-sections");
    let src = fixture(&dir, "spfail.rs", LIBTEST_SP_FAIL_SRC);
    // Under `--nocapture` the panic reports go to the guest's
    // stderr as the panics run, and the failure sections on
    // stdout carry the harness's notes alone - no copy of the
    // report. The non-string note names the payload by its
    // `TypeId`, in std's form.
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "--test-threads=1",
            "--guest-arg",
            "--nocapture",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(
        stdout.contains(
            "---- non_string stdout ----\nnote: expected panic with string value,\n found non-string value: `TypeId(0x"
        ),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "---- wrong_msg stdout ----\nnote: panic did not contain expected string\n      panic message: \"bye\"\n expected substring: \"hello\""
        ),
        "stdout={stdout}"
    );
    assert!(
        !stdout.contains("panicked at"),
        "the panic reports must not leak into the sections: stdout={stdout}"
    );
    assert!(
        stderr.contains("thread 'non_string' (1) panicked at ")
            && stderr.contains("Box<dyn Any>")
            && stderr.contains("thread 'wrong_msg' (1) panicked at "),
        "stderr={stderr}"
    );
}

#[test]
fn run_mode_libtest_capture_failure_sections_carry_the_report() {
    let dir = temp_dir("libtest-capture-sections");
    let src = fixture(&dir, "spfail.rs", LIBTEST_SP_FAIL_SRC);
    // In capture mode the same failures' sections carry the
    // captured panic report ahead of the note, as the native
    // harness's captured sections do.
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "--test-threads=1"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(
        stdout.contains("thread 'non_string' (1) panicked at ")
            && stdout.contains("found non-string value: `TypeId(0x"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("thread 'wrong_msg' (1) panicked at ")
            && stdout.contains("note: panic did not contain expected string"),
        "stdout={stdout}"
    );
    assert!(stderr.is_empty(), "stderr={stderr}");
}

const LIBTEST_JSON_SRC: &str = r#"
#[test]
fn a() {
    println!("print from successful test");
}
#[test]
fn b() {
    assert!(false);
}
#[test]
#[should_panic]
fn c() {
    assert!(false);
}
#[test]
#[ignore = "msg"]
fn d() {
    assert!(false);
}
"#;

#[test]
fn run_mode_libtest_json_format_output() {
    // Lane IH: `--format json` together with `-Z
    // unstable-options` is the JSON Lines event stream (the
    // shape tests/run-make/libtest-json snapshots), for both
    // runs and `--list` discovery.
    let dir = temp_dir("libtest-json");
    let src = fixture(&dir, "f.rs", LIBTEST_JSON_SRC);
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "-Zunstable-options",
            "--guest-arg",
            "--format=json",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(
        stdout.contains("{ \"type\": \"suite\", \"event\": \"started\", \"test_count\": 4 }"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("{ \"type\": \"test\", \"name\": \"a\", \"event\": \"ok\" }"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("\"event\": \"failed\", \"stdout\": \"\\nthread 'b'"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "{ \"type\": \"test\", \"name\": \"d\", \"event\": \"ignored\", \"message\": \"msg\" }"
        ),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "{ \"type\": \"suite\", \"event\": \"failed\", \"passed\": 2, \"failed\": 1, \
             \"ignored\": 1, \"measured\": 0, \"filtered_out\": 0"
        ),
        "stdout={stdout}"
    );
    // `--show-output` adds the captured output to passing
    // tests, including the report of an expected panic.
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "-Zunstable-options",
            "--guest-arg",
            "--format=json",
            "--guest-arg",
            "--show-output",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(
        stdout.contains(
            "\"name\": \"a\", \"event\": \"ok\", \"stdout\": \"print from successful test\\n\""
        ),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("\"name\": \"c\", \"event\": \"ok\", \"stdout\": \"\\nthread 'c'"),
        "stdout={stdout}"
    );
    // The discovery listing: descriptor spans ride along.
    let out = run_driver_extra(
        &src,
        &[
            "--rustc-arg",
            "--test",
            "--guest-arg",
            "--list",
            "--guest-arg",
            "--format=json",
            "--guest-arg",
            "-Zunstable-options",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout={stdout}");
    assert!(
        stdout.contains("{ \"type\": \"suite\", \"event\": \"discovery\" }"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "\"event\": \"discovered\", \"name\": \"d\", \"ignore\": true, \
             \"ignore_message\": \"msg\""
        ),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains(
            "{ \"type\": \"suite\", \"event\": \"completed\", \"tests\": 4, \
             \"benchmarks\": 0, \"total\": 4, \"ignored\": 1 }"
        ),
        "stdout={stdout}"
    );
}

#[test]
fn run_mode_libtest_ignored_modes() {
    // Lane IH: `--ignored` runs only the ignored tests (the
    // rest count as filtered out); `--include-ignored` runs
    // them with the rest.
    let dir = temp_dir("libtest-ignored");
    let src = fixture(&dir, "f.rs", LIBTEST_JSON_SRC);
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "--ignored"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(stdout.contains("running 1 test"), "stdout={stdout}");
    assert!(
        stdout.contains("0 passed; 1 failed; 0 ignored"),
        "stdout={stdout}"
    );
    assert!(stdout.contains("3 filtered out"), "stdout={stdout}");
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "--include-ignored"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(101), "stdout={stdout}");
    assert!(stdout.contains("running 4 tests"), "stdout={stdout}");
    assert!(
        stdout.contains("2 passed; 2 failed; 0 ignored"),
        "stdout={stdout}"
    );
}

#[test]
fn run_mode_libtest_invalid_rust_test_threads() {
    let dir = temp_dir("libtest-threads");
    let src = fixture(&dir, "listing.rs", LISTING_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .args(["--rustc-arg", "--test"])
        .env("RUST_TEST_THREADS", "foo")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("should be a positive integer"),
        "stderr={stderr}"
    );
}

// ---------------------------------------------------------------------------
// The libtest terse format (issue #532): `--quiet` /
// `--format terse` prints libtest's TerseFormatter output --
// one short result per test (`.` / `i`), a progress line every
// 88 columns and before each failure, and failures as
// `{name} --- FAILED`. The harness model used to print the
// pretty format for every format flag.
// ---------------------------------------------------------------------------

const TERSE_SRC: &str = r#"
#[test]
fn a_pass() {}
#[test]
fn b_fail() { panic!("boom"); }
#[test]
#[ignore]
fn c_ignored() {}
#[test]
fn d_pass() {}
"#;

#[test]
fn run_mode_libtest_terse_format() {
    let dir = temp_dir("libtest-terse");
    let src = fixture(&dir, "terse.rs", TERSE_SRC);
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "--test", "--guest-arg", "--quiet"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr} stdout={stdout}");
    // The pass prints ".", the failure first completes the
    // progress line (" 1/4") and then stands on its own line,
    // the ignored test prints "i", the last pass ".".
    assert!(
        stdout.starts_with("\nrunning 4 tests\n. 1/4\nb_fail --- FAILED\ni."),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("test result: FAILED. 2 passed; 1 failed; 1 ignored"),
        "stdout={stdout}"
    );
    // The pretty per-test lines must not appear.
    assert!(!stdout.contains("... ok"), "stdout={stdout}");
    assert!(!stdout.contains("... FAILED"), "stdout={stdout}");
    assert!(!stdout.contains("... ignored"), "stdout={stdout}");
}

const PANIC_ABORT_CRATE_SRC: &str = r#"
#![feature(test)]

extern crate test;

use std::io::Write;
use test::Bencher;

#[test]
fn it_works() {
    assert_eq!(1 + 1, 2);
}

#[test]
#[should_panic]
fn it_panics() {
    assert_eq!(1 + 1, 4);
}

#[test]
fn it_fails() {
    println!("hello, world");
    writeln!(std::io::stderr(), "testing321").unwrap();
    assert_eq!(1 + 1, 5);
}

#[test]
fn it_exits() {
    std::process::exit(123);
}

#[bench]
fn benchmark(b: &mut Bencher) {
    b.iter(|| assert_eq!(1 + 1, 2));
}
"#;

/// A `panic=abort` test binary enters through
/// `test_main_env_args_abort`: each test is accounted for as
/// the harness's spawned subprocess — a panic the descriptor
/// does not expect aborts the child (a plain failure whose
/// section is the child's stdout, the stderr delimiter, and
/// the child's stderr with the hook's report, the test
/// running on the child's main thread), an expected panic
/// exits the child with libtest's success code, a test
/// body's own `process::exit` is read from its code, and a
/// bench descriptor runs as a test with a single-iteration
/// bencher. The run as a whole reports the failures and
/// exits 101, like libtest.
#[test]
fn run_mode_panic_abort_harness_accounts_for_spawned_tests() {
    let dir = temp_dir("panic-abort-harness");
    let src = fixture(&dir, "panic_abort.rs", PANIC_ABORT_CRATE_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("--test")
        .arg("--rustc-arg")
        .arg("-Cpanic=abort")
        .arg("--rustc-arg")
        .arg("-Zpanic_abort_tests")
        .arg("--guest-arg")
        .arg("--test-threads=1")
        .env("RUSTC_BOOTSTRAP", "1")
        .env("RUST_BACKTRACE", "0")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(101),
        "a run with failures exits 101; stderr={stderr}"
    );
    assert_eq!(stderr, "", "the harness reports on stdout alone");
    // The line of `it_fails`' assertion in the fixture.
    let assert_line = PANIC_ABORT_CRATE_SRC
        .lines()
        .position(|l| l.contains("assert_eq!(1 + 1, 5)"))
        .expect("the fixture carries the assertion")
        + 1;
    let expected = format!(
        "\nrunning 5 tests\n\
         test benchmark ... ok\n\
         test it_exits ... FAILED\n\
         test it_fails ... FAILED\n\
         test it_panics - should panic ... ok\n\
         test it_works ... ok\n\
         \nfailures:\n\
         \n---- it_exits stdout ----\n\
         ---- it_exits stderr ----\n\
         note: got unexpected return code 123\n\
         ---- it_fails stdout ----\n\
         hello, world\n\
         ---- it_fails stderr ----\n\
         testing321\n\
         \nthread 'main' (1) panicked at {}:{}:5:\n\
         assertion `left == right` failed\n\
         \u{20} left: 2\n\
         \u{20}right: 5\n\
         note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n\
         \n\nfailures:\n\
         \u{20}\u{20}\u{20}\u{20}it_exits\n\
         \u{20}\u{20}\u{20}\u{20}it_fails\n\
         \ntest result: FAILED. 3 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n\n",
        src.display(),
        assert_line
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), expected);
}

const PANIC_ABORT_NOCAPTURE_SRC: &str = r#"
#[test]
fn it_works() {
    println!("about to succeed");
}

#[test]
#[should_panic]
fn it_panics() {
    println!("about to panic");
    assert_eq!(1 + 1, 4);
}

#[test]
fn it_fails() {
    println!("about to fail");
    assert_eq!(1 + 1, 4);
}
"#;

/// The same harness under `--nocapture`: the children's
/// stdio is inherited, so their output flows live, and each
/// child's panic report — backtrace note included, every
/// child being a fresh process — goes to the run's stderr
/// as the panic happens, a passing should-panic test's
/// report included. The failure section captures nothing.
#[test]
fn run_mode_panic_abort_harness_nocapture_streams_child_reports() {
    let dir = temp_dir("panic-abort-nocapture");
    let src = fixture(&dir, "panic_abort_nc.rs", PANIC_ABORT_NOCAPTURE_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("--test")
        .arg("--rustc-arg")
        .arg("-Cpanic=abort")
        .arg("--rustc-arg")
        .arg("-Zpanic_abort_tests")
        .arg("--guest-arg")
        .arg("--test-threads=1")
        .arg("--guest-arg")
        .arg("--nocapture")
        .env("RUSTC_BOOTSTRAP", "1")
        .env("RUST_BACKTRACE", "0")
        .output()
        .expect("failed to run sparagmos-driver");
    assert_eq!(out.status.code(), Some(101), "the run has a failure");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        stdout,
        "\nrunning 3 tests\n\
         test it_fails ... about to fail\nFAILED\n\
         test it_panics - should panic ... about to panic\nok\n\
         test it_works ... about to succeed\nok\n\
         \nfailures:\n\
         \n---- it_fails stdout ----\n\
         ---- it_fails stderr ----\n\
         \n\nfailures:\n\
         \u{20}\u{20}\u{20}\u{20}it_fails\n\
         \ntest result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n\n",
        "nocapture stdout must match native"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    let reports = stderr.matches("panicked at").count();
    assert_eq!(
        reports, 2,
        "both panicking children print their report; stderr={stderr}"
    );
    let notes = stderr
        .matches("note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace")
        .count();
    assert_eq!(
        notes, 2,
        "each child is a fresh process and spends its own backtrace note; stderr={stderr}"
    );
    assert!(
        stderr.contains("thread 'main' (1) panicked at"),
        "the children report on their main thread; stderr={stderr}"
    );
}

/// The harness's spawned subprocess re-enters the same entry
/// with `__RUST_TEST_INVOKE` naming its test: the named test
/// runs in that process and the process leaves by the
/// child's verdict — libtest's success exit code (50) for a
/// pass, an abort for a panic. Modelled by running the
/// driver itself with the variable set.
#[test]
fn run_mode_panic_abort_secondary_process_runs_one_named_test() {
    let dir = temp_dir("panic-abort-secondary");
    let src = fixture(&dir, "panic_abort.rs", PANIC_ABORT_CRATE_SRC);
    let run_secondary = |name: &str| {
        Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
            .arg("run")
            .arg(&src)
            .arg("--rustc-arg")
            .arg("--test")
            .arg("--rustc-arg")
            .arg("-Cpanic=abort")
            .arg("--rustc-arg")
            .arg("-Zpanic_abort_tests")
            .env("RUSTC_BOOTSTRAP", "1")
            .env("RUST_BACKTRACE", "0")
            .env("__RUST_TEST_INVOKE", name)
            .output()
            .expect("failed to run sparagmos-driver")
    };
    // A passing test: the process exits with TR_OK (50) and
    // prints nothing.
    let out = run_secondary("it_works");
    assert_eq!(
        out.status.code(),
        Some(50),
        "a passing secondary exits with libtest's TR_OK; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "");
    // An expected panic is also a pass.
    let out = run_secondary("it_panics");
    assert_eq!(out.status.code(), Some(50), "an expected panic exits TR_OK");
    // A panicking test: its report reaches the process's
    // stderr and the process aborts (the driver mirrors a
    // guest abort as exit 134, 128 + SIGABRT).
    let out = run_secondary("it_fails");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "a failing secondary aborts; stderr={stderr}"
    );
    assert!(
        stderr.contains("thread 'main' (1) panicked at"),
        "the child's hook prints the report before the abort; stderr={stderr}"
    );
    assert!(
        stderr.contains("assertion `left == right` failed"),
        "the report carries the panic message; stderr={stderr}"
    );
}
