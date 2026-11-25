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
