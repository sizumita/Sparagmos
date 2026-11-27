//! Run-mode tests for the `tests/ui/runtime/on-broken-pipe`
//! records (Issue #532): the guest's own `SIGPIPE`
//! disposition under each `-Zon-broken-pipe` mode, read back
//! through the records' own `sigpipe_utils` helper, and the
//! disposition a spawned child starts with.
//!
//! The `error` / `kill` / `not-used` shapes were already
//! served by the signal model (Issue #513) through an
//! auxiliary crate, and are pinned here in the records'
//! exact auxiliary shape. The `inherit` record was not
//! served: its main source declares `extern crate libc;`
//! itself (to install `SIG_IGN` through `libc::signal`),
//! and the main compilation got no sysroot extern pin, so
//! its front end failed with E0464 (multiple candidates for
//! `libc`) in a sysroot carrying the rustc-dev component —
//! the auxiliary builds already received the same pin.
//! The `child-processes` shapes pin the spawn-time rule for
//! all four modes against a native helper that reads its
//! own disposition back through `sigaction`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits
/// (the registry shape of the driver_signals tests, #537).
static SCRATCH_DIRS: std::sync::Mutex<Vec<std::path::PathBuf>> =
    std::sync::Mutex::new(Vec::new());

fn register_scratch_dir(dir: &std::path::Path) {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| unsafe {
        let _ = libc::atexit(remove_scratch_dirs);
    });
    if let Ok(mut dirs) = SCRATCH_DIRS.lock() {
        if !dirs.iter().any(|d| d == dir) {
            dirs.push(dir.to_path_buf());
        }
    }
}

extern "C" fn remove_scratch_dirs() {
    if let Ok(dirs) = SCRATCH_DIRS.try_lock() {
        for dir in dirs.iter() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn fixture(dir: &Path, name: &str, src: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    path
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-obp-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn run_driver_args(src: &Path, aux: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--aux")
        .arg(format!("sigpipe_utils={}", aux.display()))
        .args(extra)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn assert_run_ok(out: &Output, tag: &str, expected_stdout: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "case {tag} must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "case {tag} stdout mismatch; stderr={stderr}"
    );
}

/// The records' own auxiliary crate, verbatim in shape:
/// a zeroed `sigaction` structure, a query with a null
/// action, the handler field read back.
const SIGPIPE_UTILS_AUX: &str = r#"
#![feature(rustc_private)]
extern crate libc;

pub enum SignalHandler {
    Ignore,
    Default,
}

pub fn assert_sigpipe_handler(expected_handler: SignalHandler) {
    {
        let actual = unsafe {
            let mut actual: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGPIPE, std::ptr::null(), &mut actual);
            actual.sa_sigaction
        };
        let expected = match expected_handler {
            SignalHandler::Ignore => libc::SIG_IGN,
            SignalHandler::Default => libc::SIG_DFL,
        };
        assert_eq!(actual, expected, "actual and expected SIGPIPE disposition differs");
    }
}
"#;

fn mode_main(expected: &str, done: &str) -> String {
    format!(
        r#"
fn main() {{
    extern crate sigpipe_utils;

    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::{expected});
    println!("{done}");
}}
"#
    )
}

/// `error.rs`, `kill.rs` and `not-used.rs`: the guest's own
/// disposition under `-Zon-broken-pipe=error` (ignored),
/// `-Zon-broken-pipe=kill` (default) and no flag (ignored).
#[test]
fn run_mode_on_broken_pipe_error_kill_and_not_used() {
    let dir = temp_dir("modes");
    let aux = fixture(&dir, "sigpipe-utils.rs", SIGPIPE_UTILS_AUX);
    let cases: [(&str, &str, &[&str]); 3] = [
        ("error", "Ignore", &["--rustc-arg", "-Zon-broken-pipe=error"]),
        ("kill", "Default", &["--rustc-arg", "-Zon-broken-pipe=kill"]),
        ("not-used", "Ignore", &[]),
    ];
    for (name, expected, extra) in cases {
        let src = fixture(
            &dir,
            &format!("{name}.rs"),
            &mode_main(expected, &format!("{name} ok")),
        );
        let out = run_driver_args(&src, &aux, extra);
        assert_run_ok(&out, name, &format!("{name} ok\n"));
    }
}

/// A native helper that reports whether its own `SIGPIPE`
/// disposition is the one named by its first argument
/// (`SIG_IGN` or `SIG_DFL`), through a `sigaction` query
/// declared directly against the system C library (the
/// shape of the records' `assert-sigpipe-disposition`
/// auxiliary binary). It is built with
/// `-Zon-broken-pipe=inherit`, as that binary's own
/// revisions are, so its runtime leaves the disposition
/// it was exec'd with untouched.
const CHECK_HELPER_SRC: &str = r#"
#[repr(C)]
struct SigAction {
    handler: usize,
    mask: [u64; 16],
    flags: i32,
    restorer: usize,
}

extern "C" {
    fn sigaction(signum: i32, act: *const SigAction, oldact: *mut SigAction) -> i32;
}

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_default();
    let expected: usize = match arg.as_str() {
        "SIG_IGN" => 1,
        "SIG_DFL" => 0,
        other => panic!("bad arg {other}"),
    };
    let mut actual = SigAction { handler: 999, mask: [0; 16], flags: 0, restorer: 0 };
    let rc = unsafe { sigaction(13, std::ptr::null(), &mut actual) };
    assert_eq!(rc, 0);
    assert_eq!(
        actual.handler, expected,
        "SIGPIPE disposition differs: got {} want {}",
        actual.handler, expected
    );
    println!("child disposition ok {}", arg);
}
"#;

fn build_check_helper(dir: &Path) -> PathBuf {
    let bin_dir = dir.join("auxiliary").join("bin");
    std::fs::create_dir_all(&bin_dir).unwrap();
    let src = fixture(dir, "check.rs", CHECK_HELPER_SRC);
    let out_path = bin_dir.join("check");
    let build = Command::new("rustup")
        .args(["run", "1.100.0", "rustc"])
        .arg(&src)
        .arg("--edition")
        .arg("2021")
        .arg("-Zon-broken-pipe=inherit")
        .arg("-o")
        .arg(&out_path)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to build the disposition helper");
    assert!(
        build.status.success(),
        "the disposition helper must compile: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    out_path
}

/// `inherit.rs`: the guest is compiled with
/// `-Zon-broken-pipe=kill`, spawns a child that must start
/// with `SIG_DFL`, then installs `SIG_IGN` for `SIGPIPE`
/// through `libc::signal` — declared by the main source
/// itself, the shape whose front end the sysroot extern
/// pin fixes — and spawns a child that must now start
/// with `SIG_IGN`.
#[test]
fn run_mode_on_broken_pipe_inherit_child_disposition() {
    let dir = temp_dir("inherit");
    let aux = fixture(&dir, "sigpipe-utils.rs", SIGPIPE_UTILS_AUX);
    let helper = build_check_helper(&dir);
    let main_src = format!(
        r#"
#![feature(rustc_private)]
extern crate libc;

fn main() {{
    assert!(std::process::Command::new("{helper}")
        .arg("SIG_DFL")
        .status()
        .unwrap()
        .success());
    unsafe {{
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    }}
    assert!(std::process::Command::new("{helper}")
        .arg("SIG_IGN")
        .status()
        .unwrap()
        .success());
    println!("inherit ok");
}}
"#,
        helper = helper.display()
    );
    let src = fixture(&dir, "inherit.rs", &main_src);
    let out = run_driver_args(
        &src,
        &aux,
        &["--guest-process-host", "--rustc-arg", "-Zon-broken-pipe=kill"],
    );
    assert_run_ok(
        &out,
        "inherit",
        "child disposition ok SIG_DFL\nchild disposition ok SIG_IGN\ninherit ok\n",
    );
}

/// `child-processes.rs`, all four revisions: the guest
/// checks its own disposition through `sigpipe_utils`,
/// then spawns the helper, which must observe the
/// disposition the spawn-time rule names for the mode —
/// a reset to `SIG_DFL` under the default setup, the
/// guest's own disposition under an explicitly named mode.
#[test]
fn run_mode_on_broken_pipe_child_processes_modes() {
    let dir = temp_dir("child-processes");
    let aux = fixture(&dir, "sigpipe-utils.rs", SIGPIPE_UTILS_AUX);
    let helper = build_check_helper(&dir);
    // (mode, guest expectation, child expectation, flags)
    let cases: [(&str, &str, &str, &[&str]); 4] = [
        ("default", "Ignore", "SIG_DFL", &[]),
        (
            "kill",
            "Default",
            "SIG_DFL",
            &["--rustc-arg", "-Zon-broken-pipe=kill"],
        ),
        (
            "error",
            "Ignore",
            "SIG_IGN",
            &["--rustc-arg", "-Zon-broken-pipe=error"],
        ),
        (
            "inherit",
            "Default",
            "SIG_DFL",
            &["--rustc-arg", "-Zon-broken-pipe=inherit"],
        ),
    ];
    for (mode, we_expect, child_expects, flags) in cases {
        let main_src = format!(
            r#"
extern crate sigpipe_utils;

use sigpipe_utils::*;

fn main() {{
    assert_sigpipe_handler(SignalHandler::{we_expect});
    assert!(std::process::Command::new("{helper}")
        .arg("{child_expects}")
        .status()
        .unwrap()
        .success());
    println!("child-processes {mode} ok");
}}
"#,
            helper = helper.display()
        );
        let src = fixture(&dir, &format!("child-{mode}.rs"), &main_src);
        let mut extra: Vec<&str> = vec!["--guest-process-host"];
        extra.extend_from_slice(flags);
        let out = run_driver_args(&src, &aux, &extra);
        assert_run_ok(
            &out,
            &format!("child-processes {mode}"),
            &format!(
                "child disposition ok {child_expects}\nchild-processes {mode} ok\n"
            ),
        );
    }
}
