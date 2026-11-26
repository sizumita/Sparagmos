//! Run-mode tests for the signal model (Issue #513):
//! `libc::sigaction` registration and query, synchronous
//! `libc::raise` delivery to the installed handler, and
//! `libc::atexit` handlers running at program exit.
//!
//! The guest programs use libc through an auxiliary crate
//! (the driver's own sysroot pinning serves the aux build,
//! as in the driver_run FFI tests); the model code paths
//! exercised are the ones the `tests/ui/runtime/
//! signal-alternate-stack-cleanup.rs` record takes, whose
//! main body the first test mirrors. On the base tree every
//! program here stops at the sigaction call (no converted
//! body, no model).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(dir: &Path, name: &str, src: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    path
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-sig-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run_driver(src: &Path, aux: &Path) -> Output {
    run_driver_args(src, aux, &[])
}

fn run_driver_args(src: &Path, aux: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--aux")
        .arg(format!("sigaux={}", aux.display()))
        .args(extra)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn run_ok_args(
    tag: &str,
    aux_src: &str,
    main_src: &str,
    extra: &[&str],
    expected_stdout: &str,
) {
    let dir = temp_dir(tag);
    let aux = fixture(&dir, "sigaux.rs", aux_src);
    let src = fixture(&dir, "case.rs", main_src);
    let out = run_driver_args(&src, &aux, extra);
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

fn run_ok(tag: &str, aux_src: &str, main_src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let aux = fixture(&dir, "sigaux.rs", aux_src);
    let src = fixture(&dir, "case.rs", main_src);
    let out = run_driver(&src, &aux);
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

/// The target record's shape: a SA_SIGINFO handler for
/// SIGWINCH installed with sigaction, an atexit callback
/// that raises the signal, and the handler observing the
/// delivery at program exit.
const SIGINFO_AUX: &str = r#"
#![feature(rustc_private)]
extern crate libc;

use libc::*;

unsafe extern "C" fn signal_handler(signum: c_int, _: *mut siginfo_t, _: *mut c_void) {
    assert_eq!(signum, SIGWINCH);
    println!("handled {signum}");
}

extern "C" fn send_signal() {
    unsafe {
        raise(SIGWINCH);
    }
}

pub fn install_and_register() {
    unsafe {
        let mut action: sigaction = std::mem::zeroed();
        action.sa_flags = (SA_ONSTACK | SA_SIGINFO) as _;
        action.sa_sigaction = signal_handler as sighandler_t;
        sigaction(SIGWINCH, &action, std::ptr::null_mut());
        atexit(send_signal);
    }
}
"#;

const SIGINFO_MAIN: &str = r#"
extern crate sigaux;

fn main() {
    sigaux::install_and_register();
    println!("main done");
}
"#;

/// A one-argument handler (no SA_SIGINFO), queried back
/// through oldact, replaced, and delivered by a direct
/// raise; a raise of a signal whose default disposition
/// is ignore, before anything is installed.
const PLAIN_AUX: &str = r#"
#![feature(rustc_private)]
extern crate libc;

use libc::*;

unsafe extern "C" fn handler_a(signum: c_int) {
    println!("A {signum}");
}

unsafe extern "C" fn handler_b(signum: c_int) {
    println!("B {signum}");
}

pub fn run() {
    unsafe {
        // Default disposition of SIGWINCH is ignore.
        assert_eq!(raise(SIGWINCH), 0);
        println!("default ok");

        let mut action: sigaction = std::mem::zeroed();
        action.sa_sigaction = handler_a as sighandler_t;
        assert_eq!(sigaction(SIGUSR1, &action, std::ptr::null_mut()), 0);

        // Query the installed action back.
        let mut old: sigaction = std::mem::zeroed();
        assert_eq!(sigaction(SIGUSR1, std::ptr::null(), &mut old), 0);
        assert_eq!(old.sa_sigaction, handler_a as sighandler_t);
        assert_eq!(old.sa_flags, 0);

        // Replace it; the raise now reaches handler B.
        let mut next: sigaction = std::mem::zeroed();
        next.sa_sigaction = handler_b as sighandler_t;
        assert_eq!(sigaction(SIGUSR1, &next, std::ptr::null_mut()), 0);
        assert_eq!(raise(SIGUSR1), 0);
        println!("raised ok");
    }
}
"#;

const PLAIN_MAIN: &str = r#"
extern crate sigaux;

fn main() {
    sigaux::run();
}
"#;

/// SIG_IGN suppresses delivery of a signal whose default
/// disposition is termination, and the uncatchable
/// signals refuse installation.
const IGN_AUX: &str = r#"
#![feature(rustc_private)]
extern crate libc;

use libc::*;

pub fn run() {
    unsafe {
        let mut action: sigaction = std::mem::zeroed();
        action.sa_sigaction = SIG_IGN as sighandler_t;
        assert_eq!(sigaction(SIGTERM, &action, std::ptr::null_mut()), 0);
        assert_eq!(raise(SIGTERM), 0);
        println!("survived");

        assert_eq!(sigaction(SIGKILL, &action, std::ptr::null_mut()), -1);
        assert_eq!(sigaction(0, &action, std::ptr::null_mut()), -1);
        println!("refusals ok");
    }
}
"#;

const IGN_MAIN: &str = r#"
extern crate sigaux;

fn main() {
    sigaux::run();
}
"#;

/// Exit handlers run at program exit in reverse
/// registration order.
const ORDER_AUX: &str = r#"
#![feature(rustc_private)]
extern crate libc;

use libc::*;

extern "C" fn first() {
    println!("first registered");
}

extern "C" fn second() {
    println!("second registered");
}

pub fn register() {
    unsafe {
        atexit(first);
        atexit(second);
    }
}
"#;

const ORDER_MAIN: &str = r#"
extern crate sigaux;

fn main() {
    sigaux::register();
    println!("body");
}
"#;

#[test]
fn run_mode_sigaction_siginfo_handler_via_atexit_raise() {
    run_ok("siginfo", SIGINFO_AUX, SIGINFO_MAIN, "main done\nhandled 28\n");
}

#[test]
fn run_mode_sigaction_plain_handler_query_and_raise() {
    run_ok(
        "plain",
        PLAIN_AUX,
        PLAIN_MAIN,
        "default ok\nB 10\nraised ok\n",
    );
}

#[test]
fn run_mode_sigaction_sig_ign_and_refusals() {
    run_ok("ign", IGN_AUX, IGN_MAIN, "survived\nrefusals ok\n");
}

/// A query (null `act`) before the guest itself has
/// installed anything — the shape of the
/// `tests/ui/runtime/on-broken-pipe` records, whose helper
/// reads the SIGPIPE disposition back at startup. For
/// SIGPIPE the answer is the disposition the runtime
/// initialisation installed (`SIG_IGN` under the default
/// setup): an all-zero structure apart from the handler,
/// written over whatever the answer buffer already held.
/// A signal the runtime does not set up (SIGTERM here)
/// answers with the default disposition instead.
const QUERY_AUX: &str = r#"
#![feature(rustc_private)]
extern crate libc;

use libc::*;

pub fn run() {
    unsafe {
        // The on-broken-pipe helper's shape: a zeroed
        // buffer, a null action, the handler read back.
        // The runtime initialisation has installed
        // SIG_IGN for SIGPIPE before main.
        let mut actual: sigaction = std::mem::zeroed();
        assert_eq!(sigaction(SIGPIPE, std::ptr::null(), &mut actual), 0);
        assert_eq!(actual.sa_sigaction, SIG_IGN);
        assert_eq!(actual.sa_flags, 0);
        println!("init query ok");

        // Non-zero field values in the answer buffer are
        // overwritten, not left behind.
        let mut seeded: sigaction = std::mem::zeroed();
        seeded.sa_sigaction = 0xABAB as sighandler_t;
        seeded.sa_flags = 0x1234;
        assert_eq!(sigaction(SIGPIPE, std::ptr::null(), &mut seeded), 0);
        assert_eq!(seeded.sa_sigaction, SIG_IGN);
        assert_eq!(seeded.sa_flags, 0);
        println!("seeded query ok");

        // A signal the runtime does not set up answers
        // with the default disposition.
        let mut other: sigaction = std::mem::zeroed();
        assert_eq!(sigaction(SIGTERM, std::ptr::null(), &mut other), 0);
        assert_eq!(other.sa_sigaction, SIG_DFL);

        // Installing afterwards reports the runtime's
        // disposition as the previous one, and a query
        // then reports the installed disposition.
        let mut action: sigaction = std::mem::zeroed();
        action.sa_sigaction = SIG_DFL as sighandler_t;
        let mut prev: sigaction = std::mem::zeroed();
        assert_eq!(sigaction(SIGPIPE, &action, &mut prev), 0);
        assert_eq!(prev.sa_sigaction, SIG_IGN);
        let mut now: sigaction = std::mem::zeroed();
        assert_eq!(sigaction(SIGPIPE, std::ptr::null(), &mut now), 0);
        assert_eq!(now.sa_sigaction, SIG_DFL);
        println!("round trip ok");
    }
}
"#;

const QUERY_MAIN: &str = r#"
extern crate sigaux;

fn main() {
    sigaux::run();
}
"#;

#[test]
fn run_mode_sigaction_query_of_runtime_init_and_never_installed() {
    run_ok(
        "query",
        QUERY_AUX,
        QUERY_MAIN,
        "init query ok\nseeded query ok\nround trip ok\n",
    );
}

/// The startup SIGPIPE disposition under each
/// `-Zon-broken-pipe` mode the driver resolves: `kill`
/// and `inherit` leave the default handler in place,
/// `error` installs `SIG_IGN` (as the default setup
/// does, which the query test above covers).
const MODE_AUX: &str = r#"
#![feature(rustc_private)]
extern crate libc;

use libc::*;

pub fn run() {
    unsafe {
        let mut actual: sigaction = std::mem::zeroed();
        assert_eq!(sigaction(SIGPIPE, std::ptr::null(), &mut actual), 0);
        println!("handler {}", actual.sa_sigaction);
    }
}
"#;

const MODE_MAIN: &str = r#"
extern crate sigaux;

fn main() {
    sigaux::run();
}
"#;

#[test]
fn run_mode_sigpipe_kill_mode_query() {
    run_ok_args(
        "killmode",
        MODE_AUX,
        MODE_MAIN,
        &["--rustc-arg", "-Zon-broken-pipe=kill"],
        "handler 0\n",
    );
}

#[test]
fn run_mode_sigpipe_inherit_mode_query() {
    run_ok_args(
        "inheritmode",
        MODE_AUX,
        MODE_MAIN,
        &["--rustc-arg", "-Zon-broken-pipe=inherit"],
        "handler 0\n",
    );
}

#[test]
fn run_mode_sigpipe_error_mode_query() {
    run_ok_args(
        "errormode",
        MODE_AUX,
        MODE_MAIN,
        &["--rustc-arg", "-Zon-broken-pipe=error"],
        "handler 1\n",
    );
}

/// The disposition a spawned child starts with (std's
/// spawn-time rule, exercised end to end through the
/// process host): under the default setup the child is
/// reset to `SIG_DFL`, so a shell that raises SIGPIPE
/// on itself dies of it; under `-Zon-broken-pipe=error`
/// the child inherits the guest's `SIG_IGN` and
/// survives.
const CHILD_MAIN: &str = r#"
fn main() {
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg("kill -PIPE $$; echo survived")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    println!("child stdout: {}", text.trim());
    println!("child success: {}", out.status.success());
}
"#;

#[test]
fn run_mode_sigpipe_child_disposition() {
    let dir = temp_dir("child");
    let aux = fixture(&dir, "sigaux.rs", MODE_AUX);
    let src = fixture(&dir, "case.rs", CHILD_MAIN);
    let out = run_driver_args(&src, &aux, &["--guest-process-host"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "default-mode case must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "child stdout: \nchild success: false\n",
        "default-mode child must be reset to SIG_DFL; stderr={stderr}"
    );
    let out = run_driver_args(
        &src,
        &aux,
        &[
            "--guest-process-host",
            "--rustc-arg",
            "-Zon-broken-pipe=error",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "error-mode case must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "child stdout: survived\nchild success: true\n",
        "error-mode child must inherit SIG_IGN; stderr={stderr}"
    );
}

#[test]
fn run_mode_atexit_reverse_order() {
    run_ok(
        "order",
        ORDER_AUX,
        ORDER_MAIN,
        "body\nsecond registered\nfirst registered\n",
    );
}
