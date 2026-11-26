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
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--aux")
        .arg(format!("sigaux={}", aux.display()))
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
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

#[test]
fn run_mode_atexit_reverse_order() {
    run_ok(
        "order",
        ORDER_AUX,
        ORDER_MAIN,
        "body\nsecond registered\nfirst registered\n",
    );
}
