//! Pre-main constructors: the `#[used]` function-pointer
//! statics a crate graph places in an initialiser section
//! (`.init_array` and its platform spellings) run before the
//! entry point in run mode, in section order (Issue #513:
//! `tests/ui/runtime/stdout-before-main.rs` and
//! `tests/ui/attributes/used_with_archive.rs`).

use super::*;

const STDOUT_BEFORE_MAIN_SRC: &str = r#"
use std::os::raw::c_int;

#[used]
#[link_section = ".init_array"]
static INIT: extern "C" fn(c_int, *const *const u8, *const *const u8) = {
    extern "C" fn init(_: c_int, _: *const *const u8, _: *const *const u8) {
        print!("Hello from before ");
    }
    init
};

fn main() {
    println!("main!");
}
"#;

#[test]
fn run_mode_init_array_ctor_runs_before_main() {
    // The record's shape: a three-argument (argc/argv/envp)
    // constructor in the main crate's `.init_array` prints
    // the first half of the expected line before `main`.
    let dir = temp_dir("premain-stdout");
    let src = fixture(&dir, "premain.rs", STDOUT_BEFORE_MAIN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the constructor program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Hello from before main!\n",
        "the constructor's output precedes the entry point's"
    );
}

const USED_WITH_ARCHIVE_MAIN_SRC: &str = r#"
extern crate used_pre_main_constructor;

fn main() {
    println!("main");
}
"#;

#[test]
fn run_mode_used_aux_ctor_runs_before_main() {
    // The record's shape: the main crate names no item of
    // the auxiliary crate; the crate's `#[used]` constructor
    // static, its section spelled per platform by `cfg_attr`
    // as in the record, still runs before `main`.
    let dir = temp_dir("premain-archive");
    let aux = fixture(
        &dir,
        "used_pre_main_constructor.rs",
        USED_PRE_MAIN_AUX_WITH_SECTION_SRC,
    );
    let src = fixture(&dir, "used_with_archive.rs", USED_WITH_ARCHIVE_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("used_pre_main_constructor", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the archive constructor program must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "constructor\nmain\n",
        "the unused auxiliary crate's constructor precedes `main`"
    );
}

const USED_PRE_MAIN_AUX_WITH_SECTION_SRC: &str = r#"
#![crate_type = "rlib"]

#[used]
#[cfg_attr(any(target_os = "linux", target_os = "android"), link_section = ".init_array")]
#[cfg_attr(any(target_os = "macos", target_os = "ios"), link_section = "__DATA,__mod_init_func,mod_init_funcs")]
#[cfg_attr(windows, link_section = ".CRT$XCU")]
static CONSTRUCTOR: extern "C" fn() = constructor;

#[cfg_attr(any(target_os = "linux", target_os = "android"), link_section = ".text.startup")]
extern "C" fn constructor() {
    print!("constructor\n");
}
"#;

const INIT_PRIORITY_SRC: &str = r#"
#[used]
#[link_section = ".init_array.01000"]
static A: extern "C" fn() = a;
extern "C" fn a() {
    print!("A");
}
#[used]
#[link_section = ".init_array"]
static B: extern "C" fn() = b;
extern "C" fn b() {
    print!("B");
}
#[used]
#[link_section = ".init_array.00050"]
static C: extern "C" fn() = c;
extern "C" fn c() {
    print!("C");
}
fn main() {
    println!("main");
}
"#;

#[test]
fn run_mode_init_array_priority_order() {
    // Priority sections run in ascending priority order
    // ahead of the plain section: C (.00050), A (.01000),
    // B (plain), then `main`.
    let dir = temp_dir("premain-order");
    let src = fixture(&dir, "order.rs", INIT_PRIORITY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the ordered constructors must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "CABmain\n",
        "constructors run in section priority order before `main`"
    );
}

const INIT_ARGC_SRC: &str = r#"
use std::os::raw::{c_char, c_int};

#[used]
#[link_section = ".init_array"]
static INIT: extern "C" fn(c_int, *const *const c_char, *const *const c_char) = init;

extern "C" fn init(argc: c_int, argv: *const *const c_char, envp: *const *const c_char) {
    println!(
        "argc={} argv_null={} envp_null={}",
        argc,
        argv.is_null(),
        envp.is_null()
    );
}

fn main() {
    println!("main");
}
"#;

#[test]
fn run_mode_ctor_receives_platform_arguments() {
    // The invocation convention is the platform's triple:
    // with no guest arguments the constructor observes
    // argc 1 (the program path) and non-null argv/envp.
    let dir = temp_dir("premain-argc");
    let src = fixture(&dir, "argc.rs", INIT_ARGC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the argument-receiving constructor must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "argc=1 argv_null=false envp_null=false\nmain\n",
        "the constructor receives the (argc, argv, envp) triple"
    );
}

const INIT_PANIC_SRC: &str = r#"
#[used]
#[link_section = ".init_array"]
static INIT: extern "C" fn() = init;

extern "C" fn init() {
    panic!("ctor boom");
}

fn main() {
    println!("main");
}
"#;

#[test]
fn run_mode_panicking_ctor_aborts() {
    // A panic may not unwind out of an `extern "C"`
    // constructor: natively the process aborts (134) after
    // printing the panic report, and `main` never runs.
    let dir = temp_dir("premain-panic");
    let src = fixture(&dir, "panic.rs", INIT_PANIC_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "a panicking constructor aborts the run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "",
        "the entry point never runs after a constructor abort"
    );
    assert!(
        stderr.contains("ctor boom"),
        "the panic report names the constructor's panic: {stderr}"
    );
}
