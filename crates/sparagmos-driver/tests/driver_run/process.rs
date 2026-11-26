use super::{Command, Output, Path, fixture, run_driver, run_driver_extra, temp_dir};


/// Run the driver with real child processes enabled
/// (`--guest-process-host`): the guest's `std::process`
/// children start on the host.
fn run_driver_with_process_host(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .arg("--guest-process-host")
        .output()
        .expect("failed to run sparagmos-driver")
}

const PROCESS_EXIT_SRC: &str = r#"
fn main() {
    println!("before");
    std::process::exit(7);
}
"#;

const EXIT_CODE_SRC: &str = r#"
fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(3)
}
"#;

#[test]
fn run_mode_process_exit_and_exit_code() {
    let dir = temp_dir("exit-codes");
    let src = fixture(&dir, "exit.rs", PROCESS_EXIT_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(7), "process::exit(7)");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "before\n");
    let src = fixture(&dir, "exitcode.rs", EXIT_CODE_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(3), "ExitCode::from(3)");
}

const RESULT_OK_SRC: &str = r#"
fn main() -> Result<(), String> {
    println!("hello");
    Ok(())
}
"#;

const RESULT_ERR_SRC: &str = r#"
fn main() -> Result<(), String> {
    Err("boom".to_string())
}
"#;

const RESULT_ERR_IO_SRC: &str = r#"
fn main() -> Result<(), std::io::Error> {
    Err(std::io::Error::from_raw_os_error(2))
}
"#;

#[test]
fn run_mode_termination_result_report() {
    // A `fn main() -> Result<_, _>` entry reports through
    // `Termination`: `Ok` exits 0, `Err` writes `Error: {err:?}`
    // to stderr and exits with `ExitCode::FAILURE` (1).
    let dir = temp_dir("termination-result");
    let src = fixture(&dir, "ok.rs", RESULT_OK_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(0), "Result::Ok");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hello\n");
    let src = fixture(&dir, "err.rs", RESULT_ERR_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Result::Err(String)");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"boom\"\n"
    );
    let src = fixture(&dir, "err_io.rs", RESULT_ERR_IO_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Result::Err(io::Error)");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: Os { code: 2, kind: NotFound, \
         message: \"No such file or directory\" }\n"
    );
}

const RESULT_ERR_REVERSE_I32_SRC: &str = r#"
use std::cmp::Reverse;

fn main() -> Result<(), Reverse<i32>> {
    Err(Reverse(42))
}
"#;

const RESULT_ERR_REVERSE_PATHBUF_SRC: &str = r#"
use std::cmp::Reverse;
use std::path::PathBuf;

fn main() -> Result<(), Reverse<PathBuf>> {
    Err(Reverse(PathBuf::from("hello")))
}
"#;

const RESULT_ERR_WRAPPING_STRING_SRC: &str = r#"
use std::num::Wrapping;

fn main() -> Result<(), Wrapping<String>> {
    Err(Wrapping(String::from("hello")))
}
"#;

const RESULT_ERR_WRAPPING_I32_SRC: &str = r#"
use std::num::Wrapping;

fn main() -> Result<(), Wrapping<i32>> {
    Err(Wrapping(7))
}
"#;

#[test]
fn run_mode_termination_reverse_wrapping_report() {
    // The one-field transparent wrappers `Reverse` / `Wrapping`
    // as a report payload: the rendering matches native, and
    // the payload's drop glue is accounted for by the held
    // value's glue (the wrappers have no `Drop` impl of their
    // own), so the run exits 1 like native instead of failing
    // the drop accounting after rendering. Paired with
    // `run_mode_termination_range_from_pathbuf_report`.
    let dir = temp_dir("termination-reverse-wrapping");
    let src = fixture(&dir, "err_reverse_i32.rs", RESULT_ERR_REVERSE_I32_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Result::Err(Reverse<i32>)");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: Reverse(42)\n"
    );
    let src = fixture(
        &dir,
        "err_reverse_pathbuf.rs",
        RESULT_ERR_REVERSE_PATHBUF_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(1),
        "Result::Err(Reverse<PathBuf>)"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: Reverse(\"hello\")\n"
    );
    let src = fixture(
        &dir,
        "err_wrapping_string.rs",
        RESULT_ERR_WRAPPING_STRING_SRC,
    );
    let out = run_driver(&src);
    assert_eq!(
        out.status.code(),
        Some(1),
        "Result::Err(Wrapping<String>)"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"hello\"\n"
    );
    let src = fixture(&dir, "err_wrapping_i32.rs", RESULT_ERR_WRAPPING_I32_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Result::Err(Wrapping<i32>)");
    assert_eq!(String::from_utf8_lossy(&out.stderr), "Error: 7\n");
    assert!(
        out.stdout.is_empty(),
        "the report writes nothing to stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

const RESULT_OK_EXIT_CODE_SRC: &str = r#"
fn main() -> Result<std::process::ExitCode, String> {
    Ok(std::process::ExitCode::from(3))
}
"#;

const RESULT_NESTED_ERR_SRC: &str = r#"
fn main() -> Result<Result<(), String>, String> {
    Ok(Err("inner".to_string()))
}
"#;

const RESULT_NESTED_OK_SRC: &str = r#"
fn main() -> Result<Result<(), String>, String> {
    println!("deep");
    Ok(Ok(()))
}
"#;

const RESULT_NESTED_EXIT_CODE_SRC: &str = r#"
fn main() -> Result<Result<std::process::ExitCode, String>, String> {
    Ok(Ok(std::process::ExitCode::from(5)))
}
"#;

const RESULT_ERR_RC_SRC: &str = r#"
fn main() -> Result<(), std::rc::Rc<String>> {
    Err(std::rc::Rc::new(String::from("boom")))
}
"#;

#[test]
fn run_mode_termination_result_delegates_and_renders_live() {
    // `Result<T, E>::report` delegates `Ok` to `T::report`: an
    // `ExitCode` payload yields its code (not an unconditional
    // 0), and a nested `Result` recurses -- its own `Err`
    // prints and fails, its own `Ok(ExitCode)` yields the code.
    // An `Err` payload that points into the run's allocations
    // (an `Rc`) renders its actual Debug text, not the value's
    // structural form: the report is rendered by the interpreter
    // that executed the entry, before its state is discarded.
    let dir = temp_dir("termination-delegate");
    let src = fixture(&dir, "ok_exit_code.rs", RESULT_OK_EXIT_CODE_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(3), "Ok(ExitCode::from(3))");
    let src = fixture(&dir, "nested_err.rs", RESULT_NESTED_ERR_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Ok(Err(..))");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"inner\"\n"
    );
    let src = fixture(&dir, "nested_ok.rs", RESULT_NESTED_OK_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(0), "Ok(Ok(()))");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "deep\n");
    let src = fixture(&dir, "nested_exit_code.rs", RESULT_NESTED_EXIT_CODE_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(5), "Ok(Ok(ExitCode::from(5)))");
    let src = fixture(&dir, "err_rc.rs", RESULT_ERR_RC_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Err(Rc<String>)");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"boom\"\n"
    );
}

const RESULT_ERR_PANICKING_DEBUG_SRC: &str = r#"
use std::fmt;
struct Boom;
impl fmt::Debug for Boom {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("fmt exploded");
    }
}
fn main() -> Result<(), Boom> {
    Err(Boom)
}
"#;

const RESULT_ERR_EXITING_DEBUG_SRC: &str = r#"
use std::fmt;
struct Exiter;
impl fmt::Debug for Exiter {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        std::process::exit(7);
    }
}
fn main() -> Result<(), Exiter> {
    Err(Exiter)
}
"#;

const RESULT_ERR_TLS_DTOR_ORDER_SRC: &str = r#"
use std::cell::RefCell;
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        eprintln!("tls dtor ran");
    }
}
thread_local! {
    static GUARD: RefCell<Option<Guard>> = RefCell::new(None);
}
fn main() -> Result<(), String> {
    GUARD.with(|g| *g.borrow_mut() = Some(Guard));
    Err("boom".to_string())
}
"#;

const RESULT_ERR_TLS_READING_DEBUG_SRC: &str = r#"
use std::cell::RefCell;
use std::fmt;
thread_local! {
    static VAL: RefCell<Option<String>> = RefCell::new(None);
}
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = VAL.with(|v| v.borrow().clone());
        write!(f, "val={s:?}")
    }
}
fn main() -> Result<(), E> {
    VAL.with(|v| *v.borrow_mut() = Some("alive".to_string()));
    Err(E)
}
"#;

#[test]
fn run_mode_termination_report_guest_outcomes_and_tls_order() {
    // The report's rendering is guest execution: a panic inside
    // the `Err` payload's `Debug` body is the run's outcome (the
    // guest-panic exit, reported like any guest panic), and a
    // `process::exit` there passes its code through -- neither
    // collapses into the report's own failure exit with a
    // structural stand-in for the payload. The report also
    // applies before the entry thread's TLS destructors run, so
    // its line precedes destructor output in stderr, and a
    // `Debug` body that reads TLS still sees it alive.
    let dir = temp_dir("termination-report-outcomes");
    let src = fixture(&dir, "err_panicking_debug.rs", RESULT_ERR_PANICKING_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "panic in Debug::fmt");
    // The report's `Error: ` prefix is written before the
    // formatting starts (the native order), so it stands even
    // though the formatting panicked: it precedes the panic's
    // own report (std's default hook output, printed when the
    // panic is raised).
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "Error: \nthread 'main' (1) panicked at {}:6:9:\nfmt exploded\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
            src.display()
        )
        .as_str()
    );
    let src = fixture(&dir, "err_exiting_debug.rs", RESULT_ERR_EXITING_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(7), "process::exit in Debug::fmt");
    // Likewise the already-written prefix is the whole stderr
    // when the formatting exits the process.
    assert_eq!(String::from_utf8_lossy(&out.stderr), "Error: ");
    let src = fixture(&dir, "err_tls_dtor_order.rs", RESULT_ERR_TLS_DTOR_ORDER_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Err with TLS destructor");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"boom\"\ntls dtor ran\n"
    );
    let src = fixture(&dir, "err_tls_reading_debug.rs", RESULT_ERR_TLS_READING_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Debug::fmt reads TLS");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: val=Some(\"alive\")\n"
    );
}


const RESULT_ERR_DROPPING_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "E") }
}
impl Drop for E {
    fn drop(&mut self) { eprintln!("error dropped"); }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_DROP_EXITS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "E") }
}
impl Drop for E {
    fn drop(&mut self) {
        eprintln!("error dropped");
        std::process::exit(7);
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_DROP_PANICS_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "E") }
}
impl Drop for E {
    fn drop(&mut self) { panic!("drop panicked"); }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_DROP_TLS_ORDER_SRC: &str = r#"
use std::cell::RefCell;
use std::fmt;
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) { eprintln!("tls dtor ran"); }
}
thread_local! {
    static GUARD: RefCell<Option<Guard>> = RefCell::new(None);
}
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "E") }
}
impl Drop for E {
    fn drop(&mut self) { eprintln!("error dropped"); }
}
fn main() -> Result<(), E> {
    GUARD.with(|g| *g.borrow_mut() = Some(Guard));
    Err(E)
}
"#;

const RESULT_ERR_SIDE_EFFECT_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        eprintln!("side effect");
        write!(f, "E")
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_SIDE_EFFECT_EXIT_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        eprintln!("side effect");
        std::process::exit(5);
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

const RESULT_ERR_SIDE_EFFECT_PANIC_DEBUG_SRC: &str = r#"
use std::fmt;
struct E;
impl fmt::Debug for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        eprintln!("side effect");
        panic!("fmt panicked");
    }
}
fn main() -> Result<(), E> { Err(E) }
"#;

#[test]
fn run_mode_termination_report_drops_payload_and_prefix_order() {
    // The native `Result::report` writes the `Error: ` prefix
    // before it formats the payload, prints the formatted
    // payload, and then consumes and drops the error -- all
    // before the thread's TLS destructors run. So: stderr the
    // payload's `Debug` body writes follows the prefix (and
    // the prefix stands when the formatting exits or panics);
    // the payload's destructor runs after the whole
    // `Error: ...` line, its own guest outcome (an exit, a
    // panic) replaces the report's failure exit, and it still
    // precedes TLS destructor output.
    let dir = temp_dir("termination-report-drop");
    let src = fixture(&dir, "err_dropping.rs", RESULT_ERR_DROPPING_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "dropping Err payload");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: E\nerror dropped\n"
    );
    let src = fixture(&dir, "err_drop_exits.rs", RESULT_ERR_DROP_EXITS_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(7), "process::exit in payload Drop");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: E\nerror dropped\n"
    );
    let src = fixture(&dir, "err_drop_panics.rs", RESULT_ERR_DROP_PANICS_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "panic in payload Drop");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "Error: E\n\nthread 'main' (1) panicked at {}:8:26:\ndrop panicked\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
            src.display()
        )
        .as_str()
    );
    let src = fixture(&dir, "err_drop_tls_order.rs", RESULT_ERR_DROP_TLS_ORDER_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "payload Drop before TLS dtors");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: E\nerror dropped\ntls dtor ran\n"
    );
    let src = fixture(&dir, "err_side_effect_debug.rs", RESULT_ERR_SIDE_EFFECT_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Debug::fmt writes to stderr");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: side effect\nE\n"
    );
    let src = fixture(&dir, "err_side_effect_exit_debug.rs", RESULT_ERR_SIDE_EFFECT_EXIT_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(5), "process::exit in Debug::fmt keeps prefix");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: side effect\n"
    );
    let src = fixture(&dir, "err_side_effect_panic_debug.rs", RESULT_ERR_SIDE_EFFECT_PANIC_DEBUG_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "panic in Debug::fmt keeps prefix");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "Error: side effect\n\nthread 'main' (1) panicked at {}:7:9:\nfmt panicked\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
            src.display()
        )
        .as_str()
    );
}

// ---- Lane II: process model (issue #144) ----
// The process model never starts a child process. Spawning a
// program that cannot be resolved answers the io::Error the OS
// would produce (shape and kind included, through the guest
// bitpacked representation and the modelled OS error table); a
// program that does resolve is a named unsupported error, never
// a silent success.

const SPAWN_NONEXISTENT_SRC: &str = r#"
use std::io::ErrorKind;
use std::process::Command;
fn main() {
    let e = Command::new("definitely-not-a-real-program-xyz").spawn().unwrap_err();
    println!("kind={:?}", e.kind());
    println!("raw={:?}", e.raw_os_error());
    assert_eq!(e.kind(), ErrorKind::NotFound);
    println!("spawn-nonexistent-ok");
}
"#;

#[test]
fn run_mode_process_spawn_nonexistent_error_kind() {
    let dir = temp_dir("proc-spawn-nonexistent");
    let src = fixture(&dir, "spawn_nonexistent.rs", SPAWN_NONEXISTENT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "kind=NotFound\nraw=Some(2)\nspawn-nonexistent-ok\n"
    );
}

const SPAWN_ERROR_SHAPES_SRC: &str = r#"
use std::io::ErrorKind;
use std::process::Command;
fn main() {
    let dir = Command::new("/tmp").spawn().unwrap_err().kind();
    assert_eq!(dir, ErrorKind::PermissionDenied);
    let nul = Command::new("whatever").arg("a\0b").spawn().unwrap_err().kind();
    assert_eq!(nul, ErrorKind::InvalidInput);
    let cwd = Command::new("whatever")
        .current_dir("/no/such/dir-xyz")
        .spawn()
        .unwrap_err()
        .kind();
    assert_eq!(cwd, ErrorKind::NotFound);
    println!("error-shapes-ok");
}
"#;

#[test]
fn run_mode_process_spawn_error_shapes() {
    let dir = temp_dir("proc-spawn-shapes");
    let src = fixture(&dir, "spawn_shapes.rs", SPAWN_ERROR_SHAPES_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "error-shapes-ok\n");
}

const CURRENT_EXE_SRC: &str = r#"
fn main() {
    let p = std::env::current_exe().unwrap();
    let q = std::env::current_exe().unwrap();
    assert!(p == q);
    assert!(std::process::id() > 0);
    println!("current-exe-ok");
}
"#;

#[test]
fn run_mode_process_current_exe_and_id() {
    let dir = temp_dir("proc-current-exe");
    let src = fixture(&dir, "current_exe.rs", CURRENT_EXE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "current-exe-ok\n");
}

const SPAWN_RESOLVABLE_SRC: &str = r#"
fn main() {
    let _ = std::process::Command::new("ls").spawn();
    println!("must not reach here");
}
"#;

#[test]
fn run_mode_process_spawn_resolvable_is_named_error() {
    let dir = temp_dir("proc-spawn-resolvable");
    let src = fixture(&dir, "spawn_resolvable.rs", SPAWN_RESOLVABLE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("does not start child processes"),
        "stderr={stderr}"
    );
}

// ---- Process host (issue #515) ----
// With `--guest-process-host`, a run's children really start
// on the host: `output` / `status` report the child's real
// exit status and captured streams, `spawn` hands the guest
// a live `Child`, and a child naming the guest's own
// executable re-runs the guest program under the driver.

const PROCESS_HOST_OUTPUT_SRC: &str = r#"
use std::process::Command;
fn main() {
    let out = Command::new("echo").arg("hello-child").output().unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "hello-child\n");
    let out = Command::new("sh")
        .arg("-c")
        .arg("echo oops >&2; exit 3")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3));
    assert_eq!(String::from_utf8(out.stderr).unwrap(), "oops\n");
    let status = Command::new("false").status().unwrap();
    assert!(!status.success());
    assert_eq!(status.code(), Some(1));
    println!("process-host-output-ok");
}
"#;

#[test]
fn run_mode_process_host_output_and_status() {
    let dir = temp_dir("proc-host-output");
    let src = fixture(&dir, "process_host_output.rs", PROCESS_HOST_OUTPUT_SRC);
    let out = run_driver_with_process_host(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "process-host-output-ok\n"
    );
}

const PROCESS_HOST_ENV_CWD_SRC: &str = r#"
use std::process::Command;
fn main() {
    let out = Command::new("sh")
        .arg("-c")
        .arg("echo $MARKER; pwd")
        .env("MARKER", "zap")
        .current_dir("/tmp")
        .output()
        .unwrap();
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("zap\n"), "env missing: {text}");
    assert!(text.contains("/tmp\n"), "cwd missing: {text}");
    println!("process-host-env-cwd-ok");
}
"#;

#[test]
fn run_mode_process_host_env_and_cwd() {
    let dir = temp_dir("proc-host-env-cwd");
    let src = fixture(&dir, "process_host_env_cwd.rs", PROCESS_HOST_ENV_CWD_SRC);
    let out = run_driver_with_process_host(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "process-host-env-cwd-ok\n"
    );
}

const PROCESS_HOST_SPAWN_SRC: &str = r#"
use std::process::{Command, Stdio};
fn main() {
    // spawn + wait_with_output over a piped stream.
    let child = Command::new("echo")
        .arg("piped-bytes")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    assert!(child.id() > 0);
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "piped-bytes\n");
    // try_wait observes a running child, then wait reaps it.
    let mut child = Command::new("sleep").arg("0.3").spawn().unwrap();
    assert!(child.try_wait().unwrap().is_none());
    let status = child.wait().unwrap();
    assert!(status.success());
    let again = child.try_wait().unwrap().unwrap();
    assert!(again.success());
    // kill ends a child; its status reports no exit code.
    let mut victim = Command::new("sleep").arg("100").spawn().unwrap();
    victim.kill().unwrap();
    let status = victim.wait().unwrap();
    assert!(!status.success());
    assert_eq!(status.code(), None);
    println!("process-host-spawn-ok");
}
"#;

#[test]
fn run_mode_process_host_spawn_wait_kill() {
    let dir = temp_dir("proc-host-spawn");
    let src = fixture(&dir, "process_host_spawn.rs", PROCESS_HOST_SPAWN_SRC);
    let out = run_driver_with_process_host(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "process-host-spawn-ok\n"
    );
}

const PROCESS_HOST_SELF_SRC: &str = r#"
use std::env;
use std::process::Command;
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "greet" => {
                println!("child-stdout {}", env::var("WHO").unwrap_or_default());
            }
            "fail" => {
                eprintln!("child-stderr");
                std::process::exit(7);
            }
            _ => {}
        }
        return;
    }
    let me = env::current_exe().unwrap();
    let out = Command::new(&me)
        .arg("greet")
        .env("WHO", "world")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "child-stdout world\n"
    );
    let status = Command::new(&me).arg("fail").status().unwrap();
    assert_eq!(status.code(), Some(7));
    println!("process-host-self-ok");
}
"#;

#[test]
fn run_mode_process_host_self_reexec() {
    let dir = temp_dir("proc-host-self");
    let src = fixture(&dir, "process_host_self.rs", PROCESS_HOST_SELF_SRC);
    let out = run_driver_with_process_host(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "process-host-self-ok\n"
    );
    // The self-child's own stderr joins the guest's through
    // the `status` relay.
    assert!(stderr.contains("child-stderr"), "stderr={stderr}");
}

const PROCESS_HOST_SELF_SPAWN_SRC: &str = r#"
use std::env;
use std::io::Read;
use std::process::{Command, Stdio};
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "child" {
        eprintln!("spawned-child-stderr");
        println!("spawned-child-stdout");
        std::process::exit(3);
    }
    // spawn + wait_with_output: the child's stderr reaches
    // the guest as a stream, as it does for a host child.
    let out = Command::new(&args[0])
        .arg("child")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3));
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "spawned-child-stdout\n"
    );
    assert_eq!(
        String::from_utf8(out.stderr).unwrap(),
        "spawned-child-stderr\n"
    );
    // Reading the stderr handle before wait observes the
    // same stream.
    let mut child = Command::new(&args[0])
        .arg("child")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut text = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut text)
        .unwrap();
    assert_eq!(text, "spawned-child-stderr\n");
    assert_eq!(child.wait().unwrap().code(), Some(3));
    // The same through the guest's own executable path.
    let me = env::current_exe().unwrap();
    let out = Command::new(&me)
        .arg("child")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3));
    assert_eq!(
        String::from_utf8(out.stderr).unwrap(),
        "spawned-child-stderr\n"
    );
    println!("process-host-self-spawn-ok");
}
"#;

#[test]
fn run_mode_process_host_self_spawn_stderr() {
    let dir = temp_dir("proc-host-self-spawn");
    let src = fixture(
        &dir,
        "process_host_self_spawn.rs",
        PROCESS_HOST_SELF_SPAWN_SRC,
    );
    let out = run_driver_with_process_host(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "process-host-self-spawn-ok\n"
    );
}

const PROCESS_HOST_BROKEN_PIPE_SRC: &str = r#"
use std::env;
use std::io::Read;
use std::process::{Command, Stdio};
fn main() {
    let mut args = env::args();
    let _me = args.next().unwrap();
    if let Some(arg) = args.next() {
        if arg == "consume" {
            // Read exactly one line from stdin, print it, exit:
            // the producer's pipe then has no reader left.
            let mut line = Vec::new();
            let mut byte = [0u8; 1];
            loop {
                let n = std::io::stdin().read(&mut byte).unwrap();
                if n == 0 || byte[0] == b'\n' {
                    break;
                }
                line.push(byte[0]);
            }
            println!("{}", String::from_utf8(line).unwrap());
            return;
        }
        // Producer: print the line until the pipe breaks.
        for _ in 0..100_000 {
            println!("{arg}");
        }
        unreachable!("the broken pipe must panic first");
    }
    let me = env::current_exe().unwrap();
    let mut producer = Command::new(&me)
        .arg("this line shall appear exactly once on stdout")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    // The producer's stdout handle becomes the consumer's
    // stdin through `Stdio::from(ChildStdout)`.
    let mut consumer = Command::new(&me)
        .arg("consume")
        .stdin(producer.stdout.take().unwrap())
        .spawn()
        .unwrap();
    let consumer_status = consumer.wait().unwrap();
    assert!(consumer_status.success());
    let producer_status = producer.wait().unwrap();
    // The producer's `println!` panics on the broken pipe.
    assert_eq!(producer_status.code(), Some(101));
    println!("broken-pipe-stdio-ok");
}
"#;

#[test]
fn run_mode_process_host_stdio_from_child_stdout_broken_pipe() {
    let dir = temp_dir("proc-host-broken-pipe");
    let src = fixture(
        &dir,
        "process_host_broken_pipe.rs",
        PROCESS_HOST_BROKEN_PIPE_SRC,
    );
    let out = run_driver_with_process_host(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "this line shall appear exactly once on stdout\nbroken-pipe-stdio-ok\n"
    );
    // The producer's broken-pipe panic report joins the
    // guest's stderr through the wait relay, located inside
    // std's `print_to` as a native run reports it.
    assert!(
        stderr.contains("failed printing to stdout: Broken pipe (os error 32)"),
        "stderr={stderr}"
    );
    assert!(
        stderr.contains("library/std/src/io/stdio.rs:1209:9"),
        "stderr={stderr}"
    );
}

const PROCESS_HOST_STDIO_FROM_SRC: &str = r#"
use std::fs::File;
use std::io::{Read, Write};
use std::process::{Command, Stdio};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "emit-out" {
        println!("hello-from-child");
        return;
    }
    if args.len() > 1 && args[1] == "cat" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).unwrap();
        print!("{buf}");
        return;
    }
    if args.len() > 1 && args[1] == "emit-err" {
        eprintln!("guest-err-line");
        std::process::exit(3);
    }
    let dir = &args[1];
    let me = std::env::current_exe().unwrap();

    // A child's stdout is a guest `File`, through the
    // explicit `Stdio::from(File)` conversion.
    let out_path = format!("{dir}/out.txt");
    let file = File::create(&out_path).unwrap();
    let status = Command::new(&me)
        .arg("emit-out")
        .stdout(Stdio::from(file))
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(
        std::fs::read_to_string(&out_path).unwrap(),
        "hello-from-child\n"
    );

    // A child's stdin is a guest `File`, likewise.
    let in_path = format!("{dir}/in.txt");
    let mut f = File::create(&in_path).unwrap();
    f.write_all(b"line-from-file\n").unwrap();
    drop(f);
    let infile = File::open(&in_path).unwrap();
    let out = Command::new(&me)
        .arg("cat")
        .stdin(Stdio::from(infile))
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "line-from-file\n");

    // A child's stderr is a guest `File`, through `Into`.
    let err_path = format!("{dir}/err.txt");
    let errfile = File::create(&err_path).unwrap();
    let err_stdio: Stdio = errfile.into();
    let status = Command::new(&me)
        .arg("emit-err")
        .stderr(err_stdio)
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(3));
    assert_eq!(
        std::fs::read_to_string(&err_path).unwrap(),
        "guest-err-line\n"
    );

    // A taken child pipe handle converts explicitly too.
    let mut producer = Command::new(&me)
        .arg("emit-out")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let taken = producer.stdout.take().unwrap();
    let out = Command::new(&me)
        .arg("cat")
        .stdin(Stdio::from(taken))
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "hello-from-child\n");
    assert!(producer.wait().unwrap().success());
    println!("stdio-from-ok");
}
"#;

#[test]
fn run_mode_process_host_stdio_from_file_and_child_handles() {
    let dir = temp_dir("proc-host-stdio-from");
    let src = fixture(
        &dir,
        "process_host_stdio_from.rs",
        PROCESS_HOST_STDIO_FROM_SRC,
    );
    let dir_arg = dir.to_string_lossy().into_owned();
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--guest-process-host")
        .arg("--guest-arg")
        .arg(&dir_arg)
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "stdio-from-ok\n");
}

const PROCESS_HOST_FAILURE_SRC: &str = r#"
use std::io::ErrorKind;
use std::process::Command;
fn main() {
    let e = Command::new("definitely-not-a-real-program-xyz")
        .spawn()
        .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::NotFound);
    let e = Command::new("definitely-not-a-real-program-xyz")
        .output()
        .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::NotFound);
    println!("process-host-failure-ok");
}
"#;

#[test]
fn run_mode_process_host_spawn_failure() {
    let dir = temp_dir("proc-host-failure");
    let src = fixture(&dir, "process_host_failure.rs", PROCESS_HOST_FAILURE_SRC);
    let out = run_driver_with_process_host(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "process-host-failure-ok\n"
    );
}

const C_MAIN_SRC: &str = r#"
#![no_std]
#![no_main]

extern crate std as other;

#[no_mangle]
extern "C" fn main(argc: core::ffi::c_int, argv: *const *const u8) -> core::ffi::c_int {
    // The modelled entry arguments: the program path alone,
    // NUL-terminated, followed by a null terminator slot.
    if argc != 1 || argv.is_null() {
        return 3;
    }
    unsafe {
        let path = *argv;
        if path.is_null() || *path == 0 || !(*argv.add(1)).is_null() {
            return 4;
        }
    }
    0
}
"#;

#[test]
fn run_mode_c_convention_main_gets_argc_argv() {
    let dir = temp_dir("c-main");
    let src = fixture(&dir, "c_main.rs", C_MAIN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
}

// ---- Lane IW: thread spawn under optimization (issue #158) ----
// With `-O`, `mem::zeroed` inside the spawned closure is
// inlined into a `write_bytes` plus a `MaybeUninit` ->
// array `Transmute` cast in the closure body itself; the
// spawn / join path must carry it.

const SPAWN_ZEROED_OPT_SRC: &str = r#"
use std::thread;
fn main() {
    let h = thread::spawn(|| {
        let a: [u8; 65536] = unsafe { std::mem::zeroed() };
        a[0] as i32 + a[65535] as i32
    });
    std::process::exit(h.join().unwrap());
}
"#;

#[test]
fn run_mode_thread_spawn_zeroed_array_under_optimization() {
    // The joined sum of two zeroed bytes is the exit code:
    // reaching it at all proves the closure's `write_bytes`
    // plus `Transmute` executed and the handle joined.
    let dir = temp_dir("spawn-zeroed-opt");
    let src = fixture(&dir, "spawn_zeroed.rs", SPAWN_ZEROED_OPT_SRC);
    let out = run_driver_extra(&src, &["--rustc-arg", "-O"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "");
}

const SPAWN_GATE_SRC: &str = r#"
use std::process::Command;
fn main() {
    let target = std::env::args().nth(1).unwrap();
    let err = Command::new(&target).status().unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
    println!("gate ok");
}
"#;

#[test]
fn run_mode_spawn_of_non_executable_is_an_io_error() {
    // The spawn gate still applies to programs the run has
    // not claimed: a data file is not executable, and the
    // guest observes the permission error as a value.
    let dir = temp_dir("spawn-gate");
    let data = fixture(&dir, "data.txt", "not a program\n");
    let src = fixture(&dir, "gate.rs", SPAWN_GATE_SRC);
    let data_arg = data.to_string_lossy().into_owned();
    let out = run_driver_extra(&src, &["--guest-arg", &data_arg]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "gate guest must exit 0; stdout={stdout} stderr={stderr}"
    );
    assert_eq!(stdout, "gate ok\n");
}
const CHILD_STDIO_SRC: &str = r#"
use std::env;
use std::io::{BufRead, Write};
use std::process::{Command, Stdio};
fn main() {
    let me = env::args_os().next().unwrap();
    if env::args().nth(1).as_deref() == Some("child") {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            println!("{}", line.unwrap());
        }
        return;
    }
    let mut child = Command::new(&me)
        .arg("child")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"test1\ntest2\ntest3")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b"test1\ntest2\ntest3\n");
    println!("handles ok");
}
"#;

#[test]
fn run_mode_child_stdio_handles_round_trip() {
    // The `Child` value's stdio slots carry the child's pipe
    // ends as file handles: the parent writes the child's
    // stdin through the handle, `wait_with_output` closes the
    // end it still holds (so the child's line iteration sees
    // EOF) and collects the child's stdout, and the child's
    // own `std::io::stdin` reads the pipe the parent wired.
    let dir = temp_dir("child-stdio");
    let src = fixture(&dir, "childstdio.rs", CHILD_STDIO_SRC);
    let out = run_driver_with_process_host(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "child-stdio guest must exit 0; stdout={stdout} stderr={stderr}"
    );
    assert_eq!(stdout, "handles ok\n");
}

const CHILD_WAIT_STDIN_SRC: &str = r#"
use std::env;
use std::io::{BufRead, Write};
use std::process::{Command, Stdio};
fn main() {
    let me = env::args_os().next().unwrap();
    if env::args().nth(1).as_deref() == Some("child") {
        println!("foo");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        let n = std::io::stdin().lock().read_line(&mut line).unwrap();
        assert_eq!(n, 0);
        return;
    }
    let mut child = Command::new(&me)
        .arg("child")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let status = child.wait().unwrap();
    assert!(status.success());
    println!("wait ok");
}
"#;

#[test]
fn run_mode_child_wait_closes_held_stdin() {
    // `Child::wait` drops the stdin handle the `Child` still
    // holds before waiting, so a child reading its stdin to
    // the end observes EOF instead of blocking on a parent
    // that never writes.
    let dir = temp_dir("child-wait-stdin");
    let src = fixture(&dir, "childwaitstdin.rs", CHILD_WAIT_STDIN_SRC);
    let out = run_driver_with_process_host(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "child-wait guest must exit 0; stdout={stdout} stderr={stderr}"
    );
    assert_eq!(stdout, "wait ok\n");
}

const CHILD_TAKEN_STDOUT_SRC: &str = r#"
use std::env;
use std::io::Read;
use std::process::{Command, Stdio};
fn main() {
    let me = env::args_os().next().unwrap();
    if env::args().nth(1).as_deref() == Some("child") {
        println!("child text");
        return;
    }
    let mut child = Command::new(&me)
        .arg("child")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut taken = child.stdout.take().unwrap();
    let status = child.wait().unwrap();
    assert!(status.success());
    let mut text = String::new();
    taken.read_to_string(&mut text).unwrap();
    assert_eq!(text, "child text\n");
    println!("taken ok");
}
"#;

#[test]
fn run_mode_child_taken_stdout_belongs_to_the_guest() {
    // A stdio handle the guest takes out of the `Child` is
    // the guest's to read: `wait` neither collects nor
    // closes it, and reading it after the child ends yields
    // the child's bytes.
    let dir = temp_dir("child-taken-stdout");
    let src = fixture(&dir, "childtaken.rs", CHILD_TAKEN_STDOUT_SRC);
    let out = run_driver_with_process_host(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "taken-stdout guest must exit 0; stdout={stdout} stderr={stderr}"
    );
    assert_eq!(stdout, "taken ok\n");
}

const CHILD_BROKEN_PIPE_SRC: &str = r#"
use std::env;
use std::io::prelude::*;
use std::io;
use std::process::{Command, Stdio};
fn test() {
    let _ = io::stdin().read_line(&mut String::new());
    io::stdout().write(&[1]).unwrap();
    assert!(io::stdout().flush().is_err());
}
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "test" {
        return test();
    }
    let mut p = Command::new(&args[0])
        .stdout(Stdio::piped())
        .stdin(Stdio::piped())
        .arg("test")
        .spawn()
        .unwrap();
    drop(p.stdout.take());
    assert!(p.wait().unwrap().success());
    println!("pipe ok");
}
"#;

#[test]
fn run_mode_child_observes_the_broken_pipe_at_flush() {
    // A re-run child's stdout is the pipe the parent wired:
    // when the parent drops the read end, the child's writes
    // are buffered by its line writer (the small write
    // succeeds) and the flush observes the broken pipe, as
    // natively — the write-through sink carries the stream's
    // errors into the guest.
    let dir = temp_dir("child-broken-pipe");
    let src = fixture(&dir, "childpipe.rs", CHILD_BROKEN_PIPE_SRC);
    let out = run_driver_with_process_host(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "broken-pipe guest must exit 0; stdout={stdout} stderr={stderr}"
    );
    assert_eq!(stdout, "pipe ok\n");
}

const SPAWN_RESULT_DEBUG_SRC: &str = r#"
use std::env;
use std::process::Command;
fn main() {
    let me = env::args_os().next().unwrap();
    if env::args().nth(1).as_deref() == Some("child") {
        return;
    }
    let st = Command::new(&me).arg("child").status();
    println!("{:?}", st);
}
"#;

#[test]
fn run_mode_exit_status_debug_names_the_wait_status() {
    // std's `Debug` for `ExitStatus` is a manual impl naming
    // the raw wait status, and a guest printing a spawn
    // result observes exactly that text.
    let dir = temp_dir("exit-status-debug");
    let src = fixture(&dir, "exitstatus.rs", SPAWN_RESULT_DEBUG_SRC);
    let out = run_driver_with_process_host(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "exit-status guest must exit 0; stdout={stdout} stderr={stderr}"
    );
    assert_eq!(stdout, "Ok(ExitStatus(unix_wait_status(0)))\n");
}
