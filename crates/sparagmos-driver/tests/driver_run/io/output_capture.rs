// `std::io::set_output_capture`: the per-thread output
// capture std's `print_to` and the default panic report
// divert into (the internal_output_capture surface the
// rustc suite's panic / thread tests exercise).

const OUTPUT_CAPTURE_TASK_STDERR_SRC: &str = r##"
#![feature(internal_output_capture)]

use std::sync::{Arc, Mutex};
use std::{io, str, thread};

fn main() {
    let data = Arc::new(Mutex::new(Vec::new()));
    let res = thread::Builder::new()
        .spawn({
            let data = data.clone();
            move || {
                io::set_output_capture(Some(data));
                panic!("Hello, world!")
            }
        })
        .unwrap()
        .join();
    assert!(res.is_err());

    let output = data.lock().unwrap();
    let output = str::from_utf8(&output).unwrap();
    assert!(output.contains("Hello, world!"), "capture was: {output}");
    println!("task stderr ok");
}
"##;

#[test]
fn run_mode_output_capture_task_stderr() {
    // The rustc suite's `task-stderr` record: a spawned
    // thread installs the capture, then panics; the default
    // panic report must land in the capture vector (not on
    // the guest's stderr), where the main thread reads it
    // back through the shared `Arc<Mutex<Vec<u8>>>`.
    let dir = temp_dir("output-capture-task-stderr");
    let src = fixture(&dir, "task_stderr.rs", OUTPUT_CAPTURE_TASK_STDERR_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "task stderr ok\n");
    assert!(
        !stderr.contains("Hello, world!"),
        "the panic report was captured, not printed: {stderr}"
    );
}

const OUTPUT_CAPTURE_PANIC_DISPLAY_SRC: &str = r##"
#![feature(internal_output_capture)]

use std::fmt;
use std::fmt::{Display, Formatter};
use std::io::set_output_capture;
use std::sync::{Arc, Mutex};

pub struct A;

impl Display for A {
    fn fmt(&self, _f: &mut Formatter<'_>) -> fmt::Result {
        panic!();
    }
}

fn main() {
    set_output_capture(Some(Arc::new(Mutex::new(Vec::new()))));
    assert!(
        std::panic::catch_unwind(|| {
            eprintln!("{}", A);
        })
        .is_err()
    );
}
"##;

#[test]
fn run_mode_output_capture_panic_during_display() {
    // The rustc suite's `panic-during-display-formatting`
    // record: with a capture installed, an `eprintln!` whose
    // `Display` impl panics must still surface the panic to
    // `catch_unwind` (the format runs before the diverted
    // write, exactly as std's `print_to` formats first).
    let dir = temp_dir("output-capture-panic-display");
    let src = fixture(
        &dir,
        "panic_display.rs",
        OUTPUT_CAPTURE_PANIC_DISPLAY_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
}

const OUTPUT_CAPTURE_PRINT_DIVERT_SRC: &str = r##"
#![feature(internal_output_capture)]

use std::io::set_output_capture;
use std::sync::{Arc, Mutex};

fn main() {
    let data = Arc::new(Mutex::new(Vec::new()));
    let prev = set_output_capture(Some(data.clone()));
    assert!(prev.is_none());
    println!("captured stdout");
    eprintln!("captured stderr");
    let prev = set_output_capture(None);
    assert!(prev.is_some());
    let output = data.lock().unwrap();
    let text = String::from_utf8_lossy(&output).into_owned();
    assert!(text.contains("captured stdout"), "capture was: {text}");
    assert!(text.contains("captured stderr"), "capture was: {text}");
    println!("divert ok");
}
"##;

#[test]
fn run_mode_output_capture_print_divert_and_restore() {
    // While a capture is installed, both `println!` and
    // `eprintln!` append to the capture vector and nothing
    // reaches the guest's streams; removing the capture
    // returns the previous one and restores normal printing.
    let dir = temp_dir("output-capture-print-divert");
    let src = fixture(&dir, "print_divert.rs", OUTPUT_CAPTURE_PRINT_DIVERT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "divert ok\n");
    assert!(!stderr.contains("captured"), "stderr={stderr}");
}
