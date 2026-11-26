// Printing to stdout: `println!` output and compile-time
// output redirection tests.

const PRINTLN_SRC: &str = r#"
fn main() { println!("hi"); }
"#;

#[test]
fn run_mode_println_prints_to_stdout() {
    let dir = temp_dir("println");
    let src = fixture(&dir, "println.rs", PRINTLN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "println! runs through the interpreter's runtime print support; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hi\n",
        "the guest's stdout is the driver's stdout"
    );
}

const LOG_SYNTAX_SRC: &str = r#"
#![feature(log_syntax)]
fn main() {
    log_syntax!("compile-time marker");
    println!("guest marker");
}
"#;

#[test]
fn run_mode_log_syntax_output_stays_off_stdout() {
    // `log_syntax!` prints to the *compiler's* stdout at compile
    // time. The compiler runs in-process, so without redirection
    // that output would land in the driver's stdout — which run
    // mode reserves for the guest program's output. It must appear
    // on stderr (compiler-side diagnostics) instead, leaving stdout
    // byte-identical to the guest's own output.
    let dir = temp_dir("logsyntax");
    let src = fixture(&dir, "logsyntax.rs", LOG_SYNTAX_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the program itself runs fine; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "guest marker\n",
        "stdout carries the guest's output alone"
    );
    assert!(
        stderr.contains("compile-time marker"),
        "the log_syntax output is redirected to stderr: {stderr}"
    );
}

const PRINTLN_DYNAMIC_COUNT_SRC: &str = r#"fn main() {
    let w = 6usize;
    println!("{:w$}", 42);
    println!("{:.*}", 2, 3.14159f64);
}
"#;

#[test]
fn run_mode_println_dynamic_width_and_precision() {
    // A dynamic width / precision (`{:w$}`, `{:.*}`) reaches
    // the formatter through `rt::Argument::from_usize`, which
    // current core calls with a *reference* to the count (the
    // format-args capture tuple holds references): the builtin
    // reads through the reference to the count value.
    let dir = temp_dir("println-dynamic-count");
    let src = fixture(&dir, "println_count.rs", PRINTLN_DYNAMIC_COUNT_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "    42\n3.14\n");
}
