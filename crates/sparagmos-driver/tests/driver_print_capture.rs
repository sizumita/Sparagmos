//! Compiler-stdout capture (`--compiler-stdout`) and the printed
//! output it carries, end to end through the driver binary.
//!
//! A `-Zunpretty` dump or a `--print` answer is written by the
//! embedded compiler to its own stdout. Convert mode normally
//! folds compile-time stdout into stderr (the driver process's
//! stdout is its own report); `--compiler-stdout <path>`
//! captures it to a file instead, for a caller that judges the
//! printed output. The printed text must be rustc's own: the
//! conversion's callbacks set the session's trimmed-paths
//! option for printing invocations, as the real rustc entry
//! point does, so a printed Stable MIR body names `Vec` where
//! rustc prints `Vec`, not `std::vec::Vec`.

use std::process::Command;

fn run_driver(case: &str, source: &str, extra: &[&str])
              -> (i32, String, String, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-print-capture-{}-{}",
        std::process::id(),
        case
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("case.rs");
    std::fs::write(&src, source).unwrap();
    let capture = dir.join("compiler-stdout.txt");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg(&src).arg("--edition").arg("2021");
    cmd.env("RUSTC_BOOTSTRAP", "1");
    for a in extra {
        if *a == "@CAPTURE@" {
            cmd.arg("--compiler-stdout").arg(&capture);
        } else {
            cmd.arg(a);
        }
    }
    let out = cmd.output().expect("failed to run sparagmos-driver");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        capture,
    )
}

const VEC_SOURCE: &str = "\
fn bar(vec: &mut Vec<i32>) -> Vec<i32> {
    let mut new_vec = vec.clone();
    new_vec.push(1);
    new_vec
}
fn main() {}
";

#[test]
fn stable_mir_dump_is_captured_with_trimmed_paths() {
    let (code, stdout, stderr, capture) = run_driver(
        "smir", VEC_SOURCE,
        &["@CAPTURE@", "--rustc-arg", "-Zunpretty=stable-mir",
          "--rustc-arg", "-Zmir-opt-level=0"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let dump = std::fs::read_to_string(&capture)
        .expect("the capture file was not produced");
    assert!(dump.contains("Vec<i32>"), "dump={dump}");
    assert!(!dump.contains("std::vec::Vec"), "dump={dump}");
    // The driver's own report stays on its stdout.
    assert!(stdout.contains("compiler skipped conversion"),
            "stdout={stdout}");
}

#[test]
fn print_answer_is_captured() {
    let (code, stdout, stderr, capture) = run_driver(
        "print", "fn main() {}\n",
        &["@CAPTURE@", "--rustc-arg", "--print",
          "--rustc-arg", "host-tuple"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let answer = std::fs::read_to_string(&capture)
        .expect("the capture file was not produced");
    let answer = answer.trim();
    assert!(!answer.is_empty() && !answer.contains(char::is_whitespace),
            "answer={answer:?}");
    assert!(stdout.contains("compiler skipped conversion"),
            "stdout={stdout}");
}

#[test]
fn compiler_stdout_goes_to_stderr_without_capture() {
    // No --compiler-stdout: compile-time printing keeps the
    // standing discipline (stderr), and no capture file appears.
    let (code, stdout, stderr, capture) = run_driver(
        "stderr", VEC_SOURCE,
        &["--rustc-arg", "-Zunpretty=stable-mir",
          "--rustc-arg", "-Zmir-opt-level=0"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stderr.contains("Vec<i32>"), "stderr={stderr}");
    assert!(!capture.exists());
}
