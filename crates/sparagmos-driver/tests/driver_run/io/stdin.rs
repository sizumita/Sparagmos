// Guest stdin tests.

const GUEST_STDIN_SRC: &str = r#"
use std::io::BufRead;
fn main() {
    let mut line = String::new();
    let n = std::io::stdin().lock().read_line(&mut line).unwrap();
    println!("got {}: {}", n, line.trim());
}
"#;

#[test]
fn run_mode_guest_reads_its_own_stdin() {
    // The guest's `std::io::stdin` is the run's own stdin
    // in the file table: a line read consumes the piped
    // bytes the driver process was fed.
    let dir = temp_dir("guest-stdin");
    let src = fixture(&dir, "gueststdin.rs", GUEST_STDIN_SRC);
    let mut child = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn sparagmos-driver");
    use std::io::Write;
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"hello\n")
        .unwrap();
    let out = child.wait_with_output().expect("driver run failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "guest-stdin guest must exit 0; stdout={stdout} stderr={stderr}"
    );
    assert_eq!(stdout, "got 6: hello\n");
}
