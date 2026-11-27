//! Guest-visible stack overflow: a guest that would overflow
//! its native stack observes the native outcome -- the
//! runtime's `has overflowed its stack` report on stderr and
//! an abort (the driver's 134 crash exit) -- at the point the
//! charged frame sizes cross the thread's stack budget, not
//! at the interpreter's call-depth backstop.

use super::{fixture, run_driver, temp_dir};

const RECURSE_SRC: &str = r#"
#[allow(unconditional_recursion)]
fn recurse() {
    let local = [0u64; 1024];
    std::hint::black_box(&local);
    recurse();
}

fn main() {
    std::thread::spawn(recurse).join().unwrap();
}
"#;

#[test]
fn spawned_thread_recursion_overflows_its_stack() {
    let dir = temp_dir("stack-overflow-recurse");
    let src = fixture(&dir, "recurse.rs", RECURSE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "a stack overflow aborts the run (134); stderr={stderr}"
    );
    assert!(
        stderr.contains("thread '<unknown>' has overflowed its stack\n"),
        "the runtime's overflow report names the unnamed thread: {stderr}"
    );
    assert!(
        stderr.contains("fatal runtime error: stack overflow, aborting\n"),
        "the runtime's fatal line follows the report: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "nothing reaches stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

const BIG_FRAME_SRC: &str = r#"
fn main() {
    std::thread::Builder::new()
        .stack_size(512 * 1024)
        .spawn(|| {
            let local = [0u8; 1024 * 1024];
            std::hint::black_box(&local);
        })
        .unwrap()
        .join()
        .unwrap();
}
"#;

#[test]
fn single_frame_larger_than_the_thread_stack_overflows_at_entry() {
    let dir = temp_dir("stack-overflow-frame");
    let src = fixture(&dir, "big_frame.rs", BIG_FRAME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "one oversized frame overflows a small stack at its entry; stderr={stderr}"
    );
    assert!(
        stderr.contains("has overflowed its stack\n"),
        "the runtime's overflow report is on stderr: {stderr}"
    );
}

const MAIN_RECURSE_SRC: &str = r#"
#[allow(unconditional_recursion)]
fn recurse() {
    let local = [0u64; 1024];
    std::hint::black_box(&local);
    recurse();
}

fn main() {
    recurse();
}
"#;

#[test]
fn main_thread_recursion_overflows_its_stack() {
    let dir = temp_dir("stack-overflow-main");
    let src = fixture(&dir, "main_recurse.rs", MAIN_RECURSE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "the main thread's budget is its native stack size; stderr={stderr}"
    );
    assert!(
        stderr.contains("thread 'main' has overflowed its stack\n"),
        "the overflow report names the main thread: {stderr}"
    );
}

const WITHIN_BUDGET_SRC: &str = r#"
fn depth(n: u32) -> u32 {
    let local = [0u64; 64];
    std::hint::black_box(&local);
    if n == 0 { 0 } else { depth(n - 1) + 1 }
}

fn main() {
    assert_eq!(depth(2000), 2000);
    println!("done");
}
"#;

#[test]
fn recursion_within_the_stack_budget_completes() {
    let dir = temp_dir("stack-within-budget");
    let src = fixture(&dir, "within.rs", WITHIN_BUDGET_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "deep but small-frame recursion stays within budget; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "done\n");
}
