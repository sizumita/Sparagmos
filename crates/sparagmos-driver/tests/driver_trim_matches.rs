//! `str::trim_matches` / `trim_start_matches` / `trim_end_matches`
//! with a pattern argument must compute the trimmed slice for
//! every pattern kind (`&str`, `char`, `&[char]`, a function
//! pointer such as `char::is_alphanumeric`, and a closure).
//! Before these were modelled, the calls ran core's collected
//! searcher bodies, whose per-character stepping handed the
//! nested `starts_with` search a scalar where the string data
//! belongs: the run stopped with "type mismatch in string
//! argument: expected string data, found unsigned integer"
//! (the failure that kept `tests/ui/backtrace/backtrace.rs`
//! in the #511 panic cluster, via its
//! `contains_verbose_expected` helper).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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
    let dir = std::env::temp_dir()
        .join(format!("sparagmos-driver-trim-matches-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn run_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .output()
        .expect("failed to run sparagmos-driver")
}

/// Run one fixture and require a clean exit with exactly the
/// expected stdout lines (one per `println!` in the fixture).
fn expect_output(tag: &str, src: &str, expected: &[&str]) {
    let dir = temp_dir(tag);
    let file = fixture(&dir, "case.rs", src);
    let out = run_driver(&file);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{tag}: expected exit 0, stdout: {stdout}, stderr: {stderr}"
    );
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines, expected, "{tag}: stdout mismatch, stderr: {stderr}");
}

#[test]
fn trim_matches_pattern_kinds() {
    expect_output(
        "kinds",
        r#"
fn main() {
    println!("[{}]", "11foo1bar11".trim_start_matches('1'));
    println!("[{}]", "123foo1bar123".trim_start_matches(char::is_numeric));
    let x: &[char] = &['1', '2'];
    println!("[{}]", "12foo1bar12".trim_start_matches(x));
    println!("[{}]", "11foo1bar11".trim_matches('1'));
    println!("[{}]", "11foo1bar11".trim_end_matches('1'));
    println!("[{}]", "aaa".trim_matches('a'));
    println!("[{}]", "abc".trim_start_matches(""));
    println!("[{}]", "abab".trim_start_matches("ab"));
    println!("[{}]", "abab".trim_end_matches("ab"));
    println!("[{}]", "111".trim_matches(char::is_numeric));
    println!("[{}]", "ééfooé".trim_matches('é'));
}
"#,
        &[
            "[foo1bar11]",
            "[foo1bar123]",
            "[foo1bar12]",
            "[foo1bar]",
            "[11foo1bar]",
            "[]",
            "[abc]",
            "[]",
            "[]",
            "[]",
            "[foo]",
        ],
    );
}

#[test]
fn trim_start_matches_fn_pointer_predicate() {
    expect_output(
        "fnptr",
        r#"
fn main() {
    let t = "abc123]xyz".trim_start_matches(char::is_alphanumeric);
    println!("[{}]", t);
    let u = "  xyz  ".trim_matches(char::is_whitespace);
    println!("[{}]", u);
}
"#,
        &["[]xyz]", "[xyz]"],
    );
}

#[test]
fn trim_start_matches_closure_predicate_counts_calls() {
    expect_output(
        "closure",
        r#"
fn main() {
    let mut n = 0;
    let t = "aaab".trim_start_matches(|c: char| { n += 1; c == 'a' });
    println!("[{}] n={}", t, n);
    let v = "xyxfooxyx".trim_matches(|c: char| c == 'x' || c == 'y');
    println!("[{}]", v);
}
"#,
        &["[b] n=4", "[foo]"],
    );
}

/// The `backtrace.rs` shape: a `match_indices` iterator stepped
/// by `any`, whose closure slices the haystack, trims a
/// `char::is_alphanumeric` run and probes the remainder with
/// `starts_with`.
#[test]
fn trim_inside_match_indices_any_closure() {
    expect_output(
        "verbose",
        r#"
fn contains_verbose_expected(s: &str, fn_name: &str) -> bool {
    let prefix = " backtrace";
    let suffix = &format!("::{fn_name}");
    s.match_indices(prefix).any(|(i, _)| {
        s[i + prefix.len()..]
            .trim_start_matches('[')
            .trim_start_matches(char::is_alphanumeric)
            .trim_start_matches(']')
            .starts_with(suffix)
    })
}

fn main() {
    let s = "   3:     0x5555 - <backtrace::double::Double as std::ops::Drop>::drop\n";
    println!("verbose={}", contains_verbose_expected(s, "double"));
    println!("other={}", contains_verbose_expected(s, "foo"));
}
"#,
        &["verbose=true", "other=false"],
    );
}
