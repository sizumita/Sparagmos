//! Coverage-run regression tests: execute small instrumented
//! guests with `sparagmos-driver coverage-run` and check the
//! rendered report's counts and instantiation subviews.

use std::path::Path;
use std::process::{Command, Output};

use super::{fixture, temp_dir};

fn run_coverage(src: &Path, aux: &[(&str, &Path)], extra_rustc_args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
    cmd.arg("coverage-run").arg(src).arg("--edition").arg("2021");
    cmd.arg("--guest-process-host");
    for (name, path) in aux {
        cmd.arg("--aux").arg(format!("{}={}", name, path.display()));
    }
    let mut args = vec!["-C", "instrument-coverage", "-Copt-level=2"];
    args.extend_from_slice(extra_rustc_args);
    for a in args {
        cmd.arg("--rustc-arg").arg(a);
    }
    cmd.output().expect("failed to run sparagmos-driver")
}

/// The execution-count field of the first report line that
/// contains `needle` (the second `|`-separated field of an
/// llvm-cov text line), or `None` when no line matches.
fn line_count(report: &str, needle: &str) -> Option<String> {
    report.lines().find_map(|line| {
        if !line.contains(needle) {
            return None;
        }
        let mut fields = line.split('|');
        fields.next()?;
        fields.next().map(|f| f.trim().to_string())
    })
}

const INLINED_GENERIC_SRC: &str = r#"
use std::fmt::Display;

fn main() {
    permutations(&['a', 'b']);
}

#[inline(always)]
fn permutations<T: Copy + Display>(xs: &[T]) {
    let mut ys = xs.to_owned();
    permutate(&mut ys, 0);
}

fn permutate<T: Copy + Display>(xs: &mut [T], k: usize) {
    let n = length(xs);
    if k == n {
        display(xs);
    } else {
        for i in k..n {
            swap(xs, i, k);
            permutate(xs, k + 1);
            swap(xs, i, k);
        }
    }
}

fn length<T>(xs: &[T]) -> usize {
    xs.len()
}

#[inline]
fn swap<T: Copy>(xs: &mut [T], i: usize, j: usize) {
    let t = xs[i];
    xs[i] = xs[j];
    xs[j] = t;
}

fn display<T: Display>(xs: &[T]) {
    for x in xs {
        print!("{}", x);
    }
    println!();
}
"#;

/// With `-Zinline-mir`, the tiny generic `length` is inlined
/// into its caller and never converted as a body of its own;
/// its coverage statements execute inside the caller's body,
/// attributed to it through the inlined scope. The attribution
/// must name the instance with the caller's arguments
/// substituted (the identity the capture records), or the
/// statements resolve to no record and the function reports as
/// never executed. `permutate` runs 5 times for a 2-element
/// permutation, so `length` counts 5.
#[test]
fn coverage_run_inlined_generic_counts_against_its_own_record() {
    let dir = temp_dir("coverage-inlined-generic");
    let src = fixture(&dir, "inlined_generic.rs", INLINED_GENERIC_SRC);
    let out = run_coverage(&src, &[], &["-Zinline-mir"]);
    assert!(
        out.status.success(),
        "coverage-run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(
        line_count(&report, "fn length<T>(xs: &[T]) -> usize {").as_deref(),
        Some("5"),
        "length must count its own executions:\n{report}"
    );
    assert_eq!(
        line_count(&report, "fn permutate<T: Copy + Display>").as_deref(),
        Some("5"),
        "permutate's own count is unchanged:\n{report}"
    );
}

const AUX_LIB_SRC: &str = r#"
pub fn entry() {
    internal_generic(1u32);
}

pub fn internal_generic<T: Copy>(x: T) -> T {
    x
}

pub fn bin_only_generic<T: Copy>(x: T) -> T {
    x
}

pub fn never_generic<T: Copy>(x: T) -> T {
    x
}
"#;

const AUX_MAIN_SRC: &str = r#"
extern crate covaux;

fn main() {
    covaux::entry();
    covaux::bin_only_generic(2u32);
}
"#;

/// An aux crate's generic function that the aux crate's own
/// code uses is a mono item of the aux compilation, so its
/// codegen emits no unused record for it: the report must not
/// invent an `Unexecuted instantiation` subview for it. A
/// generic used only from the binary crate does get the unused
/// record from the aux compilation (and its subview), and a
/// never-used generic renders with count 0.
#[test]
fn coverage_run_aux_crate_home_used_generic_has_no_unexecuted_subview() {
    let dir = temp_dir("coverage-aux-home-used");
    let aux = fixture(&dir, "covaux.rs", AUX_LIB_SRC);
    let src = fixture(&dir, "aux_main.rs", AUX_MAIN_SRC);
    let out = run_coverage(
        &src,
        &[("covaux", &aux)],
        &["-C", "opt-level=3"],
    );
    assert!(
        out.status.success(),
        "coverage-run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        !report.contains("Unexecuted instantiation: covaux::internal_generic"),
        "home-used generic must not render as unexecuted:\n{report}"
    );
    assert_eq!(
        report
            .matches("Unexecuted instantiation: covaux::bin_only_generic")
            .count(),
        1,
        "bin-only generic keeps its one unexecuted subview:\n{report}"
    );
    assert_eq!(
        line_count(&report, "pub fn internal_generic<T: Copy>(x: T) -> T {").as_deref(),
        Some("1"),
        "the home-used generic executed once:\n{report}"
    );
    assert_eq!(
        line_count(&report, "pub fn never_generic<T: Copy>(x: T) -> T {").as_deref(),
        Some("0"),
        "the never-used generic renders unexecuted:\n{report}"
    );
}

const WRITE_MACRO_FMT_SRC: &str = r#"
struct DebugTest;

impl std::fmt::Debug for DebugTest {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "cool")?;
        for _i in 0..10 {
            write!(f, "cool")?;
        }
        Ok(())
    }
}

fn main() {
    println!("{:?}", DebugTest);
}
"#;

/// A guest `fmt` body that writes through the `write!` macro:
/// at opt-level 2 core's `Formatter::write_fmt` is inlined into
/// the body, which then reads the formatter's own fields -- the
/// `buf` trait object (field 1), whose `Write` dispatch must
/// land in the same capture the body's direct `write_str` calls
/// use. The record this bounds is the coverage suite's
/// `iffy/loops_branches` (and `iffy/issue-84561`).
#[test]
fn coverage_run_write_macro_in_fmt_body_reaches_the_capture() {
    let dir = temp_dir("coverage-write-macro-fmt");
    let src = fixture(&dir, "write_macro_fmt.rs", WRITE_MACRO_FMT_SRC);
    let out = run_coverage(&src, &[], &[]);
    assert!(
        out.status.success(),
        "coverage-run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(
        line_count(&report, "fn fmt(&self, f: &mut std::fmt::Formatter)").as_deref(),
        Some("1"),
        "the fmt body executed once:\n{report}"
    );
    assert_eq!(
        line_count(&report, "for _i in 0..10 {").as_deref(),
        Some("1"),
        "the loop header executed once:\n{report}"
    );
}

const FORMATTER_OPTIONS_SRC: &str = r#"
struct AltTest;

impl std::fmt::Debug for AltTest {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "alt")?;
        }
        write!(f, "cool")
    }
}

fn main() {
    println!("{:#?}", AltTest);
    println!("{:?}", AltTest);
}
"#;

/// A guest `fmt` body that reads its formatter's options
/// through an inlined accessor (`Formatter::alternate` reads
/// the options' flag word): the options field (field 0) is
/// synthesized from the capture the placeholder seeded, so the
/// `{:#?}` call takes the alternate branch and the plain
/// `{:?}` call does not.
#[test]
fn coverage_run_fmt_body_reads_the_captured_options() {
    let dir = temp_dir("coverage-formatter-options");
    let src = fixture(&dir, "formatter_options.rs", FORMATTER_OPTIONS_SRC);
    let out = run_coverage(&src, &[], &[]);
    assert!(
        out.status.success(),
        "coverage-run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(
        line_count(&report, "write!(f, \"alt\")?;").as_deref(),
        Some("1"),
        "only the alternate call writes the alt text:\n{report}"
    );
    assert_eq!(
        line_count(&report, "fn fmt(&self, f: &mut std::fmt::Formatter)").as_deref(),
        Some("2"),
        "the fmt body executed for both placeholders:\n{report}"
    );
}
