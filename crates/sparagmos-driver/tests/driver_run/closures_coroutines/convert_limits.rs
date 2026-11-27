// Convert-mode limits: closure type-size growth stays a bounded failure.

/// The shape of tests/ui/recursion/infinite-instantiation-via-
/// nested-closures.rs, reduced: four methods pass the generic
/// parameter along a cycle, and two of the hops substitute a
/// fresh closure type that mentions the previous parameter
/// twice (as the enclosing substitution and as a capture), so
/// the demanded arguments' expansion doubles per cycle while
/// the nesting depth grows only linearly.
const SIZE_INCREASING_CLOSURES_SRC: &str = r#"
#![crate_type = "rlib"]
#[derive(Clone)]
struct A(B);

impl A {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let A(term) = self;
        term.matches(f);
    }
}

#[derive(Clone)]
enum B {
    Variant1,
    Variant2(C),
}

impl B {
    pub fn matches<F: Fn()>(&self, f: &F) {
        match self {
            B::Variant2(factor) => factor.matches(&|| ()),
            _ => unreachable!(""),
        }
    }
}

#[derive(Clone)]
struct C(D);

impl C {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let C(base) = self;
        base.matches(&|| {
            C(base.clone()).matches(f);
        });
    }
}

#[derive(Clone)]
struct D(Box<A>);

impl D {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let D(a) = self;
        a.matches(f);
    }
}

pub fn matches() {
    A(B::Variant1).matches(&|| ());
}
"#;

#[test]
fn convert_mode_size_increasing_closures_is_a_size_limit_failure() {
    // The closure-mediated shape of the same disease: the
    // demanded arguments' expansion doubles per cycle through
    // the closure arguments, so the size cap, not the depth
    // cap, is what stops the collection.
    let dir = temp_dir("mono-size-increasing-closures");
    let src = fixture(&dir, "size_increasing_closures.rs", SIZE_INCREASING_CLOSURES_SRC);
    let out = summarize_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "closure-mediated size-increasing instantiation must fail compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("sparagmos-driver: compilation failed")
            && stderr.contains("reached the recursion limit while instantiating")
            && stderr.contains("(instantiation size limit)"),
        "the failure must name the instantiation size limit; stderr={stderr}"
    );
}

// ----- Conversion type-size limit (Issue #481) -------------------------------

/// The source of the nested-closure exponential: each `dup`
/// call wraps the previous closure, and a closure type spells
/// its captured type twice, so the type tree of the final
/// binding doubles per call.
fn nested_dup_closures_src(depth: usize) -> String {
    let mut src = String::from(
        "fn dup(f: impl Fn(i32) -> i32) -> impl Fn(i32) -> i32 {\n    move |a| f(a * 2)\n}\n\nfn main() {\n    let f = |a| a;\n",
    );
    for _ in 0..depth {
        src.push_str("    let f = dup(f);\n");
    }
    src.push_str("    println!(\"{}\", f(1));\n}\n");
    src
}

#[test]
fn convert_mode_nested_closure_type_explosion_is_bounded() {
    // Thirty levels of nesting spell a type tree of ~2^30
    // nodes for the final binding. No single demand trips the
    // collection size cap before conversion starts
    // materializing that tree, so the conversion size limit is
    // what stops it: the run completes with `main` skipped for
    // that reason instead of growing until the runner's memory
    // bound kills the process (Issue #481).
    let dir = temp_dir("nested-closure-type-explosion");
    let src = fixture(&dir, "nested.rs", &nested_dup_closures_src(30));
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "the conversion must complete and report; status={:?}\nstdout={stdout}",
        out.status
    );
    assert!(
        stdout.contains("expanded type tree exceeds the conversion size limit"),
        "main must be skipped by the conversion size limit:\n{stdout}"
    );
    // The refusal happens once, up front: the body pre-scan in
    // `convert_body` rejects the body for its over-limit local
    // before any local or statement is converted. Without it,
    // every mention of the doubling types was converted (and
    // refused) separately — the same reason reported 52 times
    // and gigabytes materialized for a body that can only ever
    // be skipped (Issue #481).
    assert!(
        stdout.contains(
            "SKIPPED (expanded type tree exceeds the conversion size limit (2097152 nodes) x1)"
        ),
        "main must be refused exactly once, by the body pre-scan:\n{stdout}"
    );
}

#[test]
fn convert_mode_moderate_nested_closures_still_convert() {
    // Ten levels stay far below the limit: the program
    // converts exactly as it does without the bound.
    let dir = temp_dir("nested-closure-moderate");
    let src = fixture(&dir, "nested10.rs", &nested_dup_closures_src(10));
    let out = summarize_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "status={:?}\nstdout={stdout}",
        out.status
    );
    assert!(
        stdout.contains("::main: blocks=") && stdout.contains("[converted]"),
        "main must convert:\n{stdout}"
    );
    assert!(
        !stdout.contains("conversion size limit"),
        "no type may trip the size limit:\n{stdout}"
    );
}
