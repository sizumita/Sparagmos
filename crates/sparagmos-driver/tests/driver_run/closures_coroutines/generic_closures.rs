// Closures in generic contexts: generic closure instantiations and higher-order FnOnce parameters.

#[test]
fn run_mode_generic_context_closure_runs() {
    let dir = temp_dir("closure-generic-run");
    let src = fixture(&dir, "closure_generic.rs", CLOSURE_GENERIC_CONTEXT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure inside a generic function must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "7\n");
}

const CLOSURE_GENERIC_TWO_TYPES_SRC: &str = r#"
fn twice<T: Copy + std::ops::Add<Output = T>>(v: T) -> T {
    let f = |x: T| x + x;
    f(v)
}
fn main() {
    println!("{}", twice::<i32>(21));
    println!("{}", twice::<f64>(1.5));
}
"#;

#[test]
fn run_mode_generic_closure_two_instantiations() {
    // Two monomorphic instances of one closure definition coexist
    // in the program and each call reaches its own body.
    let dir = temp_dir("closure-generic-two-types");
    let src = fixture(
        &dir,
        "closure_generic_two_types.rs",
        CLOSURE_GENERIC_TWO_TYPES_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "both instantiations of the generic-context closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n3\n");
}

const GENERIC_HOF_FNONCE_SRC: &str = r#"
fn apply<F: FnOnce() -> i32>(f: F) -> i32 { f() }
fn main() {
    let base = 40;
    let r = apply(|| base + 2);
    println!("{r}");
}
"#;

#[test]
fn run_mode_generic_hof_fnonce_with_closure() {
    // A generic higher-order function applied to a closure: the
    // `FnOnce::call_once` call inside the generic body resolves to
    // the closure instance the caller monomorphized it with.
    let dir = temp_dir("generic-hof-fnonce");
    let src = fixture(&dir, "generic_hof_fnonce.rs", GENERIC_HOF_FNONCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a generic HOF applied to a closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}
