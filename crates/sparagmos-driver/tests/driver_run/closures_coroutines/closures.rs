// Basic closure forms: captures, FnMut state, closures stored in structs, FnOnce moves, returned and nested closures.

const CLOSURE_CAPTURE_SRC: &str = r#"
fn main() {
    let base = 40;
    let add = |x: i32| x + base;
    println!("{}", add(2));
}
"#;

const CLOSURE_NO_CAPTURE_SRC: &str = r#"
fn main() {
    let twice = |x: i32| x * 2;
    println!("{}", twice(21));
}
"#;

const CLOSURE_FNMUT_SRC: &str = r#"
fn main() {
    let mut count = 0;
    let mut inc = || { count += 1; count };
    println!("{}", inc());
    println!("{}", inc());
}
"#;

const CLOSURE_IN_STRUCT_SRC: &str = r#"
struct Holder<F: Fn(i32) -> i32> { f: F }
fn main() {
    let k = 5;
    let h = Holder { f: |x: i32| x + k };
    println!("{}", (h.f)(7));
}
"#;

const CLOSURE_FNONCE_MOVE_OUT_SRC: &str = r#"
fn main() {
    let s = String::from("abc");
    let f = move || -> String { s };
    let out = f();
    println!("{}", out);
}
"#;

const CLOSURE_FNONCE_IN_STRUCT_SRC: &str = r#"
struct Once<F: FnOnce() -> String> { f: F }
fn main() {
    let s = String::from("xyz");
    let o = Once { f: move || s };
    let out = (o.f)();
    println!("{}", out);
}
"#;

const CLOSURE_RETURNED_SRC: &str = r#"
fn make_adder(n: i32) -> impl Fn(i32) -> i32 {
    move |x| x + n
}
fn main() {
    let add = make_adder(30);
    println!("{}", add(12));
}
"#;

const CLOSURE_NESTED_SRC: &str = r#"
fn main() {
    let outer = |x: i32| {
        let inner = |y: i32| y + 1;
        inner(x) * 2
    };
    println!("{}", outer(10));
}
"#;

#[test]
fn run_mode_closure_with_capture() {
    let dir = temp_dir("closure-capture");
    let src = fixture(&dir, "closure_capture.rs", CLOSURE_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure capturing a local must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_closure_without_capture() {
    let dir = temp_dir("closure-no-capture");
    let src = fixture(&dir, "closure_no_capture.rs", CLOSURE_NO_CAPTURE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a captureless closure must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_fnmut_closure_keeps_state() {
    let dir = temp_dir("closure-fnmut");
    let src = fixture(&dir, "closure_fnmut.rs", CLOSURE_FNMUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnMut closure must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1\n2\n",
        "the captured counter must persist across calls"
    );
}

#[test]
fn run_mode_closure_stored_in_struct() {
    let dir = temp_dir("closure-in-struct");
    let src = fixture(&dir, "closure_in_struct.rs", CLOSURE_IN_STRUCT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure stored in a struct field must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "12\n");
}

#[test]
fn run_mode_fnonce_closure_moves_capture_out() {
    let dir = temp_dir("closure-fnonce-move-out");
    let src = fixture(&dir, "closure_fnonce_move_out.rs", CLOSURE_FNONCE_MOVE_OUT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnOnce closure moving its capture out must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "abc\n");
}

#[test]
fn run_mode_fnonce_closure_in_struct_field() {
    let dir = temp_dir("closure-fnonce-in-struct");
    let src = fixture(
        &dir,
        "closure_fnonce_in_struct.rs",
        CLOSURE_FNONCE_IN_STRUCT_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an FnOnce closure stored in a struct field must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "xyz\n");
}

#[test]
fn run_mode_closure_returned_from_function() {
    let dir = temp_dir("closure-returned");
    let src = fixture(&dir, "closure_returned.rs", CLOSURE_RETURNED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a closure returned from a function must be callable; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}

#[test]
fn run_mode_nested_closures() {
    let dir = temp_dir("closure-nested");
    let src = fixture(&dir, "closure_nested.rs", CLOSURE_NESTED_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested closures must run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "22\n");
}
