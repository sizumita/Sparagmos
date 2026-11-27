use super::{Command, fixture, run_driver, temp_dir};

// Validity assertions (`mem::zeroed`, `mem::uninitialized`,
// `MaybeUninit::assume_init`): conversion bakes the compiler's
// own validity verdict for each call site, and a rejected site
// aborts the run with the `panic_nounwind` report code
// generation emits natively (exit 134, the message naming the
// type, the non-unwinding line). Accepted sites run on.

fn run_abort_case(tag: &str, src: &str, message: &str) {
    let dir = temp_dir(tag);
    let src = fixture(&dir, "case.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "case {tag} must abort (134); stderr={stderr}"
    );
    assert!(
        stderr.contains(message),
        "case {tag} stderr must carry the validity message {message:?}: {stderr}"
    );
    assert!(
        stderr.contains("thread caused non-unwinding panic. aborting."),
        "case {tag} stderr must carry the non-unwinding line: {stderr}"
    );
}

#[test]
fn zeroed_reference_aborts_with_the_std_message() {
    run_abort_case(
        "zeroed-ref",
        r#"
fn main() {
    let _v = unsafe { std::mem::zeroed::<&i32>() };
}
"#,
        "attempted to zero-initialize type `&i32`, which is invalid",
    );
}

#[test]
fn uninitialized_reference_aborts_with_the_std_message() {
    run_abort_case(
        "uninit-ref",
        r#"
fn main() {
    let _v = unsafe { std::mem::uninitialized::<&i32>() };
}
"#,
        "attempted to leave type `&i32` uninitialized, which is invalid",
    );
}

#[test]
fn assume_init_of_uninhabited_aborts_with_the_std_message() {
    run_abort_case(
        "assume-init-never",
        r#"
fn main() {
    let _v: ! = unsafe { std::mem::MaybeUninit::<!>::uninit().assume_init() };
}
"#,
        "attempted to instantiate uninhabited type `!`",
    );
}

#[test]
fn zeroed_enum_without_a_zero_discriminant_aborts() {
    run_abort_case(
        "zeroed-no-null",
        r#"
#[repr(i32)]
enum NoNull {
    A(i32) = 1,
    B(i32) = 2,
}
fn main() {
    let _v = unsafe { std::mem::zeroed::<NoNull>() };
}
"#,
        "attempted to zero-initialize type `NoNull`, which is invalid",
    );
}

#[test]
fn valid_zeroed_and_uninitialized_forms_run() {
    let dir = temp_dir("validity-healthy");
    let src = fixture(
        &dir,
        "case.rs",
        r#"
use std::mem::{self, MaybeUninit};
struct Plain {
    a: u8,
    b: u32,
}
struct ZeroSized;
fn main() {
    unsafe {
        assert_eq!(mem::zeroed::<bool>(), false);
        assert_eq!(mem::zeroed::<[u64; 2]>(), [0, 0]);
        let p = mem::zeroed::<Plain>();
        assert_eq!(p.a, 0);
        assert_eq!(p.b, 0);
        assert!(mem::zeroed::<Option<&i32>>().is_none());
        let _e = mem::zeroed::<[!; 0]>();
        let _m = mem::uninitialized::<MaybeUninit<bool>>();
        let _a = mem::uninitialized::<[!; 0]>();
        let _u: () = mem::uninitialized::<()>();
        let _z = mem::uninitialized::<ZeroSized>();
        // A function pointer left uninitialized in the lax
        // check: the 0x01 fill denotes an address-only
        // pointer value, and the value is never called.
        let _f = mem::uninitialized::<fn()>();
    }
    println!("healthy");
}
"#,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "healthy forms must run to completion; stderr={stderr}"
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("healthy"),
        "healthy forms must reach the end of main"
    );
}

#[test]
fn uninitialized_int_aborts_only_with_strict_init_checks() {
    let dir = temp_dir("validity-strict");
    let src = fixture(
        &dir,
        "case.rs",
        r#"
fn main() {
    let _v = unsafe { std::mem::uninitialized::<i32>() };
    println!("survived");
}
"#,
    );
    // The lax check (the default session) accepts the 0x01
    // pattern for a plain integer.
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "lax mode must accept uninitialized::<i32>(); stderr={stderr}"
    );
    // The strict session's own check rejects it.
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .args(["--rustc-arg", "-Zstrict-init-checks"])
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(134),
        "strict mode must abort uninitialized::<i32>(); stderr={stderr}"
    );
    assert!(
        stderr.contains("attempted to leave type `i32` uninitialized, which is invalid"),
        "strict mode stderr must carry the validity message: {stderr}"
    );
}
