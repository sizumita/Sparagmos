//! Panic containment and speculative-query discipline: the
//! driver must never panic (exit 101) on guest input, and a
//! speculative compiler query that the reference compilation
//! never poses must not fail a program rustc accepts.
//!
//! The fixtures are the shapes of the compile-check records of
//! Issue #510: screens asking a still-generic instance for its
//! type ("needs further instantiation"), screens walking a
//! body whose types reach rustc_public's unimplemented
//! unsafe-binder / const-expression conversions, an ABI query
//! for an instance fabricated without arguments, constants
//! whose allocated bytes are not valid values of their type
//! (transmuted bool/char, uninitialised integers), a legacy
//! symbol-mangling query for an instance mentioning an
//! anonymous constant, and an auxiliary crate selecting its
//! definitions through the revision `--cfg`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn driver() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sparagmos-driver"))
}

fn write_fixture(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-containment-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let path = dir.join(format!("{name}.rs"));
    std::fs::write(&path, src).expect("write fixture");
    path
}

/// Convert `fixture` (convert mode: no subcommand), returning
/// (exit code, stdout, stderr). Mirrors the subset runner's
/// invocations, which carry RUSTC_BOOTSTRAP=1 for the embedded
/// nightly rustc.
fn convert(name: &str, src: &str, rustc_args: &[&str], extra: &[&str]) -> (Option<i32>, String, String) {
    let fixture = write_fixture(name, src);
    convert_path(&fixture, rustc_args, extra)
}

fn convert_path(fixture: &Path, rustc_args: &[&str], extra: &[&str]) -> (Option<i32>, String, String) {
    let mut cmd = Command::new(driver());
    cmd.arg(fixture).arg("--edition").arg("2021");
    cmd.env("RUSTC_BOOTSTRAP", "1");
    cmd.env_remove("SPARAGMOS_DUMP");
    for a in rustc_args {
        cmd.arg("--rustc-arg").arg(a);
    }
    for a in extra {
        cmd.arg(a);
    }
    let out = cmd.output().expect("spawn driver");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The screen over an instance asks for the instance's type;
/// for a still-generic instance (here the `variant_count`
/// callee behind a generic associated constant) that question
/// panics inside rustc_public with "needs further
/// instantiation". The screen must have no opinion instead of
/// crashing the driver (the shape of
/// tests/ui/consts/issue-79137-monomorphic.rs).
const VARIANT_COUNT_RS: &str = "\
#![feature(variant_count)]

pub struct GetVariantCount<T>(T);

impl<T> GetVariantCount<T> {
    pub const VALUE: usize = std::mem::variant_count::<T>();
}

const fn check_variant_count<T>() -> bool {
    matches!(GetVariantCount::<Option<T>>::VALUE, GetVariantCount::<Option<()>>::VALUE)
}

fn main() {
    assert!(check_variant_count::<()>());
}
";

#[test]
fn generic_instance_screen_does_not_crash() {
    let (code, _stdout, stderr) = convert("variant_count", VARIANT_COUNT_RS, &[], &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
}

/// The same screen question for a generic `MaybeUninit::uninit`
/// array repeat (the shape of
/// tests/codegen-llvm/issues/issue-96274.rs).
const MAYBE_UNINIT_RS: &str = "\
#![crate_type = \"lib\"]

use std::mem::MaybeUninit;

pub fn maybe_uninit_const<T>() -> [MaybeUninit<T>; 8192] {
    [const { MaybeUninit::uninit() }; 8192]
}
";

#[test]
fn generic_uninit_screen_does_not_crash() {
    let (code, _stdout, stderr) = convert("maybe_uninit", MAYBE_UNINIT_RS, &[], &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
}

/// A body whose local's type is an unsafe binder: walking the
/// body's types for the projection screen reaches
/// rustc_public's unimplemented `ty::UnsafeBinder` conversion.
/// The screen must have no opinion, and the body's own
/// conversion then reports the named unsupported reason
/// instead of the driver crashing (the shape of
/// tests/ui/unsafe-binders/simple.rs).
const UNSAFE_BINDER_RS: &str = "\
#![feature(unsafe_binders)]

fn main() {
    let _x: unsafe<'a> &'a ();
}
";

#[test]
fn unsafe_binder_body_skips_with_named_reason() {
    let (code, stdout, stderr) = convert("unsafe_binder", UNSAFE_BINDER_RS, &[], &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert!(
        stdout.contains("body mentions an unsafe binder"),
        "expected the named unsafe-binder skip reason, got: {stdout}"
    );
}

/// A generic const expression in an associated constant's type
/// (the shape of tests/ui/const-generics/issues/
/// issue-105821.rs): the screen walk reaches rustc_public's
/// unimplemented constant-kind conversion at ty.rs:577. The
/// driver must not crash.
const CONST_EXPR_RS: &str = "\
#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params, generic_const_exprs)]
#![allow(dead_code)]

const fn catone<const M: usize>(_a: &[u8; M]) -> [u8; M + 1]
where
    [(); M + 1]:,
{
    unimplemented!()
}

struct Catter<const A: &'static [u8]>;
impl<const A: &'static [u8]> Catter<A>
where
    [(); A.len() + 1]:,
{
    const ZEROS: &'static [u8; A.len()] = &[0_u8; A.len()];
    const R: &'static [u8] = &catone(Self::ZEROS);
}

fn main() {}
";

#[test]
fn const_expr_screen_does_not_crash() {
    let (code, _stdout, stderr) = convert("const_expr", CONST_EXPR_RS, &[], &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
}

/// An allocated constant whose bytes are not a valid value of
/// its type: a transmuted bool (0xff), a transmuted char
/// (0x7fffffff) and a union read through its unset field (all
/// bytes uninitialised). The bits are the value; the bodies
/// mentioning them convert instead of being skipped (the
/// shape of tests/mir-opt/const_prop/transmute.rs).
const TRANSMUTED_CONSTS_RS: &str = "\
#![crate_type = \"lib\"]

use std::mem::transmute;

pub unsafe fn invalid_char() -> char {
    unsafe { transmute(i32::MAX) }
}

pub unsafe fn invalid_bool() -> bool {
    unsafe { transmute(-1_i8) }
}

pub unsafe fn undef_union_as_integer() -> u32 {
    union Union32 {
        value: u32,
        unit: (),
    }
    unsafe { transmute(Union32 { unit: () }) }
}
";

#[test]
fn transmuted_scalar_consts_convert() {
    let (code, stdout, stderr) =
        convert("transmuted_consts", TRANSMUTED_CONSTS_RS, &["-O"], &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
    assert!(
        stdout.contains("skipped: 0"),
        "expected no skipped bodies, got: {stdout}"
    );
}

/// A `Drop` implementation generic over a lifetime: noting the
/// drop function's ABI fabricates the instance without generic
/// arguments, and the compiler's instantiation panics with
/// "no args were provided in instantiate". The ABI is
/// best-effort bookkeeping; the driver must not crash (the
/// shape of tests/ui/coroutine/drop-control-flow.rs).
const GENERIC_DROP_RS: &str = "\
#![feature(coroutines, stmt_expr_attributes)]
#![allow(dead_code)]

struct Ptr;
impl<'a> Drop for Ptr {
    fn drop(&mut self) {}
}

fn one_armed_if(arg: bool) {
    let _ = #[coroutine] || {
        let arr = [Ptr];
        if arg {
            drop(arr);
        }
        yield;
    };
}

fn main() {
    one_armed_if(true);
}
";

#[test]
fn lifetime_generic_drop_abi_does_not_crash() {
    let (code, _stdout, stderr) = convert("generic_drop", GENERIC_DROP_RS, &[], &[]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");
}

/// The legacy symbol mangler cannot name an instance whose
/// arguments mention an anonymous constant (it reports an
/// internal error as a session diagnostic). The symbol query
/// is speculative — native code generation never names the
/// generic instances the driver registers — so the diagnostic
/// is discounted and the instance gets no symbol, instead of
/// the report failing a compilation rustc accepts (the shape
/// of tests/ui/symbol-names/issue-76365.rs, legacy revision).
const ANON_CONST_RS: &str = "\
pub struct Bar<const F: bool>;

impl Bar<true> {
    pub fn foo() {}
}

impl<const F: bool> Bar<F> {
    pub fn bar() {}
}

fn main() {}
";

#[test]
fn legacy_mangling_anon_const_symbol_does_not_fail() {
    let (code, _stdout, stderr) = convert(
        "anon_const",
        ANON_CONST_RS,
        &[
            "-Z",
            "unstable-options",
            "-C",
            "symbol-mangling-version=legacy",
            "--crate-type=lib",
        ],
        &[],
    );
    // The mangler's report is still printed when it is
    // emitted; what the containment discounts is its effect
    // on the compilation's verdict.
    assert_eq!(code, Some(0), "driver failed: {stderr}");
}

/// An auxiliary crate that selects its definitions through the
/// revision `--cfg` (an incremental test's auxiliary crate):
/// the `--aux` spec's `;cfg=` attribute reaches the auxiliary
/// build, as compiletest's own compile arguments carry the
/// revision cfg to auxiliary builds. Without the attribute the
/// auxiliary build has no `f` at all and fails.
const AUX_CFG_MAIN_RS: &str = "\
extern crate a;

fn main() {
    assert_eq!(a::f(), 1);
}
";

const AUX_CFG_A_RS: &str = "\
#[cfg(rpass1)]
pub fn f() -> u32 {
    1
}

#[cfg(rpass2)]
pub fn f() -> u32 {
    2
}
";

#[test]
fn aux_revision_cfg_reaches_aux_build() {
    let aux = write_fixture("aux_cfg_a", AUX_CFG_A_RS);
    let spec = format!("a={};cfg=rpass1", aux.display());
    let (code, _stdout, stderr) =
        convert("aux_cfg_main", AUX_CFG_MAIN_RS, &[], &["--aux", &spec]);
    assert_eq!(code, Some(0), "driver failed: {stderr}");

    let spec = format!("a={}", aux.display());
    let (code, _stdout, stderr) =
        convert("aux_cfg_main", AUX_CFG_MAIN_RS, &[], &["--aux", &spec]);
    // Without the cfg the auxiliary crate builds empty (both
    // variants configured out) and the main compilation fails
    // with E0425 — the shape the incremental cross-crate
    // records failed with before the cfg was forwarded.
    assert_eq!(code, Some(1), "expected the cfg-less build to fail");
    assert!(
        stderr.contains("cannot find function `f`"),
        "expected an E0425 for the configured-out item, got: {stderr}"
    );
}
