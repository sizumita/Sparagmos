//! Panic containment and speculative-query discipline: the
//! driver must never panic (exit 101) on guest input, and a
//! speculative compiler query that the reference compilation
//! never poses must not fail a program rustc accepts. The
//! boundary has a second side: when the compiler's own bug
//! machinery fires inside work the verdict itself decides —
//! materializing or forcing a body the collector collects —
//! containment must not downgrade it to a clean failure
//! either; the verdict is the compiler crash (exit 101) the
//! reference compilation reports. The fixtures at the end of
//! this file pin both sides of that boundary (Issue #509).
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

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits.
/// Tests spawn the driver per case and used to leave every
/// fixture directory behind (#537); the registry keeps the
/// per-test `temp_dir` shape and sweeps at exit instead.
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

fn driver() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sparagmos-driver"))
}

fn write_fixture(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-containment-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    register_scratch_dir(&dir);
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

/// The shape of tests/crashes/121575.rs: materializing the
/// body of the recursive `PrimRec` instance makes the
/// compiler's own session raise its pinned ICE (normalizing
/// a `ConstKind::Expr` is unimplemented in the compiler).
/// The same run first contains an unrelated compiler panic
/// from a speculative layout probe (the `std::fmt::Write`
/// dummy-binder panic); the crash verdict must name the
/// pinned ICE, not the probe panic, and must not be
/// downgraded to the materialization-diagnostics failure
/// the run used to end with.
const CRASH_MATERIALIZE_RS: &str = r##"
#![feature(generic_const_exprs)]

use std::array;

trait PrimRec<const N: usize, const O: usize> {
    fn eval(&self, x: [usize; N]) -> [usize; O];
}

struct Zero;

impl<const N: usize> PrimRec<N, 1> for Zero {
    fn eval(&self, _: [usize; N]) -> [usize; 1] {
        [0]
    }
}

struct Const(usize);

impl<const N: usize> PrimRec<N, 1> for Const {
    fn eval(&self, _: [usize; N]) -> [usize; 1] {
        [self.0]
    }
}

struct S;

impl PrimRec<1, 1> for S {
    fn eval(&self, x: [usize; 1]) -> [usize; 1] {
        [x[0] + 1]
    }
}

struct Proj<const I: usize>;

impl<const N: usize, const I: usize> PrimRec<N, 1> for Proj<I> {
    fn eval(&self, x: [usize; N]) -> [usize; 1] {
        [x[I]]
    }
}

struct Merge<const N: usize, const O1: usize, const O2: usize, A: PrimRec<N, O1>, B: PrimRec<N, O2>>(
    A,
    B,
);

fn concat<const M: usize, const N: usize>(a: [usize; M], b: [usize; N]) -> [usize; M + N] {
    array::from_fn(|i| if i < M { a[i] } else { b[i - M] })
}

impl<const N: usize, const O1: usize, const O2: usize, A: PrimRec<N, O1>, B: PrimRec<N, O2>>
    PrimRec<N, { O1 + O2 }> for Merge<N, O1, O2, A, B>
{
    fn eval(&self, x: [usize; N]) -> [usize; O1 + O2] {
        concat(self.0.eval(x), self.1.eval(x))
    }
}

struct Compose<const N: usize, const I: usize, const O: usize, A: PrimRec<N, I>, B: PrimRec<I, O>>(
    A,
    B,
);

impl<const N: usize, const I: usize, const O: usize, A: PrimRec<N, I>, B: PrimRec<I, O>>
    PrimRec<N, O> for Compose<N, I, O, A, B>
{
    fn eval(&self, x: [usize; N]) -> [usize; O] {
        self.1.eval(self.0.eval(x))
    }
}

struct Rec<const N: usize, const O: usize, Base: PrimRec<N, O>, F: PrimRec<{ O + (N + 1) }, O>>(
    Base,
    F,
);

fn tail<const N: usize>(x: [usize; N + 1]) -> [usize; N] {
    array::from_fn(|i| x[i + 1])
}

fn cons<const N: usize>(x: usize, xs: [usize; N]) -> [usize; N + 1] {
    array::from_fn(|i| if i == 0 { x } else { xs[i - 1] })
}

impl<const N: usize, const O: usize, Base: PrimRec<N, O>, F: PrimRec<{ O + (N + 1) }, O>>
    PrimRec<{ N + 1 }, O> for Rec<N, O, Base, F>
{
    fn eval(&self, x: [usize; N + 1]) -> [usize; O] {
        match (x[0], tail(x)) {
            (0, x) => self.0.eval(x),
            (y, x) => {
                let xy = cons(y - 1, x);
                let input = concat(self.eval(xy), xy);
                self.1.eval(input)
            }
        }
    }
}

fn main() {
    let one = Compose(Zero, S);
    dbg!(one.eval([]));
    let add: Rec<1, 1, Proj<0>, Compose<3, 1, 1, Proj<0>, S>> =
        Rec(Proj::<0>, Compose(Proj::<0>, S));
    dbg!(add.eval([3, 2]));
}
"##;

#[test]
fn compiler_bug_in_collected_materialization_is_a_crash() {
    let (code, _stdout, stderr) = convert("crash_materialize", CRASH_MATERIALIZE_RS, &[], &[]);
    assert_eq!(code, Some(101), "stderr: {stderr}");
    assert!(
        stderr.contains("mir/consts.rs:338"),
        "the crash report must name the pinned ICE: {stderr}"
    );
    assert!(
        !stderr.contains("ty/predicate.rs"),
        "the earlier probe panic must not be the reported crash: {stderr}"
    );
}

/// The shape of tests/crashes/138008.rs: the verdict's
/// per-instance force asks the compiler for the root
/// function's ABI, whose layout query raises the compiler's
/// pinned ICE for the `repr(simd)` type. The force is the
/// collector's own, so the verdict is a crash, not the bare
/// compilation failure the run used to end with.
const CRASH_FN_ABI_RS: &str = r##"
#![feature(repr_simd)]
const C: usize = 16;

#[repr(simd)]
pub struct Foo([u8; C]);
pub unsafe fn foo(a: Foo) {}
"##;

#[test]
fn compiler_bug_in_collector_fn_abi_force_is_a_crash() {
    let (code, _stdout, stderr) = convert(
        "crash_fn_abi",
        CRASH_FN_ABI_RS,
        &["--crate-type=lib", "-Copt-level=0"],
        &[],
    );
    assert_eq!(code, Some(101), "stderr: {stderr}");
    assert!(
        stderr.contains("ty/layout.rs:1411"),
        "the crash report must name the pinned ICE: {stderr}"
    );
}

/// The shape of tests/crashes/139120.rs: the verdict's walk
/// mentions an unsize cast to `&dyn FooFn`, and resolving
/// the vtable's instance for the function-pointer impl
/// raises the compiler's pinned ICE. The vtable allocation
/// is a force of the walk itself, so the verdict is a
/// crash, not the bare compilation failure the run used to
/// end with.
const CRASH_VTABLE_RS: &str = r##"
pub trait Foo {
    type Bar<'a>;
}

pub struct FooImpl {}

impl Foo for FooImpl {
    type Bar<'a> = ();
}

pub trait FooFn {
    fn bar(&self);
}

impl<T: Foo> FooFn for fn(T, T::Bar<'_>) {
    fn bar(&self) {}
}

fn foo<T: Foo>(f: fn(T, T::Bar<'_>)) {
    let _: &dyn FooFn = &f;
}

fn main() {
    foo(|_: FooImpl, _| {});
}
"##;

#[test]
fn compiler_bug_in_collector_vtable_force_is_a_crash() {
    let (code, _stdout, stderr) = convert("crash_vtable", CRASH_VTABLE_RS, &[], &[]);
    assert_eq!(code, Some(101), "stderr: {stderr}");
    assert!(
        stderr.contains("ty/instance.rs:640"),
        "the crash report must name the pinned ICE: {stderr}"
    );
}

/// The shape of tests/crashes/54888.rs: the run contains
/// the `std::fmt::Write` dummy-binder compiler panic inside
/// a speculative layout probe and converts successfully.
/// The panic is a string-payload compiler panic on a probe
/// the reference compilation never runs; it must keep its
/// verdict (a successful conversion).
const PROBE_FMT_RS: &str = r##"
#![feature(unsize, coerce_unsized)]

use std::{
    ops::CoerceUnsized,
    marker::Unsize,
};

#[repr(C)]
struct Ptr<T: ?Sized>(Box<T>);

impl<T: ?Sized, U: ?Sized> CoerceUnsized<Ptr<U>> for Ptr<T>
where
    T: Unsize<U>,
{}


fn main() {
    let foo = Ptr(Box::new(5)) as Ptr<dyn std::any::Any>;
}
"##;

#[test]
fn speculative_probe_panics_keep_their_verdict() {
    let (code, _stdout, stderr) = convert("probe_fmt", PROBE_FMT_RS, &[], &[]);
    assert_eq!(code, Some(0), "stderr: {stderr}");
}

/// The shape of tests/crashes/154367.rs: the run contains
/// compiler-location panics with string payloads — the
/// bridge's unimplemented unsafe-binder conversion on a
/// speculative screen, and the symbol mangler's
/// `unimplemented!()` for the unsafe-binder type inside the
/// verdict pass's own duplicate-symbol check. Neither is
/// the compiler's bug machinery, and the program converts;
/// both must keep that verdict.
const PROBE_MANGLING_RS: &str = r##"
#![feature(unsafe_binders)]

fn main() { panic::<unsafe<'a> &'a ()>(); }
fn panic<T>() { panic!() }
"##;

#[test]
fn compiler_site_string_panics_keep_their_verdict() {
    let (code, _stdout, stderr) =
        convert("probe_mangling", PROBE_MANGLING_RS, &["-Copt-level=0"], &[]);
    assert_eq!(code, Some(0), "stderr: {stderr}");
}

/// The shape of
/// tests/ui/const-generics/generic_const_exprs/poly-const-uneval-ice-106423.rs
/// (Issue #510): generic const expressions appear in types
/// (`[(); N * 2 + 1]: Sized`) and in an associated constant
/// built by repeating another constant. Stable MIR's body
/// builder eagerly evaluates such constants when it
/// materializes a body, and `Const::eval` dies on a
/// `ty::ConstKind::Expr` with the compiler's bug machinery
/// (mir/consts.rs:338) -- an evaluation the native pipeline
/// never performs on a type-level const, for a program rustc
/// accepts. The firing is contained inside the fetch, and
/// neither its ICE diagnostic nor its bug record may decide
/// the verdict; the verdict pass's own symbol naming of an
/// item whose arguments mention the same const expression is
/// contained the same way. The program converts.
const POLY_CONST_RS: &str = r##"
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(unused)]

use core::mem::MaybeUninit;

pub struct Arr<T, const N: usize> {
    v: [MaybeUninit<T>; N],
}

impl<T, const N: usize> Arr<T, N> {
    const ELEM: MaybeUninit<T> = MaybeUninit::uninit();
    const INIT: [MaybeUninit<T>; N] = [Self::ELEM; N];

    fn new() -> Self {
        Arr { v: Self::INIT }
    }
}

pub struct BaFormatFilter<const N: usize> {}

pub enum DigitalFilter<const N: usize>
where
    [(); N * 2 + 1]: Sized,
    [(); N * 2]: Sized,
{
    Ba(BaFormatFilter<{ N * 2 + 1 }>),
}

pub fn iirfilter_st_copy<const N: usize, const M: usize>(_: [f32; M]) -> DigitalFilter<N>
where
    [(); N * 2 + 1]: Sized,
    [(); N * 2]: Sized,
{
    let zpk = zpk2tf_st(&Arr::<f32, { N * 2 }>::new(), &Arr::<f32, { N * 2 }>::new());
    DigitalFilter::Ba(zpk)
}

pub fn zpk2tf_st<const N: usize>(_z: &Arr<f32, N>, _p: &Arr<f32, N>) -> BaFormatFilter<{ N + 1 }>
where
    [(); N + 1]: Sized,
{
    BaFormatFilter {}
}

fn main() {
    iirfilter_st_copy::<4, 2>([10., 50.]);
}
"##;

#[test]
fn const_expr_normalize_in_materialization_keeps_its_verdict() {
    let (code, stdout, stderr) = convert("poly_const", POLY_CONST_RS, &[], &[]);
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert!(
        !stderr.contains("compilation failed"),
        "a contained const-expr firing must not fail the compilation: {stderr}"
    );
    assert!(
        stdout.contains("fn poly_const::main:"),
        "the target crate converts: {stdout}"
    );
}
