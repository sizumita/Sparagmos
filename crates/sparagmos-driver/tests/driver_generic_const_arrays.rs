//! Generic associated constants whose bodies build arrays from
//! the definition's own parameters (Issue #510): an array length
//! or repeat count written from a const parameter has no target
//! value in the generic definition, so converting the definition
//! body fails with `Ty::Array (length not evaluable)` /
//! `Rvalue::Repeat (count not evaluable)` -- and a type mentioning
//! a generic const expression over the parameters cannot be
//! converted by Stable MIR at all. Those are failures on the
//! definition's own parameters, the same shape as a
//! `ConstantKind::Param` mention: the constant has no monomorphic
//! instance as an item, and monomorphized bodies that use it carry
//! its evaluated value, so the definition is accounted
//! `NOT DEMANDED` (generic constant definition), not skipped.
//! A monomorphically demanded constant of the same shape still
//! converts and executes through its evaluated value.

use std::path::PathBuf;
use std::process::Command;

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits.
/// Tests spawn the driver per case and used to leave every
/// fixture directory behind (#537); the registry keeps the
/// per-test `temp_dir` shape and sweeps at exit instead.
static SCRATCH_DIRS: std::sync::Mutex<Vec<std::path::PathBuf>> = std::sync::Mutex::new(Vec::new());

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
        "sparagmos-driver-generic-const-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    register_scratch_dir(&dir);
    let path = dir.join(format!("{name}.rs"));
    std::fs::write(&path, src).expect("write fixture");
    path
}

/// Run the driver in convert mode over a fixture (the fixtures
/// use unstable features, so the bootstrap env the driver's own
/// build uses is forwarded). Returns (stdout, status) with
/// stdout's whitespace collapsed, so report lines can be matched
/// without depending on their wrapping.
fn convert(name: &str, src: &str) -> (String, std::process::ExitStatus) {
    let fixture = write_fixture(name, src);
    let out = Command::new(driver())
        .arg(&fixture)
        .args(["--edition", "2021"])
        .env("RUSTC_BOOTSTRAP", "1")
        .env_remove("SPARAGMOS_DUMP")
        .output()
        .expect("spawn driver");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    (
        stdout.split_whitespace().collect::<Vec<_>>().join(" "),
        out.status,
    )
}

/// The shape of tests/ui/const-generics/generic_const_exprs/
/// poly-const-uneval-ice-106423.rs: a generic associated
/// constant initialised by repeating an element `N` times,
/// where `N` is the impl's own const parameter.
const ARR_RS: &str = "\
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(unused)]
use std::mem::MaybeUninit;
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
fn main() {}
";

#[test]
fn generic_repeat_constant_is_not_demanded() {
    let (stdout, status) = convert("arr_init", ARR_RS);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains("Arr::<T, N>::INIT: NOT DEMANDED (generic constant definition"),
        "INIT should be accounted not demanded: {stdout}"
    );
    let target = stdout
        .split("target-crate converted:")
        .nth(1)
        .expect("target-crate report line");
    let target_skipped = target
        .split("skipped:")
        .nth(1)
        .expect("target-crate skipped count");
    assert!(
        target_skipped.trim_start().starts_with('0'),
        "no target-crate function should be skipped: {stdout}"
    );
    assert!(
        !stdout.contains("INIT: SKIPPED"),
        "INIT must not stay skipped: {stdout}"
    );
}

/// The shape of tests/ui/const-generics/issues/issue-105821.rs:
/// an associated constant array sized by `A.len()` for a
/// `&'static [u8]` const parameter, and a second constant whose
/// type is built from a generic const expression over it.
const CATTER_RS: &str = "\
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
fn generic_const_expr_constants_are_not_demanded() {
    let (stdout, status) = convert("catter", CATTER_RS);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains("Catter::<A>::ZEROS: NOT DEMANDED (generic constant definition"),
        "ZEROS should be accounted not demanded: {stdout}"
    );
    assert!(
        stdout.contains("Catter::<A>::R: NOT DEMANDED (generic constant definition"),
        "R should be accounted not demanded: {stdout}"
    );
    assert!(
        !stdout.contains("ZEROS: SKIPPED") && !stdout.contains("R: SKIPPED"),
        "neither constant must stay skipped: {stdout}"
    );
}

/// The same array-repeat constant, demanded monomorphically:
/// the using body carries the evaluated value, so the program
/// converts and runs.
const ARR_RUN_RS: &str = "\
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
use std::mem::MaybeUninit;
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
fn main() {
    let a = Arr::<u8, 3>::new();
    println!(\"{}\", a.v.len());
}
";

#[test]
fn monomorphically_demanded_repeat_constant_executes() {
    let fixture = write_fixture("arr_run", ARR_RUN_RS);
    let out = Command::new(driver())
        .args(["run", "--edition", "2021"])
        .arg(&fixture)
        .env("RUSTC_BOOTSTRAP", "1")
        .env_remove("SPARAGMOS_DUMP")
        .output()
        .expect("spawn driver");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "run failed: {stdout}");
    assert_eq!(stdout, "3\n");
}
