//! Instance-constant conversion residuals (census B-6):
//! constant definitions whose own body fetch dies evaluating
//! an instance constant (`Failed to evaluate instance
//! constant` -- the evaluation is `ErrorHandled::TooGeneric`)
//! because the initializer resolves an associated constant
//! through an assumed trait bound. Such a constant is a
//! generic definition in effect: its value exists only under
//! a monomorphization whose context provides the bound, the
//! monomorphization collector never evaluates it as a
//! definition, and monomorphized bodies that use it carry its
//! evaluated value -- so it is accounted `NOT DEMANDED`
//! (generic constant definition), the accounting the
//! parameter-shaped generic constants already get, not
//! skipped. A constant whose initializer is concrete still
//! converts, and a function body failing the same way stays
//! skipped (its execution would need the constant).

use std::path::PathBuf;
use std::process::Command;

/// Fixture directories this test process created under the
/// system temp directory, removed when the process exits.
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
        "sparagmos-driver-instance-const-{}-{name}",
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

fn assert_no_target_skips(stdout: &str) {
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
}

/// The shape of tests/ui/associated-consts/
/// issue-88599-ref-self.rs: an associated constant of a
/// concrete impl whose initializer reads the same constant
/// through `&Self`, a resolution the impl's own where clause
/// assumes (`&'a Self: First`). Evaluating the definition is
/// too generic; nothing monomorphic can demand it.
const REF_SELF_RS: &str = "\
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
pub trait First {
    const CONST: usize;
}
pub trait Second {}
impl<'a> First for dyn Second
where
    &'a Self: First,
{
    const CONST: usize = <&Self>::CONST;
}
fn main() {}
";

#[test]
fn assoc_const_through_ref_self_is_not_demanded() {
    let (stdout, status) = convert("ref_self", REF_SELF_RS);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains(
            "<(dyn ref_self::Second + 'static) as ref_self::First>::CONST: NOT DEMANDED \
             (generic constant definition (body mentions an instance constant"
        ),
        "CONST should be accounted not demanded: {stdout}"
    );
    assert!(
        !stdout.contains("CONST: SKIPPED"),
        "CONST must not stay skipped: {stdout}"
    );
    assert_no_target_skips(&stdout);
}

/// The shape of tests/ui/generic-const-items/
/// hkl_where_bounds.rs: a generic const item whose
/// initializer resolves an associated constant through the
/// item's own (nonsense) where bound. The bound prevents the
/// constant from being evaluated at all as a definition.
const WHERE_BOUNDS_RS: &str = "\
#![feature(generic_const_items)]
#![allow(incomplete_features)]
#![allow(dead_code)]
trait Trait {
    const ASSOC: u32;
}
const ASSOC: u32 = <&'static ()>::ASSOC where for<'a> &'a (): Trait;
fn main() {}
";

#[test]
fn generic_const_item_where_bound_is_not_demanded() {
    let (stdout, status) = convert("where_bounds", WHERE_BOUNDS_RS);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        stdout.contains(
            "where_bounds::ASSOC: NOT DEMANDED \
             (generic constant definition (body mentions an instance constant"
        ),
        "ASSOC should be accounted not demanded: {stdout}"
    );
    assert!(
        !stdout.contains("ASSOC: SKIPPED"),
        "ASSOC must not stay skipped: {stdout}"
    );
    assert_no_target_skips(&stdout);
}

/// A constant whose initializer resolves an associated
/// constant of a concrete impl: the evaluation is not too
/// generic, so the constant converts (it is not re-accounted
/// as a generic definition).
const CONCRETE_RS: &str = "\
#![allow(dead_code)]
trait Trait {
    const ASSOC: u32;
}
impl Trait for u8 {
    const ASSOC: u32 = 7;
}
const ASSOC: u32 = <u8 as Trait>::ASSOC;
fn main() {}
";

#[test]
fn concrete_assoc_const_initializer_still_converts() {
    let (stdout, status) = convert("concrete", CONCRETE_RS);
    assert!(status.success(), "driver failed: {stdout}");
    assert!(
        !stdout.contains("concrete::ASSOC: NOT DEMANDED"),
        "a concrete constant must not be accounted not demanded: {stdout}"
    );
    assert!(
        !stdout.contains("concrete::ASSOC: SKIPPED"),
        "a concrete constant must not be skipped: {stdout}"
    );
    assert_no_target_skips(&stdout);
}
