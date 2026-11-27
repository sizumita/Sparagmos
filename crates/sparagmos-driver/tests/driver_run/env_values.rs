use super::{fixture, run_driver, run_driver_with_aux, temp_dir};

// ----- builtin-owned definitions demanded as values --------------------------
//
// A direct call to a builtin-owned definition is baked into a
// `CallBuiltin` at conversion, but the same definition demanded
// *as a value* (reified to a function pointer, or handed to a
// combinator like `unwrap_or_else`) has no body to collect and
// reaches the interpreter as a bare name. The runtime name
// fallback serves the host-state models from that name: the
// process environment, the clocks, the current thread and the
// standard streams read only the live state their `CallBuiltin`
// execution reads.

const TEMP_DIR_VALUE_SRC: &str = r#"
fn main() {
    let p = None::<std::path::PathBuf>.unwrap_or_else(std::env::temp_dir);
    println!("temp={}", p.display());
    let f: fn() -> std::path::PathBuf = std::env::temp_dir;
    assert_eq!(f(), p);
}
"#;

#[test]
fn run_mode_env_temp_dir_demanded_as_value() {
    let dir = temp_dir("env-temp-dir-value");
    let src = fixture(&dir, "tempdir_value.rs", TEMP_DIR_VALUE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "temp_dir as a value must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("temp={}\n", std::env::temp_dir().display()),
        "the guest sees the host temp dir, as a direct call would print"
    );
}

const CURRENT_DIR_FN_PTR_SRC: &str = r#"
fn main() {
    let f: fn() -> std::io::Result<std::path::PathBuf> = std::env::current_dir;
    println!("cwd={}", f().unwrap().display());
}
"#;

#[test]
fn run_mode_env_current_dir_as_fn_pointer() {
    let dir = temp_dir("env-current-dir-value");
    let src = fixture(&dir, "currentdir_value.rs", CURRENT_DIR_FN_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "current_dir as a function pointer must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("cwd={}\n", std::env::current_dir().unwrap().display()),
        "the guest sees the host working directory"
    );
}

const INSTANT_NOW_FN_PTR_SRC: &str = r#"
fn main() {
    let f: fn() -> std::time::Instant = std::time::Instant::now;
    let t = f();
    println!("elapsed={}", t.elapsed().as_nanos() >= 0);
    let g: fn() -> u32 = std::process::id;
    println!("pid={}", g() > 0);
}
"#;

#[test]
fn run_mode_clock_and_process_queries_as_fn_pointers() {
    let dir = temp_dir("clock-process-value");
    let src = fixture(&dir, "clock_process_value.rs", INSTANT_NOW_FN_PTR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Instant::now / process::id as function pointers must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "elapsed=true\npid=true\n"
    );
}

const AUX_TEMP_DIR_LIB_SRC: &str = r#"
pub fn base_dir() -> std::path::PathBuf {
    // The shape `run_make_support::tempfile` uses: the fallback
    // for a missing override is the builtin-owned function
    // itself, handed over as a value.
    std::env::var_os("SPARAGMOS_AUXGAP_NO_SUCH_VAR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}
"#;

const AUX_TEMP_DIR_MAIN_SRC: &str = r#"
extern crate auxlib;
fn main() {
    println!("base={}", auxlib::base_dir().display());
}
"#;

#[test]
fn run_mode_aux_crate_temp_dir_demanded_as_value() {
    // The demand originates inside the auxiliary crate's
    // converted body, not the root crate's: builtin provision
    // does not depend on which crate's body issues it.
    let dir = temp_dir("aux-temp-dir-value");
    let aux = fixture(&dir, "auxlib.rs", AUX_TEMP_DIR_LIB_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_TEMP_DIR_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an aux body's value demand for temp_dir must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("base={}\n", std::env::temp_dir().display())
    );
}
