use super::{Command, Output, Path, fixture, temp_dir};

// The debug-mode session tests are split by
// sub-feature into the files included below. `include!` keeps
// every test in this module, so test paths stay
// `debug_mode::<name>`.

include!("casts.rs");
include!("cast_chains.rs");
include!("unions.rs");
include!("whatis.rs");
include!("lldb_formats.rs");
include!("rendering.rs");
include!("session.rs");

fn debug_driver(dir: &Path, src: &Path, script: &str) -> Output {
    let script_path = dir.join("session.script");
    std::fs::write(&script_path, script).unwrap();
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("debug")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--debug-script")
        .arg(&script_path)
        .arg("--debug-dialect")
        .arg("gdb")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn debug_driver_lldb(dir: &Path, src: &Path, script: &str) -> Output {
    let script_path = dir.join("session.script");
    std::fs::write(&script_path, script).unwrap();
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("debug")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--debug-script")
        .arg(&script_path)
        .arg("--debug-dialect")
        .arg("lldb")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn debug_driver_session(
    dir: &Path,
    src: &Path,
    script: &str,
    dialect: &str,
) -> Output {
    // The subset runner's session invocation: the test's `-g`
    // compile flag and the pass-disabled MIR pipeline the
    // debugger sessions execute.
    let script_path = dir.join("session.script");
    std::fs::write(&script_path, script).unwrap();
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("debug")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--rustc-arg")
        .arg("-g")
        .arg("--rustc-arg")
        .arg("-Zmir-opt-level=0")
        .arg("--debug-script")
        .arg(&script_path)
        .arg("--debug-dialect")
        .arg(dialect)
        .output()
        .expect("failed to run sparagmos-driver")
}

fn debug_driver_bootstrapped(dir: &Path, src: &Path, script: &str) -> Output {
    let script_path = dir.join("session.script");
    std::fs::write(&script_path, script).unwrap();
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("debug")
        .arg(src)
        .arg("--edition")
        .arg("2021")
        .arg("--debug-script")
        .arg(&script_path)
        .arg("--debug-dialect")
        .arg("gdb")
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}
