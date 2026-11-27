//! The packed-stack / backchain verdict, end to end
//! through the driver binary.
//!
//! rustc rejects, as code generation computes each
//! function's attributes
//! (`rustc_codegen_llvm::attributes::packed_stack_attr`),
//! an s390x compilation with `-Zpacked-stack` in which a
//! codegen'd function enables the `backchain` target
//! feature -- through the session (the target spec or
//! `-C target-feature`) or through its own
//! `#[target_feature]` -- without `soft-float`: LLVM
//! cannot generate valid function entry points for that
//! combination under the default ABI. A metadata-only
//! session never reaches code generation, so conversion
//! accepted these programs and the `backchain_attr`
//! revision of tests/ui/target-feature/
//! packedstack-combinations (Issue #509) flipped to
//! "expected failure but conversion succeeded". The
//! driver replays the verdict in `mono_check` over the
//! session's internal target features and the codegen
//! attributes of the items code generation would define;
//! each test below pins one rejection shape and the
//! nearby acceptances that guard it against
//! over-rejection (packed-stack alone, the attribute
//! without `-Zpacked-stack`, and backchain on a
//! soft-float target, whose spec enables `soft-float`).
//! The predicate was pinned against native rustc
//! 1.100.0 before implementing: the rejection shapes
//! exit 0 under `--emit=metadata` and 1 under
//! `--emit=link`; the acceptances exit 0 under both.

use std::process::Command;

struct Case {
    dir: std::path::PathBuf,
}

impl Case {
    fn new(tag: &str) -> Case {
        let dir = std::env::temp_dir().join(format!(
            "sparagmos-driver-packed-stack-verdicts-{}-{}",
            std::process::id(),
            tag
        ));
        std::fs::create_dir_all(&dir).unwrap();
        register_scratch_dir(&dir);
        Case { dir }
    }

    fn write(&self, name: &str, source: &str) -> std::path::PathBuf {
        let path = self.dir.join(name);
        std::fs::write(&path, source).unwrap();
        path
    }

    /// Run the driver in convert mode (no subcommand) on
    /// `src` with the given extra arguments; these sources
    /// are feature-gated (`s390x_target_feature`), so
    /// RUSTC_BOOTSTRAP=1 is always set.
    fn convert(&self, src: &std::path::Path, extra: &[&str]) -> (i32, String, String) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
        cmd.arg(src).arg("--edition").arg("2021");
        for a in extra {
            cmd.arg(a);
        }
        cmd.env("RUSTC_BOOTSTRAP", "1");
        let out = cmd.output().expect("failed to run sparagmos-driver");
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }
}

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

const S390X_PACKED_STACK: &[&str] = &[
    "--rustc-arg",
    "--target=s390x-unknown-linux-gnu",
    "--rustc-arg",
    "-Zpacked-stack",
];

#[test]
fn backchain_attribute_rejected() {
    // The record's own shape (tests/ui/target-feature/
    // packedstack-combinations, backchain_attr revision):
    // an exported function whose `#[target_feature]`
    // enables `backchain` under `-Zpacked-stack`.
    let case = Case::new("backchain-attr");
    let src = case.write(
        "case.rs",
        "#![feature(s390x_target_feature)]\n\
         #![crate_type = \"rlib\"]\n\
         #[no_mangle]\n\
         #[target_feature(enable = \"backchain\")]\n\
         pub fn test() {}\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, S390X_PACKED_STACK);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("compilation failed"), "stderr={stderr}");
    assert!(
        stderr.contains("`-Zpacked-stack` is incompatible with `backchain` target feature"),
        "stderr={stderr}"
    );
}

#[test]
fn backchain_from_session_rejected() {
    // The session-carried shape of the same verdict:
    // `backchain` enabled for every function by
    // `-C target-feature`, with no attribute anywhere.
    let case = Case::new("backchain-session");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[no_mangle]\n\
         pub fn test() {}\n",
    );
    let (code, _stdout, stderr) = case.convert(
        &src,
        &[
            "--rustc-arg",
            "--target=s390x-unknown-linux-gnu",
            "--rustc-arg",
            "-Zpacked-stack",
            "--rustc-arg",
            "-Ctarget-feature=+backchain",
        ],
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("`-Zpacked-stack` is incompatible with `backchain` target feature"),
        "stderr={stderr}"
    );
}

#[test]
fn packed_stack_without_backchain_converts() {
    // The record's only_packedstack shape: packed-stack
    // alone enables neither feature, so nothing is
    // rejected.
    let case = Case::new("packed-stack-alone");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[no_mangle]\n\
         pub fn test() {}\n",
    );
    let (code, stdout, stderr) = case.convert(&src, S390X_PACKED_STACK);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn backchain_attribute_without_packed_stack_converts() {
    // The attribute alone is not the rejection: without
    // `-Zpacked-stack` the same function converts.
    let case = Case::new("backchain-no-packed-stack");
    let src = case.write(
        "case.rs",
        "#![feature(s390x_target_feature)]\n\
         #![crate_type = \"rlib\"]\n\
         #[no_mangle]\n\
         #[target_feature(enable = \"backchain\")]\n\
         pub fn test() {}\n",
    );
    let (code, stdout, stderr) =
        case.convert(&src, &["--rustc-arg", "--target=s390x-unknown-linux-gnu"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn backchain_on_softfloat_target_converts() {
    // The record's with_softfloat shape: the
    // s390x-unknown-none-softfloat target spec enables
    // `soft-float`, which lifts the rejection even with
    // session-carried `backchain`.
    let case = Case::new("backchain-softfloat-target");
    let src = case.write(
        "case.rs",
        "#![no_std]\n\
         #![crate_type = \"rlib\"]\n\
         #[no_mangle]\n\
         pub fn test() {}\n",
    );
    let (code, stdout, stderr) = case.convert(
        &src,
        &[
            "--rustc-arg",
            "--target=s390x-unknown-none-softfloat",
            "--rustc-arg",
            "-Zpacked-stack",
            "--rustc-arg",
            "-Ctarget-feature=+backchain",
        ],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn backchain_attribute_is_a_note_in_check_mode() {
    // The verdict is codegen-level: under
    // `--emit=metadata` the pinned compiler accepts the
    // crate, so in check mode the verdict is reported as
    // a note and the compilation succeeds (Issue #509).
    let case = Case::new("backchain-attr-check");
    let src = case.write(
        "case.rs",
        "#![feature(s390x_target_feature)]\n\
         #![crate_type = \"rlib\"]\n\
         #[no_mangle]\n\
         #[target_feature(enable = \"backchain\")]\n\
         pub fn test() {}\n",
    );
    let mut extra: Vec<&str> = vec!["--compiletest-check-mode"];
    extra.extend_from_slice(S390X_PACKED_STACK);
    let (code, _stdout, stderr) = case.convert(&src, &extra);
    assert_eq!(code, 0, "stderr={stderr}");
    assert!(
        stderr.contains("not applied in check mode"),
        "stderr={stderr}"
    );
    assert!(stderr.contains("packed-stack"), "stderr={stderr}");
}
