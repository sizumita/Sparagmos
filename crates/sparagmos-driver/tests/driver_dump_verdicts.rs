//! The symbol-name / def-path dump verdict, end to end
//! through the driver binary.
//!
//! rustc's dump test hook
//! (`rustc_symbol_mangling::test::dump_symbol_names_and_def_paths`)
//! runs as code generation starts: for every local item
//! carrying `#[rustc_dump_symbol_name]` or
//! `#[rustc_dump_def_path]` it emits an error carrying the
//! item's mangled symbol name or def path, so the
//! compilation fails. A metadata-only session never reaches
//! code generation, so conversion accepted these programs
//! and the compile-check records built on the hook
//! (tests/ui/symbol-names/*, tests/ui/splat/splat-mangling,
//! the dyn-compat symbol-mangling record and the two
//! doc_examples dump records of Issue #509) flipped to
//! "expected failure but conversion succeeded". The driver
//! replays the verdict in `mono_check` over the attribute
//! evidence itself; each test below pins one rejection
//! shape and the nearby acceptance that guards it against
//! over-rejection.

use std::process::Command;

struct Case {
    dir: std::path::PathBuf,
}

impl Case {
    fn new(tag: &str) -> Case {
        let dir = std::env::temp_dir().join(format!(
            "sparagmos-driver-dump-verdicts-{}-{}",
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
    /// `src`; these sources are feature-gated on
    /// `rustc_attrs`, so RUSTC_BOOTSTRAP=1 is always set.
    fn convert(&self, src: &std::path::Path) -> (i32, String, String) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
        cmd.arg(src).arg("--edition").arg("2021");
        cmd.env("RUSTC_BOOTSTRAP", "1");
        let out = cmd.output().expect("failed to run sparagmos-driver");
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }
}

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

#[test]
fn dump_symbol_name_rejected() {
    let case = Case::new("dump-symbol-name");
    let src = case.write(
        "case.rs",
        "#![feature(rustc_attrs)]\n\
         #[rustc_dump_symbol_name]\n\
         fn mangled() {}\n\
         fn main() { mangled(); }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("compilation failed"), "stderr={stderr}");
    assert!(stderr.contains("symbol-name"), "stderr={stderr}");
}

#[test]
fn dump_def_path_rejected() {
    let case = Case::new("dump-def-path");
    let src = case.write(
        "case.rs",
        "#![feature(rustc_attrs)]\n\
         #[rustc_dump_def_path]\n\
         fn main() {}\n",
    );
    let (code, _stdout, stderr) = case.convert(&src);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("compilation failed"), "stderr={stderr}");
    assert!(stderr.contains("def-path"), "stderr={stderr}");
}

#[test]
fn dump_symbol_name_on_impl_method_rejected() {
    // The hook walks every crate owner, impl items
    // included (the shape of tests/ui/symbol-names/impl1).
    let case = Case::new("dump-impl-method");
    let src = case.write(
        "case.rs",
        "#![feature(rustc_attrs)]\n\
         struct S;\n\
         impl S {\n\
             #[rustc_dump_symbol_name]\n\
             fn method(&self) {}\n\
         }\n\
         fn main() { S.method(); }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("symbol-name"), "stderr={stderr}");
}

#[test]
fn dump_symbol_name_on_foreign_fn_rejected() {
    // The doc_examples shape: a foreign declaration is a
    // crate owner too, and the hook reports on it.
    let case = Case::new("dump-foreign");
    let src = case.write(
        "case.rs",
        "#![feature(rustc_attrs)]\n\
         #![crate_type = \"rlib\"]\n\
         unsafe extern \"C\" {\n\
             #[rustc_dump_symbol_name]\n\
             fn abort();\n\
         }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("symbol-name"), "stderr={stderr}");
}

#[test]
fn rustc_attrs_without_dump_converts() {
    // The feature alone is not the rejection: with no
    // dump attribute anywhere, the hook reports nothing.
    let case = Case::new("no-dump");
    let src = case.write(
        "case.rs",
        "#![feature(rustc_attrs)]\n\
         fn plain() {}\n\
         fn main() { plain(); }\n",
    );
    let (code, stdout, stderr) = case.convert(&src);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}
