//! Collector-verdict replays, end to end through the driver binary.
//!
//! rustc rejects some programs only after monomorphization
//! begins: duplicate link symbols among the collected items, a
//! panic runtime whose strategy mismatches the session's
//! (linked statically, or pulled in through the `std` dylib
//! of a `-C prefer-dynamic` link), and
//! the per-instance `check_mono_item` checks (ABI target
//! features — including definitions no entry point reaches,
//! rooted by the reachability analysis alone — and the
//! `large_assignments` move check). Conversion
//! used to accept all of these, so compile-check records whose
//! expected outcome is a failure of one of those kinds flipped
//! to "expected failure but conversion succeeded" (Issue #509).
//! The driver now replays the verdicts in `mono_check`; each
//! test below pins one rejection and the nearby acceptance that
//! guards it against over-rejection.

use std::process::Command;

struct Case {
    dir: std::path::PathBuf,
}

impl Case {
    fn new(tag: &str) -> Case {
        let dir = std::env::temp_dir().join(format!(
            "sparagmos-driver-mono-verdicts-{}-{}",
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

    /// Run the driver in convert mode (no subcommand) on `src`
    /// with the given extra arguments; `bootstrap` sets
    /// RUSTC_BOOTSTRAP=1 for feature-gated sources.
    fn convert(
        &self,
        src: &std::path::Path,
        extra: &[&str],
        bootstrap: bool,
    ) -> (i32, String, String) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
        cmd.arg(src).arg("--edition").arg("2021");
        for a in extra {
            cmd.arg(a);
        }
        if bootstrap {
            cmd.env("RUSTC_BOOTSTRAP", "1");
        }
        let out = cmd.output().expect("failed to run sparagmos-driver");
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }
}

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

#[test]
fn duplicate_export_name_rejected() {
    let case = Case::new("dupe-export");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[export_name = \"fail\"]\npub fn a() {}\n\
         #[export_name = \"fail\"]\npub fn b() {}\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("compilation failed"), "stderr={stderr}");
    assert!(stderr.contains("already defined"), "stderr={stderr}");
}

#[test]
fn duplicate_symbol_across_fn_and_static_rejected() {
    let case = Case::new("dupe-cross");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[export_name = \"same\"]\npub fn a() {}\n\
         #[export_name = \"same\"]\npub static S: u32 = 0;\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("already defined"), "stderr={stderr}");
}

#[test]
fn single_export_name_converts() {
    let case = Case::new("export-one");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[export_name = \"fail\"]\npub fn a() {}\n",
    );
    let (code, stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn distinct_export_names_convert() {
    let case = Case::new("export-two");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[export_name = \"one\"]\npub fn a() {}\n\
         #[export_name = \"two\"]\npub fn b() {}\n",
    );
    let (code, stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn entry_symbol_collision_rejected() {
    // A `#[no_mangle] fn main` in a binary crate defines the
    // entry symbol the synthesized entry wrapper also defines.
    let case = Case::new("entry-dupe");
    let src = case.write("case.rs", "#[no_mangle]\nfn main() {}\n");
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("entry symbol"), "stderr={stderr}");
}

#[test]
fn entry_symbol_foreign_declaration_rejected() {
    // A foreign declaration of `main` also collides with the
    // synthesized entry wrapper: the code generator looks the
    // entry symbol up among the declared values.
    let case = Case::new("entry-foreign");
    let src = case.write(
        "case.rs",
        "#![allow(warnings)]\n\
         fn main() {\n\
             extern \"Rust\" { fn main(); }\n\
             unsafe { main(); }\n\
         }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("entry symbol"), "stderr={stderr}");
}

#[test]
fn foreign_declaration_converts() {
    // Declaring (and calling) an unrelated foreign function is
    // ordinary FFI and must keep converting.
    let case = Case::new("foreign-ok");
    let src = case.write(
        "case.rs",
        "extern \"C\" { fn puts(s: *const u8) -> i32; }\n\
         fn main() { unsafe { puts(b\"x\\0\".as_ptr()); } }\n",
    );
    let (code, stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn large_assignments_rejected() {
    let case = Case::new("large-move");
    let src = case.write(
        "case.rs",
        "#![feature(large_assignments)]\n\
         #![move_size_limit = \"1000\"]\n\
         #![deny(large_assignments)]\n\
         #![allow(unused)]\n\
         #[derive(Copy, Clone)]\n\
         struct Data([u8; 9999]);\n\
         fn one_arg(_: Data) {}\n\
         fn main() { one_arg(Data([0; 9999])); }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("moving 9999 bytes"), "stderr={stderr}");
}

#[test]
fn large_move_below_limit_converts() {
    let case = Case::new("large-ok");
    let src = case.write(
        "case.rs",
        "#![feature(large_assignments)]\n\
         #![move_size_limit = \"1000\"]\n\
         #![deny(large_assignments)]\n\
         #![allow(unused)]\n\
         #[derive(Copy, Clone)]\n\
         struct Data([u8; 999]);\n\
         fn one_arg(_: Data) {}\n\
         fn main() { one_arg(Data([0; 999])); }\n",
    );
    let (code, stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn simd_abi_check_rejected() {
    // An `extern "C"` signature mentioning `__m256` requires
    // the `avx` target feature in the caller's feature set;
    // with avx disabled the ABI check rejects the call.
    let case = Case::new("simd-abi");
    let src = case.write(
        "case.rs",
        "#![feature(portable_simd)]\n#![feature(simd_ffi)]\n\
         use std::arch::x86_64::*;\n\
         unsafe extern \"C\" fn f(_: __m256) {}\n\
         fn main() { unsafe { let x: __m256 = std::mem::transmute([0.0f32; 8]); f(x); } }\n",
    );
    let (code, _stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "-Ctarget-feature=-avx"],
        true,
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("compilation failed"), "stderr={stderr}");
    assert!(
        stderr.contains("requires the `avx` target feature"),
        "stderr={stderr}"
    );
}

#[test]
fn simd_abi_with_feature_enabled_converts() {
    let case = Case::new("simd-abi-ok");
    let src = case.write(
        "case.rs",
        "#![feature(portable_simd)]\n#![feature(simd_ffi)]\n\
         use std::arch::x86_64::*;\n\
         unsafe extern \"C\" fn f(_: __m256) {}\n\
         fn main() { unsafe { let x: __m256 = std::mem::transmute([0.0f32; 8]); f(x); } }\n",
    );
    let (code, stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "-Ctarget-feature=+avx"],
        true,
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

const PANIC_RUNTIME_ABORT: &str = "\
//@ compile-flags:-C panic=abort
#![feature(panic_runtime)]
#![crate_type = \"rlib\"]
#![no_std]
#![panic_runtime]

#[no_mangle]
pub extern \"C\" fn __rust_maybe_catch_panic() {}

#[no_mangle]
pub extern \"C\" fn __rust_start_panic() {}

#[no_mangle]
pub extern \"C\" fn rust_eh_personality() {}
";

const PANIC_RUNTIME_LANG_ITEMS: &str = "\
#![crate_type = \"rlib\"]
#![no_std]
#![feature(lang_items)]

use core::panic::PanicInfo;

#[panic_handler]
fn panic_impl(info: &PanicInfo) -> ! { loop {} }
#[lang = \"eh_personality\"]
fn eh_personality() {}
";

const PANIC_MAIN: &str = "\
#![no_std]
#![no_main]

extern crate panic_runtime_abort;
extern crate panic_runtime_lang_items;
";

#[test]
fn panic_runtime_strategy_mismatch_rejected() {
    // The session unwinds (the default) but the only linked
    // panic runtime was compiled with panic=abort.
    let case = Case::new("panic-mismatch");
    let abort = case.write("panic-runtime-abort.rs", PANIC_RUNTIME_ABORT);
    let lang = case.write("panic-runtime-lang-items.rs", PANIC_RUNTIME_LANG_ITEMS);
    let src = case.write("case.rs", PANIC_MAIN);
    let aux_abort = format!("panic_runtime_abort={}", abort.display());
    let aux_lang = format!("panic_runtime_lang_items={}", lang.display());
    let (code, _stdout, stderr) = case.convert(
        &src,
        &["--aux", &aux_abort, "--aux", &aux_lang],
        true,
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("is not compiled with this crate's panic strategy"),
        "stderr={stderr}"
    );
}

#[test]
fn panic_runtime_matching_strategy_converts() {
    // The same crate graph with the session itself aborting:
    // the runtime's strategy now matches.
    let case = Case::new("panic-match");
    let abort = case.write("panic-runtime-abort.rs", PANIC_RUNTIME_ABORT);
    let lang = case.write("panic-runtime-lang-items.rs", PANIC_RUNTIME_LANG_ITEMS);
    let src = case.write("case.rs", PANIC_MAIN);
    let aux_abort = format!("panic_runtime_abort={}", abort.display());
    let aux_lang = format!("panic_runtime_lang_items={}", lang.display());
    let (code, stdout, stderr) = case.convert(
        &src,
        &[
            "--aux",
            &aux_abort,
            "--aux",
            &aux_lang,
            "--rustc-arg",
            "-Cpanic=abort",
        ],
        true,
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn simd_abi_definition_in_rootless_lib_rejected() {
    // A library crate has no entry point and nothing calls
    // this `#[no_mangle]` definition: the collector roots it
    // through the reachability analysis alone (its export
    // level is below the rlib threshold), so the definition's
    // own ABI check fires — with avx disabled, an
    // `extern "C"` definition taking `__m256` is rejected for
    // its signature, with no call site involved.
    let case = Case::new("simd-def-lib");
    let src = case.write(
        "case.rs",
        "#![feature(portable_simd)]\n#![feature(simd_ffi)]\n\
         #![crate_type = \"rlib\"]\n\
         #![allow(improper_ctypes_definitions)]\n\
         use std::arch::x86_64::*;\n\
         #[no_mangle]\n\
         pub unsafe extern \"C\" fn f(_: __m256) {}\n",
    );
    let (code, _stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "-Ctarget-feature=-avx"],
        true,
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("requires the `avx` target feature"),
        "stderr={stderr}"
    );
}

#[test]
fn simd_abi_definition_in_rootless_lib_with_feature_converts() {
    // The same rootless definition with the feature enabled:
    // the definition is rooted and forced, and the ABI check
    // finds nothing to reject.
    let case = Case::new("simd-def-lib-ok");
    let src = case.write(
        "case.rs",
        "#![feature(portable_simd)]\n#![feature(simd_ffi)]\n\
         #![crate_type = \"rlib\"]\n\
         #![allow(improper_ctypes_definitions)]\n\
         use std::arch::x86_64::*;\n\
         #[no_mangle]\n\
         pub unsafe extern \"C\" fn f(_: __m256) {}\n",
    );
    let (code, stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "-Ctarget-feature=+avx"],
        true,
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn prefer_dynamic_panic_runtime_mismatch_rejected() {
    // Under `-C prefer-dynamic` the executable links the
    // `std` dylib, whose recorded dylib dependency formats
    // include the `panic_unwind` runtime it was built with as
    // a static dependency; the injected abort runtime is
    // therefore never activated, and the linked runtime's
    // strategy mismatches the abort session.
    let case = Case::new("panic-dylib");
    let src = case.write("case.rs", "fn main() {}\n");
    let (code, _stdout, stderr) = case.convert(
        &src,
        &[
            "--rustc-arg",
            "-Cpanic=abort",
            "--rustc-arg",
            "-Cprefer-dynamic",
        ],
        false,
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("panic_unwind")
            && stderr.contains("is not compiled with this crate's panic strategy"),
        "stderr={stderr}"
    );
}

#[test]
fn prefer_dynamic_matching_panic_runtime_converts() {
    // The same dynamic linkage with the session's strategy
    // left at unwind: the runtime the sweep links is the one
    // the session wants.
    let case = Case::new("panic-dylib-ok");
    let src = case.write("case.rs", "fn main() {}\n");
    let (code, stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "-Cprefer-dynamic"],
        false,
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn hello_world_converts() {
    let case = Case::new("hello");
    let src = case.write("case.rs", "fn main() { println!(\"hello\"); }\n");
    let (code, stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn hello_world_panic_abort_converts() {
    // A std program under -Cpanic=abort links the aborting
    // std: exactly one panic runtime, matching strategy.
    let case = Case::new("hello-abort");
    let src = case.write("case.rs", "fn main() { println!(\"hello\"); }\n");
    let (code, stdout, stderr) =
        case.convert(&src, &["--rustc-arg", "-Cpanic=abort"], false);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}
