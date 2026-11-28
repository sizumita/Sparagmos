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
//! Two further verdicts are pinned here: a call to an intrinsic
//! that must be overridden but has no backend shim (the internal
//! error code generation raises for it), and a
//! `#[linkage = "common"]` static with a non-zero initializer
//! (the LLVM verifier's rejection of the module). A
//! `global_asm!` item's constant operands are pinned as well:
//! emitting the assembly evaluates them unconditionally, so an
//! operand whose evaluation fails (E0080) is a compilation
//! failure even though the operands live outside every body
//! and no required-constant list names them. The
//! The missing-MIR verdict is pinned as well: the collector
//! fatally rejects a demand to monomorphize an item whose
//! optimized MIR its home crate did not encode (an item
//! outside the home crate's reachable set, or any item of a
//! metadata-only build). The driver's own auxiliary builds
//! compile with `-Zalways-encode-mir`, so an auxiliary's
//! metadata encodes that MIR anyway and conversion accepted
//! the demand; the verdict replays the native encoding rule
//! for the auxiliary crates, over the conclusive evidence
//! (a metadata-only auxiliary build, or a demand from a local
//! body for a non-trait-impl item under a non-public module
//! that no public re-export exposes).
//! The target-cpu session validation is pinned as well:
//! rustc rejects, as code generation starts, a target whose
//! spec requires an explicit cpu when none is set, and a
//! `-C target-cpu` the target spec lists as unsupported --
//! both over session state alone, so conversion accepted
//! them. The
//! LTO / prefer-dynamic session validation is pinned as well:
//! rustc rejects `-C lto` with `-C prefer-dynamic` when every
//! output of the crate is a 'staticlib', 'bin' or 'cdylib'
//! output, over the session's options alone, but only as
//! code generation starts, so conversion accepted the
//! combination. The layout-failure replay is pinned as
//! well: a type whose layout does not exist (a layout cycle,
//! E0391) is rejected by rustc only once collected code
//! demands the layout — through a `size_of`/`align_of`
//! required constant, or through the ABI of a collected
//! instance — while conversion's speculative layout queries
//! fail for such types behind discounted diagnostics, and
//! also fail, with the same discount, for types the
//! reference compilation never lays out (the query depth
//! limit). The driver records the discounted failures by
//! type and replays them only at a collected demand. The
//! driver now
//! replays the verdicts in `mono_check`; each
//! test below pins one rejection and the nearby acceptance that
//! guards it against over-rejection. A rejected compilation
//! also stops the compiler session before the phases past
//! analysis: the diagnostics behind a verdict were discounted
//! at their fetches, so the compiler's own error gate cannot
//! stop code generation, and code generation would evaluate
//! the very constants a verdict rests on and panic on them --
//! the rejection must surface as a compilation failure, never
//! as a compiler crash.

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

    /// Run the driver in convert mode on `src` with auxiliary
    /// crates: each `(name, path)` pair is handed over with
    /// `--aux`, the shape the subset runner uses for a test's
    /// `//@ aux-crate:` / `//@ aux-build:` files (the driver
    /// compiles them itself, with `-Zalways-encode-mir`).
    fn convert_aux(
        &self,
        src: &std::path::Path,
        aux: &[(&str, &std::path::Path)],
    ) -> (i32, String, String) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"));
        cmd.arg(src).arg("--edition").arg("2021");
        for (name, path) in aux {
            cmd.arg("--aux").arg(format!("{}={}", name, path.display()));
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

#[test]
fn generic_export_name_rejected() {
    // An exported symbol cannot name a generic item: rustc
    // rejects the crate ("functions generic over types or
    // consts must be mangled"), where the driver used to panic
    // asking `Instance::mono` for the generic definition while
    // collecting the duplicate-symbol check's roots.
    let case = Case::new("generic-export");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[export_name = \"foo\"]\npub fn foo<T>() {}\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("must be mangled"), "stderr={stderr}");
}

#[test]
fn generic_no_mangle_rejected() {
    let case = Case::new("generic-no-mangle");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[no_mangle]\npub fn foo<T>() {}\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("must be mangled"), "stderr={stderr}");
}

#[test]
fn non_generic_export_name_accepted() {
    let case = Case::new("plain-export");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[export_name = \"foo\"]\npub fn foo() {}\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn unbounded_instantiation_recursion_is_a_recursion_limit_failure() {
    // The opaque-output recursion of
    // tests/ui/codegen/normalization-overflow/
    // recursion-issue-139659.rs: materializing `recurse`'s
    // body normalizes its types, and the normalization itself
    // recurses without bound inside the trait solver. rustc's
    // collector reports the instantiation recursion limit for
    // the same program; the driver must report that failure
    // (exit 1), not die on the normalization panic (exit 101).
    let case = Case::new("normalize-overflow");
    let src = case.write(
        "case.rs",
        "trait Trait { type Output; }\n\
         impl<O, F: Fn() -> O> Trait for F { type Output = O; }\n\
         struct Wrap<P>(P);\n\
         struct WrapOutput<O>(O);\n\
         impl<P: Trait> Trait for Wrap<P> { type Output = WrapOutput<P::Output>; }\n\
         fn wrap<P: Trait>(x: P) -> impl Trait { Wrap(x) }\n\
         fn consume<P: Trait>(_: P) -> P::Output { unimplemented!() }\n\
         pub fn recurse() -> impl Sized { consume(wrap(recurse)) }\n\
         pub fn main() {}\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("reached the recursion limit while instantiating"),
        "stderr={stderr}"
    );
}

#[test]
fn simd_select_mismatched_lengths_rejected() {
    // The derived PartialEq body over the repr(simd) mask
    // type must never be fetched (the MIR validator bans
    // projections into SIMD types, MCP#838); with it screened,
    // conversion completes and the monomorphization check
    // reports the intrinsic misuse in `main` — where the
    // driver used to die on the validator ICE (exit 101)
    // while asking for the derived body.
    let case = Case::new("simd-select");
    let src = case.write(
        "case.rs",
        "#![feature(repr_simd, core_intrinsics)]\n\
         #![allow(non_camel_case_types)]\n\
         use std::intrinsics::simd::simd_select;\n\
         #[repr(simd)]\n#[derive(Copy, Clone)]\npub struct u32x4(pub [u32; 4]);\n\
         #[repr(simd)]\n#[derive(Copy, Clone, PartialEq)]\nstruct b8x4(pub [i8; 4]);\n\
         #[repr(simd)]\n#[derive(Copy, Clone, PartialEq)]\nstruct b8x8(pub [i8; 8]);\n\
         fn main() {\n\
             let m8 = b8x8([0, 0, 0, 0, 0, 0, 0, 0]);\n\
             let x = u32x4([0, 0, 0, 0]);\n\
             unsafe { simd_select(m8, x, x); }\n\
         }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("invalid monomorphization of `simd_select`"),
        "stderr={stderr}"
    );
}
#[test]
fn oversized_vtable_const_rejected_without_debuginfo() {
    // `try_as_dyn` over `[u8; usize::MAX]` fails inside core:
    // the vtable constant's evaluation reports E0080 ("values
    // of the type ... are too big for the target
    // architecture"). Conversion materializes the instance,
    // discounts the diagnostic at the fetch and records it;
    // the monomorphization walk turns the record into the
    // compilation verdict. With debuginfo the session used to
    // reach the same rejection through code generation's own
    // error path, but without debuginfo code generation
    // evaluated the erroneous constant first and panicked
    // ("erroneous constant missed by mono item collection"),
    // so the driver reported a compiler crash (exit 101)
    // instead of the failure. The verdict now stops the
    // session before code generation in both shapes.
    let case = Case::new("vtable-oversized-const");
    let src = case.write(
        "case.rs",
        "#![feature(try_as_dyn)]\n\
         trait Trait {}\n\
         pub fn foo(x: &[u8; usize::MAX]) {\n\
             let _ = std::any::try_as_dyn::<[u8; usize::MAX], dyn Trait>(x);\n\
         }\n",
    );
    for debuginfo in ["0", "2"] {
        let flag = format!("debuginfo={debuginfo}");
        let (code, _stdout, stderr) = case.convert(
            &src,
            &[
                "--rustc-arg",
                "--crate-type=lib",
                "--rustc-arg",
                "--emit=mir",
                "--rustc-arg",
                "-C",
                "--rustc-arg",
                &flag,
            ],
            true,
        );
        assert_eq!(code, 1, "debuginfo={debuginfo} stderr={stderr}");
        assert!(
            stderr.contains("compilation failed"),
            "debuginfo={debuginfo} stderr={stderr}"
        );
        assert!(
            stderr.contains("too big for the target architecture"),
            "debuginfo={debuginfo} stderr={stderr}"
        );
        assert!(
            !stderr.contains("conversion panicked"),
            "debuginfo={debuginfo} stderr={stderr}"
        );
    }
}

#[test]
fn unshimmed_must_override_intrinsic_rejected() {
    // A `#[rustc_intrinsic]` declared without a body must be
    // overridden by the codegen backend; this backend shims no
    // `const_deallocate`, so a call to it is the internal error
    // rustc raises (tests/ui/intrinsics/not-overridden.rs).
    let case = Case::new("not-overridden");
    let src = case.write(
        "case.rs",
        "#![feature(intrinsics)]\n\
         #[rustc_intrinsic]\n\
         pub const unsafe fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize);\n\
         fn main() { unsafe { const_deallocate(std::ptr::null_mut(), 1, 1); } }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("compilation failed"), "stderr={stderr}");
    assert!(stderr.contains("must be overridden"), "stderr={stderr}");
}

#[test]
fn shimmed_must_override_intrinsic_converts() {
    // `ctpop` must be overridden too, but the builtin registry
    // serves it, so the call is shimmed and converts.
    let case = Case::new("ctpop-served");
    let src = case.write(
        "case.rs",
        "#![feature(core_intrinsics)]\n\
         fn main() { let _n = unsafe { core::intrinsics::ctpop(7u32) }; }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn library_declared_codegen_served_intrinsic_converts() {
    // A `#[rustc_intrinsic]` declaration has no MIR body in any
    // backend: code generation serves the call in the caller.
    // In a library conversion — a crate being built, like a
    // sysroot unit in a build-std compilation — such a
    // declaration is surface other crates call, so the
    // collector credits it (and demands naming it) without a
    // body. Core's lib.rs declares the codegen-served float
    // and fence intrinsics; read as target-crate skips, they
    // failed the cargo-shim build-std conversion of core.
    let case = Case::new("lib-codegen-intrinsic");
    let src = case.write(
        "case.rs",
        "#![feature(intrinsics)]\n\
         #[rustc_intrinsic]\n\
         fn sqrtf32(x: f32) -> f32;\n\
         pub fn root(x: f32) -> f32 { unsafe { sqrtf32(x) } }\n",
    );
    let (code, stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "--crate-type=lib"],
        true,
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.lines().any(|l| {
            l.starts_with("target-crate converted:") && l.ends_with("skipped: 0")
        }),
        "the intrinsic declaration must count as converted: {stdout}"
    );
}

#[test]
fn program_declared_codegen_served_intrinsic_stays_skipped() {
    // The same declaration in a program conversion stays an
    // ordinary conversion skip: the interpreter itself would
    // have to serve the call at run time, and it has no model
    // for the sized float spellings. (The verdict still
    // forgives the call — code generation would have served
    // it — so the conversion exits 0.)
    let case = Case::new("prog-codegen-intrinsic");
    let src = case.write(
        "case.rs",
        "#![feature(intrinsics)]\n\
         #[rustc_intrinsic]\n\
         fn sqrtf32(x: f32) -> f32;\n\
         fn main() { let _x = unsafe { sqrtf32(2.0f32) }; }\n",
    );
    let (code, stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.lines().any(|l| {
            l.starts_with("target-crate converted:") && l.ends_with("skipped: 1")
        }),
        "the intrinsic declaration must stay skipped: {stdout}"
    );
}

#[test]
fn locally_declared_caller_location_converts() {
    // `caller_location` is served by the interpreter's model
    // under both its `std::` and its `core::` canonical
    // spellings: when core itself is the target crate, the
    // intrinsic lookup re-canonicalizes the definition's name
    // to the `core::` form, which the registry missed, so the
    // override verdict rejected core's own build (intrinsic
    // `caller_location` must be overridden by codegen backend,
    // but isn't). A guest crate declaring the intrinsic itself
    // reaches the same lookup.
    let case = Case::new("local-caller-location");
    let src = case.write(
        "case.rs",
        "#![feature(intrinsics)]\n\
         #[rustc_intrinsic]\n\
         fn caller_location() -> &'static std::panic::Location<'static>;\n\
         fn main() { let _l = caller_location(); }\n",
    );
    let (code, stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
}

#[test]
fn codegen_only_float_intrinsic_converts() {
    // `sqrtf32` must be overridden too, and the interpreter
    // has no model for the sized float spellings — but code
    // generation lowers them, so the call is shimmed by the
    // backend the verdict mirrors and converts. (The verdict
    // originally replayed only the interpreter's registry
    // and rejected this valid program.)
    let case = Case::new("sqrtf32-codegen-only");
    let src = case.write(
        "case.rs",
        "#![feature(core_intrinsics)]\n\
         fn main() { let _x = unsafe { core::intrinsics::sqrtf32(2.0f32) }; }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn codegen_only_prefetch_intrinsic_converts() {
    // A locally declared `prefetch_read_data` (the minicore
    // shape) is likewise served by code generation's prefetch
    // lowering while the interpreter leaves it unmodelled:
    // an ordinary conversion skip, not the must-be-overridden
    // error.
    let case = Case::new("prefetch-codegen-only");
    let src = case.write(
        "case.rs",
        "#![feature(intrinsics)]\n\
         #[rustc_intrinsic]\n\
         fn prefetch_read_data<T, const LOCALITY: i32>(data: *const T);\n\
         fn main() { let v: u32 = 0; unsafe { prefetch_read_data::<u32, 0>(&v as *const u32); } }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn codegen_only_black_box_intrinsic_converts() {
    // `black_box` must be overridden too, and the
    // interpreter has no model for it — but code
    // generation lowers it (an inline-asm barrier
    // around the value), so a call in the minicore
    // shape — a locally declared body-less intrinsic,
    // as the codegen-llvm ABI tests link — is shimmed
    // by the backend the verdict mirrors and converts.
    // (The verdict originally reported it, rejecting
    // those valid programs.)
    let case = Case::new("black-box-codegen-only");
    let src = case.write(
        "case.rs",
        "#![feature(intrinsics)]\n\
         #[rustc_intrinsic]\n\
         pub const fn black_box<T>(dummy: T) -> T;\n\
         fn main() { let _x = black_box(41u32); }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn codegen_only_raw_eq_intrinsic_converts() {
    // `raw_eq` must be overridden too, and the
    // interpreter has no model for it — but code
    // generation lowers it to a load-and-compare of
    // the pointee, so the calls std's slice and array
    // equality bodies make are shimmed by the backend
    // the verdict mirrors and convert. The case is
    // compiled at opt-level 3, as the records are:
    // below it the equality is dispatched as an
    // interpreter builtin and the call never lands in
    // a converted body. (The verdict originally
    // reported them, rejecting the valid programs of
    // tests/codegen-llvm/array-equality.rs and
    // slice-ref-equality.rs.)
    let case = Case::new("raw-eq-codegen-only");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"lib\"]\n\
         #[no_mangle]\n\
         pub fn array_eq_value(a: [u16; 3], b: [u16; 3]) -> bool { a == b }\n",
    );
    let (code, _stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "-Copt-level=3"],
        true,
    );
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn comptime_intrinsic_converts() {
    // `offset_of` must be overridden too, but it is a
    // `#[rustc_comptime]` intrinsic: the compiler rejects
    // its calls outside constant contexts, so the calls
    // the `offset_of!` desugar places in inline const
    // blocks and promoted constants are evaluated as
    // constants and never reach code generation — there
    // is no backend shim to miss. (The verdict originally
    // reported them, rejecting the valid programs of
    // tests/ui/offset-of/*.)
    let case = Case::new("offset-of-comptime");
    let src = case.write(
        "case.rs",
        "struct S { v: u8, w: u16 }\n\
         fn return_static_slice() -> &'static [usize] {\n\
         &[core::mem::offset_of!(S, v), core::mem::offset_of!(S, w)] }\n\
         fn use_reference() -> usize {\n\
         let r = &core::mem::offset_of!(S, v); *r * 6 }\n\
         fn main() { let _ = return_static_slice(); let _ = use_reference(); }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn comptime_type_name_const_argument_converts() {
    // `type_name` is a `#[rustc_comptime]` intrinsic too,
    // and the shape of tests/ui/const-generics/
    // intrinsics-type_name-as-const-argument.rs demands it
    // differently from `offset_of!`: the call sits in the
    // anonymous constant of a where clause's const generic
    // argument, a demand rooted by the trait predicate
    // rather than by a body the collector walks into. It
    // is still constant evaluation only -- the constant is
    // resolved before code generation, which never sees
    // the call -- so the must-be-overridden verdict must
    // not fire. (At the verdict's introduction this record
    // failed with the same rejection as the offset_of
    // family.)
    let case = Case::new("type-name-const-arg");
    let src = case.write(
        "case.rs",
        "#![allow(incomplete_features)]\n\
         #![feature(adt_const_params, unsized_const_params, generic_const_exprs)]\n\
         #![feature(core_intrinsics)]\n\
         #![feature(const_type_name)]\n\
         trait Trait<const S: &'static str> {}\n\
         struct Bug<T>\n\
         where\n\
             T: Trait<{ std::intrinsics::type_name::<T>() }>,\n\
         {\n\
             t: T,\n\
         }\n\
         fn main() {}\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn comptime_offset_of_promotion_shape_converts() {
    // The shapes of tests/ui/inline-const/
    // dont-eval-const-block-during-promotion.rs: since
    // `offset_of!` desugars to an inline const block, the
    // comptime call sits in the index and divisor
    // positions constant promotion examines, including
    // indexes into a local array promotion can never
    // lift. The verdict must not fire for any of them,
    // and conversion must accept the program without
    // evaluating the blocks.
    let case = Case::new("offset-of-promotion");
    let src = case.write(
        "case.rs",
        "use std::mem::offset_of;\n\
         struct Thing(i32);\n\
         fn main() {\n\
         let x = &([0][const { 0 }] & 0);\n\
         let y = &(1 / const { 1 });\n\
         let temp = [0u8];\n\
         let _ = &(temp[offset_of!(Thing, 0)] & 0u8);\n\
         let temp2: &[u8] = &[0u8];\n\
         let _ = &(temp2[const { 0usize }] & 0u8);\n\
         (x, y); }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn fallback_body_intrinsic_converts() {
    // `rotate_left` is declared in core with a fallback body,
    // so it need not be overridden: the method's body calls
    // the intrinsic, and conversion converts the fallback.
    let case = Case::new("rotate-fallback");
    let src = case.write(
        "case.rs",
        "fn main() { let _v = 1u32.rotate_left(2); }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn common_linkage_nonzero_initializer_rejected() {
    // A common global is a tentative definition: its
    // initializer must be all zero, and the LLVM verifier
    // rejects the module otherwise
    // (tests/ui/linkage-attr/common-linkage-non-zero-init.rs).
    let case = Case::new("common-nonzero");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"lib\"]\n\
         #![feature(linkage)]\n\
         #[linkage = \"common\"]\n\
         #[no_mangle]\n\
         pub static TEST: bool = true;\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("compilation failed"), "stderr={stderr}");
    assert!(stderr.contains("zero initializer"), "stderr={stderr}");
}

#[test]
fn common_linkage_zero_initializer_converts() {
    // The same declaration with a zero initializer is the
    // legal tentative definition and converts.
    let case = Case::new("common-zero");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"lib\"]\n\
         #![feature(linkage)]\n\
         #[linkage = \"common\"]\n\
         #[no_mangle]\n\
         pub static TEST: bool = false;\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], true);
    assert_eq!(code, 0, "stderr={stderr}");
}


#[test]
fn adapter_tower_instantiation_recursion_is_a_recursion_limit_failure() {
    // The packet-decoder recursion of
    // tests/ui/codegen/normalization-overflow/
    // recursion-issue-122823.rs (and -92004.rs, the same
    // program): each `operator_packet` instantiation calls
    // `decode_packets` on an iterator wrapped in one more
    // `Peekable<Take<...>>` layer, so instantiation nests
    // without bound and rustc reports the recursion limit.
    // The driver must report that same failure (exit 1) --
    // and reach it promptly: the verdict is final the
    // moment the collector's depth cap trips, so conversion
    // must not keep materializing the tower's bodies, and
    // screening each demanded instance must not re-walk
    // the tower suffix once per path that reaches it.
    let case = Case::new("adapter-tower");
    let src = case.write(
        "case.rs",
        "use std::vec::Vec;\n\
         use std::iter::Peekable;\n\
         \n\
         pub fn main() {\n\
             let packet = decode(vec![1,0,1,0]);\n\
         }\n\
         \n\
         pub fn decode(bitstream: Vec<u64>) -> Packet {\n\
             let mut bitstream_itr = bitstream.into_iter().peekable();\n\
             return match decode_packet(&mut bitstream_itr) {\n\
                 Some(p) => p,\n\
                 None    => panic!(\"expected outer packet\"),\n\
             }\n\
         }\n\
         \n\
         pub fn decode_packets<I: Iterator<Item = u64>>(itr: &mut Peekable<I>) -> Vec<Packet> {\n\
             let mut res = Vec::new();\n\
             loop {\n\
                 match decode_packet(itr) {\n\
                     Some(p) => { res.push(p); },\n\
                     None    => break\n\
                 }\n\
             }\n\
         \n\
             return res;\n\
         }\n\
         \n\
         pub fn decode_packet<I: Iterator<Item = u64>>(itr: &mut Peekable<I>) -> Option<Packet> {\n\
             let version = extend_number(0, itr, 3)?;\n\
             let type_id = extend_number(0, itr, 3)?;\n\
             return operator_packet(version, type_id, itr);\n\
         }\n\
         \n\
         pub fn operator_packet<I: Iterator<Item = u64>>(version: u64, type_id: u64, itr: &mut Peekable<I>) -> Option<Packet> {\n\
             let p = OperatorPacket {\n\
                 version: version,\n\
                 type_id: type_id,\n\
                 packets: decode_packets(&mut itr.take(0).peekable()),\n\
             };\n\
         \n\
             return Some(Packet::Operator(p));\n\
         }\n\
         \n\
         pub fn extend_number<I: Iterator<Item = u64>>(num: u64, itr: &mut Peekable<I>, take: u64) -> Option<u64> {\n\
             let mut value = num;\n\
             for _ in 0..take {\n\
                 value *= 2;\n\
                 value += itr.next()?;\n\
             }\n\
         \n\
             return Some(value);\n\
         }\n\
         \n\
         #[derive(Debug)]\n\
         pub enum Packet {\n\
             Operator(OperatorPacket),\n\
         }\n\
         \n\
         #[derive(Debug)]\n\
         pub struct OperatorPacket {\n\
             version: u64,\n\
             type_id: u64,\n\
             packets: Vec<Packet>\n\
         }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("reached the recursion limit while instantiating"),
        "stderr={stderr}"
    );
}

#[test]
fn finite_adapter_tower_converts() {
    // The finite form of the same shape: a bounded stack
    // of `Take`/`Peekable` adapters, demanded and converted
    // in full. The projection screen's memoized walks and
    // the resolution scan's per-ADT buckets must not turn
    // a bounded tower into a rejection.
    let case = Case::new("finite-tower");
    let src = case.write(
        "case.rs",
        "pub fn total<I: Iterator<Item = u32>>(itr: &mut std::iter::Peekable<I>) -> u32 {\n\
             let mut sum = 0;\n\
             while let Some(x) = itr.next() {\n\
                 sum += x;\n\
                 let _ = itr.peek();\n\
             }\n\
             sum\n\
         }\n\
         \n\
         pub fn main() {\n\
             let v: Vec<u32> = vec![1, 2, 3, 4, 5];\n\
             let mut itr = v.iter().copied().take(4).peekable().take(3).peekable();\n\
             let n = total(&mut itr);\n\
             println!(\"{n}\");\n\
         }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn global_asm_const_eval_failure_rejected() {
    // tests/ui/asm/fail-const-eval-issue-121099: the operands
    // of a `global_asm!` item are evaluated when the assembly
    // is emitted, so a failing operand is a compilation error
    // (E0080) -- but the operands are anonymous constants
    // outside every body, so conversion alone never evaluates
    // them and used to accept the crate.
    let case = Case::new("global-asm-bad-const");
    let src = case.write(
        "case.rs",
        "use std::arch::global_asm;\n\
         \n\
         fn main() {}\n\
         \n\
         global_asm!(\"/* {} */\", const 1 << 500);\n\
         global_asm!(\"/* {} */\", const 1 / 0);\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("compilation failed"), "stderr={stderr}");
    assert!(stderr.contains("E0080"), "stderr={stderr}");
}

#[test]
fn global_asm_const_operands_accepted() {
    // The acceptance guarding the rejection above: operands
    // that evaluate cleanly keep the crate converting.
    let case = Case::new("global-asm-good-const");
    let src = case.write(
        "case.rs",
        "use std::arch::global_asm;\n\
         \n\
         fn main() {}\n\
         \n\
         global_asm!(\"/* {} */\", const 1 << 5);\n\
         global_asm!(\"/* {} */\", const 40 + 2);\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn lto_prefer_dynamic_rejected() {
    // The record's own shape (tests/ui/lto/issue-11154):
    // `-C lto -C prefer-dynamic` on a binary crate. rustc
    // rejects the combination when code generation starts;
    // conversion must report the same failure.
    let case = Case::new("lto-prefer-dynamic");
    let src = case.write("case.rs", "fn main() {}\n");
    let (code, _stdout, stderr) = case.convert(
        &src,
        &[
            "--rustc-arg",
            "-C",
            "--rustc-arg",
            "lto",
            "--rustc-arg",
            "-C",
            "--rustc-arg",
            "prefer-dynamic",
        ],
        false,
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("cannot prefer dynamic linking when performing LTO"),
        "stderr={stderr}"
    );
}

#[test]
fn lto_thin_prefer_dynamic_rejected() {
    // Thin LTO is LTO for this validation as well.
    let case = Case::new("lto-thin-prefer-dynamic");
    let src = case.write("case.rs", "fn main() {}\n");
    let (code, _stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "-Clto=thin", "--rustc-arg", "-Cprefer-dynamic"],
        false,
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("cannot prefer dynamic linking when performing LTO"),
        "stderr={stderr}"
    );
}

#[test]
fn lto_prefer_dynamic_staticlib_cdylib_rejected() {
    // A library crate whose outputs are only LTO outputs
    // (staticlib + cdylib) is rejected the same way.
    let case = Case::new("lto-prefer-dynamic-lib-outputs");
    let src = case.write("case.rs", "pub fn f() {}\n");
    let (code, _stdout, stderr) = case.convert(
        &src,
        &[
            "--rustc-arg",
            "--crate-type=staticlib",
            "--rustc-arg",
            "--crate-type=cdylib",
            "--rustc-arg",
            "-Clto",
            "--rustc-arg",
            "-Cprefer-dynamic",
        ],
        false,
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("cannot prefer dynamic linking when performing LTO"),
        "stderr={stderr}"
    );
}

#[test]
fn lto_prefer_dynamic_rlib_converts() {
    // The acceptance guarding the rejections above: an rlib
    // output lifts the rejection, because no LTO link of
    // this crate happens -- the bitcode travels to a later
    // link.
    let case = Case::new("lto-prefer-dynamic-rlib");
    let src = case.write("case.rs", "pub fn f() {}\n");
    let (code, stdout, stderr) = case.convert(
        &src,
        &[
            "--rustc-arg",
            "--crate-type=rlib",
            "--rustc-arg",
            "-Clto",
            "--rustc-arg",
            "-Cprefer-dynamic",
        ],
        false,
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn lto_without_prefer_dynamic_converts() {
    // LTO alone is an ordinary compilation.
    let case = Case::new("lto-only");
    let src = case.write("case.rs", "fn main() {}\n");
    let (code, stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "-Clto"],
        false,
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn prefer_dynamic_without_lto_converts() {
    // Prefer-dynamic alone is an ordinary compilation.
    let case = Case::new("prefer-dynamic-only");
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
fn explicit_target_cpu_required_avr_rejected() {
    // The record's own shape (tests/ui/target-cpu/
    // explicit-target-cpu, avr_nocpu revision): the AVR
    // target spec sets `need_explicit_cpu`, so a session
    // with no `-C target-cpu` is a fatal error when code
    // generation starts; conversion must report the same
    // failure.
    let case = Case::new("target-cpu-required-avr");
    let src = case.write(
        "case.rs",
        "#![feature(no_core)]\n\
         #![no_core]\n\
         #![crate_type = \"rlib\"]\n",
    );
    let (code, _stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "--target=avr-none"],
        true,
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains(
            "target requires explicitly specifying a cpu with `-C target-cpu`"
        ),
        "stderr={stderr}"
    );
}

#[test]
fn explicit_target_cpu_required_amdgcn_rejected() {
    // The other `need_explicit_cpu` target in the pinned
    // tree (tests/ui/target-cpu/explicit-target-cpu,
    // amdgcn_nocpu revision), same validation.
    let case = Case::new("target-cpu-required-amdgcn");
    let src = case.write(
        "case.rs",
        "#![feature(no_core)]\n\
         #![no_core]\n\
         #![crate_type = \"rlib\"]\n",
    );
    let (code, _stdout, stderr) = case.convert(
        &src,
        &["--rustc-arg", "--target=amdgcn-amd-amdhsa"],
        true,
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains(
            "target requires explicitly specifying a cpu with `-C target-cpu`"
        ),
        "stderr={stderr}"
    );
}

#[test]
fn explicit_target_cpu_supplied_converts() {
    // The acceptance guarding the rejections above: the
    // same target with an explicit cpu converts.
    let case = Case::new("target-cpu-supplied-avr");
    let src = case.write(
        "case.rs",
        "#![feature(no_core)]\n\
         #![no_core]\n\
         #![crate_type = \"rlib\"]\n",
    );
    let (code, stdout, stderr) = case.convert(
        &src,
        &[
            "--rustc-arg",
            "--target=avr-none",
            "--rustc-arg",
            "-Ctarget-cpu=atmega328p",
        ],
        true,
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn unsupported_target_cpu_rejected() {
    // The record's own shape (tests/ui/target-cpu/
    // unsupported-target-cpu): `sm_60` is in the NVPTX
    // target spec's `unsupported_cpus` list, so rustc
    // fatally rejects it when code generation starts;
    // conversion must report the same failure.
    let case = Case::new("target-cpu-unsupported-nvptx");
    let src = case.write(
        "case.rs",
        "#![feature(no_core)]\n\
         #![no_core]\n\
         #![crate_type = \"rlib\"]\n",
    );
    let (code, _stdout, stderr) = case.convert(
        &src,
        &[
            "--rustc-arg",
            "--target=nvptx64-nvidia-cuda",
            "--rustc-arg",
            "-Ctarget-cpu=sm_60",
        ],
        true,
    );
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("target cpu `sm_60` is known but unsupported"),
        "stderr={stderr}"
    );
}

#[test]
fn supported_target_cpu_converts() {
    // The acceptance guarding the rejection above: a cpu
    // the NVPTX spec does not list as unsupported converts.
    let case = Case::new("target-cpu-supported-nvptx");
    let src = case.write(
        "case.rs",
        "#![feature(no_core)]\n\
         #![no_core]\n\
         #![crate_type = \"rlib\"]\n",
    );
    let (code, stdout, stderr) = case.convert(
        &src,
        &[
            "--rustc-arg",
            "--target=nvptx64-nvidia-cuda",
            "--rustc-arg",
            "-Ctarget-cpu=sm_70",
        ],
        true,
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn abandoned_force_ice_diagnostic_discounted() {
    // The required-const walk forces `check_mono_item` on
    // instances the collector itself would never collect; on
    // an instance whose types do not normalize in the
    // monomorphized environment (here the projections of an
    // impl whose `Self` carries the unprovable assumed bound
    // `&'a [u8]: Archive`), the force panics inside the query
    // after emitting its ICE diagnostic into the session.
    // The abandoned force's diagnostic must not fail the
    // compilation: rustc accepts the crate (this is the shape
    // of tests/ui/mir/issue-107691.rs, and of
    // tests/ui/traits/next-solver/generalize/
    // hr-alias-universe-lowering-ambiguity.rs, whose forced
    // type is a closure over a projection).
    let case = Case::new("abandoned-force-ice");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"lib\"]\n\
         pub trait Archive {\n\
             type Archived;\n\
             type Resolver;\n\
             fn resolve(resolver: Self::Resolver, out: *mut Self::Archived);\n\
         }\n\
         pub type Archived<T> = <T as Archive>::Archived;\n\
         pub type Resolver<T> = <T as Archive>::Resolver;\n\
         pub struct Record<'a> { _payload: &'a [u8] }\n\
         pub struct ArchivedRecord<'a> where &'a [u8]: Archive {\n\
             _payload: Archived<&'a [u8]>,\n\
         }\n\
         pub struct RecordResolver<'a> where &'a [u8]: Archive {\n\
             _payload: Resolver<&'a [u8]>,\n\
         }\n\
         impl<'a> Archive for Record<'a> where &'a [u8]: Archive {\n\
             type Archived = ArchivedRecord<'a>;\n\
             type Resolver = RecordResolver<'a>;\n\
             fn resolve(_resolver: Self::Resolver, _out: *mut Self::Archived) {}\n\
         }\n",
    );
    let (code, stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn duplicate_export_name_is_a_note_in_check_mode() {
    // A duplicate export symbol is a codegen-level verdict
    // (the partitioning check): under `--emit=metadata` the
    // pinned compiler accepts the crate, so in check mode the
    // verdict is reported as a note and the compilation
    // succeeds (Issue #509).
    let case = Case::new("dupe-export-check");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[export_name = \"fail\"]\npub fn a() {}\n\
         #[export_name = \"fail\"]\npub fn b() {}\n",
    );
    let (code, _stdout, stderr) =
        case.convert(&src, &["--compiletest-check-mode"], false);
    assert_eq!(code, 0, "stderr={stderr}");
    assert!(
        stderr.contains("not applied in check mode"),
        "stderr={stderr}"
    );
    assert!(stderr.contains("already defined"), "stderr={stderr}");
}

#[test]
fn simd_abi_check_is_not_a_verdict_in_check_mode() {
    // The ABI check is a `check_mono_item` finding: in check
    // mode the collector walk does not run, so the same call
    // the verdict rejects in a full build converts cleanly
    // (Issue #509).
    let case = Case::new("simd-abi-check");
    let src = case.write(
        "case.rs",
        "#![feature(portable_simd)]\n#![feature(simd_ffi)]\n\
         use std::arch::x86_64::*;\n\
         unsafe extern \"C\" fn f(_: __m256) {}\n\
         fn main() { unsafe { let x: __m256 = std::mem::transmute([0.0f32; 8]); f(x); } }\n",
    );
    let (code, _stdout, stderr) = case.convert(
        &src,
        &[
            "--compiletest-check-mode",
            "--rustc-arg",
            "-Ctarget-feature=-avx",
        ],
        true,
    );
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn generic_export_name_still_rejected_in_check_mode() {
    // The generic-export verdict is metadata-visible: the
    // pinned compiler rejects the crate under
    // `--emit=metadata` too, so check mode keeps it a verdict
    // (Issue #509).
    let case = Case::new("generic-export-check");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         #[export_name = \"foo\"]\npub fn foo<T>() {}\n",
    );
    let (code, _stdout, stderr) =
        case.convert(&src, &["--compiletest-check-mode"], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(stderr.contains("must be mangled"), "stderr={stderr}");
}
#[test]
fn recursion_limit_instantiation_is_a_failure_not_a_crash() {
    // A generic function that instantiates itself with a
    // strictly larger type on every call: the collector
    // rejects the crate with "reached the recursion limit
    // while instantiating" and never codegen-checks any of
    // the chain's instances. The walk used to force
    // `check_mono_item` on the chain's instances anyway; past
    // a few dozen layers the ABI check's layout computation
    // fails on the adapter types and its error handler raises
    // the failure as an internal compiler error, which the
    // witness conversion reported as a compiler crash (exit
    // 101) for a compilation the compiler itself fails
    // cleanly. The walk now tracks the collector's
    // instantiation depths, defers a force's crash behind
    // the collection-phase verdicts, and contains a
    // mention's resolution panic to that mention, so the
    // chain reaches the limit verdict (this is the shape of
    // tests/ui/codegen/normalization-overflow/
    // recursion-issue-118590.rs).
    let case = Case::new("recursion-limit-instantiation");
    let src = case.write(
        "case.rs",
        "fn main() {\n\
             recurse(std::iter::empty::<()>());\n\
         }\n\
         fn recurse(nums: impl Iterator) {\n\
             if true { return; }\n\
             recurse(nums.skip(42).peekable())\n\
         }\n",
    );
    let (code, stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stderr.contains("compilation failed"),
        "stderr={stderr}"
    );
    assert!(
        !stderr.contains("Stable MIR conversion panicked"),
        "stderr={stderr}"
    );
}

#[test]
fn deep_but_finite_instantiation_converts() {
    // The same adapter nesting, bounded and written out in
    // the entry point: no definition is instantiated inside
    // itself along any mention path, so the collector
    // collects every instance and the compilation succeeds.
    // The recursion-limit verdict must not fire, and the
    // walk's forces on the deep (but collected) instances
    // must not turn into a crash.
    let case = Case::new("deep-finite-instantiation");
    let mut source = String::from(
        "fn consume(nums: impl Iterator<Item = ()>) { let _ = nums.count(); }\n\
         fn main() {\n\
             let it = std::iter::empty::<()>();\n",
    );
    for _ in 0..40 {
        source.push_str("    let it = it.skip(1).peekable();\n");
    }
    source.push_str("    consume(it);\n}\n");
    let src = case.write("case.rs", &source);
    let (code, stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn trait_method_no_mangle_accepted() {
    // `#[no_mangle]` on a trait method declaration is inert:
    // rustc warns the attribute cannot be used on trait items
    // and its `InvalidNoMangleItems` lint never visits them,
    // so the generic-export verdict must not reject the crate
    // for it (the shape of
    // tests/ui/feature-gates/issue-43106-gating-of-builtin-attrs.rs).
    // The free-function rejections above pin the other side.
    let case = Case::new("trait-no-mangle");
    let src = case.write(
        "case.rs",
        "#![crate_type = \"rlib\"]\n\
         trait Tr {\n\
             #[no_mangle]\n\
             fn foo();\n\
             #[no_mangle]\n\
             fn bar() {}\n\
         }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stderr={stderr}");
}

#[test]
fn missing_mir_priv_bounds_aux_rejected() {
    // The record's own shape (tests/ui/privacy/
    // missing-mir-priv-bounds): a private associated-type
    // bound leaks a type from a private module of the
    // auxiliary crate, and monomorphizing the local caller
    // demands that type's inherent generic method. The
    // auxiliary's native metadata encodes no MIR for it (it
    // is outside the auxiliary's reachable set), so rustc
    // fails with "missing optimized MIR"; the driver's
    // auxiliary build encodes it under
    // `-Zalways-encode-mir`, and the verdict replays the
    // native rule.
    let case = Case::new("missing-mir-priv");
    let aux = case.write(
        "missing-mir-priv-bounds-extern.rs",
        r##"pub trait ToPriv {
    type AssocPriv;
}

pub trait PubTr {
    #[expect(private_bounds)]
    type Assoc: ToPriv<AssocPriv = Priv>;
}

struct Dummy;
struct DummyToPriv;
impl PubTr for Dummy {
    type Assoc = DummyToPriv;
}
impl ToPriv for DummyToPriv {
    type AssocPriv = Priv;
}

pub fn get_dummy() -> impl PubTr {
    Dummy
}

struct Priv;

pub trait GetUnreachable {
    type Assoc;
}

mod m {
    pub struct Unreachable;

    impl Unreachable {
        #[expect(dead_code)]
        pub fn generic<T>() {}
    }

    impl crate::GetUnreachable for crate::Priv {
        type Assoc = Unreachable;
    }
}
"##,
    );
    let src = case.write(
        "case.rs",
        r##"//@ aux-crate:dep=missing-mir-priv-bounds-extern.rs

extern crate dep;
use dep::{get_dummy, GetUnreachable, PubTr, ToPriv};

fn main() {
    wut(get_dummy());
}

fn wut<T: PubTr>(_: T) {
    <T as Access>::AccessAssoc::generic::<i32>();
}

trait Access: PubTr {
    type AccessAssoc;
}

impl<T: PubTr> Access for T {
    type AccessAssoc = <<T::Assoc as ToPriv>::AssocPriv as GetUnreachable>::Assoc;
}
"##,
    );
    let (code, _stdout, stderr) = case.convert_aux(&src, &[("dep", &aux)]);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("missing optimized MIR"),
        "stderr={stderr}"
    );
}

#[test]
fn missing_mir_metadata_only_aux_rejected() {
    // The record's own shape (tests/ui/rmeta/
    // no_optimized_mir): the auxiliary's own build is
    // metadata-only (`--emit=metadata` among its compile
    // flags), so a native metadata encodes no optimized MIR
    // at all, and monomorphizing its plain public function
    // from the main crate is the collector's fatal error.
    let case = Case::new("missing-mir-rmeta");
    let aux = case.write(
        "rmeta_aux.rs",
        "//@ compile-flags: --emit=metadata\n\npub fn missing_optimized_mir() {}\n",
    );
    let src = case.write(
        "case.rs",
        "//@ aux-crate:dep=rmeta_aux.rs\nextern crate dep;\nfn main() { dep::missing_optimized_mir(); }\n",
    );
    let (code, _stdout, stderr) = case.convert_aux(&src, &[("dep", &aux)]);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("missing optimized MIR"),
        "stderr={stderr}"
    );
}

#[test]
fn missing_mir_transitive_macro_aux_rejected() {
    // The record's own shape (tests/ui/definition-reachable/
    // transitive): a public macro of the auxiliary crate
    // expands, in the main crate, to a call of a plain
    // function under two private modules. The function is
    // neither generic nor inlinable and outside the
    // auxiliary's reachable set, so its MIR is not encoded
    // natively and the demand is the collector's fatal error.
    let case = Case::new("missing-mir-transitive");
    let aux = case.write(
        "transitive-macro.rs",
        r##"#![feature(decl_macro)]

mod mod1 {
    mod mod2 {
        pub fn foo() {}
    }

    pub(crate) macro m1() {
        mod2::foo()
    }
}

pub macro m() {
    mod1::m1!()
}
"##,
    );
    let src = case.write(
        "case.rs",
        "//@ aux-build:transitive-macro.rs\nextern crate transitive_macro;\nfn main() {\n    transitive_macro::m!();\n}\n",
    );
    let (code, _stdout, stderr) = case.convert_aux(&src, &[("transitive_macro", &aux)]);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("missing optimized MIR"),
        "stderr={stderr}"
    );
}

#[test]
fn missing_mir_glob_reexported_generic_converts() {
    // The record's own shape (tests/ui/imports/
    // ambiguous-reachable): a generic function under a
    // private module of the auxiliary crate, re-exported
    // from the auxiliary's root by a public glob (a second
    // glob brings a non-generic namesake, which is what
    // makes the import ambiguous). The function is on the
    // auxiliary's public API surface, so it is in the home
    // crate's reachable set and its MIR is encoded natively;
    // the module-ancestor evidence alone must not conclude
    // otherwise.
    let case = Case::new("missing-mir-glob-reexport");
    let aux = case.write(
        "ambiguous-reachable-extern.rs",
        r##"mod m1 {
    pub fn generic<T>() {
        let x = 10;
        let y = 11;
        println!("hello {x} world {:?}", y);
    }
}

mod m2 {
    pub fn generic() {}
}

pub use m1::*;
pub use m2::*;
"##,
    );
    let src = case.write(
        "case.rs",
        "//@ aux-crate:dep=ambiguous-reachable-extern.rs\n\n#![allow(ambiguous_glob_imports)]\n\nextern crate dep;\n\nfn main() {\n    dep::generic::<u8>();\n}\n",
    );
    let (code, stdout, stderr) = case.convert_aux(&src, &[("dep", &aux)]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn missing_mir_named_reexported_generic_converts() {
    // The named sibling of the glob shape above: a generic
    // function under a private module of the auxiliary
    // crate, re-exported from the root by name. Same
    // reachable-set membership, same native encoding.
    let case = Case::new("missing-mir-named-reexport");
    let aux = case.write(
        "named-reexport-extern.rs",
        "mod m1 {\n    pub fn generic<T>(x: T) -> T { x }\n}\n\npub use m1::generic;\n",
    );
    let src = case.write(
        "case.rs",
        "//@ aux-crate:dep=named-reexport-extern.rs\n\nextern crate dep;\n\nfn main() {\n    let _ = dep::generic::<u8>(3);\n}\n",
    );
    let (code, stdout, stderr) = case.convert_aux(&src, &[("dep", &aux)]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn aux_private_helper_demanded_through_aux_caller_converts() {
    // The guard for the missing-MIR verdict's demand-origin
    // evidence: a generic helper under a private module of
    // the auxiliary crate, demanded through the auxiliary's
    // own public generic caller. The home crate's
    // reachability visitor follows the caller's body, so the
    // helper is in the reachable set, its MIR is encoded
    // natively, and rustc accepts the program.
    let case = Case::new("missing-mir-helper-ok");
    let aux = case.write(
        "guard_aux.rs",
        "mod m {\n    pub fn helper<T>(x: T) -> T { x }\n}\npub fn caller<T>(x: T) -> T { m::helper(x) }\npub fn plain() -> i32 { 7 }\n",
    );
    let src = case.write(
        "case.rs",
        "extern crate dep;\nfn main() {\n    let _ = dep::caller(1i32);\n    let _ = dep::plain();\n}\n",
    );
    let (code, stdout, stderr) = case.convert_aux(&src, &[("dep", &aux)]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn layout_cycle_required_const_rejected() {
    // The layout of `S<S<()>>` does not exist: the field
    // type resolves to the struct itself, one level deeper
    // (Issue #111176). rustc reports the cycle when the
    // collector evaluates the `SizedTypeProperties::SIZE`
    // required constant of the collected `size_of`
    // instance; conversion's own layout query for the type
    // fails behind the speculative discount, so only the
    // recorded failure, replayed at the constant's demand,
    // rejects the program.
    let case = Case::new("layout-cycle-const");
    let src = case.write(
        "case.rs",
        "use std::mem;\n\
         pub struct S<T: Tr> {\n\
             pub f: <T as Tr>::I,\n\
         }\n\
         pub trait Tr {\n\
             type I: Tr;\n\
         }\n\
         impl<T: Tr> Tr for S<T> {\n\
             type I = S<S<T>>;\n\
         }\n\
         impl Tr for () {\n\
             type I = ();\n\
         }\n\
         fn foo<T: Tr>() -> usize {\n\
             mem::size_of::<S<T>>()\n\
         }\n\
         fn main() {\n\
             println!(\"{}\", foo::<S<()>>());\n\
         }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("cycle detected when computing layout"),
        "stderr={stderr}"
    );
    assert!(
        stderr.contains("demands the layout of"),
        "stderr={stderr}"
    );
}

#[test]
fn layout_cycle_post_mono_rejected() {
    // The same cycle, demanded without any measuring
    // constant: collecting `abi::<()>` lays out its
    // signature's `Option<Wrapper<()>>` for the ABI check,
    // and `Wrapper<()>`'s only field is itself. The
    // collector force that replays the check panics on the
    // cycle with an opaque payload; the verdict rests on
    // the recorded failure of a signature type.
    let case = Case::new("layout-cycle-post-mono");
    let src = case.write(
        "case.rs",
        "trait Trait {\n\
             type Assoc;\n\
         }\n\
         impl Trait for () {\n\
             type Assoc = Wrapper<()>;\n\
         }\n\
         struct Wrapper<T: Trait> {\n\
             _x: <T as Trait>::Assoc,\n\
         }\n\
         fn abi<T: Trait>(_: Option<Wrapper<T>>) {}\n\
         fn indirect<T: Trait>() {\n\
             abi::<T>(None);\n\
         }\n\
         fn main() {\n\
             indirect::<()>();\n\
         }\n",
    );
    let (code, _stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 1, "stderr={stderr}");
    assert!(
        stderr.contains("cycle detected when computing layout"),
        "stderr={stderr}"
    );
    assert!(
        stderr.contains("demands the layout of"),
        "stderr={stderr}"
    );
}

#[test]
fn size_of_acyclic_type_converts() {
    // The acceptance beside the required-constant verdict:
    // the same shape with a terminating associated type.
    // The collected `size_of` instance demands the layout
    // of `S<S<()>>`, whose layout exists, so no record
    // exists and nothing fires.
    let case = Case::new("layout-acyclic-sizeof");
    let src = case.write(
        "case.rs",
        "use std::mem;\n\
         pub struct S<T: Tr> {\n\
             pub f: <T as Tr>::I,\n\
         }\n\
         pub trait Tr {\n\
             type I: Tr;\n\
         }\n\
         impl<T: Tr> Tr for S<T> {\n\
             type I = ();\n\
         }\n\
         impl Tr for () {\n\
             type I = ();\n\
         }\n\
         fn foo<T: Tr>() -> usize {\n\
             mem::size_of::<S<T>>()\n\
         }\n\
         fn main() {\n\
             println!(\"{}\", foo::<S<()>>());\n\
         }\n",
    );
    let (code, stdout, stderr) = case.convert(&src, &[], false);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
}

#[test]
fn undemanded_layout_failure_converts() {
    // The acceptance beside both layout verdicts: under
    // the next solver with a tight recursion limit,
    // conversion's speculative layout queries for the
    // coroutine witness types (`ExtData`, `Context`) fail
    // with "queries overflow the depth limit!" and are
    // discounted and recorded — but no collected instance
    // demands those layouts, through a constant or an ABI,
    // and the reference compilation never poses the query
    // either, so the program converts.
    let case = Case::new("layout-undemanded");
    let src = case.write(
        "case.rs",
        "#![recursion_limit = \"6\"]\n\
         async fn foo1() {}\n\
         async fn foo2() {\n\
             foo1().await\n\
         }\n\
         async fn foo3() {\n\
             foo2().await\n\
         }\n\
         async fn foo4() {\n\
             foo3().await\n\
         }\n\
         async fn foo5() {\n\
             foo4().await\n\
         }\n\
         fn assert_send<T: Send>(_: T) {}\n\
         fn main() {\n\
             assert_send(foo5());\n\
         }\n",
    );
    let (code, stdout, stderr) =
        case.convert(&src, &["--rustc-arg", "-Znext-solver"], true);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("converted:"), "stdout={stdout}");
    // The discount was genuinely exercised: the depth-limit
    // diagnostic was emitted (and discounted) on the way.
    assert!(
        stderr.contains("queries overflow the depth limit"),
        "stderr={stderr}"
    );
}
