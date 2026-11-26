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
//! (the LLVM verifier's rejection of the module). The driver now
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
