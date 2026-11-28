use super::{
    Command, aux_spec, fixture, run_driver, run_driver_with_aux, run_driver_with_aux_specs,
    temp_dir,
};


// ----- auxiliary crates (--aux) ----------------------------------------------

const AUX_BASE_SRC: &str = r#"
pub fn base_val() -> i32 { 40 }
"#;

const AUX_BROKEN_SRC: &str = r#"
pub fn broken( -> i32 { 1 }
"#;

const AUX_LIB_SRC: &str = r#"
fn secret(x: i32) -> i32 { x * 3 }
pub fn add(a: i32, b: i32) -> i32 { a + b }
pub fn triple(x: i32) -> i32 { secret(x) }
pub fn generic_id<T>(x: T) -> T { x }
"#;

const AUX_MAIN_SRC: &str = r#"
extern crate auxlib;
fn main() {
    println!("add={}", auxlib::add(2, 3));
    println!("triple={}", auxlib::triple(7));
}
"#;

const AUX_MID_SRC: &str = r#"
extern crate base;
pub fn with_base() -> i32 { base::base_val() + 2 }
"#;

const AUX_NESTED_MAIN_SRC: &str = r#"
extern crate mid;
fn main() { println!("val={}", mid::with_base()); }
"#;

#[test]
fn run_mode_aux_compile_failure_exits_two() {
    let dir = temp_dir("aux-broken");
    let aux = fixture(&dir, "auxlib.rs", AUX_BROKEN_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "an uncompilable auxiliary crate is a conversion failure; stderr={stderr}"
    );
    assert!(
        stderr.contains("auxiliary crate `auxlib` failed to compile"),
        "the reason names the auxiliary crate: {stderr}"
    );
}

#[test]
fn run_mode_aux_crate_functions_execute() {
    // Functions of an auxiliary crate -- including its private
    // helpers -- are converted into the same program and execute in
    // the interpreter; the crate's generic functions are skipped
    // without affecting the run.
    let dir = temp_dir("aux-basic");
    let aux = fixture(&dir, "auxlib.rs", AUX_LIB_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "aux calls must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "add=5\ntriple=21\n",
        "aux function results reach the guest's stdout"
    );
}


const AUX_STATICS_SRC: &str = r#"
pub static ANSWER: u64 = 42;
pub static NEGATIVE: i32 = -7;
"#;

const AUX_STATICS_MAIN_SRC: &str = r#"
extern crate auxlib;
fn main() {
    println!("{}", auxlib::ANSWER);
    println!("{}", auxlib::NEGATIVE);
}
"#;

#[test]
fn run_mode_aux_crate_statics_print() {
    // Printing statics defined in an auxiliary crate under
    // `-O`: the optimized body references the static directly
    // from the raw `format_args!` argument, and the
    // interpreter's Display recovery reads the static's declared
    // type from the program's static type tables. The driver's
    // item scan records local statics only, so the declared type
    // must be recorded at the reference that names the static --
    // without it this run stops at the Display recovery
    // (Issue #513).
    let dir = temp_dir("aux-statics");
    let aux = fixture(&dir, "auxlib.rs", AUX_STATICS_SRC);
    let src = fixture(&dir, "auxmain.rs", AUX_STATICS_MAIN_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("-O")
        .arg("--aux")
        .arg(format!("auxlib={}", aux.display()))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "aux statics must print; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n-7\n");
}


const AUX_EII_DECL_SRC: &str = r#"
#![feature(extern_item_impls)]

#[eii(eii1)]
pub static DECL1: u64;
"#;

const AUX_EII_DECL_MAIN_SRC: &str = r#"
extern crate decllib as codegen;

#[codegen::eii1]
static EII1_IMPL: u64 = 5;

fn main() {
    println!("{}", EII1_IMPL);
    println!("{}", codegen::DECL1);
}
"#;

#[test]
fn run_mode_aux_crate_eii_decl_reads_impl_seed() {
    // An extern-item-impl (EII) declaration in an auxiliary
    // crate whose implementation is defined in the main crate:
    // reading the declaration must read the implementation's
    // storage. The compiler's registry of implementations is
    // per crate -- the auxiliary crate's map names no
    // implementation for its declaration, because the
    // implementation is recorded in the main crate's map -- so
    // resolving the declaration from its own crate's registry
    // alone finds nothing, the reference falls back to an
    // uninitialized foreign-static seed, and the read stops as
    // an uninitialized-place read (Issue #513, the
    // tests/ui/eii/static/cross_crate_decl shape).
    let dir = temp_dir("aux-eii-decl");
    let aux = fixture(&dir, "decllib.rs", AUX_EII_DECL_SRC);
    let src = fixture(&dir, "eiideclmain.rs", AUX_EII_DECL_MAIN_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("-O")
        .arg("--aux")
        .arg(format!("decllib={}", aux.display()))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the EII declaration must read its implementation's value; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "5\n5\n");
}


const EII_FN_LOCAL_SRC: &str = r#"
#![feature(extern_item_impls)]

mod decl {
    #[eii(eii1)]
    pub fn decl1(x: u64) -> u64 { x + 1 }
}

fn main() {
    println!("{}", decl::decl1(41));
}
"#;

#[test]
fn run_mode_eii_fn_local_default_executes() {
    // An extern-item-impl (EII) function declaration in the
    // crate itself, carrying a default body: the declaration
    // item has no MIR body of its own (the default lives in a
    // compiler-generated item), so the call stopped as a call
    // to a function with no converted body. The registered
    // implementation's body is the declaration's body (the
    // tests/ui/eii/default/local_crate shape, Issue #513).
    let dir = temp_dir("eii-fn-local");
    let src = fixture(&dir, "eiifnlocal.rs", EII_FN_LOCAL_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the EII default body must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}


const EII_FN_AUX_DEFAULT_SRC: &str = r#"
#![feature(extern_item_impls)]

#[eii(eii1)]
pub fn decl1(x: u64) -> u64 { x + 1 }
"#;

const EII_FN_AUX_DEFAULT_MAIN_SRC: &str = r#"
extern crate decllib;

fn main() {
    println!("{}", decllib::decl1(41));
}
"#;

#[test]
fn run_mode_aux_crate_eii_fn_default_executes() {
    // The declaration and its default body both live in an
    // auxiliary crate; the main crate only calls the
    // declaration (the tests/ui/eii/default/call_default
    // shape, Issue #513).
    let dir = temp_dir("eii-fn-aux-default");
    let aux = fixture(&dir, "decllib.rs", EII_FN_AUX_DEFAULT_SRC);
    let src = fixture(&dir, "eiifnauxmain.rs", EII_FN_AUX_DEFAULT_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("decllib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the aux EII default body must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42\n");
}


const EII_FN_AUX_DECL_SRC: &str = r#"
#![feature(extern_item_impls)]

#[eii(eii1)]
pub fn decl1(x: u64) -> u64;
"#;

const EII_FN_AUX_IMPL_MAIN_SRC: &str = r#"
#![feature(extern_item_impls)]

extern crate decllib as codegen;

#[codegen::eii1]
fn eii1_impl(x: u64) -> u64 { x + 100 }

fn main() {
    println!("{}", codegen::decl1(41));
}
"#;

#[test]
fn run_mode_aux_crate_eii_fn_explicit_impl_executes() {
    // The declaration lives in an auxiliary crate and the
    // explicit implementation in the main crate: the
    // compiler's registry of implementations is per crate, so
    // the implementation is recorded only in the main crate's
    // map (the tests/ui/eii/linking/codegen_cross_crate
    // shape, Issue #513).
    let dir = temp_dir("eii-fn-aux-impl");
    let aux = fixture(&dir, "decllib.rs", EII_FN_AUX_DECL_SRC);
    let src = fixture(&dir, "eiifnimplmain.rs", EII_FN_AUX_IMPL_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("decllib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the explicit EII implementation must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "141\n");
}


const EII_FN_AUX_OVERRIDE_MAIN_SRC: &str = r#"
#![feature(extern_item_impls)]

extern crate decllib as codegen;

#[codegen::eii1]
fn eii1_impl(x: u64) -> u64 { x + 100 }

fn main() {
    println!("{}", codegen::decl1(41));
}
"#;

#[test]
fn run_mode_aux_crate_eii_fn_explicit_impl_overrides_default() {
    // The declaration carries a default body in the auxiliary
    // crate and the main crate registers an explicit
    // implementation: the explicit implementation overrides
    // the default (the tests/ui/eii/default/call_impl shape,
    // Issue #513).
    let dir = temp_dir("eii-fn-aux-override");
    let aux = fixture(&dir, "decllib.rs", EII_FN_AUX_DEFAULT_SRC);
    let src = fixture(&dir, "eiifnoverridemain.rs", EII_FN_AUX_OVERRIDE_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("decllib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the explicit EII implementation must override the default; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "141\n");
}


const EII_FN_SCOPED_DECL_SRC: &str = r#"
//@ no-prefer-dynamic
//@[dylib] compile-flags: --crate-type=dylib -Cprefer-dynamic
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

#[eii(eii1)]
pub fn decl1(x: u64) {
    println!("default {x}");
}
"#;

const EII_FN_SCOPED_IMPL_SRC: &str = r#"
//@ no-prefer-dynamic
//@[dylib-impl] compile-flags: --crate-type=dylib -Cprefer-dynamic
//@ aux-build: decl_with_default.rs
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

extern crate decl_with_default as decl;

#[decl::eii1]
fn other(x: u64) {
    println!("1{x}");
}
"#;

const EII_FN_SCOPED_MAIN_SRC: &str = r#"
extern crate decl_with_default;
extern crate impl1;

fn main() {
    decl_with_default::decl1(10);
}
"#;

#[test]
fn run_mode_aux_crate_eii_fn_scoped_crate_type_flags() {
    // The declaration crate and the implementation crate of
    // tests/ui/eii/default/call_impl.rs, whose own
    // `//@[rev] compile-flags:` each scope a dylib build to one
    // revision (the declaration's to a revision this test does
    // not even have). The auxiliary builds applied scoped flags
    // in every revision, so both crates came out as dylibs in
    // both revisions, and rustc's EII final-artifact check
    // counted the default implementation inside the dylib
    // dependency as a second implementation beside the explicit
    // one: the impl1 build failed with "multiple implementations
    // of `#[eii1]`" and the record stopped unsupported. Scoped
    // flags now apply only to their own revision, and the
    // explicit implementation runs in both revisions (the
    // subset runner stamps the active revision as the spec's
    // `;cfg=`, normalized: dashes as underscores).
    for (tag, cfg) in [("rlib", "rlib"), ("dylib", "dylib_impl")] {
        let dir = temp_dir(&format!("eii-fn-scoped-{tag}"));
        let decl = fixture(&dir, "decl_with_default.rs", EII_FN_SCOPED_DECL_SRC);
        let imp = fixture(&dir, "impl1.rs", EII_FN_SCOPED_IMPL_SRC);
        let src = fixture(&dir, "eiifnscopedmain.rs", EII_FN_SCOPED_MAIN_SRC);
        let out = run_driver_with_aux_specs(
            &src,
            None,
            &[
                aux_spec("decl_with_default", &decl, &format!(";cfg={cfg}")),
                aux_spec("impl1", &imp, &format!(";cfg={cfg}")),
            ],
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "revision {tag}: the explicit EII implementation must run; stderr={stderr}"
        );
        assert_eq!(String::from_utf8_lossy(&out.stdout), "110\n");
    }
}


const AUX_PM_SRC: &str = r#"
extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn answer(_input: TokenStream) -> TokenStream {
    "42".parse().unwrap()
}
"#;

const AUX_PM_MAIN_SRC: &str = r#"
extern crate pmaux;
fn main() { println!("answer={}", pmaux::answer!()); }
"#;

// `async` is an identifier only in edition 2015.
const AUX_ED2015_SRC: &str = r#"
pub fn async() -> i32 { 11 }
"#;

// `gen` is a reserved keyword only in edition 2024.
const AUX_ED2021_SRC: &str = r#"
pub fn gen() -> i32 { 7 }
"#;

const AUX_EDITIONS_MAIN_SRC: &str = r#"
extern crate auxa;
extern crate auxb;
fn main() {
    println!("a={}", auxa::r#async());
    println!("b={}", auxb::r#gen());
}
"#;

#[test]
fn run_mode_aux_proc_macro_crate_type_attribute() {
    // A proc-macro auxiliary crate whose source declares no crate
    // type (the shape compiletest's `//@ proc-macro:` produces):
    // the `;crate-type=proc-macro` attribute builds it as a proc
    // macro, and the main crate's macro invocation runs inside the
    // compiler before the guest executes.
    let dir = temp_dir("aux-proc-macro");
    let pm = fixture(&dir, "pmaux.rs", AUX_PM_SRC);
    let src = fixture(&dir, "pmmain.rs", AUX_PM_MAIN_SRC);
    let out = run_driver_with_aux_specs(
        &src,
        None,
        &[aux_spec("pmaux", &pm, ";crate-type=proc-macro")],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "proc-macro aux must build and expand; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "answer=42\n");
}

#[test]
fn run_mode_aux_proc_macro_without_attribute_fails() {
    // The same source without the attribute is built as an rlib,
    // which rustc rejects for `#[proc_macro]` items: a conversion
    // failure (exit 2), not a silent mis-build.
    let dir = temp_dir("aux-proc-macro-plain");
    let pm = fixture(&dir, "pmaux.rs", AUX_PM_SRC);
    let src = fixture(&dir, "pmmain.rs", AUX_PM_MAIN_SRC);
    let out = run_driver_with_aux_specs(&src, None, &[aux_spec("pmaux", &pm, "")]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("auxiliary crate `pmaux` failed to compile"),
        "the reason names the auxiliary crate: {stderr}"
    );
}

#[test]
fn run_mode_aux_per_crate_editions() {
    // Two auxiliary crates with different editions, both different
    // from the main crate's edition 2024: each `;edition=` wins for
    // its own crate (compiletest builds each auxiliary with the
    // edition the auxiliary file itself declares).
    let dir = temp_dir("aux-editions");
    let auxa = fixture(&dir, "auxa.rs", AUX_ED2015_SRC);
    let auxb = fixture(&dir, "auxb.rs", AUX_ED2021_SRC);
    let src = fixture(&dir, "edmain.rs", AUX_EDITIONS_MAIN_SRC);
    let out = run_driver_with_aux_specs(
        &src,
        Some("2024"),
        &[
            aux_spec("auxa", &auxa, ";edition=2015"),
            aux_spec("auxb", &auxb, ";edition=2021"),
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "per-aux editions must apply; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "a=11\nb=7\n");
}

#[test]
fn run_mode_aux_without_edition_attribute_uses_main_edition() {
    // Without `;edition=`, the auxiliary crate is built with the
    // main edition (2024), under which `gen` is reserved: the
    // build fails as a conversion failure.
    let dir = temp_dir("aux-edition-default");
    let auxb = fixture(&dir, "auxb.rs", AUX_ED2021_SRC);
    let src = fixture(&dir, "edmain.rs", AUX_EDITIONS_MAIN_SRC);
    let out = run_driver_with_aux_specs(
        &src,
        Some("2024"),
        &[aux_spec("auxb", &auxb, "")],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("auxiliary crate `auxb` failed to compile"),
        "the reason names the auxiliary crate: {stderr}"
    );
}

#[test]
fn run_mode_aux_namespaced_binding_name() {
    // Lane CI: an `--aux` spec name is an `--extern` binding
    // string, not a crate name: `my_api::utils` is a namespaced
    // binding (compiletest's open-ns tests, enabled by the test's
    // own `-Z namespaced-crates` compile-flag). The crate builds
    // under its file stem and binds under the namespaced name;
    // passing the binding to `--crate-name` made rustc reject
    // the build ("invalid character ':' in crate name").
    let dir = temp_dir("aux-namespaced");
    let aux = fixture(
        &dir,
        "open-ns-my_api_utils.rs",
        "pub fn val() -> i32 { 7 }\n",
    );
    let src = fixture(
        &dir,
        "nsmain2.rs",
        "//@ aux-crate:my_api::utils=open-ns-my_api_utils.rs\n\
         fn main() { println!(\"v={}\", my_api::utils::val()); }\n",
    );
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("-Z")
        .arg("--rustc-arg")
        .arg("namespaced-crates")
        .arg("--aux")
        .arg(aux_spec("my_api::utils", &aux, ""))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "namespaced aux binding must build and run; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=7\n");
}

#[test]
fn run_mode_aux_commented_crate_name_is_not_a_declaration() {
    // Lane CI: `// #![crate_name = "my_api::core"]` at the top of
    // an auxiliary source is a comment, not a declaration. Reading
    // it as one made the driver search for an artifact under the
    // namespaced name and report "produced no linkable artifact"
    // for a build that had in fact succeeded under the file stem.
    let dir = temp_dir("aux-commented-name");
    let aux = fixture(
        &dir,
        "thing-core.rs",
        "// #![crate_name = \"my_api::core\"]\n\
         pub fn val() -> i32 { 9 }\n",
    );
    let src = fixture(
        &dir,
        "cmain.rs",
        "extern crate thing_core;\n\
         fn main() { println!(\"v={}\", thing_core::val()); }\n",
    );
    let out = run_driver_with_aux_specs(
        &src,
        None,
        &[aux_spec("thing_core", &aux, "")],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "commented crate_name must be ignored; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=9\n");
}

#[test]
fn run_mode_aux_build_name_binding() {
    // Lane CI round 2: a consumer can declare an auxiliary crate
    // by the name the crate's own source declares
    // (`#![crate_name = "buildname"]`), not only by the `--aux`
    // spec name (the svh crates and issue-6919 shape). Binding
    // only the spec name left the `extern crate buildname;` item
    // unresolved (E0463).
    let dir = temp_dir("aux-build-name");
    let aux = fixture(
        &dir,
        "iss-x.rs",
        "#![crate_name = \"buildname\"]\npub fn val() -> i32 { 33 }\n",
    );
    let src = fixture(
        &dir,
        "bmain.rs",
        "extern crate buildname;\n\
         fn main() { println!(\"v={}\", buildname::val()); }\n",
    );
    let out = run_driver_with_aux(&src, &[("iss_x", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "build-name declaration must bind; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=33\n");
}

#[test]
fn run_mode_aux_extra_filename_artifact() {
    // Lane CI round 2: an auxiliary source whose own
    // compile-flags carry `-C extra-filename=-1` (the
    // crateresolve shape) produces `lib<name>-1.rlib`; the
    // artifact search only recognised `lib<name>.rlib` and
    // reported the successful build as producing no linkable
    // artifact.
    let dir = temp_dir("aux-extra-filename");
    let aux = fixture(
        &dir,
        "xflag.rs",
        "//@ compile-flags: -C extra-filename=-1\n\
         pub fn val() -> i32 { 5 }\n",
    );
    let src = fixture(
        &dir,
        "xmain.rs",
        "extern crate xflag;\n\
         fn main() { println!(\"v={}\", xflag::val()); }\n",
    );
    let out = run_driver_with_aux(&src, &[("xflag", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "extra-filename artifact must be found; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "v=5\n");
}

const AUX_DUP_NAME_SRC_1: &str = r#"
//@ compile-flags: -C extra-filename=-1
#![crate_name = "crateresolve1"]
#![crate_type = "lib"]
pub fn f() -> isize { 10 }
"#;

const AUX_DUP_NAME_SRC_2: &str = r#"
//@ compile-flags: -C extra-filename=-2
#![crate_name = "crateresolve1"]
#![crate_type = "lib"]
pub fn f() -> isize { 20 }
"#;

const AUX_DUP_NAME_SRC_3: &str = r#"
//@ compile-flags: -C extra-filename=-3
#![crate_name = "crateresolve1"]
#![crate_type = "lib"]
pub fn f() -> isize { 30 }
"#;

const AUX_DUP_NAME_MAIN_SRC: &str = r#"
extern crate crateresolve1;
fn main() {}
"#;

#[test]
fn run_mode_aux_duplicate_name_extra_filename_rejected() {
    // Three auxiliary crates share one crate name and are
    // distinguished only by `-C extra-filename` (the
    // crateresolve shape behind tests/ui/error-codes/E0464.rs
    // and E0523.rs): the main compilation must see all three
    // artifacts as separate candidates and refuse with E0464.
    // The artifact location used to keep the first directory
    // entry on ranking ties, so on a filesystem whose readdir
    // order favoured it every build was handed the SAME
    // artifact -- one candidate, no error -- and the two
    // records flipped with the machine they ran on.
    let dir = temp_dir("aux-dup-name");
    let aux1 = fixture(&dir, "crateresolve1-1.rs", AUX_DUP_NAME_SRC_1);
    let aux2 = fixture(&dir, "crateresolve1-2.rs", AUX_DUP_NAME_SRC_2);
    let aux3 = fixture(&dir, "crateresolve1-3.rs", AUX_DUP_NAME_SRC_3);
    let src = fixture(&dir, "dupmain.rs", AUX_DUP_NAME_MAIN_SRC);
    let out = run_driver_with_aux(
        &src,
        &[
            ("crateresolve1_1", &aux1),
            ("crateresolve1_2", &aux2),
            ("crateresolve1_3", &aux3),
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "duplicate candidates must fail the compilation; stderr={stderr}"
    );
    assert!(
        stderr.contains("E0464") && stderr.contains("multiple candidates"),
        "the failure is the duplicate-candidate refusal: {stderr}"
    );
}

#[test]
fn run_mode_aux_distinct_names_with_extra_filename_bind_their_own() {
    // Positive control for the duplicate-name case above:
    // distinct crate names that each carry an extra-filename
    // still bind their OWN artifacts -- the values differ, so a
    // crossed binding would print the wrong sum.
    let dir = temp_dir("aux-distinct-extra-filename");
    let aux_a = fixture(
        &dir,
        "ponda.rs",
        "//@ compile-flags: -C extra-filename=-a\npub fn val() -> i32 { 3 }\n",
    );
    let aux_b = fixture(
        &dir,
        "pondb.rs",
        "//@ compile-flags: -C extra-filename=-b\npub fn val() -> i32 { 4 }\n",
    );
    let src = fixture(
        &dir,
        "distmain.rs",
        "extern crate ponda;\nextern crate pondb;\n\
         fn main() { println!(\"sum={}\", ponda::val() + pondb::val()); }\n",
    );
    let out = run_driver_with_aux(&src, &[("ponda", &aux_a), ("pondb", &aux_b)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "distinct-name extra-filename artifacts must bind; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "sum=7\n");
}

#[test]
fn run_mode_aux_modified_binding_stays_out_of_prelude() {
    // Lane CI round 2: an `aux-crate:` binding with modifiers
    // (`priv,noprelude:somedep`) is the binding itself — the
    // driver must not ALSO bind the crate under its unmodified
    // build name, or the crate lands in the extern prelude the
    // `noprelude` modifier exists to keep it out of, and
    // tests/ui/extern-flag/multiple-opts.rs stops failing the
    // way it must. The `extern crate somedep;` item inside the
    // module still resolves through the modified binding; the
    // prelude use in `main` cannot, so compilation fails
    // (run-mode exit 2).
    let dir = temp_dir("aux-modified-binding");
    let aux = fixture(
        &dir,
        "somedep.rs",
        "pub struct S;\npub fn somefun() {}\n",
    );
    let src = fixture(
        &dir,
        "momain.rs",
        "//@ aux-crate:priv,noprelude:somedep=somedep.rs\n\
         #![warn(exported_private_dependencies)]\n\
         pub mod m {\n\
             extern crate somedep;\n\
             pub struct PublicType { pub field: somedep::S }\n\
         }\n\
         fn main() { somedep::somefun(); }\n",
    );
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--rustc-arg")
        .arg("-Z")
        .arg("--rustc-arg")
        .arg("unstable-options")
        .arg("--aux")
        .arg(aux_spec("priv,noprelude:somedep", &aux, ""))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a noprelude binding must keep the crate out of the prelude; stderr={stderr}"
    );
    assert!(
        stderr.contains("cannot find"),
        "the failure is the unresolved prelude use: {stderr}"
    );
}

#[test]
fn run_mode_nested_aux_crates_execute() {
    // Auxiliary crates are passed in dependency order: `mid` uses
    // `base`, and the main crate uses `mid`.
    let dir = temp_dir("aux-nested");
    let base = fixture(&dir, "base.rs", AUX_BASE_SRC);
    let mid = fixture(&dir, "mid.rs", AUX_MID_SRC);
    let src = fixture(&dir, "nestedmain.rs", AUX_NESTED_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("base", &base), ("mid", &mid)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "nested aux calls must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "val=42\n");
}

const AUX_PREBUILT_STUB_SRC: &str = r#"
pub fn stub_val() -> i32 { 7 }
"#;

const AUX_USES_PREBUILT_SRC: &str = r#"
extern crate stub;

pub fn dep_val() -> i32 { stub::stub_val() + 1 }
"#;

const AUX_USES_PREBUILT_MAIN_SRC: &str = r#"
extern crate depaux;

fn main() {
    println!("val={}", depaux::dep_val());
}
"#;

#[test]
fn run_mode_aux_crate_binds_prebuilt_aux_lib() {
    // A prebuilt library auxiliary (--aux-lib) precedes the
    // auxiliary builds: an auxiliary source that declares it
    // (`extern crate stub;` here; `extern crate minicore;` in an
    // `//@ add-minicore` test's auxiliary file) binds it during
    // its own build, not only in the main compilation. Without
    // the prebuilt entries in the auxiliary builds' visibility
    // list, this run stops at the auxiliary build with E0463.
    // The stub rlib is built the way the driver's own auxiliary
    // builds are (MIR encoded), with the pinned toolchain the
    // harness's native side invokes as well.
    let dir = temp_dir("aux-prebuilt");
    let stub_src = fixture(&dir, "stub.rs", AUX_PREBUILT_STUB_SRC);
    let rlib = dir.join("libstub.rlib");
    let build = Command::new("rustup")
        .args(["run", "1.100.0", "rustc"])
        .arg(&stub_src)
        .arg("--crate-type")
        .arg("rlib")
        .arg("--crate-name")
        .arg("stub")
        .arg("-Zalways-encode-mir")
        .arg("-o")
        .arg(&rlib)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to build the prebuilt stub");
    assert!(
        build.status.success(),
        "the prebuilt stub must compile: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let dep = fixture(&dir, "depaux.rs", AUX_USES_PREBUILT_SRC);
    let src = fixture(&dir, "prebuiltmain.rs", AUX_USES_PREBUILT_MAIN_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--aux")
        .arg(format!("depaux={}", dep.display()))
        .arg("--aux-lib")
        .arg(format!("stub={};collect=no", rlib.display()))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an auxiliary crate must bind a prebuilt auxiliary it declares; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "val=8\n");
}


#[test]
fn run_mode_aux_prebuilt_display_impl_demanded_through_to_string() {
    // A generic `Display` impl that lives in a prebuilt library
    // crate and is referenced only from inside alloc's
    // `ToString::to_string` blanket body: the blanket body is
    // builtin-served and never converted, so unless the reach
    // walk walks it, the impl's body is never demanded and
    // formatting the value stops at the structural fallback
    // ("Display formatting for an ADT value"). The mid
    // auxiliary plays the support crate's role (it calls
    // `to_string` on the prebuilt crate's type, once by value
    // and once through a `&mut` receiver); the prebuilt rlib
    // plays the dependency crate's role.
    let dir = temp_dir("aux-prebuilt-display");
    let dep_src = fixture(&dir, "dispdep.rs", AUX_DISPLAY_DEP_SRC);
    let rlib = dir.join("libdispdep.rlib");
    let build = Command::new("rustup")
        .args(["run", "1.100.0", "rustc"])
        .arg(&dep_src)
        .arg("--crate-type")
        .arg("rlib")
        .arg("--crate-name")
        .arg("dispdep")
        .arg("-Zalways-encode-mir")
        .arg("-o")
        .arg(&rlib)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to build the prebuilt dependency");
    assert!(
        build.status.success(),
        "the prebuilt dependency must compile: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let mid = fixture(&dir, "dispmid.rs", AUX_DISPLAY_MID_SRC);
    let src = fixture(&dir, "dispmain.rs", AUX_DISPLAY_MAIN_SRC);
    let out = Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(&src)
        .arg("--aux")
        .arg(format!("dispmid={}", mid.display()))
        .arg("--aux-lib")
        .arg(format!("dispdep={}", rlib.display()))
        .output()
        .expect("failed to run sparagmos-driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a Display impl referenced only through the builtin-served \
         to_string blanket body must still be demanded; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "wrap:abc:3|wrap:xy:1\n");
}

const AUX_DISPLAY_DEP_SRC: &str = r#"
#![crate_name = "dispdep"]

use std::fmt;

pub struct Wrapper<T: AsRef<str>> {
    pub inner: T,
    pub n: usize,
}

impl<T: AsRef<str>> fmt::Display for Wrapper<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wrap:{}:{}", self.inner.as_ref(), self.n)
    }
}

"#;

const AUX_DISPLAY_MID_SRC: &str = r#"
#![crate_name = "dispmid"]

extern crate dispdep;

pub fn render() -> String {
    let w = dispdep::Wrapper { inner: "abc".to_string(), n: 3 };
    let mut w2 = dispdep::Wrapper { inner: "xy".to_string(), n: 1 };
    let header = &mut w2;
    format!("{}|{}", w.to_string(), header.to_string())
}
"#;

const AUX_DISPLAY_MAIN_SRC: &str = r#"
extern crate dispmid;

fn main() {
    println!("{}", dispmid::render());
}
"#;

const AUX_SAMENAME_DROP_SRC: &str = r#"
#![crate_name = "sharedname"]

pub struct Guard<X: std::fmt::Debug> { pub val: X }

impl<X: std::fmt::Debug> Drop for Guard<X> {
    fn drop(&mut self) { println!("dropping {:?}", self.val); }
}

pub struct Plain(pub u32);

impl Drop for Plain {
    fn drop(&mut self) { println!("dropping plain {}", self.0); }
}

pub mod lib {
    pub struct X(pub u8);

    impl Drop for X {
        fn drop(&mut self) {
            assert_eq!(self.0, 5);
            println!("x dropped with {}", self.0);
        }
    }
}
"#;

const AUX_SAMENAME_DROP_MAIN_SRC: &str = r#"
extern crate sharedname as aux;
fn main() {
    let _g = aux::Guard { val: 7u32 };
    let _p = aux::Plain(3);
    println!("body");
}
"#;

const AUX_SAMENAME_MODULE_DROP_MAIN_SRC: &str = r#"
extern crate sharedname as aux;
fn main() {
    {
        let _x = aux::lib::X(5);
        println!("inside");
    }
    println!("after");
}
"#;

#[test]
fn run_mode_aux_same_crate_name_drops_run() {
    // An auxiliary crate carrying the entry crate's own name
    // (the test harness's aux-build convention): its types
    // render under the local crate's name, so the drop demand
    // dispatched on the rendered name alone looked for their
    // `Drop` impls among the local crate's items only, never
    // demanded the destructor bodies, and the run stopped at
    // the drop with an unknown function (Issue #513: the
    // cross-crate `Drop` records). Both the generic and the
    // concrete destructor must run, in drop order.
    let dir = temp_dir("aux-samename-drop");
    let aux = fixture(&dir, "sharedname-aux.rs", AUX_SAMENAME_DROP_SRC);
    let src = fixture(&dir, "sharedname.rs", AUX_SAMENAME_DROP_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("sharedname", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "same-named aux drops must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "body\ndropping plain 3\ndropping 7\n",
        "the auxiliary destructors run in drop order"
    );
}

#[test]
fn run_mode_aux_same_crate_name_module_drop_runs() {
    // The same-named auxiliary crate's type in a submodule,
    // dropped at the end of a block scope.
    let dir = temp_dir("aux-samename-module-drop");
    let aux = fixture(&dir, "sharedname-aux.rs", AUX_SAMENAME_DROP_SRC);
    let src = fixture(&dir, "sharedname.rs", AUX_SAMENAME_MODULE_DROP_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("sharedname", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "same-named aux module drop must execute; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "inside\nx dropped with 5\nafter\n",
        "the destructor runs at the end of the block"
    );
}

const AUX_CTOR_SRC: &str = r#"
#[derive(Debug, PartialEq)]
pub struct RemoteC(pub u32);

#[derive(Debug, PartialEq)]
pub struct RemoteG<T>(pub T);
"#;

const AUX_CTOR_MAIN_SRC: &str = r#"
extern crate auxlib;
use auxlib::{RemoteC, RemoteG};
fn main() {
    let virtual_remotec: &dyn Fn(_) -> RemoteC = &RemoteC;
    assert_eq!(virtual_remotec(1), RemoteC(1));
    let virtual_remoteg: &dyn Fn(_) -> RemoteG<u32> = &RemoteG;
    assert_eq!(virtual_remoteg(1), RemoteG(1));
    println!("ctor ok");
}
"#;

#[test]
fn run_mode_aux_ctor_as_dyn_fn_value() {
    // The issue-39823 shape: an auxiliary crate's tuple-struct
    // constructor coerced to a `&dyn Fn` value. The mention
    // reaches the driver inside a promoted constant, and a
    // foreign constructor has no MIR body in the crate metadata,
    // so the constant-function collection pass synthesizes the
    // constructor body the same way the reach walk does.
    let dir = temp_dir("aux-ctor-value");
    let aux = fixture(&dir, "auxlib.rs", AUX_CTOR_SRC);
    let src = fixture(&dir, "ctor_main.rs", AUX_CTOR_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("auxlib", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an aux constructor used as a dyn Fn value must execute; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ctor ok\n");
}


const AUX_TRAIT_DEP_SRC: &str = r#"
pub trait Trait {}
pub struct OtherType;
pub fn do_something<T: Trait>(_: T) {}
"#;

const AUX_TRAIT_FAIL_MAIN_SRC: &str = r#"
extern crate dependency;
use dependency::{OtherType, do_something};
fn main() {
    do_something(OtherType);
}
"#;

const AUX_TRAIT_CLEAN_MAIN_SRC: &str = r#"
extern crate dependency;
use dependency::do_something;
struct MyType;
impl dependency::Trait for MyType {}
fn main() {
    do_something(MyType);
    println!("clean ok");
}
"#;

#[test]
fn run_mode_failing_crate_diagnostic_uses_trimmed_def_paths() {
    // A converted crate's session carries the trimmed def-path
    // map (as rustc's own entry configures it), so the
    // diagnostics it emits print the visible spelling of an
    // imported item: the trait-bound note names the bound's
    // function `do_something`, not `dependency::do_something`,
    // exactly as the native compiler prints it.
    let dir = temp_dir("aux-trimmed-diag");
    let aux = fixture(&dir, "dependency.rs", AUX_TRAIT_DEP_SRC);
    let src = fixture(&dir, "failmain.rs", AUX_TRAIT_FAIL_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("dependency", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "a trait-bound failure is a conversion failure; stderr={stderr}"
    );
    assert!(
        stderr.contains("required by a bound in `do_something`"),
        "the bound note uses the trimmed spelling: {stderr}"
    );
    assert!(
        !stderr.contains("required by a bound in `dependency::do_something`"),
        "the bound note must not use the full def path: {stderr}"
    );
}

#[test]
fn run_mode_clean_conversion_stays_clear_of_the_trimmed_paths_check() {
    // Evaluating the trimmed-path map is recorded by the
    // diagnostic context as a promise that a diagnostic
    // follows, and a clean conversion breaks that promise: it
    // prints types for its own naming without emitting one.
    // The conversion therefore runs under
    // `with_no_trimmed_paths!`, and converting a clean crate
    // that exercises generic calls into an auxiliary crate
    // must not trip the teardown panic.
    let dir = temp_dir("aux-trimmed-clean");
    let aux = fixture(&dir, "dependency.rs", AUX_TRAIT_DEP_SRC);
    let src = fixture(&dir, "cleanmain.rs", AUX_TRAIT_CLEAN_MAIN_SRC);
    let out = run_driver_with_aux(&src, &[("dependency", &aux)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a clean conversion must succeed; stderr={stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "clean ok\n");
    assert!(
        !stderr.contains("diagnostics were expected"),
        "the trimmed-paths teardown check must not fire: {stderr}"
    );
}
