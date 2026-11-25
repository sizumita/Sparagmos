use super::{Command, aux_spec, fixture, run_driver_with_aux, run_driver_with_aux_specs, temp_dir};


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
