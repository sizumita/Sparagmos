use super::{Command, Path, fixture, run_driver_extra, temp_dir};


const TOOL_GUEST_SRC: &str = r#"
use std::process::Command;

fn main() {
    // The harness passes the tool program's path as the first
    // guest argument; every compile request below is answered
    // by the driver's own conversion machinery (the tool
    // interception), never by a real child process.
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    std::fs::write("t.rs", "pub fn f() -> u32 { 7 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("t.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "good compile failed");
    assert!(std::path::Path::new("libt.rlib").exists(), "rlib artifact missing");

    // dep-info is real text naming the input.
    let out = Command::new(&rustc)
        .arg("--crate-type=rlib")
        .arg("--emit=dep-info")
        .arg("--out-dir")
        .arg(".")
        .arg("t.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "dep-info compile failed");
    let dep = std::fs::read("t.d").unwrap();
    assert!(!dep.is_empty(), "dep-info is empty");

    // A failing source: exit code 1 and diagnostics on stderr.
    std::fs::write("bad.rs", "fn main() { let _x: i32 = \"s\"; }\n").unwrap();
    let out = Command::new(&rustc).arg("bad.rs").output().unwrap();
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    assert!(!out.stderr.is_empty(), "diagnostics missing");

    // `status()` reports the code without an Output value.
    let st = Command::new(&rustc)
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("t.rs")
        .status()
        .unwrap();
    assert!(st.success());

    // A consumer bound through `--extern` resolves the
    // artifact back to its source.
    std::fs::write("u.rs", "extern crate t; fn main() { assert_eq!(t::f(), 7); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--extern")
        .arg("t=libt.rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("u.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "extern consumer failed");

    // `--print` rides the informational path.
    let out = Command::new(&rustc)
        .arg("--print")
        .arg("target-list")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(!out.stdout.is_empty(), "print output missing");

    println!("tool guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_answers_compiler_calls() {
    let dir = temp_dir("tool-intercept");
    let src = fixture(&dir, "tool_guest.rs", TOOL_GUEST_SRC);
    // The tool program is this very driver binary: the string
    // only has to name a program that would start (so the
    // process model lets the call through to the interception),
    // and the handler answers with the driver's own conversion.
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(
        &src,
        &["--tool-program", driver, "--guest-arg", driver],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "tool guest ok\n", "stderr={stderr}");
}

const ARTIFACT_GUEST_SRC: &str = r#"
use std::process::Command;

fn main() {
    // The harness passes the tool program's path as the first
    // guest argument; compiler calls are answered by the tool
    // interception, and running a binary built during the run
    // is answered from the interception's artifact registry:
    // the recorded source is interpreted with the bindings its
    // build invocation carried.
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    // A library, then a binary that uses it through `--extern`.
    std::fs::write("t.rs", "pub fn f() -> u32 { 7 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("t.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "library compile failed");
    std::fs::write(
        "u.rs",
        "extern crate t; fn main() { println!(\"u says {}\", t::f()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--extern")
        .arg("t=libt.rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("u.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "binary compile failed");
    let out = Command::new(dir.join("u")).output().unwrap();
    assert!(out.status.success(), "running u failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "u says 7\n");

    // Arguments, stderr, and `status()` ride the same path.
    std::fs::write(
        "b.rs",
        "fn main() { let a: Vec<String> = std::env::args().collect(); println!(\"b args: {}\", a[1..].join(\",\")); eprintln!(\"b err\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("b.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "b compile failed");
    let out = Command::new(dir.join("b")).arg("x").arg("y").output().unwrap();
    assert!(out.status.success(), "running b failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "b args: x,y\n");
    assert_eq!(String::from_utf8(out.stderr).unwrap(), "b err\n");
    let st = Command::new(dir.join("b")).status().unwrap();
    assert!(st.success(), "status on b failed");

    // A nonzero exit code is the program's own.
    std::fs::write("c.rs", "fn main() { std::process::exit(3); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("c.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "c compile failed");
    let out = Command::new(dir.join("c")).output().unwrap();
    assert_eq!(out.status.code(), Some(3));

    println!("artifact guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_runs_built_binaries() {
    let dir = temp_dir("tool-artifact");
    let src = fixture(&dir, "artifact_guest.rs", ARTIFACT_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The guest's one `status()` run (the `b` binary)
    // inherits the guest's streams, so the binary's own
    // line stands in the guest's stdout ahead of the
    // final marker.
    assert_eq!(stdout, "b args: \nartifact guest ok\n", "stderr={stderr}");
}

const ARTIFACT_FIDELITY_GUEST_SRC: &str = r#"
use std::process::Command;

fn build(rustc: &str, name: &str, src: &str, extra: &[&str]) {
    std::fs::write(format!("{name}.rs"), src).unwrap();
    let mut cmd = Command::new(rustc);
    cmd.arg("--edition=2021").arg("--out-dir").arg(".");
    for a in extra {
        cmd.arg(a);
    }
    cmd.arg(format!("{name}.rs"));
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "build of {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    // The producing invocation's compilation options ride
    // along: a binary built with `--cfg foo` runs its cfg
    // branch when the recipe executes it.
    build(
        &rustc,
        "cfgprobe",
        "fn main() { if cfg!(foo) { println!(\"cfg foo on\"); } else { println!(\"cfg foo off\"); } }\n",
        &["--cfg", "foo"],
    );
    let out = Command::new(dir.join("cfgprobe")).output().unwrap();
    assert!(out.status.success(), "running cfgprobe failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "cfg foo on\n");

    // A built program's exit code 2 is the program's own,
    // not a driver failure: both the `process::exit` form
    // and the `ExitCode` form reach the recipe as code 2.
    build(&rustc, "exit2", "fn main() { std::process::exit(2); }\n", &[]);
    let out = Command::new(dir.join("exit2")).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    build(
        &rustc,
        "exit2b",
        "fn main() -> std::process::ExitCode { std::process::ExitCode::from(2) }\n",
        &[],
    );
    let out = Command::new(dir.join("exit2b")).output().unwrap();
    assert_eq!(out.status.code(), Some(2));

    // The built program's stderr is relayed byte for byte:
    // no newline is invented, CRLF survives, and a guest
    // line that begins with the driver's own prefix is
    // still the guest's.
    build(&rustc, "errprobe", "fn main() { eprint!(\"error\"); }\n", &[]);
    let out = Command::new(dir.join("errprobe")).output().unwrap();
    assert_eq!(out.stderr, b"error".to_vec());
    build(
        &rustc,
        "errprobe2",
        "fn main() { eprint!(\"a\\r\\nb\\n\"); eprintln!(\"sparagmos-driver impostor\"); }\n",
        &[],
    );
    let out = Command::new(dir.join("errprobe2")).output().unwrap();
    assert_eq!(
        out.stderr,
        b"a\r\nb\nsparagmos-driver impostor\n".to_vec()
    );

    // A panicking built program: its own stderr comes
    // first, then the panic report, and the code is the
    // panic code.
    build(
        &rustc,
        "panicprobe",
        "fn main() { eprintln!(\"before panic\"); panic!(\"artifact boom\"); }\n",
        &[],
    );
    let out = Command::new(dir.join("panicprobe")).output().unwrap();
    assert_eq!(out.status.code(), Some(101));
    assert_eq!(
        String::from_utf8(out.stderr).unwrap(),
        format!(
            "before panic\n\nthread 'main' (1) panicked at {}:1:40:\n\
             artifact boom\n\
             note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
            dir.join("panicprobe.rs").display()
        )
    );

    // The built program's argv[0] is the name it was
    // invoked under, not the source path the run is
    // reconstructed from.
    build(
        &rustc,
        "argvprobe",
        "fn main() { println!(\"argv0={}\", std::env::args().next().unwrap()); }\n",
        &[],
    );
    let expect0 = dir.join("argvprobe");
    let out = Command::new(&expect0).output().unwrap();
    assert!(out.status.success(), "running argvprobe failed");
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        format!("argv0={}\n", expect0.display())
    );

    // The compilation environment and the run environment
    // are kept apart: `option_env!` sees the value the
    // build was invoked with, the program's own `env::var`
    // sees the value it was invoked with, and a variable
    // set only for the build does not leak into the run.
    std::fs::write(
        "envprobe.rs",
        "fn main() { println!(\"baked={:?} runtime={:?} leak={:?}\", option_env!(\"BUILD_VALUE\"), std::env::var(\"BUILD_VALUE\").unwrap_or_default(), std::env::var(\"BUILD_ONLY\").ok()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("envprobe.rs")
        .env("BUILD_VALUE", "old")
        .env("BUILD_ONLY", "build-secret")
        .output()
        .unwrap();
    assert!(out.status.success(), "envprobe build failed");
    let out = Command::new(dir.join("envprobe"))
        .env("BUILD_VALUE", "new")
        .output()
        .unwrap();
    assert!(out.status.success(), "running envprobe failed");
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "baked=Some(\"old\") runtime=\"new\" leak=None\n"
    );

    // A built binary denotes the source its build
    // converted: rewriting the source afterwards does not
    // change what an earlier artifact runs, and deleting
    // the source does not orphan it.
    std::fs::write("ver.rs", "fn main() { println!(\"version 1\"); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("-o")
        .arg("ver1")
        .arg("ver.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "ver1 build failed");
    std::fs::write("ver.rs", "fn main() { println!(\"version 2\"); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("-o")
        .arg("ver2")
        .arg("ver.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "ver2 build failed");
    let out = Command::new(dir.join("ver1")).output().unwrap();
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "version 1\n");
    let out = Command::new(dir.join("ver2")).output().unwrap();
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "version 2\n");
    build(&rustc, "gone", "fn main() { println!(\"still here\"); }\n", &[]);
    std::fs::remove_file("gone.rs").unwrap();
    let out = Command::new(dir.join("gone")).output().unwrap();
    assert!(out.status.success(), "running gone failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "still here\n");

    // The same holds for a library the binary was built
    // against: the library's recorded source, not the
    // file as it later stands, is what the binary's run
    // converts.
    std::fs::write("libv.rs", "pub fn f() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("libv.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "libv build failed");
    std::fs::write(
        "useslib.rs",
        "extern crate libv; fn main() { println!(\"lib says {}\", libv::f()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--extern")
        .arg("libv=liblibv.rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("useslib.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "useslib build failed");
    std::fs::write("libv.rs", "pub fn f() -> u32 { 2 }\n").unwrap();
    let out = Command::new(dir.join("useslib")).output().unwrap();
    assert!(out.status.success(), "running useslib failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");

    // A binary keeps the library version it was built
    // against: rebuilding the library at the same output
    // path afterwards does not change what the earlier
    // binary runs.
    build(
        &rustc,
        "libw",
        "pub fn f() -> u32 { 1 }\n",
        &["--crate-type=rlib"],
    );
    build(
        &rustc,
        "usesw",
        "extern crate libw; fn main() { println!(\"lib says {}\", libw::f()); }\n",
        &["--extern", "libw=liblibw.rlib"],
    );
    build(
        &rustc,
        "libw",
        "pub fn f() -> u32 { 2 }\n",
        &["--crate-type=rlib"],
    );
    let out = Command::new(dir.join("usesw")).output().unwrap();
    assert!(out.status.success(), "running usesw failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");

    // Two library outputs built from one source at
    // different times bind by the artifact each binary
    // named, not by whichever artifact the source path
    // resolves to later: a binary linked against the
    // second build runs the second build's code.
    build(
        &rustc,
        "libx",
        "pub fn f() -> u32 { 1 }\n",
        &["--crate-type=rlib", "-o", "libx_a.rlib"],
    );
    build(
        &rustc,
        "libx",
        "pub fn f() -> u32 { 2 }\n",
        &["--crate-type=rlib", "-o", "libx_b.rlib"],
    );
    build(
        &rustc,
        "usesx",
        "extern crate libx; fn main() { println!(\"lib says {}\", libx::f()); }\n",
        &["--extern", "libx=libx_b.rlib"],
    );
    let out = Command::new(dir.join("usesx")).output().unwrap();
    assert!(out.status.success(), "running usesx failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 2\n");

    // A binary bound through the `extern crate` scan (the
    // recipe passes no `--extern`) keeps the library
    // version its build bound: rebuilding the library at
    // the same output path afterwards does not change
    // what the earlier binary runs.
    build(
        &rustc,
        "scanlib",
        "pub fn f() -> u32 { 1 }\n",
        &["--crate-type=rlib"],
    );
    build(
        &rustc,
        "usesscan",
        "extern crate scanlib; fn main() { println!(\"lib says {}\", scanlib::f()); }\n",
        &[],
    );
    build(
        &rustc,
        "scanlib",
        "pub fn f() -> u32 { 2 }\n",
        &["--crate-type=rlib"],
    );
    let out = Command::new(dir.join("usesscan")).output().unwrap();
    assert!(out.status.success(), "running usesscan failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");

    // The same through a source rewrite with no rebuild:
    // the scan-bound pin is the tree the binary's build
    // converted, not the file as it later stands.
    build(
        &rustc,
        "rewlib",
        "pub fn f() -> u32 { 1 }\n",
        &["--crate-type=rlib"],
    );
    build(
        &rustc,
        "usesrew",
        "extern crate rewlib; fn main() { println!(\"lib says {}\", rewlib::f()); }\n",
        &[],
    );
    std::fs::write("rewlib.rs", "pub fn f() -> u32 { 2 }\n").unwrap();
    let out = Command::new(dir.join("usesrew")).output().unwrap();
    assert!(out.status.success(), "running usesrew failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");

    // A built binary denotes its whole source tree, not
    // just the crate root: rewriting the root AND its
    // module file after the build changes nothing the
    // earlier binary runs.
    let mtree = dir.join("mtree");
    std::fs::create_dir_all(&mtree).unwrap();
    std::fs::write(
        mtree.join("modprobe.rs"),
        "mod helper; fn main() { println!(\"value {}\", helper::v()); }\n",
    )
    .unwrap();
    std::fs::write(mtree.join("helper.rs"), "pub fn v() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&mtree)
        .arg(mtree.join("modprobe.rs"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "modprobe build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write(
        mtree.join("modprobe.rs"),
        "mod helper; fn main() { println!(\"value {} newroot\", helper::v()); }\n",
    )
    .unwrap();
    std::fs::write(mtree.join("helper.rs"), "pub fn v() -> u32 { 2 }\n").unwrap();
    let out = Command::new(mtree.join("modprobe")).output().unwrap();
    assert!(out.status.success(), "running modprobe failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "value 1\n");

    // Modifying only the module file leaves the root's
    // bytes untouched, and still must not change what
    // the earlier binary runs: the snapshot compares
    // the whole tree, not the root alone.
    let htree = dir.join("htree");
    std::fs::create_dir_all(&htree).unwrap();
    std::fs::write(
        htree.join("modprobe.rs"),
        "mod helper; fn main() { println!(\"value {}\", helper::v()); }\n",
    )
    .unwrap();
    std::fs::write(htree.join("helper.rs"), "pub fn v() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&htree)
        .arg(htree.join("modprobe.rs"))
        .output()
        .unwrap();
    assert!(out.status.success(), "htree modprobe build failed");
    std::fs::write(htree.join("helper.rs"), "pub fn v() -> u32 { 2 }\n").unwrap();
    let out = Command::new(htree.join("modprobe")).output().unwrap();
    assert!(out.status.success(), "running htree modprobe failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "value 1\n");

    // An `include!` input is part of the tree too: the
    // root that only includes its `main` builds a binary,
    // and modifying only the included file afterwards
    // does not change what it runs.
    let itree = dir.join("itree");
    std::fs::create_dir_all(&itree).unwrap();
    std::fs::write(itree.join("incprobe.rs"), "include!(\"part.rs\");\n").unwrap();
    std::fs::write(
        itree.join("part.rs"),
        "fn main() { println!(\"part 1\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&itree)
        .arg(itree.join("incprobe.rs"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "incprobe build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write(
        itree.join("part.rs"),
        "fn main() { println!(\"part 2\"); }\n",
    )
    .unwrap();
    let out = Command::new(itree.join("incprobe")).output().unwrap();
    assert!(out.status.success(), "running incprobe failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "part 1\n");

    // Deleting the whole tree after the build does not
    // orphan the binary: the snapshot holds every file.
    let dtree = dir.join("dtree");
    std::fs::create_dir_all(&dtree).unwrap();
    std::fs::write(
        dtree.join("modprobe.rs"),
        "mod helper; fn main() { println!(\"value {}\", helper::v()); }\n",
    )
    .unwrap();
    std::fs::write(dtree.join("helper.rs"), "pub fn v() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&dtree)
        .arg(dtree.join("modprobe.rs"))
        .output()
        .unwrap();
    assert!(out.status.success(), "dtree modprobe build failed");
    std::fs::remove_file(dtree.join("modprobe.rs")).unwrap();
    std::fs::remove_file(dtree.join("helper.rs")).unwrap();
    let out = Command::new(dtree.join("modprobe")).output().unwrap();
    assert!(out.status.success(), "running dtree modprobe failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "value 1\n");

    // A library's tree rides its pin the same way:
    // modifying only the library's module file after the
    // binary was built, or deleting the library's tree,
    // does not change what the binary runs.
    let tlib = dir.join("tlib");
    std::fs::create_dir_all(&tlib).unwrap();
    std::fs::write(
        tlib.join("treelib.rs"),
        "mod helper; pub fn f() -> u32 { helper::v() }\n",
    )
    .unwrap();
    std::fs::write(tlib.join("helper.rs"), "pub fn v() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--crate-type=rlib")
        .arg("--out-dir")
        .arg(&tlib)
        .arg(tlib.join("treelib.rs"))
        .output()
        .unwrap();
    assert!(out.status.success(), "treelib build failed");
    std::fs::write(
        tlib.join("uses.rs"),
        "extern crate treelib; fn main() { println!(\"lib says {}\", treelib::f()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--extern")
        .arg(format!("treelib={}", tlib.join("libtreelib.rlib").display()))
        .arg("--out-dir")
        .arg(&tlib)
        .arg(tlib.join("uses.rs"))
        .output()
        .unwrap();
    assert!(out.status.success(), "tlib uses build failed");
    std::fs::write(tlib.join("helper.rs"), "pub fn v() -> u32 { 2 }\n").unwrap();
    let out = Command::new(tlib.join("uses")).output().unwrap();
    assert!(out.status.success(), "running tlib uses failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");
    std::fs::remove_file(tlib.join("treelib.rs")).unwrap();
    std::fs::remove_file(tlib.join("helper.rs")).unwrap();
    let out = Command::new(tlib.join("uses")).output().unwrap();
    assert!(out.status.success(), "running tlib uses after delete failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "lib says 1\n");

    // A bare program name invoked under a
    // command-specific `PATH` denotes the artifact that
    // search finds, not a namesake in the working
    // directory; the spelled name stays its argv[0].
    // Without the `PATH` entry the same bare name does
    // not resolve at all -- the working directory is not
    // searched.
    let pbin = dir.join("pbin");
    std::fs::create_dir_all(&pbin).unwrap();
    std::fs::write(
        pbin.join("pathprobe.rs"),
        "fn main() { println!(\"bin prog argv0={}\", std::env::args().next().unwrap()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&pbin)
        .arg(pbin.join("pathprobe.rs"))
        .output()
        .unwrap();
    assert!(out.status.success(), "pbin pathprobe build failed");
    build(
        &rustc,
        "pathprobe",
        "fn main() { println!(\"cwd prog\"); }\n",
        &[],
    );
    let out = Command::new("pathprobe")
        .env("PATH", &pbin)
        .output()
        .unwrap();
    assert!(out.status.success(), "running pathprobe via PATH failed");
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "bin prog argv0=pathprobe\n"
    );
    assert!(Command::new("pathprobe").output().is_err());

    println!("artifact fidelity guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_built_binary_fidelity() {
    let dir = temp_dir("tool-artifact-fidelity");
    let src = fixture(&dir, "artifact_fidelity_guest.rs", ARTIFACT_FIDELITY_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "artifact fidelity guest ok\n", "stderr={stderr}");
}


const STATUS_RELAY_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "p.rs",
        "fn main() { println!(\"child out\"); eprintln!(\"child err\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("p.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    println!("before");
    let st = Command::new(dir.join("p")).status().unwrap();
    assert!(st.success(), "status failed");
    println!("after");
}"#;

#[test]
fn run_mode_tool_interception_status_relays_inherited_streams() {
    let dir = temp_dir("tool-status-relay");
    let src = fixture(&dir, "status_relay_guest.rs", STATUS_RELAY_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // `status()` inherits the child's streams: the built
    // program's lines stand in the guest's own streams,
    // in invocation order around the guest's prints.
    assert_eq!(stdout, "before\nchild out\nafter\n", "stderr={stderr}");
    assert_eq!(stderr, "child err\n", "stdout={stdout}");
}

const FILE_SNAPSHOT_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    let fbin = dir.join("fbin");
    std::fs::create_dir_all(&fbin).unwrap();
    std::fs::write(
        fbin.join("fprobe.rs"),
        "fn main() { println!(\"{}\", file!()); }\n",
    )
    .unwrap();
    std::fs::write(fbin.join("other.txt"), "one\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&fbin)
        .arg(fbin.join("fprobe.rs"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(fbin.join("fprobe")).output().unwrap();
    assert!(out.status.success(), "first run failed");
    println!("first: {}", String::from_utf8(out.stdout).unwrap().trim());
    // Change a recorded file of the tree (not the root): the
    // next run converts from the snapshot.
    std::fs::write(fbin.join("other.txt"), "two\n").unwrap();
    let out = Command::new(fbin.join("fprobe")).output().unwrap();
    assert!(out.status.success(), "second run failed");
    println!("second: {}", String::from_utf8(out.stdout).unwrap().trim());
}"#;

#[test]
fn run_mode_tool_interception_snapshot_keeps_source_spellings() {
    let dir = temp_dir("tool-file-snapshot");
    let src = fixture(&dir, "file_snapshot_guest.rs", FILE_SNAPSHOT_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The second run converts from the source snapshot
    // (the guest rewrote a file of the tree); its
    // `file!()` spelling stays the recorded source's.
    let probe = dir.join("fbin").join("fprobe.rs");
    assert_eq!(
        stdout,
        format!("first: {0}\nsecond: {0}\n", probe.display()),
        "stderr={stderr}"
    );
}

const CWD_SEPARATION_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "prog.rs",
        "extern crate dep; fn main() { println!(\"prog cwd {}\", std::env::current_dir().unwrap().display()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--extern")
        .arg("dep=libs/libdep.rlib")
        .arg("--out-dir")
        .arg(".")
        .arg("prog.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let other = dir.join("elsewhere");
    std::fs::create_dir_all(&other).unwrap();
    let out = Command::new(dir.join("prog"))
        .current_dir(&other)
        .output()
        .unwrap();
    assert!(out.status.success(), "run failed");
    println!("ran: {}", String::from_utf8(out.stdout).unwrap().trim());
}"#;

#[test]
fn run_mode_tool_interception_artifact_run_separates_directories() {
    let dir = temp_dir("tool-artifact-cwd");
    // A prebuilt dependency the run did not produce: a
    // real rlib from the pinned toolchain's own rustc,
    // which the guest names by a path relative to its
    // build directory.
    let libs = dir.join("libs");
    std::fs::create_dir_all(&libs).unwrap();
    let dep = fixture(&dir, "dep.rs", "pub fn f() -> u32 { 7 }\n");
    let rustc = Path::new(env!("SPARAGMOS_BUILD_SYSROOT"))
        .join("bin")
        .join("rustc");
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--crate-type=rlib")
        .arg("--crate-name")
        .arg("dep")
        .arg("-o")
        .arg(libs.join("libdep.rlib"))
        .arg(&dep)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "dep build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let src = fixture(&dir, "cwd_guest.rs", CWD_SEPARATION_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The run re-converts in the build's directory (the
    // relative `--extern` resolves there) and executes
    // in the directory the invocation requested.
    assert_eq!(
        stdout,
        format!("ran: prog cwd {}\n", dir.join("elsewhere").display()),
        "stderr={stderr}"
    );
}

const NEVER_BUILT_GUEST_SRC: &str = r#"
fn main() {
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    // The file exists and is executable, so the process
    // model's spawn gate passes; the run just never produced
    // it, so the interception cannot answer for it.
    let _out = std::process::Command::new(dir.join("never-built"))
        .output()
        .unwrap();
    println!("never-built guest unexpectedly ran the program");
}
"#;

#[test]
fn run_mode_tool_interception_unproduced_program_stays_unsupported() {
    let dir = temp_dir("tool-unproduced");
    let src = fixture(&dir, "never_built_guest.rs", NEVER_BUILT_GUEST_SRC);
    let fake = dir.join("never-built");
    std::fs::write(&fake, b"").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("is not an artifact this run produced"),
        "stderr={stderr}"
    );
}

const RELATIVE_CHILD_DIR_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::create_dir_all("sub").unwrap();
    std::fs::write("sub/prog.rs", "fn main() { println!(\"child five\"); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg("sub")
        .arg("sub/prog.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new("./prog").current_dir("sub").output().unwrap();
    assert!(out.status.success(), "run failed");
    println!("got: {}", String::from_utf8(out.stdout).unwrap().trim());
}"#;

#[test]
fn run_mode_tool_interception_relative_child_dir_resolves_once() {
    let dir = temp_dir("tool-relative-child-dir");
    let src = fixture(
        &dir,
        "relative_child_dir_guest.rs",
        RELATIVE_CHILD_DIR_GUEST_SRC,
    );
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The spawn resolution applies the child's directory
    // to the spelled `./prog` once: the artifact lookup
    // stands the resolved `sub/./prog` on the directory the
    // resolution ran in, not on the execution directory
    // (which would look under `sub/sub/`).
    assert_eq!(stdout, "got: child five\n", "stderr={stderr}");
}

const STDIO_NULL_GUEST_SRC: &str = r#"use std::process::{Command, Stdio};

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "p.rs",
        "fn main() { println!(\"child out\"); eprintln!(\"child err\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("p.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    println!("before");
    let st = Command::new(dir.join("p"))
        .stdout(Stdio::null())
        .status()
        .unwrap();
    assert!(st.success());
    let st = Command::new(dir.join("p"))
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(st.success());
    println!("after");
}"#;

#[test]
fn run_mode_tool_interception_status_honors_stdio_null() {
    let dir = temp_dir("tool-stdio-null");
    let src = fixture(&dir, "stdio_null_guest.rs", STDIO_NULL_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // Each `status()` call relays only the stream it
    // inherits: the stdout-nulled call contributes its
    // stderr line, the stderr-nulled call its stdout line,
    // and the nulled streams' bytes appear nowhere.
    assert_eq!(stdout, "before\nchild out\nafter\n", "stderr={stderr}");
    assert_eq!(stderr, "child err\n", "stdout={stdout}");
}

const STDIO_OUTPUT_GUEST_SRC: &str = r#"use std::process::{Command, Stdio};

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "p.rs",
        "fn main() { println!(\"child out\"); eprintln!(\"child err\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("p.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(dir.join("p"))
        .stdout(Stdio::inherit())
        .output()
        .unwrap();
    println!(
        "inherit collected stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    println!(
        "inherit collected stderr: {:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(dir.join("p"))
        .stderr(Stdio::null())
        .output()
        .unwrap();
    println!(
        "null collected stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    println!(
        "null collected stderr: {:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(dir.join("p"))
        .stdout(Stdio::piped())
        .output()
        .unwrap();
    println!(
        "piped collected stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}"#;

#[test]
fn run_mode_tool_interception_output_honors_stdio_overrides() {
    let dir = temp_dir("tool-stdio-output");
    let src = fixture(&dir, "stdio_output_guest.rs", STDIO_OUTPUT_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // Under `output()`, an inherited stream joins the
    // guest's own stream instead of the `Output`, a nulled
    // stream is neither collected nor shown, and an
    // explicit pipe collects as the default does.
    assert_eq!(
        stdout,
        "child out\n\
         inherit collected stdout: \"\"\n\
         inherit collected stderr: \"child err\\n\"\n\
         null collected stdout: \"child out\\n\"\n\
         null collected stderr: \"\"\n\
         piped collected stdout: \"child out\\n\"\n",
        "stderr={stderr}"
    );
    assert_eq!(stderr, "", "stdout={stdout}");
}

const NON_UTF8_ARG_GUEST_SRC: &str = r#"use std::os::unix::ffi::OsStringExt;
use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "p.rs",
        "use std::os::unix::ffi::OsStrExt; fn main() { let arg = std::env::args_os().nth(1).unwrap(); println!(\"child arg bytes: {:?}\", arg.as_bytes()); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("p.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let arg = std::ffi::OsString::from_vec(vec![b'a', 0xff, b'b']);
    let out = Command::new(dir.join("p")).arg(arg).output().unwrap();
    assert!(out.status.success(), "run failed");
    // The child observes the argument byte for byte.
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "child arg bytes: [97, 255, 98]\n"
    );
    println!("arg guest ran the program");
}"#;

#[test]
fn run_mode_tool_interception_non_utf8_argument_is_carried() {
    let dir = temp_dir("tool-non-utf8-arg");
    let src = fixture(&dir, "non_utf8_arg_guest.rs", NON_UTF8_ARG_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The argument's bytes are carried to the run unchanged
    // (in an arguments file, since they are not valid UTF-8),
    // and the child observes them byte for byte -- never a
    // lossy replacement. The guest itself asserts the bytes
    // the child prints; the run's success and the guest's
    // completion marker are what is observable here.
    assert_eq!(
        out.status.code(),
        Some(0),
        "stdout={stdout} stderr={stderr}"
    );
    assert_eq!(stdout, "arg guest ran the program\n", "stderr={stderr}");
}

const ADDED_MODULE_CANDIDATE_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    let tree = dir.join("mtree");
    std::fs::create_dir_all(&tree).unwrap();
    std::fs::write(
        tree.join("modprobe.rs"),
        "mod helper; fn main() { println!(\"value {}\", helper::v()); }\n",
    )
    .unwrap();
    std::fs::write(tree.join("helper.rs"), "pub fn v() -> u32 { 1 }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&tree)
        .arg(tree.join("modprobe.rs"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Every recorded file stands unchanged, but a second
    // module candidate appears beside the one the build
    // resolved: the live tree no longer describes the
    // recorded conversion (a fresh conversion of it fails
    // with E0761), while the program the build produced
    // is unchanged.
    std::fs::create_dir_all(tree.join("helper")).unwrap();
    std::fs::write(
        tree.join("helper").join("mod.rs"),
        "pub fn v() -> u32 { 2 }\n",
    )
    .unwrap();
    let out = Command::new(tree.join("modprobe")).output().unwrap();
    assert!(out.status.success(), "run failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "value 1\n");
    println!("added module candidate guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_added_module_candidate_uses_snapshot() {
    let dir = temp_dir("tool-added-module-candidate");
    let src = fixture(
        &dir,
        "added_module_candidate_guest.rs",
        ADDED_MODULE_CANDIDATE_GUEST_SRC,
    );
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The added `helper/mod.rs` ends the recorded tree's
    // intact reading, so the run converts the recorded
    // snapshot (which holds only `helper.rs`) and the
    // built program runs as built.
    assert_eq!(
        stdout, "added module candidate guest ok\n",
        "stderr={stderr}"
    );
}

const OVERSIZED_NEIGHBOUR_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    let tree = dir.join("bigtree");
    std::fs::create_dir_all(&tree).unwrap();
    std::fs::write(
        tree.join("bigprobe.rs"),
        "fn main() { println!(\"value {}\", 7); }\n",
    )
    .unwrap();
    // An unrelated file far past the snapshot byte budget
    // stands in the crate's directory. The build's snapshot
    // of the tree must leave it unread -- its size alone
    // ends the whole-tree reading, before any of its bytes
    // are allocated -- and the artifact must still run as
    // built.
    let blob = std::fs::File::create(tree.join("blob.bin")).unwrap();
    blob.set_len(128 << 20).unwrap();
    drop(blob);
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(&tree)
        .arg(tree.join("bigprobe.rs"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(tree.join("bigprobe")).output().unwrap();
    assert!(out.status.success(), "run failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "value 7\n");
    println!("oversized neighbour guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_oversized_neighbour_file_stays_unread() {
    let dir = temp_dir("tool-oversized-neighbour");
    let src = fixture(
        &dir,
        "oversized_neighbour_guest.rs",
        OVERSIZED_NEIGHBOUR_GUEST_SRC,
    );
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The oversized neighbour ends the snapshot's
    // whole-tree reading on its size alone (the recorded
    // tree keeps the root alone), the intact reading still
    // holds for the recorded root, and the built program
    // runs as built.
    assert_eq!(stdout, "oversized neighbour guest ok\n", "stderr={stderr}");
}

const RENAMED_BINARY_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write("a.rs", "fn main() { println!(\"from a\"); }\n").unwrap();
    std::fs::write("b.rs", "fn main() { println!(\"from b\"); }\n").unwrap();
    for name in ["a", "b"] {
        let out = Command::new(&rustc)
            .arg("--edition=2021")
            .arg("--out-dir")
            .arg(".")
            .arg(format!("{name}.rs"))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "build failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // Move b over a: the file at a's path is now b's file,
    // and executing the path must run b's program, as it
    // would natively.
    std::fs::rename(dir.join("b"), dir.join("a")).unwrap();
    let out = Command::new(dir.join("a")).output().unwrap();
    assert!(
        out.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    println!("ran a: {}", String::from_utf8(out.stdout).unwrap().trim());
}
"#;

#[test]
fn run_mode_tool_interception_renamed_binary_runs_the_replacement() {
    let dir = temp_dir("tool-renamed-binary");
    let src = fixture(&dir, "renamed_binary_guest.rs", RENAMED_BINARY_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(&src, &["--tool-program", driver, "--guest-arg", driver]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    // The registry follows the file's identity across the
    // rename: the path `a` now holds b's file, so the run
    // executes b's program, not the program originally
    // recorded under `a`.
    assert_eq!(stdout, "ran a: from b\n", "stderr={stderr}");
}

const ENV_FILE_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write("p.rs", "fn main() { println!(\"child ran\"); }\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("p.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(dir.join("p")).output().unwrap();
    assert!(out.status.success(), "run failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "child ran\n");
    println!("env file guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_guest_env_file_is_private() {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = temp_dir("tool-env-file");
    let src = fixture(&dir, "env_file_guest.rs", ENV_FILE_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let started = std::time::SystemTime::now();
    // The driver is spawned with its temporary directory
    // pointed at a watch directory of this test's own: the
    // shared temporary directory can hold millions of
    // entries on a busy host, and sweeping it can take
    // longer than the run itself, which would leave the
    // file unobserved even though it stood owner-only for
    // the whole run.
    let watch = dir.join("tmp-watch");
    std::fs::create_dir_all(&watch).unwrap();
    let mut child = Command::new(driver)
        .arg("run")
        .arg(&src)
        .arg("--tool-program")
        .arg(driver)
        .arg("--guest-arg")
        .arg(driver)
        .env("TMPDIR", &watch)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn sparagmos-driver");
    // The guest environment file stands in the driver's
    // temporary directory for the whole artifact run; watch
    // for it while the driver works and record the
    // permissions of every one created during this run.
    let mut seen: Vec<(String, u32)> = Vec::new();
    while child
        .try_wait()
        .expect("failed to poll the driver")
        .is_none()
    {
        if let Ok(entries) = std::fs::read_dir(&watch) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if !name.starts_with("sparagmos-artifact-env-") {
                    continue;
                }
                let Ok(md) = entry.metadata() else { continue };
                if md.modified().map(|m| m < started).unwrap_or(true) {
                    continue; // a leftover of an earlier run
                }
                let mode = md.permissions().mode() & 0o777;
                if !seen.contains(&(name.clone(), mode)) {
                    seen.push((name, mode));
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let out = child
        .wait_with_output()
        .expect("failed to read the driver's output");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "env file guest ok\n", "stderr={stderr}");
    assert!(
        !seen.is_empty(),
        "no guest environment file was observed during the run"
    );
    // The file holds the driver's whole environment,
    // credentials included: every one observed must be
    // owner-only.
    assert!(
        seen.iter().all(|(_, mode)| *mode == 0o600),
        "guest environment files were not owner-only: {seen:?}"
    );
}

const STDERR_FILE_GUEST_SRC: &str = r#"use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();
    std::fs::write(
        "p.rs",
        "fn main() { print!(\"child out\"); eprint!(\"child err\"); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg("p.rs")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(dir.join("p")).output().unwrap();
    assert!(out.status.success(), "run failed");
    // The built program's streams arrive byte for byte:
    // its stderr in particular keeps its missing trailing
    // newline on the way through the stderr channel.
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "child out");
    assert_eq!(String::from_utf8(out.stderr).unwrap(), "child err");
    println!("stderr file guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_guest_stderr_file_is_private() {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = temp_dir("tool-stderr-file");
    let src = fixture(&dir, "stderr_file_guest.rs", STDERR_FILE_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let started = std::time::SystemTime::now();
    // As in the environment-file test: watch a temporary
    // directory of this test's own (see there for why).
    let watch = dir.join("tmp-watch");
    std::fs::create_dir_all(&watch).unwrap();
    let mut child = Command::new(driver)
        .arg("run")
        .arg(&src)
        .arg("--tool-program")
        .arg(driver)
        .arg("--guest-arg")
        .arg(driver)
        .env("TMPDIR", &watch)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn sparagmos-driver");
    // The guest stderr channel stands in the driver's
    // temporary directory for the whole artifact run; watch
    // for it while the driver works and record the
    // permissions of every one created during this run.
    let mut seen: Vec<(String, u32)> = Vec::new();
    while child
        .try_wait()
        .expect("failed to poll the driver")
        .is_none()
    {
        if let Ok(entries) = std::fs::read_dir(&watch) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if !name.starts_with("sparagmos-artifact-stderr-") {
                    continue;
                }
                let Ok(md) = entry.metadata() else { continue };
                if md.modified().map(|m| m < started).unwrap_or(true) {
                    continue; // a leftover of an earlier run
                }
                let mode = md.permissions().mode() & 0o777;
                if !seen.contains(&(name.clone(), mode)) {
                    seen.push((name, mode));
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let out = child
        .wait_with_output()
        .expect("failed to read the driver's output");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "stderr file guest ok\n", "stderr={stderr}");
    assert!(
        !seen.is_empty(),
        "no guest stderr channel was observed during the run"
    );
    // The channel carries the guest's stderr -- the
    // recipe's diagnostics and any secrets among them:
    // every one observed must be owner-only.
    assert!(
        seen.iter().all(|(_, mode)| *mode == 0o600),
        "guest stderr channels were not owner-only: {seen:?}"
    );
}

#[test]
fn run_mode_tool_interception_snapshot_tree_is_private() {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = temp_dir("tool-snapshot-privacy");
    let src = fixture(
        &dir,
        "snapshot_privacy_guest.rs",
        ADDED_MODULE_CANDIDATE_GUEST_SRC,
    );
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let started = std::time::SystemTime::now();
    let mut child = Command::new(driver)
        .arg("run")
        .arg(&src)
        .arg("--tool-program")
        .arg(driver)
        .arg("--guest-arg")
        .arg(driver)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn sparagmos-driver");
    // The guest's added module candidate ends the recorded
    // tree's intact reading, so the artifact run converts a
    // materialised snapshot of the recorded tree under the
    // shared temporary directory. The snapshot copies every
    // recorded neighbouring file verbatim -- a private file
    // beside the crate root among them -- so the whole
    // materialised tree must stand owner-only. It outlives
    // the driver, so sweep for it while the driver works
    // and once more after it exits.
    let sweep = |seen: &mut Vec<(String, u32, bool)>| {
        let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("sparagmos-artifact-sources-") {
                continue;
            }
            let Ok(md) = entry.metadata() else { continue };
            if md.modified().map(|m| m < started).unwrap_or(true) {
                continue; // a leftover of an earlier run
            }
            let mut stack = vec![entry.path()];
            while let Some(p) = stack.pop() {
                let Ok(md) = std::fs::metadata(&p) else {
                    continue;
                };
                let item = (
                    p.to_string_lossy().into_owned(),
                    md.permissions().mode() & 0o777,
                    md.is_dir(),
                );
                if !seen.contains(&item) {
                    seen.push(item);
                }
                if md.is_dir() {
                    if let Ok(entries) = std::fs::read_dir(&p) {
                        stack.extend(entries.flatten().map(|e| e.path()));
                    }
                }
            }
        }
    };
    let mut seen: Vec<(String, u32, bool)> = Vec::new();
    while child
        .try_wait()
        .expect("failed to poll the driver")
        .is_none()
    {
        sweep(&mut seen);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let out = child
        .wait_with_output()
        .expect("failed to read the driver's output");
    sweep(&mut seen);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        stdout, "added module candidate guest ok\n",
        "stderr={stderr}"
    );
    assert!(
        seen.iter().any(|(_, _, is_dir)| *is_dir) && seen.iter().any(|(_, _, is_dir)| !*is_dir),
        "no materialised snapshot tree was observed during the run: {seen:?}"
    );
    assert!(
        seen.iter()
            .all(|(_, mode, is_dir)| *mode == if *is_dir { 0o700 } else { 0o600 }),
        "materialised snapshot entries were not owner-only: {seen:?}"
    );
}

const TEST_CRATE_GUEST_SRC: &str = r##"
use std::process::Command;

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    // A `--test` build of a root with no `main` (the
    // harness supplies it): the primary artifact is the
    // test binary named after the crate, not a library.
    std::fs::write(
        "t.rs",
        "#[test]\nfn a() { assert_eq!(1 + 1, 2); }\n",
    )
    .unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--test")
        .arg("--out-dir")
        .arg(".")
        .arg("t.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "test compile failed");
    assert!(std::path::Path::new("t").exists(), "test binary artifact missing");
    assert!(
        !std::path::Path::new("libt.rlib").exists(),
        "test build recorded a library"
    );

    // The recorded binary runs through the artifact seam:
    // libtest's own main drives the one test.
    let out = Command::new("./t").output().unwrap();
    assert!(out.status.success(), "test binary run failed");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("test result: ok"),
        "libtest summary missing: {text}"
    );
    assert!(text.contains("1 passed"), "pass count missing: {text}");

    println!("test crate guest ok");
}
"##;

#[test]
fn run_mode_tool_interception_test_crate_binary() {
    let dir = temp_dir("tool-test-crate");
    let src = fixture(&dir, "test_crate_guest.rs", TEST_CRATE_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(
        &src,
        &["--tool-program", driver, "--guest-arg", driver],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "test crate guest ok\n", "stderr={stderr}");
}

const ARTIFACT_NAMING_GUEST_SRC: &str = r##"
use std::process::Command;

fn compile(rustc: &str, args: &[&str]) -> std::process::Output {
    Command::new(rustc).args(args).output().unwrap()
}

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    // A crate name declared in the source and a glued
    // `-Cextra-filename` suffix: the artifact file carries
    // both (libdependency2.rlib, not a file named after the
    // source), and a consumer naming that file resolves it.
    std::fs::write(
        "dep.rs",
        "#![crate_name = \"dependency\"]\n#![crate_type = \"rlib\"]\npub fn f() -> u32 { 5 }\n",
    )
    .unwrap();
    let out = compile(&rustc, &["--edition=2021", "dep.rs", "-Cextra-filename=2"]);
    assert!(
        out.status.success(),
        "dep build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        std::path::Path::new("libdependency2.rlib").exists(),
        "declared-name artifact missing"
    );
    std::fs::write(
        "use_dep.rs",
        "extern crate dependency; fn main() { assert_eq!(dependency::f(), 5); }\n",
    )
    .unwrap();
    let out = compile(
        &rustc,
        &["--edition=2021", "use_dep.rs", "--extern", "dependency=libdependency2.rlib"],
    );
    assert!(
        out.status.success(),
        "declared-name consumer failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // A crate type declared in the source: the artifact is
    // the dylib file a real rustc writes for it.
    std::fs::write(
        "dy.rs",
        "#![crate_type = \"dylib\"]\npub fn g() -> u32 { 6 }\n",
    )
    .unwrap();
    let out = compile(&rustc, &["--edition=2021", "dy.rs"]);
    assert!(
        out.status.success(),
        "dylib build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        std::path::Path::new("libdy.so").exists(),
        "declared dylib artifact missing"
    );

    // A metadata-only build writes only the .rmeta (no rlib
    // beside it), and the recorded artifact still binds a
    // consumer -- under the build's own crate type, not an
    // `rmeta` pseudo-type the session would reject.
    std::fs::write("meta.rs", "pub fn h() -> u32 { 7 }\n").unwrap();
    let out = compile(
        &rustc,
        &["--edition=2021", "meta.rs", "--crate-type", "lib", "--emit", "metadata"],
    );
    assert!(
        out.status.success(),
        "metadata build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        std::path::Path::new("libmeta.rmeta").exists(),
        "rmeta artifact missing"
    );
    assert!(
        !std::path::Path::new("libmeta.rlib").exists(),
        "metadata build left an rlib"
    );
    std::fs::write(
        "use_meta.rs",
        "extern crate meta; fn main() { assert_eq!(meta::h(), 7); }\n",
    )
    .unwrap();
    let out = compile(
        &rustc,
        &["--edition=2021", "use_meta.rs", "--extern", "meta=libmeta.rmeta"],
    );
    assert!(
        out.status.success(),
        "metadata consumer failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // An explicit `--extern` is the binding for its name:
    // a source file (or a missing path) named as the extern
    // fails on its own, even though a library of that name
    // was built earlier in the run.
    std::fs::write("bar.rs", "pub fn b() -> u32 { 8 }\n").unwrap();
    let out = compile(&rustc, &["--edition=2021", "bar.rs", "--crate-type", "rlib"]);
    assert!(out.status.success(), "bar build failed");
    std::fs::write(
        "foo.rs",
        "extern crate bar; fn main() { bar::b(); }\n",
    )
    .unwrap();
    let out = compile(&rustc, &["--edition=2021", "foo.rs", "--extern", "bar=foo.rs"]);
    assert!(!out.status.success(), "source-file extern unexpectedly bound");
    let out = compile(&rustc, &["--edition=2021", "foo.rs", "--extern", "bar=no-exist"]);
    assert!(!out.status.success(), "missing extern unexpectedly bound");

    // A chain bound by `extern crate` scanning alone: the
    // middle crate's build named no `--extern`, so the final
    // conversion only finds the leaf if a binding carries
    // the dependencies its own build recorded.
    std::fs::write("leaf.rs", "pub fn l() -> u32 { 9 }\n").unwrap();
    let out = compile(&rustc, &["--edition=2021", "leaf.rs", "--crate-type", "rlib"]);
    assert!(out.status.success(), "leaf build failed");
    std::fs::write(
        "mid.rs",
        "extern crate leaf; pub fn m() -> u32 { leaf::l() }\n",
    )
    .unwrap();
    let out = compile(&rustc, &["--edition=2021", "mid.rs", "--crate-type", "rlib"]);
    assert!(
        out.status.success(),
        "mid build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write(
        "top.rs",
        "extern crate mid; fn main() { assert_eq!(mid::m(), 9); }\n",
    )
    .unwrap();
    let out = compile(&rustc, &["--edition=2021", "top.rs"]);
    assert!(
        out.status.success(),
        "top build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // A crate built with `-Zforce-unstable-if-unmarked`
    // keeps that stability marking when a later invocation
    // binds the artifact: the consumer meets the
    // rustc_private gate the real artifact raises.
    std::fs::write("unstable.rs", "pub fn u() {}\n").unwrap();
    let out = compile(
        &rustc,
        &[
            "--edition=2021",
            "unstable.rs",
            "-Zforce-unstable-if-unmarked",
            "--crate-type",
            "rlib",
        ],
    );
    assert!(
        out.status.success(),
        "unstable build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write("gated.rs", "extern crate alloc; fn main() {}\n").unwrap();
    let out = compile(
        &rustc,
        &["--edition=2021", "gated.rs", "--extern", "alloc=libunstable.rlib"],
    );
    assert!(
        !out.status.success(),
        "force-unstable consumer unexpectedly bound"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("rustc_private"),
        "rustc_private diagnostic missing: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    println!("artifact naming guest ok");
}
"##;

#[test]
fn run_mode_tool_interception_artifact_naming_and_binding() {
    let dir = temp_dir("tool-artifact-naming");
    let src = fixture(&dir, "artifact_naming_guest.rs", ARTIFACT_NAMING_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(
        &src,
        &["--tool-program", driver, "--guest-arg", driver],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "artifact naming guest ok\n", "stderr={stderr}");
}

const DEP_INFO_GUEST_SRC: &str = r#"
use std::process::Command;

fn main() {
    // The harness passes the tool program's path as the first
    // guest argument; every compile request below is answered
    // by the driver's tool interception.
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    // A type error lies past the dep-info phase boundary: the
    // real compiler answers a dep-info-only invocation on
    // this source with success and a `.d` file, and the type
    // error is never even computed, so it cannot appear on
    // stderr either.
    std::fs::write("erroneous.rs", "fn main() {\n    let a: u8 = \"a\";\n}\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--emit=dep-info")
        .arg("erroneous.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "dep-info-only on a type-error source failed");
    assert!(std::path::Path::new("erroneous.d").exists(), "dep-info file missing");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(!stderr.contains("E0308"), "type error leaked into dep-info stderr: {stderr}");

    // A borrow error is past the boundary too.
    std::fs::write("borrowerr.rs", "fn main() {\n    let v = vec![1, 2, 3];\n    let r = &v;\n    let w = v;\n    let _ = (r, w);\n}\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--emit=dep-info")
        .arg("borrowerr.rs")
        .output()
        .unwrap();
    assert!(out.status.success(), "dep-info-only on a borrow-error source failed");

    // A parse error is at the boundary: it still fails, with
    // the diagnostics on stderr.
    std::fs::write("parsebad.rs", "fn main() {\n    let a: u8 = ;\n}\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--emit=dep-info")
        .arg("parsebad.rs")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "dep-info-only on a parse-error source succeeded");
    assert!(!out.stderr.is_empty(), "parse diagnostics missing");

    // An unresolved import is settled by resolution, before
    // the boundary: it still fails.
    std::fs::write("resolvebad.rs", "use totally_nonexistent_crate::Thing;\nfn main() {}\n").unwrap();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--emit=dep-info")
        .arg("resolvebad.rs")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "dep-info-only on an unresolved-import source succeeded");

    // Any further emit kind restores the full judgement: the
    // same type-error source fails with dep-info,link and
    // with metadata alone.
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--emit=dep-info,link")
        .arg("erroneous.rs")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "dep-info,link on a type-error source succeeded");
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("--emit=metadata")
        .arg("erroneous.rs")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "metadata on a type-error source succeeded");

    println!("dep-info guest ok");
}
"#;

#[test]
fn run_mode_tool_interception_dep_info_verdict() {
    let dir = temp_dir("tool-dep-info");
    let src = fixture(&dir, "dep_info_guest.rs", DEP_INFO_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(
        &src,
        &["--tool-program", driver, "--guest-arg", driver],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "dep-info guest ok\n", "stderr={stderr}");
}

const DYLIB_EXISTENCE_GUEST_SRC: &str = r##"
use std::process::Command;

fn build(rustc: &str, name: &str, src: &str) {
    std::fs::write(format!("{name}.rs"), src).unwrap();
    let out = Command::new(rustc)
        .arg("--edition=2021")
        .arg("--out-dir")
        .arg(".")
        .arg(format!("{name}.rs"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build of {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn main() {
    let rustc = std::env::args().nth(1).unwrap();
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().to_path_buf();
    std::env::set_current_dir(&dir).unwrap();

    // A binary linked against a recorded dylib runs while
    // the library stands. Once the recipe removes the
    // library, the run is the loader's failure, as a
    // native execution reports it: a failure status and a
    // diagnostic naming the library that is gone.
    build(
        &rustc,
        "chainlib",
        "#![crate_type = \"dylib\"]\npub fn v() -> u32 { 7 }\n",
    );
    build(
        &rustc,
        "useslib",
        "extern crate chainlib; fn main() { println!(\"v={}\", chainlib::v()); }\n",
    );
    let out = Command::new(dir.join("useslib")).output().unwrap();
    assert!(out.status.success(), "first run of useslib failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "v=7\n");

    std::fs::remove_file("libchainlib.so").unwrap();
    let out = Command::new(dir.join("useslib")).output().unwrap();
    assert!(
        !out.status.success(),
        "run of useslib survived its library's removal"
    );
    assert_eq!(out.status.code(), Some(127));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("libchainlib.so"),
        "loader diagnostic does not name the library: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // A statically linked dependency is not the loader's
    // concern: removing its rlib after the build changes
    // nothing a run can observe.
    build(&rustc, "staticlib", "#![crate_type = \"rlib\"]\npub fn w() -> u32 { 9 }\n");
    build(
        &rustc,
        "usesstatic",
        "extern crate staticlib; fn main() { println!(\"w={}\", staticlib::w()); }\n",
    );
    let out = Command::new(dir.join("usesstatic")).output().unwrap();
    assert!(out.status.success(), "first run of usesstatic failed");
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "w=9\n");

    std::fs::remove_file("libstaticlib.rlib").unwrap();
    let out = Command::new(dir.join("usesstatic")).output().unwrap();
    assert!(
        out.status.success(),
        "run of usesstatic broke with its rlib removed"
    );
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "w=9\n");

    println!("dylib existence guest ok");
}
"##;

#[test]
fn run_mode_tool_interception_deleted_dylib_fails_the_run() {
    let dir = temp_dir("tool-dylib-existence");
    let src = fixture(&dir, "dylib_existence_guest.rs", DYLIB_EXISTENCE_GUEST_SRC);
    let driver = env!("CARGO_BIN_EXE_sparagmos-driver");
    let out = run_driver_extra(
        &src,
        &["--tool-program", driver, "--guest-arg", driver],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "dylib existence guest ok\n", "stderr={stderr}");
}
