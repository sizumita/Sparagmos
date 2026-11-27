use super::{fixture, run_driver, run_driver_extra, run_ok, temp_dir};


const ENV_ARGS_SRC: &str = r#"
fn main() {
    for a in std::env::args() {
        println!("arg:{a}");
    }
    let mut os = std::env::args_os();
    println!("os-first {}", os.next().is_some());
    println!("count {}", std::env::args().count());
}
"#;

#[test]
fn run_mode_env_args_reach_the_guest() {
    let dir = temp_dir("env-args");
    let src = fixture(&dir, "args.rs", ENV_ARGS_SRC);
    let out = run_driver_extra(&src, &["--guest-arg", "one", "--guest-arg", "two"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 5, "stdout={stdout}");
    // argv[0] is the guest's program path: the input source.
    assert!(lines[0].starts_with("arg:"), "stdout={stdout}");
    assert!(lines[0].ends_with("args.rs"), "stdout={stdout}");
    assert_eq!(lines[1], "arg:one");
    assert_eq!(lines[2], "arg:two");
    assert_eq!(lines[3], "os-first true");
    assert_eq!(lines[4], "count 3");
}

const RESULT_ERR_RANGE_FROM_SRC: &str = r#"
use std::ops::RangeFrom;
use std::path::PathBuf;

fn main() -> Result<(), RangeFrom<PathBuf>> {
    Err(RangeFrom {
        start: PathBuf::from("hello"),
    })
}
"#;

#[test]
fn run_mode_termination_range_from_pathbuf_report() {
    // The report renders the payload through the detached
    // `Debug` path (the payload's own formatting is never
    // demanded by the guest): a `RangeFrom` payload keeps
    // its handwritten form, `"hello"..`, rather than
    // collapsing to the held byte string.
    let dir = temp_dir("termination-range-from");
    let src = fixture(&dir, "err_range_from.rs", RESULT_ERR_RANGE_FROM_SRC);
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(1), "Result::Err(RangeFrom<PathBuf>)");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "Error: \"hello\"..\n"
    );
    assert!(
        out.stdout.is_empty(),
        "the report writes nothing to stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

// ---- Lane HU: body-less call models (issue #130) ----
// A trait method named by its plain trait path reaches the
// interpreter with no body under the name when it arrives as a
// function value or inside a body the converter kept in generic
// form; the value-semantics builtins serve those calls at run
// time. `HashMap::new` draws its `RandomState` keys from the
// modelled `hashmap_random_keys`.

const CLONED_SUM_SRC: &str = r#"
fn main() {
    let v = vec![1i64, 2, 3];
    let s: i64 = v.iter().cloned().sum();
    assert_eq!(s, 6);
    println!("ok");
}
"#;

#[test]
fn run_mode_cloned_sum_serves_trait_path_clone() {
    let dir = temp_dir("cloned-sum");
    let src = fixture(&dir, "cloned_sum.rs", CLONED_SUM_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const PATH_TO_STR_SRC: &str = r#"
use std::path::Path;

fn main() {
    // `Path::to_str` goes through `str::from_utf8` on the
    // path's bytes.
    let p = Path::new("abc");
    assert_eq!(p.to_str().unwrap(), "abc");
    let ok = std::str::from_utf8(b"abc").unwrap();
    assert_eq!(ok, "abc");
    let bad = std::str::from_utf8(&[0x61, 0xFF, 0x62]).unwrap_err();
    assert_eq!(bad.valid_up_to(), 1);
    assert_eq!(bad.error_len(), Some(1));
    let incomplete = std::str::from_utf8(&[0x61, 0xE3, 0x81]).unwrap_err();
    assert_eq!(incomplete.valid_up_to(), 1);
    assert_eq!(incomplete.error_len(), None);
    println!("path-to-str-ok");
}
"#;

#[test]
fn run_mode_path_to_str_and_from_utf8() {
    let dir = temp_dir("path-to-str");
    let src = fixture(&dir, "path_to_str.rs", PATH_TO_STR_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Path::to_str and str::from_utf8 run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "path-to-str-ok\n",
        "utf8 validation results and error fields match std"
    );
}

const PATH_DISPLAY_FMT_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

fn main() {
    // The `fmt` bodies of the path / OS-string types are
    // foreign code, so their text comes from the byte
    // content the values hold: `Display` adapters forward
    // to the bytes (lossy), `Debug` is the `OsStr` form,
    // and neither is the aggregate expansion.
    let p = Path::new("/tmp/foo bar.txt");
    println!("{}", p.display());
    println!("{:?}", p);
    println!("{:?}", p.display());
    let pb = PathBuf::from("/tmp/foo bar.txt");
    println!("{}", pb.display());
    println!("{:?}", pb);
    let os = OsStr::new("hello");
    println!("{}", os.display());
    println!("{:?}", os);
    let osn = OsString::from("hello");
    println!("{}", osn.display());
    println!("{:?}", osn);
    println!("{:>24}", p.display());
    println!("{:>24?}", pb);
    println!("{}", p.parent().unwrap().display());
    println!("{}", p.to_string_lossy());
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let bad = OsStr::from_bytes(b"/tmp/a\xFFb\x80c");
        println!("{}", bad.display());
        println!("{:?}", bad);
    }
    let w = std::num::Wrapping(42u32);
    println!("{}", w);
    println!("{:x}", w);
    let r = std::cmp::Reverse(7);
    println!("{:?}", r);
}
"#;

#[test]
fn run_mode_path_display_and_os_str_formatting() {
    let dir = temp_dir("path-display-fmt");
    let src = fixture(&dir, "path_display_fmt.rs", PATH_DISPLAY_FMT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Path/OsStr Display and Debug formatting runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "/tmp/foo bar.txt\n\"/tmp/foo bar.txt\"\n\"/tmp/foo bar.txt\"\n/tmp/foo bar.txt\n\"/tmp/foo bar.txt\"\nhello\n\"hello\"\nhello\n\"hello\"\n        /tmp/foo bar.txt\n\"/tmp/foo bar.txt\"\n/tmp\n/tmp/foo bar.txt\n/tmp/a\u{FFFD}b\u{FFFD}c\n\"/tmp/a\\xFFb\\x80c\"\n42\n2a\nReverse(7)\n",
        "path and OS-string formatting matches std byte for byte"
    );
}

const PATH_DISPLAY_PRECISION_SRC: &str = r#"
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

fn main() {
    // Precision and width on the `display()` adapters: the
    // adapters' `Display` text is the lossy content rendered
    // under the caller's options, so precision truncates and
    // width pads, exactly as `str` itself formats (measured
    // against native rustc 1.100.0).
    let os = OsStr::new("hello");
    println!("[{:.1}]", os.display());
    println!("[{:.0}]", os.display());
    println!("[{:>10.1}]", os.display());
    println!("[{:>10}]", os.display());
    let p = Path::new("hello");
    println!("[{:.1}]", p.display());
    println!("[{:>10.1}]", p.display());
    println!("[{:.3}]", PathBuf::from("hello world").display());
    println!("[{:.2}]", OsString::from("hello").display());
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let bad = OsStr::from_bytes(b"a\xFFbcdef");
        println!("[{:.2}]", bad.display());
        println!("[{:>12.2}]", bad.display());
    }
}
"#;

#[test]
fn run_mode_path_display_precision_and_width() {
    let dir = temp_dir("path-display-precision");
    let src = fixture(
        &dir,
        "path_display_precision.rs",
        PATH_DISPLAY_PRECISION_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "display-adapter precision and width runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "[h]\n[]\n[         h]\n[     hello]\n[h]\n[         h]\n[hel]\n[he]\n[a\u{FFFD}]\n[          a\u{FFFD}]\n",
        "display adapters truncate by precision and pad by width like str"
    );
}

const PATH_NEW_SRC: &str = r#"
use std::path::Path;

fn main() {
    // `Path::new` reaches the path through raw pointer casts
    // over the string data: the projections through the
    // `Path`/`OsStr` wrappers read the string itself.
    let p = Path::new("abc");
    assert_eq!(p.as_os_str(), "abc");
    assert_eq!(p.as_os_str().len(), 3);
    assert_eq!(p.as_os_str().as_encoded_bytes(), b"abc");
    let o = std::ffi::OsStr::new("xy");
    assert_eq!(o.len(), 2);
    println!("path-new-ok");
}
"#;

#[test]
fn run_mode_path_new_and_os_str_projections() {
    let dir = temp_dir("path-new");
    let src = fixture(&dir, "path_new.rs", PATH_NEW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Path::new over a string literal runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "path-new-ok\n",
        "the path projections read the string data"
    );
}

const PATH_PARTS_SRC: &str = r#"
use std::path::Path;

fn main() {
    let p = Path::new("/foo/bar/baz.txt");
    assert_eq!(p.parent().unwrap(), Path::new("/foo/bar"));
    assert_eq!(p.file_name().unwrap(), "baz.txt");
    assert_eq!(p.extension().unwrap(), "txt");
    assert_eq!(p.file_stem().unwrap(), "baz");
    let root = Path::new("/");
    assert_eq!(root.parent(), None);
    assert_eq!(root.file_name(), None);
    let rel = Path::new("foo");
    assert_eq!(rel.parent().unwrap(), Path::new(""));
    assert_eq!(rel.extension(), None);
    let hidden = Path::new("/a/.gitignore");
    assert_eq!(hidden.file_stem().unwrap(), ".gitignore");
    assert_eq!(hidden.extension(), None);
    println!("path-parts-ok");
}
"#;

#[test]
fn run_mode_path_part_accessors() {
    let dir = temp_dir("path-parts");
    let src = fixture(&dir, "path_parts.rs", PATH_PARTS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the Path parsing accessors run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "path-parts-ok\n",
        "parent/file_name/file_stem/extension match std"
    );
}

const PATHBUF_DEREF_SRC: &str = r#"
fn main() {
    // `PathBuf` dereferences to `Path`: the methods below are
    // `Path` methods reached through the `Deref` impl.
    let p = std::env::current_exe().unwrap();
    assert!(p.as_os_str().len() > 0);
    assert!(p.parent().is_some());
    assert!(p.file_name().is_some());
    println!("pathbuf-deref-ok");
}
"#;

#[test]
fn run_mode_pathbuf_deref_to_path() {
    let dir = temp_dir("pathbuf-deref");
    let src = fixture(&dir, "pathbuf_deref.rs", PATHBUF_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "PathBuf deref to Path runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "pathbuf-deref-ok\n",
        "the dereferenced path reads the buffer's byte string"
    );
}

const PATH_COMPONENTS_SRC: &str = r#"
use std::path::{Component, Path};

fn main() {
    // The component kinds and the `as_os_str` view of each:
    // `Component::as_os_str`'s body transmutes a string
    // constant through a byte slice to the `OsStr` sys type,
    // and the consumers below run through library bodies.
    let kinds: Vec<u8> = Path::new("/a/../c")
        .components()
        .map(|c| match c {
            Component::Prefix(_) => 0,
            Component::RootDir => 1,
            Component::CurDir => 2,
            Component::ParentDir => 3,
            Component::Normal(_) => 4,
        })
        .collect();
    assert_eq!(kinds, vec![1, 4, 3, 4]);
    let lens: Vec<usize> = Path::new("/../x")
        .components()
        .map(|c| c.as_os_str().len())
        .collect();
    assert_eq!(lens, vec![1, 2, 1]);
    assert_eq!(Path::new("").components().count(), 0);
    assert_eq!(Path::new("./a").components().count(), 2);
    let mut sum = 0usize;
    for c in Path::new("a//b/").components() {
        sum += c.as_os_str().len();
    }
    assert_eq!(sum, 2);
    println!("path-components-ok");
}
"#;

#[test]
fn run_mode_path_components_iteration() {
    let dir = temp_dir("path-components");
    let src = fixture(&dir, "path_components.rs", PATH_COMPONENTS_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Path::components iteration runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "path-components-ok\n",
        "component kinds and os_str views match std"
    );
}

const PATH_COMPONENTS_CONSUMERS_SRC: &str = r#"
use std::path::Path;

fn main() {
    // `count` is a library body whose closure drops each
    // component it consumes; the drop glue of the component
    // types is empty (they hold only string slices).
    assert_eq!(Path::new("a/b/c/d").components().count(), 4);
    // Front and back steps interleave over one state.
    let mut it = Path::new("a/b/c/d").components();
    assert_eq!(it.next().unwrap().as_os_str().len(), 1);
    assert_eq!(it.next_back().unwrap().as_os_str().len(), 1);
    assert_eq!(it.count(), 2);
    let mut it = Path::new("a").components();
    assert!(it.next().is_some());
    assert!(it.next_back().is_none());
    assert!(it.next().is_none());
    // `rev` composes the back step; an explicit `drop` of a
    // component accounts its (empty) glue as well.
    let names: Vec<String> = Path::new("a/bb")
        .components()
        .rev()
        .map(|c| c.as_os_str().to_str().unwrap().to_string())
        .collect();
    assert_eq!(names, vec!["bb".to_string(), "a".to_string()]);
    let c = Path::new("/x").components().next().unwrap();
    drop(c);
    println!("path-components-consumers-ok");
}
"#;

#[test]
fn run_mode_path_components_consumers() {
    let dir = temp_dir("path-components-consumers");
    let src = fixture(
        &dir,
        "path_components_consumers.rs",
        PATH_COMPONENTS_CONSUMERS_SRC,
    );
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Components consumers (count/next_back/rev/drop) run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "path-components-consumers-ok\n",
        "front/back steps and consumer drops match std"
    );
}

const PATHBUF_BUILD_SRC: &str = r#"
use std::path::{Path, PathBuf};

fn main() {
    let mut pb = PathBuf::from("/tmp");
    pb.push("a");
    pb.push("b/c");
    assert_eq!(pb.to_str().unwrap(), "/tmp/a/b/c");
    assert!(pb.pop());
    assert_eq!(pb.to_str().unwrap(), "/tmp/a/b");
    // An absolute addition replaces the whole buffer.
    pb.push("/abs");
    assert_eq!(pb.to_str().unwrap(), "/abs");
    // The root and the empty buffer have nothing to pop.
    let mut root = PathBuf::from("/");
    assert!(!root.pop());
    assert_eq!(root.to_str().unwrap(), "/");
    let mut empty = PathBuf::new();
    assert!(!empty.pop());
    empty.push("x");
    assert_eq!(empty.to_str().unwrap(), "x");
    // One-byte buffers dereference to their path as well.
    let one = PathBuf::from(String::from("s"));
    assert_eq!(one.to_str().unwrap(), "s");
    assert_eq!(one.components().count(), 1);
    // `to_path_buf` copies a borrowed path into a buffer.
    let copied: PathBuf = Path::new("q/r").to_path_buf();
    assert_eq!(copied.to_str().unwrap(), "q/r");
    println!("pathbuf-build-ok");
}
"#;

#[test]
fn run_mode_pathbuf_construction_and_mutation() {
    let dir = temp_dir("pathbuf-build");
    let src = fixture(&dir, "pathbuf_build.rs", PATHBUF_BUILD_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "PathBuf construction and mutation run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "pathbuf-build-ok\n",
        "from/push/pop/new/to_path_buf match std"
    );
}

const PATHBUF_FROM_JOIN_SRC: &str = r#"
use std::ffi::OsString;
use std::path::PathBuf;

fn main() {
    // A `String` a converted body built -- slice `join` and
    // `concat` go through `join_generic_copy` -- is the real
    // aggregate form, its buffer elements in the `MaybeUninit`
    // union form; the generic `From` conversions read it
    // through the string readers like every other string
    // consumer.
    let p = PathBuf::from(["a", "b"].join("/"));
    println!("{}|{:?}|{}", p.display(), p, p.as_os_str().len());
    let p1 = PathBuf::from(["a/b"].join("/"));
    println!("{}", p1.display());
    let empty: Vec<&str> = Vec::new();
    let pe = PathBuf::from(empty.join("/"));
    println!("[{}]", pe.display());
    let pc = PathBuf::from(["foo", "bar"].concat());
    println!("{}", pc.display());
    let pi: PathBuf = ["a", "b"].join("/").into();
    println!("{}", pi.display());
    let o = OsString::from(["foo", "bar", "baz"].join("/"));
    println!("{}|{}", o.to_string_lossy(), o.len());
    let om = OsString::from(["\u{3b1}", "\u{3b2}"].concat());
    println!("{}|{}", om.to_string_lossy(), om.len());
    // The modelled form keeps working.
    let pm = PathBuf::from(String::from("a/b"));
    println!("{}", pm.display());
    println!("from-join-ok");
}
"#;

#[test]
fn run_mode_pathbuf_and_osstring_from_join_built_string() {
    let dir = temp_dir("pathbuf-from-join");
    let src = fixture(&dir, "pathbuf_from_join.rs", PATHBUF_FROM_JOIN_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "PathBuf/OsString from a join-built String run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a/b|\"a/b\"|3\na/b\n[]\nfoobar\na/b\nfoo/bar/baz|11\n\u{3b1}\u{3b2}|4\na/b\nfrom-join-ok\n",
        "From<String> over the real aggregate form matches native"
    );
}

// ---- Lane JL: one io::Error representation (issue #173) ----
// Every io::Error producer (the fs layer, the process model,
// From<ErrorKind>) builds the bitpacked `Repr` layout std's own
// converted `Error::kind` / `raw_os_error` bodies read. The fs
// layer previously built an aggregate `Error { repr }` shape
// those bodies cannot consume.

const FS_NUL_KIND_SRC: &str = r#"
use std::io::ErrorKind;
fn main() {
    let e = std::fs::metadata("\0").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    assert_eq!(e.raw_os_error(), None);
    let exe = std::env::current_exe().unwrap();
    let e = std::fs::rename(&exe, "\0").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    let e = std::fs::remove_file("\0").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    println!("fs-nul-kind-ok");
}
"#;

#[test]
fn run_mode_fs_nul_path_error_kind() {
    let dir = temp_dir("fs-nul-kind");
    let src = fixture(&dir, "fs_nul_kind.rs", FS_NUL_KIND_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "fs-nul-kind-ok\n");
}

const PATH_COMPONENT_EQ_SRC: &str = r#"
use std::path::Path;
fn e(a: &str, b: &str) -> bool {
    Path::new(a) == Path::new(b)
}
fn main() {
    println!("eq1 {}", e("a//b", "a/b"));
    println!("eq2 {}", e("a/", "a"));
    println!("eq3 {}", e("a/./b", "a/b"));
    println!("eq4 {}", e("./a", "a"));
    println!("eq5 {}", e("/a", "a"));
    println!("eq6 {}", e("", "."));
    println!("eq7 {}", e("../a", "../a"));
    println!("eq8 {}", e("//a", "/a"));
    println!("eq9 {}", e("a/.", "a"));
    println!("eq10 {}", e("a/../b", "a/b"));
    println!("eq11 {}", e("/", "//"));
    println!("eq12 {}", e("a", "a/"));
    println!("cross {}", Path::new("a/b") == "a//b");
    println!("cross2 {}", "a//b" == Path::new("a/b"));
    println!("ne {}", Path::new("a//b") != Path::new("a/b"));
}
"#;

#[test]
fn run_mode_path_component_equality() {
    run_ok(
        "path-component-eq",
        PATH_COMPONENT_EQ_SRC,
        "eq1 true\neq2 true\neq3 true\neq4 false\neq5 false\neq6 false\neq7 true\neq8 true\neq9 true\neq10 false\neq11 true\neq12 true\ncross true\ncross2 true\nne false\n",
    );
}

const PATH_HASH_EQ_SRC: &str = r#"
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
fn hp(p: &str) -> u64 {
    let mut h = DefaultHasher::new();
    Path::new(p).hash(&mut h);
    h.finish()
}
fn main() {
    println!("h1 {}", hp("a//b") == hp("a/b"));
    println!("h2 {}", hp("a/") == hp("a"));
    println!("h3 {}", hp("a/./b") == hp("a/b"));
    println!("h4 {}", hp("a/b") == hp("a/b"));
    println!("h5 {}", hp("/a/b") == hp("/a/b"));
    println!("h6 {}", hp("x") != hp("y"));
    // The key path a hash map exercises: the same key spelled
    // two ways must hash and compare equal.
    let stored = Path::new("some/key");
    let looked_up = Path::new("some//key");
    println!("key {}", stored == looked_up && hp("some/key") == hp("some//key"));
}
"#;

#[test]
fn run_mode_path_hash_eq_consistency() {
    run_ok(
        "path-hash-eq",
        PATH_HASH_EQ_SRC,
        "h1 true\nh2 true\nh3 true\nh4 true\nh5 true\nh6 true\nkey true\n",
    );
}

const ENV_GUEST_SRC: &str = r#"
fn main() {
    assert!(std::env::var_os("SPARAGMOS_NO_SUCH_VAR").is_none());
    match std::env::var("SPARAGMOS_NO_SUCH_VAR") {
        Err(std::env::VarError::NotPresent) => {}
        other => panic!("expected NotPresent, got {other:?}"),
    }
    std::env::set_var("SPARAGMOS_GUEST_VAR", "hello");
    assert_eq!(std::env::var("SPARAGMOS_GUEST_VAR").unwrap(), "hello");
    assert_eq!(
        std::env::var_os("SPARAGMOS_GUEST_VAR").unwrap().to_str().unwrap(),
        "hello"
    );
    std::env::remove_var("SPARAGMOS_GUEST_VAR");
    assert!(std::env::var_os("SPARAGMOS_GUEST_VAR").is_none());
    let cwd = std::env::current_dir().unwrap();
    assert!(cwd.is_absolute());
    assert!(std::env::temp_dir().is_absolute());
    println!("env guest ok");
}
"#;

#[test]
fn run_mode_env_builtins_roundtrip() {
    let dir = temp_dir("env-guest");
    let src = fixture(&dir, "env_guest.rs", ENV_GUEST_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "env guest ok\n", "stderr={stderr}");
}

const FS_GUEST_SRC: &str = r#"
fn main() {
    let src0 = std::env::args().next().unwrap();
    let dir = std::path::Path::new(&src0).parent().unwrap().join("work");
    std::fs::create_dir_all(dir.join("a/b")).unwrap();
    std::fs::write(dir.join("a/b/f.txt"), b"abc").unwrap();
    assert_eq!(
        std::fs::read(dir.join("a/b/f.txt")).unwrap(),
        vec![b'a', b'b', b'c']
    );
    let md = std::fs::metadata(dir.join("a/b/f.txt")).unwrap();
    assert!(md.is_file());
    assert_eq!(md.len(), 3);
    let mut names = Vec::new();
    for e in std::fs::read_dir(dir.join("a")).unwrap() {
        let e = e.unwrap();
        names.push(e.file_name().to_str().unwrap().to_string());
    }
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "b");
    match std::fs::read(dir.join("missing")) {
        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
        Ok(_) => panic!("missing file read succeeded"),
    }
    match std::fs::create_dir(dir.join("a")) {
        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists),
        Ok(_) => panic!("second create_dir succeeded"),
    }
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(!dir.exists());
    println!("fs guest ok");
}
"#;

#[test]
fn run_mode_fs_builtins_roundtrip() {
    let dir = temp_dir("fs-guest");
    let src = fixture(&dir, "fs_guest.rs", FS_GUEST_SRC);
    let out = run_driver(&src);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(stdout, "fs guest ok\n", "stderr={stderr}");
}

const PATH_ORDERING_SRC: &str = r#"
use std::cmp::Ordering;
use std::path::Path;

fn main() {
    // std's `Path` ordering compares the parsed component
    // sequences, not the raw bytes: the first components
    // here are "a" and "a.b", so "a/c" < "a.b/c" — byte
    // order would give the opposite ('.' = 0x2e < '/' =
    // 0x2f).
    assert_eq!(Path::new("a/c").cmp(Path::new("a.b/c")), Ordering::Less);
    assert!(Path::new("a/c") < Path::new("a.b/c"));
    assert!(Path::new("a.b/c") > Path::new("a/c"));
    assert!(Path::new("a/c") <= Path::new("a.b/c"));
    // Component kinds order by kind before any name bytes:
    // root directory, then current directory, then parent
    // directory, then normal names.
    assert_eq!(Path::new("/a").cmp(Path::new("./a")), Ordering::Less);
    assert_eq!(Path::new("./a").cmp(Path::new("../a")), Ordering::Less);
    assert_eq!(Path::new("../a").cmp(Path::new("a")), Ordering::Less);
    assert_eq!(Path::new("/a").cmp(Path::new("a")), Ordering::Less);
    // A strict component prefix is smaller, and paths whose
    // components parse the same compare equal.
    assert_eq!(Path::new("a").cmp(Path::new("a/b")), Ordering::Less);
    assert_eq!(Path::new("a/..").cmp(Path::new("a")), Ordering::Greater);
    assert_eq!(Path::new("foo").cmp(Path::new("foo")), Ordering::Equal);
    assert_eq!(Path::new("a//b").cmp(Path::new("a/b")), Ordering::Equal);
    assert_eq!(
        Path::new("a/b").partial_cmp(Path::new("a/b")),
        Some(Ordering::Equal)
    );
    println!("path-ordering-ok");
}
"#;

#[test]
fn run_mode_path_component_ordering() {
    run_ok("path-ordering", PATH_ORDERING_SRC, "path-ordering-ok\n");
}

const PATH_SORT_SRC: &str = r#"
use std::ffi::OsString;
use std::path::PathBuf;

fn main() {
    // Sorting owned buffers goes through the generic sort
    // machinery, whose comparisons carry no type tag; the
    // order must still be the component order.
    let mut v = vec![
        PathBuf::from("a.b/c"),
        PathBuf::from("a/c"),
        PathBuf::from("/x"),
    ];
    v.sort();
    assert_eq!(v[0], PathBuf::from("/x"));
    assert_eq!(v[1], PathBuf::from("a/c"));
    assert_eq!(v[2], PathBuf::from("a.b/c"));
    let mut w = vec![PathBuf::from("a.b/c"), PathBuf::from("a/c")];
    w.sort_unstable();
    assert_eq!(w[0], PathBuf::from("a/c"));
    assert_eq!(w[1], PathBuf::from("a.b/c"));
    // Binary search uses the same ordering, and a
    // component-equal key ("a//c") finds its entry.
    assert_eq!(v.binary_search(&PathBuf::from("a/c")), Ok(1));
    assert!(v.binary_search(&PathBuf::from("a//c")).is_ok());
    // The neighbouring string-like types keep their byte
    // order in the same generic machinery.
    let mut oss = vec![OsString::from("a/c"), OsString::from("a.b/c")];
    oss.sort();
    assert_eq!(oss[0], OsString::from("a.b/c"));
    let mut vecs = vec![b"a/c".to_vec(), b"a.b/c".to_vec()];
    vecs.sort();
    assert_eq!(vecs[0], b"a.b/c".to_vec());
    println!("path-sort-ok");
}
"#;

#[test]
fn run_mode_pathbuf_sort_component_order() {
    run_ok("path-sort", PATH_SORT_SRC, "path-sort-ok\n");
}

const PATH_ANCESTORS_SRC: &str = r#"
use std::path::Path;

fn main() {
    // `ancestors` yields the path itself, then each parent
    // up to (and including) the root / the empty path.
    let anc: Vec<&Path> = Path::new("/a/b/c").ancestors().collect();
    assert_eq!(
        anc,
        vec![
            Path::new("/a/b/c"),
            Path::new("/a/b"),
            Path::new("/a"),
            Path::new("/")
        ]
    );
    let anc2: Vec<String> = Path::new("a/b")
        .ancestors()
        .map(|p| p.to_str().unwrap().to_string())
        .collect();
    assert_eq!(
        anc2,
        vec!["a/b".to_string(), "a".to_string(), "".to_string()]
    );
    assert_eq!(Path::new("").ancestors().count(), 1);
    assert_eq!(Path::new("/").ancestors().count(), 1);
    // The parents are lexical: a trailing separator takes
    // no part, and `..` components are kept as components.
    let anc3: Vec<String> = Path::new("a/b/")
        .ancestors()
        .map(|p| p.to_str().unwrap().to_string())
        .collect();
    assert_eq!(
        anc3,
        vec!["a/b/".to_string(), "a".to_string(), "".to_string()]
    );
    let anc4: Vec<String> = Path::new("a/../b")
        .ancestors()
        .map(|p| p.to_str().unwrap().to_string())
        .collect();
    assert_eq!(
        anc4,
        vec![
            "a/../b".to_string(),
            "a/..".to_string(),
            "a".to_string(),
            "".to_string()
        ]
    );
    // An exhausted iterator stays exhausted.
    let mut it = Path::new("x/y").ancestors();
    assert_eq!(it.next().unwrap(), Path::new("x/y"));
    assert_eq!(it.next().unwrap(), Path::new("x"));
    assert_eq!(it.next().unwrap(), Path::new(""));
    assert!(it.next().is_none());
    assert!(it.next().is_none());
    println!("path-ancestors-ok");
}
"#;

#[test]
fn run_mode_path_ancestors() {
    run_ok("path-ancestors", PATH_ANCESTORS_SRC, "path-ancestors-ok\n");
}

const PATH_SET_EXTENSION_SRC: &str = r#"
use std::path::PathBuf;

fn main() {
    let mut pb = PathBuf::from("foo.rs");
    assert!(pb.set_extension("txt"));
    assert_eq!(pb.to_str().unwrap(), "foo.txt");
    // An empty extension removes the old one (and the dot).
    assert!(pb.set_extension(""));
    assert_eq!(pb.to_str().unwrap(), "foo");
    // A path without an extension gains one.
    let mut bare = PathBuf::from("foo");
    assert!(bare.set_extension("md"));
    assert_eq!(bare.to_str().unwrap(), "foo.md");
    // No file name: the buffer is left alone, result false.
    let mut dot = PathBuf::from("..");
    assert!(!dot.set_extension("txt"));
    assert_eq!(dot.to_str().unwrap(), "..");
    let mut root = PathBuf::from("/");
    assert!(!root.set_extension("txt"));
    assert_eq!(root.to_str().unwrap(), "/");
    // `with_extension` replaces only the final extension.
    let w = PathBuf::from("a/b.tar.gz").with_extension("zip");
    assert_eq!(w.to_str().unwrap(), "a/b.tar.zip");
    let w2 = PathBuf::from("a/b").with_extension("md");
    assert_eq!(w2.to_str().unwrap(), "a/b.md");
    // The extension argument may be an OsStr too.
    let w3 = PathBuf::from("a/b.rs").with_extension(std::ffi::OsStr::new("toml"));
    assert_eq!(w3.to_str().unwrap(), "a/b.toml");
    println!("path-set-extension-ok");
}
"#;

#[test]
fn run_mode_pathbuf_set_extension() {
    run_ok(
        "path-set-extension",
        PATH_SET_EXTENSION_SRC,
        "path-set-extension-ok\n",
    );
}

const PATHBUF_DEDUP_SRC: &str = r#"
use std::path::PathBuf;

fn main() {
    let mut v: Vec<PathBuf> = vec![
        PathBuf::from("a/b"),
        PathBuf::from("a/b"),
        PathBuf::from("c"),
        PathBuf::from("a-b"),
        PathBuf::from("a-b"),
        PathBuf::from("a-b"),
        PathBuf::from("d/e"),
        PathBuf::from("c"),
    ];
    v.dedup();
    println!("after len: {}", v.len());
    for p in &v {
        println!("item: {}", p.to_str().unwrap());
    }
    // Path equality is component-wise: "a//b", "a/./b" and
    // "a/b/" are the same path as "a/b".
    let mut w: Vec<PathBuf> = vec![
        PathBuf::from("a/b"),
        PathBuf::from("a//b"),
        PathBuf::from("a/./b"),
        PathBuf::from("a/b/"),
    ];
    w.dedup();
    println!("normalised dedup len: {}", w.len());
    println!("pathbuf-dedup-ok");
}
"#;

#[test]
fn run_mode_pathbuf_dedup() {
    run_ok(
        "pathbuf-dedup",
        PATHBUF_DEDUP_SRC,
        "after len: 5\nitem: a/b\nitem: c\nitem: a-b\nitem: d/e\nitem: c\nnormalised dedup len: 1\npathbuf-dedup-ok\n",
    );
}

const PATH_MIXED_CMP_SRC: &str = r#"
use std::path::{Path, PathBuf};

fn generic_lt<T: PartialOrd>(a: T, b: T) -> bool { a < b }

fn main() {
    let buf_ab = PathBuf::from("a/b");
    let buf_a_b = PathBuf::from("a-b");
    let p_ab: &Path = Path::new("a/b");
    assert!(buf_ab.as_path() < buf_a_b.as_path());
    assert!(p_ab < &buf_a_b);
    assert!(&buf_ab.as_path() < &buf_a_b);
    assert!(buf_ab.as_path() == p_ab);
    assert!(generic_lt(buf_ab.as_path(), buf_a_b.as_path()));
    assert!(generic_lt(&buf_ab, &buf_a_b));
    assert_eq!(std::cmp::min(buf_ab.as_path(), buf_a_b.as_path()), p_ab);
    // A sort through the generic Ord adapter keeps component
    // order ("a/b" < "a-b" component-wise, not byte-wise).
    let mut v: Vec<&Path> = vec![buf_a_b.as_path(), buf_ab.as_path()];
    v.sort();
    assert!(v[0] == p_ab);
    let mut w: Vec<PathBuf> = vec![
        PathBuf::from("a/b"),
        PathBuf::from("c"),
        PathBuf::from("a/b"),
        PathBuf::from("a-b"),
        PathBuf::from("c"),
    ];
    w.sort();
    w.dedup();
    assert_eq!(w.len(), 3);
    assert_eq!(w[0].to_str().unwrap(), "a/b");
    assert_eq!(w[1].to_str().unwrap(), "a-b");
    assert_eq!(w[2].to_str().unwrap(), "c");
    println!("path-mixed-cmp-ok");
}
"#;

#[test]
fn run_mode_path_mixed_cmp() {
    run_ok("path-mixed-cmp", PATH_MIXED_CMP_SRC, "path-mixed-cmp-ok\n");
}

const PATH_REF_SORT_SRC: &str = r#"
use std::path::{Path, PathBuf};

fn main() {
    // Sorting a Vec of wide references (each element is a fat
    // pointer: two words) exercises the sort's internal
    // element copies at the element type's true stride; a
    // second sort over a Vec of a different element type runs
    // in the same program.
    let a = PathBuf::from("a/b");
    let b = PathBuf::from("a-b");
    let c = PathBuf::from("c");
    let mut v: Vec<&Path> = vec![c.as_path(), b.as_path(), a.as_path()];
    v.sort();
    assert_eq!(v[0].to_str().unwrap(), "a/b");
    assert_eq!(v[1].to_str().unwrap(), "a-b");
    assert_eq!(v[2].to_str().unwrap(), "c");
    let mut w: Vec<PathBuf> = vec![
        PathBuf::from("a/b"),
        PathBuf::from("c"),
        PathBuf::from("a/b"),
        PathBuf::from("a-b"),
        PathBuf::from("c"),
    ];
    w.sort();
    w.dedup();
    assert_eq!(w.len(), 3);
    println!("path-ref-sort-ok");
}
"#;

#[test]
fn run_mode_path_ref_sort() {
    run_ok("path-ref-sort", PATH_REF_SORT_SRC, "path-ref-sort-ok\n");
}

const PATHBUF_PTR_SWAP_SRC: &str = r#"
use std::path::PathBuf;

fn main() {
    // A raw-pointer element copy (the shift a dedup performs)
    // and a raw-pointer swap must move the whole element
    // value, wrapper chain included.
    let mut v: Vec<PathBuf> = vec![PathBuf::from("a/b"), PathBuf::from("c"), PathBuf::from("d/e")];
    unsafe {
        let base = v.as_mut_ptr();
        std::ptr::copy_nonoverlapping(base.add(2), base.add(1), 1);
    }
    println!("copied: {}", v[1].to_str().unwrap());
    let mut u: Vec<PathBuf> = vec![PathBuf::from("m/n"), PathBuf::from("o")];
    unsafe {
        std::ptr::swap(u.as_mut_ptr(), u.as_mut_ptr().add(1));
    }
    println!("swapped 0: {}", u[0].to_str().unwrap());
    println!("swapped 1: {}", u[1].to_str().unwrap());
    let mut s: Vec<String> = vec!["hello".to_string(), "world".to_string()];
    unsafe {
        std::ptr::swap(s.as_mut_ptr(), s.as_mut_ptr().add(1));
    }
    println!("strings: {} {}", s[0], s[1]);
    println!("pathbuf-ptr-swap-ok");
}
"#;

#[test]
fn run_mode_pathbuf_ptr_swap() {
    run_ok(
        "pathbuf-ptr-swap",
        PATHBUF_PTR_SWAP_SRC,
        "copied: d/e\nswapped 0: o\nswapped 1: m/n\nstrings: world hello\npathbuf-ptr-swap-ok\n",
    );
}

const PATH_FMT_SRC: &str = r#"
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Wrap {
    p: PathBuf,
    n: i32,
}

fn main() {
    // The Debug of the path / OS-string wrapper types is the
    // quoted content, not the wrapper structure; width and
    // precision are not consulted by std's impls.
    println!("{:?}", PathBuf::from("a/b"));
    println!("{:?}", Path::new("a/b"));
    println!("{:?}", OsString::from("os/s"));
    println!("{:?}", vec![PathBuf::from("x"), PathBuf::from("y/z")]);
    println!("{:.2?}", PathBuf::from("abcdef"));
    println!("{:>12?}", PathBuf::from("ab"));
    println!("{:?}", Wrap { p: PathBuf::from("in/struct"), n: 3 });
    println!("{:?}", PathBuf::from("a\\b\"c"));
    println!("{:?}", PathBuf::from(""));
    println!("path-fmt-ok");
}
"#;

#[test]
fn run_mode_path_and_os_string_debug_formats_content() {
    let dir = temp_dir("path-fmt");
    let src = fixture(&dir, "path_fmt.rs", PATH_FMT_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "path Debug formatting runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "\"a/b\"\n\"a/b\"\n\"os/s\"\n[\"x\", \"y/z\"]\n\"abcdef\"\n\"ab\"\nWrap { p: \"in/struct\", n: 3 }\n\"a\\\\b\\\"c\"\n\"\"\npath-fmt-ok\n",
        "PathBuf/Path/OsString Debug renders the quoted content like std"
    );
}

const PATH_DISPLAY_SRC: &str = r#"
use std::path::{Path, PathBuf};

fn main() {
    println!("{}", Path::new("a/b").display());
    println!("{}", PathBuf::from("dir/file.txt").display());
    let s = Path::new("x/y").display().to_string();
    println!("{s}");
    let pb = PathBuf::from("p/q");
    println!("{}", pb.display());
    println!("path-display-ok");
}
"#;

#[test]
fn run_mode_path_display_and_to_string() {
    let dir = temp_dir("path-display");
    let src = fixture(&dir, "path_display.rs", PATH_DISPLAY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "path Display formatting runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a/b\ndir/file.txt\nx/y\np/q\npath-display-ok\n",
        "Path::display renders the path content and to_string converts it"
    );
}

const NON_UTF8_PATH_DISPLAY_SRC: &str = r#"
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;

fn main() {
    // A non-UTF-8 `OsString` (built from raw bytes): its
    // `&OsStr` form is a fat pointer to the byte elements in
    // this model, and the `OsStr` / `Slice` field projections
    // inside `OsStr::len` and `Path::display` walk the
    // one-field wrapper chain over that fat pointer.
    let s = OsString::from_vec(vec![0x61, 0xff, 0x62]);
    println!("len={}", s.len());
    let pb = std::path::PathBuf::from(s);
    println!("{}", pb.display());
    println!("len={}", pb.as_os_str().len());
    println!("non-utf8-path-display-ok");
}
"#;

#[test]
fn run_mode_non_utf8_path_display() {
    let dir = temp_dir("non-utf8-path-display");
    let src = fixture(&dir, "non_utf8_path_display.rs", NON_UTF8_PATH_DISPLAY_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "non-UTF-8 path display runs; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=3\na\u{fffd}b\nlen=3\nnon-utf8-path-display-ok\n",
        "display() renders the lossy content and len() reads the byte length"
    );
}

const PATHBUF_AS_PATH_SRC: &str = r#"
use std::path::{Path, PathBuf};
fn main() {
    let pb = PathBuf::from("abc");
    let p = pb.as_path();
    println!("len {}", p.as_os_str().len());
    println!("eq {}", p == Path::new("abc"));
    println!("done");
}
"#;

#[test]
fn run_mode_pathbuf_as_path() {
    run_ok(
        "pathbuf-as-path",
        PATHBUF_AS_PATH_SRC,
        "len 3\neq true\ndone\n",
    );
}

// `std::path::Path::is_file` / `is_dir` have no MIR body in the
// distributed std; the interpreter answers them from host
// metadata (a failed metadata read is `false`, as in std).
const PATH_IS_FILE_DIR_SRC: &str = r#"
fn main() {
    let dir = std::env::temp_dir().join("sparagmos-is-file-probe");
    let file = dir.join("probe.txt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(&file, b"x").unwrap();
    println!("file: is_file={} is_dir={}", file.is_file(), file.is_dir());
    println!("dir: is_file={} is_dir={}", dir.is_file(), dir.is_dir());
    let missing = dir.join("missing-427");
    println!(
        "missing: is_file={} is_dir={}",
        missing.is_file(),
        missing.is_dir()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
"#;

#[test]
fn run_mode_path_is_file_is_dir() {
    run_ok(
        "path-is-file-dir",
        PATH_IS_FILE_DIR_SRC,
        "file: is_file=true is_dir=false\n\
         dir: is_file=false is_dir=true\n\
         missing: is_file=false is_dir=false\n",
    );
}

const COW_OS_STR_OWNED_DEREF_SRC: &str = r#"use std::borrow::Cow;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

fn report(tag: &str, cow: &Cow<'_, OsStr>) {
    let s: &OsStr = &**cow;
    println!("{tag} len={} to_str={:?} lossy={} debug={:?}",
        s.len(), s.to_str(), s.to_string_lossy(), s);
}

fn main() {
    // Owned through `Cow::from(OsString)`.
    let from: Cow<'_, OsStr> = Cow::from(OsString::from("hello os"));
    report("from", &from);
    // Owned constructed directly.
    let direct: Cow<'_, OsStr> = Cow::Owned(OsString::from("direct owned"));
    report("direct", &direct);
    // Owned from a PathBuf's OsString.
    let pb = PathBuf::from("/tmp/some/dir");
    let from_pb: Cow<'_, OsStr> = Cow::Owned(pb.into_os_string());
    report("pathbuf", &from_pb);
    // Owned whose OsString came from `to_owned` on a borrowed
    // OsStr (the bare string form of the value).
    let from_ref: Cow<'_, OsStr> = Cow::from(OsStr::new("relative/path").to_owned());
    report("to_owned", &from_ref);
    // Owned over a real-aggregate OsString (with_capacity + push).
    let mut built = OsString::with_capacity(16);
    built.push("built ");
    built.push("by push");
    let agg: Cow<'_, OsStr> = Cow::Owned(built);
    report("aggregate", &agg);
    // Borrowed control and cross-form equality / ordering.
    let borrowed: Cow<'_, OsStr> = Cow::Borrowed(OsStr::new("hello os"));
    report("borrowed", &borrowed);
    println!("eq={} lt={}", from == borrowed, from < direct);
    // Back out again.
    let back: OsString = from.into_owned();
    println!("back={back:?}");
    println!("cow os str owned deref ok");
}
"#;

#[test]
fn run_mode_cow_os_str_owned_deref() {
    // The `Deref` builtin's `Cow` arm returned the owned
    // field itself for a `Cow<OsStr>`: an `OsString`
    // aggregate where the guest expected the `&OsStr`
    // reference shape, so every deref consumer stopped on
    // the type mismatch. The owned field now denominates
    // through the shared OS-string byte readers (whichever
    // stored form it takes: canonical enum, real aggregate,
    // or the bare string form `to_owned` produces) and
    // derefs to the sub-path value, exactly as
    // `OsString::deref` itself does. `Cow::from` /
    // `Into` into a `Cow` target are covered here too:
    // they had no arm at all before.
    let dir = temp_dir("cow-os-str-owned-deref");
    let src = fixture(&dir, "cow_os_str_owned_deref.rs", COW_OS_STR_OWNED_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "owned Cow<OsStr> deref must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "from len=8 to_str=Some(\"hello os\") lossy=hello os debug=\"hello os\"\ndirect len=12 to_str=Some(\"direct owned\") lossy=direct owned debug=\"direct owned\"\npathbuf len=13 to_str=Some(\"/tmp/some/dir\") lossy=/tmp/some/dir debug=\"/tmp/some/dir\"\nto_owned len=13 to_str=Some(\"relative/path\") lossy=relative/path debug=\"relative/path\"\naggregate len=13 to_str=Some(\"built by push\") lossy=built by push debug=\"built by push\"\nborrowed len=8 to_str=Some(\"hello os\") lossy=hello os debug=\"hello os\"\neq=true lt=false\nback=\"hello os\"\ncow os str owned deref ok\n"
    );
}

const COW_PATH_OWNED_DEREF_SRC: &str = r#"use std::borrow::Cow;
use std::path::{Path, PathBuf};

fn main() {
    // Owned through `Cow::from(PathBuf)`.
    let owned: Cow<'_, Path> = Cow::from(PathBuf::from("/a/b/c.txt"));
    let p: &Path = &*owned;
    println!("len={}", p.as_os_str().len());
    println!("debug={p:?}");
    println!("file_name={:?}", p.file_name());
    println!("parent={:?}", owned.parent());
    println!("is_absolute={}", owned.is_absolute());
    // Owned constructed directly.
    let direct: Cow<'_, Path> = Cow::Owned(PathBuf::from("rel/file.rs"));
    println!("direct_ext={:?}", direct.extension());
    // Borrowed control and cross-form equality.
    let borrowed: Cow<'_, Path> = Cow::Borrowed(Path::new("/a/b/c.txt"));
    println!("eq={}", owned == borrowed);
    println!("cow path owned deref ok");
}
"#;

#[test]
fn run_mode_cow_path_owned_deref() {
    // The same owned-form hole for `Cow<Path>`: the owned
    // `PathBuf` field denominates through the shared path
    // byte readers and derefs to the sub-path value, as
    // `PathBuf::deref` does.
    let dir = temp_dir("cow-path-owned-deref");
    let src = fixture(&dir, "cow_path_owned_deref.rs", COW_PATH_OWNED_DEREF_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "owned Cow<Path> deref must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "len=10\ndebug=\"/a/b/c.txt\"\nfile_name=Some(\"c.txt\")\nparent=Some(\"/a/b\")\nis_absolute=true\ndirect_ext=Some(\"rs\")\neq=true\ncow path owned deref ok\n"
    );
}

const COW_FROM_REFERENCE_SRC: &str = r#"use std::borrow::Cow;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

fn main() {
    // From<&B> constructions (Borrowed variant).
    let a: Cow<'_, str> = Cow::from("literal text");
    println!("a={} borrowed={}", &*a, matches!(a, Cow::Borrowed(_)));

    let os = OsString::from("ref os");
    let b: Cow<'_, OsStr> = Cow::from(os.as_os_str());
    println!("b_len={} borrowed={}", (&*b).len(), matches!(b, Cow::Borrowed(_)));

    let bytes: &[u8] = &[9, 8, 7];
    let c: Cow<'_, [u8]> = Cow::from(bytes);
    println!("c_len={} borrowed={}", c.len(), matches!(c, Cow::Borrowed(_)));

    // From<&Owned> constructions (Borrowed over the deref shape).
    let owned_string = String::from("ref string");
    let d: Cow<'_, str> = Cow::from(&owned_string);
    println!("d={} borrowed={}", &*d, matches!(d, Cow::Borrowed(_)));

    let e: Cow<'_, OsStr> = Cow::from(&os);
    println!("e_len={} borrowed={}", (&*e).len(), matches!(e, Cow::Borrowed(_)));

    let pb = PathBuf::from("/ref/path");
    let f: Cow<'_, Path> = Cow::from(&pb);
    println!("f_debug={:?} borrowed={}", &*f, matches!(f, Cow::Borrowed(_)));

    let v = vec![5u8, 6];
    let g: Cow<'_, [u8]> = Cow::from(&v);
    println!("g_len={} borrowed={}", g.len(), matches!(g, Cow::Borrowed(_)));

    // Into: owned sources.
    let h: Cow<'_, OsStr> = OsString::from("into os").into();
    println!("h_len={} owned={}", (&*h).len(), matches!(h, Cow::Owned(_)));

    let i: Cow<'_, str> = String::from("into str").into();
    println!("i={} owned={}", &*i, matches!(i, Cow::Owned(_)));
}
"#;

#[test]
fn run_mode_cow_from_reference_sources() {
    // `From<&T>` / `From<&Owned>` for a `Cow` target build
    // the borrowed variant: over the reference itself for
    // a borrowed-type source, and over the dereferenced
    // value for a reference to the owned type. The
    // `&Vec<T>` source must not be confused with the
    // owned `Vec<T>` source (that mix-up wrapped the
    // reference as the owned variant).
    let dir = temp_dir("cow-from-reference");
    let src = fixture(&dir, "cow_from_reference.rs", COW_FROM_REFERENCE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "Cow from reference sources must run; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "a=literal text borrowed=true\nb_len=6 borrowed=true\nc_len=3 borrowed=true\nd=ref string borrowed=true\ne_len=6 borrowed=true\nf_debug=\"/ref/path\" borrowed=true\ng_len=2 borrowed=true\nh_len=7 owned=true\ni=into str owned=true\n"
    );
}
