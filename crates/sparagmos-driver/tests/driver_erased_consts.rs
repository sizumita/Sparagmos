//! Type-erased constants: a constant whose stored pointer has a
//! zero-sized pointee type (the shape `io::Error`'s bit-packed
//! `SimpleMessage` form and hand-rolled type erasure both take —
//! a `NonNull<()>` field naming an allocation of some other type)
//! cannot be decoded through the declared pointee type: the
//! allocation's bytes are decoded type-free, with provenance
//! relocations, and the typed value is materialised at the cast
//! that recovers the real type, from the byte-layout plan the
//! conversion records for that cast. The allocation's identity
//! survives: repeated casts of the same erased pointer compare
//! equal, a cast back to the erased type compares equal to the
//! original pointer, and a read through the recovered type sees
//! the stored fields, nested references and enum tags included.

use std::path::PathBuf;
use std::process::Command;

fn driver() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sparagmos-driver"))
}

fn write_fixture(name: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-erased-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let path = dir.join(format!("{name}.rs"));
    std::fs::write(&path, src).expect("write fixture");
    path
}

fn run_fixture(name: &str, src: &str) -> (String, String, std::process::ExitStatus) {
    let fixture = write_fixture(name, src);
    let out = Command::new(driver())
        .args(["run", "--edition", "2021"])
        .arg(&fixture)
        .env_remove("SPARAGMOS_DUMP")
        .output()
        .expect("spawn driver");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status,
    )
}

fn run_ok(name: &str, src: &str, expected_stdout: &str) {
    let (stdout, stderr, status) = run_fixture(name, src);
    assert_eq!(
        status.code(),
        Some(0),
        "erased-const case {name} must run to completion; stderr={stderr}"
    );
    assert_eq!(
        stdout, expected_stdout,
        "erased-const case {name} stdout mismatch; stderr={stderr}"
    );
}

const NONNULL_STRUCT_RS: &str = r#"
use std::ptr::NonNull;

struct Msg {
    kind: u32,
    message: &'static str,
}

struct Erased {
    p: NonNull<()>,
}

const MSG: Msg = Msg { kind: 7, message: "hello erased" };
const ERASED: Erased = Erased {
    p: unsafe { NonNull::new_unchecked(&MSG as *const Msg as *mut ()) },
};

fn main() {
    let m = unsafe { &*ERASED.p.as_ptr().cast::<Msg>() };
    println!("kind={} msg={}", m.kind, m.message);
    let m2 = unsafe { &*ERASED.p.as_ptr().cast::<Msg>() };
    println!("kind={} msg={}", m2.kind, m2.message);
    println!("same: {}", std::ptr::eq(m, m2));
}
"#;

#[test]
fn nonnull_erased_const_struct_readback() {
    run_ok(
        "nonnull_erased",
        NONNULL_STRUCT_RS,
        "kind=7 msg=hello erased\nkind=7 msg=hello erased\nsame: true\n",
    );
}

const ZST_ROUND_TRIP_RS: &str = r#"
use std::ptr::NonNull;

struct Msg {
    kind: u32,
    message: &'static str,
}

struct Erased {
    p: NonNull<()>,
}

const MSG: Msg = Msg { kind: 7, message: "hello erased" };
const ERASED: Erased = Erased {
    p: unsafe { NonNull::new_unchecked(&MSG as *const Msg as *mut ()) },
};

fn main() {
    let raw = ERASED.p.as_ptr();
    println!("null: {}", raw.is_null());
    let typed = raw.cast::<Msg>();
    println!("null2: {}", typed.is_null());
    let m = unsafe { &*typed };
    println!("kind={}", m.kind);
    println!("msg={}", m.message);
}
"#;

#[test]
fn erased_pointer_zst_cast_round_trip() {
    run_ok(
        "zst_round_trip",
        ZST_ROUND_TRIP_RS,
        "null: false\nnull2: false\nkind=7\nmsg=hello erased\n",
    );
}

const ENUM_NESTED_RS: &str = r#"
use std::ptr::NonNull;

enum Payload {
    Num(u32),
    Text(&'static str),
}

struct Inner {
    label: &'static str,
    vals: [u16; 3],
}

struct Entry {
    payload: Payload,
    inner: Inner,
}

struct Erased {
    p: NonNull<()>,
}

const ENTRY_A: Entry = Entry {
    payload: Payload::Text("enum text"),
    inner: Inner { label: "nested", vals: [3, 1, 4] },
};
const ENTRY_B: Entry = Entry {
    payload: Payload::Num(41),
    inner: Inner { label: "second", vals: [9, 8, 7] },
};
const ERASED_A: Erased = Erased {
    p: unsafe { NonNull::new_unchecked(&ENTRY_A as *const Entry as *mut ()) },
};
const ERASED_B: Erased = Erased {
    p: unsafe { NonNull::new_unchecked(&ENTRY_B as *const Entry as *mut ()) },
};

fn show(e: &Entry) {
    match &e.payload {
        Payload::Num(n) => println!("num={n}"),
        Payload::Text(t) => println!("text={t}"),
    }
    println!("label={} vals={:?}", e.inner.label, e.inner.vals);
}

fn main() {
    let a = unsafe { &*ERASED_A.p.as_ptr().cast::<Entry>() };
    show(a);
    let b = unsafe { &*ERASED_B.p.as_ptr().cast::<Entry>() };
    show(b);
}
"#;

#[test]
fn erased_const_with_enum_and_nested_ref() {
    run_ok(
        "erased_nested",
        ENUM_NESTED_RS,
        "text=enum text\nlabel=nested vals=[3, 1, 4]\nnum=41\nlabel=second vals=[9, 8, 7]\n",
    );
}

const READ_EXACT_RS: &str = r#"
use std::io::Read;
fn main() {
    let mut empty: &[u8] = &[];
    let mut buf = [0u8; 4];
    let err = empty.read_exact(&mut buf).unwrap_err();
    println!("kind: {:?}", err.kind());
    println!("raw: {:?}", err.raw_os_error());
    println!("display: {}", err);
}
"#;

#[test]
fn read_exact_eof_kind_and_raw_os_error() {
    // The error value itself — an erased `SimpleMessage`
    // allocation — survives conversion: its kind and raw OS
    // error read back exactly, and its `Display` renders the
    // message text decoded from the preserved constant block
    // (OJ, #299).
    let (stdout, stderr, status) = run_fixture("read_exact_eof", READ_EXACT_RS);
    assert_eq!(
        stdout,
        "kind: UnexpectedEof\nraw: None\ndisplay: failed to fill whole buffer\n"
    );
    assert_eq!(status.code(), Some(0), "stderr={stderr}");
}

const IDENTITY_ROUND_TRIP_RS: &str = r#"
use std::ptr::NonNull;

struct Msg {
    kind: u32,
    message: &'static str,
}

struct Erased {
    p: NonNull<()>,
}

const MSG: Msg = Msg { kind: 7, message: "hello erased" };
const ERASED: Erased = Erased {
    p: unsafe { NonNull::new_unchecked(&MSG as *const Msg as *mut ()) },
};

fn main() {
    let raw = ERASED.p.as_ptr();
    let typed = raw.cast::<Msg>();
    let back = typed as *mut ();
    println!("eq: {}", std::ptr::eq(raw, back));
    println!("addr eq: {}", raw as usize == back as usize);
    println!("kind={} msg={}", unsafe { (*typed).kind }, unsafe { (*typed).message });
}
"#;

#[test]
fn erased_pointer_identity_survives_materialising_cast() {
    // The materialised value is a typed view of the one
    // erased allocation, not a new allocation: a pointer cast
    // to the concrete type and back to the erased type
    // compares equal to the original pointer, by `ptr::eq`
    // and by address, as it does natively.
    run_ok(
        "identity_round_trip",
        IDENTITY_ROUND_TRIP_RS,
        "eq: true\naddr eq: true\nkind=7 msg=hello erased\n",
    );
}

const STR_SLICE_FIELD_RS: &str = r#"
use std::ptr::NonNull;

struct Holder {
    p: NonNull<()>,
}

struct Inner {
    names: &'static [&'static str],
}

const NAMES: [&'static str; 2] = ["a", "bb"];
const INNER: Inner = Inner { names: &NAMES };
const H: Holder = Holder {
    p: unsafe { NonNull::new_unchecked(&INNER as *const Inner as *mut ()) },
};

fn main() {
    let inner = unsafe { &*H.p.as_ptr().cast::<Inner>() };
    println!("{} {}", inner.names[0], inner.names[1]);
}
"#;

#[test]
fn erased_const_with_str_slice_field() {
    // The slice's elements are fat pointers (two words
    // each): the second element is read at its real stride,
    // not from the first element's length word.
    run_ok("str_slice_field", STR_SLICE_FIELD_RS, "a bb\n");
}

const RECURSIVE_CAST_RS: &str = r#"
struct Node {
    a: *const Node,
    b: *const Node,
    c: *const Node,
    v: u32,
}

fn main() {
    let n = Node {
        a: std::ptr::null(),
        b: std::ptr::null(),
        c: std::ptr::null(),
        v: 3,
    };
    let erased = &n as *const Node as *const ();
    let back = erased as *const Node;
    println!("v={}", unsafe { (*back).v });
}
"#;

#[test]
fn recursive_type_cast_converts() {
    // Recording the materialisation plan of a cast to a
    // recursive type must not unroll the type graph into a
    // tree: with three pointer fields per level the unguarded
    // expansion exhausts memory during conversion. The
    // recursive edge gets no plan; the program converts and
    // runs.
    let fixture = write_fixture("recursive_cast", RECURSIVE_CAST_RS);
    let out = Command::new(driver())
        .arg(&fixture)
        .args(["--edition", "2021"])
        .env_remove("SPARAGMOS_DUMP")
        .output()
        .expect("spawn driver");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a cast to a recursive type must convert; stderr={stderr}"
    );
    run_ok("recursive_cast_run", RECURSIVE_CAST_RS, "v=3\n");
}

const PRETTY_DEBUG_RS: &str = r#"
use std::io::{self, Read};

fn main() {
    let data: &[u8] = &[1, 2, 3];
    let mut buf = [0u8; 8];
    let mut slice = data;
    let err = slice.read_exact(&mut buf).unwrap_err();
    println!("pretty={:#?}", err);
    println!("plain={:?}", err);
    let res: io::Result<()> = Err(err);
    println!("nested={:#?}", res);
    let e2 = io::Error::from_raw_os_error(2);
    println!("os pretty={:#?}", e2);
    let e3 = io::Error::from(io::ErrorKind::NotFound);
    println!("kind pretty={:#?}", e3);
}
"#;

#[test]
fn io_error_pretty_debug_uses_multiline_layout() {
    // core renders the `io::Error` representations through
    // `debug_struct` / `debug_tuple`, so under `{:#?}` they
    // take the multi-line layout — fields indented one level
    // past the error's own nesting, the closing delimiter at
    // the error's indent — while `{:?}` stays the compact
    // one-line form. The structural formatter must propagate
    // the current indentation into the error's fields instead
    // of always emitting the compact form.
    run_ok(
        "io_error_pretty_debug",
        PRETTY_DEBUG_RS,
        "pretty=Error {\n    kind: UnexpectedEof,\n    message: \"failed to fill whole buffer\",\n}\nplain=Error { kind: UnexpectedEof, message: \"failed to fill whole buffer\" }\nnested=Err(\n    Error {\n        kind: UnexpectedEof,\n        message: \"failed to fill whole buffer\",\n    },\n)\nos pretty=Os {\n    code: 2,\n    kind: NotFound,\n    message: \"No such file or directory\",\n}\nkind pretty=Kind(\n    NotFound,\n)\n",
    );
}
