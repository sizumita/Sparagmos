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
//! equal, and a read through the recovered type sees the stored
//! fields, nested references and enum tags included.

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
    // error read back exactly. Rendering the error's `Display`
    // is a separate, still-open formatting gap (the erased
    // message text is not decoded yet), so the run stops there
    // with the named unsupported error after printing the two
    // lines above.
    let (stdout, stderr, status) = run_fixture("read_exact_eof", READ_EXACT_RS);
    assert_eq!(stdout, "kind: UnexpectedEof\nraw: None\n");
    assert_eq!(status.code(), Some(2), "stderr={stderr}");
    assert!(
        stderr.contains("Display formatting for an ADT value"),
        "stderr={stderr}"
    );
}
