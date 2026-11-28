//! Byte-form `ptr::read` tests: build small fixture programs
//! in a temp dir, execute them with `sparagmos-driver run`,
//! and check the exit-code contract (0 = normal termination,
//! 101 = guest panic, 2 = could not convert/execute, with the
//! reason on stderr). The fixtures cover the materialisation
//! of typed values out of byte storage through a raw pointer:
//! a valid enum read from a byte array, a struct with an enum
//! field, `read_unaligned` at an offset, a tuple, a payload
//! enum in both variants, a `copy_nonoverlapping` into a
//! `MaybeUninit` slot read back with `assume_init`, an
//! `assume_init_read` of a byte-written slot, the write side
//! (an enum written through a byte pointer, read back as a
//! byte and as the enum), a reference transmute read, and the
//! invalid-tag read, whose tag names no variant: the value
//! model holds no such enum value, so the read fails with the
//! validity error the transmute path reports for the same
//! materialisation, naming the discriminant problem instead
//! of a representation mismatch in the consuming assignment.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

fn fixture(dir: &Path, name: &str, src: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    path
}

fn run_driver(src: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sparagmos-driver"))
        .arg("run")
        .arg(src)
        .output()
        .expect("failed to run sparagmos-driver")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-ptr-read-bytes-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn run_ok(tag: &str, name: &str, src: &str) -> String {
    let dir = temp_dir(tag);
    let path = fixture(&dir, name, src);
    let out = run_driver(&path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "fixture must run to completion; stderr={stderr}"
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn run_err(tag: &str, name: &str, src: &str) -> String {
    let dir = temp_dir(tag);
    let path = fixture(&dir, name, src);
    let out = run_driver(&path);
    assert_eq!(
        out.status.code(),
        Some(2),
        "fixture must stop with the could-not-execute code"
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

const TWO: &str = "#[derive(Debug, PartialEq)]\nenum Two { A, B }\n";

/// A valid enum materialised from a byte array through
/// `ptr::read`: the tag byte re-forms the enum value, and
/// both the comparison and the derived `Debug` see it.
#[test]
fn read_valid_enum_from_bytes() {
    let stdout = run_ok(
        "valid-enum",
        "valid_enum.rs",
        &format!(
            r#"{TWO}
fn main() {{
    let bytes: [u8; 1] = [1];
    let v: Two = unsafe {{ std::ptr::read(bytes.as_ptr().cast::<Two>()) }};
    println!("is_b={{}} dbg={{:?}}", v == Two::B, v);
}}
"#
        ),
    );
    assert_eq!(stdout, "is_b=true dbg=B\n");
}

/// A struct with an enum field read whole out of a byte
/// array: each field's span constructs from its own bytes.
#[test]
fn read_struct_with_enum_field_from_bytes() {
    let stdout = run_ok(
        "struct-enum-field",
        "struct_enum_field.rs",
        &format!(
            r#"{TWO}
#[derive(Debug)]
struct S {{ x: u32, e: Two }}
fn main() {{
    let mut bytes = [0u8; 8];
    bytes[0..4].copy_from_slice(&42u32.to_ne_bytes());
    bytes[4] = 1;
    let s: S = unsafe {{ std::ptr::read(bytes.as_ptr().cast::<S>()) }};
    println!("x={{}} is_b={{}}", s.x, s.e == Two::B);
}}
"#
        ),
    );
    assert_eq!(stdout, "x=42 is_b=true\n");
}

/// `read_unaligned` at a nonzero offset inside a byte array:
/// the pointer's stored element step prices through the
/// birth array, the value through the accessing type.
#[test]
fn read_unaligned_at_offset() {
    let stdout = run_ok(
        "read-unaligned",
        "read_unaligned.rs",
        &format!(
            r#"{TWO}
fn main() {{
    let buf: [u8; 4] = [0xAA, 1, 0xBB, 0xCC];
    let v: Two = unsafe {{ std::ptr::read_unaligned(buf.as_ptr().add(1).cast::<Two>()) }};
    println!("is_b={{}}", v == Two::B);
}}
"#
        ),
    );
    assert_eq!(stdout, "is_b=true\n");
}

/// A tuple read out of a byte array: the fields sit at the
/// declaration-order layout the value model stores them in.
#[test]
fn read_tuple_from_bytes() {
    let stdout = run_ok(
        "tuple",
        "tuple_from_bytes.rs",
        r#"
fn main() {
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&123456u64.to_ne_bytes());
    bytes[8..12].copy_from_slice(&7u32.to_ne_bytes());
    let t: (u64, u32) = unsafe { std::ptr::read(bytes.as_ptr().cast::<(u64, u32)>()) };
    println!("t={} {}", t.0, t.1);
}
"#,
    );
    assert_eq!(stdout, "t=123456 7\n");
}

/// A payload enum read from a hand-built byte image, in the
/// payload variant and in the fieldless variant alike: the
/// tag bytes re-form the value at the variant they name,
/// carrying the payload bytes across.
#[test]
fn read_payload_enum_both_variants() {
    let stdout = run_ok(
        "payload-enum",
        "payload_enum.rs",
        r#"
#[derive(Debug, PartialEq)]
enum E { A(u32), B }
fn main() {
    let mut bimg = [0u8; 8];
    bimg[0..4].copy_from_slice(&1u32.to_ne_bytes());
    let v: E = unsafe { std::ptr::read(bimg.as_ptr().cast::<E>()) };
    match v { E::A(x) => println!("A {}", x), E::B => println!("B") }
    let mut aimg = [0u8; 8];
    aimg[4..8].copy_from_slice(&77u32.to_ne_bytes());
    let v2: E = unsafe { std::ptr::read(aimg.as_ptr().cast::<E>()) };
    match v2 { E::A(x) => println!("A {}", x), E::B => println!("B") }
}
"#,
    );
    assert_eq!(stdout, "B\nA 77\n");
}

/// A `copy_nonoverlapping` of one element out of a byte array
/// into a `MaybeUninit` slot, read back with `assume_init`:
/// the copy denominates its element by the element type,
/// constructing it from the byte the source storage held.
#[test]
fn copy_into_maybe_uninit_then_assume_init() {
    let stdout = run_ok(
        "copy-assume-init",
        "copy_assume_init.rs",
        &format!(
            r#"{TWO}
fn main() {{
    let bytes: [u8; 1] = [1];
    let mut slot: std::mem::MaybeUninit<Two> = std::mem::MaybeUninit::uninit();
    unsafe {{
        std::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<Two>(), slot.as_mut_ptr(), 1);
        let v = slot.assume_init();
        println!("is_b={{}}", v == Two::B);
    }}
}}
"#
        ),
    );
    assert_eq!(stdout, "is_b=true\n");
}

/// A `MaybeUninit` slot written through a byte pointer and
/// read with `assume_init_read`: the read's stored field
/// step prices through the slot's own type.
#[test]
fn assume_init_read_of_byte_written_slot() {
    let stdout = run_ok(
        "assume-init-read",
        "assume_init_read.rs",
        &format!(
            r#"{TWO}
fn main() {{
    let mut slot: std::mem::MaybeUninit<Two> = std::mem::MaybeUninit::uninit();
    unsafe {{ std::ptr::write(slot.as_mut_ptr().cast::<u8>(), 1) }};
    let v = unsafe {{ slot.assume_init_read() }};
    println!("is_b={{}}", v == Two::B);
}}
"#
        ),
    );
    assert_eq!(stdout, "is_b=true\n");
}

/// The write side of the same view: an enum written through
/// a byte-array pointer cast to the enum type lands as its
/// tag byte in the array, and the array reads back both as
/// the byte and, through `ptr::read`, as the enum.
#[test]
fn write_enum_through_byte_pointer() {
    let stdout = run_ok(
        "write-through",
        "write_through.rs",
        &format!(
            r#"{TWO}
fn main() {{
    let mut bytes: [u8; 1] = [0];
    unsafe {{ *bytes.as_mut_ptr().cast::<Two>() = Two::B }};
    println!("byte={{}}", bytes[0]);
    let v: Two = unsafe {{ std::ptr::read(bytes.as_ptr().cast::<Two>()) }};
    println!("is_b={{}}", v == Two::B);
}}
"#
        ),
    );
    assert_eq!(stdout, "byte=1\nis_b=true\n");
}

/// A reference formed by casting a byte pointer to an enum
/// pointer: dereferencing it reads the enum the tag byte
/// names, not the bare byte.
#[test]
fn ref_transmute_read() {
    let stdout = run_ok(
        "ref-transmute",
        "ref_transmute.rs",
        &format!(
            r#"{TWO}
fn main() {{
    let bytes: [u8; 1] = [1];
    let r: &Two = unsafe {{ &*bytes.as_ptr().cast::<Two>() }};
    println!("is_b={{}}", *r == Two::B);
}}
"#
        ),
    );
    assert_eq!(stdout, "is_b=true\n");
}

/// An invalid tag read through `ptr::read`: the tag names no
/// variant, and the value model holds no enum value for such
/// a discriminant, so the read stops with the validity error
/// naming the discriminant problem — not with a
/// representation mismatch in the assignment that consumes
/// the value.
#[test]
fn read_invalid_enum_tag_is_validity_error() {
    let stderr = run_err(
        "invalid-tag",
        "invalid_tag.rs",
        &format!(
            r#"{TWO}
fn main() {{
    let bytes: [u8; 1] = [9];
    let v: Two = unsafe {{ std::ptr::read(bytes.as_ptr().cast::<Two>()) }};
    println!("is_a={{}}", v == Two::A);
}}
"#
        ),
    );
    assert!(
        stderr.contains("a discriminant value of the target enum"),
        "stderr must name the discriminant problem; stderr={stderr}"
    );
    assert!(
        stderr.contains("bit pattern 0x9"),
        "stderr must name the offending pattern; stderr={stderr}"
    );
}
