//! Run-mode tests for cross-field partial writes through a
//! transmuted view of an enum's storage (Issue #513): a
//! `#[repr(C)]` struct/union view of an enum is written field
//! by field (tag first, then the active variant's payload
//! fields, some of them nested inside a payload union), and
//! the value is then read back through the enum's own type.
//! The writes and reads denote byte spans of one storage, so
//! the enum must re-form at the written tag with exactly the
//! written payload. On the base tree every program here
//! fails: the writes compose a nested `{tag, payload}` shape
//! inside the enum value, and the reads stop at an
//! uninitialised or type-mismatched place.

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
        "sparagmos-driver-uc-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
}

fn run_ok(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver(&src_path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "case {tag} must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "case {tag} stdout mismatch; stderr={stderr}"
    );
}

const ENUM_DEF: &str = r#"
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum MyEnum {
    A(u32),
    B { x: u8, y: i16, z: u8 },
    C,
    D(Option<u32>),
    E(std::time::Duration),
}
"#;

#[test]
fn run_mode_union_crossfield_repr_c_struct_view() {
    let src = format!(
        r#"use std::mem;
{ENUM_DEF}
#[repr(C)]
struct MyEnumRepr {{
    tag: MyEnumTag,
    payload: MyEnumPayload,
}}

#[repr(C)]
union MyEnumPayload {{
    A: MyEnumVariantA,
    B: MyEnumVariantB,
    D: MyEnumVariantD,
    E: MyEnumVariantE,
}}

#[repr(C)] #[derive(Copy, Clone)] enum MyEnumTag {{ A, B, C, D, E }}
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantA(u32);
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantB {{ x: u8, y: i16, z: u8 }}
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantD(Option<u32>);
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantE(std::time::Duration);

fn main() {{
    unsafe {{
        let mut dest: MyEnum = mem::zeroed();
        let destr: &mut MyEnumRepr = mem::transmute(&mut dest);
        destr.tag = MyEnumTag::B;
        destr.payload.B.x = 206;
        destr.payload.B.y = 1145;
        destr.payload.B.z = 78;
        assert_eq!(dest, MyEnum::B {{ x: 206, y: 1145, z: 78 }});
        let y = destr.payload.B.y;
        assert_eq!(y, 1145);
        destr.tag = MyEnumTag::C;
        assert_eq!(dest, MyEnum::C);
        destr.tag = MyEnumTag::D;
        destr.payload.D.0 = Some(407);
        assert_eq!(dest, MyEnum::D(Some(407)));
        destr.payload.D.0 = None;
        assert_eq!(dest, MyEnum::D(None));
        destr.tag = MyEnumTag::E;
        destr.payload.E.0 = std::time::Duration::new(100, 0);
        assert_eq!(dest, MyEnum::E(std::time::Duration::new(100, 0)));
        destr.tag = MyEnumTag::A;
        destr.payload.A.0 = 17;
        assert_eq!(dest, MyEnum::A(17));
    }}
    println!("repr-c ok");
}}
"#
    );
    run_ok("repr-c", &src, "repr-c ok\n");
}

#[test]
fn run_mode_union_crossfield_repr_int_union_view() {
    let src = format!(
        r#"use std::mem;
#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum MyEnum {{
    A(u32),
    B {{ x: u8, y: i16, z: u8 }},
    C,
    D(Option<u32>),
    E(std::time::Duration),
}}

#[repr(C)]
union MyEnumRepr {{
    A: MyEnumVariantA,
    B: MyEnumVariantB,
    C: MyEnumVariantC,
    D: MyEnumVariantD,
    E: MyEnumVariantE,
}}

#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantA(MyEnumTag, u32);
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantB {{ tag: MyEnumTag, x: u8, y: i16, z: u8 }}
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantC(MyEnumTag);
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantD(MyEnumTag, Option<u32>);
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantE(MyEnumTag, std::time::Duration);
#[repr(u8)] #[derive(Copy, Clone)] enum MyEnumTag {{ A, B, C, D, E }}

fn main() {{
    unsafe {{
        let mut dest: MyEnum = mem::zeroed();
        let destr: &mut MyEnumRepr = mem::transmute(&mut dest);
        destr.B.tag = MyEnumTag::B;
        destr.B.x = 206;
        destr.B.y = 1145;
        destr.B.z = 78;
        assert_eq!(dest, MyEnum::B {{ x: 206, y: 1145, z: 78 }});
        destr.C.0 = MyEnumTag::C;
        assert_eq!(dest, MyEnum::C);
        destr.D.0 = MyEnumTag::D;
        destr.D.1 = Some(407);
        assert_eq!(dest, MyEnum::D(Some(407)));
        destr.E.0 = MyEnumTag::E;
        destr.E.1 = std::time::Duration::new(100, 0);
        assert_eq!(dest, MyEnum::E(std::time::Duration::new(100, 0)));
    }}
    println!("repr-int ok");
}}
"#
    );
    run_ok("repr-int", &src, "repr-int ok\n");
}

#[test]
fn run_mode_union_crossfield_repeated_parse() {
    let src = format!(
        r#"use std::mem;
{ENUM_DEF}
#[repr(C)]
struct MyEnumRepr {{
    tag: MyEnumTag,
    payload: MyEnumPayload,
}}

#[repr(C)]
union MyEnumPayload {{
    A: MyEnumVariantA,
    B: MyEnumVariantB,
    D: MyEnumVariantD,
    E: MyEnumVariantE,
}}

#[repr(C)] #[derive(Copy, Clone)] enum MyEnumTag {{ A, B, C, D, E }}
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantA(u32);
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantB {{ x: u8, y: i16, z: u8 }}
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantD(Option<u32>);
#[repr(C)] #[derive(Copy, Clone)] struct MyEnumVariantE(std::time::Duration);

fn parse(dest: &mut MyEnum, tag: u8, word: u32) {{
    unsafe {{
        let destr: &mut MyEnumRepr = mem::transmute(dest);
        match tag {{
            0 => {{
                destr.tag = MyEnumTag::A;
                destr.payload.A.0 = word;
            }}
            1 => {{
                destr.tag = MyEnumTag::B;
                destr.payload.B.x = word as u8;
                destr.payload.B.y = (word >> 8) as i16;
                destr.payload.B.z = (word >> 24) as u8;
            }}
            _ => {{
                destr.tag = MyEnumTag::C;
            }}
        }}
    }}
}}

fn main() {{
    let mut dest: MyEnum = unsafe {{ mem::zeroed() }};
    let mut output = vec![];
    parse(&mut dest, 0, 17);
    output.push(dest);
    parse(&mut dest, 1, 0x4E_47_21_CE);
    output.push(dest);
    parse(&mut dest, 2, 0);
    output.push(dest);
    parse(&mut dest, 0, 99);
    output.push(dest);
    assert_eq!(
        output,
        vec![
            MyEnum::A(17),
            MyEnum::B {{ x: 206, y: 18209, z: 78 }},
            MyEnum::C,
            MyEnum::A(99),
        ]
    );
    println!("repeated ok");
}}
"#
    );
    run_ok("repeated", &src, "repeated ok\n");
}
