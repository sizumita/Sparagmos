//! Array-pointer denomination tests: build small fixture
//! programs in a temp dir, execute them with
//! `sparagmos-driver run`, and check the exit-code contract
//! (0 = normal termination, 101 = guest panic, 2 = could not
//! convert/execute, with the reason on stderr). The fixtures
//! cover pointer arithmetic whose pointee is an array type:
//! an offset denominated in whole arrays over a stored
//! sequence must land on the element the guest's byte
//! displacement denotes, whatever the sequence's element
//! type (a `MaybeUninit` union, an ADT, a scalar), whatever
//! carries the position (a bare local, a static, a sub-slice
//! of one), and in either direction. Before the fix, the
//! sequence position of an array local whose element the
//! bare size table cannot price (a union or ADT element)
//! went unjudged, so the offset was carried as a raw byte
//! displacement and the read or write through it stopped at
//! the value boundary; and a position inside a sub-slice was
//! invisible to the denomination guard, so the value-shape
//! absorption denominated the sub-sequence in the offset's
//! own units and read the wrong elements back.

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
        "sparagmos-driver-array-ptr-denom-{}-{tag}",
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

const MU_BUF: &str = r#"use std::mem::MaybeUninit;
fn make_buf() -> [MaybeUninit<u64>; 6] {
    let mut buf: [MaybeUninit<u64>; 6] = [const { MaybeUninit::uninit() }; 6];
    for i in 0..6 { buf[i] = MaybeUninit::new((i * 10) as u64); }
    buf
}
"#;

/// The pin shape: a `*const [u64; 3]` over a
/// `[MaybeUninit<u64>; 6]` buffer, stepped by one whole
/// array, reads the buffer's second half.
#[test]
fn mu_array_ptr_add_reads_second_array() {
    let stdout = run_ok(
        "mu-add-read",
        "mu_add_read.rs",
        &format!(
            r#"{MU_BUF}
fn main() {{
    let buf = make_buf();
    unsafe {{
        let p = buf.as_ptr() as *const [u64; 3];
        let a0 = &*p;
        println!("{{}} {{}} {{}}", a0[0], a0[1], a0[2]);
        let a1 = &*p.add(1);
        println!("{{}} {{}} {{}}", a1[0], a1[1], a1[2]);
    }}
}}
"#
        ),
    );
    assert_eq!(stdout, "0 10 20\n30 40 50\n");
}

/// The write side of the pin shape: element writes through
/// the stepped array pointer land in the second half.
#[test]
fn mu_array_ptr_add_writes_elements() {
    let stdout = run_ok(
        "mu-add-write",
        "mu_add_write.rs",
        &format!(
            r#"{MU_BUF}
fn main() {{
    let mut buf = make_buf();
    unsafe {{
        let p = buf.as_mut_ptr() as *mut [u64; 3];
        (*p.add(1))[0] = 99;
        (*p.add(1))[2] = 77;
    }}
    unsafe {{
        println!("{{}} {{}} {{}} {{}} {{}} {{}}", buf[0].assume_init(), buf[1].assume_init(), buf[2].assume_init(), buf[3].assume_init(), buf[4].assume_init(), buf[5].assume_init());
    }}
}}
"#
        ),
    );
    assert_eq!(stdout, "0 10 20 99 40 77\n");
}

/// A smaller array pointee over the same buffer: both the
/// first and the second step land on their elements.
#[test]
fn mu_array_ptr_two_element_array_steps() {
    let stdout = run_ok(
        "mu-add2",
        "mu_add2.rs",
        &format!(
            r#"{MU_BUF}
fn main() {{
    let buf = make_buf();
    unsafe {{
        let p = buf.as_ptr() as *const [u64; 2];
        let a1 = &*p.add(1);
        println!("{{}} {{}}", a1[0], a1[1]);
        let a2 = &*p.add(2);
        println!("{{}} {{}}", a2[0], a2[1]);
    }}
}}
"#
        ),
    );
    assert_eq!(stdout, "20 30\n40 50\n");
}

/// Byte-sized elements: the array step is four bytes, and
/// the denomination is the element's, not the byte's.
#[test]
fn mu_array_ptr_byte_elements() {
    let stdout = run_ok(
        "mu-u8",
        "mu_u8.rs",
        r#"use std::mem::MaybeUninit;
fn main() {
    let mut buf: [MaybeUninit<u8>; 8] = [const { MaybeUninit::uninit() }; 8];
    for i in 0..8 { buf[i] = MaybeUninit::new((i * 10) as u8); }
    unsafe {
        let p = buf.as_ptr() as *const [u8; 4];
        let a1 = &*p.add(1);
        println!("{} {} {} {}", a1[0], a1[1], a1[2], a1[3]);
    }
}
"#,
    );
    assert_eq!(stdout, "40 50 60 70\n");
}

/// A negative array offset from a one-past-the-end pointer
/// formed by a scalar step: the two denominations compose.
#[test]
fn mu_array_ptr_offset_back_from_end() {
    let stdout = run_ok(
        "mu-neg",
        "mu_neg.rs",
        &format!(
            r#"{MU_BUF}
fn main() {{
    let buf = make_buf();
    unsafe {{
        let end = buf.as_ptr().add(6) as *const [u64; 3];
        let a = &*end.offset(-1);
        println!("{{}} {{}} {{}}", a[0], a[1], a[2]);
    }}
}}
"#
        ),
    );
    assert_eq!(stdout, "30 40 50\n");
}

/// Indexing the stepped pointer directly, without forming
/// the intermediate reference explicitly.
#[test]
fn mu_array_ptr_index_deref() {
    let stdout = run_ok(
        "mu-index",
        "mu_index.rs",
        &format!(
            r#"{MU_BUF}
fn main() {{
    let buf = make_buf();
    unsafe {{
        let p = buf.as_ptr() as *const [u64; 3];
        println!("{{}}", (*p.add(1))[2]);
        println!("{{}}", (*p.add(1))[0]);
    }}
}}
"#
        ),
    );
    assert_eq!(stdout, "50\n30\n");
}

/// A negative array offset from a base pointer the scalar
/// denomination formed mid-buffer, and the base itself.
#[test]
fn mu_array_ptr_offset_back_from_scalar_base() {
    let stdout = run_ok(
        "mu-mid-neg",
        "mu_mid_neg.rs",
        &format!(
            r#"{MU_BUF}
fn main() {{
    let buf = make_buf();
    unsafe {{
        let p = buf.as_ptr().add(3) as *const [u64; 3];
        let a = &*p.offset(-1);
        println!("{{}} {{}} {{}}", a[0], a[1], a[2]);
        let b = &*p.add(0);
        println!("{{}} {{}} {{}}", b[0], b[1], b[2]);
    }}
}}
"#
        ),
    );
    assert_eq!(stdout, "0 10 20\n30 40 50\n");
}

/// A whole-value consumer of the stepped reference: the
/// iterator sum reads every element of the second array.
#[test]
fn mu_array_ptr_iter_sum() {
    let stdout = run_ok(
        "mu-sum",
        "mu_sum.rs",
        &format!(
            r#"{MU_BUF}
fn main() {{
    let buf = make_buf();
    unsafe {{
        let p = buf.as_ptr() as *const [u64; 3];
        let a1 = &*p.add(1);
        let s: u64 = a1.iter().sum();
        println!("{{}}", s);
    }}
}}
"#
        ),
    );
    assert_eq!(stdout, "120\n");
}

/// An ADT element type on plain storage: an
/// `[Option<u64>; 6]` buffer stepped by `[Option<u64>; 3]`.
/// The bare size table cannot price the element (its name
/// carries no generic arguments), so the position went
/// unjudged exactly as the union element's did.
#[test]
fn adt_element_array_ptr_add() {
    let stdout = run_ok(
        "option-elem",
        "option_elem.rs",
        r#"fn main() {
    let buf: [Option<u64>; 6] = [Some(0), Some(10), Some(20), Some(30), Some(40), Some(50)];
    unsafe {
        let p = buf.as_ptr() as *const [Option<u64>; 3];
        let a1 = &*p.add(1);
        println!("{:?} {:?} {:?}", a1[0], a1[1], a1[2]);
    }
}
"#,
    );
    assert_eq!(stdout, "Some(30) Some(40) Some(50)\n");
}

/// A slice-derived array pointer over a plain buffer: the
/// position sits behind a sub-slice step, where the
/// denomination guard used to be blind and the absorption
/// denominated the sub-sequence in the offset's own units
/// (one array step read back as one element step).
#[test]
fn slice_array_ptr_add_read() {
    let stdout = run_ok(
        "slice-read",
        "slice_read.rs",
        r#"fn main() {
    let buf: [u64; 6] = [0, 10, 20, 30, 40, 50];
    let s: &[u64] = &buf[..];
    unsafe {
        let p = s.as_ptr() as *const [u64; 3];
        let a1 = &*p.add(1);
        println!("{} {} {}", a1[0], a1[1], a1[2]);
    }
}
"#,
    );
    assert_eq!(stdout, "30 40 50\n");
}

/// The write side of the slice-derived shape.
#[test]
fn slice_array_ptr_add_write() {
    let stdout = run_ok(
        "slice-write",
        "slice_write.rs",
        r#"fn main() {
    let mut buf: [u64; 6] = [0, 10, 20, 30, 40, 50];
    let s: &mut [u64] = &mut buf[..];
    unsafe {
        let p = s.as_mut_ptr() as *mut [u64; 3];
        (*p.add(1))[1] = 99;
    }
    println!("{} {} {} {} {} {}", buf[0], buf[1], buf[2], buf[3], buf[4], buf[5]);
}
"#,
    );
    assert_eq!(stdout, "0 10 20 30 99 50\n");
}

/// A tail sub-slice's array pointer: the step ranges over
/// the sub-sequence the sub-slice step denotes.
#[test]
fn subslice_tail_array_ptr_add() {
    let stdout = run_ok(
        "subslice-tail",
        "subslice_tail.rs",
        r#"fn main() {
    let buf: [u64; 8] = [0, 10, 20, 30, 40, 50, 60, 70];
    let s: &[u64] = &buf[2..];
    unsafe {
        let p = s.as_ptr() as *const [u64; 2];
        let a1 = &*p.add(1);
        println!("{} {}", a1[0], a1[1]);
    }
}
"#,
    );
    assert_eq!(stdout, "40 50\n");
}

/// An element pointer formed inside a slice, cast to an
/// array pointer and stepped: the position carries both a
/// sub-slice step and an index step, and the array offset
/// composes over both.
#[test]
fn slice_elem_ptr_then_array_add() {
    let stdout = run_ok(
        "slice-elem-then-array",
        "slice_elem_then_array.rs",
        r#"fn main() {
    let buf: [u64; 8] = [0, 10, 20, 30, 40, 50, 60, 70];
    let s: &[u64] = &buf[..];
    unsafe {
        let p = s.as_ptr().add(1) as *const [u64; 3];
        let a1 = &*p.add(1);
        println!("{} {} {}", a1[0], a1[1], a1[2]);
    }
}
"#,
    );
    assert_eq!(stdout, "40 50 60\n");
}

/// The static form of the pin shape: a `static mut`
/// `MaybeUninit` buffer stepped by an array pointer (the
/// static position's element went unpriced the same way,
/// and the absorption mis-denominated it instead).
#[test]
fn static_mu_array_ptr_add() {
    let stdout = run_ok(
        "static-mu",
        "static_mu.rs",
        r#"use std::mem::MaybeUninit;
static mut BUF: [MaybeUninit<u64>; 6] = [const { MaybeUninit::uninit() }; 6];
fn main() {
    unsafe {
        for i in 0..6 { BUF[i] = MaybeUninit::new((i * 10) as u64); }
        let p = std::ptr::addr_of!(BUF) as *const [u64; 3];
        let a1 = &*p.add(1);
        println!("{} {} {}", a1[0], a1[1], a1[2]);
    }
}
"#,
    );
    assert_eq!(stdout, "30 40 50\n");
}

/// The stepped array pointer cast back to a scalar pointer
/// (which leaves a first-element entry step on the
/// position), then stepped in scalar units: the offset
/// still ranges over the buffer's elements.
#[test]
fn stepped_array_ptr_cast_back_scalar_add() {
    let stdout = run_ok(
        "cast-back",
        "cast_back.rs",
        &format!(
            r#"{MU_BUF}
fn main() {{
    let buf = make_buf();
    unsafe {{
        let p = buf.as_ptr() as *const [u64; 3];
        let q = p.add(1) as *const u64;
        println!("{{}}", *q);
        println!("{{}}", *q.add(2));
    }}
}}
"#
        ),
    );
    assert_eq!(stdout, "30\n50\n");
}

/// Control: a plain scalar-element buffer stepped by an
/// array pointer (this shape was already denominated
/// correctly and must stay so).
#[test]
fn plain_array_ptr_add_control() {
    let stdout = run_ok(
        "plain-control",
        "plain_control.rs",
        r#"fn main() {
    let buf: [u64; 6] = [0, 10, 20, 30, 40, 50];
    unsafe {
        let p = buf.as_ptr() as *const [u64; 3];
        let a0 = &*p;
        println!("{} {} {}", a0[0], a0[1], a0[2]);
        let a1 = &*p.add(1);
        println!("{} {} {}", a1[0], a1[1], a1[2]);
    }
}
"#,
    );
    assert_eq!(stdout, "0 10 20\n30 40 50\n");
}
