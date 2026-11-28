//! Run-mode tests for byte writes and reads through raw
//! pointers into union-typed places (Issue #513): a
//! `*mut u8` aimed at a union's storage denotes the union's
//! bytes, not one of its fields, so the write composes the
//! union's storage image and the value takes its byte form;
//! later field reads decode from those bytes and later byte
//! reads see them back. On the base tree every program here
//! stops: the byte view had no union arm, so a raw-pointer
//! byte write into a union place declined through every
//! write path and fell to a whole-value type mismatch.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-ubw-{}-{tag}",
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

fn run_err(tag: &str, src: &str) {
    let dir = temp_dir(tag);
    let src_path = fixture(&dir, "case.rs", src);
    let out = run_driver(&src_path);
    assert_ne!(
        out.status.code(),
        Some(0),
        "case {tag} must not run to completion; stdout={}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn run_mode_union_byte_write_at_base_read_back_through_fields() {
    run_ok(
        "base",
        r#"
union U {
    a: u8,
    b: usize,
}

fn main() {
    let mut u = U { b: 0 };
    unsafe {
        (&mut u as *mut U as *mut u8).write(1);
        println!("{}", u.a);
        println!("{}", u.b);
    }
}
"#,
        "1\n1\n",
    );
}

#[test]
fn run_mode_union_byte_write_displaced_read_back_through_wide_field() {
    run_ok(
        "displaced",
        r#"
union U {
    a: u8,
    storage: usize,
}

fn main() {
    let mut u = U { storage: 0 };
    unsafe {
        (&mut u as *mut U as *mut u8).add(3).write(0xAB);
        println!("{:x}", u.storage);
    }
}
"#,
        "ab000000\n",
    );
}

#[test]
fn run_mode_union_byte_write_preserves_unwritten_bytes() {
    run_ok(
        "preserve",
        r#"
union U {
    a: u8,
    storage: usize,
}

fn main() {
    let mut u = U { storage: 0x0102_0304_0506_0708 };
    unsafe {
        (&mut u as *mut U as *mut u8).write(0xFF);
        println!("{:x}", u.storage);
        // The written byte reads back through the pointer too.
        let p = &u as *const U as *const u8;
        println!("{}", p.read());
        println!("{}", p.add(7).read());
    }
}
"#,
        "1020304050607ff\n255\n1\n",
    );
}

#[test]
fn run_mode_union_byte_reads_of_field_written_union() {
    run_ok(
        "reads",
        r#"
pub union Bag {
    uninit: (),
    _storage: usize,
}

fn make_1u8_bag() -> Bag {
    let mut bag = Bag { uninit: () };
    unsafe { (&mut bag as *mut Bag as *mut u8).write(1); }
    bag
}

fn main() {
    let bag = make_1u8_bag();
    let val = unsafe { (&bag as *const Bag as *const u8).read() };
    println!("{val}");
    println!("{}", unsafe { bag._storage });
}
"#,
        "1\n1\n",
    );
}

#[test]
fn run_mode_union_byte_write_nested_in_struct() {
    run_ok(
        "nested",
        r#"
union U {
    a: u8,
    b: usize,
}

struct W {
    u: U,
    tail: u8,
}

fn main() {
    let mut w = W { u: U { b: 0 }, tail: 3 };
    unsafe {
        (&mut w.u as *mut U as *mut u8).write(5);
        println!("{}", w.u.b);
        println!("{}", w.tail);
    }
}
"#,
        "5\n3\n",
    );
}

#[test]
fn run_mode_union_byte_write_outside_storage_is_an_error() {
    run_err(
        "outside",
        r#"
union U {
    a: u8,
    b: usize,
}

fn main() {
    let mut u = U { b: 0 };
    unsafe {
        (&mut u as *mut U as *mut u8).add(8).write(1);
    }
    println!("wrote");
}
"#,
    );
}

#[test]
fn run_mode_struct_byte_write_unchanged() {
    run_ok(
        "struct-control",
        r#"
struct S {
    a: usize,
    b: u8,
}

fn main() {
    let mut s = S { a: 0, b: 0 };
    unsafe {
        let p = &mut s as *mut S as *mut u8;
        p.write(7);
        p.add(3).write(0xAB);
        println!("{:x}", s.a);
        println!("{}", s.b);
    }
}
"#,
        "ab000007\n0\n",
    );
}

#[test]
fn run_mode_generic_union_byte_write_at_base_read_back_through_fields() {
    run_ok(
        "generic-base",
        r#"
union BagOfBits<T: Copy> {
    val: T,
    bytes: [u8; 8],
}

fn main() {
    let mut u = BagOfBits::<u64> { val: 0 };
    unsafe {
        (&mut u as *mut BagOfBits<u64> as *mut u8).write(1);
        println!("{}", u.val);
        println!("{}", u.bytes[0]);
    }
}
"#,
        "1\n1\n",
    );
}

#[test]
fn run_mode_generic_union_byte_write_displaced_read_back_through_wide_field() {
    run_ok(
        "generic-displaced",
        r#"
union BagOfBits<T: Copy> {
    val: T,
    bytes: [u8; 8],
}

fn main() {
    let mut u = BagOfBits::<u64> { val: 0 };
    unsafe {
        (&mut u as *mut BagOfBits<u64> as *mut u8).add(3).write(0xAB);
        println!("{:x}", u.val);
    }
}
"#,
        "ab000000\n",
    );
}

#[test]
fn run_mode_generic_struct_byte_write_read_back_through_fields() {
    run_ok(
        "generic-struct",
        r#"
struct Wrap<T> {
    x: T,
    pad: u8,
}

fn main() {
    let mut w = Wrap::<u64> { x: 0, pad: 0 };
    unsafe {
        (&mut w as *mut Wrap<u64> as *mut u8).write(1);
        (&mut w as *mut Wrap<u64> as *mut u8).add(8).write(7);
        println!("{}", w.x);
        println!("{}", w.pad);
    }
}
"#,
        "1\n7\n",
    );
}

// Two recorded instances of one definition make the bare
// `Ty::Adt` name ambiguous: the byte view must decline
// rather than denominate the place by a guessed instance
// (see `adt_instance_key`). A per-body denomination
// channel, if one is ever recorded, updates this test
// deliberately.
#[test]
fn run_mode_generic_union_byte_write_two_instances_stays_declined() {
    run_err(
        "generic-two-instances",
        r#"
union BagOfBits<T: Copy> {
    val: T,
    bytes: [u8; 8],
}

fn main() {
    let small = BagOfBits::<u32> { val: 7 };
    println!("{}", unsafe { small.val });
    let mut u = BagOfBits::<u64> { val: 0 };
    unsafe {
        (&mut u as *mut BagOfBits<u64> as *mut u8).write(1);
        println!("{}", u.val);
    }
}
"#,
    );
}

// A byte write displaced into a const-seeded static
// `MaybeUninit<T>` (Issue #513): the write resolves
// through the union pointer cast to a byte position of
// the union's storage, and the payload read through
// `assume_init_ref` must see it. The displacement used
// to absorb into the union value's field vector as an
// element index (the fields of an ADT are not elements
// of the byte's size), landing one past the held field
// where the write was silently dropped; and a write
// that did land converted the union to its byte form,
// which the payload descent's plain field steps then
// refused to read.
#[test]
fn run_mode_maybeuninit_static_byte_write_displaced_payload_read() {
    run_ok(
        "maybeuninit-static-displaced",
        r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

static CELL: Cell0 = Cell0(UnsafeCell::new(MaybeUninit::new(0u32)));
struct Cell0(UnsafeCell<MaybeUninit<u32>>);
unsafe impl Sync for Cell0 {}

fn main() {
    unsafe {
        let p = CELL.0.get();
        let bp = p as *mut u8;
        *bp.add(1) = 0xAB;
        println!("{:x}", (*p).assume_init_ref());
    }
}
"#,
        "ab00\n",
    );
}

// The same write against a nonzero seed: the bytes the
// write does not touch survive it.
#[test]
fn run_mode_maybeuninit_static_byte_write_preserves_seed_bytes() {
    run_ok(
        "maybeuninit-static-seed",
        r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

static CELL: Cell0 = Cell0(UnsafeCell::new(MaybeUninit::new(0x11223344u32)));
struct Cell0(UnsafeCell<MaybeUninit<u32>>);
unsafe impl Sync for Cell0 {}

fn main() {
    unsafe {
        let p = CELL.0.get();
        let bp = p as *mut u8;
        *bp.add(1) = 0xAB;
        println!("{:x}", (*p).assume_init_ref());
    }
}
"#,
        "1122ab44\n",
    );
}

// A byte write at the base of the union's storage with
// no displacement is a byte of the storage, not a
// replacement for the union value: the seed's other
// bytes survive.
#[test]
fn run_mode_maybeuninit_static_byte_write_at_base_preserves_seed() {
    run_ok(
        "maybeuninit-static-base",
        r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

static CELL: Cell0 = Cell0(UnsafeCell::new(MaybeUninit::new(0x11223344u32)));
struct Cell0(UnsafeCell<MaybeUninit<u32>>);
unsafe impl Sync for Cell0 {}

fn main() {
    unsafe {
        let p = CELL.0.get();
        let bp = p as *mut u8;
        *bp = 0xAB;
        println!("{:x}", (*p).assume_init_ref());
    }
}
"#,
        "112233ab\n",
    );
}

// A displaced write followed by a base write compose in
// the storage: each byte lands at its own position.
#[test]
fn run_mode_maybeuninit_static_byte_writes_compose() {
    run_ok(
        "maybeuninit-static-compose",
        r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

static CELL: Cell0 = Cell0(UnsafeCell::new(MaybeUninit::new(0u32)));
struct Cell0(UnsafeCell<MaybeUninit<u32>>);
unsafe impl Sync for Cell0 {}

fn main() {
    unsafe {
        let p = CELL.0.get();
        let bp = p as *mut u8;
        *bp.add(1) = 0xAB;
        *bp.add(0) = 0x11;
        println!("{:x}", (*p).assume_init_ref());
    }
}
"#,
        "ab11\n",
    );
}

// The last byte of the payload's storage.
#[test]
fn run_mode_maybeuninit_static_byte_write_last_byte() {
    run_ok(
        "maybeuninit-static-last",
        r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

static CELL: Cell0 = Cell0(UnsafeCell::new(MaybeUninit::new(0u32)));
struct Cell0(UnsafeCell<MaybeUninit<u32>>);
unsafe impl Sync for Cell0 {}

fn main() {
    unsafe {
        let p = CELL.0.get();
        let bp = p as *mut u8;
        *bp.add(3) = 0xAB;
        println!("{:x}", (*p).assume_init_ref());
    }
}
"#,
        "ab000000\n",
    );
}

// A wide payload: a byte displaced into a u64
// `MaybeUninit`.
#[test]
fn run_mode_maybeuninit_static_byte_write_u64_payload() {
    run_ok(
        "maybeuninit-static-u64",
        r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

static CELL: Cell0 = Cell0(UnsafeCell::new(MaybeUninit::new(0u64)));
struct Cell0(UnsafeCell<MaybeUninit<u64>>);
unsafe impl Sync for Cell0 {}

fn main() {
    unsafe {
        let p = CELL.0.get();
        let bp = p as *mut u8;
        *bp.add(5) = 0xAB;
        println!("{:x}", (*p).assume_init_ref());
    }
}
"#,
        "ab0000000000\n",
    );
}

// The local shape: a byte write through a byte pointer
// cast from a local `MaybeUninit` pointer, read back
// through `assume_init_ref`.
#[test]
fn run_mode_maybeuninit_local_byte_write_payload_read() {
    run_ok(
        "maybeuninit-local",
        r#"
use std::mem::MaybeUninit;

fn main() {
    unsafe {
        let mut m = MaybeUninit::<u32>::new(0);
        let p = &mut m as *mut MaybeUninit<u32>;
        let bp = p as *mut u8;
        *bp.add(1) = 0xAB;
        println!("{:x}", *m.assume_init_ref());
    }
}
"#,
        "ab00\n",
    );
}

// Byte reads back through the same byte pointer see the
// written byte and the seed's untouched byte.
#[test]
fn run_mode_maybeuninit_static_byte_readback_through_pointer() {
    run_ok(
        "maybeuninit-static-byteread",
        r#"
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

static CELL: Cell0 = Cell0(UnsafeCell::new(MaybeUninit::new(0u32)));
struct Cell0(UnsafeCell<MaybeUninit<u32>>);
unsafe impl Sync for Cell0 {}

fn main() {
    unsafe {
        let p = CELL.0.get();
        let bp = p as *mut u8;
        *bp.add(1) = 0xAB;
        println!("{:x} {:x}", *bp.add(1), *bp.add(0));
    }
}
"#,
        "ab 0\n",
    );
}

// A byte-displaced store into an element of a
// `[MaybeUninit<u32>; 2]` array: the pointer cast from the
// array's base denotes a flat byte position across the
// elements, and the payload read through `assume_init_ref`
// sees the spliced byte. On the base tree the store stops
// with "write through a pointer displaced 5 bytes from a
// value boundary": the byte view priced the array's
// element size with the bare-name walk, the element type
// names `MaybeUninit` by its bare definition name (the IR
// types erase generic arguments), and no recorded layout
// answers to that name, so the position walk declined and
// the projected assignment judged the flat offset instead.
#[test]
fn run_mode_maybeuninit_array_displaced_byte_write_second_element() {
    run_ok(
        "maybeuninit-array-displaced-write-elem1",
        r#"
use std::mem::MaybeUninit;

fn main() {
    unsafe {
        let mut arr: [MaybeUninit<u32>; 2] = [MaybeUninit::new(0x11223344), MaybeUninit::new(0x55667788)];
        let base = arr.as_mut_ptr() as *mut u8;
        *base.add(5) = 0xAB;
        println!("{:x} {:x}", arr[0].assume_init_ref(), arr[1].assume_init_ref());
    }
}
"#,
        "11223344 5566ab88\n",
    );
}

// The same store displaced into the first element.
#[test]
fn run_mode_maybeuninit_array_displaced_byte_write_first_element() {
    run_ok(
        "maybeuninit-array-displaced-write-elem0",
        r#"
use std::mem::MaybeUninit;

fn main() {
    unsafe {
        let mut arr: [MaybeUninit<u32>; 2] = [MaybeUninit::new(0x11223344), MaybeUninit::new(0x55667788)];
        let base = arr.as_mut_ptr() as *mut u8;
        *base.add(1) = 0xAB;
        println!("{:x} {:x}", arr[0].assume_init_ref(), arr[1].assume_init_ref());
    }
}
"#,
        "1122ab44 55667788\n",
    );
}

// The same store displaced exactly onto the second
// element's first byte.
#[test]
fn run_mode_maybeuninit_array_displaced_byte_write_element_boundary() {
    run_ok(
        "maybeuninit-array-displaced-write-boundary",
        r#"
use std::mem::MaybeUninit;

fn main() {
    unsafe {
        let mut arr: [MaybeUninit<u32>; 2] = [MaybeUninit::new(0x11223344), MaybeUninit::new(0x55667788)];
        let base = arr.as_mut_ptr() as *mut u8;
        *base.add(4) = 0xAB;
        println!("{:x} {:x}", arr[0].assume_init_ref(), arr[1].assume_init_ref());
    }
}
"#,
        "11223344 556677ab\n",
    );
}

// A byte-displaced read of the array's storage prices the
// same position walk: the byte at flat offset 5 is the
// second element's second byte.
#[test]
fn run_mode_maybeuninit_array_displaced_byte_read() {
    run_ok(
        "maybeuninit-array-displaced-read",
        r#"
use std::mem::MaybeUninit;

fn main() {
    unsafe {
        let arr: [MaybeUninit<u32>; 2] = [MaybeUninit::new(0x11223344), MaybeUninit::new(0x55667788)];
        let base = arr.as_ptr() as *const u8;
        println!("{:x}", *base.add(5));
    }
}
"#,
        "77\n",
    );
}
