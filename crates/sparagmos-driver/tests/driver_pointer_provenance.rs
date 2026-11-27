//! Pointer equality / provenance / synthetic-address regression
//! tests (Issue #529): the shapes the rustc suite's
//! `equal-pointers-unequal` cluster and its neighbours pin --
//! EII statics sharing one address, a packed field's unaligned
//! address, zero-sized types at their dangling addresses, and
//! raw fat pointers (slice, dyn, and DST-struct) ordering by
//! (data address, metadata word) rather than by pointee.
//!
//! The dyn and DST-struct cases failed on the unmodified
//! interpreter: dyn raw-pointer comparisons fell through to
//! the pointee comparison (a type-mismatch error for
//! differently-shaped pointees at one address), the ordering
//! of two word-array transmutations of dyn pointers errored
//! the same way, and DST-struct field pointers whose addresses
//! hash (their field sizes are ambiguous from the bare ADT
//! names) ordered by hash instead of by field position.

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

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sparagmos-pp-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    register_scratch_dir(&dir);
    dir
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
        .arg("--edition")
        .arg("2021")
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .expect("failed to run sparagmos-driver")
}

fn assert_run(tag: &str, src: &str, expected_stdout: &str) {
    let dir = temp_dir(tag);
    let src = fixture(&dir, "prog.rs", src);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "program must run to completion; stderr={stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        expected_stdout,
        "stdout mismatch; stderr={stderr}"
    );
}

// tests/ui/eii/static/same_address.rs: an EII declaration and
// its implementation are one storage, so their addresses are
// equal.
#[test]
fn eii_static_shares_its_implementations_address() {
    let src = r#"#![feature(extern_item_impls)]
#[eii(hello)]
static HELLO: u64;
#[hello]
static HELLO_IMPL: u64 = 5;
fn main() {
    assert_eq!(
        &HELLO as *const u64 as usize,
        &HELLO_IMPL as *const u64 as usize,
    );
    println!("ok");
}
"#;
    assert_run("eii-addr", src, "ok\n");
}

// tests/ui/mir/alignment/packed.rs: a packed field's address
// is its base plus its packed offset (1), not the aligned
// offset its type would suggest, so it is not aligned.
#[test]
fn packed_field_address_is_unaligned() {
    let src = r#"
#[repr(packed)]
struct Misaligner {
    _head: u8,
    tail: u64,
}

fn main() {
    let memory = [Misaligner { _head: 0, tail: 0}, Misaligner { _head: 0, tail: 0}];
    let ptr0 = std::ptr::addr_of!(memory[0].tail);
    let ptr1 = std::ptr::addr_of!(memory[0].tail);
    assert!(!ptr0.is_aligned() || !ptr1.is_aligned());
    unsafe {
        let ptr = std::ptr::addr_of!(memory[0]);
        let _tail = (*ptr).tail;
        let ptr = std::ptr::addr_of!(memory[1]);
        let _tail = (*ptr).tail;
    }
    println!("ok");
}
"#;
    assert_run("packed", src, "ok\n");
}

// tests/ui/consts/zst_no_llvm_alloc.rs: a zero-sized type's
// reference is its alignment as an address, displaced exactly
// by wrapping byte offsets, while a real zero-sized static
// keeps its own unique address.
#[test]
fn zst_references_price_at_their_alignment() {
    let src = r#"
#[repr(align(4))]
struct Foo;
static FOO: Foo = Foo;
const A: *const () = (&() as *const ()).wrapping_byte_add(2);
const B: *const () = (&Foo as *const _ as *const ()).wrapping_byte_add(usize::MAX);
const C: *const () = (&Foo as *const _ as *const ()).wrapping_byte_add(2);
fn main() {
    let x: &'static () = &();
    assert_eq!(x as *const () as usize, 1);
    let x: &'static Foo = &Foo;
    assert_eq!(x as *const Foo as usize, 4);
    assert_eq!(A.addr(), 3);
    assert_eq!(B.addr(), 3);
    assert_eq!(C.addr(), 6);
    #[allow(dangling_pointers_from_temporaries)]
    {
        assert_eq!(<Vec<i32>>::new().as_ptr(), <&[i32]>::default().as_ptr());
        assert_eq!(<Box<[i32]>>::default().as_ptr(), (&[]).as_ptr());
    };
    assert_ne!(&FOO as *const Foo as usize, 4);
    println!("ok");
}
"#;
    assert_run("zst", src, "ok\n");
}

// tests/ui/stdlib-unit-tests/raw-fat-ptr.rs, in full: raw
// fat pointers order by (data address, metadata) -- slice
// pointers by address then length, dyn pointers by address
// then vtable (also through a transmute to a word pair), and
// DST-struct pointers by their fields' positions.
#[test]
fn raw_fat_pointer_ordering() {
    let src = r#"
use std::mem;

fn assert_inorder<T: PartialEq + PartialOrd>(a: &[T]) {
    for i in 0..a.len() {
        for j in 0..a.len() {
            if i < j {
                assert!(a[i] < a[j]);
                assert!(a[i] <= a[j]);
                assert!(!(a[i] == a[j]));
                assert!(a[i] != a[j]);
                assert!(!(a[i] >= a[j]));
                assert!(!(a[i] > a[j]));
            } else if i == j {
                assert!(!(a[i] < a[j]));
                assert!(a[i] <= a[j]);
                assert!(a[i] == a[j]);
                assert!(!(a[i] != a[j]));
                assert!(a[i] >= a[j]);
                assert!(!(a[i] > a[j]));
            } else {
                assert!(!(a[i] < a[j]));
                assert!(!(a[i] <= a[j]));
                assert!(!(a[i] == a[j]));
                assert!(a[i] != a[j]);
                assert!(a[i] >= a[j]);
                assert!(a[i] > a[j]);
            }
        }
    }
}

trait Foo { fn foo(&self) -> usize; }
impl<T> Foo for T {
    fn foo(&self) -> usize {
        mem::size_of::<T>()
    }
}

#[allow(dead_code)]
struct S<T:?Sized>(u32, T);

fn main() {
    let mut array = [0,1,2,3,4];
    let mut array2 = [5,6,7,8,9];

    let mut ptrs: Vec<*const [u8]> = vec![
        &array[0..0], &array[0..1], &array, &array[1..]
    ];

    let array_addr = &array as *const [u8] as *const u8 as usize;
    let array2_addr = &array2 as *const [u8] as *const u8 as usize;
    if array2_addr < array_addr {
        ptrs.insert(0, &array2);
    } else {
        ptrs.push(&array2);
    }
    assert_inorder(&ptrs);

    let mut ptrs: Vec<*mut [u8]> = vec![
        &mut array[0..0], &mut array[0..1], &mut array, &mut array[1..]
    ];

    let array_addr = &mut array as *mut [u8] as *mut u8 as usize;
    let array2_addr = &mut array2 as *mut [u8] as *mut u8 as usize;
    if array2_addr < array_addr {
        ptrs.insert(0, &mut array2);
    } else {
        ptrs.push(&mut array2);
    }
    assert_inorder(&ptrs);

    let mut u8_ = (0u8, 1u8);
    let mut u32_ = (4u32, 5u32);

    let buf: &mut [*const dyn Foo] = &mut [
        &u8_, &u8_.0,
        &u32_, &u32_.0,
    ];
    buf.sort_by(|u,v| {
        let u : [*const (); 2] = unsafe { mem::transmute(*u) };
        let v : [*const (); 2] = unsafe { mem::transmute(*v) };
        u.cmp(&v)
    });
    assert_inorder(buf);

    let buf: &mut [*mut dyn Foo] = &mut [
        &mut u8_, &mut u8_.0,
        &mut u32_, &mut u32_.0,
    ];
    buf.sort_by(|u,v| {
        let u : [*const (); 2] = unsafe { mem::transmute(*u) };
        let v : [*const (); 2] = unsafe { mem::transmute(*v) };
        u.cmp(&v)
    });
    assert_inorder(buf);

    let ss: (S<[u8; 2]>,
             S<[u8; 3]>,
             S<[u8; 2]>) = (
        S(7, [8, 9]),
        S(10, [11, 12, 13]),
        S(4, [5, 6])
    );
    assert_inorder(&[
        &ss.0 as *const S<[u8]>,
        &ss.1 as *const S<[u8]>,
        &ss.2 as *const S<[u8]>
    ]);
    println!("ok");
}
"#;
    assert_run("raw-fat-ptr", src, "ok\n");
}

// A zero-length `copy_from_slice` between subslices of two
// distinct Vec allocations: the `ptr::copy_nonoverlapping`
// ub_checks precondition (aligned, non-null, non-overlapping)
// must accept the pair. The two slice pointers price through
// the Vec/RawVec field chain, whose steps the recorded field
// layouts also price; at this change's original base the two
// empty ranges priced onto one address and the precondition
// aborted the run (the driver/corpus failure that blocked
// PR #921). The full-length copy after it pins that the same
// field pricing still lands the elements exactly.
#[test]
fn zero_length_copy_between_distinct_vecs_passes_ub_checks() {
    let src = r#"
fn main() {
    let a: Vec<i32> = (0..10).collect();
    let mut b: Vec<i32> = vec![0; 10];
    b[..0].copy_from_slice(&a[..0]);
    b.copy_from_slice(&a);
    println!("copied {:?}", b);
    let disjoint = (a.as_ptr() as usize) != (b.as_ptr() as usize);
    println!("disjoint {}", disjoint);
}
"#;
    assert_run(
        "zero-length-copy-distinct-vecs",
        src,
        "copied [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\ndisjoint true\n",
    );
}
