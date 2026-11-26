//! Transmute regrouping tests: build small fixture programs
//! in a temp dir, execute them with `sparagmos-driver run`,
//! and check the exit-code contract (0 = normal termination,
//! 101 = guest panic, 2 = could not convert/execute, with the
//! reason on stderr). The fixtures cover the bit-stream
//! reinterpretation (width-crossing array regrouping at any
//! total width, with per-element uninit propagation), the
//! wrapped-array and `Simd<T, N>` storage crossings, the
//! fat-pointer <-> DST-wrapper-struct identity, the
//! (pointer, length) tuple <-> fat reference crossing, and
//! the thin pointer <-> union crossing, and the raw-pointer
//! cast regrouping of a slice's elements (a `[[u8; 3]; 2]`
//! storage ranged over as `[[u8; 2]]`), for the element
//! reads, the comparisons, and the `Debug` formatting of
//! the regrouped view alike, and for the reads and writes
//! an `iter_mut` over the regrouped view performs through
//! the element references its steps hand out.

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

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-transmute-regroup-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
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

/// `[i16; 50]` -> `[f32; 25]`: each target lane is the
/// little-endian regrouping of two source elements (the
/// lower-indexed element in the low half). The expected bit
/// patterns are computed here, independently of the driver,
/// and the round trip back must reproduce the source.
#[test]
fn regroup_i16_to_f32_bit_exact() {
    let stdout = run_ok(
        "regroup-widen",
        "regroup_widen.rs",
        r#"
fn main() {
    let mut a = [0i16; 50];
    for j in 0..25 {
        a[2 * j] = (j as i32 * 7 - 100) as i16;
        a[2 * j + 1] = (1000 - j as i32 * 13) as i16;
    }
    let b: [f32; 25] = unsafe { std::mem::transmute(a) };
    for f in b {
        println!("{}", f.to_bits());
    }
    let c: [i16; 50] = unsafe { std::mem::transmute(b) };
    for (x, y) in a.iter().zip(c.iter()) {
        assert_eq!(x, y);
    }
    println!("back-ok");
}
"#,
    );
    let mut expected = String::new();
    for j in 0..25i32 {
        let lo = (j * 7 - 100) as i16;
        let hi = (1000 - j * 13) as i16;
        let bits = ((hi as u16 as u32) << 16) | (lo as u16 as u32);
        expected.push_str(&format!("{bits}\n"));
    }
    expected.push_str("back-ok\n");
    assert_eq!(stdout, expected);
}

/// `[u32; 10]` -> `[u8; 40]`: the 1-for-4 narrowing
/// direction; each word's bytes appear least-significant
/// first.
#[test]
fn regroup_u32_to_u8_bit_exact() {
    let stdout = run_ok(
        "regroup-narrow",
        "regroup_narrow.rs",
        r#"
fn main() {
    let mut a = [0u32; 10];
    for j in 0..10 {
        a[j] = 0x0102_0304u32.wrapping_add((j as u32) << 8);
    }
    let b: [u8; 40] = unsafe { std::mem::transmute(a) };
    for byte in b {
        println!("{byte}");
    }
}
"#,
    );
    let mut expected = String::new();
    for j in 0..10u32 {
        let w = 0x0102_0304u32.wrapping_add(j << 8);
        for k in 0..4 {
            expected.push_str(&format!("{}\n", (w >> (8 * k)) & 0xff));
        }
    }
    assert_eq!(stdout, expected);
}

/// A one-field wrapper over a long array -> the plain byte
/// array (and back): `Mu64([u64; 4])` -> `[u8; 32]`.
#[test]
fn wrapped_array_to_bytes() {
    let stdout = run_ok(
        "wrapped-array",
        "wrapped_array.rs",
        r#"
struct Mu64([u64; 4]);
fn main() {
    let m = Mu64([0x0807_0605_0403_0201, 1, 1, 1]);
    let b: [u8; 32] = unsafe { std::mem::transmute(m) };
    for i in 0..8 {
        println!("{}", b[i]);
    }
    println!("{}", b[8]);
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        bytes[i] = (i * 5 + 1) as u8;
    }
    let m2: Mu64 = unsafe { std::mem::transmute(bytes) };
    println!("{}", m2.0[0]);
    println!("{}", m2.0[3]);
}
"#,
    );
    let mut expected = String::new();
    for i in 1..=8u64 {
        expected.push_str(&format!("{i}\n"));
    }
    expected.push_str("1\n");
    let word = |k: u64| -> u64 {
        let mut w = 0u64;
        for i in 0..8u64 {
            let byte = ((k * 8 + i) * 5 + 1) as u64;
            w |= byte << (8 * i);
        }
        w
    };
    expected.push_str(&format!("{}\n", word(0)));
    expected.push_str(&format!("{}\n", word(3)));
    assert_eq!(stdout, expected);
}

/// Uninit propagation per element: lanes 10 and 11 of the
/// source are uninit, so exactly target lane 5 is poisoned;
/// every other lane reads back bit-exact, and reading the
/// poisoned lane is an uninitialised-read error, not a
/// fabricated value.
#[test]
fn regroup_uninit_lane_propagation() {
    let stdout = run_ok(
        "regroup-uninit",
        "regroup_uninit.rs",
        r#"
use std::mem::MaybeUninit;
fn main() {
    let mut a = [MaybeUninit::<i16>::uninit(); 50];
    for i in 0..50 {
        a[i] = MaybeUninit::new((i as i32 * 3 - 20) as i16);
    }
    a[10] = MaybeUninit::uninit();
    a[11] = MaybeUninit::uninit();
    let b: [MaybeUninit<f32>; 25] = unsafe { std::mem::transmute(a) };
    for (j, lane) in b.iter().enumerate() {
        if j == 5 {
            continue;
        }
        let bits = unsafe { lane.assume_init_ref() }.to_bits();
        let lo = (2 * j as i32 * 3 - 20) as i16;
        let hi = ((2 * j + 1) as i32 * 3 - 20) as i16;
        let want = ((hi as u16 as u32) << 16) | (lo as u16 as u32);
        assert_eq!(bits, want, "lane {j}");
    }
    println!("lanes-ok");
}
"#,
    );
    assert_eq!(stdout, "lanes-ok\n");

    let dir = temp_dir("regroup-uninit-read");
    let path = fixture(
        &dir,
        "regroup_uninit_read.rs",
        r#"
use std::mem::MaybeUninit;
fn main() {
    let mut a = [MaybeUninit::<i16>::uninit(); 50];
    for i in 0..50 {
        a[i] = MaybeUninit::new((i as i32 * 3 - 20) as i16);
    }
    a[10] = MaybeUninit::uninit();
    a[11] = MaybeUninit::uninit();
    let b: [MaybeUninit<f32>; 25] = unsafe { std::mem::transmute(a) };
    let bad: f32 = unsafe { b[5].assume_init() };
    println!("{}", bad.to_bits());
}
"#,
    );
    let out = run_driver(&path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "reading the poisoned lane must fail; stderr={stderr}"
    );
    // The driver's error form for reading an uninitialised
    // `f32` through `assume_init` (the same form a plain
    // uninit `MaybeUninit<f32>` read produces): the value
    // behind the poisoned lane is the uninitialised marker,
    // whose assignment to an `f32` place is rejected.
    assert!(
        stderr.contains("expected Float(F32), found unit"),
        "the failure is the uninitialised read: {stderr}"
    );
    assert!(
        out.stdout.is_empty(),
        "no fabricated lane value is printed: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

/// Thin pointer <-> union: `*const u8` transmuted into a
/// union whose field 0 is the pointer, read back through
/// the reference field, and transmuted back out.
#[test]
fn union_pointer_roundtrip() {
    let stdout = run_ok(
        "union-ptr",
        "union_ptr.rs",
        r#"
union U2<'a> {
    p: *const u8,
    r: &'a u8,
}
fn main() {
    let x = 41u8;
    let p0: *const u8 = &x;
    let u = unsafe { std::mem::transmute::<*const u8, U2>(p0) };
    let r = unsafe { u.r };
    println!("{}", *r);
    let p = unsafe { std::mem::transmute::<U2, *const u8>(u) };
    println!("{}", unsafe { *p });
}
"#,
    );
    assert_eq!(stdout, "41\n41\n");
}

/// Fat reference <-> reference to a DST-wrapper struct:
/// `&[u8]` -> `&X` where `struct X([u8])`, and the sized-
/// fields-before-the-tail form (`struct Baz { f1, f2: str }`
/// reached from a `&str` and read back).
#[test]
fn fat_ref_to_dst_struct() {
    let stdout = run_ok(
        "fat-dst",
        "fat_dst.rs",
        r#"
pub struct X([u8]);
fn xlen(x: &X) -> usize {
    match *x {
        X(ref s) => s.len(),
    }
}
struct Baz {
    f1: usize,
    f2: str,
}
fn main() {
    let s: &[u8] = &[10, 20, 30];
    let x: &X = unsafe { std::mem::transmute(s) };
    println!("{}", xlen(x));
    let text: &str = "hello";
    let b: &Baz = unsafe { std::mem::transmute(text) };
    let back2: &str = unsafe { std::mem::transmute(b) };
    println!("{back2}");
}
"#,
    );
    assert_eq!(stdout, "3\nhello\n");
}

/// (pointer, length) tuple -> fat reference to a
/// DST-wrapper struct: the `PackedSized::unsize` transmute.
#[test]
fn tuple_to_fat_dst_ref() {
    let stdout = run_ok(
        "tuple-fat",
        "tuple_fat.rs",
        r#"
#[repr(packed, C)]
pub struct PackedSized {
    f: u8,
    d: [u32; 3],
}
#[repr(packed, C)]
pub struct PackedUnsized {
    f: u8,
    d: [u32],
}
impl PackedSized {
    fn unsize(&self, len: usize) -> &PackedUnsized {
        unsafe { std::mem::transmute((self, len)) }
    }
}
fn main() {
    let s = PackedSized { f: 9, d: [100, 200, 300] };
    let u = s.unsize(3);
    let back: (&PackedSized, usize) =
        unsafe { std::mem::transmute(u) };
    println!("{}", back.1);
    println!("{}", back.0.f);
}
"#,
    );
    assert_eq!(stdout, "3\n9\n");
}

/// A local `#[repr(simd)]` wrapper whose storage length is
/// a const parameter the type representation does not
/// carry: `Simd<u8, 2>` <-> `[u8; 2]`, the count fixed by
/// the array side.
#[test]
fn simd_local_wrap_transmute() {
    let stdout = run_ok(
        "simd-local",
        "simd_local.rs",
        r#"
#![feature(repr_simd)]
#![allow(incomplete_features)]
#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);
fn main() {
    let v = Simd::<u8, 2>([7, 9]);
    let a: [u8; 2] = unsafe { std::mem::transmute(v) };
    println!("{} {}", a[0], a[1]);
    let w: Simd<u8, 2> = unsafe { std::mem::transmute([3u8, 4]) };
    let b: [u8; 2] = unsafe { std::mem::transmute(w) };
    println!("{} {}", b[0], b[1]);
}
"#,
    );
    assert_eq!(stdout, "7 9\n3 4\n");
}

/// A raw-pointer cast regroups a slice's elements across
/// the storage's inner array boundary: the `[[u8; 3]; 2]`
/// storage ranged over as `[[u8; 2]]` reads as the byte
/// stream re-chunked into pairs, for the element reads, the
/// slice equality, and the slice comparison alike (the
/// `tests/ui/cast/cast-rfc0401.rs` shape).
#[test]
fn cast_slice_regroups_elements() {
    let stdout = run_ok(
        "cast-regroup",
        "cast_regroup.rs",
        r#"
fn main() {
    let l: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    let p: *const [[u8; 3]] = &l;
    let p: &[[u8; 2]] = unsafe { &*(p as *const [[u8; 2]]) };
    assert_eq!(p, [[3, 2], [6, 4]]);
    let e0 = p[0];
    let e1 = p[1];
    assert_eq!(e0, [3, 2]);
    assert_eq!(e1, [6, 4]);
    assert!(p != &[[3u8, 2], [6, 5]][..]);
    assert_eq!(
        p.partial_cmp(&[[3u8, 2], [6, 5]][..]),
        Some(std::cmp::Ordering::Less)
    );
    assert_eq!(
        p.partial_cmp(&[[3u8, 2], [6, 4]][..]),
        Some(std::cmp::Ordering::Equal)
    );
    println!("regroup-ok");
}
"#,
    );
    assert_eq!(stdout, "regroup-ok\n");
}

/// The regrouped comparison genuinely judges the regrouped
/// bytes: an expectation that differs in the regrouped
/// second element (the byte past the storage's inner array
/// boundary) fails the assertion.
#[test]
fn cast_slice_regroup_mismatch_panics() {
    let dir = temp_dir("cast-regroup-neg");
    let path = fixture(
        &dir,
        "cast_regroup_neg.rs",
        r#"
fn main() {
    let l: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    let p: *const [[u8; 3]] = &l;
    let p: &[[u8; 2]] = unsafe { &*(p as *const [[u8; 2]]) };
    assert_eq!(p, [[3, 2], [6, 5]]);
    println!("must-not-print");
}
"#,
    );
    let out = run_driver(&path);
    assert_eq!(
        out.status.code(),
        Some(101),
        "a mismatching regrouped expectation must panic; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The `Debug` formatting of a regrouped slice view renders
/// the access-side element shape, not the storage's birth
/// shape: the byte view rebuilds the elements under the
/// formatting tag's element type, on every route a slice
/// value reaches the formatter through (a direct argument, a
/// double reference, a tuple or array element, a `Vec`
/// entry, a derived-`Debug` struct field, the alternate
/// pretty form), and an element argument reached through
/// the cast is composed at the access-side offset. The
/// non-regrouped control keeps the birth shape, and a
/// `[u16; 2]` storage ranged over as bytes reads back as
/// its byte stream.
#[test]
fn cast_slice_regroup_debug_output() {
    let stdout = run_ok(
        "cast-regroup-debug",
        "cast_regroup_debug.rs",
        r#"
#[derive(Debug)]
struct Wrap<'a> { s: &'a [[u8; 2]] }
fn main() {
    let l: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    let p: *const [[u8; 3]] = &l;
    let p: &[[u8; 2]] = unsafe { &*(p as *const [[u8; 2]]) };
    println!("direct: {:?}", p);
    println!("elem: {:?}", p[0]);
    println!("tuple: {:?}", (p, 7u8));
    let q: &[[u8; 3]] = &l;
    println!("control: {:?}", q);
    println!("deref-ref: {:?}", &p);
    println!("arr-of-ref: {:?}", [p]);
    let v: Vec<&[[u8; 2]]> = vec![p];
    println!("vec: {:?}", v);
    println!("pretty: {:#?}", p);
    println!("wrap: {:?}", Wrap { s: p });
    let m: [u16; 2] = [0x0201, 0x0403];
    let b: &[u8] = unsafe { &*(&m as *const [u16; 2] as *const [u8; 4] as *const [u8]) };
    println!("bytes-of-u16: {:?}", b);
    let back: &[u16] = unsafe { &*(&m as *const [u16; 2] as *const [u16]) };
    println!("u16-slice: {:?}", back);
}
"#,
    );
    assert_eq!(
        stdout,
        "direct: [[3, 2], [6, 4]]\n\
         elem: [3, 2]\n\
         tuple: ([[3, 2], [6, 4]], 7)\n\
         control: [[3, 2, 6], [4, 5, 1]]\n\
         deref-ref: [[3, 2], [6, 4]]\n\
         arr-of-ref: [[[3, 2], [6, 4]]]\n\
         vec: [[[3, 2], [6, 4]]]\n\
         pretty: [\n\
         \u{20}   [\n\
         \u{20}       3,\n\
         \u{20}       2,\n\
         \u{20}   ],\n\
         \u{20}   [\n\
         \u{20}       6,\n\
         \u{20}       4,\n\
         \u{20}   ],\n\
         ]\n\
         wrap: Wrap { s: [[3, 2], [6, 4]] }\n\
         bytes-of-u16: [1, 2, 3, 4]\n\
         u16-slice: [513, 1027]\n"
    );
}

/// Iteration over a regrouped slice yields the elements
/// under the cast's shape: the `iter` / `into_iter`
/// snapshots compose their elements from the storage's
/// bytes under the element type the iterator's own type
/// names, as the element reads and the formatting do.
#[test]
fn cast_slice_regroup_iteration() {
    let stdout = run_ok(
        "cast-regroup-iteration",
        "cast_regroup_iteration.rs",
        r#"
fn main() {
    let l: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    let p: *const [[u8; 3]] = &l;
    let p: &[[u8; 2]] = unsafe { &*(p as *const [[u8; 2]]) };
    for e in p.iter() {
        println!("iter: {:?}", e);
    }
    let v: Vec<[u8; 2]> = p.iter().copied().collect();
    println!("collected: {:?}", v);
    let mut sum = 0u32;
    for e in p {
        for x in e {
            sum += *x as u32;
        }
    }
    println!("sum: {}", sum);
    println!("count: {}", p.iter().count());
    for (i, e) in p.iter().enumerate() {
        println!("enum {}: {:?}", i, e);
    }
    for e in p.iter().rev() {
        println!("rev: {:?}", e);
    }
    // A whole-slice re-slice of the regrouped view keeps
    // the denomination (its base starts at the storage's
    // first byte).
    let q: &[[u8; 2]] = &p[..];
    for e in q.iter() {
        println!("whole: {:?}", e);
    }
    // The slice adapter steps compose their elements the
    // same way.
    for w in p.windows(2) {
        println!("win: {:?}", w);
    }
    for c in p.chunks(1) {
        println!("chunk: {:?}", c);
    }
    // A widening regrouping: the same storage ranged over
    // as `u16` elements.
    let s: &[u16] = unsafe { &*(&l as *const [[u8; 3]] as *const [u16]) };
    for x in s.iter() {
        println!("u16: {}", x);
    }
    // The non-regrouped control is unchanged.
    let plain: &[u8] = &[10, 20, 30];
    for x in plain.iter() {
        println!("plain: {}", x);
    }
}
"#,
    );
    assert_eq!(
        stdout,
        "iter: [3, 2]\n\
         iter: [6, 4]\n\
         collected: [[3, 2], [6, 4]]\n\
         sum: 15\n\
         count: 2\n\
         enum 0: [3, 2]\n\
         enum 1: [6, 4]\n\
         rev: [6, 4]\n\
         rev: [3, 2]\n\
         whole: [3, 2]\n\
         whole: [6, 4]\n\
         win: [[3, 2], [6, 4]]\n\
         chunk: [[3, 2]]\n\
         chunk: [[6, 4]]\n\
         u16: 515\n\
         u16: 1030\n\
         plain: 10\n\
         plain: 20\n\
         plain: 30\n"
    );
}

/// The nested element of a regrouped slice view through the
/// formatting machinery: the format argument for `p[1][0]`
/// reaches the formatter as a target whose steps the birth
/// type (`[[u8; 3]; 2]`) denominates differently from the
/// view (`[[u8; 2]]`), so the argument's value comes from
/// the denomination the dereference recorded on the target
/// (see `RefProj::View`). A reference to the element bound
/// first and dereferenced later resolves through the same
/// record.
#[test]
fn cast_slice_regroup_nested_fmt_element() {
    let stdout = run_ok(
        "cast-nested-fmt",
        "cast_nested_fmt.rs",
        r#"
fn main() {
    let l: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    let p: *const [[u8; 3]] = &l;
    let p: &[[u8; 2]] = unsafe { &*(p as *const [[u8; 2]]) };
    // body read (control)
    let x = p[1][0];
    println!("body: {x}");
    // fmt argument through the cast: nested element
    println!("fmt: {}", p[1][0]);
    println!("fmt-debug: {:?}", p[1][0]);
    // stored reference bound first, dereferenced later
    let r = &p[1][0];
    println!("stored: {}", *r);
    println!("stored-fmt: {:?}", r);
    // first element for contrast
    println!("fmt00: {}", p[0][0]);
    println!("fmt10-arr: {:?}", p[1]);
}
"#,
    );
    assert_eq!(
        stdout,
        "body: 6\n\
         fmt: 6\n\
         fmt-debug: 6\n\
         stored: 6\n\
         stored-fmt: 6\n\
         fmt00: 3\n\
         fmt10-arr: [6, 4]\n"
    );
}

/// A thin pointer cast regrouping the storage it points
/// at: the `[u8; 2]` the cast names is the denomination the
/// dereference records, so the index steps the `Index`
/// screen appends to the stored target price through it
/// (the second byte of the storage), not through the birth
/// type's row. The slice cast over the same storage keeps
/// its own element denomination.
#[test]
fn thin_cast_regroup_index() {
    let stdout = run_ok(
        "thin-cast",
        "thin_cast.rs",
        r#"
fn main() {
    let l: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    let r: &[u8; 2] = unsafe { &*(&l as *const [[u8; 3]; 2] as *const [u8; 2]) };
    println!("body: {}", r[1]);
    let rr = &r[1];
    println!("stored: {}", *rr);
    println!("fmt: {}", r[1]);
    let s: &[u8] = unsafe { &*(&l as *const [[u8; 3]; 2] as *const [u8; 4] as *const [u8]) };
    println!("slice-body: {}", s[3]);
    let sr = &s[3];
    println!("slice-stored: {}", *sr);
}
"#,
    );
    assert_eq!(
        stdout,
        "body: 2\n\
         stored: 2\n\
         fmt: 2\n\
         slice-body: 4\n\
         slice-stored: 4\n"
    );
}

/// A non-zero-offset re-slice of a regrouped view: the
/// `Subslice` step of `&p[1..]` prices through the
/// denomination the dereference recorded (its offset is
/// one `[u8; 2]` element in, not one birth row), for the
/// `Debug` rendering, the iteration, and the format
/// argument alike; and writes through the denominated view
/// land on the view's bytes of the birth storage.
#[test]
fn cast_slice_regroup_reslice_and_write() {
    let stdout = run_ok(
        "cast-reslice",
        "cast_reslice.rs",
        r#"
fn main() {
    let mut a: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    let base: *const [[u8; 3]] = &a;
    let p: &[[u8; 2]] = unsafe { &*(base as *const [[u8; 2]]) };
    let r = &p[1..];
    println!("reslice: {:?}", r);
    let mut sum = 0u32;
    for row in r {
        for x in row {
            sum += *x as u32;
        }
    }
    println!("reslice sum: {sum}");
    println!("reslice fmt: {}", r[0][1]);
    let r0 = &p[0..];
    println!("reslice0: {:?}", r0);
    {
        let basem: *mut [[u8; 3]] = &mut a;
        let pm: &mut [[u8; 2]] = unsafe { &mut *(basem as *mut [[u8; 2]]) };
        pm[1][0] = 40;
        let q: &mut [u8; 2] = &mut pm[0];
        q[1] = 20;
    }
    println!("after writes: {:?}", a);
}
"#,
    );
    assert_eq!(
        stdout,
        "reslice: [[6, 4]]\n\
         reslice sum: 10\n\
         reslice fmt: 4\n\
         reslice0: [[3, 2], [6, 4]]\n\
         after writes: [[3, 20, 40], [4, 5, 1]]\n"
    );
}

/// Mutable iteration over a regrouped slice: the element
/// references an `IterMut` step hands out denote their
/// elements by a byte displacement over the position the
/// cast's denomination recorded (the step's pointer
/// arithmetic, not an index step), so the reads and the
/// writes through them price through the recorded view at
/// the displaced position -- the second element's bytes
/// are the storage's third and fourth bytes, in both
/// directions.
#[test]
fn cast_slice_regroup_iter_mut() {
    let stdout = run_ok(
        "cast-regroup-iter-mut",
        "cast_regroup_iter_mut.rs",
        r#"
fn main() {
    let mut a: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    {
        let basem: *mut [[u8; 3]] = &mut a;
        let pm: &mut [[u8; 2]] = unsafe { &mut *(basem as *mut [[u8; 2]]) };
        for e in pm.iter_mut() {
            println!("elem: {:?}", e);
        }
        let mut sum = 0u32;
        for e in pm.iter_mut() {
            for x in e.iter() {
                sum += *x as u32;
            }
        }
        println!("sum: {sum}");
        for e in pm.iter_mut() {
            e[0] = 40;
            e[1] = 41;
        }
    }
    println!("after: {:?}", a);
}
"#,
    );
    assert_eq!(
        stdout,
        "elem: [3, 2]\n\
         elem: [6, 4]\n\
         sum: 15\n\
         after: [[40, 41, 40], [41, 5, 1]]\n"
    );
}

/// The back steps and the whole-element writes of a
/// mutable iteration over a regrouped view: `rev` displaces
/// the end pointer backwards over the denominated position,
/// a whole-element assignment through the yielded reference
/// splices the element's bytes at the displaced span, and
/// a scalar-element view (the same storage ranged over as
/// `u16` elements) mutates through the same pricing. The
/// non-regrouped control is unchanged.
#[test]
fn cast_slice_regroup_iter_mut_rev_and_whole() {
    let stdout = run_ok(
        "cast-regroup-iter-mut-rev",
        "cast_regroup_iter_mut_rev.rs",
        r#"
fn main() {
    let mut a: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    {
        let basem: *mut [[u8; 3]] = &mut a;
        let pm: &mut [[u8; 2]] = unsafe { &mut *(basem as *mut [[u8; 2]]) };
        for e in pm.iter_mut().rev() {
            println!("rev: {:?}", e);
            e[1] = 99;
        }
    }
    println!("after rev: {:?}", a);
    let mut b: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    {
        let basem: *mut [[u8; 3]] = &mut b;
        let pm: &mut [[u8; 2]] = unsafe { &mut *(basem as *mut [[u8; 2]]) };
        for e in pm.iter_mut() {
            *e = [9, 8];
        }
    }
    println!("after whole: {:?}", b);
    let mut c: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    {
        let s: &mut [u16] = unsafe { &mut *(&mut c as *mut [[u8; 3]] as *mut [u16]) };
        for x in s.iter_mut() {
            println!("u16: {}", x);
            *x = 7;
        }
    }
    println!("after u16: {:?}", c);
    let mut d: [[u8; 2]; 2] = [[1, 2], [3, 4]];
    for e in d.iter_mut() {
        e[0] = 50;
    }
    println!("control: {:?}", d);
}
"#,
    );
    assert_eq!(
        stdout,
        "rev: [6, 4]\n\
         rev: [3, 2]\n\
         after rev: [[3, 99, 6], [99, 5, 1]]\n\
         after whole: [[9, 8, 9], [8, 5, 1]]\n\
         u16: 515\n\
         u16: 1030\n\
         after u16: [[7, 0, 7], [0, 5, 1]]\n\
         control: [[50, 2], [50, 4]]\n"
    );
}
