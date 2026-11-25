use super::{fixture, run_driver, run_driver_extra, run_ok, temp_dir};


const TRANSMUTE_FLOAT_INT_SRC: &str = r#"
fn inp32() -> f32 { let mut s = 0.0f32; let mut i = 0; while i < 7 { s += 0.5; i += 1; } s }
fn inp64() -> f64 { let mut s = 0.0f64; let mut i = 0; while i < 7 { s += 1.0; i += 1; } s }
fn main() {
    let f = inp32();
    let bits: u32 = unsafe { std::mem::transmute(f) };
    let back: f32 = unsafe { std::mem::transmute(bits) };
    let g = inp64();
    let gbits: u64 = unsafe { std::mem::transmute(g) };
    let sg: i64 = unsafe { std::mem::transmute(g) };
    println!("{:x} {} {:x} {}", bits, back, gbits, sg);
    println!("{:x} {} {}", f.to_bits(), f32::from_bits(bits), f64::from_bits(gbits));
}
"#;

#[test]
fn run_mode_transmute_float_int_bit_reinterpretation() {
    run_ok(
        "transmute-float-int",
        TRANSMUTE_FLOAT_INT_SRC,
        "40600000 3.5 401c000000000000 4619567317775286272\n40600000 3.5 7\n",
    );
}

const TRAIT_OVERFLOW_SRC: &str = r#"
use std::ops::*;
fn main() {
    assert_eq!(i8::neg(-0x80), -0x80);
    assert_eq!(u8::add(0xff, 1), 0_u8);
    println!("ok");
}
"#;

#[test]
fn run_mode_overflow_asserts_follow_the_session_setting() {
    let dir = temp_dir("overflow-assert");
    let src = fixture(&dir, "neg.rs", TRAIT_OVERFLOW_SRC);
    // The default session checks overflow: the trait bodies'
    // overflow asserts fire and the guest panics (101).
    let out = run_driver(&src);
    assert_eq!(out.status.code(), Some(101), "default session must panic");
    // `-C overflow-checks=off`: the same asserts are not evaluated
    // and the wrapped results stand.
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-C", "--rustc-arg", "overflow-checks=off"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
    // `-C debug-assertions=no` implies overflow checks off as well.
    let out = run_driver_extra(
        &src,
        &["--rustc-arg", "-C", "--rustc-arg", "debug-assertions=no"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

const LITERAL_CMP_SRC: &str = r#"
fn main() {
    let a: u8 = 0;
    let b: u8 = 255;
    assert!(a < b, "locals failed");
    assert!(0u8 < 255u8, "consts failed");
    println!("ok");
}
"#;

#[test]
fn run_mode_literal_operands_compare_under_their_own_type() {
    // A literal-vs-literal comparison reads each literal under its
    // own recorded type: `255u8` is unsigned 255, not -1.
    run_ok("literal-cmp", LITERAL_CMP_SRC, "ok\n");
}

const TIME_SRC: &str = r#"
fn main() {
    let a = std::time::Instant::now();
    let b = std::time::Instant::now();
    assert!(b >= a);
    let s = std::time::SystemTime::now();
    let d = s.duration_since(std::time::UNIX_EPOCH).unwrap();
    assert!(d.as_secs() > 1_600_000_000);
    println!("time ok");
}
"#;

const TIME_OVERFLOW_SRC: &str = r#"
fn main() {
    let t = std::time::Instant::now();
    let _ = t + std::time::Duration::MAX;
}
"#;

#[test]
fn run_mode_time_now_and_overflow() {
    let dir = temp_dir("time");
    let src = fixture(&dir, "time.rs", TIME_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "time ok\n");
    let src = fixture(&dir, "time_overflow.rs", TIME_OVERFLOW_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(101), "stderr={stderr}");
    assert!(
        stderr.contains("overflow when adding duration to instant"),
        "stderr={stderr}"
    );
}

const SIMD_CAST_SRC: &str = r#"
#![feature(repr_simd, core_intrinsics)]
#[repr(simd)]
#[derive(Debug, Copy, Clone)]
struct Simd<T, const N: usize>([T; N]);
fn main() {
    let a: Simd<i8, 3> = Simd([10, -3, -9]);
    let b: Simd<i16, 3> = unsafe { std::intrinsics::simd::simd_cast(a) };
    println!("{:?}", b);
}
"#;

#[test]
fn run_mode_simd_cast_sign_extends() {
    // `simd_cast` from a signed element type sign-extends from
    // the source width (the shape of tests/ui/simd/
    // issue-39720.rs): -3 and -9 stay negative as i16 lanes.
    run_ok("simd-cast", SIMD_CAST_SRC, "Simd([10, -3, -9])\n");
}

const F32_BYTE_INDEX_READ_SRC: &str = r#"
static SX: f32 = 1.0;
struct Wrap { f: f32 }
fn main() {
    let x: f32 = 1.0;
    let bs = unsafe { std::slice::from_raw_parts(&x as *const f32 as *const u8, 4) };
    println!("local {} {} {} {}", bs[0], bs[1], bs[2], bs[3]);
    let ss = unsafe { std::slice::from_raw_parts(&SX as *const f32 as *const u8, 4) };
    println!("stat {} {} {} {}", ss[0], ss[1], ss[2], ss[3]);
    let w = Wrap { f: 2.5 };
    let fb = unsafe { std::slice::from_raw_parts(&w.f as *const f32 as *const u8, 4) };
    println!("field {} {} {} {}", fb[0], fb[1], fb[2], fb[3]);
    let bx = Box::new(1.0f32);
    let bb = unsafe { std::slice::from_raw_parts(&*bx as *const f32 as *const u8, 4) };
    println!("boxed {} {} {} {}", bb[0], bb[1], bb[2], bb[3]);
    let n: u32 = 0x0a0b0c0d;
    let nb = unsafe { std::slice::from_raw_parts(&n as *const u32 as *const u8, 4) };
    println!("int {} {} {} {}", nb[0], nb[1], nb[2], nb[3]);
}
"#;

#[test]
fn run_mode_byte_index_read_of_f32_storage() {
    // Index reads through a `&[u8]` view of an f32's storage
    // yield the f32's own bytes, not the widened f64 model's
    // (Issue #475): 1.0f32 is [0, 0, 128, 63], where the f64
    // form's low bytes are all zero. The local, static, field
    // and boxed positions all route through the projectors'
    // index arm, whose declared type in hand judges the byte.
    // A multi-byte integer's bytes are the same under either
    // form: the guard that the declared-width path leaves the
    // kinds that were already right alone.
    let dir = temp_dir("f32-byte-index-read");
    let src = fixture(&dir, "f32_byte_index_read.rs", F32_BYTE_INDEX_READ_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "local 0 0 128 63\nstat 0 0 128 63\nfield 0 0 32 64\nboxed 0 0 128 63\nint 13 12 11 10\n"
    );
}

const F32_BYTE_INDEX_WRITE_SRC: &str = r#"
fn main() {
    let mut y: f32 = 0.0;
    let ws = unsafe { std::slice::from_raw_parts_mut(&mut y as *mut f32 as *mut u8, 4) };
    ws[3] = 0x3f;
    ws[2] = 0x80;
    println!("local {}", y.to_bits());
    let mut bx = Box::new(0.0f32);
    let bw = unsafe { std::slice::from_raw_parts_mut(&mut *bx as *mut f32 as *mut u8, 4) };
    bw[3] = 0x3f;
    bw[2] = 0x80;
    println!("boxed {}", bx.to_bits());
    let mut z: f32 = 0.0;
    let zw = unsafe { std::slice::from_raw_parts_mut(&mut z as *mut f32 as *mut u8, 4) };
    zw[0] = 0x00;
    zw[1] = 0x00;
    zw[2] = 0x20;
    zw[3] = 0xc0;
    println!("full {}", z);
}
"#;

#[test]
fn run_mode_byte_index_write_of_f32_storage() {
    // Byte writes through a `&mut [u8]` view of an f32's
    // storage splice into the f32's own bits (Issue #475):
    // bytes [0, 0, 128, 63] assemble 1.0f32. Splicing into
    // the widened f64 form instead loses the write entirely
    // when the value is read back at its declared width.
    let dir = temp_dir("f32-byte-index-write");
    let src = fixture(&dir, "f32_byte_index_write.rs", F32_BYTE_INDEX_WRITE_SRC);
    let out = run_driver(&src);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "local 1065353216\nboxed 1065353216\nfull -2.5\n"
    );
}
