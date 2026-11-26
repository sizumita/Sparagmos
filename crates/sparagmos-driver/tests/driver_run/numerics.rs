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

const OVERFLOW_AUX_YES_SRC: &str = r#"
//@ compile-flags: -C overflow-checks=yes
#![crate_type = "lib"]
use std::ops::Add;
pub fn add_one(x: u8) -> u8 { u8::add(x, 1) }
pub fn inc(x: u8) -> u8 { x + 1 }
"#;

const OVERFLOW_AUX_NO_SRC: &str = r#"
//@ compile-flags: -C overflow-checks=no
#![crate_type = "lib"]
use std::ops::Add;
pub fn add_one(x: u8) -> u8 { u8::add(x, 1) }
pub fn inc(x: u8) -> u8 { x + 1 }
"#;

const OVERFLOW_PER_CRATE_MAIN_SRC: &str = r#"
extern crate ocyes;
extern crate ocno;
fn main() {
    // The checked core operation (`u8::add`, an
    // `#[rustc_inherit_overflow_checks]` body) follows the
    // crate that instantiates it, not the main session.
    assert_eq!(ocno::add_one(255), 0);
    assert!(std::panic::catch_unwind(|| ocyes::add_one(255)).is_err());
    // The auxiliary bodies' own checked operations follow
    // their own crate's setting the same way.
    assert_eq!(ocno::inc(255), 0);
    assert!(std::panic::catch_unwind(|| ocyes::inc(255)).is_err());
    println!("ok");
}
"#;

#[test]
fn run_mode_overflow_checks_follow_the_instantiating_crate() {
    // Overflow checks are a per-crate codegen setting: two
    // auxiliary crates built with opposite `-C
    // overflow-checks` step the same checked operations, and
    // each follows its own crate's setting -- under both
    // session configurations of the main compilation. (The
    // suite shape that needs this is
    // tests/ui/iterators/rangefrom-overflow-2crates.rs, whose
    // two auxiliaries step one `RangeFromIter` in turn.)
    let dir = temp_dir("overflow-per-crate");
    let yes = fixture(&dir, "ocyes.rs", OVERFLOW_AUX_YES_SRC);
    let no = fixture(&dir, "ocno.rs", OVERFLOW_AUX_NO_SRC);
    let src = fixture(&dir, "main.rs", OVERFLOW_PER_CRATE_MAIN_SRC);
    let yes_arg = format!("ocyes={}", yes.display());
    let no_arg = format!("ocno={}", no.display());
    let out = run_driver_extra(&src, &["--aux", &yes_arg, "--aux", &no_arg]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr={stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
    let out = run_driver_extra(
        &src,
        &[
            "--aux",
            &yes_arg,
            "--aux",
            &no_arg,
            "--rustc-arg",
            "-C",
            "--rustc-arg",
            "overflow-checks=off",
        ],
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

const PUNNED_SCALAR_VIEW_FMT_SRC: &str = r#"
fn main() {
    let x: f32 = 1.0;
    let view = unsafe { std::slice::from_raw_parts(&x as *const f32 as *const u32, 1) };
    println!("r={:?}", &view[0]);
    let y: u32 = 0x3F80_0000;
    let view2 = unsafe { std::slice::from_raw_parts(&y as *const u32 as *const f32, 1) };
    println!("r2={:?}", &view2[0]);
    let v: u32 = view[0];
    println!("v={}", v);
    let v2: f32 = view2[0];
    println!("v2={}", v2);
}
"#;

#[test]
fn run_mode_punned_scalar_view_fmt_reads_the_element_kind() {
    // A scalar-backed slice whose element type is the
    // storage's width but a different kind (Issue #479):
    // a consumer that resolves the element reference
    // directly (formatting) sees the element under its own
    // kind -- the storage's bits reinterpreted -- not the
    // backing scalar. The typed reads below agree with the
    // formatted ones. Native rustc 1.100.0's output.
    run_ok(
        "punned-scalar-view-fmt",
        PUNNED_SCALAR_VIEW_FMT_SRC,
        "r=1065353216\nr2=1.0\nv=1065353216\nv2=1\n",
    );
}

const PUNNED_SCALAR_VIEW_ITER_SRC: &str = r#"
fn main() {
    let y: u32 = 0x3F80_0000;
    let view = unsafe { std::slice::from_raw_parts(&y as *const u32 as *const f32, 1) };
    for v in view.iter() {
        println!("v={:?}", v);
    }
    let x: f32 = 1.0;
    let view2 = unsafe { std::slice::from_raw_parts(&x as *const f32 as *const u32, 1) };
    for v in view2.iter() {
        println!("w={:?}", v);
    }
    let s: Vec<f32> = view.iter().copied().collect();
    println!("s={:?}", s);
}
"#;

#[test]
fn run_mode_punned_scalar_view_iter_yields_the_element_kind() {
    // The iterator snapshot of a punned scalar view
    // (Issue #479): the yielded element is the storage's
    // bytes under the element's kind, both for a consumer
    // that resolves the yielded reference directly and for
    // a typed consumer (`copied`). Native rustc 1.100.0's
    // output.
    run_ok(
        "punned-scalar-view-iter",
        PUNNED_SCALAR_VIEW_ITER_SRC,
        "v=1.0\nw=1065353216\ns=[1.0]\n",
    );
}

const PUNNED_SCALAR_VIEW_KINDS_SRC: &str = r#"
static SX: f32 = -2.5;
fn main() {
    let a: f32 = -2.5;
    let va = unsafe { std::slice::from_raw_parts(&a as *const f32 as *const i32, 1) };
    println!("a={:?} ai={}", &va[0], va.iter().next().unwrap());
    let b: f64 = 0.5;
    let vb = unsafe { std::slice::from_raw_parts(&b as *const f64 as *const u64, 1) };
    println!("b={:?}", &vb[0]);
    let d: u16 = 0xFFFF;
    let vd = unsafe { std::slice::from_raw_parts(&d as *const u16 as *const i16, 1) };
    println!("d={:?} di={}", &vd[0], vd.iter().next().unwrap());
    let e: u32 = 0x41;
    let ve = unsafe { std::slice::from_raw_parts(&e as *const u32 as *const char, 1) };
    println!("e={:?} ei={:?}", &ve[0], ve.iter().next().unwrap());
    let f: char = '\u{1F600}';
    let vf = unsafe { std::slice::from_raw_parts(&f as *const char as *const u32, 1) };
    println!("f={:?}", &vf[0]);
    let i: u8 = 1;
    let vi = unsafe { std::slice::from_raw_parts(&i as *const u8 as *const bool, 1) };
    println!("i={:?} ii={}", &vi[0], vi.iter().next().unwrap());
    let vs = unsafe { std::slice::from_raw_parts(&SX as *const f32 as *const u32, 1) };
    println!("s={:?} si={}", &vs[0], vs.iter().next().unwrap());
    let bx = Box::new(1.0f32);
    let vb2 = unsafe { std::slice::from_raw_parts(&*bx as *const f32 as *const u32, 1) };
    println!("bx={:?} bi={}", &vb2[0], vb2.iter().next().unwrap());
    let j: f32 = 3.25;
    let vj = std::slice::from_ref(&j);
    println!("j={:?} ji={}", &vj[0], vj.iter().next().unwrap());
}
"#;

#[test]
fn run_mode_punned_scalar_view_other_kinds_and_storages() {
    // The punned reading across scalar kind pairs
    // (signedness, float/integer, char, bool) and storage
    // positions (local, static, boxed), with a same-kind
    // `from_ref` view as the no-reinterpretation guard.
    // Native rustc 1.100.0's output.
    run_ok(
        "punned-scalar-view-kinds",
        PUNNED_SCALAR_VIEW_KINDS_SRC,
        "a=-1071644672 ai=-1071644672\nb=4602678819172646912\nd=-1 di=-1\ne='A' ei='A'\nf=128512\ni=true ii=true\ns=3223322624 si=3223322624\nbx=1065353216 bi=1065353216\nj=3.25 ji=3.25\n",
    );
}

const CHAR_REFERENCE_FMT_SRC: &str = r#"
fn main() {
    let c: char = 'A';
    println!("c={:?}", &c);
    let arr: [char; 1] = ['A'];
    println!("ac={:?}", &arr[0]);
    for v in arr.iter() {
        println!("ai={:?}", v);
    }
    println!("d={}", &c);
}
"#;

#[test]
fn run_mode_char_reference_formats_as_char() {
    // A `char` reached through a reference (an explicit
    // `&c`, an element reference, a yielded reference)
    // formats as the character, not its code point: the
    // argument's tag names the reference, and the char
    // reading judges the pointee tag. Native rustc
    // 1.100.0's output.
    run_ok(
        "char-reference-fmt",
        CHAR_REFERENCE_FMT_SRC,
        "c='A'\nac='A'\nai='A'\nd=A\n",
    );
}

const NONZERO_FMT_SRC: &str = r#"
use std::num::NonZero;
fn main() {
    let a = NonZero::new(42u32).unwrap();
    println!("dbg {:?}", a);
    println!("disp {}", a);
    println!("hex {:x}", a);
    let b = NonZero::new(-3i64).unwrap();
    println!("neg {:?}", b);
    let c = NonZero::new(7usize).unwrap();
    println!("usz {:?} list {:?}", c, vec![a]);
    println!("pad {:>6}|{:<6}|", a, a);
}
"#;

#[test]
fn run_mode_nonzero_formats_as_inner_scalar() {
    // `core::num::NonZero<T>`'s fmt impls all delegate to
    // the primitive's own impl over the stored scalar; the
    // value arrives as nested one-field wrappers (the
    // `NonZeroInner` associated type), which the formatter
    // unwraps to the scalar. The outputs below are native
    // rustc 1.100.0's.
    run_ok(
        "nonzero-fmt",
        NONZERO_FMT_SRC,
        "dbg 42\ndisp 42\nhex 2a\nneg -3\nusz 7 list [42]\npad     42|42    |\n",
    );
}
