use super::run_ok;

// The `hint::black_box` instances the census records were stuck
// on: collection skips the instance when the argument type is one
// a converted body cannot represent in a local (an `f16` value, a
// reference to an extern type), so the call site needs the
// builtin registration of the `hint` spelling itself, not the
// collected body that serves the representable instances.

const BB_F16_SRC: &str = r#"
#![feature(f16)]
fn main() {
    println!("{}", std::hint::black_box(1.5f16));
    println!("{}", std::hint::black_box(-0.25f16));
}
"#;

#[test]
fn run_mode_black_box_f16() {
    run_ok("black-box-f16", BB_F16_SRC, "1.5\n-0.25\n");
}

const BB_EXTERN_REF_SRC: &str = r#"
#![feature(extern_types)]

extern "C" {
    type Opaque;
}

struct Newtype(Opaque);

struct S {
    i: i32,
    j: i32,
    a: Newtype,
}

fn main() {
    let buf = [0i32; 4];
    let x: &Newtype = unsafe { &*(&buf as *const _ as *const Newtype) };
    // Projecting to the newtype works, because it is always at offset 0.
    let field = &x.0;
    std::hint::black_box(field);
    let x: &S = unsafe { &*(&buf as *const _ as *const S) };
    let field = &x.i;
    std::hint::black_box(field);
    let field = &x.j;
    std::hint::black_box(field);
    println!("ok");
}
"#;

#[test]
fn run_mode_black_box_extern_type_reference() {
    run_ok("black-box-extern-ref", BB_EXTERN_REF_SRC, "ok\n");
}

const BB_BITS_SRC: &str = r#"
use std::hint::black_box;
fn main() {
    // The runtime side of the float bit-conversion checks: the
    // values pass through `black_box` so they are computed at
    // run time, over the scalar, boolean and 128-bit shapes.
    assert_eq!(black_box((1f32).to_bits()), 0x3f800000);
    assert_eq!(black_box((1337f64).to_bits()), 0x4094e40000000000);
    assert_eq!(black_box(7u128), 7u128);
    assert!(black_box(true));
    println!("ok");
}
"#;

#[test]
fn run_mode_black_box_bit_conversion_shapes() {
    run_ok("black-box-bits", BB_BITS_SRC, "ok\n");
}

const BB_AS_VALUE_SRC: &str = r#"
fn main() {
    // `black_box` demanded as a function item and invoked
    // through the value: no body is collected for a
    // builtin-owned definition, so the run-time name fallback
    // serves the call with the identity model.
    let f: fn(u32) -> u32 = std::hint::black_box;
    println!("{}", f(41) + 1);
}
"#;

#[test]
fn run_mode_black_box_as_function_value() {
    run_ok("black-box-as-value", BB_AS_VALUE_SRC, "42\n");
}

const CLASSIFY_SRC: &str = r#"
use std::hint::black_box;
fn main() {
    // The classification suite's shape: runtime values forced
    // through `black_box`, classified across the float widths
    // (the `f16` arm is the one whose `black_box` instance has
    // no converted body).
    println!("{:?}", (black_box(0.0f64) / black_box(0.0f64)).classify());
    println!("{:?}", black_box(-1.5f64).classify());
    println!("{}", black_box(2.5f32).is_normal());
    println!("{:?}", black_box(-0.0f32).classify());
}
"#;

#[test]
fn run_mode_float_classify_through_black_box() {
    run_ok(
        "float-classify-black-box",
        CLASSIFY_SRC,
        "Nan\nNormal\ntrue\nZero\n",
    );
}

// `std::intrinsics::fabs` is a generic const fn over the
// `FloatPrimitive` widths. Its instances over the representable
// widths convert from the trait-method body, but the `f16`
// instance is skipped at collection (a converted body cannot
// represent the value in a local), so its call sites need the
// builtin registration of the intrinsic itself -- the same
// no-body registration path as `black_box` above.

const FABS_SRC: &str = r#"
#![feature(core_intrinsics)]
fn main() {
    let x = std::hint::black_box(-1.5f64);
    println!("{}", unsafe { std::intrinsics::fabs(x) });
    let y = std::hint::black_box(-0.5f32);
    println!("{}", unsafe { std::intrinsics::fabs(y) });
}
"#;

#[test]
fn run_mode_fabs_intrinsic() {
    run_ok("fabs-intrinsic", FABS_SRC, "1.5\n0.5\n");
}

const FABS_F16_SRC: &str = r#"
#![feature(core_intrinsics, f16)]
fn main() {
    let x = std::hint::black_box(-1.5f16);
    println!("{}", unsafe { std::intrinsics::fabs(x) });
}
"#;

#[test]
fn run_mode_fabs_intrinsic_f16() {
    run_ok("fabs-intrinsic-f16", FABS_F16_SRC, "1.5\n");
}

const FABS_F128_SRC: &str = r#"
#![feature(core_intrinsics, f128)]
fn main() {
    let x = std::hint::black_box(-1.5f128);
    println!("{:x}", unsafe { std::intrinsics::fabs(x) }.to_bits());
}
"#;

#[test]
fn run_mode_fabs_intrinsic_f128() {
    run_ok(
        "fabs-intrinsic-f128",
        FABS_F128_SRC,
        "3fff8000000000000000000000000000\n",
    );
}
