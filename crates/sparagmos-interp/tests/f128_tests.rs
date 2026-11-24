//! Exact IEEE 754 binary128 (`f128`) arithmetic: the software
//! module in `sparagmos_interp::f128` computes on raw bit patterns
//! with round-to-nearest-even at every step and never routes a
//! value through `f64`. Every expected bit pattern below was
//! computed independently (exact rational arithmetic rounded to
//! 113 significand bits), not produced by the implementation.

use sparagmos_interp::f128;
use std::cmp::Ordering;

const ZERO: u128 = 0;
const NEG_ZERO: u128 = 1 << 127;
const ONE: u128 = 0x3fff_0000_0000_0000_0000_0000_0000_0000;
const NEG_ONE: u128 = 0xbfff_0000_0000_0000_0000_0000_0000_0000;
const NEG_TWO: u128 = 0xc000_0000_0000_0000_0000_0000_0000_0000;
const TWO: u128 = 0x4000_0000_0000_0000_0000_0000_0000_0000;
const THREE: u128 = 0x4000_8000_0000_0000_0000_0000_0000_0000;
const FOUR: u128 = 0x4001_0000_0000_0000_0000_0000_0000_0000;
const HALF: u128 = 0x3ffe_0000_0000_0000_0000_0000_0000_0000;
const ONE_HALF: u128 = 0x3fff_8000_0000_0000_0000_0000_0000_0000;
const NEG_ONE_HALF: u128 = 0xbfff_8000_0000_0000_0000_0000_0000_0000;
const TWO_HALF: u128 = 0x4000_4000_0000_0000_0000_0000_0000_0000;
const THREE_HALF: u128 = 0x4000_c000_0000_0000_0000_0000_0000_0000;
const FIVE_HALF: u128 = 0x4001_6000_0000_0000_0000_0000_0000_0000;
const POINT_ONE: u128 = 0x3ffb_9999_9999_9999_9999_9999_9999_999a;
const POINT_TWO: u128 = 0x3ffc_9999_9999_9999_9999_9999_9999_999a;
const INF: u128 = 0x7fff_0000_0000_0000_0000_0000_0000_0000;
const NAN: u128 = f128::CANONICAL_NAN;
const MAX_FINITE: u128 = 0x7ffe_ffff_ffff_ffff_ffff_ffff_ffff_ffff;
const MIN_SUBNORMAL: u128 = 1;

#[test]
fn classification() {
    assert!(f128::is_nan(NAN));
    assert!(!f128::is_nan(ONE));
    assert!(f128::is_infinite(INF));
    assert!(f128::is_infinite(f128::neg(INF)));
    assert!(f128::is_finite(MAX_FINITE));
    assert!(!f128::is_finite(INF));
    assert!(f128::is_zero(ZERO) && f128::is_zero(NEG_ZERO));
    assert!(f128::is_sign_negative(NEG_ZERO));
    assert!(f128::is_sign_positive(ZERO));
    assert!(f128::is_normal(ONE));
    assert!(f128::is_subnormal(MIN_SUBNORMAL));
    assert!(!f128::is_subnormal(ONE));
}

#[test]
fn add_sub_exact() {
    assert_eq!(f128::add(ONE_HALF, ONE_HALF), THREE);
    // The sum of the two rounded operands (not the rounding of
    // the real number 0.3, which would end in ...3333).
    assert_eq!(
        f128::add(POINT_ONE, POINT_TWO),
        0x3ffd_3333_3333_3333_3333_3333_3333_3334
    );
    assert_eq!(f128::sub(THREE, ONE_HALF), ONE_HALF);
    assert_eq!(f128::sub(ONE, ONE), ZERO);
    // Opposite signs cancel to +0 under round-to-nearest.
    assert_eq!(f128::add(ONE, NEG_ONE), ZERO);
    assert_eq!(f128::add(NEG_ONE, NEG_ONE), NEG_TWO);
    // Subnormal arithmetic is exact.
    assert_eq!(f128::add(MIN_SUBNORMAL, MIN_SUBNORMAL), 2);
    // Overflow yields infinity.
    assert_eq!(f128::add(MAX_FINITE, MAX_FINITE), INF);
    // Infinities and NaNs.
    assert_eq!(f128::add(INF, ONE), INF);
    assert!(f128::is_nan(f128::add(INF, f128::neg(INF))));
    assert!(f128::is_nan(f128::add(NAN, ONE)));
}

#[test]
fn mul_div_exact() {
    assert_eq!(
        f128::mul(ONE_HALF, ONE_HALF),
        0x4000_2000_0000_0000_0000_0000_0000_0000
    );
    assert_eq!(f128::mul(NEG_ONE_HALF, TWO), f128::neg(THREE));
    assert_eq!(f128::mul(MIN_SUBNORMAL, TWO), 2);
    // Half of the smallest subnormal is exactly a tie: rounds to
    // even, i.e. to zero.
    assert_eq!(f128::mul(MIN_SUBNORMAL, HALF), ZERO);
    assert_eq!(
        f128::div(ONE, THREE),
        0x3ffd_5555_5555_5555_5555_5555_5555_5555
    );
    assert_eq!(f128::div(THREE, ONE_HALF), TWO);
    assert_eq!(
        f128::div(THREE, HALF),
        0x4001_8000_0000_0000_0000_0000_0000_0000
    );
    assert_eq!(f128::div(ONE, ZERO), INF);
    assert_eq!(f128::div(ONE, NEG_ZERO), f128::neg(INF));
    assert!(f128::is_nan(f128::div(ZERO, ZERO)));
    assert_eq!(f128::div(MIN_SUBNORMAL, TWO), ZERO);
}

#[test]
fn rem_is_exact_fmod() {
    assert_eq!(f128::rem(FIVE_HALF, TWO), ONE_HALF);
    assert_eq!(f128::rem(NEG_ONE_HALF, TWO), NEG_ONE_HALF);
    assert_eq!(f128::rem(ONE_HALF, TWO), ONE_HALF);
    assert_eq!(f128::rem(TWO, TWO), ZERO);
    assert_eq!(f128::rem(f128::neg(TWO), TWO), NEG_ZERO);
    assert_eq!(f128::rem(ONE, INF), ONE);
    assert!(f128::is_nan(f128::rem(INF, ONE)));
    assert!(f128::is_nan(f128::rem(ONE, ZERO)));
}

#[test]
fn sqrt_exact() {
    assert_eq!(
        f128::sqrt(TWO),
        0x3fff_6a09_e667_f3bc_c908_b2fb_1366_ea95
    );
    assert_eq!(f128::sqrt(FOUR), TWO);
    assert_eq!(f128::sqrt(ZERO), ZERO);
    assert_eq!(f128::sqrt(NEG_ZERO), NEG_ZERO);
    assert_eq!(f128::sqrt(INF), INF);
    assert!(f128::is_nan(f128::sqrt(NEG_ONE)));
    assert!(f128::is_nan(f128::sqrt(f128::neg(INF))));
}

#[test]
fn rounding_modes() {
    assert_eq!(f128::floor(NEG_ONE_HALF), NEG_TWO);
    assert_eq!(f128::ceil(NEG_ONE_HALF), NEG_ONE);
    assert_eq!(f128::trunc(NEG_ONE_HALF), NEG_ONE);
    assert_eq!(f128::floor(ONE_HALF), ONE);
    assert_eq!(f128::ceil(ONE_HALF), TWO);
    // Ties to even: 2.5 -> 2, 3.5 -> 4.
    assert_eq!(f128::round_ties_even(TWO_HALF), TWO);
    assert_eq!(f128::round_ties_even(THREE_HALF), FOUR);
    // Rust `round`: halves away from zero.
    assert_eq!(f128::round_half_away(TWO_HALF), THREE);
    assert_eq!(f128::round_half_away(f128::neg(TWO_HALF)), f128::neg(THREE));
    assert_eq!(
        f128::fract(NEG_ONE_HALF),
        0xbffe_0000_0000_0000_0000_0000_0000_0000
    );
    assert_eq!(f128::fract(ONE_HALF), HALF);
    assert_eq!(f128::fract(TWO), ZERO);
    assert_eq!(f128::recip(TWO), HALF);
    assert_eq!(f128::recip(HALF), TWO);
    assert_eq!(f128::signum(NEG_ONE_HALF), NEG_ONE);
    assert_eq!(f128::signum(HALF), ONE);
}

#[test]
fn powi_exact() {
    assert_eq!(
        f128::powi(TWO, 10),
        0x4009_0000_0000_0000_0000_0000_0000_0000
    );
    assert_eq!(
        f128::powi(TWO, -2),
        0x3ffd_0000_0000_0000_0000_0000_0000_0000
    );
    assert_eq!(f128::powi(ONE_HALF, 0), ONE);
    assert_eq!(
        f128::powi(THREE, 2),
        0x4002_2000_0000_0000_0000_0000_0000_0000
    );
}

#[test]
fn comparisons() {
    assert!(f128::eq(NEG_ZERO, ZERO));
    assert!(!f128::eq(NAN, NAN));
    assert!(f128::eq(POINT_ONE, POINT_ONE));
    assert_eq!(f128::partial_cmp(NAN, ONE), None);
    assert_eq!(
        f128::partial_cmp(NEG_ONE_HALF, POINT_ONE),
        Some(Ordering::Less)
    );
    assert_eq!(f128::partial_cmp(INF, MAX_FINITE), Some(Ordering::Greater));
    assert_eq!(f128::partial_cmp(NEG_ZERO, ZERO), Some(Ordering::Equal));
    assert_eq!(
        f128::partial_cmp(f128::neg(INF), NEG_ONE),
        Some(Ordering::Less)
    );
    assert_eq!(f128::min(POINT_ONE, POINT_TWO), POINT_ONE);
    assert_eq!(f128::max(POINT_ONE, POINT_TWO), POINT_TWO);
    assert_eq!(f128::min(NAN, POINT_TWO), POINT_TWO);
    assert_eq!(f128::max(POINT_TWO, NAN), POINT_TWO);
    // Signed zeros: min keeps -0, max keeps +0.
    assert_eq!(f128::min(NEG_ZERO, ZERO), NEG_ZERO);
    assert_eq!(f128::max(NEG_ZERO, ZERO), ZERO);
}

#[test]
fn int_casts_saturate() {
    let big = 0x4062_93e5_939a_08ce_9dbd_4800_0000_0000u128; // 1e30
    assert_eq!(f128::to_int(ONE_HALF, 32), 1);
    assert_eq!(f128::to_int(NEG_ONE_HALF, 32), -1);
    assert_eq!(f128::to_int(big, 32), i128::from(i32::MAX));
    assert_eq!(f128::to_int(f128::neg(big), 32), i128::from(i32::MIN));
    assert_eq!(f128::to_int(INF, 64), i128::from(i64::MAX));
    assert_eq!(f128::to_int(NAN, 64), 0);
    assert_eq!(f128::to_uint(NEG_ONE_HALF, 32), 0);
    assert_eq!(f128::to_uint(big, 64), u128::from(u64::MAX));
    assert_eq!(f128::to_uint(ONE_HALF, 128), 1);
    assert_eq!(f128::from_int(-3), f128::neg(THREE));
    assert_eq!(f128::from_int(0), ZERO);
    assert_eq!(
        f128::from_uint(1024),
        0x4009_0000_0000_0000_0000_0000_0000_0000
    );
    // 2^113 is exactly representable; from_uint must not lose it.
    assert_eq!(
        f128::from_uint(1 << 113),
        0x4070_0000_0000_0000_0000_0000_0000_0000
    );
}

#[test]
fn narrower_and_wider_conversions() {
    // f128 0.1 narrows to the f64 nearest 0.1.
    assert_eq!(f128::to_f64_bits(POINT_ONE), 0x3fb9_9999_9999_999a);
    assert_eq!(f128::to_f32_bits(POINT_ONE), 0x3dcc_cccd);
    assert_eq!(f128::to_f16_bits(ONE), 0x3c00);
    assert_eq!(f128::to_f64_bits(f128::neg(INF)), 0xfff0_0000_0000_0000);
    // Widening an f64 is exact: f64 0.1 becomes its exact value in
    // binary128 (note the zero tail after the f64 significand).
    assert_eq!(
        f128::from_f64(0x3fb9_9999_9999_999a),
        0x3ffb_9999_9999_9999_a000_0000_0000_0000
    );
    assert_eq!(f128::from_f64(0), ZERO);
    assert_eq!(f128::from_f64(0x8000_0000_0000_0000), NEG_ZERO);
    // Widening an f32 agrees with widening through f64 (exact).
    let f32_bits = 0x3dcc_cccd_u32;
    assert_eq!(
        f128::from_f32(f32_bits),
        f128::from_f64((f32::from_bits(f32_bits) as f64).to_bits())
    );
    // Round trip f128 -> f64 -> f128 for a value f64 holds exactly.
    assert_eq!(f128::from_f64(f128::to_f64_bits(ONE_HALF)), ONE_HALF);
}

#[test]
fn f16_helpers() {
    assert_eq!(f128::f64_to_f16_bits(1.0), 0x3c00);
    assert_eq!(f128::f64_to_f16_bits(-2.0), 0xc000);
    assert_eq!(f128::f64_to_f16_bits(65504.0), 0x7bff);
    assert_eq!(f128::f64_to_f16_bits(0.0), 0);
    assert_eq!(f128::f64_to_f16_bits(-0.0), 0x8000);
    assert_eq!(f128::f64_to_f16_bits(f64::INFINITY), 0x7c00);
    assert_eq!(f128::f16_bits_to_f64(0x3c00), 1.0);
    assert_eq!(f128::f16_bits_to_f64(0x0001), f64::powi(2.0, -24));
    assert_eq!(f128::f16_bits_to_f64(0x7bff), 65504.0);
    // Round trip over a sample of patterns at binade boundaries.
    for bits in [0x0000u16, 0x03ff, 0x0400, 0x3555, 0x7bff, 0x8000, 0xbc00] {
        assert_eq!(f128::f64_to_f16_bits(f128::f16_bits_to_f64(bits)), bits);
    }
}

#[test]
fn f16_nan_patterns_round_trip() {
    // Every NaN pattern — signaling and quiet, both signs, payloads
    // at the extraction boundaries — survives the trip through the
    // stored `f64` form bit-exactly: the widening is payload-aligned
    // and the narrowing is its exact inverse, so neither the sign
    // nor the signaling bit is lost.
    for bits in [
        0x7c01u16, 0x7d00, 0x7dff, 0x7e00, 0x7e01, 0x7fff, 0xfc01, 0xfd00, 0xfdff, 0xfe00, 0xfe01,
        0xffff,
    ] {
        assert_eq!(f128::f64_to_f16_bits(f128::f16_bits_to_f64(bits)), bits);
    }
    // The widening itself: sign preserved, payload at the top of
    // the `f64` fraction, quiet bit NOT forced.
    assert_eq!(
        f128::f16_bits_to_f64(0x7c01).to_bits(),
        0x7ff0_0400_0000_0000
    );
    assert_eq!(
        f128::f16_bits_to_f64(0xfc01).to_bits(),
        0xfff0_0400_0000_0000
    );
    assert_eq!(
        f128::f16_bits_to_f64(0x7e01).to_bits(),
        0x7ff8_0400_0000_0000
    );
}

#[test]
fn f32_nan_patterns_round_trip() {
    // The `f32` storage pair follows the same rule: a NaN keeps
    // its sign, payload and signaling bit in both directions.
    for bits in [
        0x7f80_0001u32,
        0x7fbf_ffff,
        0x7fc0_0000,
        0x7fdf_ffff,
        0xff80_0001,
        0xffc0_0000,
    ] {
        assert_eq!(
            f128::f64_to_f32_bits_exact(f128::f32_bits_to_f64_exact(bits)),
            bits
        );
    }
    assert_eq!(
        f128::f32_bits_to_f64_exact(0x7f80_0001).to_bits(),
        0x7ff0_0000_2000_0000
    );
}
