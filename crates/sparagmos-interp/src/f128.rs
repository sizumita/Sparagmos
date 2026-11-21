//! Exact IEEE 754 binary128 (`f128`) arithmetic on raw bit patterns.
//!
//! Rust 1.100.0 has no native `f128` arithmetic on this host, and the
//! interpreter's `f64`-widened float model cannot represent `f128`
//! values: rounding an `f128` through `f64` would silently lose
//! precision, which this project does not accept as a main path. This
//! module therefore implements binary128 arithmetic directly on the
//! 128-bit encoding, with round-to-nearest-even at every step, exact
//! subnormal handling, and saturating float/integer casts — the same
//! semantics `core` defines for `f128`.
//!
//! NaN *payloads* produced by arithmetic are not part of Rust's
//! guaranteed semantics (a NaN operand's payload is propagated
//! quieted here, and an invalid operation with no NaN operand yields
//! the canonical quiet NaN); every non-NaN result is exact.

use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Bit layout
// ---------------------------------------------------------------------------

const SIGN: u128 = 1 << 127;
const FRAC_MASK: u128 = (1 << 112) - 1;
const QUIET_BIT: u128 = 1 << 111;
const EXP_MASK: u32 = 0x7fff;
const BIAS: i32 = 16383;
/// The canonical quiet NaN (positive, payload = quiet bit only).
pub const CANONICAL_NAN: u128 = 0x7fff_8000_0000_0000_0000_0000_0000_0000;
const INFINITY: u128 = 0x7fff_0000_0000_0000_0000_0000_0000_0000;
pub const ONE: u128 = 0x3fff_0000_0000_0000_0000_0000_0000_0000;

fn sign_of(bits: u128) -> bool {
    bits & SIGN != 0
}

fn exp_of(bits: u128) -> u32 {
    ((bits >> 112) as u32) & EXP_MASK
}

fn frac_of(bits: u128) -> u128 {
    bits & FRAC_MASK
}

pub fn is_nan(bits: u128) -> bool {
    exp_of(bits) == EXP_MASK && frac_of(bits) != 0
}

pub fn is_infinite(bits: u128) -> bool {
    exp_of(bits) == EXP_MASK && frac_of(bits) == 0
}

pub fn is_finite(bits: u128) -> bool {
    exp_of(bits) != EXP_MASK
}

pub fn is_zero(bits: u128) -> bool {
    bits & !SIGN == 0
}

pub fn is_sign_positive(bits: u128) -> bool {
    !sign_of(bits)
}

pub fn is_sign_negative(bits: u128) -> bool {
    sign_of(bits)
}

pub fn is_normal(bits: u128) -> bool {
    let e = exp_of(bits);
    e != 0 && e != EXP_MASK
}

pub fn is_subnormal(bits: u128) -> bool {
    exp_of(bits) == 0 && frac_of(bits) != 0
}

/// Quiet a NaN bit pattern in place (sets the quiet bit, keeps the
/// sign and payload). Non-NaN patterns are returned unchanged.
fn quieted(bits: u128) -> u128 {
    if is_nan(bits) {
        bits | QUIET_BIT
    } else {
        bits
    }
}

pub fn neg(bits: u128) -> u128 {
    bits ^ SIGN
}

pub fn abs(bits: u128) -> u128 {
    bits & !SIGN
}

// ---------------------------------------------------------------------------
// A small arbitrary-precision unsigned integer (little-endian u32 limbs)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
struct Big {
    limbs: Vec<u32>,
}

impl Big {
    fn zero() -> Big {
        Big { limbs: Vec::new() }
    }

    fn from_u128(mut v: u128) -> Big {
        let mut limbs = Vec::new();
        while v != 0 {
            limbs.push(v as u32);
            v >>= 32;
        }
        Big { limbs }
    }

    fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    fn is_odd(&self) -> bool {
        self.limbs.first().is_some_and(|l| l & 1 == 1)
    }

    fn bit_len(&self) -> usize {
        match self.limbs.last() {
            None => 0,
            Some(&top) => (self.limbs.len() - 1) * 32 + (32 - top.leading_zeros() as usize),
        }
    }

    fn bit(&self, i: usize) -> bool {
        let limb = i / 32;
        limb < self.limbs.len() && (self.limbs[limb] >> (i % 32)) & 1 == 1
    }

    fn one_shl(n: usize) -> Big {
        let mut limbs = vec![0u32; n / 32];
        limbs.push(1u32 << (n % 32));
        Big { limbs }
    }

    fn shl(&self, n: usize) -> Big {
        if self.is_zero() {
            return Big::zero();
        }
        let limb_shift = n / 32;
        let bit_shift = n % 32;
        let mut limbs = vec![0u32; limb_shift];
        let mut carry = 0u32;
        for &l in &self.limbs {
            limbs.push((l << bit_shift) | carry);
            carry = if bit_shift == 0 {
                0
            } else {
                l >> (32 - bit_shift)
            };
        }
        if carry != 0 {
            limbs.push(carry);
        }
        let mut b = Big { limbs };
        b.normalize();
        b
    }

    fn shr(&self, n: usize) -> Big {
        let limb_shift = n / 32;
        if limb_shift >= self.limbs.len() {
            return Big::zero();
        }
        let bit_shift = n % 32;
        let mut limbs = Vec::with_capacity(self.limbs.len() - limb_shift);
        for i in limb_shift..self.limbs.len() {
            let low = self.limbs[i] >> bit_shift;
            let high = if bit_shift == 0 || i + 1 >= self.limbs.len() {
                0
            } else {
                self.limbs[i + 1] << (32 - bit_shift)
            };
            limbs.push(low | high);
        }
        let mut b = Big { limbs };
        b.normalize();
        b
    }

    /// The low `n` bits (`self mod 2^n`).
    fn low_bits(&self, n: usize) -> Big {
        if n == 0 {
            return Big::zero();
        }
        let need = n.div_ceil(32);
        if self.limbs.len() < need {
            return self.clone();
        }
        let mut limbs = self.limbs[..need].to_vec();
        let rem = n % 32;
        if rem != 0 {
            if let Some(top) = limbs.last_mut() {
                *top &= (1u32 << rem) - 1;
            }
        }
        let mut b = Big { limbs };
        b.normalize();
        b
    }

    fn normalize(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    fn cmp_mag(&self, other: &Big) -> Ordering {
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len().cmp(&other.limbs.len());
        }
        for i in (0..self.limbs.len()).rev() {
            if self.limbs[i] != other.limbs[i] {
                return self.limbs[i].cmp(&other.limbs[i]);
            }
        }
        Ordering::Equal
    }

    fn add(&self, other: &Big) -> Big {
        let n = self.limbs.len().max(other.limbs.len());
        let mut limbs = Vec::with_capacity(n + 1);
        let mut carry = 0u64;
        for i in 0..n {
            let a = u64::from(*self.limbs.get(i).unwrap_or(&0));
            let b = u64::from(*other.limbs.get(i).unwrap_or(&0));
            let s = a + b + carry;
            limbs.push(s as u32);
            carry = s >> 32;
        }
        if carry != 0 {
            limbs.push(carry as u32);
        }
        let mut r = Big { limbs };
        r.normalize();
        r
    }

    /// `self - other`; requires `self >= other`.
    fn sub(&self, other: &Big) -> Big {
        let mut limbs = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0i64;
        for i in 0..self.limbs.len() {
            let a = i64::from(self.limbs[i]);
            let b = i64::from(*other.limbs.get(i).unwrap_or(&0));
            let mut d = a - b - borrow;
            if d < 0 {
                d += 1 << 32;
                borrow = 1;
            } else {
                borrow = 0;
            }
            limbs.push(d as u32);
        }
        let mut r = Big { limbs };
        r.normalize();
        r
    }

    fn mul(&self, other: &Big) -> Big {
        if self.is_zero() || other.is_zero() {
            return Big::zero();
        }
        let mut limbs = vec![0u32; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &b) in other.limbs.iter().enumerate() {
                let idx = i + j;
                let t = u64::from(limbs[idx]) + u64::from(a) * u64::from(b) + carry;
                limbs[idx] = t as u32;
                carry = t >> 32;
            }
            let mut idx = i + other.limbs.len();
            while carry != 0 {
                let t = u64::from(limbs[idx]) + carry;
                limbs[idx] = t as u32;
                carry = t >> 32;
                idx += 1;
            }
        }
        let mut r = Big { limbs };
        r.normalize();
        r
    }

    fn inc(&self) -> Big {
        self.add(&Big::from_u128(1))
    }

    /// `self mod divisor` by streaming bits from the top: the running
    /// remainder stays below `2 * divisor`, so this is cheap even when
    /// `self` is thousands of bits wide (the `fmod` case).
    fn rem(&self, divisor: &Big) -> Big {
        let mut r = Big::zero();
        for i in (0..self.bit_len()).rev() {
            r = r.shl(1);
            if self.bit(i) {
                r = r.inc();
            }
            if r.cmp_mag(divisor) != Ordering::Less {
                r = r.sub(divisor);
            }
        }
        r
    }
}

// ---------------------------------------------------------------------------
// Decoding and the generic rounding packer
// ---------------------------------------------------------------------------

enum Parts {
    Zero,
    Infinite,
    Nan,
    /// value = mant * 2^exp2, mant > 0
    Finite { mant: Big, exp2: i32 },
}

fn decode(bits: u128) -> Parts {
    let exp = exp_of(bits);
    let frac = frac_of(bits);
    if exp == EXP_MASK {
        if frac == 0 {
            Parts::Infinite
        } else {
            Parts::Nan
        }
    } else if exp == 0 {
        if frac == 0 {
            Parts::Zero
        } else {
            Parts::Finite {
                mant: Big::from_u128(frac),
                exp2: 1 - BIAS - 112,
            }
        }
    } else {
        Parts::Finite {
            mant: Big::from_u128(frac | (1 << 112)),
            exp2: exp as i32 - BIAS - 112,
        }
    }
}

/// Round `mant >> shift` to nearest, ties to even, where the true
/// value carries an additional positive tail smaller than one unit of
/// `mant`'s least significant bit iff `sticky` (the tail breaks a tie
/// upward, and never crosses a non-tie boundary).
fn round_shift(mant: &Big, shift: usize, sticky: bool) -> Big {
    if shift == 0 {
        return mant.clone();
    }
    let kept = mant.shr(shift);
    let dropped = mant.low_bits(shift);
    let half = Big::one_shl(shift - 1);
    let inc = match dropped.cmp_mag(&half) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => sticky || kept.is_odd(),
    };
    if inc {
        kept.inc()
    } else {
        kept
    }
}

/// Pack `value = (mant + tail) * 2^exp2` (with `0 <= tail < 1`, and
/// `tail > 0` iff `sticky`) into a binary float encoding with `prec`
/// significand bits, exponent bias `bias` and `exp_bits` exponent
/// bits. Round-to-nearest-even; overflow yields infinity, underflow
/// yields a subnormal or a signed zero.
fn pack_generic(
    sign: bool,
    mant: &Big,
    exp2: i32,
    sticky: bool,
    prec: usize,
    bias: i32,
    exp_bits: u32,
) -> u128 {
    let sign_bit = if sign {
        1u128 << (exp_bits as usize + (prec - 1))
    } else {
        0
    };
    if mant.is_zero() {
        return sign_bit;
    }
    let frac_bits = prec - 1;
    let emin = 1 - bias; // exponent of the smallest normal
    let emax = bias; // exponent of the largest normal
    let lead = mant.bit_len() as i32 - 1 + exp2; // value in [2^lead, 2^(lead+1))
    let exp_mask = (1u128 << exp_bits) - 1;
    if lead >= emin {
        // Normal range: keep the top `prec` bits.
        let shift = mant.bit_len() as i32 - prec as i32;
        let (mut kept, mut e2) = if shift > 0 {
            (round_shift(mant, shift as usize, sticky), exp2 + shift)
        } else {
            (mant.shl((-shift) as usize), exp2 + shift)
        };
        if kept.bit_len() > prec {
            // Rounding carried out of the top bit: the value is an
            // exact power of two at the next exponent.
            kept = kept.shr(1);
            e2 += 1;
        }
        let field = e2 + frac_bits as i32 + bias;
        if field > emax + bias {
            return sign_bit | (exp_mask << frac_bits);
        }
        let kept_u = kept.to_u128_lossless();
        let frac = kept_u & ((1u128 << frac_bits) - 1);
        sign_bit | ((field as u128) << frac_bits) | frac
    } else {
        // Subnormal range: the resolution is 2^(emin - frac_bits).
        let shift = (emin - frac_bits as i32) - exp2;
        let kept = if shift > 0 {
            round_shift(mant, shift as usize, sticky)
        } else {
            mant.shl((-shift) as usize)
        };
        // The encoding is sign | kept directly: kept == 2^frac_bits
        // is exactly the smallest normal's encoding.
        sign_bit | kept.to_u128_lossless()
    }
}

impl Big {
    fn to_u128_lossless(&self) -> u128 {
        debug_assert!(self.bit_len() <= 128);
        let mut v = 0u128;
        for (i, &l) in self.limbs.iter().enumerate() {
            v |= (l as u128) << (32 * i);
        }
        v
    }
}

/// Pack into the f128 encoding.
fn pack(sign: bool, mant: &Big, exp2: i32, sticky: bool) -> u128 {
    pack_generic(sign, mant, exp2, sticky, 113, BIAS, 15)
}

fn with_sign(bits: u128, sign: bool) -> u128 {
    (bits & !SIGN) | if sign { SIGN } else { 0 }
}

fn infinity(sign: bool) -> u128 {
    if sign {
        INFINITY | SIGN
    } else {
        INFINITY
    }
}

fn zero(sign: bool) -> u128 {
    if sign {
        SIGN
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

fn addsub(a: u128, b: u128, negate_b: bool) -> u128 {
    let sa = sign_of(a);
    let sb = sign_of(b) ^ negate_b;
    if is_nan(a) {
        return quieted(a);
    }
    if is_nan(b) {
        return quieted(b) ^ if negate_b { SIGN } else { 0 };
    }
    match (decode(a), decode(b)) {
        (Parts::Infinite, Parts::Infinite) => {
            if sa != sb {
                CANONICAL_NAN
            } else {
                infinity(sa)
            }
        }
        (Parts::Infinite, _) => infinity(sa),
        (_, Parts::Infinite) => infinity(sb),
        (Parts::Zero, Parts::Zero) => {
            // Equal signs keep the sign; opposite signs give +0
            // (round-to-nearest).
            if sa == sb {
                zero(sa)
            } else {
                0
            }
        }
        (Parts::Zero, _) => with_sign(b, sb),
        (_, Parts::Zero) => a,
        (
            Parts::Finite { mant: ma, exp2: ea },
            Parts::Finite { mant: mb, exp2: eb },
        ) => {
            let e = ea.min(eb);
            let ma = ma.shl((ea - e) as usize);
            let mb = mb.shl((eb - e) as usize);
            if sa == sb {
                pack(sa, &ma.add(&mb), e, false)
            } else {
                match ma.cmp_mag(&mb) {
                    Ordering::Equal => 0, // x - x = +0 (round-to-nearest)
                    Ordering::Greater => pack(sa, &ma.sub(&mb), e, false),
                    Ordering::Less => pack(sb, &mb.sub(&ma), e, false),
                }
            }
        }
        _ => unreachable!("NaN handled above"),
    }
}

pub fn add(a: u128, b: u128) -> u128 {
    addsub(a, b, false)
}

pub fn sub(a: u128, b: u128) -> u128 {
    addsub(a, b, true)
}

pub fn mul(a: u128, b: u128) -> u128 {
    let sa = sign_of(a);
    let sb = sign_of(b);
    if is_nan(a) {
        return quieted(a);
    }
    if is_nan(b) {
        return quieted(b);
    }
    match (decode(a), decode(b)) {
        (Parts::Infinite, Parts::Zero) | (Parts::Zero, Parts::Infinite) => CANONICAL_NAN,
        (Parts::Infinite, _) | (_, Parts::Infinite) => infinity(sa ^ sb),
        (Parts::Zero, _) | (_, Parts::Zero) => zero(sa ^ sb),
        (
            Parts::Finite { mant: ma, exp2: ea },
            Parts::Finite { mant: mb, exp2: eb },
        ) => pack(sa ^ sb, &ma.mul(&mb), ea + eb, false),
        _ => unreachable!("NaN handled above"),
    }
}

/// Floor division of `num / den` computed to `scale` fractional bits:
/// returns `(q, r)` with `q = floor(num * 2^scale / den)` and
/// `r = num * 2^scale - q * den` (so `r != 0` is the sticky tail).
/// Both `num` and `den` are at most 113 bits; the streaming loop
/// keeps the remainder below `2 * den` in a `u128`.
fn divmod_scaled(num: &Big, den: &Big, scale: usize) -> (Big, bool) {
    let den_u = den.to_u128_lossless();
    debug_assert!(den_u != 0);
    let total = num.bit_len() + scale;
    let mut rem: u128 = 0;
    let mut q = Big::zero();
    for i in (0..total).rev() {
        rem <<= 1;
        let bit = if i >= scale {
            num.bit(i - scale)
        } else {
            false
        };
        if bit {
            rem |= 1;
        }
        if rem >= den_u {
            rem -= den_u;
            q = q.add(&Big::one_shl(i));
        }
    }
    (q, rem != 0)
}

pub fn div(a: u128, b: u128) -> u128 {
    let sa = sign_of(a);
    let sb = sign_of(b);
    if is_nan(a) {
        return quieted(a);
    }
    if is_nan(b) {
        return quieted(b);
    }
    match (decode(a), decode(b)) {
        (Parts::Infinite, Parts::Infinite) | (Parts::Zero, Parts::Zero) => CANONICAL_NAN,
        (Parts::Infinite, _) => infinity(sa ^ sb),
        (_, Parts::Infinite) => zero(sa ^ sb),
        (Parts::Zero, _) => zero(sa ^ sb),
        (_, Parts::Zero) => infinity(sa ^ sb),
        (
            Parts::Finite { mant: ma, exp2: ea },
            Parts::Finite { mant: mb, exp2: eb },
        ) => {
            // Scale so the quotient carries ~330 significant bits:
            // far more than the 113 the packer keeps, with the
            // remainder as an exact sticky tail.
            let scale = 320usize.saturating_sub(ma.bit_len()) + mb.bit_len();
            let (q, sticky) = divmod_scaled(&ma, &mb, scale);
            pack(sa ^ sb, &q, ea - eb - scale as i32, sticky)
        }
        _ => unreachable!("NaN handled above"),
    }
}

/// Rust's `%` for floats: the exact `fmod` — the remainder of
/// truncated division, with the sign of the dividend. The result is
/// always exactly representable.
pub fn rem(a: u128, b: u128) -> u128 {
    let sa = sign_of(a);
    if is_nan(a) {
        return quieted(a);
    }
    if is_nan(b) {
        return quieted(b);
    }
    match (decode(a), decode(b)) {
        (Parts::Infinite, _) => CANONICAL_NAN,
        (_, Parts::Zero) => CANONICAL_NAN,
        (_, Parts::Infinite) => a,
        (Parts::Zero, _) => a,
        (
            Parts::Finite { mant: ma, exp2: ea },
            Parts::Finite { mant: mb, exp2: eb },
        ) => {
            // |a| < |b| -> the remainder is `a` itself.
            let ord = {
                let la = ma.bit_len() as i32 + ea;
                let lb = mb.bit_len() as i32 + eb;
                if la != lb {
                    la.cmp(&lb)
                } else {
                    let e = ea.min(eb);
                    ma.shl((ea - e) as usize).cmp_mag(&mb.shl((eb - e) as usize))
                }
            };
            if ord == Ordering::Less {
                return a;
            }
            if ord == Ordering::Equal {
                return zero(sa);
            }
            let e = ea.min(eb);
            let num = ma.shl((ea - e) as usize);
            let den = mb.shl((eb - e) as usize);
            let r = num.rem(&den);
            if r.is_zero() {
                zero(sa)
            } else {
                pack(sa, &r, e, false)
            }
        }
        _ => unreachable!("NaN handled above"),
    }
}

/// Integer square root of `n` (floor), by the classic bit-pair
/// method. Returns `(root, is_exact)`.
fn isqrt(n: &Big) -> (Big, bool) {
    if n.is_zero() {
        return (Big::zero(), true);
    }
    let mut rest = n.clone();
    let mut root = Big::zero();
    // Highest power of four not exceeding n.
    let mut bit = Big::one_shl(n.bit_len() - 1 & !1);
    while !bit.is_zero() {
        let trial = root.add(&bit);
        if rest.cmp_mag(&trial) != Ordering::Less {
            rest = rest.sub(&trial);
            root = root.shr(1).add(&bit);
        } else {
            root = root.shr(1);
        }
        bit = bit.shr(2);
    }
    let exact = rest.is_zero();
    (root, exact)
}

pub fn sqrt(a: u128) -> u128 {
    if is_nan(a) {
        return quieted(a);
    }
    if is_zero(a) || is_infinite(a) && !sign_of(a) {
        return a;
    }
    if sign_of(a) {
        // sqrt of a negative (including -inf) is NaN.
        return CANONICAL_NAN;
    }
    match decode(a) {
        Parts::Finite { mant, exp2 } => {
            let (m, e) = if exp2 % 2 != 0 {
                (mant.shl(1), exp2 - 1)
            } else {
                (mant, exp2)
            };
            // Scale by 2^280 so the root carries ~197 bits.
            let n = m.shl(280);
            let (root, exact) = isqrt(&n);
            pack(false, &root, e / 2 - 140, !exact)
        }
        _ => unreachable!("non-finite handled above"),
    }
}

// ---------------------------------------------------------------------------
// Rounding to integral values
// ---------------------------------------------------------------------------

enum IntRound {
    Floor,
    Ceil,
    Trunc,
    TiesEven,
    HalfAway,
}

/// Round the finite decoded value to an integer according to
/// `mode`, returning the f128 encoding of the result.
fn round_to_int(sign: bool, mant: &Big, exp2: i32, mode: IntRound) -> u128 {
    if exp2 >= 0 {
        // Already an integer.
        return pack(sign, mant, exp2, false);
    }
    let shift = (-exp2) as usize;
    let int_part = mant.shr(shift);
    let frac_part = mant.low_bits(shift);
    let has_frac = !frac_part.is_zero();
    let inc = match mode {
        IntRound::Trunc => false,
        IntRound::Floor => sign && has_frac,
        IntRound::Ceil => !sign && has_frac,
        IntRound::TiesEven | IntRound::HalfAway => {
            if !has_frac {
                false
            } else {
                let half = Big::one_shl(shift - 1);
                match frac_part.cmp_mag(&half) {
                    Ordering::Greater => true,
                    Ordering::Less => false,
                    Ordering::Equal => match mode {
                        IntRound::TiesEven => int_part.is_odd(),
                        _ => true, // half away from zero
                    },
                }
            }
        }
    };
    let mag = if inc { int_part.inc() } else { int_part };
    if mag.is_zero() {
        return zero(sign);
    }
    pack(sign, &mag, 0, false)
}

fn unary_round(a: u128, mode: IntRound) -> u128 {
    if is_nan(a) {
        return quieted(a);
    }
    if is_infinite(a) || is_zero(a) {
        return a;
    }
    match decode(a) {
        Parts::Finite { mant, exp2 } => round_to_int(sign_of(a), &mant, exp2, mode),
        _ => unreachable!(),
    }
}

pub fn floor(a: u128) -> u128 {
    unary_round(a, IntRound::Floor)
}

pub fn ceil(a: u128) -> u128 {
    unary_round(a, IntRound::Ceil)
}

pub fn trunc(a: u128) -> u128 {
    unary_round(a, IntRound::Trunc)
}

pub fn round_ties_even(a: u128) -> u128 {
    unary_round(a, IntRound::TiesEven)
}

/// Rust's `round`: halfway cases round away from zero.
pub fn round_half_away(a: u128) -> u128 {
    unary_round(a, IntRound::HalfAway)
}

pub fn fract(a: u128) -> u128 {
    if is_nan(a) {
        return quieted(a);
    }
    if is_infinite(a) {
        return CANONICAL_NAN;
    }
    if is_zero(a) {
        return a;
    }
    match decode(a) {
        Parts::Finite { mant, exp2 } => {
            if exp2 >= 0 {
                return zero(sign_of(a));
            }
            let shift = (-exp2) as usize;
            let frac_part = mant.low_bits(shift);
            if frac_part.is_zero() {
                zero(sign_of(a))
            } else {
                pack(sign_of(a), &frac_part, exp2, false)
            }
        }
        _ => unreachable!(),
    }
}

pub fn recip(a: u128) -> u128 {
    div(ONE, a)
}

pub fn signum(a: u128) -> u128 {
    if is_nan(a) {
        return quieted(a);
    }
    if sign_of(a) {
        neg(ONE)
    } else {
        ONE
    }
}

pub fn min(a: u128, b: u128) -> u128 {
    if is_nan(a) {
        return if is_nan(b) { CANONICAL_NAN } else { b };
    }
    if is_nan(b) {
        return a;
    }
    if is_zero(a) && is_zero(b) {
        return if sign_of(a) || sign_of(b) { neg(0) } else { 0 };
    }
    match partial_cmp(a, b) {
        Some(Ordering::Less) | Some(Ordering::Equal) => a,
        _ => b,
    }
}

pub fn max(a: u128, b: u128) -> u128 {
    if is_nan(a) {
        return if is_nan(b) { CANONICAL_NAN } else { b };
    }
    if is_nan(b) {
        return a;
    }
    if is_zero(a) && is_zero(b) {
        return if sign_of(a) && sign_of(b) { neg(0) } else { 0 };
    }
    match partial_cmp(a, b) {
        Some(Ordering::Greater) | Some(Ordering::Equal) => a,
        _ => b,
    }
}

pub fn powi(a: u128, n: i32) -> u128 {
    if n == 0 {
        return ONE;
    }
    let mut result = ONE;
    let mut base = a;
    let mut e = n.unsigned_abs();
    while e > 0 {
        if e & 1 == 1 {
            result = mul(result, base);
        }
        e >>= 1;
        if e > 0 {
            base = mul(base, base);
        }
    }
    if n < 0 {
        recip(result)
    } else {
        result
    }
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

/// IEEE equality: NaN is unequal to everything, +0 equals -0.
pub fn eq(a: u128, b: u128) -> bool {
    if is_nan(a) || is_nan(b) {
        return false;
    }
    if is_zero(a) && is_zero(b) {
        return true;
    }
    a == b
}

pub fn partial_cmp(a: u128, b: u128) -> Option<Ordering> {
    if is_nan(a) || is_nan(b) {
        return None;
    }
    if is_zero(a) && is_zero(b) {
        return Some(Ordering::Equal);
    }
    let (sa, sb) = (sign_of(a), sign_of(b));
    if sa != sb {
        return Some(if sa { Ordering::Less } else { Ordering::Greater });
    }
    // Same sign: the encoding is magnitude-ordered.
    let mag = (a & !SIGN).cmp(&(b & !SIGN));
    Some(if sa { mag.reverse() } else { mag })
}

// ---------------------------------------------------------------------------
// Casts
// ---------------------------------------------------------------------------

/// Truncating conversion to a signed integer of `bits` bits,
/// saturating at the target range; NaN becomes 0 (Rust `as`
/// semantics).
pub fn to_int(a: u128, bits: u32) -> i128 {
    if is_nan(a) {
        return 0;
    }
    let sign = sign_of(a);
    if is_infinite(a) {
        return if sign {
            if bits >= 128 { i128::MIN } else { -(1i128 << (bits - 1)) }
        } else if bits >= 128 {
            i128::MAX
        } else {
            (1i128 << (bits - 1)) - 1
        };
    }
    match decode(a) {
        Parts::Finite { mant, exp2 } => {
            let mag = if exp2 >= 0 {
                mant.shl(exp2 as usize)
            } else {
                mant.shr((-exp2) as usize)
            };
            // Clamp magnitude to the sign-dependent limit.
            let limit = if sign {
                Big::one_shl((bits - 1) as usize)
            } else {
                Big::one_shl((bits - 1) as usize).sub(&Big::from_u128(1))
            };
            if mag.cmp_mag(&limit) == Ordering::Greater {
                return if sign {
                    if bits >= 128 { i128::MIN } else { -(1i128 << (bits - 1)) }
                } else if bits >= 128 {
                    i128::MAX
                } else {
                    (1i128 << (bits - 1)) - 1
                };
            }
            let v = mag.to_u128_lossless() as i128;
            if sign {
                // wrapping: mag == 2^127 is i128::MIN's magnitude.
                0i128.wrapping_sub(v)
            } else {
                v
            }
        }
        _ => 0,
    }
}

/// Truncating conversion to an unsigned integer of `bits` bits,
/// saturating at the target range; NaN and negative values become 0.
pub fn to_uint(a: u128, bits: u32) -> u128 {
    // NaN becomes 0; every negative value (including -inf and -0)
    // saturates to 0.
    if is_nan(a) || sign_of(a) {
        return 0;
    }
    if is_infinite(a) {
        return if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
    }
    match decode(a) {
        Parts::Finite { mant, exp2 } => {
            let mag = if exp2 >= 0 {
                mant.shl(exp2 as usize)
            } else {
                mant.shr((-exp2) as usize)
            };
            let limit = if bits >= 128 {
                Big::from_u128(u128::MAX)
            } else {
                Big::from_u128((1u128 << bits) - 1)
            };
            if mag.cmp_mag(&limit) == Ordering::Greater {
                return limit.to_u128_lossless();
            }
            mag.to_u128_lossless()
        }
        _ => 0,
    }
}

pub fn from_int(v: i128) -> u128 {
    if v == 0 {
        return 0;
    }
    let sign = v < 0;
    let mag = Big::from_u128(v.unsigned_abs());
    pack(sign, &mag, 0, false)
}

pub fn from_uint(v: u128) -> u128 {
    if v == 0 {
        return 0;
    }
    pack(false, &Big::from_u128(v), 0, false)
}

/// Convert to a narrower IEEE format (`exp_bits`, `frac_bits`),
/// returning the target bit pattern in the low bits of a `u128`.
/// NaN payloads are carried across (top-aligned, quieted), as the
/// hardware conversions do.
fn to_narrower(a: u128, exp_bits: u32, frac_bits: usize, bias: i32) -> u128 {
    let sign = sign_of(a);
    let sign_bit = if sign { 1u128 << (exp_bits as usize + frac_bits) } else { 0 };
    if is_nan(a) {
        let payload = frac_of(a) >> (112 - frac_bits);
        let exp_all = ((1u128 << exp_bits) - 1) << frac_bits;
        let quiet = 1u128 << (frac_bits - 1);
        return sign_bit | exp_all | payload | quiet;
    }
    if is_infinite(a) {
        let exp_all = ((1u128 << exp_bits) - 1) << frac_bits;
        return sign_bit | exp_all;
    }
    if is_zero(a) {
        return sign_bit;
    }
    match decode(a) {
        Parts::Finite { mant, exp2 } => {
            // pack_generic produces the target format's layout
            // directly (sign at the target's top bit).
            pack_generic(sign, &mant, exp2, false, frac_bits + 1, bias, exp_bits)
        }
        _ => unreachable!(),
    }
}

pub fn to_f64_bits(a: u128) -> u64 {
    to_narrower(a, 11, 52, 1023) as u64
}

pub fn to_f32_bits(a: u128) -> u32 {
    to_narrower(a, 8, 24 - 1, 127) as u32
}

pub fn to_f16_bits(a: u128) -> u16 {
    to_narrower(a, 5, 10, 15) as u16
}

/// Exact widening of an `f64` bit pattern to `f128`.
pub fn from_f64(bits: u64) -> u128 {
    let sign = bits >> 63 != 0;
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    if exp == 0x7ff {
        if frac == 0 {
            return infinity(sign);
        }
        // NaN: payload top-aligned, quieted.
        let payload = (frac as u128) << (112 - 52);
        return (if sign { SIGN } else { 0 }) | INFINITY | payload | QUIET_BIT;
    }
    if exp == 0 && frac == 0 {
        return zero(sign);
    }
    let (mant, exp2) = if exp == 0 {
        (Big::from_u128(frac as u128), -1074)
    } else {
        (Big::from_u128((frac | (1 << 52)) as u128), exp - 1023 - 52)
    };
    pack(sign, &mant, exp2, false)
}

/// Exact widening of an `f32` bit pattern to `f128`.
pub fn from_f32(bits: u32) -> u128 {
    let sign = bits >> 31 != 0;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & ((1u32 << 23) - 1);
    if exp == 0xff {
        if frac == 0 {
            return infinity(sign);
        }
        let payload = (frac as u128) << (112 - 23);
        return (if sign { SIGN } else { 0 }) | INFINITY | payload | QUIET_BIT;
    }
    if exp == 0 && frac == 0 {
        return zero(sign);
    }
    let (mant, exp2) = if exp == 0 {
        (Big::from_u128(frac as u128), -149)
    } else {
        (Big::from_u128((frac | (1 << 23)) as u128), exp - 127 - 23)
    };
    pack(sign, &mant, exp2, false)
}

/// The `f16` bit pattern nearest to `v` (round-to-nearest-even),
/// computed by the generic packer.
pub fn f64_to_f16_bits(v: f64) -> u16 {
    let bits = v.to_bits();
    let sign = bits >> 63 != 0;
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    if exp == 0x7ff {
        let sign_bit = if sign { 0x8000u16 } else { 0 };
        return if frac == 0 {
            sign_bit | 0x7c00
        } else {
            sign_bit | 0x7e00 // canonical quiet NaN
        };
    }
    if exp == 0 && frac == 0 {
        return if sign { 0x8000 } else { 0 };
    }
    let (mant, exp2) = if exp == 0 {
        (Big::from_u128(frac as u128), -1074)
    } else {
        (Big::from_u128((frac | (1 << 52)) as u128), exp - 1023 - 52)
    };
    pack_generic(sign, &mant, exp2, false, 11, 15, 5) as u16
}

/// The `f64` value of an `f16` bit pattern (exact).
pub fn f16_bits_to_f64(bits: u16) -> f64 {
    let sign = if bits >> 15 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = f64::from(bits & 0x3ff);
    if exp == 0x1f {
        if bits & 0x3ff == 0 {
            return sign * f64::INFINITY;
        }
        return f64::NAN;
    }
    if exp == 0 {
        // Subnormal: frac * 2^-24.
        return sign * frac * f64::powi(2.0, -24);
    }
    sign * (1.0 + frac / 1024.0) * f64::powi(2.0, exp - 15)
}
