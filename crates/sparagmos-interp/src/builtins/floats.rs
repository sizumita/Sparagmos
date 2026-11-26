//! Float method execution (f16/f32/f64/f128) for the float builtins.



use crate::error::{InterpError, InterpResult};
use crate::value::Value;

use super::*;

/// Execute one primitive `f128` method on raw bit patterns.
pub(super) fn exec_float128_method(
    name: &str,
    op: FloatOp,
    vals: &[Value],
) -> InterpResult<Value> {
    use crate::f128;
    let arity = |n: usize| -> InterpResult<()> {
        if vals.len() == n {
            Ok(())
        } else {
            Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                format!("{n} argument(s)"),
                format!("{} argument(s)", vals.len()),
            ))
        }
    };
    let as_bits = |v: &Value| -> InterpResult<u128> {
        match v {
            Value::Float128(bits) => Ok(*bits),
            other => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "f128",
                other.kind_name(),
            )),
        }
    };
    use FloatOp as O;
    if op == O::FromBits {
        arity(1)?;
        let bits = match vals[0] {
            Value::Uint(v) => v,
            Value::Int(v) => v as u128,
            ref other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "integer",
                    other.kind_name(),
                ));
            }
        };
        return Ok(Value::Float128(bits));
    }
    arity(if matches!(op, O::Powi | O::Min | O::Max) {
        2
    } else {
        1
    })?;
    let x = as_bits(&vals[0])?;
    Ok(match op {
        O::IsNan => Value::Bool(f128::is_nan(x)),
        O::IsInfinite => Value::Bool(f128::is_infinite(x)),
        O::IsFinite => Value::Bool(f128::is_finite(x)),
        O::IsSignPositive => Value::Bool(f128::is_sign_positive(x)),
        O::IsSignNegative => Value::Bool(f128::is_sign_negative(x)),
        O::Abs => Value::Float128(f128::abs(x)),
        O::Sqrt => Value::Float128(f128::sqrt(x)),
        O::Floor => Value::Float128(f128::floor(x)),
        O::Ceil => Value::Float128(f128::ceil(x)),
        O::Round => Value::Float128(f128::round_half_away(x)),
        O::RoundTiesEven => Value::Float128(f128::round_ties_even(x)),
        O::Trunc => Value::Float128(f128::trunc(x)),
        O::Fract => Value::Float128(f128::fract(x)),
        O::Recip => Value::Float128(f128::recip(x)),
        O::Signum => Value::Float128(f128::signum(x)),
        O::ToBits => Value::Uint(x),
        O::Powi => {
            let n = match vals[1] {
                Value::Int(v) => v as i32,
                Value::Uint(v) => v as i32,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "i32",
                        other.kind_name(),
                    ));
                }
            };
            Value::Float128(f128::powi(x, n))
        }
        O::Min => Value::Float128(f128::min(x, as_bits(&vals[1])?)),
        O::Max => Value::Float128(f128::max(x, as_bits(&vals[1])?)),
        O::Exp
        | O::Exp2
        | O::ExpM1
        | O::Ln
        | O::Ln1p
        | O::Log
        | O::Log2
        | O::Log10
        | O::Sin
        | O::Cos
        | O::Tan
        | O::Asin
        | O::Acos
        | O::Atan
        | O::Atan2
        | O::SinCos
        | O::Sinh
        | O::Cosh
        | O::Tanh
        | O::Asinh
        | O::Acosh
        | O::Atanh
        | O::Cbrt
        | O::Hypot
        | O::MulAdd
        | O::DivEuclid
        | O::RemEuclid
        | O::CopySign
        | O::Powf
        | O::ToDegrees
        | O::ToRadians
        | O::IsNormal
        | O::Classify
        | O::ToBeBytes
        | O::ToLeBytes
        | O::ToNeBytes
        | O::FromBeBytes
        | O::FromLeBytes
        | O::FromNeBytes => {
            return Err(unsupported(format!(
                "f128 method {name} is not runtime-supported"
            )));
        }
        O::FromBits => unreachable!("handled above"),
    })
}

/// Execute one primitive `f16` method. The receiver rides as its
/// exact `f64` widening. How each operation is evaluated follows
/// how the native `f16` operation itself computes:
///
/// * The elementary operations whose `f64` evaluation of two `f16`
///   inputs is exact (`abs`, `sqrt`, the rounding family, `fract`,
///   `recip`, `signum`, `min`, `max`) compute in `f64` and round
///   the result back to `f16`.
/// * The libm-backed family computes at `f32` precision: on this
///   platform LLVM promotes the `f16` intrinsics to their `f32`
///   forms (and std's `f16` wrappers call the `f32` `cmath`
///   functions), so the host `f32` operation on the exactly
///   widened input, rounded once to `f16`, is the native result.
///   Computing this family in `f64` instead can differ by one
///   `f16` ulp.
/// * The composed operations (`div_euclid`, `rem_euclid`, `log`,
///   `atanh`, `powi`) follow their core/std/compiler-rt
///   definitions step by step, rounding each intermediate that
///   the definition itself produces at `f16` precision: in
///   `div_euclid` the quotient rounds to `f16` before it is
///   truncated, so `1.0.div_euclid(0x3267)` is `5.0`, not the
///   `4.0` an all-`f64` evaluation returns. `mul_add` is exact
///   (a single rounding of the exact product-sum), which the
///   host `f64` fused multiply-add reproduces.
/// * `to_degrees` / `to_radians` multiply by the `f16` constants
///   core itself uses, not by the `f64` conversion factors.
pub(super) fn exec_float16_method(
    name: &str,
    op: FloatOp,
    vals: &[Value],
) -> InterpResult<Value> {
    use crate::f128::{f16_bits_to_f64, f64_to_f16_bits};
    let arity = |n: usize| -> InterpResult<()> {
        if vals.len() == n {
            Ok(())
        } else {
            Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                format!("{n} argument(s)"),
                format!("{} argument(s)", vals.len()),
            ))
        }
    };
    let as_float = |v: &Value| -> InterpResult<f64> {
        match v {
            Value::Float(x) => Ok(*x),
            other => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "f16",
                other.kind_name(),
            )),
        }
    };
    use FloatOp as O;
    if op == O::FromBits {
        arity(1)?;
        let bits = match vals[0] {
            Value::Uint(v) => v,
            Value::Int(v) => v as u128,
            ref other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "integer",
                    other.kind_name(),
                ));
            }
        };
        return Ok(Value::Float(f16_bits_to_f64(bits as u16)));
    }
    if matches!(op, O::FromBeBytes | O::FromLeBytes | O::FromNeBytes) {
        // An associated function like `from_bits`: no receiver,
        // the single argument is the `[u8; 2]` byte array.
        arity(1)?;
        let bytes = as_byte_array(name, &vals[0])?;
        let bits = match op {
            O::FromBeBytes => u16::from_be_bytes(bytes),
            O::FromLeBytes => u16::from_le_bytes(bytes),
            _ => u16::from_ne_bytes(bytes),
        };
        return Ok(Value::Float(f16_bits_to_f64(bits)));
    }
    arity(match op {
        O::MulAdd => 3,
        O::Powi
        | O::Powf
        | O::Min
        | O::Max
        | O::Log
        | O::Atan2
        | O::Hypot
        | O::DivEuclid
        | O::RemEuclid
        | O::CopySign => 2,
        _ => 1,
    })?;
    let x = as_float(&vals[0])?;
    // The exact `f16` value of an intermediate computed from `f16`
    // inputs, as its `f64` widening.
    let f16_of = |r: f64| f16_bits_to_f64(f64_to_f16_bits(r));
    let round = |r: f64| Value::Float(f16_of(r));
    // The libm-backed family evaluates at `f32` precision (see the
    // function comment): the receiver widened to `f32` is exact.
    let x32 = x as f32;
    Ok(match op {
        O::IsNan => Value::Bool(x.is_nan()),
        O::IsInfinite => Value::Bool(x.is_infinite()),
        O::IsFinite => Value::Bool(x.is_finite()),
        O::IsSignPositive => Value::Bool(x.is_sign_positive()),
        O::IsSignNegative => Value::Bool(x.is_sign_negative()),
        O::Abs => round(x.abs()),
        O::Sqrt => round(x.sqrt()),
        // The rounding family evaluates at `f32` precision, like
        // the libm family: the native `f16` operations promote to
        // `f32`, whose rounding instructions quiet a signaling NaN
        // (payload preserved), while the host `f64` library
        // floor / ceil / trunc return a signaling NaN unchanged.
        // For `f16` inputs the `f32` results are exact, so no
        // finite result changes. `round_ties_even` at `f64`
        // precision likewise leaves a signaling NaN unchanged on
        // this host, so it takes the `f32` path too; `round` and
        // `fract` at `f64` already quiet signaling NaNs and match
        // the native results bit for bit (measured over the f16
        // probe battery), so they stay at `f64`.
        O::Floor => round(x32.floor() as f64),
        O::Ceil => round(x32.ceil() as f64),
        O::Round => round(x.round()),
        O::RoundTiesEven => round(x32.round_ties_even() as f64),
        O::Trunc => round(x32.trunc() as f64),
        O::Fract => round(x.fract()),
        O::Recip => round(x.recip()),
        O::Signum => round(x.signum()),
        O::Exp => round(x32.exp() as f64),
        O::Exp2 => round(x32.exp2() as f64),
        O::ExpM1 => round(x32.exp_m1() as f64),
        O::Ln => round(x32.ln() as f64),
        O::Ln1p => round(x32.ln_1p() as f64),
        O::Log2 => round(x32.log2() as f64),
        O::Log10 => round(x32.log10() as f64),
        O::Sin => round(x32.sin() as f64),
        O::Cos => round(x32.cos() as f64),
        O::Tan => round(x32.tan() as f64),
        O::Asin => round(x32.asin() as f64),
        O::Acos => round(x32.acos() as f64),
        O::Atan => round(x32.atan() as f64),
        O::Sinh => round(x32.sinh() as f64),
        O::Cosh => round(x32.cosh() as f64),
        O::Tanh => round(x32.tanh() as f64),
        O::Asinh => round(x32.asinh() as f64),
        O::Acosh => round(x32.acosh() as f64),
        O::Atanh => {
            // std defines `f16::atanh` by the formula
            // `0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()`,
            // evaluated at `f16` precision step by step (the
            // subtraction and the division round to `f16`; the
            // `ln_1p` is the `f16` one).
            let den = f16_of(1.0 - x);
            let q = f16_of(2.0 * x / den);
            round(0.5 * f16_of((q as f32).ln_1p() as f64))
        }
        O::Cbrt => round(x32.cbrt() as f64),
        O::SinCos => {
            let (s, c) = x32.sin_cos();
            Value::Tuple(vec![round(s as f64), round(c as f64)])
        }
        O::Log => {
            let y = as_float(&vals[1])?;
            // std defines `f16::log` as `self.ln() / base.ln()`:
            // both logarithms are `f16` values before the division,
            // and the division itself rounds to `f16`.
            round(f16_of(x32.ln() as f64) / f16_of((y as f32).ln() as f64))
        }
        O::Atan2 => round(x32.atan2(as_float(&vals[1])? as f32) as f64),
        O::Hypot => round(x32.hypot(as_float(&vals[1])? as f32) as f64),
        O::DivEuclid => {
            let y = as_float(&vals[1])?;
            // core's definition, step by step at `f16` precision:
            // `q = (self / rhs).trunc()`, adjusted by one when the
            // remainder is negative. The division rounds to `f16`
            // before the truncation; the remainder of two `f16`
            // values is exact in `f64`; the adjustment rounds.
            let q = f16_of(x / y).trunc();
            if x % y < 0.0 {
                if y > 0.0 {
                    round(q - 1.0)
                } else {
                    round(q + 1.0)
                }
            } else {
                Value::Float(q)
            }
        }
        O::RemEuclid => {
            let y = as_float(&vals[1])?;
            // core's definition: `r = self % rhs` (exact), then
            // `r + rhs.abs()` when `r` is negative; that sum
            // rounds to `f16`.
            let r = x % y;
            if r < 0.0 {
                round(r + y.abs())
            } else {
                Value::Float(r)
            }
        }
        // `fmaf16` is exact: the product and the sum are formed
        // without intermediate rounding and round once to `f16`,
        // which the host `f64` fused multiply-add reproduces
        // (an `f32` fused multiply-add rounds the sum to `f32`
        // first and can land one `f16` ulp lower).
        O::MulAdd => round(x.mul_add(as_float(&vals[1])?, as_float(&vals[2])?)),
        // core multiplies by its own `f16` constants
        // (57.2957795130823208767981548141051703_f16 = 0x5329 and
        // 0.017453292519943295769236907684886_f16 = 0x2478), whose
        // values differ from the `f64` conversion factors.
        O::ToDegrees => round(x * f16_bits_to_f64(0x5329)),
        O::ToRadians => round(x * f16_bits_to_f64(0x2478)),
        O::ToBits => Value::Uint(f64_to_f16_bits(x) as u128),
        // `is_normal` and `classify` are judged from the `u16`
        // bit pattern, not from the widened `f64`: an `f16`
        // subnormal widens to a normal `f64`, so the widened
        // value's own classification would differ.
        O::IsNormal => {
            let exp = (f64_to_f16_bits(x) >> 10) & 0x1f;
            Value::Bool(exp != 0 && exp != 0x1f)
        }
        O::Classify => fp_category_value(f64_to_f16_bits(x)),
        O::ToBeBytes => byte_array_value(f64_to_f16_bits(x).to_be_bytes()),
        O::ToLeBytes => byte_array_value(f64_to_f16_bits(x).to_le_bytes()),
        O::ToNeBytes => byte_array_value(f64_to_f16_bits(x).to_ne_bytes()),
        O::FromBeBytes | O::FromLeBytes | O::FromNeBytes => {
            unreachable!("handled above")
        }
        O::Powi => {
            let n = match vals[1] {
                Value::Int(v) => v as i32,
                Value::Uint(v) => v as i32,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "i32",
                        other.kind_name(),
                    ));
                }
            };
            // `powif16` runs the compiler-rt power loop at `f16`
            // precision: every multiplication rounds to `f16`
            // (the operands stay `f16` values, so their products
            // are exact in `f64` before that rounding), and a
            // negative exponent inverts last with one `f16`
            // division. Evaluating the power in `f32` or `f64`
            // instead disagrees by several ulps on a large part
            // of the input space.
            let mut pw = n.unsigned_abs();
            let mut mul = 1.0;
            let mut a = x;
            loop {
                if pw & 1 != 0 {
                    mul = f16_of(mul * a);
                }
                pw >>= 1;
                if pw == 0 {
                    break;
                }
                a = f16_of(a * a);
            }
            if n < 0 {
                round(1.0 / mul)
            } else {
                Value::Float(mul)
            }
        }
        O::Powf => round(x32.powf(as_float(&vals[1])? as f32) as f64),
        O::Min => round(x.min(as_float(&vals[1])?)),
        O::Max => round(x.max(as_float(&vals[1])?)),
        O::CopySign => {
            let y = as_float(&vals[1])?;
            // Bit composition, not arithmetic (the shape of
            // core's own fallback body for `copysignf16`): the
            // result keeps self's magnitude bits — a NaN payload
            // included, unquieted — and takes the sign argument's
            // sign bit.
            let magnitude = f64_to_f16_bits(x) & 0x7fff;
            let sign = f64_to_f16_bits(y) & 0x8000;
            Value::Float(f16_bits_to_f64(magnitude | sign))
        }
        O::FromBits => unreachable!("handled above"),
    })
}

/// Execute one primitive float method. The operation runs at the
/// method's own width — an `f32` method computes on the `f32` value
/// and the result widens back (widening is exact) — so rounding, NaN
/// propagation and signed-zero behaviour are the native ones by
/// construction: the host `f32` / `f64` methods are the definitions
/// core and std themselves use.
pub(super) fn exec_float_method(
    name: &str,
    f32_width: bool,
    op: FloatOp,
    vals: &[Value],
) -> InterpResult<Value> {
    let arity = |n: usize| -> InterpResult<()> {
        if vals.len() == n {
            Ok(())
        } else {
            Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                format!("{n} argument(s)"),
                format!("{} argument(s)", vals.len()),
            ))
        }
    };
    let as_float = |v: &Value| -> InterpResult<f64> {
        match v {
            Value::Float(x) => Ok(*x),
            other => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "float",
                other.kind_name(),
            )),
        }
    };
    use FloatOp as O;
    // `from_bits` is an associated function: no receiver, the single
    // argument is the bit pattern.
    if op == O::FromBits {
        arity(1)?;
        let bits = match vals[0] {
            Value::Uint(v) => v,
            Value::Int(v) => v as u128,
            ref other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "integer",
                    other.kind_name(),
                ));
            }
        };
        return Ok(Value::Float(if f32_width {
            crate::f128::f32_bits_to_f64_exact(bits as u32)
        } else {
            f64::from_bits(bits as u64)
        }));
    }
    arity(match op {
        O::MulAdd => 3,
        O::Powi
        | O::Powf
        | O::Min
        | O::Max
        | O::Log
        | O::Atan2
        | O::Hypot
        | O::DivEuclid
        | O::RemEuclid
        | O::CopySign => 2,
        _ => 1,
    })?;
    let x = as_float(&vals[0])?;
    macro_rules! unary {
        ($method:ident) => {
            if f32_width {
                (x as f32).$method() as f64
            } else {
                x.$method()
            }
        };
    }
    macro_rules! predicate {
        ($method:ident) => {
            if f32_width {
                (x as f32).$method()
            } else {
                x.$method()
            }
        };
    }
    Ok(match op {
        O::IsNan => Value::Bool(predicate!(is_nan)),
        O::IsInfinite => Value::Bool(predicate!(is_infinite)),
        O::IsFinite => Value::Bool(predicate!(is_finite)),
        O::IsSignPositive => Value::Bool(predicate!(is_sign_positive)),
        O::IsSignNegative => Value::Bool(predicate!(is_sign_negative)),
        O::Abs => Value::Float(unary!(abs)),
        O::Sqrt => Value::Float(unary!(sqrt)),
        O::Floor => Value::Float(unary!(floor)),
        O::Ceil => Value::Float(unary!(ceil)),
        O::Round => Value::Float(unary!(round)),
        O::RoundTiesEven => Value::Float(unary!(round_ties_even)),
        O::Trunc => Value::Float(unary!(trunc)),
        O::Fract => Value::Float(unary!(fract)),
        O::Recip => Value::Float(unary!(recip)),
        O::Signum => Value::Float(unary!(signum)),
        O::ToDegrees => Value::Float(unary!(to_degrees)),
        O::ToRadians => Value::Float(unary!(to_radians)),
        O::Exp => Value::Float(unary!(exp)),
        O::Exp2 => Value::Float(unary!(exp2)),
        O::ExpM1 => Value::Float(unary!(exp_m1)),
        O::Ln => Value::Float(unary!(ln)),
        O::Ln1p => Value::Float(unary!(ln_1p)),
        O::Log2 => Value::Float(unary!(log2)),
        O::Log10 => Value::Float(unary!(log10)),
        O::Sin => Value::Float(unary!(sin)),
        O::Cos => Value::Float(unary!(cos)),
        O::Tan => Value::Float(unary!(tan)),
        O::Asin => Value::Float(unary!(asin)),
        O::Acos => Value::Float(unary!(acos)),
        O::Atan => Value::Float(unary!(atan)),
        O::Sinh => Value::Float(unary!(sinh)),
        O::Cosh => Value::Float(unary!(cosh)),
        O::Tanh => Value::Float(unary!(tanh)),
        O::Asinh => Value::Float(unary!(asinh)),
        O::Acosh => Value::Float(unary!(acosh)),
        O::Atanh => Value::Float(unary!(atanh)),
        O::Cbrt => Value::Float(unary!(cbrt)),
        O::SinCos => {
            let (s, c) = if f32_width {
                let (s, c) = (x as f32).sin_cos();
                (s as f64, c as f64)
            } else {
                x.sin_cos()
            };
            Value::Tuple(vec![Value::Float(s), Value::Float(c)])
        }
        O::Log => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).log(y as f32) as f64
            } else {
                x.log(y)
            })
        }
        O::Atan2 => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).atan2(y as f32) as f64
            } else {
                x.atan2(y)
            })
        }
        O::Hypot => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).hypot(y as f32) as f64
            } else {
                x.hypot(y)
            })
        }
        O::DivEuclid => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).div_euclid(y as f32) as f64
            } else {
                x.div_euclid(y)
            })
        }
        O::RemEuclid => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).rem_euclid(y as f32) as f64
            } else {
                x.rem_euclid(y)
            })
        }
        O::MulAdd => {
            let a = as_float(&vals[1])?;
            let b = as_float(&vals[2])?;
            Value::Float(if f32_width {
                (x as f32).mul_add(a as f32, b as f32) as f64
            } else {
                x.mul_add(a, b)
            })
        }
        O::ToBits => Value::Uint(if f32_width {
            crate::f128::f64_to_f32_bits_exact(x) as u128
        } else {
            x.to_bits() as u128
        }),
        O::Powi => {
            let n = match vals[1] {
                Value::Int(v) => v as i32,
                Value::Uint(v) => v as i32,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "i32",
                        other.kind_name(),
                    ));
                }
            };
            Value::Float(if f32_width {
                (x as f32).powi(n) as f64
            } else {
                x.powi(n)
            })
        }
        O::Powf => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).powf(y as f32) as f64
            } else {
                x.powf(y)
            })
        }
        O::Min => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).min(y as f32) as f64
            } else {
                x.min(y)
            })
        }
        O::Max => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).max(y as f32) as f64
            } else {
                x.max(y)
            })
        }
        O::CopySign => {
            let y = as_float(&vals[1])?;
            // Bit composition at the method's own width (the
            // shape of core's fallback bodies for `copysignf32` /
            // `copysignf64`): self's magnitude bits, the sign
            // argument's sign bit; a NaN payload passes through
            // unquieted.
            Value::Float(if f32_width {
                let magnitude = crate::f128::f64_to_f32_bits_exact(x) & 0x7fff_ffff;
                let sign = crate::f128::f64_to_f32_bits_exact(y) & 0x8000_0000;
                crate::f128::f32_bits_to_f64_exact(magnitude | sign)
            } else {
                f64::from_bits(
                    (x.to_bits() & !(1u64 << 63)) | (y.to_bits() & (1u64 << 63)),
                )
            })
        }
        O::IsNormal
        | O::Classify
        | O::ToBeBytes
        | O::ToLeBytes
        | O::ToNeBytes
        | O::FromBeBytes
        | O::FromLeBytes
        | O::FromNeBytes => {
            return Err(unsupported(format!(
                "float method {name} is not runtime-supported"
            )));
        }
        O::FromBits => unreachable!("handled above"),
    })
}

/// The two bytes of a `[u8; 2]` argument (the byte-array
/// argument of the `f16::from_*_bytes` associated functions):
/// arrays ride as tuples of their elements; a byte-array
/// constant arrives as raw bytes.
fn as_byte_array(name: &str, v: &Value) -> InterpResult<[u8; 2]> {
    let byte = |e: &Value| -> InterpResult<u8> {
        match e {
            Value::Uint(v) => Ok(*v as u8),
            Value::Int(v) => Ok(*v as u8),
            other => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "u8",
                other.kind_name(),
            )),
        }
    };
    match v {
        Value::Tuple(elems) if elems.len() == 2 => Ok([byte(&elems[0])?, byte(&elems[1])?]),
        Value::Bytes(b) if b.len() == 2 => Ok([b[0], b[1]]),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "[u8; 2]",
            other.kind_name(),
        )),
    }
}

/// A `[u8; 2]` value: arrays ride as tuples of their elements.
fn byte_array_value(bytes: [u8; 2]) -> Value {
    Value::Tuple(bytes.iter().map(|b| Value::Uint(u128::from(*b))).collect())
}

/// A `core::num::FpCategory` value for an `f16` bit pattern:
/// the variants in declaration order (Nan, Infinite, Zero,
/// Subnormal, Normal), judged from the bits themselves (an
/// `f16` subnormal widens to a normal `f64`, so the widened
/// value cannot be classified in its place).
fn fp_category_value(bits: u16) -> Value {
    let exp = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;
    let variant = if exp == 0x1f {
        if mantissa != 0 { 0 } else { 1 }
    } else if exp == 0 {
        if mantissa != 0 { 3 } else { 2 }
    } else {
        4
    };
    Value::Enum {
        variant,
        fields: Vec::new(),
    }
}
