//! SIMD lane execution and lane value helpers.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    float_to_int,
    float_to_uint,
    round_to_f16, synthetic_address,
    Frame, Interpreter,
};
use crate::value::{RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// The lane shape of a vector type: (lane width in bits,
    /// signed lanes, float width in bits when the lanes are
    /// floats). Taken from the element type in the tag when it
    /// names one, else from the vector's layout and lane count,
    /// else from the lanes' own representation.
    fn simd_lane_info(&self, tag: &str, lanes: &[Value]) -> (u32, bool, Option<u32>) {
        if let Some(elem) = vector_elem_tag(tag) {
            if let Some((bits, signed)) = int_tag(elem) {
                return (bits, signed, None);
            }
            match elem {
                "f16" => return (16, true, Some(16)),
                "f32" => return (32, true, Some(32)),
                "f64" => return (64, true, Some(64)),
                "bool" => return (8, false, None),
                _ => {}
            }
        }
        if let Some((size, _)) = self.layout_of(tag) {
            if !lanes.is_empty() && size % lanes.len() as u128 == 0 {
                let bits = ((size / lanes.len() as u128 * 8) as u32).min(128);
                return match lanes.first() {
                    Some(Value::Float(_)) => (bits, true, Some(bits)),
                    Some(Value::Int(_)) => (bits, true, None),
                    _ => (bits, false, None),
                };
            }
        }
        match lanes.first() {
            Some(Value::Float(_)) => (64, true, Some(64)),
            Some(Value::Int(_)) => (128, true, None),
            _ => (128, false, None),
        }
    }

    /// One `std::intrinsics::simd::*` operation over the lanes
    /// of its vector arguments. Integer lanes wrap at the lane
    /// width; a shift amount is taken modulo the lane width (the
    /// hardware behaviour); a zero integer divisor lane fails the
    /// guest the way a scalar division by zero does.
    pub(super) fn exec_simd(
        &self,
        op: SimdOp,
        frames: &mut Vec<Frame>,
        generic_args: &[String],
        vals: &[Value],
    ) -> InterpResult<Value> {
        use SimdOp as S;
        let tag0 = generic_args.first().map(String::as_str).unwrap_or("");
        match op {
            S::Add
            | S::Sub
            | S::Mul
            | S::Div
            | S::Rem
            | S::And
            | S::Or
            | S::Xor
            | S::Shl
            | S::Shr
            | S::SaturatingAdd
            | S::SaturatingSub
            | S::CarrylessMul
            | S::FunnelShl
            | S::FunnelShr
            | S::MaximumNumber
            | S::MinimumNumber => {
                let a = simd_lanes(&vals[0])?;
                let b = simd_lanes(&vals[1])?;
                let info = self.simd_lane_info(tag0, &a);
                let mut out = Vec::with_capacity(a.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    out.push(simd_binop(op, x, y, info)?);
                }
                simd_wrap(&vals[0], out)
            }
            S::Eq | S::Ne | S::Lt | S::Le | S::Gt | S::Ge => {
                let a = simd_lanes(&vals[0])?;
                let b = simd_lanes(&vals[1])?;
                let info = self.simd_lane_info(tag0, &a);
                let mut out = Vec::with_capacity(a.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    let yes = simd_compare(op, x, y, info)?;
                    out.push(Value::Int(if yes { -1 } else { 0 }));
                }
                simd_wrap(&vals[0], out)
            }
            S::Neg
            | S::Fabs
            | S::Ceil
            | S::Floor
            | S::Trunc
            | S::Round
            | S::RoundTiesEven
            | S::Fsqrt
            | S::Fexp
            | S::Fexp2
            | S::Flog
            | S::Flog2
            | S::Flog10
            | S::Fsin
            | S::Fcos
            | S::Bswap
            | S::BitReverse
            | S::Ctlz
            | S::Ctpop
            | S::Cttz => {
                let a = simd_lanes(&vals[0])?;
                let info = self.simd_lane_info(tag0, &a);
                let mut out = Vec::with_capacity(a.len());
                for x in a.iter() {
                    out.push(simd_unop(op, x, info)?);
                }
                simd_wrap(&vals[0], out)
            }
            S::Fma => {
                let a = simd_lanes(&vals[0])?;
                let b = simd_lanes(&vals[1])?;
                let c = simd_lanes(&vals[2])?;
                let info = self.simd_lane_info(tag0, &a);
                let mut out = Vec::with_capacity(a.len());
                for ((x, y), z) in a.iter().zip(b.iter()).zip(c.iter()) {
                    let (x, y, z) = (
                        simd_as_f64(x, "simd lane")?,
                        simd_as_f64(y, "simd lane")?,
                        simd_as_f64(z, "simd lane")?,
                    );
                    let r = if info.2 == Some(32) {
                        (x as f32).mul_add(y as f32, z as f32) as f64
                    } else {
                        x.mul_add(y, z)
                    };
                    out.push(Value::Float(simd_round_float(r, info.2)));
                }
                simd_wrap(&vals[0], out)
            }
            S::Splat => {
                // The generic arguments name the element type and
                // the vector type, in one order or the other; the
                // scalar one is the element. Raw pointers are
                // scalar elements too (a vector of pointers is a
                // supported `simd_splat` target).
                let scalar = generic_args
                    .iter()
                    .find(|t| {
                        let t = t.as_str();
                        int_tag(t).is_some()
                            || matches!(t, "f16" | "f32" | "f64" | "bool" | "char")
                            || t.starts_with("*const ")
                            || t.starts_with("*mut ")
                    })
                    .ok_or_else(|| {
                        unsupported("simd_splat without an element type is not runtime-supported")
                    })?;
                let vector = generic_args
                    .iter()
                    .find(|t| t.as_str() != scalar.as_str())
                    .ok_or_else(|| {
                        unsupported("simd_splat without a vector type is not runtime-supported")
                    })?;
                let count = match simd_tag_lane_count(vector) {
                    Some(n) => n,
                    None => {
                        let (vsize, _) = self.layout_of(vector).ok_or_else(|| {
                            unsupported("simd_splat vector size is not runtime-supported")
                        })?;
                        let (esize, _) = self.layout_of(scalar).ok_or_else(|| {
                            unsupported("simd_splat element size is not runtime-supported")
                        })?;
                        if esize == 0 || vsize % esize != 0 {
                            return Err(unsupported(
                                "simd_splat lane count is not runtime-supported",
                            ));
                        }
                        (vsize / esize) as usize
                    }
                };
                let lanes = vec![vals[0].clone(); count];
                if vector.trim_start().starts_with('[') {
                    Ok(Value::Tuple(lanes))
                } else {
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Tuple(lanes)],
                    })
                }
            }
            S::Extract => {
                let lanes = simd_lanes(&vals[0])?;
                let idx = simd_index(&vals[1])?;
                lanes.get(idx).cloned().ok_or_else(|| InterpError::Panic {
                    message: format!("simd lane index {idx} out of range"),
                    payload: None,
                })
            }
            S::Insert => {
                let mut lanes = simd_lanes(&vals[0])?;
                let idx = simd_index(&vals[1])?;
                if idx >= lanes.len() {
                    return Err(InterpError::Panic {
                        message: format!("simd lane index {idx} out of range"),
                        payload: None,
                    });
                }
                lanes[idx] = vals[2].clone();
                simd_wrap(&vals[0], lanes)
            }
            S::Shuffle => {
                let a = simd_lanes(&vals[0])?;
                let b = simd_lanes(&vals[1])?;
                let idx = simd_lanes(&vals[2])?;
                let mut out = Vec::with_capacity(idx.len());
                for i in idx.iter() {
                    let i = simd_index(i)?;
                    let lane = if i < a.len() {
                        a.get(i)
                    } else {
                        b.get(i - a.len())
                    };
                    out.push(lane.cloned().ok_or_else(|| InterpError::Panic {
                        message: format!("simd shuffle index {i} out of range"),
                        payload: None,
                    })?);
                }
                simd_wrap(&vals[0], out)
            }
            S::Cast => {
                let src = simd_lanes(&vals[0])?;
                let src_info = self.simd_lane_info(tag0, &src);
                let dst_tag = generic_args.get(1).map(String::as_str).unwrap_or("");
                let dst_elem = vector_elem_tag(dst_tag).ok_or_else(|| {
                    unsupported("simd cast target element type is not runtime-supported")
                })?;
                let mut out = Vec::with_capacity(src.len());
                for lane in src.iter() {
                    out.push(simd_cast_lane(lane, src_info, dst_elem)?);
                }
                simd_wrap(&vals[0], out)
            }
            S::Bitmask => {
                let lanes = simd_lanes(&vals[0])?;
                let mut bits: u128 = 0;
                for (i, lane) in lanes.iter().enumerate() {
                    if simd_lane_active(lane)? {
                        bits |= 1u128 << i;
                    }
                }
                // The result type is the call's second generic
                // argument, and `simd_bitmask` is declared over
                // two shapes for it: an integer, or a byte array
                // holding the mask's little-endian byte
                // decomposition.
                if let Some(bytes) = generic_args
                    .get(1)
                    .and_then(|tag| tag.trim().strip_prefix("[u8;"))
                    .and_then(|rest| rest.trim().strip_suffix(']'))
                    .and_then(|n| n.trim().parse::<usize>().ok())
                {
                    let out: Vec<Value> = (0..bytes)
                        .map(|i| Value::Uint((bits >> (8 * i)) & 0xff))
                        .collect();
                    return Ok(Value::Tuple(out));
                }
                Ok(Value::Uint(bits))
            }
            S::Select => {
                let mask = simd_lanes(&vals[0])?;
                let yes = simd_lanes(&vals[1])?;
                let no = simd_lanes(&vals[2])?;
                let mut out = Vec::with_capacity(yes.len());
                for (i, lane) in yes.iter().enumerate() {
                    out.push(if simd_lane_active(&mask[i])? {
                        lane.clone()
                    } else {
                        no[i].clone()
                    });
                }
                simd_wrap(&vals[1], out)
            }
            S::SelectBitmask => {
                // The mask arrives as an integer or in the byte
                // array form `simd_bitmask` also produces; the
                // array's bytes compose little-endian.
                let bits = match &vals[0] {
                    Value::Tuple(items) => {
                        let mut bits: u128 = 0;
                        for (i, item) in items.iter().enumerate() {
                            bits |=
                                as_u128(item, "simd select bitmask")? << (8 * i);
                        }
                        bits
                    }
                    Value::Bytes(bytes) => bytes.iter().enumerate().fold(
                        0u128,
                        |acc, (i, b)| acc | ((*b as u128) << (8 * i)),
                    ),
                    other => as_u128(other, "simd select bitmask")?,
                };
                let yes = simd_lanes(&vals[1])?;
                let no = simd_lanes(&vals[2])?;
                let mut out = Vec::with_capacity(yes.len());
                for (i, lane) in yes.iter().enumerate() {
                    out.push(if (bits >> i) & 1 == 1 {
                        lane.clone()
                    } else {
                        no[i].clone()
                    });
                }
                simd_wrap(&vals[1], out)
            }
            S::Gather => {
                let defaults = simd_lanes(&vals[0])?;
                let ptrs = simd_lanes(&vals[1])?;
                let mask = simd_lanes(&vals[2])?;
                let mut out = Vec::with_capacity(defaults.len());
                for (i, default) in defaults.iter().enumerate() {
                    if simd_lane_active(&mask[i])? {
                        let target = self.deref_value(frames, &ptrs[i])?;
                        out.push(self.read_resolved(frames, &target)?);
                    } else {
                        out.push(default.clone());
                    }
                }
                simd_wrap(&vals[0], out)
            }
            S::Scatter => {
                let values = simd_lanes(&vals[0])?;
                let ptrs = simd_lanes(&vals[1])?;
                let mask = simd_lanes(&vals[2])?;
                for (i, value) in values.iter().enumerate() {
                    if simd_lane_active(&mask[i])? {
                        let target = self.deref_value(frames, &ptrs[i])?;
                        self.write_resolved(frames, &target, value.clone())?;
                    }
                }
                Ok(Value::Unit)
            }
            S::MaskedLoad => {
                let mask = simd_lanes(&vals[0])?;
                let defaults = simd_lanes(&vals[2])?;
                let target = self.deref_value(frames, &vals[1])?;
                let aggregate = matches!(
                    self.read_resolved(frames, &target),
                    Ok(Value::Tuple(_)) | Ok(Value::Enum { .. })
                );
                let mut out = Vec::with_capacity(defaults.len());
                for (i, default) in defaults.iter().enumerate() {
                    if simd_lane_active(&mask[i])? {
                        let value = if aggregate || i > 0 {
                            // The sequence accessor, not the plain
                            // indexed one: the pointer may denote
                            // a single element (an element pointer
                            // into an array), whose lanes are its
                            // siblings in the enclosing aggregate,
                            // not indices into the element itself.
                            self.read_seq_at(frames, &target, i)?
                        } else {
                            self.read_resolved(frames, &target)?
                        };
                        out.push(value);
                    } else {
                        out.push(default.clone());
                    }
                }
                simd_wrap(&vals[2], out)
            }
            S::MaskedStore => {
                let mask = simd_lanes(&vals[0])?;
                let values = simd_lanes(&vals[2])?;
                let target = self.deref_value(frames, &vals[1])?;
                let mut aggregate = false;
                let mut uninit = false;
                match self.read_resolved(frames, &target) {
                    Ok(Value::Tuple(_)) | Ok(Value::Enum { .. }) => aggregate = true,
                    Err(InterpError::UninitializedRead { .. }) => uninit = true,
                    _ => {}
                }
                if !aggregate
                    && uninit
                    && mask.iter().any(|m| simd_lane_active(m).unwrap_or(false))
                {
                    // Materialize the region uninitialized so the
                    // masked lanes can land; the others stay
                    // uninitialized, as they should.
                    self.write_resolved(
                        frames,
                        &target,
                        Value::Tuple(vec![Value::Uninit; values.len()]),
                    )?;
                    aggregate = true;
                }
                for (i, value) in values.iter().enumerate() {
                    if simd_lane_active(&mask[i])? {
                        if aggregate || i > 0 {
                            // Sequence access, as in MaskedLoad:
                            // an element pointer's lanes are its
                            // siblings, not indices into it.
                            self.write_seq_at(frames, &target, i, value.clone())?;
                        } else {
                            self.write_resolved(frames, &target, value.clone())?;
                        }
                    }
                }
                Ok(Value::Unit)
            }
            S::ArithOffset => {
                let ptrs = simd_lanes(&vals[0])?;
                let offsets = simd_lanes(&vals[1])?;
                // The element type of the pointer vector names
                // the pointee; the interpreter's own offset
                // machinery applies the per-lane offsets, wrapping
                // as the intrinsic's name promises.
                let elem_size = vector_elem_tag(tag0)
                    .and_then(|elem| {
                        let elem = elem.trim();
                        elem.strip_prefix("*const ")
                            .or_else(|| elem.strip_prefix("*mut "))
                    })
                    .and_then(|pointee| self.layout_of(pointee.trim()))
                    .map(|(size, _)| size as u64);
                let mut out = Vec::with_capacity(ptrs.len());
                for (ptr, off) in ptrs.iter().zip(offsets.iter()) {
                    out.push(self.ptr_offset_value(
                        frames,
                        ptr.clone(),
                        as_i128(off, "simd lane")?,
                        elem_size,
                        true,
                    )?);
                }
                simd_wrap(&vals[0], out)
            }
            S::ReduceAddOrdered
            | S::ReduceAddUnordered
            | S::ReduceMulOrdered
            | S::ReduceMulUnordered
            | S::ReduceMax
            | S::ReduceMin
            | S::ReduceAnd
            | S::ReduceOr
            | S::ReduceXor => {
                let lanes = simd_lanes(&vals[0])?;
                let info = self.simd_lane_info(tag0, &lanes);
                simd_reduce(op, &lanes, vals.get(1), info)
            }
            S::ReduceAll | S::ReduceAny => {
                let lanes = simd_lanes(&vals[0])?;
                let mut result = op == S::ReduceAll;
                for lane in lanes.iter() {
                    let active = simd_lane_active(lane)?;
                    if op == S::ReduceAll {
                        result &= active;
                    } else {
                        result |= active;
                    }
                }
                Ok(Value::Bool(result))
            }
            S::CastPtr => Ok(vals[0].clone()),
            S::ExposeProvenance => {
                let lanes = simd_lanes(&vals[0])?;
                let mut out = Vec::with_capacity(lanes.len());
                for lane in lanes.iter() {
                    let addr = self
                        .arithmetic_price(lane, None, Some(frames))
                        .or_else(|| synthetic_address(lane))
                        .ok_or_else(|| {
                        unsupported("simd_expose_provenance of this lane is not runtime-supported")
                    })?;
                    out.push(Value::Uint(addr));
                }
                simd_wrap(&vals[0], out)
            }
            S::WithExposedProvenance => {
                let lanes = simd_lanes(&vals[0])?;
                let mut out = Vec::with_capacity(lanes.len());
                for lane in lanes.iter() {
                    let addr = as_u128(lane, "simd lane")?;
                    let addr = u64::try_from(addr).map_err(|_| {
                        unsupported("simd_with_exposed_provenance address is not runtime-supported")
                    })?;
                    out.push(Value::Ref(RefTarget::Address(addr)));
                }
                simd_wrap(&vals[0], out)
            }
        }
    }
}

/// A float lane's value.
fn simd_as_f64(value: &Value, ctx: &str) -> InterpResult<f64> {
    match value {
        Value::Float(x) => Ok(*x),
        other => Err(InterpError::type_mismatch(ctx, "float lane", other.kind_name())),
    }
}

/// The lane count named by a vector type tag: the length of
/// an array tag `[E; N]`, or a trailing const argument
/// (`Simd<i32, 4>`).
fn simd_tag_lane_count(tag: &str) -> Option<usize> {
    let tag = tag.trim();
    if let Some(inner) = tag.strip_prefix('[') {
        let (_, n) = inner.rsplit_once(';')?;
        return n.trim().trim_end_matches(']').trim().parse().ok();
    }
    let open = tag.find('<')?;
    let close = tag.rfind('>')?;
    if close <= open {
        return None;
    }
    let parts = split_top_level(&tag[open + 1..close]);
    parts.last().and_then(|p| p.trim().parse().ok())
}

/// The lanes of a vector value: the elements of its lane
/// array, reached through the one-field wrapper a SIMD type
/// is. A bare tuple is the lanes directly.
fn simd_lanes(value: &Value) -> InterpResult<Vec<Value>> {
    match value {
        Value::Tuple(lanes) => Ok(lanes.clone()),
        Value::Enum { fields, .. } if fields.len() == 1 => simd_lanes(&fields[0]),
        _ => Err(InterpError::type_mismatch(
            "simd operation",
            "vector value",
            value.kind_name(),
        )),
    }
}

/// Rebuild a vector value with `lanes` in the shape of
/// `template` (the value the lanes were taken from).
fn simd_wrap(template: &Value, lanes: Vec<Value>) -> InterpResult<Value> {
    match template {
        Value::Tuple(_) => Ok(Value::Tuple(lanes)),
        Value::Enum { variant, fields } if fields.len() == 1 => Ok(Value::Enum {
            variant: *variant,
            fields: vec![simd_wrap(&fields[0], lanes)?],
        }),
        _ => Err(InterpError::type_mismatch(
            "simd operation",
            "vector value",
            template.kind_name(),
        )),
    }
}

/// Whether a mask lane selects its element.
fn simd_lane_active(lane: &Value) -> InterpResult<bool> {
    match lane {
        Value::Int(v) => Ok(*v != 0),
        Value::Uint(v) => Ok(*v != 0),
        Value::Bool(b) => Ok(*b),
        Value::Float(f) => Ok(*f != 0.0),
        _ => Err(InterpError::type_mismatch(
            "simd mask lane",
            "integer or boolean lane",
            lane.kind_name(),
        )),
    }
}

/// A lane index argument (`u32` in the guest).
fn simd_index(value: &Value) -> InterpResult<usize> {
    let raw = as_u128(value, "simd lane index")?;
    usize::try_from(raw).map_err(|_| InterpError::Panic {
        message: format!("simd lane index {raw} out of range"),
        payload: None,
    })
}

/// Round a float result to a SIMD lane's width.
fn simd_round_float(x: f64, float_bits: Option<u32>) -> f64 {
    match float_bits {
        Some(16) => round_to_f16(x),
        Some(32) => (x as f32) as f64,
        _ => x,
    }
}

/// IEEE 754 maximumNumber: a NaN operand yields the other
/// operand, and +0 is greater than -0.
fn maximum_number(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a == b {
        if a.is_sign_positive() {
            a
        } else {
            b
        }
    } else {
        a.max(b)
    }
}

/// IEEE 754 minimumNumber: a NaN operand yields the other
/// operand, and -0 is less than +0.
fn minimum_number(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else {
        a.min(b)
    }
}

/// One element-wise binary SIMD operation on two lanes of the
/// shape `info` = (bits, signed, float width).
fn simd_binop(
    op: SimdOp,
    a: &Value,
    b: &Value,
    info: (u32, bool, Option<u32>),
) -> InterpResult<Value> {
    use SimdOp as S;
    let (bits, signed, float_bits) = info;
    if float_bits.is_some() {
        let x = simd_as_f64(a, "simd lane")?;
        let y = simd_as_f64(b, "simd lane")?;
        let r = match op {
            S::Add => x + y,
            S::Sub => x - y,
            S::Mul => x * y,
            S::Div => x / y,
            S::Rem => x % y,
            S::MaximumNumber => maximum_number(x, y),
            S::MinimumNumber => minimum_number(x, y),
            _ => {
                return Err(unsupported(
                    "this simd operation on float lanes is not runtime-supported",
                ))
            }
        };
        return Ok(Value::Float(simd_round_float(r, float_bits)));
    }
    if matches!(op, S::SaturatingAdd | S::SaturatingSub) {
        let mask = bit_mask(bits);
        let pa = as_u128(a, "simd lane")? & mask;
        let pb = as_u128(b, "simd lane")? & mask;
        let pattern = if signed {
            let (sa, sb) = (sign_extend(pa, bits), sign_extend(pb, bits));
            let max = sign_extend(mask >> 1, bits);
            let min = -max - 1;
            let r = if op == S::SaturatingAdd {
                sa.saturating_add(sb)
            } else {
                sa.saturating_sub(sb)
            };
            (r.clamp(min, max) as u128) & mask
        } else {
            let r = if op == S::SaturatingAdd {
                pa.saturating_add(pb)
            } else {
                pa.saturating_sub(pb)
            };
            r.min(mask)
        };
        return Ok(wrap_simd_int(pattern, bits, signed));
    }
    let mask = bit_mask(bits);
    let pa = as_u128(a, "simd lane")? & mask;
    let pb = as_u128(b, "simd lane")? & mask;
    let pattern = match op {
        S::Add => pa.wrapping_add(pb),
        S::Sub => pa.wrapping_sub(pb),
        S::Mul => pa.wrapping_mul(pb),
        S::Div | S::Rem => {
            if pb == 0 {
                return Err(InterpError::Panic {
                    message: "attempt to divide by zero in a SIMD lane".to_string(),
                    payload: None,
                });
            }
            if signed {
                let sa = sign_extend(pa, bits);
                let sb = sign_extend(pb, bits);
                if op == S::Div {
                    sa.checked_div(sb).map(|v| v as u128).unwrap_or(pa)
                } else {
                    sa.checked_rem(sb).map(|v| v as u128).unwrap_or(0)
                }
            } else if op == S::Div {
                pa / pb
            } else {
                pa % pb
            }
        }
        S::And => pa & pb,
        S::Or => pa | pb,
        S::Xor => pa ^ pb,
        S::Shl => pa << ((pb % u128::from(bits)) as u32),
        S::Shr => {
            let s = (pb % u128::from(bits)) as u32;
            if signed {
                (sign_extend(pa, bits) >> s) as u128
            } else {
                pa >> s
            }
        }
        S::CarrylessMul => {
            let mut result: u128 = 0;
            let mut aa = pa;
            let mut bb = pb;
            while bb != 0 {
                if bb & 1 == 1 {
                    result ^= aa;
                }
                aa = aa.wrapping_shl(1);
                bb >>= 1;
            }
            result
        }
        S::FunnelShl => {
            let s = (pb % u128::from(bits)) as u32;
            if s == 0 {
                pa
            } else {
                (pa << s) | (pb >> (bits - s))
            }
        }
        S::FunnelShr => {
            let s = (pb % u128::from(bits)) as u32;
            if s == 0 {
                pb
            } else {
                (pa << (bits - s)) | (pb >> s)
            }
        }
        _ => {
            return Err(unsupported(
                "this simd operation on integer lanes is not runtime-supported",
            ))
        }
    };
    Ok(wrap_simd_int(pattern, bits, signed))
}

/// Wrap a bit pattern to a SIMD integer lane of `bits` width.
fn wrap_simd_int(pattern: u128, bits: u32, signed: bool) -> Value {
    let pattern = pattern & bit_mask(bits);
    if signed {
        Value::Int(sign_extend(pattern, bits))
    } else {
        Value::Uint(pattern)
    }
}

/// One SIMD comparison of two lanes.
fn simd_compare(
    op: SimdOp,
    a: &Value,
    b: &Value,
    info: (u32, bool, Option<u32>),
) -> InterpResult<bool> {
    use SimdOp as S;
    let (bits, signed, float_bits) = info;
    let ordering = if float_bits.is_some() {
        let x = simd_as_f64(a, "simd lane")?;
        let y = simd_as_f64(b, "simd lane")?;
        return Ok(match op {
            S::Eq => x == y,
            S::Ne => x != y,
            S::Lt => x < y,
            S::Le => x <= y,
            S::Gt => x > y,
            S::Ge => x >= y,
            _ => unreachable!(),
        });
    } else if signed {
        sign_extend(as_u128(a, "simd lane")? & bit_mask(bits), bits)
            .cmp(&sign_extend(as_u128(b, "simd lane")? & bit_mask(bits), bits))
    } else {
        (as_u128(a, "simd lane")? & bit_mask(bits))
            .cmp(&(as_u128(b, "simd lane")? & bit_mask(bits)))
    };
    Ok(match op {
        S::Eq => ordering == std::cmp::Ordering::Equal,
        S::Ne => ordering != std::cmp::Ordering::Equal,
        S::Lt => ordering == std::cmp::Ordering::Less,
        S::Le => ordering != std::cmp::Ordering::Greater,
        S::Gt => ordering == std::cmp::Ordering::Greater,
        S::Ge => ordering != std::cmp::Ordering::Less,
        _ => unreachable!(),
    })
}

/// One element-wise unary SIMD operation on a lane.
fn simd_unop(op: SimdOp, a: &Value, info: (u32, bool, Option<u32>)) -> InterpResult<Value> {
    use SimdOp as S;
    let (bits, signed, float_bits) = info;
    if float_bits.is_some() {
        let x = simd_as_f64(a, "simd lane")?;
        let r = match op {
            S::Neg => -x,
            S::Fabs => x.abs(),
            S::Ceil => x.ceil(),
            S::Floor => x.floor(),
            S::Trunc => x.trunc(),
            S::Round => x.round(),
            S::RoundTiesEven => x.round_ties_even(),
            S::Fsqrt => x.sqrt(),
            S::Fexp => x.exp(),
            S::Fexp2 => x.exp2(),
            S::Flog => x.ln(),
            S::Flog2 => x.log2(),
            S::Flog10 => x.log10(),
            S::Fsin => x.sin(),
            S::Fcos => x.cos(),
            _ => {
                return Err(unsupported(
                    "this simd operation on float lanes is not runtime-supported",
                ))
            }
        };
        return Ok(Value::Float(simd_round_float(r, float_bits)));
    }
    let mask = bit_mask(bits);
    let p = as_u128(a, "simd lane")? & mask;
    let pattern = match op {
        S::Neg => 0u128.wrapping_sub(p),
        S::Bswap => p.swap_bytes() >> (128 - bits),
        S::BitReverse => p.reverse_bits() >> (128 - bits),
        S::Ctlz => u128::from(p.leading_zeros() - (128 - bits)),
        S::Ctpop => u128::from(p.count_ones()),
        S::Cttz => u128::from(if p == 0 { bits } else { p.trailing_zeros() }),
        _ => {
            return Err(unsupported(
                "this simd operation on integer lanes is not runtime-supported",
            ))
        }
    };
    Ok(wrap_simd_int(pattern, bits, signed))
}

/// One lane of a `simd_cast` / `simd_as`: the numeric cast to
/// the element type `dst_elem`, with `as` semantics.
fn simd_cast_lane(
    lane: &Value,
    src_info: (u32, bool, Option<u32>),
    dst_elem: &str,
) -> InterpResult<Value> {
    let (src_bits, src_signed, src_float) = src_info;
    if let Some((dst_bits, dst_signed)) = int_tag(dst_elem) {
        return Ok(match lane {
            Value::Float(f) => {
                if dst_signed {
                    Value::Int(float_to_int(*f, dst_bits))
                } else {
                    Value::Uint(float_to_uint(*f, dst_bits))
                }
            }
            Value::Bool(b) => {
                if dst_signed {
                    Value::Int(i128::from(*b))
                } else {
                    Value::Uint(u128::from(*b))
                }
            }
            _ => {
                let masked = as_u128(lane, "simd lane")? & bit_mask(src_bits);
                // A signed source sign-extends from its own width
                // before the value is wrapped to the destination
                // width (the extension the float destination below
                // applies to an integer lane).
                let pattern = if src_signed {
                    sign_extend(masked, src_bits) as u128
                } else {
                    masked
                };
                wrap_simd_int(pattern, dst_bits, dst_signed)
            }
        });
    }
    match dst_elem {
        "f16" | "f32" | "f64" => {
            let dst_float = match dst_elem {
                "f16" => Some(16),
                "f32" => Some(32),
                _ => Some(64),
            };
            let x = match lane {
                Value::Float(f) => *f,
                Value::Bool(b) => f64::from(*b),
                Value::Int(_) => {
                    sign_extend(as_u128(lane, "simd lane")? & bit_mask(src_bits), src_bits) as f64
                }
                Value::Uint(_) => (as_u128(lane, "simd lane")? & bit_mask(src_bits)) as f64,
                _ => {
                    return Err(InterpError::type_mismatch(
                        "simd cast lane",
                        "numeric lane",
                        lane.kind_name(),
                    ))
                }
            };
            let _ = src_float;
            Ok(Value::Float(simd_round_float(x, dst_float)))
        }
        "bool" => Ok(Value::Bool(simd_lane_active(lane)?)),
        _ => Err(unsupported(format!(
            "simd cast to element type {dst_elem} is not runtime-supported"
        ))),
    }
}

/// A SIMD reduction over `lanes`; `init` is the extra operand
/// of the ordered add/mul forms.
fn simd_reduce(
    op: SimdOp,
    lanes: &[Value],
    init: Option<&Value>,
    info: (u32, bool, Option<u32>),
) -> InterpResult<Value> {
    use SimdOp as S;
    let (bits, signed, float_bits) = info;
    match op {
        S::ReduceAddOrdered | S::ReduceAddUnordered | S::ReduceMulOrdered
        | S::ReduceMulUnordered => {
            let add = matches!(op, S::ReduceAddOrdered | S::ReduceAddUnordered);
            let binop = if add { S::Add } else { S::Mul };
            let mut acc = match init {
                Some(v) => v.clone(),
                None => {
                    if float_bits.is_some() {
                        Value::Float(if add { 0.0 } else { 1.0 })
                    } else if add {
                        wrap_simd_int(0, bits, signed)
                    } else {
                        wrap_simd_int(1, bits, signed)
                    }
                }
            };
            for lane in lanes {
                acc = simd_binop(binop, &acc, lane, info)?;
            }
            Ok(acc)
        }
        S::ReduceMax | S::ReduceMin => {
            let mut iter = lanes.iter();
            let mut acc = iter
                .next()
                .cloned()
                .ok_or_else(|| unsupported("simd reduction of an empty vector"))?;
            for lane in iter {
                let take = if float_bits.is_some() {
                    let (x, y) = (simd_as_f64(&acc, "simd lane")?, simd_as_f64(lane, "simd lane")?);
                    if x.is_nan() || y.is_nan() {
                        acc = Value::Float(f64::NAN);
                        continue;
                    }
                    if op == S::ReduceMax {
                        y > x
                    } else {
                        y < x
                    }
                } else {
                    let ord = if signed {
                        sign_extend(as_u128(lane, "simd lane")? & bit_mask(bits), bits).cmp(
                            &sign_extend(as_u128(&acc, "simd lane")? & bit_mask(bits), bits),
                        )
                    } else {
                        (as_u128(lane, "simd lane")? & bit_mask(bits))
                            .cmp(&(as_u128(&acc, "simd lane")? & bit_mask(bits)))
                    };
                    if op == S::ReduceMax {
                        ord == std::cmp::Ordering::Greater
                    } else {
                        ord == std::cmp::Ordering::Less
                    }
                };
                if take {
                    acc = lane.clone();
                }
            }
            Ok(acc)
        }
        S::ReduceAnd | S::ReduceOr | S::ReduceXor => {
            let mask = bit_mask(bits);
            let mut acc: u128 = if op == S::ReduceAnd { mask } else { 0 };
            for lane in lanes {
                let p = as_u128(lane, "simd lane")? & mask;
                match op {
                    S::ReduceAnd => acc &= p,
                    S::ReduceOr => acc |= p,
                    _ => acc ^= p,
                }
            }
            Ok(wrap_simd_int(acc, bits, signed))
        }
        _ => Err(unsupported("this simd reduction is not runtime-supported")),
    }
}

/// Registration fragment of the builtin name registry: the SIMD intrinsics.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn simd_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::intrinsics::simd::simd_add" | "core::intrinsics::simd::simd_add" => {
            Builtin::SimdOp(SimdOp::Add)
        }
        "std::intrinsics::simd::simd_sub" | "core::intrinsics::simd::simd_sub" => {
            Builtin::SimdOp(SimdOp::Sub)
        }
        "std::intrinsics::simd::simd_mul" | "core::intrinsics::simd::simd_mul" => {
            Builtin::SimdOp(SimdOp::Mul)
        }
        "std::intrinsics::simd::simd_div" | "core::intrinsics::simd::simd_div" => {
            Builtin::SimdOp(SimdOp::Div)
        }
        "std::intrinsics::simd::simd_rem" | "core::intrinsics::simd::simd_rem" => {
            Builtin::SimdOp(SimdOp::Rem)
        }
        "std::intrinsics::simd::simd_and" | "core::intrinsics::simd::simd_and" => {
            Builtin::SimdOp(SimdOp::And)
        }
        "std::intrinsics::simd::simd_or" | "core::intrinsics::simd::simd_or" => {
            Builtin::SimdOp(SimdOp::Or)
        }
        "std::intrinsics::simd::simd_xor" | "core::intrinsics::simd::simd_xor" => {
            Builtin::SimdOp(SimdOp::Xor)
        }
        "std::intrinsics::simd::simd_shl" | "core::intrinsics::simd::simd_shl" => {
            Builtin::SimdOp(SimdOp::Shl)
        }
        "std::intrinsics::simd::simd_shr" | "core::intrinsics::simd::simd_shr" => {
            Builtin::SimdOp(SimdOp::Shr)
        }
        "std::intrinsics::simd::simd_eq" | "core::intrinsics::simd::simd_eq" => {
            Builtin::SimdOp(SimdOp::Eq)
        }
        "std::intrinsics::simd::simd_ne" | "core::intrinsics::simd::simd_ne" => {
            Builtin::SimdOp(SimdOp::Ne)
        }
        "std::intrinsics::simd::simd_lt" | "core::intrinsics::simd::simd_lt" => {
            Builtin::SimdOp(SimdOp::Lt)
        }
        "std::intrinsics::simd::simd_le" | "core::intrinsics::simd::simd_le" => {
            Builtin::SimdOp(SimdOp::Le)
        }
        "std::intrinsics::simd::simd_gt" | "core::intrinsics::simd::simd_gt" => {
            Builtin::SimdOp(SimdOp::Gt)
        }
        "std::intrinsics::simd::simd_ge" | "core::intrinsics::simd::simd_ge" => {
            Builtin::SimdOp(SimdOp::Ge)
        }
        "std::intrinsics::simd::simd_neg" | "core::intrinsics::simd::simd_neg" => {
            Builtin::SimdOp(SimdOp::Neg)
        }
        "std::intrinsics::simd::simd_fabs" | "core::intrinsics::simd::simd_fabs" => {
            Builtin::SimdOp(SimdOp::Fabs)
        }
        "std::intrinsics::simd::simd_ceil" | "core::intrinsics::simd::simd_ceil" => {
            Builtin::SimdOp(SimdOp::Ceil)
        }
        "std::intrinsics::simd::simd_floor" | "core::intrinsics::simd::simd_floor" => {
            Builtin::SimdOp(SimdOp::Floor)
        }
        "std::intrinsics::simd::simd_trunc" | "core::intrinsics::simd::simd_trunc" => {
            Builtin::SimdOp(SimdOp::Trunc)
        }
        "std::intrinsics::simd::simd_round" | "core::intrinsics::simd::simd_round" => {
            Builtin::SimdOp(SimdOp::Round)
        }
        "std::intrinsics::simd::simd_round_ties_even"
        | "core::intrinsics::simd::simd_round_ties_even" => {
            Builtin::SimdOp(SimdOp::RoundTiesEven)
        }
        "std::intrinsics::simd::simd_fsqrt" | "core::intrinsics::simd::simd_fsqrt" => {
            Builtin::SimdOp(SimdOp::Fsqrt)
        }
        "std::intrinsics::simd::simd_fexp" | "core::intrinsics::simd::simd_fexp" => {
            Builtin::SimdOp(SimdOp::Fexp)
        }
        "std::intrinsics::simd::simd_fexp2" | "core::intrinsics::simd::simd_fexp2" => {
            Builtin::SimdOp(SimdOp::Fexp2)
        }
        "std::intrinsics::simd::simd_flog" | "core::intrinsics::simd::simd_flog" => {
            Builtin::SimdOp(SimdOp::Flog)
        }
        "std::intrinsics::simd::simd_flog2" | "core::intrinsics::simd::simd_flog2" => {
            Builtin::SimdOp(SimdOp::Flog2)
        }
        "std::intrinsics::simd::simd_flog10" | "core::intrinsics::simd::simd_flog10" => {
            Builtin::SimdOp(SimdOp::Flog10)
        }
        "std::intrinsics::simd::simd_fsin" | "core::intrinsics::simd::simd_fsin" => {
            Builtin::SimdOp(SimdOp::Fsin)
        }
        "std::intrinsics::simd::simd_fcos" | "core::intrinsics::simd::simd_fcos" => {
            Builtin::SimdOp(SimdOp::Fcos)
        }
        "std::intrinsics::simd::simd_bswap" | "core::intrinsics::simd::simd_bswap" => {
            Builtin::SimdOp(SimdOp::Bswap)
        }
        "std::intrinsics::simd::simd_bitreverse" | "core::intrinsics::simd::simd_bitreverse" => {
            Builtin::SimdOp(SimdOp::BitReverse)
        }
        "std::intrinsics::simd::simd_ctlz" | "core::intrinsics::simd::simd_ctlz" => {
            Builtin::SimdOp(SimdOp::Ctlz)
        }
        "std::intrinsics::simd::simd_ctpop" | "core::intrinsics::simd::simd_ctpop" => {
            Builtin::SimdOp(SimdOp::Ctpop)
        }
        "std::intrinsics::simd::simd_cttz" | "core::intrinsics::simd::simd_cttz" => {
            Builtin::SimdOp(SimdOp::Cttz)
        }
        "std::intrinsics::simd::simd_saturating_add"
        | "core::intrinsics::simd::simd_saturating_add" => {
            Builtin::SimdOp(SimdOp::SaturatingAdd)
        }
        "std::intrinsics::simd::simd_saturating_sub"
        | "core::intrinsics::simd::simd_saturating_sub" => {
            Builtin::SimdOp(SimdOp::SaturatingSub)
        }
        "std::intrinsics::simd::simd_carryless_mul"
        | "core::intrinsics::simd::simd_carryless_mul" => Builtin::SimdOp(SimdOp::CarrylessMul),
        "std::intrinsics::simd::simd_funnel_shl" | "core::intrinsics::simd::simd_funnel_shl" => {
            Builtin::SimdOp(SimdOp::FunnelShl)
        }
        "std::intrinsics::simd::simd_funnel_shr" | "core::intrinsics::simd::simd_funnel_shr" => {
            Builtin::SimdOp(SimdOp::FunnelShr)
        }
        "std::intrinsics::simd::simd_maximum_number_nsz"
        | "core::intrinsics::simd::simd_maximum_number_nsz" => {
            Builtin::SimdOp(SimdOp::MaximumNumber)
        }
        "std::intrinsics::simd::simd_minimum_number_nsz"
        | "core::intrinsics::simd::simd_minimum_number_nsz" => {
            Builtin::SimdOp(SimdOp::MinimumNumber)
        }
        "std::intrinsics::simd::simd_splat" | "core::intrinsics::simd::simd_splat" => {
            Builtin::SimdOp(SimdOp::Splat)
        }
        "std::intrinsics::simd::simd_extract" | "core::intrinsics::simd::simd_extract" => {
            Builtin::SimdOp(SimdOp::Extract)
        }
        "std::intrinsics::simd::simd_extract_dyn" | "core::intrinsics::simd::simd_extract_dyn" => {
            Builtin::SimdOp(SimdOp::Extract)
        }
        "std::intrinsics::simd::simd_insert" | "core::intrinsics::simd::simd_insert" => {
            Builtin::SimdOp(SimdOp::Insert)
        }
        "std::intrinsics::simd::simd_insert_dyn" | "core::intrinsics::simd::simd_insert_dyn" => {
            Builtin::SimdOp(SimdOp::Insert)
        }
        "std::intrinsics::simd::simd_shuffle" | "core::intrinsics::simd::simd_shuffle" => {
            Builtin::SimdOp(SimdOp::Shuffle)
        }
        "std::intrinsics::simd::simd_cast" | "core::intrinsics::simd::simd_cast" => {
            Builtin::SimdOp(SimdOp::Cast)
        }
        "std::intrinsics::simd::simd_as" | "core::intrinsics::simd::simd_as" => {
            Builtin::SimdOp(SimdOp::Cast)
        }
        "std::intrinsics::simd::simd_bitmask" | "core::intrinsics::simd::simd_bitmask" => {
            Builtin::SimdOp(SimdOp::Bitmask)
        }
        "std::intrinsics::simd::simd_select" | "core::intrinsics::simd::simd_select" => {
            Builtin::SimdOp(SimdOp::Select)
        }
        "std::intrinsics::simd::simd_select_bitmask"
        | "core::intrinsics::simd::simd_select_bitmask" => Builtin::SimdOp(SimdOp::SelectBitmask),
        "std::intrinsics::simd::simd_gather" | "core::intrinsics::simd::simd_gather" => {
            Builtin::SimdOp(SimdOp::Gather)
        }
        "std::intrinsics::simd::simd_scatter" | "core::intrinsics::simd::simd_scatter" => {
            Builtin::SimdOp(SimdOp::Scatter)
        }
        "std::intrinsics::simd::simd_masked_load" | "core::intrinsics::simd::simd_masked_load" => {
            Builtin::SimdOp(SimdOp::MaskedLoad)
        }
        "std::intrinsics::simd::simd_masked_store"
        | "core::intrinsics::simd::simd_masked_store" => Builtin::SimdOp(SimdOp::MaskedStore),
        "std::intrinsics::simd::simd_arith_offset"
        | "core::intrinsics::simd::simd_arith_offset" => Builtin::SimdOp(SimdOp::ArithOffset),
        "std::intrinsics::simd::simd_reduce_add_ordered"
        | "core::intrinsics::simd::simd_reduce_add_ordered" => {
            Builtin::SimdOp(SimdOp::ReduceAddOrdered)
        }
        "std::intrinsics::simd::simd_reduce_add_unordered"
        | "core::intrinsics::simd::simd_reduce_add_unordered" => {
            Builtin::SimdOp(SimdOp::ReduceAddUnordered)
        }
        "std::intrinsics::simd::simd_reduce_mul_ordered"
        | "core::intrinsics::simd::simd_reduce_mul_ordered" => {
            Builtin::SimdOp(SimdOp::ReduceMulOrdered)
        }
        "std::intrinsics::simd::simd_reduce_mul_unordered"
        | "core::intrinsics::simd::simd_reduce_mul_unordered" => {
            Builtin::SimdOp(SimdOp::ReduceMulUnordered)
        }
        "std::intrinsics::simd::simd_reduce_max" | "core::intrinsics::simd::simd_reduce_max" => {
            Builtin::SimdOp(SimdOp::ReduceMax)
        }
        "std::intrinsics::simd::simd_reduce_min" | "core::intrinsics::simd::simd_reduce_min" => {
            Builtin::SimdOp(SimdOp::ReduceMin)
        }
        "std::intrinsics::simd::simd_reduce_and" | "core::intrinsics::simd::simd_reduce_and" => {
            Builtin::SimdOp(SimdOp::ReduceAnd)
        }
        "std::intrinsics::simd::simd_reduce_or" | "core::intrinsics::simd::simd_reduce_or" => {
            Builtin::SimdOp(SimdOp::ReduceOr)
        }
        "std::intrinsics::simd::simd_reduce_xor" | "core::intrinsics::simd::simd_reduce_xor" => {
            Builtin::SimdOp(SimdOp::ReduceXor)
        }
        "std::intrinsics::simd::simd_reduce_all" | "core::intrinsics::simd::simd_reduce_all" => {
            Builtin::SimdOp(SimdOp::ReduceAll)
        }
        "std::intrinsics::simd::simd_reduce_any" | "core::intrinsics::simd::simd_reduce_any" => {
            Builtin::SimdOp(SimdOp::ReduceAny)
        }
        "std::intrinsics::simd::simd_fma" | "core::intrinsics::simd::simd_fma" => {
            Builtin::SimdOp(SimdOp::Fma)
        }
        "std::intrinsics::simd::simd_relaxed_fma" | "core::intrinsics::simd::simd_relaxed_fma" => {
            Builtin::SimdOp(SimdOp::Fma)
        }
        "std::intrinsics::simd::simd_cast_ptr" | "core::intrinsics::simd::simd_cast_ptr" => {
            Builtin::SimdOp(SimdOp::CastPtr)
        }
        "std::intrinsics::simd::simd_expose_provenance"
        | "core::intrinsics::simd::simd_expose_provenance" => {
            Builtin::SimdOp(SimdOp::ExposeProvenance)
        }
        "std::intrinsics::simd::simd_with_exposed_provenance"
        | "core::intrinsics::simd::simd_with_exposed_provenance" => {
            Builtin::SimdOp(SimdOp::WithExposedProvenance)
        }
        _ => return None,
    })
}
