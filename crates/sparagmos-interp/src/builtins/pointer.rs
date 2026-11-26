//! Pointer identity and distance for the pointer builtins.



use crate::error::InterpResult;
use crate::interp::{
    Interpreter, Resolved,
};
use crate::value::{RefProj, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// The signed distance, in elements of the pointee type the
    /// generic arguments name, between two resolved pointer
    /// targets, when both denote positions in the same place,
    /// static or allocation. Positions are compared through their
    /// projections: the interpreter's offset machinery leaves an
    /// element-aligned position as a trailing `Index` step, or —
    /// when the position cannot be resolved structurally — as a
    /// trailing byte displacement, so a trailing step of either
    /// shape is read back as the position's element index.
    /// A projection whose only position step is a trailing field
    /// step is ambiguous -- the field step can be an element
    /// position in a tuple-like base or the last step of the path
    /// to a sequence the field holds (whose base the projection
    /// then denotes) -- so both readings are offered and the one
    /// the two targets share decides. `None` when the two targets
    /// share no reading (their distance is then meaningless in
    /// the guest too).
/// A structural fingerprint of a value (an FNV-1a walk over
/// its shape and scalar leaves), used only as an identity key
/// for pointer-distance purposes: two pointers whose base
/// values share a fingerprint are treated as pointing into the
/// same allocation.
fn value_fingerprint(value: &Value) -> u64 {
    fn mix(h: &mut u64, x: u64) {
        *h ^= x;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
    fn walk(h: &mut u64, value: &Value) {
        match value {
            Value::Unit => mix(h, 1),
            Value::Bool(b) => {
                mix(h, 2);
                mix(h, *b as u64);
            }
            Value::Int(v) => {
                mix(h, 3);
                mix(h, *v as u64);
            }
            Value::Uint(v) => {
                mix(h, 4);
                mix(h, *v as u64);
            }
            Value::Float(v) => {
                mix(h, 5);
                mix(h, v.to_bits());
            }
            Value::Str(s) => {
                mix(h, 6);
                mix(h, s.len() as u64);
                for b in s.as_bytes() {
                    mix(h, *b as u64);
                }
            }
            Value::Bytes(b) => {
                mix(h, 7);
                mix(h, b.len() as u64);
                for byte in b {
                    mix(h, *byte as u64);
                }
            }
            Value::Tuple(items) => {
                mix(h, 8);
                mix(h, items.len() as u64);
                for item in items {
                    walk(h, item);
                }
            }
            Value::Enum { variant, fields } => {
                mix(h, 9);
                mix(h, *variant as u64);
                mix(h, fields.len() as u64);
                for field in fields {
                    walk(h, field);
                }
            }
            Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => {
                mix(h, 11);
                if let Some(addr) = crate::interp::synthetic_address(value) {
                    mix(h, addr as u64);
                    mix(h, (addr >> 64) as u64);
                }
            }
            Value::FnPtr(name) => {
                mix(h, 12);
                for b in name.as_bytes() {
                    mix(h, *b as u64);
                }
            }
            Value::Closure { def, .. } => {
                mix(h, 13);
                for b in def.as_bytes() {
                    mix(h, *b as u64);
                }
            }
            other => {
                // The remaining shapes (uninitialized slots,
                // formatting arguments, SIMD vectors, ...) do
                // not underlie slice pointers in practice; a
                // Debug-shape hash keeps the key deterministic.
                mix(h, 14);
                let text = format!("{other:?}");
                mix(h, text.len() as u64);
                for b in text.as_bytes() {
                    mix(h, *b as u64);
                }
            }
        }
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    walk(&mut h, value);
    h
}

    pub(super) fn pointer_distance(
        &self,
        x: &Resolved,
        y: &Resolved,
        generic_args: &[String],
    ) -> InterpResult<Option<i128>> {
        let tag = generic_args.first().map(String::as_str).unwrap_or("");
        let elem_size = self.layout_of(tag).map(|(size, _)| size);
        // Split a target into its base identity and the element
        // index its projection's element position denotes, if
        // any.
        let split = |target: &Resolved| -> Option<Vec<(String, i128)>> {
            let (base, projection): (String, &[RefProj]) = match target {
                Resolved::Place {
                    frame_pos,
                    local,
                    projection,
                } => (format!("place:{frame_pos}:{local}"), projection),
                Resolved::Static { name, projection } => (format!("static:{name}"), projection),
                Resolved::Alloc { id, projection } => (format!("alloc:{id}"), projection),
                // Two pointers into the same value tree (a
                // cloned copy of one base value -- the shape a
                // slice iterator's start/end pair takes, both
                // derived from one fat reference) are the same
                // allocation for distance purposes: the base
                // value's structural fingerprint is the identity.
                // Pointers into different allocations that happen
                // to hold equal values would be conflated, but a
                // cross-allocation `offset_from` is undefined
                // behaviour in the guest, so no defined program
                // can observe the difference.
                Resolved::Value(base, projection) => {
                    (format!("value:{:016x}", Self::value_fingerprint(base)), projection)
                }
            };
            let key = |steps: &[RefProj]| {
                steps
                    .iter()
                    .map(|p| format!("{p:?}"))
                    .collect::<Vec<_>>()
                    .join("/")
            };
            // A trailing byte displacement is itself the
            // position, in element units of the intrinsic's
            // element type (the reading this split has always
            // used).
            if let Some((RefProj::ByteOffset(d), rest)) = projection.split_last() {
                let size = elem_size.filter(|sz| *sz > 0)? as i128;
                if d % size != 0 {
                    return None;
                }
                // A byte displacement carried past an index
                // step composes with it: the pair denotes the
                // element the displacement lands on, under the
                // same base key the plain index step denotes,
                // so a byte-advanced pointer and the element
                // pointer it lands on measure against each
                // other (and against a plain index pointer).
                if let Some((RefProj::Index(i), rest)) = rest.split_last() {
                    return Some(vec![(
                        format!("{base}|{}", key(rest)),
                        *i as i128 + d / size,
                    )]);
                }
                return Some(vec![(format!("{base}|{}", key(rest)), d / size)]);
            }
            // Otherwise the position is the projection's element
            // step (see `element_position_step`): everything
            // before and after it — the base path and any wrapper
            // entry into the element itself — is the shared
            // identity the two pointers must agree on.
            match element_position_step(projection) {
                Some((at, i)) => {
                    let mut steps: Vec<&RefProj> = projection[..at].iter().collect();
                    steps.extend(projection[at + 1..].iter());
                    let key = steps
                        .iter()
                        .map(|p| format!("{p:?}"))
                        .collect::<Vec<_>>()
                        .join("/");
                    let mut candidates =
                        vec![(format!("{base}|{key}"), i as i128)];
                    // When the element step is the projection's
                    // trailing field step (the fallback
                    // `element_position_step` applies when the
                    // projection carries no index step at all),
                    // the projection is ambiguous: the field step
                    // can equally be the last step of the path to
                    // a sequence the field holds, the whole
                    // projection then denoting that sequence's
                    // base -- element 0, the reading
                    // `array_position` gives a bare field base.
                    // Offer that reading as a second candidate;
                    // which reading the two targets share decides
                    // (a pointer to an element of the sequence and
                    // a pointer to its base must measure against
                    // each other).
                    if matches!(projection[at], RefProj::Field(_)) {
                        let full = projection
                            .iter()
                            .map(|p| format!("{p:?}"))
                            .collect::<Vec<_>>()
                            .join("/");
                        candidates.push((format!("{base}|{full}"), 0));
                    }
                    // Whole-value entry steps after the element
                    // step (a transparent wrapper's field-0
                    // entry, a downcast) enter the element
                    // itself: a pointer formed through the
                    // wrapper and a plain element pointer denote
                    // positions in the same sequence, so offer
                    // the reading whose key drops the entry
                    // steps as well.
                    let suffix = &projection[at + 1..];
                    if !suffix.is_empty()
                        && suffix.iter().all(|step| {
                            matches!(
                                step,
                                RefProj::Field(0) | RefProj::Downcast(_)
                            )
                        })
                    {
                        let prefix_key = projection[..at]
                            .iter()
                            .map(|p| format!("{p:?}"))
                            .collect::<Vec<_>>()
                            .join("/");
                        candidates.push((
                            format!("{base}|{prefix_key}"),
                            i as i128,
                        ));
                    }
                    Some(candidates)
                }
                None => Some(vec![(format!("{base}|{}", key(projection)), 0)]),
            }
        };
        Ok(match (split(x), split(y)) {
            (Some(xs), Some(ys)) => xs.iter().find_map(|(bx, ix)| {
                ys.iter()
                    .find(|(by, _)| by == bx)
                    .map(|(_, iy)| ix - iy)
            }),
            _ => None,
        })
    }
}
