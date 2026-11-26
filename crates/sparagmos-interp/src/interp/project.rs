//! The projectors and union/shaped access.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};
use std::borrow::Cow;

/// How a write through one field of a union composes with a
/// value another field of the same union currently holds (see
/// `Interpreter::union_crossing_write`).
enum UnionCross {
    /// The held field's span fits inside the written field's
    /// span: the union keeps its aggregate form, now holding
    /// the written field, whose pre-write value the read side
    /// decodes from the held field's bits.
    Switch,
    /// The held field is wider than the written field, or the
    /// union's storage is already in byte form: the union's
    /// storage continues as bytes (`Value::UnionBytes`) — the
    /// held field's byte image, with the written field's image
    /// spliced over the prefix it covers — so the bytes past
    /// the written field survive for later reads.
    Bytes { source: Vec<Option<u8>> },
}

impl<'a> Interpreter<'a> {
    /// Project `value` through `projection`, resolving union
    /// field steps against the program's union tables: the plain
    /// projector ([`project_value`]) is type-free, so runs of
    /// ordinary steps go through it and each union step is
    /// applied here, where the scalar shapes are in reach.
    pub(crate) fn project_value_t(
        &self,
        value: Value,
        projection: &[RefProj],
        func: &str,
        place: &str,
        root_ty: Option<Ty>,
    ) -> InterpResult<Value> {
        let mut value = value;
        let mut cur_ty = root_ty;
        let mut i = 0;
        while i < projection.len() {
            if let RefProj::UnionField { idx, table } = projection[i] {
                value = self.union_field_read(value, idx, table, func, place)?;
                cur_ty = stepped_field_ty(Some(self.program()), cur_ty.as_ref(), 0, idx);
                i += 1;
            } else {
                let start = i;
                while i < projection.len()
                    && !matches!(projection[i], RefProj::UnionField { .. })
                {
                    i += 1;
                }
                let segment = &projection[start..i];
                value = project_value_ev(
                    Some(self.program()),
                    cur_ty.clone(),
                    value,
                    segment,
                    func,
                    place,
                )?;
                cur_ty = step_ty_across(Some(self.program()), cur_ty, segment);
            }
        }
        Ok(value)
    }

    /// Project `value` through `projection` by reference: the
    /// borrowing form of [`Self::project_value_t`]. The steps
    /// are exactly those of `project_value_t` -- runs of
    /// ordinary steps go through [`project_value_ref_ev`], the
    /// borrowing form of the segment projector, and each union
    /// step is applied by [`Self::union_field_read`] -- so the
    /// result and the errors are the owning projection's. The
    /// difference is cost only: the owning form's caller must
    /// hand it a clone of the base value, which for a base that
    /// is a whole allocation (a hash table's bucket array, say)
    /// copies every element on every read, while this form
    /// lends the base and clones only the projected result and
    /// the value a union step is applied to (the union instance
    /// in hand at that step, not the base around it).
    pub(crate) fn project_value_ref_t(
        &self,
        value: &Value,
        projection: &[RefProj],
        func: &str,
        place: &str,
        root_ty: Option<Ty>,
    ) -> InterpResult<Value> {
        // The value in hand: a borrow of the input tree, or an
        // owned intermediate (a union step's constructed value,
        // or a step's materialised result) staged in `stage`.
        let mut stage: Option<Value> = None;
        let mut cur: &Value = value;
        let mut cur_ty = root_ty;
        let mut i = 0;
        while i < projection.len() {
            if let RefProj::UnionField { idx, table } = projection[i] {
                let next = self.union_field_read(cur.clone(), idx, table, func, place)?;
                stage = Some(next);
                cur = stage.as_ref().expect("staged union value");
                cur_ty = stepped_field_ty(Some(self.program()), cur_ty.as_ref(), 0, idx);
                i += 1;
            } else {
                let start = i;
                while i < projection.len()
                    && !matches!(projection[i], RefProj::UnionField { .. })
                {
                    i += 1;
                }
                let segment = &projection[start..i];
                let next = project_value_ref_ev(
                    Some(self.program()),
                    cur_ty.clone(),
                    cur,
                    segment,
                    func,
                    place,
                )?;
                cur_ty = step_ty_across(Some(self.program()), cur_ty, segment);
                match next {
                    Cow::Borrowed(v) => cur = v,
                    Cow::Owned(v) => {
                        stage = Some(v);
                        cur = stage.as_ref().expect("staged segment value");
                    }
                }
            }
        }
        Ok(cur.clone())
    }

    /// Read field `idx` of a union `value`. The value holds one
    /// field's value — the last one written; every other slot is
    /// uninitialized (see the union arm of aggregate evaluation
    /// and of `navigate_mut`). Reading the held field returns it
    /// as it stands. Reading any other field reinterprets the
    /// held field's bits as this field's type, little-endian: a
    /// narrower field reads the low bytes, and a wider field
    /// reads zeros above the held field's width (the bytes a
    /// native read would find there are unspecified, so the
    /// model fixes them at zero rather than fabricating data).
    /// A union holding no live field is an uninitialized read;
    /// a field whose type has no scalar shape, and a value
    /// holding several live fields (which the write side never
    /// produces), are explicit errors.
    fn union_field_read(
        &self,
        value: Value,
        idx: usize,
        table: usize,
        func: &str,
        place: &str,
    ) -> InterpResult<Value> {
        // A wholly uninitialized base holds no live field: that
        // is the limiting case of the uninitialized read below,
        // not a type mismatch (a partially initialized place can
        // reach here with its union slot never materialized).
        if value == Value::Uninit {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        }
        // A union in its byte form (see `Value::UnionBytes`):
        // the field's value is decoded from the storage bytes
        // through the field's scalar shape — the same
        // reinterpretation the aggregate form performs through
        // the held field, with the bytes the writes left
        // behind as the source of truth.
        if let Value::UnionBytes { bytes } = &value {
            let shapes = self.program.union_tables.get(table).ok_or_else(|| {
                InterpError::type_mismatch(
                    format!("union field read on {place}"),
                    "a union instance with a scalar-shape table",
                    "a union instance without one",
                )
            })?;
            let dst = shapes.get(idx).and_then(|o| o.as_ref()).ok_or_else(|| {
                InterpError::type_mismatch(
                    format!("union field read on {place}"),
                    "a scalar-shaped union field",
                    "a union field whose type has no scalar shape",
                )
            })?;
            return self.union_bytes_decode(bytes, dst, func, place);
        }
        let Value::Enum { ref fields, .. } = value else {
            return Err(InterpError::type_mismatch(
                format!("union field read on {place}"),
                "union (ADT) value",
                value.kind_name(),
            ));
        };
        if let Some(held) = fields.get(idx) {
            if *held != Value::Uninit {
                return Ok(held.clone());
            }
        }
        let mut live = fields
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != Value::Uninit);
        let Some((src_idx, stored)) = live.next() else {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        };
        if live.next().is_some() {
            return Err(InterpError::type_mismatch(
                format!("union field read on {place}"),
                "a union value holding one field",
                "a union value holding several fields",
            ));
        }
        let stored = stored.clone();
        let shapes = self.program.union_tables.get(table).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("union field read on {place}"),
                "a union instance with a scalar-shape table",
                "a union instance without one",
            )
        })?;
        let src = shapes.get(src_idx).and_then(|o| o.as_ref());
        let dst = shapes.get(idx).and_then(|o| o.as_ref());
        let (Some(src), Some(dst)) = (src, dst) else {
            return Err(InterpError::type_mismatch(
                format!("union field read on {place}"),
                "scalar-shaped union fields on both sides of the read",
                "a union field whose type has no scalar shape",
            ));
        };
        let bits = self.flatten_scalar(&stored, src)?;
        self.construct_scalar(bits, dst)
    }

    /// The byte width a union field's scalar shape spans, when
    /// the width is a whole number of bytes (every shape the
    /// union tables record is: scalar widths are byte-sized by
    /// construction and aggregate widths are layout sizes).
    /// A zero-width shape spans nothing a write could cover.
    fn union_shape_width(shape: &ScalarShape) -> Option<usize> {
        let bits = shape.bits();
        if bits == 0 || bits % 8 != 0 {
            return None;
        }
        Some((bits / 8) as usize)
    }

    /// Splice `src` over `dst` at byte offset `at`, growing
    /// `dst` with uninitialised bytes where the splice reaches
    /// past its current end.
    fn splice_union_bytes(dst: &mut Vec<Option<u8>>, at: usize, src: &[Option<u8>]) {
        if dst.len() < at + src.len() {
            dst.resize(at + src.len(), None);
        }
        dst[at..at + src.len()].copy_from_slice(src);
    }

    /// The storage bytes of a union field's value under its
    /// scalar shape: one entry per byte the shape spans, `None`
    /// for a byte no part of the value initialises. The outer
    /// `None` is a value the byte rendering cannot express
    /// under the shape (a shape past the scalar pattern domain
    /// with no structural decomposition, or a value whose form
    /// the shape does not describe); callers fall back to the
    /// behaviour the value model had before byte storage.
    fn union_field_bytes(&self, value: &Value, shape: &ScalarShape) -> Option<Vec<Option<u8>>> {
        let width = Self::union_shape_width(shape)?;
        if matches!(value, Value::Uninit) {
            return Some(vec![None; width]);
        }
        // An aggregate shape renders structurally first, at
        // any width: the pattern flattening below cannot
        // express a value some part of which is uninitialised
        // (a partially written tuple field of the union), and
        // the byte image must keep exactly those parts
        // uninitialised. A value whose form the shape does not
        // describe falls through to the flattening.
        if let Some(bytes) = self.union_field_bytes_structural(value, shape, width) {
            return Some(bytes);
        }
        if width <= 16 {
            let bits = self.flatten_scalar(value, shape).ok()?;
            return Some((0..width).map(|i| Some((bits >> (8 * i)) as u8)).collect());
        }
        None
    }

    /// The structural byte rendering of an aggregate-shaped
    /// union field's value (see [`Self::union_field_bytes`]):
    /// per-part images spliced at the parts' offsets over a
    /// zero-filled padding image, with an uninitialised part's
    /// span left uninitialised. `None` where the value's form
    /// is not the aggregate the shape describes.
    fn union_field_bytes_structural(
        &self,
        value: &Value,
        shape: &ScalarShape,
        width: usize,
    ) -> Option<Vec<Option<u8>>> {
        match shape {
            ScalarShape::Array(elem, n) => {
                let Value::Tuple(items) = value else {
                    return None;
                };
                if items.len() as u64 != *n {
                    return None;
                }
                let mut out = Vec::with_capacity(width);
                for item in items {
                    out.extend(self.union_field_bytes(item, elem)?);
                }
                Some(out)
            }
            ScalarShape::AdtFields { fields, .. } => {
                let Value::Enum { fields: vals, .. } = value else {
                    return None;
                };
                let mut out = vec![Some(0u8); width];
                for (index, offset, fshape) in fields {
                    let part = self.union_field_bytes(vals.get(*index as usize)?, fshape)?;
                    Self::splice_union_bytes(&mut out, (*offset / 8) as usize, &part);
                }
                Some(out)
            }
            ScalarShape::Fields { fields, .. } => {
                let Value::Tuple(vals) = value else {
                    return None;
                };
                let mut out = vec![Some(0u8); width];
                for (i, (offset, fshape)) in fields.iter().enumerate() {
                    let part = self.union_field_bytes(vals.get(i)?, fshape)?;
                    Self::splice_union_bytes(&mut out, (*offset / 8) as usize, &part);
                }
                Some(out)
            }
            ScalarShape::AdtField { index, inner, .. } => {
                // The payload sits at offset 0 and every other
                // field is zero-sized, so the byte image is the
                // payload's own image at the shape's full width.
                let payload = match value {
                    Value::Enum { fields, .. } => fields.get(*index as usize)?,
                    other => other,
                };
                let mut out = self.union_field_bytes(payload, inner)?;
                out.resize(width, Some(0));
                Some(out)
            }
            _ => None,
        }
    }

    /// One part of a union field's value decoded from storage
    /// bytes (see [`Self::union_bytes_decode`]): `None` where
    /// the bytes cannot be rendered under the shape, and a
    /// wholly uninitialised span decodes to the uninitialised
    /// marker, so a read of that part raises the error at its
    /// own point of use.
    fn union_bytes_part(&self, span: &[Option<u8>], shape: &ScalarShape) -> Option<Value> {
        if span.iter().all(|b| b.is_none()) {
            return Some(Value::Uninit);
        }
        let width = Self::union_shape_width(shape)?;
        debug_assert_eq!(span.len(), width);
        // An aggregate shape decodes structurally first, at
        // any width: the pattern construction below zero-fills
        // a part no write covered, where the part must stay
        // uninitialised so its own read raises the error. A
        // shape the structural decoding cannot render falls
        // through to the construction.
        if let Some(value) = self.union_bytes_part_structural(span, shape) {
            return Some(value);
        }
        if width <= 16 {
            let mut bits: u128 = 0;
            for (i, b) in span.iter().enumerate() {
                if let Some(b) = b {
                    bits |= u128::from(*b) << (8 * i);
                }
            }
            return self.construct_scalar(bits, shape).ok();
        }
        None
    }

    /// The structural decoding of one part of a union field's
    /// value from storage bytes (see [`Self::union_bytes_part`]):
    /// each part decodes from its own span, so a part no write
    /// covered stays uninitialised. `None` where the shape is
    /// not an aggregate the decoding decomposes.
    fn union_bytes_part_structural(&self, span: &[Option<u8>], shape: &ScalarShape) -> Option<Value> {
        match shape {
            ScalarShape::Array(elem, n) => {
                let ew = Self::union_shape_width(elem)?;
                let mut items = Vec::with_capacity(*n as usize);
                for i in 0..*n as usize {
                    items.push(self.union_bytes_part(&span[i * ew..(i + 1) * ew], elem)?);
                }
                Some(Value::Tuple(items))
            }
            ScalarShape::AdtFields {
                count,
                fill,
                fields,
                ..
            } => {
                let mut vals = vec![Value::Uninit; *count];
                for (i, c) in fill {
                    vals[*i as usize] = self.eval_constant(c).ok()?;
                }
                for (index, offset, fshape) in fields {
                    let fw = Self::union_shape_width(fshape)?;
                    let at = (*offset / 8) as usize;
                    vals[*index as usize] = self.union_bytes_part(&span[at..at + fw], fshape)?;
                }
                Some(Value::Enum {
                    variant: 0,
                    fields: vals,
                })
            }
            ScalarShape::Fields { fields, .. } => {
                let mut items = Vec::with_capacity(fields.len());
                for (offset, fshape) in fields {
                    let fw = Self::union_shape_width(fshape)?;
                    let at = (*offset / 8) as usize;
                    items.push(self.union_bytes_part(&span[at..at + fw], fshape)?);
                }
                Some(Value::Tuple(items))
            }
            ScalarShape::AdtField {
                index,
                count,
                fill,
                inner,
            } => {
                let mut vals = vec![Value::Uninit; *count];
                for (i, c) in fill {
                    vals[*i as usize] = self.eval_constant(c).ok()?;
                }
                vals[*index as usize] = self.union_bytes_part(span, inner)?;
                Some(Value::Enum {
                    variant: 0,
                    fields: vals,
                })
            }
            _ => None,
        }
    }

    /// Decode a union field's value from the union's storage
    /// bytes (the read side of `Value::UnionBytes`): the bytes
    /// the field's span covers, reinterpreted through the
    /// field's scalar shape. A span no write has covered is an
    /// uninitialised read; a partially covered span reads its
    /// uncovered bytes as zero, the convention the aggregate
    /// form's cross-field read already keeps for the bytes
    /// past the held field's width.
    fn union_bytes_decode(
        &self,
        bytes: &[Option<u8>],
        shape: &ScalarShape,
        func: &str,
        place: &str,
    ) -> InterpResult<Value> {
        let width = Self::union_shape_width(shape).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("union field read on {place}"),
                "a byte-wide union field shape",
                "a union field shape with no byte width",
            )
        })?;
        let mut span: Vec<Option<u8>> = bytes.iter().take(width).cloned().collect();
        span.resize(width, None);
        if span.iter().all(|b| b.is_none()) {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        }
        self.union_bytes_part(&span, shape).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("union field read on {place}"),
                "a union field value its bytes decode to",
                "storage bytes no value of the field's shape matches",
            )
        })
    }

    /// Whether a write through union field `idx` (of the union
    /// instance with scalar-shape table `table`) into the union
    /// value `current` crosses fields, and how the write
    /// composes with what the union holds (see
    /// [`Self::union_crossing_write`]). `None` when the write
    /// stays inside the held field, the union holds nothing, or
    /// a field the composition needs has no scalar shape.
    fn union_cross_plan(&self, current: &Value, idx: usize, table: usize) -> Option<UnionCross> {
        let shapes = self.program.union_tables.get(table)?;
        let dst = shapes.get(idx).and_then(|o| o.as_ref())?;
        let dst_width = Self::union_shape_width(dst)?;
        match current {
            Value::UnionBytes { bytes } => Some(UnionCross::Bytes {
                source: bytes.clone(),
            }),
            Value::Enum { fields, .. } => {
                // The widest field other than the written one
                // that holds a value: its span is what the
                // write must compose with. (A union value built
                // by guest writes holds at most one field; a
                // materialised constant can hold several, all
                // read from the same bytes, so the widest one's
                // image covers the others' spans.)
                let mut source: Option<(&Value, &ScalarShape, usize)> = None;
                for (i, v) in fields.iter().enumerate() {
                    if i == idx || *v == Value::Uninit {
                        continue;
                    }
                    let Some(shape) = shapes.get(i).and_then(|o| o.as_ref()) else {
                        continue;
                    };
                    let Some(w) = Self::union_shape_width(shape) else {
                        continue;
                    };
                    if source.is_none_or(|(_, _, sw)| w > sw) {
                        source = Some((v, shape, w));
                    }
                }
                let (held, held_shape, held_width) = source?;
                if held_width <= dst_width {
                    Some(UnionCross::Switch)
                } else {
                    Some(UnionCross::Bytes {
                        source: self.union_field_bytes(held, held_shape)?,
                    })
                }
            }
            _ => None,
        }
    }

    /// The aggregate union value holding field `idx`'s value
    /// and no other field: the form a cross-field write leaves
    /// when the written field's span covers the held field's.
    /// The field vector keeps the length the current value's
    /// form already accounts for.
    fn union_holding(idx: usize, temp: Value, current: &Value) -> Value {
        let len = match current {
            Value::Enum { fields, .. } => fields.len().max(idx + 1),
            _ => idx + 1,
        };
        let mut fields = vec![Value::Uninit; len];
        fields[idx] = temp;
        Value::Enum {
            variant: 0,
            fields,
        }
    }

    /// Store `value` at the place `head` (a projection prefix)
    /// of `target` denotes, through the same projected
    /// assignment the write path performs for the target kind.
    fn store_union_at_head(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        head: &[RefProj],
        value: Value,
    ) -> InterpResult<()> {
        match target {
            Resolved::Value(..) => Err(InterpError::type_mismatch(
                "assignment",
                "mutable place",
                "constant aggregate",
            )),
            Resolved::Static { name, .. } => {
                let mut store = self.statics.borrow_mut();
                let slot = store
                    .get_mut(name)
                    .ok_or_else(|| self.missing_static(name))?;
                let place_desc = format!("static {name}");
                let root_ty = self.program().static_tys.get(name).cloned();
                assign_projected(
                    slot,
                    head,
                    value,
                    "<static>",
                    &place_desc,
                    false,
                    Some(self.program()),
                    root_ty,
                )
            }
            Resolved::Alloc { id, .. } => {
                let head = self.elem_block_projection(*id, head);
                let mut store = self.allocs.borrow_mut();
                let slot = store
                    .get_mut(id)
                    .ok_or_else(|| self.missing_alloc(*id))?;
                let place_desc = format!("allocation #{id}");
                let alloc_root = self.alloc_ty(*id);
                let alloc_root = match alloc_root {
                    Some(element_ty) if self.is_elem_block(*id) => {
                        Some(Ty::Slice(Box::new(element_ty)))
                    }
                    other => other,
                };
                grow_tuples_along(slot, &head, Some(self.program()), alloc_root.clone());
                assign_projected(
                    slot,
                    &head,
                    value,
                    "<allocation>",
                    &place_desc,
                    true,
                    Some(self.program()),
                    alloc_root,
                )
            }
            Resolved::Place {
                frame_pos, local, ..
            } => {
                let func = frames[*frame_pos].func.clone();
                let root_ty = self
                    .program()
                    .get(&func)
                    .and_then(|body| body.local_ty(Local(*local)))
                    .cloned();
                let frame = &mut frames[*frame_pos];
                let place_desc = describe_resolved(*local, head);
                let slot = frame.locals.get_mut(*local).ok_or_else(|| InterpError::UnknownLocal {
                    func: func.clone(),
                    local: *local,
                })?;
                assign_projected(
                    slot,
                    head,
                    value,
                    &func,
                    &place_desc,
                    false,
                    Some(self.program()),
                    root_ty,
                )
            }
        }
    }

    /// A write or in-place modification whose projection
    /// crosses a union field boundary: the projection steps
    /// into field `idx` of a union whose stored value another
    /// field's write produced (or whose storage is already in
    /// byte form). Plain navigation cannot compose such a
    /// write — the value model holds one field's value per
    /// union, so a partial write into a different field has
    /// nowhere to land — so the write is intercepted here, at
    /// the same level the vector length slot's writes are
    /// intercepted: the union's current value is read, the
    /// written field's value is materialised from it (decoded
    /// through the field's scalar shape), `apply` performs the
    /// write inside that field value, and the composed union
    /// value is stored back at the projection's head.
    ///
    /// Composition keeps the aggregate form when the held
    /// field's span fits inside the written field's span (the
    /// surviving bytes are exactly the written field's new
    /// value) and switches the union to its byte form
    /// (`Value::UnionBytes`) when the held field is wider, so
    /// the bytes past the written field survive for later
    /// reads of the held field.
    ///
    /// Returns `None` when the write does not cross fields —
    /// no union step, the addressed field is the held one, the
    /// union holds nothing yet, or the fields involved have no
    /// scalar shapes to compose through — and the caller takes
    /// its ordinary path, whose verdicts are unchanged. A
    /// decline happens only before `apply` runs: once the
    /// write is applied, composition always stores a value.
    /// `vec_len_tail_ok` records whether the caller's `apply`
    /// handles a trailing vector-length step itself (the
    /// projected assignment does; in-place navigation does
    /// not, and the length-slot pre-pass owns that case).
    pub(crate) fn union_crossing_write<R>(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        vec_len_tail_ok: bool,
        apply: impl FnOnce(&mut Value, &[RefProj], &str, &str) -> InterpResult<R>,
    ) -> Option<InterpResult<R>> {
        // The effective projection: an allocation target's
        // projection is canonicalised exactly as the write
        // path canonicalises it, so the head and the rest are
        // the steps the store will navigate.
        let effective;
        let projection: &[RefProj] = match target {
            Resolved::Alloc { id, projection } => {
                effective = self.elem_block_projection(*id, projection);
                &effective
            }
            Resolved::Place { projection, .. } | Resolved::Static { projection, .. } => projection,
            Resolved::Value(..) => return None,
        };
        let pos = projection
            .iter()
            .position(|s| matches!(s, RefProj::UnionField { .. }))?;
        let RefProj::UnionField { idx, table } = projection[pos] else {
            unreachable!("the position names a union field step")
        };
        let head = &projection[..pos];
        let rest = &projection[pos + 1..];
        if !vec_len_tail_ok && matches!(rest.last(), Some(RefProj::VecLen)) {
            return None;
        }
        let (func, place_desc) = match target {
            Resolved::Place {
                frame_pos, local, ..
            } => (
                frames[*frame_pos].func.clone(),
                describe_resolved(*local, projection),
            ),
            Resolved::Static { name, .. } => ("<static>".to_string(), format!("static {name}")),
            Resolved::Alloc { id, .. } => {
                ("<allocation>".to_string(), format!("allocation #{id}"))
            }
            Resolved::Value(..) => unreachable!("value targets declined above"),
        };
        let head_target = match target {
            Resolved::Place {
                frame_pos, local, ..
            } => Resolved::Place {
                frame_pos: *frame_pos,
                local: *local,
                projection: head.to_vec(),
            },
            Resolved::Static { name, .. } => Resolved::Static {
                name: name.clone(),
                projection: head.to_vec(),
            },
            Resolved::Alloc { id, .. } => Resolved::Alloc {
                id: *id,
                projection: head.to_vec(),
            },
            Resolved::Value(..) => unreachable!("value targets declined above"),
        };
        let current = self.read_resolved(frames, &head_target).ok()?;
        let plan = self.union_cross_plan(&current, idx, table)?;
        // The written field's value before the write, decoded
        // from what the union holds; a whole-field write
        // replaces it wholesale and needs no decode. A decode
        // failure declines: the ordinary path produces its own
        // verdict for the same write.
        let mut temp = if rest.is_empty() {
            Value::Uninit
        } else {
            match self.union_field_read(current.clone(), idx, table, &func, &place_desc) {
                Ok(v) => v,
                Err(_) => return None,
            }
        };
        let result = match apply(&mut temp, rest, &func, &place_desc) {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
        };
        let new_union = match &plan {
            UnionCross::Bytes { source } => {
                let shape = self
                    .program
                    .union_tables
                    .get(table)
                    .and_then(|shapes| shapes.get(idx))
                    .and_then(|o| o.as_ref());
                match shape.and_then(|shape| self.union_field_bytes(&temp, shape)) {
                    Some(image) => {
                        let mut bytes = source.clone();
                        Self::splice_union_bytes(&mut bytes, 0, &image);
                        Value::UnionBytes { bytes }
                    }
                    // The written value has no byte image under
                    // its shape: the write is honoured in the
                    // aggregate form holding the written field.
                    // The held field's tail bytes are lost —
                    // still strictly more than the model without
                    // byte storage could do with this write,
                    // which was to error.
                    None => Self::union_holding(idx, temp, &current),
                }
            }
            UnionCross::Switch => Self::union_holding(idx, temp, &current),
        };
        Some(
            self.store_union_at_head(frames, target, head, new_union)
                .map(|()| result),
        )
    }

    /// A scalar `Transmute` reinterpretation: flatten the
    /// operand value to its bit pattern through the source
    /// shape, then construct the target shape's value from the
    /// same bits.
    /// The lane decomposition of a vector shape: the shape of
    /// one lane and the lane count. Where a one-field ADT
    /// wrapper chain bottoms out decides the element form. Over
    /// an array the wrappers spread to the array's lanes. Over
    /// an aggregate record (a struct or tuple payload) the
    /// wrapper is itself the element: readers address the
    /// payload through the wrapper's field step, so the slot
    /// holds the wrapped value whole -- decomposing to the
    /// payload would write it where the wrapper belongs and
    /// lose the layer every other reader addresses. Over
    /// anything else the wrappers are storage-transparent: a
    /// wrapper around a scalar is stored as the scalar itself
    /// (a hash table's slots hold bare control bytes, which
    /// the byte-composing readers require).
    fn shape_lanes(shape: &ScalarShape) -> (ScalarShape, u64) {
        match shape {
            ScalarShape::Array(elem, n) => ((**elem).clone(), *n),
            ScalarShape::AdtField { inner, .. } => {
                let mut cur = shape;
                while let ScalarShape::AdtField { inner, .. } = cur {
                    cur = inner;
                }
                match cur {
                    ScalarShape::Array(elem, n) => ((**elem).clone(), *n),
                    ScalarShape::AdtFields { .. } | ScalarShape::Fields { .. } => {
                        (shape.clone(), 1)
                    }
                    _ => Self::shape_lanes(inner),
                }
            }
            other => (other.clone(), 1),
        }
    }

    /// Flatten a stored element to a lane's bit pattern: the
    /// plain scalar flattening, seeing through one-field wrapper
    /// aggregates (a raw block's elements are the lane types'
    /// own values — a control byte is the wrapper struct around
    /// the byte, not the bare integer).
    fn flatten_lane(&self, value: &Value, lane: &ScalarShape) -> InterpResult<u128> {
        if let Ok(bits) = self.flatten_scalar(value, lane) {
            return Ok(bits);
        }
        if let Value::Enum { fields, .. } = value {
            if fields.len() == 1 {
                return self.flatten_lane(&fields[0], lane);
            }
        }
        self.flatten_scalar(value, lane)
    }

    /// The bit pattern of the vector value a raw element block
    /// holds at `target`: lane `i` is the element at sequence
    /// offset `i`, flattened through the lane shape.
    fn shaped_bits_from_elements(
        &self,
        frames: &[Frame],
        target: &Resolved,
        shape: &ScalarShape,
    ) -> InterpResult<u128> {
        let (lane, count) = Self::shape_lanes(shape);
        let lane_bits = lane.bits();
        // A target whose stored units are bytes (a byte string,
        // a raw block of control bytes) holds the vector as a
        // byte stream: the lanes are packed from consecutive
        // bytes, not read one byte per lane. The first unit
        // decides: a byte-sized scalar (or a one-field wrapper
        // over one, like a control-byte tag) means byte units.
        let first = self.read_seq_at(frames, target, 0)?;
        let byte_like = |v: &Value| -> bool {
            match v {
                Value::Uint(b) => *b <= 0xFF,
                Value::Int(b) => (-128..=255).contains(b),
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    matches!(&fields[0], Value::Uint(b) if *b <= 0xFF)
                        || matches!(&fields[0], Value::Int(b) if (-128..=255).contains(b))
                }
                _ => false,
            }
        };
        if lane_bits > 8 && byte_like(&first) {
            let byte_lane = ScalarShape::Int {
                bits: 8,
                signed: false,
            };
            let nbytes = (shape.bits() / 8) as usize;
            let mut out = 0u128;
            for k in 0..nbytes {
                let unit = if k == 0 {
                    first.clone()
                } else {
                    self.read_seq_at(frames, target, k)?
                };
                let b = self.flatten_lane(&unit, &byte_lane)?;
                out |= (b & 0xFF) << (8 * (k as u32));
            }
            return Ok(out);
        }
        let mut out = 0u128;
        for i in 0..count {
            let element = if i == 0 {
                first.clone()
            } else {
                self.read_seq_at(frames, target, i as usize)?
            };
            let bits = self.flatten_lane(&element, &lane)?;
            out |= bits << ((i as u32) * lane_bits);
        }
        Ok(out)
    }

    /// A shaped whole-pointee load (see [`Rvalue::ShapedLoad`]):
    /// a pointee stored whole loads as itself, canonicalised
    /// through its shape; a pointee in a raw element block whose
    /// stored elements are not the value (a byte buffer read as
    /// a vector) is assembled from the consecutive elements.
    pub(crate) fn shaped_load(
        &self,
        frames: &[Frame],
        target: &Resolved,
        shape: &ScalarShape,
        ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        let elem_block = matches!(target, Resolved::Alloc { id, .. } if self.is_elem_block(*id));
        // A byte-displaced place/static target: the load's
        // bytes are the byte view's to compose (the resolved
        // read below has no byte-level view of the base value's
        // storage).
        if let Some(view) = self.byte_view_shaped_load(frames, target, shape) {
            return view;
        }
        match self.read_resolved_typed(frames, target, ty) {
            Ok(v) => {
                // A `MaybeUninit` load whose storage reads back as
                // the union form (an uninitialized slot, or a union
                // stored whole) is the union value itself.
                if matches!(ty, Some(t) if is_maybe_uninit_ty(t))
                    && matches!(&v, Value::Enum { fields, .. } if fields.len() == 1)
                {
                    return Ok(v);
                }
                if let Ok(bits) = self.flatten_scalar(&v, shape) {
                    return self.construct_scalar(bits, shape);
                }
                // A sequence-like value base (a byte string or an
                // element tuple behind a constant pointer) holds
                // the vector as its consecutive elements, like a
                // raw block does.
                let seq_base = matches!(
                    target,
                    Resolved::Value(base, _)
                        if matches!(base, Value::Bytes(_) | Value::Tuple(_))
                );
                if elem_block || seq_base {
                    let bits = self.shaped_bits_from_elements(frames, target, shape)?;
                    return self.construct_scalar(bits, shape);
                }
                // A scalar load aimed at an aggregate position
                // (a byte pointer at a struct's base): the bytes
                // at the position are the load's value, composed
                // through the byte view.
                if let Some(view) = self.byte_view_compose(frames, target, shape) {
                    return view;
                }
                Ok(v)
            }
            Err(e) => {
                if elem_block && matches!(e, InterpError::UninitializedRead { .. }) {
                    let bits = self.shaped_bits_from_elements(frames, target, shape)?;
                    return self.construct_scalar(bits, shape);
                }
                // A byte string behind field steps: the steps
                // below the dereference select the payload field
                // of a wrapper (a vector type's storage field is
                // field 0 at offset 0), so a chain of field-0
                // steps over a byte-string base denotes the
                // shape's bits at the start of the string — the
                // plain projection cannot enter a byte string,
                // but the shaped load names exactly those bits.
                if let Resolved::Value(base, projection) = target {
                    if let Value::Bytes(bytes) = base {
                        if projection
                            .iter()
                            .all(|p| matches!(p, RefProj::Field(0)))
                            && !projection.is_empty()
                        {
                            let nbytes = (shape.bits() / 8) as usize;
                            if bytes.len() >= nbytes {
                                let mut bits = 0u128;
                                for (k, b) in bytes[..nbytes].iter().enumerate() {
                                    bits |= u128::from(*b) << (8 * (k as u32));
                                }
                                return self.construct_scalar(bits, shape);
                            }
                        }
                    }
                }
                Err(e)
            }
        }
    }

    /// A shaped whole-pointee store (see
    /// [`StatementKind::ShapedStore`]): into a raw element block
    /// the value goes lane by lane over the consecutive elements
    /// at the target, each new lane keeping the wrapper shape of
    /// the element it replaces; anywhere else the store is the
    /// plain whole-value one. The lane reading applies only at
    /// the block's own flat positions: a projection that enters
    /// a structured element (a field, downcast or union-field
    /// step) denotes one stored position inside that element,
    /// not the start of the block's flat sequence, so a store
    /// there is the plain whole-value one — writing the lanes
    /// there would append an element step inside the field and
    /// shred the value it holds.
    pub(crate) fn shaped_store(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: Value,
        shape: &ScalarShape,
        root_ty: Option<&Ty>,
    ) -> InterpResult<()> {
        let elem_block = matches!(target, Resolved::Alloc { id, projection }
            if self.is_elem_block(*id)
                && !projection.iter().any(|step| matches!(
                    step,
                    RefProj::Field(_)
                        | RefProj::Downcast(_)
                        | RefProj::UnionField { .. }
                )));
        if elem_block {
            if let Ok(bits) = self.flatten_scalar(&value, shape) {
                let (lane, count) = Self::shape_lanes(shape);
                let lane_bits = lane.bits();
                let mask = bit_mask(lane_bits);
                for i in 0..count {
                    let lane_value = self.construct_scalar(
                        (bits >> ((i as u32) * lane_bits)) & mask,
                        &lane,
                    )?;
                    let stored = match self.read_seq_at(frames, target, i as usize) {
                        Ok(current) => crate::builtins::rewrap_like(&current, lane_value),
                        Err(_) => lane_value,
                    };
                    self.write_seq_at(frames, target, i as usize, stored)?;
                }
                return Ok(());
            }
        }
        // A byte-displaced place/static target: the store's
        // bytes are spliced through the byte view of the base
        // value's storage (see `shaped_load`).
        if let Some(view) = self.byte_view_shaped_store(frames, target, &value, shape) {
            return view;
        }
        self.write_resolved_typed(frames, target, value, root_ty)
    }

    /// Map the elements a slice / array pointer denotes
    /// through `map`, for the slice element reinterpretation
    /// casts (see `Rvalue::SliceEnumToInt`): the result is a
    /// fresh value target holding a snapshot of the mapped
    /// elements, in the same shape the source pointer had.
    /// The storage the operand denotes is never rewritten
    /// (see `Rvalue::WrapSliceElems`); a pointer whose
    /// pointee is not an element aggregate keeps the identity
    /// value, and element reads through it keep the explicit
    /// type mismatch they already give.
    pub(crate) fn map_slice_pointer_elems(
        &self,
        frames: &[Frame],
        value: &Value,
        map: impl Fn(&Value) -> InterpResult<Value>,
    ) -> InterpResult<Value> {
        match value {
            Value::FatRef { base, len } => match &**base {
                Value::Tuple(items) => {
                    let mapped = items
                        .iter()
                        .map(&map)
                        .collect::<InterpResult<Vec<_>>>()?;
                    Ok(Value::FatRef {
                        base: Box::new(Value::Tuple(mapped)),
                        len: *len,
                    })
                }
                Value::Ref(_) => {
                    let (elems, _) = self.slice_elements(frames, value)?;
                    let mapped = elems
                        .iter()
                        .map(&map)
                        .collect::<InterpResult<Vec<_>>>()?;
                    // The fresh base is the bare element tuple,
                    // not a value target entered at index 0: an
                    // index step over the fat deref composes
                    // into place/static/alloc targets only, and
                    // a value target already standing at an
                    // element position would take a second index
                    // step into the element itself. The bare
                    // tuple denotes the whole sequence, so the
                    // deref denotes the slice storage and index
                    // steps address its elements directly.
                    Ok(Value::FatRef {
                        base: Box::new(Value::Tuple(mapped)),
                        len: *len,
                    })
                }
                _ => Ok(value.clone()),
            },
            Value::Ref(_) => {
                let pointee = self.resolve_ref(frames, value)?;
                match &*pointee {
                    Value::Tuple(items) => {
                        let mapped = items
                            .iter()
                            .map(&map)
                            .collect::<InterpResult<Vec<_>>>()?;
                        Ok(Value::Ref(RefTarget::value(Value::Tuple(mapped))))
                    }
                    _ => Ok(value.clone()),
                }
            }
            _ => Ok(value.clone()),
        }
    }

    pub(crate) fn reinterpret_value(
        &self,
        value: &Value,
        src: &ScalarShape,
        dst: &ScalarShape,
    ) -> InterpResult<Value> {
        // A pointer leaf the two shapes line up on is the same
        // pointer on both sides, not a bit pattern: flattening
        // it to its synthetic address and constructing an
        // address-only pointer back severs it from the
        // allocation it names, and the pointee becomes
        // unreachable. The `assume_init` transmute
        // (`Box<MaybeUninit<T>, A>` to `Box<T, A>`) is the
        // canonical case: the box aggregate is identical on both
        // sides, only the pointee type changes. Where the shapes
        // line up structurally, walk them together and carry
        // pointer leaves across as the values they are; every
        // other shape pair goes through the bit pattern, which
        // is also what a genuine pointer <-> integer transmute
        // means.
        if let Some(v) = self.reinterpret_preserving(value, src, dst) {
            return Ok(v);
        }
        // An array pair of the same length whose elements
        // do not all line up structurally: reinterpret
        // element by element. The whole-value bit path
        // below packs the array into a single `u128`, which
        // caps the total width at 128 bits; the elementwise
        // walk is the same reinterpretation at any length
        // (element `i` occupies the `i`-th lane of the
        // pattern in both readings). A byte block is the
        // one non-tuple array value: its elements are its
        // bytes.
        if let (
            ScalarShape::Array(selem, sn),
            ScalarShape::Array(delem, dn),
        ) = (src, dst)
        {
            if sn == dn {
                let items: Option<Vec<Value>> = match value {
                    Value::Tuple(items) if items.len() as u64 == *sn => {
                        Some(items.clone())
                    }
                    Value::Bytes(b)
                        if selem.bits() == 8 && b.len() as u64 == *sn =>
                    {
                        Some(
                            b.iter()
                                .map(|byte| Value::Uint(u128::from(*byte)))
                                .collect(),
                        )
                    }
                    _ => None,
                };
                if let Some(items) = items {
                    let mut out = Vec::with_capacity(items.len());
                    for item in &items {
                        out.push(self.reinterpret_value(item, selem, delem)?);
                    }
                    return Ok(Value::Tuple(out));
                }
            }
        }
        // A pair of bit-stream shapes (arrays and
        // one-payload wrappers over them) whose leaf
        // streams have the same total width: reinterpret
        // through the stream — the width-crossing
        // regrouping and the long wrapped array, at any
        // total width. The whole-pattern path below packs
        // into a single `u128`, so a stream pair beyond
        // 128 bits must not fall through to it.
        if let Some(r) = self.reinterpret_stream(value, src, dst) {
            return r;
        }
        let bits = self.flatten_scalar(value, src)?;
        self.construct_scalar(bits, dst)
    }

    /// The bit-stream reinterpretation of `value` from
    /// shape `src` to shape `dst`, or `None` when the pair
    /// is not a stream pair this path serves (the caller
    /// falls back to the whole-pattern path). The source
    /// flattens leaf by leaf into a bit stream (little-
    /// endian: earlier leaves occupy the lower bits); an
    /// uninit leaf poisons exactly the stream bits it
    /// occupies, and a target leaf is uninit iff any of
    /// its bits are. Served pairs: pure scalar leaf
    /// streams of equal nonzero total width, beyond 128
    /// bits or with a one-payload wrapper on either side
    /// — the pairs the whole-pattern path cannot flatten
    /// or construct. Every other pair keeps the pattern
    /// path, so its behaviour is unchanged.
    fn reinterpret_stream(
        &self,
        value: &Value,
        src: &ScalarShape,
        dst: &ScalarShape,
    ) -> Option<InterpResult<Value>> {
        if !shape_is_stream(src) || !shape_is_stream(dst) {
            return None;
        }
        let total = src.bits();
        if total == 0 || total != dst.bits() {
            return None;
        }
        let wrapped = matches!(src, ScalarShape::AdtField { .. })
            || matches!(dst, ScalarShape::AdtField { .. });
        if total <= 128 && !wrapped {
            return None;
        }
        Some(self.reinterpret_stream_inner(value, src, dst))
    }

    fn reinterpret_stream_inner(
        &self,
        value: &Value,
        src: &ScalarShape,
        dst: &ScalarShape,
    ) -> InterpResult<Value> {
        // Flatten the source into (bits, width, init)
        // leaves, then pack them into a word stream with
        // an init mask of the same layout.
        let mut flat: Vec<(u128, u32, bool)> = Vec::new();
        self.flatten_stream(value, src, &mut flat)?;
        let total: usize =
            flat.iter().map(|(_, w, _)| *w as usize).sum();
        let mut words = vec![0u64; total.div_ceil(64)];
        let mut mask = vec![0u64; total.div_ceil(64)];
        let mut offset = 0usize;
        for (bits, width, init) in &flat {
            stream_insert(
                &mut words,
                &mut mask,
                offset,
                *bits,
                *width,
                *init,
            );
            offset += *width as usize;
        }
        let mut offset = 0usize;
        self.construct_stream(dst, &words, &mask, &mut offset)
    }

    /// Flatten `value` through a stream shape into
    /// `(bits, width, init)` leaves in storage order.
    fn flatten_stream(
        &self,
        value: &Value,
        shape: &ScalarShape,
        out: &mut Vec<(u128, u32, bool)>,
    ) -> InterpResult<()> {
        match shape {
            ScalarShape::Array(elem, n) => match value {
                Value::Tuple(items) if items.len() as u64 == *n => {
                    for item in items {
                        self.flatten_stream(item, elem, out)?;
                    }
                    Ok(())
                }
                Value::Bytes(b)
                    if elem.bits() == 8 && b.len() as u64 == *n =>
                {
                    for byte in b {
                        out.push((u128::from(*byte), 8, true));
                    }
                    Ok(())
                }
                Value::Uninit => {
                    flatten_stream_uninit(shape, out);
                    Ok(())
                }
                other => Err(InterpError::type_mismatch(
                    "transmute",
                    "an array value of the source shape's length",
                    other.kind_name(),
                )),
            },
            ScalarShape::AdtField { index, inner, .. } => {
                match value {
                    Value::Enum { fields, .. } => {
                        match fields.get(*index as usize) {
                            Some(field) => {
                                self.flatten_stream(field, inner, out)
                            }
                            None => Err(InterpError::type_mismatch(
                                "transmute",
                                "an ADT value holding the payload \
                                 field",
                                "an ADT value without it",
                            )),
                        }
                    }
                    Value::Uninit => {
                        flatten_stream_uninit(shape, out);
                        Ok(())
                    }
                    // The storage form: the payload value
                    // itself where the aggregate would sit.
                    _ => self.flatten_stream(value, inner, out),
                }
            }
            leaf => {
                if matches!(value, Value::Uninit) {
                    out.push((0, leaf.bits(), false));
                    return Ok(());
                }
                let bits = self.flatten_scalar(value, leaf)?;
                out.push((bits, leaf.bits(), true));
                Ok(())
            }
        }
    }

    /// Build the value of stream shape `shape` from the
    /// bit stream at `offset`, in storage order.
    fn construct_stream(
        &self,
        shape: &ScalarShape,
        words: &[u64],
        mask: &[u64],
        offset: &mut usize,
    ) -> InterpResult<Value> {
        match shape {
            ScalarShape::Array(elem, n) => {
                let mut items = Vec::with_capacity(*n as usize);
                for _ in 0..*n {
                    items.push(self.construct_stream(
                        elem, words, mask, offset,
                    )?);
                }
                Ok(Value::Tuple(items))
            }
            ScalarShape::AdtField {
                index,
                count,
                fill,
                inner,
            } => {
                let payload =
                    self.construct_stream(inner, words, mask, offset)?;
                let mut fields = vec![Value::Uninit; *count];
                for (i, c) in fill {
                    fields[*i as usize] = self.eval_constant(c)?;
                }
                fields[*index as usize] = payload;
                Ok(Value::Enum {
                    variant: 0,
                    fields,
                })
            }
            leaf => {
                let width = leaf.bits();
                let (bits, init) =
                    stream_extract(words, mask, *offset, width);
                *offset += width as usize;
                if init {
                    self.construct_scalar(bits, leaf)
                } else {
                    Ok(Value::Uninit)
                }
            }
        }
    }

    /// The value of shape `dst` that is `value` (of shape
    /// `src`) with every pointer leaf carried across unchanged,
    /// or `None` when the two shapes do not line up
    /// structurally around a pointer leaf (the caller then
    /// reinterprets through the bit pattern). A non-pointer
    /// value at a pointer leaf also yields `None`, so the bit
    /// path keeps reporting it as the type mismatch it is.
    fn reinterpret_preserving(
        &self,
        value: &Value,
        src: &ScalarShape,
        dst: &ScalarShape,
    ) -> Option<Value> {
        match (src, dst) {
            (ScalarShape::Ptr, ScalarShape::Ptr) => match value {
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => {
                    Some(value.clone())
                }
                _ => None,
            },
            (ScalarShape::FnPtr, ScalarShape::FnPtr) => match value {
                Value::FnPtr(_) | Value::Closure { .. } => {
                    Some(value.clone())
                }
                _ => None,
            },
            (
                ScalarShape::AdtField {
                    index: si,
                    count: sc,
                    inner: sinner,
                    ..
                },
                ScalarShape::AdtField {
                    index: di,
                    count: dc,
                    fill: dfill,
                    inner: dinner,
                },
            ) if si == di && sc == dc => {
                match value {
                    Value::Enum { variant, fields } => {
                        let payload = fields.get(*si as usize)?;
                        let payload =
                            self.reinterpret_preserving(payload, sinner, dinner)?;
                        let mut fields = fields.clone();
                        fields[*si as usize] = payload;
                        Some(Value::Enum {
                            variant: *variant,
                            fields,
                        })
                    }
                    // The storage form: the payload value itself
                    // where the aggregate would sit (the same
                    // convention `flatten_scalar` encodes). The
                    // result is re-wrapped in the target
                    // aggregate, whose remaining fields are the
                    // baked-in fill constants.
                    _ => {
                        let payload =
                            self.reinterpret_preserving(value, sinner, dinner)?;
                        let mut fields = vec![Value::Uninit; *dc];
                        for (i, c) in dfill {
                            fields[*i as usize] = self.eval_constant(c).ok()?;
                        }
                        fields[*di as usize] = payload;
                        Some(Value::Enum {
                            variant: 0,
                            fields,
                        })
                    }
                }
            }
            // Unwrapping one side: the payload of a wrapper
            // shape reinterpreted as the bare target shape (and
            // the reverse, wrapping).
            (
                ScalarShape::AdtField {
                    index,
                    inner,
                    ..
                },
                _,
            ) => {
                let payload = match value {
                    Value::Enum { fields, .. } => fields.get(*index as usize)?,
                    _ => value,
                };
                self.reinterpret_preserving(payload, inner, dst)
            }
            (
                _,
                ScalarShape::AdtField {
                    index,
                    count,
                    fill,
                    inner,
                },
            ) => {
                let payload = self.reinterpret_preserving(value, src, inner)?;
                let mut fields = vec![Value::Uninit; *count];
                for (i, c) in fill {
                    fields[*i as usize] = self.eval_constant(c).ok()?;
                }
                fields[*index as usize] = payload;
                Some(Value::Enum {
                    variant: 0,
                    fields,
                })
            }
            (
                ScalarShape::NicheEnum {
                    inner: sinner,
                    untagged_variant: suv,
                    niche_variants: snv,
                    niche_start: sns,
                    niche_shift: sshift,
                    niche_bits: sbits,
                },
                ScalarShape::NicheEnum {
                    inner: dinner,
                    untagged_variant: duv,
                    niche_variants: dnv,
                    niche_start: dns,
                    niche_shift: dshift,
                    niche_bits: dbits,
                },
            ) if suv == duv
                && snv == dnv
                && sns == dns
                && sshift == dshift
                && sbits == dbits =>
            match value {
                Value::Enum { variant, fields } if variant == suv => {
                    let payload = fields.first()?;
                    let payload =
                        self.reinterpret_preserving(payload, sinner, dinner)?;
                    let mut fields = fields.clone();
                    fields[0] = payload;
                    Some(Value::Enum {
                        variant: *variant,
                        fields,
                    })
                }
                Value::Enum { .. } => Some(value.clone()),
                _ => None,
            },
            (
                ScalarShape::NicheEnumFields {
                    fields: sfields,
                    untagged_variant: suv,
                    niche_variants: snv,
                    niche_start: sns,
                    niche_shift: sshift,
                    niche_bits: sbits,
                    ..
                },
                ScalarShape::NicheEnumFields {
                    fields: dfields,
                    untagged_variant: duv,
                    niche_variants: dnv,
                    niche_start: dns,
                    niche_shift: dshift,
                    niche_bits: dbits,
                    ..
                },
            ) if suv == duv
                && snv == dnv
                && sns == dns
                && sshift == dshift
                && sbits == dbits =>
            match value {
                Value::Enum { variant, fields } if variant == suv => {
                    let mut out = fields.clone();
                    for ((si, _, ss), (di, _, ds)) in
                        sfields.iter().zip(dfields.iter())
                    {
                        if si != di {
                            return None;
                        }
                        let field = fields.get(*si as usize)?;
                        out[*si as usize] =
                            self.reinterpret_preserving(field, ss, ds)?;
                    }
                    Some(Value::Enum {
                        variant: *variant,
                        fields: out,
                    })
                }
                Value::Enum { .. } => Some(value.clone()),
                _ => None,
            },
            (
                ScalarShape::Array(selem, sn),
                ScalarShape::Array(delem, dn),
            ) if sn == dn => match value {
                Value::Tuple(items) if items.len() as u64 == *sn => {
                    let mut out = Vec::with_capacity(items.len());
                    for item in items {
                        out.push(self.reinterpret_preserving(item, selem, delem)?);
                    }
                    Some(Value::Tuple(out))
                }
                _ => None,
            },
            _ => None,
        }
    }
}

/// Whether a shape decomposes into a bit stream: scalar
/// leaves (integers, floats, `bool`, `char`) under arrays
/// and one-payload wrappers. The interpreter-side mirror
/// of the conversion's stream predicate — the conversion
/// only emits stream pairs, but the check is repeated
/// here so a hand-built program gets the same reading.
fn shape_is_stream(shape: &ScalarShape) -> bool {
    match shape {
        ScalarShape::Int { .. }
        | ScalarShape::Float { .. }
        | ScalarShape::Bool
        | ScalarShape::Char => true,
        ScalarShape::Array(elem, _) => shape_is_stream(elem),
        ScalarShape::AdtField { inner, .. } => shape_is_stream(inner),
        _ => false,
    }
}

/// The leaves of a stream shape read from wholly uninit
/// storage: every leaf uninit at its own width.
fn flatten_stream_uninit(
    shape: &ScalarShape,
    out: &mut Vec<(u128, u32, bool)>,
) {
    match shape {
        ScalarShape::Array(elem, n) => {
            for _ in 0..*n {
                flatten_stream_uninit(elem, out);
            }
        }
        ScalarShape::AdtField { inner, .. } => {
            flatten_stream_uninit(inner, out);
        }
        leaf => out.push((0, leaf.bits(), false)),
    }
}

/// Insert `width` bits of `bits` into the word stream at
/// bit `offset`, and the all-ones pattern into the mask
/// stream when `init` (an uninit leaf's mask bits stay
/// clear, whatever its value bits are).
fn stream_insert(
    words: &mut [u64],
    mask: &mut [u64],
    mut offset: usize,
    mut bits: u128,
    mut width: u32,
    init: bool,
) {
    while width > 0 {
        let wi = offset / 64;
        let shift = (offset % 64) as u32;
        let take = width.min(64 - shift);
        let chunk_mask: u64 = if take == 64 {
            u64::MAX
        } else {
            (1u64 << take) - 1
        };
        let chunk = (bits as u64) & chunk_mask;
        words[wi] |= chunk << shift;
        if init {
            mask[wi] |= chunk_mask << shift;
        }
        bits >>= take;
        width -= take;
        offset += take as usize;
    }
}

/// Extract `width` bits at bit `offset` from the word
/// stream, with whether every one of them is init.
fn stream_extract(
    words: &[u64],
    mask: &[u64],
    mut offset: usize,
    mut width: u32,
) -> (u128, bool) {
    let mut bits: u128 = 0;
    let mut init = true;
    let mut done: u32 = 0;
    while width > 0 {
        let wi = offset / 64;
        let shift = (offset % 64) as u32;
        let take = width.min(64 - shift);
        let chunk_mask: u64 = if take == 64 {
            u64::MAX
        } else {
            (1u64 << take) - 1
        };
        let chunk = (words[wi] >> shift) & chunk_mask;
        bits |= u128::from(chunk) << done;
        if (mask[wi] >> shift) & chunk_mask != chunk_mask {
            init = false;
        }
        done += take;
        width -= take;
        offset += take as usize;
    }
    (bits, init)
}

pub(crate) fn project_value(
    value: Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<Value> {
    project_value_ev(None, None, value, projection, func, place)
}

/// [`project_value`] with type evidence: `root_ty` is the
/// declared type of `value`, walked alongside the projection
/// so the transparent-wrapper descent can tell a wrapper's
/// payload -- a complete value of the type the projection
/// addresses -- from a partially materialised aggregate's
/// first field, which is a value of a strictly smaller type
/// (see `wrapper_descent_sanctioned`). Callers without
/// evidence use `project_value`, whose descent judgments are
/// the shape-only ones.
pub(crate) fn project_value_ev(
    program: Option<&Program>,
    root_ty: Option<Ty>,
    value: Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<Value> {
    let composed;
    let projection = if projection_needs_composition(projection) {
        composed = compose_projection(&value, projection, func, place);
        &composed[..]
    } else {
        projection
    };
    let mut value = value;
    let mut step_idx = 0;
    // The state a coroutine downcast named, for the field step
    // that follows it (see the `Downcast` arm below).
    let mut downcast_state: Option<u32> = None;
    // The declared type of the value in hand, stepped alongside
    // it (see `project_value_ev`).
    let mut cur_ty = root_ty;
    while step_idx < projection.len() {
        let step = &projection[step_idx];
        step_idx += 1;
        let pos = step_idx - 1;
        if value == Value::Uninit {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        }
        let step_state = downcast_state.take();
        // A one-field wrapper the storage acquired through a
        // transparent-wrapper pointer (a `MaybeUninit` pointee
        // materialised by projected writes that entered field 0
        // — see `ptr_field_value`): the payload's own readers
        // address the payload's fields directly on the storage
        // (a `Box<Leaf>` after `assume_init` reads the leaf's
        // fields where the wrapper's field 0 holds them), so an
        // out-of-range field step descends into the wrapper's
        // only field first. In-range steps are unaffected, and a
        // step that stays out of range keeps its error.
        if let RefProj::Field(idx) = step {
            loop {
                let descend = match &value {
                    Value::Enum { fields, .. } => {
                        fields.len() == 1
                            && fields.get(*idx).is_none()
                            && wrapper_payload(&fields[0])
                            && wrapper_descent_sanctioned(
                                program,
                                cur_ty.as_ref(),
                                &value,
                            )
                    }
                    _ => false,
                };
                if !descend {
                    break;
                }
                cur_ty = descent_payload_ty(program, cur_ty.as_ref(), &value);
                value = match value {
                    Value::Enum { mut fields, .. } => {
                        fields.pop().expect("one field checked")
                    }
                    _ => unreachable!("descent checked the shape"),
                };
            }
        }
        // The declared type of the value in hand, before this
        // step moves it: the index arm's single-array-field pun
        // consults it (see there).
        let step_base_ty = cur_ty.clone();
        // The declared type steps alongside the value, so the
        // next step's descent judgment sees the type of the
        // value it judges. A step the type walk cannot follow
        // ends the evidence; later judgments are the shape-only
        // ones, exactly as for a caller without evidence.
        cur_ty = match step {
            RefProj::Field(idx) => {
                let variant = match &value {
                    Value::Enum { variant, .. } => *variant as usize,
                    _ => 0,
                };
                stepped_field_ty(program, cur_ty.as_ref(), variant, *idx)
            }
            RefProj::Index(_) => stepped_index_ty(cur_ty.as_ref()),
            RefProj::UnionField { idx, .. } => {
                stepped_field_ty(program, cur_ty.as_ref(), 0, *idx)
            }
            RefProj::Downcast(_) => cur_ty,
            _ => None,
        };
        value = match step {
            RefProj::Field(idx) => match value {
                Value::Tuple(ref items) => items.get(*idx).cloned().ok_or_else(|| {
                    InterpError::UninitializedRead {
                        func: func.to_string(),
                        place: place.to_string(),
                    }
                })?,
                Value::Enum { ref fields, .. } => {
                    fields.get(*idx).cloned().ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?
                }
                Value::Coroutine {
                    ref fields,
                    ref saved,
                    ..
                } => {
                    // A field step through a downcast reads that
                    // state's saved storage; without one it reads
                    // the upvar storage (see `Value::Coroutine`).
                    // State 0 is the unresumed state, whose fields
                    // ARE the upvars, so its downcast addresses the
                    // upvar storage too.
                    let storage = match step_state {
                        Some(0) | None => Some(fields),
                        Some(state) => saved.get(state as usize),
                    };
                    storage
                        .and_then(|fields| fields.get(*idx))
                        .cloned()
                        .ok_or_else(|| InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        })?
                }
                Value::Closure { ref env, .. } => {
                    env.get(*idx).cloned().ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?
                }
                // Field 0 of a string value is the string
                // itself: the byte-string wrappers std layers
                // over the same bytes (`String` over its buffer,
                // `OsString` over its `Buf`, `PathBuf` over its
                // `OsString`, `Path` over its `OsStr`, `OsStr`
                // over its byte slice) are all one-field chains
                // whose every layer denotes the same byte string
                // in this model, and the `AsRef` / `Deref` glue
                // that walks them (e.g. the projections inside
                // `env::split_paths`) reads the same value at
                // every step. Consumers that need the buffer in
                // vector form reach it through the modelled
                // `Vec` API, not through these projections.
                Value::Str(_) if *idx == 0 => value,
                // A reference-counted handle in trait-object
                // form: the `DynRef` *is* the `Rc` / `Arc` /
                // `Weak` value, so a field step reads the
                // handle's own field (field 0 is its `ptr`, the
                // `NonNull` over the `RcInner` allocation).
                // Code that walks the handle's inner structure
                // through the raw pointer (`Rc::inner` and with
                // it `strong_count`, `downgrade`, ...) depends on
                // this: resolving the step to the trait object
                // itself would send the pointer's dereference to
                // the pointee instead of the `RcInner`.
                Value::DynRef { base, .. } if is_rc_arc_handle(&base) => {
                    project_value(*base, std::slice::from_ref(step), func, place)?
                }
                // Field 0 of a fat pointer value is the pointer
                // field of its smart-pointer wrapper (a `Box`'s
                // `Unique`): in this model the `DynRef` *is* that
                // pointer, metadata included.
                Value::DynRef { .. } if *idx == 0 => value,
                // The same wrapper chain for a thin pointer: a
                // `Box`'s field 0 is its `Unique`, whose field 0
                // is the `NonNull`, whose field 0 is the raw
                // pointer — and in this model the box's one field
                // is the pointer value itself, so each of those
                // steps yields it unchanged. (A reference to a
                // place never sits at a field step of a value
                // projection: place references are resolved before
                // value projection begins, so a `Ref` here is a
                // pointer value extracted from a wrapper.)
                Value::Ref(_) if *idx == 0 => value,
                // The wrapper's one remaining field is its
                // zero-sized `PhantomData` marker (a `Unique` /
                // `NonNull` holds the pointer at field 0 and the
                // marker at field 1): the model keeps no storage
                // for it, so the step yields the marker value
                // itself.
                Value::Ref(_) => Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                },
                // The same wrapper chain for a byte string: a
                // `Path`'s field 0 is its `OsStr`, whose field 0
                // is the byte slice, and in this model the value
                // at the base of that chain is the string itself
                // (the constructors reach it through raw pointer
                // casts: `Path::new` casts `&str` to `&Path`).
                // Each wrapper's only field sits at offset 0 with
                // exactly the string's representation, so the
                // step yields the value unchanged.
                Value::Bytes(_) if *idx == 0 => value,
                // The same wrapper chain over a slice value: a
                // fat pointer in this model *is* the slice it
                // denotes (a `&OsStr` over non-UTF-8 bytes is a
                // fat pointer to the byte elements, the shape
                // `sub_path_value` builds), and the wrappers std
                // layers over a slice (`OsStr` over its `Slice`,
                // `Slice` over its `[u8]`, `Path` over its
                // `OsStr`) are one-field chains whose only field
                // is that slice, so a field-0 step yields the fat
                // pointer unchanged, exactly as the string and
                // byte-string forms above. (A field step on a fat
                // pointer only ever arrives through such a
                // wrapper chain: a plain slice place takes index
                // steps, never field steps.)
                Value::FatRef { .. } if *idx == 0 => value,
                // A one-field wrapper over a scalar whose storage
                // the model holds as the bare scalar (a control
                // byte reached through a byte-string constant:
                // the tag's storage is the byte itself): field 0
                // of the wrapper is the scalar it wraps.
                Value::Int(_) | Value::Uint(_) | Value::Float(_) | Value::Bool(_)
                    if *idx == 0 =>
                {
                    value
                }
                // String data reached through its own wrapper
                // descent: a `String` the model holds as bare
                // string data has no materialised `Vec` /
                // `RawVec` / `Unique` / `NonNull` structure, yet
                // converted code (the inlined `Borrow::borrow`
                // for `Cow`, `String::as_ptr`, ...) walks exactly
                // that one-field chain to the data. Field 0 of
                // the chain is the string data itself, yielded
                // unchanged like the byte-string wrapper chain
                // above: the consumers that need a pointer form
                // the reference at their own site (reference
                // formation reads the target's value to recover
                // the fat length, so the read must stay the
                // data). `Field(1)` at any step is the vector's
                // length slot, the string's length.
                Value::Str(_) if *idx == 0 => value,
                Value::Str(ref text) if *idx == 1 => {
                    Value::Uint(text.len() as u128)
                }
                // The two fields of a modelled `fmt::Arguments`
                // (see the struct's encoding in core::fmt): field
                // 0 is the `template: NonNull<u8>`, field 1 the
                // `args: NonNull<rt::Argument>`. The model keeps
                // neither as storage, so the projection answers
                // with the field's own value: a one-field `NonNull`
                // wrapper around a pointer to the modelled bytes
                // / argument array, or — for the `from_str` form,
                // whose `args` field is not a pointer at all — the
                // tagged length word `(len << 1) | 1` as an
                // address-only pointer, which is exactly what
                // `Arguments::as_str` transmutes it back to.
                Value::FmtArguments(ref args) => {
                    use crate::value::FmtArguments;
                    let ptr = match (idx, args) {
                        (0, FmtArguments::Plain(text)) => {
                            Value::Ref(RefTarget::Value(
                                Box::new(Value::Str(text.clone())),
                                Vec::new(),
                            ))
                        }
                        (0, FmtArguments::Template { template, .. }) => {
                            Value::Ref(RefTarget::Value(
                                Box::new(Value::Bytes(template.clone())),
                                Vec::new(),
                            ))
                        }
                        (1, FmtArguments::Plain(text)) => {
                            Value::Ref(RefTarget::Address(
                                ((text.len() as u64) << 1) | 1,
                            ))
                        }
                        (1, FmtArguments::Template { args, .. }) => {
                            Value::Ref(RefTarget::Value(
                                Box::new(Value::Tuple(args.clone())),
                                Vec::new(),
                            ))
                        }
                        _ => {
                            return Err(InterpError::type_mismatch(
                                format!("field projection on {place}"),
                                "field 0 or 1 of fmt::Arguments",
                                format!("field {idx}"),
                            ));
                        }
                    };
                    Value::Enum {
                        variant: 0,
                        fields: vec![ptr],
                    }
                }
                // A sparse vector value (see `Value::SparseVec`):
                // a field step addresses an element, as the drop
                // glue's per-element walk addresses a materialised
                // vector's elements — the stored elements read as
                // themselves (the spare-capacity tail past the
                // declared length included: it is storage the
                // vector still holds), the unmaterialised tail as
                // the uninitialised slots it denotes, and past
                // the storage the step finds no slot, as a
                // materialised aggregate's missing field does not.
                Value::SparseVec {
                    ref elems,
                    cap,
                    ..
                } => {
                    if *idx as u128 >= cap.max(elems.len() as u128) {
                        return Err(InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        });
                    }
                    elems
                        .get(*idx)
                        .cloned()
                        .unwrap_or(Value::Uninit)
                }
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("field projection on {place}"),
                        "tuple or ADT value",
                        other.kind_name(),
                    ));
                }
            },
            RefProj::UnionField { .. } => {
                // Union field reads need the program's scalar
                // shapes, which this type-free projector does not
                // carry: every interpreter read path routes union
                // steps through `project_value_t` instead. Landing
                // here means a caller applied a stored projection
                // directly; that is an explicit error, not a slot
                // read.
                return Err(InterpError::type_mismatch(
                    format!("union field projection on {place}"),
                    "a union field read through the typed projection path",
                    "the untyped value projector",
                ));
            }
            RefProj::Index(idx) => match value {
                Value::FatRef {
                    ref base,
                    len,
                } => {
                    if *idx as u128 >= len {
                        return Err(index_oob(len as usize, *idx));
                    }
                    project_value((**base).clone(), &[RefProj::Index(*idx)], func, place)?
                }
                // A sparse vector value (see `Value::SparseVec`):
                // the stored elements read as themselves (the
                // spare-capacity tail past the declared length
                // included, as for the field step above), the
                // unmaterialised tail as the uninitialised slots
                // it denotes, and past the storage the index is
                // out of bounds, as for the materialised
                // aggregate. (A slice read is bounded by its fat
                // length before it delegates here, so the wider
                // storage bound does not widen slice reads.)
                Value::SparseVec {
                    ref elems,
                    cap,
                    ..
                } => {
                    let extent = cap.max(elems.len() as u128);
                    if *idx as u128 >= extent {
                        return Err(index_oob(extent as usize, *idx));
                    }
                    elems
                        .get(*idx)
                        .cloned()
                        .unwrap_or(Value::Uninit)
                }
                Value::Tuple(ref items) => items
                    .get(*idx)
                    .cloned()
                    .ok_or_else(|| index_oob(items.len(), *idx))?,
                Value::Bytes(ref b) => b
                    .get(*idx)
                    .map(|byte| Value::Uint(*byte as u128))
                    .ok_or_else(|| index_oob(b.len(), *idx))?,
                Value::Str(ref s) => s
                    .as_bytes()
                    .get(*idx)
                    .map(|byte| Value::Uint(*byte as u128))
                    .ok_or_else(|| index_oob(s.len(), *idx))?,
                // A single-field aggregate whose only field is
                // an array is layout-identical to the array
                // itself, and guest code indexes through that
                // pun (a pointer cast between a `#[repr(simd)]`
                // struct and `[T; N]`, as `Simd::as_array`
                // does): the index step continues into the
                // field's elements. The shapes collide when a
                // `Vec` value holds exactly one element and
                // that element is itself a tuple or a byte
                // block: the value is then a one-field
                // aggregate whose only field is a tuple, and
                // shape alone cannot say whether the index
                // selects the element or ranges inside it.
                // The declared type decides: a base the
                // evidence types as a `Vec` is the element
                // aggregate the arm below reads, so the pun
                // does not apply to it. Without evidence the
                // shape-only judgment stands, as elsewhere.
                Value::Enum { ref fields, .. }
                    if fields.len() == 1
                        && matches!(
                            fields[0],
                            Value::Tuple(_) | Value::Bytes(_)
                        )
                        && !matches!(
                            &step_base_ty,
                            Some(ty) if is_vec_adt_ty(ty)
                        ) =>
                {
                    project_value(
                        fields[0].clone(),
                        &[RefProj::Index(*idx)],
                        func,
                        place,
                    )?
                }
                // A `Vec` value is the aggregate of its elements
                // (see the `Vec::from` / `Deref` builtin model):
                // indexing the vector's own place reads the
                // element field. No other ADT value can be the
                // base of an index projection in valid MIR.
                Value::Enum { ref fields, .. } => fields
                    .get(*idx)
                    .cloned()
                    .ok_or_else(|| index_oob(fields.len(), *idx))?,
                ref other => {
                    // A scalar's storage is byte-addressable
                    // through a raw byte pointer (see
                    // `scalar_storage_byte`): the index step
                    // reads the byte at that offset of the
                    // storage, the only element it has. The
                    // declared type in hand judges the byte
                    // under the storage's own width (see
                    // `scalar_storage_byte_as`).
                    let kind = byte_scalar_of_ty(step_base_ty.as_ref());
                    if let Some(len) = scalar_storage_len_as(other, kind) {
                        if *idx >= len {
                            return Err(index_oob(len, *idx));
                        }
                        Value::Uint(u128::from(
                            scalar_storage_byte_as(other, kind, *idx)
                                .expect("index inside the storage"),
                        ))
                    } else {
                        return Err(InterpError::type_mismatch(
                            format!("index projection on {place}"),
                            "array, byte or string value",
                            other.kind_name(),
                        ));
                    }
                }
            },
            RefProj::Subslice { offset, len } => {
                let (start, end) = (*offset, *offset + *len);
                match value {
                    Value::Tuple(ref items) => {
                        check_subslice(items.len(), start, *len)?;
                        Value::Tuple(items[start..end].to_vec())
                    }
                    Value::Bytes(ref b) => {
                        check_subslice(b.len(), start, *len)?;
                        Value::Bytes(b[start..end].to_vec())
                    }
                    Value::Str(ref s) => match s.get(start..end) {
                        Some(sub) => Value::Str(sub.to_string()),
                        None => {
                            return Err(InterpError::panic(format!(
                                "byte range {start}..{end} is not a valid slice of the string at {place}"
                            )))
                        }
                    },
                    // A `Vec` value is the aggregate of its
                    // elements: its sub-slice is the sub-range's
                    // elements as a slice value (a tuple of the
                    // elements), not a shorter `Vec` aggregate --
                    // the window is a slice, and consumers that
                    // dispatch on the value's form (formatting,
                    // pointer casts) read it as one.
                    Value::Enum { ref fields, .. } => {
                        check_subslice(fields.len(), start, *len)?;
                        Value::Tuple(fields[start..end].to_vec())
                    }
                    // A sparse vector value (see
                    // `Value::SparseVec`): the window's elements
                    // as a slice value, as for the aggregate form
                    // above -- the stored elements read as
                    // themselves and the unmaterialised tail as
                    // the uninitialised slots it denotes, exactly
                    // as the sparse `Index` arm reads one element.
                    // The window is bounded by the declared
                    // length: the elements of the vector value
                    // are its first `len`, whatever spare
                    // capacity the storage accounts beyond them.
                    Value::SparseVec {
                        ref elems,
                        len: vec_len,
                        ..
                    } => {
                        let total =
                            usize::try_from(vec_len).unwrap_or(usize::MAX);
                        check_subslice(total, start, *len)?;
                        Value::Tuple(
                            (start..end)
                                .map(|i| {
                                    elems.get(i).cloned().unwrap_or(Value::Uninit)
                                })
                                .collect(),
                        )
                    }
                    // A sub-slice of a fat reference value (a
                    // constant slice's representation): a zero
                    // offset only adjusts the carried length; a
                    // materialised base is projected directly.
                    Value::FatRef { ref base, len: fat_len } => {
                        if start == 0 {
                            if end as u128 > fat_len {
                                return Err(InterpError::panic(format!(
                                    "range end index {end} out of range for slice of length {fat_len}"
                                )));
                            }
                            Value::FatRef {
                                base: base.clone(),
                                len: end as u128,
                            }
                        } else {
                            match &**base {
                                Value::Tuple(_) | Value::Bytes(_) | Value::Str(_) => {
                                    project_value(
                                        (**base).clone(),
                                        &[RefProj::Subslice { offset: start, len: *len }],
                                        func,
                                        place,
                                    )?
                                }
                                other => {
                                    return Err(InterpError::type_mismatch(
                                        format!("subslice projection on {place}"),
                                        "materialised slice value",
                                        other.kind_name(),
                                    ))
                                }
                            }
                        }
                    }
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("subslice projection on {place}"),
                            "array, slice or string value",
                            other.kind_name(),
                        ));
                    }
                }
            }
            RefProj::Downcast(variant) => match value {
                // A coroutine's downcast names the state whose saved
                // storage the following field step uses; the access
                // is valid whatever the current tag is (the body
                // writes a state's fields before setting the tag to
                // that state). The tag itself does not move here:
                // the state is remembered for the next step only.
                Value::Coroutine { .. } => {
                    downcast_state = Some(*variant);
                    value
                }
                Value::Enum {
                    variant: current, ..
                } => {
                    if current != *variant {
                        return Err(InterpError::type_mismatch(
                            format!("downcast of {place}"),
                            format!("variant {variant}"),
                            format!("variant {current}"),
                        ));
                    }
                    value
                }
                Value::Tuple(_) if *variant == 0 => value,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("downcast of {place}"),
                        "ADT value",
                        other.kind_name(),
                    ));
                }
            },
            RefProj::ByteOffset(disp) => {
                if *disp == 0 {
                    value
                } else if *disp >= 0 {
                    // A byte-displaced pointer into an element
                    // block (an allocation's storage, whose
                    // slots are byte positions): the aggregate
                    // stored at the displaced position is the
                    // pointee, when the step that follows is
                    // not the index step of the byte view below.
                    let elem_block_value = if !matches!(
                        projection.get(step_idx),
                        Some(RefProj::Index(_))
                    ) {
                        match &value {
                            Value::Tuple(items) => items
                                .get(*disp as usize)
                                .filter(|elem| {
                                    matches!(
                                        elem,
                                        Value::Enum { .. } | Value::Tuple(_)
                                    )
                                })
                                .cloned(),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if let Some(elem) = elem_block_value {
                        elem
                    } else {
                        let kind = byte_scalar_of_ty(step_base_ty.as_ref());
                        let Some(len) = scalar_storage_len_as(&value, kind) else {
                            return Err(InterpError::UnsupportedProjection {
                                kind: format!(
                                    "read through a pointer displaced {disp} bytes from a value boundary"
                                ),
                            });
                        };
                        let Some(&RefProj::Index(k)) = projection.get(step_idx) else {
                            return Err(InterpError::UnsupportedProjection {
                                kind: format!(
                                    "read through a pointer displaced {disp} bytes from a value boundary"
                                ),
                            });
                        };
                        // A byte read through a pointer displaced into
                        // a scalar's storage: the index step that
                        // follows selects the byte at the combined
                        // offset, the read side of the byte view
                        // `scalar_storage_byte` describes.
                        let at = *disp as usize + k;
                        step_idx += 1;
                        match scalar_storage_byte_as(&value, kind, at) {
                            Some(byte) => Value::Uint(u128::from(byte)),
                            None => return Err(index_oob(len, at)),
                        }
                    }
                } else {
                    // The position does not coincide with a value
                    // boundary, and the model has no byte-level
                    // view of a value's storage to reinterpret.
                    return Err(InterpError::UnsupportedProjection {
                        kind: format!(
                            "read through a pointer displaced {disp} bytes from a value boundary"
                        ),
                    });
                }
            }
            RefProj::VecLen => {
                // The length slot of a vector value (see the
                // variant's documentation): the read answers
                // the declared length, in whichever form the
                // value holds it.
                let len = vec_value_len(&value).ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("vector length slot of {place}"),
                        "vector value",
                        value.kind_name(),
                    )
                })?;
                Value::Uint(len)
            }
        };
    }
    Ok(value)
}

/// The declared length of a vector value in any of its forms:
/// the sparse form's recorded length, the real aggregate
/// form's length slot, or the element list's own length (the
/// same three answers the length-slot arm of place resolution
/// gives). `None` for a value that is not a vector.
pub(crate) fn vec_value_len(value: &Value) -> Option<u128> {
    match value {
        Value::SparseVec { len, .. } => Some(*len),
        Value::Enum { fields, .. } => {
            match crate::builtins::vec_real_aggregate_parts(value) {
                Some((_, len)) => Some(len as u128),
                None => Some(fields.len() as u128),
            }
        }
        _ => None,
    }
}

/// Assign `value` to the slot `projection` selects inside the
/// stored value `slot` (the final step of a `write_resolved`
/// once the target's base slot is in hand). One shape needs a
/// read-modify-write instead of a slot assignment: a byte
/// written through a trailing index step into a scalar's
/// storage (see [`scalar_storage_byte`]) has no byte slot to
/// hand out, so it is spliced into the scalar in place. Every
/// other shape navigates to the selected slot and assigns, with
/// Grow every tuple an `Index` step of `projection` addresses
/// along the path the step denotes, filling with `Uninit` slots.
/// Allocation storage materialises an array field element by
/// element (a fresh node's key area has no array value yet: its
/// tuple is exactly as long as the highest index written), so an
/// allocation write's path is grown before navigation refuses an
/// index past a partial tuple's end. A built array value in a
/// local or a static keeps its length: the same step there is the
/// guest's own out-of-bounds error, which `navigate_mut` reports.
/// The walk mirrors navigation only where the current value's
/// shape already determines the next slot; anything else (an
/// `Uninit` slot above all) is left for navigation to
/// materialise or refuse.
///
/// The declared type rides along when the caller has it, so
/// the wrapper-descent retries are judged by the same type
/// evidence as the navigation itself (see
/// `wrapper_descent_sanctioned`): growing inside a partially
/// materialised aggregate's first field would mutate a sibling
/// the write does not address, ahead of the correctly typed
/// assignment. Without evidence the shape judgment stands.
pub(crate) fn grow_tuples_along(
    value: &mut Value,
    projection: &[RefProj],
    program: Option<&Program>,
    ty: Option<Ty>,
) {
    let Some((step, rest)) = projection.split_first() else {
        return;
    };
    // The wrapper-descent judgment for this node, taken
    // before the walk borrows the value mutably.
    let sanctioned = wrapper_descent_sanctioned(program, ty.as_ref(), value);
    let payload_ty = descent_payload_ty(program, ty.as_ref(), value);
    match (value, step) {
        (Value::Tuple(items), RefProj::Index(idx)) => {
            // The materialisation bound the navigation itself
            // enforces (`navigate_mut_grow` errors at this
            // index): growing to it here would resize the
            // storage before that error can fire -- and at the
            // extreme index the size computation itself
            // overflows. Leave the storage alone and let the
            // navigation produce its error.
            if *idx >= crate::builtins::MAX_VEC_SET_LEN_ELEMENTS {
                return;
            }
            if *idx >= items.len() {
                items.resize(*idx + 1, Value::Uninit);
            }
            let elem_ty = stepped_index_ty(ty.as_ref());
            grow_tuples_along(&mut items[*idx], rest, program, elem_ty);
        }
        (Value::Tuple(items), RefProj::Field(idx)) => {
            let field_ty = stepped_field_ty(program, ty.as_ref(), 0, *idx);
            if let Some(item) = items.get_mut(*idx) {
                grow_tuples_along(item, rest, program, field_ty);
            }
        }
        (Value::Enum { fields, variant }, RefProj::Field(idx)) => {
            let variant = *variant as usize;
            if *idx < fields.len() {
                let field_ty = stepped_field_ty(program, ty.as_ref(), variant, *idx);
                grow_tuples_along(&mut fields[*idx], rest, program, field_ty);
            } else if fields.len() == 1 && wrapper_payload(&fields[0]) && sanctioned {
                // The transparent-wrapper descent navigation
                // performs: the step retries one level down.
                grow_tuples_along(&mut fields[0], projection, program, payload_ty);
            }
        }
        (Value::Enum { fields, variant }, RefProj::Index(idx)) => {
            let variant = *variant as usize;
            if fields.len() == 1
                && matches!(fields[0], Value::Tuple(_) | Value::Bytes(_))
                && sanctioned
            {
                grow_tuples_along(&mut fields[0], projection, program, payload_ty);
            } else if *idx < fields.len() {
                let field_ty = stepped_field_ty(program, ty.as_ref(), variant, *idx);
                grow_tuples_along(&mut fields[*idx], rest, program, field_ty);
            }
        }
        (value, RefProj::Downcast(_)) => grow_tuples_along(value, rest, program, ty),
        _ => {}
    }
}

/// exactly the errors that navigation produced before.
/// The declared type of field `idx` of `variant` of the ADT
/// (or tuple) `ty`, from the program's field-type table -- the
/// free-function form of `Interpreter::adt_field_ty`, for the
/// navigation type walk below.
/// The recorded field-type table for a definition name. The
/// table is keyed by the defining crate's path, while a type
/// spelled through a re-export carries the re-exporting crate's
/// path (`std::option::Option` for the definition recorded as
/// `core::option::Option`); the std-family roots are tried in
/// turn so both spellings find the one entry.
fn adt_field_tys_of<'p>(program: &'p Program, name: &str) -> Option<&'p Vec<Vec<Option<Ty>>>> {
    if let Some(entry) = program.adt_field_tys.get(name) {
        return Some(entry);
    }
    let (_, rest) = name.split_once("::")?;
    for root in ["core", "std", "alloc"] {
        if let Some(entry) = program.adt_field_tys.get(&format!("{root}::{rest}")) {
            return Some(entry);
        }
    }
    None
}

fn adt_field_ty_in(program: &Program, ty: &Ty, variant: usize, idx: usize) -> Option<Ty> {
    match ty {
        Ty::Adt(name) => adt_field_tys_of(program, name)?
            .get(variant)?
            .get(idx)?
            .clone(),
        Ty::Tuple(ts) => ts.get(idx).cloned(),
        _ => None,
    }
}

/// The number of fields the program's field-type table declares
/// for `variant` of the ADT `name`, when recorded.
fn adt_variant_field_count(program: &Program, name: &str, variant: usize) -> Option<usize> {
    adt_field_tys_of(program, name)?
        .get(variant)
        .map(|fields| fields.len())
}

/// Walk `ty` across a whole projection at the type level
/// alone (no value at hand), for callers that project a value
/// in segments and must carry the declared type across a
/// segment boundary (`project_value_t`'s union-field
/// segmentation). A step the walk cannot follow ends the
/// evidence (`None`).
pub(crate) fn step_ty_across(
    program: Option<&Program>,
    ty: Option<Ty>,
    projection: &[RefProj],
) -> Option<Ty> {
    let mut ty = ty;
    let mut variant = 0usize;
    for step in projection {
        ty = match step {
            RefProj::Field(idx) => {
                let next = stepped_field_ty(program, ty.as_ref(), variant, *idx);
                variant = 0;
                next
            }
            RefProj::UnionField { idx, .. } => {
                stepped_field_ty(program, ty.as_ref(), 0, *idx)
            }
            RefProj::Downcast(v) => {
                variant = *v as usize;
                ty
            }
            RefProj::Index(_) => stepped_index_ty(ty.as_ref()),
            _ => None,
        };
        if ty.is_none() {
            return None;
        }
    }
    ty
}

/// The declared type one field step down from `ty`, when both
/// the program's field-type table and the current type are
/// known; the type walk in `navigate_mut_grow` uses it to keep
/// each slot's declared type attached to the slot.
fn stepped_field_ty(
    program: Option<&Program>,
    ty: Option<&Ty>,
    variant: usize,
    idx: usize,
) -> Option<Ty> {
    match (program, ty) {
        (Some(program), Some(ty)) => adt_field_ty_in(program, ty, variant, idx),
        _ => None,
    }
}

/// The declared element type one index step down from `ty`.
fn stepped_index_ty(ty: Option<&Ty>) -> Option<Ty> {
    match ty {
        Some(Ty::Array(elem, _)) | Some(Ty::Slice(elem)) => Some((**elem).clone()),
        _ => None,
    }
}

/// Whether `value` is a *complete* value of `ty` under the
/// program's declared field types: every declared field of an
/// aggregate is present and itself complete, and scalars match
/// in kind. A type the tables do not record cannot be judged
/// incomplete and counts as complete (the caller's shape-based
/// judgment then stands); a recorded type is judged strictly.
/// Completeness is what separates a wrapper's payload -- a
/// whole value of the type the wrapper wraps -- from a
/// partially materialised aggregate's first field, which is a
/// value of a strictly smaller type and so can never be
/// complete for the aggregate's own type.
fn value_complete_for_ty(program: &Program, value: &Value, ty: &Ty) -> bool {
    value_complete_for_ty_at(program, value, ty, 0)
}

fn value_complete_for_ty_at(
    program: &Program,
    value: &Value,
    ty: &Ty,
    depth: usize,
) -> bool {
    if depth > 4 {
        return true;
    }
    match (ty, value) {
        (Ty::Adt(name), Value::Enum { variant, fields }) => {
            match program
                .adt_field_tys
                .get(name)
                .and_then(|variants| variants.get(*variant as usize))
            {
                Some(declared) => {
                    fields.len() == declared.len()
                        && fields.iter().zip(declared.iter()).all(|(field, field_ty)| {
                            // A field whose declared type the
                            // table could not track (its head is
                            // a generic parameter) cannot be
                            // judged incomplete.
                            match field_ty {
                                Some(field_ty) => value_complete_for_ty_at(
                                    program,
                                    field,
                                    field_ty,
                                    depth + 1,
                                ),
                                None => true,
                            }
                        })
                }
                None => true,
            }
        }
        (Ty::Adt(name), Value::Unit) => match program.adt_field_tys.get(name) {
            Some(variants) => variants.iter().any(|declared| declared.is_empty()),
            None => true,
        },
        (Ty::Tuple(ts), Value::Tuple(items)) => {
            items.len() == ts.len()
                && items
                    .iter()
                    .zip(ts.iter())
                    .all(|(item, t)| value_complete_for_ty_at(program, item, t, depth + 1))
        }
        (Ty::Tuple(ts), Value::Unit) => ts.is_empty(),
        (Ty::Array(elem, n), Value::Tuple(items)) => {
            items.len() as u64 == *n
                && items
                    .iter()
                    .all(|item| value_complete_for_ty_at(program, item, elem, depth + 1))
        }
        (Ty::Uint(_), Value::Uint(_))
        | (Ty::Int(_), Value::Int(_))
        | (Ty::Bool, Value::Bool(_))
        | (Ty::Float(_), Value::Float(_) | Value::Float128(_))
        | (Ty::Unit, Value::Unit)
        | (Ty::Str, Value::Str(_) | Value::Bytes(_)) => true,
        (Ty::Ref(_), Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }) => true,
        // An aggregate type over a non-aggregate value (or the
        // wrong aggregate form) is a clear mismatch.
        (Ty::Adt(_) | Ty::Tuple(_), _) => false,
        _ => true,
    }
}

/// Whether the transparent-wrapper descent in
/// `navigate_mut_grow`'s field arm may fire for a slot holding
/// a one-field enum whose sole field is an aggregate: the
/// descent presumes the stored field is a wrapper layer around
/// the value the rest of the projection addresses. Type
/// evidence can refute that presumption, in one case: the
/// variant the stored value selects (the only variant, for a
/// struct) declares at least two fields, and the stored field
/// is not a *complete* value of that type. Then the stored
/// field is the selected variant's own first field and the
/// slot is that variant's value materialised field by field,
/// its remaining fields elided or not yet written: the step
/// belongs to the slot's own fields, which the ordinary field
/// arm grows into place (and a read answers through the
/// canonical-value fallback). The same judgment applies to
/// a tuple root, whose declared fields are its element types.
/// Every other declared shape
/// leaves the descent standing: a step into a variant with at
/// most one declared field can only address its sole field's
/// interior. With no type evidence the stored form decides,
/// as before.
/// The declared type of a wrapper's payload after a
/// sanctioned descent: the payload is a value of the slot's
/// declared type (the retargeted-pointer case), except when
/// the declared type is the one-field wrapper itself, whose
/// interior is its sole field's type. (The same computation
/// `navigate_mut_grow` performs inline at its descent.)
fn descent_payload_ty(
    program: Option<&Program>,
    ty: Option<&Ty>,
    slot: &Value,
) -> Option<Ty> {
    let slot_variant = match slot {
        Value::Enum { variant, .. } => *variant as usize,
        _ => 0,
    };
    match (program, ty) {
        (Some(program), Some(adt @ Ty::Adt(name)))
            if adt_variant_field_count(program, name, slot_variant) == Some(1) =>
        {
            adt_field_ty_in(program, adt, slot_variant, 0)
        }
        (_, Some(Ty::Tuple(ts))) if ts.len() == 1 => ts.first().cloned(),
        _ => ty.cloned(),
    }
}

fn wrapper_descent_sanctioned(
    program: Option<&Program>,
    ty: Option<&Ty>,
    slot: &Value,
) -> bool {
    let (
        Some(program),
        Some(ty),
        Value::Enum {
            variant, fields, ..
        },
    ) = (program, ty, slot)
    else {
        return true;
    };
    // The declared shape that decides is the *selected*
    // variant's field count (a tuple's element count, for a
    // tuple root: a tuple materialised field by field has the
    // same one-field stored form as a struct): the stored
    // form is a value of that variant, and the step addresses
    // its fields. Declaring exactly one field leaves the
    // descent standing. Declaring more fields makes the
    // one-field stored form that variant's own partially
    // materialised value, and the completeness check decides.
    // Declaring no fields at all refutes the descent: such a
    // type has no interior for the step to address and no
    // one-field stored form of its own.
    let (declared_len, slot_ty) = match ty {
        Ty::Adt(name) => {
            let Some(variants) = adt_field_tys_of(program, name) else {
                // The type is a named ADT the program's field
                // table does not record: there is no evidence
                // the stored form is a wrapper around a value
                // of this type rather than the type's own
                // partially materialised form, so the descent
                // does not stand. (With no type evidence at
                // all -- no program or a non-ADT, non-tuple
                // type -- the shape judgment above stands as
                // before.)
                return false;
            };
            let Some(selected) = variants.get(*variant as usize) else {
                // The stored variant is not among the
                // declared ones: there is no evidence the
                // stored form is a wrapper, so the descent
                // does not stand.
                return false;
            };
            (selected.len(), Ty::Adt(name.clone()))
        }
        Ty::Tuple(ts) => (ts.len(), Ty::Tuple(ts.clone())),
        _ => return true,
    };
    if declared_len == 0 {
        // A type declaring no fields has no sole field whose
        // interior the step could address, and its own value is
        // a unit, never a one-field aggregate: the evidence
        // cannot describe this slot. That is the shape a type
        // walk produces when it steps through a union's
        // unit-typed field (a `MaybeUninit`'s `uninit` field)
        // while the value walk selects the union's payload --
        // the two walks have parted, and the step belongs to
        // the slot's own stored fields, which the ordinary
        // field arm grows into place.
        return false;
    }
    if declared_len == 1 {
        return true;
    }
    value_complete_for_ty(program, &fields[0], &slot_ty)
}

pub(crate) fn assign_projected(
    slot: &mut Value,
    projection: &[RefProj],
    value: Value,
    func: &str,
    place: &str,
    grow: bool,
    program: Option<&Program>,
    ty: Option<Ty>,
) -> InterpResult<()> {
    // The byte-splice pre-passes below read the projection's
    // last step as a byte position inside the value the head
    // denotes; an element-position index run (see
    // `compose_projection`) must be composed first, or the
    // splice mistakes the run's last index for a byte offset
    // into the element at the run's head.
    let composed;
    let (projection, ty) = if projection_needs_composition(projection) {
        composed = compose_projection(slot, projection, func, place);
        if composed.as_slice() == projection {
            // Composition left the steps nested as written
            // (the element at the run's head is itself an
            // aggregate, so the steps do not compose): the
            // declared types still correspond step by step,
            // so the type evidence stands.
            (&composed[..], ty)
        } else {
            // A composed projection's steps no longer
            // correspond to the declared types step by step
            // (an index run flattened into one step spans
            // several element types), so the type walk only
            // follows an uncomposed projection.
            (&composed[..], None)
        }
    } else {
        (projection, ty)
    };
    // The pre-passes' head navigations carry the type evidence
    // the caller supplied (when the projection was not
    // composed): a shape-only head walk can take the wrapper
    // descent into a partially materialised aggregate's first
    // field and splice the byte into a sibling the write does
    // not address.
    if let Some((&RefProj::ByteOffset(disp), head)) = projection.split_last() {
        // A byte written through a pointer displaced into a
        // scalar's storage, with no index step after the
        // displacement (a one-byte bulk write, or the direct
        // write of a single element): the byte value being
        // written is what makes the position a byte position.
        if disp >= 0 {
            if let Some(byte) = value_as_storage_byte(&value) {
                if let Ok(head_slot) = navigate_mut_grow(
                    slot,
                    head,
                    func,
                    place,
                    grow,
                    program,
                    ty.clone(),
                ) {
                    // The head slot's declared kind, when the
                    // type walk follows the head projection: a
                    // float storage narrower than the model
                    // form takes the byte at its own width
                    // (see `splice_scalar_storage_byte_as`).
                    let head_kind = byte_scalar_of_ty(
                        step_ty_across(program, ty.clone(), head).as_ref(),
                    );
                    if splice_scalar_storage_byte_as(head_slot, head_kind, disp as usize, byte)
                    {
                        return Ok(());
                    }
                }
            }
        }
    }
    if let Some((&RefProj::Index(idx), head)) = projection.split_last() {
        // A byte pointer displaced into the storage first
        // (`p.add(k)`) puts a byte-offset step immediately before
        // the index step; the byte position is the sum.
        let (idx, head, byte_displaced) = match head.split_last() {
            Some((&RefProj::ByteOffset(disp), head)) if disp >= 0 => {
                (idx + disp as usize, head, true)
            }
            _ => (idx, head, false),
        };
        if let Some(byte) = value_as_storage_byte(&value) {
            if let Ok(head_slot) = navigate_mut_grow(
                slot,
                head,
                func,
                place,
                grow,
                program,
                ty.clone(),
            ) {
                // The splice is a byte position inside a scalar's
                // storage. It applies when the projection is
                // byte-displaced (its shape says so outright), or
                // when the head slot actually holds a scalar. An
                // unwritten head slot is not evidence of a scalar:
                // it is just as often an array field no write has
                // materialised yet (a fresh node's key area being
                // filled element by element), and splicing the
                // first elements into a fabricated scalar there
                // would pack them into one integer and lose the
                // array. Such a write belongs to the ordinary
                // navigation below, which materialises the array.
                // The head slot's declared kind rides along as
                // at the byte-offset splice above.
                let head_kind = byte_scalar_of_ty(
                    step_ty_across(program, ty.clone(), head).as_ref(),
                );
                if (byte_displaced || scalar_storage_len(head_slot).is_some())
                    && splice_scalar_storage_byte_as(head_slot, head_kind, idx, byte)
                {
                    return Ok(());
                }
            }
        }
    }
    if let Some((&RefProj::VecLen, head)) = projection.split_last() {
        // A write to a vector's length slot (see
        // [`RefProj::VecLen`]): the slot is not storage inside
        // the element-list value, so the write applies the
        // `Vec::set_len` storage change to the vector value
        // the head denotes.
        let head_slot =
            navigate_mut_grow(slot, head, func, place, grow, program, ty.clone())?;
        return assign_vec_len(head_slot, value, place);
    }
    let target = navigate_mut_grow(slot, projection, func, place, grow, program, ty)?;
    *target = value;
    Ok(())
}

/// A write to a vector's length slot (see [`RefProj::VecLen`]),
/// applied to the vector value itself: the declared length
/// changes and the stored elements stay — the tail past the
/// new length is spare capacity a later append overwrites,
/// exactly the storage change the place-shaped write in
/// `Interpreter::write_vec_len_slot` performs for a direct
/// place. The growth padding here is the bare uninitialised
/// marker rather than the element type's shaped uninitialised
/// form the place-shaped write derives from the vector's
/// element tag: this projection is type-free, and the slots a
/// growth declares are the ones the guest writes before it
/// reads them (the length a `SetLenOnDrop` guard writes
/// covers slots its fill loop already stored), so the
/// padding's shape is observable only to a read of a
/// never-written slot, which is an uninitialised read either
/// way.
/// Update a vector value's recorded length to `new_len`
/// without rebuilding its storage, when the storage form
/// allows it: the real aggregate form's length field is
/// rewritten, and a sparse vector whose stored elements
/// already cover `new_len` (the state `Vec::reserve` leaves
/// behind, which the `extend_with` fill loop then grows
/// through one element at a time) just records the new
/// length — its capacity already accounts for the rest.
/// Returns whether the update was applied.
fn set_vec_len_in_place(slot: &mut Value, new_len: u128) -> bool {
    if crate::builtins::vec_real_aggregate_parts(slot).is_some() {
        if let Value::Enum { fields, .. } = slot {
            fields[1] = Value::Uint(new_len);
            return true;
        }
        return false;
    }
    match slot {
        Value::SparseVec { elems, len, .. } if new_len <= elems.len() as u128 => {
            *len = new_len;
            true
        }
        _ => false,
    }
}

pub(crate) fn assign_vec_len(slot: &mut Value, value: Value, place: &str) -> InterpResult<()> {
    let found = value.kind_name();
    let new_len = match value {
        Value::Uint(n) => usize::try_from(n).ok(),
        Value::Int(n) if n >= 0 => usize::try_from(n).ok(),
        _ => None,
    }
    .ok_or_else(|| {
        InterpError::type_mismatch(
            format!("vector length slot of {place}"),
            "integer length",
            found,
        )
    })?;
    if set_vec_len_in_place(slot, new_len as u128) {
        return Ok(());
    }
    // A vector in its real aggregate form holds the length in
    // the slot itself: the store lands on that field, and the
    // elements (which live behind the buffer pointer) are
    // untouched.
    if let Some((buf, _)) = crate::builtins::vec_real_aggregate_parts(slot) {
        *slot = Value::Enum {
            variant: 0,
            fields: vec![buf, Value::Uint(new_len as u128)],
        };
        return Ok(());
    }
    let prev_cap = crate::builtins::vec_stored_len(slot).unwrap_or(0);
    let elems = match std::mem::replace(slot, Value::Uninit) {
        Value::Enum { fields, .. } => fields,
        Value::SparseVec { elems, .. } => elems,
        other => {
            return Err(InterpError::type_mismatch(
                format!("vector length slot of {place}"),
                "vector value",
                other.kind_name(),
            ));
        }
    };
    *slot = crate::builtins::vec_with_len(elems, new_len, &Value::Uninit, prev_cap);
    Ok(())
}

/// Project through `projection` by reference: the borrowing form
/// of [`project_value`]. Steps that only navigate (field, index
/// into an aggregate, downcast) lend the sub-value out of the
/// input; steps that construct a fresh value (byte/string indexing,
/// sub-slices) materialise it and finish the remaining steps with
/// the owning projection. Errors are exactly those of
/// `project_value`.
pub(crate) fn project_value_ref<'a>(
    value: &'a Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<Cow<'a, Value>> {
    project_value_ref_ev(None, None, value, projection, func, place)
}

/// [`project_value_ref`] with type evidence, as in
/// `project_value_ev`.
pub(crate) fn project_value_ref_ev<'a>(
    program: Option<&Program>,
    root_ty: Option<Ty>,
    value: &'a Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<Cow<'a, Value>> {
    let composed;
    let projection = if projection_needs_composition(projection) {
        composed = compose_projection(value, projection, func, place);
        &composed[..]
    } else {
        projection
    };
    let mut cur: &'a Value = value;
    let mut cur_ty = root_ty;
    let mut rest = projection;
    // The state a coroutine downcast named, for the field step
    // that follows it (see the `Downcast` arm below).
    let mut downcast_state: Option<u32> = None;
    while let Some((step, tail)) = rest.split_first() {
        if *cur == Value::Uninit {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        }
        rest = tail;
        let step_state = downcast_state.take();
        // The declared type of the value in hand, before this
        // step moves it: the index arm's single-array-field pun
        // consults it (see `project_value_ev`).
        let step_base_ty = cur_ty.clone();
        // The declared type steps alongside the value (see
        // `project_value_ev`); the field step steps inside its
        // arm, past its wrapper descent.
        if !matches!(step, RefProj::Field(_)) {
            cur_ty = match step {
                RefProj::Index(_) => stepped_index_ty(cur_ty.as_ref()),
                RefProj::UnionField { idx, .. } => {
                    stepped_field_ty(program, cur_ty.as_ref(), 0, *idx)
                }
                RefProj::Downcast(_) => cur_ty,
                _ => None,
            };
        }
        match step {
            RefProj::Field(idx) => {
                // The transparent-wrapper descent of
                // `project_value`, in borrowing form: retry the
                // step on the wrapper's only field.
                while let Value::Enum { fields, .. } = cur {
                    if fields.len() == 1
                        && fields.get(*idx).is_none()
                        && wrapper_payload(&fields[0])
                        && wrapper_descent_sanctioned(program, cur_ty.as_ref(), cur)
                    {
                        cur_ty = descent_payload_ty(program, cur_ty.as_ref(), cur);
                        cur = &fields[0];
                    } else {
                        break;
                    }
                }
                let variant = match cur {
                    Value::Enum { variant, .. } => *variant as usize,
                    _ => 0,
                };
                cur = match cur {
                    Value::Tuple(items) => items.get(*idx).ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?,
                    Value::Enum { fields, .. } => fields.get(*idx).ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?,
                    // A sparse vector value, as in
                    // `project_value`: a field step addresses an
                    // element (the spare-capacity tail past the
                    // declared length is stored storage and lends
                    // as the rest does); the unmaterialised tail
                    // has no storage to lend, so the walk
                    // continues into an owned uninitialised slot.
                    Value::SparseVec { elems, cap, .. } => {
                        if *idx as u128 >= (*cap).max(elems.len() as u128) {
                            return Err(InterpError::UninitializedRead {
                                func: func.to_string(),
                                place: place.to_string(),
                            });
                        }
                        match elems.get(*idx) {
                            Some(v) => v,
                            None => {
                                return Ok(Cow::Owned(project_value(
                                    Value::Uninit,
                                    rest,
                                    func,
                                    place,
                                )?));
                            }
                        }
                    }
                    Value::Coroutine { fields, saved, .. } => {
                        // As in `project_value`: a downcast routes
                        // the field step to that state's saved
                        // storage, otherwise (and for state 0, the
                        // unresumed state) the upvar storage.
                        let storage = match step_state {
                            Some(0) | None => Some(fields),
                            Some(state) => saved.get(state as usize),
                        };
                        storage
                            .and_then(|fields| fields.get(*idx))
                            .ok_or_else(|| InterpError::UninitializedRead {
                                func: func.to_string(),
                                place: place.to_string(),
                            })?
                    }
                    Value::Closure { env, .. } => env.get(*idx).ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?,
                    // A reference-counted handle in trait-object
                    // form, as in `project_value`: the field step
                    // reads the handle's own field, so the walk
                    // continues into the base handle with this
                    // step and the remaining ones.
                    Value::DynRef { base, .. }
                        if is_rc_arc_handle(base) =>
                    {
                        let mut sub: Vec<RefProj> =
                            vec![RefProj::Field(*idx)];
                        sub.extend_from_slice(rest);
                        return project_value_ref(base, &sub, func, place);
                    }
                    Value::DynRef { .. } if *idx == 0 => cur,
                    // The thin-pointer wrapper chain, as in
                    // `project_value`: a `Box`'s field 0 is its
                    // `Unique`, whose field 0 is the `NonNull`,
                    // whose field 0 is the raw pointer -- and in
                    // this model the box's one field is the
                    // pointer value itself, so each of those steps
                    // yields it unchanged. (A reference to a place
                    // never sits at a field step of a value
                    // projection: place references are resolved
                    // before value projection begins, so a `Ref`
                    // here is a pointer value extracted from a
                    // wrapper -- the box stored inside another
                    // aggregate is the case that reaches it.)
                    Value::Ref(_) if *idx == 0 => cur,
                    // The wrapper's zero-sized `PhantomData`
                    // marker field, as in `project_value`: the
                    // model keeps no storage for it, so the step
                    // materializes the marker value as an owned
                    // value, like the synthesized `fmt::Arguments`
                    // fields below.
                    Value::Ref(_) => {
                        let owned = Value::Enum {
                            variant: 0,
                            fields: Vec::new(),
                        };
                        if tail.is_empty() {
                            return Ok(Cow::Owned(owned));
                        }
                        return Ok(Cow::Owned(
                            project_value_ref(&owned, tail, func, place)?
                                .into_owned(),
                        ));
                    }
                    // A one-field wrapper over a scalar whose
                    // storage the model holds as the bare scalar,
                    // as in `project_value`: field 0 of the
                    // wrapper is the scalar it wraps.
                    Value::Int(_) | Value::Uint(_) | Value::Float(_)
                    | Value::Bool(_)
                        if *idx == 0 =>
                    {
                        cur
                    }
                    // The byte-string wrapper chain, as in
                    // `project_value`: field 0 of the transparent
                    // `Path`/`OsStr`/`CStr` wrappers over a string
                    // value is the string itself, and over a fat
                    // pointer (the non-UTF-8 form of the same
                    // values) it is the fat pointer itself.
                    Value::Str(_) | Value::Bytes(_) | Value::FatRef { .. }
                        if *idx == 0 =>
                    {
                        cur
                    }
                    Value::FmtArguments(_) => {
                        // The modelled `fmt::Arguments` fields are
                        // synthesized values, not storage (see the
                        // matching arm in `project_value`), so the
                        // projection materializes an owned field and
                        // any remaining steps project into it.
                        let owned = project_value(
                            cur.clone(),
                            std::slice::from_ref(step),
                            func,
                            place,
                        )?;
                        if tail.is_empty() {
                            return Ok(Cow::Owned(owned));
                        }
                        return Ok(Cow::Owned(
                            project_value_ref(&owned, tail, func, place)?.into_owned(),
                        ));
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("field projection on {place}"),
                            "tuple or ADT value",
                            other.kind_name(),
                        ))
                    }
                };
                cur_ty = stepped_field_ty(program, cur_ty.as_ref(), variant, *idx);
            }
            RefProj::Index(idx) => match cur {
                Value::FatRef { base, len } => {
                    if *idx as u128 >= *len {
                        return Err(index_oob(*len as usize, *idx));
                    }
                    match project_value_ref(base, std::slice::from_ref(step), func, place)? {
                        Cow::Borrowed(v) => cur = v,
                        Cow::Owned(v) => {
                            return Ok(Cow::Owned(project_value(v, rest, func, place)?));
                        }
                    }
                }
                // A sparse vector value, as in `project_value`:
                // the stored elements lend their storage (the
                // spare-capacity tail past the declared length
                // included), the unmaterialised tail continues
                // into an owned uninitialised slot.
                Value::SparseVec { elems, cap, .. } => {
                    let extent = (*cap).max(elems.len() as u128);
                    if *idx as u128 >= extent {
                        return Err(index_oob(extent as usize, *idx));
                    }
                    match elems.get(*idx) {
                        Some(v) => {
                            cur = v;
                        }
                        None => {
                            return Ok(Cow::Owned(project_value(
                                Value::Uninit,
                                rest,
                                func,
                                place,
                            )?));
                        }
                    }
                }
                Value::Tuple(items) => {
                    cur = items.get(*idx).ok_or_else(|| index_oob(items.len(), *idx))?;
                }
                Value::Enum { fields, .. }
                    if fields.len() == 1
                        && matches!(
                            fields[0],
                            Value::Tuple(_) | Value::Bytes(_)
                        )
                        && !matches!(
                            &step_base_ty,
                            Some(ty) if is_vec_adt_ty(ty)
                        ) =>
                {
                    // The single-array-field pun, as in
                    // `project_value`: the index step continues
                    // into the field's elements. A base the
                    // type evidence names as a `Vec` is the
                    // element aggregate instead (see
                    // `project_value_ev`), so the pun is gated
                    // on the evidence as there.
                    match project_value_ref(
                        &fields[0],
                        std::slice::from_ref(step),
                        func,
                        place,
                    )? {
                        Cow::Borrowed(v) => cur = v,
                        Cow::Owned(v) => {
                            return Ok(Cow::Owned(project_value(
                                v, rest, func, place,
                            )?));
                        }
                    }
                }
                Value::Enum { fields, .. } => {
                    cur = fields.get(*idx).ok_or_else(|| index_oob(fields.len(), *idx))?;
                }
                Value::Bytes(b) => {
                    let byte = b.get(*idx).ok_or_else(|| index_oob(b.len(), *idx))?;
                    let v = Value::Uint(*byte as u128);
                    return Ok(Cow::Owned(project_value(v, rest, func, place)?));
                }
                Value::Str(text) => {
                    let byte = text
                        .as_bytes()
                        .get(*idx)
                        .ok_or_else(|| index_oob(text.len(), *idx))?;
                    let v = Value::Uint(*byte as u128);
                    return Ok(Cow::Owned(project_value(v, rest, func, place)?));
                }
                Value::Bool(_)
                | Value::Int(_)
                | Value::Uint(_)
                | Value::Float(_)
                | Value::Float128(_) => {
                    // A scalar's storage byte, as in
                    // `project_value` (see `scalar_storage_byte`),
                    // under the declared type in hand (see
                    // `scalar_storage_byte_as`).
                    let kind = byte_scalar_of_ty(step_base_ty.as_ref());
                    let len = scalar_storage_len_as(cur, kind).expect("scalar storage");
                    if *idx >= len {
                        return Err(index_oob(len, *idx));
                    }
                    let byte = scalar_storage_byte_as(cur, kind, *idx)
                        .expect("index inside the storage");
                    let v = Value::Uint(u128::from(byte));
                    return Ok(Cow::Owned(project_value(v, rest, func, place)?));
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("index projection on {place}"),
                        "array, byte or string value",
                        other.kind_name(),
                    ))
                }
            },
            RefProj::Subslice { offset, len } => {
                let (start, end) = (*offset, *offset + *len);
                let v = match cur {
                    Value::Tuple(items) => {
                        check_subslice(items.len(), start, *len)?;
                        Value::Tuple(items[start..end].to_vec())
                    }
                    Value::Bytes(b) => {
                        check_subslice(b.len(), start, *len)?;
                        Value::Bytes(b[start..end].to_vec())
                    }
                    Value::Str(text) => match text.get(start..end) {
                        Some(sub) => Value::Str(sub.to_string()),
                        None => {
                            return Err(InterpError::panic(format!(
                                "byte range {start}..{end} is not a valid slice of the string at {place}"
                            )))
                        }
                    },
                    Value::Enum { fields, .. } => {
                        check_subslice(fields.len(), start, *len)?;
                        Value::Tuple(fields[start..end].to_vec())
                    }
                    Value::SparseVec {
                        elems,
                        len: vec_len,
                        ..
                    } => {
                        let total =
                            usize::try_from(*vec_len).unwrap_or(usize::MAX);
                        check_subslice(total, start, *len)?;
                        Value::Tuple(
                            (start..end)
                                .map(|i| {
                                    elems.get(i).cloned().unwrap_or(Value::Uninit)
                                })
                                .collect(),
                        )
                    }
                    Value::FatRef {
                        base,
                        len: fat_len,
                    } => {
                        if start == 0 {
                            if end as u128 > *fat_len {
                                return Err(InterpError::panic(format!(
                                    "range end index {end} out of range for slice of length {fat_len}"
                                )));
                            }
                            Value::FatRef {
                                base: base.clone(),
                                len: end as u128,
                            }
                        } else {
                            match &**base {
                                Value::Tuple(_) | Value::Bytes(_) | Value::Str(_) => {
                                    project_value_ref(
                                        base,
                                        std::slice::from_ref(step),
                                        func,
                                        place,
                                    )?
                                    .into_owned()
                                }
                                other => {
                                    return Err(InterpError::type_mismatch(
                                        format!("subslice projection on {place}"),
                                        "materialised slice value",
                                        other.kind_name(),
                                    ))
                                }
                            }
                        }
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("subslice projection on {place}"),
                            "array, slice or string value",
                            other.kind_name(),
                        ))
                    }
                };
                return Ok(Cow::Owned(project_value(v, rest, func, place)?));
            }
            RefProj::Downcast(variant) => match cur {
                Value::Coroutine { .. } => {
                    downcast_state = Some(*variant);
                }
                Value::Enum {
                    variant: current, ..
                } => {
                    if current != variant {
                        return Err(InterpError::type_mismatch(
                            format!("downcast of {place}"),
                            format!("variant {variant}"),
                            format!("variant {current}"),
                        ));
                    }
                }
                Value::Tuple(_) if *variant == 0 => {}
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("downcast of {place}"),
                        "ADT value",
                        other.kind_name(),
                    ))
                }
            },
            RefProj::UnionField { .. } => {
                // Same contract as `project_value`: union field
                // reads need the program's scalar shapes and route
                // through the typed projector (`project_value_t`).
                return Err(InterpError::type_mismatch(
                    format!("union field projection on {place}"),
                    "a union field read through the typed projection path",
                    "the untyped value projector",
                ));
            }
            RefProj::VecLen => {
                // The length slot is not storage inside the
                // vector value, so the step materialises the
                // declared length, as the unmaterialised tail
                // in the field arm above materialises its slot.
                let len = vec_value_len(cur).ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("vector length slot of {place}"),
                        "vector value",
                        cur.kind_name(),
                    )
                })?;
                return Ok(Cow::Owned(project_value(
                    Value::Uint(len),
                    rest,
                    func,
                    place,
                )?));
            }
            RefProj::ByteOffset(disp) => {
                if *disp != 0 {
                    // A byte read through a pointer displaced into
                    // a scalar's storage, as in `project_value`:
                    // the index step that follows selects the byte
                    // at the combined offset.
                    if *disp >= 0 {
                        let kind = byte_scalar_of_ty(step_base_ty.as_ref());
                        if let (Some(len), Some(&RefProj::Index(k))) =
                            (scalar_storage_len_as(cur, kind), rest.first())
                        {
                            let at = *disp as usize + k;
                            let byte = scalar_storage_byte_as(cur, kind, at)
                                .ok_or_else(|| index_oob(len, at))?;
                            let v = Value::Uint(u128::from(byte));
                            return Ok(Cow::Owned(project_value(v, &rest[1..], func, place)?));
                        }
                        // An element block's aggregate at the
                        // displaced byte position, as in
                        // `project_value`.
                        if !matches!(rest.first(), Some(RefProj::Index(_))) {
                            if let Value::Tuple(items) = cur {
                                if let Some(elem) = items
                                    .get(*disp as usize)
                                    .filter(|elem| {
                                        matches!(
                                            elem,
                                            Value::Enum { .. } | Value::Tuple(_)
                                        )
                                    })
                                    .cloned()
                                {
                                    return Ok(Cow::Owned(project_value(
                                        elem, rest, func, place,
                                    )?));
                                }
                            }
                        }
                    }
                    // The position does not coincide with a value
                    // boundary, and the model has no byte-level
                    // view of a value's storage to reinterpret.
                    return Err(InterpError::UnsupportedProjection {
                        kind: format!(
                            "read through a pointer displaced {disp} bytes from a value boundary"
                        ),
                    });
                }
            }
        }
    }
    Ok(Cow::Borrowed(cur))
}

/// Navigate a projection mutably inside a stored value and return the
/// final slot (write direction).
pub(crate) fn navigate_mut<'x>(
    slot: &'x mut Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<&'x mut Value> {
    navigate_mut_grow(slot, projection, func, place, false, None, None)
}

/// [`navigate_mut`] with the sequence-growth policy explicit.
/// Growth -- extending a tuple slot past its stored length
/// with uninitialized slots on an index write -- is the
/// sparse-prefix convention for allocation storage, whose
/// stored form is a prefix of an extent the model does not
/// record. A place local's stored form is the complete value,
/// so an index write past its length there is the bounds
/// violation the guest's own index semantics name; only
/// allocation writes grow.
///
/// `ty` is the declared type of the value in `slot`, when the
/// caller knows it; the walk steps it alongside the projection
/// (through the program's field-type table) so each slot's own
/// declared type rides with it, and the wrapper descent in the
/// field arm consults it (see `wrapper_descent_sanctioned`).
/// Steps whose interior the declared types do not describe
/// (byte positions, vector storage, coroutine state) drop the
/// walk, and the judgments there stay shape-only.
pub(crate) fn navigate_mut_grow<'x>(
    slot: &'x mut Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
    grow: bool,
    program: Option<&Program>,
    ty: Option<Ty>,
) -> InterpResult<&'x mut Value> {
    if projection.is_empty() {
        return Ok(slot);
    }
    let composed;
    let (projection, ty) = if projection_needs_composition(projection) {
        composed = compose_projection(slot, projection, func, place);
        // As in `assign_projected`: the type walk follows the
        // projection when composition left it unchanged, and
        // only a rewritten projection drops the evidence.
        if composed.as_slice() == projection {
            (&composed[..], ty)
        } else {
            (&composed[..], None)
        }
    } else {
        (projection, ty)
    };
    if *slot == Value::Uninit {
        // Partial initialization: MIR fills an aggregate place
        // field by field — a `catch_unwind` Data union written
        // through a raw pointer by the callee, a struct local
        // stored field-wise — so a field, union-field or downcast
        // write into a never-written slot materializes the
        // aggregate with every other slot still uninitialized, the
        // same sparse shape a union aggregate value has. The slots
        // the write does not touch keep raising
        // `UninitializedRead` when read, so no value is fabricated.
        // An index step into an uninitialized place is sequence
        // storage that was never built (an array field of a
        // partially materialised aggregate): it materializes as
        // the sparse prefix below. Any other step has no honest
        // shape to materialize and stays an error.
        match projection.first() {
            Some(RefProj::Field(idx)) | Some(RefProj::UnionField { idx, .. }) => {
                *slot = Value::Enum {
                    variant: 0,
                    fields: vec![Value::Uninit; *idx + 1],
                };
            }
            Some(RefProj::Downcast(variant)) => {
                *slot = Value::Enum {
                    variant: *variant,
                    fields: Vec::new(),
                };
            }
            Some(RefProj::Index(idx)) => {
                // An index write into a never-built slot: the
                // slot is sequence storage (an array field of a
                // partially materialised aggregate, written
                // element-first through a raw pointer, is the
                // canonical case), and its honest shape is the
                // sparse prefix the rest of the model uses for
                // sequence storage -- the slots up to the written
                // index, all uninitialized -- the same
                // materialisation an element block's first write
                // performs in `write_resolved`. Bounded by the
                // same materialisation limit as a sparse
                // vector's tail; past it the write stays an
                // uninitialized error rather than an allocation
                // that exhausts the host.
                if *idx >= crate::builtins::MAX_VEC_SET_LEN_ELEMENTS {
                    return Err(InterpError::UninitializedRead {
                        func: func.to_string(),
                        place: place.to_string(),
                    });
                }
                *slot = Value::Tuple(vec![Value::Uninit; *idx + 1]);
            }
            _ => {
                return Err(InterpError::UninitializedRead {
                    func: func.to_string(),
                    place: place.to_string(),
                });
            }
        }
    }
    let (step, rest) = projection.split_first().expect("non-empty");
    match step {
        RefProj::VecLen => {
            // The length slot is not storage inside the vector
            // value, so there is no slot to lend: whole-value
            // writes to it go through `assign_projected`'s
            // length-slot pre-pass, and read-modify-write
            // through `Interpreter::modify_resolved`'s.
            return Err(InterpError::type_mismatch(
                format!("vector length slot of {place}"),
                "a slot inside the vector value",
                "the vector's length slot",
            ));
        }
        RefProj::UnionField { idx, .. } => {
            // A union holds one field's value: writing field
            // `idx` makes it the held field, so every other slot
            // is cleared (a later read of another field then
            // reinterprets this field's bits; see
            // `union_field_read`). Writing *inside* the field
            // (`u.a.0 = ...`) navigates into the held value as
            // usual. When no field holds a value yet -- the
            // storage is fresh, every slot uninitialized -- the
            // write inside field `idx` is partial initialization
            // of that field: the recursion below materializes
            // the field's storage sparsely, exactly as a write
            // into a never-written slot does at the head of this
            // function, and the slots the write does not touch
            // stay uninitialized. But when another field DOES
            // hold the union's value, a field that holds no value
            // has nothing to write inside of: the bytes a native
            // write would preserve there belong to the held
            // field's representation, which this model does not
            // hold as bytes, so that stays an uninitialized
            // error rather than a fabricated partial value.
            let Value::Enum { fields, .. } = slot else {
                return Err(InterpError::type_mismatch(
                    format!("union field write on {place}"),
                    "union (ADT) value",
                    slot.kind_name(),
                ));
            };
            if fields.len() <= *idx {
                fields.resize(*idx + 1, Value::Uninit);
            }
            if !rest.is_empty()
                && fields[*idx] == Value::Uninit
                && fields
                    .iter()
                    .enumerate()
                    .any(|(i, field)| i != *idx && *field != Value::Uninit)
            {
                return Err(InterpError::UninitializedRead {
                    func: func.to_string(),
                    place: place.to_string(),
                });
            }
            for (i, field) in fields.iter_mut().enumerate() {
                if i != *idx {
                    *field = Value::Uninit;
                }
            }
            navigate_mut_grow(
                &mut fields[*idx],
                rest,
                func,
                place,
                grow,
                program,
                stepped_field_ty(program, ty.as_ref(), 0, *idx),
            )
        }
        RefProj::ByteOffset(disp) => {
            if *disp == 0 {
                navigate_mut_grow(slot, rest, func, place, grow, program, ty)
            } else {
                Err(InterpError::UnsupportedProjection {
                    kind: format!(
                        "write through a pointer displaced {disp} bytes from a value boundary"
                    ),
                })
            }
        }
        RefProj::Field(idx) => {
            if let Value::DynRef { base, .. } = slot {
                if is_rc_arc_handle(base) {
                    // A reference-counted handle in trait-object
                    // form, as in `project_value`: the step applies
                    // to the handle aggregate the base holds, whose
                    // field 0 is its `ptr`.
                    let Value::DynRef { base, .. } = slot else {
                        unreachable!("shape checked by the guard")
                    };
                    return navigate_mut_grow(base, projection, func, place, grow, None, None);
                }
            }
            if *idx == 0
                && matches!(
                    slot,
                    Value::Str(_) | Value::Bytes(_) | Value::FatRef { .. }
                )
            {
                // The byte-string wrapper chain, as in
                // `project_value`: a write to field 0 of the
                // transparent `Path`/`OsStr` wrappers over a
                // string value lands on the string itself, and
                // over a fat pointer (the non-UTF-8 form of the
                // same values) on the fat pointer itself.
                return navigate_mut_grow(
                    slot,
                    rest,
                    func,
                    place,
                    grow,
                    program,
                    stepped_field_ty(program, ty.as_ref(), 0, 0),
                );
            }
            if *idx == 0
                && matches!(
                    slot,
                    Value::Ref(_)
                        | Value::DynRef { .. }
                        | Value::Int(_)
                        | Value::Uint(_)
                        | Value::Float(_)
                        | Value::Bool(_)
                )
            {
                // The one-field wrapper chains `project_value`
                // reads through: a write to field 0 of a thin
                // pointer value (a `Box`'s `Unique`/`NonNull`
                // chain, whose one field is the pointer itself),
                // of a trait-object pointer (the `DynRef` *is*
                // the box's pointer field, metadata included, as
                // in `project_value`), or of a wrapper over a
                // bare scalar lands on the held value, which is
                // the slot itself.
                return navigate_mut_grow(slot, rest, func, place, grow, None, None);
            }
            if let Value::Coroutine { fields, .. } = slot {
                // The state storage grows to the written slot: a
                // state field stored for the first time extends the
                // storage, and slots skipped over are uninitialized,
                // like a field that has not been stored yet.
                if fields.len() <= *idx {
                    fields.resize(*idx + 1, Value::Uninit);
                }
                return navigate_mut_grow(&mut fields[*idx], rest, func, place, grow, None, None);
            }
            // The field vector of the value in the slot may be
            // sparse or invalidated: a union value covers only
            // the fields written so far (see the union arm of
            // aggregate evaluation), and an enum place being
            // (re)initialized in place has no live payload slots
            // for the variant being written (SetDiscriminant
            // invalidated the old payload; a downcast write to a
            // different variant rematerialized it empty). Grow
            // the vector with Uninit padding up to the written
            // index; slots never written keep reading as
            // uninitialized.
            // The transparent-wrapper descent of
            // `project_value`, in writing form: a field step
            // past the end of a one-field wrapper whose only
            // field is an aggregate continues into that field
            // with the whole projection, instead of growing
            // the wrapper with padding it does not have.
            if matches!(
                slot,
                Value::Enum { fields, .. }
                    if fields.len() == 1
                        && fields.len() <= *idx
                        && wrapper_payload(&fields[0])
            ) {
                // The shape alone does not prove a wrapper: a
                // struct being initialized field by field has
                // exactly this shape once its first (aggregate)
                // field is stored, and descending there would
                // write the next field inside the first one.
                // Type evidence arbitrates (see
                // `wrapper_descent_sanctioned`); without it the
                // descent stands, as before. When the evidence
                // refutes the wrapper, execution falls through
                // to the field arm below, which grows the value's
                // own field vector into place.
                if wrapper_descent_sanctioned(program, ty.as_ref(), slot) {
                    let slot_variant = match slot {
                        Value::Enum { variant, .. } => *variant as usize,
                        _ => unreachable!("descent checked the shape"),
                    };
                    // The type the rest of the projection
                    // addresses inside the payload: the payload
                    // is a value of the slot's declared type (the
                    // retargeted-pointer case), except when the
                    // declared type is the one-field wrapper
                    // itself, whose interior is its sole field's
                    // type.
                    let payload_ty = match (program, &ty) {
                        (Some(program), Some(adt @ Ty::Adt(name)))
                            if adt_variant_field_count(program, name, slot_variant)
                                == Some(1) =>
                        {
                            adt_field_ty_in(program, adt, slot_variant, 0)
                        }
                        (_, Some(Ty::Tuple(ts))) if ts.len() == 1 => {
                            ts.first().cloned()
                        }
                        _ => ty.clone(),
                    };
                    let Value::Enum { fields, .. } = slot else {
                        unreachable!("descent checked the shape")
                    };
                    return navigate_mut_grow(
                        &mut fields[0],
                        projection,
                        func,
                        place,
                        false,
                        program,
                        payload_ty,
                    );
                }
            }
            if let Value::Enum { fields, variant } = slot {
                let variant = *variant as usize;
                if fields.len() <= *idx {
                    fields.resize(*idx + 1, Value::Uninit);
                }
                return navigate_mut_grow(
                    &mut fields[*idx],
                    rest,
                    func,
                    place,
                    grow,
                    program,
                    stepped_field_ty(program, ty.as_ref(), variant, *idx),
                );
            }
            let sub_ty = if matches!(&*slot, Value::Tuple(_)) {
                stepped_field_ty(program, ty.as_ref(), 0, *idx)
            } else {
                None
            };
            let items = match slot {
                Value::Tuple(items) => items,
                Value::Closure { env, .. } => env,
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("field write on {place}"),
                        "tuple or ADT value",
                        other.kind_name(),
                    ));
                }
            };
            let item = items.get_mut(*idx).ok_or_else(|| InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            })?;
            navigate_mut_grow(item, rest, func, place, grow, program, sub_ty)
        }
        RefProj::Index(idx) => match slot {
            Value::Tuple(items) => {
                // A tuple slot is sequence storage (an array or
                // an element block). Allocation storage grows
                // past its stored prefix with uninitialized
                // slots (see `navigate_mut_grow`); anywhere else
                // the stored length is the value's length and an
                // index past it is a bounds violation.
                if *idx >= items.len() {
                    if !grow || *idx >= crate::builtins::MAX_VEC_SET_LEN_ELEMENTS {
                        return Err(index_oob(items.len(), *idx));
                    }
                    items.resize(*idx + 1, Value::Uninit);
                }
                navigate_mut_grow(
                    &mut items[*idx],
                    rest,
                    func,
                    place,
                    grow,
                    program,
                    stepped_index_ty(ty.as_ref()),
                )
            }
            // A `Vec` value is the aggregate of its elements:
            // indexing the vector's own place writes the element
            // field (the read direction in `project_value`). A
            // write at the index one past the last element is
            // the append point — the tail write a `Vec::append`
            // performs through the buffer pointer before it
            // declares the new length. It grows the STORAGE
            // only: the value first takes the sparse form (see
            // `Value::SparseVec`), whose declared length the
            // write leaves untouched, and the sparse arm below
            // performs the write — a length is only ever
            // declared by the length bookkeeping (a length-slot
            // write, `set_len`, the modelled appends), never by
            // a raw write, and the read direction stays bounded
            // by the declared length for the same reason. The
            // single-array-field pun is exempt: its index steps
            // range over the field's elements, not the
            // aggregate's.
            Value::Enum { .. } => {
                // A slot the type evidence names as a `Vec`
                // holds the element aggregate, never the
                // single-array-field pun's wrapper (see
                // `project_value_ev`): neither the pun descent
                // below nor its exemption from the append point
                // applies to it.
                let vec_typed =
                    matches!(&ty, Some(t) if is_vec_adt_ty(t));
                let append_point = match slot {
                    Value::Enum { fields, .. } => {
                        *idx == fields.len()
                            && (vec_typed
                                || !(fields.len() == 1
                                    && matches!(
                                        fields[0],
                                        Value::Tuple(_) | Value::Bytes(_)
                                    )))
                    }
                    _ => false,
                };
                if append_point {
                    let value = std::mem::replace(slot, Value::Uninit);
                    let Value::Enum { fields, .. } = value else {
                        unreachable!("append point checked above")
                    };
                    let len = fields.len();
                    *slot = Value::SparseVec {
                        elems: fields,
                        len: len as u128,
                        cap: len as u128,
                    };
                    return navigate_mut(slot, projection, func, place);
                }
                let Value::Enum { fields, .. } = slot else {
                    unreachable!("enum arm")
                };
                if fields.len() == 1
                    && matches!(
                        fields[0],
                        Value::Tuple(_) | Value::Bytes(_)
                    )
                    && !vec_typed
                {
                    // The single-array-field pun, as in
                    // `project_value`: the write continues into
                    // the field's elements.
                    let mut full = Vec::with_capacity(rest.len() + 1);
                    full.push(RefProj::Index(*idx));
                    full.extend_from_slice(rest);
                    navigate_mut_grow(&mut fields[0], &full, func, place, grow, None, None)
                } else if *idx >= fields.len() {
                    return Err(index_oob(fields.len(), *idx));
                } else {
                    navigate_mut_grow(&mut fields[*idx], rest, func, place, grow, None, None)
                }
            }
            // A sparse vector value (see `Value::SparseVec`):
            // a write to a stored element navigates into it —
            // including a spare-capacity slot past the declared
            // length (the tail writes a `Vec::append` performs
            // before it declares the new length) and the slot
            // one past the stored end, which extends the storage
            // by the one element being written; a write into
            // the unmaterialised tail materialises the gap with
            // uninitialised slots up to the written index —
            // bounded by the same materialisation limit as
            // `Vec::set_len`, past which the write stays an
            // explicit error rather than an allocation that
            // exhausts the host. The storage the value accounts
            // (its capacity) is the bound a write must land
            // in: the declared length, the stored elements and
            // the accounted spare capacity are all storage, and
            // a write that extends the stored list extends the
            // account with it.
            Value::SparseVec { elems, cap, .. } => {
                let extent = (*cap).max(elems.len() as u128);
                if *idx as u128 > extent
                    || (*idx as u128 == extent && *idx > elems.len())
                {
                    return Err(index_oob(extent as usize, *idx));
                }
                if *idx >= elems.len()
                    && *idx >= crate::builtins::MAX_VEC_SET_LEN_ELEMENTS
                {
                    return Err(InterpError::UnsupportedProjection {
                        kind: format!(
                            "index write at {idx} into a vector whose elements are not materialised"
                        ),
                    });
                }
                if *idx >= elems.len() {
                    elems.resize(*idx + 1, Value::Uninit);
                    if (elems.len() as u128) > *cap {
                        *cap = elems.len() as u128;
                    }
                }
                navigate_mut_grow(&mut elems[*idx], rest, func, place, grow, None, None)
            }
            Value::Bytes(b) => {
                if *idx >= b.len() {
                    return Err(index_oob(b.len(), *idx));
                }
                // Byte payloads come from constants and have no
                // `Value` slot to hand out; array locals built by
                // aggregates are tuples and take the arm above.
                Err(InterpError::type_mismatch(
                    format!("index write on {place}"),
                    "mutable array slot",
                    "constant bytes",
                ))
            }
            other => Err(InterpError::type_mismatch(
                format!("index write on {place}"),
                "array value",
                other.kind_name(),
            )),
        },
        RefProj::Subslice { offset, len } => {
            // A sub-slice has no single slot to hand out: the
            // navigation continues into one of its elements (an
            // `Index` step composes the offset into the base) or
            // into a nested sub-slice (the offsets compose). A
            // write to the whole sub-slice would have to splice
            // the base's element sequence in place; no MIR
            // assignment targets an unsized place that way, so it
            // stays an explicit error rather than a silent
            // partial write.
            match rest.first() {
                Some(RefProj::Index(j)) => {
                    let j = *j;
                    if j >= *len {
                        return Err(index_oob(*len, j));
                    }
                    let target = *offset + j;
                    let rest = &rest[1..];
                    match slot {
                        Value::Tuple(items) => {
                            if target >= items.len() {
                                return Err(index_oob(items.len(), target));
                            }
                            navigate_mut_grow(
                                &mut items[target],
                                rest,
                                func,
                                place,
                                grow,
                                program,
                                stepped_index_ty(ty.as_ref()),
                            )
                        }
                        Value::Enum { fields, .. } => {
                            if target >= fields.len() {
                                return Err(index_oob(fields.len(), target));
                            }
                            navigate_mut_grow(
                                &mut fields[target],
                                rest,
                                func,
                                place,
                                grow,
                                None,
                                None,
                            )
                        }
                        // A sparse vector value (see
                        // `Value::SparseVec`): the composed index
                        // addresses a storage slot under the same
                        // discipline as the sparse `Index` arm --
                        // the write must land within the storage
                        // the value accounts, a slot one past the
                        // stored end extends the storage by the
                        // element being written, and a write into
                        // the unmaterialised tail materialises the
                        // gap with uninitialised slots, bounded by
                        // the same materialisation limit.
                        Value::SparseVec { elems, cap, .. } => {
                            let extent = (*cap).max(elems.len() as u128);
                            if target as u128 > extent {
                                return Err(index_oob(extent as usize, target));
                            }
                            if target >= elems.len()
                                && target
                                    >= crate::builtins::MAX_VEC_SET_LEN_ELEMENTS
                            {
                                return Err(InterpError::UnsupportedProjection {
                                    kind: format!(
                                        "index write at {target} into a vector whose elements are not materialised"
                                    ),
                                });
                            }
                            if target >= elems.len() {
                                elems.resize(target + 1, Value::Uninit);
                                if (elems.len() as u128) > *cap {
                                    *cap = elems.len() as u128;
                                }
                            }
                            navigate_mut_grow(
                                &mut elems[target],
                                rest,
                                func,
                                place,
                                grow,
                                None,
                                None,
                            )
                        }
                        other => Err(InterpError::type_mismatch(
                            format!("subslice element write on {place}"),
                            "array value",
                            other.kind_name(),
                        )),
                    }
                }
                Some(RefProj::Subslice {
                    offset: inner_offset,
                    len: inner_len,
                }) => {
                    if inner_offset + inner_len > *len {
                        return Err(InterpError::panic(format!(
                            "range end index {} out of range for slice of length {len}",
                            inner_offset + inner_len
                        )));
                    }
                    let mut composed = vec![RefProj::Subslice {
                        offset: offset + inner_offset,
                        len: *inner_len,
                    }];
                    composed.extend_from_slice(&rest[1..]);
                    navigate_mut_grow(slot, &composed, func, place, grow, program, ty)
                }
                _ => Err(InterpError::type_mismatch(
                    format!("write through sub-slice on {place}"),
                    "an element or nested sub-slice of the sub-slice",
                    "the whole sub-slice",
                )),
            }
        }
        RefProj::Downcast(variant) => match slot {
            Value::Coroutine { .. }
                if matches!(rest.first(), Some(RefProj::Field(_))) =>
            {
                // A coroutine's field writes go through the downcast
                // of the state being entered, whatever the current
                // tag is, and land in that state's saved storage -
                // not in the upvar storage a field step without a
                // downcast addresses (see `Value::Coroutine`); the
                // unresumed state's downcast is the one exception,
                // handled below. The state storage grows to the
                // written slot, like the upvar storage in the
                // `Field` arm above.
                let (idx, tail) = match rest.split_first() {
                    Some((RefProj::Field(idx), tail)) => (*idx, tail),
                    _ => unreachable!(),
                };
                let Value::Coroutine { fields, saved, .. } = slot else {
                    unreachable!()
                };
                // State 0 is the unresumed state: its fields are
                // the upvars, so the write lands in the upvar
                // storage, matching the read paths.
                let state = if *variant == 0 {
                    &mut *fields
                } else {
                    if saved.len() <= *variant as usize {
                        saved.resize(*variant as usize + 1, Vec::new());
                    }
                    &mut saved[*variant as usize]
                };
                if state.len() <= idx {
                    state.resize(idx + 1, Value::Uninit);
                }
                navigate_mut_grow(&mut state[idx], tail, func, place, grow, None, None)
            }
            // A downcast with no field step after it (a whole-state
            // place) denotes the coroutine value itself.
            Value::Coroutine { .. } => {
                navigate_mut_grow(slot, rest, func, place, grow, None, None)
            }
            Value::Enum { .. } => {
                if let Value::Enum {
                    variant: current,
                    fields,
                } = slot
                {
                    if current != variant {
                        // A mutable navigation through the downcast of
                        // a variant other than the one the tag names is
                        // an in-place reinitialization: the old variant's
                        // payload is invalidated and the new variant's
                        // storage is rematerialized empty (the Field arm
                        // pads it with Uninit slots as fields are
                        // written). The tag switches here so that the
                        // closing SetDiscriminant, when the MIR emits
                        // one after the stores, names the held variant
                        // and leaves the stored fields in place.
                        *current = *variant;
                        fields.clear();
                    }
                }
                // A field step right after a downcast names one of
                // the variant's own fields: consume it here, growing
                // the field vector as the `Field` arm would, so the
                // wrapper descent there never sees a post-downcast
                // step (a variant's field 0 is not a transparent
                // wrapper's payload).
                if matches!(rest.first(), Some(RefProj::Field(_))) {
                    let Value::Enum { fields, .. } = slot else {
                        unreachable!("enum checked above")
                    };
                    let Some((RefProj::Field(idx), tail)) = rest.split_first()
                    else {
                        unreachable!("field step checked above")
                    };
                    let idx = *idx;
                    if fields.len() <= idx {
                        fields.resize(idx + 1, Value::Uninit);
                    }
                    return navigate_mut_grow(
                        &mut fields[idx],
                        tail,
                        func,
                        place,
                        grow,
                        program,
                        stepped_field_ty(program, ty.as_ref(), *variant as usize, idx),
                    );
                }
                navigate_mut_grow(slot, rest, func, place, grow, program, ty)
            }
            Value::Tuple(_) if *variant == 0 => {
                navigate_mut_grow(slot, rest, func, place, grow, program, ty)
            }
            other => Err(InterpError::type_mismatch(
                format!("downcast write on {place}"),
                "ADT value",
                other.kind_name(),
            )),
        },
    }
}
