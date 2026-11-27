//! Byte-level views over scalar storage.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};

/// Whether a target projection denotes the root position
/// itself: no step of it moves the position. A recorded
/// denomination (see [`RefProj::View`]) names the type of
/// the value at the position; it does not move it, so a
/// projection of denomination steps only is a root
/// projection for every consumer that asks where a target
/// points (the byte view's root judgements below).
fn projection_denotes_root(projection: &[RefProj]) -> bool {
    projection
        .iter()
        .all(|s| matches!(s, RefProj::View(_)))
}

impl<'a> Interpreter<'a> {
    // ----- byte view of a value's storage ----------------------
    //
    // A raw byte pointer displaced into the middle of a value's
    // storage (a `*const u8` cast of an array's address, advanced
    // past an element boundary; a byte pointer into a struct
    // field's interior) denotes a byte position in the storage
    // of the value the target place or static holds. The load /
    // store path resolves such a position through the value's
    // declared type and the layout tables the driver recorded:
    // the position is walked from the base of the value, scalar
    // by scalar, array element by array element, ADT field by
    // ADT field (through the per-variant field spans of
    // `Program::adt_field_layouts`), and the byte it lands in is
    // read or spliced there. The view judges a position only
    // when every step is determined by recorded layout: an
    // enum's tag position (through `Program::adt_tag_layouts`)
    // is the discriminant's own bytes, a position in the
    // padding between an ADT's fields reads as the model's
    // indeterminate value and refuses a write, one outside the
    // value's storage is an explicit error, and a position
    // whose layout is not recorded (a tuple's interior,
    // an ADT with no field-layout entry, an allocation's raw
    // bytes) is left to the existing paths unchanged. Composing
    // the position from the base — never decomposing it into an
    // element index plus a within-element offset at offset time —
    // is what keeps a read that crosses an element boundary
    // exact: the walk crosses into the next element because the
    // flat position does.

    /// The layout the byte view descends by (see
    /// [`Self::byte_layout_of_ty`] and [`Self::byte_layout_of_tag`]).
    /// A scalar's bytes are its own declared width's: an integer's
    /// low bytes, a float's bit pattern at its own width (an `f32`
    /// field's bytes are its `f32` bits, not the widened `f64`
    /// storage's), a `char`'s code point.
    fn byte_layout_of_ty(&self, ty: &Ty) -> Option<ByteLayout> {
        Some(match ty {
            Ty::Bool => ByteLayout::Scalar(ByteScalar::Bool),
            Ty::Int(t) => ByteLayout::Scalar(ByteScalar::Int(*t)),
            Ty::Uint(t) => ByteLayout::Scalar(ByteScalar::Uint(*t)),
            Ty::Float(t) => ByteLayout::Scalar(ByteScalar::Float(*t)),
            Ty::Array(elem, n) => {
                let elem_layout = self.byte_layout_of_ty(elem)?;
                let elem_size = usize::try_from(self.ty_size(elem)?).ok()?;
                ByteLayout::Seq(
                    Box::new(elem_layout),
                    elem_size,
                    usize::try_from(*n).ok()?,
                )
            }
            Ty::Adt(name) => ByteLayout::Adt(
                // The IR's ADT types carry only the
                // definition's name, while the layout tables
                // are keyed by the rendered instance name:
                // denominate the layout by the instance the
                // name denotes in this program (see
                // `adt_instance_key`). An ambiguous or
                // unrecorded name keeps the bare name, and
                // every lookup below declines as before.
                self.adt_instance_key(name)
                    .unwrap_or_else(|| name.clone()),
            ),
            _ => return None,
        })
    }

    /// The byte layout of an ADT field, from its recorded
    /// tag. A field whose tag names a type with no recorded
    /// layout — a newtype wrapper the conversion never laid
    /// out as its own instance, such as the standard
    /// library's niche newtypes — falls back to the layout
    /// of the scalar leaf the stored value itself holds:
    /// the value model flattens such a wrapper to its leaf,
    /// and the field's recorded slot size fixes the leaf's
    /// width.
    fn field_byte_layout(
        &self,
        name: &str,
        variant: usize,
        field: usize,
        slot_size: usize,
        value: &Value,
    ) -> Option<ByteLayout> {
        let tag = self.adt_field_tag(name, variant, field)?;
        let resolved = self.byte_layout_of_tag(&tag);
        // A newtype over a scalar leaf — a single-field ADT
        // whose only field is a scalar of the slot's width —
        // is stored as the leaf itself in the value model
        // (the standard library's niche newtypes, flattened
        // by the builtins that construct them), so its byte
        // layout is the leaf's, whatever form the slot
        // currently holds.
        if let Some(ByteLayout::Adt(inner)) = &resolved {
            if let Some(rows) = self.program().adt_field_layouts.get(inner) {
                if rows.len() == 1 && rows[0].len() == 1 {
                    if let Some(inner_tag) = self.adt_field_tag(inner, 0, 0) {
                        if let Some(ByteLayout::Scalar(scalar)) =
                            self.byte_layout_of_tag(&inner_tag)
                        {
                            if scalar.width() == slot_size {
                                return Some(ByteLayout::Scalar(scalar));
                            }
                        }
                    }
                }
            }
        }
        if let Some(layout) = resolved {
            // A layout the value's own form cannot walk — an
            // ADT layout over a scalar leaf, the flattened
            // newtype case — is not the field's layout here.
            if !matches!(layout, ByteLayout::Adt(_))
                || !matches!(
                    value,
                    Value::Uint(_)
                        | Value::Int(_)
                        | Value::Float(_)
                        | Value::Float128(_)
                        | Value::Bool(_)
                )
            {
                return Some(layout);
            }
        }
        let scalar = match value {
            Value::Uint(_) => ByteScalar::Uint(match slot_size {
                1 => UintTy::U8,
                2 => UintTy::U16,
                4 => UintTy::U32,
                8 => UintTy::U64,
                16 => UintTy::U128,
                _ => return None,
            }),
            Value::Int(_) => ByteScalar::Int(match slot_size {
                1 => IntTy::I8,
                2 => IntTy::I16,
                4 => IntTy::I32,
                8 => IntTy::I64,
                16 => IntTy::I128,
                _ => return None,
            }),
            Value::Float(_) => ByteScalar::Float(match slot_size {
                2 => FloatTy::F16,
                4 => FloatTy::F32,
                8 => FloatTy::F64,
                _ => return None,
            }),
            Value::Float128(_) if slot_size == 16 => {
                ByteScalar::Float(FloatTy::F128)
            }
            Value::Bool(_) if slot_size == 1 => ByteScalar::Bool,
            _ => return None,
        };
        Some(ByteLayout::Scalar(scalar))
    }

    /// The byte layout of a field type named by its recorded
    /// debug-shape tag (see [`Self::byte_layout_of_ty`]): the
    /// scalar and array tags resolve structurally, an ADT tag
    /// resolves when the instance has a recorded field layout,
    /// and anything else (a tuple tag, a pointer tag, an unsized
    /// tag) has no byte layout here.
    fn byte_layout_of_tag(&self, tag: &str) -> Option<ByteLayout> {
        let tag = tag.trim();
        let scalar = match tag {
            "bool" => Some(ByteScalar::Bool),
            "char" => Some(ByteScalar::Char),
            "i8" => Some(ByteScalar::Int(IntTy::I8)),
            "i16" => Some(ByteScalar::Int(IntTy::I16)),
            "i32" => Some(ByteScalar::Int(IntTy::I32)),
            "i64" => Some(ByteScalar::Int(IntTy::I64)),
            "i128" => Some(ByteScalar::Int(IntTy::I128)),
            "isize" => Some(ByteScalar::Int(IntTy::Isize)),
            "u8" => Some(ByteScalar::Uint(UintTy::U8)),
            "u16" => Some(ByteScalar::Uint(UintTy::U16)),
            "u32" => Some(ByteScalar::Uint(UintTy::U32)),
            "u64" => Some(ByteScalar::Uint(UintTy::U64)),
            "u128" => Some(ByteScalar::Uint(UintTy::U128)),
            "usize" => Some(ByteScalar::Uint(UintTy::Usize)),
            "f16" => Some(ByteScalar::Float(FloatTy::F16)),
            "f32" => Some(ByteScalar::Float(FloatTy::F32)),
            "f64" => Some(ByteScalar::Float(FloatTy::F64)),
            "f128" => Some(ByteScalar::Float(FloatTy::F128)),
            _ => None,
        };
        if let Some(scalar) = scalar {
            return Some(ByteLayout::Scalar(scalar));
        }
        if tag.starts_with('[') && tag.ends_with(']') {
            let inner = &tag[1..tag.len() - 1];
            let (elem_tag, count) = inner.rsplit_once(';')?;
            let elem_tag = elem_tag.trim();
            let count: usize = count.trim().parse().ok()?;
            let elem_layout = self.byte_layout_of_tag(elem_tag)?;
            let (elem_size, _) = self.layout_of(elem_tag)?;
            let elem_size = usize::try_from(elem_size).ok()?;
            if self.byte_layout_size(&elem_layout)? != elem_size {
                return None;
            }
            return Some(ByteLayout::Seq(Box::new(elem_layout), elem_size, count));
        }
        if self.program().adt_field_layouts.contains_key(tag) || self.adt_is_union(tag)
        {
            return Some(ByteLayout::Adt(tag.to_string()));
        }
        None
    }

    /// The total storage size of a byte layout, when it is
    /// recorded (an ADT's size comes from its recorded layout).
    fn byte_layout_size(&self, layout: &ByteLayout) -> Option<usize> {
        match layout {
            ByteLayout::Scalar(scalar) => Some(scalar.width()),
            ByteLayout::Seq(_, elem_size, count) => elem_size.checked_mul(*count),
            ByteLayout::Adt(name) => self
                .program()
                .adt_layouts
                .get(name)
                .map(|&(size, _)| size as usize),
        }
    }

    /// Whether the ADT a type name denotes is a union.
    /// The kind is a property of the definition, shared by
    /// every instance, so the question goes through the
    /// definition-level shape channel (see
    /// `debug_shape_for_def`): a recorded instance entry
    /// answers for a bare definition name, and a rendered
    /// instance name answers for itself.
    fn adt_is_union(&self, name: &str) -> bool {
        matches!(
            self.debug_shape_for_def(name),
            Some(shape) if shape.kind == sparagmos_mir::AdtShapeKind::Union
        )
    }

    /// The storage image of a union value: one entry per
    /// byte of the union's recorded size, `Some` where the
    /// byte is initialised. Every field of a union sits at
    /// offset zero, so the image is the widest live field's
    /// own image over uninitialised storage elsewhere —
    /// the value model's invariant is that one union
    /// value's fields are readings of the same bytes, so
    /// the widest field's image covers the other fields'
    /// spans. A value already in byte form carries its
    /// image directly. A live field whose bytes the view
    /// cannot judge declines the composition (`None`);
    /// zero-sized fields hold no bytes and are skipped.
    fn union_storage_image(
        &self,
        value: &Value,
        name: &str,
    ) -> Option<Vec<Option<u8>>> {
        let size = self.program().adt_layouts.get(name)?.0 as usize;
        match value {
            Value::UnionBytes { bytes } => {
                let mut image = bytes.clone();
                image.truncate(size);
                image.resize(size, None);
                Some(image)
            }
            Value::Uninit => Some(vec![None; size]),
            Value::Enum { fields, .. } => {
                let mut widest: Option<Vec<Option<u8>>> = None;
                for (f, field) in fields.iter().enumerate() {
                    if *field == Value::Uninit {
                        continue;
                    }
                    let tag = self.adt_field_tag(name, 0, f)?;
                    let (field_size, _) = self.layout_of(&tag)?;
                    let field_size = field_size as usize;
                    if field_size == 0 {
                        continue;
                    }
                    if widest
                        .as_ref()
                        .is_some_and(|image| image.len() >= field_size)
                    {
                        continue;
                    }
                    let layout = self.byte_layout_of_tag(&tag)?;
                    let mut image = vec![None; field_size];
                    for pos in 0..field_size {
                        match self.storage_byte(field, &layout, pos) {
                            Some(Ok(StorageByte::Byte(byte))) => {
                                image[pos] = Some(byte);
                            }
                            Some(Ok(StorageByte::Padding))
                            | Some(Err(ByteVerdict::Uninit)) => {}
                            _ => return None,
                        }
                    }
                    widest = Some(image);
                }
                let mut image = vec![None; size];
                if let Some(field_image) = widest {
                    let n = field_image.len().min(size);
                    image[..n].copy_from_slice(&field_image[..n]);
                }
                Some(image)
            }
            _ => None,
        }
    }

    /// The recorded type tag of field `field` of variant
    /// `variant` of the ADT instance `name`, from its debug
    /// shape; `None` when no shape is recorded.
    pub(crate) fn adt_field_tag(&self, name: &str, variant: usize, field: usize) -> Option<String> {
        self.program()
            .adt_debug_shapes
            .get(name)?
            .variants
            .get(variant)?
            .fields
            .get(field)
            .map(|f| f.ty.clone())
    }

    /// Walk a structural projection prefix over the declared
    /// type `base_ty`, returning the byte offset of the position
    /// it denotes within the base value's storage and the
    /// declared type at that position, when every step's storage
    /// offset is determined by the type and the recorded layout
    /// tables. A field step into an ADT prices through the
    /// instance's recorded field layout; the field's own type is
    /// recovered only when its recorded tag names another
    /// laid-out ADT instance (the descent from the base needs no
    /// more: it re-walks the flat position through the field
    /// tags itself). A tuple field step is not priced: a tuple's
    /// field offsets are the compiler's choice and are not
    /// recorded. The third result is the storage size of the
    /// position the last step names, priced from the same
    /// layout rows (a field's recorded size, an array element's
    /// size, a subslice's span): it survives the steps the type
    /// walk cannot follow — a field whose tag names no laid-out
    /// instance, as a union field's does, ends the walk's type
    /// but not its size. A downcast keeps the size of the
    /// position it selects a variant of; a view step resets it,
    /// the view's own type being the evidence there.
    fn projection_storage_offset(
        &self,
        base_ty: &Ty,
        prefix: &[RefProj],
    ) -> Option<(usize, Option<Ty>, Option<usize>)> {
        let mut ty: Option<Ty> = Some(base_ty.clone());
        let mut off: usize = 0;
        let mut size: Option<usize> = None;
        let mut variant: usize = 0;
        for step in prefix {
            match step {
                RefProj::Field(i) => match &ty {
                    Some(Ty::Adt(name)) => {
                        // The field rows and field tags are
                        // recorded under the rendered instance
                        // name; the walk's type names the
                        // definition alone (see
                        // `adt_instance_key`).
                        let key = self
                            .adt_instance_key(name)
                            .unwrap_or_else(|| name.clone());
                        let rows = self.program().adt_field_layouts.get(&key)?;
                        let row = rows.get(variant)?;
                        let &(field_off, field_size) = row.get(*i)?;
                        off = off.checked_add(usize::try_from(field_off).ok()?)?;
                        size = Some(usize::try_from(field_size).ok()?);
                        ty = self
                            .adt_field_tag(&key, variant, *i)
                            .filter(|tag| {
                                self.program().adt_field_layouts.contains_key(tag)
                            })
                            .map(Ty::Adt);
                        variant = 0;
                    }
                    _ => return None,
                },
                RefProj::Index(k) => match &ty {
                    Some(Ty::Array(elem, _)) => {
                        let elem_size = usize::try_from(self.ty_size(elem)?).ok()?;
                        off = off.checked_add(k.checked_mul(elem_size)?)?;
                        size = Some(elem_size);
                        ty = Some((**elem).clone());
                    }
                    _ => return None,
                },
                RefProj::Downcast(v) => match &ty {
                    Some(Ty::Adt(_)) | None => {
                        variant = *v as usize;
                    }
                    _ => return None,
                },
                RefProj::View(view) => {
                    // The recorded denomination (see
                    // `RefProj::View`): the steps after it
                    // price through the view's type.
                    ty = Some(view.clone());
                    size = None;
                    variant = 0;
                }
                RefProj::Subslice { offset, len } => match &ty {
                    Some(Ty::Array(elem, _)) => {
                        let elem_size = usize::try_from(self.ty_size(elem)?).ok()?;
                        off = off.checked_add(offset.checked_mul(elem_size)?)?;
                        size = Some(usize::try_from(*len).ok()?.checked_mul(elem_size)?);
                        ty = Some(Ty::Array(elem.clone(), u64::try_from(*len).ok()?));
                    }
                    _ => return None,
                },
                _ => return None,
            }
        }
        Some((off, ty, size))
    }

    /// The flat byte position a byte-displaced projection
    /// denotes within the storage of a value of type `base_ty`,
    /// plus `extra` further bytes (a sequence access's byte
    /// index, or a wider access's byte offset within the
    /// position). `None` when the projection is not a displaced
    /// position this view judges: no displacement step, a step
    /// other than the caller's own after the displacement, a
    /// union field step anywhere, or a prefix step whose
    /// storage offset the type does not determine. `Err`
    /// carries the position when it is negative — a judged
    /// position before the value's storage.
    fn byte_view_position(
        &self,
        base_ty: &Ty,
        projection: &[RefProj],
        extra: i128,
    ) -> Option<Result<usize, i128>> {
        if projection
            .iter()
            .any(|s| matches!(s, RefProj::UnionField { .. }))
        {
            return None;
        }
        let at = projection
            .iter()
            .position(|s| matches!(s, RefProj::ByteOffset(_)))?;
        let RefProj::ByteOffset(disp) = projection[at] else {
            unreachable!()
        };
        if !projection[at + 1..].is_empty() {
            return None;
        }
        let (off, ..) = self.projection_storage_offset(base_ty, &projection[..at])?;
        let flat = (off as i128).checked_add(disp)?.checked_add(extra)?;
        if flat < 0 {
            return Some(Err(flat));
        }
        Some(Ok(usize::try_from(flat).ok()?))
    }

    /// Read the byte at flat position `pos` of `value`'s
    /// storage under `layout` (see the byte view above).
    /// `None` when the position's layout is not fully recorded
    /// (the caller then keeps the existing path); a judged
    /// position answers the byte it holds — a definite byte of
    /// the storage, or the indeterminate byte of a padding
    /// position — or the verdict it meets.
    fn storage_byte(
        &self,
        value: &Value,
        layout: &ByteLayout,
        pos: usize,
    ) -> Option<Result<StorageByte, ByteVerdict>> {
        if *value == Value::Uninit {
            return Some(Err(ByteVerdict::Uninit));
        }
        match layout {
            ByteLayout::Scalar(kind) => {
                if pos >= kind.width() {
                    return Some(Err(ByteVerdict::Outside(pos)));
                }
                let bits = scalar_bits_of_kind(value, *kind)?;
                Some(Ok(StorageByte::Byte((bits >> (8 * pos)) as u8)))
            }
            ByteLayout::Seq(elem, elem_size, count) => {
                if *elem_size == 0 {
                    return None;
                }
                let i = pos / elem_size;
                if i >= *count {
                    return Some(Err(ByteVerdict::Outside(pos)));
                }
                match value {
                    Value::Tuple(items) => {
                        let item = items.get(i)?;
                        self.storage_byte(item, elem, pos % elem_size)
                    }
                    Value::Bytes(bytes)
                        if *elem_size == 1
                            && matches!(
                                **elem,
                                ByteLayout::Scalar(ByteScalar::Uint(UintTy::U8))
                            ) =>
                    {
                        match bytes.get(pos) {
                            Some(b) => Some(Ok(StorageByte::Byte(*b))),
                            None => Some(Err(ByteVerdict::Outside(pos))),
                        }
                    }
                    Value::Str(text)
                        if *elem_size == 1
                            && matches!(
                                **elem,
                                ByteLayout::Scalar(ByteScalar::Uint(UintTy::U8))
                            ) =>
                    {
                        match text.as_bytes().get(pos) {
                            Some(b) => Some(Ok(StorageByte::Byte(*b))),
                            None => Some(Err(ByteVerdict::Outside(pos))),
                        }
                    }
                    _ => None,
                }
            }
            ByteLayout::Adt(name) => {
                // A union's fields all share offset zero, so
                // no single field's slot holds the storage a
                // byte access denotes: the byte view over a
                // union answers from the value's composed
                // storage image (see `union_storage_image`).
                // Unions carry no recorded field-layout rows,
                // so without this the field walk below
                // declines every byte access to a union place.
                if self.adt_is_union(name) {
                    let image = self.union_storage_image(value, name)?;
                    return match image.get(pos) {
                        Some(Some(byte)) => Some(Ok(StorageByte::Byte(*byte))),
                        Some(None) => Some(Err(ByteVerdict::Uninit)),
                        None => Some(Err(ByteVerdict::Outside(pos))),
                    };
                }
                let Value::Enum { variant, fields } = value else {
                    return None;
                };
                let rows = self.program().adt_field_layouts.get(name)?;
                let row = rows.get(*variant as usize)?;
                let mut hit = None;
                for (f, &(field_off, field_size)) in row.iter().enumerate() {
                    let (field_off, field_size) =
                        (field_off as usize, field_size as usize);
                    if field_size > 0 && pos >= field_off && pos < field_off + field_size
                    {
                        hit = Some((f, field_off));
                        break;
                    }
                }
                let Some((f, field_off)) = hit else {
                    // No field covers the position: an enum's
                    // tag span holds the discriminant's own
                    // bytes, a position past the value's storage
                    // is outside it, and anything else inside
                    // the storage is padding, whose byte has
                    // no definite value.
                    if let Some(tag) = self.program().adt_tag_layouts.get(name) {
                        let (tag_off, tag_size) =
                            (tag.offset as usize, tag.size as usize);
                        if tag_size > 0
                            && pos >= tag_off
                            && pos < tag_off + tag_size
                        {
                            let discr =
                                *tag.discriminants.get(*variant as usize)?;
                            let byte = (discr >> (8 * (pos - tag_off))) as u8;
                            return Some(Ok(StorageByte::Byte(byte)));
                        }
                    }
                    if let Some(total) = self.byte_layout_size(layout) {
                        if pos >= total {
                            return Some(Err(ByteVerdict::Outside(pos)));
                        }
                    }
                    return Some(Ok(StorageByte::Padding));
                };
                let field = fields.get(f)?;
                let field_layout = self.field_byte_layout(
                    name,
                    *variant as usize,
                    f,
                    row[f].1 as usize,
                    field,
                )?;
                // The recorded slot can exceed the field
                // type's own size (its tail padding, rounded
                // to the containing type's alignment); only a
                // slot smaller than the type is inconsistent.
                if self.byte_layout_size(&field_layout)? > row[f].1 as usize {
                    return None;
                }
                self.storage_byte(field, &field_layout, pos - field_off)
            }
        }
    }

    /// Splice `byte` into the byte at flat position `pos` of
    /// `slot`'s storage under `layout`, in place (the write
    /// side of the byte view; see [`Self::storage_byte`] for the
    /// `None` contract). A scalar slot that was never written is
    /// zero storage for this purpose — the convention
    /// `splice_scalar_storage_byte` already keeps: a buffer
    /// being filled byte by byte is written whole before it is
    /// read, so the bytes the fill has not reached are never
    /// observed. An unwritten aggregate has no storage to
    /// splice into and is left to the existing path.
    fn splice_storage_byte(
        &self,
        slot: &mut Value,
        layout: &ByteLayout,
        pos: usize,
        byte: u8,
    ) -> Option<Result<(), ByteVerdict>> {
        match layout {
            ByteLayout::Scalar(kind) => {
                if pos >= kind.width() {
                    return Some(Err(ByteVerdict::Outside(pos)));
                }
                let shift = 8 * pos;
                let splice = |bits: u128| -> u128 {
                    (bits & !(0xffu128 << shift)) | (u128::from(byte) << shift)
                };
                match (kind, &*slot) {
                    (ByteScalar::Int(t), Value::Int(i)) => {
                        let bits = t.bits();
                        let mask: u128 = if bits >= 128 {
                            u128::MAX
                        } else {
                            (1u128 << bits) - 1
                        };
                        let mut next = splice((*i as u128) & mask);
                        if bits < 128 && (next >> (bits - 1)) & 1 == 1 {
                            next |= !mask;
                        }
                        *slot = Value::Int(next as i128);
                    }
                    (ByteScalar::Int(t), Value::Uninit) => {
                        let bits = t.bits();
                        let mask: u128 = if bits >= 128 {
                            u128::MAX
                        } else {
                            (1u128 << bits) - 1
                        };
                        let mut next = splice(0);
                        if bits < 128 && (next >> (bits - 1)) & 1 == 1 {
                            next |= !mask;
                        }
                        *slot = Value::Int(next as i128);
                    }
                    (ByteScalar::Uint(t), Value::Uint(u)) => {
                        *slot = Value::Uint(splice(*u) & t.max());
                    }
                    (ByteScalar::Uint(t), Value::Uninit) => {
                        *slot = Value::Uint(splice(0) & t.max());
                    }
                    (ByteScalar::Char, Value::Uint(u)) => {
                        let v = splice(*u) as u32;
                        if char::from_u32(v).is_none() {
                            return Some(Err(ByteVerdict::InvalidChar(v)));
                        }
                        *slot = Value::Uint(u128::from(v));
                    }
                    (ByteScalar::Char, Value::Uninit) => {
                        let v = splice(0) as u32;
                        if char::from_u32(v).is_none() {
                            return Some(Err(ByteVerdict::InvalidChar(v)));
                        }
                        *slot = Value::Uint(u128::from(v));
                    }
                    (ByteScalar::Bool, Value::Bool(_))
                    | (ByteScalar::Bool, Value::Uninit) => {
                        *slot = Value::Bool(byte != 0);
                    }
                    (ByteScalar::Float(FloatTy::F16), Value::Float(f)) => {
                        let bits = splice(u128::from(crate::f128::f64_to_f16_bits(*f)));
                        *slot = Value::Float(crate::f128::f16_bits_to_f64(bits as u16));
                    }
                    (ByteScalar::Float(FloatTy::F16), Value::Uninit) => {
                        let bits = splice(0);
                        *slot = Value::Float(crate::f128::f16_bits_to_f64(bits as u16));
                    }
                    (ByteScalar::Float(FloatTy::F32), Value::Float(f)) => {
                        let bits = splice(u128::from(crate::f128::f64_to_f32_bits_exact(*f)));
                        *slot = Value::Float(crate::f128::f32_bits_to_f64_exact(bits as u32));
                    }
                    (ByteScalar::Float(FloatTy::F32), Value::Uninit) => {
                        *slot = Value::Float(crate::f128::f32_bits_to_f64_exact(splice(0) as u32));
                    }
                    (ByteScalar::Float(FloatTy::F64), Value::Float(f)) => {
                        let bits = splice(u128::from(f.to_bits()));
                        *slot = Value::Float(f64::from_bits(bits as u64));
                    }
                    (ByteScalar::Float(FloatTy::F64), Value::Uninit) => {
                        *slot = Value::Float(f64::from_bits(splice(0) as u64));
                    }
                    (ByteScalar::Float(FloatTy::F128), Value::Float128(bits)) => {
                        *slot = Value::Float128(splice(*bits));
                    }
                    (ByteScalar::Float(FloatTy::F128), Value::Uninit) => {
                        *slot = Value::Float128(splice(0));
                    }
                    _ => return None,
                }
                Some(Ok(()))
            }
            ByteLayout::Seq(elem, elem_size, count) => {
                if *elem_size == 0 {
                    return None;
                }
                let i = pos / elem_size;
                if i >= *count {
                    return Some(Err(ByteVerdict::Outside(pos)));
                }
                match slot {
                    Value::Tuple(items) => {
                        let item = items.get_mut(i)?;
                        self.splice_storage_byte(item, elem, pos % elem_size, byte)
                    }
                    Value::Bytes(bytes)
                        if *elem_size == 1
                            && matches!(
                                **elem,
                                ByteLayout::Scalar(ByteScalar::Uint(UintTy::U8))
                            ) =>
                    {
                        match bytes.get_mut(pos) {
                            Some(b) => {
                                *b = byte;
                                Some(Ok(()))
                            }
                            None => Some(Err(ByteVerdict::Outside(pos))),
                        }
                    }
                    _ => None,
                }
            }
            ByteLayout::Adt(name) => {
                // The write side of the union byte view (see
                // `storage_byte`): the byte lands in the
                // composed storage image and the slot takes
                // the byte form — the one value form that
                // holds a union's bytes without naming a
                // field. Later field reads decode from it
                // through the union projection machinery, and
                // later byte accesses read the image back.
                if self.adt_is_union(name) {
                    let mut image = self.union_storage_image(slot, name)?;
                    if pos >= image.len() {
                        return Some(Err(ByteVerdict::Outside(pos)));
                    }
                    image[pos] = Some(byte);
                    *slot = Value::UnionBytes { bytes: image };
                    return Some(Ok(()));
                }
                // Uninitialised storage under an ADT layout
                // starts as the first variant's skeleton: the
                // tag bytes spliced into it re-form the value
                // at the variant they name (carrying the other
                // spliced bytes across), and field bytes land
                // in the skeleton's fields — the same formation
                // a byte write into a fresh `MaybeUninit` slot
                // performs.
                if matches!(slot, Value::Uninit) {
                    let rows = self.program().adt_field_layouts.get(name)?;
                    let first = rows.first()?;
                    *slot = Value::Enum {
                        variant: 0,
                        fields: vec![Value::Uninit; first.len()],
                    };
                }
                let (variant, row) = {
                    let Value::Enum { variant, .. } = &*slot else {
                        return None;
                    };
                    let rows = self.program().adt_field_layouts.get(name)?;
                    let row = rows.get(*variant as usize)?.clone();
                    (*variant, row)
                };
                let mut hit = None;
                for (f, &(field_off, field_size)) in row.iter().enumerate() {
                    let (field_off, field_size) =
                        (field_off as usize, field_size as usize);
                    if field_size > 0 && pos >= field_off && pos < field_off + field_size
                    {
                        hit = Some((f, field_off));
                        break;
                    }
                }
                let Some((f, field_off)) = hit else {
                    // No field covers the position: an enum's
                    // tag span is the discriminant's own
                    // storage, spliced as such; anything else
                    // is padding, which has no storage to
                    // splice into.
                    if let Some(tag) =
                        self.program().adt_tag_layouts.get(name).cloned()
                    {
                        let (tag_off, tag_size) =
                            (tag.offset as usize, tag.size as usize);
                        if tag_size > 0
                            && pos >= tag_off
                            && pos < tag_off + tag_size
                        {
                            return self.splice_tag_byte(
                                slot, name, &tag, pos, byte,
                            );
                        }
                    }
                    return Some(Err(ByteVerdict::Padding(pos)));
                };
                let field_layout = {
                    let Value::Enum { fields, .. } = &*slot else {
                        return None;
                    };
                    let field = fields.get(f)?;
                    self.field_byte_layout(
                        name,
                        variant as usize,
                        f,
                        row[f].1 as usize,
                        field,
                    )?
                };
                // As on the read side: a recorded slot larger
                // than the field type is its tail padding,
                // not an inconsistency.
                if self.byte_layout_size(&field_layout)? > row[f].1 as usize {
                    return None;
                }
                let Value::Enum { fields, .. } = slot else {
                    return None;
                };
                let field = fields.get_mut(f)?;
                self.splice_storage_byte(field, &field_layout, pos - field_off, byte)
            }
        }
    }

    /// Splice `byte` into the stored discriminant tag of
    /// the enum value in `slot` (the write side of the tag
    /// span; see [`Self::storage_byte`] for the read side).
    /// The tag's bytes are the discriminant value's own, so
    /// the splice recomputes the value the tag names. A
    /// result naming the current variant changes nothing
    /// else — the value model stores no separate tag. A
    /// result naming another variant re-forms the value as
    /// that variant over the same storage: the new variant's
    /// fields start uninitialised and every definite byte of
    /// the old storage outside the tag is spliced into them,
    /// so a field the two variants share storage with keeps
    /// its bytes and a field over old padding or unwritten
    /// storage stays uninitialised. A result naming no
    /// variant, or a re-formation whose bytes the view
    /// cannot all judge, declines (`None`): the model cannot
    /// hold an enum value for a discriminant no variant has.
    fn splice_tag_byte(
        &self,
        slot: &mut Value,
        name: &str,
        tag: &sparagmos_mir::EnumTag,
        pos: usize,
        byte: u8,
    ) -> Option<Result<(), ByteVerdict>> {
        let Value::Enum { variant, .. } = &*slot else {
            return None;
        };
        let cur = *variant as usize;
        let size = tag.size as usize;
        let off = tag.offset as usize;
        let mask: u128 = if size >= 16 {
            u128::MAX
        } else {
            (1u128 << (8 * size)) - 1
        };
        let cur_bits = tag.discriminants.get(cur)? & mask;
        let shift = 8 * (pos - off);
        let new_bits =
            (cur_bits & !(0xffu128 << shift)) | (u128::from(byte) << shift);
        let target = tag
            .discriminants
            .iter()
            .position(|d| d & mask == new_bits)?;
        if target == cur {
            return Some(Ok(()));
        }
        let layout = ByteLayout::Adt(name.to_string());
        let total = self.byte_layout_size(&layout)?;
        let old = slot.clone();
        let rows = self.program().adt_field_layouts.get(name)?;
        let new_row = rows.get(target)?;
        let mut reformed = Value::Enum {
            variant: target as u32,
            fields: vec![Value::Uninit; new_row.len()],
        };
        for p in 0..total {
            if p >= off && p < off + size {
                // The tag itself: correct by construction.
                continue;
            }
            let b = match self.storage_byte(&old, &layout, p)? {
                Ok(StorageByte::Byte(b)) => b,
                Ok(StorageByte::Padding) => continue,
                Err(ByteVerdict::Uninit) => continue,
                Err(_) => return None,
            };
            match self.splice_storage_byte(&mut reformed, &layout, p, b)? {
                Ok(()) => {}
                Err(ByteVerdict::Padding(_)) => {}
                Err(_) => return None,
            }
        }
        *slot = reformed;
        Some(Ok(()))
    }

    /// The error a judged byte-view verdict reports.
    fn byte_view_error(
        &self,
        verdict: ByteVerdict,
        func: &str,
        place: &str,
        read: bool,
    ) -> InterpError {
        let verb = if read { "read" } else { "write" };
        match verdict {
            ByteVerdict::Uninit => InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            },
            ByteVerdict::Outside(pos) => InterpError::UnsupportedProjection {
                kind: format!(
                    "{verb} through a byte pointer at position {pos}, \
                     outside the value's storage"
                ),
            },
            ByteVerdict::Padding(pos) => InterpError::UnsupportedProjection {
                kind: format!(
                    "{verb} through a byte pointer at position {pos}, \
                     which falls in the padding of the value's layout"
                ),
            },
            ByteVerdict::InvalidChar(v) => InterpError::UnsupportedProjection {
                kind: format!("byte write assembles the invalid char value {v:#x}"),
            },
        }
    }

    /// The base value, declared base type, and error context
    /// of a place/static/allocation target, for the byte view
    /// (see [`Self::byte_view_read`]). `None` for targets the
    /// view does not judge (a constant aggregate, a raw
    /// element block) and for a base whose declared type is
    /// not recorded: a place's is in its home body's table, a
    /// static's in the program's, and an allocation's in the
    /// record its creation point left (see the `alloc_tys`
    /// field) -- the same three sources the `MaybeUninit`
    /// entry gate consults.
    pub(crate) fn byte_view_base<'t>(
        &self,
        frames: &[Frame],
        target: &'t Resolved,
    ) -> Option<(&'t Vec<RefProj>, Value, Ty, String, String)> {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let frame = frames.get(*frame_pos)?;
                let base = frame.locals.get(*local)?.clone();
                let body = self.program().functions.get(&frame.func)?;
                let base_ty = body.local_tys.get(*local)?.clone();
                Some((
                    projection,
                    base,
                    base_ty,
                    frame.func.clone(),
                    describe_resolved(*local, projection),
                ))
            }
            Resolved::Static { name, projection } => {
                let base = self.static_base(name).ok()?;
                let base_ty = self.program().static_tys.get(name)?.clone();
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| "<static>".to_string());
                Some((
                    projection,
                    base,
                    base_ty,
                    func,
                    format!("static {name}"),
                ))
            }
            Resolved::Alloc { id, projection } => {
                // A raw element block's record, when one
                // exists, is its element type, which does not
                // denominate the block's whole storage; the
                // block's positions keep the element paths.
                if self.is_elem_block(*id) {
                    return None;
                }
                let base = self.alloc_base(*id).ok()?;
                let base_ty = self.alloc_ty(*id)?;
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| "<allocation>".to_string());
                Some((
                    projection,
                    base,
                    base_ty,
                    func,
                    format!("allocation #{id}"),
                ))
            }
            _ => None,
        }
    }

    /// A shaped whole-pointee load through a byte-displaced
    /// target (see [`Self::shaped_load`]): the load's bytes are
    /// composed one by one through the byte view, little-endian,
    /// and the composed pattern is constructed through the
    /// shape — the same composition `read_punned_int` performs
    /// for a type-punned pointer. `None` when the byte view
    /// does not judge the target or any of the load's bytes.
    /// A load whose bytes are all indeterminate (padding)
    /// answers the indeterminate value itself; a load only
    /// partly over indeterminate bytes has no honest scalar
    /// to construct and is not judged.
    pub(crate) fn byte_view_shaped_load(
        &self,
        frames: &[Frame],
        target: &Resolved,
        shape: &ScalarShape,
    ) -> Option<InterpResult<Value>> {
        let width = (shape.bits() / 8) as usize;
        if width == 0 || width > 16 {
            return None;
        }
        let mut bits: u128 = 0;
        let mut indet = 0usize;
        for k in 0..width {
            match self.byte_view_read(frames, target, k as i128, false)? {
                Ok(Value::Uint(b)) => bits |= (b & 0xff) << (8 * k),
                Ok(Value::Uninit) => indet += 1,
                Ok(_) => return None,
                Err(e) => return Some(Err(e)),
            }
        }
        if indet > 0 {
            if indet == width {
                return Some(Ok(Value::Uninit));
            }
            return None;
        }
        Some(self.construct_scalar(bits, shape))
    }

    /// Compose a shaped scalar load at a target's structural
    /// position through the byte view: a byte pointer (or a
    /// scalar format argument) aimed at the base of an
    /// aggregate value denotes the bytes at that position, not
    /// the aggregate — a `*const u8` at a struct's address
    /// reads the struct's first byte. The composition applies
    /// only when the value at the position is not already a
    /// scalar the shape flattens (that load is the plain one),
    /// and only when every byte of the shape's width is judged
    /// by the recorded layout; anything else is left to the
    /// existing paths.
    pub(crate) fn byte_view_compose(
        &self,
        frames: &[Frame],
        target: &Resolved,
        shape: &ScalarShape,
    ) -> Option<InterpResult<Value>> {
        if !matches!(
            shape,
            ScalarShape::Int { .. }
                | ScalarShape::Float { .. }
                | ScalarShape::Bool
                | ScalarShape::Char
        ) {
            return None;
        }
        let width = (shape.bits() / 8) as usize;
        if width == 0 || width > 16 {
            return None;
        }
        let (projection, base, base_ty, func, place_desc) =
            self.byte_view_base(frames, target)?;
        if projection
            .iter()
            .any(|s| matches!(s, RefProj::ByteOffset(_)))
        {
            return None;
        }
        let (pos, ..) = self.projection_storage_offset(&base_ty, projection)?;
        let position_value =
            project_value(base.clone(), projection, &func, &place_desc).ok()?;
        // A position value that already IS a value of the
        // shape's own kind is the plain load. One that merely
        // flattens under the shape (an unsigned integer under
        // a signed shape of the same width) is a different
        // reading of the position's bits, and the composition
        // below is its reading.
        if self.flatten_scalar(&position_value, shape).is_ok()
            && scalar_agrees_with_shape(&position_value, shape)
        {
            return None;
        }
        let layout = self.byte_layout_of_ty(&base_ty)?;
        let mut bits: u128 = 0;
        let mut indet = 0usize;
        for k in 0..width {
            match self.storage_byte(&base, &layout, pos + k)? {
                Ok(StorageByte::Byte(byte)) => {
                    bits |= u128::from(byte) << (8 * k)
                }
                // A composition over padding bytes: wholly
                // indeterminate bytes compose the indeterminate
                // value; a partly indeterminate composition
                // has no honest scalar and is not judged.
                Ok(StorageByte::Padding) => indet += 1,
                Err(verdict) => {
                    return Some(Err(self.byte_view_error(
                        verdict,
                        &func,
                        &place_desc,
                        true,
                    )))
                }
            }
        }
        if indet > 0 {
            if indet == width {
                return Some(Ok(Value::Uninit));
            }
            return None;
        }
        Some(self.construct_scalar(bits, shape))
    }

    /// The element a scalar-backed slice denotes when the
    /// slice's element type is a scalar of the storage's own
    /// width but a different kind (a punned view: a `&[f32]`
    /// over a `u32` place, a `&[u32]` over an `f32`): the
    /// storage's bytes under its declared layout, constructed
    /// through the element's shape. The storage's own value is
    /// a different reading of the same bytes, so a snapshot
    /// holding it (an iterator's element snapshot) shows the
    /// wrong value to every consumer that resolves the element
    /// directly. `None` when the view is not such a pun --
    /// either side not a scalar, the kinds in agreement, the
    /// widths apart, or the storage's bytes beyond judgement --
    /// in which case the element is the storage value itself.
    pub(crate) fn punned_scalar_element(
        &self,
        frames: &[Frame],
        target: &Resolved,
        elem_tag: &str,
    ) -> Option<Value> {
        let shape = scalar_shape_of_tag(elem_tag)?;
        let (_, _, root_ty, _, _) = self.byte_view_base(frames, target)?;
        let root_kind = match self.byte_layout_of_ty(&root_ty) {
            Some(ByteLayout::Scalar(kind)) => kind,
            _ => return None,
        };
        if scalar_kind_key(root_kind) == scalar_shape_key(&shape) {
            return None;
        }
        let width = root_kind.width();
        if width != (shape.bits() / 8) as usize {
            return None;
        }
        let bytes = self.scalar_storage_layout_bytes(frames, target, width)?;
        let mut bits: u128 = 0;
        for (i, byte) in bytes.iter().enumerate() {
            bits |= u128::from(*byte) << (8 * i);
        }
        self.construct_scalar(bits, &shape).ok()
    }

    /// The pointee a format argument of scalar type `ty`
    /// (the argument's rendered type tag) denotes through a
    /// byte-displaced target: the tag is the load's width, so
    /// the byte view composes the value exactly (see
    /// [`Self::byte_view_shaped_load`]). `None` when the tag
    /// names no scalar shape, the target is not a reference to
    /// a judged position, or the view does not judge it. The
    /// `Pointer` trait never reaches here: it asks for the
    /// argument's address, not its pointee (the caller keeps
    /// it on the plain path).
    pub(crate) fn byte_view_fmt_arg(
        &self,
        frames: &[Frame],
        target: &Value,
        ty: &str,
    ) -> Option<InterpResult<Value>> {
        let shape = scalar_shape_of_tag(ty)?;
        let mut resolved = self.deref_value(frames, target).ok()?;
        // The argument's target can denote a place holding the
        // reference itself (an argument spilled through a
        // local, the shape a raw `core::fmt::rt::Argument`
        // carries): the plain resolution follows the reference
        // chain to the pointee, and the byte view must reach
        // the same position before it can judge it. The hop
        // guard mirrors `resolve_ref`'s: the shapes that
        // resolve to themselves are not followed.
        for _ in 0..8 {
            if let Some(view) = self
                .byte_view_shaped_load(frames, &resolved, &shape)
                .or_else(|| self.byte_view_compose(frames, &resolved, &shape))
            {
                return Some(view);
            }
            let value = self.read_resolved(frames, &resolved).ok()?;
            // A position the byte view cannot place (an
            // element position of a sequence form, above all)
            // whose plainly read value flattens under the
            // shape but is not a value of the shape's own
            // kind: the storage holds the pointee's bits
            // under a different scalar kind (a punned view),
            // and the tag is the reading that counts, so the
            // value is constructed from those bits. A boolean
            // shape has no flattening from an integer value;
            // the byte it denotes is the pattern.
            if !matches!(
                value,
                Value::Ref(_)
                    | Value::FatRef { .. }
                    | Value::DynRef { .. }
            ) && !scalar_agrees_with_shape(&value, &shape)
            {
                let bits = match self.flatten_scalar(&value, &shape) {
                    Ok(bits) => Some(bits),
                    Err(_) => match (&shape, &value) {
                        (ScalarShape::Bool, Value::Uint(b)) => {
                            Some(u128::from(*b != 0))
                        }
                        (ScalarShape::Bool, Value::Int(b)) => {
                            Some(u128::from(*b != 0))
                        }
                        _ => None,
                    },
                };
                if let Some(bits) = bits {
                    if let Ok(constructed) =
                        self.construct_scalar(bits, &shape)
                    {
                        return Some(Ok(constructed));
                    }
                }
            }
            resolved = match &value {
                Value::Ref(RefTarget::Null)
                | Value::Ref(RefTarget::Address(_)) => break,
                Value::FatRef { base, .. }
                    if matches!(
                        &**base,
                        Value::Tuple(_)
                            | Value::SparseVec { .. }
                            | Value::Ref(RefTarget::Address(_))
                    ) =>
                {
                    break
                }
                Value::Ref(_) | Value::FatRef { .. } => {
                    self.deref_value(frames, &value).ok()?
                }
                _ => break,
            };
        }
        None
    }

    /// The pointee a format argument of array type `ty` (the
    /// argument's rendered type tag) denotes through a fat
    /// pointer cast that regrouped the storage's elements:
    /// the argument's target is an element place of the birth
    /// storage (a `[[u8; 3]; 2]` array) while the tag names
    /// the element under the cast's shape (`[u8; 2]`), so the
    /// element is composed from the storage's bytes at the
    /// access-side offset -- the same composition a body place
    /// read through the cast performs (see
    /// [`Self::foreign_read`]). `None` when the tag names no
    /// array type, the target is not a single index step over
    /// an array birth, the tag's shape is the birth element's
    /// own (the ordinary resolution already reads the
    /// access-side value), the span runs past the storage, or
    /// a byte cannot be judged.
    pub(crate) fn regrouped_fmt_arg(
        &self,
        frames: &[Frame],
        target: &Value,
        ty: &str,
    ) -> Option<InterpResult<Value>> {
        let mut tag = ty;
        loop {
            tag = tag.trim_start();
            if let Some(rest) = tag.strip_prefix("&mut ") {
                tag = rest;
            } else if let Some(rest) = tag.strip_prefix('&') {
                tag = rest;
            } else if let Some(rest) = tag.strip_prefix("*const ") {
                tag = rest;
            } else if let Some(rest) = tag.strip_prefix("*mut ") {
                tag = rest;
            } else {
                break;
            }
        }
        let access = crate::builtins::ty_from_tag(tag)?;
        if !matches!(access, Ty::Array(..)) {
            return None;
        }
        let resolved = self.deref_value(frames, target).ok()?;
        let (birth, projection) = self.target_birth(frames, &resolved)?;
        let Ty::Array(birth_elem, _) = &birth else {
            return None;
        };
        // A denomination the target's dereference recorded
        // (see `RefProj::View`) does not move the position:
        // stripped of it, the step sequence is the element
        // place this composition prices, and its index
        // counts in the recorded view's elements -- the
        // shape the tag names (`access`), so a recorded
        // sequence view denominating by another element is
        // not this composition.
        for step in projection {
            if let RefProj::View(Ty::Slice(e) | Ty::Array(e, _)) = step {
                if **e != access {
                    return None;
                }
            }
        }
        let mut steps = projection
            .iter()
            .filter(|s| !matches!(s, RefProj::View(_)));
        let (Some(RefProj::Index(k)), None) = (steps.next(), steps.next())
        else {
            return None;
        };
        let k = *k;
        if access == **birth_elem {
            return None;
        }
        let elem_size = usize::try_from(self.ty_size(&access)?).ok()?;
        if elem_size == 0 {
            return None;
        }
        let off = k.checked_mul(elem_size)?;
        let birth_size = usize::try_from(self.ty_size(&birth)?).ok()?;
        if off.checked_add(elem_size)? > birth_size {
            return None;
        }
        let root = self.target_root_value(frames, &resolved)?;
        let layout = self.byte_layout_of_ty(&birth)?;
        let mut bytes = Vec::with_capacity(elem_size);
        for i in 0..elem_size {
            match self.foreign_byte(&root, &layout, off + i)? {
                ForeignByte::Byte(b) => bytes.push(Some(b)),
                ForeignByte::Padding => bytes.push(None),
                ForeignByte::Uninit => return None,
            }
        }
        self.construct_from_bytes(&access, &bytes).map(Ok)
    }

    /// A shaped whole-pointee store through a byte-displaced
    /// target (see [`Self::shaped_store`]): the value's bytes
    /// are spliced into a clone of the base value through the
    /// byte view, and the composed base is written back whole,
    /// so a position the view cannot judge leaves the store
    /// exactly as it was (the caller keeps the existing path)
    /// and a judged position is never half-written.
    pub(crate) fn byte_view_shaped_store(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: &Value,
        shape: &ScalarShape,
    ) -> Option<InterpResult<()>> {
        let bits = self.flatten_scalar(value, shape).ok()?;
        let width = (shape.bits() / 8) as usize;
        if width == 0 || width > 16 {
            return None;
        }
        let (projection, mut base, base_ty, func, place_desc) =
            self.byte_view_base(frames, target)?;
        let pos = if projection
            .iter()
            .any(|s| matches!(s, RefProj::ByteOffset(_)))
        {
            match self.byte_view_position(&base_ty, projection, 0)? {
                Ok(p) => p,
                Err(neg) => {
                    return Some(Err(InterpError::UnsupportedProjection {
                        kind: format!(
                            "write through a pointer displaced {neg} bytes from a value boundary"
                        ),
                    }))
                }
            }
        } else {
            // A shaped store aimed at a structural position
            // whose storage is wider than the shape — a byte
            // store at an element's first byte, whose target
            // the offset evaluation absorbed into the element
            // index, or at an aggregate's base: the plain
            // whole-value store would replace the whole element
            // or aggregate with the byte, so the shape's bytes
            // are spliced at the position instead. A position
            // the shape already covers exactly (a plain scalar
            // store) keeps the whole-value store.
            let (off, pos_ty, pos_size) = self.projection_storage_offset(&base_ty, projection)?;
            let position_value =
                project_value(base.clone(), projection, &func, &place_desc).ok()?;
            let wider_scalar = matches!(
                (&pos_ty, &position_value),
                (Some(t), Value::Int(_) | Value::Uint(_) | Value::Float(_) | Value::Float128(_) | Value::Bool(_))
                    if self.ty_size(t).is_some_and(|s| s as usize > width)
            );
            let aggregate = matches!(
                position_value,
                Value::Tuple(_) | Value::Enum { .. } | Value::Bytes(_) | Value::Str(_)
            );
            // A shape covering the position exactly is a
            // whole-value store of that position: the plain
            // store denominates the value by the position's
            // own type, while splicing the shape's flattened
            // bytes through the base's byte layout rebuilds
            // the position from its storage image and loses
            // the denomination of any field whose byte
            // layout is a bare scalar (a newtype payload
            // stored through a dereference -- the
            // `Capture::elem` store of an `assert!` capture
            // -- comes back as the bare integer). The splice
            // exists for a shape narrower than the position
            // it lands in, whose whole-value store would
            // replace more than the shape denotes.
            if let Some(t) = &pos_ty {
                if self
                    .ty_size(t)
                    .is_some_and(|s| s as usize == width)
                {
                    return None;
                }
            }
            // The type evidence above ends where the walk's
            // type does — at a union field, whose tag names no
            // laid-out instance (union layouts are not recorded
            // in the field layout tables) — but the walk still
            // priced the position's size out of the same layout
            // rows the splice descends, and a shape covering
            // that size exactly is the same whole-value store:
            // splicing it rebuilds the position from its
            // storage image, so a union position takes the byte
            // form, which the payload-descent reads that a
            // `MaybeUninit` place's readers perform cannot
            // denominate (they carry no union table), where the
            // plain store keeps the value's own form.
            if pos_size.is_some_and(|s| s == width) {
                return None;
            }
            if !wider_scalar && !aggregate {
                return None;
            }
            off
        };
        let layout = self.byte_layout_of_ty(&base_ty)?;
        for k in 0..width {
            let byte = ((bits >> (8 * k)) & 0xff) as u8;
            match self.splice_storage_byte(&mut base, &layout, pos + k, byte)? {
                Ok(()) => {}
                Err(ByteVerdict::Padding(_)) => {}
                Err(verdict) => {
                    return Some(Err(self.byte_view_error(
                        verdict,
                        &func,
                        &place_desc,
                        false,
                    )))
                }
            }
        }
        let base_target = match target {
            Resolved::Place {
                frame_pos, local, ..
            } => Resolved::Place {
                frame_pos: *frame_pos,
                local: *local,
                projection: Vec::new(),
            },
            Resolved::Static { name, .. } => Resolved::Static {
                name: name.clone(),
                projection: Vec::new(),
            },
            Resolved::Alloc { id, .. } => Resolved::Alloc {
                id: *id,
                projection: Vec::new(),
            },
            _ => return None,
        };
        Some(self.write_resolved(frames, &base_target, base))
    }

    /// Read the byte a byte-displaced place/static target
    /// denotes, `extra` bytes past the displaced position (a
    /// sequence access's byte index; zero for a direct deref of
    /// a byte pointer, whose caller has established the access
    /// is one byte wide). `None` when the byte view does not
    /// judge the target (see the byte view above); the caller
    /// then keeps the existing path. A sequence access over a
    /// bare scalar target is the existing indexed byte view's
    /// own case (its storage is the widened slot's), so the
    /// view declines it there and judges scalars only for the
    /// direct deref, whose width is the declared type's.
    pub(crate) fn byte_view_read(
        &self,
        frames: &[Frame],
        target: &Resolved,
        extra: i128,
        seq: bool,
    ) -> Option<InterpResult<Value>> {
        // A value target holding an erased constant block: the
        // block's bytes are the storage itself, so a byte read
        // needs no layout -- the position is the projection's
        // byte displacement (a step that denominates in elements
        // of an unknown size is not priceable over raw bytes and
        // declines, as does a sequence access, whose caller owns
        // the element denomination). A byte the constant left
        // uninitialised reads as the indeterminate value, as a
        // padding byte does below.
        if !seq {
            if let Resolved::Value(base, projection) = target {
                if let Value::ErasedBlock { bytes, .. } = base {
                    let mut pos: i128 = extra;
                    let mut priceable = true;
                    for step in projection {
                        match step {
                            RefProj::ByteOffset(d) => pos += d,
                            RefProj::Index(0) | RefProj::View(_) => {}
                            _ => {
                                priceable = false;
                                break;
                            }
                        }
                    }
                    if priceable && pos >= 0 {
                        match bytes.get(usize::try_from(pos).unwrap_or(usize::MAX)) {
                            Some(Some(byte)) => {
                                return Some(Ok(Value::Uint(u128::from(*byte))));
                            }
                            Some(None) => return Some(Ok(Value::Uninit)),
                            None => {}
                        }
                    }
                }
            }
        }
        let (projection, base, base_ty, func, place_desc) =
            self.byte_view_base(frames, target)?;
        let pos = match self.byte_view_position(&base_ty, projection, extra)? {
            Ok(p) => p,
            Err(neg) => {
                return Some(Err(InterpError::UnsupportedProjection {
                    kind: format!(
                        "read through a pointer displaced {neg} bytes from a value boundary"
                    ),
                }))
            }
        };
        let layout = self.byte_layout_of_ty(&base_ty)?;
        if seq && matches!(layout, ByteLayout::Scalar(_)) {
            return None;
        }
        match self.storage_byte(&base, &layout, pos)? {
            Ok(StorageByte::Byte(byte)) => {
                Some(Ok(Value::Uint(u128::from(byte))))
            }
            // A padding byte has no definite value: the read
            // yields the model's indeterminate value, which a
            // later use judges like any uninitialised read.
            Ok(StorageByte::Padding) => Some(Ok(Value::Uninit)),
            Err(verdict) => {
                Some(Err(self.byte_view_error(verdict, &func, &place_desc, true)))
            }
        }
    }

    /// A whole-value store of an aggregate (array) value
    /// through a byte-displaced target (see
    /// [`Self::byte_view_shaped_store`] for the scalar form):
    /// the value's byte image is spliced into a clone of the
    /// base value through the byte view at the displaced
    /// position, and the composed base is written back whole.
    /// This is the store a transmuted reference performs when
    /// it denotes an array at a byte offset inside differently
    /// typed storage (a `[u8; 32]` written one byte into a
    /// `[u64; 8]`): the position crosses the base storage's
    /// element boundaries, so no structural projection can
    /// denote it, but every byte of it is judged by the byte
    /// view. `None` when the target is not a displaced position
    /// this view judges, the value has no complete byte image,
    /// or a byte cannot be spliced: the caller then keeps the
    /// existing path and its verdicts.
    pub(crate) fn byte_view_displaced_store(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: &Value,
        place_ty: &Ty,
    ) -> Option<InterpResult<()>> {
        let size = usize::try_from(self.ty_size(place_ty)?).ok()?;
        if size == 0 {
            return None;
        }
        let (projection, mut base, base_ty, func, place_desc) =
            self.byte_view_base(frames, target)?;
        if !projection
            .iter()
            .any(|s| matches!(s, RefProj::ByteOffset(_)))
        {
            return None;
        }
        let pos = match self.byte_view_position(&base_ty, projection, 0)? {
            Ok(p) => p,
            Err(neg) => {
                return Some(Err(InterpError::UnsupportedProjection {
                    kind: format!(
                        "write through a pointer displaced {neg} bytes from a value boundary"
                    ),
                }))
            }
        };
        let bytes = self.written_value_bytes(value, place_ty, size)?;
        let layout = self.byte_layout_of_ty(&base_ty)?;
        for (i, b) in bytes.iter().enumerate() {
            let Some(b) = b else { continue };
            match self.splice_storage_byte(&mut base, &layout, pos + i, *b)? {
                Ok(()) => {}
                Err(ByteVerdict::Padding(_)) => {}
                Err(verdict) => {
                    return Some(Err(self.byte_view_error(
                        verdict,
                        &func,
                        &place_desc,
                        false,
                    )))
                }
            }
        }
        let base_target = match target {
            Resolved::Place {
                frame_pos, local, ..
            } => Resolved::Place {
                frame_pos: *frame_pos,
                local: *local,
                projection: Vec::new(),
            },
            Resolved::Static { name, .. } => Resolved::Static {
                name: name.clone(),
                projection: Vec::new(),
            },
            Resolved::Alloc { id, .. } => Resolved::Alloc {
                id: *id,
                projection: Vec::new(),
            },
            _ => return None,
        };
        Some(self.write_resolved(frames, &base_target, base))
    }

    /// The load side of [`Self::byte_view_displaced_store`]:
    /// the bytes at the displaced position are read through
    /// the byte view of the base value's storage and composed
    /// into a value of the place's declared type. `None` under
    /// the same conditions as the store; a judged
    /// uninitialised byte in the span is the load's error, as
    /// on the foreign path.
    pub(crate) fn byte_view_displaced_load(
        &self,
        frames: &[Frame],
        target: &Resolved,
        place_ty: &Ty,
    ) -> Option<InterpResult<Value>> {
        let size = usize::try_from(self.ty_size(place_ty)?).ok()?;
        if size == 0 {
            return None;
        }
        let (projection, base, base_ty, func, place_desc) =
            self.byte_view_base(frames, target)?;
        if !projection
            .iter()
            .any(|s| matches!(s, RefProj::ByteOffset(_)))
        {
            return None;
        }
        let pos = match self.byte_view_position(&base_ty, projection, 0)? {
            Ok(p) => p,
            Err(neg) => {
                return Some(Err(InterpError::UnsupportedProjection {
                    kind: format!(
                        "read through a pointer displaced {neg} bytes from a value boundary"
                    ),
                }))
            }
        };
        let layout = self.byte_layout_of_ty(&base_ty)?;
        let mut bytes = Vec::with_capacity(size);
        for i in 0..size {
            match self.foreign_byte(&base, &layout, pos + i)? {
                ForeignByte::Byte(b) => bytes.push(Some(b)),
                ForeignByte::Padding => bytes.push(None),
                ForeignByte::Uninit => {
                    return Some(Err(InterpError::UninitializedRead {
                        func,
                        place: place_desc,
                    }))
                }
            }
        }
        self.construct_from_bytes(place_ty, &bytes).map(Ok)
    }

    /// Write `value` (a byte value) through a byte-displaced
    /// place/static target, `extra` bytes past the displaced
    /// position (see [`Self::byte_view_read`] for the contract).
    pub(crate) fn byte_view_write(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        extra: i128,
        value: &Value,
        seq: bool,
    ) -> Option<InterpResult<()>> {
        let byte = value_as_storage_byte(value)?;
        let (projection, base_ty, func, place_desc) = match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let frame = frames.get(*frame_pos)?;
                let body = self.program().functions.get(&frame.func)?;
                let base_ty = body.local_tys.get(*local)?.clone();
                (
                    projection,
                    base_ty,
                    frame.func.clone(),
                    describe_resolved(*local, projection),
                )
            }
            Resolved::Static { name, projection } => {
                let base_ty = self.program().static_tys.get(name)?.clone();
                (
                    projection,
                    base_ty,
                    "<static>".to_string(),
                    format!("static {name}"),
                )
            }
            Resolved::Alloc { id, projection } => {
                if self.is_elem_block(*id) {
                    return None;
                }
                let base_ty = self.alloc_ty(*id)?;
                (
                    projection,
                    base_ty,
                    "<allocation>".to_string(),
                    format!("allocation #{id}"),
                )
            }
            _ => return None,
        };
        let pos = match self.byte_view_position(&base_ty, projection, extra)? {
            Ok(p) => p,
            Err(neg) => {
                return Some(Err(InterpError::UnsupportedProjection {
                    kind: format!(
                        "write through a pointer displaced {neg} bytes from a value boundary"
                    ),
                }))
            }
        };
        let layout = self.byte_layout_of_ty(&base_ty)?;
        if seq && matches!(layout, ByteLayout::Scalar(_)) {
            return None;
        }
        let verdict = match target {
            Resolved::Place {
                frame_pos, local, ..
            } => {
                let slot = frames.get_mut(*frame_pos)?.locals.get_mut(*local)?;
                self.splice_storage_byte(slot, &layout, pos, byte)?
            }
            Resolved::Static { name, .. } => {
                let mut store = self.statics.borrow_mut();
                let slot = store.get_mut(name)?;
                self.splice_storage_byte(slot, &layout, pos, byte)?
            }
            Resolved::Alloc { id, .. } => {
                let mut store = self.allocs.borrow_mut();
                let slot = store.get_mut(id)?;
                self.splice_storage_byte(slot, &layout, pos, byte)?
            }
            _ => return None,
        };
        Some(match verdict {
            Ok(()) => Ok(()),
            // A byte written into padding is discarded: padding
            // holds no value (the read side symmetrically
            // yields the indeterminate value for it), and
            // byte-wise copies over an aggregate write its
            // padding bytes along with its fields.
            Err(ByteVerdict::Padding(_)) => Ok(()),
            Err(verdict) => Err(self.byte_view_error(verdict, &func, &place_desc, false)),
        })
    }

    /// The declared type of the value a place denotes, walked
    /// over the place's own projection (the type a direct deref
    /// of the place reads or writes). `None` when a step's type
    /// is not determined by the local's declared type alone (a
    /// field of an ADT, a deref of an ADT-typed local such as a
    /// `Box`).
    pub(crate) fn place_result_ty(&self, body: &Body, place: &Place) -> Option<Ty> {
        let mut ty = body.local_ty(place.local)?.clone();
        // The variant the walk is inside: field steps on an ADT
        // read the field type of the variant a preceding downcast
        // selected (0 when none did -- a struct's only variant,
        // or a field step the MIR spells without one).
        let mut variant = 0usize;
        for elem in &place.projection {
            ty = projection_step_ty(&ty, elem)?;
        }
        Some(ty)
    }

    /// The declared type of the value at the root of the target
    /// a place resolves to: the place walk's type after its last
    /// `Deref` step (the pointee type the resolved target's own
    /// projection starts from), or the local's declared type
    /// when the place never dereferences. Projected-assignment
    /// navigation walks this type alongside the resolved
    /// projection so its wrapper judgments can consult the
    /// declared types (see `wrapper_descent_sanctioned`).
    pub(crate) fn resolved_root_ty(&self, body: &Body, place: &Place) -> Option<Ty> {
        let mut ty = body.local_ty(place.local)?.clone();
        let mut root = ty.clone();
        let mut variant = 0usize;
        for elem in &place.projection {
            match (elem, &ty) {
                (ProjectionElem::Deref, Ty::Ref(inner)) => {
                    ty = (**inner).clone();
                    root = ty.clone();
                    variant = 0;
                }
                (ProjectionElem::Field(i), Ty::Tuple(ts)) => {
                    ty = ts.get(*i)?.clone();
                }
                (ProjectionElem::Field(i), Ty::Adt(_)) => {
                    ty = self.adt_field_ty(&ty, variant, *i)?;
                }
                (ProjectionElem::Index(_), Ty::Array(elem, _))
                | (ProjectionElem::Index(_), Ty::Slice(elem))
                | (ProjectionElem::ConstantIndex { .. }, Ty::Array(elem, _))
                | (ProjectionElem::ConstantIndex { .. }, Ty::Slice(elem)) => {
                    ty = (**elem).clone();
                }
                (ProjectionElem::Downcast(v), _) => {
                    variant = *v as usize;
                }
                (ProjectionElem::Subslice { .. }, Ty::Array(elem, _)) => {
                    ty = Ty::Array(elem.clone(), 0);
                }
                (ProjectionElem::Subslice { .. }, Ty::Slice(elem)) => {
                    ty = Ty::Slice(elem.clone());
                }
                _ => return None,
            }
        }
        Some(root)
    }

    /// Whether the byte view judges byte positions over the
    /// position `projection` denotes in a place/static target:
    /// the position's declared type is an ADT instance with a
    /// recorded field layout and the value there is the ADT
    /// aggregate. Over such a position a byte displacement
    /// denotes a storage byte, never an element index into the
    /// field vector, so `offset_projection` must not absorb it
    /// into one: an ADT's fields are not its elements, and a
    /// byte offset reinterpreted as a field index names a
    /// different field than the byte's (a byte-2 read over a
    /// three-field struct returned the third field's first
    /// byte). Positions the view cannot judge — a tuple field's
    /// interior, an ADT with no recorded layout, a value that
    /// is not the aggregate — keep the absorption unchanged.
    pub(crate) fn byte_view_covers(
        &self,
        frames: &[Frame],
        place: Option<(u64, usize)>,
        static_name: Option<&str>,
        projection: &[RefProj],
        base: Option<&Value>,
    ) -> bool {
        let base_ty = match (static_name, place) {
            (Some(name), _) => self.program().static_tys.get(name).cloned(),
            (None, Some((frame_id, local))) => {
                let pos = self.frame_pos(frames, frame_id).ok();
                pos.and_then(|pos| {
                    let frame = frames.get(pos)?;
                    let body = self.program().functions.get(&frame.func)?;
                    body.local_tys.get(local).cloned()
                })
            }
            (None, None) => None,
        };
        let Some(base_ty) = base_ty else { return false };
        let Some((_, Some(Ty::Adt(name)), _)) =
            self.projection_storage_offset(&base_ty, projection)
        else {
            return false;
        };
        // The recorded field layout lives under the
        // rendered instance name (see `adt_instance_key`).
        let key = self.adt_instance_key(&name).unwrap_or(name);
        if !self.program().adt_field_layouts.contains_key(&key) {
            return false;
        }
        let Some(base) = base else { return false };
        matches!(
            project_value(base.clone(), projection, "<offset>", "pointer target"),
            Ok(Value::Enum { .. })
        )
    }

    /// The integer a sequence target denotes as a whole, when a
    /// bulk-memory operation's pointer names a scalar place: the
    /// address of an integer local cast to a byte pointer (the
    /// shape `ptr::copy_nonoverlapping` into a `u32` in
    /// `core::hash::sip`'s integer loads takes) makes the
    /// operation's sequence the scalar's own bytes, which the
    /// indexed access cannot express. A target ending in an index
    /// step denotes one element of an aggregate instead and keeps
    /// the element treatment, as does any non-integer value.
    pub(crate) fn seq_scalar(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let projection = match target {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. }
            | Resolved::Value(_, projection) => projection,
        };
        if matches!(projection.last(), Some(RefProj::Index(_))) {
            return Ok(None);
        }
        // A byte-displaced target denotes a byte inside the
        // scalar's storage, not the scalar as a whole: the
        // indexed byte view (read / write _resolved_at) owns
        // those reads, and the whole-value read of a displaced
        // position is an unsupported projection, not a scalar.
        if projection
            .iter()
            .any(|s| matches!(s, RefProj::ByteOffset(d) if *d != 0))
        {
            return Ok(None);
        }
        match self.read_resolved(frames, target) {
            Ok(v @ (Value::Int(_) | Value::Uint(_))) => Ok(Some(v)),
            Ok(_) => Ok(None),
            Err(InterpError::UninitializedRead { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Whether `target` denotes a scalar value whose declared
    /// type is a scalar of width `width`: the target's storage
    /// is then exactly one element of that width (the shape
    /// `slice::from_ref` / `slice::from_mut` over a scalar
    /// place forms — the fat pointer's base is the scalar's own
    /// place). The target must denote the storage itself (an
    /// empty projection): a target ending at an element
    /// position, or displaced into the storage, denotes a part
    /// of some other value, which the sibling and byte readings
    /// judge instead.
    pub(crate) fn target_is_scalar_storage(
        &self,
        frames: &[Frame],
        target: &Resolved,
        width: usize,
    ) -> bool {
        let Some((projection, _, root_ty, _, _)) =
            self.byte_view_base(frames, target)
        else {
            return false;
        };
        if !projection_denotes_root(projection) {
            return false;
        }
        let root_width = match self.byte_layout_of_ty(&root_ty) {
            Some(ByteLayout::Scalar(kind)) => kind.width(),
            _ => return false,
        };
        if root_width != width {
            return false;
        }
        matches!(
            self.read_resolved(frames, target),
            Ok(v) if scalar_storage_len(&v).is_some()
        )
    }

    /// The first `count` bytes of the scalar storage
    /// `target` denotes, read under the storage's declared
    /// layout (see `storage_byte`): a float storage yields
    /// its own width's bytes, not the widened model's. `None`
    /// when the target does not denote the storage itself (an
    /// empty projection) as a scalar with a known layout, or
    /// a byte position holds no definite storage byte.
    pub(crate) fn scalar_storage_layout_bytes(
        &self,
        frames: &[Frame],
        target: &Resolved,
        count: usize,
    ) -> Option<Vec<u8>> {
        let (projection, _, root_ty, _, _) =
            self.byte_view_base(frames, target)?;
        if !projection_denotes_root(projection) {
            return None;
        }
        let layout = self.byte_layout_of_ty(&root_ty)?;
        if !matches!(layout, ByteLayout::Scalar(_)) {
            return None;
        }
        let scalar = self.read_resolved(frames, target).ok()?;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            match self.storage_byte(&scalar, &layout, i) {
                Some(Ok(StorageByte::Byte(b))) => out.push(b),
                _ => return None,
            }
        }
        Some(out)
    }

    /// Whether the position `target` resolves to is the root
    /// of a scalar storage viewed through a byte-width element
    /// of a different scalar kind (a `&[u8]` over a `bool`):
    /// the position denotes the storage's byte, not the
    /// scalar. Only a positively identified mismatch answers
    /// true; an element type that is not a byte-width scalar,
    /// or a root kind that cannot be determined, answers
    /// false.
    pub(crate) fn scalar_storage_is_byte_view(
        &self,
        frames: &[Frame],
        target: &Resolved,
        elem_ty: Option<&Ty>,
    ) -> bool {
        let Some(elem_ty) = elem_ty else {
            return false;
        };
        let elem_kind = match self.byte_layout_of_ty(elem_ty) {
            Some(ByteLayout::Scalar(kind)) if kind.width() == 1 => kind,
            _ => return false,
        };
        let Some((projection, _, root_ty, _, _)) =
            self.byte_view_base(frames, target)
        else {
            return false;
        };
        if !projection_denotes_root(projection) {
            return false;
        }
        match self.byte_layout_of_ty(&root_ty) {
            Some(ByteLayout::Scalar(root_kind)) => root_kind != elem_kind,
            _ => false,
        }
    }

    /// Whether the position a fat dereference left in `target`
    /// is the storage of a scalar-backed slice whose element
    /// type is `elem_ty`: the element is then the scalar the
    /// target denotes itself (see
    /// [`Self::target_is_scalar_storage`]). A byte-width
    /// element of a different scalar kind never denotes the
    /// storage: a byte view of a boolean's storage (a `&[u8]`
    /// over a `bool`) denotes the storage's byte, not the
    /// boolean, and the byte projection owns those reads. (A
    /// wider element of a different kind but the storage's
    /// width still denotes the storage: a typed read of the
    /// position reinterprets the storage's bytes, while the
    /// byte projection reads only one byte of it.) An element
    /// type with no scalar layout (an ADT, a reference, ...)
    /// never denotes the storage.
    pub(crate) fn fat_denotes_scalar_element(
        &self,
        frames: &[Frame],
        target: &Resolved,
        elem_ty: Option<&Ty>,
    ) -> bool {
        let Some(elem_ty) = elem_ty else {
            return false;
        };
        let elem_kind = match self.byte_layout_of_ty(elem_ty) {
            Some(ByteLayout::Scalar(kind)) => kind,
            _ => return false,
        };
        if elem_kind.width() == 1 {
            let Some((projection, _, root_ty, _, _)) =
                self.byte_view_base(frames, target)
            else {
                return false;
            };
            if projection_denotes_root(projection) {
                let root_kind = match self.byte_layout_of_ty(&root_ty) {
                    Some(ByteLayout::Scalar(kind)) => kind,
                    _ => return false,
                };
                if root_kind != elem_kind {
                    return false;
                }
            }
        }
        self.target_is_scalar_storage(frames, target, elem_kind.width())
    }

    /// Whether the position a fat dereference left in `target`
    /// is a scalar's own storage while the slice's element type
    /// `elem_ty` is a scalar of a different width (and not a
    /// byte): a punned view of the storage, whose elements the
    /// storage does not hold as values. The bare index step's
    /// reading of such a position is a byte of the storage, so
    /// the access must decline explicitly rather than read one.
    pub(crate) fn fat_denotes_punned_scalar_element(
        &self,
        frames: &[Frame],
        target: &Resolved,
        elem_ty: Option<&Ty>,
    ) -> bool {
        let Some(elem_ty) = elem_ty else {
            return false;
        };
        let elem_width = match self.byte_layout_of_ty(elem_ty) {
            Some(ByteLayout::Scalar(kind)) => kind.width(),
            _ => return false,
        };
        if elem_width == 1 {
            return false;
        }
        let Some((projection, _, root_ty, _, _)) =
            self.byte_view_base(frames, target)
        else {
            return false;
        };
        if !projection_denotes_root(projection) {
            return false;
        }
        let root_width = match self.byte_layout_of_ty(&root_ty) {
            Some(ByteLayout::Scalar(kind)) => kind.width(),
            _ => return false,
        };
        if root_width == elem_width {
            return false;
        }
        matches!(
            self.read_resolved(frames, target),
            Ok(v) if scalar_storage_len(&v).is_some()
        )
    }

    /// Read an integer of `bits` width at `target` by composing
    /// the bytes of the byte sequence the target addresses, when
    /// the target denotes one byte of such a sequence: the shape
    /// a type-punned pointer has (a `*const u16` into a `[u8]`
    /// buffer, the access `unaligned_volatile_load` performs).
    /// The interpreter's values are not byte-addressed, so a
    /// wider-than-byte access through such a pointer can only be
    /// honoured by composing the stored bytes (little-endian, the
    /// only target layout the model uses). `None` when the target
    /// does not denote a byte sequence position with enough bytes
    /// left — the plain element read is then the access's meaning.
    pub(crate) fn read_punned_int(
        &self,
        frames: &[Frame],
        target: &Resolved,
        bits: u32,
        signed: bool,
    ) -> InterpResult<Option<Value>> {
        let n = (bits / 8) as usize;
        if n <= 1 {
            return Ok(None);
        }
        let mut raw: u128 = 0;
        for k in 0..n {
            let byte = match self.read_seq_at(frames, target, k) {
                Ok(Value::Uint(b)) if b <= 0xFF => b,
                Ok(Value::Int(b)) if (0..=0xFF).contains(&b) => b as u128,
                _ => return Ok(None),
            };
            raw |= byte << (8 * k);
        }
        if signed {
            let shift = 128 - bits;
            Ok(Some(Value::Int(((raw << shift) as i128) >> shift)))
        } else {
            let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
            Ok(Some(Value::Uint(raw & mask)))
        }
    }

    /// The store matching [`Self::read_punned_int`]: decompose
    /// `value`'s low `bits` into bytes over the byte sequence at
    /// `target`. Returns `false` (and writes nothing) when the
    /// target does not denote a byte sequence position with
    /// enough bytes left.
    pub(crate) fn write_punned_int(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        bits: u32,
        value: &Value,
    ) -> InterpResult<bool> {
        let n = (bits / 8) as usize;
        if n <= 1 {
            return Ok(false);
        }
        // Probe first: every byte of the access must land on a
        // stored byte, so a non-byte target writes nothing.
        for k in 0..n {
            match self.read_seq_at(frames, target, k) {
                Ok(Value::Uint(b)) if b <= 0xFF => {}
                Ok(Value::Int(b)) if (0..=0xFF).contains(&b) => {}
                _ => return Ok(false),
            }
        }
        let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
        let raw = match value {
            Value::Uint(u) => *u & mask,
            Value::Int(i) => (*i as u128) & mask,
            _ => return Ok(false),
        };
        for k in 0..n {
            let byte = (raw >> (8 * k)) & 0xFF;
            self.write_seq_at(frames, target, k, Value::Uint(byte))?;
        }
        Ok(true)
    }

    /// The number of slots one element of a bulk access spans
    /// in the sequence `target` denotes, when the access's
    /// element type is wider than the sequence's own elements
    /// (a `usize` copy into a byte array's storage reached
    /// through a unit-changing pointer cast chain: the
    /// sequence's slots are bytes, so one element spans eight).
    /// The sequence's own element size is the one conversion
    /// recorded for the target's field path (the same
    /// `field_elem_sizes` record the pointer arithmetic model
    /// prices with), or the array element size for a bare
    /// array local. Any shape whose sizes are not recorded, do
    /// not divide evenly, or match already spans one slot.
    fn seq_slot_stride(
        &self,
        frames: &[Frame],
        target: &Resolved,
        path: &[RefProj],
        elem: Option<&Ty>,
    ) -> usize {
        let Some(elem) = elem else { return 1 };
        let Some(elem_size) = self.ty_size(elem) else {
            return 1;
        };
        let Resolved::Place {
            frame_pos, local, ..
        } = target
        else {
            return 1;
        };
        let Some(frame) = frames.get(*frame_pos) else {
            return 1;
        };
        let Some(body) = self.program().functions.get(&frame.func) else {
            return 1;
        };
        let mut field_path = Vec::new();
        for step in path {
            match step {
                RefProj::Field(f) => field_path.push(*f),
                _ => return 1,
            }
        }
        let target_elem_size = if field_path.is_empty() {
            match body.local_tys.get(*local) {
                Some(Ty::Array(elem_ty, _)) => self.ty_size(elem_ty),
                _ => None,
            }
        } else {
            body.field_elem_sizes
                .get(*local)
                .and_then(|sizes| sizes.get(&field_path))
                .copied()
        };
        match target_elem_size {
            Some(t) if t > 0 && elem_size % t == 0 => (elem_size / t) as usize,
            _ => 1,
        }
    }

    /// The size in bytes of one slot of the sequence `target`
    /// denotes at `path` (the target's projection up to the
    /// sequence's base), when conversion recorded it: the
    /// `field_elem_sizes` record for the field path (whose
    /// components are field indices -- a union field step is a
    /// field index in that record, as conversion walks unions
    /// like structs), or the array element size for a bare
    /// array local. A path with no record of its own falls
    /// back to the declared types along the path, walked
    /// through the fields' recorded type tags, when the path
    /// ends at an array. `None` for every other shape: a
    /// non-place target, a path with a step that is not a
    /// field step, or no record and no such walk for the path.
    pub(crate) fn seq_slot_size(
        &self,
        frames: &[Frame],
        target: &Resolved,
        path: &[RefProj],
    ) -> Option<u64> {
        let Resolved::Place {
            frame_pos, local, ..
        } = target
        else {
            return None;
        };
        let frame = frames.get(*frame_pos)?;
        let body = self.program().functions.get(&frame.func)?;
        let mut field_path = Vec::new();
        for step in path {
            match step {
                RefProj::Field(f) => field_path.push(*f),
                RefProj::UnionField { idx, .. } => field_path.push(*idx),
                _ => return None,
            }
        }
        if field_path.is_empty() {
            match body.local_tys.get(*local) {
                Some(Ty::Array(elem_ty, _)) => self.ty_size(elem_ty),
                _ => None,
            }
        } else if let Some(size) = body
            .field_elem_sizes
            .get(*local)
            .and_then(|sizes| sizes.get(&field_path))
            .copied()
        {
            Some(size)
        } else {
            // No record for the path: conversion's walk does
            // not enter every wrapper (a union's fields above
            // all), so a path through such a wrapper to an
            // array field has no record of its own. The array's
            // element size is still derivable from the declared
            // type the path denotes.
            match self.target_denoted_ty(frames, target, path) {
                Some(Ty::Array(elem, _)) => self.ty_size(&elem),
                _ => None,
            }
        }
    }

    /// The declared type a place target's projection denotes,
    /// walked through the program's field-type table (the walk
    /// [`Self::seq_slot_size`]'s fallback takes). `None` when
    /// the target is not a place, a step is not a field step,
    /// or a field's type is not recorded.
    pub(crate) fn target_denoted_ty(
        &self,
        frames: &[Frame],
        target: &Resolved,
        path: &[RefProj],
    ) -> Option<Ty> {
        let Resolved::Place {
            frame_pos, local, ..
        } = target
        else {
            return None;
        };
        let frame = frames.get(*frame_pos)?;
        let body = self.program().functions.get(&frame.func)?;
        let mut ty = body.local_tys.get(*local)?.clone();
        let mut variant = 0usize;
        for step in path {
            let (i, union) = match step {
                RefProj::Field(i) => (*i, false),
                RefProj::UnionField { idx, .. } => (*idx, true),
                RefProj::Downcast(v) => {
                    variant = *v as usize;
                    continue;
                }
                _ => return None,
            };
            let Ty::Adt(name) = &ty else {
                return None;
            };
            let name = name.clone();
            ty = self.instance_field_ty(
                &name,
                &ty,
                if union { 0 } else { variant },
                i,
            )?;
            variant = 0;
        }
        Some(ty)
    }

    /// The directly-resolved target of element `index` of the
    /// sequence starting at `target`, when `target` denotes a
    /// single element of it (see [`Self::read_seq_at`]); `None`
    /// when the plain indexed access applies. An unreadable
    /// element (a buffer being filled element by element) still
    /// denotes its position, so the adjustment applies to it as
    /// well; any other read failure is the failure the plain
    /// access would report, so it propagates unchanged. A value
    /// target carries its projection like the other targets, so
    /// an element pointer into a constant aggregate adjusts the
    /// same way; the index step the access composes with may
    /// sit behind whole-value entry steps (a transparent
    /// wrapper's field-0 entry, a downcast), which the composed
    /// target keeps; a target with no such index step takes the
    /// plain access.
    pub(crate) fn seq_element_target(
        &self,
        frames: &[Frame],
        target: &Resolved,
        index: usize,
        elem: Option<&Ty>,
    ) -> InterpResult<Option<Resolved>> {
        // A byte-denominated allocation target (see
        // `elem_block_projection`) reads its elements at the
        // byte slot, as the plain read/write paths do.
        let byte_translated = match target {
            Resolved::Alloc { id, projection } => Some(Resolved::Alloc {
                id: *id,
                projection: self.elem_block_projection(*id, projection),
            }),
            _ => None,
        };
        let target = byte_translated.as_ref().unwrap_or(target);
        // A raw element block target carrying a single index
        // step denotes a position in the block's flat sequence:
        // the sequence continues at the following slots, never
        // inside the value stored at this slot (which may be a
        // structured element a byte-granular copy moves whole).
        if let Resolved::Alloc { id, projection } = target {
            if self.is_elem_block(*id) {
                if let [RefProj::Index(a)] = projection.as_slice() {
                    return Ok(Some(Resolved::Alloc {
                        id: *id,
                        projection: vec![RefProj::Index(a + index)],
                    }));
                }
            }
        }
        // A raw element block target carrying two index steps —
        // an offset absorbed into an element index, followed by
        // the first-element step of an array-to-element cast —
        // denotes the flat sequence starting at the sum of the
        // two indices, unless the value stored at the first
        // index is itself an aggregate the second index ranges
        // over (a nested array value in the block, including
        // the single-array-field pun `project_value` reads
        // through): that shape keeps the trailing-index
        // reading below.
        if let Resolved::Alloc { id, projection } = target {
            if self.is_elem_block(*id) {
                if let [RefProj::Index(a), RefProj::Index(b)] = projection.as_slice() {
                    let (a, b) = (*a, *b);
                    let head = Resolved::Alloc {
                        id: *id,
                        projection: vec![RefProj::Index(a)],
                    };
                    match self.read_resolved(frames, &head) {
                        Ok(v) => {
                            // The nested reading is only
                            // available while the trailing index
                            // ranges inside the stored value: a
                            // byte-granular sequence running past
                            // the value's own length is the flat
                            // continuation (a raw table's resize
                            // copying a structured bucket value
                            // byte by byte reads past the bucket's
                            // field count immediately).
                            let inner_len = match &v {
                                Value::Tuple(items) => Some(items.len()),
                                Value::Bytes(bytes) => Some(bytes.len()),
                                Value::Str(text) => Some(text.len()),
                                Value::Enum { fields, .. }
                                    if fields.len() == 1 =>
                                {
                                    match &fields[0] {
                                        Value::Tuple(items) => Some(items.len()),
                                        Value::Bytes(bytes) => Some(bytes.len()),
                                        _ => None,
                                    }
                                }
                                _ => None,
                            };
                            let nested = match &v {
                                Value::Tuple(_)
                                | Value::Bytes(_)
                                | Value::Str(_)
                                | Value::FatRef { .. } => {
                                    inner_len.is_none_or(|l| b + index < l)
                                }
                                Value::Enum { fields, .. } => {
                                    fields.len() == 1
                                        && matches!(
                                            fields[0],
                                            Value::Tuple(_) | Value::Bytes(_)
                                        )
                                        && inner_len.is_none_or(|l| b + index < l)
                                }
                                _ => false,
                            };
                            if !nested {
                                return Ok(Some(Resolved::Alloc {
                                    id: *id,
                                    projection: vec![RefProj::Index(a + b + index)],
                                }));
                            }
                        }
                        Err(InterpError::UninitializedRead { .. }) => {
                            return Ok(Some(Resolved::Alloc {
                                id: *id,
                                projection: vec![RefProj::Index(a + b + index)],
                            }));
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        let projection = match target {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. }
            | Resolved::Value(_, projection) => projection,
        };
        // The element step the access composes with: the
        // projection's trailing index step, or the index step
        // its trailing whole-value entry steps enter (field-0
        // and downcast steps -- the shape an array-to-element
        // pointer cast through a transparent wrapper leaves:
        // `[.., Index(k), Field(0)]` denotes element `k` of the
        // outer sequence entered through the wrapper, so the
        // sequence continues at the following elements entered
        // the same way, and the composed access advances the
        // index step and keeps the entry steps). Any other
        // trailing shape (a genuine field of the element, a byte
        // displacement) takes the plain indexed access.
        let mut at = projection.len();
        while at > 0 {
            match projection[at - 1] {
                RefProj::Field(0) | RefProj::Downcast(_) => at -= 1,
                _ => break,
            }
        }
        let Some((&RefProj::Index(start), head)) = projection[..at].split_last()
        else {
            // A bare sequence base (a leading field path with
            // no element step) whose slots are narrower than
            // the access's elements: the plain indexed access
            // would land the access's element `index` in slot
            // `index`, but the element spans `stride` slots and
            // sits at slot `index * stride` (a byte array's
            // storage filled with wider elements through a
            // unit-changing pointer cast chain). Compose the
            // element step here; any other bare shape takes
            // the plain access.
            if at == projection.len() {
                let stride = self.seq_slot_stride(frames, target, projection, elem);
                if stride > 1 {
                    let mut projection = projection.to_vec();
                    projection.push(RefProj::Index(index * stride));
                    return Ok(Some(match target {
                        Resolved::Place {
                            frame_pos, local, ..
                        } => Resolved::Place {
                            frame_pos: *frame_pos,
                            local: *local,
                            projection,
                        },
                        Resolved::Static { name, .. } => Resolved::Static {
                            name: name.clone(),
                            projection,
                        },
                        Resolved::Alloc { id, .. } => Resolved::Alloc {
                            id: *id,
                            projection,
                        },
                        Resolved::Value(base, _) => {
                            Resolved::Value(base.clone(), projection)
                        }
                    }));
                }
            }
            return Ok(None);
        };
        let entry = &projection[at..];
        // The slots one element of this access spans in the
        // sequence the target denotes (see `seq_slot_stride`):
        // the composed step advances by whole elements of the
        // access, priced in the sequence's own slots.
        let stride = self.seq_slot_stride(frames, target, head, elem);
        match self.read_resolved(frames, target) {
            // Whatever value sits at the position — a structured
            // element included — the target denotes that element
            // of the outer sequence: the composed index below is
            // the sequence access. (Reading *into* the element
            // instead hands a bulk copy the element's first
            // component: a vector of tuples built from an array
            // stored the first leaf where the tuple belonged.)
            Ok(_) | Err(InterpError::UninitializedRead { .. }) => {}
            // A position one past the container's end — the
            // append point a tail write targets (a `Vec::append`
            // copy's destination) — reads as an index panic.
            // The position still denotes its slot: the composed
            // access re-validates, so a read there reports the
            // same panic and a write extends the aggregate, as
            // `navigate_mut`'s element arms provide.
            Err(ref e) if is_index_oob_error(e) => {}
            Err(e) => return Err(e),
        }
        let mut projection = head.to_vec();
        projection.push(RefProj::Index(start + index * stride));
        projection.extend_from_slice(entry);
        Ok(Some(match target {
            Resolved::Place {
                frame_pos, local, ..
            } => Resolved::Place {
                frame_pos: *frame_pos,
                local: *local,
                projection,
            },
            Resolved::Static { name, .. } => Resolved::Static {
                name: name.clone(),
                projection,
            },
            Resolved::Alloc { id, .. } => Resolved::Alloc {
                id: *id,
                projection,
            },
            Resolved::Value(base, _) => Resolved::Value(base.clone(), projection),
        }))
    }

    /// The live place a `&RefCell` / `&mut RefCell` argument points
    /// at. Borrowing needs a place: the flag lives inside the cell's
    /// storage and must be written back, which a constant aggregate
    /// (`RefTarget::Value`) cannot do — that case is an explicit
    /// unsupported error, never a silent no-op.
    pub(crate) fn refcell_place(&self, value: &Value, name: &str) -> InterpResult<RefTarget> {
        match value {
            Value::Ref(
                target @ (RefTarget::Place { .. }
                | RefTarget::Static { .. }
                | RefTarget::Alloc { .. }),
            ) => Ok(target.clone()),
            Value::Ref(_) => Err(InterpError::UnsupportedTerminator {
                kind: format!(
                    "runtime support: {name} on a RefCell with no live place \
                     (constant aggregate) cannot track borrows"
                ),
            }),
            other => Err(InterpError::type_mismatch(
                name,
                "reference to RefCell",
                other.kind_name(),
            )),
        }
    }

    /// Read the borrow flag of the `Cell` aggregate `flag_cell`
    /// points at: 0 = unused, -1 = mutably borrowed, n > 0 = n
    /// shared borrows.
    pub(crate) fn refcell_flag(
        &self,
        frames: &[Frame],
        flag_cell: &RefTarget,
    ) -> InterpResult<i128> {
        let resolved = self.deref_value(frames, &Value::Ref(flag_cell.clone()))?;
        let cell = self.read_resolved(frames, &resolved)?;
        match cell {
            Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
                Value::Enum { fields: inner, .. } if inner.len() == 1 => match inner[0] {
                    Value::Int(flag) => Ok(flag),
                    // A flag materialised from a constant whose
                    // signed context was lost arrives as an
                    // unsigned word; the flag values (0, -1,
                    // small counts) reinterpret exactly.
                    Value::Uint(flag) => Ok(flag as i128),
                    ref other => Err(InterpError::type_mismatch(
                        "RefCell borrow flag",
                        "isize flag",
                        other.kind_name(),
                    )),
                },
                other => Err(InterpError::type_mismatch(
                    "RefCell borrow flag",
                    "UnsafeCell aggregate",
                    other.kind_name(),
                )),
            },
            ref other => Err(InterpError::type_mismatch(
                "RefCell borrow flag",
                "Cell aggregate",
                other.kind_name(),
            )),
        }
    }

    /// Write the borrow flag of the `Cell` aggregate `flag_cell`
    /// points at, rebuilding the `Cell { UnsafeCell { flag } }`
    /// aggregate in place.
    pub(crate) fn refcell_set_flag(
        &mut self,
        frames: &mut Vec<Frame>,
        flag_cell: &RefTarget,
        flag: i128,
    ) -> InterpResult<()> {
        let resolved = self.deref_value(frames, &Value::Ref(flag_cell.clone()))?;
        let cell = Value::Enum {
            variant: 0,
            fields: vec![Value::Enum {
                variant: 0,
                fields: vec![Value::Int(flag)],
            }],
        };
        self.write_resolved(frames, &resolved, cell)
    }

    /// Release the borrow held by a `Ref` / `RefMut` guard value:
    /// field 1 is the reference to the cell's borrow-flag `Cell`.
    /// A shared release decrements the count; a mutable release
    /// resets the flag to unused (0).
    pub(crate) fn release_refcell_borrow(
        &mut self,
        frames: &mut Vec<Frame>,
        guard: &Value,
        mutable: bool,
    ) -> InterpResult<()> {
        let fields = match refcell_guard_fields(guard) {
            Some(fields) => fields,
            None => {
                return Err(InterpError::type_mismatch(
                    "RefCell guard drop",
                    "Ref/RefMut guard aggregate",
                    guard.kind_name(),
                ))
            }
        };
        let flag_target = match refcell_guard_ref(&fields[1]) {
            Some(target) => target.clone(),
            None => {
                return Err(InterpError::type_mismatch(
                    "RefCell guard drop",
                    "reference to the borrow-flag Cell",
                    fields[1].kind_name(),
                ))
            }
        };
        let flag = self.refcell_flag(frames, &flag_target)?;
        let next = if mutable { 0 } else { flag - 1 };
        self.refcell_set_flag(frames, &flag_target, next)
    }

    // ----- foreign-view accesses ------------------------------
    //
    // A place reached through a reference whose pointee type
    // differs from the type the storage was born with — the
    // shape a `mem::transmute` of `&mut Enum` into `&mut Repr`
    // produces, where the repr is a struct or union laid out
    // over the same bytes — is a foreign access: the
    // projection was converted against the accessing type, but
    // the stored value is a value of the birth type, and the
    // structural projectors would compose the accessing type's
    // field tree inside the birth value's field slots, leaving
    // a shape no birth-typed read can interpret (the tag value
    // sitting in the payload slot, the payload union nested
    // beside it, the discriminant never written). The byte
    // view owns such accesses instead: the projection's byte
    // span is priced through the accessing type's recorded
    // layouts, and the bytes move through the birth value's
    // storage with the same primitives the displaced-pointer
    // view uses — a write of the tag field's bytes re-forms
    // the enum at the named variant with the surviving storage
    // spliced into it, exactly as the storage's bytes dictate,
    // and a read constructs the accessing type's value from
    // the span's bytes. An access is foreign only when the
    // accessing and birth root types are different ADT
    // instances and the reference's pointee extent agrees
    // with the span its stored prefix denotes in the birth
    // storage; anything the view cannot price or move
    // declines to the ordinary paths, which keep their
    // current behaviour there. The same view serves a
    // sequence root on both sides — a pointer cast that
    // regroups an array storage's elements under a wider or
    // narrower element shape (the `[[u8; 3]; 2]` storage read
    // as `[[u8; 2]]` that RFC 0401 blesses): the birth side
    // is the array the storage was born as, the accessing
    // side the array or slice the cast names, and the span
    // prices through the accessing side's element size.

    /// The declared (birth) root type of a resolved target and
    /// its root-relative projection, for the foreign view's
    /// detection (see above). `None` for a value target (a
    /// constant's storage has no birth type to translate
    /// against), for an element block (whose storage the
    /// element model owns), and for a target whose birth type
    /// is not recorded.
    fn target_birth<'t>(
        &self,
        frames: &[Frame],
        target: &'t Resolved,
    ) -> Option<(Ty, &'t [RefProj])> {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let frame = frames.get(*frame_pos)?;
                let body = self.program().get(&frame.func)?;
                let ty = body.local_tys.get(*local)?.clone();
                Some((ty, projection))
            }
            Resolved::Static { name, projection } => {
                let ty = self.program().static_tys.get(name)?.clone();
                Some((ty, projection))
            }
            Resolved::Alloc { id, projection } => {
                if self.is_elem_block(*id) {
                    return None;
                }
                let ty = self.alloc_ty(*id)?;
                Some((ty, projection))
            }
            Resolved::Value(..) => None,
        }
    }

    /// The (function, place description) pair an error about
    /// `target` reports, mirroring `read_resolved_typed`.
    fn target_desc(&self, frames: &[Frame], target: &Resolved) -> (String, String) {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let func = frames
                    .get(*frame_pos)
                    .map(|f| f.func.clone())
                    .unwrap_or_default();
                (func, describe_resolved(*local, projection))
            }
            Resolved::Static { name, .. } => {
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| "<static>".to_string());
                (func, format!("static {name}"))
            }
            Resolved::Alloc { id, .. } => {
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| "<allocation>".to_string());
                (func, format!("allocation #{id}"))
            }
            Resolved::Value(..) => (
                "<constant>".to_string(),
                "constant aggregate".to_string(),
            ),
        }
    }

    /// The birth root type of a foreign access (see the
    /// section comment): the accessing root type and the
    /// target's birth root type denominate the storage
    /// differently — different ADT instances, or sequence
    /// roots (an array birth read as an array or slice of a
    /// differently-shaped element) — and the target's
    /// projection contains no step whose span the layouts do
    /// not determine (a byte displacement is the displaced
    /// view's own case, a sub-slice has no single span, the
    /// vector length slot is not storage). The extent
    /// agreement the access needs — an ADT view denoting the
    /// whole storage, a sequence view's bytes covered by the
    /// birth storage — is [`Self::foreign_span`]'s check, not
    /// this one's.
    fn foreign_birth_ty(
        &self,
        frames: &[Frame],
        target: &Resolved,
        access_root: &Ty,
    ) -> Option<Ty> {
        let (birth, projection) = self.target_birth(frames, target)?;
        if projection.iter().any(|s| {
            matches!(
                s,
                RefProj::ByteOffset(_)
                    | RefProj::Subslice { .. }
                    | RefProj::VecLen
            )
        }) {
            return None;
        }
        // The same type on both sides is the ordinary path's
        // own access, not a view over the storage.
        if access_root == &birth {
            return None;
        }
        match (access_root, &birth) {
            (Ty::Adt(access_name), Ty::Adt(birth_name)) => {
                if !self.program().adt_layouts.contains_key(access_name)
                    || !self.program().adt_layouts.contains_key(birth_name)
                {
                    return None;
                }
            }
            // A sequence root on both sides (see the section
            // comment): the birth side is the sized array the
            // storage was born as — a slice is never storage's
            // own type — and its bytes must be extractable.
            (Ty::Array(..) | Ty::Slice(_), Ty::Array(..)) => {
                self.byte_layout_of_ty(&birth)?;
            }
            _ => return None,
        }
        Some(birth)
    }

    /// The byte span (offset, size) a root-relative projection
    /// denotes within the storage of a value of type `root_ty`,
    /// priced through the recorded field layouts — the foreign
    /// view's addressing (see the section comment). A field
    /// step prices through the instance's recorded layout row
    /// for the variant the walk is inside; a union field step
    /// adds nothing (every field starts at the union's base)
    /// and takes the field's recorded span as the size. `None`
    /// when a step's offset the recorded layouts do not
    /// determine (a tuple step, an unrecorded field type on
    /// the way in, an index into a non-array).
    fn access_span(
        &self,
        root_ty: &Ty,
        projection: &[RefProj],
        root_extent: usize,
    ) -> Option<(usize, usize)> {
        let mut ty: Option<Ty> = Some(root_ty.clone());
        let mut off: usize = 0;
        let mut size: usize = root_extent;
        let mut variant: usize = 0;
        for (n, step) in projection.iter().enumerate() {
            let last = n + 1 == projection.len();
            match step {
                RefProj::Field(i) | RefProj::UnionField { idx: i, .. } => {
                    let union = matches!(step, RefProj::UnionField { .. });
                    let cur = ty.as_ref()?;
                    let Ty::Adt(name) = cur else {
                        return None;
                    };
                    let (fo, fs) = if union {
                        // A union's fields all start at its
                        // base; a field's span is its own size.
                        // The recorded layout row says so when
                        // the instance has one; a union instance
                        // that appears only as another ADT's
                        // field may have none, and then the
                        // field's declared type sizes it.
                        match self
                            .program()
                            .adt_field_layouts
                            .get(name)
                            .and_then(|rows| rows.first())
                            .and_then(|row| row.get(*i))
                        {
                            Some(&(fo, fs)) => (fo, fs),
                            None => {
                                let fty = self.instance_field_ty(
                                    name, cur, 0, *i,
                                )?;
                                (0, self.ty_size(&fty)?)
                            }
                        }
                    } else {
                        let rows =
                            self.program().adt_field_layouts.get(name)?;
                        let row = rows.get(variant)?;
                        *row.get(*i)?
                    };
                    off = off.checked_add(usize::try_from(fo).ok()?)?;
                    size = usize::try_from(fs).ok()?;
                    if !last {
                        ty = Some(self.instance_field_ty(
                            name,
                            cur,
                            if union { 0 } else { variant },
                            *i,
                        )?);
                        variant = 0;
                    }
                }
                RefProj::Downcast(v) => {
                    variant = *v as usize;
                }
                RefProj::Index(k) => {
                    let cur = ty.as_ref()?;
                    let elem = match cur {
                        Ty::Array(elem, _) | Ty::Slice(elem) => elem,
                        _ => return None,
                    };
                    let es = usize::try_from(self.ty_size(elem)?).ok()?;
                    off = off.checked_add(k.checked_mul(es)?)?;
                    size = es;
                    if !last {
                        ty = Some((**elem).clone());
                    }
                }
                RefProj::View(view) => {
                    // The recorded denomination of the
                    // dereference this access performs (see
                    // `RefProj::View`): the steps after it
                    // price through the view's type, which is
                    // the accessing type this walk prices by.
                    // A trailing record denotes the whole
                    // pointee, whose span is its own size.
                    ty = Some(view.clone());
                    variant = 0;
                    if last {
                        size = usize::try_from(self.ty_size(view)?).ok()?;
                    }
                }
                _ => return None,
            }
        }
        Some((off, size))
    }

    /// The byte span a foreign access denotes, split at the
    /// access's own dereference: the target's projection is
    /// the reference's stored prefix (formed when the
    /// reference was created) followed by the steps the
    /// access's place adds past its dereference (formed
    /// against the accessing type). The tail prices over the
    /// accessing type. A non-empty prefix declines: the steps
    /// a stored reference carries were formed against the
    /// type of the place that created it — the birth type for
    /// an ordinary borrow, an accessing view for a borrow
    /// taken through one — and nothing at the access records
    /// which, so a span priced for it could name the wrong
    /// bytes; such accesses (a reference extracted from a
    /// transmuted view and used elsewhere) keep the ordinary
    /// structural path. For a reference to the storage root
    /// itself the prefix is empty and the tail is the whole
    /// projection.
    fn foreign_span(
        &self,
        frames: &[Frame],
        body: &Body,
        target: &Resolved,
        access_root: &Ty,
        birth: &Ty,
        place: &Place,
    ) -> Option<(usize, usize)> {
        let (_, projection) = self.target_birth(frames, target)?;
        let tail_len = place
            .projection
            .iter()
            .rposition(|e| matches!(e, ProjectionElem::Deref))
            .map(|p| place.projection.len() - p - 1)?;
        // A denomination step the target recorded at its
        // dereference (see `RefProj::View`) is not a step of
        // the tail: it names the type the tail prices
        // through, which is the accessing type this span
        // prices by (see the `View` arm of `access_span`).
        let step_count = projection
            .iter()
            .filter(|s| !matches!(s, RefProj::View(_)))
            .count();
        if step_count != tail_len {
            return None;
        }
        let birth_size = usize::try_from(self.ty_size(birth)?).ok()?;
        // The extent of the storage the accessing root ranges
        // over, and the agreement it keeps with the birth
        // storage: an ADT view denotes the whole storage, so
        // the two sizes are equal; a sequence view denotes its
        // elements' bytes — an array's own size, a slice's fat
        // length times its element size — which the birth
        // storage must cover (a cast may regroup a prefix of
        // the storage it points into).
        let access_extent = match access_root {
            Ty::Adt(name) => {
                let size =
                    usize::try_from(self.program().adt_layouts.get(name)?.0)
                        .ok()?;
                if size != birth_size {
                    return None;
                }
                size
            }
            Ty::Array(..) => {
                let size = usize::try_from(self.ty_size(access_root)?).ok()?;
                if size > birth_size {
                    return None;
                }
                size
            }
            Ty::Slice(elem) => {
                let fat_len = self.fat_prefix_deref_len(frames, body, place)?;
                let elem_size = usize::try_from(self.ty_size(elem)?).ok()?;
                let size = usize::try_from(fat_len)
                    .ok()?
                    .checked_mul(elem_size)?;
                if size > birth_size {
                    return None;
                }
                size
            }
            _ => return None,
        };
        let (off, size) = self.access_span(access_root, projection, access_extent)?;
        if off.checked_add(size)? > access_extent {
            return None;
        }
        Some((off, size))
    }

    /// One byte of a value's storage, for the foreign view
    /// (see [`Self::storage_byte`]): a definite byte, an
    /// indeterminate (padding) byte, or a byte of storage
    /// never written. `None` when the layout cannot judge the
    /// position at all.
    fn foreign_byte(
        &self,
        value: &Value,
        layout: &ByteLayout,
        pos: usize,
    ) -> Option<ForeignByte> {
        match self.storage_byte(value, layout, pos)? {
            Ok(StorageByte::Byte(b)) => Some(ForeignByte::Byte(b)),
            Ok(StorageByte::Padding) => Some(ForeignByte::Padding),
            Err(ByteVerdict::Uninit) => Some(ForeignByte::Uninit),
            Err(_) => None,
        }
    }

    /// The byte image of a complete value of type `ty` being
    /// written through a foreign view: per position, the
    /// definite byte the value's storage holds there, or
    /// `None` for a position the value does not determine
    /// (its padding, which a write leaves as it lies). The
    /// whole image is `None` (the access declines) when any
    /// position's byte the value leaves uninitialised or the
    /// layout cannot judge — a partial image spliced into
    /// foreign storage would invent bytes. A niche-encoded
    /// enum value in a niche variant has no field bytes to
    /// extract: its image is the niche scalar's own pattern
    /// at the recorded offset, padding elsewhere; the same
    /// holds for the scalar form such a value flattens to.
    fn written_value_bytes(
        &self,
        value: &Value,
        ty: &Ty,
        size: usize,
    ) -> Option<Vec<Option<u8>>> {
        if let Ty::Adt(name) = ty {
            if let Some(niche) = self.program().adt_niches.get(name) {
                let scalar = match value {
                    Value::Enum { variant, .. }
                        if *variant != niche.untagged_variant
                            && (niche.niche_variants.0..=niche.niche_variants.1)
                                .contains(variant) =>
                    {
                        Some(
                            niche.niche_start
                                + u128::from(variant - niche.niche_variants.0),
                        )
                    }
                    Value::Uint(bits) => Some(*bits),
                    Value::Int(bits) => Some(*bits as u128),
                    _ => None,
                };
                if let Some(scalar) = scalar {
                    let width = usize::try_from(niche.tag_bits / 8).ok()?;
                    let off = usize::try_from(niche.niche_offset).ok()?;
                    if off + width > size {
                        return None;
                    }
                    let mut bytes = vec![None; size];
                    for i in 0..width {
                        bytes[off + i] = Some((scalar >> (8 * i)) as u8);
                    }
                    return Some(bytes);
                }
            }
        }
        let layout = self.byte_layout_of_ty(ty)?;
        let mut bytes = Vec::with_capacity(size);
        for pos in 0..size {
            match self.foreign_byte(value, &layout, pos)? {
                ForeignByte::Byte(b) => bytes.push(Some(b)),
                ForeignByte::Padding => bytes.push(None),
                ForeignByte::Uninit => {
                    return None;
                }
            }
        }
        Some(bytes)
    }

    /// Construct the value of type `ty` whose storage bytes
    /// are `bytes` (the foreign view's read): a definite byte
    /// per position, `None` where the source storage held
    /// padding there. `None` (the access declines) when the
    /// type's value cannot be built from bytes here — a tuple
    /// (whose field offsets are not recorded), a pointer, or
    /// a byte the construction cannot judge. An ADT value is
    /// built by splicing the bytes into an uninitialised
    /// skeleton of the type, so an enum's tag bytes re-form
    /// the skeleton at the variant they name, carrying the
    /// already-spliced payload bytes across; a niche-encoded
    /// enum whose niche window holds a niche variant's
    /// pattern is that variant's value directly.
    fn construct_from_bytes(&self, ty: &Ty, bytes: &[Option<u8>]) -> Option<Value> {
        match ty {
            Ty::Bool | Ty::Int(_) | Ty::Uint(_) | Ty::Float(_) => {
                let mut bits: u128 = 0;
                for (i, b) in bytes.iter().enumerate() {
                    bits |= u128::from((*b)?) << (8 * i);
                }
                let shape = match ty {
                    Ty::Bool => ScalarShape::Bool,
                    Ty::Int(t) => ScalarShape::Int {
                        bits: t.bits(),
                        signed: true,
                    },
                    Ty::Uint(t) => ScalarShape::Int {
                        bits: t.bits(),
                        signed: false,
                    },
                    Ty::Float(t) => ScalarShape::Float { bits: t.bits() },
                    _ => unreachable!(),
                };
                self.construct_scalar(bits, &shape).ok()
            }
            Ty::Array(elem, n) => {
                let count = usize::try_from(*n).ok()?;
                let es = usize::try_from(self.ty_size(elem)?).ok()?;
                let mut items = Vec::with_capacity(count);
                for i in 0..count {
                    let slice = bytes.get(i * es..(i + 1) * es)?;
                    items.push(self.construct_from_bytes(elem, slice)?);
                }
                Some(Value::Tuple(items))
            }
            Ty::Adt(name) => {
                if let Some(niche) = self.program().adt_niches.get(name) {
                    let width = usize::try_from(niche.tag_bits / 8).ok()?;
                    let off = usize::try_from(niche.niche_offset).ok()?;
                    let mut bits: u128 = 0;
                    let mut judged = true;
                    for i in 0..width {
                        match bytes.get(off + i) {
                            Some(Some(b)) => bits |= u128::from(*b) << (8 * i),
                            _ => {
                                judged = false;
                                break;
                            }
                        }
                    }
                    if judged
                        && bits >= niche.niche_start
                        && bits - niche.niche_start
                            <= u128::from(
                                niche.niche_variants.1 - niche.niche_variants.0,
                            )
                    {
                        let variant = niche.niche_variants.0
                            + u32::try_from(bits - niche.niche_start).ok()?;
                        let fields = self
                            .program()
                            .adt_field_layouts
                            .get(name)
                            .and_then(|rows| rows.get(variant as usize))
                            .map(|row| vec![Value::Uninit; row.len()])
                            .unwrap_or_default();
                        return Some(Value::Enum { variant, fields });
                    }
                }
                let rows = self.program().adt_field_layouts.get(name)?;
                let row0 = rows.first()?;
                let mut skeleton = Value::Enum {
                    variant: 0,
                    fields: vec![Value::Uninit; row0.len()],
                };
                let layout = ByteLayout::Adt(name.clone());
                for (pos, b) in bytes.iter().enumerate() {
                    let Some(b) = b else {
                        continue;
                    };
                    match self.splice_storage_byte(&mut skeleton, &layout, pos, *b)? {
                        Ok(()) => {}
                        Err(ByteVerdict::Padding(_)) => {}
                        Err(_) => return None,
                    }
                }
                Some(skeleton)
            }
            _ => None,
        }
    }

    /// The current root value of a resolved target, cloned;
    /// `None` when the target holds no storage the foreign
    /// view may move (a value target, a missing local, an
    /// unseeded static, a freed allocation) — the access
    /// declines and the ordinary path judges the target.
    fn target_root_value(&self, frames: &[Frame], target: &Resolved) -> Option<Value> {
        match target {
            Resolved::Place {
                frame_pos, local, ..
            } => frames.get(*frame_pos)?.locals.get(*local).cloned(),
            Resolved::Static { name, .. } => self.static_base(name).ok(),
            Resolved::Alloc { id, .. } => self.alloc_base(*id).ok(),
            Resolved::Value(..) => None,
        }
    }

    /// Store `value` as the root value of a resolved target;
    /// `None` under the same conditions as
    /// [`Self::target_root_value`].
    fn store_target_root(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: Value,
    ) -> Option<InterpResult<()>> {
        match target {
            Resolved::Place {
                frame_pos, local, ..
            } => {
                let slot = frames.get_mut(*frame_pos)?.locals.get_mut(*local)?;
                *slot = value;
                Some(Ok(()))
            }
            Resolved::Static { name, .. } => {
                let mut store = self.statics.borrow_mut();
                let slot = store.get_mut(name)?;
                *slot = value;
                Some(Ok(()))
            }
            Resolved::Alloc { id, .. } => {
                let mut store = self.allocs.borrow_mut();
                let slot = store.get_mut(id)?;
                *slot = value;
                Some(Ok(()))
            }
            Resolved::Value(..) => None,
        }
    }

    /// The accessing root type of a place, for the foreign
    /// view: the pointee type the place's last dereference
    /// denotes, walked from the place's local with the same
    /// steps [`Self::resolved_root_ty`] takes, plus the union
    /// field step that walk cannot price (a union field's
    /// declared type is its field type at variant 0). `None`
    /// when the walk meets a step it cannot type.
    fn foreign_access_root(&self, body: &Body, place: &Place) -> Option<Ty> {
        self.foreign_place_walk(body, place).map(|(root, _)| root)
    }

    /// The declared type of the value a place denotes, for
    /// the foreign view: the full walk of
    /// [`Self::foreign_access_root`], returning the type at
    /// the end of the projection (the value the access reads
    /// or writes) alongside the accessing root type.
    /// (`place_result_ty` cannot serve here: it prices no
    /// field step into an ADT.)
    fn foreign_place_ty(&self, body: &Body, place: &Place) -> Option<Ty> {
        self.foreign_place_walk(body, place).map(|(_, ty)| ty)
    }

    /// The walk behind [`Self::foreign_access_root`] and
    /// [`Self::foreign_place_ty`]: (accessing root type,
    /// result type) of a place.
    /// The type of an ADT field for the foreign walks: the
    /// declared-type table's skeleton, upgraded to the field's
    /// full instance when the skeleton names only the generic
    /// definition (its bare name has no recorded layout) and
    /// the parent's recorded field tag names an instance that
    /// does. The byte machinery is keyed by instance, so a
    /// place whose type stays a bare definition name cannot
    /// be extracted or constructed.
    fn instance_field_ty(
        &self,
        parent_name: &str,
        parent: &Ty,
        variant: usize,
        field: usize,
    ) -> Option<Ty> {
        let ty = self.adt_field_ty(parent, variant, field)?;
        if let Ty::Adt(name) = &ty {
            if !self.program().adt_layouts.contains_key(name) {
                if let Some(tag) = self.adt_field_tag(parent_name, variant, field) {
                    let tag = tag.trim();
                    if self.program().adt_layouts.contains_key(tag) {
                        return Some(Ty::Adt(tag.to_string()));
                    }
                }
            }
        }
        Some(ty)
    }

    fn foreign_place_walk(
        &self,
        body: &Body,
        place: &Place,
    ) -> Option<(Ty, Ty)> {
        let mut ty = body.local_ty(place.local)?.clone();
        let mut root = ty.clone();
        let mut variant = 0usize;
        for elem in &place.projection {
            match (elem, &ty) {
                (ProjectionElem::Deref, Ty::Ref(inner)) => {
                    ty = (**inner).clone();
                    root = ty.clone();
                    variant = 0;
                }
                (ProjectionElem::Field(i), Ty::Tuple(ts)) => {
                    ty = ts.get(*i)?.clone();
                }
                (ProjectionElem::Field(i), Ty::Adt(name)) => {
                    let name = name.clone();
                    ty = self.instance_field_ty(&name, &ty, variant, *i)?;
                }
                (ProjectionElem::UnionField { idx, .. }, Ty::Adt(name)) => {
                    let name = name.clone();
                    ty = self.instance_field_ty(&name, &ty, 0, *idx)?;
                }
                (ProjectionElem::Index(_), Ty::Array(elem, _))
                | (ProjectionElem::Index(_), Ty::Slice(elem))
                | (ProjectionElem::ConstantIndex { .. }, Ty::Array(elem, _))
                | (ProjectionElem::ConstantIndex { .. }, Ty::Slice(elem)) => {
                    ty = (**elem).clone();
                }
                (ProjectionElem::Downcast(v), _) => {
                    variant = *v as usize;
                }
                (ProjectionElem::Subslice { .. }, Ty::Array(elem, _)) => {
                    ty = Ty::Array(elem.clone(), 0);
                }
                (ProjectionElem::Subslice { .. }, Ty::Slice(elem)) => {
                    ty = Ty::Slice(elem.clone());
                }
                _ => return None,
            }
        }
        Some((root, ty))
    }

    /// Whether a birth denomination and a pointee of these
    /// shapes can denominate the same storage at all: the
    /// sequence and ADT pairs are judged by their own arms,
    /// and of the rest only same-kind pairs are comparable.
    /// A mixed pair (a container ADT against the sequence it
    /// holds, an ADT against a scalar payload, a sequence
    /// against an ADT view) is a formation the access-side
    /// machinery already prices -- a `Vec` dereference to its
    /// slice above all -- not a regrouping of the storage's
    /// elements, so it records no denomination.
    fn comparable_shape(a: &Ty, b: &Ty) -> bool {
        use std::mem::discriminant;
        discriminant(a) == discriminant(b)
    }

    /// Record the denomination of a dereference on the
    /// target it resolved to (see
    /// [`RefProj::View`]): the steps a walk appends after the
    /// dereference price through the pointee type, while the
    /// target's consumers price its steps through the birth
    /// type of the storage the target's root names -- the two
    /// agree for an ordinary borrow and differ exactly when
    /// the pointer's formation regrouped the storage (a raw
    /// pointer cast, a transmute). The record is appended only
    /// where the denominations provably differ: the birth
    /// type at the resolved position is stepped from the
    /// target's birth root type, and a sequence pointee
    /// denominates its index steps by its element, any other
    /// comparable pointee by itself. Two ADT instances over
    /// one storage (a container against its inner
    /// representation, a sized value against its unsized
    /// wrapper) record nothing: their steps are field steps,
    /// which the tables price from the birth type, and the
    /// access-side foreign view prices the formation itself;
    /// a stored record would re-denominate the root storage
    /// under the pointee's layout, which no consumer prices.
    /// A position the type tables cannot step to, a storage
    /// with no byte layout, and a value target (a
    /// constant's storage has no birth type) record nothing,
    /// keeping today's targets.
    pub(crate) fn denominate_fat_deref(
        &self,
        frames: &[Frame],
        resolved: Resolved,
        pointee: &Ty,
    ) -> Resolved {
        let (birth, projection) = match self.target_birth(frames, &resolved) {
            Some(pair) => pair,
            None => return resolved,
        };
        if self.byte_layout_of_ty(&birth).is_none() {
            return resolved;
        }
        let pos_ty = step_ty_across(
            Some(self.program()),
            Some(birth.clone()),
            projection,
        );
        // A stepped empty tuple is the walk's placeholder for
        // a position whose denomination the tables do not
        // record (a union field above all): like a position
        // the tables cannot step to, it records nothing --
        // the placeholder denominates no storage, so its
        // difference from the pointee is not a regrouping.
        if matches!(pos_ty, Some(Ty::Tuple(ref items)) if items.is_empty()) {
            return resolved;
        }
        // The ordinary comparison: sequences by their
        // elements, comparable shapes by themselves, ADT pairs
        // and mixed shapes not at all (see the arms).
        let ordinary = match (pos_ty.as_ref(), pointee) {
            (
                Some(Ty::Array(e1, _) | Ty::Slice(e1)),
                Ty::Array(e2, _) | Ty::Slice(e2),
            ) => e1 != e2,
            (Some(Ty::Adt(_)), Ty::Adt(_)) => false,
            (Some(other), _) => {
                Self::comparable_shape(other, pointee)
                    && other != pointee
            }
            (None, _) => false,
        };
        // One position the ordinary comparison misjudges: a
        // fat dereference denotes its pointee through the
        // pointee's first element, so the resolved target can
        // end in an index step landing on an ELEMENT of the
        // sequence the dereference denominates -- the stepped
        // type is then the element type, which the ordinary
        // comparison cannot set against the sequence. A
        // sequence pointee denominates by its element, so the
        // comparison there is element against element. An
        // index step landing on a nested sequence (a row of a
        // matrix) is not this position: its stepped type is
        // the sequence itself, and the ordinary comparison
        // already judges it.
        let element_position = matches!(
            projection.last(),
            Some(RefProj::Index(_))
        ) && !matches!(
            pos_ty,
            Some(Ty::Array(..) | Ty::Slice(..))
        );
        let differ = if element_position {
            match pointee {
                Ty::Array(e2, _) | Ty::Slice(e2) => {
                    // A position the type tables cannot step
                    // to records nothing, as above.
                    match pos_ty.as_ref() {
                        Some(t) => t != e2.as_ref(),
                        None => false,
                    }
                }
                _ => ordinary,
            }
        } else {
            ordinary
        };
        if !differ {
            return resolved;
        }
        let step = RefProj::View(pointee.clone());
        match resolved {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(step);
                Resolved::Place {
                    frame_pos,
                    local,
                    projection,
                }
            }
            Resolved::Static {
                name,
                mut projection,
            } => {
                projection.push(step);
                Resolved::Static { name, projection }
            }
            Resolved::Alloc {
                id,
                mut projection,
            } => {
                projection.push(step);
                Resolved::Alloc { id, projection }
            }
            Resolved::Value(..) => resolved,
        }
    }

    /// The byte span a target carrying a recorded denomination
    /// (see [`RefProj::View`]) denotes within its root
    /// storage: the (offset, span size, denoted type) the
    /// target's projection prices to, segment by segment --
    /// the steps before the first denomination step price
    /// through the root's birth type, each denomination step
    /// replaces the pricing type for the steps that follow
    /// it. `None` when the target carries no denomination
    /// step, is a value target or an element block (whose
    /// storage has no birth type to price against), a step's
    /// span the recorded layouts do not determine, or the
    /// span runs past the storage. The denoted type is the
    /// type of the value the whole projection denotes, for
    /// the caller's construction (read) and byte extraction
    /// (write).
    fn view_span(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> Option<(usize, usize, Ty)> {
        let (birth, projection) = self.target_birth(frames, target)?;
        if !projection
            .iter()
            .any(|s| matches!(s, RefProj::View(_)))
        {
            return None;
        }
        let birth_size = usize::try_from(self.ty_size(&birth)?).ok()?;
        let mut ty: Option<Ty> = Some(birth);
        let mut off: usize = 0;
        let mut size: Option<usize> = Some(birth_size);
        let mut variant: usize = 0;
        for (n, step) in projection.iter().enumerate() {
            let last = n + 1 == projection.len();
            match step {
                RefProj::View(view) => {
                    // The recorded denomination: the steps
                    // after it price through the view's type.
                    // A trailing record denotes the whole
                    // pointee, whose span is its own size (an
                    // unsized pointee's span is not recorded
                    // anywhere, so the walk declines).
                    ty = Some(view.clone());
                    variant = 0;
                    size = if last {
                        Some(usize::try_from(self.ty_size(view)?).ok()?)
                    } else {
                        None
                    };
                }
                RefProj::Field(i) | RefProj::UnionField { idx: i, .. } => {
                    let union = matches!(step, RefProj::UnionField { .. });
                    let cur = ty.as_ref()?;
                    let Ty::Adt(name) = cur else { return None };
                    let (fo, fs) = if union {
                        let field_ty = self.instance_field_ty(
                            name,
                            cur,
                            0,
                            *i,
                        )?;
                        let fs = usize::try_from(
                            self.ty_size(&field_ty)?,
                        )
                        .ok()?;
                        (0u64, fs as u64)
                    } else {
                        let layouts = self
                            .program()
                            .adt_field_layouts
                            .get(name)?;
                        let rows = layouts.get(variant)?;
                        let row = rows.get(*i)?;
                        (row.0, row.1)
                    };
                    let fty = self.instance_field_ty(
                        name,
                        cur,
                        if union { 0 } else { variant },
                        *i,
                    )?;
                    off = off.checked_add(usize::try_from(fo).ok()?)?;
                    size = Some(usize::try_from(fs).ok()?);
                    ty = Some(fty);
                    variant = 0;
                }
                RefProj::Downcast(v) => {
                    variant = *v as usize;
                }
                RefProj::Index(k) => {
                    let cur = ty.as_ref()?;
                    let elem = match cur {
                        Ty::Array(elem, _) | Ty::Slice(elem) => elem,
                        _ => return None,
                    };
                    let es = usize::try_from(self.ty_size(elem)?).ok()?;
                    off = off.checked_add(k.checked_mul(es)?)?;
                    size = Some(es);
                    ty = Some((**elem).clone());
                    variant = 0;
                }
                RefProj::Subslice { offset, len } => {
                    // A re-slice of a denominated view: the
                    // step's denomination is the type the
                    // preceding steps (or the recorded view)
                    // denote, so the offset prices through
                    // its element, as the access-side walk
                    // prices its own.
                    let cur = ty.as_ref()?;
                    let elem = match cur {
                        Ty::Array(elem, _) | Ty::Slice(elem) => elem,
                        _ => return None,
                    };
                    let es = usize::try_from(self.ty_size(elem)?).ok()?;
                    off = off.checked_add(offset.checked_mul(es)?)?;
                    size = Some(len.checked_mul(es)?);
                    ty = Some(match cur {
                        Ty::Array(..) => Ty::Array(
                            elem.clone(),
                            u64::try_from(*len).ok()?,
                        ),
                        _ => Ty::Slice(elem.clone()),
                    });
                    variant = 0;
                }
                RefProj::ByteOffset(disp) => {
                    // A byte displacement on top of the
                    // denominated position: the pointer
                    // arithmetic that formed the target (an
                    // iterator step's advance over the view's
                    // elements above all) moved the pointer
                    // this many bytes from the position the
                    // preceding steps denote. The displaced
                    // pointer denotes a value of the same type
                    // at the displaced position -- a pointer
                    // to the view's element advanced by one
                    // element denotes the next element -- so
                    // the walk moves the offset and keeps the
                    // denomination and the span the preceding
                    // steps priced. A span the walk has not
                    // priced (a displaced whole slice) keeps
                    // the final size check declining, and a
                    // displacement that runs the position
                    // before the storage or the span past it
                    // declines there as well.
                    if *disp >= 0 {
                        off = off.checked_add(usize::try_from(*disp).ok()?)?;
                    } else {
                        off = off.checked_sub(usize::try_from(disp.checked_neg()?).ok()?)?;
                    }
                }
                _ => return None,
            }
        }
        let size = size?;
        if size == 0 || off.checked_add(size)? > birth_size {
            return None;
        }
        Some((off, size, ty?))
    }

    /// Read a target whose projection carries a recorded
    /// denomination (see [`RefProj::View`]): the stored
    /// reference's own record of the view its formation
    /// took, for the consumers that meet the target without
    /// an accessing place (a format argument's target, a
    /// dereference of a stored reference). The span's bytes
    /// come out of the birth value's storage and the result
    /// is constructed as a value of the denoted type -- the
    /// stored-target form of [`Self::foreign_read`]. `None`
    /// when the view does not judge the target; a judged
    /// uninitialised byte in the span is the read's error,
    /// as on the foreign path.
    pub(crate) fn view_read(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> Option<InterpResult<Value>> {
        let (birth, _) = self.target_birth(frames, target)?;
        let (off, size, leaf) = self.view_span(frames, target)?;
        // The span a step sequence prices can be wider than
        // the denoted value (a field's layout span covers its
        // padding); the value is its own size's prefix, as on
        // the foreign path. A slice's span is its elements'.
        let value_size = match &leaf {
            Ty::Slice(elem) => {
                let elem_size =
                    usize::try_from(self.ty_size(elem)?).ok()?;
                if elem_size == 0 || size % elem_size != 0 {
                    return None;
                }
                size
            }
            _ => usize::try_from(self.ty_size(&leaf)?).ok()?,
        };
        if value_size > size {
            return None;
        }
        let size = value_size;
        let root = self.target_root_value(frames, target)?;
        let layout = self.byte_layout_of_ty(&birth)?;
        let mut bytes = Vec::with_capacity(size);
        for i in 0..size {
            match self.foreign_byte(&root, &layout, off + i)? {
                ForeignByte::Byte(b) => bytes.push(Some(b)),
                ForeignByte::Padding => bytes.push(None),
                ForeignByte::Uninit => {
                    let (func, desc) = self.target_desc(frames, target);
                    return Some(Err(InterpError::UninitializedRead {
                        func,
                        place: desc,
                    }));
                }
            }
        }
        let built = match &leaf {
            Ty::Slice(elem) => {
                let elem_size =
                    usize::try_from(self.ty_size(elem)?).ok()?;
                let mut items = Vec::with_capacity(size / elem_size);
                for chunk in bytes.chunks(elem_size) {
                    items.push(self.construct_from_bytes(elem, chunk)?);
                }
                Some(Value::Tuple(items))
            }
            _ => self.construct_from_bytes(&leaf, &bytes),
        };
        built.map(Ok)
    }

    /// The write side of [`Self::view_read`] (see
    /// [`Self::foreign_write`] for the place-shaped form):
    /// the value's bytes are spliced into the birth storage
    /// at the span the recorded denomination prices. A
    /// projection crossing a union field declines: the union
    /// crossing machinery composes such writes with the held
    /// field's bytes, as on the ordinary path.
    pub(crate) fn view_write(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: &Value,
    ) -> Option<InterpResult<()>> {
        let (birth, projection) = self.target_birth(frames, target)?;
        if projection
            .iter()
            .any(|s| matches!(s, RefProj::UnionField { .. }))
        {
            return None;
        }
        let (off, size, leaf) = self.view_span(frames, target)?;
        let value_size = usize::try_from(self.ty_size(&leaf)?).ok()?;
        if value_size > size {
            return None;
        }
        let size = value_size;
        let bytes = self.written_value_bytes(value, &leaf, size)?;
        let root = self.target_root_value(frames, target)?;
        let layout = self.byte_layout_of_ty(&birth)?;
        let mut composed = root;
        for (i, b) in bytes.iter().enumerate() {
            let Some(b) = b else { continue };
            match self.splice_storage_byte(
                &mut composed,
                &layout,
                off + i,
                *b,
            ) {
                Some(Ok(())) => {}
                Some(Err(ByteVerdict::Padding(_))) => {}
                Some(Err(verdict)) => {
                    let (func, desc) =
                        self.target_desc(frames, target);
                    return Some(Err(self.byte_view_error(
                        verdict, &func, &desc, false,
                    )));
                }
                None => return None,
            }
        }
        self.store_target_root(frames, target, composed)
    }

    /// Read a place through a foreign view (see the section
    /// comment): the span's bytes come out of the birth
    /// value's storage and the result is constructed as a
    /// value of the place's own (accessing) type. `None` when
    /// the access is not foreign or the view cannot judge it;
    /// a judged uninitialised byte in the span is the read's
    /// error, as on the ordinary path.
    /// The elements of a fat pointer whose base denotes the
    /// root of a storage born as an array of a differently-
    /// shaped element: the pointer's formation regrouped the
    /// storage's bytes under `elem_ty` (the `[[u8; 3]; 2]`
    /// storage a `*const [[u8; 2]]` ranges over), so the
    /// elements are constructed from the storage's bytes
    /// under that type, `len` of them from the storage's
    /// start. The base may reach the root through a leading
    /// step that starts at the storage's first byte under
    /// any denomination (a whole-slice re-slice, a
    /// first-element pointer); a step starting further in
    /// is denominated by the view that formed it, which the
    /// target does not record, so the view declines it.
    /// `None` when the base does not denote such a storage
    /// root (a stored projection, a value target, an
    /// allocation element block), the storage's own element
    /// already is `elem_ty` (the ordinary element readings
    /// denominate it correctly), the regrouped extent runs
    /// past the storage, or a byte cannot be judged.
    pub(crate) fn foreign_slice_elements(
        &self,
        frames: &[Frame],
        value: &Value,
        elem_ty: &Ty,
    ) -> Option<Vec<Value>> {
        let Value::FatRef { base, len } = value else {
            return None;
        };
        let target = self.deref_value(frames, base).ok()?;
        let (birth, projection) = self.target_birth(frames, &target)?;
        // The base must denote the storage root. A leading
        // step that starts at the storage's first byte (a
        // whole-slice re-slice, a first-element pointer)
        // leaves it there, and so does a denomination the
        // base's dereference recorded (see `RefProj::View`),
        // which does not move the position: the base still
        // denotes the storage root, under the recorded view.
        // The composition below denominates by `elem_ty`, so
        // a recorded sequence view must denominate by the
        // same element; any other step, or a view of another
        // shape, is not this composition.
        let root_position = match projection {
            []
            | [RefProj::Subslice { offset: 0, .. }]
            | [RefProj::Index(0)] => true,
            _ => projection.iter().all(|step| {
                matches!(
                    step,
                    RefProj::View(Ty::Slice(e) | Ty::Array(e, _)) if **e == *elem_ty
                )
            }),
        };
        if !root_position {
            return None;
        }
        let Ty::Array(birth_elem, _) = &birth else {
            return None;
        };
        if **birth_elem == *elem_ty {
            return None;
        }
        let elem_size = usize::try_from(self.ty_size(elem_ty)?).ok()?;
        if elem_size == 0 {
            return None;
        }
        let count = usize::try_from(*len).ok()?;
        let extent = count.checked_mul(elem_size)?;
        if extent > usize::try_from(self.ty_size(&birth)?).ok()? {
            return None;
        }
        let root = self.target_root_value(frames, &target)?;
        let layout = self.byte_layout_of_ty(&birth)?;
        let mut bytes = Vec::with_capacity(extent);
        for i in 0..extent {
            match self.foreign_byte(&root, &layout, i)? {
                ForeignByte::Byte(b) => bytes.push(Some(b)),
                ForeignByte::Padding => bytes.push(None),
                ForeignByte::Uninit => return None,
            }
        }
        let mut items = Vec::with_capacity(count);
        for chunk in bytes.chunks(elem_size) {
            items.push(self.construct_from_bytes(elem_ty, chunk)?);
        }
        Some(items)
    }

    pub(crate) fn foreign_read(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
        target: &Resolved,
    ) -> Option<InterpResult<Value>> {
        let access_root = self.foreign_access_root(body, place)?;
        let birth = self.foreign_birth_ty(frames, target, &access_root)?;
        let (off, size) =
            self.foreign_span(frames, body, target, &access_root, &birth, place)?;
        if size == 0 {
            return None;
        }
        let place_ty = self.foreign_place_ty(body, place)?;
        // As on the write path, the span can carry the slot's
        // tail padding past the value's own extent; the value
        // read back is the value's extent. A slice value's
        // extent is its span: the fat length times the element
        // size, which the span priced as the whole access.
        let value_size = match &place_ty {
            Ty::Slice(elem) => {
                let elem_size = usize::try_from(self.ty_size(elem)?).ok()?;
                if elem_size == 0 || size % elem_size != 0 {
                    return None;
                }
                size
            }
            _ => usize::try_from(self.ty_size(&place_ty)?).ok()?,
        };
        if value_size > size {
            return None;
        }
        let size = value_size;
        let root = self.target_root_value(frames, target)?;
        let layout = self.byte_layout_of_ty(&birth)?;
        let mut bytes = Vec::with_capacity(size);
        for i in 0..size {
            match self.foreign_byte(&root, &layout, off + i)? {
                ForeignByte::Byte(b) => bytes.push(Some(b)),
                ForeignByte::Padding => bytes.push(None),
                ForeignByte::Uninit => {
                    let (func, desc) = self.target_desc(frames, target);
                    return Some(Err(InterpError::UninitializedRead {
                        func,
                        place: desc,
                    }));
                }
            }
        }
        let built = match &place_ty {
            Ty::Slice(elem) => {
                // A slice value is its element tuple: each
                // element is constructed from its own byte
                // chunk of the span.
                let elem_size =
                    usize::try_from(self.ty_size(elem)?).ok()?;
                let mut items = Vec::with_capacity(size / elem_size);
                for chunk in bytes.chunks(elem_size) {
                    items.push(self.construct_from_bytes(elem, chunk)?);
                }
                Some(Value::Tuple(items))
            }
            _ => self.construct_from_bytes(&place_ty, &bytes),
        };
        built.map(Ok)
    }

    /// Set an enum place's discriminant through a foreign
    /// view: the discriminant's own bytes are spliced into
    /// the birth storage at the tag position the place's
    /// span denotes, re-forming the birth value at the named
    /// variant (the field writes that follow land in the new
    /// variant's fields). A direct-tag enum's bytes are its
    /// recorded tag's; a niche-encoded enum's are the niche
    /// scalar's pattern for a niche variant. Setting the
    /// untagged variant of a niche-encoded enum declines:
    /// its niche scalar is the variant's own field value,
    /// which the field write that follows a whole-value
    /// assignment provides, and a bare discriminant set has
    /// no byte of its own to write.
    pub(crate) fn foreign_set_discriminant(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        place: &Place,
        target: &Resolved,
        variant: u32,
    ) -> Option<InterpResult<()>> {
        let access_root = self.foreign_access_root(body, place)?;
        let birth = self.foreign_birth_ty(frames, target, &access_root)?;
        let (off, _size) =
            self.foreign_span(frames, body, target, &access_root, &birth, place)?;
        let place_ty = self.foreign_place_ty(body, place)?;
        let Ty::Adt(name) = &place_ty else {
            return None;
        };
        let (base, bytes): (usize, Vec<u8>) =
            if let Some(tag) = self.program().adt_tag_layouts.get(name) {
                let discr = *tag.discriminants.get(variant as usize)?;
                (
                    off + tag.offset as usize,
                    (0..tag.size as usize)
                        .map(|i| (discr >> (8 * i)) as u8)
                        .collect(),
                )
            } else if let Some(niche) = self.program().adt_niches.get(name) {
                if variant == niche.untagged_variant
                    || !(niche.niche_variants.0..=niche.niche_variants.1)
                        .contains(&variant)
                {
                    return None;
                }
                let scalar = niche.niche_start
                    + u128::from(variant - niche.niche_variants.0);
                let width = usize::try_from(niche.tag_bits / 8).ok()?;
                (
                    off + niche.niche_offset as usize,
                    (0..width)
                        .map(|i| (scalar >> (8 * i)) as u8)
                        .collect(),
                )
            } else {
                return None;
            };
        let root = self.target_root_value(frames, target)?;
        let layout = self.byte_layout_of_ty(&birth)?;
        let mut composed = root;
        for (i, byte) in bytes.iter().enumerate() {
            match self.splice_storage_byte(&mut composed, &layout, base + i, *byte) {
                Some(Ok(())) => {}
                Some(Err(ByteVerdict::Padding(_))) => {}
                Some(Err(verdict)) => {
                    let (func, desc) = self.target_desc(frames, target);
                    return Some(Err(self.byte_view_error(
                        verdict, &func, &desc, false,
                    )));
                }
                None => return None,
            }
        }
        self.store_target_root(frames, target, composed)
    }

    /// Write `value` to a place through a foreign view (see
    /// the section comment): the value's byte image is
    /// spliced into the birth value's storage at the span the
    /// accessing projection denotes, and the composed birth
    /// value is stored back — so a later birth-typed read
    /// sees exactly the value the written bytes form. The
    /// splice runs on a clone of the stored value: a view
    /// that declines midway leaves the storage untouched and
    /// the ordinary path runs instead.
    pub(crate) fn foreign_write(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        place: &Place,
        target: &Resolved,
        value: &Value,
    ) -> Option<InterpResult<()>> {
        let access_root = self.foreign_access_root(body, place)?;
        let birth = self.foreign_birth_ty(frames, target, &access_root)?;
        let (off, size) =
            self.foreign_span(frames, body, target, &access_root, &birth, place)?;
        if size == 0 {
            return None;
        }
        let place_ty = self.foreign_place_ty(body, place)?;
        // The span prices the place's slot in the accessing
        // type, whose recorded extent can carry the slot's
        // tail padding (a field whose recorded size rounds up
        // to the containing struct's alignment); the value's
        // own extent is what a write of the value determines.
        // A span smaller than the value cannot hold it.
        let value_size = usize::try_from(self.ty_size(&place_ty)?).ok()?;
        if value_size > size {
            return None;
        }
        let size = value_size;
        let bytes = self.written_value_bytes(value, &place_ty, size)?;
        let root = self.target_root_value(frames, target)?;
        let layout = self.byte_layout_of_ty(&birth)?;
        let mut composed = root;
        for (i, b) in bytes.iter().enumerate() {
            let Some(b) = b else {
                continue;
            };
            match self.splice_storage_byte(&mut composed, &layout, off + i, *b) {
                Some(Ok(())) => {}
                Some(Err(ByteVerdict::Padding(_))) => {}
                Some(Err(verdict)) => {
                    let (func, desc) = self.target_desc(frames, target);
                    return Some(Err(self.byte_view_error(
                        verdict, &func, &desc, false,
                    )));
                }
                None => {
                    return None;
                }
            }
        }
        self.store_target_root(frames, target, composed)
    }
}

/// The scalar shape a declared type names, when it names a
/// scalar at all: the `Ty` form of [`scalar_shape_of_tag`].
pub(crate) fn scalar_shape_of_ty(ty: &Ty) -> Option<ScalarShape> {
    Some(match ty {
        Ty::Bool => ScalarShape::Bool,
        Ty::Int(t) => ScalarShape::Int {
            bits: t.bits(),
            signed: true,
        },
        Ty::Uint(t) => ScalarShape::Int {
            bits: t.bits(),
            signed: false,
        },
        Ty::Float(t) => ScalarShape::Float { bits: t.bits() },
        _ => return None,
    })
}

/// The scalar shape a rendered type tag names (a format
/// argument's type tag, a builtin's element tag), with any
/// reference or pointer wrappers stripped: `&u32`, `*const f32`
/// and `u32` all name the same shape. `None` when the tag names
/// no scalar type.
pub(crate) fn scalar_shape_of_tag(tag: &str) -> Option<ScalarShape> {
    let mut tag = tag;
    loop {
        tag = tag.trim_start();
        if let Some(rest) = tag.strip_prefix("&mut ") {
            tag = rest;
        } else if let Some(rest) = tag.strip_prefix('&') {
            tag = rest;
        } else if let Some(rest) = tag.strip_prefix("*const ") {
            tag = rest;
        } else if let Some(rest) = tag.strip_prefix("*mut ") {
            tag = rest;
        } else {
            break;
        }
    }
    match tag.trim() {
        "bool" => Some(ScalarShape::Bool),
        "char" => Some(ScalarShape::Char),
        "i8" => Some(ScalarShape::Int { bits: 8, signed: true }),
        "i16" => Some(ScalarShape::Int { bits: 16, signed: true }),
        "i32" => Some(ScalarShape::Int { bits: 32, signed: true }),
        "i64" => Some(ScalarShape::Int { bits: 64, signed: true }),
        "i128" => Some(ScalarShape::Int { bits: 128, signed: true }),
        "isize" => Some(ScalarShape::Int { bits: 64, signed: true }),
        "u8" => Some(ScalarShape::Int { bits: 8, signed: false }),
        "u16" => Some(ScalarShape::Int { bits: 16, signed: false }),
        "u32" => Some(ScalarShape::Int { bits: 32, signed: false }),
        "u64" => Some(ScalarShape::Int { bits: 64, signed: false }),
        "u128" => Some(ScalarShape::Int { bits: 128, signed: false }),
        "usize" => Some(ScalarShape::Int { bits: 64, signed: false }),
        "f16" => Some(ScalarShape::Float { bits: 16 }),
        "f32" => Some(ScalarShape::Float { bits: 32 }),
        "f64" => Some(ScalarShape::Float { bits: 64 }),
        "f128" => Some(ScalarShape::Float { bits: 128 }),
        _ => None,
    }
}

/// The (kind class, bit width) a byte-view scalar kind is
/// compared by: two scalar readings of one storage agree
/// exactly when their keys do.
fn scalar_kind_key(kind: ByteScalar) -> (u8, u32) {
    match kind {
        ByteScalar::Int(t) => (1, t.bits()),
        ByteScalar::Uint(t) => (2, t.bits()),
        ByteScalar::Float(t) => (3, t.bits()),
        ByteScalar::Bool => (4, 8),
        ByteScalar::Char => (5, 32),
    }
}

/// Whether `value` is already a value of `shape`'s own
/// kind: a signed shape over a signed value, an unsigned
/// shape over an unsigned one, a float shape over a float
/// value, a boolean over a boolean, a char over its code
/// point. A value that merely flattens under the shape (an
/// unsigned integer under a signed shape of the same width)
/// is a different reading of the same bits, not the shape's
/// value.
pub(crate) fn scalar_agrees_with_shape(
    value: &Value,
    shape: &ScalarShape,
) -> bool {
    matches!(
        (shape, value),
        (ScalarShape::Int { signed: true, .. }, Value::Int(_))
            | (ScalarShape::Int { signed: false, .. }, Value::Uint(_))
            | (ScalarShape::Float { bits: 128 }, Value::Float128(_))
            | (ScalarShape::Float { .. }, Value::Float(_))
            | (ScalarShape::Bool, Value::Bool(_))
            | (ScalarShape::Char, Value::Uint(_))
    )
}

/// The key of [`scalar_kind_key`] for a scalar shape; a
/// non-scalar shape's key matches no scalar kind's.
fn scalar_shape_key(shape: &ScalarShape) -> (u8, u32) {
    match shape {
        ScalarShape::Int { bits, signed } => {
            if *signed {
                (1, *bits)
            } else {
                (2, *bits)
            }
        }
        ScalarShape::Float { bits } => (3, *bits),
        ScalarShape::Bool => (4, 8),
        ScalarShape::Char => (5, 32),
        _ => (0, 0),
    }
}

/// A scalar shape for the byte view of a value's storage (see
/// `Interpreter::storage_byte`): the scalar's bytes are its own
/// declared width's, extracted from the stored value's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ByteScalar {
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Bool,
    /// A `char` field's storage: its code point's four bytes.
    /// (`char` locals ride as `u32` in the IR type model; the
    /// shape arises from ADT field tags.)
    Char,
}

impl ByteScalar {
    /// The scalar's storage width in bytes.
    fn width(self) -> usize {
        match self {
            ByteScalar::Int(t) => (t.bits() / 8) as usize,
            ByteScalar::Uint(t) => (t.bits() / 8) as usize,
            ByteScalar::Float(t) => (t.bits() / 8) as usize,
            ByteScalar::Bool => 1,
            ByteScalar::Char => 4,
        }
    }
}

/// One byte of a value's storage as the foreign view judges
/// it (see `Interpreter::foreign_byte`).
enum ForeignByte {
    /// A definite byte of the storage.
    Byte(u8),
    /// A byte of padding: inside the storage, covered by no
    /// field and no tag, with no definite value.
    Padding,
    /// A byte of storage that was never written.
    Uninit,
}

/// The layout the byte view of a value's storage descends by
/// (see `Interpreter::storage_byte`).
#[derive(Clone, Debug)]
enum ByteLayout {
    Scalar(ByteScalar),
    /// Elements of one layout repeated: (element layout,
    /// element size in bytes, element count).
    Seq(Box<ByteLayout>, usize, usize),
    /// An ADT instance by its rendered name, resolved through
    /// the recorded field layouts and debug shapes.
    Adt(String),
}

/// The verdict a judged byte-view position meets (see
/// `Interpreter::storage_byte`).
#[derive(Clone, Copy, Debug)]
enum ByteVerdict {
    /// The byte's storage was never written.
    Uninit,
    /// The position is outside the value's storage.
    Outside(usize),
    /// A write position falls in the padding between an
    /// ADT's fields: it has no storage to splice into. (A
    /// read of a padding position is not a verdict: it
    /// answers the indeterminate byte, see [`StorageByte`].)
    Padding(usize),
    /// A byte write into a `char` field assembled a value that
    /// is not a `char`.
    InvalidChar(u32),
}

/// What a judged byte-view position holds (see
/// `Interpreter::storage_byte`).
#[derive(Clone, Copy, Debug)]
enum StorageByte {
    /// A definite byte of the value's storage: a scalar's own
    /// byte, a field's byte, or an enum tag's byte.
    Byte(u8),
    /// A byte of an ADT's padding: the position is inside the
    /// value's storage but covered by no field and no tag, so
    /// it has no definite value. A read of it yields the value
    /// model's indeterminate value rather than an error or a
    /// fabricated byte.
    Padding,
}

/// The failure a value projector or place resolution reports
/// when an index step names a position past the end of the value
/// it ranges over. A projection only names a position; it is not
/// a checked operation. Checked accesses panic at their own site
/// -- the MIR `Assert` terminator for a source-level index, the
/// modelled `Index` operation for a library index -- before any
/// projection is evaluated, so a position past the stored value
/// was reached through an unchecked route (raw-pointer
/// arithmetic, `get_unchecked`, a bulk-memory intrinsic) and the
/// model holds no storage there. That is an explicit error in the
/// model's terms, never a fabricated guest panic: the guest
/// indexed nothing.
/// Whether an error is a projection bounds failure naming a
/// position past the stored value — the failure `index_oob`
/// reports. Position-judging consumers (the append point a tail
/// write targets in `seq_element_target`, an allocation's
/// unwritten prefix slot in `read_resolved`) discriminate on
/// this failure: the position still denotes its slot, whatever
/// form the failure takes. The explicit projection error is the
/// current form; the guest-panic form predates it, and the
/// modelled index operations still raise that form directly.
pub(crate) fn is_index_oob_error(e: &InterpError) -> bool {
    match e {
        InterpError::Panic { message, .. } => {
            message.starts_with("index out of bounds: the len is ")
        }
        InterpError::UnsupportedProjection { kind } => {
            kind.contains("past the end of the stored value")
        }
        _ => false,
    }
}

pub(crate) fn index_oob(len: usize, index: usize) -> InterpError {
    InterpError::UnsupportedProjection {
        kind: format!(
            "index {index} past the end of the stored value (length {len})"
        ),
    }
}

/// Byte `index` of an integer value's little-endian
/// representation (see `Interpreter::seq_scalar`). The model's
/// word is 128 bits; a byte past it is outside every integer
/// type the model holds, so it is an explicit error rather
/// than a fabricated byte.
/// The scalar leaves of a structured scalar value (a vector
/// value's tree of integer/float/bool leaves inside its
/// aggregate wrappers), in storage order; `None` when any part
/// of the tree is not a scalar leaf or a plain aggregate over
/// them. Byte-granular bulk operations (the byte copy an
/// unaligned vector load lowers to) address such a value's
/// bytes, which the indexed access cannot express: the leaves'
/// widths are not recorded in the values themselves.
fn scalar_tree_leaves(value: &Value) -> Option<Vec<&Value>> {
    fn walk<'v>(value: &'v Value, out: &mut Vec<&'v Value>) -> Option<()> {
        match value {
            Value::Int(_) | Value::Uint(_) | Value::Float(_) | Value::Bool(_) => {
                out.push(value);
            }
            Value::Tuple(items) => {
                for item in items {
                    walk(item, out)?;
                }
            }
            Value::Enum { fields, .. } => {
                for field in fields {
                    walk(field, out)?;
                }
            }
            _ => return None,
        }
        Some(())
    }
    let mut out = Vec::new();
    walk(value, &mut out)?;
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Whether `value` is a structured scalar tree (an aggregate
/// over scalar leaves) rather than a plain scalar or a
/// sequence value the indexed access already handles.
pub(crate) fn is_scalar_tree(value: &Value) -> bool {
    matches!(value, Value::Tuple(_) | Value::Enum { .. })
        && scalar_tree_leaves(value).is_some()
}

/// The leaf width a byte view of a scalar tree uses: every
/// leaf must share one kind, and the total bit count must
/// divide evenly into a width the model can store. Vector
/// values — the trees this serves — are uniform by
/// construction (a `Simd<T, N>` storage array, a wrapper over
/// one scalar); a mixed tree has no honest byte view here and
/// keeps the plain access's outcome.
fn scalar_tree_leaf_width(leaves: &[&Value], total_bits: u32) -> Option<u32> {
    let count = leaves.len() as u32;
    if count == 0 || total_bits % count != 0 {
        return None;
    }
    let width = total_bits / count;
    let kind = leaves[0];
    if !leaves.iter().all(|leaf| {
        matches!(
            (kind, leaf),
            (Value::Int(_), Value::Int(_))
                | (Value::Uint(_), Value::Uint(_))
                | (Value::Float(_), Value::Float(_))
                | (Value::Bool(_), Value::Bool(_))
        )
    }) {
        return None;
    }
    match kind {
        Value::Int(_) | Value::Uint(_) if matches!(width, 8 | 16 | 32 | 64 | 128) => Some(width),
        Value::Float(_) if matches!(width, 32 | 64) => Some(width),
        Value::Bool(_) if width == 8 => Some(width),
        _ => None,
    }
}

/// Rebuild a scalar tree with its leaf bits taken from
/// `bytes` (little-endian, in storage order), preserving the
/// tree's shape and leaf kinds. `None` when the tree has no
/// uniform byte view (see [`scalar_tree_leaf_width`]).
pub(crate) fn byte_patch_tree(current: &Value, bytes: &[u8]) -> Option<Value> {
    let leaves = scalar_tree_leaves(current)?;
    let width = scalar_tree_leaf_width(&leaves, (bytes.len() as u32).checked_mul(8)?)?;
    let mut pos = 0usize;
    fn build(template: &Value, bytes: &[u8], width: u32, pos: &mut usize) -> Option<Value> {
        let nbytes = (width / 8) as usize;
        let take = |pos: &mut usize| -> Option<u128> {
            let mut bits = 0u128;
            for i in 0..nbytes {
                bits |= u128::from(*bytes.get(*pos + i)?) << (8 * i);
            }
            *pos += nbytes;
            Some(bits)
        };
        match template {
            Value::Int(_) => {
                let bits = take(pos)?;
                Some(Value::Int(wrap_signed(bits as i128, width)))
            }
            Value::Uint(_) => {
                let bits = take(pos)?;
                Some(Value::Uint(wrap_unsigned(bits, width)))
            }
            Value::Float(_) => {
                let bits = take(pos)?;
                Some(Value::Float(match width {
                    32 => crate::f128::f32_bits_to_f64_exact(bits as u32),
                    _ => f64::from_bits(bits as u64),
                }))
            }
            Value::Bool(_) => {
                let bits = take(pos)?;
                Some(Value::Bool(bits != 0))
            }
            Value::Tuple(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(build(item, bytes, width, pos)?);
                }
                Some(Value::Tuple(out))
            }
            Value::Enum { variant, fields } => {
                let mut out = Vec::with_capacity(fields.len());
                for field in fields {
                    out.push(build(field, bytes, width, pos)?);
                }
                Some(Value::Enum {
                    variant: *variant,
                    fields: out,
                })
            }
            _ => None,
        }
    }
    build(current, bytes, width, &mut pos)
}

/// Serialize a scalar tree to its little-endian bytes when it
/// has a uniform byte view of exactly `total` bytes (the
/// mirror of [`byte_patch_tree`], for a vector store's source).
pub(crate) fn tree_to_bytes(value: &Value, total: usize) -> Option<Vec<u8>> {
    let leaves = scalar_tree_leaves(value)?;
    let width = scalar_tree_leaf_width(&leaves, (total as u32).checked_mul(8)?)?;
    let nbytes = (width / 8) as usize;
    let mut out = Vec::with_capacity(total);
    for leaf in leaves {
        let bits: u128 = match leaf {
            Value::Int(i) => (*i as u128) & bit_mask(width),
            Value::Uint(u) => *u & bit_mask(width),
            Value::Float(f) => match width {
                32 => u128::from(crate::f128::f64_to_f32_bits_exact(*f)),
                _ => u128::from(f.to_bits()),
            },
            Value::Bool(b) => u128::from(*b),
            _ => return None,
        };
        for i in 0..nbytes {
            out.push((bits >> (8 * i)) as u8);
        }
    }
    if out.len() == total {
        Some(out)
    } else {
        None
    }
}

pub(crate) fn scalar_byte(value: &Value, index: usize) -> InterpResult<u8> {
    if index >= 16 {
        return Err(InterpError::UnsupportedProjection {
            kind: format!(
                "byte {index} of a scalar value is outside the model's word"
            ),
        });
    }
    let bits = match value {
        Value::Int(v) => *v as u128,
        Value::Uint(v) => *v,
        other => {
            return Err(InterpError::type_mismatch(
                "scalar byte access",
                "integer value",
                other.kind_name(),
            ));
        }
    };
    Ok((bits >> (index * 8)) as u8)
}

/// The integer value with byte `index` of its little-endian
/// representation replaced by `byte` (see
/// [`scalar_byte`]). The signed form keeps its two's-complement
/// word: patching a byte of a negative value patches the
/// representation the native write would touch.
pub(crate) fn scalar_with_byte(value: &Value, index: usize, byte: u8) -> InterpResult<Value> {
    if index >= 16 {
        return Err(InterpError::UnsupportedProjection {
            kind: format!(
                "byte {index} of a scalar value is outside the model's word"
            ),
        });
    }
    let shift = index * 8;
    let mask = !(0xFFu128 << shift);
    Ok(match value {
        Value::Int(v) => {
            Value::Int(((*v as u128 & mask) | ((byte as u128) << shift)) as i128)
        }
        Value::Uint(v) => Value::Uint((*v & mask) | ((byte as u128) << shift)),
        other => {
            return Err(InterpError::type_mismatch(
                "scalar byte access",
                "integer value",
                other.kind_name(),
            ));
        }
    })
}

/// Apply constant projection steps to a value (read direction).
/// The canonical value of a type that is zero-sized by its IR
/// shape alone (see `Interpreter::zst_local_value`): unit, tuples
/// of zero-sized types, arrays of zero-sized elements and empty
/// arrays, zero-capture closures, function items. ADT values need
/// the driver's shape tables and are only synthesized at
/// conversion time.
pub(crate) fn zst_value_of_ty(ty: &Ty) -> Option<Value> {
    match ty {
        Ty::Unit => Some(Value::Unit),
        Ty::Tuple(ts) => {
            let mut vals = Vec::with_capacity(ts.len());
            for t in ts {
                vals.push(zst_value_of_ty(t)?);
            }
            Some(Value::Tuple(vals))
        }
        Ty::Array(_, 0) => Some(Value::Tuple(Vec::new())),
        Ty::Array(elem, n) if *n <= 4096 => {
            let v = zst_value_of_ty(elem)?;
            Some(Value::Tuple(vec![v; *n as usize]))
        }
        Ty::Closure(def, 0) => Some(Value::Closure {
            def: def.clone(),
            env: Vec::new(),
        }),
        Ty::FnDef(name) => Some(Value::FnPtr(name.clone())),
        _ => None,
    }
}

/// The bits of a scalar value's storage under a declared
/// scalar kind: the value serialized at its own declared
/// width — an `f32`'s bits are its `f32` pattern, not the
/// widened `f64` model's, and an `f16`'s its `f16` pattern.
/// `None` when the value does not hold a scalar of the kind.
/// The byte view's scalar arm (`Interpreter::storage_byte`)
/// and the projectors' declared-width byte arms (see
/// [`scalar_storage_byte_as`]) share this extraction.
fn scalar_bits_of_kind(value: &Value, kind: ByteScalar) -> Option<u128> {
    Some(match (kind, value) {
        (ByteScalar::Int(_), Value::Int(i)) => *i as u128,
        (ByteScalar::Uint(_), Value::Uint(u)) => *u,
        (ByteScalar::Char, Value::Uint(u)) => *u,
        (ByteScalar::Bool, Value::Bool(b)) => u128::from(*b),
        (ByteScalar::Float(FloatTy::F16), Value::Float(f)) => {
            u128::from(crate::f128::f64_to_f16_bits(*f))
        }
        (ByteScalar::Float(FloatTy::F32), Value::Float(f)) => {
            u128::from((*f as f32).to_bits())
        }
        (ByteScalar::Float(FloatTy::F64), Value::Float(f)) => u128::from(f.to_bits()),
        (ByteScalar::Float(FloatTy::F128), Value::Float128(bits)) => *bits,
        _ => return None,
    })
}

/// The declared scalar kind a position's type evidence
/// names, when it names a scalar at all: the scalar arms of
/// the byte view's layout resolution
/// (`Interpreter::byte_layout_of_ty`), in the form the
/// projectors' evidence takes — a `Ty`, not a resolved
/// layout.
pub(crate) fn byte_scalar_of_ty(ty: Option<&Ty>) -> Option<ByteScalar> {
    Some(match ty? {
        Ty::Bool => ByteScalar::Bool,
        Ty::Int(t) => ByteScalar::Int(*t),
        Ty::Uint(t) => ByteScalar::Uint(*t),
        Ty::Float(t) => ByteScalar::Float(*t),
        _ => return None,
    })
}

/// Whether a declared scalar kind is a float narrower than
/// the model's `f64` storage: the kinds whose declared bytes
/// the evidence-free scalar storage forms
/// ([`scalar_storage_byte`], [`scalar_storage_len`],
/// [`splice_scalar_storage_byte`]) do not describe, because
/// they serialize the widened `f64` bits. Every other kind's
/// evidence-free answers are already its declared ones (an
/// integer's low bytes, a boolean's byte, an `f128`'s raw
/// bits), so only these kinds make type evidence change a
/// byte arm's answer.
fn narrowed_float_kind(kind: ByteScalar) -> bool {
    matches!(
        kind,
        ByteScalar::Float(FloatTy::F16) | ByteScalar::Float(FloatTy::F32)
    )
}

/// [`scalar_storage_len`] under declared-kind evidence (see
/// [`byte_scalar_of_ty`]): a float storage narrower than the
/// model form is exactly its declared width long; every
/// other scalar keeps the evidence-free length.
pub(crate) fn scalar_storage_len_as(value: &Value, kind: Option<ByteScalar>) -> Option<usize> {
    match (kind, value) {
        (Some(kind), Value::Float(_)) if narrowed_float_kind(kind) => Some(kind.width()),
        _ => scalar_storage_len(value),
    }
}

/// [`scalar_storage_byte`] under declared-kind evidence (see
/// [`byte_scalar_of_ty`]): a float storage narrower than the
/// model form yields its own width's bytes, and no byte past
/// them; every other scalar keeps the evidence-free byte,
/// which is already its declared one.
pub(crate) fn scalar_storage_byte_as(value: &Value, kind: Option<ByteScalar>, idx: usize) -> Option<u8> {
    match (kind, value) {
        (Some(kind), Value::Float(_)) if narrowed_float_kind(kind) => {
            if idx >= kind.width() {
                return None;
            }
            let bits = scalar_bits_of_kind(value, kind)?;
            Some((bits >> (8 * idx)) as u8)
        }
        _ => scalar_storage_byte(value, idx),
    }
}

/// The number of bytes in a scalar value's storage, for the
/// byte-level view of that storage a raw byte pointer has (see
/// [`scalar_storage_byte`]). Integers ride in a 128-bit slot, so
/// their storage is the full 16 bytes; the guest only ever
/// addresses the bytes of the value's own declared width, which
/// are the low ones.
pub(crate) fn scalar_storage_len(value: &Value) -> Option<usize> {
    match value {
        Value::Bool(_) => Some(1),
        Value::Float(_) => Some(8),
        Value::Int(_) | Value::Uint(_) | Value::Float128(_) => Some(16),
        _ => None,
    }
}

/// Byte `idx` of a scalar value's storage, little-endian: the
/// byte a `*const u8` cast of the value's address reads at that
/// offset. Bulk byte copies move data in and out of scalar
/// buffers this way (SipHash's `Hasher::write` assembles its
/// next word in a `u64` local), so an index step applied to a
/// scalar value denotes this byte, the only element a scalar's
/// storage has. `None` for a non-scalar value or an index past
/// the storage.
pub(crate) fn scalar_storage_byte(value: &Value, idx: usize) -> Option<u8> {
    let len = scalar_storage_len(value)?;
    if idx >= len {
        return None;
    }
    let bits: u128 = match value {
        Value::Bool(b) => u128::from(*b),
        Value::Int(i) => *i as u128,
        Value::Uint(u) => *u,
        Value::Float(f) => u128::from(f.to_bits()),
        Value::Float128(bits) => *bits,
        _ => return None,
    };
    Some((bits >> (8 * idx)) as u8)
}

/// The byte a written value carries when it is a byte value: a
/// `u8` (or an `i8`, or a `bool`) written through a byte pointer.
pub(crate) fn value_as_storage_byte(value: &Value) -> Option<u8> {
    match value {
        Value::Uint(u) if *u <= 0xff => Some(*u as u8),
        Value::Int(i) if (-128..=127).contains(i) => Some(*i as u8),
        Value::Bool(b) => Some(*b as u8),
        _ => None,
    }
}

/// Splice `byte` into byte `idx` of a scalar value's storage
/// (see [`scalar_storage_byte`]), in place. An unwritten slot is
/// zero storage for this purpose: a buffer being filled byte by
/// byte is written whole before it is read, so the bytes the
/// fill has not reached yet are never observed. Returns whether
/// the splice applied (the slot held a scalar, or no value, and
/// the index is inside the storage).
fn splice_scalar_storage_byte(slot: &mut Value, idx: usize, byte: u8) -> bool {
    let len = match &*slot {
        Value::Uninit => 16,
        other => match scalar_storage_len(other) {
            Some(len) => len,
            None => return false,
        },
    };
    if idx >= len {
        return false;
    }
    let bits: u128 = match &*slot {
        Value::Bool(b) => u128::from(*b),
        Value::Int(i) => *i as u128,
        Value::Uint(u) => *u,
        Value::Float(f) => u128::from(f.to_bits()),
        Value::Float128(bits) => *bits,
        _ => 0,
    };
    let bits = (bits & !(0xffu128 << (8 * idx))) | (u128::from(byte) << (8 * idx));
    *slot = match &*slot {
        Value::Bool(_) => Value::Bool(bits != 0),
        Value::Int(_) => Value::Int(bits as i128),
        Value::Float(_) => Value::Float(f64::from_bits(bits as u64)),
        Value::Float128(_) => Value::Float128(bits),
        _ => Value::Uint(bits),
    };
    true
}

/// [`splice_scalar_storage_byte`] under declared-kind
/// evidence (see [`byte_scalar_of_ty`]): a byte written into
/// a float storage narrower than the model form splices into
/// the storage's own width's bits — the write side of
/// [`scalar_storage_byte_as`], with the same bit conventions
/// as the byte view's write arm
/// (`Interpreter::splice_storage_byte`). A byte position
/// past the declared width does not apply. Every other
/// scalar keeps the evidence-free splice, whose widening
/// assumption is already its declared behaviour there.
pub(crate) fn splice_scalar_storage_byte_as(
    slot: &mut Value,
    kind: Option<ByteScalar>,
    idx: usize,
    byte: u8,
) -> bool {
    // A byte position at or past the declared width is not a
    // byte of this scalar's storage: splicing there would
    // widen the stored value past its type (the model's word
    // is wider than every declared integer width), corrupting
    // the value the storage holds. The evidence-free splice
    // below keeps the widened-word convention for callers
    // without declared-kind evidence.
    if let Some(kind) = kind {
        if idx >= kind.width() {
            return false;
        }
    }
    match (kind, &*slot) {
        (Some(kind), Value::Float(f)) if narrowed_float_kind(kind) => {
            if idx >= kind.width() {
                return false;
            }
            let shift = 8 * idx;
            let splice = |bits: u128| -> u128 {
                (bits & !(0xffu128 << shift)) | (u128::from(byte) << shift)
            };
            *slot = match kind {
                ByteScalar::Float(FloatTy::F16) => {
                    let bits = splice(u128::from(crate::f128::f64_to_f16_bits(*f)));
                    Value::Float(crate::f128::f16_bits_to_f64(bits as u16))
                }
                ByteScalar::Float(FloatTy::F32) => {
                    let bits = splice(u128::from(crate::f128::f64_to_f32_bits_exact(*f)));
                    Value::Float(crate::f128::f32_bits_to_f64_exact(bits as u32))
                }
                _ => unreachable!("narrowed float kind checked"),
            };
            true
        }
        _ => splice_scalar_storage_byte(slot, idx, byte),
    }
}

/// The field values of a `Ref` / `RefMut` guard value: field 0
/// is the reference to the cell's value, field 1 the reference
/// to the borrow-flag `Cell`. Both guard shapes in circulation
/// are accepted — the modelled two-field aggregate the borrow
/// builtins build (`refcell_guard`), and core's real layout,
/// whose `RefMut` form carries a trailing `PhantomData` marker
/// field past the two reference fields. The marker holds no
/// borrow state, so fields past the first two are ignored.
pub(crate) fn refcell_guard_fields(guard: &Value) -> Option<&[Value]> {
    match guard {
        Value::Enum { fields, .. } if fields.len() >= 2 => Some(fields),
        _ => None,
    }
}

/// The reference a `Ref` / `RefMut` guard field holds, peeling
/// the single-field wrapper aggregates core's real layout stores
/// it under: the guard the borrow builtins build holds each
/// reference bare, but a guard a converted guest body built from
/// core's own definitions wraps the value reference in
/// `NonNull<T>` and the borrow-flag reference in
/// `BorrowRef` / `BorrowRefMut`, each a one-field struct in this
/// value model. (The fmt engine applies the same descent to a
/// guard's first field.)
pub(crate) fn refcell_guard_ref(field: &Value) -> Option<&RefTarget> {
    let mut cur = field;
    loop {
        match cur {
            Value::Ref(target) => return Some(target),
            Value::Enum { fields, .. } if fields.len() == 1 => cur = &fields[0],
            _ => return None,
        }
    }
}

/// Extend a `RefCell` place target with a field path inside the
/// cell. Only place targets can be extended; guard construction
/// starts from `Interpreter::refcell_place`, which has already
/// rejected the rest.
pub(crate) fn refcell_sub_target(cell: &RefTarget, path: &[RefProj]) -> RefTarget {
    match cell {
        RefTarget::Place {
            frame,
            local,
            projection,
        } => {
            let mut projection = projection.clone();
            projection.extend_from_slice(path);
            RefTarget::Place {
                frame: *frame,
                local: *local,
                projection,
            }
        }
        RefTarget::Static { name, projection } => {
            let mut projection = projection.clone();
            projection.extend_from_slice(path);
            RefTarget::Static {
                name: name.clone(),
                projection,
            }
        }
        RefTarget::Alloc { id, projection } => {
            let mut projection = projection.clone();
            projection.extend_from_slice(path);
            RefTarget::Alloc {
                id: *id,
                projection,
            }
        }
        // Unreachable through `refcell_place`; keep the target as-is
        // rather than panicking in the model.
        other => other.clone(),
    }
}
