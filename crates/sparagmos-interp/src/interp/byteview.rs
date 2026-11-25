//! Byte-level views over scalar storage.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};

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
            Ty::Adt(name) => ByteLayout::Adt(name.clone()),
            _ => return None,
        })
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
        if self.program().adt_field_layouts.contains_key(tag) {
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
    /// recorded.
    fn projection_storage_offset(
        &self,
        base_ty: &Ty,
        prefix: &[RefProj],
    ) -> Option<(usize, Option<Ty>)> {
        let mut ty: Option<Ty> = Some(base_ty.clone());
        let mut off: usize = 0;
        let mut variant: usize = 0;
        for step in prefix {
            match step {
                RefProj::Field(i) => match &ty {
                    Some(Ty::Adt(name)) => {
                        let rows = self.program().adt_field_layouts.get(name)?;
                        let row = rows.get(variant)?;
                        let &(field_off, _) = row.get(*i)?;
                        off = off.checked_add(usize::try_from(field_off).ok()?)?;
                        ty = self
                            .adt_field_tag(name, variant, *i)
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
                RefProj::Subslice { offset, len } => match &ty {
                    Some(Ty::Array(elem, _)) => {
                        let elem_size = usize::try_from(self.ty_size(elem)?).ok()?;
                        off = off.checked_add(offset.checked_mul(elem_size)?)?;
                        ty = Some(Ty::Array(elem.clone(), u64::try_from(*len).ok()?));
                    }
                    _ => return None,
                },
                _ => return None,
            }
        }
        Some((off, ty))
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
        let (off, _) = self.projection_storage_offset(base_ty, &projection[..at])?;
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
                let tag = self.adt_field_tag(name, *variant as usize, f)?;
                let field_layout = self.byte_layout_of_tag(&tag)?;
                if self.byte_layout_size(&field_layout)? != row[f].1 as usize {
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
                let tag = self.adt_field_tag(name, variant as usize, f)?;
                let field_layout = self.byte_layout_of_tag(&tag)?;
                if self.byte_layout_size(&field_layout)? != row[f].1 as usize {
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
        let (pos, _) = self.projection_storage_offset(&base_ty, projection)?;
        let position_value =
            project_value(base.clone(), projection, &func, &place_desc).ok()?;
        if self.flatten_scalar(&position_value, shape).is_ok() {
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

    /// The pointee a format argument of scalar type `ty`
    /// (the argument's rendered type tag) denotes through a
    /// byte-displaced target: the tag is the load's width, so
    /// the byte view composes the value exactly (see
    /// [`Self::byte_view_shaped_load`]). `None` when the tag
    /// names no scalar shape, the target is not a reference to
    /// a judged position, or the view does not judge it.
    pub(crate) fn byte_view_fmt_arg(
        &self,
        frames: &[Frame],
        target: &Value,
        ty: &str,
    ) -> Option<InterpResult<Value>> {
        let shape = match ty.trim() {
            "bool" => ScalarShape::Bool,
            "char" => ScalarShape::Char,
            "i8" => ScalarShape::Int { bits: 8, signed: true },
            "i16" => ScalarShape::Int { bits: 16, signed: true },
            "i32" => ScalarShape::Int { bits: 32, signed: true },
            "i64" => ScalarShape::Int { bits: 64, signed: true },
            "i128" => ScalarShape::Int { bits: 128, signed: true },
            "isize" => ScalarShape::Int { bits: 64, signed: true },
            "u8" => ScalarShape::Int { bits: 8, signed: false },
            "u16" => ScalarShape::Int { bits: 16, signed: false },
            "u32" => ScalarShape::Int { bits: 32, signed: false },
            "u64" => ScalarShape::Int { bits: 64, signed: false },
            "u128" => ScalarShape::Int { bits: 128, signed: false },
            "usize" => ScalarShape::Int { bits: 64, signed: false },
            "f16" => ScalarShape::Float { bits: 16 },
            "f32" => ScalarShape::Float { bits: 32 },
            "f64" => ScalarShape::Float { bits: 64 },
            "f128" => ScalarShape::Float { bits: 128 },
            _ => return None,
        };
        let resolved = self.deref_value(frames, target).ok()?;
        self.byte_view_shaped_load(frames, &resolved, &shape)
            .or_else(|| self.byte_view_compose(frames, &resolved, &shape))
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
            let (off, pos_ty) = self.projection_storage_offset(&base_ty, projection)?;
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
        let Some((_, Some(Ty::Adt(name)))) =
            self.projection_storage_offset(&base_ty, projection)
        else {
            return false;
        };
        if !self.program().adt_field_layouts.contains_key(&name) {
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
        if !projection.is_empty() {
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
        if !projection.is_empty() {
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
        if !projection.is_empty() {
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
            if projection.is_empty() {
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
        if !projection.is_empty() {
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
    /// same way; a target whose projection has no trailing index
    /// step takes the plain access.
    pub(crate) fn seq_element_target(
        &self,
        frames: &[Frame],
        target: &Resolved,
        index: usize,
        _elem: Option<&Ty>,
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
        let Some((&RefProj::Index(start), head)) = projection.split_last() else {
            return Ok(None);
        };
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
        projection.push(RefProj::Index(start + index));
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
        let flag_ref = match guard {
            Value::Enum { fields, .. } if fields.len() == 2 => &fields[1],
            other => {
                return Err(InterpError::type_mismatch(
                    "RefCell guard drop",
                    "Ref/RefMut guard aggregate",
                    other.kind_name(),
                ))
            }
        };
        let flag_target = match flag_ref {
            Value::Ref(target) => target.clone(),
            other => {
                return Err(InterpError::type_mismatch(
                    "RefCell guard drop",
                    "reference to the borrow-flag Cell",
                    other.kind_name(),
                ))
            }
        };
        let flag = self.refcell_flag(frames, &flag_target)?;
        let next = if mutable { 0 } else { flag - 1 };
        self.refcell_set_flag(frames, &flag_target, next)
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
