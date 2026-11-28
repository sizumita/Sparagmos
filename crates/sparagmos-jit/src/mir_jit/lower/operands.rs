use sparagmos_mir as mir;

use crate::ir::{BinOp as IrBinOp, Instr, Operand as IrOperand};
use crate::program::{StaticBlob, StaticWord};

use super::*;

impl<'a> FnLower<'a> {
    pub(super) fn place_ty(&self, place: &mir::Place) -> Result<mir::Ty, JitError> {
        Ok(match self.classify_place(place)? {
            PlaceLoc::Slots { ty, .. } => ty,
            PlaceLoc::Mem { ty, .. } => ty,
        })
    }


    /// Best-effort type of an operand: from its place, when it has one.
    pub(super) fn operand_place_ty(&self, op: &mir::Operand) -> Option<mir::Ty> {
        op.place().and_then(|p| self.place_ty(p).ok())
    }

    /// The float type of an operand, from its place or — for a float
    /// constant — from the constant itself. Casts need the source
    /// type even when the operand has no place.
    pub(super) fn operand_float_ty(&self, op: &mir::Operand) -> Option<mir::Ty> {
        if let Some(t) = self.operand_place_ty(op) {
            if float_kind(&t).is_some() {
                return Some(t);
            }
        }
        if let mir::Operand::Constant(c) = op {
            if let mir::ConstKind::Value(mir::ConstValue::Scalar(mir::Scalar::Float {
                ty,
                ..
            })) = &c.literal
            {
                if matches!(ty, mir::FloatTy::F32 | mir::FloatTy::F64) {
                    return Some(mir::Ty::Float(*ty));
                }
            }
        }
        None
    }

    /// Materialise an operand as a single IR value in canonical form
    /// for `ctx`. Multi-leaf operands are rejected here; whole-value
    /// copies are handled by the `Use` path.
    pub(super) fn operand_value(&mut self, op: &mir::Operand, ctx: &mir::Ty) -> Result<u32, JitError> {
        match op {
            mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                let loc = self.classify_place(p)?;
                match &loc {
                    PlaceLoc::Slots { first, count, .. } => {
                        if *count != 1 {
                            return Err(
                                self.err("operand of aggregate type in scalar position")
                            );
                        }
                        Ok(*first)
                    }
                    PlaceLoc::Mem { count, .. } => {
                        if *count != Some(1) {
                            return Err(
                                self.err("operand of aggregate type in scalar position")
                            );
                        }
                        // The word at the address is the pointee's
                        // canonical form, exactly as if it had been
                        // read from its own slot.
                        let dst = self.fresh();
                        self.load_mem_to(&loc, dst);
                        Ok(dst)
                    }
                }
            }
            mir::Operand::Constant(c) => {
                if let Some(slot) = self.const_data_addr(c, ctx) {
                    return Ok(slot);
                }
                if let mir::ConstKind::Value(v) = &c.literal {
                    if is_pointer_shaped(v) {
                        // A pointer-shaped constant in a single-word
                        // position is a thin pointer (a reference to
                        // a scalar constant, a byte string used as a
                        // raw pointer): materialize its words and
                        // take the first — the address.
                        let pointee = match ctx {
                            mir::Ty::Ref(_, p, _) | mir::Ty::RawPtr(p, _) => Some(&**p),
                            _ => None,
                        };
                        let words = self.pointer_const_words(v, pointee, 1)?;
                        let dst = self.fresh();
                        self.emit_static_words(&words, dst);
                        return Ok(dst);
                    }
                }
                let v = self.const_i64(c, ctx)?;
                let dst = self.fresh();
                self.emit(Instr::Const { dst, imm: v });
                Ok(dst)
            }
        }
    }

    /// Materialise a two-word operand — a fat pointer value — as
    /// (address word, length word) value slots.
    pub(super) fn operand_pair(&mut self, op: &mir::Operand) -> Result<(u32, u32), JitError> {
        match op {
            mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                let loc = self.classify_place(p)?;
                match &loc {
                    PlaceLoc::Slots { first, count, .. } => {
                        if *count != 2 {
                            return Err(
                                self.err("operand of non-fat type in fat position")
                            );
                        }
                        Ok((*first, *first + 1))
                    }
                    PlaceLoc::Mem { count, .. } => {
                        if *count != Some(2) {
                            return Err(
                                self.err("operand of non-fat type in fat position")
                            );
                        }
                        let a = self.fresh();
                        let b = self.fresh();
                        debug_assert_eq!(b, a + 1);
                        self.load_mem_to(&loc, a);
                        Ok((a, b))
                    }
                }
            }
            mir::Operand::Constant(c) => match &c.literal {
                mir::ConstKind::Value(v) if is_pointer_shaped(v) => {
                    // A fat pointer constant: the address of its
                    // pointee blob and its metadata word.
                    let words = self.pointer_const_words(v, None, 2)?;
                    let a = self.fresh();
                    let b = self.fresh();
                    debug_assert_eq!(b, a + 1);
                    self.emit_static_words(&words, a);
                    Ok((a, b))
                }
                _ => Err(self.err("fat pointer constant")),
            },
        }
    }


    /// Write an operand's leaves — `flatten(ctx)` of them — into
    /// the consecutive slots starting at `dst`. The operand may
    /// live in the frame, in memory, or be a constant (materialised
    /// leaf by leaf). Used for aggregate fields and whole-value
    /// copies of multi-leaf values.
    pub(super) fn store_operand_leaves(
        &mut self,
        op: &mir::Operand,
        ctx: &mir::Ty,
        dst: u32,
    ) -> Result<(), JitError> {
        let count = flatten(ctx, &self.func.name, self.layouts)?;
        match op {
            mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                let loc = self.classify_place(p)?;
                match &loc {
                    PlaceLoc::Slots { first, count: c, .. } => {
                        if *c != count {
                            return Err(self.err("operand shape mismatch"));
                        }
                        for k in 0..count as u32 {
                            self.emit(Instr::Copy {
                                dst: dst + k,
                                src: IrOperand::Value(first + k),
                            });
                        }
                        Ok(())
                    }
                    PlaceLoc::Mem { count: c, .. } => {
                        if *c != Some(count) {
                            return Err(self.err("operand shape mismatch"));
                        }
                        self.load_mem_to(&loc, dst);
                        Ok(())
                    }
                }
            }
            mir::Operand::Constant(c) => match &c.literal {
                mir::ConstKind::Value(v) => self.const_value_leaves(v, ctx, dst),
                _ => Err(self.err("non-value constant in aggregate position")),
            },
        }
    }

    /// The leaves of an operand as value slots (a fresh scratch
    /// run for a constant or a memory value; the place's own slots
    /// for a frame value). `count` must equal the operand type's
    /// leaf count.
    pub(super) fn operand_leaves(
        &mut self,
        op: &mir::Operand,
        ctx: &mir::Ty,
        count: usize,
    ) -> Result<Vec<u32>, JitError> {
        match op {
            mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                let loc = self.classify_place(p)?;
                match &loc {
                    PlaceLoc::Slots { first, count: c, .. } => {
                        if *c != count {
                            return Err(self.err("operand shape mismatch"));
                        }
                        Ok((0..count as u32).map(|k| first + k).collect())
                    }
                    PlaceLoc::Mem { count: c, .. } => {
                        if *c != Some(count) {
                            return Err(self.err("operand shape mismatch"));
                        }
                        let base = self.fresh();
                        for _ in 1..count {
                            self.fresh();
                        }
                        self.load_mem_to(&loc, base);
                        Ok((0..count as u32).map(|k| base + k).collect())
                    }
                }
            }
            mir::Operand::Constant(_) => {
                let base = self.fresh();
                for _ in 1..count {
                    self.fresh();
                }
                self.store_operand_leaves(op, ctx, base)?;
                Ok((0..count as u32).map(|k| base + k).collect())
            }
        }
    }

    /// Materialise a constant value's leaves into the consecutive
    /// slots starting at `dst`, under the type context `ctx`:
    /// scalars are one leaf (their canonical word), tuples are
    /// their fields in order, an ADT constant is its tag (when the
    /// layout has one) plus each field in its layout slot.
    pub(super) fn const_value_leaves(
        &mut self,
        v: &mir::ConstValue,
        ctx: &mir::Ty,
        dst: u32,
    ) -> Result<(), JitError> {
        // A byte-array constant in reference position: its
        // address is the statics blob in guest-array layout, on
        // this path exactly as on the operand paths (see
        // `byte_array_blob_index`) — a value's address cannot
        // depend on which lowering path materialised it.
        if let mir::ConstValue::Bytes(b) = v {
            if let Some(index) = self.byte_array_blob_index(b, ctx) {
                self.emit(Instr::StaticAddr { dst, index });
                return Ok(());
            }
        }
        match (v, ctx) {
            (mir::ConstValue::Adt { variant, fields }, mir::Ty::Adt(def, _)) => {
                let layout = self.adt_layout(ctx)?;
                let slots = layout
                    .variants
                    .get(*variant as usize)
                    .ok_or_else(|| self.err("ADT constant variant outside the layout"))?;
                if slots.len() != fields.len() {
                    return Err(self.err("ADT constant field count mismatch"));
                }
                if layout.tag {
                    let tag = self.discriminant_value(*def, *variant)?;
                    self.emit(Instr::Const { dst, imm: tag });
                }
                for (field, slot) in fields.iter().zip(slots.iter()) {
                    let fctx = slot.ty.clone();
                    self.const_value_leaves(field, &fctx, dst + slot.offset as u32)?;
                }
                Ok(())
            }
            (mir::ConstValue::Tuple(fields), mir::Ty::Tuple(tys)) => {
                if fields.len() != tys.len() {
                    return Err(self.err("tuple constant field count mismatch"));
                }
                let mut off = 0usize;
                for (field, ty) in fields.iter().zip(tys.iter()) {
                    self.const_value_leaves(field, ty, dst + off as u32)?;
                    off += flatten(ty, &self.func.name, self.layouts)?;
                }
                Ok(())
            }
            _ if is_pointer_shaped(v) => {
                // A pointer-shaped constant (a string, a slice, a
                // reference to another constant): its leaves are
                // the address of its pointee blob plus its metadata
                // words.
                let width = flatten(ctx, &self.func.name, self.layouts)?;
                let pointee = match ctx {
                    mir::Ty::Ref(_, p, _) | mir::Ty::RawPtr(p, _) => Some(&**p),
                    _ => None,
                };
                let words = self.pointer_const_words(v, pointee, width)?;
                self.emit_static_words(&words, dst);
                Ok(())
            }
            _ => {
                if flatten(ctx, &self.func.name, self.layouts)? == 0 {
                    // A zero-leaf value (unit / never) stores nothing.
                    return Ok(());
                }
                if int128_signed(ctx).is_some() {
                    // A 128-bit scalar constant is the word pair of
                    // its full-width pattern, low word first.
                    let w = self.const_value_u128(v, ctx)?;
                    self.emit(Instr::Const {
                        dst,
                        imm: w as u64 as i64,
                    });
                    self.emit(Instr::Const {
                        dst: dst + 1,
                        imm: (w >> 64) as u64 as i64,
                    });
                    return Ok(());
                }
                let w = self.const_value_i64(v, ctx)?;
                self.emit(Instr::Const { dst, imm: w });
                Ok(())
            }
        }
    }

    /// Emit static words into consecutive slots starting at `dst`:
    /// an immediate word is a constant, an address word is the
    /// address of the interned static blob.
    fn emit_static_words(&mut self, words: &[StaticWord], dst: u32) {
        for (k, w) in words.iter().enumerate() {
            match w {
                StaticWord::Imm(x) => self.emit(Instr::Const {
                    dst: dst + k as u32,
                    imm: *x,
                }),
                StaticWord::AddrOf(idx) => self.emit(Instr::StaticAddr {
                    dst: dst + k as u32,
                    index: *idx,
                }),
            }
        }
    }

    /// The leaf words of a pointer-shaped constant, as static
    /// words: the address of the constant's pointee blob plus the
    /// metadata word(s) of the value's shape. `pointee` is the
    /// pointee type when the caller knows it (a slice-value
    /// constant needs its element type, a reference constant its
    /// pointee's layout); `width` is the constant value's leaf
    /// count in its position.
    fn pointer_const_words(
        &mut self,
        v: &mir::ConstValue,
        pointee: Option<&mir::Ty>,
        width: usize,
    ) -> Result<Vec<StaticWord>, JitError> {
        use mir::ConstValue as V;
        match v {
            V::Str(s) => {
                if width != 2 {
                    return Err(self.err("string constant in a non-pointer position"));
                }
                let idx = self
                    .statics
                    .intern(StaticBlob::Bytes(s.as_bytes().to_vec()));
                Ok(vec![
                    StaticWord::AddrOf(idx),
                    StaticWord::Imm(s.len() as i64),
                ])
            }
            V::Bytes(b) => {
                let idx = self.statics.intern(StaticBlob::Bytes(b.clone()));
                match width {
                    1 => Ok(vec![StaticWord::AddrOf(idx)]),
                    2 => Ok(vec![
                        StaticWord::AddrOf(idx),
                        StaticWord::Imm(b.len() as i64),
                    ]),
                    _ => Err(self.err("byte-string constant in an unexpected position")),
                }
            }
            V::Slice { data, meta } => {
                if width != 2 {
                    return Err(self.err("slice constant in a non-pointer position"));
                }
                let idx = self.statics.intern(StaticBlob::Bytes(data.clone()));
                Ok(vec![
                    StaticWord::AddrOf(idx),
                    StaticWord::Imm(*meta as i64),
                ])
            }
            V::SliceVal(elems) => {
                if width != 2 {
                    return Err(self.err("slice-value constant in a non-pointer position"));
                }
                let Some(mir::Ty::Slice(elem)) = pointee else {
                    return Err(self.err("slice-value constant without a slice type"));
                };
                let elem = (**elem).clone();
                let words = self.const_blob_words(elems, &elem)?;
                let idx = self.statics.intern(StaticBlob::Words(words));
                Ok(vec![
                    StaticWord::AddrOf(idx),
                    StaticWord::Imm(elems.len() as i64),
                ])
            }
            V::Ref(inner) => {
                if is_pointer_shaped(inner) {
                    if width == 1 {
                        // A thin reference to a fat constant (a
                        // `&&str` literal): the value is the
                        // address of a static blob holding the
                        // constant's own pointer words, in the
                        // descending layout every guest value
                        // uses (a reference to a reference nests
                        // the same way, one blob per level).
                        let inner_width = match &**inner {
                            V::Str(_) | V::Slice { .. } | V::SliceVal(_) => 2,
                            _ => 1,
                        };
                        let words =
                            self.pointer_const_words(inner, pointee, inner_width)?;
                        let idx = self.statics.intern(StaticBlob::Words(words));
                        return Ok(vec![StaticWord::AddrOf(idx)]);
                    }
                    // A reference to a pointer-shaped constant is
                    // the constant's own pointer value.
                    return self.pointer_const_words(inner, pointee, width);
                }
                let Some(pty) = pointee else {
                    return Err(self.err("reference constant without a pointee type"));
                };
                if width != 1 {
                    return Err(self.err("fat reference constant to a value"));
                }
                let pty = pty.clone();
                let words = self.const_static_words(inner, &pty)?;
                let idx = self.statics.intern(StaticBlob::Words(words));
                Ok(vec![StaticWord::AddrOf(idx)])
            }
            _ => Err(self.err("unsupported pointer constant")),
        }
    }

    /// The static words of a constant value laid out in a blob: a
    /// scalar is one immediate word, a tuple / array / ADT is its
    /// leaves in order (an ADT prefixed by its discriminant word),
    /// a pointer-shaped value is its pointer words.
    fn const_static_words(
        &mut self,
        v: &mir::ConstValue,
        ty: &mir::Ty,
    ) -> Result<Vec<StaticWord>, JitError> {
        use mir::ConstValue as V;
        match v {
            V::Tuple(fields) => {
                let mut out = Vec::new();
                match ty {
                    mir::Ty::Tuple(tys) => {
                        if fields.len() != tys.len() {
                            return Err(self.err("tuple constant arity mismatch"));
                        }
                        for (f, t) in fields.iter().zip(tys.iter()) {
                            out.extend(self.const_static_words(f, t)?);
                        }
                    }
                    // A promoted array constant is a tuple of its
                    // elements.
                    mir::Ty::Array(elem, mir::ArrayLen::Known(n)) => {
                        if fields.len() != *n as usize {
                            return Err(self.err("array constant arity mismatch"));
                        }
                        let elem = (**elem).clone();
                        for f in fields {
                            out.extend(self.const_static_words(f, &elem)?);
                        }
                    }
                    _ => return Err(self.err("tuple constant without a tuple type")),
                }
                Ok(out)
            }
            V::Adt { variant, fields } => {
                let mir::Ty::Adt(def, _) = ty else {
                    return Err(self.err("ADT constant without an ADT type"));
                };
                let layout = self.adt_layout(ty)?;
                let slots = layout
                    .variants
                    .get(*variant as usize)
                    .ok_or_else(|| self.err("ADT constant variant outside the layout"))?
                    .clone();
                if slots.len() != fields.len() {
                    return Err(self.err("ADT constant field count mismatch"));
                }
                let mut out = vec![StaticWord::Imm(0); layout.total];
                if layout.tag {
                    out[0] = StaticWord::Imm(self.discriminant_value(*def, *variant)?);
                }
                for (field, slot) in fields.iter().zip(slots.iter()) {
                    let fty = slot.ty.clone();
                    let words = self.const_static_words(field, &fty)?;
                    for (k, w) in words.into_iter().enumerate() {
                        out[slot.offset + k] = w;
                    }
                }
                Ok(out)
            }
            _ if is_pointer_shaped(v) => {
                let width = flatten(ty, &self.func.name, self.layouts)?;
                let pointee = match ty {
                    mir::Ty::Ref(_, p, _) | mir::Ty::RawPtr(p, _) => Some(&**p),
                    _ => None,
                };
                self.pointer_const_words(v, pointee, width)
            }
            _ if int128_signed(ty).is_some() => {
                let w = self.const_value_u128(v, ty)?;
                Ok(vec![
                    StaticWord::Imm(w as u64 as i64),
                    StaticWord::Imm((w >> 64) as u64 as i64),
                ])
            }
            _ => Ok(vec![StaticWord::Imm(self.const_value_i64(v, ty)?)]),
        }
    }

    /// The static words of a sequence of same-typed constant
    /// values (a slice-value constant's elements).
    fn const_blob_words(
        &mut self,
        elems: &[mir::ConstValue],
        elem_ty: &mir::Ty,
    ) -> Result<Vec<StaticWord>, JitError> {
        let mut out = Vec::new();
        for e in elems {
            out.extend(self.const_static_words(e, elem_ty)?);
        }
        Ok(out)
    }

    /// Interpret a constant under the type context it is used in.
    pub(super) fn const_i64(&self, c: &mir::Constant, ctx: &mir::Ty) -> Result<i64, JitError> {
        match &c.literal {
            mir::ConstKind::Value(v) => self.const_value_i64(v, ctx),
            mir::ConstKind::Ty(t) => Err(self.err(&format!(
                "type constant {} outside call position",
                t.kind_name()
            ))),
            mir::ConstKind::Unevaluated { def, .. } => {
                Err(self.err(&format!("unevaluated constant {def}")))
            }
        }
    }

    /// The full 128-bit pattern of a constant under a 128-bit
    /// integer context: the scalar's own stored pattern, masked to
    /// the context width (which is the full width here). Mirrors
    /// [`Self::const_value_i64`]'s treatment of the scalar forms.
    fn const_value_u128(
        &self,
        v: &mir::ConstValue,
        _ctx: &mir::Ty,
    ) -> Result<u128, JitError> {
        Ok(match v {
            mir::ConstValue::Bool(b) => *b as u128,
            mir::ConstValue::Char(ch) => *ch as u128,
            mir::ConstValue::ZeroSized => 0,
            mir::ConstValue::Scalar(mir::Scalar::Int { data, .. }) => *data,
            _ => {
                return Err(self.err("constant without a 128-bit integer pattern"));
            }
        })
    }

    fn const_value_i64(&self, v: &mir::ConstValue, ctx: &mir::Ty) -> Result<i64, JitError> {
        Ok(match v {
            mir::ConstValue::Bool(b) => *b as i64,
            mir::ConstValue::Char(ch) => *ch as i64,
            mir::ConstValue::ZeroSized => 0,
            mir::ConstValue::Scalar(mir::Scalar::Int { data, size }) => {
                // `data` holds the value in `size` bytes; normalise it
                // to the context type's canonical form.
                match int_kind(ctx) {
                    Some(k) => {
                        let bits = k.bits as u32;
                        let masked = if bits >= 128 {
                            *data
                        } else {
                            data & ((1u128 << bits) - 1)
                        };
                        if k.signed && bits < 128 {
                            let shift = 128 - bits;
                            ((masked << shift) as i128 >> shift) as i64
                        } else {
                            masked as i64
                        }
                    }
                    None => {
                        // No integer context: fall back to the
                        // scalar's own size, read as unsigned.
                        let bits = (*size as u32) * 8;
                        if bits >= 128 {
                            *data as i64
                        } else {
                            (data & ((1u128 << bits) - 1)) as i64
                        }
                    }
                }
            }
            mir::ConstValue::Scalar(mir::Scalar::Float { bits, ty }) => {
                // A float value's slot word is its bit pattern: the
                // full 64 bits for f64, the low 32 (zero-extended)
                // for f32.
                match ty {
                    mir::FloatTy::F64 => *bits as u64 as i64,
                    mir::FloatTy::F32 => *bits as u32 as i64,
                    other => {
                        return Err(self.err(&format!("{} constant", other.name())));
                    }
                }
            }
            mir::ConstValue::Scalar(mir::Scalar::Ptr { .. }) => {
                return Err(self.err("pointer constant"));
            }
            mir::ConstValue::FnDef(_) => {
                return Err(self.err("function pointer constant"));
            }
            mir::ConstValue::Closure(..) => {
                return Err(self.err("closure constant"));
            }
            mir::ConstValue::Str(_) => return Err(self.err("string constant")),
            mir::ConstValue::Bytes(_) => return Err(self.err("byte-string constant")),
            mir::ConstValue::Slice { .. } => return Err(self.err("slice constant")),
            mir::ConstValue::ByRef { .. } => {
                return Err(self.err("by-reference constant"));
            }
            mir::ConstValue::ErasedAlloc { .. } => {
                return Err(self.err("erased-allocation constant"));
            }
            mir::ConstValue::IntPtr(_) => {
                return Err(self.err("address pointer constant"));
            }
            mir::ConstValue::Tuple(_) => {
                return Err(self.err("tuple constant"));
            }
            mir::ConstValue::Adt { variant, fields } => {
                if !fields.is_empty() {
                    return Err(self.err("ADT constant with fields"));
                }
                let mir::Ty::Adt(def, _) = ctx else {
                    return Err(self.err("ADT constant without an ADT type context"));
                };
                // A fieldless variant constant is its tag alone.
                self.discriminant_value(*def, *variant)?
            }
            mir::ConstValue::Ref(_) => {
                return Err(self.err("reference constant"));
            }
            mir::ConstValue::NullPtr => {
                return Err(self.err("null pointer constant"));
            }
            mir::ConstValue::Uninit => {
                return Err(self.err("uninitialized constant"));
            }
            mir::ConstValue::SliceVal(_) => {
                return Err(self.err("slice-value constant"));
            }
            mir::ConstValue::StaticRef { .. } => {
                return Err(self.err("static reference constant"));
            }
            mir::ConstValue::VTablePtr(_) => {
                return Err(self.err("vtable pointer constant"));
            }
            mir::ConstValue::FatPtr { .. } => {
                return Err(self.err("fat pointer constant"));
            }
            mir::ConstValue::DynRef { .. } => {
                return Err(self.err("dyn reference constant"));
            }
        })
    }

    /// Emit the normalisation that re-establishes canonical form for
    /// `ty` on the value in slot `slot` (in place).
    pub(super) fn normalize(&mut self, slot: u32, ty: &mir::Ty) {
        let Some(k) = int_kind(ty) else { return };
        let bits = k.bits as u32;
        if bits >= 64 {
            return;
        }
        if k.signed {
            let shift = 64 - bits;
            self.emit(Instr::BinOp {
                dst: slot,
                op: IrBinOp::Shl,
                lhs: IrOperand::Value(slot),
                rhs: IrOperand::Const(shift as i64),
            });
            self.emit(Instr::BinOp {
                dst: slot,
                op: IrBinOp::Sar,
                lhs: IrOperand::Value(slot),
                rhs: IrOperand::Const(shift as i64),
            });
        } else {
            let mask = ((1u64 << bits) - 1) as i64;
            self.emit(Instr::BinOp {
                dst: slot,
                op: IrBinOp::And,
                lhs: IrOperand::Value(slot),
                rhs: IrOperand::Const(mask),
            });
        }
    }

    // ----- statements --------------------------------------------------------

}
