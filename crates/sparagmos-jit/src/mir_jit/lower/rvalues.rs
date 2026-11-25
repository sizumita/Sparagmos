use sparagmos_mir as mir;

use crate::ir::{Instr, Operand as IrOperand};

use super::*;

impl<'a> FnLower<'a> {
    pub(super) fn lower_statement(&mut self, stmt: &mir::StatementKind) -> Result<(), JitError> {
        use mir::StatementKind as S;
        match stmt {
            S::Assign(pair) => {
                let (place, rvalue) = &**pair;
                self.lower_assign(place, rvalue)
            }
            // Statements with no runtime effect on values.
            S::StorageLive(_)
            | S::StorageDead(_)
            | S::Nop
            | S::FakeRead(_)
            | S::PlaceMention(_)
            | S::Retag(..)
            | S::AscribeUserType(..)
            | S::Coverage(..)
            | S::ConstEvalCounter => Ok(()),
            S::SetDiscriminant {
                place,
                variant_index,
            } => {
                let loc = self.classify_place(place)?;
                if let PlaceLoc::Mem { ty, .. } = &loc {
                    let mir::Ty::Adt(def, _) = ty else {
                        return Err(self.err("SetDiscriminant on a non-ADT place"));
                    };
                    let layout = self.adt_layout(ty)?;
                    // Validates the variant against the table.
                    let tag = self.discriminant_value(*def, *variant_index)?;
                    if layout.tag {
                        // The value's first leaf in memory is its
                        // tag: store just that leaf.
                        let mut tag_loc = loc.clone();
                        if let PlaceLoc::Mem { count, .. } = &mut tag_loc {
                            *count = Some(1);
                        }
                        let tmp = self.fresh();
                        self.emit(Instr::Const { dst: tmp, imm: tag });
                        self.store_mem_from(&tag_loc, tmp);
                    }
                    return Ok(());
                }
                let (dst, _, ty) = self.place_leaves(place)?;
                let mir::Ty::Adt(def, _) = &ty else {
                    return Err(self.err("SetDiscriminant on a non-ADT place"));
                };
                let layout = self.adt_layout(&ty)?;
                // Validates the variant against the table.
                let tag = self.discriminant_value(*def, *variant_index)?;
                if layout.tag {
                    // The place's first leaf is its tag: write the
                    // variant's discriminant value from the table.
                    self.emit(Instr::Const { dst, imm: tag });
                }
                // A single-variant ADT has no tag leaf: its
                // discriminant is implicit, and setting it (to the
                // only variant) writes nothing.
                Ok(())
            }
            S::Intrinsic(intr) => match &**intr {
                // `assume` is an optimisation hint: the JIT emits
                // nothing for it. (On a false operand the guest has
                // undefined behaviour; the interpreter reports it as
                // a guest panic, the JIT simply proceeds — both are
                // legitimate readings of UB.)
                mir::NonDivergingIntrinsic::Assume(_) => Ok(()),
                mir::NonDivergingIntrinsic::CopyNonOverlapping { .. } => {
                    Err(self.err("Intrinsic statement CopyNonOverlapping"))
                }
            },
        }
    }

    /// The value of a fresh thin reference to `target`: the stack
    /// address of the target place's first leaf, computed from the
    /// place's resolution (for a target itself in memory — a field
    /// of a pointee, a slice element — the materialised address).
    fn ref_value(&mut self, target: &mir::Place) -> Result<u32, JitError> {
        let loc = self.classify_place(target)?;
        match &loc {
            PlaceLoc::Slots { first, .. } => {
                let dst = self.fresh();
                self.emit(Instr::SlotAddr {
                    dst,
                    slot: *first,
                });
                Ok(dst)
            }
            PlaceLoc::Mem { addr, checks, .. } => {
                self.emit_checks(checks);
                Ok(self.materialize_addr(addr))
            }
        }
    }

    /// The word(s) of a fresh reference of type `ref_ty` to
    /// `target`: one word (the address) for a thin reference, two
    /// (address, length) for a fat one. A fat reference's target is
    /// an array place (its length is the array's), or a slice /
    /// `str` place reached through a dereference (its length is
    /// the source fat pointer's).
    fn ref_words(
        &mut self,
        target: &mir::Place,
        ref_ty: &mir::Ty,
    ) -> Result<Vec<u32>, JitError> {
        let pointee = match ref_ty {
            mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _) => (**pointee).clone(),
            _ => {
                return Err(self.err(&format!(
                    "reference creation with non-pointer type {}",
                    ref_ty.kind_name()
                )));
            }
        };
        if !matches!(pointee, mir::Ty::Slice(_) | mir::Ty::Str) {
            return Ok(vec![self.ref_value(target)?]);
        }
        let loc = self.classify_place(target)?;
        let (addr_v, len_v) = match &loc {
            PlaceLoc::Slots { first, ty, .. } => {
                let mir::Ty::Array(_, mir::ArrayLen::Known(n)) = ty else {
                    return Err(self.err("fat reference to a non-array place"));
                };
                let n = *n;
                let d = self.fresh();
                self.emit(Instr::SlotAddr {
                    dst: d,
                    slot: *first,
                });
                (d, self.len_value(&LenSrc::Const(n)))
            }
            PlaceLoc::Mem { ty, len, addr, checks, .. } => {
                let len_src = match ty {
                    mir::Ty::Slice(_) | mir::Ty::Str => len.clone().ok_or_else(|| {
                        self.err("fat reference to a slice place without a length")
                    })?,
                    mir::Ty::Array(_, mir::ArrayLen::Known(n)) => LenSrc::Const(*n),
                    _ => return Err(self.err("fat reference to a non-array place")),
                };
                self.emit_checks(checks);
                let a = self.materialize_addr(addr);
                (a, self.len_value(&len_src))
            }
        };
        Ok(vec![addr_v, len_v])
    }

    /// `place = &target` / `place = &raw target`: the reference
    /// value is the target's address (and, for a fat reference,
    /// its length), as in [`Self::ref_words`].
    fn lower_ref_creation(
        &mut self,
        place: &mir::Place,
        target: &mir::Place,
    ) -> Result<(), JitError> {
        let (dst, count, dst_ty) = self.place_leaves(place)?;
        let words = self.ref_words(target, &dst_ty)?;
        if words.len() != count {
            return Err(self.err("reference value shape mismatch"));
        }
        for (k, w) in words.iter().enumerate() {
            self.emit(Instr::Copy {
                dst: dst + k as u32,
                src: IrOperand::Value(*w),
            });
        }
        Ok(())
    }




    /// The word(s) of an rvalue being assigned into a memory
    /// place, computed into scratch slots with the ordinary rvalue
    /// lowerings. Only the rvalue shapes the slot model can hold
    /// are accepted; anything else is an explicit error.
    fn rvalue_words(
        &mut self,
        rvalue: &mir::Rvalue,
        dst_ty: &mir::Ty,
        count: usize,
    ) -> Result<Vec<u32>, JitError> {
        use mir::Rvalue as R;
        match rvalue {
            R::Use(op) => match count {
                1 => Ok(vec![self.operand_value(op, dst_ty)?]),
                2 if int128_signed(dst_ty).is_some() => {
                    self.operand_leaves(op, dst_ty, count)
                }
                2 => {
                    // Any two-word value (a fat pointer, a
                    // two-field aggregate constant, ...): its
                    // own two leaves.
                    self.operand_leaves(op, dst_ty, count)
                }
                _ => self.operand_leaves(op, dst_ty, count),
            },
            R::BinaryOp(op, pair) => {
                if count != 1 && !(count == 2 && int128_signed(dst_ty).is_some()) {
                    return Err(self.err("binary operation assigned through a dereference"));
                }
                let (lhs, rhs) = &**pair;
                let tmp = self.fresh();
                if count == 2 {
                    self.fresh();
                }
                self.lower_binop_to(tmp, dst_ty, *op, lhs, rhs)?;
                Ok((0..count as u32).map(|k| tmp + k).collect())
            }
            R::UnaryOp(op, operand) => {
                if count != 1 && !(count == 2 && int128_signed(dst_ty).is_some()) {
                    return Err(self.err("unary operation assigned through a dereference"));
                }
                let tmp = self.fresh();
                if count == 2 {
                    self.fresh();
                }
                self.lower_unop_to(tmp, dst_ty, *op, operand)?;
                Ok((0..count as u32).map(|k| tmp + k).collect())
            }
            R::Cast(kind, operand, target) => {
                if count == 2
                    && (int128_signed(target).is_some()
                        || matches!(kind, mir::CastKind::Transmute))
                {
                    // A two-word value the cast machinery writes
                    // leaf by leaf (a 128-bit result, a transmute
                    // between two-word types), not an unsize pair.
                    let tmp = self.fresh();
                    self.fresh();
                    self.lower_cast_to(tmp, kind, operand, target)?;
                    return Ok(vec![tmp, tmp + 1]);
                }
                if count == 2 {
                    let (a, b) = self.unsize_pair(kind, operand, target)?;
                    return Ok(vec![a, b]);
                }
                if count != 1 {
                    return Err(self.err("cast assigned through a dereference"));
                }
                let tmp = self.fresh();
                self.lower_cast_to(tmp, kind, operand, target)?;
                Ok(vec![tmp])
            }
            R::Reborrow(_, _, src_place) => match count {
                1 => Ok(vec![
                    self.operand_value(&mir::Operand::Copy(src_place.clone()), dst_ty)?,
                ]),
                2 => {
                    let (a, b) =
                        self.operand_pair(&mir::Operand::Copy(src_place.clone()))?;
                    Ok(vec![a, b])
                }
                _ => Err(self.err("reborrow of an aggregate value")),
            },
            R::Ref(_, _, target) | R::RawPtr(_, target) => {
                let words = self.ref_words(target, dst_ty)?;
                if words.len() != count {
                    return Err(self.err("reference value shape mismatch"));
                }
                Ok(words)
            }
            R::Aggregate(kind, ops) => {
                // Build the aggregate into a fresh run of scratch
                // slots (tag first for an ADT, then each field at
                // its layout offset); the caller stores the words
                // through the destination's address.
                let base = self.fresh();
                for _ in 1..count {
                    self.fresh();
                }
                match &**kind {
                    mir::AggregateKind::Tuple => {
                        let mir::Ty::Tuple(tys) = dst_ty else {
                            return Err(self.err("tuple aggregate type mismatch"));
                        };
                        if tys.len() != ops.len() {
                            return Err(self.err("tuple aggregate field count mismatch"));
                        }
                        let mut off = 0usize;
                        for (op, ty) in ops.iter().zip(tys.iter()) {
                            self.store_operand_leaves(op, ty, base + off as u32)?;
                            off += flatten(ty, &self.func.name, self.layouts)?;
                        }
                    }
                    mir::AggregateKind::Adt {
                        def,
                        variant_index,
                        field_index,
                        ..
                    } => {
                        if field_index.is_some() {
                            return Err(self.err(
                                "ADT aggregate with an active-field index (union)",
                            ));
                        }
                        let layout = self.adt_layout(dst_ty)?;
                        if count != layout.total {
                            return Err(self.err("ADT aggregate shape mismatch"));
                        }
                        let slots = layout
                            .variants
                            .get(*variant_index as usize)
                            .ok_or_else(|| {
                                self.err("ADT aggregate variant outside the layout")
                            })?
                            .clone();
                        if slots.len() != ops.len() {
                            return Err(self.err("ADT aggregate field count mismatch"));
                        }
                        if layout.tag {
                            let tag = self.discriminant_value(*def, *variant_index)?;
                            self.emit(Instr::Const { dst: base, imm: tag });
                        }
                        for (op, slot) in ops.iter().zip(slots.iter()) {
                            self.store_operand_leaves(
                                op,
                                &slot.ty,
                                base + slot.offset as u32,
                            )?;
                        }
                    }
                    mir::AggregateKind::Array(_) => {
                        let (elem, n) = match dst_ty {
                            mir::Ty::Array(elem, mir::ArrayLen::Known(n)) => {
                                (elem, *n as usize)
                            }
                            _ => {
                                return Err(self.err(
                                    "array aggregate with a non-array place type",
                                ))
                            }
                        };
                        if ops.len() != n {
                            return Err(self.err("array aggregate arity mismatch"));
                        }
                        let per = flatten(elem, &self.func.name, self.layouts)?;
                        let mut off = 0usize;
                        for op in ops {
                            self.store_operand_leaves(op, elem, base + off as u32)?;
                            off += per;
                        }
                    }
                    mir::AggregateKind::RawPtr { data_ty, .. } => {
                        self.store_raw_ptr_aggregate(data_ty, ops, base, count)?;
                    }
                    other => {
                        return Err(self.err(&format!(
                            "{} aggregate assigned through a dereference",
                            other.kind_name()
                        )));
                    }
                }
                Ok((0..count as u32).map(|k| base + k).collect())
            }
            other => Err(self.err(&format!(
                "{} rvalue assigned through a dereference",
                other.kind_name()
            ))),
        }
    }

    /// Lower `place = rvalue` for a place in memory: compute the
    /// value into scratch slots with the ordinary rvalue lowerings,
    /// then store it through the place's address, so the write
    /// lands on the original place.
    fn lower_assign_through_mem(
        &mut self,
        loc: &PlaceLoc,
        rvalue: &mir::Rvalue,
    ) -> Result<(), JitError> {
        let PlaceLoc::Mem { count, ty, .. } = loc else {
            unreachable!("lower_assign_through_mem on a frame place")
        };
        let Some(count) = *count else {
            return Err(self.err("assignment to an unsized place"));
        };
        let ty = ty.clone();
        let words = self.rvalue_words(rvalue, &ty, count)?;
        // Store from consecutive scratch slots.
        let base = self.fresh();
        for _ in 1..count {
            self.fresh();
        }
        for (k, w) in words.iter().enumerate() {
            self.emit(Instr::Copy {
                dst: base + k as u32,
                src: IrOperand::Value(*w),
            });
        }
        self.store_mem_from(loc, base);
        Ok(())
    }

    /// `operand as &[T]` where the operand is a `&[T; N]`: the
    /// pair is the operand's address word and the array's length.
    /// Other unsize coercions (to a trait object, from a raw
    /// pointer — the interpreter accepts only a reference source)
    /// stay unsupported.
    fn unsize_pair(
        &mut self,
        kind: &mir::CastKind,
        operand: &mir::Operand,
        target: &mir::Ty,
    ) -> Result<(u32, u32), JitError> {
        let mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize) = kind else {
            return Err(self.err(&format!("cast {kind:?} with a two-word result")));
        };
        let src_ty = self
            .operand_place_ty(operand)
            .ok_or_else(|| self.err("Unsize cast with an unknown source type"))?;
        let (mir::Ty::Ref(_, src_pointee, _), mir::Ty::Ref(_, dst_pointee, _)) =
            (&src_ty, target)
        else {
            return Err(self.err("Unsize cast between non-reference types"));
        };
        match (&**src_pointee, &**dst_pointee) {
            (mir::Ty::Array(elem, mir::ArrayLen::Known(n)), mir::Ty::Slice(elem2))
                if elem == elem2 =>
            {
                let n = *n;
                let addr = self.operand_value(operand, &src_ty)?;
                let l = self.len_value(&LenSrc::Const(n));
                Ok((addr, l))
            }
            _ => Err(self.err(&format!(
                "Unsize cast from {} to {}",
                src_pointee.kind_name(),
                dst_pointee.kind_name()
            ))),
        }
    }

    /// The length of the value at `place`, for a `Len` rvalue: an
    /// array's declared length, or a slice / `str` place's length
    /// word.
    fn len_of_place(&mut self, place: &mir::Place) -> Result<u32, JitError> {
        let loc = self.classify_place(place)?;
        match &loc {
            PlaceLoc::Slots { ty, .. } => match ty {
                mir::Ty::Array(_, mir::ArrayLen::Known(n)) => {
                    Ok(self.len_value(&LenSrc::Const(*n)))
                }
                _ => Err(self.err(&format!("Len of type {}", ty.kind_name()))),
            },
            PlaceLoc::Mem {
                ty, len, checks, ..
            } => {
                // The interpreter resolves the place before reading
                // its length, so the place's checks (an index's
                // bounds, a sub-slice's range) fire here too.
                self.emit_checks(checks);
                match ty {
                    mir::Ty::Array(_, mir::ArrayLen::Known(n)) => {
                        Ok(self.len_value(&LenSrc::Const(*n)))
                    }
                    mir::Ty::Slice(_) | mir::Ty::Str => {
                        let l = len.clone().ok_or_else(|| {
                            self.err("Len of a slice place without a length")
                        })?;
                        Ok(self.len_value(&l))
                    }
                    _ => Err(self.err(&format!("Len of type {}", ty.kind_name()))),
                }
            }
        }
    }


    fn lower_assign(&mut self, place: &mir::Place, rvalue: &mir::Rvalue) -> Result<(), JitError> {
        use mir::Rvalue as R;
        // An assignment to a place in memory (`*p = ...`,
        // `(*p).field = ...`, `arr[i] = ...`) computes the value
        // into scratch slots and stores it through the place's
        // address, so the write lands on the original place.
        let dst_loc = self.classify_place(place)?;
        if let PlaceLoc::Mem { .. } = dst_loc {
            return self.lower_assign_through_mem(&dst_loc, rvalue);
        }
        match rvalue {
            R::Use(op) => {
                let (dst, dst_count, dst_ty) = self.place_leaves(place)?;
                match op {
                    mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                        let src_loc = self.classify_place(p)?;
                        if let PlaceLoc::Mem { count, .. } = &src_loc {
                            if *count != Some(dst_count) {
                                return Err(
                                    self.err("Use copy with mismatched value shapes")
                                );
                            }
                            self.load_mem_to(&src_loc, dst);
                            return Ok(());
                        }
                        let (src, src_count, _) = self.place_leaves(p)?;
                        if src_count != dst_count {
                            return Err(self.err("Use copy with mismatched value shapes"));
                        }
                        for k in 0..dst_count as u32 {
                            self.emit(Instr::Copy {
                                dst: dst + k,
                                src: IrOperand::Value(src + k),
                            });
                        }
                        Ok(())
                    }
                    mir::Operand::Constant(c) => {
                        if dst_count == 1 {
                            if let Some(slot) = self.const_data_addr(c, &dst_ty) {
                                self.emit(Instr::Copy {
                                    dst,
                                    src: IrOperand::Value(slot),
                                });
                                return Ok(());
                            }
                            match &c.literal {
                                // A pointer-shaped constant (a
                                // reference to a scalar constant, a
                                // byte string) materialises through
                                // the leaf path like a wide one.
                                mir::ConstKind::Value(v) if is_pointer_shaped(v) => {
                                    self.const_value_leaves(v, &dst_ty, dst)?;
                                }
                                _ => {
                                    let v = self.const_i64(c, &dst_ty)?;
                                    self.emit(Instr::Const { dst, imm: v });
                                }
                            }
                        } else if dst_count > 1 {
                            // A multi-leaf constant (a tuple or an
                            // ADT with payload fields) materialises
                            // leaf by leaf under the destination's
                            // type.
                            match &c.literal {
                                mir::ConstKind::Value(v) => {
                                    self.const_value_leaves(v, &dst_ty, dst)?;
                                }
                                _ => {
                                    return Err(
                                        self.err("constant of aggregate type")
                                    )
                                }
                            }
                        }
                        // A zero-leaf (unit / never) destination has
                        // nothing to store.
                        Ok(())
                    }
                }
            }
            R::BinaryOp(op, pair) => {
                let (lhs, rhs) = &**pair;
                let (dst, _, dst_ty) = self.place_leaves(place)?;
                self.lower_binop_to(dst, &dst_ty, *op, lhs, rhs)
            }
            R::CheckedBinaryOp(op, pair) => {
                let (lhs, rhs) = &**pair;
                let (dst, _, dst_ty) = self.place_leaves(place)?;
                self.lower_checked_binop_to(dst, &dst_ty, *op, lhs, rhs)
            }
            R::UnaryOp(op, operand) => {
                let (dst, count, dst_ty) = self.place_leaves(place)?;
                if count != 1 && int128_signed(&dst_ty).is_none() {
                    return Err(self.err("unary operation on an aggregate value"));
                }
                self.lower_unop_to(dst, &dst_ty, *op, operand)
            }
            R::Cast(kind, operand, target) => {
                let (dst, count, _) = self.place_leaves(place)?;
                if count == 2
                    && (int128_signed(target).is_some()
                        || matches!(kind, mir::CastKind::Transmute))
                {
                    return self.lower_cast_to(dst, kind, operand, target);
                }
                if count == 2 {
                    let (a, b) = self.unsize_pair(kind, operand, target)?;
                    self.emit(Instr::Copy {
                        dst,
                        src: IrOperand::Value(a),
                    });
                    self.emit(Instr::Copy {
                        dst: dst + 1,
                        src: IrOperand::Value(b),
                    });
                    return Ok(());
                }
                if count != 1 {
                    return Err(self.err("cast of an aggregate value"));
                }
                self.lower_cast_to(dst, kind, operand, target)
            }
            R::Aggregate(kind, ops) => self.lower_aggregate(place, kind, ops),
            R::Repeat(op, count) => self.lower_repeat(place, op, *count),
            R::Ref(_, _, target) => self.lower_ref_creation(place, target),
            R::ThreadLocalRef(_) => Err(self.err("ThreadLocalRef rvalue")),
            R::RawPtr(_, target) => self.lower_ref_creation(place, target),
            R::Len(len_place) => {
                let (dst, count, _) = self.place_leaves(place)?;
                if count != 1 {
                    return Err(self.err("Len result is not a scalar"));
                }
                let v = self.len_of_place(len_place)?;
                self.emit(Instr::Copy {
                    dst,
                    src: IrOperand::Value(v),
                });
                Ok(())
            }
            R::Discriminant(src_place) => {
                let (dst, dst_count, dst_ty) = self.place_leaves(place)?;
                if dst_count != 1 {
                    return Err(self.err("Discriminant result is not a scalar"));
                }
                let src_loc = self.classify_place(src_place)?;
                if let PlaceLoc::Mem { ty, .. } = &src_loc {
                    let mir::Ty::Adt(def, _) = ty else {
                        return Err(self.err("Discriminant of a non-ADT place"));
                    };
                    let layout = self.adt_layout(ty)?;
                    if layout.tag {
                        // The value's first leaf in memory holds
                        // the discriminant value itself.
                        let mut tag_loc = src_loc.clone();
                        if let PlaceLoc::Mem { count, .. } = &mut tag_loc {
                            *count = Some(1);
                        }
                        self.load_mem_to(&tag_loc, dst);
                    } else {
                        // A single-variant ADT's discriminant is
                        // the constant of its only variant.
                        let v = self.discriminant_value(*def, 0)?;
                        self.emit(Instr::Const { dst, imm: v });
                    }
                    self.normalize(dst, &dst_ty);
                    return Ok(());
                }
                let (src, _, src_ty) = self.place_leaves(src_place)?;
                let mir::Ty::Adt(def, _) = &src_ty else {
                    return Err(self.err("Discriminant of a non-ADT place"));
                };
                let layout = self.adt_layout(&src_ty)?;
                if layout.tag {
                    // The value's first leaf holds the discriminant
                    // value itself; the read canonicalises it
                    // through the destination's (discriminant) type,
                    // so `SwitchInt` on the result compares against
                    // the same values the interpreter produces.
                    self.emit(Instr::Copy {
                        dst,
                        src: IrOperand::Value(src),
                    });
                } else {
                    let v = self.discriminant_value(*def, 0)?;
                    self.emit(Instr::Const { dst, imm: v });
                }
                self.normalize(dst, &dst_ty);
                Ok(())
            }
            R::Reborrow(_, _, src_place) => {
                // A bitwise copy of the pointer value held at the
                // place, reinterpreted as the target pointer type:
                // exactly a `Use` copy of the place's value.
                let use_rv = mir::Rvalue::Use(mir::Operand::Copy(src_place.clone()));
                self.lower_assign(place, &use_rv)
            }
            R::NullaryOp(..) => Err(self.err("NullaryOp rvalue")),
            R::ShallowInitBox(..) => Err(self.err("ShallowInitBox rvalue")),
            R::CopyForDeref(src_place) => {
                let (dst, count, dst_ty) = self.place_leaves(place)?;
                let src_ty = self.place_ty(src_place)?;
                if !matches!(src_ty, mir::Ty::Ref(..) | mir::Ty::RawPtr(..)) {
                    // The interpreter's other arm: the place's own
                    // value, as a `Use` copy.
                    return self.lower_assign(
                        place,
                        &mir::Rvalue::Use(mir::Operand::Copy(src_place.clone())),
                    );
                }
                // The value is the pointee: read through one
                // dereference of the source place, leaf by leaf
                // (a payload ADT pointee copies its whole layout).
                let _ = count;
                let mut projection = src_place.projection.clone();
                projection.push(mir::PlaceElem::Deref);
                let deref_place = mir::Place {
                    local: src_place.local,
                    projection,
                };
                self.store_operand_leaves(
                    &mir::Operand::Copy(deref_place),
                    &dst_ty,
                    dst,
                )?;
                Ok(())
            }
        }
    }

    /// `Rvalue::Repeat`: build the destination array by copying one
    /// element value into every element slot. The element is
    /// evaluated once, as in the interpreter (which materialises the
    /// same value `count` times). Element shapes the slot model
    /// cannot hold never reach here — `flatten` has already rejected
    /// the destination's type — and a constant element of aggregate
    /// type stays unsupported, as aggregate constants are elsewhere.
    fn lower_repeat(
        &mut self,
        place: &mir::Place,
        op: &mir::Operand,
        count: u64,
    ) -> Result<(), JitError> {
        // The interpreter's conversion bound on repeat counts,
        // mirrored so both engines accept the same programs.
        if count > 1_000_000 {
            return Err(self.err(&format!("Repeat of {count} elements")));
        }
        let (dst, dst_count, dst_ty) = self.place_leaves(place)?;
        let mir::Ty::Array(elem_ty, _) = &dst_ty else {
            return Err(self.err("Repeat rvalue with a non-array destination"));
        };
        let elem_ty = (**elem_ty).clone();
        let per = flatten(&elem_ty, &self.func.name, self.layouts)?;
        if per == 0 {
            // A zero-leaf element (unit / never) stores nothing; the
            // destination necessarily has zero leaves too.
            return Ok(());
        }
        let total = (count as usize).checked_mul(per).unwrap_or(usize::MAX);
        if total != dst_count {
            return Err(self.err("Repeat count does not match the array length"));
        }
        match op {
            mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                let (src, src_count, _) = self.place_leaves(p)?;
                if src_count != per {
                    return Err(self.err("Repeat element shape mismatch"));
                }
                for i in 0..count as usize {
                    for k in 0..per as u32 {
                        self.emit(Instr::Copy {
                            dst: dst + (i * per) as u32 + k,
                            src: IrOperand::Value(src + k),
                        });
                    }
                }
                Ok(())
            }
            mir::Operand::Constant(_) => {
                let leaves = self.operand_leaves(op, &elem_ty, per)?;
                for i in 0..count as usize {
                    for (k, leaf) in leaves.iter().enumerate() {
                        self.emit(Instr::Copy {
                            dst: dst + (i * per) as u32 + k as u32,
                            src: IrOperand::Value(*leaf),
                        });
                    }
                }
                Ok(())
            }
        }
    }

}
