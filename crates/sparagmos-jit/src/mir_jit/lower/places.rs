use sparagmos_mir as mir;

use crate::ir::{
    BinOp as IrBinOp,
    Instr,
    Operand as IrOperand,
    Terminator as IrTerminator,
};

use super::*;

impl<'a> FnLower<'a> {
    /// The value slot of a dynamic `Index` projection's local, and
    /// whether the index's type is signed. The interpreter reads
    /// the index as a `usize`, panicking on a negative signed value
    /// with its negative-index message; the JIT supports both
    /// signednesses (the check a signed index needs is recorded
    /// alongside the bounds check) and rejects any non-integer
    /// index type by name.
    fn index_slot(&self, local: mir::Local) -> Result<(u32, bool), JitError> {
        let decl = &self.body.local_decls[local.index()];
        let signed = match decl.ty {
            mir::Ty::Uint(_) => false,
            mir::Ty::Int(_) => true,
            _ => {
                return Err(self.err(&format!(
                    "Index projection with index type {}",
                    decl.ty.kind_name()
                )));
            }
        };
        let (slot, count) = self.local_slots[local.index()];
        if count != 1 {
            return Err(self.err("Index projection with a multi-word index"));
        }
        Ok((slot, signed))
    }


    /// The resolved length source of a sub-slice selecting
    /// `from..to` (or `from..len - to` when `from_end`) from a base
    /// whose length source is `base_len`: the interpreter's
    /// resolution of the same projection. A base length known at
    /// lowering time resolves the count outright; a range that
    /// does not fit resolves to a placeholder count instead — the
    /// place's sub-slice check panics before any use reads the
    /// length, exactly as the interpreter panics while resolving
    /// the place.
    fn subslice_len(base_len: &LenSrc, from: u64, to: u64, from_end: bool) -> LenSrc {
        match base_len {
            LenSrc::Const(n) => {
                let end = if from_end { n.checked_sub(to) } else { Some(to) };
                let count = end.and_then(|e| e.checked_sub(from)).unwrap_or(0);
                LenSrc::Const(count)
            }
            _ => {
                if from_end {
                    LenSrc::Tail {
                        base: Box::new(base_len.clone()),
                        skip: to.saturating_add(from),
                    }
                } else {
                    LenSrc::Const(to.checked_sub(from).unwrap_or(0))
                }
            }
        }
    }

    /// Resolve a place against the slot model: either leaves of a
    /// local in the current frame, or leaves in memory, reached
    /// through dereferences and dynamic indices. Pure (emits no
    /// code); the memory form carries the bounds checks a use of
    /// the place must emit first.
    pub(super) fn classify_place(&self, place: &mir::Place) -> Result<PlaceLoc, JitError> {
        let (local_first, _) = self.local_slots[place.local.index()];
        let mut ty = self.body.local_decls[place.local.index()].ty.clone();
        let mut offset = 0usize;
        let mut variant = 0u32;
        let mut i = 0usize;
        while i < place.projection.len() {
            match &place.projection[i] {
                mir::PlaceElem::Deref => {
                    let pointee = match &ty {
                        mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _) => {
                            (**pointee).clone()
                        }
                        _ => {
                            return Err(self.err(&format!(
                                "dereference of a non-pointer type {}",
                                ty.kind_name()
                            )));
                        }
                    };
                    // Validates the pointer's own shape (a trait
                    // object pointee is rejected by `flatten`).
                    flatten(&ty, &self.func.name, self.layouts)?;
                    let ptr_slot = local_first + offset as u32;
                    let len = match &pointee {
                        mir::Ty::Slice(_) | mir::Ty::Str => {
                            Some(LenSrc::Slot(ptr_slot + 1))
                        }
                        _ => None,
                    };
                    let addr = AddrExpr {
                        source: PtrSrc::Slot(ptr_slot),
                        const_off: 0,
                        dyn_terms: Vec::new(),
                    };
                    return self.classify_mem(place, i + 1, addr, pointee, len, Vec::new());
                }
                mir::PlaceElem::Index(local) => {
                    let mir::Ty::Array(elem, len) = &ty else {
                        return Err(self.err(&format!(
                            "Index projection on non-array type {}",
                            ty.kind_name()
                        )));
                    };
                    let mir::ArrayLen::Known(n) = len else {
                        return Err(
                            self.err("Index projection on an array of unknown length")
                        );
                    };
                    let n = *n;
                    let elem = (**elem).clone();
                    let (idx_slot, signed) = self.index_slot(*local)?;
                    let per = flatten(&elem, &self.func.name, self.layouts)?;
                    let addr = AddrExpr {
                        source: PtrSrc::SlotAddr(local_first + offset as u32),
                        const_off: 0,
                        dyn_terms: vec![(IndexSrc::Slot(idx_slot), per)],
                    };
                    let checks = vec![PlaceCheck::Index(IndexCheck {
                        index: IndexSrc::Slot(idx_slot),
                        len: LenSrc::Const(n),
                        neg: signed.then(|| place_display(place)),
                    })];
                    return self.classify_mem(place, i + 1, addr, elem, None, checks);
                }
                mir::PlaceElem::Subslice { from, to, from_end } => {
                    let mir::Ty::Array(elem, arr_len) = &ty else {
                        return Err(self.err(&format!(
                            "Subslice projection on non-array type {}",
                            ty.kind_name()
                        )));
                    };
                    let mir::ArrayLen::Known(n) = arr_len else {
                        return Err(
                            self.err("Subslice projection on an array of unknown length")
                        );
                    };
                    let n = *n;
                    let elem = (**elem).clone();
                    let per = flatten(&elem, &self.func.name, self.layouts)?;
                    let base_len = LenSrc::Const(n);
                    let checks = vec![PlaceCheck::Subslice(SubsliceCheck {
                        from: *from,
                        to: *to,
                        from_end: *from_end,
                        len: base_len.clone(),
                    })];
                    let len = Self::subslice_len(&base_len, *from, *to, *from_end);
                    let addr = AddrExpr {
                        source: PtrSrc::SlotAddr(local_first + offset as u32),
                        const_off: subslice_offset(*from, per),
                        dyn_terms: Vec::new(),
                    };
                    return self.classify_mem(
                        place,
                        i + 1,
                        addr,
                        mir::Ty::Slice(Box::new(elem)),
                        Some(len),
                        checks,
                    );
                }
                elem => {
                    self.frame_step(&mut offset, &mut ty, &mut variant, elem)?;
                    i += 1;
                }
            }
        }
        let count = flatten(&ty, &self.func.name, self.layouts)?;
        Ok(PlaceLoc::Slots {
            first: local_first + offset as u32,
            count,
            ty,
        })
    }

    /// Continue [`Self::classify_place`] in memory: the remaining
    /// projections apply to the pointee at `addr`.
    fn classify_mem(
        &self,
        place: &mir::Place,
        mut i: usize,
        mut addr: AddrExpr,
        mut ty: mir::Ty,
        mut len: Option<LenSrc>,
        mut checks: Vec<PlaceCheck>,
    ) -> Result<PlaceLoc, JitError> {
        let mut variant = 0u32;
        while i < place.projection.len() {
            match &place.projection[i] {
                mir::PlaceElem::Field(idx, field_ty) => {
                    if let mir::Ty::Adt(..) = &ty {
                        let layout = self.adt_layout(&ty)?;
                        let slot = layout.field(variant, *idx).ok_or_else(|| {
                            self.err("Field projection outside the ADT layout")
                        })?;
                        addr.const_off += slot.offset;
                        ty = field_ty.clone();
                        variant = 0;
                        len = None;
                        i += 1;
                        continue;
                    }
                    let mir::Ty::Tuple(fields) = &ty else {
                        return Err(self.err(&format!(
                            "Field projection on non-tuple type {}",
                            ty.kind_name()
                        )));
                    };
                    let idx = *idx as usize;
                    if idx >= fields.len() {
                        return Err(self.err("Field projection index out of range"));
                    }
                    for f in &fields[..idx] {
                        addr.const_off += flatten(f, &self.func.name, self.layouts)?;
                    }
                    ty = field_ty.clone();
                    len = None;
                }
                mir::PlaceElem::ConstantIndex {
                    offset: idx,
                    min_length: _,
                    from_end,
                } => {
                    if matches!(ty, mir::Ty::Slice(_) | mir::Ty::Str) {
                        // A literal index into a slice: the bound is
                        // the runtime length, so the index becomes a
                        // checked dynamic term with a constant index.
                        if *from_end {
                            return Err(
                                self.err("ConstantIndex from the end of a slice")
                            );
                        }
                        let (elem_ty, per) = match &ty {
                            mir::Ty::Slice(elem) => {
                                ((**elem).clone(), flatten(elem, &self.func.name, self.layouts)?)
                            }
                            _ => (mir::Ty::Uint(mir::UintTy::U8), 1),
                        };
                        let len_src = len.clone().ok_or_else(|| {
                            self.err("index into a slice whose length is not available")
                        })?;
                        addr.dyn_terms.push((IndexSrc::Const(*idx), per));
                        checks.push(PlaceCheck::Index(IndexCheck {
                            index: IndexSrc::Const(*idx),
                            len: len_src,
                            neg: None,
                        }));
                        ty = elem_ty;
                        len = None;
                        i += 1;
                        continue;
                    }
                    let mir::Ty::Array(elem, arr_len) = &ty else {
                        return Err(self.err(&format!(
                            "ConstantIndex projection on non-array type {}",
                            ty.kind_name()
                        )));
                    };
                    let mir::ArrayLen::Known(arr_len) = arr_len else {
                        return Err(self.err(
                            "ConstantIndex projection on an array of unknown length",
                        ));
                    };
                    let idx = if *from_end {
                        match arr_len.checked_sub(*idx).filter(|j| *j < *arr_len) {
                            Some(j) => j,
                            None => {
                                return Err(
                                    self.err("ConstantIndex projection out of range")
                                );
                            }
                        }
                    } else if idx < arr_len {
                        *idx
                    } else {
                        return Err(self.err("ConstantIndex projection out of range"));
                    };
                    addr.const_off += idx as usize * flatten(elem, &self.func.name, self.layouts)?;
                    ty = (**elem).clone();
                    len = None;
                }
                mir::PlaceElem::Index(local) => {
                    let (elem_ty, len_src) = match &ty {
                        mir::Ty::Array(elem, arr_len) => {
                            let mir::ArrayLen::Known(n) = arr_len else {
                                return Err(self.err(
                                    "Index projection on an array of unknown length",
                                ));
                            };
                            ((**elem).clone(), LenSrc::Const(*n))
                        }
                        mir::Ty::Slice(elem) => {
                            let l = len.clone().ok_or_else(|| {
                                self.err("index into a slice whose length is not available")
                            })?;
                            ((**elem).clone(), l)
                        }
                        mir::Ty::Str => {
                            let l = len.clone().ok_or_else(|| {
                                self.err("index into a str whose length is not available")
                            })?;
                            (mir::Ty::Uint(mir::UintTy::U8), l)
                        }
                        _ => {
                            return Err(self.err(&format!(
                                "Index projection on non-array type {}",
                                ty.kind_name()
                            )));
                        }
                    };
                    let (idx_slot, signed) = self.index_slot(*local)?;
                    let per = flatten(&elem_ty, &self.func.name, self.layouts)?;
                    addr.dyn_terms.push((IndexSrc::Slot(idx_slot), per));
                    checks.push(PlaceCheck::Index(IndexCheck {
                        index: IndexSrc::Slot(idx_slot),
                        len: len_src,
                        neg: signed.then(|| place_display(place)),
                    }));
                    ty = elem_ty;
                    len = None;
                }
                mir::PlaceElem::Deref => {
                    let pointee = match &ty {
                        mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _) => {
                            (**pointee).clone()
                        }
                        _ => {
                            return Err(self.err(&format!(
                                "dereference of a non-pointer type {}",
                                ty.kind_name()
                            )));
                        }
                    };
                    // Rejects a trait-object pointee, as in frame.
                    flatten(&ty, &self.func.name, self.layouts)?;
                    let pair_addr = addr.clone();
                    match &pointee {
                        mir::Ty::Slice(_) | mir::Ty::Str => {
                            let mut len_addr = addr.clone();
                            len_addr.const_off += 1;
                            addr = AddrExpr {
                                source: PtrSrc::Loaded(Box::new(pair_addr)),
                                const_off: 0,
                                dyn_terms: Vec::new(),
                            };
                            len = Some(LenSrc::Mem(Box::new(len_addr)));
                        }
                        _ => {
                            addr = AddrExpr {
                                source: PtrSrc::Loaded(Box::new(pair_addr)),
                                const_off: 0,
                                dyn_terms: Vec::new(),
                            };
                            len = None;
                        }
                    }
                    ty = pointee;
                }
                mir::PlaceElem::Downcast(_, v) => {
                    if !matches!(ty, mir::Ty::Adt(..)) {
                        return Err(self.err(&format!(
                            "Downcast projection on non-ADT type {}",
                            ty.kind_name()
                        )));
                    }
                    variant = *v;
                }
                mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => {
                    ty = t.clone();
                    variant = 0;
                }
                mir::PlaceElem::Subslice { from, to, from_end } => {
                    let (elem_ty, base_len) = match &ty {
                        mir::Ty::Array(elem, arr_len) => {
                            let mir::ArrayLen::Known(n) = arr_len else {
                                return Err(self.err(
                                    "Subslice projection on an array of unknown length",
                                ));
                            };
                            ((**elem).clone(), LenSrc::Const(*n))
                        }
                        mir::Ty::Slice(elem) => {
                            let l = len.clone().ok_or_else(|| {
                                self.err("sub-slice of a slice whose length is not available")
                            })?;
                            ((**elem).clone(), l)
                        }
                        mir::Ty::Str => {
                            let l = len.clone().ok_or_else(|| {
                                self.err("sub-slice of a str whose length is not available")
                            })?;
                            (mir::Ty::Uint(mir::UintTy::U8), l)
                        }
                        _ => {
                            return Err(self.err(&format!(
                                "Subslice projection on type {}",
                                ty.kind_name()
                            )));
                        }
                    };
                    let per = flatten(&elem_ty, &self.func.name, self.layouts)?;
                    addr.const_off = addr.const_off.saturating_add(subslice_offset(*from, per));
                    checks.push(PlaceCheck::Subslice(SubsliceCheck {
                        from: *from,
                        to: *to,
                        from_end: *from_end,
                        len: base_len.clone(),
                    }));
                    len = Some(Self::subslice_len(&base_len, *from, *to, *from_end));
                    ty = if matches!(ty, mir::Ty::Str) {
                        mir::Ty::Str
                    } else {
                        mir::Ty::Slice(Box::new(elem_ty))
                    };
                }
            }
            i += 1;
        }
        // A slice / `str` place itself has no word representation:
        // it is only ever the target of a reference, a `Len`, or a
        // further projection, and those read the pair from the
        // address and length source instead.
        let count = match &ty {
            mir::Ty::Slice(_) | mir::Ty::Str => None,
            _ => Some(flatten(&ty, &self.func.name, self.layouts)?),
        };
        Ok(PlaceLoc::Mem {
            addr,
            count,
            ty,
            len,
            checks,
        })
    }

    /// Resolve a place to (first leaf slot, leaf count, type).
    /// Frame places only: a place in memory is an error here, and
    /// the callers that accept one classify it themselves.
    pub(super) fn place_leaves(&self, place: &mir::Place) -> Result<(u32, usize, mir::Ty), JitError> {
        match self.classify_place(place)? {
            PlaceLoc::Slots { first, count, ty } => Ok((first, count, ty)),
            PlaceLoc::Mem { .. } => Err(self.err("place in memory (through a dereference or an index)")),
        }
    }


    /// Emit the checks a memory place needs, in projection order.
    /// A failed check is the interpreter's panic for the projection
    /// that needed it, with the runtime values in the message.
    pub(super) fn emit_checks(&mut self, checks: &[PlaceCheck]) {
        for check in checks {
            match check {
                PlaceCheck::Index(check) => self.emit_index_check(check),
                PlaceCheck::Subslice(check) => self.emit_subslice_check(check),
            }
        }
    }

    /// A constant materialised into a fresh value slot.
    pub(super) fn const_slot(&mut self, n: u64) -> u32 {
        let d = self.fresh();
        self.emit(Instr::Const {
            dst: d,
            imm: n as i64,
        });
        d
    }

    /// Continue in a fresh block when `ok` is nonzero; jump to a
    /// panic block holding `panic_instrs` when it is zero.
    fn branch_unless(&mut self, ok: u32, panic_instrs: Vec<Instr>) {
        let pblk = self.new_block();
        self.seal_block(pblk, panic_instrs, IrTerminator::Unreachable);
        let cont = self.new_block();
        self.seal_and_continue(
            IrTerminator::Branch {
                cond: IrOperand::Value(ok),
                then_blk: cont,
                else_blk: pblk,
            },
            cont,
        );
    }

    /// Emit `ok = lhs <op> rhs` and continue only when it holds,
    /// panicking with `panic_instrs` otherwise.
    fn guard_cmp(&mut self, op: IrBinOp, lhs: u32, rhs: u32, panic_instrs: Vec<Instr>) {
        let ok = self.fresh();
        self.emit(Instr::BinOp {
            dst: ok,
            op,
            lhs: IrOperand::Value(lhs),
            rhs: IrOperand::Value(rhs),
        });
        self.branch_unless(ok, panic_instrs);
    }

    /// Emit an indexed place's bounds check: for a signed index,
    /// the interpreter's negative-index panic first (the slot holds
    /// the sign-extended value, so a signed `>= 0` test is exact,
    /// and a value that passes it compares against the length on
    /// its mathematical value), then the unsigned bounds check
    /// with the interpreter's index panic.
    fn emit_index_check(&mut self, check: &IndexCheck) {
        let len_v = self.len_value(&check.len);
        let idx_v = match check.index {
            IndexSrc::Slot(s) => s,
            IndexSrc::Const(n) => self.const_slot(n),
        };
        if let Some(place) = &check.neg {
            let place = self.places.intern(place);
            let ok = self.fresh();
            self.emit(Instr::BinOp {
                dst: ok,
                op: IrBinOp::Ge,
                lhs: IrOperand::Value(idx_v),
                rhs: IrOperand::Const(0),
            });
            self.branch_unless(
                ok,
                vec![Instr::PanicNegativeIndex {
                    index: IrOperand::Value(idx_v),
                    place,
                }],
            );
        }
        self.guard_cmp(
            IrBinOp::LtU,
            idx_v,
            len_v,
            vec![Instr::PanicIndexOob {
                len: IrOperand::Value(len_v),
                index: IrOperand::Value(idx_v),
            }],
        );
    }

    /// Emit a sub-slice projection's range check, in the
    /// interpreter's order: with `from_end`, a `to` past the base's
    /// length is the range-end panic before the end is even
    /// resolved; then a start past the end is the inverted-range
    /// panic; then (for a plain range) an end past the length is
    /// the range-end panic.
    fn emit_subslice_check(&mut self, check: &SubsliceCheck) {
        let len_v = self.len_value(&check.len);
        let end_v = if check.from_end {
            let to_v = self.const_slot(check.to);
            self.guard_cmp(
                IrBinOp::LeU,
                to_v,
                len_v,
                vec![Instr::PanicRangeEnd {
                    end: IrOperand::Value(to_v),
                    len: IrOperand::Value(len_v),
                }],
            );
            let d = self.fresh();
            self.emit(Instr::BinOp {
                dst: d,
                op: IrBinOp::Sub,
                lhs: IrOperand::Value(len_v),
                rhs: IrOperand::Value(to_v),
            });
            d
        } else {
            self.const_slot(check.to)
        };
        let from_v = self.const_slot(check.from);
        self.guard_cmp(
            IrBinOp::LeU,
            from_v,
            end_v,
            vec![Instr::PanicSliceIndex {
                from: IrOperand::Value(from_v),
                end: IrOperand::Value(end_v),
            }],
        );
        if !check.from_end {
            self.guard_cmp(
                IrBinOp::LeU,
                end_v,
                len_v,
                vec![Instr::PanicRangeEnd {
                    end: IrOperand::Value(end_v),
                    len: IrOperand::Value(len_v),
                }],
            );
        }
    }

    /// The value of a length source, materialised into a slot.
    pub(super) fn len_value(&mut self, len: &LenSrc) -> u32 {
        match len {
            LenSrc::Const(n) => {
                let d = self.fresh();
                self.emit(Instr::Const {
                    dst: d,
                    imm: *n as i64,
                });
                d
            }
            LenSrc::Slot(s) => *s,
            LenSrc::Mem(a) => {
                let addr = self.materialize_addr(a);
                let d = self.fresh();
                self.emit(Instr::LoadPtr {
                    dst: d,
                    ptr: IrOperand::Value(addr),
                });
                d
            }
            LenSrc::Tail { base, skip } => {
                let b = self.len_value(base);
                let d = self.fresh();
                self.emit(Instr::BinOp {
                    dst: d,
                    op: IrBinOp::Sub,
                    lhs: IrOperand::Value(b),
                    rhs: IrOperand::Const(*skip as i64),
                });
                d
            }
        }
    }

    /// Emit the computation of a memory place's address; returns
    /// the slot holding it. Bounds checks are *not* emitted here —
    /// callers emit the place's checks first.
    pub(super) fn materialize_addr(&mut self, addr: &AddrExpr) -> u32 {
        let mut base = match &addr.source {
            PtrSrc::Slot(s) => *s,
            PtrSrc::SlotAddr(s) => {
                let d = self.fresh();
                self.emit(Instr::SlotAddr { dst: d, slot: *s });
                d
            }
            PtrSrc::Loaded(inner) => {
                let a = self.materialize_addr(inner);
                let d = self.fresh();
                self.emit(Instr::LoadPtr {
                    dst: d,
                    ptr: IrOperand::Value(a),
                });
                d
            }
        };
        for (idx, per) in &addr.dyn_terms {
            let idx = match idx {
                IndexSrc::Slot(s) => *s,
                IndexSrc::Const(n) => {
                    let d = self.fresh();
                    self.emit(Instr::Const {
                        dst: d,
                        imm: *n as i64,
                    });
                    d
                }
            };
            let per = *per;
            let t = self.fresh();
            self.emit(Instr::BinOp {
                dst: t,
                op: IrBinOp::Mul,
                lhs: IrOperand::Value(idx),
                rhs: IrOperand::Const((per * 8) as i64),
            });
            let d = self.fresh();
            self.emit(Instr::BinOp {
                dst: d,
                op: IrBinOp::Sub,
                lhs: IrOperand::Value(base),
                rhs: IrOperand::Value(t),
            });
            base = d;
        }
        if addr.const_off != 0 {
            let d = self.fresh();
            self.emit(Instr::BinOp {
                dst: d,
                op: IrBinOp::Sub,
                lhs: IrOperand::Value(base),
                rhs: IrOperand::Const((addr.const_off * 8) as i64),
            });
            base = d;
        }
        base
    }

    /// Load a memory place's leaves into consecutive slots starting
    /// at `dst` (bounds checks first).
    pub(super) fn load_mem_to(&mut self, loc: &PlaceLoc, dst: u32) {
        let PlaceLoc::Mem {
            addr,
            count,
            checks,
            ..
        } = loc
        else {
            unreachable!("load_mem_to on a frame place")
        };
        let count = count.expect("load of an unsized place");
        self.emit_checks(checks);
        let base = self.materialize_addr(addr);
        for k in 0..count as u32 {
            let a = if k == 0 {
                base
            } else {
                let d = self.fresh();
                self.emit(Instr::BinOp {
                    dst: d,
                    op: IrBinOp::Sub,
                    lhs: IrOperand::Value(base),
                    rhs: IrOperand::Const((k * 8) as i64),
                });
                d
            };
            self.emit(Instr::LoadPtr {
                dst: dst + k,
                ptr: IrOperand::Value(a),
            });
        }
    }

    /// Store consecutive slots starting at `src` into a memory
    /// place (bounds checks first).
    pub(super) fn store_mem_from(&mut self, loc: &PlaceLoc, src: u32) {
        let PlaceLoc::Mem {
            addr,
            count,
            checks,
            ..
        } = loc
        else {
            unreachable!("store_mem_from on a frame place")
        };
        let count = count.expect("store of an unsized place");
        self.emit_checks(checks);
        let base = self.materialize_addr(addr);
        for k in 0..count as u32 {
            let a = if k == 0 {
                base
            } else {
                let d = self.fresh();
                self.emit(Instr::BinOp {
                    dst: d,
                    op: IrBinOp::Sub,
                    lhs: IrOperand::Value(base),
                    rhs: IrOperand::Const((k * 8) as i64),
                });
                d
            };
            self.emit(Instr::StorePtr {
                ptr: IrOperand::Value(a),
                src: IrOperand::Value(src + k),
            });
        }
    }


}
