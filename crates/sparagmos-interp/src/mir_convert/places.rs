//! Converter: place conversion.


use sparagmos_mir as mir;

use crate::ir;

use super::converter::*;
use super::errors::*;
use super::helpers::*;
use super::program::*;


impl<'a> Converter<'a> {
    // ----- places ----------------------------------------------------------

    pub(super) fn convert_place(
        &self,
        body: &mir::Body,
        place: &mir::Place,
    ) -> Result<ir::Place, ConvertError> {
        let mut out = ir::Place::from_index(place.local.index());
        // The type before each projection step, tracked the way
        // `place_ty` tracks it: a `Field` step whose base type
        // is a union converts to a union field step (its read
        // reinterprets the held field's bits; see
        // `ir::ProjectionElem::UnionField`), every other step to
        // the plain step it always was. Once the walk loses the
        // type, later steps convert plainly too.
        let mut ty = body
            .local_decls
            .get(place.local.index())
            .map(|decl| decl.ty.clone());
        // A coroutine place's `Downcast(v)` step followed by a
        // `Field(i)` step names state field `i` of state `v` in
        // the compiler's state-machine layout. When the driver
        // recorded the instance's slot map, the pair converts to
        // the single layout slot of that field (see
        // `mir::CoroutineSlots`): a state field keeps one storage
        // cell in every state that contains it, which the
        // positional field index does not express. A downcast not
        // followed by a field step, and every projection of an
        // instance with no recorded map, converts exactly as
        // before.
        let mut coroutine_downcast: Option<(u32, &mir::CoroutineSlots)> = None;
        for elem in &place.projection {
            while let Some(mir::Ty::Pat(inner, _)) = &ty {
                ty = Some(inner.as_ref().clone());
            }
            if let Some((variant, slots)) = coroutine_downcast.take() {
                if let mir::PlaceElem::Field(idx, field_ty) = elem {
                    if let Some(slot) = slots
                        .variants
                        .get(variant as usize)
                        .and_then(|v| v.get(*idx as usize))
                    {
                        out.projection.push(ir::ProjectionElem::Field(*slot as usize));
                        ty = Some(field_ty.clone());
                        continue;
                    }
                }
                out.projection.push(ir::ProjectionElem::Downcast(variant));
            }
            out.projection.push(match elem {
                mir::PlaceElem::Field(idx, field_ty) => {
                    let step = match &ty {
                        Some(mir::Ty::Adt(def, args)) if self.is_union(def) => {
                            match self.union_table_for(def, args) {
                                Some(table) => ir::ProjectionElem::UnionField {
                                    idx: *idx as usize,
                                    table,
                                },
                                None => ir::ProjectionElem::Field(*idx as usize),
                            }
                        }
                        _ => ir::ProjectionElem::Field(*idx as usize),
                    };
                    ty = Some(field_ty.clone());
                    step
                }
                mir::PlaceElem::Deref => {
                    ty = match ty {
                        Some(
                            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _),
                        ) => Some(*inner),
                        _ => None,
                    };
                    ir::ProjectionElem::Deref
                }
                mir::PlaceElem::Index(local) => {
                    ty = match ty {
                        Some(mir::Ty::Array(elem, _) | mir::Ty::Slice(elem)) => {
                            Some(*elem)
                        }
                        _ => None,
                    };
                    ir::ProjectionElem::Index(ir::Local(local.index()))
                }
                mir::PlaceElem::ConstantIndex {
                    offset, from_end, ..
                } => {
                    ty = match ty {
                        Some(mir::Ty::Array(elem, _) | mir::Ty::Slice(elem)) => {
                            Some(*elem)
                        }
                        _ => None,
                    };
                    ir::ProjectionElem::ConstantIndex {
                        offset: *offset as u64,
                        from_end: *from_end,
                    }
                }
                mir::PlaceElem::Downcast(_, variant) => {
                    if let Some(mir::Ty::Coroutine(def, args)) = &ty {
                        let key = crate::builtins::ty_tag(
                            &mir::Ty::Coroutine(*def, args.clone()),
                            self.names,
                        );
                        if let Some(slots) = self.coroutine_layouts.get(&key) {
                            coroutine_downcast = Some((*variant, slots));
                            // The downcast step is emitted (or
                            // folded into the following field step)
                            // at the top of the next iteration, so
                            // this arm never pushes.
                            continue;
                        }
                    }
                    ir::ProjectionElem::Downcast(*variant)
                }
                mir::PlaceElem::Subslice { from, to, from_end } => {
                    if let Some(mir::Ty::Array(elem, _)) = &ty {
                        ty = Some(mir::Ty::Slice(elem.clone()));
                    }
                    ir::ProjectionElem::Subslice {
                        from: *from,
                        to: *to,
                        from_end: *from_end,
                    }
                }
                other => {
                    return Err(ConvertError::UnsupportedProjection {
                        kind: other.kind_name().to_string(),
                    })
                }
            });
        }
        if let Some((variant, _)) = coroutine_downcast.take() {
            out.projection.push(ir::ProjectionElem::Downcast(variant));
        }
        Ok(out)
    }

    /// Whether a scalar shape has a pointer leaf (see the
    /// pointer exception in [`Self::deref_vector_shape`]).
    fn shape_contains_ptr(shape: &ir::ScalarShape) -> bool {
        match shape {
            ir::ScalarShape::Ptr
            | ir::ScalarShape::FnPtr
            | ir::ScalarShape::FatPtr(_) => true,
            ir::ScalarShape::Fields { fields, .. } => {
                fields.iter().any(|(_, s)| Self::shape_contains_ptr(s))
            }
            ir::ScalarShape::Array(elem, _) => Self::shape_contains_ptr(elem),
            ir::ScalarShape::AdtField { inner, .. } => Self::shape_contains_ptr(inner),
            ir::ScalarShape::NicheEnum { inner, .. } => Self::shape_contains_ptr(inner),
            ir::ScalarShape::AdtFields { fields, .. } => fields
                .iter()
                .any(|(_, _, s)| Self::shape_contains_ptr(s)),
            ir::ScalarShape::NicheEnumFields { fields, .. } => fields
                .iter()
                .any(|(_, _, s)| Self::shape_contains_ptr(s)),
            ir::ScalarShape::TaggedEnum { variants, .. } => variants
                .iter()
                .any(|v| v.fields.iter().any(|(_, _, s)| Self::shape_contains_ptr(s))),
            _ => false,
        }
    }

    /// The scalar shape of a dereference through `place`:
    /// the shape a shaped load / store of the place runs
    /// through. The dereference may sit below field steps (a
    /// vector stored in a wrapper struct behind a pointer:
    /// `(*p).0` where the field is the vector or its storage
    /// array; the byte inside a tag wrapper: `(*p).0` over a
    /// control byte), because the pointee behind such a
    /// pointer can be stored as raw bytes the plain projection
    /// cannot enter. `None` for any other place: no
    /// dereference, or a place type with no scalar shape (its
    /// loads and stores are plain whole-value ones).
    pub(super) fn deref_vector_shape(
        &self,
        body: &mir::Body,
        place: &mir::Place,
    ) -> Option<ir::ScalarShape> {
        if !place
            .projection
            .iter()
            .any(|e| matches!(e, mir::PlaceElem::Deref))
        {
            return None;
        }
        let ty = self.place_ty(body, place)?;
        let shape = self.scalar_shape(&ty, 0)?;
        // Any scalar shape qualifies, not only vector (array)
        // ones: a pointee stored as raw bytes — a control byte
        // in the static empty group is the case at hand — can
        // sit below a dereference under a field step of any
        // scalar-shaped type (the byte inside its tag wrapper),
        // and the plain projection cannot enter a byte string.
        // For a pointee stored whole the shaped load/store is
        // the canonicalising round trip, so the broader gate
        // changes nothing there.
        //
        // A shape with a pointer leaf is the exception: the
        // round trip through a pointer shape flattens a
        // reference to its synthetic address and constructs an
        // address-only pointer back, which is not the stored
        // value's identity (a place or allocation reference
        // denotes its target; the address-only form denotes
        // nothing readable). Pointers in this model are stored
        // whole, never as bytes, so there is nothing for a
        // shaped load/store to assemble — the plain projection
        // is the only honest path.
        if Self::shape_contains_ptr(&shape) {
            return None;
        }
        Some(shape)
    }

    /// Best-effort type of a place (used only to type constants in
    /// operand position; `None` when it cannot be derived).
    pub(super) fn place_ty(&self, body: &mir::Body, place: &mir::Place) -> Option<mir::Ty> {
        let mut ty = body.local_decls.get(place.local.index())?.ty.clone();
        for elem in &place.projection {
            // A pattern type is its base type for every
            // projection step.
            while let mir::Ty::Pat(inner, _) = ty {
                ty = *inner;
            }
            ty = match elem {
                mir::PlaceElem::Field(_, t) => t.clone(),
                mir::PlaceElem::Deref => match ty {
                    mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => *inner,
                    _ => return None,
                },
                mir::PlaceElem::Index(_) | mir::PlaceElem::ConstantIndex { .. } => match ty {
                    mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => *elem,
                    _ => return None,
                },
                mir::PlaceElem::Downcast(..) => ty,
                mir::PlaceElem::Subslice { .. } => match ty {
                    mir::Ty::Array(elem, _) => mir::Ty::Slice(elem),
                    t => t,
                },
                mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => t.clone(),
            };
        }
        Some(ty)
    }

    /// Whether a place's declared type is the coroutine
    /// transform's context cell type, `std::future::ResumeTy`.
    pub(super) fn is_resume_ty_place(&self, body: &mir::Body, place: &mir::Place) -> bool {
        match self.place_ty(body, place) {
            Some(mir::Ty::Adt(def, _)) => {
                self.def_display_name(&def) == "std::future::ResumeTy"
            }
            _ => false,
        }
    }

    /// The discriminant table of the `Self` type of a comparison
    /// call: the callee's first generic argument, falling back to
    /// the receiver operand's type (a constant receiver carries no
    /// place type). `None` unless that type is an enum with a
    /// recorded table.
    pub(super) fn comparison_discriminants(
        &self,
        body: &mir::Body,
        args: &[mir::Operand],
        raw_generic_args: &[mir::Ty],
    ) -> Option<Vec<i128>> {
        let self_ty = raw_generic_args.first().cloned().or_else(|| {
            args.first()
                .and_then(|a| self.operand_place_ty(body, a))
        })?;
        let self_ty = match self_ty {
            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => *inner,
            t => t,
        };
        match self_ty {
            mir::Ty::Adt(def, _) => {
                let table = self.discriminants.get(&def)?;
                if table.is_empty() {
                    None
                } else {
                    Some(table.clone())
                }
            }
            _ => None,
        }
    }

    pub(super) fn operand_place_ty(&self, body: &mir::Body, op: &mir::Operand) -> Option<mir::Ty> {
        match op.place() {
            Some(place) => self.place_ty(body, place),
            None => None,
        }
    }

    /// The element size of a `BinaryOp::Offset`, from the pointer
    /// operand's type (its place type, else the result type — the
    /// operation's result is the pointer type itself), when the
    /// pointee's size is exactly computable from the type alone
    /// (see [`mir_ty_size`]). `None` leaves the size to execution,
    /// which derives it from the executing body's declared type of
    /// the pointer operand.
    pub(super) fn offset_elem_size(
        &self,
        body: &mir::Body,
        pair: &(mir::Operand, mir::Operand),
        dest_ty: Option<&mir::Ty>,
    ) -> Option<u64> {
        let ty = self
            .operand_place_ty(body, &pair.0)
            .or_else(|| dest_ty.cloned())?;
        let pointee = match ty {
            mir::Ty::RawPtr(inner, _) => *inner,
            mir::Ty::Ref(_, inner, _) => *inner,
            _ => return None,
        };
        if let Some(size) = mir_ty_size(&pointee) {
            return Some(size);
        }
        // A composite pointee's size is its recorded layout: an
        // ADT's size lives in the layout table (the
        // `MaybeUninit<Chunk>` pointees of the byte-swap
        // machinery are the canonical case), and a tuple's size
        // is its fields' layouts composed — a hashbrown bucket
        // pointer's pointee is a `(K, V)` tuple whose fields may
        // themselves be ADTs.
        self.ty_layout(&pointee).map(|(size, _)| size)
    }

    /// The source pointee size for a pointer-to-pointer cast
    /// that needs its projection index rewritten in byte units
    /// (`Rvalue::PtrCastUnits`): both pointees' layouts are
    /// known, they differ, and the source is thin. `None` for
    /// every other cast, which keeps its existing form.
    pub(super) fn cast_units_from_size(
        &self,
        ctx: &Option<mir::Ty>,
        target: &mir::Ty,
    ) -> Option<u64> {
        let src = ptr_pointee(ctx.as_ref()?)?;
        let dst = ptr_pointee(target)?;
        if matches!(
            src,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ) {
            return None;
        }
        let from = self.ty_layout(src).map(|(size, _)| size)?;
        let to = self.ty_layout(dst).map(|(size, _)| size)?;
        if from != to && from > 0 {
            Some(from)
        } else {
            None
        }
    }

    /// The native `(size, align)` of a type whose layout is
    /// structurally known or recorded in the layout table:
    /// scalars by their width, arrays and tuples by composing
    /// their elements' layouts (declaration order, each field
    /// aligned to its own alignment, the total rounded up to
    /// the widest), ADTs by their recorded layout. `None` when
    /// any component's layout is unknown.
    pub(super) fn ty_layout(&self, ty: &mir::Ty) -> Option<(u64, u64)> {
        use mir::Ty as T;
        Some(match ty {
            T::Bool => (1, 1),
            T::Char => (4, 4),
            T::Int(t) => {
                let b = u64::from(t.bits() / 8);
                (b, b)
            }
            T::Uint(t) => {
                let b = u64::from(t.bits() / 8);
                (b, b)
            }
            T::Float(t) => {
                let b = u64::from(t.bits() / 8);
                (b, b)
            }
            T::Array(elem, mir::ArrayLen::Known(n)) => {
                let (es, ea) = self.ty_layout(elem)?;
                (es.checked_mul(*n)?, ea)
            }
            T::RawPtr(pointee, _) | T::Ref(_, pointee, _) => {
                if matches!(
                    &**pointee,
                    T::Slice(_) | T::Str | T::Dynamic(..)
                ) {
                    (16, 8)
                } else {
                    (8, 8)
                }
            }
            T::FnPtr(_) => (8, 8),
            T::Tuple(ts) => {
                let mut size = 0u64;
                let mut align = 1u64;
                for f in ts {
                    let (fs, fa) = self.ty_layout(f)?;
                    align = align.max(fa);
                    size = size.checked_add(fa - 1)? / fa * fa;
                    size = size.checked_add(fs)?;
                }
                (size.checked_add(align - 1)? / align * align, align)
            }
            T::Adt(def, args) => {
                let tag = crate::builtins::ty_tag(
                    &mir::Ty::Adt(*def, args.clone()),
                    self.names,
                );
                *self.adt_layouts.get(&tag)?
            }
            T::Pat(inner, _) => self.ty_layout(inner)?,
            T::Never => (0, 1),
            _ => return None,
        })
    }

}
