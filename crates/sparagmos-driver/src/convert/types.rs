//! Type conversion, ADT shape/layout recording, and place conversion.

use super::*;

impl Converter {
    pub(crate) fn place(&mut self, p: &mir::Place) -> sm::Place {
        let mut place = sm::Place::from_local(sm::Local(p.local as u32));
        for elem in &p.projection {
            place.projection.push(self.projection_elem(elem));
        }
        place
    }

    fn projection_elem(&mut self, elem: &mir::ProjectionElem) -> sm::PlaceElem {
        use mir::ProjectionElem as P;
        match elem {
            P::Deref => sm::PlaceElem::Deref,
            P::Field(idx, ty) => sm::PlaceElem::Field(*idx as u32, self.ty(ty)),
            P::Index(local) => sm::PlaceElem::Index(sm::Local(*local as u32)),
            P::ConstantIndex {
                offset,
                min_length,
                from_end,
            } => sm::PlaceElem::ConstantIndex {
                offset: *offset,
                min_length: *min_length,
                from_end: *from_end,
            },
            P::Subslice { from, to, from_end } => sm::PlaceElem::Subslice {
                from: *from,
                to: *to,
                from_end: *from_end,
            },
            P::Downcast(variant) => sm::PlaceElem::Downcast(None, variant.to_index() as u32),
            P::OpaqueCast(ty) => sm::PlaceElem::OpaqueCast(self.ty(ty)),
        }
    }

    pub(crate) fn type_args(&mut self, args: &GenericArgs) -> Vec<sm::Ty> {
        args.0
            .iter()
            .filter_map(|a| match a {
                GenericArgKind::Type(t) => Some(self.ty(t)),
                GenericArgKind::Const(_) | GenericArgKind::Lifetime(_) => None,
            })
            .collect()
    }

    pub fn ty(&mut self, ty: &rustc_public::ty::Ty) -> sm::Ty {
        let Some(kind) = guarded_ty_kind(ty) else {
            // An unsafe binder type erases to the type it
            // wraps (see `unsafe_binder_inner_ty`): the binder
            // is transparent at the value level, so the body
            // converts with the wrapped type in its place.
            if let Some(inner) = unsafe_binder_inner_ty(ty) {
                return self.ty(&inner);
            }
            // A coroutine-closure type, whose kind the bridge
            // cannot hand over (see `guarded_ty_kind`): neither
            // its structure nor a layout can be asked. The value
            // it types is created by a coroutine-closure
            // aggregate, which carries the closure's definition
            // and arguments itself, so the body still converts
            // with the type recorded as the unit placeholder.
            return sm::Ty::unit();
        };
        if !crate::reach::ty_within_expanded_size(ty, crate::reach::MAX_TY_EXPANDED_SIZE) {
            // The type's expanded tree is past the conversion
            // size limit (see `MAX_TY_EXPANDED_SIZE`): refuse it
            // here, before any of it is materialized, and let
            // the body carrying it fail with this reason like
            // any other unsupported construct.
            self.err(format!(
                "expanded type tree exceeds the conversion size limit ({} nodes)",
                crate::reach::MAX_TY_EXPANDED_SIZE
            ));
            return sm::Ty::Error;
        }
        let converted = self.ty_inner(ty);
        match &kind {
            TyKind::RigidTy(RigidTy::Adt(def, args)) => {
                self.record_layout(ty, def, args, &converted);
                self.record_debug_shape(def, args, &converted);
                self.record_instance_shape(ty, def, args, &converted);
            }
            TyKind::RigidTy(RigidTy::Closure(..))
            | TyKind::RigidTy(RigidTy::Coroutine(..))
            | TyKind::RigidTy(RigidTy::CoroutineClosure(..)) => {
                self.record_coroutine_layout(ty, &converted);
            }
            _ => {}
        }
        if let TyKind::RigidTy(RigidTy::Coroutine(..)) = &kind {
            self.record_coroutine_slots(ty, &converted);
        }
        if matches!(
            kind,
            TyKind::RigidTy(RigidTy::Closure(..))
                | TyKind::RigidTy(RigidTy::Coroutine(..))
        ) {
            self.record_env_shape(ty, &kind, &converted);
        }
        converted
    }

    fn ty_inner(&mut self, ty: &rustc_public::ty::Ty) -> sm::Ty {
        match ty.kind() {
            TyKind::RigidTy(r) => self.rigid_ty(ty, &r),
            TyKind::Alias(kind, alias) => {
                let k = match kind {
                    SmirAliasKind::Projection => sm::AliasKind::Projection,
                    SmirAliasKind::Opaque => sm::AliasKind::Opaque,
                    SmirAliasKind::Free => sm::AliasKind::Free,
                    SmirAliasKind::Inherent => sm::AliasKind::Inherent,
                };
                let d = self.crate_def_id(&alias.def_id);
                sm::Ty::Alias(k, d, self.type_args(&alias.args))
            }
            TyKind::Param(p) => sm::Ty::Param(p.index, p.name.clone()),
            TyKind::Bound(idx, _) => sm::Ty::Bound(idx as u32),
        }
    }

    /// Record the declared shape of an ADT under its converted
    /// `DefId`, once per ADT (see `sm::AdtShape`). The entry is
    /// inserted before the field types are converted so that a
    /// recursive ADT terminates: the recursive reference observes
    /// the placeholder (no variants), which no consumer mistakes
    /// for a complete shape.
    ///
    /// Shape recording is auxiliary to the conversion of the body
    /// in progress: converting a field type can hit constructs the
    /// body itself never uses (pattern types on an unrelated std
    /// ADT's fields, say), so the current body's error list is
    /// saved and restored around the field conversions — a body
    /// that genuinely uses such a type still reports the error
    /// through its own conversion path.
    fn record_shape(&mut self, def: &AdtDef, id: sm::AdtDefId) {
        if self.adt_shapes.contains_key(&id) {
            return;
        }
        let kind = match def.kind() {
            AdtKind::Struct => sm::AdtShapeKind::Struct,
            AdtKind::Enum => sm::AdtShapeKind::Enum,
            AdtKind::Union => sm::AdtShapeKind::Union,
        };
        self.adt_shapes.insert(
            id,
            sm::AdtShape {
                kind,
                variants: Vec::new(),
            },
        );
        let saved_errors = std::mem::take(&mut self.errors);
        let mut variants = Vec::new();
        for variant in def.variants() {
            let mut fields = Vec::new();
            for field in variant.fields() {
                let fty = field.ty();
                fields.push(self.ty(&fty));
            }
            variants.push(fields);
        }
        self.errors = saved_errors;
        self.adt_shapes.insert(id, sm::AdtShape { kind, variants });
    }

    /// Record the discriminant values of an enum ADT under its
    /// converted `DefId`, once per ADT. Structs and unions have no
    /// `Discriminant` semantics to record (a struct value is its
    /// single variant; unions are not modelled as enum values).
    fn record_discriminants(&mut self, def: &AdtDef, id: sm::AdtDefId) {
        if self.adt_discriminants.contains_key(&id) {
            return;
        }
        if def.kind() != AdtKind::Enum {
            return;
        }
        let values: Vec<i128> = (0..def.num_variants())
            .map(|i| {
                let d = def.discriminant_for_variant(VariantIdx::to_val(i));
                discr_value(&d).unwrap_or(d.val as i128)
            })
            .collect();
        self.adt_discriminants.insert(id, values);
    }

    /// Record the derived-`Debug` shape of one monomorphic ADT
    /// instance under its rendered instance name, once per distinct
    /// instance (see `sm::AdtDebugShapes`). Only instances with
    /// no remaining generic parameters get an entry:
    /// a parameter would render into a field tag that names no real
    /// type, and the instance's own tag would collide across
    /// instantiations.
    ///
    /// The field types are taken from the definition (in terms of
    /// its generic parameters, exactly like `record_shape`) and the
    /// instance's converted arguments are substituted into them
    /// here, in converted form. Substituting through the compiler
    /// instead (`FieldDef::ty_with_args`) is not an option: it can
    /// ask the compiler to normalize a projection it cannot
    /// normalize, which panics inside the compiler and poisons the
    /// query it runs in.
    ///
    /// Two conversion losses are repaired before the substitution.
    /// A parameter index counts every parameter kind (lifetimes
    /// and consts included) while the converted arguments keep
    /// only the type arguments, so the arguments are first aligned
    /// to the parameter positions (a placeholder sits at the other
    /// kinds' positions, which no type parameter ever occupies).
    /// And an array length that is a const parameter converts to
    /// a nameless parameter form, so the instance's const argument
    /// is resolved back into it by a parallel walk of the
    /// definition's field type (`resolve_param_array_lens`).
    ///
    /// Shape recording is auxiliary to the conversion in progress,
    /// exactly like `record_shape`: the current body's error list
    /// is saved and restored around the field conversions, and a
    /// placeholder entry is inserted before the fields are
    /// converted so a recursive ADT terminates. The unfolding is
    /// also depth-capped as a belt-and-braces bound (instances past
    /// the cap stay unsupported, like any other unrecorded
    /// instance).
    fn record_debug_shape(&mut self, def: &AdtDef, args: &GenericArgs, converted: &sm::Ty) {
        let kind = match def.kind() {
            AdtKind::Struct => sm::AdtShapeKind::Struct,
            AdtKind::Enum => sm::AdtShapeKind::Enum,
            AdtKind::Union => sm::AdtShapeKind::Union,
        };
        if ty_has_param(converted) || self.debug_shape_depth >= 16 {
            return;
        }
        let key = self.layout_key(converted);
        if self.adt_debug_shapes.contains_key(&key) {
            return;
        }
        let sm::Ty::Adt(_, instance_args) = converted else {
            return;
        };
        let instance_args = instance_args.clone();
        let mut converted_types = instance_args.iter();
        let aligned_args: Vec<sm::Ty> = args
            .0
            .iter()
            .enumerate()
            .map(|(i, a)| match a {
                GenericArgKind::Type(_) => converted_types
                    .next()
                    .cloned()
                    .unwrap_or(sm::Ty::Error),
                GenericArgKind::Const(_) | GenericArgKind::Lifetime(_) => {
                    sm::Ty::Param(i as u32, String::new())
                }
            })
            .collect();
        self.adt_debug_shapes.insert(
            key.clone(),
            sm::AdtDebugShape {
                kind,
                variants: Vec::new(),
            },
        );
        self.debug_shape_depth += 1;
        let saved_errors = std::mem::take(&mut self.errors);
        let mut variants = Vec::new();
        for variant in def.variants() {
            let mut fields = Vec::new();
            for field in variant.fields() {
                let fty = field.ty();
                let converted_field = self.ty(&fty);
                let converted_field =
                    resolve_param_array_lens(&fty, &converted_field, args);
                let substituted = substitute_params(&converted_field, &aligned_args);
                let tag = self.layout_key(&substituted);
                // A tuple field's recorded name is its position
                // written out as a number; anything else is a
                // declared field name.
                let name = if field.name.bytes().all(|b| b.is_ascii_digit())
                    && !field.name.is_empty()
                {
                    None
                } else {
                    Some(field.name.clone())
                };
                fields.push(sm::AdtDebugField { name, ty: tag });
            }
            variants.push(sm::AdtDebugVariant {
                name: variant.name(),
                fields,
            });
        }
        self.errors = saved_errors;
        self.debug_shape_depth -= 1;
        self.adt_debug_shapes
            .insert(key, sm::AdtDebugShape { kind, variants });
    }

    /// Record the compiler-computed layout of one monomorphic ADT
    /// instance under its rendered instance name, once per distinct
    /// instance. The layout query is the compiler's own, so
    /// `repr(C)` / `repr(packed)` / `repr(align)` and enum tag
    /// placement are reflected exactly. Instances the query cannot
    /// answer (generic parameters among the arguments, unsized
    /// types, trait objects) are skipped: they stay explicitly
    /// unsupported in the interpreter.
    fn record_layout(
        &mut self,
        ty: &rustc_public::ty::Ty,
        def: &AdtDef,
        args: &rustc_public::ty::GenericArgs,
        converted: &sm::Ty,
    ) {
        if !self.layout_attempted_tys.insert(*ty) {
            return;
        }
        let key = self.layout_key(converted);
        if !self.layout_attempted.insert(key.clone()) {
            return;
        }
        self.record_niche(ty, def, args, &key);
        // The maybe-unsized screen was this query's only guard,
        // but it answers conservatively for ADTs whose fields
        // mention a type alias — `Cow<str>`'s `Owned` field is
        // `<str as ToOwned>::Owned` — screening out instances
        // the compiler lays out without doubt (the niche
        // recorder above already relies on exactly that). The
        // query below is contained, so attempting it for a
        // screened instance whose unsizedness is only suspected
        // costs at most the contained failure the unscreened
        // path already tolerates: the instance simply gets no
        // entry and stays unsupported.
        // One class must not be asked at all: an instance the
        // sizedness walk proves genuinely unsized — a slice,
        // `str` or trait object reachable through its fields,
        // at any depth, with the instance's arguments composed
        // in — is ill-formed for layout, and asking does not
        // fail inside the query. The layout computation records
        // a delayed bug instead, which the compiler flushes as
        // an abort when the run ends, outside every containment
        // the driver has. The niche recorder's screen already
        // judges by exactly this walk; the layout table follows
        // it. (An earlier form of this guard looked only at the
        // instance's own declared field types, which let
        // wrappers through — an enum over ill-formed structs,
        // a generic ADT instantiated with `str` — and aborted
        // the whole compilation for them.)
        // The layout query is not total: for some types that can
        // appear in converted bodies (ill-formed bounds, types
        // the layout computation itself rejects) it panics
        // inside the compiler. The table is auxiliary — a
        // failed query must never fail the conversion — so the
        // query is contained: the instance simply gets no entry
        // and stays unsupported. (The panic is silent under the
        // driver's panic-capture hook, which is installed while
        // the compiler runs.)
        if adt_sizedness(def, args, 0) == Sizedness::Unsized {
            self.record_dst_layout(def, args, &key);
            return;
        }
        let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        let Ok(Ok(layout)) = layout else {
            return;
        };
        let shape = layout.shape();
        if shape.is_unsized() {
            return;
        }
        self.adt_layouts.insert(
            key.clone(),
            (shape.size.bytes() as u64, shape.abi_align),
        );
        self.record_field_layouts(def, args, &shape, &key);
        self.record_tag_layout(def, &shape, &key);
    }

    /// Record the runtime layout descriptor of one genuinely
    /// unsized ADT instance (see `sm::AdtDstLayout`): the sized
    /// prefix's extent and the one unsized field, from which
    /// the interpreter composes the instance's dynamic layout.
    /// The instance's own layout is never queried -- a proven
    /// unsized instance is ill-formed for the layout query (see
    /// `record_layout`) -- so the prefix is computed here from
    /// the sized fields' own contained layout queries, laid out
    /// the way the compiler lays them out among themselves: in
    /// declaration order with alignment padding for a `repr(C)`
    /// struct, and as a plain sum for the default
    /// representation (whose layout orders the sized fields by
    /// decreasing alignment, which inserts no padding between
    /// them), each field's alignment capped by the repr's pack.
    /// Only a struct or union with exactly one unsized field
    /// gets a descriptor; every other shape stays unsupported.
    fn record_dst_layout(
        &mut self,
        def: &AdtDef,
        args: &rustc_public::ty::GenericArgs,
        key: &str,
    ) {
        use rustc_public::ty::AdtKind;
        let kind = def.kind();
        if kind != AdtKind::Struct && kind != AdtKind::Union {
            return;
        }
        if def.num_variants() != 1 {
            return;
        }
        let Some(variant) =
            def.variant(rustc_public::ty::VariantIdx::to_val(0))
        else {
            return;
        };
        let fields = variant.fields();
        let repr = def.repr();
        if repr.flags.is_simd {
            return;
        }
        let pack = repr.pack;
        let mut prefix_end: u64 = 0;
        let mut prefix_align: u64 = 1;
        let mut tail: Option<(usize, rustc_public::ty::Ty)> = None;
        for (index, field) in fields.iter().enumerate() {
            let fty = field.ty_with_args(args);
            // Convert the field type as well: its own recording
            // hooks (a nested DST instance's descriptor among
            // them) hang off the conversion.
            let _ = self.ty(&fty);
            match ty_sizedness(&field.ty(), Some(args), 1) {
                Sizedness::Unsized => {
                    if tail.is_some() {
                        return;
                    }
                    tail = Some((index, fty));
                }
                Sizedness::Sized => {
                    // The same coroutine screen as
                    // `record_field_layouts`: such a field's
                    // layout query is not total.
                    if ty_mentions_coroutine(&fty) {
                        return;
                    }
                    let queried = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| fty.layout()),
                    );
                    let Ok(Ok(layout)) = queried else {
                        return;
                    };
                    let shape = layout.shape();
                    if shape.is_unsized() {
                        return;
                    }
                    let size = shape.size.bytes() as u64;
                    let eff_align = match pack {
                        Some(cap) => shape.abi_align.min(cap),
                        None => shape.abi_align,
                    };
                    prefix_align = prefix_align.max(eff_align);
                    if kind == AdtKind::Union {
                        prefix_end = prefix_end.max(size);
                    } else if repr.flags.is_c {
                        prefix_end =
                            (prefix_end + eff_align - 1) & !(eff_align - 1);
                        prefix_end += size;
                    } else {
                        prefix_end += size;
                    }
                }
                Sizedness::Unknown => return,
            }
        }
        let Some((tail_index, tail_ty)) = tail else {
            return;
        };
        if kind == AdtKind::Struct && tail_index + 1 != fields.len() {
            return;
        }
        if let Some(align) = repr.align {
            prefix_align = prefix_align.max(align);
        }
        let tail_converted = self.ty(&tail_ty);
        let tail_tag = self.layout_key(&tail_converted);
        self.adt_dst_layouts.insert(
            key.to_string(),
            sm::AdtDstLayout {
                prefix_end,
                prefix_align,
                pack,
                tail_field: tail_index as u32,
                tail_ty: tail_tag,
                union: kind == AdtKind::Union,
            },
        );
    }

    /// Record the per-variant field byte layout of one ADT
    /// instance (see `sm::AdtFieldLayouts`) from the layout
    /// `record_layout` already queried. Variant field offsets
    /// come from the layout shape itself; each field's size
    /// comes from the field type's own contained layout query.
    /// An instance whose fields cannot all be sized gets no
    /// entry, exactly as a failed whole-layout query leaves
    /// `adt_layouts` without one. Unions get no entry: their
    /// fields share one offset and the interpreter reads them
    /// through its own reinterpretation machinery.
    fn record_field_layouts(
        &mut self,
        def: &AdtDef,
        args: &rustc_public::ty::GenericArgs,
        shape: &rustc_public::abi::LayoutShape,
        key: &str,
    ) {
        use rustc_public::abi::{FieldsShape, VariantsShape};
        if def.kind() == AdtKind::Union {
            return;
        }
        // (variant index, field offsets in declaration order)
        let variant_offsets: Vec<(usize, Vec<u64>)> = match &shape.variants {
            VariantsShape::Empty => return,
            VariantsShape::Single { index } => {
                let offsets = match &shape.fields {
                    FieldsShape::Arbitrary { offsets } => {
                        offsets.iter().map(|o| o.bytes() as u64).collect()
                    }
                    FieldsShape::Array { stride, count } => (0..*count)
                        .map(|i| (i as u64) * (stride.bytes() as u64))
                        .collect(),
                    FieldsShape::Primitive | FieldsShape::Union(_) => return,
                };
                vec![(index.to_index(), offsets)]
            }
            VariantsShape::Multiple { variants, .. } => variants
                .iter()
                .enumerate()
                .map(|(v, variant)| {
                    (
                        v,
                        variant
                            .offsets
                            .iter()
                            .map(|o| o.bytes() as u64)
                            .collect::<Vec<u64>>(),
                    )
                })
                .collect(),
        };
        let width = variant_offsets
            .iter()
            .map(|(v, _)| v + 1)
            .max()
            .unwrap_or(0);
        let mut out: Vec<Vec<(u64, u64)>> = vec![Vec::new(); width];
        for (v, offsets) in &variant_offsets {
            let Some(variant) = def.variant(rustc_public::ty::VariantIdx::to_val(*v))
            else {
                return;
            };
            let fields = variant.fields();
            if fields.len() != offsets.len() {
                return;
            }
            let mut row = Vec::with_capacity(offsets.len());
            for (field, &off) in fields.iter().zip(offsets.iter()) {
                let fty = field.ty_with_args(args);
                // A tuple field type's own storage layout is
                // recorded as well, through the conversion
                // hook that records every tuple type a body
                // mentions: the interpreter's scalar shape of
                // the field reads the tuple's fields at the
                // recorded offsets, and the instantiated
                // tuple type may occur in no body.
                if matches!(
                    guarded_ty_kind(&fty),
                    Some(TyKind::RigidTy(RigidTy::Tuple(_)))
                ) {
                    let _ = self.ty(&fty);
                }
                // A field type mentioning a coroutine is never
                // queried: computing such a type's layout can
                // drive the compiler into the coroutine's own
                // destructor-shim construction, which is not
                // total — an `async_drop_in_place` coroutine's
                // layout query aborts the whole compilation
                // with an internal error the containment cannot
                // catch. The interpreter's byte view loses
                // nothing: a coroutine value is not an aggregate
                // of fields it could descend into.
                if ty_mentions_coroutine(&fty) {
                    return;
                }
                let queried = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| fty.layout()),
                );
                let Ok(Ok(flayout)) = queried else { return };
                row.push((off, flayout.shape().size.bytes() as u64));
            }
            out[*v] = row;
        }
        self.adt_field_layouts.insert(key.to_string(), out);
    }

    /// Record the direct discriminant tag layout of one
    /// enum instance (see `sm::AdtTagLayouts`) from the
    /// layout `record_layout` already queried. Only a
    /// multi-variant enum with the direct tag encoding
    /// stores a tag: its offset is the tag field's offset in
    /// the layout's top-level field list, and its width is
    /// the tag scalar's own (the stored tag can be narrower
    /// than the enum's discriminant type). Each variant's
    /// discriminant value is recorded as the bit pattern
    /// the tag stores. An instance whose discriminant
    /// values cannot be read gets no entry.
    fn record_tag_layout(
        &mut self,
        def: &AdtDef,
        shape: &rustc_public::abi::LayoutShape,
        key: &str,
    ) {
        use rustc_public::abi::{FieldsShape, TagEncoding, VariantsShape};
        if def.kind() != AdtKind::Enum {
            return;
        }
        let VariantsShape::Multiple {
            tag,
            tag_encoding: TagEncoding::Direct,
            tag_field,
            ..
        } = &shape.variants
        else {
            return;
        };
        let FieldsShape::Arbitrary { offsets } = &shape.fields else {
            return;
        };
        let Some(tag_off) = offsets.get(*tag_field) else {
            return;
        };
        let Some((tag_size, _)) = tag_scalar_size_signed(tag) else {
            return;
        };
        if tag_size == 0 || tag_size > 16 {
            return;
        }
        let mut discriminants = Vec::with_capacity(def.num_variants());
        for i in 0..def.num_variants() {
            let d = def
                .discriminant_for_variant(rustc_public::ty::VariantIdx::to_val(i));
            let Some(v) = discr_value(&d) else { return };
            discriminants.push(v as u128);
        }
        self.adt_tag_layouts.insert(
            key.to_string(),
            sm::EnumTag {
                offset: tag_off.bytes() as u64,
                size: tag_size as u64,
                discriminants,
            },
        );
    }

    /// Record the niche discriminant encoding of one
    /// monomorphic enum instance (see `sm::AdtNiches`); the
    /// caller dedups through `layout_attempted`, so this runs
    /// once per distinct instance.
    ///
    /// The recording runs its own contained layout query
    /// instead of sharing `record_layout`'s result, because
    /// the instances that matter most never reach that
    /// result: the maybe-unsized screen that guards the
    /// layout table answers conservatively for ADTs whose
    /// fields mention a type alias, which is exactly the
    /// `Option<NonZero<T>>` family this table exists for.
    /// The compiler's layout query has no such doubt and
    /// answers for those instances directly (and is
    /// memoized, so asking again for enums that pass the
    /// screen costs nothing measurable). Only multi-variant
    /// enums can carry a niche encoding at all.
    fn record_niche(
        &mut self,
        ty: &rustc_public::ty::Ty,
        def: &AdtDef,
        args: &GenericArgs,
        key: &str,
    ) {
        if def.kind() != AdtKind::Enum || def.num_variants() < 2 {
            return;
        }
        // Genuinely unsized instances are skipped without
        // querying: forcing a layout for one (the
        // trivial-bounds test cases, whose fields are slices
        // under assumed `Sized` bounds) trips the compiler's
        // own delayed bugs. Instances whose sizedness the
        // walk cannot prove — the alias-carrying
        // `Option<NonZero<T>>` family — go through the
        // contained query below, which is the only way to
        // learn their encoding; an uncomputable layout
        // simply records nothing.
        if adt_sizedness(def, args, 0) == Sizedness::Unsized {
            return;
        }
        let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        let Ok(Ok(layout)) = layout else {
            return;
        };
        let shape = layout.shape();
        if shape.is_unsized() {
            return;
        }
        let VariantsShape::Multiple {
            tag,
            tag_encoding:
                TagEncoding::Niche {
                    untagged_variant,
                    niche_variants,
                    niche_start,
                },
            tag_field,
            ..
        } = &shape.variants
        else {
            return;
        };
        let offsets = match &shape.fields {
            FieldsShape::Arbitrary { offsets } => {
                offsets.iter().map(|o| o.bytes()).collect::<Vec<_>>()
            }
            _ => return,
        };
        let Some((tag_bytes, _)) = tag_scalar_size_signed(tag) else {
            return;
        };
        let Some(offset) = offsets.get(*tag_field) else {
            return;
        };
        self.adt_niches.insert(
            key.to_string(),
            sm::EnumNiche {
                untagged_variant: untagged_variant.to_index() as u32,
                niche_variants: (
                    niche_variants.start().to_index() as u32,
                    niche_variants.end().to_index() as u32,
                ),
                niche_start: *niche_start,
                tag_field: *tag_field as u32,
                tag_bits: (tag_bytes * 8) as u32,
                niche_offset: *offset as u64,
            },
        );
    }

    /// Record the field types of one monomorphic ADT instance
    /// whose generic shape mentions a type alias or a const
    /// generic array length (see `sm::AdtInstanceShapes`), once
    /// per distinct instance.
    ///
    /// The generic shape's field types are the definition's own,
    /// so a projection alias among them reaches the interpreter
    /// unresolved — the `NonZero<T>` field is declared as
    /// `<T as ZeroablePrimitive>::NonZeroInner`, and the
    /// interpreter has no trait solver — and an array length
    /// that is a const generic parameter (`[T; N]`, the field
    /// of every `core::simd` vector) reaches it unresolvable by
    /// substitution: the converter's instantiated types drop
    /// const arguments, so no substitution can ever produce the
    /// concrete length. The compiler resolves both:
    /// `FieldDef::ty_with_args` instantiates a field's declared
    /// type with the instance's arguments and normalizes it in a
    /// monomorphic environment, which is exactly the resolution
    /// the shape lacks. The query is not total: on a projection
    /// that mentions a generic parameter it fails *fatally* —
    /// an internal compiler error that no catch_unwind
    /// contains — because the codegen typing environment's
    /// parameter environment is empty. Instances are therefore
    /// screened before any query: `instance_mentions_param`
    /// rejects instances whose arguments mention a parameter,
    /// and a contained layout query gates the rest — an
    /// instance whose layout the compiler cannot compute has
    /// field projections that only a where-clause on the
    /// definition justifies, which normalize just as fatally.
    /// Unrecorded instances keep the generic shape's
    /// behaviour downstream.
    ///
    /// The recording is auxiliary, like `record_shape`'s: the
    /// body's error list is saved and restored around the field
    /// conversions, and any error growth abandons the entry — a
    /// body that genuinely uses such a type still reports through
    /// its own conversion path.
    fn record_instance_shape(
        &mut self,
        ty: &rustc_public::ty::Ty,
        def: &AdtDef,
        args: &GenericArgs,
        converted: &sm::Ty,
    ) {
        let sm::Ty::Adt(id, _) = converted else {
            return;
        };
        let key = self.layout_key(converted);
        if !self.instance_shape_attempted.insert(key.clone()) {
            return;
        }
        let Some(shape) = self.adt_shapes.get(id) else {
            return;
        };
        if !shape
            .variants
            .iter()
            .flatten()
            .any(|t| ty_mentions_alias(t) || ty_mentions_param_len(t))
        {
            return;
        }
        // `ty_with_args` normalizes in a codegen typing
        // environment whose parameter environment is empty: a
        // projection that mentions a generic parameter — a
        // type or const parameter, or a free region — cannot
        // normalize there, and the failure is *fatal* to the
        // whole compilation, not a catchable panic (the
        // compiler's `Failed to normalize ... maybe try
        // try_normalize_erasing_regions` ICE). Only
        // parameter-free instances are recorded; the rest
        // keep the generic shape's behaviour downstream.
        if instance_mentions_param(args) {
            return;
        }
        // Gate on the compiler's own layout query for the
        // instance (contained, like every layout query in
        // this driver; the query is memoized, so asking again
        // here costs nothing measurable). Computing a layout
        // normalizes the field types in the same codegen
        // environment, so an instance with no computable
        // layout has field projections `ty_with_args` would
        // fail fatally on — typically projections justified
        // only by a where-clause on the definition, whose
        // trait has no unconditional implementation for the
        // substituted arguments at all.
        let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        if !matches!(layout, Ok(Ok(_))) {
            return;
        }
        let kind = shape.kind;
        let saved_errors = std::mem::take(&mut self.errors);
        let mut variants = Vec::new();
        let mut complete = true;
        'variants: for variant in def.variants() {
            let mut fields = Vec::new();
            for field in variant.fields() {
                let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    field.ty_with_args(args)
                }));
                let Ok(resolved) = resolved else {
                    complete = false;
                    break 'variants;
                };
                fields.push(self.ty(&resolved));
            }
            variants.push(fields);
        }
        if complete && self.errors.is_empty() {
            self.adt_instance_shapes
                .insert(key, sm::AdtShape { kind, variants });
        }
        self.errors = saved_errors;
    }

    /// The `adt_layouts` key for a converted type: its `ty_tag`
    /// rendering against the layout name table, which is kept in
    /// sync with `def_names` (the interpreter renders call-site
    /// tags against the same names).
    /// Render the term of an existential projection predicate
    /// — the `101` of `Trait<N = 101>`, the type of
    /// `Trait<Item = u8>` — as it appears in the type's name.
    fn existential_term(&mut self, term: &rustc_public::ty::TermKind) -> String {
        use rustc_public::ty::TermKind;
        match term {
            TermKind::Type(t) => {
                let converted = self.ty(t);
                self.layout_key(&converted)
            }
            TermKind::Const(c) => match c.kind() {
                rustc_public::ty::TyConstKind::Value(ty, alloc) => {
                    match self.const_from_alloc(&ty, &alloc) {
                        sm::ConstValue::Scalar(sm::Scalar::Int { data, size }) => {
                            if matches!(
                                guarded_ty_kind(&ty),
                                Some(TyKind::RigidTy(RigidTy::Int(_)))
                            ) {
                                let shift = 128 - u32::from(size) * 8;
                                format!("{}", ((data << shift) as i128) >> shift)
                            } else {
                                format!("{data}")
                            }
                        }
                        sm::ConstValue::Bool(b) => b.to_string(),
                        sm::ConstValue::Char(ch) => format!("'{ch}'"),
                        sm::ConstValue::Str(s) => format!("{s:?}"),
                        _ => "<const>".to_string(),
                    }
                }
                rustc_public::ty::TyConstKind::Param(p) => p.name.clone(),
                rustc_public::ty::TyConstKind::Unevaluated(def, _) => def.name(),
                _ => "<const>".to_string(),
            },
        }
    }

    pub fn layout_key(&mut self, converted: &sm::Ty) -> String {
        while self.layout_names.len() < self.def_names.len() {
            let index = self.layout_names.len();
            self.layout_names.insert(
                sm::DefId::new(0, index as u32),
                self.def_names[index].clone(),
            );
        }
        sparagmos_interp::builtins::ty_tag(converted, &self.layout_names)
    }

    /// Record the compiler-computed layout of one monomorphic
    /// closure or coroutine instance under its type tag, once per
    /// distinct instance. A coroutine's layout is the compiler's
    /// own state-machine layout (which captures live across each
    /// suspension point), and a closure's is its capture aggregate;
    /// neither can be derived from the converted MIR, so both come
    /// from the same layout query the ADT path uses. The entry
    /// lands in `adt_layouts`, the table the
    /// `size_of_val` / `align_of_val` builtins consult by tag.
    /// Instances the query cannot answer (generic parameters among
    /// the arguments) are skipped under the same containment as
    /// the ADT path: the query is caught, and the instance stays
    /// explicitly unsupported.
    fn record_coroutine_layout(&mut self, ty: &rustc_public::ty::Ty, converted: &sm::Ty) {
        if !self.layout_attempted_tys.insert(*ty) {
            return;
        }
        let key = self.layout_key(converted);
        if !self.layout_attempted.insert(key.clone()) {
            return;
        }
        let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        let Ok(Ok(layout)) = layout else {
            return;
        };
        let shape = layout.shape();
        if shape.is_unsized() {
            return;
        }
        self.adt_layouts.insert(
            key,
            (shape.size.bytes() as u64, shape.abi_align),
        );
    }

    /// Record the compiler-computed storage layout of one
    /// monomorphic tuple type under its rendered name, once
    /// per distinct type (see `sm::TupleLayouts`): the
    /// tuple's total storage size and each field's (offset,
    /// size) in declaration order, from the same layout
    /// query the ADT path uses. A tuple's storage order is
    /// the compiler's choice — fields may be reordered and
    /// padding inserted — so the interpreter's scalar shape
    /// of a tuple is built from this record, never from
    /// declaration order. Types the query cannot answer
    /// (generic parameters among the elements, unsized
    /// elements) are skipped under the same containment as
    /// the ADT path: the query is caught, and the type
    /// simply gets no entry. A type mentioning a coroutine
    /// is not asked at all, exactly as in
    /// `record_field_layouts`: computing such a type's
    /// layout can drive the compiler into the coroutine's
    /// own destructor-shim construction, which is not
    /// total. The mention is judged on the converted type,
    /// never on the compiler's: asking the compiler's type
    /// for its kind is itself the question that panics for
    /// coroutine-closure types (see `guarded_ty_kind`), and
    /// this recorder runs inside type conversion, where
    /// such a panic would take the whole body with it.
    fn record_tuple_layout(
        &mut self,
        ty: &rustc_public::ty::Ty,
        elem_tys: &[rustc_public::ty::Ty],
        converted: &sm::Ty,
    ) {
        if !self.layout_attempted_tys.insert(*ty) {
            return;
        }
        let key = self.layout_key(converted);
        if !self.layout_attempted.insert(key.clone()) {
            return;
        }
        if converted_mentions_coroutine(converted) {
            return;
        }
        let layout =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        let Ok(Ok(layout)) = layout else {
            return;
        };
        let shape = layout.shape();
        if shape.is_unsized() {
            return;
        }
        let FieldsShape::Arbitrary { offsets } = &shape.fields else {
            return;
        };
        if offsets.len() != elem_tys.len() {
            return;
        }
        let mut spans = Vec::with_capacity(elem_tys.len());
        for (elem_ty, off) in elem_tys.iter().zip(offsets.iter()) {
            let queried = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| elem_ty.layout()),
            );
            let Ok(Ok(elayout)) = queried else { return };
            spans.push((
                off.bytes() as u64,
                elayout.shape().size.bytes() as u64,
            ));
        }
        self.tuple_layouts
            .insert(key, (shape.size.bytes() as u64, spans));
    }

    /// Record the storage-slot map of one monomorphic coroutine
    /// instance (see `sm::CoroutineSlots`) from the compiler's
    /// layout of the instance. The layout's per-variant field
    /// offsets are absolute, and a state field keeps one offset
    /// in every variant that contains it, so slots are assigned
    /// per distinct offset across all variants, after the upvar
    /// slots — except the zero-sized fields that share an offset
    /// with the field after them inside one variant, which are
    /// distinct cells and get private slots (see below).
    /// Instances whose layout the query cannot answer,
    /// and shapes that do not match the state-machine form (the
    /// outer fields are the upvars followed by the discriminant
    /// tag), get no entry; their projections keep the positional
    /// conversion.
    fn record_coroutine_slots(
        &mut self,
        ty: &rustc_public::ty::Ty,
        converted: &sm::Ty,
    ) {
        use rustc_public::abi::{FieldsShape, VariantsShape};
        if !self.coroutine_slots_attempted.insert(*ty) {
            return;
        }
        let sm::Ty::Coroutine(..) = converted else {
            return;
        };
        let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        let Ok(Ok(layout)) = layout else {
            return;
        };
        let shape = layout.shape();
        let FieldsShape::Arbitrary { offsets } = &shape.fields else {
            return;
        };
        let VariantsShape::Multiple { tag_field, variants, .. } = &shape.variants
        else {
            return;
        };
        // The outer fields of a coroutine layout are the upvars
        // followed by the discriminant tag, whose field index the
        // layout itself reports: the upvar count is the tag's
        // index. (The converted generic arguments do not reliably
        // start with the upvars tuple - a generic coroutine's
        // arguments begin with the definition's own parameters.)
        let n_upvars = *tag_field;
        if offsets.len() != n_upvars + 1 {
            return;
        }
        // One slot per distinct offset, shared by every variant
        // that has a field there — with one exception. Inside a
        // single variant, several fields can sit at one offset:
        // a zero-sized field consumes no bytes, so the field
        // after it starts at the same offset. Those fields are
        // distinct storage cells (the zero-sized value and the
        // value after it are both live in that state), and
        // merging them would let each overwrite the other. In
        // offset order, every field of such a group but the last
        // is zero-sized (the next field starts at its offset, so
        // it spans no bytes), so the last field owns the offset's
        // shared slot and each earlier field gets a private slot
        // of its own, used by no other variant.
        let mut state_offsets: Vec<u64> = Vec::new();
        for variant in variants {
            for offset in &variant.offsets {
                let offset = offset.bytes() as u64;
                if !state_offsets.contains(&offset) {
                    state_offsets.push(offset);
                }
            }
        }
        let mut next_slot = n_upvars + state_offsets.len();
        let variants: Vec<Vec<u32>> = variants
            .iter()
            .map(|variant| {
                let offsets: Vec<u64> =
                    variant.offsets.iter().map(|o| o.bytes() as u64).collect();
                // The owner of each offset in this variant: the
                // last field, in (offset, field index) order,
                // among the fields at that offset.
                let mut owner: std::collections::HashMap<u64, usize> =
                    std::collections::HashMap::new();
                let mut order: Vec<usize> = (0..offsets.len()).collect();
                order.sort_by_key(|&i| (offsets[i], i));
                for &i in &order {
                    owner.insert(offsets[i], i);
                }
                offsets
                    .iter()
                    .enumerate()
                    .map(|(i, offset)| {
                        if owner[&offset] == i {
                            let pos = state_offsets
                                .iter()
                                .position(|o| o == offset)
                                .expect("offset was collected above");
                            (n_upvars + pos) as u32
                        } else {
                            let slot = next_slot as u32;
                            next_slot += 1;
                            slot
                        }
                    })
                    .collect()
            })
            .collect();
        let key = self.layout_key(converted);
        self.coroutine_layouts.insert(key, sm::CoroutineSlots { variants });
    }

    /// Record the debugger shape of one monomorphic closure or
    /// coroutine instance under its rendered instance name, once
    /// per distinct instance (see `sm::EnvDebugShapes`).
    ///
    /// The shape is read off the instance body's own debug info.
    /// The compiler records each captured variable there with the
    /// name the debuginfo gives the environment field - the
    /// captured place's name, `__`-joined along the place's
    /// projections - addressed by a projection from the
    /// environment base local: a `Field` step for a capture, a
    /// `Downcast` + `Field` pair for one of a coroutine's saved
    /// locals. A `Deref` step after the field step marks a
    /// by-reference capture, whose debuginfo name carries the
    /// `_ref__` prefix. The base local is identified by its
    /// declared type, which strips (through references and a
    /// `Pin` wrapper) to this instance's own definition.
    ///
    /// Recording is auxiliary to the conversion in progress,
    /// exactly like `record_debug_shape`: the current body's
    /// error list is saved and restored around it. An environment
    /// field the debug info does not name (the compiler emits no
    /// entry for a zero-sized capture) is recovered by type from
    /// the instance's upvars tuple with no name, rather than
    /// dropping the whole entry: consumers that need only the
    /// field type tags (the closure clone path among them) still
    /// get a complete shape, and the debugger renders no invented
    /// names - a nameless field simply never matches a name
    /// lookup and keeps the value on the generic render path.
    fn record_env_shape(
        &mut self,
        ty: &rustc_public::ty::Ty,
        kind: &TyKind,
        converted: &sm::Ty,
    ) {
        if !self.env_shape_attempted.insert(*ty) {
            return;
        }
        if ty_has_param(converted) {
            return;
        }
        let (def_name, label, args, body) = match kind {
            TyKind::RigidTy(RigidTy::Closure(def, args)) => {
                let Some(ck) = closure_kind(args) else {
                    return;
                };
                let resolved = resolve_closure_guarded(*def, args, ck);
                let key = resolved
                    .as_ref()
                    .map(|inst| inst.name())
                    .unwrap_or_else(|| def.name());
                let body = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| {
                        instance_item_body(&key, resolved, || def.body())
                    }),
                );
                let Ok(Some(body)) = body else {
                    return;
                };
                (def.name(), "closure", args, body)
            }
            TyKind::RigidTy(RigidTy::Coroutine(def, args)) => {
                let resolved = resolve_instance_guarded(FnDef(def.def_id()), args);
                let key = resolved
                    .as_ref()
                    .map(|inst| inst.name())
                    .unwrap_or_else(|| def.name());
                let body = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| {
                        instance_item_body(&key, resolved, || def.body())
                    }),
                );
                let Ok(Some(body)) = body else {
                    return;
                };
                (def.name(), "coroutine", args, body)
            }
            _ => return,
        };
        let Some(upvar_tys) = env_upvar_tys(args) else {
            return;
        };
        let key = self.layout_key(converted);
        let saved_errors = std::mem::take(&mut self.errors);
        let shape = self.env_shape_from_body(
            &body,
            &def_name,
            env_dwarf_name(&def_name, label),
            &upvar_tys,
            &key,
            label == "coroutine",
        );
        self.errors = saved_errors;
        if let Some(shape) = shape {
            self.env_debug_shapes.insert(key, shape);
        }
    }

    /// Assemble one environment shape from an instance body's
    /// debug info (see `record_env_shape`). `key` is the
    /// instance's rendered name, under which its coroutine
    /// storage-slot map (if any) is also recorded.
    fn env_shape_from_body(
        &mut self,
        body: &mir::Body,
        def_name: &str,
        dwarf_name: String,
        upvar_tys: &[rustc_public::ty::Ty],
        key: &str,
        is_coroutine: bool,
    ) -> Option<sm::EnvDebugShape> {
        let upvar_count = upvar_tys.len();
        // Parse the debug entries into capture and saved-local
        // candidates, grouped by their base local.
        struct Candidate {
            base: usize,
            variant: Option<usize>,
            field: usize,
            by_ref: bool,
            name: String,
            fty: rustc_public::ty::Ty,
        }
        let mut candidates: Vec<Candidate> = Vec::new();
        for info in &body.var_debug_info {
            let mir::VarDebugInfoContents::Place(place) = &info.value
            else {
                continue;
            };
            let proj = &place.projection;
            let mut i = 0;
            while matches!(proj.get(i), Some(mir::ProjectionElem::Deref)) {
                i += 1;
            }
            let (variant, field, fty, rest) = match (proj.get(i), proj.get(i + 1)) {
                (
                    Some(mir::ProjectionElem::Downcast(v)),
                    Some(mir::ProjectionElem::Field(f, fty)),
                ) => (Some(v.to_index()), *f, *fty, i + 2),
                (Some(mir::ProjectionElem::Field(f, fty)), _) => {
                    (None, *f, *fty, i + 1)
                }
                _ => continue,
            };
            // A capture's place ends at the field step or at the
            // dereference of the reference the capture holds; a
            // longer tail denotes a sub-place of a variable the
            // environment does not hold a field for.
            if variant.is_none()
                && rest != proj.len()
                && !(rest + 1 == proj.len()
                    && matches!(proj.get(rest), Some(mir::ProjectionElem::Deref)))
            {
                continue;
            }
            let by_ref = variant.is_none()
                && matches!(proj.get(rest), Some(mir::ProjectionElem::Deref));
            candidates.push(Candidate {
                base: place.local,
                variant,
                field,
                by_ref,
                name: info.name.clone(),
                fty,
            });
        }
        // The environment base: the one candidate base whose
        // declared type denotes this instance's definition.
        let locals = body.locals();
        let mut base: Option<usize> = None;
        for cand in &candidates {
            if base == Some(cand.base) {
                continue;
            }
            let Some(decl) = locals.get(cand.base) else {
                continue;
            };
            if env_def_name(&decl.ty).as_deref() == Some(def_name) {
                if base.is_some() {
                    return None;
                }
                base = Some(cand.base);
            }
        }
        let base = base?;
        // Field descriptors, in the shape's own terms: the
        // debuginfo name (with a by-reference capture's `_ref__`
        // prefix) and the field type's rendered tag.
        let mut upvars: HashMap<usize, sm::AdtDebugField> = HashMap::new();
        let mut saved: HashMap<(usize, usize), sm::AdtDebugField> = HashMap::new();
        for cand in candidates.into_iter().filter(|c| c.base == base) {
            let converted_field = self.ty(&cand.fty);
            let tag = self.layout_key(&converted_field);
            let name = if cand.by_ref {
                format!("_ref__{}", cand.name)
            } else {
                cand.name
            };
            let field = sm::AdtDebugField {
                name: Some(name),
                ty: tag,
            };
            match cand.variant {
                None => {
                    upvars.entry(cand.field).or_insert(field);
                }
                Some(v) => {
                    saved.entry((v, cand.field)).or_insert(field);
                }
            }
        }
        // A field the debug info does not name is recovered
        // from the upvars tuple: the tuple carries every
        // capture's type in environment field order, so the
        // field's type tag is exact; only its name stays `None`
        // (see `record_env_shape`).
        let mut filled: Vec<sm::AdtDebugField> = Vec::with_capacity(upvar_count);
        for i in 0..upvar_count {
            if let Some(field) = upvars.get(&i) {
                filled.push(field.clone());
                continue;
            }
            let converted_field = self.ty(&upvar_tys[i]);
            filled.push(sm::AdtDebugField {
                name: None,
                ty: self.layout_key(&converted_field),
            });
        }
        let upvars = filled;
        if !is_coroutine {
            return Some(sm::EnvDebugShape {
                dwarf_name,
                upvars,
                variants: Vec::new(),
                slotted: false,
            });
        }
        if let Some(slots) = self.coroutine_layouts.get(key).cloned() {
            // Name every slot from the saved-local candidates:
            // a candidate's (variant, field) address names the
            // slot the layout map assigns to it.
            let mut by_slot: HashMap<u32, sm::AdtDebugField> = HashMap::new();
            for ((v, f), field) in &saved {
                let slot = slots.variants.get(*v)?.get(*f)?;
                by_slot.insert(*slot, field.clone());
            }
            let mut variants = Vec::with_capacity(slots.variants.len());
            for (v, vslots) in slots.variants.iter().enumerate() {
                let mut fields = Vec::with_capacity(vslots.len());
                for slot in vslots {
                    fields.push(by_slot.get(slot)?.clone());
                }
                variants.push(sm::EnvDebugVariant {
                    name: coroutine_variant_name(v),
                    fields,
                    slots: vslots.clone(),
                });
            }
            return Some(sm::EnvDebugShape {
                dwarf_name,
                upvars,
                variants,
                slotted: true,
            });
        }
        // No storage-slot map: the saved storage is positional,
        // one vector per state in the body's own variant order.
        let n_variants = saved
            .keys()
            .map(|(v, _)| v + 1)
            .max()
            .unwrap_or(0)
            .max(3);
        let mut variants = Vec::with_capacity(n_variants);
        for v in 0..n_variants {
            let count = saved.keys().filter(|(sv, _)| *sv == v).count();
            let mut fields = Vec::with_capacity(count);
            for f in 0..count {
                fields.push(saved.get(&(v, f))?.clone());
            }
            variants.push(sm::EnvDebugVariant {
                name: coroutine_variant_name(v),
                fields,
                slots: (0..count as u32).collect(),
            });
        }
        Some(sm::EnvDebugShape {
            dwarf_name,
            upvars,
            variants,
            slotted: false,
        })
    }

    /// Record the debuginfo spelling of a foreign (extern-block)
    /// type's name in place of the Stable MIR spelling. The name
    /// `DefId::name` records omits the extern block's scope
    /// segment, but the compiler's debuginfo names the type
    /// through its scope chain (`type_names::{extern#0}::ForeignType1`),
    /// and a debugger's type query reports that spelling (the
    /// debuginfo suite's type-names record pins it). The recorded
    /// name is the only identity a foreign type carries into the
    /// program, so it is rewritten here, at the one conversion
    /// point every mention of the type passes through. When the
    /// compiler context is unavailable or the name cannot be
    /// computed under the resolution containment, the Stable MIR
    /// spelling stands.
    fn record_foreign_name(&mut self, def: &rustc_public::ty::ForeignDef, id: sm::DefId) {
        let Some(tcx) = compiler_tcx() else {
            return;
        };
        let (result, _emitted) = with_instance_resolution_contained(|| {
            use rustc_codegen_ssa::debuginfo::type_names;
            let internal = rustc_public::rustc_internal::internal(tcx, def.def_id());
            let ty = tcx.type_of(internal).skip_binder();
            type_names::compute_debuginfo_type_name(tcx, ty, true)
        });
        if let Ok(name) = result {
            if let Some(slot) = self.def_names.get_mut(id.index as usize) {
                *slot = name;
            }
        }
    }

    fn rigid_ty(&mut self, ty: &rustc_public::ty::Ty, r: &RigidTy) -> sm::Ty {
        match r {
            RigidTy::Bool => sm::Ty::Bool,
            RigidTy::Char => sm::Ty::Char,
            RigidTy::Int(t) => sm::Ty::Int(int_ty(*t)),
            RigidTy::Uint(t) => sm::Ty::Uint(uint_ty(*t)),
            RigidTy::Float(t) => sm::Ty::Float(float_ty(*t)),
            RigidTy::Never => sm::Ty::Never,
            RigidTy::Str => sm::Ty::Str,
            RigidTy::Adt(def, args) => {
                let d = self.crate_def_id(def);
                self.record_discriminants(def, d);
                self.record_shape(def, d);
                sm::Ty::Adt(d, self.type_args(args))
            }
            RigidTy::Foreign(def) => {
                let d = self.crate_def_id(def);
                self.record_foreign_name(def, d);
                sm::Ty::Foreign(d)
            }
            RigidTy::Array(elem, len) => {
                let e = self.ty(elem);
                let l = match len.eval_target_usize() {
                    Ok(n) => sm::ArrayLen::Known(n),
                    Err(_) => {
                        self.err("Ty::Array (length not evaluable)");
                        sm::ArrayLen::Param("?".to_string())
                    }
                };
                sm::Ty::Array(Box::new(e), l)
            }
            RigidTy::Slice(elem) => sm::Ty::Slice(Box::new(self.ty(elem))),
            RigidTy::RawPtr(elem, m) => sm::Ty::RawPtr(Box::new(self.ty(elem)), mutability(*m)),
            RigidTy::Ref(region, elem, m) => sm::Ty::Ref(
                region_of(region),
                Box::new(self.ty(elem)),
                mutability(*m),
            ),
            RigidTy::FnDef(def, args) => {
                let d = self.fn_def_id(def, args);
                sm::Ty::FnDef(d, self.type_args(args))
            }
            RigidTy::FnPtr(poly) => {
                let sig = &poly.value;
                let mut ios = sig.inputs_and_output.clone();
                let output = match ios.pop() {
                    Some(o) => self.ty(&o),
                    None => {
                        self.err("Ty::FnPtr (empty signature)");
                        sm::Ty::unit()
                    }
                };
                let inputs = ios.iter().map(|t| self.ty(t)).collect();
                let mut s = sm::FnSig::new(inputs, output);
                s.c_variadic = sig.c_variadic;
                s.safety = match sig.safety {
                    mir::Safety::Safe => sm::Safety::Safe,
                    mir::Safety::Unsafe => sm::Safety::Unsafe,
                };
                s.abi = abi_of(&sig.abi);
                sm::Ty::FnPtr(Box::new(s))
            }
            RigidTy::Dynamic(bounds, _) => {
                // The bounds as name strings. A projection
                // predicate is an associated-item binding of
                // one of the trait bounds (`Trait<N = 101>`):
                // it folds into that trait's name as a generic
                // argument rather than standing as a bound of
                // its own. Each entry keeps the trait path it
                // was rendered from beside the rendered name:
                // folding a binding rewrites the rendered name
                // (`Trait` becomes `Trait<A=u8>`), so matching
                // a later binding of the same trait against the
                // rendered names would miss the entry and stand
                // the binding up as a separate bound.
                let mut names: Vec<(String, String)> = Vec::new();
                let mut bindings: Vec<(String, String, String)> = Vec::new();
                for b in bounds.iter() {
                    match &b.value {
                        rustc_public::ty::ExistentialPredicate::Trait(t) => {
                            let name = t.def_id.name();
                            // The trait's own generic arguments
                            // (the existential trait reference's
                            // own arguments; `Self` is not among
                            // them)
                            // are part of the type's debuginfo
                            // name (`dyn Trait2<i32, Struct2>`);
                            // dropping them loses the arguments
                            // from every rendering of the bound.
                            let args: Vec<String> = t
                                .generic_args
                                .0
                                .iter()
                                .filter_map(|a| match a {
                                    rustc_public::ty::GenericArgKind::Type(
                                        ty,
                                    ) => {
                                        let converted = self.ty(ty);
                                        Some(self.layout_key(&converted))
                                    }
                                    _ => None,
                                })
                                .collect();
                            let rendered = if args.is_empty() {
                                name.clone()
                            } else {
                                format!("{name}<{}>", args.join(", "))
                            };
                            names.push((name, rendered));
                        }
                        rustc_public::ty::ExistentialPredicate::AutoTrait(d) => {
                            let name = d.name();
                            names.push((name.clone(), name));
                        }
                        rustc_public::ty::ExistentialPredicate::Projection(p) => {
                            let item = p.def_id.name();
                            let term = self.existential_term(&p.term);
                            match item.rsplit_once("::") {
                                Some((trait_path, assoc)) => bindings.push((
                                    trait_path.to_string(),
                                    assoc.to_string(),
                                    term,
                                )),
                                None => names.push((item.clone(), item)),
                            }
                        }
                    }
                }
                for (trait_path, assoc, term) in bindings {
                    let binding = format!("{assoc}={term}");
                    match names
                        .iter_mut()
                        .find(|(base, _)| base == &trait_path)
                    {
                        Some((_, slot)) => {
                            if slot.ends_with('>') {
                                slot.insert_str(
                                    slot.len() - 1,
                                    &format!(", {binding}"),
                                );
                            } else {
                                *slot = format!("{slot}<{binding}>");
                            }
                        }
                        None => names.push((
                            trait_path.clone(),
                            format!("{trait_path}<{binding}>"),
                        )),
                    }
                }
                let names = names
                    .into_iter()
                    .map(|(_, rendered)| rendered)
                    .collect();
                sm::Ty::Dynamic(names, Box::new(sm::Ty::unit()), sm::Region::Erased)
            }
            RigidTy::Closure(def, args) => {
                let d = self.closure_def_id(def, args);
                sm::Ty::Closure(d, self.type_args(args))
            }
            RigidTy::Coroutine(def, args) => {
                let d = self.crate_def_id(def);
                sm::Ty::Coroutine(d, self.type_args(args))
            }
            RigidTy::CoroutineClosure(def, args) => {
                let d = self.crate_def_id(def);
                self.async_closure_defs.insert(d);
                sm::Ty::Coroutine(d, self.type_args(args))
            }
            RigidTy::CoroutineWitness(def, args) => {
                let d = self.crate_def_id(def);
                sm::Ty::CoroutineWitness(d, self.type_args(args))
            }
            RigidTy::Tuple(elems) => {
                let converted =
                    sm::Ty::Tuple(elems.iter().map(|t| self.ty(t)).collect());
                self.record_tuple_layout(ty, elems, &converted);
                converted
            }
            RigidTy::Pat(inner, _) => {
                // A pattern type is a refinement of its base
                // type with the same representation, so it
                // converts as the base type everywhere: body
                // locals, constants, and ADT shape fields alike
                // (std's own `NonNull` carries a pattern-typed
                // pointer field, and shape consumers judge fields
                // by their base types). Construction in a valid
                // program guarantees the range; layout facts that
                // depend on it (niches) come from the rustc
                // layout tables, not from this type.
                self.ty(inner)
            }
        }
    }

    pub(crate) fn cast_kind(&mut self, kind: &mir::CastKind) -> sm::CastKind {
        use mir::CastKind as C;
        match kind {
            C::PointerExposeAddress => sm::CastKind::PointerExposeProvenance,
            C::PointerWithExposedProvenance => sm::CastKind::PointerWithExposedProvenance,
            C::PointerCoercion(c) => sm::CastKind::PointerCoercion(pointer_coercion(*c)),
            C::IntToInt => sm::CastKind::IntToInt,
            C::FloatToInt => sm::CastKind::FloatToInt,
            C::FloatToFloat => sm::CastKind::FloatToFloat,
            C::IntToFloat => sm::CastKind::IntToFloat,
            C::PtrToPtr => sm::CastKind::PtrToPtr,
            C::FnPtrToPtr => sm::CastKind::FnPtrToPtr,
            C::Transmute => sm::CastKind::Transmute,
            C::Subtype => sm::CastKind::Subtype,
            C::BoxDerefTransmute => sm::CastKind::BoxDerefTransmute,
        }
    }
}
