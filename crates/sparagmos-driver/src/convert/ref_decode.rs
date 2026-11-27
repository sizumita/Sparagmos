//! Function-pointer and reference constant decoding, and static slice/offset path recording.

use super::*;

impl Converter {
    /// Decode a reference (or raw pointer) stored in `alloc` at
    /// `offset`, pointing at a value of type `inner`. The result is
    /// the reference's value in the interpreter's model: string and
    /// byte data directly, a fat slice value for decoded slices, and
    /// `ConstValue::Ref` over the decoded pointee otherwise.
    /// Decode a function pointer stored in `alloc` at `offset`:
    /// its provenance names the target instance. A plain function
    /// item becomes the callable of that item. A closure coerced to
    /// a function pointer points at the closure's `Fn` trait adapter
    /// shim (const evaluation picks the `FnOnce` one), whose `Self`
    /// argument is the closure type; a direct closure instance's
    /// own type is the closure type. Both identify the closure
    /// body, which is the callable, with an empty environment (only
    /// non-capturing closures coerce to function pointers).
    /// Anything else (a vtable, a null pointer with no provenance)
    /// has no callable constant form and stays undecoded.
    pub(crate) fn decode_fn_pointer(
        &mut self,
        alloc: &Allocation,
        offset: usize,
    ) -> Option<sm::ConstValue> {
        let prov = prov_at(alloc, offset)?;
        let mir::alloc::GlobalAlloc::Function(inst) =
            mir::alloc::GlobalAlloc::from(prov.0)
        else {
            return None;
        };
        self.fn_pointer_value(&inst)
    }

    /// The callable constant naming a function instance: a plain
    /// function item becomes the callable of that item; a closure
    /// (named directly, or through the `Fn` adapter shim const
    /// evaluation picks for a coercion) becomes the closure value
    /// with an empty environment.
    pub(crate) fn fn_pointer_value(
        &mut self,
        inst: &rustc_public::mir::mono::Instance,
    ) -> Option<sm::ConstValue> {
        if matches!(inst.kind, InstanceKind::Shim) {
            for arg in &inst.args().0 {
                if let GenericArgKind::Type(t) = arg {
                    if let TyKind::RigidTy(RigidTy::Closure(def, args)) = t.kind() {
                        self.note_const_closure(&def, &args);
                        let d = self.closure_def_id(&def, &args);
                        return Some(sm::ConstValue::Closure(d));
                    }
                }
            }
        }
        if let TyKind::RigidTy(RigidTy::Closure(def, args)) = inst.ty().kind() {
            self.note_const_closure(&def, &args);
            let d = self.closure_def_id(&def, &args);
            return Some(sm::ConstValue::Closure(d));
        }
        self.note_const_fn(inst);
        let d = self.instance_def_id(inst.def.def_id(), inst);
        Some(sm::ConstValue::FnDef(d))
    }

    /// Record a closure materialized as a value out of a constant
    /// (see the `const_closure_mentions` field): the demand passes
    /// collect the bodies of the closures recorded here, because
    /// no type in any converted body mentions them.
    pub(crate) fn note_const_closure(
        &mut self,
        def: &rustc_public::ty::ClosureDef,
        args: &rustc_public::ty::GenericArgs,
    ) {
        let key = (def.def_id(), format!("{args:?}"));
        if self.const_closure_mentions_seen.insert(key) {
            self.const_closure_mentions.push((*def, args.clone()));
        }
    }

    /// Record a function item materialized as a value out of a
    /// constant (see the `const_fn_mentions` field). The record
    /// keeps the resolved instance itself: constant evaluation
    /// has already monomorphized the pointer, so no resolution
    /// is left for the collection pass.
    pub(crate) fn note_const_fn(&mut self, inst: &rustc_public::mir::mono::Instance) {
        let key = (inst.def.def_id(), format!("{:?}", inst.args()));
        if self.const_fn_mentions_seen.insert(key) {
            self.const_fn_mentions.push(inst.clone());
        }
    }

    /// Decode a reference whose provenance names a foreign
    /// static (an item of an `extern` block). A foreign static
    /// has no initializer to decode — asking the compiler for
    /// one ICEs (see `static_is_foreign`) — so the reference
    /// denotes the static's program-level storage with an
    /// uninitialized seed: address-only uses (the common case:
    /// `addr_of!`, raw-pointer casts, comparisons) observe the
    /// storage's identity, writes through a mutable reference
    /// land in the store like any other static's, and reads of
    /// never-written storage meet the interpreter's
    /// uninitialized-read error instead of an invented value.
    /// A pointee whose decoded form IS the contents (a `str`)
    /// cannot be represented this way and stays undecodable,
    /// as does a slice reference whose backing array cannot be
    /// located in the static's declared type (see
    /// `static_slice_path`).
    fn decode_foreign_static_ref(
        &mut self,
        def: &mir::mono::StaticDef,
        inner: &rustc_public::ty::Ty,
        alloc: &Allocation,
        offset: usize,
        target_off: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        let def_id = self.crate_def_id(def);
        // The item scan does not enumerate foreign statics, so
        // the declared type is recorded here, at the first decode
        // that names the static: the interpreter expands a
        // whole-static reference's seed to the recorded type's
        // extent, and a fat slice reference's seed in particular
        // must not be sized by the slice's own (possibly prefix)
        // length. The query is the one the projection
        // computations below already make of this definition --
        // a foreign static is a real item, unlike the anonymous
        // promoted statics the type query is unsafe for -- and it
        // is contained the way the item scan contains it.
        if !self.static_tys.contains_key(&def_id) {
            if let Ok(ty) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| def.ty()))
            {
                self.note_static_ty(def, ty);
            }
        }
        match inner.kind() {
            TyKind::RigidTy(RigidTy::Slice(elem)) => {
                if matches!(elem.kind(), TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))) {
                    return None;
                }
                let count = read_uint_at(alloc, offset + PTR_SIZE, PTR_SIZE)? as usize;
                // The projection names the slice's backing array
                // inside the static's declared type (the whole
                // array, an array field, an array element): an
                // empty projection for a field's array would make
                // the slice view index the static's whole value
                // instead of the field. A slice whose backing
                // array cannot be located stays undecodable.
                let projection =
                    self.static_slice_path(&def.ty(), target_off, &elem, count, depth)?;
                Some(sm::ConstValue::StaticRef {
                    def: def_id,
                    value: Box::new(sm::ConstValue::Uninit),
                    projection,
                    len: Some(count as u64),
                })
            }
            TyKind::RigidTy(RigidTy::Str) => None,
            _ => {
                // The projection is computed exactly as for a
                // non-foreign static, at offset zero too: a
                // pointer to the first element or field of an
                // aggregate static denotes that sub-value, not
                // the whole static (an empty projection would
                // make a write through the pointer replace the
                // complete storage). `static_offset_path` yields
                // the empty path itself when the pointee IS the
                // whole static.
                let projection =
                    self.static_offset_path(&def.ty(), target_off, inner, depth)?;
                Some(sm::ConstValue::StaticRef {
                    def: def_id,
                    value: Box::new(sm::ConstValue::Uninit),
                    projection,
                    len: None,
                })
            }
        }
    }

    pub(crate) fn decode_ref(
        &mut self,
        inner: &rustc_public::ty::Ty,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        if depth == 0 {
            self.const_budget = MAX_CONST_TOTAL_ELEMS;
            self.const_scan_budget = MAX_CONST_SCAN_BYTES;
            self.const_work_budget = MAX_CONST_WORK_BYTES;
            self.const_work_failed = false;
            self.clear_alloc_caches();
        }
        // A pointee type passing through a coroutine-closure
        // type cannot be asked its kind (see `guarded_ty_kind`)
        // and has no decodable value form; the caller keeps the
        // `ByRef` identity instead.
        if guarded_ty_kind(inner).is_none() {
            return None;
        }
        // A `dyn` pointee makes the pointer a fat pointer: the
        // second word is the vtable, whose provenance names the
        // concrete type the data word denotes. The pair has its
        // own decoder (see `decode_dyn_ref`).
        if matches!(inner.kind(), TyKind::RigidTy(RigidTy::Dynamic(..))) {
            return self.decode_dyn_ref(inner, alloc, offset, depth);
        }
        let prov = prov_at(alloc, offset);
        let Some(prov) = prov else {
            // No provenance at the pointer slot: a stored
            // zero word is a null pointer (Lane CK). A
            // nonzero word is a dangling address from
            // `without_provenance` (an empty `Vec`'s buffer
            // pointer, `NonNull::dangling`): only the address
            // is observable, as an address-only reference
            // whose reads are explicit errors (Lane CE).
            let addr = read_uint_at(alloc, offset, PTR_SIZE)?;
            if addr == 0 {
                return Some(sm::ConstValue::NullPtr);
            }
            // A fat pointer (slice / str pointee) carries a second
            // word: the length. A dangling fat pointer constant
            // (from_raw_parts in a const context) has no
            // pointee, but its address identity and length are
            // exactly observable, so it decodes as a FatPtr
            // rather than dropping the metadata word.
            if matches!(
                inner.kind(),
                TyKind::RigidTy(RigidTy::Slice(_) | RigidTy::Str)
            ) {
                let meta = read_uint_at(alloc, offset + PTR_SIZE, PTR_SIZE)?;
                return Some(sm::ConstValue::FatPtr {
                    addr: addr as u64,
                    meta: meta as u64,
                });
            }
            return Some(sm::ConstValue::IntPtr(addr as u64));
        };
        let target_off = read_uint_at(alloc, offset, PTR_SIZE)? as usize;
        let (target, static_def) = match self.resolve_alloc(prov) {
            ResolvedAlloc::Memory(target) => (target, None),
            // A reference to a static: the pointee is the static's
            // own storage. The decoded value is the static's
            // initializer allocation (the value the storage holds
            // before the program runs); the reference keeps the
            // static's identity so the interpreter can route reads
            // and writes through the program-level statics store
            // instead of a per-constant snapshot.
            ResolvedAlloc::Other(mir::alloc::GlobalAlloc::Static(def)) => {
                // An extern-item-impl (EII) declaration is a
                // foreign item - it has no initializer of its
                // own, and evaluating it panics - but its storage
                // is the registered implementation's storage:
                // decode the reference against the implementation,
                // an ordinary static, so reads and writes through
                // the declaration land in the one storage.
                let def = if static_is_foreign(&def) {
                    match eii_impl_static(&def) {
                        Some(impl_def) => impl_def,
                        None => {
                            return self.decode_foreign_static_ref(
                                &def, inner, alloc, offset, target_off, depth,
                            );
                        }
                    }
                } else {
                    def
                };
                // A static defined in another crate is a real
                // item the item scan does not enumerate (the scan
                // covers local items only), so its declared type
                // is recorded here, at the first reference that
                // names it: the program's static tables are keyed
                // from this record (a whole-static reference's
                // seed is decoded against it below, and the
                // interpreter's declared-type recovery for static
                // references consults it). Local statics keep the
                // scan's record: the ones the scan misses are
                // anonymous promoted statics, whose type query can
                // ICE (see the scan in `collect`), so they are not
                // asked here. The query is contained the way the
                // scan contains it.
                if !def.krate().is_local {
                    let def_id = self.crate_def_id(&def);
                    if !self.static_tys.contains_key(&def_id) {
                        if let Ok(ty) = std::panic::catch_unwind(
                            std::panic::AssertUnwindSafe(|| def.ty()),
                        ) {
                            self.note_static_ty(&def, ty);
                        }
                    }
                }
                let init = self.static_initializer(&def)?;
                (init, Some(def))
            }
            // A pointer at a function item: the item is zero-sized,
            // so the pointer's value is the callable itself. This
            // covers both references to function items reached
            // through pointer-typed fields and pointers to functions
            // stored under a cast (`&(foo as *const u8)`).
            ResolvedAlloc::Other(mir::alloc::GlobalAlloc::Function(inst)) => {
                return self.fn_pointer_value(&inst);
            }
            ResolvedAlloc::Other(mir::alloc::GlobalAlloc::TypeId { ty: tid_ty }) => {
                // A `TypeId` value is a 128-bit hash of the type,
                // stored as two pointer-sized segments whose first
                // segment's provenance is this symbolic allocation.
                // The concrete hash function is unobservable - only
                // equality between TypeIds matters - so synthesize
                // the segments from a stable hash of the type. A
                // reference to the allocation denotes the TypeId
                // value itself; a pointer field pointing at it (the
                // value's own first segment) denotes the segment's
                // bits.
                let (lo, hi) = type_id_segments(&tid_ty)?;
                if matches!(inner.kind(), TyKind::RigidTy(RigidTy::Adt(..))) {
                    return Some(sm::ConstValue::Ref(Box::new(sm::ConstValue::Adt {
                        variant: 0,
                        fields: vec![sm::ConstValue::Tuple(vec![
                            sm::ConstValue::IntPtr(lo),
                            sm::ConstValue::IntPtr(hi),
                        ])],
                    })));
                }
                return Some(sm::ConstValue::IntPtr(lo));
            }
            // A vtable pointer: vtables are not materialised, but
            // the pointer's identity is observable (raw-pointer
            // equality / ordering, `ptr::eq` on trait objects), so
            // it decodes to a stable identity number for the
            // (concrete type, trait) pair.
            ResolvedAlloc::Other(mir::alloc::GlobalAlloc::VTable(vty, trait_ref)) => {
                let id = vtable_id(&vty, &trait_ref);
                self.note_vtable_identity(id, &vty, &trait_ref);
                return Some(sm::ConstValue::VTablePtr(id));
            }
            // `resolve_alloc` returns memory allocations in
            // the `Memory` variant, never wrapped in `Other`.
            ResolvedAlloc::Other(mir::alloc::GlobalAlloc::Memory(_)) => {
                return None;
            }
            // The work budget refused the materialisation:
            // the constant stays undecoded.
            ResolvedAlloc::Refused => {
                return None;
            }
        };
        match inner.kind() {
            TyKind::RigidTy(RigidTy::Str) => {
                // The string's bytes are exactly the fat pointer's
                // length word (the second word at the pointer
                // slot), as for the slice arm below: the backing
                // allocation may extend past the string.
                let len = read_uint_at(alloc, offset + PTR_SIZE, PTR_SIZE)? as usize;
                let bytes = alloc_bytes_in(&target, target_off..target_off + len)?;
                if !self.spend_const_budget(bytes.len()) {
                    return None;
                }
                String::from_utf8(bytes).ok().map(sm::ConstValue::Str)
            }
            TyKind::RigidTy(RigidTy::Slice(elem)) => {
                let count = read_uint_at(alloc, offset + PTR_SIZE, PTR_SIZE)? as usize;
                // A slice reference into a static's storage keeps
                // the static's identity, as a thin static reference
                // does: a detached snapshot of the elements would
                // make writes through a mutable slice static
                // (`static mut S: &mut [T] = &mut [..]`) land in a
                // copy no later read observes.
                if let Some(def) = &static_def {
                    if let Some(c) = self.decode_static_slice_ref(
                        def, &elem, &target, target_off, count, depth,
                    ) {
                        return Some(c);
                    }
                }
                if matches!(elem.kind(), TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))) {
                    let bytes = alloc_bytes_in(&target, target_off..target_off + count)?;
                    if !self.spend_const_budget(bytes.len()) {
                        return None;
                    }
                    return Some(sm::ConstValue::Bytes(bytes));
                }
                let stride = elem.layout().ok()?.shape().size.bytes();
                if stride == 0 {
                    // A zero-sized element type stores nothing
                    // per element: every element of the constant
                    // reads the same zero bytes at the base, so
                    // one decode replicated to the length is the
                    // exact value (a `[Z; 2]` slice constant, Z a
                    // unit struct, is the canonical case). The
                    // replication is charged to the expansion
                    // budget like the per-element decodes below.
                    if count == 0 {
                        return Some(sm::ConstValue::SliceVal(Vec::new()));
                    }
                    if !self.spend_const_budget(count) {
                        return None;
                    }
                    let elem_value =
                        self.decode_const(&elem, &target, target_off, depth + 1)?;
                    return Some(sm::ConstValue::SliceVal(vec![elem_value; count]));
                }
                let mut elems = Vec::new();
                for i in 0..count {
                    elems.push(self.decode_const(
                        &elem,
                        &target,
                        target_off + i * stride,
                        depth + 1,
                    )?);
                }
                Some(sm::ConstValue::SliceVal(elems))
            }
            TyKind::RigidTy(RigidTy::Array(elem, _))
                if matches!(elem.kind(), TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))) =>
            {
                // Byte arrays keep the established `Bytes` form (the
                // fmt pieces template is the canonical case).
                let bytes = alloc_bytes_in(&target, target_off..target.bytes.len())?;
                if !self.spend_const_budget(bytes.len()) {
                    return None;
                }
                Some(sm::ConstValue::Bytes(bytes))
            }
            _ => {
                // A type-erased pointee: the declared type is
                // zero-sized but the allocation region it denotes
                // is not — the pointer was stored through an
                // erased slot (a `NonNull<()>` field is the
                // canonical case), and its real type is named
                // only by the casts the guest later performs.
                // Preserve the allocation exactly so those casts
                // can materialise it (see `decode_erased_alloc`);
                // when the block cannot be built for an ordinary
                // reason, the declared type's decode below
                // stands; when the expansion budget ran out, the
                // constant stays undecoded instead.
                if self.pointee_is_erased(inner, &target, target_off) {
                    // A block that cannot be built because the
                    // expansion budget -- or the scan budget,
                    // whose exhaustion can equally deny a decode
                    // its `Uninit` shortcut, or the work budget,
                    // whose exhaustion can equally refuse a
                    // backing allocation the decode needs -- ran
                    // out must not fall
                    // back to the declared type's decode below:
                    // that decode names the zero-sized declared
                    // type, so the constant would hold a
                    // reference to a unit value where the source
                    // holds the whole allocation -- and a
                    // second, identical pointer in the same
                    // constant that did decode would compare
                    // unequal to it. An ordinary failure to
                    // build the block (a relocation that cannot
                    // be decoded type-free) keeps the declared
                    // decode, as before.
                    self.const_budget_failed = false;
                    self.const_scan_failed = false;
                    self.const_work_failed = false;
                    if let Some(block) =
                        self.decode_erased_alloc(&target, target_off, depth + 1)
                    {
                        return Some(sm::ConstValue::Ref(Box::new(block)));
                    }
                    if self.const_budget_failed || self.const_scan_failed || self.const_work_failed
                    {
                        return None;
                    }
                }
                if let Some(def) = &static_def {
                    // A static seed that cannot be built because the
                    // expansion budget -- or the scan budget,
                    // whose exhaustion can equally deny the seed
                    // decode its `Uninit` shortcut, or the work
                    // budget, whose exhaustion can equally refuse
                    // a backing allocation the decode needs -- ran
                    // out must not fall back to
                    // the snapshot decode below: the snapshot names
                    // only the pointee's current value, so the
                    // pointer loses the static's identity -- a
                    // second, identical pointer in the same constant
                    // that did decode as a static reference would
                    // not share storage with it, and writes through
                    // one would not be observed through the other.
                    // An ordinary failure to determine the
                    // projection keeps the snapshot, as before.
                    self.const_budget_failed = false;
                    self.const_scan_failed = false;
                    self.const_work_failed = false;
                    if let Some(c) =
                        self.decode_static_ref(def, inner, &target, target_off, depth)
                    {
                        return Some(c);
                    }
                    if self.const_budget_failed || self.const_scan_failed || self.const_work_failed
                    {
                        return None;
                    }
                }
                // A byte-sized pointee whose target allocation
                // extends past it: the pointer was cast in a
                // const context from a wider pointer into the
                // same allocation (a byte pointer into a static
                // byte table is the canonical case), so it
                // denotes the allocation from the offset on, not
                // the single byte — the same snapshot the byte
                // array arm above takes. A pointer whose
                // allocation is exactly the pointee keeps the
                // scalar decode.
                if matches!(
                    inner.kind(),
                    TyKind::RigidTy(
                        RigidTy::Uint(SmirUintTy::U8)
                            | RigidTy::Int(SmirIntTy::I8)
                            | RigidTy::Bool
                    )
                ) {
                    if let Some(raw) = alloc_bytes_in(&target, target_off..target.bytes.len()) {
                        if raw.len() > 1 {
                            if !self.spend_const_budget(raw.len()) {
                                return None;
                            }
                            return Some(sm::ConstValue::Ref(Box::new(
                                sm::ConstValue::Bytes(raw),
                            )));
                        }
                    }
                }
                // A pointer to a zero-sized pointee at a
                // nonzero offset into its allocation: the
                // pointee's value decode below reads nothing,
                // so the offset would be dropped, but the
                // offset is the pointer's whole observable
                // content. Native codegen places a zero-sized
                // allocation at its alignment (the dangling
                // address) and the offset moves from there, so
                // the pointer decodes as that address. A static
                // pointee keeps the identity path above: its
                // address is the static's own, not a dangling
                // address.
                if static_def.is_none() && target_off != 0 && target.bytes.is_empty() {
                    if let Ok(layout) = inner.layout() {
                        if layout.shape().size.bytes() == 0 {
                            // The dangling base is the target
                            // allocation's own alignment, not
                            // the pointee type's: a pointer
                            // cast to a weaker-aligned type
                            // keeps the address it was created
                            // with.
                            let align = target.align.max(1);
                            return Some(sm::ConstValue::IntPtr(
                                align.wrapping_add(target_off as u64),
                            ));
                        }
                    }
                }
                if let Some(pointee) =
                    self.decode_const(inner, &target, target_off, depth + 1)
                {
                    return Some(sm::ConstValue::Ref(Box::new(pointee)));
                }
                // A partially initialised pointee defeats the
                // typed decode -- a scalar leaf spanning
                // initialised and uninitialised bytes has no
                // value form (a union written through one field
                // and read through its bytes is the canonical
                // case) -- but the allocation itself is exact:
                // preserve it as an erased block, the form a
                // type-erased pointee takes above, and let the
                // materialisation at the guest's casts and reads
                // interpret it. A decode that failed on budget
                // keeps the constant undecoded, as the other
                // fallbacks in this function rule.
                if !self.const_budget_failed && !self.const_work_failed {
                    if let Ok(size) = inner.layout().map(|l| l.shape().size.bytes()) {
                        let size = size as usize;
                        if size > 0
                            && self.range_partially_uninit(&target, target_off, size)
                        {
                            if let Some(block) = self.decode_erased_alloc(
                                &target,
                                target_off,
                                depth + 1,
                            ) {
                                return Some(sm::ConstValue::Ref(Box::new(block)));
                            }
                        }
                    }
                }
                None
            }
        }
    }

    /// Decode a fat reference (or raw pointer) whose pointee is
    /// a `dyn` trait type at `offset` in `alloc`, as a
    /// [`sm::ConstValue::DynRef`]: the data pointer decoded as a
    /// reference to the concrete type, paired with the vtable
    /// identity -- the concrete type itself and the bounds of
    /// the `dyn` type the pointer was decoded from. The vtable
    /// identity comes from the provenance of the fat pointer's
    /// second word, which names the (concrete type, trait) pair
    /// the vtable serves; without that provenance the pointer's
    /// dispatch identity is unknowable and the constant stays
    /// undecoded (`None`, the caller's fallbacks apply, exactly
    /// as before this decoder existed).
    fn decode_dyn_ref(
        &mut self,
        inner: &rustc_public::ty::Ty,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        let vtable_prov = prov_at(alloc, offset + PTR_SIZE);
        let Some(vtable_prov) = vtable_prov else {
            // No vtable provenance: the pointer's dispatch
            // identity is unknowable. A data word that names an
            // allocation stays undecoded, as before; a data word
            // with no provenance of its own is the address-only
            // (or null) value a thin pointer decodes to -- a
            // dangling fat pointer minted from raw integers,
            // whose only observable property is its address.
            let has_data_prov = prov_at(alloc, offset).is_some();
            if has_data_prov {
                return None;
            }
            let addr = read_uint_at(alloc, offset, PTR_SIZE)?;
            return Some(if addr == 0 {
                sm::ConstValue::NullPtr
            } else {
                sm::ConstValue::IntPtr(addr as u64)
            });
        };
        let mir::alloc::GlobalAlloc::VTable(concrete_ty, trait_ref) =
            mir::alloc::GlobalAlloc::from(vtable_prov.0)
        else {
            return None;
        };
        // The vtable this pointer carries is a trait-object
        // formation no body cast names when it happens inside a
        // constant -- a static's initializer, or a promoted
        // constant like the generated test harness's tests
        // array -- so record the vtable's identity exactly as
        // a standalone vtable pointer's decode does. The demand
        // pass then converts the vtable's method bodies (see
        // `Converter::vtable_demand`), without which a dispatch
        // through this pointer finds no concrete body for a
        // method only a vtable entry names (a trait default the
        // impl does not override).
        let id = vtable_id(&concrete_ty, &trait_ref);
        self.note_vtable_identity(id, &concrete_ty, &trait_ref);
        // The bounds the pointer answers to, in the name form
        // body conversion gives this same `dyn` type.
        let sm::Ty::Dynamic(traits, _, _) = self.ty(inner) else {
            return None;
        };
        // The data pointer, decoded exactly as a reference
        // constant of the concrete type decodes: the recursive
        // call follows the data word's own provenance (a static
        // reference when the pointee is a static's storage, a
        // snapshot otherwise). A data word with no provenance is
        // a null or dangling address, as for a thin pointer.
        let has_data_prov = prov_at(alloc, offset).is_some();
        let base = if has_data_prov {
            self.decode_ref(&concrete_ty, alloc, offset, depth + 1)?
        } else {
            let addr = read_uint_at(alloc, offset, PTR_SIZE)?;
            if addr == 0 {
                sm::ConstValue::NullPtr
            } else {
                sm::ConstValue::IntPtr(addr as u64)
            }
        };
        Some(sm::ConstValue::DynRef {
            base: Box::new(base),
            concrete: self.ty(&concrete_ty),
            traits,
            type_id: type_id_segments(&concrete_ty),
        })
    }

    /// Record the (concrete type, principal trait) pair a
    /// vtable identity number denotes. The concrete type
    /// converts through `ty`, exactly as a vtable constant's
    /// own decode renders it, and the trait is the principal's
    /// def name, the form a `Ty::Dynamic` bound carries.
    pub(super) fn note_vtable_identity(
        &mut self,
        id: u64,
        vty: &rustc_public::ty::Ty,
        trait_ref: &Option<rustc_public::ty::Binder<rustc_public::ty::ExistentialTraitRef>>,
    ) {
        if self.vtable_identities.contains_key(&id) {
            return;
        }
        let concrete = self.ty(vty);
        let traits = trait_ref
            .as_ref()
            .map(|binder| vec![binder.value.def_id.name()])
            .unwrap_or_default();
        // The concrete type's `TypeId` segments, from the
        // compiler's own spelling of the type: a trait object
        // recovered through this identity answers `dyn Any`
        // downcasts with them (see `ir::VTableIdentity`).
        let type_id = type_id_segments(vty);
        self.vtable_identities
            .insert(id, (concrete, traits, type_id));
        if let Some(binder) = trait_ref {
            // As in `dyn_trait_refs`: erase the existential
            // arguments' binder regions so the demanded
            // reference can be asked for its vtable.
            let erased = ExistentialTraitRef {
                def_id: binder.value.def_id,
                generic_args: erase_bound_regions_args(&binder.value.generic_args),
            };
            self.vtable_demand.push(erased.with_self_ty(*vty));
        }
    }

    /// Decode a reference whose provenance names the static
    /// `def`: a [`sm::ConstValue::StaticRef`] carrying the static's
    /// full initializer as the store seed and the projection from
    /// the static's base to the pointee. `None` when the projection
    /// cannot be determined (the caller then falls back to a
    /// snapshot of the pointee: reads stay correct, writes are
    /// rejected like any other constant-aggregate write).
    fn decode_static_ref(
        &mut self,
        def: &mir::mono::StaticDef,
        inner: &rustc_public::ty::Ty,
        alloc: &Allocation,
        target_off: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        let def_id = self.crate_def_id(def);
        // The static's declared type, recorded by the item scan
        // (see `note_static_ty`). A static with no recorded type —
        // a foreign static, or an anonymous promoted one — is in
        // practice reached through references that denote it
        // whole, whose pointee type is the static's own type, so a
        // pointee-typed seed stays consistent for it.
        let Some(static_ty) = self.static_tys.get(&def_id).cloned() else {
            let seed = self.decode_const(inner, alloc, target_off, depth + 1)?;
            return Some(sm::ConstValue::StaticRef {
                def: def_id,
                value: Box::new(seed),
                projection: Vec::new(),
                len: None,
            });
        };
        let projection = self.static_offset_path(&static_ty, target_off, inner, depth)?;
        // A reference back to this static from inside its own
        // seed decode closes an initializer cycle: the seed
        // cannot be decoded by value (the decode would recurse
        // until the depth budget refuses the whole constant),
        // so the nested reference carries the uninitialized
        // placeholder seed the `static_seed_decoding` field
        // describes. The enclosing decode produces the real
        // seed, and the interpreter's seed table ranks it above
        // the placeholder.
        if !self.static_seed_decoding.insert(def_id) {
            return Some(sm::ConstValue::StaticRef {
                def: def_id,
                value: Box::new(sm::ConstValue::Uninit),
                projection,
                len: None,
            });
        }
        let seed = self.decode_const(&static_ty, alloc, 0, depth + 1);
        self.static_seed_decoding.remove(&def_id);
        let seed = seed?;
        Some(sm::ConstValue::StaticRef {
            def: def_id,
            value: Box::new(seed),
            projection,
            len: None,
        })
    }

    /// Decode a fat slice reference whose provenance names the
    /// static `def`, as a [`sm::ConstValue::StaticRef`] carrying
    /// the slice length: the evaluated value is a fat reference
    /// over the static reference, in the shape the runtime
    /// unsizing cast builds, so writes through it reach the
    /// static's storage. The pointee must be the static's whole
    /// array storage — the shape a promoted array in a static
    /// initializer takes (`static mut S: &mut [T] = &mut [..]`),
    /// and a reference to a whole array static unsized to a
    /// slice. Anything else (an interior subslice, an array
    /// field inside a larger static) returns `None`, and the
    /// caller falls back to a snapshot of the elements: reads
    /// stay correct, writes are rejected like any other
    /// constant-aggregate write.
    fn decode_static_slice_ref(
        &mut self,
        def: &mir::mono::StaticDef,
        elem: &rustc_public::ty::Ty,
        alloc: &Allocation,
        target_off: usize,
        count: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if target_off != 0 {
            return None;
        }
        let def_id = self.crate_def_id(def);
        // As in `decode_static_ref`: a slice reference back to
        // a static whose seed decode is on the path closes an
        // initializer cycle and carries the placeholder seed
        // (see the `static_seed_decoding` field).
        if !self.static_seed_decoding.insert(def_id) {
            return Some(sm::ConstValue::StaticRef {
                def: def_id,
                value: Box::new(sm::ConstValue::Uninit),
                projection: Vec::new(),
                len: Some(count as u64),
            });
        }
        let seed = self.decode_static_slice_seed(def_id, elem, alloc, count, depth);
        self.static_seed_decoding.remove(&def_id);
        let seed = seed?;
        Some(sm::ConstValue::StaticRef {
            def: def_id,
            value: Box::new(seed),
            projection: Vec::new(),
            len: Some(count as u64),
        })
    }

    /// The seed of the array static a fat slice reference
    /// denotes whole (see `decode_static_slice_ref`): the
    /// static's initializer decoded against its declared array
    /// type, or, for an anonymous promoted static with no
    /// recorded type, the target allocation decoded element by
    /// element exactly as the array arm of `decode_const`
    /// decodes an array value.
    fn decode_static_slice_seed(
        &mut self,
        def_id: sm::DefId,
        elem: &rustc_public::ty::Ty,
        alloc: &Allocation,
        count: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        match self.static_tys.get(&def_id).cloned() {
            Some(static_ty) => {
                // The static's declared type must be the array
                // the slice covers whole: an unsizing of the
                // static itself.
                let TyKind::RigidTy(RigidTy::Array(arr_elem, arr_len)) =
                    static_ty.kind()
                else {
                    return None;
                };
                if !Self::ty_eq_ignoring_regions(&arr_elem, elem)
                    || arr_len.eval_target_usize().ok()? != count as u64
                {
                    return None;
                }
                self.decode_const(&static_ty, alloc, 0, depth + 1)
            }
            None => {
                let stride = elem.layout().ok()?.shape().size.bytes();
                if stride == 0 {
                    // A zero-sized element type stores nothing
                    // per element: one decode replicated to the
                    // length is the exact value, as in
                    // `decode_ref`'s slice arm.
                    if count == 0 {
                        return Some(sm::ConstValue::Tuple(Vec::new()));
                    }
                    if !self.spend_const_budget(count) {
                        return None;
                    }
                    let elem_value = self.decode_const(elem, alloc, 0, depth + 1)?;
                    return Some(sm::ConstValue::Tuple(vec![elem_value; count]));
                }
                let mut elems = Vec::new();
                for i in 0..count {
                    elems.push(self.decode_const(elem, alloc, i * stride, depth + 1)?);
                }
                Some(sm::ConstValue::Tuple(elems))
            }
        }
    }

    /// Whether two types denote the same type up to their
    /// region (lifetime) arguments. A static's declared type
    /// carries the `'static` region where the pointee type of a
    /// pointer at the static carries the erased region, so plain
    /// equality misses references that denote the static whole
    /// (a `*mut &i64` at a `static mut R: &i64`): the reference
    /// then degrades to a snapshot of the initializer and writes
    /// through it are rejected as constant-aggregate writes.
    /// Regions never affect a value's storage shape, so the
    /// projection paths computed for one spelling hold for the
    /// other.
    fn ty_eq_ignoring_regions(a: &Ty, b: &Ty) -> bool {
        if a == b {
            return true;
        }
        match (a.kind(), b.kind()) {
            (
                TyKind::RigidTy(RigidTy::Ref(_, ea, ma)),
                TyKind::RigidTy(RigidTy::Ref(_, eb, mb)),
            ) => ma == mb && Self::ty_eq_ignoring_regions(&ea, &eb),
            (
                TyKind::RigidTy(RigidTy::RawPtr(ea, ma)),
                TyKind::RigidTy(RigidTy::RawPtr(eb, mb)),
            ) => ma == mb && Self::ty_eq_ignoring_regions(&ea, &eb),
            (
                TyKind::RigidTy(RigidTy::Slice(ea)),
                TyKind::RigidTy(RigidTy::Slice(eb)),
            ) => Self::ty_eq_ignoring_regions(&ea, &eb),
            (
                TyKind::RigidTy(RigidTy::Array(ea, la)),
                TyKind::RigidTy(RigidTy::Array(eb, lb)),
            ) => {
                Self::ty_eq_ignoring_regions(&ea, &eb)
                    && format!("{la:?}") == format!("{lb:?}")
            }
            (
                TyKind::RigidTy(RigidTy::Tuple(ea)),
                TyKind::RigidTy(RigidTy::Tuple(eb)),
            ) => {
                ea.len() == eb.len()
                    && ea
                        .iter()
                        .zip(eb.iter())
                        .all(|(x, y)| Self::ty_eq_ignoring_regions(x, y))
            }
            (
                TyKind::RigidTy(RigidTy::Adt(da, aa)),
                TyKind::RigidTy(RigidTy::Adt(db, ab)),
            ) => da == db && Self::generic_args_eq_ignoring_regions(&aa, &ab),
            (
                TyKind::RigidTy(RigidTy::FnDef(da, aa)),
                TyKind::RigidTy(RigidTy::FnDef(db, ab)),
            ) => da == db && Self::generic_args_eq_ignoring_regions(&aa, &ab),
            _ => false,
        }
    }

    /// The argument-wise companion of
    /// [`Self::ty_eq_ignoring_regions`]: lifetime arguments are
    /// ignored, type arguments recurse, and constant arguments
    /// compare by their rendered form (a mismatch keeps the
    /// caller's fallback, as before).
    fn generic_args_eq_ignoring_regions(a: &GenericArgs, b: &GenericArgs) -> bool {
        a.0.len() == b.0.len()
            && a.0.iter().zip(b.0.iter()).all(|(x, y)| match (x, y) {
                (GenericArgKind::Type(ta), GenericArgKind::Type(tb)) => {
                    Self::ty_eq_ignoring_regions(ta, tb)
                }
                (GenericArgKind::Lifetime(_), GenericArgKind::Lifetime(_)) => true,
                (GenericArgKind::Const(ca), GenericArgKind::Const(cb)) => {
                    format!("{ca:?}") == format!("{cb:?}")
                }
                _ => false,
            })
    }

    /// The projection path from the base of a value of type `ty`
    /// to the sub-value at byte offset `off`, when that sub-value
    /// has type `inner`: layout-guided descent through struct and
    /// union fields, tuple elements and array elements. Enum
    /// payloads are not descended (the active variant is a fact of
    /// the stored tag this layer does not read for projections);
    /// a whole-value reference is the empty path.
    fn static_offset_path(
        &mut self,
        ty: &rustc_public::ty::Ty,
        off: usize,
        inner: &rustc_public::ty::Ty,
        depth: usize,
    ) -> Option<Vec<sm::PlaceElem>> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        if off == 0 && Self::ty_eq_ignoring_regions(ty, inner) {
            return Some(Vec::new());
        }
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Adt(def, args)) => match def.kind() {
                AdtKind::Struct => {
                    let variant = def.variant(VariantIdx::to_val(0))?;
                    let offsets = arbitrary_offsets(ty)?;
                    let fields: Vec<(rustc_public::ty::Ty, usize)> = variant
                        .fields()
                        .iter()
                        .map(|f| f.ty_with_args(&args))
                        .zip(offsets)
                        .collect();
                    let (i, fty, mut rest) =
                        self.record_offset_path(&fields, off, inner, depth)?;
                    let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                    path.append(&mut rest);
                    Some(path)
                }
                AdtKind::Union => {
                    let variant = def.variant(VariantIdx::to_val(0))?;
                    let fields: Vec<(rustc_public::ty::Ty, usize)> = variant
                        .fields()
                        .iter()
                        .map(|f| (f.ty_with_args(&args), 0))
                        .collect();
                    let (i, fty, mut rest) =
                        self.record_offset_path(&fields, off, inner, depth)?;
                    let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                    path.append(&mut rest);
                    Some(path)
                }
                AdtKind::Enum => None,
            },
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                let offsets = arbitrary_offsets(ty)?;
                let fields: Vec<(rustc_public::ty::Ty, usize)> =
                    elems.iter().cloned().zip(offsets).collect();
                let (i, fty, mut rest) = self.record_offset_path(&fields, off, inner, depth)?;
                let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                path.append(&mut rest);
                Some(path)
            }
            TyKind::RigidTy(RigidTy::Array(elem, _)) => {
                let stride = elem.layout().ok()?.shape().size.bytes() as usize;
                if stride == 0 || off % stride != 0 {
                    return None;
                }
                let idx = off / stride;
                let count =
                    ty.layout().ok()?.shape().size.bytes() as usize / stride;
                if idx >= count {
                    return None;
                }
                let mut rest = self.static_offset_path(&elem, 0, inner, depth + 1)?;
                let mut path = vec![sm::PlaceElem::ConstantIndex {
                    offset: idx as u64,
                    min_length: count as u64,
                    from_end: false,
                }];
                path.append(&mut rest);
                Some(path)
            }
            _ => None,
        }
    }

    /// The projection path from the base of a value of type
    /// `ty` to the array at byte offset `off` whose element type
    /// is `elem` and whose length is `count` \u2014 the backing
    /// array of a fat slice reference into a static's storage.
    /// The descent mirrors `static_offset_path`, with the array
    /// itself as the leaf: a slice starting at an array's base
    /// and fitting inside it denotes that array (the empty path;
    /// the reference's length word bounds the view), and an
    /// array element that is itself the backing array is
    /// reached through a constant index.
    fn static_slice_path(
        &mut self,
        ty: &rustc_public::ty::Ty,
        off: usize,
        elem: &rustc_public::ty::Ty,
        count: usize,
        depth: usize,
    ) -> Option<Vec<sm::PlaceElem>> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Array(arr_elem, arr_len)) => {
                if arr_elem == *elem && off == 0 {
                    // The slice starts at this array's base and
                    // denotes the array itself when it fits.
                    if count as u64 <= arr_len.eval_target_usize().ok()? {
                        return Some(Vec::new());
                    }
                    return None;
                }
                // An array element that is itself the backing
                // array, reached at its own offset.
                let stride = arr_elem.layout().ok()?.shape().size.bytes() as usize;
                if stride == 0 || off % stride != 0 {
                    return None;
                }
                let idx = off / stride;
                let total =
                    ty.layout().ok()?.shape().size.bytes() as usize / stride;
                if idx >= total {
                    return None;
                }
                let mut rest =
                    self.static_slice_path(&arr_elem, 0, elem, count, depth + 1)?;
                let mut path = vec![sm::PlaceElem::ConstantIndex {
                    offset: idx as u64,
                    min_length: total as u64,
                    from_end: false,
                }];
                path.append(&mut rest);
                Some(path)
            }
            TyKind::RigidTy(RigidTy::Adt(def, args)) => match def.kind() {
                AdtKind::Struct => {
                    let variant = def.variant(VariantIdx::to_val(0))?;
                    let offsets = arbitrary_offsets(ty)?;
                    let fields: Vec<(rustc_public::ty::Ty, usize)> = variant
                        .fields()
                        .iter()
                        .map(|f| f.ty_with_args(&args))
                        .zip(offsets)
                        .collect();
                    let (i, fty, mut rest) =
                        self.record_slice_path(&fields, off, elem, count, depth)?;
                    let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                    path.append(&mut rest);
                    Some(path)
                }
                AdtKind::Union => {
                    let variant = def.variant(VariantIdx::to_val(0))?;
                    let fields: Vec<(rustc_public::ty::Ty, usize)> = variant
                        .fields()
                        .iter()
                        .map(|f| (f.ty_with_args(&args), 0))
                        .collect();
                    let (i, fty, mut rest) =
                        self.record_slice_path(&fields, off, elem, count, depth)?;
                    let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                    path.append(&mut rest);
                    Some(path)
                }
                AdtKind::Enum => None,
            },
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                let offsets = arbitrary_offsets(ty)?;
                let fields: Vec<(rustc_public::ty::Ty, usize)> =
                    elems.iter().cloned().zip(offsets).collect();
                let (i, fty, mut rest) =
                    self.record_slice_path(&fields, off, elem, count, depth)?;
                let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                path.append(&mut rest);
                Some(path)
            }
            _ => None,
        }
    }

    /// Find the record field (type and byte offset within the
    /// record) that contains `off` and from which the backing
    /// array of a slice reference (element type `elem`, length
    /// `count`) is reachable, returning the field index, the
    /// field's type, and the remaining path from that field.
    /// Mirrors `record_offset_path`, with `static_slice_path`
    /// as the reachability test.
    fn record_slice_path(
        &mut self,
        fields: &[(rustc_public::ty::Ty, usize)],
        off: usize,
        elem: &rustc_public::ty::Ty,
        count: usize,
        depth: usize,
    ) -> Option<(usize, rustc_public::ty::Ty, Vec<sm::PlaceElem>)> {
        for (i, (fty, foff)) in fields.iter().enumerate() {
            if *foff > off {
                continue;
            }
            let size = fty.layout().ok()?.shape().size.bytes() as usize;
            // A field contains `off` when it starts at or before
            // it and ends after it; a zero-sized field contains
            // only its own offset.
            let contains = if size == 0 { *foff == off } else { off < foff + size };
            if !contains {
                continue;
            }
            if let Some(rest) =
                self.static_slice_path(fty, off - foff, elem, count, depth + 1)
            {
                return Some((i, fty.clone(), rest));
            }
        }
        None
    }

    /// Find the record field (type and byte offset within the
    /// record) that contains `off` and from which a pointee of
    /// type `inner` is reachable, returning the field index, the
    /// field's type, and the remaining path from that field.
    fn record_offset_path(
        &mut self,
        fields: &[(rustc_public::ty::Ty, usize)],
        off: usize,
        inner: &rustc_public::ty::Ty,
        depth: usize,
    ) -> Option<(usize, rustc_public::ty::Ty, Vec<sm::PlaceElem>)> {
        for (i, (fty, foff)) in fields.iter().enumerate() {
            if *foff > off {
                continue;
            }
            let size = fty.layout().ok()?.shape().size.bytes() as usize;
            // A field contains `off` when it starts at or before it
            // and ends after it; a zero-sized field contains only
            // its own offset.
            let contains = if size == 0 { *foff == off } else { off < foff + size };
            if !contains {
                continue;
            }
            if let Some(rest) = self.static_offset_path(fty, off - foff, inner, depth + 1) {
                return Some((i, fty.clone(), rest));
            }
        }
        None
    }
}
