//! Converter entry points: DefId assignment, body conversion, and source-line resolution.

use super::*;

impl Converter {
    pub fn new() -> Self {
        TY_MENTIONS_PARAM_CACHE.with(|cache| cache.borrow_mut().clear());
        TRAITS_WITH_IMPLS.with(|cache| *cache.borrow_mut() = None);
        crate::reach::clear_expanded_size_cache();
        Self {
            def_ids: HashMap::new(),
            instance_ids: HashMap::new(),
            def_names: Vec::new(),
            intrinsic_defs: std::collections::HashSet::new(),
            async_closure_defs: std::collections::HashSet::new(),
            vtable_identities: HashMap::new(),
            vtable_methods_cache: HashMap::new(),
            vtable_demand: Vec::new(),
            adt_discriminants: sm::AdtDiscriminants::new(),
            adt_layouts: sm::AdtLayouts::new(),
            adt_dst_layouts: sm::AdtDstLayouts::new(),
            adt_field_layouts: sm::AdtFieldLayouts::new(),
            tuple_layouts: sm::TupleLayouts::new(),
            adt_tag_layouts: sm::AdtTagLayouts::new(),
            mat_plans: HashMap::new(),
            coroutine_layouts: sm::CoroutineLayouts::new(),
            coroutine_slots_attempted: std::collections::HashSet::new(),
            layout_attempted: std::collections::HashSet::new(),
            layout_attempted_tys: std::collections::HashSet::new(),
            layout_names: std::collections::HashMap::new(),
            adt_shapes: sm::AdtShapes::new(),
            adt_debug_shapes: sm::AdtDebugShapes::new(),
            env_debug_shapes: sm::EnvDebugShapes::new(),
            env_shape_attempted: std::collections::HashSet::new(),
            debug_shape_depth: 0,
            thread_locals: sm::ThreadLocals::new(),
            extra_static_seeds: sm::ExtraStaticSeeds::new(),
            adt_instance_shapes: sm::AdtInstanceShapes::new(),
            adt_niches: sm::AdtNiches::new(),
            source_files: Vec::new(),
            source_file_ids: HashMap::new(),
            instance_shape_attempted: std::collections::HashSet::new(),
            static_tys: HashMap::new(),
            static_tys_mir: sm::StaticTys::new(),
            track_caller_defs: HashMap::new(),
            track_caller_decl_pending: Vec::new(),
            fn_abis: HashMap::new(),
            fn_shim_locations: [None, None, None],
            coroutine_shim_location: None,
            track_caller_coroutines: Vec::new(),
            link_symbols: HashMap::new(),
            noting_track_caller: false,
            errors: Vec::new(),
            const_closure_mentions: Vec::new(),
            const_closure_mentions_seen: std::collections::HashSet::new(),
            const_fn_mentions: Vec::new(),
            const_fn_mentions_seen: std::collections::HashSet::new(),
            const_budget: MAX_CONST_TOTAL_ELEMS,
            const_budget_failed: false,
            const_scan_failed: false,
            const_scan_budget: MAX_CONST_SCAN_BYTES,
            const_work_failed: false,
            const_work_budget: MAX_CONST_WORK_BYTES,
            static_seed_decoding: std::collections::HashSet::new(),
            alloc_cache: std::collections::HashMap::new(),
            static_inits: std::collections::HashMap::new(),
            foreign_static_defs: std::collections::HashMap::new(),
            alloc_cache_bytes: 0,
            const_memo: std::collections::HashMap::new(),
            source_line_cache: std::collections::HashMap::new(),
        }
    }

    pub(crate) fn err(&mut self, what: impl Into<String>) {
        self.errors.push(what.into());
    }

    /// Map a Stable MIR `DefId` to a Sparagmos `DefId`. Stable MIR does
    /// not expose rustc's raw `(krate, index)` pair, so ids are assigned
    /// sequentially within one driver run (`krate` is always 0 in v0);
    /// `def_names` keeps the corresponding item name for each index.
    pub(crate) fn def_id(&mut self, id: rustc_public::DefId) -> sm::DefId {
        if let Some(&idx) = self.def_ids.get(&id) {
            return sm::DefId::new(0, idx);
        }
        let idx = self.def_names.len() as u32;
        self.def_ids.insert(id, idx);
        self.def_names.push(id.name());
        sm::DefId::new(0, idx)
    }

    /// Record that the compiler classifies the instance `inst`
    /// as a compiler intrinsic (see `intrinsic_defs`). The id
    /// recorded is the instance id the program's call operands
    /// carry, so the interpreter's gate compares like with like.
    pub fn note_intrinsic(&mut self, inst: &Instance) {
        let id = self.instance_def_id(inst.def.def_id(), inst);
        self.intrinsic_defs.insert(id);
    }

    pub fn crate_def_id<D: CrateDef>(&mut self, def: &D) -> sm::DefId {
        self.def_id(def.def_id())
    }

    /// Record the declared type of a static item. Called by the
    /// item scan before any conversion, and by the foreign-static
    /// reference decoder for the statics the scan does not
    /// enumerate: a reference into a static's interior is decoded
    /// against the static's own type (its seed is the full
    /// initializer and the reference carries a projection into
    /// it), and the type query is only safe for real items, so it
    /// is asked only at points that hold one.
    pub fn note_static_ty<D: CrateDef>(&mut self, def: &D, ty: rustc_public::ty::Ty) {
        let id = self.crate_def_id(def);
        let mir_ty = self.ty(&ty);
        self.static_tys_mir.insert(id, mir_ty);
        self.static_tys.insert(id, ty);
    }

    /// Record the definition of the thread-local static `item`
    /// (already assigned the Sparagmos id `id`) in the program's
    /// thread-locals table: its declared type and its initializer,
    /// decoded from the static's initializer allocation the way a
    /// plain static's seed is decoded in `decode_static_ref`. The
    /// declared type comes from the item scan's table when the
    /// static was recorded there; otherwise it is asked of the
    /// item itself, guarded the way the item scan guards it (this
    /// is rvalue conversion, not the constant decoder). A static
    /// whose type or initializer cannot be obtained gets no entry:
    /// the interpreter then reports its `ThreadLocalRef` as an
    /// explicit error instead of inventing storage.
    pub(crate) fn note_thread_local(&mut self, id: sm::DefId, item: &rustc_public::CrateItem) {
        if self.thread_locals.contains_key(&id) {
            return;
        }
        let Some(ty) = self.static_tys.get(&id).cloned().or_else(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| item.ty())).ok()
        }) else {
            return;
        };
        let Ok(def) = mir::mono::StaticDef::try_from(item.clone()) else {
            return;
        };
        // A foreign thread-local static (an `extern` block
        // declaration) has no initializer of its own, but when
        // a definition carrying the imported link symbol is
        // part of the session (an auxiliary crate's
        // `#[thread_local] #[no_mangle]` static, say), the
        // declaration's storage IS the definition's storage,
        // exactly as for a plain foreign static (see
        // `foreign_static_seed`): record the definition's
        // initializer under the declaration's id. A foreign
        // thread-local with no definition in the session keeps
        // no entry and stays the explicit error.
        let def = if static_is_foreign(&def) {
            match foreign_static_definition(&def) {
                Some(definition) => definition,
                None => return,
            }
        } else {
            def
        };
        // Seeding decodes a root constant (the initializer is
        // decoded at the root of `decode_const` just below), so
        // the evaluation in `static_initializer` starts from a
        // full materialisation-work budget rather than whatever
        // an unrelated earlier constant left behind.
        self.const_work_budget = MAX_CONST_WORK_BYTES;
        self.const_work_failed = false;
        let Some(alloc) = self.static_initializer(&def) else {
            return;
        };
        let Some(init) = self.decode_const(&ty, &alloc, 0, 0) else {
            return;
        };
        let ty = self.ty(&ty);
        self.thread_locals.insert(id, sm::ThreadLocalDef { ty, init });
    }

    /// Map a resolved function instance, reached through the
    /// definition `through`, to a Sparagmos `DefId`. The `def_names`
    /// entry is `through`'s name (the spelling call sites and the
    /// builtin registry use), so several instances of one generic
    /// function share a name but never an id.
    /// The instance registry: (referencing definition, instance)
    /// -> the Sparagmos `DefId` the instance was registered
    /// under. Read by the coverage capture to link mono
    /// instances to the bodies they were converted under.
    pub fn instance_ids(
        &self,
    ) -> &std::collections::HashMap<(rustc_public::DefId, Instance), u32> {
        &self.instance_ids
    }

    pub fn instance_def_id(
        &mut self,
        through: rustc_public::DefId,
        inst: &Instance,
    ) -> sm::DefId {
        let key = (through, *inst);
        if let Some(&idx) = self.instance_ids.get(&key) {
            return sm::DefId::new(0, idx);
        }
        let idx = self.def_names.len() as u32;
        self.instance_ids.insert(key, idx);
        self.def_names.push(through.name());
        let id = sm::DefId::new(0, idx);
        self.note_link_symbol(id, inst);
        self.note_track_caller_instance(id, inst);
        self.note_fn_abi(id, inst);
        id
    }

    /// Record the declared ABI of a registered function
    /// instance (see [`Converter::fn_abis`]). Runs once per
    /// registered instance — every instance resolution in the
    /// driver flows through `instance_def_id`, so every body
    /// the program can enter has its ABI recorded under the
    /// very id the body is registered with.
    fn note_fn_abi(&mut self, id: sm::DefId, inst: &Instance) {
        if let Some(abi) = instance_fn_abi(inst) {
            self.fn_abis.insert(id, abi);
        }
    }

    /// Record the declared ABI of a function definition under
    /// its own (non-instance) id: the registration the
    /// auxiliary-crate item loop uses for bodies no instance
    /// demand has registered (see [`Converter::fn_abis`]).
    pub fn note_fn_abi_for_def(&mut self, id: sm::DefId, def: &FnDef) {
        if let Some(abi) = def_fn_abi(def) {
            self.fn_abis.insert(id, abi);
        }
    }

    /// If `inst` is a `#[track_caller]` instance with a body,
    /// record its definition-location stamp (see
    /// `track_caller_defs`). Runs once per registered instance —
    /// every instance resolution in the driver flows through
    /// `instance_def_id`, so every track-caller body the program
    /// can enter is recorded, whether it is ever called directly
    /// or only reached through a function pointer or a vtable.
    fn note_track_caller_instance(&mut self, id: sm::DefId, inst: &Instance) {
        if self.noting_track_caller || !inst.requires_caller_location() {
            return;
        }
        // A virtual instance has no body of its own to take a
        // definition span from (asking the compiler for one is
        // an ICE); the impl instances a dispatch can land on
        // are registered separately and carry the stamps.
        if matches!(inst.kind, InstanceKind::Virtual { .. }) {
            return;
        }
        self.noting_track_caller = true;
        let stamp = std::panic::catch_unwind(
            std::panic::AssertUnwindSafe(|| {
                with_body_materialization(&inst.name(), || inst.body())
                    .and_then(|body| self.def_location_stamp(&body))
            }),
        )
        .ok()
        .flatten();
        self.noting_track_caller = false;
        if let Some(stamp) = stamp {
            self.track_caller_defs.insert(id, stamp);
        } else {
            // No body to take a definition span from: an extern
            // declaration (or an instance whose body the
            // compiler cannot materialise). Its stamp is
            // fabricated from its definition span at the next
            // body conversion (see
            // `fabricate_track_caller_decl_stamps`).
            if !self
                .track_caller_decl_pending
                .iter()
                .any(|(pending, _)| *pending == id)
            {
                self.track_caller_decl_pending.push((id, *inst));
            }
        }
    }

    /// Fabricate the definition-location stamps of the
    /// body-less `#[track_caller]` instances noted so far
    /// (extern declarations): a terminator of `host` re-spanned
    /// to each declaration's definition span and re-scoped to
    /// the host's root scope, resolved with the compiler's own
    /// caller-location resolution -- the fabrication
    /// `def_location_stamp` performs for definition stamps and
    /// `note_coroutine_shim_location` for the resume shim's.
    /// Runs at the end of every body conversion, with the body
    /// just converted as the host: a declaration is noted while
    /// some body that names it converts, so that body is at
    /// hand. A declaration whose span or stamp the compiler
    /// cannot produce simply gets no stamp, like an instance
    /// the compiler cannot name gets no link symbol.
    fn fabricate_track_caller_decl_stamps(&mut self, host: &mir::Body) {
        while let Some((id, inst)) = self.track_caller_decl_pending.pop() {
            if self.track_caller_defs.contains_key(&id) {
                continue;
            }
            let Some(span) = track_caller_decl_def_span(&inst) else {
                continue;
            };
            self.noting_track_caller = true;
            let stamp = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut term = host.blocks.first()?.terminator.clone();
                term.source_info.span = span;
                term.source_info.scope = 0;
                self.call_site_stamp(host, &term)
            }))
            .ok()
            .flatten();
            self.noting_track_caller = false;
            if let Some(stamp) = stamp {
                self.track_caller_defs.insert(id, stamp);
            }
        }
    }

    /// Record the link symbol of the instance registered as
    /// `id` (see `link_symbols`). Runs once per registered
    /// instance — every instance resolution in the driver flows
    /// through `instance_def_id`, so every function the program
    /// can name is recorded, extern declarations included. The
    /// symbol is the compiler's own: `#[no_mangle]` and
    /// `#[export_name]` definitions carry their export name,
    /// extern declarations the name they import (a
    /// `#[link_name]` included), and everything else its
    /// mangled name, which no other instance shares. Asking
    /// the compiler can panic for instances it cannot name (a
    /// virtual instance, say); the panic is contained and the
    /// instance simply gets no symbol, so it never binds by
    /// name.
    fn note_link_symbol(&mut self, id: sm::DefId, inst: &Instance) {
        let symbol = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.link_symbol_of(inst)
        }))
        .ok()
        .flatten();
        if let Some(symbol) = symbol {
            self.link_symbols.insert(id, symbol);
        }
    }

    /// Record the link symbol of the function item `def`,
    /// registered under the Sparagmos `DefId` `id` without an
    /// instance resolution: the auxiliary-crate loop in the
    /// driver binary converts `FnDef` items directly, so their
    /// symbols are noted here instead of in `note_link_symbol`.
    /// The definition is resolved to its monomorphic instance
    /// for the naming; a definition that does not resolve (a
    /// generic one, say) gets no symbol.
    pub fn note_link_symbol_for_def(&mut self, id: sm::DefId, def: rustc_public::DefId) {
        if self.link_symbols.contains_key(&id) {
            return;
        }
        let symbol = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let inst = resolve_instance_guarded(FnDef(def), &GenericArgs(Vec::new()))?;
            self.link_symbol_of(&inst)
        }))
        .ok()
        .flatten();
        if let Some(symbol) = symbol {
            self.link_symbols.insert(id, symbol);
        }
    }

    /// The link symbol the compiler assigns to `inst`, when a
    /// compiler context is installed (outside one there is no
    /// compiler to ask, and no symbol is recorded).
    ///
    /// The query is speculative in the same way an instance
    /// resolution is: the driver asks for the symbols of
    /// generic instances the compiler's own code generation
    /// never names (it names only monomorphized ones), and a
    /// mangler can fail on such an instance as a session
    /// diagnostic rather than a panic — the legacy mangler
    /// reports an internal error for an instance whose
    /// arguments mention an anonymous constant. Left standing,
    /// that one report poisons the whole compilation of a
    /// program rustc accepts. The query therefore runs under
    /// the resolution containment: diagnostics it emits are
    /// discounted, and an instance the compiler cannot name —
    /// by diagnostic or by panic — simply gets no symbol.
    fn link_symbol_of(&self, inst: &Instance) -> Option<String> {
        let tcx = compiler_tcx()?;
        let internal = rustc_public::rustc_internal::internal(tcx, inst);
        let (result, emitted) =
            with_instance_resolution_contained(|| tcx.symbol_name(internal).name.to_string());
        if emitted {
            return None;
        }
        match result {
            Ok(symbol) => Some(symbol),
            Err(_payload) => {
                // The containment returns the payload; the
                // callers' contract is that an unnameable
                // instance gets no symbol, so the panic is
                // consumed here, hook record included (the
                // discipline `resolve_instance_guarded`
                // follows for the same slot).
                let _ = take_captured_panic();
                None
            }
        }
    }

    /// The `&'static Location` of a body's own definition: the
    /// stamp the compiler's call shims present to a
    /// `#[track_caller]` callee entered without a call-site
    /// stamp. Computed with the compiler's own resolution over
    /// a terminator of the body re-spanned to the body's
    /// definition span (the same fabrication
    /// `caller_location_stamp` performs for call sites, whose
    /// terminator already carries the call span).
    fn def_location_stamp(&mut self, body: &mir::Body) -> Option<sm::ConstValue> {
        let mut term = body.blocks.first()?.terminator.clone();
        term.source_info.span = body.span;
        // The first terminator's scope can be an inlined frame's
        // scope (at high MIR optimization levels the body's first
        // call is often `Location::caller` inlined into the body,
        // whose scope names that call site inside the body), and
        // the compiler's caller-location resolution follows the
        // scope, not the span: left standing, the definition
        // stamp resolves to the inlined call's own site in the
        // body instead of the body's definition. The body's own
        // (root) scope carries no inlined call site, so the
        // resolution falls back to the re-spanned definition span.
        term.source_info.scope = 0;
        let stamp = self.call_site_stamp(body, &term)?;
        Some(stamp)
    }

    /// Decode the caller-location stamp the compiler resolves
    /// for the call edge `term` of `body` (see
    /// `caller_location_stamp` for the call-site use).
    fn call_site_stamp(
        &mut self,
        body: &mir::Body,
        term: &mir::Terminator,
    ) -> Option<sm::ConstValue> {
        let mc = body.caller_location(term, None);
        let constant = self.constant(&mir::ConstOperand {
            span: term.source_info.span,
            user_ty: None,
            const_: mc,
        });
        match constant.literal {
            sm::ConstKind::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Record the location stamp of one `Fn`-family vtable
    /// shim: the stamp of the shim body's own call into the
    /// callable body, which is the location a track-caller
    /// closure entered through a `dyn` receiver observes (the
    /// compiler attributes the generated `Fn` impl to the trait
    /// method's definition in core). `kind` is 0 for `call`, 1
    /// for `call_mut`, 2 for `call_once`. The first shim seen
    /// for a kind wins: the location is a property of the trait
    /// method, not of the callable.
    pub fn note_fn_shim_location(&mut self, kind: usize, shim_body: &mir::Body) {
        if self.fn_shim_locations[kind].is_some() || self.noting_track_caller {
            return;
        }
        self.noting_track_caller = true;
        let stamp = shim_body
            .blocks
            .iter()
            .filter_map(|b| match &b.terminator.kind {
                mir::TerminatorKind::Call { .. } => Some(&b.terminator),
                _ => None,
            })
            .next()
            .and_then(|term| self.call_site_stamp(shim_body, term));
        self.noting_track_caller = false;
        if let Some(stamp) = stamp {
            self.fn_shim_locations[kind] = Some(stamp);
        }
    }

    /// Record the location stamp of the `Coroutine::resume`
    /// vtable shim: the definition location of the `resume`
    /// method in core's `Coroutine` trait, which is the
    /// location a track-caller coroutine entered through a
    /// `dyn` receiver observes (the compiler attributes the
    /// resume glue to the trait method's definition, as it
    /// does the `Fn` shims to theirs). The stamp is fabricated
    /// on the body of one concrete resume instance: a
    /// terminator of that body re-spanned to the method's
    /// definition span, resolved with the compiler's own
    /// caller-location resolution (the fabrication
    /// `def_location_stamp` performs for definition stamps).
    pub fn note_coroutine_shim_location(
        &mut self,
        host: &mir::Body,
        trait_def: &rustc_public::ty::TraitDef,
    ) {
        if self.coroutine_shim_location.is_some() || self.noting_track_caller {
            return;
        }
        let Some(span) = coroutine_resume_def_span(trait_def) else {
            return;
        };
        self.noting_track_caller = true;
        let stamp = (|| {
            let mut term = host.blocks.first()?.terminator.clone();
            term.source_info.span = span;
            self.call_site_stamp(host, &term)
        })();
        self.noting_track_caller = false;
        if let Some(stamp) = stamp {
            self.coroutine_shim_location = Some(stamp);
        }
    }

    /// Record the plain definition path of a `#[track_caller]`
    /// coroutine the dyn demand pass met through its resume
    /// shim (see `note_coroutine_shim` in the binary). The
    /// interpreter registers a coroutine body under its own
    /// key, which the resume instance's definition stamp does
    /// not name; this record is what marks that body as a
    /// track-caller body when the program is converted (see
    /// `mark_caller_location_bodies` in the interpreter).
    pub fn note_track_caller_coroutine(&mut self, path: String) {
        if !self.track_caller_coroutines.contains(&path) {
            self.track_caller_coroutines.push(path);
        }
    }

    /// The Sparagmos `DefId` for an `FnDef` type as it appears in a
    /// body: the id of the monomorphic instance the call resolves to.
    /// When the instance cannot be resolved (the definition is not a
    /// resolvable function item), falls back to the definition's own
    /// id, which no converted body carries, so the call stays
    /// unresolved downstream.
    pub(crate) fn fn_def_id(&mut self, def: &FnDef, args: &GenericArgs) -> sm::DefId {
        match resolve_instance_guarded(*def, args) {
            Some(inst) => self.instance_def_id(def.def_id(), &inst),
            None => self.def_id(def.def_id()),
        }
    }


    /// The Sparagmos `DefId` for a closure type as it appears in a
    /// body: the id of the monomorphic closure instance the type
    /// denotes, resolved through `Instance::resolve_closure` with
    /// the closure's own kind — the closure counterpart of
    /// `fn_def_id` for function items. Two instances of one closure
    /// definition (the same closure demanded from different generic
    /// contexts) are different bodies and must not share an id.
    /// A closure type whose arguments still mention generic
    /// parameters (the closure seen from inside the generic body
    /// that defines it) or whose instance cannot be resolved keeps
    /// the definition's own id, which no converted body carries,
    /// so the reference stays unresolved downstream.
    pub fn closure_def_id(
        &mut self,
        def: &rustc_public::ty::ClosureDef,
        args: &GenericArgs,
    ) -> sm::DefId {
        let raw = def.def_id();
        if !args_mention_params(args) {
            if let Some(kind) = closure_kind(args) {
                if let Some(inst) = resolve_closure_guarded(*def, args, kind) {
                    return self.instance_def_id(raw, &inst);
                }
            }
        }
        self.def_id(raw)
    }


    /// Convert one Stable MIR body, containing the panics of
    /// rustc_public 1.100.0's known unimplemented conversions
    /// whose payloads identify them (coroutine-closure types,
    /// unsafe-binder projections, instance-constant evaluation):
    /// a body hitting one fails with the corresponding
    /// `*_REASON` instead of the panic escaping. Any other panic
    /// is resumed — this containment must never paper over an
    /// unrelated compiler failure. (The bare "not implemented"
    /// payloads are identified by location one layer up, in the
    /// driver's `convert_body_isolated`.)
    pub fn convert_body_contained(
        &mut self,
        body: &mir::Body,
    ) -> Result<sm::Body, Vec<String>> {
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.convert_body(body)
        })) {
            Ok(result) => result,
            Err(payload) => {
                if let Some(reason) = stable_mir_panic_reason(&*payload) {
                    Err(vec![reason.to_string()])
                } else {
                    std::panic::resume_unwind(payload);
                }
            }
        };
        // The body just converted is the host for any
        // definition stamps of body-less `#[track_caller]`
        // declarations its conversion noted.
        self.fabricate_track_caller_decl_stamps(body);
        result
    }

    /// Convert one Stable MIR body. On any unsupported construct, returns
    /// the list of reasons (variant names) instead of a body.
    pub fn convert_body(&mut self, body: &mir::Body) -> Result<sm::Body, Vec<String>> {
        self.errors.clear();
        // Issue #481, second half: refuse a body carrying an
        // over-limit local type *before* converting anything in
        // it. The per-type guard in `Converter::ty` already
        // refuses such a type, but only after the locals and
        // statements that spell smaller-but-still-enormous
        // prefixes of the same exponential chain have been
        // materialized one by one: the nested-closure record's
        // main spells thirty bindings whose types double per
        // binding, so conversion built (and dropped) the whole
        // tower up to the limit for every one of the dozens of
        // mentions before the largest type tripped the guard
        // (about 1.4GB peak and 27s for a body that can only
        // ever be skipped). The locals are converted
        // unconditionally, so a body with an over-limit local
        // fails either way; the pre-scan (the same memoized
        // expanded-size walk, costing O(min(expanded size,
        // limit)) per distinct type) makes it fail once, up
        // front, with the same reason. Types that appear only
        // in statements keep the per-type guard in
        // `Converter::ty` as their bound.
        for decl in body.locals() {
            if !crate::reach::ty_within_expanded_size(
                &decl.ty,
                crate::reach::MAX_TY_EXPANDED_SIZE,
            ) {
                return Err(vec![format!(
                    "expanded type tree exceeds the conversion size limit ({} nodes)",
                    crate::reach::MAX_TY_EXPANDED_SIZE
                )]);
            }
        }
        let locals: Vec<sm::LocalDecl> = body
            .locals()
            .iter()
            .map(|d| sm::LocalDecl {
                ty: self.ty(&d.ty),
                mutability: mutability(d.mutability),
            })
            .collect();
        let mut source_lines: Vec<sm::BlockSourceLines> =
            Vec::with_capacity(body.blocks.len());
        let mut unsize_type_ids: Vec<(u32, sm::Place, (u64, u64))> =
            Vec::new();
        let mut unsize_vtable_methods: Vec<(
            u32,
            sm::Place,
            Vec<String>,
        )> = Vec::new();
        let mut blocks: Vec<sm::BasicBlockData> = body
            .blocks
            .iter()
            .enumerate()
            .map(|(index, b)| {
                let (block, lines, ids, methods) =
                    self.basic_block(body, index as u32, b);
                source_lines.push(lines);
                unsize_type_ids.extend(ids);
                unsize_vtable_methods.extend(methods);
                block
            })
            .collect();
        Self::mark_cleanup_region(&mut blocks);
        // The line extent of each source scope in this body,
        // from the resolved lines of the statements and
        // terminators stated directly in it (same-file positions
        // only: the extent is a line range in the body's own
        // file). A debug entry's visibility is its scope, so the
        // entry's recorded extent is its scope's extent, which
        // is what the debugger's name resolution needs to tell a
        // live declaration from a shadowed one whose scope has
        // ended (see var_debug_info).
        let body_file = self.source_line(body.span, body.span).file;
        let mut scope_lines: HashMap<u32, (u32, u32)> = HashMap::new();
        let mut note_scope = |scope_lines: &mut HashMap<u32, (u32, u32)>,
                              scope: u32,
                              pos: sm::SourceLine| {
            if pos.is_known() && pos.file == body_file {
                let entry = scope_lines.entry(scope).or_insert((pos.line, pos.line));
                entry.0 = entry.0.min(pos.line);
                entry.1 = entry.1.max(pos.line);
            }
        };
        // The statements' and terminators' lines were already
        // resolved when the blocks were converted (the
        // `source_lines` above); resolving every span a second
        // time here would double the span-resolution cost of
        // every body, which dominates the conversion of a
        // large body.
        for (bb, lines) in body.blocks.iter().zip(&source_lines) {
            for (stmt, pos) in bb.statements.iter().zip(&lines.statements) {
                note_scope(&mut scope_lines, stmt.source_info.scope, *pos);
            }
            note_scope(
                &mut scope_lines,
                bb.terminator.source_info.scope,
                lines.terminator,
            );
        }
        let debug_info: Vec<sm::VarDebugInfo> = body
            .var_debug_info
            .iter()
            .map(|info| self.var_debug_info(info, &scope_lines))
            .collect();
        if self.errors.is_empty() {
            Self::initialize_const_bound_locals(body, &mut blocks, &debug_info);
            let mut b = sm::Body::new(locals, body.arg_locals().len(), blocks);
            b.var_debug_info = debug_info;
            b.source_lines = source_lines;
            b.unsize_type_ids = unsize_type_ids;
            b.unsize_vtable_methods = unsize_vtable_methods;
            Ok(b)
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    /// Recover the body's cleanup region, which Stable MIR does
    /// not state: it exposes neither `BasicBlockData::is_cleanup`
    /// nor the reason of an `UnwindAction::Terminate`. Both are
    /// structural. A cleanup region is entered only through
    /// `UnwindAction::Cleanup` edges and never branches back into
    /// the body's normal blocks, so the region is exactly the set
    /// of blocks reachable from those edges' targets, less the
    /// blocks reachable from the entry without crossing a cleanup
    /// edge (a block on a normal path is not a cleanup block).
    /// Blocks of the region are marked `is_cleanup`, and a
    /// `Terminate` action stated in the region takes the
    /// `InCleanup` reason — the distinction rustc's own MIR
    /// states directly, and the one codegen uses to choose
    /// between `panic_in_cleanup` ("panic in a destructor during
    /// cleanup") and `panic_cannot_unwind` ("panic in a function
    /// that cannot unwind").
    fn mark_cleanup_region(blocks: &mut [sm::BasicBlockData]) {
        fn unwind_of(kind: &sm::TerminatorKind) -> Option<&sm::UnwindAction> {
            match kind {
                sm::TerminatorKind::Drop { unwind, .. }
                | sm::TerminatorKind::Call { unwind, .. }
                | sm::TerminatorKind::Assert { unwind, .. }
                | sm::TerminatorKind::FalseUnwind { unwind, .. }
                | sm::TerminatorKind::InlineAsm { unwind, .. } => Some(unwind),
                _ => None,
            }
        }
        // Blocks reachable from the entry over normal edges only:
        // a terminator's successors less its cleanup target.
        let mut normal = vec![false; blocks.len()];
        if !blocks.is_empty() {
            let mut stack = vec![0usize];
            while let Some(i) = stack.pop() {
                if i >= blocks.len() || normal[i] {
                    continue;
                }
                normal[i] = true;
                let cleanup_target = blocks[i]
                    .terminator
                    .as_ref()
                    .and_then(|t| unwind_of(&t.kind))
                    .and_then(|u| u.cleanup_block());
                for s in blocks[i].successors() {
                    if Some(s) != cleanup_target {
                        stack.push(s.index());
                    }
                }
            }
        }
        // The cleanup region: everything reachable from a cleanup
        // edge's target (through normal and cleanup edges alike —
        // both stay inside the region), minus the normal blocks.
        let mut cleanup = vec![false; blocks.len()];
        let mut stack: Vec<usize> = Vec::new();
        for data in blocks.iter() {
            if let Some(t) = data
                .terminator
                .as_ref()
                .and_then(|t| unwind_of(&t.kind))
                .and_then(|u| u.cleanup_block())
            {
                stack.push(t.index());
            }
        }
        while let Some(i) = stack.pop() {
            if i >= blocks.len() || cleanup[i] || normal[i] {
                continue;
            }
            cleanup[i] = true;
            for s in blocks[i].successors() {
                stack.push(s.index());
            }
        }
        for (i, data) in blocks.iter_mut().enumerate() {
            if !cleanup[i] {
                continue;
            }
            data.is_cleanup = true;
            let Some(term) = data.terminator.as_mut() else {
                continue;
            };
            let unwind = match &mut term.kind {
                sm::TerminatorKind::Drop { unwind, .. }
                | sm::TerminatorKind::Call { unwind, .. }
                | sm::TerminatorKind::Assert { unwind, .. }
                | sm::TerminatorKind::FalseUnwind { unwind, .. }
                | sm::TerminatorKind::InlineAsm { unwind, .. } => Some(unwind),
                _ => None,
            };
            if let Some(unwind) = unwind {
                if matches!(unwind, sm::UnwindAction::Terminate(_)) {
                    *unwind =
                        sm::UnwindAction::Terminate(sm::UnwindTerminateReason::InCleanup);
                }
            }
        }
    }

    fn var_debug_info(
        &mut self,
        info: &mir::VarDebugInfo,
        scope_lines: &HashMap<u32, (u32, u32)>,
    ) -> sm::VarDebugInfo {
        let value = match &info.value {
            mir::VarDebugInfoContents::Place(p) => {
                sm::VarDebugInfoContents::Place(self.place(p))
            }
            mir::VarDebugInfoContents::Const(c) => {
                sm::VarDebugInfoContents::Constant(Box::new(self.constant(c)))
            }
        };
        // The entry's recorded (start, end) lines: the extent
        // of the lexical scope the variable is visible in,
        // which the debugger interface uses to pick the right
        // entry when one name is declared in several scopes.
        // The entry's own span is only the declaration point;
        // the extent is the line range of the entry's source
        // scope as stated in the body (computed by the caller),
        // widened to cover the declaration. A scope that states
        // nothing leaves the declaration point standing alone.
        // (This model's spans are otherwise dummy; the
        // debug-info span is the one place a resolved extent is
        // carried, as lines, not byte offsets.)
        let (lo, hi) = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            info.source_info.span.get_lines()
        })) {
            Ok(lines) => {
                let (mut lo, mut hi) = (lines.start_line as u32, lines.end_line as u32);
                if let Some(&(smin, smax)) = scope_lines.get(&info.source_info.scope) {
                    lo = lo.min(smin);
                    hi = hi.max(smax);
                }
                (lo, hi)
            }
            Err(_) => (0, 0),
        };
        sm::VarDebugInfo {
            name: info.name.clone(),
            source_info: sm::SourceInfo {
                span: sm::Span { lo, hi },
                scope: 0,
            },
            value,
        }
    }

    /// Optimized MIR can hold a local whose only definition was
    /// const-propagated away: the binding's value survives only as
    /// a `VarDebugInfoContents::Const` entry, while the local
    /// itself is still borrowed, dropped and otherwise used as a
    /// place (a match binding moved out of a constant scrutinee is
    /// the canonical shape). Such a local has no assignment
    /// anywhere in the body, so an executor meets it uninitialized.
    /// When a constant debug entry and a never-written local are
    /// the unique pair of their type, restore the definition as an
    /// assignment at the head of the entry block: the value is a
    /// constant, so it is the same at every point of the body.
    /// Ambiguous shapes (several never-written locals, or several
    /// constants, of one type) are left as they are: a wrong
    /// pairing would silently invent a value, which is worse than
    /// the explicit uninitialized failure.
    fn initialize_const_bound_locals(
        body: &mir::Body,
        blocks: &mut [sm::BasicBlockData],
        debug_info: &[sm::VarDebugInfo],
    ) {
        fn mark(written: &mut [bool], place: &sm::Place) {
            // Any write rooted at the local disqualifies it, not
            // just a whole-local assignment: a local the body
            // builds or mutates through projections (a union
            // overwritten field by field, say) is defined by the
            // body, whatever its debug info claims.
            written[place.local.0 as usize] = true;
        }
        let mut written = vec![false; body.locals().len()];
        for i in 1..=body.arg_locals().len() {
            written[i] = true;
        }
        for block in blocks.iter() {
            for stmt in &block.statements {
                match &stmt.kind {
                    sm::StatementKind::Assign(pair) => mark(&mut written, &pair.0),
                    sm::StatementKind::SetDiscriminant { place, .. } => {
                        mark(&mut written, place)
                    }
                    _ => {}
                }
            }
            if let Some(term) = &block.terminator {
                match &term.kind {
                    sm::TerminatorKind::Call { destination, .. } => {
                        mark(&mut written, destination)
                    }
                    sm::TerminatorKind::Yield { resume_arg, .. } => {
                        mark(&mut written, resume_arg)
                    }
                    sm::TerminatorKind::Drop {
                        place, replace, ..
                    } if *replace => mark(&mut written, place),
                    sm::TerminatorKind::InlineAsm { operands, .. } => {
                        for op in operands {
                            if matches!(
                                op.kind,
                                sm::AsmOperandKind::Out | sm::AsmOperandKind::InOut
                            ) {
                                if let Some(place) = &op.out_place {
                                    mark(&mut written, place);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // Constant debug entries with their source types; the
        // converted debug info preserves the source order, so the
        // index names the same entry in both.
        let consts: Vec<(usize, Ty)> = body
            .var_debug_info
            .iter()
            .enumerate()
            .filter_map(|(i, info)| match &info.value {
                mir::VarDebugInfoContents::Const(c) => Some((i, c.const_.ty())),
                _ => None,
            })
            .collect();
        if consts.is_empty() || blocks.is_empty() {
            return;
        }
        let decl_tys: Vec<Ty> = body.locals().iter().map(|d| d.ty).collect();
        let mut init: Vec<sm::Statement> = Vec::new();
        for (local, ty) in decl_tys.iter().enumerate() {
            if written[local] {
                continue;
            }
            // The local pairs only when it is the sole never-written
            // local of its type and the constant is the sole
            // constant debug entry of that type.
            if decl_tys.iter().enumerate().any(|(other, t)| {
                other != local && t == ty && !written[other]
            }) {
                continue;
            }
            let mut matching = consts.iter().filter(|(_, t)| t == ty);
            let Some((info_idx, _)) = matching.next() else {
                continue;
            };
            if matching.next().is_some() {
                continue;
            }
            if let sm::VarDebugInfoContents::Constant(k) = &debug_info[*info_idx].value {
                // A zero-sized literal carries no value shape: the
                // definition it would restore cannot be rebuilt
                // from it (a union constant decodes to one, and
                // assigning the resulting unit to the union's local
                // is a type error, not a restoration).
                if matches!(k.literal, sm::ConstKind::Value(sm::ConstValue::ZeroSized)) {
                    continue;
                }
                init.push(sm::Statement::new(sm::StatementKind::Assign(Box::new((
                    sm::Place::from_local(sm::Local(local as u32)),
                    sm::Rvalue::Use(sm::Operand::Constant(k.clone())),
                )))));
            }
        }
        if init.is_empty() {
            return;
        }
        init.extend(blocks[0].statements.drain(..));
        blocks[0].statements = init;
    }

    fn basic_block(
        &mut self,
        body: &mir::Body,
        bb_index: u32,
        bb: &mir::BasicBlock,
    ) -> (
        sm::BasicBlockData,
        sm::BlockSourceLines,
        Vec<(u32, sm::Place, (u64, u64))>,
        Vec<(u32, sm::Place, Vec<String>)>,
    ) {
        let statements: Vec<sm::Statement> = bb
            .statements
            .iter()
            .map(|s| sm::Statement::new(self.statement_kind(&s.kind)))
            .collect();
        // The `TypeId` segments of each unsizing cast's concrete
        // type in this block, keyed the way the converted body
        // states them (see `sm::Body::unsize_type_ids`), and the
        // compiler-resolved vtable method names of the same
        // casts (see `sm::Body::unsize_vtable_methods`). The two
        // recordings are independent: either can succeed where
        // the other cannot name its result.
        let mut unsize_type_ids = Vec::new();
        let mut unsize_vtable_methods = Vec::new();
        for (stmt, converted) in bb.statements.iter().zip(&statements) {
            let ids = self.unsize_cast_type_id(body, stmt);
            let methods = self.unsize_cast_vtable_methods(body, stmt);
            if ids.is_none() && methods.is_none() {
                continue;
            }
            if let sm::StatementKind::Assign(pair) = &converted.kind {
                if let Some(ids) = ids {
                    unsize_type_ids.push((bb_index, pair.0.clone(), ids));
                }
                if let Some(methods) = methods {
                    unsize_vtable_methods
                        .push((bb_index, pair.0.clone(), methods));
                }
            }
        }
        let lines = sm::BlockSourceLines {
            statements: bb
                .statements
                .iter()
                .map(|s| self.source_line(s.source_info.span, body.span))
                .collect(),
            terminator: self.terminator_line(body, bb),
        };
        let terminator = self.terminator_kind(body, &bb.terminator);
        let mut data = sm::BasicBlockData::new(sm::Terminator::new(terminator));
        data.statements = statements;
        // Stable MIR (1.100.0) does not expose `is_cleanup`; the
        // cleanup region is recovered structurally in
        // `convert_body` (see `mark_cleanup_region`).
        data.is_cleanup = false;
        (data, lines, unsize_type_ids, unsize_vtable_methods)
    }

    /// The compiler-resolved method instance names of the
    /// vtable the trait reference `tr` denotes: the entries of
    /// the compiler's own vtable for the reference's `Self`,
    /// supertrait vtables included through their `TraitVPtr`
    /// entries, in vtable order. Resolution crosses into the
    /// compiler and can panic on exotic traits; a panicking
    /// reference records nothing, exactly as the dyn demand
    /// pass treats it. Results are memoised per reference (see
    /// `Converter::vtable_methods_cache`).
    ///
    /// A reference that still mentions a generic parameter is
    /// never resolved: it arises from an unsizing cast inside a
    /// generic body, where the cast's "concrete" type is the
    /// parameter itself (or contains it), and asking the
    /// compiler for its vtable ICEs in instance resolution
    /// (`failed to resolve instance for <&T as Debug>::fmt`).
    /// The ICE diagnostic is emitted into the session before
    /// any containment can downgrade it, so the whole
    /// compilation fails; such a reference records nothing and
    /// dispatch keeps the pre-existing resolution chain.
    pub(super) fn vtable_method_names(
        &mut self,
        tr: &rustc_public::ty::TraitRef,
    ) -> Vec<String> {
        let key = format!("{tr:?}");
        if let Some(names) = self.vtable_methods_cache.get(&key) {
            return names.clone();
        }
        let tr_args = tr.args();
        if args_mention_params(&tr_args) {
            self.vtable_methods_cache
                .insert(key, Vec::new());
            return Vec::new();
        }
        let mut names: Vec<String> = Vec::new();
        let mut queue =
            std::collections::VecDeque::from([tr.clone()]);
        let mut seen: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        while let Some(tr) = queue.pop_front() {
            if !seen.insert(format!("{tr:?}")) {
                continue;
            }
            let Ok(entries) = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| tr.vtable_entries()),
            ) else {
                continue;
            };
            for entry in entries {
                match entry {
                    rustc_public::ty::VtblEntry::Method(inst) => {
                        let name = inst.name().to_string();
                        if !names.contains(&name) {
                            names.push(name);
                        }
                    }
                    rustc_public::ty::VtblEntry::TraitVPtr(super_ref) => {
                        queue.push_back(super_ref);
                    }
                    _ => {}
                }
            }
        }
        self.vtable_methods_cache.insert(key, names.clone());
        names
    }

    /// The compiler-resolved vtable method instance names for
    /// the (concrete type, principal trait) pair a vtable
    /// constant denotes: the trait reference is rebuilt with
    /// the concrete type as `Self`, exactly as
    /// `note_vtable_identity` rebuilds it for the demand pass,
    /// and asked for its vtable. Empty when the constant
    /// carries no principal trait reference.
    pub(super) fn vtable_methods_for(
        &mut self,
        vty: &rustc_public::ty::Ty,
        trait_ref: &Option<
            rustc_public::ty::Binder<
                rustc_public::ty::ExistentialTraitRef,
            >,
        >,
    ) -> Vec<String> {
        let Some(binder) = trait_ref else {
            return Vec::new();
        };
        let erased = rustc_public::ty::ExistentialTraitRef {
            def_id: binder.value.def_id,
            generic_args: erase_bound_regions_args(
                &binder.value.generic_args,
            ),
        };
        let tr = erased.with_self_ty(*vty);
        self.vtable_method_names(&tr)
    }

    /// The compiler-resolved vtable method instance names of
    /// an unsizing cast statement, when the statement is a cast
    /// to a `dyn` target and the concrete type can be named by
    /// walking the cast's source and target types in parallel
    /// (see `unsize_concrete_ty`): for each trait bound of the
    /// `dyn` type, the trait reference the cast materializes --
    /// the concrete type as `Self`, the bound's own arguments
    /// with their binder regions erased, as the dyn demand
    /// pass forms it -- is asked for its vtable, and the method
    /// instance names of all bounds are collected in bound
    /// order. `None` when the statement is not such a cast, the
    /// concrete type is itself a `dyn` type (a supertrait
    /// upcast: the value's own vtable carries over), or no
    /// bound's vtable names a method.
    fn unsize_cast_vtable_methods(
        &mut self,
        body: &mir::Body,
        stmt: &mir::Statement,
    ) -> Option<Vec<String>> {
        let mir::StatementKind::Assign(_, rv) = &stmt.kind else {
            return None;
        };
        let mir::Rvalue::Cast(
            mir::CastKind::PointerCoercion(
                mir::PointerCoercion::Unsize,
            ),
            operand,
            target,
        ) = rv
        else {
            return None;
        };
        let locals = body.locals();
        let source = operand.ty(&locals).ok()?;
        let concrete = unsize_concrete_ty(&source, target)?;
        if matches!(
            guarded_ty_kind(&concrete),
            Some(TyKind::RigidTy(RigidTy::Dynamic(..)))
        ) {
            return None;
        }
        let bounds = unsize_dyn_bounds(&source, target)?;
        let mut names: Vec<String> = Vec::new();
        for bound in &bounds {
            let rustc_public::ty::ExistentialPredicate::Trait(et) =
                &bound.value
            else {
                continue;
            };
            let erased = rustc_public::ty::ExistentialTraitRef {
                def_id: et.def_id,
                generic_args: erase_bound_regions_args(
                    &et.generic_args,
                ),
            };
            let tr = erased.with_self_ty(concrete);
            for name in self.vtable_method_names(&tr) {
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
        if names.is_empty() {
            None
        } else {
            Some(names)
        }
    }

    /// The `TypeId` segments of the concrete type of an unsizing
    /// cast statement, when the statement is a cast to a `dyn`
    /// target and the concrete type can be named by walking the
    /// cast's source and target types in parallel (see
    /// `unsize_concrete_ty`). A cast whose concrete type is
    /// itself a `dyn` type (a supertrait upcast: the value
    /// carries its own identity, segments included) and a cast
    /// whose shapes diverge record nothing.
    fn unsize_cast_type_id(
        &mut self,
        body: &mir::Body,
        stmt: &mir::Statement,
    ) -> Option<(u64, u64)> {
        let mir::StatementKind::Assign(_, rv) = &stmt.kind else {
            return None;
        };
        let mir::Rvalue::Cast(
            mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize),
            operand,
            target,
        ) = rv
        else {
            return None;
        };
        let locals = body.locals();
        let source = operand.ty(&locals).ok()?;
        let concrete = unsize_concrete_ty(&source, target)?;
        if matches!(
            guarded_ty_kind(&concrete),
            Some(TyKind::RigidTy(RigidTy::Dynamic(..)))
        ) {
            return None;
        }
        type_id_segments(&concrete)
    }

    /// The source line of one block's terminator. A call is
    /// attributed -- by the compiler's own location resolution
    /// and by native line tables alike -- to the span of its
    /// function operand (the `fn_span` recorded for the call):
    /// for a free function that is the call expression's own
    /// start, and for a method call it is the method name,
    /// while the terminator's own span covers the whole
    /// receiver expression. A method chain split across lines
    /// is therefore attributed to the line of the segment being
    /// called (`.baz()`), not the chain's first line -- the
    /// same span `caller_location_stamp` stamps `#[track_caller]`
    /// locations from. An indirect call (a closure or function
    /// pointer held in a local) has no callee span and keeps
    /// the terminator's own span, as does a callee span the
    /// compiler cannot place.
    fn terminator_line(
        &mut self,
        body: &mir::Body,
        bb: &mir::BasicBlock,
    ) -> sm::SourceLine {
        if let mir::TerminatorKind::Call { func, .. } = &bb.terminator.kind {
            if let mir::Operand::Constant(c) = func {
                let line = self.source_line(c.span, body.span);
                if line.is_known() {
                    return line;
                }
            }
        }
        self.source_line(bb.terminator.source_info.span, body.span)
    }

    /// Resolve one Stable MIR span to the (file, line) the debugger
    /// interface works in, interning the file name into
    /// `source_files`. The span is opaque outside the compiler, so
    /// the resolution must happen here, through the span's own
    /// stable queries; a span the compiler cannot place (a dummy or
    /// foreign-macro span) yields [`sm::SourceLine::UNKNOWN`]
    /// instead of failing the conversion — line information is an
    /// overlay on the body, never a conversion requirement.
    fn source_line(
        &mut self,
        span: rustc_public::ty::Span,
        body_span: rustc_public::ty::Span,
    ) -> sm::SourceLine {
        // The resolution is a pure function of the two spans
        // (the file interning it performs hands out ids by
        // file name, so a repeated resolution would return
        // the same line), and it crosses into the compiler
        // for every call; a body's statements share spans
        // heavily, so resolutions are memoised (see the
        // `source_line_cache` field).
        if let Some(&line) = self.source_line_cache.get(&(span, body_span)) {
            return line;
        }
        let line = self.source_line_uncached(span, body_span);
        self.source_line_cache.insert((span, body_span), line);
        line
    }

    fn source_line_uncached(
        &mut self,
        span: rustc_public::ty::Span,
        body_span: rustc_public::ty::Span,
    ) -> sm::SourceLine {
        let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.collapsed_filename_lines(span, body_span)
                .unwrap_or_else(|| (span.get_filename(), span.get_lines()))
        }));
        let Ok((file, lines)) = resolved else {
            return sm::SourceLine::UNKNOWN;
        };
        if lines.start_line == 0 {
            return sm::SourceLine::UNKNOWN;
        }
        let file_id = match self.source_file_ids.get(&file) {
            Some(&id) => id,
            None => {
                let id = self.source_files.len() as u32;
                self.source_files.push(file.clone());
                self.source_file_ids.insert(file, id);
                id
            }
        };
        sm::SourceLine {
            file: file_id,
            line: lines.start_line as u32,
            col: lines.start_col as u32,
        }
    }

    /// The span of a method call's name within the call's
    /// terminator span, recovered from the call's own source
    /// text. A `#[track_caller]` stamp names the call's
    /// `fn_span` — for a method call, the method name — but at
    /// high MIR optimization levels the function operand can
    /// arrive as a promoted constant whose span was erased, and
    /// the terminator's own span starts at the receiver
    /// expression (the head of the chain), not the segment being
    /// called. The name is the last `.name` occurrence in the
    /// terminator span's source text whose following token opens
    /// the call's argument list: a method chain is written
    /// left-to-right, so the segment being called is the last
    /// one in its own span, ahead of any same-named call in the
    /// receiver or the arguments. Anything else (a free-function
    /// or associated-function call, whose span already starts at
    /// the function path; a span with no source text) yields
    /// `None` and the caller keeps the terminator span.
    pub(crate) fn method_call_name_span(
        &self,
        term_span: rustc_public::ty::Span,
        canonical: &str,
    ) -> Option<rustc_public::ty::Span> {
        let name = canonical.rsplit("::").next()?;
        if name.is_empty()
            || !name.chars().all(|ch| ch == '_' || ch.is_alphanumeric())
        {
            return None;
        }
        let tcx = compiler_tcx()?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let internal_span =
                rustc_public::rustc_internal::internal(tcx, term_span);
            let snippet =
                tcx.sess.source_map().span_to_snippet(internal_span).ok()?;
            let mut found: Option<usize> = None;
            let mut start = 0;
            while let Some(rel) = snippet[start..].find(name) {
                let idx = start + rel;
                let after = snippet[idx + name.len()..]
                    .chars()
                    .find(|ch| !ch.is_whitespace());
                let before = snippet[..idx]
                    .chars()
                    .rev()
                    .find(|ch| !ch.is_whitespace());
                if before == Some('.') && matches!(after, Some('(' | '<' | ':')) {
                    found = Some(idx);
                }
                start = idx + name.len();
            }
            let idx = found?;
            let lo = rustc_span::BytePos(internal_span.lo().0 + idx as u32);
            let named = internal_span.with_lo(lo);
            Some(rustc_public::rustc_internal::stable(named))
        }));
        result.ok().flatten()
    }

    /// Whether the function operand at `operand_span` names
    /// the callee of the call whose terminator span is
    /// `term_span`: in the call's own source text, the first
    /// non-whitespace character after the operand is the
    /// argument list's `(`. A genuine call — a free or
    /// associated function path, a method name — is written
    /// that way. A compiler-rewritten call can instead carry
    /// a moved argument in the function position: the
    /// `const_eval_select` substitution moves the runtime
    /// branch operand into a direct call, keeping the
    /// operand's argument span, and that operand is followed
    /// by the enclosing call's `)` or `,`. For such a call the
    /// location the compiler names is the terminator's own
    /// span (the original call expression), not the operand's.
    /// `None` when the source text cannot adjudicate (no
    /// compiler context, a span with no snippet, an operand
    /// outside the terminator span or in another file): the
    /// caller keeps its standing assumption.
    pub(crate) fn operand_opens_call(
        &self,
        operand_span: rustc_public::ty::Span,
        term_span: rustc_public::ty::Span,
    ) -> Option<bool> {
        if operand_span.get_filename() != term_span.get_filename() {
            return None;
        }
        let tcx = compiler_tcx()?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let term_i = rustc_public::rustc_internal::internal(tcx, term_span);
            let op_i = rustc_public::rustc_internal::internal(tcx, operand_span);
            if op_i.lo() < term_i.lo() || op_i.hi() > term_i.hi() {
                return None;
            }
            let snippet =
                tcx.sess.source_map().span_to_snippet(term_i).ok()?;
            let offset = (op_i.hi().0 - term_i.lo().0) as usize;
            let after = snippet.get(offset..)?;
            Some(after.chars().find(|ch| !ch.is_whitespace()) == Some('('))
        }));
        result.ok().flatten()
    }

    /// The (filename, lines) of a statement span as the
    /// compiler's own debuginfo reports them: the span is first
    /// walked outward through its macro expansion chain to the
    /// outermost call site whose expansion collapses debuginfo
    /// (`rustc_span::hygiene::walk_chain_collapsed` -- the exact
    /// resolution codegen applies to every statement's debug
    /// location in `adjusted_span_and_dbg_scope`), bounded by
    /// the function body's span. A statement written inside a
    /// collapsing macro (a `#[collapse_debuginfo]` macro, an
    /// external macro under the default flag, a builtin macro)
    /// is therefore attributed to the macro's invocation line,
    /// which is the line a debugger stops on. Without a compiler
    /// context (a conversion driven outside the callback) the
    /// span stands unresolved and the caller uses its raw
    /// position.
    pub(crate) fn collapsed_filename_lines(
        &self,
        span: rustc_public::ty::Span,
        body_span: rustc_public::ty::Span,
    ) -> Option<(rustc_public::Filename, rustc_public::ty::LineInfo)> {
        let tcx = compiler_tcx()?;
        let internal_span = rustc_public::rustc_internal::internal(tcx, span);
        let internal_bound = rustc_public::rustc_internal::internal(tcx, body_span);
        let collapsed = rustc_span::hygiene::walk_chain_collapsed(internal_span, internal_bound);
        let public: rustc_public::ty::Span = rustc_public::rustc_internal::stable(collapsed);
        Some((public.get_filename(), public.get_lines()))
    }
}

/// The definition span of a body-less `#[track_caller]`
/// instance (an extern declaration), read through the
/// compiler context installed for the conversion: the span
/// the compiler's own track-caller shim for the declaration
/// presents as the caller location of a call through a
/// function-item value naming it. Outside a compilation
/// callback there is no definition to read.
fn track_caller_decl_def_span(inst: &Instance) -> Option<rustc_public::ty::Span> {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal = rustc_public::rustc_internal::internal(tcx, inst);
        Some(rustc_public::rustc_internal::stable(
            tcx.def_span(internal.def.def_id()),
        ))
    })
}

/// The definition span of the `resume` method of the
/// `Coroutine` trait, read from the trait's own definition
/// through the compiler context installed for the
/// conversion. Outside a compilation callback there is no
/// definition to read.
fn coroutine_resume_def_span(
    trait_def: &rustc_public::ty::TraitDef,
) -> Option<rustc_public::ty::Span> {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal_trait =
            rustc_public::rustc_internal::internal(tcx, trait_def.def_id());
        let resume = rustc_span::Symbol::intern("resume");
        let item = tcx
            .associated_item_def_ids(internal_trait)
            .iter()
            .find(|id| tcx.item_name(**id) == resume)?;
        Some(rustc_public::rustc_internal::stable(tcx.def_span(*item)))
    })
}

/// Whether a compiler type mentions a `dyn` type through
/// references, raw pointers and ADT generic arguments -- the
/// traversal the interpreter's conversion performs on the
/// converted type when it decides an unsizing cast's target.
fn dyn_in_ty(ty: &rustc_public::ty::Ty) -> bool {
    match guarded_ty_kind(ty) {
        Some(TyKind::RigidTy(RigidTy::Dynamic(..))) => true,
        Some(TyKind::RigidTy(RigidTy::Ref(_, inner, _) | RigidTy::RawPtr(inner, _))) => {
            dyn_in_ty(&inner)
        }
        Some(TyKind::RigidTy(RigidTy::Adt(_, args))) => args
            .0
            .iter()
            .any(|a| matches!(a, GenericArgKind::Type(t) if dyn_in_ty(t))),
        _ => false,
    }
}

/// The concrete type an unsizing cast's value answers for, read
/// off the compiler's types: walk the source and target types
/// in parallel along the target's wrapper path to its `dyn`
/// type and return the source type at the same position (the
/// walk the interpreter's conversion performs on the converted
/// types, so the two name the same type). `None` when the
/// target holds no `dyn` type or the shapes diverge.
/// The bounds of the `dyn` type inside `target`, found by
/// walking the cast's source and target types in parallel the
/// way `unsize_concrete_ty` walks them for the concrete type.
fn unsize_dyn_bounds(
    src: &rustc_public::ty::Ty,
    target: &rustc_public::ty::Ty,
) -> Option<
    Vec<
        rustc_public::ty::Binder<
            rustc_public::ty::ExistentialPredicate,
        >,
    >,
> {
    match (guarded_ty_kind(src)?, guarded_ty_kind(target)?) {
        (_, TyKind::RigidTy(RigidTy::Dynamic(bounds, _))) => {
            Some(bounds)
        }
        (
            TyKind::RigidTy(RigidTy::Ref(_, s, _)),
            TyKind::RigidTy(RigidTy::Ref(_, t, _)),
        ) => unsize_dyn_bounds(&s, &t),
        (
            TyKind::RigidTy(RigidTy::RawPtr(s, _)),
            TyKind::RigidTy(RigidTy::RawPtr(t, _)),
        ) => unsize_dyn_bounds(&s, &t),
        (
            TyKind::RigidTy(RigidTy::Adt(_, sargs)),
            TyKind::RigidTy(RigidTy::Adt(_, targs)),
        ) => {
            let s_types: Vec<&rustc_public::ty::Ty> = sargs
                .0
                .iter()
                .filter_map(|a| match a {
                    GenericArgKind::Type(t) => Some(t),
                    _ => None,
                })
                .collect();
            let t_types: Vec<&rustc_public::ty::Ty> = targs
                .0
                .iter()
                .filter_map(|a| match a {
                    GenericArgKind::Type(t) => Some(t),
                    _ => None,
                })
                .collect();
            for (sa, ta) in s_types.iter().zip(t_types.iter()) {
                if dyn_in_ty(ta) {
                    return unsize_dyn_bounds(sa, ta);
                }
            }
            None
        }
        _ => None,
    }
}

fn unsize_concrete_ty(
    src: &rustc_public::ty::Ty,
    target: &rustc_public::ty::Ty,
) -> Option<rustc_public::ty::Ty> {
    match (guarded_ty_kind(src)?, guarded_ty_kind(target)?) {
        (_, TyKind::RigidTy(RigidTy::Dynamic(..))) => Some(*src),
        (
            TyKind::RigidTy(RigidTy::Ref(_, s, _)),
            TyKind::RigidTy(RigidTy::Ref(_, t, _)),
        ) => unsize_concrete_ty(&s, &t),
        (
            TyKind::RigidTy(RigidTy::RawPtr(s, _)),
            TyKind::RigidTy(RigidTy::RawPtr(t, _)),
        ) => unsize_concrete_ty(&s, &t),
        (
            TyKind::RigidTy(RigidTy::Adt(_, sargs)),
            TyKind::RigidTy(RigidTy::Adt(_, targs)),
        ) => {
            let s_types: Vec<&rustc_public::ty::Ty> = sargs
                .0
                .iter()
                .filter_map(|a| match a {
                    GenericArgKind::Type(t) => Some(t),
                    _ => None,
                })
                .collect();
            let t_types: Vec<&rustc_public::ty::Ty> = targs
                .0
                .iter()
                .filter_map(|a| match a {
                    GenericArgKind::Type(t) => Some(t),
                    _ => None,
                })
                .collect();
            for (sa, ta) in s_types.iter().zip(t_types.iter()) {
                if dyn_in_ty(ta) {
                    return unsize_concrete_ty(sa, ta);
                }
            }
            None
        }
        _ => None,
    }
}
