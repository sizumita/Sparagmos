//! The per-body converter: its state, construction, and body/block conversion.

use std::collections::{HashMap, HashSet};

use sparagmos_mir as mir;

use crate::ir;
use crate::value::{Value};

use super::adt_fields::*;
use super::errors::*;
use super::helpers::*;
use super::program::*;
use super::resolve::*;


/// The dispatch chosen for a call that names an `Fn` trait method
/// (see `Converter::fn_trait_or_direct`).
pub(super) enum FnResolution {
    /// Evaluate the receiver as a callable value (`CallTrait`).
    /// The `bool` is whether the call carries a caller-location
    /// stamp as a third argument (the driver appends one when
    /// the resolved `Fn` shim instance is `#[track_caller]`).
    Trait(ir::FnCallKind, Vec<mir::Ty>, bool),
    /// Call the named body directly.
    Direct(String),
}

pub(super) struct Converter<'a> {
    /// Program key -> body, for callee argument typing.
    pub(super) bodies: HashMap<String, &'a mir::Body>,
    /// `DefId` -> program key, for resolving `FnDef` call operands.
    pub(super) def_names: HashMap<mir::DefId, String>,
    /// `DefId` -> name for every named item (the driver's table,
    /// including body-less callees and ADT types).
    pub(super) names: &'a HashMap<mir::DefId, String>,
    /// Declared ABIs of the program's function instances (see
    /// `MirProgram::fn_abis`).
    fn_abis: &'a HashMap<mir::DefId, mir::Abi>,
    /// Definitions the compiler classifies as intrinsics (see
    /// `MirProgram::intrinsic_defs`).
    pub(super) intrinsic_defs: &'a std::collections::HashSet<mir::DefId>,
    /// Declared ADT field types recovered from `Field` projections
    /// across the whole program (see [`AdtFieldTys`]).
    pub(super) adt_fields: AdtFieldTys,
    /// Discriminant tables of the program (see `MirProgram`).
    pub(super) discriminants: &'a mir::AdtDiscriminants,
    /// ADT shape tables of the program (see `MirProgram`).
    pub(super) adt_shapes: &'a mir::AdtShapes,
    /// Instance ADT shape tables of the program (see `MirProgram`).
    pub(super) adt_instance_shapes: &'a mir::AdtInstanceShapes,
    pub(super) adt_niches: &'a mir::AdtNiches,
    /// Coroutine storage-slot tables of the program (see `MirProgram`).
    pub(super) coroutine_layouts: &'a mir::CoroutineLayouts,
    /// Private storage slots for provably zero-sized coroutine
    /// state fields (see `Converter::coroutine_field_slot`):
    /// per instance key, the slot given to each zero-sized
    /// field type. Computed in `Converter::new` from the
    /// program-wide field-type scan, so every conversion round
    /// assigns the same slots.
    pub(super) coroutine_zst_slots: HashMap<String, HashMap<String, u32>>,
    /// ADT instance layouts of the program (see `MirProgram`).
    pub(super) adt_layouts: &'a mir::AdtLayouts,
    /// Runtime layout descriptors of the program's dynamically
    /// sized ADT instances (see `MirProgram`). Consulted to
    /// decide whether a type is unsized (see
    /// `Converter::ty_is_unsized`).
    pub(super) adt_dst_layouts: &'a mir::AdtDstLayouts,
    /// Per-variant field layouts of the program's ADT
    /// instances (see `MirProgram`).
    pub(super) adt_field_layouts: &'a mir::AdtFieldLayouts,
    /// Storage layouts of the program's tuple types (see
    /// `MirProgram`).
    pub(super) tuple_layouts: &'a mir::TupleLayouts,
    /// Direct tag layouts of the program's enum instances
    /// (see `MirProgram`).
    pub(super) adt_tag_layouts: &'a mir::AdtTagLayouts,
    /// ADT instantiations with compiler-certified empty drop
    /// glue (see `MirProgram::drop_trivial_adts`).
    pub(super) drop_trivial_adts: &'a std::collections::HashSet<String>,
    /// ADT instantiations certified to have no `Drop` impl
    /// (see `MirProgram::drop_no_impl_adts`).
    pub(super) drop_no_impl_adts: &'a std::collections::HashSet<String>,
    /// Demanded destructor bodies by instantiation
    /// (see `MirProgram::dtor_bodies`).
    pub(super) dtor_bodies: &'a std::collections::HashMap<String, String>,
    /// Operand counts of ADT aggregate constructions (see
    /// [`AdtAggregateCounts`]).
    pub(super) adt_aggregates: AdtAggregateCounts,
    /// First path segment of the entry function (`{crate}::main`):
    /// the crate whose items the driver's name table enumerates
    /// completely, `Drop` impls included. `None` when the program
    /// has no `main` function.
    pub(super) local_crate: Option<String>,
    /// The bodies trait-impl call resolution may target (see
    /// `resolve_impl_callee`): `None` disables resolution (the
    /// first conversion phase in `convert_all`), `Some` restricts
    /// it to the named bodies (later fixpoint rounds).
    pub(super) impl_targets: Option<HashSet<String>>,
    /// The bodies that failed conversion in the previous fixpoint
    /// round, keyed by program key with the function's plain name:
    /// a call naming one of them is served another way when it
    /// honestly can be (see `resolve_body_callee`). Empty in the
    /// first conversion phase.
    pub(super) unconverted: HashMap<String, String>,
    /// The guest's allocation shims, when the program defines a
    /// `#[global_allocator]`: symbol (`__rust_alloc`, ...) ->
    /// program key of the shim body (`<crate>::_::__rust_alloc`,
    /// ...). In a linked binary the std allocation entry points
    /// bind to these shims; the conversion binds them the same
    /// way (see `resolve_body_callee`). A
    /// `__rust_alloc_error_handler` entry (the shim a
    /// `#[alloc_error_handler]` generates) rides along for the
    /// allocation-error model; no call resolution reads it.
    pub(super) alloc_shims: HashMap<String, String>,
    /// Link symbols of the program's instances (see
    /// `MirProgram::link_symbols`).
    pub(super) link_symbols: &'a HashMap<mir::DefId, String>,
    /// Declared types of the program's statics (see
    /// `MirProgram::static_tys`).
    pub(super) static_tys: &'a mir::StaticTys,
    /// Initializer seeds of the program's statics, collected
    /// while converting `StaticRef` constants (see
    /// [`Self::register_static_seed`]): static key -> (priority,
    /// seed). Behind a `RefCell` because constant conversion
    /// takes `&self`; `convert_all` merges each round's table
    /// into the program's (see `ir::Program::static_seeds`).
    pub(super) static_seeds: std::cell::RefCell<HashMap<String, (u8, ir::Constant)>>,
    /// Payload types of the program's `panic_any` call sites,
    /// keyed by the call's payload type tag (the first
    /// generic-argument tag the call records, which is the
    /// string the raised payload carries as its type). Behind
    /// a `RefCell` because terminator conversion takes
    /// `&self`; the glue itself is built in `convert_all`,
    /// once conversion has settled (see
    /// `ir::Program::payload_drop_glues`).
    pub(super) payload_tys: std::cell::RefCell<HashMap<String, mir::Ty>>,
    /// The bodies a body-less extern declaration can bind to,
    /// by link symbol: symbol -> program key, holding only the
    /// symbols exactly one converted body exports (a symbol
    /// two bodies claim binds nothing, as a duplicate symbol
    /// is a link error natively). See `link_symbol_route`.
    pub(super) link_exports: HashMap<String, String>,
    /// Coverage record index by mangled symbol, from the
    /// program's coverage table: a coverage statement the driver
    /// attributed to an inlined-from instance counts against
    /// that instance's record.
    pub(super) coverage_by_symbol: HashMap<String, u32>,
    /// Scalar-shape tables of the program's union instances (see
    /// [`ir::Program::union_tables`]), interned in
    /// `Converter::new` by [`Self::compute_union_tables`]: every
    /// conversion round recomputes the same tables, so the
    /// indices baked into converted places agree across rounds.
    pub(super) union_tables: Vec<Vec<Option<ir::ScalarShape>>>,
    /// Union instance -> index into `union_tables`.
    union_table_ids: Vec<(mir::AdtDefId, Vec<mir::Ty>, usize)>,
    /// Per-capture "taken by reference" flags of the program's
    /// async closures (see [`ir::Program::async_closure_captures`]),
    /// computed in `Converter::new` from the creation aggregates:
    /// a pure function of the program, so every conversion round
    /// computes the same table.
    pub(super) async_closure_captures: HashMap<String, Vec<ir::AsyncCapture>>,
    /// Materialisation plans of the program (see
    /// [`MirProgram::mat_plans`]), interned in `Converter::new`
    /// into `mat_table` in sorted-key order: every conversion
    /// round interns the same table, so the indices baked into
    /// converted casts agree across rounds.
    mat_table_ids: HashMap<String, usize>,
    pub(super) mat_table: Vec<ir::MatPlan>,
}

impl<'a> Converter<'a> {
    pub(super) fn new(program: &'a MirProgram, keys: &[String]) -> Self {
        let mut bodies = HashMap::new();
        let mut def_names = HashMap::new();
        let mut adt_fields = AdtFieldTys::default();
        let mut adt_aggregates = AdtAggregateCounts::default();
        for (f, key) in program.functions.iter().zip(keys) {
            bodies.insert(key.clone(), &f.body);
            if let Some(def) = f.def_id {
                def_names.insert(def, key.clone());
            }
            scan_body_fields(&mut adt_fields, &f.body);
            scan_body_aggregates(&mut adt_aggregates, &mut adt_fields, &f.body);
        }
        let coroutine_zst_slots = coroutine_zst_slot_map(program, &adt_fields);
        // A closure body collected through an `Fn`-trait call
        // instance is recorded under the instance's key (the trait
        // method's name), not under the closure's own `DefId`:
        // register the closure `DefId` for that key so closure
        // values and closure types resolve to the body (see
        // `closure_body_alias`).
        for (f, key) in program.functions.iter().zip(keys) {
            if let Some(def) = closure_body_alias(f) {
                def_names.entry(def).or_insert_with(|| key.clone());
            }
        }
        let local_crate = program
            .functions
            .iter()
            .find(|f| f.name == "main" || f.name.ends_with("::main"))
            .map(|f| f.name.split("::").next().unwrap_or("").to_string());
        let mut alloc_shims: HashMap<String, String> = HashMap::new();
        for key in keys {
            for symbol in [
                "__rust_alloc",
                "__rust_dealloc",
                "__rust_realloc",
                "__rust_alloc_zeroed",
                // The generated shim a `#[alloc_error_handler]`
                // produces, scanned into the same table: the
                // allocation-error model reads it from the
                // program (the conversion-side routing below
                // binds only the four allocation entry points,
                // so its presence changes no call resolution).
                "__rust_alloc_error_handler",
            ] {
                if key.contains("::_::") && key.ends_with(symbol) {
                    alloc_shims
                        .entry(symbol.to_string())
                        .or_insert_with(|| key.clone());
                }
            }
        }
        let mut link_exports: HashMap<String, Option<String>> = HashMap::new();
        for (f, key) in program.functions.iter().zip(keys) {
            let Some(def) = f.def_id else { continue };
            let Some(symbol) = program.link_symbols.get(&def) else {
                continue;
            };
            match link_exports.get(symbol) {
                None => {
                    link_exports.insert(symbol.clone(), Some(key.clone()));
                }
                Some(Some(existing)) if *existing != *key => {
                    link_exports.insert(symbol.clone(), None);
                }
                _ => {}
            }
        }
        let link_exports: HashMap<String, String> = link_exports
            .into_iter()
            .filter_map(|(symbol, key)| key.map(|key| (symbol, key)))
            .collect();
        let mut conv = Converter {
            bodies,
            def_names,
            names: &program.fn_names,
            fn_abis: &program.fn_abis,
            intrinsic_defs: &program.intrinsic_defs,
            adt_fields,
            discriminants: &program.adt_discriminants,
            adt_shapes: &program.adt_shapes,
            adt_instance_shapes: &program.adt_instance_shapes,
            adt_niches: &program.adt_niches,
            coroutine_layouts: &program.coroutine_layouts,
            coroutine_zst_slots,
            adt_layouts: &program.adt_layouts,
            adt_dst_layouts: &program.adt_dst_layouts,
            adt_field_layouts: &program.adt_field_layouts,
            tuple_layouts: &program.tuple_layouts,
            adt_tag_layouts: &program.adt_tag_layouts,
            drop_trivial_adts: &program.drop_trivial_adts,
            drop_no_impl_adts: &program.drop_no_impl_adts,
            dtor_bodies: &program.dtor_bodies,
            adt_aggregates,
            local_crate,
            alloc_shims,
            link_symbols: &program.link_symbols,
            static_tys: &program.static_tys,
            static_seeds: std::cell::RefCell::new(HashMap::new()),
            payload_tys: std::cell::RefCell::new(HashMap::new()),
            link_exports,
            coverage_by_symbol: program
                .coverage
                .functions
                .iter()
                .enumerate()
                .map(|(i, f)| (f.symbol.clone(), i as u32))
                .collect(),
            impl_targets: None,
            unconverted: HashMap::new(),
            union_tables: Vec::new(),
            union_table_ids: Vec::new(),
            async_closure_captures: HashMap::new(),
            mat_table_ids: HashMap::new(),
            mat_table: Vec::new(),
        };
        conv.compute_union_tables(program);
        conv.compute_async_closure_captures(program);
        conv.intern_mat_plans(program);
        conv
    }

    /// A converter for a joint-fixpoint round in `convert_all`:
    /// trait-impl call resolution restricted to `targets` (the
    /// previous round's clean bodies), and calls naming a body in
    /// `unconverted` (the previous round's conversion failures)
    /// served another way where that is honest (see
    /// `resolve_body_callee`).
    pub(super) fn with_resolution(
        program: &'a MirProgram,
        keys: &[String],
        targets: HashSet<String>,
        unconverted: HashMap<String, String>,
    ) -> Self {
        let mut conv = Self::new(program, keys);
        conv.impl_targets = Some(targets);
        conv.unconverted = unconverted;
        conv
    }

    /// Whether the ADT is a union.
    pub(super) fn is_union(&self, def: &mir::AdtDefId) -> bool {
        self.adt_shapes
            .get(def)
            .map(|shape| shape.kind == mir::AdtShapeKind::Union)
            .unwrap_or(false)
    }

    /// Record the per-capture plan of every async closure the
    /// program creates (see [`async_capture_plan`]), keyed by the
    /// closure's program
    /// key - the key the created `Value::Closure` carries.
    fn compute_async_closure_captures(&mut self, program: &MirProgram) {
        // Definitions the driver recorded as coroutine closures
        // from type mentions and zero-sized constants, which no
        // creation aggregate names (a zero-sized async closure
        // is created by a constant, not an aggregate): the
        // definition is an async closure, with its captures'
        // by-reference flags unknown - an empty entry, whose
        // per-capture lookups fall back like a missing one.
        for def in &program.async_closure_defs {
            self.async_closure_captures
                .entry(self.body_key(def))
                .or_insert_with(Vec::new);
        }
        for func in &program.functions {
            for bb in &func.body.basic_blocks {
                for stmt in &bb.statements {
                    let mir::StatementKind::Assign(pair) = &stmt.kind else {
                        continue;
                    };
                    let (_, rv) = &**pair;
                    let mir::Rvalue::Aggregate(kind, _) = rv else {
                        continue;
                    };
                    let mir::AggregateKind::CoroutineClosure {
                        def,
                        generic_args,
                    } = &**kind
                    else {
                        continue;
                    };
                    let plan = {
                        let field_ty = |pointee: &mir::Ty, field: u32| -> Option<mir::Ty> {
                            match pointee {
                                mir::Ty::Tuple(ts) => ts.get(field as usize).cloned(),
                                mir::Ty::Adt(def, args) => self
                                    .adt_fields
                                    .get(*def, args, 0, field)
                                    .cloned()
                                    .or_else(|| {
                                        // The projection-recovered
                                        // table only holds a field
                                        // some collected body
                                        // projects, and a capture's
                                        // pointee fields may be
                                        // projected nowhere in the
                                        // program (their accesses
                                        // live in the closure's own
                                        // body, which conversion
                                        // never sees). The declared
                                        // shape is complete by
                                        // construction, so fall back
                                        // to it, as other consumers
                                        // of the table do.
                                        self.adt_shapes
                                            .get(def)
                                            .and_then(|shape| shape.variants.first())
                                            .and_then(|fields| {
                                                fields.get(field as usize)
                                            })
                                            .map(|fty| subst_ty(fty, args))
                                    }),
                                _ => None,
                            }
                        };
                        async_capture_plan(generic_args, &field_ty)
                    };
                    if let Some(plan) = plan {
                        // A computed plan replaces the empty
                        // placeholder the `async_closure_defs` pass
                        // above recorded for the same definition
                        // (that placeholder stands only for a
                        // definition no creation aggregate names);
                        // among computed plans the first wins.
                        let entry = self
                            .async_closure_captures
                            .entry(self.body_key(def))
                            .or_insert_with(Vec::new);
                        if entry.is_empty() {
                            *entry = plan;
                        }
                    }
                }
            }
        }
    }

    /// Intern the scalar-shape tables of every union instance a
    /// place in the program can project a field of. The instance
    /// set is the closure of the bodies' local types and the
    /// projection-recovered field types under field, pointee and
    /// element types: a place walk starts at a local, and each
    /// step's type is one of those (a `Field` step's type is the
    /// previous type's field type, a `Deref` step's the pointee,
    /// an index step's the element). The closure is a pure
    /// function of the program, so every conversion round
    /// interns the same tables in the same order.
    /// Intern the program's materialisation plans (see
    /// [`MirProgram::mat_plans`]) in sorted-key order. The
    /// interned set is a pure function of the program, so every
    /// conversion round interns the same table in the same
    /// order.
    fn intern_mat_plans(&mut self, program: &MirProgram) {
        let mut keys: Vec<&String> = program.mat_plans.keys().collect();
        keys.sort();
        for key in keys {
            let idx = self.mat_table.len();
            self.mat_table.push(program.mat_plans[key].clone());
            self.mat_table_ids.insert(key.clone(), idx);
        }
    }

    /// The materialisation plan index for a cast target type,
    /// when the driver recorded a plan for its pointee and the
    /// pointee is a type a materialised value can be read as
    /// (not an unsized or `dyn` pointee, whose cast keeps the
    /// metadata word instead).
    pub(super) fn mat_plan_index(&self, target: &mir::Ty) -> Option<usize> {
        if std::env::var_os("NT_DISABLE_MAT").is_some() {
            return None;
        }
        let pointee = ptr_pointee(target)?;
        if matches!(
            pointee,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ) || self.pointee_is_byte_string_dst(pointee)
        {
            return None;
        }
        // A fat target whose pointee is a DST-wrapper ADT (or
        // a tuple-tailed pointee): the pointee is unsized
        // through its tail, exactly as a slice / `str` / `dyn`
        // pointee is, so the cast keeps the metadata word
        // instead of materialising the operand into the
        // pointee's storage form -- materialising a fat
        // operand strips its metadata (`eval_materialize`
        // reduces a fat operand to its base), and a later
        // `PtrMetadata` on the result then finds only the
        // storage bytes.
        if self.fat_tail_meta(target).is_some() {
            return None;
        }
        // A zero-sized pointee: the cast is the identity. The
        // erased block such a pointer denotes keeps its bytes
        // for a later cast to a sized type; materialising it
        // into the zero-sized type's value would discard them.
        if let Some((0, _)) = self.ty_layout(pointee) {
            return None;
        }
        let key = format!("{pointee:?}");
        self.mat_table_ids.get(&key).copied()
    }

    fn compute_union_tables(&mut self, program: &MirProgram) {
        // Each worklist entry carries the chain of (ADT
        // definition, instance size) pairs whose field
        // expansion produced it, where the size is the
        // instance type's node count (see `ty_node_count`).
        // An ADT instance is not expanded when its
        // definition already appears in the chain with an
        // instance no larger than it: a generative
        // definition -- `Expanding<T> { b: *const
        // Expanding<(T, T)> }` in tests/debuginfo/
        // recursive-struct.rs (Issue #480) -- re-enters
        // with strictly larger arguments at every step,
        // minting a new, twice-as-large type forever, and
        // before this guard the walk followed it without
        // bound (the driver grew at ~190 MB/s past 22 GB
        // and was OOM-killed). Re-entries that *shrink* --
        // the `Box<Box<...>>` peel, a recursive type
        // re-entered with the same arguments (whose field
        // types the ancestor's expansion already produced)
        // -- still expand, so ordinary recursive types
        // (`Node { next: Option<Box<Node>> }`, the
        // `LongCycle1 -> ... -> LongCycle4 -> LongCycle1`
        // family) are walked exactly as before. Not
        // expanding a non-shrinking re-entry loses nothing
        // the walk could ever finish: any union instance
        // inside it whose exact instance a body actually
        // uses is still seeded directly (every local type
        // and every recorded instance field type is a
        // seed), and a union that only exists at a deeper
        // generative level has infinitely many sibling
        // instances besides.
        let mut visited: Vec<mir::Ty> = Vec::new();
        let mut worklist: Vec<(mir::Ty, Vec<(mir::AdtDefId, u32)>)> = Vec::new();
        for f in &program.functions {
            for decl in &f.body.local_decls {
                worklist.push((decl.ty.clone(), Vec::new()));
            }
        }
        for (_, ty) in &self.adt_fields.entries {
            worklist.push((ty.clone(), Vec::new()));
        }
        while let Some((ty, chain)) = worklist.pop() {
            let mut ty = ty;
            while let mir::Ty::Pat(inner, _) = ty {
                ty = *inner;
            }
            if visited.contains(&ty) {
                continue;
            }
            visited.push(ty.clone());
            match &ty {
                mir::Ty::Adt(def, args) => {
                    if self.is_union(def) {
                        self.intern_union_table(def, args);
                    }
                    let size = ty_node_count(&ty);
                    if chain.iter().any(|(d, n)| d == def && *n <= size) {
                        continue;
                    }
                    let mut chain = chain.clone();
                    chain.push((*def, size));
                    let variant_count = self
                        .adt_shapes
                        .get(def)
                        .map(|shape| shape.variants.len())
                        .unwrap_or(0);
                    for variant in 0..variant_count {
                        if let Some(ftys) =
                            self.instantiated_field_tys(def, args, variant as u32)
                        {
                            worklist.extend(
                                ftys.into_iter().map(|fty| (fty, chain.clone())),
                            );
                        }
                    }
                }
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => {
                    worklist.push(((**inner).clone(), chain));
                }
                mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => {
                    worklist.push(((**elem).clone(), chain));
                }
                mir::Ty::Tuple(ts) => worklist.extend(
                    ts.iter().cloned().map(|t| (t, chain.clone())),
                ),
                _ => {}
            }
        }
    }

    /// Intern one union instance's per-field scalar shapes.
    fn intern_union_table(&mut self, def: &mir::AdtDefId, args: &[mir::Ty]) {
        if self
            .union_table_ids
            .iter()
            .any(|(d, a, _)| d == def && a.as_slice() == args)
        {
            return;
        }
        let fields = self
            .instantiated_field_tys(def, args, 0)
            .unwrap_or_default();
        let shapes: Vec<Option<ir::ScalarShape>> = fields
            .iter()
            .map(|fty| self.union_field_shape(fty))
            .collect();
        let idx = self.union_tables.len();
        self.union_tables.push(shapes);
        self.union_table_ids.push((*def, args.to_vec(), idx));
    }

    /// The table index of a union instance interned by
    /// [`Self::compute_union_tables`]; `None` for an instance
    /// the closure did not reach (the place then converts as a
    /// plain field projection, the pre-union-fields behaviour).
    pub(super) fn union_table_for(&self, def: &mir::AdtDefId, args: &[mir::Ty]) -> Option<usize> {
        self.union_table_ids
            .iter()
            .find(|(d, a, _)| d == def && a.as_slice() == args)
            .map(|(_, _, idx)| *idx)
    }

    /// The scalar shape of one union field's type: the transmute
    /// shape when the type has one, plus a tuple of equal scalar
    /// shapes — its elements occupy consecutive slots of one
    /// width in every field order, so the tuple's bit pattern is
    /// the array pattern over the shared element shape (and a
    /// tuple value and an array value are the same tuple of
    /// elements in this model).
    fn union_field_shape(&self, ty: &mir::Ty) -> Option<ir::ScalarShape> {
        if let Some(shape) = self.scalar_shape(ty, 0) {
            return Some(shape);
        }
        if let mir::Ty::Tuple(ts) = ty {
            if ts.is_empty() || ts.len() > 128 {
                return None;
            }
            let first = self.scalar_shape(&ts[0], 0)?;
            if first.bits() == 0 || first.bits().checked_mul(ts.len() as u32)? > 128 {
                return None;
            }
            for t in &ts[1..] {
                if self.scalar_shape(t, 0).as_ref() != Some(&first) {
                    return None;
                }
            }
            return Some(ir::ScalarShape::Array(
                Box::new(first),
                ts.len() as u64,
            ));
        }
        None
    }

    /// The element size of a local's declared type when the
    /// type is a sequence type: an array's element type, or an
    /// ADT's first generic argument (a `Vec`'s element type —
    /// the argument the value model holds the ADT's elements
    /// under). Recorded on the converted body (see
    /// `ir::Body::local_elem_sizes`) because the IR type drops
    /// the generic arguments the size is computed from.
    fn local_elem_size(&self, ty: &mir::Ty) -> Option<u64> {
        let elem = match ty {
            mir::Ty::Array(elem, _) => elem,
            mir::Ty::Adt(_, args) => args.first()?,
            _ => return None,
        };
        match self.ty_layout(elem) {
            Some((size, _)) if size > 0 => Some(size as u64),
            _ => None,
        }
    }

    /// The discriminant table of the body's return type, when
    /// that type is a fieldless (C-like) enum: the integer each
    /// variant stands for, in variant order. Baked into the
    /// converted body (see `ir::Body::return_discriminants`)
    /// while the definition's shape and discriminant table are
    /// at hand. `None` for every other return type: an enum
    /// with any fielded variant is not represented as a bare
    /// integer, and its return values cross as aggregates.
    fn return_discriminants(&self, body: &mir::Body) -> Option<Vec<i128>> {
        let ret_ty = &body.local_decls.first()?.ty;
        let mir::Ty::Adt(def, _) = ret_ty else {
            return None;
        };
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Enum
            || !shape.variants.iter().all(|fields| fields.is_empty())
        {
            return None;
        }
        self.discriminants.get(def).cloned()
    }

    pub(super) fn convert_body(&self, func: &MirFunction, key: &str) -> Result<ir::Body, ConvertError> {
        BODY_CONVERSIONS.with(|c| c.set(c.get() + 1));
        let body = &func.body;
        let mut local_tys = Vec::with_capacity(body.local_decls.len());
        for (i, decl) in body.local_decls.iter().enumerate() {
            local_tys.push(self.convert_ty(&decl.ty, &format!("local _{i} of `{key}`"))?);
        }
        // A local whose type is a coroutine-closure type arrives
        // with the unit placeholder as its declared type (the
        // driver cannot ask that type's kind through Stable MIR;
        // see `guarded_ty_kind` in the driver). The one statement
        // that creates such a value - the coroutine-closure
        // aggregate assigned to the bare local - names the
        // closure's definition and captures, so the local's type
        // is the closure type the aggregate describes, exactly as
        // a plain closure local's declared type names its own.
        for bb in &body.basic_blocks {
            for stmt in &bb.statements {
                let mir::StatementKind::Assign(pair) = &stmt.kind else {
                    continue;
                };
                let (place, rv) = &**pair;
                if !place.projection.is_empty() {
                    continue;
                }
                let mir::Rvalue::Aggregate(kind, ops) = rv else {
                    continue;
                };
                let mir::AggregateKind::CoroutineClosure {
                    def,
                    generic_args,
                } = &**kind
                else {
                    continue;
                };
                let upvars = match generic_args.iter().rev().nth(1) {
                    Some(mir::Ty::Tuple(ts)) => ts.len(),
                    _ => ops.len(),
                };
                let idx = place.local.index();
                if idx < local_tys.len() {
                    local_tys[idx] = ir::Ty::Closure(self.body_key(def), upvars);
                }
            }
        }
        let fn_abi = func.def_id.and_then(|id| self.fn_abis.get(&id));
        let mut blocks = Vec::with_capacity(body.basic_blocks.len());
        for (index, bb) in body.basic_blocks.iter().enumerate() {
            blocks.push(self.convert_block(body, index as u32, bb, fn_abi)?);
        }
        let mut out = ir::Body::new(key.to_string(), body.arg_count, local_tys, blocks);
        out.return_discriminants = self.return_discriminants(body);
        out.local_elem_sizes = body
            .local_decls
            .iter()
            .map(|decl| self.local_elem_size(&decl.ty))
            .collect();
        out.local_sizes = body
            .local_decls
            .iter()
            .map(|decl| self.ty_layout(&decl.ty).map(|(size, _)| size))
            .collect();
        out.char_locals = body
            .local_decls
            .iter()
            .map(|decl| matches!(decl.ty, mir::Ty::Char))
            .collect();
        out.field_elem_sizes = body
            .local_decls
            .iter()
            .map(|decl| {
                let mut sizes = std::collections::HashMap::new();
                let mut budget = 4096usize;
                self.collect_field_elem_sizes(
                    &decl.ty,
                    &mut Vec::new(),
                    &mut sizes,
                    0,
                    &mut budget,
                );
                sizes
            })
            .collect();
        out.source_lines = body
            .source_lines
            .iter()
            .map(|b| ir::BlockSourceLines {
                statements: b
                    .statements
                    .iter()
                    .map(|s| ir::SourceLine { file: s.file, line: s.line, col: s.col })
                    .collect(),
                terminator: ir::SourceLine {
                    file: b.terminator.file,
                    line: b.terminator.line,
                    col: b.terminator.col,
                },
            })
            .collect();
        // The debugger interface's variable table, from the
        // body's var-debug-info entries (see `ir::DebugVar`).
        out.debug_vars = body
            .var_debug_info
            .iter()
            .filter_map(|info| {
                // Whether the variable is a `char` or a reference
                // to one at the source level, from the type its
                // debug info was recorded against.
                let ty_is_char = |ty: &mir::Ty| match ty {
                    mir::Ty::Char => true,
                    mir::Ty::Ref(_, inner, _) => matches!(&**inner, mir::Ty::Char),
                    _ => false,
                };
                let lines = (info.source_info.span.lo, info.source_info.span.hi);
                match &info.value {
                    mir::VarDebugInfoContents::Local(local) => Some(ir::DebugVarEntry {
                        name: info.name.clone(),
                        source: ir::DebugVar::Place(local.0 as usize, Vec::new()),
                        is_char: ty_is_char(&body.local_decls.get(local.0 as usize)?.ty),
                        ty: Some(crate::builtins::ty_tag(
                            &body.local_decls.get(local.0 as usize)?.ty,
                            self.names,
                        )),
                        lines,
                    }),
                    mir::VarDebugInfoContents::Place(place) => {
                        // The projection, in the interpreter's
                        // resolved form: only field/downcast steps
                        // carry over (the same restriction the
                        // decoder's local-projection reader works
                        // under); a variable reached through
                        // anything else is not carried.
                        let mut proj: Vec<crate::RefProj> = Vec::new();
                        for elem in &place.projection {
                            match elem {
                                mir::PlaceElem::Field(i, _) => {
                                    proj.push(crate::RefProj::Field(*i as usize))
                                }
                                mir::PlaceElem::Downcast(_, v) => {
                                    proj.push(crate::RefProj::Downcast(*v))
                                }
                                _ => return None,
                            }
                        }
                        // Only plain locals carry the char fact; a
                        // projected variable's element type is not
                        // tracked here.
                        let is_char = place.projection.is_empty()
                            && ty_is_char(&body.local_decls.get(place.local.0 as usize)?.ty);
                        // The projected type's tag, when the
                        // projection walk can recover it.
                        let ty = self
                            .debug_projected_ty(
                                &body.local_decls.get(place.local.0 as usize)?.ty,
                                &proj,
                            )
                            .map(|t| crate::builtins::ty_tag(&t, self.names));
                        Some(ir::DebugVarEntry {
                            name: info.name.clone(),
                            source: ir::DebugVar::Place(place.local.0 as usize, proj),
                            is_char,
                            ty,
                            lines,
                        })
                    }
                    mir::VarDebugInfoContents::Constant(c) => {
                        let constant = self.convert_constant(c, c.ty.as_ref()).ok()?;
                        let is_char = c.ty.as_ref().map(|t| ty_is_char(t)).unwrap_or(false);
                        Some(ir::DebugVarEntry {
                            name: info.name.clone(),
                            source: ir::DebugVar::Constant(constant),
                            is_char,
                            ty: c
                                .ty
                                .as_ref()
                                .map(|t| crate::builtins::ty_tag(t, self.names)),
                            lines,
                        })
                    }
                }
            })
            .collect();
        out.unwind = body
            .basic_blocks
            .iter()
            .map(|bb| {
                use mir::TerminatorKind as T;
                match bb.terminator.as_ref().map(|t| &t.kind) {
                    Some(
                        T::Call { unwind, .. }
                        | T::Assert { unwind, .. }
                        | T::Drop { unwind, .. }
                        | T::InlineAsm { unwind, .. },
                    ) => convert_unwind_action(unwind),
                    _ => ir::UnwindAction::Continue,
                }
            })
            .collect();
        // A block whose call conversion diverted to one of the
        // nounwind ub-check helpers (see `ub_check_terminator`):
        // every panic the collected body can raise crosses a
        // `Terminate` boundary inside the body (its
        // `panic_nounwind` / `panic_nounwind_fmt` call), so the
        // call block's action is `Terminate` too — a panic the
        // builtin node raises aborts exactly as the body's did,
        // instead of unwinding to a `catch_unwind`. (The
        // precondition shims are `#[rustc_nounwind]`, so their
        // blocks already carry `Terminate`; this keeps the
        // `ub_checks::runtime` predicate's blocks consistent.)
        for (i, block) in out.blocks.iter().enumerate() {
            if matches!(out.unwind[i], ir::UnwindAction::Terminate(_)) {
                continue;
            }
            if let ir::TerminatorKind::CallBuiltin { name, .. } = &block.terminator {
                if crate::builtins::ub_check_helper(name)
                    .is_some_and(|helper| helper.panics_are_nounwind())
                {
                    out.unwind[i] =
                        ir::UnwindAction::Terminate(ir::UnwindTerminateReason::CannotUnwind);
                }
            }
        }
        out.zst_locals = body
            .local_decls
            .iter()
            .map(|decl| self.zst_value(&decl.ty, 0))
            .collect();
        out.zst_fields = body
            .local_decls
            .iter()
            .map(|decl| {
                let mut map = std::collections::HashMap::new();
                // The walk budget bounds the conversion cost on
                // bodies whose types are enormous (iterator chains
                // with type lengths in the millions): the paths it
                // serves are shallow wrappers, collected long
                // before the budget runs out.
                let mut budget = 256usize;
                self.collect_zst_fields(&decl.ty, &mut Vec::new(), &mut map, 0, &mut budget);
                map
            })
            .collect();
        Ok(out)
    }

    /// The canonical value of a type that is provably zero-sized,
    /// or `None` when the type is not provably zero-sized (or its
    /// value is not uniquely determined by the type). rustc MIR
    /// never writes the place of a zero-sized local in many bodies
    /// — a unit struct constructor's body is a bare `return` — so
    /// a never-written read of such a place denotes this value.
    ///
    /// The proof is structural: unit, and tuples of zero-sized
    /// types; arrays of zero-sized elements and empty arrays;
    /// structs and enums with a single inhabited variant whose
    /// fields are all zero-sized (from the driver's shape table,
    /// with the instantiation's arguments substituted for the
    /// definition's parameters — a variant with an uninhabited
    /// field type can never be constructed, so `Option<Empty>`
    /// still has the single value `None`); closures whose
    /// captures are all single-valued; function items. Enums with two or more inhabited variants
    /// are excluded (the type does not determine the variant), as
    /// are unions with more than one field (the type does not
    /// determine the active field; a single-field union's value
    /// is its one field's value) and opaque aliases (the hidden
    /// type is not visible here).
    pub(super) fn zst_value(&self, ty: &mir::Ty, depth: u32) -> Option<Value> {
        let mut budget = MAX_ZST_VALUE_NODES;
        match self.unique_value(ty, depth, &mut budget) {
            UniqueValue::One(v) => Some(v),
            _ => None,
        }
    }

    /// The one value of the element type of a
    /// `box_assume_init_into_vec_unsafe` argument
    /// (`Box<MaybeUninit<[T; N]>>`), when the element type has
    /// exactly one value: the element the source array's
    /// elided writes would have stored in each unwritten slot.
    /// `None` when the element type has many values (an
    /// unwritten slot is then a genuine uninitialized read, as
    /// before) or the argument's shape is not the expected
    /// box-of-array.
    pub(super) fn into_vec_elem_value(&self, body: &mir::Body, arg: &mir::Operand) -> Option<Value> {
        fn array_elem(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Array(elem, _) => Some(elem),
                mir::Ty::Adt(_, args) => args.iter().find_map(array_elem),
                _ => None,
            }
        }
        let ty = self.operand_place_ty(body, arg)?;
        self.zst_value(array_elem(&ty)?, 0)
    }

    /// Collect the canonical values of the single-valued
    /// projection paths of `ty` into `out`, keyed by path (see
    /// `ir::Body::zst_fields`). Structs, unions and tuples expose
    /// their fields under bare field projections; an enum's fields
    /// Record the element size of every sequence-typed field
    /// reachable from `ty` through struct and tuple fields, by
    /// field-index path (see `ir::Body::field_elem_sizes`). The
    /// walk mirrors `collect_zst_fields`: enum payloads are not
    /// entered (their positions carry a downcast step the
    /// sequence judgment does not read), and the walk is depth-
    /// and budget-bounded like that one.
    fn collect_field_elem_sizes(
        &self,
        ty: &mir::Ty,
        path: &mut Vec<usize>,
        out: &mut std::collections::HashMap<Vec<usize>, u64>,
        depth: u32,
        budget: &mut usize,
    ) {
        if depth > 8 || *budget == 0 {
            return;
        }
        *budget -= 1;
        let mut ty = ty.clone();
        while let mir::Ty::Pat(inner, _) = ty {
            ty = *inner;
        }
        let fields: Vec<mir::Ty> = match &ty {
            mir::Ty::Tuple(ts) => ts.clone(),
            mir::Ty::Adt(def, args) => {
                let Some(shape) = self.adt_shapes.get(def) else {
                    return;
                };
                if shape.kind == mir::AdtShapeKind::Enum {
                    return;
                }
                match self.instantiated_field_tys(def, args, 0) {
                    Some(fields) => fields,
                    None => return,
                }
            }
            _ => return,
        };
        for (i, field_ty) in fields.iter().enumerate() {
            path.push(i);
            if let Some(size) = self.local_elem_size(field_ty) {
                out.insert(path.clone(), size);
            }
            self.collect_field_elem_sizes(field_ty, path, out, depth + 1, budget);
            path.pop();
        }
    }

    /// sit behind a downcast step, so each variant's fields are
    /// collected under that variant's downcast step. The runtime
    /// lookup only consults a recorded path after projection
    /// itself walked it and came back uninitialized — a downcast
    /// to a variant the stored value does not hold fails as a
    /// type mismatch, never as uninitialized — so a path recorded
    /// under one variant's downcast is only ever used for a value
    /// that actually holds that variant. A sequence — an array, a
    /// slice, or a `Vec`, whose modelled value holds its elements
    /// under bare index projections — exposes its element type
    /// under an index step: a zero-sized element's store may be
    /// elided like a ZST field's (a `vec!` of a zero-sized enum
    /// reaches execution with no element write in MIR at all), so
    /// the element's single value is recorded for the read that
    /// comes back uninitialized. The walk is depth- and
    /// size-capped: it serves the shallow paths MIR actually
    /// projects (a wrapper or two around a zero-sized field), not
    /// arbitrary recursive types.
    fn collect_zst_fields(
        &self,
        ty: &mir::Ty,
        path: &mut Vec<ir::ZstPathStep>,
        out: &mut std::collections::HashMap<Vec<ir::ZstPathStep>, Value>,
        depth: u32,
        budget: &mut usize,
    ) {
        if depth > 8 || out.len() > 4096 || *budget == 0 {
            return;
        }
        *budget -= 1;
        let mut ty = ty.clone();
        while let mir::Ty::Pat(inner, _) = ty {
            ty = *inner;
        }
        match &ty {
            mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => {
                let elem = (**elem).clone();
                self.collect_zst_element(&elem, path, out, depth, budget);
                return;
            }
            mir::Ty::Adt(def, args)
                if self.names.get(def).map(String::as_str)
                    == Some("std::vec::Vec") =>
            {
                // A `Vec`'s modelled value holds the elements
                // themselves, so the element type sits under an
                // index step from the Vec place, in addition to
                // the struct fields the walk below records.
                if let Some(elem) = args.first().cloned() {
                    self.collect_zst_element(&elem, path, out, depth, budget);
                }
            }
            _ => {}
        }
        let fields: Vec<mir::Ty> = match &ty {
            mir::Ty::Tuple(ts) => ts.clone(),
            mir::Ty::Adt(def, args) => {
                let Some(shape) = self.adt_shapes.get(def) else {
                    return;
                };
                if shape.kind == mir::AdtShapeKind::Enum {
                    for vi in 0..shape.variants.len() {
                        let Some(fields) =
                            self.instantiated_field_tys(def, args, vi as u32)
                        else {
                            continue;
                        };
                        if *budget == 0 {
                            return;
                        }
                        *budget -= 1;
                        path.push(ir::ZstPathStep::Downcast(vi));
                        self.collect_zst_field_list(
                            &fields,
                            path,
                            out,
                            depth + 1,
                            budget,
                        );
                        path.pop();
                        if out.len() > 4096 {
                            return;
                        }
                    }
                    return;
                }
                match self.instantiated_field_tys(def, args, 0) {
                    Some(fields) => fields,
                    None => return,
                }
            }
            _ => return,
        };
        self.collect_zst_field_list(&fields, path, out, depth, budget);
    }

    /// Record the single-valued values of one field list (a
    /// struct's, a union's, a tuple's, or one enum variant's
    /// fields) under `path`, recursing into each field's own
    /// sub-paths.
    fn collect_zst_field_list(
        &self,
        fields: &[mir::Ty],
        path: &mut Vec<ir::ZstPathStep>,
        out: &mut std::collections::HashMap<Vec<ir::ZstPathStep>, Value>,
        depth: u32,
        budget: &mut usize,
    ) {
        for (i, field_ty) in fields.iter().enumerate() {
            path.push(ir::ZstPathStep::Field(i));
            if let Some(v) = self.zst_value(field_ty, 0) {
                out.insert(path.clone(), v);
            }
            self.collect_zst_fields(field_ty, path, out, depth + 1, budget);
            path.pop();
            if out.len() > 4096 {
                return;
            }
        }
    }

    /// Record the single value of a sequence's element type
    /// under an index step at `path`, and recurse into the
    /// element type's own sub-paths beneath that step (a
    /// single-valued field of the elements sits under
    /// `[Index, Field(..)]`, as a struct's sits under
    /// `[Field(..)]`).
    fn collect_zst_element(
        &self,
        elem: &mir::Ty,
        path: &mut Vec<ir::ZstPathStep>,
        out: &mut std::collections::HashMap<Vec<ir::ZstPathStep>, Value>,
        depth: u32,
        budget: &mut usize,
    ) {
        path.push(ir::ZstPathStep::Index);
        if let Some(v) = self.zst_value(elem, 0) {
            out.insert(path.clone(), v);
        }
        self.collect_zst_fields(elem, path, out, depth + 1, budget);
        path.pop();
    }

    /// Whether a type has exactly one value (and what it is), no
    /// value at all, or many. The one-value case is what
    /// [`Self::zst_value`] synthesizes: a type with a single value
    /// is zero-sized in every layout that matters here, and rustc
    /// MIR may leave its places unwritten. An enum contributes its
    /// unique inhabited variant: a variant with an uninhabited
    /// field type (`Empty`, `!`, a struct containing one) can
    /// never be constructed, so `Option<Empty>` has the single
    /// value `None` even though the enum declares two variants.
    ///
    /// Every value constructed charges its node count against
    /// `budget` (see [`MAX_ZST_VALUE_NODES`]); a construction that
    /// does not fit reports `Many`, so the synthesis never
    /// materialises an enormous value tree. The walk's recursion
    /// depth is capped separately (see [`MAX_ZST_VALUE_DEPTH`]).
    fn unique_value(
        &self,
        ty: &mir::Ty,
        depth: u32,
        budget: &mut u64,
    ) -> UniqueValue {
        if depth > MAX_ZST_VALUE_DEPTH {
            return UniqueValue::Many;
        }
        match ty {
            mir::Ty::Never => UniqueValue::Uninhabited,
            mir::Ty::Tuple(ts) if ts.is_empty() => {
                if !spend_zst_nodes(budget, 1) {
                    return UniqueValue::Many;
                }
                UniqueValue::One(Value::Unit)
            }
            mir::Ty::Tuple(ts) => {
                let mut vals = Vec::with_capacity(ts.len());
                for t in ts {
                    match self.unique_value(t, depth + 1, budget) {
                        UniqueValue::One(v) => vals.push(v),
                        UniqueValue::Uninhabited => return UniqueValue::Uninhabited,
                        UniqueValue::Many => return UniqueValue::Many,
                    }
                }
                if !spend_zst_nodes(budget, 1) {
                    return UniqueValue::Many;
                }
                UniqueValue::One(Value::Tuple(vals))
            }
            mir::Ty::Array(elem, len) => {
                let n = match len {
                    mir::ArrayLen::Known(n) => *n,
                    _ => return UniqueValue::Many,
                };
                if n == 0 {
                    if !spend_zst_nodes(budget, 1) {
                        return UniqueValue::Many;
                    }
                    return UniqueValue::One(Value::Tuple(Vec::new()));
                }
                if n > 4096 {
                    return UniqueValue::Many;
                }
                match self.unique_value(elem, depth + 1, budget) {
                    UniqueValue::One(v) => {
                        // Replicating the element value `n` times
                        // materialises `n - 1` further copies of
                        // its whole tree: charge them before
                        // building, so nested zero-sized arrays
                        // cannot multiply a small type into an
                        // enormous value (the per-level length cap
                        // above alone allows 4096 nodes per level,
                        // multiplied across levels).
                        let extra =
                            value_nodes(&v).saturating_mul(u64::from(n) - 1);
                        if !spend_zst_nodes(budget, extra.saturating_add(1)) {
                            return UniqueValue::Many;
                        }
                        UniqueValue::One(Value::Tuple(vec![v; n as usize]))
                    }
                    UniqueValue::Uninhabited => UniqueValue::Uninhabited,
                    UniqueValue::Many => UniqueValue::Many,
                }
            }
            mir::Ty::Adt(def, args) => {
                let Some(shape) = self.adt_shapes.get(def) else {
                    return UniqueValue::Many;
                };
                if shape.kind == mir::AdtShapeKind::Union
                    && shape.variants.first().map_or(0, |f| f.len()) != 1
                {
                    // A union's active field is not determined by
                    // its type — except a single-field union, whose
                    // one field's value the loop below determines
                    // like a struct's.
                    return UniqueValue::Many;
                }
                let mut single: Option<(u32, Vec<Value>)> = None;
                for (vi, variant_fields) in shape.variants.iter().enumerate() {
                    let mut fields = Vec::with_capacity(variant_fields.len());
                    let mut inhabited = true;
                    for (fi, fty) in variant_fields.iter().enumerate() {
                        // A field type recovered from a projection
                        // in some body is already instantiated for
                        // this `args`; the shape's declared type is
                        // in terms of the definition's parameters,
                        // whose indices also count lifetimes and
                        // consts that the converted argument list
                        // omits, so substitution can misalign.
                        // Prefer the recovered type when one exists.
                        let fty = self
                            .adt_fields
                            .get(*def, args, vi as u32, fi as u32)
                            .cloned()
                            .unwrap_or_else(|| subst_ty(fty, args));
                        match self.unique_value(&fty, depth + 1, budget) {
                            UniqueValue::One(v) => fields.push(v),
                            UniqueValue::Uninhabited => {
                                inhabited = false;
                                break;
                            }
                            // An inhabited variant whose field
                            // has many values gives the type many
                            // values.
                            UniqueValue::Many => return UniqueValue::Many,
                        }
                    }
                    if !inhabited {
                        continue;
                    }
                    if single.is_some() {
                        return UniqueValue::Many;
                    }
                    single = Some((vi as u32, fields));
                }
                match single {
                    Some((variant, fields)) => {
                        if !spend_zst_nodes(budget, 1) {
                            return UniqueValue::Many;
                        }
                        UniqueValue::One(Value::Enum { variant, fields })
                    }
                    None => UniqueValue::Uninhabited,
                }
            }
            mir::Ty::Closure(def, args) => match args.last() {
                Some(mir::Ty::Tuple(ts)) => {
                    let mut env = Vec::with_capacity(ts.len());
                    for t in ts {
                        match self.unique_value(t, depth + 1, budget) {
                            UniqueValue::One(v) => env.push(v),
                            UniqueValue::Uninhabited => {
                                return UniqueValue::Uninhabited
                            }
                            UniqueValue::Many => return UniqueValue::Many,
                        }
                    }
                    if !spend_zst_nodes(budget, 1) {
                        return UniqueValue::Many;
                    }
                    UniqueValue::One(Value::Closure {
                        def: self.def_display_name(def),
                        env,
                    })
                }
                _ => UniqueValue::Many,
            },
            mir::Ty::FnDef(def, _) => {
                if !spend_zst_nodes(budget, 1) {
                    return UniqueValue::Many;
                }
                UniqueValue::One(Value::FnPtr(self.body_key(def)))
            }
            _ => UniqueValue::Many,
        }
    }

    fn convert_block(
        &self,
        body: &mir::Body,
        bb_index: u32,
        bb: &mir::BasicBlockData,
        fn_abi: Option<&mir::Abi>,
    ) -> Result<ir::BasicBlockData, ConvertError> {
        let mut statements = Vec::with_capacity(bb.statements.len());
        for stmt in &bb.statements {
            statements.push(self.convert_statement(body, bb_index, stmt)?);
        }
        let terminator = match &bb.terminator {
            Some(term) => {
                self.convert_terminator(
                    body,
                    bb_index,
                    &term.kind,
                    fn_abi,
                    !bb.statements.is_empty(),
                )?
            }
            None => ir::TerminatorKind::Unsupported("missing terminator".to_string()),
        };
        Ok(ir::BasicBlockData::new(statements, terminator))
    }

}

/// The private storage slots of one program's provably
/// zero-sized coroutine state fields (see
/// `Converter::coroutine_field_slot`): per coroutine instance
/// key, one slot per zero-sized field type, allocated above
/// the instance's recorded slot map in field-type order. The
/// evidence is the program-wide field-type scan: a state field
/// that is ever projected has its declared type there, keyed
/// by the same (definition, arguments, variant, field) the
/// projection carried. A pure function of the program, so
/// every conversion round computes the same table.
fn coroutine_zst_slot_map(
    program: &MirProgram,
    adt_fields: &AdtFieldTys,
) -> HashMap<String, HashMap<String, u32>> {
    let mut by_instance: HashMap<String, std::collections::BTreeSet<String>> =
        HashMap::new();
    for (key, field_ty) in &adt_fields.entries {
        let coroutine_ty = mir::Ty::Coroutine(key.def, key.generic_args.clone());
        let tag = crate::builtins::ty_tag(&coroutine_ty, &program.fn_names);
        if !program.coroutine_layouts.contains_key(&tag) {
            continue;
        }
        if !ty_proven_zero_sized(field_ty, &program.adt_layouts, &program.fn_names) {
            continue;
        }
        by_instance
            .entry(tag)
            .or_default()
            .insert(crate::builtins::ty_tag(field_ty, &program.fn_names));
    }
    let mut out = HashMap::new();
    for (tag, field_tys) in by_instance {
        let slots = &program.coroutine_layouts[&tag];
        let mut next = slots
            .variants
            .iter()
            .flatten()
            .max()
            .map(|max| *max + 1)
            .unwrap_or(0);
        let mut inner = HashMap::new();
        for field_ty in field_tys {
            inner.insert(field_ty, next);
            next += 1;
        }
        out.insert(tag, inner);
    }
    out
}

/// Whether a type is provably zero-sized from the evidence
/// conversion keeps: the unit and never types, tuples and
/// arrays of provably zero-sized elements (an array of a
/// zero-sized element is zero-sized at any length, and a
/// length-0 array is zero-sized at any element), function
/// item types, and ADT / closure / coroutine instances whose
/// recorded layout size is zero. Anything else -- including
/// an instance with no recorded layout -- is not proven.
fn ty_proven_zero_sized(
    ty: &mir::Ty,
    adt_layouts: &mir::AdtLayouts,
    names: &HashMap<mir::DefId, String>,
) -> bool {
    match ty {
        mir::Ty::Never => true,
        mir::Ty::Tuple(elems) => {
            elems.iter().all(|elem| ty_proven_zero_sized(elem, adt_layouts, names))
        }
        mir::Ty::Pat(inner, _) => ty_proven_zero_sized(inner, adt_layouts, names),
        mir::Ty::Array(elem, len) => {
            matches!(len, mir::ArrayLen::Known(0))
                || ty_proven_zero_sized(elem, adt_layouts, names)
        }
        mir::Ty::FnDef(..) => true,
        mir::Ty::Adt(..) | mir::Ty::Closure(..) | mir::Ty::Coroutine(..) => {
            let key = crate::builtins::ty_tag(ty, names);
            matches!(adt_layouts.get(&key), Some((0, _)))
        }
        _ => false,
    }
}

/// The number of nodes in a converted type, counted with a
/// saturating cap: the count only steers the
/// generative-recursion guard in
/// `Converter::compute_union_tables`, which compares
/// instances of one definition along an expansion chain, so
/// exact counts past the cap (where every comparison
/// already reads as "at least as large") would change
/// nothing.
fn ty_node_count(ty: &mir::Ty) -> u32 {
    const CAP: u32 = 512;
    fn count(ty: &mir::Ty, budget: &mut u32) -> u32 {
        if *budget == 0 {
            return 0;
        }
        *budget -= 1;
        let children: u32 = match ty {
            mir::Ty::Adt(_, args)
            | mir::Ty::FnDef(_, args)
            | mir::Ty::Closure(_, args)
            | mir::Ty::Coroutine(_, args)
            | mir::Ty::CoroutineWitness(_, args)
            | mir::Ty::Alias(_, _, args)
            | mir::Ty::Tuple(args) => {
                args.iter().map(|t| count(t, budget)).sum()
            }
            mir::Ty::Array(inner, _)
            | mir::Ty::Pat(inner, _)
            | mir::Ty::Slice(inner)
            | mir::Ty::RawPtr(inner, _)
            | mir::Ty::Ref(_, inner, _)
            | mir::Ty::Dynamic(_, inner, _) => count(inner, budget),
            mir::Ty::FnPtr(sig) => sig
                .inputs
                .iter()
                .map(|t| count(t, budget))
                .sum::<u32>()
                + count(&sig.output, budget),
            _ => 0,
        };
        1 + children
    }
    let mut budget = CAP;
    count(ty, &mut budget)
}
