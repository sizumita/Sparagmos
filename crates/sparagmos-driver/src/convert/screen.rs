//! The unnormalizable-projection screen: whether a projection can normalize in a monomorphized typing environment.

use super::*;

/// The recorded reason for a definition the conversion does
/// not pose the compiler's monomorphizing queries for,
/// because its types mention a trait projection that cannot
/// normalize in a fully-monomorphized typing environment
/// (see [`unnormalizable_projection_in_instance`]). The
/// constant-item accounting in the binary recognizes this
/// prefix when it re-accounts a screened constant definition
/// as not demanded, the way it does for a definition that
/// fails only on its own parameters.
pub const UNNORMALIZABLE_PROJECTION_REASON: &str =
    "mentions a projection that does not normalize in a \
     monomorphized typing environment";

thread_local! {
    /// The traits that have at least one implementation in
    /// the running compilation, built once (see
    /// `trait_has_impls`).
    pub(crate) static TRAITS_WITH_IMPLS: std::cell::RefCell<
        Option<HashSet<rustc_public::DefId>>,
    > = const { std::cell::RefCell::new(None) };
}

/// Whether `trait_def` has at least one implementation in
/// the running compilation. Building the table enumerates
/// `all_trait_impls` and instantiates each impl's trait
/// reference, which neither substitutes nor normalizes, so
/// it poses no query the projection screen exists to avoid.
fn trait_has_impls(trait_def: rustc_public::DefId) -> bool {
    TRAITS_WITH_IMPLS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let set = cache.get_or_insert_with(|| {
            let mut set: HashSet<rustc_public::DefId> = HashSet::new();
            for impl_def in rustc_public::all_trait_impls() {
                set.insert(impl_def.trait_impl().value.def_id.def_id());
            }
            set
        });
        set.contains(&trait_def)
    })
}

thread_local! {
    /// Per-trait memo for `trait_is_compiler_implemented`.
    static COMPILER_IMPLEMENTED_TRAITS: std::cell::RefCell<
        Option<HashMap<rustc_public::DefId, bool>>,
    > = const { std::cell::RefCell::new(None) };
}

/// Whether the compiler itself provides `trait_def`'s
/// implementations, so the implementation table is no evidence
/// about the trait at all. The compiler marks such traits
/// `#[rustc_deny_explicit_impl]`: user implementations are
/// forbidden outright, and the compiler supplies the
/// implementations from its own knowledge — every type
/// implements `Pointee` with the metadata its kind carries,
/// every enum implements `DiscriminantKind`, the sized
/// hierarchy and `Unsize` hold by construction. Such a
/// trait's projections normalize for the self types the
/// compiler assigns them to, table or no table, so the screen
/// must not reject them.
///
/// The mark is read two ways, because the attribute is not
/// encoded in dependency metadata: directly, for a trait
/// whose attributes the running compilation itself reads,
/// and by lang item for core's own marked traits, whose lang
/// items are the identity that survives into a dependent
/// compilation. Outside a compilation callback there is no
/// compiler context to consult (see `compiler_tcx`); the
/// trait is reported user-implemented, preserving the
/// screen's plain judgment.
fn trait_is_compiler_implemented(trait_def: rustc_public::DefId) -> bool {
    COMPILER_IMPLEMENTED_TRAITS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let memo = cache.get_or_insert_with(HashMap::new);
        if let Some(&known) = memo.get(&trait_def) {
            return known;
        }
        let implemented = compiler_tcx().is_some_and(|tcx| {
            let internal = rustc_public::rustc_internal::internal(tcx, trait_def);
            let marked = tcx
                .get_attrs(internal, rustc_span::sym::rustc_deny_explicit_impl)
                .into_iter()
                .next()
                .is_some();
            let lang_items = tcx.lang_items();
            let core_compiler_trait = [
                rustc_attr_ir::lang_items::LangItem::PointeeTrait,
                rustc_attr_ir::lang_items::LangItem::DiscriminantKind,
                rustc_attr_ir::lang_items::LangItem::Sized,
                rustc_attr_ir::lang_items::LangItem::MetaSized,
                rustc_attr_ir::lang_items::LangItem::PointeeSized,
                rustc_attr_ir::lang_items::LangItem::Unsize,
                rustc_attr_ir::lang_items::LangItem::Destruct,
                rustc_attr_ir::lang_items::LangItem::Tuple,
                rustc_attr_ir::lang_items::LangItem::BikeshedGuaranteedNoDrop,
            ]
            .iter()
            .any(|item| lang_items.get(*item) == Some(internal));
            marked || core_compiler_trait
        });
        memo.insert(trait_def, implemented);
        implemented
    })
}

/// Whether the projection `alias` cannot normalize in a
/// fully-monomorphized typing environment. The one judgment
/// the implementation table decides: the projection's trait
/// has no implementation at all, in any crate of the
/// compilation. No instantiation of a definition mentioning
/// the projection can satisfy its bounds then, so the
/// projection normalizes for no self type — parameter,
/// concrete, or otherwise — with two exceptions the screen
/// checks separately: a trait the compiler itself implements
/// (see `trait_is_compiler_implemented`), whose
/// implementations never reach the table, and a trait-object
/// self type, whose own existential projection predicate
/// binds the item and so normalizes the projection without
/// any implementation (see `dyn_binding_normalizes`).
///
/// Nothing else is judged. Whether an implementation
/// *covers* a concrete self type is not decidable from the
/// table alone: a `dyn` self type normalizes through its
/// own existential projection predicate, closures and
/// coroutines implement the `Fn` family and `Future`
/// through compiler-synthesized implementations with no
/// `impl` item, and a conditional implementation's bounds
/// are not the table's to check. A screen that guessed at
/// coverage would refuse queries rustc answers fine.
fn projection_cannot_normalize(alias: &rustc_public::ty::AliasTy) -> bool {
    let Some(trait_def) = alias.def_id.def_id().parent() else {
        return false;
    };
    !trait_has_impls(trait_def) && !trait_is_compiler_implemented(trait_def)
}

/// Whether the projection `alias` normalizes through an
/// existential associated-type binding of its self type:
/// the self type, resolved as far as the screen's own
/// evidence reaches (see `resolve_screen_self_ty`), is a
/// trait object whose predicates bind this very associated
/// item (`dyn Trait<Assoc = Term>`). Such a projection
/// normalizes to the binding's term in a fully-monomorphized
/// environment even though the trait has no implementation
/// at all — the object type itself carries the equality —
/// so the absence of implementations is no proof it cannot
/// normalize, and the screen must not reject it. Any other
/// shape — an unresolved parameter, a non-object self type,
/// an object binding a different item of the trait — leaves
/// the implementation-table judgment in force.
fn dyn_binding_normalizes(
    alias: &rustc_public::ty::AliasTy,
    ambient: Option<&GenericArgs>,
    depth: usize,
) -> bool {
    // A projection's generic arguments are its trait
    // reference's, whose first argument is the self type.
    let Some(GenericArgKind::Type(self_ty)) = alias.args.0.first() else {
        return false;
    };
    let Some(resolved) = resolve_screen_self_ty(self_ty, ambient, depth) else {
        return false;
    };
    let Some(TyKind::RigidTy(RigidTy::Dynamic(preds, _))) = guarded_ty_kind(&resolved) else {
        return false;
    };
    preds.iter().any(|pred| {
        matches!(
            &pred.value,
            ExistentialPredicate::Projection(p)
                if p.def_id.def_id() == alias.def_id.def_id()
        )
    })
}

thread_local! {
    /// Whether the screen's type resolution gave up at
    /// its depth cap during the judgment in progress:
    /// set by `resolve_screen_self_ty` when its budget
    /// runs out, and read -- and cleared -- by the one
    /// judgment that consults resolution (see
    /// `projection_screen_ty`), so exhaustion recorded
    /// anywhere else cannot leak into a verdict.
    static RESOLUTION_DEPTH_EXHAUSTED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Note that the resolution in progress exhausted its
/// depth budget (see `RESOLUTION_DEPTH_EXHAUSTED`).
fn note_resolution_depth_exhausted() {
    RESOLUTION_DEPTH_EXHAUSTED.with(|flag| flag.set(true));
}

/// Whether the resolution since the last take exhausted
/// its depth budget, clearing the record (see
/// `RESOLUTION_DEPTH_EXHAUSTED`).
fn take_resolution_depth_exhausted() -> bool {
    RESOLUTION_DEPTH_EXHAUSTED.with(|flag| flag.replace(false))
}

thread_local! {
    /// Whether the resolution in progress found the
    /// implementations matching a projection in a
    /// disagreement it cannot settle (see
    /// `projection_impl_assigned_ty`): set by the decision
    /// when it runs, and read -- and cleared -- by the
    /// one judgment that consults resolution (see
    /// `projection_screen_ty`), the way the depth record
    /// is (see `RESOLUTION_DEPTH_EXHAUSTED`), so a
    /// disagreement recorded anywhere else cannot leak
    /// into a verdict.
    static RESOLUTION_INCONCLUSIVE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Note that the resolution in progress reached a
/// disagreement it cannot settle (see
/// `RESOLUTION_INCONCLUSIVE`).
fn note_resolution_inconclusive() {
    RESOLUTION_INCONCLUSIVE.with(|flag| flag.set(true));
}

/// Whether the resolution since the last take reached a
/// disagreement it cannot settle, clearing the record
/// (see `RESOLUTION_INCONCLUSIVE`).
fn take_resolution_inconclusive() -> bool {
    RESOLUTION_INCONCLUSIVE.with(|flag| flag.replace(false))
}

thread_local! {
    /// Whether the projection walk in progress also judges
    /// a projection by its self type being a parameter (see
    /// `projection_self_is_parameter`): set by
    /// `unnormalizable_projection_in_instance` for the
    /// duration of its one judgment and cleared when it
    /// returns, the way the resolution records are read and
    /// cleared by the one judgment that consults them (see
    /// `RESOLUTION_DEPTH_EXHAUSTED`). The body screen (see
    /// `unnormalizable_projection_in_body`) never sets it:
    /// a definition's own bodies mention projections over
    /// the definition's parameters throughout, and posing
    /// no query about them is not the body screen's call --
    /// the judgment belongs to the instance screen, whose
    /// instances are the ones the compiler's monomorphizing
    /// queries would be posed for.
    static PARAM_SELF_JUDGMENT: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Set whether the projection walk in progress applies the
/// parameter-self judgment (see `PARAM_SELF_JUDGMENT`).
fn set_param_self_judgment(active: bool) {
    PARAM_SELF_JUDGMENT.with(|flag| flag.set(active));
}

/// Whether the projection walk in progress applies the
/// parameter-self judgment (see `PARAM_SELF_JUDGMENT`).
fn param_self_judgment_active() -> bool {
    PARAM_SELF_JUDGMENT.with(|flag| flag.get())
}

/// Whether the projection's self type is a type parameter
/// the ambient arguments do not bind. The first of a
/// projection's generic arguments is its trait reference's
/// self type; resolving it through the ambient arguments,
/// the way the screen's walk resolves parameters, ends at
/// a parameter no argument binds.
///
/// Such a projection cannot normalize in a
/// fully-monomorphized typing environment: there is no
/// implementation to select for an abstract self type and
/// no caller bound to project through, so the compiler's
/// non-trying `normalize_erasing_regions` fails on it the
/// way it fails on a projection no implementation covers
/// -- the query must not be posed. This is the shape a
/// demand through an anonymous constant inside a generic
/// function produces: the constant's instance carries the
/// function's own parameters, and the associated function
/// it calls is demanded with a projection over one of
/// them (`MaybeUninit::<<R as Try>::Output>::uninit` in
/// `core::array::try_from_fn` is the recorded case).
/// rustc's own collector never poses the query either:
/// it collects only concrete instances, and a projection
/// over a parameter is resolved, if ever, at the concrete
/// instantiation that substitutes the parameter away.
fn projection_self_is_parameter(
    alias: &rustc_public::ty::AliasTy,
    ambient: Option<&GenericArgs>,
) -> bool {
    let Some(GenericArgKind::Type(self_ty)) = alias.args.0.first() else {
        return false;
    };
    let mut ty = *self_ty;
    // The step bound is a termination guard for a
    // parameter the ambient arguments bind to itself (an
    // identity instantiation), not evidence: running out
    // of it reaches no judgment.
    for _ in 0..16 {
        match guarded_ty_kind(&ty) {
            Some(TyKind::Param(param)) => {
                match ambient.and_then(|a| a.0.get(param.index as usize)) {
                    Some(GenericArgKind::Type(bound)) => ty = *bound,
                    _ => return true,
                }
            }
            _ => return false,
        }
    }
    false
}

/// Resolve a type as far as the screen's own evidence
/// reaches: parameters through the ambient arguments the
/// way the screen's walk resolves them, and projections
/// through the values the implementation table and
/// trait-object bindings assign them (see
/// `projection_resolved_ty`). Returns the first type that
/// is neither a parameter nor a projection, or `None` when
/// a step has no answer in that evidence. No normalization
/// query is posed — the queries the screen exists to
/// avoid — so a projection whose value the evidence does
/// not determine simply stops the resolution.
fn resolve_screen_self_ty(ty: &Ty, ambient: Option<&GenericArgs>, depth: usize) -> Option<Ty> {
    let mut resolved: Ty = *ty;
    let mut depth = depth;
    loop {
        if depth >= 8 {
            // The budget is a termination guard, not
            // evidence: a resolution that runs out of it
            // has reached no answer, and the absence must
            // not be read as one (see
            // `RESOLUTION_DEPTH_EXHAUSTED`).
            note_resolution_depth_exhausted();
            return None;
        }
        let kind = guarded_ty_kind(&resolved)?;
        match kind {
            TyKind::Param(param) => match ambient.and_then(|a| a.0.get(param.index as usize)) {
                Some(GenericArgKind::Type(t)) => resolved = *t,
                _ => return None,
            },
            TyKind::Alias(SmirAliasKind::Projection, inner) => {
                resolved = projection_resolved_ty(&inner, ambient, depth + 1)?;
            }
            TyKind::Alias(..) => return None,
            _ => return Some(resolved),
        }
        depth += 1;
    }
}

/// The type a projection normalizes to, when the screen's
/// own evidence determines it. Two sources of evidence,
/// both read without posing a normalization query: the
/// projection's self type is itself a trait object binding
/// the item, in which case the binding's term is the
/// value; or an implementation of the projection's trait
/// applies to the resolved self type, in which case the
/// type that implementation assigns the item — read from
/// the implementation's own definition — is the value.
/// Anything else (no applicable implementation, several
/// disagreeing) determines nothing and returns `None`.
pub(crate) fn projection_resolved_ty(
    alias: &rustc_public::ty::AliasTy,
    ambient: Option<&GenericArgs>,
    depth: usize,
) -> Option<Ty> {
    let Some(GenericArgKind::Type(self_ty)) = alias.args.0.first() else {
        return None;
    };
    if let Some(resolved_self) = resolve_screen_self_ty(self_ty, ambient, depth + 1) {
        if let Some(TyKind::RigidTy(RigidTy::Dynamic(preds, _))) = guarded_ty_kind(&resolved_self) {
            for pred in preds.iter() {
                let ExistentialPredicate::Projection(p) = &pred.value else {
                    continue;
                };
                if p.def_id.def_id() == alias.def_id.def_id() {
                    if let TermKind::Type(term) = &p.term {
                        return Some(*term);
                    }
                }
            }
        }
    }
    projection_impl_assigned_ty(alias, ambient, depth)
}

thread_local! {
    /// The running compilation's implementations grouped
    /// by trait, built once (see `impls_of_trait`).
    static IMPLS_BY_TRAIT: std::cell::RefCell<
        Option<HashMap<rustc_public::DefId, Vec<ImplDef>>>,
    > = const { std::cell::RefCell::new(None) };
}

/// The implementations of `trait_def` in the running
/// compilation. Building the table enumerates
/// `all_trait_impls` and reads each impl's trait
/// reference, which neither substitutes nor normalizes,
/// so it poses no query the projection screen exists to
/// avoid (the same table `trait_has_impls` builds).
fn impls_of_trait(trait_def: rustc_public::DefId) -> Vec<ImplDef> {
    IMPLS_BY_TRAIT.with(|cache| {
        let mut cache = cache.borrow_mut();
        let map = cache.get_or_insert_with(|| {
            let mut map: HashMap<rustc_public::DefId, Vec<ImplDef>> = HashMap::new();
            for impl_def in rustc_public::all_trait_impls() {
                map.entry(impl_def.trait_impl().value.def_id.def_id())
                    .or_default()
                    .push(impl_def);
            }
            map
        });
        map.get(&trait_def).cloned().unwrap_or_default()
    })
}

/// One implementation's trait reference, materialized
/// once for the resolution scan (see `impl_scan_entries`).
#[derive(Clone)]
struct ImplScanEntry {
    impl_def: ImplDef,
    self_ty: Ty,
    args: GenericArgs,
}

/// The implementations of a trait organized for the
/// resolution scan in `projection_impl_assigned_ty`.
/// An implementation whose self type is headed by an
/// ADT can only match a self type headed by the same
/// ADT -- the structural match compares rigid heads
/// before anything else -- so those implementations
/// are kept in per-ADT buckets and a scan walks only
/// the bucket for the resolved self type's head, plus
/// the implementations whose self type has no ADT head
/// (a parameter, a reference, an alias, ...), which
/// can match any head. The trait references are
/// materialized once per compilation and kept in
/// `TRAIT_IMPL_SCANS`: reading `ImplDef::trait_impl`
/// crosses the compiler bridge and rebuilds the
/// reference -- the self type included -- on every
/// call, so a scan that re-read it for every
/// implementation of a widely implemented trait, for
/// every projection the screen resolves, would spend
/// its time repeating bridge traffic the first scan
/// already paid for.
struct TraitImplScan {
    generic: Vec<ImplScanEntry>,
    by_head: HashMap<rustc_public::DefId, Vec<ImplScanEntry>>,
}

thread_local! {
    /// The running compilation's per-trait resolution
    /// scans, built lazily (see `impl_scan_entries`),
    /// kept for the compilation like `IMPLS_BY_TRAIT`.
    static TRAIT_IMPL_SCANS: std::cell::RefCell<
        Option<HashMap<rustc_public::DefId, TraitImplScan>>,
    > = const { std::cell::RefCell::new(None) };
}

/// The scan entries that can match a self type with
/// the given ADT head. A `None` head -- a resolved
/// self type that is not an ADT -- can match any
/// implementation, so every entry is returned.
fn impl_scan_entries(
    trait_def: rustc_public::DefId,
    head: Option<rustc_public::DefId>,
) -> Vec<ImplScanEntry> {
    TRAIT_IMPL_SCANS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let scans = cache.get_or_insert_with(HashMap::new);
        let scan = scans.entry(trait_def).or_insert_with(|| {
            let mut scan = TraitImplScan {
                generic: Vec::new(),
                by_head: HashMap::new(),
            };
            for impl_def in impls_of_trait(trait_def) {
                let tref = impl_def.trait_impl().value;
                let entry = ImplScanEntry {
                    impl_def,
                    self_ty: tref.self_ty(),
                    args: tref.args().clone(),
                };
                match guarded_ty_kind(&entry.self_ty) {
                    Some(TyKind::RigidTy(RigidTy::Adt(def, _))) => {
                        scan.by_head.entry(def.def_id()).or_default().push(entry);
                    }
                    _ => scan.generic.push(entry),
                }
            }
            scan
        });
        match head {
            Some(head) => {
                let mut entries = scan.generic.clone();
                if let Some(bucket) = scan.by_head.get(&head) {
                    entries.extend(bucket.iter().cloned());
                }
                entries
            }
            None => {
                let mut entries = scan.generic.clone();
                for bucket in scan.by_head.values() {
                    entries.extend(bucket.iter().cloned());
                }
                entries
            }
        }
    })
}

/// The type the applicable implementation assigns a
/// projection's item, read from the implementation table:
/// the implementations of the projection's trait whose
/// whole trait reference matches the projection's — the
/// self type against the projection's resolved self type,
/// the remaining arguments against the projection's own
/// (see `match_impl_self_ty`) — must agree on one assigned
/// type, read from the matching implementation's own
/// definition through the compiler context. The query
/// posed is the definition query for the item's type,
/// which neither substitutes into nor normalizes the
/// screened definitions — the same class of question
/// `trait_has_impls`' table building already poses.
///
/// Agreement among the matching implementations decides
/// the value, with one refinement for disagreement (see
/// `decide_impl_assigned_ty`): a conditional
/// implementation — one declaring predicates the
/// structural match does not check — matches self types
/// it may not apply to, so its disagreement with an
/// unconditional implementation proves nothing. A
/// disagreement no unconditional implementation settles
/// is recorded as an inconclusive resolution (see
/// `RESOLUTION_INCONCLUSIVE`): it rests on ground the
/// structural match does not check, so it is no proof
/// the projection cannot normalize either.
fn projection_impl_assigned_ty(
    alias: &rustc_public::ty::AliasTy,
    ambient: Option<&GenericArgs>,
    depth: usize,
) -> Option<Ty> {
    let trait_def = alias.def_id.def_id().parent()?;
    let Some(GenericArgKind::Type(self_ty)) = alias.args.0.first() else {
        return None;
    };
    let resolved_self = resolve_screen_self_ty(self_ty, ambient, depth + 1)?;
    let head = match guarded_ty_kind(&resolved_self) {
        Some(TyKind::RigidTy(RigidTy::Adt(def, _))) => Some(def.def_id()),
        _ => None,
    };
    let mut candidates: Vec<(Ty, ImplDef)> = Vec::new();
    for entry in impl_scan_entries(trait_def, head) {
        let impl_def = entry.impl_def;
        // A projection carrying fewer arguments than the
        // trait reference cannot be this implementation's.
        // Arguments beyond the trait reference's are the
        // associated item's own -- a generic associated
        // type's -- and instantiate the item alongside the
        // implementation's arguments when its assigned type
        // is read, below.
        if alias.args.0.len() < entry.args.0.len() {
            continue;
        }
        let mut bindings: HashMap<u32, GenericArgKind> = HashMap::new();
        if !match_impl_self_ty(
            &entry.self_ty,
            &resolved_self,
            ambient,
            &mut bindings,
            depth + 1,
        ) {
            continue;
        }
        // Matching the self type alone does not identify
        // the implementation: a trait can be implemented
        // for the same self type at several of its own
        // argument instantiations (`Carrier<u8> for ()`
        // beside `Carrier<bool> for ()`), each assigning
        // the item a different type. The trait reference's
        // remaining arguments must match the projection's
        // too, binding implementation parameters that
        // appear only in those positions.
        if !entry
            .args
            .0
            .iter()
            .skip(1)
            .zip(alias.args.0.iter().skip(1))
            .all(|(i, t)| match_impl_arg(i, t, ambient, &mut bindings, depth + 1))
        {
            continue;
        }
        // An implementation matches the trait reference,
        // so it is the applicable one; if its assigned
        // type cannot be read, no other implementation's
        // can stand in for it.
        let mut args = impl_instantiation_args(&impl_def, &bindings)?;
        args.0
            .extend_from_slice(&alias.args.0[entry.args.0.len()..]);
        let ty = match impl_item_assigned_ty(&impl_def, alias, &args) {
            Some(ty) => ty,
            None => {
                // The implementation may inherit the item
                // from the trait's own default
                // (`associated_type_defaults`), declaring
                // no item of its own: the assigned type is
                // then the default, instantiated with the
                // trait reference this match established --
                // the resolved self type and the
                // projection's own remaining arguments,
                // composed with the ambient frame (the
                // item's own arguments travel with them,
                // past the trait reference's).
                let mut trait_args = vec![GenericArgKind::Type(resolved_self)];
                trait_args.extend_from_slice(
                    &compose_args(&GenericArgs(alias.args.0[1..].to_vec()), ambient).0,
                );
                trait_item_default_ty(alias, &GenericArgs(trait_args))?
            }
        };
        candidates.push((ty, impl_def));
    }
    let had_candidates = !candidates.is_empty();
    let decided = decide_impl_assigned_ty(candidates);
    if decided.is_none() && had_candidates {
        // Implementations matched and their assigned
        // types were read, but the decision has no
        // verdict for their disagreement: the match is
        // structural, so the disagreement can rest on
        // ground it does not check — a conditional
        // implementation's predicates, a function-pointer
        // signature's bound lifetimes — and proves
        // nothing about whether the projection
        // normalizes. Record the resolution as
        // inconclusive so the screen abstains (see
        // `RESOLUTION_INCONCLUSIVE`).
        note_resolution_inconclusive();
    }
    decided
}

/// The assigned type the matching implementations decide
/// on (see `projection_impl_assigned_ty`): the one type
/// they all assign, when they agree. On disagreement, the
/// unconditional implementations — ones declaring no
/// predicates — decide among themselves: a conditional
/// implementation matches structurally without its
/// predicates being checked, so it may not apply to this
/// self type at all, and in a program that compiles it
/// cannot apply alongside an unconditional one (coherence
/// forbids overlapping implementations), which makes the
/// unconditional agreement the applicable value and the
/// conditional disagreement no evidence against it. When
/// the unconditional implementations disagree among
/// themselves, or none is unconditional, the candidates
/// determine nothing, and the caller records that
/// outcome as an inconclusive resolution when candidates
/// existed (see `projection_impl_assigned_ty`).
fn decide_impl_assigned_ty(candidates: Vec<(Ty, ImplDef)>) -> Option<Ty> {
    if let Some((first, _)) = candidates.first() {
        if candidates.iter().all(|(ty, _)| ty == first) {
            return Some(*first);
        }
    }
    let mut unconditional: Option<Ty> = None;
    for (ty, impl_def) in &candidates {
        if impl_is_conditional(impl_def) {
            continue;
        }
        match unconditional {
            None => unconditional = Some(*ty),
            Some(prev) if prev == *ty => {}
            Some(_) => return None,
        }
    }
    unconditional
}

/// Whether an implementation's applicability is
/// conditional: it declares predicates — parameter bounds
/// or where-clauses — that structural matching does not
/// check, so it may not apply to a self type it matches.
/// The query posed is the definition query for the
/// implementation's explicit clauses (its predicates),
/// which neither substitutes into nor normalizes the
/// screened definitions — the same class of question
/// `impl_item_assigned_ty` poses.
fn impl_is_conditional(impl_def: &ImplDef) -> bool {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal = rustc_public::rustc_internal::internal(tcx, impl_def.def_id());
        Some(!tcx.explicit_clauses_of(internal).clauses.is_empty())
    })
    .unwrap_or(false)
}

/// The generic arguments instantiating an implementation
/// from the arguments its self-type match bound (see
/// `match_impl_self_ty`). A parameter the match never
/// bound has no value the table determines — except a
/// lifetime, which the assigned type's identity does not
/// turn on for the screen's judgment and which is read as
/// erased — so the instantiation fails rather than
/// guessing.
fn impl_instantiation_args(
    impl_def: &ImplDef,
    bindings: &HashMap<u32, GenericArgKind>,
) -> Option<GenericArgs> {
    let generics = impl_def.generics_of();
    let mut args = Vec::with_capacity(generics.params.len());
    for param in &generics.params {
        match bindings.get(&param.index) {
            Some(arg) => args.push(arg.clone()),
            None => match param.kind {
                rustc_public::ty::GenericParamDefKind::Lifetime => {
                    args.push(GenericArgKind::Lifetime(Region {
                        kind: RegionKind::ReErased,
                    }));
                }
                _ => return None,
            },
        }
    }
    Some(GenericArgs(args))
}

/// The type an implementation assigns the projection's
/// item, read from the implementation's own associated
/// item definition through the compiler context installed
/// for the conversion. Outside a compilation callback
/// there is no definition to read.
fn impl_item_assigned_ty(
    impl_def: &ImplDef,
    alias: &rustc_public::ty::AliasTy,
    args: &GenericArgs,
) -> Option<Ty> {
    let item = impl_def.associated_items().into_iter().find(|item| {
        matches!(item.kind, AssocKind::Type { .. })
            && matches!(
                &item.container,
                AssocContainer::TraitImpl(owner)
                    if owner.def_id() == alias.def_id.def_id()
            )
    })?;
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal_item = rustc_public::rustc_internal::internal(tcx, item.def_id.def_id());
        let internal_args = rustc_public::rustc_internal::internal(tcx, args.clone());
        let internal_ty = tcx
            .type_of(internal_item)
            .instantiate(tcx, internal_args)
            .skip_normalization();
        Some(rustc_public::rustc_internal::stable(internal_ty))
    })
}

/// The type a projection's trait item defaults to, when
/// the trait declares a default for it
/// (`associated_type_defaults`) and an implementation
/// inherits the item instead of declaring it: the
/// default, read from the trait item's own definition
/// through the compiler context and instantiated with
/// `trait_args` -- the trait reference the implementation
/// match established, plus the item's own arguments. The
/// query posed is the definition query for the item's
/// type, the same class of question
/// `impl_item_assigned_ty` poses. An item the trait
/// declares without a default has nothing to inherit, so
/// nothing is read for it.
fn trait_item_default_ty(
    alias: &rustc_public::ty::AliasTy,
    trait_args: &GenericArgs,
) -> Option<Ty> {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal_item =
            rustc_public::rustc_internal::internal(tcx, alias.def_id.def_id());
        if !tcx
            .associated_item(internal_item)
            .defaultness(tcx)
            .is_default()
        {
            return None;
        }
        let internal_args =
            rustc_public::rustc_internal::internal(tcx, trait_args.clone());
        let internal_ty = tcx
            .type_of(internal_item)
            .instantiate(tcx, internal_args)
            .skip_normalization();
        Some(rustc_public::rustc_internal::stable(internal_ty))
    })
}

/// Whether a type mentions a generic parameter -- a type
/// or constant parameter, anywhere inside it. Lifetime
/// parameters do not count: the instantiation an
/// implementation match feeds reads an unbound lifetime as
/// erased (see `impl_instantiation_args`), so a lifetime
/// needs no binding the way a type or a constant does.
/// The walk poses no query beyond the kind questions the
/// guarded form answers.
fn ty_contains_param(ty: &Ty) -> bool {
    match guarded_ty_kind(ty) {
        Some(TyKind::Param(_)) => true,
        Some(TyKind::RigidTy(rigid)) => match rigid {
            RigidTy::Adt(_, args)
            | RigidTy::FnDef(_, args)
            | RigidTy::Closure(_, args)
            | RigidTy::Coroutine(_, args)
            | RigidTy::CoroutineClosure(_, args)
            | RigidTy::CoroutineWitness(_, args) => args.0.iter().any(generic_arg_contains_param),
            RigidTy::Ref(_, inner, _)
            | RigidTy::RawPtr(inner, _)
            | RigidTy::Slice(inner)
            | RigidTy::Pat(inner, _) => ty_contains_param(&inner),
            RigidTy::Array(elem, len) => ty_contains_param(&elem) || const_contains_param(&len),
            RigidTy::Tuple(ts) => ts.iter().any(ty_contains_param),
            RigidTy::FnPtr(sig) => sig
                .skip_binder()
                .inputs_and_output
                .iter()
                .any(ty_contains_param),
            RigidTy::Dynamic(preds, _) => preds.iter().any(|pred| match &pred.value {
                ExistentialPredicate::Trait(tref) => {
                    tref.generic_args.0.iter().any(generic_arg_contains_param)
                }
                ExistentialPredicate::Projection(p) => {
                    p.generic_args.0.iter().any(generic_arg_contains_param)
                        || match &p.term {
                            TermKind::Type(t) => ty_contains_param(t),
                            TermKind::Const(c) => const_contains_param(c),
                        }
                }
                ExistentialPredicate::AutoTrait(_) => false,
            }),
            _ => false,
        },
        Some(TyKind::Alias(_, alias)) => alias.args.0.iter().any(generic_arg_contains_param),
        _ => false,
    }
}

/// Whether a generic argument mentions a generic
/// parameter (see `ty_contains_param`).
fn generic_arg_contains_param(arg: &GenericArgKind) -> bool {
    match arg {
        GenericArgKind::Type(t) => ty_contains_param(t),
        GenericArgKind::Const(c) => const_contains_param(c),
        GenericArgKind::Lifetime(_) => false,
    }
}

/// Whether a constant mentions a generic parameter (see
/// `ty_contains_param`).
fn const_contains_param(c: &TyConst) -> bool {
    match c.kind() {
        TyConstKind::Param(_) => true,
        TyConstKind::Unevaluated(_, args) => args.0.iter().any(generic_arg_contains_param),
        _ => false,
    }
}

/// Whether a type is or contains a projection (an alias
/// of any kind) — at the top, or nested inside a compound
/// shape the way `ty_contains_param` finds a parameter
/// nested in one. A binding disagreement over such a type
/// rests on ground the structural match does not check,
/// the same way a disagreement over the projection itself
/// does (see `bind_impl_arg`).
fn ty_contains_alias(ty: &Ty) -> bool {
    match guarded_ty_kind(ty) {
        Some(TyKind::Alias(..)) => true,
        Some(TyKind::RigidTy(rigid)) => match rigid {
            RigidTy::Adt(_, args)
            | RigidTy::FnDef(_, args)
            | RigidTy::Closure(_, args)
            | RigidTy::Coroutine(_, args)
            | RigidTy::CoroutineClosure(_, args)
            | RigidTy::CoroutineWitness(_, args) => {
                args.0.iter().any(generic_arg_contains_alias)
            }
            RigidTy::Ref(_, inner, _)
            | RigidTy::RawPtr(inner, _)
            | RigidTy::Slice(inner)
            | RigidTy::Pat(inner, _) => ty_contains_alias(&inner),
            RigidTy::Array(elem, len) => ty_contains_alias(&elem) || const_contains_alias(&len),
            RigidTy::Tuple(ts) => ts.iter().any(ty_contains_alias),
            RigidTy::FnPtr(sig) => sig
                .skip_binder()
                .inputs_and_output
                .iter()
                .any(ty_contains_alias),
            RigidTy::Dynamic(preds, _) => preds.iter().any(|pred| match &pred.value {
                ExistentialPredicate::Trait(tref) => {
                    tref.generic_args.0.iter().any(generic_arg_contains_alias)
                }
                ExistentialPredicate::Projection(p) => {
                    p.generic_args.0.iter().any(generic_arg_contains_alias)
                        || match &p.term {
                            TermKind::Type(t) => ty_contains_alias(t),
                            TermKind::Const(c) => const_contains_alias(c),
                        }
                }
                ExistentialPredicate::AutoTrait(_) => false,
            }),
            _ => false,
        },
        _ => false,
    }
}

/// Whether a generic argument is or contains a projection
/// (see `ty_contains_alias`).
fn generic_arg_contains_alias(arg: &GenericArgKind) -> bool {
    match arg {
        GenericArgKind::Type(t) => ty_contains_alias(t),
        GenericArgKind::Const(c) => const_contains_alias(c),
        GenericArgKind::Lifetime(_) => false,
    }
}

/// Whether a constant's unevaluated arguments mention a
/// type that is or contains a projection (see
/// `ty_contains_alias`).
fn const_contains_alias(c: &TyConst) -> bool {
    match c.kind() {
        TyConstKind::Unevaluated(_, args) => args.0.iter().any(generic_arg_contains_alias),
        _ => false,
    }
}

/// Whether an implementation with the trait-reference
/// self type `impl_self` applies to the resolved self
/// type `target`, recording the implementation's generic
/// arguments in `bindings`. The match is structural: an
/// implementation parameter binds the corresponding
/// target argument (and must agree with any earlier
/// binding of the same parameter), every other shape
/// must agree exactly, and a target shape the match
/// cannot judge — an unresolved parameter, a projection
/// the screen cannot resolve — does not match. The match
/// poses no query beyond the kind questions the guarded
/// form answers.
fn match_impl_self_ty(
    impl_self: &Ty,
    target: &Ty,
    ambient: Option<&GenericArgs>,
    bindings: &mut HashMap<u32, GenericArgKind>,
    depth: usize,
) -> bool {
    let Some(ikind) = guarded_ty_kind(impl_self) else {
        return false;
    };
    match ikind {
        TyKind::Param(param) => {
            // The target is written in the frame the
            // screen is walking, so the parameters inside
            // it are first resolved through the ambient
            // frame -- the type form of the resolution the
            // constant match performs for its target (see
            // `match_impl_const`) -- recursively (see
            // `substitute_ty`), so a parameter nested
            // inside a compound target is substituted the
            // way a bare parameter target is: walking
            // `Holder<u32>`, the self type `Wrap<(T,)>`
            // binds the implementation's `A` to `(u32,)`,
            // not to the unsubstituted `(T,)` the trait
            // argument is then compared against. A
            // parameter the ambient frame leaves
            // unresolved stays a parameter. Without it, an
            // implementation parameter appearing in both
            // the self type and another trait argument
            // binds the target's unsubstituted form
            // through the self type and is then compared
            // against the substituted one, rejecting an
            // applicable implementation.
            let substituted;
            let target = match ambient {
                Some(ambient) => {
                    substituted = substitute_ty(target, ambient, 0, false);
                    &substituted
                }
                None => target,
            };
            // A projection target is resolved before it
            // is bound, the way a rigid self type's nested
            // target is (see the `RigidTy` arm below): the
            // substitution above resolves parameters but
            // leaves projections standing, so an
            // implementation parameter appearing in both
            // the self type and another trait argument
            // (`Carrier<A> for Wrap<A>` matched against
            // `Carrier<dyn Trait<Assoc = u32>>` for
            // `Wrap<T::Item>`) would bind the projection
            // itself through the self type and then be
            // rejected against the object type the
            // projection denotes at the trait argument.
            // A projection the evidence cannot resolve
            // keeps its substituted form; a disagreement
            // over that form is inconclusive rather than a
            // mismatch (see `bind_impl_arg`).
            let resolved;
            let target = match guarded_ty_kind(target) {
                Some(TyKind::Alias(..)) => {
                    match resolve_screen_self_ty(target, ambient, depth + 1) {
                        Some(t) => {
                            resolved = t;
                            &resolved
                        }
                        None => target,
                    }
                }
                _ => target,
            };
            bind_impl_arg(bindings, param.index, GenericArgKind::Type(*target), ambient, depth)
        }
        TyKind::RigidTy(irigid) => {
            // An interned-identity agreement settles the
            // match only when the implementation's self
            // type mentions no parameter. The two sides'
            // parameters belong to different items -- the
            // implementation's own and the walked frame's --
            // and identically indexed ones compare equal
            // without binding anything, leaving the
            // instantiation the assigned-type read performs
            // (see `impl_instantiation_args`) without the
            // argument it needs. A parameter-mentioning
            // self type is matched structurally instead, so
            // its parameters bind through the ambient frame
            // the way any other's do.
            let identity_decides = !ty_contains_param(impl_self);
            if identity_decides && impl_self == target {
                return true;
            }
            // A nested target position can still hold a
            // parameter or a projection, which the same
            // resolution settles before comparing.
            let resolved_target = match guarded_ty_kind(target) {
                Some(TyKind::Param(_)) | Some(TyKind::Alias(..)) => {
                    match resolve_screen_self_ty(target, ambient, depth + 1) {
                        Some(t) => t,
                        None => return false,
                    }
                }
                _ => *target,
            };
            if identity_decides && *impl_self == resolved_target {
                return true;
            }
            let Some(TyKind::RigidTy(trigid)) = guarded_ty_kind(&resolved_target) else {
                return false;
            };
            match (irigid, trigid) {
                (RigidTy::Adt(idef, iargs), RigidTy::Adt(tdef, targs)) => {
                    idef.def_id() == tdef.def_id()
                        && match_impl_args(&iargs, &targs, ambient, bindings, depth)
                }
                (RigidTy::Ref(_, ii, im), RigidTy::Ref(_, ti, tm)) => {
                    im == tm && match_impl_self_ty(&ii, &ti, ambient, bindings, depth)
                }
                (RigidTy::RawPtr(ii, im), RigidTy::RawPtr(ti, tm)) => {
                    im == tm && match_impl_self_ty(&ii, &ti, ambient, bindings, depth)
                }
                (RigidTy::Slice(ii), RigidTy::Slice(ti)) => {
                    match_impl_self_ty(&ii, &ti, ambient, bindings, depth)
                }
                (RigidTy::Pat(ii, _), RigidTy::Pat(ti, _)) => {
                    match_impl_self_ty(&ii, &ti, ambient, bindings, depth)
                }
                (RigidTy::Array(ie, ilen), RigidTy::Array(te, tlen)) => {
                    match_impl_self_ty(&ie, &te, ambient, bindings, depth)
                        && match_impl_const(&ilen, &tlen, ambient, bindings, depth)
                }
                (RigidTy::Tuple(its), RigidTy::Tuple(tts)) => {
                    its.len() == tts.len()
                        && its
                            .iter()
                            .zip(tts.iter())
                            .all(|(i, t)| match_impl_self_ty(i, t, ambient, bindings, depth))
                }
                (RigidTy::FnPtr(isig), RigidTy::FnPtr(tsig)) => {
                    // A generic implementation for a
                    // function pointer (`impl<A> Trait
                    // for fn(A)`) matches a concrete
                    // pointer type signature by
                    // signature: the calling convention
                    // must agree exactly, and each input
                    // and the output match the way self
                    // types match, binding the
                    // implementation's parameters. The
                    // signatures' own bound variables are
                    // their late-bound lifetimes, which
                    // the type match does not turn on: a
                    // signature distinguished from the
                    // target's only by its bound lifetimes
                    // (`fn(&'static u8)` beside
                    // `for<'a> fn(&'a u8)`) still matches,
                    // so its implementation can stand as a
                    // candidate beside the applicable one.
                    // That is safe because a disagreement
                    // among the candidates is recorded as
                    // an inconclusive resolution (see
                    // `RESOLUTION_INCONCLUSIVE`), never as
                    // proof the projection cannot
                    // normalize.
                    let iv = isig.skip_binder();
                    let tv = tsig.skip_binder();
                    iv.c_variadic == tv.c_variadic
                        && iv.safety == tv.safety
                        && iv.abi == tv.abi
                        && iv.inputs_and_output.len() == tv.inputs_and_output.len()
                        && iv
                            .inputs_and_output
                            .iter()
                            .zip(tv.inputs_and_output.iter())
                            .all(|(i, t)| match_impl_self_ty(i, t, ambient, bindings, depth))
                }
                (RigidTy::Dynamic(ipreds, _), RigidTy::Dynamic(tpreds, _)) => {
                    // A generic implementation for a
                    // trait object (`impl<A> Trait for
                    // dyn Other<A>`) matches a concrete
                    // object type predicate by predicate
                    // (see `match_impl_dynamic_preds`):
                    // the object's own bound variables are
                    // what both lists' arguments are
                    // written against, and the region
                    // decides nothing the type match
                    // turns on, the way regions inside
                    // types are ignored throughout.
                    match_impl_dynamic_preds(&ipreds, &tpreds, ambient, bindings, depth)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// Whether an implementation's trait-object self type
/// matches the target object type (see
/// `match_impl_self_ty`): every existential predicate the
/// implementation's object spells must have a counterpart
/// of the same trait in the target's predicate list, with
/// arguments matching the way trait arguments match and a
/// projection predicate's term matching the way its kind
/// matches, binding the implementation's parameters. The
/// target may spell further predicates the implementation
/// does not (an object type can carry bounds beyond the
/// ones an implementation's self type names); only the
/// implementation's own predicates constrain the match.
/// The auto traits are the exception: they are part of
/// the object type's identity -- `dyn Other` and
/// `dyn Other + Send` are different self types with
/// different implementations -- so both objects must
/// carry the same auto traits, or the `dyn Other`
/// implementation would also match a `dyn Other + Send`
/// target and leave two disagreeing candidates where
/// the native resolution has exactly one.
/// A failed counterpart attempt leaves no bindings
/// behind, so a later counterpart is tried against the
/// bindings as they stood.
fn match_impl_dynamic_preds(
    ipreds: &[rustc_public::ty::Binder<rustc_public::ty::ExistentialPredicate>],
    tpreds: &[rustc_public::ty::Binder<rustc_public::ty::ExistentialPredicate>],
    ambient: Option<&GenericArgs>,
    bindings: &mut HashMap<u32, GenericArgKind>,
    depth: usize,
) -> bool {
    // The auto traits must agree as a multiset (see the
    // doc comment above): collect each side's auto-trait
    // definitions and compare their multiplicities before
    // the per-predicate subset match below.
    let auto_traits =
        |preds: &[rustc_public::ty::Binder<rustc_public::ty::ExistentialPredicate>]| {
            preds
                .iter()
                .filter_map(|pred| match &pred.value {
                    ExistentialPredicate::AutoTrait(def) => Some(def.def_id()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
    let iautos = auto_traits(ipreds);
    let tautos = auto_traits(tpreds);
    let same_auto_traits = iautos.len() == tautos.len()
        && iautos.iter().all(|id| {
            iautos.iter().filter(|other| *other == id).count()
                == tautos.iter().filter(|other| *other == id).count()
        });
    if !same_auto_traits {
        return false;
    }
    ipreds.iter().all(|ipred| {
        tpreds.iter().any(|tpred| {
            let saved = bindings.clone();
            let matched = match (&ipred.value, &tpred.value) {
                (ExistentialPredicate::Trait(itref), ExistentialPredicate::Trait(ttref)) => {
                    itref.def_id.def_id() == ttref.def_id.def_id()
                        && match_impl_args(
                            &itref.generic_args,
                            &ttref.generic_args,
                            ambient,
                            bindings,
                            depth,
                        )
                }
                (ExistentialPredicate::Projection(ip), ExistentialPredicate::Projection(tp)) => {
                    ip.def_id.def_id() == tp.def_id.def_id()
                        && match_impl_args(
                            &ip.generic_args,
                            &tp.generic_args,
                            ambient,
                            bindings,
                            depth,
                        )
                        && match (&ip.term, &tp.term) {
                            (TermKind::Type(it), TermKind::Type(tt)) => {
                                match_impl_self_ty(it, tt, ambient, bindings, depth)
                            }
                            (TermKind::Const(ic), TermKind::Const(tc)) => {
                                match_impl_const(ic, tc, ambient, bindings, depth)
                            }
                            _ => false,
                        }
                }
                (ExistentialPredicate::AutoTrait(idef), ExistentialPredicate::AutoTrait(tdef)) => {
                    idef.def_id() == tdef.def_id()
                }
                _ => false,
            };
            if !matched {
                *bindings = saved;
            }
            matched
        })
    })
}

/// Whether the generic arguments of an implementation's
/// trait reference (or of an ADT the self type mentions)
/// match the target's, binding the implementation's
/// parameters (see `match_impl_self_ty`).
fn match_impl_args(
    iargs: &GenericArgs,
    targs: &GenericArgs,
    ambient: Option<&GenericArgs>,
    bindings: &mut HashMap<u32, GenericArgKind>,
    depth: usize,
) -> bool {
    iargs.0.len() == targs.0.len()
        && iargs
            .0
            .iter()
            .zip(targs.0.iter())
            .all(|(i, t)| match_impl_arg(i, t, ambient, bindings, depth))
}

/// Whether one generic argument of an implementation
/// matches the corresponding target argument (see
/// `match_impl_args`).
fn match_impl_arg(
    iarg: &GenericArgKind,
    targ: &GenericArgKind,
    ambient: Option<&GenericArgs>,
    bindings: &mut HashMap<u32, GenericArgKind>,
    depth: usize,
) -> bool {
    match (iarg, targ) {
        (GenericArgKind::Type(it), GenericArgKind::Type(tt)) => {
            match_impl_self_ty(it, tt, ambient, bindings, depth)
        }
        (GenericArgKind::Const(ic), GenericArgKind::Const(tc)) => {
            match_impl_const(ic, tc, ambient, bindings, depth)
        }
        (GenericArgKind::Lifetime(iregion), GenericArgKind::Lifetime(tregion)) => {
            // The target region is written in the frame
            // the screen is walking, so a lifetime
            // parameter there is first resolved through
            // the ambient frame -- the lifetime form of
            // the resolution the constant match performs
            // for its target (see `match_impl_const`) --
            // and a parameter the ambient frame leaves
            // unresolved stays a parameter.
            let resolved;
            let tregion = match &tregion.kind {
                RegionKind::ReEarlyParam(param) => {
                    match ambient.and_then(|a| a.0.get(param.index as usize)) {
                        Some(GenericArgKind::Lifetime(bound)) => {
                            resolved = bound.clone();
                            &resolved
                        }
                        _ => tregion,
                    }
                }
                _ => tregion,
            };
            match &iregion.kind {
                RegionKind::ReEarlyParam(param) => bind_impl_arg(
                    bindings,
                    param.index,
                    GenericArgKind::Lifetime(tregion.clone()),
                    ambient,
                    depth,
                ),
                // A lifetime never decides which type an
                // implementation assigns the item -- the
                // instantiation reads lifetimes as erased
                // (see `impl_instantiation_args`), and
                // the type match ignores the regions
                // inside types the same way -- so a
                // comparison the regions leave
                // inconclusive (an erased region on
                // either side, a target parameter the
                // ambient frame does not resolve) is not
                // a mismatch: the implementation stays a
                // candidate and the assigned-type
                // agreement check decides. Two concrete
                // regions must still agree.
                _ => {
                    iregion == tregion
                        || matches!(iregion.kind, RegionKind::ReErased)
                        || matches!(
                            tregion.kind,
                            RegionKind::ReErased
                                | RegionKind::ReEarlyParam(_)
                                | RegionKind::ReBound(..)
                        )
                }
            }
        }
        _ => false,
    }
}

/// Whether an implementation's constant argument matches
/// the target's: an implementation constant parameter
/// binds the target constant, every other constant must
/// be the same constant. The target constant is written
/// in the frame the screen is walking, so a constant
/// parameter there is first resolved through the ambient
/// frame -- the constant form of the resolution the type
/// match performs for its target (see
/// `match_impl_self_ty`) -- and a parameter the ambient
/// frame leaves unresolved stays a parameter. An
/// unevaluated constant on each side matches another for
/// the same definition whose arguments match in turn, so
/// a parameter nested inside one is resolved (or bound)
/// the same way. Any other comparison an unevaluated
/// constant takes part in is inconclusive rather than a
/// mismatch: telling `{ N + 0 }` from the concrete `3`
/// takes evaluating it, which the screen never does, so
/// the implementation stays a candidate and the
/// assigned-type agreement check decides -- the lifetime
/// match in `match_impl_arg` treats its inconclusive
/// comparisons the same way.
fn match_impl_const(
    iconst: &TyConst,
    tconst: &TyConst,
    ambient: Option<&GenericArgs>,
    bindings: &mut HashMap<u32, GenericArgKind>,
    depth: usize,
) -> bool {
    let resolved;
    let tconst = match tconst.kind() {
        TyConstKind::Param(param) => {
            match ambient.and_then(|a| a.0.get(param.index as usize)) {
                Some(GenericArgKind::Const(bound)) => {
                    resolved = bound.clone();
                    &resolved
                }
                _ => tconst,
            }
        }
        _ => tconst,
    };
    match iconst.kind() {
        TyConstKind::Param(param) => {
            bind_impl_arg(
                bindings,
                param.index,
                GenericArgKind::Const(tconst.clone()),
                ambient,
                depth,
            )
        }
        TyConstKind::Unevaluated(idef, iargs) => match tconst.kind() {
            TyConstKind::Unevaluated(tdef, targs) if idef.def_id() == tdef.def_id() => {
                match_impl_args(&iargs, &targs, ambient, bindings, depth)
            }
            _ => true,
        },
        kind => match tconst.kind() {
            TyConstKind::Unevaluated(..) => true,
            _ => kind == tconst.kind(),
        },
    }
}

/// Record an implementation parameter's matched argument,
/// requiring agreement with any earlier binding of the
/// same parameter.
fn bind_impl_arg(
    bindings: &mut HashMap<u32, GenericArgKind>,
    index: u32,
    arg: GenericArgKind,
    ambient: Option<&GenericArgs>,
    depth: usize,
) -> bool {
    match bindings.get(&index) {
        Some(prev) => {
            if *prev == arg {
                return true;
            }
            // Two concrete types can spell the same type
            // with different region spellings: a type read
            // from a definition keeps the regions its
            // declaration carries (`dyn Trait<Assoc =
            // u32>` read from an implementation's item is
            // `+ 'static`), while the same type reached
            // through a monomorphized body is region-erased
            // (the `+ 'static` is erased). Interned
            // identity tells those spellings apart, but
            // the type match the binding feeds ignores the
            // regions inside types throughout (see
            // `match_impl_arg`), so concrete rigid types
            // -- ones mentioning no parameter, whose
            // agreement a parameter could still turn on --
            // are also compared structurally, in a scratch
            // binding map the comparison cannot disturb.
            if let (GenericArgKind::Type(prev_ty), GenericArgKind::Type(arg_ty)) =
                (prev, &arg)
            {
                if !ty_contains_param(prev_ty)
                    && !ty_contains_param(arg_ty)
                    && matches!(guarded_ty_kind(prev_ty), Some(TyKind::RigidTy(_)))
                    && matches!(guarded_ty_kind(arg_ty), Some(TyKind::RigidTy(_)))
                {
                    let mut scratch: HashMap<u32, GenericArgKind> = HashMap::new();
                    if match_impl_self_ty(prev_ty, arg_ty, ambient, &mut scratch, depth + 1)
                    {
                        return true;
                    }
                }
            }
            // A disagreement one side of which is or
            // contains a projection the resolution could
            // not settle (see the parameter arm of
            // `match_impl_self_ty`, which binds the
            // resolved form when it can) rests on ground
            // the match does not check: the projection may
            // denote the very type it is compared against,
            // so the disagreement proves nothing about
            // whether the outer projection normalizes. The
            // containment is recursive (see
            // `ty_contains_alias`): a projection nested
            // inside a compound binding -- a tuple holding
            // `T::Item`, bound as one argument -- is the
            // same unchecked ground as the projection
            // itself, and an outermost-shape check alone
            // misses it.
            // Record the resolution as inconclusive the
            // way an unsettled disagreement among the
            // matching implementations is recorded (see
            // `RESOLUTION_INCONCLUSIVE`); the
            // implementation itself does not stand as a
            // candidate on unchecked ground.
            let unresolved_projection =
                |a: &GenericArgKind| generic_arg_contains_alias(a);
            if unresolved_projection(prev) || unresolved_projection(&arg) {
                note_resolution_inconclusive();
            }
            false
        }
        None => {
            bindings.insert(index, arg);
            true
        }
    }
}
/// Walk a type for a projection that cannot normalize (see
/// [`projection_cannot_normalize`]), returning the recorded
/// reason for the first one found. The walk mirrors
/// `ty_sizedness`'s discipline: declared field types through
/// ADT boundaries, parameters resolved one frame deep
/// through the ambient arguments (see `compose_args`), a
/// depth cap, and no compiler query beyond the kind
/// questions the guarded form answers — the walk itself
/// never poses the normalization it screens for. A type the
/// walk cannot judge (an unaskable kind, an unresolved
/// parameter) counts as no finding.
/// The types one screen question has already walked clean,
/// keyed by the shallowest depth at which a walk of the type
/// completed (see `projection_screen_ty`).
type ScreenVisited = HashMap<Ty, usize>;

/// Walk a type for a projection that cannot normalize (see
/// `unnormalizable_projection_in_instance`), memoizing clean
/// walks within the question.
///
/// A walk entered without an ambient substitution cannot
/// consult one anywhere in its subtree — only a declared
/// type's parameters do, and declared types are entered with
/// the ambient they resolve through, never without — so its
/// outcome is a function of the type and the remaining depth
/// budget alone, and the budget only ever shrinks the search:
/// a type walked clean at a shallower depth (a larger budget)
/// is clean at any deeper one. The compiler interns types as
/// a DAG, but the walk is a tree walk, so a shared subterm —
/// the suffix of a growing adapter tower, reached once per
/// field and argument path that mentions it — would
/// otherwise be re-walked once per path, exponentially often
/// in the tower's height: screening one demanded instance of
/// a tower-shaped program cost more than converting it. The
/// record is created per question by the callers, so no
/// verdict leaks across questions.
fn projection_screen_ty(
    ty: &Ty,
    ambient: Option<&GenericArgs>,
    depth: usize,
    visited: &mut ScreenVisited,
) -> Option<String> {
    if ambient.is_none() {
        if let Some(&walked_depth) = visited.get(ty) {
            if walked_depth <= depth {
                return None;
            }
        }
    }
    let result = projection_screen_ty_walk(ty, ambient, depth, visited);
    if result.is_none() && ambient.is_none() {
        visited
            .entry(*ty)
            .and_modify(|d| *d = (*d).min(depth))
            .or_insert(depth);
    }
    result
}

fn projection_screen_ty_walk(
    ty: &Ty,
    ambient: Option<&GenericArgs>,
    depth: usize,
    visited: &mut ScreenVisited,
) -> Option<String> {
    if depth >= 8 {
        return None;
    }
    let kind = guarded_ty_kind(ty)?;
    match kind {
        TyKind::Alias(alias_kind, alias) => {
            if alias_kind == SmirAliasKind::Projection && projection_cannot_normalize(&alias) {
                // The judgment rests on resolving the
                // projection's self type (see
                // `dyn_binding_normalizes`); a resolution
                // that exhausts its depth budget on the
                // way, or reaches a disagreement among
                // the matching implementations it cannot
                // settle, has reached no answer, so either
                // outcome during this judgment leaves the
                // screen without an opinion rather than
                // rejecting a projection it could not
                // finish resolving. The records are
                // cleared first, so an outcome from an
                // earlier judgment cannot leak into this
                // one.
                let _ = take_resolution_depth_exhausted();
                let _ = take_resolution_inconclusive();
                if !dyn_binding_normalizes(&alias, ambient, depth)
                    && !take_resolution_depth_exhausted()
                    && !take_resolution_inconclusive()
                {
                    return Some(format!(
                        "{UNNORMALIZABLE_PROJECTION_REASON}: \
                         {alias:?}'s trait has no implementation"
                    ));
                }
            }
            if alias_kind == SmirAliasKind::Projection
                && param_self_judgment_active()
                && projection_self_is_parameter(&alias, ambient)
            {
                return Some(format!(
                    "{UNNORMALIZABLE_PROJECTION_REASON}: \
                     {alias:?}'s self type is a parameter"
                ));
            }
            projection_screen_args(&alias.args, ambient, depth + 1, visited)
        }
        TyKind::RigidTy(r) => match r {
            RigidTy::Adt(def, a2) => {
                let composed = compose_screen_args(&a2, ambient);
                projection_screen_args(&a2, ambient, depth + 1, visited).or_else(|| {
                    def.variants().iter().find_map(|variant| {
                        variant.fields().iter().find_map(|field| {
                            projection_screen_ty(
                                &field.ty(),
                                Some(&composed),
                                depth + 1,
                                visited,
                            )
                        })
                    })
                })
            }
            RigidTy::Ref(_, inner, _)
            | RigidTy::RawPtr(inner, _)
            | RigidTy::Slice(inner)
            | RigidTy::Pat(inner, _) => {
                projection_screen_ty(&inner, ambient, depth + 1, visited)
            }
            RigidTy::Array(elem, len) => {
                projection_screen_ty(&elem, ambient, depth + 1, visited).or_else(|| {
                    projection_screen_const(&len, ambient, depth + 1, visited)
                })
            }
            RigidTy::Tuple(ts) => ts
                .iter()
                .find_map(|t| projection_screen_ty(t, ambient, depth + 1, visited)),
            RigidTy::FnDef(_, a2) => {
                projection_screen_args(&a2, ambient, depth + 1, visited).or_else(|| {
                    ty.kind().fn_sig().and_then(|sig| {
                        sig.skip_binder()
                            .inputs_and_output
                            .iter()
                            .find_map(|t| {
                                projection_screen_ty(t, None, depth + 1, visited)
                            })
                    })
                })
            }
            RigidTy::FnPtr(sig) => sig
                .skip_binder()
                .inputs_and_output
                .iter()
                .find_map(|t| projection_screen_ty(t, ambient, depth + 1, visited)),
            RigidTy::Closure(_, a2)
            | RigidTy::Coroutine(_, a2)
            | RigidTy::CoroutineWitness(_, a2) => {
                projection_screen_args(&a2, ambient, depth + 1, visited)
            }
            RigidTy::Dynamic(preds, _) => preds.iter().find_map(|pred| {
                match &pred.value {
                    ExistentialPredicate::Trait(tref) => {
                        projection_screen_args(
                            &tref.generic_args,
                            ambient,
                            depth + 1,
                            visited,
                        )
                    }
                    ExistentialPredicate::Projection(pred) => {
                        projection_screen_args(
                            &pred.generic_args,
                            ambient,
                            depth + 1,
                            visited,
                        )
                    }
                    ExistentialPredicate::AutoTrait(_) => None,
                }
            }),
            _ => None,
        },
        TyKind::Param(p) => {
            match ambient.and_then(|a| a.0.get(p.index as usize)) {
                Some(GenericArgKind::Type(t)) => {
                    projection_screen_ty(t, None, depth + 1, visited)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Walk generic arguments for a projection that cannot
/// normalize (see `projection_screen_ty`).
fn projection_screen_args(
    args: &GenericArgs,
    ambient: Option<&GenericArgs>,
    depth: usize,
    visited: &mut ScreenVisited,
) -> Option<String> {
    if depth >= 8 {
        return None;
    }
    args.0.iter().find_map(|arg| match arg {
        GenericArgKind::Type(t) => projection_screen_ty(t, ambient, depth, visited),
        GenericArgKind::Const(c) => projection_screen_const(c, ambient, depth, visited),
        GenericArgKind::Lifetime(_) => None,
    })
}

/// Walk a constant for a projection that cannot normalize
/// (see `projection_screen_ty`): only an unevaluated
/// constant's arguments can mention one.
fn projection_screen_const(
    c: &rustc_public::ty::TyConst,
    ambient: Option<&GenericArgs>,
    depth: usize,
    visited: &mut ScreenVisited,
) -> Option<String> {
    if depth >= 8 {
        return None;
    }
    match c.kind() {
        TyConstKind::Unevaluated(_, args) => {
            projection_screen_args(&args, ambient, depth, visited)
        }
        _ => None,
    }
}

/// Whether a panic hook record names one of rustc_public
/// 1.100.0's known `unimplemented!()` conversion sites — the
/// same sites the driver binary's location classification
/// maps to named skip reasons (the tail-call terminator, the
/// unsafe-binder type, and the unconverted constant kinds).
/// A panic there is a known conversion gap, not a crash.
fn record_names_unimplemented_site(record: &str) -> bool {
    record.contains("rustc_public/src/unstable/convert/stable/mir.rs:775:")
        || record.contains("rustc_public/src/unstable/convert/stable/ty.rs:471:")
        || record.contains("rustc_public/src/unstable/convert/stable/ty.rs:577:")
}

/// Run a screen question under the containment
/// `guarded_ty_kind` applies to kind questions. The screen
/// asks questions some types cannot answer — an async
/// closure instance's type cannot be asked at all, and an
/// instance constant's evaluation can be too generic — and
/// those questions panic with one of rustc_public's known
/// unimplemented conversions. The screen also asks for an
/// instance's type, which panics with "needs further
/// instantiation" while the instance is still generic
/// (the same gap `guarded_abi_query` tolerates: the
/// question is speculative, and a body the question would
/// have vetoed is fetched afterwards under
/// `fetch_body_contained`'s own containment), and a screen
/// walk over a body's types can reach a type whose kind
/// conversion is one of rustc_public's known
/// `unimplemented!()` sites — vetoing the body is not the
/// screen's call there either, because the body's own
/// conversion classifies the site into its named skip
/// reason. Such a panic means the question has no answer
/// the screen can use, so the screen has no opinion; the
/// panic hook's record of it is consumed, so no outer
/// containment mistakes it for its own. Any other panic is
/// resumed: the screen must never paper over an unrelated
/// compiler failure.
pub(crate) fn screen_question<T>(question: impl FnOnce() -> Option<T>) -> Option<T> {
    clear_captured_panic();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(question)) {
        Ok(answer) => answer,
        Err(payload) => {
            if stable_mir_panic_reason(&*payload).is_some() {
                let _ = take_captured_panic();
                return None;
            }
            // The payload of a panic that crossed the bridge
            // is opaque, so the hook's record of the same
            // panic — taken after the slot was cleared above,
            // hence this very panic's — is the witness that
            // identifies it.
            match take_captured_panic() {
                Some(record)
                    if record.contains("FIXME(async_closures)")
                        || record.contains("needs further instantiation")
                        || record_names_unimplemented_site(&record) =>
                {
                    None
                }
                other => {
                    if let Some(record) = other {
                        record_captured_panic(record);
                    }
                    std::panic::resume_unwind(payload)
                }
            }
        }
    }
}

/// Whether posing the compiler's monomorphizing queries
/// for this instance — materializing its body through
/// `Instance::body`, or resolving one of its associated
/// items during conversion — would ask the compiler to
/// normalize a trait projection that cannot normalize in a
/// fully-monomorphized typing environment, returning the
/// recorded reason if so. Two judgments are made: a
/// projection whose trait has no implementation at all
/// (see `projection_cannot_normalize`), and a projection
/// whose self type is a parameter no argument binds (see
/// `projection_self_is_parameter`) — the second is this
/// screen's alone, never the body screen's.
///
/// Both queries normalize under
/// `TypingEnv::fully_monomorphized` (reveal-all, no caller
/// bounds) through the non-trying
/// `normalize_erasing_regions`, whose failure is a delayed
/// bug: the diagnostic is stored and flushed at the end of
/// the run, so the compilation is poisoned even where the
/// panic itself is contained. The query must not be posed.
/// rustc's own monomorphization collector never poses it
/// either: it instantiates only definitions demanded by
/// other instantiated code, and a projection no
/// implementation covers can never be demanded by a
/// compilable program — the definitions that mention one
/// (a function taking the projection as an argument, a
/// constructor whose field it is, a generic constant whose
/// body resolves a call through it) are exactly the
/// definitions that stay uninstantiated. The screen is
/// structural: it walks the instance's type (for a
/// function, its signature with the arguments applied), its
/// generic arguments, and the declared field types of the
/// ADTs those mention, and reports the first projection
/// whose trait has no implementation at all (see
/// `projection_cannot_normalize`).
pub fn unnormalizable_projection_in_instance(
    inst: &Instance,
) -> Option<String> {
    // The two walks are separate screen questions: asking
    // a still-generic instance for its type can panic
    // ("needs further instantiation", see
    // `screen_question`), and that containment must not
    // void the judgment the instance's own arguments
    // already carry -- the arguments are where a demand
    // through an anonymous constant leaves the enclosing
    // function's parameters, the recorded shape of the
    // parameter-self judgment.
    set_param_self_judgment(true);
    let result = screen_question(|| {
        let mut visited = HashMap::new();
        projection_screen_args(&inst.args(), None, 0, &mut visited)
    })
    .or_else(|| {
        screen_question(|| {
            let mut visited = HashMap::new();
            projection_screen_ty(&inst.ty(), None, 0, &mut visited)
        })
    });
    set_param_self_judgment(false);
    result
}

/// Whether a body mentions a projection that cannot
/// normalize (see [`unnormalizable_projection_in_instance`]),
/// returning the recorded reason for the first one found.
/// The body is a definition body — fetched without
/// normalization — and the danger is inside its conversion:
/// resolving an associated call whose arguments mention
/// such a projection poses the normalization (an anonymous
/// constant inside a generic function, calling an
/// associated function through a projection over the
/// function's own parameter, is the known shape). The walk
/// covers the locals' types and the call terminators'
/// operand types; a call's callee constant carries the
/// call's generic arguments in its function type.
pub fn unnormalizable_projection_in_body(
    body: &mir::Body,
) -> Option<String> {
    screen_question(|| unnormalizable_projection_in_body_walk(body))
}

fn unnormalizable_projection_in_body_walk(
    body: &mir::Body,
) -> Option<String> {
    let mut visited = HashMap::new();
    for local in body.locals() {
        if let Some(reason) = projection_screen_ty(&local.ty, None, 0, &mut visited) {
            return Some(reason);
        }
    }
    for block in &body.blocks {
        if let mir::TerminatorKind::Call { func, args, .. } =
            &block.terminator.kind
        {
            for op in std::iter::once(func).chain(args.iter()) {
                let reason = match op {
                    mir::Operand::Constant(c) => {
                        projection_screen_ty(&c.const_.ty(), None, 0, &mut visited)
                            .or_else(|| match c.const_.kind() {
                                ConstantKind::Ty(tc) => {
                                    projection_screen_const(&tc, None, 0, &mut visited)
                                }
                                ConstantKind::Unevaluated(uneval) => {
                                    projection_screen_args(
                                        &uneval.args,
                                        None,
                                        0,
                                        &mut visited,
                                    )
                                }
                                _ => None,
                            })
                    }
                    mir::Operand::RuntimeChecks(_) => None,
                    mir::Operand::Copy(place) | mir::Operand::Move(place) => {
                        place.ty(body.locals()).ok().and_then(|ty| {
                            projection_screen_ty(&ty, None, 0, &mut visited)
                        })
                    }
                };
                if reason.is_some() {
                    return reason;
                }
            }
        }
    }
    None
}
