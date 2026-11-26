//! Collection of the local, closure, const and coroutine instances a crate item demands.

use super::*;

fn fail_all(
    demands: &mut HashMap<rustc_public::DefId, Demand>,
    owners: &[rustc_public::DefId],
    reason: String,
) {
    for owner in owners {
        *demands
            .entry(*owner)
            .or_default()
            .failures
            .entry(reason.clone())
            .or_default() += 1;
    }
}

/// Fail the owners of one demand fetch (`fetch_instance_body`)
/// with the fetch's reason, translating the body-fetch
/// normalization overflow (see
/// `convert::BODY_FETCH_NORMALIZE_OVERFLOW`) into the
/// recursion-limit compilation failure: the compiler itself
/// cannot materialize that instance, so the demand failure is
/// recorded as "recursion limit reached" and the first such
/// instance is named on the demand state's recursion-limit
/// channel, exactly as the demand passes' own depth mirror
/// records its trips.
fn fail_fetch(
    demands: &mut HashMap<rustc_public::DefId, Demand>,
    owners: &[rustc_public::DefId],
    recursion_limit: &mut Option<String>,
    inst_name: &str,
    reason: String,
) {
    if reason == sparagmos_driver::convert::BODY_FETCH_NORMALIZE_OVERFLOW {
        if recursion_limit.is_none() {
            *recursion_limit = Some(format!(
                "reached the recursion limit while instantiating `{}`",
                inst_name
            ));
        }
        fail_all(demands, owners, "recursion limit reached".to_string());
    } else {
        fail_all(demands, owners, reason);
    }
}

/// Record the first body fetch whose whole-body normalization
/// overflowed the trait solver on the item loop's and the
/// reachability collection's shared channel, naming the
/// instance in the compilation-failure form rustc's collector
/// reports for the same overflow.
pub(crate) fn note_body_fetch_recursion_limit(slot: &mut Option<String>, inst: &Instance) {
    if slot.is_none() {
        *slot = Some(format!(
            "reached the recursion limit while instantiating `{}`",
            inst.name()
        ));
    }
}

/// State shared by the demand passes of one program
/// ([`collect_local_instances`] runs first over the converted
/// local bodies, then over the foreign bodies the reachability
/// collection obtained): which instances were already considered,
/// how many instance bodies were taken against
/// [`MAX_LOCAL_INSTANCES`], which definitions already own their
/// plain display name, and the per-definition outcomes the passes
/// accumulated.
#[derive(Default)]
pub(crate) struct InstanceDemandState {
    seen: HashSet<(rustc_public::DefId, Instance)>,
    unresolved_seen: HashSet<(rustc_public::DefId, String)>,
    plain_taken: HashSet<rustc_public::DefId>,
    taken: usize,
    pub(crate) demands: HashMap<rustc_public::DefId, Demand>,
    /// Outcomes accumulated for closures defined in other crates
    /// (see [`collect_closure_instances`] and
    /// [`collect_const_closure_instances`]): a foreign closure has
    /// no item-loop report entry to fill in, so the caller records
    /// these with [`apply_foreign_closure_demands`].
    pub(crate) foreign_closure_demands: HashMap<rustc_public::DefId, Demand>,
    /// Outcomes accumulated for plain functions defined in other
    /// crates by [`collect_const_fn_instances`]: the counterpart
    /// of `foreign_closure_demands` for function items, recorded
    /// with the same [`apply_foreign_closure_demands`].
    pub(crate) foreign_fn_demands: HashMap<rustc_public::DefId, Demand>,
    /// The first recursion-limit failure the demand passes hit
    /// (see [`RECURSION_LIMIT_DEPTH`]), reported as a compilation
    /// failure alongside the monomorphization checks.
    pub(crate) recursion_limit: Option<String>,
}

/// One demanded instance whose body was fetched during the
/// demand scan of [`collect_local_instances`] but whose
/// conversion is deferred until the scan drains (see the note
/// at the fetch site there).
struct PendingInstance {
    referenced: rustc_public::DefId,
    inst: rustc_public::mir::mono::Instance,
    owners: Vec<rustc_public::DefId>,
    ref_name: String,
    ibody: rustc_public::mir::Body,
}

/// Convert one deferred instance body and record the outcome
/// exactly as the demand scan's inline conversion did: the
/// body joins the program under its display name (the plain
/// definition name for the first converted instance of a
/// definition, the instance-suffixed name afterwards), the
/// owners' converted counts rise, and a conversion failure is
/// recorded against every owner.
#[allow(clippy::too_many_arguments)]
fn convert_pending_instance(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    plain_taken: &mut HashSet<rustc_public::DefId>,
    taken: &mut usize,
    demands: &mut HashMap<rustc_public::DefId, Demand>,
    pending: PendingInstance,
) {
    let PendingInstance {
        referenced,
        inst,
        owners,
        ref_name,
        ibody,
    } = pending;
    match convert_body_isolated(conv, &ibody) {
        Ok(converted) => {
            // No representability screen: these are local
            // functions, and the item loop converts local
            // bodies without one. A body the interpreter
            // cannot represent stays in the program and is
            // recorded unconverted by the interpreter's own
            // per-body isolation if it is ever called.
            *taken += 1;
            let counts = (
                ibody.blocks.len(),
                ibody.locals().len(),
                ibody.arg_locals().len(),
            );
            // Display name: the first converted instance of
            // the referenced definition carries its plain
            // name; further instances are suffixed with the
            // instance name (the reachability collection's
            // rule for foreign instances).
            let display = if plain_taken.insert(referenced) {
                ref_name.clone()
            } else {
                format!("{} [{}]", ref_name, inst.name())
            };
            bodies.push((display, converted));
            body_def_ids.push(conv.instance_def_id(referenced, &inst));
            body_def_names.push(ref_name);
            for owner in &owners {
                let demand = demands.entry(*owner).or_default();
                demand.converted += 1;
                if demand.first_counts.is_none() {
                    demand.first_counts = Some(counts);
                }
            }
        }
        Err(errs) => {
            for e in errs {
                fail_all(demands, &owners, e);
            }
        }
    }
}

/// Convert the monomorphic instances of generic local function
/// definitions that a set of bodies actually calls.
///
/// A generic definition has no instance as an item, so the item
/// loop cannot convert it; but every call site names the
/// definition through an `FnDef` target with concrete generic
/// arguments, and resolving those targets yields the instances
/// the program demands — the same mechanism the reachability
/// collection uses for foreign functions. This pass walks `seeds`
/// (and, transitively, the instance bodies themselves) for
/// `FnDef` targets, resolves the ones belonging to deferred
/// definitions, and converts each distinct instance, registering
/// it under the same `(referenced definition, instance)` key call
/// sites resolve through.
///
/// The pass runs several times per program with shared `state`:
/// first over the converted local bodies, then — interleaved with
/// the reachability collection — over the foreign bodies that
/// collection obtained, which can demand further local instances
/// (a std adapter calling a trait method of a local generic type
/// is the common shape). Sharing the state keeps the per-program
/// cap and the instance deduplication in force across passes.
/// Outcomes accumulate in the state; the caller records them in
/// the report with [`apply_instance_demands`] once the last pass
/// has run. The Stable MIR bodies of the instances this pass
/// converted are returned so the reachability collection can walk
/// them for foreign targets too.
pub(crate) fn collect_local_instances(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    deferred: &[DeferredFn],
    seeds: &[rustc_public::mir::Body],
    state: &mut InstanceDemandState,
) -> Vec<rustc_public::mir::Body> {
    use std::collections::VecDeque;

    let InstanceDemandState {
        seen,
        unresolved_seen,
        plain_taken,
        taken,
        demands,
        foreign_closure_demands: _,
        foreign_fn_demands: _,
        recursion_limit,
    } = state;

    // A recursion-limit failure recorded by an earlier pass
    // seals the compilation verdict (the demand state's slot is
    // the verdict chain's first input and is never cleared), so
    // no demand collected here can change the outcome; skip the
    // pass entirely.
    if recursion_limit.is_some() {
        return Vec::new();
    }

    let deferred_defs: HashSet<rustc_public::DefId> = deferred.iter().map(|d| d.def).collect();
    let deferred_ctors: HashSet<rustc_public::DefId> =
        deferred.iter().filter(|d| d.ctor).map(|d| d.def).collect();
    let mut queue: VecDeque<(rustc_public::mir::Body, DemandPath)> =
        seeds.iter().map(|b| (b.clone(), DemandPath::new())).collect();
    let mut extra_seeds: Vec<rustc_public::mir::Body> = Vec::new();
    // Conversions are deferred until the demand scan below
    // drains: the scan (target resolution, the size and depth
    // caps, the body fetch) is what decides whether the demand
    // graph is finite, and a size- or depth-increasing
    // instantiation chain is only known to be one once the
    // caps trip — converting each body as it is fetched spends
    // the full conversion cost of every level of such a chain
    // (each level's types larger than the last, its projection
    // resolutions posing trait goals that overflow the solver
    // and are reported one by one) before the trip can seal
    // the compilation failure that makes the conversions moot.
    // rustc's collector, which the caps mirror, never converts
    // at all: it records the demanded instances and reports
    // the recursion limit the moment the depth cap trips.
    // Deferring keeps the scan's decisions identical — the
    // same targets are resolved, capped and fetched in the
    // same order, and the conversions then run in fetch order
    // — while a scan that seals a recursion-limit failure
    // skips them wholesale.
    let mut pending: std::collections::VecDeque<PendingInstance> =
        std::collections::VecDeque::new();

    while let Some((body, path)) = queue.pop_front() {
        // A recursion-limit failure seals the compilation
        // verdict (see the entry check): the demands of the
        // bodies still queued cannot change it, and the
        // deferred conversions are skipped with the scan.
        if recursion_limit.is_some() {
            break;
        }
        // Instance resolution and body access go through the
        // compiler's internals and can panic on exotic targets
        // (higher-ranked and late-bound types are known cases).
        // Isolate each scanned body: a panic abandons that body's
        // remaining targets instead of the whole conversion, so
        // instance collection can never make a crate convert worse
        // than the item loop alone did.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // The demand targets of this body: its `FnDef` call
        // targets, plus every `FnDef` constant it materializes
        // that names a deferred definition. A constructor used
        // as a value (assigned, passed on, returned) never
        // appears as a call target of the body that demands it,
        // so call targets alone would miss the demand (Lane CF,
        // constructors); the same holds for a plain generic
        // function used as a value, so the scan follows `FnDef`
        // constants for every deferred definition (Lane CP).
        // Targets are filtered by deferred membership below, so
        // constants naming anything else cost one hash lookup.
        let mut targets = sparagmos_driver::reach::call_targets(&body);
        if !deferred.is_empty() {
            // Constants are resolved and filtered by the same
            // referenced-or-resolved deferred check as call targets
            // below, not by the referenced definition alone: a
            // constant can name a trait method whose resolution is
            // the deferred definition (an impl method). `core::fmt`'s
            // argument constructors materialize exactly that shape —
            // the formatter stored in an `Argument` is the trait
            // method `FnDef` constant in the constructor's body,
            // which the reachability pass walks for this purpose.
            targets.extend(sparagmos_driver::reach::fndef_constants(&body));
        }
        for (def, args) in targets {
            let referenced = def.def_id();
            let inst = match sparagmos_driver::convert::resolve_instance_guarded(def, &args) {
                Some(inst) => inst,
                None => {
                    if deferred_defs.contains(&referenced)
                        && unresolved_seen.insert((referenced, format!("{args:?}")))
                    {
                        *demands
                            .entry(referenced)
                            .or_default()
                            .failures
                            .entry("instance resolution failed".to_string())
                            .or_default() += 1;
                    }
                    continue;
                }
            };
            let resolved = inst.def.def_id();
            if !deferred_defs.contains(&referenced) && !deferred_defs.contains(&resolved) {
                continue;
            }
            if !seen.insert((referenced, inst)) {
                continue;
            }
            // The instance is credited to every deferred definition
            // it belongs to: the referenced definition (a direct
            // call, or a trait method whose default body this is)
            // and the resolved definition when it differs (an impl
            // method demanded through its trait).
            let mut owners: Vec<rustc_public::DefId> = Vec::new();
            if deferred_defs.contains(&referenced) {
                owners.push(referenced);
            }
            if resolved != referenced && deferred_defs.contains(&resolved) {
                owners.push(resolved);
            }
            // The expanded-size cap (see
            // `sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE`):
            // a size-increasing recursion nests its definition
            // only linearly deep, so the depth check below cannot
            // bind before the type expansion exhausts memory.
            // The size of the demanded arguments is the quantity
            // that explodes, so it is the one capped here;
            // tripping it is reported exactly like the depth
            // limit, since it is the same rustc failure, reached
            // through the expansion rustc's interning keeps
            // implicit.
            if !sparagmos_driver::reach::generic_args_within_expanded_size(
                &args,
                sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE,
            ) {
                if recursion_limit.is_none() {
                    *recursion_limit = Some(format!(
                        "reached the recursion limit while instantiating `{}` (instantiation size limit)",
                        def.name()
                    ));
                }
                fail_all(&mut *demands, &owners, "recursion limit reached".to_string());
                continue;
            }
            // The recursion-limit check mirrors rustc's mono
            // collector: if this instance nests its definition
            // deeper along the current demand path than the
            // recursion limit, instantiating it is the failure
            // rustc reports, however large the collection as a
            // whole legitimately is. A merely large but finite
            // demand never nests one definition this deep along a
            // single path, so the collection cap below stays the
            // only consequence for it.
            let depth = path.get(&resolved).copied().unwrap_or(0) + 1;
            if depth > RECURSION_LIMIT_DEPTH {
                if recursion_limit.is_none() {
                    *recursion_limit = Some(format!(
                        "reached the recursion limit while instantiating `{}`",
                        inst.name()
                    ));
                }
                fail_all(&mut *demands, &owners, "recursion limit reached".to_string());
                continue;
            }
            let mut child_path = path.clone();
            *child_path.entry(resolved).or_default() += 1;
            if *taken >= MAX_LOCAL_INSTANCES {
                fail_all(&mut *demands, &owners, "collection limit reached".to_string());
                continue;
            }
            // Mirror the reachability collection's guards before
            // asking for the body: a virtual instance has no MIR
            // body, and asking the compiler for one panics inside
            // it (and the panic's ICE diagnostic poisons the whole
            // compilation even when the unwind is caught).
            if matches!(inst.kind, rustc_public::mir::mono::InstanceKind::Virtual { .. }) {
                fail_all(
                    &mut *demands,
                    &owners,
                    "dynamic dispatch target (virtual instance)".to_string(),
                );
                continue;
            }
            if !inst.has_body() {
                // A compiler intrinsic the interpreter models
                // is served by the builtin, the way a demanded
                // data constructor is served by its callers'
                // aggregate rvalues: there is no body to
                // convert, and the demand is met. The model is
                // looked up by the intrinsic's own name (see
                // `builtins::intrinsic_lookup`): an intrinsic
                // the guest crate declares itself is named by
                // its declaration site, which the qualified
                // registry never matches. An intrinsic with no
                // model stays an unmet demand.
                if matches!(
                    inst.kind,
                    rustc_public::mir::mono::InstanceKind::Intrinsic
                ) {
                    conv.note_intrinsic(&inst);
                    let canonical = sparagmos_interp::builtins::canonical_name(
                        &inst.def.def_id().name(),
                    );
                    if sparagmos_interp::builtins::intrinsic_lookup(&canonical).is_some() {
                        for owner in &owners {
                            demands.entry(*owner).or_default().converted += 1;
                        }
                        continue;
                    }
                }
                let reason = match inst.kind {
                    rustc_public::mir::mono::InstanceKind::Intrinsic => {
                        "compiler intrinsic (no fallback body)"
                    }
                    _ => "no MIR body available",
                };
                fail_all(&mut *demands, &owners, reason.to_string());
                continue;
            }
            // Mirror the virtual guard for compiler-generated
            // impl bodies over a SIMD type: the derived body
            // projects the ADT's fields, the MIR validator bans
            // projections into SIMD types (MCP#838), and the
            // ICE diagnostic emitted while asking poisons the
            // whole compilation even when the panic is
            // contained. The derived rendering is served
            // without the body (the interpreter's structural
            // Debug shapes, or the trait's builtin model), so
            // the demand is recorded unmet instead of fetched.
            // Hand-written methods over SIMD types do not trip
            // the ban (their bodies choose their own accesses)
            // and are fetched as usual.
            if sparagmos_driver::convert::generated_simd_impl_method(&inst) {
                fail_all(
                    &mut *demands,
                    &owners,
                    "compiler-generated impl body over a SIMD type".to_string(),
                );
                continue;
            }
            let ibody = match fetch_instance_body(&inst) {
                Ok(Some(b)) => b,
                Ok(None) => {
                    fail_all(&mut *demands, &owners, "no MIR body available".to_string());
                    continue;
                }
                Err(reason) => {
                    fail_fetch(
                        &mut *demands,
                        &owners,
                        recursion_limit,
                        &inst.name(),
                        reason,
                    );
                    continue;
                }
            };
            // The fetched body joins the scan queue and the
            // seed set exactly as an already-converted body
            // would; only the conversion itself is deferred
            // (see `pending` above).
            queue.push_back((ibody.clone(), child_path.clone()));
            extra_seeds.push(ibody.clone());
            pending.push_back(PendingInstance {
                referenced,
                inst,
                owners,
                ref_name: def.name(),
                ibody,
            });
            // The collection cap judges conversions, so it
            // must see the same `taken` count an inline
            // conversion would have produced by now: flush the
            // deferred conversions before the cap can bind, so
            // no cap check ever runs against a count that
            // pending conversions would still change. Away
            // from the cap the check below cannot fire in
            // either arrangement (taken + pending stays under
            // it), which is what makes the deferral exact.
            if *taken + pending.len() >= MAX_LOCAL_INSTANCES {
                while let Some(item) = pending.pop_front() {
                    convert_pending_instance(
                        &mut *conv,
                        &mut *bodies,
                        &mut *body_def_ids,
                        &mut *body_def_names,
                        &mut *plain_taken,
                        &mut *taken,
                        &mut *demands,
                        item,
                    );
                }
            }
        }
        }));
        if outcome.is_err() {
            // The panic capture slot keeps the first panic's
            // message; this panic was contained, so clear the
            // slot — a later, genuinely fatal panic must report
            // its own message.
            let _ = take_captured_panic();
        }
    }

    // Run the deferred conversions in fetch order — unless
    // the scan sealed a recursion-limit failure, in which case
    // the compilation is failed with that verdict and the
    // converted bodies would be discarded unread. Each
    // conversion is isolated the way the scan isolates each
    // body: a panic the conversion's own containment resumes
    // abandons that one conversion, not the rest.
    if recursion_limit.is_none() {
        while let Some(item) = pending.pop_front() {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                convert_pending_instance(
                    &mut *conv,
                    &mut *bodies,
                    &mut *body_def_ids,
                    &mut *body_def_names,
                    &mut *plain_taken,
                    &mut *taken,
                    &mut *demands,
                    item,
                );
            }));
            if outcome.is_err() {
                let _ = take_captured_panic();
            }
        }
    }

    extra_seeds
}

/// Convert the monomorphic instances of closures defined in
/// generic contexts that a set of bodies actually mentions.
///
/// A closure defined inside a generic function has no instance as
/// an item (see [`closure_instance`]), but every monomorphic
/// instance of the enclosing definition mentions the closure type
/// with the enclosing substitution applied — in a closure
/// aggregate, a local's type, or a call target's generic
/// arguments. This pass walks `seeds` (and, transitively, the
/// closure bodies themselves) for such mentions via
/// [`sparagmos_driver::reach::closure_targets`], resolves each
/// concrete one through `Instance::resolve_closure` with the
/// closure's own kind, and converts it, registering the body under
/// the same `(closure definition, instance)` key
/// [`sparagmos_driver::convert::Converter::closure_def_id`] assigns
/// to references. Mentions whose arguments still mention
/// parameters are not demands this pass can satisfy — the
/// enclosing body is itself generic — and are left to the pass
/// over the monomorphic instances. Outcomes accumulate in the
/// shared demand state (cap and deduplication are shared with the
/// function-instance collection); the caller records them with
/// [`apply_instance_demands`]. The Stable MIR bodies converted
/// here are returned so the reachability collection can walk them
/// for foreign targets too.
///
/// The same pass also collects closures defined in *other*
/// crates. A foreign closure is never an item of the local crate,
/// so no item loop defers it, and the reachability collection
/// follows `FnDef` targets and constants only — a closure created
/// inside a generic foreign function (the closure
/// `std::sys::thread_local::destroy::<T>` passes to
/// `abort_on_dtor_unwind`, the one inside `slice::binary_search`)
/// is mentioned concretely by the collected foreign body yet
/// demanded by no pass, and calls to it fail as unknown
/// functions. Foreign outcomes accumulate in their own map of
/// the shared state and are recorded with
/// [`apply_foreign_closure_demands`]: there is no local item to
/// credit them to.
pub(crate) fn collect_closure_instances(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    deferred: &[DeferredFn],
    local_defs: &HashSet<rustc_public::DefId>,
    seeds: &[rustc_public::mir::Body],
    collected: &HashSet<sparagmos_mir::DefId>,
    state: &mut InstanceDemandState,
) -> Vec<rustc_public::mir::Body> {
    use std::collections::VecDeque;

    let InstanceDemandState {
        seen,
        unresolved_seen,
        plain_taken,
        taken,
        demands,
        foreign_closure_demands,
        foreign_fn_demands: _,
        recursion_limit,
    } = state;

    // A recursion-limit failure recorded by an earlier pass
    // seals the compilation verdict (the demand state's slot is
    // the verdict chain's first input and is never cleared), so
    // no demand collected here can change the outcome; skip the
    // pass entirely.
    if recursion_limit.is_some() {
        return Vec::new();
    }

    let deferred_names: HashMap<rustc_public::DefId, &str> =
        deferred.iter().map(|d| (d.def, d.name.as_str())).collect();
    let mut queue: VecDeque<rustc_public::mir::Body> = seeds.iter().cloned().collect();
    let mut extra_seeds: Vec<rustc_public::mir::Body> = Vec::new();

    while let Some(body) = queue.pop_front() {
        // Instance resolution goes through the compiler's
        // internals and can panic on exotic targets; isolate each
        // scanned body exactly as the function-instance collection
        // does, so closure collection can never make a crate
        // convert worse than the item loop alone did.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for (def, args) in sparagmos_driver::reach::closure_targets(&body) {
                let referenced = def.def_id();
                // Two kinds of closures are collected here. One
                // defined in a generic context of the local crate
                // was deferred by the item loop and is accounted
                // through the shared demand state, like a generic
                // function. One defined in another crate is
                // demanded by no other pass: no item loop iterates
                // foreign items, and the reachability collection
                // follows `FnDef` targets only, so a foreign
                // closure a converted body mentions concretely
                // (the closure inside a generic std function is
                // the common case) is collected here or nowhere;
                // it is accounted in the state's foreign map
                // instead. A local closure outside a generic
                // context is the item loop's business and is
                // skipped.
                let (base_name, foreign) = match deferred_names.get(&referenced) {
                    Some(&name) => (name.to_string(), false),
                    None => {
                        if local_defs.contains(&referenced) {
                            continue;
                        }
                        (referenced.name(), true)
                    }
                };
                let target_demands: &mut HashMap<rustc_public::DefId, Demand> = if foreign {
                    &mut *foreign_closure_demands
                } else {
                    &mut *demands
                };
                if args_mention_params(&args) {
                    continue;
                }
                let owners = [referenced];
                // The expanded-size cap, as in the function
                // demand pass: a closure demanded with
                // size-increasing arguments is the same
                // recursion-limit situation. A foreign closure
                // is the exception: its arguments come from a
                // body the compiler already monomorphized, so
                // the instance is finite by construction (the
                // foreign demand graph of a deeply nested but
                // legitimate program measures far past the cap;
                // see the reachability collection, which exempts
                // foreign demand for the same reason). Tripping
                // the cap there fails only that demand, recorded
                // under the same reason; it must not fail the
                // whole compilation as a recursion-limit error.
                if !sparagmos_driver::reach::generic_args_within_expanded_size(
                    &args,
                    sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE,
                ) {
                    if !foreign && recursion_limit.is_none() {
                        *recursion_limit = Some(format!(
                            "reached the recursion limit while instantiating `{}` (instantiation size limit)",
                            base_name
                        ));
                    }
                    fail_all(&mut *target_demands, &owners, "recursion limit reached".to_string());
                    continue;
                }
                let Some(kind) = closure_kind(&args) else {
                    if coroutine_closure_args(&args) {
                        // An async closure demand: the definition
                        // has no closure instance of its own (the
                        // compiler's closure resolution rejects
                        // coroutine closures), and its execution
                        // bodies — the AsyncFn shim and the
                        // coroutine — are collected by the instance
                        // and coroutine demands under their own
                        // definitions. Record no demand for the
                        // closure definition itself: the deferred
                        // item is accounted not demanded, under the
                        // coroutine-closure reason it was deferred
                        // with.
                        continue;
                    }
                    if unresolved_seen.insert((referenced, format!("{args:?}"))) {
                        *target_demands
                            .entry(referenced)
                            .or_default()
                            .failures
                            .entry(
                                "closure kind is not recoverable from its generic arguments"
                                    .to_string(),
                            )
                            .or_default() += 1;
                    }
                    continue;
                };
                let inst = match sparagmos_driver::convert::resolve_closure_guarded(def, &args, kind) {
                    Some(inst) => inst,
                    None => {
                        if unresolved_seen.insert((referenced, format!("{args:?}"))) {
                            *target_demands
                                .entry(referenced)
                                .or_default()
                                .failures
                                .entry("closure instance resolution failed".to_string())
                                .or_default() += 1;
                        }
                        continue;
                    }
                };
                if !seen.insert((referenced, inst)) {
                    continue;
                }
                // The reachability collection resolves the
                // closures in a call's generic arguments itself,
                // so an instance it already converted must not be
                // converted again here: two bodies under one
                // program DefId collide when the program is built.
                // Its copy (and its report entry) stands.
                let collected_id = conv.instance_def_id(referenced, &inst);
                if collected.contains(&collected_id) {
                    continue;
                }
                if *taken >= MAX_LOCAL_INSTANCES {
                    fail_all(&mut *target_demands, &owners, "collection limit reached".to_string());
                    continue;
                }
                if !inst.has_body() {
                    fail_all(&mut *target_demands, &owners, "no MIR body available".to_string());
                    continue;
                }
                // As in `collect_local_instances`: a
                // compiler-generated impl body over a SIMD type
                // must not be asked for (MCP#838 validation ICE).
                if sparagmos_driver::convert::generated_simd_impl_method(&inst) {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "compiler-generated impl body over a SIMD type".to_string(),
                    );
                    continue;
                }
                let ibody = match fetch_instance_body(&inst) {
                    Ok(Some(b)) => b,
                    Ok(None) => {
                        fail_all(
                            &mut *demands,
                            &owners,
                            "no MIR body available".to_string(),
                        );
                        continue;
                    }
                    Err(reason) => {
                        fail_fetch(
                            &mut *demands,
                            &owners,
                            recursion_limit,
                            &inst.name(),
                            reason,
                        );
                        continue;
                    }
                };
                match convert_body_isolated(&mut *conv, &ibody) {
                    Ok(converted) => {
                        *taken += 1;
                        queue.push_back(ibody.clone());
                        extra_seeds.push(ibody.clone());
                        let counts =
                            (ibody.blocks.len(), ibody.locals().len(), ibody.arg_locals().len());
                        let display = if plain_taken.insert(referenced) {
                            base_name.to_string()
                        } else {
                            format!("{} [{}]", base_name, inst.name())
                        };
                        bodies.push((display, converted));
                        body_def_ids.push(collected_id);
                        body_def_names.push(base_name.to_string());
                        let demand = target_demands.entry(referenced).or_default();
                        demand.converted += 1;
                        if demand.first_counts.is_none() {
                            demand.first_counts = Some(counts);
                        }
                    }
                    Err(errs) => {
                        for e in errs {
                            fail_all(&mut *target_demands, &owners, e);
                        }
                    }
                }
            }
        }));
        if outcome.is_err() {
            // Contained panic: clear the capture slot, as the
            // function-instance collection does.
            let _ = take_captured_panic();
        }
    }

    extra_seeds
}

/// Convert the closures that converted bodies materialize as
/// *values* out of constants.
///
/// A closure can be demanded without any body's types mentioning
/// it: const evaluation stores it inside a constant allocation —
/// a function pointer in a constant struct, a closure value
/// nested in a constant aggregate — and body conversion decodes
/// the allocation into a closure value (see
/// `Converter::const_closure_mentions`). The canonical case is
/// `std::hash::RandomState::new::KEYS`: the `LocalKey` value is a
/// constant whose function-pointer field names the thread-local
/// initializer closure, so every `HashMap` program materializes
/// that closure as a value while no local declaration, operand
/// type, or aggregate anywhere mentions its type. The
/// type-driven passes ([`collect_closure_instances`],
/// [`sparagmos_driver::reach::closure_targets`]) cannot see it,
/// and the foreign constant's own body is never converted (no
/// item loop iterates foreign items), so without this pass the
/// closure's body is collected by no pass and calls to it fail
/// as unknown functions.
///
/// The mentions are exactly the constants conversion itself
/// decoded — constants of reached bodies and the initializers of
/// statics they reference — so the pass demands nothing
/// conversion did not already touch. Routing and accounting
/// mirror the closure demand pass: a closure the item loop
/// deferred (a generic-context local closure) is accounted in
/// the shared demand state, a local closure outside a generic
/// context is the item loop's business and is skipped, and a
/// closure defined in another crate is accounted in the state's
/// foreign map (see [`apply_foreign_closure_demands`]). The
/// shared `seen` set deduplicates against the type-driven pass:
/// whichever pass reaches an instance first converts it.
///
/// Coroutine mentions are not part of this record: a constant
/// cannot hold a coroutine value (the constant decoder has no
/// coroutine value form — coroutines are created by aggregates
/// in bodies, never stored in allocations), so there is no
/// constant-content coroutine mention to scan for.
pub(crate) fn collect_const_closure_instances(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    deferred: &[DeferredFn],
    local_defs: &HashSet<rustc_public::DefId>,
    collected: &HashSet<sparagmos_mir::DefId>,
    state: &mut InstanceDemandState,
) -> Vec<rustc_public::mir::Body> {
    let InstanceDemandState {
        seen,
        unresolved_seen,
        plain_taken,
        taken,
        demands,
        foreign_closure_demands,
        foreign_fn_demands: _,
        recursion_limit,
    } = state;

    // A recursion-limit failure recorded by an earlier pass
    // seals the compilation verdict (the demand state's slot is
    // the verdict chain's first input and is never cleared), so
    // no demand collected here can change the outcome; skip the
    // pass entirely.
    if recursion_limit.is_some() {
        return Vec::new();
    }

    let deferred_names: HashMap<rustc_public::DefId, &str> =
        deferred.iter().map(|d| (d.def, d.name.as_str())).collect();
    let mut converted_bodies: Vec<rustc_public::mir::Body> = Vec::new();

    // Converting one mentioned closure's body decodes its own
    // constants, which can mention further closures; drain the
    // converter's record until a round adds nothing. Each
    // distinct mention is recorded once and each conversion is
    // deduplicated and capped through the shared state, so the
    // loop terminates.
    loop {
        let mentions = std::mem::take(&mut conv.const_closure_mentions);
        if mentions.is_empty() {
            break;
        }
        for (def, args) in &mentions {
            let referenced = def.def_id();
            let (base_name, foreign) = match deferred_names.get(&referenced) {
                Some(&name) => (name.to_string(), false),
                None => {
                    if local_defs.contains(&referenced) {
                        continue;
                    }
                    (referenced.name(), true)
                }
            };
            if args_mention_params(args) {
                continue;
            }
            // Instance resolution goes through the compiler's
            // internals and can panic on exotic targets; isolate
            // each mention exactly as the other demand passes
            // isolate each scanned body.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let target_demands: &mut HashMap<rustc_public::DefId, Demand> = if foreign {
                    &mut *foreign_closure_demands
                } else {
                    &mut *demands
                };
                let owners = [referenced];
                // The expanded-size cap, as in the closure
                // demand pass: a closure demanded with
                // size-increasing arguments is the same
                // recursion-limit situation. As there, a
                // foreign closure's oversized arguments fail
                // only its own demand: the instance comes from
                // a constant the compiler already monomorphized,
                // so its size is finite by construction and is
                // not a recursion-limit violation.
                if !sparagmos_driver::reach::generic_args_within_expanded_size(
                    args,
                    sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE,
                ) {
                    if !foreign && recursion_limit.is_none() {
                        *recursion_limit = Some(format!(
                            "reached the recursion limit while instantiating `{}` (instantiation size limit)",
                            base_name
                        ));
                    }
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "recursion limit reached".to_string(),
                    );
                    return;
                }
                let Some(kind) = closure_kind(args) else {
                    if coroutine_closure_args(args) {
                        // An async closure demand: no demand is
                        // recorded for the closure definition
                        // itself (see collect_closure_instances).
                        return;
                    }
                    if unresolved_seen.insert((referenced, format!("{args:?}"))) {
                        fail_all(
                            &mut *target_demands,
                            &owners,
                            "closure kind is not recoverable from its generic arguments"
                                .to_string(),
                        );
                    }
                    return;
                };
                let inst = match sparagmos_driver::convert::resolve_closure_guarded(*def, args, kind) {
                    Some(inst) => inst,
                    None => {
                        if unresolved_seen.insert((referenced, format!("{args:?}"))) {
                            fail_all(
                                &mut *target_demands,
                                &owners,
                                "closure instance resolution failed".to_string(),
                            );
                        }
                        return;
                    }
                };
                if !seen.insert((referenced, inst)) {
                    return;
                }
                // Already converted by the reachability
                // collection (see collect_closure_instances):
                // a second copy under the same program DefId
                // would collide when the program is built.
                let collected_id = conv.instance_def_id(referenced, &inst);
                if collected.contains(&collected_id) {
                    return;
                }
                if *taken >= MAX_LOCAL_INSTANCES {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "collection limit reached".to_string(),
                    );
                    return;
                }
                if !inst.has_body() {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "no MIR body available".to_string(),
                    );
                    return;
                }
                let Some(ibody) = sparagmos_driver::convert::with_body_materialization(
                    &inst.name(),
                    || inst.body(),
                ) else {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "no MIR body available".to_string(),
                    );
                    return;
                };
                match convert_body_isolated(&mut *conv, &ibody) {
                    Ok(converted) => {
                        *taken += 1;
                        converted_bodies.push(ibody.clone());
                        let counts =
                            (ibody.blocks.len(), ibody.locals().len(), ibody.arg_locals().len());
                        let display = if plain_taken.insert(referenced) {
                            base_name.clone()
                        } else {
                            format!("{} [{}]", base_name, inst.name())
                        };
                        bodies.push((display, converted));
                        body_def_ids.push(collected_id);
                        body_def_names.push(base_name.clone());
                        let demand = target_demands.entry(referenced).or_default();
                        demand.converted += 1;
                        if demand.first_counts.is_none() {
                            demand.first_counts = Some(counts);
                        }
                    }
                    Err(errs) => {
                        for e in errs {
                            fail_all(&mut *target_demands, &owners, e);
                        }
                    }
                }
            }));
            if outcome.is_err() {
                // Contained panic: clear the capture slot, as the
                // other demand passes do.
                let _ = take_captured_panic();
            }
        }
    }

    converted_bodies
}

/// Convert the plain functions that converted bodies
/// materialize as *values* out of constants: the counterpart of
/// [`collect_const_closure_instances`] for function items.
///
/// A function pointer stored inside a constant allocation - a
/// `RawWakerVTable`'s clone / wake / wake_by_ref / drop entries
/// are the canonical case - names a body no converted body's
/// call operands mention: the call goes through the decoded
/// pointer value at run time, and the operand-level `FnDef`
/// scan ([`sparagmos_driver::reach::fndef_constants`]) only sees
/// constants whose *type* is the function item, never pointers
/// nested inside an aggregate allocation. The decoded value is
/// the only mention, recorded by conversion itself (see
/// `Converter::const_fn_mentions`), and constant evaluation has
/// already resolved each mention to its monomorphic instance,
/// so this pass converts exactly those instances. Routing and
/// accounting mirror the constant-closure pass: a deferred
/// (generic-context) local definition is accounted in the
/// shared demand state, a local definition outside a generic
/// context is the item loop's business and is skipped, and a
/// function defined in another crate is accounted in the
/// state's foreign map (see [`apply_foreign_closure_demands`]).
pub(crate) fn collect_const_fn_instances(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    deferred: &[DeferredFn],
    local_defs: &HashSet<rustc_public::DefId>,
    collected: &HashSet<sparagmos_mir::DefId>,
    state: &mut InstanceDemandState,
) -> Vec<rustc_public::mir::Body> {
    let InstanceDemandState {
        seen,
        unresolved_seen: _,
        plain_taken,
        taken,
        demands,
        foreign_closure_demands: _,
        foreign_fn_demands,
        recursion_limit,
    } = state;

    // A recursion-limit failure recorded by an earlier pass
    // seals the compilation verdict (the demand state's slot is
    // the verdict chain's first input and is never cleared), so
    // no demand collected here can change the outcome; skip the
    // pass entirely.
    if recursion_limit.is_some() {
        return Vec::new();
    }

    let deferred_names: HashMap<rustc_public::DefId, &str> =
        deferred.iter().map(|d| (d.def, d.name.as_str())).collect();
    let mut converted_bodies: Vec<rustc_public::mir::Body> = Vec::new();

    // Converting one mentioned function's body decodes its own
    // constants, which can mention further functions; drain the
    // converter's record until a round adds nothing, exactly as
    // the constant-closure pass drains its record.
    loop {
        let mentions = std::mem::take(&mut conv.const_fn_mentions);
        if mentions.is_empty() {
            break;
        }
        for inst in &mentions {
            let referenced = inst.def.def_id();
            let (base_name, foreign) = match deferred_names.get(&referenced) {
                Some(&name) => (name.to_string(), false),
                None => {
                    if local_defs.contains(&referenced) {
                        continue;
                    }
                    (referenced.name(), true)
                }
            };
            let args = inst.args();
            if args_mention_params(&args) {
                continue;
            }
            // Body access goes through the compiler's internals
            // and can panic on exotic targets; isolate each
            // mention exactly as the other demand passes isolate
            // each scanned body.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let target_demands: &mut HashMap<rustc_public::DefId, Demand> = if foreign {
                    &mut *foreign_fn_demands
                } else {
                    &mut *demands
                };
                let owners = [referenced];
                // The expanded-size cap, as in the other demand
                // passes. A foreign function's oversized
                // arguments fail only its own demand: the
                // instance comes from a constant the compiler
                // already monomorphized, so its size is finite
                // by construction.
                if !sparagmos_driver::reach::generic_args_within_expanded_size(
                    &args,
                    sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE,
                ) {
                    if !foreign && recursion_limit.is_none() {
                        *recursion_limit = Some(format!(
                            "reached the recursion limit while instantiating `{}` (instantiation size limit)",
                            base_name
                        ));
                    }
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "recursion limit reached".to_string(),
                    );
                    return;
                }
                if !seen.insert((referenced, *inst)) {
                    return;
                }
                // Already converted by the reachability
                // collection: a second copy under the same
                // program DefId would collide when the program
                // is built.
                let collected_id = conv.instance_def_id(referenced, inst);
                if collected.contains(&collected_id) {
                    return;
                }
                if *taken >= MAX_LOCAL_INSTANCES {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "collection limit reached".to_string(),
                    );
                    return;
                }
                // Mirror the instance demand pass's guard before
                // asking for the body: a virtual instance has no
                // MIR body, and asking the compiler for one
                // panics inside it.
                if matches!(inst.kind, rustc_public::mir::mono::InstanceKind::Virtual { .. }) {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "dynamic dispatch target (virtual instance)".to_string(),
                    );
                    return;
                }
                if !inst.has_body() {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "no MIR body available".to_string(),
                    );
                    return;
                }
                let Some(ibody) = sparagmos_driver::convert::with_body_materialization(
                    &inst.name(),
                    || inst.body(),
                ) else {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "no MIR body available".to_string(),
                    );
                    return;
                };
                match convert_body_isolated(&mut *conv, &ibody) {
                    Ok(converted) => {
                        *taken += 1;
                        converted_bodies.push(ibody.clone());
                        let counts =
                            (ibody.blocks.len(), ibody.locals().len(), ibody.arg_locals().len());
                        let display = if plain_taken.insert(referenced) {
                            base_name.clone()
                        } else {
                            format!("{} [{}]", base_name, inst.name())
                        };
                        bodies.push((display, converted));
                        body_def_ids.push(collected_id);
                        body_def_names.push(base_name.clone());
                        let demand = target_demands.entry(referenced).or_default();
                        demand.converted += 1;
                        if demand.first_counts.is_none() {
                            demand.first_counts = Some(counts);
                        }
                    }
                    Err(errs) => {
                        for e in errs {
                            fail_all(&mut *target_demands, &owners, e);
                        }
                    }
                }
            }));
            if outcome.is_err() {
                // Contained panic: clear the capture slot, as the
                // other demand passes do.
                let _ = take_captured_panic();
            }
        }
    }

    converted_bodies
}

/// Convert the demanded instances of the local coroutines
/// (async blocks, `gen` blocks, and the futures of `async fn`s
/// defined as separate items) that converted bodies create or
/// hold. A coroutine item has no monomorphic instance as an item
/// — `Instance::try_from` rejects it like a closure — so the item
/// loop defers it with the generic functions, but neither the
/// instance demand (which follows call targets and `FnDef`
/// constants) nor the closure demand (which follows closure
/// types) ever mentions it: a coroutine is demanded by the
/// `AggregateKind::Coroutine` rvalue that creates it and by the
/// coroutine-typed locals that hold it. This pass scans for
/// exactly those mentions ([`sparagmos_driver::reach::coroutine_targets`])
/// and converts each distinct demanded instance, recording the
/// outcome in the shared demand state so
/// [`apply_instance_demands`] accounts the deferred item.
///
/// The body comes from resolving the coroutine definition as an
/// instance (the monomorphization the compiler itself performs
/// for the creation site); when resolution refuses the
/// definition, the definition's own body is used — the arguments
/// are concrete here, so that body is already the monomorphic
/// one. Converted bodies reference a coroutine by its plain
/// definition key (the key the creation aggregate converts
/// through), so the first demanded instance's body is registered
/// under that key; a further instance of the same definition
/// counts as a converted demand without registering a second
/// body, the same one-body-per-definition shape the aggregate
/// conversion itself assumes.
pub(crate) fn collect_coroutine_instances(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    deferred: &[DeferredFn],
    seeds: &[rustc_public::mir::Body],
    state: &mut InstanceDemandState,
) -> Vec<rustc_public::mir::Body> {
    use std::collections::VecDeque;

    let InstanceDemandState {
        seen,
        unresolved_seen,
        plain_taken,
        taken,
        demands,
        foreign_closure_demands: _,
        foreign_fn_demands: _,
        recursion_limit,
    } = state;

    // A recursion-limit failure recorded by an earlier pass
    // seals the compilation verdict (the demand state's slot is
    // the verdict chain's first input and is never cleared), so
    // no demand collected here can change the outcome; skip the
    // pass entirely.
    if recursion_limit.is_some() {
        return Vec::new();
    }

    let deferred_names: HashMap<rustc_public::DefId, &str> =
        deferred.iter().map(|d| (d.def, d.name.as_str())).collect();
    let mut queue: VecDeque<rustc_public::mir::Body> = seeds.iter().cloned().collect();
    let mut extra_seeds: Vec<rustc_public::mir::Body> = Vec::new();

    while let Some(body) = queue.pop_front() {
        // Instance resolution and body access go through the
        // compiler's internals and can panic on exotic targets;
        // isolate each scanned body exactly as the other demand
        // passes do.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for (def, args) in sparagmos_driver::reach::coroutine_targets(&body) {
                let referenced = def.def_id();
                let Some(&base_name) = deferred_names.get(&referenced) else {
                    continue;
                };
                if coroutine_closure_args(&args) {
                    // An async closure demanded as a coroutine
                    // target: it has no instance of its own (see
                    // the closure collection), so there is no
                    // demand to record for its definition; the
                    // coroutine body inside it is a separate
                    // definition, collected by the reachability
                    // pass under its own name.
                    continue;
                }
                if args_mention_params(&args) {
                    continue;
                }
                let owners = [referenced];
                // The expanded-size cap, as in the function
                // demand pass: a coroutine demanded with
                // size-increasing arguments is the same
                // recursion-limit situation.
                if !sparagmos_driver::reach::generic_args_within_expanded_size(
                    &args,
                    sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE,
                ) {
                    if recursion_limit.is_none() {
                        *recursion_limit = Some(format!(
                            "reached the recursion limit while instantiating `{}` (instantiation size limit)",
                            base_name
                        ));
                    }
                    fail_all(&mut *demands, &owners, "recursion limit reached".to_string());
                    continue;
                }
                // Obtain the instance body: resolution first,
                // the definition's own body as the fallback.
                let ibody = match sparagmos_driver::convert::resolve_instance_guarded(
                    rustc_public::ty::FnDef(referenced),
                    &args,
                ) {
                    Some(inst) => {
                        if !seen.insert((referenced, inst)) {
                            continue;
                        }
                        if !inst.has_body() {
                            fail_all(
                                &mut *demands,
                                &owners,
                                "no MIR body available".to_string(),
                            );
                            continue;
                        }
                        // As in `collect_local_instances`: a
                        // compiler-generated impl body over a
                        // SIMD type must not be asked for
                        // (MCP#838 validation ICE).
                        if sparagmos_driver::convert::generated_simd_impl_method(&inst) {
                            fail_all(
                                &mut *demands,
                                &owners,
                                "compiler-generated impl body over a SIMD type"
                                    .to_string(),
                            );
                            continue;
                        }
                        fetch_instance_body(&inst)
                    }
                    None => {
                        if !unresolved_seen.insert((referenced, format!("{args:?}"))) {
                            continue;
                        }
                        fetch_query_normalize_aware(&def.name(), false, || def.body())
                    }
                };
                let ibody = match ibody {
                    Ok(Some(b)) => b,
                    Ok(None) => {
                        fail_all(
                            &mut *demands,
                            &owners,
                            "no MIR body available".to_string(),
                        );
                        continue;
                    }
                    Err(reason) => {
                        fail_fetch(
                            &mut *demands,
                            &owners,
                            recursion_limit,
                            base_name,
                            reason,
                        );
                        continue;
                    }
                };
                if *taken >= MAX_LOCAL_INSTANCES {
                    fail_all(
                        &mut *demands,
                        &owners,
                        "collection limit reached".to_string(),
                    );
                    continue;
                }
                match convert_body_isolated(conv, &ibody) {
                    Ok(converted) => {
                        *taken += 1;
                        queue.push_back(ibody.clone());
                        extra_seeds.push(ibody.clone());
                        let counts =
                            (ibody.blocks.len(), ibody.locals().len(), ibody.arg_locals().len());
                        let demand = demands.entry(referenced).or_default();
                        demand.converted += 1;
                        if demand.first_counts.is_none() {
                            demand.first_counts = Some(counts);
                        }
                        // Register the body under the plain
                        // definition key the creation aggregates
                        // reference, unless an earlier instance of
                        // this definition already registered one.
                        let plain = conv.crate_def_id(&def);
                        if !body_def_ids.contains(&plain) {
                            let display = if plain_taken.insert(referenced) {
                                base_name.to_string()
                            } else {
                                format!("{} [coroutine]", base_name)
                            };
                            bodies.push((display, converted));
                            body_def_ids.push(plain);
                            body_def_names.push(base_name.to_string());
                        }
                    }
                    Err(errs) => {
                        for e in errs {
                            fail_all(&mut *demands, &owners, e);
                        }
                    }
                }
            }
        }));
        if outcome.is_err() {
            // Contained panic: clear the capture slot, as the
            // other demand passes do.
            let _ = take_captured_panic();
        }
    }

    extra_seeds
}

/// Record the demand outcomes accumulated by
/// [`collect_local_instances`] in the report: a deferred
/// definition is converted when every instance demanded of it
/// converted; when some instance could not be converted it stays
/// skipped with that instance's reasons. A definition nothing
/// demands is accounted [`FnOutcome::NotDemanded`]: no converted
/// body reaches it, so it is not part of the program — rustc's
/// monomorphization collector never instantiates it either —
/// and its lack of an instance does not make the conversion
/// incomplete. The record itself stays, under the item loop's
/// original no-instance reason; it is never counted converted.
/// One exception: a *constructor* nothing demands is accounted
/// converted without a body — its direct uses are aggregate
/// rvalues in the callers (already converted with them), and the
/// constructor item has no body in any monomorphization that was
/// not demanded, so there is nothing left unconverted (the same
/// accounting as a trivial constant).
pub(crate) fn apply_instance_demands(
    report: &mut ConversionReport,
    deferred: &[DeferredFn],
    demands: &HashMap<rustc_public::DefId, Demand>,
) {
    for dfn in deferred {
        let Some(demand) = demands.get(&dfn.def) else {
            let entry = &mut report.functions[dfn.report_idx];
            if dfn.ctor {
                entry.outcome = FnOutcome::Converted;
                report.bodyless_converted += 1;
            } else {
                let reason = match &entry.outcome {
                    FnOutcome::Skipped { reasons } if reasons.len() == 1 => {
                        reasons.keys().next().expect("one reason").clone()
                    }
                    _ => "no monomorphic instance available".to_string(),
                };
                entry.outcome = FnOutcome::NotDemanded { reason };
            }
            continue;
        };
        let entry = &mut report.functions[dfn.report_idx];
        if demand.converted > 0 && demand.failures.is_empty() {
            let (blocks, locals, arg_count) = demand.first_counts.unwrap_or((0, 0, 0));
            entry.blocks = blocks;
            entry.locals = locals;
            entry.arg_count = arg_count;
            entry.outcome = FnOutcome::Converted;
        } else {
            entry.outcome = FnOutcome::Skipped {
                reasons: demand.failures.clone(),
            };
        }
    }
}

/// Record the demand outcomes accumulated for closures defined
/// in other crates in the report: one entry per closure
/// definition, with the accounting [`apply_instance_demands`]
/// gives a deferred local definition — converted when every
/// demanded instance converted, skipped with the accumulated
/// reasons otherwise. A foreign closure has no item-loop
/// placeholder entry (no item loop iterates foreign items), so
/// the entries are appended here, past the target-crate section
/// of the report, like the reachability collection's entries.
pub(crate) fn apply_foreign_closure_demands(
    report: &mut ConversionReport,
    demands: &HashMap<rustc_public::DefId, Demand>,
) {
    let mut defs: Vec<&rustc_public::DefId> = demands.keys().collect();
    defs.sort_by_key(|d| d.name());
    for def in defs {
        let demand = &demands[def];
        let converted = demand.converted > 0 && demand.failures.is_empty();
        let (blocks, locals, arg_count) = if converted {
            demand.first_counts.unwrap_or((0, 0, 0))
        } else {
            (0, 0, 0)
        };
        let outcome = if converted {
            FnOutcome::Converted
        } else {
            FnOutcome::Skipped {
                reasons: demand.failures.clone(),
            }
        };
        report.functions.push(FnReport {
            name: def.name(),
            blocks,
            locals,
            arg_count,
            outcome,
        });
    }
}

/// Register converted impl method bodies under the trait keys
/// their call sites resolve through (Issue #64).
///
/// A call to a trait method names the trait declaration in its
/// `FnDef` target, so body conversion keys the call by
/// `(trait definition, resolved instance)`. The item loop instead
/// converts an impl method as an item of its own and keys it by
/// `(impl definition, instance)`. The two keys never meet, so a
/// trait call failed with "no converted body" even though the impl
/// body was in the program. (Generic impl methods escape this: the
/// demanded-instance collection keys them by the referenced
/// definition already, and the reachability collection does the
/// same for foreign bodies. The item loop is the only registrar
/// that keys by the resolved definition.)
///
/// This pass closes the gap the same way the rest of the driver
/// keys bodies — by the referenced definition: it scans the
/// converted bodies' Stable MIR call targets, and whenever a
/// target resolves to an instance whose body is present under the
/// resolved definition's key but not under the referenced
/// definition's key, it registers a copy of that body under the
/// referenced key. One pass suffices: the copy's own call targets
/// are the original's, already keyed the same way. Aliases carry
/// no report entries — the impl method's own entry already
/// accounts for the body; the alias is the same body under the
/// key call sites use.
/// Whether `name` (a definition name as the driver's name
/// table renders it) is one of `core::fmt`'s formatting trait
/// methods — the methods `core::fmt::rt::Argument`'s
/// constructors store as the formatting function.
fn is_fmt_trait_method(name: &str) -> bool {
    let Some(rest) = name
        .strip_prefix("std::fmt::")
        .or_else(|| name.strip_prefix("core::fmt::"))
    else {
        return false;
    };
    matches!(
        rest,
        "Display::fmt"
            | "Debug::fmt"
            | "Octal::fmt"
            | "LowerHex::fmt"
            | "UpperHex::fmt"
            | "Binary::fmt"
            | "LowerExp::fmt"
            | "UpperExp::fmt"
            | "Pointer::fmt"
    )
}

pub(crate) fn register_trait_key_aliases(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    sources: &[rustc_public::mir::Body],
) {
    use std::collections::{HashMap, HashSet};

    let mut present: HashMap<sparagmos_mir::DefId, usize> = body_def_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    let mut used_names: HashSet<String> = bodies.iter().map(|(n, _)| n.clone()).collect();

    for source in sources {
        // Instance resolution goes through the compiler's
        // internals and can panic on exotic targets; isolate each
        // scanned body exactly as the instance collection does, so
        // aliasing can never make a crate convert worse.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut additions: Vec<(String, Body, sparagmos_mir::DefId, String)> = Vec::new();
            let mut pending: HashSet<sparagmos_mir::DefId> = HashSet::new();
            for (def, args) in sparagmos_driver::reach::call_targets(source) {
                let referenced = def.def_id();
                let Some(inst) = sparagmos_driver::convert::resolve_instance_guarded(def, &args) else {
                    continue;
                };
                if matches!(inst.kind, rustc_public::mir::mono::InstanceKind::Virtual { .. }) {
                    continue;
                }
                // A call through an `Fn`-family trait whose Self
                // type is a closure resolves to the closure
                // instance itself. Its body must NOT be aliased
                // under the trait key: the call passes the
                // arguments as one RustCall tuple, while the
                // closure body takes them spread, and the
                // interpreter's trait-call path performs that
                // adaptation when it invokes the closure value.
                // A direct call to an alias copy would hand the
                // body the unspread tuple.
                if matches!(inst.ty().kind(), TyKind::RigidTy(RigidTy::Closure(..))) {
                    continue;
                }
                let resolved = inst.def.def_id();
                if resolved == referenced {
                    continue;
                }
                let call_key = conv.instance_def_id(referenced, &inst);
                if present.contains_key(&call_key) || !pending.insert(call_key) {
                    continue;
                }
                let impl_key = conv.instance_def_id(resolved, &inst);
                if impl_key == call_key {
                    continue;
                }
                let Some(&idx) = present.get(&impl_key) else {
                    continue;
                };
                let ref_name = def.name();
                let display = if used_names.insert(ref_name.clone()) {
                    ref_name.clone()
                } else {
                    let suffixed = format!("{} [{}]", ref_name, inst.name());
                    used_names.insert(suffixed.clone());
                    suffixed
                };
                additions.push((display, bodies[idx].1.clone(), call_key, ref_name));
            }
            // `core::fmt`'s argument constructors materialize
            // the formatting function as an `FnDef` constant
            // naming the trait method, never as a call target,
            // so the loop above never sees the demand. The
            // interpreter's fmt engine addresses the impl body
            // by its instance name (the name the item loop
            // records non-generic impls under), so register
            // that alias as well: the body may be keyed by the
            // referenced definition (the instance-demand pass
            // and the reachability collection) or by the
            // resolved one (the item loop).
            for (def, args) in sparagmos_driver::reach::fndef_constants(source) {
                let referenced_name = def.def_id().name();
                if !is_fmt_trait_method(&referenced_name) {
                    continue;
                }
                let Some(inst) = sparagmos_driver::convert::resolve_instance_guarded(def, &args) else {
                    continue;
                };
                if matches!(
                    inst.kind,
                    rustc_public::mir::mono::InstanceKind::Virtual { .. }
                ) {
                    continue;
                }
                let instance_name = inst.name().to_string();
                if used_names.contains(&instance_name) {
                    continue;
                }
                let call_key = conv.instance_def_id(def.def_id(), &inst);
                let impl_key = conv.instance_def_id(inst.def.def_id(), &inst);
                let Some(&idx) = present
                    .get(&call_key)
                    .or_else(|| present.get(&impl_key))
                else {
                    continue;
                };
                used_names.insert(instance_name.clone());
                additions.push((
                    instance_name,
                    bodies[idx].1.clone(),
                    call_key,
                    referenced_name,
                ));
            }
            additions
        }));
        match outcome {
            Ok(additions) => {
                for (display, body, call_key, ref_name) in additions {
                    present.insert(call_key, bodies.len());
                    bodies.push((display, body));
                    body_def_ids.push(call_key);
                    body_def_names.push(ref_name);
                }
            }
            Err(_) => {
                // Contained panic: clear the capture slot, as the
                // instance collection does — a later, genuinely
                // fatal panic must report its own message.
                let _ = take_captured_panic();
            }
        }
    }
}
