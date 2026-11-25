//! Reachability collection of non-local function bodies.
//!
//! The driver converts the local crate's items; calls from those
//! bodies into library functions (std / core / alloc, trait impls in
//! other crates, ...) otherwise have no body in the resulting
//! program, so the interpreter can only run them when it models them
//! as builtins. This module walks the call graph of the converted
//! local bodies on the Stable MIR side, resolves every `FnDef` call
//! target to a monomorphic instance, and fetches the bodies Stable
//! MIR can provide, so they are converted and included in the
//! program like local functions.
//!
//! What can and cannot be obtained (measured against Rust 1.100.0):
//! - Generic functions and `#[inline]` / const functions of other
//!   crates: a body is available, eagerly monomorphized with all
//!   constants evaluated (`Vec::push`, `Option::unwrap`,
//!   `core::panicking::assert_failed`, trait impl methods, ...).
//! - Non-generic, non-inline functions of other crates: no body is
//!   shipped in the crate metadata, so they stay body-less
//!   (`core::panicking::panic`, `std::rt::panic_fmt`, the
//!   `__rust_alloc*` allocation symbols, `str::to_uppercase`, ...).
//! - Compiler intrinsics: no body, except the few that carry a
//!   fallback body (e.g. `integer_max`), which is collected like any
//!   other body.
//! - Dynamic dispatch has no `FnDef` call target at all and is not
//!   affected by this collection.
//!
//! Boundaries:
//! - Functions the interpreter implements as builtins
//!   (`sparagmos_interp::builtins`) are not collected: the runtime's
//!   modelled implementation owns those definition names, and the
//!   interpreter prefers converted bodies over builtins, so a
//!   collected body would shadow the model. The exclusion is per
//!   definition, matching the builtin registry's granularity.
//! - Targets whose resolved instance belongs to the local crate are
//!   the local item loop's business and are skipped here.
//! - One function's conversion failure never stops the others: each
//!   collected body is converted on its own and failures are
//!   recorded per function in the report.
//! - Collection is capped at [`MAX_COLLECTED_BODIES`] bodies per
//!   program; the caller passes the remaining budget to each pass
//!   (the driver runs further passes over bodies the first pass
//!   unlocked), and targets past the cap are recorded with that
//!   reason instead of being collected, so the cutoff is visible
//!   in the report.
//! - The order bodies are scanned in is the pass's
//!   [`CollectionOrder`]: convert mode walks first-in-first-out,
//!   while the executing modes walk execution-first (demand
//!   distance from the entry, call edges before value mentions),
//!   so when the cap binds the bodies cut are the weakest-demand
//!   periphery rather than whatever the walk reached last.
//! - `FnDef` constants a body materializes as *values* are demand
//!   targets on equal footing with call targets (Lane CP): an
//!   external constructor or function used as a value (`let f =
//!   Some;`) gets its instance collected the same way, subject to
//!   the same builtin exclusion and cap.
//! - The Stable MIR bodies obtained during the walk are returned
//!   alongside the converted bodies ([`CollectedFns::walked_bodies`]):
//!   targets resolving to local items are skipped here, but a
//!   collected foreign body can be the only place that demands a
//!   monomorphic instance of a generic local definition, and the
//!   caller's local-instance collection scans these bodies for
//!   exactly those demands.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::mir::{self, AggregateKind, Operand, Rvalue, StatementKind, TerminatorKind};
use rustc_public::ty::{ClosureDef, GenericArgKind, GenericArgs, RigidTy, Ty, TyKind};
use rustc_public::CrateDef;

use sparagmos_mir as sm;

use crate::convert::{
    args_mention_params, closure_kind, guarded_ty_kind, stable_mir_panic_reason, Converter,
};
use crate::report::{FnOutcome, FnReport};

/// Cap on collected bodies per program; see the module docs.
pub const MAX_COLLECTED_BODIES: usize = 8192;

/// Cap on the expanded size (node count) of one demanded
/// instance's generic arguments, where the size of a type is the
/// number of nodes in its full expansion: a tuple `(T, T)` counts
/// one plus twice the size of `T`, an ADT counts one plus the
/// sizes of its generic arguments, a reference counts one plus
/// the size of its pointee, and a leaf type counts one.
///
/// rustc interns types, so its recursion limit — a bound on how
/// deep one definition's instantiations nest along a demand
/// path — is all it needs: each nesting level costs it O(1) new
/// type nodes. This driver instead materializes types as trees
/// when it converts a body, so a size-increasing recursion (each
/// instantiation of a definition demanding the next with a
/// doubled type, as in `fn recurse(&self) { (self,
/// self).recurse() }`) costs memory exponential in the nesting
/// depth: the depth and instance-count caps are reached only
/// after the process has allocated more than any machine has.
/// The expanded-size cap is the bound that actually binds for
/// such programs, and tripping it is reported through the same
/// recursion-limit channel as the depth cap, since the
/// situation is exactly rustc's "reached the recursion limit
/// while instantiating". The cap value is set by conversion
/// cost, not by what a type can theoretically reach: converting
/// a type re-resolves the closure and function instances among
/// its nodes, so conversion time grows superlinearly in the
/// expanded size. At 32768 nodes a worst-case size-increasing
/// chain still finishes in seconds with its memory bounded,
/// while legitimate local programs stay far below it (no test
/// in the rustc suite demands a local instance past a few
/// thousand nodes).
///
/// The cap guards the LOCAL collection passes in the driver
/// only. Foreign demands are deliberately not capped: their
/// argument spellings grow by substitution composition even
/// for a finite instance set — collecting
/// tests/ui/iterators/iter-map-fold-type-length.rs demands
/// `Iterator::fold` at expansions up to 540,651 nodes for a
/// twelve-deep `Map` tower, and the base driver completes it —
/// so a per-instance cap there refuses legitimate collection.
/// The unbounded growth this cap exists to stop is driven by
/// local instantiation, and a local size failure aborts the
/// conversion before foreign bodies are converted.
pub const MAX_INSTANTIATION_EXPANDED_SIZE: usize = 1 << 15;

/// Whether the expanded size of `args` (see
/// [`MAX_INSTANTIATION_EXPANDED_SIZE`]) stays within `limit`.
/// The walk counts nodes against a budget and stops as soon as
/// the budget is spent, so deciding the question costs
/// O(min(expanded size, limit)): the walk itself can never be
/// the runaway allocation it guards against.
pub fn generic_args_within_expanded_size(args: &GenericArgs, limit: usize) -> bool {
    struct Walk {
        budget: usize,
    }
    impl Walk {
        fn node(&mut self) -> bool {
            if self.budget == 0 {
                return false;
            }
            self.budget -= 1;
            true
        }
        fn args(&mut self, args: &GenericArgs) -> bool {
            args.0.iter().all(|arg| self.arg(arg))
        }
        fn arg(&mut self, arg: &GenericArgKind) -> bool {
            match arg {
                GenericArgKind::Type(ty) => self.ty(ty),
                GenericArgKind::Const(konst) => match konst.kind() {
                    rustc_public::ty::TyConstKind::Unevaluated(_, inner) => {
                        self.node() && self.args(&inner)
                    }
                    rustc_public::ty::TyConstKind::Value(ty, _)
                    | rustc_public::ty::TyConstKind::ZSTValue(ty) => self.ty(&ty),
                    _ => self.node(),
                },
                GenericArgKind::Lifetime(_) => true,
            }
        }
        fn ty(&mut self, ty: &Ty) -> bool {
            if let Some(size) = EXPANDED_SIZE_CACHE.with(|cache| cache.borrow().get(ty).copied())
            {
                if size > self.budget {
                    return false;
                }
                self.budget -= size;
                return true;
            }
            let budget_before = self.budget;
            if !self.node() {
                return false;
            }
            // A type passing through a coroutine-closure type
            // cannot be asked its kind (see `guarded_ty_kind`);
            // it counts as one node, like any other leaf.
            let Some(kind) = guarded_ty_kind(ty) else {
                EXPANDED_SIZE_CACHE.with(|cache| {
                    cache.borrow_mut().insert(*ty, 1);
                });
                return true;
            };
            let within = match kind {
                TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                    elems.iter().all(|elem| self.ty(elem))
                }
                TyKind::RigidTy(RigidTy::Ref(_, inner, _))
                | TyKind::RigidTy(RigidTy::RawPtr(inner, _))
                | TyKind::RigidTy(RigidTy::Slice(inner))
                | TyKind::RigidTy(RigidTy::Pat(inner, _)) => self.ty(&inner),
                TyKind::RigidTy(RigidTy::Array(inner, _)) => self.ty(&inner),
                TyKind::RigidTy(RigidTy::FnPtr(sig)) => {
                    sig.value.inputs_and_output.iter().all(|t| self.ty(t))
                }
                TyKind::RigidTy(RigidTy::Adt(_, args))
                | TyKind::RigidTy(RigidTy::FnDef(_, args))
                | TyKind::RigidTy(RigidTy::Closure(_, args))
                | TyKind::RigidTy(RigidTy::Coroutine(_, args))
                | TyKind::RigidTy(RigidTy::CoroutineClosure(_, args))
                | TyKind::RigidTy(RigidTy::CoroutineWitness(_, args)) => self.args(&args),
                TyKind::RigidTy(RigidTy::Dynamic(bounds, _)) => bounds.iter().all(|bound| {
                    match &bound.value {
                        rustc_public::ty::ExistentialPredicate::Trait(t) => {
                            self.args(&t.generic_args)
                        }
                        rustc_public::ty::ExistentialPredicate::Projection(p) => {
                            self.args(&p.generic_args)
                        }
                        rustc_public::ty::ExistentialPredicate::AutoTrait(_) => true,
                    }
                }),
                TyKind::Alias(_, alias) => self.args(&alias.args),
                _ => true,
            };
            if within {
                let size = budget_before - self.budget;
                EXPANDED_SIZE_CACHE.with(|cache| {
                    cache.borrow_mut().insert(*ty, size);
                });
            }
            within
        }
    }
    Walk { budget: limit }.args(args)
}

thread_local! {
    /// Exact expanded sizes of types already measured by
    /// [`generic_args_within_expanded_size`], keyed by the
    /// type handle and shared by every check of one
    /// compilation. The argument graph is a DAG (an interned
    /// type cannot be a proper subterm of itself), and a
    /// deeply nested instance shares its subterms across
    /// every occurrence and every demand pass: re-expanding
    /// them per check makes the guard cost the full expanded
    /// size per demanded instance, which dwarfs the
    /// conversion it guards. Only exact sizes are stored: a
    /// type whose measurement ran out of budget is not
    /// memoized (its size is only known to exceed the budget
    /// it was measured against). Cleared whenever a fresh
    /// `Converter` starts a compilation (see
    /// [`clear_expanded_size_cache`]), so handles from an
    /// earlier compilation in the same process are never
    /// read back.
    static EXPANDED_SIZE_CACHE: std::cell::RefCell<std::collections::HashMap<Ty, usize>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Drop every memoized expanded size (see
/// [`EXPANDED_SIZE_CACHE`]). Called when a new compilation
/// starts, because type handles are only meaningful within
/// the compilation that interned them.
pub fn clear_expanded_size_cache() {
    EXPANDED_SIZE_CACHE.with(|cache| cache.borrow_mut().clear());
}


/// The outcome of one reachability pass: the converted foreign
/// bodies (in discovery order) and a report entry for every
/// distinct foreign target that was considered.
pub struct CollectedFns {
    /// `(display name, body)` for each successfully converted body.
    pub bodies: Vec<(String, sm::Body)>,
    /// Sparagmos `DefId` of each entry of `bodies` (same order).
    pub body_def_ids: Vec<sm::DefId>,
    /// Definition name of each entry of `bodies` (same order); the
    /// display name is the definition name for the first converted
    /// instance of a definition and carries the instance name as a
    /// suffix for further instances, while this stays the plain
    /// definition name.
    pub body_def_names: Vec<String>,
    /// One report entry per distinct foreign target considered.
    pub reports: Vec<FnReport>,
    /// The Stable MIR bodies of the successfully converted foreign
    /// functions, parallel to `bodies`. Callers that need to walk
    /// the foreign call graph once more (the trait-key alias pass
    /// in the driver) use these; the converted bodies alone no
    /// longer carry the `(referenced definition, instance)` pairs
    /// their call targets were resolved through.
    pub stable_bodies: Vec<mir::Body>,
    /// Whether the collection cap stopped further collection.
    pub limit_reached: bool,
    /// The Stable MIR bodies obtained from foreign instances
    /// during the walk, in discovery order (the seeds are not
    /// included). A body is listed whether or not its conversion
    /// succeeded: its call targets were walked either way.
    pub walked_bodies: Vec<mir::Body>,
}

enum RecordKind {
    /// A Stable MIR body was obtained; conversion is attempted
    /// after the walk.
    WithBody { inst: Instance, body: mir::Body },
    /// No Stable MIR body exists for this target, but the target
    /// is a data constructor and its body — a one-block aggregate
    /// of the arguments, the same shape the compiler synthesizes
    /// for a local constructor — was synthesized directly in
    /// sparagmos-mir form by [`synthetic_ctor_body`].
    SyntheticCtor { inst: Instance, body: sm::Body },
    /// No body is included for this target; `reason` says why.
    NoBody { reason: String },
}

struct Record {
    /// The `FnDef` definition the target was reached through.
    def: rustc_public::DefId,
    def_name: String,
    /// Detail used to disambiguate instances of one definition in
    /// display names (the resolved instance's name, or the generic
    /// arguments when resolution failed).
    detail: String,
    kind: RecordKind,
}

/// Fetch one instance's body, containing the panics of
/// rustc_public 1.100.0's known unimplemented conversions
/// whose payloads identify them (`stable_mir_panic_reason`):
/// materializing a foreign body can hit the bridge's
/// unreachable instance-constant evaluation, among others.
/// A contained panic is reported as the named reason (the
/// caller records the instance as body-less with it); any
/// other panic is resumed — this containment must never paper
/// over an unrelated compiler failure.
fn fetch_body_contained(
    inst: &Instance,
) -> Result<Option<mir::Body>, &'static str> {
    // An instance whose types mention a projection that
    // does not normalize: fetching its body would pose that
    // normalization and poison the compilation (see
    // `crate::convert::unnormalizable_projection_in_instance`).
    // Record it body-less with the screen's reason instead.
    if crate::convert::unnormalizable_projection_in_instance(inst).is_some() {
        return Err(crate::convert::UNNORMALIZABLE_PROJECTION_REASON);
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::convert::with_body_materialization(&inst.name(), || inst.body())
    })) {
        Ok(body) => Ok(body),
        Err(payload) => {
            if let Some(reason) = stable_mir_panic_reason(&*payload) {
                Err(reason)
            } else {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

/// How one [`collect_reachable`] pass orders the bodies it
/// scans. The order decides both the order bodies are recorded
/// (and so converted) in and — when the collection cap binds —
/// which bodies the cap cuts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CollectionOrder {
    /// First-in-first-out breadth-first order: entry seeds,
    /// then seeds, then bodies in the order their demand was
    /// first recorded. This is the historical behaviour, and
    /// convert mode always uses it: its conversion report is a
    /// published measurement contract (the demand accounting
    /// included), and it must not change.
    Fifo,
    /// Execution-first order, for the executing modes: bodies
    /// are scanned by demand distance from the entry seeds
    /// (the entry body itself first, the remaining seeds one
    /// tier below it), and at equal distance a body demanded
    /// through a call edge is scanned before one demanded only
    /// as a value (an `FnDef` constant). The set collected when
    /// the cap does not bind is the same as
    /// [`CollectionOrder::Fifo`]; when the cap binds, the
    /// bodies cut are the weakest-demand periphery instead of
    /// whatever the breadth-first walk happened to reach last,
    /// so the executable core is complete before speculative
    /// periphery is collected.
    ExecutionFirst,
}

/// The strength of the edge that demanded a body. A call edge
/// is stronger than a value mention: a body materializes an
/// `FnDef` constant for many reasons (a comparison, a function
/// table entry) that never call it, while a call target runs
/// whenever its caller runs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Demand {
    Call = 0,
    Value = 1,
}

/// One queued body in execution-first order.
struct FrontierEntry<B> {
    distance: u32,
    demand: Demand,
    /// Insertion sequence; makes the order total and stable
    /// (equal keys pop in insertion order, as the FIFO does).
    seq: u64,
    body: B,
}

impl<B> PartialEq for FrontierEntry<B> {
    fn eq(&self, other: &Self) -> bool {
        (self.distance, self.demand, self.seq)
            == (other.distance, other.demand, other.seq)
    }
}

impl<B> Eq for FrontierEntry<B> {}

impl<B> PartialOrd for FrontierEntry<B> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<B> Ord for FrontierEntry<B> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `BinaryHeap` pops the greatest element; the walk
        // wants the smallest (distance, demand, seq) first.
        (other.distance, other.demand, other.seq)
            .cmp(&(self.distance, self.demand, self.seq))
    }
}

/// The walk frontier of one [`collect_reachable`] pass: the
/// bodies still to scan, popped in the pass's
/// [`CollectionOrder`].
struct Frontier<B> {
    order: CollectionOrder,
    fifo: VecDeque<B>,
    heap: BinaryHeap<FrontierEntry<B>>,
    seq: u64,
}

impl<B> Frontier<B> {
    fn new(order: CollectionOrder) -> Frontier<B> {
        Frontier {
            order,
            fifo: VecDeque::new(),
            heap: BinaryHeap::new(),
            seq: 0,
        }
    }

    fn push(&mut self, distance: u32, demand: Demand, body: B) {
        match self.order {
            CollectionOrder::Fifo => self.fifo.push_back(body),
            CollectionOrder::ExecutionFirst => {
                let seq = self.seq;
                self.seq += 1;
                self.heap.push(FrontierEntry {
                    distance,
                    demand,
                    seq,
                    body,
                });
            }
        }
    }

    /// Pop the next body to scan, with the demand distance it
    /// was pushed at. The distance is meaningful only in
    /// execution-first order; the FIFO does not track it and
    /// this returns 0 there.
    fn pop(&mut self) -> Option<(u32, B)> {
        match self.order {
            CollectionOrder::Fifo => self.fifo.pop_front().map(|body| (0, body)),
            CollectionOrder::ExecutionFirst => self
                .heap
                .pop()
                .map(|entry| (entry.distance, entry.body)),
        }
    }
}

/// Collect the bodies of non-local functions reachable from the
/// converted local bodies (`seeds`, Stable MIR form). `local_defs`
/// holds the `DefId`s of the local crate's items; targets resolving
/// to one of them are skipped (they are collected as local items).
/// At most `cap` bodies are collected in this pass; the caller
/// passes the program's remaining budget ([`MAX_COLLECTED_BODIES`]
/// minus what earlier passes collected).
///
/// `order` is the pass's [`CollectionOrder`]. In
/// execution-first order the caller also passes the entry
/// point's body as `entry_seeds` (empty when the crate has no
/// entry point, and always empty in FIFO order's callers): it
/// is scanned ahead of `seeds`, and the remaining seeds sit
/// one demand tier below it.
pub fn collect_reachable(
    conv: &mut Converter,
    seeds: &[mir::Body],
    local_defs: &HashSet<rustc_public::DefId>,
    cap: usize,
    order: CollectionOrder,
    entry_seeds: &[mir::Body],
) -> CollectedFns {
    let mut records: Vec<Record> = Vec::new();
    let mut by_instance: HashMap<(rustc_public::DefId, Instance), ()> = HashMap::new();
    let mut unresolved_seen: HashSet<(rustc_public::DefId, String)> = HashSet::new();
    let mut builtin_seen: HashSet<rustc_public::DefId> = HashSet::new();
    let mut frontier = Frontier::new(order);
    for body in entry_seeds {
        frontier.push(0, Demand::Call, body.clone());
    }
    // With entry seeds, the remaining seeds sit one demand
    // tier below the entry: their demands must not crowd out
    // the entry's own closure under the cap. Without entry
    // seeds they are the walk's roots, at distance 0.
    let seed_distance = if entry_seeds.is_empty() { 0 } else { 1 };
    for body in seeds {
        frontier.push(seed_distance, Demand::Call, body.clone());
    }
    let mut bodies_taken = 0usize;
    let mut limit_reached = false;
    let mut walked_bodies: Vec<mir::Body> = Vec::new();

    while let Some((distance, body)) = frontier.pop() {
        // Demand targets of this body: its `FnDef` call targets,
        // plus every `FnDef` constant it materializes. A foreign
        // function or constructor used *as a value* (assigned to a
        // local, passed to a higher-order function, returned) never
        // appears as a call target of the body that demands it, so
        // call targets alone miss the demand — `let f = Some; f(3)`
        // is the common case. The per-target pipeline below is
        // unchanged: builtin-owned definitions stay excluded,
        // local targets stay the item loop's business, instances
        // are deduplicated, and the shared cap applies.
        let mut targets: Vec<(rustc_public::ty::FnDef, GenericArgs, Demand)> =
            call_targets(&body)
                .into_iter()
                .map(|(def, args)| (def, args, Demand::Call))
                .collect();
        targets.extend(
            fndef_constants(&body)
                .into_iter()
                .map(|(def, args)| (def, args, Demand::Value)),
        );
        for (def, args, demand) in targets {
            let def_id = def.def_id();
            let def_name = def_id.name();
            // Builtin-owned definitions are not collected (see the
            // module docs). The registry is keyed by canonical
            // definition name, so this check is per definition.
            let canonical = sparagmos_interp::builtins::canonical_name(&def_name);
            if sparagmos_interp::builtins::lookup_qualified(&canonical).is_some()
                && !consumer_instance_needs_body(&def, &canonical, &args)
                && !partial_eq_ref_instance_needs_body(&canonical, &args)
            {
                if builtin_seen.insert(def_id) {
                    records.push(Record {
                        def: def_id,
                        def_name: def_name.clone(),
                        detail: def_name.clone(),
                        kind: RecordKind::NoBody {
                            reason: "implemented as an interpreter builtin; body not collected"
                                .to_string(),
                        },
                    });
                }
                // `core::fmt`'s argument constructors are builtins,
                // but the formatter function each one stores — and,
                // through that function's body, the `fmt` method of
                // the type being formatted — is real code whose
                // body the interpreter needs: formatting runs the
                // type's own `Debug`/`Display` implementation
                // against a captured `Formatter`. Walk the
                // constructor's body (never collected: the builtin
                // serves the call) so the formatter function it
                // names is demanded through the normal pipeline
                // below, and the trait method through the formatter
                // function's own body in turn.
                if is_fmt_argument_constructor(&canonical) {
                    if let Some(inst) = crate::convert::resolve_instance_guarded(def, &args) {
                        if let Ok(Some(ctor_body)) = fetch_body_contained(&inst) {
                            // The walk queue demands the formatter
                            // function through the pipeline above;
                            // `walked_bodies` is what the caller's
                            // local-instance pass scans, and the
                            // trait method the formatter names is a
                            // local instance when the type is local.
                            frontier.push(distance + 1, demand, ctor_body.clone());
                            walked_bodies.push(ctor_body);
                        }
                    }
                }
                // The excluded body is also where the demand for
                // the closures handed to it would have come from:
                // a generic consumer's body calls its closure
                // through an Fn-trait shim, and collecting that
                // call is what used to bring the closure's body
                // into the program (the interpreter's consumers —
                // `Iterator::fold` and friends — call the closure
                // value directly). Create the same demand from
                // the call site instead: the concrete closure
                // types among the call's generic arguments get
                // their bodies collected here. Local closures are
                // the closure-demand pass's business, and mentions
                // still mentioning parameters are left to the
                // monomorphic instance bodies.
                for (cdef, cargs) in closures_in_args(&args) {
                    let cdef_id = cdef.def_id();
                    if local_defs.contains(&cdef_id) || args_mention_params(&cargs) {
                        continue;
                    }
                    let Some(kind) = closure_kind(&cargs) else {
                        continue;
                    };
                    let inst = match crate::convert::resolve_closure_guarded(cdef, &cargs, kind) {
                        Some(inst) => inst,
                        None => {
                            let key = (cdef_id, format!("{cargs:?}"));
                            if unresolved_seen.insert(key) {
                                records.push(Record {
                                    def: cdef_id,
                                    def_name: cdef_id.name(),
                                    detail: truncate(&format!("{cargs:?}"), 100),
                                    kind: RecordKind::NoBody {
                                        reason: "closure instance resolution failed"
                                            .to_string(),
                                    },
                                });
                            }
                            continue;
                        }
                    };
                    if by_instance.contains_key(&(cdef_id, inst)) {
                        continue;
                    }
                    by_instance.insert((cdef_id, inst), ());
                    let detail = inst.name().to_string();
                    let kind = if !inst.has_body() {
                        RecordKind::NoBody {
                            reason: "no MIR body available".to_string(),
                        }
                    } else if bodies_taken >= cap {
                        limit_reached = true;
                        RecordKind::NoBody {
                            reason: "collection limit reached".to_string(),
                        }
                    } else if let Some(reason) =
                        crate::convert::unnormalizable_projection_in_instance(
                            &inst,
                        )
                    {
                        RecordKind::NoBody { reason }
                    } else if let Some(cbody) = crate::convert::with_body_materialization(
                        &inst.name(),
                        || inst.body(),
                    ) {
                        bodies_taken += 1;
                        frontier.push(distance + 1, demand, cbody.clone());
                        walked_bodies.push(cbody.clone());
                        RecordKind::WithBody { inst, body: cbody }
                    } else {
                        RecordKind::NoBody {
                            reason: "no MIR body available".to_string(),
                        }
                    };
                    records.push(Record {
                        def: cdef_id,
                        def_name: cdef_id.name(),
                        detail,
                        kind,
                    });
                }
                continue;
            }
            let inst = match crate::convert::resolve_instance_guarded(def, &args) {
                Some(inst) => inst,
                None => {
                    let key = (def_id, format!("{args:?}"));
                    if unresolved_seen.insert(key) {
                        records.push(Record {
                            def: def_id,
                            detail: truncate(&format!("{args:?}"), 100),
                            def_name,
                            kind: RecordKind::NoBody {
                                reason: "instance resolution failed".to_string(),
                            },
                        });
                    }
                    continue;
                }
            };
            // A target that resolves to a local-crate item is
            // collected by the local item loop, not here.
            if local_defs.contains(&inst.def.def_id()) {
                continue;
            }
            if by_instance.contains_key(&(def_id, inst)) {
                continue;
            }
            by_instance.insert((def_id, inst), ());
            let detail = inst.name().to_string();
            let no_body = |reason: &str| RecordKind::NoBody {
                reason: reason.to_string(),
            };
            let kind = match inst.kind {
                InstanceKind::Virtual { .. } => {
                    no_body("dynamic dispatch target (virtual instance)")
                }
                _ if !inst.has_body() => match inst.kind {
                    InstanceKind::Intrinsic => {
                        conv.note_intrinsic(&inst);
                        // A compiler intrinsic the interpreter
                        // models is served by the builtin, like
                        // the builtin-owned definitions excluded
                        // above; the model is looked up by the
                        // intrinsic's own name (see
                        // `builtins::intrinsic_lookup`), because
                        // an intrinsic declared by another crate
                        // is named by its declaration site, which
                        // the qualified registry never matches.
                        let canonical =
                            sparagmos_interp::builtins::canonical_name(&def_name);
                        if sparagmos_interp::builtins::intrinsic_lookup(&canonical)
                            .is_some()
                        {
                            no_body(
                                "implemented as an interpreter builtin; body not collected",
                            )
                        } else {
                            no_body("compiler intrinsic (no fallback body)")
                        }
                    }
                    _ if bodies_taken >= cap => {
                        limit_reached = true;
                        no_body("collection limit reached")
                    }
                    // A data constructor of another crate has no
                    // MIR body in the crate metadata (the compiler
                    // synthesizes the shim only for local defs);
                    // synthesize the equivalent body directly.
                    _ => match synthetic_ctor_body(conv, &def, &inst) {
                        Some(body) => {
                            bodies_taken += 1;
                            RecordKind::SyntheticCtor { inst, body }
                        }
                        None => no_body("no MIR body available"),
                    },
                },
                _ if bodies_taken >= cap => {
                    limit_reached = true;
                    no_body("collection limit reached")
                }
                _ => match fetch_body_contained(&inst) {
                    Ok(Some(body)) => {
                        bodies_taken += 1;
                        frontier.push(distance + 1, demand, body.clone());
                        walked_bodies.push(body.clone());
                        RecordKind::WithBody { inst, body }
                    }
                    Ok(None) => no_body("no MIR body available"),
                    Err(reason) => no_body(reason),
                },
            };
            records.push(Record {
                def: def_id,
                def_name,
                detail,
                kind,
            });
        }
    }

    // Convert each collected body on its own; a failure is recorded
    // for that function and never stops the others.
    let mut converted: Vec<Option<sm::Body>> = Vec::with_capacity(records.len());
    let mut failed: Vec<Option<std::collections::BTreeMap<String, usize>>> =
        Vec::with_capacity(records.len());
    for rec in &records {
        match &rec.kind {
            RecordKind::SyntheticCtor { body, .. } => {
                match unrepresentable_local(body) {
                    Some(what) => {
                        let mut reasons: std::collections::BTreeMap<String, usize> =
                            Default::default();
                        reasons.insert(
                            format!(
                                "body uses a type the runtime cannot represent ({what})"
                            ),
                            1,
                        );
                        converted.push(None);
                        failed.push(Some(reasons));
                    }
                    None => {
                        converted.push(Some(body.clone()));
                        failed.push(None);
                    }
                }
            }
            RecordKind::WithBody { body, .. } => match conv.convert_body_contained(body) {
                Ok(b) => match unrepresentable_local(&b) {
                    Some(what) => {
                        let mut reasons: std::collections::BTreeMap<String, usize> =
                            Default::default();
                        reasons.insert(
                            format!(
                                "body uses a type the runtime cannot represent ({what})"
                            ),
                            1,
                        );
                        converted.push(None);
                        failed.push(Some(reasons));
                    }
                    None => {
                        converted.push(Some(b));
                        failed.push(None);
                    }
                },
                Err(errs) => {
                    let mut reasons: std::collections::BTreeMap<String, usize> =
                        Default::default();
                    for e in errs {
                        *reasons.entry(e).or_default() += 1;
                    }
                    converted.push(None);
                    failed.push(Some(reasons));
                }
            },
            RecordKind::NoBody { .. } => {
                converted.push(None);
                failed.push(None);
            }
        }
    }

    // Display names: within one definition, the first successfully
    // converted instance carries the plain definition name (the
    // name call sites and coverage tooling use); other instances
    // are suffixed with their instance name. If nothing converted,
    // the first record of the definition carries the plain name.
    let mut plain_owner: HashMap<rustc_public::DefId, usize> = HashMap::new();
    for (i, rec) in records.iter().enumerate() {
        let rank = if converted[i].is_some() {
            0
        } else if matches!(
            rec.kind,
            RecordKind::WithBody { .. } | RecordKind::SyntheticCtor { .. }
        ) {
            1
        } else {
            2
        };
        match plain_owner.get(&rec.def) {
            None => {
                plain_owner.insert(rec.def, i);
            }
            Some(&cur) => {
                let cur_rank = if converted[cur].is_some() {
                    0
                } else if matches!(
                    records[cur].kind,
                    RecordKind::WithBody { .. } | RecordKind::SyntheticCtor { .. }
                ) {
                    1
                } else {
                    2
                };
                if rank < cur_rank {
                    plain_owner.insert(rec.def, i);
                }
            }
        }
    }
    let display_name = |i: usize, rec: &Record| -> String {
        if plain_owner.get(&rec.def) == Some(&i) {
            rec.def_name.clone()
        } else {
            format!("{} [{}]", rec.def_name, rec.detail)
        }
    };

    let mut out = CollectedFns {
        bodies: Vec::new(),
        body_def_ids: Vec::new(),
        body_def_names: Vec::new(),
        reports: Vec::new(),
        stable_bodies: Vec::new(),
        limit_reached,
        walked_bodies,
    };
    for (i, rec) in records.iter().enumerate() {
        let name = display_name(i, rec);
        let (blocks, locals, arg_count) = match &rec.kind {
            RecordKind::WithBody { body, .. } => {
                (body.blocks.len(), body.locals().len(), body.arg_locals().len())
            }
            RecordKind::SyntheticCtor { body, .. } => (
                body.basic_blocks.len(),
                body.local_decls.len(),
                body.arg_count,
            ),
            RecordKind::NoBody { .. } => (0, 0, 0),
        };
        let outcome = match (&rec.kind, &converted[i], &failed[i]) {
            (RecordKind::WithBody { inst, body }, Some(_), _) => {
                out.bodies.push((name.clone(), converted[i].clone().unwrap()));
                out.body_def_ids.push(conv.instance_def_id(rec.def, inst));
                out.body_def_names.push(rec.def_name.clone());
                out.stable_bodies.push(body.clone());
                FnOutcome::Converted
            }
            (RecordKind::SyntheticCtor { inst, .. }, Some(_), _) => {
                out.bodies.push((name.clone(), converted[i].clone().unwrap()));
                out.body_def_ids.push(conv.instance_def_id(rec.def, inst));
                out.body_def_names.push(rec.def_name.clone());
                FnOutcome::Converted
            }
            (RecordKind::WithBody { .. }, None, Some(reasons)) => {
                FnOutcome::Skipped {
                    reasons: reasons.clone(),
                }
            }
            (RecordKind::SyntheticCtor { .. }, None, Some(reasons)) => {
                FnOutcome::Skipped {
                    reasons: reasons.clone(),
                }
            }
            (RecordKind::NoBody { reason }, _, _) => {
                let mut reasons = std::collections::BTreeMap::new();
                reasons.insert(reason.clone(), 1);
                FnOutcome::Skipped { reasons }
            }
            _ => unreachable!("record state mismatch"),
        };
        out.reports.push(FnReport {
            name,
            blocks,
            locals,
            arg_count,
            outcome,
        });
    }
    out
}

/// Synthesize the body of a foreign data constructor (a
/// tuple-struct or tuple-variant constructor of another crate).
///
/// Such a constructor has no MIR body in the crate metadata —
/// `Instance::has_body` is false for it — because the compiler
/// synthesizes the constructor shim only for definitions of the
/// local crate (Lane CF converts those from the synthesized Stable
/// MIR body). The shim's shape is fixed: one block that aggregates
/// the arguments into the ADT and returns it, so the equivalent
/// body is built here directly in sparagmos-mir form.
///
/// Identification is conservative. The target must have an
/// instantiated signature whose output is an ADT (not a union: a
/// union constructor writes a single field, a different aggregate
/// shape), the definition's last path segment must name the
/// variant the constructor builds (the only variant of a struct,
/// or the matching variant of an enum), and the signature's input
/// count must equal that variant's field count. Anything else
/// returns `None` and the target keeps its "no MIR body available"
/// record.
fn synthetic_ctor_body(
    conv: &mut Converter,
    def: &rustc_public::ty::FnDef,
    inst: &Instance,
) -> Option<sm::Body> {
    let sig = guarded_ty_kind(&inst.ty())?.fn_sig()?.skip_binder();
    let (output, inputs) = sig.inputs_and_output.split_last()?;
    let TyKind::RigidTy(RigidTy::Adt(adt_def, adt_args)) =
        guarded_ty_kind(output)? else {
        return None;
    };
    if matches!(adt_def.kind(), rustc_public::ty::AdtKind::Union) {
        return None;
    }
    let def_name = def.def_id().name();
    let last = def_name.rsplit("::").next()?;
    let variants = adt_def.variants();
    let (variant_index, variant) = if variants.len() == 1 {
        (0u32, variants.first()?)
    } else {
        variants
            .iter()
            .enumerate()
            .find(|(_, v)| v.name() == last)
            .map(|(i, v)| (i as u32, v))?
    };
    if variant.name() != last || variant.fields().len() != inputs.len() {
        return None;
    }
    let mut local_decls = vec![sm::LocalDecl::new(conv.ty(output))];
    for input in inputs {
        local_decls.push(sm::LocalDecl::new(conv.ty(input)));
    }
    let generic_args = adt_args
        .0
        .iter()
        .filter_map(|a| match a {
            rustc_public::ty::GenericArgKind::Type(t) => Some(conv.ty(t)),
            _ => None,
        })
        .collect();
    let operands = (1..=inputs.len())
        .map(|i| sm::Operand::Move(sm::Place::from_local(sm::Local(i as u32))))
        .collect();
    let stmt = sm::Statement::new(sm::StatementKind::Assign(Box::new((
        sm::Place::from_local(sm::Local(0)),
        sm::Rvalue::Aggregate(
            Box::new(sm::AggregateKind::Adt {
                def: conv.crate_def_id(&adt_def),
                variant_index,
                generic_args,
                user_ty: None,
                field_index: None,
            }),
            operands,
        ),
    ))));
    let mut block = sm::BasicBlockData::new(sm::Terminator::new(sm::TerminatorKind::Return));
    block.statements.push(stmt);
    Some(sm::Body::new(local_decls, inputs.len(), vec![block]))
}

/// The `FnDef` call targets of one Stable MIR body: every `Call`
/// terminator whose func operand is a constant of `FnDef` type.
/// Calls through places (function pointers, closures) have no
/// `FnDef` target and are not call-graph edges here.
pub fn call_targets(body: &mir::Body) -> Vec<(rustc_public::ty::FnDef, GenericArgs)> {
    let mut out = Vec::new();
    for bb in &body.blocks {
        let TerminatorKind::Call { func, .. } = &bb.terminator.kind else {
            continue;
        };
        let Operand::Constant(c) = func else {
            continue;
        };
        // A constant whose type passes through a coroutine-
        // closure type cannot be asked its kind (see
        // `guarded_ty_kind`); it names no FnDef target that
        // could be extracted here.
        if let Some(TyKind::RigidTy(RigidTy::FnDef(def, args))) =
            guarded_ty_kind(&c.const_.ty())
        {
            out.push((def, args));
        }
    }
    out
}

/// The `FnDef` constants materialized anywhere in one Stable MIR
/// body: every constant operand of `FnDef` type, in statement
/// rvalues and terminator operands alike. Unlike [`call_targets`],
/// this also finds function items used *as values* — assigned to a
/// local, passed as an argument, stored in an aggregate — whose
/// instances a body can demand without ever naming them in a
/// `Call` terminator of its own. (A tuple-struct constructor is
/// the common case: direct construction is an aggregate rvalue,
/// so the constructor item only ever appears as an `FnDef`
/// constant when it is used as a value.)
pub fn fndef_constants(body: &mir::Body) -> Vec<(rustc_public::ty::FnDef, GenericArgs)> {
    fn visit(op: &Operand, out: &mut Vec<(rustc_public::ty::FnDef, GenericArgs)>) {
        let Operand::Constant(c) = op else { return };
        // As in `call_targets`: an unaskable constant type
        // names no extractable FnDef target.
        if let Some(TyKind::RigidTy(RigidTy::FnDef(def, args))) =
            guarded_ty_kind(&c.const_.ty())
        {
            out.push((def, args));
        }
    }
    fn visit_rvalue(rv: &mir::Rvalue, out: &mut Vec<(rustc_public::ty::FnDef, GenericArgs)>) {
        use mir::Rvalue as R;
        match rv {
            R::Use(op, _) | R::Cast(_, op, _) | R::Repeat(op, _) | R::UnaryOp(_, op) => {
                visit(op, out)
            }
            R::BinaryOp(_, l, r) | R::CheckedBinaryOp(_, l, r) => {
                visit(l, out);
                visit(r, out);
            }
            R::Aggregate(_, ops) => {
                for op in ops {
                    visit(op, out);
                }
            }
            // The remaining rvalues read places, not operands.
            _ => {}
        }
    }
    let mut out = Vec::new();
    for bb in &body.blocks {
        for stmt in &bb.statements {
            match &stmt.kind {
                mir::StatementKind::Assign(_, rv) => visit_rvalue(rv, &mut out),
                mir::StatementKind::Intrinsic(intr) => match intr {
                    mir::NonDivergingIntrinsic::Assume(op) => visit(op, &mut out),
                    mir::NonDivergingIntrinsic::CopyNonOverlapping(cno) => {
                        visit(&cno.src, &mut out);
                        visit(&cno.dst, &mut out);
                        visit(&cno.count, &mut out);
                    }
                },
                _ => {}
            }
        }
        match &bb.terminator.kind {
            TerminatorKind::Call { func, args, .. } => {
                visit(func, &mut out);
                for arg in args {
                    visit(arg, &mut out);
                }
            }
            TerminatorKind::Assert { cond, .. } => visit(cond, &mut out),
            TerminatorKind::SwitchInt { discr, .. } => visit(discr, &mut out),
            _ => {}
        }
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s.to_string()
    }
}

/// A converted body is only collected when every local's type has a
/// runtime representation. The interpreter (and the JIT) reject the
/// remaining types — trait objects, closures, coroutines, function
/// items and pointers, `f16` / `f128`, arrays of unknown length —
/// when converting a program, so a body using one could never
/// execute; including it would also fail the conversion of the
/// whole program, even for callers that never reach it. Such
/// functions stay body-less, with the offending type recorded as
/// the reason. ADT types are accepted without inspecting their
/// arguments, matching how the runtime converts them (by name).
pub fn unrepresentable_local(body: &sm::Body) -> Option<String> {
    /// The kind name of the first type inside `ty` that has no
    /// runtime representation, if any.
    fn offending(ty: &sm::Ty) -> Option<String> {
        match ty {
            sm::Ty::Bool
            | sm::Ty::Int(_)
            | sm::Ty::Uint(_)
            | sm::Ty::Char
            | sm::Ty::Never
            | sm::Ty::Str
            // Callable values are representable: the interpreter
            // models function items and closures as values, and a
            // body that merely holds one in a local (the shape of
            // every generic adapter in core, e.g. `Iterator::fold`
            // holding the closure it was given) converts like a
            // local body does — local bodies are not screened at
            // all. The closure/function bodies such a value names
            // are demanded through the call graph as usual.
            | sm::Ty::FnDef(..)
            | sm::Ty::FnPtr(..)
            | sm::Ty::Closure(..)
            | sm::Ty::Dynamic(..)
            // A coroutine type is representable: the interpreter
            // models the state object the coroutine transform
            // produces (its generic arguments - upvars, resume /
            // yield / return types, witness - are not themselves
            // locals and need no representation).
            | sm::Ty::Coroutine(..)
            | sm::Ty::Adt(..) => None,
            sm::Ty::Float(ft) => match ft {
                sm::FloatTy::F32 | sm::FloatTy::F64 | sm::FloatTy::F128 => None,
                _ => Some(ty.kind_name().to_string()),
            },
            sm::Ty::Tuple(ts) => ts.iter().find_map(offending),
            sm::Ty::Ref(_, inner, _) | sm::Ty::RawPtr(inner, _) => offending(inner),
            sm::Ty::Array(elem, len) => {
                if matches!(len, sm::ArrayLen::Known(_)) {
                    offending(elem)
                } else {
                    Some(ty.kind_name().to_string())
                }
            }
            sm::Ty::Slice(elem) => offending(elem),
            _ => Some(ty.kind_name().to_string()),
        }
    }
    body.local_decls
        .iter()
        .enumerate()
        .find_map(|(i, decl)| offending(&decl.ty).map(|kind| format!("{kind} in local _{i}")))
}

/// The closure types one Stable MIR body mentions: every
/// `Closure(def, args)` type reachable from the body's local
/// declarations, its operands' types, and its closure aggregate
/// rvalues, walking nested types (a closure captured by another
/// closure, a closure in a call target's generic arguments, ...).
/// The arguments are the body's own substitution applied — inside
/// a monomorphic instance body they are concrete; inside the
/// generic definition's body they still mention parameters, and
/// the caller decides what that means for each entry.
fn walk_closure_args(args: &GenericArgs, out: &mut Vec<(ClosureDef, GenericArgs)>) {
    for arg in &args.0 {
        match arg {
            GenericArgKind::Type(ty) => walk_closure_ty(ty, out),
            GenericArgKind::Const(konst) => match konst.kind() {
                rustc_public::ty::TyConstKind::Unevaluated(_, inner) => {
                    walk_closure_args(&inner, out)
                }
                rustc_public::ty::TyConstKind::Value(ty, _)
                | rustc_public::ty::TyConstKind::ZSTValue(ty) => walk_closure_ty(&ty, out),
                _ => {}
            },
            GenericArgKind::Lifetime(_) => {}
        }
    }
}

fn walk_closure_ty(ty: &Ty, out: &mut Vec<(ClosureDef, GenericArgs)>) {
    // A type whose conversion tree passes through a coroutine-
    // closure type cannot be asked its kind at all (the bridge
    // leaves that conversion unimplemented); nothing can be
    // extracted from it here. The aggregate that creates an async
    // closure's value carries the demand instead (see
    // `closure_targets`).
    let Some(kind) = guarded_ty_kind(ty) else {
        return;
    };
    match kind {
        TyKind::RigidTy(RigidTy::Closure(def, args)) => {
            out.push((def, args.clone()));
            walk_closure_args(&args, out);
        }
        TyKind::RigidTy(RigidTy::Ref(_, inner, _))
        | TyKind::RigidTy(RigidTy::RawPtr(inner, _))
        | TyKind::RigidTy(RigidTy::Slice(inner)) => walk_closure_ty(&inner, out),
        TyKind::RigidTy(RigidTy::Array(inner, _)) => walk_closure_ty(&inner, out),
        TyKind::RigidTy(RigidTy::Pat(inner, _)) => walk_closure_ty(&inner, out),
        TyKind::RigidTy(RigidTy::Tuple(elems)) => {
            for elem in &elems {
                walk_closure_ty(elem, out);
            }
        }
        TyKind::RigidTy(RigidTy::FnPtr(sig)) => {
            for t in &sig.value.inputs_and_output {
                walk_closure_ty(t, out);
            }
        }
        TyKind::RigidTy(RigidTy::Adt(_, args))
        | TyKind::RigidTy(RigidTy::FnDef(_, args))
        | TyKind::RigidTy(RigidTy::Coroutine(_, args))
        | TyKind::RigidTy(RigidTy::CoroutineClosure(_, args))
        | TyKind::RigidTy(RigidTy::CoroutineWitness(_, args)) => walk_closure_args(&args, out),
        TyKind::RigidTy(RigidTy::Dynamic(bounds, _)) => {
            for bound in &bounds {
                match &bound.value {
                    rustc_public::ty::ExistentialPredicate::Trait(t) => {
                        walk_closure_args(&t.generic_args, out)
                    }
                    rustc_public::ty::ExistentialPredicate::Projection(p) => {
                        walk_closure_args(&p.generic_args, out)
                    }
                    rustc_public::ty::ExistentialPredicate::AutoTrait(_) => {}
                }
            }
        }
        TyKind::Alias(_, alias) => walk_closure_args(&alias.args, out),
        _ => {}
    }
}

/// Whether a call target naming one of the iterator-consumer
/// builtins (`Iterator::count` / `fold` / `for_each` /
/// `collect`, `FromIterator::from_iter`) must still have its
/// instance body collected: the builtins drive the modelled
/// `next` state machine, which only the builtin-produced
/// iterator states satisfy, so for any other iterator type the
/// instance body (including its specializations) is the
/// faithful execution and conversion prefers it when present.
/// The iterator type is the call's first generic argument, or
/// the third for `from_iter` (whose first is the collection).
fn consumer_instance_needs_body(
    def: &rustc_public::ty::FnDef,
    canonical: &str,
    args: &GenericArgs,
) -> bool {
    // `Index` / `IndexMut`: the builtins read the receiver as a
    // sequence (slices, arrays, `Vec`, `String`), which is the
    // faithful semantics only for those receivers. Any other
    // `Self` type — a map's key lookup above all — has an impl
    // body that means something else entirely, and the
    // interpreter's impl-call resolution prefers a collected
    // body over the builtin, so the instance must be collected.
    // The `Self` type is the call's first generic argument.
    if matches!(
        canonical,
        "std::ops::Index::index" | "std::ops::IndexMut::index_mut"
    ) {
        let Some(GenericArgKind::Type(ty)) = args.0.first() else {
            return false;
        };
        return !index_self_modelled(ty);
    }
    // The comparison builtins (`PartialEq::eq` and family)
    // compare values structurally — the derived semantics.
    // A manual impl that compares something else is silently
    // wrong under the builtin, but demanding every unmodelled
    // type's body is not safe either: a derived body may
    // lower its comparison to shapes the conversion leaves
    // unsupported (a whole-value transmute, a discriminant
    // `BinaryOp Cmp`), and a collected body then replaces a
    // working builtin with a runtime failure. The demand is
    // therefore limited to the measured case: `Cow`, whose
    // manual impl compares the borrowed and owned contents
    // across variants — a comparison the structural builtin
    // gets wrong (different variants are never structurally
    // equal), and the common manual-equality hash key. The
    // `Self` type is the call's first generic argument.
    if matches!(
        canonical,
        "std::cmp::PartialEq::eq"
            | "std::cmp::PartialEq::ne"
            | "std::cmp::PartialOrd::partial_cmp"
            | "std::cmp::PartialOrd::lt"
            | "std::cmp::PartialOrd::le"
            | "std::cmp::PartialOrd::gt"
            | "std::cmp::PartialOrd::ge"
            | "std::cmp::Ord::cmp"
    ) {
        let Some(GenericArgKind::Type(ty)) = args.0.first() else {
            return false;
        };
        return matches!(
            guarded_ty_kind(ty),
            Some(TyKind::RigidTy(RigidTy::Adt(def, _)))
                if def.name() == "std::borrow::Cow"
        );
    }
    // `Default::default`: the builtin constructs the pure
    // shapes (`default_value_for_tag`: primitives, `String`,
    // `Vec`, `Option`, cells, tuples, arrays) and the
    // allocating wrappers (`Box` / `Rc` / `Arc`, `RandomState`)
    // only. A derived (or manual) `Default` for any other ADT
    // -- the regex builders above all -- is not served by the
    // builtin at runtime, so the instance body is the only
    // faithful execution and must be collected.
    if matches!(
        canonical,
        "std::default::Default::default" | "core::default::Default::default"
    ) {
        let Some(GenericArgKind::Type(ty)) = args.0.first() else {
            return false;
        };
        return !default_self_served(ty);
    }
    // Hash-collection iteration: the `next` / `into_iter`
    // builtins model iterator states for sequences, not a raw
    // table's control-byte walk, so hash iterator and hash
    // collection instances keep their bodies (the interpreter's
    // conversion prefers a collected body for these shapes).
    if matches!(
        canonical,
        "std::iter::Iterator::next" | "std::iter::IntoIterator::into_iter"
    ) {
        let Some(GenericArgKind::Type(ty)) = args.0.first() else {
            return false;
        };
        return if canonical.ends_with("next") {
            hash_iterator_ty(ty)
                || (str_iter_ty(ty) && !str_split_char_ty(ty))
                || foreign_adt_ty(ty)
        } else {
            hash_collection_ty(ty)
                || drain_ty(ty)
                || str_iter_ty(ty)
                || foreign_adt_ty(ty)
        };
    }
    let iter_arg = match canonical {
        "std::iter::Iterator::count"
        | "std::iter::Iterator::fold"
        | "std::iter::Iterator::for_each"
        | "std::iter::Iterator::collect" => 0,
        "std::iter::FromIterator::from_iter" => 2,
        "std::iter::DoubleEndedIterator::next_back" => {
            // `next_back` differs from the consumers above: the
            // builtin serves the call directly, so a collected
            // body is needed exactly for the iterator types the
            // builtin's back step does not model — every other
            // type (adapters whose bodies compose modelled
            // steps, and user-defined iterators) keeps its body
            // as the primary execution, with the builtin as the
            // fallback when no clean body is available.
            let Some(GenericArgKind::Type(ty)) = args.0.first() else {
                return false;
            };
            return !iterator_back_modelled(ty);
        }
        "std::iter::ExactSizeIterator::len" => {
            // `len` likewise: the builtin serves the call
            // directly, reading the remaining count off the
            // slice-state iterators' unified bounds, so a
            // collected body is needed exactly for the
            // iterator types it does not model — every other
            // type (the trait default, which delegates to
            // `size_hint`, and the types' own overrides) keeps
            // its body as the primary execution.
            let Some(GenericArgKind::Type(ty)) = args.0.first() else {
                return false;
            };
            return !slice_iter_len_modelled(ty);
        }
        _ => return false,
    };
    let Some(GenericArgKind::Type(ty)) = args.0.get(iter_arg) else {
        return false;
    };
    if iterator_self_modelled(ty) {
        return false;
    }
    // A `fold` instance whose body drives the iterator through
    // the `TrustedRandomAccess` specialization (core's adapter
    // helpers reach `__iterator_get_unchecked` and read the
    // sides through stored index/length counters) cannot run on
    // a modelled adapter state: the state machine maintains no
    // such counters, and the sides' unchecked-access bodies
    // address their real layouts, not the model states. The
    // builtin's fold — the same `next` loop the trait's default
    // body spells out — is the faithful execution for those
    // instances, and every other consumer (`count`, `for_each`,
    // `last`, `sum`, `reduce`, ...) reaches it through its own
    // collected default body, which delegates to `fold`. The
    // remaining instances — the default body itself and the
    // composition overrides that delegate to the inner
    // iterator's consumers — keep their bodies as today.
    if canonical == "std::iter::Iterator::fold"
        && iterator_adapter_modelled(ty)
        && fold_instance_tra_specialized(def, args)
    {
        return false;
    }
    true
}

/// Whether this type is one of the core adapter types whose
/// state the interpreter's modelled `next` machine steps by
/// composition (the adapter arms of `next_on_state` in
/// sparagmos-interp): their states are produced by the builtin
/// machinery, so a consumer body that bypasses `next` cannot
/// run on them. Judged by the iterator type's ADT name, as in
/// `iterator_self_modelled`.
fn iterator_adapter_modelled(ty: &Ty) -> bool {
    let Some(TyKind::RigidTy(RigidTy::Adt(def, _))) = guarded_ty_kind(ty) else {
        return false;
    };
    matches!(
        def.name().as_str(),
        "std::iter::Chain"
            | "std::iter::Copied"
            | "std::iter::Enumerate"
            | "std::iter::Map"
            | "std::iter::Rev"
            | "std::iter::Skip"
            | "std::iter::SkipWhile"
            | "std::iter::Take"
            | "std::iter::Zip"
    )
}

/// Whether the resolved `Iterator::fold` instance is served by
/// core's `TrustedRandomAccess` specialization: the instance
/// body, followed through the adapter-helper bodies it
/// delegates to (the `*Impl` / `Spec*` traits under
/// `core::iter::adapters`), calls an unchecked-access entry
/// (`__iterator_get_unchecked`, or an adapter's own
/// `get_unchecked`). Composition overrides delegate to the
/// inner iterator's trait consumers instead and never reach
/// those entries. The walk is bounded; any resolution or body
/// failure answers "not specialized", keeping today's
/// collection.
fn fold_instance_tra_specialized(
    def: &rustc_public::ty::FnDef,
    args: &GenericArgs,
) -> bool {
    let mut queue: VecDeque<(rustc_public::ty::FnDef, GenericArgs)> =
        VecDeque::from([(def.clone(), args.clone())]);
    let mut visited: HashSet<String> = HashSet::new();
    let mut scanned = 0usize;
    while let Some((d, a)) = queue.pop_front() {
        if scanned >= 16 {
            break;
        }
        let Some(inst) = crate::convert::resolve_instance_guarded(d, &a) else {
            continue;
        };
        if !visited.insert(inst.name().to_string()) {
            continue;
        }
        scanned += 1;
        let Ok(Some(body)) = fetch_body_contained(&inst) else {
            continue;
        };
        for (tdef, targs) in call_targets(&body) {
            let tname =
                sparagmos_interp::builtins::canonical_name(&tdef.def_id().name());
            if tname.contains("__iterator_get_unchecked")
                || tname.ends_with("::get_unchecked")
            {
                return true;
            }
            if tname.contains("::adapters::") {
                queue.push_back((tdef, targs));
            }
        }
    }
    false
}

/// Whether a call target naming `PartialEq::eq` / `ne` whose
/// `Self` type is a reference must still have its instance body
/// collected: that instance is core's blanket impl for `&A`,
/// whose body is a dereference plus a call to `A`'s own `eq`.
/// The builtin's structural comparison cannot serve it — it
/// compares the pointees field by field, which is a different
/// result whenever `A`'s `eq` is hand-written or derived over a
/// field with a hand-written `eq` (a derived `PartialEq` compares
/// its fields through exactly this blanket impl). The blanket
/// body is tiny and its inner call resolves by the ordinary
/// rules, so collecting it is the faithful execution; every
/// non-reference `Self` keeps the builtin model.
fn partial_eq_ref_instance_needs_body(canonical: &str, args: &GenericArgs) -> bool {
    if !matches!(
        canonical,
        "std::cmp::PartialEq::eq"
            | "core::cmp::PartialEq::eq"
            | "std::cmp::PartialEq::ne"
            | "core::cmp::PartialEq::ne"
    ) {
        return false;
    }
    let Some(GenericArgKind::Type(ty)) = args.0.first() else {
        return false;
    };
    matches!(
        guarded_ty_kind(ty),
        Some(TyKind::RigidTy(RigidTy::Ref(..)))
    )
}
/// The receiver types the `Index` builtins serve directly:
/// the sequence types whose elements the builtin reads —
/// arrays, slices, `str`, references to them, `Vec` and
/// `String`. Judged on the `Self` type of the indexing impl.
fn index_self_modelled(ty: &Ty) -> bool {
    match guarded_ty_kind(ty) {
        Some(
            TyKind::RigidTy(RigidTy::Array(..))
            | TyKind::RigidTy(RigidTy::Slice(..))
            | TyKind::RigidTy(RigidTy::Str)
            | TyKind::RigidTy(RigidTy::Ref(..))
            | TyKind::RigidTy(RigidTy::RawPtr(..)),
        ) => true,
        Some(TyKind::RigidTy(RigidTy::Adt(def, _))) => matches!(
            def.name().as_str(),
            "std::vec::Vec" | "alloc::vec::Vec" | "std::string::String"
        ),
        _ => false,
    }
}

/// Whether this type is a hash-collection iterator (the
/// `hash_map` / `hash_set` view types or hashbrown's raw
/// iterators), judged by the ADT definition name.
fn hash_iterator_ty(ty: &Ty) -> bool {
    // The `&mut I` blanket `Iterator` impl names the reference
    // as its `Self` type.
    if let Some(TyKind::RigidTy(RigidTy::Ref(_, inner, _))) = guarded_ty_kind(ty) {
        return hash_iterator_ty(&inner);
    }
    let Some(TyKind::RigidTy(RigidTy::Adt(def, _))) = guarded_ty_kind(ty) else {
        return false;
    };
    let name = def.name();
    name.starts_with("std::collections::hash_map::")
        || name.starts_with("std::collections::hash_set::")
        || name.starts_with("hashbrown::map::")
        || name.starts_with("hashbrown::set::")
        || name.starts_with("hashbrown::raw::RawIter")
        || name.starts_with("hashbrown::raw::RawIntoIter")
        || name.starts_with("hashbrown::raw::RawDrain")
        || name.starts_with("hashbrown::raw::RawExtractIf")
}

/// Whether this `IntoIterator` `Self` type is a hash
/// collection, or a reference to one, or a reference to a
/// hash-collection iterator (the `&mut I` blanket impl).
fn hash_collection_ty(ty: &Ty) -> bool {
    if hash_iterator_ty(ty) {
        return true;
    }
    match guarded_ty_kind(ty) {
        Some(TyKind::RigidTy(RigidTy::Adt(def, _))) => matches!(
            def.name().as_str(),
            "std::collections::HashMap" | "std::collections::HashSet"
        ),
        Some(TyKind::RigidTy(RigidTy::Ref(_, inner, _))) => {
            hash_collection_ty(&inner) || hash_iterator_ty(&inner)
        }
        _ => false,
    }
}

/// Whether `canonical` (a builtin lookup name) is one of
/// `core::fmt::rt::Argument`'s value constructors — the functions
/// that pair a reference to the value being formatted with the
/// formatting function for its type. `from_usize` is not one: it
/// formats an argument index and stores no formatter function.
fn is_fmt_argument_constructor(canonical: &str) -> bool {
    canonical.starts_with("core::fmt::rt::Argument::new_")
}


/// Whether the `Default` builtin serves this `Self` type
/// structurally: the pure shapes of `default_value_for_tag`
/// (anything that is not an ADT -- primitives, references,
/// tuples, arrays -- recurses into element calls, which are
/// judged on their own instances) plus the ADT names the
/// builtin constructs directly.
fn default_self_served(ty: &Ty) -> bool {
    let Some(TyKind::RigidTy(RigidTy::Adt(def, _))) = guarded_ty_kind(ty) else {
        return true;
    };
    matches!(
        def.name().as_str(),
        "std::string::String"
            | "alloc::string::String"
            | "std::vec::Vec"
            | "alloc::vec::Vec"
            | "std::option::Option"
            | "core::option::Option"
            | "std::marker::PhantomData"
            | "core::marker::PhantomData"
            | "std::cell::Cell"
            | "core::cell::Cell"
            | "std::cell::RefCell"
            | "core::cell::RefCell"
            | "std::boxed::Box"
            | "alloc::boxed::Box"
            | "std::rc::Rc"
            | "alloc::rc::Rc"
            | "std::sync::Arc"
            | "alloc::sync::Arc"
            | "std::hash::RandomState"
    )
}

/// Whether `ty` is an ADT from outside the standard library
/// (a crate's own iterator wrapper, a support crate's type):
/// no builtin models its iteration step, so the real body is
/// the only route.
fn foreign_adt_ty(ty: &Ty) -> bool {
    let Some(TyKind::RigidTy(RigidTy::Adt(def, _))) = guarded_ty_kind(ty) else {
        return false;
    };
    let name = def.name();
    !name.starts_with("std::") && !name.starts_with("alloc::") && !name.starts_with("core::")
}

/// Whether `ty` is one of the `str` pattern iterators
/// (`Split`, `Matches`, `Lines`, ...) whose `IntoIterator`
/// identity the builtin does not serve: the blanket body is
/// collected for them instead. `Chars` / `Bytes` are served
/// by the builtin and stay excluded.
fn str_iter_ty(ty: &Ty) -> bool {
    let Some(TyKind::RigidTy(RigidTy::Adt(def, _))) = guarded_ty_kind(ty) else {
        return false;
    };
    let name = def.name();
    if !name.contains("::str::") {
        return false;
    }
    matches!(
        name.rsplit("::").next(),
        Some(
            "Split"
                | "RSplit"
                | "SplitTerminator"
                | "RSplitTerminator"
                | "SplitN"
                | "RSplitN"
                | "SplitInclusive"
                | "RSplitInclusive"
                | "Matches"
                | "MatchIndices"
                | "CharIndices"
                | "Lines"
                | "SplitWhitespace"
                | "SplitAsciiWhitespace"
                | "EncodeUtf16"
        )
    )
}

/// Whether `ty` is one of the `str` split iterators over a
/// `char` pattern (`Split`, `RSplit`, `SplitTerminator`,
/// `RSplitTerminator`, `SplitInclusive`): the interpreter's
/// builtin replays both steps of this family for char
/// patterns (`str_split_step` in sparagmos-interp), so their
/// `next` instances stay builtin-served and out of
/// `str_iter_ty`'s body demand. Collecting the real body
/// replaces the replay with an execution that dies inside
/// `core::slice::memchr::memrchr_aligned`, which has no
/// converted body. Non-char patterns keep the collected
/// body: the replay rejects non-char searcher shapes.
fn str_split_char_ty(ty: &Ty) -> bool {
    let Some(TyKind::RigidTy(RigidTy::Adt(def, args))) = guarded_ty_kind(ty) else {
        return false;
    };
    let name = def.name();
    if !name.contains("::str::") {
        return false;
    }
    if !matches!(
        name.rsplit("::").next(),
        Some(
            "Split"
                | "RSplit"
                | "SplitTerminator"
                | "RSplitTerminator"
                | "SplitInclusive"
        )
    ) {
        return false;
    }
    // The pattern is the ADT's type argument (after the
    // haystack lifetime); the replay serves exactly `char`.
    let pattern = args.0.iter().find_map(|arg| match arg {
        GenericArgKind::Type(ty) => Some(ty),
        _ => None,
    });
    matches!(
        pattern,
        Some(ty) if matches!(guarded_ty_kind(ty), Some(TyKind::RigidTy(RigidTy::Char)))
    )
}

/// Whether this `IntoIterator` `Self` type is a `Drain`-class
/// iterator: its `into_iter` is the identity over a state the
/// builtin sequence models do not produce, so the instance
/// body is the faithful execution.
fn drain_ty(ty: &Ty) -> bool {
    let Some(TyKind::RigidTy(RigidTy::Adt(def, _))) = guarded_ty_kind(ty) else {
        return false;
    };
    def.name().rsplit("::").next() == Some("Drain")
}

/// The Stable MIR counterpart of the interpreter's modelled
/// `next` shapes (see `iterator_next_modelled` in
/// sparagmos-interp): ranges, the `[items, index]` iterators,
/// `Empty`, and the mpsc channel iterators, judged by the
/// iterator type's ADT name.
fn iterator_self_modelled(ty: &Ty) -> bool {
    let Some(TyKind::RigidTy(RigidTy::Adt(def, _))) = guarded_ty_kind(ty) else {
        return false;
    };
    matches!(
        def.name().as_str(),
        "std::ops::Range"
            | "std::ops::RangeInclusive"
            | "std::slice::Iter"
            | "std::collections::btree_set::Iter"
            | "alloc::collections::btree_set::Iter"
            | "std::collections::btree_map::Iter"
            | "alloc::collections::btree_map::Iter"
            | "std::collections::btree_map::Keys"
            | "alloc::collections::btree_map::Keys"
            | "std::collections::btree_map::Values"
            | "alloc::collections::btree_map::Values"
            | "std::collections::btree_set::IntoIter"
            | "alloc::collections::btree_set::IntoIter"
            | "std::collections::btree_map::IntoIter"
            | "alloc::collections::btree_map::IntoIter"
            | "std::vec::IntoIter"
            | "alloc::vec::IntoIter"
            | "std::env::Args"
            | "std::env::ArgsOs"
            | "std::array::IntoIter"
            | "std::iter::Empty"
            | "std::sync::mpsc::Iter"
            | "std::sync::mpsc::IntoIter"
            | "std::str::Chars"
            | "core::str::Chars"
            | "std::str::Bytes"
            | "core::str::Bytes"
    )
}

/// Whether this type is a slice-state iterator whose
/// `ExactSizeIterator::len` the interpreter's builtin
/// serves (see `iterator_len_modelled` in
/// sparagmos-interp): the `slice::Iter` / `IterMut` pair
/// and the B-tree iterators sharing their state layout.
fn slice_iter_len_modelled(ty: &Ty) -> bool {
    let Some(TyKind::RigidTy(RigidTy::Adt(def, _))) = guarded_ty_kind(ty) else {
        return false;
    };
    matches!(
        def.name().as_str(),
        "std::slice::Iter"
            | "std::slice::IterMut"
            | "std::collections::btree_set::Iter"
            | "alloc::collections::btree_set::Iter"
            | "std::collections::btree_map::Iter"
            | "alloc::collections::btree_map::Iter"
            | "std::collections::btree_map::Keys"
            | "alloc::collections::btree_map::Keys"
            | "std::collections::btree_map::Values"
            | "alloc::collections::btree_map::Values"
    )
}

/// The Stable MIR counterpart of the interpreter's modelled
/// `next_back` shapes (see `iterator_next_back_modelled` in
/// sparagmos-interp): the leaf iterators whose states the
/// builtin's back step transitions directly. Adapter types
/// (`Rev`, `Enumerate`, `Map`, ...) are deliberately absent:
/// their bodies compose the modelled steps and stay the
/// primary execution.
fn iterator_back_modelled(ty: &Ty) -> bool {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return false;
    };
    match def.name().as_str() {
        "std::ops::Range"
        | "std::ops::RangeInclusive"
        | "std::slice::Iter"
        | "std::slice::IterMut"
        | "std::collections::btree_set::Iter"
        | "alloc::collections::btree_set::Iter"
        | "std::collections::btree_map::Iter"
        | "alloc::collections::btree_map::Iter"
        | "std::collections::btree_map::Keys"
        | "alloc::collections::btree_map::Keys"
        | "std::collections::btree_map::Values"
        | "alloc::collections::btree_map::Values"
        | "std::collections::btree_set::IntoIter"
        | "alloc::collections::btree_set::IntoIter"
        | "std::collections::btree_map::IntoIter"
        | "alloc::collections::btree_map::IntoIter"
        | "std::vec::IntoIter"
        | "alloc::vec::IntoIter"
        | "std::array::IntoIter"
        | "std::option::IntoIter"
        | "std::option::Iter"
        | "std::option::IterMut"
        | "std::iter::Empty"
        | "std::iter::Once"
        | "std::env::Args"
        | "std::env::ArgsOs"
        | "std::str::Chars"
        | "core::str::Chars"
        | "std::str::Bytes"
        | "core::str::Bytes"
        | "std::str::Split"
        | "core::str::Split"
        | "std::str::RSplit"
        | "core::str::RSplit"
        | "std::str::SplitTerminator"
        | "core::str::SplitTerminator"
        | "std::str::RSplitTerminator"
        | "core::str::RSplitTerminator"
        | "std::str::SplitInclusive"
        | "core::str::SplitInclusive" => true,
        // A `Rev`'s back step is its inner iterator's front
        // step, so it is modelled exactly when the inner
        // iterator's `next` is.
        "std::iter::Rev" => first_type_arg(&args).is_some_and(iterator_self_modelled),
        // These adapters' back steps compose the inner back
        // step, so they are modelled exactly when it is. (An
        // `Enumerate` body would also work when collected, but
        // it computes the index through the inner iterator's
        // `len` body, which for a pointer-pair state such as
        // `slice::Iter`'s is not executable; the builtin
        // computes it from the modelled size hint instead.)
        "std::iter::Enumerate" | "std::iter::Copied" | "std::iter::Cloned" | "std::iter::Map" | "std::iter::Filter" | "std::iter::FilterMap" => {
            first_type_arg(&args).is_some_and(iterator_back_modelled)
        }
        // A `Zip`'s back step composes the two sides' back
        // steps, so it is modelled exactly when both sides'
        // are.
        "std::iter::Zip" => {
            let mut type_args = args.0.iter().filter_map(|arg| match arg {
                GenericArgKind::Type(ty) => Some(ty),
                _ => None,
            });
            match (type_args.next(), type_args.next()) {
                (Some(a), Some(b)) => {
                    iterator_back_modelled(a) && iterator_back_modelled(b)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// The first type among a set of generic arguments.
fn first_type_arg(args: &GenericArgs) -> Option<&Ty> {
    args.0.iter().find_map(|arg| match arg {
        GenericArgKind::Type(ty) => Some(ty),
        _ => None,
    })
}

/// The closure types among one set of generic arguments.
pub fn closures_in_args(args: &GenericArgs) -> Vec<(ClosureDef, GenericArgs)> {
    let mut out = Vec::new();
    walk_closure_args(args, &mut out);
    out
}

pub fn closure_targets(body: &mir::Body) -> Vec<(ClosureDef, GenericArgs)> {
    let locals = body.locals();
    let mut out = Vec::new();
    for decl in locals {
        walk_closure_ty(&decl.ty, &mut out);
    }
    let mut operand_ty = |op: &Operand, out: &mut Vec<(ClosureDef, GenericArgs)>| {
        if let Ok(ty) = op.ty(&locals) {
            walk_closure_ty(&ty, out);
        }
    };
    for bb in &body.blocks {
        for stmt in &bb.statements {
            let StatementKind::Assign(_, rvalue) = &stmt.kind else {
                continue;
            };
            match rvalue {
                Rvalue::Aggregate(AggregateKind::Closure(def, args), ops) => {
                    out.push((*def, args.clone()));
                    walk_closure_args(args, &mut out);
                    for op in ops {
                        operand_ty(op, &mut out);
                    }
                }
                Rvalue::Aggregate(AggregateKind::CoroutineClosure(def, args), ops) => {
                    // An async closure's value is created here, and
                    // the aggregate is the one place that carries
                    // its definition and arguments: the closure's
                    // *type* cannot be asked for them (the bridge
                    // leaves coroutine-closure type conversion
                    // unimplemented). The demand is recorded under
                    // the same `ClosureDef` key the closure
                    // collection resolves, since an async closure
                    // is instantiated through the closure machinery
                    // like any other closure.
                    out.push((ClosureDef(def.def_id()), args.clone()));
                    walk_closure_args(args, &mut out);
                    for op in ops {
                        operand_ty(op, &mut out);
                    }
                }
                Rvalue::Aggregate(_, ops) => {
                    for op in ops {
                        operand_ty(op, &mut out);
                    }
                }
                Rvalue::Use(op, _)
                | Rvalue::Repeat(op, _)
                | Rvalue::UnaryOp(_, op) => operand_ty(op, &mut out),
                Rvalue::Cast(_, op, ty) => {
                    operand_ty(op, &mut out);
                    walk_closure_ty(ty, &mut out);
                }
                Rvalue::BinaryOp(_, a, b) | Rvalue::CheckedBinaryOp(_, a, b) => {
                    operand_ty(a, &mut out);
                    operand_ty(b, &mut out);
                }
                _ => {}
            }
        }
        match &bb.terminator.kind {
            TerminatorKind::Call { func, args, .. } => {
                operand_ty(func, &mut out);
                for op in args {
                    operand_ty(op, &mut out);
                }
            }
            TerminatorKind::Assert { cond, .. } => operand_ty(cond, &mut out),
            TerminatorKind::SwitchInt { discr, .. } => operand_ty(discr, &mut out),
            _ => {}
        }
    }
    out
}

/// The coroutine types one Stable MIR body mentions: every
/// `Coroutine(def, args)` type reachable from the body's local
/// declarations, its operands' types, and its coroutine aggregate
/// rvalues, walking nested types. This is the coroutine counterpart
/// of [`closure_targets`]: a coroutine (an `async` block, an
/// `async fn`'s future, a `gen` block) is created by an
/// `AggregateKind::Coroutine` rvalue in the enclosing body and held
/// in locals of coroutine type, but it is neither a call target nor
/// a closure, so neither the instance demand nor the closure demand
/// ever sees it — without this scan a demanded coroutine stays
/// unconverted however concrete its context is. The arguments are
/// the body's own substitution applied, as in [`closure_targets`].
pub fn coroutine_targets(body: &mir::Body) -> Vec<(rustc_public::ty::CoroutineDef, GenericArgs)> {
    fn walk_args(
        args: &GenericArgs,
        out: &mut Vec<(rustc_public::ty::CoroutineDef, GenericArgs)>,
    ) {
        for arg in &args.0 {
            match arg {
                GenericArgKind::Type(ty) => walk_ty(ty, out),
                GenericArgKind::Const(konst) => match konst.kind() {
                    rustc_public::ty::TyConstKind::Unevaluated(_, inner) => {
                        walk_args(&inner, out)
                    }
                    rustc_public::ty::TyConstKind::Value(ty, _)
                    | rustc_public::ty::TyConstKind::ZSTValue(ty) => walk_ty(&ty, out),
                    _ => {}
                },
                GenericArgKind::Lifetime(_) => {}
            }
        }
    }

    fn walk_ty(ty: &Ty, out: &mut Vec<(rustc_public::ty::CoroutineDef, GenericArgs)>) {
        // A type passing through a coroutine-closure type
        // cannot be asked its kind (see `guarded_ty_kind`);
        // nothing can be extracted from it here either.
        let Some(kind) = guarded_ty_kind(ty) else {
            return;
        };
        match kind {
            TyKind::RigidTy(RigidTy::Coroutine(def, args)) => {
                out.push((def, args.clone()));
                walk_args(&args, out);
            }
            TyKind::RigidTy(RigidTy::Ref(_, inner, _))
            | TyKind::RigidTy(RigidTy::RawPtr(inner, _))
            | TyKind::RigidTy(RigidTy::Slice(inner)) => walk_ty(&inner, out),
            TyKind::RigidTy(RigidTy::Array(inner, _)) => walk_ty(&inner, out),
            TyKind::RigidTy(RigidTy::Pat(inner, _)) => walk_ty(&inner, out),
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                for elem in &elems {
                    walk_ty(elem, out);
                }
            }
            TyKind::RigidTy(RigidTy::FnPtr(sig)) => {
                for t in &sig.value.inputs_and_output {
                    walk_ty(t, out);
                }
            }
            TyKind::RigidTy(RigidTy::Adt(_, args))
            | TyKind::RigidTy(RigidTy::FnDef(_, args))
            | TyKind::RigidTy(RigidTy::Closure(_, args))
            | TyKind::RigidTy(RigidTy::CoroutineClosure(_, args))
            | TyKind::RigidTy(RigidTy::CoroutineWitness(_, args)) => walk_args(&args, out),
            TyKind::RigidTy(RigidTy::Dynamic(bounds, _)) => {
                for bound in &bounds {
                    match &bound.value {
                        rustc_public::ty::ExistentialPredicate::Trait(t) => {
                            walk_args(&t.generic_args, out)
                        }
                        rustc_public::ty::ExistentialPredicate::Projection(p) => {
                            walk_args(&p.generic_args, out)
                        }
                        rustc_public::ty::ExistentialPredicate::AutoTrait(_) => {}
                    }
                }
            }
            TyKind::Alias(_, alias) => walk_args(&alias.args, out),
            _ => {}
        }
    }

    let locals = body.locals();
    let mut out = Vec::new();
    for decl in locals {
        walk_ty(&decl.ty, &mut out);
    }
    let mut operand_ty = |op: &Operand,
                          out: &mut Vec<(rustc_public::ty::CoroutineDef, GenericArgs)>| {
        if let Ok(ty) = op.ty(&locals) {
            walk_ty(&ty, out);
        }
    };
    for bb in &body.blocks {
        for stmt in &bb.statements {
            let StatementKind::Assign(_, rvalue) = &stmt.kind else {
                continue;
            };
            match rvalue {
                Rvalue::Aggregate(AggregateKind::Coroutine(def, args), ops) => {
                    out.push((*def, args.clone()));
                    walk_args(args, &mut out);
                    for op in ops {
                        operand_ty(op, &mut out);
                    }
                }
                Rvalue::Aggregate(_, ops) => {
                    for op in ops {
                        operand_ty(op, &mut out);
                    }
                }
                Rvalue::Use(op, _) | Rvalue::Repeat(op, _) | Rvalue::UnaryOp(_, op) => {
                    operand_ty(op, &mut out)
                }
                Rvalue::Cast(_, op, ty) => {
                    operand_ty(op, &mut out);
                    walk_ty(ty, &mut out);
                }
                Rvalue::BinaryOp(_, a, b) | Rvalue::CheckedBinaryOp(_, a, b) => {
                    operand_ty(a, &mut out);
                    operand_ty(b, &mut out);
                }
                _ => {}
            }
        }
        match &bb.terminator.kind {
            TerminatorKind::Call { func, args, .. } => {
                operand_ty(func, &mut out);
                for op in args {
                    operand_ty(op, &mut out);
                }
            }
            TerminatorKind::Assert { cond, .. } => operand_ty(cond, &mut out),
            TerminatorKind::SwitchInt { discr, .. } => operand_ty(discr, &mut out),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pop_all<B>(frontier: &mut Frontier<B>) -> Vec<B> {
        let mut out = Vec::new();
        while let Some((_, body)) = frontier.pop() {
            out.push(body);
        }
        out
    }

    #[test]
    fn fifo_frontier_pops_in_push_order() {
        let mut frontier = Frontier::new(CollectionOrder::Fifo);
        frontier.push(0, Demand::Call, "entry");
        frontier.push(1, Demand::Value, "other");
        frontier.push(0, Demand::Value, "third");
        assert_eq!(pop_all(&mut frontier), vec!["entry", "other", "third"]);
    }

    #[test]
    fn execution_first_frontier_orders_by_distance_then_demand() {
        let mut frontier = Frontier::new(CollectionOrder::ExecutionFirst);
        frontier.push(1, Demand::Call, "seed");
        frontier.push(0, Demand::Call, "entry");
        frontier.push(1, Demand::Value, "value-of-seed");
        frontier.push(2, Demand::Call, "deep-call");
        frontier.push(1, Demand::Call, "call-of-entry");
        assert_eq!(
            pop_all(&mut frontier),
            vec![
                "entry",
                "seed",
                "call-of-entry",
                "value-of-seed",
                "deep-call"
            ]
        );
    }

    #[test]
    fn execution_first_frontier_is_stable_for_equal_keys() {
        let mut frontier = Frontier::new(CollectionOrder::ExecutionFirst);
        frontier.push(1, Demand::Call, "first");
        frontier.push(1, Demand::Call, "second");
        frontier.push(1, Demand::Call, "third");
        assert_eq!(pop_all(&mut frontier), vec!["first", "second", "third"]);
    }
}
