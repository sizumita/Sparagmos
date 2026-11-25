//! Per-item conversion entry: converting one crate item and classifying closure instances.

use super::*;

/// Convert one function (a local item or an auxiliary-crate
/// function) and record the outcome. `body` is the function's
/// monomorphic MIR body: `Err(reason)` records the function as
/// skipped with that reason (no monomorphic instance), and
/// `Ok(None)` records it as skipped with no body available.
pub(crate) fn convert_item(
    conv: &mut Converter,
    report: &mut ConversionReport,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    name: String,
    def_id: sparagmos_mir::DefId,
    body: Result<Option<rustc_public::mir::Body>, String>,
) {
    let body = match body {
        Ok(Some(b)) => b,
        Ok(None) => {
            let mut reasons = BTreeMap::new();
            reasons.insert("no MIR body available".to_string(), 1);
            report.functions.push(FnReport {
                name,
                blocks: 0,
                locals: 0,
                arg_count: 0,
                outcome: FnOutcome::Skipped { reasons },
            });
            return;
        }
        Err(reason) => {
            let mut reasons = BTreeMap::new();
            reasons.insert(reason, 1);
            report.functions.push(FnReport {
                name,
                blocks: 0,
                locals: 0,
                arg_count: 0,
                outcome: FnOutcome::Skipped { reasons },
            });
            return;
        }
    };
    let blocks = body.blocks.len();
    let locals = body.locals().len();
    let arg_count = body.arg_locals().len();
    match convert_body_isolated(&mut *conv, &body) {
        Ok(converted) => {
            bodies.push((name.clone(), converted));
            body_def_ids.push(def_id);
            body_def_names.push(name.clone());
            report.functions.push(FnReport {
                name,
                blocks,
                locals,
                arg_count,
                outcome: FnOutcome::Converted,
            });
        }
        Err(errs) => {
            let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
            for e in errs {
                *reasons.entry(e).or_default() += 1;
            }
            report.functions.push(FnReport {
                name,
                blocks: 0,
                locals: 0,
                arg_count: 0,
                outcome: FnOutcome::Skipped { reasons },
            });
        }
    }
}

/// Record `name` as skipped for one concrete `reason`.
pub(crate) fn skip_fn(report: &mut ConversionReport, name: &str, reason: String) {
    let mut reasons = BTreeMap::new();
    reasons.insert(reason, 1);
    report.functions.push(FnReport {
        name: name.to_string(),
        blocks: 0,
        locals: 0,
        arg_count: 0,
        outcome: FnOutcome::Skipped { reasons },
    });
}

/// What came of trying to instantiate an item as a closure after
/// `Instance::try_from` rejected it.
pub(crate) enum ClosureAttempt {
    /// A monomorphic closure instance whose body can be converted.
    Instance(Instance),
    /// The item is not a closure at all.
    NotClosure,
    /// The item is a closure that stays skipped, with the reason.
    Unsupported(String),
    /// The closure is defined in a generic context: its own
    /// arguments still mention the enclosing definition's
    /// parameters, so it has no instance as an item — but a
    /// monomorphic instance of the enclosing definition mentions
    /// the closure with concrete arguments, and the closure
    /// collection after the item loop converts the instances
    /// actually demanded (the same deferral generic functions get).
    GenericContext,
    /// The item is an async closure: asking for its type hits the
    /// bridge's unimplemented coroutine-closure type conversion,
    /// so no instance can be formed from the item alone — but a
    /// converted body that creates the closure carries its
    /// definition and arguments in the coroutine-closure
    /// aggregate, and the closure collection after the item loop
    /// accounts the instances actually demanded (the same
    /// deferral generic functions get).
    CoroutineClosure,
}

/// Instantiate a local closure item's body.
///
/// `Instance::try_from` rejects every closure item: a closure
/// definition always carries generic parameters (its parent
/// arguments plus its signature and capture types), so it never has
/// a mono instance *as an item*. A closure defined in a concrete
/// context can still be instantiated from the generic arguments of
/// its own type via `Instance::resolve_closure`, requesting the
/// closure's own kind so no `FnOnce` adapter shim is produced and
/// the instance's body is the closure body itself. Closures whose
/// arguments still mention generic parameters (defined inside a
/// generic function, method or impl) have no single monomorphic
/// body to collect as an item — the same limitation as generic
/// functions — so the item loop defers it and the closure
/// collection after the loop converts the instances that
/// monomorphic bodies actually demand. Async closures are
/// coroutine closures: merely asking for an async closure
/// item's type hits the unimplemented coroutine-closure type
/// conversion, so no instance can be formed from the item —
/// the guard's reason identifies the item, and it is deferred
/// the same way, to be accounted from the coroutine-closure
/// aggregates of converted bodies (the `CoroutineClosure` arm
/// of `closure_instance_kind` below only fires if a future
/// Stable MIR implements the type conversion).
pub(crate) fn closure_instance(item: &rustc_public::CrateItem) -> ClosureAttempt {
    match guard_stable_mir_query("", || closure_instance_kind(item)) {
        Ok(attempt) => attempt,
        Err(reason) if reason == COROUTINE_CLOSURE_TYPE_REASON => {
            ClosureAttempt::CoroutineClosure
        }
        Err(reason) => ClosureAttempt::Unsupported(reason),
    }
}

/// The kind inspection behind [`closure_instance`]: asking for
/// an async closure item's type kind is what materializes the
/// unimplemented coroutine-closure type conversion, so the
/// whole inspection runs under the guard.
fn closure_instance_kind(item: &rustc_public::CrateItem) -> ClosureAttempt {
    match item.ty().kind() {
        TyKind::RigidTy(RigidTy::CoroutineClosure(..)) => ClosureAttempt::Unsupported(
            "async closure body is not converted (coroutine closures are unsupported)".to_string(),
        ),
        TyKind::RigidTy(RigidTy::Closure(def, args)) => {
            if args_mention_params(&args) {
                return ClosureAttempt::GenericContext;
            }
            let Some(kind) = closure_kind(&args) else {
                return ClosureAttempt::Unsupported(
                    "closure kind is not recoverable from its generic arguments".to_string(),
                );
            };
            match sparagmos_driver::convert::resolve_closure_guarded(def, &args, kind) {
                Some(instance) => ClosureAttempt::Instance(instance),
                None => ClosureAttempt::Unsupported(
                    "closure instance resolution failed".to_string(),
                ),
            }
        }
        _ => ClosureAttempt::NotClosure,
    }
}

/// A local function item that has no monomorphic instance as an
/// item (a generic function definition): it is deferred during the
/// item loop and accounted for by [`collect_local_instances`] once
/// the instances its call sites demand are known.
///
/// Tuple-struct and tuple-variant constructors (`ItemKind::Ctor`)
/// are deferred the same way: a generic constructor has no
/// instance as an item either. A constructor differs from a plain
/// function in one respect — when nothing demands it as a value,
/// there is nothing left unconverted: its direct uses are
/// aggregate rvalues in the callers, and the constructor item
/// itself has no body in any monomorphization that was not
/// demanded. [`apply_instance_demands`] accounts such a
/// constructor as converted without a body.
pub(crate) struct DeferredFn {
    pub(crate) def: rustc_public::DefId,
    pub(crate) name: String,
    /// Index of this item's placeholder entry in the report.
    pub(crate) report_idx: usize,
    /// Whether this deferred item is a data constructor
    /// (`ItemKind::Ctor(CtorKind::Fn)`).
    pub(crate) ctor: bool,
}

/// Cap on monomorphic instance bodies collected for generic local
/// definitions per program, mirroring the reachability
/// collection's cap for foreign bodies.
pub(crate) const MAX_LOCAL_INSTANCES: usize = 8192;

/// Depth at which the demand passes report rustc's
/// monomorphization recursion limit: the number of instances of
/// one definition nested along a single demand path (each a
/// distinct instance — a repeated instance is deduplicated and
/// never deepens the path). This mirrors
/// `check_recursion_limit` in rustc's mono collector, which
/// tracks exactly this per-definition path depth against the
/// crate's recursion limit and fails compilation past it
/// ("reached the recursion limit while instantiating ...").
/// 128 is rustc's default recursion limit; a crate's
/// `#![recursion_limit]` attribute is not exposed through
/// Stable MIR, so the default is what the driver can mirror.
pub(crate) const RECURSION_LIMIT_DEPTH: u16 = 128;

/// How many instances of each definition sit on the demand path
/// that led to one body: the chain of instances whose bodies
/// demanded the next, from a collection root down to this body.
/// Only repetitions matter to the recursion-limit check, but the
/// full map is kept so any definition repeating deeper down the
/// path is counted correctly.
pub(crate) type DemandPath = std::collections::HashMap<rustc_public::DefId, u16>;

/// Outcome accumulated for one deferred definition across every
/// demand pass of a program.
#[derive(Default)]
pub(crate) struct Demand {
    pub(crate) converted: usize,
    pub(crate) failures: BTreeMap<String, usize>,
    pub(crate) first_counts: Option<(usize, usize, usize)>,
}
