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

use std::collections::{HashMap, HashSet, VecDeque};

use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::mir::{self, AggregateKind, Operand, Rvalue, StatementKind, TerminatorKind};
use rustc_public::ty::{ClosureDef, GenericArgKind, GenericArgs, RigidTy, Ty, TyKind};
use rustc_public::CrateDef;

use sparagmos_mir as sm;

use crate::convert::Converter;
use crate::report::{FnOutcome, FnReport};

/// Cap on collected bodies per program; see the module docs.
pub const MAX_COLLECTED_BODIES: usize = 1024;

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

/// Collect the bodies of non-local functions reachable from the
/// converted local bodies (`seeds`, Stable MIR form). `local_defs`
/// holds the `DefId`s of the local crate's items; targets resolving
/// to one of them are skipped (they are collected as local items).
/// At most `cap` bodies are collected in this pass; the caller
/// passes the program's remaining budget ([`MAX_COLLECTED_BODIES`]
/// minus what earlier passes collected).
pub fn collect_reachable(
    conv: &mut Converter,
    seeds: &[mir::Body],
    local_defs: &HashSet<rustc_public::DefId>,
    cap: usize,
) -> CollectedFns {
    let mut records: Vec<Record> = Vec::new();
    let mut by_instance: HashMap<(rustc_public::DefId, Instance), ()> = HashMap::new();
    let mut unresolved_seen: HashSet<(rustc_public::DefId, String)> = HashSet::new();
    let mut builtin_seen: HashSet<rustc_public::DefId> = HashSet::new();
    let mut queue: VecDeque<mir::Body> = seeds.iter().cloned().collect();
    let mut bodies_taken = 0usize;
    let mut limit_reached = false;
    let mut walked_bodies: Vec<mir::Body> = Vec::new();

    while let Some(body) = queue.pop_front() {
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
        let mut targets = call_targets(&body);
        targets.extend(fndef_constants(&body));
        for (def, args) in targets {
            let def_id = def.def_id();
            let def_name = def_id.name();
            // Builtin-owned definitions are not collected (see the
            // module docs). The registry is keyed by canonical
            // definition name, so this check is per definition.
            let canonical = sparagmos_interp::builtins::canonical_name(&def_name);
            if sparagmos_interp::builtins::lookup(&canonical).is_some() {
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
                continue;
            }
            let inst = match Instance::resolve(def, &args) {
                Ok(inst) => inst,
                Err(_) => {
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
                        no_body("compiler intrinsic (no fallback body)")
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
                _ => match inst.body() {
                    Some(body) => {
                        bodies_taken += 1;
                        queue.push_back(body.clone());
                        walked_bodies.push(body.clone());
                        RecordKind::WithBody { inst, body }
                    }
                    None => no_body("no MIR body available"),
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
    let sig = inst.ty().kind().fn_sig()?.skip_binder();
    let (output, inputs) = sig.inputs_and_output.split_last()?;
    let TyKind::RigidTy(RigidTy::Adt(adt_def, adt_args)) = output.kind() else {
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
        if let TyKind::RigidTy(RigidTy::FnDef(def, args)) = c.const_.ty().kind() {
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
        if let TyKind::RigidTy(RigidTy::FnDef(def, args)) = c.const_.ty().kind() {
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
                sm::FloatTy::F32 | sm::FloatTy::F64 => None,
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
pub fn closure_targets(body: &mir::Body) -> Vec<(ClosureDef, GenericArgs)> {
    fn walk_args(args: &GenericArgs, out: &mut Vec<(ClosureDef, GenericArgs)>) {
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

    fn walk_ty(ty: &Ty, out: &mut Vec<(ClosureDef, GenericArgs)>) {
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Closure(def, args)) => {
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
            | TyKind::RigidTy(RigidTy::Coroutine(_, args))
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
    let mut operand_ty = |op: &Operand, out: &mut Vec<(ClosureDef, GenericArgs)>| {
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
                Rvalue::Aggregate(AggregateKind::Closure(def, args), ops) => {
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
                Rvalue::Use(op, _)
                | Rvalue::Repeat(op, _)
                | Rvalue::UnaryOp(_, op) => operand_ty(op, &mut out),
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
