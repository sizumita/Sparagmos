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
//! - The Stable MIR bodies obtained during the walk are returned
//!   alongside the converted bodies ([`CollectedFns::walked_bodies`]):
//!   targets resolving to local items are skipped here, but a
//!   collected foreign body can be the only place that demands a
//!   monomorphic instance of a generic local definition, and the
//!   caller's local-instance collection scans these bodies for
//!   exactly those demands.

use std::collections::{HashMap, HashSet, VecDeque};

use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::mir::{self, Operand, TerminatorKind};
use rustc_public::ty::{GenericArgs, RigidTy, TyKind};
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
        for (def, args) in call_targets(&body) {
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
                    _ => no_body("no MIR body available"),
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
            RecordKind::WithBody { body, .. } => match conv.convert_body(body) {
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
        } else if matches!(rec.kind, RecordKind::WithBody { .. }) {
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
                } else if matches!(records[cur].kind, RecordKind::WithBody { .. }) {
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
        limit_reached,
        walked_bodies,
    };
    for (i, rec) in records.iter().enumerate() {
        let name = display_name(i, rec);
        let (blocks, locals, arg_count) = match &rec.kind {
            RecordKind::WithBody { body, .. } => {
                (body.blocks.len(), body.locals().len(), body.arg_locals().len())
            }
            RecordKind::NoBody { .. } => (0, 0, 0),
        };
        let outcome = match (&rec.kind, &converted[i], &failed[i]) {
            (RecordKind::WithBody { inst, .. }, Some(_), _) => {
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
            | sm::Ty::Dynamic(..)
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
