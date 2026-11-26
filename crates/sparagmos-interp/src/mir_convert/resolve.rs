//! Callee resolution: demand consultations, resolution rounds and the per-body conversion drivers.

use std::collections::{HashMap, HashSet};

use sparagmos_mir as mir;

use crate::ir;

use super::converter::*;
use super::drop_glue::DropGlueBuilder;
use super::errors::*;
use super::program::*;


/// The key each function of `program` is stored under in the
/// converted program.
///
/// A function's key is its name — except when several functions
/// share one name. That is the case for anonymous `const _` items:
/// the driver names every one of them by the same item path (a
/// crate's `::_`), while their bodies are distinct items. It is
/// also the case for a `--test` crate's two `main` functions —
/// the user's and the generated harness entry — which share the
/// definition name `<crate>::main`. Bodies
/// sharing a name are disambiguated by their `DefId`, which is
/// also how every reference to them resolves (call and constant
/// operands name a `DefId`, never the shared display name), so a
/// reference still finds exactly its own body and the name-keyed
/// tables stay coherent. A shared name whose bodies cannot be
/// told apart by `DefId` — a missing id, or the same id twice —
/// is a genuine duplicate and fails conversion as a whole.
/// If the recorded body `f` is a closure body that the driver
/// collected through an `Fn`-trait call instance — recognisable by
/// its recorded name being one of the `Fn` / `FnMut` / `FnOnce`
/// call methods (either `std::` or `core::` spelling, with or
/// without the driver's demand-context suffix) and its first
/// argument being the closure environment — return the `DefId` of
/// the closure it is the body of. The driver resolves such an
/// instance straight to the closure's body and records it under
/// the instance's own key, so without this registration the
/// closure's `DefId` names no body and every reference to the
/// closure value falls back to its display name.
///
/// A body that is the call *shim* itself is not the closure body:
/// it takes the environment plus the call's argument tuple as its
/// two arguments (the second argument's type is the closure
/// signature's input tuple) and forwards to the closure body, so
/// it is recognised by that shape and skipped.
/// Whether a body is a closure's own body (as opposed to the
/// `Fn`-trait method's shim or a genuine trait implementation):
/// its first parameter is the closure environment type, and it is
/// not shim-shaped — the shim takes the environment plus the whole
/// argument tuple, matching the closure signature's single tuple
/// input, while the closure's own body takes the environment and
/// the unpacked parameters.
pub(super) fn body_is_closure_body(body: &mir::Body) -> bool {
    let Some(decl) = body.local_decls.get(1) else {
        return false;
    };
    let env_ty = match &decl.ty {
        mir::Ty::Ref(_, inner, _) => inner.as_ref(),
        other => other,
    };
    let mir::Ty::Closure(_, args) = env_ty else {
        return false;
    };
    if body.arg_count == 2 {
        let sig = args.iter().find_map(|t| match t {
            mir::Ty::FnPtr(sig) => Some(sig),
            _ => None,
        });
        if let (Some(sig), Some(d2)) = (sig, body.local_decls.get(2)) {
            if sig.inputs.len() == 1 && sig.inputs[0] == d2.ty {
                return false;
            }
        }
    }
    true
}

/// Whether an `Fn`-trait call's `Self` type denotes a closure
/// or a trait object (possibly behind a reference or pointer):
/// callables the `CallTrait` dispatch resolves from the
/// receiver value. A concrete ADT `Self` instead names a
/// hand-written `Fn` impl, whose collected shim body the
/// direct resolution should keep.
/// The coroutine-closure aggregate that creates local `local`'s
/// value in `body`, if the body creates it: the closure's
/// definition and the types of its captures (the upvars tuple of
/// the aggregate's generic arguments). A coroutine-closure type
/// is recorded by the driver as the unit placeholder (its kind
/// is unaskable through Stable MIR), so the aggregate that
/// creates the value is the one place a body's conversion can
/// learn what the value is.
pub(super) fn coroutine_closure_creation(
    body: &mir::Body,
    local: usize,
) -> Option<(mir::DefId, Vec<mir::Ty>)> {
    for bb in &body.basic_blocks {
        for stmt in &bb.statements {
            let mir::StatementKind::Assign(pair) = &stmt.kind else {
                continue;
            };
            let (place, rv) = &**pair;
            if place.local.index() != local || !place.projection.is_empty() {
                continue;
            }
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
            let upvars = match generic_args.iter().rev().nth(1) {
                Some(mir::Ty::Tuple(ts)) => ts.clone(),
                _ => Vec::new(),
            };
            return Some((*def, upvars));
        }
    }
    None
}

/// The per-capture "taken by reference" flags of an async
/// closure, from its creation aggregate's generic arguments:
/// the upvars tuple (second-to-last) gives the environment's
/// field types, and the by-reference signature (last, a
/// fn-pointer type whose output is the tuple of the types the
/// closure's coroutine borrows for a call) gives each capture's
/// borrowed type. A capture whose borrowed type is a reference
/// TO its environment type was taken by value - the call borrows
/// the environment field itself. Any other capture was taken by
/// reference - the environment field is itself a reference, and
/// the call reborrows it. `None` when the arguments do not have
/// the expected shape.
pub(super) fn async_capture_by_ref(generic_args: &[mir::Ty]) -> Option<Vec<bool>> {
    let upvars = match generic_args.iter().rev().nth(1) {
        Some(mir::Ty::Tuple(ts)) => ts,
        _ => return None,
    };
    let borrowed = match generic_args.last() {
        Some(mir::Ty::FnPtr(sig)) => match &*sig.output {
            mir::Ty::Tuple(ts) => ts,
            _ => return None,
        },
        _ => return None,
    };
    if upvars.len() != borrowed.len() {
        return None;
    }
    Some(
        upvars
            .iter()
            .zip(borrowed.iter())
            .map(|(env_ty, borrowed_ty)| {
                !matches!(borrowed_ty, mir::Ty::Ref(_, inner, _) if &**inner == env_ty)
            })
            .collect(),
    )
}

pub(super) fn fn_self_is_closure_or_dyn(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::Closure(..) | mir::Ty::Dynamic(..) => true,
        mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => {
            fn_self_is_closure_or_dyn(inner)
        }
        // An ADT `Self` wraps the callable in its first type
        // argument (`Box<dyn FnOnce>` — with its allocator
        // argument behind it — is the `Self` of its `call_once`
        // call, `Box<closure>` of a boxed closure's): look
        // through it.
        mir::Ty::Adt(_, args) => match args.first() {
            Some(first) => fn_self_is_closure_or_dyn(first),
            None => false,
        }
        _ => false,
    }
}

pub(super) fn closure_body_alias(f: &MirFunction) -> Option<mir::DefId> {
    let base = f.name.split(" [").next().unwrap_or(&f.name);
    if !matches!(
        base,
        "std::ops::Fn::call"
            | "core::ops::Fn::call"
            | "std::ops::FnMut::call_mut"
            | "core::ops::FnMut::call_mut"
            | "std::ops::FnOnce::call_once"
            | "core::ops::FnOnce::call_once"
    ) {
        return None;
    }
    let env_ty = &f.body.local_decls.get(1)?.ty;
    let env_ty = match env_ty {
        mir::Ty::Ref(_, inner, _) => inner.as_ref(),
        other => other,
    };
    let mir::Ty::Closure(def, args) = env_ty else {
        return None;
    };
    if f.body.arg_count == 2 {
        // The shim shape: (environment, argument tuple).
        let sig = args.iter().find_map(|t| match t {
            mir::Ty::FnPtr(sig) => Some(sig),
            _ => None,
        })?;
        if sig.inputs.len() == 1 && sig.inputs[0] == f.body.local_decls.get(2)?.ty {
            return None;
        }
    }
    Some(*def)
}

/// One body's resolution consultations: the keys whose membership in
/// a round's resolution inputs can change how the body converts.
///
/// A body's conversion consults the round inputs only through the
/// call operands in its terminators — `resolve_body_callee` tests
/// the direct callee's key against the unconverted set,
/// `alloc_shim_route` and `link_symbol_route` test their keys the
/// same way, and `resolve_impl_callee` tests the impl candidate's
/// key against the clean-target set (the exhaustive read sites of
/// `Converter::unconverted` / `Converter::impl_targets`). Every one
/// of those keys is a deterministic function of the call operand
/// alone, computed from tables that do not change between rounds.
/// So a body's converted form can only change when one of these
/// keys' answers changes — the basis of the demand-driven worklist
/// in `lazy_resolution_rounds`.
#[derive(Default)]
struct ResolutionConsult {
    /// Keys tested against the round's unconverted set.
    unconverted_keys: HashSet<String>,
    /// Keys tested against the round's clean-target set.
    target_keys: HashSet<String>,
}

/// Compute one body's resolution consultations under `conv` (any
/// converter over the same program; the key computations do not
/// depend on the round state). `impl_cache` memoises
/// `impl_callee_cached` across bodies.
fn body_consultations(
    conv: &Converter,
    func: &MirFunction,
    impl_cache: &mut HashMap<String, Option<String>>,
) -> ResolutionConsult {
    let mut consult = ResolutionConsult::default();
    for bb in &func.body.basic_blocks {
        let Some(term) = &bb.terminator else {
            continue;
        };
        let mir::TerminatorKind::Call { func: callee, .. } = &term.kind else {
            continue;
        };
        if let Some(key) = conv.resolve_callee(callee) {
            consult.unconverted_keys.insert(key);
        }
        if let Some(key) = conv.alloc_shim_key(callee) {
            consult.unconverted_keys.insert(key);
        }
        if let Some(key) = conv.link_symbol_key(callee) {
            consult.unconverted_keys.insert(key);
        }
        if let Some(key) = impl_callee_cached(conv, callee, impl_cache) {
            consult.target_keys.insert(key);
        }
    }
    consult
}

/// `Converter::find_impl_callee` memoised for the consultation
/// scan. The computation is a deterministic function of the
/// callee's name, its `Self` type's concrete name and its generic
/// arguments' rendered tags, and the same callee recurs across
/// thousands of bodies, while a miss costs a scan of every body
/// key — so the cache key joins exactly those determinants (the
/// separator cannot occur in any of them).
fn impl_callee_cached(
    conv: &Converter,
    callee: &mir::Operand,
    cache: &mut HashMap<String, Option<String>>,
) -> Option<String> {
    let (name, tags, raw_generic_args) = conv.resolve_external_callee(callee)?;
    let self_name = raw_generic_args
        .first()
        .map(|t| conv.concrete_ty_name(t))
        .unwrap_or_default();
    let mut key = name;
    key.push('\u{1}');
    key.push_str(&self_name);
    for tag in &tags {
        key.push('\u{1}');
        key.push_str(tag);
    }
    cache
        .entry(key)
        .or_insert_with(|| conv.find_impl_callee(callee))
        .clone()
}

/// The answers a body's consultations receive from one round's
/// inputs, as sorted vectors: the consulted unconverted keys that
/// are unconverted, and the consulted target keys that are clean
/// targets. A body is stale in a later round exactly when these
/// answers differ from the ones its current form was converted
/// under.
type ConsultAnswers = (Vec<String>, Vec<String>);

fn consult_answers(
    consult: &ResolutionConsult,
    targets: &HashSet<String>,
    unconverted: &HashMap<String, String>,
) -> ConsultAnswers {
    let mut unconverted_hits: Vec<String> = consult
        .unconverted_keys
        .iter()
        .filter(|key| unconverted.contains_key(*key))
        .cloned()
        .collect();
    unconverted_hits.sort();
    let mut target_hits: Vec<String> = consult
        .target_keys
        .iter()
        .filter(|key| targets.contains(*key))
        .cloned()
        .collect();
    target_hits.sort();
    (unconverted_hits, target_hits)
}

/// Demand-driven variant of the resolution rounds, used by the run
/// entry points (Issue #426).
///
/// The eager rounds re-convert every body in every round, but a
/// body's converted form depends on the round inputs only through
/// its own consultations (see `ResolutionConsult`), so this
/// worklist re-converts a body only when the answers to its
/// consultations changed since its last conversion, and stops when
/// no body is stale. The fixpoint reached is the eager one: when
/// the loop stops, every body's form reflects the final round
/// inputs, exactly as after a full round whose inputs no longer
/// change. A failed re-conversion keeps the body's previous form,
/// mirroring the eager merge, and still records the attempt's
/// answers — conversion under the same answers is deterministic,
/// so the body would fail again identically.
fn lazy_resolution_rounds(
    program: &MirProgram,
    keys: &[String],
    probe: &Converter,
    out: &mut ir::Program,
    all_static_seeds: &mut HashMap<String, (u8, ir::Constant)>,
) {
    let mut impl_cache: HashMap<String, Option<String>> = HashMap::new();
    let consultations: Vec<ResolutionConsult> = program
        .functions
        .iter()
        .map(|func| body_consultations(probe, func, &mut impl_cache))
        .collect();
    // The answers each body's current form was converted under.
    // Phase 1 ran with resolution off — empty inputs, so the empty
    // answer vectors.
    let mut last_answers: Vec<ConsultAnswers> = vec![(Vec::new(), Vec::new()); keys.len()];
    let mut used: Option<(HashSet<String>, HashMap<String, String>)> = None;
    for _ in 0..MAX_RESOLUTION_ROUNDS {
        let targets = clean_targets(out);
        let unconverted = unconverted_bodies(program, keys, out);
        if used
            .as_ref()
            .is_some_and(|(t, u)| *t == targets && *u == unconverted)
        {
            break;
        }
        if targets.is_empty() && unconverted.is_empty() {
            break;
        }
        used = Some((targets.clone(), unconverted.clone()));
        let stale: Vec<usize> = (0..keys.len())
            .filter(|&i| {
                consult_answers(&consultations[i], &targets, &unconverted) != last_answers[i]
            })
            .collect();
        if stale.is_empty() {
            break;
        }
        let conv =
            Converter::with_resolution(program, keys, targets.clone(), unconverted.clone());

        for &i in &stale {
            last_answers[i] = consult_answers(&consultations[i], &targets, &unconverted);
            if let Ok(body) = conv.convert_body(&program.functions[i], &keys[i]) {
                out.unconverted.remove(&keys[i]);
                out.functions.insert(keys[i].clone(), body);
            }
        }
        merge_static_seeds(&mut *all_static_seeds, &conv);
    }
}

/// The eager resolution rounds of `convert_all_with`: re-convert
/// every body against each round's resolution inputs until the
/// inputs stop changing. The convert and debug entry points use
/// this path unchanged.
fn eager_resolution_rounds(
    program: &MirProgram,
    keys: &[String],
    out: &mut ir::Program,
    all_static_seeds: &mut HashMap<String, (u8, ir::Constant)>,
) {
    let mut used: Option<(HashSet<String>, HashMap<String, String>)> = None;
    for _ in 0..MAX_RESOLUTION_ROUNDS {
        let targets = clean_targets(out);
        let unconverted = unconverted_bodies(program, keys, out);
        if used
            .as_ref()
            .is_some_and(|(t, u)| *t == targets && *u == unconverted)
        {
            break;
        }
        if targets.is_empty() && unconverted.is_empty() {
            break;
        }
        let conv = Converter::with_resolution(
            program,
            &keys,
            targets.clone(),
            unconverted.clone(),
        );
        let (resolved, _) = convert_each(&conv, program, &keys);
        merge_static_seeds(&mut *all_static_seeds, &conv);
        for (key, body) in resolved.functions {
            out.unconverted.remove(&key);
            out.functions.insert(key, body);
        }
        used = Some((targets, unconverted));
    }
}

thread_local! {
    /// Diagnostic counter of body conversions performed on this
    /// thread (see `conversions_performed`).
    pub(super) static BODY_CONVERSIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The number of bodies conversion has (re-)converted on this
/// thread so far, across all entry points. A diagnostic for the
/// resolution rounds' work: the run entry points' demand-driven
/// rounds keep this close to the phase-1 body count, where the
/// eager rounds multiply it by the round count (Issue #426).
pub fn conversions_performed() -> usize {
    BODY_CONVERSIONS.with(|c| c.get())
}

/// [`convert_program`] plus the per-function conversion failures,
/// for the run entry points: the entry function's own failure is a
/// whole-run conversion failure, exactly as before per-body
/// isolation, so callers need the failures beside the program.
pub(super) fn convert_all(
    program: &MirProgram,
) -> Result<(ir::Program, HashMap<String, ConvertError>), ConvertError> {
    convert_all_with(program, false)
}

/// `convert_all` for the run entry points, with demand-driven
/// resolution rounds (see `lazy_resolution_rounds`, Issue #426):
/// the program produced is the eager fixpoint; only the
/// re-conversion work differs.
pub(super) fn convert_all_lazy(
    program: &MirProgram,
) -> Result<(ir::Program, HashMap<String, ConvertError>), ConvertError> {
    convert_all_with(program, true)
}

fn convert_all_with(
    program: &MirProgram,
    lazy: bool,
) -> Result<(ir::Program, HashMap<String, ConvertError>), ConvertError> {
    let keys = program_keys(program)?;
    // First phase: convert every body with trait-impl call
    // resolution off (the historical behaviour), learning which
    // bodies convert at all.
    let conv = Converter::new(program, &keys);
    let (mut out, mut failures) = convert_each(&conv, program, &keys);
    let mut all_static_seeds: HashMap<String, (u8, ir::Constant)> = HashMap::new();
    merge_static_seeds(&mut all_static_seeds, &conv);
    out.union_tables = conv.union_tables.clone();
    out.async_closure_captures = conv.async_closure_captures.clone();
    out.mat_plans = conv.mat_table.clone();
    // The ADT field-type table: each definition's declared field
    // types, reduced to the skeletons place-walk type tracking
    // can follow (see `field_ty_skeleton`), keyed by the same
    // definition name the converted `Ty::Adt` tags carry.
    out.adt_field_tys = program
        .adt_shapes
        .iter()
        .map(|(def, shape)| {
            let variants = shape
                .variants
                .iter()
                .map(|fields| {
                    fields
                        .iter()
                        .map(|ty| conv.field_ty_skeleton(ty))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            (conv.def_display_name(def), variants)
        })
        .collect();
    // Joint fixpoint of conversion, cleanliness and the
    // builtin-served set. Each round recomputes, from the merged
    // program so far:
    //
    // * the *clean* bodies (see `clean_targets`) — the bodies
    //   trait-impl call resolution may target, and
    // * the *unconverted* bodies — the bodies that failed
    //   conversion. A call naming one of them is served another
    //   way when it honestly can be (see `resolve_body_callee`):
    //   by a clean impl body for the same trait call, or by the
    //   builtin model of the callee.
    //
    // and converts every body again with those two sets, merging
    // the results in (a body whose re-conversion fails keeps its
    // previous form; a converted body is never lost). Iterating
    // matters because the sets feed each other: resolving a trait
    // call inside a body can make that body clean, which makes it
    // a resolution target for the next round, and a body withheld
    // only because it called an unconverted body becomes clean
    // once that call is builtin-served. The rounds are monotone —
    // the clean set only grows and the unconverted set only
    // shrinks, because a re-conversion only retargets calls to
    // bodies proven clean in the previous round or replaces an
    // edge to an unconverted body with a neutral builtin call —
    // so the loop reaches a fixpoint: in the final program, every
    // resolution target is clean in that same program, and every
    // diverted call names a body that is unconverted in that same
    // program. The round cap is a defensive bound only.
    if lazy {
        lazy_resolution_rounds(program, &keys, &conv, &mut out, &mut all_static_seeds);
    } else {
        eager_resolution_rounds(program, &keys, &mut out, &mut all_static_seeds);
    }
    failures.retain(|key, _| !out.functions.contains_key(key));
    out.adt_layouts = program.adt_layouts.clone();
    out.adt_dst_layouts = program.adt_dst_layouts.clone();
    // The vtable identities the driver recorded as it minted
    // vtable-pointer constants, with each concrete type's name
    // rendered and its drop glue built the way a
    // conversion-baked `DynRef` constant carries them.
    out.vtable_identities = program
        .vtable_identities
        .iter()
        .map(|(id, (concrete, traits))| {
            (
                *id,
                ir::VTableIdentity {
                    concrete: conv.concrete_ty_name(concrete),
                    traits: traits.clone(),
                    glue: DropGlueBuilder::build(&conv, concrete),
                },
            )
        })
        .collect();
    out.adt_field_layouts = program.adt_field_layouts.clone();
    out.adt_tag_layouts = program.adt_tag_layouts.clone();
    out.adt_niches = program.adt_niches.clone();
    out.fn_display_names = program.fn_display_names.clone();
    out.crate_overflow_checks = program.crate_overflow_checks.clone();
    out.adt_debug_shapes = program.adt_debug_shapes.clone();
    out.needs_drop_verdicts = program.needs_drop_verdicts.clone();
    out.env_debug_shapes = program.env_debug_shapes.clone();
    out.alloc_shims = conv.alloc_shims.clone();
    out.panic_handler = program.panic_handler.clone();
    out.links_std = program.links_std;
    merge_static_seeds(&mut all_static_seeds, &conv);
    out.static_seeds = all_static_seeds
        .into_iter()
        .map(|(name, (_, seed))| (name, seed))
        .collect();
    // The thread-locals table: each recorded static's initializer,
    // converted with its declared type as context and keyed by the
    // same static key the `ThreadLocalRef` rvalues carry. A seed
    // that does not convert is dropped; evaluating that static's
    // `ThreadLocalRef` then fails with an explicit error.
    out.thread_locals = program
        .thread_locals
        .iter()
        .filter_map(|(def, defn)| {
            let seed = conv.convert_const_value(&defn.init, Some(&defn.ty)).ok()?;
            let seed = conv.expand_uninit_constant(&seed, &defn.ty, 0);
            Some((conv.static_key(def), seed))
        })
        .collect();
    // The driver-recorded seeds of statics no converted body
    // references (see `MirProgram::extra_static_seeds`): each
    // initializer, converted with its declared type as
    // context, enters the seed table under the static's name.
    // A name a converted reference also seeded keeps that
    // seed — it carries the reference path's own expansion —
    // and a seed that does not convert is dropped; naming
    // the static then fails with the explicit no-seed error.
    for (name, defn) in program.extra_static_seeds.iter() {
        if out.static_seeds.contains_key(name) {
            continue;
        }
        if let Ok(seed) = conv.convert_const_value(&defn.init, Some(&defn.ty)) {
            let seed = conv.expand_uninit_constant(&seed, &defn.ty, 0);
            out.static_seeds.insert(name.clone(), seed);
        }
    }
    // The statics' declared types, keyed by the same static key
    // the `StaticRef` values carry, for the interpreter's
    // pointer-offset denomination over statics.
    out.static_tys = program
        .static_tys
        .iter()
        .filter_map(|(def, ty)| {
            let ty = conv.convert_ty(ty, "a static's declared type").ok()?;
            Some((conv.static_key(def), ty))
        })
        .collect();
    // The statics' declared types as rendered type tags, keyed
    // like the type table above: the debugger interface's
    // record of each static's full instance type (see
    // `ir::Program::static_debug_tags`).
    out.static_debug_tags = program
        .static_tys
        .iter()
        .map(|(def, ty)| {
            (
                conv.static_key(def),
                crate::builtins::ty_tag(ty, conv.names),
            )
        })
        .collect();
    // The definition-location stamps, keyed by the program key
    // of the body each belongs to (the `DefId` a function's
    // calls and constants resolve through), in the same
    // constant form the driver-appended call-site stamps take.
    for (f, key) in program.functions.iter().zip(keys.iter()) {
        let Some(def) = f.def_id else { continue };
        let Some(stamp) = program.track_caller_defs.get(&def) else {
            continue;
        };
        if let Ok(c) = conv.convert_const_value(stamp, None) {
            out.track_caller_defs.insert(key.clone(), c);
        }
    }
    out.fn_shim_locations = [
        program.fn_shim_locations[0]
            .as_ref()
            .and_then(|v| conv.convert_const_value(v, None).ok()),
        program.fn_shim_locations[1]
            .as_ref()
            .and_then(|v| conv.convert_const_value(v, None).ok()),
        program.fn_shim_locations[2]
            .as_ref()
            .and_then(|v| conv.convert_const_value(v, None).ok()),
    ];
    mark_caller_location_bodies(&mut out);
    Ok((out, failures))
}

/// Mark the `#[track_caller]` bodies of a converted program: the
/// driver appends the caller-location stamp as one extra
/// argument to every direct call of such a function, so a body
/// whose converted callers all pass `arg_count + 1` arguments is
/// a track-caller body. Each marked body gains a synthetic final
/// local (typed as the location reference) that the interpreter
/// binds the stamp to (see `Body::caller_location_local`).
fn mark_caller_location_bodies(program: &mut ir::Program) {
    // Bodies the driver recorded a definition-location stamp
    // for are track-caller bodies by the compiler's own
    // resolution, whether or not any converted direct call
    // passes a stamp: a body reached only through a function
    // pointer or a vtable is never called directly at all.
    let mut marked: Vec<String> = program
        .track_caller_defs
        .keys()
        .filter(|k| program.functions.contains_key(*k))
        .cloned()
        .collect();
    for body in program.functions.values() {
        for block in &body.blocks {
            if let ir::TerminatorKind::Call { func, args, .. } = &block.terminator {
                if let Some(callee) = program.functions.get(func) {
                    if args.len() == callee.arg_count + 1
                        && callee.caller_location_local.is_none()
                        && !callee_takes_va_list(callee)
                        && !marked.contains(func)
                    {
                        marked.push(func.clone());
                    }
                }
            }
        }
    }
    for name in marked {
        let Some(body) = program.functions.get_mut(&name) else {
            continue;
        };
        let idx = body.local_tys.len();
        body.local_tys.push(ir::Ty::Ref(Box::new(ir::Ty::Adt(
            "std::panic::Location".to_string(),
        ))));
        if !body.zst_locals.is_empty() {
            body.zst_locals.push(None);
        }
        body.caller_location_local = Some(idx);
    }
}

/// Whether the body's last declared parameter is a C variable
/// argument list (`std::ffi::VaList`, spelled through `core`
/// inside core's own bodies): the body is a C-variadic function,
/// so a caller passing `arg_count + 1` values is passing one
/// tail argument, not a caller-location stamp, and the
/// track-caller heuristic in [`mark_caller_location_bodies`]
/// must not claim it.
fn callee_takes_va_list(body: &ir::Body) -> bool {
    body.arg_count >= 1
        && matches!(
            body.local_tys.get(body.arg_count),
            Some(ir::Ty::Adt(name))
                if name == "std::ffi::VaList" || name == "core::ffi::VaList"
        )
}

/// The defensive bound on the joint-fixpoint rounds in
/// `convert_all` (see there). The rounds are monotone, so the
/// bound is never the reason the loop stops in practice.
const MAX_RESOLUTION_ROUNDS: usize = 16;

/// The bodies of `out` that failed conversion, keyed by program
/// key with the function's plain name (the key itself can carry a
/// `DefId` disambiguation suffix; see `program_keys`). Later
/// fixpoint rounds use the table to serve calls naming an
/// unconverted body another way (see `resolve_body_callee`).
fn unconverted_bodies(
    program: &MirProgram,
    keys: &[String],
    out: &ir::Program,
) -> HashMap<String, String> {
    program
        .functions
        .iter()
        .zip(keys)
        .filter(|(_, key)| out.unconverted.contains_key(*key))
        .map(|(f, key)| (key.clone(), f.name.clone()))
        .collect()
}

/// The bodies of a converted program that trait-impl call
/// resolution may target in the next fixpoint round (see
/// `convert_all`):
/// the *clean* bodies — running one can never hit an unmodelled
/// construct or a body that is not itself clean.
///
/// A body is unclean when
/// * it contains an unsupported statement or terminator (an
///   unsupported rvalue rides inside an assignment statement),
/// * a direct `Call` edge names a body that did not convert, or a
///   body that is itself unclean, or
/// * its drop glue (on a `Drop` terminator or a `mem::drop`
///   builtin call) calls an unclean destructor, or contains glue
///   conversion could not account for (`DropPlan::Unknown` /
///   `DropPlan::Untyped`: executing it judges or fails on the
///   value's own shape, which is exactly what resolution must not
///   newly expose a caller to).
///
/// `CallBuiltin` edges are neutral (builtins are the fallback
/// model itself), and so are `CallTrait` / `CallIndirect` edges
/// and `Call`s whose target name is no program body at all: a
/// vtable-dispatch call names the trait method, and the concrete
/// target is chosen from the receiver's value at run time.
/// Computed as a fixpoint: uncleanliness propagates backwards
/// along call and destructor edges until nothing changes.
fn clean_targets(program: &ir::Program) -> HashSet<String> {
    // Direct edges from each body to the program bodies it can
    // enter, plus the bodies that are unclean on their own.
    let mut edges: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut unclean: HashSet<&str> = HashSet::new();
    for (key, body) in &program.functions {
        let mut targets: Vec<&str> = Vec::new();
        for block in &body.blocks {
            for stmt in &block.statements {
                match &stmt.kind {
                    ir::StatementKind::Unsupported(_)
                    | ir::StatementKind::Assign(_, ir::Rvalue::Unsupported(_)) => {
                        unclean.insert(key.as_str());
                    }
                    _ => {}
                }
            }
            match &block.terminator {
                ir::TerminatorKind::Unsupported(_) => {
                    unclean.insert(key.as_str());
                }
                ir::TerminatorKind::Call { func, .. } => {
                    // An edge only when the name is a program body
                    // (converted, or recorded as unconverted); a
                    // name in neither table is a vtable-dispatch
                    // call, whose target the receiver's value
                    // chooses at run time.
                    if program.functions.contains_key(func)
                        || program.unconverted.contains_key(func)
                    {
                        targets.push(func.as_str());
                    }
                }
                ir::TerminatorKind::Drop { glue, .. } => {
                    if !drop_glue_edges(glue, &mut targets) {
                        unclean.insert(key.as_str());
                    }
                }
                ir::TerminatorKind::CallBuiltin {
                    drop_glue: Some(glue),
                    ..
                } => {
                    if !drop_glue_edges(glue, &mut targets) {
                        unclean.insert(key.as_str());
                    }
                }
                _ => {}
            }
        }
        edges.insert(key.as_str(), targets);
    }
    // Propagate: a body with an edge to an unconverted or unclean
    // body is itself unclean.
    loop {
        let mut changed = false;
        for (key, targets) in &edges {
            if unclean.contains(key) {
                continue;
            }
            if targets
                .iter()
                .any(|t| !program.functions.contains_key(*t) || unclean.contains(t))
            {
                unclean.insert(key);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    program
        .functions
        .keys()
        .filter(|k| !unclean.contains(k.as_str()))
        .cloned()
        .collect()
}

/// Whether a drop glue's plans call the destructor body `key`
/// anywhere: through `Seq` / `At` / `Each` / `Fields` /
/// `Variants` / reference-count steps and the glue's ADT table.
pub(super) fn glue_calls_dtor(glue: &ir::DropGlue, key: &str) -> bool {
    fn walk(
        plan: &ir::DropPlan,
        glue: &ir::DropGlue,
        key: &str,
        seen: &mut std::collections::HashSet<usize>,
    ) -> bool {
        use ir::DropPlan as P;
        match plan {
            P::CallDtor { func } => func == key,
            P::Seq(plans) => plans.iter().any(|p| walk(p, glue, key, seen)),
            P::At { plan, .. } => walk(plan, glue, key, seen),
            P::Each { elem } => walk(elem, glue, key, seen),
            P::Elided { plan, .. } => walk(plan, glue, key, seen),
            P::Fields { fields, .. } => {
                fields.iter().any(|p| walk(p, glue, key, seen))
            }
            P::Variants { variants } => {
                variants.iter().any(|p| walk(p, glue, key, seen))
            }
            P::RcHandleDrop { value } => walk(value, glue, key, seen),
            P::BoxPointee { pointee } => walk(pointee, glue, key, seen),
            P::DynOrPlain { plain } => walk(plain, glue, key, seen),
            P::Adt { index } => {
                seen.insert(*index)
                    && glue
                        .adts
                        .get(*index)
                        .is_some_and(|p| walk(p, glue, key, seen))
            }
            _ => false,
        }
    }
    let mut seen = std::collections::HashSet::new();
    walk(&glue.root, glue, key, &mut seen)
}

/// Walk one terminator's drop glue, pushing the destructor bodies
/// it can call onto `targets`. Returns `false` when the glue
/// contains a plan conversion could not account for
/// (`DropPlan::Unknown` / `DropPlan::Untyped`).
fn drop_glue_edges<'a>(glue: &'a ir::DropGlue, targets: &mut Vec<&'a str>) -> bool {
    fn walk<'a>(
        plan: &'a ir::DropPlan,
        adts: &'a [ir::DropPlan],
        targets: &mut Vec<&'a str>,
        seen: &mut HashSet<usize>,
    ) -> bool {
        match plan {
            ir::DropPlan::Trivial => true,
            // A dyn value's glue is chosen from the value at run
            // time, like a vtable call's target: no static edges.
            ir::DropPlan::Dyn => true,
            // Releasing a RefCell borrow calls no destructor body.
            ir::DropPlan::RefCellRelease { .. } => true,
            ir::DropPlan::ChannelHandleDrop { .. } => true,
            // Closing a file handle calls no destructor body.
            ir::DropPlan::FileHandleDrop => true,
            // Closing a socket handle calls no destructor body.
            ir::DropPlan::SocketHandleDrop => true,
            // Removing a string drain's range calls no
            // destructor body.
            ir::DropPlan::StringDrainDrop => true,
            // Releasing a weak reference calls no destructor body.
            ir::DropPlan::RcWeakRelease => true,
            // A waker's release is chosen from the value at run
            // time (a handle release, or the vtable entry the
            // value itself carries): no static edges.
            ir::DropPlan::WakerDrop => true,
            // An `io::Error` representation's release is judged
            // from the word at run time, and the payload's glue is
            // the trait object's own (carried by the value, like
            // `Dyn`): no static edges.
            ir::DropPlan::IoErrorRepr => true,
            ir::DropPlan::RcHandleDrop { value } => walk(value, adts, targets, seen),
            // The dyn branch is chosen from the value at run time
            // (no static edges, like `Dyn` itself); the plain
            // branch's edges are the subplan's.
            ir::DropPlan::DynOrPlain { plain } => walk(plain, adts, targets, seen),
            // The pointee's storage is judged from the value at
            // run time; the glue that runs is the subplan's.
            ir::DropPlan::BoxPointee { pointee } => walk(pointee, adts, targets, seen),
            // Judged from the value at run time; calls no
            // destructor body.
            ir::DropPlan::Dynamic => true,
            ir::DropPlan::Unknown | ir::DropPlan::Untyped => false,
            ir::DropPlan::Seq(plans) => plans.iter().all(|p| walk(p, adts, targets, seen)),
            ir::DropPlan::At { plan, .. } => walk(plan, adts, targets, seen),
            ir::DropPlan::CallDtor { func } | ir::DropPlan::GuardDtor { func } => {
                targets.push(func.as_str());
                true
            }
            ir::DropPlan::Each { elem } => walk(elem, adts, targets, seen),
            ir::DropPlan::Elided { plan, .. } => walk(plan, adts, targets, seen),
            ir::DropPlan::Fields { fields, .. } => {
                fields.iter().all(|p| walk(p, adts, targets, seen))
            }
            ir::DropPlan::Variants { variants } => {
                variants.iter().all(|p| walk(p, adts, targets, seen))
            }
            // An ADT plan can refer to itself (recursive types);
            // `seen` keeps the walk finite — a revisited plan's
            // edges were collected by the first visit.
            ir::DropPlan::Adt { index } => {
                if !seen.insert(*index) {
                    return true;
                }
                match adts.get(*index) {
                    Some(plan) => walk(plan, adts, targets, seen),
                    None => false,
                }
            }
        }
    }
    let mut seen = HashSet::new();
    if !walk(&glue.root, &glue.adts, targets, &mut seen) {
        return false;
    }
    // Table entries the root never referenced are walked too, so
    // their destructor edges and unaccounted plans count as well.
    for (index, plan) in glue.adts.iter().enumerate() {
        if seen.insert(index) && !walk(plan, &glue.adts, targets, &mut seen) {
            return false;
        }
    }
    true
}

/// Convert every body of `program` with `conv`, isolating failures
/// per body (see [`convert_program`]).
fn convert_each(
    conv: &Converter,
    program: &MirProgram,
    keys: &[String],
) -> (ir::Program, HashMap<String, ConvertError>) {
    let mut out = ir::Program::new();
    out.source_files = program.source_files.clone();
    out.coverage = program.coverage.clone();
    let mut failures: HashMap<String, ConvertError> = HashMap::new();
    for (func, key) in program.functions.iter().zip(keys) {
        match conv.convert_body(func, key) {
            Ok(body) => {
                out.add_function(body);
            }
            Err(e) => {
                out.add_unconverted(key.clone(), e.to_string());
                failures.insert(key.clone(), e);
            }
        }
    }
    (out, failures)
}
