//! Demand-driven body acquisition: drop, clone, wake and dyn instances demanded by converted bodies.

use super::*;

/// Convert one local body (an item-loop body, or a constant body
/// obtained from the item itself) and record the outcome; a
/// successfully converted body also becomes a reachability seed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn convert_local_body(
    conv: &mut Converter,
    report: &mut ConversionReport,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    seed_bodies: &mut Vec<rustc_public::mir::Body>,
    name: String,
    def_id: sparagmos_mir::DefId,
    body: rustc_public::mir::Body,
) {
    let blocks = body.blocks.len();
    let locals = body.locals().len();
    let arg_count = body.arg_locals().len();
    match convert_body_isolated(&mut *conv, &body) {
        Ok(converted) => {
            seed_bodies.push(body);
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
                blocks,
                locals,
                arg_count,
                outcome: FnOutcome::Skipped { reasons },
            });
        }
    }
}

/// The local crate's `Drop` impl methods, keyed by their full
/// instance-style name (`<crate::Type as std::ops::Drop>::drop`).
/// The trait method declaration itself is not among a crate's
/// function definitions, but every impl method is, and resolving
/// an impl method with the self type's generic arguments yields
/// the monomorphic destructor instance directly.
fn local_drop_impl_methods() -> std::collections::HashMap<String, rustc_public::ty::FnDef> {
    drop_impl_methods(rustc_public::local_crate().fn_defs())
}

/// The `Drop::drop` methods of one crate's function definitions,
/// keyed by the self type's path without its parameter list
/// (see `local_drop_impl_methods`).
fn drop_impl_methods(
    fn_defs: Vec<rustc_public::ty::FnDef>,
) -> std::collections::HashMap<String, rustc_public::ty::FnDef> {
    let mut out = std::collections::HashMap::new();
    for f in fn_defs {
        let name = f.name();
        if !(name.ends_with(" as std::ops::Drop>::drop")
            || name.ends_with(" as core::ops::Drop>::drop")
            || name.ends_with(" as std::ops::Drop>::pin_drop")
            || name.ends_with(" as core::ops::Drop>::pin_drop"))
        {
            continue;
        }
        // `<SELF as ...::Drop>::drop`: the key is the self type's
        // canonical path (see `drop_type_key`). The self type is
        // the part of the leading balanced group before its
        // top-level ` as ` separator -- splitting at the first
        // ` as ` in the raw name is wrong for a type defined
        // inside another impl's method (a cleanup guard inside a
        // `drop` method is the common shape): its name is
        // `<<OUTER<..> as ..>::method::Local<..> as ..>::drop`,
        // whose first ` as ` belongs to the OUTER type's group.
        let Some(inner_end) = balanced_group_end(&name, 0) else {
            continue;
        };
        let inner = &name[1..inner_end];
        let Some(as_pos) = top_level_as_pos(inner) else {
            continue;
        };
        let base = drop_type_key(&inner[..as_pos]);
        out.entry(base).or_insert(f);
    }
    out
}

/// The byte index of the `>` matching the `<` at byte `open`
/// in `s`, or `None` when the group never closes.
fn balanced_group_end(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, ch) in s.char_indices() {
        if i < open {
            continue;
        }
        match ch {
            '<' => depth += 1,
            '>' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// The byte position of the last top-level (outside every
/// `<...>` group) ` as ` separator in `s`, or `None` when there
/// is none. A rendered ` as ` separator is the last one at the
/// top level: the trait path after it contains no separator.
fn top_level_as_pos(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut found = None;
    let mut offset = 0usize;
    let mut rest = s;
    while !rest.is_empty() {
        if depth == 0 && rest.starts_with(" as ") {
            found = Some(offset);
            offset += 4;
            rest = &rest[4..];
            continue;
        }
        let ch = rest.chars().next().expect("non-empty");
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ => {}
        }
        offset += ch.len_utf8();
        rest = &rest[ch.len_utf8()..];
    }
    found
}

/// The key under which a type's `Drop` impl is recorded (by
/// `drop_impl_methods`) and looked up (by the demand and
/// certification passes): the type's canonical path, computed
/// identically from the impl method's self-type rendering and
/// from a candidate ADT's rendered name. The canonical path is
/// the rendered path with every `<...>` group removed, except
/// that a `<X as Trait>` group -- the rendering of a type whose
/// path passes through an impl, as for a type defined inside
/// another item's body -- stands for X itself and is replaced
/// by X's canonical path. Both sides render such a nested type
/// through the enclosing impl (the ADT as
/// `<OUTER<..> as ..>::method::Local`, its impl method as
/// `<<OUTER<..> as ..>::method::Local<..> as ..::Drop>::drop`),
/// and only the canonical path makes the two renderings meet,
/// at `OUTER::method::Local`.
fn drop_type_key(rendered: &str) -> String {
    let mut out = String::with_capacity(rendered.len());
    let mut rest = rendered;
    while !rest.is_empty() {
        if rest.starts_with('<') {
            match balanced_group_end(rest, 0) {
                Some(end) => {
                    let inner = &rest[1..end];
                    if let Some(as_pos) = top_level_as_pos(inner) {
                        out.push_str(&drop_type_key(&inner[..as_pos]));
                    }
                    rest = &rest[end + 1..];
                    continue;
                }
                None => {
                    // An unbalanced group: keep the remainder
                    // verbatim rather than invent a key.
                    out.push_str(rest);
                    break;
                }
            }
        }
        let ch = rest.chars().next().expect("non-empty");
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

/// Cap on drop-impl bodies demanded per program, mirroring the
/// reachability collection's cap.
const MAX_DEMANDED_DTOR_BODIES: usize = 256;

/// The type of the place a `Drop` terminator drops, following the
/// place's projections. `None` when a projection's resulting type
/// cannot be determined locally (the demand pass then simply does
/// not see this drop; conversion reports it as before).
fn drop_place_ty(body: &rustc_public::mir::Body, place: &rustc_public::mir::Place) -> Option<Ty> {
    use rustc_public::mir::ProjectionElem as P;
    let mut ty = body.locals().get(place.local)?.ty;
    for elem in &place.projection {
        // The intermediate types are of unknown provenance (a
        // projection can step onto any field type), so their
        // kinds go through the guarded question; an unaskable
        // intermediate type ends the walk with `None`, the same
        // answer the unrecognized shapes below give.
        ty = match elem {
            P::Field(_, t) => *t,
            P::Deref => match sparagmos_driver::convert::guarded_ty_kind(&ty)? {
                TyKind::RigidTy(RigidTy::Ref(_, inner, _)) => inner,
                TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => inner,
                TyKind::RigidTy(RigidTy::Adt(def, ref args)) => {
                    // `Box<T>` dereferences to `T`.
                    if def.name() != "std::boxed::Box" {
                        return None;
                    }
                    args.0.iter().find_map(|a| match a {
                        GenericArgKind::Type(t) => Some(*t),
                        _ => None,
                    })?
                }
                _ => return None,
            },
            P::Index(_) | P::ConstantIndex { .. } | P::Subslice { .. } => {
                match sparagmos_driver::convert::guarded_ty_kind(&ty)? {
                    TyKind::RigidTy(RigidTy::Array(elem, _)) => elem,
                    TyKind::RigidTy(RigidTy::Slice(elem)) => elem,
                    _ => return None,
                }
            }
            P::Downcast(_) => ty,
            P::OpaqueCast(t) => *t,
        };
    }
    Some(ty)
}

/// Every place in `body` whose projection contains a
/// `Downcast` step (see the certification candidate scan in
/// `certify_drop_trivial` / `certify_drop_no_impl`).
fn downcast_places(body: &rustc_public::mir::Body) -> Vec<rustc_public::mir::Place> {
    use rustc_public::mir::{Operand, ProjectionElem, Rvalue, StatementKind, TerminatorKind};
    fn operand_place<'a>(op: &'a Operand, out: &mut Vec<&'a rustc_public::mir::Place>) {
        if let Operand::Copy(place) | Operand::Move(place) = op {
            out.push(place);
        }
    }
    fn rvalue_places<'a>(rv: &'a Rvalue, out: &mut Vec<&'a rustc_public::mir::Place>) {
        match rv {
            Rvalue::Use(op, _)
            | Rvalue::Repeat(op, _)
            | Rvalue::Cast(_, op, _)
            | Rvalue::UnaryOp(_, op) => operand_place(op, out),
            Rvalue::BinaryOp(_, l, r) | Rvalue::CheckedBinaryOp(_, l, r) => {
                operand_place(l, out);
                operand_place(r, out);
            }
            Rvalue::Aggregate(_, ops) => {
                for op in ops {
                    operand_place(op, out);
                }
            }
            Rvalue::Ref(_, _, place)
            | Rvalue::AddressOf(_, place)
            | Rvalue::Len(place)
            | Rvalue::Discriminant(place) => out.push(place),
            _ => {}
        }
    }
    let mut refs: Vec<&rustc_public::mir::Place> = Vec::new();
    for bb in &body.blocks {
        for stmt in &bb.statements {
            match &stmt.kind {
                StatementKind::Assign(place, rv) => {
                    refs.push(place);
                    rvalue_places(rv, &mut refs);
                }
                StatementKind::SetDiscriminant { place, .. } => refs.push(place),
                StatementKind::Intrinsic(intr) => match intr {
                    rustc_public::mir::NonDivergingIntrinsic::Assume(op) => {
                        operand_place(op, &mut refs)
                    }
                    rustc_public::mir::NonDivergingIntrinsic::CopyNonOverlapping(cno) => {
                        operand_place(&cno.src, &mut refs);
                        operand_place(&cno.dst, &mut refs);
                        operand_place(&cno.count, &mut refs);
                    }
                },
                _ => {}
            }
        }
        match &bb.terminator.kind {
            TerminatorKind::Drop { place, .. } => refs.push(place),
            TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } => {
                operand_place(func, &mut refs);
                for arg in args {
                    operand_place(arg, &mut refs);
                }
                refs.push(destination);
            }
            TerminatorKind::SwitchInt { discr, .. } => operand_place(discr, &mut refs),
            TerminatorKind::Assert { cond, .. } => operand_place(cond, &mut refs),
            _ => {}
        }
    }
    refs.into_iter()
        .filter(|place| {
            place
                .projection
                .iter()
                .any(|elem| matches!(elem, ProjectionElem::Downcast(_)))
        })
        .cloned()
        .collect()
}

/// Whether the type tree of `ty` contains an alias type (an
/// associated-type projection, an inherent alias or an opaque
/// type) anywhere the substitution in `drop_candidates` would
/// carry it: ADT and closure arguments, tuple elements, array
/// and slice elements, and reference / raw-pointer pointees.
/// Kinds are asked through the guarded question, and a type
/// whose kind cannot be asked for — or a tree deeper than the
/// cap — counts as containing one: the screen exists to keep
/// the compiler's normalization out of shapes it can fail on,
/// so doubt resolves to not substituting.
fn type_tree_has_alias(ty: &Ty, depth: usize) -> bool {
    if depth >= 8 {
        return true;
    }
    let Some(kind) = sparagmos_driver::convert::guarded_ty_kind(ty) else {
        return true;
    };
    match kind {
        TyKind::Alias(..) => true,
        TyKind::RigidTy(RigidTy::Adt(_, args)) | TyKind::RigidTy(RigidTy::Closure(_, args)) => {
            args.0.iter().any(|a| match a {
                GenericArgKind::Type(t) => type_tree_has_alias(t, depth + 1),
                _ => false,
            })
        }
        TyKind::RigidTy(RigidTy::Tuple(ts)) => {
            ts.iter().any(|t| type_tree_has_alias(t, depth + 1))
        }
        TyKind::RigidTy(RigidTy::Array(elem, _) | RigidTy::Slice(elem)) => {
            type_tree_has_alias(&elem, depth + 1)
        }
        TyKind::RigidTy(RigidTy::Ref(_, inner, _) | RigidTy::RawPtr(inner, _)) => {
            type_tree_has_alias(&inner, depth + 1)
        }
        // A function pointer's signature is part of the type
        // tree for this screen's purpose: folding a substituted
        // field type back to Stable MIR normalizes the aliases
        // inside the signature exactly like the ones in an ADT's
        // arguments (a query cache's fingerprint function type,
        // `fn(&mut StableHashState, &QueryCache::Value) ->
        // Fingerprint`, is the shape that reached the fold
        // through a vtable struct's function-pointer field).
        TyKind::RigidTy(RigidTy::FnPtr(..)) => fn_sig_tree_has_alias(ty, depth),
        TyKind::RigidTy(RigidTy::FnDef(_, args)) => {
            args.0.iter().any(|a| match a {
                GenericArgKind::Type(t) => type_tree_has_alias(t, depth + 1),
                _ => false,
            }) || fn_sig_tree_has_alias(ty, depth)
        }
        // A trait object's projection bindings carry types the
        // same fold normalizes; walk the predicates' arguments
        // and type terms.
        TyKind::RigidTy(RigidTy::Dynamic(bounds, _)) => bounds.iter().any(|b| match &b.value {
            rustc_public::ty::ExistentialPredicate::Trait(t) => t
                .generic_args
                .0
                .iter()
                .any(|a| match a {
                    GenericArgKind::Type(t) => type_tree_has_alias(t, depth + 1),
                    _ => false,
                }),
            rustc_public::ty::ExistentialPredicate::Projection(p) => {
                p.generic_args.0.iter().any(|a| match a {
                    GenericArgKind::Type(t) => type_tree_has_alias(t, depth + 1),
                    _ => false,
                }) || matches!(&p.term, rustc_public::ty::TermKind::Type(t)
                        if type_tree_has_alias(t, depth + 1))
            }
            _ => false,
        }),
        _ => false,
    }
}

/// Whether the signature of a function-shaped type (a function
/// pointer or a function definition) mentions an alias in any
/// input or output type, at the next depth of
/// [`type_tree_has_alias`]. An unreadable signature counts as
/// mentioning one: the screen's safe direction is to skip the
/// substitution.
fn fn_sig_tree_has_alias(ty: &Ty, depth: usize) -> bool {
    let Some(kind) = sparagmos_driver::convert::guarded_ty_kind(ty) else {
        return true;
    };
    let Some(sig) = kind.fn_sig() else {
        return true;
    };
    let sig = sig.skip_binder();
    sig.inputs()
        .iter()
        .any(|t| type_tree_has_alias(t, depth + 1))
        || type_tree_has_alias(&sig.output(), depth + 1)
}

/// Collect every ADT instantiation in the type tree of `ty` —
/// the type itself, its generic arguments, its variants' field
/// types (with the arguments substituted), tuple elements, array
/// elements and closure captures — as drop-impl demand candidates.
/// References and raw pointers are not followed (their pointees
/// are not dropped with the value), and aliases are left alone
/// (their expansion is not available locally).
fn drop_candidates(
    ty: &Ty,
    out: &mut Vec<(Ty, rustc_public::ty::AdtDef, GenericArgs)>,
    seen: &mut std::collections::HashSet<Ty>,
) {
    // The walk reaches every type in the dropped places' trees,
    // including types whose kind rustc_public cannot answer for
    // (coroutine-closure types panic inside `Ty::kind`); the
    // guarded question is the driver's standing convention for
    // types of unknown provenance. An unaskable type contributes
    // no candidates, exactly like the shapes the walk does not
    // descend into.
    let Some(kind) = sparagmos_driver::convert::guarded_ty_kind(ty) else {
        return;
    };
    match kind {
        TyKind::RigidTy(RigidTy::Adt(def, args)) => {
            // The dedup key is the type value itself, never a
            // debug rendering: rendering a `Ty` asks its kind,
            // which panics for the types this walk must tolerate
            // (a coroutine-closure type nested in the arguments).
            // The interned identity is exact for this purpose.
            if !seen.insert(*ty) {
                return;
            }
            out.push((*ty, def, args.clone()));
            for a in &args.0 {
                if let GenericArgKind::Type(t) = a {
                    drop_candidates(t, out, seen);
                }
            }
            for variant in def.variants() {
                for field in variant.fields() {
                    let declared = field.ty();
                    // Substituting the instance's arguments into
                    // a field type makes the compiler fold the
                    // result back to Stable MIR, which normalizes
                    // every alias the substituted type contains —
                    // and an associated-type projection whose
                    // arguments the erasing-regions folder cannot
                    // normalize fails there. The failure is not
                    // containable after the fact: alongside the
                    // panic, the compiler records a delayed bug,
                    // which the end of the run flushes as an
                    // abort of the whole compilation. So the
                    // substitution is only attempted when neither
                    // the field's declared tree nor the instance's
                    // arguments contain an alias at all (the
                    // substitution cannot create one). With an
                    // alias present, the declared type is walked
                    // instead: its concrete members still yield
                    // their candidates, and the alias itself is
                    // left alone, as aliases are everywhere in
                    // this walk. The containment stays as a
                    // backstop for shapes the screen cannot see.
                    if type_tree_has_alias(&declared, 0)
                        || args.0.iter().any(|a| match a {
                            GenericArgKind::Type(t) => type_tree_has_alias(t, 0),
                            _ => false,
                        })
                    {
                        drop_candidates(&declared, out, seen);
                        continue;
                    }
                    let Ok(field_ty) = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| field.ty_with_args(&args)),
                    ) else {
                        continue;
                    };
                    drop_candidates(&field_ty, out, seen);
                }
            }
        }
        TyKind::RigidTy(RigidTy::Tuple(ts)) => {
            for t in &ts {
                drop_candidates(t, out, seen);
            }
        }
        TyKind::RigidTy(RigidTy::Array(elem, _) | RigidTy::Slice(elem)) => {
            drop_candidates(&elem, out, seen);
        }
        TyKind::RigidTy(RigidTy::Closure(_, args)) => {
            for a in &args.0 {
                if let GenericArgKind::Type(t) = a {
                    drop_candidates(t, out, seen);
                }
            }
        }
        _ => {}
    }
}

/// Resolve a `Drop` impl method for a concrete self type.
///
/// `self_args` are the self type's generic arguments, in the
/// self type's declaration order. The impl method's own
/// parameter list is the impl's, whose names and declaration
/// order are the impl author's choice: `impl<V, U> Drop for
/// S<U, V>` is legal and assigns the parameters the other way
/// round, so resolving with the self type's arguments
/// positionally monomorphizes the WRONG instance (the
/// destructor of `S<u16, u8>` for a value of `S<u8, u16>`).
/// The impl's arguments are instead recovered by unifying the
/// impl's self-type pattern (from the impl's trait reference,
/// whose arguments are still the impl's parameters) with the
/// concrete self type. When unification does not determine
/// every impl parameter, the positional resolution is used,
/// as before.
fn resolve_drop_instance(
    impl_fn: &rustc_public::ty::FnDef,
    self_ty: &Ty,
    self_args: &GenericArgs,
) -> Option<rustc_public::mir::mono::Instance> {
    let positional = |args: &GenericArgs| {
        sparagmos_driver::convert::resolve_instance_guarded(*impl_fn, args)
            .filter(|inst| inst.has_body())
    };
    let unified = (|| -> Option<GenericArgs> {
        let parent = impl_fn.def_id().parent()?;
        let impl_def = rustc_public::ty::ImplDef(parent);
        let pattern = impl_def.trait_impl().value.self_ty();
        let mut bindings: std::collections::HashMap<u32, GenericArgKind> =
            std::collections::HashMap::new();
        unify_self_ty(&pattern, self_ty, &mut bindings)?;
        let generics = impl_def.generics_of();
        let mut impl_args = Vec::with_capacity(generics.params.len());
        for param in &generics.params {
            impl_args.push(bindings.get(&param.index)?.clone());
        }
        Some(GenericArgs(impl_args))
    })();
    match unified {
        Some(impl_args) => sparagmos_driver::convert::resolve_instance_guarded(*impl_fn, &impl_args)
            .filter(|inst| inst.has_body())
            .or_else(|| positional(self_args)),
        None => positional(self_args),
    }
}

/// Unify an impl's self-type pattern with the concrete self
/// type, recording the generic argument each impl parameter
/// (by parameter index) takes. `None` when the two do not
/// have the same shape, or a concrete pattern argument does
/// not match the concrete type's.
fn unify_self_ty(
    pattern: &Ty,
    concrete: &Ty,
    bindings: &mut std::collections::HashMap<u32, GenericArgKind>,
) -> Option<()> {
    use rustc_public::ty::{RigidTy, TyKind};
    // Either side's arguments can carry a type whose kind
    // rustc_public cannot answer for; such a pair can only be
    // judged by the identity fallback below.
    let kinds = (
        sparagmos_driver::convert::guarded_ty_kind(pattern),
        sparagmos_driver::convert::guarded_ty_kind(concrete),
    );
    match kinds {
        (Some(TyKind::Param(param)), _) => {
            bindings.insert(param.index, GenericArgKind::Type(*concrete));
            Some(())
        }
        (Some(TyKind::RigidTy(RigidTy::Adt(def_p, args_p))), Some(TyKind::RigidTy(RigidTy::Adt(def_c, args_c)))) => {
            if def_p != def_c || args_p.0.len() != args_c.0.len() {
                return None;
            }
            for (pa, ca) in args_p.0.iter().zip(args_c.0.iter()) {
                unify_self_arg(pa, ca, bindings)?;
            }
            Some(())
        }
        (Some(TyKind::RigidTy(RigidTy::Ref(_, inner_p, _))), Some(TyKind::RigidTy(RigidTy::Ref(_, inner_c, _))))
        | (Some(TyKind::RigidTy(RigidTy::RawPtr(inner_p, _))), Some(TyKind::RigidTy(RigidTy::RawPtr(inner_c, _)))) => {
            unify_self_ty(&inner_p, &inner_c, bindings)
        }
        (Some(TyKind::RigidTy(RigidTy::Tuple(ts_p))), Some(TyKind::RigidTy(RigidTy::Tuple(ts_c)))) => {
            if ts_p.len() != ts_c.len() {
                return None;
            }
            for (tp, tc) in ts_p.iter().zip(ts_c.iter()) {
                unify_self_ty(tp, tc, bindings)?;
            }
            Some(())
        }
        (Some(TyKind::RigidTy(RigidTy::Array(elem_p, _))), Some(TyKind::RigidTy(RigidTy::Array(elem_c, _))))
        | (Some(TyKind::RigidTy(RigidTy::Slice(elem_p))), Some(TyKind::RigidTy(RigidTy::Slice(elem_c)))) => {
            unify_self_ty(&elem_p, &elem_c, bindings)
        }
        // Any other pattern shape must be identical to the
        // concrete type (a fully concrete self type, or a
        // shape this unification does not descend into).
        _ => {
            if pattern == concrete {
                Some(())
            } else {
                None
            }
        }
    }
}

/// Unify one generic argument of a self-type pattern with
/// the concrete argument (see `unify_self_ty`).
fn unify_self_arg(
    pattern: &GenericArgKind,
    concrete: &GenericArgKind,
    bindings: &mut std::collections::HashMap<u32, GenericArgKind>,
) -> Option<()> {
    use rustc_public::ty::TyConstKind;
    match (pattern, concrete) {
        (GenericArgKind::Type(tp), GenericArgKind::Type(tc)) => {
            unify_self_ty(tp, tc, bindings)
        }
        (GenericArgKind::Const(cp), GenericArgKind::Const(_)) => {
            match cp.kind() {
                TyConstKind::Param(param) => {
                    bindings.insert(param.index, concrete.clone());
                    Some(())
                }
                // A concrete pattern constant must match.
                _ => {
                    if pattern == concrete {
                        Some(())
                    } else {
                        None
                    }
                }
            }
        }
        (GenericArgKind::Lifetime(rp), GenericArgKind::Lifetime(_)) => {
            use rustc_public::ty::RegionKind;
            match &rp.kind {
                RegionKind::ReEarlyParam(region) => {
                    bindings.insert(region.index, concrete.clone());
                    Some(())
                }
                // A concrete pattern region must match.
                _ => {
                    if pattern == concrete {
                        Some(())
                    } else {
                        None
                    }
                }
            }
        }
        _ => None,
    }
}

/// The function definitions of one foreign crate, for the
/// merged foreign scans (`demand_drop_instances`,
/// `certify_drop_no_impl`). Enumerating a crate's functions
/// walks the compiler's metadata tables, and not every crate
/// kind has the entries that walk needs: a proc-macro crate's
/// registered macro items live outside the regular item tables,
/// and enumerating such a crate panics inside the compiler.
/// A crate that cannot be enumerated contributes no definitions
/// to the scan — for a proc-macro crate that is also the exact
/// answer, since its items are not linkable into the compiled
/// program and so cannot hold a `Drop` impl for a type the
/// program can hold. The containment is per crate, in the style
/// of the demand passes' guarded compiler queries: one
/// unenumerable crate must not abort the whole scan (or the
/// conversion with it).
fn foreign_fn_defs(krate: &rustc_public::Crate) -> Vec<rustc_public::ty::FnDef> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| krate.fn_defs()))
        .unwrap_or_default()
}

/// Demand a converted body for the `Drop` impl of every local ADT
/// instantiation that the converted bodies drop (see the call site
/// in `collect`). Each demanded body is converted like a local item
/// and added to the seed bodies, so the reachability collection
/// walks its callees too; demanded bodies are scanned for further
/// drops in turn. An ADT whose impl does not resolve, or whose
/// instance has no body, is left as it was: the interpreter keeps
/// refusing its drop with the reason it already reports.
///
/// The returned accounting credits each converted destructor
/// instance to its impl method's definition (instance count and
/// the first instance's shape) and records each failed conversion
/// under the same definition, so the caller can fold both into
/// the demand accounting: a generic `Drop` impl is itself a
/// deferred definition, and without the credit its placeholder
/// entry would stand against the conversion even though the
/// demanded instance was converted here.
/// The iterator type a call to a modelled consumer of an owned
/// iterator consumes, or `None` when the callee is not such a
/// consumer. The type is the callee generic argument the
/// interpreter's dispatch reads as the iterator type: the first
/// for the `Iterator` methods, the third for
/// `FromIterator::from_iter`, the last for the `Vec` from-iter
/// shim. The modelled consumers drop the iterator inside their
/// real bodies, so both the demand pass and the certification
/// pass treat the type as a drop candidate.
fn consumer_iterator_ty(
    def: &rustc_public::ty::FnDef,
    args: &rustc_public::ty::GenericArgs,
) -> Option<Ty> {
    let nth = match def.name().as_str() {
        "std::iter::Iterator::collect"
        | "std::iter::Iterator::count"
        | "std::iter::Iterator::fold"
        | "std::iter::Iterator::for_each" => Some(0),
        "std::iter::FromIterator::from_iter" => Some(2),
        "std::vec::spec_from_iter_nested::SpecFromIterNested::from_iter" => None,
        _ => return None,
    };
    let types: Vec<&Ty> = args
        .0
        .iter()
        .filter_map(|a| match a {
            GenericArgKind::Type(t) => Some(t),
            _ => None,
        })
        .collect();
    match nth {
        Some(i) => types.get(i).map(|t| **t),
        None => types.last().map(|t| **t),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn demand_drop_instances(
    conv: &mut Converter,
    report: &mut ConversionReport,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    seed_bodies: &mut Vec<rustc_public::mir::Body>,
) -> (
    std::collections::HashMap<rustc_public::DefId, (usize, (usize, usize, usize))>,
    std::collections::HashMap<rustc_public::DefId, BTreeMap<String, usize>>,
    std::collections::HashMap<String, String>,
) {
    let local_crate = rustc_public::local_crate();
    let local_crate_name = local_crate.name.as_str();
    let impl_methods = local_drop_impl_methods();
    // Drop impls of foreign crates, built once on first use.
    // An impl lives in its self type's defining crate (the
    // orphan rule leaves no other home for a `Drop` impl), but
    // a type's rendered name need not start with that crate:
    // facade crates re-export other crates' types under their
    // own path (`std::array::Guard`, the cleanup guard inside
    // `core`'s array clone, renders with the `std` facade), and
    // the impl method's key is rendered in the same facade path
    // space. Scanning only the named crate therefore misses
    // exactly the impls of re-exported types, so the scan
    // covers every external crate; the keys are full rendered
    // paths, unique per type, which keeps the merged map exact.
    let mut foreign_impl_methods: Option<
        std::collections::HashMap<String, rustc_public::ty::FnDef>,
    > = None;
    let mut credited: std::collections::HashMap<
        rustc_public::DefId,
        (usize, (usize, usize, usize)),
    > = std::collections::HashMap::new();
    let mut failures: std::collections::HashMap<
        rustc_public::DefId,
        BTreeMap<String, usize>,
    > = std::collections::HashMap::new();
    let mut dtor_bodies: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    if impl_methods.is_empty() && seed_bodies.is_empty() {
        return (credited, failures, dtor_bodies);
    }
    let mut seen_candidates: std::collections::HashSet<Ty> = Default::default();
    let mut demanded = 0usize;
    let mut scan_from = 0usize;
    while scan_from < seed_bodies.len() {
        let batch: Vec<rustc_public::mir::Body> = seed_bodies[scan_from..].to_vec();
        scan_from = seed_bodies.len();
        let mut candidates = Vec::new();
        let mut seen = Default::default();
        for body in &batch {
            for bb in &body.blocks {
                let rustc_public::mir::TerminatorKind::Drop { place, .. } = &bb.terminator.kind
                else {
                    continue;
                };
                if let Some(ty) = drop_place_ty(body, place) {
                    drop_candidates(&ty, &mut candidates, &mut seen);
                }
            }
            // An unsizing cast to a trait object makes the
            // source pointee droppable through the object's
            // vtable: the converter builds the pointee's glue
            // into the cast, which needs the same destructor
            // bodies a terminator would demand. The pointee of
            // the cast operand (a reference, raw pointer or box
            // around it, or the value itself) is a candidate.
            for bb in &body.blocks {
                for stmt in &bb.statements {
                    let rustc_public::mir::StatementKind::Assign(
                        _,
                        rustc_public::mir::Rvalue::Cast(
                            rustc_public::mir::CastKind::PointerCoercion(
                                rustc_public::mir::PointerCoercion::Unsize,
                            ),
                            operand,
                            target,
                        ),
                    ) = &stmt.kind
                    else {
                        continue;
                    };
                    let Some(target_kind) =
                        sparagmos_driver::convert::guarded_ty_kind(target)
                    else {
                        continue;
                    };
                    if !matches!(target_kind, TyKind::RigidTy(RigidTy::Dynamic(..)))
                        && !matches!(target_kind, TyKind::RigidTy(RigidTy::Adt(..)))
                    {
                        continue;
                    }
                    let (rustc_public::mir::Operand::Copy(place)
                    | rustc_public::mir::Operand::Move(place)) = operand
                    else {
                        continue;
                    };
                    let Some(src) = drop_place_ty(body, place) else {
                        continue;
                    };
                    let pointee = match sparagmos_driver::convert::guarded_ty_kind(&src) {
                        Some(TyKind::RigidTy(RigidTy::Ref(_, inner, _))) => inner,
                        Some(TyKind::RigidTy(RigidTy::RawPtr(inner, _))) => inner,
                        Some(TyKind::RigidTy(RigidTy::Adt(def, ref args)))
                            if def.name() == "std::boxed::Box" =>
                        {
                            match args.0.iter().find_map(|a| match a {
                                GenericArgKind::Type(t) => Some(*t),
                                _ => None,
                            }) {
                                Some(t) => t,
                                None => continue,
                            }
                        }
                        _ => src,
                    };
                    drop_candidates(&pointee, &mut candidates, &mut seen);
                }
            }
            // A `std::mem::drop` call runs the dropped type's glue
            // without a `Drop` terminator, and the interpreter
            // accounts for that glue when it converts the call,
            // which needs the same destructor bodies a terminator
            // would demand.
            for (def, args) in sparagmos_driver::reach::call_targets(body) {
                if def.name() != "std::mem::drop" {
                    continue;
                }
                if let Some(ty) = args.0.iter().find_map(|a| match a {
                    GenericArgKind::Type(t) => Some(*t),
                    _ => None,
                }) {
                    drop_candidates(&ty, &mut candidates, &mut seen);
                }
            }
            // A call to a modelled consumer of an owned iterator
            // (`collect`, `from_iter`, `fold`, ...) drops the
            // iterator inside the callee's real body, where no
            // `Drop` terminator for it exists in any collected
            // body; the interpreter accounts for that glue when
            // it converts the call, which needs the same
            // destructor bodies a terminator would demand.
            for (def, args) in sparagmos_driver::reach::call_targets(body) {
                if let Some(ty) = consumer_iterator_ty(&def, &args) {
                    drop_candidates(&ty, &mut candidates, &mut seen);
                }
            }
        }
        // Local types first: the shared cap must not let a
        // foreign destructor crowd out the entry crate's own.
        let mut candidates = candidates;
        candidates.sort_by_key(|(_, def, _)| {
            def.name().split("::").next() != Some(local_crate_name)
        });
        let mut added = false;
        for (cand_ty, def, args) in candidates {
            if demanded >= MAX_DEMANDED_DTOR_BODIES {
                break;
            }
            if !seen_candidates.insert(cand_ty) {
                continue;
            }
            if args_mention_params(&args) {
                continue;
            }
            // The destructor body to demand: the entry crate's
            // impls come from the local scan; a foreign type's
            // impl comes from its own crate's scan. Types the
            // interpreter's value model owns (the modelled std
            // containers) are matched by name in the glue builder
            // before it ever consults a collected body, so
            // collecting their destructors is inert for them; for
            // every other type the collected body is the glue the
            // builder runs.
            let def_name = def.name();
            let type_key = drop_type_key(&def_name);
            let type_crate = type_key.split("::").next().unwrap_or("");
            let impl_fn = if type_crate == local_crate_name {
                impl_methods.get(&type_key)
            } else {
                let methods = foreign_impl_methods.get_or_insert_with(|| {
                    let mut defs = Vec::new();
                    for krate in rustc_public::external_crates() {
                        if !krate.is_local {
                            defs.extend(foreign_fn_defs(&krate));
                        }
                    }
                    drop_impl_methods(defs)
                });
                methods.get(&type_key)
            };
            let Some(impl_fn) = impl_fn else {
                continue;
            };
            // The impl's generic arguments are recovered from
            // the self type by unification (see
            // `resolve_drop_instance`); a resolution failure
            // simply means the shape differs or this ADT has no
            // applicable impl, and a compiler panic on an odd
            // instantiation must not take the whole conversion
            // down with it.
            let impl_fn = *impl_fn;
            let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                resolve_drop_instance(&impl_fn, &cand_ty, &args)
                    .filter(|inst| inst.has_body())
                    .and_then(|inst| {
                        sparagmos_driver::convert::with_body_materialization(
                            &inst.name(),
                            || inst.body(),
                        )
                        .map(|body| (inst, body))
                    })
            }));
            let Ok(Some((inst, sm_body))) = resolved else {
                continue;
            };
            // Identity, not spelling: the item loop may already
            // have converted this very instance under a differently
            // spelled name (the item records the declared lifetime
            // parameters, e.g. `<'a>`, where the resolved instance
            // renders erased lifetimes as `'_`). Registering the
            // same destructor twice leaves the interpreter's
            // destructor lookup with two candidates, which it
            // treats as ambiguous and refuses the drop, so an
            // already-converted instance is skipped by its DefId.
            let def_id = conv.instance_def_id(inst.def.def_id(), &inst);
            if body_def_ids.contains(&def_id) {
                continue;
            }
            let name = inst.name().to_string();
            if bodies.iter().any(|(n, _)| n == &name) {
                continue;
            }
            let before = bodies.len();
            let report_before = report.functions.len();
            convert_item(
                conv,
                report,
                bodies,
                body_def_ids,
                body_def_names,
                name,
                def_id,
                Ok(Some(sm_body.clone())),
            );
            if bodies.len() > before {
                seed_bodies.push(sm_body);
                demanded += 1;
                added = true;
                // Record the demanded body under the converted
                // type's key: the interpreter's glue builder
                // cannot reconstruct the compiler's instance
                // name for every generic-argument shape (a
                // closure argument renders differently), so the
                // exact name is handed over directly.
                let key = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    format!("{:?}", conv.ty(&cand_ty))
                }));
                if let Ok(key) = key {
                    dtor_bodies.insert(key, inst.name().to_string());
                }
                if let Some(entry) = report.functions.last() {
                    let counts = (entry.blocks, entry.locals, entry.arg_count);
                    let slot = credited
                        .entry(inst.def.def_id())
                        .or_insert((0, counts));
                    slot.0 += 1;
                }
            } else if report.functions.len() > report_before {
                // The demanded instance could not be converted:
                // the failure stands against the impl definition,
                // exactly as a call-graph demand failure would.
                if let Some(FnOutcome::Skipped { reasons }) =
                    report.functions.last().map(|e| &e.outcome)
                {
                    let slot = failures.entry(inst.def.def_id()).or_default();
                    for (k, v) in reasons {
                        *slot.entry(k.clone()).or_default() += v;
                    }
                }
            }
        }
        if !added {
            break;
        }
    }
    (credited, failures, dtor_bodies)
}

/// Cap on tuple/array `Clone` bodies demanded per program,
/// mirroring the drop demand's cap.
const MAX_DEMANDED_CLONE_BODIES: usize = 256;

/// Demand a converted body for the `Clone` impl of every tuple
/// and array type the converted bodies clone. Those impls are
/// macro-generated inside `core`, so the call-graph collection
/// never reaches them (the trait method's name is served by an
/// interpreter builtin, which ends the demand there), and the
/// builtin's structural clone bypasses the elements' own
/// `Clone` impls — wrong for any element whose impl is manual,
/// and unable to reproduce the cleanup a panicking element
/// clone triggers. With the body collected, the call dispatches
/// to it like any other trait call, element impls and unwind
/// cleanup included. Other `Self` shapes keep the builtin:
/// their impls are local items or modelled types.
pub(crate) fn demand_clone_instances(
    conv: &mut Converter,
    report: &mut ConversionReport,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    seed_bodies: &mut Vec<rustc_public::mir::Body>,
) {
    let mut seen: std::collections::HashSet<(rustc_public::DefId, GenericArgs)> =
        Default::default();
    let mut demanded = 0usize;
    let mut scan_from = 0usize;
    while scan_from < seed_bodies.len() {
        let batch: Vec<rustc_public::mir::Body> = seed_bodies[scan_from..].to_vec();
        scan_from = seed_bodies.len();
        let mut targets: Vec<(rustc_public::ty::FnDef, GenericArgs)> = Vec::new();
        for body in &batch {
            for (def, args) in sparagmos_driver::reach::call_targets(body) {
                let name = def.name();
                if name != "std::clone::Clone::clone"
                    && name != "core::clone::Clone::clone"
                {
                    continue;
                }
                let Some(self_ty) = args.0.iter().find_map(|a| match a {
                    GenericArgKind::Type(t) => Some(*t),
                    _ => None,
                }) else {
                    continue;
                };
                if !matches!(
                    sparagmos_driver::convert::guarded_ty_kind(&self_ty),
                    Some(
                        TyKind::RigidTy(RigidTy::Tuple(_))
                            | TyKind::RigidTy(RigidTy::Array(..))
                    )
                ) {
                    continue;
                }
                if args_mention_params(&args) {
                    continue;
                }
                // The dedup key is the identity pair itself,
                // never a debug rendering (rendering an argument
                // type asks its kind, which panics for types the
                // demand walks must tolerate).
                if seen.insert((def.def_id(), args.clone())) {
                    targets.push((def, args));
                }
            }
        }
        let mut added = false;
        for (def, args) in targets {
            if demanded >= MAX_DEMANDED_CLONE_BODIES {
                break;
            }
            // A tuple/array `Clone` resolves to a compiler
            // shim instance: `has_body` reports false for
            // shims, but `body()` still hands over the
            // synthesized body, which is all conversion needs.
            let resolved = sparagmos_driver::convert::resolve_instance_guarded(def, &args).and_then(|inst| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    sparagmos_driver::convert::with_body_materialization(
                        &inst.name(),
                        || inst.body(),
                    )
                    .map(|body| (inst, body))
                }))
                .ok()
                .flatten()
            });
            let Some((inst, sm_body)) = resolved else {
                continue;
            };
            let def_id = conv.instance_def_id(inst.def.def_id(), &inst);
            if body_def_ids.contains(&def_id) {
                continue;
            }
            let name = inst.name().to_string();
            if bodies.iter().any(|(n, _)| n == &name) {
                continue;
            }
            let before = bodies.len();
            convert_item(
                conv,
                report,
                bodies,
                body_def_ids,
                body_def_names,
                name,
                def_id,
                Ok(Some(sm_body.clone())),
            );
            if bodies.len() > before {
                seed_bodies.push(sm_body);
                demanded += 1;
                added = true;
            }
        }
        if !added {
            break;
        }
    }
}

/// Cap on `Wake` / `LocalWake` bodies demanded per program,
/// mirroring the clone demand's cap.
const MAX_DEMANDED_WAKE_BODIES: usize = 256;

/// Demand the `Wake` / `LocalWake` impl method bodies
/// (`wake` and `wake_by_ref`) of every type a converted body
/// builds a `std::task::Waker` / `LocalWaker` from.
///
/// The interpreter models a waker's wake dispatch on the waker
/// value's shape (see `Builtin::WakerWake` in
/// `sparagmos-interp`): for a waker built by the modelled
/// `From<Arc<T: Wake>>` / `From<Rc<T: LocalWake>>` conversion it
/// calls the pointee's impl methods by name. `wake` is usually
/// collected anyway (it is an impl item of the entry crate, or
/// demanded through the vtable of a `dyn Wake`), but
/// `wake_by_ref`'s *default* body - the common case, an impl
/// that defines only `wake` - is an instance of the trait item
/// that no call target names and no collected constant
/// mentions: the only mention would be the `RawWakerVTable`
/// constant the real `From` body builds, and that body is
/// builtin-owned, so it is never collected. This pass is the
/// demand the modelled conversion stands in for: it scans the
/// seed bodies' call targets for the `From` / `Into`
/// conversions whose target is a waker type, takes the
/// pointee type out of the smart-pointer source, and converts
/// the two impl method instances of that type, resolving the
/// trait items directly (which yields the override where one
/// exists and the trait's default body where none does).
pub(crate) fn demand_wake_instances(
    conv: &mut Converter,
    report: &mut ConversionReport,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    seed_bodies: &mut Vec<rustc_public::mir::Body>,
) {
    // The (pointee type, is-local-waker) pairs the seed bodies
    // build wakers from, in first-mention order.
    let mut pointees: Vec<(Ty, bool)> = Vec::new();
    for body in seed_bodies.iter() {
        for (def, args) in sparagmos_driver::reach::call_targets(body) {
            let name = def.name();
            let from = name == "std::convert::From::from"
                || name == "core::convert::From::from";
            let into = name == "std::convert::Into::into"
                || name == "core::convert::Into::into";
            if !from && !into {
                continue;
            }
            if args_mention_params(&args) {
                continue;
            }
            let tys: Vec<Ty> = args
                .0
                .iter()
                .filter_map(|a| match a {
                    GenericArgKind::Type(t) => Some(*t),
                    _ => None,
                })
                .collect();
            if tys.len() < 2 {
                continue;
            }
            // `From::from` renders [target, source]; `Into::into`
            // renders [source, target] (its `Self` is the source).
            let (target, source) = if from {
                (tys[0], tys[1])
            } else {
                (tys[1], tys[0])
            };
            let Some(TyKind::RigidTy(RigidTy::Adt(tdef, _))) =
                sparagmos_driver::convert::guarded_ty_kind(&target)
            else {
                continue;
            };
            let (local, ptr_names): (bool, [&str; 2]) = match tdef.name().as_str() {
                "std::task::Waker" | "core::task::Waker" => {
                    (false, ["std::sync::Arc", "alloc::sync::Arc"])
                }
                "std::task::LocalWaker" | "core::task::LocalWaker" => {
                    (true, ["std::rc::Rc", "alloc::rc::Rc"])
                }
                _ => continue,
            };
            let Some(TyKind::RigidTy(RigidTy::Adt(sdef, sargs))) =
                sparagmos_driver::convert::guarded_ty_kind(&source)
            else {
                continue;
            };
            if !ptr_names.contains(&sdef.name().as_str()) {
                continue;
            }
            let Some(pointee) = sargs.0.iter().find_map(|a| match a {
                GenericArgKind::Type(t) => Some(*t),
                _ => None,
            }) else {
                continue;
            };
            if !pointees.contains(&(pointee, local)) {
                pointees.push((pointee, local));
            }
        }
    }
    if pointees.is_empty() {
        return;
    }
    // The trait items themselves: method declarations are
    // `AssocFn` definitions of the trait's crate, so they are
    // among its `fn_defs` (unlike an impl method, the item is
    // what resolving a default body starts from).
    let mut wake_item: Option<rustc_public::ty::FnDef> = None;
    let mut wake_by_ref_item: Option<rustc_public::ty::FnDef> = None;
    let mut local_wake_item: Option<rustc_public::ty::FnDef> = None;
    let mut local_wake_by_ref_item: Option<rustc_public::ty::FnDef> = None;
    let mut crates = vec![rustc_public::local_crate()];
    crates.extend(rustc_public::external_crates());
    for krate in &crates {
        for f in krate.fn_defs() {
            match f.name().as_str() {
                "std::task::Wake::wake" | "core::task::Wake::wake" => {
                    wake_item.get_or_insert(f);
                }
                "std::task::Wake::wake_by_ref" | "core::task::Wake::wake_by_ref" => {
                    wake_by_ref_item.get_or_insert(f);
                }
                "std::task::LocalWake::wake" | "core::task::LocalWake::wake" => {
                    local_wake_item.get_or_insert(f);
                }
                "std::task::LocalWake::wake_by_ref"
                | "core::task::LocalWake::wake_by_ref" => {
                    local_wake_by_ref_item.get_or_insert(f);
                }
                _ => {}
            }
        }
    }
    let mut seen: std::collections::HashSet<(rustc_public::DefId, GenericArgs)> =
        Default::default();
    let mut demanded = 0usize;
    for (pointee, local) in &pointees {
        let items = if *local {
            [local_wake_item, local_wake_by_ref_item]
        } else {
            [wake_item, wake_by_ref_item]
        };
        for item in items.into_iter().flatten() {
            if demanded >= MAX_DEMANDED_WAKE_BODIES {
                return;
            }
            let args = GenericArgs(vec![GenericArgKind::Type(*pointee)]);
            if !seen.insert((item.def_id(), args.clone())) {
                continue;
            }
            // Instance resolution goes through the compiler's
            // internals and can panic on exotic targets; isolate
            // it exactly as the clone demand does.
            let resolved = sparagmos_driver::convert::resolve_instance_guarded(item, &args).and_then(|inst| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    sparagmos_driver::convert::with_body_materialization(
                        &inst.name(),
                        || inst.body(),
                    )
                    .map(|body| (inst, body))
                }))
                .ok()
                .flatten()
            });
            let Some((inst, sm_body)) = resolved else {
                continue;
            };
            let def_id = conv.instance_def_id(inst.def.def_id(), &inst);
            if body_def_ids.contains(&def_id) {
                continue;
            }
            let name = inst.name().to_string();
            if bodies.iter().any(|(n, _)| n == &name) {
                continue;
            }
            let before = bodies.len();
            convert_item(
                conv,
                report,
                bodies,
                body_def_ids,
                body_def_names,
                name,
                def_id,
                Ok(Some(sm_body.clone())),
            );
            if bodies.len() > before {
                seed_bodies.push(sm_body);
                demanded += 1;
            }
        }
    }
}

/// The ADT instantiations among the types the collected bodies
/// drop that have no `Drop` impl at all, keyed like
/// `certify_drop_trivial`'s result. A `Drop` impl for a type can
/// only live in the type's defining crate (the orphan rule: the
/// trait is `core`'s, so no other crate may implement it for the
/// type), so the scan must cover that crate — but a type's
/// rendered name need not start with it: facade crates
/// re-export other crates' types under their own path
/// (`std::array::Guard`, the cleanup guard inside `core`'s
/// array clone, is defined in `core`), and the impl keys are
/// rendered in the same facade path space. The foreign scan
/// therefore covers every external crate, and absence from
/// that merged key set — full rendered paths, unique per
/// type — is absence everywhere. An impl under any generic
/// spelling blocks certification: it keeps the interpreter's
/// refusal for the type.
/// Cap on ZST `Fn`-family bodies demanded per program across
/// all [`demand_zst_fn_instances`] calls, mirroring the dyn
/// demand's cap.
const MAX_DEMANDED_ZST_FN_BODIES: usize = 256;

/// The `Fn` / `FnMut` / `FnOnce` impl methods among `fn_defs`,
/// keyed by the self type's path without its parameter list
/// (see `drop_type_key`), one entry per (trait, method) impl.
/// An impl method's name is
/// `<SELF as std::ops::Fn<Args>>::call` (and its `FnMut` /
/// `FnOnce` counterparts, under either the `std::` or the
/// `core::` spelling of the trait path).
fn fn_impl_methods(
    fn_defs: Vec<rustc_public::ty::FnDef>,
) -> std::collections::HashMap<String, Vec<rustc_public::ty::FnDef>> {
    let mut out: std::collections::HashMap<
        String,
        Vec<rustc_public::ty::FnDef>,
    > = std::collections::HashMap::new();
    for f in fn_defs {
        let name = f.name();
        let Some(inner_end) = balanced_group_end(&name, 0) else {
            continue;
        };
        let method = &name[inner_end + 1..];
        if !matches!(method, "::call" | "::call_mut" | "::call_once") {
            continue;
        }
        let inner = &name[1..inner_end];
        let Some(as_pos) = top_level_as_pos(inner) else {
            continue;
        };
        let trait_part = inner[as_pos + 4..].trim();
        let trait_base = trait_part
            .split('<')
            .next()
            .unwrap_or(trait_part)
            .trim();
        let expected = match method {
            "::call" => ["std::ops::Fn", "core::ops::Fn"],
            "::call_mut" => ["std::ops::FnMut", "core::ops::FnMut"],
            _ => ["std::ops::FnOnce", "core::ops::FnOnce"],
        };
        if !expected.contains(&trait_base) {
            continue;
        }
        let base = drop_type_key(&inner[..as_pos]);
        out.entry(base).or_default().push(f);
    }
    out
}

/// Whether the ADT instantiation `(def, args)` is provably
/// zero-sized from its structure alone: every variant's every
/// field is itself provably zero-sized (an empty ADT, a tuple
/// of such, a unit). The question is deliberately *not* asked
/// through the compiler's layout machinery: a layout query on
/// a type with an unsized field records a delayed compiler
/// bug that fails the whole compilation at the end, even when
/// the query itself is caught - and the types this demand
/// serves (core's `impl_fn_for_zst!` family and its kin) are
/// all fieldless or phantom-only, which the structural
/// question decides exactly.
fn adt_provable_zst(
    def: &rustc_public::ty::AdtDef,
    args: &GenericArgs,
    depth: usize,
) -> bool {
    if depth > 8 {
        return false;
    }
    let variants = def.variants();
    if variants.is_empty() {
        return false;
    }
    variants.iter().all(|variant| {
        variant.fields().iter().all(|field| {
            let declared = field.ty();
            // Substituting the instance's arguments can hit the
            // same normalization failures the drop walk screens
            // for; with an alias anywhere, the declared type is
            // judged instead (its parameters then simply fail
            // the question unless the field is a fieldless ADT).
            let judged = if type_tree_has_alias(&declared, 0)
                || args.0.iter().any(|a| match a {
                    GenericArgKind::Type(t) => type_tree_has_alias(t, 0),
                    _ => false,
                }) {
                declared
            } else {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    field.ty_with_args(args)
                })) {
                    Ok(t) => t,
                    Err(_) => declared,
                }
            };
            ty_provable_zst(&judged, args, depth + 1)
        })
    })
}

/// The field-type half of [`adt_provable_zst`]: a type is
/// provably zero-sized when it is an ADT instantiation that
/// is, a tuple of such types, or a generic parameter whose
/// argument (in the enclosing ADT's argument list, by
/// parameter index) is. Anything else - a scalar, a reference,
/// an array, a string or slice shape - is not, conservatively:
/// a callable the question misses keeps the interpreter's
/// existing outcome, never a wrong body.
fn ty_provable_zst(ty: &Ty, args: &GenericArgs, depth: usize) -> bool {
    if depth > 8 {
        return false;
    }
    let Some(kind) = sparagmos_driver::convert::guarded_ty_kind(ty) else {
        return false;
    };
    match kind {
        TyKind::RigidTy(RigidTy::Adt(def, inner_args)) => {
            adt_provable_zst(&def, &inner_args, depth)
        }
        TyKind::RigidTy(RigidTy::Tuple(ts)) => {
            ts.iter().all(|t| ty_provable_zst(t, args, depth + 1))
        }
        TyKind::Param(param) => match args.0.get(param.index as usize) {
            Some(GenericArgKind::Type(t)) => {
                ty_provable_zst(t, &GenericArgs(Vec::new()), depth + 1)
            }
            _ => false,
        },
        _ => false,
    }
}

/// Demand a converted body for the `Fn` / `FnMut` / `FnOnce`
/// impl methods of every zero-sized ADT type the scanned bodies
/// mention.
///
/// A zero-sized callable (core's `impl_fn_for_zst!` family -
/// `str::LinesMap` behind `str::lines`, `char::CharEscapeDebug`,
/// `char::IsWhitespace`, and their siblings) is consumed by the
/// interpreter's iterator adapter steppers, which are builtins:
/// no collected body ever calls the callable's `Fn::call` body,
/// so the call-graph collection has no edge to demand it by, and
/// the value itself is an untyped empty ADT the interpreter
/// cannot dispatch from alone. Demanding the impl bodies here -
/// for every zero-sized ADT with an `Fn`-family impl, whatever
/// its name - puts them in the program under their instance
/// names, where the interpreter's type-directed dispatch for
/// empty-ADT callables finds them (see
/// `Interpreter::call_callable_sync_typed`). Converted bodies
/// join `seed_bodies`, so the reachability collection walks
/// their callees too; the caller reruns this pass over the
/// collected foreign bodies, whose own type mentions (a library
/// adapter's state type, above all) are where these callables
/// are usually named.
#[allow(clippy::too_many_arguments)]
pub(crate) fn demand_zst_fn_instances(
    conv: &mut Converter,
    report: &mut ConversionReport,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    seed_bodies: &mut Vec<rustc_public::mir::Body>,
    demanded: &mut usize,
) {
    // Every ADT instantiation the scanned bodies mention, in
    // first-mention order (the same walk the drop demand uses).
    let mut candidates: Vec<(Ty, rustc_public::ty::AdtDef, GenericArgs)> =
        Vec::new();
    let mut seen: std::collections::HashSet<Ty> = Default::default();
    for body in seed_bodies.iter() {
        for decl in body.locals() {
            drop_candidates(&decl.ty, &mut candidates, &mut seen);
        }
    }
    let zst_candidates: Vec<_> = candidates
        .into_iter()
        .filter(|(_, def, args)| {
            !args_mention_params(args) && adt_provable_zst(def, args, 0)
        })
        .collect();
    if zst_candidates.is_empty() {
        return;
    }
    // The impl scan covers the local crate and every external
    // crate, built once here: an `Fn` impl lives in its self
    // type's defining crate (the orphan rule), but a type's
    // rendered name need not start with that crate (facade
    // re-exports), and the impl keys are rendered in the same
    // facade path space - the same reasoning as the drop
    // demand's foreign scan.
    let mut all_defs: Vec<rustc_public::ty::FnDef> =
        rustc_public::local_crate().fn_defs();
    for krate in rustc_public::external_crates() {
        if !krate.is_local {
            all_defs.extend(foreign_fn_defs(&krate));
        }
    }
    let impl_methods = fn_impl_methods(all_defs);
    if impl_methods.is_empty() {
        return;
    }
    let mut seen_instances: std::collections::HashSet<String> =
        Default::default();
    for (cand_ty, def, args) in &zst_candidates {
        if *demanded >= MAX_DEMANDED_ZST_FN_BODIES {
            return;
        }
        let type_key = drop_type_key(&def.name());
        let Some(impl_fns) = impl_methods.get(&type_key) else {
            continue;
        };
        for impl_fn in impl_fns {
            if *demanded >= MAX_DEMANDED_ZST_FN_BODIES {
                return;
            }
            let impl_fn = *impl_fn;
            // The impl's generic arguments are recovered from
            // the self type by unification, exactly as for a
            // `Drop` impl (see `resolve_drop_instance`); a
            // resolution failure or a compiler panic on an odd
            // instantiation simply leaves this candidate
            // undemanded.
            let resolved = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| {
                    resolve_drop_instance(&impl_fn, cand_ty, args)
                        .filter(|inst| inst.has_body())
                        .and_then(|inst| {
                            sparagmos_driver::convert::with_body_materialization(
                                &inst.name(),
                                || inst.body(),
                            )
                            .map(|body| (inst, body))
                        })
                }),
            );
            let Ok(Some((inst, sm_body))) = resolved else {
                continue;
            };
            if !seen_instances.insert(inst.name().to_string()) {
                continue;
            }
            let def_id = conv.instance_def_id(inst.def.def_id(), &inst);
            if body_def_ids.contains(&def_id) {
                continue;
            }
            let name = inst.name().to_string();
            if bodies.iter().any(|(n, _)| n == &name) {
                continue;
            }
            let before = bodies.len();
            convert_item(
                conv,
                report,
                bodies,
                body_def_ids,
                body_def_names,
                name,
                def_id,
                Ok(Some(sm_body.clone())),
            );
            if bodies.len() > before {
                seed_bodies.push(sm_body);
                *demanded += 1;
            }
        }
    }
}

pub(crate) fn certify_drop_no_impl(
    conv: &mut Converter,
    bodies: &[rustc_public::mir::Body],
) -> std::collections::HashSet<String> {
    let mut candidates = Vec::new();
    let mut seen = Default::default();
    for body in bodies {
        for bb in &body.blocks {
            let rustc_public::mir::TerminatorKind::Drop { place, .. } = &bb.terminator.kind
            else {
                continue;
            };
            if let Some(ty) = drop_place_ty(body, place) {
                drop_candidates(&ty, &mut candidates, &mut seen);
            }
        }
        for (def, args) in sparagmos_driver::reach::call_targets(body) {
            if def.name() != "std::mem::drop" {
                continue;
            }
            if let Some(ty) = args.0.iter().find_map(|a| match a {
                GenericArgKind::Type(t) => Some(*t),
                _ => None,
            }) {
                drop_candidates(&ty, &mut candidates, &mut seen);
            }
        }
        // The types of places projected through a `Downcast`
        // step: a coroutine's state fields are projected that
        // way (a `Downcast` + `Field` pair from the state
        // object), and they are the one class of field type the
        // candidate walk cannot reach from a dropped type — a
        // coroutine definition declares no variant fields the
        // walk could substitute into. The interpreter's glue
        // builder plans a dropped coroutine state by state from
        // exactly these projected field types, so a state field
        // type no scan certifies (a foreign marker type like
        // `std::future::ResumeTy`, saved in every suspended
        // state) would refuse the whole drop.
        for place in downcast_places(body) {
            if let Some(ty) = drop_place_ty(body, &place) {
                drop_candidates(&ty, &mut candidates, &mut seen);
            }
        }
        // The modelled owned-iterator consumers, with the same
        // candidate source as the demand pass: the consumed
        // iterator's glue is built at conversion, and its field
        // tree is certified (or not) exactly as a dropped
        // place's would be.
        for (def, args) in sparagmos_driver::reach::call_targets(body) {
            if let Some(ty) = consumer_iterator_ty(&def, &args) {
                drop_candidates(&ty, &mut candidates, &mut seen);
            }
        }
    }
    let local_crate_name = rustc_public::local_crate().name;
    let mut scans: std::collections::HashMap<String, Option<std::collections::HashSet<String>>> =
        std::collections::HashMap::new();
    // The merged foreign impl-name set (see the doc comment
    // above), built on first use and shared by every foreign
    // candidate.
    let mut foreign_names: Option<std::collections::HashSet<String>> = None;
    let mut out = std::collections::HashSet::new();
    let mut queried = 0usize;
    for (ty, def, args) in candidates {
        if queried >= MAX_DROP_CERTIFICATIONS {
            break;
        }
        if args_mention_params(&args) {
            continue;
        }
        queried += 1;
        let def_name = def.name();
        let type_key = drop_type_key(&def_name);
        let type_crate = type_key.split("::").next().unwrap_or("");
        let scan = scans.entry(type_crate.to_string()).or_insert_with(|| {
            if type_crate == local_crate_name.as_str() {
                Some(local_drop_impl_methods().into_keys().collect())
            } else {
                Some(
                    foreign_names
                        .get_or_insert_with(|| {
                            let mut defs = Vec::new();
                            for krate in rustc_public::external_crates() {
                                if !krate.is_local {
                                    defs.extend(foreign_fn_defs(&krate));
                                }
                            }
                            drop_impl_methods(defs).into_keys().collect()
                        })
                        .clone(),
                )
            }
        });
        let Some(impl_names) = scan else {
            continue;
        };
        if impl_names.contains(&type_key) {
            continue;
        }
        let key = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            format!("{:?}", conv.ty(&ty))
        }));
        if let Ok(key) = key {
            out.insert(key);
        }
    }
    out
}

/// Cap on drop-glue certifications per program: certification
/// is one compiler query per distinct instantiation, and a
/// program dropping more distinct types than this keeps the
/// uncertified (refused) treatment for the tail.
const MAX_DROP_CERTIFICATIONS: usize = 1024;

/// The ADT instantiations among the types the collected bodies
/// drop whose drop glue the compiler certifies empty, keyed by
/// the `Debug` rendering of the converted type (the key the
/// interpreter's glue builder consults). A foreign type's
/// destructor cannot be ruled out from the name table — an impl
/// the collection never demanded leaves no trace there — but
/// the compiler's own drop-in-place resolution is complete: an
/// empty shim means no destructor and no field glue anywhere in
/// the type's tree. A query that panics (an unaskable type, an
/// odd instantiation) certifies nothing.
pub(crate) fn certify_drop_trivial(
    conv: &mut Converter,
    bodies: &[rustc_public::mir::Body],
) -> std::collections::HashSet<String> {
    // Resolving `drop_in_place` requires the `drop_glue` lang
    // item; a crate graph without one (a minicore) makes the
    // query emit a fatal diagnostic instead of panicking, which
    // the `catch_unwind` below cannot contain. Certify nothing
    // there: no type is proven trivial, and conversion of the
    // crate itself is unaffected.
    if !sparagmos_driver::convert::drop_glue_lang_item_defined() {
        return std::collections::HashSet::new();
    }
    let mut candidates = Vec::new();
    let mut seen = Default::default();
    for body in bodies {
        for bb in &body.blocks {
            let rustc_public::mir::TerminatorKind::Drop { place, .. } = &bb.terminator.kind
            else {
                continue;
            };
            if let Some(ty) = drop_place_ty(body, place) {
                drop_candidates(&ty, &mut candidates, &mut seen);
            }
        }
        // A `std::mem::drop` call runs the dropped type's glue
        // without a `Drop` terminator; its argument type is a
        // certification candidate for the same reason.
        for (def, args) in sparagmos_driver::reach::call_targets(body) {
            if def.name() != "std::mem::drop" {
                continue;
            }
            if let Some(ty) = args.0.iter().find_map(|a| match a {
                GenericArgKind::Type(t) => Some(*t),
                _ => None,
            }) {
                drop_candidates(&ty, &mut candidates, &mut seen);
            }
        }
        // The types of places projected through a `Downcast`
        // step: a coroutine's state fields are projected that
        // way (a `Downcast` + `Field` pair from the state
        // object), and they are the one class of field type the
        // candidate walk cannot reach from a dropped type — a
        // coroutine definition declares no variant fields the
        // walk could substitute into. The interpreter's glue
        // builder plans a dropped coroutine state by state from
        // exactly these projected field types, so a state field
        // type no scan certifies (a foreign marker type like
        // `std::future::ResumeTy`, saved in every suspended
        // state) would refuse the whole drop.
        for place in downcast_places(body) {
            if let Some(ty) = drop_place_ty(body, &place) {
                drop_candidates(&ty, &mut candidates, &mut seen);
            }
        }
    }
    let mut out = std::collections::HashSet::new();
    let mut queried = 0usize;
    for (ty, _def, args) in candidates {
        if queried >= MAX_DROP_CERTIFICATIONS {
            break;
        }
        if args_mention_params(&args) {
            continue;
        }
        queried += 1;
        let certified = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let inst = Instance::resolve_drop_in_place(ty);
            if !inst.is_empty_shim() {
                return None;
            }
            Some(format!("{:?}", conv.ty(&ty)))
        }));
        if let Ok(Some(key)) = certified {
            out.insert(key);
        }
    }
    out
}

/// The `dyn` trait references one Stable MIR body forms: every
/// unsizing cast whose target mentions a `dyn` type and whose
/// source pointee is a concrete sized type, as a `TraitRef` with
/// that type as `Self` — the vtable the cast materializes, in
/// Stable MIR terms. Casts from another `dyn` type (a supertrait
/// upcast) name no concrete type and are skipped: the sized
/// formation of the same value elsewhere in the program demands
/// the full vtable, supertraits included. Casts whose pointee
/// still mentions a type parameter are skipped the same way: a
/// monomorphic instance body carries the concrete cast.
fn dyn_trait_refs(body: &rustc_public::mir::Body) -> Vec<rustc_public::ty::TraitRef> {
    /// One wrapper layer on the way from a cast's target type to
    /// the `dyn` type inside it: a reference layer, a raw-pointer
    /// layer, or the generic-argument index of an ADT layer.
    #[derive(Clone, Copy)]
    enum DynStep {
        Ref,
        RawPtr,
        Arg(usize),
    }
    /// The bounds of the `dyn` type inside `ty`, together with
    /// the wrapper path to it:
    /// the vtable a cast materializes belongs to the source type
    /// the same path names — the layers above the `dyn` are the
    /// pointer the value travels through, and everything at the
    /// path's end is the trait's `Self`, wrapper layers included:
    /// `Box<Box<T>>` cast to `Box<dyn Trait>` forms the vtable of
    /// `Box<T>` (the trait is implemented for `Box<T>`), not of
    /// `T`. Stripping every pointer wrapper instead (the fallback
    /// in `pointee`) names a `Self` no impl answers for.
    fn find_dyn_path(
        ty: &Ty,
    ) -> Option<(
        Vec<rustc_public::ty::Binder<rustc_public::ty::ExistentialPredicate>>,
        Vec<DynStep>,
    )> {
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Dynamic(bounds, _)) => Some((bounds, Vec::new())),
            TyKind::RigidTy(RigidTy::Ref(_, inner, _)) => {
                let (bounds, mut path) = find_dyn_path(&inner)?;
                path.insert(0, DynStep::Ref);
                Some((bounds, path))
            }
            TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => {
                let (bounds, mut path) = find_dyn_path(&inner)?;
                path.insert(0, DynStep::RawPtr);
                Some((bounds, path))
            }
            TyKind::RigidTy(RigidTy::Adt(_, args)) => {
                for (i, arg) in args.0.iter().enumerate() {
                    if let GenericArgKind::Type(t) = arg {
                        if let Some((bounds, mut path)) = find_dyn_path(t) {
                            path.insert(0, DynStep::Arg(i));
                            return Some((bounds, path));
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
    /// The source type a `dyn` path names (see `find_dyn_path`):
    /// apply each wrapper step of the target to the source.
    /// `None` when the shapes diverge (the caller falls back to
    /// the heuristic walk in `pointee`).
    fn pointee_at(ty: &Ty, path: &[DynStep]) -> Option<Ty> {
        let mut ty = *ty;
        for step in path {
            ty = match (step, ty.kind()) {
                (DynStep::Ref, TyKind::RigidTy(RigidTy::Ref(_, inner, _))) => inner,
                (DynStep::RawPtr, TyKind::RigidTy(RigidTy::RawPtr(inner, _))) => inner,
                (DynStep::Arg(i), TyKind::RigidTy(RigidTy::Adt(_, args))) => {
                    match args.0.get(*i) {
                        Some(GenericArgKind::Type(t)) => *t,
                        _ => return None,
                    }
                }
                _ => return None,
            };
        }
        Some(ty)
    }
    fn pointee(ty: &Ty) -> Ty {
        fn first_type_arg(args: &rustc_public::ty::GenericArgs) -> Option<Ty> {
            args.0.iter().find_map(|a| match a {
                GenericArgKind::Type(t) => Some(*t),
                _ => None,
            })
        }
        // The heuristic walk, kept as the fallback for shapes
        // the target-driven path (`find_dyn_path` / `pointee_at`)
        // cannot follow: the first step is the plain pointee step
        // (any ADT gives up its first type argument) and the walk
        // continues through the standard pointer wrappers (`Pin`,
        // `Box`, `Rc`, `Arc`, the `Weak`s) and references until
        // the sized type itself. Any other ADT stops the walk, as
        // its first argument is not a pointee.
        let mut ty = match ty.kind() {
            TyKind::RigidTy(RigidTy::Ref(_, inner, _)) => inner,
            TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => inner,
            TyKind::RigidTy(RigidTy::Adt(_, args)) => {
                first_type_arg(&args).unwrap_or(*ty)
            }
            _ => *ty,
        };
        loop {
            match ty.kind() {
                TyKind::RigidTy(RigidTy::Ref(_, inner, _)) => ty = inner,
                TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => ty = inner,
                TyKind::RigidTy(RigidTy::Adt(def, args)) => {
                    match def.name().as_str() {
                        "std::pin::Pin"
                        | "core::pin::Pin"
                        | "std::boxed::Box"
                        | "alloc::boxed::Box"
                        | "std::rc::Rc"
                        | "alloc::rc::Rc"
                        | "std::sync::Arc"
                        | "alloc::sync::Arc"
                        | "std::rc::Weak"
                        | "alloc::rc::Weak"
                        | "std::sync::Weak"
                        | "alloc::sync::Weak" => match first_type_arg(&args) {
                            Some(inner) => ty = inner,
                            None => return ty,
                        },
                        _ => return ty,
                    }
                }
                _ => return ty,
            }
        }
    }
    let locals = body.locals();
    let mut out = Vec::new();
    for bb in &body.blocks {
        for stmt in &bb.statements {
            let rustc_public::mir::StatementKind::Assign(_, rv) = &stmt.kind else {
                continue;
            };
            let rustc_public::mir::Rvalue::Cast(kind, op, target) = rv else {
                continue;
            };
            if !matches!(
                kind,
                rustc_public::mir::CastKind::PointerCoercion(
                    rustc_public::mir::PointerCoercion::Unsize
                )
            ) {
                continue;
            }
            let Some((bounds, dyn_path)) = find_dyn_path(target) else {
                continue;
            };
            let Ok(src) = op.ty(&locals) else {
                continue;
            };
            let concrete = pointee_at(&src, &dyn_path).unwrap_or_else(|| pointee(&src));
            if matches!(concrete.kind(), TyKind::RigidTy(RigidTy::Dynamic(..))) {
                continue;
            }
            if args_mention_params(&GenericArgs(vec![GenericArgKind::Type(concrete)])) {
                continue;
            }
            for bound in &bounds {
                if let rustc_public::ty::ExistentialPredicate::Trait(et) = &bound.value {
                    out.push(et.with_self_ty(concrete));
                }
            }
        }
    }
    out
}

/// Cap on vtable method bodies demanded per program across all
/// [`demand_dyn_instances`] calls, mirroring the drop demand's cap.
const MAX_DEMANDED_DYN_BODIES: usize = 256;

/// Demand a converted body for every method of every vtable the
/// scanned bodies form (see [`dyn_trait_refs`]).
///
/// A method called through a `dyn` receiver has no demand edge the
/// other passes can follow: resolving the trait method with the
/// `dyn` type itself yields a virtual instance, which has no body,
/// so a trait *default* method — whose only body is the trait's
/// own, instantiated with the concrete `Self` — is never demanded
/// at all, and the interpreter's vtable lookup finds neither an
/// impl body nor a default body at the call. The body a vtable
/// dispatch actually reaches is the instance the compiler puts in
/// the concrete type's vtable: an impl method when the impl
/// provides one, the trait's default body instantiated with the
/// concrete `Self` when it does not. Demanding exactly those
/// instances (the compiler's own vtable entries, supertrait
/// vtables included through their `TraitVPtr` entries) puts the
/// dispatch target in the program under the instance name the
/// interpreter's lookup constructs
/// (`<{concrete} as {trait}>::{method}`). An instance an earlier
/// pass already converted is skipped by identity and by name, so
/// a collected impl body is never shadowed by the default body —
/// the vtable itself already encodes that priority.
///
/// Instances with no MIR body available (a non-generic,
/// non-inline library method is the common case) are left alone:
/// the dispatch stays the interpreter's explicit unsupported
/// error, as before. `Fn`-family traits are skipped: their vtable
/// entries are call shims, and `dyn Fn` calls are executed through
/// the callable value, never through this lookup. Every converted
/// instance whose definition is a deferred definition is recorded
/// in `credited` (instance count and first instance's shape); the
/// caller credits it in the demand accounting once the demand
/// passes have run (see the call sites in `collect`).
#[allow(clippy::too_many_arguments)]
/// Record the location stamp of the `Fn`-family vtable shim
/// for the trait reference `tr` (whose `Self` is a concrete
/// callable type), if this trait is one of the `Fn` traits and
/// its shim location is not recorded yet. The vtable's method
/// entry for the trait's own call method is the shim instance;
/// its body is the generated shim, and the stamp of the shim's
/// call into the callable body is the trait method's definition
/// location in core.
fn note_fn_shim(
    conv: &mut Converter,
    tr: &rustc_public::ty::TraitRef,
) {
    let (kind, method) = match tr.def_id.name().as_str() {
        "std::ops::Fn" | "core::ops::Fn" => (0, "call"),
        "std::ops::FnMut" | "core::ops::FnMut" => (1, "call_mut"),
        "std::ops::FnOnce" | "core::ops::FnOnce" => (2, "call_once"),
        _ => return,
    };
    if conv.fn_shim_locations[kind].is_some() {
        return;
    }
    let Ok(entries) = std::panic::catch_unwind(
        std::panic::AssertUnwindSafe(|| tr.vtable_entries()),
    ) else {
        return;
    };
    for entry in entries {
        let rustc_public::ty::VtblEntry::Method(inst) = entry else {
            continue;
        };
        if !inst.name().ends_with(method) {
            continue;
        }
        let Ok(Some(shim_body)) = std::panic::catch_unwind(
            std::panic::AssertUnwindSafe(|| {
                sparagmos_driver::convert::with_body_materialization(
                    &inst.name(),
                    || inst.body(),
                )
            }),
        ) else {
            continue;
        };
        conv.note_fn_shim_location(kind, &shim_body);
        return;
    }
}

pub(crate) fn demand_dyn_instances(
    conv: &mut Converter,
    report: &mut ConversionReport,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    sources: &[rustc_public::mir::Body],
    deferred_defs: &HashSet<rustc_public::DefId>,
    credited: &mut HashMap<rustc_public::DefId, (usize, (usize, usize, usize))>,
    demanded: &mut usize,
) -> Vec<rustc_public::mir::Body> {
    use std::collections::VecDeque;
    let mut queue: VecDeque<rustc_public::mir::Body> = sources.iter().cloned().collect();
    let mut seen_refs: HashSet<String> = HashSet::new();
    let mut new_bodies: Vec<rustc_public::mir::Body> = Vec::new();
    // The trait references of the vtables the constants
    // decoded so far form (see `Converter::vtable_demand`):
    // no scanned body's cast names them, so they ride the
    // first scanned body's iteration.
    let mut extra_refs: Vec<rustc_public::ty::TraitRef> = if queue.is_empty() {
        // No scanned body to ride: leave the recorded refs for
        // a later call of this pass.
        Vec::new()
    } else {
        std::mem::take(&mut conv.vtable_demand)
    };
    while let Some(body) = queue.pop_front() {
        // Vtable queries go through the compiler's internals and
        // can panic on exotic traits; isolate each scanned body
        // exactly as the other demand passes do, so this pass can
        // never make a crate convert worse.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut converted_now: Vec<rustc_public::mir::Body> = Vec::new();
            for tref in dyn_trait_refs(&body)
                .into_iter()
                .chain(extra_refs.drain(..))
            {
                if !seen_refs.insert(format!("{tref:?}")) {
                    continue;
                }
                let mut vtables: VecDeque<rustc_public::ty::TraitRef> =
                    VecDeque::from([tref]);
                while let Some(tr) = vtables.pop_front() {
                    if matches!(
                        tr.def_id.name().as_str(),
                        "std::ops::Fn"
                            | "core::ops::Fn"
                            | "std::ops::FnMut"
                            | "core::ops::FnMut"
                            | "std::ops::FnOnce"
                            | "core::ops::FnOnce"
                    ) {
                        // `Fn`-trait dispatch is the interpreter's
                        // own `CallTrait` machinery, so no vtable
                        // bodies are demanded — but the vtable's
                        // method entry is the compiler-generated
                        // shim whose location a track-caller
                        // closure entered through this `dyn`
                        // receiver observes, so record it before
                        // skipping (see
                        // `Converter::note_fn_shim_location`).
                        note_fn_shim(conv, &tr);
                        continue;
                    }
                    let Ok(entries) = std::panic::catch_unwind(
                        std::panic::AssertUnwindSafe(|| tr.vtable_entries()),
                    ) else {
                        continue;
                    };
                    for entry in entries {
                        match entry {
                            rustc_public::ty::VtblEntry::TraitVPtr(super_ref) => {
                                if seen_refs.insert(format!("{super_ref:?}")) {
                                    vtables.push_back(super_ref);
                                }
                            }
                            rustc_public::ty::VtblEntry::Method(inst) => {
                                if *demanded >= MAX_DEMANDED_DYN_BODIES {
                                    continue;
                                }
                                if !matches!(
                                    inst.kind,
                                    rustc_public::mir::mono::InstanceKind::Item
                                ) {
                                    continue;
                                }
                                let Ok(true) = std::panic::catch_unwind(
                                    std::panic::AssertUnwindSafe(|| inst.has_body()),
                                ) else {
                                    continue;
                                };
                                let Ok(Some(sbody)) = std::panic::catch_unwind(
                                    std::panic::AssertUnwindSafe(|| {
                                        sparagmos_driver::convert::with_body_materialization(
                                            &inst.name(),
                                            || inst.body(),
                                        )
                                    }),
                                ) else {
                                    continue;
                                };
                                let def_id =
                                    conv.instance_def_id(inst.def.def_id(), &inst);
                                if body_def_ids.contains(&def_id) {
                                    continue;
                                }
                                let name = inst.name().to_string();
                                if bodies.iter().any(|(n, _)| n == &name) {
                                    continue;
                                }
                                let before = bodies.len();
                                convert_item(
                                    conv,
                                    report,
                                    bodies,
                                    body_def_ids,
                                    body_def_names,
                                    name,
                                    def_id,
                                    Ok(Some(sbody.clone())),
                                );
                                if bodies.len() > before {
                                    *demanded += 1;
                                    converted_now.push(sbody.clone());
                                    if deferred_defs.contains(&inst.def.def_id()) {
                                        let entry = credited
                                            .entry(inst.def.def_id())
                                            .or_insert_with(|| {
                                                (
                                                    0,
                                                    (
                                                        sbody.blocks.len(),
                                                        sbody.locals().len(),
                                                        sbody.arg_locals().len(),
                                                    ),
                                                )
                                            });
                                        entry.0 += 1;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            converted_now
        }));
        match outcome {
            Ok(converted_now) => {
                for sbody in converted_now {
                    queue.push_back(sbody.clone());
                    new_bodies.push(sbody);
                }
            }
            Err(_) => {
                let _ = take_captured_panic();
            }
        }
    }
    new_bodies
}

/// Account for a constant that has no MIR body anywhere: a
/// *trivial* constant (a literal, a path to another constant, a
/// const parameter). rustc represents such constants without a
/// body by design — its `is_trivial_const` check is what makes
/// body retrieval return `None` for an otherwise available local
/// constant — because the value is folded into every use site:
/// the bodies that use the constant carry the evaluated value,
/// so nothing is missing from the program and there is no body
/// to convert. The constant is therefore accounted as converted,
/// with no body, instead of being reported as a missing body.
pub(crate) fn account_trivial_const(report: &mut ConversionReport, name: String) {
    report.bodyless_converted += 1;
    report.functions.push(FnReport {
        name,
        blocks: 0,
        locals: 0,
        arg_count: 0,
        outcome: FnOutcome::Converted,
    });
}
