//! Generic-argument composition and substitution over Stable MIR types, and sizedness of converted types.

use super::*;

/// Whether a converted type mentions a type alias anywhere
/// inside it (see `Converter::record_instance_shape`).
/// Whether an ADT instance's generic arguments mention a
/// generic parameter: a type parameter, a const parameter,
/// a bound variable, or a free region (early-param or
/// placeholder). A field's declared type can only mention
/// the ADT's own parameters, so the field types after
/// substitution are parameter-free exactly when the
/// instance's arguments are — which is the condition for
/// normalizing them in the driver's codegen typing
/// environment (see `Converter::record_instance_shape`).
pub(crate) fn instance_mentions_param(args: &GenericArgs) -> bool {
    use rustc_public::visitor::Visitable;
    struct MentionsParam(bool);
    impl rustc_public::visitor::Visitor for MentionsParam {
        type Break = ();
        fn visit_ty(
            &mut self,
            ty: &rustc_public::ty::Ty,
        ) -> std::ops::ControlFlow<()> {
            match ty.kind() {
                rustc_public::ty::TyKind::Param(_)
                | rustc_public::ty::TyKind::Bound(..) => {
                    self.0 = true;
                    std::ops::ControlFlow::Break(())
                }
                _ => ty.super_visit(self),
            }
        }
        fn visit_const(
            &mut self,
            c: &rustc_public::ty::TyConst,
        ) -> std::ops::ControlFlow<()> {
            match c.kind() {
                rustc_public::ty::TyConstKind::Param(_)
                | rustc_public::ty::TyConstKind::Bound(..) => {
                    self.0 = true;
                    std::ops::ControlFlow::Break(())
                }
                _ => c.super_visit(self),
            }
        }
        fn visit_reg(
            &mut self,
            reg: &rustc_public::ty::Region,
        ) -> std::ops::ControlFlow<()> {
            match reg.kind {
                rustc_public::ty::RegionKind::ReErased
                | rustc_public::ty::RegionKind::ReStatic
                | rustc_public::ty::RegionKind::ReBound(..) => {
                    std::ops::ControlFlow::Continue(())
                }
                _ => {
                    self.0 = true;
                    std::ops::ControlFlow::Break(())
                }
            }
        }
    }
    let mut screen = MentionsParam(false);
    for arg in &args.0 {
        match arg {
            GenericArgKind::Type(t) => {
                let _ = t.visit(&mut screen);
            }
            GenericArgKind::Const(c) => {
                let _ = c.visit(&mut screen);
            }
            GenericArgKind::Lifetime(r) => {
                let _ = r.visit(&mut screen);
            }
        }
        if screen.0 {
            return true;
        }
    }
    false
}

/// Whether a type mentions an array length the converter
/// cannot resolve by substitution: a const generic parameter
/// (or an unevaluated const) stands where a concrete length
/// belongs (see `Converter::record_instance_shape`).
pub(crate) fn ty_mentions_param_len(ty: &sm::Ty) -> bool {
    match ty {
        sm::Ty::Array(elem, len) => {
            !matches!(len, sm::ArrayLen::Known(_)) || ty_mentions_param_len(elem)
        }
        sm::Ty::Adt(_, args)
        | sm::Ty::FnDef(_, args)
        | sm::Ty::Closure(_, args)
        | sm::Ty::Coroutine(_, args)
        | sm::Ty::CoroutineWitness(_, args)
        | sm::Ty::Alias(_, _, args) => args.iter().any(ty_mentions_param_len),
        sm::Ty::Slice(elem)
        | sm::Ty::RawPtr(elem, _)
        | sm::Ty::Ref(_, elem, _)
        | sm::Ty::Pat(elem, _) => ty_mentions_param_len(elem),
        sm::Ty::FnPtr(sig) => {
            sig.inputs.iter().any(ty_mentions_param_len)
                || ty_mentions_param_len(&sig.output)
        }
        sm::Ty::Dynamic(_, self_ty, _) => ty_mentions_param_len(self_ty),
        sm::Ty::Tuple(ts) => ts.iter().any(ty_mentions_param_len),
        _ => false,
    }
}

pub(crate) fn ty_mentions_alias(ty: &sm::Ty) -> bool {
    match ty {
        sm::Ty::Alias(..) => true,
        sm::Ty::Adt(_, args)
        | sm::Ty::FnDef(_, args)
        | sm::Ty::Closure(_, args)
        | sm::Ty::Coroutine(_, args)
        | sm::Ty::CoroutineWitness(_, args) => args.iter().any(ty_mentions_alias),
        sm::Ty::Array(elem, _)
        | sm::Ty::Slice(elem)
        | sm::Ty::RawPtr(elem, _)
        | sm::Ty::Ref(_, elem, _)
        | sm::Ty::Pat(elem, _) => ty_mentions_alias(elem),
        sm::Ty::FnPtr(sig) => {
            sig.inputs.iter().any(ty_mentions_alias) || ty_mentions_alias(&sig.output)
        }
        sm::Ty::Dynamic(_, self_ty, _) => ty_mentions_alias(self_ty),
        sm::Ty::Tuple(ts) => ts.iter().any(ty_mentions_alias),
        _ => false,
    }
}

/// Whether an ADT instance may be unsized — or otherwise unsafe
/// to ask the compiler for its layout. The check walks the
/// *declared* field types (`FieldDef::ty`, which neither
/// substitutes nor normalizes; the substituting variant
/// `ty_with_args` normalizes projections and fails fatally on
/// ones that do not normalize in a codegen environment). A field
/// may be unsized when its type is a slice, `str`, a trait object,
/// an alias (whose sizedness is not decidable here), a parameter
/// bound to such a type by the instance's arguments, or a nested
/// ADT that may be unsized. Such types are only legal with the
/// unsized field in tail position — except under `trivial_bounds`,
/// where they can sit anywhere and the compiler's layout
/// computation itself ICEs on them (rustc never computes their
/// layouts). Screened instances get no layout entry and stay
/// explicitly unsupported. Parameter resolution goes one frame
/// deep at each ADT boundary, composing the instance arguments
/// with the ambient frame's (see `compose_args`): a parameter
/// the ambient frame leaves unresolved counts as suspect, which
/// only ever skips a query, never enables one.
/// Compose an ADT's instance arguments with the ambient
/// frame's arguments: a parameter of the ambient frame,
/// wherever it appears inside an argument -- directly, or
/// nested inside another ADT, reference, tuple, or the
/// like -- is replaced by the ambient frame's argument for
/// it. The substitution is structural and rebuilds only
/// the shapes `Ty::from_rigid_kind` constructs; a shape it
/// cannot rebuild (an alias type, a constant whose
/// substituted form has no interned identity of its own to
/// take) keeps its argument unsubstituted, exactly as a
/// non-parameter argument was always treated. The walkers
/// below resolve parameters one frame deep; composing at
/// every ADT boundary keeps a parameter chain resolvable
/// across nested generic ADTs (the `ManuallyDrop<T>` field
/// inside `MaybeUninit<T>` is the canonical case) without
/// ever enabling a query the one-frame rule would skip: an
/// argument the ambient frame itself leaves a parameter
/// stays a parameter, and stays suspect.
pub(crate) fn compose_args(
    args: &rustc_public::ty::GenericArgs,
    ambient: Option<&rustc_public::ty::GenericArgs>,
) -> rustc_public::ty::GenericArgs {
    let Some(ambient) = ambient else {
        return args.clone();
    };
    substitute_args(args, ambient, 0, false)
}

/// Compose arguments for the projection screen's field
/// walk (see `compose_args`), additionally resolving a
/// projection argument through the screen's own evidence
/// (see `projection_resolved_ty`) before it crosses the
/// ADT boundary. A projection argument's parameters belong
/// to the enclosing frame: carried into the ADT's frame
/// unresolved, they are read against that frame instead --
/// the alias's own self parameter resolving to the very
/// argument that carries it, until the depth cap gives
/// up. Resolving it here, with the enclosing frame still
/// in hand, substitutes the type the evidence assigns it;
/// a projection the evidence does not determine is left
/// as it was, exactly as `compose_args` leaves it.
pub(crate) fn compose_screen_args(
    args: &rustc_public::ty::GenericArgs,
    ambient: Option<&rustc_public::ty::GenericArgs>,
) -> rustc_public::ty::GenericArgs {
    let Some(ambient) = ambient else {
        return args.clone();
    };
    substitute_args(args, ambient, 0, true)
}

/// `args` with the ambient frame's parameters substituted
/// throughout, recursively (see `compose_args`).
fn substitute_args(
    args: &GenericArgs,
    ambient: &GenericArgs,
    depth: usize,
    resolve_aliases: bool,
) -> GenericArgs {
    GenericArgs(
        args.0
            .iter()
            .map(|a| match a {
                GenericArgKind::Type(t) => {
                    GenericArgKind::Type(substitute_ty(t, ambient, depth, resolve_aliases))
                }
                GenericArgKind::Const(c) => {
                    match substitute_const(c, ambient, depth, resolve_aliases) {
                        Some(c2) => GenericArgKind::Const(c2),
                        None => a.clone(),
                    }
                }
                _ => a.clone(),
            })
            .collect(),
    )
}

/// A type with the ambient frame's parameters substituted
/// throughout, recursively (see `compose_args`). A shape
/// that cannot be rebuilt -- an alias the resolution
/// leaves undetermined (see `compose_screen_args`), a
/// type whose kind cannot be asked, nesting past the
/// depth cap -- is returned unsubstituted.
pub(crate) fn substitute_ty(ty: &Ty, ambient: &GenericArgs, depth: usize, resolve_aliases: bool) -> Ty {
    if depth >= 8 {
        return *ty;
    }
    let Some(kind) = guarded_ty_kind(ty) else {
        return *ty;
    };
    match kind {
        TyKind::Param(p) => match ambient.0.get(p.index as usize) {
            Some(GenericArgKind::Type(t)) => *t,
            _ => *ty,
        },
        TyKind::RigidTy(r) => match r {
            RigidTy::Adt(def, a2) => Ty::from_rigid_kind(RigidTy::Adt(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::Ref(region, inner, mutability) => Ty::from_rigid_kind(RigidTy::Ref(
                region,
                substitute_ty(&inner, ambient, depth + 1, resolve_aliases),
                mutability,
            )),
            RigidTy::RawPtr(inner, mutability) => Ty::from_rigid_kind(RigidTy::RawPtr(
                substitute_ty(&inner, ambient, depth + 1, resolve_aliases),
                mutability,
            )),
            RigidTy::Slice(inner) => Ty::from_rigid_kind(RigidTy::Slice(substitute_ty(
                &inner,
                ambient,
                depth + 1,
                resolve_aliases,
            ))),
            RigidTy::Pat(inner, pat) => Ty::from_rigid_kind(RigidTy::Pat(
                substitute_ty(&inner, ambient, depth + 1, resolve_aliases),
                pat,
            )),
            RigidTy::Array(elem, len) => {
                match substitute_const(&len, ambient, depth + 1, resolve_aliases) {
                    Some(len) => Ty::from_rigid_kind(RigidTy::Array(
                        substitute_ty(&elem, ambient, depth + 1, resolve_aliases),
                        len,
                    )),
                    None => *ty,
                }
            }
            RigidTy::Tuple(ts) => Ty::from_rigid_kind(RigidTy::Tuple(
                ts.iter()
                    .map(|t| substitute_ty(t, ambient, depth + 1, resolve_aliases))
                    .collect(),
            )),
            RigidTy::FnDef(def, a2) => Ty::from_rigid_kind(RigidTy::FnDef(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::FnPtr(sig) => {
                let Binder {
                    value: fsig,
                    bound_vars,
                } = sig;
                Ty::from_rigid_kind(RigidTy::FnPtr(Binder {
                    value: FnSig {
                        inputs_and_output: fsig
                            .inputs_and_output
                            .iter()
                            .map(|t| substitute_ty(t, ambient, depth + 1, resolve_aliases))
                            .collect(),
                        c_variadic: fsig.c_variadic,
                        safety: fsig.safety,
                        abi: fsig.abi,
                    },
                    bound_vars,
                }))
            }
            RigidTy::Closure(def, a2) => Ty::from_rigid_kind(RigidTy::Closure(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::Coroutine(def, a2) => Ty::from_rigid_kind(RigidTy::Coroutine(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::CoroutineClosure(def, a2) => Ty::from_rigid_kind(RigidTy::CoroutineClosure(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::CoroutineWitness(def, a2) => Ty::from_rigid_kind(RigidTy::CoroutineWitness(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::Dynamic(preds, region) => Ty::from_rigid_kind(RigidTy::Dynamic(
                preds
                    .iter()
                    .map(|pred| substitute_pred(pred, ambient, depth + 1, resolve_aliases))
                    .collect(),
                region,
            )),
            _ => *ty,
        },
        TyKind::Alias(SmirAliasKind::Projection, alias) if resolve_aliases => {
            // A projection argument belongs to the
            // enclosing frame (see `compose_screen_args`):
            // substitute the type the screen's own
            // evidence assigns it, completed with the
            // same frame. A projection the evidence does
            // not determine is left unsubstituted.
            match projection_resolved_ty(&alias, Some(ambient), depth + 1) {
                Some(resolved) => substitute_ty(&resolved, ambient, depth + 1, resolve_aliases),
                None => *ty,
            }
        }
        _ => *ty,
    }
}

/// An existential predicate with the ambient frame's
/// parameters substituted throughout (see
/// `substitute_ty`). The binder's own bound variables are
/// the object type's, not the frame's, and are kept.
fn substitute_pred(
    pred: &Binder<ExistentialPredicate>,
    ambient: &GenericArgs,
    depth: usize,
    resolve_aliases: bool,
) -> Binder<ExistentialPredicate> {
    let value = match &pred.value {
        ExistentialPredicate::Trait(tref) => ExistentialPredicate::Trait(ExistentialTraitRef {
            def_id: tref.def_id,
            generic_args: substitute_args(&tref.generic_args, ambient, depth, resolve_aliases),
        }),
        ExistentialPredicate::Projection(p) => {
            let term = match &p.term {
                TermKind::Type(t) => {
                    TermKind::Type(substitute_ty(t, ambient, depth, resolve_aliases))
                }
                TermKind::Const(c) => match substitute_const(c, ambient, depth, resolve_aliases) {
                    Some(c2) => TermKind::Const(c2),
                    None => TermKind::Const(c.clone()),
                },
            };
            ExistentialPredicate::Projection(ExistentialProjection {
                def_id: p.def_id,
                generic_args: substitute_args(&p.generic_args, ambient, depth, resolve_aliases),
                term,
            })
        }
        ExistentialPredicate::AutoTrait(def) => ExistentialPredicate::AutoTrait(*def),
    };
    Binder {
        value,
        bound_vars: pred.bound_vars.clone(),
    }
}

/// A constant with the ambient frame's parameters
/// substituted (see `compose_args`), or `None` when the
/// substituted form cannot be represented: a substituted
/// constant needs an interned identity of its own, which
/// only a constant the frame itself supplies has. An
/// unevaluated constant whose arguments substitute to
/// themselves is returned unchanged.
fn substitute_const(
    c: &TyConst,
    ambient: &GenericArgs,
    depth: usize,
    resolve_aliases: bool,
) -> Option<TyConst> {
    match c.kind() {
        TyConstKind::Param(p) => match ambient.0.get(p.index as usize) {
            Some(GenericArgKind::Const(bound)) => Some(bound.clone()),
            _ => None,
        },
        TyConstKind::Unevaluated(_, args) => {
            if substitute_args(args, ambient, depth, resolve_aliases) == *args {
                Some(c.clone())
            } else {
                None
            }
        }
        _ => Some(c.clone()),
    }
}

fn adt_may_be_unsized(
    def: &AdtDef,
    args: &rustc_public::ty::GenericArgs,
    depth: usize,
) -> bool {
    if depth >= 8 {
        return true;
    }
    def.variants().iter().any(|variant| {
        variant
            .fields()
            .iter()
            .any(|field| ty_may_be_unsized(&field.ty(), Some(args), depth + 1))
    })
}

/// The sizedness of an ADT instance as far as the driver's
/// own walk can tell, refining `adt_may_be_unsized`'s bool:
/// `Unsized` only when a genuinely unsized type (a slice,
/// `str`, or a trait object) is reachable through the fields;
/// `Unknown` when the walk cannot prove either way — the
/// case of field types that mention a type alias, whose
/// sizedness only the compiler's layout computation knows.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Sizedness {
    Sized,
    Unsized,
    Unknown,
}

pub(crate) fn adt_sizedness(
    def: &AdtDef,
    args: &rustc_public::ty::GenericArgs,
    depth: usize,
) -> Sizedness {
    if depth >= 8 {
        return Sizedness::Unknown;
    }
    let mut result = Sizedness::Sized;
    for variant in def.variants() {
        for field in variant.fields() {
            match ty_sizedness(&field.ty(), Some(args), depth + 1) {
                Sizedness::Unsized => return Sizedness::Unsized,
                Sizedness::Unknown => result = Sizedness::Unknown,
                Sizedness::Sized => {}
            }
        }
    }
    result
}

pub(crate) fn ty_sizedness(
    ty: &rustc_public::ty::Ty,
    args: Option<&rustc_public::ty::GenericArgs>,
    depth: usize,
) -> Sizedness {
    if depth >= 8 {
        return Sizedness::Unknown;
    }
    // The composed arguments can resolve a parameter to any
    // concrete type in the program, including one whose kind
    // the bridge cannot hand over (a coroutine-closure type):
    // the question goes through the guarded form, and an
    // unaskable type is `Unknown`, like an alias.
    let Some(kind) = guarded_ty_kind(ty) else {
        return Sizedness::Unknown;
    };
    match kind {
        TyKind::RigidTy(r) => match r {
            RigidTy::Slice(_) | RigidTy::Str | RigidTy::Dynamic(..) => {
                Sizedness::Unsized
            }
            RigidTy::Adt(def, a2) => adt_sizedness(&def, &compose_args(&a2, args), depth),
            RigidTy::Array(elem, _) => ty_sizedness(&elem, args, depth),
            RigidTy::Tuple(ts) => {
                let mut result = Sizedness::Sized;
                for t in ts.iter() {
                    match ty_sizedness(t, args, depth) {
                        Sizedness::Unsized => return Sizedness::Unsized,
                        Sizedness::Unknown => result = Sizedness::Unknown,
                        Sizedness::Sized => {}
                    }
                }
                result
            }
            _ => Sizedness::Sized,
        },
        TyKind::Param(p) => match args.and_then(|a| a.0.get(p.index as usize)) {
            Some(rustc_public::ty::GenericArgKind::Type(t)) => {
                ty_sizedness(t, None, depth + 1)
            }
            _ => Sizedness::Unknown,
        },
        _ => Sizedness::Unknown,
    }
}

fn ty_may_be_unsized(
    ty: &rustc_public::ty::Ty,
    args: Option<&rustc_public::ty::GenericArgs>,
    depth: usize,
) -> bool {
    if depth >= 8 {
        return true;
    }
    // See `ty_sizedness`: the composed arguments can carry a
    // type whose kind cannot be asked for; it counts as
    // possibly unsized, like every shape this walk cannot
    // resolve.
    let Some(kind) = guarded_ty_kind(ty) else {
        return true;
    };
    match kind {
        TyKind::RigidTy(r) => match r {
            RigidTy::Slice(_) | RigidTy::Str | RigidTy::Dynamic(..) => true,
            RigidTy::Adt(def, a2) => {
                adt_may_be_unsized(&def, &compose_args(&a2, args), depth)
            }
            RigidTy::Array(elem, _) => ty_may_be_unsized(&elem, args, depth),
            RigidTy::Tuple(ts) => ts.iter().any(|t| ty_may_be_unsized(t, args, depth)),
            _ => false,
        },
        TyKind::Param(p) => match args.and_then(|a| a.0.get(p.index as usize)) {
            Some(rustc_public::ty::GenericArgKind::Type(t)) => {
                ty_may_be_unsized(t, None, depth + 1)
            }
            _ => true,
        },
        _ => true,
    }
}
