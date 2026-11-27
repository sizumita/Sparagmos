//! dyn trait-object helpers: trait discovery and unsize pointees.


use sparagmos_mir as mir;


use super::converter::*;


/// The trait bounds of the `dyn` type inside `ty`, if any: the
/// search goes through references and raw pointers and into ADT
/// generic arguments (a `Box<dyn Trait>` / `Rc<dyn Trait>` target
/// carries the `dyn` type in its argument).
pub(super) fn dyn_traits_in(ty: &mir::Ty) -> Option<Vec<String>> {
    match ty {
        mir::Ty::Dynamic(traits, _, _) => Some(traits.clone()),
        mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => dyn_traits_in(inner),
        mir::Ty::Adt(_, args) => args.iter().find_map(dyn_traits_in),
        _ => None,
    }
}

/// Whether the `dyn` type inside `ty` is reached through
/// reference and raw-pointer layers only (no ADT generic
/// argument in between): a cast whose target has this shape
/// and whose source names a trait object under the same
/// layers reborrows the stored trait-object pointer instead
/// of erasing a new value.
pub(super) fn dyn_reached_through_refs_only(ty: &mir::Ty) -> bool {
    match dyn_path_in(ty) {
        Some(path) => path
            .iter()
            .all(|step| matches!(step, DynStep::Ref | DynStep::RawPtr)),
        None => false,
    }
}

/// The sized pointee an unsizing coercion starts from: the inner
/// type of a reference or raw pointer, the first generic argument
/// of a pointer-shaped ADT (`Box`, `Rc`, `Arc` share the shape), or
/// the type itself for a bare value operand.
pub(super) fn unsize_pointee(ty: &mir::Ty) -> &mir::Ty {
    match ty {
        mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => inner,
        mir::Ty::Adt(_, args) => args.first().unwrap_or(ty),
        other => other,
    }
}

/// One wrapper layer on the way from a cast's target type to
/// the `dyn` type inside it: a reference layer, a raw-pointer
/// layer, or the generic-argument index of an ADT layer.
#[derive(Clone, Copy)]
enum DynStep {
    Ref,
    RawPtr,
    Arg(usize),
}

/// The wrapper path from a type to the `dyn` type inside it (the
/// traversal of `dyn_traits_in`, recording each layer).
fn dyn_path_in(ty: &mir::Ty) -> Option<Vec<DynStep>> {
    match ty {
        mir::Ty::Dynamic(..) => Some(Vec::new()),
        mir::Ty::Ref(_, inner, _) => {
            let mut path = dyn_path_in(inner)?;
            path.insert(0, DynStep::Ref);
            Some(path)
        }
        mir::Ty::RawPtr(inner, _) => {
            let mut path = dyn_path_in(inner)?;
            path.insert(0, DynStep::RawPtr);
            Some(path)
        }
        mir::Ty::Adt(_, args) => {
            for (i, arg) in args.iter().enumerate() {
                if let Some(mut path) = dyn_path_in(arg) {
                    path.insert(0, DynStep::Arg(i));
                    return Some(path);
                }
            }
            None
        }
        _ => None,
    }
}

/// The source type a `dyn` path names (see `dyn_path_in`): apply
/// each wrapper step of the target to the source. `None` when
/// the shapes diverge (the caller falls back to the heuristic
/// walk in `unsize_concrete_pointee`).
fn pointee_at_path<'t>(mut ty: &'t mir::Ty, path: &[DynStep]) -> Option<&'t mir::Ty> {
    for step in path {
        ty = match (step, ty) {
            (DynStep::Ref, mir::Ty::Ref(_, inner, _)) => inner,
            (DynStep::RawPtr, mir::Ty::RawPtr(inner, _)) => inner,
            (DynStep::Arg(i), mir::Ty::Adt(_, args)) => args.get(*i)?,
            _ => return None,
        };
    }
    Some(ty)
}

/// The concrete type an unsizing cast's *value* answers for: the
/// trait object the cast forms is a `dyn` of the type the
/// target's own wrapper path names in the source — the layers
/// above the `dyn` are the pointer the value travels through,
/// and everything at the path's end is the trait's `Self`,
/// wrapper layers included: `Box<Box<T>>` cast to `Box<dyn
/// Trait>` carries the concrete tag `Box<T>` (the trait is
/// implemented for `Box<T>`), while `Pin<Box<T>>` cast to
/// `Pin<Box<dyn Trait>>` names `T`. The tag must name that type
/// (see `Value::DynRef`): it is the vtable a dispatch looks up.
/// When the target's path cannot be followed in the source, the
/// heuristic walk below applies: the first step is the plain
/// pointee step (any ADT gives up its first argument) and
/// further steps only pass through the standard pointer wrappers
/// (`Pin`, `Box`, `Rc`, `Arc`, the `Weak`s) and references, so a
/// user type in the chain stops the walk.
pub(super) fn unsize_concrete_pointee<'t>(
    conv: &Converter<'_>,
    src: &'t mir::Ty,
    target: &mir::Ty,
) -> &'t mir::Ty {
    if let Some(path) = dyn_path_in(target) {
        if let Some(ty) = pointee_at_path(src, &path) {
            return ty;
        }
    }
    let mut ty = unsize_pointee(src);
    loop {
        match ty {
            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => ty = inner,
            mir::Ty::Adt(def, args) => {
                let name = conv
                    .names
                    .get(def)
                    .or_else(|| conv.def_names.get(def))
                    .map(|n| n.as_str());
                match name {
                    Some(
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
                        | "alloc::sync::Weak",
                    ) => match args.first() {
                        Some(inner) => ty = inner,
                        None => return ty,
                    },
                    _ => return ty,
                }
            }
            other => return other,
        }
    }
}
