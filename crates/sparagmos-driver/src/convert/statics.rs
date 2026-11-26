//! Foreign (extern-block) statics and the extern-item-impl (EII) storage they resolve to.

use super::*;

/// Whether the static `def` is a foreign static (an item of an
/// `extern` block). A foreign static has no initializer by
/// construction, and asking the compiler to evaluate one ICEs
/// inside the typeck query ("can't type-check body of ..."):
/// evaluating a static initializer goes through the item's MIR
/// body, which a foreign item does not have. The ICE diagnostic
/// poisons the whole compilation even when the panic itself is
/// contained, so the evaluation must not be attempted — callers
/// handle the storage without it (a reference decoded against
/// an uninitialized seed, or no decode at all).
/// Outside a compilation callback there is no context to ask;
/// the static is reported non-foreign, preserving the caller's
/// plain behavior.
pub(crate) fn static_is_foreign(def: &mir::mono::StaticDef) -> bool {
    let Some(tcx) = compiler_tcx() else {
        return false;
    };
    let internal = rustc_public::rustc_internal::internal(tcx, def.def_id());
    tcx.is_foreign_item(internal)
}

/// The static that provides the storage of the extern-item-impl
/// (EII) declaration `def`, when `def` is one. An EII
/// declaration (`#[eii(name)] static X: T;`, or one carrying a
/// default initializer) is a foreign item: it has no
/// initializer of its own and evaluating it panics (see
/// `static_is_foreign`). The value lives in the one
/// implementation registered for the declaration in this
/// compilation - an explicit `#[name] static Y: T = ..;` or
/// the declaration's own default, which the compiler registers
/// as an implementation too - and that implementation is an
/// ordinary static whose storage IS the declaration's storage.
/// The compiler's `externally_implementable_items` registry
/// names them; an explicit implementation overrides the
/// default (see the selection in the body). `None` when the
/// item is not an EII declaration, when no implementation is
/// registered, when the registrations do not resolve to one
/// implementation, or when the implementation cannot be named
/// as a `StaticDef`.
pub(crate) fn eii_impl_static(
    def: &mir::mono::StaticDef,
) -> Option<mir::mono::StaticDef> {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal =
            rustc_public::rustc_internal::internal(tcx, def.def_id());
        // The registry is per crate: a crate's map names the
        // implementations recorded for that crate, so an
        // implementation defined in a different crate than the
        // declaration -- the declaration in an auxiliary crate,
        // the implementation in the crate being compiled, or
        // the other way round -- appears only in its own
        // crate's map, never in the declaration's. The
        // compiler's own EII check merges every crate's map
        // for exactly this reason (see
        // the check_externally_implementable_items pass in
        // rustc_passes), and the resolution does the same:
        // every registration of this declaration, from the
        // local crate and every external crate, is a candidate.
        let mut impl_ids: Vec<rustc_span::def_id::DefId> = Vec::new();
        for cnum in tcx
            .crates(())
            .iter()
            .chain(std::iter::once(&rustc_span::def_id::LOCAL_CRATE))
        {
            let map = tcx.externally_implementable_items(*cnum);
            if let Some((_, impls)) = map.get(&internal) {
                for id in impls.keys() {
                    if !impl_ids.contains(id) {
                        impl_ids.push(*id);
                    }
                }
            }
        }
        if impl_ids.is_empty() {
            return None;
        }
        // The registrations hold every implementation of the
        // declaration, the declaration's own default among
        // them when it has one: the default is registered
        // under an item carrying the declaration's own name.
        // An explicit implementation overrides the default,
        // so with several registrations the implementation
        // is the one whose name is not the declaration's.
        // Several such registrations are a duplication the
        // compiler itself reports; none at all (a default
        // registered beside a same-named item in another
        // module) leaves the declaration unimplemented here.
        let impl_internal = if impl_ids.len() == 1 {
            impl_ids[0]
        } else {
            let decl_name = tcx.item_name(internal);
            let explicit: Vec<_> = impl_ids
                .iter()
                .filter(|id| tcx.item_name(**id) != decl_name)
                .collect();
            if explicit.len() == 1 {
                *explicit[0]
            } else {
                return None;
            }
        };
        let impl_id = rustc_public::rustc_internal::stable(impl_internal);
        let mut crates = vec![rustc_public::local_crate()];
        crates.extend(rustc_public::external_crates());
        crates
            .iter()
            .flat_map(|krate| krate.statics())
            .find(|candidate| candidate.def_id() == impl_id)
    })
}
