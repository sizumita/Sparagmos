//! Foreign (extern-block) items and the extern-item-impl (EII) implementations they resolve to.

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

/// The instance that provides the body of the extern-item-impl
/// (EII) declaration `inst` names, when it names one. An EII
/// function declaration (`#[eii(name)] pub fn f(..);`, or one
/// carrying a default body) is a foreign item: it has no MIR
/// body of its own, so a call to it stops as a call to a
/// function with no converted body. The body lives in the one
/// implementation registered for the declaration in this
/// compilation - an explicit `#[name] fn g(..) { .. }` or the
/// declaration's own default, which the compiler registers as
/// an implementation too - and that implementation is an
/// ordinary function whose body, instantiated with the
/// declaration's own generic arguments (an implementation
/// carries the declaration's signature), IS the declaration's
/// body. The compiler's `externally_implementable_items`
/// registry names them; an explicit implementation overrides
/// the default (see the selection in the body). `None` when
/// the item is not an EII declaration, when no implementation
/// is registered, or when the registrations do not resolve
/// to one implementation.
pub fn eii_impl_instance(
    inst: &rustc_public::mir::mono::Instance,
) -> Option<rustc_public::mir::mono::Instance> {
    let impl_def = screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal =
            rustc_public::rustc_internal::internal(tcx, inst.def.def_id());
        // The registry is per crate, exactly as for statics
        // (see `eii_impl_static`): a crate's map names the
        // implementations recorded for that crate, so an
        // implementation defined in a different crate than
        // the declaration appears only in its own crate's
        // map. Merge every crate's map, the way the
        // compiler's own EII check does.
        let mut explicit: Vec<rustc_span::def_id::DefId> = Vec::new();
        let mut defaults: Vec<rustc_span::def_id::DefId> = Vec::new();
        for cnum in tcx
            .crates(())
            .iter()
            .chain(std::iter::once(&rustc_span::def_id::LOCAL_CRATE))
        {
            let map = tcx.externally_implementable_items(*cnum);
            if let Some((_, impls)) = map.get(&internal) {
                for (id, imp) in impls {
                    let list = if imp.is_default {
                        &mut defaults
                    } else {
                        &mut explicit
                    };
                    if !list.contains(id) {
                        list.push(*id);
                    }
                }
            }
        }
        // An explicit implementation overrides the default.
        // More than one explicit implementation is a
        // duplication the compiler itself reports, and a
        // declaration carries at most one default, so
        // anything but exactly one chosen registration
        // leaves the declaration unimplemented here.
        let chosen = if explicit.len() == 1 {
            explicit[0]
        } else if explicit.is_empty() && defaults.len() == 1 {
            defaults[0]
        } else {
            return None;
        };
        Some(rustc_public::rustc_internal::stable(chosen))
    })?;
    resolve_instance_guarded(rustc_public::ty::FnDef(impl_def), &inst.args())
}

impl Converter {
    /// Record the seeds of core's generated unicode skiplist
    /// tables (the `core::unicode::unicode_data` property
    /// modules' `SHORT_OFFSET_RUNS` and `OFFSETS` statics) in
    /// [`Converter::extra_static_seeds`].
    ///
    /// The tables' only reader is each module's
    /// `lookup_slow`, which has no MIR body in the shipped
    /// core, so no converted body ever references these
    /// statics and the ordinary seed path — a seed rides
    /// inside a converted `StaticRef` constant — cannot
    /// reach them. The `lookup_slow` runtime model reads the
    /// tables from the program's seed table by name, so they
    /// are recorded directly: every static of every crate
    /// whose name has the table shape is decoded against its
    /// declared type, exactly as a whole-static reference's
    /// seed is decoded. A table whose initializer cannot be
    /// decoded gets no entry; naming it at run time then
    /// fails with the explicit no-seed error, never a
    /// guessed answer.
    pub fn note_unicode_data_tables(&mut self) {
        let mut crates = vec![rustc_public::local_crate()];
        crates.extend(rustc_public::external_crates());
        for def in crates.iter().flat_map(|krate| crate_statics(krate)) {
            let name = def.name();
            if !is_unicode_data_table(&name) || self.extra_static_seeds.contains_key(&name) {
                continue;
            }
            let Ok(ty) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| def.ty()))
            else {
                continue;
            };
            let Some(alloc) = self.static_initializer(&def) else {
                continue;
            };
            if let Some(seed) = self.decode_const(&ty, &alloc, 0, 0) {
                let sm_ty = self.ty(&ty);
                self.extra_static_seeds
                    .insert(name, sm::StaticSeedDef { ty: sm_ty, init: seed });
            }
        }
    }
}

/// The statics of one crate, enumerated under the same
/// containment as the demand passes' foreign scans (see
/// `foreign_fn_defs` in the driver binary): enumerating a
/// crate's items asks the compiler for each item's kind, and
/// a kind query an earlier contained panic poisoned aborts
/// with the compiler's fatal error instead of answering,
/// taking the whole compilation with it. A crate that cannot
/// be enumerated contributes no statics here — the unicode
/// tables live in core, so a foreign crate that fails
/// enumeration loses nothing this pass exists to record —
/// and one unenumerable crate must not abort the conversion.
fn crate_statics(krate: &rustc_public::Crate) -> Vec<mir::mono::StaticDef> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| krate.statics()))
        .unwrap_or_default()
}

/// Whether the item name is one of core's generated unicode
/// skiplist tables: a `SHORT_OFFSET_RUNS` or `OFFSETS`
/// static inside a `unicode_data` property module.
fn is_unicode_data_table(name: &str) -> bool {
    name.contains("::unicode_data::")
        && (name.ends_with("::SHORT_OFFSET_RUNS") || name.ends_with("::OFFSETS"))
}

/// The link symbol the compiler assigns to the static `def`,
/// when it can be asked: `#[no_mangle]` / `#[export_name]`
/// definitions carry their export name, foreign declarations
/// the name they import (a `#[link_name]` included), and every
/// other static its mangled name, which no other static
/// shares. Asking can panic for shapes the compiler cannot
/// name; the question is contained and such a static simply
/// has no symbol here.
fn static_link_symbol(def: &mir::mono::StaticDef) -> Option<String> {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal = rustc_public::rustc_internal::internal(tcx, def.def_id());
        let instance = rustc_middle::ty::Instance::mono(tcx, internal);
        Some(tcx.symbol_name(instance).name.to_string())
    })
}

/// The defined static whose storage a foreign static's import
/// binds to, when the session contains one: a declaration in
/// an `extern` block (or a `#[linkage]` declaration) names a
/// link symbol, and a definition anywhere in the session's
/// crates — the auxiliary crates a test links included —
/// carrying the same symbol is the storage the declaration's
/// references denote. The match is by the compiler's own
/// symbol names, so it holds across crate boundaries and
/// export attributes. A foreign static with no definition in
/// the session (a C library's variable, say) gets `None`.
pub(crate) fn foreign_static_definition(
    def: &mir::mono::StaticDef,
) -> Option<mir::mono::StaticDef> {
    let symbol = static_link_symbol(def)?;
    for krate in
        std::iter::once(rustc_public::local_crate()).chain(rustc_public::external_crates())
    {
        for candidate in crate_statics(&krate) {
            if candidate.def_id() == def.def_id() {
                continue;
            }
            if static_link_symbol(&candidate).as_deref() != Some(symbol.as_str()) {
                continue;
            }
            if static_is_foreign(&candidate) {
                continue;
            }
            return Some(candidate);
        }
    }
    None
}

/// Whether the foreign static `def` imports its symbol with
/// weak linkage (`#[linkage = "extern_weak"]`): the symbol is
/// allowed to be undefined, and an undefined weak symbol's
/// address is the null address, so a program that takes the
/// declaration's value reads the null pointer rather than
/// storage contents.
pub(crate) fn static_is_extern_weak(def: &mir::mono::StaticDef) -> bool {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal = rustc_public::rustc_internal::internal(tcx, def.def_id());
        // A foreign item's `#[linkage]` is recorded in the
        // codegen attrs' import linkage (the field the
        // compiler's own `get_static` applies when it declares
        // the import); the plain linkage field carries the
        // attribute of definitions only.
        Some(
            tcx.codegen_fn_attrs(internal).import_linkage
                == Some(rustc_attr_ir::Linkage::ExternalWeak),
        )
    })
    .unwrap_or(false)
}
