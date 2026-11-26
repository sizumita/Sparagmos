//! Debugger display names of function instances.
//!
//! A debugger's function listing (gdb's `info functions`) names
//! every monomorphized instance the way the compiler's own debug
//! info does: the scope is the item's def-path rendered with
//! `{impl#N}` / `{closure#N}` segments — or, for a method of an
//! inherent impl, the instantiated self type — and the item name
//! carries the instance's generic arguments
//! (`function_names::GenericStruct<i32, i32>::impl_function<i32, i32>`,
//! `function_names::{impl#1}::trait_function`). The interpreter's
//! program keys spell instances differently (the demangled form,
//! e.g. `<TestStruct1 as TestTrait1>::trait_function`), so while
//! collecting a debug session's program the driver records, for
//! every instance its demand path resolves, the display name and
//! the declaration signature the debug info would carry, computed
//! with the compiler's own debuginfo name machinery
//! (`rustc_codegen_ssa::debuginfo::type_names`, the same code
//! that names DWARF DIEs). The debugger interface lists
//! instances from this table (see `ir::Program::fn_display_names`).

use super::*;

use std::sync::atomic::{AtomicBool, Ordering};

/// Whether this process is collecting a debugger session's
/// program (the `debug` mode), set once from the parsed mode
/// before the compiler runs. Instance recording is gated on it:
/// the table serves only the debugger interface, and computing
/// display names for every resolved instance of an ordinary
/// conversion would tax modes that never read them.
static DEBUG_SESSION: AtomicBool = AtomicBool::new(false);

/// Mark whether this process collects a debugger session's
/// program. Called by the binary alongside its other mode flags.
pub fn set_debug_session(on: bool) {
    DEBUG_SESSION.store(on, Ordering::Relaxed);
}

thread_local! {
    /// Instances resolved on the guarded demand path during
    /// this compilation, deduplicated by their program key
    /// (the `Instance::name` spelling, which is unique per
    /// instance), in resolution order.
    static FN_INSTANCE_RECORDS: std::cell::RefCell<(HashSet<String>, Vec<Instance>)> =
        std::cell::RefCell::new((HashSet::new(), Vec::new()));
}

/// Record a resolved instance for the display-name table.
/// Called by the guarded resolvers on every success; a no-op
/// outside a debug session.
pub(crate) fn record_fn_instance(inst: &Instance) {
    if !DEBUG_SESSION.load(Ordering::Relaxed) {
        return;
    }
    FN_INSTANCE_RECORDS.with(|records| {
        let mut records = records.borrow_mut();
        if records.0.insert(inst.name()) {
            records.1.push(inst.clone());
        }
    });
}

/// Drain the recorded instances and compute the display-name
/// table: one `(display name, signature)` pair per distinct
/// instance, the signature being the declaration tail a debug
/// info listing prints after the name — `(params)` plus
/// ` -> ret` for a non-unit, non-never return. Instances whose
/// entry cannot be computed (an exotic instance kind, a name
/// computation that panics) are skipped under the resolution
/// containment discipline. Empty outside a debug session.
pub fn drain_fn_display_names() -> Vec<(String, String)> {
    let instances = FN_INSTANCE_RECORDS.with(|records| {
        let mut records = records.borrow_mut();
        records.0.clear();
        std::mem::take(&mut records.1)
    });
    if instances.is_empty() {
        return Vec::new();
    }
    let Some(tcx) = compiler_tcx() else {
        return Vec::new();
    };
    let mut seen: HashSet<String> = HashSet::new();
    let mut table = Vec::new();
    for inst in &instances {
        let (result, _emitted) =
            with_instance_resolution_contained(|| fn_display_entry(tcx, inst));
        if let Ok(Some((display, signature))) = result {
            if seen.insert(display.clone()) {
                table.push((display, signature));
            }
        }
    }
    table
}

/// The display name and declaration signature of one resolved
/// instance, computed the way the compiler's debuginfo computes
/// them. The name mirrors `dbg_scope_fn` in
/// `rustc_codegen_llvm`'s debuginfo module: the unqualified item
/// name (`push_item_name` renders `{impl#N}` / `{closure#N}`
/// segments) plus the instance's generic arguments truncated to
/// the enclosing function's generics and rendered by
/// `push_generic_args`. The scope mirrors
/// `get_containing_scope`: for a method of an inherent impl
/// whose instantiated self type is an ADT (other than `Box`),
/// the self type's debuginfo name — the subprogram DIE is a
/// child of the type's DIE — and otherwise the parent item's
/// qualified name, whose impl segments spell `{impl#N}`.
fn fn_display_entry(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    inst: &Instance,
) -> Option<(String, String)> {
    use rustc_codegen_ssa::debuginfo::type_names;
    let internal = rustc_public::rustc_internal::internal(tcx, inst.clone());
    let def_id = internal.def_id();
    let mut name = String::with_capacity(64);
    type_names::push_item_name(tcx, def_id, false, &mut name);
    let enclosing = tcx.typeck_root_def_id(def_id);
    let generics = tcx.generics_of(enclosing);
    let args = internal.args.truncate_to(tcx, generics);
    type_names::push_generic_args(
        tcx,
        tcx.normalize_erasing_regions(
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            rustc_middle::ty::Unnormalized::new_wip(args),
        ),
        &mut name,
    );
    let scope = match tcx.inherent_impl_of_assoc(def_id) {
        Some(impl_id) => {
            let self_ty = tcx.instantiate_and_normalize_erasing_regions(
                internal.args,
                rustc_middle::ty::TypingEnv::fully_monomorphized(),
                tcx.type_of(impl_id),
            );
            match self_ty.kind() {
                rustc_middle::ty::TyKind::Adt(def, _) if !def.is_box() => {
                    type_names::compute_debuginfo_type_name(tcx, self_ty, true)
                }
                _ => qualified_parent_name(tcx, def_id)?,
            }
        }
        None => qualified_parent_name(tcx, def_id)?,
    };
    let signature = if matches!(
        tcx.def_key(def_id).disambiguated_data.data,
        rustc_hir::definitions::DefPathData::Closure
    ) {
        // A closure's or coroutine's subprogram DIE carries no
        // return type and a single formal parameter, the
        // environment pointer: `*mut <env type name>`, where the
        // env type is the closure type itself under its debuginfo
        // name (the `{closure_env#N}` form with the enclosing
        // function's arguments).
        let env_ty = tcx.normalize_erasing_regions(
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            rustc_middle::ty::Unnormalized::new_wip(internal.ty(
                tcx,
                rustc_middle::ty::TypingEnv::fully_monomorphized(),
            )),
        );
        let env = type_names::compute_debuginfo_type_name(tcx, env_ty, true);
        format!("(*mut {env})")
    } else {
        let sig = internal
            .ty(tcx, rustc_middle::ty::TypingEnv::fully_monomorphized())
            .fn_sig(tcx);
        let sig = tcx.instantiate_bound_regions_with_erased(sig);
        // The debuginfo name machinery is fed the types codegen
        // would show it: monomorphised and normalised. The
        // instantiated signature still carries the declaration's
        // projection aliases (a trait method's associated-type
        // output, `Index::Output` for a `Vec` index), which the
        // name computation rejects outright, so normalise the
        // signature the way the generic arguments above are
        // normalised before naming any of its types.
        let sig = tcx.normalize_erasing_regions(
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            rustc_middle::ty::Unnormalized::new_wip(sig),
        );
        let mut s = String::from("(");
        for (index, input) in sig.inputs().iter().enumerate() {
            if index > 0 {
                s.push_str(", ");
            }
            s.push_str(&type_names::compute_debuginfo_type_name(tcx, *input, true));
        }
        s.push(')');
        match sig.output().kind() {
            rustc_middle::ty::TyKind::Tuple(tys) if tys.is_empty() => {}
            rustc_middle::ty::TyKind::Never => {}
            _ => {
                s.push_str(" -> ");
                s.push_str(&type_names::compute_debuginfo_type_name(
                    tcx,
                    sig.output(),
                    true,
                ));
            }
        }
        s
    };
    Some((format!("{scope}::{name}"), signature))
}

/// The qualified name of an item's parent item: the crate name
/// for a crate-root child, `crate::Mod` for a module member,
/// `crate::{impl#N}` for an impl member, the enclosing function
/// for a closure.
fn qualified_parent_name(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    def_id: rustc_span::def_id::DefId,
) -> Option<String> {
    use rustc_codegen_ssa::debuginfo::type_names;
    let parent = tcx.def_key(def_id).parent?;
    let parent_id = rustc_span::def_id::DefId {
        krate: def_id.krate,
        index: parent,
    };
    let mut s = String::new();
    type_names::push_item_name(tcx, parent_id, true, &mut s);
    Some(s)
}
