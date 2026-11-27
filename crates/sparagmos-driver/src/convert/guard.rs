//! Compiler-context slot, body-materialization and instance-resolution containment, and the emission registries those guards feed.

use super::*;

thread_local! {
    /// The running compilation's compiler context, handed to
    /// the library by the binary's compiler callback (which
    /// receives it from `rustc_public`'s `run_with_tcx` entry).
    /// [`Converter::collapsed_filename_lines`] consults it to
    /// resolve statement spans the way the compiler's own
    /// debuginfo does. The context is valid only for the length
    /// of the callback; the binary clears the slot when the
    /// callback returns. The lifetime is erased to 'static for
    /// storage: the stored value is the callback's own context,
    /// which outlives every conversion performed inside it, and
    /// it is never read outside the callback.
    static COMPILER_TCX: std::cell::Cell<Option<rustc_middle::ty::TyCtxt<'static>>> =
        const { std::cell::Cell::new(None) };
}

/// Install (or clear, with `None`) the compiler context the
/// span resolution consults. Called by the binary's compiler
/// callback around the conversion entry.
pub fn set_compiler_tcx(tcx: Option<rustc_middle::ty::TyCtxt<'_>>) {
    let erased = tcx.map(|t| unsafe {
        std::mem::transmute::<rustc_middle::ty::TyCtxt<'_>, rustc_middle::ty::TyCtxt<'static>>(t)
    });
    COMPILER_TCX.with(|slot| slot.set(erased));
}

pub fn compiler_tcx() -> Option<rustc_middle::ty::TyCtxt<'static>> {
    COMPILER_TCX.with(|slot| slot.get())
}

/// Run `f` — a Stable MIR body materialization — with the
/// diagnostics its eager constant evaluation emits discounted.
///
/// `Instance::body()` and its siblings hand back a body that is
/// already monomorphized with every constant operand evaluated
/// (rustc_public's body builder evaluates them as it converts).
/// That evaluation is speculative from this driver's point of
/// view: the driver materializes the bodies of items and
/// instances the compiler's own monomorphization collector
/// would never collect — unreachable local functions, instances
/// demanded only from dead code, bodies fetched to inspect a
/// shape — and a constant that fails evaluation there (a
/// panicking associated constant behind `if false` is the shape
/// of Issue #417) is reported by the const-eval query as a
/// session error even though the body itself is still produced,
/// with the constant left unevaluated. Left standing in the
/// session's error count, that one report fails the whole
/// compilation of a program rustc accepts.
///
/// The discount is scoped to body materialization on purpose.
/// Which constant failures are compilation errors is decided by
/// the driver's own authorities, never by a side effect of
/// fetching a body: the front end for the constants it evaluates
/// itself before collection begins (statics, constants in item
/// types), and `mono_check`'s replay of the monomorphization
/// collector for the required constants of the instances that
/// collector actually collects. Diagnostics emitted outside a
/// materialization keep their usual effect.
///
/// The discount is recorded, not forgotten, and it is scoped
/// by the class of diagnostic. The diagnostics a materialization
/// can emit fall in two classes with different verdicts:
///
/// - Lint errors (the known-panics reports are the known shape):
///   rustc lints every local body it builds, collected or not,
///   so a lint error from materializing a body is a compilation
///   error under rustc too. These are never discounted into a
///   conversion: the fetch records the body's key in the lint
///   registry, and the compile verdict fails on any entry.
/// - Evaluation errors (E0080, including the validity errors of
///   a body's constant operands): rustc produces these only for
///   the bodies its monomorphization collector actually
///   collects. These are discounted at the fetch, but the
///   body's key (the `def_path_str_with_args` form the Stable
///   MIR name of the instance already has) is recorded, and
///   `mono_check` — which replays the collector — turns a hit
///   on a collected instance into a finding. Bodies the
///   collector never reaches stay discounted for good.
///
/// The classes are told apart by whether the session gained a
/// non-lint error during the fetch. A fetch that gains both
/// classes at once is recorded as an evaluation error; for a
/// collected body the collector replay still fails it, so the
/// only residue is an uncollected body carrying both a lint
/// violation and a failing constant, which converts.
pub fn with_body_materialization<R>(key: &str, f: impl FnOnce() -> R) -> R {
    let Some(tcx) = compiler_tcx() else {
        return f();
    };
    let dcx = tcx.dcx();
    let errors_before = dcx.err_count();
    let non_lint_before = dcx.has_errors_excluding_lint_errors().is_some();
    // The bug-witness snapshot for the crash registry below:
    // taken before the fetch, diffed after it.
    let witnesses_before = compiler_bug_witness_len();
    // Contained, not handled: a panic out of the fetch keeps its
    // payload for the driver's own containment layers, but the
    // diagnostics emitted before it belong to the same
    // speculative materialization and are discounted either way.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    // A compiler-bug panic inside the fetch (see
    // `captured_panic_is_compiler_bug`) is recorded under the
    // body's key, beside the diagnostic registries below: the
    // fetch may be speculative, so the record is inert unless
    // the collector replay collects the body — a hit there is
    // a compiler crash, the verdict native reaches when its
    // own materialization of a collected body dies on the
    // compiler's bug (see `materialization_bug`).
    if !key.is_empty() {
        if let Some(record) = compiler_bug_witness_since(witnesses_before) {
            MATERIALIZATION_BUGS.with(|map| {
                map.borrow_mut()
                    .entry(key.to_string())
                    .or_insert(record);
            });
        }
    }
    if dcx.err_count() > errors_before {
        let non_lint_after = dcx.has_errors_excluding_lint_errors().is_some();
        if non_lint_after && !non_lint_before {
            note_materialization_emission(key);
        } else {
            note_materialization_lint(key);
        }
        dcx.reset_err_count();
    }
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

thread_local! {
    /// Keys of the bodies whose materialization emitted
    /// evaluation diagnostics that `with_body_materialization`
    /// discounted, in the `def_path_str_with_args` form (a
    /// generic definition's own body is keyed by its plain def
    /// path). Written during collection, read by the compile
    /// verdict (`mono_check::check_required_consts`); one
    /// compilation runs per process, so nothing clears it.
    static MATERIALIZATION_EMISSIONS: std::cell::RefCell<std::collections::HashSet<String>> =
        std::cell::RefCell::new(std::collections::HashSet::new());

    /// Keys of the bodies whose materialization emitted a lint
    /// error. Same lifecycle as `MATERIALIZATION_EMISSIONS`.
    static MATERIALIZATION_LINTS: std::cell::RefCell<std::collections::HashSet<String>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

thread_local! {
    /// Keys of the bodies whose materialization raised a
    /// compiler-bug panic (see
    /// `captured_panic_is_compiler_bug`), with the panic's
    /// record. Same lifecycle as `MATERIALIZATION_EMISSIONS`:
    /// written during collection, read by the compile verdict
    /// (the collector replay in `mono_check`); one compilation
    /// runs per process, so nothing clears it.
    static MATERIALIZATION_BUGS: std::cell::RefCell<
        std::collections::HashMap<String, String>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The compiler-bug panic record of the materialization of
/// the body named by `key`, if it raised one (see
/// `with_body_materialization`).
pub fn materialization_bug(key: &str) -> Option<String> {
    MATERIALIZATION_BUGS.with(|map| map.borrow().get(key).cloned())
}

fn note_materialization_emission(key: &str) {
    if key.is_empty() {
        return;
    }
    MATERIALIZATION_EMISSIONS.with(|set| {
        set.borrow_mut().insert(key.to_string());
    });
}

fn note_materialization_lint(key: &str) {
    if key.is_empty() {
        return;
    }
    MATERIALIZATION_LINTS.with(|set| {
        set.borrow_mut().insert(key.to_string());
    });
}

/// Run `f` — an instance resolution on the demand or
/// reachability path — with the diagnostics it emits
/// discounted, reporting beside the outcome whether any were.
///
/// Resolution is speculative in the same way body
/// materialization is (see `with_body_materialization`): the
/// driver's demand passes resolve instances the compiler's
/// own monomorphization collector never resolves in that form
/// — the collector resolves with the `try_` variants and
/// simply does not collect an item that fails to normalize —
/// and a resolution can fail inside the compiler in ways that
/// are reported as session diagnostics: an instance whose
/// signature mentions an associated-type projection the
/// erasing-regions folder cannot normalize (a query cache's
/// fingerprint function-pointer type is the shape Issue #459
/// isolated) makes the non-try normalization report an
/// internal error, and left standing in the session's error
/// count that one report fails the whole compilation of a
/// program rustc accepts, even though the resolution itself
/// was contained and recorded as failed by its caller. (The
/// budget-4096 all-loss itself fired at a sibling speculative
/// operation - the drop demand's field substitution, screened
/// in `main::type_tree_has_alias` - whose delayed bug the
/// same session poisoning carried; this guard closes the
/// resolution side of the class.)
///
/// The discount follows the materialization discipline. When
/// the run raised the session's non-lint error count, the
/// error count is reset and the returned flag is set: a
/// resolution that emitted diagnostics is a failed resolution
/// whatever it returned, and the caller records it with its
/// usual "instance resolution failed" reason. The caller also
/// records the emission (see `resolution_emitted`) under the
/// attempted instantiation's own name — never under the
/// definition's argument-free key. Unlike a body
/// materialization, whose diagnostics are properties of the
/// one generic body fetched per definition, a resolution's
/// outcome depends on the generic arguments attempted (the
/// resolution query normalizes them), so a record keyed by
/// the definition would condemn every other instantiation
/// of it. `mono_check`'s collector replay turns a record on
/// an instance the collector actually collects into a
/// finding, so a diagnostic rustc would report for a
/// collected instance still fails the compilation.
/// Lint-class growth is never discounted — a resolution is
/// not expected to lint at all, and leaving the growth
/// standing keeps its usual effect, exactly as at a
/// materialization.
///
/// A panic out of the resolution is discounted the same way;
/// the payload is returned for the caller to resume or
/// contain as it already does.
pub(crate) fn with_instance_resolution_contained<R>(
    f: impl FnOnce() -> R,
) -> (Result<R, Box<dyn std::any::Any + Send>>, bool) {
    let Some(tcx) = compiler_tcx() else {
        return (Ok(f()), false);
    };
    let dcx = tcx.dcx();
    let errors_before = dcx.err_count();
    let non_lint_before = dcx.has_errors_excluding_lint_errors().is_some();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    let mut emitted = false;
    if dcx.err_count() > errors_before {
        let non_lint_after = dcx.has_errors_excluding_lint_errors().is_some();
        if non_lint_after && !non_lint_before {
            dcx.reset_err_count();
            emitted = true;
        }
    }
    (result, emitted)
}

/// Resolve a function item instance on the demand path under
/// `with_instance_resolution_contained`, collapsing every
/// failure — a resolution error, a contained panic, or
/// diagnostics emitted on the way — to `None`: on the demand
/// path each is the same "instance resolution failed"
/// outcome, which the caller records as such. An emission is
/// recorded under the resolved instance's own name when the
/// resolution produced one before failing. When it produced
/// none there is no record: an instantiation the resolution
/// fails is one the collector's identical `try_resolve` does
/// not collect either, so a record could never legitimately
/// fire for it — and keyed by the definition instead, it
/// would fire against every collected instantiation of the
/// definition, condemning instantiations that resolve
/// cleanly.
pub fn resolve_instance_guarded(
    def: rustc_public::ty::FnDef,
    args: &rustc_public::ty::GenericArgs,
) -> Option<Instance> {
    let (result, emitted) = with_instance_resolution_contained(|| {
        Instance::resolve(def, args)
    });
    let inst = match result {
        Ok(resolved) => resolved.ok()?,
        Err(_payload) => {
            // This resolver contains the panic: it becomes the
            // "instance resolution failed" outcome, so the
            // resolver consumes the panic's hook record, as
            // every other containment in this module does. The
            // record is first-wins and the location-based
            // classification of a later panic peeks at the slot
            // without clearing it, so a record left standing
            // here would be read as that later panic's own and
            // misclassify it.
            let _ = take_captured_panic();
            return None;
        }
    };
    if emitted {
        note_resolution_emission(&inst.name());
        return None;
    }
    record_fn_instance(&inst);
    Some(inst)
}

/// The closure counterpart of `resolve_instance_guarded`.
pub fn resolve_closure_guarded(
    def: rustc_public::ty::ClosureDef,
    args: &rustc_public::ty::GenericArgs,
    kind: rustc_public::ty::ClosureKind,
) -> Option<Instance> {
    let (result, emitted) = with_instance_resolution_contained(|| {
        Instance::resolve_closure(def, args, kind)
    });
    let inst = match result {
        Ok(resolved) => resolved.ok()?,
        Err(_payload) => {
            // Contained panic: consume its hook record, exactly
            // as in `resolve_instance_guarded`.
            let _ = take_captured_panic();
            return None;
        }
    };
    if emitted {
        note_resolution_emission(&inst.name());
        return None;
    }
    record_fn_instance(&inst);
    Some(inst)
}

thread_local! {
    /// Keys of the instances whose demand-path resolution
    /// emitted diagnostics that the resolution discount
    /// recorded: only the resolved instance's own name (the
    /// `def_path_str_with_args` form the collector replay
    /// consults), and only when the resolution produced that
    /// instance before failing. Written during collection,
    /// read by the compile verdict
    /// (`mono_check::check_required_consts`); one compilation
    /// runs per process, so nothing clears it.
    static RESOLUTION_EMISSIONS: std::cell::RefCell<std::collections::HashSet<String>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

fn note_resolution_emission(key: &str) {
    if key.is_empty() {
        return;
    }
    RESOLUTION_EMISSIONS.with(|set| {
        set.borrow_mut().insert(key.to_string());
    });
}

/// Whether resolving the instance named by `key` emitted
/// diagnostics that were discounted at the resolution
/// (see `resolve_instance_guarded`).
pub fn resolution_emitted(key: &str) -> bool {
    RESOLUTION_EMISSIONS.with(|set| set.borrow().contains(key))
}

/// Whether materializing the body named by `key` emitted
/// evaluation diagnostics that were discounted at the fetch
/// (see `with_body_materialization`).
pub fn materialization_emitted(key: &str) -> bool {
    MATERIALIZATION_EMISSIONS.with(|set| set.borrow().contains(key))
}

/// A body whose materialization emitted a lint error, if any:
/// rustc lints every local body, so each one is a compilation
/// error there (see `with_body_materialization`).
pub fn materialization_lint() -> Option<String> {
    MATERIALIZATION_LINTS.with(|set| set.borrow().iter().next().cloned())
}

/// Whether the running compilation's crate graph defines the
/// `drop_glue` lang item. A crate compiled against a minicore
/// (which defines only the lang items it names itself) has none:
/// asking the compiler to resolve `drop_in_place` there emits a
/// fatal "requires `drop_glue` lang_item" diagnostic that no
/// `catch_unwind` can contain, so drop-glue certification must
/// not ask (see `certify_drop_trivial` in the binary). Outside a
/// compilation callback there is no graph to consult; the item
/// is reported defined, preserving the caller's plain behavior.
pub fn drop_glue_lang_item_defined() -> bool {
    let Some(tcx) = compiler_tcx() else {
        return true;
    };
    tcx.lang_items()
        .get(rustc_attr_ir::lang_items::LangItem::DropGlue)
        .is_some()
}

/// The Stable MIR `DefId` of the local item the
/// `#[panic_handler]` attribute marks, when the crate
/// defines one: the definition the unwinder delivers every
/// panic to in a `no_std` program. (In this toolchain the
/// panic handler is an attribute-marked item, not a lang
/// item.) The allocation-error model delivers alloc's
/// default (`__rdl`) panic to the same definition. Outside a
/// compilation callback there is no graph to consult and the
/// item is reported absent.
pub fn panic_handler_def() -> Option<rustc_public::DefId> {
    let tcx = compiler_tcx()?;
    let internal = tcx
        .lang_items()
        .get(rustc_attr_ir::lang_items::LangItem::PanicImpl)?;
    Some(rustc_public::rustc_internal::stable(internal))
}
