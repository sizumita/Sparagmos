//! Monomorphization validity checks that rustc performs while
//! monomorphizing a crate for code generation and that MIR
//! conversion alone does not reproduce.
//!
//! MIR conversion is deliberately permissive: it translates the
//! bodies Stable MIR hands over, and a body that translates cleanly
//! is reported as converted. Some programs nevertheless fail to
//! compile under rustc because a check that only runs at
//! monomorphization time rejects them. When the driver also skipped
//! that check, its verdict disagreed with rustc's on
//! failure-expecting tests (the "expected failure but conversion
//! succeeded" bucket). The checks here mirror rustc's own, over the
//! same evidence rustc uses — the monomorphic bodies that were
//! converted, with the call targets' substituted types:
//!
//! - Invalid SIMD intrinsic monomorphizations (rustc's E0511, from
//!   `generic_simd_intrinsic` in rustc_codegen_llvm): every
//!   `simd_*` intrinsic requires its first argument's type to be a
//!   SIMD vector type, except `simd_splat` (whose *return* type
//!   must be a vector) and `simd_select_bitmask` (whose *second*
//!   argument must be a vector). A scalar where a vector is
//!   required is a hard compile error under rustc.
//! - Invalid SIMD layouts: a SIMD type with zero lanes or more
//!   lanes than the monomorphization lane limit has no layout;
//!   rustc rejects the crate when code generation asks for one.
//!   Every local of a converted body is materialized by code
//!   generation, so a SIMD-typed local whose layout does not exist
//!   is the same failure.
//! - Types too big for the target architecture: a type whose size
//!   exceeds the target's object-size bound has no layout; rustc's
//!   layout computation itself reports `values of the type ... are
//!   too big for the target architecture`, and monomorphization
//!   surfaces that failure for every type it must materialize.
//!   The check asks the compiler's own layout query (the same one
//!   the driver's layout table uses) about every local type of a
//!   converted body, and about the type arguments of calls whose
//!   lowering is defined by those layouts: intrinsic calls, which
//!   code generation lowers in the caller from the type arguments'
//!   layouts, and the `core::mem` size/alignment measuring
//!   functions (`size_of`, `align_of`, `size_of_val`, ...), whose
//!   instantiation evaluates the measured type's layout. Only the
//!   layout query's own too-big error counts; any other layout
//!   failure (unsized, parametric, uncomputable here) is left
//!   alone, exactly as for the SIMD layout check.
//!
//! - Invalid atomic intrinsic monomorphizations (also E0511,
//!   from `codegen_intrinsic_call` in rustc_codegen_ssa): the
//!   `atomic_*` intrinsics require the type of the value at the
//!   memory operand to be a basic integer or raw pointer type;
//!   the read-modify-write arithmetic group additionally
//!   requires the operand value to have that same type (or
//!   `usize` when the memory type is a raw pointer).
//! - The per-intrinsic SIMD argument rules of
//!   `generic_simd_intrinsic` beyond the entry check above:
//!   element-type requirements for the saturating arithmetic,
//!   bitmask, gather/scatter, masked load/store and reduction
//!   intrinsics (the ordered reductions' initial value included),
//!   length and element agreement for shuffles,
//!   selects and comparisons, insert/extract element agreement,
//!   and the constant index bounds of `simd_shuffle`,
//!   `simd_insert` and `simd_extract` (the index values are read
//!   from the call's constant operand, the same evidence code
//!   generation reads; named constants reach this check already
//!   evaluated by the compiler's own constant conversion).
//!
//! - Required constants of collected instances (rustc's E0080 at
//!   monomorphization time): the mono collector evaluates every
//!   required constant of every instance it collects — dead code
//!   included — and follows the items a body mentions (calls,
//!   drops, unsizing casts, closures recorded before optimization)
//!   to instances whose own constants it then evaluates the same
//!   way. Stable MIR bodies are post-optimization, so a mention
//!   that lives only in dead code is gone from them and the
//!   failing constant behind it is never demanded; conversion
//!   alone therefore succeeds where rustc fails. The check below
//!   (`check_required_consts`) replays the collector's walk over
//!   the compiler's internal bodies, which keep both the required
//!   constants and the mentioned items from before optimization.
//!
//! - The collector's own per-instance monomorphization checks
//!   (the `check_mono_item` query: `mono_checks::abi_check`'s
//!   target-feature and ABI rules, and `mono_checks::move_check`'s
//!   `large_assignments` lint): the collector forces that query
//!   for every instance it collects, before scanning the
//!   instance's items, and the diagnostics it emits are the
//!   compilation's own. The walk below forces the same query for
//!   every instance it processes, and a rise in the session's
//!   diagnostic error count while forcing it is the rejection
//!   rustc reports for that instance.
//!
//! - Duplicate link symbols over the collected mono items
//!   (rustc_monomorphize's partitioning asserts all mono-item
//!   symbols are distinct, fatally), and the entry symbol
//!   defined both by a mono item and by the entry wrapper code
//!   generation synthesizes (`rustc_codegen_ssa`'s
//!   multiple-main check): the symbol of every instance the
//!   walk collected, of every local static and of every local
//!   `global_asm!` item is the compiler's own `symbol_name`, so
//!   a name two distinct items share is the same failure.
//!
//! - The panic-runtime validation of the crate graph
//!   (`rustc_metadata::dependency_format::verify_ok`): code
//!   generation forces the `dependency_formats` query, whose
//!   computation validates that at most one panic runtime is
//!   linked and that its strategy — and every linked crate's
//!   required strategy — is compatible with the session's.
//!   The driver's session emits only metadata, so the query's
//!   lists come back empty and the validation never runs; the
//!   check below mirrors `verify_ok` over the linkage list the
//!   computation's static path produces (see
//!   `panic_runtime_finding`).
//!
//! - The LTO / prefer-dynamic session validation: rustc
//!   rejects `-C lto` (thin or fat) combined with
//!   `-C prefer-dynamic` when every output the crate produces
//!   is one LTO is performed for -- 'staticlib', 'bin' and
//!   'cdylib' ("cannot prefer dynamic linking when performing
//!   LTO"). The validation reads only the session's options
//!   and crate types, but rustc performs it as code generation
//!   starts, which a metadata-only session never reaches; the
//!   check below replays it over the driver's own session
//!   state (see `lto_prefer_dynamic_finding`).
//!
//! - Calls to intrinsics that must be overridden but are not
//!   (rustc_codegen_ssa's code generation raises an internal
//!   error — "intrinsic ... must be overridden by codegen
//!   backend, but isn't" — when such a call reaches it
//!   unshimmed): a `#[rustc_intrinsic]` item declared without
//!   a body has no fallback, so every backend must shim every
//!   call to it; code generation serves a call from its
//!   generic intrinsic handling first and the backend's own
//!   shims second, and a call neither serves is the error.
//!   The shim set that decides the verdict is code
//!   generation's own name-based dispatch, which is wider
//!   than this interpreter's builtin registry: the registry
//!   models the intrinsics the interpreter can execute,
//!   while code generation also lowers intrinsics the
//!   interpreter leaves unmodelled (the sized float
//!   families, the prefetch family, and a handful of
//!   singletons), whose calls stay ordinary conversion
//!   skips here. `check_intrinsic_overrides` therefore
//!   consults both sets — the dispatch conversion itself
//!   performs for a body-less callee, and the positively
//!   verified codegen-only list in
//!   `codegen_served_unmodelled` — and reports the call
//!   only when neither serves it. An intrinsic with a
//!   fallback body is never reported: the body is the
//!   fallback, and leaving it unshimmed is legal. A
//!   `#[rustc_comptime]` intrinsic is never reported
//!   either: the compiler confines its calls to constant
//!   contexts, which are evaluated before code generation,
//!   so no call to one ever reaches the backend (see
//!   `comptime_intrinsic`).
//!
//! - `#[linkage = "common"]` statics with a non-zero
//!   initializer (the LLVM verifier's "'common' global must
//!   have a zero initializer", a fatal error when code
//!   generation emits the module): a common global is a
//!   tentative definition the linker merges, so its
//!   initializer must be the zero initializer. The check
//!   below (`common_linkage_finding`) reads the linkage
//!   from the compiler's own codegen attributes and the
//!   initializer from the compiler's own static evaluation,
//!   and reports the static when the initializer carries a
//!   relocation (a pointer constant is not a zero
//!   initializer) or a non-zero byte.
//!
//! - The target-cpu session validation of code generation
//!   (`rustc_codegen_ssa::codegen_crate`'s opening checks,
//!   mirrored in `create_metadata` for AVR): rustc fatally
//!   rejects a target whose spec sets `need_explicit_cpu`
//!   when the session sets no `-C target-cpu`, and a
//!   `-C target-cpu` the target spec lists in
//!   `unsupported_cpus` ("known but unsupported"). Both
//!   predicates read only session state -- the target spec
//!   and the codegen options -- but rustc performs them as
//!   code generation starts, which a metadata-only session
//!   never reaches; the check below replays them over the
//!   driver's own session state (see
//!   `target_cpu_finding`). This is the same shape as the
//!   LTO / prefer-dynamic replay above, and it closes the
//!   last two records of the original class-L group that
//!   are session-state predicates at all (the data-layout
//!   and LLVM-pass records of that group need LLVM's own
//!   tables, as recorded on Issue #509).
//!
//! - The packed-stack / backchain incompatibility of code
//!   generation's function attributes
//!   (`rustc_codegen_llvm::attributes::packed_stack_attr`,
//!   computed for every function code generation defines):
//!   on s390x with `-Zpacked-stack`, a function that has the
//!   `backchain` target feature enabled -- by the session
//!   (the target spec or `-C target-feature`) or by its own
//!   `#[target_feature]` -- without the `soft-float`
//!   feature is rejected, because LLVM cannot generate
//!   valid function entry points for that combination
//!   under the default ABI. The predicate reads only the
//!   session's target and internal target features and the
//!   function's own codegen attributes, all of which the
//!   driver's session already carries, so the check below
//!   (`packed_stack_finding`) replays it over the items
//!   code generation would define (the collector walk's
//!   instances plus the exported roots, the same item set
//!   as the duplicate-symbol verdict). This closes the
//!   `backchain_attr` revision of
//!   tests/ui/target-feature/packedstack-combinations
//!   (Issue #509): the session-carried revision of that
//!   record is already rejected inside the reference
//!   compilation's own minicore build, and the neighbouring
//!   revisions -- packed-stack alone, and backchain with a
//!   soft-float target -- stay accepted, as they are under
//!   rustc. (Verified against the pinned rustc 1.100.0:
//!   each rejection shape exits 0 under `--emit=metadata`
//!   and 1 under `--emit=link`, and each neighbouring
//!   acceptance exits 0 under both.)
//!
//! - Missing optimized MIR for a demanded instance of an
//!   auxiliary crate (the collector's `NoOptimizedMir`, from
//!   `should_codegen_locally` in rustc_monomorphize): when the
//!   collector must monomorphize an instance locally and its
//!   home crate did not encode the item's optimized MIR, rustc
//!   fails fatally. For crates rustc built itself the metadata
//!   answers `is_mir_available` truthfully and the collector
//!   hook raises that fatal on its own. The crates this driver
//!   builds as auxiliaries are the exception: they are
//!   compiled with `-Zalways-encode-mir` (the main session
//!   must find their bodies to convert them), so their
//!   metadata encodes MIR a native build would have left out
//!   -- the MIR of items outside the home crate's reachable
//!   set (the encoder's `should_encode_mir` rule), and every
//!   MIR at all when the auxiliary's own build was
//!   metadata-only. The walk replays the native rule for the
//!   auxiliary crates (see `missing_mir_finding`): a
//!   metadata-only auxiliary encodes nothing, and for the
//!   rest an item is reported only on conclusive evidence --
//!   the demand arrived from a local body (an item the home
//!   crate's reachability visitor reaches is demanded through
//!   the home crate's own bodies), the item is not a trait
//!   impl item (those are seeded into the reachable set with
//!   their trait), and a module on its ancestor chain is not
//!   public. Anything not conclusive is treated as encoded:
//!   the check only narrows, never widens.
//!
//! - The symbol-name / def-path dump verdict of code
//!   generation's opening test hook
//!   (`rustc_symbol_mangling::test::dump_symbol_names_and_def_paths`,
//!   called from `start_codegen` only when the output types
//!   should codegen): for every local item carrying
//!   `#[rustc_dump_symbol_name]` or `#[rustc_dump_def_path]`,
//!   rustc emits an error carrying the item's mangled symbol
//!   name or def path, so the compilation fails -- the dump
//!   is the test harness's output, delivered as a diagnostic.
//!   The hook never runs in a metadata-only session, so
//!   conversion accepted these programs. The predicate is
//!   pure session/HIR state -- the `rustc_attrs` feature is
//!   enabled and a local item carries one of the two
//!   attributes -- which the driver's session already
//!   carries, so the check below (`dump_symbol_finding`)
//!   replays it exactly: it fires for precisely the items
//!   the hook reports on. (Verified against the pinned
//!   rustc 1.100.0: a dump-carrying source exits 0 under
//!   `--emit=metadata` and 1 under `--emit=link`.)
//!
//! - Check mode: when the compilation being judged is a
//!   compiletest check compilation (`--emit=metadata` — a
//!   `check-pass` / `check-fail` test, or an incremental
//!   `cpass` revision), the collector and code generation
//!   never run in the reference compilation, so the verdicts
//!   above that replay them cannot decide it: the
//!   required-constant walk, the `check_mono_item` forces,
//!   the duplicate-symbol, common-linkage, panic-runtime,
//!   LTO / prefer-dynamic, target-cpu, dump and
//!   packed-stack verdicts, the
//!   unshimmed-intrinsic
//!   verdict, and the body checks (SIMD / atomic / layout).
//!   Each was verified against the pinned compiler to stay
//!   silent under `--emit=metadata` while firing in a full
//!   build (Issue #509). In check mode the driver computes
//!   the session-level ones and reports them as notes, and
//!   the verdict rests on the metadata-visible findings:
//!   the materialization-lint finding above and the
//!   generic-export verdict (which does fire under
//!   `--emit=metadata`). See `set_check_mode`.
//!
//! Anything not provable from the converted bodies is left alone:
//! the checks only report a failure when the invalid shape is
//! positively present, mirroring the corresponding rustc error.

use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::mir::{Body, Operand, TerminatorKind};
use rustc_public::ty::{AdtKind, RigidTy, Ty, TyKind};
use rustc_public::{CrateDef, CrateDefType};

use crate::convert::guarded_ty_kind;

/// Scan the Stable MIR bodies that were converted for this program
/// and return the first monomorphization failure rustc would have
/// reported, if any.
pub fn check_bodies(bodies: &[Body]) -> Option<String> {
    let mut layout_memo: std::collections::HashMap<(Ty, bool), Option<String>> =
        std::collections::HashMap::new();
    for body in bodies {
        if let Some(err) = check_body(body, &mut layout_memo) {
            return Some(err);
        }
    }
    None
}

/// The unshimmed-intrinsic verdict over the converted bodies:
/// a call to an intrinsic that must be overridden, which this
/// backend serves with no shim, is the internal error
/// rustc_codegen_ssa raises when the same call reaches its
/// code generation (see the module documentation). Like the
/// collector verdicts in `check_required_consts`, the driver
/// chains this check only in the compilation-verdict modes:
/// in the executing modes the interpreter, not a compilation
/// verdict, decides what a call does.
pub fn check_intrinsic_overrides(bodies: &[Body]) -> Option<String> {
    for body in bodies {
        for bb in &body.blocks {
            let TerminatorKind::Call { func, .. } = &bb.terminator.kind else {
                continue;
            };
            if let Some(err) = check_override_call(func) {
                return Some(err);
            }
        }
    }
    None
}

/// The verdict for one call: `Some` when the callee is an
/// intrinsic that must be overridden and no builtin lookup
/// serves it. The resolution discipline is `check_call`'s:
/// the callee resolves to an intrinsic instance, and only
/// the positively-present shape is reported.
fn check_override_call(func: &Operand) -> Option<String> {
    let Operand::Constant(c) = func else {
        return None;
    };
    let Some(kind) = guarded_ty_kind(&c.const_.ty()) else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::FnDef(def, args)) = &kind else {
        return None;
    };
    let (def, args) = (*def, args.clone());
    let intrinsic = def.as_intrinsic()?;
    if !intrinsic.must_be_overridden() {
        return None;
    }
    let canonical = sparagmos_interp::builtins::canonical_name(&def.name());
    let bare = canonical.rsplit("::").next().unwrap_or(&canonical);
    // A compile-time intrinsic's calls are confined to
    // constant contexts and evaluated there; the call
    // never reaches code generation, so the override
    // error is unreachable for it.
    if comptime_intrinsic(bare) {
        return None;
    }
    let inst = Instance::resolve(def, &args).ok()?;
    if !matches!(inst.kind, InstanceKind::Intrinsic) {
        return None;
    }
    // Replay the dispatch conversion performs for a body-less
    // callee (see `mir_convert`'s external-callee chain): the
    // qualified registry lookup, the foreign-helper lookup,
    // then the intrinsic lookup. A hit in that dispatch is a
    // shim for the call — and every name it serves is one
    // code generation shims as well — but code generation's
    // own dispatch serves further names this interpreter
    // does not model, so the codegen-only list is consulted
    // too. No hit in either is code generation's fallback,
    // which for a must-be-overridden intrinsic is the
    // internal error. (The remaining conversion dispatches
    // key on trait methods and format machinery, which an
    // intrinsic definition never names.)
    let served = sparagmos_interp::builtins::lookup_qualified(&canonical).is_some()
        || sparagmos_interp::builtins::foreign_helper_lookup(&canonical).is_some()
        || sparagmos_interp::builtins::intrinsic_lookup(&canonical).is_some()
        || codegen_served_unmodelled(bare);
    if served {
        return None;
    }
    Some(format!(
        "intrinsic `{}` must be overridden by codegen backend, \
         but isn't (no backend shim serves this call)",
        intrinsic.fn_name()
    ))
}

/// Whether the named intrinsic is a compile-time
/// intrinsic: in the pinned core it is declared with
/// `#[rustc_comptime]`, and the compiler rejects any call
/// to it outside a constant context ("comptime fns can
/// only be called at compile time" — verified against
/// the pinned compiler with a runtime-context call to
/// `offset_of`). Every call that survives into a
/// converted body therefore sits inside a constant —
/// an inline const block, a promoted constant, a const
/// item — and is resolved by constant evaluation; it
/// never reaches code generation, so the
/// must-be-overridden error cannot fire for it. The
/// verdict exempts these names outright instead of
/// routing them through the served sets: code
/// generation does not shim them (they are not
/// `codegen_served_unmodelled` material), it simply
/// never sees them.
///
/// The set is the pinned core's `#[rustc_comptime]`
/// declarations, extracted from the Rust 1.100.0 core
/// sources; every one of them is a body-less
/// `#[rustc_intrinsic]` declaration, so each is a
/// must-be-overridden candidate the exemption has to
/// cover. As with the codegen list, the judgement is
/// by bare name: the test suite's own minicore-style
/// declarations name the same intrinsics.
fn comptime_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "align_of"
            | "field_offset"
            | "field_representing_type_actual_type_id"
            | "field_representing_type_name"
            | "field_representing_type_offset"
            | "needs_drop"
            | "non_exhaustive"
            | "offset_of"
            | "size_of"
            | "size_of_type_id"
            | "type_id"
            | "type_id_array_len"
            | "type_id_element_ty"
            | "type_id_fields"
            | "type_id_field_representing_type"
            | "type_id_function_ptr"
            | "type_id_generics"
            | "type_id_is_signed"
            | "type_id_points_mutably"
            | "type_id_points_to"
            | "type_id_variants"
            | "type_id_vtable"
            | "type_name"
            | "type_of"
            | "variant_count"
            | "variant_name"
            | "variant_non_exhaustive"
    )
}

/// Whether rustc's code generation shims the named intrinsic
/// even though the interpreter has no model for it. The
/// verdict in `check_intrinsic_overrides` mirrors code
/// generation's error, so its served set is code generation's
/// dispatch, not the interpreter's registry: a name belongs
/// here only with positive evidence that the pinned
/// compiler (Rust 1.100.0) lowers it — each name below was
/// verified by compiling a body-less declaration and call
/// with the real compiler and observing it accept the
/// program the interpreter leaves unmodelled. Names code
/// generation does not shim (`const_deallocate` and its kin)
/// must never be added: they are the verdict's reason to
/// exist.
fn codegen_served_unmodelled(name: &str) -> bool {
    if matches!(
        name,
        // The prefetch family: lowered to the target's
        // prefetch hint. The interpreter has no cache model,
        // so the call stays an ordinary conversion skip.
        "prefetch_read_data"
            | "prefetch_write_data"
            | "prefetch_read_instruction"
            | "prefetch_write_instruction"
            // The algebraic float family: the interpreter
            // models the strict and `*_fast` spellings, code
            // generation lowers the algebraic ones as well.
            | "fadd_algebraic"
            | "fsub_algebraic"
            | "fmul_algebraic"
            | "fdiv_algebraic"
            | "frem_algebraic"
            // Singletons code generation lowers directly:
            // the single-thread fence (its multi-thread
            // sibling is modelled), the debug trap, the
            // hint intrinsics, the black-box hint (an
            // inline-asm barrier around the value), the
            // raw memory equality (a load-and-compare of
            // the pointee), the vtable queries, and the
            // offload device count (lowered to the OpenMP
            // runtime call).
            | "atomic_singlethreadfence"
            | "black_box"
            | "breakpoint"
            | "cold_path"
            | "exact_div"
            | "is_val_statically_known"
            | "offload_get_num_devices"
            | "ptr_mask"
            | "raw_eq"
            | "return_address"
            | "select_unpredictable"
            | "vtable_align"
            | "vtable_size"
    ) {
        return true;
    }
    // The sized float families: for each operation stem
    // below, code generation lowers the f16 / f32 / f64 /
    // f128 spellings (the interpreter models only the
    // generic float operations). The spellings join the
    // width directly (`sqrtf32`, `fmaf64`) or after an
    // underscore (`round_ties_even_f32`,
    // `maximum_number_nsz_f64`); every stem–width
    // combination was verified against the pinned compiler.
    let stem = name
        .strip_suffix("f128")
        .or_else(|| name.strip_suffix("f16"))
        .or_else(|| name.strip_suffix("f32"))
        .or_else(|| name.strip_suffix("f64"))
        .map(|s| s.strip_suffix('_').unwrap_or(s));
    match stem {
        Some(stem) => matches!(
            stem,
            "sqrt"
                | "ceil"
                | "floor"
                | "trunc"
                | "round"
                | "round_ties_even"
                | "fma"
                | "fmuladd"
                | "pow"
                | "powi"
                | "copysign"
                | "maximum"
                | "minimum"
                | "maximum_number_nsz"
                | "minimum_number_nsz"
        ),
        None => false,
    }
}

fn check_body(
    body: &Body,
    layout_memo: &mut std::collections::HashMap<(Ty, bool), Option<String>>,
) -> Option<String> {
    for local in body.locals() {
        let ty = local.ty;
        if is_simd(ty) && simd_layout_invalid(ty) {
            return Some(format!(
                "the SIMD type `{ty}` has an invalid number of lanes \
                 (zero, or more than the monomorphization lane limit)"
            ));
        }
        if let Some(err) = layout_too_big(ty, layout_memo, false) {
            return Some(err);
        }
    }
    for bb in &body.blocks {
        let TerminatorKind::Call { func, args, .. } = &bb.terminator.kind else {
            continue;
        };
        if let Some(err) = check_call(func, args) {
            return Some(err);
        }
        if let Some(err) = check_measured_call(func, layout_memo) {
            return Some(err);
        }
    }
    None
}

/// The failure rustc reports when the layout of `ty` does not
/// exist because the type is too big for the target architecture,
/// or `None` when the layout exists or fails for any other
/// reason. The layout query can panic inside the compiler for
/// shapes it cannot represent; a panic means the layout could
/// not be determined here, not that it is invalid, so it is
/// treated as "no finding" (the same discipline as the SIMD
/// layout check). Results are memoized per scan: the walked
/// bodies repeat the same types heavily.
fn layout_too_big(
    ty: Ty,
    memo: &mut std::collections::HashMap<(Ty, bool), Option<String>>,
    bypass_screen: bool,
) -> Option<String> {
    if let Some(hit) = memo.get(&(ty, bypass_screen)) {
        return hit.clone();
    }
    let found = raw_layout_too_big(ty, bypass_screen);
    memo.insert((ty, bypass_screen), found.clone());
    found
}

fn raw_layout_too_big(ty: Ty, bypass_screen: bool) -> Option<String> {
    if !bypass_screen && !layout_query_safe(ty) {
        return None;
    }
    let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
    let Err(err) = (match layout {
        Ok(r) => r,
        Err(_) => return None,
    }) else {
        return None;
    };
    // Stable MIR exposes the layout failure only as the
    // compiler's diagnostic text; the failures this scan reports
    // are the ones whose text carries rustc's own wording for
    // the error — a type too big for the target architecture, or
    // a layout cycle (E0391), which monomorphization surfaces
    // for the same materialized types.
    let text = err.to_string();
    for needle in ["values of the type", "cycle detected when computing layout"] {
        if let Some(at) = text.find(needle) {
            return Some(text[at..].to_string());
        }
    }
    None
}

/// Whether asking the compiler for `ty`'s layout is safe for
/// this scan. The layout query is not total: for a type with an
/// unsized field in a non-final position (accepted under
/// trivial bounds but never materializable) or a projection
/// alias in field position (whose normalization can be
/// ambiguous), the query does not return an error at all — it
/// creates a delayed internal compiler error that poisons the
/// rest of the compilation. The scan therefore walks the type
/// structurally first, asking the compiler nothing: a type is
/// safe when every component in field position is a concrete
/// sized shape, with unsized components only in tail position
/// (the last field of a struct or single-variant enum, the
/// last element of a tuple, or nested under either — a
/// multi-variant ADT and a union have no tail position; see
/// the ADT arm of the walk). Pointer and reference types are
/// walked through as well: their layout needs the pointee's
/// metadata, which is computed from the pointee's tail. Type
/// parameters are resolved one frame at a time through the
/// enclosing ADT's arguments —
/// never through the compiler's instantiating-and-normalizing
/// accessors — and anything unresolvable is unsafe. Recursive
/// types are cut off by a visited set: a type already being
/// walked is assumed safe, as its own walk is in progress in
/// a context at least as strict (non-tail occurrences are
/// always walked before the tail occurrence).
fn layout_query_safe(ty: Ty) -> bool {
    let mut visited = std::collections::HashSet::new();
    walk_layout_type(ty, &[], true, &mut visited, 0)
}

/// Walk `ty` structurally. `frames` is the stack of generic
/// argument lists of the ADTs whose fields are being walked,
/// innermost last: a type parameter in `ty` refers to the
/// innermost frame, and the type it resolves to was written
/// one frame further out, so it is walked with the innermost
/// frame popped. `tail` marks tail position (see above).
fn walk_layout_type(
    ty: Ty,
    frames: &[rustc_public::ty::GenericArgs],
    tail: bool,
    visited: &mut std::collections::HashSet<Ty>,
    depth: usize,
) -> bool {
    if depth > 256 {
        return false;
    }
    // A type passing through a coroutine-closure type cannot be
    // asked its kind (see `guarded_ty_kind`); its layout safety
    // cannot be judged structurally, so it is not safe to query.
    let Some(kind) = guarded_ty_kind(&ty) else {
        return false;
    };
    match kind {
        TyKind::RigidTy(RigidTy::Adt(def, args)) => {
            if !visited.insert(ty) {
                return true;
            }
            // Tail position exists only in a struct-like type.
            // The layout calculator lays out a struct (and a
            // single-variant enum, and a tuple) with at most one
            // unsized field, in last position; but every variant
            // of a multi-variant ADT is laid out prefixed by the
            // discriminant, where an unsized field is always an
            // error, and a union's fields all overlap, so no
            // field is in tail position. Under the monomorphic
            // typing environment this query runs in, that error
            // is not returned — it is recorded as a delayed bug
            // that aborts the whole compilation when flushed
            // (the layout of `Poll<str>` in
            // tests/ui/mir/gvn-nonsensical-sized-str.rs). A
            // possibly unsized field of a multi-variant ADT or
            // a union therefore makes the query unsafe wherever
            // it sits.
            let has_tail = def.variants().len() <= 1 && def.kind() != AdtKind::Union;
            let mut frames = frames.to_vec();
            frames.push(args);
            def.variants().iter().all(|variant| {
                let fields = variant.fields();
                let last = fields.len().saturating_sub(1);
                fields.iter().enumerate().all(|(i, field)| {
                    walk_layout_type(
                        field.ty(),
                        &frames,
                        tail && i == last && has_tail,
                        visited,
                        depth + 1,
                    )
                })
            })
        }
        TyKind::RigidTy(RigidTy::Tuple(elems)) => {
            let last = elems.len().saturating_sub(1);
            elems.iter().enumerate().all(|(i, elem)| {
                walk_layout_type(*elem, frames, tail && i == last, visited, depth + 1)
            })
        }
        TyKind::RigidTy(RigidTy::Array(elem, _)) => {
            walk_layout_type(elem, frames, false, visited, depth + 1)
        }
        TyKind::RigidTy(RigidTy::Pat(inner, _)) => {
            walk_layout_type(inner, frames, tail, visited, depth + 1)
        }
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _))
        | TyKind::RigidTy(RigidTy::RawPtr(pointee, _)) => {
            walk_layout_type(pointee, frames, true, visited, depth + 1)
        }
        TyKind::RigidTy(RigidTy::Slice(_))
        | TyKind::RigidTy(RigidTy::Str)
        | TyKind::RigidTy(RigidTy::Dynamic(..)) => tail,
        TyKind::Param(param) => {
            let Some((innermost, outer)) = frames.split_last() else {
                return false;
            };
            match innermost.0.get(param.index as usize) {
                Some(rustc_public::ty::GenericArgKind::Type(arg)) => {
                    walk_layout_type(*arg, outer, tail, visited, depth + 1)
                }
                _ => false,
            }
        }
        TyKind::Alias(..) | TyKind::Bound(..) => false,
        TyKind::RigidTy(_) => true,
    }
}

/// Check the type arguments of a call whose lowering is defined
/// by those types' layouts — an intrinsic call, or one of the
/// `core::mem` size/alignment measuring functions — for a type
/// too big for the target architecture. Instantiating such a
/// call evaluates the measured type's layout in rustc, so a
/// too-big argument is the same monomorphization failure as a
/// too-big local. The layout query is asked without the
/// structural pre-screen here: rustc itself performs exactly
/// this query when it instantiates the call, so the query
/// cannot fail in a way rustc's own compilation of the same
/// program avoids. Type arguments of ordinary calls are *not*
/// checked: an ordinary generic function may never materialize
/// a type it merely names (a `PhantomData` parameter is the
/// standard shape), and rustc instantiates it without asking
/// for the type's layout.
fn check_measured_call(
    func: &Operand,
    layout_memo: &mut std::collections::HashMap<(Ty, bool), Option<String>>,
) -> Option<String> {
    let Operand::Constant(c) = func else {
        return None;
    };
    let Some(kind) = guarded_ty_kind(&c.const_.ty()) else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::FnDef(def, args)) = &kind else {
        return None;
    };
    let (def, args) = (*def, args.clone());
    let intrinsic = Instance::resolve(def, &args)
        .map(|inst| matches!(inst.kind, InstanceKind::Intrinsic))
        .unwrap_or(false);
    if !intrinsic && !is_mem_measure_fn(&def.name()) {
        return None;
    }
    for arg in &args.0 {
        if let rustc_public::ty::GenericArgKind::Type(ty) = arg {
            if let Some(err) = layout_too_big(*ty, layout_memo, true) {
                return Some(err);
            }
        }
    }
    None
}

/// Whether `name` is one of the `core::mem` functions whose
/// result *is* the measured type's layout (`size_of` and
/// `align_of` evaluate the layout as a constant; `size_of_val`
/// and the `min_align_of` family forward to the corresponding
/// intrinsics at instantiation).
fn is_mem_measure_fn(name: &str) -> bool {
    let mut segments = name.rsplit("::");
    let (Some(last), Some(parent)) = (segments.next(), segments.next()) else {
        return false;
    };
    parent == "mem"
        && matches!(
            last,
            "size_of"
                | "align_of"
                | "size_of_val"
                | "align_of_val"
                | "min_align_of"
                | "min_align_of_val"
        )
}

/// Whether the layout of a SIMD type exists. The layout query can
/// panic inside the compiler for shapes it cannot represent; a
/// panic means the layout could not be determined here, not that
/// it is invalid, so it is treated as "no finding".
/// Whether the layout query positively reports `ty`'s lane
/// count as invalid -- the failure rustc's own wording names
/// ("has zero elements", "has more elements than the limit").
/// Any other layout outcome is no finding: a layout can be
/// uncomputable here for reasons rustc accepts (a lane count
/// named by a constant the layout query cannot evaluate in
/// this context reports "unknown layout", and rustc's own
/// code generation evaluates it fine), and a panic means the
/// layout could not be determined, not that it is invalid --
/// the same discipline as the too-big check.
fn simd_layout_invalid(ty: Ty) -> bool {
    let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
    let Err(err) = (match layout {
        Ok(r) => r,
        Err(_) => return false,
    }) else {
        return false;
    };
    let text = err.to_string();
    text.contains("has zero elements") || text.contains("has more elements than the limit")
}

/// Whether a type is a SIMD vector type (a `#[repr(simd)]` ADT or
/// `core::simd::Simd`), the notion rustc's own check uses.
fn is_simd(ty: Ty) -> bool {
    // A type passing through a coroutine-closure type cannot be
    // asked its kind (see `guarded_ty_kind`) and is not a SIMD
    // type this check can judge.
    guarded_ty_kind(&ty).is_some_and(|kind| kind.is_simd())
}

fn check_call(func: &Operand, call_args: &[Operand]) -> Option<String> {
    let Operand::Constant(c) = func else {
        return None;
    };
    let Some(kind) = guarded_ty_kind(&c.const_.ty()) else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::FnDef(def, args)) = &kind else {
        return None;
    };
    let (def, args) = (*def, args.clone());
    // Only compiler intrinsics are checked: among them the
    // `simd_*` and `atomic_*` families carry monomorphization
    // rules of this shape.
    let inst = Instance::resolve(def, &args).ok()?;
    if !matches!(inst.kind, InstanceKind::Intrinsic) {
        return None;
    }
    let full_name = def.name();
    let name = full_name.rsplit("::").next().unwrap_or(&full_name);
    let sig = kind.fn_sig()?;
    let sig = &sig.value;
    let inputs = sig.inputs();
    if name.starts_with("atomic_") {
        return check_atomic(name, inputs);
    }
    if !name.starts_with("simd_") {
        return None;
    }
    // Mirror `generic_simd_intrinsic`'s entry checks: the vector
    // argument whose type must be a SIMD type differs for the two
    // intrinsics handled before the general case.
    let (vector_ty, what) = match name {
        "simd_splat" => (Some(sig.output()), "return"),
        "simd_select_bitmask" => (inputs.get(1).copied(), "second argument"),
        _ => (inputs.first().copied(), "input"),
    };
    let vector_ty = vector_ty?;
    if !is_simd(vector_ty) {
        return Some(format!(
            "invalid monomorphization of `{name}` intrinsic: \
             expected SIMD {what} type, found non-SIMD `{vector_ty}`"
        ));
    }
    check_simd_family(name, inputs, sig.output(), call_args)
}

/// Mirror the E0511 checks rustc_codegen_ssa performs for the
/// `atomic_*` intrinsics (`codegen_intrinsic_call`): the type of
/// the value at the memory operand — the pointee of the first
/// argument, which is `*const T` for loads and `*mut T` for the
/// rest — must be a basic integer type (an integer of any width,
/// pointer-sized included; `bool` and `char` do not qualify) or
/// a raw pointer type. The read-modify-write arithmetic group
/// (`atomic_xadd` and its siblings, whose second argument is the
/// operand value) additionally requires the operand's type to
/// equal the memory type, or to be `usize` when the memory type
/// is a raw pointer. `atomic_fence` and
/// `atomic_singlethreadfence` carry no type rule and are left
/// alone, as is any shape not positively present.
fn check_atomic(name: &str, inputs: &[Ty]) -> Option<String> {
    let dst = inputs.first().copied()?;
    let Some(dst_kind) = guarded_ty_kind(&dst) else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::RawPtr(pointee, _)) = dst_kind else {
        return None;
    };
    let invalid = || {
        Some(format!(
            "invalid monomorphization of `{name}` intrinsic: \
             expected basic integer or pointer type, found `{pointee}`"
        ))
    };
    match name {
        "atomic_load" | "atomic_store" | "atomic_xchg" | "atomic_cxchg"
        | "atomic_cxchgweak" | "atomic_max" | "atomic_min" | "atomic_umax"
        | "atomic_umin" => {
            if is_int_or_uint(pointee)
                || guarded_ty_kind(&pointee)
                    .is_some_and(|k| matches!(k, TyKind::RigidTy(RigidTy::RawPtr(..))))
            {
                None
            } else {
                invalid()
            }
        }
        "atomic_xadd" | "atomic_xsub" | "atomic_and" | "atomic_or" | "atomic_xor"
        | "atomic_nand" => {
            let val = inputs.get(1).copied()?;
            let ok = (is_int_or_uint(pointee) && val == pointee)
                || (guarded_ty_kind(&pointee)
                    .is_some_and(|k| matches!(k, TyKind::RigidTy(RigidTy::RawPtr(..))))
                    && guarded_ty_kind(&val).is_some_and(|k| {
                        matches!(
                            k,
                            TyKind::RigidTy(RigidTy::Uint(rustc_public::ty::UintTy::Usize))
                        )
                    }));
            if ok { None } else { invalid() }
        }
        _ => None,
    }
}

/// Whether a type is an integer type (signed or unsigned, any
/// width, pointer-sized included).
fn is_int_or_uint(ty: Ty) -> bool {
    guarded_ty_kind(&ty).is_some_and(|k| {
        matches!(k, TyKind::RigidTy(RigidTy::Int(_) | RigidTy::Uint(_)))
    })
}

/// The shape of a SIMD vector type as `(lane count, element
/// type)`, read from the single array field every `#[repr(simd)]`
/// type is built on, or from several fields all of the same type;
/// `None` when the type is not a SIMD type or its lane count is
/// not a concrete value here.
fn simd_shape(ty: Ty) -> Option<(u64, Ty)> {
    if !is_simd(ty) {
        return None;
    }
    let Some(ty_kind) = guarded_ty_kind(&ty) else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty_kind else {
        return None;
    };
    let variants = def.variants();
    let [variant] = variants.as_slice() else {
        return None;
    };
    let fields = variant.fields();
    let [field] = fields.as_slice() else {
        // A `#[repr(simd)]` type may instead be several fields
        // all of the same type `T`; its shape is then the field
        // count and `T`, read with the instance's arguments
        // substituted.
        if fields.len() < 2 {
            return None;
        }
        let elem = fields[0].ty_with_args(&args);
        if fields.iter().all(|f| f.ty_with_args(&args) == elem) {
            return Some((fields.len() as u64, elem));
        }
        return None;
    };
    // The field's declared type is written in terms of the
    // ADT's own generic parameters (`Simd<T, N>`'s field is
    // `[T; N]`); substitute the instance's arguments, the same
    // substitution code generation works with.
    let Some(field_kind) = guarded_ty_kind(&field.ty()) else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::Array(elem, len)) = field_kind else {
        return None;
    };
    let elem = subst_param(elem, &args);
    // The lane count, resolving a count written as the ADT's
    // own const parameter through the instance's arguments.
    let lanes = match len.eval_target_usize() {
        Ok(n) => n,
        Err(_) => {
            let rustc_public::ty::TyConstKind::Param(p) = len.kind() else {
                return None;
            };
            match args.0.get(p.index as usize) {
                Some(rustc_public::ty::GenericArgKind::Const(inner)) => {
                    inner.eval_target_usize().ok()?
                }
                _ => return None,
            }
        }
    };
    Some((lanes, elem))
}

/// Resolve a type written as one of the enclosing ADT's type
/// parameters through the instance's generic arguments; any
/// other type is returned unchanged.
fn subst_param(ty: Ty, args: &rustc_public::ty::GenericArgs) -> Ty {
    match guarded_ty_kind(&ty) {
        Some(TyKind::Param(p)) => match args.0.get(p.index as usize) {
            Some(rustc_public::ty::GenericArgKind::Type(t)) => *t,
            _ => ty,
        },
        _ => ty,
    }
}

/// The bit width of an integer type, from its layout (the layout
/// of a primitive integer always exists); `None` when it cannot
/// be determined here.
fn int_bit_width(ty: Ty) -> Option<u64> {
    if !is_int_or_uint(ty) {
        return None;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout())) {
        Ok(Ok(layout)) => Some(layout.shape().size.bytes() as u64 * 8),
        _ => None,
    }
}

/// The `u32` elements of a constant index operand, read from the
/// operand's allocation — the same values code generation reads
/// when it checks `simd_shuffle` indices. `None` when the
/// operand is not a constant whose allocation is exactly the
/// index vector's bytes.
fn const_u32_vector(op: &Operand, lanes: u64) -> Option<Vec<u32>> {
    use rustc_public::ty::ConstantKind as CK;
    let Operand::Constant(c) = op else {
        return None;
    };
    let read = |alloc: &rustc_public::ty::Allocation| -> Option<Vec<u32>> {
        if alloc.bytes.len() as u64 != 4 * lanes {
            return None;
        }
        (0..lanes)
            .map(|i| {
                let at = (4 * i) as usize;
                alloc.read_partial_uint(at..at + 4).ok().map(|v| v as u32)
            })
            .collect()
    };
    match c.const_.kind() {
        CK::Allocated(alloc) => read(&alloc),
        CK::Ty(tc) => match tc.kind() {
            rustc_public::ty::TyConstKind::Value(_, alloc) => read(&alloc),
            _ => None,
        },
        _ => None,
    }
}

/// The value of a constant scalar index operand (the `u32`
/// index of `simd_insert` / `simd_extract`), read from the
/// operand's allocation. `None` when the operand is not a
/// constant with a readable 4-byte value.
fn const_u32_scalar(op: &Operand) -> Option<u32> {
    use rustc_public::ty::ConstantKind as CK;
    let Operand::Constant(c) = op else {
        return None;
    };
    let read = |alloc: &rustc_public::ty::Allocation| -> Option<u32> {
        alloc.read_partial_uint(0..4).ok().map(|v| v as u32)
    };
    match c.const_.kind() {
        CK::Allocated(alloc) => read(&alloc),
        CK::Ty(tc) => match tc.kind() {
            rustc_public::ty::TyConstKind::Value(_, alloc) => read(&alloc),
            _ => None,
        },
        _ => None,
    }
}

/// The per-intrinsic argument rules of `generic_simd_intrinsic`
/// (rustc_codegen_llvm) beyond the entry check in `check_call`:
/// element-type requirements, length and element agreement
/// between arguments and the return type, and the constant
/// index bounds of `simd_shuffle`. Each rule fires only when
/// the invalid shape is positively present in the call's
/// substituted signature (or, for the shuffle bounds, in the
/// call's constant index operand).
fn check_simd_family(
    name: &str,
    inputs: &[Ty],
    ret_ty: Ty,
    call_args: &[Operand],
) -> Option<String> {
    let input = |i: usize| inputs.get(i).copied();
    let err = |msg: String| {
        Some(format!(
            "invalid monomorphization of `{name}` intrinsic: {msg}"
        ))
    };
    let mask_elem_err = |elem: Ty| {
        err(format!(
            "expected mask element type to be an integer, found `{elem}`"
        ))
    };
    match name {
        "simd_saturating_add" | "simd_saturating_sub" => {
            let in_ty = input(0)?;
            let (_, elem) = simd_shape(in_ty)?;
            if is_int_or_uint(elem) {
                None
            } else {
                err(format!(
                    "expected element type `{elem}` of vector type `{in_ty}` \
                     to be a signed or unsigned integer type"
                ))
            }
        }
        "simd_bitmask" => {
            let in_ty = input(0)?;
            let (len, elem) = simd_shape(in_ty)?;
            if !is_int_or_uint(elem) {
                return mask_elem_err(elem);
            }
            // The return must be an unsigned integer with
            // `max(len, 8).next_power_of_two()` bits, or a byte
            // array with `ceil(len / 8)` elements.
            let expected_bits = len.max(8).next_power_of_two();
            let ok = match guarded_ty_kind(&ret_ty) {
                Some(TyKind::RigidTy(RigidTy::Uint(_))) => {
                    int_bit_width(ret_ty) == Some(expected_bits)
                }
                Some(TyKind::RigidTy(RigidTy::Array(e, n))) => {
                    guarded_ty_kind(&e).is_some_and(|k| {
                        matches!(
                            k,
                            TyKind::RigidTy(RigidTy::Uint(rustc_public::ty::UintTy::U8))
                        )
                    }) && n.eval_target_usize().ok() == Some(len.div_ceil(8))
                }
                _ => false,
            };
            if ok {
                None
            } else {
                err(format!(
                    "cannot return `{ret_ty}`, expected `u{expected_bits}` \
                     or `[u8; {}]`",
                    len.div_ceil(8)
                ))
            }
        }
        "simd_shuffle" => {
            let in_ty = input(0)?;
            let (in_len, in_elem) = simd_shape(in_ty)?;
            let idx_ty = input(2)?;
            let Some((n, idx_elem)) = simd_shape(idx_ty) else {
                return err(
                    "simd_shuffle index must be a SIMD vector of `u32` elements"
                        .to_string(),
                );
            };
            if !guarded_ty_kind(&idx_elem).is_some_and(|k| {
                matches!(
                    k,
                    TyKind::RigidTy(RigidTy::Uint(rustc_public::ty::UintTy::U32))
                )
            }) {
                return err(
                    "simd_shuffle index must be a SIMD vector of `u32` elements"
                        .to_string(),
                );
            }
            if !is_simd(ret_ty) {
                return err(format!(
                    "expected SIMD return type, found non-SIMD `{ret_ty}`"
                ));
            }
            let (out_len, out_elem) = simd_shape(ret_ty)?;
            if out_len != n {
                return err(format!(
                    "expected return type with length {n} (same as the \
                     index type `{idx_ty}`), found `{ret_ty}` with \
                     length {out_len}"
                ));
            }
            if out_elem != in_elem {
                return err(format!(
                    "expected return element type `{in_elem}` (element of \
                     input `{in_ty}`), found `{out_elem}`"
                ));
            }
            if let Some(indices) = const_u32_vector(call_args.get(2)?, n) {
                let total = in_len * 2;
                for (i, v) in indices.iter().enumerate() {
                    if u64::from(*v) >= total {
                        return err(format!(
                            "expected index {i} to be less than {total}, \
                             found {v}"
                        ));
                    }
                }
            }
            None
        }
        "simd_gather" | "simd_scatter" => {
            let in_ty = input(0)?;
            let (in_len, in_elem) = simd_shape(in_ty)?;
            let ptr_ty = input(1)?;
            if !is_simd(ptr_ty) {
                return err(format!(
                    "expected SIMD second argument type, found \
                     non-SIMD `{ptr_ty}`"
                ));
            }
            let (ptr_len, ptr_elem) = simd_shape(ptr_ty)?;
            if ptr_len != in_len {
                return err(format!(
                    "expected second argument with length {in_len} (same \
                     as input type `{in_ty}`), found `{ptr_ty}` with \
                     length {ptr_len}"
                ));
            }
            let mask_ty = input(2)?;
            if !is_simd(mask_ty) {
                return err(format!(
                    "expected SIMD third argument type, found \
                     non-SIMD `{mask_ty}`"
                ));
            }
            let (mask_len, mask_elem) = simd_shape(mask_ty)?;
            if mask_len != in_len {
                return err(format!(
                    "expected third argument with length {in_len} (same \
                     as input type `{in_ty}`), found `{mask_ty}` with \
                     length {mask_len}"
                ));
            }
            if name == "simd_gather" {
                if !is_simd(ret_ty) {
                    return err(format!(
                        "expected SIMD return type, found non-SIMD `{ret_ty}`"
                    ));
                }
                if ret_ty != in_ty {
                    return err(format!(
                        "expected return type `{in_ty}`, found `{ret_ty}`"
                    ));
                }
            }
            // The pointer vector's elements must be pointers to
            // the input's element type. Code generation also
            // requires mutable pointers for scatter, but that
            // requirement is enforced only when a call is
            // lowered, and this scan also covers constant bodies
            // code generation never lowers (a masked store
            // through a `*const` in an unevaluated constant is
            // accepted by rustc), so mutability is not checked
            // here.
            let ptr_ok = match guarded_ty_kind(&ptr_elem) {
                Some(TyKind::RigidTy(RigidTy::RawPtr(p, _))) => p == in_elem,
                _ => false,
            };
            if !ptr_ok {
                return err(format!(
                    "expected element type of second argument `{ptr_ty}` \
                     to be a pointer to the element type `{in_elem}`"
                ));
            }
            if !is_int_or_uint(mask_elem) {
                return mask_elem_err(mask_elem);
            }
            None
        }
        "simd_masked_load" | "simd_masked_store" => {
            let mask_ty = input(0)?;
            let (mask_len, mask_elem) = simd_shape(mask_ty)?;
            let values_ty = input(2)?;
            if !is_simd(values_ty) {
                return err(format!(
                    "expected SIMD third argument type, found \
                     non-SIMD `{values_ty}`"
                ));
            }
            let (values_len, values_elem) = simd_shape(values_ty)?;
            if values_len != mask_len {
                return err(format!(
                    "expected third argument with length {mask_len} \
                     (same as input type `{mask_ty}`), found \
                     `{values_ty}` with length {values_len}"
                ));
            }
            if name == "simd_masked_load" && ret_ty != values_ty {
                return err(format!(
                    "expected return type `{values_ty}`, found `{ret_ty}`"
                ));
            }
            // The pointer must point at the values' element
            // type. (Pointer mutability is not checked, for the
            // reason given at the gather/scatter check above.)
            let pointer_ty = input(1)?;
            let ptr_ok = match guarded_ty_kind(&pointer_ty) {
                Some(TyKind::RigidTy(RigidTy::RawPtr(p, _))) => p == values_elem,
                _ => false,
            };
            if !ptr_ok {
                return err(format!(
                    "expected element type of second argument \
                     `{pointer_ty}` to be a pointer to the element \
                     type `{values_elem}`"
                ));
            }
            if !is_int_or_uint(mask_elem) {
                return mask_elem_err(mask_elem);
            }
            None
        }
        "simd_reduce_add_ordered" | "simd_reduce_mul_ordered"
        | "simd_reduce_add_unordered" | "simd_reduce_mul_unordered" => {
            let in_ty = input(0)?;
            let (_, elem) = simd_shape(in_ty)?;
            // The ordered pair takes the initial value as its
            // second argument, and it must have the element type.
            if matches!(
                name,
                "simd_reduce_add_ordered" | "simd_reduce_mul_ordered"
            ) {
                if let Some(initial) = input(1) {
                    if initial != elem {
                        return err(format!(
                            "expected initial value type `{elem}` \
                             (element of input `{in_ty}`), found \
                             `{initial}`"
                        ));
                    }
                }
            }
            if ret_ty != elem {
                return err(format!(
                    "expected return type `{elem}` (element of input \
                     `{in_ty}`), found `{ret_ty}`"
                ));
            }
            if is_int_or_uint(elem)
                || guarded_ty_kind(&elem)
                    .is_some_and(|k| matches!(k, TyKind::RigidTy(RigidTy::Float(_))))
            {
                None
            } else {
                err(format!(
                    "unsupported {name} from `{in_ty}` with element \
                     `{elem}` to `{ret_ty}`"
                ))
            }
        }
        "simd_reduce_min" | "simd_reduce_max" | "simd_reduce_and"
        | "simd_reduce_or" | "simd_reduce_xor" => {
            let in_ty = input(0)?;
            let (_, elem) = simd_shape(in_ty)?;
            if ret_ty != elem {
                return err(format!(
                    "expected return type `{elem}` (element of input \
                     `{in_ty}`), found `{ret_ty}`"
                ));
            }
            if is_int_or_uint(elem) {
                None
            } else {
                err(format!(
                    "unsupported {name} from `{in_ty}` with element \
                     `{elem}` to `{ret_ty}`"
                ))
            }
        }
        "simd_reduce_all" | "simd_reduce_any" => {
            let in_ty = input(0)?;
            let (_, elem) = simd_shape(in_ty)?;
            if is_int_or_uint(elem) {
                None
            } else {
                err(format!(
                    "unsupported {name} from `{in_ty}` with element \
                     `{elem}` to `{ret_ty}`"
                ))
            }
        }
        "simd_eq" | "simd_ne" | "simd_lt" | "simd_le" | "simd_gt" | "simd_ge" => {
            let in_ty = input(0)?;
            let (in_len, _) = simd_shape(in_ty)?;
            if !is_simd(ret_ty) {
                return err(format!(
                    "expected SIMD return type, found non-SIMD `{ret_ty}`"
                ));
            }
            let (out_len, out_elem) = simd_shape(ret_ty)?;
            if out_len != in_len {
                return err(format!(
                    "expected return type with length {in_len} (same as \
                     input type `{in_ty}`), found `{ret_ty}` with \
                     length {out_len}"
                ));
            }
            if !is_int_or_uint(out_elem) {
                return err(format!(
                    "expected return element type to be an integer, \
                     found `{out_elem}`"
                ));
            }
            None
        }
        "simd_select" => {
            let mask_ty = input(0)?;
            let (m_len, m_elem) = simd_shape(mask_ty)?;
            let v_ty = input(1)?;
            if !is_simd(v_ty) {
                return err(format!(
                    "expected SIMD second argument type, found \
                     non-SIMD `{v_ty}`"
                ));
            }
            let (v_len, _) = simd_shape(v_ty)?;
            if m_len != v_len {
                return err(format!(
                    "expected mask length {m_len} to match the vector \
                     length {v_len} of `{v_ty}`"
                ));
            }
            if !is_int_or_uint(m_elem) {
                return mask_elem_err(m_elem);
            }
            None
        }
        "simd_splat" => {
            let (_, out_elem) = simd_shape(ret_ty)?;
            let val_ty = input(0)?;
            if val_ty == out_elem {
                None
            } else {
                err(format!(
                    "expected element type `{out_elem}` of vector type \
                     `{ret_ty}`, found the splatted value's type \
                     `{val_ty}`"
                ))
            }
        }
        "simd_select_bitmask" => {
            let vec_ty = input(1)?;
            let (len, _) = simd_shape(vec_ty)?;
            let mask_ty = input(0)?;
            // Same accepted mask shapes as `simd_bitmask`'s
            // return: an unsigned integer with
            // `max(len, 8).next_power_of_two()` bits, or a byte
            // array with `ceil(len / 8)` elements.
            let expected_bits = len.max(8).next_power_of_two();
            let ok = match guarded_ty_kind(&mask_ty) {
                Some(TyKind::RigidTy(RigidTy::Int(_)))
                | Some(TyKind::RigidTy(RigidTy::Uint(_))) => {
                    int_bit_width(mask_ty) == Some(expected_bits)
                }
                Some(TyKind::RigidTy(RigidTy::Array(e, n))) => {
                    guarded_ty_kind(&e).is_some_and(|k| {
                        matches!(
                            k,
                            TyKind::RigidTy(RigidTy::Uint(rustc_public::ty::UintTy::U8))
                        )
                    }) && n.eval_target_usize().ok() == Some(len.div_ceil(8))
                }
                _ => false,
            };
            if ok {
                None
            } else {
                err(format!(
                    "invalid bitmask `{mask_ty}`, expected `u{expected_bits}` \
                     or `[u8; {}]`",
                    len.div_ceil(8)
                ))
            }
        }
        "simd_insert" | "simd_insert_dyn" => {
            let in_ty = input(0)?;
            let (lanes, elem) = simd_shape(in_ty)?;
            let val_ty = input(2)?;
            if val_ty != elem {
                return err(format!(
                    "expected inserted type `{elem}` (element of input \
                     `{in_ty}`), found `{val_ty}`"
                ));
            }
            if let Some(idx) = call_args.get(1).and_then(const_u32_scalar) {
                if u64::from(idx) >= lanes {
                    return err(format!(
                        "expected index to be less than {lanes}, \
                         found {idx}"
                    ));
                }
            }
            None
        }
        "simd_extract" | "simd_extract_dyn" => {
            let in_ty = input(0)?;
            let (lanes, elem) = simd_shape(in_ty)?;
            if ret_ty != elem {
                return err(format!(
                    "expected return type `{elem}` (element of input \
                     `{in_ty}`), found `{ret_ty}`"
                ));
            }
            if let Some(idx) = call_args.get(1).and_then(const_u32_scalar) {
                if u64::from(idx) >= lanes {
                    return err(format!(
                        "expected index to be less than {lanes}, \
                         found {idx}"
                    ));
                }
            }
            None
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------
// Required constants of collected instances
// ---------------------------------------------------------------------

/// Upper bound on the instances one required-constant walk visits.
/// Conversion itself collects at most a few thousand instance
/// bodies per program; the walk additionally follows mentions into
/// instances conversion never demanded (dead-code mentions), and
/// the cap keeps a pathological mention graph from turning the
/// check into a second full collection. Reaching the cap only
/// narrows the check, never widens it.
const MAX_REQUIRED_CONST_WALK: usize = 8192;

fn fully_monomorphized<'tcx>() -> rustc_middle::ty::TypingEnv<'tcx> {
    rustc_middle::ty::TypingEnv::fully_monomorphized()
}

/// Mirror of the constant evaluation rustc's monomorphization
/// collector performs (`MirUsedCollector` in rustc_monomorphize):
/// for every instance it collects, the collector evaluates the
/// body's required constants — every unevaluated constant operand
/// of the pre-optimization body, dead code included — and a
/// constant whose evaluation is already reported as an error
/// (E0080) fails the whole compilation. It reaches those bodies
/// not only through live calls: the items a body *mentions*
/// (calls, drops, unsizing casts and closure coercions recorded
/// before optimization, so dead code still mentions them) and the
/// function pointers inside evaluated constant values are all
/// collected and checked the same way.
///
/// The driver's own collection walks Stable MIR bodies, which are
/// post-optimization: a mention that lives only in dead code is
/// gone from them, and the failing constant it leads to is never
/// demanded, so conversion succeeds where rustc fails (the
/// `tests/ui/consts/required-consts` family). This check replays
/// the collector's walk over the compiler's internal bodies —
/// which keep both the required constants and the mentioned items
/// from before optimization — starting from the same roots
/// the collector itself starts from (see
/// `RequiredConstWalk::collect_roots`).
///
/// Only a positively reported evaluation error is a finding: a
/// constant that stays too generic after substitution, an instance
/// whose body the compiler cannot produce here, and any internal
/// panic while walking are all "could not determine", never a
/// failure — the same discipline as the layout check above.
/// (Evaluating a failing constant also emits rustc's own E0080
/// diagnostic into the session, which is how the failure surfaces
/// in practice — the same emission that fails the compilation
/// under rustc; the returned string is the driver's own record of
/// the finding.)
///
/// The same walk also carries the collector's remaining
/// verdicts, each described at its site: the per-instance
/// `check_mono_item` force in `RequiredConstWalk::process`, the
/// exported-symbol verdict over the crate's export roots
/// (`generic_export_finding`), the duplicate-symbol verdict
/// over the collected items (`duplicate_symbol_finding`), and
/// the panic-runtime validation of the crate graph
/// (`panic_runtime_finding`), which needs no walk, and the
/// LTO / prefer-dynamic session validation
/// (`lto_prefer_dynamic_finding`), which needs no walk
/// either, and the target-cpu session validation
/// (`target_cpu_finding`), which needs no walk either.
/// The dump verdict (`dump_symbol_finding`) needs no walk
/// either. The packed-stack verdict
/// (`packed_stack_finding`) runs over the walk's item set,
/// like the duplicate-symbol verdict.
pub fn check_required_consts() -> Option<String> {
    let tcx = crate::convert::compiler_tcx()?;
    // A lint error recorded at a body materialization is a
    // finding of its own: rustc lints every local body, so the
    // body need not be one the collector collects for the
    // diagnostic to be a compilation error (see
    // `convert::with_body_materialization`).
    if let Some(key) = crate::convert::materialization_lint() {
        return Some(format!(
            "materialization of `{key}` emitted a lint error \
             (rustc lints every local body; see the diagnostics \
             above)"
        ));
    }
    if check_mode() {
        // A check compilation never runs the collector, so
        // the walk does not run either: its findings replay
        // collector work, and its constant evaluations emit
        // the very diagnostics the check compilation cannot
        // produce. (Its crash witnessing belongs to the
        // collector replay too; conversion's own crash
        // containment is unaffected.) The generic-export
        // verdict is metadata-visible, so it still decides;
        // the session-level codegen verdicts are computed
        // for the record and reported as notes. The
        // duplicate-symbol verdict runs over the export
        // roots alone — the collected-instance set is the
        // walk's.
        if let Some(finding) = generic_export_finding(tcx) {
            return Some(finding);
        }
        let roots_only = std::collections::HashSet::new();
        for finding in [
            duplicate_symbol_finding(tcx, &roots_only),
            common_linkage_finding(tcx),
            panic_runtime_finding(tcx),
            lto_prefer_dynamic_finding(tcx),
            target_cpu_finding(tcx),
            dump_symbol_finding(tcx),
            packed_stack_finding(tcx, &roots_only),
        ] {
            if let Some(finding) = finding {
                note_check_mode_verdict(finding);
            }
        }
        *MONO_ITEM_FINDING.lock().unwrap() = None;
        return None;
    }
    let mut walk = RequiredConstWalk {
        tcx,
        visited: std::collections::HashSet::new(),
        visited_allocs: std::collections::HashSet::new(),
        demanded_from_local: std::collections::HashSet::new(),
        processing_local: true,
        queue: std::collections::VecDeque::new(),
        finding: None,
        crash: None,
        pending_crash: None,
        deferred_layout_finding: None,
        recursion_limit_finding: None,
        mono_finding: None,
        force_abandoned: false,
        child_depths: std::rc::Rc::new(std::collections::HashMap::new()),
    };
    walk.collect_roots();
    walk.run();
    *MONO_ITEM_FINDING.lock().unwrap() = walk.mono_finding.take();
    // A crash the walk established pre-empts every finding:
    // it is reported through the crash slot, not as a
    // compilation failure (see `convert::note_compiler_crash`).
    if let Some(record) = walk.crash.take() {
        crate::convert::note_compiler_crash(record);
        return None;
    }
    // The collector's recursion-limit verdict pre-empts a
    // crash the walk's forces recorded on the way (see
    // `pending_crash`): the compiler aborts the compilation
    // in collection, before any codegen-phase force exists.
    if let Some(finding) = walk.recursion_limit_finding.take() {
        return Some(finding);
    }
    if let Some(record) = walk.pending_crash.take() {
        crate::convert::note_compiler_crash(record);
        return None;
    }
    walk.finding
        .or_else(|| walk.deferred_layout_finding.take())
        .or_else(|| generic_export_finding(tcx))
        .or_else(|| duplicate_symbol_finding(tcx, &walk.visited))
        .or_else(|| common_linkage_finding(tcx))
        .or_else(|| panic_runtime_finding(tcx))
        .or_else(|| lto_prefer_dynamic_finding(tcx))
        .or_else(|| target_cpu_finding(tcx))
        .or_else(|| dump_symbol_finding(tcx))
        .or_else(|| packed_stack_finding(tcx, &walk.visited))
}

/// The `check_mono_item` finding of the most recent
/// `check_required_consts` walk, held for the verdict chain's
/// last step (`take_mono_item_finding`).
static MONO_ITEM_FINDING: std::sync::Mutex<Option<String>> =
    std::sync::Mutex::new(None);

/// The deferred `check_mono_item` verdict: the rejection the
/// required-constant walk's per-instance forces produced, if
/// any. It is chained after `check_bodies` in the driver, so a
/// body check that names the same rejection more precisely
/// reports first. Reading clears the slot; the next
/// `check_required_consts` call overwrites it regardless.
pub fn take_mono_item_finding() -> Option<String> {
    MONO_ITEM_FINDING.lock().unwrap().take()
}

/// Whether this process judges a compiletest check
/// compilation, set once from the driver's
/// `--compiletest-check-mode` flag before the compiler runs
/// (see `set_check_mode`).
static CHECK_MODE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Record whether this process judges a compiletest check
/// compilation (`--emit=metadata`): the runner sets the flag
/// for the views compiletest compiles in check mode
/// (check-pass / check-fail and directive-less UI views,
/// incremental `cpass` revisions). In check mode the
/// codegen-level verdicts of this module are reported as
/// notes instead of deciding the compilation, because the
/// reference compilation never runs the monomorphization
/// collector or code generation and cannot raise them
/// (Issue #509).
pub fn set_check_mode(on: bool) {
    CHECK_MODE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether this process judges a compiletest check
/// compilation (see `set_check_mode`).
pub fn check_mode() -> bool {
    CHECK_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

/// The codegen-level verdicts a check-mode compilation
/// computed but did not apply, in the order they were
/// recorded; drained by `take_check_mode_notes`.
static CHECK_MODE_NOTES: std::sync::Mutex<Vec<String>> =
    std::sync::Mutex::new(Vec::new());

/// Record a codegen-level verdict a check-mode compilation
/// computed but did not apply (see `set_check_mode`).
pub fn note_check_mode_verdict(finding: String) {
    CHECK_MODE_NOTES.lock().unwrap().push(finding);
}

/// Divert a codegen-level verdict in check mode: the
/// finding is recorded as a note and does not fail the
/// compilation. Outside check mode the finding passes
/// through unchanged.
pub fn divert_check_mode(finding: Option<String>) -> Option<String> {
    if check_mode() {
        if let Some(finding) = finding {
            note_check_mode_verdict(finding);
        }
        None
    } else {
        finding
    }
}

/// The recorded check-mode notes (see
/// `note_check_mode_verdict`), drained.
pub fn take_check_mode_notes() -> Vec<String> {
    std::mem::take(&mut *CHECK_MODE_NOTES.lock().unwrap())
}

/// The auxiliary crates this driver process built for the
/// compilation being judged, as (metadata crate name, the
/// build was metadata-only in native shape) pairs, published
/// by the driver before the main compilation runs (see
/// `set_aux_crates`). The missing-MIR verdict consults it:
/// only these crates can carry the `-Zalways-encode-mir`
/// encoding the auxiliary builds use, so only for them does
/// `is_mir_available` answer more than a native build's
/// metadata would.
static AUX_CRATES: std::sync::Mutex<Vec<(String, bool)>> = std::sync::Mutex::new(Vec::new());

/// Publish the auxiliary crates built for this compilation
/// (see `AUX_CRATES`). Called once per driver run, before the
/// main compilation's session starts; a later call replaces
/// the list.
pub fn set_aux_crates(crates: Vec<(String, bool)>) {
    *AUX_CRATES.lock().unwrap() = crates;
}

/// Whether the named crate is one this process built as an
/// auxiliary, and if so whether its native build was
/// metadata-only (`--emit=metadata` among its own compile
/// flags: no output the encoder counts as code generation,
/// so a native metadata carries no optimized MIR at all).
/// `None` for every other crate, whose metadata answers
/// `is_mir_available` truthfully.
fn aux_native_metadata_only(crate_name: &str) -> Option<bool> {
    AUX_CRATES
        .lock()
        .unwrap()
        .iter()
        .find(|(name, _)| name == crate_name)
        .map(|(_, metadata_only)| *metadata_only)
}

/// The exported-symbol verdict over the crate's export roots,
/// mirroring the check rustc performs when it names the
/// crate's exported symbols: a function carrying an export
/// attribute (`#[export_name]` / `#[no_mangle]`) that is
/// generic over types or consts has no single symbol to
/// export, and rustc rejects the crate ("functions generic
/// over types or consts must be mangled"). The root set is
/// the one `duplicate_symbol_finding` enumerates: the local
/// functions whose codegen attributes name a symbol or drop
/// mangling. A generic root is rejected here, so the
/// duplicate-symbol check below never has to name it.
fn generic_export_finding<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
) -> Option<String> {
    for local_def_id in tcx.iter_local_def_id() {
        let def_id = local_def_id.to_def_id();
        if !matches!(
            tcx.def_kind(def_id),
            rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn
        ) {
            continue;
        }
        // An associated item declared in a trait is not an
        // export candidate at all: the attribute is inert on
        // trait items (rustc warns it "cannot be used on"
        // them), and the compiler's own check — the
        // `InvalidNoMangleItems` late lint — visits free
        // functions and constants only, never trait items.
        // A generic trait method carrying the attribute is
        // therefore accepted, as in
        // tests/ui/feature-gates/issue-43106-gating-of-builtin-attrs.rs,
        // where `#[no_mangle]` sits on the methods of a trait.
        if matches!(
            tcx.def_kind(def_id),
            rustc_hir::def::DefKind::AssocFn
        ) && matches!(
            tcx.associated_item(def_id).container,
            rustc_middle::ty::AssocContainer::Trait
        ) {
            continue;
        }
        let attrs = tcx.codegen_fn_attrs(def_id);
        let exported = attrs.symbol_name.is_some()
            || attrs.flags.contains(
                rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags::NO_MANGLE,
            );
        if exported && tcx.generics_of(def_id).requires_monomorphization(tcx) {
            return Some(format!(
                "functions generic over types or consts must be mangled \
                 (rustc rejects the exported function `{}`: an exported \
                 symbol cannot name a generic item)",
                tcx.def_path_str(def_id)
            ));
        }
    }
    None
}

/// The duplicate-symbol verdict over the mono items the
/// required-constant walk collected, mirroring the partitioning
/// assertion in rustc_monomorphize (`assert_symbols_are_distinct`,
/// a fatal error) and the entry-symbol check in
/// rustc_codegen_ssa's entry-wrapper creation.
///
/// The item set is the walk's: every function instance the
/// collector would codegen locally, plus every local static and
/// every local `global_asm!` item (both are mono-item roots of
/// their own). Symbols are the compiler's own `symbol_name`
/// values, so two distinct items sharing one is exactly the
/// collision rustc reports. Asking for a symbol can panic for
/// shapes the compiler cannot name; such an item simply takes
/// no part in the check.
fn duplicate_symbol_finding<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    visited: &std::collections::HashSet<rustc_middle::ty::Instance<'tcx>>,
) -> Option<String> {
    // The item set, deduplicated by item (an item the walk
    // reached and an exported root below are the same mono
    // item, never two): the walk's instances, plus the local
    // statics and `global_asm!` items, plus the local functions
    // carrying an export attribute (`#[export_name]` /
    // `#[no_mangle]`) — the collector roots its collection at
    // the exported items as well as at the entry point, so an
    // exported function is a mono item even in a library crate
    // that nothing calls.
    let mut items: std::collections::HashSet<rustc_middle::ty::Instance<'tcx>> =
        visited.clone();
    for local_def_id in tcx.iter_local_def_id() {
        let def_id = local_def_id.to_def_id();
        let is_export_root = match tcx.def_kind(def_id) {
            rustc_hir::def::DefKind::Static { .. }
            | rustc_hir::def::DefKind::GlobalAsm => true,
            rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn => {
                let attrs = tcx.codegen_fn_attrs(def_id);
                attrs.symbol_name.is_some()
                    || attrs.flags.contains(
                        rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags::NO_MANGLE,
                    )
            }
            _ => false,
        };
        if is_export_root {
            // A generic function is not a mono item — asking
            // `Instance::mono` for one panics — so it takes no
            // part in the item set: its instances enter through
            // the walk when demanded with concrete arguments,
            // and an exported generic function is a compilation
            // error of its own, which `generic_export_finding`
            // reports.
            let generic_fn = matches!(
                tcx.def_kind(def_id),
                rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn
            ) && tcx
                .generics_of(def_id)
                .requires_monomorphization(tcx);
            if !generic_fn {
                items.insert(rustc_middle::ty::Instance::mono(tcx, def_id));
            }
        }
    }
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut duplicate: Option<String> = None;
    for instance in &items {
        if !tcx.should_codegen_locally(*instance) {
            continue;
        }
        // The naming runs under the resolution containment
        // (see `convert::with_instance_resolution_contained`):
        // the walk's item set is broader than the set the
        // native partitioning names -- it visits instances
        // for their constants, including ones whose generic
        // arguments still mention a const expression the
        // mangler cannot print (`ty::ConstKind::Expr`, Issue
        // #510) -- and a naming failure there, by panic or by
        // diagnostic, must leave no trace in the session: the
        // item simply takes no part in the check, as the
        // contract above states. A panic's hook record is
        // consumed for the same reason (the discipline
        // `link_symbol_of` follows).
        let (symbol, _emitted) = crate::convert::with_instance_resolution_contained(|| {
            tcx.symbol_name(*instance).name.to_string()
        });
        match symbol {
            Ok(symbol) => {
                if duplicate.is_none() && !seen.insert(symbol.clone()) {
                    duplicate = Some(symbol);
                }
            }
            Err(_payload) => {
                let _ = crate::convert::take_captured_panic();
            }
        }
    }
    if let Some(symbol) = duplicate {
        return Some(format!(
            "symbol `{symbol}` is already defined \
             (rustc rejects duplicate symbols among the \
             monomorphized items of a crate)"
        ));
    }
    // The entry wrapper code generation synthesizes for an
    // executable's entry point carries the target's entry
    // symbol; a mono item that already defines that symbol
    // makes the wrapper a second definition, which
    // rustc_codegen_ssa reports fatally. A foreign declaration
    // of the symbol counts for this check (the code generator
    // looks the symbol up among the declared values), though
    // declarations are not definitions and never take part in
    // the duplicate count above.
    if tcx.entry_fn(()).is_some()
        && tcx
            .crate_types()
            .iter()
            .any(|ty| *ty == rustc_structures::CrateType::Executable)
    {
        let entry_name: &str = tcx.sess.target.entry_name.as_ref();
        let declared = tcx.iter_local_def_id().any(|local_def_id| {
            let def_id = local_def_id.to_def_id();
            if !tcx.is_foreign_item(def_id) {
                return false;
            }
            // Same containment as the duplicate count above:
            // a declaration the compiler cannot name takes no
            // part in the check and leaves no trace.
            let (symbol, _emitted) =
                crate::convert::with_instance_resolution_contained(|| {
                    tcx.symbol_name(rustc_middle::ty::Instance::mono(tcx, def_id))
                        .name
                        .to_string()
                });
            match symbol {
                Ok(symbol) => symbol == entry_name,
                Err(_payload) => {
                    let _ = crate::convert::take_captured_panic();
                    false
                }
            }
        });
        if seen.contains(entry_name) || declared {
            return Some(format!(
                "entry symbol `{entry_name}` declared multiple times \
                 (a monomorphized item already defines the symbol \
                 the entry wrapper synthesizes)"
            ));
        }
    }
    None
}

/// Mirror of the panic-runtime validation rustc performs when
/// code generation computes a crate's dependency formats
/// (`rustc_metadata::dependency_format::verify_ok`): at most one
/// panic runtime may be linked, its strategy must be the
/// session's, and every other linked crate's required panic
/// strategy must be compatible with it (with the
/// immediate-abort rule applied to the whole graph first).
///
/// The validation is reproduced, not forced: the driver's
/// session emits only metadata, under which the real
/// `dependency_formats` computation produces empty lists and
/// validates nothing, so the linkage list of each crate type
/// is reconstructed here from the facts the session does hold
/// (dependency kinds, the dylib dependency formats recorded in
/// each crate's metadata, and the session's linkage options),
/// following the computation's own construction:
///
/// - the preferred linkage of a crate type is static, except
///   that an executable with `-C prefer-dynamic` (and no
///   crt-static), a dylib with `-C prefer-dynamic`, and a
///   static library built with
///   `-Zstaticlib-prefer-dynamic` prefer dynamic linkage;
/// - the static list is the local crate and every
///   unconditional dependency (`attempt_static`), with the
///   injected panic runtime activated when no panic runtime
///   is linked yet;
/// - the dynamic list is the sweep over the graph's dylibs:
///   each dylib is linked dynamically and the crates its
///   recorded dylib dependency formats name are linked as
///   recorded (a crate recorded `RequireStatic` is included
///   in the dylib), after which unconditional crates no dylib
///   claims are linked statically. The injected panic runtime
///   is activated only when the sweep linked no panic
///   runtime — which is how a `prefer-dynamic` executable
///   whose standard library was built with the unwind runtime
///   comes to link `panic_unwind` (recorded as a static
///   dependency of the `std` dylib) under an abort session.
///
/// Two facts of the metadata session shape the reconstruction.
/// The loaded source of every crate reports an rmeta and no
/// rlib or dylib, so which crates are available as dylibs
/// cannot be read from the sources; a crate's metadata records
/// its dylib dependency formats exactly when the crate was
/// itself compiled as a dylib (the formats are the linkage
/// that dylib was built with), so the crates with non-empty
/// recorded formats are taken as the graph's dylibs. (A dylib
/// with no upstream crates records an empty list and escapes
/// the identification; it can be neither a panic runtime nor
/// a crate with a required panic strategy in any graph the
/// formats describe, and the sweep's static fill still links
/// it when it is unconditional.) And whether the static
/// construction would fall back to the dynamic sweep for want
/// of an rlib source is likewise not decidable here, so the
/// static list is used whenever the preferred linkage is
/// static. An rlib-only compilation computes an empty list
/// natively and validates nothing, so it is skipped the same
/// way.
/// The common-linkage verdict over the crate's local
/// statics, mirroring the LLVM verifier's rejection of a
/// common global with a non-zero initializer ("'common'
/// global must have a zero initializer"), which code
/// generation hits fatally when it emits the module.
///
/// The static set is the one `duplicate_symbol_finding`
/// enumerates (every local static), the linkage is the
/// compiler's own codegen attribute for the static, and
/// the initializer is the compiler's own static
/// evaluation, so a static reported here is exactly one
/// whose emitted global the verifier rejects. Only
/// positive evidence counts: a relocation anywhere in the
/// initializer (a pointer constant is not a zero
/// initializer), or a non-zero byte in the evaluated
/// initializer. A foreign static has no initializer to
/// evaluate, and a static whose evaluation fails takes no
/// part — the required-constant walk reports evaluation
/// failures of collected statics itself.
fn common_linkage_finding<'tcx>(tcx: rustc_middle::ty::TyCtxt<'tcx>) -> Option<String> {
    for local_def_id in tcx.iter_local_def_id() {
        let def_id = local_def_id.to_def_id();
        if !matches!(tcx.def_kind(def_id), rustc_hir::def::DefKind::Static { .. }) {
            continue;
        }
        let attrs = tcx.codegen_fn_attrs(def_id);
        if attrs.linkage != Some(rustc_attr_ir::Linkage::Common) {
            continue;
        }
        let Ok(alloc) = tcx.eval_static_initializer(def_id) else {
            continue;
        };
        let inner = alloc.inner();
        let bytes = inner.inspect_with_uninit_and_ptr_outside_interpreter(0..inner.len());
        let relocated = inner.provenance().ptrs().values().next().is_some();
        if relocated || bytes.iter().any(|&b| b != 0) {
            return Some(format!(
                "static `{}` has common linkage but a non-zero initializer \
                 ('common' global must have a zero initializer)",
                tcx.def_path_str(def_id)
            ));
        }
    }
    None
}

/// The LTO / prefer-dynamic verdict, over the session's own
/// option state (see the module documentation): LTO is
/// requested (`-C lto` thin or fat), dynamic linking is
/// preferred, and every crate type the session produces is
/// an output LTO is performed for. A crate type outside
/// that set lifts the rejection exactly as it does under
/// rustc: an rlib output carries its bitcode to a later
/// link, so no LTO link of this crate happens, and a dylib
/// or proc-macro output is rejected separately, for using
/// LTO at all without `-Zdylib-lto`.
fn lto_prefer_dynamic_finding(
    tcx: rustc_middle::ty::TyCtxt<'_>,
) -> Option<String> {
    use rustc_structures::CrateType;

    if !tcx.sess.opts.cg.prefer_dynamic
        || tcx.sess.lto() == rustc_session::config::Lto::No
    {
        return None;
    }
    let types = tcx.crate_types();
    let lto_outputs_only = !types.is_empty()
        && types.iter().all(|ty| {
            matches!(
                ty,
                CrateType::Executable
                    | CrateType::StaticLib
                    | CrateType::Cdylib
            )
        });
    if lto_outputs_only {
        return Some(
            "cannot prefer dynamic linking when performing LTO \
             (rustc rejects `-C lto` with `-C prefer-dynamic` \
             when every output of the crate is a 'staticlib', \
             'bin' or 'cdylib' output)"
                .to_string(),
        );
    }
    None
}

/// The target-cpu verdict over the session's own state,
/// mirroring the two fatal checks `rustc_codegen_ssa`
/// performs as code generation starts (`codegen_crate`,
/// and `create_metadata` for the AVR explicit-cpu case):
/// a target whose spec sets `need_explicit_cpu` has no
/// default cpu, so a session that sets no `-C target-cpu`
/// cannot codegen for it at all; and a `-C target-cpu`
/// named in the target spec's `unsupported_cpus` list is
/// known to the target but unsupported by it. Both
/// predicates read only the target spec and the session's
/// codegen options -- the same evidence the driver's
/// session already carries -- so the replay is exact:
/// it fires for precisely the sessions rustc rejects.
/// (Verified against the pinned rustc 1.100.0: each
/// rejection shape exits 0 under `--emit=metadata` and 1
/// under `--emit=link`, and each neighbouring acceptance
/// -- an explicit supported cpu -- exits 0 under both.)
fn target_cpu_finding(tcx: rustc_middle::ty::TyCtxt<'_>) -> Option<String> {
    let target_cpu = &tcx.sess.opts.cg.target_cpu;
    if tcx.sess.target.need_explicit_cpu && target_cpu.is_none() {
        return Some(
            "target requires explicitly specifying a cpu with \
             `-C target-cpu` (rustc rejects a target whose spec \
             requires an explicit cpu when the session sets none, \
             at code generation)"
                .to_string(),
        );
    }
    if let Some(target_cpu) = target_cpu {
        let listed = tcx
            .sess
            .target
            .unsupported_cpus
            .iter()
            .any(|cpu| &**cpu == target_cpu.as_str());
        if listed {
            return Some(format!(
                "target cpu `{target_cpu}` is known but unsupported \
                 (rustc rejects a `-C target-cpu` the target spec \
                 lists as unsupported, at code generation)"
            ));
        }
    }
    None
}

/// The symbol-name / def-path dump verdict, mirroring
/// `rustc_symbol_mangling::test::dump_symbol_names_and_def_paths`,
/// the test hook `start_codegen` runs as code generation
/// starts (and only when the output types should codegen,
/// which a metadata-only session's never do): when the
/// `rustc_attrs` feature is enabled, every local item
/// carrying `#[rustc_dump_symbol_name]` or
/// `#[rustc_dump_def_path]` is reported with an error
/// carrying its mangled symbol name or its def path, so
/// the compilation fails. The evidence is the attribute
/// itself on a local item -- the same evidence the hook
/// walks -- so the replay fires for precisely the
/// compilations rustc rejects this way, and stays silent
/// when the feature is off (the attributes cannot be
/// parsed as dump attributes then, and the hook returns
/// before walking) or no local item carries either one.
fn dump_symbol_finding(tcx: rustc_middle::ty::TyCtxt<'_>) -> Option<String> {
    if !tcx.features().rustc_attrs() {
        return None;
    }
    for local_def_id in tcx.iter_local_def_id() {
        let def_id = local_def_id.to_def_id();
        for dump in [
            (
                rustc_attr_ir::find_attr!(tcx, def_id, RustcDumpSymbolName(..)),
                "symbol-name",
            ),
            (
                rustc_attr_ir::find_attr!(tcx, def_id, RustcDumpDefPath(..)),
                "def-path",
            ),
        ] {
            if dump.0 {
                return Some(format!(
                    "`{}` carries a rustc dump attribute, so code \
                     generation reports its {} as an error \
                     (rustc's dump test hook rejects the \
                     compilation at code generation)",
                    tcx.def_path_str(def_id),
                    dump.1
                ));
            }
        }
    }
    None
}

/// The packed-stack / backchain verdict, mirroring
/// `packed_stack_attr` in rustc_codegen_llvm's function
/// attribute computation, which runs for every function
/// code generation defines: on s390x, when
/// `-Zpacked-stack` is set, a function whose target
/// features enable `backchain` without also enabling
/// `soft-float` is an error ("`-Zpacked-stack` is
/// incompatible with `backchain` target feature"),
/// because LLVM cannot generate valid function entry
/// points for the combination under the default ABI.
/// A function's features are the session's internal
/// target features (the target spec's defaults plus
/// `-C target-feature`) merged with the function's own
/// `#[target_feature]` list, exactly as the attribute
/// computation reads them. The evidence is the session
/// state and the codegen attributes of the items code
/// generation would define -- the collector walk's
/// instances plus the exported roots, the item set of
/// the duplicate-symbol verdict -- so the replay fires
/// for precisely the compilations rustc rejects this
/// way, and stays silent on every other target, without
/// `-Zpacked-stack`, and whenever `soft-float` is
/// enabled alongside `backchain` (a soft-float target's
/// spec enables it, which is what lifts the rejection
/// for the record's `with_softfloat` revision).
/// (Verified against the pinned rustc 1.100.0: the
/// attribute-carried and session-carried shapes each
/// exit 0 under `--emit=metadata` and 1 under
/// `--emit=link`; packed-stack alone and the
/// soft-float-target shape exit 0 under both.)
fn packed_stack_finding<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    visited: &std::collections::HashSet<rustc_middle::ty::Instance<'tcx>>,
) -> Option<String> {
    if tcx.sess.target.arch != rustc_target::spec::Arch::S390x
        || !tcx.sess.opts.unstable_opts.packed_stack
    {
        return None;
    }
    let session_backchain =
        tcx.sess.internal_target_features.contains(&rustc_span::sym::backchain);
    let session_soft_float =
        tcx.sess.internal_target_features.contains(&rustc_span::sym::soft_float);
    // The item set, as in `duplicate_symbol_finding`: the
    // walk's instances plus the local functions carrying
    // an export attribute -- the collector roots its
    // collection at the exported items, so an exported
    // function is a mono item even in a library crate
    // that nothing calls (the record's shape).
    let mut items: std::collections::HashSet<rustc_middle::ty::Instance<'tcx>> =
        visited.clone();
    for local_def_id in tcx.iter_local_def_id() {
        let def_id = local_def_id.to_def_id();
        if !matches!(
            tcx.def_kind(def_id),
            rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn
        ) {
            continue;
        }
        let attrs = tcx.codegen_fn_attrs(def_id);
        let exported = attrs.symbol_name.is_some()
            || attrs.flags.contains(
                rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags::NO_MANGLE,
            );
        if exported
            && !tcx
                .generics_of(def_id)
                .requires_monomorphization(tcx)
        {
            items.insert(rustc_middle::ty::Instance::mono(tcx, def_id));
        }
    }
    for instance in &items {
        if !tcx.should_codegen_locally(*instance) {
            continue;
        }
        let def_id = instance.def_id();
        if !matches!(
            tcx.def_kind(def_id),
            rustc_hir::def::DefKind::Fn
                | rustc_hir::def::DefKind::AssocFn
                | rustc_hir::def::DefKind::Closure
        ) {
            continue;
        }
        let attrs = tcx.codegen_fn_attrs(def_id);
        let have_backchain = session_backchain
            || attrs
                .target_features
                .iter()
                .any(|feature| feature.name == rustc_span::sym::backchain);
        let have_soft_float = session_soft_float
            || attrs
                .target_features
                .iter()
                .any(|feature| feature.name == rustc_span::sym::soft_float);
        if have_backchain && !have_soft_float {
            return Some(format!(
                "`-Zpacked-stack` is incompatible with \
                 `backchain` target feature (rustc rejects \
                 the combination when code generation \
                 defines `{}`, whose target features enable \
                 `backchain` without `soft-float`)",
                tcx.def_path_str(def_id)
            ));
        }
    }
    None
}

fn panic_runtime_finding(tcx: rustc_middle::ty::TyCtxt<'_>) -> Option<String> {
    use rustc_structures::CrateType;

    let crates = tcx.crates(());
    let desired = tcx.sess.panic_strategy();
    for ty in tcx.crate_types() {
        // The preferred linkage of the crate type
        // (`dependency_format::calculate_type`): an rlib
        // computes no linkage list at all.
        let dynamic = match ty {
            CrateType::Executable => {
                tcx.sess.opts.cg.prefer_dynamic && !tcx.sess.crt_static(Some(*ty))
            }
            CrateType::Dylib | CrateType::Cdylib | CrateType::Sdylib => {
                tcx.sess.opts.cg.prefer_dynamic
            }
            CrateType::StaticLib => tcx.sess.opts.unstable_opts.staticlib_prefer_dynamic,
            CrateType::ProcMacro => false,
            CrateType::Rlib => continue,
        };
        let (linked, conflict) = if dynamic {
            dynamic_linkage_list(tcx, crates, desired)
        } else {
            (static_linkage_list(tcx, crates, desired), None)
        };
        if let Some(finding) = conflict {
            return Some(finding);
        }
        if let Some(finding) = verify_panic_linkage(tcx, &linked, desired) {
            return Some(finding);
        }
    }
    None
}

/// The panic runtime the loader injected for the session
/// strategy: the panic-runtime crate whose name matches the
/// strategy's runtime — the name the loader itself injects
/// under, which is how the crate store's injected-runtime
/// slot is populated.
fn injected_panic_runtime(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    crates: &[rustc_span::def_id::CrateNum],
    desired: rustc_target::spec::PanicStrategy,
) -> Option<rustc_span::def_id::CrateNum> {
    use rustc_target::spec::PanicStrategy;
    let wanted = match desired {
        PanicStrategy::Unwind => Some("panic_unwind"),
        PanicStrategy::Abort => Some("panic_abort"),
        PanicStrategy::ImmediateAbort => None,
    }?;
    crates
        .iter()
        .copied()
        .find(|&cnum| tcx.is_panic_runtime(cnum) && tcx.crate_name(cnum).as_str() == wanted)
}

/// The static linkage list (`dependency_format`'s
/// `attempt_static` result): the local crate and every
/// unconditional dependency, with the injected panic runtime
/// activated when no panic runtime is linked yet
/// (`activate_injected_dep`).
fn static_linkage_list(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    crates: &[rustc_span::def_id::CrateNum],
    desired: rustc_target::spec::PanicStrategy,
) -> Vec<rustc_span::def_id::CrateNum> {
    use rustc_crate_store::CrateDepKind;
    use rustc_span::def_id::LOCAL_CRATE;

    let mut linked: Vec<rustc_span::def_id::CrateNum> = vec![LOCAL_CRATE];
    for &cnum in crates {
        if tcx.crate_dep_kind(cnum) == CrateDepKind::Unconditional {
            linked.push(cnum);
        }
    }
    if !linked.iter().any(|&cnum| tcx.is_panic_runtime(cnum)) {
        if let Some(injected) = injected_panic_runtime(tcx, crates, desired) {
            linked.push(injected);
        }
    }
    linked
}

/// The dynamic linkage list (`dependency_format`'s sweep for a
/// dynamically-preferring crate type), in crate-number order,
/// with the injected panic runtime activated when the sweep
/// linked no panic runtime. The second return value is the
/// finding for a crate the sweep would have to link in two
/// formats, which the computation rejects outright
/// (`CrateDepMultiple`).
fn dynamic_linkage_list(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    crates: &[rustc_span::def_id::CrateNum],
    desired: rustc_target::spec::PanicStrategy,
) -> (Vec<rustc_span::def_id::CrateNum>, Option<String>) {
    use rustc_crate_store::{CrateDepKind, LinkagePreference};
    use rustc_span::def_id::{CrateNum, LOCAL_CRATE};

    // The graph's dylibs, identified by the recorded dylib
    // dependency formats as described at `panic_runtime_finding`.
    let all_dylibs: Vec<CrateNum> = crates
        .iter()
        .copied()
        .filter(|&cnum| {
            !tcx.crate_dep_kind(cnum).macros_only()
                && !tcx.dylib_dependency_formats(cnum).is_empty()
        })
        .collect();
    // With `rustc_private`, a crate some dylib statically
    // requires is linked inside that dylib and is not swept
    // as a dylib in its own right.
    let mut upstream_in_dylibs: std::collections::HashSet<CrateNum> =
        std::collections::HashSet::new();
    if tcx.features().rustc_private() {
        for &cnum in &all_dylibs {
            for &(depnum, style) in tcx.dylib_dependency_formats(cnum) {
                if style == LinkagePreference::RequireStatic {
                    upstream_in_dylibs.insert(depnum);
                }
            }
        }
    }
    let mut formats: std::collections::HashMap<CrateNum, LinkagePreference> =
        std::collections::HashMap::new();
    let mut conflict: Option<CrateNum> = None;
    let mut add_library = |cnum: CrateNum, link: LinkagePreference| {
        if let Some(&existing) = formats.get(&cnum) {
            if (existing != link || link == LinkagePreference::RequireStatic)
                && conflict.is_none()
            {
                conflict = Some(cnum);
            }
        }
        formats.insert(cnum, link);
    };
    for &cnum in &all_dylibs {
        if upstream_in_dylibs.contains(&cnum) {
            continue;
        }
        add_library(cnum, LinkagePreference::RequireDynamic);
        for &(depnum, style) in tcx.dylib_dependency_formats(cnum) {
            add_library(depnum, style);
        }
    }
    if let Some(cnum) = conflict {
        return (
            Vec::new(),
            Some(format!(
                "crate `{}` is required to be available in multiple \
                 formats (rustc rejects a crate that would be linked \
                 both statically and dynamically)",
                tcx.crate_name(cnum)
            )),
        );
    }
    // Unconditional crates no dylib claims and that are not
    // themselves dylibs are linked statically.
    for &cnum in crates {
        if !all_dylibs.contains(&cnum)
            && !formats.contains_key(&cnum)
            && tcx.crate_dep_kind(cnum) == CrateDepKind::Unconditional
        {
            formats.insert(cnum, LinkagePreference::RequireStatic);
        }
    }
    let mut linked: Vec<CrateNum> = vec![LOCAL_CRATE];
    for &cnum in crates {
        if formats.contains_key(&cnum) {
            linked.push(cnum);
        }
    }
    if !linked.iter().any(|&cnum| tcx.is_panic_runtime(cnum)) {
        if let Some(injected) = injected_panic_runtime(tcx, crates, desired) {
            linked.push(injected);
            linked.sort_unstable();
        }
    }
    (linked, None)
}

/// The validation itself (`dependency_format::verify_ok`) over
/// one constructed linkage list.
fn verify_panic_linkage(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    linked: &[rustc_span::def_id::CrateNum],
    desired: rustc_target::spec::PanicStrategy,
) -> Option<String> {
    use rustc_span::def_id::{CrateNum, LOCAL_CRATE};
    use rustc_target::spec::PanicStrategy;

    let strategy_name = |strategy: PanicStrategy| match strategy {
        PanicStrategy::Unwind => "unwind",
        PanicStrategy::Abort => "abort",
        PanicStrategy::ImmediateAbort => "immediate-abort",
    };
    // The immediate-abort rule applies to the whole graph
    // before any single runtime is selected.
    if linked
        .iter()
        .any(|&cnum| tcx.required_panic_strategy(cnum) == Some(PanicStrategy::ImmediateAbort))
    {
        for &cnum in linked {
            if tcx.required_panic_strategy(cnum) != Some(PanicStrategy::ImmediateAbort) {
                return Some(format!(
                    "the crate `{}` was compiled with a panic strategy \
                     which is incompatible with `immediate-abort` \
                     (rustc validates the panic strategies of the \
                     whole crate graph at code generation)",
                    tcx.crate_name(cnum)
                ));
            }
        }
    }
    let mut runtime: Option<(CrateNum, PanicStrategy)> = None;
    for &cnum in linked {
        if !tcx.is_panic_runtime(cnum) {
            continue;
        }
        if let Some((prev, _)) = runtime {
            return Some(format!(
                "cannot link together two panic runtimes: {} and {} \
                 (rustc links at most one panic runtime into an output)",
                tcx.crate_name(prev),
                tcx.crate_name(cnum)
            ));
        }
        // A panic runtime whose strategy the metadata cannot
        // determine is a compiler bug in rustc; here it simply
        // takes no part in the validation.
        runtime = tcx.required_panic_strategy(cnum).map(|s| (cnum, s));
    }
    let (runtime_cnum, found_strategy) = runtime?;
    if found_strategy != desired {
        return Some(format!(
            "the linked panic runtime `{}` is not compiled with this \
             crate's panic strategy `{}` (rustc validates the linked \
             panic runtime against the session strategy at code \
             generation)",
            tcx.crate_name(runtime_cnum),
            strategy_name(desired)
        ));
    }
    for &cnum in linked {
        if cnum == runtime_cnum || tcx.is_compiler_builtins(cnum) {
            continue;
        }
        if let Some(found) = tcx.required_panic_strategy(cnum) {
            if found != desired {
                return Some(format!(
                    "the crate `{}` requires panic strategy `{}` which \
                     is incompatible with this crate's strategy of `{}` \
                     (rustc validates the panic strategies of the \
                     whole crate graph at code generation)",
                    tcx.crate_name(cnum),
                    strategy_name(found),
                    strategy_name(desired)
                ));
            }
        }
        if cnum != LOCAL_CRATE
            && tcx.panic_in_drop_strategy(cnum) != tcx.sess.opts.unstable_opts.panic_in_drop
        {
            return Some(format!(
                "the crate `{}` was compiled with a panic-in-drop \
                 strategy which is incompatible with this crate's \
                 (rustc validates the panic strategies of the whole \
                 crate graph at code generation)",
                tcx.crate_name(cnum)
            ));
        }
    }
    None
}

/// The instance name the recursion-limit verdict reports,
/// shrunk the way the collector's own diagnostic shrinks it
/// (`shrunk_instance_name` in rustc_monomorphize): past the
/// limit the name spells a type nested hundreds of layers
/// deep, so only its head and tail are kept.
fn shrunk_instance_name(instance: rustc_middle::ty::Instance<'_>) -> String {
    let name = instance.to_string();
    let chars: Vec<char> = name.chars().collect();
    if chars.len() <= 64 {
        return name;
    }
    let head: String = chars[..32].iter().collect();
    let tail: String = chars[chars.len() - 32..].iter().collect();
    format!("{head}...{tail}")
}

/// Whether every type `instance` drags in normalizes in the
/// monomorphized environment, asked the way the collector
/// asks it: the `try_` variant of instantiate-and-normalize,
/// whose failure is exactly "the collector does not collect
/// this item". The question is asked not only of the
/// instance's signature but through the fields of the ADTs
/// the signature and arguments mention: a drop glue's
/// pointee carries its projections in its fields, not in the
/// glue's signature, and it is the field projection's failure
/// to normalize that keeps the collector from ever forcing
/// the glue. A panic in the probe answers that the types
/// normalize: the question could not be asked, which is not
/// the abandonment's premise.
fn instance_types_normalize<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    instance: rustc_middle::ty::Instance<'tcx>,
) -> bool {
    let typing_env = fully_monomorphized();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut work: Vec<rustc_middle::ty::Ty<'tcx>> =
            vec![instance.ty(tcx, typing_env)];
        work.extend(instance.args.iter().filter_map(|arg| arg.as_type()));
        let mut seen = std::collections::HashSet::new();
        while let Some(ty) = work.pop() {
            if !seen.insert(ty) {
                continue;
            }
            if instance
                .try_instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    rustc_middle::ty::EarlyBinder::bind(tcx, ty),
                )
                .is_err()
            {
                return false;
            }
            for sub in ty.walk() {
                if let Some(sub_ty) = sub.as_type() {
                    if let rustc_middle::ty::TyKind::Adt(def, args) = sub_ty.kind() {
                        for field in def.all_fields() {
                            match tcx.try_normalize_erasing_regions(
                                typing_env,
                                field.ty(tcx, args),
                            ) {
                                Ok(field_ty) => work.push(field_ty),
                                Err(_) => return false,
                            }
                        }
                    }
                }
            }
        }
        true
    }))
    .unwrap_or(true)
}

struct RequiredConstWalk<'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    visited: std::collections::HashSet<rustc_middle::ty::Instance<'tcx>>,
    visited_allocs: std::collections::HashSet<rustc_middle::mir::interpret::AllocId>,
    /// Instances some local-crate processing demanded (a
    /// mention, a constant chase or a root of the local
    /// crate): the missing-MIR verdict's demand-origin
    /// evidence (see `missing_mir_finding`). An instance
    /// demanded only through another upstream crate's bodies
    /// is not in this set.
    demanded_from_local: std::collections::HashSet<rustc_middle::ty::Instance<'tcx>>,
    /// Whether the instance currently being processed is a
    /// local-crate instance; `enqueue` reads it to maintain
    /// `demanded_from_local`. True while the roots are
    /// collected (the roots are the local crate's own).
    processing_local: bool,
    queue: std::collections::VecDeque<(
        rustc_middle::ty::Instance<'tcx>,
        std::rc::Rc<std::collections::HashMap<rustc_span::def_id::DefId, usize>>,
    )>,
    finding: Option<String>,
    /// The record of a compiler-bug panic a force of this
    /// walk raised (see `convert::captured_panic_is_compiler_bug`):
    /// the walk's forces are the collector's own, so the
    /// compiler dying inside one is a crash verdict, kept
    /// apart from the findings and reported as one by
    /// `check_required_consts`. This slot is for the crash
    /// verdicts that stand immediately: a body-materialization
    /// bug recorded during conversion, which `process`
    /// surfaces for the instance that body belongs to.
    crash: Option<String>,
    /// The first compiler-bug witness a force of this walk
    /// raised, held back instead of ending the walk: the
    /// walk's forces replay the compiler's codegen-phase
    /// per-instance checks, but the collector's own verdicts
    /// are collection-phase, and in the compiler collection
    /// runs first — a compilation the collector aborts (the
    /// recursion limit below) never reaches the forces at
    /// all. So a force's crash is only a verdict once the
    /// walk has run its course without a collection-phase
    /// verdict pre-empting it; `check_required_consts`
    /// reports it after the recursion-limit finding.
    pending_crash: Option<String>,
    /// The collector's instantiation recursion-limit verdict
    /// (`check_recursion_limit` in rustc_monomorphize), kept
    /// apart from `finding`: it is a collection-phase failure
    /// and pre-empts a pending force crash (see
    /// `pending_crash`), which ordinary findings do not.
    recursion_limit_finding: Option<String>,
    /// The first `check_mono_item` rejection the walk's forces
    /// produced, kept apart from `finding`: it is reported
    /// after `check_bodies`, whose findings name some of the
    /// same rejections more precisely.
    mono_finding: Option<String>,
    /// The first layout verdict a panicked `check_mono_item`
    /// force produced (see `process`): the recorded layout
    /// failure of a type the instance's collection demands.
    /// Kept apart from `finding` and reported only once the
    /// walk has run its course, like a force's crash (see
    /// `pending_crash`): the ABI demand it replays is a
    /// codegen-phase event, and collection-phase verdicts
    /// pre-empt it.
    deferred_layout_finding: Option<String>,
    /// Set by `process` when this instance's `check_mono_item`
    /// force took the abandonment branch (the force panicked
    /// with no errors counted before it) AND the instance is
    /// of the abandoned kind: its types do not normalize in
    /// the monomorphized environment, so the collector would
    /// never force it (see `instance_types_normalize`). The
    /// compiler-bug witness such a force leaves is the
    /// abandoned force's own, discounted with its diagnostics,
    /// so `run` does not convert it into `crash`. A force
    /// abandoned on an instance whose types DO normalize is
    /// one the collector itself makes for a collected
    /// instance, and its witness stays a crash.
    force_abandoned: bool,
    /// The instantiation recursion-depth map the instances
    /// enqueued while the current instance is processed are
    /// queued under: for each definition, how many nested
    /// instantiations of it stand above them on the path the
    /// walk reached them by — the counts the collector's own
    /// `check_recursion_limit` consults (see
    /// `beyond_recursion_limit`). Roots are seeded under the
    /// empty map; `run` replaces this with the popped
    /// instance's child map before processing it.
    child_depths:
        std::rc::Rc<std::collections::HashMap<rustc_span::def_id::DefId, usize>>,
}

impl<'tcx> RequiredConstWalk<'tcx> {
    fn run(&mut self) {
        while let Some((instance, depths)) = self.queue.pop_front() {
            if self.finding.is_some()
                || self.crash.is_some()
                || self.recursion_limit_finding.is_some()
            {
                return;
            }
            // The collector's instantiation recursion limit
            // (`check_recursion_limit` in rustc_monomorphize):
            // an instance the walk reached through more nested
            // instantiations of its own definition than the
            // session's recursion limit allows is one the
            // collector rejects with a fatal error before it is
            // ever collected, so the walk records that verdict
            // and never processes the instance. The verdict
            // matters beyond this instance: the forces on the
            // shallower instances of a runaway chain are not
            // safe either — the ABI check inside
            // `check_mono_item` asks for the layout of types
            // nested scores of layers deep, the layout
            // computation fails, and the check's error handler
            // raises that failure as an internal compiler
            // error — yet the compiler itself never makes any
            // of those forces, because collection aborts
            // first. That is why a force's crash is deferred
            // (see `pending_crash`) and this finding pre-empts
            // it.
            if self.beyond_recursion_limit(instance, &depths) {
                if self.recursion_limit_finding.is_none() {
                    self.recursion_limit_finding = Some(format!(
                        "reached the recursion limit while instantiating `{}`",
                        shrunk_instance_name(instance)
                    ));
                }
                continue;
            }
            // The depth map this instance's mentions are
            // enqueued under: its own definition's count rises
            // by one for the descent, as in the collector.
            let mut child_depths = (*depths).clone();
            *child_depths.entry(instance.def_id()).or_insert(0) += 1;
            self.child_depths = std::rc::Rc::new(child_depths);
            // The bug-witness snapshot for this instance's
            // processing, diffed below.
            let witnesses_before = crate::convert::compiler_bug_witness_len();
            self.force_abandoned = false;
            // Instance-body access and constant evaluation go
            // through the compiler's internals and can panic on
            // exotic instances (foreign bodies with no MIR here,
            // shim shapes the queries do not expect). A panic
            // abandons that instance's walk, never the check.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.process(instance);
            }));
            // A compiler-bug panic inside the processing is
            // not an exotic instance: every force the walk
            // makes is one the compiler's own collector makes
            // for an instance it collects (the per-instance
            // `check_mono_item` force, the required-constant
            // evaluations, the vtable allocations of mentioned
            // unsize casts), so the compiler dying inside one
            // is the compiler crashing on a collected
            // instance — a crash verdict. The verdict is
            // deferred, not immediate (see `pending_crash`):
            // the walk first has to run its course, because a
            // collection-phase verdict further along the
            // mention graph — the recursion limit above —
            // pre-empts it, the way collection pre-empts
            // codegen in the compiler. The diff is scoped to
            // this instance's processing, so bug panics
            // contained on speculative paths the walk does not
            // decide (and the findings computed after the
            // walk) never reach it. One exemption: a force
            // `process` abandoned on an instance whose types do
            // not normalize (see `force_abandoned`) leaves its
            // bug witness discounted with its diagnostics: the
            // collector would never make that force, so the
            // compiler dying inside it is not a crash on a
            // collected instance.
            if self.crash.is_none()
                && self.pending_crash.is_none()
                && !self.force_abandoned
            {
                if let Some(record) =
                    crate::convert::compiler_bug_witness_since(witnesses_before)
                {
                    self.pending_crash = Some(record);
                }
            }
        }
    }

    /// The collector's recursion-limit question for one
    /// queued instance (`check_recursion_limit` in
    /// rustc_monomorphize): `depths` counts, per definition,
    /// how many nested instantiations of it stand above this
    /// instance on the path the walk reached it by, and the
    /// instance is beyond the limit when that count for its
    /// own definition — quartered for `drop_in_place`, whose
    /// tight monomorphization loops the collector gives more
    /// margin — is not within the session's recursion limit.
    /// The walk discovers instances breadth-first, so the
    /// path it holds for an instance is the shallowest one:
    /// an instance is declared beyond the limit only when
    /// every path to it is, which is exactly the set of
    /// instances the collector cannot collect either.
    fn beyond_recursion_limit(
        &self,
        instance: rustc_middle::ty::Instance<'tcx>,
        depths: &std::collections::HashMap<rustc_span::def_id::DefId, usize>,
    ) -> bool {
        let def_id = instance.def_id();
        let depth = depths.get(&def_id).copied().unwrap_or(0);
        let adjusted =
            if matches!(
                instance.def,
                rustc_middle::ty::InstanceKind::Shim(
                    rustc_middle::ty::ShimKind::DropGlue(..)
                )
            ) {
                depth / 4
            } else {
                depth
            };
        !self.tcx.recursion_limit().value_within_limit(adjusted)
    }

    fn enqueue(&mut self, instance: rustc_middle::ty::Instance<'tcx>) {
        use rustc_middle::ty::InstanceKind;
        match instance.def {
            // Intrinsics have no body to walk, and a virtual
            // instance's body is the trait definition's (its
            // constants are only evaluated for the concrete
            // instances the walk reaches separately).
            InstanceKind::Intrinsic(_)
            | InstanceKind::LlvmIntrinsic(_)
            | InstanceKind::Virtual(_, _) => return,
            _ => {}
        }
        if self.visited.len() >= MAX_REQUIRED_CONST_WALK {
            return;
        }
        if self.processing_local {
            self.demanded_from_local.insert(instance);
        }
        if self.visited.insert(instance) {
            self.queue
                .push_back((instance, self.child_depths.clone()));
        }
    }

    /// Seed the walk with the roots the monomorphization
    /// collector itself starts from (`RootCollector` in
    /// rustc_monomorphize, under the lazy collection strategy —
    /// the eager strategy is selected only by `-Zlink-dead-code`,
    /// which the driver does not pass):
    ///
    /// - the crate's entry point, when it has one;
    /// - every non-generic local function the reachability
    ///   analysis marks as a root — the entry point, exported and
    ///   `#[used]` functions, and the non-generic functions they
    ///   reference — plus functions carrying the internal-symbol
    ///   flags the collector also treats as roots;
    /// - every local static: collecting a static evaluates its
    ///   initializer, and a non-nested static's type also gets
    ///   its drop glue collected.
    /// - every `global_asm!` item's constant operands: the item
    ///   is a root of its own and emitting its assembly evaluates
    ///   the operands unconditionally; they are anonymous
    ///   constants parented by the item, evaluated where they
    ///   are found (see `collect_global_asm_const`).
    ///
    /// The instances the driver's own demand passes resolved are
    /// NOT roots. Conversion translates every local body,
    /// reachable or not, so a demand originating in an
    /// unreferenced private function would seed the walk with an
    /// instance rustc never collects, and a failing constant
    /// inside it would fail a compilation rustc accepts.
    /// Everything the collector does collect is reached from
    /// these roots by the mention walk, exactly as in the
    /// compiler.
    fn collect_roots(&mut self) {
        let tcx = self.tcx;
        // The collector's root predicate for functions is
        // `tcx.is_reachable_non_generic`, but that query's
        // provider returns an empty map when the session does
        // not codegen (the `should_codegen` gate in
        // `rustc_codegen_ssa::back::symbol_export`), and this
        // driver's session emits only metadata — through the
        // query, no function would ever be a root, and in a
        // crate with no entry point (a library, or a test whose
        // definitions the entry point never mentions) the walk
        // would have no roots at all. `reachable_non_generics_helper`
        // is the provider's own computation without the gate
        // (exposed for Miri, whose sessions have the same
        // metadata-only shape); testing its result against the
        // session's export threshold is exactly what the
        // query's local provider does in a code-generating
        // session. A panic in the computation abandons it and
        // leaves the walk with the entry point and the flag
        // roots alone, never a finding.
        let reachable_non_generics = std::panic::catch_unwind(
            std::panic::AssertUnwindSafe(|| {
                rustc_codegen_ssa::back::symbol_export::reachable_non_generics_helper(tcx)
            }),
        )
        .unwrap_or_default();
        let export_threshold =
            rustc_codegen_ssa::back::symbol_export::crates_export_threshold(
                tcx.crate_types(),
            );
        let is_collector_root = |def_id: rustc_span::def_id::DefId| {
            reachable_non_generics.get(&def_id).is_some_and(|info| {
                info.level.is_below_threshold(export_threshold)
            })
        };
        if let Some((entry_def_id, _)) = tcx.entry_fn(()) {
            self.enqueue(rustc_middle::ty::Instance::mono(tcx, entry_def_id));
        }
        for local_def_id in tcx.iter_local_def_id() {
            if self.finding.is_some() {
                return;
            }
            let def_id = local_def_id.to_def_id();
            // The root queries can panic on exotic items; a
            // panic abandons that root, never the check (the
            // same discipline as the per-instance walk in `run`).
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Foreign items (the contents of `extern`
                // blocks) are never roots: the collector's item
                // visitor does not process them, an extern
                // function has no body to collect, and an extern
                // static has no initializer to evaluate (asking
                // for one is an internal compiler error).
                if tcx.is_foreign_item(def_id) {
                    return;
                }
                match tcx.def_kind(def_id) {
                    rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn => {
                        if tcx.generics_of(def_id).requires_monomorphization(tcx) {
                            return;
                        }
                        let flags = tcx.codegen_fn_attrs(def_id).flags;
                        if is_collector_root(def_id)
                            || flags.intersects(
                                rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL
                                    | rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags::EXTERNALLY_IMPLEMENTABLE_ITEM,
                            )
                        {
                            self.enqueue(rustc_middle::ty::Instance::mono(tcx, def_id));
                        }
                    }
                    rustc_hir::def::DefKind::Static { nested, .. } => {
                        self.collect_root_static(def_id, nested);
                    }
                    rustc_hir::def::DefKind::AnonConst => {
                        // The constant operands of a `global_asm!`
                        // item are anonymous constants parented by
                        // the item; the item is a collector root of
                        // its own, so they are collected (and
                        // evaluated) with it.
                        if matches!(
                            tcx.def_kind(tcx.parent(def_id)),
                            rustc_hir::def::DefKind::GlobalAsm
                        ) {
                            self.collect_global_asm_const(def_id);
                        }
                    }
                    _ => {}
                }
            }));
        }
    }

    /// Collect one root static the way the collector visits a
    /// `MonoItem::Static`: evaluate its initializer and follow
    /// the allocations it points into (an evaluation error
    /// reported there is the same failure class as a required
    /// constant's), and collect the drop glue of a non-nested
    /// static's type.
    fn collect_root_static(&mut self, def_id: rustc_span::def_id::DefId, nested: bool) {
        let tcx = self.tcx;
        if !nested {
            let instance = rustc_middle::ty::Instance::mono(tcx, def_id);
            let ty = instance.ty(tcx, fully_monomorphized());
            let glue = rustc_middle::ty::Instance::resolve_drop_glue(tcx, ty);
            self.enqueue(glue);
        }
        match tcx.eval_static_initializer(def_id) {
            Ok(alloc) => self.chase_allocation(alloc),
            Err(rustc_middle::mir::interpret::ErrorHandled::Reported(..)) => {
                self.finding = Some(format!(
                    "initializer of static `{}` failed to evaluate \
                     (rustc reports the evaluation error during \
                     monomorphization collection)",
                    tcx.def_path_str(def_id)
                ));
            }
            Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_)) => {}
        }
    }

    /// Collect one `global_asm!` constant operand the way the
    /// collector treats a `MonoItem::GlobalAsm`: emitting the
    /// item's assembly evaluates every constant operand
    /// unconditionally, so a constant whose evaluation is
    /// reported as an error (E0080) fails the whole compilation
    /// (tests/ui/asm/fail-const-eval-issue-121099). The operands
    /// live outside every body, so no body's required-constant
    /// list ever names them and the walk must evaluate them
    /// where they are found. Only a positively reported
    /// evaluation error is a finding, as everywhere in this
    /// walk; the evaluated value is chased like a required
    /// constant's. (The item's `sym` operands name items the
    /// collector also collects; not following them only narrows
    /// the check, never widens it.)
    fn collect_global_asm_const(&mut self, def_id: rustc_span::def_id::DefId) {
        let tcx = self.tcx;
        // An anonymous constant carries one synthetic type
        // parameter (its own type, `<const_ty>`), so
        // `Instance::mono` refuses it; the identity arguments
        // are the instance the operand is evaluated under, the
        // same shape the compiler's own global-assembly
        // emission builds for it.
        let gid = rustc_middle::mir::interpret::GlobalId {
            instance: rustc_middle::ty::Instance::new_raw(
                def_id,
                rustc_middle::ty::GenericArgs::identity_for_item(tcx, def_id),
            ),
            promoted: None,
        };
        let evaluated =
            tcx.const_eval_global_id(fully_monomorphized(), gid, tcx.def_span(def_id));
        match evaluated {
            Ok(value) => self.chase_value(value),
            Err(rustc_middle::mir::interpret::ErrorHandled::Reported(..)) => {
                self.finding = Some(format!(
                    "constant operand of the `global_asm!` item `{}` \
                     failed to evaluate (rustc reports the evaluation \
                     error when it emits the global assembly)",
                    tcx.def_path_str(tcx.parent(def_id))
                ));
            }
            Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_)) => {}
        }
    }

    /// Instantiate a value read from `instance`'s body with the
    /// instance's arguments, normalizing projections away. The
    /// collector does this with the `try_` variant and simply
    /// does not collect an item that fails to normalize (the
    /// panicking variant reports an internal error instead), so
    /// a failure here yields `None` and the item is skipped.
    fn monomorphize<T>(&self, instance: rustc_middle::ty::Instance<'tcx>, value: T) -> Option<T>
    where
        T: rustc_middle::ty::TypeFoldable<rustc_middle::ty::TyCtxt<'tcx>>,
    {
        instance
            .try_instantiate_mir_and_normalize_erasing_regions(
                self.tcx,
                fully_monomorphized(),
                rustc_middle::ty::EarlyBinder::bind(self.tcx, value),
            )
            .ok()
    }

    /// The collector's missing-MIR verdict for an instance of
    /// a crate this driver built as an auxiliary (see the
    /// module note): `Some` when the collector's own predicate
    /// chain says the instance must be monomorphized locally
    /// (`should_codegen_locally` in rustc_monomorphize, up to
    /// its MIR-availability test) and the item's optimized MIR
    /// is conclusively one a native build of the auxiliary
    /// would not have encoded (the `should_encode_mir` rule in
    /// rustc_metadata without `-Zalways-encode-mir`). The
    /// finding is the walk's own; the session's availability
    /// answer is never consulted for the verdict itself,
    /// because the auxiliary's metadata always answers true.
    fn missing_mir_finding(
        &self,
        instance: rustc_middle::ty::Instance<'tcx>,
    ) -> Option<String> {
        let tcx = self.tcx;
        let def_id = instance.def.def_id_if_not_guaranteed_local_codegen()?;
        // The encoder's MIR group: functions, methods and
        // closures. Constructors are the collector's own
        // exception, and statics are never monomorphized from
        // an upstream crate (the chain below excludes both:
        // constructors are not in this kind set, and a static
        // is reachable-non-generic or foreign in every shape
        // that reaches here).
        if !matches!(
            tcx.def_kind(def_id),
            rustc_hir::def::DefKind::Fn
                | rustc_hir::def::DefKind::AssocFn
                | rustc_hir::def::DefKind::Closure
        ) {
            return None;
        }
        if tcx.is_foreign_item(def_id) || def_id.is_local() {
            return None;
        }
        // The collector links these instead of monomorphizing
        // them locally, so their MIR is never demanded.
        if tcx.is_reachable_non_generic(def_id)
            || instance.upstream_monomorphization(tcx).is_some()
        {
            return None;
        }
        // Only the crates this process built as auxiliaries
        // can carry the always-encode-mir lie; every other
        // crate's metadata answers truthfully, and a truthful
        // absence already fails through the collector hook in
        // the gate below.
        let metadata_only =
            aux_native_metadata_only(tcx.crate_name(def_id.krate).as_str())?;
        let native_encoded = if metadata_only {
            // A metadata-only native build encodes no
            // optimized MIR at all (the encoder's rule needs
            // an output code generation counts).
            false
        } else if !self.demanded_from_local.contains(&instance) {
            // Demanded only through upstream bodies: an item
            // in the home crate's reachable set is demanded
            // exactly that way (the reachability visitor
            // follows the bodies of the items it seeds), so
            // nothing here is conclusive.
            true
        } else if is_trait_impl_item(tcx, def_id) {
            // Impl items are seeded into the reachable set
            // with their trait; their encoding does not turn
            // on the module evidence below.
            true
        } else {
            !has_nonpublic_module_ancestor(tcx, def_id)
        };
        if native_encoded {
            return None;
        }
        use rustc_middle::ty::print::{with_no_trimmed_paths, with_resolve_crate_name};
        let key = with_resolve_crate_name!(with_no_trimmed_paths!(
            tcx.def_path_str_with_args(instance.def.def_id(), instance.args)
        ));
        Some(format!(
            "missing optimized MIR for `{key}` in the crate `{}` \
             (rustc reports this when the collector must \
             monomorphize an item whose optimized MIR its home \
             crate did not encode)",
            tcx.crate_name(def_id.krate),
        ))
    }

    fn process(&mut self, instance: rustc_middle::ty::Instance<'tcx>) {
        let tcx = self.tcx;
        self.processing_local = instance.def.def_id().is_local();
        // The collector's missing-MIR verdict for auxiliary
        // crates (see `missing_mir_finding`): it precedes the
        // codegen-locally gate below, whose own hook raises
        // the fatal only from truthful metadata, and an
        // auxiliary's metadata is not truthful.
        if self.finding.is_none() {
            if let Some(finding) = self.missing_mir_finding(instance) {
                self.finding = Some(finding);
                return;
            }
        }
        // The collector only walks instances it would codegen
        // locally: a foreign instance's constants were evaluated
        // when its home crate was compiled (or never, if it was
        // never collected there either), and its body may not be
        // available here at all.
        if !tcx.should_codegen_locally(instance) {
            return;
        }
        // The instance's display keys, shared by the verdicts
        // below: the instance's own key, and its definition's —
        // a generic definition's body is materialized once,
        // under the definition's key, and its diagnostics are
        // properties of every instance of it.
        use rustc_middle::ty::print::{with_no_trimmed_paths, with_resolve_crate_name};
        let key = with_resolve_crate_name!(with_no_trimmed_paths!(
            tcx.def_path_str_with_args(instance.def.def_id(), instance.args)
        ));
        let def_key = with_resolve_crate_name!(with_no_trimmed_paths!(
            tcx.def_path_str(instance.def.def_id())
        ));
        // The collector forces the compiler's per-instance
        // monomorphization checks for every instance it
        // collects, before scanning the instance's items
        // (`collect_items_of_instance` in rustc_monomorphize):
        // the `check_mono_item` query runs the ABI target-feature
        // rules and the `large_assignments` move check, and its
        // diagnostics are compilation errors there. Force the
        // same query here. The diagnostics are the compiler's
        // own, emitted into the session as the query runs, so a
        // rise in the diagnostic error count across the force is
        // the rejection rustc reports for this instance; a panic
        // inside the query abandons the force, never the walk
        // (the same discipline as the per-instance walk in
        // `run`). The finding is recorded separately from the
        // walk's own and reported last in the verdict chain:
        // the body checks of `check_bodies` name some of the
        // same rejections more precisely (the SIMD lane limit
        // is one), and their findings take precedence.
        if self.mono_finding.is_none() {
            let before = tcx.dcx().err_count();
            let forced = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tcx.ensure_ok().check_mono_item(instance);
            }));
            if forced.is_ok() && tcx.dcx().err_count() > before {
                self.mono_finding = Some(format!(
                    "monomorphization checks rejected `{key}` \
                     (see the diagnostics above)"
                ));
                // The diagnostics stay printed, but the error
                // count is reset (the discounting discipline of
                // the speculative body fetches in `convert`):
                // the verdict is the driver's to report through
                // the finding chain, not the session's.
                tcx.dcx().reset_err_count();
            } else if forced.is_err() && before == 0 && tcx.dcx().err_count() > 0 {
                // The force panicked, so it is abandoned — but a
                // `bug!` inside the query emits its ICE diagnostic
                // into the session before the panic unwinds, and
                // that diagnostic alone would fail the compilation
                // at the end of the run. The instance is one the
                // collector itself would never force (its types
                // do not normalize in the monomorphized
                // environment — a projection whose impl carries
                // an unprovable assumed bound, say), so the
                // abandoned force's diagnostics are discounted
                // the same way: with no errors counted before the
                // force, resetting restores exactly that state.
                // The force's compiler-bug witness is discounted
                // with its diagnostics only when the instance is
                // of the abandoned kind — its types do not
                // normalize in the monomorphized environment, so
                // the collector would never make this force. A
                // force abandoned on an instance whose types do
                // normalize is the collector's own force on a
                // collected instance, and the compiler dying
                // inside it stays a crash verdict in `run`.
                self.force_abandoned = !instance_types_normalize(tcx, instance);
                tcx.dcx().reset_err_count();
            }
            // A force that panicked may have died in its ABI
            // check, on the layout of a type the instance's
            // collection demands: the panic reaches the walk
            // with an opaque payload (the query system
            // delivers a layout cycle by resume, not as a
            // diagnostic the walk can read), so the verdict
            // is tied to the conversion's own record — when a
            // type among the instance's arguments or its
            // instantiated signature's types is one whose
            // speculative layout query failed with discounted
            // diagnostics (see `convert::guarded_layout`),
            // the failure is this compilation's. The verdict
            // is deferred like a force's crash (see
            // `pending_crash`): collection runs before the
            // ABI demands in the compiler, so a
            // collection-phase verdict further along the
            // walk pre-empts it. An instance whose types do
            // not normalize is one the collector never
            // forces (see the abandonment branch above), so
            // it makes no ABI demand to replay — and posing
            // its signature's queries is exactly what the
            // abandoned force died doing.
            if forced.is_err()
                && self.deferred_layout_finding.is_none()
                && instance_types_normalize(tcx, instance)
            {
                if let Some(ty) = demanded_layout_failure(tcx, instance) {
                    self.deferred_layout_finding = Some(format!(
                        "collecting `{key}` demands the layout of                          `{ty}`, whose computation failed (rustc                          reports the layout error for the instances                          it collects; see the diagnostics above)"
                    ));
                }
            }
        }
        // A body the collector collects is one rustc itself
        // materializes in its own pipeline, so the evaluation
        // diagnostics of materializing it are compilation errors
        // there too: the validity errors of its constant
        // operands are the shape this check adds (evaluation
        // errors proper are also found by the required-constant
        // evaluation below, and lint errors are recorded
        // separately, for every local body). The driver's speculative
        // body fetches discount those diagnostics at the fetch
        // (a body the collector never reaches must not fail the
        // compilation — Issue #417) but record the body's key;
        // a hit here turns the record into the verdict. Both the
        // instance's own key and its definition's are consulted:
        // a generic definition's body is materialized once,
        // under the definition's key, and its diagnostics are
        // properties of every instance of it.
        //
        // A compiler-bug panic recorded for the body's
        // materialization (see `convert::materialization_bug`)
        // is the stronger verdict of the same shape: the
        // compiler crashed materializing a body the collector
        // collects, which native reports as a crash, not as a
        // diagnostics finding.
        if let Some(record) = crate::convert::materialization_bug(&key)
            .or_else(|| crate::convert::materialization_bug(&def_key))
        {
            if crate::convert::captured_panic_is_const_expr_normalize(&record)
                && !key.contains('{')
                && !def_key.contains('{')
            {
                // A const-expr-normalize record for an instance
                // whose own generic arguments mention no const
                // expression is not a crash verdict (Issue
                // #510): the firing is an artifact of Stable
                // MIR materializing the body -- it eagerly
                // evaluates the type-level const expressions of
                // the body's where-clauses, which the native
                // pipeline never evaluates as values -- so the
                // check falls through to the diagnostics
                // registries below, which the fetch likewise
                // left untouched for this firing. An instance
                // whose own arguments do mention a const
                // expression (the `{ N + 1 }` trait argument of
                // the crash-materialize shape) is one the
                // native collector itself cannot collect
                // without normalizing that expression, and the
                // record keeps its crash verdict.
            } else {
                self.crash = Some(record);
                return;
            }
        }
        {
            if crate::convert::materialization_emitted(&key)
                || crate::convert::materialization_emitted(&def_key)
            {
                self.finding = Some(format!(
                    "materialization of `{key}` emitted diagnostics \
                     (rustc reports them for the bodies it collects; \
                     see the diagnostics above)"
                ));
                return;
            }
            // The same discipline for demand-path instance
            // resolution (see `convert::resolve_instance_guarded`),
            // with one difference: only the instance's own key
            // is consulted. A materialization diagnostic is a
            // property of the one generic body fetched per
            // definition, but a resolution's outcome depends on
            // the generic arguments attempted, so a record keyed
            // by the definition would condemn instantiations
            // that resolve cleanly.
            if crate::convert::resolution_emitted(&key) {
                self.finding = Some(format!(
                    "resolution of `{key}` emitted diagnostics \
                     (rustc reports them for the instances it collects; \
                     see the diagnostics above)"
                ));
                return;
            }
        }
        // Shim instances are handled by kind. Drop glue and
        // fn-pointer shim bodies are walked: a type's destructor
        // and field drops live in the former, and the latter's
        // body is what calls the pointee instance. The reify and
        // closure `call_once` shims are instead followed to the
        // instance they forward to — building a reify shim's
        // body for a closure is not even safe (it asks for the
        // closure's `fn_sig`, which the compiler refuses with
        // "use `args.as_closure().sig()`"), while the target
        // instance is the one whose constants the collector
        // evaluates. The remaining shims are synthetic thunks
        // whose targets the walk reaches by other edges (vtable
        // methods via unsizing casts) or not at all
        // (data-dependent callees).
        if let rustc_middle::ty::InstanceKind::Shim(shim) = instance.def {
            use rustc_middle::ty::ShimKind;
            match shim {
                ShimKind::DropGlue(..) | ShimKind::AsyncDropGlue(..) | ShimKind::FnPtr(..) => {}
                ShimKind::Reify(def_id, _) => {
                    self.enqueue_resolved(def_id, instance.args);
                    return;
                }
                ShimKind::ClosureOnce { closure, .. } => {
                    self.enqueue_resolved(closure, instance.args);
                    return;
                }
                _ => return,
            }
        }
        let body = tcx.instance_mir(instance.def);
        self.process_body(instance, body);
    }

    /// Resolve `(def_id, args)` the way the collector resolves a
    /// mentioned function item and queue the instance for the
    /// walk. Unresolvable or generic instances are not collected
    /// by the compiler either, so they are simply not queued.
    fn enqueue_resolved(
        &mut self,
        def_id: rustc_span::def_id::DefId,
        args: rustc_middle::ty::GenericArgsRef<'tcx>,
    ) {
        let tcx = self.tcx;
        let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
        let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rustc_middle::ty::Instance::try_resolve(tcx, typing_env, def_id, args)
        }));
        if let Ok(Ok(Some(target))) = resolved {
            self.enqueue(target);
        }
    }

    fn process_body(
        &mut self,
        instance: rustc_middle::ty::Instance<'tcx>,
        body: &rustc_middle::mir::Body<'tcx>,
    ) {
        use rustc_middle::mir::visit::Visitor as _;
        let tcx = self.tcx;
        // The collector evaluates every required constant of the
        // body it collects (the list is taken from the
        // pre-optimization body, so dead code is included).
        for const_op in body.required_consts() {
            let Some(mono_const) = self.monomorphize(instance, const_op.const_) else {
                continue;
            };
            // A required constant that measures a type's
            // layout (see `is_layout_measuring_const`) is the
            // collector itself demanding that layout: when
            // the conversion's speculative layout query for
            // the measured type failed and its diagnostics
            // were discounted (see `convert::guarded_layout`),
            // the demand makes the failure this compilation's
            // verdict, the way the evaluation error below
            // does for a constant that fails to evaluate. A
            // measured type nothing recorded stays silent:
            // the query may be one the reference compilation
            // never poses.
            if let rustc_middle::mir::Const::Unevaluated(uc, _) = &mono_const {
                if is_layout_measuring_const(tcx, uc.def) {
                    if let Some(ty) = recorded_layout_failure(uc.args) {
                        self.finding = Some(format!(
                            "a required constant of `{}` demands the layout of                              `{}`, whose computation failed (rustc reports                              the layout error for the instances it collects;                              see the diagnostics above)",
                            tcx.def_path_str(instance.def.def_id()),
                            ty
                        ));
                        return;
                    }
                }
            }
            match mono_const.eval(tcx, fully_monomorphized(), const_op.span) {
                Ok(value) => self.chase_value(value),
                Err(rustc_middle::mir::interpret::ErrorHandled::Reported(..)) => {
                    self.finding = Some(required_const_finding(tcx, instance));
                    return;
                }
                Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_)) => {}
            }
            if self.finding.is_some() {
                return;
            }
        }
        // The items the body mentions: their instances' required
        // constants are evaluated by the collector exactly like
        // this body's. Item bodies carry the mentions the compiler
        // recorded before optimization; shim bodies may carry
        // none, so there the mentions are recomputed with the
        // same visitor the compiler uses (`MentionScan`).
        let mut mentions: Vec<rustc_middle::mir::MentionedItem<'tcx>> = body
            .mentioned_items()
            .iter()
            .map(|spanned| spanned.node)
            .collect();
        if mentions.is_empty() {
            let mut scan = MentionScan {
                tcx,
                body,
                mentioned: Vec::new(),
            };
            scan.visit_body(body);
            mentions = scan.mentioned;
        }
        for item in mentions {
            let Some(item) = self.monomorphize(instance, item) else {
                continue;
            };
            // A mention's resolution goes through the
            // compiler's trait and layout machinery, which
            // has its own recursion ceilings: on a type
            // nested deeply enough (a runaway instantiation
            // chain's adapter types, say) resolving one
            // mention — the drop glue of a deeply-nested
            // type is the shape this was measured on —
            // panics where the shallower mentions resolved
            // cleanly. The panic abandons that mention,
            // never the body's remaining mentions: the
            // collector's descent along them is what
            // carries a runaway chain to the recursion-limit
            // verdict in `run`, and losing the whole body to
            // one mention's ceiling would strand the chain
            // below it. A compiler-bug panic among them is
            // still recorded by the hook and read by the
            // scoped witness diff in `run`, exactly as if it
            // had unwound to there.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.visit_mentioned(item);
            }));
            if self.finding.is_some() {
                return;
            }
        }
        // Constant operands of the reachable blocks: the scan
        // evaluates each one and chases the value into the
        // allocations it points at (which can hold further
        // instances and statics). A function item or closure
        // used as a plain value is not collected from here; it
        // is collected through the mentioned items above where
        // it is actually used as a function.
        let mut scan = OperandScan {
            walk: self,
            instance,
        };
        for (bb, data) in rustc_middle::mir::traversal::mono_reachable(body, tcx, instance) {
            scan.visit_basic_block_data(bb, data);
            if scan.walk.finding.is_some() {
                return;
            }
        }
    }
    fn visit_mentioned(&mut self, item: rustc_middle::mir::MentionedItem<'tcx>) {
        use rustc_middle::mir::MentionedItem;
        let tcx = self.tcx;
        match item {
            MentionedItem::Fn(ty) => {
                if let rustc_middle::ty::FnDef(def_id, args) = *ty.kind() {
                    if let Some(args) = args.no_bound_vars() {
                        if let Ok(Some(instance)) = rustc_middle::ty::Instance::try_resolve(
                            tcx,
                            fully_monomorphized(),
                            def_id,
                            args,
                        ) {
                            self.enqueue(instance);
                        }
                    }
                }
            }
            MentionedItem::Drop(ty) => {
                let instance = rustc_middle::ty::Instance::resolve_drop_glue(tcx, ty);
                self.enqueue(instance);
            }
            MentionedItem::Closure(ty) => {
                if let rustc_middle::ty::Closure(def_id, args) = *ty.kind() {
                    let instance = rustc_middle::ty::Instance::resolve_closure(
                        tcx,
                        def_id,
                        args,
                        rustc_middle::ty::ClosureKind::FnOnce,
                    );
                    self.enqueue(instance);
                }
            }
            MentionedItem::UnsizeCast {
                source_ty,
                target_ty,
            } => {
                // Only the direct shape is chased: a concrete type
                // unsized to a trait object, whose vtable's method
                // instances the collector collects. The cast's
                // operand types are the pointer types (`&Vec<T>`
                // unsizing to `&dyn Trait`); the tails the vtable
                // is formed for are found by traversing the two
                // operand types together, the way the collector's
                // tail search does (see `find_unsize_tails`). (A
                // vtable formed through the tail of a custom
                // unsizing struct is reached here through the
                // constant value carrying it whenever one was
                // evaluated, and the remaining shapes only narrow
                // the check.)
                let (source_ty, target_ty) = find_unsize_tails(tcx, source_ty, target_ty);
                if target_ty.is_trait() && !source_ty.is_trait() {
                    if let rustc_middle::ty::Dynamic(data, _) = target_ty.kind() {
                        if let Some(principal) = data.principal() {
                            let principal = tcx.instantiate_bound_regions_with_erased(principal);
                            let alloc_id = tcx.vtable_allocation((source_ty, Some(principal)));
                            self.chase_alloc(alloc_id);
                        }
                    }
                }
            }
        }
    }

    /// Follow a constant value into the allocation it points at:
    /// the collector collects the function instances and statics
    /// inside evaluated constants (a function pointer constant
    /// whose target body holds a failing required constant is the
    /// `collect-in-dead-fnptr-in-const` shape).
    fn chase_value(&mut self, value: rustc_middle::mir::ConstValue) {
        use rustc_middle::mir::interpret::Scalar;
        match value {
            rustc_middle::mir::ConstValue::Scalar(Scalar::Ptr(ptr, _)) => {
                self.chase_alloc(ptr.provenance.alloc_id());
            }
            rustc_middle::mir::ConstValue::Indirect { alloc_id, .. }
            | rustc_middle::mir::ConstValue::Slice { alloc_id, .. } => {
                self.chase_alloc(alloc_id);
            }
            _ => {}
        }
    }

    fn chase_alloc(&mut self, alloc_id: rustc_middle::mir::interpret::AllocId) {
        if self.finding.is_some() || !self.visited_allocs.insert(alloc_id) {
            return;
        }
        use rustc_middle::mir::interpret::GlobalAlloc;
        match self.tcx.global_alloc(alloc_id) {
            GlobalAlloc::Function { instance, .. } => self.enqueue(instance),
            GlobalAlloc::Static(def_id) => {
                // A static the collector reaches is code-generated,
                // which evaluates its initializer; an evaluation
                // error there is the same class of failure. A
                // foreign static (an `extern` block item) is never
                // collected and has no initializer to evaluate
                // (asking for one is an internal compiler error),
                // so it is left alone, as are thread-local statics,
                // whose initialization is lazy and likewise not
                // evaluated by the collector.
                if self.tcx.is_foreign_item(def_id) || self.tcx.is_thread_local_static(def_id) {
                    return;
                }
                match self.tcx.eval_static_initializer(def_id) {
                    Ok(alloc) => self.chase_allocation(alloc),
                    Err(rustc_middle::mir::interpret::ErrorHandled::Reported(..)) => {
                        self.finding = Some(format!(
                            "initializer of static `{}` failed to evaluate \
                             (rustc reports the evaluation error during \
                             monomorphization collection)",
                            self.tcx.def_path_str(def_id)
                        ));
                    }
                    Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_)) => {}
                }
            }
            // A type-id allocation holds no instances or statics.
            GlobalAlloc::TypeId { .. } => {}
            GlobalAlloc::Memory(alloc) => self.chase_allocation(alloc),
            GlobalAlloc::VTable(ty, dyn_ty) => {
                // Collect the vtable's method instances the way
                // the collector does for a vtable allocation
                // (`create_mono_items_for_vtable_methods`): the
                // entries of the principal trait, plus the drop
                // glue when the type needs dropping. (Chasing the
                // vtable allocation itself cannot reach them: the
                // allocation id is the one this very call is
                // visiting, so the visited set would turn the
                // chase away.)
                if let Some(principal) = dyn_ty.principal() {
                    let principal = self.tcx.instantiate_bound_regions_with_erased(principal);
                    let trait_ref = principal.with_self_ty(self.tcx, ty);
                    for entry in self.tcx.vtable_entries(trait_ref) {
                        if let rustc_middle::ty::VtblEntry::Method(instance) = entry {
                            self.enqueue(*instance);
                        }
                    }
                    if ty.needs_drop(self.tcx, fully_monomorphized()) {
                        let glue = rustc_middle::ty::Instance::resolve_drop_glue(self.tcx, ty);
                        self.enqueue(glue);
                    }
                }
            }
        }
    }

    fn chase_allocation(&mut self, alloc: rustc_middle::mir::interpret::ConstAllocation<'tcx>) {
        let ptrs = alloc.inner().provenance().ptrs();
        for &prov in ptrs.values() {
            self.chase_alloc(prov.alloc_id());
            if self.finding.is_some() {
                return;
            }
        }
    }
}

/// Find the tails of an unsizing cast's operand types the way the
/// collector's `find_tails_for_unsizing` does: the source and the
/// target are traversed together, never independently. One pointer
/// layer (a reference, a raw pointer or a box) is peeled from both
/// sides at once, and from there the two types descend in lockstep
/// (see `unsize_lockstep_tails`) for as long as they keep the same
/// shape, stopping when they diverge — in particular when the
/// target reaches the trait object. So `&&W<T>` unsizing to
/// `&dyn Trait` forms the vtable for `&W<T>`: the traversal stops
/// with the target at `dyn Trait` after peeling one layer.
/// Peeling each side on its own would strip both source
/// references and select the implementation for `W<T>` instead.
fn find_unsize_tails<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    source_ty: rustc_middle::ty::Ty<'tcx>,
    target_ty: rustc_middle::ty::Ty<'tcx>,
) -> (rustc_middle::ty::Ty<'tcx>, rustc_middle::ty::Ty<'tcx>) {
    let pointees = match (source_ty.kind(), target_ty.kind()) {
        (
            &rustc_middle::ty::Ref(_, source_pointee, _),
            &rustc_middle::ty::Ref(_, target_pointee, _)
            | &rustc_middle::ty::RawPtr(target_pointee, _),
        )
        | (
            &rustc_middle::ty::RawPtr(source_pointee, _),
            &rustc_middle::ty::RawPtr(target_pointee, _),
        ) => Some((source_pointee, target_pointee)),
        _ => match (source_ty.boxed_ty(), target_ty.boxed_ty()) {
            (Some(source_boxed), Some(target_boxed)) => Some((source_boxed, target_boxed)),
            _ => None,
        },
    };
    match pointees {
        Some((source, target)) => unsize_lockstep_tails(tcx, source, target),
        None => (source_ty, target_ty),
    }
}

/// The lockstep half of the tail search (the collector's
/// `struct_lockstep_tails_for_codegen`): descend both types
/// through the tail field of a struct both sides instantiate, or
/// through the last element of equal-length tuples, for as long
/// as they keep the same shape.
fn unsize_lockstep_tails<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    mut source: rustc_middle::ty::Ty<'tcx>,
    mut target: rustc_middle::ty::Ty<'tcx>,
) -> (rustc_middle::ty::Ty<'tcx>, rustc_middle::ty::Ty<'tcx>) {
    loop {
        match (source.kind(), target.kind()) {
            (
                &rustc_middle::ty::Adt(source_def, source_args),
                &rustc_middle::ty::Adt(target_def, target_args),
            ) if source_def == target_def && source_def.is_struct() => {
                let Some(field) = source_def.non_enum_variant().tail_opt() else {
                    break;
                };
                source = tcx
                    .normalize_erasing_regions(fully_monomorphized(), field.ty(tcx, source_args));
                target = tcx
                    .normalize_erasing_regions(fully_monomorphized(), field.ty(tcx, target_args));
            }
            (&rustc_middle::ty::Tuple(source_tys), &rustc_middle::ty::Tuple(target_tys))
                if source_tys.len() == target_tys.len() =>
            {
                let (Some(&source_last), Some(&target_last)) =
                    (source_tys.last(), target_tys.last())
                else {
                    break;
                };
                source = source_last;
                target = target_last;
            }
            _ => break,
        }
    }
    (source, target)
}

/// Whether `def` is one of the `SizedTypeProperties`
/// associated constants whose evaluation is defined by the
/// measured type's layout — `SIZE`, `ALIGN`, `ALIGNMENT`
/// and `IS_ZST` in `core::mem` (re-exported through
/// `std::mem`): a required constant of this shape in a
/// collected body is the collector demanding the layout of
/// the constant's self type, the demand the native
/// compilation answers with the layout query itself.
fn is_layout_measuring_const(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    def: rustc_span::def_id::DefId,
) -> bool {
    let path = tcx.def_path_str(def);
    (path.starts_with("core::mem::SizedTypeProperties::")
        || path.starts_with("std::mem::SizedTypeProperties::"))
        && ["::SIZE", "::ALIGN", "::ALIGNMENT", "::IS_ZST"]
            .iter()
            .any(|name| path.ends_with(name))
}

/// A type search for the first type whose speculative
/// layout query failed during conversion with discounted
/// diagnostics (see `convert::layout_failure_recorded`).
struct RecordedLayoutFailure {
    hit: Option<rustc_middle::ty::Ty<'static>>,
}

impl<'tcx> rustc_middle::ty::TypeVisitor<rustc_middle::ty::TyCtxt<'tcx>>
    for RecordedLayoutFailure
{
    type Result = std::ops::ControlFlow<()>;

    fn visit_ty(
        &mut self,
        ty: rustc_middle::ty::Ty<'tcx>,
    ) -> Self::Result {
        if crate::convert::layout_failure_recorded(ty) {
            // The registry's types live in the running
            // compilation's context, which outlives the
            // walk (the same erasure as the registry's own).
            self.hit = Some(unsafe {
                std::mem::transmute::<
                    rustc_middle::ty::Ty<'tcx>,
                    rustc_middle::ty::Ty<'static>,
                >(ty)
            });
            return std::ops::ControlFlow::Break(());
        }
        use rustc_middle::ty::TypeSuperVisitable as _;
        ty.super_visit_with(self)
    }
}

/// The first recorded layout failure among the types
/// `value` mentions, rendered for a finding, if any.
fn recorded_layout_failure<'tcx, V>(value: V) -> Option<String>
where
    V: rustc_middle::ty::TypeVisitable<rustc_middle::ty::TyCtxt<'tcx>>,
{
    let mut search = RecordedLayoutFailure { hit: None };
    let _ = value.visit_with(&mut search);
    search.hit.map(|ty| {
        use rustc_middle::ty::print::{with_no_trimmed_paths, with_resolve_crate_name};
        with_resolve_crate_name!(with_no_trimmed_paths!(format!("{ty}")))
    })
}

/// The recorded layout failure a collected instance's own
/// collection demands, if any: a failure among the types of
/// the instance's generic arguments (the demanded type
/// itself for a drop glue) or of its instantiated
/// signature — the types the per-instance force's ABI
/// check lays out. The signature is read under containment:
/// an instance whose signature the queries refuse (a shim
/// shape) simply has no signature demand to replay.
fn demanded_layout_failure<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    instance: rustc_middle::ty::Instance<'tcx>,
) -> Option<String> {
    if let Some(hit) = recorded_layout_failure(instance.args) {
        return Some(hit);
    }
    if !matches!(instance.def, rustc_middle::ty::InstanceKind::Item(_)) {
        return None;
    }
    let sig = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let sig = instance
            .ty(tcx, fully_monomorphized())
            .fn_sig(tcx);
        let sig = tcx.instantiate_bound_regions_with_erased(sig);
        tcx.try_normalize_erasing_regions(
            fully_monomorphized(),
            rustc_middle::ty::Unnormalized::new_wip(sig),
        )
        .ok()
    }));
    let Ok(Some(sig)) = sig else {
        return None;
    };
    recorded_layout_failure(sig)
}

fn required_const_finding(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    instance: rustc_middle::ty::Instance<'_>,
) -> String {
    format!(
        "required constant of `{}` failed to evaluate \
         (rustc reports the evaluation error during \
         monomorphization collection)",
        tcx.def_path_str(instance.def.def_id())
    )
}

/// Recompute a body's mentioned items the way the compiler's
/// `MentionedItems` pass does (rustc_mir_transform's
/// `MentionedItemsVisitor`, kept in sync with the mono collector's
/// own visitor): calls and tail calls, drops, inline-asm function
/// symbols, unsizing casts that may form a vtable, and closures or
/// function items coerced to function pointers. The pass stores
/// its result on item bodies; shim bodies may carry no recording,
/// so for those the recording is reproduced here from the same
/// input the pass would read.
/// Whether `def_id` is a trait's own associated item or an
/// item of a trait impl: those are seeded into a crate's
/// reachable set with their trait, so the module evidence of
/// `missing_mir_finding` does not decide their encoding.
fn is_trait_impl_item(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    def_id: rustc_span::def_id::DefId,
) -> bool {
    if !matches!(tcx.def_kind(def_id), rustc_hir::def::DefKind::AssocFn) {
        return false;
    }
    matches!(
        tcx.associated_item(def_id).container,
        rustc_middle::ty::AssocContainer::Trait
            | rustc_middle::ty::AssocContainer::TraitImpl(_)
    )
}

/// Whether some module on `def_id`'s ancestor chain in its
/// home crate is not public: the item then cannot be named
/// from outside the crate through any path, so only a mention
/// from a reachable body or signature could have put it in
/// the reachable set the metadata encoder consults.
fn has_nonpublic_module_ancestor(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    def_id: rustc_span::def_id::DefId,
) -> bool {
    let mut cur = def_id;
    while let Some(parent) = tcx.opt_parent(cur) {
        if matches!(tcx.def_kind(parent), rustc_hir::def::DefKind::Mod)
            && !matches!(
                tcx.visibility(parent),
                rustc_middle::ty::Visibility::Public
            )
        {
            return true;
        }
        cur = parent;
    }
    false
}

struct MentionScan<'a, 'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    body: &'a rustc_middle::mir::Body<'tcx>,
    mentioned: Vec<rustc_middle::mir::MentionedItem<'tcx>>,
}

impl<'tcx> rustc_middle::mir::visit::Visitor<'tcx> for MentionScan<'_, 'tcx> {
    fn visit_terminator(
        &mut self,
        terminator: &rustc_middle::mir::Terminator<'tcx>,
        location: rustc_middle::mir::Location,
    ) {
        self.super_terminator(terminator, location);
        use rustc_middle::mir::MentionedItem;
        match &terminator.kind {
            rustc_middle::mir::TerminatorKind::Call { func, .. }
            | rustc_middle::mir::TerminatorKind::TailCall { func, .. } => {
                self.mentioned
                    .push(MentionedItem::Fn(func.ty(self.body, self.tcx)));
            }
            rustc_middle::mir::TerminatorKind::Drop { place, .. } => {
                self.mentioned
                    .push(MentionedItem::Drop(place.ty(self.body, self.tcx).ty));
            }
            rustc_middle::mir::TerminatorKind::InlineAsm { operands, .. } => {
                for op in operands {
                    if let rustc_middle::mir::InlineAsmOperand::SymFn { value } = op {
                        self.mentioned.push(MentionedItem::Fn(value.const_.ty()));
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_rvalue(
        &mut self,
        rvalue: &rustc_middle::mir::Rvalue<'tcx>,
        location: rustc_middle::mir::Location,
    ) {
        self.super_rvalue(rvalue, location);
        use rustc_middle::mir::MentionedItem;
        use rustc_middle::ty::adjustment::PointerCoercion;
        match *rvalue {
            rustc_middle::mir::Rvalue::Cast(
                rustc_middle::mir::CastKind::PointerCoercion(PointerCoercion::Unsize, _),
                ref operand,
                target_ty,
            ) => {
                // Detect unsizing casts that require vtables
                // (mirrors the pass: array-to-str/slice unsizing
                // is the one shape that never does).
                let source_ty = operand.ty(self.body, self.tcx);
                let may_involve_vtable = match (
                    source_ty.builtin_deref(true).map(|t| t.kind()),
                    target_ty.builtin_deref(true).map(|t| t.kind()),
                ) {
                    (
                        Some(rustc_middle::ty::Array(..)),
                        Some(rustc_middle::ty::Str | rustc_middle::ty::Slice(..)),
                    ) => false,
                    _ => true,
                };
                if may_involve_vtable {
                    self.mentioned.push(MentionedItem::UnsizeCast {
                        source_ty,
                        target_ty,
                    });
                }
            }
            rustc_middle::mir::Rvalue::Cast(
                rustc_middle::mir::CastKind::PointerCoercion(
                    PointerCoercion::ClosureFnPointer(_),
                    _,
                ),
                ref operand,
                _,
            ) => {
                self.mentioned
                    .push(MentionedItem::Closure(operand.ty(self.body, self.tcx)));
            }
            rustc_middle::mir::Rvalue::Cast(
                rustc_middle::mir::CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer(_), _),
                ref operand,
                _,
            ) => {
                self.mentioned
                    .push(MentionedItem::Fn(operand.ty(self.body, self.tcx)));
            }
            _ => {}
        }
    }
}

/// The constant-operand scan of one visited instance's reachable
/// blocks (see `RequiredConstWalk::process_body`).
struct OperandScan<'a, 'tcx> {
    walk: &'a mut RequiredConstWalk<'tcx>,
    instance: rustc_middle::ty::Instance<'tcx>,
}

impl<'tcx> rustc_middle::mir::visit::Visitor<'tcx> for OperandScan<'_, 'tcx> {
    fn visit_const_operand(
        &mut self,
        const_op: &rustc_middle::mir::ConstOperand<'tcx>,
        _location: rustc_middle::mir::Location,
    ) {
        let tcx = self.walk.tcx;
        let Some(mono_const) = self.walk.monomorphize(self.instance, const_op.const_) else {
            return;
        };
        // A bare function-item or closure operand is NOT
        // collected: both are plain values (a function item is
        // zero-sized, and so is a captureless closure), and
        // merely using one as a value (passing it to a generic
        // function that never calls it, storing it, black-boxing
        // it) does not make the collector codegen its body, so
        // resolving it here would demand bodies rustc never
        // collects and fail compilations rustc accepts. The
        // instances behind them are collected where the value
        // is actually used as a function: a call is a mentioned
        // item (a closure call names the `Fn`/`FnMut`/`FnOnce`
        // trait method, whose resolution reaches the closure
        // instance), a coercion to a function pointer is a
        // mentioned item too (both handled in
        // `visit_mentioned`), and an evaluated function-pointer
        // value points into a function allocation, which the
        // evaluation below chases.
        if self.walk.finding.is_some() {
            return;
        }
        match mono_const.eval(tcx, fully_monomorphized(), const_op.span) {
            Ok(value) => self.walk.chase_value(value),
            Err(rustc_middle::mir::interpret::ErrorHandled::Reported(..)) => {
                self.walk.finding = Some(required_const_finding(tcx, self.instance));
            }
            Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_)) => {}
        }
    }
}
