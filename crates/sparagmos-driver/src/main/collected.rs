//! The collected program: the conversion result stashed by the compiler callback for the driver to execute.

use super::*;

pub(crate) struct Collected {
    pub(crate) report: ConversionReport,
    pub(crate) bodies: Vec<(String, Body)>,
    /// `DefId` of each entry of `bodies` (same order), as assigned by
    /// the converter; lets the interpreter resolve `FnDef` call targets
    /// to the converted functions.
    pub(crate) body_def_ids: Vec<sparagmos_mir::DefId>,
    /// Definition name of each entry of `bodies` (same order). For
    /// local items this equals the item name; for collected foreign
    /// instances it is the definition name even when the display
    /// name carries an instance suffix.
    pub(crate) body_def_names: Vec<String>,
    /// Declared ABIs of the collected function instances, keyed
    /// by the `DefId` their bodies carry (see
    /// `Converter::fn_abis`).
    pub(crate) fn_abis: std::collections::HashMap<sparagmos_mir::DefId, sparagmos_mir::Abi>,
    pub(crate) def_names: Vec<String>,
    /// Sparagmos ids of the definitions the compiler classifies
    /// as compiler intrinsics (see `Converter::intrinsic_defs`).
    pub(crate) intrinsic_defs: std::collections::HashSet<sparagmos_mir::DefId>,
    /// Sparagmos ids of the coroutine-closure (async closure)
    /// definitions conversion met (see
    /// `Converter::async_closure_defs`).
    pub(crate) async_closure_defs: std::collections::HashSet<sparagmos_mir::DefId>,
    /// Vtable identities minted while decoding constants (see
    /// `Converter::vtable_identities`).
    pub(crate) vtable_identities:
        std::collections::HashMap<u64, (sparagmos_mir::Ty, Vec<String>, (u64, u64))>,
    /// Discriminant values of the converted enum ADTs, for the
    /// interpreter (see `Converter::adt_discriminants`).
    pub(crate) adt_discriminants: sparagmos_mir::AdtDiscriminants,
    /// Layouts of the converted monomorphic ADT instances, for the
    /// interpreter (see `Converter::adt_layouts`).
    pub(crate) adt_layouts: sparagmos_mir::AdtLayouts,
    /// Runtime layout descriptors of the converted
    /// dynamically sized ADT instances, for the interpreter
    /// (see `Converter::adt_dst_layouts`).
    pub(crate) adt_dst_layouts: sparagmos_mir::AdtDstLayouts,
    /// Per-variant field byte layouts of the converted
    /// monomorphic ADT instances, for the interpreter (see
    /// `Converter::adt_field_layouts`).
    pub(crate) adt_field_layouts: sparagmos_mir::AdtFieldLayouts,
    /// Storage layouts of the converted monomorphic tuple
    /// types, for the interpreter (see
    /// `Converter::tuple_layouts`).
    pub(crate) tuple_layouts: sparagmos_mir::TupleLayouts,
    /// Direct discriminant tag layouts of the converted
    /// monomorphic enum instances, for the interpreter (see
    /// `Converter::adt_tag_layouts`).
    pub(crate) adt_tag_layouts: sparagmos_mir::AdtTagLayouts,
    /// Storage-slot maps of the converted monomorphic coroutine
    /// instances, for the interpreter (see
    /// `Converter::coroutine_layouts`).
    pub(crate) coroutine_layouts: sparagmos_mir::CoroutineLayouts,
    /// Declared shapes of the converted ADTs, for the interpreter
    /// (see `Converter::adt_shapes`).
    pub(crate) adt_shapes: sparagmos_mir::AdtShapes,
    /// Derived-`Debug` shapes of the converted monomorphic ADT
    /// instances, for the interpreter (see
    /// `Converter::adt_debug_shapes`).
    pub(crate) adt_debug_shapes: sparagmos_mir::AdtDebugShapes,
    /// Debugger shapes of the converted monomorphic closure and
    /// coroutine instances, for the interpreter (see
    /// `Converter::env_debug_shapes`).
    pub(crate) env_debug_shapes: sparagmos_mir::EnvDebugShapes,
    /// Thread-local statics of the converted program, for the
    /// interpreter (see `Converter::thread_locals`).
    pub(crate) thread_locals: sparagmos_mir::ThreadLocals,
    /// Seeds of statics no converted body references, for
    /// the interpreter (see `Converter::extra_static_seeds`).
    pub(crate) extra_static_seeds: sparagmos_mir::ExtraStaticSeeds,
    /// Declared types of the converted program's statics, for
    /// the interpreter (see `Converter::static_tys_mir`).
    pub(crate) static_tys: sparagmos_mir::StaticTys,
    /// Field types of monomorphic ADT instances whose generic
    /// shape mentions a type alias, for the interpreter
    /// (see `Converter::adt_instance_shapes`).
    pub(crate) adt_instance_shapes: sparagmos_mir::AdtInstanceShapes,
    pub(crate) adt_niches: sparagmos_mir::AdtNiches,
    /// Materialisation plans for pointer-cast pointee types,
    /// for the interpreter (see `Converter::mat_plans`).
    pub(crate) mat_plans: std::collections::HashMap<String, sparagmos_interp::ir::MatPlan>,
    /// ADT instantiations whose drop glue the compiler certifies
    /// empty, for the interpreter (see `certify_drop_trivial`).
    pub(crate) drop_trivial_adts: std::collections::HashSet<String>,
    pub(crate) drop_no_impl_adts: std::collections::HashSet<String>,
    /// The compiler's `needs_drop` verdict for every type a
    /// collected body's `core::mem::needs_drop` call asks
    /// about, for the interpreter (see `certify_needs_drop`).
    pub(crate) needs_drop_verdicts: std::collections::HashMap<String, bool>,
    pub(crate) dtor_bodies: std::collections::HashMap<String, String>,
    /// Number of leading entries of `bodies` that come from the main
    /// crate itself (the rest are auxiliary-crate functions). The
    /// entry point is only ever searched among these: an auxiliary
    /// crate may define its own `main`-named function, which is not
    /// the program's entry point.
    pub(crate) local_body_count: usize,
    /// Number of leading entries of `report.functions` that belong
    /// to the target crate itself (the rest are auxiliary-crate and
    /// collected library functions). Compile verdicts judge
    /// conversion completeness on the target crate, the way
    /// compiletest does: the library is precompiled, so a collected
    /// library body that stays unconverted does not make the
    /// target crate's compilation incomplete.
    pub(crate) local_report_count: usize,
    /// A monomorphization failure rustc would have reported for
    /// this program (see `sparagmos_driver::mono_check`), detected
    /// after conversion. When set, the compilation as a whole has
    /// failed even though every converted body translated cleanly.
    pub(crate) compile_error: Option<String>,
    /// The entry point rustc designated for this crate, as a
    /// Sparagmos `DefId` into `body_def_ids`, when the crate has
    /// one. In a `--test` crate this is the generated harness
    /// main, which shares the definition name `<crate>::main`
    /// with the user's own `main`; run mode enters through this
    /// id, never through the name alone.
    pub(crate) entry_def_id: Option<sparagmos_mir::DefId>,
    /// The body name of the crate graph's `panic_handler`
    /// lang item, when the graph defines one and its body was
    /// collected (see `MirProgram::panic_handler`).
    pub(crate) panic_handler: Option<String>,
    /// Whether the crate graph links `std` (see
    /// `MirProgram::links_std`).
    pub(crate) links_std: bool,
    /// Definition-location stamps of the program's
    /// `#[track_caller]` instances (see
    /// `Converter::track_caller_defs`), for the interpreter.
    pub(crate) track_caller_defs:
        std::collections::HashMap<sparagmos_mir::DefId, sparagmos_mir::ConstValue>,
    /// The `Fn` / `FnMut` / `FnOnce` vtable-shim location
    /// stamps (see `Converter::fn_shim_locations`), for the
    /// interpreter.
    pub(crate) fn_shim_locations: [Option<sparagmos_mir::ConstValue>; 3],
    /// The `Coroutine::resume` vtable shim's location stamp
    /// (see `Converter::coroutine_shim_location`), for the
    /// program build.
    pub(crate) coroutine_shim_location: Option<sparagmos_mir::ConstValue>,
    /// Plain definition paths of the program's
    /// `#[track_caller]` coroutines (see
    /// `Converter::track_caller_coroutines`), for the
    /// program build.
    pub(crate) track_caller_coroutines: Vec<String>,
    /// Link symbols of the program's instances (see
    /// `Converter::link_symbols`), for the interpreter.
    pub(crate) link_symbols: std::collections::HashMap<sparagmos_mir::DefId, String>,
    /// The source-file table the bodies' source lines index into
    /// (see `Converter::source_files`), for the interpreter's
    /// debugger interface.
    pub(crate) source_files: Vec<String>,
    /// The crate's coverage mapping table, captured when the
    /// compilation was instrumented for coverage (see
    /// `sparagmos_driver::coverage`); empty otherwise.
    pub(crate) coverage: sparagmos_mir::CoverageMapping,
    /// Debugger display names of the resolved function
    /// instances (see
    /// `sparagmos_driver::convert::drain_fn_display_names`),
    /// for the interpreter's program. Empty outside debug mode.
    pub(crate) fn_display_names: Vec<(String, String, String)>,
    /// Resolved overflow-checks settings of the separately
    /// compiled auxiliary crates, keyed by crate name, as
    /// recorded by `main` from each crate's own build (see
    /// `AUX_OVERFLOW_CHECKS`), for the interpreter's program.
    pub(crate) crate_overflow_checks: Vec<(String, bool)>,
}

static COLLECTED: OnceLock<Mutex<Option<Collected>>> = OnceLock::new();

/// Names of the auxiliary crates (--aux) whose functions `collect`
/// must convert into the program, in `--aux` order. Set by `main`
/// before the compiler run; empty for single-file inputs.
pub(crate) static AUX_CRATES: OnceLock<Vec<String>> = OnceLock::new();

pub(crate) fn aux_crates() -> &'static [String] {
    AUX_CRATES.get().map(|v| v.as_slice()).unwrap_or(&[])
}

/// Resolved overflow-checks settings of the collected
/// auxiliary crates (--aux), keyed by crate name: the value
/// each crate's own build flags resolve to, the way
/// `session_overflow_checks` resolves the main compilation's.
/// Set by `main` before the compiler run, alongside
/// `AUX_CRATES`; empty for single-file inputs. The interpreter
/// resolves a shared library body's effective setting from
/// the call stack against this table (see
/// `ir::Program::crate_overflow_checks`).
pub(crate) static AUX_OVERFLOW_CHECKS: OnceLock<Vec<(String, bool)>> =
    OnceLock::new();

pub(crate) fn aux_overflow_checks() -> &'static [(String, bool)] {
    AUX_OVERFLOW_CHECKS
        .get()
        .map(|v| v.as_slice())
        .unwrap_or(&[])
}

pub(crate) fn stash(c: Collected) {
    let cell = COLLECTED.get_or_init(|| Mutex::new(None));
    *cell.lock().unwrap() = Some(c);
}

pub(crate) fn take() -> Option<Collected> {
    COLLECTED.get()?.lock().unwrap().take()
}
