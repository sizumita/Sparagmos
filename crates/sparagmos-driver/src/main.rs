//! `sparagmos-driver` — the Sparagmos rustc driver.
//!
//! Usage:
//! - `sparagmos-driver <file.rs> [--dump <path>] [--edition <ed>]`
//!   `[--rustc-arg <arg>]...` — convert and print a per-function
//!   summary.
//! - `sparagmos-driver run <file.rs> [--edition <ed>]`
//!   `[--rustc-arg <arg>]...` — convert and execute the program's
//!   `main` in the Sparagmos interpreter (`sparagmos-interp`), in
//!   this process.
//! - `sparagmos-driver run-jit <file.rs> [--edition <ed>]`
//!   `[--rustc-arg <arg>]...` — convert and execute the program's
//!   `main` with the Sparagmos JIT (`sparagmos-jit`), in this
//!   process. This is the JIT counterpart of `run`, on the same
//!   main path (same conversion, same collected program); only the
//!   execution engine differs. Exit codes follow run mode: 0 = the
//!   guest terminated normally, 101 = the guest panicked, 2 = the
//!   program could not be converted or JIT-compiled/executed
//!   (concrete reason on stderr, prefixed
//!   `sparagmos-driver run-jit:`). The JIT execution entry takes
//!   no guest arguments and no modelled `argc`/`argv`, and an
//!   entry returning `std::process::ExitCode` is reported as
//!   unsupported rather than guessed: those shapes are outside
//!   the JIT harness's entry contract (see
//!   `sparagmos-harness`'s `JitRunner`, which this mode mirrors).
//! - `sparagmos-driver jit-compile <file.rs> [--edition <ed>]`
//!   `[--rustc-arg <arg>]...` — convert exactly as convert mode
//!   does, print the same conversion report, and then attempt to
//!   JIT-compile (lower) the whole converted program with
//!   `sparagmos-jit`. The report gains one line: `jit-compiled:
//!   yes` when the lowering succeeded, or `jit-compiled: no`
//!   plus a `sparagmos-driver jit-compile: unsupported: <reason>`
//!   line on stderr when it did not. Exit codes: 0 = compilation
//!   succeeded and JIT lowering succeeded, 2 = compilation
//!   succeeded but JIT lowering is unsupported, 1 = the
//!   compilation failed (as convert mode), 101 = the compiler
//!   crashed (as convert mode).
//!
//! Each `--rustc-arg` forwards one argument verbatim to the embedded
//! rustc invocation, appended after this driver's own flags — the
//! position compiletest gives a test's `//@ compile-flags`, so a
//! caller measuring the rustc test suite can honour per-test flags
//! (and per-revision `--cfg`s) without a second invocation path.
//!
//! Both modes also accept `--aux <name>=<file.rs>` (repeatable): an
//! auxiliary source file that is compiled as a library crate named
//! `<name>` with the pinned rustc and linked into the main
//! compilation with `--extern`, the way compiletest's
//! `//@ aux-build:` / `//@ aux-crate:` directives do. Two optional
//! per-crate attributes may follow the path, separated by `;`:
//! `;crate-type=<type>` builds the crate as that crate type (the
//! runner passes `proc-macro` for compiletest's `//@ proc-macro:`
//! auxiliaries, whose sources do not declare a crate type), and
//! `;edition=<ed>` builds it with that edition instead of the main
//! `--edition` (compiletest builds each auxiliary crate with the
//! edition its own `//@ edition:` directive declares). An in-source
//! `#![crate_type]` declaration still decides the crate type when
//! one is present. Pass auxiliary
//! crates in dependency order (a crate before the crates that use
//! it): each auxiliary compilation sees the previously built ones
//! through `--extern`. The functions of auxiliary crates are
//! converted into the same program as the main crate's functions,
//! so calls from the main crate into an auxiliary crate execute in
//! the interpreter like any other call. Auxiliary crates whose
//! declared crate type is `proc-macro` or `cdylib` are compiled and
//! linked the same way, but their functions are not converted:
//! proc macros run inside the compiler at compile time, and a
//! cdylib exposes no Rust API to call.
//!
//! Crate type: run mode pins `--crate-type=lib` (unless the caller
//! forwards one). Convert mode pins nothing, mirroring compiletest,
//! which names no crate type for a test file: a forwarded
//! `--crate-type` or a source-level `#![crate_type]` declaration
//! governs when present, and rustc's default (bin) -- with its
//! entry-point checks and dead-code reachability roots -- applies
//! otherwise. To convert a library file in convert mode, forward
//! `--rustc-arg --crate-type=rlib` (the subset runner does this for
//! the mir-opt suite, as compiletest does).
//!
//! Both modes compile `<file.rs>` as a single crate with the pinned
//! Rust 1.100.0 toolchain, obtain every local function's MIR through
//! Stable MIR (`rustc_public`), and convert it to `sparagmos_mir::Body`.
//! Build/run instructions: see the crate README.
//!
//! Run mode also caps lints (`--cap-lints=allow`): the lib pin
//! removes the entry-point reachability roots a binary gets, so a
//! denied lint could otherwise fail the front end on `main` itself
//! where the native binary compile -- the run's reference --
//! succeeded; lint levels are not part of an execution verdict.
//!
//! Run-mode exit codes: 0 = the guest program terminated normally,
//! 101 = the guest program panicked (message on stderr), 2 = the
//! program could not be converted or executed (concrete reason on
//! stderr). Run mode writes the guest program's captured stdout to
//! its own stdout (in every outcome — a native program's output is
//! likewise observable when it panics).
//!
//! Convert-mode exit codes: 0 = the crate converted (per-function
//! outcomes are in the report; an informational or parse-only
//! invocation -- `--version`, `--print`, `--explain`,
//! `-Z parse-crate-root-only` -- converts nothing and still
//! succeeds, because the compilation the compiler was asked to
//! perform did), 1 = the compilation failed (a
//! front-end rejection, a `-Z treat-err-as-bug` abort on a reported
//! diagnostic, or a driver-side monomorphization failure),
//! 101 = the compiler crashed: an internal compiler error (a
//! delayed bug flushed at the end of the run) or a panic that
//! escaped the compiler run. A crash is reported only for a genuine
//! internal failure of the compiler; conversion limits the driver
//! contains per body or per query keep their named reasons in the
//! report and do not change the exit code.
//!
//! Conversion can also fail *inside* Stable MIR itself: some MIR
//! constructs make `rustc_public` panic (`unimplemented!()`). Those
//! panics are caught at the compiler-run boundary: run mode reports
//! them like any other conversion failure (exit 2, with the panic
//! message as the reason), while convert mode reports an escaped
//! panic as a compiler crash (exit 101, above). While the
//! compiler runs, its stdout is redirected to stderr: compile-time
//! output (e.g. `log_syntax!`)
//! is compiler-side diagnostics, and this process's stdout is
//! reserved for the driver's report — in run mode, for the guest's
//! output alone.

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_span;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::{Mutex, OnceLock};

use rustc_public::mir::mono::Instance;
use rustc_public::ty::{
    ClosureKind, ExistentialPredicate, GenericArgKind, GenericArgs, IntTy, RigidTy, Ty,
    TyConstKind, TyKind,
};
use rustc_public::{CrateDef, CrateDefType};
use rustc_public::ItemKind;

use sparagmos_driver::convert::{
    args_mention_params, clear_captured_panic, closure_kind, coroutine_closure_args,
    peek_captured_panic, record_captured_panic, stable_mir_panic_reason,
    take_captured_panic, Converter, CONST_EXPR_REASON, COROUTINE_CLOSURE_TYPE_REASON,
    INSTANCE_CONST_REASON, TAIL_CALL_REASON, UNSAFE_BINDER_REASON,
};
use sparagmos_driver::report::{ConversionReport, FnOutcome, FnReport};
use sparagmos_mir::Body;

struct Collected {
    report: ConversionReport,
    bodies: Vec<(String, Body)>,
    /// `DefId` of each entry of `bodies` (same order), as assigned by
    /// the converter; lets the interpreter resolve `FnDef` call targets
    /// to the converted functions.
    body_def_ids: Vec<sparagmos_mir::DefId>,
    /// Definition name of each entry of `bodies` (same order). For
    /// local items this equals the item name; for collected foreign
    /// instances it is the definition name even when the display
    /// name carries an instance suffix.
    body_def_names: Vec<String>,
    /// Declared ABIs of the collected function instances, keyed
    /// by the `DefId` their bodies carry (see
    /// `Converter::fn_abis`).
    fn_abis: std::collections::HashMap<sparagmos_mir::DefId, sparagmos_mir::Abi>,
    def_names: Vec<String>,
    /// Discriminant values of the converted enum ADTs, for the
    /// interpreter (see `Converter::adt_discriminants`).
    adt_discriminants: sparagmos_mir::AdtDiscriminants,
    /// Layouts of the converted monomorphic ADT instances, for the
    /// interpreter (see `Converter::adt_layouts`).
    adt_layouts: sparagmos_mir::AdtLayouts,
    /// Per-variant field byte layouts of the converted
    /// monomorphic ADT instances, for the interpreter (see
    /// `Converter::adt_field_layouts`).
    adt_field_layouts: sparagmos_mir::AdtFieldLayouts,
    /// Storage layouts of the converted monomorphic tuple
    /// types, for the interpreter (see
    /// `Converter::tuple_layouts`).
    tuple_layouts: sparagmos_mir::TupleLayouts,
    /// Direct discriminant tag layouts of the converted
    /// monomorphic enum instances, for the interpreter (see
    /// `Converter::adt_tag_layouts`).
    adt_tag_layouts: sparagmos_mir::AdtTagLayouts,
    /// Storage-slot maps of the converted monomorphic coroutine
    /// instances, for the interpreter (see
    /// `Converter::coroutine_layouts`).
    coroutine_layouts: sparagmos_mir::CoroutineLayouts,
    /// Declared shapes of the converted ADTs, for the interpreter
    /// (see `Converter::adt_shapes`).
    adt_shapes: sparagmos_mir::AdtShapes,
    /// Derived-`Debug` shapes of the converted monomorphic ADT
    /// instances, for the interpreter (see
    /// `Converter::adt_debug_shapes`).
    adt_debug_shapes: sparagmos_mir::AdtDebugShapes,
    /// Debugger shapes of the converted monomorphic closure and
    /// coroutine instances, for the interpreter (see
    /// `Converter::env_debug_shapes`).
    env_debug_shapes: sparagmos_mir::EnvDebugShapes,
    /// Thread-local statics of the converted program, for the
    /// interpreter (see `Converter::thread_locals`).
    thread_locals: sparagmos_mir::ThreadLocals,
    /// Declared types of the converted program's statics, for
    /// the interpreter (see `Converter::static_tys_mir`).
    static_tys: sparagmos_mir::StaticTys,
    /// Field types of monomorphic ADT instances whose generic
    /// shape mentions a type alias, for the interpreter
    /// (see `Converter::adt_instance_shapes`).
    adt_instance_shapes: sparagmos_mir::AdtInstanceShapes,
    adt_niches: sparagmos_mir::AdtNiches,
    /// Materialisation plans for pointer-cast pointee types,
    /// for the interpreter (see `Converter::mat_plans`).
    mat_plans: std::collections::HashMap<String, sparagmos_interp::ir::MatPlan>,
    /// ADT instantiations whose drop glue the compiler certifies
    /// empty, for the interpreter (see `certify_drop_trivial`).
    drop_trivial_adts: std::collections::HashSet<String>,
    drop_no_impl_adts: std::collections::HashSet<String>,
    dtor_bodies: std::collections::HashMap<String, String>,
    /// Number of leading entries of `bodies` that come from the main
    /// crate itself (the rest are auxiliary-crate functions). The
    /// entry point is only ever searched among these: an auxiliary
    /// crate may define its own `main`-named function, which is not
    /// the program's entry point.
    local_body_count: usize,
    /// Number of leading entries of `report.functions` that belong
    /// to the target crate itself (the rest are auxiliary-crate and
    /// collected library functions). Compile verdicts judge
    /// conversion completeness on the target crate, the way
    /// compiletest does: the library is precompiled, so a collected
    /// library body that stays unconverted does not make the
    /// target crate's compilation incomplete.
    local_report_count: usize,
    /// A monomorphization failure rustc would have reported for
    /// this program (see `sparagmos_driver::mono_check`), detected
    /// after conversion. When set, the compilation as a whole has
    /// failed even though every converted body translated cleanly.
    compile_error: Option<String>,
    /// The entry point rustc designated for this crate, as a
    /// Sparagmos `DefId` into `body_def_ids`, when the crate has
    /// one. In a `--test` crate this is the generated harness
    /// main, which shares the definition name `<crate>::main`
    /// with the user's own `main`; run mode enters through this
    /// id, never through the name alone.
    entry_def_id: Option<sparagmos_mir::DefId>,
    /// Definition-location stamps of the program's
    /// `#[track_caller]` instances (see
    /// `Converter::track_caller_defs`), for the interpreter.
    track_caller_defs:
        std::collections::HashMap<sparagmos_mir::DefId, sparagmos_mir::ConstValue>,
    /// The `Fn` / `FnMut` / `FnOnce` vtable-shim location
    /// stamps (see `Converter::fn_shim_locations`), for the
    /// interpreter.
    fn_shim_locations: [Option<sparagmos_mir::ConstValue>; 3],
    /// Link symbols of the program's instances (see
    /// `Converter::link_symbols`), for the interpreter.
    link_symbols: std::collections::HashMap<sparagmos_mir::DefId, String>,
    /// The source-file table the bodies' source lines index into
    /// (see `Converter::source_files`), for the interpreter's
    /// debugger interface.
    source_files: Vec<String>,
}

static COLLECTED: OnceLock<Mutex<Option<Collected>>> = OnceLock::new();

/// Names of the auxiliary crates (--aux) whose functions `collect`
/// must convert into the program, in `--aux` order. Set by `main`
/// before the compiler run; empty for single-file inputs.
static AUX_CRATES: OnceLock<Vec<String>> = OnceLock::new();

fn aux_crates() -> &'static [String] {
    AUX_CRATES.get().map(|v| v.as_slice()).unwrap_or(&[])
}

fn stash(c: Collected) {
    let cell = COLLECTED.get_or_init(|| Mutex::new(None));
    *cell.lock().unwrap() = Some(c);
}

fn take() -> Option<Collected> {
    COLLECTED.get()?.lock().unwrap().take()
}

/// The process panic hook, as returned by `std::panic::take_hook`.
type PanicHook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync + 'static>;

/// Format a panic as a concrete reason: the payload text when the
/// payload is a string, plus the panic location when known. The
/// compiler also panics with non-string marker payloads; those are
/// reported as such, with the location still identifying where the
/// conversion gave up.
fn panic_reason(info: &std::panic::PanicHookInfo<'_>) -> String {
    let message = if let Some(s) = info.payload().downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_string()
    };
    match info.location() {
        Some(loc) => format!(
            "{message} (panicked at {}:{}:{})",
            loc.file(),
            loc.line(),
            loc.column()
        ),
        None => message,
    }
}

/// The report text for a delayed bug the compiler created during
/// compilation and flushed at the end of the run.
const DELAYED_BUG_ABORT: &str = "the compiler aborted on a delayed bug it created during \
    compilation (see the internal compiler error above)";

/// Whether a panic that escaped the compiler run is the flush of
/// a delayed bug the compiler created during compilation: such a
/// bug is flushed at the end of the run by panicking with the
/// non-string `DelayedBugPanic` marker (rustc_errors/src/lib.rs:
/// 1499), and its own text is an internal compiler error the
/// compiler has already printed. The flush only panics when the
/// compilation would otherwise have finished, so the compilation
/// died on the compiler's own internal error: a compiler crash,
/// which the caller reports as one (Issue #66).
fn is_delayed_bug_flush(reason: &str) -> bool {
    reason.contains("rustc_errors/src/lib.rs:1499:")
}

/// When a panic that escaped the compiler run is the compiler
/// aborting the compilation on a diagnostic it had already
/// reported, the compilation-failure reason to report for it;
/// `None` for anything else. One shape: an error emitted under
/// `-Z treat-err-as-bug` panics with "aborting due to
/// `-Z treat-err-as-bug=…`" (rustc_errors'
/// `panic_if_treat_err_as_bug`). The flag orders exactly this
/// abort, so the outcome stays a compilation failure carrying the
/// reported diagnostic, not a crash.
fn treat_err_abort_reason(reason: &str) -> Option<String> {
    if reason.contains("treat-err-as-bug") {
        Some(reason.to_string())
    } else {
        None
    }
}

/// Reason text from a caught panic payload, when it is a string.
fn payload_reason(payload: &(dyn std::any::Any + Send)) -> Option<String> {
    if let Some(s) = payload.downcast_ref::<&str>() {
        Some((*s).to_string())
    } else if let Some(s) = payload.downcast_ref::<String>() {
        Some(s.clone())
    } else {
        None
    }
}

/// The skip reason for a caught panic whose payload is the
/// bare text "not implemented": several of rustc_public
/// 1.100.0's unimplemented conversions panic with exactly that
/// payload (`unimplemented!()` with no message), so the
/// payload alone cannot tell them apart — but the panic
/// hook's record of the same panic carries its location,
/// which names the site. The record must be this very panic:
/// its message part is the same bare text. (The toolchain is
/// pinned, so the locations are stable.)
fn unimplemented_site_reason(
    payload: &(dyn std::any::Any + Send),
) -> Option<&'static str> {
    if payload_reason(payload).as_deref() != Some("not implemented") {
        return None;
    }
    let captured = peek_captured_panic()?;
    let (message, location) = captured.split_once(" (panicked at ")?;
    if message != "not implemented" {
        return None;
    }
    if location.contains("rustc_public/src/unstable/convert/stable/mir.rs:775:") {
        Some(TAIL_CALL_REASON)
    } else if location
        .contains("rustc_public/src/unstable/convert/stable/ty.rs:471:")
    {
        Some(UNSAFE_BINDER_REASON)
    } else if location
        .contains("rustc_public/src/unstable/convert/stable/ty.rs:577:")
    {
        Some(CONST_EXPR_REASON)
    } else {
        None
    }
}

/// Run a Stable MIR query that may hit one of rustc_public
/// 1.100.0's known unimplemented conversions (inspecting the
/// type kind of an async closure item materializes it, unlike
/// a body fetch, whose types stay opaque until conversion —
/// but a body fetch materializes the whole body, terminators
/// and projections included). A panic from one of those
/// specific unimplemented conversions is contained and
/// reported as its named reason (`stable_mir_panic_reason`
/// for the payloads that identify themselves,
/// `unimplemented_site_reason` for the bare "not implemented"
/// ones); any other panic is resumed — this guard must never
/// paper over an unrelated compiler failure. The query also
/// runs as a body materialization where it fetches one, so the
/// diagnostics its eager constant evaluation emits are
/// discounted the same way (see
/// `sparagmos_driver::convert::with_body_materialization`).
fn guard_stable_mir_query<T>(
    key: &str,
    query: impl FnOnce() -> T,
) -> Result<T, String> {
    sparagmos_driver::convert::with_body_materialization(key, || {
        guard_stable_mir_query_contained(query)
    })
}

fn guard_stable_mir_query_contained<T>(
    query: impl FnOnce() -> T,
) -> Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(query)) {
        Ok(value) => Ok(value),
        Err(payload) => {
            let reason = stable_mir_panic_reason(&*payload)
                .map(str::to_string)
                .or_else(|| unimplemented_site_reason(&*payload).map(str::to_string));
            if let Some(reason) = reason {
                // Contained panic: clear the capture slot, as the
                // instance collection does — a later, genuinely
                // fatal panic must report its own message.
                let _ = take_captured_panic();
                Err(reason)
            } else {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

/// Convert one body through the Stable MIR containment (see
/// `Converter::convert_body_contained`), clearing the panic
/// capture slot when a containment fired: the hook recorded
/// the contained panic's message, and a later, genuinely fatal
/// panic must report its own. A panic the inner containment
/// resumes because its payload is the bare "not implemented"
/// is caught once more here and identified by its recorded
/// location (`unimplemented_site_reason`).
fn convert_body_isolated(
    conv: &mut Converter,
    body: &rustc_public::mir::Body,
) -> Result<Body, Vec<String>> {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        conv.convert_body_contained(body)
    }));
    let result = match outcome {
        Ok(result) => result,
        Err(payload) => {
            if let Some(reason) = unimplemented_site_reason(&*payload) {
                let _ = take_captured_panic();
                return Err(vec![reason.to_string()]);
            }
            std::panic::resume_unwind(payload);
        }
    };
    if matches!(&result, Err(errs) if errs.iter().any(|e| is_contained_reason(e))) {
        let _ = take_captured_panic();
    }
    result
}

/// Fetch one demanded instance's body through the Stable MIR
/// containment (see [`guard_stable_mir_query`]): materializing a
/// body can hit one of rustc_public's known unimplemented
/// conversions (a tail-call terminator is the known case), and in
/// the demand passes that panic must become the definition's
/// named demand failure. An uncontained fetch would unwind to
/// the per-body scan guard, which abandons the scan silently and
/// leaves a demanded definition looking undemanded. `Ok(None)`
/// is an instance with no body available; `Err` is the named
/// reason.
fn fetch_instance_body(
    inst: &Instance,
) -> Result<Option<rustc_public::mir::Body>, String> {
    guard_stable_mir_query(&inst.name(), || inst.body())
}

/// Whether this instance is a compiler-generated impl method
/// (a derived impl) over a `#[repr(simd)]` `Self` type. A
/// derived method carries the very span of its impl — both are
/// the derive attribute's span — while a hand-written method's
/// span is its own definition inside the impl block, never the
/// impl's. The distinction matters because a derived body is
/// synthesized by projecting the ADT's fields, and projecting
/// into a SIMD type is exactly what rustc's MIR validator
/// bans (MCP#838): asking for such a body emits the ICE
/// diagnostic that poisons the whole compilation, even when
/// the panic itself is contained. A hand-written body chooses
/// its own accesses (the `as_array` pointer-cast idiom is the
/// known shape) and is safe to fetch.
fn generated_simd_impl_method(inst: &Instance) -> bool {
    use rustc_public::CrateDef as _;
    let def = inst.def.def_id();
    let Some(parent) = def.parent() else {
        return false;
    };
    if def.span() != parent.span() {
        return false;
    }
    // The instance's own arguments are relative to the impl
    // method's definition (its `T`, `N`, ...), not the trait's,
    // so the `Self` type is read off the method's receiver: the
    // first input of the instance's signature, behind the
    // reference a `&self` receiver adds.
    let Some(sig) = sparagmos_driver::convert::guarded_ty_kind(&inst.ty())
        .and_then(|kind| kind.fn_sig())
        .map(|sig| sig.skip_binder())
    else {
        return false;
    };
    let Some(receiver) = sig.inputs_and_output.first() else {
        return false;
    };
    let self_ty = match sparagmos_driver::convert::guarded_ty_kind(receiver) {
        Some(TyKind::RigidTy(RigidTy::Ref(_, inner, _))) => inner,
        _ => receiver.clone(),
    };
    sparagmos_driver::convert::guarded_ty_kind(&self_ty)
        .is_some_and(|kind| kind.is_simd())
}

/// Whether a conversion error is one of the contained Stable
/// MIR panics' named reasons (see `stable_mir_panic_reason`).
fn is_contained_reason(reason: &str) -> bool {
    reason == COROUTINE_CLOSURE_TYPE_REASON
        || reason == UNSAFE_BINDER_REASON
        || reason == INSTANCE_CONST_REASON
}

/// Install a panic hook that records the first panic's message into
/// the library's panic-capture slot (see
/// `sparagmos_driver::convert::record_captured_panic`) and prints
/// nothing. While the compiler runs, the
/// default panic dump must not leak into this process's stderr: run
/// mode reserves stderr for the contract messages, and the captured
/// message is reported as the conversion-failure reason instead.
/// Returns the previous hook, to be restored by `end_panic_capture`
/// as soon as the compiler run is over — panics outside the run
/// (e.g. in the interpreter) keep the normal reporting.
fn begin_panic_capture() -> PanicHook {
    clear_captured_panic();
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        record_captured_panic(panic_reason(info));
    }));
    previous
}

fn end_panic_capture(previous: PanicHook) {
    std::panic::set_hook(previous);
}

/// Guard that points the process's stdout file descriptor at stderr
/// for the duration of the compiler run. The compiler runs
/// in-process, so anything it prints to *its* stdout at compile
/// time (the `log_syntax!` macro is the known case) would otherwise
/// land in this process's stdout — which run mode reserves for the
/// guest program's output, and summary mode for the driver's own
/// report. Compiler-side output is diagnostic in nature, so it
/// belongs on stderr. Restore explicitly via `restore` once the run
/// is over; dropping the guard restores as a safety net.
#[cfg(unix)]
struct StdoutToStderr {
    saved: Option<std::os::unix::io::RawFd>,
}

#[cfg(unix)]
impl StdoutToStderr {
    fn new() -> Self {
        use std::io::Write;
        // Flush anything buffered before swapping the descriptor,
        // so it cannot surface on the wrong stream afterwards.
        let _ = std::io::stdout().flush();
        let saved = unsafe { libc::dup(libc::STDOUT_FILENO) };
        if saved < 0 {
            return Self { saved: None };
        }
        if unsafe { libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) } < 0 {
            unsafe { libc::close(saved) };
            return Self { saved: None };
        }
        Self { saved: Some(saved) }
    }

    fn restore(&mut self) {
        use std::io::Write;
        if let Some(saved) = self.saved.take() {
            // Flush compile-time output still buffered in Rust's
            // stdout handle so it goes to stderr, as intended,
            // instead of leaking into the restored stdout.
            let _ = std::io::stdout().flush();
            unsafe {
                libc::dup2(saved, libc::STDOUT_FILENO);
                libc::close(saved);
            }
        }
    }
}

#[cfg(unix)]
impl Drop for StdoutToStderr {
    fn drop(&mut self) {
        self.restore();
    }
}

/// No-op counterpart on non-Unix platforms (the driver is developed
/// and measured on Linux; the descriptor swap has no portable
/// equivalent there).
#[cfg(not(unix))]
struct StdoutToStderr;

#[cfg(not(unix))]
impl StdoutToStderr {
    fn new() -> Self {
        Self
    }

    fn restore(&mut self) {}
}

/// Convert one function (a local item or an auxiliary-crate
/// function) and record the outcome. `body` is the function's
/// monomorphic MIR body: `Err(reason)` records the function as
/// skipped with that reason (no monomorphic instance), and
/// `Ok(None)` records it as skipped with no body available.
fn convert_item(
    conv: &mut Converter,
    report: &mut ConversionReport,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    name: String,
    def_id: sparagmos_mir::DefId,
    body: Result<Option<rustc_public::mir::Body>, String>,
) {
    let body = match body {
        Ok(Some(b)) => b,
        Ok(None) => {
            let mut reasons = BTreeMap::new();
            reasons.insert("no MIR body available".to_string(), 1);
            report.functions.push(FnReport {
                name,
                blocks: 0,
                locals: 0,
                arg_count: 0,
                outcome: FnOutcome::Skipped { reasons },
            });
            return;
        }
        Err(reason) => {
            let mut reasons = BTreeMap::new();
            reasons.insert(reason, 1);
            report.functions.push(FnReport {
                name,
                blocks: 0,
                locals: 0,
                arg_count: 0,
                outcome: FnOutcome::Skipped { reasons },
            });
            return;
        }
    };
    let blocks = body.blocks.len();
    let locals = body.locals().len();
    let arg_count = body.arg_locals().len();
    match convert_body_isolated(&mut *conv, &body) {
        Ok(converted) => {
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
                blocks: 0,
                locals: 0,
                arg_count: 0,
                outcome: FnOutcome::Skipped { reasons },
            });
        }
    }
}

/// Record `name` as skipped for one concrete `reason`.
fn skip_fn(report: &mut ConversionReport, name: &str, reason: String) {
    let mut reasons = BTreeMap::new();
    reasons.insert(reason, 1);
    report.functions.push(FnReport {
        name: name.to_string(),
        blocks: 0,
        locals: 0,
        arg_count: 0,
        outcome: FnOutcome::Skipped { reasons },
    });
}

/// What came of trying to instantiate an item as a closure after
/// `Instance::try_from` rejected it.
enum ClosureAttempt {
    /// A monomorphic closure instance whose body can be converted.
    Instance(Instance),
    /// The item is not a closure at all.
    NotClosure,
    /// The item is a closure that stays skipped, with the reason.
    Unsupported(String),
    /// The closure is defined in a generic context: its own
    /// arguments still mention the enclosing definition's
    /// parameters, so it has no instance as an item — but a
    /// monomorphic instance of the enclosing definition mentions
    /// the closure with concrete arguments, and the closure
    /// collection after the item loop converts the instances
    /// actually demanded (the same deferral generic functions get).
    GenericContext,
    /// The item is an async closure: asking for its type hits the
    /// bridge's unimplemented coroutine-closure type conversion,
    /// so no instance can be formed from the item alone — but a
    /// converted body that creates the closure carries its
    /// definition and arguments in the coroutine-closure
    /// aggregate, and the closure collection after the item loop
    /// accounts the instances actually demanded (the same
    /// deferral generic functions get).
    CoroutineClosure,
}

/// Instantiate a local closure item's body.
///
/// `Instance::try_from` rejects every closure item: a closure
/// definition always carries generic parameters (its parent
/// arguments plus its signature and capture types), so it never has
/// a mono instance *as an item*. A closure defined in a concrete
/// context can still be instantiated from the generic arguments of
/// its own type via `Instance::resolve_closure`, requesting the
/// closure's own kind so no `FnOnce` adapter shim is produced and
/// the instance's body is the closure body itself. Closures whose
/// arguments still mention generic parameters (defined inside a
/// generic function, method or impl) have no single monomorphic
/// body to collect as an item — the same limitation as generic
/// functions — so the item loop defers it and the closure
/// collection after the loop converts the instances that
/// monomorphic bodies actually demand. Async closures are
/// coroutine closures: merely asking for an async closure
/// item's type hits the unimplemented coroutine-closure type
/// conversion, so no instance can be formed from the item —
/// the guard's reason identifies the item, and it is deferred
/// the same way, to be accounted from the coroutine-closure
/// aggregates of converted bodies (the `CoroutineClosure` arm
/// of `closure_instance_kind` below only fires if a future
/// Stable MIR implements the type conversion).
fn closure_instance(item: &rustc_public::CrateItem) -> ClosureAttempt {
    match guard_stable_mir_query("", || closure_instance_kind(item)) {
        Ok(attempt) => attempt,
        Err(reason) if reason == COROUTINE_CLOSURE_TYPE_REASON => {
            ClosureAttempt::CoroutineClosure
        }
        Err(reason) => ClosureAttempt::Unsupported(reason),
    }
}

/// The kind inspection behind [`closure_instance`]: asking for
/// an async closure item's type kind is what materializes the
/// unimplemented coroutine-closure type conversion, so the
/// whole inspection runs under the guard.
fn closure_instance_kind(item: &rustc_public::CrateItem) -> ClosureAttempt {
    match item.ty().kind() {
        TyKind::RigidTy(RigidTy::CoroutineClosure(..)) => ClosureAttempt::Unsupported(
            "async closure body is not converted (coroutine closures are unsupported)".to_string(),
        ),
        TyKind::RigidTy(RigidTy::Closure(def, args)) => {
            if args_mention_params(&args) {
                return ClosureAttempt::GenericContext;
            }
            let Some(kind) = closure_kind(&args) else {
                return ClosureAttempt::Unsupported(
                    "closure kind is not recoverable from its generic arguments".to_string(),
                );
            };
            match Instance::resolve_closure(def, &args, kind) {
                Ok(instance) => ClosureAttempt::Instance(instance),
                Err(e) => {
                    ClosureAttempt::Unsupported(format!("closure instance resolution failed: {e}"))
                }
            }
        }
        _ => ClosureAttempt::NotClosure,
    }
}

/// A local function item that has no monomorphic instance as an
/// item (a generic function definition): it is deferred during the
/// item loop and accounted for by [`collect_local_instances`] once
/// the instances its call sites demand are known.
///
/// Tuple-struct and tuple-variant constructors (`ItemKind::Ctor`)
/// are deferred the same way: a generic constructor has no
/// instance as an item either. A constructor differs from a plain
/// function in one respect — when nothing demands it as a value,
/// there is nothing left unconverted: its direct uses are
/// aggregate rvalues in the callers, and the constructor item
/// itself has no body in any monomorphization that was not
/// demanded. [`apply_instance_demands`] accounts such a
/// constructor as converted without a body.
struct DeferredFn {
    def: rustc_public::DefId,
    name: String,
    /// Index of this item's placeholder entry in the report.
    report_idx: usize,
    /// Whether this deferred item is a data constructor
    /// (`ItemKind::Ctor(CtorKind::Fn)`).
    ctor: bool,
}

/// Cap on monomorphic instance bodies collected for generic local
/// definitions per program, mirroring the reachability
/// collection's cap for foreign bodies.
const MAX_LOCAL_INSTANCES: usize = 1024;

/// Depth at which the demand passes report rustc's
/// monomorphization recursion limit: the number of instances of
/// one definition nested along a single demand path (each a
/// distinct instance — a repeated instance is deduplicated and
/// never deepens the path). This mirrors
/// `check_recursion_limit` in rustc's mono collector, which
/// tracks exactly this per-definition path depth against the
/// crate's recursion limit and fails compilation past it
/// ("reached the recursion limit while instantiating ...").
/// 128 is rustc's default recursion limit; a crate's
/// `#![recursion_limit]` attribute is not exposed through
/// Stable MIR, so the default is what the driver can mirror.
const RECURSION_LIMIT_DEPTH: u16 = 128;

/// How many instances of each definition sit on the demand path
/// that led to one body: the chain of instances whose bodies
/// demanded the next, from a collection root down to this body.
/// Only repetitions matter to the recursion-limit check, but the
/// full map is kept so any definition repeating deeper down the
/// path is counted correctly.
type DemandPath = std::collections::HashMap<rustc_public::DefId, u16>;

/// Outcome accumulated for one deferred definition across every
/// demand pass of a program.
#[derive(Default)]
struct Demand {
    converted: usize,
    failures: BTreeMap<String, usize>,
    first_counts: Option<(usize, usize, usize)>,
}

fn fail_all(
    demands: &mut HashMap<rustc_public::DefId, Demand>,
    owners: &[rustc_public::DefId],
    reason: String,
) {
    for owner in owners {
        *demands
            .entry(*owner)
            .or_default()
            .failures
            .entry(reason.clone())
            .or_default() += 1;
    }
}

/// State shared by the demand passes of one program
/// ([`collect_local_instances`] runs first over the converted
/// local bodies, then over the foreign bodies the reachability
/// collection obtained): which instances were already considered,
/// how many instance bodies were taken against
/// [`MAX_LOCAL_INSTANCES`], which definitions already own their
/// plain display name, and the per-definition outcomes the passes
/// accumulated.
#[derive(Default)]
struct InstanceDemandState {
    seen: HashSet<(rustc_public::DefId, Instance)>,
    unresolved_seen: HashSet<(rustc_public::DefId, String)>,
    plain_taken: HashSet<rustc_public::DefId>,
    taken: usize,
    demands: HashMap<rustc_public::DefId, Demand>,
    /// Outcomes accumulated for closures defined in other crates
    /// (see [`collect_closure_instances`] and
    /// [`collect_const_closure_instances`]): a foreign closure has
    /// no item-loop report entry to fill in, so the caller records
    /// these with [`apply_foreign_closure_demands`].
    foreign_closure_demands: HashMap<rustc_public::DefId, Demand>,
    /// Outcomes accumulated for plain functions defined in other
    /// crates by [`collect_const_fn_instances`]: the counterpart
    /// of `foreign_closure_demands` for function items, recorded
    /// with the same [`apply_foreign_closure_demands`].
    foreign_fn_demands: HashMap<rustc_public::DefId, Demand>,
    /// The first recursion-limit failure the demand passes hit
    /// (see [`RECURSION_LIMIT_DEPTH`]), reported as a compilation
    /// failure alongside the monomorphization checks.
    recursion_limit: Option<String>,
}

/// Convert the monomorphic instances of generic local function
/// definitions that a set of bodies actually calls.
///
/// A generic definition has no instance as an item, so the item
/// loop cannot convert it; but every call site names the
/// definition through an `FnDef` target with concrete generic
/// arguments, and resolving those targets yields the instances
/// the program demands — the same mechanism the reachability
/// collection uses for foreign functions. This pass walks `seeds`
/// (and, transitively, the instance bodies themselves) for
/// `FnDef` targets, resolves the ones belonging to deferred
/// definitions, and converts each distinct instance, registering
/// it under the same `(referenced definition, instance)` key call
/// sites resolve through.
///
/// The pass runs several times per program with shared `state`:
/// first over the converted local bodies, then — interleaved with
/// the reachability collection — over the foreign bodies that
/// collection obtained, which can demand further local instances
/// (a std adapter calling a trait method of a local generic type
/// is the common shape). Sharing the state keeps the per-program
/// cap and the instance deduplication in force across passes.
/// Outcomes accumulate in the state; the caller records them in
/// the report with [`apply_instance_demands`] once the last pass
/// has run. The Stable MIR bodies of the instances this pass
/// converted are returned so the reachability collection can walk
/// them for foreign targets too.
fn collect_local_instances(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    deferred: &[DeferredFn],
    seeds: &[rustc_public::mir::Body],
    state: &mut InstanceDemandState,
) -> Vec<rustc_public::mir::Body> {
    use std::collections::VecDeque;

    let InstanceDemandState {
        seen,
        unresolved_seen,
        plain_taken,
        taken,
        demands,
        foreign_closure_demands: _,
        foreign_fn_demands: _,
        recursion_limit,
    } = state;

    let deferred_defs: HashSet<rustc_public::DefId> = deferred.iter().map(|d| d.def).collect();
    let deferred_ctors: HashSet<rustc_public::DefId> =
        deferred.iter().filter(|d| d.ctor).map(|d| d.def).collect();
    let mut queue: VecDeque<(rustc_public::mir::Body, DemandPath)> =
        seeds.iter().map(|b| (b.clone(), DemandPath::new())).collect();
    let mut extra_seeds: Vec<rustc_public::mir::Body> = Vec::new();

    while let Some((body, path)) = queue.pop_front() {
        // Instance resolution and body access go through the
        // compiler's internals and can panic on exotic targets
        // (higher-ranked and late-bound types are known cases).
        // Isolate each scanned body: a panic abandons that body's
        // remaining targets instead of the whole conversion, so
        // instance collection can never make a crate convert worse
        // than the item loop alone did.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // The demand targets of this body: its `FnDef` call
        // targets, plus every `FnDef` constant it materializes
        // that names a deferred definition. A constructor used
        // as a value (assigned, passed on, returned) never
        // appears as a call target of the body that demands it,
        // so call targets alone would miss the demand (Lane CF,
        // constructors); the same holds for a plain generic
        // function used as a value, so the scan follows `FnDef`
        // constants for every deferred definition (Lane CP).
        // Targets are filtered by deferred membership below, so
        // constants naming anything else cost one hash lookup.
        let mut targets = sparagmos_driver::reach::call_targets(&body);
        if !deferred.is_empty() {
            // Constants are resolved and filtered by the same
            // referenced-or-resolved deferred check as call targets
            // below, not by the referenced definition alone: a
            // constant can name a trait method whose resolution is
            // the deferred definition (an impl method). `core::fmt`'s
            // argument constructors materialize exactly that shape —
            // the formatter stored in an `Argument` is the trait
            // method `FnDef` constant in the constructor's body,
            // which the reachability pass walks for this purpose.
            targets.extend(sparagmos_driver::reach::fndef_constants(&body));
        }
        for (def, args) in targets {
            let referenced = def.def_id();
            let inst = match Instance::resolve(def, &args) {
                Ok(inst) => inst,
                Err(_) => {
                    if deferred_defs.contains(&referenced)
                        && unresolved_seen.insert((referenced, format!("{args:?}")))
                    {
                        *demands
                            .entry(referenced)
                            .or_default()
                            .failures
                            .entry("instance resolution failed".to_string())
                            .or_default() += 1;
                    }
                    continue;
                }
            };
            let resolved = inst.def.def_id();
            if !deferred_defs.contains(&referenced) && !deferred_defs.contains(&resolved) {
                continue;
            }
            if !seen.insert((referenced, inst)) {
                continue;
            }
            // The instance is credited to every deferred definition
            // it belongs to: the referenced definition (a direct
            // call, or a trait method whose default body this is)
            // and the resolved definition when it differs (an impl
            // method demanded through its trait).
            let mut owners: Vec<rustc_public::DefId> = Vec::new();
            if deferred_defs.contains(&referenced) {
                owners.push(referenced);
            }
            if resolved != referenced && deferred_defs.contains(&resolved) {
                owners.push(resolved);
            }
            // The expanded-size cap (see
            // `sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE`):
            // a size-increasing recursion nests its definition
            // only linearly deep, so the depth check below cannot
            // bind before the type expansion exhausts memory.
            // The size of the demanded arguments is the quantity
            // that explodes, so it is the one capped here;
            // tripping it is reported exactly like the depth
            // limit, since it is the same rustc failure, reached
            // through the expansion rustc's interning keeps
            // implicit.
            if !sparagmos_driver::reach::generic_args_within_expanded_size(
                &args,
                sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE,
            ) {
                if recursion_limit.is_none() {
                    *recursion_limit = Some(format!(
                        "reached the recursion limit while instantiating `{}` (instantiation size limit)",
                        def.name()
                    ));
                }
                fail_all(&mut *demands, &owners, "recursion limit reached".to_string());
                continue;
            }
            // The recursion-limit check mirrors rustc's mono
            // collector: if this instance nests its definition
            // deeper along the current demand path than the
            // recursion limit, instantiating it is the failure
            // rustc reports, however large the collection as a
            // whole legitimately is. A merely large but finite
            // demand never nests one definition this deep along a
            // single path, so the collection cap below stays the
            // only consequence for it.
            let depth = path.get(&resolved).copied().unwrap_or(0) + 1;
            if depth > RECURSION_LIMIT_DEPTH {
                if recursion_limit.is_none() {
                    *recursion_limit = Some(format!(
                        "reached the recursion limit while instantiating `{}`",
                        inst.name()
                    ));
                }
                fail_all(&mut *demands, &owners, "recursion limit reached".to_string());
                continue;
            }
            let mut child_path = path.clone();
            *child_path.entry(resolved).or_default() += 1;
            if *taken >= MAX_LOCAL_INSTANCES {
                fail_all(&mut *demands, &owners, "collection limit reached".to_string());
                continue;
            }
            // Mirror the reachability collection's guards before
            // asking for the body: a virtual instance has no MIR
            // body, and asking the compiler for one panics inside
            // it (and the panic's ICE diagnostic poisons the whole
            // compilation even when the unwind is caught).
            if matches!(inst.kind, rustc_public::mir::mono::InstanceKind::Virtual { .. }) {
                fail_all(
                    &mut *demands,
                    &owners,
                    "dynamic dispatch target (virtual instance)".to_string(),
                );
                continue;
            }
            if !inst.has_body() {
                let reason = match inst.kind {
                    rustc_public::mir::mono::InstanceKind::Intrinsic => {
                        "compiler intrinsic (no fallback body)"
                    }
                    _ => "no MIR body available",
                };
                fail_all(&mut *demands, &owners, reason.to_string());
                continue;
            }
            // Mirror the virtual guard for compiler-generated
            // impl bodies over a SIMD type: the derived body
            // projects the ADT's fields, the MIR validator bans
            // projections into SIMD types (MCP#838), and the
            // ICE diagnostic emitted while asking poisons the
            // whole compilation even when the panic is
            // contained. The derived rendering is served
            // without the body (the interpreter's structural
            // Debug shapes, or the trait's builtin model), so
            // the demand is recorded unmet instead of fetched.
            // Hand-written methods over SIMD types do not trip
            // the ban (their bodies choose their own accesses)
            // and are fetched as usual.
            if generated_simd_impl_method(&inst) {
                fail_all(
                    &mut *demands,
                    &owners,
                    "compiler-generated impl body over a SIMD type".to_string(),
                );
                continue;
            }
            let ibody = match fetch_instance_body(&inst) {
                Ok(Some(b)) => b,
                Ok(None) => {
                    fail_all(&mut *demands, &owners, "no MIR body available".to_string());
                    continue;
                }
                Err(reason) => {
                    fail_all(&mut *demands, &owners, reason);
                    continue;
                }
            };
            match convert_body_isolated(&mut *conv, &ibody) {
                Ok(converted) => {
                    // No representability screen: these are local
                    // functions, and the item loop converts local
                    // bodies without one. A body the interpreter
                    // cannot represent stays in the program and is
                    // recorded unconverted by the interpreter's own
                    // per-body isolation if it is ever called.
                    *taken += 1;
                    queue.push_back((ibody.clone(), child_path.clone()));
                    extra_seeds.push(ibody.clone());
                    let counts = (ibody.blocks.len(), ibody.locals().len(), ibody.arg_locals().len());
                    // Display name: the first converted instance of
                    // the referenced definition carries its plain
                    // name; further instances are suffixed with the
                    // instance name (the reachability collection's
                    // rule for foreign instances).
                    let ref_name = def.name();
                    let display = if plain_taken.insert(referenced) {
                        ref_name.clone()
                    } else {
                        format!("{} [{}]", ref_name, inst.name())
                    };
                    bodies.push((display, converted));
                    body_def_ids.push(conv.instance_def_id(referenced, &inst));
                    body_def_names.push(ref_name);
                    for owner in &owners {
                        let demand = demands.entry(*owner).or_default();
                        demand.converted += 1;
                        if demand.first_counts.is_none() {
                            demand.first_counts = Some(counts);
                        }
                    }
                }
                Err(errs) => {
                    for e in errs {
                        fail_all(&mut *demands, &owners, e);
                    }
                }
            }
        }
        }));
        if outcome.is_err() {
            // The panic capture slot keeps the first panic's
            // message; this panic was contained, so clear the
            // slot — a later, genuinely fatal panic must report
            // its own message.
            let _ = take_captured_panic();
        }
    }

    extra_seeds
}

/// Convert the monomorphic instances of closures defined in
/// generic contexts that a set of bodies actually mentions.
///
/// A closure defined inside a generic function has no instance as
/// an item (see [`closure_instance`]), but every monomorphic
/// instance of the enclosing definition mentions the closure type
/// with the enclosing substitution applied — in a closure
/// aggregate, a local's type, or a call target's generic
/// arguments. This pass walks `seeds` (and, transitively, the
/// closure bodies themselves) for such mentions via
/// [`sparagmos_driver::reach::closure_targets`], resolves each
/// concrete one through `Instance::resolve_closure` with the
/// closure's own kind, and converts it, registering the body under
/// the same `(closure definition, instance)` key
/// [`sparagmos_driver::convert::Converter::closure_def_id`] assigns
/// to references. Mentions whose arguments still mention
/// parameters are not demands this pass can satisfy — the
/// enclosing body is itself generic — and are left to the pass
/// over the monomorphic instances. Outcomes accumulate in the
/// shared demand state (cap and deduplication are shared with the
/// function-instance collection); the caller records them with
/// [`apply_instance_demands`]. The Stable MIR bodies converted
/// here are returned so the reachability collection can walk them
/// for foreign targets too.
///
/// The same pass also collects closures defined in *other*
/// crates. A foreign closure is never an item of the local crate,
/// so no item loop defers it, and the reachability collection
/// follows `FnDef` targets and constants only — a closure created
/// inside a generic foreign function (the closure
/// `std::sys::thread_local::destroy::<T>` passes to
/// `abort_on_dtor_unwind`, the one inside `slice::binary_search`)
/// is mentioned concretely by the collected foreign body yet
/// demanded by no pass, and calls to it fail as unknown
/// functions. Foreign outcomes accumulate in their own map of
/// the shared state and are recorded with
/// [`apply_foreign_closure_demands`]: there is no local item to
/// credit them to.
fn collect_closure_instances(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    deferred: &[DeferredFn],
    local_defs: &HashSet<rustc_public::DefId>,
    seeds: &[rustc_public::mir::Body],
    collected: &HashSet<sparagmos_mir::DefId>,
    state: &mut InstanceDemandState,
) -> Vec<rustc_public::mir::Body> {
    use std::collections::VecDeque;

    let InstanceDemandState {
        seen,
        unresolved_seen,
        plain_taken,
        taken,
        demands,
        foreign_closure_demands,
        foreign_fn_demands: _,
        recursion_limit,
    } = state;

    let deferred_names: HashMap<rustc_public::DefId, &str> =
        deferred.iter().map(|d| (d.def, d.name.as_str())).collect();
    let mut queue: VecDeque<rustc_public::mir::Body> = seeds.iter().cloned().collect();
    let mut extra_seeds: Vec<rustc_public::mir::Body> = Vec::new();

    while let Some(body) = queue.pop_front() {
        // Instance resolution goes through the compiler's
        // internals and can panic on exotic targets; isolate each
        // scanned body exactly as the function-instance collection
        // does, so closure collection can never make a crate
        // convert worse than the item loop alone did.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for (def, args) in sparagmos_driver::reach::closure_targets(&body) {
                let referenced = def.def_id();
                // Two kinds of closures are collected here. One
                // defined in a generic context of the local crate
                // was deferred by the item loop and is accounted
                // through the shared demand state, like a generic
                // function. One defined in another crate is
                // demanded by no other pass: no item loop iterates
                // foreign items, and the reachability collection
                // follows `FnDef` targets only, so a foreign
                // closure a converted body mentions concretely
                // (the closure inside a generic std function is
                // the common case) is collected here or nowhere;
                // it is accounted in the state's foreign map
                // instead. A local closure outside a generic
                // context is the item loop's business and is
                // skipped.
                let (base_name, foreign) = match deferred_names.get(&referenced) {
                    Some(&name) => (name.to_string(), false),
                    None => {
                        if local_defs.contains(&referenced) {
                            continue;
                        }
                        (referenced.name(), true)
                    }
                };
                let target_demands: &mut HashMap<rustc_public::DefId, Demand> = if foreign {
                    &mut *foreign_closure_demands
                } else {
                    &mut *demands
                };
                if args_mention_params(&args) {
                    continue;
                }
                let owners = [referenced];
                // The expanded-size cap, as in the function
                // demand pass: a closure demanded with
                // size-increasing arguments is the same
                // recursion-limit situation. A foreign closure
                // is the exception: its arguments come from a
                // body the compiler already monomorphized, so
                // the instance is finite by construction (the
                // foreign demand graph of a deeply nested but
                // legitimate program measures far past the cap;
                // see the reachability collection, which exempts
                // foreign demand for the same reason). Tripping
                // the cap there fails only that demand, recorded
                // under the same reason; it must not fail the
                // whole compilation as a recursion-limit error.
                if !sparagmos_driver::reach::generic_args_within_expanded_size(
                    &args,
                    sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE,
                ) {
                    if !foreign && recursion_limit.is_none() {
                        *recursion_limit = Some(format!(
                            "reached the recursion limit while instantiating `{}` (instantiation size limit)",
                            base_name
                        ));
                    }
                    fail_all(&mut *target_demands, &owners, "recursion limit reached".to_string());
                    continue;
                }
                let Some(kind) = closure_kind(&args) else {
                    if coroutine_closure_args(&args) {
                        // An async closure demand: the definition
                        // has no closure instance of its own (the
                        // compiler's closure resolution rejects
                        // coroutine closures), and its execution
                        // bodies — the AsyncFn shim and the
                        // coroutine — are collected by the instance
                        // and coroutine demands under their own
                        // definitions. Record no demand for the
                        // closure definition itself: the deferred
                        // item is accounted not demanded, under the
                        // coroutine-closure reason it was deferred
                        // with.
                        continue;
                    }
                    if unresolved_seen.insert((referenced, format!("{args:?}"))) {
                        *target_demands
                            .entry(referenced)
                            .or_default()
                            .failures
                            .entry(
                                "closure kind is not recoverable from its generic arguments"
                                    .to_string(),
                            )
                            .or_default() += 1;
                    }
                    continue;
                };
                let inst = match Instance::resolve_closure(def, &args, kind) {
                    Ok(inst) => inst,
                    Err(_) => {
                        if unresolved_seen.insert((referenced, format!("{args:?}"))) {
                            *target_demands
                                .entry(referenced)
                                .or_default()
                                .failures
                                .entry("closure instance resolution failed".to_string())
                                .or_default() += 1;
                        }
                        continue;
                    }
                };
                if !seen.insert((referenced, inst)) {
                    continue;
                }
                // The reachability collection resolves the
                // closures in a call's generic arguments itself,
                // so an instance it already converted must not be
                // converted again here: two bodies under one
                // program DefId collide when the program is built.
                // Its copy (and its report entry) stands.
                let collected_id = conv.instance_def_id(referenced, &inst);
                if collected.contains(&collected_id) {
                    continue;
                }
                if *taken >= MAX_LOCAL_INSTANCES {
                    fail_all(&mut *target_demands, &owners, "collection limit reached".to_string());
                    continue;
                }
                if !inst.has_body() {
                    fail_all(&mut *target_demands, &owners, "no MIR body available".to_string());
                    continue;
                }
                // As in `collect_local_instances`: a
                // compiler-generated impl body over a SIMD type
                // must not be asked for (MCP#838 validation ICE).
                if generated_simd_impl_method(&inst) {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "compiler-generated impl body over a SIMD type".to_string(),
                    );
                    continue;
                }
                let ibody = match fetch_instance_body(&inst) {
                    Ok(Some(b)) => b,
                    Ok(None) => {
                        fail_all(
                            &mut *demands,
                            &owners,
                            "no MIR body available".to_string(),
                        );
                        continue;
                    }
                    Err(reason) => {
                        fail_all(&mut *demands, &owners, reason);
                        continue;
                    }
                };
                match convert_body_isolated(&mut *conv, &ibody) {
                    Ok(converted) => {
                        *taken += 1;
                        queue.push_back(ibody.clone());
                        extra_seeds.push(ibody.clone());
                        let counts =
                            (ibody.blocks.len(), ibody.locals().len(), ibody.arg_locals().len());
                        let display = if plain_taken.insert(referenced) {
                            base_name.to_string()
                        } else {
                            format!("{} [{}]", base_name, inst.name())
                        };
                        bodies.push((display, converted));
                        body_def_ids.push(collected_id);
                        body_def_names.push(base_name.to_string());
                        let demand = target_demands.entry(referenced).or_default();
                        demand.converted += 1;
                        if demand.first_counts.is_none() {
                            demand.first_counts = Some(counts);
                        }
                    }
                    Err(errs) => {
                        for e in errs {
                            fail_all(&mut *target_demands, &owners, e);
                        }
                    }
                }
            }
        }));
        if outcome.is_err() {
            // Contained panic: clear the capture slot, as the
            // function-instance collection does.
            let _ = take_captured_panic();
        }
    }

    extra_seeds
}

/// Convert the closures that converted bodies materialize as
/// *values* out of constants.
///
/// A closure can be demanded without any body's types mentioning
/// it: const evaluation stores it inside a constant allocation —
/// a function pointer in a constant struct, a closure value
/// nested in a constant aggregate — and body conversion decodes
/// the allocation into a closure value (see
/// `Converter::const_closure_mentions`). The canonical case is
/// `std::hash::RandomState::new::KEYS`: the `LocalKey` value is a
/// constant whose function-pointer field names the thread-local
/// initializer closure, so every `HashMap` program materializes
/// that closure as a value while no local declaration, operand
/// type, or aggregate anywhere mentions its type. The
/// type-driven passes ([`collect_closure_instances`],
/// [`sparagmos_driver::reach::closure_targets`]) cannot see it,
/// and the foreign constant's own body is never converted (no
/// item loop iterates foreign items), so without this pass the
/// closure's body is collected by no pass and calls to it fail
/// as unknown functions.
///
/// The mentions are exactly the constants conversion itself
/// decoded — constants of reached bodies and the initializers of
/// statics they reference — so the pass demands nothing
/// conversion did not already touch. Routing and accounting
/// mirror the closure demand pass: a closure the item loop
/// deferred (a generic-context local closure) is accounted in
/// the shared demand state, a local closure outside a generic
/// context is the item loop's business and is skipped, and a
/// closure defined in another crate is accounted in the state's
/// foreign map (see [`apply_foreign_closure_demands`]). The
/// shared `seen` set deduplicates against the type-driven pass:
/// whichever pass reaches an instance first converts it.
///
/// Coroutine mentions are not part of this record: a constant
/// cannot hold a coroutine value (the constant decoder has no
/// coroutine value form — coroutines are created by aggregates
/// in bodies, never stored in allocations), so there is no
/// constant-content coroutine mention to scan for.
fn collect_const_closure_instances(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    deferred: &[DeferredFn],
    local_defs: &HashSet<rustc_public::DefId>,
    collected: &HashSet<sparagmos_mir::DefId>,
    state: &mut InstanceDemandState,
) -> Vec<rustc_public::mir::Body> {
    let InstanceDemandState {
        seen,
        unresolved_seen,
        plain_taken,
        taken,
        demands,
        foreign_closure_demands,
        foreign_fn_demands: _,
        recursion_limit,
    } = state;

    let deferred_names: HashMap<rustc_public::DefId, &str> =
        deferred.iter().map(|d| (d.def, d.name.as_str())).collect();
    let mut converted_bodies: Vec<rustc_public::mir::Body> = Vec::new();

    // Converting one mentioned closure's body decodes its own
    // constants, which can mention further closures; drain the
    // converter's record until a round adds nothing. Each
    // distinct mention is recorded once and each conversion is
    // deduplicated and capped through the shared state, so the
    // loop terminates.
    loop {
        let mentions = std::mem::take(&mut conv.const_closure_mentions);
        if mentions.is_empty() {
            break;
        }
        for (def, args) in &mentions {
            let referenced = def.def_id();
            let (base_name, foreign) = match deferred_names.get(&referenced) {
                Some(&name) => (name.to_string(), false),
                None => {
                    if local_defs.contains(&referenced) {
                        continue;
                    }
                    (referenced.name(), true)
                }
            };
            if args_mention_params(args) {
                continue;
            }
            // Instance resolution goes through the compiler's
            // internals and can panic on exotic targets; isolate
            // each mention exactly as the other demand passes
            // isolate each scanned body.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let target_demands: &mut HashMap<rustc_public::DefId, Demand> = if foreign {
                    &mut *foreign_closure_demands
                } else {
                    &mut *demands
                };
                let owners = [referenced];
                // The expanded-size cap, as in the closure
                // demand pass: a closure demanded with
                // size-increasing arguments is the same
                // recursion-limit situation. As there, a
                // foreign closure's oversized arguments fail
                // only its own demand: the instance comes from
                // a constant the compiler already monomorphized,
                // so its size is finite by construction and is
                // not a recursion-limit violation.
                if !sparagmos_driver::reach::generic_args_within_expanded_size(
                    args,
                    sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE,
                ) {
                    if !foreign && recursion_limit.is_none() {
                        *recursion_limit = Some(format!(
                            "reached the recursion limit while instantiating `{}` (instantiation size limit)",
                            base_name
                        ));
                    }
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "recursion limit reached".to_string(),
                    );
                    return;
                }
                let Some(kind) = closure_kind(args) else {
                    if coroutine_closure_args(args) {
                        // An async closure demand: no demand is
                        // recorded for the closure definition
                        // itself (see collect_closure_instances).
                        return;
                    }
                    if unresolved_seen.insert((referenced, format!("{args:?}"))) {
                        fail_all(
                            &mut *target_demands,
                            &owners,
                            "closure kind is not recoverable from its generic arguments"
                                .to_string(),
                        );
                    }
                    return;
                };
                let inst = match Instance::resolve_closure(*def, args, kind) {
                    Ok(inst) => inst,
                    Err(_) => {
                        if unresolved_seen.insert((referenced, format!("{args:?}"))) {
                            fail_all(
                                &mut *target_demands,
                                &owners,
                                "closure instance resolution failed".to_string(),
                            );
                        }
                        return;
                    }
                };
                if !seen.insert((referenced, inst)) {
                    return;
                }
                // Already converted by the reachability
                // collection (see collect_closure_instances):
                // a second copy under the same program DefId
                // would collide when the program is built.
                let collected_id = conv.instance_def_id(referenced, &inst);
                if collected.contains(&collected_id) {
                    return;
                }
                if *taken >= MAX_LOCAL_INSTANCES {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "collection limit reached".to_string(),
                    );
                    return;
                }
                if !inst.has_body() {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "no MIR body available".to_string(),
                    );
                    return;
                }
                let Some(ibody) = sparagmos_driver::convert::with_body_materialization(
                    &inst.name(),
                    || inst.body(),
                ) else {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "no MIR body available".to_string(),
                    );
                    return;
                };
                match convert_body_isolated(&mut *conv, &ibody) {
                    Ok(converted) => {
                        *taken += 1;
                        converted_bodies.push(ibody.clone());
                        let counts =
                            (ibody.blocks.len(), ibody.locals().len(), ibody.arg_locals().len());
                        let display = if plain_taken.insert(referenced) {
                            base_name.clone()
                        } else {
                            format!("{} [{}]", base_name, inst.name())
                        };
                        bodies.push((display, converted));
                        body_def_ids.push(collected_id);
                        body_def_names.push(base_name.clone());
                        let demand = target_demands.entry(referenced).or_default();
                        demand.converted += 1;
                        if demand.first_counts.is_none() {
                            demand.first_counts = Some(counts);
                        }
                    }
                    Err(errs) => {
                        for e in errs {
                            fail_all(&mut *target_demands, &owners, e);
                        }
                    }
                }
            }));
            if outcome.is_err() {
                // Contained panic: clear the capture slot, as the
                // other demand passes do.
                let _ = take_captured_panic();
            }
        }
    }

    converted_bodies
}

/// Convert the plain functions that converted bodies
/// materialize as *values* out of constants: the counterpart of
/// [`collect_const_closure_instances`] for function items.
///
/// A function pointer stored inside a constant allocation - a
/// `RawWakerVTable`'s clone / wake / wake_by_ref / drop entries
/// are the canonical case - names a body no converted body's
/// call operands mention: the call goes through the decoded
/// pointer value at run time, and the operand-level `FnDef`
/// scan ([`sparagmos_driver::reach::fndef_constants`]) only sees
/// constants whose *type* is the function item, never pointers
/// nested inside an aggregate allocation. The decoded value is
/// the only mention, recorded by conversion itself (see
/// `Converter::const_fn_mentions`), and constant evaluation has
/// already resolved each mention to its monomorphic instance,
/// so this pass converts exactly those instances. Routing and
/// accounting mirror the constant-closure pass: a deferred
/// (generic-context) local definition is accounted in the
/// shared demand state, a local definition outside a generic
/// context is the item loop's business and is skipped, and a
/// function defined in another crate is accounted in the
/// state's foreign map (see [`apply_foreign_closure_demands`]).
fn collect_const_fn_instances(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    deferred: &[DeferredFn],
    local_defs: &HashSet<rustc_public::DefId>,
    collected: &HashSet<sparagmos_mir::DefId>,
    state: &mut InstanceDemandState,
) -> Vec<rustc_public::mir::Body> {
    let InstanceDemandState {
        seen,
        unresolved_seen: _,
        plain_taken,
        taken,
        demands,
        foreign_closure_demands: _,
        foreign_fn_demands,
        recursion_limit,
    } = state;

    let deferred_names: HashMap<rustc_public::DefId, &str> =
        deferred.iter().map(|d| (d.def, d.name.as_str())).collect();
    let mut converted_bodies: Vec<rustc_public::mir::Body> = Vec::new();

    // Converting one mentioned function's body decodes its own
    // constants, which can mention further functions; drain the
    // converter's record until a round adds nothing, exactly as
    // the constant-closure pass drains its record.
    loop {
        let mentions = std::mem::take(&mut conv.const_fn_mentions);
        if mentions.is_empty() {
            break;
        }
        for inst in &mentions {
            let referenced = inst.def.def_id();
            let (base_name, foreign) = match deferred_names.get(&referenced) {
                Some(&name) => (name.to_string(), false),
                None => {
                    if local_defs.contains(&referenced) {
                        continue;
                    }
                    (referenced.name(), true)
                }
            };
            let args = inst.args();
            if args_mention_params(&args) {
                continue;
            }
            // Body access goes through the compiler's internals
            // and can panic on exotic targets; isolate each
            // mention exactly as the other demand passes isolate
            // each scanned body.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let target_demands: &mut HashMap<rustc_public::DefId, Demand> = if foreign {
                    &mut *foreign_fn_demands
                } else {
                    &mut *demands
                };
                let owners = [referenced];
                // The expanded-size cap, as in the other demand
                // passes. A foreign function's oversized
                // arguments fail only its own demand: the
                // instance comes from a constant the compiler
                // already monomorphized, so its size is finite
                // by construction.
                if !sparagmos_driver::reach::generic_args_within_expanded_size(
                    &args,
                    sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE,
                ) {
                    if !foreign && recursion_limit.is_none() {
                        *recursion_limit = Some(format!(
                            "reached the recursion limit while instantiating `{}` (instantiation size limit)",
                            base_name
                        ));
                    }
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "recursion limit reached".to_string(),
                    );
                    return;
                }
                if !seen.insert((referenced, *inst)) {
                    return;
                }
                // Already converted by the reachability
                // collection: a second copy under the same
                // program DefId would collide when the program
                // is built.
                let collected_id = conv.instance_def_id(referenced, inst);
                if collected.contains(&collected_id) {
                    return;
                }
                if *taken >= MAX_LOCAL_INSTANCES {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "collection limit reached".to_string(),
                    );
                    return;
                }
                // Mirror the instance demand pass's guard before
                // asking for the body: a virtual instance has no
                // MIR body, and asking the compiler for one
                // panics inside it.
                if matches!(inst.kind, rustc_public::mir::mono::InstanceKind::Virtual { .. }) {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "dynamic dispatch target (virtual instance)".to_string(),
                    );
                    return;
                }
                if !inst.has_body() {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "no MIR body available".to_string(),
                    );
                    return;
                }
                let Some(ibody) = sparagmos_driver::convert::with_body_materialization(
                    &inst.name(),
                    || inst.body(),
                ) else {
                    fail_all(
                        &mut *target_demands,
                        &owners,
                        "no MIR body available".to_string(),
                    );
                    return;
                };
                match convert_body_isolated(&mut *conv, &ibody) {
                    Ok(converted) => {
                        *taken += 1;
                        converted_bodies.push(ibody.clone());
                        let counts =
                            (ibody.blocks.len(), ibody.locals().len(), ibody.arg_locals().len());
                        let display = if plain_taken.insert(referenced) {
                            base_name.clone()
                        } else {
                            format!("{} [{}]", base_name, inst.name())
                        };
                        bodies.push((display, converted));
                        body_def_ids.push(collected_id);
                        body_def_names.push(base_name.clone());
                        let demand = target_demands.entry(referenced).or_default();
                        demand.converted += 1;
                        if demand.first_counts.is_none() {
                            demand.first_counts = Some(counts);
                        }
                    }
                    Err(errs) => {
                        for e in errs {
                            fail_all(&mut *target_demands, &owners, e);
                        }
                    }
                }
            }));
            if outcome.is_err() {
                // Contained panic: clear the capture slot, as the
                // other demand passes do.
                let _ = take_captured_panic();
            }
        }
    }

    converted_bodies
}

/// Convert the demanded instances of the local coroutines
/// (async blocks, `gen` blocks, and the futures of `async fn`s
/// defined as separate items) that converted bodies create or
/// hold. A coroutine item has no monomorphic instance as an item
/// — `Instance::try_from` rejects it like a closure — so the item
/// loop defers it with the generic functions, but neither the
/// instance demand (which follows call targets and `FnDef`
/// constants) nor the closure demand (which follows closure
/// types) ever mentions it: a coroutine is demanded by the
/// `AggregateKind::Coroutine` rvalue that creates it and by the
/// coroutine-typed locals that hold it. This pass scans for
/// exactly those mentions ([`sparagmos_driver::reach::coroutine_targets`])
/// and converts each distinct demanded instance, recording the
/// outcome in the shared demand state so
/// [`apply_instance_demands`] accounts the deferred item.
///
/// The body comes from resolving the coroutine definition as an
/// instance (the monomorphization the compiler itself performs
/// for the creation site); when resolution refuses the
/// definition, the definition's own body is used — the arguments
/// are concrete here, so that body is already the monomorphic
/// one. Converted bodies reference a coroutine by its plain
/// definition key (the key the creation aggregate converts
/// through), so the first demanded instance's body is registered
/// under that key; a further instance of the same definition
/// counts as a converted demand without registering a second
/// body, the same one-body-per-definition shape the aggregate
/// conversion itself assumes.
fn collect_coroutine_instances(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    deferred: &[DeferredFn],
    seeds: &[rustc_public::mir::Body],
    state: &mut InstanceDemandState,
) -> Vec<rustc_public::mir::Body> {
    use std::collections::VecDeque;

    let InstanceDemandState {
        seen,
        unresolved_seen,
        plain_taken,
        taken,
        demands,
        foreign_closure_demands: _,
        foreign_fn_demands: _,
        recursion_limit,
    } = state;

    let deferred_names: HashMap<rustc_public::DefId, &str> =
        deferred.iter().map(|d| (d.def, d.name.as_str())).collect();
    let mut queue: VecDeque<rustc_public::mir::Body> = seeds.iter().cloned().collect();
    let mut extra_seeds: Vec<rustc_public::mir::Body> = Vec::new();

    while let Some(body) = queue.pop_front() {
        // Instance resolution and body access go through the
        // compiler's internals and can panic on exotic targets;
        // isolate each scanned body exactly as the other demand
        // passes do.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for (def, args) in sparagmos_driver::reach::coroutine_targets(&body) {
                let referenced = def.def_id();
                let Some(&base_name) = deferred_names.get(&referenced) else {
                    continue;
                };
                if coroutine_closure_args(&args) {
                    // An async closure demanded as a coroutine
                    // target: it has no instance of its own (see
                    // the closure collection), so there is no
                    // demand to record for its definition; the
                    // coroutine body inside it is a separate
                    // definition, collected by the reachability
                    // pass under its own name.
                    continue;
                }
                if args_mention_params(&args) {
                    continue;
                }
                let owners = [referenced];
                // The expanded-size cap, as in the function
                // demand pass: a coroutine demanded with
                // size-increasing arguments is the same
                // recursion-limit situation.
                if !sparagmos_driver::reach::generic_args_within_expanded_size(
                    &args,
                    sparagmos_driver::reach::MAX_INSTANTIATION_EXPANDED_SIZE,
                ) {
                    if recursion_limit.is_none() {
                        *recursion_limit = Some(format!(
                            "reached the recursion limit while instantiating `{}` (instantiation size limit)",
                            base_name
                        ));
                    }
                    fail_all(&mut *demands, &owners, "recursion limit reached".to_string());
                    continue;
                }
                // Obtain the instance body: resolution first,
                // the definition's own body as the fallback.
                let ibody = match Instance::resolve(
                    rustc_public::ty::FnDef(referenced),
                    &args,
                ) {
                    Ok(inst) => {
                        if !seen.insert((referenced, inst)) {
                            continue;
                        }
                        if !inst.has_body() {
                            fail_all(
                                &mut *demands,
                                &owners,
                                "no MIR body available".to_string(),
                            );
                            continue;
                        }
                        // As in `collect_local_instances`: a
                        // compiler-generated impl body over a
                        // SIMD type must not be asked for
                        // (MCP#838 validation ICE).
                        if generated_simd_impl_method(&inst) {
                            fail_all(
                                &mut *demands,
                                &owners,
                                "compiler-generated impl body over a SIMD type"
                                    .to_string(),
                            );
                            continue;
                        }
                        fetch_instance_body(&inst)
                    }
                    Err(_) => {
                        if !unresolved_seen.insert((referenced, format!("{args:?}"))) {
                            continue;
                        }
                        guard_stable_mir_query(&def.name(), || def.body())
                    }
                };
                let ibody = match ibody {
                    Ok(Some(b)) => b,
                    Ok(None) => {
                        fail_all(
                            &mut *demands,
                            &owners,
                            "no MIR body available".to_string(),
                        );
                        continue;
                    }
                    Err(reason) => {
                        fail_all(&mut *demands, &owners, reason);
                        continue;
                    }
                };
                if *taken >= MAX_LOCAL_INSTANCES {
                    fail_all(
                        &mut *demands,
                        &owners,
                        "collection limit reached".to_string(),
                    );
                    continue;
                }
                match convert_body_isolated(conv, &ibody) {
                    Ok(converted) => {
                        *taken += 1;
                        queue.push_back(ibody.clone());
                        extra_seeds.push(ibody.clone());
                        let counts =
                            (ibody.blocks.len(), ibody.locals().len(), ibody.arg_locals().len());
                        let demand = demands.entry(referenced).or_default();
                        demand.converted += 1;
                        if demand.first_counts.is_none() {
                            demand.first_counts = Some(counts);
                        }
                        // Register the body under the plain
                        // definition key the creation aggregates
                        // reference, unless an earlier instance of
                        // this definition already registered one.
                        let plain = conv.crate_def_id(&def);
                        if !body_def_ids.contains(&plain) {
                            let display = if plain_taken.insert(referenced) {
                                base_name.to_string()
                            } else {
                                format!("{} [coroutine]", base_name)
                            };
                            bodies.push((display, converted));
                            body_def_ids.push(plain);
                            body_def_names.push(base_name.to_string());
                        }
                    }
                    Err(errs) => {
                        for e in errs {
                            fail_all(&mut *demands, &owners, e);
                        }
                    }
                }
            }
        }));
        if outcome.is_err() {
            // Contained panic: clear the capture slot, as the
            // other demand passes do.
            let _ = take_captured_panic();
        }
    }

    extra_seeds
}

/// Record the demand outcomes accumulated by
/// [`collect_local_instances`] in the report: a deferred
/// definition is converted when every instance demanded of it
/// converted; when some instance could not be converted it stays
/// skipped with that instance's reasons. A definition nothing
/// demands is accounted [`FnOutcome::NotDemanded`]: no converted
/// body reaches it, so it is not part of the program — rustc's
/// monomorphization collector never instantiates it either —
/// and its lack of an instance does not make the conversion
/// incomplete. The record itself stays, under the item loop's
/// original no-instance reason; it is never counted converted.
/// One exception: a *constructor* nothing demands is accounted
/// converted without a body — its direct uses are aggregate
/// rvalues in the callers (already converted with them), and the
/// constructor item has no body in any monomorphization that was
/// not demanded, so there is nothing left unconverted (the same
/// accounting as a trivial constant).
fn apply_instance_demands(
    report: &mut ConversionReport,
    deferred: &[DeferredFn],
    demands: &HashMap<rustc_public::DefId, Demand>,
) {
    for dfn in deferred {
        let Some(demand) = demands.get(&dfn.def) else {
            let entry = &mut report.functions[dfn.report_idx];
            if dfn.ctor {
                entry.outcome = FnOutcome::Converted;
                report.bodyless_converted += 1;
            } else {
                let reason = match &entry.outcome {
                    FnOutcome::Skipped { reasons } if reasons.len() == 1 => {
                        reasons.keys().next().expect("one reason").clone()
                    }
                    _ => "no monomorphic instance available".to_string(),
                };
                entry.outcome = FnOutcome::NotDemanded { reason };
            }
            continue;
        };
        let entry = &mut report.functions[dfn.report_idx];
        if demand.converted > 0 && demand.failures.is_empty() {
            let (blocks, locals, arg_count) = demand.first_counts.unwrap_or((0, 0, 0));
            entry.blocks = blocks;
            entry.locals = locals;
            entry.arg_count = arg_count;
            entry.outcome = FnOutcome::Converted;
        } else {
            entry.outcome = FnOutcome::Skipped {
                reasons: demand.failures.clone(),
            };
        }
    }
}

/// Record the demand outcomes accumulated for closures defined
/// in other crates in the report: one entry per closure
/// definition, with the accounting [`apply_instance_demands`]
/// gives a deferred local definition — converted when every
/// demanded instance converted, skipped with the accumulated
/// reasons otherwise. A foreign closure has no item-loop
/// placeholder entry (no item loop iterates foreign items), so
/// the entries are appended here, past the target-crate section
/// of the report, like the reachability collection's entries.
fn apply_foreign_closure_demands(
    report: &mut ConversionReport,
    demands: &HashMap<rustc_public::DefId, Demand>,
) {
    let mut defs: Vec<&rustc_public::DefId> = demands.keys().collect();
    defs.sort_by_key(|d| d.name());
    for def in defs {
        let demand = &demands[def];
        let converted = demand.converted > 0 && demand.failures.is_empty();
        let (blocks, locals, arg_count) = if converted {
            demand.first_counts.unwrap_or((0, 0, 0))
        } else {
            (0, 0, 0)
        };
        let outcome = if converted {
            FnOutcome::Converted
        } else {
            FnOutcome::Skipped {
                reasons: demand.failures.clone(),
            }
        };
        report.functions.push(FnReport {
            name: def.name(),
            blocks,
            locals,
            arg_count,
            outcome,
        });
    }
}

/// Register converted impl method bodies under the trait keys
/// their call sites resolve through (Issue #64).
///
/// A call to a trait method names the trait declaration in its
/// `FnDef` target, so body conversion keys the call by
/// `(trait definition, resolved instance)`. The item loop instead
/// converts an impl method as an item of its own and keys it by
/// `(impl definition, instance)`. The two keys never meet, so a
/// trait call failed with "no converted body" even though the impl
/// body was in the program. (Generic impl methods escape this: the
/// demanded-instance collection keys them by the referenced
/// definition already, and the reachability collection does the
/// same for foreign bodies. The item loop is the only registrar
/// that keys by the resolved definition.)
///
/// This pass closes the gap the same way the rest of the driver
/// keys bodies — by the referenced definition: it scans the
/// converted bodies' Stable MIR call targets, and whenever a
/// target resolves to an instance whose body is present under the
/// resolved definition's key but not under the referenced
/// definition's key, it registers a copy of that body under the
/// referenced key. One pass suffices: the copy's own call targets
/// are the original's, already keyed the same way. Aliases carry
/// no report entries — the impl method's own entry already
/// accounts for the body; the alias is the same body under the
/// key call sites use.
/// Whether `name` (a definition name as the driver's name
/// table renders it) is one of `core::fmt`'s formatting trait
/// methods — the methods `core::fmt::rt::Argument`'s
/// constructors store as the formatting function.
fn is_fmt_trait_method(name: &str) -> bool {
    let Some(rest) = name
        .strip_prefix("std::fmt::")
        .or_else(|| name.strip_prefix("core::fmt::"))
    else {
        return false;
    };
    matches!(
        rest,
        "Display::fmt"
            | "Debug::fmt"
            | "Octal::fmt"
            | "LowerHex::fmt"
            | "UpperHex::fmt"
            | "Binary::fmt"
            | "LowerExp::fmt"
            | "UpperExp::fmt"
            | "Pointer::fmt"
    )
}

fn register_trait_key_aliases(
    conv: &mut Converter,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    sources: &[rustc_public::mir::Body],
) {
    use std::collections::{HashMap, HashSet};

    let mut present: HashMap<sparagmos_mir::DefId, usize> = body_def_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    let mut used_names: HashSet<String> = bodies.iter().map(|(n, _)| n.clone()).collect();

    for source in sources {
        // Instance resolution goes through the compiler's
        // internals and can panic on exotic targets; isolate each
        // scanned body exactly as the instance collection does, so
        // aliasing can never make a crate convert worse.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut additions: Vec<(String, Body, sparagmos_mir::DefId, String)> = Vec::new();
            let mut pending: HashSet<sparagmos_mir::DefId> = HashSet::new();
            for (def, args) in sparagmos_driver::reach::call_targets(source) {
                let referenced = def.def_id();
                let Ok(inst) = Instance::resolve(def, &args) else {
                    continue;
                };
                if matches!(inst.kind, rustc_public::mir::mono::InstanceKind::Virtual { .. }) {
                    continue;
                }
                // A call through an `Fn`-family trait whose Self
                // type is a closure resolves to the closure
                // instance itself. Its body must NOT be aliased
                // under the trait key: the call passes the
                // arguments as one RustCall tuple, while the
                // closure body takes them spread, and the
                // interpreter's trait-call path performs that
                // adaptation when it invokes the closure value.
                // A direct call to an alias copy would hand the
                // body the unspread tuple.
                if matches!(inst.ty().kind(), TyKind::RigidTy(RigidTy::Closure(..))) {
                    continue;
                }
                let resolved = inst.def.def_id();
                if resolved == referenced {
                    continue;
                }
                let call_key = conv.instance_def_id(referenced, &inst);
                if present.contains_key(&call_key) || !pending.insert(call_key) {
                    continue;
                }
                let impl_key = conv.instance_def_id(resolved, &inst);
                if impl_key == call_key {
                    continue;
                }
                let Some(&idx) = present.get(&impl_key) else {
                    continue;
                };
                let ref_name = def.name();
                let display = if used_names.insert(ref_name.clone()) {
                    ref_name.clone()
                } else {
                    let suffixed = format!("{} [{}]", ref_name, inst.name());
                    used_names.insert(suffixed.clone());
                    suffixed
                };
                additions.push((display, bodies[idx].1.clone(), call_key, ref_name));
            }
            // `core::fmt`'s argument constructors materialize
            // the formatting function as an `FnDef` constant
            // naming the trait method, never as a call target,
            // so the loop above never sees the demand. The
            // interpreter's fmt engine addresses the impl body
            // by its instance name (the name the item loop
            // records non-generic impls under), so register
            // that alias as well: the body may be keyed by the
            // referenced definition (the instance-demand pass
            // and the reachability collection) or by the
            // resolved one (the item loop).
            for (def, args) in sparagmos_driver::reach::fndef_constants(source) {
                let referenced_name = def.def_id().name();
                if !is_fmt_trait_method(&referenced_name) {
                    continue;
                }
                let Ok(inst) = Instance::resolve(def, &args) else {
                    continue;
                };
                if matches!(
                    inst.kind,
                    rustc_public::mir::mono::InstanceKind::Virtual { .. }
                ) {
                    continue;
                }
                let instance_name = inst.name().to_string();
                if used_names.contains(&instance_name) {
                    continue;
                }
                let call_key = conv.instance_def_id(def.def_id(), &inst);
                let impl_key = conv.instance_def_id(inst.def.def_id(), &inst);
                let Some(&idx) = present
                    .get(&call_key)
                    .or_else(|| present.get(&impl_key))
                else {
                    continue;
                };
                used_names.insert(instance_name.clone());
                additions.push((
                    instance_name,
                    bodies[idx].1.clone(),
                    call_key,
                    referenced_name,
                ));
            }
            additions
        }));
        match outcome {
            Ok(additions) => {
                for (display, body, call_key, ref_name) in additions {
                    present.insert(call_key, bodies.len());
                    bodies.push((display, body));
                    body_def_ids.push(call_key);
                    body_def_names.push(ref_name);
                }
            }
            Err(_) => {
                // Contained panic: clear the capture slot, as the
                // instance collection does — a later, genuinely
                // fatal panic must report its own message.
                let _ = take_captured_panic();
            }
        }
    }
}

/// Convert one local body (an item-loop body, or a constant body
/// obtained from the item itself) and record the outcome; a
/// successfully converted body also becomes a reachability seed.
#[allow(clippy::too_many_arguments)]
fn convert_local_body(
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
        _ => false,
    }
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
        Instance::resolve(*impl_fn, args)
            .ok()
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
        Some(impl_args) => Instance::resolve(*impl_fn, &impl_args)
            .ok()
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
fn demand_drop_instances(
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
fn demand_clone_instances(
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
            let resolved = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| {
                    Instance::resolve(def, &args)
                        .ok()
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
fn demand_wake_instances(
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
            let resolved = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| {
                    Instance::resolve(item, &args)
                        .ok()
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
fn certify_drop_no_impl(
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
fn certify_drop_trivial(
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

fn demand_dyn_instances(
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
    while let Some(body) = queue.pop_front() {
        // Vtable queries go through the compiler's internals and
        // can panic on exotic traits; isolate each scanned body
        // exactly as the other demand passes do, so this pass can
        // never make a crate convert worse.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut converted_now: Vec<rustc_public::mir::Body> = Vec::new();
            for tref in dyn_trait_refs(&body) {
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
fn account_trivial_const(report: &mut ConversionReport, name: String) {
    report.bodyless_converted += 1;
    report.functions.push(FnReport {
        name,
        blocks: 0,
        locals: 0,
        arg_count: 0,
        outcome: FnOutcome::Converted,
    });
}

/// The compiler callback, in the `run_with_tcx` form: the
/// context is installed for the conversion's span resolution
/// (see `sparagmos_driver::convert::set_compiler_tcx`) and
/// cleared on the way out, including on a panic.
fn collect_with_tcx(tcx: rustc_middle::ty::TyCtxt) -> ControlFlow<()> {
    struct ClearTcx;
    impl Drop for ClearTcx {
        fn drop(&mut self) {
            sparagmos_driver::convert::set_compiler_tcx(None);
        }
    }
    sparagmos_driver::convert::set_compiler_tcx(Some(tcx));
    let _clear = ClearTcx;
    collect()
}

fn collect() -> ControlFlow<()> {
    let mut conv = Converter::new();
    let mut report = ConversionReport::default();
    let mut bodies: Vec<(String, Body)> = Vec::new();
    let mut body_def_ids: Vec<sparagmos_mir::DefId> = Vec::new();
    let mut body_def_names: Vec<String> = Vec::new();
    let local_defs: std::collections::HashSet<rustc_public::DefId> =
        rustc_public::all_local_items()
            .iter()
            .map(|i| i.def_id())
            .collect();
    // Stable MIR bodies of the converted local functions; the
    // reachability collection walks their call graph afterwards.
    let mut seed_bodies: Vec<rustc_public::mir::Body> = Vec::new();

    // Generic function definitions deferred by the item loop; the
    // instance collection after the loop accounts for them.
    let mut deferred: Vec<DeferredFn> = Vec::new();
    // Closures defined in generic contexts, deferred the same way;
    // the closure collection after the loop accounts for them.
    let mut deferred_closures: Vec<DeferredFn> = Vec::new();

    let items = rustc_public::all_local_items();
    // Record the local statics' declared types before converting
    // anything: a reference into a static's interior is decoded
    // against the static's own type (see Converter::decode_static_ref),
    // and the type query is only safe for real items — an anonymous
    // promoted static's type_of can ICE the compilation — so it is
    // asked here, for items, never from the constant decoder.
    for item in items.iter() {
        if matches!(item.kind(), ItemKind::Static) {
            if let Ok(ty) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| item.ty()))
            {
                conv.note_static_ty(item, ty);
            }
        }
    }
    for item in items.iter() {
        let name = item.name();
        let item_instance = Instance::try_from(item.clone());
        // Assign (or fetch) this item's Sparagmos DefId up front, so a
        // converted body can be handed to the interpreter together with
        // the id that other bodies use to call it. Function items are
        // keyed by their resolved instance (the same key `FnDef` call
        // targets resolve through); other items keep the definition's
        // own id.
        let mut def_id = match &item_instance {
            Ok(inst) => conv.instance_def_id(item.def_id(), inst),
            Err(_) => conv.crate_def_id(item),
        };
        let instance = match item_instance {
            Ok(i) => i,
            Err(_) => match closure_instance(item) {
                ClosureAttempt::Instance(i) => {
                    // A closure body is keyed by its instance, like
                    // a function item: references to the closure
                    // type resolve through `closure_def_id` to the
                    // same key, and two instances of one closure
                    // definition stay distinct bodies.
                    def_id = conv.instance_def_id(item.def_id(), &i);
                    i
                }
                ClosureAttempt::GenericContext => {
                    // A closure defined inside a generic function,
                    // method or impl: no instance as an item, but
                    // monomorphic instances of the enclosing
                    // definition mention it with concrete arguments.
                    // Defer it like a generic function: a placeholder
                    // report entry keeps the target-crate accounting
                    // in place, and the closure collection after
                    // this loop fills in the outcome.
                    let mut reasons = BTreeMap::new();
                    reasons.insert(
                        "closure is defined in a generic context (its generic arguments \
                         still mention parameters); only monomorphic closures are converted"
                            .to_string(),
                        1,
                    );
                    deferred_closures.push(DeferredFn {
                        def: item.def_id(),
                        name: name.clone(),
                        report_idx: report.functions.len(),
                        ctor: false,
                    });
                    report.functions.push(FnReport {
                        name,
                        blocks: 0,
                        locals: 0,
                        arg_count: 0,
                        outcome: FnOutcome::Skipped { reasons },
                    });
                    continue;
                }
                ClosureAttempt::CoroutineClosure => {
                    // An async closure: its type cannot be asked
                    // (see `closure_instance`), so it has no
                    // instance as an item, but a converted body
                    // that creates it mentions it with concrete
                    // arguments in the coroutine-closure
                    // aggregate. Defer it like a generic-context
                    // closure: a placeholder report entry keeps
                    // the target-crate accounting in place, and
                    // the closure collection after this loop fills
                    // in the outcome.
                    let mut reasons = BTreeMap::new();
                    reasons.insert(COROUTINE_CLOSURE_TYPE_REASON.to_string(), 1);
                    deferred_closures.push(DeferredFn {
                        def: item.def_id(),
                        name: name.clone(),
                        report_idx: report.functions.len(),
                        ctor: false,
                    });
                    report.functions.push(FnReport {
                        name,
                        blocks: 0,
                        locals: 0,
                        arg_count: 0,
                        outcome: FnOutcome::Skipped { reasons },
                    });
                    continue;
                }
                ClosureAttempt::NotClosure => {
                    if matches!(item.kind(), ItemKind::Fn)
                        || matches!(
                            item.kind(),
                            ItemKind::Ctor(rustc_public::CtorKind::Fn)
                        )
                    {
                        // A generic function definition (or a
                        // generic data constructor): it has no
                        // monomorphic instance as an item, but its
                        // call sites may demand concrete instances.
                        // Defer it: a placeholder report entry keeps
                        // the target-crate accounting in place, and
                        // the instance collection after this loop
                        // fills in the outcome.
                        let mut reasons = BTreeMap::new();
                        reasons.insert(
                            format!(
                                "no monomorphic instance available (kind={:?})",
                                item.kind()
                            ),
                            1,
                        );
                        deferred.push(DeferredFn {
                            def: item.def_id(),
                            name: name.clone(),
                            report_idx: report.functions.len(),
                            ctor: matches!(item.kind(), ItemKind::Ctor(..)),
                        });
                        report.functions.push(FnReport {
                            name,
                            blocks: 0,
                            locals: 0,
                            arg_count: 0,
                            outcome: FnOutcome::Skipped { reasons },
                        });
                        continue;
                    }
                    if matches!(item.kind(), ItemKind::Const) {
                        // A constant that requires monomorphization
                        // as an item (an anonymous constant inside a
                        // generic function, an associated constant
                        // of a generic impl, ...): it has no
                        // instance, but the constant's own MIR body
                        // can still be converted, the way the
                        // demanded instances of a generic function
                        // are. Monomorphized bodies that use the
                        // constant carry its evaluated value, so the
                        // converted body is registered under the
                        // constant's own definition id — the key
                        // unevaluated constant references resolve
                        // through. A constant whose own body is
                        // unavailable too is a trivial constant
                        // (see `account_trivial_const`).
                        match guard_stable_mir_query(&item.name(), || item.body()) {
                            Ok(Some(body)) => {
                                let report_idx = report.functions.len();
                                convert_local_body(
                                    &mut conv,
                                    &mut report,
                                    &mut bodies,
                                    &mut body_def_ids,
                                    &mut body_def_names,
                                    &mut seed_bodies,
                                    name,
                                    def_id,
                                    body,
                                );
                                // A conversion that fails only on
                                // the constant's own parameters has
                                // met the generic definition itself,
                                // not a monomorphic body: a
                                // `ConstantKind::Param` in the body
                                // means the body's constants still
                                // mention the parameters of the
                                // definition (or of its enclosing
                                // item), which only substitution by
                                // monomorphization removes — and this
                                // constant has no instance as an item
                                // (that is why it reached this arm).
                                // Monomorphized bodies that use the
                                // constant carry its evaluated value,
                                // so no converted body needs this
                                // definition body, the way a generic
                                // function definition nothing demands
                                // is not part of the program either.
                                // Account it `NotDemanded`, keeping
                                // the parameter record; a failure
                                // with any other reason mixed in
                                // stays skipped, as a genuine gap.
                                if let Some(entry) =
                                    report.functions.get_mut(report_idx)
                                {
                                    let param_reasons = match &entry.outcome {
                                        FnOutcome::Skipped { reasons }
                                            if !reasons.is_empty()
                                                && reasons.keys().all(|r| {
                                                    r.starts_with(
                                                        "ConstantKind::Param(",
                                                    ) || r
                                                        == "ConstantKind::Bound"
                                                }) =>
                                        {
                                            Some(
                                                reasons
                                                    .iter()
                                                    .map(|(k, v)| {
                                                        format!("{k} x{v}")
                                                    })
                                                    .collect::<Vec<_>>()
                                                    .join(", "),
                                            )
                                        }
                                        _ => None,
                                    };
                                    if let Some(param_reasons) = param_reasons {
                                        entry.outcome =
                                            FnOutcome::NotDemanded {
                                                reason: format!(
                                                    "generic constant definition                                                      ({param_reasons}); only                                                      monomorphic instances are                                                      converted, and                                                      monomorphized bodies that                                                      use the constant carry                                                      its evaluated value"
                                                ),
                                            };
                                    }
                                }
                            }
                            Ok(None) => account_trivial_const(&mut report, name),
                            Err(reason) => skip_fn(&mut report, &name, reason),
                        }
                        continue;
                    }
                    skip_fn(
                        &mut report,
                        &name,
                        format!("no monomorphic instance available (kind={:?})", item.kind()),
                    );
                    continue;
                }
                ClosureAttempt::Unsupported(reason) => {
                    skip_fn(&mut report, &name, reason);
                    continue;
                }
            },
        };
        let body = match guard_stable_mir_query(&instance.name(), || instance.body()) {
            Ok(Some(body)) => body,
            Ok(None) => {
                if matches!(item.kind(), ItemKind::Const) {
                    // The constant has an instance but no body: a
                    // trivial constant (see `account_trivial_const`).
                    account_trivial_const(&mut report, name);
                    continue;
                }
                let mut reasons = BTreeMap::new();
                reasons.insert("no MIR body available".to_string(), 1);
                report.functions.push(FnReport {
                    name,
                    blocks: 0,
                    locals: 0,
                    arg_count: 0,
                    outcome: FnOutcome::Skipped { reasons },
                });
                continue;
            }
            Err(reason) => {
                skip_fn(&mut report, &name, reason);
                continue;
            }
        };
        convert_local_body(
            &mut conv,
            &mut report,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &mut seed_bodies,
            name,
            def_id,
            body,
        );
    }

    let local_body_count = bodies.len();
    let local_report_count = report.functions.len();

    // Demand monomorphic bodies for the `Drop` impls of the local
    // types the converted bodies drop. A generic `Drop` impl is
    // never instantiated by the item loop (it has no monomorphic
    // instance as an item) and never reached by the call-graph
    // collection (drop glue is not a call edge), so without this
    // pass the interpreter's glue builder finds the impl recorded
    // but bodiless and refuses the drop.
    let (mut drop_credited, mut drop_failures, mut dtor_bodies) = demand_drop_instances(
        &mut conv,
        &mut report,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &mut seed_bodies,
    );

    // Demand the `core`-generated `Clone` bodies for tuple and
    // array types, for the same reason: the trait method's name
    // is builtin-served, so the call-graph collection stops
    // there and the elements' own `Clone` impls would never run.
    demand_clone_instances(
        &mut conv,
        &mut report,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &mut seed_bodies,
    );

    // Demand the `Wake` / `LocalWake` impl methods of the types
    // the converted bodies build wakers from: a waker's wake
    // dispatch is modelled in the interpreter, which calls the
    // impl methods by name, and `wake_by_ref`'s default body is
    // an instance nothing else ever demands (no call site names
    // it, and the vtable that would mention it is built by the
    // modelled `From` conversion, whose body is not collected).
    demand_wake_instances(
        &mut conv,
        &mut report,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &mut seed_bodies,
    );

    // Convert the monomorphic instances of the generic local
    // function definitions that the converted bodies call; the
    // instance bodies join the reachability seeds below, so the
    // foreign functions they call are collected too.
    let mut demand_state = InstanceDemandState::default();
    let instance_seeds = collect_local_instances(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &deferred,
        &seed_bodies,
        &mut demand_state,
    );
    // The program DefIds and report names collected so far,
    // shared with every merge and demand pass below: the
    // reachability collection resolves some of the same closure
    // instances the closure demand passes do, and a body may
    // enter the program only once under its DefId (report
    // entries only once under their display name).
    let mut collected_ids: HashSet<sparagmos_mir::DefId> =
        body_def_ids.iter().copied().collect();
    let mut report_names: HashSet<String> =
        report.functions.iter().map(|f| f.name.clone()).collect();
    // Convert the demanded instances of closures defined in
    // generic contexts: the converted local bodies and the
    // demanded function instances mention them with concrete
    // arguments. The closure bodies join the reachability seeds,
    // so the foreign functions they call are collected too.
    let mut closure_scan = seed_bodies.clone();
    closure_scan.extend(instance_seeds.iter().cloned());
    let closure_seeds = collect_closure_instances(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &deferred_closures,
        &local_defs,
        &closure_scan,
        &collected_ids,
        &mut demand_state,
    );
    // Convert the demanded instances of the local coroutines
    // the converted bodies create or hold: a coroutine is
    // demanded by its creation aggregate and its coroutine-typed
    // locals, which neither the instance demand nor the closure
    // demand follows.
    let mut coroutine_scan = closure_scan;
    coroutine_scan.extend(closure_seeds.iter().cloned());
    let coroutine_seeds = collect_coroutine_instances(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &deferred,
        &coroutine_scan,
        &mut demand_state,
    );
    // Convert the closures the conversions so far materialized
    // as values out of constants (see
    // collect_const_closure_instances): their bodies join the
    // reachability seeds and the fixpoint scan set like the
    // closure pass's.
    let const_seeds = collect_const_closure_instances(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &deferred_closures,
        &local_defs,
        &collected_ids,
        &mut demand_state,
    );
    // And the plain functions the same conversions materialized
    // as values out of constants (see
    // collect_const_fn_instances): their bodies join the same
    // seed sets.
    let const_fn_seeds = collect_const_fn_instances(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &deferred,
        &local_defs,
        &collected_ids,
        &mut demand_state,
    );
    // Local bodies converted by the closure and coroutine passes
    // have not been scanned for the *function* instances they
    // demand (each demand pass scans only for its own kind of
    // target). The fixpoint loop below scans them with the
    // instance demand, and each round's newly converted closure
    // and coroutine bodies become the next round's scan set, so
    // every converted body is eventually scanned by every pass.
    let mut local_scan: Vec<rustc_public::mir::Body> = closure_seeds.clone();
    local_scan.extend(coroutine_seeds.iter().cloned());
    local_scan.extend(const_seeds.iter().cloned());
    local_scan.extend(const_fn_seeds.iter().cloned());
    let mut reach_seeds = seed_bodies;
    reach_seeds.extend(instance_seeds);
    reach_seeds.extend(closure_seeds);
    reach_seeds.extend(coroutine_seeds);
    reach_seeds.extend(const_seeds);
    reach_seeds.extend(const_fn_seeds);

    // Demand the vtable method bodies of the `dyn` values the
    // bodies scanned so far form (trait default methods reached
    // only through a vtable are the case this serves; see
    // `demand_dyn_instances`). The demanded bodies join the
    // reachability seeds, so the functions they call are
    // collected too. The state is shared with the fixpoint rounds
    // below, which rescan the bodies those rounds convert.
    let deferred_def_set: HashSet<rustc_public::DefId> =
        deferred.iter().map(|d| d.def).collect();
    let mut dyn_credited: HashMap<
        rustc_public::DefId,
        (usize, (usize, usize, usize)),
    > = HashMap::new();
    let mut dyn_demanded = 0usize;
    let dyn_seeds = demand_dyn_instances(
        &mut conv,
        &mut report,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &reach_seeds.clone(),
        &deferred_def_set,
        &mut dyn_credited,
        &mut dyn_demanded,
    );
    reach_seeds.extend(dyn_seeds);

    // Collect the bodies of non-local functions reachable from the
    // converted local bodies (std / core / alloc and other library
    // functions), so calls into them have bodies in the program too.
    let reached = sparagmos_driver::reach::collect_reachable(
        &mut conv,
        &reach_seeds,
        &local_defs,
        sparagmos_driver::reach::MAX_COLLECTED_BODIES,
    );
    collected_ids.extend(body_def_ids.iter().copied());
    let mut foreign_taken = 0usize;
    let mut frontier = reached.walked_bodies;
    for ((name, body), (def_id, def_name)) in reached
        .bodies
        .into_iter()
        .zip(reached.body_def_ids.into_iter().zip(reached.body_def_names))
    {
        if collected_ids.insert(def_id) {
            foreign_taken += 1;
            bodies.push((name, body));
            body_def_ids.push(def_id);
            body_def_names.push(def_name);
        }
    }
    for entry in reached.reports {
        if report_names.insert(entry.name.clone()) {
            report.functions.push(entry);
        }
    }

    // A generic local definition can also be demanded from inside
    // a collected foreign body (a std adapter calling a trait
    // method of a local generic type): the reachability collection
    // skips targets that resolve to local items, and the demand
    // pass above only walked local bodies, so such instances were
    // never converted. Alternate the two passes to a fixpoint:
    // scan the newly collected foreign bodies for demanded local
    // instances, then collect the foreign functions those instance
    // bodies call, whose bodies may in turn demand more local
    // instances. Later reachability passes re-walk (and
    // re-convert) targets earlier passes already collected; the
    // duplicates are dropped by DefId for bodies and by display
    // name for report entries (the sets above, which the initial
    // merge already used), and both collections keep their
    // per-program caps across passes (shared demand state, and a
    // remaining-budget argument for reachability).
    // Stable MIR sources for the trait-key alias pass below: the
    // local seeds plus every foreign body the reachability passes
    // walk, across all fixpoint rounds.
    let mut alias_sources = reach_seeds.clone();
    alias_sources.extend(frontier.iter().cloned());
    loop {
        let ids_before = body_def_ids.len();
        let mut fn_scan = frontier.clone();
        fn_scan.extend(local_scan.iter().cloned());
        let new_instances = collect_local_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred,
            &fn_scan,
            &mut demand_state,
        );
        // Newly collected foreign bodies and demanded instances
        // can also mention generic-context closures concretely.
        let mut closure_scan = frontier.clone();
        closure_scan.extend(new_instances.iter().cloned());
        let new_closure_bodies = collect_closure_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred_closures,
            &local_defs,
            &closure_scan,
            &collected_ids,
            &mut demand_state,
        );
        // Newly collected bodies can likewise create or hold
        // local coroutines concretely.
        let mut coroutine_scan = closure_scan;
        coroutine_scan.extend(new_closure_bodies.iter().cloned());
        let new_coroutine_bodies = collect_coroutine_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred,
            &coroutine_scan,
            &mut demand_state,
        );
        // Bodies converted or collected this round can form
        // `dyn` values whose vtables no earlier pass demanded
        // (a demanded generic instance unsizing its parameter's
        // concrete type is the common shape).
        let mut dyn_scan = fn_scan.clone();
        dyn_scan.extend(new_instances.iter().cloned());
        dyn_scan.extend(new_closure_bodies.iter().cloned());
        dyn_scan.extend(new_coroutine_bodies.iter().cloned());
        let new_dyn_bodies = demand_dyn_instances(
            &mut conv,
            &mut report,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &dyn_scan,
            &deferred_def_set,
            &mut dyn_credited,
            &mut dyn_demanded,
        );
        // Newly converted bodies also decode constants, which
        // can materialize further closures as values (the
        // `RandomState::new::KEYS` shape: a foreign constant's
        // function pointer naming a closure no type mentions).
        let new_const_bodies = collect_const_closure_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred_closures,
            &local_defs,
            &collected_ids,
            &mut demand_state,
        );
        let new_const_fn_bodies = collect_const_fn_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred,
            &local_defs,
            &collected_ids,
            &mut demand_state,
        );
        if new_instances.is_empty()
            && new_closure_bodies.is_empty()
            && new_coroutine_bodies.is_empty()
            && new_dyn_bodies.is_empty()
            && new_const_bodies.is_empty()
            && new_const_fn_bodies.is_empty()
        {
            break;
        }
        local_scan = new_closure_bodies.clone();
        local_scan.extend(new_coroutine_bodies.iter().cloned());
        local_scan.extend(new_dyn_bodies.iter().cloned());
        local_scan.extend(new_const_bodies.iter().cloned());
        local_scan.extend(new_const_fn_bodies.iter().cloned());
        for id in &body_def_ids[ids_before..] {
            collected_ids.insert(*id);
        }
        if foreign_taken >= sparagmos_driver::reach::MAX_COLLECTED_BODIES {
            break;
        }
        let mut reach_input = new_instances;
        reach_input.extend(new_closure_bodies.iter().cloned());
        reach_input.extend(new_coroutine_bodies.iter().cloned());
        reach_input.extend(new_dyn_bodies.iter().cloned());
        reach_input.extend(new_const_bodies.iter().cloned());
        let reached = sparagmos_driver::reach::collect_reachable(
            &mut conv,
            &reach_input,
            &local_defs,
            sparagmos_driver::reach::MAX_COLLECTED_BODIES - foreign_taken,
        );
        frontier = reached.walked_bodies;
        alias_sources.extend(frontier.iter().cloned());
        alias_sources.extend(new_closure_bodies.iter().cloned());
        alias_sources.extend(new_coroutine_bodies.iter().cloned());
        alias_sources.extend(new_dyn_bodies.iter().cloned());
        alias_sources.extend(new_const_bodies.iter().cloned());
        for ((name, body), (def_id, def_name)) in reached
            .bodies
            .into_iter()
            .zip(reached.body_def_ids.into_iter().zip(reached.body_def_names))
        {
            if collected_ids.insert(def_id) {
                foreign_taken += 1;
                bodies.push((name, body));
                body_def_ids.push(def_id);
                body_def_names.push(def_name);
            }
        }
        for entry in reached.reports {
            if report_names.insert(entry.name.clone()) {
                report.functions.push(entry);
            }
        }
    }
    // The reachability conversions of the final round above can
    // record further constant-content closure mentions (their
    // conversions decode new constants), so drain that demand to
    // a fixpoint as well: demand what the mentions name, walk
    // what it converts for the foreign functions those closures
    // call, and repeat until a round converts nothing.
    loop {
        let ids_before = body_def_ids.len();
        let new_const_bodies = collect_const_closure_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred_closures,
            &local_defs,
            &collected_ids,
            &mut demand_state,
        );
        let new_const_fn_bodies = collect_const_fn_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred,
            &local_defs,
            &collected_ids,
            &mut demand_state,
        );
        if new_const_bodies.is_empty() && new_const_fn_bodies.is_empty() {
            break;
        }
        for id in &body_def_ids[ids_before..] {
            collected_ids.insert(*id);
        }
        alias_sources.extend(new_const_bodies.iter().cloned());
        alias_sources.extend(new_const_fn_bodies.iter().cloned());
        if foreign_taken >= sparagmos_driver::reach::MAX_COLLECTED_BODIES {
            break;
        }
        let mut walk_seeds = new_const_bodies.clone();
        walk_seeds.extend(new_const_fn_bodies.iter().cloned());
        let reached = sparagmos_driver::reach::collect_reachable(
            &mut conv,
            &walk_seeds,
            &local_defs,
            sparagmos_driver::reach::MAX_COLLECTED_BODIES - foreign_taken,
        );
        alias_sources.extend(reached.walked_bodies.iter().cloned());
        for ((name, body), (def_id, def_name)) in reached
            .bodies
            .into_iter()
            .zip(reached.body_def_ids.into_iter().zip(reached.body_def_names))
        {
            if collected_ids.insert(def_id) {
                foreign_taken += 1;
                bodies.push((name, body));
                body_def_ids.push(def_id);
                body_def_names.push(def_name);
            }
        }
        for entry in reached.reports {
            if report_names.insert(entry.name.clone()) {
                report.functions.push(entry);
            }
        }
    }
    // Second drop-demand round over the full collected body
    // set: the first round ran before the reachability
    // collection, so it only saw the entry crate's bodies and
    // the type trees of the types they drop. Foreign bodies
    // collected by reachability drop foreign machinery of
    // their own (hashbrown's allocation guards are the
    // motivating case), and those drops need the same
    // destructor bodies. Instances the first round already
    // converted are skipped by identity inside the pass.
    {
        let mut all_bodies = alias_sources.clone();
        // The destructor bodies this round converts call
        // functions of their own (the array clone guard's drop
        // calls a slice-index precondition check, for one), and
        // drop glue is not a call edge, so no earlier
        // reachability wave walked them: walk each newly
        // converted destructor body's call edges, merge what
        // they reach, and demand drops again over the enlarged
        // set, until a round converts no new destructor body.
        loop {
            let seeds_before = all_bodies.len();
            let bodies_before = bodies.len();
            let (credited2, failures2, dtor_bodies2) = demand_drop_instances(
                &mut conv,
                &mut report,
                &mut bodies,
                &mut body_def_ids,
                &mut body_def_names,
                &mut all_bodies,
            );
            for (def, (count, counts)) in credited2 {
                let slot = drop_credited.entry(def).or_insert((0, counts));
                slot.0 += count;
            }
            for (def, reasons) in failures2 {
                let slot = drop_failures.entry(def).or_default();
                for (k, v) in reasons {
                    *slot.entry(k).or_default() += v;
                }
            }
            dtor_bodies.extend(dtor_bodies2);
            if bodies.len() == bodies_before {
                break;
            }
            if foreign_taken >= sparagmos_driver::reach::MAX_COLLECTED_BODIES {
                break;
            }
            let walk_seeds: Vec<rustc_public::mir::Body> =
                all_bodies[seeds_before..].to_vec();
            let reached = sparagmos_driver::reach::collect_reachable(
                &mut conv,
                &walk_seeds,
                &local_defs,
                sparagmos_driver::reach::MAX_COLLECTED_BODIES - foreign_taken,
            );
            alias_sources.extend(reached.walked_bodies.iter().cloned());
            all_bodies.extend(reached.walked_bodies.iter().cloned());
            for ((name, body), (def_id, def_name)) in reached
                .bodies
                .into_iter()
                .zip(reached.body_def_ids.into_iter().zip(reached.body_def_names))
            {
                if collected_ids.insert(def_id) {
                    foreign_taken += 1;
                    bodies.push((name, body));
                    body_def_ids.push(def_id);
                    body_def_names.push(def_name);
                }
            }
            for entry in reached.reports {
                if report_names.insert(entry.name.clone()) {
                    report.functions.push(entry);
                }
            }
        }
    }
    // Credit the vtable instances the dyn demand converted for
    // deferred definitions (a trait default method is the common
    // case): the "dynamic dispatch target (virtual instance)"
    // failure the demand passes recorded for such a definition is
    // served by those bodies, so it no longer stands against it.
    // Any other failure of the same definition is untouched, and
    // a definition no vtable instance was converted for keeps the
    // failure — its dyn calls stay unsupported at run time.
    for (def, (count, counts)) in &dyn_credited {
        let demand = demand_state.demands.entry(*def).or_default();
        demand.converted += count;
        if demand.first_counts.is_none() {
            demand.first_counts = Some(*counts);
        }
        demand
            .failures
            .remove("dynamic dispatch target (virtual instance)");
    }
    // A deferred trait method with a default body whose only
    // demand failure is the virtual-instance one was never
    // demanded as a monomorphic body: the demand came from
    // dynamic-dispatch sites in generic bodies, and no vtable
    // formation anywhere in the program names a concrete `Self`
    // for it (a formed vtable would have credited the definition
    // above). The default body is demandable in principle, so
    // nothing convertible was left unconverted; drop the demand
    // entry and the definition is accounted like an un-demanded
    // one. A definition with any other failure, one an instance
    // of which did convert, or a trait method without a default
    // body (no body of its own exists to demand) keeps its
    // accounting.
    for dfn in &deferred {
        let Some(demand) = demand_state.demands.get(&dfn.def) else {
            continue;
        };
        if demand.converted > 0
            || demand.failures.len() != 1
            || !demand
                .failures
                .contains_key("dynamic dispatch target (virtual instance)")
        {
            continue;
        }
        let has_default_body = guard_stable_mir_query("", || {
            rustc_public::ty::FnDef(dfn.def).has_body()
        })
        .unwrap_or(false);
        if has_default_body {
            demand_state.demands.remove(&dfn.def);
        }
    }
    // Credit the destructor instances the drop demand converted
    // to their impl definitions (and record its failures) the
    // same way: drop glue is not a call edge, so without this the
    // generic `Drop` impl's placeholder entry would stand against
    // the conversion even when the demanded instance converted.
    for (def, (count, counts)) in &drop_credited {
        let demand = demand_state.demands.entry(*def).or_default();
        demand.converted += count;
        if demand.first_counts.is_none() {
            demand.first_counts = Some(*counts);
        }
    }
    for (def, reasons) in &drop_failures {
        let demand = demand_state.demands.entry(*def).or_default();
        for (k, v) in reasons {
            *demand.failures.entry(k.clone()).or_default() += v;
        }
    }
    apply_instance_demands(&mut report, &deferred, &demand_state.demands);
    apply_instance_demands(&mut report, &deferred_closures, &demand_state.demands);
    apply_foreign_closure_demands(&mut report, &demand_state.foreign_closure_demands);
    apply_foreign_closure_demands(&mut report, &demand_state.foreign_fn_demands);

    // Register converted impl method bodies under the trait keys
    // their call sites resolve through (Issue #64). Demands can
    // come from any converted body: local items, demanded local
    // instances, and collected foreign bodies alike.
    register_trait_key_aliases(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &alias_sources,
    );

    // Auxiliary crates (--aux): the main compilation links them as
    // dependencies, so their functions live in this same Stable MIR
    // session and share the converter's DefId assignment with the
    // local items. Convert them into the same program, after the
    // local functions. A generic auxiliary function has no
    // monomorphic body to convert and is recorded as skipped, the
    // same treatment local generic functions get. Genericity is
    // checked up front from the function's generic parameters (any
    // type or const parameter, its own or an enclosing
    // definition's): resolving a generic function to an instance
    // with empty generic arguments hands back its polymorphic body
    // and panics inside the compiler when it is instantiated.
    for crate_name in aux_crates() {
        for krate in rustc_public::find_crates(crate_name) {
            // An auxiliary crate can share the main crate's name
            // (see the E0519 note in `main`); `find_crates` then
            // also returns the local crate, whose functions the loop
            // above already collected.
            if krate.is_local {
                continue;
            }
            for fn_def in krate.fn_defs() {
                let name = fn_def.name();
                if bodies.iter().any(|(n, _)| n == &name) {
                    // Already converted (the reachability
                    // collection above picks up auxiliary
                    // functions called from the main crate).
                    continue;
                }
                let def_id = conv.crate_def_id(&fn_def);
                conv.note_fn_abi_for_def(def_id, &fn_def);
                conv.note_link_symbol_for_def(def_id, fn_def.def_id());
                let generic = fn_def.generics_of().params.iter().any(|param| {
                    matches!(
                        param.kind,
                        rustc_public::ty::GenericParamDefKind::Type { .. }
                            | rustc_public::ty::GenericParamDefKind::Const { .. }
                    )
                });
                let body = if generic {
                    Err("no monomorphic instance available \
                         (auxiliary crate function)"
                        .to_string())
                } else {
                    guard_stable_mir_query(&fn_def.name(), || fn_def.body())
                };
                convert_item(
                    &mut conv,
                    &mut report,
                    &mut bodies,
                    &mut body_def_ids, &mut body_def_names,
                    name,
                    def_id,
                    body,
                );
            }
        }
    }



    // Monomorphization failures rustc reports around MIR
    // conversion: the recursion limit the demand passes hit while
    // collecting instances (a mono-collection error, so it takes
    // precedence), the required-constant evaluation the mono
    // collector performs over the instances it collects
    // (also a collection-time error, so it comes next), and the
    // checks of `sparagmos_driver::mono_check` over the converted
    // bodies (invalid SIMD intrinsic instantiations and SIMD
    // layouts, types too big for the target architecture). A
    // failure here fails the whole compilation, the way the
    // corresponding rustc error does. The body scan covers the
    // Stable MIR bodies conversion walked (local seeds, demanded
    // instances, and the reachability frontier); auxiliary-crate
    // bodies converted by the loop above are not retained in that
    // set and are not scanned.
    let compile_error = demand_state
        .recursion_limit
        .clone()
        .or_else(|| {
            if !COMPILE_VERDICT_MODE.load(std::sync::atomic::Ordering::Relaxed) {
                return None;
            }
            sparagmos_driver::mono_check::check_required_consts()
        })
        .or_else(|| sparagmos_driver::mono_check::check_bodies(&alias_sources));

    let entry_def_id = entry_point_def_id(&mut conv, &items);

    // Certify the dropped types whose drop glue the compiler
    // knows is empty, over the same body set the monomorphization
    // checks scan: the interpreter's glue builder consults the
    // set for the foreign types its name table cannot rule a
    // destructor out for.
    let drop_trivial_adts = certify_drop_trivial(&mut conv, &alias_sources);
    let drop_no_impl_adts = certify_drop_no_impl(&mut conv, &alias_sources);

    stash(Collected {
        report,
        bodies,
        body_def_ids,
        body_def_names,
        fn_abis: conv.fn_abis.clone(),
        local_body_count,
        local_report_count,
        entry_def_id,
        def_names: conv.def_names.clone(),
        adt_discriminants: conv.adt_discriminants.clone(),
        compile_error,
        adt_layouts: conv.adt_layouts.clone(),
        adt_field_layouts: conv.adt_field_layouts.clone(),
        tuple_layouts: conv.tuple_layouts.clone(),
        adt_tag_layouts: conv.adt_tag_layouts.clone(),
        coroutine_layouts: conv.coroutine_layouts.clone(),
        adt_shapes: conv.adt_shapes.clone(),
        adt_debug_shapes: conv.adt_debug_shapes.clone(),
        env_debug_shapes: conv.env_debug_shapes.clone(),
        thread_locals: conv.thread_locals.clone(),
        static_tys: conv.static_tys_mir.clone(),
        adt_instance_shapes: conv.adt_instance_shapes.clone(),
        adt_niches: conv.adt_niches.clone(),
        mat_plans: conv.mat_plans.clone(),
        drop_trivial_adts,
        drop_no_impl_adts,
        dtor_bodies,
        track_caller_defs: conv.track_caller_defs.clone(),
        fn_shim_locations: conv.fn_shim_locations.clone(),
        link_symbols: conv.link_symbols.clone(),
        source_files: conv.source_files.clone(),
    });
    ControlFlow::Continue(())
}

/// The Sparagmos `DefId` of the entry point rustc designated
/// for this crate, if it has one: the user's `main` in a binary
/// crate, the generated harness main in a `--test` crate. The
/// two share the definition name `<crate>::main` in a test
/// crate, so the entry cannot be recovered from names; it is
/// resolved here through the same instance the item loop
/// converted, to the id that body carries in
/// `Collected::body_def_ids`.
fn entry_point_def_id(
    conv: &mut Converter,
    items: &[rustc_public::CrateItem],
) -> Option<sparagmos_mir::DefId> {
    let entry = std::panic::catch_unwind(rustc_public::entry_fn).ok()??;
    let item = items.iter().find(|item| item.def_id() == entry.def_id())?;
    match Instance::try_from(item.clone()) {
        Ok(inst) => Some(conv.instance_def_id(item.def_id(), &inst)),
        Err(_) => None,
    }
}

fn is_main(name: &str) -> bool {
    name == "main" || name.ends_with("::main")
}

/// Names of `FnDef` call targets that have no converted body in this
/// program (library / core functions, `println!` machinery, ...), in
/// the driver's `def_names` spelling. Used to make run-mode failure
/// reasons concrete about *which* call could not be resolved.
fn external_callees(c: &Collected) -> Vec<String> {
    let defined: BTreeSet<&str> = c.body_def_names.iter().map(|n| n.as_str()).collect();
    let mut out: BTreeSet<String> = BTreeSet::new();
    for (_, body) in &c.bodies {
        for bb in &body.basic_blocks {
            let Some(term) = &bb.terminator else { continue };
            let sparagmos_mir::TerminatorKind::Call { func, .. } = &term.kind else {
                continue;
            };
            let sparagmos_mir::Operand::Constant(konst) = func else {
                continue;
            };
            let sparagmos_mir::ConstKind::Ty(sparagmos_mir::Ty::FnDef(def, _)) = &konst.literal
            else {
                continue;
            };
            if def.krate != 0 {
                continue;
            }
            if let Some(name) = c.def_names.get(def.index as usize) {
                if !defined.contains(name.as_str()) {
                    out.insert(name.clone());
                }
            }
        }
    }
    out.into_iter().collect()
}

/// Execute the collected program's `main` in the interpreter and
/// return the run-mode process exit code (see the crate docs).
/// Whether the entry body at `entry_index` returns
/// `std::process::ExitCode` (its local 0 is that ADT).
fn entry_returns_exit_code(c: &Collected, entry_index: usize) -> bool {
    let Some((_, body)) = c.bodies.get(entry_index) else {
        return false;
    };
    let Some(decl) = body.local_decls.first() else {
        return false;
    };
    let sparagmos_mir::Ty::Adt(def, _) = &decl.ty else {
        return false;
    };
    c.def_names
        .get(def.index as usize)
        // The compiler-generated test harness main names the
        // type by the path visible in its generated module
        // (`<crate>::test::ExitCode`), not the defining path;
        // the value shape checked by `exit_code_value` keeps
        // the match honest.
        .map(|n| n == "std::process::ExitCode" || n.ends_with("::ExitCode"))
        .unwrap_or(false)
}

/// The code byte of an `ExitCode` value: the wrapper aggregate
/// around the platform aggregate around the byte, the shape the
/// `ExitCode::from` builtin builds and the collected
/// `Termination::report` body passes through.
fn exit_code_value(value: &sparagmos_interp::Value) -> Option<i32> {
    let sparagmos_interp::Value::Enum { fields, .. } = value else {
        return None;
    };
    let [inner] = fields.as_slice() else {
        return None;
    };
    let sparagmos_interp::Value::Enum { fields, .. } = inner else {
        return None;
    };
    let [byte] = fields.as_slice() else {
        return None;
    };
    match byte {
        sparagmos_interp::Value::Uint(v) => Some(*v as u8 as i32),
        sparagmos_interp::Value::Int(v) => Some(*v as u8 as i32),
        _ => None,
    }
}

/// Whether the entry body at `entry_index` returns
/// `core::result::Result` (its local 0 is that ADT): a
/// `fn main() -> Result<_, _>`, whose `Termination::report`
/// the driver applies itself (see `execute`).
fn entry_returns_result(c: &Collected, entry_index: usize) -> bool {
    let Some((_, body)) = c.bodies.get(entry_index) else {
        return false;
    };
    let Some(decl) = body.local_decls.first() else {
        return false;
    };
    let sparagmos_mir::Ty::Adt(def, _) = &decl.ty else {
        return false;
    };
    c.def_names
        .get(def.index as usize)
        .map(|n| n == "core::result::Result" || n == "std::result::Result")
        .unwrap_or(false)
}

/// The outcome of applying `Termination::report` to a returned
/// `Result` value, following the standard library's own
/// delegation: `Err(e)` prints `Error: {e:?}` and exits with
/// `ExitCode::FAILURE`; `Ok(v)` reports exactly as `v`'s own
/// `Termination` does -- `()` succeeds silently, an `ExitCode`
/// payload yields its code, a nested `Result` recurses.
enum TerminationReport<'a> {
    /// The report ends the process with this code, printing
    /// nothing.
    Code(i32),
    /// The report prints `Error: {payload:?}` -- the payload,
    /// rendered under the type tag of the `Err` type at the
    /// level that produced it -- drops the payload, and exits
    /// with `ExitCode::FAILURE`. The `Err` type itself rides
    /// along so the executing interpreter can build the
    /// payload's drop glue from it.
    Error(
        &'a sparagmos_interp::Value,
        String,
        &'a sparagmos_mir::Ty,
    ),
}

/// `Termination::report` for the entry's returned value, walked
/// along the entry's declared return type `ty` (a
/// `core::result::Result` ADT; see [`entry_result_ty`]): the
/// value alone does not say which of its payloads are themselves
/// `Termination` values, the type does. `None` when the value
/// does not have the one-field-per-variant shape its type
/// declares, or when an `Ok` payload's type has no `Termination`
/// reading the driver models -- an unhandled payload is
/// classified unsupported, never reported as a success with the
/// wrong exit code.
fn result_report<'a>(
    c: &Collected,
    names: &std::collections::HashMap<sparagmos_mir::DefId, String>,
    ty: &'a sparagmos_mir::Ty,
    value: &'a sparagmos_interp::Value,
) -> Option<TerminationReport<'a>> {
    let sparagmos_mir::Ty::Adt(def, args) = ty else {
        return None;
    };
    let is_result = c
        .def_names
        .get(def.index as usize)
        .map(|n| n == "core::result::Result" || n == "std::result::Result")
        .unwrap_or(false);
    if !is_result {
        return None;
    }
    let sparagmos_interp::Value::Enum { variant, fields } = value else {
        return None;
    };
    let [payload] = fields.as_slice() else {
        return None;
    };
    match variant {
        1 => {
            let err_ty = args.get(1)?;
            Some(TerminationReport::Error(
                payload,
                sparagmos_interp::builtins::ty_tag(err_ty, names),
                err_ty,
            ))
        }
        0 => ok_payload_report(c, names, args.first()?, payload),
        _ => None,
    }
}

/// `Termination::report` for an `Ok` payload, delegated to the
/// payload type's own `Termination` impl, as
/// `Result<T, E>::report` delegates `Ok(value)` to
/// `T::report()`.
fn ok_payload_report<'a>(
    c: &Collected,
    names: &std::collections::HashMap<sparagmos_mir::DefId, String>,
    ty: &'a sparagmos_mir::Ty,
    value: &'a sparagmos_interp::Value,
) -> Option<TerminationReport<'a>> {
    match ty {
        // `()` reports success and prints nothing.
        sparagmos_mir::Ty::Tuple(ts) if ts.is_empty() => {
            matches!(value, sparagmos_interp::Value::Unit).then_some(TerminationReport::Code(0))
        }
        sparagmos_mir::Ty::Adt(def, _) => {
            let name = c.def_names.get(def.index as usize)?;
            if name == "std::process::ExitCode" || name.ends_with("::ExitCode") {
                exit_code_value(value).map(TerminationReport::Code)
            } else if name == "core::result::Result" || name == "std::result::Result" {
                result_report(c, names, ty, value)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The declared return type of the entry body at `entry_index`
/// (its local 0), when it is an ADT: the type the
/// [`result_report`] `Termination` walk follows.
fn entry_result_ty<'a>(c: &'a Collected, entry_index: usize) -> Option<&'a sparagmos_mir::Ty> {
    let (_, body) = c.bodies.get(entry_index)?;
    let decl = body.local_decls.first()?;
    let sparagmos_mir::Ty::Adt(..) = &decl.ty else {
        return None;
    };
    Some(&decl.ty)
}

/// The session's effective overflow-checks setting, resolved from
/// the final rustc argument list the way rustc resolves it
/// (`Session::overflow_checks`): the last `-C overflow-checks`
/// wins; without one, the last `-C debug-assertions`; without
/// either, checks are on exactly when the opt-level is 0 (`-O`
/// selects opt-level 2). The interpreter needs the resolved value
/// because an overflow `Assert` inside a
/// `#[rustc_inherit_overflow_checks]` body is present in the MIR
/// either way — dropping it when the session has checks off is a
/// codegen-time decision the driver must reproduce (see
/// `InterpConfig::overflow_checks`).
fn session_overflow_checks(args: &[String]) -> bool {
    // One pass over the arguments, tracking the last value of each
    // relevant option. A codegen option appears as `-C opt[=v]`,
    // `-Copt[=v]`, `--codegen opt[=v]` or `--codegen=opt[=v]`.
    let mut overflow: Option<Option<String>> = None;
    let mut debug_assertions: Option<Option<String>> = None;
    let mut opt_is_zero = true;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "-O" {
            opt_is_zero = false;
            i += 1;
            continue;
        }
        let text: Option<&str> = if a == "-C" || a == "--codegen" {
            i += 1;
            args.get(i).map(String::as_str)
        } else if let Some(rest) = a.strip_prefix("--codegen=") {
            Some(rest)
        } else if let Some(rest) = a.strip_prefix("-C") {
            Some(rest)
        } else {
            None
        };
        if let Some(t) = text {
            let (name, value) = match t.split_once('=') {
                Some((n, v)) => (n, Some(v.to_string())),
                None => (t, None),
            };
            match name {
                "overflow-checks" | "overflow_checks" => overflow = Some(value),
                "debug-assertions" | "debug_assertions" => debug_assertions = Some(value),
                "opt-level" => {
                    if let Some(v) = &value {
                        opt_is_zero = v == "0";
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    // A value-less boolean option counts as on; the spelled-out
    // negative values are the only off readings.
    let as_bool = |v: &Option<String>| !matches!(v.as_deref(), Some("off" | "no" | "false"));
    if let Some(v) = &overflow {
        return as_bool(v);
    }
    if let Some(v) = &debug_assertions {
        return as_bool(v);
    }
    opt_is_zero
}

/// The source tree one build converted: every file under
/// the crate root's directory at the moment of the build,
/// as (path relative to that directory, bytes) pairs, the
/// root itself among them under its file name. A crate is
/// more than its root file -- `mod` files and `include!`
/// inputs live beside it -- so an artifact denoting "the
/// program its build produced" must denote the tree, not
/// the root alone: rewriting only a module file after the
/// build must not change what an earlier artifact runs,
/// and rewriting or deleting the root must not orphan the
/// artifact while its modules survive in the snapshot.
type SourceTree = Vec<(std::path::PathBuf, Vec<u8>)>;

/// The bounds past which a directory is not read as a
/// crate's source tree: past them the snapshot keeps the
/// root alone (the root-only behaviour) rather than
/// copying a tree the build could not have read whole.
const SOURCE_TREE_MAX_FILES: usize = 4096;
const SOURCE_TREE_MAX_BYTES: usize = 64 << 20;
const SOURCE_TREE_MAX_DEPTH: usize = 32;

/// Read `path` holding at most `limit` bytes: `None` when
/// the file cannot be opened or read, or when it holds
/// more than `limit` bytes. The size stands checked from
/// the metadata before any byte is read, and the read
/// itself is capped at `limit + 1` bytes, so a file that
/// grows between the check and the read is caught by the
/// extra byte rather than allocated whole: a snapshot
/// read must never take in more than its bound, however
/// large a file in the tree stands on disk.
fn read_bounded(path: &std::path::Path, limit: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    if std::fs::metadata(path).ok()?.len() > limit as u64 {
        return None;
    }
    let file = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes).ok()?;
    if bytes.len() > limit {
        return None;
    }
    Some(bytes)
}

/// Snapshot the source tree rooted at `root` (see
/// `SourceTree`): the root's own bytes and, recursively,
/// every regular file under the root's directory, in
/// deterministic (sorted) order.
fn snapshot_source_tree(root: &std::path::Path) -> SourceTree {
    let root_name = root
        .file_name()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("source.rs"));
    let root_bytes = read_bounded(root, SOURCE_TREE_MAX_BYTES).unwrap_or_default();
    let mut tree: SourceTree = vec![(root_name, root_bytes)];
    if let Some(base) = root.parent() {
        let mut budget = (SOURCE_TREE_MAX_FILES, SOURCE_TREE_MAX_BYTES);
        if !collect_source_tree(base, base, 0, &mut tree, &mut budget) {
            // Over the bounds: keep the root alone.
            tree.truncate(1);
            return tree;
        }
    }
    let mut seen = std::collections::HashSet::new();
    tree.retain(|(rel, _)| seen.insert(rel.clone()));
    tree.sort();
    tree
}

/// The recursive walk under `snapshot_source_tree`; `false`
/// when the tree runs past the snapshot bounds.
fn collect_source_tree(
    base: &std::path::Path,
    dir: &std::path::Path,
    depth: usize,
    tree: &mut SourceTree,
    budget: &mut (usize, usize),
) -> bool {
    if depth > SOURCE_TREE_MAX_DEPTH {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        // An unreadable directory contributes nothing.
        return true;
    };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            if !collect_source_tree(base, &path, depth + 1, tree, budget) {
                return false;
            }
        } else if path.is_file() {
            if budget.0 == 0 {
                return false;
            }
            // The size stands checked before any read: a
            // file past the remaining byte budget ends the
            // walk without its bytes ever being allocated,
            // however large it stands on disk.
            match std::fs::metadata(&path) {
                Ok(meta) if meta.len() > budget.1 as u64 => return false,
                Ok(_) => {}
                Err(_) => continue,
            }
            // The read itself stays bounded by the same
            // budget, so a file that grows between the
            // check and the read ends the walk the same
            // way instead of being allocated whole; a file
            // that cannot be read at all leaves the walk
            // the same way, since a tree this walk cannot
            // read whole is not one it can snapshot.
            let Some(bytes) = read_bounded(&path, budget.1) else {
                return false;
            };
            budget.0 -= 1;
            budget.1 -= bytes.len();
            let Ok(rel) = path.strip_prefix(base) else {
                continue;
            };
            tree.push((rel.to_path_buf(), bytes));
        }
    }
    true
}

/// Whether the tree standing at `root`'s directory still
/// denotes the conversion the build made: every file of
/// `tree` holds the bytes the build converted, and no
/// source file has been added beside them. An addition
/// can change what a re-conversion reads even when every
/// recorded file stands unchanged -- a `mod helper;` the
/// build resolved to the recorded `helper.rs` gains a
/// second candidate when `helper/mod.rs` appears beside
/// it, and converting the live tree then fails with
/// E0761 although the program the build produced is
/// unchanged -- so an added `.rs` file anywhere in the
/// tree ends the intact reading too, and the run falls
/// back to the recorded snapshot (see
/// `tree_has_added_source_file`).
fn source_tree_intact(root: &std::path::Path, tree: &SourceTree) -> bool {
    let Some(base) = root.parent() else {
        return false;
    };
    if !tree.iter().all(|(rel, bytes)| {
        read_bounded(&base.join(rel), bytes.len()).is_some_and(|now| now == *bytes)
    }) {
        return false;
    }
    let recorded: std::collections::HashSet<&std::path::Path> =
        tree.iter().map(|(rel, _)| rel.as_path()).collect();
    let mut budget = SOURCE_TREE_MAX_FILES;
    !tree_has_added_source_file(base, base, 0, &recorded, &mut budget)
}

/// Whether a `.rs` file the recorded tree does not hold
/// has appeared anywhere under `dir` (recursing from
/// `base`): the addition scan behind `source_tree_intact`.
/// Only `.rs` files take part -- they are the files whose
/// presence steers source resolution (`mod` candidates
/// and `#[path]` targets). An added file of any other
/// kind cannot change what a conversion reads: a file
/// the build's conversion did read (an `include!` input,
/// say) stood in the tree at build time, and so stands
/// recorded. The scan gives up -- reports no addition --
/// once it has looked at more entries than the snapshot
/// bounds allow: a tree that large was never recorded
/// whole (see `snapshot_source_tree`), and its runs keep
/// the recorded-files-only reading rather than falling
/// back to a snapshot that does not hold the tree.
fn tree_has_added_source_file(
    base: &std::path::Path,
    dir: &std::path::Path,
    depth: usize,
    recorded: &std::collections::HashSet<&std::path::Path>,
    budget: &mut usize,
) -> bool {
    if depth > SOURCE_TREE_MAX_DEPTH {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        // An unreadable directory contributes nothing.
        return false;
    };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        let path = entry.path();
        if path.is_dir() {
            if tree_has_added_source_file(base, &path, depth + 1, recorded, budget) {
                return true;
            }
        } else if path.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
            let Ok(rel) = path.strip_prefix(base) else {
                continue;
            };
            if !recorded.contains(rel) {
                return true;
            }
        }
    }
    false
}

/// One artifact a tool invocation produced, as the registry
/// remembers it: the name it was built under, the source it was
/// converted from, and the attributes the producing invocation
/// carried -- which is what a later `--extern` naming the
/// artifact binds through `--aux`.
struct ToolArtifact {
    name: String,
    source: std::path::PathBuf,
    /// The source tree as it stood when the producing
    /// invocation converted it (see `SourceTree`): a recipe
    /// may rewrite or delete the source files after
    /// building, and a later run of the artifact must still
    /// execute the program its build produced (see
    /// `ToolRegistry::effective_source`).
    source_tree: SourceTree,
    /// The working directory of the producing invocation:
    /// the directory its relative compilation paths (a
    /// forwarded `--extern` path, a `-L` directory) were
    /// resolved against when it converted. A later run of
    /// the artifact re-converts under this directory, not
    /// the execution's, so those paths denote the same
    /// files at both times (see `artifact_run`).
    build_cwd: std::path::PathBuf,
    edition: Option<String>,
    crate_type: Option<String>,
    /// The compilation environment the producing invocation
    /// ran under (its request's environment assignments and
    /// clear flag): what a later run of the artifact (see
    /// `artifact_run`) spawns its re-conversion with, so
    /// compile-time environment reads (`option_env!`, `env!`)
    /// see the values the build saw, while the interpreted
    /// program itself runs under the execution request's
    /// environment.
    compile_env_clear: bool,
    compile_envs: Vec<(Vec<u8>, Option<Vec<u8>>)>,
    /// The `--aux` specs the producing invocation itself
    /// carried (its `--extern` bindings, already translated):
    /// what a later run of the artifact (see `artifact_run`)
    /// hands to the run subprocess so the artifact's source
    /// converts against the same libraries its build used.
    aux_specs: Vec<String>,
    /// The dependency version each entry of `aux_specs`
    /// bound when this artifact's build linked it, in the
    /// same order: the dependency artifact's source path
    /// and the source tree its build converted. A later run of
    /// this artifact resolves its `--aux` specs against
    /// these pins, not against the registry as it then
    /// stands: the registry replaces a rebuilt artifact's
    /// entry and keys nothing by version, so resolving by
    /// source path at execution time would hand the run
    /// whatever build of the dependency happens to be
    /// current then -- a library rebuilt after this
    /// artifact was, or another output built from the
    /// same source.
    aux_pins: Vec<(std::path::PathBuf, SourceTree)>,
    /// The compilation-shaping options the producing
    /// invocation carried (its forwarded rustc arguments --
    /// `--cfg`, `--check-cfg`, codegen and lint options, and
    /// the rest of that family): what a later run of the
    /// artifact (see `artifact_run`) replays through
    /// `--rustc-arg` so the recorded source converts under
    /// the same configuration its build used, not merely
    /// against the same libraries.
    rustc_args: Vec<String>,
    /// The identity (device and inode) of the placeholder
    /// file the producing invocation left at its output
    /// path, as it stood when the build recorded it: how a
    /// later execution lookup tells this artifact's file
    /// from another's after the recipe renames one over the
    /// other (see `ToolRegistry::get_for_run`).
    file_id: Option<(u64, u64)>,
}

/// The artifact registry of one run (see `execute`): artifact
/// paths in the order the guest produced them (a recipe builds
/// a crate before naming it as an `--extern` input, so build
/// order is dependency order).
#[derive(Default)]
struct ToolRegistry {
    entries: std::sync::Mutex<Vec<(std::path::PathBuf, ToolArtifact)>>,
    /// Source trees materialised to directories, keyed by
    /// the artifact's recorded source path, with the tree's
    /// content hash: a rebuilt artifact at the same path
    /// with a different tree gets a different directory.
    /// The value is the materialised root file's path.
    snapshot_files:
        std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, (u64, std::path::PathBuf)>>,
}

/// A detached copy of one recorded artifact: the registry
/// hands these out, so a caller can hold an artifact while
/// the registry keeps accepting records.
fn cloned_artifact(a: &ToolArtifact) -> ToolArtifact {
    ToolArtifact {
        name: a.name.clone(),
        source: a.source.clone(),
        source_tree: a.source_tree.clone(),
        build_cwd: a.build_cwd.clone(),
        edition: a.edition.clone(),
        crate_type: a.crate_type.clone(),
        compile_env_clear: a.compile_env_clear,
        compile_envs: a.compile_envs.clone(),
        aux_specs: a.aux_specs.clone(),
        aux_pins: a.aux_pins.clone(),
        rustc_args: a.rustc_args.clone(),
        file_id: a.file_id,
    }
}

/// The identity (device and inode numbers) of the file at
/// `path`: what distinguishes one artifact's placeholder
/// file from another's after a recipe moves files around,
/// given every placeholder holds the same (empty) bytes.
/// `None` when the file cannot be stat'ed.
fn file_identity(path: &std::path::Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt as _;
    let md = std::fs::metadata(path).ok()?;
    Some((md.dev(), md.ino()))
}

impl ToolRegistry {
    fn get(&self, path: &std::path::Path) -> Option<ToolArtifact> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, a)| cloned_artifact(a))
    }

    /// The artifact an execution of the file at `path` must
    /// run. The registry's records are keyed by the path a
    /// build produced its output at, but a recipe can move
    /// files after building them -- `std::fs::rename` of one
    /// artifact over another's path is the plain case -- and
    /// the interpreter's filesystem builtins perform such
    /// moves directly on the host, with no word to this
    /// registry. So the lookup follows the file itself: when
    /// the file now at `path` carries an identity a build
    /// recorded, that build's artifact is the program the
    /// path now denotes, wherever it was recorded under.
    /// When the path's recorded artifact is no longer the
    /// file there and no recorded identity matches, what
    /// stands there is not an artifact this run produced,
    /// and the execution is refused rather than answered
    /// with a stale program. When the file cannot be
    /// stat'ed (it is gone), the path record alone decides,
    /// as before.
    fn get_for_run(&self, path: &std::path::Path) -> Result<Option<ToolArtifact>, String> {
        let entries = self.entries.lock().unwrap();
        if let Some(id) = file_identity(path) {
            if let Some((_, a)) = entries.iter().find(|(_, a)| a.file_id == Some(id)) {
                return Ok(Some(cloned_artifact(a)));
            }
            if entries.iter().any(|(p, _)| p == path) {
                return Err(format!(
                    "tool interception: the file at `{}` was replaced after it was built; what stands there is not an artifact this run produced",
                    path.display()
                ));
            }
            return Ok(None);
        }
        Ok(entries
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, a)| cloned_artifact(a)))
    }

    /// The source a conversion of `artifact` must read: the
    /// recorded path itself while every file of the recorded
    /// tree still holds the bytes the build converted (the
    /// common case, and the one whose `file!` spellings stay
    /// the build's), and a materialised snapshot of the
    /// recorded tree -- relative layout preserved, so the
    /// root's `mod` and `include!` inputs stand beside it --
    /// once the recipe has rewritten or deleted any of it.
    fn effective_source(&self, artifact: &ToolArtifact) -> std::path::PathBuf {
        self.effective_source_of(&artifact.source, &artifact.source_tree)
    }

    /// `effective_source` for a bare (source path, tree)
    /// pair: an artifact's own source, or one recorded
    /// dependency pin (see `ToolArtifact::aux_pins`).
    fn effective_source_of(
        &self,
        source: &std::path::Path,
        tree: &SourceTree,
    ) -> std::path::PathBuf {
        if source_tree_intact(source, tree) {
            return source.to_path_buf();
        }
        use std::hash::Hasher as _;
        use std::os::unix::ffi::OsStrExt as _;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for (rel, bytes) in tree {
            hasher.write(rel.as_os_str().as_bytes());
            hasher.write(bytes);
        }
        let hash = hasher.finish();
        let mut files = self.snapshot_files.lock().unwrap();
        if let Some((h, path)) = files.get(source) {
            if *h == hash {
                return path.clone();
            }
        }
        let root_name = source
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "source.rs".to_string());
        let base =
            std::env::temp_dir().join(format!("sparagmos-artifact-sources-{}", std::process::id()));
        let dir = base.join(format!("{hash:016x}"));
        for (rel, bytes) in tree {
            let target = dir.join(rel);
            if let Some(parent) = target.parent() {
                let _ = std::fs::create_dir_all(parent);
                // The materialised tree copies every recorded
                // neighbouring file verbatim -- a private file
                // beside the crate root among them -- so it
                // must not stand in the shared temporary
                // directory under the default permissions:
                // keep every directory of it owner-only (the
                // restriction is re-applied on each visit, so
                // a directory an older run left at this pid's
                // name is tightened too).
                for ancestor in parent.ancestors() {
                    if !ancestor.starts_with(&base) {
                        break;
                    }
                    restrict_to_owner(ancestor, 0o700);
                    if ancestor == base {
                        break;
                    }
                }
            }
            // Owner-only from the moment the file exists,
            // and tightened again after the write in case a
            // file of that name already stood there.
            use std::io::Write as _;
            use std::os::unix::fs::OpenOptionsExt as _;
            let _ = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&target)
                .and_then(|mut f| f.write_all(bytes));
            restrict_to_owner(&target, 0o600);
        }
        let path = dir.join(root_name);
        files.insert(source.to_path_buf(), (hash, path.clone()));
        path
    }

    /// The path-prefix remap a conversion reading the
    /// source at `source` from its snapshot needs, so the
    /// converted program's source-location spellings
    /// (`file!` and its kin) stay the build's: when the
    /// recorded tree no longer stands intact, the
    /// conversion reads the materialised snapshot under
    /// the returned scratch directory, while the build
    /// read the same relative layout under the recorded
    /// source's own directory -- the pair to hand rustc as
    /// `--remap-path-prefix <from>=<to>`. `None` when the
    /// recorded path itself is read (the intact case,
    /// whose spellings already are the build's).
    fn snapshot_remap(
        &self,
        source: &std::path::Path,
        tree: &SourceTree,
    ) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        if source_tree_intact(source, tree) {
            return None;
        }
        let materialised = self.effective_source_of(source, tree);
        let from = materialised.parent()?.to_path_buf();
        let to = source.parent()?.to_path_buf();
        Some((from, to))
    }

    /// One `--aux` spec with its source path routed
    /// through a recorded dependency pin rather than the
    /// registry's current state: the conversion the spec
    /// feeds reads the tree the pinned build converted,
    /// even if the dependency has since been rebuilt (at
    /// the same output path, or as another output from
    /// the same source).
    fn pinned_aux_spec(&self, spec: &str, pin: &(std::path::PathBuf, SourceTree)) -> String {
        let Some((name, rest)) = spec.split_once('=') else {
            return spec.to_string();
        };
        let attrs = match rest.split_once(';') {
            Some((_, a)) => format!(";{a}"),
            None => String::new(),
        };
        format!(
            "{}={}{}",
            name,
            self.effective_source_of(&pin.0, &pin.1).display(),
            attrs
        )
    }

    /// `specs` with each entry routed through its
    /// recorded pin (see `pinned_aux_spec`). Specs and
    /// pins are recorded together by `tool_run`, so a
    /// spec without a pin does not arise for a recorded
    /// artifact; should one appear anyway it stands as
    /// recorded -- no resolution against the registry's
    /// current state happens anywhere on this path.
    fn pinned_aux_specs(
        &self,
        specs: &[String],
        pins: &[(std::path::PathBuf, SourceTree)],
    ) -> Vec<String> {
        specs
            .iter()
            .enumerate()
            .map(|(i, s)| match pins.get(i) {
                Some(pin) => self.pinned_aux_spec(s, pin),
                None => s.clone(),
            })
            .collect()
    }

    /// The `--aux` specs a run of `artifact` hands its
    /// subprocess: the specs its build recorded, each
    /// resolved against the dependency version that
    /// build bound (see `ToolArtifact::aux_pins`).
    fn artifact_aux_specs(&self, artifact: &ToolArtifact) -> Vec<String> {
        self.pinned_aux_specs(&artifact.aux_specs, &artifact.aux_pins)
    }

    fn record(&self, path: std::path::PathBuf, artifact: ToolArtifact) {
        let mut entries = self.entries.lock().unwrap();
        if let Some(slot) = entries.iter_mut().find(|(p, _)| *p == path) {
            slot.1 = artifact;
        } else {
            entries.push((path, artifact));
        }
    }

    fn all(&self) -> Vec<(std::path::PathBuf, ToolArtifact)> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .map(|(p, a)| (p.clone(), cloned_artifact(a)))
            .collect()
    }
}

/// The per-invocation cap on one conversion subprocess: a
/// conversion that has not finished by then is not going to,
/// and the recipe's own harness applies its (larger) budget to
/// the whole run.
const TOOL_RUN_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(180);

/// An absolute form of `path`, resolved against `cwd` when the
/// guest spelled it relative. Lexical only: the registry keys
/// are built the same way, so the spellings a recipe uses for
/// an artifact it produced meet the keys it looks up later.
fn absolutize(cwd: &std::path::Path, path: &str) -> std::path::PathBuf {
    let p = std::path::PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        cwd.join(p)
    }
}

/// The file name a real rustc gives the artifact of one crate
/// type.
fn artifact_filename(crate_name: &str, crate_type: &str) -> String {
    match crate_type {
        "bin" => crate_name.to_string(),
        "dylib" | "cdylib" | "proc-macro" => format!("lib{crate_name}.so"),
        "staticlib" => format!("lib{crate_name}.a"),
        "rmeta" => format!("lib{crate_name}.rmeta"),
        _ => format!("lib{crate_name}.rlib"),
    }
}

/// The `--aux` spec binding one registered artifact.
fn aux_spec_for(name: &str, artifact: &ToolArtifact) -> String {
    let mut spec = format!("{}={}", name, artifact.source.display());
    if let Some(edition) = &artifact.edition {
        spec += &format!(";edition={edition}");
    }
    if let Some(crate_type) = &artifact.crate_type {
        spec += &format!(";crate-type={crate_type}");
    }
    spec
}

/// The subprocess's stderr with the driver's own report lines
/// removed: what remains is the compiler's diagnostics (and,
/// for `--print` invocations, the printed output, which the
/// session discipline routes to stderr), which is what a real
/// compiler child would have produced.
fn child_stream(stderr: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(stderr);
    let mut out = String::new();
    for line in text.lines() {
        if !line.starts_with("sparagmos-driver:") {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.into_bytes()
}

/// Run this driver binary as a subprocess (convert mode for a
/// tool invocation, run mode for an artifact invocation) and
/// collect (exit code, stdout, stderr). The guest command's
/// environment is applied over the inherited one, as a real
/// spawn would apply it.
fn run_convert_subprocess(
    exe: &std::path::Path,
    tail: &[String],
    cwd: &std::path::Path,
    env_clear: bool,
    envs: &[(Vec<u8>, Option<Vec<u8>>)],
) -> Result<(i32, Vec<u8>, Vec<u8>), String> {
    use std::io::Read as _;
    use std::os::unix::ffi::OsStrExt as _;
    let mut cmd = std::process::Command::new(exe);
    cmd.args(tail)
        .current_dir(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if env_clear {
        cmd.env_clear();
    }
    for (key, value) in envs {
        match value {
            Some(v) => {
                cmd.env(
                    std::ffi::OsStr::from_bytes(key),
                    std::ffi::OsStr::from_bytes(v),
                );
            }
            None => {
                cmd.env_remove(std::ffi::OsStr::from_bytes(key));
            }
        }
    }
    let mut child = cmd.spawn().map_err(|e| {
        format!("tool interception: cannot start the tool subprocess: {e}")
    })?;
    let mut out_pipe = child.stdout.take().expect("piped stdout");
    let mut err_pipe = child.stderr.take().expect("piped stderr");
    let out_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = out_pipe.read_to_end(&mut buf);
        buf
    });
    let err_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = err_pipe.read_to_end(&mut buf);
        buf
    });
    let deadline = std::time::Instant::now() + TOOL_RUN_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(
                        "tool interception: the tool subprocess timed out"
                            .to_string(),
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => {
                return Err(format!(
                    "tool interception: waiting on the tool subprocess failed: {e}"
                ));
            }
        }
    };
    let stdout = out_thread.join().unwrap_or_default();
    let stderr = err_thread.join().unwrap_or_default();
    Ok((status.code().unwrap_or(101), stdout, stderr))
}

/// Create a fresh file in the shared temporary directory
/// that no other account can read or predict: the name
/// carries the process id, a per-process sequence, and the
/// wall-clock nanosecond stamp; the open is `create_new`,
/// so an existing namesake is never truncated (the name is
/// retried instead); and the permissions are owner-only
/// from the moment the file exists. For files whose
/// contents are worth keeping from other accounts on a
/// shared host -- the guest environment file is the one
/// here (see `write_guest_env_file`).
/// Set `path`'s permissions to the owner-only `mode`,
/// best-effort. For the directories and files of a
/// materialised source snapshot (see
/// `ToolRegistry::effective_source_of`): the snapshot
/// copies recorded neighbouring files verbatim, so its
/// scratch tree in the shared temporary directory must
/// not be traversable or readable by other accounts.
fn restrict_to_owner(path: &std::path::Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt as _;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode));
}

fn create_private_temp_file(tag: &str) -> std::io::Result<(std::path::PathBuf, std::fs::File)> {
    use std::os::unix::fs::OpenOptionsExt as _;
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..100 {
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "sparagmos-{tag}-{}-{seq}-{nanos:x}",
            std::process::id()
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
        {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "no free private temporary file name",
    ))
}

/// Write one complete environment assignment (the environment
/// an interpreted program is to run under, as `artifact_run`
/// computed it from the execution request) to a fresh private
/// temporary file (see `create_private_temp_file`), in the
/// length-prefixed binary form `read_guest_env_file` reads:
/// a sequence of key/value records, each two u32-LE-length-
/// prefixed byte strings. The assignment is complete, not a
/// delta: the reader replaces its whole environment with it,
/// so no variable from the conversion's environment can leak
/// into the program's. The file is private because the
/// assignment is this process's whole environment --
/// inherited credentials and tokens included -- and it
/// stands in the shared temporary directory for the run's
/// whole duration. Returns the file's path; the caller
/// removes it once the run is over.
fn write_guest_env_file(env: &[(Vec<u8>, Vec<u8>)]) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write as _;
    let mut buf = Vec::new();
    for (key, value) in env {
        buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
        buf.extend_from_slice(value);
    }
    let (path, mut file) = create_private_temp_file("artifact-env")?;
    if let Err(e) = file.write_all(&buf) {
        let _ = std::fs::remove_file(&path);
        return Err(e);
    }
    Ok(path)
}

/// Read back a file `write_guest_env_file` wrote.
fn read_guest_env_file(path: &std::path::Path) -> std::io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let buf = std::fs::read(path)?;
    let mut env = Vec::new();
    let mut at = 0usize;
    let take = |buf: &[u8], at: &mut usize| -> std::io::Result<Vec<u8>> {
        let len = |buf: &[u8], at: &mut usize| -> std::io::Result<usize> {
            let bytes = buf.get(*at..*at + 4).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated env file")
            })?;
            *at += 4;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()) as usize)
        };
        let n = len(buf, at)?;
        let bytes = buf.get(*at..*at + n).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated env file")
        })?;
        *at += n;
        Ok(bytes.to_vec())
    };
    while at < buf.len() {
        let key = take(&buf, &mut at)?;
        let value = take(&buf, &mut at)?;
        env.push((key, value));
    }
    Ok(env)
}

/// Replace this process's environment with the complete
/// assignment `env`: the run entry point applies the guest
/// environment file with this after the conversion is done,
/// so the interpreted program's environment reads (which the
/// interpreter answers from the host environment) see the
/// environment the program was invoked with, not the one its
/// source was converted under.
fn apply_guest_env(env: &[(Vec<u8>, Vec<u8>)]) {
    use std::os::unix::ffi::OsStrExt as _;
    for (key, _) in std::env::vars_os() {
        std::env::remove_var(&key);
    }
    for (key, value) in env {
        std::env::set_var(
            std::ffi::OsStr::from_bytes(key),
            std::ffi::OsStr::from_bytes(value),
        );
    }
}

/// The literal `include!` paths a source text names, in
/// order: `include!` followed by a string literal, the
/// only spelling rustc accepts for the path.
fn literal_include_paths(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut paths = Vec::new();
    let mut i = 0;
    while let Some(at) = text[i..].find("include!") {
        let mut j = i + at + "include!".len();
        i = j;
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'(' {
            continue;
        }
        j += 1;
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'"' {
            continue;
        }
        j += 1;
        let mut path = String::new();
        while j < bytes.len() && bytes[j] != b'"' {
            if bytes[j] == b'\\' && j + 1 < bytes.len() {
                j += 1;
            }
            path.push(bytes[j] as char);
            j += 1;
        }
        if j < bytes.len() {
            paths.push(path);
            i = j;
        }
    }
    paths
}

/// Whether the crate rooted at `root` declares a `main`
/// function: the root's own text, or the text of a file
/// the root splices in through a literal `include!`
/// (transitively, each include resolved against the
/// including file's directory). A `mod` file cannot hold
/// the crate's entry point, but an included file is root
/// text by other means, and a root that only includes
/// its `main` still builds a binary under a real rustc --
/// the root-only reading would record such a build as a
/// library, and the artifact could never be run.
fn crate_root_has_main(root: &std::path::Path, root_text: &str) -> bool {
    if root_text.contains("fn main") {
        return true;
    }
    let base = root.parent().unwrap_or(root);
    let mut frontier: Vec<std::path::PathBuf> = literal_include_paths(root_text)
        .iter()
        .map(|rel| base.join(rel))
        .collect();
    let mut seen: std::collections::HashSet<std::path::PathBuf> =
        frontier.iter().cloned().collect();
    let mut steps = 0;
    while let Some(path) = frontier.pop() {
        steps += 1;
        if steps > 64 {
            break;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if text.contains("fn main") {
            return true;
        }
        let base = path.parent().unwrap_or(&path).to_path_buf();
        for rel in literal_include_paths(&text) {
            let next = base.join(rel);
            if seen.insert(next.clone()) {
                frontier.push(next);
            }
        }
    }
    false
}

/// Answer one intercepted tool invocation (a run-make recipe's
/// compiler call) with the driver's own conversion machinery:
/// the invocation is re-run as this binary in convert mode over
/// the guest's input, and the fabricated process outcome is the
/// conversion's verdict -- exit 0 only when the target crate
/// converted with nothing skipped, with the compiler's
/// diagnostics as the child's stderr. Artifacts a real
/// compiler would have produced are recorded in the registry
/// (and created as placeholders, with real text for dep-info
/// and MIR output) so later invocations can name them through
/// `--extern`, which then binds the original source through
/// `--aux`.
fn tool_run(
    request: &sparagmos_interp::ToolRunRequest,
    registry: &ToolRegistry,
    exe: &std::path::Path,
) -> Result<sparagmos_interp::ToolRunOutcome, String> {
    use std::os::unix::ffi::OsStrExt as _;
    use std::os::unix::ffi::OsStringExt as _;
    let cwd: std::path::PathBuf = match &request.cwd {
        Some(dir) => std::path::PathBuf::from(std::ffi::OsString::from_vec(dir.clone())),
        None => std::env::current_dir()
            .map_err(|e| format!("tool interception: no working directory: {e}"))?,
    };
    // A guest-spelled directory is relative to this
    // process's own working directory (the parent the
    // child would have inherited it from); make it stand
    // on its own, so the paths resolved against it -- the
    // recorded build directory among them -- keep their
    // meaning wherever they are read back.
    let cwd = if cwd.is_absolute() {
        cwd
    } else {
        std::env::current_dir()
            .map_err(|e| format!("tool interception: no working directory: {e}"))?
            .join(cwd)
    };
    let args: Vec<String> = request
        .args
        .iter()
        .map(|a| String::from_utf8_lossy(a).into_owned())
        .collect();

    // Translate the guest's rustc arguments.
    let mut input: Option<String> = None;
    let mut edition: Option<String> = None;
    let mut crate_name: Option<String> = None;
    let mut crate_types: Vec<String> = Vec::new();
    let mut forward: Vec<String> = Vec::new();
    let mut aux: Vec<String> = Vec::new();
    // The dependency version each `aux` entry binds, in
    // the same order (see `ToolArtifact::aux_pins`): the
    // specs are resolved through these pins, so a build
    // -- and the artifact it records -- keeps the
    // dependency builds it actually named.
    let mut aux_pins: Vec<(std::path::PathBuf, SourceTree)> = Vec::new();
    let mut emits: Vec<(String, Option<String>)> = Vec::new();
    let mut out_file: Option<String> = None;
    let mut out_dir: Option<String> = None;
    let mut informational = false;
    let mut positional_only = false;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].clone();
        i += 1;
        if positional_only {
            if input.is_none() {
                input = Some(arg);
            } else {
                forward.push(arg);
            }
            continue;
        }
        if arg == "--" {
            positional_only = true;
            continue;
        }
        // Split a `--flag=value` spelling; short flags keep their
        // value attached (`-Copt-level=2` is one token already).
        let (flag, inline): (&str, Option<String>) = if arg.starts_with("--") {
            match arg.split_once('=') {
                Some((f, v)) => (f, Some(v.to_string())),
                None => (arg.as_str(), None),
            }
        } else {
            (arg.as_str(), None)
        };
        let mut take_value = |inline: Option<String>, i: &mut usize| -> Option<String> {
            match inline {
                Some(v) => Some(v),
                None => {
                    let v = args.get(*i).cloned();
                    if v.is_some() {
                        *i += 1;
                    }
                    v
                }
            }
        };
        match flag {
            "--edition" => edition = take_value(inline, &mut i),
            "--crate-name" => crate_name = take_value(inline, &mut i),
            "--crate-type" => {
                if let Some(v) = take_value(inline, &mut i) {
                    crate_types.push(v.clone());
                    forward.push(format!("--crate-type={v}"));
                }
            }
            "--emit" => {
                if let Some(v) = take_value(inline, &mut i) {
                    for part in v.split(',') {
                        let (kind, path) = match part.split_once('=') {
                            Some((k, p)) => (k.to_string(), Some(p.to_string())),
                            None => (part.to_string(), None),
                        };
                        emits.push((kind, path));
                    }
                }
            }
            "-o" => out_file = take_value(inline, &mut i),
            "--out-dir" => out_dir = take_value(inline, &mut i),
            "--extern" => {
                if let Some(v) = take_value(inline, &mut i) {
                    let (name, path) = match v.split_once('=') {
                        Some((n, p)) => (n.to_string(), Some(p.to_string())),
                        None => (v.clone(), None),
                    };
                    let bound = path.as_ref().and_then(|p| {
                        let abs = absolutize(&cwd, p);
                        registry.get(&abs).map(|art| (name.clone(), art))
                    });
                    match bound {
                        Some((name, art)) => {
                            aux_pins.push((art.source.clone(), art.source_tree.clone()));
                            aux.push(aux_spec_for(&name, &art));
                        }
                        None => {
                            // A real file (a sysroot or prebuilt
                            // rlib) or a name the session must
                            // resolve itself: forward verbatim.
                            forward.push("--extern".to_string());
                            forward.push(v);
                        }
                    }
                }
            }
            "--print" => {
                informational = true;
                forward.push("--print".to_string());
                if let Some(v) = take_value(inline, &mut i) {
                    forward.push(v);
                }
            }
            "--version" | "-V" | "--explain" => {
                informational = true;
                forward.push(arg.clone());
            }
            "--cfg" | "--check-cfg" | "-C" | "-Z" | "-L" | "--target"
            | "--error-format" | "--cap-lints" | "--remap-path-prefix"
            | "--color" | "--diagnostic-width" | "-A" | "-W" | "-D" | "-F" => {
                forward.push(flag.to_string());
                if let Some(v) = take_value(inline, &mut i) {
                    forward.push(v);
                }
            }
            _ if flag.starts_with('-') && flag.len() > 1 => {
                // Any other flag spelling (single-token forms
                // like `-g`, `--pretty=expanded`): forward as-is.
                forward.push(arg.clone());
            }
            _ => {
                if input.is_none() {
                    input = Some(arg);
                } else {
                    forward.push(arg);
                }
            }
        }
    }

    if informational {
        // `--print` / `--version` / `--explain`: the convert
        // session answers these itself (the compiler skips
        // conversion and reports success); the printed text is
        // the child's stdout. An invocation without an input
        // runs over an empty scratch crate.
        let input = match input {
            Some(path) => path,
            None => {
                let scratch = std::env::temp_dir().join(format!(
                    "sparagmos-tool-informational-{}.rs",
                    std::process::id()
                ));
                if !scratch.exists() {
                    let _ = std::fs::write(&scratch, "");
                }
                scratch.display().to_string()
            }
        };
        let mut tail = vec![input];
        for spec in registry.pinned_aux_specs(&aux, &aux_pins) {
            tail.push("--aux".to_string());
            tail.push(spec);
        }
        for tok in &forward {
            tail.push("--rustc-arg".to_string());
            tail.push(tok.clone());
        }
        let (code, _stdout, stderr) =
            run_convert_subprocess(exe, &tail, &cwd, request.env_clear, &request.envs)?;
        let text = child_stream(&stderr);
        return Ok(if code == 0 {
            sparagmos_interp::ToolRunOutcome {
                exit_code: 0,
                stdout: text,
                stderr: Vec::new(),
            }
        } else {
            sparagmos_interp::ToolRunOutcome {
                exit_code: code,
                stdout: Vec::new(),
                stderr: text,
            }
        });
    }

    let Some(input) = input else {
        return Err(
            "tool interception: the invocation names no input file".to_string(),
        );
    };
    let input_abs = absolutize(&cwd, &input);
    let edition = edition.unwrap_or_else(|| "2015".to_string());
    let crate_name = crate_name.unwrap_or_else(|| {
        input_abs
            .file_stem()
            .map(|s| s.to_string_lossy().replace('-', "_"))
            .unwrap_or_else(|| "crate".to_string())
    });
    let source_text = std::fs::read_to_string(&input_abs).unwrap_or_default();
    // The source tree this build converts, snapshotted
    // into every artifact it records: the recipe may
    // rewrite or delete the sources once the build returns,
    // and the artifact must still denote the program this
    // invocation produced.
    let source_tree = snapshot_source_tree(&input_abs);

    // Registered artifacts the input declares through
    // `extern crate` items bind too (the driver's session
    // binding counts them as declarations, as for `--aux`).
    for (_, art) in registry.all() {
        let needle = format!("extern crate {}", art.name);
        if source_text.contains(&needle)
            && !aux.iter().any(|s| s == &aux_spec_for(&art.name, &art))
        {
            aux_pins.push((art.source.clone(), art.source_tree.clone()));
            aux.push(aux_spec_for(&art.name, &art));
        }
    }

    // A MIR emit is served by the same subprocess through the
    // driver's `--dump`: decide its path before running.
    let out_base = match &out_dir {
        Some(dir) => absolutize(&cwd, dir),
        None => cwd.clone(),
    };
    let mir_dump: Option<std::path::PathBuf> = emits
        .iter()
        .find(|(kind, _)| kind == "mir")
        .map(|(_, path)| match path {
            Some(p) => absolutize(&cwd, p),
            None => out_base.join(format!("{crate_name}.mir")),
        });

    let mut tail = vec![
        input_abs.display().to_string(),
        "--edition".to_string(),
        edition.clone(),
    ];
    for spec in registry.pinned_aux_specs(&aux, &aux_pins) {
        tail.push("--aux".to_string());
        tail.push(spec);
    }
    for tok in &forward {
        tail.push("--rustc-arg".to_string());
        tail.push(tok.clone());
    }
    if let Some(dump) = &mir_dump {
        tail.push("--dump".to_string());
        tail.push(dump.display().to_string());
    }
    let (code, stdout, stderr) =
        run_convert_subprocess(exe, &tail, &cwd, request.env_clear, &request.envs)?;

    // The verdict: exit 0 only when the target crate converted
    // with nothing skipped. The report's target-crate line is
    // the authority (a plain exit 0 also covers informational
    // invocations and demand-driven partial conversions).
    let report = String::from_utf8_lossy(&stdout);
    let skipped = report.lines().find_map(|line| {
        let at = line.find("target-crate converted:")?;
        let rest = &line[at..];
        let at = rest.find("skipped:")?;
        rest[at + "skipped:".len()..]
            .trim()
            .split_whitespace()
            .next()?
            .parse::<usize>()
            .ok()
    });
    let child_stderr = child_stream(&stderr);
    if code != 0 || skipped != Some(0) {
        return Ok(sparagmos_interp::ToolRunOutcome {
            exit_code: if code == 101 { 101 } else { 1 },
            stdout: Vec::new(),
            stderr: child_stderr,
        });
    }

    // Success: record and create the artifacts a real compiler
    // would have produced, so later invocations and the
    // recipe's own file checks find them.
    let types: Vec<String> = if crate_types.is_empty() {
        let has_main = crate_root_has_main(&input_abs, &source_text);
        let wants_metadata = emits.iter().any(|(k, _)| k == "metadata");
        if wants_metadata {
            vec!["rmeta".to_string()]
        } else if has_main {
            vec!["bin".to_string()]
        } else {
            vec!["rlib".to_string()]
        }
    } else {
        crate_types.clone()
    };
    let mut primary: Option<std::path::PathBuf> = None;
    let mut record = |path: std::path::PathBuf,
                      crate_type: Option<String>,
                      registry: &ToolRegistry,
                      primary: &mut Option<std::path::PathBuf>| {
        if primary.is_none() {
            *primary = Some(path.clone());
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, b"");
        if crate_type.as_deref() == Some("bin") {
            // A real compiler's binary output is executable:
            // the recipe's spawn of it (and its own executable
            // checks) must find it so.
            use std::os::unix::fs::PermissionsExt as _;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755));
        }
        let file_id = file_identity(&path);
        registry.record(
            path,
            ToolArtifact {
                name: crate_name.clone(),
                source: input_abs.clone(),
                source_tree: source_tree.clone(),
                build_cwd: cwd.clone(),
                edition: Some(edition.clone()),
                crate_type,
                compile_env_clear: request.env_clear,
                compile_envs: request.envs.clone(),
                aux_specs: aux.clone(),
                aux_pins: aux_pins.clone(),
                rustc_args: forward.clone(),
                file_id,
            },
        );
    };
    if let Some(out) = &out_file {
        // `-o` names the invocation's one output (a real rustc
        // refuses `-o` with several crate types), so the type is
        // known exactly when the invocation carried one.
        record(
            absolutize(&cwd, out),
            (types.len() == 1).then(|| types[0].clone()),
            registry,
            &mut primary,
        );
    } else {
        for ty in &types {
            record(
                out_base.join(artifact_filename(&crate_name, ty)),
                Some(ty.clone()),
                registry,
                &mut primary,
            );
        }
    }
    for (kind, path) in &emits {
        match kind.as_str() {
            "metadata" => {
                let p = match path {
                    Some(p) => absolutize(&cwd, p),
                    None => out_base.join(artifact_filename(&crate_name, "rmeta")),
                };
                record(p, Some("rmeta".to_string()), registry, &mut primary);
            }
            "dep-info" => {
                let p = match path {
                    Some(p) => absolutize(&cwd, p),
                    None => out_base.join(format!("{crate_name}.d")),
                };
                if let Some(parent) = p.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let mut deps = vec![input_abs.clone()];
                for spec in &aux {
                    if let Some((_, src)) = spec.split_once('=') {
                        let src = src.split(';').next().unwrap_or(src);
                        deps.push(std::path::PathBuf::from(src));
                    }
                }
                let target = primary
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| crate_name.clone());
                let mut text = format!(
                    "{}: {}",
                    target,
                    deps.iter()
                        .map(|d| d.display().to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                for d in &deps {
                    text += &format!("\n{}:", d.display());
                }
                text.push('\n');
                let _ = std::fs::write(&p, text);
            }
            "llvm-ir" => {
                let p = match path {
                    Some(p) => absolutize(&cwd, p),
                    None => out_base.join(format!("{crate_name}.ll")),
                };
                if let Some(parent) = p.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(&p, b"");
            }
            "asm" => {
                let p = match path {
                    Some(p) => absolutize(&cwd, p),
                    None => out_base.join(format!("{crate_name}.s")),
                };
                if let Some(parent) = p.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(&p, b"");
            }
            _ => {}
        }
    }

    Ok(sparagmos_interp::ToolRunOutcome {
        exit_code: 0,
        stdout: Vec::new(),
        stderr: child_stderr,
    })
}

/// Answer one invocation of a program the guest itself
/// produced earlier in this run (a run-make recipe running a
/// binary it just built): the artifact registry knows the
/// path, so the run is this driver binary in run mode over
/// the recorded source, with the `--aux` bindings and the
/// compilation-shaping options the producing invocation
/// carried and the guest's arguments as the program's own.
/// This is the stage-3 harness's reading of "execute the
/// built binary": the program's input/output and exit code
/// are the interpreter's, not a native binary's (see
/// docs/CI.md). Programs the registry does not know, and
/// artifacts that are not binaries, stay unsupported -- as
/// does a recorded program the driver cannot itself run, so
/// the harness keeps sorting those recipes as unsupported
/// rather than failed.
///
/// The run is reconstructed from what the producing
/// invocation carried, not from the file system as it now
/// stands: the source it converts is the recorded path while
/// every file of its tree still holds the build's bytes, and
/// the recorded snapshot once the recipe has rewritten or
/// deleted any of it (see `ToolRegistry::effective_source`),
/// and its `--aux` bindings
/// likewise resolve against the dependency versions its build
/// bound, not the registry's current entries (see
/// `ToolArtifact::aux_pins`); the program's argv[0]
/// is the name the recipe invoked it under, not the source
/// path; and the two environments are kept apart -- the
/// subprocess is spawned under the producing invocation's
/// compilation environment, so compile-time environment
/// reads see the build's values, and the execution request's
/// environment is handed over as a complete assignment the
/// subprocess applies to itself once the conversion is done
/// (see `--guest-env-file`), so the program's own environment
/// reads see the values it was invoked with and nothing from
/// the build leaks into them.
///
/// The two working directories are kept apart the same
/// way: the subprocess is spawned in the producing
/// invocation's working directory (see
/// `ToolArtifact::build_cwd`), so the compilation-shaping
/// arguments it replays -- a forwarded relative `--extern`
/// path, a `-L` directory -- resolve to the files they
/// denoted at build time, and the requested execution
/// directory is handed over as `--guest-cwd`, which the
/// subprocess enters once its conversion is done, so the
/// program itself runs where the recipe invoked it.
///
/// The lookup path the interpreter's spawn resolution
/// settled on is read against the directory that
/// resolution ran in -- this process's own working
/// directory -- not the execution directory: the
/// resolution already applied the child's directory to a
/// spelled path (and a relative `PATH` entry is relative
/// to the searching process), so standing it on the
/// execution directory would apply the child's directory
/// twice. The guest arguments travel to the subprocess as
/// command-line strings, so an argument that is not valid
/// UTF-8 cannot be carried unchanged and makes the
/// invocation unsupported instead of lossy.
///
/// The subprocess keeps the two stderr channels apart: the
/// recorded program's own stderr travels through the file
/// named by `--guest-stderr-file` and is relayed byte for
/// byte, while the subprocess's stderr carries only the
/// driver's own lines. That separation is also what tells a
/// driver failure from the program's own exit: the run
/// subprocess exits 2 both when it cannot run the recorded
/// program and when the program itself exits 2, but only
/// the driver failure reports a `sparagmos-driver` line on
/// its stderr.
fn artifact_run(
    request: &sparagmos_interp::ToolRunRequest,
    registry: &ToolRegistry,
    exe: &std::path::Path,
) -> Result<sparagmos_interp::ToolRunOutcome, String> {
    use std::os::unix::ffi::OsStrExt as _;
    use std::os::unix::ffi::OsStringExt as _;
    let cwd: std::path::PathBuf = match &request.cwd {
        Some(dir) => std::path::PathBuf::from(std::ffi::OsString::from_vec(dir.clone())),
        None => std::env::current_dir()
            .map_err(|e| format!("tool interception: no working directory: {e}"))?,
    };
    // As in `tool_run`: a guest-spelled directory is
    // relative to this process's own, and the execution
    // directory is handed to a subprocess that starts in
    // the build's directory, so it must stand on its own.
    let cwd = if cwd.is_absolute() {
        cwd
    } else {
        std::env::current_dir()
            .map_err(|e| format!("tool interception: no working directory: {e}"))?
            .join(cwd)
    };
    let program = String::from_utf8_lossy(&request.program).into_owned();
    // The lookup follows the path the interpreter's spawn
    // resolution settled on, not the spelled name: a bare
    // name invoked under a command-specific `PATH` denotes
    // the artifact that search found, not a namesake in
    // the working directory (see
    // `ToolRunRequest::resolved_program`). The spelled
    // name remains what the run presents as argv[0].
    let path = match &request.resolved_program {
        Some(resolved) => {
            let resolved = std::path::PathBuf::from(std::ffi::OsString::from_vec(resolved.clone()));
            if resolved.is_absolute() {
                resolved
            } else {
                // The resolution settled this path against
                // this process's own working directory (the
                // directory the interpreter's spawn search
                // ran in), not the execution directory: a
                // spelling like `sub/./prog` already carries
                // the child's directory, and standing it on
                // the execution directory would apply that
                // directory twice.
                std::env::current_dir()
                    .map_err(|e| format!("tool interception: no working directory: {e}"))?
                    .join(resolved)
            }
        }
        None => absolutize(&cwd, &program),
    };
    let artifact = match registry.get_for_run(&path) {
        Ok(Some(artifact)) => artifact,
        Ok(None) => {
            return Err(format!(
                "tool interception: program `{program}` is not an artifact this run produced"
            ));
        }
        Err(e) => return Err(e),
    };
    if artifact.crate_type.as_deref() != Some("bin") {
        return Err(format!(
            "tool interception: artifact `{program}` is not a binary this run produced"
        ));
    }
    // The recorded program's stderr gets its own channel
    // (a scratch file this run names), so nothing of it is
    // filtered or rewritten on its way to the recipe.
    static ARTIFACT_RUN_SEQ: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);
    let seq = ARTIFACT_RUN_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stderr_path = std::env::temp_dir().join(format!(
        "sparagmos-artifact-stderr-{}-{seq}",
        std::process::id()
    ));
    // The environment the recorded program runs under, as a
    // complete assignment: this process's environment (the
    // guest's own, which the interpreter answers its
    // environment reads from) with the execution request's
    // assignments applied -- or, when the request clears the
    // environment, exactly the request's assignments.
    let mut guest_env: Vec<(Vec<u8>, Vec<u8>)> = if request.env_clear {
        Vec::new()
    } else {
        std::env::vars_os()
            .map(|(k, v)| (k.as_bytes().to_vec(), v.as_bytes().to_vec()))
            .collect()
    };
    for (key, value) in &request.envs {
        match value {
            Some(v) => {
                if let Some(slot) = guest_env.iter_mut().find(|(k, _)| k == key) {
                    slot.1 = v.clone();
                } else {
                    guest_env.push((key.clone(), v.clone()));
                }
            }
            None => guest_env.retain(|(k, _)| k != key),
        }
    }
    let env_path = write_guest_env_file(&guest_env).map_err(|e| {
        format!("tool interception: cannot record the guest environment: {e}")
    })?;
    // The name the program reads as its argv[0]: the explicit
    // argv0 the invocation carried, else the program name as
    // the recipe spelled it -- never the source path the run
    // is reconstructed from.
    let program_name = match &request.argv0 {
        Some(argv0) => String::from_utf8_lossy(argv0).into_owned(),
        None => program.clone(),
    };
    let mut tail = vec![
        "run".to_string(),
        registry.effective_source(&artifact).display().to_string(),
    ];
    if let Some(edition) = &artifact.edition {
        tail.push("--edition".to_string());
        tail.push(edition.clone());
    }
    for spec in registry.artifact_aux_specs(&artifact) {
        tail.push("--aux".to_string());
        tail.push(spec);
    }
    for tok in &artifact.rustc_args {
        tail.push("--rustc-arg".to_string());
        tail.push(tok.clone());
    }
    // A recorded source tree that no longer stands intact
    // is read from its materialised snapshot under a
    // scratch directory; remap that directory back to the
    // recorded source's own directory, so the program's
    // source-location spellings stay the build's (see
    // `ToolRegistry::snapshot_remap`). The artifact's own
    // tree and every pinned dependency's tree take part.
    let mut remap_trees: Vec<(&std::path::Path, &SourceTree)> =
        vec![(&artifact.source, &artifact.source_tree)];
    for pin in &artifact.aux_pins {
        remap_trees.push((&pin.0, &pin.1));
    }
    for (source, tree) in remap_trees {
        if let Some((from, to)) = registry.snapshot_remap(source, tree) {
            tail.push("--rustc-arg".to_string());
            tail.push("--remap-path-prefix".to_string());
            tail.push("--rustc-arg".to_string());
            tail.push(format!("{}={}", from.display(), to.display()));
        }
    }
    tail.push("--guest-cwd".to_string());
    tail.push(cwd.display().to_string());
    tail.push("--guest-stderr-file".to_string());
    tail.push(stderr_path.display().to_string());
    tail.push("--guest-env-file".to_string());
    tail.push(env_path.display().to_string());
    tail.push("--guest-program-name".to_string());
    tail.push(program_name);
    // The arguments travel as command-line strings, and
    // the interpreter's argument model holds text: an
    // argument that is not valid UTF-8 cannot be carried
    // without changing its bytes, so the invocation is
    // unsupported rather than silently lossy.
    for arg in &request.args {
        if std::str::from_utf8(arg).is_err() {
            return Err(format!(
                "tool interception: running artifact `{program}`: a guest argument is not valid UTF-8, which the run cannot carry unchanged"
            ));
        }
        tail.push("--guest-arg".to_string());
        tail.push(String::from_utf8_lossy(arg).into_owned());
    }
    let outcome = run_convert_subprocess(
        exe,
        &tail,
        &artifact.build_cwd,
        artifact.compile_env_clear,
        &artifact.compile_envs,
    );
    let guest_stderr = std::fs::read(&stderr_path).unwrap_or_default();
    let _ = std::fs::remove_file(&stderr_path);
    let _ = std::fs::remove_file(&env_path);
    let (code, stdout, stderr) = outcome?;
    if code == 2 {
        // Exit 2 is both the subprocess's driver-failure code
        // and a code the recorded program itself may exit
        // with; the driver's own stderr channel tells them
        // apart (see the doc comment above). A driver failure
        // makes the recipe's invocation unsupported, with the
        // subprocess's own named reason when it gave one; no
        // driver line means the program ran and chose 2.
        let text = String::from_utf8_lossy(&stderr);
        let reason = text.lines().find_map(|line| {
            line.strip_prefix("sparagmos-driver run: unsupported: ")
                .or_else(|| line.strip_prefix("sparagmos-driver: compilation failed: "))
                .map(|rest| rest.to_string())
        });
        let driver_failed = reason.is_some()
            || text.lines().any(|line| line.starts_with("sparagmos-driver"));
        if driver_failed {
            let reason =
                reason.unwrap_or_else(|| "the recorded program could not be run".to_string());
            return Err(format!(
                "tool interception: running artifact `{program}`: {reason}"
            ));
        }
    }
    Ok(sparagmos_interp::ToolRunOutcome {
        exit_code: code,
        stdout,
        stderr: guest_stderr,
    })
}

/// A converted program ready to run, with its entry point
/// resolved: the shared front half of the run and debug entry
/// points (the two differ only in how the interpreter is driven).
struct BuiltProgram {
    program: sparagmos_interp::MirProgram,
    /// The program key of the entry body (see `execute`).
    entry: String,
    /// The display name of the entry body.
    entry_name: String,
    /// Index of the entry body in `Collected::bodies`.
    entry_index: usize,
    /// The arguments the entry invocation supplies.
    entry_args: Vec<sparagmos_interp::Value>,
    /// The entry body's declared argument count.
    entry_arg_count: usize,
}

/// Build the interpreter program from a collection and resolve
/// its entry point. `program_name` is the name the entry
/// invocation presents as argv[0] (the invocation path, which
/// a caller reconstructing a run may name separately from the
/// source it converted). On failure the run-mode error has
/// already been reported and the `Err` carries the process
/// exit code.
fn build_program(
    c: &Collected,
    program_name: &str,
    guest_args: &[String],
) -> Result<BuiltProgram, i32> {
    let mut program = sparagmos_interp::MirProgram::new();
    for ((name, body), def_id) in c.bodies.iter().zip(c.body_def_ids.iter()) {
        program.functions.push(sparagmos_interp::MirFunction {
            name: name.clone(),
            def_id: Some(*def_id),
            body: body.clone(),
        });
    }
    // The interpreter names ADT locals and resolves runtime-support
    // calls through the full name table (every named item, including
    // body-less std functions and types). `def_names` is indexed by
    // the DefId index within crate 0.
    for (index, name) in c.def_names.iter().enumerate() {
        program.fn_names.insert(
            sparagmos_mir::DefId::new(0, index as u32),
            name.clone(),
        );
    }
    program.fn_abis = c.fn_abis.clone();
    program.adt_discriminants = c.adt_discriminants.clone();
    program.adt_layouts = c.adt_layouts.clone();
    program.adt_field_layouts = c.adt_field_layouts.clone();
    program.tuple_layouts = c.tuple_layouts.clone();
    program.adt_tag_layouts = c.adt_tag_layouts.clone();
    program.coroutine_layouts = c.coroutine_layouts.clone();
    program.adt_shapes = c.adt_shapes.clone();
    program.adt_debug_shapes = c.adt_debug_shapes.clone();
    program.env_debug_shapes = c.env_debug_shapes.clone();
    program.thread_locals = c.thread_locals.clone();
    program.static_tys = c.static_tys.clone();
    program.adt_instance_shapes = c.adt_instance_shapes.clone();
    program.adt_niches = c.adt_niches.clone();
    program.mat_plans = c.mat_plans.clone();
    program.drop_trivial_adts = c.drop_trivial_adts.clone();
    program.drop_no_impl_adts = c.drop_no_impl_adts.clone();
    program.dtor_bodies = c.dtor_bodies.clone();
    program.track_caller_defs = c.track_caller_defs.clone();
    program.fn_shim_locations = c.fn_shim_locations.clone();
    program.link_symbols = c.link_symbols.clone();
    program.source_files = c.source_files.clone();
    // The entry body: the one rustc designated (see
    // `Collected::entry_def_id`) when collection recorded it, and
    // otherwise the first local body named `main`. A `--test`
    // crate has two bodies named `<crate>::main` — the user's and
    // the harness's — and only the designated one is the entry:
    // the user's `main` is an ordinary function there, neither
    // the entry nor run twice.
    let entry_index = c
        .entry_def_id
        .and_then(|def| {
            c.body_def_ids[..c.local_body_count]
                .iter()
                .position(|id| *id == def)
        })
        .or_else(|| {
            c.bodies[..c.local_body_count]
                .iter()
                .position(|(n, _)| is_main(n))
        });
    let Some(entry_index) = entry_index else {
        for f in &c.report.functions {
            if is_main(&f.name) {
                if let FnOutcome::Skipped { reasons } = &f.outcome {
                    let rs: Vec<String> =
                        reasons.iter().map(|(k, v)| format!("{k} x{v}")).collect();
                    eprintln!(
                        "sparagmos-driver run: unsupported: `main` was not converted ({})",
                        rs.join(", ")
                    );
                    return Err(2);
                }
            }
        }
        eprintln!(
            "sparagmos-driver run: unsupported: no `main` function among the converted functions"
        );
        return Err(2);
    };
    // The interpreter keys a body whose name another body shares
    // under a `DefId`-disambiguated key; hand it the key
    // conversion stored this body under, not the bare name (the
    // bare name of a `--test` crate's harness main is also the
    // user's `main`, and names no stored body at all).
    let entry = sparagmos_interp::program_key_of(&program, &program.functions[entry_index]);
    let entry_name = program.functions[entry_index].name.clone();
    // A `#[no_main]` entry in the C convention
    // (`extern "C" fn main(argc, argv)`) declares two parameters;
    // the interpreter's call machinery checks the argument count,
    // so the entry invocation supplies the modelled `argc`/`argv`
    // a native `_start` would pass: the program path followed by
    // the guest arguments, each a NUL-terminated byte string, and
    // a null terminator for the array. A standard entry declares
    // no parameters and receives none, as before.
    let entry_arg_count = c.bodies[entry_index].1.arg_count;
    let entry_args: Vec<sparagmos_interp::Value> = if entry_arg_count == 2 {
        let mut argv: Vec<sparagmos_interp::Value> = Vec::new();
        let mut all_args = vec![program_name.to_string()];
        all_args.extend(guest_args.iter().cloned());
        for arg in &all_args {
            let mut bytes = arg.as_bytes().to_vec();
            bytes.push(0);
            argv.push(sparagmos_interp::Value::Ref(
                sparagmos_interp::RefTarget::value(sparagmos_interp::Value::Bytes(bytes)),
            ));
        }
        argv.push(sparagmos_interp::Value::Ref(
            sparagmos_interp::RefTarget::Null,
        ));
        vec![
            sparagmos_interp::Value::Int(all_args.len() as i128),
            // The array pointer denotes its first element, as a
            // native argv does: element reads and pointer
            // arithmetic over the array go through the index
            // projection.
            sparagmos_interp::Value::Ref(sparagmos_interp::RefTarget::Value(
                Box::new(sparagmos_interp::Value::Tuple(argv)),
                vec![sparagmos_interp::RefProj::Index(0)],
            )),
        ]
    } else {
        Vec::new()
    };
    Ok(BuiltProgram {
        program,
        entry,
        entry_name,
        entry_index,
        entry_args,
        entry_arg_count,
    })
}

fn execute(
    c: &Collected,
    input: &str,
    guest_args: &[String],
    overflow_checks: bool,
    tool_programs: &[String],
    guest_stderr_file: Option<&str>,
    program_name: Option<&str>,
) -> i32 {
    // The name the program reads as its argv[0]: the input
    // path by default (the stand-in for the compiled binary's
    // path), or the name the caller names separately when it
    // reconstructs a run under a different invocation name
    // (an artifact run answering a recipe: see `artifact_run`).
    let program_name = program_name.unwrap_or(input);
    let built = match build_program(c, program_name, guest_args) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let BuiltProgram {
        program,
        entry,
        entry_name,
        entry_index,
        entry_args,
        entry_arg_count,
    } = built;
    // Tool interception (run-make recipes): when the harness
    // named tool programs, the interpreter routes the guest's
    // invocations of them to `tool_run`, which answers with this
    // driver binary's own conversion machinery instead of a
    // process. The artifact registry is shared by every
    // invocation of the run: a recipe builds a crate once and
    // names the artifact as an `--extern` input to later
    // invocations -- and, through the fallback runner, runs the
    // artifact itself, which `artifact_run` answers by
    // interpreting the recorded source.
    let tool_interception = if tool_programs.is_empty() {
        None
    } else {
        let registry = std::sync::Arc::new(ToolRegistry::default());
        let exe = std::env::current_exe().unwrap_or_default();
        let fallback_registry = registry.clone();
        let fallback_exe = exe.clone();
        Some(sparagmos_interp::ToolInterception {
            programs: tool_programs
                .iter()
                .map(|prog| prog.as_bytes().to_vec())
                .collect(),
            runner: sparagmos_interp::ToolRunner(std::sync::Arc::new(
                move |request| tool_run(request, &registry, &exe),
            )),
            // Programs the guest itself produced during the run
            // (a recipe executing a binary it built) are answered
            // from the same registry: running the artifact means
            // interpreting its recorded source (see
            // `artifact_run`).
            fallback: Some(sparagmos_interp::ToolRunner(std::sync::Arc::new(
                move |request| artifact_run(request, &fallback_registry, &fallback_exe),
            ))),
        })
    };
    let config = sparagmos_interp::InterpConfig {
        guest_args: guest_args.to_vec(),
        program_path: program_name.to_string(),
        overflow_checks,
        tool_interception,
        ..sparagmos_interp::InterpConfig::default()
    };
    // A `Result` entry's `Termination` report renders its `Err`
    // payload against the interpreter that executed the entry:
    // the payload may point into that run's allocations and
    // statics, which a detached rendering after the run cannot
    // resolve. The report closure names the payload and its
    // type tag (see `result_report`); the report's exit code is
    // applied below, from the same walk over the returned value.
    let report = |value: &sparagmos_interp::Value| {
        let ty = entry_result_ty(c, entry_index)?;
        match result_report(c, &program.fn_names, ty, value) {
            Some(TerminationReport::Error(payload, tag, err_ty)) => {
                Some((payload.clone(), tag, err_ty.clone()))
            }
            _ => None,
        }
    };
    let run = sparagmos_interp::interpret_mir_captured_with_report(
        &program,
        &entry,
        entry_args,
        config,
        report,
    );
    // The guest's stdout is this process's stdout, whatever the
    // outcome was; the guest's stderr likewise goes to stderr,
    // unless the caller named a separate channel for it (an
    // artifact run answering a recipe: see `artifact_run`), in
    // which case this process's stderr carries only the
    // driver's own lines.
    {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(run.stdout.as_bytes());
        let _ = out.flush();
        let mut relay_to_stderr = true;
        if let Some(path) = guest_stderr_file {
            relay_to_stderr = std::fs::write(path, run.stderr.as_bytes()).is_err();
        }
        if relay_to_stderr {
            let mut err = std::io::stderr().lock();
            let _ = err.write_all(run.stderr.as_bytes());
            let _ = err.flush();
        }
    }
    // Lines this entry point renders onto the guest's stderr
    // itself (the panic and abort reports below): they are
    // part of the guest's stream, so with a separate guest
    // stderr channel they join it there, after the output
    // already relayed, instead of this process's stderr.
    let guest_report_line = |line: &str| {
        use std::io::Write as _;
        if let Some(path) = guest_stderr_file {
            if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path) {
                if writeln!(file, "{line}").is_ok() {
                    return;
                }
            }
        }
        eprintln!("{line}");
    };
    match run.result {
        Ok(sparagmos_interp::Value::Unit) => 0,
        Ok(ref value) if entry_returns_exit_code(c, entry_index) => {
            // The entry reported through `Termination` (a
            // `fn main() -> ExitCode`, which is also the shape of
            // a `--test` harness main): the reported code is the
            // process exit code, exactly as `lang_start` would
            // apply it natively.
            match exit_code_value(value) {
                Some(code) => code,
                None => {
                    eprintln!(
                        "sparagmos-driver run: unsupported: entry `{entry_name}` returned a malformed ExitCode value"
                    );
                    2
                }
            }
        }
        Ok(sparagmos_interp::Value::Int(code)) if entry_arg_count == 2 => {
            // The C-convention entry's return value is the
            // process exit status, as `_start` applies it.
            code as i32
        }
        Ok(sparagmos_interp::Value::Uint(code)) if entry_arg_count == 2 => {
            code as i32
        }
        Ok(ref value) if entry_returns_result(c, entry_index) => {
            // The entry reported through `Termination` (a
            // `fn main() -> Result<_, _>`): `lang_start` applies
            // `Termination::report` to the returned value
            // natively -- `Ok` delegates to the payload's own
            // report, `Err` writes `Error: {err:?}` to stderr
            // and exits with `ExitCode::FAILURE`. The driver
            // applies the same report to the value the entry
            // returned; the `Err` payload's Debug text was
            // rendered by the executing interpreter during the
            // run (see the `interpret_mir_captured_with_report`
            // call above).
            match entry_result_ty(c, entry_index)
                .and_then(|ty| result_report(c, &program.fn_names, ty, value))
            {
                Some(TerminationReport::Code(code)) => code,
                Some(TerminationReport::Error(..)) => {
                    // The report's `Error: ...` line is already
                    // in the captured stderr, written by the
                    // interpreter that executed the entry at the
                    // report's native position -- after the
                    // entry returned, before its thread's TLS
                    // destructors ran (with the value's
                    // structural form standing in when the
                    // payload does not render even against the
                    // live state). Only the exit code remains
                    // to apply.
                    1
                }
                None => {
                    eprintln!(
                        "sparagmos-driver run: unsupported: entry `{entry_name}` returned a malformed Result value"
                    );
                    2
                }
            }
        }
        Ok(other) => {
            eprintln!(
                "sparagmos-driver run: unsupported: entry `{entry_name}` returned {other}, expected ()"
            );
            2
        }
        Err(sparagmos_interp::MirRunError::Interp(
            sparagmos_interp::InterpError::ProcessExit { code },
        )) => {
            // `std::process::exit(code)`: the guest's normal
            // termination; the code is the process exit code.
            code
        }
        Err(sparagmos_interp::MirRunError::Interp(sparagmos_interp::InterpError::Panic {
            message,
            ..
        })) => {
            guest_report_line(&format!("thread 'main' panicked: {message}"));
            101
        }
        Err(sparagmos_interp::MirRunError::Interp(sparagmos_interp::InterpError::Abort {
            message,
        })) => {
            // A panic that escaped a cleanup (a native double
            // panic aborts the process): the guest-panic exit, with
            // the abort reason on stderr.
            guest_report_line(&format!("thread 'main' aborted: {message}"));
            101
        }
        Err(e) => {
            eprintln!("sparagmos-driver run: unsupported: {e}");
            let ext = external_callees(c);
            if !ext.is_empty() {
                eprintln!(
                    "sparagmos-driver run: calls to functions with no converted body: {}",
                    ext.join(", ")
                );
            }
            2
        }
    }
}

/// The `debug` entry point: run a debugger session over the
/// converted program (see [`sparagmos_interp::debug`]), driven by
/// a session script the harness produced from the test file's
/// debugger sections (`--debug-script <path>`). The script is one
/// command per line:
///
/// ```text
/// break <line>     a breakpoint in the input file (repeatable)
/// break-at <file>:<line>
///                  a breakpoint in the named file (repeatable)
/// break-fn <path>  a function breakpoint: stop at the first
///                  position of each frame of the named body
/// pretty-printers  render gdb prints through the sysroot Rust
///                  pretty printers (the harness emits this for
///                  gdb sessions whose test does not opt out with
///                  `disable-gdb-pretty-printers`)
/// type-format <dec|unsigned|hex> <type>
///                  an lldb `type format add` registration for
///                  a char-family type (repeatable)
/// run              start (the session start; at most one, first)
/// continue         resume to the next breakpoint
/// next             step over
/// step             step into
/// finish           step out
/// print <expr>     gdb-form print (also `print/d`, `print/x`,
///                  `print/t`, `print/c` for the value format)
/// v <expr>         lldb-form print
/// whatis <expr>    print the expression's declared type
/// ptype <expr>     print the expression's type with its layout
/// repr <name>      the debugger_tester repr pseudo-command
/// frame <n>        select frame n (0 = innermost)
/// up / down        move the frame selection out / in one frame
/// show-line        emit the current stop line again
/// info-functions [-q] [<pattern>]
///                  list the program's matching functions
/// info-locals      list the selected frame's locals
/// ```
///
/// The session transcript is printed to stdout (the guest's own
/// stdout first, then one line per stop or print). The exit
/// contract is run mode's, adapted to a session: 0 when the
/// session ran (the program's own end, normal or not, is part of
/// the session — the harness judges the transcript), 2 when a
/// conversion-side failure kept the session from starting.
fn execute_debug(
    c: &Collected,
    input: &str,
    guest_args: &[String],
    overflow_checks: bool,
    script_path: Option<&str>,
    dialect_name: &str,
    hide_locals: bool,
) -> i32 {
    let Some(script_path) = script_path else {
        eprintln!("sparagmos-driver debug: --debug-script is required");
        return 2;
    };
    let dialect = match dialect_name {
        "gdb" => sparagmos_interp::DebugDialect::Gdb,
        "lldb" => sparagmos_interp::DebugDialect::Lldb,
        other => {
            eprintln!("sparagmos-driver debug: unknown dialect `{other}` (gdb|lldb)");
            return 2;
        }
    };
    let script_text = match std::fs::read_to_string(script_path) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("sparagmos-driver debug: cannot read script {script_path}: {e}");
            return 2;
        }
    };
    let parsed = match parse_debug_script(&script_text) {
        Ok(parsed) => parsed,
        Err(reason) => {
            eprintln!("sparagmos-driver debug: bad script: {reason}");
            return 2;
        }
    };
    let breakpoints = parsed.break_lines;
    let commands = parsed.commands;
    let pretty_printers = parsed.pretty_printers;
    let built = match build_program(c, input, guest_args) {
        Ok(b) => b,
        Err(code) => return code,
    };
    let config = sparagmos_interp::InterpConfig {
        guest_args: guest_args.to_vec(),
        program_path: input.to_string(),
        overflow_checks,
        ..sparagmos_interp::InterpConfig::default()
    };
    let mut session = match sparagmos_interp::debug_mir(
        &built.program,
        &built.entry,
        built.entry_args,
        config,
    ) {
        Ok(session) => session,
        Err(e) => {
            eprintln!("sparagmos-driver run: unsupported: {e}");
            return 2;
        }
    };
    for line in &breakpoints {
        session.set_breakpoint(input, *line);
    }
    for (file, line) in &parsed.break_files {
        session.set_breakpoint(file, *line);
    }
    for name in &parsed.break_fns {
        session.set_fn_breakpoint(name);
    }
    session.set_hide_locals(hide_locals);
    session.set_pretty_printers(pretty_printers);
    // Register the source text of every file the converted
    // program references, not just the main input: a session
    // can stop in an included file or an auxiliary crate, and
    // a stop line renders only from registered text.
    let files: Vec<String> = session.source_files().to_vec();
    for file in &files {
        if let Ok(text) = std::fs::read_to_string(file) {
            session.set_source_text(file, &text);
        }
    }
    let run = session.run_script(commands, dialect);
    print!("{}", run.stdout);
    for line in &run.transcript {
        println!("{line}");
    }
    if let Err(msg) = &run.outcome {
        eprintln!("sparagmos-driver debug: program ended: {msg}");
    }
    0
}

/// A parsed debugger session script (see [`execute_debug`]).
struct ParsedDebugScript {
    /// Breakpoint lines in the input file.
    break_lines: Vec<u32>,
    /// Breakpoints in a named file: (file, line) pairs.
    break_files: Vec<(String, u32)>,
    /// Function breakpoints: body-name patterns.
    break_fns: Vec<String>,
    /// The session's commands, in script order.
    commands: Vec<sparagmos_interp::DebugCommand>,
    /// Whether gdb prints render through the pretty printers.
    pretty_printers: bool,
}

/// Parse a debugger session script (see [`execute_debug`]) into
/// its breakpoint lines and its command list.
fn parse_debug_script(text: &str) -> Result<ParsedDebugScript, String> {
    use sparagmos_interp::{DebugCommand, DebugPrint, PrintFormat};
    let mut breakpoints = Vec::new();
    let mut break_files = Vec::new();
    let mut break_fns = Vec::new();
    let mut commands = Vec::new();
    let mut pretty_printers = false;
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let bad = |what: &str| format!("line {}: {what}", lineno + 1);
        let (head, rest) = match line.split_once(char::is_whitespace) {
            Some((h, r)) => (h, r.trim()),
            None => (line, ""),
        };
        match head {
            "pretty-printers" => {
                if !rest.is_empty() {
                    return Err(bad("`pretty-printers` takes no argument"));
                }
                pretty_printers = true;
            }
            "type-format" => {
                let (fmt, ty) = rest
                    .split_once(char::is_whitespace)
                    .ok_or_else(|| bad(&format!("bad type-format `{rest}`")))?;
                let format = match fmt {
                    "dec" => sparagmos_interp::LldbNumFormat::Decimal,
                    "unsigned" => sparagmos_interp::LldbNumFormat::Unsigned,
                    "hex" => sparagmos_interp::LldbNumFormat::Hex,
                    other => {
                        return Err(bad(&format!("unknown type format `{other}`")));
                    }
                };
                commands.push(DebugCommand::SetLldbFormat {
                    ty: ty.trim().to_string(),
                    format,
                });
            }
            "break" => {
                let n: u32 = rest
                    .parse()
                    .map_err(|_| bad(&format!("bad breakpoint line `{rest}`")))?;
                breakpoints.push(n);
            }
            "break-at" => {
                let (file, line) = rest
                    .rsplit_once(':')
                    .ok_or_else(|| bad(&format!("bad file breakpoint `{rest}`")))?;
                let n: u32 = line
                    .parse()
                    .map_err(|_| bad(&format!("bad breakpoint line `{rest}`")))?;
                break_files.push((file.to_string(), n));
            }
            "break-fn" => {
                if rest.is_empty() {
                    return Err(bad("`break-fn` takes a function path"));
                }
                break_fns.push(rest.to_string());
            }
            "run" => commands.push(DebugCommand::Run),
            "continue" => commands.push(DebugCommand::Continue),
            "next" => commands.push(DebugCommand::Next),
            "step" => commands.push(DebugCommand::Step),
            "finish" => commands.push(DebugCommand::Finish),
            "show-line" => commands.push(DebugCommand::ShowLine),
            "frame" => {
                let n: usize = rest
                    .parse()
                    .map_err(|_| bad(&format!("bad frame index `{rest}`")))?;
                commands.push(DebugCommand::FrameSelect(n));
            }
            "v" => commands.push(DebugCommand::Print(DebugPrint {
                expr: rest.to_string(),
                format: None,
            })),
            "print" => commands.push(DebugCommand::Print(DebugPrint {
                expr: rest.to_string(),
                format: None,
            })),
            "whatis" => commands.push(DebugCommand::WhatIs(DebugPrint {
                expr: rest.to_string(),
                format: None,
            })),
            "ptype" => commands.push(DebugCommand::Ptype(DebugPrint {
                expr: rest.to_string(),
                format: None,
            })),
            "repr" => commands.push(DebugCommand::Repr(rest.to_string())),
            "up" => commands.push(DebugCommand::FrameUp),
            "down" => commands.push(DebugCommand::FrameDown),
            "info-locals" => commands.push(DebugCommand::InfoLocals),
            "info-functions" | "info-functions-q" => {
                commands.push(DebugCommand::InfoFunctions {
                    pattern: if rest.is_empty() || rest == "-" {
                        None
                    } else {
                        Some(rest.to_string())
                    },
                    quiet: head == "info-functions-q",
                });
            }
            _ if head.starts_with("print/") => {
                let format = match &head["print/".len()..] {
                    "d" => Some(PrintFormat::Decimal),
                    "u" => Some(PrintFormat::Unsigned),
                    "x" => Some(PrintFormat::Hex),
                    "t" => Some(PrintFormat::Binary),
                    "c" => Some(PrintFormat::Char),
                    other => return Err(bad(&format!("unknown print format `{other}`"))),
                };
                commands.push(DebugCommand::Print(DebugPrint {
                    expr: rest.to_string(),
                    format,
                }));
            }
            _ => return Err(bad(&format!("unknown command `{head}`"))),
        }
    }
    Ok(ParsedDebugScript {
        break_lines: breakpoints,
        break_files,
        break_fns,
        commands,
        pretty_printers,
    })
}

/// The crate type an auxiliary source declares for itself
/// (`#![crate_type = "..."]`), if any. When a source declares one,
/// the declaration decides the artifact kind and the driver passes
/// no `--crate-type` of its own.
fn declared_crate_type(src: &str) -> Option<String> {
    declared_inner_attr(src, "crate_type")
}

/// The crate name an auxiliary source declares for itself
/// (`#![crate_name = "..."]`), if any. When a source declares one,
/// the declaration is the crate's name (compiletest links the crate
/// under it, and the test refers to it by it), so the driver passes
/// no `--crate-name` of its own — rustc rejects a `--crate-name`
/// that differs from the declaration.
fn declared_crate_name(src: &str) -> Option<String> {
    declared_inner_attr(src, "crate_name")
}

/// Read an inner-attribute string value (`#![<attr> = "..."]`) from
/// a source file. Only a real inner attribute at the top of the
/// file counts — one written inside a comment does not (the
/// open-ns auxiliary `open-ns-my_api_core.rs` carries
/// `// #![crate_name = "my_api::core"]` as a comment, and treating
/// it as a declaration made the driver look for an artifact under
/// a namespaced name the build never produced).
fn declared_inner_attr(src: &str, attr: &str) -> Option<String> {
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if !line.starts_with("#![") {
            break;
        }
        let Some(rest) = line[3..].trim_start().strip_prefix(attr) else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('"') else {
            continue;
        };
        let end = rest.find('"')?;
        return Some(rest[..end].to_string());
    }
    None
}

/// First `error` line of a compiler's stderr, for failure reasons.
fn first_error_line(stderr: &str) -> String {
    for line in stderr.lines() {
        let line = line.trim();
        if line.starts_with("error") {
            return line.chars().take(200).collect();
        }
    }
    stderr
        .lines()
        .next()
        .unwrap_or("<no compiler output>")
        .chars()
        .take(200)
        .collect()
}

/// Index just past the balanced `open`..`close` group at the
/// start of `s` (`s` must start with `open`), or `None` when the
/// group never closes. Nesting of the same bracket pair counts;
/// string literals inside are not lexed (attribute arguments in
/// test sources do not hide brackets inside strings).
fn skip_balanced(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in s.char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(i + c.len_utf8());
            }
        }
    }
    None
}

/// The crate name an `extern crate <name>` item on this line
/// declares, if the line is one. Attributes (`#[no_link]`,
/// `#[macro_use]`, possibly several in a row, possibly carrying
/// arguments) and a visibility (`pub`, `pub(crate)`) may precede
/// the item — auxiliary sources write e.g.
/// `#[macro_use] #[no_link] extern crate empty_crate_1 as t1;`,
/// and missing those declarations left the dependency unbound
/// (E0463 in the dependent's build). An `as` rename does not
/// change the declared crate name, and `extern crate self as
/// ...` declares no external crate.
fn extern_crate_item_name(line: &str) -> Option<String> {
    let mut rest = line.trim_start();
    loop {
        if rest.starts_with("#[") {
            let brackets = &rest[1..];
            rest = brackets[skip_balanced(brackets, '[', ']')?..].trim_start();
        } else if let Some(after) = rest.strip_prefix("pub") {
            if after.starts_with('(') {
                rest = after[skip_balanced(after, '(', ')')?..].trim_start();
            } else if after.starts_with(|c: char| c.is_whitespace()) {
                rest = after.trim_start();
            } else {
                break;
            }
        } else {
            break;
        }
    }
    let rest = rest.strip_prefix("extern crate ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() || name == "self" {
        None
    } else {
        Some(name)
    }
}

/// The crate names a flag list declares through `--extern`:
/// both the valued form (`--extern name=path` /
/// `--extern=name=path`) and the locator form (`--extern name` /
/// `--extern=name`), where rustc is told the name and locates
/// the crate itself. A test that locates an auxiliary crate this
/// way (tests/ui/resolve/extern-prelude.rs passes `--extern Vec`
/// for an auxiliary crate whose source declares
/// `#![crate_name = "Vec"]`) declares the name exactly as much
/// as an `extern crate` item does, so binding decisions count
/// it. The name is the part before the first `=`, verbatim
/// (modifiers included, as in `declared_aux_binding_names`).
fn locator_extern_names(flags: &[String]) -> Vec<String> {
    let mut names = Vec::new();
    let mut iter = flags.iter();
    while let Some(flag) = iter.next() {
        let value = if flag == "--extern" {
            iter.next().map(|v| v.as_str())
        } else {
            flag.strip_prefix("--extern=")
        };
        if let Some(value) = value {
            let name = value.split('=').next().unwrap_or("");
            if !name.is_empty() {
                names.push(name.to_string());
            }
        }
    }
    names
}

/// The subset of a source's declarations that are true
/// `--extern` binding declarations: the names of its
/// `//@ aux-crate:` directives, verbatim (modifiers and
/// namespaced paths included). Unlike an `aux-build:` stem or
/// an `extern crate` item, an `aux-crate:` name is the binding
/// itself: when it carries modifiers (`priv,noprelude:somedep`)
/// those modifiers are the semantics (noprelude keeps the crate
/// out of the extern prelude), so a consumer that declared one
/// must not also receive the crate under its unmodified build
/// name — that second binding silently undoes the modifiers
/// (tests/ui/extern-flag/multiple-opts.rs relies on the prelude
/// NOT containing the crate).
fn declared_binding_names(src_text: &str) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for line in src_text.lines() {
        let Some(at) = line.find("//@") else { continue };
        let rest = line[at + 3..].trim_start();
        let rest = if rest.starts_with('[') {
            match rest.find(']') {
                Some(end) => rest[end + 1..].trim_start(),
                None => continue,
            }
        } else {
            rest
        };
        if let Some(value) = rest.strip_prefix("aux-crate:") {
            if let Some((name, _)) = value.trim().split_once('=') {
                names.insert(name.trim().to_string());
            }
        }
    }
    names
}

/// The `--extern` binding names a source file declares as its
/// dependencies: for `//@ aux-crate:` the name before the `=`
/// verbatim (modifiers and namespaced paths included), for
/// `//@ aux-build:` / `//@ proc-macro:` the file stem with dashes
/// as underscores, and for an `extern crate <name>;` item the
/// crate name. Scoped directives (`//@[rev] ...`) count too: the
/// union over revisions is what a caller that does not track
/// revisions can bind, and it is still exactly the set the file
/// declares somewhere — never unrelated crates.
fn declared_aux_binding_names(src_text: &str) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for line in src_text.lines() {
        if let Some(name) = extern_crate_item_name(line) {
            names.insert(name);
        }
        let Some(at) = line.find("//@") else { continue };
        let rest = line[at + 3..].trim_start();
        let rest = if rest.starts_with('[') {
            match rest.find(']') {
                Some(end) => rest[end + 1..].trim_start(),
                None => continue,
            }
        } else {
            rest
        };
        if let Some(value) = rest.strip_prefix("aux-crate:") {
            if let Some((name, _)) = value.trim().split_once('=') {
                names.insert(name.trim().to_string());
            }
        } else {
            for directive in ["aux-build:", "proc-macro:"] {
                if let Some(value) = rest.strip_prefix(directive) {
                    let fname = value.trim().split_whitespace().next().unwrap_or("");
                    if let Some(stem) = std::path::Path::new(fname).file_stem() {
                        names.insert(stem.to_string_lossy().replace('-', "_"));
                    }
                }
            }
        }
    }
    names
}

/// The flags a source file's own `//@ compile-flags:` directives
/// declare (whitespace-split; scoped directives included, as in
/// [`declared_aux_binding_names`]). compiletest compiles each
/// auxiliary crate with its own file's flags; the main
/// compilation's flags come from the caller instead.
fn declared_compile_flags(src_text: &str) -> Vec<String> {
    let mut flags = Vec::new();
    for line in src_text.lines() {
        let Some(at) = line.find("//@") else { continue };
        let rest = line[at + 3..].trim_start();
        let rest = if rest.starts_with('[') {
            match rest.find(']') {
                Some(end) => rest[end + 1..].trim_start(),
                None => continue,
            }
        } else {
            rest
        };
        if let Some(value) = rest.strip_prefix("compile-flags:") {
            flags.extend(value.split_whitespace().map(str::to_string));
        }
    }
    flags
}

/// Expand compiletest's `{{...}}` template variables in an
/// auxiliary crate's declared compile flags. compiletest expands
/// every directive value before compiling an auxiliary crate with
/// the crate's own flags; passing `{{src-base}}` verbatim makes a
/// flag like `--remap-path-prefix={{src-base}}=remapped` a no-op,
/// so paths baked into the auxiliary crate's metadata (a macro's
/// `file!()`) stay unremapped. `{{src-base}}` is the auxiliary
/// file's suite root (the nearest ancestor directory whose parent
/// is `tests`, as in the subset runner) and `{{cwd}}` the driver's
/// working directory. Variables that name directories only
/// compiletest owns (`{{build-base}}` and friends) are left as-is:
/// no auxiliary crate's flags use them.
fn expand_aux_flag_variables(flag: &str, src_path: &str) -> String {
    if !flag.contains("{{") {
        return flag.to_string();
    }
    let src = std::path::Path::new(src_path);
    let mut src_base = src.parent().map(|p| p.to_path_buf());
    for ancestor in src.ancestors() {
        let is_suite_root = ancestor
            .parent()
            .is_some_and(|p| p.file_name().is_some_and(|n| n == "tests"));
        if is_suite_root {
            src_base = Some(ancestor.to_path_buf());
            break;
        }
    }
    let mut out = flag.to_string();
    if let Some(base) = &src_base {
        out = out.replace("{{src-base}}", &base.to_string_lossy());
    }
    if let Ok(cwd) = std::env::current_dir() {
        out = out.replace("{{cwd}}", &cwd.to_string_lossy());
    }
    out
}

/// One `--aux` specification: the crate name and source path from
/// `<name>=<file.rs>`, plus the optional per-crate attributes that
/// followed the path (`;crate-type=<type>`, `;edition=<ed>`).
struct AuxSpec {
    name: String,
    path: String,
    crate_type: Option<String>,
    edition: Option<String>,
}

/// Parse one `--aux` value: `<name>=<file.rs>` optionally followed
/// by `;crate-type=<type>` and/or `;edition=<ed>` attributes.
/// Anything else is a usage error.
fn parse_aux_spec(spec: &str) -> Result<AuxSpec, String> {
    let (name, rest) = spec
        .split_once('=')
        .filter(|(n, r)| !n.is_empty() && !r.is_empty())
        .ok_or_else(|| format!("--aux needs a <name>=<file.rs> value, got: {spec}"))?;
    let mut parts = rest.split(';');
    let path = parts.next().unwrap_or("");
    if path.is_empty() {
        return Err(format!("--aux needs a <name>=<file.rs> value, got: {spec}"));
    }
    let mut parsed = AuxSpec {
        name: name.to_string(),
        path: path.to_string(),
        crate_type: None,
        edition: None,
    };
    for attr in parts {
        let (key, value) = attr
            .split_once('=')
            .filter(|(_, v)| !v.is_empty())
            .ok_or_else(|| format!("--aux attribute needs <key>=<value>, got: {attr}"))?;
        match key {
            "crate-type" => parsed.crate_type = Some(value.to_string()),
            "edition" => parsed.edition = Some(value.to_string()),
            _ => return Err(format!("unknown --aux attribute: {key}")),
        }
    }
    Ok(parsed)
}

/// One compiled auxiliary crate.
struct AuxBuilt {
    /// The name the crate was compiled under: its source's own
    /// `#![crate_name]` when it declares one, else the source file
    /// stem with dashes as underscores — the name compiletest's
    /// auxiliary builds carry, and the name the crate's metadata
    /// (and so the Stable MIR session) knows it by.
    name: String,
    /// The name dependents and the main compilation bind the crate
    /// under with `--extern`: the `--aux` spec's name verbatim. It
    /// may carry `--extern` modifiers (`priv:foo`,
    /// `priv,noprelude:foo`) or a namespaced-crates path
    /// (`my_api::utils`), neither of which is a valid `--crate-name`
    /// for the auxiliary build itself.
    extern_name: String,
    /// Artifact to link with `--extern`; `None` for crate types
    /// that expose no Rust API (cdylib and friends).
    artifact: Option<std::path::PathBuf>,
    /// Whether the crate's functions are guest code to convert
    /// (false for proc-macro and cdylib crates).
    collect: bool,
}

/// The names under which a consumer may declare a built
/// auxiliary crate: the `--extern` binding name (the `--aux`
/// spec's name verbatim) and the build name (the source's
/// declared `#![crate_name]`, else the file stem). Which one a
/// declaration uses depends on the declaring source: an
/// `aux-crate:` directive names the binding, while an
/// `extern crate` item that refers to a crate by the name its
/// own source declares (`#![crate_name = "a"]` in the svh
/// crates, `issue6919_3`, `Vec`) uses the build name. Matching
/// only the binding name left those dependencies unbound
/// (E0463 in the dependent's or the main compilation).
fn binding_names_for(built: &AuxBuilt) -> [&str; 2] {
    [built.extern_name.as_str(), built.name.as_str()]
}

/// The names to bind `built` under for a consumer declaring
/// `declared`: every name of `binding_names_for(built)` the
/// consumer declared (binding name first), or none. A consumer
/// can declare both names of the same crate — a test file
/// carries `//@ aux-build:iss-6919.rs` (the binding name) and
/// `extern crate issue6919_3;` (the build name) — and its
/// `extern crate` item looks the crate up by the name it
/// writes, so binding only one of the two leaves the other
/// reference unresolved (E0463). When two built crates share a
/// declared name (two svh `a` crates, the duplicate-rlib `foo`
/// crates), each is bound in build order and rustc's own
/// `--extern` precedence decides, as under compiletest.
///
/// One exception: when the consumer declared the binding name
/// through a true binding declaration (`binding_declared`: an
/// `aux-crate:` directive or a locator `--extern`), that
/// binding alone is emitted. Its modifiers are load-bearing —
/// with `priv,noprelude:somedep` bound, an additional
/// unmodified `--extern somedep=` would put the crate in the
/// extern prelude the modifier exists to keep it out of, and
/// `extern crate somedep;` items still resolve through the
/// modified binding (rustc matches them by crate name).
fn bind_names_for<'a>(
    built: &'a AuxBuilt,
    declared: &std::collections::HashSet<String>,
    binding_declared: &std::collections::HashSet<String>,
) -> Vec<&'a str> {
    if binding_declared.contains(&built.extern_name) {
        return vec![built.extern_name.as_str()];
    }
    binding_names_for(built)
        .into_iter()
        .filter(|name| declared.contains(*name))
        .collect()
}

/// Compile one auxiliary source file with the pinned rustc into
/// `aux_dir`, under the source's declared `#![crate_name]` or the
/// file-stem name compiletest's auxiliary builds use (never the
/// `--aux` spec's binding name: that string may carry `--extern`
/// modifiers or a namespaced-crates path, which are binding syntax,
/// not crate names). Previously built
/// auxiliary crates are visible through `--extern`, so callers pass
/// the crates in dependency order. The crate is built with
/// `spec.edition` when the caller attached one (compiletest builds
/// each auxiliary crate with its own edition) and with
/// `default_edition` otherwise; its crate type is the source's own
/// `#![crate_type]` declaration when present, else the caller's
/// `spec.crate_type`, else `rlib`. On failure, returns the first
/// compiler error line as the reason.
fn compile_aux(
    sysroot: &str,
    default_edition: &str,
    aux_dir: &std::path::Path,
    spec: &AuxSpec,
    previous: &[AuxBuilt],
) -> Result<AuxBuilt, String> {
    let name = spec.name.as_str();
    let src_path = spec.path.as_str();
    let edition = spec.edition.as_deref().unwrap_or(default_edition);
    let src_text = std::fs::read_to_string(src_path)
        .map_err(|e| format!("cannot read auxiliary source {src_path}: {e}"))?;
    let declared = declared_crate_type(&src_text);
    let effective_type = declared
        .as_deref()
        .or(spec.crate_type.as_deref())
        .unwrap_or("rlib");
    let declared_name = declared_crate_name(&src_text);
    // The build name: the declared `#![crate_name]`, else the file
    // stem with dashes as underscores (compiletest's
    // path_to_crate_name). The spec's binding name is used only for
    // `--extern`, never as `--crate-name`.
    let stem_name = std::path::Path::new(src_path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().replace('-', "_"));
    let effective_name = declared_name.as_deref().or(stem_name.as_deref()).unwrap_or(name);
    let rustc = std::path::Path::new(sysroot).join("bin").join("rustc");
    let mut cmd = std::process::Command::new(&rustc);
    // `-Zalways-encode-mir` makes the compiler encode every
    // function's MIR into the crate metadata; without it, the
    // metadata of a library crate only carries MIR for generic and
    // inline functions, and the main compilation's Stable MIR
    // session would find no bodies for the auxiliary functions.
    // The flag is unstable, so the auxiliary compilation runs with
    // RUSTC_BOOTSTRAP=1 — the same condition this driver itself is
    // built under, and the condition compiletest runs under (a
    // development compiler).
    cmd.env("RUSTC_BOOTSTRAP", "1");
    cmd.arg(src_path)
        .arg("-Zalways-encode-mir")
        .arg(format!("--edition={edition}"))
        .arg(format!("--out-dir={}", aux_dir.display()))
        .arg(format!("--sysroot={sysroot}"))
        .arg("-L")
        .arg(format!("dependency={}", aux_dir.display()));
    if declared.is_none() {
        cmd.arg(format!("--crate-type={effective_type}"));
    }
    if declared_name.is_none() {
        cmd.arg(format!("--crate-name={effective_name}"));
    }
    // Bind only the auxiliary crates this source declares as its
    // own dependencies (compiletest binds an auxiliary crate's
    // declared deps, not every crate built so far): binding an
    // unrelated crate can change the outcome — a `priv:`-modified
    // or namespaced binding is invalid without the flags that
    // accompany its declaration, and a crate visible as a direct
    // dependency is not a private one. Undeclared transitive deps
    // still resolve through the `-L dependency` directory above.
    let mut deps = declared_aux_binding_names(&src_text);
    let own_flags = declared_compile_flags(&src_text);
    deps.extend(locator_extern_names(&own_flags));
    let mut binding_declared = declared_binding_names(&src_text);
    binding_declared.extend(locator_extern_names(&own_flags));
    for prev in previous {
        if let Some(artifact) = &prev.artifact {
            for bind in bind_names_for(prev, &deps, &binding_declared) {
                cmd.arg("--extern")
                    .arg(format!("{}={}", bind, artifact.display()));
            }
        }
    }
    // The auxiliary crate's own `//@ compile-flags:` (compiletest
    // compiles each auxiliary crate with its own file's flags,
    // with the `{{...}}` template variables expanded).
    for flag in declared_compile_flags(&src_text) {
        cmd.arg(expand_aux_flag_variables(&flag, src_path));
    }
    let out = cmd
        .output()
        .map_err(|e| format!("failed to run rustc for auxiliary crate `{name}`: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "auxiliary crate `{name}` failed to compile: {}",
            first_error_line(&stderr)
        ));
    }
    // Locate the produced artifact: lib<name>.rlib / .rmeta /
    // .so. A build carrying `-C extra-filename=<suffix>` (an
    // auxiliary source's own compile-flags can set one — the
    // crateresolve crates use `-C extra-filename=-1`) names its
    // artifact lib<name><suffix>.rlib instead, with the suffix
    // between the crate name and the extension; accept that
    // shape too, ranked below the plain lib<name>. artifact. A
    // `-` after the name can only be that separator: crate names
    // never contain one (declared names forbid it, and file
    // stems had theirs replaced with underscores).
    let mut artifact: Option<std::path::PathBuf> = None;
    let mut best_rank = (u8::MAX, u8::MAX);
    if let Ok(entries) = std::fs::read_dir(aux_dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().into_owned();
            let group = if fname.starts_with(&format!("lib{effective_name}.")) {
                0
            } else if fname.starts_with(&format!("lib{effective_name}-")) {
                1
            } else {
                continue;
            };
            let ext = match fname.rsplit('.').next() {
                Some("rlib") => 0,
                Some("rmeta") => 1,
                Some("so") | Some("dylib") | Some("dll") => 2,
                _ => continue,
            };
            if (group, ext) < best_rank {
                best_rank = (group, ext);
                artifact = Some(entry.path());
            }
        }
    }
    let (link, collect) = match effective_type {
        "rlib" | "lib" | "dylib" => (true, true),
        "proc-macro" => (true, false),
        _ => (false, false),
    };
    if link && artifact.is_none() {
        return Err(format!(
            "auxiliary crate `{name}` produced no linkable artifact in {}",
            aux_dir.display()
        ));
    }
    Ok(AuxBuilt {
        name: effective_name.to_string(),
        extern_name: name.to_string(),
        artifact: if link { artifact } else { None },
        collect,
    })
}

/// The JIT program input for a collected program: the same
/// bodies, definition ids and name table the interpreter path
/// consumes, in the shape `sparagmos-jit` (and the harness's
/// `JitRunner`, through the dump) takes.
fn jit_program_input(c: &Collected) -> sparagmos_jit::MirProgramInput {
    let functions = c
        .bodies
        .iter()
        .zip(c.body_def_ids.iter())
        .map(|((name, body), def_id)| sparagmos_jit::MirFunctionInput {
            name: name.clone(),
            def_id: Some(*def_id),
            body: body.clone(),
        })
        .collect();
    let names = c
        .def_names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            (
                sparagmos_mir::DefId::new(0, index as u32),
                name.clone(),
            )
        })
        .collect();
    sparagmos_jit::MirProgramInput {
        functions,
        names,
        adt_discriminants: c.adt_discriminants.clone(),
    }
}

/// Attempt to JIT-compile (lower) the whole collected program.
/// `Err` carries the JIT's named unsupported reason.
fn jit_compile_program(c: &Collected) -> Result<sparagmos_jit::JitProgram, String> {
    sparagmos_jit::JitProgram::compile(&jit_program_input(c)).map_err(|e| match e {
        sparagmos_jit::JitError::Unsupported(what) => what,
        sparagmos_jit::JitError::Backend(e) => format!("jit backend error: {e}"),
    })
}

/// Execute the collected program's `main` with the JIT and return
/// the run-jit process exit code (see the crate docs). The entry
/// contract mirrors the harness's `JitRunner`: the entry is called
/// with no arguments, a normal return is exit 0, and a guest panic
/// is exit 101 with the panic message on stderr. Shapes the JIT
/// entry cannot model — guest arguments, a C-convention
/// `main(argc, argv)`, an entry returning `ExitCode` — are
/// reported as unsupported (exit 2), never guessed.
fn execute_jit(c: &Collected, guest_args: &[String]) -> i32 {
    if !guest_args.is_empty() {
        eprintln!(
            "sparagmos-driver run-jit: unsupported: guest arguments are not supported by JIT execution"
        );
        return 2;
    }
    let entry_index = c
        .entry_def_id
        .and_then(|def| {
            c.body_def_ids[..c.local_body_count]
                .iter()
                .position(|id| *id == def)
        })
        .or_else(|| {
            c.bodies[..c.local_body_count]
                .iter()
                .position(|(n, _)| is_main(n))
        });
    let Some(entry_index) = entry_index else {
        for f in &c.report.functions {
            if is_main(&f.name) {
                if let FnOutcome::Skipped { reasons } = &f.outcome {
                    let rs: Vec<String> =
                        reasons.iter().map(|(k, v)| format!("{k} x{v}")).collect();
                    eprintln!(
                        "sparagmos-driver run-jit: unsupported: `main` was not converted ({})",
                        rs.join(", ")
                    );
                    return 2;
                }
            }
        }
        eprintln!(
            "sparagmos-driver run-jit: unsupported: no `main` function among the converted functions"
        );
        return 2;
    };
    if c.bodies[entry_index].1.arg_count == 2 {
        eprintln!(
            "sparagmos-driver run-jit: unsupported: C-convention entry `main(argc, argv)` is not supported by JIT execution"
        );
        return 2;
    }
    if entry_returns_exit_code(c, entry_index) {
        eprintln!(
            "sparagmos-driver run-jit: unsupported: entry returning `std::process::ExitCode` is not supported by JIT execution"
        );
        return 2;
    }
    if entry_returns_result(c, entry_index) {
        eprintln!(
            "sparagmos-driver run-jit: unsupported: entry returning `core::result::Result` is not supported by JIT execution"
        );
        return 2;
    }
    let prog = match jit_compile_program(c) {
        Ok(p) => p,
        Err(reason) => {
            eprintln!("sparagmos-driver run-jit: unsupported: {reason}");
            let ext = external_callees(c);
            if !ext.is_empty() {
                eprintln!(
                    "sparagmos-driver run-jit: calls to functions with no converted body: {}",
                    ext.join(", ")
                );
            }
            return 2;
        }
    };
    match prog.call_index(entry_index, &[]) {
        Ok(_) => {
            let (stdout, stderr) = prog.guest_output();
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(stdout.as_bytes());
            let _ = out.flush();
            let mut err = std::io::stderr().lock();
            let _ = err.write_all(stderr.as_bytes());
            let _ = err.flush();
            0
        }
        Err(sparagmos_jit::JitCallError::Panic(rec)) => {
            let (stdout, stderr) = prog.guest_output();
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(stdout.as_bytes());
            let _ = out.flush();
            let mut err = std::io::stderr().lock();
            let _ = err.write_all(stderr.as_bytes());
            let _ = err.flush();
            eprintln!("thread 'main' panicked: {}", rec.message);
            101
        }
        Err(other) => {
            eprintln!("sparagmos-driver run-jit: unsupported: {other}");
            2
        }
    }
}

/// Print the convert-mode conversion report for a collected
/// program (the report `jit-compile` prints as well, before its
/// own `jit-compiled:` line).
fn print_convert_report(c: &Collected, input: &str) {
    println!(
        "sparagmos-driver: {input} (Stable MIR, {})",
        env!("SPARAGMOS_BUILD_RUSTC_VERSION")
    );
    for f in &c.report.functions {
        match &f.outcome {
            FnOutcome::Converted => println!(
                "fn {}: blocks={} locals={} args={} [converted]",
                f.name, f.blocks, f.locals, f.arg_count
            ),
            FnOutcome::Skipped { reasons } => {
                let rs: Vec<String> =
                    reasons.iter().map(|(k, v)| format!("{k} x{v}")).collect();
                println!("fn {}: SKIPPED ({})", f.name, rs.join(", "));
            }
            FnOutcome::NotDemanded { reason } => {
                println!("fn {}: NOT DEMANDED ({})", f.name, reason);
            }
        }
    }
    println!(
        "converted: {}, skipped: {}",
        c.report.converted_count(),
        c.report.skipped_count()
    );
    println!(
        "converted without a body: {}",
        c.report.bodyless_converted_count()
    );
    let not_demanded = c.report.not_demanded_count();
    if not_demanded > 0 {
        println!("not demanded: {not_demanded}");
    }
    {
        let target = &c.report.functions[..c.local_report_count];
        let t_converted = target
            .iter()
            .filter(|f| matches!(f.outcome, FnOutcome::Converted))
            .count();
        let t_skipped = target.iter().filter(|f| f.is_skipped()).count();
        println!("target-crate converted: {t_converted}, skipped: {t_skipped}");
        let t_not_demanded = target
            .iter()
            .filter(|f| matches!(f.outcome, FnOutcome::NotDemanded { .. }))
            .count();
        if t_not_demanded > 0 {
            println!("target-crate not demanded: {t_not_demanded}");
        }
        if t_skipped > 0 {
            let mut totals: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            for f in target {
                if let FnOutcome::Skipped { reasons } = &f.outcome {
                    for (k, v) in reasons {
                        *totals.entry(k.as_str()).or_default() += v;
                    }
                }
            }
            println!("target-crate unsupported:");
            for (k, v) in &totals {
                println!("  {k}: {v}");
            }
        }
    }
    let totals = c.report.unsupported_totals();
    if !totals.is_empty() {
        println!("unsupported:");
        for (k, v) in &totals {
            println!("  {k}: {v}");
        }
    }
}

/// The driver's invocation mode, named by the first argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// No subcommand: convert only, print the conversion report.
    Convert,
    /// `run`: convert and execute in the interpreter.
    Run,
    /// `run-jit`: convert and execute with the JIT.
    RunJit,
    /// `jit-compile`: convert, print the report, and attempt JIT
    /// lowering of the whole converted program.
    JitCompile,
    /// `debug`: convert and execute in the interpreter under a
    /// debugger session.
    Debug,
}

impl Mode {
    /// Run-shaped modes share run mode's front-end discipline
    /// (the `--crate-type=lib` pin and `--cap-lints=allow`) and
    /// its failure contract (a failed compilation is reported as
    /// unsupported, exit 2).
    fn is_run(self) -> bool {
        matches!(self, Mode::Run | Mode::RunJit)
    }

    /// Modes whose verdict is a compilation outcome (the program
    /// converts, or fails to compile) rather than an execution
    /// outcome.
    fn is_compile_verdict(self) -> bool {
        matches!(self, Mode::Convert | Mode::JitCompile)
    }
}

/// Whether this process's mode judges a compilation verdict, set
/// once from the parsed mode before the compiler runs and read by
/// `collect` (which the compiler callback reaches without the
/// mode at hand). The required-constant check in `mono_check` is
/// gated on it: in the executing modes the interpreter runs
/// programs through its own models — including bodies rustc's
/// collector would reject, and bodies it cannot fetch at all —
/// and a compilation verdict must not pre-empt execution there.
static COMPILE_VERDICT_MODE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The stack size of the thread the driver runs on: the
/// interpreter's evaluation, projection and value walks are
/// host-recursive, so a guest program with thousands of live
/// frames (each frame costing a chain of host calls) overflows
/// the default main-thread stack long before it reaches the
/// interpreter's own frame limit. The reservation is virtual;
/// pages are committed only as the stack actually grows.
const DRIVER_STACK_SIZE: usize = 8 * 1024 * 1024 * 1024;

fn main() {
    let worker = std::thread::Builder::new()
        .name("sparagmos-driver".to_string())
        .stack_size(DRIVER_STACK_SIZE)
        .spawn(driver_main)
        .expect("sparagmos-driver: failed to spawn the driver thread");
    if worker.join().is_err() {
        // The worker panicked: a driver crash, which the exit
        // contract reports as 101.
        eprintln!("sparagmos-driver: internal error: driver thread panicked");
        std::process::exit(101);
    }
}

fn driver_main() {
    let argv: Vec<String> = std::env::args().collect();
    let mode = match argv.get(1).map(|a| a.as_str()) {
        Some("run") => Mode::Run,
        Some("run-jit") => Mode::RunJit,
        Some("jit-compile") => Mode::JitCompile,
        Some("debug") => Mode::Debug,
        _ => Mode::Convert,
    };
    let run_mode = mode.is_run();
    COMPILE_VERDICT_MODE.store(
        mode.is_compile_verdict(),
        std::sync::atomic::Ordering::Relaxed,
    );
    let debug_mode = mode == Mode::Debug;
    // The debug entry point compiles and collects exactly like
    // run mode (it executes the same program, under a debugger
    // session), so every compilation-shaping and error-contract
    // decision below keys on the executing modes together.
    let exec_mode = run_mode || debug_mode;
    let mut debug_script: Option<String> = None;
    let mut debug_dialect = "gdb".to_string();
    let mut input: Option<String> = None;
    let mut aux_specs: Vec<AuxSpec> = Vec::new();
    let mut aux_lib_specs: Vec<(String, std::path::PathBuf)> = Vec::new();
    let mut tool_programs: Vec<String> = Vec::new();
    let mut dump_path: Option<String> = None;
    let mut guest_stderr_file: Option<String> = None;
    let mut guest_env_file: Option<String> = None;
    let mut guest_cwd: Option<String> = None;
    let mut guest_program_name: Option<String> = None;
    let mut edition = "2021".to_string();
    let mut sysroot: Option<String> = None;
    let mut rustc_args: Vec<String> = Vec::new();
    let mut guest_args: Vec<String> = Vec::new();
    let mut i = if mode == Mode::Convert { 1 } else { 2 };
    while i < argv.len() {
        match argv[i].as_str() {
            "--dump" => {
                i += 1;
                dump_path = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--dump needs a path");
                    std::process::exit(2);
                }));
            }
            "--guest-stderr-file" => {
                i += 1;
                guest_stderr_file = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--guest-stderr-file needs a path");
                    std::process::exit(2);
                }));
            }
            "--guest-env-file" => {
                i += 1;
                guest_env_file = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--guest-env-file needs a path");
                    std::process::exit(2);
                }));
            }
            "--guest-cwd" => {
                i += 1;
                guest_cwd = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--guest-cwd needs a path");
                    std::process::exit(2);
                }));
            }
            "--guest-program-name" => {
                i += 1;
                guest_program_name = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--guest-program-name needs a value");
                    std::process::exit(2);
                }));
            }
            "--edition" => {
                i += 1;
                edition = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--edition needs a value");
                    std::process::exit(2);
                });
            }
            "--rustc-arg" => {
                i += 1;
                rustc_args.push(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--rustc-arg needs a value");
                    std::process::exit(2);
                }));
            }
            "--guest-arg" => {
                i += 1;
                guest_args.push(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--guest-arg needs a value");
                    std::process::exit(2);
                }));
            }
            "--debug-script" => {
                i += 1;
                debug_script = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--debug-script needs a path");
                    std::process::exit(2);
                }));
            }
            "--debug-dialect" => {
                i += 1;
                debug_dialect = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--debug-dialect needs a value (gdb|lldb)");
                    std::process::exit(2);
                });
            }
            "--aux" => {
                i += 1;
                let spec = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--aux needs a <name>=<file.rs> value");
                    std::process::exit(2);
                });
                match parse_aux_spec(&spec) {
                    Ok(parsed) => aux_specs.push(parsed),
                    Err(reason) => {
                        eprintln!("{reason}");
                        std::process::exit(2);
                    }
                }
            }
            "--aux-lib" => {
                i += 1;
                let spec = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--aux-lib needs a <name>=<path> value");
                    std::process::exit(2);
                });
                match spec.split_once('=') {
                    Some((name, path)) if !name.is_empty() && !path.is_empty() => {
                        aux_lib_specs.push((name.to_string(), std::path::PathBuf::from(path)));
                    }
                    _ => {
                        eprintln!("--aux-lib needs a <name>=<path> value, got: {spec}");
                        std::process::exit(2);
                    }
                }
            }
            "--tool-program" => {
                i += 1;
                tool_programs.push(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--tool-program needs a path");
                    std::process::exit(2);
                }));
            }
            "--sysroot" => {
                i += 1;
                sysroot = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--sysroot needs a path");
                    std::process::exit(2);
                }));
            }
            "-h" | "--help" => {
    println!("usage: sparagmos-driver [run|run-jit|jit-compile|debug] <file.rs> [--aux <name>=<file.rs>[;crate-type=<type>][;edition=<ed>]] [--aux-lib <name>=<rlib>] [--tool-program <path>] [--rustc-arg <arg>] [--guest-arg <arg>] [--dump <path>] [--guest-stderr-file <path>] [--edition <ed>] [--sysroot <path>] [--debug-script <path>] [--debug-dialect <gdb|lldb>]");
                return;
            }
            other if input.is_none() && !other.starts_with('-') => {
                input = Some(other.to_string());
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let Some(input) = input else {
        eprintln!("usage: sparagmos-driver [run|run-jit|jit-compile|debug] <file.rs> [--dump <path>]");
        std::process::exit(2);
    };
    let sysroot = sysroot
        .or_else(|| std::env::var("SPARAGMOS_SYSROOT").ok())
        .unwrap_or_else(|| env!("SPARAGMOS_BUILD_SYSROOT").to_string());

    // Only the MIR is needed: emit metadata into a scratch directory so
    // running the driver never leaves build artifacts in the caller's
    // working directory.
    let out_dir = std::env::temp_dir().join(format!("sparagmos-driver-out-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&out_dir);

    // Compile the auxiliary crates first, in the order given (each
    // one sees the previously built ones through --extern), so the
    // main compilation below can link them and `collect` can convert
    // their functions from the same Stable MIR session.
    let aux_dir = out_dir.join("aux");
    let _ = std::fs::create_dir_all(&aux_dir);
    let mut aux_built: Vec<AuxBuilt> = Vec::new();
    for spec in &aux_specs {
        match compile_aux(&sysroot, &edition, &aux_dir, spec, &aux_built) {
            Ok(built) => aux_built.push(built),
            Err(reason) => {
                if mode == Mode::RunJit {
                    eprintln!("sparagmos-driver run-jit: unsupported: {reason}");
                    std::process::exit(2);
                }
                if exec_mode {
                    eprintln!("sparagmos-driver run: unsupported: {reason}");
                    std::process::exit(2);
                }
                eprintln!("sparagmos-driver: {reason}");
                std::process::exit(1);
            }
        }
    }
    // Prebuilt library auxiliaries (--aux-lib): the artifact
    // already exists (a support library built once by the
    // harness), so there is nothing to compile; the entry takes
    // part in binding and collection exactly like a compiled
    // auxiliary. The binding is unconditional -- the crate is
    // part of the harness's contract with the guest (a recipe
    // never declares it) -- so `main` adds the name to the
    // session's declared set below.
    for (name, path) in &aux_lib_specs {
        aux_built.push(AuxBuilt {
            name: name.clone(),
            extern_name: name.clone(),
            artifact: Some(path.clone()),
            collect: true,
        });
    }
    let collect_names: Vec<String> = aux_built
        .iter()
        .filter(|b| b.collect)
        .map(|b| b.name.clone())
        .collect();
    let _ = AUX_CRATES.set(collect_names);

    let mut args: Vec<String> = vec![
        "sparagmos-driver".to_string(),
        input.clone(),
        format!("--edition={edition}"),
        "--emit=metadata".to_string(),
        format!("--out-dir={}", out_dir.display()),
        format!("--sysroot={sysroot}"),
    ];
    // Crate type. Run mode pins `lib` unless the caller forwards a
    // `--crate-type` of its own: execution looks `main` up among the
    // converted functions, and the pin keeps the crate shape the
    // interpreter was built against the only one in the invocation.
    //
    // Convert mode instead mirrors compiletest, which passes no
    // `--crate-type` for a test file at all (its one exception,
    // mir-opt's `rlib`, is forwarded by the caller): a forwarded
    // `--crate-type` or a source-level `#![crate_type]` declaration
    // governs when present, and rustc's own default (bin) applies
    // otherwise -- including the entry-point checks (no `main`
    // function, `main`'s signature) and the dead-code reachability
    // roots, under which `main` is a root rather than an ordinary
    // private function. Pinning `lib` in convert mode made the front
    // end disagree with compiletest on exactly those points: a
    // `#![deny(dead_code)]` test whose `main` nothing calls was
    // rejected here ("function `main` is never used") and accepted
    // there, and entry-point errors never fired here.
    let caller_sets_crate_type = rustc_args
        .iter()
        .any(|a| a == "--crate-type" || a.starts_with("--crate-type="));
    if exec_mode && !caller_sets_crate_type {
        args.push("--crate-type=lib".to_string());
    }
    // Lint levels are not part of an execution verdict, and the
    // lib pin above changes them: it removes the entry-point
    // reachability roots a binary crate gets, so a lint a test
    // denies -- `#![deny(dead_code)]` in the source, a forwarded
    // `-D warnings` -- can fire on `main` itself and fail the
    // front end where the native binary compile (the run's
    // reference) succeeded. Run mode therefore caps lints at
    // allow, restoring the native compile's outcome under the
    // pinned crate shape. A lint that fails the native compile
    // keeps a test from ever reaching the driver, so the cap
    // admits nothing the reference rejected. Convert mode caps
    // nothing: its front end mirrors compiletest's, lint errors
    // included.
    if exec_mode {
        args.push("--cap-lints=allow".to_string());
    }
    // Caller-supplied rustc arguments (a test's `//@ compile-flags`,
    // per-revision `--cfg`s) go last, as they do under compiletest.
    // The `--extern` names among them are captured first: the
    // auxiliary binding below counts them as declarations.
    let locator_names = locator_extern_names(&rustc_args);
    args.extend(rustc_args);
    if !aux_built.is_empty() {
        args.push("-L".to_string());
        args.push(format!("dependency={}", aux_dir.display()));
        // A prebuilt auxiliary's own dependencies live next to
        // it (the build's deps directory), not in `aux_dir`.
        for (_, path) in &aux_lib_specs {
            if let Some(parent) = path.parent() {
                args.push("-L".to_string());
                args.push(format!("dependency={}", parent.display()));
            }
        }
        // Bind only the auxiliary crates the test file itself
        // declares (compiletest binds a test's own aux directives;
        // transitive auxiliary crates reach the main compilation
        // through their dependents' re-exports and the `-L`
        // directory, not as direct `--extern`s — a transitive crate
        // bound directly stops being a private dependency, which
        // changes lint outcomes like exported_private_dependencies).
        let main_text = std::fs::read_to_string(&input).unwrap_or_default();
        let mut top_names = declared_aux_binding_names(&main_text);
        top_names.extend(locator_names.iter().cloned());
        // Prebuilt auxiliaries bind unconditionally (see the
        // --aux-lib note at their construction).
        top_names.extend(aux_lib_specs.iter().map(|(name, _)| name.clone()));
        let mut binding_declared = declared_binding_names(&main_text);
        binding_declared.extend(locator_names.iter().cloned());
        for built in &aux_built {
            if let Some(artifact) = &built.artifact {
                for bind in bind_names_for(built, &top_names, &binding_declared) {
                    args.push("--extern".to_string());
                    args.push(format!("{}={}", bind, artifact.display()));
                }
            }
        }
        // An auxiliary crate may carry the same name as the main
        // crate (compiletest has tests whose aux file is the test
        // file's own name, e.g. `llvm-pr32379.rs` with
        // `aux-build:llvm_pr32379.rs`). Compiled with the same
        // default `-C metadata`, rustc then rejects the main
        // compilation with E0519 (the crate is indistinguishable
        // from its dependency). compiletest distinguishes its test
        // compilations with a per-test `-C metadata`; do the same
        // for the main compilation when a name collision is present.
        let main_crate_name = std::path::Path::new(&input)
            .file_stem()
            .map(|stem| stem.to_string_lossy().replace('-', "_"));
        if let Some(main_name) = main_crate_name {
            if aux_built.iter().any(|b| b.name == main_name) {
                args.push("-C".to_string());
                args.push("metadata=sparagmos-main".to_string());
            }
        }
    }
    // The compiler runs in-process, under two disciplines:
    // - A panic raised inside Stable MIR conversion (the
    //   `unimplemented!()`s hit by explicit tail calls, async
    //   closures and generators) must not kill this process: it is
    //   caught here and reported as a conversion failure, with the
    //   panic message as the concrete reason (exit 2 in run mode).
    // - Anything the compiler prints to its stdout at compile time
    //   (e.g. `log_syntax!` output) is redirected to stderr: this
    //   process's stdout is reserved for the driver's own report —
    //   in run mode, for the guest program's output alone.
    let overflow_checks = session_overflow_checks(&args);
    let previous_hook = begin_panic_capture();
    let mut stdout_guard = StdoutToStderr::new();
    let run_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rustc_public::run_with_tcx!(args.as_slice(), collect_with_tcx)
    }));
    stdout_guard.restore();
    end_panic_capture(previous_hook);
    let result = match run_outcome {
        Ok(result) => result,
        Err(payload) => {
            let reason = take_captured_panic()
                .or_else(|| payload_reason(&*payload))
                .unwrap_or_else(|| "panic with a non-string payload".to_string());
            if is_delayed_bug_flush(&reason) {
                // The compiler died on an internal error of its own
                // (see `is_delayed_bug_flush`): a compiler crash
                // (Issue #66). Convert mode reports it with the
                // crash exit code, the code the native compiler
                // dies with on the same flush; run mode keeps its
                // single unsupported code, and the
                // compilation-failure report, for every
                // conversion-side failure.
                if exec_mode {
                    eprintln!("sparagmos-driver: compilation failed: {DELAYED_BUG_ABORT}");
                    std::process::exit(2);
                }
                eprintln!("sparagmos-driver: compiler crashed: {DELAYED_BUG_ABORT}");
                std::process::exit(101);
            }
            if let Some(abort) = treat_err_abort_reason(&reason) {
                // The compiler itself aborted the compilation on a
                // diagnostic it had already reported, under a flag
                // ordering exactly that: a compilation failure, not
                // a conversion gap, so it takes the
                // compilation-failure report (and its exit codes),
                // exactly like a front-end rejection recorded by
                // the callbacks.
                eprintln!("sparagmos-driver: compilation failed: {abort}");
                std::process::exit(if exec_mode { 2 } else { 1 });
            }
            if mode == Mode::RunJit {
                eprintln!(
                    "sparagmos-driver run-jit: unsupported: Stable MIR conversion panicked: {reason}"
                );
                std::process::exit(2);
            }
            if exec_mode {
                eprintln!(
                    "sparagmos-driver run: unsupported: Stable MIR conversion panicked: {reason}"
                );
                std::process::exit(2);
            }
            // Any other panic that escapes the compiler run is an
            // internal failure of the compiler itself (Issue #66):
            // convert mode reports it as a crash, not as a clean
            // compilation failure. Panics the conversion contains
            // per body or per query never reach this point; they
            // keep their named reasons in the report.
            eprintln!("sparagmos-driver: Stable MIR conversion panicked: {reason}");
            std::process::exit(101);
        }
    };
    let collected = take();
    match (result, collected) {
        (Ok(()), Some(c)) => {
            if let Some(e) = &c.compile_error {
                eprintln!("sparagmos-driver: compilation failed: {e}");
                // Same contract as a front-end failure: exit 1 in
                // convert mode, exit 2 (unsupported) in run mode.
                std::process::exit(if exec_mode { 2 } else { 1 });
            }
            if debug_mode {
                // Limited debug info (`-C debuginfo=0` or `=1`,
                // in joined or split form) carries no variable
                // information; the session models that for
                // `info locals`.
                let hide_locals = args.iter().any(|a| {
                    matches!(
                        a.as_str(),
                        "debuginfo=0" | "debuginfo=1" | "-Cdebuginfo=0"
                            | "-Cdebuginfo=1"
                    )
                });
                std::process::exit(execute_debug(
                    &c,
                    &input,
                    &guest_args,
                    overflow_checks,
                    debug_script.as_deref(),
                    &debug_dialect,
                    hide_locals,
                ));
            }
            if mode == Mode::Run || mode == Mode::RunJit {
                // A caller reconstructing a run (an artifact run
                // answering a recipe) hands the environment the
                // program was invoked with as a complete
                // assignment in a file: apply it now that the
                // conversion is done, so the conversion above
                // ran under the environment it was spawned with
                // (the producing invocation's compilation
                // environment) while the program's own
                // environment reads see its invocation
                // environment (see `artifact_run`).
                if let Some(path) = &guest_env_file {
                    match read_guest_env_file(std::path::Path::new(path)) {
                        Ok(env) => apply_guest_env(&env),
                        Err(e) => {
                            eprintln!(
                                "sparagmos-driver run: unsupported: cannot read the guest environment file {path}: {e}"
                            );
                            std::process::exit(2);
                        }
                    }
                }
                // A caller reconstructing a run converts in
                // one directory and executes in another (an
                // artifact run converts in the build's
                // directory): it hands the execution
                // directory over the same way, and the
                // program enters it now that the conversion
                // is done (see `artifact_run`).
                if let Some(dir) = &guest_cwd {
                    if let Err(e) = std::env::set_current_dir(dir) {
                        eprintln!(
                            "sparagmos-driver run: unsupported: cannot enter the guest working directory {dir}: {e}"
                        );
                        std::process::exit(2);
                    }
                }
                if let Some(path) = &dump_path {
                    let text =
                        sparagmos_driver::dump::render(&input, &c.report, &c.bodies, &c.body_def_ids, &c.def_names);
                    if let Err(e) = std::fs::write(path, text) {
                        eprintln!("failed to write dump {path}: {e}");
                        std::process::exit(1);
                    }
                }
                if mode == Mode::RunJit {
                    std::process::exit(execute_jit(&c, &guest_args));
                }
                std::process::exit(execute(
                    &c,
                    &input,
                    &guest_args,
                    overflow_checks,
                    &tool_programs,
                    guest_stderr_file.as_deref(),
                    guest_program_name.as_deref(),
                ));
            }
            print_convert_report(&c, &input);
            if mode == Mode::JitCompile {
                match jit_compile_program(&c) {
                    Ok(_) => println!("jit-compiled: yes"),
                    Err(reason) => {
                        println!("jit-compiled: no");
                        eprintln!(
                            "sparagmos-driver jit-compile: unsupported: {reason}"
                        );
                        std::process::exit(2);
                    }
                }
            }
            if let Some(path) = &dump_path {
                let text =
                    sparagmos_driver::dump::render(&input, &c.report, &c.bodies, &c.body_def_ids, &c.def_names);
                if let Err(e) = std::fs::write(path, text) {
                    eprintln!("failed to write dump {path}: {e}");
                    std::process::exit(1);
                }
                println!("dump: {path}");
            }
        }
        (Ok(()), None) => {
            eprintln!("sparagmos-driver: internal error: no results collected");
            std::process::exit(1);
        }
        (Err(rustc_public::CompilerError::Skipped), _) if !exec_mode => {
            // The compiler finished without ever handing the
            // callbacks a crate: it was asked for information
            // (--version, --print, --explain) or told to stop after
            // parsing the crate root (-Z parse-crate-root-only).
            // The compilation it was asked to perform succeeded --
            // these paths report no error -- so convert mode
            // reports an empty, successful conversion instead of a
            // compilation failure. Run mode has no program to
            // execute on this path and keeps the failure arm below.
            println!(
                "sparagmos-driver: {input} (Stable MIR, {})",
                env!("SPARAGMOS_BUILD_RUSTC_VERSION")
            );
            println!(
                "compiler skipped conversion: informational or parse-only invocation"
            );
            println!("converted: 0, skipped: 0");
            println!("target-crate converted: 0, skipped: 0");
            if mode == Mode::JitCompile {
                // No crate was handed to the callbacks, so there
                // is no program to lower; the empty conversion the
                // compiler was asked for succeeded, and JIT
                // lowering of an empty program succeeds trivially.
                println!("jit-compiled: yes");
            }
        }
        (Err(e), _) => {
            eprintln!("sparagmos-driver: compilation failed: {e:?}");
            // In run mode a failed compilation is a conversion failure,
            // which the run contract reports as exit 2.
            std::process::exit(if exec_mode { 2 } else { 1 });
        }
    }
}
