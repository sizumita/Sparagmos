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
//! Run-mode exit codes: 0 = the guest program terminated normally,
//! 101 = the guest program panicked (message on stderr), 2 = the
//! program could not be converted or executed (concrete reason on
//! stderr). Run mode writes the guest program's captured stdout to
//! its own stdout (in every outcome — a native program's output is
//! likewise observable when it panics).
//!
//! Conversion can also fail *inside* Stable MIR itself: some MIR
//! constructs make `rustc_public` panic (`unimplemented!()`). Those
//! panics are caught at the compiler-run boundary and reported like
//! any other conversion failure (run mode: exit 2, with the panic
//! message as the reason). While the compiler runs, its stdout is
//! redirected to stderr: compile-time output (e.g. `log_syntax!`)
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
    args_mention_params, closure_kind, is_coroutine_closure_panic, Converter,
    COROUTINE_CLOSURE_TYPE_REASON,
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
    def_names: Vec<String>,
    /// Discriminant values of the converted enum ADTs, for the
    /// interpreter (see `Converter::adt_discriminants`).
    adt_discriminants: sparagmos_mir::AdtDiscriminants,
    /// Layouts of the converted monomorphic ADT instances, for the
    /// interpreter (see `Converter::adt_layouts`).
    adt_layouts: sparagmos_mir::AdtLayouts,
    /// Declared shapes of the converted ADTs, for the interpreter
    /// (see `Converter::adt_shapes`).
    adt_shapes: sparagmos_mir::AdtShapes,
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

/// First panic message observed while the panic-capture hook is
/// installed (see `begin_panic_capture`). The hook also fires for
/// panics on the compiler's own threads and for panics the compiler
/// handles internally; the recorded message is only consumed when a
/// panic actually escapes the compiler run (the `catch_unwind` in
/// `main`), so a handled panic never changes the outcome.
static PANIC_CAPTURE: Mutex<Option<String>> = Mutex::new(None);

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

/// Run a Stable MIR query that may hit rustc_public 1.100.0's
/// unimplemented coroutine-closure type conversion (inspecting
/// the type kind of an async closure item materializes it,
/// unlike a body fetch, whose types stay opaque until
/// conversion). A panic from that specific unimplemented
/// conversion is contained and reported as
/// `COROUTINE_CLOSURE_TYPE_REASON`; any other panic is resumed —
/// this guard must never paper over an unrelated compiler
/// failure.
fn guard_coroutine_closure_query<T>(query: impl FnOnce() -> T) -> Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(query)) {
        Ok(value) => Ok(value),
        Err(payload) => {
            if is_coroutine_closure_panic(&*payload) {
                // Contained panic: clear the capture slot, as the
                // instance collection does — a later, genuinely
                // fatal panic must report its own message.
                let _ = take_captured_panic();
                Err(COROUTINE_CLOSURE_TYPE_REASON.to_string())
            } else {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

/// Convert one body through the coroutine-closure containment
/// (see `Converter::convert_body_contained`), clearing the panic
/// capture slot when the containment fired: the hook recorded
/// the contained panic's message, and a later, genuinely fatal
/// panic must report its own.
fn convert_body_isolated(
    conv: &mut Converter,
    body: &rustc_public::mir::Body,
) -> Result<Body, Vec<String>> {
    let result = conv.convert_body_contained(body);
    if matches!(&result, Err(errs) if errs.iter().any(|e| e == COROUTINE_CLOSURE_TYPE_REASON)) {
        let _ = take_captured_panic();
    }
    result
}

/// Install a panic hook that records the first panic's message into
/// `PANIC_CAPTURE` and prints nothing. While the compiler runs, the
/// default panic dump must not leak into this process's stderr: run
/// mode reserves stderr for the contract messages, and the captured
/// message is reported as the conversion-failure reason instead.
/// Returns the previous hook, to be restored by `end_panic_capture`
/// as soon as the compiler run is over — panics outside the run
/// (e.g. in the interpreter) keep the normal reporting.
fn begin_panic_capture() -> PanicHook {
    if let Ok(mut slot) = PANIC_CAPTURE.lock() {
        *slot = None;
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        if let Ok(mut slot) = PANIC_CAPTURE.lock() {
            if slot.is_none() {
                *slot = Some(panic_reason(info));
            }
        }
    }));
    previous
}

fn end_panic_capture(previous: PanicHook) {
    std::panic::set_hook(previous);
}

fn take_captured_panic() -> Option<String> {
    PANIC_CAPTURE.lock().ok().and_then(|mut slot| slot.take())
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
/// coroutine closures; their bodies are coroutines, which are
/// not converted — and merely asking for an async closure
/// item's type hits the unimplemented coroutine-closure type
/// conversion, so the guard accounts the item unsupported
/// before its kind is ever seen (the `CoroutineClosure` arm
/// below only fires if a future Stable MIR implements the
/// conversion).
fn closure_instance(item: &rustc_public::CrateItem) -> ClosureAttempt {
    match guard_coroutine_closure_query(|| closure_instance_kind(item)) {
        Ok(attempt) => attempt,
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
            for (def, args) in sparagmos_driver::reach::fndef_constants(&body) {
                if deferred_defs.contains(&def.def_id()) {
                    targets.push((def, args));
                }
            }
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
            let Some(ibody) = inst.body() else {
                fail_all(&mut *demands, &owners, "no MIR body available".to_string());
                continue;
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
fn collect_closure_instances(
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
        recursion_limit: _,
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
                let Some(&base_name) = deferred_names.get(&referenced) else {
                    continue;
                };
                if args_mention_params(&args) {
                    continue;
                }
                let owners = [referenced];
                let Some(kind) = closure_kind(&args) else {
                    if unresolved_seen.insert((referenced, format!("{args:?}"))) {
                        *demands
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
                            *demands
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
                if *taken >= MAX_LOCAL_INSTANCES {
                    fail_all(&mut *demands, &owners, "collection limit reached".to_string());
                    continue;
                }
                if !inst.has_body() {
                    fail_all(&mut *demands, &owners, "no MIR body available".to_string());
                    continue;
                }
                let Some(ibody) = inst.body() else {
                    fail_all(&mut *demands, &owners, "no MIR body available".to_string());
                    continue;
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
                        body_def_ids.push(conv.instance_def_id(referenced, &inst));
                        body_def_names.push(base_name.to_string());
                        let demand = demands.entry(referenced).or_default();
                        demand.converted += 1;
                        if demand.first_counts.is_none() {
                            demand.first_counts = Some(counts);
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
            // function-instance collection does.
            let _ = take_captured_panic();
        }
    }

    extra_seeds
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
        recursion_limit: _,
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
                if args_mention_params(&args) {
                    continue;
                }
                let owners = [referenced];
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
                        inst.body()
                    }
                    Err(_) => {
                        if !unresolved_seen.insert((referenced, format!("{args:?}"))) {
                            continue;
                        }
                        def.body()
                    }
                };
                let Some(ibody) = ibody else {
                    fail_all(
                        &mut *demands,
                        &owners,
                        "no MIR body available".to_string(),
                    );
                    continue;
                };
                if *taken >= MAX_LOCAL_INSTANCES {
                    fail_all(
                        &mut *demands,
                        &owners,
                        "collection limit reached".to_string(),
                    );
                    continue;
                }
                match conv.convert_body(&ibody) {
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
/// skipped with that instance's reasons, and a definition nothing
/// demands keeps the item loop's "no monomorphic instance
/// available" record. One exception: a *constructor* nothing
/// demands is accounted converted without a body — its direct
/// uses are aggregate rvalues in the callers (already converted
/// with them), and the constructor item has no body in any
/// monomorphization that was not demanded, so there is nothing
/// left unconverted (the same accounting as a trivial constant).
fn apply_instance_demands(
    report: &mut ConversionReport,
    deferred: &[DeferredFn],
    demands: &HashMap<rustc_public::DefId, Demand>,
) {
    for dfn in deferred {
        let Some(demand) = demands.get(&dfn.def) else {
            if dfn.ctor {
                report.functions[dfn.report_idx].outcome = FnOutcome::Converted;
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
    let mut out = std::collections::HashMap::new();
    for f in rustc_public::local_crate().fn_defs() {
        let name = f.name();
        if !(name.ends_with(" as std::ops::Drop>::drop")
            || name.ends_with(" as core::ops::Drop>::drop"))
        {
            continue;
        }
        // `<crate::Type<Params> as ...::Drop>::drop`: the key is
        // the self type's path without its parameter list.
        let Some(rest) = name.strip_prefix('<') else {
            continue;
        };
        let Some((self_part, _)) = rest.split_once(" as ") else {
            continue;
        };
        let base = self_part.split('<').next().unwrap_or(self_part);
        out.entry(base.to_string()).or_insert(f);
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
        ty = match elem {
            P::Field(_, t) => *t,
            P::Deref => match ty.kind() {
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
            P::Index(_) | P::ConstantIndex { .. } | P::Subslice { .. } => match ty.kind() {
                TyKind::RigidTy(RigidTy::Array(elem, _)) => elem,
                TyKind::RigidTy(RigidTy::Slice(elem)) => elem,
                _ => return None,
            },
            P::Downcast(_) => ty,
            P::OpaqueCast(t) => *t,
        };
    }
    Some(ty)
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
    seen: &mut std::collections::HashSet<String>,
) {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(def, args)) => {
            let key = format!("{:?} {:?}", def.def_id(), args);
            if !seen.insert(key) {
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
                    drop_candidates(&field.ty_with_args(&args), out, seen);
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

/// Demand a converted body for the `Drop` impl of every local ADT
/// instantiation that the converted bodies drop (see the call site
/// in `collect`). Each demanded body is converted like a local item
/// and added to the seed bodies, so the reachability collection
/// walks its callees too; demanded bodies are scanned for further
/// drops in turn. An ADT whose impl does not resolve, or whose
/// instance has no body, is left as it was: the interpreter keeps
/// refusing its drop with the reason it already reports.
#[allow(clippy::too_many_arguments)]
fn demand_drop_instances(
    conv: &mut Converter,
    report: &mut ConversionReport,
    bodies: &mut Vec<(String, Body)>,
    body_def_ids: &mut Vec<sparagmos_mir::DefId>,
    body_def_names: &mut Vec<String>,
    seed_bodies: &mut Vec<rustc_public::mir::Body>,
) {
    let local_crate = rustc_public::local_crate();
    let local_crate_name = local_crate.name.as_str();
    let impl_methods = local_drop_impl_methods();
    if impl_methods.is_empty() {
        return;
    }
    let mut seen_candidates: std::collections::HashSet<String> = Default::default();
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
        }
        let mut added = false;
        for (_, def, args) in candidates {
            if demanded >= MAX_DEMANDED_DTOR_BODIES {
                break;
            }
            // Only local types: for any other crate's type the
            // interpreter cannot assume the collected body is the
            // whole story (its value model owns the std types whose
            // glue it can run at all). Type definitions are not
            // items, so locality is judged by the name's crate
            // segment, the same test the glue builder applies.
            if def.name().split("::").next() != Some(local_crate_name) {
                continue;
            }
            let key = format!("{:?} {:?}", def.def_id(), args);
            if !seen_candidates.insert(key) {
                continue;
            }
            if args_mention_params(&args) {
                continue;
            }
            let Some(impl_fn) = impl_methods.get(&def.name()) else {
                continue;
            };
            // The impl's generic parameters are the self type's, in
            // declaration order, for the standard
            // `impl<...> Drop for Type<...>` shape; a resolution
            // failure simply means the shape differs or this ADT
            // has no applicable impl, and a compiler panic on an
            // odd instantiation must not take the whole conversion
            // down with it.
            let impl_fn = *impl_fn;
            let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Instance::resolve(impl_fn, &args)
                    .ok()
                    .filter(|inst| inst.has_body())
                    .and_then(|inst| inst.body().map(|body| (inst, body)))
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
    report.functions.push(FnReport {
        name,
        blocks: 0,
        locals: 0,
        arg_count: 0,
        outcome: FnOutcome::Converted,
    });
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
                        match guard_coroutine_closure_query(|| item.body()) {
                            Ok(Some(body)) => convert_local_body(
                                &mut conv,
                                &mut report,
                                &mut bodies,
                                &mut body_def_ids,
                                &mut body_def_names,
                                &mut seed_bodies,
                                name,
                                def_id,
                                body,
                            ),
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
        let body = match guard_coroutine_closure_query(|| instance.body()) {
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
    demand_drop_instances(
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
        &closure_scan,
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
    // Local bodies converted by the closure and coroutine passes
    // have not been scanned for the *function* instances they
    // demand (each demand pass scans only for its own kind of
    // target). The fixpoint loop below scans them with the
    // instance demand, and each round's newly converted closure
    // and coroutine bodies become the next round's scan set, so
    // every converted body is eventually scanned by every pass.
    let mut local_scan: Vec<rustc_public::mir::Body> = closure_seeds.clone();
    local_scan.extend(coroutine_seeds.iter().cloned());
    let mut reach_seeds = seed_bodies;
    reach_seeds.extend(instance_seeds);
    reach_seeds.extend(closure_seeds);
    reach_seeds.extend(coroutine_seeds);

    // Collect the bodies of non-local functions reachable from the
    // converted local bodies (std / core / alloc and other library
    // functions), so calls into them have bodies in the program too.
    let reached = sparagmos_driver::reach::collect_reachable(
        &mut conv,
        &reach_seeds,
        &local_defs,
        sparagmos_driver::reach::MAX_COLLECTED_BODIES,
    );
    let mut foreign_taken = reached.bodies.len();
    let mut frontier = reached.walked_bodies;
    bodies.extend(reached.bodies);
    body_def_ids.extend(reached.body_def_ids);
    body_def_names.extend(reached.body_def_names);
    report.functions.extend(reached.reports);

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
    // name for report entries, and both collections keep their
    // per-program caps across passes (shared demand state, and a
    // remaining-budget argument for reachability).
    let mut collected_ids: HashSet<sparagmos_mir::DefId> =
        body_def_ids.iter().copied().collect();
    // Stable MIR sources for the trait-key alias pass below: the
    // local seeds plus every foreign body the reachability passes
    // walk, across all fixpoint rounds.
    let mut alias_sources = reach_seeds.clone();
    alias_sources.extend(frontier.iter().cloned());
    let mut report_names: HashSet<String> =
        report.functions.iter().map(|f| f.name.clone()).collect();
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
            &closure_scan,
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
        if new_instances.is_empty()
            && new_closure_bodies.is_empty()
            && new_coroutine_bodies.is_empty()
        {
            break;
        }
        local_scan = new_closure_bodies.clone();
        local_scan.extend(new_coroutine_bodies.iter().cloned());
        for id in &body_def_ids[ids_before..] {
            collected_ids.insert(*id);
        }
        if foreign_taken >= sparagmos_driver::reach::MAX_COLLECTED_BODIES {
            break;
        }
        let mut reach_input = new_instances;
        reach_input.extend(new_closure_bodies.iter().cloned());
        reach_input.extend(new_coroutine_bodies.iter().cloned());
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
    apply_instance_demands(&mut report, &deferred, &demand_state.demands);
    apply_instance_demands(&mut report, &deferred_closures, &demand_state.demands);

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
                    guard_coroutine_closure_query(|| fn_def.body())
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
    // precedence), and the checks of `sparagmos_driver::mono_check`
    // over the converted bodies (invalid SIMD intrinsic
    // instantiations and SIMD layouts, types too big for the
    // target architecture). A failure here fails the whole
    // compilation, the way the corresponding rustc error does.
    // The scan covers the Stable MIR bodies conversion walked
    // (local seeds, demanded instances, and the reachability
    // frontier); auxiliary-crate bodies converted by the loop
    // above are not retained in that set and are not scanned.
    let compile_error = demand_state
        .recursion_limit
        .clone()
        .or_else(|| sparagmos_driver::mono_check::check_bodies(&alias_sources));

    stash(Collected {
        report,
        bodies,
        body_def_ids,
        body_def_names,
        local_body_count,
        local_report_count,
        def_names: conv.def_names.clone(),
        adt_discriminants: conv.adt_discriminants.clone(),
        compile_error,
        adt_layouts: conv.adt_layouts.clone(),
        adt_shapes: conv.adt_shapes.clone(),
    });
    ControlFlow::Continue(())
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
fn execute(c: &Collected, guest_args: &[String]) -> i32 {
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
    program.adt_discriminants = c.adt_discriminants.clone();
    program.adt_layouts = c.adt_layouts.clone();
    program.adt_shapes = c.adt_shapes.clone();
    let Some(entry) = c.bodies[..c.local_body_count]
        .iter()
        .map(|(n, _)| n)
        .find(|n| is_main(n))
    else {
        for f in &c.report.functions {
            if is_main(&f.name) {
                if let FnOutcome::Skipped { reasons } = &f.outcome {
                    let rs: Vec<String> =
                        reasons.iter().map(|(k, v)| format!("{k} x{v}")).collect();
                    eprintln!(
                        "sparagmos-driver run: unsupported: `main` was not converted ({})",
                        rs.join(", ")
                    );
                    return 2;
                }
            }
        }
        eprintln!(
            "sparagmos-driver run: unsupported: no `main` function among the converted functions"
        );
        return 2;
    };
    let config = sparagmos_interp::InterpConfig {
        guest_args: guest_args.to_vec(),
        ..sparagmos_interp::InterpConfig::default()
    };
    let run = sparagmos_interp::interpret_mir_captured_with_config(
        &program,
        entry,
        Vec::new(),
        config,
    );
    // The guest's stdout is this process's stdout, whatever the
    // outcome was.
    {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(run.stdout.as_bytes());
        let _ = out.flush();
    }
    match run.result {
        Ok(sparagmos_interp::Value::Unit) => 0,
        Ok(other) => {
            eprintln!(
                "sparagmos-driver run: unsupported: entry `{entry}` returned {other}, expected ()"
            );
            2
        }
        Err(sparagmos_interp::MirRunError::Interp(sparagmos_interp::InterpError::Panic {
            message,
            ..
        })) => {
            eprintln!("thread 'main' panicked: {message}");
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
    // compiles each auxiliary crate with its own file's flags).
    for flag in declared_compile_flags(&src_text) {
        cmd.arg(flag);
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

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let run_mode = argv.get(1).map(|a| a == "run").unwrap_or(false);
    let mut input: Option<String> = None;
    let mut aux_specs: Vec<AuxSpec> = Vec::new();
    let mut dump_path: Option<String> = None;
    let mut edition = "2021".to_string();
    let mut sysroot: Option<String> = None;
    let mut rustc_args: Vec<String> = Vec::new();
    let mut guest_args: Vec<String> = Vec::new();
    let mut i = if run_mode { 2 } else { 1 };
    while i < argv.len() {
        match argv[i].as_str() {
            "--dump" => {
                i += 1;
                dump_path = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--dump needs a path");
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
            "--sysroot" => {
                i += 1;
                sysroot = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--sysroot needs a path");
                    std::process::exit(2);
                }));
            }
            "-h" | "--help" => {
    println!("usage: sparagmos-driver [run] <file.rs> [--aux <name>=<file.rs>[;crate-type=<type>][;edition=<ed>]] [--rustc-arg <arg>] [--guest-arg <arg>] [--dump <path>] [--edition <ed>] [--sysroot <path>]");
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
        eprintln!("usage: sparagmos-driver [run] <file.rs> [--dump <path>]");
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
                if run_mode {
                    eprintln!("sparagmos-driver run: unsupported: {reason}");
                    std::process::exit(2);
                }
                eprintln!("sparagmos-driver: {reason}");
                std::process::exit(1);
            }
        }
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
    if run_mode && !caller_sets_crate_type {
        args.push("--crate-type=lib".to_string());
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
    let previous_hook = begin_panic_capture();
    let mut stdout_guard = StdoutToStderr::new();
    let run_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rustc_public::run!(args.as_slice(), collect)
    }));
    stdout_guard.restore();
    end_panic_capture(previous_hook);
    let result = match run_outcome {
        Ok(result) => result,
        Err(payload) => {
            let reason = take_captured_panic()
                .or_else(|| payload_reason(&*payload))
                .unwrap_or_else(|| "panic with a non-string payload".to_string());
            if run_mode {
                eprintln!(
                    "sparagmos-driver run: unsupported: Stable MIR conversion panicked: {reason}"
                );
                std::process::exit(2);
            }
            eprintln!("sparagmos-driver: Stable MIR conversion panicked: {reason}");
            std::process::exit(1);
        }
    };
    let collected = take();
    match (result, collected) {
        (Ok(()), Some(c)) => {
            if let Some(e) = &c.compile_error {
                eprintln!("sparagmos-driver: compilation failed: {e}");
                // Same contract as a front-end failure: exit 1 in
                // convert mode, exit 2 (unsupported) in run mode.
                std::process::exit(if run_mode { 2 } else { 1 });
            }
            if run_mode {
                if let Some(path) = &dump_path {
                    let text =
                        sparagmos_driver::dump::render(&input, &c.report, &c.bodies, &c.def_names);
                    if let Err(e) = std::fs::write(path, text) {
                        eprintln!("failed to write dump {path}: {e}");
                        std::process::exit(1);
                    }
                }
                std::process::exit(execute(&c, &guest_args));
            }
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
                }
            }
            println!(
                "converted: {}, skipped: {}",
                c.report.converted_count(),
                c.report.skipped_count()
            );
            {
                let target = &c.report.functions[..c.local_report_count];
                let t_converted = target
                    .iter()
                    .filter(|f| matches!(f.outcome, FnOutcome::Converted))
                    .count();
                let t_skipped = target.len() - t_converted;
                println!("target-crate converted: {t_converted}, skipped: {t_skipped}");
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
            if let Some(path) = &dump_path {
                let text =
                    sparagmos_driver::dump::render(&input, &c.report, &c.bodies, &c.def_names);
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
        (Err(e), _) => {
            eprintln!("sparagmos-driver: compilation failed: {e:?}");
            // In run mode a failed compilation is a conversion failure,
            // which the run contract reports as exit 2.
            std::process::exit(if run_mode { 2 } else { 1 });
        }
    }
}
