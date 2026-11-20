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
//! `//@ aux-build:` / `//@ aux-crate:` directives do. Pass auxiliary
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

use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;
use std::sync::{Mutex, OnceLock};

use rustc_public::mir::mono::Instance;
use rustc_public::ty::{
    ClosureKind, ExistentialPredicate, GenericArgKind, GenericArgs, IntTy, RigidTy, Ty,
    TyConstKind, TyKind,
};
use rustc_public::CrateDef;

use sparagmos_driver::convert::Converter;
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
    match conv.convert_body(&body) {
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
/// body to collect — the same limitation as generic functions —
/// and stay skipped. Async closures are coroutine closures; their
/// bodies are coroutines, which are not converted.
fn closure_instance(item: &rustc_public::CrateItem) -> ClosureAttempt {
    match item.ty().kind() {
        TyKind::RigidTy(RigidTy::CoroutineClosure(..)) => ClosureAttempt::Unsupported(
            "async closure body is not converted (coroutine closures are unsupported)".to_string(),
        ),
        TyKind::RigidTy(RigidTy::Closure(def, args)) => {
            if args_mention_params(&args) {
                return ClosureAttempt::Unsupported(
                    "closure is defined in a generic context (its generic arguments \
                     still mention parameters); only monomorphic closures are converted"
                        .to_string(),
                );
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

/// The closure kind (`Fn` / `FnMut` / `FnOnce`) recorded in a
/// closure type's generic arguments. rustc appends three synthetic
/// arguments to a closure's parent arguments — the kind, the
/// signature as a function-pointer type, and the tupled captures —
/// and encodes the kind itself as an integer type: `i8` for `Fn`,
/// `i16` for `FnMut`, `i32` for `FnOnce` (this is rustc's
/// `Ty::from_closure_kind`; the encoding is part of the Stable MIR
/// boundary this crate pins to Rust 1.100.0).
fn closure_kind(args: &GenericArgs) -> Option<ClosureKind> {
    let kind_arg = args.0.iter().rev().nth(2)?;
    let GenericArgKind::Type(ty) = kind_arg else {
        return None;
    };
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Int(IntTy::I8)) => Some(ClosureKind::Fn),
        TyKind::RigidTy(RigidTy::Int(IntTy::I16)) => Some(ClosureKind::FnMut),
        TyKind::RigidTy(RigidTy::Int(IntTy::I32)) => Some(ClosureKind::FnOnce),
        _ => None,
    }
}

/// Whether any generic argument still mentions a type or const
/// parameter, i.e. the item is not fully monomorphic.
fn args_mention_params(args: &GenericArgs) -> bool {
    args.0.iter().any(|arg| match arg {
        GenericArgKind::Type(ty) => ty_mentions_param(ty),
        GenericArgKind::Const(konst) => const_mentions_param(konst),
        GenericArgKind::Lifetime(_) => false,
    })
}

fn const_mentions_param(konst: &rustc_public::ty::TyConst) -> bool {
    match konst.kind() {
        TyConstKind::Param(_) => true,
        TyConstKind::Unevaluated(_, args) => args_mention_params(args),
        TyConstKind::Value(ty, _) | TyConstKind::ZSTValue(ty) => ty_mentions_param(ty),
        _ => false,
    }
}

fn ty_mentions_param(ty: &Ty) -> bool {
    match ty.kind() {
        TyKind::Param(_) => true,
        TyKind::Alias(_, alias) => args_mention_params(&alias.args),
        TyKind::RigidTy(rigid) => match rigid {
            RigidTy::Ref(_, inner, _) | RigidTy::RawPtr(inner, _) | RigidTy::Slice(inner) => {
                ty_mentions_param(&inner)
            }
            RigidTy::Array(inner, len) => ty_mentions_param(&inner) || const_mentions_param(&len),
            RigidTy::Pat(inner, _) => ty_mentions_param(&inner),
            RigidTy::Tuple(elems) => elems.iter().any(ty_mentions_param),
            RigidTy::FnPtr(sig) => sig.value.inputs_and_output.iter().any(ty_mentions_param),
            RigidTy::Adt(_, args)
            | RigidTy::FnDef(_, args)
            | RigidTy::Closure(_, args)
            | RigidTy::Coroutine(_, args)
            | RigidTy::CoroutineClosure(_, args)
            | RigidTy::CoroutineWitness(_, args) => args_mention_params(&args),
            RigidTy::Dynamic(bounds, _) => bounds.iter().any(|b| match &b.value {
                ExistentialPredicate::Trait(t) => args_mention_params(&t.generic_args),
                ExistentialPredicate::Projection(p) => args_mention_params(&p.generic_args),
                ExistentialPredicate::AutoTrait(_) => false,
            }),
            _ => false,
        },
        _ => false,
    }
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

    let items = rustc_public::all_local_items();
    for item in items.iter() {
        let name = item.name();
        let item_instance = Instance::try_from(item.clone());
        // Assign (or fetch) this item's Sparagmos DefId up front, so a
        // converted body can be handed to the interpreter together with
        // the id that other bodies use to call it. Function items are
        // keyed by their resolved instance (the same key `FnDef` call
        // targets resolve through); other items keep the definition's
        // own id.
        let def_id = match &item_instance {
            Ok(inst) => conv.instance_def_id(item.def_id(), inst),
            Err(_) => conv.crate_def_id(item),
        };
        let instance = match item_instance {
            Ok(i) => i,
            Err(_) => match closure_instance(item) {
                ClosureAttempt::Instance(i) => i,
                ClosureAttempt::NotClosure => {
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
        let Some(body) = instance.body() else {
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
        };
        let blocks = body.blocks.len();
        let locals = body.locals().len();
        let arg_count = body.arg_locals().len();
        match conv.convert_body(&body) {
            Ok(converted) => {
                seed_bodies.push(body);
                bodies.push((name.clone(), converted));
                body_def_ids.push(def_id);
                body_def_names.push(item.def_id().name());
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

    let local_body_count = bodies.len();
    let local_report_count = report.functions.len();

    // Collect the bodies of non-local functions reachable from the
    // converted local bodies (std / core / alloc and other library
    // functions), so calls into them have bodies in the program too.
    let reached = sparagmos_driver::reach::collect_reachable(&mut conv, &seed_bodies, &local_defs);
    bodies.extend(reached.bodies);
    body_def_ids.extend(reached.body_def_ids);
    body_def_names.extend(reached.body_def_names);
    report.functions.extend(reached.reports);

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
                    Ok(fn_def.body())
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



    stash(Collected {
        report,
        bodies,
        body_def_ids,
        body_def_names,
        local_body_count,
        local_report_count,
        def_names: conv.def_names.clone(),
        adt_discriminants: conv.adt_discriminants.clone(),
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
fn execute(c: &Collected) -> i32 {
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
    let run = sparagmos_interp::interpret_mir_captured(&program, entry, Vec::new());
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
/// a source file: the first quoted string after the first
/// occurrence of the attribute name.
fn declared_inner_attr(src: &str, attr: &str) -> Option<String> {
    let idx = src.find(attr)?;
    let rest = &src[idx + attr.len()..];
    let q1 = rest.find('"')?;
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('"')?;
    Some(rest[..q2].to_string())
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

/// One compiled auxiliary crate.
struct AuxBuilt {
    name: String,
    /// Artifact to link with `--extern`; `None` for crate types
    /// that expose no Rust API (cdylib and friends).
    artifact: Option<std::path::PathBuf>,
    /// Whether the crate's functions are guest code to convert
    /// (false for proc-macro and cdylib crates).
    collect: bool,
}

/// Compile one auxiliary source file with the pinned rustc, as a
/// library crate named `name`, into `aux_dir`. Previously built
/// auxiliary crates are visible through `--extern`, so callers pass
/// the crates in dependency order. On failure, returns the first
/// compiler error line as the reason.
fn compile_aux(
    sysroot: &str,
    edition: &str,
    aux_dir: &std::path::Path,
    name: &str,
    src_path: &str,
    previous: &[AuxBuilt],
) -> Result<AuxBuilt, String> {
    let src_text = std::fs::read_to_string(src_path)
        .map_err(|e| format!("cannot read auxiliary source {src_path}: {e}"))?;
    let declared = declared_crate_type(&src_text);
    let effective_type = declared.as_deref().unwrap_or("rlib");
    let declared_name = declared_crate_name(&src_text);
    let effective_name = declared_name.as_deref().unwrap_or(name);
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
        cmd.arg("--crate-type=rlib");
    }
    if declared_name.is_none() {
        cmd.arg(format!("--crate-name={name}"));
    }
    for prev in previous {
        if let Some(artifact) = &prev.artifact {
            cmd.arg("--extern")
                .arg(format!("{}={}", prev.name, artifact.display()));
        }
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
    // Locate the produced artifact: lib<name>.rlib / .rmeta / .so.
    let mut artifact: Option<std::path::PathBuf> = None;
    let mut best_rank = u8::MAX;
    if let Ok(entries) = std::fs::read_dir(aux_dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().into_owned();
            if !fname.starts_with(&format!("lib{effective_name}.")) {
                continue;
            }
            let rank = match fname.rsplit('.').next() {
                Some("rlib") => 0,
                Some("rmeta") => 1,
                Some("so") | Some("dylib") | Some("dll") => 2,
                _ => continue,
            };
            if rank < best_rank {
                best_rank = rank;
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
        artifact: if link { artifact } else { None },
        collect,
    })
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let run_mode = argv.get(1).map(|a| a == "run").unwrap_or(false);
    let mut input: Option<String> = None;
    let mut aux_specs: Vec<(String, String)> = Vec::new();
    let mut dump_path: Option<String> = None;
    let mut edition = "2021".to_string();
    let mut sysroot: Option<String> = None;
    let mut rustc_args: Vec<String> = Vec::new();
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
            "--aux" => {
                i += 1;
                let spec = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--aux needs a <name>=<file.rs> value");
                    std::process::exit(2);
                });
                match spec.split_once('=') {
                    Some((name, path)) if !name.is_empty() && !path.is_empty() => {
                        aux_specs.push((name.to_string(), path.to_string()));
                    }
                    _ => {
                        eprintln!("--aux needs a <name>=<file.rs> value, got: {spec}");
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
    println!("usage: sparagmos-driver [run] <file.rs> [--aux <name>=<file.rs>] [--rustc-arg <arg>] [--dump <path>] [--edition <ed>] [--sysroot <path>]");
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
    for (name, src) in &aux_specs {
        match compile_aux(&sysroot, &edition, &aux_dir, name, src, &aux_built) {
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
    args.extend(rustc_args);
    if !aux_built.is_empty() {
        args.push("-L".to_string());
        args.push(format!("dependency={}", aux_dir.display()));
        for built in &aux_built {
            if let Some(artifact) = &built.artifact {
                args.push("--extern".to_string());
                args.push(format!("{}={}", built.name, artifact.display()));
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
            if run_mode {
                if let Some(path) = &dump_path {
                    let text =
                        sparagmos_driver::dump::render(&input, &c.report, &c.bodies, &c.def_names);
                    if let Err(e) = std::fs::write(path, text) {
                        eprintln!("failed to write dump {path}: {e}");
                        std::process::exit(1);
                    }
                }
                std::process::exit(execute(&c));
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
