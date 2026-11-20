//! `sparagmos-driver` — the Sparagmos rustc driver.
//!
//! Usage:
//! - `sparagmos-driver <file.rs> [--dump <path>] [--edition <ed>]` —
//!   convert and print a per-function summary.
//! - `sparagmos-driver run <file.rs> [--edition <ed>]` — convert and
//!   execute the program's `main` in the Sparagmos interpreter
//!   (`sparagmos-interp`), in this process.
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
    def_names: Vec<String>,
    /// Discriminant values of the converted enum ADTs, for the
    /// interpreter (see `Converter::adt_discriminants`).
    adt_discriminants: sparagmos_mir::AdtDiscriminants,
}

static COLLECTED: OnceLock<Mutex<Option<Collected>>> = OnceLock::new();

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

fn collect() -> ControlFlow<()> {
    let mut conv = Converter::new();
    let mut report = ConversionReport::default();
    let mut bodies: Vec<(String, Body)> = Vec::new();
    let mut body_def_ids: Vec<sparagmos_mir::DefId> = Vec::new();

    let items = rustc_public::all_local_items();
    for item in items.iter() {
        let name = item.name();
        // Assign (or fetch) this item's Sparagmos DefId up front, so a
        // converted body can be handed to the interpreter together with
        // the id that other bodies use to call it.
        let def_id = conv.crate_def_id(item);
        let instance = match Instance::try_from(item.clone()) {
            Ok(i) => i,
            Err(_) => {
                let mut reasons = BTreeMap::new();
                reasons.insert(
                    format!("no monomorphic instance available (kind={:?})", item.kind()),
                    1,
                );
                report.functions.push(FnReport {
                    name,
                    blocks: 0,
                    locals: 0,
                    arg_count: 0,
                    outcome: FnOutcome::Skipped { reasons },
                });
                continue;
            }
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
                bodies.push((name.clone(), converted));
                body_def_ids.push(def_id);
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

    stash(Collected {
        report,
        bodies,
        body_def_ids,
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
    let defined: BTreeSet<&str> = c.bodies.iter().map(|(n, _)| n.as_str()).collect();
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
    let Some(entry) = c.bodies.iter().map(|(n, _)| n).find(|n| is_main(n)) else {
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

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let run_mode = argv.get(1).map(|a| a == "run").unwrap_or(false);
    let mut input: Option<String> = None;
    let mut dump_path: Option<String> = None;
    let mut edition = "2021".to_string();
    let mut sysroot: Option<String> = None;
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
            "--sysroot" => {
                i += 1;
                sysroot = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--sysroot needs a path");
                    std::process::exit(2);
                }));
            }
            "-h" | "--help" => {
                println!("usage: sparagmos-driver [run] <file.rs> [--dump <path>] [--edition <ed>] [--sysroot <path>]");
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
    let args: Vec<String> = vec![
        "sparagmos-driver".to_string(),
        input.clone(),
        format!("--edition={edition}"),
        "--crate-type=lib".to_string(),
        "--emit=metadata".to_string(),
        format!("--out-dir={}", out_dir.display()),
        format!("--sysroot={sysroot}"),
    ];
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
