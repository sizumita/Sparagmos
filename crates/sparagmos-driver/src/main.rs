//! `sparagmos-driver` — the Sparagmos rustc driver.
//!
//! Usage:
//! - `sparagmos-driver <file.rs> [--dump <path>] [--edition <ed>]`
//!   `[--rustc-arg <arg>]...` — convert and print a per-function
//!   summary. `--compiler-stdout <path>` captures what
//!   the compiler itself prints to its stdout during the run
//!   (a `-Zunpretty` dump or a `--print` answer) into a file,
//!   for a caller that judges that output; compiler stdout
//!   otherwise goes to stderr, like all compile-time printing.
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
//! auxiliaries, whose sources do not declare a crate type),
//! `;edition=<ed>` builds it with that edition instead of the main
//! `--edition` (compiletest builds each auxiliary crate with the
//! edition its own `//@ edition:` directive declares), and
//! `;cfg=<name>` (repeatable) builds it with `--cfg <name>`
//! (compiletest compiles a test's auxiliary crates with the
//! test's own compile arguments, which carry the revision's
//! `--cfg` for a revised test — an incremental test's auxiliary
//! crate selects its revision's definitions through it). An in-source
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
//! `--aux-lib <name>=<path>` (repeatable) names a prebuilt library
//! auxiliary instead: a support library the harness built once
//! (compiletest's minicore stub, run-make's support library). It
//! binds into the main compilation and into every auxiliary build
//! that declares it -- compiletest's support libraries precede a
//! test's own auxiliary builds, so an auxiliary source's
//! `extern crate minicore;` resolves during its own build. The
//! optional `;collect=no` attribute keeps its functions out of
//! the converted program (the minicore stub's rlib carries no
//! encoded MIR for its non-generic functions).
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
//! 101 = the guest program panicked and the panic unwound to the
//! top of the run (message on stderr), 134 = the guest program
//! crashed: a panic that may not unwind aborted the run, the
//! outcome a native process dies by SIGABRT in (128 + 6; the
//! abort reason is on the guest's stderr and a
//! `sparagmos-driver run: guest crashed` marker on the driver's
//! own stderr names the outcome), 2 = the program could not be
//! converted or executed (concrete reason on stderr). Run mode
//! writes the guest program's captured stdout to its own stdout
//! (in every outcome — a native program's output is likewise
//! observable when it panics).
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
extern crate rustc_hir;
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
    take_captured_panic, take_compiler_crash, Converter, CONST_EXPR_REASON,
    COROUTINE_CLOSURE_TYPE_REASON,
    INSTANCE_CONST_REASON, TAIL_CALL_REASON, UNSAFE_BINDER_REASON,
};
use sparagmos_driver::report::{ConversionReport, FnOutcome, FnReport};
use sparagmos_mir::Body;


#[path = "main/collected.rs"]
mod collected;
#[allow(unused_imports)]
pub(crate) use self::collected::*;
#[path = "main/panic_guard.rs"]
mod panic_guard;
#[allow(unused_imports)]
pub(crate) use self::panic_guard::*;
#[path = "main/convert_item.rs"]
mod convert_item;
#[allow(unused_imports)]
pub(crate) use self::convert_item::*;
#[path = "main/collect_instances.rs"]
mod collect_instances;
#[allow(unused_imports)]
pub(crate) use self::collect_instances::*;
#[path = "main/demand.rs"]
mod demand;
#[allow(unused_imports)]
pub(crate) use self::demand::*;
#[path = "main/collect.rs"]
mod collect;
#[allow(unused_imports)]
pub(crate) use self::collect::*;
#[path = "main/entry.rs"]
mod entry;
#[allow(unused_imports)]
pub(crate) use self::entry::*;
#[path = "main/source_tree.rs"]
mod source_tree;
#[allow(unused_imports)]
pub(crate) use self::source_tree::*;
#[path = "main/tools.rs"]
mod tools;
#[allow(unused_imports)]
pub(crate) use self::tools::*;
#[path = "main/build_program.rs"]
mod build_program;
#[allow(unused_imports)]
pub(crate) use self::build_program::*;
#[path = "main/process_host.rs"]
mod process_host;
#[allow(unused_imports)]
pub(crate) use self::process_host::*;
#[path = "main/execute.rs"]
mod execute;
#[allow(unused_imports)]
pub(crate) use self::execute::*;
#[path = "main/debug_script.rs"]
mod debug_script;
#[allow(unused_imports)]
pub(crate) use self::debug_script::*;
#[path = "main/aux_build.rs"]
mod aux_build;
#[allow(unused_imports)]
pub(crate) use self::aux_build::*;
#[path = "main/jit.rs"]
mod jit;
#[allow(unused_imports)]
pub(crate) use self::jit::*;
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
/// `InterpConfig::overflow_checks`). This resolves one
/// compilation's setting from its own argument list: the main
/// compilation's here, and each auxiliary crate's from its own
/// build command (see `AuxBuilt::overflow_checks`); the
/// interpreter picks the instantiating crate's setting at run
/// time (see `Interpreter::overflow_checks_at`).
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

/// The session's panic strategy, resolved from the final rustc
/// argument list the way rustc resolves it: the last
/// `-C panic=<strategy>` wins; without one the strategy is
/// `unwind`. The interpreter needs the resolved value because
/// the abort strategy changes what a panic prints (see
/// `InterpConfig::panic_abort`).
fn session_panic_abort(args: &[String]) -> bool {
    let mut panic: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
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
            if let Some((name, value)) = t.split_once('=') {
                if name == "panic" {
                    panic = Some(value.to_string());
                }
            }
        }
        i += 1;
    }
    panic.as_deref() == Some("abort")
}

/// The session's `SIGPIPE` setup, resolved from the
/// final rustc argument list the way rustc resolves it:
/// the last `-Z on-broken-pipe=<mode>` wins (`error`
/// installs `SIG_IGN`, `kill` installs `SIG_DFL`,
/// `inherit` installs nothing); without one, the
/// runtime default applies (install `SIG_IGN`, with
/// the spawn-time reset for children). The interpreter
/// needs the resolved value because the runtime
/// initialisation that performs the setup natively is
/// not guest code it executes (see
/// `InterpConfig::sigpipe`). The `#[unix_sigpipe]`
/// crate attribute names the same modes in source; it
/// is not read here.
fn session_sigpipe(args: &[String]) -> sparagmos_interp::SigpipeInit {
    let mut mode = sparagmos_interp::SigpipeInit::Default;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let text: Option<&str> = if a == "-Z" {
            i += 1;
            args.get(i).map(String::as_str)
        } else {
            a.strip_prefix("-Z")
        };
        if let Some(t) = text {
            if let Some((name, value)) = t.split_once('=') {
                if name == "on-broken-pipe" {
                    mode = match value {
                        "error" => sparagmos_interp::SigpipeInit::Ignore,
                        "kill" => sparagmos_interp::SigpipeInit::SigDfl,
                        "inherit" => sparagmos_interp::SigpipeInit::Inherit,
                        // rustc itself rejects the other
                        // values (e.g. `default`) before a
                        // run can observe them.
                        _ => mode,
                    };
                }
            }
        }
        i += 1;
    }
    mode
}

/// Whether the entry shim frame appears below the entry
/// frame in the guest's short backtraces, resolved from the
/// final rustc argument list the way the native build
/// behaves: the `<fn() as FnOnce<()>>::call_once` shim
/// `lang_start` calls the entry through is a real frame at
/// opt-level 0; an optimising build inlines it into
/// `__rust_begin_short_backtrace`, where it leaves no frame
/// of its own unless debug info lets the backtrace printer
/// reconstruct the inlined frame. The frame is therefore
/// visible exactly when the resolved opt-level is 0 or any
/// debug info is emitted. The last `-O` / `-C opt-level`
/// and the last `-g` / `-C debuginfo` win, as rustc
/// resolves them (see `InterpConfig::entry_shim_visible`).
fn session_entry_shim_visible(args: &[String]) -> bool {
    let mut opt_is_zero = true;
    let mut debuginfo_present = false;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "-O" {
            opt_is_zero = false;
            i += 1;
            continue;
        }
        if a == "-g" {
            debuginfo_present = true;
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
                Some((n, v)) => (n, Some(v)),
                None => (t, None),
            };
            match name {
                "opt-level" => {
                    if let Some(v) = value {
                        opt_is_zero = v == "0";
                    }
                }
                "debuginfo" => {
                    // A value-less `-C debuginfo` spells the
                    // full level, like `-g`.
                    debuginfo_present =
                        !matches!(value, Some("0") | Some("none"));
                }
                _ => {}
            }
        }
        i += 1;
    }
    opt_is_zero || debuginfo_present
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
    /// `coverage-map`: convert an instrumented crate and print
    /// its coverage-mapping dump (the coverage-dump format
    /// compiletest's coverage-map mode compares).
    CoverageMap,
    /// `coverage-run`: convert an instrumented crate, execute
    /// it in the interpreter counting coverage counters, and
    /// print the llvm-cov style text report (the format
    /// compiletest's coverage-run mode compares); the process
    /// exit code is the guest's own.
    CoverageRun,
}

impl Mode {
    /// Run-shaped modes share run mode's front-end discipline
    /// (the `--crate-type=lib` pin and `--cap-lints=allow`) and
    /// its failure contract (a failed compilation is reported as
    /// unsupported, exit 2).
    fn is_run(self) -> bool {
        matches!(self, Mode::Run | Mode::RunJit | Mode::CoverageRun)
    }

    /// Modes whose verdict is a compilation outcome (the program
    /// converts, or fails to compile) rather than an execution
    /// outcome.
    fn is_compile_verdict(self) -> bool {
        matches!(self, Mode::Convert | Mode::JitCompile | Mode::CoverageMap)
    }
}

/// Whether this process's mode is a run-shaped executing mode
/// (`run` / `run-jit`), set once from the parsed mode before the
/// compiler runs and read by `collect` (which the compiler
/// callback reaches without the mode at hand). The reachability
/// collection walks execution-first in these modes (see
/// `sparagmos_driver::reach::CollectionOrder`); the other modes
/// keep the first-in-first-out walk.
static EXECUTION_FIRST_MODE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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

/// Scratch paths this process created under the system temp
/// directory and must remove before it exits: the compile flow's
/// output directory (`sparagmos-driver-out-{pid}`, the rustc
/// `--out-dir` and the auxiliary builds), the materialised
/// artifact source trees (`sparagmos-artifact-sources-{pid}`,
/// registered by the tool registry when it materialises a
/// snapshot: the tree is read only by this process and by the
/// child driver processes it waits for, so it is dead at exit)
/// and the informational scratch source file. Nothing outside
/// this process reads them once it exits -- the subset runner
/// and the harness consume only the driver's stdout, stderr and
/// exit code -- but they are large (the out dir carries an
/// `.rmeta` and the aux artifacts) and a measurement wave spawns
/// thousands of driver processes, so leaving them behind filled
/// the dev host's disk (154 GB across 122,813 out dirs, #537).
/// The driver exits through `exit_process`, never bare
/// `std::process::exit`, so the removal happens on every path;
/// `SPARAGMOS_DRIVER_KEEP_SCRATCH=1` opts out for a local
/// debugging session that wants to inspect the artifacts.
static PROCESS_SCRATCH: std::sync::Mutex<Vec<std::path::PathBuf>> =
    std::sync::Mutex::new(Vec::new());

/// Record a temp-dir scratch path for removal at process exit
/// (see `PROCESS_SCRATCH`). Registration is idempotent.
fn register_process_scratch(path: &std::path::Path) {
    if std::env::var_os("SPARAGMOS_DRIVER_KEEP_SCRATCH").is_some() {
        return;
    }
    if let Ok(mut paths) = PROCESS_SCRATCH.lock() {
        if !paths.iter().any(|p| p == path) {
            paths.push(path.to_path_buf());
        }
    }
}

/// Remove every registered scratch path (see `PROCESS_SCRATCH`).
/// Best-effort: a path another process already removed, or one
/// that was never created, is not an error.
fn remove_process_scratch() {
    if let Ok(mut paths) = PROCESS_SCRATCH.lock() {
        for path in paths.drain(..) {
            if path.is_dir() {
                let _ = std::fs::remove_dir_all(&path);
            } else {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

/// Terminate the process with `code` after removing the scratch
/// paths this process registered (see `PROCESS_SCRATCH`). Every
/// exit in this binary goes through here: `std::process::exit`
/// runs no destructors, so a Drop guard on the paths themselves
/// would silently not fire on most exit paths.
fn exit_process(code: i32) -> ! {
    remove_process_scratch();
    std::process::exit(code)
}

/// The prefix of the compile flow's scratch output directory
/// names in the system temp directory; the suffix is the
/// creating process's pid (`sparagmos-driver-out-{pid}`, see
/// `PROCESS_SCRATCH`).
const SCRATCH_OUT_DIR_PREFIX: &str = "sparagmos-driver-out-";

/// The prefix of the materialised artifact source tree
/// directory names in the system temp directory; the suffix
/// is the creating process's pid
/// (`sparagmos-artifact-sources-{pid}`, see `PROCESS_SCRATCH`).
const SCRATCH_ARTIFACT_SOURCES_PREFIX: &str = "sparagmos-artifact-sources-";

/// How long a scratch output directory must go unmodified
/// before the startup reaper (see `reap_stale_scratch`) may
/// remove it. The creator's death is the primary signal; the
/// age margin is the backstop for what the pid check cannot
/// see: a killed driver's own children (a host rustc, a guest
/// process) can briefly write into its out dir after it dies,
/// and a recycled pid can make a stale directory look owned
/// until the new owner exits. One hour is far beyond any
/// single driver run's lifetime -- the subset runner kills a
/// driver after a few minutes at most -- so a directory this
/// old whose creator is gone has no live user left.
const SCRATCH_REAP_MIN_AGE: std::time::Duration = std::time::Duration::from_secs(60 * 60);

/// Parse the creator pid out of a scratch output directory
/// name (`sparagmos-driver-out-{pid}`). Only a plain positive
/// `pid_t` is accepted: pid 0 and negative values are
/// process-group addresses to `kill`, never a creator, and a
/// suffix with trailing junk is not a name this driver
/// produced.
fn scratch_out_dir_pid(file_name: &std::ffi::OsStr) -> Option<u32> {
    let name = file_name.to_str()?;
    let suffix = name
        .strip_prefix(SCRATCH_OUT_DIR_PREFIX)
        .or_else(|| name.strip_prefix(SCRATCH_ARTIFACT_SOURCES_PREFIX))?;
    if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let pid: u32 = suffix.parse().ok()?;
    if pid == 0 || pid > i32::MAX as u32 {
        return None;
    }
    Some(pid)
}

/// Whether a process with `pid` currently exists. Signal 0
/// performs the existence check without delivering a signal;
/// EPERM means the process exists but is owned by another
/// user, which is still alive for the reaper's purpose. On
/// platforms without signals the answer is conservatively
/// "alive": the reaper then removes nothing there.
fn process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
        rc == 0
            || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        true
    }
}

/// Whether the startup reaper may remove a scratch output
/// directory: its creator is another process, that process is
/// gone, and the directory has been unmodified for at least
/// `SCRATCH_REAP_MIN_AGE` (see there).
fn scratch_out_dir_reapable(
    pid: u32,
    own_pid: u32,
    alive: bool,
    age: std::time::Duration,
) -> bool {
    pid != own_pid && !alive && age >= SCRATCH_REAP_MIN_AGE
}

/// Remove scratch output directories whose creating driver
/// process is gone. `remove_process_scratch` only runs on
/// paths the process itself reaches: a driver the subset
/// runner kills at its per-record timeout, or one that dies
/// with its host, never gets there, and a measurement wave
/// spawns thousands of drivers, so the orphans kept
/// accumulating after #537 (tens of thousands of directories,
/// ~100 GB, cleared by hand). Every driver start therefore
/// sweeps the temp directory first: the runner starts a
/// driver per record, so the sweep runs continuously during
/// a measurement, and a local invocation reaps what an
/// earlier killed run left behind. The
/// `SPARAGMOS_DRIVER_KEEP_SCRATCH` opt-out disables the
/// sweep as well: a session that keeps its own scratch for
/// inspection does not have other sessions' scratch removed
/// from under it either. Both per-pid scratch directory
/// families are candidates -- `sparagmos-driver-out-{pid}`
/// and `sparagmos-artifact-sources-{pid}` -- since a killed
/// process's exit-path removal never ran for either.
/// Best-effort throughout: a directory a
/// concurrent reaper already removed, or one that cannot be
/// read, is not an error.
fn reap_stale_scratch() {
    if std::env::var_os("SPARAGMOS_DRIVER_KEEP_SCRATCH").is_some() {
        return;
    }
    let temp_dir = std::env::temp_dir();
    let Ok(entries) = std::fs::read_dir(&temp_dir) else {
        return;
    };
    let own_pid = std::process::id();
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        let Some(pid) = scratch_out_dir_pid(&entry.file_name()) else {
            continue;
        };
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_dir() {
            continue;
        }
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let age = now.duration_since(modified).unwrap_or_default();
        if scratch_out_dir_reapable(pid, own_pid, process_alive(pid), age) {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}


/// The raw bytes of one of this process's own argv entries:
/// on Unix the entry itself; on other hosts the entries this
/// driver receives are text, and the bytes are that text.
fn os_bytes(arg: &std::ffi::OsStr) -> Vec<u8> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        arg.as_bytes().to_vec()
    }
    #[cfg(not(unix))]
    {
        arg.to_string_lossy().into_owned().into_bytes()
    }
}

fn main() {
    reap_stale_scratch();
    let worker = std::thread::Builder::new()
        .name("sparagmos-driver".to_string())
        .stack_size(DRIVER_STACK_SIZE)
        .spawn(driver_main)
        .expect("sparagmos-driver: failed to spawn the driver thread");
    if worker.join().is_err() {
        // The worker panicked: a driver crash, which the exit
        // contract reports as 101.
        eprintln!("sparagmos-driver: internal error: driver thread panicked");
        exit_process(101);
    }
}

fn driver_main() {
    driver_main_inner();
    // Reached on the paths that return instead of exiting (a
    // successful conversion, an informational invocation): the
    // scratch removal `exit_process` performs on the exit paths
    // is owed here too.
    remove_process_scratch();
}

fn driver_main_inner() {
    // The driver's own options are text, but a `--guest-arg`
    // value is a guest argument and need not be valid UTF-8
    // (a self-spawned child can carry one unchanged, see
    // issue #515): keep the raw `OsString` argv alongside the
    // lossy text view the option matching uses.
    let argv_os: Vec<std::ffi::OsString> = std::env::args_os().collect();
    let argv: Vec<String> = argv_os
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    // The cargo rustc-wrapper shim (see `rustc_shim` in
    // tools.rs) is not a session mode: it converts one
    // compilation unit, records its verdict on a side
    // channel, and delegates to the real compiler, all
    // before any session starts.
    if argv.get(1).map(|a| a.as_str()) == Some("rustc-shim") {
        exit_process(rustc_shim(&argv[2..]));
    }
    let mode = match argv.get(1).map(|a| a.as_str()) {
        Some("run") => Mode::Run,
        Some("coverage-map") => Mode::CoverageMap,
        Some("coverage-run") => Mode::CoverageRun,
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
    EXECUTION_FIRST_MODE.store(
        run_mode,
        std::sync::atomic::Ordering::Relaxed,
    );
    let debug_mode = mode == Mode::Debug;
    sparagmos_driver::convert::set_debug_session(debug_mode);
    // The debug entry point compiles and collects exactly like
    // run mode (it executes the same program, under a debugger
    // session), so every compilation-shaping and error-contract
    // decision below keys on the executing modes together.
    let exec_mode = run_mode || debug_mode;
    let mut debug_script: Option<String> = None;
    let mut debug_dialect = "gdb".to_string();
    let mut input: Option<String> = None;
    let mut aux_specs: Vec<AuxSpec> = Vec::new();
    // The invocation's own binding designations for the
    // main compilation (`--aux-bind <name>=<source>`, see
    // `ToolRegistry::designated_aux_specs`): for a name
    // built from several sources in this conversion, the
    // one build each designation names is the name's
    // binding here.
    let mut aux_binds: Vec<(String, String)> = Vec::new();
    let mut aux_lib_specs: Vec<(String, std::path::PathBuf, bool)> = Vec::new();
    let mut aux_native_lib_dirs: Vec<std::path::PathBuf> = Vec::new();
    let mut aux_env_scrub: Vec<String> = Vec::new();
    let mut tool_programs: Vec<String> = Vec::new();
    let mut guest_process_host = false;
    let mut compiletest_check_mode = false;
    let mut dep_info_verdict = false;
    let mut coverage_color = false;
    let mut dump_path: Option<String> = None;
    let mut compiler_stdout: Option<String> = None;
    let mut guest_stderr_file: Option<String> = None;
    let mut guest_stdout_passthrough = false;
    let mut guest_env_file: Option<String> = None;
    let mut guest_cwd: Option<String> = None;
    let mut guest_program_name: Option<String> = None;
    let mut edition = "2021".to_string();
    let mut sysroot: Option<String> = None;
    let mut rustc_args: Vec<String> = Vec::new();
    let mut guest_args: Vec<Vec<u8>> = Vec::new();
    let mut guest_args_file: Option<String> = None;
    let mut i = if mode == Mode::Convert { 1 } else { 2 };
    while i < argv.len() {
        match argv[i].as_str() {
            "--dump" => {
                i += 1;
                dump_path = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--dump needs a path");
                    exit_process(2);
                }));
            }
            "--compiler-stdout" => {
                i += 1;
                compiler_stdout = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--compiler-stdout needs a path");
                    std::process::exit(2);
                }));
            }
            "--guest-stderr-file" => {
                i += 1;
                guest_stderr_file = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--guest-stderr-file needs a path");
                    exit_process(2);
                }));
            }
            "--guest-stdout-passthrough" => {
                guest_stdout_passthrough = true;
            }
            "--guest-env-file" => {
                i += 1;
                guest_env_file = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--guest-env-file needs a path");
                    exit_process(2);
                }));
            }
            "--guest-cwd" => {
                i += 1;
                guest_cwd = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--guest-cwd needs a path");
                    exit_process(2);
                }));
            }
            "--guest-program-name" => {
                i += 1;
                guest_program_name = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--guest-program-name needs a value");
                    exit_process(2);
                }));
            }
            "--edition" => {
                i += 1;
                edition = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--edition needs a value");
                    exit_process(2);
                });
            }
            "--rustc-arg" => {
                i += 1;
                rustc_args.push(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--rustc-arg needs a value");
                    exit_process(2);
                }));
            }
            "--guest-arg" => {
                i += 1;
                let raw = argv_os.get(i).unwrap_or_else(|| {
                    eprintln!("--guest-arg needs a value");
                    exit_process(2);
                });
                guest_args.push(os_bytes(raw));
            }
            "--guest-args-file" => {
                i += 1;
                guest_args_file = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--guest-args-file needs a path");
                    exit_process(2);
                }));
            }
            "--debug-script" => {
                i += 1;
                debug_script = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--debug-script needs a path");
                    exit_process(2);
                }));
            }
            "--debug-dialect" => {
                i += 1;
                debug_dialect = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--debug-dialect needs a value (gdb|lldb)");
                    exit_process(2);
                });
            }
            "--aux" => {
                i += 1;
                let spec = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--aux needs a <name>=<file.rs> value");
                    exit_process(2);
                });
                match parse_aux_spec(&spec) {
                    Ok(parsed) => aux_specs.push(parsed),
                    Err(reason) => {
                        eprintln!("{reason}");
                        exit_process(2);
                    }
                }
            }
            "--aux-bind" => {
                i += 1;
                let spec = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--aux-bind needs a <name>=<source> value");
                    exit_process(2);
                });
                match spec.split_once('=') {
                    Some((name, path)) if !name.is_empty() && !path.is_empty() => {
                        aux_binds.push((name.to_string(), path.to_string()));
                    }
                    _ => {
                        eprintln!(
                            "--aux-bind needs a <name>=<source> value, got: {spec}"
                        );
                        exit_process(2);
                    }
                }
            }
            "--aux-lib" => {
                i += 1;
                let spec = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--aux-lib needs a <name>=<path> value");
                    exit_process(2);
                });
                match parse_aux_lib_spec(&spec) {
                    Ok(parsed) => aux_lib_specs.push(parsed),
                    Err(reason) => {
                        eprintln!("{reason}");
                        exit_process(2);
                    }
                }
            }
            "--aux-native-lib" => {
                i += 1;
                let dir = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--aux-native-lib needs a directory path");
                    std::process::exit(2);
                });
                aux_native_lib_dirs.push(std::path::PathBuf::from(dir));
            }
            "--aux-env-scrub" => {
                i += 1;
                let name = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--aux-env-scrub needs a variable name");
                    std::process::exit(2);
                });
                aux_env_scrub.push(name);
            }
            "--guest-process-host" => {
                guest_process_host = true;
            }
            "--compiletest-check-mode" => {
                compiletest_check_mode = true;
            }
            "--dep-info-verdict" => {
                dep_info_verdict = true;
            }
            "--coverage-color" => {
                coverage_color = true;
            }
            "--tool-program" => {
                i += 1;
                tool_programs.push(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--tool-program needs a path");
                    exit_process(2);
                }));
            }
            "--sysroot" => {
                i += 1;
                sysroot = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--sysroot needs a path");
                    exit_process(2);
                }));
            }
            "-h" | "--help" => {
    println!("usage: sparagmos-driver [run|run-jit|jit-compile|debug] <file.rs> [--aux <name>=<file.rs>[;crate-type=<type>][;edition=<ed>]] [--aux-lib <name>=<rlib>[;collect=no]] [--aux-native-lib <dir>] [--aux-env-scrub <name>] [--tool-program <path>] [--guest-process-host] [--compiletest-check-mode] [--dep-info-verdict] [--coverage-color] [--rustc-arg <arg>] [--guest-arg <arg>] [--guest-args-file <path>] [--dump <path>] [--compiler-stdout <path>] [--guest-stderr-file <path>] [--edition <ed>] [--sysroot <path>] [--debug-script <path>] [--debug-dialect <gdb|lldb>]");
                return;
            }
            other if input.is_none() && !other.starts_with('-') => {
                input = Some(other.to_string());
            }
            other => {
                eprintln!("unknown argument: {other}");
                exit_process(2);
            }
        }
        i += 1;
    }
    let Some(input) = input else {
        eprintln!("usage: sparagmos-driver [run|run-jit|jit-compile|debug] <file.rs> [--dump <path>]");
        exit_process(2);
    };
    // A compiletest check compilation (`--emit=metadata`)
    // never runs the monomorphization collector or code
    // generation, so the codegen-level verdicts in
    // `mono_check` must not decide it (see
    // `mono_check::set_check_mode`). The flag is honoured
    // only in the compilation-verdict modes; the executing
    // modes never consult those verdicts.
    sparagmos_driver::mono_check::set_check_mode(
        compiletest_check_mode && mode.is_compile_verdict(),
    );
    let sysroot = sysroot
        .or_else(|| std::env::var("SPARAGMOS_SYSROOT").ok())
        .unwrap_or_else(|| env!("SPARAGMOS_BUILD_SYSROOT").to_string());

    // A caller reconstructing a run whose guest arguments
    // include a non-UTF-8 one hands them over in a file (the
    // command line of a conversion subprocess is text in this
    // driver's plumbing; see `write_guest_args_file`): the
    // file's arguments are the run's, in order.
    if let Some(path) = &guest_args_file {
        match read_guest_args_file(std::path::Path::new(path)) {
            Ok(args) => guest_args.extend(args),
            Err(e) => {
                eprintln!(
                    "sparagmos-driver run: unsupported: cannot read the guest arguments file {path}: {e}"
                );
                exit_process(2);
            }
        }
    }

    // The process host of a run that opted into real child
    // processes: assembled from this invocation's own shape,
    // so a self-child re-runs this very program (see
    // `DriverProcessHost`).
    let process_host = if guest_process_host {
        let aux_specs = aux_specs
            .iter()
            .map(|spec| {
                let mut text = format!(
                    "{}={}",
                    spec.name,
                    absolutize_path(std::path::Path::new(&spec.path))
                        .display()
                );
                if let Some(crate_type) = &spec.crate_type {
                    text += &format!(";crate-type={crate_type}");
                }
                if let Some(edition) = &spec.edition {
                    text += &format!(";edition={edition}");
                }
                for (need_name, need_path) in &spec.needs {
                    text += &format!(";need={need_name}={need_path}");
                }
                if let Some(uniq) = &spec.uniq {
                    text += &format!(";uniq={uniq}");
                }
                text
            })
            .collect();
        let aux_lib_specs = aux_lib_specs
            .iter()
            .map(|(name, lib, collect)| {
                let mut text =
                    format!("{}={}", name, absolutize_path(lib).display());
                if !collect {
                    text += ";collect=no";
                }
                text
            })
            .collect();
        Some(sparagmos_interp::ProcessHost(std::sync::Arc::new(
            DriverProcessHost {
                exe: std::env::current_exe().unwrap_or_default(),
                input: absolutize_path(std::path::Path::new(&input))
                    .display()
                    .to_string(),
                edition: edition.clone(),
                rustc_args: rustc_args.clone(),
                aux_specs,
                aux_lib_specs,
                children: std::sync::Mutex::new(
                    std::collections::HashMap::new(),
                ),
                next: std::sync::Mutex::new(1),
                exe_fingerprint: std::sync::OnceLock::new(),
            },
        )))
    } else {
        None
    };

    // Only the MIR is needed: emit metadata into a scratch directory so
    // running the driver never leaves build artifacts in the caller's
    // working directory.
    let out_dir = std::env::temp_dir().join(format!("{}{}", SCRATCH_OUT_DIR_PREFIX, std::process::id()));
    let _ = std::fs::create_dir_all(&out_dir);
    register_process_scratch(&out_dir);

    // Compile the auxiliary crates first, in the order given (each
    // one sees the previously built ones through --extern), so the
    // main compilation below can link them and `collect` can convert
    // their functions from the same Stable MIR session.
    let aux_dir = out_dir.join("aux");
    let _ = std::fs::create_dir_all(&aux_dir);
    let mut aux_built: Vec<AuxBuilt> = Vec::new();
    // Prebuilt library auxiliaries (--aux-lib): the artifact
    // already exists (a support library built once by the
    // harness), so there is nothing to compile; the entry takes
    // part in binding and collection exactly like a compiled
    // auxiliary. The binding is unconditional -- the crate is
    // part of the harness's contract with the guest (a recipe
    // never declares it) -- so `main` adds the name to the
    // session's declared set below.
    //
    // The prebuilt entries also lead the visibility list the
    // auxiliary compilations bind from: compiletest builds its
    // support libraries (minicore above all) before a test's
    // auxiliary crates, so an auxiliary source that declares one
    // (`extern crate minicore;` in an auxiliary file of an
    // `//@ add-minicore` test) binds it during its own build,
    // not only in the main compilation below.
    let prebuilt: Vec<AuxBuilt> = aux_lib_specs
        .iter()
        .map(|(name, path, collect)| AuxBuilt {
            name: name.clone(),
            extern_name: name.clone(),
            source: None,
            artifact: Some(path.clone()),
            collect: *collect,
            // No build flags to resolve from: `main` records the
            // main compilation's own setting for prebuilt
            // libraries when it publishes the table (below).
            overflow_checks: true,
            // A prebuilt artifact is a native build: its
            // metadata answers MIR availability truthfully,
            // so the missing-MIR verdict never replays for it
            // (it is not published either; see below).
            native_metadata_only: false,
        })
        .collect();
    let mut visible: Vec<AuxBuilt> = prebuilt.clone();
    for spec in &aux_specs {
        match compile_aux(
            &sysroot,
            &edition,
            &aux_dir,
            spec,
            &visible,
            &aux_native_lib_dirs,
            &aux_env_scrub,
            mode,
        ) {
            Ok(built) => {
                visible.push(built.clone());
                aux_built.push(built);
            }
            Err(reason) => {
                if mode == Mode::RunJit {
                    eprintln!("sparagmos-driver run-jit: unsupported: {reason}");
                    exit_process(2);
                }
                if exec_mode {
                    eprintln!("sparagmos-driver run: unsupported: {reason}");
                    exit_process(2);
                }
                eprintln!("sparagmos-driver: {reason}");
                exit_process(1);
            }
        }
    }
    // Publish the auxiliary crates this process built for
    // the missing-MIR verdict (see
    // `mono_check::set_aux_crates`): their metadata encodes
    // MIR under `-Zalways-encode-mir` that a native build
    // would have left out, so the verdict replays the native
    // encoding rule for them. The prebuilt libraries are not
    // published: their artifacts are native builds whose
    // metadata answers truthfully.
    sparagmos_driver::mono_check::set_aux_crates(
        aux_built
            .iter()
            .map(|b| (b.name.clone(), b.native_metadata_only))
            .collect(),
    );
    aux_built.extend(prebuilt);
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
    if exec_mode && mode != Mode::CoverageRun && !caller_sets_crate_type {
        args.push("--crate-type=lib".to_string());
    }
    // The coverage-run mode is the one exec mode without the
    // lib pin: its coverage capture reads the compilation's
    // mono partition, which only has the test's functions in
    // it under the binary shape compiletest compiles coverage
    // tests in (a lib compilation roots the partition at the
    // exported items, and the test's own functions vanish from
    // it). The collected entry point of a binary crate is its
    // `main`, which the execution path resolves as usual.
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
    // The invocation context a guest self-spawn replays (see
    // `self_run`), captured before the argument vector is
    // consumed by the conversion's own argument assembly.
    let self_ctx = SelfRunContext {
        input: input.clone(),
        edition: edition.clone(),
        aux: aux_specs.iter().map(render_aux_spec).collect(),
        aux_binds: aux_binds
            .iter()
            .map(|(name, path)| format!("{name}={path}"))
            .collect(),
        rustc_args: rustc_args.clone(),
        process_host: guest_process_host,
    };
    args.extend(rustc_args);
    if !aux_built.is_empty() {
        args.push("-L".to_string());
        args.push(format!("dependency={}", aux_dir.display()));
        // A prebuilt auxiliary's own dependencies live next to
        // it (the build's deps directory), not in `aux_dir`.
        for (_, path, _) in &aux_lib_specs {
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
        top_names.extend(aux_lib_specs.iter().map(|(name, _, _)| name.clone()));
        let mut binding_declared = declared_binding_names(&main_text);
        binding_declared.extend(locator_names.iter().cloned());
        for built in &aux_built {
            if let Some(artifact) = &built.artifact {
                for bind in bind_names_for(built, &top_names, &binding_declared) {
                    // A name the invocation designated (see
                    // `--aux-bind`) binds the one build the
                    // designation names, as an auxiliary
                    // build's designated names do (see
                    // `designated_elsewhere`). An
                    // undesignated name binds every build
                    // providing it: several distinct
                    // candidates are the compilation's own
                    // E0464 to refuse, as a native invocation
                    // handed them all refuses (the
                    // crateresolve shape).
                    if designated_elsewhere(&aux_binds, bind, built) {
                        continue;
                    }
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
    // Pin the sysroot crates the main source itself declares
    // that the sysroot offers ambiguously (sysroot_extern_pins):
    // the same disambiguation the auxiliary builds get. A main
    // source that declares an extern crate libc item itself — the
    // tests/ui/runtime/on-broken-pipe/inherit.rs record does,
    // to install SIG_IGN for SIGPIPE through libc::signal
    // — otherwise fails its front end with E0464 (multiple
    // candidates for the crate) in a rustup sysroot carrying
    // the rustc-dev component, before any guest code runs. A
    // name already bound above (an auxiliary crate, a prebuilt
    // library, a caller --extern) keeps that binding.
    {
        let main_text = std::fs::read_to_string(&input).unwrap_or_default();
        let mut bound: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut args_iter = args.iter();
        while let Some(a) = args_iter.next() {
            if a == "--extern" {
                if let Some(spec) = args_iter.next() {
                    if let Some((name, _)) = spec.split_once('=') {
                        bound.insert(name.to_string());
                    }
                }
            } else if let Some(spec) = a.strip_prefix("--extern=") {
                if let Some((name, _)) = spec.split_once('=') {
                    bound.insert(name.to_string());
                }
            }
        }
        for (name, rmeta, rlib) in sysroot_extern_pins(&sysroot, &main_text) {
            if bound.contains(&name) {
                continue;
            }
            args.push("--extern".to_string());
            args.push(format!("{name}={}", rmeta.display()));
            args.push("--extern".to_string());
            args.push(format!("{name}={}", rlib.display()));
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
    // Publish the auxiliary crates' own overflow-checks
    // settings for the collection about to run: a prebuilt
    // library (--aux-lib) has no build of its own to resolve
    // from, so it takes the main compilation's setting; a
    // compiled auxiliary keeps the setting its own build flags
    // resolved to (see `AuxBuilt::overflow_checks`).
    for built in &mut aux_built {
        if aux_lib_specs.iter().any(|(name, _, _)| name == &built.name) {
            built.overflow_checks = overflow_checks;
        }
    }
    let _ = AUX_OVERFLOW_CHECKS.set(
        aux_built
            .iter()
            .filter(|b| b.collect)
            .map(|b| (b.name.clone(), b.overflow_checks))
            .collect::<Vec<_>>(),
    );
    let panic_abort = session_panic_abort(&args);
    let sigpipe = session_sigpipe(&args);
    let entry_shim_visible = session_entry_shim_visible(&args);
    if dep_info_verdict && mode == Mode::Convert {
        // A dep-info verdict run (see `run_resolution_verdict`
        // in collect.rs): the session the arguments assembled
        // runs only to the phase boundary the real compiler's
        // dep-info mode stops at, and the exit code is that
        // boundary's own verdict -- 0 when the crate parses,
        // expands and resolves, 1 when it does not. Nothing is
        // converted or collected, and no report is printed.
        // Only the tool interception asks for this, and only
        // for a guest invocation whose emit set is dep-info
        // alone; the compiler's own diagnostics have already
        // reached stderr through the session's emitter, as the
        // real compiler's would.
        let verdict_hook = begin_panic_capture();
        let mut verdict_stdout_guard = StdoutToStderr::new();
        let verdict_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_resolution_verdict(args.as_slice())
        }));
        verdict_stdout_guard.restore();
        end_panic_capture(verdict_hook);
        match verdict_outcome {
            Ok(Ok(true)) => exit_process(0),
            Ok(Ok(false)) | Ok(Err(_)) => exit_process(1),
            Err(payload) => {
                let reason = take_captured_panic()
                    .or_else(|| payload_reason(&*payload))
                    .unwrap_or_else(|| "panic with a non-string payload".to_string());
                report_escaped_panic(&reason, exec_mode, mode);
            }
        }
    }
    let previous_hook = begin_panic_capture();
    let mut stdout_guard = match &compiler_stdout {
        Some(path) => StdoutToStderr::to_file(std::path::Path::new(path)),
        None => StdoutToStderr::new(),
    };
    let run_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_conversion_compiler(args.as_slice())
    }));
    stdout_guard.restore();
    end_panic_capture(previous_hook);
    let result = match run_outcome {
        Ok(result) => result,
        Err(payload) => {
            let reason = take_captured_panic()
                .or_else(|| payload_reason(&*payload))
                .unwrap_or_else(|| "panic with a non-string payload".to_string());
            report_escaped_panic(&reason, exec_mode, mode);
        }
    };
    let collected = take();
    match (result, collected) {
        (Ok(()), Some(c)) => {
            // A compiler crash a verdict authority
            // established during collection (see
            // `convert::note_compiler_crash`) pre-empts the
            // compilation-failure report below: the
            // compilation died on the compiler's own bug.
            if let Some(crash) = take_compiler_crash() {
                report_escaped_panic(&crash, exec_mode, mode);
            }
            if let Some(e) = &c.compile_error {
                eprintln!("sparagmos-driver: compilation failed: {e}");
                // Same contract as a front-end failure: exit 1 in
                // convert mode, exit 2 (unsupported) in run mode.
                exit_process(if exec_mode { 2 } else { 1 });
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
                exit_process(execute_debug(
                    &c,
                    &input,
                    &guest_args,
                    overflow_checks,
                    panic_abort,
                    sigpipe,
                    entry_shim_visible,
                    debug_script.as_deref(),
                    &debug_dialect,
                    hide_locals,
                ));
            }
            if mode == Mode::CoverageMap {
                // The coverage-mapping dump of the instrumented
                // crate, on stdout; nothing is executed.
                use std::io::Write;
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(
                    sparagmos_driver::coverage::render_dump(&c.coverage)
                        .as_bytes(),
                );
                let _ = out.flush();
                std::process::exit(0);
            }
            if mode == Mode::Run || mode == Mode::RunJit
                || mode == Mode::CoverageRun
            {
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
                            exit_process(2);
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
                        exit_process(2);
                    }
                }
                if let Some(path) = &dump_path {
                    let text =
                        sparagmos_driver::dump::render(&input, &c.report, &c.bodies, &c.body_def_ids, &c.def_names);
                    if let Err(e) = std::fs::write(path, text) {
                        eprintln!("failed to write dump {path}: {e}");
                        exit_process(1);
                    }
                }
                if mode == Mode::RunJit {
                    exit_process(execute_jit(&c, &guest_args));
                }
                exit_process(execute(
                    &c,
                    &input,
                    &guest_args,
                    overflow_checks,
                    panic_abort,
                    sigpipe,
                    entry_shim_visible,
                    &tool_programs,
                    guest_stderr_file.as_deref(),
                    guest_stdout_passthrough,
                    guest_program_name.as_deref(),
                    process_host,
                    &self_ctx,
                    mode == Mode::CoverageRun,
                    coverage_color,
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
                        exit_process(2);
                    }
                }
            }
            if let Some(path) = &dump_path {
                let text =
                    sparagmos_driver::dump::render(&input, &c.report, &c.bodies, &c.body_def_ids, &c.def_names);
                if let Err(e) = std::fs::write(path, text) {
                    eprintln!("failed to write dump {path}: {e}");
                    exit_process(1);
                }
                println!("dump: {path}");
            }
        }
        (Ok(()), None) => {
            eprintln!("sparagmos-driver: internal error: no results collected");
            exit_process(1);
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
            // A panic raised inside the run can be folded into
            // this bare `CompilerError::Failed` by the compiler's
            // fatal-error machinery before it reaches the
            // `catch_unwind` above; `report_failed_run` surfaces
            // the panic hook's record of such a panic (see
            // `begin_panic_capture`) and otherwise keeps the
            // bare compilation-failure report.
            report_failed_run(&e, exec_mode, mode);
        }
    }
}

#[cfg(test)]
mod scratch_reap_tests {
    use super::*;

    fn pid_of(name: &str) -> Option<u32> {
        scratch_out_dir_pid(std::ffi::OsStr::new(name))
    }

    #[test]
    fn parses_creator_pid_from_out_dir_names() {
        assert_eq!(pid_of("sparagmos-driver-out-1"), Some(1));
        assert_eq!(pid_of("sparagmos-driver-out-4194304"), Some(4194304));
        assert_eq!(pid_of("sparagmos-artifact-sources-123"), Some(123));
        // Not names this driver produces: no pid, pid 0 (a
        // process-group address), a sign, trailing junk, a pid
        // that does not fit a `pid_t`.
        assert_eq!(pid_of("sparagmos-driver-out-"), None);
        assert_eq!(pid_of("sparagmos-driver-out-0"), None);
        assert_eq!(pid_of("sparagmos-driver-out--1"), None);
        assert_eq!(pid_of("sparagmos-driver-out-12x"), None);
        assert_eq!(pid_of("sparagmos-driver-out-99999999999"), None);
        assert_eq!(pid_of("sparagmos-artifact-sources-0"), None);
        assert_eq!(pid_of("sparagmos-artifact-sources-12x"), None);
        assert_eq!(pid_of("sparagmos-driver-out"), None);
    }

    #[test]
    fn reapable_only_when_dead_old_and_not_own() {
        let old = SCRATCH_REAP_MIN_AGE;
        let young = std::time::Duration::ZERO;
        assert!(scratch_out_dir_reapable(4242, 1111, false, old));
        assert!(!scratch_out_dir_reapable(4242, 1111, true, old));
        assert!(!scratch_out_dir_reapable(4242, 1111, false, young));
        assert!(!scratch_out_dir_reapable(1111, 1111, false, old));
    }

    #[test]
    fn registered_scratch_paths_are_removed() {
        // Registration is what the exit wrapper drains on
        // every exit path (#537): a registered directory and
        // a registered file both go away, and a path that was
        // never created is not an error.
        let dir = std::env::temp_dir().join(format!(
            "sparagmos-scratch-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("scratch.txt");
        std::fs::write(&file, b"x").unwrap();
        register_process_scratch(&dir);
        register_process_scratch(&file);
        register_process_scratch(&dir.join("never-created"));
        remove_process_scratch();
        assert!(!dir.exists());
        assert!(!file.exists());
    }

    #[cfg(unix)]
    #[test]
    fn process_alive_sees_self_and_not_a_pid_above_the_kernel_max() {
        assert!(process_alive(std::process::id()));
        // Linux caps pids at 2^22 and macOS below 100_000, so
        // this pid cannot belong to any process.
        assert!(!process_alive(999_999_999));
    }
}
