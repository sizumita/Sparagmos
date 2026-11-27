//! Execution of a built program in the interpreter, including debug mode.

use super::*;

/// An absolute form of `path` against this process's working
/// directory (a self-child's invocation is assembled from
/// paths that must survive the child's own working
/// directory).
pub(crate) fn absolutize_path(path: &std::path::Path) -> std::path::PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

pub(crate) fn execute(
    c: &Collected,
    input: &str,
    guest_args: &[String],
    overflow_checks: bool,
    panic_abort: bool,
    sigpipe: sparagmos_interp::SigpipeInit,
    tool_programs: &[String],
    guest_stderr_file: Option<&str>,
    guest_stdout_passthrough: bool,
    program_name: Option<&str>,
    process_host: Option<sparagmos_interp::ProcessHost>,
    self_ctx: &SelfRunContext,
    coverage_report: bool,
    coverage_color: bool,
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
        // No tool programs were named: the one program the
        // host answers for is the guest's own. A guest
        // spawning itself (the `needs-subprocess` test shape:
        // a parent run checking a child run of the same
        // program) is routed to `self_run`, which replays
        // this run's invocation in a driver subprocess; every
        // other program keeps the process model's standing
        // answers, and `spawn` of even the guest's own
        // program stays unsupported (no `Child` handle can be
        // fabricated for a subprocess run).
        let exe = std::env::current_exe().unwrap_or_default();
        let self_ctx = self_ctx.clone();
        Some(sparagmos_interp::ToolInterception {
            programs: vec![program_name.as_bytes().to_vec()],
            runner: sparagmos_interp::ToolRunner(std::sync::Arc::new(
                move |request| self_run(request, &self_ctx, &exe),
            )),
            fallback: None,
        })
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
        panic_abort,
        sigpipe,
        // Real child processes coexist with tool
        // interception in any run that configured a host:
        // the interception is consulted first (the named
        // tool programs through the runner, the artifacts
        // this run produced through the fallback), and a
        // program neither claims is the host's to serve.
        // A run naming tool programs and configuring a
        // host together is how a recipe drives a real
        // orchestrator (cargo) whose own compiler
        // invocations the run still answers for -- the
        // stage-3 cargo shim (issue #359). A run with no
        // host configured keeps the interception's
        // standing answers for everything it claims.
        process_host: process_host.clone(),
        tool_interception,
        // A run the process model re-ran as a child (see
        // `self_run`) writes the guest's stdout through to
        // this process's own stdout as it happens, errors
        // included: that stream is the pipe (or terminal) the
        // parent arranged, and a parent that has closed it
        // must be observable to the guest, as natively.
        stdout_sink: if guest_stdout_passthrough {
            use std::os::fd::AsFd as _;
            std::io::stdout()
                .as_fd()
                .try_clone_to_owned()
                .ok()
                .map(|fd| {
                    sparagmos_interp::StdoutSink(std::sync::Arc::new(
                        std::fs::File::from(fd),
                    ))
                })
        } else {
            None
        },
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
        if !coverage_report && !guest_stdout_passthrough {
            let _ = out.write_all(run.stdout.as_bytes());
            let _ = out.flush();
        }
        let mut relay_to_stderr = true;
        if let Some(path) = guest_stderr_file {
            relay_to_stderr = std::fs::write(path, run.stderr.as_bytes()).is_err();
            if !relay_to_stderr {
                // The caller allocated the channel
                // owner-only (see `artifact_run`), and
                // writing into the standing file keeps
                // that; restate it so the channel is
                // owner-only even when this write itself
                // had to create the file.
                restrict_to_owner(std::path::Path::new(path), 0o600);
            }
        }
        if relay_to_stderr {
            let mut err = std::io::stderr().lock();
            let _ = err.write_all(run.stderr.as_bytes());
            let _ = err.flush();
        }
    }

    let exit_code = match run.result {
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
            ..
        })) => {
            // The guest's panic report (std's default hook
            // output) is already in the captured stderr: the
            // interpreter prints it when the panic is raised.
            // Only the exit code remains to apply.
            101
        }
        Err(sparagmos_interp::MirRunError::Interp(sparagmos_interp::InterpError::Abort {
            ..
        })) => {
            // The guest crashed (a native double panic aborts
            // the process): whatever the abort printed — the
            // panic report and the runtime's abort lines — is
            // already in the captured stderr, printed by the
            // interpreter at the abort point. A marker on the
            // driver's own stderr names the outcome (a parent
            // reconstructing this run reads the crash from it),
            // and the exit code is the crash code (128 + SIGABRT)
            // the native process's death carries, so a crash
            // stays distinct from a guest panic that unwound to
            // the top of the run (101).
            eprintln!("sparagmos-driver run: guest crashed: abort");
            134
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
    };
    if coverage_report {
        // The llvm-cov style report rendered from the run's
        // counted counters: the coverage-run mode's stdout, in
        // place of the guest's own (discarded above).
        use std::io::Write;
        let report = sparagmos_driver::coverage::render_report(
            &program.coverage,
            &run.coverage_counts,
            coverage_color,
        );
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(report.as_bytes());
        let _ = out.flush();
    }
    exit_code
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
pub(crate) fn execute_debug(
    c: &Collected,
    input: &str,
    guest_args: &[String],
    overflow_checks: bool,
    panic_abort: bool,
    sigpipe: sparagmos_interp::SigpipeInit,
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
        panic_abort,
        sigpipe,
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
