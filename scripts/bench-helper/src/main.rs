//! Stage-timing helper for the performance baseline (scripts/bench.py).
//!
//! Loads one driver dump (`sparagmos-driver <file.rs> --dump <path>`,
//! the same text representation the harness runners use) and times
//! the stages that follow driver-side conversion, in-process:
//!
//! * `parse`   — dump text -> program (harness `parse_dump`)
//! * `convert` — sparagmos-mir program -> interpreter IR
//!               (`sparagmos_interp::convert_program`)
//! * `interp`  — interpreter execution of the converted program
//!               (`Interpreter::run`)
//! * `jitc`    — JIT compilation of the whole program
//!               (`JitProgram::compile`)
//! * `jit`     — execution of the compiled program's `main`
//!               (`JitProgram::call`)
//!
//! Modes:
//! * `check`          — run each path once and print its status
//!                      (default interpreter limits, exactly what the
//!                      driver's run mode does), so the caller can
//!                      verify every path performs the computation
//!                      and record concrete unsupported reasons.
//! * `interp <reps>`  — print per-rep samples for convert and interp.
//!                      The interpreter runs with a raised step/call
//!                      budget: the default budget (1,000,000 steps)
//!                      stops long-running programs, and this mode
//!                      exists to time pure execution of programs
//!                      beyond that budget. `check` mode reports
//!                      what the default budget does to them.
//! * `jit <reps>`     — print per-rep samples for jitc and jit.
//!
//! Output is `key=value` lines on stdout, times in milliseconds.

use std::time::Instant;

use sparagmos_harness::parse_dump;
use sparagmos_interp::{convert_program, InterpConfig, Interpreter, MirFunction, MirProgram};
use sparagmos_jit::{JitCallError, JitError, JitProgram, MirProgramInput};

fn is_main(name: &str) -> bool {
    name.rsplit("::").next() == Some("main")
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn load(path: &str) -> Result<(MirProgramInput, String), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read dump: {e}"))?;
    let t = Instant::now();
    let parsed = parse_dump(&text).map_err(|e| format!("cannot parse dump: {e}"))?;
    let entry = parsed
        .program
        .functions
        .iter()
        .map(|f| f.name.clone())
        .find(|n| is_main(n))
        .ok_or_else(|| "no `main` function among the converted bodies".to_string())?;
    println!("parse_ms={:.3}", ms(t.elapsed()));
    Ok((parsed.program, entry))
}

/// The interpreter-side program shape, built field by field from the
/// JIT-side input shape (both wrap the same sparagmos-mir bodies and
/// the same definition-name table).
fn interp_program(input: &MirProgramInput) -> MirProgram {
    MirProgram {
        functions: input
            .functions
            .iter()
            .map(|f| MirFunction {
                name: f.name.clone(),
                def_id: f.def_id,
                body: f.body.clone(),
            })
            .collect(),
        fn_names: input.names.clone(),
        ..Default::default()
    }
}

fn mode_check(input: &MirProgramInput, entry: &str) {
    // Interpreter, through the same entry point the driver's run
    // mode uses (default limits).
    let program = interp_program(input);
    let run = sparagmos_interp::interpret_mir_captured(&program, entry, Vec::new());
    match &run.result {
        Ok(_) => println!("interp_status=ok"),
        Err(e) => println!("interp_status=error: {e}"),
    }
    // JIT.
    match JitProgram::compile(input) {
        Ok(prog) => match prog.call(entry, &[]) {
            Ok(_) => println!("jit_status=ok"),
            Err(JitCallError::Panic(rec)) => {
                println!("jit_status=guest_panic: {}", rec.message)
            }
            Err(other) => println!("jit_status=error: {other}"),
        },
        Err(JitError::Unsupported(what)) => println!("jit_status=unsupported: {what}"),
        Err(JitError::Backend(e)) => println!("jit_status=backend_error: {e}"),
    }
}

fn mode_interp(input: &MirProgramInput, entry: &str, reps: usize) -> Result<(), String> {
    let program = interp_program(input);
    // Conversion samples.
    let mut ir = None;
    for i in 0..reps {
        let t = Instant::now();
        let converted = convert_program(&program)
            .map_err(|e| format!("MIR conversion failed: {e}"))?;
        println!("convert_ms_{i}={:.3}", ms(t.elapsed()));
        ir = Some(converted);
    }
    let ir = ir.expect("reps >= 1");
    // Execution samples, raised budget (see module docs).
    let config = InterpConfig {
        max_steps: 10_000_000_000,
        max_call_depth: 1_000_000,
        guest_args: Vec::new(),
        program_path: String::new(),
        overflow_checks: true,
    };
    for i in 0..reps {
        let mut interp = Interpreter::with_config(&ir, config.clone());
        let t = Instant::now();
        let result = interp.run(entry, Vec::new());
        let elapsed = t.elapsed();
        match result {
            Ok(_) => println!("interp_ms_{i}={:.3}", ms(elapsed)),
            Err(e) => return Err(format!("interpreter run failed: {e}")),
        }
    }
    Ok(())
}

fn mode_jit(input: &MirProgramInput, entry: &str, reps: usize) -> Result<(), String> {
    let mut prog = None;
    for i in 0..reps {
        let t = Instant::now();
        let compiled = JitProgram::compile(input).map_err(|e| match e {
            JitError::Unsupported(what) => format!("unsupported: {what}"),
            JitError::Backend(e) => format!("jit backend error: {e}"),
        })?;
        println!("jitc_ms_{i}={:.3}", ms(t.elapsed()));
        prog = Some(compiled);
    }
    let prog = prog.expect("reps >= 1");
    for i in 0..reps {
        let t = Instant::now();
        let result = prog.call(entry, &[]);
        let elapsed = t.elapsed();
        match result {
            Ok(_) => println!("jit_ms_{i}={:.3}", ms(elapsed)),
            Err(JitCallError::Panic(rec)) => {
                return Err(format!("guest panicked: {}", rec.message))
            }
            Err(other) => return Err(format!("jit call failed: {other}")),
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (dump, mode, reps) = match args.as_slice() {
        [_, dump, mode] => (dump.clone(), mode.clone(), 3usize),
        [_, dump, mode, reps] => (
            dump.clone(),
            mode.clone(),
            reps.parse().unwrap_or_else(|_| {
                eprintln!("bad reps: {reps}");
                std::process::exit(2);
            }),
        ),
        _ => {
            eprintln!("usage: sparagmos-bench-helper <dump> <check|interp|jit> [reps]");
            std::process::exit(2);
        }
    };
    if reps == 0 {
        eprintln!("reps must be >= 1");
        std::process::exit(2);
    }
    let (input, entry) = match load(&dump) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let result = match mode.as_str() {
        "check" => {
            mode_check(&input, &entry);
            Ok(())
        }
        "interp" => mode_interp(&input, &entry, reps),
        "jit" => mode_jit(&input, &entry, reps),
        other => Err(format!("unknown mode: {other}")),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
