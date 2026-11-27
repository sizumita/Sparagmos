//! JIT execution of a collected program.

use super::*;

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
pub(crate) fn jit_compile_program(c: &Collected) -> Result<sparagmos_jit::JitProgram, String> {
    sparagmos_jit::JitProgram::compile(&jit_program_input(c)).map_err(|e| match e {
        sparagmos_jit::JitError::Unsupported(what) => what,
        sparagmos_jit::JitError::Backend(e) => format!("jit backend error: {e}"),
    })
}

/// Execute the collected program's `main` with the JIT and return
/// the run-jit process exit code (see the crate docs). The entry
/// contract mirrors the harness's `JitRunner`: the entry is called
/// with no arguments, a normal return is exit 0, and a guest panic
/// is exit 101 with std's default panic report on stderr. Shapes the JIT
/// entry cannot model — guest arguments, a C-convention
/// `main(argc, argv)`, an entry returning `ExitCode` — are
/// reported as unsupported (exit 2), never guessed.
pub(crate) fn execute_jit(c: &Collected, guest_args: &[Vec<u8>]) -> i32 {
    if !guest_args.is_empty() {
        eprintln!(
            "sparagmos-driver run-jit: unsupported: guest arguments are not supported by JIT execution"
        );
        return 2;
    }
    let entry_index = c
        .entry_def_id
        .and_then(|def| {
            // The designated entry is authoritative wherever
            // its body sits: an imported `main` from a foreign
            // crate is a demanded body, past the local prefix.
            c.body_def_ids.iter().position(|id| *id == def)
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
            // std's default panic report for the guest panic
            // (#540): the JIT path's invented
            // `thread 'main' panicked:` line is gone, as on the
            // interpreter path; the report goes through the
            // same stderr lock, after the guest's own stderr.
            let report = sparagmos_jit::std_default_panic_report(
                &rec.message,
                rec.location.as_deref(),
            );
            let _ = err.write_all(report.as_bytes());
            let _ = err.flush();
            101
        }
        Err(other) => {
            eprintln!("sparagmos-driver run-jit: unsupported: {other}");
            2
        }
    }
}
