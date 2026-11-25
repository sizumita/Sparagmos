//! Entry points: whole-program conversion and the one-shot runner on top of it.


use sparagmos_mir as mir;

use crate::error::InterpError;
use crate::interp::Interpreter;
use crate::ir;
use crate::value::{Value};

use super::converter::*;
use super::drop_glue::*;
use super::errors::*;
use super::program::*;
use super::resolve::*;


/// Resource bound on `Rvalue::Repeat` element counts, mirroring the
/// interpreter's step / call-depth limits: past this many elements the
/// aggregate is not materialised and the rvalue converts to an
/// explicit unsupported error instead of aborting on allocation.
pub(super) const MAX_REPEAT_ELEMENTS: u64 = 1_000_000;

/// Reduce a raw `SwitchInt` target value to the canonical form the
/// interpreter compares against. MIR stores target values as the
/// discriminant type's bit pattern, zero-extended to 128 bits
/// (a `-2i32` branch key is stored as 4294967294), while the
/// interpreter holds signed values sign-extended (`switch_bits`),
/// so a signed type's keys are sign-extended here from the type's
/// width. Unsigned and non-integer discriminants already agree.
/// The JIT applies the same normalisation in its lowering.
pub(super) fn normalize_switch_value(v: u128, ty: Option<&mir::Ty>) -> u128 {
    match ty {
        Some(mir::Ty::Int(it)) => {
            let bits = it.bits();
            if bits >= 128 {
                v
            } else {
                let shift = 128 - bits;
                (((v << shift) as i128) >> shift) as u128
            }
        }
        _ => v,
    }
}

/// Convert a whole MIR program into an interpreter [`ir::Program`].
///
/// Conversion is isolated per body: a function whose body cannot be
/// converted is recorded in the program's unconverted table with
/// its failure reason instead of failing the whole conversion (see
/// the module docs). Only a structurally malformed program — two
/// functions that neither name nor `DefId` can tell apart — fails
/// conversion as a whole.
pub fn convert_program(program: &MirProgram) -> Result<ir::Program, ConvertError> {
    Ok(convert_all(program)?.0)
}

/// Convert `program` and run `entry` with `args` in the interpreter.
///
/// Bodies are converted in isolation (see [`convert_program`]):
/// an unconvertible helper that is never called does not affect
/// the run, and calling one fails with its conversion reason. If
/// the entry function itself cannot be converted, the run fails
/// with that conversion error before anything executes.
pub fn interpret_mir(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
) -> Result<Value, MirRunError> {
    let (ir_program, failures) = convert_all_lazy(program)?;
    if let Some(e) = failures.get(entry) {
        return Err(MirRunError::Convert(e.clone()));
    }
    let mut interp = Interpreter::new(&ir_program);
    let result = interp.run(entry, args);
    interp.finish_stdout();
    Ok(result?)
}

/// The outcome of a run that also captures the guest's stdout: the
/// run result plus everything the guest printed before it returned
/// or failed (a native program's output is likewise observable when
/// it panics).
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedRun {
    pub result: Result<Value, MirRunError>,
    pub stdout: String,
    pub stderr: String,
}

/// Like [`interpret_mir`], but also returns the captured guest stdout.
pub fn interpret_mir_captured(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
) -> CapturedRun {
    interpret_mir_captured_with_config(
        program,
        entry,
        args,
        crate::InterpConfig::default(),
    )
}

/// Like [`interpret_mir_captured`], with an explicit interpreter
/// configuration (the driver hands the guest program's
/// command-line arguments through it; see
/// [`crate::InterpConfig::guest_args`]).
pub fn interpret_mir_captured_with_config(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
    config: crate::InterpConfig,
) -> CapturedRun {
    interpret_mir_captured_impl(program, entry, args, config, |_| None)
}

/// Like [`interpret_mir_captured_with_config`], but when the
/// entry returns normally, `report` is consulted with the
/// returned value while the interpreter that executed the entry
/// is still alive: it names the value a `Termination` report
/// must render (the `Err` payload of a `Result` entry), the
/// type tag to render it under, and the payload's type. The
/// payload's `Debug` text is rendered by that interpreter --
/// whose allocation store and statics the payload may point
/// into -- and the report's `Error: ...` line joins the
/// captured stderr at the report's native position: after the
/// entry returns, before the entry thread's TLS destructors
/// run. The `Error: ` prefix is written before the rendering
/// starts, as natively the report writes it before formatting
/// the payload, so output the payload's `Debug` body itself
/// writes to stderr follows the prefix, and the prefix stands
/// even when the formatting does not return normally. The
/// rendering itself streams when it goes through the payload's
/// own `Debug` body: the body's formatter writes land in the
/// captured stderr as they happen, interleaved with the body's
/// own stderr writes, as natively. After a
/// rendering that returns normally (or falls back to the
/// payload's structural form), the payload's drop glue runs
/// against the still-live interpreter, as the native report
/// consumes and drops the error after printing it: a guest
/// outcome the destructor itself produces (a panic, a
/// `std::process::exit`) becomes the run's result. A guest
/// outcome the rendering itself produces likewise becomes
/// the run's result, as it does natively: a `process::exit`
/// and an abort leave the payload undropped (the run is
/// already on its way out, before any cleanup), while a panic
/// unwinds through the report and drops the payload as its
/// cleanup, the drop's own guest outcome replacing the panic
/// (a destructor that exits the process exits it; a second
/// panic aborts). A payload that does
/// not render even against the live state (it reaches into
/// the run's frames) stands in with its structural form. The
/// caller still applies the report's exit code.
pub fn interpret_mir_captured_with_report(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
    config: crate::InterpConfig,
    report: impl FnOnce(&Value) -> Option<(Value, String, mir::Ty)>,
) -> CapturedRun {
    interpret_mir_captured_impl(program, entry, args, config, report)
}

fn interpret_mir_captured_impl(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
    config: crate::InterpConfig,
    report: impl FnOnce(&Value) -> Option<(Value, String, mir::Ty)>,
) -> CapturedRun {
    match convert_all_lazy(program) {
        Ok((ir_program, failures)) => {
            // The entry function's own conversion failure is a
            // whole-run failure with no guest output, as before.
            if let Some(e) = failures.get(entry) {
                return CapturedRun {
                    result: Err(MirRunError::from(e.clone())),
                    stdout: String::new(),
                    stderr: String::new(),
                };
            }
            let mut interp = Interpreter::with_config(&ir_program, config);
            // The termination report renders against the live
            // interpreter, before its allocations and statics go
            // away with it and before the entry thread's TLS
            // destructors run (see
            // [`Interpreter::run_with_report`]).
            let result = interp
                .run_with_report(entry, args, |interp, frames, value| {
                    let Some((payload, tag, err_ty)) = report(value) else {
                        return Ok(None);
                    };
                    // Drop the payload against the live
                    // interpreter, as the native report consumes
                    // the error it reports: build the drop glue
                    // from the payload's type and run it. A
                    // payload type whose glue conversion cannot
                    // account for is an explicit error, never a
                    // silent skip -- the same policy as a `Drop`
                    // terminator whose glue cannot be built (it
                    // stays unsupported) and a `dyn` value whose
                    // carried glue is `None` (dropping it fails
                    // explicitly; see `UnsizeMeta::Dyn` in `ir`).
                    // The drop's own guest outcome (a panicking
                    // destructor, a `process::exit` in one)
                    // propagates as the caller's.
                    let drop_payload = |interp: &mut Interpreter,
                                        frames: &mut Vec<crate::interp::Frame>,
                                        payload: Value|
                     -> Result<(), InterpError> {
                        let keys = program_keys(program).map_err(|e| {
                            InterpError::UnsupportedTerminator {
                                kind: format!(
                                    "Drop (Termination report payload: {e})"
                                ),
                            }
                        })?;
                        let glue = {
                            let conv = Converter::new(program, &keys);
                            DropGlueBuilder::build(&conv, &err_ty)
                        };
                        let Some(glue) = glue else {
                            return Err(InterpError::UnsupportedTerminator {
                                kind: "Drop (Termination report payload glue                                  cannot be accounted for)".to_string(),
                            });
                        };
                        interp.drop_value_sync(frames, payload, &glue)
                    };
                    // The native report writes the `Error: `
                    // prefix before it formats the payload, so
                    // the prefix precedes anything the payload's
                    // `Debug` body itself writes to stderr, and
                    // it is already in the captured stderr when
                    // the formatting does not return normally (a
                    // panic, a `process::exit`): push it now,
                    // before the rendering starts, rather than
                    // as part of the text returned afterwards.
                    interp.push_error_output("Error: ");
                    let text = match interp.format_debug_detached(&payload, &tag) {
                        Ok(text) => text,
                        // A guest outcome of the rendering itself
                        // (a panic, an abort, a `process::exit`
                        // in the payload's `Debug` body) is the
                        // run's outcome, not a rendering failure.
                        // A panic unwinds natively through the
                        // report, dropping the error as it goes:
                        // run the payload's drop as the panic's
                        // cleanup, and let the drop's own guest
                        // outcome replace the panic (a destructor
                        // that exits the process exits it; a
                        // destructor that panics while unwinding
                        // aborts, as a panic in a cleanup does).
                        // An abort and a `process::exit` keep the
                        // no-drop behavior: the run is already on
                        // its way out, before any cleanup. The
                        // prefix pushed above stays in the captured
                        // stderr, as does any text the rendering
                        // streamed before it failed.
                        Err(e) if is_guest_outcome(&e) => {
                            if matches!(e, InterpError::Panic { .. }) {
                                match drop_payload(interp, frames, payload) {
                                    Ok(()) => return Err(e),
                                    Err(InterpError::Panic {
                                        message, ..
                                    }) => {
                                        return Err(InterpError::Abort {
                                            message: format!(
                                                "panic `{message}` raised in the drop of the Termination report payload while unwinding: panicking while unwinding aborts the run"
                                            ),
                                        });
                                    }
                                    Err(e2) => return Err(e2),
                                }
                            }
                            return Err(e);
                        }
                        // The payload does not render even
                        // against the live state (it reaches into
                        // the run's frames): its structural form
                        // stands in, at the report's own position
                        // in the captured stderr.
                        Err(_) => Some(format!("{payload}")),
                    };
                    // The rendered line is complete before the
                    // payload is dropped, exactly as natively:
                    // write its ending now, so output the
                    // payload's destructor writes follows the
                    // whole line, not the prefix alone. A
                    // rendering through the payload's own `Debug`
                    // body already streamed its text into the
                    // captured stderr as it ran (`None`); only
                    // the newline remains to write.
                    match &text {
                        Some(text) => {
                            interp.push_error_output(&format!("{text}\n"))
                        }
                        None => interp.push_error_output("\n"),
                    }
                    // The native report consumes the error and
                    // drops it after printing, before TLS
                    // teardown: run the payload's drop glue
                    // against the live interpreter and propagate
                    // its outcome (a panicking destructor, a
                    // `process::exit` in one) as the run's.
                    drop_payload(interp, frames, payload)?;
                    Ok(None)
                })
                .map_err(MirRunError::from);
            interp.finish_stdout();
            CapturedRun {
                result,
                stdout: interp.output().to_string(),
                stderr: interp.error_output().to_string(),
            }
        }
        Err(e) => CapturedRun {
            result: Err(MirRunError::from(e)),
            stdout: String::new(),
            stderr: String::new(),
        },
    }
}

/// Whether an interpreter error is a *guest* outcome -- the
/// guest's own control flow reaching the top of a run (a panic,
/// a panic that may not unwind, a `std::process::exit`) --
/// rather than an interpreter/oracle failure. A guest outcome
/// is observable program behavior wherever it occurs, including
/// inside a `Termination` report's rendering; an oracle failure
/// is not, and never replaces a run's result.
fn is_guest_outcome(e: &InterpError) -> bool {
    matches!(
        e,
        InterpError::Panic { .. } | InterpError::Abort { .. } | InterpError::ProcessExit { .. }
    )
}

/// Convert `program` and open a debugger session on `entry`
/// (see [`crate::debug::DebugSession`]): the caller sets its
/// breakpoints and source text, then runs a command script; the
/// session drives the interpreter itself, so nothing executes
/// until the script runs. Conversion failures are reported the
/// same way as for a run: a helper's failure is isolated, the
/// entry's own failure is returned here.
pub fn debug_mir(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
    config: crate::InterpConfig,
) -> Result<crate::debug::DebugSession, MirRunError> {
    let (ir_program, failures) = convert_all(program).map_err(MirRunError::from)?;
    if let Some(e) = failures.get(entry) {
        return Err(MirRunError::from(e.clone()));
    }
    Ok(crate::debug::DebugSession::new(
        ir_program,
        entry.to_string(),
        args,
        config,
    ))
}
