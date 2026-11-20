//! The harness's Sparagmos runner on the project's main path.
//!
//! Main path: `.rs` --(Stable MIR driver, Lane I)--> `sparagmos-mir`
//! bodies --(sparagmos-interp)--> execution result. This module is the
//! execution half of that path: [`run_mir_program`] executes a
//! [`MirProgram`] in the interpreter and shapes the outcome as a
//! [`RunResult`].
//!
//! [`InterpRunner`] is the [`SparagmosRunner`] used for corpus cases.
//! Until the driver is integrated it cannot obtain MIR for a `.rs`
//! file, so it honestly reports every case as unsupported ("awaiting
//! sparagmos-driver") instead of guessing output. Once Lane I lands,
//! `run` will feed the driver's MIR into [`run_mir_program`]; nothing
//! else in the harness changes. There is deliberately no other input
//! path (project rule: main path only).

use sparagmos_interp::mir_convert::{convert_program, MirProgram};
use sparagmos_interp::{InterpError, Interpreter, Value};

use crate::{CorpusCase, RunResult, SparagmosRunner, UnsupportedReason};

/// Corpus runner executing driver-produced MIR in the interpreter.
/// See the module docs for its current (driver-waiting) behaviour.
#[derive(Debug, Default, Clone, Copy)]
pub struct InterpRunner;

impl SparagmosRunner for InterpRunner {
    fn run(&self, case: &CorpusCase) -> Result<RunResult, UnsupportedReason> {
        Err(UnsupportedReason::Feature(format!(
            "awaiting Stable MIR driver (sparagmos-driver, Lane I): no MIR for `{}` yet",
            case.name
        )))
    }
}

/// Execute `entry` (a zero-argument function, e.g. `main`) of a MIR
/// program obtained via the driver, in the interpreter, and shape the
/// outcome as a [`RunResult`]:
///
/// * normal return of `()` → exit code 0, empty stdout (output produced
///   by guest `println!` calls is *not* modelled yet — such calls are
///   unresolved callees and surface as `Unsupported`, never as faked
///   output);
/// * a guest panic → exit code 101 (the real Rust panic exit code) with
///   the panic message on stderr;
/// * anything the interpreter/converter cannot execute yet →
///   [`UnsupportedReason::Feature`] with the concrete reason.
pub fn run_mir_program(program: &MirProgram, entry: &str) -> Result<RunResult, UnsupportedReason> {
    let ir_program =
        convert_program(program).map_err(|e| UnsupportedReason::Feature(e.to_string()))?;
    let mut interp = Interpreter::new(&ir_program);
    match interp.run(entry, Vec::new()) {
        Ok(Value::Unit) => Ok(RunResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(0),
        }),
        Ok(other) => Err(UnsupportedReason::Feature(format!(
            "entry `{entry}` returned {other}, expected ()"
        ))),
        Err(InterpError::Panic { message }) => Ok(RunResult {
            stdout: String::new(),
            stderr: format!("thread 'main' panicked at sparagmos-interp:\n{message}\n"),
            exit_code: Some(101),
        }),
        Err(e) => Err(UnsupportedReason::Feature(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparagmos_interp::MirFunction;
    use sparagmos_mir as mir;
    use std::path::PathBuf;

    fn unit_const() -> mir::Operand {
        mir::Operand::constant(mir::Constant::from_value(mir::ConstValue::ZeroSized))
    }

    fn block(stmts: Vec<mir::Statement>, kind: mir::TerminatorKind) -> mir::BasicBlockData {
        let mut bb = mir::BasicBlockData::new(mir::Terminator::new(kind));
        bb.statements = stmts;
        bb
    }

    fn set_unit() -> mir::Statement {
        mir::Statement::assign(
            mir::Place::from_local(mir::Local(0)),
            mir::Rvalue::Use(unit_const()),
        )
    }

    fn program_with_main(body: mir::Body) -> MirProgram {
        let mut prog = MirProgram::new();
        prog.functions.push(MirFunction {
            name: "main".to_string(),
            def_id: None,
            body,
        });
        prog
    }

    #[test]
    fn unit_main_exits_zero() {
        let body = mir::Body::new(
            vec![mir::LocalDecl::new(mir::Ty::unit())],
            0,
            vec![block(vec![set_unit()], mir::TerminatorKind::Return)],
        );
        let res = run_mir_program(&program_with_main(body), "main").expect("unit main runs");
        assert_eq!(res.exit_code, Some(0));
        assert_eq!(res.stdout, "");
    }

    #[test]
    fn panicking_main_exits_101_with_message() {
        let body = mir::Body::new(
            vec![
                mir::LocalDecl::new(mir::Ty::unit()),
                mir::LocalDecl::new(mir::Ty::Bool),
            ],
            0,
            vec![
                block(
                    vec![],
                    mir::TerminatorKind::Assert {
                        cond: mir::Operand::constant(mir::Constant::bool(false)),
                        expected: true,
                        msg: Box::new(mir::AssertMessage::Custom("boom".to_string())),
                        target: mir::BasicBlock(1),
                        unwind: mir::UnwindAction::Continue,
                    },
                ),
                block(vec![set_unit()], mir::TerminatorKind::Return),
            ],
        );
        let res = run_mir_program(&program_with_main(body), "main").expect("panic is a run result");
        assert_eq!(res.exit_code, Some(101));
        assert!(res.stderr.contains("boom"), "stderr: {:?}", res.stderr);
    }

    #[test]
    fn unsupported_main_reports_feature_not_a_fake_result() {
        let body = mir::Body::new(
            vec![
                mir::LocalDecl::new(mir::Ty::unit()),
                mir::LocalDecl::new(mir::Ty::Uint(mir::UintTy::U32)),
            ],
            0,
            vec![block(
                vec![],
                mir::TerminatorKind::Drop {
                    place: mir::Place::from_local(mir::Local(1)),
                    target: mir::BasicBlock(0),
                    unwind: mir::UnwindAction::Continue,
                    replace: false,
                },
            )],
        );
        let err = run_mir_program(&program_with_main(body), "main")
            .expect_err("Drop main must be unsupported");
        match err {
            UnsupportedReason::Feature(what) => {
                assert!(what.contains("Drop"), "reason: {what}")
            }
            other => panic!("expected Feature reason, got {other}"),
        }
    }

    #[test]
    fn corpus_runner_reports_driver_waiting() {
        let case = CorpusCase::new(PathBuf::from("arith_int.rs"));
        let err = InterpRunner
            .run(&case)
            .expect_err("no driver yet, so no execution");
        match err {
            UnsupportedReason::Feature(what) => {
                assert!(what.contains("sparagmos-driver"), "reason: {what}")
            }
            other => panic!("expected Feature reason, got {other}"),
        }
    }
}
