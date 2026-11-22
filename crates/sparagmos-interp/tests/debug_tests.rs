//! Integration tests for the interpreter's debugger interface:
//! line breakpoints, stepping, and scalar printing over a
//! hand-built program with recorded source lines and debug
//! variables, in the shape the MIR converter produces.

use sparagmos_interp::ir::*;
use sparagmos_interp::{
    DebugCommand, DebugDialect, DebugPrint, DebugSession, InterpConfig,
};

const SOURCE: &str = "fn main() {\n    let x = 40 + 2;\n    zzz();\n}\n";

fn line(n: u32) -> SourceLine {
    SourceLine { file: 0, line: n }
}

/// fn main() { let x = 40 + 2; zzz(); }
/// block 0 (line 2): _1 = Add(40, 2); goto block 1
/// block 1 (line 3): return
fn sample_program() -> Program {
    program_with_vars(vec![DebugVarEntry {
        name: "x".to_string(),
        source: DebugVar::Place(1, vec![]),
        is_char: false,
        lines: (2, 2),
    }])
}

fn program_with_vars(debug_vars: Vec<DebugVarEntry>) -> Program {
    let mut body = Body::new(
        "main",
        0,
        vec![Ty::i64(), Ty::i64()],
        vec![
            BasicBlockData::new(
                vec![Statement::assign(
                    Place::from_index(1),
                    Rvalue::binary(
                        BinOp::Add,
                        Operand::const_int(40),
                        Operand::const_int(2),
                    ),
                )],
                TerminatorKind::Goto { target: BasicBlock(1) },
            ),
            BasicBlockData::new(
                vec![Statement::assign(
                    Place::from_index(0),
                    Rvalue::Use(Operand::copy_local(1)),
                )],
                TerminatorKind::Return,
            ),
        ],
    );
    body.source_lines = vec![
        BlockSourceLines { statements: vec![line(2)], terminator: line(2) },
        BlockSourceLines { statements: vec![line(3)], terminator: line(3) },
    ];
    body.debug_vars = debug_vars;
    let mut p = Program::new();
    p.add_function(body);
    p.source_files = vec!["test.rs".to_string()];
    p
}

fn session(program: Program) -> DebugSession {
    let mut s = DebugSession::new(
        program,
        "main".to_string(),
        vec![],
        InterpConfig::default(),
    );
    s.set_source_text("test.rs", SOURCE);
    s
}

fn print(expr: &str) -> DebugCommand {
    DebugCommand::Print(DebugPrint { expr: expr.to_string(), format: None })
}

#[test]
fn breakpoint_stop_and_scalar_print() {
    let mut s = session(sample_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("x")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "zzz();"),
        "transcript: {:?}",
        run.transcript
    );
    assert!(
        run.transcript.iter().any(|l| l == "$1 = 42"),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn no_breakpoint_runs_to_completion_silently() {
    let mut s = session(sample_program());
    let run = s.run_script(vec![DebugCommand::Run], DebugDialect::Gdb);
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().all(|l| !l.starts_with('$')),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn step_over_from_a_breakpoint_reaches_the_next_line() {
    let mut s = session(sample_program());
    s.set_breakpoint("test.rs", 2);
    let run = s.run_script(
        vec![DebugCommand::Run, DebugCommand::Next, print("x")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    // The next-stop is line 3, after the assignment executed.
    assert!(
        run.transcript.iter().any(|l| l == "$1 = 42"),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn unknown_variable_is_a_transcript_error_not_a_failure() {
    let mut s = session(sample_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("y")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript
            .iter()
            .any(|l| l.contains("no variable `y` in scope")),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn constant_debug_variable_prints_its_recorded_value() {
    // A constant-valued variable whose storage the optimizer
    // eliminated carries its value in its debug info; the print
    // reads it from there.
    let p = program_with_vars(vec![DebugVarEntry {
        name: "shadowed".to_string(),
        source: DebugVar::Constant(Constant::Int(231)),
        is_char: false,
        lines: (2, 2),
    }]);
    let mut s = session(p);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("shadowed")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "$1 = 231"),
        "transcript: {:?}",
        run.transcript
    );
}
