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
        ty: None,
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
        ty: None,
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

/// A program whose main builds aggregate values — a named-field
/// struct, an enum, a tuple, an array, a nested struct, a `Vec`
/// in the std model's element form, and an `Option` — with the
/// debug-variable type tags and debug shapes the converter
/// records for them.
fn aggregate_program() -> Program {
    use sparagmos_mir::{AdtDebugField, AdtDebugShape, AdtDebugVariant, AdtShapeKind};
    fn field(name: Option<&str>, ty: &str) -> AdtDebugField {
        AdtDebugField { name: name.map(str::to_string), ty: ty.to_string() }
    }
    fn shape(kind: AdtShapeKind, variants: Vec<AdtDebugVariant>) -> AdtDebugShape {
        AdtDebugShape { kind, variants }
    }
    fn variant(name: &str, fields: Vec<AdtDebugField>) -> AdtDebugVariant {
        AdtDebugVariant { name: name.to_string(), fields }
    }
    fn var(name: &str, local: usize, ty: &str) -> DebugVarEntry {
        DebugVarEntry {
            name: name.to_string(),
            source: DebugVar::Place(local, vec![]),
            is_char: false,
            ty: Some(ty.to_string()),
            lines: (2, 2),
        }
    }
    fn adt(variant: u32, operands: Vec<Operand>) -> Rvalue {
        Rvalue::AggregateAdt {
            variant,
            operands,
            field_index: None,
            units_from: None,
        }
    }
    let debug_vars = vec![
        var("p", 1, "test::Point"),
        var("e", 2, "test::E"),
        var("t", 3, "(i32, bool)"),
        var("a", 4, "[i32; 3]"),
        var("q", 5, "test::Outer"),
        var("v", 6, "alloc::vec::Vec<i32>"),
        var("o", 7, "std::option::Option<i32>"),
    ];
    let mut body = Body::new(
        "main",
        0,
        vec![
            Ty::Unit,
            Ty::Adt("test::Point".to_string()),
            Ty::Adt("test::E".to_string()),
            Ty::Tuple(vec![Ty::i64(), Ty::Bool]),
            Ty::Array(Box::new(Ty::i64()), 3),
            Ty::Adt("test::Outer".to_string()),
            Ty::Adt("alloc::vec::Vec<i32>".to_string()),
            Ty::Adt("std::option::Option<i32>".to_string()),
        ],
        vec![
            BasicBlockData::new(
                vec![
                    Statement::assign(
                        Place::from_index(1),
                        adt(0, vec![Operand::const_int(10), Operand::const_int(20)]),
                    ),
                    Statement::assign(
                        Place::from_index(2),
                        adt(1, vec![Operand::const_uint(7)]),
                    ),
                    Statement::assign(
                        Place::from_index(3),
                        Rvalue::AggregateTuple(vec![
                            Operand::const_int(1),
                            Operand::const_bool(true),
                        ]),
                    ),
                    Statement::assign(
                        Place::from_index(4),
                        Rvalue::AggregateTuple(vec![
                            Operand::const_int(5),
                            Operand::const_int(5),
                            Operand::const_int(5),
                        ]),
                    ),
                    Statement::assign(
                        Place::from_index(5),
                        adt(
                            0,
                            vec![
                                Operand::copy_local(1),
                                Operand::const_int(3),
                            ],
                        ),
                    ),
                    Statement::assign(
                        Place::from_index(6),
                        adt(
                            0,
                            vec![
                                Operand::const_int(1),
                                Operand::const_int(2),
                                Operand::const_int(3),
                            ],
                        ),
                    ),
                    Statement::assign(
                        Place::from_index(7),
                        adt(1, vec![Operand::const_int(4)]),
                    ),
                ],
                TerminatorKind::Goto { target: BasicBlock(1) },
            ),
            BasicBlockData::new(
                vec![Statement::assign(
                    Place::from_index(0),
                    Rvalue::Use(Operand::const_unit()),
                )],
                TerminatorKind::Return,
            ),
        ],
    );
    body.source_lines = vec![
        BlockSourceLines { statements: vec![line(2); 7], terminator: line(2) },
        BlockSourceLines { statements: vec![line(3)], terminator: line(3) },
    ];
    body.debug_vars = debug_vars;
    let mut p = Program::new();
    p.add_function(body);
    p.source_files = vec!["test.rs".to_string()];
    p.adt_debug_shapes.insert(
        "test::Point".to_string(),
        shape(
            AdtShapeKind::Struct,
            vec![variant(
                "Point",
                vec![field(Some("x"), "i32"), field(Some("y"), "i32")],
            )],
        ),
    );
    p.adt_debug_shapes.insert(
        "test::E".to_string(),
        shape(
            AdtShapeKind::Enum,
            vec![
                variant("A", vec![]),
                variant("B", vec![field(None, "u32")]),
            ],
        ),
    );
    p.adt_debug_shapes.insert(
        "test::Outer".to_string(),
        shape(
            AdtShapeKind::Struct,
            vec![variant(
                "Outer",
                vec![
                    field(Some("inner"), "test::Point"),
                    field(Some("tag"), "i32"),
                ],
            )],
        ),
    );
    p.adt_debug_shapes.insert(
        "std::option::Option<i32>".to_string(),
        shape(
            AdtShapeKind::Enum,
            vec![
                variant("None", vec![]),
                variant("Some", vec![field(None, "i32")]),
            ],
        ),
    );
    p
}

#[test]
fn gdb_prints_aggregate_values_in_rust_form() {
    let mut s = session(aggregate_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            print("p"),
            print("e"),
            print("t"),
            print("a"),
            print("q"),
            print("o"),
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "$1 = test::Point {x: 10, y: 20}",
        "$2 = test::E::B(7)",
        "$3 = (1, true)",
        "$4 = [5, 5, 5]",
        "$5 = test::Outer {inner: test::Point {x: 10, y: 20}, tag: 3}",
        // The printed name is the definition path, not the
        // `std` re-export path the recorded tag carries.
        "$6 = core::option::Option<i32>::Some(4)",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

#[test]
fn lldb_prints_aggregate_values_in_summary_form() {
    let mut s = session(aggregate_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            print("p"),
            print("e"),
            print("t"),
            print("a"),
            print("q"),
            print("v"),
        ],
        DebugDialect::Lldb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "(test::Point) p = {x:10, y:20}",
        "(test::E) e = B(7)",
        "= (1, true)",
        "= (5, 5, 5)",
        "(test::Outer) q = {inner:{x:10, y:20}, tag:3}",
        "(alloc::vec::Vec<int>) v = size=3 { [0] = 1 [1] = 2 [2] = 3 }",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

/// A program whose debug variables are constants the conversion
/// recorded: a slice, a string slice, and an unassigned `Vec`
/// (the pretty-uninitialized-vec shape).
fn constant_sequence_program() -> Program {
    fn cvar(name: &str, constant: Constant, ty: &str) -> DebugVarEntry {
        DebugVarEntry {
            name: name.to_string(),
            source: DebugVar::Constant(constant),
            is_char: false,
            ty: Some(ty.to_string()),
            lines: (2, 2),
        }
    }
    let p = program_with_vars(vec![
        cvar(
            "slice",
            Constant::Ref(Box::new(Constant::Tuple(vec![
                Constant::Int(0),
                Constant::Int(1),
                Constant::Int(2),
            ]))),
            "&[i32]",
        ),
        cvar(
            "text",
            Constant::Ref(Box::new(Constant::Str(
                "string slice".to_string(),
            ))),
            "&str",
        ),
        cvar("vec", Constant::Uninit, "alloc::vec::Vec<i32>"),
    ]);
    p
}

#[test]
fn gdb_pretty_printers_render_provider_forms() {
    // With the printers enabled (compiletest's default), a
    // value whose type has a sysroot provider renders in the
    // provider's form; a value with no provider keeps the raw
    // form.
    let mut s = session(aggregate_program());
    s.set_pretty_printers(true);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("v"), print("p")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "$1 = Vec(size=3) = {1, 2, 3}",
        "$2 = test::Point {x: 10, y: 20}",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

#[test]
fn gdb_pretty_printers_render_slice_str_and_uninit_vec() {
    let mut s = session(constant_sequence_program());
    s.set_pretty_printers(true);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            print("slice"),
            print("text"),
            print("vec"),
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "$1 = &[i32](size=3) = {0, 1, 2}",
        "$2 = \"string slice\"",
        "$3 = Vec(size=0)",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

#[test]
fn lldb_provider_lines_for_slice_and_str() {
    let mut s = session(constant_sequence_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("slice"), print("text")],
        DebugDialect::Lldb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "(&[i32]) slice = size=3 { [0] = 0 [1] = 1 [2] = 2 }",
        "(&str) text = \"string slice\" { [0] = 's' [1] = 't' [2] = 'r' \
         [3] = 'i' [4] = 'n' [5] = 'g' [6] = ' ' [7] = 's' [8] = 'l' \
         [9] = 'i' [10] = 'c' [11] = 'e' }",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

#[test]
fn gdb_without_pretty_printers_keeps_raw_vec_form() {
    // The printers are opt-in per session: without them the
    // `Vec` value has no raw rendering (its element aggregate
    // does not match the library layout in the shape table),
    // exactly as before.
    let mut s = session(aggregate_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("v")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript
            .iter()
            .all(|l| !l.contains("Vec(size=")),
        "transcript: {:?}",
        run.transcript
    );
}
