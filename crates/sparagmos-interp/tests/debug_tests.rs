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
    SourceLine { file: 0, line: n, col: 0 }
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
        run.transcript.iter().any(|l| l == "3\t    zzz();"),
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
            payload_entry: None,
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
fn gdb_print_elements_truncates_printed_strings() {
    // gdb's `set print elements` reaches the session as a
    // command: a print after it truncates a long string at
    // the limit, the way the real debugger's does, while a
    // print before it (under the default limit) does not.
    fn cvar(name: &str, constant: Constant, ty: &str) -> DebugVarEntry {
        DebugVarEntry {
            name: name.to_string(),
            source: DebugVar::Constant(constant),
            is_char: false,
            ty: Some(ty.to_string()),
            lines: (2, 2),
        }
    }
    let p = program_with_vars(vec![cvar(
        "owned",
        Constant::Str("IAMA optional string!".to_string()),
        "alloc::string::String",
    )]);
    let mut s = session(p);
    s.set_pretty_printers(true);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            print("owned"),
            DebugCommand::SetPrintElements(5),
            print("owned"),
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "$1 = \"IAMA optional string!\"",
        "$2 = \"IAMA \"...",
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

// ---------------------------------------------------------------------------
// The D2 band: statics (before and after `run`), type queries,
// the `repr` pseudo-command, named-field steps, function
// breakpoints, the info listings, and lldb type formats.
// ---------------------------------------------------------------------------

fn typed_scalar_program() -> Program {
    program_with_vars(vec![DebugVarEntry {
        name: "x".to_string(),
        source: DebugVar::Place(1, vec![]),
        is_char: false,
        ty: Some("i64".to_string()),
        lines: (2, 2),
    }])
}

#[test]
fn whatis_reports_the_recorded_type() {
    let mut s = session(typed_scalar_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            DebugCommand::WhatIs(DebugPrint {
                expr: "x".to_string(),
                format: None,
            }),
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "type = i64"),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn whatis_of_a_function_pointer_uses_the_raw_pointer_spelling() {
    // A function pointer's recorded tag is the type's kind
    // name (`FnPtr`); gdb spells the type as a raw pointer to
    // the unit-signature function.
    let p = program_with_vars(vec![DebugVarEntry {
        name: "f".to_string(),
        source: DebugVar::Constant(Constant::Int(0)),
        is_char: false,
        ty: Some("FnPtr".to_string()),
        lines: (2, 2),
    }]);
    let mut s = session(p);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            DebugCommand::WhatIs(DebugPrint {
                expr: "f".to_string(),
                format: None,
            }),
        ],
        DebugDialect::Gdb,
    );
    assert!(
        run.transcript.iter().any(|l| l == "type = *mut fn ()"),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn ptype_of_a_struct_prints_its_layout() {
    let mut s = session(aggregate_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            DebugCommand::Ptype(DebugPrint {
                expr: "p".to_string(),
                format: None,
            }),
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "type = struct test::Point {",
        "    i32 x,",
        "    i32 y,",
        "}",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

#[test]
fn repr_reports_ok_for_a_renderable_variable() {
    let mut s = session(typed_scalar_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, DebugCommand::Repr("x".to_string())],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "x: Ok"),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn named_field_step_prints_the_field() {
    let mut s = session(aggregate_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("p.x"), print("p.y")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in ["$1 = 10", "$2 = 20"] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

/// A program whose main mentions a static (the mention is
/// what seeds the debugger's static store), with the static's
/// declared type recorded the way the converter records it.
fn static_program() -> Program {
    let mut body = Body::new(
        "main",
        0,
        vec![Ty::Unit, Ty::Ref(Box::new(Ty::i64()))],
        vec![
            BasicBlockData::new(
                vec![Statement::assign(
                    Place::from_index(1),
                    Rvalue::Use(Operand::Constant(Constant::StaticRef {
                        name: "test::COUNTER".to_string(),
                        projection: vec![],
                        len: None,
                    })),
                )],
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
        BlockSourceLines { statements: vec![line(2)], terminator: line(2) },
        BlockSourceLines { statements: vec![line(3)], terminator: line(3) },
    ];
    let mut p = Program::new();
    p.add_function(body);
    p.source_files = vec!["test.rs".to_string()];
    p.static_debug_tags
        .insert("test::COUNTER".to_string(), "i64".to_string());
    p.static_seeds
        .insert("test::COUNTER".to_string(), Constant::Int(7));
    p
}

#[test]
fn static_prints_before_run_at_its_initial_value() {
    // A print ahead of `run` reads the static at its initial
    // value, by its path or by its plain name alike.
    let mut s = session(static_program());
    let run = s.run_script(
        vec![print("test::COUNTER"), print("COUNTER"), DebugCommand::Run],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "$1 = 7"),
        "transcript: {:?}",
        run.transcript
    );
    assert!(
        run.transcript.iter().any(|l| l == "$2 = 7"),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn info_functions_lists_bodies_in_both_declaration_forms() {
    let mut s = session(typed_scalar_program());
    let run = s.run_script(
        vec![
            DebugCommand::InfoFunctions { pattern: None, quiet: true },
            DebugCommand::InfoFunctions { pattern: None, quiet: false },
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in ["static fn main();", "fn main();"] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

#[test]
fn info_functions_prefers_the_display_name_table() {
    let mut p = program_with_vars(vec![]);
    // A body whose key already spells the instance the DWARF
    // way but whose own local types would render a bare
    // signature: the table's signature must take its place.
    p.add_function(Body::new(
        "x::generic_func<i32>",
        0,
        vec![Ty::Unit],
        vec![BasicBlockData::new(vec![], TerminatorKind::Return)],
    ));
    p.fn_display_names = vec![
        (
            "x::generic_func<i32>".to_string(),
            "(i32) -> i32".to_string(),
        ),
        // An instance with no converted body is listed from
        // the table alone.
        ("x::helper<u8>".to_string(), "()".to_string()),
    ];
    let mut s = session(p);
    let run = s.run_script(
        vec![
            DebugCommand::InfoFunctions { pattern: None, quiet: true },
            DebugCommand::InfoFunctions {
                pattern: Some("helper".to_string()),
                quiet: true,
            },
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "static fn x::generic_func<i32>(i32) -> i32;",
        "static fn x::helper<u8>();",
        "static fn main() -> i64;",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
    assert!(
        !run.transcript
            .iter()
            .any(|l| l == "static fn x::generic_func<i32>();"),
        "body-derived signature leaked: {:?}",
        run.transcript
    );
}

#[test]
fn info_locals_lists_live_locals_and_honours_limited_debuginfo() {
    let mut s = session(typed_scalar_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, DebugCommand::InfoLocals],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "x = 42"),
        "transcript: {:?}",
        run.transcript
    );

    let mut s = session(typed_scalar_program());
    s.set_hide_locals(true);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, DebugCommand::InfoLocals],
        DebugDialect::Gdb,
    );
    assert!(
        run.transcript.iter().any(|l| l == "No locals."),
        "transcript: {:?}",
        run.transcript
    );
}

/// main (lines 2-3) calls helper (lines 5-6) and returns.
fn two_function_program() -> Program {
    let mut main = Body::new(
        "main",
        0,
        vec![Ty::Unit, Ty::Unit],
        vec![
            BasicBlockData::new(
                vec![],
                TerminatorKind::Call {
                    func: "test::helper".to_string(),
                    args: vec![],
                    destination: Place::from_index(1),
                    target: Some(BasicBlock(1)),
                },
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
    main.source_lines = vec![
        BlockSourceLines { statements: vec![], terminator: line(2) },
        BlockSourceLines { statements: vec![line(3)], terminator: line(3) },
    ];
    let mut helper = Body::new(
        "test::helper",
        0,
        vec![Ty::Unit],
        vec![BasicBlockData::new(
            vec![Statement::assign(
                Place::from_index(0),
                Rvalue::Use(Operand::const_unit()),
            )],
            TerminatorKind::Return,
        )],
    );
    helper.source_lines = vec![BlockSourceLines {
        statements: vec![line(5)],
        terminator: line(5),
    }];
    let mut p = Program::new();
    p.add_function(main);
    p.add_function(helper);
    p.source_files = vec!["test.rs".to_string()];
    p
}

fn two_function_session() -> DebugSession {
    let mut s = DebugSession::new(
        two_function_program(),
        "main".to_string(),
        vec![],
        InterpConfig::default(),
    );
    s.set_source_text(
        "test.rs",
        "fn main() {\n    helper();\n}\n\nfn helper() {\n    zzz();\n}\n",
    );
    s
}

#[test]
fn function_breakpoint_stops_at_the_callees_first_line() {
    let mut s = two_function_session();
    s.set_fn_breakpoint("test::helper");
    let run = s.run_script(vec![DebugCommand::Run], DebugDialect::Gdb);
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l.starts_with("5\t")),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn frame_up_emits_the_callers_line() {
    let mut s = two_function_session();
    s.set_fn_breakpoint("test::helper");
    let run = s.run_script(
        vec![DebugCommand::Run, DebugCommand::FrameUp],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    // After `up`, the shown line is main's call line (line 2).
    let stop = run
        .transcript
        .iter()
        .position(|l| l.starts_with("5\t"))
        .expect("the helper stop line");
    assert!(
        run.transcript[stop + 1..].iter().any(|l| l.starts_with("2\t")),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn lldb_hex_format_registration_renders_an_i8_in_hex() {
    use sparagmos_interp::LldbNumFormat;
    let p = program_with_vars(vec![DebugVarEntry {
        name: "b".to_string(),
        source: DebugVar::Constant(Constant::Int(111)),
        is_char: false,
        ty: Some("i8".to_string()),
        lines: (2, 2),
    }]);
    let mut s = session(p);
    s.set_lldb_format("i8", LldbNumFormat::Hex);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("b")],
        DebugDialect::Lldb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "= 0x6f"),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn lldb_static_print_names_the_variable_in_its_typed_line() {
    // lldb's `(Type) name = value` head line for a static
    // printed by path: the checks that name the variable
    // anchor on it.
    let mut s = session(static_program());
    let run = s.run_script(
        vec![print("test::COUNTER"), DebugCommand::Run],
        DebugDialect::Lldb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript
            .iter()
            .any(|l| l == "(i64) test::COUNTER = 7"),
        "transcript: {:?}",
        run.transcript
    );
}

// ---------------------------------------------------------------------------
// Projected primitive type names (a field / element / pointee
// step must keep the concrete primitive type the shape table
// recorded, not collapse it to a width-less descriptor) and the
// `length` resolution order (a recorded field of that name wins
// over the sequence pseudo-field).
// ---------------------------------------------------------------------------

fn whatis(expr: &str) -> DebugCommand {
    DebugCommand::WhatIs(DebugPrint {
        expr: expr.to_string(),
        format: None,
    })
}

#[test]
fn whatis_of_projected_primitives_keeps_the_concrete_name() {
    let mut s = session(aggregate_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            whatis("p.x"),
            whatis("a[0]"),
            whatis("t.0"),
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert_eq!(
        run.transcript
            .iter()
            .filter(|l| *l == "type = i32")
            .count(),
        3,
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn whatis_of_a_dereferenced_pointee_keeps_the_concrete_name() {
    let p = program_with_vars(vec![DebugVarEntry {
        name: "r".to_string(),
        source: DebugVar::Constant(Constant::Ref(Box::new(Constant::Int(42)))),
        is_char: false,
        ty: Some("&i32".to_string()),
        lines: (2, 2),
    }]);
    let mut s = session(p);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, whatis("*r")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "type = i32"),
        "transcript: {:?}",
        run.transcript
    );
}

/// A struct with an ordinary field named `length` and a float
/// field: `test::Bag { length: i32, ratio: f64 }`.
fn named_length_program() -> Program {
    use sparagmos_mir::{AdtDebugField, AdtDebugShape, AdtDebugVariant, AdtShapeKind};
    let mut body = Body::new(
        "main",
        0,
        vec![Ty::Unit, Ty::Adt("test::Bag".to_string())],
        vec![
            BasicBlockData::new(
                vec![Statement::assign(
                    Place::from_index(1),
                    Rvalue::AggregateAdt {
                        variant: 0,
                        operands: vec![
                            Operand::const_int(7),
                            Operand::Constant(Constant::Float(2.5, FloatTy::F64)),
                        ],
                        field_index: None,
                        units_from: None,
                        payload_entry: None,
                    },
                )],
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
        BlockSourceLines { statements: vec![line(2)], terminator: line(2) },
        BlockSourceLines { statements: vec![line(3)], terminator: line(3) },
    ];
    body.debug_vars = vec![DebugVarEntry {
        name: "b".to_string(),
        source: DebugVar::Place(1, vec![]),
        is_char: false,
        ty: Some("test::Bag".to_string()),
        lines: (2, 2),
    }];
    let mut p = Program::new();
    p.add_function(body);
    p.source_files = vec!["test.rs".to_string()];
    p.adt_debug_shapes.insert(
        "test::Bag".to_string(),
        AdtDebugShape {
            kind: AdtShapeKind::Struct,
            variants: vec![AdtDebugVariant {
                name: "Bag".to_string(),
                fields: vec![
                    AdtDebugField {
                        name: Some("length".to_string()),
                        ty: "i32".to_string(),
                    },
                    AdtDebugField {
                        name: Some("ratio".to_string()),
                        ty: "f64".to_string(),
                    },
                ],
            }],
        },
    );
    p
}

#[test]
fn length_named_field_wins_over_the_sequence_pseudo_field() {
    let mut s = session(named_length_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("b.length"), whatis("b.length")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in ["$1 = 7", "type = i32"] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

#[test]
fn whatis_of_a_float_field_keeps_the_concrete_name() {
    let mut s = session(named_length_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, whatis("b.ratio")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "type = f64"),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn length_pseudo_field_still_reads_a_sequence() {
    let mut s = session(constant_sequence_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("slice.length")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "$1 = 3"),
        "transcript: {:?}",
        run.transcript
    );
}

/// A program whose main builds a union value through one
/// field: `let u = test::U { bits: 0x3f800000 }` for
/// `union U { bits: u32, value: f32 }`. The value model
/// stores only the written field, so reading the other
/// field reinterprets that storage.
fn union_program() -> Program {
    use sparagmos_mir::{AdtDebugField, AdtDebugShape, AdtDebugVariant, AdtShapeKind};
    let mut body = Body::new(
        "main",
        0,
        vec![Ty::Unit, Ty::Adt("test::U".to_string())],
        vec![
            BasicBlockData::new(
                vec![Statement::assign(
                    Place::from_index(1),
                    Rvalue::AggregateAdt {
                        variant: 0,
                        operands: vec![Operand::const_uint(1065353216)],
                        field_index: None,
                        units_from: None,
                        payload_entry: None,
                    },
                )],
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
        BlockSourceLines {
            statements: vec![line(2)],
            terminator: line(2),
        },
        BlockSourceLines {
            statements: vec![line(3)],
            terminator: line(3),
        },
    ];
    body.debug_vars = vec![DebugVarEntry {
        name: "u".to_string(),
        source: DebugVar::Place(1, vec![]),
        is_char: false,
        ty: Some("test::U".to_string()),
        lines: (2, 2),
    }];
    let mut p = Program::new();
    p.add_function(body);
    p.source_files = vec!["test.rs".to_string()];
    p.adt_debug_shapes.insert(
        "test::U".to_string(),
        AdtDebugShape {
            kind: AdtShapeKind::Union,
            variants: vec![AdtDebugVariant {
                name: "U".to_string(),
                fields: vec![
                    AdtDebugField {
                        name: Some("bits".to_string()),
                        ty: "u32".to_string(),
                    },
                    AdtDebugField {
                        name: Some("value".to_string()),
                        ty: "f32".to_string(),
                    },
                ],
            }],
        },
    );
    p
}

#[test]
fn union_field_print_reinterprets_the_stored_field() {
    // Inspecting one field of a union agrees with printing
    // the whole union: the field the program did not write
    // reads through the written field's storage instead of
    // reporting an uninitialized read.
    let mut s = session(union_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            print("u"),
            print("u.value"),
            print("u.bits"),
            whatis("u.value"),
            whatis("u.bits"),
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "$1 = test::U {bits: 1065353216, value: 1}",
        "$2 = 1",
        "$3 = 1065353216",
        "type = f32",
        "type = u32",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

/// A program whose first statement stores a zero-sized value:
/// block 0 (line 2) `_1 = const ()` (no code emitted), then
/// (line 3) `_2 = 40 + 2`; block 1 (line 4) returns `_2`.
fn zst_first_program() -> Program {
    let mut body = Body::new(
        "main",
        0,
        vec![Ty::i64(), Ty::Unit, Ty::i64()],
        vec![
            BasicBlockData::new(
                vec![
                    Statement::assign(
                        Place::from_index(1),
                        Rvalue::Use(Operand::const_unit()),
                    ),
                    Statement::assign(
                        Place::from_index(2),
                        Rvalue::binary(
                            BinOp::Add,
                            Operand::const_int(40),
                            Operand::const_int(2),
                        ),
                    ),
                ],
                TerminatorKind::Goto { target: BasicBlock(1) },
            ),
            BasicBlockData::new(
                vec![Statement::assign(
                    Place::from_index(0),
                    Rvalue::Use(Operand::copy_local(2)),
                )],
                TerminatorKind::Return,
            ),
        ],
    );
    body.source_lines = vec![
        BlockSourceLines {
            statements: vec![line(2), line(3)],
            terminator: line(3),
        },
        BlockSourceLines { statements: vec![line(4)], terminator: line(4) },
    ];
    body.debug_vars = vec![DebugVarEntry {
        name: "x".to_string(),
        source: DebugVar::Place(2, vec![]),
        is_char: false,
        ty: None,
        lines: (3, 3),
    }];
    let mut p = Program::new();
    p.add_function(body);
    p.source_files = vec!["test.rs".to_string()];
    p
}

const ZST_SOURCE: &str =
    "fn main() {\n    let a = ();\n    let x = 40 + 2;\n}\n";

#[test]
fn breakpoint_on_a_zero_sized_line_slides_to_the_next_code_line() {
    // A native debugger binds a breakpoint on a line whose only
    // statement emits no code (a zero-sized store) at the next
    // line that carries code.
    let mut s = DebugSession::new(
        zst_first_program(),
        "main".to_string(),
        vec![],
        InterpConfig::default(),
    );
    s.set_source_text("test.rs", ZST_SOURCE);
    s.set_breakpoint("test.rs", 2);
    let run = s.run_script(vec![DebugCommand::Run], DebugDialect::Gdb);
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "3\t    let x = 40 + 2;"),
        "transcript: {:?}",
        run.transcript
    );
    assert!(
        run.transcript.iter().all(|l| !l.starts_with("2\t")),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn stepping_never_stops_on_a_zero_sized_line() {
    let mut s = DebugSession::new(
        zst_first_program(),
        "main".to_string(),
        vec![],
        InterpConfig::default(),
    );
    s.set_source_text("test.rs", ZST_SOURCE);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, DebugCommand::Next],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().all(|l| !l.starts_with("2\t")),
        "transcript: {:?}",
        run.transcript
    );
}

/// A unit-returning main: block 1 stores the unit return value
/// on the signature line (line 1), as MIR spans `_0 = const ()`
/// there, before the `Return` on the closing-brace line (3).
fn unit_return_program() -> Program {
    let mut body = Body::new(
        "main",
        0,
        vec![Ty::Unit, Ty::i64()],
        vec![
            BasicBlockData::new(
                vec![Statement::assign(
                    Place::from_index(1),
                    Rvalue::Use(Operand::const_int(42)),
                )],
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
        BlockSourceLines { statements: vec![line(2)], terminator: line(2) },
        BlockSourceLines { statements: vec![line(1)], terminator: line(3) },
    ];
    body.debug_vars = vec![DebugVarEntry {
        name: "x".to_string(),
        source: DebugVar::Place(1, vec![]),
        is_char: false,
        ty: None,
        lines: (2, 2),
    }];
    let mut p = Program::new();
    p.add_function(body);
    p.source_files = vec!["test.rs".to_string()];
    p
}

#[test]
fn step_over_the_last_statement_lands_on_the_closing_brace_not_the_signature() {
    let mut s = DebugSession::new(
        unit_return_program(),
        "main".to_string(),
        vec![],
        InterpConfig::default(),
    );
    s.set_source_text("test.rs", "fn main() {\n    let x = 42;\n}\n");
    s.set_breakpoint("test.rs", 2);
    let run = s.run_script(
        vec![DebugCommand::Run, DebugCommand::Next],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "3\t}"),
        "transcript: {:?}",
        run.transcript
    );
    assert!(
        run.transcript.iter().all(|l| !l.starts_with("1\t")),
        "transcript: {:?}",
        run.transcript
    );
}

/// The macro-expansion shape: a user statement on line 2, a
/// compiler temp's store on line 3, and the call that uses the
/// temp on line 4 (the using terminator's span, after macro
/// collapse, sits on a different line than the temp's span).
fn temp_store_program() -> Program {
    let mut helper = Body::new(
        "helper",
        1,
        vec![Ty::Unit, Ty::i64()],
        vec![BasicBlockData::new(vec![], TerminatorKind::Return)],
    );
    helper.source_lines =
        vec![BlockSourceLines { statements: vec![], terminator: line(4) }];
    let mut body = Body::new(
        "main",
        0,
        vec![Ty::Unit, Ty::i64(), Ty::i64(), Ty::Unit],
        vec![
            BasicBlockData::new(
                vec![
                    Statement::assign(
                        Place::from_index(1),
                        Rvalue::Use(Operand::const_int(1)),
                    ),
                    Statement::assign(
                        Place::from_index(2),
                        Rvalue::Use(Operand::const_int(2)),
                    ),
                ],
                TerminatorKind::Call {
                    func: "helper".to_string(),
                    args: vec![Operand::copy_local(2)],
                    destination: Place::from_index(3),
                    target: Some(BasicBlock(1)),
                },
            ),
            BasicBlockData::new(vec![], TerminatorKind::Return),
        ],
    );
    body.source_lines = vec![
        BlockSourceLines {
            statements: vec![line(2), line(3)],
            terminator: line(4),
        },
        BlockSourceLines { statements: vec![], terminator: line(5) },
    ];
    body.debug_vars = vec![DebugVarEntry {
        name: "x".to_string(),
        source: DebugVar::Place(1, vec![]),
        is_char: false,
        ty: None,
        lines: (2, 2),
    }];
    let mut p = Program::new();
    p.add_function(body);
    p.add_function(helper);
    p.source_files = vec!["test.rs".to_string()];
    p
}

#[test]
fn step_over_skips_a_temp_store_for_the_call_it_feeds() {
    let mut s = DebugSession::new(
        temp_store_program(),
        "main".to_string(),
        vec![],
        InterpConfig::default(),
    );
    s.set_source_text(
        "test.rs",
        "fn main() {\n    let x = 1;\n    let y = 2;\n    helper();\n}\n",
    );
    s.set_breakpoint("test.rs", 2);
    let run = s.run_script(
        vec![DebugCommand::Run, DebugCommand::Next],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "4\t    helper();"),
        "transcript: {:?}",
        run.transcript
    );
    assert!(
        run.transcript.iter().all(|l| !l.starts_with("3\t")),
        "transcript: {:?}",
        run.transcript
    );
}


#[test]
fn history_reference_rereads_an_earlier_print_gdb() {
    let mut s = session(sample_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("x"), print("$1")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "$1 = 42"),
        "transcript: {:?}",
        run.transcript
    );
    // The history print is itself a print: its value takes
    // the next history slot.
    assert!(
        run.transcript.iter().any(|l| l == "$2 = 42"),
        "transcript: {:?}",
        run.transcript
    );
}

/// A program with two in-scope values of different content:
/// `x`, the computed local (42 at the line-3 stop), and `c`, a
/// constant-valued variable (7).
fn two_value_program() -> Program {
    program_with_vars(vec![
        DebugVarEntry {
            name: "x".to_string(),
            source: DebugVar::Place(1, vec![]),
            is_char: false,
            ty: None,
            lines: (2, 2),
        },
        DebugVarEntry {
            name: "c".to_string(),
            source: DebugVar::Constant(Constant::Int(7)),
            is_char: false,
            ty: None,
            lines: (2, 2),
        },
    ])
}

#[test]
fn history_comparison_prints_the_equality_boolean_gdb() {
    let mut s = session(two_value_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            print("x"),
            print("c"),
            print("$1 == $2"),
            print("$1 != $2"),
            print("$1 == $1"),
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in ["$1 = 42", "$2 = 7", "$3 = false", "$4 = true", "$5 = true"] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in transcript: {:?}",
            run.transcript
        );
    }
}

#[test]
fn failed_print_keeps_its_history_slot_gdb() {
    let mut s = session(sample_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            print("y"),
            print("x"),
            print("$2 == $2"),
            print("$1"),
        ],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    // The failed print took $1, so `x` is $2 and its
    // self-comparison is $3; reading $1 back reports the
    // failed print's empty slot.
    assert!(
        run.transcript.iter().any(|l| l == "$2 = 42"),
        "transcript: {:?}",
        run.transcript
    );
    assert!(
        run.transcript.iter().any(|l| l == "$3 = true"),
        "transcript: {:?}",
        run.transcript
    );
    assert!(
        run.transcript
            .iter()
            .any(|l| l.contains("history value `$1` has no value")),
        "transcript: {:?}",
        run.transcript
    );
}

#[test]
fn history_comparison_lldb_numbers_from_zero() {
    let mut s = session(two_value_program());
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![
            DebugCommand::Run,
            print("x"),
            print("c"),
            print("$0 == $1"),
            print("$0 != $1"),
        ],
        DebugDialect::Lldb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript.iter().any(|l| l == "= false"),
        "transcript: {:?}",
        run.transcript
    );
    assert!(
        run.transcript.iter().any(|l| l == "= true"),
        "transcript: {:?}",
        run.transcript
    );
}

/// A program that writes a union local through the given
/// fields (one aggregate write per entry, in order) and stops
/// at a line where the union can be printed. `fields` are the
/// declared (name, type tag) pairs of `test::U`.
fn union_written_program(
    fields: Vec<(&str, &str)>,
    writes: Vec<(Option<u32>, u128)>,
) -> Program {
    use sparagmos_mir::{AdtDebugField, AdtDebugShape, AdtDebugVariant, AdtShapeKind};
    let statements: Vec<Statement> = writes
        .iter()
        .map(|(field_index, value)| {
            Statement::assign(
                Place::from_index(1),
                Rvalue::AggregateAdt {
                    variant: 0,
                    operands: vec![Operand::const_uint(*value)],
                    field_index: *field_index,
                    units_from: None,
                    payload_entry: None,
                },
            )
        })
        .collect();
    let n_statements = statements.len();
    let mut body = Body::new(
        "main",
        0,
        vec![Ty::Unit, Ty::Adt("test::U".to_string())],
        vec![
            BasicBlockData::new(statements, TerminatorKind::Goto { target: BasicBlock(1) }),
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
        BlockSourceLines {
            statements: vec![line(2); n_statements],
            terminator: line(2),
        },
        BlockSourceLines {
            statements: vec![line(3)],
            terminator: line(3),
        },
    ];
    body.debug_vars = vec![DebugVarEntry {
        name: "u".to_string(),
        source: DebugVar::Place(1, vec![]),
        is_char: false,
        ty: Some("test::U".to_string()),
        lines: (2, 2),
    }];
    let mut p = Program::new();
    p.add_function(body);
    p.source_files = vec!["test.rs".to_string()];
    p.adt_debug_shapes.insert(
        "test::U".to_string(),
        AdtDebugShape {
            kind: AdtShapeKind::Union,
            variants: vec![AdtDebugVariant {
                name: "U".to_string(),
                fields: fields
                    .iter()
                    .map(|(name, ty)| AdtDebugField {
                        name: Some(name.to_string()),
                        ty: ty.to_string(),
                    })
                    .collect(),
            }],
        },
    );
    p
}

#[test]
fn union_print_reinterprets_a_pointer_sibling() {
    // A raw-pointer field shares the written integer field's
    // storage: the whole-union print reads it back as the
    // address the integer's bytes name (as gdb does), instead
    // of failing the whole print on the pointer field.
    let mut s = session(union_written_program(
        vec![("bits", "u32"), ("p", "*const u8")],
        vec![(None, 0x04030201)],
    ));
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("u"), print("u.p"), print("u.bits")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "$1 = test::U {bits: 67305985, p: 0x4030201}",
        "$2 = 0x4030201",
        "$3 = 67305985",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

#[test]
fn union_print_degrades_an_unreadable_struct_sibling() {
    // A struct field's layout is not part of the recorded
    // debug shape, so its bytes cannot be reinterpreted; the
    // whole-union print degrades that field in place (the
    // form gdb uses for a field it cannot read) and still
    // prints the written field.
    use sparagmos_mir::{AdtDebugField, AdtDebugShape, AdtDebugVariant, AdtShapeKind};
    let mut program = union_written_program(
        vec![("bits", "u32"), ("s", "test::S")],
        vec![(None, 131073)],
    );
    program.adt_debug_shapes.insert(
        "test::S".to_string(),
        AdtDebugShape {
            kind: AdtShapeKind::Struct,
            variants: vec![AdtDebugVariant {
                name: "S".to_string(),
                fields: vec![
                    AdtDebugField {
                        name: Some("x".to_string()),
                        ty: "u16".to_string(),
                    },
                    AdtDebugField {
                        name: Some("y".to_string()),
                        ty: "u16".to_string(),
                    },
                ],
            }],
        },
    );
    let mut s = session(program);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("u"), print("u.bits")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "$1 = test::U {bits: 131073, s: <error reading variable>}",
        "$2 = 131073",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}

#[test]
fn union_print_reads_a_fat_sibling_from_scalar_storage() {
    // A slice field read out of a written integer field's
    // storage is the fat pointer those words name; in this
    // dialect a fat pointer prints as its address, like the
    // same value printed standalone.
    let mut s = session(union_written_program(
        vec![("bits", "u128"), ("s", "&[u8]")],
        vec![(None, 5)],
    ));
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("u"), print("u.s")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    assert!(
        run.transcript
            .iter()
            .any(|l| l.starts_with("$1 = test::U {bits: 5, s: 0x")),
        "missing fat-sibling whole print in {:?}",
        run.transcript
    );
    assert!(
        run.transcript.iter().any(|l| l.starts_with("$2 = 0x")),
        "missing fat-sibling field print in {:?}",
        run.transcript
    );
}

#[test]
fn union_print_reads_fields_from_byte_storage() {
    // A union written through two fields is held as raw
    // storage bytes; the whole-union print reads every field
    // out of those bytes, including the pointer field whose
    // tail bytes no write covered (they read as zero).
    use sparagmos_mir::{AdtDebugField, AdtDebugShape, AdtDebugVariant, AdtShapeKind};
    let mut body = Body::new(
        "main",
        0,
        vec![Ty::Unit, Ty::Adt("test::U".to_string())],
        vec![
            BasicBlockData::new(
                vec![
                    Statement::assign(
                        Place::from_index(1),
                        Rvalue::AggregateAdt {
                            variant: 0,
                            operands: vec![Operand::const_uint(0x04030201)],
                            field_index: Some(0),
                            units_from: None,
                            payload_entry: None,
                        },
                    ),
                    Statement::assign(
                        Place {
                            local: sparagmos_interp::ir::Local(1),
                            projection: vec![ProjectionElem::UnionField { idx: 1, table: 0 }],
                        },
                        Rvalue::Use(Operand::const_uint(0x9999)),
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
        BlockSourceLines {
            statements: vec![line(2), line(2)],
            terminator: line(2),
        },
        BlockSourceLines {
            statements: vec![line(3)],
            terminator: line(3),
        },
    ];
    body.debug_vars = vec![DebugVarEntry {
        name: "u".to_string(),
        source: DebugVar::Place(1, vec![]),
        is_char: false,
        ty: Some("test::U".to_string()),
        lines: (2, 2),
    }];
    let mut program = Program::new();
    program.add_function(body);
    program.source_files = vec!["test.rs".to_string()];
    program.union_tables = vec![vec![
        Some(ScalarShape::Int { bits: 32, signed: false }),
        Some(ScalarShape::Int { bits: 16, signed: false }),
        None,
    ]];
    program.adt_debug_shapes.insert(
        "test::U".to_string(),
        AdtDebugShape {
            kind: AdtShapeKind::Union,
            variants: vec![AdtDebugVariant {
                name: "U".to_string(),
                fields: vec![
                    AdtDebugField {
                        name: Some("a".to_string()),
                        ty: "u32".to_string(),
                    },
                    AdtDebugField {
                        name: Some("b".to_string()),
                        ty: "u16".to_string(),
                    },
                    AdtDebugField {
                        name: Some("p".to_string()),
                        ty: "*const u8".to_string(),
                    },
                ],
            }],
        },
    );
    let mut s = session(program);
    s.set_breakpoint("test.rs", 3);
    let run = s.run_script(
        vec![DebugCommand::Run, print("u"), print("u.a"), print("u.b")],
        DebugDialect::Gdb,
    );
    assert!(run.outcome.is_ok(), "outcome: {:?}", run.outcome);
    for want in [
        "$1 = test::U {a: 67344793, b: 39321, p: 0x4039999}",
        "$2 = 67344793",
        "$3 = 39321",
    ] {
        assert!(
            run.transcript.iter().any(|l| l == want),
            "missing `{want}` in {:?}",
            run.transcript
        );
    }
}
