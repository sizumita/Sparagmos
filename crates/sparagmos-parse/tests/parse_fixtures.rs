//! Fixture-based tests: every `tests/fixtures/*.mir` file is real output of
//! `rustup run 1.100.0 rustc --edition 2021 --emit=mir` on the sibling `.rs`
//! source (binop also used `-C overflow-checks=off`). All of them must parse
//! without error; specific bodies are checked in detail below.

use sparagmos_parse::ast::*;
use sparagmos_parse::parse_mir;
use std::path::PathBuf;

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

fn all_fixtures_parse() -> Vec<(String, MirFile)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "mir").unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.path());
    assert!(!entries.is_empty(), "no .mir fixtures found");
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(entry.path()).unwrap();
        let parsed =
            parse_mir(&src).unwrap_or_else(|e| panic!("fixture {name} failed to parse: {e}"));
        assert!(!parsed.bodies.is_empty(), "fixture {name} has no bodies");
        out.push((name, parsed));
    }
    out
}

#[test]
fn every_fixture_parses() {
    let files = all_fixtures_parse();
    assert_eq!(
        files.len(),
        4,
        "expected 4 fixtures, got: {:?}",
        files.iter().map(|f| &f.0).collect::<Vec<_>>()
    );
}

#[test]
fn const_return_fixture_details() {
    let file = parse_mir(&fixture("const_return.mir")).unwrap();
    let answer = file
        .bodies
        .iter()
        .find(|b| b.name == "answer")
        .expect("answer body");
    assert_eq!(answer.return_ty, Ty::Int("i32".into()));
    match &answer.blocks[0].statements[0] {
        Statement::Assign(_, Rvalue::Use(Operand::Constant(Const::Int { value: 1, .. }))) => {}
        other => panic!("unexpected: {other:?}"),
    }
    // main must contain a Call terminator to `answer`
    let main = file
        .bodies
        .iter()
        .find(|b| b.name == "main")
        .expect("main body");
    assert!(main.blocks.iter().any(|b| matches!(&b.terminator,
        Terminator::Call { func: Operand::Constant(Const::Named(n)), .. } if n == "answer")));
}

#[test]
fn binop_fixture_details() {
    let file = parse_mir(&fixture("binop.mir")).unwrap();
    let add = file
        .bodies
        .iter()
        .find(|b| b.name == "add")
        .expect("add body");
    assert!(matches!(
        &add.blocks[0].statements[0],
        Statement::Assign(_, Rvalue::BinaryOp(BinOp::Add, _, _))
    ));
    let calc = file
        .bodies
        .iter()
        .find(|b| b.name == "calc")
        .expect("calc body");
    let ops: Vec<&BinOp> = calc
        .blocks
        .iter()
        .flat_map(|b| &b.statements)
        .filter_map(|s| match s {
            Statement::Assign(_, Rvalue::BinaryOp(op, _, _)) => Some(op),
            _ => None,
        })
        .collect();
    assert!(ops.contains(&&BinOp::Sub) && ops.contains(&&BinOp::Mul) && ops.contains(&&BinOp::Div));
    // calc contains assert terminators generated for division checks
    assert!(calc
        .blocks
        .iter()
        .any(|b| matches!(b.terminator, Terminator::Assert { .. })));
    // AddWithOverflow-free build still has named const i32::MIN inside an Eq
    assert!(calc
        .blocks
        .iter()
        .flat_map(|b| &b.statements)
        .any(|s| matches!(s,
        Statement::Assign(_, Rvalue::BinaryOp(BinOp::Eq, _, rhs))
            if matches!(&**rhs, Operand::Constant(Const::Named(n)) if n == "i32::MIN"))));
}

#[test]
fn switch_fixture_details() {
    let file = parse_mir(&fixture("switch.mir")).unwrap();
    let pick = file
        .bodies
        .iter()
        .find(|b| b.name == "pick")
        .expect("pick body");
    let switch = pick
        .blocks
        .iter()
        .find_map(|b| match &b.terminator {
            Terminator::SwitchInt {
                targets, otherwise, ..
            } => Some((targets.clone(), *otherwise)),
            _ => None,
        })
        .expect("switchInt terminator");
    assert_eq!(switch.0.len(), 1);
    assert_eq!(switch.0[0].0, 0);
    assert!(switch.1.is_some());
    // pick branches rejoin with goto
    assert!(pick
        .blocks
        .iter()
        .any(|b| matches!(b.terminator, Terminator::Goto { .. })));
    // Gt comparison appears as a binary op
    assert!(pick
        .blocks
        .iter()
        .flat_map(|b| &b.statements)
        .any(|s| matches!(s, Statement::Assign(_, Rvalue::BinaryOp(BinOp::Gt, _, _)))));
}

#[test]
fn copy_move_fixture_details() {
    let file = parse_mir(&fixture("copy_move.mir")).unwrap();
    let neg = file
        .bodies
        .iter()
        .find(|b| b.name == "neg")
        .expect("neg body");
    assert!(neg
        .blocks
        .iter()
        .flat_map(|b| &b.statements)
        .any(|s| matches!(s, Statement::Assign(_, Rvalue::UnaryOp(UnOp::Neg, _)))));
    let copy_demo = file
        .bodies
        .iter()
        .find(|b| b.name == "copy_demo")
        .expect("copy_demo body");
    assert!(matches!(
        &copy_demo.blocks[0].statements[0],
        Statement::Assign(_, Rvalue::Use(Operand::Copy(_)))
    ));
    // Vec<i32> argument type is preserved as a Named type with generics
    let move_demo = file
        .bodies
        .iter()
        .find(|b| b.name == "move_demo")
        .expect("move_demo body");
    assert!(matches!(&move_demo.args[0].ty, Ty::Named(n) if n.contains("Vec")));
    // debug vars collected, including from scopes
    let main = file
        .bodies
        .iter()
        .find(|b| b.name == "main")
        .expect("main body");
    assert!(main.debug_vars.iter().any(|d| d.name == "_a"));
    assert!(main.locals.iter().any(|l| l.local == Local(2)));
}

#[test]
fn checked_add_fixture_from_source_text() {
    // AddWithOverflow form (default overflow checks on) as an inline fixture,
    // matching the shape rustc 1.100.0 emits for `a + b`.
    let src = "fn add(_1: i32, _2: i32) -> i32 {\n    let mut _0: i32;\n    let mut _3: (i32, bool);\n    bb0: {\n        _3 = AddWithOverflow(copy _1, copy _2);\n        assert(!move (_3.1: bool), \"overflow\", copy _1, copy _2) -> [success: bb1, unwind continue];\n    }\n    bb1: {\n        _0 = move (_3.0: i32);\n        return;\n    }\n}\n";
    let file = parse_mir(src).unwrap();
    let add = &file.bodies[0];
    assert!(matches!(
        &add.blocks[0].statements[0],
        Statement::Assign(_, Rvalue::BinaryOp(BinOp::AddWithOverflow, _, _))
    ));
    assert!(matches!(
        add.blocks[0].terminator,
        Terminator::Assert { .. }
    ));
    // typed field projection `(_3.0: i32)` as a move operand
    match &add.blocks[1].statements[0] {
        Statement::Assign(_, Rvalue::Use(Operand::Move(place))) => {
            assert_eq!(place.local, Local(3));
            assert!(matches!(
                place.projections[0],
                Projection::Field {
                    index: 0,
                    ty: Some(Ty::Int(_))
                }
            ));
        }
        other => panic!("unexpected: {other:?}"),
    }
}
