//! Coverage counting: the interpreter counts a coverage
//! statement against the record of the instance the statement
//! belongs to -- for a statement inlined from another instance,
//! that instance's record, not the executing body's (the way
//! codegen emits the counter increment into the owner's
//! counters). A statement attributed to an instance with no
//! coverage record counts nothing: no counter was emitted for
//! it natively either.

use sparagmos_interp::mir_convert::{interpret_mir_captured, MirProgram};
use sparagmos_mir as mir;

fn unit_ty() -> mir::Ty {
    mir::Ty::Tuple(Vec::new())
}

fn decls(tys: &[mir::Ty]) -> Vec<mir::LocalDecl> {
    tys.iter().cloned().map(mir::LocalDecl::new).collect()
}

fn block(stmts: Vec<mir::Statement>, kind: mir::TerminatorKind) -> mir::BasicBlockData {
    let mut bb = mir::BasicBlockData::new(mir::Terminator::new(kind));
    bb.statements = stmts;
    bb
}

fn coverage_stmt(bcb: u32) -> mir::Statement {
    mir::Statement::new(mir::StatementKind::Coverage(bcb))
}

fn record(
    symbol: &str,
    name: &str,
    body_names: &[&str],
) -> mir::CovFunction {
    mir::CovFunction {
        symbol: symbol.to_string(),
        name: name.to_string(),
        is_used: true,
        binary_only: false,
        body_names: body_names.iter().map(|s| s.to_string()).collect(),
        expressions: Vec::new(),
        mappings: vec![mir::CovMapping {
            kind: mir::CovMappingKind::Code { bcb: 0 },
            file: 0,
            start_line: 1,
            start_col: 1,
            end_line: 1,
            end_col: 10,
        }],
        term_for_bcb: vec![Some(mir::CovTerm::Counter(0))],
        counter_for_bcb: vec![Some(0)],
        num_counters: 1,
    }
}

fn table(functions: Vec<mir::CovFunction>) -> mir::CoverageMapping {
    mir::CoverageMapping {
        files: vec!["test.rs".to_string()],
        functions,
    }
}

/// `fn main() { <cov 0: own> <cov 0: inlined from callee> }`
/// (an inlined statement carries the owner's BCB numbering, so
/// both statements name BCB 0 -- of different functions).
fn main_body() -> mir::Body {
    let mut body = mir::Body::new(
        decls(&[unit_ty()]),
        0,
        vec![block(
            vec![coverage_stmt(0), coverage_stmt(0)],
            mir::TerminatorKind::Return,
        )],
    );
    body.coverage_owners = vec![None, Some("callee_sym".to_string())];
    body
}

#[test]
fn inlined_statement_counts_against_its_owner() {
    let mut prog = MirProgram::new();
    prog.add("main", main_body());
    prog.coverage = table(vec![
        record("main_sym", "test::main", &["main"]),
        record("callee_sym", "test::callee", &[]),
    ]);

    let run = interpret_mir_captured(&prog, "main", vec![]);
    assert!(run.result.is_ok(), "main runs: {:?}", run.result);
    // The first statement feeds main's record; the inlined
    // statement feeds the callee's record, even though the
    // callee has no converted body of its own.
    assert_eq!(run.coverage_counts, vec![vec![1], vec![1]]);
}

#[test]
fn statement_for_a_recordless_owner_counts_nothing() {
    let mut body = mir::Body::new(
        decls(&[unit_ty()]),
        0,
        vec![block(
            vec![coverage_stmt(0)],
            mir::TerminatorKind::Return,
        )],
    );
    body.coverage_owners = vec![Some("missing_sym".to_string())];
    let mut prog = MirProgram::new();
    prog.add("main", body);
    prog.coverage = table(vec![record("main_sym", "test::main", &["main"])]);

    let run = interpret_mir_captured(&prog, "main", vec![]);
    assert!(run.result.is_ok(), "main runs: {:?}", run.result);
    assert_eq!(run.coverage_counts, vec![vec![0]]);
}

#[test]
fn expression_terms_evaluate_with_saturating_subtraction() {
    let mut f = record("s", "test::f", &[]);
    f.expressions = vec![mir::CovExpression {
        lhs: mir::CovTerm::Counter(0),
        subtract: true,
        rhs: mir::CovTerm::Counter(1),
    }];
    let expr = mir::CovTerm::Expression(0);
    assert_eq!(f.eval_term(expr, &[5, 3]), 2);
    assert_eq!(f.eval_term(expr, &[1, 3]), 0, "subtraction saturates");
    // An unused function's terms all read as zero.
    f.is_used = false;
    assert_eq!(f.term_of(0), mir::CovTerm::Zero);
}
