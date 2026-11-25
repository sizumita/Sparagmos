//! Coverage mapping data for one program, captured from the
//! compiler's coverage instrumentation (the data rustc's codegen
//! turns into the LLVM coverage mapping sections; see
//! `rustc_middle::mir::coverage` and the `coverageinfo::mapgen`
//! code that consumes it).
//!
//! The driver records this table when the crate was compiled with
//! `-C instrument-coverage`; without instrumentation there is no
//! coverage data and the table is empty. The interpreter counts
//! physical-counter hits while executing the program's
//! [`StatementKind::Coverage`](crate::StatementKind::Coverage)
//! statements, and the counts together with this table reproduce
//! both coverage outputs compiletest judges: the coverage-mapping
//! dump (`coverage-dump`) and the `llvm-cov` text report. Nothing
//! in the table is estimated: mappings, counter terms and
//! expressions are the compiler's own, transcribed verbatim.

/// A coverage counter term: how the execution count of the basic
/// coverage block (BCB) it is attached to is derived from the
/// program's physical counters at report time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CovTerm {
    /// The constant zero (the BCB is never counted, or its counter
    /// was optimised away).
    Zero,
    /// The value of physical counter `n`.
    Counter(u32),
    /// The value of counter expression `n` (see
    /// [`CovFunction::expressions`]); the expression's operator is
    /// part of the expression, not of the term.
    Expression(u32),
}

/// One counter expression of a function: `lhs op rhs`, where `op`
/// is subtraction when `subtract` is set and addition otherwise.
/// Operands are terms; an operand that names another expression
/// carries that expression's own operator when rendered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CovExpression {
    pub lhs: CovTerm,
    pub subtract: bool,
    pub rhs: CovTerm,
}

/// What a coverage mapping attaches its counter(s) to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CovMappingKind {
    /// A code region counted by the term of `bcb`.
    Code { bcb: u32 },
    /// A branch region counted by the terms of the true and
    /// false BCBs.
    Branch { true_bcb: u32, false_bcb: u32 },
}

/// One coverage mapping: a source region and the counter(s) that
/// count its executions. Line and column numbers are 1-based;
/// columns are byte columns, as the LLVM coverage mapping format
/// stores them. `file` is a global file id into
/// [`CoverageMapping::files`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CovMapping {
    pub kind: CovMappingKind,
    pub file: u32,
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

/// The coverage record of one function instance (one entry of
/// the coverage mapping's function table).
#[derive(Clone, Debug, PartialEq)]
pub struct CovFunction {
    /// The instance's mangled symbol name: the record's identity
    /// and the sort key the global file table is built in.
    pub symbol: String,
    /// The demangled name, as the coverage outputs print it.
    pub name: String,
    /// Whether the function participates in codegen. An unused
    /// function's mappings all count as zero (its record exists
    /// so the function still appears in reports).
    pub is_used: bool,
    /// Whether the record lives only in the linked binary's
    /// coverage mapping, not in this crate's own mapping dump:
    /// a foreign instance that was not monomorphised in this
    /// crate (its record was emitted by its home crate's
    /// codegen, and merged into the executable at link time).
    /// The coverage-run report includes such records; the
    /// coverage-map dump of this crate's IR does not.
    pub binary_only: bool,
    /// The program keys of the converted bodies whose execution
    /// feeds this record's counters (an instance can be
    /// converted under more than one key; executing any of them
    /// counts for this record). Empty for unused functions and
    /// for instances the conversion did not produce (their
    /// counters stay zero).
    pub body_names: Vec<String>,
    /// The function's counter expressions, in the compiler's
    /// order (the order the mapping format encodes them, before
    /// the writer's minimisation).
    pub expressions: Vec<CovExpression>,
    /// The function's mappings, in the compiler's order.
    pub mappings: Vec<CovMapping>,
    /// Per-BCB counter term (index = BCB), as the compiler's
    /// codegen info assigned it; `None` where no term was
    /// assigned (such a BCB is not referred to by any mapping).
    pub term_for_bcb: Vec<Option<CovTerm>>,
    /// Per-BCB physical counter (index = BCB): the counter the
    /// BCB's coverage statement increments when executed.
    pub counter_for_bcb: Vec<Option<u32>>,
    /// The number of physical counters the function was assigned.
    pub num_counters: u32,
}

impl CovFunction {
    /// The counter term a mapping BCB counts with in this record:
    /// unused functions count everything as zero, as the native
    /// codegen emits for them.
    pub fn term_of(&self, bcb: u32) -> CovTerm {
        if !self.is_used {
            return CovTerm::Zero;
        }
        self.term_for_bcb
            .get(bcb as usize)
            .copied()
            .flatten()
            .unwrap_or(CovTerm::Zero)
    }

    /// Evaluate a counter term against this function's physical
    /// counter values (indexed by counter id). Expression values
    /// saturate at zero for subtraction, as LLVM's counter
    /// evaluation does.
    pub fn eval_term(&self, term: CovTerm, counters: &[u64]) -> u64 {
        match term {
            CovTerm::Zero => 0,
            CovTerm::Counter(id) => counters.get(id as usize).copied().unwrap_or(0),
            CovTerm::Expression(id) => {
                let Some(expr) = self.expressions.get(id as usize) else {
                    return 0;
                };
                let lhs = self.eval_term(expr.lhs, counters);
                let rhs = self.eval_term(expr.rhs, counters);
                if expr.subtract {
                    lhs.saturating_sub(rhs)
                } else {
                    lhs.saturating_add(rhs)
                }
            }
        }
    }

    /// The execution count of one BCB under the given physical
    /// counter values.
    pub fn bcb_count(&self, bcb: u32, counters: &[u64]) -> u64 {
        self.eval_term(self.term_of(bcb), counters)
    }
}

/// The coverage mapping table of one program.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoverageMapping {
    /// The global file table, in the order the native codegen
    /// builds it: entry 0 is the compilation working directory
    /// (the format's base directory), the remaining entries are
    /// the source files the functions' mappings refer to, in
    /// first-appearance order over the records sorted by symbol
    /// name. A mapping's `file` field indexes this table.
    pub files: Vec<String>,
    /// One record per function instance that carries coverage
    /// data, in symbol-name order.
    pub functions: Vec<CovFunction>,
}

impl CoverageMapping {
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// The record whose counters the named body feeds, if any.
    pub fn record_for_body(&self, body_name: &str) -> Option<usize> {
        self.functions
            .iter()
            .position(|f| f.body_names.iter().any(|n| n == body_name))
    }
}
