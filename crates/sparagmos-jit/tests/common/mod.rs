//! Builders shared by the dual-engine JIT test suites.

use std::collections::HashMap;

use sparagmos_mir as mir;

/// The interpreter-side program for a dual-engine test: the given
/// functions under the given name table, with the shared ADT
/// discriminant table recorded, and every other program table at
/// its default.
///
/// Built through `MirProgram::new()` plus field assignment, never
/// an exhaustive struct literal: when `MirProgram` grows a field,
/// this helper is the one place that decides its test-side value
/// (the default, unless a suite passes it in), and the test sites
/// stay untouched.
pub fn interp_program(
    functions: Vec<sparagmos_interp::MirFunction>,
    fn_names: HashMap<mir::DefId, String>,
    adt_discriminants: mir::AdtDiscriminants,
) -> sparagmos_interp::MirProgram {
    let mut program = sparagmos_interp::MirProgram::new();
    program.functions = functions;
    program.fn_names = fn_names;
    program.adt_discriminants = adt_discriminants;
    program
}
