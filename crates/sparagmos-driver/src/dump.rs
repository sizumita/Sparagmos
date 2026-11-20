//! Dump format v0 for converted MIR bodies.
//!
//! A plain-text, line-oriented format owned by this crate:
//!
//! ```text
//! sparagmos-mir-dump v0
//! source: <input file>
//! functions: <count>
//!
//! === fn <name> (args=<n>, blocks=<n>, locals=<n>)
//! <Debug representation of sparagmos_mir::Body>
//! ```
//!
//! Only successfully converted functions are dumped; skipped functions are
//! listed in a trailing `=== skipped` section with their reasons. The
//! harness is expected to read this file in a later lane.

use std::fmt::Write as _;

use crate::report::{ConversionReport, FnOutcome};
use sparagmos_mir::Body;

pub fn render(
    source: &str,
    report: &ConversionReport,
    bodies: &[(String, Body)],
    def_names: &[String],
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "sparagmos-mir-dump v0");
    let _ = writeln!(out, "source: {source}");
    let _ = writeln!(out, "functions: {}", bodies.len());
    if !def_names.is_empty() {
        let _ = writeln!(out, "def-names:");
        for (i, n) in def_names.iter().enumerate() {
            let _ = writeln!(out, "  DefId(0:{i}) = {n}");
        }
    }
    for (name, body) in bodies {
        let _ = writeln!(
            out,
            "\n=== fn {name} (args={}, blocks={}, locals={})",
            body.arg_count,
            body.basic_blocks.len(),
            body.local_decls.len()
        );
        let _ = writeln!(out, "{body:#?}");
    }
    let skipped: Vec<_> = report
        .functions
        .iter()
        .filter(|f| !f.is_converted())
        .collect();
    if !skipped.is_empty() {
        let _ = writeln!(out, "\n=== skipped");
        for f in skipped {
            if let FnOutcome::Skipped { reasons } = &f.outcome {
                let rs: Vec<String> =
                    reasons.iter().map(|(k, v)| format!("{k} x{v}")).collect();
                let _ = writeln!(out, "{}: {}", f.name, rs.join(", "));
            }
        }
    }
    out
}
