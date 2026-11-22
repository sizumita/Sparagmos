//! Dump format v0 for converted MIR bodies.
//!
//! A plain-text, line-oriented format owned by this crate:
//!
//! ```text
//! sparagmos-mir-dump v0
//! source: <input file>
//! functions: <count>
//!
//! === fn <name> (def=DefId(0:<n>), args=<n>, blocks=<n>, locals=<n>)
//! <Debug representation of sparagmos_mir::Body>
//! ```
//!
//! The `def=` field is the function's own definition id - the
//! same id the in-process JIT input carries - because a name is
//! not an identity: several collected bodies can share one name
//! (instances of one generic item, an impl forwarder collected
//! under its trait's name), and a reader that guesses the id
//! from the name can hand the JIT a different function than the
//! driver collected.
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
    body_def_ids: &[sparagmos_mir::DefId],
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
    for ((name, body), def_id) in bodies.iter().zip(body_def_ids.iter()) {
        let _ = writeln!(
            out,
            "\n=== fn {name} (def=DefId({}:{}), args={}, blocks={}, locals={})",
            def_id.krate,
            def_id.index,
            body.arg_count,
            body.basic_blocks.len(),
            body.local_decls.len()
        );
        let _ = writeln!(out, "{body:#?}");
    }
    let skipped: Vec<_> = report
        .functions
        .iter()
        .filter(|f| f.is_skipped())
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
    let not_demanded: Vec<_> = report
        .functions
        .iter()
        .filter(|f| matches!(f.outcome, FnOutcome::NotDemanded { .. }))
        .collect();
    if !not_demanded.is_empty() {
        let _ = writeln!(out, "\n=== not demanded");
        for f in not_demanded {
            if let FnOutcome::NotDemanded { reason } = &f.outcome {
                let _ = writeln!(out, "{}: {}", f.name, reason);
            }
        }
    }
    out
}
