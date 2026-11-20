//! Conversion accounting: what was converted, what was skipped, and which
//! unsupported variants were hit. Nothing is dropped silently: every
//! unsupported construct is counted here, per function.

use std::collections::BTreeMap;

/// Outcome for a single local item.
#[derive(Clone, Debug)]
pub enum FnOutcome {
    /// Successfully converted to a `sparagmos_mir::Body`.
    Converted,
    /// Skipped. `reasons` maps a human-readable reason (usually the
    /// unsupported variant name) to the number of occurrences.
    Skipped { reasons: BTreeMap<String, usize> },
}

#[derive(Clone, Debug)]
pub struct FnReport {
    pub name: String,
    /// Number of basic blocks in the Stable MIR body (0 if no body).
    pub blocks: usize,
    /// Number of locals in the Stable MIR body (0 if no body).
    pub locals: usize,
    pub arg_count: usize,
    pub outcome: FnOutcome,
}

impl FnReport {
    pub fn is_converted(&self) -> bool {
        matches!(self.outcome, FnOutcome::Converted)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConversionReport {
    pub functions: Vec<FnReport>,
}

impl ConversionReport {
    pub fn converted_count(&self) -> usize {
        self.functions.iter().filter(|f| f.is_converted()).count()
    }
    pub fn skipped_count(&self) -> usize {
        self.functions.len() - self.converted_count()
    }
    /// Unsupported-variant totals across all skipped functions.
    pub fn unsupported_totals(&self) -> BTreeMap<String, usize> {
        let mut totals: BTreeMap<String, usize> = BTreeMap::new();
        for f in &self.functions {
            if let FnOutcome::Skipped { reasons } = &f.outcome {
                for (k, v) in reasons {
                    *totals.entry(k.clone()).or_default() += v;
                }
            }
        }
        totals
    }
}
