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
    /// A generic definition no converted body demands an instance
    /// of: it has no monomorphic instance in this program, the way
    /// rustc's monomorphization collector never instantiates it
    /// either. The definition is not part of the program, so its
    /// absence does not make the conversion incomplete — but it
    /// is recorded, not dropped: `reason` is the item loop's
    /// original no-instance record. A definition that *is*
    /// demanded and whose instances cannot be converted stays
    /// `Skipped` with the instances' reasons.
    NotDemanded { reason: String },
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
    pub fn is_skipped(&self) -> bool {
        matches!(self.outcome, FnOutcome::Skipped { .. })
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConversionReport {
    pub functions: Vec<FnReport>,
    /// Definitions counted as converted that have no body: an
    /// un-demanded data constructor (accounted converted in
    /// `apply_instance_demands`, because its uses are aggregate
    /// rvalues in already-converted callers) and a trivial constant
    /// (accounted by `account_trivial_const`, because its value is
    /// folded into every use site). Neither has a body to convert;
    /// the count is reported separately so the converted total can
    /// be split into with-body and without-body parts.
    pub bodyless_converted: usize,
}

impl ConversionReport {
    pub fn converted_count(&self) -> usize {
        self.functions.iter().filter(|f| f.is_converted()).count()
    }
    pub fn skipped_count(&self) -> usize {
        self.functions.iter().filter(|f| f.is_skipped()).count()
    }
    pub fn not_demanded_count(&self) -> usize {
        self.functions
            .iter()
            .filter(|f| matches!(f.outcome, FnOutcome::NotDemanded { .. }))
            .count()
    }
    /// Definitions counted as converted without a body (see the
    /// field's documentation).
    pub fn bodyless_converted_count(&self) -> usize {
        self.bodyless_converted
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
