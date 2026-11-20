//! Variant tables for coverage accounting (metric 1 and sub-metric 1b in
//! `docs/COVERAGE.md`).
//!
//! Two distinct groups of tables live here, and they must not be
//! conflated:
//!
//! 1. **The metric-1 denominator** ([`METRIC1_STATEMENT_KINDS`],
//!    [`METRIC1_TERMINATOR_KINDS`], [`METRIC1_RVALUE_KINDS`]): the variant
//!    names of the three main MIR enums of **Rust 1.100.0**, mechanically
//!    extracted from `compiler/rustc_middle/src/mir/syntax.rs` at tag
//!    `1.100.0` (see `docs/COVERAGE.md` for the extraction procedure).
//!    12 + 15 + 13 = 40 variants; [`metric1_variant_count`] is the one
//!    number that serves as the denominator of metric 1.
//!
//! 2. **The crate-model tables** (all remaining constants, listed by
//!    [`model_tables`]): the variants the `sparagmos-mir` type model
//!    itself defines. They are the bookkeeping for sub-metric 1b (each
//!    enum counted separately) and for the per-module sample tests,
//!    which construct one sample of every variant and check its
//!    `kind_name()` against these tables, so a variant cannot be added
//!    to an enum without also appearing here (or vice versa) without a
//!    test failing. They are *not* the metric-1 denominator: the model
//!    is a dialect of rustc MIR, so the sets differ — the model has
//!    `StatementKind::Retag` where rustc 1.100.0 has
//!    `BackwardIncompatibleDropHint`, names the unwind terminators
//!    `Resume` / `Terminate` where rustc has `UnwindResume` /
//!    `UnwindTerminate`, and splits `Len`, `CheckedBinaryOp`,
//!    `NullaryOp` and `ShallowInitBox` out of `Rvalue` where rustc
//!    1.100.0 folds them into other variants (and has `WrapUnsafeBinder`
//!    and `Reborrow`, which the model does not).

/// Metric-1 denominator: `StatementKind` variants of Rust 1.100.0 (12).
pub const METRIC1_STATEMENT_KINDS: &[&str] = &[
    "Assign",
    "FakeRead",
    "SetDiscriminant",
    "StorageLive",
    "StorageDead",
    "PlaceMention",
    "AscribeUserType",
    "Coverage",
    "Intrinsic",
    "ConstEvalCounter",
    "Nop",
    "BackwardIncompatibleDropHint",
];

/// Metric-1 denominator: `TerminatorKind` variants of Rust 1.100.0 (15).
pub const METRIC1_TERMINATOR_KINDS: &[&str] = &[
    "Goto",
    "SwitchInt",
    "UnwindResume",
    "UnwindTerminate",
    "Return",
    "Unreachable",
    "Drop",
    "Call",
    "TailCall",
    "Assert",
    "Yield",
    "CoroutineDrop",
    "FalseEdge",
    "FalseUnwind",
    "InlineAsm",
];

/// Metric-1 denominator: `Rvalue` variants of Rust 1.100.0 (13).
pub const METRIC1_RVALUE_KINDS: &[&str] = &[
    "Use",
    "Repeat",
    "Ref",
    "ThreadLocalRef",
    "RawPtr",
    "Cast",
    "BinaryOp",
    "UnaryOp",
    "Discriminant",
    "Aggregate",
    "CopyForDeref",
    "WrapUnsafeBinder",
    "Reborrow",
];

/// (enum name, variant count) for the three metric-1 denominator tables.
pub fn metric1_tables() -> Vec<(&'static str, usize)> {
    vec![
        ("StatementKind", METRIC1_STATEMENT_KINDS.len()),
        ("TerminatorKind", METRIC1_TERMINATOR_KINDS.len()),
        ("Rvalue", METRIC1_RVALUE_KINDS.len()),
    ]
}

/// The denominator of metric 1 (MIR variant coverage): 40.
pub fn metric1_variant_count() -> usize {
    metric1_tables().iter().map(|(_, n)| n).sum()
}

// ---------------------------------------------------------------------------
// Crate-model tables (sub-metric 1b bookkeeping). These list the variants
// the sparagmos-mir model defines; they are not the metric-1 denominator.
// ---------------------------------------------------------------------------

pub const TY_KINDS: &[&str] = &[
    "Bool",
    "Char",
    "Int",
    "Uint",
    "Float",
    "Adt",
    "Foreign",
    "Str",
    "Array",
    "Pat",
    "Slice",
    "RawPtr",
    "Ref",
    "FnDef",
    "FnPtr",
    "Dynamic",
    "Closure",
    "Coroutine",
    "CoroutineWitness",
    "Alias",
    "Param",
    "Bound",
    "Placeholder",
    "Infer",
    "Error",
    "Never",
    "Tuple",
];
pub const INT_TYS: &[&str] = &["i8", "i16", "i32", "i64", "i128", "isize"];
pub const UINT_TYS: &[&str] = &["u8", "u16", "u32", "u64", "u128", "usize"];
pub const FLOAT_TYS: &[&str] = &["f16", "f32", "f64", "f128"];
pub const PROJECTION_ELEMS: &[&str] = &[
    "Deref",
    "Field",
    "Index",
    "ConstantIndex",
    "Subslice",
    "Downcast",
    "OpaqueCast",
    "Subtype",
];
pub const OPERAND_KINDS: &[&str] = &["Copy", "Move", "Constant"];
pub const BIN_OPS: &[&str] = &[
    "Add",
    "AddWithOverflow",
    "AddUnchecked",
    "Sub",
    "SubWithOverflow",
    "SubUnchecked",
    "Mul",
    "MulWithOverflow",
    "MulUnchecked",
    "Div",
    "Rem",
    "BitXor",
    "BitAnd",
    "BitOr",
    "Shl",
    "ShlUnchecked",
    "Shr",
    "ShrUnchecked",
    "Eq",
    "Lt",
    "Le",
    "Ne",
    "Ge",
    "Gt",
    "Offset",
    "Cmp",
];
pub const UN_OPS: &[&str] = &["Not", "Neg", "PtrMetadata"];
pub const POINTER_COERCIONS: &[&str] = &[
    "ReifyFnPointer",
    "ClosureFnPointer",
    "UnsafeFnPointer",
    "Unsize",
    "MutToConstPointer",
    "ArrayToPointer",
    "DynStar",
];
pub const CAST_KINDS: &[&str] = &[
    "PointerExposeProvenance",
    "PointerWithExposedProvenance",
    "PointerCoercion",
    "IntToInt",
    "FloatToInt",
    "FloatToFloat",
    "IntToFloat",
    "PtrToPtr",
    "FnPtrToPtr",
    "Transmute",
    "Subtype",
];
pub const AGGREGATE_KINDS: &[&str] = &[
    "Array",
    "Tuple",
    "Adt",
    "Closure",
    "Coroutine",
    "CoroutineClosure",
    "RawPtr",
];
pub const NULLARY_OPS: &[&str] = &["SizeOf", "AlignOf", "OffsetOf", "UbChecks"];
pub const RVALUE_KINDS: &[&str] = &[
    "Use",
    "Repeat",
    "Ref",
    "ThreadLocalRef",
    "RawPtr",
    "Len",
    "Cast",
    "BinaryOp",
    "CheckedBinaryOp",
    "NullaryOp",
    "UnaryOp",
    "Discriminant",
    "Aggregate",
    "ShallowInitBox",
    "CopyForDeref",
];
pub const STATEMENT_KINDS: &[&str] = &[
    "Assign",
    "FakeRead",
    "SetDiscriminant",
    "StorageLive",
    "StorageDead",
    "Retag",
    "PlaceMention",
    "AscribeUserType",
    "Coverage",
    "Intrinsic",
    "ConstEvalCounter",
    "Nop",
];
pub const TERMINATOR_KINDS: &[&str] = &[
    "Goto",
    "SwitchInt",
    "Resume",
    "Terminate",
    "Return",
    "Unreachable",
    "Drop",
    "Call",
    "Assert",
    "Yield",
    "CoroutineDrop",
    "FalseEdge",
    "FalseUnwind",
    "InlineAsm",
    "TailCall",
];

/// (table name, variant count) for every crate-model table. These are the
/// sub-metric 1b bookkeeping tables — not the metric-1 denominator (see
/// [`metric1_tables`]).
pub fn model_tables() -> Vec<(&'static str, usize)> {
    vec![
        ("Ty", TY_KINDS.len()),
        ("IntTy", INT_TYS.len()),
        ("UintTy", UINT_TYS.len()),
        ("FloatTy", FLOAT_TYS.len()),
        ("ProjectionElem", PROJECTION_ELEMS.len()),
        ("Operand", OPERAND_KINDS.len()),
        ("BinOp", BIN_OPS.len()),
        ("UnOp", UN_OPS.len()),
        ("PointerCoercion", POINTER_COERCIONS.len()),
        ("CastKind", CAST_KINDS.len()),
        ("AggregateKind", AGGREGATE_KINDS.len()),
        ("NullaryOp", NULLARY_OPS.len()),
        ("Rvalue", RVALUE_KINDS.len()),
        ("StatementKind", STATEMENT_KINDS.len()),
        ("TerminatorKind", TERMINATOR_KINDS.len()),
    ]
}

/// Total number of variants across the crate-model tables: what the
/// `sparagmos-mir` model defines. This is not a coverage denominator —
/// the metric-1 denominator is [`metric1_variant_count`] (40).
pub fn model_variant_total() -> usize {
    model_tables().iter().map(|(_, n)| n).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_unique(name: &str, table: &[&str]) {
        assert!(!table.is_empty(), "{name} empty");
        let mut sorted = table.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), table.len(), "{name} has duplicates");
    }

    #[test]
    fn tables_are_nonempty_and_unique() {
        for (name, table) in [
            ("TY_KINDS", TY_KINDS),
            ("BIN_OPS", BIN_OPS),
            ("RVALUE_KINDS", RVALUE_KINDS),
            ("STATEMENT_KINDS", STATEMENT_KINDS),
            ("TERMINATOR_KINDS", TERMINATOR_KINDS),
            ("METRIC1_STATEMENT_KINDS", METRIC1_STATEMENT_KINDS),
            ("METRIC1_TERMINATOR_KINDS", METRIC1_TERMINATOR_KINDS),
            ("METRIC1_RVALUE_KINDS", METRIC1_RVALUE_KINDS),
        ] {
            assert_unique(name, table);
        }
    }

    #[test]
    fn metric1_denominator_is_40() {
        assert_eq!(METRIC1_STATEMENT_KINDS.len(), 12);
        assert_eq!(METRIC1_TERMINATOR_KINDS.len(), 15);
        assert_eq!(METRIC1_RVALUE_KINDS.len(), 13);
        assert_eq!(metric1_variant_count(), 40);
    }

    #[test]
    fn model_total_is_sum_of_model_tables() {
        assert_eq!(
            model_variant_total(),
            model_tables().iter().map(|(_, n)| n).sum::<usize>()
        );
    }

    /// The crate model is a dialect of rustc MIR: where the two table
    /// groups differ, they differ on purpose. Pin the known differences
    /// so the metric-1 denominator and the model tables cannot silently
    /// drift into being treated as the same set.
    #[test]
    fn metric1_and_model_tables_differ_where_the_dialect_does() {
        // StatementKind: model has Retag, rustc 1.100.0 has
        // BackwardIncompatibleDropHint instead.
        assert!(STATEMENT_KINDS.contains(&"Retag"));
        assert!(!METRIC1_STATEMENT_KINDS.contains(&"Retag"));
        assert!(METRIC1_STATEMENT_KINDS.contains(&"BackwardIncompatibleDropHint"));
        assert!(!STATEMENT_KINDS.contains(&"BackwardIncompatibleDropHint"));
        // TerminatorKind: model names the unwind terminators Resume /
        // Terminate, rustc 1.100.0 calls them UnwindResume /
        // UnwindTerminate.
        assert!(TERMINATOR_KINDS.contains(&"Resume"));
        assert!(TERMINATOR_KINDS.contains(&"Terminate"));
        assert!(METRIC1_TERMINATOR_KINDS.contains(&"UnwindResume"));
        assert!(METRIC1_TERMINATOR_KINDS.contains(&"UnwindTerminate"));
        // Rvalue: the model splits out Len / CheckedBinaryOp / NullaryOp /
        // ShallowInitBox (15 kinds); rustc 1.100.0 has WrapUnsafeBinder
        // and Reborrow instead (13 kinds).
        assert_eq!(RVALUE_KINDS.len(), 15);
        for name in ["Len", "CheckedBinaryOp", "NullaryOp", "ShallowInitBox"] {
            assert!(RVALUE_KINDS.contains(&name), "model Rvalue has {name}");
            assert!(
                !METRIC1_RVALUE_KINDS.contains(&name),
                "metric-1 Rvalue folds {name} into other variants"
            );
        }
        for name in ["WrapUnsafeBinder", "Reborrow"] {
            assert!(METRIC1_RVALUE_KINDS.contains(&name));
            assert!(!RVALUE_KINDS.contains(&name));
        }
    }
}
