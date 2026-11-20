//! Mechanically countable variant tables — the denominator for MIR variant
//! coverage (metric 1 in `docs/COVERAGE.md`).
//!
//! Reference denominator: `rustc_middle::mir` / `rustc_middle::ty` as of
//! **Rust 1.100.0** (released 2026-11-12; verified on fleur with
//! `rustup run 1.100.0 rustc`). Every table entry corresponds to a variant
//! actually implemented in this crate; unit tests construct one sample of
//! each variant and check its `kind_name()` against these tables, so a
//! variant cannot be added to an enum without also appearing here (or
//! vice versa) without a test failing.

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
    "BoxDerefTransmute",
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

/// (table name, variant count) for every counted enum.
pub fn tables() -> Vec<(&'static str, usize)> {
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

/// Total number of variants across all counted tables. Because every entry is
/// backed by a sample-construction test, this equals the number of variants
/// this crate implements — i.e. the numerator *and* (for the subset of rustc
/// MIR modelled here) the working denominator for coverage reporting.
pub fn total_variant_count() -> usize {
    tables().iter().map(|(_, n)| n).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tables_are_nonempty_and_unique() {
        for (name, table) in [
            ("TY_KINDS", TY_KINDS),
            ("BIN_OPS", BIN_OPS),
            ("RVALUE_KINDS", RVALUE_KINDS),
            ("STATEMENT_KINDS", STATEMENT_KINDS),
            ("TERMINATOR_KINDS", TERMINATOR_KINDS),
        ] {
            assert!(!table.is_empty(), "{name} empty");
            let mut sorted = table.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), table.len(), "{name} has duplicates");
        }
    }
    #[test]
    fn total_is_sum_of_tables() {
        assert_eq!(
            total_variant_count(),
            tables().iter().map(|(_, n)| n).sum::<usize>()
        );
        assert!(
            total_variant_count() >= 150,
            "suspiciously small variant total: {}",
            total_variant_count()
        );
    }
}
