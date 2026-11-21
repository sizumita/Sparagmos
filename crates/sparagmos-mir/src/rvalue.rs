//! `Rvalue` and its operator/kind enums — full variant sets of rustc MIR (1.100.0 reference).
use crate::operand::Operand;
use crate::place::Place;
use crate::ty::{DefId, Mutability, Region, Ty};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    AddWithOverflow,
    AddUnchecked,
    Sub,
    SubWithOverflow,
    SubUnchecked,
    Mul,
    MulWithOverflow,
    MulUnchecked,
    Div,
    Rem,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    ShlUnchecked,
    Shr,
    ShrUnchecked,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
    Offset,
    Cmp,
}
impl BinOp {
    pub const ALL: &[BinOp] = &[
        Self::Add,
        Self::AddWithOverflow,
        Self::AddUnchecked,
        Self::Sub,
        Self::SubWithOverflow,
        Self::SubUnchecked,
        Self::Mul,
        Self::MulWithOverflow,
        Self::MulUnchecked,
        Self::Div,
        Self::Rem,
        Self::BitXor,
        Self::BitAnd,
        Self::BitOr,
        Self::Shl,
        Self::ShlUnchecked,
        Self::Shr,
        Self::ShrUnchecked,
        Self::Eq,
        Self::Lt,
        Self::Le,
        Self::Ne,
        Self::Ge,
        Self::Gt,
        Self::Offset,
        Self::Cmp,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::AddWithOverflow => "AddWithOverflow",
            Self::AddUnchecked => "AddUnchecked",
            Self::Sub => "Sub",
            Self::SubWithOverflow => "SubWithOverflow",
            Self::SubUnchecked => "SubUnchecked",
            Self::Mul => "Mul",
            Self::MulWithOverflow => "MulWithOverflow",
            Self::MulUnchecked => "MulUnchecked",
            Self::Div => "Div",
            Self::Rem => "Rem",
            Self::BitXor => "BitXor",
            Self::BitAnd => "BitAnd",
            Self::BitOr => "BitOr",
            Self::Shl => "Shl",
            Self::ShlUnchecked => "ShlUnchecked",
            Self::Shr => "Shr",
            Self::ShrUnchecked => "ShrUnchecked",
            Self::Eq => "Eq",
            Self::Lt => "Lt",
            Self::Le => "Le",
            Self::Ne => "Ne",
            Self::Ge => "Ge",
            Self::Gt => "Gt",
            Self::Offset => "Offset",
            Self::Cmp => "Cmp",
        }
    }
    pub fn is_checkable(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Sub | Self::Mul | Self::Shl | Self::Shr
        )
    }
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Eq | Self::Lt | Self::Le | Self::Ne | Self::Ge | Self::Gt | Self::Cmp
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Not,
    Neg,
    PtrMetadata,
}
impl UnOp {
    pub const ALL: &[UnOp] = &[Self::Not, Self::Neg, Self::PtrMetadata];
    pub fn name(self) -> &'static str {
        match self {
            Self::Not => "Not",
            Self::Neg => "Neg",
            Self::PtrMetadata => "PtrMetadata",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointerCoercion {
    ReifyFnPointer,
    ClosureFnPointer,
    UnsafeFnPointer,
    Unsize,
    MutToConstPointer,
    ArrayToPointer,
    DynStar,
}
impl PointerCoercion {
    pub const ALL: &[PointerCoercion] = &[
        Self::ReifyFnPointer,
        Self::ClosureFnPointer,
        Self::UnsafeFnPointer,
        Self::Unsize,
        Self::MutToConstPointer,
        Self::ArrayToPointer,
        Self::DynStar,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::ReifyFnPointer => "ReifyFnPointer",
            Self::ClosureFnPointer => "ClosureFnPointer",
            Self::UnsafeFnPointer => "UnsafeFnPointer",
            Self::Unsize => "Unsize",
            Self::MutToConstPointer => "MutToConstPointer",
            Self::ArrayToPointer => "ArrayToPointer",
            Self::DynStar => "DynStar",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CastKind {
    PointerExposeProvenance,
    PointerWithExposedProvenance,
    PointerCoercion(PointerCoercion),
    IntToInt,
    FloatToInt,
    FloatToFloat,
    IntToFloat,
    PtrToPtr,
    FnPtrToPtr,
    Transmute,
    Subtype,
    /// Extract the raw pointer inside a `Box`: the operand is the
    /// `Unique<T>` field of the box (a chain of single-field wrappers
    /// around a raw pointer) and the result is that raw pointer.
    /// Emitted by rustc deref/drop elaboration for `Box` places.
    BoxDerefTransmute,
}
impl CastKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::PointerExposeProvenance => "PointerExposeProvenance",
            Self::PointerWithExposedProvenance => "PointerWithExposedProvenance",
            Self::PointerCoercion(_) => "PointerCoercion",
            Self::IntToInt => "IntToInt",
            Self::FloatToInt => "FloatToInt",
            Self::FloatToFloat => "FloatToFloat",
            Self::IntToFloat => "IntToFloat",
            Self::PtrToPtr => "PtrToPtr",
            Self::FnPtrToPtr => "FnPtrToPtr",
            Self::Transmute => "Transmute",
            Self::Subtype => "Subtype",
            Self::BoxDerefTransmute => "BoxDerefTransmute",
        }
    }
    pub fn samples() -> Vec<CastKind> {
        vec![
            Self::PointerExposeProvenance,
            Self::PointerWithExposedProvenance,
            Self::PointerCoercion(PointerCoercion::Unsize),
            Self::IntToInt,
            Self::FloatToInt,
            Self::FloatToFloat,
            Self::IntToFloat,
            Self::PtrToPtr,
            Self::FnPtrToPtr,
            Self::Transmute,
            Self::Subtype,
            Self::BoxDerefTransmute,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FakeBorrowKind {
    Deep,
    Shallow,
}
#[derive(Clone, Debug, PartialEq)]
pub enum BorrowKind {
    Shared,
    Fake(FakeBorrowKind),
    Mut { allow_two_phase_borrow: bool },
}
impl BorrowKind {
    pub fn mutable() -> Self {
        Self::Mut {
            allow_two_phase_borrow: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NullaryOp {
    SizeOf,
    AlignOf,
    OffsetOf(Vec<u32>),
    UbChecks,
}
impl NullaryOp {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::SizeOf => "SizeOf",
            Self::AlignOf => "AlignOf",
            Self::OffsetOf(_) => "OffsetOf",
            Self::UbChecks => "UbChecks",
        }
    }
    pub fn samples() -> Vec<NullaryOp> {
        vec![
            Self::SizeOf,
            Self::AlignOf,
            Self::OffsetOf(vec![0]),
            Self::UbChecks,
        ]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AggregateKind {
    Array(Ty),
    Tuple,
    Adt {
        def: DefId,
        variant_index: u32,
        generic_args: Vec<Ty>,
        user_ty: Option<u32>,
        field_index: Option<u32>,
    },
    Closure {
        def: DefId,
        generic_args: Vec<Ty>,
    },
    Coroutine {
        def: DefId,
        generic_args: Vec<Ty>,
    },
    CoroutineClosure {
        def: DefId,
        generic_args: Vec<Ty>,
    },
    RawPtr {
        data_ty: Ty,
        mutability: Mutability,
    },
}
impl AggregateKind {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Array(_) => "Array",
            Self::Tuple => "Tuple",
            Self::Adt { .. } => "Adt",
            Self::Closure { .. } => "Closure",
            Self::Coroutine { .. } => "Coroutine",
            Self::CoroutineClosure { .. } => "CoroutineClosure",
            Self::RawPtr { .. } => "RawPtr",
        }
    }
    pub fn samples() -> Vec<AggregateKind> {
        let d = DefId::new(0, 0);
        vec![
            Self::Array(Ty::Bool),
            Self::Tuple,
            Self::Adt {
                def: d,
                variant_index: 0,
                generic_args: vec![],
                user_ty: None,
                field_index: None,
            },
            Self::Closure {
                def: d,
                generic_args: vec![],
            },
            Self::Coroutine {
                def: d,
                generic_args: vec![],
            },
            Self::CoroutineClosure {
                def: d,
                generic_args: vec![],
            },
            Self::RawPtr {
                data_ty: Ty::Bool,
                mutability: Mutability::Not,
            },
        ]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Rvalue {
    Use(Operand),
    Repeat(Operand, u64),
    Ref(Region, BorrowKind, Place),
    ThreadLocalRef(DefId),
    RawPtr(Mutability, Place),
    Len(Place),
    Cast(CastKind, Operand, Ty),
    BinaryOp(BinOp, Box<(Operand, Operand)>),
    CheckedBinaryOp(BinOp, Box<(Operand, Operand)>),
    NullaryOp(NullaryOp, Ty),
    UnaryOp(UnOp, Operand),
    Discriminant(Place),
    Aggregate(Box<AggregateKind>, Vec<Operand>),
    ShallowInitBox(Operand, Ty),
    CopyForDeref(Place),
    /// A reborrow: a bitwise copy of the pointer/reference value held
    /// at the place, reinterpreted as the target type (the layout is
    /// guaranteed equal by the `CoerceShared` contract). Mirrors
    /// Stable MIR's `Rvalue::Reborrow(Ty, Mutability, Place)`: with
    /// `Mutability::Mut` the value keeps its type; otherwise it is
    /// coerced to the (shared) target type.
    Reborrow(Ty, Mutability, Place),
}
impl Rvalue {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Use(_) => "Use",
            Self::Repeat(..) => "Repeat",
            Self::Ref(..) => "Ref",
            Self::ThreadLocalRef(_) => "ThreadLocalRef",
            Self::RawPtr(..) => "RawPtr",
            Self::Len(_) => "Len",
            Self::Cast(..) => "Cast",
            Self::BinaryOp(..) => "BinaryOp",
            Self::CheckedBinaryOp(..) => "CheckedBinaryOp",
            Self::NullaryOp(..) => "NullaryOp",
            Self::UnaryOp(..) => "UnaryOp",
            Self::Discriminant(_) => "Discriminant",
            Self::Aggregate(..) => "Aggregate",
            Self::ShallowInitBox(..) => "ShallowInitBox",
            Self::CopyForDeref(_) => "CopyForDeref",
            Self::Reborrow(..) => "Reborrow",
        }
    }
    pub fn samples() -> Vec<Rvalue> {
        use crate::operand::Constant;
        use crate::place::Local;
        let p = || Place::from_local(Local(1));
        let op = || Operand::constant(Constant::int(1));
        vec![
            Self::Use(op()),
            Self::Repeat(op(), 3),
            Self::Ref(Region::Erased, BorrowKind::Shared, p()),
            Self::ThreadLocalRef(DefId::new(0, 0)),
            Self::RawPtr(Mutability::Not, p()),
            Self::Len(p()),
            Self::Cast(CastKind::IntToInt, op(), Ty::Bool),
            Self::BinaryOp(BinOp::Add, Box::new((op(), op()))),
            Self::CheckedBinaryOp(BinOp::Add, Box::new((op(), op()))),
            Self::NullaryOp(NullaryOp::SizeOf, Ty::Bool),
            Self::UnaryOp(UnOp::Not, op()),
            Self::Discriminant(p()),
            Self::Aggregate(Box::new(AggregateKind::Tuple), vec![]),
            Self::ShallowInitBox(op(), Ty::Bool),
            Self::CopyForDeref(p()),
            Self::Reborrow(Ty::Bool, Mutability::Not, p()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage;
    #[test]
    fn binop_table_matches() {
        assert_eq!(BinOp::ALL.len(), coverage::BIN_OPS.len());
        for op in BinOp::ALL {
            assert!(coverage::BIN_OPS.contains(&op.name()));
        }
    }
    #[test]
    fn unop_and_coercion_tables() {
        assert_eq!(UnOp::ALL.len(), coverage::UN_OPS.len());
        assert_eq!(
            PointerCoercion::ALL.len(),
            coverage::POINTER_COERCIONS.len()
        );
    }
    #[test]
    fn cast_samples_cover_table() {
        let names: Vec<&str> = CastKind::samples().iter().map(|c| c.name()).collect();
        assert_eq!(names.len(), coverage::CAST_KINDS.len());
        for n in coverage::CAST_KINDS {
            assert!(names.contains(n));
        }
    }
    #[test]
    fn rvalue_samples_cover_table() {
        let names: Vec<&str> = Rvalue::samples().iter().map(|r| r.kind_name()).collect();
        assert_eq!(names.len(), coverage::RVALUE_KINDS.len());
        for n in coverage::RVALUE_KINDS {
            assert!(names.contains(n), "missing {n}");
        }
    }
    #[test]
    fn aggregate_and_nullary_cover() {
        assert_eq!(
            AggregateKind::samples().len(),
            coverage::AGGREGATE_KINDS.len()
        );
        assert_eq!(NullaryOp::samples().len(), coverage::NULLARY_OPS.len());
    }
}
