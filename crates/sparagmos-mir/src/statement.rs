//! `Statement` / `StatementKind`.
use crate::operand::Operand;
use crate::place::{Local, Place};
use crate::rvalue::Rvalue;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}
impl Span {
    pub const DUMMY: Span = Span { lo: 0, hi: 0 };
    pub fn new(lo: u32, hi: u32) -> Self {
        Self { lo, hi }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceInfo {
    pub span: Span,
    pub scope: u32,
}
impl SourceInfo {
    pub const DUMMY: SourceInfo = SourceInfo {
        span: Span::DUMMY,
        scope: 0,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FakeReadCause {
    ForMatchedGuard,
    ForGuardBinding,
    ForLet,
    ForIndexOf,
    ForVarBinding,
}
impl FakeReadCause {
    pub const ALL: &[FakeReadCause] = &[
        Self::ForMatchedGuard,
        Self::ForGuardBinding,
        Self::ForLet,
        Self::ForIndexOf,
        Self::ForVarBinding,
    ];
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RetagKind {
    FnEntry,
    TwoPhase,
    Raw,
    Default,
}
impl RetagKind {
    pub const ALL: &[RetagKind] = &[Self::FnEntry, Self::TwoPhase, Self::Raw, Self::Default];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variance {
    Covariant,
    Invariant,
    Contravariant,
    Bivariant,
}

/// Minimal form of rustc's `UserTypeProjection`.
#[derive(Clone, Debug, PartialEq)]
pub struct UserTypeProjection {
    pub user_ty_index: u32,
    pub projections: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NonDivergingIntrinsic {
    CopyNonOverlapping {
        src: Operand,
        dst: Operand,
        count: Operand,
    },
    Assume(Operand),
}
impl NonDivergingIntrinsic {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::CopyNonOverlapping { .. } => "CopyNonOverlapping",
            Self::Assume(_) => "Assume",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum StatementKind {
    Assign(Box<(Place, Rvalue)>),
    FakeRead(Box<(FakeReadCause, Place)>),
    SetDiscriminant {
        place: Box<Place>,
        variant_index: u32,
    },
    StorageLive(Local),
    StorageDead(Local),
    Retag(RetagKind, Box<Place>),
    PlaceMention(Box<Place>),
    AscribeUserType(Box<(Place, UserTypeProjection)>, Variance),
    Coverage,
    Intrinsic(Box<NonDivergingIntrinsic>),
    ConstEvalCounter,
    Nop,
}
impl StatementKind {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Assign(_) => "Assign",
            Self::FakeRead(_) => "FakeRead",
            Self::SetDiscriminant { .. } => "SetDiscriminant",
            Self::StorageLive(_) => "StorageLive",
            Self::StorageDead(_) => "StorageDead",
            Self::Retag(..) => "Retag",
            Self::PlaceMention(_) => "PlaceMention",
            Self::AscribeUserType(..) => "AscribeUserType",
            Self::Coverage => "Coverage",
            Self::Intrinsic(_) => "Intrinsic",
            Self::ConstEvalCounter => "ConstEvalCounter",
            Self::Nop => "Nop",
        }
    }
    pub fn samples() -> Vec<StatementKind> {
        use crate::operand::Constant;
        let p = || Place::from_local(Local(1));
        vec![
            Self::Assign(Box::new((
                p(),
                Rvalue::Use(Operand::constant(Constant::int(0))),
            ))),
            Self::FakeRead(Box::new((FakeReadCause::ForLet, p()))),
            Self::SetDiscriminant {
                place: Box::new(p()),
                variant_index: 0,
            },
            Self::StorageLive(Local(1)),
            Self::StorageDead(Local(1)),
            Self::Retag(RetagKind::Default, Box::new(p())),
            Self::PlaceMention(Box::new(p())),
            Self::AscribeUserType(
                Box::new((
                    p(),
                    UserTypeProjection {
                        user_ty_index: 0,
                        projections: vec![],
                    },
                )),
                Variance::Covariant,
            ),
            Self::Coverage,
            Self::Intrinsic(Box::new(NonDivergingIntrinsic::Assume(Operand::constant(
                Constant::bool(true),
            )))),
            Self::ConstEvalCounter,
            Self::Nop,
        ]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Statement {
    pub source_info: SourceInfo,
    pub kind: StatementKind,
}
impl Statement {
    pub fn new(kind: StatementKind) -> Self {
        Self {
            source_info: SourceInfo::DUMMY,
            kind,
        }
    }
    pub fn assign(place: Place, rv: Rvalue) -> Self {
        Self::new(StatementKind::Assign(Box::new((place, rv))))
    }
    pub fn nop() -> Self {
        Self::new(StatementKind::Nop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage;
    #[test]
    fn statement_samples_cover_table() {
        let names: Vec<&str> = StatementKind::samples()
            .iter()
            .map(|s| s.kind_name())
            .collect();
        assert_eq!(names.len(), coverage::STATEMENT_KINDS.len());
        for n in coverage::STATEMENT_KINDS {
            assert!(names.contains(n), "missing {n}");
        }
    }
}
