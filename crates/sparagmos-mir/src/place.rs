//! `Local`, `LocalDecl`, `Place` and projections.
use crate::ty::{Mutability, Ty};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Local(pub u32);
pub const RETURN_PLACE: Local = Local(0);
impl Local {
    pub const RETURN_PLACE: Local = Local(0);
    pub fn index(self) -> usize {
        self.0 as usize
    }
    pub fn is_return_place(self) -> bool {
        self.0 == 0
    }
}
impl From<u32> for Local {
    fn from(v: u32) -> Self {
        Self(v)
    }
}
impl From<usize> for Local {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalDecl {
    pub ty: Ty,
    pub mutability: Mutability,
}
impl LocalDecl {
    pub fn new(ty: Ty) -> Self {
        Self {
            ty,
            mutability: Mutability::Not,
        }
    }
    pub fn mutable(ty: Ty) -> Self {
        Self {
            ty,
            mutability: Mutability::Mut,
        }
    }
}

/// Mirrors `rustc_middle::mir::ProjectionElem`.
#[derive(Clone, Debug, PartialEq)]
pub enum PlaceElem {
    Deref,
    Field(u32, Ty),
    Index(Local),
    ConstantIndex {
        offset: u64,
        min_length: u64,
        from_end: bool,
    },
    Subslice {
        from: u64,
        to: u64,
        from_end: bool,
    },
    Downcast(Option<String>, u32),
    OpaqueCast(Ty),
    Subtype(Ty),
}
impl PlaceElem {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Deref => "Deref",
            Self::Field(..) => "Field",
            Self::Index(_) => "Index",
            Self::ConstantIndex { .. } => "ConstantIndex",
            Self::Subslice { .. } => "Subslice",
            Self::Downcast(..) => "Downcast",
            Self::OpaqueCast(_) => "OpaqueCast",
            Self::Subtype(_) => "Subtype",
        }
    }
    pub fn samples() -> Vec<PlaceElem> {
        vec![
            Self::Deref,
            Self::Field(0, Ty::Bool),
            Self::Index(Local(1)),
            Self::ConstantIndex {
                offset: 0,
                min_length: 1,
                from_end: false,
            },
            Self::Subslice {
                from: 0,
                to: 1,
                from_end: false,
            },
            Self::Downcast(None, 0),
            Self::OpaqueCast(Ty::Bool),
            Self::Subtype(Ty::Bool),
        ]
    }
}
/// Backwards-friendly alias matching rustc naming.
pub type ProjectionElem = PlaceElem;

#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub local: Local,
    pub projection: Vec<PlaceElem>,
}
impl Place {
    pub fn from_local(local: Local) -> Self {
        Self {
            local,
            projection: Vec::new(),
        }
    }
    pub fn return_place() -> Self {
        Self::from_local(RETURN_PLACE)
    }
    pub fn field(mut self, field: u32, ty: Ty) -> Self {
        self.projection.push(PlaceElem::Field(field, ty));
        self
    }
    pub fn deref(mut self) -> Self {
        self.projection.push(PlaceElem::Deref);
        self
    }
    pub fn index(mut self, idx: Local) -> Self {
        self.projection.push(PlaceElem::Index(idx));
        self
    }
    pub fn downcast(mut self, name: Option<String>, variant: u32) -> Self {
        self.projection.push(PlaceElem::Downcast(name, variant));
        self
    }
    pub fn is_direct(&self) -> bool {
        self.projection.is_empty()
    }
    /// Type after stripping a leading `Deref` chain count (best-effort, no typeck).
    pub fn deref_count(&self) -> usize {
        self.projection
            .iter()
            .filter(|p| matches!(p, PlaceElem::Deref))
            .count()
    }
}
impl From<Local> for Place {
    fn from(l: Local) -> Self {
        Self::from_local(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage;
    #[test]
    fn projection_samples_cover_table() {
        let names: Vec<&str> = PlaceElem::samples().iter().map(|p| p.kind_name()).collect();
        assert_eq!(names.len(), coverage::PROJECTION_ELEMS.len());
        for n in coverage::PROJECTION_ELEMS {
            assert!(names.contains(n));
        }
    }
    #[test]
    fn place_builders() {
        let p = Place::from_local(Local(6))
            .downcast(Some("Some".into()), 1)
            .field(0, Ty::Uint(crate::ty::UintTy::Usize));
        assert_eq!(p.projection.len(), 2);
        assert!(!p.is_direct());
        assert!(Place::return_place().is_direct());
    }
}
