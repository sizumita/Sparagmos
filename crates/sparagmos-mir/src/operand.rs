//! `Operand`, `Constant`, `ConstValue`, `Scalar`.
use crate::place::Place;
use crate::ty::{DefId, FloatTy, Ty};

#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Int { data: u128, size: u8 },
    Float { bits: u128, ty: FloatTy },
    Ptr { alloc_id: u64, offset: u64 },
}
impl Scalar {
    pub fn int(v: u64) -> Self {
        Self::Int {
            data: v as u128,
            size: 8,
        }
    }
    pub fn bool(v: bool) -> Self {
        Self::Int {
            data: v as u128,
            size: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConstValue {
    Scalar(Scalar),
    Slice { data: Vec<u8>, meta: u64 },
    ByRef { alloc_id: u64, offset: u64 },
    ZeroSized,
    Bool(bool),
    Char(char),
    Str(String),
    Bytes(Vec<u8>),
    /// A tuple or array value, element by element. Arrays ride as
    /// their elements; the length lives in the type.
    Tuple(Vec<ConstValue>),
    /// An ADT value: variant index plus field values in definition
    /// order.
    Adt { variant: u32, fields: Vec<ConstValue> },
    /// A thin reference (or raw pointer) to a constant value: the
    /// materialised form of a reference constant whose pointee is
    /// itself a constant value.
    Ref(Box<ConstValue>),
    /// A wholly uninitialized constant of the constant's type
    /// (the payload of a `MaybeUninit::uninit()` constant):
    /// evaluating it yields the uninitialized value, and reading
    /// it is the interpreter's usual uninitialized-read error.
    Uninit,
    /// A null pointer constant: an allocation word of zero with no
    /// provenance. Equality with another null pointer holds; any
    /// dereference is an explicit error in the interpreter (the
    /// `RefTarget::Null` value).
    NullPtr,
    /// A fat reference to decoded elements: a slice constant whose
    /// elements are not plain bytes.
    SliceVal(Vec<ConstValue>),
    /// A function pointer value, decoded from an allocation whose
    /// provenance names this function item: the callable value of
    /// the item itself (the same target a  function
    /// item constant names in operand position).
    FnDef(DefId),
    /// A function pointer value, decoded from an allocation whose
    /// provenance names this closure (through its  adapter
    /// shim): the closure coerced to a function pointer. Only
    /// non-capturing closures coerce to function pointers, so the
    /// environment is empty by construction.
    Closure(DefId),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConstKind {
    Value(ConstValue),
    Ty(Ty),
    Unevaluated { def: DefId, args: Vec<Ty> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Constant {
    pub literal: ConstKind,
    /// Optional user-type annotation index (minimal form of rustc's `UserTypeAnnotationIndex`).
    pub user_ty: Option<u32>,
}
impl Constant {
    pub fn from_value(v: ConstValue) -> Self {
        Self {
            literal: ConstKind::Value(v),
            user_ty: None,
        }
    }
    pub fn int(v: u64) -> Self {
        Self::from_value(ConstValue::Scalar(Scalar::int(v)))
    }
    pub fn bool(v: bool) -> Self {
        Self::from_value(ConstValue::Bool(v))
    }
    pub fn str_(v: &str) -> Self {
        Self::from_value(ConstValue::Str(v.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant(Box<Constant>),
}
impl Operand {
    pub fn copy(p: Place) -> Self {
        Self::Copy(p)
    }
    pub fn move_(p: Place) -> Self {
        Self::Move(p)
    }
    pub fn constant(c: Constant) -> Self {
        Self::Constant(Box::new(c))
    }
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Copy(_) => "Copy",
            Self::Move(_) => "Move",
            Self::Constant(_) => "Constant",
        }
    }
    pub fn place(&self) -> Option<&Place> {
        match self {
            Self::Copy(p) | Self::Move(p) => Some(p),
            Self::Constant(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::place::Local;
    #[test]
    fn operand_kinds() {
        assert_eq!(
            Operand::copy(Place::from_local(Local(1))).kind_name(),
            "Copy"
        );
        assert_eq!(
            Operand::constant(Constant::bool(true)).kind_name(),
            "Constant"
        );
        assert!(Operand::constant(Constant::int(3)).place().is_none());
    }
}
