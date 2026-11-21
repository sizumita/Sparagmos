//! Type model mirroring `rustc_middle::ty::TyKind` (reference: toolchain 1.100.0).
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DefId {
    pub krate: u32,
    pub index: u32,
}
impl DefId {
    pub const fn new(krate: u32, index: u32) -> Self {
        Self { krate, index }
    }
}
impl fmt::Display for DefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DefId({}:{})", self.krate, self.index)
    }
}
pub type AdtDefId = DefId;

/// Discriminant values of one ADT, indexed by variant index:
/// `values[i]` is the discriminant of variant `i`, as reported by
/// the compiler for that variant, stored as an `i128`. A
/// `Discriminant` rvalue yields these values; the interpreter
/// stores them through the discriminant type of the destination
/// place, which canonicalises them for comparison against
/// `SwitchInt` branch keys.
///
/// An enum without explicit discriminants has the sequential values
/// `0, 1, 2, ...`; a struct has a single entry, `0`. Consumers that
/// find no entry for an ADT treat the variant index itself as the
/// value, which is exact for enums with sequential discriminants.
pub type DiscriminantValues = Vec<i128>;

/// Discriminant tables for the ADTs of a program, keyed by the
/// `DefId` of the ADT. The driver fills one entry per enum it
/// converts; the interpreter reads them when evaluating
/// `Discriminant` rvalues.
pub type AdtDiscriminants = std::collections::HashMap<AdtDefId, DiscriminantValues>;

/// Layout of one monomorphic ADT instance: `(size, align)` in bytes,
/// as computed by the compiler's own layout query (so `repr(C)`,
/// `repr(packed)`, `repr(align)` and enum tag placement are already
/// reflected). Keyed by the rendered instance name — the same string
/// the interpreter's type-tag rendering produces for the ADT type
/// (e.g. `Aligned`, `Pair<i32, u8>`) — because layouts differ per
/// monomorphization and the `size_of`/`align_of` builtins only ever
/// see the rendered tag. The driver fills one entry per ADT instance
/// it converts whose layout the compiler can give; unsized types and
/// types whose layout is not computable (generic parameters, trait
/// objects) get no entry and stay explicitly unsupported in the
/// interpreter.
pub type AdtLayouts = std::collections::HashMap<String, (u64, u64)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
}
impl IntTy {
    pub const ALL: &[IntTy] = &[
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::I128,
        Self::Isize,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::I128 => "i128",
            Self::Isize => "isize",
        }
    }
    pub fn bits(self) -> u32 {
        match self {
            Self::I8 => 8,
            Self::I16 => 16,
            Self::I32 => 32,
            Self::I64 => 64,
            Self::I128 => 128,
            Self::Isize => 64,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UintTy {
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
}
impl UintTy {
    pub const ALL: &[UintTy] = &[
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::U128,
        Self::Usize,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::U128 => "u128",
            Self::Usize => "usize",
        }
    }
    pub fn bits(self) -> u32 {
        match self {
            Self::U8 => 8,
            Self::U16 => 16,
            Self::U32 => 32,
            Self::U64 => 64,
            Self::U128 => 128,
            Self::Usize => 64,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatTy {
    F16,
    F32,
    F64,
    F128,
}
impl FloatTy {
    pub const ALL: &[FloatTy] = &[Self::F16, Self::F32, Self::F64, Self::F128];
    pub fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F128 => "f128",
        }
    }
    pub fn bits(self) -> u32 {
        match self {
            Self::F16 => 16,
            Self::F32 => 32,
            Self::F64 => 64,
            Self::F128 => 128,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mutability {
    Not,
    Mut,
}
impl Mutability {
    pub fn is_mut(self) -> bool {
        matches!(self, Self::Mut)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Region {
    Erased,
    Static,
    Var(u32),
    Placeholder,
    Named(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Safety {
    Safe,
    Unsafe,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Abi {
    Rust,
    C,
    System,
    Stdcall,
    Fastcall,
    RustCall,
    RustIntrinsic,
    Unadjusted,
    Other(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FnSig {
    pub inputs: Vec<Ty>,
    pub output: Box<Ty>,
    pub c_variadic: bool,
    pub safety: Safety,
    pub abi: Abi,
}
impl FnSig {
    pub fn new(inputs: Vec<Ty>, output: Ty) -> Self {
        Self {
            inputs,
            output: Box::new(output),
            c_variadic: false,
            safety: Safety::Safe,
            abi: Abi::Rust,
        }
    }
}

/// Length of a `[T; N]` array type.
#[derive(Clone, Debug, PartialEq)]
pub enum ArrayLen {
    Known(u64),
    Param(String),
    Unevaluated(DefId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AliasKind {
    Projection,
    Opaque,
    Free,
    Inherent,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Bool,
    Char,
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Adt(AdtDefId, Vec<Ty>),
    Foreign(DefId),
    Str,
    Array(Box<Ty>, ArrayLen),
    /// `rustc` `TyKind::Pat` (pattern types, unstable).
    Pat(Box<Ty>, String),
    Slice(Box<Ty>),
    RawPtr(Box<Ty>, Mutability),
    Ref(Region, Box<Ty>, Mutability),
    FnDef(DefId, Vec<Ty>),
    FnPtr(Box<FnSig>),
    /// `dyn Trait` — trait names kept as strings (no trait solver in Sparagmos yet).
    Dynamic(Vec<String>, Box<Ty>, Region),
    Closure(DefId, Vec<Ty>),
    Coroutine(DefId, Vec<Ty>),
    CoroutineWitness(DefId, Vec<Ty>),
    Alias(AliasKind, DefId, Vec<Ty>),
    Param(u32, String),
    Bound(u32),
    Placeholder(String),
    Infer(String),
    Error,
    Never,
    Tuple(Vec<Ty>),
}

impl Ty {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Bool => "Bool",
            Self::Char => "Char",
            Self::Int(_) => "Int",
            Self::Uint(_) => "Uint",
            Self::Float(_) => "Float",
            Self::Adt(..) => "Adt",
            Self::Foreign(_) => "Foreign",
            Self::Str => "Str",
            Self::Array(..) => "Array",
            Self::Pat(..) => "Pat",
            Self::Slice(_) => "Slice",
            Self::RawPtr(..) => "RawPtr",
            Self::Ref(..) => "Ref",
            Self::FnDef(..) => "FnDef",
            Self::FnPtr(_) => "FnPtr",
            Self::Dynamic(..) => "Dynamic",
            Self::Closure(..) => "Closure",
            Self::Coroutine(..) => "Coroutine",
            Self::CoroutineWitness(..) => "CoroutineWitness",
            Self::Alias(..) => "Alias",
            Self::Param(..) => "Param",
            Self::Bound(_) => "Bound",
            Self::Placeholder(_) => "Placeholder",
            Self::Infer(_) => "Infer",
            Self::Error => "Error",
            Self::Never => "Never",
            Self::Tuple(_) => "Tuple",
        }
    }
    pub fn unit() -> Self {
        Self::Tuple(Vec::new())
    }
    pub fn is_unit(&self) -> bool {
        matches!(self, Self::Tuple(v) if v.is_empty())
    }
    pub fn is_never(&self) -> bool {
        matches!(self, Self::Never)
    }
    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Bool)
    }
    pub fn is_integral(&self) -> bool {
        matches!(self, Self::Int(_) | Self::Uint(_) | Self::Char | Self::Bool)
    }
    pub fn is_floating(&self) -> bool {
        matches!(self, Self::Float(_))
    }
    pub fn is_signed(&self) -> bool {
        matches!(self, Self::Int(_))
    }
    /// One representative instance of every `Ty` variant (used by coverage tests).
    pub fn samples() -> Vec<Ty> {
        let d = DefId::new(0, 0);
        vec![
            Self::Bool,
            Self::Char,
            Self::Int(IntTy::I32),
            Self::Uint(UintTy::U64),
            Self::Float(FloatTy::F64),
            Self::Adt(d, vec![]),
            Self::Foreign(d),
            Self::Str,
            Self::Array(Box::new(Self::Bool), ArrayLen::Known(3)),
            Self::Pat(Box::new(Self::Bool), "pat".into()),
            Self::Slice(Box::new(Self::Bool)),
            Self::RawPtr(Box::new(Self::Bool), Mutability::Not),
            Self::Ref(Region::Erased, Box::new(Self::Bool), Mutability::Not),
            Self::FnDef(d, vec![]),
            Self::FnPtr(Box::new(FnSig::new(vec![], Self::unit()))),
            Self::Dynamic(vec!["Send".into()], Box::new(Self::unit()), Region::Erased),
            Self::Closure(d, vec![]),
            Self::Coroutine(d, vec![]),
            Self::CoroutineWitness(d, vec![]),
            Self::Alias(AliasKind::Projection, d, vec![]),
            Self::Param(0, "T".into()),
            Self::Bound(0),
            Self::Placeholder("p".into()),
            Self::Infer("_".into()),
            Self::Error,
            Self::Never,
            Self::Tuple(vec![]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage;
    #[test]
    fn ty_samples_cover_table() {
        let names: Vec<&str> = Ty::samples().iter().map(|t| t.kind_name()).collect();
        assert_eq!(names.len(), coverage::TY_KINDS.len());
        for n in coverage::TY_KINDS {
            assert!(names.contains(n), "missing Ty sample {n}");
        }
    }
    #[test]
    fn int_uint_float_tables() {
        assert_eq!(IntTy::ALL.len(), coverage::INT_TYS.len());
        assert_eq!(UintTy::ALL.len(), coverage::UINT_TYS.len());
        assert_eq!(FloatTy::ALL.len(), coverage::FLOAT_TYS.len());
    }
}
