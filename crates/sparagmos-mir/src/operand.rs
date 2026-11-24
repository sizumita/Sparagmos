//! `Operand`, `Constant`, `ConstValue`, `Scalar`.
use crate::place::{Place, PlaceElem};
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
    /// A pointer constant with no provenance: the stored integer
    /// is the address itself — a null pointer, or a dangling
    /// pointer minted by `without_provenance`. It denotes no
    /// readable pointee; only its address identity is observable.
    IntPtr(u64),
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
    /// A constant allocation preserved exactly — its bytes and
    /// its provenance relocations — because the reference that
    /// denotes it declares a zero-sized pointee type while the
    /// allocation itself is not empty: the pointer was stored
    /// through a type-erased slot (a `NonNull<()>` field is the
    /// canonical case), so the pointee's real type is named only
    /// by the pointer casts the guest later performs. The
    /// interpreter materialises the block into the cast target's
    /// type at those casts (see `sparagmos_interp::ir::MatPlan`);
    /// until then the block is the pointee's identity, exactly
    /// like the snapshot a typed decode would have produced.
    ErasedAlloc {
        /// The allocation's bytes from the denoted offset on;
        /// `None` bytes are uninitialised.
        bytes: Vec<Option<u8>>,
        /// Provenance relocations inside the block: the byte
        /// offset of a pointer slot and the reference value the
        /// slot holds, decoded type-free (a nested block for a
        /// memory allocation, a function value for a function
        /// allocation, an address for a dangling word).
        relocs: Vec<(u32, ConstValue)>,
    },
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
    /// A reference (or raw pointer) to a `static` item. The
    /// pointee is the static's own storage, identified by the
    /// static's `DefId`; `value` is the decoded initializer — the
    /// value the storage holds before the program runs. Unlike
    /// [`ConstValue::Ref`], whose pointee is an immutable snapshot,
    /// a static reference observes (and, through a mutable
    /// reference, updates) the one program-level storage of that
    /// static: reads and writes through it are read-after-write
    /// consistent across the whole run.
    StaticRef {
        def: DefId,
        value: Box<ConstValue>,
        /// The path from the static's base to the pointee this
        /// reference denotes: empty for a reference to the static
        /// itself, a field/index path for a reference into its
        /// interior (`static B: &usize = &A.a;`). `value` is always
        /// the static's full initializer, never the interior value:
        /// seeding the store with an interior value would corrupt
        /// every other access to the same static.
        projection: Vec<PlaceElem>,
        /// The element count when the reference is a fat slice
        /// reference (`&[T]` over the static's array storage, the
        /// shape `static mut S: &mut [T] = &mut [...]` produces):
        /// the evaluated value is a fat reference over the static
        /// reference. `None` for a thin reference.
        len: Option<u64>,
    },
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
    /// A vtable pointer value, decoded from an allocation whose
    /// provenance names a vtable (`GlobalAlloc::VTable`). Vtables
    /// are not materialised in this model; the only observable
    /// property of a vtable pointer is its identity (equality and
    /// ordering of the raw pointers a program forms from it), so
    /// the value is a stable identity number for the (concrete
    /// type, trait) pair, assigned by the driver.
    VTablePtr(u64),
    /// A fat pointer constant with no provenance: a dangling
    /// address plus its metadata word (a slice length), as minted
    /// by `slice::from_raw_parts(NonNull::dangling().as_ptr(), 0)`
    /// in a const context. Only the pointer identity and the
    /// metadata are observable; there is no pointee to read.
    FatPtr { addr: u64, meta: u64 },
    /// A fat reference (or raw pointer) to a `dyn Trait` value in
    /// constant position: the data pointer constant plus the
    /// vtable identity. Vtables are not materialised in this
    /// model; the vtable is represented by what it is a table
    /// *for* -- the concrete pointee type and the trait bounds
    /// the pointer answers to. The concrete type rides as a type,
    /// not a name, so conversion renders its name and builds its
    /// drop glue with the same machinery a runtime unsizing cast
    /// uses (see `sparagmos_interp::ir::UnsizeMeta::Dyn`).
    /// `traits` holds the bounds of the `dyn` type the pointer
    /// was decoded from, as the name strings of [`Ty::Dynamic`].
    /// `base` is the data pointer constant in the shape the sized
    /// value's own reference takes: a [`ConstValue::Ref`]
    /// snapshot of the pointee, a [`ConstValue::StaticRef`] when
    /// the pointee is a static's storage, or an address-only /
    /// null value when the stored data word has no provenance.
    DynRef {
        base: Box<ConstValue>,
        concrete: Ty,
        traits: Vec<String>,
    },
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
    /// The constant's own type, when the producer recorded it and
    /// it is a plain scalar type (int / uint / bool / char / float).
    /// A scalar literal's bits carry no signedness of their own, so
    /// a consumer with no typing context from the operation (a
    /// comparison of two literals is the canonical case) reads the
    /// bits under this type instead of guessing from the sign bit.
    pub ty: Option<Ty>,
}
impl Constant {
    pub fn from_value(v: ConstValue) -> Self {
        Self {
            literal: ConstKind::Value(v),
            user_ty: None,
            ty: None,
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
