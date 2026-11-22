//! Minimal MIR-like IR executed by the interpreter.
//!
//! This is Lane C's own model, shaped after rustc MIR as of Rust 1.100.0
//! (`rustc_middle::mir`): a [`Body`] is a list of basic blocks, each block
//! is statements plus exactly one terminator, statements are assignments,
//! and expressions are operands / binary / unary operations.
//!
//! Deliberate simplifications for the first cut (see the crate-level docs
//! for the assumed Lane A API and the expected differences):
//! * Functions are identified by name (`String`), not by `DefId`.
//! * A call's callee is a plain name, not a constant `FnDef` operand.
//!   Calls into code with no converted body (std/core runtime support)
//!   use the dedicated [`TerminatorKind::CallBuiltin`] instead.
//! * Places support field / deref / index / constant-index / downcast
//!   projections; references are resolved values (see
//!   [`crate::value::RefTarget`]).
//! * Types cover scalars, tuples, ADTs (by name), references, slices,
//!   arrays and `str`; no lifetimes, no trait objects.

use std::collections::HashMap;
use std::fmt;

pub use sparagmos_mir::AsmStep;

use crate::value::{RefProj, Value};

/// Index of a local variable. Local 0 is always the return place,
/// locals `1..=arg_count` are the arguments (as in rustc MIR).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Local(pub usize);

impl Local {
    /// The return place, `_0`.
    pub const RETURN: Local = Local(0);
}

impl fmt::Display for Local {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_{}", self.0)
    }
}

/// Index of a basic block inside a [`Body`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BasicBlock(pub usize);

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// Signed integer types (bit width matters for overflow semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
}

impl IntTy {
    /// Width in bits. `isize` is 64 on the x86_64 target Sparagmos aims at.
    pub fn bits(self) -> u32 {
        match self {
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
            IntTy::Isize => 64,
        }
    }
    /// Inclusive `(min, max)` representable in this type.
    pub fn range(self) -> (i128, i128) {
        let bits = self.bits();
        if bits >= 128 {
            (i128::MIN, i128::MAX)
        } else {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        }
    }
}

/// Unsigned integer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UintTy {
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
}

impl UintTy {
    /// Width in bits. `usize` is 64 on x86_64.
    pub fn bits(self) -> u32 {
        match self {
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
            UintTy::Usize => 64,
        }
    }
    /// Largest representable value.
    pub fn max(self) -> u128 {
        let bits = self.bits();
        if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        }
    }
}

/// Floating-point types. `f32` / `f64` values are stored widened
/// to `f64` (see `Value::Float`); `f16` widens exactly and its
/// operations round back to `f16` precision. `f128` has no exact
/// `f64` form, so it is stored as its raw bit pattern
/// (`Value::Float128`) and computed by the exact software
/// arithmetic in [`crate::f128`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatTy {
    F16,
    F32,
    F64,
    F128,
}

impl FloatTy {
    /// Width in bits.
    pub fn bits(self) -> u32 {
        match self {
            FloatTy::F16 => 16,
            FloatTy::F32 => 32,
            FloatTy::F64 => 64,
            FloatTy::F128 => 128,
        }
    }
}

/// The type model used for width / overflow semantics and for
/// validating values written to places.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ty {
    Unit,
    Bool,
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    /// Tuple type, e.g. the `(T, bool)` produced by a checked operation.
    Tuple(Vec<Ty>),
    Never,
    /// A struct/enum/union type, identified by its full path as
    /// rendered by the driver (e.g. `std::fmt::Arguments`). ADT values
    /// are variant + fields; the name is what runtime support for
    /// std types (iterators, `String`, ...) dispatches on.
    Adt(String),
    /// A reference or raw pointer to the given pointee type. The two
    /// share one representation: a resolved [`crate::value::RefTarget`]
    /// (or a fat pointer for slice pointees).
    Ref(Box<Ty>),
    /// A slice type `[T]`; only appears behind a reference.
    Slice(Box<Ty>),
    /// An array type `[T; N]` with a known length. Arrays are valued
    /// as tuples of their elements.
    Array(Box<Ty>, u64),
    /// The `str` type; only appears behind a reference or as the type
    /// of a string-data place.
    Str,
    /// A function pointer type (`fn(A) -> B`). The signature is not
    /// modelled: call arguments are values, and their count is
    /// checked against the callee body's arity at the call site.
    FnPtr,
    /// A function item type: the zero-sized type of one specific
    /// function, identified by the function's name.
    FnDef(String),
    /// A closure type: the closure body's name in the program, and
    /// the number of captured variables (the length of the upvars
    /// tuple). The count matters because a zero-capture closure is
    /// zero-sized: rustc MIR never writes its place, so the
    /// interpreter synthesises the (unique) empty-environment value
    /// when such a place is read.
    Closure(String, usize),
    /// A `dyn Trait` type: the trait bounds' names (principal trait
    /// first), as the driver renders them. The type is unsized: it
    /// only appears behind a reference or raw pointer, or as the
    /// type of a place reached through a fat pointer's dereference.
    /// Values of this type are [`crate::value::Value::DynRef`] fat
    /// pointers; the erased concrete type is carried by the value,
    /// not the type.
    Dynamic(Vec<String>),
    /// A coroutine type (an `async` block / `async fn` future, a
    /// `gen` block): the coroutine body's name, as the driver
    /// renders the definition. By the time a body reaches this
    /// conversion, rustc's coroutine state transform has already run:
    /// the value is the state object - a discriminant (the state:
    /// unresumed, one per suspension point, returned, panicked) plus
    /// the fields live in that state - and the coroutine body is an
    /// ordinary function that dispatches on the discriminant, saves
    /// state through downcast field writes, and returns
    /// `Poll` / `CoroutineState` values. Values of this type are
    /// [`crate::value::Value::Coroutine`].
    Coroutine(String),
}

impl Ty {
    pub fn i8() -> Ty {
        Ty::Int(IntTy::I8)
    }
    pub fn i32() -> Ty {
        Ty::Int(IntTy::I32)
    }
    pub fn i64() -> Ty {
        Ty::Int(IntTy::I64)
    }
    pub fn i128() -> Ty {
        Ty::Int(IntTy::I128)
    }
    pub fn u32() -> Ty {
        Ty::Uint(UintTy::U32)
    }
    pub fn u64() -> Ty {
        Ty::Uint(UintTy::U64)
    }
    pub fn usize_() -> Ty {
        Ty::Uint(UintTy::Usize)
    }
}

/// One step of a place projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionElem {
    /// Tuple / aggregate field access, e.g. `_1.0`.
    Field(usize),
    /// Field access on a union value, e.g. `_1.0` where `_1` is a
    /// union: the read reinterprets the bits of whichever field
    /// the value currently holds (see [`Program::union_tables`]),
    /// and the write makes this field the held one. `table`
    /// indexes the union instance's per-field scalar shapes in
    /// the program's union tables.
    UnionField { idx: usize, table: usize },
    /// `*_1`: follow a reference value.
    Deref,
    /// `_1[_2]`: index by a local (read in the executing frame).
    Index(Local),
    /// `_1[3]` / `_1[len - 2]`: index by a constant offset.
    ConstantIndex { offset: u64, from_end: bool },
    /// Project an enum value to one variant's fields.
    Downcast(u32),
    /// `_1[from..to]`: the sub-slice of an array or slice place.
    /// With `from_end`, `to` counts back from the base's length
    /// (the range is `from..len - to`); otherwise it is the
    /// range's end. The bounds are the constants MIR carries;
    /// resolution checks them against the base's actual length.
    Subslice { from: u64, to: u64, from_end: bool },
}

/// A place: a local plus a (possibly empty) projection chain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Place {
    pub local: Local,
    pub projection: Vec<ProjectionElem>,
}

impl Place {
    pub fn local(local: Local) -> Place {
        Place {
            local,
            projection: Vec::new(),
        }
    }
    pub fn from_index(index: usize) -> Place {
        Place::local(Local(index))
    }
    /// Append a tuple-field projection, e.g. `place.field(0)`.
    pub fn field(mut self, field: usize) -> Place {
        self.projection.push(ProjectionElem::Field(field));
        self
    }
    pub fn is_direct_local(&self) -> bool {
        self.projection.is_empty()
    }
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.local)?;
        for elem in &self.projection {
            match elem {
                ProjectionElem::Field(i) => write!(f, ".{i}")?,
                ProjectionElem::UnionField { idx, .. } => write!(f, ".{idx}")?,
                ProjectionElem::Deref => write!(f, ".*")?,
                ProjectionElem::Index(l) => write!(f, "[{l}]")?,
                ProjectionElem::ConstantIndex { offset, from_end } => {
                    if *from_end {
                        write!(f, "[len - {offset}]")?
                    } else {
                        write!(f, "[{offset}]")?
                    }
                }
                ProjectionElem::Downcast(v) => write!(f, " as variant {v}")?,
                ProjectionElem::Subslice { from, to, from_end } => {
                    if *from_end {
                        write!(f, "[{from}..len - {to}]")?
                    } else {
                        write!(f, "[{from}..{to}]")?
                    }
                }
            }
        }
        Ok(())
    }
}

/// A compile-time constant.
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Unit,
    Bool(bool),
    Int(i128),
    Uint(u128),
    /// A floating-point constant, stored widened to `f64` (an `f32`
    /// value widens exactly, so nothing is lost).
    Float(f64),
    /// An `f128` constant: the raw IEEE 754 binary128 bit pattern.
    Float128(u128),
    /// String data (a `str` / `&str` constant).
    Str(String),
    /// Raw bytes (a byte-array constant, e.g. an fmt pieces template).
    Bytes(Vec<u8>),
    /// A tuple or array constant, element by element.
    Tuple(Vec<Constant>),
    /// An ADT constant: variant index plus field constants.
    Adt { variant: u32, fields: Vec<Constant> },
    /// A reference to a constant value.
    Ref(Box<Constant>),
    /// A pointer constant with no provenance (dangling):
    /// evaluates to an address-only reference value.
    IntPtr(u64),
    /// A wholly uninitialized constant: evaluates to the
    /// uninitialized value.
    Uninit,
    /// A null pointer constant: evaluates to the null reference
    /// value (equality with another null holds, dereference is an
    /// explicit error).
    NullPtr,
    /// A reference to a `static` item's storage: `name` is the
    /// static's program key and `value` is its initializer constant.
    /// Evaluating it seeds the interpreter's statics store from
    /// `value` on first touch and yields a reference to the store,
    /// so writes through the reference are observed by later reads
    /// (unlike [`Constant::Ref`], whose pointee is an immutable
    /// snapshot).
    StaticRef {
        name: String,
        value: Box<Constant>,
        /// The path from the static's base to the pointee (empty
        /// for a reference to the static itself).
        projection: Vec<RefProj>,
        /// The element count when the reference is a fat slice
        /// reference over the static's array storage: evaluation
        /// yields a fat reference over the static reference.
        /// `None` for a thin reference.
        len: Option<u128>,
    },
    /// A slice constant with decoded (non-byte) elements.
    SliceVal(Vec<Constant>),
    /// A vtable pointer constant: evaluates to an address-only
    /// reference whose address is the vtable's identity number
    /// (see `sparagmos_mir::ConstValue::VTablePtr`), so pointer
    /// equality / ordering between vtable pointers behaves as it
    /// does natively (same vtable equal, distinct ones ordered).
    VTablePtr(u64),
    /// A provenance-less fat pointer constant (dangling address +
    /// slice length): evaluates to a fat reference over an
    /// address-only base.
    FatPtr { addr: u64, meta: u64 },
    /// A function item constant: evaluates to the callable value
    /// naming the function.
    FnDef(String),
    /// A non-capturing closure constant (a closure value in
    /// constant position, or a closure coerced to a function
    /// pointer by a constant): evaluates to the closure value
    /// naming the closure body, with an empty environment.
    Closure(String),
}

/// An operand: the right-hand side leaf of most rvalues.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant(Constant),
}

impl Operand {
    pub fn copy_place(place: Place) -> Operand {
        Operand::Copy(place)
    }
    pub fn copy_local(index: usize) -> Operand {
        Operand::Copy(Place::from_index(index))
    }
    pub fn move_local(index: usize) -> Operand {
        Operand::Move(Place::from_index(index))
    }
    pub fn const_int(v: i128) -> Operand {
        Operand::Constant(Constant::Int(v))
    }
    pub fn const_uint(v: u128) -> Operand {
        Operand::Constant(Constant::Uint(v))
    }
    pub fn const_bool(v: bool) -> Operand {
        Operand::Constant(Constant::Bool(v))
    }
    pub fn const_unit() -> Operand {
        Operand::Constant(Constant::Unit)
    }
}

/// Binary operators (subset of rustc MIR `BinOp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Three-way comparison (`BinOp::Cmp`): the result is a
    /// `core::cmp::Ordering` value, not a boolean.
    Cmp,
}

/// Unary operators (rustc MIR `UnOp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    Neg,
    Not,
    /// `PtrMetadata`: the length metadata of a fat pointer.
    PtrMetadata,
}

/// One side of a scalar `Transmute` reinterpretation
/// ([`Rvalue::Reinterpret`]): how a value of this shape flattens
/// to a bit pattern (little-endian, the x86_64 target's order)
/// and how a bit pattern constructs a value of this shape. Only
/// shapes whose every bit is accounted for by scalar leaves are
/// described; the two sides of a reinterpretation always agree
/// on the total width, at most 128 bits.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarShape {
    /// An integer of the given width. Flattening masks the
    /// stored value to the width (a signed source is stored
    /// sign-extended, so masking recovers its bit pattern);
    /// construction wraps into the width with the given
    /// signedness.
    Int { bits: u32, signed: bool },
    /// A float of the given width (16, 32 or 64). The stored
    /// value is an `f64` (an `f32` / `f16` widened exactly), so
    /// flattening narrows it back to its own width's bit pattern
    /// and construction widens the pattern's value back.
    Float { bits: u32 },
    /// A `bool`: one byte, flattened as 0 or 1; construction
    /// reads any nonzero pattern as `true`, the same convention
    /// as a cast to `bool`.
    Bool,
    /// A `char`: flattened as its 32-bit code point (chars are
    /// stored as unsigned integers); construction rejects a
    /// pattern that is not a valid code point rather than
    /// fabricating a `char`.
    Char,
    /// A thin pointer: flattened as its synthetic address (the
    /// identity [`crate::interp::synthetic_address`] assigns the
    /// pointer value), constructed as an address-only reference
    /// holding the pattern — the same reading the pointer <->
    /// integer transmutes give a bare pointer, extended to the
    /// pointer payloads of wrapper and niche-encoded shapes
    /// (`NonNull<T>`, `Option<NonNull<T>>`).
    Ptr,
    /// An array of a scalar element shape: element `i` occupies
    /// bits `[i * elem_bits, (i + 1) * elem_bits)`.
    Array(Box<ScalarShape>, u64),
    /// A one-payload-field ADT (a struct, a union, or a
    /// single-variant enum): the payload field at `index`
    /// carries the whole bit pattern; the remaining fields are
    /// zero-sized with uniquely determined values, baked in as
    /// `fill` constants for construction and ignored when
    /// flattening. `count` is the ADT's total field count.
    AdtField {
        index: u32,
        count: usize,
        fill: Vec<(u32, Constant)>,
        inner: Box<ScalarShape>,
    },
    /// A niche-encoded enum: the payload-carrying variant
    /// (`untagged_variant`) stores its single field's own bit
    /// pattern (the `inner` shape), and the fieldless variants
    /// `niche_variants.0..=niche_variants.1` are stored as
    /// `niche_start + (variant - niche_variants.0)` in that
    /// same field — the encoding `Option<NonZero<T>>` and its
    /// kin use for their discriminant.
    NicheEnum {
        inner: Box<ScalarShape>,
        untagged_variant: u32,
        niche_variants: (u32, u32),
        niche_start: u128,
    },
    /// A fieldless enum: the whole bit pattern is the
    /// variant's discriminant, held in a tag of `bits` width.
    /// Flattening yields the discriminant's bit pattern in
    /// that width; construction selects the variant whose
    /// discriminant's pattern matches and rejects a pattern
    /// no variant has (the same contract as the
    /// integer-to-enum transmute).
    FieldlessEnum {
        bits: u32,
        discriminants: Vec<i128>,
    },
}

impl ScalarShape {
    /// The total width in bits this shape accounts for.
    pub fn bits(&self) -> u32 {
        match self {
            ScalarShape::Int { bits, .. } | ScalarShape::Float { bits } => *bits,
            ScalarShape::Bool => 8,
            ScalarShape::Char => 32,
            ScalarShape::Ptr => 64,
            ScalarShape::Array(elem, n) => elem.bits() * (*n as u32),
            ScalarShape::AdtField { inner, .. } => inner.bits(),
            ScalarShape::NicheEnum { inner, .. } => inner.bits(),
            ScalarShape::FieldlessEnum { bits, .. } => *bits,
        }
    }
}

/// A right-hand side expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Rvalue {
    Use(Operand),
    BinaryOp(BinOp, Box<(Operand, Operand)>),
    /// Like rustc MIR: yields a `(value, overflowed)` tuple, wrapping.
    CheckedBinaryOp(BinOp, Box<(Operand, Operand)>),
    UnaryOp(UnOp, Operand),
    /// Build a tuple value from operands. Array aggregates ride as
    /// tuples too (the length lives in the declared type).
    AggregateTuple(Vec<Operand>),
    /// Build an ADT value: a variant index plus field values.
    AggregateAdt {
        variant: u32,
        operands: Vec<Operand>,
        /// The active field of a union aggregate: a union value is
        /// built from one operand, which initializes this field
        /// rather than field 0. `None` for struct/enum aggregates,
        /// whose operands fill the variant's fields from index 0.
        field_index: Option<u32>,
        /// A pointer-payload denomination rewrite (see
        /// [`Rvalue::PtrCastUnits`]): when a wrapper-to-wrapper
        /// transmute retargets the payload pointer at a
        /// differently sized pointee, the payload field (the
        /// operand index in the second element) has its trailing
        /// projection index rewritten in byte units at the
        /// source pointee size (the first element) as the
        /// aggregate is built. `None` for every other aggregate.
        units_from: Option<(u64, usize)>,
    },
    /// Build a closure value: the closure body's name plus the
    /// captured environment values (the aggregate's operands, in
    /// field order).
    ClosureAggregate { def: String, operands: Vec<Operand> },
    /// Build a raw pointer from parts (`AggregateKind::RawPtr`,
    /// what `ptr::from_raw_parts` lowers to): the data pointer
    /// operand plus, for a slice pointee, the length operand.
    /// Evaluates to the data pointer value itself, or to a fat
    /// reference over it when a metadata operand is present.
    RawPtrAggregate {
        data: Operand,
        meta: Option<Operand>,
    },
    /// An `as` cast to the given scalar type (wrapping, as in Rust).
    Cast(Operand, Ty),
    /// An unsizing coercion (`PointerCoercion::Unsize`). The
    /// metadata describes what the coercion attaches, decided at
    /// conversion time from the source and target types.
    Unsize(Operand, UnsizeMeta),
    /// `BoxDerefTransmute` cast: the operand is the chain of
    /// single-field wrappers a `Box` keeps around its raw pointer
    /// (`Unique` around `NonNull` around the pointer); yields the
    /// raw pointer itself.
    BoxDerefTransmute(Operand),
    /// The data pointer of a fat pointer value (a fat-to-thin
    /// pointer cast, e.g. `<[T]>::as_ptr`'s `*const [T]` to
    /// `*const T`): evaluates to the fat pointer's base, or to the
    /// operand itself when it is not fat.
    DataPtr(Operand),
    /// A `Transmute` cast from a pointer/reference value to an
    /// integer: the pointer's address. The model has no flat memory,
    /// so the address is synthetic: a deterministic, non-zero,
    /// 4096-aligned value derived from the reference target (the
    /// same target always yields the same address). That is all the
    /// observable behavior the model's pointers can honestly offer:
    /// equality, null and alignment checks (the compiler's UB
    /// precondition checks transmute pointers to `usize` for exactly
    /// those) behave correctly; the numeric value itself is not a
    /// real address and must not be turned back into a pointer.
    PtrToInt(Operand),
    /// A pointer cast from a transparent single-field wrapper
    /// (`UnsafeCell<T>` / `MaybeUninit<T>`) to the wrapped type
    /// itself - what `UnsafeCell::get` and `MaybeUninit::as_mut_ptr`
    /// return. The wrapper is `repr(transparent)`: its only field
    /// is the value, at offset 0, so the cast pointer denotes that
    /// field. In this projection-based model the result is the
    /// operand pointer with a field-0 projection appended, decided
    /// at conversion time from the cast's source and target types.
    /// Any other pointer-to-pointer cast leaves the denoted place
    /// unchanged and stays a plain `Use`.
    ///
    /// The `u32` is the number of wrapper layers the cast enters:
    /// usually 1, but a cast that crosses nested wrappers in one
    /// step (a pointer to the thread-local storage's
    /// `UnsafeCell<MaybeUninit<T>>` cast straight to `T`, which is
    /// how std's lazy thread-local access returns its pointer after
    /// `UnsafeCell::get` and `cast` are inlined together) appends
    /// one field-0 projection per layer.
    PtrField(Operand, u32),
    /// A pointer cast from an array / slice / `str` pointee to the
    /// element type itself - what `<[T]>::as_ptr`, `<[T; N]>::as_ptr`
    /// and `str::as_ptr` return, and what `&arr as *const T` writes.
    /// Such a pointer denotes the aggregate's first element, not the
    /// aggregate: in this projection-based model the result is the
    /// operand pointer with an index-0 projection appended, decided
    /// at conversion time from the cast's source and target types.
    /// Any other pointer-to-pointer cast leaves the denoted place
    /// unchanged and stays a plain `Use`.
    /// A pointer-to-pointer cast whose source and target
    /// pointee sizes differ (decided at conversion time): the
    /// denoted position does not change, but its projection
    /// index is denominated in the SOURCE pointee's elements,
    /// while every later offset on the result is denominated
    /// in the target's. The operand is evaluated and a trailing
    /// element index is rewritten as the equivalent byte
    /// offset (`index * source size`, the second field), the
    /// unit-free form later offsets accumulate onto in bytes.
    /// A pointer without a trailing element index denotes the
    /// same position in any unit and passes through unchanged.
    PtrCastUnits(Operand, u64),
    PtrFirstElem(Operand),
    /// The reverse cast: an element pointer cast back to a pointer
    /// to the whole array (`as_ptr().cast::<[T; N]>()`). The result
    /// denotes the array as a whole, so a trailing index-0
    /// projection - the marker a first-element cast (or an
    /// `&arr[0]` borrow) left on the pointer - is removed at run
    /// time. A pointer whose projection does not end in index 0
    /// denotes an element past the first; an array starting there
    /// has no representation in this projection model, so the value
    /// passes through unchanged and reads through it keep failing
    /// with a type mismatch, as they did before.
    PtrArrayBase(Operand),
    /// An array -> array `Transmute` cast entering an element
    /// wrapper (`[T; N]` -> `[MaybeUninit<T>; N]`, or the
    /// `UnsafeCell` twin): each element takes the wrapper's
    /// value representation -- an ADT value holding the payload
    /// at field 0, the shape `MaybeUninit::new` constructions
    /// are normalized to -- so the cast maps that representation
    /// over the array's elements. (The reverse cast out of the
    /// wrapper is [`Rvalue::UnwrapArrayElems`].)
    WrapArrayElems(Operand),
    /// An array -> array `Transmute` cast leaving an element
    /// wrapper (`[MaybeUninit<T>; N]` -> `[T; N]`, or the
    /// `UnsafeCell` twin): each element's wrapper representation
    /// is unwrapped to the payload it holds.
    UnwrapArrayElems(Operand),
    /// An integer-to-pointer `Transmute` cast: the mirror image of
    /// [`Rvalue::PtrToInt`]. The integer is an address in the
    /// model's synthetic address space, so the result is an
    /// address-only reference value: it compares for equality,
    /// tests for null and transmutes back to an integer, but
    /// reading or writing through it is an explicit error (the
    /// address names no place), exactly like a pointer constant
    /// with no provenance.
    IntToPtr(Operand),
    /// A `Transmute` cast from an integer to a transparent
    /// single-payload wrapper around a thin pointer (`NonNull`,
    /// `Unique`): the reverse of transmuting the wrapper to an
    /// integer. The integer is an address in the model's
    /// synthetic address space (see [`Rvalue::IntToPtr`]), held
    /// in the wrapper's payload field; the remaining fields are
    /// zero-sized and take their unique values.
    IntToPtrWrapped {
        operand: Operand,
        payload_index: u32,
        field_count: usize,
        fill: Vec<(u32, Constant)>,
    },
    /// A `Transmute` cast from an ADT value whose scalar
    /// shape bottoms out at an integer (a `NonZero<usize>`,
    /// or a transparent wrapper chain over an integer) to a
    /// transparent pointer wrapper (`NonNull`, `Unique`):
    /// the flattened integer is the payload pointer's
    /// address -- the transmute `NonNull::without_provenance`
    /// performs. `from` is the source value's scalar shape
    /// (its flattening is the address); the remaining fields
    /// are as in [`Rvalue::IntToPtrWrapped`].
    ScalarToPtrWrapped {
        operand: Operand,
        from: ScalarShape,
        payload_index: u32,
        field_count: usize,
        fill: Vec<(u32, Constant)>,
    },
    /// A `Transmute` cast from a fieldless-enum value (or a
    /// transparent one-field wrapper around one, such as
    /// `mem::Alignment`) to a transparent pointer wrapper
    /// (`NonNull`, `Unique`): the variant's discriminant is the
    /// payload pointer's address — the transmute
    /// `NonNull::dangling` performs to turn an alignment into a
    /// dangling pointer. The operand is the enum value itself
    /// (conversion extracts the field for the wrapped form);
    /// `discriminants` is the same baked table
    /// [`Rvalue::EnumToInt`] carries (an empty table means the
    /// discriminants are the variant indices).
    EnumToPtrWrapped {
        operand: Operand,
        discriminants: Vec<i128>,
        payload_index: u32,
        field_count: usize,
        fill: Vec<(u32, Constant)>,
    },
    /// A `Transmute` cast from a fieldless enum value to an
    /// integer: yields the discriminant value of the operand's
    /// variant, taken from the table baked in at conversion time
    /// (variant index -> discriminant value; an empty table means
    /// the values are the variant indices), converted to the
    /// target scalar type with the usual wrapping cast semantics.
    EnumToInt(Operand, Vec<i128>, Ty),
    /// A `Transmute` cast from an integer to a fieldless enum
    /// value: yields the variant whose discriminant value equals
    /// the operand's bit pattern, from the table baked in at
    /// conversion time (same convention as
    /// [`Rvalue::EnumToInt`]). An integer matching no discriminant
    /// is an explicit error: the model does not fabricate an
    /// invalid enum value.
    IntToEnum(Operand, Vec<i128>),
    /// A `Transmute` cast from an integer to a transparent
    /// wrapper struct around a fieldless enum (`mem::Alignment`
    /// is the canonical case): the integer is the discriminant
    /// of the enum value the wrapper's payload field holds (the
    /// same table convention as [`Rvalue::IntToEnum`]); the
    /// remaining fields are zero-sized and take their unique
    /// values. An integer matching no discriminant is an
    /// explicit error, as for [`Rvalue::IntToEnum`].
    IntToEnumWrapped {
        operand: Operand,
        discriminants: Vec<i128>,
        payload_index: u32,
        field_count: usize,
        fill: Vec<(u32, Constant)>,
    },
    /// A `Transmute` cast between scalar-shaped types of equal
    /// total size: the operand's bit pattern, flattened through
    /// the source [`ScalarShape`], reinterpreted through the
    /// target one. This is the bit-level reinterpretation
    /// `transmute` means for the shapes the value model can hold
    /// exactly: integers, floats, `bool`, `char`, arrays of
    /// scalars, and one-payload-field ADTs over those. Aggregate
    /// <-> aggregate byte reinterpretation beyond one payload
    /// field stays unsupported: there is no flat byte memory to
    /// reinterpret through.
    Reinterpret(Operand, ScalarShape, ScalarShape),
    /// A whole-pointee load of a vector-shaped value (an ADT or
    /// array whose scalar shape is an array of lanes, e.g. an
    /// x86 vector type) through the place's dereference. Where
    /// the pointee is stored whole, the load is that value,
    /// canonicalised through its shape; where the pointee lives
    /// in a raw element block whose elements are the lanes'
    /// storage (a byte buffer read as a vector), the value is
    /// assembled from the consecutive elements at the place.
    ShapedLoad(Place, ScalarShape),
    /// A pointer offset (`BinaryOp::Offset`: `ptr.offset` /
    /// `ptr.add` / `ptr.sub` and the `byte_add` family after their
    /// cast to an element pointer): the first operand's pointer
    /// position advanced by the second operand's count, in elements
    /// of the pointer's pointee type. The third element is the
    /// pointee's size in bytes when conversion could compute it
    /// exactly (scalars, arrays of those, pointers); `None` leaves
    /// the size to execution, which derives it from the executing
    /// body's declared type of the pointer operand. Execution
    /// represents the result in the projection model: an offset that
    /// lands on an element of an array place is absorbed into the
    /// place's index projection, an address-only pointer's address
    /// advances by the byte count, and any other displacement is
    /// carried as a byte-offset projection step (see
    /// [`crate::value::RefProj::ByteOffset`]). A displacement whose
    /// size is unknown in all of these is an explicit error, never a
    /// fabricated position. The fourth element is the bit width
    /// of the count operand's MIR type when that type is unsigned
    /// (0 when it is signed): the `byte_*` family passes its
    /// `usize` count straight into the `Offset`, whose count is
    /// an `isize`, so execution reinterprets the unsigned bits as
    /// the signed value of that width.
    PtrOffset(Operand, Operand, Option<u64>, u32),
    /// `&place` (also raw-pointer creation): a resolved reference value.
    Ref(Place),
    /// The length of the value at a place (array / slice / `str`).
    Len(Place),
    /// The discriminant of the enum value at a place, with the
    /// discriminant values of the enum indexed by variant index
    /// (baked in at conversion time from the program's discriminant
    /// tables). An empty table means the values are the variant
    /// indices themselves, which is exact for enums whose
    /// discriminants are the sequential defaults.
    Discriminant(Place, Vec<i128>),
    /// `*place` read for a copy (rustc `CopyForDeref`): reads the
    /// place; when the value there is a reference, reads through it
    /// once more.
    CopyForDeref(Place),
    /// Build a coroutine state object (`AggregateKind::Coroutine`):
    /// the operands are the captured upvars, in field order. The
    /// object starts in the unresumed state (variant 0), whose
    /// fields are exactly the upvars; later states' fields are
    /// written by the coroutine body itself through downcast field
    /// writes.
    CoroutineAggregate(Vec<Operand>),
    /// A reborrow: a bitwise copy of the pointer/reference value held
    /// at the place, reinterpreted as the target pointer type. The
    /// value model does not track pointer mutability or provenance,
    /// so the copy is the value itself; the target type only matters
    /// to the destination local's declared type, which conversion
    /// already accounts for.
    Reborrow(Place),
    /// A reference to a thread-local static, named by the static's
    /// identity. The driver collects no static items at all, so there
    /// is no storage and no initializer to bind a cell to; executing
    /// this is a named explicit error, never a fabricated value.
    ThreadLocalRef(String),
    /// Any rvalue kind not modelled yet; executing it is an explicit error.
    Unsupported(String),
}

impl Rvalue {
    pub fn binary(op: BinOp, lhs: Operand, rhs: Operand) -> Rvalue {
        Rvalue::BinaryOp(op, Box::new((lhs, rhs)))
    }
    pub fn checked_binary(op: BinOp, lhs: Operand, rhs: Operand) -> Rvalue {
        Rvalue::CheckedBinaryOp(op, Box::new((lhs, rhs)))
    }
}

/// What an [`Rvalue::Unsize`] coercion attaches to the operand,
/// decided at conversion time from the source and target types
/// (mir_convert): exactly the metadata a real fat pointer carries.
#[derive(Debug, Clone, PartialEq)]
pub enum UnsizeMeta {
    /// Array-to-slice unsizing: the metadata is the element count,
    /// read off the pointee at run time.
    Slice,
    /// Sized-to-`dyn` unsizing: the metadata is the vtable, which
    /// this model represents by its identity — the concrete type's
    /// name (as the driver renders it, matching the `Self` spelling
    /// of the concrete type's impl method names) and the target
    /// type's trait bounds. `concrete` is `None` when the source is
    /// itself a `dyn` value (a supertrait upcast): the value's own
    /// concrete type carries over and only the bounds change.
    Dyn {
        concrete: Option<String>,
        traits: Vec<String>,
        /// The concrete type's drop glue, built by conversion from
        /// the cast's pointee type (the full generic instantiation,
        /// so no two instantiations can be confused). The value
        /// carries it from the cast on, like a vtable's drop shim.
        /// `None` when conversion could not account for the
        /// pointee's glue (dropping the value then fails
        /// explicitly) and for supertrait upcasts, where the value
        /// already carries its own glue.
        glue: Option<DropGlue>,
    },
}

/// Statement kinds (subset of rustc MIR `StatementKind`).
#[derive(Debug, Clone, PartialEq)]
pub enum StatementKind {
    Assign(Place, Rvalue),
    /// A whole-pointee store of a vector-shaped value through
    /// the place's dereference (the store half of
    /// [`Rvalue::ShapedLoad`]): where the pointee lives in a raw
    /// element block, the value is written lane by lane over the
    /// consecutive elements at the place, preserving each
    /// element's wrapper shape; anywhere else it is the plain
    /// whole-value store.
    ShapedStore {
        place: Place,
        value: Operand,
        shape: ScalarShape,
    },
    /// Set the discriminant of the enum value at a place, clearing
    /// its fields (as in rustc MIR).
    SetDiscriminant { place: Place, variant: u32 },
    /// Hint only in rustc MIR; a no-op here.
    StorageLive(Local),
    /// Hint only in rustc MIR; a no-op here.
    StorageDead(Local),
    /// `core::intrinsics::assume`: the operand is a boolean the
    /// programmer promises is true. Execution evaluates it; a false
    /// operand is undefined behaviour in the guest and is reported
    /// as a guest panic, a true one is a no-op.
    Assume(Operand),
    /// `core::intrinsics::copy_nonoverlapping`: copy `count` elements
    /// from the place `src` points at to the place `dst` points at.
    /// The operands are the pointer values themselves (references or
    /// raw pointers in the value model).
    CopyNonOverlapping {
        src: Operand,
        dst: Operand,
        count: Operand,
    },
    Nop,
    /// Any statement kind not modelled yet; executing it is an explicit error.
    Unsupported(String),
}

/// A single statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Statement {
    pub kind: StatementKind,
}

impl Statement {
    pub fn assign(place: Place, rvalue: Rvalue) -> Statement {
        Statement {
            kind: StatementKind::Assign(place, rvalue),
        }
    }
    pub fn nop() -> Statement {
        Statement {
            kind: StatementKind::Nop,
        }
    }
}

/// Which `Fn`-family trait method a [`TerminatorKind::CallTrait`]
/// dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FnCallKind {
    /// `Fn::call`: the receiver is a shared reference to the callable.
    Call,
    /// `FnMut::call_mut`: the receiver is a unique reference to the
    /// callable.
    CallMut,
    /// `FnOnce::call_once`: the receiver is the callable value
    /// itself, consumed by the call.
    CallOnce,
}

/// Drop glue for one dropped value, built by `mir_convert` from the
/// dropped type. The interpreter executes a plan against the value
/// actually stored at the dropped place; a part that is uninitialized
/// (moved out of, never written) is skipped, which is this value
/// model's form of rustc's drop flags. A plan is only built when
/// conversion can account for every part of the type: anything it
/// cannot account for keeps the whole `Drop` terminator unsupported.
#[derive(Debug, Clone, PartialEq)]
pub enum DropPlan {
    /// No guest-observable effect: nothing to run.
    Trivial,
    /// The glue of this part of the value is not modelled (a field or
    /// variant whose type conversion could not establish). Executing
    /// it is an explicit error, never a silent skip.
    Unknown,
    /// The type of this part (a field set by an aggregate and never
    /// projected anywhere) is unrecorded, so no glue could be built
    /// for it. Execution judges by the value's own shape: value kinds
    /// that never carry glue (scalars, references, strings) pass,
    /// tuples recurse into their elements, and an ADT value — which
    /// might hide a destructor — is an explicit error.
    Untyped,
    /// Run each plan in order.
    Seq(Vec<DropPlan>),
    /// Run `plan` against the sub-place at `path`, a projection from
    /// the value this plan applies to.
    At {
        path: Vec<RefProj>,
        plan: Box<DropPlan>,
    },
    /// Call the converted destructor body `func` (a program key) with
    /// a `&mut` reference to the value this plan applies to.
    CallDtor { func: String },
    /// Run `elem` against every element of the array / tuple / `Vec`
    /// value this plan applies to, in index order.
    Each { elem: Box<DropPlan> },
    /// Run `fields[i]` against field `i` of the tuple / ADT value
    /// this plan applies to, in field order. `complete` records that
    /// conversion proved the plan covers every field the value can
    /// hold; when it is false, a value holding more fields than the
    /// plan covers is an explicit error at run time.
    Fields {
        fields: Vec<DropPlan>,
        complete: bool,
    },
    /// Run `variants[v]` when the ADT value this plan applies to has
    /// variant index `v`. A live variant with no entry is an explicit
    /// error at run time (as if it were [`DropPlan::Unknown`]).
    Variants { variants: Vec<DropPlan> },
    /// The glue of one ADT instantiation, held in the terminator's
    /// glue table ([`DropGlue::adts`]) so recursive types stay
    /// finite: an ADT's entry is registered before its fields are
    /// planned, and a recursive occurrence refers back to it.
    Adt { index: usize },
    /// Release a `std::cell::RefCell` borrow held by the guard value
    /// this plan applies to (a `std::cell::Ref` / `RefMut` aggregate
    /// built by the `RefCell` borrow builtins in `builtins`): the
    /// guard's second field is a reference to the cell's borrow-flag
    /// `Cell`. A shared guard decrements the flag's count; a mutable
    /// guard resets it to unused (0). A guard whose flag reference
    /// can no longer be resolved (the cell's frame is gone) is an
    /// explicit error, as for any other dangling reference in this
    /// model — it is never silently skipped.
    RefCellRelease { mutable: bool },
    /// Decrement the live-handle count of the mpsc channel whose
    /// handle value this plan applies to (the `Sender` /
    /// `SyncSender` / `Receiver` drop glue in the interpreter's
    /// channel model, see `ChannelState` in `interp.rs`): dropping
    /// the last sender disconnects the channel for its receivers,
    /// and dropping the receiver for its senders. A moved-out
    /// handle has no count to decrement.
    ChannelHandleDrop { sender: bool },
    /// Drop one strong handle of a reference-counted allocation
    /// (the `Rc` / `Arc` destructor in the interpreter's allocation
    /// model, see `RefTarget::Alloc`): decrement the allocation's
    /// strong count, and when it reaches zero run `value` — the
    /// pointee type's glue — against the stored value, then
    /// release the weak bundle the strong handles collectively
    /// held (an [`DropPlan::RcWeakRelease`] the interpreter
    /// synthesizes for the allocation). A moved-out handle has no
    /// count to decrement.
    RcHandleDrop { value: Box<DropPlan> },
    /// The glue of a `std::task::Waker` / `LocalWaker`: which
    /// release runs depends on the shape the waker value has.
    /// A waker built by the modelled `From<Arc<T: Wake>>` /
    /// `From<Rc<T: LocalWake>>` conversion is a one-field
    /// aggregate carrying the handle itself, and dropping it
    /// releases that handle (the `RcHandleDrop` of the stored
    /// value, its pointee judged dynamically, as the erased
    /// pointee's glue can only be judged from the value). A
    /// waker built by `Waker::from_raw` holds a real `RawWaker`
    /// - a data pointer and a reference to a `RawWakerVTable`
    /// value - and dropping it calls the vtable's drop entry
    /// with the data pointer, exactly as `Waker`'s own `Drop`
    /// impl does. Any other shape is an explicit error.
    WakerDrop,
    /// Release one weak reference to a reference-counted
    /// allocation: decrement its weak count, freeing the storage
    /// when both counts are zero. The drop glue of a `Weak`
    /// handle; also synthesized by the interpreter itself for the
    /// strong bundle's weak release once a pointee's glue has run
    /// (the session is then rooted at the allocation itself).
    RcWeakRelease,
    /// The glue of a value of dynamic type (`dyn Trait`): the
    /// concrete type is known only from the value itself, which in
    /// this model is a `DynRef` carrying the payload value.
    /// Execution judges the payload by its shape, as
    /// [`DropPlan::Untyped`] does: scalars, strings and references
    /// own nothing, tuples and closure environments recurse, and an
    /// ADT value — whose concrete type might hide a destructor
    /// this plan cannot name — is an explicit error. This is the
    /// plan that lets a caught panic's `Box<dyn Any>` payload be
    /// dropped at scope end for the payload shapes the model can
    /// account for.
    Dynamic,
    /// The glue of a `dyn Trait` value: the dropped type is a trait
    /// object, so the glue is the *concrete* type's, which only the
    /// value knows. Executing it reads the value: a
    /// [`crate::value::Value::DynRef`] carries the concrete type's
    /// drop glue (built by conversion at the unsizing cast, the way
    /// a real vtable carries the drop shim), and the interpreter
    /// runs it against the pointee. A dyn value whose glue
    /// conversion could not account for is an explicit error, never
    /// a silent skip.
    Dyn,
}

/// What happens when execution of a basic block panics (the MIR
/// `UnwindAction` of the block terminator, preserved by conversion):
/// the cleanup block of the frame runs, the panic propagates to the
/// caller, or the run aborts. A body built by hand carries no unwind
/// table: every block then defaults to [`UnwindAction::Continue`],
/// the pre-unwinding behaviour of discarding the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnwindAction {
    /// Jump to this cleanup block in the panicking frame; the block
    /// chain ends with [`TerminatorKind::Resume`], which pops the
    /// frame and continues the unwind in the caller.
    Cleanup(BasicBlock),
    /// No cleanup in this frame: discard it and continue the unwind
    /// in the caller.
    Continue,
    /// The panic may not unwind past this point (a panic in a
    /// cleanup, or a `nounwind` boundary): the run aborts.
    Terminate,
    /// The callee was assumed never to unwind; that it did is an
    /// abort, exactly as for [`UnwindAction::Terminate`].
    Unreachable,
}

/// The drop glue of one `Drop` terminator: the plan for the dropped
/// type plus the table of ADT plans it references.
#[derive(Debug, Clone, PartialEq)]
pub struct DropGlue {
    /// Glue of the dropped type itself.
    pub root: DropPlan,
    /// Glue of each ADT instantiation reachable from the dropped
    /// type, referenced by [`DropPlan::Adt`] indices.
    pub adts: Vec<DropPlan>,
}

/// Terminator kinds (subset of rustc MIR `TerminatorKind`).
#[derive(Debug, Clone, PartialEq)]
pub enum TerminatorKind {
    Goto {
        target: BasicBlock,
    },
    SwitchInt {
        discr: Operand,
        /// `(discriminant value, target block)` pairs. Values are
        /// in the canonical comparison form produced by
        /// `Value::switch_bits` (sign-extended for signed
        /// discriminant types); conversion normalises the raw MIR
        /// bit patterns into this form.
        targets: Vec<(u128, BasicBlock)>,
        otherwise: BasicBlock,
    },
    Call {
        /// Callee name (simplification: rustc uses a constant FnDef operand).
        func: String,
        args: Vec<Operand>,
        destination: Place,
        /// Continuation block in the caller; `None` models a diverging call.
        target: Option<BasicBlock>,
    },
    /// A call through a callable value (function pointer, function
    /// item, or closure) held in a place: the `func` operand
    /// evaluates to the callable value. A closure callee is invoked
    /// with call-once semantics (its environment is consumed).
    CallIndirect {
        func: Operand,
        args: Vec<Operand>,
        destination: Place,
        target: Option<BasicBlock>,
    },
    /// A call through the `Fn` / `FnMut` / `FnOnce` trait methods
    /// (`std::ops::Fn::call` and siblings), in the shape rustc MIR
    /// emits for calling a closure: the receiver is the callable
    /// (a reference to it for `call` / `call_mut`, the value itself
    /// for `call_once`) and `args` evaluates to the argument tuple.
    CallTrait {
        kind: FnCallKind,
        receiver: Operand,
        args: Operand,
        /// The caller-location stamp the driver appended to the
        /// `Fn`-trait call when the resolved shim instance is
        /// `#[track_caller]` (a track-caller closure called
        /// directly or through a generic `Fn` bound): the
        /// interpreter binds it to the callee's location local,
        /// exactly like the stamp of a direct call. Absent for
        /// `dyn` receivers, whose call passes through the vtable
        /// shim: the shim's own location is bound instead (see
        /// `Program::fn_shim_locations`).
        stamp: Option<Operand>,
        destination: Place,
        target: Option<BasicBlock>,
    },
    /// A call into runtime support (std/core functions whose bodies
    /// only exist in the standard library): printing/formatting, fmt
    /// argument construction, iterator adapters over the modelled
    /// std types, and similar. Dispatched by `name` (the canonical
    /// callee path, generic lifetime groups stripped) in `builtins`.
    CallBuiltin {
        name: String,
        /// Rendered generic arguments of the callee (type tags the
        /// dispatch keys on, e.g. `std::ops::Range<usize>` for
        /// `Iterator::next`).
        generic_args: Vec<String>,
        args: Vec<Operand>,
        destination: Place,
        target: Option<BasicBlock>,
        /// Drop glue of the single argument when the callee is
        /// `std::mem::drop`: conversion builds the dropped type's
        /// [`DropGlue`] and the interpreter runs it against the
        /// argument place, exactly like a [`TerminatorKind::Drop`]
        /// for that place. `None` for every other builtin, and for
        /// a `mem::drop` whose argument is a constant with only
        /// trivial glue (discarding it is already faithful). A
        /// `mem::drop` whose glue conversion cannot account for
        /// the dropped type stays `Unsupported` instead.
        drop_glue: Option<DropGlue>,
        /// Discriminant values of the comparison's `Self` type,
        /// baked by conversion for the `PartialOrd` / `Ord`
        /// comparison builtins: their structural comparison orders
        /// enum values by discriminant value (the derived
        /// semantics), which the variant index alone cannot
        /// express for enums with explicit discriminants. `None`
        /// when the `Self` type is not an enum with a recorded
        /// table (the sequential default, where index and value
        /// agree).
        cmp_discriminants: Option<Vec<i128>>,
        /// Unsizing applied to the call's result before it is
        /// written to the destination, baked by conversion for a
        /// `From` / `Into` conversion whose target is a
        /// `Box<dyn Trait>`: the modelled conversion returns the
        /// plain box aggregate, and the result goes through the
        /// same unsizing a coercion cast applies
        /// ([`Rvalue::Unsize`]), so the destination holds a real
        /// `DynRef` carrying the concrete type's drop glue.
        /// `None` for every other builtin call.
        result_unsize: Option<UnsizeMeta>,
    },
    /// Drop the value at a place, running its `glue` (converted
    /// destructor calls and field glue; see `mir_convert`), and
    /// continue at `target`. Drops whose glue conversion cannot
    /// account for stay `Unsupported` instead.
    Drop {
        place: Place,
        target: BasicBlock,
        glue: DropGlue,
    },
    Return,
    /// Resume an in-flight unwind after a cleanup block chain has
    /// run: pop this frame and continue unwinding in the caller
    /// with the panic stashed in the frame when its cleanup was
    /// entered. Only reachable while unwinding; executing it at any
    /// other time is an explicit error.
    Resume,
    /// Abort the run (MIR `Terminate(Abort)`): a panic escaped a
    /// cleanup or crossed a `nounwind` boundary. Uncatchable: it is
    /// reported as [`crate::error::InterpError::Abort`], never as a
    /// guest panic a `catch_unwind` could observe.
    Abort,
    Unreachable,
    Assert {
        cond: Operand,
        expected: bool,
        target: BasicBlock,
        message: String,
        /// The assertion guards an arithmetic overflow (a MIR
        /// `AssertMessage::Overflow` / `OverflowNeg`): the kind of
        /// assertion a checked build emits after a
        /// `CheckedBinaryOp`. Whether it fires is the *session's*
        /// overflow-checks decision, not the body's: bodies marked
        /// `#[rustc_inherit_overflow_checks]` carry the assertion
        /// in their MIR regardless of the session flag, and native
        /// codegen drops it when the session has checks off. The
        /// interpreter does the same at run time (see
        /// `InterpConfig::overflow_checks`).
        overflow: bool,
    },
    /// An inline-assembly block of the evaluable class
    /// (sparagmos-mir's `inline_asm_program`): a straight-line
    /// register program over the asm operands, each a 64-bit slot
    /// seeded from its input value and written back when the
    /// program writes it. `target` is the continuation block;
    /// `None` ends the function after the steps (the
    /// naked-function `ret` shape).
    InlineAsm {
        regs: Vec<AsmReg>,
        steps: Vec<AsmStep>,
        target: Option<BasicBlock>,
    },
    /// Any terminator kind not modelled yet; executing it is an explicit error.
    Unsupported(String),
}

/// One register slot of a [`TerminatorKind::InlineAsm`]: how its
/// value enters the block and where the result goes.
#[derive(Debug, Clone, PartialEq)]
pub struct AsmReg {
    /// Input value (in / inout / const operands); `None` for a
    /// pure output, whose slot starts at zero.
    pub init: Option<Operand>,
    /// Write-back place (out / inout operands); `None` otherwise.
    pub dest: Option<Place>,
    /// Operand type width in bits (8, 16, 32 or 64).
    pub width: u32,
    /// The operand type's value kind: write-back sign-extends from
    /// `width` for `Signed`, zero-extends for `Unsigned`, and
    /// rebuilds a `bool` for `Bool`. (`char` operands are
    /// `Unsigned` at width 32: the value model carries chars as
    /// code-point integers.)
    pub kind: AsmRegKind,
}

/// The value kind of an [`AsmReg`]'s operand type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsmRegKind {
    Signed,
    Unsigned,
    Bool,
}

/// A basic block: statements followed by exactly one terminator.
#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlockData {
    pub statements: Vec<Statement>,
    pub terminator: TerminatorKind,
}

impl BasicBlockData {
    pub fn new(statements: Vec<Statement>, terminator: TerminatorKind) -> Self {
        BasicBlockData {
            statements,
            terminator,
        }
    }
}

/// The source position of one statement or terminator: an index
/// into [`Program::source_files`] plus a 1-based line number,
/// resolved by the driver from the Stable MIR span at conversion
/// time. The debugger interface ([`crate::debug`]) is the consumer:
/// breakpoints and stepping are line-based.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SourceLine {
    /// Index into the program's source-file table; `u32::MAX` when
    /// the position is unknown.
    pub file: u32,
    /// 1-based line number; 0 when unknown.
    pub line: u32,
}

impl SourceLine {
    pub const UNKNOWN: SourceLine = SourceLine { file: u32::MAX, line: 0 };

    pub fn is_known(&self) -> bool {
        self.line != 0 && self.file != u32::MAX
    }
}

/// Source lines of one basic block, parallel to [`Body::blocks`]:
/// one entry per statement, plus the terminator's. Empty in bodies
/// built by hand.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlockSourceLines {
    pub statements: Vec<SourceLine>,
    pub terminator: SourceLine,
}

/// One debugger-visible variable of a body.
#[derive(Clone, Debug, PartialEq)]
pub struct DebugVarEntry {
    /// The variable's source-level name.
    pub name: String,
    /// Where its value comes from.
    pub source: DebugVar,
    /// Whether the variable's source-level type is `char` or a
    /// reference to one (see [`Body::debug_vars`]).
    pub is_char: bool,
    /// The (start, end) source lines of the span the compiler
    /// recorded for the entry: usually a point span at the
    /// declaration. `(0, 0)` when unknown. When several entries
    /// share a name, the debugger interface resolves the name to
    /// the declaration in effect at the current line (a
    /// containing span first, then the latest declaration at or
    /// before it).
    pub lines: (u32, u32),
}

/// Where one debugger-visible variable's value comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum DebugVar {
    /// A place in the executing frame: the local, plus a
    /// field/downcast projection into it (empty for the local
    /// itself; only field/downcast projections are carried).
    Place(usize, Vec<crate::RefProj>),
    /// A constant the compiler's debug info records directly for
    /// the variable: a constant-valued variable whose storage the
    /// optimizer eliminated carries its value in its debug info
    /// instead, and a real debugger reads that same constant out
    /// of the DWARF location description.
    Constant(Constant),
}

/// A function body: argument count, local types, and basic blocks.
/// `local_tys[0]` is the return type; `local_tys.len()` is the frame size.
#[derive(Debug, Clone, PartialEq)]
pub struct Body {
    pub name: String,
    pub arg_count: usize,
    pub local_tys: Vec<Ty>,
    pub blocks: Vec<BasicBlockData>,
    /// Canonical values of the locals whose declared type is
    /// provably zero-sized, indexed by local (conversion-computed;
    /// see `mir_convert`'s `zst_value`). rustc MIR may never write
    /// such a local's place; a never-written read of it denotes the
    /// stored value. Empty for hand-built bodies: the interpreter
    /// then falls back to the shapes derivable from the IR type
    /// alone.
    pub zst_locals: Vec<Option<Value>>,
    /// Canonical values of field paths whose type has a single
    /// value, indexed by local, then by the field-index path from
    /// the local (conversion-computed; see `mir_convert`'s
    /// `collect_zst_fields`). A field of a single-valued type —
    /// typically a zero-sized field such as the closure slot of a
    /// `catch_unwind` Data union — may never be written by MIR at
    /// all (its store is elided like a ZST local's), so a read of
    /// the path that otherwise fails as uninitialized denotes this
    /// value instead. A read that succeeds from storage is never
    /// overridden: a union's storage slot can be shared with a
    /// field of a different type. Empty for hand-built bodies.
    pub zst_fields: Vec<std::collections::HashMap<Vec<usize>, Value>>,
    /// The unwind action of each block terminator, indexed by
    /// block (conversion-preserved from MIR; see [`UnwindAction`]).
    /// Empty for hand-built bodies, and a block past the end of the
    /// table defaults to [`UnwindAction::Continue`].
    pub unwind: Vec<UnwindAction>,
    /// For a `#[track_caller]` function: the index of the
    /// synthetic final local holding the caller location the
    /// driver stamped onto each call (conversion-assigned; see
    /// `mark_caller_location_bodies` in mir_convert). `None` for
    /// ordinary functions.
    pub caller_location_local: Option<usize>,
    /// Source lines of every block, parallel to `blocks` (see
    /// [`BlockSourceLines`]). Carried from the MIR body's recorded
    /// lines; empty for hand-built bodies.
    pub source_lines: Vec<BlockSourceLines>,
    /// The body's named variables, for the debugger interface
    /// (see [`DebugVarEntry`]), in declaration order.
    pub debug_vars: Vec<DebugVarEntry>,
}

impl Body {
    pub fn new(
        name: impl Into<String>,
        arg_count: usize,
        local_tys: Vec<Ty>,
        blocks: Vec<BasicBlockData>,
    ) -> Body {
        Body {
            name: name.into(),
            arg_count,
            local_tys,
            blocks,
            zst_locals: Vec::new(),
            zst_fields: Vec::new(),
            unwind: Vec::new(),
            caller_location_local: None,
            source_lines: Vec::new(),
            debug_vars: Vec::new(),
        }
    }
    pub fn local_count(&self) -> usize {
        self.local_tys.len()
    }
    pub fn local_ty(&self, local: Local) -> Option<&Ty> {
        self.local_tys.get(local.0)
    }

    /// The unwind action of the terminator of `block` (see
    /// [`UnwindAction`]); [`UnwindAction::Continue`] when the body
    /// carries no entry for the block.
    pub fn unwind_action(&self, block: usize) -> UnwindAction {
        self.unwind
            .get(block)
            .copied()
            .unwrap_or(UnwindAction::Continue)
    }
}

/// A whole program: a set of named function bodies in one module.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Program {
    /// The program's source-file table: the file names the
    /// bodies' source lines ([`Body::source_lines`]) index into.
    /// Carried from the driver's table; empty for hand-built
    /// programs.
    pub source_files: Vec<String>,
    /// Per-field scalar shapes of the program's union instances,
    /// indexed by the `table` of [`ProjectionElem::UnionField`]:
    /// entry `t` is one union instance's field list, in field
    /// order, with the [`ScalarShape`] a field's bits flatten to
    /// (`None` for a field whose type has no scalar shape:
    /// reading such a field through another field's bits is an
    /// explicit error). Filled from the MIR types at conversion;
    /// consulted when a union field is read through a field
    /// other than the one the value currently holds.
    pub union_tables: Vec<Vec<Option<ScalarShape>>>,
    /// Layouts of the program's monomorphic ADT instances
    /// (`(size, align)` in bytes), keyed by the rendered instance
    /// name (the type tag the builtins see). Filled from
    /// [`crate::mir_convert::MirProgram::adt_layouts`] at conversion;
    /// consulted by the `size_of` / `align_of` / `size_of_val`
    /// builtins for ADT tags the static layout table cannot answer.
    pub adt_layouts: std::collections::HashMap<String, (u64, u64)>,
    /// Derived-`Debug` shapes of the program's monomorphic ADT
    /// instances, keyed by the rendered instance name (the type
    /// tag the formatter sees), as recorded by the driver (see
    /// `sparagmos_mir::AdtDebugShapes`). Consulted by the `Debug`
    /// formatter when it meets an ADT value.
    pub adt_debug_shapes:
        std::collections::HashMap<String, sparagmos_mir::AdtDebugShape>,
    /// Thread-local statics of the program, keyed by the static
    /// key a `ThreadLocalRef` rvalue carries: the initializer
    /// constant each thread's storage for that static is seeded
    /// from on first access.
    pub thread_locals: std::collections::HashMap<String, Constant>,
    /// The guest allocation shims of a program that defines a
    /// `#[global_allocator]`: symbol (`__rust_alloc`,
    /// `__rust_dealloc`, `__rust_realloc`, `__rust_alloc_zeroed`)
    /// -> program key of the shim body. Calls the conversion
    /// resolved to the std entry points are bound to these
    /// bodies directly; the allocation-entry builtins (whose
    /// native forms allocate through those entry points) call
    /// them at run time, so a guest allocator observes every
    /// allocation the program makes. Empty when the program has
    /// no custom allocator.
    pub alloc_shims: std::collections::HashMap<String, String>,
    pub functions: HashMap<String, Body>,
    /// Functions that exist in the source program but whose bodies
    /// could not be converted, keyed by name, with the reason
    /// conversion failed. They deliberately hold no body: a function
    /// that is never called does not affect the run, and calling one
    /// fails with the recorded reason (see
    /// []).
    pub unconverted: HashMap<String, String>,
    /// Definition-location stamps of the program's
    /// `#[track_caller]` bodies, keyed by body key: the
    /// `&'static Location` constant of the body's own
    /// definition. Bound when a call reaches the body through
    /// a shim with no call-site stamp on the edge (a function
    /// pointer call, a vtable dispatch to an impl-only
    /// `#[track_caller]` method), which is the location the
    /// compiler's shim presents to the callee (see
    /// `Interpreter::start_call`).
    pub track_caller_defs: HashMap<String, Constant>,
    /// The location stamps of the `Fn` / `FnMut` / `FnOnce`
    /// vtable shims (index 0 is `call`, 1 is `call_mut`, 2 is
    /// `call_once`): the trait method definitions in core,
    /// which compiler-generated `Fn` impls are attributed to.
    /// Bound when a track-caller closure is entered through a
    /// `dyn` receiver.
    pub fn_shim_locations: [Option<Constant>; 3],
}

impl Program {
    pub fn new() -> Program {
        Program::default()
    }
    pub fn add_function(&mut self, body: Body) -> &mut Self {
        self.functions.insert(body.name.clone(), body);
        self
    }
    pub fn get(&self, name: &str) -> Option<&Body> {
        self.functions.get(name)
    }
    /// Record a function whose body could not be converted, with
    /// the reason conversion failed. See [].
    pub fn add_unconverted(
        &mut self,
        name: impl Into<String>,
        reason: impl Into<String>,
    ) -> &mut Self {
        self.unconverted.insert(name.into(), reason.into());
        self
    }
    /// The recorded conversion-failure reason for , if its
    /// body could not be converted.
    pub fn unconverted_reason(&self, name: &str) -> Option<&str> {
        self.unconverted.get(name).map(String::as_str)
    }
}
