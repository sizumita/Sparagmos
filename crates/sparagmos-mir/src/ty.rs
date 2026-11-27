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

/// One thread-local static of a program: its declared type and its
/// decoded initializer — the value each thread's storage starts
/// from. For `std`'s thread-local storages this is the storage
/// struct's `new()` value, not the user-visible initial value,
/// which the storage's own `get_or_init` machinery produces on
/// first access. The driver records one entry per `ThreadLocalRef`
/// target it converts, decoding the static's initializer
/// allocation against its declared type, exactly as for a plain
/// static's seed. A thread-local whose initializer cannot be
/// decoded gets no entry; evaluating its `ThreadLocalRef` then
/// fails with an explicit error.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadLocalDef {
    pub ty: Ty,
    pub init: crate::ConstValue,
}

/// Thread-local static definitions of a program, keyed by the
/// `DefId` of the static item.
pub type ThreadLocals = std::collections::HashMap<DefId, ThreadLocalDef>;

/// One driver-recorded seed of a static no converted body
/// references: its declared type and its decoded initializer,
/// in the shape [`ThreadLocalDef`] carries for thread-locals.
/// The ordinary seed path reaches a static only through a
/// converted reference to it; a static whose only readers are
/// functions with no MIR body in the shipped libraries (core's
/// generated unicode skiplist tables, read solely by the
/// property modules' `lookup_slow`) is never referenced, so
/// the driver records its seed directly, keyed by the static's
/// name — the key the interpreter's seed table and the runtime
/// model that reads the static both use.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticSeedDef {
    pub ty: Ty,
    pub init: crate::ConstValue,
}

/// Driver-recorded static seeds, keyed by the static's name
/// (see [`StaticSeedDef`]).
pub type ExtraStaticSeeds = std::collections::HashMap<String, StaticSeedDef>;

/// Declared types of a program's statics, keyed by the `DefId`
/// of the static item. The interpreter consults a static's
/// declared type when a pointer offset over the static must be
/// denominated in the type's element size (a byte displacement
/// into a static array, say): the static's seed value alone
/// does not carry the element width. Statics the driver could
/// not record (foreign items) have no entry and keep the
/// value-shape behaviour.
pub type StaticTys = std::collections::HashMap<DefId, Ty>;

/// Declared alignments of a program's statics, in bytes, keyed
/// by the `DefId` of the static item: the alignment the item's
/// own attributes demand (`#[rustc_align_static(N)]`), which can
/// exceed the declared type's alignment. Only statics carrying
/// such an attribute have an entry; every other static's
/// alignment is its type's. The interpreter prices a static's
/// synthetic address from the greater of the two, as a native
/// linker places the storage at the greater alignment.
pub type StaticAligns = std::collections::HashMap<DefId, u64>;

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

/// Per-variant field byte layouts of the monomorphic ADT
/// instances in a program: `variants[v][f]` is the (byte
/// offset, byte size) of field `f` of variant `v`, fields in
/// declaration order, as the compiler's layout query gives
/// them. Keyed like [`AdtLayouts`] by the rendered instance
/// name. The driver fills one entry per instance it records a
/// layout for; unions get no entry (their fields share one
/// offset, and the value model reads them through its own
/// reinterpretation machinery). The interpreter's byte view of
/// a value's storage (raw byte pointer reads and writes into
/// an ADT value) resolves byte positions through this table:
/// a position inside one field's span is that field's byte,
/// and a position in the padding between fields has no byte
/// to read.
pub type AdtFieldLayouts = std::collections::HashMap<String, Vec<Vec<(u64, u64)>>>;

/// How the interpreter composes the layout of one dynamically
/// sized ADT instance (a struct or union whose one unsized
/// field is the tail) at runtime: the size of a value is the
/// tail field's offset plus the tail's own dynamic size, and
/// the tail's offset is the recorded sized-prefix end rounded
/// up to the tail's effective alignment (its dynamic alignment
/// capped by the repr's pack, if any). The driver records one
/// entry per unsized instance it converts; the instance itself
/// gets no [`AdtLayouts`] entry, because its size is not a
/// constant. Keyed like [`AdtLayouts`] by the rendered instance
/// name.
#[derive(Clone, Debug, PartialEq)]
pub struct AdtDstLayout {
    /// Byte offset just past the last sized field, in the
    /// layout the compiler computes (the prefix fields laid out
    /// among themselves, the tail not yet placed): for a struct,
    /// the offset the tail field's placement starts from; for a
    /// union, the largest sized field's size.
    pub prefix_end: u64,
    /// The sized prefix's alignment contribution: the largest
    /// effective alignment among the sized fields (each capped
    /// by `pack`), raised by a `repr(align)` if the ADT has one.
    pub prefix_align: u64,
    /// The `repr(packed(N))` alignment cap, if the ADT has one.
    pub pack: Option<u64>,
    /// Declaration-order index of the one unsized field.
    pub tail_field: u32,
    /// The unsized field's type, rendered as a type tag the way
    /// [`AdtLayouts`] keys are rendered: the interpreter
    /// recurses into the tail with this tag.
    pub tail_ty: String,
    /// The ADT is a union: its fields share offset 0, so the
    /// size is the larger of the prefix end and the tail's size
    /// rather than their sum.
    pub union: bool,
}

/// Per-instance dynamically-sized ADT layout descriptors,
/// keyed like [`AdtLayouts`] (see [`AdtDstLayout`]).
pub type AdtDstLayouts = std::collections::HashMap<String, AdtDstLayout>;

/// Storage layouts of the monomorphic tuple types in a
/// program: the tuple's total storage size in bytes and, per
/// field in declaration order, the field's (byte offset, byte
/// size) within that storage, as the compiler's layout query
/// gives them. Keyed by the tuple type's rendered name (its
/// `ty_tag` rendering, e.g. `(u8, u16, u8)`). A tuple's field
/// order in storage is the compiler's choice — fields may be
/// reordered and padding inserted — so a bit-level reading of
/// a tuple (a `transmute` through the interpreter's scalar
/// shapes) is sound only at these recorded offsets; a tuple
/// type with no entry has no such reading. The driver fills
/// one entry per tuple type it converts whose layout the
/// compiler can give; generic or unsized tuple types get no
/// entry.
pub type TupleLayouts = std::collections::HashMap<String, (u64, Vec<(u64, u64)>)>;

/// The stored discriminant tag of one monomorphic enum
/// instance whose layout uses the direct tag encoding: the
/// tag is an integer scalar of `size` bytes at byte offset
/// `offset` in the value's storage, and variant `v` is stored
/// with the tag value `discriminants[v]`, held as a bit
/// pattern (the stored bytes are its low `size` bytes,
/// little-endian). Keyed by the rendered instance name, like
/// [`AdtLayouts`]. Only enums with a direct tag get an entry:
/// a niche-encoded enum has no tag storage of its own (its
/// discriminant lives inside a payload field's value range,
/// see [`AdtNiches`]), and a single-variant enum stores no
/// tag at all. The interpreter's byte view reads a byte in
/// the tag span as the discriminant's own byte and splices a
/// write there back into the value's variant.
#[derive(Clone, Debug, PartialEq)]
pub struct EnumTag {
    /// Byte offset of the tag within the enum's storage.
    pub offset: u64,
    /// Width in bytes of the stored tag scalar.
    pub size: u64,
    /// Discriminant value of each variant, by variant index.
    pub discriminants: Vec<u128>,
}

/// Per-instance direct tag layouts, keyed like [`AdtLayouts`].
pub type AdtTagLayouts = std::collections::HashMap<String, EnumTag>;

/// The storage-slot map of one monomorphic coroutine instance:
/// which slot of the interpreter's flat coroutine field vector
/// each state field of the compiler's state-machine layout lives
/// in. Keyed like [`AdtLayouts`] by the rendered instance name.
///
/// A coroutine value's fields are stored positionally: slots
/// `0..n_upvars` hold the upvars in creation order (the creation
/// aggregate fills them in that order, and bare `Field`
/// projections - how transformed bodies read upvars - stay
/// positional), and the state fields take the following slots.
/// The compiler's layout gives every state field one absolute
/// offset, the same in every variant that contains it, so slots
/// are assigned per distinct offset across all variants: two
/// variant fields at one offset (above all the `ResumeTy` slot,
/// present at the head of every suspension state) are one storage
/// cell, and fields at distinct offsets never alias even when
/// their variant-relative field indices coincide. Within one
/// variant, though, a zero-sized field shares its offset with
/// the field after it, and those two are distinct cells: the
/// offset's slot belongs to the later field, and the zero-sized
/// field gets a private slot no other variant uses. The map
/// carries offsets, not field sizes, so it cannot express the
/// same separation across variants: a zero-sized field can be
/// the only occupant of its offset in one state while that
/// offset belongs to a differently-typed field in another
/// state, and the map then hands it that field's shared slot.
/// Conversion therefore moves a field its type proves
/// zero-sized to a private slot of its own (see
/// `Converter::coroutine_field_slot` in sparagmos-interp): in
/// the real layout the field's stores write no bytes, so they
/// must not overwrite the other field's cell in the flat
/// vector either. The driver
/// fills one entry per coroutine instance whose layout the
/// compiler can give; an instance with no entry keeps the
/// positional projection conversion.
#[derive(Clone, Debug, PartialEq)]
pub struct CoroutineSlots {
    /// `variants[v][f]` is the storage slot of variant `v`'s
    /// field `f`, in the variant's own field order.
    pub variants: Vec<Vec<u32>>,
}

/// Storage-slot maps of a program's coroutine instances (see
/// [`CoroutineSlots`]).
pub type CoroutineLayouts = std::collections::HashMap<String, CoroutineSlots>;

/// The derived-`Debug` shape of one monomorphic ADT instance: the
/// names and field structure a `#[derive(Debug)]` expansion prints,
/// recorded without any function body. Keyed like [`AdtLayouts`]
/// by the rendered instance name - the type tag the interpreter's
/// formatter sees - because the field *types* (which the formatter
/// needs to format nested values with their own types known)
/// differ per monomorphization even though the names do not.
///
/// The driver fills one entry per monomorphic struct/enum instance
/// it converts (see `Converter::record_debug_shape` in the driver);
/// the interpreter's `Debug` formatter consults the table when it
/// meets an ADT value. An instance with no entry stays explicitly
/// unsupported in the formatter. Unions get no entry: `Debug` is
/// never derived for them, so formatting a union value is always a
/// handwritten impl, which this table does not model.
pub type AdtDebugShapes = std::collections::HashMap<String, AdtDebugShape>;

/// Derived-`Debug` shape of one ADT instance (see
/// [`AdtDebugShapes`]).
#[derive(Clone, Debug, PartialEq)]
pub struct AdtDebugShape {
    pub kind: AdtShapeKind,
    pub variants: Vec<AdtDebugVariant>,
}

/// One variant of an [`AdtDebugShape`]: its printed name (for a
/// struct, the type name) and its fields in declaration order.
#[derive(Clone, Debug, PartialEq)]
pub struct AdtDebugVariant {
    pub name: String,
    pub fields: Vec<AdtDebugField>,
}

/// The debugger shape of one monomorphic closure or coroutine
/// instance: the environment type name and field structure the
/// compiler's debuginfo describes, recorded without any function
/// body. Keyed like [`AdtDebugShapes`] by the rendered instance
/// name - the type tag the debugger interface sees.
///
/// The driver fills one entry per monomorphic closure/coroutine
/// instance it converts (see `Converter::record_env_shape` in
/// the driver); the debugger interface consults the table when
/// it prints a closure or coroutine value. An environment field
/// the instance body's debug info does not name (the compiler
/// emits no entry for a zero-sized capture) is recorded with
/// its type recovered from the instance's upvars tuple and a
/// `None` name. An instance with no entry stays unrenderable
/// in the debugger, exactly as an ADT with no debug shape stays
/// unformattable.
pub type EnvDebugShapes = std::collections::HashMap<String, EnvDebugShape>;

/// Debugger shape of one closure/coroutine instance (see
/// [`EnvDebugShapes`]).
#[derive(Clone, Debug, PartialEq)]
pub struct EnvDebugShape {
    /// The environment type's DWARF name: the definition path of
    /// the closure/coroutine with its final segment in the
    /// `{<kind>_env#<disambiguator>}` form the compiler's
    /// debuginfo uses (e.g. `crate::main::{closure_env#0}`).
    pub dwarf_name: String,
    /// The captured variables (upvars) in environment field
    /// order, with the names the compiler's debuginfo gives them
    /// (a by-reference capture's name carries its `_ref__`
    /// prefix). A coroutine's every state renders these after the
    /// state's own saved fields.
    pub upvars: Vec<AdtDebugField>,
    /// A coroutine's states in discriminant order (empty for a
    /// closure): each state's saved fields, not including the
    /// upvars (see `upvars`).
    pub variants: Vec<EnvDebugVariant>,
    /// Whether the variants' `slots` index the coroutine value's
    /// flat field vector (the instance has a storage-slot map in
    /// [`CoroutineLayouts`], and its saved storage lives there).
    /// When false, a variant's fields are read positionally from
    /// the value's per-state saved storage instead, and `slots`
    /// is the identity.
    pub slotted: bool,
}

/// One state of an [`EnvDebugShape`]: its printed name (the
/// DWARF variant name - `Unresumed`, `Returned`, `Panicked`, or
/// `Suspend<N>`) and its saved fields in layout order.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvDebugVariant {
    pub name: String,
    pub fields: Vec<AdtDebugField>,
    /// The storage index of each field, parallel to `fields`:
    /// the flat field-vector slot when the shape is `slotted`,
    /// else the field's position in the per-state saved vector.
    pub slots: Vec<u32>,
}

/// One field of an [`AdtDebugVariant`].
#[derive(Clone, Debug, PartialEq)]
pub struct AdtDebugField {
    /// The field name of a named-field variant; `None` for a
    /// tuple field.
    pub name: Option<String>,
    /// The rendered type tag of the field's type in this
    /// monomorphization (the same rendering [`AdtLayouts`] keys
    /// use), so the formatter can format the field's value with
    /// its type known.
    pub ty: String,
}
/// The niche discriminant encoding of one monomorphic enum
/// instance, as the compiler's layout describes it: the
/// payload-carrying variant (`untagged_variant`) has a field
/// whose invalid values — its niche — encode the variants
/// `niche_variants.0..=niche_variants.1`; variant
/// `niche_variants.0 + k` is stored as the field value
/// `niche_start + k`, and the untagged variant stores its
/// field's own value. `tag_field` is the index of the
/// niche-carrying field in the enum layout's top-level field
/// list, `tag_bits` the width of the scalar stored there, and
/// `niche_offset` its byte offset within the enum. Keyed by
/// the rendered instance name, like [`AdtLayouts`]. Only
/// enums whose layout uses this encoding get an entry.
#[derive(Clone, Debug, PartialEq)]
pub struct EnumNiche {
    /// Variant index of the payload-carrying variant.
    pub untagged_variant: u32,
    /// First and last variant indices encoded in the niche.
    pub niche_variants: (u32, u32),
    /// Field value encoding `niche_variants.0`.
    pub niche_start: u128,
    /// Index of the niche-carrying field in the enum layout.
    pub tag_field: u32,
    /// Width in bits of the scalar stored at the niche.
    pub tag_bits: u32,
    /// Byte offset of the niche scalar within the enum.
    pub niche_offset: u64,
}

/// Per-instance niche encodings, keyed like [`AdtLayouts`].
pub type AdtNiches = std::collections::HashMap<String, EnumNiche>;

/// The kind of an ADT, for shape purposes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdtShapeKind {
    Struct,
    Enum,
    Union,
}

/// The declared shape of one ADT: per variant, the field types in
/// terms of the definition's generic parameters (`Ty::Param`
/// indices refer to the definition's parameter list). The driver
/// records one entry per ADT it converts, alongside the
/// discriminant table. Consumers use the shape to reason about an
/// ADT's values without a layout engine — in particular whether an
/// instantiated type is zero-sized, in which case rustc MIR may
/// never write its places at all.
#[derive(Clone, Debug, PartialEq)]
pub struct AdtShape {
    pub kind: AdtShapeKind,
    pub variants: Vec<Vec<Ty>>,
}

/// ADT shape tables for a program, keyed by the `DefId` of the ADT
/// (see [`AdtShape`]).
pub type AdtShapes = std::collections::HashMap<AdtDefId, AdtShape>;

/// Declared field types of one *monomorphic instance* of an ADT,
/// keyed by the rendered instance name (the same key as
/// [`AdtLayouts`]). The driver records an entry only for instances
/// whose generic shape (see [`AdtShapes`]) mentions a type alias:
/// the shape's field types are the definition's own, so a
/// projection alias among them (the `NonZero<T>` field is
/// `<T as ZeroablePrimitive>::NonZeroInner`) reaches the
/// interpreter unresolved, and the interpreter has no trait solver
/// to resolve it with. An instance entry's field types are the
/// compiler's own instantiation of the declared fields with the
/// instance's arguments, which normalizes such projections to the
/// concrete types they denote.
pub type AdtInstanceShapes = std::collections::HashMap<String, AdtShape>;

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
