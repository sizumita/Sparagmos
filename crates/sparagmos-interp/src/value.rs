//! The interpreter's value model.
//!
//! Scalars are stored widened (`i128` / `u128`); the declared width
//! and signedness of a place are applied from its [`crate::ir::Ty`]
//! when the value is written (see `normalize_for_ty` in `interp`).
//!
//! Composite and reference values (added for the print/format path):
//!
//! - [`Value::Str`] is the string data of `str` / `String` values and
//!   of `&str` values. The driver materialises `&str` constants to
//!   `ConstValue::Str`, so a string constant *is* this value; taking a
//!   reference to a place holding one yields a [`Value::Ref`] as usual.
//! - [`Value::Bytes`] is raw byte data: byte-array constants such as
//!   the `&[u8; N]` pieces template that `core::fmt` passes to
//!   `Arguments::new`.
//! - [`Value::ErasedBlock`] is a type-erased constant allocation: the
//!   bytes and provenance relocations of a constant whose reference
//!   declares a zero-sized pointee, preserved exactly until a pointer
//!   cast materialises the block into the cast target's type.
//! - [`Value::Ref`] is a reference (or raw pointer) resolved at
//!   creation time to the frame, local and constant projection it
//!   points at. Frames carry unique ids so a reference stays valid
//!   while its frame is on the stack, across calls in both
//!   directions. The stored projection never contains a `Deref` step:
//!   dereferences are followed eagerly when the reference is created.
//! - [`Value::FatRef`] is a fat pointer: a base value plus the slice
//!   length metadata that `PtrMetadata` reads. The base is the
//!   [`Value::Ref`] of an array place (from an unsizing cast) or the
//!   aggregate value itself (from a slice constant).
//! - [`Value::DynRef`] is a fat pointer to a `dyn Trait` value: a
//!   base value plus the vtable identity (concrete type name and
//!   trait bounds). A method call through it resolves the concrete
//!   type's impl body in the program — the vtable lookup, with the
//!   (concrete type, trait) pair standing in for the vtable pointer
//!   the driver does not materialise.
//! - [`Value::Enum`] is every ADT aggregate: structs use variant 0,
//!   enums carry their variant index. Field projections read
//!   `fields`, `Downcast` checks the variant, `Discriminant` reads it.
//! - [`Value::FmtArg`] is one `core::fmt::rt::Argument`: the reference
//!   handed to a `new_display` / `new_debug` / ... constructor, the
//!   formatting trait it selects, and the pointee type tag. Like the
//!   real `format_args!`, the pointee is read through the reference
//!   when the arguments are formatted, not when the argument is
//!   built. The formatter function pointer of the real type cannot
//!   exist here (its target is std code with no converted body), so
//!   the trait + type tag pair stands in for it; that pair is exactly
//!   what the formatter dispatch in core keys on.
//! - [`Value::FmtCount`] is an `rt::Argument` built by
//!   `Argument::from_usize`: a width/precision count, not a value.
//! - [`Value::FnPtr`] is a function item or function pointer value:
//!   the name of the target function in the program. Function items
//!   are zero-sized in rustc MIR and reifying one into a function
//!   pointer does not change the value, so both share this form.
//! - [`Value::Closure`] is a closure value: the closure body's name
//!   plus the captured environment in field order. A closure body
//!   receives a reference to this value as its first argument and
//!   reads its captures through field projections on it, exactly as
//!   it reads `(*env).<field>` in rustc MIR.
//!
//! - [`Value::Coroutine`] is a coroutine state object (the value of
//!   a future / generator): a state discriminant plus two storage
//!   areas the coroutine transform addresses separately. `fields`
//!   is the upvar storage - the captures the creation aggregate
//!   stored - addressed by field projections with no preceding
//!   downcast, in any state. `saved` holds one field vector per
//!   state, addressed by field projections through a downcast to
//!   that state: the body writes a state's fields through its
//!   downcast *before* the discriminant is set to that state, and
//!   setting the discriminant never clears either storage. State
//!   0 is the exception: the unresumed state's fields ARE the
//!   upvars, so a downcast to state 0 addresses `fields` too. The two
//!   areas are distinct even where their indices coincide (an
//!   async block's upvar at field 0 and its first saved field at
//!   (state, 0) are different slots); whenever a value lives in
//!   both, the transform moves it with an explicit copy.
//!
//! - [`Value::PanicPayload`] is a raw panic payload in flight: the
//!   value the panic runtime hands to `catch_unwind`'s catch
//!   function as a `*mut u8`. It exists only between the
//!   `catch_unwind` intrinsic (which produces it when the tried
//!   function panics) and `std::panicking::catch_unwind::cleanup`
//!   (which boxes it into a [`Value::DynRef`], the model's
//!   `Box<dyn Any + Send>`).
//!
//! - [`Value::FmtArguments`] is a built `std::fmt::Arguments` value:
//!   either a plain string (`from_str`) or a pieces template plus the
//!   snapshot of its argument array (`Arguments::new`).

use std::fmt;

use crate::ir::Ty;

/// One step of a resolved reference target. Indices are constants:
/// an `Index` projection through a local is evaluated against the
/// frame that syntactically contains the place when the reference is
/// created, which is the only frame it can refer to.
#[derive(Clone, Debug, PartialEq)]
pub enum RefProj {
    Field(usize),
    /// The length slot of a vector value: the position a
    /// reference to a `Vec`'s `len` field denotes (`&mut vec.len`
    /// — the reference `Vec::extend_with` hands its
    /// `SetLenOnDrop` guard, which writes the final length
    /// through it when it drops). The value model holds a
    /// vector as its element list, so the slot is not storage
    /// inside the value: reading the step answers the declared
    /// length, and writing it applies the `Vec::set_len`
    /// storage change to the vector value the preceding steps
    /// denote (see `assign_projected`). The step only ever
    /// appears as the last step of a target's projection:
    /// nothing projects through a `usize`.
    VecLen,
    /// A field access on a union value (the resolved form of
    /// `ProjectionElem::UnionField`): reading reinterprets the
    /// held field's bits as field `idx`'s type, through the
    /// scalar shapes of union table `table`.
    UnionField { idx: usize, table: usize },
    Index(usize),
    Downcast(u32),
    /// A sub-slice of the value's element sequence, resolved to
    /// an absolute offset and element count against the base
    /// value's length at resolution time. Reads materialise the
    /// sub-sequence; writes navigate into it element-wise.
    Subslice { offset: usize, len: usize },
    /// A byte displacement from the position the preceding steps
    /// denote, produced by pointer arithmetic (`BinaryOp::Offset`,
    /// the `arith_offset` intrinsic) whose result does not coincide
    /// with a structural position in the value tree — a `byte_add`
    /// into the middle of a scalar's or struct's storage is the
    /// canonical case. The step faithfully denotes the position (it
    /// participates in pointer identity and in further offsets,
    /// which accumulate). Reading or writing through it resolves
    /// through the interpreter's byte view of the target value's
    /// storage when the target is a place or static whose declared
    /// type and recorded field layouts determine the position's
    /// byte exactly; where they do not, the access is an explicit
    /// error at the use site. A zero displacement never appears;
    /// offset evaluation absorbs or drops it.
    ByteOffset(i128),
    /// The denomination a dereference recorded for the
    /// steps that follow it: those steps price
    /// through this type (the dereference's pointee), not
    /// through the type of the storage the target's root names,
    /// because the pointer's formation regrouped the storage's
    /// elements under the pointee's shape (a `[[u8; 3]; 2]`
    /// storage ranged over as `[[u8; 2]]`). The step does not
    /// move the position: it names the type of the value the
    /// preceding steps denote, and the byte view prices the
    /// steps after it through that type (see
    /// `Interpreter::view_span`). It is recorded only where the
    /// two denominations provably differ (see
    /// `Interpreter::denominate_fat_deref`); structural
    /// projections skip it, keeping the birth reading they have
    /// today for the consumers the byte view does not judge.
    View(Ty),
}

/// Where a [`Value::Ref`] points.
#[derive(Clone, Debug, PartialEq)]
pub enum RefTarget {
    /// A place in a live frame, identified by the frame's unique id.
    Place {
        frame: u64,
        local: usize,
        projection: Vec<RefProj>,
    },
    /// An immutable constant aggregate (the base of a constant slice
    /// or array), projected by the stored projection: the target is
    /// the sub-value of the aggregate the projection reaches (the
    /// projection is applied when the reference is resolved, as for
    /// the other targets). An empty projection denotes the whole
    /// aggregate. Reads resolve against the stored value; writes
    /// through it are an error. The (aggregate, projection) pair is
    /// the pointer's identity, which is what lets a value target
    /// denote one element of the aggregate — the element pointers
    /// a `slice::Iter` value carries, for instance.
    Value(Box<Value>, Vec<RefProj>),
    /// A `static` item's program-level storage, identified by the
    /// static's program key. The interpreter holds one value per
    /// static (see `Interpreter`'s statics store), seeded from the
    /// static's initializer; reads copy out of the store and writes
    /// store back, so a write through any reference to the static
    /// is observed by every later read — the read-after-write
    /// consistency `static mut` and interior-mutable statics rely
    /// on. The stored projection never contains a `Deref` step, as
    /// for [`RefTarget::Place`].
    Static {
        name: String,
        projection: Vec<RefProj>,
    },
    /// A heap allocation's storage in the interpreter's
    /// program-level allocation store, identified by the
    /// allocation id a `Box` value carries as its single field.
    /// A box's pointee lives in the store, not inside the box
    /// aggregate, so copying or moving the box value (into a
    /// callee frame, a pattern scrutinee, a return slot) copies
    /// only the id — the pointer — and a reference to the pointee
    /// denotes `(id, projection)` here, never a place in the
    /// frame that happened to hold a copy of the box. That is
    /// what lets a pointee reference outlive the frame that
    /// formed it, exactly as a real heap pointer does. Reads and
    /// writes resolve against the store entry; the entry is
    /// removed when the owning box's drop glue completes.
    Alloc {
        id: u64,
        projection: Vec<RefProj>,
    },
    /// The null pointer: denotes no place and no value. Produced by
    /// `ptr::null` / `ptr::null_mut`; equal only to itself, and any
    /// dereference of it is an explicit error.
    Null,
    /// A pointer known only by its address: a null or dangling
    /// pointer constant (a `without_provenance` address, an empty
    /// allocation's buffer pointer). It denotes no readable place;
    /// address identity is its only observable property, and
    /// address 0 is the null address (equal to `Null`).
    Address(u64),
}

impl RefTarget {
    /// A reference to a whole constant aggregate value (an empty
    /// projection).
    pub fn value(value: Value) -> RefTarget {
        RefTarget::Value(Box::new(value), Vec::new())
    }

    pub fn place(frame: u64, local: usize, projection: Vec<RefProj>) -> RefTarget {
        RefTarget::Place {
            frame,
            local,
            projection,
        }
    }

    /// This target with one more projection step appended: the
    /// target denoting the sub-value the step reaches within
    /// the value this target denotes. `None` for the targets
    /// that denote no storage (the null and address-only
    /// pointers).
    pub fn projected(&self, step: RefProj) -> Option<RefTarget> {
        match self {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                let mut projection = projection.clone();
                projection.push(step);
                Some(RefTarget::Place {
                    frame: *frame,
                    local: *local,
                    projection,
                })
            }
            RefTarget::Value(value, projection) => {
                let mut projection = projection.clone();
                projection.push(step);
                Some(RefTarget::Value(value.clone(), projection))
            }
            RefTarget::Static { name, projection } => {
                let mut projection = projection.clone();
                projection.push(step);
                Some(RefTarget::Static {
                    name: name.clone(),
                    projection,
                })
            }
            RefTarget::Alloc { id, projection } => {
                let mut projection = projection.clone();
                projection.push(step);
                Some(RefTarget::Alloc {
                    id: *id,
                    projection,
                })
            }
            RefTarget::Null | RefTarget::Address(_) => None,
        }
    }
}

/// Which formatting trait an `rt::Argument` was built for; mirrors
/// the `new_*` constructor family in `core::fmt::rt`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FmtTrait {
    Display,
    Debug,
    /// `new_debug_noop`: formats as the empty string.
    DebugNoop,
    Octal,
    LowerHex,
    UpperHex,
    Pointer,
    Binary,
    LowerExp,
    UpperExp,
}

/// A built `std::fmt::Arguments` value.
#[derive(Clone, Debug, PartialEq)]
pub enum FmtArguments {
    /// `Arguments::from_str`: printed verbatim.
    Plain(String),
    /// `Arguments::new`: the encoded pieces template (see the parser
    /// in `builtins`) plus the argument values, snapshotted from the
    /// argument array when `new` runs, as the real call snapshots
    /// the array reference's elements.
    Template { template: Vec<u8>, args: Vec<Value> },
}

/// A guest panic's payload: the concrete value the panic carries
/// (the argument of `panic_any`, the message of a `panic!`), with
/// the rendered name of its concrete type — the same rendering
/// `builtins::ty_tag` produces for a generic argument, which is
/// what `dyn Any` downcasts compare against. The interpreter's
/// panic error carries one of these from the panic site to the
/// `catch_unwind` that observes it.
#[derive(Clone, Debug, PartialEq)]
pub struct PanicPayload {
    /// Rendered concrete type name, e.g. `&str`,
    /// `std::string::String`, `i32`.
    pub ty: String,
    /// The payload value itself.
    pub value: Box<Value>,
    /// The payload type's `TypeId` segments, when the raise
    /// recorded them (a `panic_any` call's generic arguments, or
    /// the segments of the `dyn Any` box a resumed payload was
    /// read out of): the box the payload is later rebuilt into
    /// carries them, so a downcast of the caught payload
    /// compares exactly. `None` for payloads the interpreter
    /// fabricates itself (a string panic, an allocation
    /// failure), whose types are region-free scalars and
    /// strings in practice.
    pub type_id: Option<(u64, u64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// The slot was never written (or was moved out of).
    Uninit,
    Unit,
    Bool(bool),
    /// Signed integer, stored sign-extended to 128 bits.
    Int(i128),
    /// Unsigned integer (also `char`, stored as its code point).
    Uint(u128),
    /// A floating-point value, stored as `f64`. An `f32` value is
    /// stored as its exact `f64` widening; writes to an `f32` place
    /// round back (see `normalize_for_ty` in `interp`), and `f32`
    /// arithmetic rounds each result, which is exact for + - * / %
    /// by the double-rounding theorem (53 intermediate bits is more
    /// than twice 24).
    Float(f64),
    /// An `f128` value: the raw IEEE 754 binary128 bit pattern.
    /// Arithmetic on it is exact (see [`crate::f128`]); it is never
    /// rounded through `f64`.
    Float128(u128),
    /// A tuple, or an array (arrays ride as tuples of their elements;
    /// the length lives in the declared type).
    Tuple(Vec<Value>),
    /// String data: the value of `str`, `String` and `&str` places.
    Str(String),
    /// Raw bytes (byte-array constants, e.g. fmt pieces templates).
    Bytes(Vec<u8>),
    /// A type-erased constant allocation, not yet materialised
    /// into a type (see `crate::ir::Constant::ErasedAlloc`): the
    /// block's bytes verbatim, plus the values its provenance
    /// relocations decoded to. The block is the pointee's
    /// identity - two references to the same constant block are
    /// equal, as for [`RefTarget::Value`] snapshots - and the
    /// only operation on it is materialisation by a pointer
    /// cast (see `crate::ir::Rvalue::Materialize`); reading it
    /// as any other kind of value is an explicit error.
    ErasedBlock {
        bytes: Vec<Option<u8>>,
        relocs: Vec<(u32, Value)>,
    },
    /// A union value held as its storage bytes: the form a
    /// union takes once a write through one field must compose
    /// with bytes an earlier write through another field left
    /// behind (see `Interpreter::union_field_read` and the
    /// cross-field union write path in `interp`). Each entry
    /// is the byte the last write covering that offset stored,
    /// or `None` where no write has covered it (an
    /// uninitialised byte). A field read reinterprets the
    /// bytes through the field's scalar shape; a field write
    /// splices the written field's byte image over the prefix
    /// it covers. The vector's length is the widest span any
    /// conversion or write has accounted for so far; storage
    /// past it is uninitialised by definition. Unlike
    /// [`Value::ErasedBlock`] no relocations are recorded:
    /// pointer payloads round-trip through synthetic addresses
    /// exactly as they do for the aggregate union form (the
    /// scalar flattening path records the exposure).
    UnionBytes { bytes: Vec<Option<u8>> },
    /// A reference or raw pointer.
    Ref(RefTarget),
    /// A fat pointer (slice reference): base + element count.
    FatRef { base: Box<Value>, len: u128 },
    /// A fat pointer to a `dyn Trait` value: the data pointer plus
    /// the vtable identity. The driver does not materialise vtables,
    /// so the vtable is represented by what it is a table *for*: the
    /// concrete type's name (matching the `Self` spelling of its
    /// impl method names in the program) and the trait bounds the
    /// pointer was formed for. `base` is the data pointer in the
    /// shape the sized value had: a [`Value::Ref`] for a reference
    /// coercion, a [`Value::FatRef`] for a slice pointee, the box
    /// aggregate itself for a `Box` coercion.
    DynRef {
        base: Box<Value>,
        concrete: String,
        traits: Vec<String>,
        /// The concrete type's drop glue, attached at the unsizing
        /// cast that formed this value (the vtable's drop shim, in
        /// this model). `None` when conversion could not account
        /// for the concrete type's glue: dropping the value is then
        /// an explicit error.
        drop_glue: Option<Box<crate::ir::DropGlue>>,
        /// The concrete type's `TypeId` segments, attached where
        /// the value was formed (an unsizing cast, a decoded
        /// constant, a vtable identity, a rebuilt panic payload):
        /// `concrete` is the type's rendered tag, which erases
        /// regions, so two region variants of one type share it;
        /// the segments are the identity `TypeId::of` itself
        /// compares, and a `dyn Any` downcast compares them when
        /// the call site recorded the target type's segments too.
        /// `None` when the forming site could not name them.
        type_id: Option<(u64, u64)>,
    },
    /// An ADT value: variant index + field values.
    Enum { variant: u32, fields: Vec<Value> },
    /// A `Vec` value whose declared length differs from the
    /// element list the value model materialises, in either
    /// direction. `elems` holds the stored elements. When `len`
    /// runs past them — the shape a `Vec::set_len` growth past
    /// the materialisation limit produces (the element list is
    /// stored value by value, so an astronomical length is not
    /// a value this model can hold element by element) — every
    /// slot from `elems.len()` up to `len` is uninitialised.
    /// When the stored elements run past `len` — the shape a
    /// length shrink (`set_len`, a length-slot write) produces —
    /// the tail past `len` is the vector's spare capacity: the
    /// slots keep their values and stay reachable as storage
    /// (through the buffer pointer, and as the slots the next
    /// append overwrites), but they are not elements: length
    /// questions, slice views, equality and the drop glue all
    /// stop at `len`, and the capacity the model reports is
    /// the storage extent, `max(len, elems.len())`. The
    /// unmaterialised slots behave exactly like the
    /// uninitialised slots a materialised `set_len` leaves:
    /// reading one is an uninitialised read, dropping one is a
    /// no-op, and only the length, the stored elements, and
    /// bounded prefixes of the whole are observable.
    ///
    /// `cap` is the storage extent the value accounts beyond
    /// what `len` and the stored list already imply: the
    /// capacity a `Vec::with_capacity` past the materialisation
    /// limit requested, which the value holds as accounted
    /// storage without holding it element by element. It is
    /// always at least `max(len, elems.len())` — every
    /// constructor and rewrite maintains that — so the storage
    /// extent (see `vec_stored_len`) is `cap` itself, and the
    /// slots from the stored end up to `cap` are spare
    /// capacity: reachable through the buffer pointer as
    /// uninitialised slots, exactly like the slots a
    /// materialised `with_capacity` holds. Without the record,
    /// a length shrink would collapse the reported capacity
    /// onto the new length and break the `len < capacity`
    /// invariant guest code relies on (the assertion inside
    /// `Vec::pop`, for one).
    SparseVec {
        elems: Vec<Value>,
        len: u128,
        cap: u128,
    },
    /// A `core::fmt::rt::Argument` formatting a pointee.
    FmtArg {
        /// The reference passed to the `new_*` constructor.
        target: Box<Value>,
        trait_: FmtTrait,
        /// Rendered pointee type (the constructor's generic
        /// argument), e.g. `i32`, `&str`, `char`.
        ty: String,
    },
    /// An `rt::Argument` count (`Argument::from_usize`).
    FmtCount(u16),
    /// A built `std::fmt::Arguments`.
    FmtArguments(FmtArguments),
    /// A `core::fmt::Formatter` that captures what guest code
    /// writes: the id of its state in the interpreter's formatter
    /// table. A guest `Debug`/`Display` implementation body
    /// receives this value behind its `&mut Formatter` parameter
    /// when the fmt engine runs the body (see `builtins`); the
    /// `Formatter` methods are builtins that append to the
    /// captured buffer.
    Formatter(u64),
    /// A `core::fmt` builder under construction (`DebugList`,
    /// `DebugSet`, `DebugMap`, `DebugStruct`, `DebugTuple`): the
    /// id of its state in the interpreter's builder table. The
    /// builder methods are builtins over this handle, so no
    /// builder aggregate ever materializes.
    FmtBuilder(u64),
    /// A C variable argument list (`std::ffi::VaList`): the id of
    /// its state in the interpreter's va-list table. The state is
    /// the ordered tail arguments of one variadic call plus the
    /// cursor `va_arg` reads from; the platform `va_list` struct
    /// (register save areas and offsets) is not modelled, only
    /// the argument sequence it walks. Cloning a list (C
    /// `va_copy`) registers a second state with the same cursor,
    /// so the two advance independently.
    VaList(u64),
    /// A function item or function pointer value: the name of the
    /// target function in the program.
    FnPtr(String),
    /// A closure value: the closure body's name plus the captured
    /// environment, in the field order of the closure aggregate.
    Closure { def: String, env: Vec<Value> },
    /// A coroutine state object: the state discriminant (0 =
    /// unresumed, then one per suspension point, then the returned
    /// and panicked states - the exact numbering is the coroutine
    /// layout's, read back by the body's entry `SwitchInt`), the
    /// upvar storage `fields`, and the per-state saved storage
    /// `saved` (indexed by state; entry 0 is never used - the
    /// unresumed state's fields are the upvars in `fields` - and
    /// states with no saved fields, including the returned and
    /// panicked states, hold an empty or absent vector). See the
    /// module docs for how this differs from [`Value::Enum`].
    Coroutine {
        variant: u32,
        fields: Vec<Value>,
        saved: Vec<Vec<Value>>,
    },
    /// A raw panic payload in flight between the `catch_unwind`
    /// intrinsic and `std::panicking::catch_unwind::cleanup` (see
    /// the module docs). Anywhere else the value is meaningless;
    /// operations on it are explicit errors.
    PanicPayload(PanicPayload),
}

impl Value {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Value::Uninit => "uninitialized",
            Value::Unit => "unit",
            Value::Bool(_) => "bool",
            Value::Int(_) => "signed integer",
            Value::Uint(_) => "unsigned integer",
            Value::Float(_) => "float",
            Value::Float128(_) => "f128",
            Value::Tuple(_) => "tuple",
            Value::Str(_) => "string",
            Value::Bytes(_) => "bytes",
            Value::ErasedBlock { .. } => "erased constant block",
            Value::UnionBytes { .. } => "union byte storage",
            Value::Ref(_) => "reference",
            Value::FatRef { .. } => "slice reference",
            Value::DynRef { .. } => "dyn reference",
            Value::Enum { .. } => "ADT value",
            Value::SparseVec { .. } => "vector value",
            Value::FmtArg { .. } => "fmt argument",
            Value::FmtCount(_) => "fmt count",
            Value::FmtArguments(_) => "fmt arguments",
            Value::Formatter(_) => "fmt formatter",
            Value::FmtBuilder(_) => "fmt builder",
            Value::VaList(_) => "va list",
            Value::FnPtr(_) => "function pointer",
            Value::Closure { .. } => "closure",
            Value::Coroutine { .. } => "coroutine",
            Value::PanicPayload(_) => "panic payload",
        }
    }

    /// Whether two values are identical bit for bit: the
    /// derived `PartialEq` comparison, except that a float
    /// compares by its bit pattern, recursively through
    /// every aggregate that can hold one. The derived
    /// equality is the guest's `==`, under which a NaN is
    /// unequal to everything including itself (and -0.0
    /// equals 0.0), so it cannot witness that a value read
    /// back from storage is the very value that was
    /// snapshotted from it: a NaN read back from its own
    /// storage is bit-identical to the snapshot, while a
    /// -0.0 in storage is not the 0.0 a canonicalised
    /// snapshot may hold. Callers that need an identity
    /// witness rather than the guest's equality use this;
    /// everything else keeps `==`.
    pub fn bit_identical(&self, other: &Value) -> bool {
        fn elems_identical(a: &[Value], b: &[Value]) -> bool {
            a.len() == b.len()
                && a.iter().zip(b.iter()).all(|(x, y)| x.bit_identical(y))
        }
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
            (Value::Tuple(a), Value::Tuple(b)) => elems_identical(a, b),
            (
                Value::Enum { variant: va, fields: fa },
                Value::Enum { variant: vb, fields: fb },
            ) => va == vb && elems_identical(fa, fb),
            (
                Value::SparseVec { elems: ea, len: la, cap: ca },
                Value::SparseVec { elems: eb, len: lb, cap: cb },
            ) => la == lb && ca == cb && elems_identical(ea, eb),
            (
                Value::FatRef { base: ba, len: la },
                Value::FatRef { base: bb, len: lb },
            ) => la == lb && ba.bit_identical(bb),
            (
                Value::DynRef { base: ba, concrete: ca, traits: ta, drop_glue: ga, type_id: ia },
                Value::DynRef { base: bb, concrete: cb, traits: tb, drop_glue: gb, type_id: ib },
            ) => ca == cb && ta == tb && ga == gb && ia == ib && ba.bit_identical(bb),
            (
                Value::ErasedBlock { bytes: ba, relocs: ra },
                Value::ErasedBlock { bytes: bb, relocs: rb },
            ) => {
                ba == bb
                    && ra.len() == rb.len()
                    && ra.iter().zip(rb.iter()).all(|((oa, va), (ob, vb))| {
                        oa == ob && va.bit_identical(vb)
                    })
            }
            (
                Value::FmtArg { target: ta, trait_: tra, ty: tya },
                Value::FmtArg { target: tb, trait_: trb, ty: tyb },
            ) => tra == trb && tya == tyb && ta.bit_identical(tb),
            (
                Value::FmtArguments(FmtArguments::Template {
                    template: ta,
                    args: aa,
                }),
                Value::FmtArguments(FmtArguments::Template {
                    template: tb,
                    args: ab,
                }),
            ) => ta == tb && elems_identical(aa, ab),
            (
                Value::Closure { def: da, env: ea },
                Value::Closure { def: db, env: eb },
            ) => da == db && elems_identical(ea, eb),
            (
                Value::Coroutine { variant: va, fields: fa, saved: sa },
                Value::Coroutine { variant: vb, fields: fb, saved: sb },
            ) => {
                va == vb
                    && elems_identical(fa, fb)
                    && sa.len() == sb.len()
                    && sa.iter()
                        .zip(sb.iter())
                        .all(|(xa, xb)| elems_identical(xa, xb))
            }
            (Value::PanicPayload(pa), Value::PanicPayload(pb)) => {
                pa.ty == pb.ty && pa.value.bit_identical(&pb.value)
            }
            _ => self == other,
        }
    }

    /// View as a boolean, if this value is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The bit pattern used by `SwitchInt`: booleans as 0/1, integers
    /// as their stored pattern. Anything else cannot be switched on.
    pub fn switch_bits(&self) -> Option<u128> {
        match self {
            Value::Bool(b) => Some(*b as u128),
            Value::Int(v) => Some(*v as u128),
            Value::Uint(v) => Some(*v),
            _ => None,
        }
    }

    /// Number of elements of an aggregate/byte/string value, for
    /// `Len` and index resolution. `Str` counts bytes, as MIR `Len`
    /// on a `str` place does.
    pub fn len(&self) -> Option<u128> {
        match self {
            Value::Tuple(items) => Some(items.len() as u128),
            Value::Bytes(b) => Some(b.len() as u128),
            Value::Str(s) => Some(s.len() as u128),
            Value::Enum { fields, .. } => Some(fields.len() as u128),
            Value::SparseVec { len, .. } => Some(*len),
            Value::FatRef { len, .. } => Some(*len),
            _ => None,
        }
    }
}

impl fmt::Display for RefTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                write!(f, "frame#{frame} local _{local}")?;
                for proj in projection {
                    match proj {
                        RefProj::Field(i) => write!(f, ".{i}")?,
                        RefProj::UnionField { idx, .. } => write!(f, ".{idx}")?,
                        RefProj::Index(i) => write!(f, "[{i}]")?,
                        RefProj::Downcast(v) => write!(f, " as variant {v}")?,
                        RefProj::Subslice { offset, len } => {
                            write!(f, "[{offset}..{}]", offset + len)?
                        }
                        RefProj::ByteOffset(d) => write!(f, " +byte {d}")?,
                        RefProj::VecLen => write!(f, ".len")?,
                        RefProj::View(ty) => write!(f, " as view {ty:?}")?,
                    }
                }
                Ok(())
            }
            RefTarget::Value(..) => write!(f, "constant aggregate"),
            RefTarget::Static { name, projection } => {
                write!(f, "static {name}")?;
                for proj in projection {
                    match proj {
                        RefProj::Field(i) => write!(f, ".{i}")?,
                        RefProj::UnionField { idx, .. } => write!(f, ".{idx}")?,
                        RefProj::Index(i) => write!(f, "[{i}]")?,
                        RefProj::Downcast(v) => write!(f, " as variant {v}")?,
                        RefProj::Subslice { offset, len } => {
                            write!(f, "[{offset}..{}]", offset + len)?
                        }
                        RefProj::ByteOffset(d) => write!(f, " +byte {d}")?,
                        RefProj::VecLen => write!(f, ".len")?,
                        RefProj::View(ty) => write!(f, " as view {ty:?}")?,
                    }
                }
                Ok(())
            }
            RefTarget::Alloc { id, projection } => {
                write!(f, "allocation #{id}")?;
                for proj in projection {
                    match proj {
                        RefProj::Field(i) => write!(f, ".{i}")?,
                        RefProj::UnionField { idx, .. } => write!(f, ".{idx}")?,
                        RefProj::Index(i) => write!(f, "[{i}]")?,
                        RefProj::Downcast(v) => write!(f, " as variant {v}")?,
                        RefProj::Subslice { offset, len } => {
                            write!(f, "[{offset}..{}]", offset + len)?
                        }
                        RefProj::ByteOffset(d) => write!(f, " +byte {d}")?,
                        RefProj::VecLen => write!(f, ".len")?,
                        RefProj::View(ty) => write!(f, " as view {ty:?}")?,
                    }
                }
                Ok(())
            }
            RefTarget::Null => write!(f, "null pointer"),
            RefTarget::Address(addr) => write!(f, "address pointer {addr:#x}"),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Uninit => write!(f, "<uninit>"),
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(v) => write!(f, "{v}"),
            Value::Uint(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Float128(bits) => write!(f, "f128:{bits:#034x}"),
            Value::Tuple(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                if items.len() == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")
            }
            Value::Str(s) => write!(f, "{s:?}"),
            Value::Bytes(b) => write!(f, "b\"{}\"", String::from_utf8_lossy(b)),
            Value::ErasedBlock { bytes, .. } => {
                write!(f, "<erased block ({} bytes)>", bytes.len())
            }
            Value::UnionBytes { bytes } => {
                write!(f, "<union bytes ({} bytes)>", bytes.len())
            }
            Value::Ref(target) => write!(f, "&({target})"),
            Value::FatRef { len, .. } => write!(f, "&[_; {len}]"),
            Value::DynRef { traits, .. } => {
                write!(f, "&dyn {}", traits.first().map(String::as_str).unwrap_or("?"))
            }
            Value::Enum { variant, fields } => {
                write!(f, "variant {variant} (")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{field}")?;
                }
                write!(f, ")")
            }
            Value::SparseVec { elems, len, cap } => {
                write!(f, "vec[{len} ({} stored, cap {cap})]", elems.len())
            }
            Value::FmtArg { ty, trait_, .. } => {
                write!(f, "<fmt {trait_:?} argument of {ty}>")
            }
            Value::FmtCount(n) => write!(f, "<fmt count {n}>"),
            Value::FmtArguments(_) => write!(f, "<fmt::Arguments>"),
            Value::Formatter(_) => write!(f, "<fmt::Formatter>"),
            Value::FmtBuilder(_) => write!(f, "<fmt builder>"),
            Value::VaList(_) => write!(f, "<va list>"),
            Value::FnPtr(name) => write!(f, "<fn {name}>"),
            Value::Closure { def, env } => {
                write!(f, "<closure {def} ({} captures)>", env.len())
            }
            Value::Coroutine { variant, fields, .. } => {
                write!(f, "<coroutine state {variant} ({} fields)>", fields.len())
            }
            Value::PanicPayload(payload) => {
                write!(f, "<panic payload of type {}>", payload.ty)
            }
        }
    }
}
