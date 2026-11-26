//! Value shape helpers and typed normalization.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};

/// The channel id carried by an mpsc handle value, if the value
/// has the handle shape (a one-field aggregate holding the id).
/// Whether a value is a scalar: a boolean, an integer or a
/// float. A scalar never holds a handle's count, so a scalar
/// found where a handle's drop glue runs is a builtin-modelled
/// stand-in for the handle -- the thread model's `Thread` value
/// is an id, although the declared type's field is an `Arc` --
/// and its release is not count-based. Drop glue built from a
/// declared type (the shape-table fallback for a variant no
/// projection evidences) can name a handle at such a position;
/// the model's scalar there has nothing to release.
pub(crate) fn is_scalar_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Bool(_)
            | Value::Int(_)
            | Value::Uint(_)
            | Value::Float(_)
            | Value::Float128(_)
    )
}

/// The allocation a `Box` value points at, if the value has the
/// shape the construction builtins produce (see
/// [`Interpreter::new_box`]): the allocation id and the
/// projection the box's pointer itself carries (empty for a
/// freshly built box; a pointer extracted from a box and stored
/// on can carry one). Only a box's own field 0 is ever an
/// allocation pointer in this model, so this shape test is
/// exact: a one-field aggregate whose field is an allocation
/// reference is a built box, and projecting its field 0 denotes
/// the pointee, not the pointer slot.
/// The `mem::Alignment` value for a byte alignment, in the
/// shape the conversion produces for a `usize` -> `Alignment`
/// transmute: the variant whose discriminant is the alignment
/// (the variants are the powers of two in order), wrapped as
/// the struct's single field. An alignment with no variant
/// (not a power of two, or beyond the enum's range) keeps the
/// plain integer shape.
pub(crate) fn alignment_value(align: u128) -> Value {
    if align >= 1 && align.is_power_of_two() {
        let exponent = align.trailing_zeros();
        if exponent <= 26 {
            return Value::Enum {
                variant: 0,
                fields: vec![Value::Enum {
                    variant: exponent,
                    fields: Vec::new(),
                }],
            };
        }
    }
    Value::Uint(align)
}

/// The pointer a chain of one-field pointer wrappers holds:
/// the innermost field of a chain of single-field aggregates
/// (`NonNull` inside `Unique`, a `Box` inside its owner
/// wrapper), when that field is a pointer value. `None` when
/// the chain ends in anything else.
pub(crate) fn ptr_wrapper_pointer(value: &Value) -> Option<Value> {
    let mut cur = value;
    loop {
        match cur {
            Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
                p @ (Value::Ref(_)
                | Value::FatRef { .. }
                | Value::DynRef { .. }) => return Some(p.clone()),
                inner @ Value::Enum { .. } => cur = inner,
                _ => return None,
            },
            _ => return None,
        }
    }
}

pub(crate) fn box_alloc_pointer(value: &Value) -> Option<(u64, Vec<RefProj>)> {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
            Value::Ref(RefTarget::Alloc { id, projection }) => {
                Some((*id, projection.clone()))
            }
            _ => None,
        },
        _ => None,
    }
}

/// The allocation pointer stored at the bottom of a
/// pointer-wrapper aggregate chain: the `Box` / `Unique` /
/// `NonNull` aggregates std code builds in MIR, where field 0
/// of each wrapper is the next wrapper down (the remaining
/// fields are the zero-sized phantom and allocator fields) and
/// the chain bottoms out at the stored pointer. A box value
/// built by the `Box::new` builtin is the degenerate chain: a
/// one-field aggregate holding the pointer directly.
pub(crate) fn wrapper_alloc_pointer(value: &Value) -> Option<(u64, Vec<RefProj>)> {
    let mut cur = value;
    loop {
        match cur {
            Value::Ref(RefTarget::Alloc { id, projection }) => {
                return Some((*id, projection.clone()));
            }
            Value::Enum { fields, .. } if !fields.is_empty() => {
                cur = &fields[0];
            }
            _ => return None,
        }
    }
}

/// The slice length an unsizing cast reads from a pointee
/// value: the element count of the pointee's own array tail. A
/// custom DST value (an ADT whose trailing field is the unsized
/// part) carries the tail's length, recursively; an empty array
/// aggregate is the unit value, of length zero; and a scalar
/// pointee counts as one element — the one-element array view
/// of an allocation a reinterpreted pointer denotes
/// (`Box::into_boxed_slice` reinterprets a `Box<T>` allocation
/// as a `[T; 1]`). Anything else has no slice length.
pub(crate) fn unsized_tail_len(pointee: &Value) -> Option<u128> {
    match pointee {
        Value::Tuple(items) => Some(items.len() as u128),
        Value::Bytes(b) => Some(b.len() as u128),
        Value::Str(s) => Some(s.len() as u128),
        Value::FatRef { len, .. } => Some(*len),
        Value::Unit => Some(0),
        Value::Enum { fields, .. } => fields.last().and_then(unsized_tail_len),
        Value::Bool(_)
        | Value::Int(_)
        | Value::Uint(_)
        | Value::Float(_)
        | Value::Float128(_) => Some(1),
        _ => None,
    }
}

/// Whether a pointee's unsized length (see
/// [`unsized_tail_len`]) rests on the pointee's own array tail
/// rather than on the one-element view of a scalar: the
/// distinction the ADT field search in `unsize_to_slice` ranks
/// candidates by.
pub(crate) fn unsized_tail_is_array(pointee: &Value) -> bool {
    match pointee {
        Value::Tuple(_)
        | Value::Bytes(_)
        | Value::Str(_)
        | Value::FatRef { .. }
        | Value::Unit => true,
        Value::Enum { fields, .. } => {
            fields.last().is_some_and(unsized_tail_is_array)
        }
        _ => false,
    }
}

/// The count leaf of one reference count of an `RcInner` /
/// `ArcInner` aggregate: field `field` (0 = strong, 1 = weak) is
/// a `Cell<usize>` / `AtomicUsize` wrapper around an
/// `UnsafeCell<usize>` holding the count.
pub(crate) fn count_leaf(inner: &Value, field: usize) -> Option<u128> {
    fn unwrap1(v: &Value) -> Option<&Value> {
        match v {
            Value::Enum { fields, .. } if fields.len() == 1 => Some(&fields[0]),
            _ => None,
        }
    }
    let fields = match inner {
        Value::Enum { fields, .. } => fields,
        _ => return None,
    };
    let leaf = unwrap1(unwrap1(fields.get(field)?)?)?;
    match leaf {
        Value::Uint(n) => Some(*n),
        _ => None,
    }
}

/// The mutable count leaf of one reference count (see
/// [`count_leaf`]).
pub(crate) fn count_leaf_mut(inner: &mut Value, field: usize) -> Option<&mut u128> {
    fn unwrap1_mut(v: &mut Value) -> Option<&mut Value> {
        match v {
            Value::Enum { fields, .. } if fields.len() == 1 => fields.first_mut(),
            _ => None,
        }
    }
    let fields = match inner {
        Value::Enum { fields, .. } => fields,
        _ => return None,
    };
    let leaf = unwrap1_mut(unwrap1_mut(fields.get_mut(field)?)?)?;
    match leaf {
        Value::Uint(n) => Some(n),
        _ => None,
    }
}

/// Whether a stored value's tree contains any scalar content
/// (a number, a boolean, text, bytes or a pointer): such a
/// value belongs to a type of nonzero size. Used as a veto
/// when a comparing operand claims a place's positions are
/// zero-sized (see [`Interpreter::trailing_zero_ok`]): the
/// claim is only ever refused, never manufactured, by stored
/// content, so a scalar-free tree (units and aggregates of
/// units) leaves the claim untouched.
pub(crate) fn value_tree_has_scalar(value: &Value) -> bool {
    match value {
        Value::Bool(_)
        | Value::Int(_)
        | Value::Uint(_)
        | Value::Float(_)
        | Value::Float128(_)
        | Value::Str(_)
        | Value::Bytes(_)
        | Value::Ref(_)
        | Value::FatRef { .. }
        | Value::DynRef { .. }
        | Value::FmtCount(_)
        | Value::FmtArguments(_)
        | Value::Formatter(_)
        | Value::FmtBuilder(_)
        | Value::VaList(_)
        | Value::FnPtr(_)
        | Value::PanicPayload(_) => true,
        Value::Uninit | Value::Unit => false,
        Value::Tuple(items) => items.iter().any(value_tree_has_scalar),
        Value::Enum { fields, .. } => fields.iter().any(value_tree_has_scalar),
        Value::SparseVec { elems, .. } => elems.iter().any(value_tree_has_scalar),
        Value::FmtArg { target, .. } => value_tree_has_scalar(target),
        // An erased constant block is the verbatim storage of a
        // real allocation: its bytes are scalar content, and so
        // is anything its provenance relocations decoded to.
        Value::ErasedBlock { bytes, relocs } => {
            !bytes.is_empty() || relocs.iter().any(|(_, v)| value_tree_has_scalar(v))
        }
        // A union in its byte form holds scalar content
        // exactly where a write covered a byte; a span no
        // write covered holds nothing.
        Value::UnionBytes { bytes } => bytes.iter().any(|b| b.is_some()),
        Value::Closure { env, .. } => env.iter().any(value_tree_has_scalar),
        Value::Coroutine { fields, saved, .. } => {
            fields.iter().any(value_tree_has_scalar)
                || saved.iter().flatten().any(value_tree_has_scalar)
        }
    }
}

impl<'a> Interpreter<'a> {
    /// Validate / wrap a value being stored into a typed direct local.
    pub(crate) fn normalize_for_ty(mut value: Value, ty: Option<&Ty>) -> InterpResult<Value> {
        if value == Value::Uninit {
            // A `MaybeUninit` place holding the bare marker
            // denotes the uninitialized union value: the store
            // takes the union's canonical form (the field-0
            // payload skeleton holding the marker), so a
            // whole-place read or move of the place sees the
            // union value, while a read of the payload itself
            // still raises `UninitializedRead` at the union
            // field read -- the point of use.
            if matches!(ty, Some(t) if is_maybe_uninit_ty(t)) {
                return Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Uninit],
                });
            }
            // Storing the uninitialized marker itself is not a
            // type mismatch: MIR materialises uninitialised
            // locals with an assignment of the `Uninit` constant
            // (the storage a `MaybeUninit` or a union is filled
            // through a pointer afterwards), and moving a
            // not-yet-initialised local into another local
            // propagates the same marker. The place simply holds
            // no value yet; any later read of it still raises
            // `UninitializedRead`, so nothing is fabricated here.
            return Ok(value);
        }
        // A `MaybeUninit` place denotes the payload's own storage:
        // the union adds no representation of its own, and std's
        // byte-wise machinery reads and writes the payload through
        // the wrapper type while the storage is shared with the
        // payload-typed place (`ptr::swap_chunk` copies a chunk of
        // one element's storage into a `MaybeUninit<[u8; N]>`
        // local and back out into the other element). A read of
        // that storage yields the payload in storage form, so the
        // store accepts the value as it stands; wrapping it into
        // the field-0 form here would put a wrapper value where
        // the payload's own readers look.
        if matches!(
            ty,
            Some(Ty::Adt(name))
                if name.starts_with("std::mem::MaybeUninit")
                    || name.starts_with("core::mem::MaybeUninit")
        ) {
            return Ok(value);
        }
        // `core::ptr::Unaligned<T>` is a one-field `repr(C,
        // packed)` wrapper whose only field IS the value it
        // wraps: `ptr::read_unaligned` moves the pointee through
        // locals of this type, and a pointee read in this model
        // yields the field value itself (the allocation holds
        // the inner value; the wrapper has no address of its
        // own). Storing the bare value into an `Unaligned` place
        // therefore wraps it in the one-field aggregate, the
        // exact inverse of the wrapper-unwrap for scalar targets
        // below.
        if let Some(Ty::Adt(name)) = ty {
            if (name == "std::ptr::Unaligned" || name == "core::ptr::Unaligned")
                && !matches!(value, Value::Enum { .. })
            {
                return Ok(Value::Enum {
                    variant: 0,
                    fields: vec![value],
                });
            }
        }
        // A `UsizeNoHighBit` place (a `RawVecInner`'s capacity
        // slot): the type is a one-field wrapper around a
        // `usize`, and the capacity the resolver's buffer-chain
        // walk reports for a vector in its element-list form
        // is the bare count (see `resolve_place_mode`).
        // Storing the bare count into the wrapper place wraps
        // it in the one-field aggregate, the exact inverse of
        // the wrapper-unwrap for scalar targets below, as for
        // `Unaligned` above.
        if let Some(Ty::Adt(name)) = ty {
            if name == "core::num::niche_types::UsizeNoHighBit"
                && matches!(value, Value::Uint(_))
            {
                return Ok(Value::Enum {
                    variant: 0,
                    fields: vec![value],
                });
            }
        }
        // A scalar place whose value arrives as a one-field wrapper
        // aggregate: reading through a raw pointer into a transparent
        // wrapper (`UnsafeCell<T>`, the `AtomicUsize`-family structs,
        // any one-field struct) yields the wrapper's aggregate value,
        // because the pointer names the wrapper place itself and the
        // model has no separate address for the field. The wrapper's
        // only field IS the scalar at that address (the layout fact
        // the pointer arithmetic in the source relies on), so the
        // store unwraps it, recursively for nested wrappers.
        //
        // The unwrap is limited to unsigned-integer, boolean and
        // float targets whose innermost leaf is a value of the same
        // kind: those are the shapes the atomic/pointer-metadata
        // assignments actually produce. A signed-integer target does
        // not unwrap: a signed field's width lives only in its
        // declared type, so reading a wrapper aggregate as a signed
        // scalar of some other width is exactly the reinterpretation
        // a cast to an unrelated pointee type asks for, and the
        // model — which has no byte memory — keeps it an explicit
        // type mismatch instead of fabricating a value (the negative
        // case `wrapper_cast_to_an_unrelated_pointee_is_not_
        // reinterpreted` pins this).
        if matches!(
            ty,
            Some(Ty::Bool) | Some(Ty::Uint(_)) | Some(Ty::Float(_))
        ) {
            let mut inner = &value;
            while let Value::Enum {
                variant: 0,
                fields,
            } = inner
            {
                if fields.len() != 1 {
                    break;
                }
                inner = &fields[0];
            }
            let same_kind = matches!(
                (ty, inner),
                (Some(Ty::Uint(_)), Value::Uint(_))
                    | (Some(Ty::Bool), Value::Bool(_))
                    | (Some(Ty::Float(_)), Value::Float(_))
            );
            if same_kind && !matches!(inner, Value::Enum { .. }) {
                value = inner.clone();
            }
        }
        // An array place whose value arrives as a one-field
        // wrapper aggregate: reading a one-field struct over an
        // array (a `#[repr(simd)]` vector, above all) through a
        // pointer cast to the array type -- `Simd::into_array`
        // dereferences a `*const [T; N]` at the vector's own
        // address -- yields the wrapper's aggregate value,
        // because the pointer names the wrapper place itself.
        // The wrapper's only field IS the array at that address,
        // so the store unwraps it, recursively for nested
        // wrappers: the aggregate counterpart of the scalar
        // unwrap above. The field must be exactly the declared
        // array (a tuple of the declared length, or the byte
        // form of a `u8` array); any other field shape keeps
        // the type mismatch.
        if let Some(Ty::Array(elem, n)) = ty {
            let mut inner = &value;
            while let Value::Enum {
                variant: 0,
                fields,
            } = inner
            {
                if fields.len() != 1 {
                    break;
                }
                let field = &fields[0];
                let is_the_array = match field {
                    Value::Tuple(items) => items.len() as u64 == *n,
                    Value::Bytes(b) => {
                        **elem == Ty::Uint(UintTy::U8) && b.len() as u64 == *n
                    }
                    _ => false,
                };
                if is_the_array {
                    value = field.clone();
                    break;
                }
                inner = field;
            }
        }
        // A tuple place whose value arrives as a one-field
        // wrapper aggregate: a tuple's own value form is the
        // tuple itself, never an aggregate, so a one-field
        // aggregate around a tuple of the declared arity is a
        // transparent storage wrapper (the `MaybeUninit` /
        // `ManuallyDrop` payload slot) the value crossed on its
        // way through raw storage -- the storage a
        // `Box::new_in` writes through a `*mut MaybeUninit<T>`
        // and a pointee-typed read takes back out. The store
        // unwraps it: the aggregate counterpart of the scalar
        // and array unwraps above.
        if let Some(Ty::Tuple(elems)) = ty {
            let arity = elems.len();
            if let Value::Enum {
                variant: 0,
                fields,
            } = &value
            {
                if fields.len() == 1 {
                    if let Value::Tuple(items) = &fields[0] {
                        if items.len() == arity {
                            value = fields[0].clone();
                        }
                    }
                }
            }
        }
        // A `String` place holds string data: the model's
        // `String` value form is the data itself, never an
        // aggregate (a field-built `String` aggregate holds its
        // `Vec<u8>` field, not string data). A one-field
        // wrapper aggregate around string data is therefore a
        // transparent storage wrapper of the same kind, and the
        // store unwraps it.
        if let Some(Ty::Adt(name)) = ty {
            if name == "std::string::String" || name == "alloc::string::String" {
                if let Value::Enum {
                    variant: 0,
                    fields,
                } = &value
                {
                    if fields.len() == 1 && matches!(fields[0], Value::Str(_)) {
                        value = fields[0].clone();
                    }
                }
            }
        }
        match (ty, &value) {
            (Some(Ty::Bool), Value::Bool(_))
            | (Some(Ty::Unit), Value::Unit)
            | (Some(Ty::Tuple(_)), Value::Tuple(_))
            | (None, _) => Ok(value),
            (Some(Ty::Int(it)), Value::Int(v)) => Ok(Value::Int(wrap_signed(*v, it.bits()))),
            (Some(Ty::Uint(ut)), Value::Uint(v)) => Ok(Value::Uint(*v & ut.max())),
            // An `f32` place holds the `f32` rounding of the stored
            // (exactly widened) value; an `f64` place holds it as-is.
            (Some(Ty::Float(FloatTy::F16)), Value::Float(v)) => {
                Ok(Value::Float(round_to_f16(*v)))
            }
            (Some(Ty::Float(FloatTy::F32)), Value::Float(v)) => {
                if v.is_nan() {
                    // The rounding above would quiet a signaling
                    // NaN; a place holds the value's exact form,
                    // so narrow and re-widen through the bit-exact
                    // pair instead.
                    Ok(Value::Float(crate::f128::f32_bits_to_f64_exact(
                        crate::f128::f64_to_f32_bits_exact(*v),
                    )))
                } else {
                    Ok(Value::Float((*v as f32) as f64))
                }
            }
            (Some(Ty::Float(FloatTy::F64)), Value::Float(_)) => Ok(value),
            (Some(Ty::Float(FloatTy::F128)), Value::Float128(_)) => Ok(value),
            // An integer payload whose kind tag differs from the
            // declared type: the bits are reinterpreted at the
            // declared width. This arises for scalar leaves of
            // structured constants, whose signedness is not stored
            // in the constant itself; MIR places and their values
            // always agree on width, so the declared type is the
            // exact interpretation.
            (Some(Ty::Int(it)), Value::Uint(v)) => {
                Ok(Value::Int(wrap_signed(*v as i128, it.bits())))
            }
            (Some(Ty::Uint(ut)), Value::Int(v)) => Ok(Value::Uint((*v as u128) & ut.max())),
            // Callable locals: a function-pointer place holds a
            // function item value or a (non-capturing) closure value;
            // a function-item place holds the item value; a closure
            // place holds the closure value.
            (Some(Ty::FnPtr), Value::FnPtr(_))
            | (Some(Ty::FnPtr), Value::Closure { .. })
            | (Some(Ty::FnDef(_)), Value::FnPtr(_))
            | (Some(Ty::Closure(..)), Value::Closure { .. }) => Ok(value),
            // A coroutine-closure (async closure) place whose
            // type the driver recorded as the unit placeholder
            // (the type's kind is unaskable through Stable MIR;
            // see `coroutine_closure_creation` in mir_convert):
            // the place's real type is the closure type, and the
            // closure value its creation aggregate produced is
            // the value the place holds. A genuine unit place
            // only ever receives a unit value in well-typed MIR,
            // so the arm cannot bless a wrong store.
            (Some(Ty::Unit), Value::Closure { .. }) => Ok(value),
            // ADT locals hold the ADT value forms: aggregates, string
            // data (`String`), and the fmt value family.
            (Some(Ty::Adt(_)), Value::Enum { .. })
            // A pointer-shaped ADT local (`Unique`, `NonNull`, a
            // `Box` field slot) can hold a built box's allocation
            // pointer directly: MIR moves the pointer between
            // these wrapper locals verbatim (a box's field 0 read
            // into a `Unique` local is the first step of every
            // `Box::into_raw`), and the pointer is the value each
            // of them denotes in this model. The target may be a
            // detached value as well as an allocation — the data
            // pointer read out of a string the model holds as
            // bare data names the string value itself.
            | (Some(Ty::Adt(_)), Value::Ref(_))
            | (Some(Ty::Adt(_)), Value::Str(_))
            // A smart-pointer ADT over a `dyn` pointee (`Box<dyn
            // Trait>`, `Rc<dyn Trait>`) holds the fat pointer.
            | (Some(Ty::Adt(_)), Value::DynRef { .. })
            | (Some(Ty::Adt(_)), Value::FmtArg { .. })
            | (Some(Ty::Adt(_)), Value::FmtCount(_))
            | (Some(Ty::Adt(_)), Value::FmtArguments(_))
            // The captured-`Formatter` and builder handles stand
            // for `core::fmt` ADT values (`Formatter` itself when
            // a body moves it, and the `DebugList` family the
            // builder constructors return by value).
            | (Some(Ty::Adt(_)), Value::Formatter(_))
            // The va-list handle stands for the `std::ffi::VaList`
            // ADT value a variadic body moves and clones.
            | (Some(Ty::Adt(_)), Value::VaList(_))
            | (Some(Ty::Adt(_)), Value::FmtBuilder(_)) => Ok(value),
            // Reference locals hold resolved references, fat pointers,
            // and — for `&str` and byte-array references — the string
            // or byte data itself (a string constant *is* its value;
            // a reference to a string place is a `Ref`).
            (Some(Ty::Ref(_)), Value::Ref(_))
            | (Some(Ty::Ref(_)), Value::FatRef { .. })
            | (Some(Ty::Ref(_)), Value::DynRef { .. })
            | (Some(Ty::Ref(_)), Value::Str(_))
            | (Some(Ty::Ref(_)), Value::Bytes(_))
            // A pointer to a function stored under a cast (a
            // function item referenced through a data-pointer type,
            // e.g. `&(foo as *const u8)` as a constant): the value
            // is the callable, which is a pointer like any other
            // for storage, address and equality purposes; reading
            // the pointee through it stays an explicit error at the
            // read site, as the model has no code bytes.
            | (Some(Ty::Ref(_)), Value::FnPtr(_))
            | (Some(Ty::Ref(_)), Value::Closure { .. }) => Ok(value),
            // A niche-encoded enum over a pointer payload stored
            // into a pointer place: the enum's representation IS
            // the payload pointer -- the ABI fact the
            // transmuted-fn-pointer return relies on (a callee
            // whose declared return type is `Option<&T>` produces
            // the payload-variant aggregate, and a caller whose
            // signature was transmuted to return `&T` stores the
            // result into a reference place). The one field is
            // the pointer itself, the same reading a one-field
            // wrapper's store unwraps to its scalar above. A
            // fieldless (niche) variant value carries no pointer
            // to read, so it stays a mismatch rather than
            // fabricating one.
            // A pointer-shaped ADT value (`NonNull`, `Unique`,
            // a `Box` inside its owner wrapper) written to a
            // raw-pointer or reference place: MIR moves these
            // transparent wrappers into pointer places verbatim
            // (the `Allocator::deallocate` body passes its
            // `NonNull<u8>` straight to `__rust_dealloc`), and
            // the pointer the wrapper chain holds is the value
            // the place denotes.
            (Some(Ty::Ref(_)), Value::Enum { .. })
                if ptr_wrapper_pointer(&value).is_some() =>
            {
                Ok(ptr_wrapper_pointer(&value).expect("guard checked"))
            }
            (Some(Ty::Str), Value::Str(_)) => Ok(value),
            (Some(Ty::Array(_, n)), Value::Tuple(items)) if items.len() as u64 == *n => Ok(value),
            (Some(Ty::Array(elem, n)), Value::Bytes(b))
                if **elem == Ty::Uint(UintTy::U8) && b.len() as u64 == *n =>
            {
                Ok(value)
            }
            // A byte read through a `*const u8` / `*const i8`
            // whose pointee is a byte-string value: the pointer
            // denotes the storage's first byte (any displacement
            // has already been projected away by the read).
            (Some(Ty::Uint(UintTy::U8)), Value::Bytes(b)) if !b.is_empty() => {
                Ok(Value::Uint(u128::from(b[0])))
            }
            (Some(Ty::Int(IntTy::I8)), Value::Bytes(b)) if !b.is_empty() => {
                Ok(Value::Int(i128::from(b[0] as i8)))
            }
            (Some(Ty::Slice(_)), Value::FatRef { .. }) => Ok(value),
            // Coroutine locals hold the state object itself. (A
            // coroutine value also reaches places typed as the ADT
            // wrappers around it - `Box`, `Pin` - as a field of the
            // wrapper's aggregate value, which is checked as the
            // wrapper, not here.)
            (Some(Ty::Coroutine(_)), Value::Coroutine { .. }) => Ok(value),
            // The async-drop transform types a coroutine body's
            // resume place as the ADT `std::future::ResumeTy`,
            // the wrapper the coroutine environment travels in;
            // the value stored there is the coroutine object
            // itself, exactly as in a coroutine-typed place.
            (Some(Ty::Adt(name)), Value::Coroutine { .. })
                if name == "std::future::ResumeTy" =>
            {
                Ok(value)
            }
            // A vector value in its sparse form (see
            // `Value::SparseVec`) is a `Vec` value: it stores
            // into a `Vec` place as it stands.
            (Some(Ty::Adt(name)), Value::SparseVec { .. })
                if name == "std::vec::Vec" || name == "alloc::vec::Vec" =>
            {
                Ok(value)
            }
            (Some(expected), other) => {
                // A zero-sized place holds the type's single
                // value, whatever shape the operand took: the
                // read that produced it may have read the
                // storage a zero-sized header shares with the
                // value behind it (a sized `ThinBox`'s
                // metadata is the unit value, read through a
                // header pointer that aliases the value
                // itself).
                match zst_value_of_ty(expected) {
                    Some(single) => Ok(single),
                    None => Err(InterpError::type_mismatch(
                        "assignment",
                        format!("{expected:?}"),
                        other.kind_name(),
                    )),
                }
            }
        }
    }
}

impl<'a> Interpreter<'a> {
    /// The bit pattern of `value` read through `shape`
    /// (little-endian: earlier array elements and lower fields
    /// occupy the lower bits).
    /// The data word a fat pointer flattens to: the base's
    /// arithmetic address where the address model derives
    /// one — so a flattened fat pointer's data word is the
    /// address the pointer compares and exposes as — else
    /// the synthetic-hash word.
    fn fat_data_word(&self, base: &Value) -> u128 {
        self.arithmetic_price(base, None, None)
            .unwrap_or_else(|| data_word(base))
    }

    pub(crate) fn flatten_scalar(
        &self,
        value: &Value,
        shape: &ScalarShape,
    ) -> InterpResult<u128> {
        match (shape, value) {
            // A union in its byte form flattens from its
            // storage bytes: a byte a write covered holds the
            // pattern there, and a byte no write covered reads
            // as zero, the convention the byte-form field read
            // keeps. (Shapes wider than a pattern have no
            // flattening; the structural byte rendering in
            // `union_field_bytes` is the path for those.)
            (shape, Value::UnionBytes { bytes }) if shape.bits() <= 128 => {
                let mut out: u128 = 0;
                for (i, b) in bytes.iter().enumerate().take(16) {
                    if let Some(b) = b {
                        out |= u128::from(*b) << (8 * i);
                    }
                }
                Ok(out & bit_mask(shape.bits()))
            }
            (ScalarShape::Int { bits, .. }, Value::Int(i)) => {
                Ok((*i as u128) & bit_mask(*bits))
            }
            (ScalarShape::Int { bits, .. }, Value::Uint(u)) => {
                Ok(*u & bit_mask(*bits))
            }
            (ScalarShape::Float { bits }, Value::Float(f)) => Ok(match bits {
                16 => u128::from(f64_to_f16_bits(*f)),
                32 => u128::from(crate::f128::f64_to_f32_bits_exact(*f)),
                _ => u128::from(f.to_bits()),
            }),
            // An `f128` value is stored as its exact bit
            // pattern, so flattening it is the identity.
            (ScalarShape::Float { bits: 128 }, Value::Float128(b)) => Ok(*b),
            (ScalarShape::Bool, Value::Bool(b)) => Ok(u128::from(*b)),
            (ScalarShape::Char, Value::Uint(u)) => Ok(*u & bit_mask(32)),
            (ScalarShape::Array(elem, n), Value::Tuple(items))
                if items.len() as u64 == *n =>
            {
                let eb = elem.bits();
                let mut out = 0u128;
                for (i, item) in items.iter().enumerate() {
                    out |= self.flatten_scalar(item, elem)? << (i as u32 * eb);
                }
                Ok(out)
            }
            // A byte-array constant's value is its bytes.
            (ScalarShape::Array(elem, n), Value::Bytes(b))
                if elem.bits() == 8 && b.len() as u64 == *n =>
            {
                let mut out = 0u128;
                for (i, byte) in b.iter().enumerate() {
                    out |= u128::from(*byte) << (8 * i);
                }
                Ok(out)
            }
            // A pointer payload stored bare (the model keeps a
            // wrapper's payload pointer as the value itself in
            // some slots — the same convention `project_value`
            // encodes for field 0 of a pointer value): flatten
            // the pointer through the payload shape directly.
            (
                ScalarShape::AdtField { index, inner, .. },
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. },
            ) if *index == 0 => self.flatten_scalar(value, inner),
            (
                ScalarShape::AdtField { index, inner, .. },
                Value::Enum { fields, .. },
            ) => match fields.get(*index as usize) {
                Some(field) => self.flatten_scalar(field, inner),
                None => Err(InterpError::type_mismatch(
                    "transmute",
                    "an ADT value holding the payload field",
                    "an ADT value without it",
                )),
            },
            // The storage form of the same payload: a write
            // through a pointer cast to the payload field's type
            // (the `Simd::load` copy through a `*mut [T; N]`
            // into the union's slot) stores the field's own
            // value where the aggregate would sit. Its bits are
            // the payload's bits, so it flattens as the payload.
            (ScalarShape::AdtField { inner, .. }, value) => {
                self.flatten_scalar(value, inner)
            },
            (
                ScalarShape::NicheEnum {
                    inner,
                    untagged_variant,
                    niche_variants,
                    niche_start,
                    niche_shift,
                    niche_bits,
                },
                Value::Enum { variant, fields },
            ) => {
                if (niche_variants.0..=niche_variants.1).contains(variant) {
                    // The encoded value is taken at the niche
                    // scalar's width before shifting: a range
                    // starting near the scalar's maximum wraps
                    // (255 + 1 encodes as 0 in a u8 scalar)
                    // instead of spilling into the adjacent
                    // payload byte.
                    Ok(((*niche_start + u128::from(*variant - niche_variants.0))
                        & bit_mask(*niche_bits))
                        << niche_shift)
                } else if variant == untagged_variant {
                    match fields.first() {
                        Some(field) => self.flatten_scalar(field, inner),
                        None => Err(InterpError::type_mismatch(
                            "transmute",
                            "an enum value holding the payload field",
                            "an enum value without it",
                        )),
                    }
                } else {
                    Err(InterpError::type_mismatch(
                        "transmute",
                        "a variant of the niche-encoded enum",
                        format!("variant {variant}"),
                    ))
                }
            }
            (
                ScalarShape::NicheEnumFields {
                    fields,
                    untagged_variant,
                    niche_variants,
                    niche_start,
                    niche_shift,
                    niche_bits,
                    ..
                },
                Value::Enum { variant, fields: vals },
            ) => {
                if variant == untagged_variant {
                    let mut out = 0u128;
                    for (index, offset, shape) in fields {
                        let field = vals.get(*index as usize).ok_or_else(
                            || {
                                InterpError::type_mismatch(
                                    "transmute",
                                    "an ADT value holding every field",
                                    "an ADT value without one",
                                )
                            },
                        )?;
                        out |= (self.flatten_scalar(field, shape)?
                            & bit_mask(shape.bits()))
                            << offset;
                    }
                    Ok(out)
                } else if (niche_variants.0..=niche_variants.1)
                    .contains(variant)
                {
                    // As in the single-field niche shape, the
                    // encoded value is taken at the niche
                    // scalar's width before shifting.
                    Ok(((*niche_start
                        + u128::from(*variant - niche_variants.0))
                        & bit_mask(*niche_bits))
                        << niche_shift)
                } else {
                    Err(InterpError::type_mismatch(
                        "transmute",
                        "a variant of the niche-encoded enum",
                        format!("variant {variant}"),
                    ))
                }
            }
            (ScalarShape::Ptr | ScalarShape::FnPtr, _) => match self
                .arithmetic_price(value, None, None)
                .or_else(|| synthetic_address(value))
            {
                Some(addr) => {
                    self.note_exposed_target(addr as u64, value);
                    Ok(addr & bit_mask(64))
                }
                None => Err(InterpError::type_mismatch(
                    "transmute",
                    "a pointer value",
                    value.kind_name(),
                )),
            },
            (
                ScalarShape::FieldlessEnum {
                    bits,
                    discriminants,
                },
                Value::Enum { variant, .. },
            ) => match discriminants.get(*variant as usize) {
                Some(d) => Ok((*d as u128) & bit_mask(*bits)),
                None => Err(InterpError::type_mismatch(
                    "transmute",
                    "a variant of the fieldless enum",
                    format!("variant {variant}"),
                )),
            },
            (ScalarShape::FatPtr(meta), value) => {
                // A fat pointer held behind a value reference
                // (an element of a constant aggregate) flattens
                // as the fat value the reference denotes.
                let value = match value {
                    Value::Ref(RefTarget::Value(inner, projection))
                        if projection.is_empty() =>
                    {
                        inner.as_ref()
                    }
                    _ => value,
                };
                let (data, meta_word) = match (meta, value) {
                    (crate::ir::FatMeta::Len, Value::FatRef { base, len }) => {
                        (self.fat_data_word(base), *len & bit_mask(64))
                    }
                    (crate::ir::FatMeta::Len, Value::Str(s)) => {
                        (str_data_word(s), s.len() as u128)
                    }
                    (
                        crate::ir::FatMeta::Vtable,
                        Value::DynRef {
                            base,
                            concrete,
                            traits,
                            ..
                        },
                    ) => (self.fat_data_word(base), vtable_word(concrete, traits)),
                    (
                        crate::ir::FatMeta::Len,
                        Value::Ref(RefTarget::Alloc { id, projection }),
                    ) if projection.is_empty() => {
                        // A fat pointer whose pointee lives in
                        // the allocation store (a box's
                        // pointer): the pointer itself is the
                        // thin store reference, and the length
                        // is the stored pointee's length.
                        let len = match self.allocs.borrow().get(id) {
                            Some(Value::Tuple(items)) => {
                                items.len() as u128
                            }
                            Some(Value::FatRef { len, .. }) => *len,
                            Some(Value::SparseVec { len, .. }) => *len,
                            Some(Value::Str(s)) => s.len() as u128,
                            Some(Value::Bytes(b)) => b.len() as u128,
                            _ => {
                                return Err(InterpError::type_mismatch(
                                    "transmute",
                                    "a fat pointer into the allocation store",
                                    value.kind_name(),
                                ))
                            }
                        };
                        (self.fat_data_word(value), len)
                    }
                    _ => {
                        return Err(InterpError::type_mismatch(
                            "transmute",
                            "a fat pointer value of the shape's metadata kind",
                            value.kind_name(),
                        ))
                    }
                };
                let bits = (data & bit_mask(64)) | (meta_word << 64);
                // The pattern alone does not denote the value
                // (see the `fat_values` field): register it, so
                // constructing the same pattern returns this
                // very pointer.
                self.fat_values.borrow_mut().insert(bits, value.clone());
                Ok(bits)
            }
            (
                ScalarShape::AdtFields { fields, .. },
                Value::Enum { fields: vals, .. },
            ) => {
                let mut out = 0u128;
                for (index, offset, shape) in fields {
                    let field =
                        vals.get(*index as usize).ok_or_else(|| {
                            InterpError::type_mismatch(
                                "transmute",
                                "an ADT value holding every field",
                                "an ADT value without one",
                            )
                        })?;
                    out |= (self.flatten_scalar(field, shape)?
                        & bit_mask(shape.bits()))
                        << offset;
                }
                Ok(out)
            }
            (
                ScalarShape::TaggedEnum {
                    tag_offset,
                    tag_bits,
                    discriminants,
                    variants,
                    ..
                },
                Value::Enum { variant, fields },
            ) => {
                let disc =
                    discriminants.get(*variant as usize).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "transmute",
                            "a variant of the tagged enum",
                            format!("variant {variant}"),
                        )
                    })?;
                let mut out =
                    (*disc & bit_mask(*tag_bits)) << tag_offset;
                for (index, offset, shape) in
                    &variants[*variant as usize].fields
                {
                    let field =
                        fields.get(*index as usize).ok_or_else(|| {
                            InterpError::type_mismatch(
                                "transmute",
                                "an enum value holding every field",
                                "an enum value without one",
                            )
                        })?;
                    out |= (self.flatten_scalar(field, shape)?
                        & bit_mask(shape.bits()))
                        << offset;
                }
                Ok(out)
            }
            (ScalarShape::Fields { fields, .. }, Value::Tuple(items))
                if items.len() == fields.len() =>
            {
                let mut out = 0u128;
                for (item, (offset, field)) in
                    items.iter().zip(fields.iter())
                {
                    out |= (self.flatten_scalar(item, field)?
                        & bit_mask(field.bits()))
                        << offset;
                }
                Ok(out)
            }
            (_, other) => Err(InterpError::type_mismatch(
                "transmute",
                "a value of the source shape",
                other.kind_name(),
            )),
        }
    }

    /// The value of `shape` whose bit pattern is `bits`
    /// (the low `shape.bits()` bits; higher bits are ignored).
    pub(crate) fn construct_scalar(
        &self,
        bits: u128,
        shape: &ScalarShape,
    ) -> InterpResult<Value> {
        match shape {
            ScalarShape::Int { bits: w, signed } => {
                if *signed {
                    Ok(Value::Int(wrap_signed(bits as i128, *w)))
                } else {
                    Ok(Value::Uint(wrap_unsigned(bits, *w)))
                }
            }
            ScalarShape::Float { bits: w } => match w {
                // An `f128` value is its exact bit pattern
                // (`Value::Float128`), so construction is the
                // identity on the pattern.
                128 => Ok(Value::Float128(bits)),
                _ => Ok(Value::Float(match w {
                    16 => crate::mir_convert::f16_to_f64(bits as u16),
                    // Rebuilding a value from its bit pattern is
                    // transport: widen exactly, as `flatten_scalar`
                    // narrows exactly, so a signaling NaN pattern
                    // survives a transmute round trip.
                    32 => crate::f128::f32_bits_to_f64_exact(bits as u32),
                    _ => f64::from_bits(bits as u64),
                })),
            }
            ScalarShape::Ptr => {
                // A data-pointer leaf constructs a data
                // pointer, even when the pattern is an address
                // a function value was exposed at: the callable
                // reading belongs to function-pointer
                // destinations (the `FnPtr` leaf below) only.
                Ok(self.materialized_ptr(bits as u64))
            }
            ScalarShape::FnPtr => {
                // An address a function pointer was exposed
                // at reconstructs that function value (see
                // the `exposed_fn_ptrs` field); any other
                // pattern is an address-only pointer as usual.
                if let Some(v) =
                    self.exposed_fn_ptrs.borrow().get(&(bits as u64))
                {
                    return Ok(v.clone());
                }
                Ok(self.materialized_ptr(bits as u64))
            }
            ScalarShape::Bool => Ok(Value::Bool(bits != 0)),
            ScalarShape::Char => match char::from_u32(bits as u32) {
                Some(c) => Ok(Value::Uint(u128::from(c as u32))),
                None => Err(InterpError::type_mismatch(
                    "transmute to char",
                    "a valid Unicode code point",
                    format!("bit pattern {bits:#x}"),
                )),
            },
            ScalarShape::Array(elem, n) => {
                let eb = elem.bits();
                let mut items = Vec::with_capacity(*n as usize);
                for i in 0..*n {
                    let part = (bits >> (i as u32 * eb)) & bit_mask(eb);
                    items.push(self.construct_scalar(part, elem)?);
                }
                Ok(Value::Tuple(items))
            }
            ScalarShape::AdtField {
                index,
                count,
                fill,
                inner,
            } => {
                let mut fields = vec![Value::Uninit; *count];
                for (i, c) in fill {
                    fields[*i as usize] = self.eval_constant(c)?;
                }
                fields[*index as usize] = self.construct_scalar(bits, inner)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields,
                })
            }
            ScalarShape::NicheEnum {
                inner,
                untagged_variant,
                niche_variants,
                niche_start,
                niche_shift,
                niche_bits,
            } => {
                let window =
                    (bits >> niche_shift) & bit_mask(*niche_bits);
                let span = u128::from(niche_variants.1 - niche_variants.0);
                // The niche range may wrap past the niche
                // scalar's maximum (a range starting at 255 in
                // a u8 scalar encodes its second variant as 0),
                // so the distance from the range's start is
                // taken at the scalar's width.
                let offset = window.wrapping_sub(*niche_start) & bit_mask(*niche_bits);
                if offset <= span {
                    Ok(Value::Enum {
                        variant: niche_variants.0 + offset as u32,
                        fields: Vec::new(),
                    })
                } else {
                    let payload = self.construct_scalar(bits, inner)?;
                    Ok(Value::Enum {
                        variant: *untagged_variant,
                        fields: vec![payload],
                    })
                }
            }
            ScalarShape::NicheEnumFields {
                count,
                fill,
                fields,
                untagged_variant,
                niche_variants,
                niche_start,
                niche_shift,
                niche_bits,
                ..
            } => {
                let window =
                    (bits >> niche_shift) & bit_mask(*niche_bits);
                let span =
                    u128::from(niche_variants.1 - niche_variants.0);
                // As in the single-field niche shape, the
                // distance from the range's start is taken at
                // the niche scalar's width (the range may wrap
                // past the scalar's maximum).
                let offset = window.wrapping_sub(*niche_start)
                    & bit_mask(*niche_bits);
                let decoded = niche_variants.0 + offset as u32;
                if offset <= span && decoded != *untagged_variant {
                    Ok(Value::Enum {
                        variant: decoded,
                        fields: Vec::new(),
                    })
                } else {
                    let mut vals = vec![Value::Uninit; *count];
                    for (i, c) in fill {
                        vals[*i as usize] = self.eval_constant(c)?;
                    }
                    for (index, offset, shape) in fields {
                        let part =
                            (bits >> offset) & bit_mask(shape.bits());
                        vals[*index as usize] =
                            self.construct_scalar(part, shape)?;
                    }
                    Ok(Value::Enum {
                        variant: *untagged_variant,
                        fields: vals,
                    })
                }
            }
            ScalarShape::AdtFields {
                count,
                fill,
                fields,
                ..
            } => {
                let mut vals = vec![Value::Uninit; *count];
                for (i, c) in fill {
                    vals[*i as usize] = self.eval_constant(c)?;
                }
                for (index, offset, shape) in fields {
                    let part =
                        (bits >> offset) & bit_mask(shape.bits());
                    vals[*index as usize] =
                        self.construct_scalar(part, shape)?;
                }
                Ok(Value::Enum {
                    variant: 0,
                    fields: vals,
                })
            }
            ScalarShape::TaggedEnum {
                tag_offset,
                tag_bits,
                discriminants,
                variants,
                ..
            } => {
                let pattern =
                    (bits >> tag_offset) & bit_mask(*tag_bits);
                let Some(variant) = discriminants
                    .iter()
                    .position(|d| *d & bit_mask(*tag_bits) == pattern)
                else {
                    return Err(InterpError::type_mismatch(
                        "transmute to a tagged enum",
                        "a discriminant value of the target enum",
                        format!("bit pattern {pattern:#x}"),
                    ));
                };
                let vshape = &variants[variant];
                let mut vals = vec![Value::Uninit; vshape.count];
                for (i, c) in &vshape.fill {
                    vals[*i as usize] = self.eval_constant(c)?;
                }
                for (index, offset, shape) in &vshape.fields {
                    let part =
                        (bits >> offset) & bit_mask(shape.bits());
                    vals[*index as usize] =
                        self.construct_scalar(part, shape)?;
                }
                Ok(Value::Enum {
                    variant: variant as u32,
                    fields: vals,
                })
            }
            ScalarShape::FatPtr(meta) => {
                // A pattern a flattened value registered
                // reconstructs that very value (see the
                // `fat_values` field); an unregistered pattern
                // is fabricated from its words only where the
                // words carry the value -- a length metadata
                // does, a vtable identity does not.
                if let Some(v) = self.fat_values.borrow().get(&bits) {
                    return Ok(v.clone());
                }
                match meta {
                    crate::ir::FatMeta::Len => Ok(Value::FatRef {
                        base: Box::new(Value::Ref(RefTarget::Address(
                            bits as u64,
                        ))),
                        len: bits >> 64,
                    }),
                    crate::ir::FatMeta::Vtable => {
                        Err(InterpError::type_mismatch(
                            "transmute to a dyn fat pointer",
                            "a bit pattern a flattened dyn value registered",
                            format!("bit pattern {bits:#x}"),
                        ))
                    }
                }
            }
            ScalarShape::Fields { fields, .. } => {
                let mut items = Vec::with_capacity(fields.len());
                for (offset, field) in fields {
                    let part = (bits >> offset) & bit_mask(field.bits());
                    items.push(self.construct_scalar(part, field)?);
                }
                Ok(Value::Tuple(items))
            }
            ScalarShape::FieldlessEnum {
                bits: width,
                discriminants,
            } => {
                let pattern = bits & bit_mask(*width);
                match discriminants
                    .iter()
                    .position(|d| (*d as u128) & bit_mask(*width) == pattern)
                {
                    Some(variant) => Ok(Value::Enum {
                        variant: variant as u32,
                        fields: Vec::new(),
                    }),
                    None => Err(InterpError::type_mismatch(
                        "transmute to a fieldless enum",
                        "a discriminant value of the target enum",
                        format!("bit pattern {pattern:#x}"),
                    )),
                }
            }
        }
    }

    /// The `DynRef` a vtable identity number denotes over the
    /// given data-pointer value, resolved through the program's
    /// vtable identity table (see
    /// `ir::Program::vtable_identities`): `None` when the
    /// identity is not one the driver recorded.
    pub(crate) fn dyn_ref_from_vtable(&self, id: u64, base: Value) -> Option<Value> {
        let identity = self.program.vtable_identities.get(&id)?;
        Some(Value::DynRef {
            base: Box::new(base),
            concrete: identity.concrete.clone(),
            traits: identity.traits.clone(),
            drop_glue: identity.glue.clone().map(Box::new),
        })
    }

    /// The vtable identity a metadata value names, if any: a
    /// `DynMetadata` value is an aggregate whose vtable word is
    /// a leaf under one-field wrappers, and a bare vtable
    /// pointer is the address itself. Only an address the
    /// driver recorded as a vtable identity counts -- the
    /// search descends aggregate fields, but an address is a
    /// vtable identity only when the table names it one.
    pub(crate) fn vtable_id_in_metadata(&self, value: &Value) -> Option<u64> {
        match value {
            Value::Ref(RefTarget::Address(id)) => self
                .program
                .vtable_identities
                .contains_key(id)
                .then_some(*id),
            Value::Enum { fields, .. } | Value::Tuple(fields) => {
                fields.iter().find_map(|f| self.vtable_id_in_metadata(f))
            }
            _ => None,
        }
    }

    /// Rebase a metadata value onto a data pointer: a `DynRef`
    /// metadata (the form the pointer-metadata model hands
    /// back for a `dyn` pointee) keeps its identity over the
    /// new base; a `DynMetadata` aggregate or bare vtable
    /// pointer resolves through the vtable identity table.
    /// `None` when the metadata names no vtable (a length, a
    /// unit, or an unrecorded identity).
    pub(crate) fn dyn_ref_from_metadata(&self, meta: &Value, base: Value) -> Option<Value> {
        match meta {
            Value::DynRef {
                concrete,
                traits,
                drop_glue,
                ..
            } => Some(Value::DynRef {
                base: Box::new(base),
                concrete: concrete.clone(),
                traits: traits.clone(),
                drop_glue: drop_glue.clone(),
            }),
            _ => {
                let id = self.vtable_id_in_metadata(meta)?;
                self.dyn_ref_from_vtable(id, base)
            }
        }
    }

    /// The metadata of a pointer value (`PtrMetadata` rvalue):
    /// a slice or string pointer's length, a dyn pointer's vtable
    /// (the fat value itself in this model, only ever passed along
    /// or compared opaquely), and the unit value for a sized
    /// pointee. A reference to a constant slice value (a static
    /// slice, materialised as a reference to the fat value) and a
    /// reference naming a place whose contents are a fat value (a
    /// slice reference formed as `&*box`, stored as a place
    /// reference) both carry that fat value's metadata.
    pub(crate) fn ptr_metadata(
        &self,
        frames: &[Frame],
        v: &Value,
        dest_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match v {
            Value::FatRef { len, .. } => Ok(Value::Uint(*len)),
            Value::Str(s) => Ok(Value::Uint(s.len() as u128)),
            Value::DynRef { .. } => Ok(v.clone()),
            Value::Ref(target) => {
                // A byte-displaced thin pointer's metadata is the
                // unit value: the pointer itself is thin whatever
                // it points into, and the displaced position is not
                // readable in this model, so the pointee read below
                // does not apply to it.
                let projection = match target {
                    RefTarget::Place { projection, .. }
                    | RefTarget::Static { projection, .. }
                    | RefTarget::Alloc { projection, .. } => Some(projection),
                    _ => None,
                };
                if let Some(projection) = projection {
                    if let Some(RefProj::ByteOffset(d)) = projection.last() {
                        if *d != 0 {
                            return Ok(Value::Unit);
                        }
                    }
                }
                let target = self.deref_value(frames, v)?;
                let pointee = self.read_resolved(frames, &target)?;
                self.metadata_of_pointee(frames, &pointee, dest_ty)
            }
            // A built box (a one-field aggregate whose field is
            // the allocation pointer) carries its pointee's
            // metadata, as the box pointer does.
            Value::Enum { fields, .. } if fields.len() == 1 => {
                let target = self.deref_value(frames, v)?;
                let pointee = self.read_resolved(frames, &target)?;
                self.metadata_of_pointee(frames, &pointee, dest_ty)
            }
            other => Err(InterpError::type_mismatch(
                "PtrMetadata",
                "fat pointer",
                other.kind_name(),
            )),
        }
    }

    /// The metadata denoted by a pointee value (see
    /// [`Self::ptr_metadata`]). A value with a dynamically sized
    /// tail — a custom DST struct value, or the array value a
    /// slice pointer names — carries the tail's metadata: the
    /// metadata of the field that carries the length (searched
    /// from the last field backwards), recursively, the element
    /// count, or a sparse vector's own length.
    /// The recursion is gated on the destination being
    /// length-typed: MIR types a sized pointee's metadata as
    /// `()`, so a `usize` destination is exactly the case where
    /// the metadata is a length; every other shape's metadata is
    /// the unit value.
    fn metadata_of_pointee(
        &self,
        frames: &[Frame],
        pointee: &Value,
        dest_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match pointee {
            Value::FatRef { len, .. } => Ok(Value::Uint(*len)),
            Value::Str(s) => Ok(Value::Uint(s.len() as u128)),
            Value::DynRef { .. } => Ok(pointee.clone()),
            Value::Ref(_) => self.ptr_metadata(frames, pointee, dest_ty),
            Value::Tuple(items) => {
                if matches!(dest_ty, Some(Ty::Uint(_))) {
                    Ok(Value::Uint(items.len() as u128))
                } else {
                    Ok(Value::Unit)
                }
            }
            // A byte-string pointee (the storage a byte-slice
            // or C-string pointer names, e.g. the `[c_char]`
            // field of a `CStr` value over a byte-string
            // constant): the same length metadata as an array
            // value's, under the same destination gate.
            Value::Bytes(b) => {
                if matches!(dest_ty, Some(Ty::Uint(_))) {
                    Ok(Value::Uint(b.len() as u128))
                } else {
                    Ok(Value::Unit)
                }
            }
            Value::Enum { fields, .. } => {
                if matches!(dest_ty, Some(Ty::Uint(_))) {
                    // The length lives in whichever field
                    // carries it: the last field first (a slice
                    // struct's buffer), then the rest — a
                    // vector value's length is its first field,
                    // behind the buffer and capacity fields.
                    // The first field (from the right) whose
                    // own metadata is a length wins; a field
                    // with no length metadata does not stop the
                    // search.
                    let mut found = Value::Unit;
                    for field in fields.iter().rev() {
                        let field = field.clone();
                        let meta =
                            self.metadata_of_pointee(frames, &field, dest_ty)?;
                        if matches!(meta, Value::Uint(_)) {
                            found = meta;
                            break;
                        }
                    }
                    Ok(found)
                } else {
                    Ok(Value::Unit)
                }
            }
            // A sparse vector value's own length is its slice
            // metadata, under the same destination gate.
            Value::SparseVec { len, .. } => {
                if matches!(dest_ty, Some(Ty::Uint(_))) {
                    Ok(Value::Uint(*len))
                } else {
                    Ok(Value::Unit)
                }
            }
            _ => Ok(Value::Unit),
        }
    }

    // ----- rvalues -----------------------------------------------------------

    /// Record the target of a thin pointer value being exposed
    /// as the integer `addr` (see the `exposed_targets` field).
    /// Address-only and null pointers record nothing: their
    /// integer form already denotes all they are.
    pub(crate) fn note_exposed_target(&self, addr: u64, value: &Value) {
        if let Value::Ref(target) = value {
            if !matches!(target, RefTarget::Address(_) | RefTarget::Null) {
                self.exposed_targets
                    .borrow_mut()
                    .insert(addr, target.clone());
            }
        }
        // A function pointer's address denotes the function
        // itself: record the identity so a pointer shape
        // constructed from the address recovers the callable
        // value (see the `exposed_fn_ptrs` field). A
        // non-capturing closure value is a function pointer
        // (it coerces to one) and records the same way.
        if matches!(value, Value::FnPtr(_))
            || matches!(value, Value::Closure { env, .. } if env.is_empty())
        {
            self.exposed_fn_ptrs
                .borrow_mut()
                .insert(addr, value.clone());
        }
    }

    /// The pointer value materialised from the integer `addr`:
    /// the target recorded when that address was exposed (see
    /// [`Self::note_exposed_target`]), or an address-only
    /// pointer for an address that was never exposed.
    pub(crate) fn materialized_ptr(&self, addr: u64) -> Value {
        if let Some(target) = self.exposed_targets.borrow().get(&addr) {
            return Value::Ref(target.clone());
        }
        Value::Ref(RefTarget::Address(addr))
    }

    /// The wrapper ADT value holding the pointer materialised
    /// from `addr` (see [`Self::materialized_ptr`]) in its
    /// payload field, with the unique values in its zero-sized
    /// fields: the result of the
    /// integer/enum-to-pointer-wrapper transmutes.
    pub(crate) fn ptr_wrapped_value(
        &self,
        addr: u64,
        payload_index: u32,
        field_count: usize,
        fill: &[(u32, Constant)],
    ) -> InterpResult<Value> {
        let mut fields = vec![Value::Uninit; field_count];
        fields[payload_index as usize] = self.materialized_ptr(addr);
        for (index, constant) in fill {
            fields[*index as usize] = self.eval_constant(constant)?;
        }
        Ok(Value::Enum {
            variant: 0,
            fields,
        })
    }

    /// The thin data pointer a trait-object value denotes: the
    /// pointer word of the fat pointer, with the metadata
    /// dropped (the extraction a cast from a `dyn` pointer to a
    /// sized pointer performs). The base a `DynRef` records is
    /// the data pointer in the shape the sized value had: a
    /// reference base is the pointer itself; a box-shaped base
    /// holds it in its field-0 wrapper chain, or holds the
    /// pointee itself as the only field of a one-field box; an
    /// upcast chain (a box whose field is itself a `DynRef`)
    /// unwraps one layer at a time; a slice base unwraps to its
    /// own data pointer. A reference-counted handle's data
    /// pointer is the pointee slot inside its inner allocation
    /// — the same position `deref_value` resolves for the
    /// handle.
    pub(crate) fn dyn_data_pointer(&self, base: &Value) -> Value {
        // The handle reading is licensed by the allocation's
        // contents (`rc_inner_id`), not by the base's shape
        // alone: a plain aggregate can share the handle's shape
        // (a `CString` value, above all), and reading it as a
        // handle would name a position inside its buffer
        // instead of the value itself.
        if let Some(id) = self.rc_inner_id(base) {
            return Value::Ref(RefTarget::Alloc {
                id,
                projection: vec![RefProj::Field(2)],
            });
        }
        // The outermost aggregate decides what the base is (the
        // discrimination `deref_value` applies to a `DynRef`
        // base). A one-field aggregate whose only field is an
        // allocation pointer is a built box, and that pointer is
        // the data pointer. A one-field aggregate holding any
        // other value holds the pointee itself inline — the
        // shape the panic-cleanup and thread-join models
        // fabricate for a caught payload, and the shape an
        // aggregate-built sized box has — so the data pointer
        // denotes that field's slot and the walk stops there:
        // descending further would name a position inside the
        // payload, not the payload. Any other aggregate is a
        // pointer wrapper (`Unique`, `NonNull`, an
        // aggregate-built `Box` with its `Unique` at field 0
        // beside the allocator field): follow its field 0 until
        // the walk reaches the pointer itself, or a value stored
        // inline at the position the walk reached. An upcast or
        // slice layer restarts the walk at the value it denotes,
        // which is a self-contained value in this model.
        let mut root = base.clone();
        let mut cur = base.clone();
        let mut path: Vec<RefProj> = Vec::new();
        loop {
            // The aggregate forms the walk descends, as a field
            // list: an ADT value or a tuple value.
            let fields = match &cur {
                Value::Enum { fields, .. } if !fields.is_empty() => Some(fields.clone()),
                Value::Tuple(items) if !items.is_empty() => Some(items.clone()),
                _ => None,
            };
            if let Some(fields) = fields {
                if path.is_empty() && fields.len() == 1 {
                    match &fields[0] {
                        // A built box: its only field is the
                        // allocation pointer, which is the data
                        // pointer itself.
                        Value::Ref(RefTarget::Alloc { .. }) => {
                            return fields[0].clone();
                        }
                        // An upcast layer sits in the box: keep
                        // walking from the value it denotes.
                        Value::DynRef { .. } | Value::FatRef { .. } => {
                            cur = fields[0].clone();
                            continue;
                        }
                        // An inline pointee: the field is the
                        // payload itself, so the data pointer
                        // denotes that field's slot in the base.
                        _ => {
                            return Value::Ref(RefTarget::Value(
                                Box::new(root),
                                vec![RefProj::Field(0)],
                            ));
                        }
                    }
                }
                path.push(RefProj::Field(0));
                cur = fields[0].clone();
                continue;
            }
            match cur {
                Value::DynRef { base, .. } | Value::FatRef { base, .. } => {
                    root = (*base).clone();
                    cur = (*base).clone();
                    path.clear();
                }
                Value::Ref(_) => return cur,
                other => {
                    // The pointee is stored inline in the base
                    // value (an aggregate-shaped box): the data
                    // pointer denotes the position the walk
                    // reached inside that value.
                    return if path.is_empty() {
                        other
                    } else {
                        Value::Ref(RefTarget::Value(Box::new(root), path))
                    };
                }
            }
        }
    }
}

/// Whether `value` has the shape of a value of type `ty`, for the
/// shapes the copy machinery must tell apart: scalars directly,
/// tuples and arrays element by element, ADT values as variant
/// aggregates, references as resolved pointers. Anything
/// unrecognised does not fit, so callers fall back to treating
/// the value by its own shape.
/// Whether `ty` is a `MaybeUninit` union. A `MaybeUninit`
/// value's canonical form is the union form -- a one-field
/// `Enum` holding the payload -- and the payload itself may be
/// uninitialized: uninitialized content is this type's ordinary
/// content, not an unwritten place. The places that move
/// `MaybeUninit` values therefore judge them by this type (see
/// `normalize_for_ty`, `exec_return` and
/// `exec_copy_nonoverlapping`), never by whether the payload
/// happens to be initialized.
pub(crate) fn is_maybe_uninit_ty(ty: &Ty) -> bool {
    matches!(
        ty,
        Ty::Adt(name)
            if name.starts_with("std::mem::MaybeUninit")
                || name.starts_with("core::mem::MaybeUninit")
    )
}

/// The canonical union form of a `MaybeUninit` element value:
/// storage read for a `MaybeUninit` place holds the payload in
/// whatever form it was stored (the union form itself, the bare
/// payload a byte-wise write left, or the uninitialized
/// marker); the value of the `MaybeUninit` place is the union
/// value over that content.
pub(crate) fn maybe_uninit_form(value: Value) -> Value {
    match &value {
        Value::Enum { variant: 0, fields } if fields.len() == 1 => value,
        _ => Value::Enum {
            variant: 0,
            fields: vec![value],
        },
    }
}

pub(crate) fn value_fits_ty(value: &Value, ty: &Ty) -> bool {
    match (value, ty) {
        (Value::Unit, Ty::Unit) => true,
        (Value::Bool(_), Ty::Bool) => true,
        (Value::Int(_), Ty::Int(_)) => true,
        (Value::Uint(_), Ty::Uint(_)) => true,
        (Value::Float(_), Ty::Float(_)) => true,
        (Value::Float128(_), Ty::Float(_)) => true,
        (Value::Tuple(items), Ty::Tuple(tys)) => {
            items.len() == tys.len()
                && items
                    .iter()
                    .zip(tys.iter())
                    .all(|(v, t)| value_fits_ty(v, t))
        }
        (Value::Tuple(items), Ty::Array(elem, n)) => {
            items.len() as u64 == *n
                && items.iter().all(|v| value_fits_ty(v, elem))
        }
        (Value::Bytes(b), Ty::Array(elem, n)) => {
            matches!(&**elem, Ty::Uint(UintTy::U8)) && b.len() as u64 == *n
        }
        (Value::Enum { .. }, Ty::Adt(_)) => true,
        (Value::Str(_), Ty::Str) => true,
        // A string value is the value model's representation of
        // the `String` ADT itself (the owned buffer's contents,
        // with no separate aggregate): a `Str` value under an
        // ADT-typed position is one `String` element, not a byte
        // sequence to index into. No other ADT is represented by
        // a bare `Str`, so the arm cannot bless a value of the
        // wrong type in a well-typed program.
        (Value::Str(_), Ty::Adt(_)) => true,
        // The same representation serves an `&str` value stored
        // in an aggregate: the element is the string data,
        // re-formed as a reference by the declared type on
        // ordinary reads.
        (Value::Str(_), Ty::Ref(pointee)) if matches!(&**pointee, Ty::Str) => {
            true
        }
        (Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }, Ty::Ref(_)) => {
            true
        }
        (Value::FnPtr(_), Ty::FnPtr) => true,
        (Value::FnPtr(_), Ty::FnDef(_)) => true,
        (Value::Closure { .. }, Ty::Closure(..)) => true,
        (Value::Coroutine { .. }, Ty::Coroutine(_)) => true,
        _ => false,
    }
}
