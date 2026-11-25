//! Value shape helpers: tags, defaults shapes and small constructors.

use std::borrow::Cow;
use std::cmp::Ordering;


use crate::error::{InterpError, InterpResult};
use crate::interp::{
    impl_qualified_self_type, rc_handle_id, Resolved,
};
use crate::ir::{BinOp, FloatTy, IntTy, Ty, UintTy};
use crate::value::{RefProj, RefTarget, Value};

use super::*;

/// Whether a rendered Self type tag denotes the ADT `name`: either
/// the bare name (a type whose generic arguments are all lifetimes
/// renders without them, e.g. `std::str::Chars`) or `name<...>`
/// with type arguments.
/// The abstract `RefCell` value, in the real struct's shape: field 0
/// is the borrow flag as a `Cell` aggregate (`Cell { UnsafeCell {
/// isize } }`, starting at 0 = unused), field 1 is the `UnsafeCell`
/// aggregate holding the value. Building the natural shape keeps
/// collected bodies that project `RefCell` fields (e.g. `into_inner`)
/// working on builtin-constructed cells.
/// A `Cell` value of the natural shape `Cell { UnsafeCell { value
/// } }`: the shape a collected `Cell` aggregate and a decoded
/// constant produce, which the `Cell` builtins and the
/// `UnsafeCell::get` pointer path must agree on.
/// The parts of a `Vec` value in its real aggregate form
/// (`[buf, len]`): the buffer value and the length, when the
/// value has exactly that shape. A modelled vector (its elements
/// as the fields) does not qualify: its second field is an
/// element, never a bare length beside a `RawVec`-shaped first
/// field.
pub(crate) fn vec_real_aggregate_parts(value: &Value) -> Option<(Value, usize)> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    if fields.len() != 2 {
        return None;
    }
    let Value::Uint(len) = fields[1] else {
        return None;
    };
    let Value::Enum {
        fields: buf_fields, ..
    } = &fields[0]
    else {
        return None;
    };
    if buf_fields.len() != 2 {
        return None;
    }
    let Value::Enum {
        fields: inner_fields, ..
    } = &buf_fields[0]
    else {
        return None;
    };
    if inner_fields.len() != 3 {
        return None;
    }
    Some((fields[0].clone(), len as usize))
}

/// The capacity a real-form `RawVec` buffer value declares:
/// the `RawVecInner` triple's middle field (the
/// `UsizeNoHighBit` capacity, a one-field wrapper around the
/// count). `None` when the buffer does not have that shape.
pub(crate) fn vec_real_aggregate_cap(buf: &Value) -> Option<u128> {
    let Value::Enum {
        fields: buf_fields, ..
    } = buf
    else {
        return None;
    };
    let Value::Enum {
        fields: inner_fields,
        ..
    } = buf_fields.first()?
    else {
        return None;
    };
    if inner_fields.len() != 3 {
        return None;
    }
    fn count_of(value: &Value) -> Option<u128> {
        match value {
            Value::Uint(n) => Some(*n),
            Value::Int(n) if *n >= 0 => Some(*n as u128),
            Value::Enum { fields, .. } if fields.len() == 1 => count_of(&fields[0]),
            Value::Tuple(items) if items.len() == 1 => count_of(&items[0]),
            _ => None,
        }
    }
    count_of(&inner_fields[1])
}

/// A `RawVec` buffer value over the element block pointer
/// `ptr`, with capacity `cap`: the `RawVecInner` triple
/// (`NonNull` pointer, `UsizeNoHighBit` capacity, the global
/// allocator marker) plus the trailing `PhantomData`.
pub(super) fn raw_vec_value(ptr: Value, cap: usize) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Enum {
                        variant: 0,
                        fields: vec![ptr],
                    },
                    Value::Enum {
                        variant: 0,
                        fields: vec![Value::Uint(cap as u128)],
                    },
                    Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    },
                ],
            },
            Value::Enum {
                variant: 0,
                fields: Vec::new(),
            },
        ],
    }
}

/// A `VecDeque` value in its real aggregate form over buffer
/// `buf` holding `len` elements from index zero: the wrapped
/// head index, the length, and the buffer.
pub(super) fn vec_deque_value(buf: Value, len: usize) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Enum {
                variant: 0,
                fields: vec![Value::Uint(0)],
            },
            Value::Uint(len as u128),
            buf,
        ],
    }
}

pub(super) fn cell_value(value: Value) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![value],
        }],
    }
}

/// The value a `Cell` aggregate of the natural shape holds;
/// `None` for any other value.
/// The single field of a one-field aggregate value: the shape of
/// the std newtype wrappers (`Pin`, `ManuallyDrop`,
/// `AssertUnwindSafe`).
/// The types whose data the value model stores as a string:
/// the string, byte-string and path types and `str` itself.
pub(super) fn is_string_like_ty(ty: &str) -> bool {
    matches!(
        ty,
        "str"
            | "std::string::String"
            | "alloc::string::String"
            | "std::ffi::OsString"
            | "std::ffi::OsStr"
            | "std::path::PathBuf"
            | "std::path::Path"
    )
}

/// The generic arguments of a rendered type, split at top-level
/// commas: `std::boxed::Box<T, A>` yields `["T", "A"]`.
pub(super) fn generic_args_of(ty: &str) -> Vec<String> {
    let Some(open) = ty.find('<') else {
        return Vec::new();
    };
    let mut depth = 0usize;
    let mut close = None;
    for (i, c) in ty[open..].char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return Vec::new();
    };
    let inner = &ty[open + 1..close];
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in inner.char_indices() {
        match c {
            '<' | '[' | '(' => depth += 1,
            '>' | ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(inner[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(inner[start..].trim().to_string());
    parts.retain(|p| !p.is_empty());
    parts
}

/// The first generic argument of a rendered type, if any.
pub(super) fn first_generic_param(ty: &str) -> Option<String> {
    generic_args_of(ty).into_iter().next()
}

/// The last generic argument of a rendered type, if any (the
/// borrowed type of a `Cow<'_, B>` is its last parameter).
pub(super) fn last_generic_param(ty: &str) -> Option<String> {
    generic_args_of(ty).into_iter().last()
}

pub(super) fn one_field_value(value: Value, what: &str) -> InterpResult<Value> {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => {
            Ok(fields.into_iter().next().expect("one field"))
        }
        other => Err(InterpError::type_mismatch(
            what,
            "one-field aggregate",
            other.kind_name(),
        )),
    }
}

/// The value produced by dereferencing a pointer-shaped value: a
/// reference, fat reference or dyn reference is its own
/// dereference; a built `Box` yields the pointer to its allocation
/// (an aggregate-built box a reference to the held pointee value);
/// an `Rc` / `Arc` handle yields a reference to the stored value
/// field. This is the shared tail of the `Box` and `Rc` arms of
/// the `Deref` / `DerefMut` builtins, factored out for `Pin<P>`,
/// which derefs as the pointer it wraps.
pub(super) fn deref_pointer_value(pointer: Value) -> InterpResult<Value> {
    if let Some(id) = rc_handle_id(&pointer) {
        return Ok(Value::Ref(RefTarget::Alloc {
            id,
            projection: vec![RefProj::Field(2)],
        }));
    }
    match pointer {
        Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => Ok(pointer),
        Value::Enum { fields, .. } if fields.len() == 1 => {
            let inner = fields.into_iter().next().expect("one field");
            match inner {
                Value::Ref(RefTarget::Alloc { .. }) => Ok(inner),
                other => Ok(Value::Ref(RefTarget::value(other))),
            }
        }
        other => Err(InterpError::type_mismatch(
            "dereference of a wrapped pointer",
            "reference, box or handle value",
            other.kind_name(),
        )),
    }
}

pub(super) fn cell_held(state: &Value) -> Option<&Value> {
    match state {
        Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
            Value::Enum { fields, .. } if fields.len() == 1 => Some(&fields[0]),
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn refcell_value(value: Value) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Enum {
                variant: 0,
                fields: vec![Value::Enum {
                    variant: 0,
                    fields: vec![Value::Int(0)],
                }],
            },
            Value::Enum {
                variant: 0,
                fields: vec![value],
            },
        ],
    }
}

/// A `Ref` / `RefMut` guard value in this model: field 0 is a
/// reference to the cell's value (through the `UnsafeCell` wrapper),
/// field 1 is a reference to the cell's borrow-flag `Cell`, which is
/// what the guard's drop glue (`DropPlan::RefCellRelease`) releases
/// through.
pub(super) fn refcell_guard(cell: &RefTarget) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Ref(crate::interp::refcell_sub_target(
                cell,
                &[RefProj::Field(1), RefProj::Field(0)],
            )),
            Value::Ref(crate::interp::refcell_sub_target(
                cell,
                &[RefProj::Field(0)],
            )),
        ],
    }
}

pub(super) fn some_value(value: Value) -> Value {
    Value::Enum {
        variant: 1,
        fields: vec![value],
    }
}

pub(super) fn none_value() -> Value {
    Value::Enum {
        variant: 0,
        fields: Vec::new(),
    }
}

/// Split the `Option` value an `Iterator::next` transition
/// returns into its payload; `None` for the `None` variant.
pub(super) fn option_payload(value: Value) -> InterpResult<Option<Value>> {
    match value {
        Value::Enum { variant: 0, .. } => Ok(None),
        Value::Enum {
            variant: 1,
            mut fields,
        } if fields.len() == 1 => Ok(Some(fields.pop().expect("len checked"))),
        other => Err(InterpError::type_mismatch(
            "Iterator::next result",
            "an Option value",
            other.kind_name(),
        )),
    }
}

/// A `core::cmp::Ordering` as a value: the variants in declaration
/// order (Less, Equal, Greater).
pub(super) fn ordering_value(ord: Ordering) -> Value {
    Value::Enum {
        variant: match ord {
            Ordering::Less => 0,
            Ordering::Equal => 1,
            Ordering::Greater => 2,
        },
        fields: Vec::new(),
    }
}

/// The IR type of a primitive tag (`i32`, `usize`, `f64`,
/// `bool`): the Self types the arithmetic trait models compute
/// on. `None` for anything else.
fn primitive_ty(tag: &str) -> Option<Ty> {
    Some(match tag {
        "i8" => Ty::Int(IntTy::I8),
        "i16" => Ty::Int(IntTy::I16),
        "i32" => Ty::Int(IntTy::I32),
        "i64" => Ty::Int(IntTy::I64),
        "i128" => Ty::Int(IntTy::I128),
        "isize" => Ty::Int(IntTy::Isize),
        "u8" => Ty::Uint(UintTy::U8),
        "u16" => Ty::Uint(UintTy::U16),
        "u32" => Ty::Uint(UintTy::U32),
        "u64" => Ty::Uint(UintTy::U64),
        "u128" => Ty::Uint(UintTy::U128),
        "usize" => Ty::Uint(UintTy::Usize),
        "f32" => Ty::Float(FloatTy::F32),
        "f64" => Ty::Float(FloatTy::F64),
        "bool" => Ty::Bool,
        _ => return None,
    })
}

/// The Self type of an impl-qualified arithmetic trait call, as
/// an IR type: the only width / signedness source the model
/// has, so a name that does not carry a primitive Self type
/// does not qualify.
pub(super) fn primitive_self_ty(name: &str) -> InterpResult<Ty> {
    let self_tag = impl_qualified_self_type(name).ok_or_else(|| {
        unsupported(format!(
            "arithmetic trait call {name} does not name its Self type"
        ))
    })?;
    primitive_ty(&self_tag).ok_or_else(|| {
        unsupported(format!(
            "arithmetic trait call for {self_tag} is not runtime-supported"
        ))
    })
}

/// Core's overflow panic messages for the arithmetic traits,
/// verbatim (they are the messages the checked impls raise).
pub(super) fn primitive_overflow_message(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "attempt to add with overflow",
        BinOp::Sub => "attempt to subtract with overflow",
        BinOp::Mul => "attempt to multiply with overflow",
        BinOp::Div => "attempt to divide with overflow",
        BinOp::Rem => "attempt to calculate the remainder with overflow",
        BinOp::Shl => "attempt to shift left with overflow",
        BinOp::Shr => "attempt to shift right with overflow",
        _ => "attempt to perform an arithmetic operation with overflow",
    }
}

/// The element-entry suffix of a projection: a trailing run of
/// field-0 entry steps preceded by the index step that selects
/// the first entered element. This is the shape a fat pointer's
/// base takes after a transparent-wrapper pointer cast into a
/// sequence (a `*const [MaybeUninit<T>]` \u{2192} `*const [T]` cast:
/// the base denotes the container's element at that index,
/// entered by the field steps — see the `FatRef` arm of
/// `ptr_field_value`). Returns the container projection (the
/// steps before the index step), the index, and the entry steps.
pub(super) fn element_entry_suffix(projection: &[RefProj]) -> Option<(Vec<RefProj>, usize, Vec<RefProj>)> {
    let mut end = projection.len();
    while end > 0 && matches!(projection[end - 1], RefProj::Field(0)) {
        end -= 1;
    }
    if end == projection.len() || end == 0 {
        return None;
    }
    let RefProj::Index(first) = projection[end - 1] else {
        return None;
    };
    Some((
        projection[..end - 1].to_vec(),
        first,
        projection[end..].to_vec(),
    ))
}

/// The projection denoting element `idx` of the sequence a
/// projection with an element-entry suffix denotes: the suffix's
/// index step advances by `idx`, the entry steps stay after it.
pub(super) fn advance_entry_projection(projection: &[RefProj], idx: usize) -> Option<Vec<RefProj>> {
    let (mut out, first, entry) = element_entry_suffix(projection)?;
    out.push(RefProj::Index(first.checked_add(idx)?));
    out.extend(entry);
    Some(out)
}

/// A resolved target with its projection replaced.
pub(super) fn resolved_with_projection(target: &Resolved, projection: Vec<RefProj>) -> Resolved {
    match target {
        Resolved::Place {
            frame_pos, local, ..
        } => Resolved::Place {
            frame_pos: *frame_pos,
            local: *local,
            projection,
        },
        Resolved::Static { name, .. } => Resolved::Static {
            name: name.clone(),
            projection,
        },
        Resolved::Alloc { id, .. } => Resolved::Alloc {
            id: *id,
            projection,
        },
        Resolved::Value(value, _) => Resolved::Value(value.clone(), projection),
    }
}

/// Whether a resolved target denotes one element of a container
/// (its projection ends at an index step): a fat pointer over
/// such a target denotes the elements from that position on, not
/// the element itself.
pub(crate) fn target_ends_at_element(target: &Resolved) -> bool {
    let projection = match target {
        Resolved::Place { projection, .. }
        | Resolved::Static { projection, .. }
        | Resolved::Alloc { projection, .. }
        | Resolved::Value(_, projection) => projection,
    };
    matches!(projection.last(), Some(RefProj::Index(_)))
}

/// The elements of an already-resolved value, borrowed where
/// possible: the borrowing form of [`value_elements`], with the
/// same results and errors.
pub(super) fn elements_of<'a>(value: Cow<'a, Value>) -> InterpResult<SliceElems<'a>> {
    match value {
        Cow::Borrowed(v) => Ok(match v {
            Value::Tuple(items) => SliceElems::Values(Cow::Borrowed(items)),
            Value::Enum { fields, .. } => SliceElems::Values(Cow::Borrowed(fields)),
            Value::SparseVec { elems, len, .. } => SliceElems::Sparse {
                stored: Cow::Borrowed(elems),
                len: usize::try_from(*len).unwrap_or(usize::MAX),
            },
            Value::Bytes(b) => SliceElems::Bytes(Cow::Borrowed(b)),
            Value::Str(text) => SliceElems::Bytes(Cow::Borrowed(text.as_bytes())),
            other => {
                return Err(InterpError::type_mismatch(
                    "slice elements",
                    "aggregate, byte or string value",
                    other.kind_name(),
                ))
            }
        }),
        Cow::Owned(v) => Ok(match v {
            Value::Tuple(items) => SliceElems::Values(Cow::Owned(items)),
            Value::Enum { fields, .. } => SliceElems::Values(Cow::Owned(fields)),
            Value::SparseVec { elems, len, .. } => SliceElems::Sparse {
                stored: Cow::Owned(elems),
                len: usize::try_from(len).unwrap_or(usize::MAX),
            },
            Value::Bytes(b) => SliceElems::Bytes(Cow::Owned(b)),
            Value::Str(text) => SliceElems::Bytes(Cow::Owned(text.into_bytes())),
            other => {
                return Err(InterpError::type_mismatch(
                    "slice elements",
                    "aggregate, byte or string value",
                    other.kind_name(),
                ))
            }
        }),
    }
}

/// The element values of an aggregate-like value: tuple items, ADT
/// fields, or the bytes of a byte/string value as integers.
pub(super) fn value_elements(value: &Value) -> InterpResult<Vec<Value>> {
    Ok(match value {
        Value::Tuple(items) => items.clone(),
        Value::Enum { fields, .. } => fields.clone(),
        Value::Bytes(b) => b.iter().map(|byte| Value::Uint(*byte as u128)).collect(),
        Value::Str(s) => s
            .as_bytes()
            .iter()
            .map(|byte| Value::Uint(*byte as u128))
            .collect(),
        // A sparse vector whose storage covers its declared
        // length (the spare-capacity form a length shrink
        // leaves) yields its declared elements; one whose
        // length runs past the stored elements has no element
        // list to hand out.
        Value::SparseVec { elems, len, .. } => {
            let len = usize::try_from(*len).unwrap_or(usize::MAX);
            if len > elems.len() {
                return Err(InterpError::type_mismatch(
                    "slice elements",
                    "aggregate, byte or string value",
                    "vector value with an unmaterialised tail",
                ));
            }
            elems[..len].to_vec()
        }
        other => {
            return Err(InterpError::type_mismatch(
                "slice elements",
                "aggregate, byte or string value",
                other.kind_name(),
            ))
        }
    })
}

pub(super) fn int_uint_eq(x: i128, y: u128) -> bool {
    x >= 0 && (x as u128) == y
}

pub(super) fn int_uint_cmp(x: i128, y: u128) -> Ordering {
    if x < 0 {
        Ordering::Less
    } else {
        (x as u128).cmp(&y)
    }
}

/// An integer index argument as a `usize`.
/// The (start, end) element bounds of a range index against a
/// receiver of length `len`, applying core's checks in core's
/// order with core's panic messages. Inclusive ends are converted
/// to exclusive ones first (an inclusive end of `usize::MAX`
/// overflows that conversion, which core reports separately).
/// A reference value one sub-slice deeper: `value` must be a
/// `Value::Ref`; the result names the same target with a
/// resolved `RefProj::Subslice` step appended, so reads and
/// writes through it alias the original storage at the
/// sub-slice's offset. A `Value` target (a constant aggregate)
/// carries the step in its projection, like the other targets:
/// reads through it stay exact, and writes through it are
/// refused by the write path, as for any constant.
/// A reference value with its projection replaced.
pub(super) fn ref_with_projection(value: &Value, projection: Vec<RefProj>) -> Value {
    match value {
        Value::Ref(RefTarget::Place {
            frame, local, ..
        }) => Value::Ref(RefTarget::place(*frame, *local, projection)),
        Value::Ref(RefTarget::Static {
            name: static_name, ..
        }) => Value::Ref(RefTarget::Static {
            name: static_name.clone(),
            projection,
        }),
        Value::Ref(RefTarget::Alloc { id, .. }) => Value::Ref(RefTarget::Alloc {
            id: *id,
            projection,
        }),
        Value::Ref(RefTarget::Value(v, _)) => {
            Value::Ref(RefTarget::Value(v.clone(), projection))
        }
        other => other.clone(),
    }
}

pub(super) fn subslice_ref(value: &Value, offset: usize, len: usize, _name: &str) -> InterpResult<Value> {
    let step = RefProj::Subslice { offset, len };
    match value {
        Value::Ref(RefTarget::Place {
            frame,
            local,
            projection,
        }) => {
            let mut projection = projection.clone();
            projection.push(step);
            Ok(Value::Ref(RefTarget::place(*frame, *local, projection)))
        }
        Value::Ref(RefTarget::Static {
            name: static_name,
            projection,
        }) => {
            let mut projection = projection.clone();
            projection.push(step);
            Ok(Value::Ref(RefTarget::Static {
                name: static_name.clone(),
                projection,
            }))
        }
        Value::Ref(RefTarget::Alloc { id, projection }) => {
            let mut projection = projection.clone();
            projection.push(step);
            Ok(Value::Ref(RefTarget::Alloc {
                id: *id,
                projection,
            }))
        }
        Value::Ref(RefTarget::Value(v, projection)) => {
            let mut projection = projection.clone();
            projection.push(step);
            Ok(Value::Ref(RefTarget::Value(v.clone(), projection)))
        }
        other => Err(InterpError::type_mismatch(
            "IndexMut::index_mut",
            "reference value",
            other.kind_name(),
        )),
    }
}

pub(super) fn range_bounds(
    kind: &RangeKindForIndex,
    bound: &dyn Fn(usize) -> InterpResult<usize>,
    len: usize,
) -> InterpResult<(usize, usize)> {
    use RangeKindForIndex::*;
    let (start, end) = match kind {
        Full => (0, len),
        Range => (bound(0)?, bound(1)?),
        From => (bound(0)?, len),
        To => (0, bound(0)?),
        Inclusive => {
            let start = bound(0)?;
            let end = bound(1)?;
            if end == usize::MAX {
                return Err(InterpError::panic("range end index overflow".to_string()));
            }
            (start, end + 1)
        }
        ToInclusive => {
            let end = bound(0)?;
            if end == usize::MAX {
                return Err(InterpError::panic("range end index overflow".to_string()));
            }
            (0, end + 1)
        }
    };
    if start > end {
        return Err(InterpError::panic(format!(
            "slice index starts at {start} but ends at {end}"
        )));
    }
    if end > len {
        return Err(InterpError::panic(format!(
            "range end index {end} out of range for slice of length {len}"
        )));
    }
    if start > len {
        return Err(InterpError::panic(format!(
            "range start index {start} out of range for slice of length {len}"
        )));
    }
    Ok((start, end))
}

pub(super) enum RangeKindForIndex {
    Full,
    Range,
    From,
    To,
    Inclusive,
    ToInclusive,
}

/// core's message for a string slice index that is not a char
/// boundary (`str::slice_error_fail`).
pub(super) fn str_slice_error(text: &str, start: usize, end: usize) -> String {
    for idx in [start, end] {
        if idx != 0 && idx != text.len() && !text.is_char_boundary(idx) {
            let lo = text[..idx].chars().next_back().map_or(0, |c| idx - c.len_utf8());
            let ch = text[lo..].chars().next().expect("char at boundary");
            let hi = lo + ch.len_utf8();
            return format!(
                "byte index {idx} is not a char boundary; it is inside {ch:?} (bytes {lo}..{hi}) of `{text}`"
            );
        }
    }
    format!("byte index {start} is not a char boundary")
}

pub(super) fn expect_index(value: &Value, name: &str) -> InterpResult<usize> {
    match value {
        Value::Uint(v) => usize::try_from(*v).map_err(|_| {
            InterpError::panic(format!("index {v} does not fit in usize"))
        }),
        Value::Int(v) if *v >= 0 => usize::try_from(*v).map_err(|_| {
            InterpError::panic(format!("index {v} does not fit in usize"))
        }),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "integer index",
            other.kind_name(),
        )),
    }
}

/// Whether a rendered type tag names an integer type.
pub(super) fn is_integer_tag(tag: &str) -> bool {
    matches!(
        tag,
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64" | "u128"
            | "usize"
    )
}

/// The element tag inside a rendered array type (`T; N`):
/// the part before the top-level semicolon, or `None` when
/// there is none (a slice's inside is not an array's).
pub(super) fn array_elem_tag(inner: &str) -> Option<&str> {
    let mut depth = 0i32;
    for (i, c) in inner.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ';' if depth == 0 => return Some(inner[..i].trim()),
            _ => {}
        }
    }
    None
}

/// The element tag inside a rendered slice type (`[T]`):
/// the whole inside, or `None` when the tag is not a
/// bracket tag or its inside is an array's (`T; N` -- the
/// semicolon reading [`array_elem_tag`] performs).
pub(super) fn slice_pointee_elem_tag(ty: &str) -> Option<&str> {
    let inner = ty.strip_prefix('[')?.strip_suffix(']')?;
    if array_elem_tag(inner).is_some() {
        return None;
    }
    Some(inner.trim())
}

/// Split a rendered type list on the commas that sit at nesting
/// depth zero (for tuple/array type tags).
pub(super) fn split_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    parts.into_iter().filter(|p| !p.is_empty()).collect()
}

/// The uninitialized value of the shape a rendered type tag fixes,
/// for `Box::new_uninit`: an array type is a tuple of uninitialized
/// elements (the count is part of the type), a `MaybeUninit<T>`
/// wrapper is transparent, and any other type is a single
/// uninitialized value. `None` for an array too large to materialize.
pub(super) fn uninit_value_for_tag(tag: &str) -> Option<Value> {
    let tag = tag.trim();
    if let Some(inner) = tag
        .strip_prefix("std::mem::MaybeUninit<")
        .and_then(|t| t.strip_suffix('>'))
    {
        return uninit_value_for_tag(inner);
    }
    if tag.starts_with('[') && tag.ends_with(']') {
        let inner = &tag[1..tag.len() - 1];
        if let Some((_, n)) = inner.rsplit_once(';') {
            if let Ok(n) = n.trim().parse::<usize>() {
                if n > 4096 {
                    return None;
                }
                // The pointee is a `MaybeUninit<[T; N]>`, and the
                // `vec!` expansion writes the elements through the
                // place `(*ptr).1.0`: the union's value field (index
                // 1), then `ManuallyDrop`'s only field. The value is
                // shaped to that layout -- field 0 is the unit
                // `uninit` field, field 1 wraps the uninitialized
                // element array -- so the projected write lands on
                // the array.
                return Some(Value::Enum {
                    variant: 0,
                    fields: vec![
                        Value::Unit,
                        Value::Enum {
                            variant: 0,
                            fields: vec![Value::Tuple(vec![Value::Uninit; n])],
                        },
                    ],
                });
            }
        }
    }
    Some(Value::Uninit)
}

/// The uninitialized form of a value of the type a rendered
/// tag names, as a `Vec` element slot holds it when the
/// vector's declared length grows over storage nothing was
/// written to (see `vec_with_len`): a `MaybeUninit<T>` element
/// is the union value whose payload slot carries the payload
/// type's own uninitialized form — the shape the
/// `MaybeUninit::uninit()` construction materializes, so a
/// slot the length declares but no write reached reads back
/// as that value, and using its payload is the uninitialized
/// read. Every other type's uninitialized form is the bare
/// `Uninit` marker: reading such a slot is an uninitialized
/// read, exactly as before.
pub(crate) fn uninit_elem_value_for_tag(tag: &str) -> Value {
    let tag = tag.trim();
    for prefix in ["std::mem::MaybeUninit<", "core::mem::MaybeUninit<"] {
        if let Some(inner) = generic_first_arg(tag, prefix) {
            return Value::Enum {
                variant: 0,
                fields: vec![uninit_elem_value_for_tag(inner)],
            };
        }
    }
    Value::Uninit
}

/// The element values inside a `box_assume_init_into_vec_unsafe`
/// argument's pointee: an array value is its element tuple; a
/// `MaybeUninit` wrapper value wraps the array in one of its
/// fields. Anything else is not an array pointee.
pub(super) fn assume_init_elements(pointee: Value) -> Option<Vec<Value>> {
    match pointee {
        Value::Tuple(elems) => Some(elems),
        // The pointee is a `MaybeUninit<[T; N]>`: either a
        // single-field wrapper (the array is the only field) or the
        // layout-shaped value `Box::new_uninit` produces (the array
        // sits in the union's value field, inside `ManuallyDrop`).
        // Search the wrapper fields for the element array.
        Value::Enum { fields, .. } if fields.len() <= 2 => {
            fields.into_iter().find_map(assume_init_elements)
        }
        _ => None,
    }
}

/// Fill the unwritten slots of `value` from the one value of
/// its type: a slot holding the uninitialized marker takes
/// the corresponding part of `canonical`, containers recurse
/// where both sides have the same shape, and a written leaf
/// is kept as it stands. This is the materialization a
/// `box_assume_init_into_vec_unsafe` call performs: the call
/// asserts the source array is fully initialized, so a slot
/// still holding the marker is one whose write was elided
/// because the value occupies no bytes.
pub(super) fn fill_uninit(value: Value, canonical: &Value) -> Value {
    match (value, canonical) {
        (Value::Uninit, _) => canonical.clone(),
        (Value::Tuple(fields), Value::Tuple(canon)) if fields.len() == canon.len() => {
            Value::Tuple(
                fields
                    .into_iter()
                    .zip(canon.iter())
                    .map(|(field, c)| fill_uninit(field, c))
                    .collect(),
            )
        }
        (
            Value::Enum {
                variant,
                fields,
            },
            Value::Enum {
                variant: cvariant,
                fields: canon,
            },
        ) if variant == *cvariant && fields.len() == canon.len() => Value::Enum {
            variant,
            fields: fields
                .into_iter()
                .zip(canon.iter())
                .map(|(field, c)| fill_uninit(field, c))
                .collect(),
        },
        (other, _) => other,
    }
}

/// The `Default` value for a rendered type tag, for the types whose
/// default is fixed by the type alone in this value model.
/// The (bit width, signedness) of a primitive integer type tag
/// as rendered by `ty_tag` (`i32`, `usize`, ...). `isize` /
/// `usize` are 64-bit on every platform this runs on.
/// An element count argument (of a memory intrinsic), as a
/// `usize`. Mirrors the count handling of the interpreter's
/// element copy.
pub(super) fn element_count(value: &Value, name: &str) -> InterpResult<usize> {
    match value {
        Value::Uint(n) => usize::try_from(*n).map_err(|_| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "count that fits in usize",
                format!("{n}"),
            )
        }),
        Value::Int(n) if *n >= 0 => Ok(*n as usize),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "unsigned integer count",
            other.kind_name(),
        )),
    }
}

/// The most scalar leaves a materialised fill value (a
/// `mem::zeroed` result, a `write_bytes` fill) may hold. The
/// interpreter's values are trees, so a fill of a large array
/// type is a tree with one value per element: the bound admits
/// a `mem::zeroed::<[u8; 1 << 20]>()` — the big-buffer shape
/// real programs use, the same order as the conversion bound
/// on `Rvalue::Repeat` aggregates — while keeping an absurd
/// type from exhausting memory.
const MAX_FILL_LEAVES: usize = 1 << 21;

/// The number of scalar leaves in the value of the type `tag`
/// names, for the fill functions' materialisation budget: a
/// scalar is one leaf, a tuple sums its parts, an array
/// multiplies. `None` for a tag outside the fill grammar.
pub(super) fn tag_leaf_count(tag: &str) -> Option<usize> {
    let tag = tag.trim();
    match tag {
        "bool" | "char" | "f16" | "f32" | "f64" | "f128" | "()" => Some(1),
        _ if int_tag(tag).is_some() => Some(1),
        _ if tag.starts_with('(') && tag.ends_with(')') => {
            let inner = &tag[1..tag.len() - 1];
            let mut total: usize = 0;
            if !inner.trim().is_empty() {
                for part in split_top_level(inner) {
                    total = total.checked_add(tag_leaf_count(part)?)?;
                }
            }
            Some(total)
        }
        _ if tag.starts_with('[') && tag.ends_with(']') => {
            let inner = &tag[1..tag.len() - 1];
            let (elem, n) = inner.rsplit_once(';')?;
            let n: usize = n.trim().parse().ok()?;
            n.checked_mul(tag_leaf_count(elem)?)
        }
        _ => None,
    }
}

/// The number of scalar leaves in a materialised `value`,
/// mirroring [`tag_leaf_count`]: aggregates sum their parts,
/// anything else is one leaf -- including an uninitialised leaf,
/// which still occupies its slot in the aggregate's shape (a
/// materialised-but-uninitialised array has the same leaf count
/// as the filled value `write_bytes` would write into it, and
/// the caller compares exactly those two counts). Byte blocks
/// and strings keep the sequence path (their fills are
/// byte-wise), and a fat reference's pointee count is not a fact
/// of the value alone: those are `None` here.
pub(super) fn value_leaf_count(value: &Value) -> Option<usize> {
    match value {
        Value::Tuple(items) => items.iter().map(value_leaf_count).sum(),
        Value::Enum { fields, .. } => fields.iter().map(value_leaf_count).sum(),
        Value::Coroutine { fields, saved, .. } => {
            let mut total = 0usize;
            for value in fields.iter().chain(saved.iter().flatten()) {
                total += value_leaf_count(value)?;
            }
            Some(total)
        }
        Value::Closure { env, .. } => env.iter().map(value_leaf_count).sum(),
        Value::Bytes(_)
        | Value::Str(_)
        | Value::FatRef { .. }
        | Value::DynRef { .. } => None,
        _ => Some(1),
    }
}

/// The value whose byte pattern is `byte` repeated, for the
/// element type `tag` names — the value `write_bytes` fills with.
/// Only types with a fixed byte interpretation are filled: for
/// any other type an arbitrary byte pattern need not be a valid
/// value, so the fill stays unsupported there. An array of a
/// fillable element type is filled element by element (the
/// shape `mem::zeroed` takes under optimisation, where the
/// intrinsic's type is the whole array), within the
/// materialisation budget.
pub(super) fn byte_fill_value(tag: &str, byte: u8) -> Option<Value> {
    let tag = tag.trim();
    if let Some((bits, signed)) = int_tag(tag) {
        let mut pattern: u128 = 0;
        for _ in 0..bits / 8 {
            pattern = (pattern << 8) | u128::from(byte);
        }
        return Some(if signed {
            Value::Int(sign_extend(pattern, bits))
        } else {
            Value::Uint(pattern)
        });
    }
    Some(match tag {
        "bool" => match byte {
            0 => Value::Bool(false),
            1 => Value::Bool(true),
            _ => return None,
        },
        // A `char` is stored as its code point; the filled code
        // point is the byte repeated four times, valid only when
        // it names a character.
        "char" => Value::Uint(u128::from(char::from_u32(u32::from(byte) * 0x0101_0101)? as u32)),
        "f16" => Value::Float(crate::mir_convert::f16_to_f64(
            u16::from(byte) * 0x0101,
        )),
        "f32" => Value::Float(f64::from(f32::from_bits(u32::from(byte) * 0x0101_0101))),
        "f64" => Value::Float(f64::from_bits(u64::from(byte) * 0x0101_0101_0101_0101)),
        _ if tag.starts_with('[') && tag.ends_with(']') => {
            let inner = &tag[1..tag.len() - 1];
            let (elem, n) = inner.rsplit_once(';')?;
            let n: usize = n.trim().parse().ok()?;
            if tag_leaf_count(tag)? > MAX_FILL_LEAVES {
                return None;
            }
            let elem = byte_fill_value(elem, byte)?;
            Value::Tuple(vec![elem; n])
        }
        _ => return None,
    })
}

/// The alignment an allocator call declares, read
/// permissively: the argument is usually a bare integer,
/// but a `mem::Alignment` (or similar newtype) argument
/// arrives as an aggregate wrapping the integer. Anything
/// unrecognised keeps the pointer-width default the
/// layout fallback has always used.
pub(super) fn alloc_align_arg(v: &Value) -> u64 {
    fn scalar(v: &Value) -> Option<u64> {
        match v {
            Value::Uint(n) => u64::try_from(*n).ok(),
            Value::Int(n) if *n >= 0 => u64::try_from(*n).ok(),
            Value::Tuple(fields) | Value::Enum { fields, .. } => {
                fields.first().and_then(scalar)
            }
            _ => None,
        }
    }
    scalar(v).filter(|a| a.is_power_of_two()).unwrap_or(8)
}

/// An allocation-size argument (`usize`, or a non-negative
/// `isize`-shaped integer) as a byte count.
pub(super) fn alloc_size_arg(name: &str, v: &Value) -> InterpResult<u128> {
    match v {
        Value::Uint(n) => Ok(*n),
        Value::Int(n) if *n >= 0 => Ok(*n as u128),
        other => Err(InterpError::type_mismatch(
            name,
            "allocation size",
            other.kind_name(),
        )),
    }
}

pub(super) fn int_tag(tag: &str) -> Option<(u32, bool)> {
    Some(match tag {
        "i8" => (8, true),
        "i16" => (16, true),
        "i32" => (32, true),
        "i64" => (64, true),
        "i128" => (128, true),
        "isize" => (64, true),
        "u8" => (8, false),
        "u16" => (16, false),
        "u32" => (32, false),
        "u64" => (64, false),
        "u128" => (128, false),
        "usize" => (64, false),
        _ => return None,
    })
}

/// The element type tag of a vector type tag: the element of
/// an array tag `[E; N]`, or the first top-level generic
/// argument of `...<E, ...>`.
pub(super) fn vector_elem_tag(tag: &str) -> Option<&str> {
    let tag = tag.trim();
    if let Some(inner) = tag.strip_prefix('[') {
        return inner.split(';').next().map(str::trim);
    }
    let open = tag.find('<')?;
    let close = tag.rfind('>')?;
    if close <= open {
        return None;
    }
    let parts = split_top_level(&tag[open + 1..close]);
    parts.into_iter().next().map(str::trim)
}

/// A mask of the low `bits` bits.
pub(super) fn bit_mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// Sign-extend the low `bits` bits of `pattern` to an `i128`.
pub(super) fn sign_extend(pattern: u128, bits: u32) -> i128 {
    if bits >= 128 {
        pattern as i128
    } else {
        let shift = 128 - bits;
        ((pattern << shift) as i128) >> shift
    }
}

/// The field values of an aggregate value: a struct rides
/// as a `Value::Enum` (variant 0), a plain aggregate as a
/// `Value::Tuple`.
/// The pointee type of a rendered smart-pointer type tag
/// (`std::sync::Arc<T>` / `std::rc::Rc<T>`, in either crate
/// spelling, with or without the allocator argument): the
/// first top-level generic argument, or `None` when the tag
/// names neither pointer.
pub(super) fn smart_ptr_pointee(source: &str, prefixes: &[&str]) -> Option<String> {
    let mut inner = None;
    for prefix in prefixes {
        if let Some(rest) = source.strip_prefix(prefix) {
            inner = Some(rest.strip_suffix('>')?);
            break;
        }
    }
    let inner = inner?;
    let mut depth = 0i32;
    for (i, c) in inner.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => return Some(inner[..i].trim().to_string()),
            _ => {}
        }
    }
    Some(inner.trim().to_string())
}

pub(super) fn aggregate_fields(value: &Value) -> Option<&[Value]> {
    match value {
        Value::Enum { fields, .. } => Some(fields),
        Value::Tuple(items) => Some(items),
        _ => None,
    }
}

/// Rebuild an aggregate with the shape of `template` and new
/// field values.
pub(super) fn aggregate_like(template: &Value, fields: Vec<Value>) -> Value {
    match template {
        Value::Enum { variant, .. } => Value::Enum {
            variant: *variant,
            fields,
        },
        _ => Value::Tuple(fields),
    }
}

pub(super) fn as_f64(value: &Value, name: &str) -> InterpResult<f64> {
    match value {
        Value::Float(f) => Ok(*f),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "float",
            other.kind_name(),
        )),
    }
}

/// The 64-bit pattern a `LARGE_INTEGER` union value holds.
/// A union value is an `Enum` holding one live field (the
/// last written; the other slots are `Uninit`): here either
/// the `QuadPart` scalar or the two-`u32` struct view, whose
/// parts combine low-then-high. A bare scalar (the shape a
/// constant aggregate can arrive in) reads as itself.
pub(super) fn union_u64_bits(value: &Value) -> Option<u64> {
    match value {
        Value::Uint(v) => Some(*v as u64),
        Value::Int(v) => Some(*v as u64),
        Value::Enum { fields, .. } => {
            let live: Vec<&Value> = fields
                .iter()
                .filter(|f| **f != Value::Uninit)
                .collect();
            match live.as_slice() {
                [only] => match only {
                    Value::Enum { fields: parts, .. } if parts.len() == 2 => {
                        let lo = match &parts[0] {
                            Value::Uint(v) => *v as u64,
                            _ => return None,
                        };
                        let hi = match &parts[1] {
                            Value::Uint(v) => *v as u64,
                            _ => return None,
                        };
                        Some(lo | (hi << 32))
                    }
                    other => union_u64_bits(other),
                },
                _ => None,
            }
        }
        Value::UnionBytes { bytes } => {
            if bytes.iter().all(|b| b.is_none()) {
                return None;
            }
            let mut bits: u64 = 0;
            for (i, b) in bytes.iter().enumerate().take(8) {
                if let Some(b) = b {
                    bits |= u64::from(*b) << (8 * i);
                }
            }
            Some(bits)
        }
        Value::Tuple(items) if items.len() == 1 => union_u64_bits(&items[0]),
        _ => None,
    }
}

/// A union value with `template`'s shape holding `bits` as
/// its 64-bit pattern: the live slot of the template (the
/// scalar, or the struct view as its two 32-bit parts) is
/// rewritten, every other slot stays `Uninit`, so the guest's
/// union field reads reinterpret the new pattern exactly as
/// they did the old one.
pub(super) fn union_with_u64_bits(template: &Value, bits: u64) -> Value {
    match template {
        Value::Enum { variant, fields } => {
            let live_idx = fields.iter().position(|f| *f != Value::Uninit);
            match live_idx {
                Some(idx) => {
                    let mut out: Vec<Value> =
                        fields.iter().map(|_| Value::Uninit).collect();
                    out[idx] = match &fields[idx] {
                        Value::Enum { variant: v2, fields: parts } if parts.len() == 2 => {
                            Value::Enum {
                                variant: *v2,
                                fields: vec![
                                    Value::Uint((bits & 0xFFFF_FFFF) as u128),
                                    Value::Uint((bits >> 32) as u128),
                                ],
                            }
                        }
                        _ => Value::Uint(bits as u128),
                    };
                    Value::Enum {
                        variant: *variant,
                        fields: out,
                    }
                }
                None => Value::Uint(bits as u128),
            }
        }
        Value::UnionBytes { bytes } => {
            let mut out = bytes.clone();
            if out.len() < 8 {
                out.resize(8, None);
            }
            for (i, b) in bits.to_le_bytes().iter().enumerate() {
                out[i] = Some(*b);
            }
            Value::UnionBytes { bytes: out }
        }
        Value::Tuple(items) if items.len() == 1 => {
            Value::Tuple(vec![union_with_u64_bits(&items[0], bits)])
        }
        _ => Value::Uint(bits as u128),
    }
}

pub(super) fn as_i128(value: &Value, name: &str) -> InterpResult<i128> {
    match value {
        Value::Int(v) => Ok(*v),
        Value::Uint(v) => Ok(*v as i128),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "integer",
            other.kind_name(),
        )),
    }
}

pub(super) fn as_u128(value: &Value, name: &str) -> InterpResult<u128> {
    match value {
        Value::Uint(v) => Ok(*v),
        Value::Int(v) => Ok(*v as u128),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "integer",
            other.kind_name(),
        )),
    }
}

/// The all-zero-bits value of the type `tag` names, for
/// `mem::zeroed`: scalars, and tuples/arrays built from scalars,
/// within the materialisation budget (see [`MAX_FILL_LEAVES`]):
/// a large array of scalars is a big tree of zero leaves, which
/// the budget admits up to millions of elements. Anything else
/// (an ADT whose zero bit pattern need not be a valid value, a
/// reference, ...) stays unsupported.
pub(super) fn zeroed_value_for_tag(tag: &str) -> Option<Value> {
    if tag_leaf_count(tag)? > MAX_FILL_LEAVES {
        return None;
    }
    let tag = tag.trim();
    Some(match tag {
        "bool" => Value::Bool(false),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => Value::Int(0),
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "char" => Value::Uint(0),
        "f32" | "f64" => Value::Float(0.0),
        "f128" => Value::Float128(0),
        "()" => Value::Unit,
        _ if tag.starts_with('(') && tag.ends_with(')') => {
            let inner = &tag[1..tag.len() - 1];
            let mut vals = Vec::new();
            if !inner.trim().is_empty() {
                for part in split_top_level(inner) {
                    vals.push(zeroed_value_for_tag(part)?);
                }
            }
            Value::Tuple(vals)
        }
        _ if tag.starts_with('[') && tag.ends_with(']') => {
            let inner = &tag[1..tag.len() - 1];
            let (elem, n) = inner.rsplit_once(';')?;
            let n: usize = n.trim().parse().ok()?;
            let elem = zeroed_value_for_tag(elem)?;
            Value::Tuple(vec![elem; n])
        }
        _ => return None,
    })
}


/// The first generic argument of a rendered generic type tag
/// (`std::cell::RefCell<std::vec::Vec<u32>>` -> the `Vec` tag), or
/// `None` when the tag does not start with `prefix`.
/// The `Ty` a rendered type tag denotes, for the forms that
/// carry one: the inverse of [`ty_tag`]'s rendering, following
/// the rules conversion itself applies when it builds a `Ty`
/// from a rustc type (an ADT instance is its definition's name
/// with the generic arguments dropped, a `char` is a `u32`, a
/// raw pointer shares the reference representation). Forms
/// whose `Ty` carries facts the rendering does not -- a `dyn`
/// type, a closure or coroutine type, a generic parameter, an
/// unevaluated array length -- answer `None`, and the caller
/// keeps the behaviour it has without a type.
pub(crate) fn ty_from_tag(tag: &str) -> Option<Ty> {
    let tag = tag.trim();
    Some(match tag {
        "bool" => Ty::Bool,
        "char" => Ty::Uint(UintTy::U32),
        "i8" => Ty::Int(IntTy::I8),
        "i16" => Ty::Int(IntTy::I16),
        "i32" => Ty::Int(IntTy::I32),
        "i64" => Ty::Int(IntTy::I64),
        "i128" => Ty::Int(IntTy::I128),
        "isize" => Ty::Int(IntTy::Isize),
        "u8" => Ty::Uint(UintTy::U8),
        "u16" => Ty::Uint(UintTy::U16),
        "u32" => Ty::Uint(UintTy::U32),
        "u64" => Ty::Uint(UintTy::U64),
        "u128" => Ty::Uint(UintTy::U128),
        "usize" => Ty::Uint(UintTy::Usize),
        "f16" => Ty::Float(FloatTy::F16),
        "f32" => Ty::Float(FloatTy::F32),
        "f64" => Ty::Float(FloatTy::F64),
        "f128" => Ty::Float(FloatTy::F128),
        "str" => Ty::Str,
        "!" => Ty::Never,
        "()" => Ty::Unit,
        _ if tag.starts_with('[') && tag.ends_with(']') => {
            let inner = &tag[1..tag.len() - 1];
            match inner.rsplit_once(';') {
                Some((elem, n)) => {
                    let n: u64 = n.trim().parse().ok()?;
                    Ty::Array(Box::new(ty_from_tag(elem.trim())?), n)
                }
                None => Ty::Slice(Box::new(ty_from_tag(inner.trim())?)),
            }
        }
        _ if tag.starts_with('(') && tag.ends_with(')') => {
            let inner = &tag[1..tag.len() - 1];
            if inner.is_empty() {
                Ty::Unit
            } else {
                let mut tys = Vec::new();
                for part in split_top_level(inner) {
                    let part = part.trim();
                    if part.is_empty() {
                        // The one-tuple's trailing comma.
                        continue;
                    }
                    tys.push(ty_from_tag(part)?);
                }
                Ty::Tuple(tys)
            }
        }
        _ if tag.starts_with('&') => {
            let inner = tag[1..].trim_start();
            let inner = inner.strip_prefix("mut ").unwrap_or(inner);
            Ty::Ref(Box::new(ty_from_tag(inner)?))
        }
        _ if tag.starts_with("*const ") || tag.starts_with("*mut ") => {
            let inner = tag.split_once(' ')?.1.trim();
            Ty::Ref(Box::new(ty_from_tag(inner)?))
        }
        _ if tag.starts_with("dyn ") => return None,
        _ => {
            // An ADT instance: the definition's path, with any
            // generic arguments dropped, exactly as conversion
            // builds the type. ADT tags are crate-qualified
            // paths; a bare single-segment name is a generic
            // parameter (or another form with no `Ty` here),
            // and a closure or coroutine definition name
            // carries its `{closure}` / `{coroutine}` marker.
            let base = match tag.find('<') {
                Some(at) => &tag[..at],
                None => tag,
            };
            if !base.contains("::") || base.contains('{') {
                return None;
            }
            Ty::Adt(base.to_string())
        }
    })
}
