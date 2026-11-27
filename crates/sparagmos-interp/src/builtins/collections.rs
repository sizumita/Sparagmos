//! Vec/VecDeque/Box storage construction for the collection builtins.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    box_alloc_pointer,
    Frame, Interpreter, Resolved,
};
use crate::ir::Ty;
use crate::value::{RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// Take the pointee out of a `Box` value a conversion
    /// consumes (`From<Box<T>>` and its kin): the box aggregate
    /// `new_box` builds, over the allocation holding the
    /// pointee. The box's allocation is released — the box is
    /// gone once its pointee has moved out — and the pointee is
    /// returned. `None` when the value is not a box over a live
    /// allocation, or the box pointer carries a projection.
    pub(super) fn unbox_take(&self, value: &Value) -> Option<Value> {
        let (id, projection) = box_alloc_pointer(value)?;
        if !projection.is_empty() {
            return None;
        }
        let pointee = self.alloc_value(id)?;
        self.free_alloc(id);
        Some(pointee)
    }

    /// The `Box<[T]>` a sequence value's elements move into: the
    /// element tuple in a fresh allocation, with the slice
    /// pointee type recorded when known. This is the shape
    /// `Vec::into_boxed_slice` builds and every slice consumer
    /// of a boxed slice (indexing, iteration, `Debug`, comparison)
    /// reads, so all construction paths that produce a boxed
    /// slice go through here. A sparse vector contributes its
    /// declared elements: the stored prefix, plus the
    /// uninitialised tail when the length runs past it
    /// (materialised up to the same limit `Vec::set_len` uses).
    pub(super) fn boxed_slice_value(
        &self,
        value: Value,
        pointee_ty: Option<Ty>,
        context: &'static str,
    ) -> InterpResult<Value> {
        let boxed = match value {
            Value::Enum { fields, .. } => self.new_box(Value::Tuple(fields)),
            Value::SparseVec { mut elems, len, .. } => {
                let len = usize::try_from(len).map_err(|_| {
                    unsupported(format!(
                        "{context} on a vector whose elements are not materialised"
                    ))
                })?;
                if len > elems.len() && len > MAX_VEC_SET_LEN_ELEMENTS {
                    return Err(unsupported(format!(
                        "{context} on a vector whose elements are not materialised"
                    )));
                }
                elems.truncate(len);
                if elems.len() < len {
                    elems.resize(len, Value::Uninit);
                }
                self.new_box(Value::Tuple(elems))
            }
            Value::Tuple(items) => self.new_box(Value::Tuple(items)),
            other => {
                return Err(InterpError::type_mismatch(
                    context,
                    "vector value",
                    other.kind_name(),
                ))
            }
        };
        if let Some(ty) = pointee_ty {
            self.record_box_ty(&boxed, ty);
        }
        Ok(boxed)
    }
}

impl<'a> Interpreter<'a> {
    /// Semantic (derived `PartialEq`) equality of two values:
    /// references are followed to the data they denote and aggregates
    /// compare element-wise. This is the equality of the *data*, not
    /// of the representation — exactly what the derived `PartialEq`
    /// impls of the modelled types compute.
    /// The logical element sequence of a `VecDeque` value (the
    /// real aggregate: wrapped head index, length, and the ring
    /// buffer), in iteration order, when the value has exactly
    /// that shape; `None` for any other value, and for a deque
    /// whose buffer slots cannot be read back (a physical slot
    /// of the logical range was never written -- not a state a
    /// well-formed deque reaches). The buffer's physical layout
    /// is not part of the sequence.
    pub(crate) fn vec_deque_elements(
        &self,
        value: &Value,
    ) -> InterpResult<Option<Vec<Value>>> {
        let Value::Enum { fields, .. } = value else {
            return Ok(None);
        };
        if fields.len() != 3 {
            return Ok(None);
        }
        let Value::Enum {
            fields: head_fields, ..
        } = &fields[0]
        else {
            return Ok(None);
        };
        let [Value::Uint(head)] = head_fields.as_slice() else {
            return Ok(None);
        };
        let Value::Uint(len) = fields[1] else {
            return Ok(None);
        };
        let Value::Enum {
            fields: buf_fields, ..
        } = &fields[2]
        else {
            return Ok(None);
        };
        if buf_fields.len() != 2 {
            return Ok(None);
        }
        let Value::Enum {
            fields: inner_fields, ..
        } = &buf_fields[0]
        else {
            return Ok(None);
        };
        if inner_fields.len() != 3 {
            return Ok(None);
        }
        let Value::Enum {
            fields: ptr_fields, ..
        } = &inner_fields[0]
        else {
            return Ok(None);
        };
        let [ptr] = ptr_fields.as_slice() else {
            return Ok(None);
        };
        let Value::Enum {
            fields: cap_fields, ..
        } = &inner_fields[1]
        else {
            return Ok(None);
        };
        let [Value::Uint(cap)] = cap_fields.as_slice() else {
            return Ok(None);
        };
        let (head, len, cap) = (*head as usize, len as usize, *cap as usize);
        if len == 0 {
            return Ok(Some(Vec::new()));
        }
        if cap == 0 {
            return Ok(None);
        }
        // The buffer's element block: the allocation the
        // `RawVec` pointer denotes, whose stored value is the
        // element tuple.
        let Value::Ref(RefTarget::Alloc { id, .. }) = ptr else {
            return Ok(None);
        };
        let Ok(buffer) = self.alloc_base(*id) else {
            return Ok(None);
        };
        let Value::Tuple(slots) = buffer else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let physical = (head + i) % cap;
            let Some(elem) = slots.get(physical) else {
                return Ok(None);
            };
            if *elem == Value::Uninit {
                return Ok(None);
            }
            out.push(elem.clone());
        }
        Ok(Some(out))
    }
}

impl<'a> Interpreter<'a> {
    /// The byte string a path argument denotes, and whether it
    /// arrived as text: a `Path` / `OsStr` value in this model
    /// is the byte string itself under its wrapper chain (see
    /// the field projection pun in `project_value`), held as a
    /// `Str` for text data or a `Bytes` for arbitrary bytes, or
    /// as a fat pointer over the bytes.
    /// Assemble a `VecDeque` value from a `From` / `Into`
    /// source: a `Vec` (modelled, or in its real aggregate
    /// form), an array, or a slice. The deque's value is the
    /// real aggregate the converted std bodies build and
    /// consume (see `vec_deque_value`): a source `Vec` in its
    /// real aggregate form hands its buffer over unchanged, as
    /// the std conversion does; every other source's elements
    /// are copied into a fresh element block at exact-fit
    /// capacity, with the head at index zero.
    pub(super) fn vec_deque_from_value(
        &self,
        frames: &[Frame],
        value: Value,
    ) -> InterpResult<Value> {
        let resolved = self.resolve_value(frames, &value)?;
        if let Some((buf, len)) = vec_real_aggregate_parts(&resolved) {
            return Ok(vec_deque_value(buf, len));
        }
        let elements = match resolved {
            Value::Enum { fields, .. } => fields,
            Value::Tuple(items) => items,
            v @ Value::SparseVec { .. } => value_elements(&v)?,
            Value::FatRef { base, .. } => match *base {
                Value::Tuple(items) => items,
                other => {
                    return Err(InterpError::type_mismatch(
                        "From::from to VecDeque",
                        "slice value over an element tuple",
                        other.kind_name(),
                    ));
                }
            },
            other => {
                return Err(InterpError::type_mismatch(
                    "From::from to VecDeque",
                    "vector, array or slice value",
                    other.kind_name(),
                ));
            }
        };
        let len = elements.len();
        let id = self.alloc_insert(Value::Tuple(elements));
        self.mark_elem_block(id);
        let buf = raw_vec_value(
            Value::Ref(RefTarget::Alloc {
                id,
                projection: Vec::new(),
            }),
            len,
        );
        Ok(vec_deque_value(buf, len))
    }

    /// `BinaryHeap<T>: From<Vec<T>>` (the `FromFrom`
    /// dispatch): std moves the vector's buffer into the
    /// heap's single `data` field and rebuilds the heap
    /// order in place (`BinaryHeap::rebuild`). The model
    /// extracts the source's elements, rebuilds the heap
    /// order over them (Floyd's sift-down pass: any valid
    /// heap arrangement pops in the same descending order
    /// std's arrangement does), and builds the heap's real
    /// aggregate over a fresh element block at exact-fit
    /// capacity, the construction `vec_deque_from_value`
    /// uses for the deque.
    pub(super) fn binary_heap_from_vec(
        &self,
        frames: &[Frame],
        value: Value,
    ) -> InterpResult<Value> {
        let resolved = self.resolve_value(frames, &value)?;
        let resolved = match self.vec_from_real_form(frames, &resolved) {
            Some(r) => r?,
            None => resolved,
        };
        let mut elements = match resolved {
            Value::Enum { fields, .. } => fields,
            Value::Tuple(items) => items,
            v @ Value::SparseVec { .. } => value_elements(&v)?,
            Value::FatRef { base, .. } => match *base {
                Value::Tuple(items) => items,
                other => {
                    return Err(InterpError::type_mismatch(
                        "From::from to BinaryHeap",
                        "slice value over an element tuple",
                        other.kind_name(),
                    ));
                }
            },
            other => {
                return Err(InterpError::type_mismatch(
                    "From::from to BinaryHeap",
                    "vector value",
                    other.kind_name(),
                ));
            }
        };
        let len = elements.len();
        for start in (0..len / 2).rev() {
            let mut i = start;
            loop {
                let left = 2 * i + 1;
                let right = left + 1;
                let mut largest = i;
                if left < len {
                    match self.values_cmp(frames, &elements[left], &elements[largest], None)? {
                        Some(std::cmp::Ordering::Greater) => largest = left,
                        Some(_) => {}
                        None => {
                            return Err(unsupported(
                                "From::from to BinaryHeap: element comparison is unordered"
                                    .to_string(),
                            ));
                        }
                    }
                }
                if right < len {
                    match self.values_cmp(frames, &elements[right], &elements[largest], None)? {
                        Some(std::cmp::Ordering::Greater) => largest = right,
                        Some(_) => {}
                        None => {
                            return Err(unsupported(
                                "From::from to BinaryHeap: element comparison is unordered"
                                    .to_string(),
                            ));
                        }
                    }
                }
                if largest == i {
                    break;
                }
                elements.swap(i, largest);
                i = largest;
            }
        }
        let id = self.alloc_insert(Value::Tuple(elements));
        self.mark_elem_block(id);
        let buf = raw_vec_value(
            Value::Ref(RefTarget::Alloc {
                id,
                projection: Vec::new(),
            }),
            len,
        );
        let vec_value = Value::Enum {
            variant: 0,
            fields: vec![buf, Value::Uint(len as u128)],
        };
        Ok(Value::Enum {
            variant: 0,
            fields: vec![vec_value],
        })
    }
}

/// The `ErrorKind::InvalidInput` discriminant.
/// The largest element count `Vec::set_len` will materialise
/// (see the `VecSetLen` builtin): the element list is stored
/// value by value, so the count is bounded well clear of the
/// memory a pathological length would need.
pub(crate) const MAX_VEC_SET_LEN_ELEMENTS: usize = 1 << 22;

/// A vector value holding `elems` at declared length `len`:
/// the plain element aggregate when the declared length is
/// exactly the stored list — a growth into the materialisation
/// limit pads with `uninit_elem`, the element type's
/// uninitialised form (see `uninit_elem_value_for_tag`), as
/// `Vec::set_len` does — and the sparse form (see
/// `Value::SparseVec`) when
/// the two differ: past the materialisation limit the length
/// runs ahead of the stored elements, and on a shrink the
/// stored elements run ahead of the length. A shrink keeps
/// the tail in storage: those slots are the vector's spare
/// capacity — still reachable through the buffer pointer (a
/// `pop` reads the element it just un-declared), counted by
/// the capacity the model reports, and overwritten by the
/// next append before any new storage is needed.
///
/// `prev_cap` is the storage extent the value being rebuilt
/// accounted (see `vec_stored_len`): the rebuilt value keeps
/// accounting at least that much storage — a length change
/// alone never shrinks the allocation a `with_capacity`
/// established — so the result takes the sparse form
/// whenever the accounted extent runs past the stored list,
/// even when length and stored list meet again.
pub(crate) fn vec_with_len(
    mut elems: Vec<Value>,
    len: usize,
    uninit_elem: &Value,
    prev_cap: u128,
) -> Value {
    let cap = prev_cap.max(len as u128).max(elems.len() as u128);
    if len == elems.len() {
        if cap == len as u128 {
            Value::Enum {
                variant: 0,
                fields: elems,
            }
        } else {
            Value::SparseVec {
                elems,
                len: len as u128,
                cap,
            }
        }
    } else if len < elems.len() {
        Value::SparseVec {
            elems,
            len: len as u128,
            cap,
        }
    } else if len <= MAX_VEC_SET_LEN_ELEMENTS {
        elems.resize(len, uninit_elem.clone());
        if cap == len as u128 {
            Value::Enum {
                variant: 0,
                fields: elems,
            }
        } else {
            Value::SparseVec {
                elems,
                len: len as u128,
                cap,
            }
        }
    } else {
        Value::SparseVec {
            elems,
            len: len as u128,
            cap,
        }
    }
}

/// The capacity a vector value's storage represents: the
/// number of element slots the value accounts for — the stored
/// list's length, or the declared length when it runs past the
/// stored list (the unmaterialised tail is allocation the
/// model accounts without holding element by element), or the
/// extent a `with_capacity` requested past the materialisation
/// limit, which the sparse form records as its `cap`. Every
/// capacity question in the model gives this answer (the
/// `Vec::capacity` builtin and the buffer chain's capacity
/// slot in place resolution alike): the element storage is
/// exactly the allocation the model holds.
pub(crate) fn vec_stored_len(value: &Value) -> Option<u128> {
    match value {
        Value::Enum { fields, .. } => Some(fields.len() as u128),
        Value::SparseVec { elems, cap, .. } => Some((*cap).max(elems.len() as u128)),
        _ => None,
    }
}

/// The minimum non-zero capacity of a `RawVec` for an element
/// of `elem_size` bytes: std's `MIN_NON_ZERO_CAP` (8 for
/// byte-sized elements, 4 up to a 1024-byte element, 1 above).
fn rawvec_min_non_zero_cap(elem_size: usize) -> usize {
    if elem_size == 1 {
        8
    } else if elem_size <= 1024 {
        4
    } else {
        1
    }
}

/// The capacity RawVec's amortized growth reaches when
/// `required` elements must fit over a current capacity of
/// `cap`: twice the current capacity or the requirement,
/// whichever is larger, floored at the minimum non-zero
/// capacity -- the floor applies at every growth, not only the
/// first allocation (growing a capacity-1 vector of 4-byte
/// elements reaches 4, not 2).
fn rawvec_amortized_capacity(cap: usize, required: usize, elem_size: usize) -> usize {
    cap.saturating_mul(2)
        .max(required)
        .max(rawvec_min_non_zero_cap(elem_size))
}

/// Grow the element storage of the vector value in `slot` to
/// `target` slots by padding with `uninit_elem`, the element
/// type's uninitialised form: the storage change RawVec's
/// reallocation performs in this model, where the capacity a
/// vector reports is the extent of the storage it holds (see
/// `vec_stored_len`). The declared length is untouched; the
/// padded slots are the spare capacity the next append
/// overwrites. A vector whose declared length runs past its
/// stored elements (the unmaterialised form) has no storage to
/// grow and is left as it stands.
fn vec_grow_storage(slot: &mut Value, target: usize, uninit_elem: &Value) {
    match slot {
        Value::Enum { fields, .. } => {
            if target > fields.len() {
                let len = fields.len();
                let elems = {
                    let mut padded = std::mem::take(fields);
                    padded.resize(target, uninit_elem.clone());
                    padded
                };
                *slot = Value::SparseVec {
                    elems,
                    len: len as u128,
                    cap: target as u128,
                };
            }
        }
        Value::SparseVec { elems, len, cap } => {
            if *len <= elems.len() as u128 && target > elems.len() {
                elems.resize(target, uninit_elem.clone());
                *cap = (*cap).max(target as u128);
            }
        }
        _ => {}
    }
}

/// Move the capacity slot of a real-aggregate vector value
/// (see `vec_real_aggregate_parts`) to `cap`: the navigation
/// mirrors `Interpreter::real_vec_cap`, writing the count back
/// in the shape the slot already holds (a bare count or the
/// one-field `UsizeNoHighBit` wrapper around one).
fn set_real_vec_cap(slot: &mut Value, cap: usize) {
    let Value::Enum { fields, .. } = slot else {
        return;
    };
    let Some(buf) = fields.first_mut() else {
        return;
    };
    let Value::Enum {
        fields: buf_fields, ..
    } = buf
    else {
        return;
    };
    let Some(inner) = buf_fields.first_mut() else {
        return;
    };
    let Value::Enum {
        fields: inner_fields,
        ..
    } = inner
    else {
        return;
    };
    let Some(cap_slot) = inner_fields.get_mut(1) else {
        return;
    };
    match cap_slot {
        Value::Uint(n) => *n = cap as u128,
        Value::Int(n) => *n = cap as i128,
        Value::Enum { fields, .. } if fields.len() == 1 => match &mut fields[0] {
            Value::Uint(n) => *n = cap as u128,
            Value::Int(n) => *n = cap as i128,
            _ => {}
        },
        _ => {}
    }
}

/// Materialise the element storage of a real-aggregate vector
/// value whose buffer pointer denotes no storage -- a null
/// pointer, or an address no allocation owns (the dangling
/// pointer std serves a fresh vector with): a fresh element
/// block of `count` uninitialised elements, with the buffer
/// pointer field rewritten to denote it, as a reallocation
/// installs the new allocation's address in the field. A
/// pointer that already denotes storage (an allocation, a
/// place, a static, a registered address) is left exactly as
/// it stands: the storage behind it grows on the writes that
/// use it.
fn ensure_real_vec_buffer_storage(
    interp: &Interpreter,
    slot: &mut Value,
    count: usize,
    uninit_elem: &Value,
) {
    let Some(ptr) = Interpreter::real_vec_buffer_ptr(slot) else {
        return;
    };
    let storage_less = match &ptr {
        Value::Ref(RefTarget::Null) => true,
        Value::Ref(RefTarget::Address(addr)) => {
            interp.address_owner(*addr).is_none()
        }
        _ => false,
    };
    if !storage_less {
        return;
    }
    let id = interp.alloc_insert(Value::Tuple(vec![
        uninit_elem.clone();
        count
    ]));
    interp.mark_elem_block(id);
    set_real_vec_buffer_ptr(
        slot,
        Value::Ref(RefTarget::Alloc {
            id,
            projection: Vec::new(),
        }),
    );
}

/// Rewrite the buffer pointer slot of a real-aggregate vector
/// value (see `vec_real_aggregate_parts`) to `ptr`: the
/// navigation mirrors `Interpreter::real_vec_buffer_ptr`,
/// writing through the one-field wrapper when the slot holds
/// one. A value whose chain does not have the recorded shape
/// is left as it stands.
fn set_real_vec_buffer_ptr(slot: &mut Value, ptr: Value) {
    let Value::Enum { fields, .. } = slot else {
        return;
    };
    let Some(buf) = fields.first_mut() else {
        return;
    };
    let Value::Enum {
        fields: buf_fields, ..
    } = buf
    else {
        return;
    };
    let Some(inner) = buf_fields.first_mut() else {
        return;
    };
    let Value::Enum {
        fields: inner_fields,
        ..
    } = inner
    else {
        return;
    };
    let Some(first) = inner_fields.first_mut() else {
        return;
    };
    match first {
        Value::Enum { fields, .. } if fields.len() == 1 => {
            if matches!(fields[0], Value::Ref(_) | Value::FatRef { .. }) {
                fields[0] = ptr;
            }
        }
        v @ (Value::Ref(_) | Value::FatRef { .. }) => *v = ptr,
        _ => {}
    }
}

/// Ensure the vector value in `slot` holds storage for
/// `additional` elements past its declared length, growing the
/// storage the way RawVec grows the allocation: amortized
/// (`reserve`, the appends) or to the exact requirement (the
/// `*_exact` spellings). A requirement the storage already
/// meets changes nothing, and a target past the materialisation
/// limit is not accounted -- the same stance `Vec::with_capacity`
/// takes toward a capacity past the limit: the storage the
/// model can hold is the allocation it reports.
pub(super) fn vec_ensure_capacity(
    interp: &Interpreter,
    slot: &mut Value,
    additional: usize,
    exact: bool,
    uninit_elem: &Value,
    elem_size: usize,
) {
    // A vector in its real aggregate form (the `Vec { buf, len }`
    // aggregate a converted body builds field by field, e.g. the
    // buffer `join_generic_copy` fills through raw pointers) is
    // not the element list its outer `Enum` shape suggests: the
    // two fields are the `RawVec` and the length, and the
    // capacity lives in the buffer chain's own slot (see
    // `vec_real_aggregate_parts`). Growth there is the
    // reallocation's bookkeeping: the capacity slot moves to the
    // reached capacity, and the element storage behind the
    // buffer pointer grows on the writes that use it, as
    // allocation storage does. A buffer pointer that denotes no
    // storage at all is the exception: it has nothing for those
    // writes to grow, so the storage is materialised and the
    // field repointed (see `ensure_real_vec_buffer_storage`).
    // A capacity the chain does not expose leaves the value as
    // it stands.
    if let Some((_, agg_len)) = vec_real_aggregate_parts(slot) {
        let Some(cap) = Interpreter::real_vec_cap(slot) else {
            return;
        };
        let cap = usize::try_from(cap).unwrap_or(usize::MAX);
        let required = agg_len.saturating_add(additional);
        let reached = if required <= cap {
            cap
        } else {
            let target = if exact {
                required
            } else {
                rawvec_amortized_capacity(cap, required, elem_size)
            };
            if target > MAX_VEC_SET_LEN_ELEMENTS {
                return;
            }
            set_real_vec_cap(slot, target);
            target
        };
        if reached > 0 {
            ensure_real_vec_buffer_storage(interp, slot, reached, uninit_elem);
        }
        return;
    }
    let (len, cap) = match &*slot {
        Value::Enum { fields, .. } => (fields.len(), fields.len()),
        Value::SparseVec { elems, len, cap } => {
            let len = usize::try_from(*len).unwrap_or(usize::MAX);
            let cap = usize::try_from(*cap).unwrap_or(usize::MAX);
            (len, cap.max(elems.len()))
        }
        _ => return,
    };
    let required = len.saturating_add(additional);
    if required <= cap {
        return;
    }
    let target = if exact {
        required
    } else {
        rawvec_amortized_capacity(cap, required, elem_size)
    };
    if target > MAX_VEC_SET_LEN_ELEMENTS {
        return;
    }
    vec_grow_storage(slot, target, uninit_elem);
}

impl<'a> Interpreter<'a> {
    /// Append `items` to a vector value in its real aggregate
    /// form (see `vec_real_aggregate_parts`): the elements are
    /// written into the storage the buffer pointer denotes, at
    /// the declared length, and the length field advances past
    /// them — the append the element form performs on its own
    /// fields, carried out on the form a converted body builds
    /// field by field. The storage must already hold the new
    /// elements (see `vec_ensure_capacity`, whose real-form
    /// branch also materialises a storage-less buffer).
    /// Returns `false`, leaving the value untouched, when it
    /// is not in the real form.
    pub(super) fn vec_append_real_form(
        &self,
        frames: &mut Vec<Frame>,
        slot: &mut Value,
        items: &[Value],
        op: &str,
    ) -> InterpResult<bool> {
        let Some((_, len)) = vec_real_aggregate_parts(slot) else {
            return Ok(false);
        };
        if !items.is_empty() {
            let ptr = Self::real_vec_buffer_ptr(slot).ok_or_else(|| {
                unsupported(format!(
                    "{op} on a vector whose buffer pointer is not a reference value"
                ))
            })?;
            let seq = self.deref_value(frames, &ptr)?;
            for (offset, item) in items.iter().enumerate() {
                self.write_seq_at(frames, &seq, len + offset, item.clone())?;
            }
        }
        if let Value::Enum { fields, .. } = slot {
            if let Some(len_slot) = fields.get_mut(1) {
                *len_slot = Value::Uint((len + items.len()) as u128);
            }
        }
        Ok(true)
    }

    /// Append `items` to the vector the target denotes,
    /// growing its storage first (see `vec_ensure_capacity`):
    /// a vector in its real aggregate form takes the buffer
    /// write path (see `vec_append_real_form`); every other
    /// shape takes the in-place element append
    /// (see `vec_append_in_place`).
    pub(super) fn vec_append_resolved(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        items: &[Value],
        op: &str,
        uninit_elem: &Value,
        elem_size: usize,
    ) -> InterpResult<()> {
        let current = self.read_resolved(frames, target)?;
        if vec_real_aggregate_parts(&current).is_some() {
            let mut value = current;
            vec_ensure_capacity(
                self,
                &mut value,
                items.len(),
                false,
                uninit_elem,
                elem_size,
            );
            self.vec_append_real_form(frames, &mut value, items, op)?;
            self.write_resolved(frames, target, value)?;
            return Ok(());
        }
        self.modify_resolved(frames, target, |slot| {
            vec_ensure_capacity(self, slot, items.len(), false, uninit_elem, elem_size);
            vec_append_in_place(slot, items, op)
        })?;
        Ok(())
    }
}

/// Append `items` to the vector value in `slot`, in place:
/// slots the storage already holds past the length (the spare
/// capacity a length shrink left behind) are overwritten in
/// order and the rest extend the storage, so length and
/// storage grow together and the value takes the plain
/// aggregate form when they meet again — unless the accounted
/// capacity still runs past the stored list, which only the
/// sparse form records. Appends past the accounted capacity
/// extend the account with the storage, as the plain form's
/// appends do. A vector whose
/// declared length runs past its stored elements cannot take
/// appends — the new elements would land past the stored
/// region — and stays the explicit error it was.
pub(super) fn vec_append_in_place(
    slot: &mut Value,
    items: &[Value],
    op: &str,
) -> InterpResult<()> {
    match slot {
        Value::Enum { fields, .. } => {
            fields.extend(items.iter().cloned());
            Ok(())
        }
        Value::SparseVec { .. } => {
            let value = std::mem::replace(slot, Value::Uninit);
            let Value::SparseVec {
                mut elems,
                len,
                cap,
            } = value
            else {
                unreachable!("matched sparse vector above")
            };
            let cur_len = usize::try_from(len).unwrap_or(usize::MAX);
            if cur_len > elems.len() {
                *slot = Value::SparseVec { elems, len, cap };
                return Err(unsupported(format!(
                    "{op} on a vector whose elements are not materialised"
                )));
            }
            let mut l = cur_len;
            for item in items {
                if l < elems.len() {
                    elems[l] = item.clone();
                } else {
                    elems.push(item.clone());
                }
                l += 1;
            }
            let cap = cap.max(l as u128).max(elems.len() as u128);
            *slot = if l == elems.len() && cap == l as u128 {
                Value::Enum {
                    variant: 0,
                    fields: elems,
                }
            } else {
                Value::SparseVec {
                    elems,
                    len: l as u128,
                    cap,
                }
            };
            Ok(())
        }
        other => Err(InterpError::type_mismatch(
            op,
            "vector value",
            other.kind_name(),
        )),
    }
}

/// Registration fragment of the builtin name registry: the `Vec::into_boxed_slice` conversion.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn vec_boxed_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        _ if canonical.starts_with("std::vec::Vec")
            && canonical.ends_with(">::into_boxed_slice") =>
        {
            Builtin::VecIntoBoxedSlice
        }
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the `Vec` surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn vec_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::new") => {
            Builtin::VecNew
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::push") => {
            Builtin::VecPush
        }
        _ if canonical.starts_with("std::vec::Vec")
            && canonical.ends_with(">::extend_from_slice") =>
        {
            Builtin::VecExtendFromSlice
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::len") => {
            Builtin::VecLen
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::is_empty") => {
            Builtin::VecIsEmpty
        }
        _ if canonical.starts_with("std::vec::Vec")
            && (canonical.ends_with(">::with_capacity")
                || canonical.ends_with(">::with_capacity_in")) =>
        {
            Builtin::VecWithCapacity
        }
        "std::vec::from_elem" => Builtin::VecFromElem,
        _ if canonical.starts_with("std::vec::Vec")
            && (canonical.ends_with(">::reserve")
                || canonical.ends_with(">::reserve_exact")) =>
        {
            Builtin::VecReserve
        }
        _ if canonical.starts_with("std::vec::Vec")
            && (canonical.ends_with(">::try_reserve")
                || canonical.ends_with(">::try_reserve_exact")) =>
        {
            Builtin::VecTryReserve
        }
        _ if canonical.starts_with("std::vec::Vec")
            && (canonical.ends_with(">::shrink_to_fit")
                || canonical.ends_with(">::shrink_to")) =>
        {
            Builtin::VecShrink
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::set_len") => {
            Builtin::VecSetLen
        }
        _ if canonical.starts_with("std::vec::Vec")
            && (canonical.ends_with(">::as_ptr") || canonical.ends_with(">::as_mut_ptr")) =>
        {
            Builtin::VecAsPtr
        }
        _ if canonical.starts_with("std::vec::Vec")
            && canonical.ends_with(">::extend_trusted") =>
        {
            Builtin::VecExtendTrusted
        }
        "std::vec::spec_from_iter_nested::SpecFromIterNested::from_iter" => {
            Builtin::VecFromIterNested
        }
        // The in-place collect entry the `SpecFromIter`
        // specialization selects when the iterator's source is a
        // `Vec` allocation (`SourceIter` + `InPlaceIterable` with
        // an `AsVecIntoIter` source: `vec::IntoIter` itself and
        // every adapter forwarding those traits). Its collected
        // body drives the source allocation's raw buffer through
        // `SourceIter::as_inner` / `AsVecIntoIter::as_into_iter`,
        // which the value model does not serve; the observable
        // semantics are the nested collect's -- the yielded
        // items, in order, in a fresh vector -- so the same
        // model serves this entry.
        "std::vec::in_place_collect::from_iter_in_place"
        | "alloc::vec::in_place_collect::from_iter_in_place" => {
            Builtin::VecFromIterNested
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::capacity") => {
            Builtin::VecCapacity
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::remove::assert_failed") => Builtin::VecRemoveAssertFailed,
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::insert_mut::assert_failed") => Builtin::VecInsertAssertFailed,
        _ => return None,
    })
}
