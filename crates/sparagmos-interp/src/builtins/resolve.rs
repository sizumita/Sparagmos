//! Reference resolution and element scanning for the builtin executors.

use std::borrow::Cow;


use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter, Resolved,
};
use crate::ir::Ty;
use crate::value::{RefProj, RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// `Deref` / `DerefMut` for `smallvec::SmallVec<[T; N]>`
    /// (smallvec 1.x): the slice the vector's elements form.
    ///
    /// The value is the crate's real struct shape
    /// `{ capacity: usize, data: SmallVecData, .. }`, and the
    /// capacity word alone selects the storage, by the crate's
    /// own invariant: at or below the inline capacity `N` the
    /// word IS the length and the elements sit in the data
    /// word's inline field; above it the word is the heap
    /// allocation's capacity and the data word's heap field
    /// holds the `(pointer, length)` pair. The data word's own
    /// shape depends on the crate's `union` feature -- a
    /// two-field union value (inline field first; the inline
    /// field is a `ManuallyDrop` over the `MaybeUninit`
    /// buffer) or an enum (`Inline` / `Heap` variants, the
    /// inline variant holding the `MaybeUninit` directly) --
    /// and the extraction follows the held value's shape
    /// rather than the feature: every layer between the data
    /// word and the element array is a one-field aggregate, so
    /// the walk that reaches the array value also counts the
    /// field steps that denote the same array in the vector's
    /// place, which is what the result's base names when the
    /// vector is place-backed (so writes through a `deref_mut`
    /// result land in the vector). A detached vector value
    /// yields a reference to the extracted array itself, as
    /// for constant aggregates.
    pub(super) fn smallvec_slice(
        &self,
        frames: &mut Vec<Frame>,
        arg: &Value,
        self_ty: &str,
        what: &str,
    ) -> InterpResult<Value> {
        let (elem_tag, inline_cap) = smallvec_array_arg(self_ty).ok_or_else(|| {
            unsupported(format!(
                "{what}: cannot read an inline capacity from {self_ty}"
            ))
        })?;
        let vec = self.resolve_value(frames, arg)?;
        let Value::Enum { fields, .. } = &vec else {
            return Err(InterpError::type_mismatch(
                what,
                "SmallVec aggregate",
                vec.kind_name(),
            ));
        };
        if fields.len() < 2 {
            return Err(InterpError::type_mismatch(
                what,
                "SmallVec aggregate with capacity and data fields",
                vec.kind_name(),
            ));
        }
        let capacity_word = value_as_usize(&fields[0]).ok_or_else(|| {
            InterpError::type_mismatch(
                what,
                "SmallVec capacity word",
                fields[0].kind_name(),
            )
        })?;
        let data = fields[1].clone();
        if capacity_word <= inline_cap {
            // Inline storage: the capacity word is the length.
            let len = capacity_word as u128;
            let Value::Enum {
                fields: data_fields,
                ..
            } = &data
            else {
                return Err(InterpError::type_mismatch(
                    what,
                    "SmallVec data aggregate",
                    data.kind_name(),
                ));
            };
            let mut node = data_fields.first().cloned().ok_or_else(|| {
                InterpError::type_mismatch(
                    what,
                    "SmallVec data with an inline field",
                    data.kind_name(),
                )
            })?;
            let mut steps = vec![RefProj::Field(1), RefProj::Field(0)];
            while let Value::Enum { fields: inner, .. } = &node {
                if inner.len() != 1 {
                    break;
                }
                node = inner[0].clone();
                steps.push(RefProj::Field(0));
            }
            // The buffer's elements were written through raw
            // pointers (the crate's own `push` / `grow` paths),
            // which leaves the array in the model's scalar-backed
            // form -- the whole initialised region packed into
            // the first slot's scalar, the remaining slots
            // uninitialised -- a form structural element reads
            // cannot take apart. Expand it to the element form
            // (only the first `len` elements are initialised;
            // the spare slots stay uninitialised) and, when the
            // vector is place-backed, write the expansion back
            // into the vector's place: the representation
            // changes, the bytes denoted do not, and every later
            // structural access (this slice's reads and writes
            // included) sees the elements themselves.
            let unpacked = self.unpack_scalar_backed_array(
                &node,
                &elem_tag,
                inline_cap,
                len as usize,
            );
            if unpacked != node {
                if let Ok(target) = self.deref_value(frames, arg) {
                    let target = match target {
                        Resolved::Place {
                            frame_pos,
                            local,
                            mut projection,
                        } => {
                            projection.extend(steps.iter().cloned());
                            Some(Resolved::Place {
                                frame_pos,
                                local,
                                projection,
                            })
                        }
                        Resolved::Static {
                            name,
                            mut projection,
                        } => {
                            projection.extend(steps.iter().cloned());
                            Some(Resolved::Static { name, projection })
                        }
                        Resolved::Alloc {
                            id,
                            mut projection,
                        } => {
                            projection.extend(steps.iter().cloned());
                            Some(Resolved::Alloc { id, projection })
                        }
                        Resolved::Value(..) => None,
                    };
                    if let Some(target) = target {
                        self.write_resolved(frames, &target, unpacked.clone())?;
                    }
                }
                node = unpacked;
            }
            // The base denotes the first element, not the
            // array itself: the form a raw pointer to the
            // buffer (the crate's own `as_ptr`) takes, which
            // is what the slice machinery (index composition,
            // element iteration bounded by the fat length)
            // expects of a base whose storage is wider than
            // the slice.
            steps.push(RefProj::Index(0));
            if let Some(place_ref) =
                self.place_projected_ref_steps(frames, arg, &steps)?
            {
                return Ok(Value::FatRef {
                    base: Box::new(place_ref),
                    len,
                });
            }
            return Ok(Value::FatRef {
                base: Box::new(Value::Ref(RefTarget::Value(
                    Box::new(node),
                    vec![RefProj::Index(0)],
                ))),
                len,
            });
        }
        // Spilled storage: the heap field holds the element
        // pointer and the length; the capacity word is the
        // allocation's capacity.
        let (ptr, len) = smallvec_heap_parts(&data).ok_or_else(|| {
            InterpError::type_mismatch(
                what,
                "SmallVec data holding a (pointer, length) heap field",
                data.kind_name(),
            )
        })?;
        Ok(Value::FatRef {
            base: Box::new(ptr),
            len: len as u128,
        })
    }

    /// The reference `step` inside the place a reference value
    /// denotes, when the reference denotes a live place: a frame
    /// place, a static's storage or an allocation, with `step`
    /// appended to its projection. Returns `None` when the
    /// reference denotes no place (a value snapshot), where the
    /// caller falls back to resolving against the value itself.
    ///
    /// Builtins that hand out references to a part of a value they
    /// were given a reference to (the pointee of an
    /// aggregate-shaped box) use this to keep the result denoting
    /// the same storage the part lives in, so writes through the
    /// result \u2014 interior mutation, `SetDiscriminant`, a
    /// `Cell::set` write-back \u2014 land in that storage instead of
    /// failing on an immutable snapshot of it.
    pub(super) fn place_projected_ref(
        &self,
        frames: &[Frame],
        arg: &Value,
        step: RefProj,
    ) -> InterpResult<Option<Value>> {
        Ok(match self.deref_value(frames, arg)? {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(step);
                Some(Value::Ref(RefTarget::Place {
                    frame: frames[frame_pos].id,
                    local,
                    projection,
                }))
            }
            Resolved::Static {
                name,
                mut projection,
            } => {
                projection.push(step);
                Some(Value::Ref(RefTarget::Static { name, projection }))
            }
            Resolved::Alloc {
                id,
                mut projection,
            } => {
                projection.push(step);
                Some(Value::Ref(RefTarget::Alloc { id, projection }))
            }
            Resolved::Value(..) => None,
        })
    }

    /// A reference to field 0 of the aggregate `arg` points at:
    /// the dereference result of a one-field newtype wrapper
    /// (`AssertUnwindSafe`). A place-backed argument yields a live
    /// place reference, so writes through the result land in the
    /// wrapper; a detached aggregate value yields a reference to a
    /// copy of the field.
    pub(crate) fn field_zero_ref(
        &self,
        frames: &[Frame],
        arg: &Value,
        what: &str,
    ) -> InterpResult<Value> {
        Ok(match self.deref_value(frames, arg)? {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(RefProj::Field(0));
                Value::Ref(RefTarget::Place {
                    frame: frames[frame_pos].id,
                    local,
                    projection,
                })
            }
            Resolved::Static {
                name,
                mut projection,
            } => {
                projection.push(RefProj::Field(0));
                Value::Ref(RefTarget::Static { name, projection })
            }
            Resolved::Alloc { id, mut projection } => {
                projection.push(RefProj::Field(0));
                Value::Ref(RefTarget::Alloc { id, projection })
            }
            Resolved::Value(base, projection) => {
                let value = self.value_target_value(&base, &projection)?;
                Value::Ref(RefTarget::value(one_field_value(value, what)?))
            }
        })
    }

    /// Read a reference value exactly one level: the value it
    /// names, without following that value further if it is itself
    /// a reference. The `Deref` / `DerefMut` impls for references
    /// have exactly this shape: `deref(&&T) -> &T` reads the
    /// argument place once and returns the inner reference itself.
    /// (`resolve_value` follows the chain to the final pointee,
    /// which hands the caller the pointee where the inner
    /// reference was asked for.)
    pub(crate) fn read_ref_once(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Value> {
        match value {
            // The null and address-only pointers denote no data;
            // one more level of reading is the value itself.
            Value::Ref(RefTarget::Null) | Value::Ref(RefTarget::Address(_)) => {
                Ok(value.clone())
            }
            Value::Ref(_) => {
                let target = self.deref_value(frames, value)?;
                self.read_resolved(frames, &target)
            }
            _ => Ok(value.clone()),
        }
    }

    /// Follow references in `value` until a non-reference value.
    pub(crate) fn resolve_value(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Value> {
        Ok(self.resolve_ref(frames, value)?.into_owned())
    }

    /// Follow references in `value` until a non-reference value,
    /// borrowing the denoted data wherever it is already
    /// materialised: the borrowing form of [`resolve_value`], with
    /// the same results and errors. The shapes that denote no data
    /// (null and address-only pointers) and a fat pointer whose
    /// base is self-contained (an element tuple — a slice constant,
    /// a materialised sub-slice — or a sparse sequence) or denotes
    /// no data (an address-only pointer: a `from_raw_parts` slice)
    /// resolve to themselves: dereferencing any of those yields
    /// the fat pointer itself, so following it would spin.
    pub(crate) fn resolve_ref<'x>(
        &self,
        frames: &'x [Frame],
        value: &'x Value,
    ) -> InterpResult<Cow<'x, Value>> {
        let mut cur: Cow<'x, Value> = Cow::Borrowed(value);
        // The length of the fat pointer the walk last followed
        // through, when that step was a fat-pointer step: a fat
        // pointer of length 1 whose pointee is a scalar denotes
        // the one-element sequence holding that scalar (a
        // `Box<T>` allocation reinterpreted as a `[T; 1]` by
        // `Box::into_boxed_slice`), so the followed value is
        // denominated as that sequence where it lands.
        let mut fat_len: Option<u128> = None;
        loop {
            let follows = match &*cur {
                Value::Ref(RefTarget::Null) | Value::Ref(RefTarget::Address(_)) => false,
                // The fat pointers `deref_value` resolves to
                // themselves: a self-contained base (an element
                // aggregate, a sparse sequence) or an
                // address-only base, which has no pointee to
                // reach. The guard here must name exactly the
                // shapes `deref_value` names, or following such
                // a pointer spins.
                Value::FatRef { base, .. }
                    if matches!(
                        &**base,
                        Value::Tuple(_)
                            | Value::SparseVec { .. }
                            | Value::Ref(RefTarget::Address(_))
                    ) =>
                {
                    false
                }
                Value::Ref(_) | Value::FatRef { .. } => true,
                _ => false,
            };
            // A fat pointer whose base denotes one element of
            // a container (an element-entry base: the string
            // view the `as_os_str` / `as_path` formations leave
            // over an owner's buffer, a raw-parts or sub-slice
            // base over `container[i]`) denotes the span of its
            // carried length from that element on, not the
            // element itself: following the base would hand the
            // caller the first element alone (a byte span of
            // any length mistaken for its one first byte, the
            // defect pinned on #513). The pointer resolves to
            // itself, as the self-contained bases above do;
            // consumers that want the span's elements read
            // them through `slice_elements`, whose sibling
            // walk is built on exactly this reading.
            let follows = follows
                && !matches!(&*cur, Value::FatRef { base, .. }
                    if self.deref_value(frames, base)
                        .is_ok_and(|target| target_ends_at_element(&target)));
            if !follows {
                break;
            }
            fat_len = match &*cur {
                Value::FatRef { len, .. } => Some(*len),
                _ => None,
            };
            let target = self.deref_value(frames, &cur)?;
            cur = self.read_resolved_ref(frames, &target)?;
        }
        if fat_len == Some(1) {
            match &*cur {
                Value::Bool(_)
                | Value::Int(_)
                | Value::Uint(_)
                | Value::Float(_)
                | Value::Float128(_) => {
                    let scalar = cur.into_owned();
                    return Ok(Cow::Owned(Value::Tuple(vec![scalar])));
                }
                _ => {}
            }
        }
        Ok(cur)
    }

    /// The bytes behind a raw C string pointer, up to (not
    /// including) the first NUL byte: the scan the C library
    /// `strlen` performs. String and byte data are scanned
    /// directly; any other aggregate is read element by
    /// element, as [`Self::read_byte_range`] reads a fixed
    /// range. Storage with no NUL in it is an explicit error,
    /// never a silently truncated length.
    pub(crate) fn read_c_str(&self, frames: &[Frame], ptr: &Value) -> InterpResult<Vec<u8>> {
        let target = self.deref_value(frames, ptr)?;
        let base = self.read_resolved(frames, &target)?;
        let direct: Option<&[u8]> = match &base {
            Value::Str(s) => Some(s.as_bytes()),
            Value::Bytes(b) => Some(b.as_slice()),
            _ => None,
        };
        if let Some(bytes) = direct {
            return match bytes.iter().position(|b| *b == 0) {
                Some(end) => Ok(bytes[..end].to_vec()),
                None => Err(unsupported(
                    "strlen of a string whose storage has no NUL terminator",
                )),
            };
        }
        let mut out = Vec::new();
        loop {
            match self.read_seq_at(frames, &target, out.len())? {
                Value::Uint(0) => return Ok(out),
                Value::Uint(b) => out.push(b as u8),
                // A signed byte element (an `i8` holding a byte
                // above 127) keeps its bit pattern: the range
                // runs from -128, not 0.
                Value::Int(b) if (-128..=255).contains(&b) => {
                    if b == 0 {
                        return Ok(out);
                    }
                    out.push(b as u8);
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "strlen element",
                        "byte",
                        other.kind_name(),
                    ));
                }
            }
        }
    }

    /// The argument of a body-less libc string conversion
    /// (`atol` and family) as a host C string: the guest
    /// bytes from the argument pointer up to the first NUL,
    /// re-terminated for the host call. The scanned bytes
    /// contain no NUL by construction, so the host string
    /// holds exactly the bytes the C function's own scan
    /// would see.
    pub(super) fn c_str_arg(&self, frames: &[Frame], ptr: &Value) -> InterpResult<std::ffi::CString> {
        let bytes = match self.scan_element_ptr(frames, ptr)? {
            Some(bytes) => bytes,
            None => self.read_c_str(frames, ptr)?,
        };
        std::ffi::CString::new(bytes).map_err(|_| {
            unsupported("libc string conversion argument contains a NUL byte")
        })
    }

    /// The width in bytes of the elements of the sequence
    /// `target`'s projection prefix denotes, from the
    /// declared type recorded for the target's root (a
    /// place's local type, a static's declared type) walked
    /// through the prefix: `Some` only when the prefix
    /// denotes an array or slice of integers. `None` when no
    /// root type is recorded, or when a step's type cannot
    /// be followed (ADT fields, downcasts, byte
    /// displacements): the caller then reads the storage by
    /// value alone, which cannot tell a wider-element array
    /// whose values fit in a byte apart from a byte array.
    fn sequence_elem_width(
        &self,
        frames: &[Frame],
        target: &RefTarget,
        prefix: &[RefProj],
    ) -> Option<usize> {
        let mut cur: Ty = match target {
            RefTarget::Place { frame, local, .. } => frames
                .iter()
                .find(|f| f.id == *frame)
                .and_then(|f| self.program().functions.get(&f.func))
                .and_then(|body| body.local_tys.get(*local))
                .cloned()?,
            RefTarget::Static { name, .. } => self.program().static_tys.get(name).cloned()?,
            _ => return None,
        };
        for step in prefix {
            cur = match step {
                RefProj::Field(k) => match cur {
                    Ty::Tuple(ts) => ts.get(*k).cloned()?,
                    _ => return None,
                },
                RefProj::Index(_) => match cur {
                    Ty::Array(elem, _) | Ty::Slice(elem) => *elem,
                    _ => return None,
                },
                RefProj::Subslice { .. } => match cur {
                    Ty::Array(..) | Ty::Slice(..) => cur,
                    _ => return None,
                },
                _ => return None,
            };
        }
        match cur {
            Ty::Array(elem, _) | Ty::Slice(elem) => match *elem {
                Ty::Int(t) => Some((t.bits() / 8) as usize),
                Ty::Uint(t) => Some((t.bits() / 8) as usize),
                _ => None,
            },
            _ => None,
        }
    }

    /// Scan a C string whose argument pointer denotes one
    /// element of a byte sequence — the shape of
    /// `CString::as_ptr()` and of slice element pointers,
    /// whose projection ends in an `Index` step. Relative
    /// reads against the element itself do not traverse the
    /// sequence (the read past the element reports the
    /// terminator), so the scan ascends the pointer's
    /// projection to the sequence it denotes an element of:
    /// the longest projection prefix resolving to a byte
    /// sequence is the base, and the remaining steps —
    /// `Index` / `Field` / `Subslice` element steps, plus
    /// byte displacements over a byte-addressed base —
    /// give the starting offset in it. `Ok(None)` when no
    /// prefix resolves to a byte sequence (the caller then
    /// uses the whole-sequence scan); a resolved sequence
    /// with no NUL at or after the offset is an explicit
    /// error, never a silently truncated string.
    fn scan_element_ptr(
        &self,
        frames: &[Frame],
        ptr: &Value,
    ) -> InterpResult<Option<Vec<u8>>> {
        let target = match ptr {
            Value::Ref(t) => t.clone(),
            _ => return Ok(None),
        };
        let projection = match &target {
            RefTarget::Place { projection, .. }
            | RefTarget::Static { projection, .. }
            | RefTarget::Alloc { projection, .. }
            | RefTarget::Value(_, projection) => projection.clone(),
            _ => return Ok(None),
        };
        for prefix_len in (0..=projection.len()).rev() {
            let suffix = &projection[prefix_len..];
            let mut elems: usize = 0;
            let mut bytes: i128 = 0;
            let mut interpretable = true;
            for step in suffix {
                match step {
                    RefProj::Index(k) | RefProj::Field(k) => elems += k,
                    RefProj::Subslice { offset, .. } => elems += offset,
                    RefProj::ByteOffset(b) => bytes += b,
                    _ => {
                        interpretable = false;
                        break;
                    }
                }
            }
            if !interpretable {
                // The offending step is in every shorter
                // prefix's suffix too.
                return Ok(None);
            }
            let candidate = match self.with_projection(&target, projection[..prefix_len].to_vec())
            {
                Some(v) => v,
                None => return Ok(None),
            };
            let resolved = match self.deref_value(frames, &candidate) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let base = match self.read_resolved(frames, &resolved) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let direct: Option<&[u8]> = match &base {
                Value::Str(s) => Some(s.as_bytes()),
                Value::Bytes(b) => Some(b.as_slice()),
                _ => None,
            };
            if let Some(storage) = direct {
                let start = elems + bytes as usize;
                if bytes < 0 || start > storage.len() {
                    continue;
                }
                return match storage[start..].iter().position(|b| *b == 0) {
                    Some(end) => Ok(Some(storage[start..start + end].to_vec())),
                    None => Err(unsupported(
                        "libc string conversion argument has no NUL terminator",
                    )),
                };
            }
            let Value::Tuple(fields) = &base else {
                continue;
            };
            // A sequence whose declared elements are wider
            // than a byte: a `char*` cast into such storage
            // denotes the storage's bytes, not its elements,
            // so scan the elements' little-endian bytes (the
            // target this model runs on is little-endian).
            // The declared type is what tells this storage
            // apart from a byte array; where no declared type
            // is recorded the scan below reads by value alone
            // and cannot tell the two apart.
            if let Some(width) = self
                .sequence_elem_width(frames, &target, &projection[..prefix_len])
                .filter(|w| *w > 1)
            {
                if bytes < 0 {
                    continue;
                }
                let Some(byte_start) = elems
                    .checked_mul(width)
                    .and_then(|s| s.checked_add(bytes as usize))
                else {
                    continue;
                };
                let mut out: Vec<u8> = Vec::new();
                let mut pos = byte_start;
                let mut candidate_live = true;
                'bytes: loop {
                    let idx = pos / width;
                    let within = pos % width;
                    let elem_bytes: [u8; 16] = match self.read_seq_at(frames, &resolved, idx) {
                        Ok(Value::Uint(v)) => v.to_le_bytes(),
                        Ok(Value::Int(v)) => v.to_le_bytes(),
                        Ok(other) => {
                            if out.is_empty() {
                                candidate_live = false;
                                break;
                            }
                            return Err(InterpError::type_mismatch(
                                "libc string conversion element",
                                "integer element",
                                other.kind_name(),
                            ));
                        }
                        Err(e) => {
                            if out.is_empty() {
                                candidate_live = false;
                                break;
                            }
                            return Err(e);
                        }
                    };
                    for &byte in &elem_bytes[within..width] {
                        if byte == 0 {
                            break 'bytes;
                        }
                        out.push(byte);
                    }
                    pos += width - within;
                }
                if candidate_live {
                    return Ok(Some(out));
                }
                continue;
            }
            // The starting element: the element steps, plus
            // any byte displacement when the stored elements
            // are themselves single bytes (a byte displacement
            // over a byte sequence is an element
            // displacement). Where no declared element
            // type is recorded, a wider-element array whose
            // values all happen to fit in a byte remains
            // indistinguishable from a byte array in the
            // value model, and a `char*` displaced into
            // such storage is outside what this scan can
            // tell apart; any other element shape abandons
            // the candidate.
            let start = if bytes == 0 {
                elems
            } else {
                let all_bytes = fields.iter().all(|f| {
                    matches!(f, Value::Uint(v) if *v <= 0xFF)
                        || matches!(f, Value::Int(v) if (-128..=255).contains(v))
                });
                if bytes < 0 || !all_bytes {
                    continue;
                }
                elems + bytes as usize
            };
            let mut out: Vec<u8> = Vec::new();
            let mut candidate_live = true;
            loop {
                match self.read_seq_at(frames, &resolved, start + out.len()) {
                    Ok(Value::Uint(0)) => break,
                    Ok(Value::Uint(b)) => out.push(b as u8),
                    // As in `read_c_str`: a signed byte element
                    // keeps its bit pattern (range from -128).
                    Ok(Value::Int(b)) if (-128..=255).contains(&b) => {
                        if b == 0 {
                            break;
                        }
                        out.push(b as u8);
                    }
                    Ok(other) => {
                        if out.is_empty() {
                            candidate_live = false;
                            break;
                        }
                        return Err(InterpError::type_mismatch(
                            "libc string conversion element",
                            "byte",
                            other.kind_name(),
                        ));
                    }
                    Err(e) => {
                        if out.is_empty() {
                            candidate_live = false;
                            break;
                        }
                        return Err(e);
                    }
                }
            }
            if candidate_live {
                return Ok(Some(out));
            }
        }
        Ok(None)
    }

    /// `target` with its projection replaced, as a reference
    /// value; `None` for target kinds that carry no
    /// projection.
    fn with_projection(&self, target: &RefTarget, projection: Vec<RefProj>) -> Option<Value> {
        Some(match target {
            RefTarget::Place { frame, local, .. } => Value::Ref(RefTarget::Place {
                frame: *frame,
                local: *local,
                projection,
            }),
            RefTarget::Static { name, .. } => Value::Ref(RefTarget::Static {
                name: name.clone(),
                projection,
            }),
            RefTarget::Alloc { id, .. } => Value::Ref(RefTarget::Alloc {
                id: *id,
                projection,
            }),
            RefTarget::Value(v, _) => Value::Ref(RefTarget::Value(v.clone(), projection)),
            _ => return None,
        })
    }
}
