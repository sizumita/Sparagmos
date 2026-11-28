//! Slice views, snapshot rescue and element access helpers.

use std::borrow::Cow;


use crate::error::{InterpError, InterpResult};
use crate::ir::Ty;
use crate::interp::{
    layered_wrapper_pointer, project_value, rc_handle_id, scalar_storage_byte,
    scalar_storage_len, Frame, Interpreter, Resolved,
};
use crate::value::{RefProj, RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// The character set of a `&[char]` pattern argument, read
    /// through `slice_elements` so the length and the sibling
    /// elements survive even when the slice's base references an
    /// individual element. `None` when the value is not a slice
    /// whose every element is a character.
    /// The bytes of a byte-slice argument, read through
    /// `slice_elements` so the length and the sibling elements
    /// survive even when the slice's base references an
    /// individual element. `None` when the value is not a slice
    /// whose every element is a byte.
    pub(super) fn byte_slice_of_fat_ref(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> Option<Vec<u8>> {
        let (elems, _) = self.slice_elements(frames, value).ok()?;
        let mut bytes = Vec::with_capacity(elems.len());
        for e in &elems {
            let Value::Uint(v) = e else { return None };
            bytes.push(u8::try_from(*v).ok()?);
        }
        Some(bytes)
    }

    pub(super) fn char_set_of_slice(&self, frames: &[Frame], value: &Value) -> Option<Vec<char>> {
        let (elems, _) = self.slice_elements(frames, value).ok()?;
        let mut set = Vec::with_capacity(elems.len());
        for e in &elems {
            let Value::Uint(v) = e else { return None };
            set.push(char::from_u32(*v as u32)?);
        }
        Some(set)
    }

    /// The element snapshot of a fat pointer whose base denotes
    /// a scalar's own storage (the `slice::from_ref` /
    /// `slice::from_mut` shape over a scalar place, or a raw
    /// view of one): the storage is the element sequence
    /// itself — one element of the scalar's own type, or the
    /// storage's bytes for a byte element. `None` for any other
    /// shape: a wider or narrower non-byte element over the
    /// storage is a punned view the plain readings judge (and
    /// reject) instead.
    pub(super) fn scalar_backed_slice_elements(
        &self,
        frames: &[Frame],
        value: &Value,
        elem_tag: &str,
    ) -> Option<Vec<Value>> {
        let Value::FatRef { base, len } = value else {
            return None;
        };
        let target = self.deref_value(frames, base).ok()?;
        // A base denoting an element position denotes that
        // element of a container, not a scalar's storage: the
        // sibling readings' shape. The position may be entered
        // through a wrapper's field steps (an element-entry
        // suffix) as well as ending at the index step itself.
        if self.base_denotes_element_span(frames, &target, *len) {
            return None;
        }
        let scalar = self.read_resolved(frames, &target).ok()?;
        if scalar_storage_len(&scalar).is_none() {
            return None;
        }
        let (size, _) = self.layout_of(elem_tag)?;
        if *len == 1
            && !matches!(elem_tag.trim(), "u8" | "i8")
            && self.target_is_scalar_storage(frames, &target, size as usize)
        {
            // One element of the storage's own width: the
            // element is the scalar itself, or -- for an
            // element of a different scalar kind -- its
            // punned reading under that kind. (Judged
            // before the byte reading below, whose one-byte case it
            // subsumes: a `bool` element — the
            // `slice::from_ref(&true)` shape — is the scalar,
            // not its byte.)
            return Some(vec![
                self.punned_scalar_element(frames, &target, elem_tag)
                    .unwrap_or(scalar),
            ]);
        }
        if size == 1 {
            // Byte elements: the storage's own bytes under its
            // declared layout, up to the fat length (a float
            // storage's bytes are its own width's, not the
            // widened model's). An `i8` element denotes the
            // byte's signed value: a consumer that resolves the
            // yielded reference directly (the formatting path)
            // reads the snapshot as the element itself, so an
            // unsigned construction would print 255 for -1.
            let signed = elem_tag.trim() == "i8";
            let layout_bytes =
                self.scalar_storage_layout_bytes(frames, &target, *len as usize);
            let mut out = Vec::new();
            for i in 0..(*len as usize) {
                let byte = match &layout_bytes {
                    Some(bytes) => *bytes.get(i)?,
                    None => {
                        if matches!(scalar, Value::Float(_) | Value::Float128(_)) {
                            // A float storage whose declared
                            // layout is not at hand has no honest
                            // byte reading: its model form is
                            // widened.
                            return None;
                        }
                        scalar_storage_byte(&scalar, i)?
                    }
                };
                out.push(if signed {
                    Value::Int(i128::from(byte as i8))
                } else {
                    Value::Uint(u128::from(byte))
                });
            }
            return Some(out);
        }
        if *len == 1 && self.target_is_scalar_storage(frames, &target, size as usize) {
            // One element of the storage's own width: the
            // element is the scalar itself, or its punned
            // reading under a different element kind.
            return Some(vec![
                self.punned_scalar_element(frames, &target, elem_tag)
                    .unwrap_or(scalar),
            ]);
        }
        None
    }

    /// The elements an iteration over the slice `value`
    /// yields under the element type `elem_ty` the
    /// consumer's own type names, when the slice is a fat
    /// pointer whose formation regrouped its storage's
    /// elements (see `foreign_slice_elements`): the
    /// elements composed from the storage's bytes under
    /// that type, rather than the stored shape the plain
    /// element reading materialises. A reference to the
    /// fat pointer is followed. `None` when the value
    /// denotes no such pointer or the view does not judge
    /// it; the caller keeps its ordinary element reading.
    pub(crate) fn regrouped_iter_elements(
        &self,
        frames: &[Frame],
        value: &Value,
        elem_ty: &Ty,
    ) -> Option<Vec<Value>> {
        let fat = match value {
            Value::FatRef { .. } => value.clone(),
            Value::Ref(_) => {
                let target = self.deref_value(frames, value).ok()?;
                match self.read_resolved(frames, &target).ok()? {
                    v @ Value::FatRef { .. } => v,
                    _ => return None,
                }
            }
            _ => return None,
        };
        self.foreign_slice_elements(frames, &fat, elem_ty)
    }

    pub(crate) fn slice_elements(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<(Vec<Value>, u128)> {
        let (elems, len) = self.slice_elements_ref(frames, value)?;
        // An owned copy of a sparse sequence's elements would
        // materialise its uninitialised tail: past the
        // materialisation limit that stays an explicit error
        // rather than an allocation that exhausts the host.
        // (Borrowing consumers use `slice_elements_ref` and read
        // the tail as the uninitialised slots they are.)
        if elems.stored_len() < elems.len()
            && elems.len() > MAX_VEC_SET_LEN_ELEMENTS
        {
            return Err(unsupported(format!(
                "a slice of {len} elements is beyond the value model's materialisation limit"
            )));
        }
        let mut elems = elems.into_vec();
        // An element of a single-valued type whose store MIR
        // elided reaches the snapshot as an uninitialised slot;
        // it denotes the type's one value (see
        // `rescue_elided_snapshot`).
        if let Some((frame_pos, local, path)) = self.seq_snapshot_key(frames, value) {
            let frame = &frames[frame_pos];
            self.rescue_elided_snapshot(frame, local, &path, &mut elems);
        }
        Ok((elems, len))
    }

    /// The (frame, local, path) key of the sequence a reference
    /// value denotes, for the elided-store rescue: the target
    /// place and the projection the sequence sits at within it.
    /// An element snapshot taken from such a place carries no
    /// type of its own, so this key is the snapshot's only
    /// route back to the canonical values conversion recorded
    /// for the place (see `Body::zst_fields`). `None` for
    /// value-backed targets (their elements were materialised
    /// by whoever built them).
    fn seq_snapshot_key(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> Option<(usize, usize, Vec<RefProj>)> {
        let target = match value {
            Value::FatRef { base, .. } => self.deref_value(frames, base).ok()?,
            _ => self.deref_value(frames, value).ok()?,
        };
        let Resolved::Place {
            frame_pos,
            local,
            projection,
        } = target
        else {
            return None;
        };
        let mut path = projection;
        if matches!(path.last(), Some(RefProj::Index(_))) {
            path.pop();
        }
        Some((frame_pos, local, path))
    }

    /// Rescue the elided-store elements of an element snapshot
    /// of the sequence at `path` in `local`: an element whose
    /// type has a single value denotes that value, which
    /// conversion recorded under the path's index step (see
    /// `Body::zst_fields`), so an element that reaches the
    /// snapshot as an uninitialised slot is replaced by it.
    /// Elements that are themselves sequences are rescued the
    /// same way one index step deeper, so a snapshot of a
    /// nested sequence (`Vec<Vec<..>>`) denotes its elements'
    /// elements too. Nothing is replaced at a path conversion
    /// recorded nothing for: an element of a many-valued type
    /// stays the uninitialised slot it is.
    fn rescue_elided_snapshot(
        &self,
        frame: &Frame,
        local: usize,
        path: &[RefProj],
        elems: &mut [Value],
    ) {
        let mut elem_path = path.to_vec();
        elem_path.push(RefProj::Index(0));
        if let Some(canonical) = self.zst_field_value(frame, local, &elem_path) {
            for elem in elems.iter_mut() {
                if *elem == Value::Uninit {
                    *elem = canonical.clone();
                }
            }
        }
        for elem in elems.iter_mut() {
            // An element that is itself a sequence is rescued
            // the same way one index step deeper; the uninit
            // slot it holds may sit several sequences down, so
            // the descent is not gated on what the element
            // immediately holds — the recorded-path lookup
            // inside decides whether anything denotes a value,
            // and a path conversion recorded nothing for (a
            // struct's fields sit under field steps, not index
            // steps) never fires.
            self.rescue_elided_value(frame, local, &elem_path, elem);
        }
    }

    /// [`Self::rescue_elided_snapshot`] on the elements a
    /// sequence value itself holds.
    fn rescue_elided_value(
        &self,
        frame: &Frame,
        local: usize,
        path: &[RefProj],
        value: &mut Value,
    ) {
        match value {
            Value::Tuple(items) => {
                self.rescue_elided_snapshot(frame, local, path, items);
                // A tuple's items sit under field steps (an
                // array's elements, rescued above, sit under an
                // index step instead; the recorded paths say
                // which of the two a given list is).
                if self.zst_recorded_branches(frame, local, path).0 {
                    self.rescue_elided_fields(frame, local, path, items);
                }
            }
            Value::Enum { variant, fields } => {
                self.rescue_elided_snapshot(frame, local, path, fields);
                // The same field list is a struct's fields under
                // plain field steps, or an enum variant's fields
                // under the downcast that selects the variant;
                // descend along whichever the recorded paths
                // take from here. (A modelled `Vec` value's
                // elements, rescued above, sit under an index
                // step instead.)
                let (field_below, downcast_below) =
                    self.zst_recorded_branches(frame, local, path);
                if field_below {
                    self.rescue_elided_fields(frame, local, path, fields);
                }
                if downcast_below {
                    let mut variant_path = path.to_vec();
                    variant_path.push(RefProj::Downcast(*variant));
                    self.rescue_elided_fields(
                        frame,
                        local,
                        &variant_path,
                        fields,
                    );
                }
            }
            Value::SparseVec { elems, len, .. } => {
                self.rescue_elided_snapshot(frame, local, path, elems);
                // The virtual tail: the slots the declared
                // length runs past the stored prefix denote the
                // element type's single value when conversion
                // recorded one, exactly as an elided stored slot
                // does. A snapshot detached from the place has
                // no later keyed rescue to read them through, so
                // they are materialised here, within the bound
                // an owned materialisation enforces; past the
                // bound (or with nothing recorded) the tail
                // stays the uninitialised slots it denotes.
                let declared = *len;
                if declared > elems.len() as u128
                    && declared <= MAX_VEC_SET_LEN_ELEMENTS as u128
                {
                    let mut elem_path = path.to_vec();
                    elem_path.push(RefProj::Index(0));
                    if let Some(canonical) =
                        self.zst_field_value(frame, local, &elem_path)
                    {
                        elems.resize(declared as usize, canonical);
                    }
                }
            }
            _ => {}
        }
    }

    /// [`Self::rescue_elided_snapshot`] along field steps: the
    /// fields of an aggregate value at `path` -- a struct's or
    /// tuple's fields under their field steps, or one enum
    /// variant's fields with the downcast already in `path`. A
    /// field whose type has a single value denotes that value,
    /// which conversion recorded under the field's step, so a
    /// field that reaches the snapshot as an uninitialised slot
    /// is replaced by it; a field that is itself an aggregate
    /// or a sequence is rescued the same way beneath its step.
    /// Nothing is replaced at a path conversion recorded
    /// nothing for.
    fn rescue_elided_fields(
        &self,
        frame: &Frame,
        local: usize,
        path: &[RefProj],
        fields: &mut [Value],
    ) {
        for (i, field) in fields.iter_mut().enumerate() {
            let mut field_path = path.to_vec();
            field_path.push(RefProj::Field(i));
            if *field == Value::Uninit {
                if let Some(canonical) =
                    self.zst_field_value(frame, local, &field_path)
                {
                    *field = canonical;
                }
            }
            self.rescue_elided_value(frame, local, &field_path, field);
        }
    }

    /// The elements of a slice whose base pointer is address-only
    /// (a `from_raw_parts` slice over an integer address: the
    /// model holds no storage behind it), when the element type
    /// has a single value (see [`single_value_for_tag`]): every
    /// element of such a slice is that value, so the elements
    /// exist without storage. `None` for every other shape — the
    /// caller keeps its original error. The length is bounded by
    /// the same materialisation limit [`slice_elements`] applies,
    /// since the result is a materialised element list.
    pub(super) fn storageless_slice_elements(
        &self,
        frames: &[Frame],
        value: &Value,
        elem_tag: &str,
    ) -> Option<(Vec<Value>, u128)> {
        let resolved = self.resolve_value(frames, value).ok()?;
        let Value::FatRef { base, len } = &resolved else {
            return None;
        };
        if !matches!(&**base, Value::Ref(RefTarget::Address(_))) {
            return None;
        }
        if *len > MAX_VEC_SET_LEN_ELEMENTS as u128 {
            return None;
        }
        let elem = single_value_for_tag(elem_tag)?;
        Some((vec![elem; *len as usize], *len))
    }

    /// The borrowing form of [`slice_elements`]: the same elements
    /// and length, but the element storage is lent out of the
    /// resolved value wherever it is already materialised, so a
    /// consumer that needs one element (or a comparison that stops
    /// early) does not pay for a snapshot of all of them.
    /// The live element reference for an index into a
    /// place-backed receiver (the shared-`Index` counterpart of
    /// the resolution `IndexMut::index_mut` performs): the
    /// receiver's own place with an `Index` step appended, so
    /// reads and interior-mutability writes reach the container's
    /// storage. `Ok(None)` when the receiver is not backed by a
    /// live place holding an indexable aggregate — the caller
    /// falls back to a snapshot of the element.
    pub(super) fn index_place_ref(
        &self,
        frames: &[Frame],
        receiver: &Value,
        idx: usize,
        fat_len: Option<u128>,
    ) -> InterpResult<Option<Value>> {
        if !matches!(receiver, Value::Ref(_) | Value::FatRef { .. }) {
            return Ok(None);
        }
        let target = self.deref_value(frames, receiver)?;
        if matches!(target, Resolved::Value(..)) {
            return Ok(None);
        }
        // A target ending in an element-entry suffix (see
        // `element_entry_suffix`) already denotes an entered
        // element of its container: the element reference is
        // that position advanced by the index, under the fat
        // length's bound — the entered element's own length and
        // shape say nothing about the sequence, so without a fat
        // length the snapshot path answers instead.
        let target_projection = match &target {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. }
            | Resolved::Value(_, projection) => projection.clone(),
        };
        let has_suffix = element_entry_suffix(&target_projection).is_some();
        if has_suffix && fat_len.is_none() {
            return Ok(None);
        }
        let entry_suffix = match fat_len {
            Some(len) => {
                has_suffix && self.entry_suffix_denotes_elements(frames, &target, len)
            }
            None => false,
        };
        // The base is only inspected (its length and form): read
        // it by borrowing, so indexing a vector does not clone
        // every element before returning one element's reference.
        let base = self.read_resolved_ref(frames, &target)?;
        let len = match fat_len {
            Some(len) => len,
            None => match base.len() {
                Some(len) => len,
                None => return Ok(None),
            },
        };
        if idx as u128 >= len {
                return Err(self.bounds_check_panic(frames, len as u128, idx as u128));
        }
        if entry_suffix {
            // The bounds check above ran against the fat length.
            let Some(projection) = advance_entry_projection(&target_projection, idx) else {
                return Ok(None);
            };
            let refr = match target {
                Resolved::Place {
                    frame_pos, local, ..
                } => Value::Ref(RefTarget::Place {
                    frame: frames[frame_pos].id,
                    local,
                    projection,
                }),
                Resolved::Static { name, .. } => {
                    Value::Ref(RefTarget::Static { name, projection })
                }
                Resolved::Alloc { id, .. } => Value::Ref(RefTarget::Alloc { id, projection }),
                Resolved::Value(..) => unreachable!("returned above"),
            };
            return Ok(Some(refr));
        }
        if !matches!(
            &*base,
            Value::Tuple(_) | Value::Enum { .. } | Value::SparseVec { .. }
        ) {
            return Ok(None);
        }
        let refr = match target {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(RefProj::Index(idx));
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
                projection.push(RefProj::Index(idx));
                Value::Ref(RefTarget::Static { name, projection })
            }
            Resolved::Alloc { id, mut projection } => {
                projection.push(RefProj::Index(idx));
                Value::Ref(RefTarget::Alloc { id, projection })
            }
            Resolved::Value(..) => unreachable!("returned above"),
        };
        Ok(Some(refr))
    }

    /// Whether the `len` elements a fat pointer denotes can be
    /// the elements of the container its base target sits in,
    /// starting at the target's trailing element index. A fat
    /// pointer formed by dereferencing a container value that
    /// itself sits at an element position (a `Vec` nested inside
    /// another container, bound by a deref pattern) denotes that
    /// value's own elements, whose count the container's
    /// remaining span can fall short of; such a span cannot be
    /// the container's elements, so the answer is no and the
    /// caller reads the pointee's own elements instead. Any
    /// shape this cannot judge answers yes, keeping the sibling
    /// reading.
    fn sibling_span_fits(&self, frames: &[Frame], target: &Resolved, len: u128) -> bool {
        let projection = match target {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. }
            | Resolved::Value(_, projection) => projection,
        };
        let Some((&RefProj::Index(start), head)) = projection.split_last() else {
            return true;
        };
        let parent = match target {
            Resolved::Place {
                frame_pos,
                local,
                ..
            } => Resolved::Place {
                frame_pos: *frame_pos,
                local: *local,
                projection: head.to_vec(),
            },
            Resolved::Static { name, .. } => Resolved::Static {
                name: name.clone(),
                projection: head.to_vec(),
            },
            Resolved::Alloc { id, .. } => Resolved::Alloc {
                id: *id,
                projection: head.to_vec(),
            },
            Resolved::Value(value, _) => Resolved::Value(value.clone(), head.to_vec()),
        };
        match self.read_resolved(frames, &parent) {
            Ok(container) => match container.len() {
                Some(count) => (start as u128) + len <= count,
                None => true,
            },
            Err(_) => true,
        }
    }

    /// Whether the value a fat pointer base target denotes is
    /// itself a sequence of exactly the fat length: the base
    /// then denotes that sequence directly -- an array value or
    /// a `Vec` value in element form sitting at an element
    /// position of an outer container, which a deref / unsizing
    /// formation ranges over its own elements -- not the outer
    /// container elements from that position on (the raw-parts
    /// reading `target_ends_at_element` describes). The two
    /// readings collide exactly when the lengths agree; as in
    /// `entry_suffix_denotes_elements`, the direct reading wins
    /// that collision.
    fn fat_base_is_the_sequence(
        &self,
        frames: &[Frame],
        target: &Resolved,
        len: u128,
    ) -> bool {
        match self.read_resolved(frames, target) {
            Ok(value) => {
                matches!(
                    value,
                    Value::Tuple(_)
                        | Value::Enum { .. }
                        | Value::SparseVec { .. }
                        | Value::Bytes(_)
                        | Value::Str(_)
                ) && value.len() == Some(len)
            }
            Err(_) => false,
        }
    }

    pub(super) fn slice_elements_ref<'x>(
        &self,
        frames: &'x [Frame],
        value: &'x Value,
    ) -> InterpResult<(SliceElems<'x>, u128)> {
        match value {
            Value::FatRef { base, len } => {
                // A base pointer denoting one element of a
                // container (a raw-parts slice formed over an
                // offset element pointer) denotes the elements
                // from that position on, not the pointed element
                // itself: read them through the sequence accessor,
                // which continues to the element's siblings in the
                // container (and reads within the element when the
                // element is itself an aggregate).
                if let Ok(target) = self.deref_value(frames, base) {
                    // A sparse pointee's elements past its stored
                    // prefix are read through the ordinary path
                    // below (which reports them as the
                    // uninitialised slots they are); the sibling
                    // walk would materialise them one by one.
                    let sparse_target = matches!(
                        self.read_resolved(frames, &target),
                        Ok(Value::SparseVec { .. })
                    );
                    // A base denoting a value that is itself a
                    // sequence of exactly the fat length denotes
                    // that sequence directly (see
                    // `fat_base_is_the_sequence`): the sibling
                    // walk below would read the elements of the
                    // outer container the value sits in where
                    // the sequence elements belong.
                    let direct_target = !sparse_target
                        && self.fat_base_is_the_sequence(frames, &target, *len);
                    if !sparse_target
                        && !direct_target
                        && target_ends_at_element(&target)
                        && self.sibling_span_fits(frames, &target, *len)
                    {
                        let mut elems = Vec::with_capacity(*len as usize);
                        for i in 0..(*len as usize) {
                            elems.push(self.read_seq_at(frames, &target, i)?);
                        }
                        let elems = SliceElems::Values(Cow::Owned(elems));
                        let elems = match self.seq_snapshot_key(frames, value) {
                            Some((frame_pos, local, path)) => elems.with_rescue(
                                |elems| {
                                    self.rescue_elided_snapshot(
                                        &frames[frame_pos],
                                        local,
                                        &path,
                                        elems,
                                    )
                                },
                            ),
                            None => elems,
                        };
                        return Ok((elems, *len));
                    }
                    // A base ending in an element-entry suffix
                    // denotes one entered element of its
                    // container; the slice's elements are the
                    // container's elements from that index on,
                    // each entered the same way.
                    if let Some(elems) =
                        self.entered_sequence_elements(frames, &target, *len)?
                    {
                        return Ok((elems, *len));
                    }
                }
                let base = self.resolve_ref(frames, base)?;
                // A fat pointer of length 1 over a scalar
                // pointee (a `Box<T>` allocation reinterpreted
                // as a `[T; 1]` by `Box::into_boxed_slice`, a
                // one-element `from_raw_parts` over a scalar):
                // the scalar is the slice's one element (the
                // one-element view `unsized_tail_len` names).
                let elems = match &*base {
                    Value::Bool(_)
                    | Value::Int(_)
                    | Value::Uint(_)
                    | Value::Float(_)
                    | Value::Float128(_)
                        if *len == 1 =>
                    {
                        SliceElems::Values(Cow::Owned(vec![base.into_owned()]))
                    }
                    // A base that resolves to a fat pointer
                    // itself (a reborrow of a stored slice
                    // value -- the shape the `&self.inner`
                    // bodies, `as_encoded_bytes` above all,
                    // produce over a stored `&OsStr`): the
                    // outer pointer denotes the inner slice's
                    // elements, so read them through the inner
                    // pointer's own element reading, truncated
                    // to the outer length as the plain reading
                    // below truncates.
                    Value::FatRef { .. } => {
                        let (inner, _) = self.slice_elements(frames, base.as_ref())?;
                        SliceElems::Values(Cow::Owned(inner)).truncated(*len as usize)
                    }
                    _ => elements_of(base)?.truncated(*len as usize),
                };
                let elems = match self.seq_snapshot_key(frames, value) {
                    Some((frame_pos, local, path)) => elems.with_rescue(|elems| {
                        self.rescue_elided_snapshot(
                            &frames[frame_pos],
                            local,
                            &path,
                            elems,
                        )
                    }),
                    None => elems,
                };
                Ok((elems, *len))
            }
            _ => {
                let resolved = self.resolve_ref(frames, value)?;
                let elems = elements_of(resolved)?;
                let len = elems.len() as u128;
                let elems = match self.seq_snapshot_key(frames, value) {
                    Some((frame_pos, local, path)) => elems.with_rescue(|elems| {
                        self.rescue_elided_snapshot(
                            &frames[frame_pos],
                            local,
                            &path,
                            elems,
                        )
                    }),
                    None => elems,
                };
                Ok((elems, len))
            }
        }
    }

    /// `Index::index` with a range argument: the sub-slice of the
    /// receiver between the range's bounds, with core's own panic
    /// messages for out-of-range bounds. The result is a fat
    /// reference aliasing the receiver's storage when the
    /// receiver is backed by live storage (see
    /// [`range_view_base`]), and a fat reference over a
    /// materialised copy of the sub-slice's elements when it is
    /// not (reads through the copy are exact; it has no storage
    /// to alias) — or, when the range runs past a sparse
    /// receiver's stored elements, over the sparse form of the
    /// range (see `Value::SparseVec`). A `str` receiver yields
    /// the sub-string, with char-boundary checks as in core.
    /// Whether a target ending in an element-entry suffix
    /// denotes its container's elements entered by the suffix
    /// (the transparent-wrapper pointer cast reading), as
    /// opposed to a sequence the base denotes directly through
    /// a genuine field step — a slice over an element's array
    /// field ends in the same projection shape, and there the
    /// field step is the sequence's own storage. The cast
    /// reading is declared only when both judgements agree: the
    /// container's span from the suffix's index covers the fat
    /// length, and the target's own value is not itself a
    /// sequence of exactly the fat length (the direct reading's
    /// shape). Any shape this cannot judge answers no, keeping
    /// the plain reading.
    pub(super) fn entry_suffix_denotes_elements(
        &self,
        frames: &[Frame],
        target: &Resolved,
        fat_len: u128,
    ) -> bool {
        let projection = match target {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. }
            | Resolved::Value(_, projection) => projection,
        };
        let Some((container_proj, start, _)) = element_entry_suffix(projection) else {
            return false;
        };
        let container = resolved_with_projection(target, container_proj);
        let Ok(container) = self.read_resolved(frames, &container) else {
            return false;
        };
        let Some(count) = container.len() else {
            return false;
        };
        if start as u128 + fat_len > count {
            return false;
        }
        let Ok(value) = self.read_resolved(frames, target) else {
            return false;
        };
        // A target value that is itself a sequence of exactly
        // the fat length is the direct reading's shape.
        let target_is_the_sequence = matches!(
            value,
            Value::Tuple(_)
                | Value::Enum { .. }
                | Value::SparseVec { .. }
                | Value::Bytes(_)
                | Value::Str(_)
        ) && value.len() == Some(fat_len);
        !target_is_the_sequence
    }

    /// Whether a fat pointer of length `fat_len` whose base
    /// resolves to `target` denotes the span of elements from
    /// the element the base denotes on, rather than that
    /// element itself: the base ends at an element position --
    /// an index step last, or an element-entry suffix (wrapper
    /// field steps after the index step: the shape a
    /// transparent-wrapper subslice formation leaves, a `CStr`
    /// range index above all) that
    /// [`Self::entry_suffix_denotes_elements`] judges to
    /// denote the container's elements. A consumer that would
    /// resolve such a base to the entered element reads a span
    /// of any length as its first element alone, so consumers
    /// that judge a base by the shape it denotes consult this
    /// reading before following or claiming the base.
    pub(super) fn base_denotes_element_span(
        &self,
        frames: &[Frame],
        target: &Resolved,
        fat_len: u128,
    ) -> bool {
        target_ends_at_element(target)
            || self.entry_suffix_denotes_elements(frames, target, fat_len)
    }

    /// The elements a fat pointer denotes when its base target
    /// ends in an element-entry suffix (see
    /// [`element_entry_suffix`]): the elements of the container
    /// the base sits in, from the suffix's index on, each entered
    /// by the suffix's field steps — the reading the base's own
    /// value (the first entered element alone) cannot give.
    /// `Ok(None)` when the base does not have that shape, or its
    /// container is not a sequence the entry can be distributed
    /// over (the caller keeps its plain reading). The containers
    /// the entry distributes over are the aggregate forms and
    /// the byte-string forms.
    fn entered_sequence_elements(
        &self,
        frames: &[Frame],
        target: &Resolved,
        len: u128,
    ) -> InterpResult<Option<SliceElems<'static>>> {
        let projection = match target {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. }
            | Resolved::Value(_, projection) => projection,
        };
        let Some((container_proj, start, entry)) = element_entry_suffix(projection) else {
            return Ok(None);
        };
        if !self.entry_suffix_denotes_elements(frames, target, len) {
            return Ok(None);
        }
        let container = resolved_with_projection(target, container_proj);
        let Ok(value) = self.read_resolved(frames, &container) else {
            return Ok(None);
        };
        let want = usize::try_from(len).unwrap_or(usize::MAX);
        match value {
            Value::Tuple(items) => {
                if items.len() < start || items.len() - start < want {
                    return Ok(None);
                }
                let mut out = Vec::with_capacity(want);
                for item in &items[start..start + want] {
                    out.push(project_value(
                        item.clone(),
                        &entry,
                        "slice elements",
                        "fat pointer base",
                    )?);
                }
                Ok(Some(SliceElems::Values(Cow::Owned(out))))
            }
            Value::SparseVec { elems, .. } => {
                let mut stored = Vec::new();
                for elem in elems.iter().skip(start).take(want) {
                    stored.push(project_value(
                        elem.clone(),
                        &entry,
                        "slice elements",
                        "fat pointer base",
                    )?);
                }
                Ok(Some(SliceElems::Sparse {
                    stored: Cow::Owned(stored),
                    len: want,
                }))
            }
            // A byte-string container (the storage form a
            // boxed byte sequence takes): its elements are its
            // bytes, each entered by the suffix's field steps
            // as the aggregate containers' elements are.
            Value::Bytes(bytes) => {
                entered_byte_elements(&bytes, start, want, &entry)
            }
            Value::Str(text) => {
                entered_byte_elements(text.as_bytes(), start, want, &entry)
            }
            _ => Ok(None),
        }
    }

    pub(super) fn index_range(
        &self,
        frames: &[Frame],
        name: &str,
        receiver: &Value,
        index: &Value,
        idx_tag: &str,
    ) -> InterpResult<Value> {
        // The bounds as stored in the range value: `Range` and
        // `RangeInclusive` carry [start, end], `RangeFrom` [start],
        // `RangeTo` / `RangeToInclusive` [end], `RangeFull` none
        // (it arrives as the unit value).
        let fields: &[Value] = match index {
            Value::Unit => &[],
            Value::Enum { fields, .. } => fields,
            Value::Tuple(fields) => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "range index value",
                    other.kind_name(),
                ))
            }
        };
        let bound = |i: usize| -> InterpResult<usize> {
            match fields.get(i) {
                Some(v) => expect_index(v, name),
                None => Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "range index value with bounds",
                    index.kind_name(),
                )),
            }
        };
        let kind = if idx_tag.contains("RangeFull") {
            RangeKindForIndex::Full
        } else if idx_tag.contains("RangeToInclusive") {
            RangeKindForIndex::ToInclusive
        } else if idx_tag.contains("RangeInclusive") {
            RangeKindForIndex::Inclusive
        } else if idx_tag.contains("RangeFrom") {
            RangeKindForIndex::From
        } else if idx_tag.contains("RangeTo") {
            RangeKindForIndex::To
        } else {
            RangeKindForIndex::Range
        };
        // Resolve the receiver's length first: for a `str` the
        // length is in bytes, as range indices on strings are byte
        // indices.
        let resolved = self.resolve_value(frames, receiver)?;
        if let Value::Str(text) = &resolved {
            // A `str` range is judged by core's own
            // `slice_error_fail`, not the slice wording: take
            // the raw extent and let its branch order speak.
            let len = text.len();
            let (start, end) = range_extent(&kind, &bound, len)?;
            return match text.get(start..end) {
                Some(sub) => Ok(Value::Str(sub.to_string())),
                None => Err(InterpError::panic(slice_error_fail_message(
                    text, start, end,
                ))),
            };
        }
        let (elems, len) = self.slice_elements_ref(frames, receiver)?;
        let len = usize::try_from(len).unwrap_or(usize::MAX);
        let (start, end) = range_bounds(&kind, &bound, len)?;
        let sub_len = end - start;
        // A receiver backed by live storage yields a fat
        // reference aliasing that storage, formed exactly as
        // `index_range_mut` forms its result, so that
        // interior-mutability writes through the sub-slice land
        // in the source storage and later source writes are read
        // through the sub-slice. Only a receiver with no live
        // storage behind it keeps the materialised copy below.
        if let Some(base) =
            self.range_view_base(frames, receiver, start, sub_len, len, name)?
        {
            return Ok(Value::FatRef {
                base: Box::new(base),
                len: sub_len as u128,
            });
        }
        Ok(Value::FatRef {
            base: Box::new(elems.sub_snapshot(start, end)),
            len: sub_len as u128,
        })
    }

    /// The base for a shared range window over a receiver with
    /// live storage behind it: [`windowed_base`] over the
    /// receiver reference itself (a plain reference receiver) or
    /// over a fat receiver's base reference. `Ok(None)` when the
    /// receiver has no live storage to alias: a fat base that is
    /// not a reference, a reference to a constant aggregate or
    /// another target without storage a subslice step can
    /// window. Storage held in the sparse form aliases like
    /// any other live storage: the projectors' `Subslice` arms
    /// cover `Value::SparseVec`.
    pub(super) fn range_view_base(
        &self,
        frames: &[Frame],
        receiver: &Value,
        start: usize,
        sub_len: usize,
        receiver_len: usize,
        name: &str,
    ) -> InterpResult<Option<Value>> {
        let base = match receiver {
            Value::Ref(_) => receiver,
            Value::FatRef { base, .. } if matches!(&**base, Value::Ref(_)) => {
                base.as_ref()
            }
            _ => return Ok(None),
        };
        let live = matches!(
            base,
            Value::Ref(
                RefTarget::Place { .. }
                    | RefTarget::Static { .. }
                    | RefTarget::Alloc { .. }
            )
        );
        if !live {
            return Ok(None);
        }
        Ok(Some(self.windowed_base(
            frames,
            base,
            start,
            sub_len,
            receiver_len,
            name,
        )?))
    }

    /// `IndexMut::index_mut` with a range argument: the mutable
    /// sub-slice of the receiver between the range's bounds, with
    /// core's own panic messages for out-of-range bounds (via
    /// [`range_bounds`], as for the shared `Index`). The result is
    /// a fat reference aliasing the receiver's storage: its base
    /// is the receiver reference with a resolved
    /// `RefProj::Subslice` step appended (the step carries the
    /// range's offset, which is what makes a range starting above
    /// zero expressible), over the receiver reference itself for a
    /// place/static receiver and over the same base for a fat
    /// receiver. A fat receiver whose base is a materialised copy
    /// has no storage to alias and stays explicitly unsupported.
    /// The base reference for a range window over a slice
    /// receiver: normally the base with a subslice step
    /// appended (see `subslice_ref`). A base ending in an
    /// element-entry suffix that denotes its container's
    /// elements (see `entry_suffix_denotes_elements`) windows
    /// the container instead: the entry index advances by the
    /// window's start and no subslice step is appended — a
    /// subslice step after the entry steps would window the
    /// first entered element rather than the sequence.
    fn windowed_base(
        &self,
        frames: &[Frame],
        base: &Value,
        start: usize,
        sub_len: usize,
        receiver_len: usize,
        name: &str,
    ) -> InterpResult<Value> {
        if let Value::Ref(target_ref) = base {
            let projection = match target_ref {
                RefTarget::Place { projection, .. }
                | RefTarget::Static { projection, .. }
                | RefTarget::Alloc { projection, .. }
                | RefTarget::Value(_, projection) => Some(projection),
                _ => None,
            };
            if let Some(projection) = projection {
                if element_entry_suffix(projection).is_some() {
                if let Ok(target) = self.deref_value(frames, base) {
                    if self.entry_suffix_denotes_elements(
                        frames,
                        &target,
                        receiver_len as u128,
                    ) {
                        if let Some(projection) =
                            advance_entry_projection(projection, start)
                        {
                            return Ok(ref_with_projection(base, projection));
                        }
                    }
                }
            }
            }
        }
        subslice_ref(base, start, sub_len, name)
    }

    pub(super) fn index_range_mut(
        &self,
        frames: &[Frame],
        name: &str,
        receiver: &Value,
        index: &Value,
        idx_tag: &str,
    ) -> InterpResult<Value> {
        // The bounds as stored in the range value, in the same
        // shapes as `index_range`.
        let fields: &[Value] = match index {
            Value::Unit => &[],
            Value::Enum { fields, .. } => fields,
            Value::Tuple(fields) => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "range index value",
                    other.kind_name(),
                ))
            }
        };
        let bound = |i: usize| -> InterpResult<usize> {
            match fields.get(i) {
                Some(v) => expect_index(v, name),
                None => Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "range index value with bounds",
                    index.kind_name(),
                )),
            }
        };
        let kind = if idx_tag.contains("RangeFull") {
            RangeKindForIndex::Full
        } else if idx_tag.contains("RangeToInclusive") {
            RangeKindForIndex::ToInclusive
        } else if idx_tag.contains("RangeInclusive") {
            RangeKindForIndex::Inclusive
        } else if idx_tag.contains("RangeFrom") {
            RangeKindForIndex::From
        } else if idx_tag.contains("RangeTo") {
            RangeKindForIndex::To
        } else {
            RangeKindForIndex::Range
        };
        // The receiver's effective length: a fat receiver's
        // carried length, otherwise the pointed-to value's length.
        let len: usize = match receiver {
            Value::FatRef { len, .. } => usize::try_from(*len).unwrap_or(usize::MAX),
            Value::Ref(_) => {
                let resolved = self.resolve_value(frames, receiver)?;
                if matches!(resolved, Value::Str(_)) {
                    return Err(unsupported(
                        "IndexMut::index_mut with a range on a str is not runtime-supported"
                            .to_string(),
                    ));
                }
                let (_, len) = self.slice_elements(frames, receiver)?;
                usize::try_from(len).unwrap_or(usize::MAX)
            }
            other => {
                return Err(InterpError::type_mismatch(
                    "IndexMut::index_mut",
                    "mutable reference",
                    other.kind_name(),
                ))
            }
        };
        let (start, end) = range_bounds(&kind, &bound, len)?;
        let sub_len = end - start;
        match receiver {
            Value::FatRef { base, .. } => match &**base {
                Value::Ref(_) => {
                    let base = self.windowed_base(frames, base, start, sub_len, len, name)?;
                    Ok(Value::FatRef {
                        base: Box::new(base),
                        len: sub_len as u128,
                    })
                }
                other => Err(unsupported(format!(
                    "IndexMut::index_mut with a range over a {} base is not runtime-supported (a materialised copy has no storage to alias)",
                    other.kind_name()
                ))),
            },
            Value::Ref(_) => {
                let base = self.windowed_base(frames, receiver, start, sub_len, len, name)?;
                Ok(Value::FatRef {
                    base: Box::new(base),
                    len: sub_len as u128,
                })
            }
            _ => unreachable!("receiver shapes checked above"),
        }
    }

    /// The bytes of the string window a receiver denotes, when
    /// the receiver (behind any number of plain reference
    /// layers) is a fat pointer: the reading
    /// [`Self::byte_slice_of_fat_ref`] gives, whose length is
    /// the fat pointer's own, so a window over part of a larger
    /// byte storage yields the window's bytes, not the
    /// storage's. `None` for any other receiver shape, and for
    /// a window whose elements are not all bytes.
    fn string_window_bytes(
        &self,
        frames: &[Frame],
        arg: &Value,
    ) -> InterpResult<Option<Vec<u8>>> {
        let mut receiver = self.read_ref_once(frames, arg)?;
        while matches!(receiver, Value::Ref(_)) {
            let next = self.read_ref_once(frames, &receiver)?;
            if next == receiver {
                break;
            }
            receiver = next;
        }
        Ok(match receiver {
            Value::FatRef { .. } => self.byte_slice_of_fat_ref(frames, &receiver),
            _ => None,
        })
    }

    /// `AsRef::as_ref` / `AsMut::as_mut`: the receiver viewed as
    /// the target type, computed from the modelled data (see the
    /// `Builtin::AsRef` note). `generic_args` are the trait's
    /// `[Self, Target]` type arguments. The blanket impls for
    /// `&T` / `&mut T` forward to `T`'s impl, so leading
    /// references are stripped from the self type and the value
    /// resolution below follows them. For the mutable form the
    /// receiver place itself is the viewed storage for the
    /// modelled containers, so the argument reference is returned
    /// unchanged; only `Box` needs its allocation pointer.
    pub(super) fn as_ref_view(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        arg: &Value,
        mutable: bool,
    ) -> InterpResult<Value> {
        let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
        let target_ty = generic_args.get(1).map(String::as_str).unwrap_or("");
        let mut base_ty = self_ty;
        while let Some(rest) = base_ty
            .strip_prefix("&mut ")
            .or_else(|| base_ty.strip_prefix('&'))
        {
            base_ty = rest.trim();
        }
        let what = if mutable { "AsMut::as_mut" } else { "AsRef::as_ref" };
        if mutable {
            match base_ty {
                t if is_string_like_ty(t)
                    || t.starts_with("std::vec::Vec<")
                    || t.starts_with("alloc::vec::Vec<")
                    || t.starts_with('[') =>
                {
                    return Ok(arg.clone());
                }
                t if t.starts_with("std::boxed::Box<") => {
                    let pointee = self.resolve_value(frames, arg)?;
                    if let Value::DynRef { .. } = pointee {
                        return Ok(pointee);
                    }
                    // The box's stored pointer: the single field
                    // of the aggregate a builtin builds, or the
                    // bottom of the field-0 wrapper chain
                    // (`Box` / `Unique` / `NonNull`) a guest body
                    // builds. A fat bottom is the slice view
                    // itself; an allocation pointer is returned
                    // as it stands.
                    let inner = match pointee {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            fields.into_iter().next().expect("len checked")
                        }
                        ref value => match layered_wrapper_pointer(value) {
                            Some(ptr) => ptr,
                            None => {
                                return Err(InterpError::type_mismatch(
                                    what,
                                    "boxed value",
                                    value.kind_name(),
                                ));
                            }
                        },
                    };
                    return Ok(match inner {
                        Value::Ref(RefTarget::Alloc { .. }) => inner,
                        Value::FatRef { .. } => inner,
                        other => Value::Ref(RefTarget::value(other)),
                    });
                }
                _ => {
                    return Err(unsupported(format!(
                        "{what} for {self_ty} is not runtime-supported"
                    )));
                }
            }
        }
        let value = self.resolve_value(frames, arg)?;
        // A string view over a window of byte storage — the
        // `char::encode_utf8` result is the shape that reaches
        // here: a fat pointer whose base denotes one element of
        // a byte array and whose length is the encoded length —
        // does not survive the resolution above: following the
        // fat pointer reads the denoted element alone, so the
        // value that arrives is a byte scalar or a one-element
        // tuple, never the window's bytes. Where the resolved
        // form is none of the shapes the string view denominates
        // and the receiver is such a fat pointer, denominate
        // the window's bytes instead, read through the fat
        // pointer itself (its length is the view's length, the
        // reading `os_str_bytes` gives an `&OsStr` argument).
        // A window whose bytes cannot be read that way — a
        // view longer than the backing storage included — keeps
        // the resolved form and the verdict it already produced.
        let window = if is_string_like_ty(base_ty)
            && (target_ty == "[u8]"
                || target_ty == "str"
                || target_ty == "std::ffi::OsStr"
                || target_ty == "std::path::Path"
                || target_ty == base_ty)
            && !matches!(
                value,
                Value::Str(_) | Value::Bytes(_) | Value::Enum { .. }
            ) {
            self.string_window_bytes(frames, arg)?
        } else {
            None
        };
        let view = match window {
            Some(bytes) => {
                let denominated = match String::from_utf8(bytes) {
                    Ok(text) => Value::Str(text),
                    Err(err) => Value::Bytes(err.into_bytes()),
                };
                self.view_value_as(frames, denominated, base_ty, target_ty, what)?
            }
            None => self.view_value_as(frames, value, base_ty, target_ty, what)?,
        };
        // `as_ref` returns a reference: a view that came out as the
        // bare pointee data (an identity view of an aggregate —
        // `Box<T> -> &T`, `Vec<T> -> &Vec<T>`, a self view) must be
        // re-wrapped as a reference to that data, or storing the
        // result fails the destination's `Ref` type check. The fat
        // and string views are already reference-shaped.
        Ok(match view {
            Value::Ref(_)
            | Value::FatRef { .. }
            | Value::DynRef { .. }
            | Value::Str(_)
            | Value::Bytes(_)
            | Value::FnPtr(_)
            | Value::Closure { .. } => view,
            other => Value::Ref(RefTarget::value(other)),
        })
    }

    /// The bytes an owned string-like value denotes in the
    /// forms the pure shape reading in `view_value_as` cannot
    /// reach, read by the accessor for its type: `String`
    /// through `string_from_real_aggregate` (the buffer behind
    /// a reference) or `string_from_element_form` (the element
    /// form whose elements keep the `MaybeUninit` union form
    /// -- the readings `resolve_string` composes), `OsString`
    /// through `os_string_bytes_any`, `PathBuf` through
    /// `path_buf_bytes_any`. Any other type, or a value in any
    /// other form, returns `None` and leaves the caller's
    /// ordinary judgement to stand.
    fn real_form_string_bytes(
        &self,
        frames: &[Frame],
        base_ty: &str,
        value: &Value,
    ) -> InterpResult<Option<Vec<u8>>> {
        match base_ty {
            "std::string::String" | "alloc::string::String" => {
                if let Some(text) =
                    self.string_from_real_aggregate(frames, value)?
                {
                    return Ok(Some(text.into_bytes()));
                }
                if let Value::Enum {
                    variant: 0,
                    fields,
                } = value
                {
                    if fields.len() == 1 {
                        if let Some(text) =
                            self.string_from_element_form(&fields[0])
                        {
                            return Ok(Some(text.into_bytes()));
                        }
                    }
                }
                Ok(None)
            }
            "std::ffi::OsString" | "std::ffi::OsStr" => {
                self.os_string_bytes_any(frames, value)
            }
            "std::path::PathBuf" | "std::path::Path" => {
                self.path_buf_bytes_any(frames, value)
            }
            _ => Ok(None),
        }
    }

    /// The value-level view behind `as_ref_view`: `value` is the
    /// receiver's data (references already followed), `base_ty`
    /// its type, `target_ty` the view's type.
    fn view_value_as(
        &self,
        frames: &[Frame],
        value: Value,
        base_ty: &str,
        target_ty: &str,
        what: &str,
    ) -> InterpResult<Value> {
        // A value still wrapped in a reference (a blanket
        // `&T` layer the type stripping already accounted for):
        // follow it and view the pointee.
        if matches!(
            value,
            Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }
        ) {
            let inner = self.resolve_value(frames, &value)?;
            if inner != value {
                return self.view_value_as(frames, inner, base_ty, target_ty, what);
            }
        }
        if is_string_like_ty(base_ty) {
            if target_ty == "[u8]" {
                return self.string_bytes_ref(frames, &value);
            }
            if target_ty == "str"
                || target_ty == "std::ffi::OsStr"
                || target_ty == "std::path::Path"
                || target_ty == base_ty
            {
                return match value {
                    Value::Str(_) | Value::Bytes(_) => Ok(value),
                    Value::Enum { .. } => {
                        // An owned string-like value in aggregate
                        // form (a `PathBuf` around an `OsString`
                        // around the buffer around the byte
                        // vector, the shape `pathbuf_value`
                        // builds): the view is the byte string at
                        // the bottom of the wrapper chain. The
                        // pure shape reading denominates the
                        // modelled forms; the forms it cannot
                        // reach -- the real aggregate form, whose
                        // buffer sits behind a reference, and a
                        // `String` element form whose elements
                        // keep the `MaybeUninit` union form --
                        // go through the readers that follow
                        // those shapes, as the byte accessors
                        // and the equality comparisons already do.
                        let bytes = match string_aggregate_bytes(&value) {
                            Some(bytes) => Some(bytes),
                            None => {
                                self.real_form_string_bytes(frames, base_ty, &value)?
                            }
                        };
                        match bytes {
                            Some(bytes) => {
                                // A `String`'s view is the string
                                // itself: the consumers of an
                                // `&str` read string data, and a
                                // byte-string view of a `String`
                                // is refused as a string argument
                                // downstream. The OS-string and
                                // path views stay byte strings.
                                if matches!(
                                    base_ty,
                                    "std::string::String" | "alloc::string::String"
                                ) {
                                    if let Ok(text) =
                                        String::from_utf8(bytes.clone())
                                    {
                                        return Ok(Value::Str(text));
                                    }
                                }
                                Ok(Value::Bytes(bytes))
                            }
                            None => Err(InterpError::type_mismatch(
                                what,
                                "string data",
                                value.kind_name(),
                            )),
                        }
                    }
                    other => Err(InterpError::type_mismatch(
                        what,
                        "string data",
                        other.kind_name(),
                    )),
                };
            }
        }
        if base_ty.starts_with("std::vec::Vec<")
            || base_ty.starts_with("alloc::vec::Vec<")
        {
            if target_ty.starts_with('[') {
                // The slice view of a vector: the fat pointer
                // denotes the vector's element sequence, in the
                // shape every slice consumer reads (the element
                // aggregate `boxed_slice_value` builds). A vector
                // in its real aggregate form denotes its elements
                // through its buffer pointer, which the fat
                // pointer then carries directly; a modelled
                // vector is its element list, which becomes the
                // element tuple. Wrapping the vector aggregate
                // itself would leave the fat pointer's base a
                // bare ADT value, which no dereference can follow
                // (the failure `Vec::as_ref` hit on element
                // access, where `len` and iteration still worked
                // because they read the fat pointer itself).
                if let Some((_, real_len)) = vec_real_aggregate_parts(&value) {
                    let ptr = Self::real_vec_buffer_ptr(&value).ok_or_else(|| {
                        unsupported(format!(
                            "{what} for {base_ty}: the vector's buffer pointer is not a reference value"
                        ))
                    })?;
                    return Ok(Value::FatRef {
                        base: Box::new(ptr),
                        len: real_len as u128,
                    });
                }
                let len = value.len().ok_or_else(|| {
                    InterpError::type_mismatch(
                        what,
                        "array value",
                        value.kind_name(),
                    )
                })?;
                let base = match value {
                    Value::Enum { fields, .. } => Value::Tuple(fields),
                    other => other,
                };
                return Ok(Value::FatRef {
                    base: Box::new(base),
                    len,
                });
            }
            if target_ty.starts_with("std::vec::Vec<")
                || target_ty.starts_with("alloc::vec::Vec<")
            {
                return Ok(value);
            }
        }
        if base_ty.starts_with('[') && target_ty.starts_with('[') {
            if !is_slice_tag(base_ty) && is_slice_tag(target_ty) {
                // An array viewed as a slice.
                let len = value.len().ok_or_else(|| {
                    InterpError::type_mismatch(
                        what,
                        "array value",
                        value.kind_name(),
                    )
                })?;
                return Ok(Value::FatRef {
                    base: Box::new(value),
                    len,
                });
            }
            return Ok(value);
        }
        if base_ty.starts_with("std::boxed::Box<") {
            if let Value::DynRef { .. } = value {
                return Ok(value);
            }
            let inner_ty = first_generic_param(base_ty).ok_or_else(|| {
                unsupported(format!("{what} for {base_ty}: no inner type"))
            })?;
            // The box's stored pointer: the single field of
            // the aggregate a builtin builds, or the bottom of
            // the field-0 wrapper chain (`Box` / `Unique` /
            // `NonNull`) a guest body builds — the shape
            // `Box::into_boxed_slice`'s body leaves behind.
            let field = match value {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    fields.into_iter().next().expect("len checked")
                }
                ref value => match layered_wrapper_pointer(value) {
                    Some(ptr) => ptr,
                    None => {
                        return Err(InterpError::type_mismatch(
                            what,
                            "boxed value",
                            value.kind_name(),
                        ));
                    }
                },
            };
            if target_ty == inner_ty {
                // The view is the pointee itself: the result is
                // a reference to the box's storage, not the
                // pointee value. The box field already is that
                // reference (the allocation pointer, or the fat
                // pointer that is the slice view itself); a
                // non-reference field is resolved and re-wrapped
                // as a value reference.
                return Ok(match field {
                    Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => {
                        field
                    }
                    other => {
                        let pointee = self.resolve_value(frames, &other)?;
                        Value::Ref(RefTarget::value(pointee))
                    }
                });
            }
            let pointee = self.resolve_value(frames, &field)?;
            return self.view_value_as(frames, pointee, &inner_ty, target_ty, what);
        }
        if base_ty.starts_with("std::rc::Rc<")
            || base_ty.starts_with("std::sync::Arc<")
        {
            if let Value::DynRef { .. } = value {
                return Ok(value);
            }
            let inner_ty = first_generic_param(base_ty).ok_or_else(|| {
                unsupported(format!("{what} for {base_ty}: no inner type"))
            })?;
            return match rc_handle_id(&value) {
                Some(id) => {
                    let pointee = self.read_resolved(
                        frames,
                        &Resolved::Alloc {
                            id,
                            projection: vec![RefProj::Field(2)],
                        },
                    )?;
                    self.view_value_as(frames, pointee, &inner_ty, target_ty, what)
                }
                None => Err(InterpError::type_mismatch(
                    what,
                    "a reference-counted handle",
                    value.kind_name(),
                )),
            };
        }
        if base_ty.starts_with("std::borrow::Cow<") {
            let inner_ty = last_generic_param(base_ty).ok_or_else(|| {
                unsupported(format!("{what} for {base_ty}: no inner type"))
            })?;
            return match value {
                Value::Enum { fields, .. } if !fields.is_empty() => {
                    let data = fields.into_iter().next().expect("checked non-empty");
                    self.view_value_as(frames, data, &inner_ty, target_ty, what)
                }
                other => Err(InterpError::type_mismatch(
                    what,
                    "a Cow value",
                    other.kind_name(),
                )),
            };
        }
        if base_ty == target_ty {
            return Ok(value);
        }
        Err(unsupported(format!(
            "{what} for {base_ty} -> {target_ty} is not runtime-supported"
        )))
    }

    /// The reference formed by appending `steps` to the place
    /// a reference value denotes, when it denotes a live place:
    /// the multi-step form of [`Self::place_projected_ref`].
    pub(super) fn place_projected_ref_steps(
        &self,
        frames: &[Frame],
        arg: &Value,
        steps: &[RefProj],
    ) -> InterpResult<Option<Value>> {
        Ok(match self.deref_value(frames, arg)? {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.extend(steps.iter().cloned());
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
                projection.extend(steps.iter().cloned());
                Some(Value::Ref(RefTarget::Static { name, projection }))
            }
            Resolved::Alloc {
                id,
                mut projection,
            } => {
                projection.extend(steps.iter().cloned());
                Some(Value::Ref(RefTarget::Alloc { id, projection }))
            }
            Resolved::Value(..) => None,
        })
    }

    /// The element form of an array value held in the
    /// model's scalar-backed form (see
    /// [`Self::smallvec_slice`]): a tuple whose first slot is
    /// one scalar carrying the initialised region's bytes and
    /// whose remaining slots are uninitialised. Only scalar
    /// element types pack this way, and only the first `len`
    /// elements are initialised, so the expansion rebuilds
    /// exactly those from the scalar's little-endian storage
    /// bytes and leaves the spare slots uninitialised. Any
    /// other shape is returned unchanged.
    pub(super) fn unpack_scalar_backed_array(
        &self,
        array: &Value,
        elem_tag: &str,
        count: usize,
        len: usize,
    ) -> Value {
        let Value::Tuple(items) = array else {
            return array.clone();
        };
        if items.len() != count || len < 2 || len > count {
            return array.clone();
        }
        if items[1..].iter().any(|item| *item != Value::Uninit) {
            return array.clone();
        }
        let scalar = &items[0];
        if crate::interp::scalar_storage_len(scalar).is_none() {
            return array.clone();
        }
        let Some((elem_size, _)) = self.layout_of(elem_tag) else {
            return array.clone();
        };
        let elem_size = elem_size as usize;
        if elem_size == 0 || len * elem_size > 16 {
            return array.clone();
        }
        let mut out: Vec<Value> = Vec::with_capacity(count);
        for i in 0..len {
            let mut bytes = [0u8; 16];
            for (k, byte) in bytes.iter_mut().take(elem_size).enumerate() {
                match crate::interp::scalar_storage_byte(scalar, i * elem_size + k) {
                    Some(b) => *byte = b,
                    None => return array.clone(),
                }
            }
            match scalar_elem_value(elem_tag, &bytes[..elem_size]) {
                Some(v) => out.push(v),
                None => return array.clone(),
            }
        }
        out.resize(count, Value::Uninit);
        Value::Tuple(out)
    }
}

/// The entered elements of a byte-string container for
/// [`Interpreter::entered_sequence_elements`]: the container's
/// bytes from `start` on, `want` of them, each entered by the
/// suffix's field steps as the aggregate containers' elements
/// are. `Ok(None)` when the container's span from `start` does
/// not cover `want` elements.
fn entered_byte_elements(
    bytes: &[u8],
    start: usize,
    want: usize,
    entry: &[RefProj],
) -> InterpResult<Option<SliceElems<'static>>> {
    if bytes.len() < start || bytes.len() - start < want {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(want);
    for byte in &bytes[start..start + want] {
        out.push(project_value(
            Value::Uint(u128::from(*byte)),
            entry,
            "slice elements",
            "fat pointer base",
        )?);
    }
    Ok(Some(SliceElems::Values(Cow::Owned(out))))
}

/// The elements a slice-like value denotes, borrowed where the
/// element storage is already materialised: aggregate elements
/// lend their storage, while byte/string elements are the bytes
/// themselves (each byte is one element, exactly what
/// [`value_elements`] materialises as an integer).
pub(crate) enum SliceElems<'a> {
    Values(Cow<'a, [Value]>),
    Bytes(Cow<'a, [u8]>),
    /// The stored prefix of a sparse sequence (see
    /// `Value::SparseVec`) together with its declared length:
    /// indices past the prefix read as the uninitialised slot.
    Sparse { stored: Cow<'a, [Value]>, len: usize },
}

impl<'a> SliceElems<'a> {
    /// The number of elements.
    pub(crate) fn len(&self) -> usize {
        match self {
            SliceElems::Values(v) => v.len(),
            SliceElems::Bytes(b) => b.len(),
            SliceElems::Sparse { len, .. } => *len,
        }
    }

    /// The number of elements actually stored: a sparse
    /// sequence's declared length runs past its stored prefix
    /// (see `Value::SparseVec`); every other form stores them
    /// all.
    pub(crate) fn stored_len(&self) -> usize {
        match self {
            SliceElems::Values(v) => v.len(),
            SliceElems::Bytes(b) => b.len(),
            SliceElems::Sparse { stored, .. } => stored.len(),
        }
    }

    /// The element at `idx`, cloned out of the element storage (a
    /// byte element is constructed as an integer).
    pub(crate) fn at(&self, idx: usize) -> Value {
        match self {
            SliceElems::Values(v) => v[idx].clone(),
            SliceElems::Bytes(b) => Value::Uint(b[idx] as u128),
            SliceElems::Sparse { stored, .. } => stored
                .get(idx)
                .cloned()
                .unwrap_or(Value::Uninit),
        }
    }

    /// The element at `idx`, borrowed when it is stored element
    /// storage (a byte element is a constructed temporary).
    pub(crate) fn get(&self, idx: usize) -> Cow<'_, Value> {
        match self {
            SliceElems::Values(v) => Cow::Borrowed(&v[idx]),
            SliceElems::Bytes(b) => Cow::Owned(Value::Uint(b[idx] as u128)),
            SliceElems::Sparse { stored, .. } => match stored.get(idx) {
                Some(v) => Cow::Borrowed(v),
                None => Cow::Owned(Value::Uninit),
            },
        }
    }

    /// Truncate to at most `len` elements (a fat reference's
    /// carried length applied to its base's full element list, as
    /// `Vec::truncate` applies it in `slice_elements`).
    fn truncated(self, len: usize) -> Self {
        match self {
            SliceElems::Values(Cow::Borrowed(s)) => {
                SliceElems::Values(Cow::Borrowed(&s[..len.min(s.len())]))
            }
            SliceElems::Values(Cow::Owned(mut v)) => {
                v.truncate(len);
                SliceElems::Values(Cow::Owned(v))
            }
            SliceElems::Bytes(Cow::Borrowed(b)) => {
                SliceElems::Bytes(Cow::Borrowed(&b[..len.min(b.len())]))
            }
            SliceElems::Bytes(Cow::Owned(mut v)) => {
                v.truncate(len);
                SliceElems::Bytes(Cow::Owned(v))
            }
            SliceElems::Sparse { stored, len: declared } => SliceElems::Sparse {
                stored: match stored {
                    Cow::Borrowed(s) => {
                        Cow::Borrowed(&s[..len.min(s.len())])
                    }
                    Cow::Owned(mut v) => {
                        v.truncate(len);
                        Cow::Owned(v)
                    }
                },
                len: declared.min(len),
            },
        }
    }

    /// These elements after applying an element rewrite (the
    /// elided-store rescue, see
    /// `Interpreter::rescue_elided_snapshot`) to the element
    /// storage: borrowed storage is materialised so the
    /// rewrite has somewhere to land. Byte elements carry no
    /// uninitialised slots and pass through. A sparse
    /// sequence's stored prefix is rewritten; its virtual tail
    /// stays the uninitialised slots it denotes until a
    /// consumer materialises it (`slice_elements` rewrites the
    /// materialised tail the same way).
    fn with_rescue(self, rescue: impl Fn(&mut [Value])) -> Self {
        match self {
            SliceElems::Values(Cow::Borrowed(items)) => {
                let mut items = items.to_vec();
                rescue(&mut items);
                SliceElems::Values(Cow::Owned(items))
            }
            SliceElems::Values(Cow::Owned(mut items)) => {
                rescue(&mut items);
                SliceElems::Values(Cow::Owned(items))
            }
            SliceElems::Sparse { stored, len } => SliceElems::Sparse {
                stored: match stored {
                    Cow::Borrowed(items) => {
                        let mut items = items.to_vec();
                        rescue(&mut items);
                        Cow::Owned(items)
                    }
                    Cow::Owned(mut items) => {
                        rescue(&mut items);
                        Cow::Owned(items)
                    }
                },
                len,
            },
            bytes => bytes,
        }
    }

    /// The elements as an owned vector (what `slice_elements`
    /// returns): byte elements materialise as integers.
    pub(crate) fn into_vec(self) -> Vec<Value> {
        match self {
            SliceElems::Values(v) => v.into_owned(),
            SliceElems::Bytes(b) => {
                b.iter().map(|byte| Value::Uint(*byte as u128)).collect()
            }
            // Only reached at or under the materialisation
            // limit (see the guard in `slice_elements`): the
            // tail materialises as the uninitialised slots it
            // denotes.
            SliceElems::Sparse { stored, len } => {
                let mut v = stored.into_owned();
                v.resize(len, Value::Uninit);
                v
            }
        }
    }

    /// The snapshot value a range index over these elements
    /// denotes (what `index_range` stores as the sub-slice's
    /// base): the materialised elements of the range — or, when
    /// the range runs past a sparse sequence's stored elements,
    /// the sparse form over the stored part of the range, its
    /// remaining slots uninitialised (see `Value::SparseVec`).
    pub(crate) fn sub_snapshot(&self, start: usize, end: usize) -> Value {
        match self {
            SliceElems::Sparse { stored, .. } if end > stored.len() => {
                // The snapshot is a fresh value, not the
                // vector itself: it accounts exactly the
                // range it denotes, no capacity beyond it.
                let elems = stored[start.min(stored.len())..].to_vec();
                let len = (end - start) as u128;
                let cap = len.max(elems.len() as u128);
                Value::SparseVec { elems, len, cap }
            }
            _ => Value::Tuple((start..end).map(|i| self.at(i)).collect()),
        }
    }
}

// `io::Error` values are built only in the bitpacked `Repr`
// layout (`io_error_value` / `io_os_error_value` /
// `io_simple_error_value` above): that is the representation
// the guest's own converted `Error::kind` / `raw_os_error`
// bodies read, so every producer - the fs layer here and the
// process model alike - goes through those constructors.
