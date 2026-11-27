//! Place resolution and projection stepping.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};

impl<'a> Interpreter<'a> {
    pub(crate) fn resolve_place(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> InterpResult<Resolved> {
        self.resolve_place_mode(frames, body, place, true)
    }




    pub(crate) fn resolve_place_mode(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
        deref_elem0: bool,
    ) -> InterpResult<Resolved> {
        let mut cur = Resolved::Place {
            frame_pos: frames.len() - 1,
            local: place.local.0,
            projection: Vec::new(),
        };
        // The declared type of the position `cur` has reached,
        // tracked alongside the walk for the steps the IR type
        // alone types (tuple fields, array elements, reference
        // pointees). An ADT field step loses it: the interpreter
        // keeps no field-type table for ADT instances, so the
        // type is unknown from there on. The dereference arm
        // consults it for the `MaybeUninit` entry decision (see
        // `peel_maybe_uninit_entry`).
        let mut cur_ty: Option<Ty> = body.local_ty(place.local).cloned();
        // The length a `Deref` step through a fat pointer carries:
        // dereferencing a sub-slice reference reaches the base
        // place, whose own length is longer, so the fat length is
        // kept alongside and applied to the immediately following
        // index step (the only step it can bound).
        let mut fat_cap: Option<u128> = None;
        // Whether the fat pointer's elements are string carriers
        // (see `fat_slice_elem_is_string_carrier`), read from the
        // place's declared type at the `Deref` step and consumed
        // by the same following index step as `fat_cap`.
        let mut fat_elem_string: Option<bool> = None;
        // The fat pointer's element type, read from the place's
        // declared type at the `Deref` step like
        // `fat_elem_string`: it judges whether an index step
        // over the dereference denotes the base storage itself
        // (a scalar-backed slice's one element; see the `Index`
        // arm).
        let mut fat_elem_ty: Option<Ty> = None;
        // The declared type of the value `cur` currently denotes,
        // tracked from the place's local type through reference
        // steps (field and index steps end the tracking: the
        // program carries no field-type table to continue it).
        // Its one consumer is the `Field` arm's `Vec` length
        // slot.
        let mut cur_ty: Option<Ty> = body.local_ty(place.local).cloned();
        // A walk inside a `Vec`'s buffer wrapper chain (the
        // `Field(0)` steps an inlined `Vec::as_mut_ptr` takes
        // through `RawVec` / `Unique` / `NonNull`): none of the
        // wrappers is storage in the value model, so the steps
        // do not move `cur`; the walk's end (or a `Deref` step)
        // completes the chain as the buffer pointer itself (see
        // `buf_chain_pointer`). `buf_base` is the vector place's
        // target the chain started from.
        let mut buf_base: Option<Resolved> = None;
        let mut buf_depth = 0usize;
        // A suspected `Vec` position the type tracking could
        // not name: a `Field` step out of a named transparent
        // wrapper (`ManuallyDrop<Vec>` above all) whose field
        // type the field-type table cannot record (a generic
        // parameter at its head) leaves the walk on the
        // wrapper's payload with no declared type, and the
        // payload may be a vector whose structural fields the
        // following steps name. The suspicion is armed by that
        // step and judged by the next field step against the
        // payload's own shape (see the `Field` arm).
        let mut buf_suspect = false;
        // A `Field(0)` step that landed on a byte-vector-shaped
        // value (an `Enum` of byte elements: the stored form of
        // a `Vec<u8>` the type tracking could not reach, e.g.
        // the buffer inside a `PathBuf`'s wrapper chain) whose
        // meaning depends on the next step: a further `Field(0)`
        // enters the buffer wrapper chain with this step as the
        // vector's buffer field, a `Field(1)` reads its length
        // slot, and anything else appends the step as the plain
        // field access it is. Holds the target from before the
        // step; the step itself is not appended yet.
        let mut pending_vec: Option<Resolved> = None;
        // The pending detection above serves one descent: a
        // path buffer reached through its wrapper chain
        // (`PathBuf` / `Path` / `OsString` / `OsStr` locals,
        // whose buffer the declared types cannot follow past
        // the first field step). It is gated on the place's
        // root type: a byte-vector-shaped value is also a
        // legitimate intermediate of unrelated place walks
        // (a hash table's control-byte buffer, a byte block's
        // storage), and hijacking those walks into a buffer
        // chain misresolves them.
        let root_is_path_chain = {
            // The root type, through any reference layers (the
            // walk may start from a `&PathBuf` / `&OsString`
            // local and deref into the chain).
            let mut root_ty = body.local_ty(place.local);
            while let Some(Ty::Ref(inner)) = root_ty {
                root_ty = Some(inner);
            }
            matches!(
                root_ty,
                Some(Ty::Adt(name))
                    if name.contains("::path::Path")
                        || name.contains("::ffi::OsStr")
            )
        };

        // The variant the walk is inside of, set by `Downcast`
        // steps and consumed by the field-type tracking (see
        // `adt_field_ty`).
        let mut cur_variant: usize = 0;
        // A fat dereference whose base is an element position of
        // the sequence it spans exactly (see the `Deref` arm):
        // the container position the following field steps
        // re-anchor to, with the storage descent depth they then
        // consume without moving (the base already descended
        // those wrapper levels on the storage side).
        let mut fat_anchor: Option<(Resolved, usize)> = None;
        let mut fat_skip: usize = 0;
        let projection_len = place.projection.len();
        for (step_idx, elem) in place.projection.iter().enumerate() {
            let is_last = step_idx + 1 == projection_len;
            // A pending byte-vector step (see the state above)
            // is decided by the step that follows it; a
            // non-field step makes it the plain field access it
            // is, appended before this step is processed.
            if !matches!(elem, ProjectionElem::Field(_)) {
                buf_suspect = false;
            }
            if pending_vec.is_some()
                && !matches!(elem, ProjectionElem::Field(_))
            {
                let pending = pending_vec.take().expect("checked");
                cur = self.project_step(
                    frames,
                    pending,
                    RefProj::Field(0),
                    body,
                    place,
                )?;
            }
            cur = match elem {
                ProjectionElem::Field(idx) => {
                    // A field step after a fat dereference whose
                    // base is a spanned sequence's element
                    // position (see the `Deref` arm): the field
                    // addresses the wrapper over the sequence, so
                    // the walk re-anchors to the sequence value's
                    // container position, and this and the
                    // following wrapper steps — the levels the
                    // base already descended on the storage side —
                    // are consumed without moving the position.
                    if let Some((container, depth)) = fat_anchor.take() {
                        cur = container;
                        fat_skip = depth;
                    }
                    // The suspicion a previous wrapper step
                    // armed (see the state above) is judged by
                    // this step; it does not survive past it
                    // unless this step re-arms it.
                    let suspect = std::mem::take(&mut buf_suspect);
                    fat_cap = None;
                    fat_elem_string = None;
                    fat_elem_ty = None;
                    // The length slot of a `Vec` (field 1) read
                    // directly, the shape an inlined `Vec::len`
                    // inside a collected std body takes
                    // (`dedup_by`, `pop`, ...): the value model
                    // holds a `Vec` as its element list, so the
                    // slot is not storage in the value — the read
                    // is the element count itself, answered as a
                    // value target.
                    if fat_skip > 0 {
                        // A wrapper level the fat base already
                        // descended (see the `Deref` arm): the
                        // position does not move, and the type
                        // tracking still advances past the field.
                        fat_skip -= 1;
                        cur_ty = match &cur_ty {
                            Some(Ty::Tuple(ts)) => ts.get(*idx as usize).cloned(),
                            Some(Ty::Array(elem, _)) => Some((**elem).clone()),
                            Some(ty) => self.adt_field_ty(ty, cur_variant, *idx),
                            _ => None,
                        };
                        cur_variant = 0;
                        cur
                    } else if let Some(pending) = pending_vec.take() {
                        if *idx == 0 {
                            // The pending step was the vector's
                            // buffer field; this step is a
                            // wrapper step of its chain.
                            let base = self.project_step(
                                frames,
                                pending,
                                RefProj::Field(0),
                                body,
                                place,
                            )?;
                            buf_base = Some(base.clone());
                            buf_depth = 2;
                            base
                        } else if *idx == 1 {
                            // The pending vector's length slot.
                            let vec_target = self.project_step(
                                frames,
                                pending,
                                RefProj::Field(0),
                                body,
                                place,
                            )?;
                            let count = self
                                .read_resolved(frames, &vec_target)?
                                .len()
                                .ok_or_else(|| {
                                    InterpError::type_mismatch(
                                        format!("length of {place}"),
                                        "vector value",
                                        "value without a length",
                                    )
                                })?;
                            cur_ty = Some(Ty::Uint(UintTy::Usize));
                            Resolved::Value(Value::Uint(count), Vec::new())
                        } else {
                            // A plain field access after all:
                            // append the pending step, then this
                            // one.
                            let stepped = self.project_step(
                                frames,
                                pending,
                                RefProj::Field(0),
                                body,
                                place,
                            )?;
                            cur_ty = None;
                            self.project_step(
                                frames,
                                stepped,
                                RefProj::Field(*idx),
                                body,
                                place,
                            )?
                        }
                    } else if buf_depth > 0 && *idx == 0 {
                        // A further wrapper step of the buffer
                        // chain: still no storage to denote.
                        buf_depth += 1;
                        cur
                    } else if buf_depth == 2 && *idx == 1 {
                        // The buffer's capacity slot, read
                        // through the chain (an inlined
                        // `Vec::capacity`, e.g. in a `set_len`
                        // precondition check): the capacity the
                        // model reports is the extent of the
                        // element storage it holds (see
                        // `crate::builtins::vec_stored_len`) —
                        // the stored elements, spare capacity
                        // included, so the answer does not move
                        // when the length does.
                        let held = self.read_resolved(frames, &cur)?;
                        let count = match Self::real_vec_cap(&held) {
                            // A vector in its real aggregate
                            // form holds the capacity in the
                            // `RawVecInner` slot itself.
                            Some(cap) => cap,
                            None => crate::builtins::vec_stored_len(&held)
                                .ok_or_else(|| {
                                    InterpError::type_mismatch(
                                        format!("capacity of {place}"),
                                        "vector value",
                                        "value without element storage",
                                    )
                                })?,
                        };
                        buf_depth = 0;
                        buf_base = None;
                        cur_ty = Some(Ty::Uint(UintTy::Usize));
                        Resolved::Value(Value::Uint(count), Vec::new())
                    } else if buf_depth > 0 && *idx >= 1 {
                        // The chain's remaining fields are the
                        // zero-sized allocator and phantom
                        // markers (`RawVec`'s phantom field at
                        // depth 1, `RawVecInner`'s allocator at
                        // depth 2): the value model holds no
                        // storage for them, and the only value
                        // they can take is the empty aggregate
                        // every zero-sized value takes.
                        buf_depth = 0;
                        buf_base = None;
                        cur_ty = None;
                        Resolved::Value(
                            Value::Enum {
                                variant: 0,
                                fields: Vec::new(),
                            },
                            Vec::new(),
                        )
                    } else if *idx == 0
                        && cur_ty.as_ref().is_some_and(is_vec_adt_ty)
                        && self.vec_position_materialised(frames, &cur)
                    {
                        // The buffer field of a `Vec`: enter the
                        // wrapper chain (see the state declared
                        // above). The step denotes the `RawVec`,
                        // which the value model does not hold.
                        buf_base = Some(cur.clone());
                        buf_depth = 1;
                        cur_ty = None;
                        cur
                    } else if *idx == 1
                        && cur_ty.as_ref().is_some_and(is_vec_adt_ty)
                    {
                        let held = self.read_resolved(frames, &cur)?;
                        let count = match crate::builtins::vec_real_aggregate_parts(
                            &held,
                        ) {
                            // A vector in its real aggregate
                            // form holds the length in the slot
                            // itself.
                            Some((_, len)) => len as u128,
                            None => held.len().ok_or_else(|| {
                                InterpError::type_mismatch(
                                    format!("length of {place}"),
                                    "vector value",
                                    "value without a length",
                                )
                            })?,
                        };
                        cur_ty = Some(Ty::Uint(UintTy::Usize));
                        Resolved::Value(Value::Uint(count), Vec::new())
                    } else if suspect
                        && cur_ty.is_none()
                        && *idx == 0
                        && self.suspected_vec_position(frames, &cur)
                    {
                        // The suspected vector position (see
                        // the state above): the wrapper's
                        // payload is a materialised vector
                        // value, so this step is its buffer
                        // field -- enter the wrapper chain
                        // exactly as for a named `Vec` position.
                        buf_base = Some(cur.clone());
                        buf_depth = 1;
                        cur_ty = None;
                        cur
                    } else if suspect
                        && cur_ty.is_none()
                        && *idx == 1
                        && self.suspected_vec_position(frames, &cur)
                    {
                        // The suspected vector's length slot,
                        // read as for a named `Vec` position.
                        let held = self.read_resolved(frames, &cur)?;
                        let count = match crate::builtins::vec_real_aggregate_parts(
                            &held,
                        ) {
                            Some((_, len)) => len as u128,
                            None => held.len().ok_or_else(|| {
                                InterpError::type_mismatch(
                                    format!("length of {place}"),
                                    "vector value",
                                    "value without a length",
                                )
                            })?,
                        };
                        cur_ty = Some(Ty::Uint(UintTy::Usize));
                        Resolved::Value(Value::Uint(count), Vec::new())
                    } else if *idx == 0 && cur_ty.is_none() && root_is_path_chain {
                        // The type tracking has no type for this
                        // step (a wrapper descent the program's
                        // tables cannot follow): if the step
                        // lands on a byte-vector-shaped value, it
                        // may be a vector's buffer field whose
                        // chain the following steps walk — hold
                        // the step pending (see the state above)
                        // instead of appending it.
                        let stepped = self.project_step(
                            frames,
                            cur.clone(),
                            RefProj::Field(0),
                            body,
                            place,
                        )?;
                        match self.read_resolved(frames, &stepped) {
                            Ok(v) if is_byte_vec_value(&v) => {
                                pending_vec = Some(cur.clone());
                                cur
                            }
                            _ => stepped,
                        }
                    } else if *idx == 0
                        && self.position_is_bare_wrapper_field(frames, &cur, &cur_ty)
                    {
                        // The position's declared type is a
                        // one-field wrapper (a single-field
                        // struct, or a one-element tuple over an
                        // array) whose storage holds the field's
                        // value bare -- the form a transmute or
                        // pointer cast between the wrapper and
                        // its field leaves behind, the storage
                        // having been written through the field's
                        // own type. Field 0 of the wrapper IS
                        // that stored value, so the step does
                        // not move (see
                        // `position_is_bare_wrapper_field`).
                        buf_depth = 0;
                        buf_base = None;
                        cur_ty = match &cur_ty {
                            Some(Ty::Tuple(ts)) => ts.first().cloned(),
                            _ => None,
                        };
                        cur
                    } else {
                        buf_depth = 0;
                        buf_base = None;
                        // Whether this step descends out of a
                        // named transparent wrapper (see
                        // `transparent_wrapper_ty`): judged
                        // before the type tracking advances
                        // past the step.
                        let wrapper_descent = *idx == 0
                            && matches!(&cur_ty, Some(t) if transparent_wrapper_ty(t));
                        cur_ty = match &cur_ty {
                            Some(Ty::Tuple(ts)) => ts.get(*idx as usize).cloned(),
                            Some(Ty::Array(elem, _)) => Some((**elem).clone()),
                            Some(ty) => self.adt_field_ty(ty, cur_variant, *idx),
                            _ => None,
                        };
                        cur_variant = 0;
                        // A descent out of a transparent
                        // wrapper whose field type the tracking
                        // cannot name (a generic parameter at
                        // its head: `ManuallyDrop<T>` above
                        // all) leaves the walk on the payload
                        // with no declared type: arm the
                        // suspicion that the payload is a
                        // vector whose structural fields the
                        // following steps name (see the state
                        // above).
                        if wrapper_descent && cur_ty.is_none() {
                            buf_suspect = true;
                        }
                        self.project_step(frames, cur, RefProj::Field(*idx), body, place)?
                    }
                }
                ProjectionElem::UnionField { idx, table } => {
                    buf_depth = 0;
                    buf_base = None;
                    cur_ty = None;
                    fat_cap = None;
                    cur_variant = 0;
                    fat_skip = 0;
                    fat_anchor = None;
                    fat_elem_string = None;
                    fat_elem_ty = None;
                    self.project_step(
                        frames,
                        cur,
                        RefProj::UnionField {
                            idx: *idx,
                            table: *table,
                        },
                        body,
                        place,
                    )?
                }
                ProjectionElem::Index(local) => {
                    buf_depth = 0;
                    buf_base = None;
                    cur_ty = None;
                    cur_variant = 0;
                    fat_skip = 0;
                    fat_anchor = None;
                    let idx = self.read_index_local(frames, body, *local, place)?;
                    let via_fat = fat_cap.is_some_and(|cap| {
                        !self.deref_denotes_storage(frames, &cur, cap, fat_elem_string)
                    });
                    if let Some(cap) = fat_cap {
                        if idx as u128 >= cap {
                            return Err(index_oob(cap as usize, idx));
                        }
                    }
                    fat_cap = None;
                    fat_elem_string = None;
                    let elem_ty = fat_elem_ty.take();
                    cur_ty = match &cur_ty {
                        Some(Ty::Array(elem, _)) | Some(Ty::Slice(elem)) => {
                            Some((**elem).clone())
                        }
                        _ => None,
                    };
                    match via_fat.then(|| compose_fat_index(&cur, idx)).flatten() {
                        Some(composed) => composed,
                        None => {
                            // A scalar-backed slice (see
                            // `target_is_scalar_storage`): element
                            // 0 is the base storage itself, so the
                            // step does not move the position; the
                            // bare index step the fallback adds
                            // would denote a byte of the storage.
                            if via_fat
                                && self.fat_denotes_punned_scalar_element(
                                    frames,
                                    &cur,
                                    elem_ty.as_ref(),
                                )
                            {
                                return Err(InterpError::type_mismatch(
                                    format!("element access in {place}"),
                                    "an element the width of the scalar \
                                     storage behind the slice",
                                    "a punned element width",
                                ));
                            }
                            if via_fat
                                && idx == 0
                                && self.fat_denotes_scalar_element(
                                    frames,
                                    &cur,
                                    elem_ty.as_ref(),
                                )
                            {
                                cur
                            } else {
                                self.project_step(
                                    frames,
                                    cur,
                                    RefProj::Index(idx),
                                    body,
                                    place,
                                )?
                            }
                        }
                    }
                }
                ProjectionElem::ConstantIndex { offset, from_end } => {
                    // The array length the tracked type carries,
                    // captured before the state resets below clear
                    // it: the storage may never have been
                    // materialised (a `Box::new_uninit` pointee's
                    // array field no write has reached), in which
                    // case the type is the only length there is.
                    let ty_len = match &cur_ty {
                        Some(Ty::Array(_, n)) => Some(*n as u128),
                        _ => None,
                    };
                    buf_depth = 0;
                    buf_base = None;
                    cur_ty = None;
                    cur_variant = 0;
                    fat_skip = 0;
                    fat_anchor = None;
                    // The length the index is bounded by: the fat
                    // length when the place was reached through a
                    // fat pointer (the value resolved at an
                    // element position has no length of its own
                    // to ask), otherwise the array type's length
                    // when the types track one, otherwise the
                    // resolved value's own. An unmaterialised
                    // array has no value to take the length from:
                    // the index is a compiler-checked constant,
                    // so resolution proceeds without the bound
                    // and the use of the place materialises the
                    // storage (a write) or reports the
                    // uninitialised read (a read), as it would
                    // for any other unwritten slot.
                    let len = match fat_cap.or(ty_len) {
                        Some(cap) => Some(cap),
                        None => match self.read_resolved(frames, &cur) {
                            Ok(v) => Some(v.len().ok_or_else(|| {
                                InterpError::type_mismatch(
                                    format!("constant index into {place}"),
                                    "array, slice or string value",
                                    "value without a length",
                                )
                            })?),
                            Err(InterpError::UninitializedRead { .. }) => None,
                            Err(e) => return Err(e),
                        },
                    };
                    let via_fat = fat_cap.is_some_and(|cap| {
                        !self.deref_denotes_storage(frames, &cur, cap, fat_elem_string)
                    });
                    fat_cap = None;
                    fat_elem_string = None;
                    let elem_ty = fat_elem_ty.take();
                    cur_ty = match &cur_ty {
                        Some(Ty::Array(elem, _)) | Some(Ty::Slice(elem)) => {
                            Some((**elem).clone())
                        }
                        _ => None,
                    };
                    let idx = match len {
                        Some(len) => {
                            if *from_end {
                                len.checked_sub(*offset as u128).ok_or_else(|| {
                                    index_oob(len as usize, *offset as usize)
                                })?
                            } else if *offset as u128 >= len {
                                return Err(index_oob(len as usize, *offset as usize));
                            } else {
                                *offset as u128
                            }
                        }
                        None => {
                            if *from_end {
                                return Err(InterpError::type_mismatch(
                                    format!("constant index into {place}"),
                                    "array value of known length",
                                    "unmaterialised array indexed from the end",
                                ));
                            }
                            *offset as u128
                        }
                    };
                    let idx = usize::try_from(idx).map_err(|_| {
                        InterpError::UnsupportedProjection {
                            kind: match len {
                                Some(len) => format!(
                                    "index {idx} past the end of the stored value (length {len})"
                                ),
                                None => format!(
                                    "index {idx} past the end of the stored value"
                                ),
                            },
                        }
                    })?;
                    match via_fat.then(|| compose_fat_index(&cur, idx)).flatten() {
                        Some(composed) => composed,
                        None => {
                            // The scalar-backed rule of the
                            // `Index` arm: element 0 is the base
                            // storage itself.
                            if via_fat
                                && self.fat_denotes_punned_scalar_element(
                                    frames,
                                    &cur,
                                    elem_ty.as_ref(),
                                )
                            {
                                return Err(InterpError::type_mismatch(
                                    format!("element access in {place}"),
                                    "an element the width of the scalar \
                                     storage behind the slice",
                                    "a punned element width",
                                ));
                            }
                            if via_fat
                                && idx == 0
                                && self.fat_denotes_scalar_element(
                                    frames,
                                    &cur,
                                    elem_ty.as_ref(),
                                )
                            {
                                cur
                            } else {
                                self.project_step(
                                    frames,
                                    cur,
                                    RefProj::Index(idx),
                                    body,
                                    place,
                                )?
                            }
                        }
                    }
                }
                ProjectionElem::Subslice { from, to, from_end } => {
                    // The array length the tracked type carries,
                    // captured before the state resets below clear
                    // it (see the constant-index arm).
                    let ty_len = match &cur_ty {
                        Some(Ty::Array(_, n)) => Some(*n as u128),
                        _ => None,
                    };
                    buf_depth = 0;
                    buf_base = None;
                    cur_ty = None;
                    cur_variant = 0;
                    fat_skip = 0;
                    fat_anchor = None;
                    // The range is resolved against the length the
                    // base place currently has: the fat length when
                    // the base was reached through a fat deref,
                    // otherwise the array type's length when the
                    // types track one (the storage may never have
                    // been materialised), otherwise the base
                    // value's own length. As for a constant index
                    // (see there), an unmaterialised array's length
                    // is unknown here and the range the MIR carries
                    // is taken as written; the use of the place
                    // materialises or reports, as appropriate. The
                    // resolved step carries an absolute offset and
                    // element count, so later steps (and reference
                    // targets formed from this place) need no
                    // knowledge of the MIR range shape.
                    let len = match fat_cap.or(ty_len) {
                        Some(cap) => Some(cap),
                        None => match self.read_resolved(frames, &cur) {
                            Ok(v) => Some(v.len().ok_or_else(|| {
                                InterpError::type_mismatch(
                                    format!("subslice of {place}"),
                                    "array, slice or string value",
                                    "value without a length",
                                )
                            })?),
                            Err(InterpError::UninitializedRead { .. }) => None,
                            Err(e) => return Err(e),
                        },
                    };
                    fat_cap = None;
                    fat_elem_string = None;
                    fat_elem_ty = None;
                    let (offset, sub_len) = match len {
                        Some(len) => subslice_offset_len(*from, *to, *from_end, len)?,
                        None => {
                            if *from_end {
                                return Err(InterpError::type_mismatch(
                                    format!("subslice of {place}"),
                                    "array value of known length",
                                    "unmaterialised array sliced from the end",
                                ));
                            }
                            (
                                *from as usize,
                                (*to as usize).saturating_sub(*from as usize),
                            )
                        }
                    };
                    cur_ty = match &cur_ty {
                        Some(Ty::Array(elem, _)) | Some(Ty::Slice(elem)) => {
                            Some(Ty::Slice(elem.clone()))
                        }
                        _ => None,
                    };
                    self.project_step(
                        frames,
                        cur,
                        RefProj::Subslice {
                            offset,
                            len: sub_len,
                        },
                        body,
                        place,
                    )?
                }
                ProjectionElem::Downcast(variant) => {
                    buf_depth = 0;
                    buf_base = None;
                    fat_cap = None;
                    fat_elem_string = None;
                    fat_elem_ty = None;
                    fat_skip = 0;
                    fat_anchor = None;
                    cur_variant = *variant as usize;
                    self.project_step(frames, cur, RefProj::Downcast(*variant), body, place)?
                }
                ProjectionElem::Deref => {
                    let pointee_ty = match &cur_ty {
                        Some(Ty::Ref(inner)) => Some((**inner).clone()),
                        _ => None,
                    };
                    // A dereference through the buffer pointer
                    // completes the wrapper chain first: the
                    // value read here is the pointee the pointer
                    // denotes, not the vector's element list.
                    let was_in_chain = buf_depth > 0;
                    if was_in_chain {
                        if let Some(base) = buf_base.take() {
                            cur = self.buf_chain_pointer(frames, &base)?;
                        }
                        buf_depth = 0;
                    }
                    let value = self.read_resolved(frames, &cur)?;
                    fat_cap = match &value {
                        Value::FatRef { len, .. } => Some(*len),
                        _ => None,
                    };
                    fat_elem_ty = if fat_cap.is_some() {
                        match &cur_ty {
                            Some(Ty::Ref(inner)) => match &**inner {
                                Ty::Slice(elem) | Ty::Array(elem, _) => {
                                    Some((**elem).clone())
                                }
                                _ => None,
                            },
                            _ => None,
                        }
                    } else {
                        None
                    };
                    fat_elem_string = match &value {
                        Value::FatRef { .. } if step_idx == 0 => body
                            .local_ty(place.local)
                            .and_then(fat_slice_elem_is_string_carrier),
                        _ => None,
                    };
                    // A dereference of a wrapper-typed reference
                    // whose target is a vector's buffer pointer
                    // re-enters the wrapper chain at the depth
                    // the wrapper type names (see
                    // `vec_buf_chain_reentry`): the reference a
                    // collected body passes across a call (a
                    // `&mut RawVec` handed to `grow_one`, a
                    // `&mut RawVecInner` in `grow_amortized`)
                    // denotes the vector's bookkeeping, not its
                    // first element, and the field steps that
                    // follow it are the chain's bookkeeping
                    // steps, judged by the same arms as a walk
                    // that entered the chain from the vector
                    // place itself.
                    let reentry = if was_in_chain {
                        None
                    } else {
                        self.vec_buf_chain_reentry(
                            frames,
                            &value,
                            pointee_ty.as_ref(),
                        )?
                    };
                    fat_anchor = None;
                    fat_skip = 0;
                    match reentry {
                        Some((base, depth)) => {
                            buf_base = Some(base.clone());
                            buf_depth = depth;
                            cur_ty = None;
                            base
                        }
                        None => {
                            let resolved = match self.deref_value(frames, &value) {
                                Ok(resolved) => resolved,
                                Err(err) => {
                                    match self.storage_less_zst_deref(
                                        &value,
                                        pointee_ty.as_ref(),
                                    ) {
                                        Some(resolved) => resolved,
                                        None => match self.fn_code_deref(
                                            &value,
                                            pointee_ty.as_ref(),
                                        ) {
                                            Some(resolved) => resolved,
                                            None => return Err(err),
                                        },
                                    }
                                }
                            };
                            // A raw allocation block holds its elements
                            // as a tuple: a dereference of its base
                            // pointer denotes the first element, not the
                            // block as a whole, whenever the steps that
                            // follow address the element itself — the
                            // dereference is the last step, or a
                            // structural step (field, downcast, union
                            // field) follows, which can only address the
                            // pointee's own structure, never the block's
                            // tuple. A sequence step (index, subslice)
                            // instead continues the block's flat
                            // sequence, so no entry is inserted for it
                            // and the flat reading the sequence machinery
                            // relies on is unchanged. The entry is
                            // inserted before the wrapper peels below so
                            // they judge the value at the element
                            // position. Fat pointers are exempt: they
                            // carry their own projection, so the resolved
                            // form is never the bare block.
                            let resolved = if deref_elem0 && matches!(value, Value::Ref(_)) {
                                let next_is_structural = matches!(
                                    place.projection.get(step_idx + 1),
                                    Some(
                                        ProjectionElem::Field(_)
                                            | ProjectionElem::Downcast(_)
                                            | ProjectionElem::UnionField { .. },
                                    )
                                );
                                match &resolved {
                                    Resolved::Alloc { id, projection }
                                        if (is_last || next_is_structural)
                                            && projection.is_empty()
                                            && self.is_elem_block(*id)
                                            && matches!(
                                                self.alloc_base(*id).ok().as_ref(),
                                                Some(Value::Tuple(_) | Value::Bytes(_))
                                            ) =>
                                    {
                                        Resolved::Alloc {
                                            id: *id,
                                            projection: vec![RefProj::Index(0)],
                                        }
                                    }
                                    _ => resolved,
                                }
                            } else {
                                resolved
                            };
                            // A thin pointer whose target is a
                            // bare array base -- a leading field
                            // path with no element step, the form
                            // pointer casts leave behind when the
                            // first-element step the pointer was
                            // formed with is stripped (a fat
                            // pointer's base kept by a fat-to-thin
                            // cast, a cast chain redenominating a
                            // byte array's elements) -- denotes
                            // the array's first element whenever
                            // the pointee is the element type or a
                            // scalar (see
                            // `bare_array_base_denotes_elem0`).
                            // The dereference being the last step,
                            // or a structural step following, is
                            // what makes the position the element
                            // itself; a following sequence step
                            // continues the array's flat reading.
                            // (The raw-block twin of this rule is
                            // the `deref_elem0` entry above.)
                            let resolved = {
                                let next_is_structural = matches!(
                                    place.projection.get(step_idx + 1),
                                    Some(
                                        ProjectionElem::Field(_)
                                            | ProjectionElem::Downcast(_)
                                            | ProjectionElem::UnionField { .. },
                                    )
                                );
                                let denotes_elem0 = (is_last || next_is_structural)
                                    && matches!(value, Value::Ref(_))
                                    && pointee_ty.as_ref().is_some_and(|pointee| {
                                        self.bare_array_base_denotes_elem0(
                                            frames, &resolved, pointee,
                                        )
                                    });
                                if denotes_elem0 {
                                    let mut resolved = resolved;
                                    match &mut resolved {
                                        Resolved::Place {
                                            projection, ..
                                        }
                                        | Resolved::Static {
                                            projection, ..
                                        } => {
                                            projection.push(RefProj::Index(0));
                                        }
                                        _ => {}
                                    }
                                    resolved
                                } else {
                                    resolved
                                }
                            };
                            let resolved = match &pointee_ty {
                                Some(pointee) => {
                                    let resolved = self.peel_maybe_uninit_entry(
                                        frames, resolved, pointee,
                                    );
                                    // A one-element tuple stored
                                    // where the pointee is the
                                    // tuple's own array element
                                    // type: the tuple is a
                                    // one-field wrapper around the
                                    // array the steps that follow
                                    // address (see
                                    // `enter_tuple_wrapped_sequence`).
                                    match pointee {
                                        Ty::Array(elem, n) => {
                                            let resolved = self
                                                .enter_tuple_wrapped_sequence(
                                                    frames, resolved, elem,
                                                    *n,
                                                );
                                            self.window_to_declared_array(
                                                frames, resolved, *n,
                                            )
                                        }
                                        _ => resolved,
                                    }
                                }
                                None => {
                                    // A dereference of an ADT-typed
                                    // place (a `Box` local) carries no
                                    // pointee type in this walk. When the
                                    // next step indexes, the position must
                                    // be array storage, whose own value
                                    // form is a tuple/byte block -- never a
                                    // one-field enum -- so a one-field enum
                                    // there is a wrapper layer the index
                                    // would otherwise consume as an
                                    // element (see `peel_wrapper_for_index`).
                                    let next_is_index = matches!(
                                        place.projection.get(step_idx + 1),
                                        Some(
                                            ProjectionElem::Index(_)
                                                | ProjectionElem::ConstantIndex { .. }
                                                | ProjectionElem::Subslice { .. },
                                        )
                                    );
                                    if next_is_index
                                        && matches!(cur_ty, Some(Ty::Adt(..)))
                                    {
                                        self.peel_wrapper_for_index(frames, resolved)
                                    } else {
                                        resolved
                                    }
                                }
                            };
                            // A fat pointer whose base is an element
                            // position of the sequence it spans
                            // exactly (the shape a byte-string
                            // wrapper's buffer pointer has): the
                            // field steps that follow address the
                            // wrapper over the sequence, so they
                            // re-anchor to the sequence value's
                            // container position (see the `Field`
                            // arm and `fat_element_anchor`). A
                            // sub-slice's base (its length shorter
                            // than the stored sequence's) offers no
                            // anchor, as do all thin dereferences.
                            fat_anchor = if matches!(value, Value::FatRef { .. }) {
                                match fat_element_anchor(&resolved) {
                                    Some((container, depth)) => {
                                        let spans = self
                                            .read_resolved(frames, &container)
                                            .ok()
                                            .and_then(|v| v.len())
                                            .is_some_and(|len| Some(len) == fat_cap);
                                        if spans {
                                            Some((container, depth))
                                        } else {
                                            None
                                        }
                                    }
                                    None => None,
                                }
                            } else {
                                None
                            };
                            // A fat pointer whose base is the
                            // sequence value's own position: the
                            // walk already denotes the sequence, so
                            // the pointee wrapper's field levels
                            // are consumed without moving the
                            // position (see
                            // `sequence_wrapper_depth`). The gate
                            // is the resolved value's own length
                            // being the fat length.
                            if fat_anchor.is_none()
                                && matches!(value, Value::FatRef { .. })
                            {
                                let depth = pointee_ty
                                    .as_ref()
                                    .map_or(0, |ty| self.sequence_wrapper_depth(ty));
                                if depth > 0
                                    && self
                                        .read_resolved(frames, &resolved)
                                        .ok()
                                        .and_then(|v| v.len())
                                        == fat_cap
                                {
                                    fat_skip = depth;
                                }
                            }
                            // A dereference whose formation
                            // regrouped the storage under the
                            // pointee's shape: the steps that
                            // follow price through the pointee,
                            // so the resolved target records
                            // the denomination for the consumers
                            // that meet it without an accessing
                            // place (see
                            // `Interpreter::denominate_fat_deref`).
                            // The wrapper levels a `fat_skip`
                            // walk consumes keep the sequence's
                            // own denomination, so they record
                            // none.
                            let resolved = if fat_skip == 0 {
                                match &pointee_ty {
                                    Some(pointee) => self
                                        .denominate_fat_deref(
                                            frames, resolved, pointee,
                                        ),
                                    _ => resolved,
                                }
                            } else {
                                resolved
                            };
                            cur_ty = pointee_ty;
                            cur_variant = 0;
                            resolved
                        }
                    }
                }
            };
        }
        // A pending byte-vector step the walk ends on is the
        // plain field access it is.
        if let Some(pending) = pending_vec.take() {
            cur = self.project_step(frames, pending, RefProj::Field(0), body, place)?;
        }
        // A walk that ends inside the buffer wrapper chain
        // denotes the buffer pointer itself -- unless the
        // vector at the chain's base is in its real aggregate
        // form. There the chain positions are the aggregate's
        // own fields (the `RawVec` at the vector's field 0,
        // the `RawVecInner` at the buffer's field 0, and so on
        // down the wrapper chain), so the position the walk
        // reached is the base extended by the field steps it
        // took into the chain. A reference formed over the
        // position then denotes the bookkeeping storage
        // itself, and the field steps a callee takes through
        // it -- the capacity slot above all -- read and write
        // the aggregate. Completing the walk as the bare
        // buffer pointer instead loses the base: the pointer
        // denotes the element storage, a dereference of a
        // wrapper-typed reference over it cannot re-enter the
        // chain (see `vec_buf_chain_reentry`), and the
        // bookkeeping steps that follow could only fabricate
        // values for slots that live in the aggregate.
        if buf_depth > 0 {
            if let Some(base) = buf_base.take() {
                let real_form = matches!(
                    self.read_resolved(frames, &base),
                    Ok(held)
                        if crate::builtins::vec_real_aggregate_parts(&held)
                            .is_some()
                );
                if real_form {
                    let mut resolved = base;
                    let projection = match &mut resolved {
                        Resolved::Place { projection, .. }
                        | Resolved::Value(_, projection)
                        | Resolved::Static { projection, .. }
                        | Resolved::Alloc { projection, .. } => projection,
                    };
                    for _ in 0..buf_depth {
                        projection.push(RefProj::Field(0));
                    }
                    cur = resolved;
                } else {
                    cur = self.buf_chain_pointer(frames, &base)?;
                }
            }
        }
        Ok(cur)
    }

    /// The `MaybeUninit` entry a dereference crosses when the
    /// pointer's declared pointee is the payload type but the
    /// storage at the resolved position still carries the
    /// union's wrapper layer.
    ///
    /// The storage keeps the wrapper form whenever the payload
    /// was written through a pointer whose own projection enters
    /// the wrapper first (the `as_mut_ptr` cast appends that
    /// entry step), and pointers that reach the same allocation
    /// without the entry step -- the transmute-retargeted box
    /// `Box::assume_init` produces is the canonical case -- then
    /// read the wrapper itself where the payload is expected.
    /// The dereference is the one place that knows both sides:
    /// the declared pointee from the projection walk and the
    /// stored value at the resolved position. When the stored
    /// value is an enum whose first field is the pointee's
    /// value form, the resolution descends the entry step, so
    /// reads, writes, moves and drops all address the payload.
    ///
    /// The descent is refused whenever the wrapper reading could
    /// be the position's own value: a `MaybeUninit` pointee is
    /// read as the wrapper by definition; a one-field ADT's own
    /// value is indistinguishable from the wrapper form (its
    /// debug shape reports one field), as is a union's; a slice
    /// pointee is never entered here. The descent is also refused
    /// when the position's own declared type -- recorded for a
    /// place that is its home local itself and for a static --
    /// is an ADT other than `MaybeUninit` itself: an enum stored
    /// at such a position is that ADT's own value, and a
    /// differently-typed pointer reaching it (a byte view of an
    /// enum's storage is the canonical case) reinterprets the
    /// storage rather than addressing a wrapper's payload. An
    /// unreadable or absent stored value keeps the unentered
    /// resolution, so error behaviour on uninitialised storage
    /// is unchanged.
    /// Whether the position `cur` denotes is a one-field
    /// wrapper's field stored bare: the position's declared
    /// type, where the walk tracks it, is a struct with a
    /// single field (or a one-element tuple over an array),
    /// and the value stored at the position is the field's own
    /// value form -- a tuple of elements or a byte block,
    /// never the wrapper's own aggregate (an ADT value is an
    /// enum in this model, and a tuple wrapper's own form is a
    /// one-element tuple holding the field). That is the
    /// storage a transmute or pointer cast between the wrapper
    /// and its field leaves behind: the storage was written
    /// through the field's type, so a `Field(0)` step at the
    /// position denotes the stored value itself, not its
    /// first element. (The scalar and string forms of the same
    /// situation are answered inside the projectors; the
    /// sequence forms are ambiguous there -- a bare tuple is
    /// also a tuple value's own form -- so this step is judged
    /// here, where the declared type is known.)
    fn position_is_bare_wrapper_field(
        &self,
        frames: &[Frame],
        cur: &Resolved,
        cur_ty: &Option<Ty>,
    ) -> bool {
        // The value at the position the step denotes. An element
        // block's root target denotes its first element, and the
        // block's own stored form is the element tuple -- a tuple
        // by construction, which says nothing about the form the
        // element is stored in -- so an element block's storage
        // is judged at the element position, not at the block.
        let position_value = match cur {
            Resolved::Alloc { id, projection }
                if projection.is_empty() && self.is_elem_block(*id) =>
            {
                self.read_resolved(
                    frames,
                    &Resolved::Alloc {
                        id: *id,
                        projection: vec![RefProj::Index(0)],
                    },
                )
            }
            _ => self.read_resolved(frames, cur),
        };
        match cur_ty {
            Some(Ty::Adt(name)) => {
                let Some(shape) = self.debug_shape_for_def(name) else {
                    return false;
                };
                if shape.kind != sparagmos_mir::AdtShapeKind::Struct
                    || shape.variants.len() != 1
                    || shape.variants[0].fields.len() != 1
                {
                    return false;
                }
                matches!(
                    position_value,
                    Ok(Value::Tuple(_) | Value::Bytes(_))
                )
            }
            Some(Ty::Tuple(ts))
                if ts.len() == 1 && matches!(ts[0], Ty::Array(..)) =>
            {
                match position_value {
                    Ok(Value::Bytes(_)) => true,
                    Ok(Value::Tuple(items)) => items.len() != 1,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// The entry for the dereference of an array-typed
    /// reference whose target holds a one-element tuple around
    /// the array: the tuple is a one-field wrapper whose only
    /// field is the array itself (a `([T; N],)` value viewed
    /// through the `&[T; N]` a transmute formed), so the steps
    /// that follow address the tuple's single item, as
    /// `enter_wrapper_field` arranges. A stored tuple whose
    /// length is the array's own length is the array's genuine
    /// form and is not entered; a one-element array stays
    /// ambiguous with its wrapper and keeps the genuine
    /// reading.
    ///
    /// A one-item tuple is ALSO the array's own sparse prefix
    /// when only its first element is materialised -- a nested
    /// array whose first row alone is written stores exactly
    /// that shape (a one-row tuple holding the row) -- so the
    /// entry is refused when the single item is already
    /// plausible as one element of the declared array: the
    /// stored form is then consistent as a prefix of the
    /// declared array type, and entering it would insert a
    /// field step the place does not have, leaving the
    /// following index steps to address the first element's
    /// own parts instead of the array's elements. The wrapper
    /// reading stands only when the item cannot be an element:
    /// a tuple or byte block is never a scalar element, so
    /// the `([u8; N],)` shape still enters.
    /// The target `resolved` with a trailing element-index
    /// step rebased to the window of `len` elements starting
    /// at that index, when the container the step ranges over
    /// holds a sequence that covers the window. `None` when
    /// the target does not end in an index step or the
    /// container is not such a sequence.
    fn rebase_element_position(
        &self,
        frames: &[Frame],
        resolved: &Resolved,
        len: u64,
    ) -> Option<Resolved> {
        fn projection_of(target: &Resolved) -> &Vec<RefProj> {
            match target {
                Resolved::Place { projection, .. }
                | Resolved::Static { projection, .. }
                | Resolved::Alloc { projection, .. } => projection,
                Resolved::Value(_, projection) => projection,
            }
        }
        fn projection_of_mut(target: &mut Resolved) -> &mut Vec<RefProj> {
            match target {
                Resolved::Place { projection, .. }
                | Resolved::Static { projection, .. }
                | Resolved::Alloc { projection, .. } => projection,
                Resolved::Value(_, projection) => projection,
            }
        }
        let Some(RefProj::Index(start)) =
            projection_of(resolved).last().cloned()
        else {
            return None;
        };
        let mut rebased = resolved.clone();
        projection_of_mut(&mut rebased).pop();
        let stored = self.read_resolved(frames, &rebased).ok()?;
        let total = match &stored {
            Value::Tuple(items) => items.len() as u64,
            Value::Bytes(bytes) => bytes.len() as u64,
            Value::SparseVec { len, .. } => {
                u64::try_from(*len).unwrap_or(u64::MAX)
            }
            _ => return None,
        };
        if start as u64 + len > total {
            return None;
        }
        projection_of_mut(&mut rebased).push(RefProj::Subslice {
            offset: start,
            len: len as usize,
        });
        Some(rebased)
    }

    /// A dereference whose pointee is an array of exactly
    /// `len` elements, over storage holding a longer element
    /// sequence: the pointer named the whole allocation (a
    /// fat pointer cast to an array pointer drops the fat
    /// metadata at the cast; a sub-slice's `as_ptr` names the
    /// allocation it was carved from), so the target the walk
    /// resolved denotes the allocation's sequence, not the
    /// array's. The declared array is the sequence's first
    /// `len` elements: the target is windowed to them, and
    /// every later step addresses inside the window. A target
    /// whose stored form is not a longer sequence is returned
    /// unchanged.
    fn window_to_declared_array(
        &self,
        frames: &[Frame],
        resolved: Resolved,
        len: u64,
    ) -> Resolved {
        // A target whose projection already passes through
        // a recorded view denomination (a slice cast's
        // regrouping of the storage it ranges over) denotes
        // exactly the element the view's steps select: the
        // cast's denomination governs the dereference, and
        // re-windowing the target against the longer birth
        // sequence behind the view would retarget it at a
        // window of elements instead of the element itself.
        let projection = match &resolved {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. } => projection,
            Resolved::Value(_, projection) => projection,
        };
        if projection
            .iter()
            .any(|step| matches!(step, RefProj::View(_)))
        {
            return resolved;
        }
        // An element position of a longer sequence (the
        // target ends in an index step, the shape a
        // sub-slice's `as_ptr` leaves): the array the
        // dereference denotes starts at that element, so the
        // trailing step becomes the window over the container
        // that starts there.
        if let Some(rebased) =
            self.rebase_element_position(frames, &resolved, len)
        {
            return rebased;
        }
        let Ok(stored) = self.read_resolved(frames, &resolved) else {
            return resolved;
        };
        let stored_len = match &stored {
            Value::Tuple(items) => items.len() as u64,
            Value::Bytes(bytes) => bytes.len() as u64,
            Value::SparseVec { len, .. } => {
                u64::try_from(*len).unwrap_or(u64::MAX)
            }
            _ => return resolved,
        };
        if stored_len <= len {
            return resolved;
        }
        let step = RefProj::Subslice {
            offset: 0,
            len: len as usize,
        };
        let mut resolved = resolved;
        match &mut resolved {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. } => projection.push(step),
            Resolved::Value(_, projection) => projection.push(step),
        }
        resolved
    }

    fn enter_tuple_wrapped_sequence(
        &self,
        frames: &[Frame],
        resolved: Resolved,
        elem: &Ty,
        len: u64,
    ) -> Resolved {
        if len == 1 {
            return resolved;
        }
        let Ok(stored) = self.read_resolved(frames, &resolved) else {
            return resolved;
        };
        let Value::Tuple(items) = &stored else {
            return resolved;
        };
        if items.len() == 1
            && matches!(items[0], Value::Tuple(_) | Value::Bytes(_))
            && !self.value_plausible_for_ty(&items[0], elem, 0)
        {
            enter_wrapper_field(resolved)
        } else {
            resolved
        }
    }

    /// Whether `value` is plausible as a value of `ty`, judged
    /// on the value's own form alone (the type-level mirror of
    /// [`Self::value_plausible_for_tag`]). Conservative in the
    /// same way: a slot not yet written is plausible for every
    /// type, a form this judgment does not model keeps the
    /// benefit of the doubt, and `false` is answered only on a
    /// clear mismatch -- a scalar, reference or ADT value is
    /// never a bare tuple or byte block, and a tuple or array
    /// value is never a bare scalar.
    fn value_plausible_for_ty(&self, value: &Value, ty: &Ty, depth: usize) -> bool {
        if matches!(value, Value::Uninit) || depth > 4 {
            return true;
        }
        match (ty, value) {
            (Ty::Array(elem, n), Value::Tuple(items)) => {
                items.len() as u64 <= *n
                    && items
                        .iter()
                        .all(|item| self.value_plausible_for_ty(item, elem, depth + 1))
            }
            // A byte block is an array's flat storage form,
            // whatever the element type: the bytes are
            // reinterpreted under the declared element type on
            // every read.
            (Ty::Array(..), Value::Bytes(_)) => true,
            (Ty::Tuple(ts), Value::Tuple(items)) => {
                items.len() <= ts.len()
                    && items
                        .iter()
                        .zip(ts.iter())
                        .all(|(item, t)| self.value_plausible_for_ty(item, t, depth + 1))
            }
            (Ty::Uint(_), Value::Uint(_)) => true,
            (Ty::Int(_), Value::Int(_)) => true,
            (Ty::Bool, Value::Bool(_)) => true,
            (Ty::Float(_), Value::Float(_) | Value::Float128(_)) => true,
            (Ty::Unit, Value::Unit) => true,
            (Ty::Str, Value::Str(_) | Value::Bytes(_)) => true,
            (
                Ty::Ref(_),
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. },
            ) => true,
            (Ty::FnPtr | Ty::FnDef(_), Value::FnPtr(_)) => true,
            (Ty::Closure(..), Value::Closure { .. }) => true,
            (Ty::Coroutine(_), Value::Coroutine { .. }) => true,
            (Ty::Adt(name), Value::Enum { variant, fields }) => {
                match self.debug_shape_for_def(name) {
                    Some(shape) => {
                        shape.kind == sparagmos_mir::AdtShapeKind::Union
                            || self.plausible_adt_enum(*variant, fields, shape, depth)
                    }
                    None => true,
                }
            }
            (Ty::Adt(name), Value::Unit) => self
                .debug_shape_for_def(name)
                .is_none_or(|shape| {
                    shape
                        .variants
                        .iter()
                        .any(|variant| variant.fields.is_empty())
                }),
            (Ty::Slice(_) | Ty::Dynamic(_) | Ty::Never, _) => true,
            _ => false,
        }
    }

    /// Whether a dereference target that is a bare array base
    /// (a place or static whose projection is a leading field
    /// path with no element step) denotes the array's first
    /// element for a pointee of `pointee`: the target's declared
    /// type is an array, and the pointee either is the array's
    /// element type or is a scalar -- a scalar pointee at an
    /// array base can only denote an element-sized value there,
    /// whatever denomination the pointer casts that formed the
    /// pointer used (a byte array's storage read through a
    /// `usize` pointer after a unit-changing cast chain). An
    /// aggregate pointee that is not the element type may
    /// denote the array as a whole and keeps the bare form.
    fn bare_array_base_denotes_elem0(
        &self,
        frames: &[Frame],
        resolved: &Resolved,
        pointee: &Ty,
    ) -> bool {
        let (root_ty, projection) = match resolved {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let Some(frame) = frames.get(*frame_pos) else {
                    return false;
                };
                let Some(home) = self.program.get(&frame.func) else {
                    return false;
                };
                let Some(root) = home.local_ty(Local(*local)) else {
                    return false;
                };
                (root.clone(), projection)
            }
            Resolved::Static { name, projection } => {
                let Some(root) = self.program.static_tys.get(name) else {
                    return false;
                };
                (root.clone(), projection)
            }
            _ => return false,
        };
        if !projection.iter().all(|step| {
            matches!(
                step,
                RefProj::Field(_) | RefProj::Downcast(_) | RefProj::UnionField { .. }
            )
        }) {
            return false;
        }
        let scalar_pointee = matches!(
            pointee,
            Ty::Bool | Ty::Int(_) | Ty::Uint(_) | Ty::Float(_)
        );
        if let Some(Ty::Array(elem, _)) =
            step_ty_across(Some(self.program), Some(root_ty), projection)
        {
            return pointee == &*elem || scalar_pointee;
        }
        // The declared-type walk cannot always judge the
        // position: the field types of a generic ADT whose
        // fields depend on its parameters are not recorded in
        // the program's table. The stored value's own shape is
        // the remaining evidence: a sequence aggregate (an
        // array's tuple of elements, a byte block) at the bare
        // base, read through a scalar pointee, denotes its
        // first element -- a scalar dereference can only name
        // an element-sized value at the position.
        if !scalar_pointee {
            return false;
        }
        let Ok(value) = self.read_resolved(frames, resolved) else {
            return false;
        };
        matches!(value, Value::Tuple(_) | Value::Bytes(_))
    }

    fn peel_maybe_uninit_entry(
        &self,
        frames: &[Frame],
        resolved: Resolved,
        pointee: &Ty,
    ) -> Resolved {
        if matches!(pointee, Ty::Adt(name)
            if name.starts_with("std::mem::MaybeUninit")
                || name.starts_with("core::mem::MaybeUninit"))
        {
            return resolved;
        }
        // The position's own declared type, where the model
        // records it, outranks the stored form: a place with no
        // projection is its home local itself, whose declared
        // type the home body's table carries, a static's
        // declared type is in the program's table, and an
        // allocation's is in the record its creation point left
        // (see the `alloc_tys` field). When that type is an ADT
        // other than `MaybeUninit` itself, an enum stored at
        // the position is that ADT's own value, never a wrapper
        // layer. Positions whose home type the model does not
        // record (an allocation created without a type, a
        // projected place) keep the stored-form evidence below.
        let home_is_foreign_adt = match &resolved {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } if projection.is_empty() => frames
                .get(*frame_pos)
                .and_then(|f| self.program.get(&f.func))
                .and_then(|b| b.local_ty(Local(*local)))
                .is_some_and(|ty| {
                    matches!(ty, Ty::Adt(_)) && !is_maybe_uninit_ty(ty)
                }),
            Resolved::Static { name, projection }
                if projection.is_empty() =>
            {
                self.program.static_tys.get(name).is_some_and(|ty| {
                    matches!(ty, Ty::Adt(_)) && !is_maybe_uninit_ty(ty)
                })
            }
            Resolved::Alloc { id, projection }
                if projection.is_empty() =>
            {
                // The record outranks the stored form only at a
                // byte-shaped position: a byte pointee denotes
                // the ADT's own storage (an enum's tag span,
                // which the byte view prices from the recorded
                // type), and there a stored enum's first field
                // is indistinguishable from a payload's first
                // byte. A wider pointee names a value inside
                // the storage -- a wrapper's payload, read
                // through a pointer cast -- whose entry the
                // stored-form evidence below judges, as it does
                // for positions without a record; and a
                // single-field struct's record never outranks,
                // because its stored form is a wrapper layer
                // around its one field at every width.
                let byte_position = matches!(
                    pointee,
                    Ty::Uint(UintTy::U8) | Ty::Int(IntTy::I8) | Ty::Bool
                );
                byte_position
                    && self.alloc_ty(*id).is_some_and(|ty| match &ty {
                        Ty::Adt(name) => {
                            !is_maybe_uninit_ty(&ty)
                                && !self.adt_is_single_field_wrapper(name)
                        }
                        _ => false,
                    })
            }
            _ => false,
        };
        if home_is_foreign_adt {
            return resolved;
        }
        let Ok(stored) = self.read_resolved(frames, &resolved) else {
            return resolved;
        };
        let Value::Enum { variant, fields } = &stored else {
            return resolved;
        };
        // The payload sits in the wrapper's first field. A
        // union skeleton can leave further member slots behind
        // it (an array payload's skeleton materialises both
        // `MaybeUninit` members, and the payload write replaces
        // only the first), so for a non-ADT pointee the entry
        // decision rests on the first field alone: the gate
        // above has already refused the positions whose own
        // declared type is a foreign ADT, so at the positions
        // that reach here an enum value is the wrapper layer.
        // For an ADT pointee the stored enum may
        // BE the ADT's own value, so the descent requires the
        // stricter wrapper evidence of `wrapper_payload_matches`:
        // a one-field stored enum whose field is itself the
        // ADT's complete value form.
        let Some(field) = fields.first() else {
            return resolved;
        };
        if *field != Value::Uninit {
            if !self.wrapper_payload_matches(field, *variant, pointee, fields.len()) {
                return resolved;
            }
            return enter_wrapper_field(resolved);
        }
        // The raw form of the same wrapper: the union value a
        // field-by-field write through the union's declared
        // members leaves behind -- the payload written into the
        // `value` member (index 1 in `MaybeUninit`) while the
        // `uninit` member's slot before it was never written.
        // The payload sits in the one live member, inside that
        // member's own one-field form (the `ManuallyDrop<T>` the
        // member is declared as), so the entry descends both
        // steps. A stored enum holding several live members is
        // not a shape the write side produces for a wrapper,
        // and a live member that is not a one-field form is not
        // this wrapper's shape; both keep the unentered
        // resolution, as does a payload that does not match the
        // pointee.
        let mut live = fields
            .iter()
            .enumerate()
            .filter(|(_, f)| **f != Value::Uninit);
        let Some((idx, stored)) = live.next() else {
            return resolved;
        };
        if live.next().is_some() {
            return resolved;
        }
        let Value::Enum { fields: inner, .. } = stored else {
            return resolved;
        };
        if inner.len() != 1 {
            return resolved;
        }
        if !self.wrapper_payload_matches(&inner[0], *variant, pointee, 1) {
            return resolved;
        }
        enter_wrapper_fields(resolved, &[idx, 0])
    }

    /// The `MaybeUninit` entry for the dereference of an
    /// ADT-typed place (a `Box` local) directly followed by an
    /// index-like step, where the walk carries no pointee type:
    /// the indexed position is array storage, and array storage's
    /// own value form in this model is a tuple or byte block, so
    /// an enum whose first field is one of those at the resolved
    /// position can only be the wrapper layer (a
    /// `Box<MaybeUninit<[T; N]>>` retargeted by `assume_init`'s
    /// transmute). Without the entry the index step reads the
    /// wrapper's sole field -- the whole array -- as element 0.
    fn peel_wrapper_for_index(
        &self,
        frames: &[Frame],
        resolved: Resolved,
    ) -> Resolved {
        let Ok(stored) = self.read_resolved(frames, &resolved) else {
            return resolved;
        };
        let Value::Enum { fields, .. } = &stored else {
            return resolved;
        };
        let Some(field) = fields.first() else {
            return resolved;
        };
        if !matches!(field, Value::Tuple(_) | Value::Bytes(_)) {
            return resolved;
        }
        enter_wrapper_field(resolved)
    }

    /// Whether `field`, the first field of the value stored at
    /// a dereferenced position, is the value form of the
    /// declared pointee type -- the evidence that the stored
    /// enum is a `MaybeUninit` wrapper around the pointee rather
    /// than the position's own value. For an ADT pointee a
    /// stored enum whose field count is the declared count of
    /// its variant in the type's debug shape, with a first field
    /// that fits that variant's declared first field, is the
    /// ADT's value itself, and only a one-field stored enum
    /// whose field is itself a value of the ADT (an enum
    /// carrying its variant's fields) is a wrapper around it.
    /// An aggregate payload may
    /// be partially materialised -- a prefix of its declared
    /// fields, the form field-by-field writes through pointers
    /// leave behind -- and still be the pointee's value, so the
    /// aggregate arms accept a prefix of the declared count.
    /// A one-field ADT's own value stays ambiguous with its
    /// wrapper and is not entered; a union's fields are read
    /// through union steps, never through this entry.
    fn wrapper_payload_matches(
        &self,
        field: &Value,
        stored_variant: u32,
        pointee: &Ty,
        stored_fields: usize,
    ) -> bool {
        match (pointee, field) {
            (Ty::Tuple(ts), Value::Tuple(items)) => items.len() <= ts.len(),
            (Ty::Array(_, n), Value::Tuple(items)) => items.len() as u64 <= *n,
            (Ty::Array(elem, n), Value::Bytes(b)) => {
                matches!(**elem, Ty::Uint(UintTy::U8)) && b.len() as u64 == *n
            }
            // A slice pointee's payload is the array value the
            // wrapper holds: an element tuple whose items all
            // fit the slice's element type, or a byte block for
            // a byte slice. The slice name carries no length of
            // its own, so any item count is the payload's.
            (Ty::Slice(elem), Value::Tuple(items)) => {
                items.iter().all(|item| value_fits_ty(item, elem))
            }
            (Ty::Slice(elem), Value::Bytes(_)) => {
                matches!(**elem, Ty::Uint(UintTy::U8))
            }
            (Ty::Uint(_), Value::Uint(_)) => true,
            (Ty::Int(_), Value::Int(_)) => true,
            (Ty::Bool, Value::Bool(_)) => true,
            (Ty::Float(_), Value::Float(_) | Value::Float128(_)) => true,
            (Ty::Unit, Value::Unit) => true,
            (
                Ty::Ref(_),
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. },
            ) => true,
            (Ty::Str, Value::Str(_) | Value::Bytes(_)) => true,
            (Ty::FnPtr | Ty::FnDef(_), Value::FnPtr(_)) => true,
            (Ty::Closure(..), Value::Closure { .. }) => true,
            (Ty::Coroutine(..), Value::Coroutine { .. }) => true,
            (
                Ty::Adt(name),
                Value::Enum {
                    variant: inner_variant,
                    fields: inner_fields,
                },
            ) => {
                // The pointee type names the definition alone,
                // while the recorded shapes are per instance and
                // their field types differ across instances, so
                // no single recorded instance can be identified
                // as the one the position holds. The refusals
                // below therefore run over every recorded
                // instance: a stored enum that is any instance's
                // own value is that instance's value, never a
                // wrapper, and a field's incompatibility with one
                // instance's declared field is not wrapper
                // evidence while another instance's field fits
                // it. (Judging by one arbitrary instance misread
                // a genuine `G<E>` value as a wrapper whenever
                // the pick landed on `G<u32>`.)
                let shapes = self.debug_shapes_for_def(name);
                if shapes.is_empty() {
                    return false;
                }
                if shapes
                    .iter()
                    .any(|shape| shape.kind == sparagmos_mir::AdtShapeKind::Union)
                {
                    return false;
                }
                // A stored enum carrying the declared field
                // count of its own variant, with a first field
                // that fits the variant's declared first field,
                // is the ADT's complete value itself, however
                // deep its one-field chains run -- never a
                // wrapper. The field fit is part of the
                // evidence: the stored variant index of a
                // wrapper is the union's field index, not a
                // variant of the pointee at all, and it can
                // coincide with a one-field variant of the
                // pointee (a `MaybeUninit<E>` payload read back
                // through an `E` pointer, where `E`'s first
                // variant takes one field of another type) --
                // there the stored field, itself a complete
                // value of the pointee, cannot be the declared
                // field's value, and the wrapper reading below
                // decides. Judging by the first field is
                // enough: a stored enum with more fields is
                // refused as a wrapper by the next check either
                // way.
                for shape in &shapes {
                    let declared = shape.variants.get(stored_variant as usize);
                    if let Some(declared) = declared {
                        if declared.fields.len() == stored_fields
                            && declared.fields.first().is_some_and(|first| {
                                self.value_plausible_for_tag(field, &first.ty, 0)
                            })
                        {
                            return false;
                        }
                    }
                }
                if stored_fields != 1 {
                    return false;
                }
                // A one-field stored enum can still be the
                // ADT's own partially materialised value -- a
                // prefix holding only its first field -- when
                // that field fits the declared first field's
                // type. It is the wrapper only when it cannot
                // be the ADT's own prefix under any recorded
                // instance and its sole field is itself a
                // plausible value of the ADT under one of them.
                for shape in &shapes {
                    let declared = shape.variants.get(stored_variant as usize);
                    if let Some(declared) = declared {
                        if let Some(first) = declared.fields.first() {
                            if self.value_plausible_for_tag(field, &first.ty, 0) {
                                return false;
                            }
                        }
                    }
                }
                shapes.iter().any(|shape| {
                    self.plausible_adt_enum(*inner_variant, inner_fields, shape, 0)
                })
            }
            _ => false,
        }
    }

    /// The recorded debug shape of the ADT definition a type
    /// name denotes. The table is keyed by the rendered
    /// instance name, while the interpreter's type names the
    /// definition alone; the judgments that consult the shape
    /// here ask only for variant field counts and the shape
    /// kind, which every instance of the definition shares, so
    /// an instance entry answers for the definition when no
    /// definition-named entry exists. Judgments that turn on a
    /// field's *type* cannot use this pick -- field types
    /// differ across instances -- and must consider every
    /// recorded instance through `debug_shapes_for_def`.
    pub(crate) fn debug_shape_for_def(
        &self,
        name: &str,
    ) -> Option<&sparagmos_mir::AdtDebugShape> {
        self.program.adt_debug_shapes.get(name).or_else(|| {
            self.program
                .adt_debug_shapes
                .iter()
                .find(|(key, _)| {
                    key.strip_prefix(name)
                        .is_some_and(|rest| rest.starts_with('<'))
                })
                .map(|(_, shape)| shape)
        })
    }

    /// Every recorded debug shape of the ADT definition a type
    /// name denotes: the definition-named entry, when one
    /// exists, plus every instance entry. See
    /// `debug_shape_for_def` for why judgments that turn on
    /// field types must consult them all.
    fn debug_shapes_for_def(
        &self,
        name: &str,
    ) -> Vec<&sparagmos_mir::AdtDebugShape> {
        let mut shapes = Vec::new();
        if let Some(shape) = self.program.adt_debug_shapes.get(name) {
            shapes.push(shape);
        }
        shapes.extend(
            self.program
                .adt_debug_shapes
                .iter()
                .filter(|(key, _)| {
                    key.strip_prefix(name)
                        .is_some_and(|rest| rest.starts_with('<'))
                })
                .map(|(_, shape)| shape),
        );
        shapes
    }

    /// Whether an enum value is a plausible (complete or
    /// partially materialised prefix) value of the ADT a debug
    /// shape describes: the variant is one of the declared
    /// ones, the fields are a prefix of the variant's declared
    /// fields, and each present field is plausible for its
    /// declared field type. The check is conservative: it
    /// answers `false` only on a clear mismatch, so a value it
    /// cannot judge counts as plausible.
    fn plausible_adt_enum(
        &self,
        variant: u32,
        fields: &[Value],
        shape: &sparagmos_mir::AdtDebugShape,
        depth: usize,
    ) -> bool {
        let Some(declared) = shape.variants.get(variant as usize) else {
            return false;
        };
        if fields.len() > declared.fields.len() {
            return false;
        }
        fields.iter().enumerate().all(|(i, field)| {
            self.value_plausible_for_tag(field, &declared.fields[i].ty, depth + 1)
        })
    }

    /// Whether `value` is plausible as a value of the type a
    /// debug-shape field tag names (see
    /// [`Self::plausible_adt_enum`] for the conservatism rule).
    /// An uninitialized slot is plausible for every type: it
    /// is the state of any field not yet written.
    fn value_plausible_for_tag(&self, value: &Value, tag: &str, depth: usize) -> bool {
        if matches!(value, Value::Uninit) || depth > 4 {
            return true;
        }
        let tag = tag.trim();
        match tag {
            "bool" => return matches!(value, Value::Bool(_)),
            "char" => return matches!(value, Value::Uint(_)),
            "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => {
                return matches!(value, Value::Uint(_))
            }
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => {
                return matches!(value, Value::Int(_))
            }
            "f16" | "f32" | "f64" | "f128" => {
                return matches!(value, Value::Float(_) | Value::Float128(_))
            }
            "str" => return matches!(value, Value::Str(_) | Value::Bytes(_)),
            "()" => return matches!(value, Value::Unit),
            _ => {}
        }
        if tag.starts_with('(') {
            return matches!(value, Value::Tuple(_) | Value::Unit);
        }
        if tag.starts_with('[') {
            return matches!(value, Value::Tuple(_) | Value::Bytes(_) | Value::Str(_));
        }
        if tag.starts_with('&') || tag.starts_with('*') {
            return matches!(
                value,
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }
            );
        }
        if let Some(shape) = self.program.adt_debug_shapes.get(tag) {
            if shape.kind == sparagmos_mir::AdtShapeKind::Union {
                return true;
            }
            return match value {
                Value::Enum { variant, fields } => {
                    self.plausible_adt_enum(*variant, fields, shape, depth)
                }
                Value::Unit => shape
                    .variants
                    .iter()
                    .any(|variant| variant.fields.is_empty()),
                _ => false,
            };
        }
        true
    }

    /// The target a place walk inside a `Vec`'s buffer wrapper
    /// chain denotes (see `resolve_place_mode`): the chain's
    /// steps walk `RawVec` / `Unique` / `NonNull` wrappers
    /// around the buffer pointer, none of which is storage in
    /// the value model, so the denoted place is the pointer
    /// itself — a reference to the vector place's first element,
    /// the same target `Vec::as_ptr` produces — as a value
    /// target.
    fn buf_chain_pointer(
        &self,
        frames: &[Frame],
        base: &Resolved,
    ) -> InterpResult<Resolved> {
        if let Ok(held) = self.read_resolved(frames, base) {
            if crate::builtins::vec_real_aggregate_parts(&held).is_some() {
                if let Some(ptr) = Self::real_vec_buffer_ptr(&held) {
                    return Ok(Resolved::Value(ptr, Vec::new()));
                }
            }
        }
        let target = match base {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(0));
                RefTarget::place(frames[*frame_pos].id, *local, projection)
            }
            Resolved::Static { name, projection } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(0));
                RefTarget::Static {
                    name: name.clone(),
                    projection,
                }
            }
            Resolved::Alloc { id, projection } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(0));
                RefTarget::Alloc {
                    id: *id,
                    projection,
                }
            }
            Resolved::Value(base_value, projection) => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(0));
                RefTarget::Value(Box::new(base_value.clone()), projection)
            }
        };
        Ok(Resolved::Value(Value::Ref(target), Vec::new()))
    }

    /// Re-entry into a `Vec`'s buffer wrapper chain through a
    /// reference (see `resolve_place_mode`'s `Deref` arm): when
    /// the dereferenced value is a vector's buffer pointer — the
    /// target a chain walk produces, the vector place with a
    /// trailing first-element step — and the reference's pointee
    /// type names a wrapper of the chain, the denoted place is
    /// the vector place itself and the chain depth is the
    /// wrapper's depth in the chain. `None` for every other
    /// reference: an ordinary element pointer (the pointee type
    /// names no wrapper), a wrapper over storage that is not a
    /// vector value, a pointer whose target is not the vector
    /// place's first element.
    pub(crate) fn vec_buf_chain_reentry(
        &self,
        frames: &[Frame],
        value: &Value,
        pointee: Option<&Ty>,
    ) -> InterpResult<Option<(Resolved, usize)>> {
        let Some(depth) = pointee.and_then(buf_chain_depth) else {
            return Ok(None);
        };
        let Value::Ref(target) = value else {
            return Ok(None);
        };
        // The buffer pointer a reference formation produced over
        // a value-resolved chain place arrives wrapped in a
        // value target holding the pointer itself; the denoted
        // target is the pointer's own.
        let target = match target {
            RefTarget::Value(inner, proj) if proj.is_empty() => match &**inner {
                Value::Ref(inner_target) => inner_target,
                _ => return Ok(None),
            },
            other => other,
        };
        let base: Resolved = match target {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                if projection.last() != Some(&RefProj::Index(0)) {
                    return Ok(None);
                }
                let frame_pos = self.frame_pos(frames, *frame)?;
                Resolved::Place {
                    frame_pos,
                    local: *local,
                    projection: projection[..projection.len() - 1].to_vec(),
                }
            }
            RefTarget::Static { name, projection } => {
                if projection.last() != Some(&RefProj::Index(0)) {
                    return Ok(None);
                }
                Resolved::Static {
                    name: name.clone(),
                    projection: projection[..projection.len() - 1].to_vec(),
                }
            }
            RefTarget::Alloc { id, projection } => {
                if projection.last() != Some(&RefProj::Index(0)) {
                    return Ok(None);
                }
                Resolved::Alloc {
                    id: *id,
                    projection: projection[..projection.len() - 1].to_vec(),
                }
            }
            _ => return Ok(None),
        };
        // The base must currently hold a vector value: a
        // wrapper-typed reference over any other storage is an
        // ordinary aggregate reference.
        let held = self.read_resolved(frames, &base)?;
        if crate::builtins::vec_stored_len(&held).is_none() {
            return Ok(None);
        }
        Ok(Some((base, depth)))

    }

    /// Whether the position a walk has reached holds a `Vec`
    /// in its materialised form: the element list as an
    /// aggregate value. The buffer-chain treatment in the
    /// `Field` arm of [`Self::resolve_place_mode`] and the
    /// length-slot write in [`Self::write_vec_len_slot`] apply
    /// only then. A `Vec`-typed position can instead hold a
    /// bare byte string - a `String` the model keeps as
    /// `Value::Str` has no materialised `Vec` / `RawVec` /
    /// `NonNull` structure, and the plain projection arms
    /// answer walks over it (field 0 is the string itself,
    /// field 1 its length; see the string arms of
    /// `project_value`). Entering the chain there would form
    /// an element pointer into a value that stores no elements.
    /// Whether the position `pos` denotes holds a value in a
    /// vector's own shape: a sparse vector, or an aggregate
    /// whose fields form an element list -- every field the
    /// same top-level shape, or, for elements of one enum
    /// type, the same shape within each variant (an enum's
    /// variants are the one place elements of one type may
    /// differ in shape at all). A struct aggregate's fields
    /// are its declared fields, of distinct types in general,
    /// and their shapes differ within the struct's single
    /// variant: a reference-counted handle, whose fields are
    /// a one-field wrapper around the allocation reference
    /// beside two empty marker aggregates, is not an element
    /// list, and a walk over it must take its field steps as
    /// the field accesses they are. (An empty or one-field
    /// aggregate is vacuous: it has no field a structural
    /// step could misread as an element.) This is the
    /// judgment the suspected-position arms apply to a
    /// payload the type tracking could not name (see the
    /// `buf_suspect` state in `resolve_place_mode`).
    fn suspected_vec_position(&self, frames: &[Frame], pos: &Resolved) -> bool {
        match self.read_resolved(frames, pos) {
            Ok(Value::SparseVec { .. }) => true,
            Ok(Value::Enum { fields, .. }) => Self::fields_form_element_list(&fields),
            _ => false,
        }
    }

    fn vec_position_materialised(&self, frames: &[Frame], pos: &Resolved) -> bool {
        matches!(
            self.read_resolved(frames, pos),
            Ok(Value::Enum { .. } | Value::SparseVec { .. })
        )
    }

    /// Whether an aggregate's fields form an element list
    /// (see [`Self::suspected_vec_position`]): every field
    /// has the same top-level shape signature, or every
    /// field is an enum value whose signature is constant
    /// within each variant. A value's shape signature is
    /// its kind, its variant for an enum value, and its
    /// field count for an aggregate (a sequence value's
    /// length is data, not shape, so byte blocks and
    /// strings carry no count).
    fn fields_form_element_list(fields: &[Value]) -> bool {
        fn shape(value: &Value) -> (&'static str, Option<u32>, Option<usize>) {
            match value {
                Value::Enum { variant, fields } => {
                    (value.kind_name(), Some(*variant), Some(fields.len()))
                }
                Value::Tuple(items) => {
                    (value.kind_name(), None, Some(items.len()))
                }
                other => (other.kind_name(), None, None),
            }
        }
        if fields.windows(2).all(|w| shape(&w[0]) == shape(&w[1])) {
            return true;
        }
        if !fields.iter().all(|f| matches!(f, Value::Enum { .. })) {
            return false;
        }
        let mut seen: Vec<(u32, (&'static str, Option<u32>, Option<usize>))> =
            Vec::new();
        for field in fields {
            let Value::Enum { variant, .. } = field else {
                unreachable!("checked above")
            };
            let sig = shape(field);
            match seen.iter().find(|(v, _)| v == variant) {
                Some((_, prev)) if *prev != sig => return false,
                Some(_) => {}
                None => seen.push((*variant, sig)),
            }
        }
        true
    }

    /// The declared type of field `idx` of variant `variant` of
    /// a value of type `ty`, from the conversion-recorded field
    /// table (see [`Program::adt_field_tys`]) for ADTs and from
    /// the type itself for tuples. `None` when the field's type
    /// is not recorded (it mentions a generic parameter at its
    /// head): place-walk type tracking ends there.
    pub(crate) fn adt_field_ty(&self, ty: &Ty, variant: usize, idx: usize) -> Option<Ty> {
        match ty {
            Ty::Adt(name) => self
                .program()
                .adt_field_tys
                .get(name)?
                .get(variant)?
                .get(idx)?
                .clone(),
            Ty::Tuple(ts) => ts.get(idx).cloned(),
            _ => None,
        }
    }

    /// The length carried by the fat pointer a place's final
    /// `Deref` step traverses (`(*p)` where `p` holds a fat
    /// reference), when there is one. Resolving that place reaches
    /// the base place and loses the fat length, so consumers that
    /// need the pointee's own length (`Len`, reference formation)
    /// recover it here from the pointer value itself.
    /// Whether the position a fat-pointer dereference resolved
    /// to holds the slice's own storage: an aggregate value whose
    /// element count is the fat length (a `Vec` value's fields are
    /// its elements; an unsized array value is its element tuple).
    /// Indexing such a position goes INTO the aggregate, so a
    /// following index step is appended as usual. When the
    /// position instead holds one element of the container the
    /// slice lives in (a raw-pointer-derived subslice base), the
    /// value there is a single element — a scalar, or an aggregate
    /// whose own count is the element's, not the slice's — and the
    /// index step composes with the base position instead (see
    /// `compose_fat_index`). A position that cannot be read does
    /// not denote storage.
    pub(crate) fn deref_denotes_storage(
        &self,
        frames: &[Frame],
        cur: &Resolved,
        fat_len: u128,
        _elem_string: Option<bool>,
    ) -> bool {
        // A position that ends at one element of a container,
        // entered through whole-value entry steps (the shape a
        // transparent-wrapper entry leaves, as in the elements of
        // a `[MaybeUninit<T>]` buffer), denotes that element as a
        // whole: the slice's elements are the container's elements
        // entered the same way, so when the slice fits inside the
        // container at the base index, the position is an element
        // position, never the slice's own storage -- whatever the
        // entered element's own field count happens to be. Judging
        // by the count alone mistook a two-field element of a
        // two-element slice (the array clone guard's partial drop)
        // for the slice's storage.
        if let Some((container, base)) = entered_element_container(cur) {
            if let Ok(value) = self.read_resolved(frames, &container) {
                if let Some(count) = value.len() {
                    if base as u128 + fat_len <= count {
                        return false;
                    }
                    // The slice does not fit in the container at
                    // the base index, so the position is not an
                    // element of that container: the field steps
                    // after the index select the slice's own
                    // storage inside the element (an array field
                    // of a node struct is the canonical case), and
                    // the position denotes that storage. Composing
                    // the index into the container's index step
                    // would address a sibling element instead.
                    return true;
                }
            }
        }
        match self.read_resolved(frames, cur) {
            // The position denotes the slice's storage only when
            // the value there is the aggregate whose elements the
            // slice ranges over. A fat pointer or string value at
            // the position is one *element* of the slice (a
            // `Vec<&str>` element is itself a fat pointer whose
            // own length can coincide with the slice's): its
            // `len()` reports the length it carries, not a
            // storage element count, and judging by it appended a
            // second index step that read into the element.
            Ok(
                value @ (Value::Tuple(_)
                | Value::Enum { .. }
                | Value::SparseVec { .. }),
            ) => value.len() == Some(fat_len),
            Ok(_) => false,
            Err(_) => false,
        }
    }



    pub(crate) fn fat_deref_len(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> Option<u128> {
        if !matches!(place.projection.last(), Some(ProjectionElem::Deref)) {
            return None;
        }
        let prefix = Place {
            local: place.local,
            projection: place.projection[..place.projection.len() - 1].to_vec(),
        };
        match self.read_place(frames, body, &prefix) {
            Ok(Value::FatRef { len, .. }) => Some(len),
            _ => None,
        }
    }

    /// The fat length denoted by the last dereference step in
    /// a place's projection: the length of the fat pointer the
    /// prefix up to that step holds. For places that walk into
    /// a fat pointee's fields (`(*p).inner`), this is the
    /// metadata those fields' borrows carry.
    pub(crate) fn fat_prefix_deref_len(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> Option<u128> {
        let pos = place
            .projection
            .iter()
            .rposition(|e| matches!(e, ProjectionElem::Deref))?;
        let prefix = Place {
            local: place.local,
            projection: place.projection[..=pos].to_vec(),
        };
        self.fat_deref_len(frames, body, &prefix)
    }

    /// The trait-object fat pointer a place's last dereference
    /// crosses, when it crosses one: the value of the pointer
    /// the step dereferences, when that value is a `dyn` fat
    /// pointer. A DST struct's unsized tail shares the
    /// enclosing fat pointer's vtable, so a borrow of a place
    /// reached through the dereference (`&(*x).ptr` for the
    /// tail field of a `Fat<dyn Trait>`, say) denotes the
    /// trait object the crossed pointer names.
    pub(crate) fn crossed_dyn_ref(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> Option<Value> {
        let pos = place
            .projection
            .iter()
            .rposition(|e| matches!(e, ProjectionElem::Deref))?;
        let prefix = Place {
            local: place.local,
            projection: place.projection[..pos].to_vec(),
        };
        match self.read_place(frames, body, &prefix) {
            Ok(v @ Value::DynRef { .. }) => Some(v),
            _ => None,
        }
    }

    /// The `dyn` identity a borrow of `place` takes when the
    /// borrow's own type is a trait object (its bounds are
    /// `dest_traits`) but the place is not itself a fat
    /// pointer, so the thin borrow the place resolution forms
    /// carries no vtable and a method call through it never
    /// engages virtual dispatch: the identity a fat pointer
    /// attached to the place carries — the value stored at
    /// the place, when that value is a fat pointer the borrow
    /// reborrows, or the fat pointer the place's last
    /// dereference crosses, when the place is the unsized
    /// tail of the pointee it denotes. `None` when neither
    /// names the borrow's principal trait: the borrow then
    /// keeps its thin form.
    pub(crate) fn dyn_borrow_source(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
        target: &Resolved,
        dest_traits: &[String],
    ) -> Option<DynBorrowSource> {
        let principal = dest_traits.first()?;
        let names_principal = |traits: &[String]| traits.iter().any(|t| t == principal);
        if let Ok(v @ Value::DynRef { .. }) = self.read_resolved(frames, target) {
            let Value::DynRef { traits, .. } = &v else {
                unreachable!()
            };
            if names_principal(traits) {
                return Some(DynBorrowSource::Stored(v));
            }
        }
        if let Some(crossed) = self.crossed_dyn_ref(frames, body, place) {
            let Value::DynRef {
                concrete,
                traits,
                drop_glue,
                ..
            } = crossed
            else {
                unreachable!()
            };
            if names_principal(&traits) {
                return Some(DynBorrowSource::Crossed {
                    concrete,
                    traits: dest_traits.to_vec(),
                    drop_glue,
                });
            }
        }
        None
    }

    /// Whether a type is dynamically sized in the model's
    /// terms: a slice, a string, a trait object or a foreign
    /// type; a tuple whose tail is dynamically sized; or an
    /// ADT whose last field is (recursively) dynamically
    /// sized — the shape of the byte-string DST wrappers
    /// (`OsStr`, `Path`, `CStr`) and of custom DST structs.
    /// Judged from the conversion-recorded field table; a
    /// bounded walk, as a recursive tail would be an
    /// infinitely sized type. A reference to an unsized type
    /// is a fat pointer: it occupies two words, not one.
    pub(crate) fn ty_is_unsized(&self, ty: &Ty) -> bool {
        fn inner(interp: &Interpreter, ty: &Ty, depth: usize) -> bool {
            if depth > 16 {
                return false;
            }
            match ty {
                Ty::Slice(_) | Ty::Str | Ty::Dynamic(..) | Ty::Foreign(_) => {
                    true
                }
                Ty::Tuple(ts) => ts.last().is_some_and(|tail| {
                    inner(interp, tail, depth + 1)
                }),
                Ty::Adt(name) => interp
                    .program()
                    .adt_field_tys
                    .get(name)
                    .is_some_and(|variants| {
                        variants.iter().any(|fields| {
                            fields.last().is_some_and(|field| {
                                field.as_ref().is_some_and(|t| {
                                    inner(interp, t, depth + 1)
                                })
                            })
                        })
                    }),
                _ => false,
            }
        }
        inner(self, ty, 0)
    }

    /// The number of field-0 wrapper levels between a DST
    /// pointee type and the sequence it wraps: the levels of
    /// `OsStr` over its sys `Slice` over `[u8]` are 2. Zero when
    /// the type is the sequence itself or its field 0 is not a
    /// wrapper over one.
    fn sequence_wrapper_depth(&self, ty: &Ty) -> usize {
        let mut depth = 0;
        let mut cur = ty.clone();
        loop {
            match &cur {
                Ty::Slice(_) | Ty::Str => return depth,
                Ty::Adt(_) => {
                    if depth >= 16 {
                        return 0;
                    }
                    match self.adt_field_ty(&cur, 0, 0) {
                        Some(next) => {
                            depth += 1;
                            cur = next;
                        }
                        None => return 0,
                    }
                }
                _ => return 0,
            }
        }
    }

    /// The fat length for a place naming a DST's unsized tail
    /// field (`(*p).tail`): the field's metadata is the
    /// enclosing value's metadata, so the length is the one
    /// [`Self::fat_deref_len`] reports for the place with the
    /// field step stripped. Consumers try this only after the
    /// field value's own length: a sized field (an array field
    /// of a DST struct) reports its own length that way, and
    /// only a tail field over storage the model cannot read
    /// as a sequence (byte storage reached through a thin
    /// base pointer, as a `CStr`'s `[c_char]` field is) needs
    /// the enclosing pointer's length.
    pub(crate) fn fat_tail_field_len(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> Option<u128> {
        if !matches!(place.projection.last(), Some(ProjectionElem::Field(_))) {
            return None;
        }
        let stripped = Place {
            local: place.local,
            projection: place.projection[..place.projection.len() - 1].to_vec(),
        };
        if let Some(len) = self.fat_deref_len(frames, body, &stripped) {
            return Some(len);
        }
        // The pointer at the deref may itself sit behind a
        // reference (a method receiver passed by autoref
        // reborrow: the parameter holds a reference to the
        // caller's fat pointer value). Follow that one level,
        // as `ptr_metadata`'s reference arm does, and take the
        // fat length of the pointer it denotes.
        if !matches!(
            stripped.projection.last(),
            Some(ProjectionElem::Deref)
        ) {
            return None;
        }
        let prefix = Place {
            local: stripped.local,
            projection: stripped.projection[..stripped.projection.len() - 1].to_vec(),
        };
        let head = self.read_place(frames, body, &prefix).ok()?;
        if !matches!(head, Value::Ref(_)) {
            return None;
        }
        let target = self.deref_value(frames, &head).ok()?;
        let pointee = self.read_resolved(frames, &target).ok()?;
        match pointee {
            Value::FatRef { len, .. } => Some(len),
            _ => None,
        }
    }

    fn project_step(
        &self,
        frames: &[Frame],
        cur: Resolved,
        step: RefProj,
        body: &Body,
        place: &Place,
    ) -> InterpResult<Resolved> {
        match cur {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(step);
                Ok(Resolved::Place {
                    frame_pos,
                    local,
                    projection,
                })
            }
            Resolved::Value(base, projection) => {
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| body.name.clone());
                let mut projection = projection.clone();
                projection.push(step);
                let projected =
                    self.project_value_t(base, &projection, &func, &place.to_string(), None)?;
                Ok(Resolved::Value(projected, Vec::new()))
            }
            Resolved::Static {
                name,
                mut projection,
            } => {
                projection.push(step);
                Ok(Resolved::Static { name, projection })
            }
            Resolved::Alloc {
                id,
                mut projection,
            } => {
                projection.push(step);
                Ok(Resolved::Alloc { id, projection })
            }
        }
    }
}

/// The guest panic rustc raises for an out-of-bounds index.
/// The (offset, element count) a MIR `Subslice` projection
/// selects from a base of `len` elements, with core's own panic
/// messages for an out-of-range selection. With `from_end`, `to`
/// counts back from the end of the base (the range is
/// `from..len - to`); otherwise `to` is the range's end.
fn subslice_offset_len(
    from: u64,
    to: u64,
    from_end: bool,
    len: u128,
) -> InterpResult<(usize, usize)> {
    let from = from as u128;
    let end = if from_end {
        len.checked_sub(to as u128).ok_or_else(|| {
            InterpError::panic(format!(
                "range end index {to} out of range for slice of length {len}"
            ))
        })?
    } else {
        to as u128
    };
    if from > end {
        return Err(InterpError::panic(format!(
            "slice index starts at {from} but ends at {end}"
        )));
    }
    if end > len {
        return Err(InterpError::panic(format!(
            "range end index {end} out of range for slice of length {len}"
        )));
    }
    let offset = usize::try_from(from).map_err(|_| {
        InterpError::panic(format!("slice index {from} does not fit in usize"))
    })?;
    let count = usize::try_from(end - from).map_err(|_| {
        InterpError::panic(format!(
            "slice length {} does not fit in usize",
            end - from
        ))
    })?;
    Ok((offset, count))
}

/// Check an already-resolved (offset, count) sub-slice selection
/// against a materialised value's element count, with the same
/// panic messages as [`subslice_offset_len`].
pub(crate) fn check_subslice(value_len: usize, offset: usize, count: usize) -> InterpResult<()> {
    let end = offset + count;
    if end > value_len {
        return Err(InterpError::panic(format!(
            "range end index {end} out of range for slice of length {value_len}"
        )));
    }
    Ok(())
}

/// Compose an index step that immediately follows a fat-pointer
/// dereference with the resolved target's trailing element index
/// (the caller has already established that the dereference does
/// not denote the slice's own storage — see
/// `Interpreter::deref_denotes_storage`).
/// A fat pointer's base denotes the slice's first element, so
/// when that base is itself an element position (the trailing
/// step is an index into the container the slice lives in), the
/// slice-relative index denotes the element `base + idx` of that
/// container, not an element of the first element: the trailing
/// step is advanced instead of a nested step being appended.
/// The base may also enter that first element through whole-value
/// entry steps (field steps and downcasts — the shape a
/// transparent-wrapper entry leaves, as in a slice over the `S`
/// values of a `[MaybeUninit<S>]` buffer): the position inside
/// the element is the same in every element, so the index step
/// before the entry steps is advanced and the entry steps are
/// kept. Any other target shape answers `None` and the caller
/// appends the step as usual.
/// Whether a projection ends at an element position: its last
/// step is an index step, or an index step followed only by
/// whole-value entry steps (field and downcast steps) -- the
/// entered-element shape (see [`entered_element_container`]).
/// Whether an allocation projection denotes the allocation's
/// base: no steps at all, or only zero steps -- an `Index(0)` or
/// a `ByteOffset(0)`, the forms pointer arithmetic and the
/// element-block denominations leave on a pointer that still
/// denotes the start of its block (a byte-denominated buffer
/// pointer carries `[Index(0)]`, for instance).
pub(crate) fn projection_is_base(projection: &[RefProj]) -> bool {
    projection
        .iter()
        .all(|s| matches!(s, RefProj::Index(0) | RefProj::ByteOffset(0)))
}

pub(crate) fn projection_ends_at_element(projection: &[RefProj]) -> bool {
    let mut pos = projection.len();
    while pos > 0 {
        match projection[pos - 1] {
            // A whole-value entry step enters a transparent
            // wrapper's only field (or downcasts to the entered
            // variant); any other field step selects a genuine
            // field of the element the index step denotes (a
            // node's key array, say), and the position denotes
            // that field's value, not the element.
            RefProj::Field(0) | RefProj::Downcast(_) => pos -= 1,
            RefProj::Index(_) => return true,
            _ => return false,
        }
    }
    false
}

/// The container position and base index of a resolved position
/// that ends at one element of a container, entered through
/// whole-value entry steps (field and downcast steps -- the shape
/// a transparent-wrapper entry leaves, as in the elements of a
/// `[MaybeUninit<T>]` buffer): the position
/// `[.., Index(k), entry...]`, with at least one entry step,
/// denotes element `k` of the container the index step ranges
/// over, entered the way every element of that container is.
/// Any other shape answers `None`.
/// The element type of the fat pointer type `ty` (a
/// reference to a slice or to an array), when the type
/// says it.
pub(crate) fn fat_slice_elem_ty(ty: &Ty) -> Option<Ty> {
    let pointee = match ty {
        Ty::Ref(pointee) => pointee,
        _ => return None,
    };
    match &**pointee {
        Ty::Slice(elem) | Ty::Array(elem, _) => Some((**elem).clone()),
        _ => None,
    }
}

/// Whether the elements of the fat pointer type `ty` (a
/// reference to a slice, or to `str`) are string carriers: an
/// element whose stored value is the string data itself (a
/// `String` value or an `&str` value, both stored as
/// [`Value::Str`]). `Some(false)` for a byte slice and for
/// `str` itself, whose elements are the string's own bytes, so
/// a string value at the base position is the slice's storage.
/// `None` when the type does not say.
pub(crate) fn fat_slice_elem_is_string_carrier(ty: &Ty) -> Option<bool> {
    let pointee = match ty {
        Ty::Ref(pointee) => pointee,
        _ => return None,
    };
    match &**pointee {
        Ty::Str => Some(false),
        Ty::Slice(elem) => match &**elem {
            Ty::Adt(name) if is_string_adt_name(name) => Some(true),
            Ty::Ref(inner) if matches!(&**inner, Ty::Str) => Some(true),
            Ty::Uint(UintTy::U8) | Ty::Int(IntTy::I8) => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Whether a bulk sequence's element type is a string carrier
/// (see `fat_slice_elem_is_string_carrier`): an element whose
/// stored value is the string data itself.
fn elem_is_string_carrier(elem: &Ty) -> bool {
    match elem {
        Ty::Adt(name) => is_string_adt_name(name),
        Ty::Ref(pointee) => matches!(&**pointee, Ty::Str),
        _ => false,
    }
}

/// Whether a value is a leaf-data aggregate: an aggregate
/// whose fields are all scalars — a byte buffer's stored
/// form, a scalar array's. Such a value is data, not a
/// wrapper around data (see `Interpreter::copy_unwrap`).
pub(crate) fn is_leaf_data_aggregate(value: &Value) -> bool {
    fn scalar(v: &Value) -> bool {
        matches!(
            v,
            Value::Int(_) | Value::Uint(_) | Value::Float(_) | Value::Bool(_)
        )
    }
    match value {
        Value::Enum { fields, .. } => {
            !fields.is_empty() && fields.iter().all(scalar)
        }
        Value::Tuple(items) => !items.is_empty() && items.iter().all(scalar),
        _ => false,
    }
}

/// Whether a value has the stored shape of a `Vec<u8>`: the
/// element-list aggregate whose elements are all bytes. (An
/// empty aggregate has no elements to judge by and is not
/// claimed: the wrapper chains this shape detection serves
/// always run over a buffer the program filled first, and an
/// empty buffer's chain denotes a dangling pointer nothing
/// reads through.)
fn is_byte_vec_value(value: &Value) -> bool {
    match value {
        Value::Enum { variant: 0, fields } => {
            !fields.is_empty()
                && fields
                    .iter()
                    .all(|f| matches!(f, Value::Uint(n) if *n <= 0xFF))
        }
        _ => false,
    }
}

/// Whether the ADT name is the string type, under either its
/// `std` or its `alloc` spelling.
fn is_string_adt_name(name: &str) -> bool {
    name == "std::string::String" || name == "alloc::string::String"
}

fn entered_element_container(cur: &Resolved) -> Option<(Resolved, usize)> {
    let projection: &[RefProj] = match cur {
        Resolved::Place { projection, .. }
        | Resolved::Static { projection, .. }
        | Resolved::Alloc { projection, .. } => projection,
        Resolved::Value(..) => return None,
    };
    let mut pos = projection.len();
    let mut entered = false;
    while pos > 0 {
        match projection[pos - 1] {
            RefProj::Field(0) | RefProj::Downcast(_) => {
                entered = true;
                pos -= 1;
            }
            RefProj::Field(_) => return None,
            RefProj::Index(k) => {
                if !entered {
                    return None;
                }
                let container_projection = projection[..pos - 1].to_vec();
                let container = match cur {
                    Resolved::Place {
                        frame_pos, local, ..
                    } => Resolved::Place {
                        frame_pos: *frame_pos,
                        local: *local,
                        projection: container_projection,
                    },
                    Resolved::Static { name, .. } => Resolved::Static {
                        name: name.clone(),
                        projection: container_projection,
                    },
                    Resolved::Alloc { id, .. } => Resolved::Alloc {
                        id: *id,
                        projection: container_projection,
                    },
                    Resolved::Value(..) => unreachable!(),
                };
                return Some((container, k));
            }
            _ => return None,
        }
    }
    None
}

/// A resolved target descended through a one-field wrapper's
/// entry step: the `Field(0)` a `MaybeUninit` (or like wrapper)
/// layer adds between a pointer and its payload, appended to
/// whichever store the target denotes.
/// A resolved target descended through several wrapper entry
/// steps in order (see [`enter_wrapper_field`]): the raw form
/// of a `MaybeUninit` value holds its payload behind the
/// union's `value` member and that member's own field, so the
/// entry appends both steps.
fn enter_wrapper_fields(resolved: Resolved, steps: &[usize]) -> Resolved {
    let mut resolved = resolved;
    for step in steps {
        resolved = match resolved {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(RefProj::Field(*step));
                Resolved::Place {
                    frame_pos,
                    local,
                    projection,
                }
            }
            Resolved::Alloc {
                id,
                mut projection,
            } => {
                projection.push(RefProj::Field(*step));
                Resolved::Alloc { id, projection }
            }
            Resolved::Static {
                name,
                mut projection,
            } => {
                projection.push(RefProj::Field(*step));
                Resolved::Static { name, projection }
            }
            Resolved::Value(base, mut projection) => {
                projection.push(RefProj::Field(*step));
                Resolved::Value(base, projection)
            }
        };
    }
    resolved
}

fn enter_wrapper_field(resolved: Resolved) -> Resolved {
    let entered = RefProj::Field(0);
    match resolved {
        Resolved::Place {
            frame_pos,
            local,
            mut projection,
        } => {
            projection.push(entered);
            Resolved::Place {
                frame_pos,
                local,
                projection,
            }
        }
        Resolved::Alloc {
            id,
            mut projection,
        } => {
            projection.push(entered);
            Resolved::Alloc { id, projection }
        }
        Resolved::Static {
            name,
            mut projection,
        } => {
            projection.push(entered);
            Resolved::Static { name, projection }
        }
        Resolved::Value(base, mut projection) => {
            projection.push(entered);
            Resolved::Value(base, projection)
        }
    }
}

/// The container anchor of a resolved position that ends at a
/// sequence element (a trailing `Index` step): the same target
/// with that step stripped — the position of the sequence value
/// itself — together with the number of steps preceding the
/// index step (the storage descent depth the position already
/// encodes). `None` for any other shape.
fn fat_element_anchor(cur: &Resolved) -> Option<(Resolved, usize)> {
    fn strip(projection: &[RefProj]) -> Option<(Vec<RefProj>, usize)> {
        match projection.last() {
            Some(RefProj::Index(_)) => Some((
                projection[..projection.len() - 1].to_vec(),
                projection.len() - 1,
            )),
            _ => None,
        }
    }
    match cur {
        Resolved::Place {
            frame_pos,
            local,
            projection,
        } => {
            let (projection, depth) = strip(projection)?;
            Some((
                Resolved::Place {
                    frame_pos: *frame_pos,
                    local: *local,
                    projection,
                },
                depth,
            ))
        }
        Resolved::Value(value, projection) => {
            let (projection, depth) = strip(projection)?;
            Some((Resolved::Value(value.clone(), projection), depth))
        }
        Resolved::Static { name, projection } => {
            let (projection, depth) = strip(projection)?;
            Some((
                Resolved::Static {
                    name: name.clone(),
                    projection,
                },
                depth,
            ))
        }
        Resolved::Alloc { id, projection } => {
            let (projection, depth) = strip(projection)?;
            Some((
                Resolved::Alloc {
                    id: *id,
                    projection,
                },
                depth,
            ))
        }
    }
}

fn compose_fat_index(cur: &Resolved, idx: usize) -> Option<Resolved> {
    fn advanced(projection: &[RefProj], idx: usize) -> Option<Vec<RefProj>> {
        // The element index the composition advances: the
        // projection's last index step, with only whole-value
        // entry steps (fields, downcasts) after it.
        let mut pos = projection.len();
        loop {
            match projection.get(pos.wrapping_sub(1)) {
                // Only a whole-value entry step (a transparent
                // wrapper's field 0, a downcast) can follow the
                // element index here; any other field step selects
                // a field of the element the index denotes (a
                // node's key array, say), and composing the new
                // index into that outer index would denote the
                // field itself rather than its element.
                Some(RefProj::Field(0)) | Some(RefProj::Downcast(_)) => {
                    pos -= 1;
                }
                Some(RefProj::Index(_)) => {
                    pos -= 1;
                    break;
                }
                _ => return None,
            }
        }
        let Some(RefProj::Index(k)) = projection.get(pos) else {
            return None;
        };
        let mut out = projection.to_vec();
        out[pos] = RefProj::Index(k.checked_add(idx)?);
        Some(out)
    }
    match cur {
        Resolved::Place {
            frame_pos,
            local,
            projection,
        } => Some(Resolved::Place {
            frame_pos: *frame_pos,
            local: *local,
            projection: advanced(projection, idx)?,
        }),
        Resolved::Static { name, projection } => Some(Resolved::Static {
            name: name.clone(),
            projection: advanced(projection, idx)?,
        }),
        Resolved::Alloc { id, projection } => Some(Resolved::Alloc {
            id: *id,
            projection: advanced(projection, idx)?,
        }),
        Resolved::Value(..) => None,
    }
}

/// The sub-aggregate a `Subslice` projection step denotes over
/// an element container value: the elements `offset..offset+len`
/// of a tuple, an ADT value whose fields are its elements, a byte
/// string or a string. Anything else (or an out-of-range window)
/// answers `None`.
pub(crate) fn subslice_value(value: &Value, offset: usize, len: usize) -> Option<Value> {
    let end = offset.checked_add(len)?;
    match value {
        Value::Tuple(items) => Some(Value::Tuple(items.get(offset..end)?.to_vec())),
        Value::Enum { fields, .. } => {
            Some(Value::Tuple(fields.get(offset..end)?.to_vec()))
        }
        Value::Bytes(b) => Some(Value::Bytes(b.get(offset..end)?.to_vec())),
        Value::Str(text) => Some(Value::Str(text.get(offset..end)?.to_string())),
        // A sparse vector value: the window's elements as a
        // tuple, as for the aggregate forms above -- the stored
        // elements as themselves and the unmaterialised tail as
        // the uninitialised slots it denotes. The window is
        // bounded by the declared length, as in the projectors.
        // A window that runs past the stored elements stays in
        // the sparse form (its stored prefix rebased to the
        // window, its declared length the window's) rather than
        // materialising one slot per element, as in the
        // projectors.
        Value::SparseVec { elems, len, .. } => {
            if end as u128 > *len {
                return None;
            }
            if end <= elems.len() {
                Some(Value::Tuple(
                    (offset..end)
                        .map(|i| elems.get(i).cloned().unwrap_or(Value::Uninit))
                        .collect(),
                ))
            } else {
                Some(Value::SparseVec {
                    elems: if offset < elems.len() {
                        elems[offset..].to_vec()
                    } else {
                        Vec::new()
                    },
                    len: (end - offset) as u128,
                    cap: (end - offset) as u128,
                })
            }
        }
        _ => None,
    }
}

/// Whether a projection contains an index step immediately
/// followed by a step that indexes further (another index, or
/// a sub-slice): the only shape [`compose_projection`] rewrites.
pub(crate) fn projection_needs_composition(projection: &[RefProj]) -> bool {
    projection.windows(2).any(|w| {
        matches!(w[0], RefProj::Index(_))
            && matches!(w[1], RefProj::Index(_) | RefProj::Subslice { .. })
    })
}

/// Whether an index step landing on `element` ranges inside
/// it (the following steps of an index run nest as written)
/// rather than denoting the next position of the sequence
/// holding it (the run composes): the element is itself a
/// sequence or an aggregate of elements. The `MaybeUninit`
/// union form over a scalar payload is the one wrapper shape
/// that does not nest — its payload is a scalar value, which
/// no index step ranges inside.
fn element_nests_index(element: &Value) -> bool {
    match element {
        Value::Tuple(_)
        | Value::FatRef { .. }
        | Value::Bytes(_)
        | Value::Str(_)
        | Value::SparseVec { .. } => true,
        Value::Enum { variant, fields } => {
            !(*variant == 0
                && fields.len() == 1
                && matches!(
                    fields[0],
                    Value::Bool(_)
                        | Value::Int(_)
                        | Value::Uint(_)
                        | Value::Float(_)
                        | Value::Float128(_)
                ))
        }
        _ => false,
    }
}

/// Compose the element-position index runs of a projection.
///
/// An index step that lands on one element of a sequence,
/// followed by steps that index further, denotes a position
/// in the sequence rather than a read of the element: the
/// first step came from a data pointer into the sequence (a
/// raw-parts slice's base, an `as_ptr` result), and the
/// following steps address the sequence's elements from that
/// position, so the run composes into a single index (or, with
/// a trailing sub-slice step, a sub-slice) from the sequence's
/// start. When the element is itself a sequence value the
/// steps nest as written (a row of a matrix, one string of a
/// table) - the same judgment `seq_element_target` makes for
/// sequence reads through an element pointer. Steps the walk
/// cannot evaluate pass through unchanged; the caller's own
/// application then reproduces their result or error.
pub(crate) fn compose_projection(
    value: &Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> Vec<RefProj> {
    let mut out: Vec<RefProj> = Vec::with_capacity(projection.len());
    let mut cur = value.clone();
    let mut i = 0;
    while i < projection.len() {
        if let RefProj::Index(a) = projection[i] {
            let mut total = a;
            let mut j = i + 1;
            while let Some(RefProj::Index(b)) = projection.get(j) {
                total += b;
                j += 1;
            }
            let subslice = match projection.get(j) {
                Some(RefProj::Subslice { offset, len }) => Some((*offset, *len)),
                _ => None,
            };
            if j > i + 1 || subslice.is_some() {
                let element = project_value(cur.clone(), &[RefProj::Index(a)], func, place);
                if let Ok(element) = element {
                    // An ADT value is the aggregate of its
                    // fields, and an index step into it is
                    // element access over them (a `Vec` value's
                    // elements, above all): a following index
                    // step ranges inside it, exactly as for a
                    // tuple, so the run does not compose. The
                    // sparse form of a vector value is the same
                    // element aggregate (only its declared
                    // length differs from the stored list), so
                    // it nests the same way: a `Vec` whose
                    // length was rewritten in place (a `pop`'s
                    // shrink, a `set_len`) must not have the
                    // steps into its elements composed into
                    // positions of the sequence holding it.
                    // The one wrapper form that is not an
                    // element aggregate is the `MaybeUninit`
                    // union form — a one-field variant-0 wrapper
                    // around a scalar payload, the shape a bulk
                    // copy leaves in an element slot: the
                    // element is the scalar in wrapper form,
                    // and an index step cannot range inside a
                    // scalar, so the run composes as it does
                    // for the bare scalar.
                    if !element_nests_index(&element) {
                        let merged = match subslice {
                            Some((offset, len)) => RefProj::Subslice {
                                offset: total + offset,
                                len,
                            },
                            None => RefProj::Index(total),
                        };
                        if let Ok(next) = project_value(
                            cur.clone(),
                            std::slice::from_ref(&merged),
                            func,
                            place,
                        ) {
                            cur = next;
                            out.push(merged);
                            i = if subslice.is_some() { j + 1 } else { j };
                            continue;
                        }
                    }
                }
            }
        }
        match project_value(cur.clone(), std::slice::from_ref(&projection[i]), func, place) {
            Ok(next) => {
                cur = next;
                out.push(projection[i].clone());
                i += 1;
            }
            Err(_) => {
                out.extend_from_slice(&projection[i..]);
                return out;
            }
        }
    }
    out
}

/// A fat pointer's base re-expressed at the sequence's first
/// element: an index step appended to the base reference's
/// projection (see `Interpreter::fat_base_denotes_sequence`).
/// A base that is not a reference passes through unchanged (the
/// caller only routes reference bases here).
pub(crate) fn fat_base_first_element(base: Value) -> Value {
    match base {
        Value::Ref(RefTarget::Place {
            frame,
            local,
            mut projection,
        }) => {
            projection.push(RefProj::Index(0));
            Value::Ref(RefTarget::Place {
                frame,
                local,
                projection,
            })
        }
        Value::Ref(RefTarget::Static {
            name,
            mut projection,
        }) => {
            projection.push(RefProj::Index(0));
            Value::Ref(RefTarget::Static { name, projection })
        }
        Value::Ref(RefTarget::Alloc {
            id,
            mut projection,
        }) => {
            projection.push(RefProj::Index(0));
            Value::Ref(RefTarget::Alloc { id, projection })
        }
        Value::Ref(RefTarget::Value(v, mut projection)) => {
            projection.push(RefProj::Index(0));
            Value::Ref(RefTarget::Value(v, projection))
        }
        other => other,
    }
}

/// The run-time half of [`Rvalue::PtrCastUnits`]: rewrite a
/// trailing element index in the pointer's projection as the
/// byte offset it denotes in the source pointee's units
/// (`index * from_size`). Every other shape — no trailing
/// index, a byte offset already, a zero index (the same
/// position in any unit), an index whose byte position does
/// not fit — passes through unchanged: the position is the
/// same, only its denomination was ambiguous.
pub(crate) fn ptr_cast_units_value(
    value: Value,
    from_size: u64,
    typed_alloc: &dyn Fn(u64) -> bool,
) -> Value {
    fn rescale(projection: &mut Vec<RefProj>, from_size: u64) {
        if let Some(RefProj::Index(k)) = projection.last() {
            let k = *k;
            if k > 0 {
                if let Some(bytes) = (k as u128).checked_mul(u128::from(from_size)) {
                    if let Ok(d) = i128::try_from(bytes) {
                        *projection.last_mut().expect("last checked") =
                            RefProj::ByteOffset(d);
                    }
                }
            }
        }
    }
    match value {
        Value::Ref(target) => Value::Ref(match target {
            // Only a raw allocation block's index is
            // denominated in the pointer's own pointee units:
            // a raw block has no declared element type, so the
            // projection the pointer carries is the only
            // denomination in play. An allocation whose
            // recorded declared type is a sequence denominates
            // its indices itself (see `alloc_index_is_typed`),
            // as a place, static or value target's index
            // belongs to the declared aggregate it projects
            // into (an array local's element index); a pointer
            // cast redenominates neither.
            RefTarget::Alloc { id, mut projection } => {
                if !typed_alloc(id) {
                    rescale(&mut projection, from_size);
                }
                RefTarget::Alloc { id, projection }
            }
            other => other,
        }),
        Value::FatRef { base, len } => Value::FatRef {
            base: Box::new(ptr_cast_units_value(*base, from_size, typed_alloc)),
            len,
        },
        Value::DynRef { base, concrete, traits, drop_glue, type_id } => Value::DynRef {
            base: Box::new(ptr_cast_units_value(*base, from_size, typed_alloc)),
            concrete,
            traits,
            drop_glue,
            type_id,
        },
        other => other,
    }
}

/// The depth a buffer-wrapper type sits at in a `Vec`'s buffer
/// wrapper chain (see `resolve_place_mode`): the `Vec`'s buffer
/// field is the `RawVec` (depth 1), whose inner field is the
/// `RawVecInner` (depth 2), whose pointer field is the `Unique`
/// (depth 3), whose pointer field is the `NonNull` (depth 4)
/// around the buffer pointer itself. Matched on the defining
/// path, with or without the generic arguments the rendered
/// name carries.
pub(crate) fn buf_chain_depth(ty: &Ty) -> Option<usize> {
    let Ty::Adt(name) = ty else { return None };
    let base = name.split('<').next().unwrap_or(name.as_str());
    match base {
        "alloc::raw_vec::RawVec" => Some(1),
        "alloc::raw_vec::RawVecInner" => Some(2),
        "core::ptr::unique::Unique" => Some(3),
        "core::ptr::non_null::NonNull" => Some(4),
        _ => None,
    }
}

/// Whether `ty` names one of the std newtype wrappers whose
/// single field is the wrapped value itself (`ManuallyDrop`,
/// `Pin`, `AssertUnwindSafe`): a field step out of one is a
/// descent into the payload, never a read of the wrapper's
/// own storage. Matched on the defining path, with or
/// without the generic arguments the rendered name carries.
pub(crate) fn transparent_wrapper_ty(ty: &Ty) -> bool {
    let Ty::Adt(name) = ty else { return false };
    let base = name.split('<').next().unwrap_or(name.as_str());
    matches!(
        base,
        "std::mem::ManuallyDrop"
            | "core::mem::ManuallyDrop"
            | "std::pin::Pin"
            | "core::pin::Pin"
            | "std::panic::AssertUnwindSafe"
            | "core::panic::AssertUnwindSafe"
    )
}

/// Whether the type is a `Vec` ADT instance, under either its
/// `std` or its `alloc` spelling.
pub(crate) fn is_vec_adt_ty(ty: &Ty) -> bool {
    match ty {
        Ty::Adt(name) => {
            name == "std::vec::Vec"
                || name == "alloc::vec::Vec"
                || vec_spelling(name, "std::vec::Vec<")
                || vec_spelling(name, "alloc::vec::Vec<")
        }
        _ => false,
    }
}

/// Whether `name` is exactly `prefix` (which ends in `<`)
/// followed by a balanced generic argument list and nothing
/// else. A type nested in the named type's impl block
/// (`Vec<T, A>::dedup_by::FillGapOnDrop`) carries the same
/// prefix but continues past the argument list, and is not
/// the type itself.
fn vec_spelling(name: &str, prefix: &str) -> bool {
    let Some(rest) = name.strip_prefix(prefix) else {
        return false;
    };
    let mut depth = 1usize;
    for (idx, ch) in rest.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return idx + ch.len_utf8() == rest.len();
                }
            }
            _ => {}
        }
    }
    false
}

/// Whether a one-field wrapper's only field is an aggregate the
/// transparent-wrapper descent may enter: a tuple, a multi-field
/// aggregate, or another wrapper in the chain. A value-holding
/// enum (a unit variant, or a wrapper around a scalar) is NOT
/// entered: the wrapper itself is the aggregate being addressed,
/// so an out-of-range field step grows it instead of burrowing
/// into the value it happens to hold (a leaf node's `parent:
/// Option<_>` None value must not become the leaf's storage).
pub(crate) fn wrapper_payload(v: &Value) -> bool {
    match v {
        Value::Tuple(_) => true,
        Value::Enum { fields, .. } => {
            fields.len() >= 2
                || (fields.len() == 1
                    && wrapper_payload(&fields[0]))
        }
        _ => false,
    }
}

/// The element count a stored capacity value carries: the
/// integer itself, or the single field of the one-field
/// aggregate (`Cap`) wrapping it.
pub(crate) fn cap_value_count(value: &Value) -> Option<usize> {
    match value {
        Value::Uint(n) => usize::try_from(*n).ok(),
        Value::Int(n) if *n >= 0 => usize::try_from(*n).ok(),
        Value::Enum { fields, .. } if fields.len() == 1 => {
            cap_value_count(&fields[0])
        }
        Value::Tuple(fields) if fields.len() == 1 => cap_value_count(&fields[0]),
        _ => None,
    }
}

/// The base place of the vector itself when `place` denotes a
/// `Vec`'s length slot: a final `Field(1)` step whose base is
/// `Vec`-typed, with the type tracked from the local through
/// reference steps exactly as `resolve_place_mode` tracks it
/// (any other step ends the tracking, so the slot is only
/// recognised in the direct shapes collected std bodies use).
/// Store `value` into the buffer bookkeeping field of a
/// vector value in its real aggregate form, following the
/// chain `steps` (the field steps after the dereference that
/// entered the buffer wrapper chain): from the vector value
/// the steps name the `RawVec` field, the `RawVecInner`
/// inside it, and the pointer (field 0) or capacity
/// (field 1) slot inside that, through the one-field
/// wrappers around each. `false` when the value does not
/// have the shape the steps denote.
pub(crate) fn aggregate_buf_slot_write(
    slot: &mut Value,
    steps: &[ProjectionElem],
    value: &Value,
) -> bool {
    fn field_mut<'a>(v: &'a mut Value, idx: usize) -> Option<&'a mut Value> {
        match v {
            Value::Enum { fields, .. } => fields.get_mut(idx),
            _ => None,
        }
    }
    fn walk<'a>(v: &'a mut Value, idxs: &[usize]) -> Option<&'a mut Value> {
        match idxs.split_first() {
            None => Some(v),
            Some((idx, tail)) => walk(field_mut(v, *idx)?, tail),
        }
    }
    // The vector's `RawVec` field and the `RawVecInner`
    // inside it precede the steps, which are the inner's
    // own field steps (see `write_vec_buf_slot`).
    let mut idxs: Vec<usize> = vec![0, 0];
    for step in steps {
        let ProjectionElem::Field(idx) = step else {
            return false;
        };
        idxs.push(*idx);
    }
    match walk(slot, &idxs) {
        Some(cur) => {
            *cur = value.clone();
            true
        }
        None => false,
    }
}

/// The declared type a projection prefix reaches from a
/// local's declared type, tracked through reference, field
/// and downcast steps with the program's field-type table.
/// This is the tracking the length-slot shape judgments
/// (`vec_len_slot_base`, `vec_elem_tag`) need when the vector
/// sits behind a field of the place's root -- a struct's
/// `Vec` field above all: an inlined `Vec::pop` inside a
/// collected std body writes `(*self).data.len`, whose prefix
/// crosses the owner's field step, and a walk that only
/// follows references never reaches the vector. A step the
/// table cannot type (a field whose recorded type mentions
/// a generic parameter at its head, an index step) ends the
/// tracking (`None`).
fn prefix_result_ty(
    program: &Program,
    body: &Body,
    local: Local,
    prefix: &[ProjectionElem],
) -> Option<Ty> {
    let mut cur_ty = body.local_ty(local)?.clone();
    let mut variant = 0usize;
    for elem in prefix {
        match elem {
            ProjectionElem::Deref => {
                cur_ty = match cur_ty {
                    Ty::Ref(pointee) => (*pointee).clone(),
                    _ => return None,
                };
            }
            ProjectionElem::Field(idx) => {
                cur_ty = adt_field_ty_in(program, &cur_ty, variant, *idx)?;
                variant = 0;
            }
            ProjectionElem::Downcast(v) => {
                variant = *v as usize;
            }
            _ => return None,
        }
    }
    Some(cur_ty)
}

pub(crate) fn vec_len_slot_base(
    program: &Program,
    body: &Body,
    place: &Place,
) -> Option<Place> {
    let (last, prefix) = place.projection.split_last()?;
    if !matches!(last, ProjectionElem::Field(1)) {
        return None;
    }
    let cur_ty = prefix_result_ty(program, body, place.local, prefix)?;
    if is_vec_adt_ty(&cur_ty) {
        let mut base = place.clone();
        base.projection = prefix.to_vec();
        Some(base)
    } else {
        None
    }
}

/// The rendered element type tag of the `Vec` a base place
/// denotes: the first generic argument of the vector's ADT
/// name, with the base's type tracked from the local through
/// the prefix steps exactly as `vec_len_slot_base` tracks it.
/// `None` when the type does not resolve to a `Vec` instance.
pub(crate) fn vec_elem_tag(
    program: &Program,
    body: &Body,
    base: &Place,
) -> Option<String> {
    let cur_ty = prefix_result_ty(program, body, base.local, &base.projection)?;
    if !is_vec_adt_ty(&cur_ty) {
        return None;
    }
    let Ty::Adt(name) = cur_ty else {
        return None;
    };
    let open = name.find('<')?;
    let close = name.rfind('>')?;
    split_generic_args(&name[open + 1..close]).into_iter().next()
}

/// Whether some strict prefix of `projection` lands on a
/// coroutine value inside `root`: the remaining steps then
/// address cells of that coroutine's state (its upvars and
/// state-saved locals), whose only later readers are the
/// coroutine's own drop glue. A prefix that cannot be walked
/// (an uninitialised or mismatched intermediate) answers no.
pub(crate) fn projection_enters_coroutine(root: &Value, projection: &[RefProj], func: &str) -> bool {
    for end in 0..projection.len() {
        match project_value(root.clone(), &projection[..end], func, "<drop glue>") {
            Ok(Value::Coroutine { .. }) => return true,
            Ok(_) => {}
            Err(_) => return false,
        }
    }
    false
}

/// The pointee type of a bare-local pointer operand: a reference
/// or raw-pointer local with no projection, whose declared type
/// says what it points at. Anything else (a constant, a projected
/// place, a non-pointer local) has no statically known pointee
/// here.
pub(crate) fn operand_pointee_ty(body: &Body, op: &Operand) -> Option<Ty> {
    let (Operand::Copy(place) | Operand::Move(place)) = op else {
        return None;
    };
    if !place.projection.is_empty() {
        return None;
    }
    match body.local_ty(place.local) {
        Some(Ty::Ref(inner)) => Some((**inner).clone()),
        _ => None,
    }
}

/// The pointee type of a bare-local pointer operand for
/// *address pricing* (see [`local_pointee_ty`]): like
/// [`operand_pointee_ty`], but an erased local's leaf is the
/// pointee its value carried before the erasure. Only the
/// address a pointer denotes is priced from this leaf, never
/// a transfer width: a copy through an erased pointer still
/// moves unit elements, exactly as its declared type says.
pub(crate) fn operand_address_pointee_ty(
    body: &Body,
    op: &Operand,
) -> Option<Ty> {
    let (Operand::Copy(place) | Operand::Move(place)) = op else {
        return None;
    };
    if !place.projection.is_empty() {
        return None;
    }
    local_pointee_ty(body, place.local)
}

/// The pointee type of a pointer local for *address pricing*,
/// from its declared type.
/// A local declared as a pointer to the unit type may hold a
/// value whose pointee the declaration erased: core's
/// `*const T::addr` (and its siblings on `*mut T` and
/// `NonNull<T>`) casts the pointer to `*const ()` before
/// transmuting it to an integer, so the erased local's declared
/// type denominates nothing and every element position of a
/// root-less target prices at the root's base. Where the body's
/// own assignments prove the value entered the erased local
/// through a cast (or a copy chain) from pointers of one and
/// the same non-unit pointee, that pointee is the denomination
/// the value's positions were formed under, and it is the
/// answer; anything else keeps the declared unit (a genuine
/// `*const ()` value's element positions are zero-width,
/// exactly as declared).
pub(crate) fn local_pointee_ty(body: &Body, local: Local) -> Option<Ty> {
    match body.local_ty(local) {
        Some(Ty::Ref(inner)) => {
            let pointee = (**inner).clone();
            if matches!(pointee, Ty::Unit) {
                erasure_source_pointee(body, local).or(Some(pointee))
            } else {
                Some(pointee)
            }
        }
        _ => None,
    }
}

/// Whether a callee name is a raw pointer `cast` method
/// (`*const T::cast` / `*mut T::cast`): a pure retype whose
/// result is the receiver value itself under a new pointee.
fn is_ptr_cast_callee(func: &str) -> bool {
    func.ends_with(">::cast")
        && (func.contains("const_ptr::<impl *const T>")
            || func.contains("mut_ptr::<impl *mut T>"))
}

/// The pointee an erased (unit-pointee) pointer local's value
/// carried before its erasure, as proven by the body's own
/// assignments (see [`local_pointee_ty`]): every assignment
/// that can produce the local's value is a cast to the erased
/// pointer type (a cast statement, a units cast, or a raw
/// pointer `cast` call), or a plain copy, of a pointer operand whose
/// formation pointee — its own declared pointee, chased through
/// further erased locals — is one and the same non-unit type.
/// `None` when the local is an argument (its value enters from
/// the caller, under a formation this body cannot see), when
/// any producer is a call result, a constant, or anything else
/// the chase cannot read, when the producers disagree, or when
/// the body is too large for the per-call scan (the pattern
/// this serves lives in leaf accessor bodies).
fn erasure_source_pointee(body: &Body, local: Local) -> Option<Ty> {
    fn source_pointee(
        body: &Body,
        op: &Operand,
        visited: &mut Vec<usize>,
    ) -> Option<Ty> {
        let (Operand::Copy(place) | Operand::Move(place)) = op else {
            return None;
        };
        if !place.projection.is_empty() {
            return None;
        }
        match body.local_ty(place.local) {
            Some(Ty::Ref(inner)) if !matches!(**inner, Ty::Unit) => {
                Some((**inner).clone())
            }
            Some(Ty::Ref(_)) => chase(body, place.local, visited),
            _ => None,
        }
    }

    fn chase(body: &Body, local: Local, visited: &mut Vec<usize>) -> Option<Ty> {
        // An argument's value enters from the caller: its
        // formation pointee is not this body's to know.
        if local.0 >= 1 && local.0 <= body.arg_count {
            return None;
        }
        if visited.contains(&local.0) {
            return None;
        }
        visited.push(local.0);
        let result = scan(body, local, visited);
        visited.pop();
        result
    }

    fn scan(body: &Body, local: Local, visited: &mut Vec<usize>) -> Option<Ty> {
        let mut found: Option<Ty> = None;
        let mut producers = 0usize;
        for block in &body.blocks {
            for stmt in &block.statements {
                match &stmt.kind {
                    StatementKind::Assign(place, rvalue)
                        if place.local == local && place.projection.is_empty() =>
                    {
                        producers += 1;
                        let candidate = match rvalue {
                            Rvalue::Use(op) => {
                                source_pointee(body, op, visited)?
                            }
                            // A units cast redenominates its
                            // result in the target pointee's
                            // units; the value it produces is
                            // the source pointer's, formed
                            // under the source operand's
                            // pointee.
                            Rvalue::PtrCastUnits(op, _) => {
                                source_pointee(body, op, visited)?
                            }
                            Rvalue::Cast(op, dest) => match dest {
                                // Only a cast *to* the erased
                                // pointer type is an erasure
                                // step; any other cast forms
                                // the pointer here, under no
                                // pointee this body records.
                                Ty::Ref(inner)
                                    if matches!(**inner, Ty::Unit) =>
                                {
                                    source_pointee(body, op, visited)?
                                }
                                _ => return None,
                            },
                            _ => return None,
                        };
                        match &found {
                            None => found = Some(candidate),
                            Some(prev) if *prev == candidate => {}
                            Some(_) => return None,
                        }
                    }
                    StatementKind::ShapedStore { place, .. }
                    | StatementKind::SetDiscriminant { place, .. }
                        if place.local == local
                            && place.projection.is_empty() =>
                    {
                        return None;
                    }
                    _ => {}
                }
            }
            // A call writing the local produces a value whose
            // formation this body cannot read — except a raw
            // pointer `cast` call, which is a pure retype: the
            // value it writes is its receiver's, formed under
            // the receiver's pointee.
            let destination = match &block.terminator {
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    ..
                } if destination.local == local
                    && destination.projection.is_empty() =>
                {
                    if !is_ptr_cast_callee(func) {
                        return None;
                    }
                    producers += 1;
                    let candidate =
                        source_pointee(body, args.first()?, visited)?;
                    match &found {
                        None => found = Some(candidate),
                        Some(prev) if *prev == candidate => {}
                        Some(_) => return None,
                    }
                    None
                }
                TerminatorKind::Call { destination, .. }
                | TerminatorKind::CallIndirect { destination, .. }
                | TerminatorKind::CallTrait { destination, .. }
                | TerminatorKind::CallBuiltin { destination, .. } => {
                    Some(destination)
                }
                _ => None,
            };
            if let Some(place) = destination {
                if place.local == local && place.projection.is_empty() {
                    return None;
                }
            }
        }
        if producers == 0 {
            return None;
        }
        found
    }

    let total: usize = body.blocks.iter().map(|b| b.statements.len()).sum();
    if total > 4096 {
        return None;
    }
    chase(body, local, &mut Vec::new())
}

#[cfg(test)]
mod resolved_route_tests {
    use super::*;
    use crate::ir::{Body, Place, Program, Ty, ZstPathStep};

    fn partial_struct() -> Value {
        Value::Enum {
            variant: 0,
            fields: vec![Value::Tuple(vec![Value::Int(10), Value::Int(20)])],
        }
    }

    /// A projected write into an allocation whose declared
    /// pointee type is on record (the type the box construction
    /// builtins record, or a caller's root type): the second
    /// field's write extends the struct's own field vector; it
    /// does not descend into the first field.
    #[test]
    fn typed_alloc_field_write_extends_the_struct() {
        let mut program = Program::default();
        program.adt_field_tys.insert(
            "probe::S".to_string(),
            vec![vec![
                Some(Ty::Tuple(vec![
                    Ty::Int(IntTy::I32),
                    Ty::Int(IntTy::I32),
                ])),
                Some(Ty::Int(IntTy::I32)),
            ]],
        );
        let interp = Interpreter::new(&program);
        let id = interp.alloc_insert(partial_struct());
        let mut frames: Vec<Frame> = Vec::new();
        let target = Resolved::Alloc {
            id,
            projection: vec![RefProj::Field(1)],
        };
        interp
            .write_resolved_typed(
                &mut frames,
                &target,
                Value::Int(30),
                Some(&Ty::Adt("probe::S".to_string())),
            )
            .expect("write");
        assert_eq!(
            interp.alloc_value(id),
            Some(Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Tuple(vec![Value::Int(10), Value::Int(20)]),
                    Value::Int(30),
                ],
            })
        );
    }

    /// A read-modify-write whose index step sits at the
    /// materialisation bound errors like the plain write does
    /// (see `write_bound_tests`), with the storage untouched.
    #[test]
    fn modify_resolved_at_materialisation_bound_errors() {
        let program = Program::default();
        let interp = Interpreter::new(&program);
        let id = interp.alloc_insert(Value::Enum {
            variant: 0,
            fields: vec![Value::Tuple(vec![Value::Int(1), Value::Int(2)])],
        });
        let mut frames: Vec<Frame> = Vec::new();
        let target = Resolved::Alloc {
            id,
            projection: vec![RefProj::Field(0), RefProj::Index(usize::MAX)],
        };
        let result = interp.modify_resolved(&mut frames, &target, |_| Ok(()));
        assert!(result.is_err(), "modify at the bound must error");
        assert_eq!(
            interp.alloc_value(id),
            Some(Value::Enum {
                variant: 0,
                fields: vec![Value::Tuple(vec![Value::Int(1), Value::Int(2)])],
            })
        );
    }

    fn s2_program() -> Program {
        let mut program = Program::new();
        program.adt_field_tys.insert(
            "probe::S2".to_string(),
            vec![vec![
                Some(Ty::Tuple(vec![
                    Ty::Int(IntTy::I32),
                    Ty::Int(IntTy::I32),
                ])),
                Some(Ty::Unit),
            ]],
        );
        let mut body = Body::new(
            "main",
            0,
            vec![Ty::Unit, Ty::Adt("probe::S2".to_string())],
            vec![],
        );
        let mut zst1 = std::collections::HashMap::new();
        zst1.insert(vec![ZstPathStep::Field(1)], Value::Unit);
        body.zst_fields = vec![std::collections::HashMap::new(), zst1];
        program.add_function(body);
        program
    }

    fn s2_frame() -> Frame {
        Frame {
            id: 0,
            func: "main".to_string(),
            locals: vec![Value::Uninit, partial_struct()],
            stack_bytes: 0,
            block: 0,
            stmt: 0,
            return_dest: None,
            return_block: None,
            unwind_payload: None,
            in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm: None,
        }
    }

    /// Successive projected writes to a local struct place:
    /// the second field's write extends the struct's own
    /// field vector; it does not descend into the first
    /// field's aggregate and corrupt it.
    #[test]
    fn local_field_write_extends_the_struct() {
        let mut program = Program::new();
        program.adt_field_tys.insert(
            "probe::S".to_string(),
            vec![vec![
                Some(Ty::Tuple(vec![Ty::Int(IntTy::I32), Ty::Int(IntTy::I32)])),
                Some(Ty::Int(IntTy::I32)),
            ]],
        );
        let body = Body::new(
            "main",
            0,
            vec![Ty::Unit, Ty::Adt("probe::S".to_string())],
            vec![],
        );
        program.add_function(body);
        let interp = Interpreter::new(&program);
        let mut frames = vec![Frame {
            id: 0,
            func: "main".to_string(),
            locals: vec![Value::Uninit, partial_struct()],
            stack_bytes: 0,
            block: 0,
            stmt: 0,
            return_dest: None,
            return_block: None,
            unwind_payload: None,
            in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm: None,
        }];
        let target = Resolved::Place {
            frame_pos: 0,
            local: 1,
            projection: vec![RefProj::Field(1)],
        };
        interp
            .write_resolved(&mut frames, &target, Value::Int(30))
            .expect("write");
        assert_eq!(
            frames[0].locals[1],
            Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Tuple(vec![Value::Int(10), Value::Int(20)]),
                    Value::Int(30),
                ],
            }
        );
        let readback = Resolved::Place {
            frame_pos: 0,
            local: 1,
            projection: vec![RefProj::Field(0), RefProj::Field(1)],
        };
        assert_eq!(
            interp.read_resolved(&frames, &readback).expect("read"),
            Value::Int(20),
        );
    }

    /// The borrowing read of a never-stored zero-sized field:
    /// the wrapper descent is judged against the struct's own
    /// declared type (the walk's root), refuses, and the
    /// recorded canonical value answers -- a sibling's contents
    /// are not the field's value. The owned read of the same
    /// place already answered this way.
    #[test]
    fn borrowed_read_of_elided_field_is_canonical() {
        let program = s2_program();
        let interp = Interpreter::new(&program);
        let body = program.get("main").unwrap();
        let frames = vec![s2_frame()];
        let place = Place::from_index(1).field(1);
        assert_eq!(
            interp.read_place(&frames, body, &place).expect("read"),
            Value::Unit,
            "the elided field must read as its canonical value"
        );
        let target = Resolved::Place {
            frame_pos: 0,
            local: 1,
            projection: vec![RefProj::Field(1)],
        };
        assert_eq!(
            interp
                .read_resolved_ref(&frames, &target)
                .expect("read")
                .into_owned(),
            Value::Unit,
        );
        assert_eq!(
            interp.read_resolved(&frames, &target).expect("read"),
            Value::Unit,
        );
    }

    fn e_program() -> Program {
        let mut program = Program::new();
        program.adt_field_tys.insert(
            "probe::E".to_string(),
            vec![
                vec![
                    Some(Ty::Tuple(vec![
                        Ty::Int(IntTy::I32),
                        Ty::Int(IntTy::I32),
                    ])),
                    Some(Ty::Unit),
                ],
                vec![Some(Ty::Int(IntTy::I32))],
            ],
        );
        let mut body = Body::new(
            "main",
            0,
            vec![Ty::Unit, Ty::Adt("probe::E".to_string())],
            vec![],
        );
        let mut zst1 = std::collections::HashMap::new();
        zst1.insert(
            vec![ZstPathStep::Downcast(0), ZstPathStep::Field(1)],
            Value::Unit,
        );
        body.zst_fields = vec![std::collections::HashMap::new(), zst1];
        program.add_function(body);
        program
    }

    /// The elided-field read of an enum place: the selected
    /// variant declares a second, zero-sized field that was
    /// never stored. The wrapper descent is judged against the
    /// *selected variant's* declared fields, exactly as for a
    /// struct: the stored first field is the variant's own, not
    /// a wrapper payload, so the descent refuses and the
    /// recorded canonical value answers -- a sibling's contents
    /// are not the field's value.
    #[test]
    fn borrowed_read_of_elided_enum_field_is_canonical() {
        let program = e_program();
        let interp = Interpreter::new(&program);
        let body = program.get("main").unwrap();
        let frames = vec![s2_frame()];
        let place = Place {
            local: Local(1),
            projection: vec![ProjectionElem::Downcast(0), ProjectionElem::Field(1)],
        };
        assert_eq!(
            interp.read_place(&frames, body, &place).expect("read"),
            Value::Unit,
            "the elided variant field must read as its canonical value"
        );
        let target = Resolved::Place {
            frame_pos: 0,
            local: 1,
            projection: vec![RefProj::Downcast(0), RefProj::Field(1)],
        };
        assert_eq!(
            interp
                .read_resolved_ref(&frames, &target)
                .expect("read")
                .into_owned(),
            Value::Unit,
        );
        assert_eq!(
            interp.read_resolved(&frames, &target).expect("read"),
            Value::Unit,
        );
    }

    /// A projected write to the second field's array element,
    /// with the struct's declared type on record: the growth
    /// pre-pass judges its wrapper-descent retry by the same
    /// type evidence as the navigation, so it does not grow
    /// an array inside the first field -- a sibling the write
    /// does not address.
    #[test]
    fn alloc_growth_prepass_keeps_sibling_array() {
        let mut program = Program::default();
        program.adt_field_tys.insert(
            "probe::S3".to_string(),
            vec![vec![
                Some(Ty::Tuple(vec![
                    Ty::Int(IntTy::I32),
                    Ty::Array(Box::new(Ty::Int(IntTy::I32)), 1),
                ])),
                Some(Ty::Array(Box::new(Ty::Int(IntTy::I32)), 4)),
            ]],
        );
        let interp = Interpreter::new(&program);
        let id = interp.alloc_insert(Value::Enum {
            variant: 0,
            fields: vec![Value::Tuple(vec![
                Value::Int(10),
                Value::Tuple(vec![Value::Int(20)]),
            ])],
        });
        let mut frames: Vec<Frame> = Vec::new();
        let target = Resolved::Alloc {
            id,
            projection: vec![RefProj::Field(1), RefProj::Index(3)],
        };
        interp
            .write_resolved_typed(
                &mut frames,
                &target,
                Value::Int(1000),
                Some(&Ty::Adt("probe::S3".to_string())),
            )
            .expect("write");
        assert_eq!(
            interp.alloc_value(id),
            Some(Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Tuple(vec![
                        Value::Int(10),
                        Value::Tuple(vec![Value::Int(20)]),
                    ]),
                    Value::Tuple(vec![
                        Value::Uninit,
                        Value::Uninit,
                        Value::Uninit,
                        Value::Int(1000),
                    ]),
                ],
            })
        );
    }

    /// A byte-sized typed write to an unwritten array field
    /// of a partially materialised struct: the byte-splice
    /// pre-pass navigates the head with the type evidence, so
    /// it cannot take the shape-only wrapper descent into the
    /// first field and splice the byte into that sibling's
    /// scalar storage; the ordinary navigation materialises
    /// the array field instead.
    #[test]
    fn byte_splice_prepass_keeps_sibling() {
        let mut program = Program::default();
        program.adt_field_tys.insert(
            "probe::S5".to_string(),
            vec![vec![
                Some(Ty::Tuple(vec![
                    Ty::Int(IntTy::I32),
                    Ty::Int(IntTy::I32),
                ])),
                Some(Ty::Array(Box::new(Ty::Uint(UintTy::U8)), 4)),
            ]],
        );
        let interp = Interpreter::new(&program);
        let id = interp.alloc_insert(Value::Enum {
            variant: 0,
            fields: vec![Value::Tuple(vec![Value::Int(10), Value::Int(20)])],
        });
        let mut frames: Vec<Frame> = Vec::new();
        let target = Resolved::Alloc {
            id,
            projection: vec![RefProj::Field(1), RefProj::Index(3)],
        };
        interp
            .write_resolved_typed(
                &mut frames,
                &target,
                Value::Uint(7),
                Some(&Ty::Adt("probe::S5".to_string())),
            )
            .expect("write");
        assert_eq!(
            interp.alloc_value(id),
            Some(Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Tuple(vec![Value::Int(10), Value::Int(20)]),
                    Value::Tuple(vec![
                        Value::Uninit,
                        Value::Uninit,
                        Value::Uninit,
                        Value::Uint(7),
                    ]),
                ],
            })
        );
    }

    /// Nested index steps into an array of partially
    /// materialised structs: the index run's head element is
    /// itself an aggregate, so composition leaves the steps
    /// nested as written, and the type evidence survives into
    /// the navigation (see `assign_projected`): writing the
    /// element struct's second field extends that struct; it
    /// does not descend into the first field.
    #[test]
    fn nested_index_field_write_extends_the_struct() {
        let mut program = Program::default();
        program.adt_field_tys.insert(
            "probe::S".to_string(),
            vec![vec![
                Some(Ty::Tuple(vec![
                    Ty::Int(IntTy::I32),
                    Ty::Int(IntTy::I32),
                ])),
                Some(Ty::Int(IntTy::I32)),
            ]],
        );
        let interp = Interpreter::new(&program);
        let id = interp.alloc_insert(Value::Tuple(vec![Value::Tuple(vec![
            partial_struct(),
        ])]));
        let mut frames: Vec<Frame> = Vec::new();
        let target = Resolved::Alloc {
            id,
            projection: vec![RefProj::Index(0), RefProj::Index(0), RefProj::Field(1)],
        };
        let root_ty = Ty::Array(
            Box::new(Ty::Array(Box::new(Ty::Adt("probe::S".to_string())), 1)),
            1,
        );
        interp
            .write_resolved_typed(&mut frames, &target, Value::Int(30), Some(&root_ty))
            .expect("write");
        assert_eq!(
            interp.alloc_value(id),
            Some(Value::Tuple(vec![Value::Tuple(vec![Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Tuple(vec![Value::Int(10), Value::Int(20)]),
                    Value::Int(30),
                ],
            }])]))
        );
    }

    /// A write through a pointer to a field inside an
    /// allocation whose declared type is on record: the
    /// caller's type evidence is the pointee type (`Inner`),
    /// but the resolved projection is rooted at the allocation
    /// (`Outer`), so the navigation roots at the allocation's
    /// own type. Writing the inner struct's second field
    /// extends that struct; it does not descend into the first
    /// field and corrupt it.
    #[test]
    fn interior_pointer_alloc_write_uses_the_allocations_type() {
        let mut program = Program::default();
        program.adt_field_tys.insert(
            "probe::Inner".to_string(),
            vec![vec![
                Some(Ty::Tuple(vec![
                    Ty::Int(IntTy::I32),
                    Ty::Int(IntTy::I32),
                ])),
                Some(Ty::Int(IntTy::I32)),
            ]],
        );
        program.adt_field_tys.insert(
            "probe::Outer".to_string(),
            vec![vec![
                Some(Ty::Int(IntTy::I32)),
                Some(Ty::Adt("probe::Inner".to_string())),
            ]],
        );
        let interp = Interpreter::new(&program);
        let id = interp.alloc_insert(Value::Enum {
            variant: 0,
            fields: vec![
                Value::Int(7),
                Value::Enum {
                    variant: 0,
                    fields: vec![Value::Tuple(vec![Value::Int(10), Value::Int(20)])],
                },
            ],
        });
        interp.record_alloc_ty(id, Ty::Adt("probe::Outer".to_string()));
        let mut frames: Vec<Frame> = Vec::new();
        let target = Resolved::Alloc {
            id,
            projection: vec![RefProj::Field(1), RefProj::Field(1)],
        };
        interp
            .write_resolved_typed(
                &mut frames,
                &target,
                Value::Int(30),
                Some(&Ty::Adt("probe::Inner".to_string())),
            )
            .expect("write");
        assert_eq!(
            interp.alloc_value(id),
            Some(Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Int(7),
                    Value::Enum {
                        variant: 0,
                        fields: vec![
                            Value::Tuple(vec![Value::Int(10), Value::Int(20)]),
                            Value::Int(30),
                        ],
                    },
                ],
            })
        );
    }

    /// The unrecorded counterpart of
    /// `interior_pointer_alloc_write_uses_the_allocations_type`:
    /// the allocation's creation point recorded no declared
    /// type, so the only type evidence anywhere is the
    /// caller's pointee type (`Inner`). The shape-only wrapper
    /// judgment still roots the navigation at the allocation's
    /// own aggregate: writing the inner struct's second field
    /// extends that struct; it does not descend into the first
    /// field and corrupt it (issue #474, fixed by PR #613).
    #[test]
    fn interior_pointer_alloc_write_without_a_recorded_type() {
        let mut program = Program::default();
        program.adt_field_tys.insert(
            "probe::Inner".to_string(),
            vec![vec![
                Some(Ty::Tuple(vec![
                    Ty::Int(IntTy::I32),
                    Ty::Int(IntTy::I32),
                ])),
                Some(Ty::Int(IntTy::I32)),
            ]],
        );
        program.adt_field_tys.insert(
            "probe::Outer".to_string(),
            vec![vec![
                Some(Ty::Int(IntTy::I32)),
                Some(Ty::Adt("probe::Inner".to_string())),
            ]],
        );
        let interp = Interpreter::new(&program);
        let id = interp.alloc_insert(Value::Enum {
            variant: 0,
            fields: vec![
                Value::Int(7),
                Value::Enum {
                    variant: 0,
                    fields: vec![Value::Tuple(vec![Value::Int(10), Value::Int(20)])],
                },
            ],
        });
        let mut frames: Vec<Frame> = Vec::new();
        let target = Resolved::Alloc {
            id,
            projection: vec![RefProj::Field(1), RefProj::Field(1)],
        };
        interp
            .write_resolved_typed(
                &mut frames,
                &target,
                Value::Int(30),
                Some(&Ty::Adt("probe::Inner".to_string())),
            )
            .expect("write");
        assert_eq!(
            interp.alloc_value(id),
            Some(Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Int(7),
                    Value::Enum {
                        variant: 0,
                        fields: vec![
                            Value::Tuple(vec![Value::Int(10), Value::Int(20)]),
                            Value::Int(30),
                        ],
                    },
                ],
            })
        );
    }

    /// The no-evidence counterpart: the allocation has no
    /// recorded type and the caller supplies no type evidence
    /// either (`write_resolved`). The shape-only judgment
    /// alone keeps the write rooted at the allocation's own
    /// aggregate, so the inner struct's second field is
    /// materialised and the first field stays intact
    /// (issue #474, fixed by PR #613).
    #[test]
    fn interior_pointer_alloc_write_without_any_type_evidence() {
        let mut program = Program::default();
        program.adt_field_tys.insert(
            "probe::Inner".to_string(),
            vec![vec![
                Some(Ty::Tuple(vec![
                    Ty::Int(IntTy::I32),
                    Ty::Int(IntTy::I32),
                ])),
                Some(Ty::Int(IntTy::I32)),
            ]],
        );
        program.adt_field_tys.insert(
            "probe::Outer".to_string(),
            vec![vec![
                Some(Ty::Int(IntTy::I32)),
                Some(Ty::Adt("probe::Inner".to_string())),
            ]],
        );
        let interp = Interpreter::new(&program);
        let id = interp.alloc_insert(Value::Enum {
            variant: 0,
            fields: vec![
                Value::Int(7),
                Value::Enum {
                    variant: 0,
                    fields: vec![Value::Tuple(vec![Value::Int(10), Value::Int(20)])],
                },
            ],
        });
        let mut frames: Vec<Frame> = Vec::new();
        let target = Resolved::Alloc {
            id,
            projection: vec![RefProj::Field(1), RefProj::Field(1)],
        };
        interp
            .write_resolved(&mut frames, &target, Value::Int(30))
            .expect("write");
        assert_eq!(
            interp.alloc_value(id),
            Some(Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Int(7),
                    Value::Enum {
                        variant: 0,
                        fields: vec![
                            Value::Tuple(vec![Value::Int(10), Value::Int(20)]),
                            Value::Int(30),
                        ],
                    },
                ],
            })
        );
    }

    /// Successive projected writes to a tuple local: the
    /// wrapper descent is judged against the tuple's declared
    /// element types, exactly as for a struct, so the second
    /// field's write extends the tuple's own storage; it does
    /// not descend into the first field's aggregate, and the
    /// unwritten field does not read as the first field's
    /// contents.
    #[test]
    fn tuple_field_write_extends_the_tuple() {
        let mut program = Program::new();
        let body = Body::new(
            "main",
            0,
            vec![
                Ty::Unit,
                Ty::Tuple(vec![
                    Ty::Tuple(vec![Ty::Int(IntTy::I32), Ty::Int(IntTy::I32)]),
                    Ty::Int(IntTy::I32),
                ]),
            ],
            vec![],
        );
        program.add_function(body);
        let interp = Interpreter::new(&program);
        let mut frames = vec![Frame {
            id: 0,
            func: "main".to_string(),
            locals: vec![
                Value::Uninit,
                Value::Enum {
                    variant: 0,
                    fields: vec![Value::Tuple(vec![Value::Int(10), Value::Int(20)])],
                },
            ],
            stack_bytes: 0,
            block: 0,
            stmt: 0,
            return_dest: None,
            return_block: None,
            unwind_payload: None,
            in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm: None,
        }];
        // The unwritten second field must not read as the
        // first field's contents.
        let unread = Resolved::Place {
            frame_pos: 0,
            local: 1,
            projection: vec![RefProj::Field(1)],
        };
        assert!(
            interp.read_resolved(&frames, &unread).is_err(),
            "the unwritten tuple field must not read through the wrapper descent"
        );
        let target = Resolved::Place {
            frame_pos: 0,
            local: 1,
            projection: vec![RefProj::Field(1)],
        };
        interp
            .write_resolved(&mut frames, &target, Value::Int(30))
            .expect("write");
        assert_eq!(
            frames[0].locals[1],
            Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Tuple(vec![Value::Int(10), Value::Int(20)]),
                    Value::Int(30),
                ],
            }
        );
        let readback = Resolved::Place {
            frame_pos: 0,
            local: 1,
            projection: vec![RefProj::Field(0), RefProj::Field(1)],
        };
        assert_eq!(
            interp.read_resolved(&frames, &readback).expect("read"),
            Value::Int(20),
        );
    }
}
