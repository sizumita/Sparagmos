//! Modelled Clone execution for values whose clone body is not converted.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    box_alloc_pointer, channel_handle_id, rc_handle_id,
    Frame, Interpreter, Resolved,
};
use crate::value::{RefProj, RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// `Iterator::next`: mutate the iterator in place through the
    /// `&mut` argument and return the `Option` value.
    /// A `Duration` value in the real field shape (`secs`,
    /// `nanos`), with the normalization `Duration::new` performs
    /// (a nanosecond count at or past one second carries into the
    /// seconds; the carry is checked, as in std).


    /// The model's `Clone::clone`: the clone of the value
    /// the `&self` argument denotes, or of the argument
    /// itself when it is not a plain reference (a fat pointer
    /// clones as the pointer it is). `self_ty` is the cloned
    /// type's tag; the cases keyed on it (the mpsc and
    /// reference-counted handle counts, the `Box` / `Vec`
    /// owned clones, the waker clone) are the ones a
    /// `Clone::clone` call observes. The `Cloned` iterator
    /// adapter's steps serve their item through this same
    /// clone, under the element tag the inner iterator's tag
    /// spells (empty when it does not: the tag-keyed cases
    /// then stay silent and the value-driven ones apply).
    pub(super) fn clone_modelled_value(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        name: &str,
        arg: &Value,
        receiver_glue: Option<&crate::ir::DropGlue>,
    ) -> InterpResult<Value> {
        // Cloning an mpsc handle creates another live
        // handle: the channel's handle count must move
        // with it, or disconnection would be observed
        // while a clone still lives.
        let chan_kind = if self_ty.starts_with("std::sync::mpsc::Sender<")
            || self_ty.starts_with("std::sync::mpsc::SyncSender<")
        {
            Some("sender")
        } else if self_ty.starts_with("std::sync::mpsc::Receiver<") {
            Some("receiver")
        } else {
            None
        };
        match arg {
            // `&self`: the clone is a copy of the pointee
            // value (which may itself be a reference value
            // when `Self` is a reference type).
            Value::Ref(_) => {
                let target = self.deref_value(frames, arg)?;
                let value = self.read_resolved(frames, &target)?;
                // A `VaList` clone is C `va_copy`: a second
                // list over the same arguments with its own
                // cursor, not another handle on the same
                // cursor.
                if let Value::VaList(id) = value {
                    return self.va_list_duplicate(id);
                }
                // A `Waker` / `LocalWaker` clone is not
                // a value copy: a waker built by
                // `from_raw` clones through its
                // `RawWakerVTable` clone entry (which is
                // where the reference count moves), and
                // the modelled `From` waker clones the
                // handle it carries.
                if self_ty == "std::task::Waker"
                    || self_ty == "std::task::LocalWaker"
                {
                    return self.clone_waker_value(frames, &value, name);
                }
                // A `Box` clone owns a fresh allocation
                // holding a clone of the pointee: copying
                // the box value itself would alias the
                // source allocation, so writes through the
                // clone would rewrite the original (and the
                // two boxes would free the same storage).
                // The pointee is cloned by
                // `clone_owned_value`, the same value-copy
                // reading this builtin applies everywhere,
                // with owned boxes re-allocated at every
                // level.
                if self_ty.starts_with("std::boxed::Box<") {
                    if let Some((id, projection)) = box_alloc_pointer(&value) {
                        let pointee = self.read_resolved(
                            frames,
                            &Resolved::Alloc {
                                id,
                                projection: projection.clone(),
                            },
                        )?;
                        // The pointee clones with its own tag
                        // known, the way an element of a
                        // boxed-slice conversion clones: a
                        // counted pointee moves its count, a
                        // collected `Clone` body runs, instead
                        // of the shape-only copy.
                        let inner_ty = generic_first_arg(self_ty, "std::boxed::Box<");
                        let pointee_glue =
                            receiver_glue.and_then(|g| g.pointee_plan());
                        let pointee_ref = RefTarget::Alloc { id, projection };
                        let cloned = self.clone_boxed_slice_element(
                            frames,
                            &pointee,
                            Some(&pointee_ref),
                            inner_ty,
                            pointee_glue.as_ref(),
                        )?;
                        let boxed = self.new_box(cloned);
                        if let Some(ty) =
                            generic_first_arg(self_ty, "std::boxed::Box<")
                                .and_then(ty_from_tag)
                        {
                            self.record_box_ty(&boxed, ty);
                        }
                        return Ok(boxed);
                    }
                }
                // A cloned closure owns its environment:
                // each capture clones with its declared tag
                // (the discipline `clone_typed_aggregate`
                // applies to the same value in the boxed-slice
                // conversion), so an owned capture re-owns
                // what it owns -- a boxed capture is a fresh
                // box, a counted capture moves its count --
                // instead of aliasing the source closure's
                // environment. An instance with no recorded
                // environment shape keeps the value copy
                // below.
                //
                // A direct `Clone::clone` call hands over the
                // receiver glue conversion baked for the
                // closure type (`receiver_glue`): when a
                // capture's clone panics, the captures cloned
                // before it drop with their own glue before
                // the panic propagates. A caller with no baked
                // glue keeps the discard treatment.
                if let Value::Closure { .. } = value {
                    let source_ref = match arg {
                        Value::Ref(target) => Some(target.clone()),
                        _ => None,
                    };
                    if let Some(cloned) = self.clone_typed_aggregate(
                        frames,
                        &value,
                        source_ref.as_ref(),
                        self_ty,
                        receiver_glue,
                    )? {
                        return Ok(cloned);
                    }
                }
                // A cloned container owns its elements: a
                // `Vec` value's fields are its elements in
                // this model, and a tuple's / array's fields
                // likewise, so the elements go through the
                // same owned clone (a `Vec<Box<T>>` clone
                // must not alias the source's boxes).
                let value = if self_ty.starts_with("std::vec::Vec<")
                    || self_ty.starts_with("alloc::vec::Vec<")
                {
                    // A `Vec` clone clones every element with
                    // the element type's own `Clone` semantics,
                    // exactly as the boxed-slice conversion
                    // clones a slice's elements (see
                    // `clone_boxed_slice_element`): a collected
                    // element `Clone` body runs once per element
                    // in order, counted handles move their
                    // counts, and a panicking element clone
                    // drops the elements cloned so far (the
                    // glue `receiver_glue` carries, baked for a
                    // `Vec` receiver in conversion) while the
                    // source vector stays intact. The elements
                    // pair with their live source references
                    // (see `clone_source_elements`), so a clone
                    // body that mutates through `&self` writes
                    // the source element, as a direct clone of
                    // that element would.
                    let elem_ty = generic_first_arg(self_ty, "std::vec::Vec<")
                        .or_else(|| generic_first_arg(self_ty, "alloc::vec::Vec<"));
                    let elem_glue = receiver_glue.and_then(|g| g.elem_plan());
                    match value {
                        Value::Enum { variant, fields } => {
                            let len = fields.len() as u128;
                            let elements = self.clone_source_elements(
                                frames.as_slice(),
                                arg,
                                fields,
                                len,
                                elem_ty,
                            );
                            let cloned = self.clone_boxed_slice_elements(
                                frames,
                                &elements,
                                elem_ty,
                                elem_glue.as_ref(),
                            )?;
                            Value::Enum {
                                variant,
                                fields: cloned,
                            }
                        }
                        // A sparse vector's stored
                        // elements clone the same way;
                        // its unmaterialised tail has
                        // nothing to clone. The clone is
                        // a fresh allocation of exactly
                        // the declared length (as
                        // `Vec::clone` allocates), so it
                        // accounts its length - the
                        // source's spare capacity does
                        // not carry over.
                        Value::SparseVec { elems, len, .. } => {
                            let elements = self.clone_source_elements(
                                frames.as_slice(),
                                arg,
                                elems,
                                len,
                                elem_ty,
                            );
                            let cloned = self.clone_boxed_slice_elements(
                                frames,
                                &elements,
                                elem_ty,
                                elem_glue.as_ref(),
                            )?;
                            let cap =
                                len.max(cloned.len() as u128);
                            Value::SparseVec {
                                elems: cloned,
                                len,
                                cap,
                            }
                        }
                        other => other,
                    }
                } else if (self_ty.starts_with('(') || self_ty.starts_with('['))
                    && matches!(value, Value::Tuple(_))
                {
                    // A tuple / array clone is the field-wise
                    // clone with each element's own tag, the
                    // discipline `clone_typed_aggregate`
                    // applies to the same shapes in the
                    // boxed-slice conversion: an element's
                    // collected `Clone` body runs instead of
                    // the shape-only copy. A shape the typed
                    // clone does not cover keeps the copy.
                    let source_ref = match arg {
                        Value::Ref(target) => Some(target.clone()),
                        _ => None,
                    };
                    match self.clone_typed_aggregate(
                        frames,
                        &value,
                        source_ref.as_ref(),
                        self_ty,
                        receiver_glue,
                    )? {
                        Some(cloned) => cloned,
                        None => match value {
                            Value::Tuple(items) => Value::Tuple(
                                items
                                    .iter()
                                    .map(|f| self.clone_owned_value(frames, f))
                                    .collect::<InterpResult<Vec<_>>>()?,
                            ),
                            other => other,
                        },
                    }
                } else {
                    value
                };
                if let (Some(kind), Some(id)) =
                    (chan_kind, channel_handle_id(&value))
                {
                    self.channel_handle_cloned(id, kind);
                }
                // Cloning a reference-counted handle makes
                // another live handle on the same
                // allocation: the count the converted
                // bodies read must move with it, or
                // uniqueness (`Rc::get_mut`, the counts,
                // drop) would be observed wrong.
                if self_ty.starts_with("std::rc::Rc<")
                    || self_ty.starts_with("std::sync::Arc<")
                {
                    if let Some(id) = rc_handle_id(&value) {
                        self.rc_inc_strong(id)?;
                    }
                } else if self_ty.starts_with("std::rc::Weak<")
                    || self_ty.starts_with("std::sync::Weak<")
                {
                    if let Some(id) = rc_handle_id(&value) {
                        self.rc_inc_weak(id)?;
                    }
                }
                Ok(value)
            }
            // A fat pointer clones as the pointer itself
            // (slices are `Clone` through their reference).
            other => Ok(other.clone()),
        }
    }

    /// Pair every element of a boxed-slice conversion's
    /// source snapshot with its live source reference (see
    /// [`Self::clone_source_element_ref`]), for the
    /// custom-`Clone` dispatch in
    /// [`Self::clone_boxed_slice_element`]. Scalar elements
    /// never dispatch to a `Clone` body and hold nothing that
    /// can, so their conversions name no references.
    pub(super) fn clone_source_elements(
        &self,
        frames: &[Frame],
        receiver: &Value,
        elements: Vec<Value>,
        len: u128,
        elem_ty: Option<&str>,
    ) -> Vec<(Value, Option<RefTarget>)> {
        let named = matches!(elem_ty, Some(ty) if !matches!(
            ty,
            "bool" | "char" | "str" | "()"
                | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
                | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
                | "f16" | "f32" | "f64" | "f128"
        ));
        elements
            .into_iter()
            .enumerate()
            .map(|(i, e)| {
                let source_ref = if named {
                    self.clone_source_element_ref(frames, receiver, i, len, &e)
                } else {
                    None
                };
                (e, source_ref)
            })
            .collect()
    }

    /// The live reference to element `idx` of the sequence a
    /// conversion is cloning from: the element's own place in
    /// the source storage (the resolution
    /// [`Self::index_place_ref`] performs for an indexed
    /// read), so a `Clone` body invoked on it that mutates
    /// through `&self` -- an inline `Cell` or `RefCell` --
    /// writes the source element, exactly as the same body
    /// called on the element directly would. `None` when the
    /// source is not place-backed, or when the place does not
    /// denote exactly the snapshotted element (an element
    /// whose store the source's MIR elided reaches the
    /// snapshot as its canonical value, not as the storage's
    /// uninitialised slot): the dispatch then keeps the
    /// snapshot argument it has always used. The denotation
    /// check is bit identity, not the guest's `==` (see
    /// [`Value::bit_identical`]): a NaN element read back
    /// from its own storage denotes its snapshot, though
    /// `==` holds it unequal to itself.
    fn clone_source_element_ref(
        &self,
        frames: &[Frame],
        receiver: &Value,
        idx: usize,
        len: u128,
        snapshot: &Value,
    ) -> Option<RefTarget> {
        let refr = self.index_place_ref(frames, receiver, idx, Some(len)).ok()??;
        let Value::Ref(target) = &refr else {
            return None;
        };
        let resolved = self.deref_value(frames, &refr).ok()?;
        let stored = self.read_resolved(frames, &resolved).ok()?;
        if stored.bit_identical(snapshot) {
            Some(target.clone())
        } else {
            None
        }
    }

    /// Re-read the value a clone source reference denotes
    /// right now, in place of the snapshot taken when the
    /// conversion began: the real field-wise construction
    /// reads each field and element from live storage as it
    /// reaches it, so a part cloned after an earlier part's
    /// `Clone` body ran observes that body's mutations
    /// through interior mutability (a sibling field's `Cell`
    /// the earlier clone rewrote, reached through a stored
    /// reference or pointer). The snapshot stays the value
    /// cloned when the target denotes no readable storage,
    /// or when the read fails: such a source has no live
    /// storage a clone body could have written through, so
    /// the snapshot is all there is to observe.
    fn clone_reread_source(
        &self,
        frames: &[Frame],
        target: &RefTarget,
        snapshot: &Value,
    ) -> Value {
        let resolved = match self.deref_value(frames, &Value::Ref(target.clone())) {
            Ok(resolved) => resolved,
            Err(_) => return snapshot.clone(),
        };
        self.read_resolved(frames, &resolved)
            .unwrap_or_else(|_| snapshot.clone())
    }

    /// Clone one element of a slice being re-owned by a
    /// `From<&[T]>` / borrowed-`Cow<[T]>` conversion to
    /// `Box<[T]>`, where the element type tag `elem_ty` is
    /// known. A reference-counted element (`Rc` / `Arc` /
    /// `Weak`) clones the way the `Clone::clone` model clones
    /// the handle itself: the clone is another live handle on
    /// the SAME allocation, with the matching count moved --
    /// the shape-only [`Self::clone_owned_value`] must not be
    /// applied to it, because its field-wise recursion reaches
    /// the handle's `NonNull` field, mistakes it for a built
    /// box, and deep-copies the counted allocation, which
    /// breaks allocation identity (`ptr_eq`, shared interior
    /// mutation, the counts themselves). A `Box<U>` element
    /// re-owns its pointee with the pointee's tag known, so a
    /// counted pointee is judged the same way one level down.
    /// An aggregate element is cloned field by field with
    /// each field's own tag propagated (see
    /// [`Self::clone_typed_aggregate`]), so a counted handle
    /// nested inside it is judged by the same discipline.
    /// A channel handle element (`Sender` / `SyncSender` /
    /// `Receiver`) likewise clones as another live handle on
    /// the same channel, with the channel's handle count
    /// moved the way [`Self::clone_modelled_value`] moves it.
    /// A `Waker` / `LocalWaker` element clones through
    /// [`Self::clone_waker_value`], the way
    /// [`Self::clone_modelled_value`] clones the same values:
    /// a `from_raw` waker's clone is its vtable's clone entry
    /// (which is where the reference count moves), so a
    /// shape-only copy would mint a second dropping handle
    /// that never acquired a reference.
    /// An element type whose `Clone` body the program
    /// collected clones through that body, exactly as a direct
    /// `Clone::clone` call on the element would (see the
    /// dispatch in the body below). The body's `&self` is the
    /// element's own source reference when the caller could
    /// name one (`value_ref`, propagated into fields by the
    /// aggregate recursion): a body that mutates through
    /// `&self` -- an inline `Cell` or `RefCell` -- writes the
    /// source element, as the same body called on the element
    /// directly would. Without a named reference the argument
    /// is the snapshot itself, the treatment such a body has
    /// always had here.
    /// Any other element goes through the shape-only owned
    /// clone. `glue` is the element type's drop glue when the
    /// caller baked one: the aggregate recursion below uses it
    /// to drop the parts already cloned when a later part's
    /// clone unwinds.
    fn clone_boxed_slice_element(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        value_ref: Option<&RefTarget>,
        elem_ty: Option<&str>,
        glue: Option<&crate::ir::DropGlue>,
    ) -> InterpResult<Value> {
        if let Some(ty) = elem_ty {
            if ty.starts_with("std::rc::Rc<")
                || ty.starts_with("alloc::rc::Rc<")
                || ty.starts_with("std::sync::Arc<")
                || ty.starts_with("alloc::sync::Arc<")
            {
                if let Some(id) = rc_handle_id(value) {
                    self.rc_inc_strong(id)?;
                    return Ok(value.clone());
                }
            } else if ty.starts_with("std::rc::Weak<")
                || ty.starts_with("alloc::rc::Weak<")
                || ty.starts_with("std::sync::Weak<")
                || ty.starts_with("alloc::sync::Weak<")
            {
                if let Some(id) = rc_handle_id(value) {
                    self.rc_inc_weak(id)?;
                    return Ok(value.clone());
                }
            } else if ty.starts_with("std::boxed::Box<")
                || ty.starts_with("alloc::boxed::Box<")
            {
                if let Some((id, projection)) = box_alloc_pointer(value) {
                    let pointee = self.read_resolved(
                        frames,
                        &Resolved::Alloc {
                            id,
                            projection: projection.clone(),
                        },
                    )?;
                    let inner_ty = generic_first_arg(ty, "std::boxed::Box<")
                        .or_else(|| generic_first_arg(ty, "alloc::boxed::Box<"));
                    let pointee_glue = glue.and_then(|g| g.pointee_plan());
                    // The pointee's own reference is its
                    // allocation target: a `Clone` body invoked
                    // on it sees the pointee's storage.
                    let pointee_ref = RefTarget::Alloc { id, projection };
                    let cloned = self.clone_boxed_slice_element(
                        frames,
                        &pointee,
                        Some(&pointee_ref),
                        inner_ty,
                        pointee_glue.as_ref(),
                    )?;
                    return Ok(self.new_box(cloned));
                }
            } else if ty.starts_with("std::sync::mpsc::Sender<")
                || ty.starts_with("std::sync::mpsc::SyncSender<")
                || ty.starts_with("std::sync::mpsc::Receiver<")
            {
                // A channel handle clones as another live
                // handle on the same channel: the channel's
                // handle count must move with the clone, or
                // dropping the clone would disconnect the
                // channel while the source handle still
                // lives (the discipline `clone_modelled_value`
                // applies to the same handles).
                if let Some(id) = channel_handle_id(value) {
                    let kind = if ty.starts_with("std::sync::mpsc::Receiver<") {
                        "receiver"
                    } else {
                        "sender"
                    };
                    self.channel_handle_cloned(id, kind);
                    return Ok(value.clone());
                }
            } else if ty == "std::task::Waker" || ty == "std::task::LocalWaker" {
                // A waker clone is not a value copy: the
                // modelled `From` waker clones the handle it
                // carries, and a `from_raw` waker clones
                // through its `RawWakerVTable` clone entry --
                // the discipline `clone_waker_value`
                // implements and `clone_modelled_value`
                // delegates to for the same values. A
                // shape-only copy would create another
                // dropping handle without acquiring another
                // reference, so dropping the boxed slice
                // could release the waker's data while the
                // source waker still lives.
                return self.clone_waker_value(frames, value, "From::from to Box<[T]>");
            }
            // A collected `Clone` body for the element type is
            // the element's real clone: a direct `clone()` call
            // on such an element runs the body (its presence is
            // what makes the call dispatch prefer it over the
            // builtin model), so the element clone here runs it
            // too — the structural clone below would silently
            // bypass a user-defined `Clone` impl, returning the
            // source's field values where the impl computes new
            // ones and never raising the impl's panics. Only
            // plain named types reach this judgment: tuples,
            // arrays and references keep the structural paths
            // (their `Clone` is the field-wise one by
            // definition, and no body is ever collected under
            // these names). A type whose `Clone` body was not
            // collected falls through to the structural clone,
            // the treatment it has always had.
            if !ty.starts_with(['(', '[', '&']) {
                for spelling in ["std::clone::Clone", "core::clone::Clone"] {
                    let name = format!("<{ty} as {spelling}>::clone");
                    if self.program().get(&name).is_some() {
                        // The body's `&self`: the element's own
                        // source reference when the caller named
                        // one, so a body that mutates through
                        // `&self` writes the source element; the
                        // snapshot itself otherwise.
                        let arg = match value_ref {
                            Some(target) => Value::Ref(target.clone()),
                            None => Value::Ref(RefTarget::value(value.clone())),
                        };
                        return self.call_sync(frames, &name, vec![arg]);
                    }
                }
            }
            if let Some(cloned) =
                self.clone_typed_aggregate(frames, value, value_ref, ty, glue)?
            {
                return Ok(cloned);
            }
        }
        self.clone_owned_value(frames, value)
    }

    /// Clone every element of a boxed-slice conversion's
    /// source snapshot (see [`Self::clone_boxed_slice_element`]),
    /// each element paired with its live source reference when
    /// the conversion could name one (see
    /// [`Self::clone_source_elements`]), dropping the elements
    /// cloned so far when a later
    /// element's clone unwinds with a guest panic: the real
    /// construction's guard drops the initialized prefix
    /// before the panic propagates. `elem_glue` is the
    /// conversion's baked element glue (see
    /// `TerminatorKind::CallBuiltin::elem_glue`); without it
    /// the prefix keeps the discard treatment it has always
    /// had.
    pub(super) fn clone_boxed_slice_elements(
        &mut self,
        frames: &mut Vec<Frame>,
        elements: &[(Value, Option<RefTarget>)],
        elem_ty: Option<&str>,
        elem_glue: Option<&crate::ir::DropGlue>,
    ) -> InterpResult<Vec<Value>> {
        let mut cloned: Vec<(Value, Option<crate::ir::DropGlue>)> =
            Vec::with_capacity(elements.len());
        for (e, e_ref) in elements {
            // The element is read from its live source right
            // before it is cloned: an earlier element's clone
            // may have rewritten it through interior
            // mutability (see `clone_reread_source`).
            let live = e_ref
                .as_ref()
                .map(|target| self.clone_reread_source(frames, target, e));
            let e = live.as_ref().unwrap_or(e);
            match self.clone_boxed_slice_element(
                frames,
                e,
                e_ref.as_ref(),
                elem_ty,
                elem_glue,
            ) {
                Ok(v) => cloned.push((v, elem_glue.cloned())),
                Err(err) => return Err(self.drop_cloned_prefix(frames, cloned, err)),
            }
        }
        Ok(cloned.into_iter().map(|(v, _)| v).collect())
    }

    /// Drop the values cloned so far in one construction (each
    /// with the glue its own part carries) after a later part's
    /// clone unwound with a guest panic, then hand back the
    /// error to propagate: the original panic, or the failure
    /// the cleanup itself met — a panic raised by the cleanup
    /// is panicking while unwinding and aborts, as it does
    /// natively. A non-panic failure of the original clone left
    /// nothing mid-construction to clean up: it propagates
    /// untouched. A part whose glue the caller could not name
    /// is discarded as before. The parts arrive in the order
    /// they drop: an initialized slice, vector or array prefix
    /// drops in element order, while a partially built tuple's
    /// or ADT's fields are construction temporaries, which
    /// drop in reverse initialization order -- those callers
    /// reverse their prefix before handing it over.
    fn drop_cloned_prefix(
        &mut self,
        frames: &mut Vec<Frame>,
        cloned: Vec<(Value, Option<crate::ir::DropGlue>)>,
        err: InterpError,
    ) -> InterpError {
        if !matches!(err, InterpError::Panic { .. }) {
            return err;
        }
        for (value, glue) in cloned {
            let Some(glue) = glue else { continue };
            match self.drop_value_sync(frames, value, &glue) {
                Ok(()) => {}
                Err(InterpError::Panic { message, .. }) => {
                    return InterpError::Abort {
                        message: format!(
                            "panic `{message}` raised while dropping a cloned \
                             prefix: panicking while unwinding aborts the run"
                        ),
                    };
                }
                Err(other) => return other,
            }
        }
        err
    }

    /// The aggregate case of [`Self::clone_boxed_slice_element`]:
    /// clone `value`, whose rendered type tag is `ty`, field by
    /// field with the fields' own tags propagated into the
    /// recursion, so a reference-counted handle nested inside
    /// the aggregate keeps its allocation identity instead of
    /// being deep-copied by the shape-only recursion. `Ok(None)`
    /// when the (value, tag) pair is not an aggregate shape
    /// this judgment covers -- the caller then applies the
    /// shape-only owned clone, exactly as before.
    ///
    /// The field tags come from the tag itself for tuples,
    /// arrays, slices and `Vec`s, and from the instance's recorded
    /// debug shape for any other ADT; a field whose tag cannot
    /// be determined is cloned shape-only. An ADT value
    /// carrying fewer fields than its recorded variant
    /// declares (a union's value carries only its live field)
    /// is left to the shape-only clone whole, since its field
    /// positions cannot be aligned with the declared ones.
    ///
    /// `value_ref` is the aggregate's own source reference
    /// when the caller could name one: each field's reference
    /// is the aggregate's with the field's step appended, so a
    /// field's `Clone` body sees the field's own storage.
    ///
    /// `glue` is the aggregate type's drop glue when the
    /// caller baked one: when a field's clone unwinds with a
    /// guest panic, the fields cloned so far drop with their
    /// own extracted glue before the panic propagates, the
    /// way the real field-wise construction's guard drops
    /// the initialized fields of the never-completed value.
    fn clone_typed_aggregate(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        value_ref: Option<&RefTarget>,
        ty: &str,
        glue: Option<&crate::ir::DropGlue>,
    ) -> InterpResult<Option<Value>> {
        // Tuples and arrays ride as a `Value::Tuple` of their
        // elements.
        if let Value::Tuple(items) = value {
            if let Some(inner) = ty.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
                let field_tys = split_top_level(inner);
                if field_tys.len() != items.len() {
                    return Ok(None);
                }
                let mut cloned: Vec<(Value, Option<crate::ir::DropGlue>)> =
                    Vec::with_capacity(items.len());
                for (i, (f, t)) in items.iter().zip(field_tys.iter()).enumerate() {
                    let field_glue = glue.and_then(|g| g.field_plan(None, i));
                    let field_ref =
                        value_ref.and_then(|t| t.projected(RefProj::Field(i)));
                    // The field is read from its live source
                    // right before it is cloned: an earlier
                    // field's clone may have rewritten it
                    // through interior mutability (see
                    // `clone_reread_source`).
                    let live = field_ref
                        .as_ref()
                        .map(|target| self.clone_reread_source(frames, target, f));
                    let f = live.as_ref().unwrap_or(f);
                    match self.clone_boxed_slice_element(
                        frames,
                        f,
                        field_ref.as_ref(),
                        Some(t),
                        field_glue.as_ref(),
                    ) {
                        Ok(v) => cloned.push((v, field_glue)),
                        Err(err) => {
                            // The fields cloned so far are
                            // construction temporaries: they drop
                            // in reverse initialization order.
                            cloned.reverse();
                            return Err(self.drop_cloned_prefix(frames, cloned, err));
                        }
                    }
                }
                return Ok(Some(Value::Tuple(
                    cloned.into_iter().map(|(v, _)| v).collect(),
                )));
            }
            if let Some(inner) = ty.strip_prefix('[').and_then(|t| t.strip_suffix(']')) {
                // An array tag `[T; N]` names its element
                // before the top-level semicolon; a slice tag
                // `[T]` -- the pointee of a `Box<[T]>` element,
                // reached through the box arm of the caller --
                // names it with the whole inside. Without the
                // slice reading, a boxed slice's pointee fell
                // to the shape-only clone, and its elements
                // lost the type-aware judgments (counted
                // handles deep-copied, custom `Clone` bodies
                // bypassed).
                let Some(elem) = array_elem_tag(inner).or_else(|| slice_pointee_elem_tag(ty)) else {
                    return Ok(None);
                };
                let elem_glue = glue.and_then(|g| g.elem_plan());
                let mut cloned: Vec<(Value, Option<crate::ir::DropGlue>)> =
                    Vec::with_capacity(items.len());
                for (i, f) in items.iter().enumerate() {
                    let elem_ref =
                        value_ref.and_then(|t| t.projected(RefProj::Index(i)));
                    let live = elem_ref
                        .as_ref()
                        .map(|target| self.clone_reread_source(frames, target, f));
                    let f = live.as_ref().unwrap_or(f);
                    match self.clone_boxed_slice_element(
                        frames,
                        f,
                        elem_ref.as_ref(),
                        Some(elem),
                        elem_glue.as_ref(),
                    ) {
                        Ok(v) => cloned.push((v, elem_glue.clone())),
                        Err(err) => {
                            // Not construction temporaries,
                            // unlike the tuple arm above: the
                            // real array or slice clone initializes
                            // element by element behind a guard
                            // that drops the initialized prefix
                            // in element order, as the slice and
                            // vector prefixes do (measured
                            // against native Rust).
                            return Err(self.drop_cloned_prefix(frames, cloned, err));
                        }
                    }
                }
                return Ok(Some(Value::Tuple(
                    cloned.into_iter().map(|(v, _)| v).collect(),
                )));
            }
            return Ok(None);
        }
        // A `Vec`'s elements all share the element tag, in
        // either of the value forms a vector takes. A
        // `Box<[T]>` pointee built by `into_boxed_slice`
        // keeps the vector's value form under the slice tag,
        // so a slice tag names the same shared element here.
        if let Some(elem) = generic_first_arg(ty, "std::vec::Vec<")
            .or_else(|| generic_first_arg(ty, "alloc::vec::Vec<"))
            .or_else(|| slice_pointee_elem_tag(ty))
        {
            let elem_glue = glue.and_then(|g| g.elem_plan());
            match value {
                Value::Enum { variant, fields } => {
                    let mut cloned: Vec<(Value, Option<crate::ir::DropGlue>)> =
                        Vec::with_capacity(fields.len());
                    for (i, f) in fields.iter().enumerate() {
                        let elem_ref =
                            value_ref.and_then(|t| t.projected(RefProj::Index(i)));
                        let live = elem_ref
                            .as_ref()
                            .map(|target| self.clone_reread_source(frames, target, f));
                        let f = live.as_ref().unwrap_or(f);
                        match self.clone_boxed_slice_element(
                            frames,
                            f,
                            elem_ref.as_ref(),
                            Some(elem),
                            elem_glue.as_ref(),
                        ) {
                            Ok(v) => cloned.push((v, elem_glue.clone())),
                            Err(err) => {
                                return Err(self.drop_cloned_prefix(frames, cloned, err))
                            }
                        }
                    }
                    return Ok(Some(Value::Enum {
                        variant: *variant,
                        fields: cloned.into_iter().map(|(v, _)| v).collect(),
                    }));
                }
                Value::SparseVec { elems, len, .. } => {
                    let mut cloned: Vec<(Value, Option<crate::ir::DropGlue>)> =
                        Vec::with_capacity(elems.len());
                    for (i, f) in elems.iter().enumerate() {
                        let elem_ref =
                            value_ref.and_then(|t| t.projected(RefProj::Index(i)));
                        let live = elem_ref
                            .as_ref()
                            .map(|target| self.clone_reread_source(frames, target, f));
                        let f = live.as_ref().unwrap_or(f);
                        match self.clone_boxed_slice_element(
                            frames,
                            f,
                            elem_ref.as_ref(),
                            Some(elem),
                            elem_glue.as_ref(),
                        ) {
                            Ok(v) => cloned.push((v, elem_glue.clone())),
                            Err(err) => {
                                return Err(self.drop_cloned_prefix(frames, cloned, err))
                            }
                        }
                    }
                    let cloned: Vec<Value> =
                        cloned.into_iter().map(|(v, _)| v).collect();
                    // The clone owns fresh storage of exactly
                    // the declared length, the discipline the
                    // `Clone::clone` model applies to the same
                    // shape.
                    let cap = (*len).max(cloned.len() as u128);
                    return Ok(Some(Value::SparseVec {
                        elems: cloned,
                        len: *len,
                        cap,
                    }));
                }
                _ => return Ok(None),
            }
        }
        // Any other ADT: the recorded debug shape of the
        // instance names each field's tag.
        if let Value::Enum { variant, fields } = value {
            if self.adt_field_tag(ty, *variant as usize, fields.len()).is_some() {
                return Ok(None);
            }
            let mut cloned: Vec<(Value, Option<crate::ir::DropGlue>)> =
                Vec::with_capacity(fields.len());
            for (i, f) in fields.iter().enumerate() {
                let field_ty = self.adt_field_tag(ty, *variant as usize, i);
                let field_glue =
                    glue.and_then(|g| g.field_plan(Some(*variant as usize), i));
                let field_ref =
                    value_ref.and_then(|t| t.projected(RefProj::Field(i)));
                let live = field_ref
                    .as_ref()
                    .map(|target| self.clone_reread_source(frames, target, f));
                let f = live.as_ref().unwrap_or(f);
                match self.clone_boxed_slice_element(
                    frames,
                    f,
                    field_ref.as_ref(),
                    field_ty.as_deref(),
                    field_glue.as_ref(),
                ) {
                    Ok(v) => cloned.push((v, field_glue)),
                    Err(err) => {
                        // Construction temporaries, as in the
                        // tuple arm above.
                        cloned.reverse();
                        return Err(self.drop_cloned_prefix(frames, cloned, err));
                    }
                }
            }
            return Ok(Some(Value::Enum {
                variant: *variant,
                fields: cloned.into_iter().map(|(v, _)| v).collect(),
            }));
        }
        // A closure value: the recorded environment shape of
        // the instance names each capture's tag, in
        // environment field order. Each capture clones with
        // its own tag judged the same way an aggregate field
        // is, so an owned capture re-owns what it owns (a
        // boxed capture is a fresh box, a counted capture
        // moves its count) instead of aliasing the source
        // closure's environment; a by-reference capture's tag
        // is a reference tag, and references clone as the
        // pointers they are, which is the sharing a reference
        // capture means. An instance with no recorded shape,
        // or whose shape does not cover the environment,
        // keeps the shape-only fallback.
        if let Value::Closure { def, env } = value {
            let Some(capture_tys) = self
                .program()
                .env_debug_shapes
                .get(ty)
                .filter(|shape| shape.upvars.len() == env.len())
                .map(|shape| {
                    shape
                        .upvars
                        .iter()
                        .map(|f| f.ty.clone())
                        .collect::<Vec<_>>()
                })
            else {
                return Ok(None);
            };
            let mut cloned: Vec<(Value, Option<crate::ir::DropGlue>)> =
                Vec::with_capacity(env.len());
            for (i, capture) in env.iter().enumerate() {
                let capture_ty = capture_tys[i].as_str();
                let capture_glue = glue.and_then(|g| g.field_plan(None, i));
                let capture_ref =
                    value_ref.and_then(|t| t.projected(RefProj::Field(i)));
                let live = capture_ref
                    .as_ref()
                    .map(|target| self.clone_reread_source(frames, target, capture));
                let capture = live.as_ref().unwrap_or(capture);
                match self.clone_boxed_slice_element(
                    frames,
                    capture,
                    capture_ref.as_ref(),
                    Some(capture_ty),
                    capture_glue.as_ref(),
                ) {
                    Ok(v) => cloned.push((v, capture_glue)),
                    Err(err) => {
                        // Construction temporaries, as in the
                        // tuple arm above.
                        cloned.reverse();
                        return Err(self.drop_cloned_prefix(frames, cloned, err));
                    }
                }
            }
            return Ok(Some(Value::Closure {
                def: def.clone(),
                env: cloned.into_iter().map(|(v, _)| v).collect(),
            }));
        }
        Ok(None)
    }

    /// Clone an owned value the way this module's `Clone::clone`
    /// model does: a value copy, except that a built box (the
    /// one-field aggregate holding an allocation pointer — the
    /// shape `box_alloc_pointer` documents as exact) is
    /// re-allocated around a recursively cloned pointee, so the
    /// clone owns its storage at every level. References, fat
    /// pointers and trait objects copy as the pointers they are,
    /// exactly as a field-wise `Clone` of a struct containing
    /// them behaves.
    fn clone_owned_value(&self, frames: &[Frame], value: &Value) -> InterpResult<Value> {
        if let Some((id, projection)) = box_alloc_pointer(value) {
            let pointee =
                self.read_resolved(frames, &Resolved::Alloc { id, projection })?;
            let cloned = self.clone_owned_value(frames, &pointee)?;
            return Ok(self.new_box(cloned));
        }
        Ok(match value {
            Value::Enum { variant, fields } => Value::Enum {
                variant: *variant,
                fields: fields
                    .iter()
                    .map(|f| self.clone_owned_value(frames, f))
                    .collect::<InterpResult<Vec<_>>>()?,
            },
            Value::Tuple(items) => Value::Tuple(
                items
                    .iter()
                    .map(|f| self.clone_owned_value(frames, f))
                    .collect::<InterpResult<Vec<_>>>()?,
            ),
            other => other.clone(),
        })
    }
}
