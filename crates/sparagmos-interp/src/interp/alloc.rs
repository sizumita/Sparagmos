//! Allocation and statics bookkeeping.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};

impl<'a> Interpreter<'a> {
    /// Record the guest-allocated buffer of a
    /// `String::with_capacity` result against the place the
    /// call wrote it to (see `string_buffers`).
    pub(crate) fn record_string_buffer(
        &self,
        frames: &Vec<Frame>,
        body: &Body,
        destination: &Place,
    ) {
        let pending = self.pending_string_buffer.borrow_mut().take();
        let Some((ptr, cap)) = pending else {
            return;
        };
        if let Ok(Resolved::Place {
            frame_pos,
            local,
            projection,
        }) = self.resolve_place(frames, body, destination)
        {
            if projection.is_empty() {
                let frame_id = frames[frame_pos].id;
                self.string_buffers
                    .borrow_mut()
                    .insert((frame_id, local), (ptr, cap));
            }
        }
    }

    /// Run the guest deallocator for the recorded buffer of a
    /// string whose place's drop glue just finished (see
    /// `string_buffers`).
    pub(crate) fn note_string_buffer_free(&mut self, frames: &mut Vec<Frame>, root: &RefTarget) {
        self.note_string_buffer_free_inner(frames, root, false);
    }

    /// `note_string_buffer_free`, optionally with the
    /// single-entry fallback for strings whose recorded place
    /// was lost through an untracked move. The fallback is only
    /// sound at a `mem::drop` site, where the dropped value is
    /// consumed by the call itself.
    pub(crate) fn note_string_buffer_free_inner(
        &mut self,
        frames: &mut Vec<Frame>,
        root: &RefTarget,
        allow_fallback: bool,
    ) {
        let RefTarget::Place {
            frame,
            local,
            projection,
        } = root
        else {
            return;
        };
        if !projection.is_empty() {
            return;
        }
        let entry = self
            .string_buffers
            .borrow_mut()
            .remove(&(*frame, *local));
        let entry = entry.or_else(|| {
            if !allow_fallback {
                return None;
            }
            // The string reached this place through a move the
            // assignment transfer did not track (for example
            // straight into a call argument): when the frame
            // holds exactly one recorded buffer, it can only be
            // this string's.
            let mut buffers = self.string_buffers.borrow_mut();
            let in_frame = buffers
                .keys()
                .filter(|(f, _)| *f == *frame)
                .copied()
                .collect::<Vec<_>>();
            if in_frame.len() == 1 {
                buffers.remove(&in_frame[0])
            } else {
                None
            }
        });
        let Some((ptr, cap)) = entry else {
            return;
        };
        let Some(shim) = self.program().alloc_shims.get("__rust_dealloc") else {
            return;
        };
        let shim = shim.clone();
        let _ = self.call_callable_to_completion(
            frames,
            &Value::FnPtr(shim),
            vec![ptr, Value::Uint(cap), alignment_value(1)],
        );
    }

    /// Run the guest deallocator for a dropped box, when the
    /// program defines one: natively the box's storage is freed
    /// by `__rust_dealloc` with the pointee's layout, which the
    /// guest allocator observes. The layout is the one recorded
    /// when the allocation builtins created the box (see
    /// `box_layouts`); a box with no recorded layout (built by
    /// another path) frees only in the model, as before. The
    /// model's own store entry is released by the caller (the
    /// guest's deallocation usually frees it itself through the
    /// system path; a guest allocator that pools the block
    /// instead is beyond what the store can represent).
    pub(crate) fn note_box_free(
        &mut self,
        frames: &mut Vec<Frame>,
        box_value: &Value,
    ) {
        let Some(shim) = self.program().alloc_shims.get("__rust_dealloc") else {
            return;
        };
        let shim = shim.clone();
        let Some((id, _)) = box_alloc_pointer(box_value) else {
            return;
        };
        let Some((size, align)) = self.box_layouts.borrow_mut().remove(&id) else {
            return;
        };
        let Value::Enum { fields, .. } = box_value else {
            return;
        };
        let Some(ptr) = fields.first().cloned() else {
            return;
        };
        let _ = self.call_callable_to_completion(
            frames,
            &Value::FnPtr(shim),
            vec![ptr, Value::Uint(size), alignment_value(align)],
        );
    }

    /// Record the pointee layout of a box the allocation
    /// builtins just created, for `note_box_free`.
    pub(crate) fn record_box_layout(&self, box_value: &Value, size: u128, align: u128) {
        if let Some((id, _)) = box_alloc_pointer(box_value) {
            self.box_layouts.borrow_mut().insert(id, (size, align));
        }
    }

    /// Record the declared pointee type of a box the
    /// construction builtins just created, for the judgments
    /// that consult an allocation's declared type (see
    /// [`Interpreter::record_alloc_ty`]).
    pub(crate) fn record_box_ty(&self, box_value: &Value, ty: Ty) {
        if let Some((id, _)) = box_alloc_pointer(box_value) {
            self.record_alloc_ty(id, ty);
        }
    }

    /// Allocate through the program's guest allocator shim
    /// (`#[global_allocator]`), when the program defines one:
    /// the allocation the native entry point
    /// (`alloc::alloc::__rust_alloc`) would bind to the shim.
    /// Returns the shim's pointer, or `None` when the program
    /// has no guest allocator (the caller then allocates in the
    /// model directly, as before).
    pub(crate) fn call_guest_alloc(
        &mut self,
        frames: &mut Vec<Frame>,
        size: u128,
        align: u128,
    ) -> InterpResult<Option<Value>> {
        let Some(shim) = self.program().alloc_shims.get("__rust_alloc") else {
            return Ok(None);
        };
        let shim = shim.clone();
        let align_arg = alignment_value(align);
        let ptr = self.call_callable_to_completion(
            frames,
            &Value::FnPtr(shim),
            vec![Value::Uint(size), align_arg],
        )?;
        Ok(Some(ptr))
    }
}

impl<'a> Interpreter<'a> {
    /// The statics-store key of thread `tid`'s storage for the
    /// thread-local static `name`. TLS storage lives in the
    /// statics store under a key no real static can carry (real
    /// keys are item paths, optionally suffixed with a `DefId`
    /// disambiguator), so reads and writes through a TLS reference
    /// route through the ordinary static machinery while the
    /// storage itself stays private to the owning thread: the
    /// reference names that thread's storage permanently, exactly
    /// as a native pointer names that thread's TLS block.
    pub(crate) fn tls_store_key(tid: u64, name: &str) -> String {
        format!("\u{1}tls\u{1}{tid}\u{1}{name}")
    }

    pub(crate) fn missing_static(&self, name: &str) -> InterpError {
        InterpError::UnsupportedTerminator {
            kind: format!(
                "static `{name}` has no storage: its initializer constant was never evaluated"
            ),
        }
    }

    /// Seed the storage of the static `name` from its initializer
    /// constant `seed`, unless it is already seeded. A static
    /// whose initializer mentions itself (or a cycle of statics)
    /// through a reference does not recurse: the nested reference
    /// evaluates to a pointer at the storage the outer seeding is
    /// about to fill, exactly as the nested reference in a native
    /// static initializer names the static's address.
    pub(crate) fn ensure_static(&self, name: &str, seed: &Constant) -> InterpResult<()> {
        if self.statics.borrow().contains_key(name) {
            return Ok(());
        }
        if !self.statics_seeding.borrow_mut().insert(name.to_string()) {
            return Ok(());
        }
        let value = self.eval_constant(seed);
        self.statics_seeding.borrow_mut().remove(name);
        let value = value?;
        self.statics.borrow_mut().insert(name.to_string(), value);
        Ok(())
    }

    /// Seed the storage of the static `name` from the
    /// initializer seed the conversion recorded for it in the
    /// program's seed table (see `ir::Program::static_seeds`),
    /// unless it is already seeded (or being seeded, for a
    /// static whose initializer mentions itself). A static
    /// with no recorded seed -- one no converted `StaticRef`
    /// named, as in a hand-built program -- is an explicit
    /// error, never a guessed value.
    pub(crate) fn ensure_static_seeded(&self, name: &str) -> InterpResult<()> {
        if self.statics.borrow().contains_key(name)
            || self.statics_seeding.borrow().contains(name)
        {
            return Ok(());
        }
        let Some(seed) = self.program.static_seeds.get(name) else {
            return Err(InterpError::UnsupportedTerminator {
                kind: format!(
                    "static `{name}` has no storage: no initializer seed \
                     was recorded for it in the program"
                ),
            });
        };
        self.ensure_static(name, seed)
    }

    /// The current value of the storage of the static whose
    /// program key names the item `item` as its last path
    /// segment, when exactly one seeded static matches. The
    /// modelled `rust_test_helpers` check functions (see
    /// `Builtin::FfiStaticMutCheckFour`) read the C library's
    /// global through this: the guest's declaration of the
    /// global and the C body denote the one storage, whose
    /// program key is the declaration's own path, so the last
    /// segment is the name both sides share. A key carrying
    /// the `#DefId(..)` disambiguator suffix names the same
    /// item once the suffix is stripped.
    pub(crate) fn static_by_item_name(&self, item: &str) -> Option<Value> {
        let store = self.statics.borrow();
        let mut found: Option<&Value> = None;
        for (key, value) in store.iter() {
            let base = key.split('#').next().unwrap_or(key);
            if base == item || base.rsplit("::").next() == Some(item) {
                if found.is_some() {
                    return None;
                }
                found = Some(value);
            }
        }
        found.cloned()
    }

    /// The current value of the static `name`'s storage.
    pub(crate) fn static_base(&self, name: &str) -> InterpResult<Value> {
        self.statics
            .borrow()
            .get(name)
            .cloned()
            .ok_or_else(|| self.missing_static(name))
    }

    /// The error for naming an allocation the store no longer
    /// holds (its owning box was dropped, or the id never named
    /// an allocation of this run).
    pub(crate) fn missing_alloc(&self, id: u64) -> InterpError {
        InterpError::type_mismatch(
            "dereference",
            "reference into a live allocation",
            format!("reference into freed allocation #{id}"),
        )
    }

    /// Allocate `pointee` in the allocation store and return the
    /// `Box` value denoting it: the one-field aggregate whose
    /// single field is the allocation pointer (see
    /// [`RefTarget::Alloc`]). This is the only shape the `Box`
    /// construction builtins produce; a box built from its MIR
    /// aggregates instead carries the `Unique`/`NonNull` wrapper
    /// chain above a pointer value.
    /// Create a heap allocation holding `value` in the
    /// allocation store and return the id that names it. The
    /// storage lives until the model frees it: for a box, when
    /// the owning box's drop glue completes; for a
    /// reference-counted allocation, when its strong and weak
    /// counts both reach zero.
    pub(crate) fn alloc_insert(&self, value: Value) -> u64 {
        let id = self.next_alloc.get();
        self.next_alloc.set(id + 1);
        self.allocs.borrow_mut().insert(id, value);
        id
    }

    /// Record the final state of the owned iterator a consumer
    /// builtin has just driven to exhaustion (see the
    /// `consumed_iter_final` field).
    pub(crate) fn note_consumed_iter_final(&self, state: Value) {
        *self.consumed_iter_final.borrow_mut() = Some(state);
    }

    /// The current value held in allocation `id`, if it is live.
    pub(crate) fn alloc_value(&self, id: u64) -> Option<Value> {
        self.allocs.borrow().get(&id).cloned()
    }

    /// Record the declared type of allocation `id` (see the
    /// `alloc_tys` field). A creation point that knows the
    /// type calls this once, when the allocation is made. The
    /// record is the allocation's own: retargeting a pointer
    /// at the same storage (an `assume_init` transmute is the
    /// canonical case) does not move it, because the storage
    /// was created for the recorded type whatever the pointer
    /// that reaches it declares.
    pub(crate) fn record_alloc_ty(&self, id: u64, ty: Ty) {
        self.alloc_tys.borrow_mut().insert(id, ty);
    }

    /// The declared type recorded for allocation `id`, if its
    /// creation point knew one.
    pub(crate) fn alloc_ty(&self, id: u64) -> Option<Ty> {
        self.alloc_tys.borrow().get(&id).cloned()
    }

    /// Whether the indices over allocation `id` are denominated
    /// by the type its creation point recorded (see the
    /// `alloc_tys` field): the recorded type is a sequence, so
    /// an index names an element of that sequence -- exactly as
    /// an index over a static array is denominated by the
    /// static's recorded type -- not a count of the units of
    /// whatever pointer currently denotes the position. The
    /// recorded element size must be known and nonzero, the
    /// same conditions under which
    /// [`Self::alloc_typed_offset_absorb`] denominates a
    /// displacement by it.
    pub(crate) fn alloc_index_is_typed(&self, id: u64) -> bool {
        match self.alloc_ty(id) {
            Some(Ty::Array(elem, _)) | Some(Ty::Slice(elem)) => {
                matches!(self.ty_size(&elem), Some(size) if size > 0)
            }
            _ => false,
        }
    }

    /// Whether the ADT `name` (a recorded allocation type, with
    /// generic arguments dropped as `Ty::Adt` carries them) is a
    /// single-field struct: a wrapper whose stored form is one
    /// layer around its one field at every width, never the
    /// position's own multi-part value. The instance's debug
    /// shape answers it (see `debug_shape_for_def`, which
    /// resolves the monomorphic instance of a bare name); an
    /// ADT with no recorded shape is not judged a wrapper.
    pub(crate) fn adt_is_single_field_wrapper(&self, name: &str) -> bool {
        match self.debug_shape_for_def(name) {
            Some(shape) => {
                shape.kind == sparagmos_mir::AdtShapeKind::Struct
                    && shape.variants.len() == 1
                    && shape.variants[0].fields.len() == 1
            }
            None => false,
        }
    }

    pub(crate) fn new_box(&self, pointee: Value) -> Value {
        let id = self.alloc_insert(pointee);
        Value::Enum {
            variant: 0,
            fields: vec![Value::Ref(RefTarget::Alloc {
                id,
                projection: Vec::new(),
            })],
        }
    }

    /// The current contents of allocation `id` (a clone; the
    /// store is the shared storage, so writes go through
    /// `write_resolved`, never through this copy).
    pub(crate) fn alloc_base(&self, id: u64) -> InterpResult<Value> {
        self.allocs
            .borrow()
            .get(&id)
            .cloned()
            .ok_or_else(|| self.missing_alloc(id))
    }

    /// Register `addr` as a synthetic address denoting
    /// allocation `id` in the store (see the `address_owners`
    /// field): the reverse lookup a guest needs when a value it
    /// owns is named only by the address an owning word carries.
    pub(crate) fn register_address_owner(&self, addr: u64, id: u64) {
        self.address_owners.borrow_mut().insert(addr, id);
    }

    /// The allocation a registered synthetic address denotes, if
    /// the address is registered.
    pub(crate) fn address_owner(&self, addr: u64) -> Option<u64> {
        self.address_owners.borrow().get(&addr).copied()
    }

    /// Remove and return the allocation a registered synthetic
    /// address denotes: its owner's drop glue has run, so the
    /// address denotes nothing from here on.
    pub(crate) fn take_address_owner(&self, addr: u64) -> Option<u64> {
        self.address_owners.borrow_mut().remove(&addr)
    }

    /// Remove allocation `id` from the store (its owning box's
    /// drop glue has completed). Removing an id that is not
    /// present is not an error: glue also runs for places whose
    /// box was fabricated rather than allocated.
    /// Mark an allocation as a raw element block (see
    /// `elem_blocks`).
    pub(crate) fn mark_elem_block(&self, id: u64) {
        self.elem_blocks.borrow_mut().insert(id);
    }

    /// Whether an allocation is a raw element block.
    /// The projection a raw element block's access uses: a
    /// lone byte offset (the denomination a differently sized
    /// pointer cast left on the pointer, see `PtrCastUnits`)
    /// denotes the block's byte slot of that offset — the
    /// block's slots are byte positions for a byte-denominated
    /// pointer exactly as they are element positions for an
    /// element-denominated one, and the two denominations are
    /// kept apart by the projection form itself. Any other
    /// projection passes through unchanged.
    pub(crate) fn elem_block_projection(
        &self,
        id: u64,
        projection: &[RefProj],
    ) -> Vec<RefProj> {
        if self.is_elem_block(id) {
            if let Some(RefProj::ByteOffset(d)) = projection.first() {
                if let Ok(i) = usize::try_from(*d) {
                    let mut out = vec![RefProj::Index(i)];
                    out.extend_from_slice(&projection[1..]);
                    return out;
                }
            }
            // A field-first projection addresses the pointee's
            // own structure: the block's base pointer denotes
            // its first element, so the element step belongs in
            // front. (A plain field step would otherwise land
            // on the block's element tuple itself, whose field
            // 0 happens to be the element and whose other
            // fields do not exist.) Index- and sub-slice-first
            // projections already denote positions in the
            // block's element sequence and keep their form.
            if matches!(
                projection.first(),
                Some(
                    RefProj::Field(_)
                        | RefProj::UnionField { .. }
                        | RefProj::Downcast(_)
                )
            ) {
                let mut out = vec![RefProj::Index(0)];
                out.extend_from_slice(projection);
                return out;
            }
        }
        projection.to_vec()
    }

    pub(crate) fn is_elem_block(&self, id: u64) -> bool {
        self.elem_blocks.borrow().contains(&id)
    }

    pub(crate) fn free_alloc(&self, id: u64) {
        self.allocs.borrow_mut().remove(&id);
        self.glue_dropped_allocs.borrow_mut().remove(&id);
        self.elem_blocks.borrow_mut().remove(&id);
        self.alloc_tys.borrow_mut().remove(&id);
    }

    /// Set the discriminant of the ADT value `value` to `variant`,
    /// as a `SetDiscriminant` statement does: a coroutine keeps both
    /// of its storages (the transform writes the entered state's
    /// fields through its downcast before the tag moves, and values
    /// saved for other states must survive the change), an enum's
    /// old payload is invalidated.
    pub(crate) fn set_discriminant_value(
        value: &mut Value,
        variant: u32,
        place: &Place,
    ) -> InterpResult<()> {
        match value {
            Value::Coroutine { variant: current, .. } => {
                *current = variant;
                // The state storage is shared between states: the
                // fields stay in place. The transform sets the tag
                // of the state it is entering only after (or before)
                // storing that state's fields through downcast
                // projections, and values saved for later states
                // must survive this tag change.
                Ok(())
            }
            Value::Enum {
                variant: current,
                fields,
            } => {
                if *current != variant {
                    *current = variant;
                    // The old payload is invalidated, as in MIR.
                    fields.clear();
                }
                // A SetDiscriminant naming the variant the place
                // already holds is a tag-only write: in the
                // writes-first order (the order the coroutine
                // transform also uses) the new variant's fields were
                // already stored through its downcast, which
                // switched the tag when it rematerialized the
                // payload, and those stores must survive this
                // statement.
                Ok(())
            }
            other => Err(InterpError::type_mismatch(
                format!("SetDiscriminant on {place}"),
                "ADT value",
                other.kind_name(),
            )),
        }
    }
}

impl<'a> Interpreter<'a> {
    /// Whether a pointer into allocation `id` denotes the
    /// allocation's own base for the allocation family (a free,
    /// or a realloc's move-out of the old block): the projection
    /// must consist of zero steps only, and a nonempty zero-step
    /// projection denotes the base only when the allocation is a
    /// raw element block. In a value-holding allocation the same
    /// step denotes a position inside the stored value -- the
    /// first element of a stored `Vec` value's buffer, reached
    /// through the owning wrapper around it -- whose storage is
    /// the value's own, not the allocation's, so freeing or
    /// moving the allocation under it would destroy the wrapper's
    /// storage instead of the buffer's.
    pub(crate) fn alloc_ptr_is_base(&self, id: u64, projection: &[RefProj]) -> bool {
        projection_is_base(projection)
            && (projection.is_empty() || self.is_elem_block(id))
    }
}
