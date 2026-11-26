//! Resolved reads and writes: the memory-model core.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};
use std::borrow::Cow;

/// One byte of a function's code image (see
/// [`Interpreter::fn_code_deref`]): a splitmix64-style
/// finaliser over the image seed and the byte's offset, so
/// the stream is deterministic and position-dependent, and
/// any byte can be served without materialising the image.
fn code_byte(seed: u64, offset: usize) -> u8 {
    let mut z = seed.wrapping_add(
        0x9E37_79B9_7F4A_7C15u64.wrapping_mul(offset as u64 + 1),
    );
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 56) as u8
}

/// The integer a code read of `bits` width assembles: the
/// image's bytes at `offset`, little-endian, as a native
/// little-endian load of the same width from the function's
/// bytes yields.
fn code_int(seed: u64, offset: usize, bits: u32) -> u128 {
    let bytes = (bits as usize).div_ceil(8);
    let mut value: u128 = 0;
    for i in 0..bytes {
        value |= u128::from(code_byte(seed, offset + i)) << (8 * i);
    }
    value
}

impl<'a> Interpreter<'a> {
    /// The base value a bulk byte operation reads at `target`:
    /// the resolved value, or -- for a pointer displaced into a
    /// scalar's storage -- the byte the pointer denotes. The
    /// operation's element is a byte by construction, so the
    /// displaced position is a byte position here; a displaced
    /// read anywhere else (a wider element load) stays the
    /// explicit projection error `project_value` raises.
    /// `elem` is the operation's element type when the call
    /// site knows it; it judges the uninitialized outcome as
    /// in [`read_resolved_typed`].
    fn read_bulk_base(
        &self,
        frames: &[Frame],
        target: &Resolved,
        elem: Option<&Ty>,
    ) -> InterpResult<Value> {
        match self.read_resolved_typed(frames, target, elem) {
            Ok(v) => Ok(v),
            Err(e) => {
                // A byte-displaced aggregate target: the byte
                // view of the base value's storage denotes the
                // byte (the scalar fallback below prices the
                // displacement within the head element's slot,
                // which is only the position when the
                // displacement stays inside that element).
                if let Some(Ok(byte)) = self.byte_view_read(frames, target, 0, false) {
                    return Ok(byte);
                }
                match self.displaced_scalar_byte(frames, target) {
                    Ok(Some(byte)) => Ok(byte),
                    _ => Err(e),
                }
            }
        }
    }

    /// The byte at the displacement of a target whose projection
    /// ends in a byte-offset step into a scalar's storage, when
    /// the base under the displacement is a scalar; `None` for
    /// every other shape.
    fn displaced_scalar_byte(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let (base_target, disp) = match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => match projection.split_last() {
                Some((&RefProj::ByteOffset(d), head)) if d >= 0 => (
                    Resolved::Place {
                        frame_pos: *frame_pos,
                        local: *local,
                        projection: head.to_vec(),
                    },
                    d as usize,
                ),
                _ => return Ok(None),
            },
            Resolved::Static { name, projection } => match projection.split_last() {
                Some((&RefProj::ByteOffset(d), head)) if d >= 0 => (
                    Resolved::Static {
                        name: name.clone(),
                        projection: head.to_vec(),
                    },
                    d as usize,
                ),
                _ => return Ok(None),
            },
            Resolved::Alloc { id, projection } => match projection.split_last() {
                Some((&RefProj::ByteOffset(d), head)) if d >= 0 => (
                    Resolved::Alloc {
                        id: *id,
                        projection: head.to_vec(),
                    },
                    d as usize,
                ),
                _ => return Ok(None),
            },
            Resolved::Value(base, projection) => match projection.split_last() {
                Some((&RefProj::ByteOffset(d), head)) if d >= 0 => (
                    Resolved::Value(base.clone(), head.to_vec()),
                    d as usize,
                ),
                _ => return Ok(None),
            },
        };
        let base = self.read_resolved(frames, &base_target)?;
        // The base's declared kind, when the target's root
        // type is recorded and the type walk follows the head
        // projection to the base's position: a float storage
        // narrower than the model form yields its own width's
        // byte (see `scalar_storage_byte_as`), the same
        // declared-width answer the byte view gives the
        // targets it judges. An unrecorded root (a value
        // target above all) keeps the evidence-free byte.
        let kind = self
            .byte_view_base(frames, &base_target)
            .and_then(|(head, _, root_ty, _, _)| {
                step_ty_across(Some(self.program()), Some(root_ty), head)
            })
            .and_then(|ty| byte_scalar_of_ty(Some(&ty)));
        Ok(scalar_storage_byte_as(&base, kind, disp).map(|byte| Value::Uint(u128::from(byte))))
    }

    /// The element copy behind `copy_nonoverlapping`, shared by the
    /// `StatementKind::Intrinsic` form (a direct
    /// `core::intrinsics::copy_nonoverlapping` call in MIR) and the
    /// `std::ptr::copy_nonoverlapping` / `std::ptr::copy` builtin
    /// (the same operation arriving as an ordinary call). The source
    /// elements are snapshotted before any write, so the copy is
    /// exact even if the ranges overlapped (which the
    /// non-overlapping contract forbids, and which makes the shared
    /// implementation a correct `memmove` for `ptr::copy`).
    ///
    /// The copy machinery's wrapper unwrap: like
    /// `builtins::unwrap_wrapper`, but the descent stops at a
    /// leaf-data aggregate (an aggregate of scalars — a byte
    /// buffer's stored form), which is the element's core, not
    /// one more wrapper layer. A one-element buffer has
    /// exactly the one-field shape a wrapper layer has (a
    /// one-byte path's byte list is the case that matters:
    /// descending into it mistakes the buffer for a wrapper
    /// and the byte inside for the element's core). The
    /// general `unwrap_wrapper` keeps its unconditional
    /// descent: its other callers (the atomic operations, the
    /// pointer unwraps) unwrap genuine scalar wrappers whose
    /// payload field is a one-field scalar aggregate.
    fn copy_unwrap<'v>(&self, value: &'v Value) -> &'v Value {
        let mut current = value;
        while let Value::Enum {
            variant: 0,
            fields,
        } = current
        {
            if fields.len() != 1 || is_leaf_data_aggregate(&fields[0]) {
                break;
            }
            current = &fields[0];
        }
        current
    }

    /// The write-side mirror of [`Self::copy_unwrap`]: rebuild
    /// `current`'s wrapper chain around a new core, stopping
    /// the descent at the same leaf-data boundary.
    fn copy_rewrap(&self, current: &Value, core: Value) -> Value {
        match current {
            Value::Enum {
                variant: 0,
                fields,
            } if fields.len() == 1 && !is_leaf_data_aggregate(&fields[0]) => {
                Value::Enum {
                    variant: 0,
                    fields: vec![self.copy_rewrap(&fields[0], core)],
                }
            }
            _ => core,
        }
    }

    /// `elem` is the copy's element type when the call site knows
    /// it (both call paths take it from the pointer operands'
    /// declared types); with `None` the copy reads the element
    /// boundary off the values' shapes alone, as it did before the
    /// type was threaded through.
    pub(crate) fn exec_copy_nonoverlapping(
        &self,
        frames: &mut Vec<Frame>,
        src_value: Value,
        dst_value: Value,
        count_value: Value,
        elem: Option<&Ty>,
    ) -> InterpResult<()> {
                let n = match count_value {
                    Value::Uint(n) => usize::try_from(n).map_err(|_| {
                        InterpError::type_mismatch(
                            "copy_nonoverlapping count",
                            "count that fits in usize",
                            format!("{n}"),
                        )
                    })?,
                    Value::Int(n) if n >= 0 => n as usize,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "copy_nonoverlapping count",
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                };
                if n == 0 {
                    // A zero-count copy touches no memory: the
                    // pointers are never dereferenced, so even a
                    // null or dangling pointer is a legal operand.
                    return Ok(());
                }
                let src_target = self.deref_value(frames, &src_value)?;
                let dst_target = self.deref_value(frames, &dst_value)?;
                // Snapshot the source elements before writing: the
                // intrinsic's contract is that the ranges do not
                // overlap, and the snapshot keeps the copy exact
                // element-for-element either way. A pointer in this
                // model denotes either an aggregate (an array place,
                // a fat pointer's base) whose elements are indexed, or
                // a single element, which a count of 1 copies directly
                // and larger counts continue at its siblings (the
                // sequence access in `read_seq_at` / `write_seq_at`).
                // A byte-granular copy whose other end is one
                // structured scalar value — the byte copy an
                // unaligned vector load/store lowers to, filling
                // or draining a vector local — moves the bytes
                // through the value's leaf structure: the
                // indexed access cannot address a vector value's
                // bytes (it would walk the storage array's lanes
                // as elements). Only whole structured values are
                // rerouted; every other shape keeps the element
                // path below. The reroute is only for copies
                // whose elements are bytes: when the element type
                // is known and is not a byte type, the count
                // counts elements, and a structured element value
                // at either end is one of the elements being
                // copied -- treating it as a byte tree would
                // patch the first element with the following
                // elements' leading bytes and drop the rest (the
                // shifted tail of a `Vec::insert` of one-field
                // wrapper elements).
                let byte_granular = match elem {
                    None => true,
                    Some(ty) => matches!(
                        ty,
                        Ty::Uint(UintTy::U8) | Ty::Int(IntTy::I8)
                    ),
                };
                if n > 1 && byte_granular {
                    let dst_tree = self
                        .read_resolved(frames, &dst_target)
                        .ok()
                        .filter(|v| is_scalar_tree(v));
                    let src_tree = if dst_tree.is_none() {
                        self.read_resolved(frames, &src_target)
                            .ok()
                            .filter(|v| is_scalar_tree(v))
                    } else {
                        None
                    };
                    if dst_tree.is_some() || src_tree.is_some() {
                        let bytes = match &src_tree {
                            Some(tree) => tree_to_bytes(tree, n),
                            None => {
                                // The source's units as bytes: a
                                // unit can be a one-field wrapper
                                // over the byte (a control byte's
                                // tag in a raw block), which the
                                // byte view unwraps.
                                fn seq_byte(value: &Value) -> Option<u8> {
                                    match value {
                                        Value::Uint(b) if *b <= 0xFF => Some(*b as u8),
                                        Value::Int(b) if (0..=255).contains(b) => {
                                            Some(*b as u8)
                                        }
                                        Value::Enum { fields, .. } if fields.len() == 1 => {
                                            seq_byte(&fields[0])
                                        }
                                        _ => None,
                                    }
                                }
                                let mut bytes = Vec::with_capacity(n);
                                let mut all_bytes = true;
                                for i in 0..n {
                                    match self.read_seq_at_with_elem(frames, &src_target, i, elem) {
                                        Ok(v) => match seq_byte(&v) {
                                            Some(b) => bytes.push(b),
                                            None => {
                                                all_bytes = false;
                                                break;
                                            }
                                        },
                                        _ => {
                                            all_bytes = false;
                                            break;
                                        }
                                    }
                                }
                                if all_bytes {
                                    Some(bytes)
                                } else {
                                    None
                                }
                            }
                        };
                        if let Some(bytes) = bytes {
                            if let Some(tree) = &dst_tree {
                                if let Some(patched) = byte_patch_tree(tree, &bytes) {
                                    self.write_resolved(frames, &dst_target, patched)?;
                                    return Ok(());
                                }
                            }
                            if let Some(tree) = &src_tree {
                                // The copy moves exactly the source
                                // value's bytes (`tree_to_bytes`
                                // only succeeds at the value's own
                                // length): write the value itself as
                                // one slot. Writing its bytes one by
                                // one would shred the structured
                                // value into byte slots that a later
                                // whole-value read cannot reassemble
                                // (a hash table's resize copying its
                                // buckets byte-granularly).
                                let tree = tree.clone();
                                self.write_resolved(frames, &dst_target, tree)?;
                                return Ok(());
                            }
                        }
                    }
                }
                let aggregate = |v: &Value| {
                    matches!(
                        v,
                        Value::Tuple(_) | Value::FatRef { .. } | Value::Bytes(_) | Value::Str(_)
                    )
                };
                let src_base = self.read_bulk_base(frames, &src_target, elem)?;
                // The elements are values of the element type. When
                // that type is known and the value found at the
                // source IS one element -- possibly inside a
                // one-field wrapper, since a pointer into a
                // transparent wrapper names the wrapper place, whose
                // only field is the pointee at that address -- the
                // first element is that value, not the first lane
                // of it. (This is the shape `Simd::load`'s copy
                // has: the element is an array, and the source is
                // the array itself.) Without a known element type
                // the shape rule applies: a non-aggregate source is
                // one element, an aggregate source is indexed.
                // A `MaybeUninit` element is the union value
                // over its storage (see `maybe_uninit_form`):
                // the base denotes one element unless it is a
                // sequence of them, and the element the copy
                // moves is the union form of the content found,
                // never the bare payload the wrapper judgement
                // below peels off for transparent one-field
                // structs -- peeling a union loses the form its
                // own readers (the union field read behind
                // `assume_init`, the whole-place move) require.
                let mu_elem = matches!(elem, Some(ty) if is_maybe_uninit_ty(ty));
                let src_single = match elem {
                    Some(_) if mu_elem => {
                        !aggregate(&src_base)
                            && !matches!(src_base, Value::SparseVec { .. })
                    }
                    Some(ty) => {
                        value_fits_ty(self.copy_unwrap(&src_base), ty)
                    }
                    None => n == 1 && !aggregate(&src_base),
                };
                let elems: Vec<Value> = if src_single {
                    let mut elems = Vec::with_capacity(n);
                    if mu_elem {
                        elems.push(maybe_uninit_form(src_base.clone()));
                        for i in 1..n {
                            let unit = self.read_seq_at_with_elem(frames, &src_target, i, elem)?;
                            elems.push(maybe_uninit_form(unit));
                        }
                    } else {
                        elems.push(self.copy_unwrap(&src_base).clone());
                        for i in 1..n {
                            elems.push(self.read_seq_at_with_elem(frames, &src_target, i, elem)?);
                        }
                    }
                    elems
                } else {
                    let mut elems = Vec::with_capacity(n);
                    for i in 0..n {
                        // A unit of the source range that was
                        // never written copies as uninit: a
                        // byte-granular copy over a raw block
                        // spans model slots that carry no value
                        // (the block's element occupies one slot;
                        // its byte size counts the rest).
                        let unit = match self.read_seq_at_with_elem(frames, &src_target, i, elem) {
                            Ok(v) => v,
                            Err(InterpError::UninitializedRead { .. }) => Value::Uninit,
                            Err(e) => return Err(e),
                        };
                        let unit = if mu_elem {
                            maybe_uninit_form(unit)
                        } else {
                            unit
                        };
                        elems.push(unit);
                    }
                    elems
                };
                // The destination, mirroring the source: with a
                // known element type it denotes one element when
                // the value found there is not an aggregate of
                // elements (a wrapper place being filled, a scalar
                // place, an element-shaped aggregate itself). An
                // unreadable (not yet written) destination is a
                // buffer being filled: the indexed path is the one
                // that can create its elements.
                let dst_current = self.read_resolved(frames, &dst_target).ok();
                let dst_single = match elem {
                    Some(ty) => match &dst_current {
                        Some(cur) => {
                            !aggregate(cur)
                                || value_fits_ty(
                                    self.copy_unwrap(cur),
                                    ty,
                                )
                        }
                        None => false,
                    },
                    None => {
                        let dst_aggregate = dst_current
                            .as_ref()
                            .map(|v| aggregate(v))
                            .unwrap_or(true);
                        n == 1 && !dst_aggregate
                    }
                };
                if dst_single {
                    // The destination place's declared type, when
                    // the target is a whole local of some frame: a
                    // place whose declared type IS the element type
                    // holds the element itself, not a wrapper
                    // around it, so its value is replaced outright;
                    // any other whole place the pointer names is a
                    // wrapper whose payload slot the element fills
                    // (the write-side mirror of `unwrap_wrapper`,
                    // as atomic stores do).
                    let dst_place_ty = match &dst_target {
                        Resolved::Place {
                            frame_pos,
                            local,
                            projection,
                        } if projection.is_empty() => frames
                            .get(*frame_pos)
                            .and_then(|f| self.program.get(&f.func))
                            .and_then(|b| b.local_ty(Local(*local)))
                            .cloned(),
                        _ => None,
                    };
                    let mut elems = elems.into_iter();
                    let first = elems.next().expect("count is nonzero");
                    // Whether the incoming first element replaces
                    // the destination value outright. Only an
                    // element the source side peeled out of a
                    // wrapper chain (`src_single`) is ever
                    // rewrapped: it arrived as the chain's core and
                    // belongs back inside the wrapper shape the
                    // destination carries. An element read whole
                    // from its sequence IS the destination element,
                    // and wrapping it would store the element
                    // inside itself. When the destination is a
                    // whole local, its declared type decides
                    // whether the peeled core needs the rewrap (a
                    // matching type means the core already is the
                    // element). A projected place (an array
                    // element, a vector buffer slot) has no
                    // declared type here, so a peeled element is
                    // rewrapped into the old element's shape --
                    // the shifted tail of a `Vec::insert`, and the
                    // form a sort's temporary takes, seen through
                    // the element's own wrapper chain.
                    let cur_is_element = match &dst_place_ty {
                        Some(_) => dst_place_ty.as_ref() == elem || !src_single,
                        None => match elem {
                            Some(_) => !src_single,
                            None => true,
                        },
                    };
                    let stored = match &dst_current {
                        Some(cur) if !cur_is_element => {
                            // The slot's current form is the
                            // authority for its representation: an
                            // incoming value that already carries
                            // that form (the element read from a
                            // sibling slot of the same buffer) is
                            // stored as it stands, and only a value
                            // arriving as the wrapped core is
                            // rebuilt into the slot's wrapper chain.
                            // Rebuilding unconditionally would wrap
                            // a whole element in a second copy of
                            // itself whenever the element type is a
                            // one-field aggregate (a `Vec<E>` tail
                            // shift's first element).
                            if crate::builtins::same_wrapper_form(cur, &first) {
                                first
                            } else {
                                self.copy_rewrap(cur, first)
                            }
                        }
                        _ => first,
                    };
                    self.write_resolved(frames, &dst_target, stored)?;
                    for (i, value) in elems.enumerate() {
                        self.write_seq_at_with_elem(frames, &dst_target, i + 1, value, elem)?;
                    }
                } else {
                    for (i, value) in elems.into_iter().enumerate() {
                        self.write_seq_at_with_elem(frames, &dst_target, i, value, elem)?;
                    }
                }
                Ok(())
    }

    // ----- place resolution --------------------------------------------------
    //
    // A place is resolved against the frame stack. `Field`, `Index`,
    // `ConstantIndex` and `Downcast` steps accumulate as constant
    // `RefProj` steps on a (frame, local) pair; a `Deref` step reads
    // the reference value found there and continues at its target.
    // `Index` locals are read in the *executing* frame: a place is
    // syntactic to the function that contains it, so its index locals
    // always live there, even when earlier steps crossed into another
    // frame through a reference.

    pub(crate) fn frame_pos(&self, frames: &[Frame], id: u64) -> InterpResult<usize> {
        frames.iter().position(|f| f.id == id).ok_or_else(|| {
            InterpError::type_mismatch(
                "dereference",
                "reference into a live frame",
                format!("reference into finished frame #{id}"),
            )
        })
    }

    /// Classify a `BoxDerefTransmute` cast operand: when the
    /// operand is a place naming a `Box` or one of its pointer
    /// wrappers (`Unique` / `NonNull`), return the resolved operand
    /// place, the resolved place of the pointee, and the number of
    /// single-field wrapper layers an aggregate-built box value
    /// carries above its pointer (the box itself carries 3 — Box,
    /// Unique, NonNull — and each field already projected away
    /// removes one). Returns `None` for any other operand, whose
    /// cast keeps the strict wrapper-chain reading.
    pub(crate) fn box_cast_places(
        &self,
        frames: &[Frame],
        body: &Body,
        operand: &Operand,
    ) -> Option<(Resolved, Resolved, usize)> {
        let place = match operand {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Constant(_) => return None,
        };
        let Ty::Adt(name) = body.local_ty(place.local)? else {
            return None;
        };
        let segment = name.rsplit("::").next().unwrap_or(name.as_str());
        let is_box = matches!(name.as_str(), "std::boxed::Box" | "alloc::boxed::Box");
        let (depth, pointee_field) = match (segment, place.projection.as_slice()) {
            ("Box", []) if is_box => (3, true),
            ("Box", [ProjectionElem::Field(0)]) if is_box => (2, false),
            ("Unique", []) => (2, false),
            ("NonNull", []) => (1, false),
            _ => return None,
        };
        let operand_target = self.resolve_place(frames, body, place).ok()?;
        // A built box (see `Interpreter::new_box`) holds its
        // pointee in the allocation store, not in the aggregate:
        // when the operand value is the box aggregate itself, or
        // the allocation pointer stored at the bottom of the
        // pointer-wrapper chain above it, the pointee place
        // is the allocation itself. Naming the allocation keeps
        // the extracted pointer valid after this frame is gone
        // and makes every copy of the box extract the same
        // pointer — the two properties of a real heap pointer
        // this cast exists to expose.
        if let Ok(value) = self.read_resolved(frames, &operand_target) {
            let alloc = box_alloc_pointer(&value)
                .or_else(|| wrapper_alloc_pointer(&value))
                .or_else(|| match &value {
                    Value::Ref(RefTarget::Alloc { id, projection }) => {
                        Some((*id, projection.clone()))
                    }
                    _ => None,
                });
            if let Some((id, projection)) = alloc {
                return Some((
                    operand_target,
                    Resolved::Alloc { id, projection },
                    depth,
                ));
            }
        }
        let Resolved::Place {
            frame_pos,
            local,
            projection,
        } = operand_target.clone()
        else {
            return None;
        };
        let mut pointee_projection = projection.clone();
        if pointee_field {
            pointee_projection.push(RefProj::Field(0));
        }
        let pointee_target = Resolved::Place {
            frame_pos,
            local,
            projection: pointee_projection,
        };
        Some((operand_target, pointee_target, depth))
    }

    /// The reference value naming a resolved place (the value a
    /// `BoxDerefTransmute` cast yields in the builtin box model).
    pub(crate) fn pointee_ref(frames: &[Frame], target: &Resolved) -> Value {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => Value::Ref(RefTarget::place(
                frames[*frame_pos].id,
                *local,
                projection.clone(),
            )),
            Resolved::Value(base, projection) => {
                Value::Ref(RefTarget::Value(
                    Box::new(base.clone()),
                    projection.clone(),
                ))
            }
            Resolved::Static { name, projection } => Value::Ref(RefTarget::Static {
                name: name.clone(),
                projection: projection.clone(),
            }),
            Resolved::Alloc { id, projection } => Value::Ref(RefTarget::Alloc {
                id: *id,
                projection: projection.clone(),
            }),
        }
    }
}

impl<'a> Interpreter<'a> {
    /// Read an `Index` projection's local in the executing frame.
    pub(crate) fn read_index_local(
        &self,
        frames: &[Frame],
        body: &Body,
        local: Local,
        place: &Place,
    ) -> InterpResult<usize> {
        let frame = frames.last().expect("frame");
        let value = frame
            .locals
            .get(local.0)
            .cloned()
            .ok_or_else(|| InterpError::UnknownLocal {
                func: body.name.clone(),
                local: local.0,
            })?;
        match value {
            Value::Uint(v) => usize::try_from(v).map_err(|_| {
                InterpError::panic(format!("index {v} does not fit in usize"))
            }),
            Value::Int(v) if v >= 0 => usize::try_from(v).map_err(|_| {
                InterpError::panic(format!("index {v} does not fit in usize"))
            }),
            Value::Int(v) => Err(InterpError::panic(format!(
                "index out of bounds: negative index {v} used for {place}"
            ))),
            Value::Uninit => Err(InterpError::UninitializedRead {
                func: body.name.clone(),
                place: place.to_string(),
            }),
            other => Err(InterpError::type_mismatch(
                format!("index of {place}"),
                "integer index",
                other.kind_name(),
            )),
        }
    }

    /// The resolution of a dereference of a storage-less
    /// pointer -- a null pointer, or an address no allocation
    /// owns (the dangling pointer std serves every zero-sized
    /// allocation with) -- whose pointee type is zero-sized:
    /// the pointee's single value, which needs no storage to
    /// denote. `None` when the pointer denotes storage, the
    /// pointee type is unknown or not zero-sized, or its
    /// single value cannot be fabricated.
    pub(super) fn storage_less_zst_deref(
        &self,
        value: &Value,
        pointee_ty: Option<&Ty>,
    ) -> Option<Resolved> {
        let storage_less = match value {
            Value::Ref(RefTarget::Null) => true,
            Value::Ref(RefTarget::Address(addr)) => {
                self.address_owner(*addr).is_none()
            }
            _ => false,
        };
        if !storage_less {
            return None;
        }
        let single = self.zst_single_value(pointee_ty?)?;
        Some(Resolved::Value(single, Vec::new()))
    }

    /// The resolution of a dereference of a pointer whose
    /// address denotes a function: a function pointer exposed
    /// as an integer (a `PtrToInt` rvalue records the identity
    /// in `exposed_fn_ptrs`) and rebuilt as a data pointer --
    /// the `foo as usize as *const u8` shape code-inspection
    /// code uses to read a function's machine code. The
    /// pointee is the function's code image: a deterministic
    /// byte stream keyed by the body's canonical form (see
    /// [`Self::code_seed`]), shaped here into the pointee
    /// type -- a scalar integer assembled little-endian from
    /// the stream, or a byte array's worth of it. Two
    /// functions whose converted bodies are canonically
    /// identical share an image, exactly as two functions
    /// codegen reduces to the same instruction stream share
    /// their bytes natively; that is the property
    /// cross-crate-inlining tests read through this shape.
    /// `None` when the address denotes no recorded function,
    /// the function has no converted body, or the pointee
    /// type is not a shape a byte stream can fill.
    pub(super) fn fn_code_deref(
        &self,
        value: &Value,
        pointee_ty: Option<&Ty>,
    ) -> Option<Resolved> {
        let Value::Ref(RefTarget::Address(addr)) = value else {
            return None;
        };
        // A function's synthetic address is its slot base
        // (see `synthetic_address`): the slot an address
        // falls in names the function, and the in-slot
        // displacement is the offset into its code image.
        let base = addr & !0xFFFFu64;
        let offset = (addr - base) as usize;
        let target = self.exposed_fn_ptrs.borrow().get(&base)?.clone();
        let name = match &target {
            Value::FnPtr(name) => name.clone(),
            Value::Closure { def, env } if env.is_empty() => def.clone(),
            _ => return None,
        };
        let seed = self.code_seed(&name)?;
        let value = match pointee_ty? {
            Ty::Bool => Value::Bool(code_byte(seed, offset) != 0),
            Ty::Uint(ty) => {
                let bits = ty.bits();
                Value::Uint(wrap_unsigned(code_int(seed, offset, bits), bits))
            }
            Ty::Int(ty) => {
                let bits = ty.bits();
                Value::Int(wrap_signed(
                    code_int(seed, offset, bits) as i128,
                    bits,
                ))
            }
            Ty::Array(elem, n)
                if matches!(&**elem, Ty::Uint(UintTy::U8)) && *n <= (1 << 20) =>
            {
                Value::Bytes(
                    (0..*n as usize)
                        .map(|i| code_byte(seed, offset + i))
                        .collect(),
                )
            }
            _ => return None,
        };
        Some(Resolved::Value(value, Vec::new()))
    }

    /// The seed of a function's code image: an FNV-1a hash
    /// of the body's canonical form -- its argument count,
    /// its locals' types and its blocks, the parts that
    /// determine what the function does. The body's name
    /// and its source and debug records take no part:
    /// natively, a function's bytes carry neither its name
    /// nor its source positions, so two bodies that differ
    /// only in those share an image here too. `None` when
    /// the program holds no converted body for the name: a
    /// builtin-modelled function has no code in this model,
    /// and a read of its bytes keeps the address-only
    /// verdict.
    fn code_seed(&self, name: &str) -> Option<u64> {
        if let Some(seed) = self.code_seeds.borrow().get(name) {
            return Some(*seed);
        }
        let body = self.program().functions.get(name)?;
        let canonical = format!(
            "{}\u{1}{:?}\u{1}{:?}",
            body.arg_count, body.local_tys, body.blocks
        );
        let mut seed: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in canonical.as_bytes() {
            seed ^= u64::from(b);
            seed = seed.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.code_seeds.borrow_mut().insert(name.to_string(), seed);
        Some(seed)
    }

    /// The single value of a zero-sized type, when the type
    /// has one and it can be fabricated: the structural forms
    /// [`zst_value_of_ty`] covers, and a single-variant ADT
    /// whose fields are all single-valued (their declared
    /// types are the program's recorded field skeletons).
    /// `None` for a type of unknown or nonzero size, a
    /// multi-variant ADT (its zero-sized values are not
    /// unique), or a field whose skeleton names no type.
    pub(crate) fn zst_single_value(&self, ty: &Ty) -> Option<Value> {
        if self.ty_size_full(ty) != Some(0) {
            return None;
        }
        if let Some(value) = zst_value_of_ty(ty) {
            return Some(value);
        }
        match ty {
            Ty::Adt(name) => {
                let variants = self.program().adt_field_tys.get(name)?;
                if variants.len() != 1 {
                    return None;
                }
                let mut fields = Vec::with_capacity(variants[0].len());
                for field in &variants[0] {
                    fields.push(self.zst_single_value(field.as_ref()?)?);
                }
                Some(Value::Enum {
                    variant: 0,
                    fields,
                })
            }
            _ => None,
        }
    }

    /// Follow a reference value to what it points at.
    pub(crate) fn deref_value(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Resolved> {
        match value {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                projection,
            }) => Ok(Resolved::Place {
                frame_pos: self.frame_pos(frames, *frame)?,
                local: *local,
                projection: projection.clone(),
            }),
            Value::Ref(RefTarget::Value(v, projection)) => {
                Ok(Resolved::Value((**v).clone(), projection.clone()))
            }
            Value::Ref(RefTarget::Static { name, projection }) => Ok(Resolved::Static {
                name: name.clone(),
                projection: projection.clone(),
            }),
            Value::Ref(RefTarget::Alloc { id, projection }) => Ok(Resolved::Alloc {
                id: *id,
                projection: projection.clone(),
            }),
            Value::Ref(RefTarget::Null) => Err(InterpError::type_mismatch(
                "dereference",
                "reference",
                "null pointer",
            )),
            Value::Ref(RefTarget::Address(addr)) => {
                // A registered synthetic address denotes its
                // allocation in the store (see `address_owners`):
                // dereferencing it reaches the stored value,
                // exactly as dereferencing the allocation pointer
                // the address was minted from would. An
                // unregistered address denotes nothing readable.
                match self.address_owner(*addr) {
                    Some(id) => Ok(Resolved::Alloc {
                        id,
                        projection: Vec::new(),
                    }),
                    None => Err(InterpError::type_mismatch(
                        "dereference",
                        "reference",
                        "address-only pointer (no pointee in this model)",
                    )),
                }
            }
            // A byte-array reference local holds the byte data
            // itself (see `normalize_for_ty`), and a `&str` value
            // that was materialised as bare string data likewise:
            // dereferencing yields the data value.
            Value::Bytes(_) | Value::Str(_) => {
                Ok(Resolved::Value(value.clone(), Vec::new()))
            }
            // Dereferencing a fat pointer whose base is a reference
            // reaches through to the base place. A self-contained
            // fat pointer (a slice constant: the base is the element
            // aggregate itself) derefs to the slice value itself, so
            // `Len` and index bounds see the fat pointer's length.
            Value::FatRef { base, .. } => match &**base {
                // A sparse element aggregate (see
                // `Value::SparseVec`) is self-contained in the
                // same way: the fat pointer's own length is the
                // slice's length.
                Value::Tuple(_) | Value::SparseVec { .. } => {
                    Ok(Resolved::Value(value.clone(), Vec::new()))
                }
                // An address-only base has no pointee to reach: the
                // fat pointer is self-contained (a dangling slice
                // constant, a `from_raw_parts` result), and
                // dereferencing it yields the slice value itself —
                // the same treatment as the aggregate-base case.
                Value::Ref(RefTarget::Address(_)) => {
                    Ok(Resolved::Value(value.clone(), Vec::new()))
                }
                _ => self.deref_value(frames, base),
            },
            // Dereferencing a dyn fat pointer reaches the value it
            // was formed over: through the base pointer for a
            // reference or slice base, and to the held pointee for
            // a box base (the box aggregate itself — exactly what
            // `Deref::deref` for `Box` yields).
            Value::DynRef { base, .. } => {
                // A reference-counted base names its pointee in the
                // allocation store, not inside the handle value.
                // The handle reading is licensed by the
                // allocation's contents (`rc_inner_id`), not by
                // the base's shape alone: a plain aggregate can
                // share the handle's shape (a `CString` value is
                // a one-field aggregate over a one-field box
                // aggregate over an allocation pointer), and
                // reading it as a handle would name a position
                // inside its buffer instead of the value.
                if let Some(id) = self.rc_inner_id(base) {
                    return Ok(Resolved::Alloc {
                        id,
                        projection: vec![RefProj::Field(2)],
                    });
                }
                match &**base {
                    // A built box base: its single field is the
                    // allocation pointer, and the pointee lives in
                    // the allocation store behind it.
                    Value::Enum { fields, .. }
                        if fields.len() == 1
                            && matches!(fields[0], Value::Ref(RefTarget::Alloc { .. })) =>
                    {
                        self.deref_value(frames, &fields[0])
                    }
                    Value::Enum { fields, .. } if fields.len() == 1 => {
                        Ok(Resolved::Value(fields[0].clone(), Vec::new()))
                    }
                    // An aggregate-built box base (a
                    // `Box::from_raw` product): the `Unique`
                    // sits at field 0 beside the allocator
                    // field, so the single-field arms above do
                    // not apply; the data-pointer extraction
                    // names the pointer the pointee lives
                    // behind.
                    Value::Enum { fields, .. } if fields.len() > 1 => {
                        let ptr = self.dyn_data_pointer(base);
                        self.deref_value(frames, &ptr)
                    }
                    _ => self.deref_value(frames, base),
                }
            },
            // Dereferencing a built box value itself (or a
            // one-field pointer wrapper holding an allocation
            // pointer, such as a `NonNull` extracted from a built
            // box) reaches the allocation's storage: the field is
            // the pointer, and the pointee lives behind it.
            Value::Enum { fields, .. }
                if fields.len() == 1
                    && matches!(fields[0], Value::Ref(RefTarget::Alloc { .. })) =>
            {
                match &fields[0] {
                    Value::Ref(RefTarget::Alloc { id, projection }) => {
                        Ok(Resolved::Alloc {
                            id: *id,
                            projection: projection.clone(),
                        })
                    }
                    _ => unreachable!("field shape checked by the guard"),
                }
            }
            other => Err(InterpError::type_mismatch(
                "dereference",
                "reference",
                other.kind_name(),
            )),
        }
    }

    /// Whether a fat pointer's base sits at the sequence
    /// container itself: its target value is the element
    /// aggregate (a tuple of elements, a sparse sequence, a byte
    /// string) and its projection does not already end at an
    /// element position (an index step). See the `FatRef` arm of
    /// [`Self::ptr_field_value`].
    fn fat_base_denotes_sequence(&self, frames: &[Frame], base: &Value) -> bool {
        let Ok(target) = self.deref_value(frames, base) else {
            return false;
        };
        let projection = match &target {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. }
            | Resolved::Value(_, projection) => projection,
        };
        if matches!(projection.last(), Some(RefProj::Index(_))) {
            return false;
        }
        matches!(
            self.read_resolved(frames, &target),
            Ok(Value::Tuple(_)
                | Value::SparseVec { .. }
                | Value::Bytes(_)
                | Value::Str(_))
        )
    }

    /// The pointer a transparent-wrapper cast produces (see
    /// [`Rvalue::PtrField`]): the operand pointer with a field-0
    /// projection appended, so it denotes the wrapper's only
    /// field. A pointer to a constant aggregate is projected
    /// eagerly (such pointers are read-only in this model anyway,
    /// so nothing observable is lost); null and address-only
    /// pointers denote no field and pass through unchanged.
    pub(crate) fn ptr_field_value(
        &self,
        frames: &[Frame],
        body: &Body,
        value: Value,
        depth: u32,
    ) -> InterpResult<Value> {
        match value {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                mut projection,
            }) => {
                projection.extend((0..depth).map(|_| RefProj::Field(0)));
                Ok(Value::Ref(RefTarget::Place {
                    frame,
                    local,
                    projection,
                }))
            }
            Value::Ref(RefTarget::Static {
                name,
                mut projection,
            }) => {
                projection.extend((0..depth).map(|_| RefProj::Field(0)));
                Ok(Value::Ref(RefTarget::Static { name, projection }))
            }
            Value::Ref(RefTarget::Alloc {
                id,
                mut projection,
            }) => {
                projection.extend((0..depth).map(|_| RefProj::Field(0)));
                Ok(Value::Ref(RefTarget::Alloc { id, projection }))
            }
            Value::Ref(RefTarget::Value(v, projection)) => {
                let mut projection = projection.clone();
                projection.extend((0..depth).map(|_| RefProj::Field(0)));
                Ok(Value::Ref(RefTarget::Value(v.clone(), projection)))
            }
            Value::Ref(RefTarget::Null) | Value::Ref(RefTarget::Address(_)) => Ok(value),
            Value::FatRef { base, len } => match *base {
                Value::Ref(_) => {
                    // A fat pointer's base denotes the sequence's
                    // first element position. When the base sits at
                    // the sequence container itself (its target is
                    // the element aggregate), the entry the cast
                    // denotes is into the first element, so the
                    // position is first re-expressed as that
                    // element (an index step) — the shape the
                    // projection composition machinery (pointer
                    // offsets, fat index composition) already
                    // ranges over when it advances the position to
                    // the later elements. A base that already ends
                    // at an element position, and one whose target
                    // is a single wrapper aggregate around the
                    // sequence (a whole-buffer
                    // `MaybeUninit<[T; N]>`), keeps the plain field
                    // entry: there the field step enters the
                    // wrapper, not an element.
                    let base = if self.fat_base_denotes_sequence(frames, &base) {
                        fat_base_first_element(*base)
                    } else {
                        *base
                    };
                    let base = self.ptr_field_value(frames, body, base, depth)?;
                    Ok(Value::FatRef {
                        base: Box::new(base),
                        len,
                    })
                }
                ref other => Err(InterpError::type_mismatch(
                    "transparent wrapper pointer cast",
                    "fat pointer over a reference",
                    other.kind_name(),
                )),
            },
            other => Err(InterpError::type_mismatch(
                "transparent wrapper pointer cast",
                "pointer to an UnsafeCell/MaybeUninit wrapper",
                other.kind_name(),
            )),
        }
    }


    /// The pointer an array / slice / `str` to element cast
    /// produces (see [`Rvalue::PtrFirstElem`]): the operand pointer
    /// retargeted at the aggregate's first element. A thin pointer
    /// denotes the aggregate itself (its type says so), so an
    /// index-0 projection is appended. A fat pointer's base can
    /// denote either the slice's own storage (a `Vec` value, an
    /// unsized array value: the aggregate whose elements are the
    /// slice's) or its first element's position in the container
    /// the slice lives in (a raw-pointer-derived subslice base).
    /// The storage case takes the appended index 0 like the thin
    /// case; the element case is already the first-element
    /// pointer, and appending index 0 there would denote the
    /// element's own first element instead. A base position is the
    /// element case exactly when its projection already ends at
    /// an element position (an index step, possibly followed by
    /// whole-value entry steps into that element) and the value
    /// there is not the slice's storage (see
    /// [`Self::deref_denotes_storage`]); a position
    /// denoting a whole scalar or aggregate (a one-element slice
    /// over a lone scalar, say) is storage whatever its shape. The projection is carried
    /// on the target like every other retargeting in this model
    /// (a value target's projection applies when the reference
    /// resolves); a fat pointer loses its metadata, as the cast's
    /// result is a thin pointer; null and address-only pointers
    /// denote no element and pass through unchanged.
    /// A `Vec` value assembled in its real aggregate form
    /// ([buffer, length], see `crate::builtins::vec_real_aggregate_parts`)
    /// translated to the element form every consumer reads:
    /// the elements are the ones the buffer pointer denotes,
    /// read through it as a slice of the declared length, and
    /// the result is the vector value at that length (see
    /// `crate::builtins::vec_with_len`). `None` when the value
    /// is not in the real form.
    pub(crate) fn real_vec_buffer_ptr(value: &Value) -> Option<Value> {
        let (buf, _) = crate::builtins::vec_real_aggregate_parts(value)?;
        let Value::Enum {
            fields: buf_fields, ..
        } = &buf
        else {
            return None;
        };
        let Value::Enum {
            fields: inner, ..
        } = buf_fields.first()?
        else {
            return None;
        };
        match inner.first()? {
            Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
                v @ (Value::Ref(_) | Value::FatRef { .. }) => Some(v.clone()),
                _ => None,
            },
            v @ (Value::Ref(_) | Value::FatRef { .. }) => Some(v.clone()),
            _ => None,
        }
    }

    /// The capacity a vector value in its real aggregate
    /// form stores in its `RawVecInner` slot (the one-field
    /// `UsizeNoHighBit` wrapper around the count), or `None`
    /// for any other shape.
    pub(crate) fn real_vec_cap(value: &Value) -> Option<u128> {
        let (buf, _) = crate::builtins::vec_real_aggregate_parts(value)?;
        let Value::Enum {
            fields: buf_fields, ..
        } = &buf
        else {
            return None;
        };
        let Value::Enum {
            fields: inner, ..
        } = buf_fields.first()?
        else {
            return None;
        };
        match inner.get(1)? {
            Value::Uint(n) => Some(*n),
            Value::Int(n) if *n >= 0 => Some(*n as u128),
            Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
                Value::Uint(n) => Some(*n),
                Value::Int(n) if *n >= 0 => Some(*n as u128),
                _ => None,
            },
            _ => None,
        }
    }

    pub(crate) fn vec_from_real_form(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> Option<InterpResult<Value>> {
        let (buf, len) = crate::builtins::vec_real_aggregate_parts(value)?;
        // The capacity the real form's `RawVec` declares: the
        // translated value keeps accounting that storage (see
        // `vec_with_len`), so a capacity the guest's own
        // aggregate records is not lost in translation.
        let prev_cap = crate::builtins::vec_real_aggregate_cap(&buf).unwrap_or(0);
        // The buffer's pointer: the `RawVec`'s inner triple
        // (`RawVecInner`) holds it in a `NonNull` wrapper at
        // field 0.
        let ptr = (|| {
            let Value::Enum {
                fields: buf_fields, ..
            } = &buf
            else {
                return None;
            };
            let Value::Enum {
                fields: inner, ..
            } = buf_fields.first()?
            else {
                return None;
            };
            match inner.first()? {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    Some(fields[0].clone())
                }
                v @ (Value::Ref(_) | Value::FatRef { .. }) => Some(v.clone()),
                _ => None,
            }
        })();
        // A zero-length vector's elements are no elements,
        // whatever its buffer pointer denotes (a dangling
        // address-only pointer an allocator model handed out
        // for a `with_capacity` buffer has no pointee to read).
        // A capacity the real form declares is still the
        // vector's storage: the sparse form accounts it over
        // the empty element list.
        if len == 0 {
            return Some(Ok(if prev_cap > 0 {
                Value::SparseVec {
                    elems: Vec::new(),
                    len: 0,
                    cap: prev_cap,
                }
            } else {
                Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                }
            }));
        }
        // Reading the elements can fail for a buffer the model
        // does not hold (an address-only allocation): the
        // translation then declines — the caller keeps the real
        // form, exactly as it would have without the
        // translation — instead of failing the return.
        let result = match ptr {
            Some(p @ Value::Ref(_)) => {
                let fat = Value::FatRef {
                    base: Box::new(p),
                    len: len as u128,
                };
                self.slice_elements(frames, &fat).map(|(elems, _)| {
                    crate::builtins::vec_with_len(elems, len, &Value::Uninit, prev_cap)
                })
            }
            Some(fat @ Value::FatRef { .. }) => self
                .slice_elements(frames, &fat)
                .map(|(elems, _)| {
                    crate::builtins::vec_with_len(elems, len, &Value::Uninit, prev_cap)
                }),
            _ => Err(InterpError::type_mismatch(
                "Vec value in its real aggregate form",
                "a buffer pointer in the RawVec fields",
                "no pointer value",
            )),
        };
        match result {
            Ok(value) => Some(Ok(value)),
            Err(_) => None,
        }
    }

    pub(crate) fn ptr_first_elem_value(
        &self,
        frames: &[Frame],
        body: &Body,
        value: Value,
        elem_string: Option<bool>,
        elem_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        // A target holding a scalar denotes the scalar's own
        // storage: its first element is the scalar itself, so
        // the first-element pointer is the pointer as it
        // stands. Appending the element index the arms below
        // add would denote a byte of the storage instead (the
        // reading `project_value` gives a bare index step on a
        // scalar), which is the shape a slice formed over a
        // scalar place (`slice::from_ref` / `slice::from_mut`)
        // takes through `as_ptr`. A constant value target is
        // not storage: its element position is the appended
        // index the arms below form, which the offset machinery
        // advances, so the shortcut is for storage targets only.
        if let Value::Ref(target) = &value {
            let storage_backed = matches!(
                target,
                RefTarget::Place { .. }
                    | RefTarget::Static { .. }
                    | RefTarget::Alloc { .. }
            );
            if storage_backed {
                if let Ok(resolved) = self.deref_value(frames, &value) {
                    if matches!(
                        self.read_resolved(frames, &resolved),
                        Ok(v) if scalar_storage_len(&v).is_some()
                    ) {
                        // A byte-width element of a different
                        // scalar kind than the storage's own (a
                        // `&[u8]` over a `bool`) does not denote
                        // the scalar itself: its first element is
                        // the storage's byte, the appended-index
                        // form the arms below produce (the same
                        // judgment `fat_denotes_scalar_element`
                        // makes for element positions).
                        if !self.scalar_storage_is_byte_view(
                            frames,
                            &resolved,
                            elem_ty,
                        ) {
                            return Ok(value);
                        }
                    }
                }
            }
        }
        match value {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                mut projection,
            }) => {
                projection.push(RefProj::Index(0));
                Ok(Value::Ref(RefTarget::Place {
                    frame,
                    local,
                    projection,
                }))
            }
            Value::Ref(RefTarget::Static {
                name,
                mut projection,
            }) => {
                projection.push(RefProj::Index(0));
                Ok(Value::Ref(RefTarget::Static { name, projection }))
            }
            Value::Ref(RefTarget::Alloc {
                id,
                mut projection,
            }) => {
                projection.push(RefProj::Index(0));
                Ok(Value::Ref(RefTarget::Alloc { id, projection }))
            }
            Value::Ref(RefTarget::Value(v, mut projection)) => {
                projection.push(RefProj::Index(0));
                Ok(Value::Ref(RefTarget::Value(v, projection)))
            }
            Value::Ref(RefTarget::Null) | Value::Ref(RefTarget::Address(_)) => Ok(value),
            Value::FatRef { base, len } => match *base {
                Value::Ref(_) => {
                    let resolved = self.deref_value(frames, &base)?;
                    let ends_at_element = match &resolved {
                        Resolved::Place { projection, .. }
                        | Resolved::Static { projection, .. }
                        | Resolved::Alloc { projection, .. }
                        | Resolved::Value(_, projection) => {
                            projection_ends_at_element(projection)
                        }
                    };
                    if ends_at_element
                        && !self.deref_denotes_storage(frames, &resolved, len, elem_string)
                    {
                        Ok(*base)
                    } else {
                        self.ptr_first_elem_value(frames, body, *base, elem_string, elem_ty)
                    }
                }
                Value::Tuple(_) | Value::Bytes(_) | Value::Str(_) => Ok(Value::Ref(
                    RefTarget::Value(base, vec![RefProj::Index(0)]),
                )),
                ref other => Err(InterpError::type_mismatch(
                    "first-element pointer cast",
                    "fat pointer over a reference or an aggregate value",
                    other.kind_name(),
                )),
            },
            other => Err(InterpError::type_mismatch(
                "first-element pointer cast",
                "pointer to an array, slice or string",
                other.kind_name(),
            )),
        }
    }

    /// The pointer an element-to-array cast produces (see
    /// [`Rvalue::PtrArrayBase`]): the operand pointer with a
    /// trailing index-0 projection removed, so it denotes the
    /// array as a whole again. Any other shape passes through
    /// unchanged: a pointer to an element past the first has no
    /// array-base representation in this model.
    /// The element values of an array value: a tuple of
    /// elements, or the bytes of a byte array (each byte is its
    /// own element value). Anything else is not an array value.
    fn array_elem_values(value: Value) -> InterpResult<Vec<Value>> {
        match value {
            Value::Tuple(elems) => Ok(elems),
            Value::Bytes(bytes) => Ok(bytes
                .into_iter()
                .map(|b| Value::Uint(b as u128))
                .collect()),
            other => Err(InterpError::type_mismatch(
                "array element wrapper transmute",
                "an array value",
                other.kind_name(),
            )),
        }
    }

    /// `[T; N]` -> `[MaybeUninit<T>; N]` (or the `UnsafeCell`
    /// twin): each element takes the wrapper's value
    /// representation, an ADT value holding the payload at
    /// field 0 -- the shape `MaybeUninit::new` constructions
    /// are normalized to.
    pub(crate) fn wrap_array_elems(value: Value) -> InterpResult<Value> {
        Ok(Value::Tuple(
            Self::array_elem_values(value)?
                .into_iter()
                .map(|elem| Value::Enum {
                    variant: 0,
                    fields: vec![elem],
                })
                .collect(),
        ))
    }

    /// The reverse cast: each element's wrapper representation
    /// is unwrapped to the payload it holds. The payload sits
    /// at field 0 in the normalized representation, or in the
    /// union's value field (index 1) in a raw aggregate that
    /// still carries the uninitialized field before it.
    pub(crate) fn unwrap_array_elems(value: Value) -> InterpResult<Value> {
        let elems = Self::array_elem_values(value)?
            .into_iter()
            .map(|elem| match elem {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    Ok(fields.into_iter().next().expect("one field"))
                }
                Value::Enum { fields, .. } if fields.len() == 2 => {
                    Ok(fields.into_iter().nth(1).expect("two fields"))
                }
                other => Err(InterpError::type_mismatch(
                    "array element wrapper transmute",
                    "a wrapper element value",
                    other.kind_name(),
                )),
            })
            .collect::<InterpResult<Vec<_>>>()?;
        Ok(Value::Tuple(elems))
    }

    pub(crate) fn ptr_array_base_value(&self, value: Value) -> InterpResult<Value> {
        match value {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                mut projection,
            }) => {
                if projection.last() == Some(&RefProj::Index(0)) {
                    projection.pop();
                }
                Ok(Value::Ref(RefTarget::Place {
                    frame,
                    local,
                    projection,
                }))
            }
            Value::Ref(RefTarget::Static {
                name,
                mut projection,
            }) => {
                if projection.last() == Some(&RefProj::Index(0)) {
                    projection.pop();
                }
                Ok(Value::Ref(RefTarget::Static { name, projection }))
            }
            Value::Ref(RefTarget::Alloc {
                id,
                mut projection,
            }) => {
                if projection.last() == Some(&RefProj::Index(0)) {
                    projection.pop();
                }
                Ok(Value::Ref(RefTarget::Alloc { id, projection }))
            }
            Value::Ref(RefTarget::Value(v, mut projection)) => {
                if projection.last() == Some(&RefProj::Index(0)) {
                    projection.pop();
                }
                Ok(Value::Ref(RefTarget::Value(v, projection)))
            }
            other => Ok(other),
        }
    }

    /// The borrowing form of [`read_resolved`]: a place read lends
    /// the projected value out of the frame's storage instead of
    /// cloning it, for consumers that only inspect the value (or
    /// clone a small part of it). Reads out of the statics and
    /// allocation stores cannot lend (the stores sit behind
    /// `RefCell`s), and a detached `Resolved::Value` is itself a
    /// copy, so those arms yield an owned value — exactly what
    /// `read_resolved` returns. Errors are identical.
    /// The outcome of a resolved read that found nothing but
    /// the uninitialized marker: at a `MaybeUninit` place the
    /// marker IS the value -- the uninitialized union value,
    /// the payload being uninitialized being that type's
    /// ordinary content (a read of the payload itself still
    /// raises at its own point of use, exactly as the payload
    /// of a written `MaybeUninit::uninit()` value does). At
    /// any other place the read stays the
    /// `UninitializedRead` the untyped layer reports.
    fn uninit_read_value(
        &self,
        ty: Option<&Ty>,
        func: String,
        place: String,
    ) -> InterpResult<Value> {
        match ty {
            Some(t) if is_maybe_uninit_ty(t) => Ok(maybe_uninit_form(Value::Uninit)),
            _ => Err(InterpError::UninitializedRead { func, place }),
        }
    }

    pub(crate) fn read_resolved_ref<'x>(
        &self,
        frames: &'x [Frame],
        target: &Resolved,
    ) -> InterpResult<Cow<'x, Value>> {
        self.read_resolved_ref_typed(frames, target, None)
    }

    /// [`read_resolved_ref`] with the declared type of the
    /// place being read (see [`read_resolved_typed`]).
    pub(crate) fn read_resolved_ref_typed<'x>(
        &self,
        frames: &'x [Frame],
        target: &Resolved,
        ty: Option<&Ty>,
    ) -> InterpResult<Cow<'x, Value>> {
        match target {
            Resolved::Value(base, projection) => {
                let value = self.value_target_value(base, projection)?;
                if value == Value::Uninit {
                    return self
                        .uninit_read_value(
                            ty,
                            "<constant>".to_string(),
                            "constant aggregate".to_string(),
                        )
                        .map(Cow::Owned);
                }
                Ok(Cow::Owned(value))
            }
            Resolved::Static { .. } | Resolved::Alloc { .. } => {
                Ok(Cow::Owned(self.read_resolved_typed(frames, target, ty)?))
            }
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                // Union field reads need the typed projector and
                // the owned path's rescue rules; they take the
                // owned read unchanged (the pre-GK behaviour).
                if projection
                    .iter()
                    .any(|step| matches!(step, RefProj::UnionField { .. }))
                {
                    return Ok(Cow::Owned(self.read_resolved_typed(frames, target, ty)?));
                }
                let frame = &frames[*frame_pos];
                let base = frame.locals.get(*local).ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: frame.func.clone(),
                        local: *local,
                    }
                })?;
                let place_desc = if projection.is_empty() {
                    String::new()
                } else {
                    describe_resolved(*local, projection)
                };
                if *base == Value::Uninit {
                    if let Some(zst) = self.zst_local_value(frame, *local) {
                        let value = project_value(zst, projection, &frame.func, &place_desc)?;
                        if value == Value::Uninit {
                            return self
                                .uninit_read_value(
                                    ty,
                                    frame.func.clone(),
                                    if projection.is_empty() {
                                        describe_resolved(*local, projection)
                                    } else {
                                        place_desc
                                    },
                                )
                                .map(Cow::Owned);
                        }
                        return Ok(Cow::Owned(value));
                    }
                }
                // The projection's type walk is rooted at the
                // local's declared type, exactly as the owned
                // read's is (see `read_resolved_typed`): the
                // caller's `ty` is the place's *result* type and
                // judges only the uninitialized outcome below.
                // Rooting the walk there instead would judge the
                // wrapper descent against the result type -- for
                // an elided field read, the field's own type --
                // and let the descent take a sibling's contents
                // for the field's value.
                let root_ty = self
                    .program()
                    .get(&frame.func)
                    .and_then(|body| body.local_ty(Local(*local)))
                    .cloned();
                let value = match project_value_ref_ev(
                    Some(self.program()),
                    root_ty,
                    base,
                    projection,
                    &frame.func,
                    &place_desc,
                ) {
                    Ok(v) if *v != Value::Uninit => v,
                    Ok(_) | Err(InterpError::UninitializedRead { .. }) => {
                        // Mirror the owned read: a single-valued
                        // field denotes its value when nothing was
                        // ever written there.
                        match self.zst_field_value(frame, *local, projection) {
                            Some(v) => return Ok(Cow::Owned(v)),
                            None => {
                                return self
                                    .uninit_read_value(
                                        ty,
                                        frame.func.clone(),
                                        if projection.is_empty() {
                                            describe_resolved(*local, projection)
                                        } else {
                                            place_desc
                                        },
                                    )
                                    .map(Cow::Owned);
                            }
                        }
                    }
                    Err(e) => return Err(e),
                };
                Ok(value)
            }
        }
    }
}

impl<'a> Interpreter<'a> {
    pub(crate) fn read_resolved(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> InterpResult<Value> {
        self.read_resolved_typed(frames, target, None)
    }

    /// [`read_resolved`] with the declared type of the place
    /// being read, when the caller knows it (a MIR place read
    /// passes its place's type; a bulk copy passes its element
    /// type). The type judges only the uninitialized outcome:
    /// a read that finds nothing but the uninitialized marker
    /// at a `MaybeUninit` place yields that type's value (see
    /// [`Self::uninit_read_value`]); every other outcome --
    /// values and errors alike -- is the untyped read's.
    pub(crate) fn read_resolved_typed(
        &self,
        frames: &[Frame],
        target: &Resolved,
        ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match target {
            Resolved::Value(base, projection) => {
                let value = self.value_target_value(base, projection)?;
                if value == Value::Uninit {
                    return self.uninit_read_value(
                        ty,
                        "<constant>".to_string(),
                        "constant aggregate".to_string(),
                    );
                }
                Ok(value)
            }
            Resolved::Static { name, projection } => {
                let base = self.static_base(name)?;
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| "<static>".to_string());
                let place_desc = format!("static {name}");
                let root_ty = self.program().static_tys.get(name).cloned();
                let value =
                    self.project_value_t(base, projection, &func, &place_desc, root_ty)?;
                if value == Value::Uninit {
                    return self.uninit_read_value(ty, func, place_desc);
                }
                Ok(value)
            }
            Resolved::Alloc { id, projection } => {
                let projection = &self.elem_block_projection(*id, projection);
                let base = self.alloc_base(*id)?;
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| "<allocation>".to_string());
                let place_desc = format!("allocation #{id}");
                // An elem block's stored tuple holds only the
                // elements written so far (it grows on write; see
                // the allocation arm of `write_resolved`): a read
                // whose first index step lands past the tuple's
                // current end addresses a slot that was never
                // written. That is an uninitialized read, not a
                // guest bounds violation -- the block's extent is
                // its byte size, which the tuple's length does
                // not measure, so the projector's bounds panic
                // does not apply to it.
                let past_unwritten = self.is_elem_block(*id)
                    && match (&base, projection.first()) {
                        (Value::Tuple(items), Some(RefProj::Index(i))) => {
                            *i >= items.len()
                        }
                        _ => false,
                    };
                let root_ty = self.alloc_ty(*id);
                let value = match self.project_value_t(
                    base.clone(),
                    projection,
                    &func,
                    &place_desc,
                    root_ty,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        // The stored form of an allocation is a
                        // prefix of its extent: an index step the
                        // projector bounds against the prefix
                        // addresses a slot that was never written,
                        // exactly as `past_unwritten` judges the
                        // block's own first step. The projector's
                        // bounds panic measures the prefix, not
                        // the allocation, so it is an uninitialized
                        // read here, never a guest bounds
                        // violation (a guest panic is control
                        // flow, not a read result).
                        let prefix_bounds = is_index_oob_error(&e);
                        if past_unwritten || prefix_bounds {
                            return self.uninit_read_value(ty, func, place_desc);
                        }
                        return Err(e);
                    }
                };
                if value == Value::Uninit {
                    return self.uninit_read_value(ty, func, place_desc);
                }
                Ok(value)
            }
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let frame = &frames[*frame_pos];
                let base = frame.locals.get(*local).cloned().ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: frame.func.clone(),
                        local: *local,
                    }
                })?;
                // A zero-sized local's place may never be written
                // by rustc MIR at all (a unit struct constructor's
                // body is a bare `return`): an unwritten read of
                // such a local denotes the type's single value.
                let base = if base == Value::Uninit {
                    self.zst_local_value(frame, *local).unwrap_or(base)
                } else {
                    base
                };
                // The description is only read inside projection
                // steps and by the error below, so with an empty
                // projection it is built only if that error fires.
                let place_desc = if projection.is_empty() {
                    String::new()
                } else {
                    describe_resolved(*local, projection)
                };
                // A field whose type has a single value denotes
                // that value when nothing was ever written there:
                // the store of a zero-sized field is elided like a
                // ZST local's (the closure slot of a `catch_unwind`
                // Data union is read back by `do_call` without any
                // body having stored it). The stored value wins
                // whenever the read succeeds — a union's storage
                // slot can be shared with a field of a different
                // type — so the precomputed value (conversion's
                // `Body::zst_fields` table) only rescues a read
                // that otherwise fails as uninitialized.
                let root_ty = self
                    .program()
                    .get(&frame.func)
                    .and_then(|body| body.local_ty(Local(*local)))
                    .cloned();
                let value = match self.project_value_t(
                    base,
                    projection,
                    &frame.func,
                    &place_desc,
                    root_ty,
                ) {
                    Ok(v) if v != Value::Uninit => v,
                    Ok(_) | Err(InterpError::UninitializedRead { .. }) => {
                        match self.zst_field_value(frame, *local, projection) {
                            Some(v) => v,
                            None => {
                                return self.uninit_read_value(
                                    ty,
                                    frame.func.clone(),
                                    if projection.is_empty() {
                                        describe_resolved(*local, projection)
                                    } else {
                                        place_desc
                                    },
                                );
                            }
                        }
                    }
                    Err(e) => return Err(e),
                };
                Ok(value)
            }
        }
    }

    /// The value of a never-written zero-sized local in `frame`
    /// (see the note in `read_resolved`); `None` for any other
    /// local. Conversion records the canonical value of every
    /// provably zero-sized local in the body (`Body::zst_locals`,
    /// computed from the driver's ADT shape tables); for
    /// hand-built bodies the shapes derivable from the IR type
    /// alone are synthesized by [`zst_value_of_ty`].
    pub(crate) fn zst_local_value(&self, frame: &Frame, local: usize) -> Option<Value> {
        let body = self.program.get(&frame.func)?;
        if let Some(Some(v)) = body.zst_locals.get(local) {
            return Some(v.clone());
        }
        zst_value_of_ty(body.local_ty(Local(local))?)
    }

    /// The value of a `MaybeUninit` local holding the bare
    /// uninitialized marker: the uninitialized union value (see
    /// `normalize_for_ty`); `None` for any other local.
    pub(crate) fn maybe_uninit_local_value(&self, frame: &Frame, local: usize) -> Option<Value> {
        let body = self.program.get(&frame.func)?;
        if is_maybe_uninit_ty(body.local_ty(Local(local))?) {
            Some(Value::Enum {
                variant: 0,
                fields: vec![Value::Uninit],
            })
        } else {
            None
        }
    }

    /// The canonical value of a single-valued projection path
    /// of a local (see the note in `read_resolved`); `None` when
    /// the projection is not a pure field/downcast/index path or
    /// conversion recorded no single-valued value for it.
    pub(crate) fn zst_field_value(
        &self,
        frame: &Frame,
        local: usize,
        projection: &[RefProj],
    ) -> Option<Value> {
        if projection.is_empty() {
            return None;
        }
        let path = Self::zst_path_of(projection)?;
        let body = self.program.get(&frame.func)?;
        body.zst_fields.get(local)?.get(&path).cloned()
    }

    /// A projection path as the canonical-value path the
    /// conversion-side walk records (see `Body::zst_fields`);
    /// `None` when the projection is not a pure
    /// field/index/downcast path.
    fn zst_path_of(projection: &[RefProj]) -> Option<Vec<ZstPathStep>> {
        let mut path = Vec::with_capacity(projection.len());
        for step in projection {
            match step {
                RefProj::Field(idx) => path.push(ZstPathStep::Field(*idx)),
                // Union fields sit in the same per-parent index
                // space the conversion-side walk records.
                RefProj::UnionField { idx, .. } => {
                    path.push(ZstPathStep::Field(*idx))
                }
                // A downcast step selects the variant whose
                // fields the following steps address; the walk
                // records each variant's fields under its
                // downcast (see `collect_zst_fields`).
                RefProj::Downcast(variant) => {
                    path.push(ZstPathStep::Downcast(*variant as usize))
                }
                // An index step into a sequence: the recorded
                // value is the element type's single value, which
                // does not depend on the index read.
                RefProj::Index(_) => path.push(ZstPathStep::Index),
                _ => return None,
            }
        }
        Some(path)
    }

    /// The kinds of step conversion recorded beneath a
    /// projection path of a local: whether any recorded
    /// canonical-value path continues from `projection` with a
    /// field step, and whether any continues with a downcast
    /// step. The snapshot rescue (see `builtins`) descends into
    /// a detached aggregate value's fields only along the steps
    /// the recorded paths actually take from the value's path:
    /// the value alone does not say whether its field list is a
    /// sequence's elements, a struct's fields or an enum
    /// variant's fields, but the recorded paths do.
    pub(crate) fn zst_recorded_branches(
        &self,
        frame: &Frame,
        local: usize,
        projection: &[RefProj],
    ) -> (bool, bool) {
        let Some(path) = Self::zst_path_of(projection) else {
            return (false, false);
        };
        let Some(body) = self.program.get(&frame.func) else {
            return (false, false);
        };
        let Some(fields) = body.zst_fields.get(local) else {
            return (false, false);
        };
        let mut field = false;
        let mut downcast = false;
        for key in fields.keys() {
            if key.len() > path.len() && key[..path.len()] == path[..] {
                match key[path.len()] {
                    ZstPathStep::Field(_) => field = true,
                    ZstPathStep::Downcast(_) => downcast = true,
                    ZstPathStep::Index => {}
                }
                if field && downcast {
                    break;
                }
            }
        }
        (field, downcast)
    }

    /// The value of a never-written local of an evidently
    /// zero-sized type, for drop glue only (see the note in
    /// `read_at`); `None` for any other local. `read_resolved`
    /// deliberately does not do this: a genuine read of a
    /// never-written place is an uninitialized read, but drop glue
    /// must still run the destructor of a ZST place whose write
    /// optimized MIR elided.
    pub(crate) fn zst_glue_value(&self, frame: &Frame, local: usize) -> Option<Value> {
        let body = self.program.get(&frame.func)?;
        let ty = body.local_ty(Local(local))?;
        Self::zst_value_of(ty)
    }

    /// The unique value of an evidently zero-sized type, when the
    /// type alone determines one. ADT values are fabricated as
    /// variant 0 with no fields: exact for unit-like and fieldless
    /// structs; for payload-carrying types the fabrication is only
    /// a drop-glue gate — field glue beneath it reads as
    /// never-written and is skipped.
    fn zst_value_of(ty: &Ty) -> Option<Value> {
        match ty {
            Ty::Unit => Some(Value::Unit),
            Ty::Tuple(tys) => tys
                .iter()
                .map(Self::zst_value_of)
                .collect::<Option<Vec<_>>>()
                .map(Value::Tuple),
            Ty::Array(elem, n) => {
                if *n > 1_000_000 {
                    return None;
                }
                let elem = Self::zst_value_of(elem)?;
                Some(Value::Tuple(vec![elem; *n as usize]))
            }
            Ty::Closure(def, 0) => Some(Value::Closure {
                def: def.clone(),
                env: Vec::new(),
            }),
            Ty::FnDef(name) => Some(Value::FnPtr(name.clone())),
            Ty::Adt(_) => Some(Value::Enum {
                variant: 0,
                fields: Vec::new(),
            }),
            _ => None,
        }
    }

    /// A cloned snapshot of an allocation's stored value, with
    /// the store's borrow released before the caller continues
    /// (a borrow taken in an `if let` scrutinee would otherwise
    /// outlive the branch and block the caller's next write).
    pub(crate) fn alloc_value_snapshot(&self, id: &u64) -> Option<Value> {
        self.allocs.borrow().get(id).cloned()
    }

    pub(crate) fn write_resolved(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: Value,
    ) -> InterpResult<()> {
        self.write_resolved_typed(frames, target, value, None)
    }

    /// [`Self::write_resolved`] with the caller's type evidence:
    /// `root_ty`, the declared type of the value at the target's
    /// root (for an allocation target, its pointee type), is
    /// walked alongside the projected assignment so the
    /// navigation's wrapper judgments can consult the declared
    /// types (see `wrapper_descent_sanctioned`). Callers without
    /// evidence use `write_resolved`, whose judgments are the
    /// shape-only ones.
    pub(crate) fn write_resolved_typed(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: Value,
        root_ty: Option<&Ty>,
    ) -> InterpResult<()> {
        // A write crossing a union field boundary composes
        // with the held field's bytes (see
        // `union_crossing_write`); every other write takes the
        // ordinary path below.
        let grow = matches!(target, Resolved::Alloc { .. });
        let mut pending = Some(value);
        let crossed = self.union_crossing_write(
            frames,
            target,
            true,
            |temp, rest, func, place| {
                let value = pending
                    .take()
                    .expect("a crossing write applies the value once");
                assign_projected(
                    temp,
                    rest,
                    value,
                    func,
                    place,
                    grow,
                    Some(self.program()),
                    None,
                )
            },
        );
        if let Some(done) = crossed {
            return done;
        }
        let value = pending
            .take()
            .expect("a declined crossing leaves the value unapplied");
        match target {
            Resolved::Value(..) => Err(InterpError::type_mismatch(
                "assignment",
                "mutable place",
                "constant aggregate",
            )),
            Resolved::Static { name, projection } => {
                let mut store = self.statics.borrow_mut();
                let slot = store
                    .get_mut(name)
                    .ok_or_else(|| self.missing_static(name))?;
                if projection.is_empty() {
                    *slot = value;
                    return Ok(());
                }
                let place_desc = format!("static {name}");
                let root_ty = self.program().static_tys.get(name).cloned();
                assign_projected(
                    slot,
                    projection,
                    value,
                    "<static>",
                    &place_desc,
                    false,
                    Some(self.program()),
                    root_ty,
                )
            }
            Resolved::Alloc { id, projection } => {
                let projection = &self.elem_block_projection(*id, projection);
                let mut store = self.allocs.borrow_mut();
                let slot = store
                    .get_mut(id)
                    .ok_or_else(|| self.missing_alloc(*id))?;
                if projection.is_empty() {
                    // A raw block's base pointer denotes its
                    // first element: writing a non-tuple value
                    // through it writes element 0, not the
                    // block as a whole. A tuple value is the
                    // block's own array value and replaces it.
                    if self.is_elem_block(*id)
                        && matches!(&*slot, Value::Tuple(_))
                        && !matches!(value, Value::Tuple(_))
                    {
                        let place_desc = format!("allocation #{id}");
                        if let Value::Tuple(items) = &mut *slot {
                            if items.is_empty() {
                                items.push(Value::Uninit);
                            }
                        }
                        let slot = navigate_mut(
                            slot,
                            &[RefProj::Index(0)],
                            "<allocation>",
                            &place_desc,
                        )?;
                        *slot = value;
                        return Ok(());
                    }
                    *slot = value;
                    return Ok(());
                }
                // A raw block's element tuple grows to the
                // written index: the allocation's byte size is
                // not a value-level element count, so unwritten
                // elements materialize as uninitialized slots
                // (reading one stays an uninitialized-read
                // error, never a fabricated value).
                if self.is_elem_block(*id) {
                    if let Some(RefProj::Index(i)) = projection.first() {
                        let i = *i;
                        // A block no write has materialized yet
                        // holds no value at all: the first element
                        // write materializes the element tuple,
                        // with the slots it skips over
                        // uninitialized, exactly as the grow
                        // below treats a short tuple.
                        // The same materialisation bound as
                        // the tuple growth: an index at or past
                        // it is the navigation's error to
                        // produce, not storage to allocate (at
                        // the extreme index the size computation
                        // itself overflows).
                        if i < crate::builtins::MAX_VEC_SET_LEN_ELEMENTS {
                            if *slot == Value::Uninit {
                                *slot = Value::Tuple(vec![Value::Uninit; i + 1]);
                            }
                            if let Value::Tuple(items) = &mut *slot {
                                if i >= items.len() {
                                    items.resize(i + 1, Value::Uninit);
                                }
                            }
                        }
                    }
                }
                let place_desc = format!("allocation #{id}");
                // The navigation's root type: the projection is
                // rooted at the allocation, so the type walked
                // alongside it is the allocation's own declared
                // type — the record its creation point made,
                // which retargeting a pointer at the same storage
                // does not move (see `record_alloc_ty`). The
                // caller's evidence names the type at the
                // *pointer's* target, which for a pointer to a
                // field inside the allocation is that field's
                // type, not the root's: walking the root-relative
                // projection from it misaligns the type walk and
                // the descent can land in a sibling. The caller's
                // type remains the fallback for an allocation
                // with no type on record, where it names the root
                // in the flows that supply it: a plain
                // allocation's root is the pointee value itself;
                // an element block's root is its element type,
                // seen as the element sequence the block's stored
                // tuple is (an element-block projection starts by
                // indexing into it).
                let alloc_root = self.alloc_ty(*id).or_else(|| root_ty.cloned());
                let alloc_root = match alloc_root {
                    Some(element_ty) if self.is_elem_block(*id) => {
                        Some(Ty::Slice(Box::new(element_ty)))
                    }
                    other => other,
                };
                grow_tuples_along(slot, projection, Some(self.program()), alloc_root.clone());
                assign_projected(
                    slot,
                    projection,
                    value,
                    "<allocation>",
                    &place_desc,
                    true,
                    Some(self.program()),
                    alloc_root,
                )
            }
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                if projection.is_empty() {
                    // Direct local of the target frame: normalize
                    // against that frame's declared type.
                    let func = frames[*frame_pos].func.clone();
                    let program: &'a Program = self.program;
                    let target_body = program.get(&func).ok_or_else(|| {
                        InterpError::UnknownFunction { name: func.clone() }
                    })?;
                    let frame = &mut frames[*frame_pos];
                    let slot = frame.locals.get_mut(*local).ok_or_else(|| {
                        InterpError::UnknownLocal {
                            func: func.clone(),
                            local: *local,
                        }
                    })?;
                    *slot = Self::normalize_for_ty(value, target_body.local_ty(Local(*local)))?;
                    return Ok(());
                }
                let func = frames[*frame_pos].func.clone();
                let root_ty = self
                    .program()
                    .get(&func)
                    .and_then(|body| body.local_ty(Local(*local)))
                    .cloned();
                let frame = &mut frames[*frame_pos];
                let place_desc = describe_resolved(*local, projection);
                let slot = frame.locals.get_mut(*local).ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: func.clone(),
                        local: *local,
                    }
                })?;
                assign_projected(
                    slot,
                    projection,
                    value,
                    &func,
                    &place_desc,
                    false,
                    Some(self.program()),
                    root_ty,
                )
            }
        }
    }

    /// Mutate the value a resolved target denotes in place,
    /// running `f` on the stored slot. Callers must preserve the
    /// value's top-level form (an `Enum` stays the same `Enum`, a
    /// `Str` stays a `Str`): write normalization
    /// ([`Self::normalize_for_ty`]) is the identity for those
    /// forms, so skipping the read-modify-write round trip through
    /// [`Self::write_resolved`] — which clones every element of an
    /// aggregate on the way out and back in — is unobservable.
    pub(crate) fn modify_resolved<R>(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        f: impl FnOnce(&mut Value) -> InterpResult<R>,
    ) -> InterpResult<R> {
        // A modification crossing a union field boundary
        // composes with the held field's bytes (see
        // `union_crossing_write`); every other target takes
        // the paths below. A projection ending in a vector
        // length step is left to the length-slot pre-pass: the
        // length is not storage inside the field value the
        // crossing materialises.
        let grow = matches!(target, Resolved::Place { .. });
        let mut pending = Some(f);
        let crossed = self.union_crossing_write(
            frames,
            target,
            false,
            |temp, rest, func, place| {
                let f = pending
                    .take()
                    .expect("a crossing modification applies once");
                let slot = navigate_mut_grow(
                    temp,
                    rest,
                    func,
                    place,
                    grow,
                    Some(self.program()),
                    None,
                )?;
                f(slot)
            },
        );
        if let Some(done) = crossed {
            return done;
        }
        let f = pending
            .take()
            .expect("a declined crossing leaves the modification unapplied");
        // A read-modify-write through a vector length-slot
        // reference (see [`RefProj::VecLen`]): the slot is not
        // storage inside the vector value, so the modification
        // runs on the declared length as a detached value and
        // the result is stored back through the whole-value
        // write, which applies the `Vec::set_len` storage
        // change to the vector itself.
        let len_head = match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } if projection.last() == Some(&RefProj::VecLen) => {
                Some(Resolved::Place {
                    frame_pos: *frame_pos,
                    local: *local,
                    projection: projection[..projection.len() - 1].to_vec(),
                })
            }
            Resolved::Static { name, projection }
                if projection.last() == Some(&RefProj::VecLen) =>
            {
                Some(Resolved::Static {
                    name: name.clone(),
                    projection: projection[..projection.len() - 1].to_vec(),
                })
            }
            Resolved::Alloc { id, projection }
                if projection.last() == Some(&RefProj::VecLen) =>
            {
                Some(Resolved::Alloc {
                    id: *id,
                    projection: projection[..projection.len() - 1].to_vec(),
                })
            }
            _ => None,
        };
        if let Some(head) = len_head {
            // Navigate to the vector value itself and update its
            // declared length where it lives: `Vec::extend_with`'s
            // fill loop increments the length once per element
            // through `SetLenOnDrop::increment_len`, so reading
            // and writing back a clone of the whole vector per
            // update (the generic whole-value route) would make
            // growth quadratic in the element count. The update
            // itself runs on the length as a detached value, then
            // [`assign_vec_len`] applies it to the navigated slot.
            macro_rules! len_slot_update {
                ($slot:expr, $desc:expr) => {{
                    let slot = $slot;
                    let len = vec_value_len(slot).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "vector length slot read-modify-write",
                            "vector value",
                            slot.kind_name(),
                        )
                    })?;
                    let mut current = Value::Uint(len);
                    let result = f(&mut current)?;
                    assign_vec_len(slot, current, $desc)?;
                    return Ok(result);
                }};
            }
            match &head {
                Resolved::Place {
                    frame_pos,
                    local,
                    projection,
                } => {
                    let frame = &mut frames[*frame_pos];
                    let func = frame.func.clone();
                    let place_desc = describe_resolved(*local, projection);
                    let slot = frame.locals.get_mut(*local).ok_or_else(|| {
                        InterpError::UnknownLocal {
                            func: func.clone(),
                            local: *local,
                        }
                    })?;
                    let slot = navigate_mut(slot, projection, &func, &place_desc)?;
                    len_slot_update!(slot, &place_desc);
                }
                Resolved::Static { name, projection } => {
                    let mut store = self.statics.borrow_mut();
                    let slot = store
                        .get_mut(name)
                        .ok_or_else(|| self.missing_static(name))?;
                    let place_desc = format!("static {name}");
                    let slot = navigate_mut(slot, projection, "<static>", &place_desc)?;
                    len_slot_update!(slot, &place_desc);
                }
                Resolved::Alloc { id, projection } => {
                    let mut store = self.allocs.borrow_mut();
                    let slot = store
                        .get_mut(id)
                        .ok_or_else(|| self.missing_alloc(*id))?;
                    let place_desc = format!("allocation #{id}");
                    let slot =
                        navigate_mut(slot, projection, "<allocation>", &place_desc)?;
                    len_slot_update!(slot, &place_desc);
                }
                Resolved::Value(..) => {
                    unreachable!("length-slot heads are never constant aggregates")
                }
            }
        }
        match target {
            Resolved::Value(..) => Err(InterpError::type_mismatch(
                "assignment",
                "mutable place",
                "constant aggregate",
            )),
            Resolved::Static { name, projection } => {
                let mut store = self.statics.borrow_mut();
                let slot = store
                    .get_mut(name)
                    .ok_or_else(|| self.missing_static(name))?;
                let place_desc = format!("static {name}");
                let slot = navigate_mut(slot, projection, "<static>", &place_desc)?;
                f(slot)
            }
            Resolved::Alloc { id, projection } => {
                let mut store = self.allocs.borrow_mut();
                let slot = store
                    .get_mut(id)
                    .ok_or_else(|| self.missing_alloc(*id))?;
                let root_ty = self.alloc_ty(*id);
                let root_ty = match root_ty {
                    Some(element_ty) if self.is_elem_block(*id) => {
                        Some(Ty::Slice(Box::new(element_ty)))
                    }
                    other => other,
                };
                grow_tuples_along(slot, projection, Some(self.program()), root_ty);
                let place_desc = format!("allocation #{id}");
                let slot = navigate_mut(slot, projection, "<allocation>", &place_desc)?;
                f(slot)
            }
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let func = frames[*frame_pos].func.clone();
                let root_ty = self
                    .program()
                    .get(&func)
                    .and_then(|body| body.local_ty(Local(*local)))
                    .cloned();
                let frame = &mut frames[*frame_pos];
                let place_desc = describe_resolved(*local, projection);
                let slot = frame.locals.get_mut(*local).ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: func.clone(),
                        local: *local,
                    }
                })?;
                let slot = navigate_mut_grow(
                    slot,
                    projection,
                    &func,
                    &place_desc,
                    false,
                    Some(self.program()),
                    root_ty,
                )?;
                f(slot)
            }
        }
    }

    /// Read element `index` of the aggregate a resolved target
    /// denotes (the element access `copy_nonoverlapping` performs).
    pub(crate) fn read_resolved_at(
        &self,
        frames: &[Frame],
        target: &Resolved,
        index: usize,
    ) -> InterpResult<Value> {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.read_resolved(
                    frames,
                    &Resolved::Place {
                        frame_pos: *frame_pos,
                        local: *local,
                        projection,
                    },
                )
            }
            Resolved::Static { name, projection } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.read_resolved(
                    frames,
                    &Resolved::Static {
                        name: name.clone(),
                        projection,
                    },
                )
            }
            Resolved::Alloc { id, projection } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.read_resolved(
                    frames,
                    &Resolved::Alloc {
                        id: *id,
                        projection,
                    },
                )
            }
            Resolved::Value(base, projection) => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                project_value(base.clone(), &projection, "<constant>", "copy_nonoverlapping source")
            }
        }
    }

    /// Write element `index` of the aggregate a resolved target
    /// denotes. Writing through a constant aggregate is the error
    /// `write_resolved` already produces for it.
    pub(crate) fn write_resolved_at(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        index: usize,
        value: Value,
    ) -> InterpResult<()> {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.write_resolved(
                    frames,
                    &Resolved::Place {
                        frame_pos: *frame_pos,
                        local: *local,
                        projection,
                    },
                    value,
                )
            }
            Resolved::Static { name, projection } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.write_resolved(
                    frames,
                    &Resolved::Static {
                        name: name.clone(),
                        projection,
                    },
                    value,
                )
            }
            Resolved::Alloc { id, projection } => {
                let mut projection = projection.clone();
                projection.push(RefProj::Index(index));
                self.write_resolved(
                    frames,
                    &Resolved::Alloc {
                        id: *id,
                        projection,
                    },
                    value,
                )
            }
            Resolved::Value(..) => self.write_resolved(frames, target, value),
        }
    }

    /// Read element `index` of the sequence a bulk-memory
    /// operation (copy / fill / compare) starting at `target`
    /// reads. Such an operation's pointer denotes the sequence's
    /// first element, which two target shapes express: an
    /// aggregate (an array place, a fat pointer's base), whose
    /// elements the plain indexed access
    /// ([`Self::read_resolved_at`]) walks, or a single element —
    /// a trailing index projection, the shape an array-to-element
    /// pointer cast produces — whose siblings continue the
    /// sequence, so the index adds to the trailing one. Which
    /// shape a target is follows from the value it denotes: an
    /// aggregate value (a nested array's row included) takes the
    /// plain access; anything else is one element of the sequence.
    pub(crate) fn read_seq_at(
        &self,
        frames: &[Frame],
        target: &Resolved,
        index: usize,
    ) -> InterpResult<Value> {
        self.read_seq_at_with_elem(frames, target, index, None)
    }

    /// [`Self::read_seq_at`] with the sequence's element type,
    /// when the caller knows it (a bulk copy's declared pointee
    /// type): a string element's stored value is the string
    /// data itself, which the untyped access cannot tell from
    /// the string's own byte storage.
    pub(crate) fn read_seq_at_with_elem(
        &self,
        frames: &[Frame],
        target: &Resolved,
        index: usize,
        elem: Option<&Ty>,
    ) -> InterpResult<Value> {
        match self.seq_element_target(frames, target, index, elem)? {
            Some(target) => self.read_resolved(frames, &target),
            None => {
                if let Some(scalar) = self.seq_scalar(frames, target)? {
                    return Ok(Value::Uint(scalar_byte(&scalar, index)? as u128));
                }
                // A byte-displaced aggregate target: the byte
                // view of the base value's storage owns the
                // read (the indexed access below cannot address
                // a byte inside an element or a field).
                if let Some(view) = self.byte_view_read(frames, target, index as i128, true)
                {
                    return view;
                }
                self.read_resolved_at(frames, target, index)
            }
        }
    }

    /// Write element `index` of the sequence a bulk-memory
    /// operation starting at `target` writes (see
    /// [`Self::read_seq_at`]).
    pub(crate) fn write_seq_at(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        index: usize,
        value: Value,
    ) -> InterpResult<()> {
        self.write_seq_at_with_elem(frames, target, index, value, None)
    }

    /// [`Self::write_seq_at`] with the sequence's element type
    /// (see [`Self::read_seq_at_with_elem`]).
    pub(crate) fn write_seq_at_with_elem(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        index: usize,
        value: Value,
        elem: Option<&Ty>,
    ) -> InterpResult<()> {
        match self.seq_element_target(frames, target, index, elem)? {
            Some(target) => self.write_resolved(frames, &target, value),
            None => {
                if let Some(scalar) = self.seq_scalar(frames, target)? {
                    let byte = match &value {
                        Value::Uint(v) if *v <= 0xFF => *v as u8,
                        Value::Int(v) if (-128..=255).contains(v) => *v as u8,
                        _ => {
                            return self.write_resolved_at(frames, target, index, value);
                        }
                    };
                    let patched = scalar_with_byte(&scalar, index, byte)?;
                    return self.write_resolved(frames, target, patched);
                }
                // A byte-displaced aggregate target: the byte
                // view of the base value's storage owns the
                // write (see `read_seq_at`).
                if let Some(view) =
                    self.byte_view_write(frames, target, index as i128, &value, true)
                {
                    return view;
                }
                self.write_resolved_at(frames, target, index, value)
            }
        }
    }
}

impl<'a> Interpreter<'a> {
    pub(crate) fn read_place(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> InterpResult<Value> {
        let target = self.resolve_place(frames, body, place)?;
        // The place's declared type, when the local's declared
        // type alone determines it: it judges the byte view
        // below and the read's uninitialized outcome (a
        // `MaybeUninit` place's uninitialized content is its
        // ordinary value; see `read_resolved_typed`).
        let place_ty = self.place_result_ty(body, place);
        // A direct deref of a byte pointer (the place's own type
        // is one byte wide) into a byte-displaced target: the
        // byte view of the base value's storage owns the read.
        if let Some(ty) = &place_ty {
            if matches!(
                ty,
                Ty::Uint(UintTy::U8) | Ty::Int(IntTy::I8) | Ty::Bool
            ) {
                if let Some(view) = self.byte_view_read(frames, &target, 0, false) {
                    return view;
                }
            }
        }
        // A read through a foreign view (a reference transmuted
        // to a different ADT over the same storage): the byte
        // view owns it when it can judge it (see
        // `Interpreter::foreign_read`).
        if let Some(read) = self.foreign_read(frames, body, place, &target) {
            return read;
        }
        // Read through the borrowing form: a projected read lends
        // the denoted value out of the frame's storage, so only
        // the projected part is cloned into the owned result,
        // where the owned read cloned the whole base aggregate
        // (every element of a vector, for a single indexed read).
        // Results and errors are identical (see
        // `read_resolved_ref`).
        Ok(self
            .read_resolved_ref_typed(frames, &target, place_ty.as_ref())?
            .into_owned())
    }

    pub(crate) fn write_place(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        place: &Place,
        value: Value,
    ) -> InterpResult<()> {
        if let Some(written) = self.write_vec_len_slot(frames, body, place, &value) {
            return written;
        }
        if let Some(written) = self.write_vec_buf_slot(frames, body, place, &value) {
            return written;
        }
        let target = self.resolve_place_mode(frames, body, place, false)?;
        let place_ty = self.place_result_ty(body, place);
        // A direct deref of a byte pointer (see `read_place`):
        // the byte view of the base value's storage owns the
        // write.
        if let Some(ty) = &place_ty {
            if matches!(
                ty,
                Ty::Uint(UintTy::U8) | Ty::Int(IntTy::I8) | Ty::Bool
            ) {
                if let Some(view) = self.byte_view_write(frames, &target, 0, &value, false)
                {
                    return view;
                }
            }
        }
        // A write through a foreign view (a reference
        // transmuted to a different ADT over the same storage):
        // the byte view owns it when it can judge it (see
        // `Interpreter::foreign_write`).
        if let Some(written) =
            self.foreign_write(frames, body, place, &target, &value)
        {
            return written;
        }
        // A write of a zero-sized value through a dereference
        // that denoted no storage (see
        // `storage_less_zst_deref`) stores nothing: the value
        // has no scalar content, and every later read of the
        // pointee fabricates the same single value.
        if matches!(target, Resolved::Value(..))
            && !value_tree_has_scalar(&value)
            && place
                .projection
                .iter()
                .any(|step| matches!(step, ProjectionElem::Deref))
        {
            return Ok(());
        }
        let root_ty = self.resolved_root_ty(body, place);
        self.write_resolved_typed(frames, &target, value, root_ty.as_ref())
    }

    /// A write to a `Vec`'s buffer bookkeeping slots through a
    /// wrapper reference (see `vec_buf_chain_reentry`): the
    /// shape a collected `RawVecInner` body takes when it
    /// stores the regrown buffer pointer and capacity
    /// (`grow_amortized`'s `(*self).0 = ...; (*self).1 = ...`,
    /// and the same stores one `RawVec` field step further
    /// out). Neither slot is storage in the element-list value.
    /// The buffer pointer is derived from the vector place
    /// itself, so storing it changes nothing: the elements do
    /// not move when the allocation is regrown — the fresh
    /// allocation the allocator produced is simply not where
    /// the model keeps them, and every later pointer read is
    /// answered from the place again. The capacity is the
    /// extent of the element storage, so storing it applies
    /// that extent to the vector value: slots past the length
    /// are the spare capacity the next append overwrites, and
    /// a shrink drops the spare slots past the new extent (the
    /// declared length itself is never cut by a capacity
    /// store). `None` when the place is not such a slot (or
    /// the stored capacity is past the model's materialisation
    /// limit, mirroring the allocation model's own limit in
    /// `Builtin::RustAlloc`).
    pub(crate) fn write_vec_buf_slot(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        place: &Place,
        value: &Value,
    ) -> Option<InterpResult<()>> {
        let local_ty = body.local_ty(place.local)?;
        let Ty::Ref(pointee) = local_ty else {
            return None;
        };
        let depth = buf_chain_depth(pointee)?;
        let proj = &place.projection;
        if proj.first() != Some(&ProjectionElem::Deref) {
            return None;
        }
        let mut steps: &[ProjectionElem] = &proj[1..];
        if depth == 1 {
            // A `RawVec` reference: the inner field step comes
            // first, and the slots are the inner's own fields.
            match steps.first() {
                Some(ProjectionElem::Field(0)) => steps = &steps[1..],
                _ => return None,
            }
        } else if depth != 2 {
            return None;
        }
        enum Slot {
            Cap,
            Ptr,
        }
        let slot = match steps {
            [ProjectionElem::Field(1)]
            | [ProjectionElem::Field(1), ProjectionElem::Field(0)] => Slot::Cap,
            [ProjectionElem::Field(0)]
            | [ProjectionElem::Field(0), ProjectionElem::Field(0)]
            | [
                ProjectionElem::Field(0),
                ProjectionElem::Field(0),
                ProjectionElem::Field(0),
            ] => Slot::Ptr,
            _ => return None,
        };
        let frame = frames.last()?;
        let ref_value = frame.locals.get(place.local.0)?.clone();
        let (base, _) = match self.vec_buf_chain_reentry(frames, &ref_value, Some(pointee)) {
            Ok(Some(pair)) => pair,
            Ok(None) => return None,
            Err(e) => return Some(Err(e)),
        };
        // A vector in its real aggregate form holds the
        // buffer bookkeeping in the `RawVec` fields
        // themselves: the store lands on the field the chain
        // steps name, and the element-list translation below
        // does not apply.
        if let Ok(held) = self.read_resolved(frames, &base) {
            if crate::builtins::vec_real_aggregate_parts(&held).is_some() {
                let steps = steps.to_vec();
                let value = value.clone();
                return Some(self.modify_resolved(frames, &base, move |slot| {
                    if aggregate_buf_slot_write(slot, &steps, &value) {
                        Ok(())
                    } else {
                        Err(InterpError::type_mismatch(
                            "Vec buffer slot write",
                            "real aggregate vector value",
                            slot.kind_name(),
                        ))
                    }
                }));
            }
        }
        match slot {
            Slot::Ptr => Some(Ok(())),
            Slot::Cap => {
                let new_cap = cap_value_count(value)?;
                if new_cap > (1 << 28) {
                    return None;
                }
                Some(self.modify_resolved(frames, &base, |slot| {
                    let (mut elems, len) =
                        match std::mem::replace(slot, Value::Uninit) {
                            Value::Enum { fields, .. } => {
                                let len = fields.len();
                                (fields, len)
                            }
                            Value::SparseVec { elems, len, .. } => (
                                elems,
                                usize::try_from(len).unwrap_or(usize::MAX),
                            ),
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "Vec capacity slot write",
                                    "vector value",
                                    other.kind_name(),
                                ));
                            }
                        };
                    // The write declares the vector's new
                    // storage extent: the sparse form accounts
                    // it as its capacity (see `Value::SparseVec`)
                    // instead of materialising the slots, and
                    // stored elements past the new extent are
                    // gone with the storage they sat in.
                    let stored = new_cap.max(len);
                    if elems.len() > stored {
                        elems.truncate(stored);
                    }
                    *slot = crate::builtins::vec_with_len(
                        elems,
                        len,
                        &Value::Uninit,
                        stored as u128,
                    );
                    Ok(())
                }))
            }
        }
    }

    /// A write to a `Vec`'s length slot, when `place` denotes
    /// one (see `vec_len_slot_base`): the shape an inlined
    /// `Vec::len` mutation takes (e.g. `pop`'s `self.len -= 1`,
    /// or a shaped store of the length in `dedup_by`). The slot
    /// is not storage in the element-list value, so the write
    /// applies the `Vec::set_len` storage change (see
    /// `crate::builtins::vec_with_len`) to the vector place
    /// itself: the declared length changes and the stored
    /// elements stay — the tail past the new length is spare
    /// capacity a later append overwrites, and a reader through
    /// the buffer pointer (the `pop` body reading the element
    /// it just un-declared) still reaches it. `None` when the
    /// place is not a length slot (or the value is not an
    /// integer length).
    pub(crate) fn write_vec_len_slot(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        place: &Place,
        value: &Value,
    ) -> Option<InterpResult<()>> {
        let base = vec_len_slot_base(body, place)?;
        let new_len = match value {
            Value::Uint(n) => usize::try_from(*n).ok()?,
            Value::Int(n) if *n >= 0 => usize::try_from(*n).ok()?,
            _ => return None,
        };
        let target = match self.resolve_place(frames, body, &base) {
            Ok(target) => target,
            Err(e) => return Some(Err(e)),
        };
        let uninit_elem = vec_elem_tag(body, &base)
            .map(|tag| crate::builtins::uninit_elem_value_for_tag(&tag))
            .unwrap_or(Value::Uninit);
        Some(self.modify_resolved(frames, &target, |slot| {
            // A vector in its real aggregate form (`[RawVec,
            // len]`, the shape a converted body builds with an
            // aggregate assignment) holds the length in the
            // slot itself: the store lands on that field, and
            // the elements (which live behind the buffer
            // pointer) are untouched. The element-list form
            // takes the `vec_with_len` translation below.
            if let Some((buf, _)) = crate::builtins::vec_real_aggregate_parts(slot) {
                *slot = Value::Enum {
                    variant: 0,
                    fields: vec![buf, Value::Uint(new_len as u128)],
                };
                return Ok(());
            }
            // The length change keeps the storage the value
            // already accounts (see `vec_with_len`): a shrink
            // (a `pop` decrement) must not collapse the
            // capacity onto the new length.
            let prev_cap = crate::builtins::vec_stored_len(slot).unwrap_or(0);
            let elems = match std::mem::replace(slot, Value::Uninit) {
                Value::Enum { fields, .. } => fields,
                Value::SparseVec { elems, .. } => elems,
                other => {
                    return Err(InterpError::type_mismatch(
                        "Vec length slot write",
                        "vector value",
                        other.kind_name(),
                    ));
                }
            };
            *slot = crate::builtins::vec_with_len(elems, new_len, &uninit_elem, prev_cap);
            Ok(())
        }))
    }
}

/// Human-readable form of a resolved place, for error messages.
pub(crate) fn describe_resolved(local: usize, projection: &[RefProj]) -> String {
    use std::fmt::Write as _;
    let mut out = format!("_{local}");
    for proj in projection {
        match proj {
            RefProj::Field(i) => {
                let _ = write!(out, ".{i}");
            }
            RefProj::UnionField { idx, .. } => {
                let _ = write!(out, ".{idx}");
            }
            RefProj::Index(i) => {
                let _ = write!(out, "[{i}]");
            }
            RefProj::Downcast(v) => {
                let _ = write!(out, " as variant {v}");
            }
            RefProj::Subslice { offset, len } => {
                let _ = write!(out, "[{offset}..{}]", offset + len);
            }
            RefProj::ByteOffset(d) => {
                let _ = write!(out, " +byte {d}");
            }
            RefProj::VecLen => {
                let _ = write!(out, ".len");
            }
        }
    }
    out
}

#[cfg(test)]
mod write_bound_tests {
    use super::*;

    /// A projected write whose index step sits at the
    /// materialisation bound must produce the navigation's
    /// interpreter error, not resize the allocation's storage
    /// first: the growth pre-pass runs ahead of the bounded
    /// navigation, and at the extreme index its size
    /// computation overflows the host's arithmetic.
    #[test]
    fn alloc_write_at_materialisation_bound_errors() {
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
        let result = interp.write_resolved(&mut frames, &target, Value::Int(7));
        assert!(result.is_err(), "write at the bound must error");
        // The storage is untouched by the rejected write.
        assert_eq!(
            interp.alloc_value(id),
            Some(Value::Enum {
                variant: 0,
                fields: vec![Value::Tuple(vec![Value::Int(1), Value::Int(2)])],
            })
        );
    }
}
