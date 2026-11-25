//! Constant materialisation.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefTarget, Value};

impl<'a> Interpreter<'a> {
    pub(crate) fn eval_operand(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        operand: &Operand,
    ) -> InterpResult<Value> {
        match operand {
            Operand::Copy(place) => {
                // Fast path, mirroring `read_local` in the decode
                // layer: a projection-less local holding a value is
                // cloned straight out of the frame's slot, skipping
                // place resolution and the resolved read. Anything
                // else — a projected place, an out-of-range index,
                // an uninitialized slot (whose zero-sized fallback
                // and exact error the general read produces) —
                // delegates to the general read unchanged.
                if place.is_direct_local() {
                    let frame = frames.last().expect("frame");
                    if let Some(value) = frame.locals.get(place.local.0) {
                        if *value != Value::Uninit {
                            return Ok(value.clone());
                        }
                    }
                }
                // A `Field` / `Downcast`-only place over a local
                // takes the decode layer's projected fast read,
                // which mirrors the general read step for step;
                // any other shape keeps the general read.
                if let Some((proj, desc)) = crate::decode::classify_local_proj(place) {
                    return self.read_local_proj(
                        frames,
                        body,
                        place.local.0 as u32,
                        &proj,
                        &desc,
                        place,
                    );
                }
                self.read_place(frames, body, place)
            }
            Operand::Move(place) => {
                // The `Copy` fast path, plus the clearing of the
                // vacated slot the general path performs below for
                // a direct local.
                if place.is_direct_local() {
                    let idx = place.local.0;
                    let frame = frames.last().expect("frame");
                    if let Some(value) = frame.locals.get(idx) {
                        if *value != Value::Uninit {
                            let value = value.clone();
                            frames.last_mut().expect("frame").locals[idx] = Value::Uninit;
                            return Ok(value);
                        }
                    }
                }
                let fast = crate::decode::classify_local_proj(place);
                let value = match &fast {
                    Some((proj, desc)) => self.read_local_proj(
                        frames,
                        body,
                        place.local.0 as u32,
                        proj,
                        desc,
                        place,
                    )?,
                    None => self.read_place(frames, body, place)?,
                };
                if place.is_direct_local() {
                    let frame = frames.last_mut().expect("frame");
                    if let Some(slot) = frame.locals.get_mut(place.local.0) {
                        *slot = Value::Uninit;
                    }
                } else if let Some((proj, desc)) = &fast {
                    self.clear_local_proj(frames, place.local.0, proj, desc)?;
                } else if let Ok(target) = self.resolve_place(frames, body, place) {
                    // A move out of a projected place leaves that
                    // part uninitialized too, so a later `Drop` of
                    // the whole value does not run the moved-out
                    // part's glue a second time (rustc's drop flags
                    // track exactly this). A heap allocation's
                    // storage clears the same way: a boxed
                    // coroutine's prologue moves its parameters out
                    // of the environment (the allocation's upvar
                    // storage), and leaving the stale copies in
                    // place would drop them a second time when the
                    // future's own glue walks the environment.
                    if matches!(
                        target,
                        Resolved::Place { .. } | Resolved::Static { .. } | Resolved::Alloc { .. }
                    ) {
                        self.write_resolved(frames, &target, Value::Uninit)?;
                    }
                }
                Ok(value)
            }
            Operand::Constant(c) => self.eval_constant(c),
        }
    }

    /// Evaluate a constant to a value. Structured constants recurse:
    /// a reference constant becomes a reference to the constant value
    /// it points at, a slice constant becomes a fat reference over
    /// its element tuple.
    pub(crate) fn eval_constant(&self, c: &Constant) -> InterpResult<Value> {
        Ok(match c {
            Constant::Unit => Value::Unit,
            Constant::Bool(b) => Value::Bool(*b),
            Constant::Int(v) => Value::Int(*v),
            Constant::Uint(v) => Value::Uint(*v),
            Constant::Float(v, _) => Value::Float(*v),
            Constant::Float128(bits) => Value::Float128(*bits),
            Constant::VTablePtr(id) => Value::Ref(RefTarget::Address(*id)),
            Constant::FatPtr { addr, meta } => Value::FatRef {
                base: Box::new(Value::Ref(RefTarget::Address(*addr))),
                len: u128::from(*meta),
            },
            Constant::DynRef {
                base,
                concrete,
                traits,
                glue,
            } => Value::DynRef {
                base: Box::new(self.eval_constant(base)?),
                concrete: concrete.clone(),
                traits: traits.clone(),
                drop_glue: glue.clone().map(Box::new),
            },
            Constant::FnDef(name) => Value::FnPtr(name.clone()),
            Constant::Closure(def, env) => Value::Closure {
                def: def.clone(),
                env: env
                    .iter()
                    .map(|c| self.eval_constant(c))
                    .collect::<InterpResult<Vec<_>>>()?,
            },
            Constant::Str(s) => Value::Str(s.clone()),
            Constant::Bytes(b) => Value::Bytes(b.clone()),
            Constant::Tuple(fields) => Value::Tuple(
                fields
                    .iter()
                    .map(|f| self.eval_constant(f))
                    .collect::<InterpResult<Vec<_>>>()?,
            ),
            Constant::Adt { variant, fields } => Value::Enum {
                variant: *variant,
                fields: fields
                    .iter()
                    .map(|f| self.eval_constant(f))
                    .collect::<InterpResult<Vec<_>>>()?,
            },
            Constant::Ref(inner) => {
                Value::Ref(RefTarget::value(self.eval_constant(inner)?))
            }
            Constant::ErasedAlloc { bytes, relocs } => {
                let relocs = relocs
                    .iter()
                    .map(|(off, v)| Ok((*off, self.eval_constant(v)?)))
                    .collect::<InterpResult<Vec<_>>>()?;
                Value::ErasedBlock {
                    bytes: bytes.clone(),
                    relocs,
                }
            }
            Constant::IntPtr(addr) => Value::Ref(RefTarget::Address(*addr)),
            Constant::NullPtr => Value::Ref(RefTarget::Null),
            Constant::Uninit => Value::Uninit,
            Constant::StaticRef {
                name,
                projection,
                len,
            } => {
                self.ensure_static_seeded(name)?;
                let base = Value::Ref(RefTarget::Static {
                    name: name.clone(),
                    projection: projection.clone(),
                });
                // A fat static reference is the static reference
                // plus its element count, in the same shape the
                // runtime unsizing cast builds: dereferencing it
                // reaches the static's storage, so writes through
                // a mutable slice static land in the one storage
                // every later read observes.
                match len {
                    Some(len) => Value::FatRef {
                        base: Box::new(base),
                        len: *len,
                    },
                    None => base,
                }
            }
            Constant::SliceVal(elems) => {
                let elems = elems
                    .iter()
                    .map(|e| self.eval_constant(e))
                    .collect::<InterpResult<Vec<_>>>()?;
                let len = elems.len() as u128;
                Value::FatRef {
                    base: Box::new(Value::Tuple(elems)),
                    len,
                }
            }
        })
    }

    /// Record that `materialised` is the typed view of the
    /// type-erased constant block `origin` (the
    /// [`Value::ErasedBlock`] a materialisation just read):
    /// address resolution denotes a pointer to the view by
    /// the block (see [`Self::canonical_target`]), so pointer
    /// equality and pointer-to-integer casts answer the same
    /// before and after a materialising cast, as they do
    /// natively -- the cast recovers the allocation's type,
    /// it does not move the allocation. The first origin
    /// recorded for a view wins: blocks whose bytes differ
    /// only where no plan reads (padding, uninitialised
    /// bytes) materialise to the same view, and address
    /// identity in this model is content-based, so either
    /// origin names the view's identity as faithfully.
    fn note_mat_origin(&self, materialised: &Value, origin: &Value) {
        self.mat_origins
            .borrow_mut()
            .entry(format!("{materialised:?}"))
            .or_insert_with(|| origin.clone());
    }

    /// Evaluate a materialising cast (see [`Rvalue::Materialize`]):
    /// an operand denoting a type-erased constant allocation
    /// materialises into the cast target's type with the
    /// recorded plan; every other operand keeps the behaviour
    /// the cast would have had without the plan (the identity,
    /// or the base extraction of [`Rvalue::DataPtr`] for a fat
    /// operand).
    pub(crate) fn eval_materialize(&self, value: Value, plan: usize) -> InterpResult<Value> {
        let value = match value {
            Value::FatRef { base, .. } => (*base).clone(),
            other => other,
        };
        let Value::Ref(target) = &value else {
            return Ok(value);
        };
        let RefTarget::Value(pointee, projection) = target else {
            return Ok(value);
        };
        // The block is the pointee itself only when the
        // reference denotes the value's base. A projection
        // denotes a position inside an aggregate (an element,
        // possibly one past the end): its denotation is the
        // pointer's identity, which the cast preserves
        // untouched, exactly as for every other non-erased
        // operand.
        let Value::ErasedBlock { bytes, relocs } = pointee.as_ref() else {
            return Ok(value);
        };
        if !projection_is_base(projection) {
            return Ok(value);
        }
        let Some(plan) = self.program.mat_plans.get(plan) else {
            return Err(InterpError::type_mismatch(
                "type-erased constant materialisation",
                "a recorded materialisation plan",
                "an out-of-range plan index",
            ));
        };
        let materialised = self.materialize_value(bytes, relocs, plan, 0)?;
        self.note_mat_origin(&materialised, pointee.as_ref());
        Ok(Value::Ref(RefTarget::value(materialised)))
    }

    /// Read the value of the type `plan` describes out of a
    /// type-erased constant block (its bytes plus its evaluated
    /// provenance relocations) at byte `offset`, producing the
    /// value the driver's typed constant decode would have
    /// produced for the same allocation (see [`MatPlan`]).
    fn materialize_value(
        &self,
        bytes: &[Option<u8>],
        relocs: &[(u32, Value)],
        plan: &MatPlan,
        offset: u64,
    ) -> InterpResult<Value> {
        match plan {
            MatPlan::Unsupported => Err(InterpError::type_mismatch(
                "type-erased constant materialisation",
                "a type with a materialisation plan",
                "an unsupported type",
            )),
            MatPlan::Unit => Ok(Value::Unit),
            MatPlan::Bool => match Self::mat_byte(bytes, offset)? {
                None => Ok(Value::Uninit),
                Some(0) => Ok(Value::Bool(false)),
                Some(1) => Ok(Value::Bool(true)),
                Some(_) => Err(InterpError::type_mismatch(
                    "type-erased constant materialisation",
                    "a bool byte of 0 or 1",
                    "another byte",
                )),
            },
            MatPlan::Char => match Self::mat_uint(bytes, offset, 4)? {
                None => Ok(Value::Uninit),
                Some(v) => {
                    if char::from_u32(v as u32).is_some() {
                        Ok(Value::Uint(v))
                    } else {
                        Err(InterpError::type_mismatch(
                            "type-erased constant materialisation",
                            "a valid char code point",
                            "an invalid code point",
                        ))
                    }
                }
            },
            MatPlan::Int { size, signed } => match Self::mat_uint(bytes, offset, *size)? {
                None => Ok(Value::Uninit),
                Some(raw) => {
                    if *signed {
                        Ok(Value::Int(Self::mat_sign_extend(raw, *size)))
                    } else {
                        Ok(Value::Uint(raw))
                    }
                }
            },
            MatPlan::Float { size } => match Self::mat_uint(bytes, offset, *size)? {
                None => Ok(Value::Uninit),
                Some(bits) => match size {
                    2 => Ok(Value::Float(crate::mir_convert::f16_to_f64(bits as u16))),
                    4 => Ok(Value::Float(crate::f128::f32_bits_to_f64_exact(bits as u32))),
                    8 => Ok(Value::Float(f64::from_bits(bits as u64))),
                    16 => Ok(Value::Float128(bits)),
                    _ => Err(InterpError::type_mismatch(
                        "type-erased constant materialisation",
                        "a float width of 2, 4, 8 or 16 bytes",
                        "another width",
                    )),
                },
            },
            MatPlan::Str => {
                let raw = Self::mat_trailing_bytes(bytes, offset)?;
                match String::from_utf8(raw) {
                    Ok(s) => Ok(Value::Str(s)),
                    Err(_) => Err(InterpError::type_mismatch(
                        "type-erased constant materialisation",
                        "valid UTF-8 string bytes",
                        "invalid UTF-8",
                    )),
                }
            }
            MatPlan::Slice(elem) => self.materialize_trailing_slice(bytes, relocs, elem, offset),
            MatPlan::Ref(inner) => self.materialize_pointer(bytes, relocs, offset, inner),
            MatPlan::FnPtr => match Self::mat_reloc(relocs, offset) {
                Some(target) => Ok(target.clone()),
                None => Self::mat_address_word(bytes, offset),
            },
            MatPlan::Tuple { fields, .. } => {
                let values = self.materialize_fields(bytes, relocs, fields, offset)?;
                Ok(Value::Tuple(values))
            }
            MatPlan::Struct { fields, .. } => {
                let values = self.materialize_fields(bytes, relocs, fields, offset)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: values,
                })
            }
            MatPlan::Array {
                elem,
                stride,
                count,
            } => {
                let mut values = Vec::new();
                for i in 0..*count {
                    values.push(self.materialize_value(
                        bytes,
                        relocs,
                        elem,
                        offset + i * stride,
                    )?);
                }
                Ok(Value::Tuple(values))
            }
            MatPlan::Enum {
                tag, variants, ..
            } => {
                let (variant, fields) =
                    self.materialize_enum_tag(bytes, relocs, tag, variants, offset)?;
                let values = self.materialize_fields(bytes, relocs, fields, offset)?;
                Ok(Value::Enum {
                    variant,
                    fields: values,
                })
            }
            MatPlan::Union { fields, .. } => {
                self.materialize_union(bytes, relocs, fields, offset)
            }
        }
    }

    /// Materialise the fields of one aggregate plan at their
    /// recorded offsets (see [`Self::materialize_value`]).
    fn materialize_fields(
        &self,
        bytes: &[Option<u8>],
        relocs: &[(u32, Value)],
        fields: &[MatField],
        offset: u64,
    ) -> InterpResult<Vec<Value>> {
        fields
            .iter()
            .map(|f| self.materialize_value(bytes, relocs, &f.plan, offset + f.offset))
            .collect()
    }

    /// The raw value of a niche tag scalar. A niche whose bytes
    /// are covered by a relocation — the pointer niche of a
    /// variant carrying a reference — has no meaningful byte
    /// value in the block (the address lives in the relocation
    /// alone, over zeroed bytes), so the raw value comes from
    /// the relocation's target: null reads as 0, an address as
    /// itself, and any other live pointer or function value as
    /// a nonzero word, which is all a niche range check can ask
    /// of it.
    fn mat_tag_raw(
        bytes: &[Option<u8>],
        relocs: &[(u32, Value)],
        offset: u64,
        size: u8,
    ) -> InterpResult<u128> {
        match Self::mat_reloc(relocs, offset) {
            Some(Value::Ref(RefTarget::Null)) => Ok(0),
            Some(Value::Ref(RefTarget::Address(word))) => Ok(u128::from(*word)),
            Some(Value::Uint(word)) => Ok(u128::from(*word)),
            Some(Value::Int(value)) => Ok(*value as u128),
            Some(_) => Ok(1),
            None => Self::mat_uint(bytes, offset, size)?.ok_or_else(|| {
                InterpError::type_mismatch(
                    "type-erased constant materialisation",
                    "an initialised enum tag",
                    "uninitialised tag bytes",
                )
            }),
        }
    }

    /// Resolve an enum plan's tag to the variant it names and
    /// that variant's field plans (see [`Self::materialize_value`]).
    fn materialize_enum_tag<'p>(
        &self,
        bytes: &[Option<u8>],
        relocs: &[(u32, Value)],
        tag: &MatTag,
        variants: &'p [Vec<MatField>],
        offset: u64,
    ) -> InterpResult<(u32, &'p [MatField])> {
        fn variant_fields<'p>(
            variants: &'p [Vec<MatField>],
            idx: u32,
        ) -> InterpResult<&'p [MatField]> {
            variants
                .get(idx as usize)
                .map(Vec::as_slice)
                .ok_or_else(|| {
                    InterpError::type_mismatch(
                        "type-erased constant materialisation",
                        "a variant the plan describes",
                        "a variant outside the plan",
                    )
                })
        }
        match tag {
            MatTag::Single(idx) => Ok((*idx, variant_fields(variants, *idx)?)),
            MatTag::Direct {
                offset: toff,
                size,
                signed,
                discriminants,
            } => {
                let raw = Self::mat_uint(bytes, offset + toff, *size)?.ok_or_else(|| {
                    InterpError::type_mismatch(
                        "type-erased constant materialisation",
                        "an initialised enum tag",
                        "uninitialised tag bytes",
                    )
                })?;
                let value = if *signed {
                    Self::mat_sign_extend(raw, *size)
                } else {
                    raw as i128
                };
                let idx = discriminants
                    .iter()
                    .position(|d| *d == value)
                    .ok_or_else(|| {
                        InterpError::type_mismatch(
                            "type-erased constant materialisation",
                            "a tag matching a variant discriminant",
                            "an unmatched tag",
                        )
                    })?;
                Ok((idx as u32, variant_fields(variants, idx as u32)?))
            }
            MatTag::Niche {
                offset: toff,
                size,
                untagged,
                niche_start,
                start,
                end,
            } => {
                let raw = Self::mat_tag_raw(bytes, relocs, offset + toff, *size)?;
                let span = u128::from(end - start);
                // The niche range may wrap past the tag
                // scalar's maximum, so the distance from the
                // range's start is taken at the scalar's width.
                let offset = raw.wrapping_sub(*niche_start) & bit_mask(u32::from(*size) * 8);
                if offset <= span {
                    let variant = start + offset as u32;
                    Ok((variant, variant_fields(variants, variant)?))
                } else {
                    Ok((*untagged, variant_fields(variants, *untagged)?))
                }
            }
        }
    }

    /// Materialise a pointer slot (see [`MatPlan::Ref`]): the
    /// slot at `offset` in the block, read as a pointer to a
    /// value of the pointee plan's type.
    fn materialize_pointer(
        &self,
        bytes: &[Option<u8>],
        relocs: &[(u32, Value)],
        offset: u64,
        inner: &MatPlan,
    ) -> InterpResult<Value> {
        // A fat pointee (`str` / slice): the slot is two words -
        // the pointer, carrying the slot's relocation, and the
        // length - and the value is the fat reference the typed
        // decode of the same slot produces.
        if matches!(inner, MatPlan::Str | MatPlan::Slice(_)) {
            let len = match Self::mat_uint(bytes, offset + 8, 8)? {
                Some(v) => v,
                None => return Ok(Value::Uninit),
            };
            let Some(target) = Self::mat_reloc(relocs, offset) else {
                // No provenance: a dangling fat pointer constant,
                // exactly the typed decode's `FatPtr` form.
                let addr = match Self::mat_uint(bytes, offset, 8)? {
                    Some(v) => v as u64,
                    None => return Ok(Value::Uninit),
                };
                return Ok(Value::FatRef {
                    base: Box::new(Value::Ref(RefTarget::Address(addr))),
                    len,
                });
            };
            let (bbytes, brelocs) = Self::mat_target_block(target)?;
            return match inner {
                MatPlan::Str => {
                    let raw = Self::mat_block_bytes(bbytes, 0, len)?;
                    match String::from_utf8(raw) {
                        Ok(s) => Ok(Value::Str(s)),
                        Err(_) => Err(InterpError::type_mismatch(
                            "type-erased constant materialisation",
                            "valid UTF-8 string bytes",
                            "invalid UTF-8",
                        )),
                    }
                }
                MatPlan::Slice(elem) => {
                    let elems =
                        self.materialize_slice_elems(bbytes, brelocs, elem, 0, len)?;
                    Ok(Value::FatRef {
                        base: Box::new(Value::Tuple(elems)),
                        len,
                    })
                }
                _ => unreachable!(),
            };
        }
        // A thin pointee.
        match Self::mat_reloc(relocs, offset) {
            // A relocation at the slot names the target block:
            // materialise it with the pointee plan, as the
            // typed decode of the same slot produces the
            // pointee's snapshot.
            Some(Value::Ref(RefTarget::Value(v, projection)))
                if matches!(v.as_ref(), Value::ErasedBlock { .. })
                    && projection_is_base(projection) =>
            {
                let Value::ErasedBlock {
                    bytes: bbytes,
                    relocs: brelocs,
                } = v.as_ref()
                else {
                    unreachable!()
                };
                let m = self.materialize_value(bbytes, brelocs, inner, 0)?;
                self.note_mat_origin(&m, v.as_ref());
                Ok(Value::Ref(RefTarget::value(m)))
            }
            // A function value, an address-only pointer, a null
            // pointer, a reference to a typed snapshot: the
            // relocation's value is the slot's value, exactly as
            // the typed decode yields it.
            Some(target) => Ok(target.clone()),
            None => Self::mat_address_word(bytes, offset),
        }
    }

    /// Materialise a slice in trailing position (see
    /// [`MatPlan::Slice`]): its elements run to the end of the
    /// block, as the typed decode of a trailing slice reads
    /// them.
    fn materialize_trailing_slice(
        &self,
        bytes: &[Option<u8>],
        relocs: &[(u32, Value)],
        elem: &MatPlan,
        offset: u64,
    ) -> InterpResult<Value> {
        if matches!(elem, MatPlan::Int { size: 1, signed: false }) {
            let raw = Self::mat_trailing_bytes(bytes, offset)?;
            return Ok(Value::Bytes(raw));
        }
        let stride = elem.size().ok_or_else(|| {
            InterpError::type_mismatch(
                "type-erased constant materialisation",
                "a sized slice element type",
                "an unsized element type",
            )
        })?;
        if stride == 0 {
            return Err(InterpError::type_mismatch(
                "type-erased constant materialisation",
                "a nonzero element size",
                "a zero-sized element type",
            ));
        }
        let remaining = (bytes.len() as u64).saturating_sub(offset);
        if remaining % stride != 0 {
            return Err(InterpError::type_mismatch(
                "type-erased constant materialisation",
                "a block holding whole elements",
                "a block with a partial element",
            ));
        }
        let count = u128::from(remaining / stride);
        let elems = self.materialize_slice_elems(bytes, relocs, elem, offset, count)?;
        Ok(Value::FatRef {
            base: Box::new(Value::Tuple(elems)),
            len: count,
        })
    }

    /// Materialise `count` slice elements of the element plan
    /// from a block at `base`, each at its stride offset.
    fn materialize_slice_elems(
        &self,
        bytes: &[Option<u8>],
        relocs: &[(u32, Value)],
        elem: &MatPlan,
        base: u64,
        count: u128,
    ) -> InterpResult<Vec<Value>> {
        let stride = elem.size().ok_or_else(|| {
            InterpError::type_mismatch(
                "type-erased constant materialisation",
                "a sized slice element type",
                "an unsized element type",
            )
        })?;
        if stride == 0 && count > 0 {
            return Err(InterpError::type_mismatch(
                "type-erased constant materialisation",
                "a nonzero element size",
                "a zero-sized element type",
            ));
        }
        let mut out = Vec::new();
        for i in 0..count {
            out.push(self.materialize_value(
                bytes,
                relocs,
                elem,
                base + i as u64 * stride,
            )?);
        }
        Ok(out)
    }

    /// Materialise a union plan (see [`MatPlan::Union`]),
    /// mirroring the driver's union decode.
    fn materialize_union(
        &self,
        bytes: &[Option<u8>],
        relocs: &[(u32, Value)],
        fields: &[MatPlan],
        offset: u64,
    ) -> InterpResult<Value> {
        // A provenance relocation at the union's offset can
        // only belong to a pointer-shaped field stored there;
        // when exactly one field is such a pointer it is the
        // live field, zero-sized fields are always read, and
        // the rest are uninitialised. Without a relocation (or
        // without a unique pointer field) every field is read
        // from the same bytes.
        let live = if Self::mat_reloc(relocs, offset).is_some() {
            let ptr_fields: Vec<usize> = fields
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    matches!(p, MatPlan::Ref(_) | MatPlan::FnPtr) && p.size() == Some(8)
                })
                .map(|(i, _)| i)
                .collect();
            if ptr_fields.len() == 1 {
                Some(ptr_fields[0])
            } else {
                None
            }
        } else {
            None
        };
        let mut out = Vec::with_capacity(fields.len());
        for (i, plan) in fields.iter().enumerate() {
            if let Some(live) = live {
                if i != live && plan.size() != Some(0) {
                    out.push(Value::Uninit);
                    continue;
                }
            }
            out.push(self.materialize_value(bytes, relocs, plan, offset)?);
        }
        Ok(Value::Enum {
            variant: 0,
            fields: out,
        })
    }

    /// One byte of a block being materialised; an out-of-range
    /// offset is an explicit error (the block does not cover
    /// the value being read).
    fn mat_byte(bytes: &[Option<u8>], offset: u64) -> InterpResult<Option<u8>> {
        match bytes.get(offset as usize) {
            Some(b) => Ok(*b),
            None => Err(InterpError::type_mismatch(
                "type-erased constant materialisation",
                "a block covering the value's bytes",
                "a shorter block",
            )),
        }
    }

    /// The `size` bytes at `offset` as a little-endian unsigned
    /// integer; `None` when any of them is uninitialised.
    fn mat_uint(bytes: &[Option<u8>], offset: u64, size: u8) -> InterpResult<Option<u128>> {
        let mut out: u128 = 0;
        for i in 0..u64::from(size) {
            match Self::mat_byte(bytes, offset + i)? {
                Some(b) => out |= u128::from(b) << (8 * i),
                None => return Ok(None),
            }
        }
        Ok(Some(out))
    }

    /// Sign-extend a `size`-byte raw pattern to an `i128`.
    fn mat_sign_extend(raw: u128, size: u8) -> i128 {
        let bits = u32::from(size) * 8;
        if bits == 0 {
            0
        } else if bits >= 128 {
            raw as i128
        } else {
            ((raw << (128 - bits)) as i128) >> (128 - bits)
        }
    }

    /// The provenance relocation recorded at `offset`, if any.
    fn mat_reloc<'v>(relocs: &'v [(u32, Value)], offset: u64) -> Option<&'v Value> {
        relocs
            .iter()
            .find(|(o, _)| u64::from(*o) == offset)
            .map(|(_, v)| v)
    }

    /// The erased block a relocation value denotes.
    fn mat_target_block(target: &Value) -> InterpResult<(&[Option<u8>], &[(u32, Value)])> {
        match target {
            Value::Ref(RefTarget::Value(v, projection)) if projection.is_empty() => {
                match v.as_ref() {
                    Value::ErasedBlock { bytes, relocs } => Ok((bytes, relocs)),
                    other => Err(InterpError::type_mismatch(
                        "type-erased constant materialisation",
                        "a reference to an erased constant block",
                        other.kind_name(),
                    )),
                }
            }
            other => Err(InterpError::type_mismatch(
                "type-erased constant materialisation",
                "a reference to an erased constant block",
                other.kind_name(),
            )),
        }
    }

    /// A pointer slot with no provenance relocation: the word
    /// itself is the pointer (null, or a dangling address),
    /// exactly as the typed decode reads it.
    fn mat_address_word(bytes: &[Option<u8>], offset: u64) -> InterpResult<Value> {
        match Self::mat_uint(bytes, offset, 8)? {
            None => Ok(Value::Uninit),
            Some(0) => Ok(Value::Ref(RefTarget::Null)),
            Some(addr) => Ok(Value::Ref(RefTarget::Address(addr as u64))),
        }
    }

    /// The block's bytes from `offset` to its end, all of which
    /// must be initialised.
    fn mat_trailing_bytes(bytes: &[Option<u8>], offset: u64) -> InterpResult<Vec<u8>> {
        let len = (bytes.len() as u64).saturating_sub(offset);
        Self::mat_block_bytes(bytes, offset, u128::from(len))
    }

    /// `len` bytes of a block at `offset`, all of which must be
    /// initialised.
    fn mat_block_bytes(bytes: &[Option<u8>], offset: u64, len: u128) -> InterpResult<Vec<u8>> {
        let mut out = Vec::new();
        for i in 0..len {
            match Self::mat_byte(bytes, offset + i as u64)? {
                Some(b) => out.push(b),
                None => {
                    return Err(InterpError::type_mismatch(
                        "type-erased constant materialisation",
                        "initialised bytes",
                        "uninitialised bytes",
                    ))
                }
            }
        }
        Ok(out)
    }
}
