//! Casts and unsizing.

use super::*;
use crate::error::{InterpError, InterpResult};
use crate::f128;
use crate::value::{RefProj, RefTarget, Value};

impl<'a> Interpreter<'a> {
    /// Apply an unsizing decided at conversion time
    /// ([`UnsizeMeta`]) to a value: the shared machinery of the
    /// `Rvalue::Unsize` coercion cast and of a `From` / `Into`
    /// conversion into a `Box<dyn Trait>`, whose result goes
    /// through the same unsizing (see
    /// `TerminatorKind::CallBuiltin::result_unsize`).
    pub(crate) fn eval_unsize(
        &self,
        frames: &[Frame],
        v: Value,
        meta: &UnsizeMeta,
    ) -> InterpResult<Value> {
        match meta {
            UnsizeMeta::Slice { array_len } => {
                self.unsize_to_slice(frames, v, *array_len)
            }
            UnsizeMeta::Dyn {
                concrete,
                traits,
                glue,
            } => match v {
                // A supertrait upcast: the data pointer, the
                // concrete type and its drop glue carry over;
                // only the bounds the pointer answers to
                // change.
                Value::DynRef {
                    base,
                    concrete: existing,
                    drop_glue,
                    ..
                } => Ok(Value::DynRef {
                    base,
                    concrete: existing,
                    traits: traits.clone(),
                    drop_glue,
                }),
                Value::Ref(_) | Value::FatRef { .. } | Value::Enum { .. } => {
                    match concrete {
                        Some(concrete) => Ok(Value::DynRef {
                            base: Box::new(v),
                            concrete: concrete.clone(),
                            traits: traits.clone(),
                            drop_glue: glue.clone().map(Box::new),
                        }),
                        // A `dyn`-to-`dyn` coercion whose operand
                        // is a reference to a place holding a
                        // trait-object pointer (a `&mut
                        // Box<dyn Trait>` argument where a `&mut
                        // dyn Trait` is expected): no concrete
                        // type is static here - it rides on the
                        // stored pointer. Read the stored pointer
                        // and reborrow it: the data pointer and
                        // the concrete type carry over, the
                        // bounds become the cast's, and the
                        // reborrow owns nothing, so no drop glue
                        // attaches.
                        None => {
                            let resolved = self.deref_value(frames, &v)?;
                            let stored = self.read_resolved(frames, &resolved)?;
                            match stored {
                                Value::DynRef {
                                    base,
                                    concrete: stored_concrete,
                                    ..
                                } => Ok(Value::DynRef {
                                    base,
                                    concrete: stored_concrete,
                                    traits: traits.clone(),
                                    drop_glue: None,
                                }),
                                _ => Err(InterpError::UnsupportedRvalue {
                                    kind: "Unsize to dyn (no concrete type recorded)"
                                        .to_string(),
                                }),
                            }
                        }
                    }
                }
                other => Err(InterpError::type_mismatch(
                    "unsizing cast to dyn",
                    "reference, fat pointer or box value",
                    other.kind_name(),
                )),
            },
        }
    }

    /// Array-to-slice unsizing of a pointer-shaped value: a
    /// reference to an array becomes a fat reference over it, and
    /// a pointer-carrying ADT value (a `Box`, the `Unique` /
    /// `NonNull` aggregate chain a MIR-built box is made of, an
    /// `Rc` / `Arc` / `Weak` handle, or a user wrapper around
    /// one) is rewrapped around the unsized pointee, so
    /// `Box<[T; N]>` becomes the same shape a directly built
    /// `Box<[T]>` has.
    fn unsize_to_slice(
        &self,
        frames: &[Frame],
        v: Value,
        array_len: Option<u64>,
    ) -> InterpResult<Value> {
        match v {
            Value::Ref(_) => {
                // With the source array's length recorded at
                // conversion, the fat reference needs nothing
                // from the pointee's stored value, which may be
                // unwritten or only partly materialised (see
                // `UnsizeMeta::Slice`).
                if let Some(n) = array_len {
                    return Ok(Value::FatRef {
                        base: Box::new(v),
                        len: n as u128,
                    });
                }
                let target = self.deref_value(frames, &v)?;
                let pointee = self.read_resolved(frames, &target)?;
                let len = unsized_tail_len(&pointee).ok_or_else(|| {
                    InterpError::type_mismatch(
                        "unsizing cast",
                        "array pointee",
                        pointee.kind_name(),
                    )
                })?;
                Ok(Value::FatRef {
                    base: Box::new(v),
                    len,
                })
            }
            // Unsizing a fat pointer into a slice pointer again
            // (a slice-typed field reached through the ADT search
            // below) leaves it as it stands.
            Value::FatRef { .. } => Ok(v),
            Value::Bytes(b) => {
                let len = b.len() as u128;
                Ok(Value::FatRef {
                    base: Box::new(Value::Bytes(b)),
                    len,
                })
            }
            Value::Tuple(items) => {
                let len = items.len() as u128;
                Ok(Value::FatRef {
                    base: Box::new(Value::Tuple(items)),
                    len,
                })
            }
            // An empty array aggregate is the unit value in this
            // model: unsizing it yields an empty slice.
            Value::Unit => Ok(Value::FatRef {
                base: Box::new(Value::Tuple(Vec::new())),
                len: 0,
            }),
            // A reference-counted handle is itself the pointer:
            // its pointee lives in the allocation store, whose
            // stored aggregate carries the length every consumer
            // derives (the `Deref` builtin, the drop glue and the
            // count adjustments all navigate the store through
            // the handle's allocation id), so unsizing the handle
            // into a slice handle changes nothing about the
            // value. Wrapping the handle's inner pointer into a
            // fat reference instead would leave the handle naming
            // a view no consumer recognises as a handle.
            Value::Enum { .. } if rc_handle_id(&v).is_some() => Ok(v),
            Value::Enum { variant, mut fields } if fields.len() == 1 => {
                let pointee = fields.pop().expect("one field");
                // A built box's field is its allocation pointer:
                // the array already lives in the allocation store
                // under that pointer, and unsizing is only a view
                // of it — every consumer derives the slice length
                // from the pointee value in the store. The box
                // keeps its shape (rewrapping the pointer into a
                // fat reference here would leave the box holding
                // a snapshot view instead of the pointer, and a
                // pointer extracted from it later would name this
                // place rather than the allocation).
                if matches!(pointee, Value::Ref(RefTarget::Alloc { .. })) {
                    return Ok(Value::Enum {
                        variant,
                        fields: vec![pointee],
                    });
                }
                let fat = self.unsize_to_slice(frames, pointee, None)?;
                Ok(Value::Enum {
                    variant,
                    fields: vec![fat],
                })
            }
            Value::Enum { variant, fields } => {
                // A multi-field ADT: the coercion propagates
                // through the one pointer-shaped field (rustc's
                // `CoerceUnsized` changes exactly one field's
                // type); the remaining fields — phantom markers,
                // counters, plain data — pass through untouched.
                // Candidates are judged from the values alone: a
                // candidate whose unsizing rests on the pointee's
                // own array tail outranks one resting on the
                // one-element view of a scalar, and among equals
                // a changed value outranks an unchanged one, with
                // the later field winning (the position a DST
                // tail occupies). When no candidate changes its
                // value — every one is an allocation-backed
                // handle, whose unsizing is the identity in this
                // model — the first candidate stands and the ADT
                // keeps its shape.
                let mut best: Option<(usize, Value)> = None;
                let mut best_strong = false;
                for (i, field) in fields.iter().enumerate() {
                    let Some(Ok(unsized_field)) =
                        self.unsize_pointer_shape(frames, field.clone())
                    else {
                        continue;
                    };
                    let strong = match field {
                        Value::Ref(_) => {
                            let target = self.deref_value(frames, field)?;
                            let pointee = self.read_resolved(frames, &target)?;
                            unsized_tail_is_array(&pointee)
                        }
                        _ => true,
                    };
                    let changed = unsized_field != *field;
                    let replace = match &best {
                        None => true,
                        Some(_) if strong && !best_strong => true,
                        Some(_) if strong == best_strong && changed => true,
                        _ => false,
                    };
                    if replace {
                        best = Some((i, unsized_field));
                        best_strong = strong;
                    }
                }
                let Some((idx, replacement)) = best else {
                    return Err(InterpError::type_mismatch(
                        "unsizing cast",
                        "reference to an array",
                        "ADT value",
                    ));
                };
                let mut fields = fields;
                fields[idx] = replacement;
                Ok(Value::Enum {
                    variant,
                    fields,
                })
            }
            other => Err(InterpError::type_mismatch(
                "unsizing cast",
                "reference to an array",
                other.kind_name(),
            )),
        }
    }

    /// Unsizing applied to a value in pointer position inside an
    /// ADT (the multi-field arm of [`Self::unsize_to_slice`]):
    /// references and fat pointers directly, wrapper chains and
    /// nested ADTs through the general unsizing, and
    /// reference-counted handles as the identity. Returns `None`
    /// when the value is not a pointer shape an unsizing can
    /// apply to — plain data, callables, trait-object pointers
    /// (their unsizing is the `Dyn` metadata's business) — so
    /// the caller can look at the ADT's other fields. A pointer
    /// shape whose own unsizing fails is likewise not a
    /// candidate: the field the coercion targets is the one
    /// whose unsizing succeeds.
    fn unsize_pointer_shape(
        &self,
        frames: &[Frame],
        v: Value,
    ) -> Option<InterpResult<Value>> {
        match v {
            Value::Ref(_) | Value::FatRef { .. } => {
                Some(self.unsize_to_slice(frames, v, None))
            }
            Value::Enum { .. } => {
                if rc_handle_id(&v).is_some() {
                    return Some(Ok(v));
                }
                match self.unsize_to_slice(frames, v, None) {
                    Ok(u) => Some(Ok(u)),
                    Err(_) => None,
                }
            }
            _ => None,
        }
    }

    /// Leave the value at a `Field` / `Downcast`-only projected
    /// place over a local of the executing frame uninitialized:
    /// the clearing a `Move` out of that place performs. The
    /// resolved target of such a place is the executing frame's
    /// local under the same projection (the classification's
    /// steps are purely syntactic), so the write goes straight
    /// through `navigate_mut` — the same write
    /// `write_resolved`'s projected `Place` arm performs, minus
    /// the resolve round trip.
    pub(crate) fn clear_local_proj(
        &self,
        frames: &mut Vec<Frame>,
        local: usize,
        proj: &[RefProj],
        desc: &str,
    ) -> InterpResult<()> {
        let func = frames.last().expect("frame").func.clone();
        let root_ty = self
            .program()
            .get(&func)
            .and_then(|body| body.local_ty(Local(local)))
            .cloned();
        let frame = frames.last_mut().expect("frame");
        let slot = frame.locals.get_mut(local).ok_or_else(|| InterpError::UnknownLocal {
            func: func.clone(),
            local,
        })?;
        let slot = navigate_mut_grow(
            slot,
            proj,
            &func,
            desc,
            false,
            Some(self.program()),
            root_ty,
        )?;
        *slot = Value::Uninit;
        Ok(())
    }
}

/// One step of the declared-type walk over a projection: the
/// type of the value denoted after applying `elem` to a value of
/// type `ty`, when the local's declared type alone determines
/// it. This is the step set `place_result_ty` judges by.
pub(crate) fn projection_step_ty(ty: &Ty, elem: &ProjectionElem) -> Option<Ty> {
    match (elem, ty) {
        (ProjectionElem::Deref, Ty::Ref(inner)) => Some((**inner).clone()),
        (ProjectionElem::Field(i), Ty::Tuple(ts)) => ts.get(*i).cloned(),
        (ProjectionElem::Index(_), Ty::Array(elem, _)) => Some((**elem).clone()),
        (ProjectionElem::Index(_), Ty::Slice(elem)) => Some((**elem).clone()),
        (ProjectionElem::ConstantIndex { .. }, Ty::Array(elem, _)) => {
            Some((**elem).clone())
        }
        (ProjectionElem::ConstantIndex { .. }, Ty::Slice(elem)) => {
            Some((**elem).clone())
        }
        (ProjectionElem::Downcast(_), t) => Some(t.clone()),
        (ProjectionElem::Subslice { .. }, Ty::Array(elem, _)) => {
            Some(Ty::Array(elem.clone(), 0))
        }
        (ProjectionElem::Subslice { .. }, Ty::Slice(elem)) => {
            Some(Ty::Slice(elem.clone()))
        }
        _ => None,
    }
}

/// The declared type of the value a cast's source place denotes,
/// walked over the place's own projection: a tuple field, an
/// array element and a dereferenced pointer (a reference and a
/// raw pointer share the `Ty::Ref` representation) all name
/// their element / pointee type, so a cast from such a place is
/// a genuine conversion at that width, not a transport of the
/// base local's stored form. `None` when a step's type is not
/// determined by the local's declared type alone (a field of an
/// ADT, a deref of an ADT-typed local such as a `Box`).
pub(crate) fn cast_place_ty(body: &Body, place: &Place) -> Option<Ty> {
    let mut ty = body.local_ty(place.local)?.clone();
    for elem in &place.projection {
        ty = projection_step_ty(&ty, elem)?;
    }
    Some(ty)
}

/// The source type a constant records for itself, when it
/// records one: a float constant keeps its own width beside the
/// widened stored value (see `Constant::Float`). Other constants
/// record no width in this IR.
pub(crate) fn constant_ty(c: &Constant) -> Option<Ty> {
    match c {
        Constant::Float(_, ty) => Some(Ty::Float(*ty)),
        Constant::Float128(_) => Some(Ty::Float(FloatTy::F128)),
        _ => None,
    }
}

/// The declared type of a cast's source operand: a place source
/// resolves through the place's declared type (see
/// [`cast_place_ty`]); a constant source answers with the type
/// the constant itself records (see [`constant_ty`]). Without a
/// source type, `eval_cast`'s float arms keep the stored form,
/// which is the exact widening of the source value.
pub(crate) fn cast_src_ty(body: &Body, operand: &Operand) -> Option<Ty> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => cast_place_ty(body, place),
        Operand::Constant(c) => constant_ty(c),
        _ => None,
    }
}

pub(crate) fn eval_cast(v: &Value, src_ty: Option<&Ty>, ty: &Ty) -> InterpResult<Value> {
    match ty {
        Ty::Int(it) => {
            // A float source saturates; it does not wrap like an
            // integer source.
            if let Value::Float(a) = v {
                return Ok(Value::Int(float_to_int(*a, it.bits())));
            }
            if let Value::Float128(a) = v {
                return Ok(Value::Int(f128::to_int(*a, it.bits())));
            }
            // A pointer source (a pointer-to-integer transmute)
            // yields the pointer's synthetic address.
            if matches!(v, Value::Ref(_) | Value::FatRef { .. } | Value::FnPtr(_)) {
                return Ok(Value::Int(wrap_signed(
                    synthetic_addr(v) as i128,
                    it.bits(),
                )));
            }
            let raw = match v {
                Value::Int(a) => *a,
                Value::Uint(a) => *a as i128,
                Value::Bool(b) => *b as i128,
                other => {
                    return Err(InterpError::type_mismatch(
                        "cast",
                        "numeric source",
                        other.kind_name(),
                    ))
                }
            };
            Ok(Value::Int(wrap_signed(raw, it.bits())))
        }
        Ty::Uint(ut) => {
            if let Value::Float(a) = v {
                return Ok(Value::Uint(float_to_uint(*a, ut.bits())));
            }
            if let Value::Float128(a) = v {
                return Ok(Value::Uint(f128::to_uint(*a, ut.bits())));
            }
            // A pointer source (a pointer-to-integer transmute)
            // yields the pointer's synthetic address.
            if matches!(v, Value::Ref(_) | Value::FatRef { .. } | Value::FnPtr(_)) {
                return Ok(Value::Uint(wrap_unsigned(synthetic_addr(v), ut.bits())));
            }
            let raw = match v {
                Value::Int(a) => *a as u128,
                Value::Uint(a) => *a,
                Value::Bool(b) => *b as u128,
                other => {
                    return Err(InterpError::type_mismatch(
                        "cast",
                        "numeric source",
                        other.kind_name(),
                    ))
                }
            };
            Ok(Value::Uint(wrap_unsigned(raw, ut.bits())))
        }
        Ty::Float(ft) => {
            // `f128` sources and targets convert through the exact
            // binary128 module, never through an `f64` rounding of
            // the value (an `f64` *storage* widening of an `f16` /
            // `f32` source is exact, so widening from `Value::Float`
            // loses nothing).
            match (v, ft) {
                (Value::Float128(a), FloatTy::F128) => return Ok(Value::Float128(*a)),
                (Value::Float128(a), FloatTy::F64) => {
                    return Ok(Value::Float(f64::from_bits(f128::to_f64_bits(*a))))
                }
                (Value::Float128(a), FloatTy::F32) => {
                    return Ok(Value::Float(
                        f32::from_bits(f128::to_f32_bits(*a)) as f64,
                    ))
                }
                (Value::Float128(a), FloatTy::F16) => {
                    return Ok(Value::Float(f128::f16_bits_to_f64(f128::to_f16_bits(
                        *a,
                    ))))
                }
                (Value::Float(a), FloatTy::F128) => {
                    return Ok(Value::Float128(f128::from_f64(a.to_bits())))
                }
                (Value::Int(a), FloatTy::F128) => {
                    return Ok(Value::Float128(f128::from_int(*a)))
                }
                (Value::Uint(a), FloatTy::F128) => {
                    return Ok(Value::Float128(f128::from_uint(*a)))
                }
                _ => {}
            }
            // Convert at the target width directly: going through
            // `f64` first would double-round wide integer sources.
            let out = match (v, ft) {
                (Value::Int(a), FloatTy::F16) => round_to_f16(*a as f64),
                (Value::Int(a), FloatTy::F32) => (*a as f32) as f64,
                (Value::Int(a), FloatTy::F64) => *a as f64,
                (Value::Uint(a), FloatTy::F16) => round_to_f16(*a as f64),
                (Value::Uint(a), FloatTy::F32) => (*a as f32) as f64,
                (Value::Uint(a), FloatTy::F64) => *a as f64,
                (Value::Float(a), FloatTy::F16) => {
                    if a.is_nan()
                        && matches!(
                            src_ty,
                            Some(Ty::Float(FloatTy::F32)) | Some(Ty::Float(FloatTy::F64))
                        )
                    {
                        // A narrowing conversion carries the payload
                        // across top-aligned and quiets the NaN, as
                        // the hardware conversion does; the result is
                        // stored as the exact widening of that `f16`
                        // pattern.
                        let bits = a.to_bits();
                        let sign = if bits >> 63 != 0 { 0x8000u16 } else { 0 };
                        let payload = ((bits >> 42) & 0x3ff) as u16;
                        f128::f16_bits_to_f64(sign | 0x7c00 | payload | 0x0200)
                    } else {
                        round_to_f16(*a)
                    }
                }
                (Value::Float(a), FloatTy::F32) => {
                    if matches!(
                        src_ty,
                        Some(Ty::Float(FloatTy::F16)) | Some(Ty::Float(FloatTy::F32))
                    ) {
                        // Widening `f16` to `f32` (or an `f32`
                        // identity cast): the stored form already
                        // is the exact widening at the target width,
                        // NaN payloads included — the native
                        // conversion preserves a signaling NaN here
                        // rather than quieting it, which the host
                        // `f64` -> `f32` conversion would not.
                        *a
                    } else {
                        (*a as f32) as f64
                    }
                }
                (Value::Float(a), FloatTy::F64) => {
                    // A widening conversion from `f16` / `f32` quiets
                    // a NaN, as the hardware conversion does; the sign
                    // and the top-aligned payload are preserved. The
                    // stored form is the exact widening, so only the
                    // quiet bit changes.
                    if a.is_nan()
                        && matches!(
                            src_ty,
                            Some(Ty::Float(FloatTy::F16)) | Some(Ty::Float(FloatTy::F32))
                        )
                    {
                        f64::from_bits(a.to_bits() | (1u64 << 51))
                    } else {
                        *a
                    }
                }
                (other, _) => {
                    return Err(InterpError::type_mismatch(
                        "cast",
                        "numeric source",
                        other.kind_name(),
                    ))
                }
            };
            Ok(Value::Float(out))
        }
        Ty::Bool => match v {
            Value::Bool(b) => Ok(Value::Bool(*b)),
            Value::Int(a) => Ok(Value::Bool(*a != 0)),
            Value::Uint(a) => Ok(Value::Bool(*a != 0)),
            other => Err(InterpError::type_mismatch(
                "cast",
                "numeric source",
                other.kind_name(),
            )),
        },
        Ty::Ref(_) => match v {
            // A fat pointer cast to a thin pointer type keeps
            // only its data pointer: the metadata half (a slice
            // length, a string length, a vtable) has no
            // counterpart in the target type.
            Value::FatRef { base, .. } => Ok((**base).clone()),
            Value::DynRef { base, .. } => Ok((**base).clone()),
            Value::Ref(_) => Ok(v.clone()),
            other => Err(InterpError::type_mismatch(
                "cast",
                "pointer source",
                other.kind_name(),
            )),
        },
        _ => Err(InterpError::UnsupportedRvalue {
            kind: format!("Cast to {ty:?}"),
        }),
    }
}

/// Convenience one-shot entry point with the default configuration.
/// Judge the drop of one value of dynamic type (the
/// [`crate::ir::DropPlan::Dynamic`] plan): the value's concrete
/// contents are all the glue can know. Scalars, strings and
/// references own nothing; a `DynRef` recurses into the payload it
/// carries; tuples and closure environments recurse into their
/// elements (neither has a destructor of its own). An ADT or
/// coroutine value might hide a destructor in its concrete type,
/// which this judgment cannot name: an explicit error, never a
/// silent skip — the same rule `DropPlan::Untyped` applies.
/// Judge the drop of a `dyn` value that carries no
/// conversion-accounted glue (see the `DropPlan::Dyn` arm): strip
/// the box shapes the interpreter's own builtins build (a one-field
/// aggregate holding the payload, possibly behind a glue-less
/// `DynRef`) and judge the payload itself with
/// [`drop_dyn_value`], whose shapes that can hide a destructor are
/// explicit errors.
/// The bottom of a pointer wrapper chain: descend field 0
/// through the aggregate layers that carry a pointer — a `Box`
/// over its `Unique`, the `Unique` over its `NonNull`, the
/// `NonNull` over the pointer itself. A layer continues the
/// chain when the value at its field 0 is the pointer (a thin,
/// fat, or trait-object pointer), or — for a multi-field layer,
/// whose remaining fields are the parts beside the pointer (an
/// allocator, a `PhantomData` marker) — the next layer. The
/// descent stops at the first value that is not a wrapper
/// layer: the pointer itself, or the pointee a fabricated
/// one-field box holds inline (its single field is the pointee
/// value, not a pointer). Returns the number of field-0 steps
/// taken and the value found there.
pub(crate) fn wrapper_chain_bottom(value: &Value) -> (usize, &Value) {
    let mut depth = 0usize;
    let mut bottom = value;
    loop {
        let Value::Enum { fields, .. } = bottom else {
            break;
        };
        let Some(first) = fields.first() else {
            break;
        };
        let continues = matches!(
            first,
            Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }
        ) || (fields.len() > 1 && matches!(first, Value::Enum { .. }));
        if !continues {
            break;
        }
        depth += 1;
        bottom = first;
    }
    (depth, bottom)
}
