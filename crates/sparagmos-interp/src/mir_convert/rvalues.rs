//! Converter: rvalue conversion.


use sparagmos_mir as mir;

use crate::ir;

use super::adt_fields::*;
use super::converter::*;
use super::drop_glue::*;
use super::dyn_unsize::*;
use super::errors::*;
use super::helpers::*;
use super::program::*;
use super::run::*;


impl<'a> Converter<'a> {
    // ----- rvalues -----------------------------------------------------------

    pub(super) fn convert_rvalue(
        &self,
        body: &mir::Body,
        rv: &mir::Rvalue,
        dest_ty: Option<&mir::Ty>,
    ) -> Result<ir::Rvalue, ConvertError> {
        use mir::Rvalue as R;
        Ok(match rv {
            R::Use(op) => {
                // A whole-pointee load of a vector-shaped value
                // through a dereference: the shaped load, which
                // assembles the value from a raw block's elements
                // when the pointee is not stored whole.
                if let mir::Operand::Copy(place) | mir::Operand::Move(place) = op {
                    if let Some(shape) = self.deref_vector_shape(body, place) {
                        return Ok(ir::Rvalue::ShapedLoad(
                            self.convert_place(body, place)?,
                            shape,
                            matches!(op, mir::Operand::Move(_)),
                        ));
                    }
                }
                ir::Rvalue::Use(self.convert_operand(body, op, dest_ty)?)
            }
            R::BinaryOp(op, pair) if *op == mir::BinOp::Offset => {
                // `BinaryOp::Offset` is, like a shift, a binary
                // operation whose operands do not share a type: a
                // pointer and an element count. The operands are
                // typed separately the same way (the pointer takes
                // the operation's context — its place type, else the
                // result type, which is the pointer type itself; the
                // count keeps its own type).
                let (lhs, rhs) = self.convert_shift_operands(body, pair, dest_ty, false)?;
                let elem_size = self.offset_elem_size(body, pair, dest_ty);
                let count_width = match self.operand_place_ty(body, &pair.1) {
                    Some(mir::Ty::Uint(uty)) => uty.bits(),
                    _ => 0,
                };
                ir::Rvalue::PtrOffset(lhs, rhs, elem_size, count_width)
            }
            R::BinaryOp(op, pair) => {
                let (lhs, rhs) = if is_shift(*op) {
                    self.convert_shift_operands(body, pair, dest_ty, false)?
                } else {
                    self.convert_binop_operands(
                        body,
                        pair,
                        dest_ty,
                        overflow_base(*op).is_some(),
                        is_comparison(*op),
                    )?
                };
                match overflow_base(*op) {
                    // `BinaryOp` with a *WithOverflow operator yields the
                    // `(value, overflowed)` tuple in rustc MIR, exactly
                    // like `CheckedBinaryOp`. A plain operator in a
                    // `BinaryOp` yields just the value.
                    Some(base) => ir::Rvalue::CheckedBinaryOp(base, Box::new((lhs, rhs))),
                    None => match plain_binop(*op) {
                        Some(b) => ir::Rvalue::BinaryOp(b, Box::new((lhs, rhs))),
                        None => ir::Rvalue::Unsupported(format!("BinaryOp {}", op.name())),
                    },
                }
            }
            R::CheckedBinaryOp(op, pair) => {
                let (lhs, rhs) = if is_shift(*op) {
                    self.convert_shift_operands(body, pair, dest_ty, true)?
                } else {
                    self.convert_binop_operands(
                        body,
                        pair,
                        dest_ty,
                        true,
                        is_comparison(*op),
                    )?
                };
                match overflow_base(*op).or_else(|| plain_binop(*op)) {
                    Some(base) => ir::Rvalue::CheckedBinaryOp(base, Box::new((lhs, rhs))),
                    None => ir::Rvalue::Unsupported(format!("CheckedBinaryOp {}", op.name())),
                }
            }
            R::UnaryOp(op, operand) => {
                let ctx = self
                    .operand_place_ty(body, operand)
                    .or_else(|| dest_ty.cloned());
                let v = self.convert_operand(body, operand, ctx.as_ref())?;
                match op {
                    mir::UnOp::Not => ir::Rvalue::UnaryOp(ir::UnOp::Not, v),
                    mir::UnOp::Neg => ir::Rvalue::UnaryOp(ir::UnOp::Neg, v),
                    mir::UnOp::PtrMetadata => ir::Rvalue::UnaryOp(ir::UnOp::PtrMetadata, v),
                }
            }
            R::Cast(kind, operand, target) => {
                // The operand is read under its own (source) type,
                // never the cast's destination type: a constant
                // operand typed by the destination loses its
                // signedness (an `u128::MAX` operand of an
                // int-to-float cast was read as `-1i128` through
                // the float destination, which names neither).
                // `convert_operand` fills a constant's context
                // from the constant's own recorded type.
                let ctx = self.operand_place_ty(body, operand).or_else(|| {
                    match operand {
                        mir::Operand::Constant(_) => None,
                        _ => dest_ty.cloned(),
                    }
                });
                let v = self.convert_operand(body, operand, ctx.as_ref())?;
                match kind {
                    mir::CastKind::IntToInt
                    | mir::CastKind::FloatToInt
                    | mir::CastKind::FloatToFloat
                    | mir::CastKind::IntToFloat => match self.convert_ty(target, "cast target") {
                        Ok(t) => ir::Rvalue::Cast(v, t),
                        Err(_) => ir::Rvalue::Unsupported(format!(
                            "Cast {} to {}",
                            kind.name(),
                            target.kind_name()
                        )),
                    },
                    mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize) => {
                        match dyn_traits_in(target) {
                            // Sized-to-`dyn`: the vtable identity is
                            // decided here, where both types are
                            // known — the concrete type from the
                            // operand's type, the bounds from the
                            // target.
                            Some(traits) => {
                                let source = self.operand_place_ty(body, operand);
                                match source {
                                    // A `dyn`-to-`dyn` coercion
                                    // (supertrait upcast): the
                                    // concrete type rides on the
                                    // value; only the bounds change.
                                    Some(src) if dyn_traits_in(&src).is_some() => {
                                        ir::Rvalue::Unsize(
                                            v,
                                            ir::UnsizeMeta::Dyn {
                                                concrete: None,
                                                traits,
                                                glue: None,
                                            },
                                        )
                                    }
                                    Some(src) => {
                                        let pointee = unsize_pointee(&src);
                                        let concrete = self.concrete_ty_name(
                                            unsize_concrete_pointee(self, &src, target),
                                        );
                                        // The value carries the
                                        // pointee's drop glue from
                                        // here, like a vtable's
                                        // drop shim (see
                                        // `UnsizeMeta::Dyn`).
                                        let glue = DropGlueBuilder::build(self, pointee);
                                        ir::Rvalue::Unsize(
                                            v,
                                            ir::UnsizeMeta::Dyn {
                                                concrete: Some(concrete),
                                                traits,
                                                glue,
                                            },
                                        )
                                    }
                                    // A constant operand does not
                                    // preserve its type in this MIR
                                    // form, so the concrete type
                                    // cannot be named (the same
                                    // limit as the ClosureFnPointer
                                    // constant case below).
                                    None => ir::Rvalue::Unsupported(format!(
                                        "Cast Unsize to {} (the concrete type is not                                          preserved in the constant operand)",
                                        target.kind_name()
                                    )),
                                }
                            }
                            None => {
                                let array_len = self
                                    .operand_place_ty(body, operand)
                                    .and_then(|src| {
                                        match unsize_pointee(&src) {
                                            mir::Ty::Array(
                                                _,
                                                mir::ArrayLen::Known(n),
                                            ) => Some(*n as u64),
                                            _ => None,
                                        }
                                    });
                                ir::Rvalue::Unsize(
                                    v,
                                    ir::UnsizeMeta::Slice { array_len },
                                )
                            }
                        }
                    }
                    // This coercion does not change the reference
                    // value in this model (a reference carries no
                    // mutability or pointee type at run time).
                    mir::CastKind::PointerCoercion(
                        mir::PointerCoercion::MutToConstPointer,
                    ) => ir::Rvalue::Use(v),
                    // An array-to-pointer coercion retargets a
                    // pointer to an array at its first element,
                    // exactly like the `PtrToPtr` cast below; the
                    // coercion kind itself guarantees the source is
                    // an array pointer and the target its element
                    // pointer, and the type check is a guard.
                    mir::CastKind::PointerCoercion(
                        mir::PointerCoercion::ArrayToPointer,
                    ) => {
                        if ctx
                            .as_ref()
                            .is_some_and(|src| self.cast_enters_first_element(src, target))
                        {
                            ir::Rvalue::PtrFirstElem(v)
                        } else {
                            ir::Rvalue::Use(v)
                        }
                    }
                    mir::CastKind::PtrToPtr => {
                        // A cast from a transparent single-field
                        // wrapper (`UnsafeCell<T>` / `MaybeUninit<T>`)
                        // to the wrapped type changes what the
                        // pointer denotes: the wrapper's only field.
                        // Every other pointer-to-pointer cast leaves
                        // the denoted place unchanged in this model.
                        let entry = ctx
                            .as_ref()
                            .and_then(|src| self.cast_entry_depth(src, target));
                        if let Some((depth, first_elem)) = entry {
                            if first_elem {
                                ir::Rvalue::PtrFieldFirstElem(v, depth)
                            } else {
                                ir::Rvalue::PtrField(v, depth)
                            }
                        } else if ctx.as_ref().is_some_and(|src| {
                            self.cast_enters_first_element(src, target)
                        }) {
                            // A cast from an array / slice / `str`
                            // pointer to an element pointer (what
                            // `as_ptr` returns and `&arr as *const T`
                            // writes) denotes the first element, not
                            // the aggregate as a whole. This precedes
                            // the `DataPtr` extraction below: dropping
                            // the metadata alone would leave the
                            // result denoting the whole aggregate.
                            ir::Rvalue::PtrFirstElem(v)
                        } else if let Some(wrap) = ctx
                            .as_ref()
                            .and_then(|src| self.slice_elem_wrap_direction(src, target))
                        {
                            // A cast between slice pointers
                            // whose element types differ by one
                            // `MaybeUninit` / `UnsafeCell` layer
                            // (`<[MaybeUninit<T>]>::assume_init`
                            // and its kin): the fat pointer and
                            // the storage it denotes are both
                            // unchanged; the representation
                            // difference between the two element
                            // views is reconciled at the access
                            // points (see the rvalue's docs).
                            if wrap {
                                ir::Rvalue::WrapSliceElems(v)
                            } else {
                                ir::Rvalue::UnwrapSliceElems(v)
                            }
                        } else if ctx.as_ref().is_some_and(|src| {
                            self.cast_returns_to_array_base(src, target)
                        }) {
                            // The reverse cast: an element pointer
                            // cast back to a pointer to the whole
                            // array denotes the array as a whole.
                            ir::Rvalue::PtrArrayBase(v)
                        } else if let Some(f) = self.cast_units_from_size(&ctx, target) {
                            // A cast between differently sized
                            // pointees (see `cast_units_from_size`):
                            // the denoted position is unchanged,
                            // but its projection index is
                            // denominated in the source pointee's
                            // elements while later offsets use the
                            // target's, so the cast rewrites the
                            // index as a byte offset. For a thin
                            // source this is the cast's only
                            // effect; fat sources never reach
                            // here (the `DataPtr` path below owns
                            // them).
                            ir::Rvalue::PtrCastUnits(v, f)
                        } else if ptr_pointee(target).is_some_and(|p| {
                            // A pointer cast to a thin target drops
                            // the metadata word when the source value
                            // is fat (`*const [T]` to `*const T`, the
                            // shape of `<[T]>::as_ptr`): the result is
                            // the data pointer. `DataPtr` evaluates to
                            // the operand itself when the value is not
                            // fat, so it is correct for thin sources
                            // too -- including constant operands, whose
                            // source type this MIR form does not
                            // preserve (`ctx` is None). A byte-string
                            // DST struct target (`CStr` and its kin)
                            // is a fat target, not a thin one: its
                            // metadata is the byte length, and the
                            // cast keeps the value whole.
                            !matches!(
                                p,
                                mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
                            ) && !self.pointee_is_byte_string_dst(p)
                        }) {
                            // A cast to a pointee type with a
                            // recorded materialisation plan: a
                            // type-erased constant allocation
                            // operand materialises into the
                            // target type; every other operand
                            // keeps the `DataPtr` behaviour.
                            match self.mat_plan_index(target) {
                                Some(idx) => ir::Rvalue::Materialize(v, idx),
                                None => ir::Rvalue::DataPtr(v),
                            }
                        } else if let Some(idx) = self.mat_plan_index(target) {
                            ir::Rvalue::Materialize(v, idx)
                        } else {
                            ir::Rvalue::Use(v)
                        }
                    }
                    // Reifying a function item into a function
                    // pointer does not change the value in this
                    // model: both are the callable value naming the
                    // function.
                    mir::CastKind::PointerCoercion(
                        mir::PointerCoercion::ReifyFnPointer,
                    ) => ir::Rvalue::Use(v),
                    // A non-capturing closure coerced to a function
                    // pointer: the closure value itself is the
                    // callable, for place operands and for constant
                    // operands alike — the driver preserves the
                    // closure type of zero-sized closure constants,
                    // and constant conversion resolves it to the
                    // closure value (an environment whose values
                    // the capture types do not determine stays an
                    // explicit unsupported there).
                    mir::CastKind::PointerCoercion(
                        mir::PointerCoercion::ClosureFnPointer,
                    ) => ir::Rvalue::Use(v),
                    mir::CastKind::BoxDerefTransmute => ir::Rvalue::BoxDerefTransmute(v),
                    // `ptr as usize`: exposing a pointer's
                    // provenance yields its address as an integer,
                    // evaluated to the pointer's synthetic address
                    // (`Rvalue::PtrToInt`) — the same treatment as
                    // the pointer-to-integer transmute below. This
                    // is the cast rustc emits for `ptr as usize` in
                    // user code (e.g. alignment checks on raw
                    // pointers). The reverse direction (an integer
                    // cast back to a pointer,
                    // PointerWithExposedProvenance) is the next
                    // arm below. (Lane BV)
                    mir::CastKind::PointerExposeProvenance => match (&ctx, target) {
                        (
                            Some(
                                mir::Ty::RawPtr(..)
                                | mir::Ty::Ref(..)
                                | mir::Ty::FnPtr(..)
                                | mir::Ty::FnDef(..),
                            ),
                            mir::Ty::Int(_) | mir::Ty::Uint(_),
                        ) => ir::Rvalue::PtrToInt(v.clone()),
                        // A constant operand carries no place type
                        // (the `ctx` above holds the cast's target
                        // type for it), so a pointer constant is
                        // judged from the converted constant itself:
                        // the constant forms that evaluate to a
                        // pointer value take the same `PtrToInt`
                        // reading, which yields the value's
                        // synthetic address exactly as for a place
                        // operand.
                        _ if matches!(target, mir::Ty::Int(_) | mir::Ty::Uint(_))
                            && is_pointer_constant(&v) =>
                        {
                            ir::Rvalue::PtrToInt(v.clone())
                        }
                        _ => ir::Rvalue::Unsupported(
                            "Cast PointerExposeProvenance".to_string(),
                        ),
                    },
                    // The reverse provenance cast (an integer
                    // address back to a pointer,
                    // `with_exposed_provenance` / `addr as *mut T`):
                    // the same address-only reference the
                    // integer -> pointer transmute produces. It
                    // compares, tests for null and casts back to
                    // an integer; reading or writing through it is
                    // an explicit error, as for any address-only
                    // pointer in this model.
                    mir::CastKind::PointerWithExposedProvenance => {
                        let src_is_int = matches!(
                            ctx,
                            Some(mir::Ty::Int(_) | mir::Ty::Uint(_))
                        ) || matches!(
                            v,
                            ir::Operand::Constant(
                                ir::Constant::Int(_) | ir::Constant::Uint(_)
                            )
                        );
                        if src_is_int && is_thin_ptr_ty(target) {
                            ir::Rvalue::IntToPtr(v.clone())
                        } else {
                            ir::Rvalue::Unsupported(
                                "Cast PointerWithExposedProvenance"
                                    .to_string(),
                            )
                        }
                    }
                    // A subtyping cast changes only lifetimes,
                    // which this model does not track (a reference
                    // value carries no lifetime): the value itself
                    // is unchanged.
                    mir::CastKind::Subtype => ir::Rvalue::Use(v),
                    // `transmute`: the shapes the value model
                    // can reinterpret exactly (pointers, integers,
                    // fieldless enums, transparent wrappers,
                    // zero-sized types) are decided in
                    // `convert_transmute`; every other shape
                    // stays an explicit unsupported.
                    mir::CastKind::Transmute => {
                        self.convert_transmute(body, operand, &v, target)
                    }
                    other => ir::Rvalue::Unsupported(format!("Cast {}", other.name())),
                }
            }
            R::Aggregate(kind, ops) => match &**kind {
                // The empty tuple is the unit value: its type
                // converts to `Ty::Unit`, so the aggregate must
                // produce `Value::Unit`, not an empty tuple value.
                mir::AggregateKind::Tuple if ops.is_empty() => {
                    ir::Rvalue::Use(ir::Operand::Constant(ir::Constant::Unit))
                }
                mir::AggregateKind::Tuple | mir::AggregateKind::Array(_) => {
                    let elem_tys: Vec<mir::Ty> = match dest_ty {
                        Some(mir::Ty::Tuple(ts)) => ts.clone(),
                        Some(mir::Ty::Array(elem, _)) => {
                            vec![(**elem).clone(); ops.len()]
                        }
                        _ => Vec::new(),
                    };
                    let mut items = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        let ctx = elem_tys.get(i);
                        items.push(self.convert_operand(body, op, ctx)?);
                    }
                    ir::Rvalue::AggregateTuple(items)
                }
                mir::AggregateKind::Adt {
                    def,
                    variant_index,
                    generic_args,
                    field_index,
                    ..
                } => {
                    // A `MaybeUninit::uninit()` construction: the
                    // union aggregate activates the `uninit: ()`
                    // field, but in this value model a
                    // `MaybeUninit<T>` holds its payload at field
                    // 0 -- the slot its pointer accesses enter --
                    // and the payload of an uninitialized
                    // `MaybeUninit` is uninitialized memory of
                    // type `T`, not the unit value the field's
                    // declaration suggests. Storing the unit value
                    // there left every access that navigates
                    // inside the payload (an element write through
                    // `as_mut_ptr` into a `MaybeUninit<[T; N]>`,
                    // say) facing a unit where the payload's own
                    // shape belongs. Emit the value the model
                    // holds for an uninitialized `MaybeUninit`:
                    // the payload slot carrying the uninitialized
                    // skeleton of `T`, the same expansion an
                    // uninitialized constant of this type receives
                    // (see `expand_uninit_constant`).
                    if let Some(active) = field_index {
                        if self.is_maybe_uninit(def) {
                            // The active field's declared type, from
                            // the ADT's shape (the field table is
                            // recovered from projections, and the
                            // `uninit` field is never projected).
                            let active_ty = self
                                .adt_shapes
                                .get(def)
                                .and_then(|shape| {
                                    shape.variants.get(*variant_index as usize)
                                })
                                .and_then(|fields| fields.get(*active as usize))
                                .map(|fty| subst_ty(fty, generic_args));
                            if let (Some(payload_ty), Some(field_ty)) =
                                (generic_args.first(), active_ty)
                            {
                                if field_ty.is_unit() {
                                    let payload = self.expand_uninit_constant(
                                        &ir::Constant::Uninit,
                                        payload_ty,
                                        0,
                                    );
                                    return Ok(ir::Rvalue::Use(ir::Operand::Constant(
                                        ir::Constant::Adt {
                                            variant: 0,
                                            fields: vec![payload],
                                        },
                                    )));
                                }
                            }
                        }
                    }
                    // For `Range` / `RangeInclusive` the two bound
                    // fields are the ADT's single generic parameter;
                    // typing literal bounds by it keeps them from
                    // being read as unsigned by default.
                    let bound_ty = {
                        let name = self.names.get(def).map(String::as_str);
                        if matches!(
                            name,
                            Some("std::ops::Range" | "std::ops::RangeInclusive")
                        ) {
                            generic_args.first()
                        } else {
                            None
                        }
                    };
                    let mut fields = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        // The field's declared type, recovered from
                        // `Field` projections program-wide, is the
                        // primary context: it is what the stored
                        // value will be read as. The Range bound rule
                        // stays as the fallback for fields no place
                        // ever projects (their values are consumed by
                        // runtime support, not by MIR places).
                        // A union aggregate carries a single operand
                        // for its active field (`field_index`), so
                        // that field's declared type is the context,
                        // not field 0's.
                        let field = match field_index {
                            Some(active) => *active,
                            None => i as u32,
                        };
                        let declared = self.adt_fields.get(
                            *def,
                            generic_args,
                            *variant_index,
                            field,
                        );
                        let ctx = declared.or_else(|| if i < 2 { bound_ty } else { None });
                        fields.push(self.convert_operand(body, op, ctx)?);
                    }
                    ir::Rvalue::AggregateAdt {
                        variant: *variant_index,
                        operands: fields,
                        field_index: *field_index,
                        units_from: None,
                    }
                }
                mir::AggregateKind::Closure { def, generic_args } => {
                    // The closure type's LAST generic argument is
                    // the upvars tuple; its element types type the
                    // capture operands.
                    let upvar_tys: Vec<mir::Ty> = match generic_args.last() {
                        Some(mir::Ty::Tuple(ts)) => ts.clone(),
                        _ => Vec::new(),
                    };
                    let mut fields = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        fields.push(self.convert_operand(body, op, upvar_tys.get(i))?);
                    }
                    ir::Rvalue::ClosureAggregate {
                        def: self.body_key(def),
                        operands: fields,
                    }
                }
                mir::AggregateKind::CoroutineClosure { def, generic_args } => {
                    // An async closure's creation aggregate: the
                    // value is the closure environment, exactly as
                    // for a plain closure - what differs is only
                    // how a call on it is dispatched (the AsyncFn
                    // family creates the coroutine the closure's
                    // captures feed). The generic arguments pack
                    // the tail as [kind, signature, upvars tuple,
                    // by-reference captures fn-pointer], so the
                    // upvars tuple is the second-to-last argument
                    // (a plain closure's is the last).
                    let upvar_tys: Vec<mir::Ty> = match generic_args.iter().rev().nth(1) {
                        Some(mir::Ty::Tuple(ts)) => ts.clone(),
                        _ => Vec::new(),
                    };
                    let mut fields = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        fields.push(self.convert_operand(body, op, upvar_tys.get(i))?);
                    }
                    ir::Rvalue::ClosureAggregate {
                        def: self.body_key(def),
                        operands: fields,
                    }
                }
                mir::AggregateKind::Coroutine { def, generic_args } => {
                    // The creation aggregate fills the unresumed
                    // state's fields: the upvars. Their declared
                    // types come from the field table (the coroutine
                    // body's own projections of state 0), with the
                    // upvars tuple - the coroutine type's first
                    // generic argument - as the fallback.
                    let upvar_tys: Vec<mir::Ty> = match generic_args.first() {
                        Some(mir::Ty::Tuple(ts)) => ts.clone(),
                        _ => Vec::new(),
                    };
                    let mut fields = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        let declared =
                            self.adt_fields.get(*def, generic_args, 0, i as u32);
                        let ctx = declared.or_else(|| upvar_tys.get(i));
                        fields.push(self.convert_operand(body, op, ctx)?);
                    }
                    ir::Rvalue::CoroutineAggregate(fields)
                }
                mir::AggregateKind::RawPtr { data_ty, .. } => {
                    // `ptr::from_raw_parts` shape: one operand for a
                    // sized pointee (the aggregate is the pointer
                    // itself), two for an unsized one (data pointer
                    // plus metadata). Slice metadata is a length and
                    // becomes a fat reference; `dyn` metadata is a
                    // `DynMetadata` value, whose vtable the
                    // evaluation resolves through the program's
                    // vtable identity table.
                    let data_ctx = mir::Ty::RawPtr(
                        Box::new(data_ty.clone()),
                        mir::Mutability::Not,
                    );
                    match (data_ty, ops.len()) {
                        (mir::Ty::Slice(_), 2) => {
                            let data =
                                self.convert_operand(body, &ops[0], Some(&data_ctx))?;
                            let meta = self.convert_operand(
                                body,
                                &ops[1],
                                Some(&mir::Ty::Uint(mir::UintTy::Usize)),
                            )?;
                            ir::Rvalue::RawPtrAggregate {
                                data,
                                meta: Some(meta),
                            }
                        }
                        (mir::Ty::Dynamic(..), 2) => {
                            // As the generic two-operand shape
                            // below: the metadata operand converts
                            // without a context, and the evaluation
                            // resolves the vtable it names.
                            let data =
                                self.convert_operand(body, &ops[0], Some(&data_ctx))?;
                            let meta = self.convert_operand(body, &ops[1], None)?;
                            ir::Rvalue::RawPtrAggregate {
                                data,
                                meta: Some(meta),
                            }
                        }
                        (_, 1) => {
                            let data =
                                self.convert_operand(body, &ops[0], Some(&data_ctx))?;
                            ir::Rvalue::RawPtrAggregate { data, meta: None }
                        }
                        (_, 2) => {
                            // Any other two-operand shape (a sized
                            // pointee with unit metadata, a `str`
                            // pointer's length, ...): the metadata
                            // operand converts without a context and
                            // the evaluation decides its meaning
                            // (unit metadata is no metadata).
                            let data =
                                self.convert_operand(body, &ops[0], Some(&data_ctx))?;
                            let meta = self.convert_operand(body, &ops[1], None)?;
                            ir::Rvalue::RawPtrAggregate {
                                data,
                                meta: Some(meta),
                            }
                        }
                        _ => ir::Rvalue::Unsupported(format!(
                            "Aggregate RawPtr ({} operands)",
                            ops.len()
                        )),
                    }
                }
                other => ir::Rvalue::Unsupported(format!("Aggregate {}", other.kind_name())),
            },
            R::Repeat(op, count) => {
                // A repeat count is guest-controlled input: a huge
                // count (e.g. `[0u8; 1 << 40]`) would make the
                // aggregate allocation abort the host process instead
                // of failing the run. Like the interpreter's step and
                // call-depth limits, the element count is a resource
                // bound; past it the rvalue stays an explicit
                // unsupported error.
                if *count > MAX_REPEAT_ELEMENTS {
                    ir::Rvalue::Unsupported(format!("Repeat of {count} elements"))
                } else {
                    let elem_ty = match dest_ty {
                        Some(mir::Ty::Tuple(ts)) => ts.first().cloned(),
                        Some(mir::Ty::Array(elem, _)) => Some((**elem).clone()),
                        _ => None,
                    };
                    let v = self.convert_operand(body, op, elem_ty.as_ref())?;
                    // A repeat evaluates its operand once and
                    // replicates the value; the operand is written
                    // as a move in MIR even for `Copy` element
                    // types (an array of `Copy` elements is itself
                    // `Copy`). Replicating a move operand would
                    // read the source place once per element and
                    // fail on the second read, the source having
                    // been moved out of. Read it as a copy instead:
                    // the value is replicated, not moved per
                    // element.
                    let v = match v {
                        ir::Operand::Move(place) => ir::Operand::Copy(place),
                        other => other,
                    };
                    ir::Rvalue::AggregateTuple(vec![v; *count as usize])
                }
            }
            R::Ref(_, _, place) => ir::Rvalue::Ref(self.convert_place(body, place)?),
            R::CopyForDeref(place) => ir::Rvalue::CopyForDeref(self.convert_place(body, place)?),
            // Raw pointer creation shares the reference representation.
            R::RawPtr(_, place) => ir::Rvalue::Ref(self.convert_place(body, place)?),
            R::Reborrow(_, _, place) => ir::Rvalue::Reborrow(self.convert_place(body, place)?),
            R::ThreadLocalRef(def) => ir::Rvalue::ThreadLocalRef(self.static_key(def)),
            R::Len(place) => ir::Rvalue::Len(self.convert_place(body, place)?),
            R::NullaryOp(op, _) => ir::Rvalue::Unsupported(format!("NullaryOp {}", op.kind_name())),
            R::Discriminant(place) => {
                let converted = self.convert_place(body, place)?;
                // Bake the discriminant table of the enum into the
                // rvalue: at execution time the place type is no
                // longer available, and the table is what separates
                // the discriminant value from the variant index.
                // No table (hand-built programs, non-ADT places)
                // leaves the table empty, which the interpreter
                // reads as "values are the variant indices".
                let discriminants = match self.place_ty(body, place) {
                    Some(mir::Ty::Adt(def, _)) => {
                        self.discriminants.get(&def).cloned().unwrap_or_default()
                    }
                    _ => Vec::new(),
                };
                ir::Rvalue::Discriminant(converted, discriminants)
            }
            R::ShallowInitBox(_, _) => ir::Rvalue::Unsupported("ShallowInitBox rvalue".to_string()),
        })
    }

}
