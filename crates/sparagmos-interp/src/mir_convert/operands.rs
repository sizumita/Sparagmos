//! Converter: constants and operands.


use sparagmos_mir as mir;

use crate::ir;
use crate::value::{RefProj};

use super::adt_fields::*;
use super::converter::*;
use super::drop_glue::*;
use super::errors::*;
use super::helpers::*;
use super::program::*;
use super::run::*;


impl<'a> Converter<'a> {
    // ----- constants & operands ---------------------------------------------

    pub(super) fn convert_constant(
        &self,
        constant: &mir::Constant,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Constant, ConvertError> {
        match &constant.literal {
            mir::ConstKind::Value(v) => {
                let k = self.convert_const_value(v, ctx)?;
                // A constant whose bytes are (partly) uninitialized
                // decodes to a bare `Uninit` whatever its type; give
                // it the aggregate skeleton of its declared type so
                // element/field writes and reads see the structure
                // (see `expand_uninit_constant`). Without a context
                // type there is no skeleton to rebuild.
                Ok(match ctx {
                    Some(ty) => self.expand_uninit_constant(&k, ty, 0),
                    None => k,
                })
            }
            // A function item in operand position is a value (the
            // zero-sized fn item, reifiable to a function pointer):
            // it becomes a callable constant naming the function.
            mir::ConstKind::Ty(mir::Ty::FnDef(def, args)) => {
                Ok(ir::Constant::FnDef(self.fn_def_value_key(def, args)))
            }
            // A closure in constant position is a value too (the
            // zero-sized closure constant a `ClosureFnPointer` cast
            // coerces, or a stored closure value). A type constant
            // carries no stored bytes, so the value is materialised
            // from the type alone: the captured environment is
            // materialised exactly when every captured value is
            // the unique value of its type (a non-capturing closure
            // trivially qualifies). A capture whose type has
            // several values cannot be materialised here and stays
            // an explicit unsupported naming the closure.
            mir::ConstKind::Ty(mir::Ty::Closure(def, args)) => {
                // The upvars tuple is the closure type's LAST
                // generic argument (see `convert_ty`); the third
                // argument is the upvars tuple only when the
                // closure has no parent generics, so a closure
                // constant inside a generic context (e.g. the
                // closure inside `Sum::sum`) was misread as
                // capturing an environment it does not have.
                let env = match args.last() {
                    Some(mir::Ty::Tuple(ts)) => ts
                        .iter()
                        .map(|t| {
                            self.zst_value(t, 0)
                                .and_then(|v| value_to_constant(&v))
                        })
                        .collect::<Option<Vec<_>>>(),
                    _ => None,
                };
                match env {
                    Some(env) => Ok(ir::Constant::Closure(
                        self.body_key(def),
                        env,
                    )),
                    None => Err(ConvertError::UnsupportedConstant {
                        kind: format!(
                            "closure constant {} with a captured environment",
                            self.def_display_name(def)
                        ),
                    }),
                }
            }
            mir::ConstKind::Ty(t) => Err(ConvertError::UnsupportedConstant {
                kind: format!("type constant {} outside call position", t.kind_name()),
            }),
            mir::ConstKind::Unevaluated { def, .. } => Err(ConvertError::UnsupportedConstant {
                kind: format!("unevaluated constant {def}"),
            }),
        }
    }

    /// A projection step of a static reference, in the
    /// interpreter's reference-projection form.
    fn convert_ref_proj(&self, elem: &mir::PlaceElem) -> Result<RefProj, ConvertError> {
        Ok(match elem {
            mir::PlaceElem::Field(idx, _) => RefProj::Field(*idx as usize),
            mir::PlaceElem::ConstantIndex {
                offset,
                from_end: false,
                ..
            } => RefProj::Index(*offset as usize),
            mir::PlaceElem::Downcast(_, variant) => RefProj::Downcast(*variant),
            other => {
                return Err(ConvertError::UnsupportedConstant {
                    kind: format!("projection {other:?} in a static reference"),
                })
            }
        })
    }

    /// Record the initializer seed a `StaticRef` conversion
    /// produced for the static `name` (see
    /// `ir::Program::static_seeds`). One seed per static is
    /// kept, ranked by how fully it was expanded: a seed
    /// expanded from the static's recorded declared type (2)
    /// outranks one expanded from a whole-static reference's
    /// own context (1), which outranks a bare seed (0) -- the
    /// seed an interior reference keeps, or one rebuilt to a
    /// slice reference's extent. The ranking gives the storage
    /// the fullest skeleton available whichever reference is
    /// evaluated first, the same outcome the recorded-type
    /// expansion exists to guarantee; seeds of one rank are
    /// interchangeable, so the first recorded wins.
    fn register_static_seed(&self, name: &str, priority: u8, seed: ir::Constant) {
        let mut seeds = self.static_seeds.borrow_mut();
        match seeds.get(name) {
            Some((existing, _)) if *existing >= priority => {}
            _ => {
                seeds.insert(name.to_string(), (priority, seed));
            }
        }
    }

    /// Expand the `Uninit` leaves of a seed constant (a
    /// thread-local's initializer, a static's initializer, an
    /// operand constant) into the aggregate skeleton of their
    /// declared type.
    ///
    /// The constant decoder collapses every all-uninitialized byte
    /// range into a bare `Uninit`, whatever its type. That loses
    /// the value's shape wherever the constant is used: a
    /// thread-local storage's `data` field, or — under optimized
    /// MIR, where `[MaybeUninit::uninit(); N]` const-propagates
    /// into a single `<uninit>` constant of the array type — an
    /// ordinary local whose elements are then written and read
    /// through dynamic indices. For a thread-local storage: the
    /// `data` field is an `UnsafeCell<MaybeUninit<T>>`, a real value
    /// of an aggregate type, and std's lazy initialization copies
    /// the `MaybeUninit` out by value (`ptr::replace` in
    /// `get_or_init_slow`) -- a read that must yield the uninitialized
    /// `MaybeUninit` itself, not fail for touching uninitialized
    /// memory. Guest code materializes that same value through the
    /// `MaybeUninit::uninit()` construction, whose conversion the
    /// aggregate arm of `convert_rvalue` rewrites to the model's
    /// field-0 payload form, so the expansion rebuilds exactly that
    /// form for a `MaybeUninit`: the payload slot carrying the
    /// payload type's own expansion. For the rest: structs and
    /// tuples get one slot per field, arrays get one slot per
    /// element, a union gets its unit-typed
    /// field (the `uninit: ()` pattern) with `Uninit` padding before
    /// it, and scalar leaves stay `Uninit`. Unions without a unit
    /// field have no canonical uninitialized form and keep the bare
    /// `Uninit`, as do enums (no variant is determined).
    pub(super) fn expand_uninit_constant(
        &self,
        value: &ir::Constant,
        ty: &mir::Ty,
        depth: usize,
    ) -> ir::Constant {
        if depth > 16 {
            return value.clone();
        }
        match (value, ty) {
            (ir::Constant::Uninit, mir::Ty::Adt(def, args)) => {
                // A `MaybeUninit<T>` is held as a single field-0
                // payload (see the aggregate conversion in
                // `convert_rvalue`): an uninitialized one carries
                // the uninitialized skeleton of `T` in that slot,
                // not the union's unit-typed `uninit` field value,
                // so a constant of this type expands to the same
                // value the `MaybeUninit::uninit()` construction
                // materializes. (Bulk consumers are unaffected by
                // the shape: `write_bytes` counts uninitialised
                // leaves as occupying their slots when it decides
                // a fill is the target's whole value.)
                if self.is_maybe_uninit(def) {
                    if let Some(payload_ty) = args.first() {
                        let payload = self.expand_uninit_constant(
                            &ir::Constant::Uninit,
                            payload_ty,
                            depth + 1,
                        );
                        return ir::Constant::Adt {
                            variant: 0,
                            fields: vec![payload],
                        };
                    }
                }
                let Some(shape) = self.adt_shapes.get(def) else {
                    return value.clone();
                };
                match shape.kind {
                    mir::AdtShapeKind::Struct => {
                        let Some(fields) = shape.variants.first() else {
                            return value.clone();
                        };
                        let fields = fields
                            .iter()
                            .enumerate()
                            .map(|(fi, fty)| {
                                let fty = self
                                    .adt_fields
                                    .get(*def, args, 0, fi as u32)
                                    .cloned()
                                    .unwrap_or_else(|| subst_ty(fty, args));
                                self.expand_uninit_constant(&ir::Constant::Uninit, &fty, depth + 1)
                            })
                            .collect();
                        ir::Constant::Adt {
                            variant: 0,
                            fields,
                        }
                    }
                    mir::AdtShapeKind::Union => {
                        let Some(fields) = shape.variants.first() else {
                            return value.clone();
                        };
                        let unit_idx = fields
                            .iter()
                            .position(|f| subst_ty(f, args).is_unit() || f.is_unit());
                        match unit_idx {
                            Some(k) => {
                                let mut fields =
                                    vec![ir::Constant::Uninit; k];
                                fields.push(ir::Constant::Unit);
                                ir::Constant::Adt {
                                    variant: 0,
                                    fields,
                                }
                            }
                            None => value.clone(),
                        }
                    }
                    mir::AdtShapeKind::Enum => value.clone(),
                }
            }
            (ir::Constant::Uninit, mir::Ty::Tuple(ts)) => ir::Constant::Tuple(
                ts.iter()
                    .map(|t| self.expand_uninit_constant(&ir::Constant::Uninit, t, depth + 1))
                    .collect(),
            ),
            (ir::Constant::Adt { variant, fields }, mir::Ty::Adt(def, args)) => {
                let Some(shape) = self.adt_shapes.get(def) else {
                    return value.clone();
                };
                let Some(field_tys) = shape.variants.get(*variant as usize) else {
                    return value.clone();
                };
                let fields = fields
                    .iter()
                    .enumerate()
                    .map(|(fi, f)| match field_tys.get(fi) {
                        Some(fty) => {
                            let fty = self
                                .adt_fields
                                .get(*def, args, *variant, fi as u32)
                                .cloned()
                                .unwrap_or_else(|| subst_ty(fty, args));
                            self.expand_uninit_constant(f, &fty, depth + 1)
                        }
                        None => f.clone(),
                    })
                    .collect();
                ir::Constant::Adt {
                    variant: *variant,
                    fields,
                }
            }
            (ir::Constant::Tuple(fields), mir::Ty::Tuple(ts)) => ir::Constant::Tuple(
                fields
                    .iter()
                    .zip(ts.iter())
                    .map(|(f, t)| self.expand_uninit_constant(f, t, depth + 1))
                    .collect(),
            ),
            (ir::Constant::Tuple(fields), mir::Ty::Array(elem, _)) => ir::Constant::Tuple(
                fields
                    .iter()
                    .map(|f| self.expand_uninit_constant(f, elem, depth + 1))
                    .collect(),
            ),
            (ir::Constant::Uninit, mir::Ty::Array(elem, len)) => {
                // An uninitialized array constant is an array of
                // uninitialized elements: expand it element by
                // element so the value carries the array's shape
                // and element writes/reads through dynamic indices
                // see the same structure the unoptimized `Repeat`
                // form materializes. Lengths that are not known
                // constants, and arrays past the repeat
                // materialization limit, keep the bare `Uninit`.
                let mir::ArrayLen::Known(n) = len else {
                    return value.clone();
                };
                if *n > MAX_REPEAT_ELEMENTS {
                    return value.clone();
                }
                let elem =
                    self.expand_uninit_constant(&ir::Constant::Uninit, elem, depth + 1);
                ir::Constant::Tuple(vec![elem; *n as usize])
            }
            _ => value.clone(),
        }
    }

    /// The modelled-contents form of an empty `Vec` / `String`
    /// constant, when `value` is one's raw-layout constant tree
    /// (see the `Adt` arm of `convert_const_value`). The raw
    /// length is the second field of the raw `Vec` tree (the
    /// first is the buffer); a `String` wraps that tree in its
    /// single field.
    fn empty_modelled_container(
        &self,
        value: &mir::ConstValue,
        ctx: Option<&mir::Ty>,
    ) -> Option<ir::Constant> {
        let Some(mir::Ty::Adt(def, _)) = ctx else {
            return None;
        };
        let name = self.names.get(def).or_else(|| self.def_names.get(def))?;
        let mir::ConstValue::Adt { fields, .. } = value else {
            return None;
        };
        let raw_vec_len = |fields: &[mir::ConstValue]| -> Option<u128> {
            match fields.get(1) {
                Some(mir::ConstValue::Scalar(mir::Scalar::Int { data, .. })) => {
                    Some(*data)
                }
                _ => None,
            }
        };
        match name.as_str() {
            "std::vec::Vec" => (raw_vec_len(fields)? == 0).then_some(ir::Constant::Adt {
                variant: 0,
                fields: Vec::new(),
            }),
            "std::string::String" => {
                let mir::ConstValue::Adt {
                    fields: inner_fields,
                    ..
                } = fields.first()?
                else {
                    return None;
                };
                (raw_vec_len(inner_fields)? == 0)
                    .then_some(ir::Constant::Str(String::new()))
            }
            _ => None,
        }
    }

    pub(super) fn convert_const_value(
        &self,
        value: &mir::ConstValue,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Constant, ConvertError> {
        Ok(match value {
            mir::ConstValue::Bool(b) => ir::Constant::Bool(*b),
            mir::ConstValue::Char(c) => ir::Constant::Uint(*c as u128),
            mir::ConstValue::ZeroSized => {
                // A zero-sized constant carries no value of its own:
                // its value is the one its context type determines.
                // When that value is a structure holding a callable
                // (a closure / function pointer inside an enum or
                // tuple - e.g. a `Fun<F>` wrapper whose only field is
                // a zero-capture closure passed as a constant
                // receiver), materialise it so projections and calls
                // through it can reach the callable. Every other
                // zero-sized constant keeps the unit reading: most
                // ZST types' unique values are plain aggregates whose
                // constant form would change existing handling.
                match ctx.and_then(|ty| self.zst_value(ty, 0)) {
                    Some(v) if value_contains_callable(&v) => {
                        match value_to_constant(&v) {
                            Some(k) => k,
                            None => ir::Constant::Unit,
                        }
                    }
                    _ => ir::Constant::Unit,
                }
            }
            mir::ConstValue::Scalar(s) => return self.convert_scalar(s, ctx),
            mir::ConstValue::Str(s) => ir::Constant::Str(s.clone()),
            mir::ConstValue::Bytes(b) => {
                // The bytes are the elements themselves only for
                // byte-sized elements (the fmt pieces template is the
                // canonical case); wider elements are packed
                // little-endian encodings this layer does not decode.
                if let Some(size) = pointee_elem_size(ctx) {
                    if size != 1 {
                        return Err(ConvertError::UnsupportedConstant {
                            kind: format!("byte constant with {size}-byte elements"),
                        });
                    }
                }
                // In pointer-to-slice position the bytes are not
                // the value: the value is the fat pointer to them
                // (a `&[u8]` static's initializer decodes to this
                // form), and a bare byte value where a slice
                // pointer belongs loses the length every consumer
                // of the pointer reads.
                if ctx_is_u8_slice_ptr(ctx) {
                    ir::Constant::SliceVal(
                        b.iter()
                            .map(|byte| ir::Constant::Uint(u128::from(*byte)))
                            .collect(),
                    )
                } else {
                    ir::Constant::Bytes(b.clone())
                }
            }
            mir::ConstValue::Slice { data, meta } => {
                // A slice constant whose metadata equals its byte
                // length is a byte slice.
                if *meta as usize == data.len() {
                    // As with the byte form above: in
                    // pointer-to-slice position the constant is
                    // the fat pointer, not the pointee data.
                    if ctx_is_u8_slice_ptr(ctx) {
                        ir::Constant::SliceVal(
                            data.iter()
                                .map(|byte| ir::Constant::Uint(u128::from(*byte)))
                                .collect(),
                        )
                    } else {
                        ir::Constant::Bytes(data.clone())
                    }
                } else {
                    return Err(ConvertError::UnsupportedConstant {
                        kind: "slice constant with non-byte elements".to_string(),
                    });
                }
            }
            mir::ConstValue::NullPtr => ir::Constant::NullPtr,
            mir::ConstValue::Uninit => ir::Constant::Uninit,
            mir::ConstValue::Tuple(fields) => ir::Constant::Tuple(
                fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        // A promoted array constant is represented
                        // as a tuple of its elements (inside a
                        // by-reference constant); under an `Array`
                        // context every field takes the element type.
                        // Without it the fields decode context-free,
                        // which misreads the sign of 128-bit elements.
                        let elem_ctx = match ctx {
                            Some(mir::Ty::Tuple(ts)) => ts.get(i),
                            Some(mir::Ty::Array(elem, _)) => Some(&**elem),
                            _ => None,
                        };
                        self.convert_const_value(f, elem_ctx)
                    })
                    .collect::<Result<_, _>>()?,
            ),
            mir::ConstValue::Adt { variant, fields } => {
                // A constant of a modelled container type arrives
                // in the type's raw layout (a `Vec` constant is
                // the nested RawVec tree const evaluation
                // produced), but the value model represents the
                // container by its contents: a `Vec::new()`
                // default field value would otherwise materialise
                // as a raw-layout aggregate whose two fields the
                // `Vec::len` builtin reads as a length of 2. An
                // empty container's contents are empty in both
                // layouts, so it translates exactly; a non-empty
                // one's elements live in the buffer allocation the
                // raw tree points into, which this layer does not
                // decode, so it keeps the raw form.
                if let Some(k) = self.empty_modelled_container(value, ctx) {
                    return Ok(k);
                }
                // The fields take the ADT's instantiated field
                // types as their context, exactly as the tuple arm
                // takes its element types: a scalar field's bits
                // carry no signedness of their own, so a `u128`
                // field holding 2^127 decoded context-free reads
                // back as a negative `i128` (the `Some(1 << 127)`
                // constant in checked_shl's test, among others).
                let field_ctxs = match ctx {
                    Some(mir::Ty::Adt(def, args)) => {
                        self.instantiated_field_tys(def, args, *variant)
                    }
                    _ => None,
                };
                let mut fields: Vec<ir::Constant> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let fctx = field_ctxs
                            .as_ref()
                            .and_then(|ts| ts.get(i))
                            .map(|t| t as &mir::Ty);
                        self.convert_const_value(f, fctx)
                    })
                    .collect::<Result<_, _>>()?;
                // A decoded `MaybeUninit<T>` constant carries one
                // slot per union field -- the `uninit: ()` slot
                // beside the `value: ManuallyDrop<T>` slot, both
                // decoded from the same bytes -- but the value
                // model holds a `MaybeUninit` as a single field-0
                // payload: the shape `MaybeUninit::new`
                // constructions store (see the aggregate
                // evaluation in the interpreter) and the shape its
                // pointer accesses read. Normalize the two-slot
                // decoded form to that shape: the payload is the
                // `value` slot's constant with the `ManuallyDrop`
                // wrapper unwrapped.
                if fields.len() == 2 {
                    if let Some(mir::Ty::Adt(def, _)) = ctx {
                        let name = self.def_display_name(def);
                        if matches!(
                            name.as_str(),
                            "std::mem::MaybeUninit" | "core::mem::MaybeUninit"
                        ) {
                            let value_slot = fields.pop().expect("two slots");
                            let payload = match value_slot {
                                ir::Constant::Adt { fields: inner, .. }
                                    if inner.len() == 1 =>
                                {
                                    inner.into_iter().next().expect("one field")
                                }
                                other => other,
                            };
                            fields = vec![payload];
                        }
                    }
                }
                ir::Constant::Adt {
                    variant: *variant,
                    fields,
                }
            }
            mir::ConstValue::Ref(inner) => {
                // The pointee's declared type types the inner value:
                // a scalar leaf carries raw bits with no signedness of
                // its own, and a pointee is read back through a deref
                // with no typed store in between to re-tag it, so
                // dropping the context here would misread an unsigned
                // leaf with its sign bit set as a negative signed
                // value (a promoted `200.0f64 as u8` printing as -56).
                let inner_ctx = match ctx {
                    Some(mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _)) => {
                        Some(&**pointee)
                    }
                    _ => None,
                };
                ir::Constant::Ref(Box::new(self.convert_const_value(inner, inner_ctx)?))
            }
            mir::ConstValue::ErasedAlloc { bytes, relocs } => {
                // The block's relocations are reference values in
                // their own right (a nested block, a function
                // value, an address); each converts with no type
                // context, exactly as the driver decoded it.
                let mut converted = Vec::with_capacity(relocs.len());
                for (off, value) in relocs {
                    converted.push((*off, self.convert_const_value(value, None)?));
                }
                ir::Constant::ErasedAlloc {
                    bytes: bytes.clone(),
                    relocs: converted,
                }
            }
            mir::ConstValue::StaticRef {
                def,
                value,
                projection,
                len,
            } => {
                // The seed is the static's full initializer, typed
                // by the pointee context exactly as for a plain
                // reference constant; the reference itself names the
                // static's program-level storage, projected from the
                // static's base to the pointee.
                let inner_ctx = match ctx {
                    Some(mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _)) => {
                        Some(&**pointee)
                    }
                    _ => None,
                };
                let seed = self.convert_const_value(value, inner_ctx)?;
                // A static whose initializer is (partly)
                // uninitialized decodes to a bare `Uninit` seed
                // with no trace of the declared type's structure:
                // element writes into the storage then land in a
                // shapeless value and element reads fail as
                // uninitialized-place reads. The seed is the
                // static's full initializer whatever this reference
                // denotes -- the whole static or a sub-value inside
                // it -- and the first reference evaluated seeds the
                // storage every later access observes, so whenever
                // the driver recorded the declared type, expand
                // the seed to that type's skeleton exactly as the
                // thread-local table does for its seeds. A bare
                // seed kept for an interior reference (a pointer
                // to one element, a slice of one field) leaves the
                // storage shapeless: a write through the projection
                // collapses the projected slot to the written
                // scalar, and a later write to a sibling element
                // then fails against the scalar where the declared
                // aggregate belongs. The recorded type must also
                // win for a slice reference: the slice's length
                // bounds only the view it denotes, never the
                // storage, so a seed rebuilt from the length of a
                // prefix slice evaluated first would shrink the
                // storage to the slice's extent. Only a reference
                // with no recorded type falls back to the pointee
                // context -- the declared type itself for a thin
                // whole-static reference, and for a fat slice
                // reference the array type the slice covers,
                // rebuilt from the element type and the carried
                // length; an interior reference with no recorded
                // type keeps the bare form, as the full type is
                // unknown to this conversion.
                let (seed, seed_priority) = match self.static_tys.get(def) {
                    Some(ty) => {
                        let ty = ty.clone();
                        (self.expand_uninit_constant(&seed, &ty, 0), 2)
                    }
                    None if projection.is_empty() => {
                        let seed_ty = match (inner_ctx, len) {
                            (Some(ty), None) => Some(ty.clone()),
                            (Some(mir::Ty::Slice(elem)), Some(n)) => {
                                Some(mir::Ty::Array(
                                    elem.clone(),
                                    mir::ArrayLen::Known(*n),
                                ))
                            }
                            (Some(ty @ mir::Ty::Array(..)), Some(_)) => {
                                Some(ty.clone())
                            }
                            _ => None,
                        };
                        match seed_ty {
                            Some(ty) => {
                                let expanded =
                                    self.expand_uninit_constant(&seed, &ty, 0);
                                // A seed rebuilt to a slice
                                // reference's extent ranks with
                                // the bare seeds: it must not
                                // shrink the storage a
                                // whole-static seed would give.
                                (expanded, if len.is_none() { 1 } else { 0 })
                            }
                            None => (seed, 0),
                        }
                    }
                    None => (seed, 0),
                };
                // The seed is recorded once in the program's
                // seed table, not carried by this constant: a
                // table of references to one static otherwise
                // retains one (expanded) copy of the static's
                // initializer per element.
                let name = self.static_key(def);
                self.register_static_seed(&name, seed_priority, seed);
                ir::Constant::StaticRef {
                    name,
                    projection: projection
                        .iter()
                        .map(|e| self.convert_ref_proj(e))
                        .collect::<Result<_, _>>()?,
                    len: len.map(u128::from),
                }
            }
            mir::ConstValue::SliceVal(elems) => ir::Constant::SliceVal(
                elems
                    .iter()
                    .map(|e| {
                        // The slice value may sit directly under a
                        // reference context (a `&[u8]` constant), so
                        // unwrap references before matching the
                        // array/slice element type.
                        let mut inner_ctx = ctx;
                        while let Some(
                            mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _),
                        ) = inner_ctx
                        {
                            inner_ctx = Some(pointee);
                        }
                        let elem_ctx = match inner_ctx {
                            Some(mir::Ty::Array(elem, _) | mir::Ty::Slice(elem)) => {
                                Some(&**elem)
                            }
                            _ => None,
                        };
                        self.convert_const_value(e, elem_ctx)
                    })
                    .collect::<Result<_, _>>()?,
            ),
            mir::ConstValue::FnDef(def) => {
                ir::Constant::FnDef(self.body_key(def))
            }
            mir::ConstValue::Closure(def) => {
                ir::Constant::Closure(self.body_key(def), Vec::new())
            }
            mir::ConstValue::IntPtr(addr) => ir::Constant::IntPtr(*addr),
            mir::ConstValue::VTablePtr(id) => ir::Constant::VTablePtr(*id),
            mir::ConstValue::FatPtr { addr, meta } => ir::Constant::FatPtr {
                addr: *addr,
                meta: *meta,
            },
            mir::ConstValue::DynRef {
                base,
                concrete,
                traits,
            } => {
                // The data pointer constant converts as a
                // reference to the concrete type the vtable
                // names: the context a sized reference constant
                // of that type carries, so the pointee's scalar
                // leaves keep their signedness. The value carries
                // the concrete type's drop glue from here, like
                // a vtable's drop shim (see `UnsizeMeta::Dyn`).
                let base_ctx = mir::Ty::Ref(
                    mir::Region::Erased,
                    Box::new(concrete.clone()),
                    mir::Mutability::Not,
                );
                let base = self.convert_const_value(base, Some(&base_ctx))?;
                let glue = DropGlueBuilder::build(self, concrete);
                ir::Constant::DynRef {
                    base: Box::new(base),
                    concrete: self.concrete_ty_name(concrete),
                    traits: traits.clone(),
                    glue,
                }
            }
            mir::ConstValue::ByRef { .. } => {
                return Err(ConvertError::UnsupportedConstant {
                    kind: "by-reference constant".to_string(),
                })
            }
        })
    }

    fn convert_scalar(
        &self,
        scalar: &mir::Scalar,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Constant, ConvertError> {
        match scalar {
            mir::Scalar::Float { bits, ty } => match ty {
                mir::FloatTy::F16 => Ok(ir::Constant::Float(
                    f16_to_f64(*bits as u16),
                    ir::FloatTy::F16,
                )),
                mir::FloatTy::F32 => Ok(ir::Constant::Float(
                    crate::f128::f32_bits_to_f64_exact(*bits as u32),
                    ir::FloatTy::F32,
                )),
                mir::FloatTy::F64 => Ok(ir::Constant::Float(
                    f64::from_bits(*bits as u64),
                    ir::FloatTy::F64,
                )),
                mir::FloatTy::F128 => Ok(ir::Constant::Float128(*bits)),
            },
            mir::Scalar::Ptr { .. } => Err(ConvertError::UnsupportedConstant {
                kind: "pointer constant".to_string(),
            }),
            mir::Scalar::Int { data, size } => {
                let data = *data;
                Ok(match ctx {
                    Some(mir::Ty::Int(it)) => ir::Constant::Int(sign_extend(data, it.bits())),
                    Some(mir::Ty::Uint(_)) | Some(mir::Ty::Char) => ir::Constant::Uint(data),
                    Some(mir::Ty::Bool) => ir::Constant::Bool(data != 0),
                    // No context: use the scalar's own storage width. A
                    // set sign bit means a negative signed value, at
                    // every width: a 16-byte pattern with its top bit
                    // set is a negative `i128`, not `2^127` unsigned
                    // (the rule `literal_pair_ty` already applies to
                    // 128-bit literal pairs).
                    _ => {
                        let bits = (*size as u32) * 8;
                        if bits > 0 && (data >> (bits - 1)) & 1 == 1 {
                            ir::Constant::Int(sign_extend(data, bits))
                        } else {
                            ir::Constant::Uint(data)
                        }
                    }
                })
            }
        }
    }

    pub(super) fn convert_operand(
        &self,
        body: &mir::Body,
        op: &mir::Operand,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Operand, ConvertError> {
        Ok(match op {
            mir::Operand::Copy(p) => ir::Operand::Copy(self.convert_place(body, p)?),
            mir::Operand::Move(p) => {
                // The coroutine transform moves the `ResumeTy`
                // out of the state object's context cell at every
                // opaque-future poll site, then reads the cell
                // again for the next poll of the same activation:
                // the cell is state-machine storage (rewritten at
                // every state entry), not an owned value the move
                // vacates. `ResumeTy` is a pointer wrapper with no
                // drop glue of its own, so reading it without
                // vacating the cell is exact, and no other move in
                // the program has this shape.
                if self.is_resume_ty_place(body, p) {
                    ir::Operand::Copy(self.convert_place(body, p)?)
                } else {
                    ir::Operand::Move(self.convert_place(body, p)?)
                }
            }
            mir::Operand::Constant(c) => {
                // With no context from the operation, a constant is
                // read under its own recorded type when it has one:
                // the bits of a scalar literal carry no signedness,
                // and the sign-bit guess in `convert_scalar` reads
                // `255u8` as `-1i8`.
                let ctx = ctx.or(c.ty.as_ref());
                ir::Operand::Constant(self.convert_constant(c, ctx)?)
            }
        })
    }

    /// Convert the two operands of a binary operation under one shared
    /// typing context.
    ///
    /// In MIR both operands of a binary operation have the same type
    /// (shifts aside — their operands are typed separately by
    /// `convert_shift_operands`), and a literal
    /// operand carries no type of its own in `sparagmos-mir` — how its
    /// bits are interpreted must come from the operation's context:
    ///
    /// 1. A place operand pins the shared type: its declared type is
    ///    the operation's operand type.
    /// 2. Otherwise the result type gives it away. A checked operation
    ///    (a `CheckedBinaryOp`, or a `BinaryOp` with a `*WithOverflow`
    ///    operator) yields the `(T, bool)` pair, so the operand type is
    ///    the tuple's first element. A plain operation yields the
    ///    operand type directly — except comparisons, which yield
    ///    `bool` and say nothing about their operands. (Typing a
    ///    literal by the `bool` destination used to collapse every
    ///    nonzero literal to `true`.)
    /// 3. Otherwise — a comparison of two literals — the operation
    ///    itself declares no type. The pair still needs ONE
    ///    interpretation: guessing per literal can yield an
    ///    `Int`/`Uint` pair that cannot even be compared. A
    ///    literal's own recorded type (both literals share one type
    ///    in MIR) is that interpretation when either side carries
    ///    one; the sign-bit reading of [`literal_pair_ty`] remains
    ///    for literals with no recorded type (the shape of rustc's
    ///    expanded Div/Rem checks: `Eq(rhs, 0)`, `Eq(rhs, -1)`,
    ///    `Eq(lhs, MIN)` over two literals).
    pub(super) fn convert_binop_operands(
        &self,
        body: &mir::Body,
        pair: &(mir::Operand, mir::Operand),
        dest_ty: Option<&mir::Ty>,
        checked_result: bool,
        comparison: bool,
    ) -> Result<(ir::Operand, ir::Operand), ConvertError> {
        let ctx = self
            .operand_place_ty(body, &pair.0)
            .or_else(|| self.operand_place_ty(body, &pair.1))
            .or_else(|| {
                if checked_result {
                    match dest_ty {
                        Some(mir::Ty::Tuple(ts)) => ts.first().cloned(),
                        _ => None,
                    }
                } else if comparison {
                    None
                } else {
                    dest_ty.cloned()
                }
            })
            .or_else(|| constant_recorded_ty(&pair.0).or_else(|| constant_recorded_ty(&pair.1)))
            .or_else(|| literal_pair_ty(&pair.0, &pair.1));
        let lhs = self.convert_operand(body, &pair.0, ctx.as_ref())?;
        let rhs = self.convert_operand(body, &pair.1, ctx.as_ref())?;
        Ok((lhs, rhs))
    }

    /// Convert the two operands of a shift (`Shl` / `Shr`, plain,
    /// unchecked or checked).
    ///
    /// A shift is the one binary operation whose operands do not
    /// share a type in MIR: the amount may have any integer type.
    /// The shared-context conversion would mistype both sides — a
    /// constant value read under the amount's type (when the amount
    /// is a place), or a constant amount read under the value's
    /// type, truncating it to the value's width and sign — so the
    /// operands are typed separately:
    ///
    /// * The value takes the operation's own context: its place
    ///   type, else the result type (a checked shift's `(T, bool)`
    ///   result gives `T`).
    /// * The amount keeps its own type. A place amount's runtime
    ///   value already carries it, and a constant amount is
    ///   converted with no context, which reads it in its own
    ///   storage width (see `convert_scalar`). That reading is
    ///   exact whenever the amount fits the value's width; an
    ///   amount that does not fit — wider than the value, negative,
    ///   or an unsigned amount whose top bit is set, read as a
    ///   negative — is an overflowing shift under either reading,
    ///   which execution reports the same way (see `eval_shift`).
    pub(super) fn convert_shift_operands(
        &self,
        body: &mir::Body,
        pair: &(mir::Operand, mir::Operand),
        dest_ty: Option<&mir::Ty>,
        checked_result: bool,
    ) -> Result<(ir::Operand, ir::Operand), ConvertError> {
        let value_ctx = self.operand_place_ty(body, &pair.0).or_else(|| {
            if checked_result {
                match dest_ty {
                    Some(mir::Ty::Tuple(ts)) => ts.first().cloned(),
                    _ => None,
                }
            } else {
                dest_ty.cloned()
            }
        });
        let lhs = self.convert_operand(body, &pair.0, value_ctx.as_ref())?;
        let rhs = self.convert_operand(body, &pair.1, None)?;
        Ok((lhs, rhs))
    }

}
