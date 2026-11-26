//! Value equality and comparison for the equality builtins.

use std::borrow::Cow;
use std::cmp::Ordering;


use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter,
};
use crate::ir::Ty;
use crate::value::{RefTarget, Value};

use super::*;


/// One operand of an equality or comparison builtin, with a
/// one-field wrapper form around a scalar (the `MaybeUninit`
/// union form a bulk copy can leave in an element slot)
/// replaced by the scalar it holds, when the other operand
/// is a bare scalar: the comparison is over the element
/// type, and only one side's storage carries the wrapper
/// form. Any other shape passes through unchanged.
fn unwrap_scalar_form<'x>(v: Cow<'x, Value>, other: &Value) -> Cow<'x, Value> {
    fn is_scalar(v: &Value) -> bool {
        matches!(
            v,
            Value::Uint(_)
                | Value::Int(_)
                | Value::Float(_)
                | Value::Float128(_)
                | Value::Bool(_)
        )
    }
    if !is_scalar(other) {
        return v;
    }
    match &*v {
        Value::Enum { variant: 0, fields }
            if fields.len() == 1 && is_scalar(&fields[0]) =>
        {
            Cow::Owned(fields[0].clone())
        }
        _ => v,
    }
}


/// Whether a comparison call's own type tags name a thin raw
/// pointer as the comparison's subject: the `Self` tag (the
/// first generic argument), with one leading reference layer
/// stripped, opens with `*const ` / `*mut `. Core's cross
/// impls put the raw pointer on the `Self` side (a
/// `&T == *const T` comparison has no impl), and the `Rhs`
/// tag is not consulted. A fat pointee (a slice, `str`, a
/// `dyn` type) makes the pointer an (address, metadata)
/// pair whose native equality also weighs the metadata, so
/// those keep the ordinary path; a sized array pointee
/// (`[T; N]`) is thin, like every other sized pointee. A raw
/// pointer nested inside an aggregate tag
/// (`Option<*const T>`) is a field comparison, which the
/// structural recursion in `values_equal` judges, not this
/// dispatch.
fn comparison_subject_is_raw_pointer(generic_args: &[String]) -> bool {
    let Some(tag) = generic_args.first() else {
        return false;
    };
    let tag = tag
        .strip_prefix("&mut ")
        .or_else(|| tag.strip_prefix('&'))
        .unwrap_or(tag);
    let Some(pointee) = tag
        .strip_prefix("*const ")
        .or_else(|| tag.strip_prefix("*mut "))
    else {
        return false;
    };
    !(pointee.starts_with("dyn ")
        || pointee == "str"
        || (pointee.starts_with('[') && !pointee.contains(';')))
}

impl<'a> Interpreter<'a> {
    /// The element lists of a slice comparison's two
    /// operands when a side's fat pointer regroups its
    /// storage's elements under the element shape the
    /// comparison's own `Self` type names (a `[[u8; 3]; 2]`
    /// storage ranged over as `[[u8; 2]]`): the regrouped
    /// side's elements are the byte view's construction
    /// (`foreign_slice_elements`), the other side's are the
    /// ordinary element reading's. `None` when the `Self`
    /// type is not a slice or neither side is such a view,
    /// in which case the ordinary comparison judges.
    fn regrouped_slice_elems(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
    ) -> Option<InterpResult<(Vec<Value>, Vec<Value>)>> {
        let self_ty = ty_from_tag(generic_args.first()?)?;
        let Ty::Slice(elem_ty) = &self_ty else {
            return None;
        };
        let ea = self.foreign_slice_elements(frames, a, elem_ty);
        let eb = self.foreign_slice_elements(frames, b, elem_ty);
        if ea.is_none() && eb.is_none() {
            return None;
        }
        Some((|| {
            let ea = match ea {
                Some(elems) => elems,
                None => self.slice_elements(frames, a)?.0,
            };
            let eb = match eb {
                Some(elems) => elems,
                None => self.slice_elements(frames, b)?.0,
            };
            Ok((ea, eb))
        })())
    }

    /// Slice equality under a regrouping view, when one
    /// applies (see `regrouped_slice_elems`).
    pub(crate) fn regrouped_slice_equal(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
    ) -> Option<InterpResult<bool>> {
        let elems = self.regrouped_slice_elems(frames, generic_args, a, b)?;
        Some((|| {
            let (ea, eb) = elems?;
            if ea.len() != eb.len() {
                return Ok(false);
            }
            self.seq_equal(frames, &ea, &eb)
        })())
    }

    /// Slice comparison under a regrouping view, when one
    /// applies (see `regrouped_slice_elems`): element-wise
    /// with the first non-`Equal` element deciding, then by
    /// length, exactly as the ordinary slice arm of
    /// `values_cmp`.
    pub(crate) fn regrouped_slice_cmp(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
        discriminants: Option<&[i128]>,
    ) -> Option<InterpResult<Option<Ordering>>> {
        let elems = self.regrouped_slice_elems(frames, generic_args, a, b)?;
        Some((|| {
            let (ea, eb) = elems?;
            for (x, y) in ea.iter().zip(eb.iter()) {
                match self.values_cmp(frames, x, y, discriminants)? {
                    Some(Ordering::Equal) => {}
                    other => return Ok(other),
                }
            }
            Ok(Some(ea.len().cmp(&eb.len())))
        })())
    }

    /// The address one operand of a raw-pointer comparison
    /// denotes: the call passes a reference to the compared
    /// value, so one dereference layer reaches the pointer
    /// value itself, and `pointer_address` prices it in the
    /// same model every other pointer identity uses. `None`
    /// when the layer cannot be read or no address can be
    /// derived, leaving the ordinary comparison in charge.
    fn raw_pointer_operand_address(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> Option<u128> {
        let compared = match value {
            Value::Ref(_) | Value::FatRef { .. } => {
                let target = self.deref_value(frames, value).ok()?;
                self.read_resolved(frames, &target).ok()?
            }
            _ => value.clone(),
        };
        self.pointer_address(frames, &compared, None)
    }

    /// Raw-pointer equality under the type's own `PartialEq`
    /// (see `comparison_subject_is_raw_pointer`): the native
    /// impls compare addresses, never the pointees the
    /// structural comparison in `values_equal` would resolve
    /// to -- which also makes a self-referential pointee (a
    /// stored pointer denoting the very aggregate that holds
    /// it) recurse without bound there. `None` when the
    /// comparison is not a thin raw-pointer one or an
    /// operand's address cannot be derived.
    pub(crate) fn raw_pointer_equal(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
    ) -> Option<bool> {
        if !comparison_subject_is_raw_pointer(generic_args) {
            return None;
        }
        let x = self.raw_pointer_operand_address(frames, a)?;
        let y = self.raw_pointer_operand_address(frames, b)?;
        Some(x == y)
    }

    /// Raw-pointer ordering, the `PartialOrd` / `Ord`
    /// counterpart of `raw_pointer_equal`: addresses order
    /// as integers.
    pub(crate) fn raw_pointer_cmp(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
    ) -> Option<Option<Ordering>> {
        if !comparison_subject_is_raw_pointer(generic_args) {
            return None;
        }
        let x = self.raw_pointer_operand_address(frames, a)?;
        let y = self.raw_pointer_operand_address(frames, b)?;
        Some(Some(x.cmp(&y)))
    }

    pub(crate) fn values_equal(
        &self,
        frames: &[Frame],
        a: &Value,
        b: &Value,
    ) -> InterpResult<bool> {
        if matches!(a, Value::FatRef { .. }) || matches!(b, Value::FatRef { .. }) {
            let (ea, la) = self.slice_elements_ref(frames, a)?;
            let (eb, lb) = self.slice_elements_ref(frames, b)?;
            if la != lb {
                return Ok(false);
            }
            return self.elems_equal(frames, &ea, &eb);
        }
        // An address-only pointer (the value
        // `simd_with_exposed_provenance` and pointer arithmetic
        // produce) denotes no place in the current frames, so
        // there is no pointee to resolve to: it compares by
        // address against the other side's synthetic address,
        // the identity every pointer value carries in this
        // model.
        if matches!(a, Value::Ref(RefTarget::Address(_)))
            || matches!(b, Value::Ref(RefTarget::Address(_)))
        {
            if let (Some(x), Some(y)) = (
                self.pointer_address(frames, a, None),
                self.pointer_address(frames, b, None),
            ) {
                return Ok(x == y);
            }
        }
        let a = self.resolve_ref(frames, a)?;
        let b = self.resolve_ref(frames, b)?;
        let a = unwrap_scalar_form(a, &b);
        let b = unwrap_scalar_form(b, &a);
        // A `VecDeque` value (the real aggregate: wrapped head,
        // length, ring buffer) compares by its logical element
        // sequence, as the type's own `PartialEq` does: the
        // physical buffer layout -- where the head sits, the
        // capacity, the buffer pointer's projection form -- is
        // not part of the value's identity, so two deques with
        // the same elements in the same order are equal however
        // their buffers happen to be arranged.
        if let (Some(ea), Some(eb)) = (
            self.vec_deque_elements(a.as_ref())?,
            self.vec_deque_elements(b.as_ref())?,
        ) {
            return Ok(ea.len() == eb.len() && self.seq_equal(frames, &ea, &eb)?);
        }
        // A vector value in its sparse form (declared length
        // below the stored elements: the spare capacity a
        // shrink or a regrowth leaves behind) compares by its
        // logical element sequence, as the type's own
        // `PartialEq` does: the storage past the length is not
        // part of the value's identity, so a sparse vector and
        // the plain aggregate form of the same elements are
        // equal. A sparse vector whose length runs past its
        // stored elements has no element list to compare and
        // keeps the structural comparison below.
        if matches!(&*a, Value::SparseVec { .. })
            || matches!(&*b, Value::SparseVec { .. })
        {
            if let (Ok(ea), Ok(eb)) = (value_elements(&a), value_elements(&b)) {
                return Ok(
                    ea.len() == eb.len() && self.seq_equal(frames, &ea, &eb)?
                );
            }
        }
        // A `dyn Trait` value denotes its concrete pointee: two
        // trait objects compare equal when their concrete types
        // agree and the pointees compare equal, and unequal when
        // the concrete types differ -- the semantics a hand-written
        // `PartialEq for dyn Trait` computes. For the primitive
        // `PartialEq` supertrait comparisons (the trait-upcasting
        // tests, whose concrete type is `i32`) the structural
        // comparison of the pointee is the concrete type's own
        // `PartialEq`; for a user-defined `PartialEq` the builtin's
        // structural evaluation is the same policy the model applies
        // to direct calls. A dyn side against a sized side compares
        // the pointee against that side.
        if matches!(&*a, Value::DynRef { .. }) || matches!(&*b, Value::DynRef { .. }) {
            if let (
                Value::DynRef { concrete: ca, .. },
                Value::DynRef { concrete: cb, .. },
            ) = (&*a, &*b)
            {
                if ca != cb {
                    return Ok(false);
                }
            }
            let pa = self.dyn_pointee(frames, &a)?;
            let pb = self.dyn_pointee(frames, &b)?;
            return self.values_equal(frames, &pa, &pb);
        }
        // A side that resolves to a self-contained fat value (a
        // slice constant or materialised sub-slice held in a local)
        // compares as a slice against the other side's elements:
        // the other side may resolve to the bare array value, whose
        // elements are exactly its own.
        if matches!(&*a, Value::FatRef { .. }) || matches!(&*b, Value::FatRef { .. }) {
            let (ea, la) = self.slice_elements_ref(frames, &a)?;
            let (eb, lb) = self.slice_elements_ref(frames, &b)?;
            if la != lb {
                return Ok(false);
            }
            return match (&ea, &eb) {
                (SliceElems::Values(x), SliceElems::Values(y)) => {
                    self.seq_equal(frames, x, y)
                }
                _ => self.elems_equal(frames, &ea, &eb),
            };
        }
        // A `String` in its real aggregate form denotes the
        // string its buffer holds (see
        // `string_from_real_aggregate`): against a string side
        // it compares as that string.
        for (side, other) in [(&*a, &*b), (&*b, &*a)] {
            if let Some(text) = self.string_from_real_aggregate(frames, side)? {
                return Ok(match other {
                    Value::Str(y) => text == *y,
                    Value::Bytes(y) => text.as_bytes() == y.as_slice(),
                    _ => false,
                });
            }
        }
        // The same denotation for a `String` whose vector is
        // in its element form. Unlike the aggregate loop
        // above this never decides against a side that yields
        // no string: two element-form values (an `OsString`
        // has the same shape) keep the structural comparison
        // below unless both sides yield their bytes. The
        // element form may sit under the one-field `String`
        // wrapper or *be* the stored value itself (the shape
        // `String::as_mut_vec` leaves in the string's
        // storage); `string_from_element_form` reads both,
        // the sparse form included.
        for (side, other) in [(&*a, &*b), (&*b, &*a)] {
            let text = self.string_from_element_form(side);
            if let Some(text) = text {
                match other {
                    Value::Str(y) => return Ok(text == *y),
                    Value::Bytes(y) => {
                        return Ok(text.as_bytes() == y.as_slice())
                    }
                    Value::Enum { variant: 0, .. } => {
                        if let Some(other_text) =
                            self.string_from_real_aggregate(frames, other)?
                                .or_else(|| {
                                    self.string_from_element_form(other)
                                })
                        {
                            return Ok(text == other_text);
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(match (&*a, &*b) {
            (Value::Ref(RefTarget::Null), Value::Ref(RefTarget::Null)) => true,
            // Address-only pointers compare by address; address 0
            // is the null address.
            (Value::Ref(RefTarget::Address(x)), Value::Ref(RefTarget::Address(y))) => {
                x == y
            }
            (Value::Ref(RefTarget::Address(0)), Value::Ref(RefTarget::Null))
            | (Value::Ref(RefTarget::Null), Value::Ref(RefTarget::Address(0))) => true,
            (Value::Ref(RefTarget::Null), _) | (_, Value::Ref(RefTarget::Null)) => false,
            (Value::Unit, Value::Unit) => true,
            (Value::Bool(x), Value::Bool(y)) => x == y,
            (Value::Int(x), Value::Int(y)) => x == y,
            (Value::Uint(x), Value::Uint(y)) => x == y,
            (Value::Int(x), Value::Uint(y)) => int_uint_eq(*x, *y),
            (Value::Uint(x), Value::Int(y)) => int_uint_eq(*y, *x),
            (Value::Str(x), Value::Str(y)) => x == y,
            (Value::Bytes(x), Value::Bytes(y)) => x == y,
            // Float equality is IEEE equality, as derived
            // `PartialEq` on a float field performs it (NaN compares
            // unequal to itself).
            (Value::Float(x), Value::Float(y)) => x == y,
            (Value::Float128(x), Value::Float128(y)) => crate::f128::eq(*x, *y),
            (Value::FnPtr(x), Value::FnPtr(y)) => x == y,
            (Value::Tuple(x), Value::Tuple(y)) => {
                x.len() == y.len() && self.seq_equal(frames, x, y)?
            }
            // A byte string and an array of byte values are the
            // same data in two of the model's shapes (a `[u8; N]`
            // constant decodes to bytes; the same array built by an
            // aggregate is a tuple).
            (Value::Bytes(x), Value::Tuple(y)) | (Value::Tuple(y), Value::Bytes(x)) => {
                x.len() == y.len()
                    && x.iter().zip(y.iter()).all(|(byte, elem)| {
                        matches!(elem, Value::Uint(v) if *v == u128::from(*byte))
                            || matches!(elem, Value::Int(v) if *v == i128::from(*byte))
                    })
            }
            // A byte string and an element-list aggregate of
            // byte values: the two shapes of one byte sequence
            // (a `Vec<u8>` the process model builds from a
            // child's captured output is the element list; a
            // byte-string constant is the bytes).
            (Value::Enum { fields: x, .. }, Value::Bytes(y))
            | (Value::Bytes(y), Value::Enum { fields: x, .. })
                if x.iter()
                    .all(|e| matches!(e, Value::Uint(_) | Value::Int(_))) =>
            {
                x.len() == y.len()
                    && x.iter().zip(y.iter()).all(|(elem, byte)| {
                        matches!(elem, Value::Uint(v) if *v == u128::from(*byte))
                            || matches!(elem, Value::Int(v) if *v == i128::from(*byte))
                    })
            }
            // A byte string against an ADT value whose single
            // field is a sequence: the holder is the sequence it
            // holds -- a vector read through an array-typed
            // pointer (`simd.as_array()`) against a byte-array
            // literal is the array against the bytes, not a
            // one-element tuple of it.
            (Value::Enum { variant: 0, fields: x }, Value::Bytes(_))
            | (Value::Bytes(_), Value::Enum { variant: 0, fields: x })
                if x.len() == 1
                    && matches!(x[0], Value::Tuple(_) | Value::Bytes(_)) =>
            {
                let field = x[0].clone();
                let bytes_side = if matches!(&*a, Value::Bytes(_)) {
                    &*a
                } else {
                    &*b
                };
                return self.values_equal(frames, &field, bytes_side);
            }
            // A byte string against a vector in its element
            // form: the ADT value's fields ARE the elements
            // (the shape a `Vec<u8>` built element by element
            // takes, see `string_from_element_form`), so the
            // field list itself is the byte sequence -- unlike
            // the holder above, whose single field holds it.
            (Value::Enum { variant: 0, fields: x }, Value::Bytes(y))
            | (Value::Bytes(y), Value::Enum { variant: 0, fields: x })
                if x.iter().all(|f| Self::element_form_byte(f).is_some()) =>
            {
                x.len() == y.len()
                    && x.iter()
                        .zip(y.iter())
                        .all(|(elem, byte)| Self::element_form_byte(elem) == Some(*byte))
            }
            // The same holder against a tuple, when the tuple
            // cannot be the holder's field list (its length
            // differs from the field count): it is the held
            // sequence's own tuple form.
            (Value::Enum { variant: 0, fields: x }, Value::Tuple(y))
            | (Value::Tuple(y), Value::Enum { variant: 0, fields: x })
                if x.len() == 1
                    && y.len() != 1
                    && matches!(x[0], Value::Tuple(_) | Value::Bytes(_)) =>
            {
                let field = x[0].clone();
                let tuple_side = if matches!(&*a, Value::Tuple(_)) {
                    &*a
                } else {
                    &*b
                };
                return self.values_equal(frames, &field, tuple_side);
            }
            // A single-variant ADT value (a tuple struct) and a
            // plain tuple of the same fields: the constant decoder
            // and aggregate construction produce the two shapes for
            // the same source value.
            (
                Value::Enum { variant: 0, fields: x },
                Value::Tuple(y),
            )
            | (
                Value::Tuple(y),
                Value::Enum { variant: 0, fields: x },
            ) => x.len() == y.len() && self.seq_equal(frames, x, y)?,
            (
                Value::Enum {
                    variant: va,
                    fields: fa,
                },
                Value::Enum {
                    variant: vb,
                    fields: fb,
                },
            ) => va == vb && fa.len() == fb.len() && self.seq_equal(frames, fa, fb)?,
            _ => {
                return Err(InterpError::type_mismatch(
                    "value equality",
                    "comparable values",
                    format!("{} and {}", a.kind_name(), b.kind_name()),
                ))
            }
        })
    }

    /// The concrete value a `dyn Trait` value denotes: the
    /// pointee of its data pointer. A box-shaped base (the
    /// one-field handle aggregate a `Box` / `Rc` / `Arc` coercion
    /// records, the same judgment the dereference of a dyn fat
    /// pointer makes) holds the pointee as its only field; a
    /// reference or slice base resolves to the value it points at.
    /// A non-dyn value denotes itself. An upcast chain (a box
    /// whose field is itself a `DynRef`) yields that `DynRef`,
    /// which the caller's recursion unwraps one layer at a time.
    fn dyn_pointee(&self, frames: &[Frame], value: &Value) -> InterpResult<Value> {
        match value {
            Value::DynRef { base, .. } => {
                // A reference-counted handle's pointee lives in
                // the shared allocation, not in the handle: the
                // handle aggregate is the pointer chain plus
                // metadata, and the value is field 2 of the
                // RcInner / ArcInner aggregate the allocation
                // holds ({strong, weak, value}). The store check
                // is the deciding evidence -- a Box handle has
                // the same shape, but its allocation holds the
                // value itself (see `rc_inner_id`).
                if let Some(id) = self.rc_inner_id(base) {
                    if let Ok(Value::Enum { fields, .. }) = self.alloc_base(id) {
                        if let Some(pointee) = fields.get(2) {
                            return Ok(pointee.clone());
                        }
                    }
                }
                match &**base {
                    Value::Enum { fields, .. } if fields.len() == 1 => {
                        Ok(fields[0].clone())
                    }
                    _ => self.resolve_value(frames, base),
                }
            }
            other => Ok(other.clone()),
        }
    }

    fn seq_equal(&self, frames: &[Frame], a: &[Value], b: &[Value]) -> InterpResult<bool> {
        for (x, y) in a.iter().zip(b.iter()) {
            if !self.values_equal(frames, x, y)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Element-wise equality of two borrowed element lists: the
    /// slice arm of `values_equal` without materialising either
    /// side's elements, so a comparison that decides early does not
    /// pay for a snapshot of the rest.
    fn elems_equal(
        &self,
        frames: &[Frame],
        a: &SliceElems<'_>,
        b: &SliceElems<'_>,
    ) -> InterpResult<bool> {
        for i in 0..a.len().min(b.len()) {
            let (x, y) = (a.get(i), b.get(i));
            if !self.values_equal(frames, &x, &y)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Semantic (derived `Ord`) comparison of two values: integers
    /// and booleans by value, strings byte-wise (as `str`'s `Ord`
    /// is), aggregates lexicographically with enum variants ordered
    /// by declaration order — the derived semantics.
/// The byte an element-form field denotes, when it denotes
/// one: a byte-valued integer, possibly holding the
/// `MaybeUninit` union form (a one-field variant-0 wrapper)
/// the buffer's spare-capacity region stores. Any other
/// shape denotes no byte.
fn element_form_byte(elem: &Value) -> Option<u8> {
    let elem = match elem {
        Value::Enum {
            variant: 0, fields, ..
        } if fields.len() == 1 => &fields[0],
        other => other,
    };
    match elem {
        Value::Uint(b) if *b <= 0xFF => Some(*b as u8),
        Value::Int(b) if (0..=255).contains(b) => Some(*b as u8),
        _ => None,
    }
}

/// The discriminant value a variant index stands for in a
/// structural comparison: the baked table's entry when the
/// comparison's `Self` type has one, else the index itself (the
/// sequential default, where the two agree).
fn cmp_discriminant(discriminants: Option<&[i128]>, variant: u32) -> i128 {
    discriminants
        .and_then(|table| table.get(variant as usize).copied())
        .unwrap_or(variant as i128)
}

    /// The comparison behind the `PartialOrd` / `Ord` builtins,
    /// with `partial_cmp` semantics: the result is `None` when the
    /// values are unordered (a float comparison involving NaN,
    /// propagated out of the first field or element whose comparison
    /// is unordered, exactly as a derived `partial_cmp` returns the
    /// first non-`Equal` field result). Callers that need a total
    /// ordering treat `None` as an error at their own site.
    pub(crate) fn values_cmp(
        &self,
        frames: &[Frame],
        a: &Value,
        b: &Value,
        discriminants: Option<&[i128]>,
    ) -> InterpResult<Option<Ordering>> {
        if matches!(a, Value::FatRef { .. }) || matches!(b, Value::FatRef { .. }) {
            let (ea, la) = self.slice_elements_ref(frames, a)?;
            let (eb, lb) = self.slice_elements_ref(frames, b)?;
            match self.elems_cmp(frames, &ea, &eb)? {
                Some(Ordering::Equal) => {}
                other => return Ok(other),
            }
            return Ok(Some(la.cmp(&lb)));
        }
        let a = self.resolve_ref(frames, a)?;
        let b = self.resolve_ref(frames, b)?;
        let a = unwrap_scalar_form(a, &b);
        let b = unwrap_scalar_form(b, &a);
        // A vector value in its sparse form (declared length
        // below the stored elements: the spare capacity a
        // growth or a shrink leaves behind) orders by its
        // logical element sequence, as the type's own `Ord`
        // does: the storage past the length is not part of the
        // value, so a sparse vector and the plain aggregate
        // form of the same elements order as those elements.
        // A sparse vector whose length runs past its stored
        // elements has no element list to compare and keeps
        // the structural comparison below.
        if matches!(&*a, Value::SparseVec { .. })
            || matches!(&*b, Value::SparseVec { .. })
        {
            if let (Ok(ea), Ok(eb)) = (value_elements(&a), value_elements(&b)) {
                return self.seq_cmp(frames, &ea, &eb);
            }
        }
        // A `String` in its real aggregate form denotes the
        // string its buffer holds (see
        // `string_from_real_aggregate`): against a string side
        // it orders as that string.
        for (side, other, flip) in [(&*a, &*b, false), (&*b, &*a, true)] {
            // The element form of the same value (the shape
            // the return-boundary translation produces) reads
            // the same way, but only against a string side:
            // the shortcut never decides a comparison between
            // two aggregate-shaped values.
            let text = match self.string_from_real_aggregate(frames, side)? {
                Some(text) => Some(text),
                // The element form, under the one-field
                // wrapper or as the stored value itself (see
                // `values_equal`): `string_from_element_form`
                // reads both, the sparse form included.
                None => self.string_from_element_form(side),
            };
            if let Some(text) = text {
                let ord = match other {
                    Value::Str(y) => text.as_bytes().cmp(y.as_bytes()),
                    Value::Bytes(y) => text.as_bytes().cmp(y.as_slice()),
                    _ => continue,
                };
                return Ok(Some(if flip { ord.reverse() } else { ord }));
            }
        }
        Ok(match (&*a, &*b) {
            (Value::Unit, Value::Unit) => Some(Ordering::Equal),
            (Value::Bool(x), Value::Bool(y)) => Some(x.cmp(y)),
            (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
            (Value::Uint(x), Value::Uint(y)) => Some(x.cmp(y)),
            (Value::Int(x), Value::Uint(y)) => Some(int_uint_cmp(*x, *y)),
            (Value::Uint(x), Value::Int(y)) => Some(int_uint_cmp(*y, *x).reverse()),
            (Value::Float(x), Value::Float(y)) => x.partial_cmp(y),
            (Value::Str(x), Value::Str(y)) => Some(x.as_bytes().cmp(y.as_bytes())),
            (Value::Bytes(x), Value::Bytes(y)) => Some(x.cmp(y)),
            (Value::Tuple(x), Value::Tuple(y)) => self.seq_cmp(frames, x, y)?,
            (
                Value::Enum {
                    variant: va,
                    fields: fa,
                },
                Value::Enum {
                    variant: vb,
                    fields: fb,
                },
            ) => match Self::cmp_discriminant(discriminants, *va)
                .cmp(&Self::cmp_discriminant(discriminants, *vb))
            {
                Ordering::Equal => self.seq_cmp(frames, fa, fb)?,
                other => Some(other),
            },
            _ => {
                return Err(InterpError::type_mismatch(
                    "value comparison",
                    "comparable values",
                    format!("{} and {}", a.kind_name(), b.kind_name()),
                ))
            }
        })
    }

    fn seq_cmp(
        &self,
        frames: &[Frame],
        a: &[Value],
        b: &[Value],
    ) -> InterpResult<Option<Ordering>> {
        for (x, y) in a.iter().zip(b.iter()) {
            match self.values_cmp(frames, x, y, None)? {
                Some(Ordering::Equal) => {}
                other => return Ok(other),
            }
        }
        Ok(Some(a.len().cmp(&b.len())))
    }

    /// Element-wise comparison of two borrowed element lists: the
    /// slice arm of `values_cmp` without materialising either
    /// side's elements. The result is the first non-`Equal` element
    /// result, or `Equal` when the common prefix compares equal
    /// (the caller applies the length tiebreak).
    fn elems_cmp(
        &self,
        frames: &[Frame],
        a: &SliceElems<'_>,
        b: &SliceElems<'_>,
    ) -> InterpResult<Option<Ordering>> {
        for i in 0..a.len().min(b.len()) {
            let (x, y) = (a.get(i), b.get(i));
            match self.values_cmp(frames, &x, &y, None)? {
                Some(Ordering::Equal) => {}
                other => return Ok(other),
            }
        }
        Ok(Some(Ordering::Equal))
    }
}

/// Registration fragment of the builtin name registry: the comparison trait surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn equality_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        // Comparison traits: the impls for the modelled data types
        // are the derived ones, i.e. semantic value comparison.
        _ if canonical.ends_with("::PartialEq::eq") => Builtin::PartialEqEq,
        _ if canonical.ends_with("::PartialEq::ne") => Builtin::PartialEqNe,
        _ if canonical.ends_with("::PartialOrd::partial_cmp") => Builtin::PartialCmp,
        _ if canonical.ends_with("::PartialOrd::lt") => Builtin::PartialOrdRel(RelOp::Lt),
        _ if canonical.ends_with("::PartialOrd::le") => Builtin::PartialOrdRel(RelOp::Le),
        _ if canonical.ends_with("::PartialOrd::gt") => Builtin::PartialOrdRel(RelOp::Gt),
        _ if canonical.ends_with("::PartialOrd::ge") => Builtin::PartialOrdRel(RelOp::Ge),
        _ if canonical.ends_with("::Ord::cmp") => Builtin::OrdCmp,
        _ if canonical.ends_with("::Clone::clone") => Builtin::CloneClone,
        _ => return None,
    })
}
