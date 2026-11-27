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
/// The element tag a scalar comparison under a punned view
/// is judged by: the compared type the call's generic
/// arguments name, with one reference layer removed (a
/// comparison of references themselves -- the derived
/// `PartialEq` of a tuple holding the element reference,
/// above all -- names the reference type, while the
/// operands the builtin receives are the references behind
/// it). A non-reference tag passes through unchanged.
/// The scalar type tag a tuple field type names, through
/// one reference layer (the element tag a punned field is
/// judged by, in the spelling `scalar_shape_of_tag`
/// reads). `None` for a field type that is not a scalar or
/// a reference to one.
fn scalar_tag_of_ty(ty: &Ty) -> Option<String> {
    match ty {
        Ty::Ref(inner) => scalar_tag_of_ty(inner),
        Ty::Bool => Some("bool".to_string()),
        Ty::Int(t) => Some(format!("{t:?}").to_lowercase()),
        Ty::Uint(t) => Some(format!("{t:?}").to_lowercase()),
        Ty::Float(t) => Some(format!("{t:?}").to_lowercase()),
        _ => None,
    }
}

fn punned_scalar_tag(tag: &str) -> &str {
    let tag = tag.trim();
    match tag.strip_prefix('&') {
        Some(rest) => {
            let rest = rest.trim_start();
            rest.strip_prefix("mut ").unwrap_or(rest).trim_start()
        }
        None => tag,
    }
}

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


/// Whether a value is a reference over a byte-form value with
/// no projection: the snapshot form a byte-sized pointee's
/// constant reference takes. The driver's pointer decode
/// denotes such a pointee (a `u8` / `i8` / `bool` whose
/// allocation extends past it -- an element of a constant
/// array, above all) by the allocation's bytes from the
/// pointee on, so the scalar the reference denotes is the
/// byte string's first byte and the remaining bytes are its
/// siblings, not part of the value. A wider scalar's
/// reference decodes to the scalar itself, never to this
/// form, so the first byte is the whole story here.
fn is_byte_tail_ref(v: &Value) -> bool {
    matches!(
        v,
        Value::Ref(RefTarget::Value(base, proj))
            if proj.is_empty() && matches!(&**base, Value::Bytes(_))
    )
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
/// Whether a value has the shape an element type names,
/// for the one judgement `box_slice_elements` makes: the
/// coarse form only (a scalar against a scalar type, an
/// aggregate against an aggregate type, an array's tuple
/// or byte form at its length). Types whose shape this
/// does not judge answer true, keeping the ordinary
/// reading.
fn elem_shape_matches(value: &Value, elem_ty: &Ty) -> bool {
    match elem_ty {
        Ty::Bool => matches!(value, Value::Bool(_)),
        Ty::Int(_) | Ty::Uint(_) => {
            matches!(value, Value::Int(_) | Value::Uint(_))
        }
        Ty::Float(_) => matches!(value, Value::Float(_) | Value::Float128(_)),
        Ty::Array(_, n) => match value {
            Value::Tuple(items) => items.len() as u64 == *n,
            Value::Bytes(b) => b.len() as u64 == *n,
            _ => false,
        },
        Ty::Slice(_) | Ty::Str => matches!(
            value,
            Value::Tuple(_)
                | Value::Bytes(_)
                | Value::Str(_)
                | Value::FatRef { .. }
                | Value::SparseVec { .. }
        ),
        Ty::Ref(_) => matches!(
            value,
            Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }
        ),
        Ty::Adt(_) | Ty::Tuple(_) => {
            matches!(value, Value::Enum { .. } | Value::Tuple(_))
        }
        _ => true,
    }
}

/// Whether a resolved value is a bare pointee snapshot:
/// the form a box's pointee takes when a consumer (a sort,
/// above all) extracted it from the box -- a byte string, a
/// string, a scalar, or an element sequence -- as opposed
/// to an aggregate that might itself be a stored box or
/// another modelled type's own form.
fn is_bare_pointee_form(v: &Value) -> bool {
    matches!(
        v,
        Value::Bytes(_)
            | Value::Str(_)
            | Value::Tuple(_)
            | Value::SparseVec { .. }
            | Value::Uint(_)
            | Value::Int(_)
            | Value::Float(_)
            | Value::Float128(_)
            | Value::Bool(_)
            | Value::Unit
    )
}

/// The pointee tag a `Box` type tag names: the first
/// generic argument inside the `Box<..>` (the allocator
/// argument follows it). `None` when the tag is not a box
/// tag.
fn box_pointee_tag(tag: &str) -> Option<&str> {
    let inner = tag
        .strip_prefix("std::boxed::Box<")
        .or_else(|| tag.strip_prefix("alloc::boxed::Box<"))?;
    let inner = inner.strip_suffix('>')?;
    // The pointee argument ends at the first top-level
    // comma (the allocator separator); nested generic
    // brackets nest.
    let mut depth = 0i32;
    for (i, ch) in inner.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => return Some(inner[..i].trim()),
            _ => {}
        }
    }
    Some(inner.trim())
}

pub(crate) fn comparison_subject_is_raw_pointer(generic_args: &[String]) -> bool {
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

/// The pointee type of a raw-pointer comparison subject (see
/// `comparison_subject_is_raw_pointer`), parsed from the same
/// tag: the element type a step into storage the pointer
/// denotes advances by. Pricing an operand that denotes an
/// element of an aggregate held as a value -- a collection
/// iterator's elements, above all -- needs it: the
/// aggregate's scalar leaves carry no width of their own, so
/// without the subject's own pointee type the offset walk
/// cannot price the element's position, the address comes
/// back underivable, and the comparison falls back to
/// comparing pointees. `None` when the subject is not a thin
/// raw pointer or the pointee tag does not parse.
fn raw_pointer_subject_pointee(generic_args: &[String]) -> Option<Ty> {
    if !comparison_subject_is_raw_pointer(generic_args) {
        return None;
    }
    let tag = generic_args.first()?;
    let tag = tag
        .strip_prefix("&mut ")
        .or_else(|| tag.strip_prefix('&'))
        .unwrap_or(tag);
    let pointee = tag
        .strip_prefix("*const ")
        .or_else(|| tag.strip_prefix("*mut "))?;
    ty_from_tag(pointee.trim())
}

/// Whether a comparison call's own type tags name a dyn raw
/// pointer as the comparison's subject: the `Self` tag (the
/// first generic argument), with one leading reference layer
/// stripped, opens with `*const dyn ` / `*mut dyn `. Dyn
/// pointers are wide: their native equality and ordering are
/// over the (data address, vtable) pair, which
/// `comparison_subject_is_raw_pointer` deliberately leaves
/// to the ordinary path -- but the ordinary path compares
/// pointees, which is neither word, so the raw-pointer arms
/// judge dyn subjects by that pair instead (see
/// `raw_pointer_operand_key`).
pub(crate) fn comparison_subject_is_dyn_pointer(generic_args: &[String]) -> bool {
    let Some(tag) = generic_args.first() else {
        return false;
    };
    let tag = tag
        .strip_prefix("&mut ")
        .or_else(|| tag.strip_prefix('&'))
        .unwrap_or(tag);
    tag.strip_prefix("*const ")
        .or_else(|| tag.strip_prefix("*mut "))
        .is_some_and(|pointee| pointee.starts_with("dyn "))
}

/// How many reference layers one operand of a raw-pointer
/// comparison sits behind: the call passes a reference to
/// the compared value, and when the subject is itself a
/// reference to a raw pointer (`&*const T` -- the shape a
/// comparison of two `&T` values in generic code takes, and
/// of `(&a) < (&b)` directly), the compared value is a
/// reference whose denoted value is the pointer, one layer
/// deeper. Pricing the outer reference instead would price
/// the slot that holds it, ordering the comparison by
/// storage position rather than by the pointers denoted.
fn raw_pointer_operand_layers(generic_args: &[String]) -> usize {
    match generic_args.first() {
        Some(tag) if tag.starts_with('&') => 2,
        _ => 1,
    }
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

    /// One operand of a scalar comparison, read under the
    /// compared type's own kind when the operand is a
    /// reference denoting a scalar storage of the same width
    /// but a different kind (a same-width punned view's
    /// element reference: `&view[0]` for a `&[u32]` over an
    /// `f32` place, Issue #479): the reference denotes the
    /// storage's bits under the element kind, not the backing
    /// scalar, so the operand is the punned reading (see
    /// `Interpreter::punned_scalar_element`). A reference
    /// spilled through a local is followed, as the fmt
    /// argument path follows it. `None` when the operand
    /// denotes no such punned element.
    fn punned_scalar_operand(
        &self,
        frames: &[Frame],
        value: &Value,
        elem_tag: &str,
    ) -> Option<Value> {
        let mut cur = value.clone();
        for _ in 0..8 {
            let target = self.deref_value(frames, &cur).ok()?;
            if let Some(v) = self.punned_scalar_element(frames, &target, elem_tag) {
                return Some(v);
            }
            let read = self.read_resolved(frames, &target).ok()?;
            match read {
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => {
                    cur = read;
                }
                _ => return None,
            }
        }
        None
    }

    /// Scalar equality under a same-width punned view, when
    /// one applies (see `punned_scalar_operand`): the punned
    /// operand is read under the compared type's kind and the
    /// other operand resolves as usual, so the pair compares
    /// as values of the compared type. `None` when neither
    /// operand is such a view, in which case the ordinary
    /// comparison judges.
    pub(crate) fn punned_scalar_equal(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
    ) -> Option<InterpResult<bool>> {
        let tag = punned_scalar_tag(generic_args.first()?);
        let pa = self.punned_scalar_operand(frames, a, tag);
        let pb = self.punned_scalar_operand(frames, b, tag);
        if pa.is_none() && pb.is_none() {
            return None;
        }
        Some((|| {
            let a = match pa {
                Some(v) => v,
                None => self.resolve_ref(frames, a)?.into_owned(),
            };
            let b = match pb {
                Some(v) => v,
                None => self.resolve_ref(frames, b)?.into_owned(),
            };
            self.values_equal(frames, &a, &b)
        })())
    }

    /// Scalar comparison under a same-width punned view,
    /// when one applies: the ordering counterpart of
    /// `punned_scalar_equal`.
    pub(crate) fn punned_scalar_cmp(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
        discriminants: Option<&[i128]>,
    ) -> Option<InterpResult<Option<Ordering>>> {
        let tag = punned_scalar_tag(generic_args.first()?);
        let pa = self.punned_scalar_operand(frames, a, tag);
        let pb = self.punned_scalar_operand(frames, b, tag);
        if pa.is_none() && pb.is_none() {
            return None;
        }
        Some((|| {
            let a = match pa {
                Some(v) => v,
                None => self.resolve_ref(frames, a)?.into_owned(),
            };
            let b = match pb {
                Some(v) => v,
                None => self.resolve_ref(frames, b)?.into_owned(),
            };
            self.values_cmp(frames, &a, &b, discriminants)
        })())
    }

    /// The field list of a resolved tuple value (a tuple
    /// value, or the one-variant aggregate form arrays and
    /// tuples also travel in). `None` for any other shape.
    fn tuple_fields(value: &Value) -> Option<Vec<Value>> {
        match value {
            Value::Tuple(items) => Some(items.clone()),
            Value::Enum { fields, .. } => Some(fields.clone()),
            _ => None,
        }
    }

    /// The per-field operands of a tuple comparison under a
    /// punned view (see `punned_scalar_operand`): each field
    /// pair, with a field that is a punned scalar element
    /// reference replaced by its punned reading and the other
    /// side of that pair resolved as usual. `None` when the
    /// compared type is not a tuple, either operand does not
    /// resolve to a tuple of the declared arity, or no field
    /// is such a view -- the ordinary structural comparison
    /// then judges, exactly as before.
    fn punned_tuple_fields(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
    ) -> Option<InterpResult<Vec<(Value, Value)>>> {
        let self_ty = ty_from_tag(generic_args.first()?)?;
        let Ty::Tuple(field_tys) = &self_ty else {
            return None;
        };
        let ra = self.resolve_ref(frames, a).ok()?;
        let rb = self.resolve_ref(frames, b).ok()?;
        let fa = Self::tuple_fields(&ra)?;
        let fb = Self::tuple_fields(&rb)?;
        if fa.len() != field_tys.len() || fb.len() != field_tys.len() {
            return None;
        }
        let mut any = false;
        let mut pairs = Vec::with_capacity(fa.len());
        for ((x, y), ty) in fa.into_iter().zip(fb).zip(field_tys.iter()) {
            let tag = scalar_tag_of_ty(ty);
            let (px, py) = match tag {
                Some(tag) => (
                    self.punned_scalar_operand(frames, &x, &tag),
                    self.punned_scalar_operand(frames, &y, &tag),
                ),
                None => (None, None),
            };
            if px.is_some() || py.is_some() {
                any = true;
            }
            pairs.push((x, y, px, py));
        }
        if !any {
            return None;
        }
        Some((|| {
            let mut out = Vec::with_capacity(pairs.len());
            for (x, y, px, py) in pairs {
                let x = match px {
                    Some(v) => v,
                    None => self.resolve_ref(frames, &x)?.into_owned(),
                };
                let y = match py {
                    Some(v) => v,
                    None => self.resolve_ref(frames, &y)?.into_owned(),
                };
                out.push((x, y));
            }
            Ok(out)
        })())
    }

    /// Tuple equality under a punned scalar view in one of
    /// its fields, when one applies (see
    /// `punned_tuple_fields`): the derived `PartialEq` of a
    /// tuple holding the element reference compares the
    /// fields under their own kinds.
    pub(crate) fn punned_tuple_equal(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
    ) -> Option<InterpResult<bool>> {
        let pairs = self.punned_tuple_fields(frames, generic_args, a, b)?;
        Some((|| {
            for (x, y) in pairs? {
                if !self.values_equal(frames, &x, &y)? {
                    return Ok(false);
                }
            }
            Ok(true)
        })())
    }

    /// Tuple comparison under a punned scalar view in one
    /// of its fields: the ordering counterpart of
    /// `punned_tuple_equal`, with the first non-`Equal`
    /// field deciding, as a derived `partial_cmp` does.
    pub(crate) fn punned_tuple_cmp(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
        discriminants: Option<&[i128]>,
    ) -> Option<InterpResult<Option<Ordering>>> {
        let pairs = self.punned_tuple_fields(frames, generic_args, a, b)?;
        Some((|| {
            for (x, y) in pairs? {
                match self.values_cmp(frames, &x, &y, discriminants)? {
                    Some(Ordering::Equal) => {}
                    other => return Ok(other),
                }
            }
            Ok(Some(Ordering::Equal))
        })())
    }

    /// The address one operand of a raw-pointer comparison
    /// denotes: the call passes a reference to the compared
    /// value, so dereferencing the operand's layers (see
    /// `raw_pointer_operand_layers`) reaches the pointer
    /// value itself, and `pointer_address` prices it in the
    /// same model every other pointer identity uses, with the
    /// subject's pointee type as the projection walk's leaf
    /// evidence (see `raw_pointer_subject_pointee`). `None`
    /// when a layer cannot be read or no address can be
    /// derived, leaving the ordinary comparison in charge.
    fn raw_pointer_operand_address(
        &self,
        frames: &[Frame],
        value: &Value,
        leaf_ty: Option<&Ty>,
        layers: usize,
    ) -> Option<u128> {
        let compared = self.raw_pointer_operand_compared(frames, value, layers)?;
        self.pointer_address(frames, &compared, leaf_ty)
    }

    /// The pointer value an operand of a raw-pointer
    /// comparison denotes, after dereferencing the layers
    /// the call passes it behind (see
    /// `raw_pointer_operand_layers`). `None` when a layer
    /// cannot be read.
    fn raw_pointer_operand_compared(
        &self,
        frames: &[Frame],
        value: &Value,
        layers: usize,
    ) -> Option<Value> {
        let mut compared = value.clone();
        for _ in 0..layers {
            compared = match &compared {
                Value::Ref(_) | Value::FatRef { .. } => {
                    let target = self.deref_value(frames, &compared).ok()?;
                    self.read_resolved(frames, &target).ok()?
                }
                _ => break,
            };
        }
        Some(compared)
    }

    /// The comparison key of one operand of a dyn raw-pointer
    /// comparison (see `comparison_subject_is_dyn_pointer`):
    /// the data address and the vtable word, the two words
    /// native wide-pointer equality and ordering are defined
    /// over. `None` when the operand does not denote a dyn
    /// pointer or its address cannot be derived.
    fn raw_pointer_operand_key(
        &self,
        frames: &[Frame],
        value: &Value,
        layers: usize,
    ) -> Option<(u128, u128)> {
        let compared = self.raw_pointer_operand_compared(frames, value, layers)?;
        let Value::DynRef {
            concrete, traits, ..
        } = &compared
        else {
            return None;
        };
        let addr = self.pointer_address(frames, &compared, None)?;
        Some((addr, crate::interp::vtable_word(concrete, traits)))
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
        if comparison_subject_is_dyn_pointer(generic_args) {
            let layers = raw_pointer_operand_layers(generic_args);
            let x = self.raw_pointer_operand_key(frames, a, layers)?;
            let y = self.raw_pointer_operand_key(frames, b, layers)?;
            return Some(x == y);
        }
        if !comparison_subject_is_raw_pointer(generic_args) {
            return None;
        }
        let pointee = raw_pointer_subject_pointee(generic_args);
        let layers = raw_pointer_operand_layers(generic_args);
        let x = self
            .raw_pointer_operand_address(frames, a, pointee.as_ref(), layers)?;
        let y = self
            .raw_pointer_operand_address(frames, b, pointee.as_ref(), layers)?;
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
        if comparison_subject_is_dyn_pointer(generic_args) {
            let layers = raw_pointer_operand_layers(generic_args);
            let x = self.raw_pointer_operand_key(frames, a, layers)?;
            let y = self.raw_pointer_operand_key(frames, b, layers)?;
            return Some(Some(x.cmp(&y)));
        }
        if !comparison_subject_is_raw_pointer(generic_args) {
            return None;
        }
        let pointee = raw_pointer_subject_pointee(generic_args);
        let layers = raw_pointer_operand_layers(generic_args);
        let compared_a = self.raw_pointer_operand_compared(frames, a, layers)?;
        let compared_b = self.raw_pointer_operand_compared(frames, b, layers)?;
        let x = self.pointer_address(frames, &compared_a, pointee.as_ref())?;
        let y = self.pointer_address(frames, &compared_b, pointee.as_ref())?;
        // Two positions of one storage whose addresses the
        // model could not price arithmetically (a field of a
        // tuple of same-named ADT instances, whose sizes the
        // bare type names leave ambiguous) fall back to
        // hashed addresses, whose order is arbitrary -- but
        // natively the positions have a definite order, and
        // the guest asserts it (the fields of one aggregate
        // in declaration order). The binary operators consult
        // the position order in exactly this case (see
        // `eval_binary`); the builtin comparison does the
        // same. Address-identical pointers keep the address
        // verdict, as there.
        if x != y {
            fn target_of<'v>(v: &'v Value) -> Option<&'v RefTarget> {
                match v {
                    Value::Ref(t) => Some(t),
                    Value::FatRef { base, .. } | Value::DynRef { base, .. } => {
                        match &**base {
                            Value::Ref(t) => Some(t),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            if let (Some(ta), Some(tb)) = (target_of(&compared_a), target_of(&compared_b)) {
                if let Some(ord) = crate::interp::positional_order(ta, tb) {
                    return Some(Some(ord));
                }
            }
        }
        Some(Some(x.cmp(&y)))
    }

    /// Equality of two `Box` values through their pointees,
    /// when the comparison's own `Self` tag names a `Box`:
    /// std's `PartialEq for Box<T>` is the pointee's own
    /// equality, but the two box values the model holds may
    /// wrap their pointers in different aggregate shapes (the
    /// one-field form a `Box::new` builds against the real
    /// `Unique` / `NonNull` / allocator nest a collected body
    /// assembles field by field, as `Box::into_boxed_slice`
    /// does), which the structural comparison below reads as
    /// different values. Extracting each side's pointer and
    /// comparing those values compares the pointees in
    /// whatever form each pointer carries. A side whose
    /// pointer cannot be extracted may still be the bare
    /// pointee itself (see the match below). `None` when the
    /// comparison is not a same-type box comparison or a
    /// side is neither a box nor a bare pointee, in which
    /// case the ordinary comparison judges. A `Path` / `PathBuf`
    /// pointee keeps the ordinary path: its equality is
    /// component-wise, not the pointee's structural one.
    pub(crate) fn box_pointee_equal(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
    ) -> Option<InterpResult<bool>> {
        let self_tag = generic_args.first()?;
        if generic_args.get(1) != Some(self_tag) {
            return None;
        }
        let pointee = box_pointee_tag(self_tag)?;
        if pointee.contains("::path::Path") {
            return None;
        }
        let pa = self.box_cmp_pointer(frames, a);
        let pb = self.box_cmp_pointer(frames, b);
        // One side may fail to extract because it is not a
        // box at all but the bare pointee: an element move
        // (a sort's rotation, above all) can leave a slot
        // holding the pointee snapshot the box denoted, and
        // a comparison between the stored box and that
        // snapshot is still the pointees' comparison. A side
        // that is neither a box nor a bare pointee form
        // keeps the ordinary path.
        let (pa, pb) = match (pa, pb) {
            (Some(pa), Some(pb)) => (pa, pb),
            (Some(ptr), None) => {
                let bare = self.resolve_ref(frames, b).ok()?;
                if !is_bare_pointee_form(&bare) {
                    return None;
                }
                return Some(self.values_equal(frames, &ptr, b));
            }
            (None, Some(ptr)) => {
                let bare = self.resolve_ref(frames, a).ok()?;
                if !is_bare_pointee_form(&bare) {
                    return None;
                }
                return Some(self.values_equal(frames, a, &ptr));
            }
            (None, None) => return None,
        };
        // A slice pointee compares element-wise. The
        // element reading needs the element type the tag
        // names: a fat pointer of length 1 formed over the
        // pointee value itself (the `into_boxed_slice`
        // formation) and one formed over an array of that
        // length hold identically shaped storage, and only
        // the element type tells which reading is the
        // slice's (see `box_slice_elements`).
        if let Some(elem_tag) = pointee
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
        {
            if let Some(elem_ty) = ty_from_tag(elem_tag) {
                let ea = self.box_slice_elements(frames, &pa, &elem_ty)?;
                let eb = self.box_slice_elements(frames, &pb, &elem_ty)?;
                return Some(
                    self.seq_equal(frames, &ea, &eb)
                        .map(|eq| ea.len() == eb.len() && eq),
                );
            }
        }
        Some(self.values_equal(frames, &pa, &pb))
    }

    /// Ordering of two `Box` values through their
    /// pointees, the `PartialOrd` / `Ord` counterpart of
    /// [`Self::box_pointee_equal`]: std's ordering impls for
    /// `Box<T>` are the pointee's own, so the pair is
    /// decided by the values the boxes' pointers denote, in
    /// whatever form each side carries them.
    ///
    /// Two presentations reach this dispatch. A tagged call
    /// (the call's own `Self` tag names the `Box`) mirrors
    /// `box_pointee_equal`'s gating exactly, the `Path`
    /// exclusion included: a path's ordering is
    /// component-wise, not the pointee's byte order, and
    /// belongs to the path dispatch, which runs before this
    /// one; the box shortcut never decides a path
    /// comparison, here as in equality. An untagged call --
    /// one the converter kept in generic form, which is how
    /// a sort over a `Vec<Box<T>>` compares -- carries no
    /// type tags, and there the stored box itself is the
    /// signal: the sort's rotation hands the comparison one
    /// side as the box aggregate a collection slot holds and
    /// the other as the bare pointee snapshot it extracted,
    /// a pair the structural comparison below has no arm
    /// for. When exactly one side is such a stored box (see
    /// `stored_box_pointer`) and the other resolves to a
    /// bare pointee form, substituting the box's pointer for
    /// the box lets the ordinary comparison judge the pair.
    /// A pair of stored boxes needs no substitution (the
    /// structural comparison already descends through the
    /// aggregates to the pointees), and a side that resolves
    /// to an aggregate is never taken for a bare pointee, so
    /// no other type's comparison is decided here.
    pub(crate) fn box_pointee_cmp(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        a: &Value,
        b: &Value,
    ) -> Option<InterpResult<Option<Ordering>>> {
        if generic_args.is_empty() {
            return self.box_mixed_cmp(frames, a, b, true, None);
        }
        let self_tag = generic_args.first()?;
        if generic_args.len() > 1 && generic_args.get(1) != Some(self_tag) {
            return None;
        }
        let pointee = box_pointee_tag(self_tag)?;
        if pointee.contains("::path::Path") {
            return None;
        }
        // A slice pointee orders element-wise (see
        // `box_pointee_equal` for why the element reading
        // needs the element type the tag names).
        let elem_ty = pointee
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
            .and_then(ty_from_tag);
        self.box_mixed_cmp(frames, a, b, false, elem_ty.as_ref())
    }

    /// The comparison of a box pair whose sides
    /// independently present as a box (its pointer
    /// extracted) or as the bare pointee (see
    /// `box_pointee_cmp` for the presentations). With
    /// `stored_only` the extraction is `stored_box_pointer`'s
    /// (the untagged signal); otherwise it is
    /// `box_cmp_pointer`'s, as in `box_pointee_equal`. Two
    /// extracted pointers compare through their pointees --
    /// element-wise under `elem_ty` when the pointee is a
    /// slice -- and a pointer against a bare side compares
    /// through the ordinary comparison, which resolves the
    /// pointer to the same pointee the bare side snapshots.
    /// A side that is neither keeps the ordinary comparison
    /// in charge (`None`).
    fn box_mixed_cmp(
        &self,
        frames: &[Frame],
        a: &Value,
        b: &Value,
        stored_only: bool,
        elem_ty: Option<&Ty>,
    ) -> Option<InterpResult<Option<Ordering>>> {
        let extract = |v: &Value| {
            if stored_only {
                self.stored_box_pointer(frames, v)
            } else {
                self.box_cmp_pointer(frames, v)
            }
        };
        match (extract(a), extract(b)) {
            (Some(pa), Some(pb)) => {
                if let Some(elem_ty) = elem_ty {
                    let ea = self.box_slice_elements(frames, &pa, elem_ty)?;
                    let eb = self.box_slice_elements(frames, &pb, elem_ty)?;
                    return Some(self.seq_cmp(frames, &ea, &eb));
                }
                Some(self.values_cmp(frames, &pa, &pb, None))
            }
            (Some(ptr), None) => {
                let bare = self.resolve_ref(frames, b).ok()?;
                if !is_bare_pointee_form(&bare) {
                    return None;
                }
                Some(self.values_cmp(frames, &ptr, b, None))
            }
            (None, Some(ptr)) => {
                let bare = self.resolve_ref(frames, a).ok()?;
                if !is_bare_pointee_form(&bare) {
                    return None;
                }
                Some(self.values_cmp(frames, a, &ptr, None))
            }
            (None, None) => None,
        }
    }

    /// The pointer a stored box value denotes, when the
    /// operand resolves to the box aggregate itself (the
    /// form a collection slot holds): the same first-field
    /// descent as [`Self::box_cmp_pointer`], except that a
    /// value which is already a pointer (or a bare pointee)
    /// is not a stored box and answers `None`. The descent
    /// reaches a pointer only through the wrapper chain a
    /// box is stored as; the other modelled string-like and
    /// collection forms end at their element lists instead
    /// (see `is_pathbuf_value`), so their comparisons are
    /// never mistaken for a box's.
    fn stored_box_pointer(&self, frames: &[Frame], arg: &Value) -> Option<Value> {
        let mut cur = self.resolve_ref(frames, arg).ok()?.into_owned();
        if !matches!(cur, Value::Enum { .. }) {
            return None;
        }
        loop {
            match cur {
                Value::Enum { fields, .. } if !fields.is_empty() => {
                    cur = fields.into_iter().next()?;
                }
                ptr @ (Value::Ref(_)
                | Value::FatRef { .. }
                | Value::DynRef { .. }) => return Some(ptr),
                _ => return None,
            }
        }
    }

    /// The elements of one side of a box-of-slice
    /// comparison, read under the slice's element type.
    /// The ordinary element reading judges a fat pointer's
    /// storage; the one ambiguous shape is a fat pointer of
    /// length 1 whose storage is a one-element sequence: it
    /// may be the slice's own storage (the array an unsize
    /// coercion ranged over) or the single element itself
    /// (a fat pointer formed over the pointee value, as
    /// `Box::into_boxed_slice` forms over the boxed value).
    /// The element type decides: when the reading's one
    /// element does not have the element type's shape but
    /// the stored value itself does, the stored value is
    /// the element. A thin pointer (the model's own box
    /// form) denotes the pointee's storage directly, whose
    /// elements are the slice's.
    fn box_slice_elements(
        &self,
        frames: &[Frame],
        ptr: &Value,
        elem_ty: &Ty,
    ) -> Option<Vec<Value>> {
        match ptr {
            Value::FatRef { base, len } => {
                let (elems, _) = self.slice_elements_ref(frames, ptr).ok()?;
                let elems = elems.into_vec();
                if *len == 1
                    && elems.len() == 1
                    && !elem_shape_matches(&elems[0], elem_ty)
                {
                    if let Ok(target) = self.deref_value(frames, base) {
                        if let Ok(stored) = self.read_resolved(frames, &target) {
                            if elem_shape_matches(&stored, elem_ty) {
                                return Some(vec![stored]);
                            }
                        }
                    }
                }
                Some(elems)
            }
            Value::Ref(_) => {
                let stored = self.resolve_ref(frames, ptr).ok()?;
                let elems = elements_of(stored).ok()?;
                Some(elems.into_vec())
            }
            _ => None,
        }
    }

    /// The pointer a box comparison operand denotes: the
    /// call passes a reference to the box, resolving it
    /// yields the box value, and the box aggregate wraps its
    /// pointer in the one-field `Unique` / `NonNull` chain
    /// with the allocator beside it, so first fields lead to
    /// the pointer. `None` when the resolved value is not
    /// such a nest ending in a pointer (a box the model holds
    /// as a bare value has no pointer to extract either: the
    /// resolution already followed it to the pointee).
    fn box_cmp_pointer(&self, frames: &[Frame], arg: &Value) -> Option<Value> {
        let mut cur = self.resolve_ref(frames, arg).ok()?.into_owned();
        loop {
            match cur {
                Value::Enum { fields, .. } if !fields.is_empty() => {
                    cur = fields.into_iter().next()?;
                }
                ptr @ (Value::Ref(_)
                | Value::FatRef { .. }
                | Value::DynRef { .. }) => return Some(ptr),
                _ => return None,
            }
        }
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
        let a_byte_tail = is_byte_tail_ref(a);
        let b_byte_tail = is_byte_tail_ref(b);
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
        // An `OsString` denotes the bytes its buffer holds,
        // in whichever of the model's shapes it is stored --
        // the element form the pure `os_string_bytes` reads,
        // or the real aggregate form
        // `os_string_bytes_from_real_aggregate` reads through
        // the buffer pointer (the shape `with_capacity` plus
        // `push` builds, whose storage arrangement -- a fresh
        // allocation per clone -- is not part of the value's
        // identity): when both sides yield their OS-string
        // bytes they compare as those bytes. A side that
        // yields no bytes decides nothing, and the pair keeps
        // the structural comparison below, exactly as the
        // element-form loop above.
        if let (Some(x), Some(y)) = (
            self.os_string_bytes_any(frames, a.as_ref())?,
            self.os_string_bytes_any(frames, b.as_ref())?,
        ) {
            return Ok(x == y);
        }
        // An `OsStr` value in its bare form -- the byte
        // string itself, as a `Box<OsStr>`'s pointee and
        // byte-view reads of OS-string storage hold it, or
        // the text form when the bytes came from a string
        // literal -- denotes the same bytes: against an
        // OS-string shape, or against the other bare form,
        // it compares as those bytes. Two bare byte strings
        // never take this arm: a byte string there may be a
        // scalar's storage, whose reading the scalar arm
        // below decides by the partner's form.
        for (side, other) in [(&*a, &*b), (&*b, &*a)] {
            let bare = match side {
                Value::Bytes(bytes) => Some(bytes.clone()),
                Value::Str(text) => Some(text.clone().into_bytes()),
                _ => None,
            };
            let Some(bare) = bare else { continue };
            let other_bytes = match other {
                Value::Bytes(_) => None,
                Value::Str(text) => Some(text.clone().into_bytes()),
                _ => self.os_string_bytes_any(frames, other)?,
            };
            if let Some(other_bytes) = other_bytes {
                return Ok(bare == other_bytes);
            }
        }
        // A scalar in its byte form against the same scalar in
        // its integer form: the pointee of a CTFE-extracted
        // reference, above all, against the integer the same
        // value takes when built at runtime. Two byte shapes
        // denote a scalar, and the reading is exact for each:
        // a bare byte string IS the scalar's storage (a byte
        // view of exactly the scalar), read whole as its
        // little-endian representation; a byte string a
        // byte-sized pointee's reference resolved to (see
        // `is_byte_tail_ref`, the flags captured before
        // resolution above) is the allocation from the pointee
        // on, and the scalar is its first byte -- read unsigned
        // against an unsigned partner, sign-extended against a
        // signed one, as a truth value against a boolean one,
        // the partner's form naming the pointee's type, as the
        // two sides of one comparison share one source type.
        // A byte string of no scalar width falls through to
        // the structural comparison and its mismatch verdict,
        // as before.
        for (side, other, tail) in [
            (&*a, &*b, a_byte_tail),
            (&*b, &*a, b_byte_tail),
        ] {
            if matches!(side, Value::Bytes(_)) {
                match other {
                    Value::Uint(y) => {
                        let v = if tail {
                            side.bytes_prefix_as_uint(1)
                        } else {
                            side.bytes_as_uint()
                        };
                        if let Some(v) = v {
                            return Ok(v == *y);
                        }
                    }
                    Value::Int(y) => {
                        let v = if tail {
                            side.bytes_prefix_as_int(1)
                        } else {
                            side.bytes_as_int()
                        };
                        if let Some(v) = v {
                            return Ok(v == *y);
                        }
                    }
                    Value::Bool(y) if tail => {
                        if let Value::Bytes(bytes) = side {
                            let first = bytes.first().copied().unwrap_or(0);
                            return Ok((first != 0) == *y);
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
        // Pointer pairs the resolution below cannot judge: an
        // address-only or null pointer denotes no data to
        // resolve to, and a dyn fat pointer's ordering is over
        // its two words, not its pointee -- the pair a raw
        // wide pointer transmuted to a word array compares by,
        // above all. Both order by the (data address, metadata
        // word) key the binary operators use (see
        // `eval_binary`): the metadata word is a dyn pointer's
        // vtable identity, and a thin side of a mixed pair
        // drops the word, as there. This is the ordering
        // counterpart of the address arm `values_equal` has
        // always had; without it these pairs fell through to
        // the structural match and its type-mismatch error.
        let pointer_key = |v: &Value| -> Option<(u128, Option<u128>)> {
            match v {
                Value::Ref(RefTarget::Null) => Some((0, None)),
                Value::DynRef {
                    concrete, traits, ..
                } => self
                    .pointer_address(frames, v, None)
                    .map(|addr| (addr, Some(crate::interp::vtable_word(concrete, traits)))),
                Value::Ref(_) | Value::FnPtr(_) | Value::Closure { .. } => {
                    self.pointer_address(frames, v, None).map(|addr| (addr, None))
                }
                _ => None,
            }
        };
        if matches!(a, Value::DynRef { .. })
            || matches!(b, Value::DynRef { .. })
            || matches!(a, Value::Ref(RefTarget::Address(_)) | Value::Ref(RefTarget::Null))
            || matches!(b, Value::Ref(RefTarget::Address(_)) | Value::Ref(RefTarget::Null))
        {
            if let (Some(x), Some(y)) = (pointer_key(a), pointer_key(b)) {
                let (x, y) = match (x, y) {
                    ((addr, Some(_)), (other, None)) => ((addr, None), (other, None)),
                    ((addr, None), (other, Some(_))) => ((addr, None), (other, None)),
                    pair => pair,
                };
                return Ok(Some(x.cmp(&y)));
            }
        }
        let a_byte_tail = is_byte_tail_ref(a);
        let b_byte_tail = is_byte_tail_ref(b);
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
        // A side that resolves to a self-contained fat value
        // (a slice constant or materialised sub-slice held in
        // a local) orders as a slice against the other side's
        // elements: the entry arm above sees only a fat value
        // handed over directly, but a reference to a stored
        // fat pointer -- the shape a reborrowed `&OsStr`
        // takes -- resolves to the fat value itself, and the
        // pair is the slice comparison the entry arm performs
        // (the same post-resolution arm `values_equal` has).
        if matches!(&*a, Value::FatRef { .. }) || matches!(&*b, Value::FatRef { .. }) {
            let (ea, la) = self.slice_elements_ref(frames, &a)?;
            let (eb, lb) = self.slice_elements_ref(frames, &b)?;
            match self.elems_cmp(frames, &ea, &eb)? {
                Some(Ordering::Equal) => {}
                other => return Ok(other),
            }
            return Ok(Some(la.cmp(&lb)));
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
        // An `OsString` denotes the bytes its buffer holds,
        // in whichever of the model's shapes it is stored
        // (see `values_equal`): when both sides yield their
        // OS-string bytes they order as those bytes, which
        // is the byte-wise lexicographic order `OsStr`'s
        // `Ord` performs. A side that yields no bytes
        // decides nothing, and the pair keeps the structural
        // comparison below.
        if let (Some(x), Some(y)) = (
            self.os_string_bytes_any(frames, a.as_ref())?,
            self.os_string_bytes_any(frames, b.as_ref())?,
        ) {
            return Ok(Some(x.cmp(&y)));
        }
        // An `OsStr` value in its bare form -- the byte
        // string itself, as a `Box<OsStr>`'s pointee and
        // byte-view reads of OS-string storage hold it, or
        // the text form when the bytes came from a string
        // literal -- denotes the same bytes (see the bare
        // arm of `values_equal`): against an OS-string
        // shape, or against the other bare form, it orders
        // as those bytes. Two bare byte strings never take
        // this arm: a byte string there may be a scalar's
        // storage, whose reading the scalar arm below
        // decides by the partner's form.
        for (side, other, flip) in [(&*a, &*b, false), (&*b, &*a, true)] {
            let bare = match side {
                Value::Bytes(bytes) => Some(bytes.clone()),
                Value::Str(text) => Some(text.clone().into_bytes()),
                _ => None,
            };
            let Some(bare) = bare else { continue };
            let other_bytes = match other {
                Value::Bytes(_) => None,
                Value::Str(text) => Some(text.clone().into_bytes()),
                _ => self.os_string_bytes_any(frames, other)?,
            };
            if let Some(other_bytes) = other_bytes {
                let ord = bare.cmp(&other_bytes);
                return Ok(Some(if flip { ord.reverse() } else { ord }));
            }
        }
        // A scalar in its byte form orders as the integer it
        // denotes, against the same scalar in its integer form
        // (see `values_equal` for the two byte shapes and why
        // each reading is exact). A byte string of no scalar
        // width falls through to the structural comparison and
        // its mismatch verdict, as before.
        for (side, other, flip, tail) in [
            (&*a, &*b, false, a_byte_tail),
            (&*b, &*a, true, b_byte_tail),
        ] {
            if matches!(side, Value::Bytes(_)) {
                let ord = match other {
                    Value::Uint(y) => {
                        let v = if tail {
                            side.bytes_prefix_as_uint(1)
                        } else {
                            side.bytes_as_uint()
                        };
                        v.map(|v| v.cmp(y))
                    }
                    Value::Int(y) => {
                        let v = if tail {
                            side.bytes_prefix_as_int(1)
                        } else {
                            side.bytes_as_int()
                        };
                        v.map(|v| v.cmp(y))
                    }
                    _ => None,
                };
                if let Some(ord) = ord {
                    return Ok(Some(if flip { ord.reverse() } else { ord }));
                }
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
