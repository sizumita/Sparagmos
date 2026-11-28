//! C string and byte argument marshalling for the FFI builtins.



use super::*;
use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter,
};
use crate::value::{RefProj, RefTarget, Value};

use super::fmt::pointee_tag;


/// The fat-pointer metadata a value carries for its innermost
/// unsized tail: a slice length, or the concrete type name a
/// `dyn` value's vtable answers for. The fat pointer of a
/// dynamically sized aggregate carries the *tail's* metadata,
/// so one value's metadata governs every aggregate layer
/// between it and the tail.
#[derive(Clone)]
enum DstMeta {
    Len(u128),
    Concrete(String),
}

fn dst_meta_of(value: &Value) -> Option<DstMeta> {
    match value {
        Value::FatRef { len, .. } => Some(DstMeta::Len(*len)),
        Value::DynRef { concrete, .. } => {
            Some(DstMeta::Concrete(concrete.clone()))
        }
        _ => None,
    }
}

fn align_up_u128(n: u128, align: u128) -> Option<u128> {
    if align <= 1 {
        return Some(n);
    }
    n.checked_add(align - 1).map(|v| v & !(align - 1))
}


impl<'a> Interpreter<'a> {
    /// The bytes of a `&[u8]` argument.
    pub(super) fn arg_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        let (items, _) = self.slice_elements(frames, value)?;
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Value::Uint(b) => out.push(b as u8),
                other => {
                    return Err(InterpError::type_mismatch(
                        "byte slice argument",
                        "byte elements",
                        other.kind_name(),
                    ))
                }
            }
        }
        Ok(out)
    }

    /// A `CString` value over `bytes` (given without the
    /// trailing NUL): the struct's single field is the
    /// owned boxed byte slice, built the way
    /// `Vec::into_boxed_slice` builds it (an allocation
    /// holding the byte elements, terminator appended).
    pub(super) fn c_string_value(&self, mut bytes: Vec<u8>) -> Value {
        bytes.push(0);
        let elems: Vec<Value> = bytes
            .iter()
            .map(|b| Value::Uint(u128::from(*b)))
            .collect();
        Value::Enum {
            variant: 0,
            fields: vec![self.new_box(Value::Tuple(elems))],
        }
    }

    /// The layout of the concrete type a `dyn` value's
    /// vtable names: `size_of_val` / `align_of_val` of a
    /// trait object are the concrete type's size and
    /// alignment, which the vtable carries -- here, the
    /// `DynRef`'s recorded concrete type, looked up in the
    /// layout table. `None` when the tag is not a `dyn`
    /// type, or the value names no concrete type with a
    /// recorded layout.
    pub(super) fn dyn_layout_of(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        tag: &str,
    ) -> Option<(u128, u128)> {
        if !pointee_tag(tag).starts_with("dyn ") {
            return None;
        }
        let resolved = self.resolve_value(frames, value).ok()?;
        let Value::DynRef { concrete, .. } = resolved else {
            return None;
        };
        self.layout_of(&concrete)
    }

    /// The `(size, align)` stored in the vtable a raw
    /// pointer value denotes: the layout of the concrete
    /// type the vtable is for, which is what the
    /// `vtable_size` / `vtable_align` intrinsics read out
    /// of the vtable's header. The pointer denotes the
    /// vtable in any of the model's forms: the `dyn` value
    /// itself, a value carrying a driver-recorded vtable
    /// identity, or the interpreter's own vtable word (the
    /// form a transmute of a fat pointer produces for the
    /// metadata half), matched against the recorded
    /// identities by recomputing each one's word.
    pub(super) fn vtable_stored_layout(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
    ) -> Option<(u128, u128)> {
        let resolved = self.resolve_value(frames, value).ok()?;
        if let Value::DynRef { concrete, .. } = &resolved {
            return self.layout_of(concrete);
        }
        if let Some(id) = self.vtable_id_in_metadata(&resolved) {
            let identity = self.program().vtable_identities.get(&id)?;
            return self.layout_of(&identity.concrete);
        }
        if let Value::Ref(RefTarget::Address(word)) = &resolved {
            if let Some(concrete) = self.vtable_word_concrete(u128::from(*word)) {
                return self.layout_of(&concrete);
            }
        }
        None
    }

    /// The `(size, align)` of a value of a dynamically sized
    /// type, composed from the value itself: the layout the
    /// `size_of_val` / `align_of_val` builtins answer when
    /// neither the static layout table nor the `dyn` vtable
    /// path can. An aggregate with an unsized tail (a struct or
    /// union with a driver-recorded descriptor, or a tuple) is
    /// its prefix plus the tail's own dynamic layout, the tail
    /// placed at the prefix end rounded up to the tail's
    /// effective alignment; a function item type is zero-sized.
    /// `None` when the composition cannot be carried out (the
    /// caller stays unsupported).
    pub(super) fn dst_size_align_of_val(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        tag: &str,
    ) -> Option<(u128, u128)> {
        self.dst_layout_of(frames, value, None, tag)
    }

    /// The recursive composition behind
    /// [`Self::dst_size_align_of_val`]: `carrier` is the value
    /// (or the reference denoting it) whose type `tag` names,
    /// and `inherited` is the fat-pointer metadata an enclosing
    /// aggregate's pointer carries for this tail (see
    /// [`DstMeta`]); a carrier carrying metadata of its own
    /// supersedes the inherited one.
    fn dst_layout_of(
        &mut self,
        frames: &mut Vec<Frame>,
        carrier: &Value,
        inherited: Option<&DstMeta>,
        tag: &str,
    ) -> Option<(u128, u128)> {
        let tag = tag.trim();
        let own = dst_meta_of(carrier);
        let meta = own.as_ref().or(inherited);
        if tag == "str" {
            let len = match meta {
                Some(DstMeta::Len(n)) => *n,
                _ => {
                    let v = self.resolve_value(frames, carrier).ok()?;
                    match v {
                        Value::Str(text) => text.len() as u128,
                        _ => self.resolve_string(frames, carrier).ok()?.len()
                            as u128,
                    }
                }
            };
            return Some((len, 1));
        }
        if super::is_slice_tag(tag) {
            let elem = &tag[1..tag.len() - 1];
            let (elem_size, elem_align) = self.layout_of(elem)?;
            let len = match meta {
                Some(DstMeta::Len(n)) => *n,
                _ => self.resolve_value(frames, carrier).ok()?.len()?,
            };
            return Some((len.checked_mul(elem_size)?, elem_align));
        }
        if pointee_tag(tag).starts_with("dyn ") {
            if let Some(DstMeta::Concrete(concrete)) = meta {
                return self.layout_of(concrete);
            }
            return self.dyn_layout_of(frames, carrier, tag);
        }
        if let Some(layout) = self.layout_of(tag) {
            return Some(layout);
        }
        if let Some(desc) =
            self.program().adt_dst_layouts.get(tag).cloned()
        {
            let (tail_size, tail_align) =
                self.dst_tail_layout(frames, carrier, meta, &desc)?;
            let eff_align = match desc.pack {
                Some(pack) => tail_align.min(u128::from(pack)),
                None => tail_align,
            };
            let prefix_align = u128::from(desc.prefix_align);
            if desc.union {
                return Some((
                    u128::from(desc.prefix_end).max(tail_size),
                    prefix_align.max(eff_align),
                ));
            }
            let tail_offset =
                align_up_u128(u128::from(desc.prefix_end), eff_align)?;
            let align = prefix_align.max(eff_align);
            // The composed size is padded up to the composed
            // alignment, as every layout's size is: a
            // `WT<(u16, [u8])>` of length 3 spans 2 + (2 + 3)
            // bytes of fields but its size is 8, not 7.
            let size =
                align_up_u128(tail_offset.checked_add(tail_size)?, align)?;
            return Some((size, align));
        }
        if tag.len() > 2 && tag.starts_with('(') && tag.ends_with(')') {
            return self.dst_tuple_layout_of(frames, carrier, meta, tag);
        }
        // A function item type is zero-sized: the carrier
        // resolves to the function pointer value itself.
        // (An intrinsic-backed item like `std::mem::size_of`
        // has no collected body, so the program's function
        // table cannot name it; the value can.)
        if let Ok(Value::FnPtr(_)) = self.resolve_value(frames, carrier) {
            return Some((0, 1));
        }
        // A function item type is zero-sized: its tag is the
        // instance name of a function of the program. The
        // tag drops the generic arguments the program's
        // function keys carry, so an exact key or a key
        // continuing with the argument list both name the
        // item.
        if self.program().functions.keys().any(|name| {
            name == tag
                || name.strip_prefix(tag).is_some_and(|rest| {
                    rest.starts_with('<') || rest.starts_with("::<")
                })
        }) {
            return Some((0, 1));
        }
        None
    }

    /// The dynamic layout of the unsized tail field a DST
    /// descriptor names: the tail value sits in the pointee
    /// aggregate's recorded field, under the fat metadata of
    /// the whole value. When the pointee cannot be read (an
    /// address-only fat pointer, whose metadata is all the
    /// value carries), the recursion runs on the carrier
    /// itself, where the metadata alone decides the `str` /
    /// slice / `dyn` tails.
    fn dst_tail_layout(
        &mut self,
        frames: &mut Vec<Frame>,
        carrier: &Value,
        meta: Option<&DstMeta>,
        desc: &sparagmos_mir::AdtDstLayout,
    ) -> Option<(u128, u128)> {
        if let Some(pointee) = self.dst_pointee(frames, carrier) {
            let fields = match &pointee {
                Value::Enum { fields, .. } => Some(fields),
                Value::Tuple(items) => Some(items),
                _ => None,
            };
            if let Some(tail) =
                fields.and_then(|f| f.get(desc.tail_field as usize))
            {
                let tail = tail.clone();
                return self.dst_layout_of(frames, &tail, meta, &desc.tail_ty);
            }
        }
        self.dst_layout_of(frames, carrier, meta, &desc.tail_ty)
    }

    /// The aggregate value a fat or thin reference denotes,
    /// for the DST layout composition: references resolve to
    /// their pointee, and a `DynRef` unwraps to the value
    /// behind its base (the sized aggregate the coercion
    /// started from).
    fn dst_pointee(
        &mut self,
        frames: &mut Vec<Frame>,
        carrier: &Value,
    ) -> Option<Value> {
        let v = self.resolve_value(frames, carrier).ok()?;
        match v {
            Value::DynRef { base, .. } => self.dst_pointee(frames, &base),
            other => Some(other),
        }
    }

    /// The layout of a tuple with an unsized last element: the
    /// sized elements' layouts are static (their sum is the
    /// prefix end, as in the static tuple layout), and the
    /// last element's layout is dynamic (see
    /// [`Self::dst_layout_of`]).
    fn dst_tuple_layout_of(
        &mut self,
        frames: &mut Vec<Frame>,
        carrier: &Value,
        meta: Option<&DstMeta>,
        tag: &str,
    ) -> Option<(u128, u128)> {
        let inner = &tag[1..tag.len() - 1];
        let parts = super::split_top_level(inner);
        let (last, prefix) = parts.split_last()?;
        // Every element before the last must have a static
        // layout: the composition is only sound when the last
        // element is the dynamic one.
        let mut prefix_end: u128 = 0;
        let mut prefix_align: u128 = 1;
        for part in prefix {
            let (size, align) = self.layout_of(part)?;
            prefix_end = prefix_end.checked_add(size)?;
            prefix_align = prefix_align.max(align);
        }
        let pointee = self.dst_pointee(frames, carrier)?;
        let Value::Tuple(items) = &pointee else {
            return None;
        };
        let tail = items.get(parts.len() - 1)?.clone();
        let (tail_size, tail_align) =
            self.dst_layout_of(frames, &tail, meta, last)?;
        if prefix.is_empty() {
            return Some((tail_size, tail_align));
        }
        let tail_offset = align_up_u128(prefix_end, tail_align)?;
        let align = prefix_align.max(tail_align);
        let size =
            align_up_u128(tail_offset.checked_add(tail_size)?, align)?;
        Some((size, align))
    }

    /// `ThinBox<T>` dereference, shared by `Deref` and
    /// `DerefMut`: the box's one stored pointer names the
    /// pointee, and the header std's construction wrote
    /// immediately ahead of the pointee -- in the same
    /// allocation, or, for a zero-sized pointee, ahead of the
    /// one-past-the-end pointer into the constant metadata
    /// allocation -- is the pointee's metadata. A `dyn`
    /// header rebases onto the data pointer as a `DynRef`; a
    /// length header makes the fat reference of a slice
    /// pointee; a sized pointee's unit header leaves the data
    /// pointer itself as the result.
    pub(super) fn thin_box_deref(
        &mut self,
        frames: &mut Vec<Frame>,
        arg: &Value,
        self_ty: &str,
        what: &str,
    ) -> InterpResult<Value> {
        let pointee_tag = self_ty
            .strip_prefix("std::boxed::ThinBox<")
            .and_then(|rest| rest.strip_suffix('>'))
            .unwrap_or("");
        let resolved = self.resolve_value(frames, arg)?;
        // The stored pointer: field 0 of the box aggregate,
        // under the one-field wrappers of `WithOpaqueHeader`
        // and `NonNull`.
        let mut data = resolved;
        loop {
            match data {
                Value::Enum { mut fields, .. } if fields.len() == 1 => {
                    data = fields.pop().expect("len checked");
                }
                Value::Enum { fields, .. } if fields.len() >= 2 => {
                    data = fields.into_iter().next().expect("len checked");
                }
                other => {
                    data = other;
                    break;
                }
            }
        }
        let Value::Ref(data_target) = data.clone() else {
            return Err(InterpError::type_mismatch(
                what,
                "thin box holding a data pointer",
                data.kind_name(),
            ));
        };
        if !(pointee_tag.starts_with("dyn ") || pointee_tag.starts_with('[') || pointee_tag == "str") {
            // A sized pointee: the header is the unit value,
            // and the data pointer is the dereference. A data
            // pointer at an element block's base denotes its
            // first element (the value the empty header
            // precedes), as a place dereference of the same
            // pointer resolves it.
            if let RefTarget::Alloc { id, projection } = &data_target {
                if projection.is_empty() {
                    if let Ok(Value::Tuple(_)) = self.alloc_base(*id) {
                        return Ok(Value::Ref(RefTarget::Alloc {
                            id: *id,
                            projection: vec![RefProj::Index(0)],
                        }));
                    }
                }
            }
            return Ok(data);
        }
        // The header word sits one metadata word (a vtable
        // pointer, or a length) ahead of the data pointer.
        // The offset is denominated the way the target's own
        // positions are: an allocation's element positions are
        // byte-denominated for the `u8` arithmetic std forms
        // this pointer with, while a constant value target's
        // index positions are denominated in the metadata
        // element itself.
        let (count, elem_size) = match data_target {
            RefTarget::Alloc { .. } => (-8, Some(1)),
            _ => (-1, Some(8)),
        };
        let header_target =
            self.offset_target(data_target.clone(), frames, count, elem_size, Some(-8), false)?;
        let header = self.resolve_value(frames, &Value::Ref(header_target))?;
        if pointee_tag.starts_with("dyn ") {
            let dynref = self
                .dyn_ref_from_metadata(&header, data.clone())
                .or_else(|| {
                    // A header read back through the byte view
                    // can arrive as the vtable identity's bits.
                    let id = match header {
                        Value::Uint(v) => u64::try_from(v).ok(),
                        Value::Int(v) => u64::try_from(v).ok(),
                        _ => None,
                    }?;
                    self.dyn_ref_from_vtable(id, data.clone())
                });
            dynref.ok_or_else(|| {
                InterpError::type_mismatch(
                    what,
                    "thin box header naming a recorded vtable",
                    header.kind_name(),
                )
            })
        } else {
            match header {
                Value::Uint(len) => Ok(Value::FatRef {
                    base: Box::new(data),
                    len,
                }),
                Value::Int(len) => Ok(Value::FatRef {
                    base: Box::new(data),
                    len: len as u128,
                }),
                ref other => Err(InterpError::type_mismatch(
                    what,
                    "thin box header holding a length",
                    other.kind_name(),
                )),
            }
        }
    }

    /// The `&CStr` a `CString` dereferences to: the fat
    /// reference over the owned buffer, terminator
    /// included. The base is the inner box's allocation
    /// pointer and the length is the buffer's byte count.
    pub(super) fn c_string_deref(
        &self,
        frames: &[Frame],
        receiver: &Value,
    ) -> InterpResult<Value> {
        let pointee = self.resolve_value(frames, receiver)?;
        match pointee {
            Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
                Value::FatRef { .. } => Ok(fields[0].clone()),
                Value::Enum {
                    fields: box_fields,
                    ..
                } if box_fields.len() == 1 => {
                    let base = box_fields[0].clone();
                    let buf = self.resolve_value(frames, &base)?;
                    let len = buf.len().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "Deref for CString",
                            "a byte buffer value",
                            buf.kind_name(),
                        )
                    })?;
                    Ok(Value::FatRef {
                        base: Box::new(base),
                        len,
                    })
                }
                other => Err(InterpError::type_mismatch(
                    "Deref for CString",
                    "a boxed byte slice",
                    other.kind_name(),
                )),
            },
            ref other => Err(InterpError::type_mismatch(
                "Deref for CString",
                "cstring aggregate",
                other.kind_name(),
            )),
        }
    }

    /// The `Box<CStr>` over the C string a fat reference
    /// denotes: the bytes themselves, terminator included,
    /// in a fresh box allocation. A stored fat reference
    /// is not a place the projection engine can index
    /// through, so the byte value is what the box's
    /// consumers (indexing, iteration, formatting) read,
    /// as for a boxed byte slice.
    pub(super) fn boxed_c_str(
        &self,
        frames: &[Frame],
        fat: &Value,
    ) -> InterpResult<Value> {
        let bytes = self.arg_bytes(frames, fat)?;
        let boxed = self.new_box(Value::Bytes(bytes));
        if let Some(ty) = ty_from_tag("std::ffi::CStr") {
            self.record_box_ty(&boxed, ty);
        }
        Ok(boxed)
    }

    /// The bytes a `&CStr` receiver denotes, without the
    /// terminating NUL: the receiver is the fat byte slice
    /// the C string was built from, terminator included, so
    /// the last byte (always a NUL in a well-formed value)
    /// is dropped.
    pub(super) fn c_str_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        let mut bytes = self.arg_bytes(frames, value)?;
        if bytes.last() == Some(&0) {
            bytes.pop();
        }
        Ok(bytes)
    }
}

/// Registration fragment of the builtin name registry: the C string and OS string surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn ffi_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "core::ffi::c_str::strlen::runtime::strlen" => Builtin::LibcStrlen,
        "std::ffi::CStr::from_bytes_with_nul" | "core::ffi::CStr::from_bytes_with_nul" => {
            Builtin::CStrFromBytesWithNul
        }
        "std::ffi::CStr::from_bytes_until_nul" | "core::ffi::CStr::from_bytes_until_nul" => {
            Builtin::CStrFromBytesUntilNul
        }
        "std::ffi::CStr::to_str" | "core::ffi::CStr::to_str" => Builtin::CStrToStr,
        "alloc::ffi::c_str::<impl std::ffi::CStr>::to_string_lossy" => {
            Builtin::CStrToStringLossy
        }
        "std::ffi::CString::new::SpecNewImpl::spec_new_impl" => {
            Builtin::CStringSpecNewImpl
        }
        "std::ffi::CString::_from_vec_unchecked"
        | "std::ffi::CString::from_vec_unchecked" => {
            Builtin::CStringFromVecUnchecked
        }
        "std::ffi::CString::from_raw" => Builtin::CStringFromRaw,
        "std::ffi::CString::into_string" => Builtin::CStringIntoString,
        "std::ffi::CString::into_boxed_c_str" => Builtin::CStringIntoBoxedCStr,
        "std::ffi::OsStr::to_ascii_lowercase" => Builtin::OsStrToAsciiLowercase,
        "std::ffi::OsStr::make_ascii_lowercase" => {
            Builtin::OsStrMakeAsciiCase(false)
        }
        "std::ffi::OsStr::make_ascii_uppercase" => {
            Builtin::OsStrMakeAsciiCase(true)
        }
        "std::ffi::OsString::to_string_lossy" => Builtin::ToStringLossy,
        "std::ffi::OsString::into_boxed_os_str" => Builtin::OsStringIntoBoxedOsStr,
        "std::ffi::OsStr::to_string_lossy" => Builtin::ToStringLossy,
        "std::path::Path::to_string_lossy" => Builtin::ToStringLossy,
        _ => return None,
    })
}
