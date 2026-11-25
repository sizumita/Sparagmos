//! C string and byte argument marshalling for the FFI builtins.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter,
};
use crate::value::{RefProj, RefTarget, Value};

use super::fmt::pointee_tag;


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
