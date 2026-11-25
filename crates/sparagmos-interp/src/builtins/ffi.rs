//! C string and byte argument marshalling for the FFI builtins.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter,
};
use crate::value::Value;


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
