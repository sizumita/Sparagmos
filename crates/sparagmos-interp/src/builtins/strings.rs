//! String and byte-slice resolution for the string builtins.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter, Resolved,
};
use crate::value::{FmtArguments, RefProj, RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// Validate a `strto*` call and read its operands: the
    /// string argument as a host C string and the `base`
    /// argument. The delegated host call passes a NULL
    /// `endptr`, so a non-NULL guest `endptr` — whose
    /// write-back into guest storage is not modelled — is
    /// an explicit error, never a silently skipped write.
    pub(super) fn strto_args(
        &self,
        frames: &[Frame],
        name: &str,
        vals: &[Value],
    ) -> InterpResult<(std::ffi::CString, std::ffi::c_int)> {
        if vals.len() != 3 {
            return Err(unsupported(format!(
                "call to {name}: expected 3 arguments, got {}",
                vals.len()
            )));
        }
        let endptr = unwrap_wrapper(&vals[1]);
        let endptr_is_null = matches!(
            endptr,
            Value::Ref(RefTarget::Null)
                | Value::Ref(RefTarget::Address(0))
                | Value::Uint(0)
                | Value::Int(0)
        );
        if !endptr_is_null {
            return Err(unsupported(format!(
                "call to {name}: a non-NULL endptr is not modelled"
            )));
        }
        let base = as_i128(&vals[2], name)? as std::ffi::c_int;
        let arg = self.c_str_arg(frames, &vals[0])?;
        Ok((arg, base))
    }

    /// The `&[u8]` value denoting a string value's bytes: a fat
    /// reference over the receiver itself when the receiver is a
    /// reference (element reads resolve through it to the string
    /// data, as for a `Vec` deref), over the receiver's own base
    /// when it is already a fat reference, or over a snapshot of
    /// the bytes when the receiver is the string data itself (the
    /// shape of a byte-slice constant).
    /// The `n` bytes starting at a raw byte pointer, for
    /// `compare_bytes`: string and byte data read directly, any
    /// other aggregate element by element.
    pub(super) fn read_byte_range(
        &self,
        frames: &[Frame],
        ptr: &Value,
        n: usize,
    ) -> InterpResult<Vec<u8>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let target = self.deref_value(frames, ptr)?;
        let base = self.read_resolved(frames, &target)?;
        let direct: Option<&[u8]> = match &base {
            Value::Str(s) => Some(s.as_bytes()),
            Value::Bytes(b) => Some(b.as_slice()),
            _ => None,
        };
        if let Some(bytes) = direct {
            if bytes.len() < n {
                return Err(InterpError::type_mismatch(
                    "compare_bytes range",
                    "byte range of the requested length",
                    format!("{} bytes", bytes.len()),
                ));
            }
            return Ok(bytes[..n].to_vec());
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            match self.read_seq_at(frames, &target, i)? {
                Value::Uint(b) => out.push(b as u8),
                // A signed byte element (an `i8` holding a byte
                // above 127) keeps its bit pattern: the range
                // runs from -128, not 0, as in `read_c_str`.
                Value::Int(b) if (-128..=255).contains(&b) => out.push(b as u8),
                other => {
                    return Err(InterpError::type_mismatch(
                        "compare_bytes element",
                        "byte",
                        other.kind_name(),
                    ));
                }
            }
        }
        Ok(out)
    }

    pub(super) fn string_bytes_ref(&self, frames: &[Frame], receiver: &Value) -> InterpResult<Value> {
        let s = self.resolve_string(frames, receiver)?;
        let len = s.len() as u128;
        let base = match receiver {
            Value::Ref(_) => receiver.clone(),
            Value::FatRef { base, .. } => (**base).clone(),
            _ => Value::Tuple(
                s.as_bytes()
                    .iter()
                    .map(|b| Value::Uint(*b as u128))
                    .collect(),
            ),
        };
        Ok(Value::FatRef {
            base: Box::new(base),
            len,
        })
    }

    /// The string data a value denotes: the value itself, or the
    /// pointee of a reference to string data.
    /// The bytes an `&OsStr` argument denotes (a process
    /// program, argument, environment entry or directory): in
    /// this model an `OsStr` is its byte string.
    pub(super) fn os_str_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        // A fat reference carries the byte string's length in
        // its own metadata: read the bytes through
        // `slice_elements` so the length and the sibling bytes
        // survive. Resolving the value first would follow a
        // base that references an individual byte to that byte
        // alone, and a string of any length (including zero)
        // would then be mistaken for the one-byte string the
        // scalar shortcut below composes.
        let from_fat_ref = matches!(value, Value::FatRef { .. });
        if from_fat_ref {
            if let Some(bytes) = self.byte_slice_of_fat_ref(frames, value) {
                return Ok(bytes);
            }
        }
        let resolved0 = self.resolve_value(frames, value)?;
        // An owned byte-string aggregate in buffer form (an
        // `OsString` or `PathBuf` wrapping a `Buf` wrapping the
        // byte vector): the unwrap loop below cannot descend
        // into the byte list itself, so read the bytes off the
        // aggregate's shape first. Where both readings apply
        // they agree (the string form's data is the same bytes).
        if let Some(bytes) =
            os_string_bytes(&resolved0).or_else(|| path_buf_bytes(&resolved0))
        {
            return Ok(bytes);
        }
        let mut resolved = resolved0;
        // The path and OS-string values of this model wrap the
        // string data in one-field aggregates (`PathBuf`
        // around `OsString` around the string itself, an owned
        // `Cow<OsStr>` around its `OsString`): unwrap those
        // layers to the data they hold.
        loop {
            match resolved {
                Value::Str(s) => return Ok(s.into_bytes()),
                Value::Bytes(b) => return Ok(b),
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    resolved = fields.into_iter().next().expect("length checked");
                }
                // A one-byte string read back through a byte
                // pointer composes the scalar byte (the byte
                // view's shaped read): the lone byte value at
                // this position is the one-byte string. A fat
                // reference never takes this shortcut: its
                // metadata states the length, and the read at
                // the top either honoured it or could not read
                // the slice at all.
                Value::Uint(v) if v <= 0xff && !from_fat_ref => {
                    return Ok(vec![v as u8])
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "OsStr argument",
                        "string or byte data",
                        other.kind_name(),
                    ))
                }
            }
        }
    }
}

impl<'a> Interpreter<'a> {
    pub(super) fn resolve_string(&self, frames: &[Frame], value: &Value) -> InterpResult<String> {
        let resolved = self.resolve_value(frames, value)?;
        match resolved {
            Value::Str(s) => Ok(s),
            ref other => {
                if let Some(text) = self.string_from_real_aggregate(frames, other)? {
                    return Ok(text);
                }
                // The vector may equally arrive in its element
                // form (the shape the return-boundary
                // translation produces). This coercion is only
                // reached where a string is expected, so the
                // shape is read as the string it denotes; the
                // shape-sniffing comparisons use
                // `string_from_real_aggregate` alone and are
                // unaffected.
                if let Value::Enum {
                    variant: 0,
                    fields,
                } = other
                {
                    if fields.len() == 1 {
                        if let Some(text) = self.string_from_element_form(&fields[0]) {
                            return Ok(text);
                        }
                    }
                }
                Err(InterpError::type_mismatch(
                    "string argument",
                    "string data",
                    other.kind_name(),
                ))
            }
        }
    }

    /// The string a `String`'s vector in its element form
    /// denotes: the elements are the bytes themselves, each
    /// possibly holding the `MaybeUninit` union form the
    /// buffer's spare-capacity region stores (see
    /// `string_from_real_aggregate`). Any other shape returns
    /// `None`.
    pub(super) fn string_from_element_form(&self, vec: &Value) -> Option<String> {
        // Wrapper newtypes between the string type and its
        // byte vector (an `OsString` is a `Buf` around the
        // `Vec<u8>`) nest single-field variant-0 aggregates
        // around the element list. Descend through them: a
        // byte vector's element list cannot itself consist of
        // one aggregate -- its elements are bytes (each
        // possibly in a one-field union wrapper, handled per
        // element below) -- so a lone aggregate field is a
        // wrapper around the vector, not an element of it.
        let mut cur = vec;
        loop {
            match cur {
                Value::Enum {
                    variant: 0, fields, ..
                } if fields.len() == 1
                    && matches!(
                        fields[0],
                        Value::Enum { .. } | Value::SparseVec { .. }
                    ) =>
                {
                    cur = &fields[0];
                }
                _ => break,
            }
        }
        let (elems, len): (&[Value], usize) = match cur {
            // A vector is a struct: its value carries variant
            // 0. (A one-field wrapper around an empty or
            // small aggregate of any other variant — an
            // `Option` payload, say — has a similar outline
            // and is not a byte vector.)
            Value::Enum {
                variant: 0, fields, ..
            } => (fields, fields.len()),
            Value::SparseVec { elems, len, .. } => {
                (elems, usize::try_from(*len).ok()?)
            }
            _ => return None,
        };
        let mut bytes = Vec::with_capacity(len);
        for elem in elems.iter().take(len) {
            let elem = match elem {
                Value::Enum { variant: 0, fields } if fields.len() == 1 => {
                    &fields[0]
                }
                other => other,
            };
            match elem {
                Value::Uint(b) if *b <= 0xFF => bytes.push(*b as u8),
                Value::Int(b) if (0..=255).contains(b) => bytes.push(*b as u8),
                _ => return None,
            }
        }
        String::from_utf8(bytes).ok()
    }

    /// The string a `String` value in its real aggregate form
    /// denotes: the one-field wrapper around a `Vec<u8>` in its
    /// real aggregate form (`[RawVec, len]`, see
    /// `vec_real_aggregate_parts`), whose buffer the converted
    /// library bodies filled through raw-pointer copies — the
    /// shape `slice::join` / `concat` build through
    /// `join_generic_copy` and hand back as a `String`. The bytes
    /// are the buffer pointer's elements, read through the
    /// pointer model like any other bulk-memory source. Any
    /// other shape returns `None` and leaves the caller's
    /// ordinary judgement to stand.
    pub(super) fn string_from_real_aggregate(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Option<String>> {
        let Value::Enum { fields, .. } = value else {
            return Ok(None);
        };
        if fields.len() != 1 {
            return Ok(None);
        }
        let Some((buf, len)) = vec_real_aggregate_parts(&fields[0]) else {
            return Ok(None);
        };
        if len == 0 {
            return Ok(Some(String::new()));
        }
        // The buffer value is the `RawVec` aggregate: its first
        // field is the `RawVecInner` triple, whose first field is
        // the one-field `NonNull` wrapper around the data pointer.
        let Value::Enum {
            fields: buf_fields, ..
        } = &buf
        else {
            return Ok(None);
        };
        let Some(Value::Enum {
            fields: inner_fields,
            ..
        }) = buf_fields.first()
        else {
            return Ok(None);
        };
        let Some(Value::Enum {
            fields: ptr_fields, ..
        }) = inner_fields.first()
        else {
            return Ok(None);
        };
        if ptr_fields.len() != 1 {
            return Ok(None);
        }
        let ptr = &ptr_fields[0];
        if !matches!(ptr, Value::Ref(_)) {
            return Ok(None);
        }
        let target = self.deref_value(frames, ptr)?;
        let mut bytes = Vec::with_capacity(len);
        for i in 0..len {
            let elem = self.read_seq_at(frames, &target, i)?;
            // A byte stored through the buffer's union view
            // (the spare-capacity region a `join` fills is
            // `MaybeUninit<u8>`-typed) keeps the union's
            // one-field form; the region reads as the byte it
            // holds, like every reader of such a region.
            let elem = match &elem {
                Value::Enum { variant: 0, fields } if fields.len() == 1 => &fields[0],
                _ => &elem,
            };
            match elem {
                Value::Uint(b) if *b <= 0xFF => bytes.push(*b as u8),
                Value::Int(b) if (0..=255).contains(b) => bytes.push(*b as u8),
                _ => return Ok(None),
            }
        }
        match String::from_utf8(bytes) {
            Ok(text) => Ok(Some(text)),
            Err(_) => Ok(None),
        }
    }

    /// The `fmt::Arguments` a builtin argument denotes. The
    /// modelled value comes from the `Arguments` constructor
    /// builtins; a body that builds the struct itself (an
    /// inlined `format_args!` expansion inside a library body —
    /// the shape edition-2015 `unreachable!` reaches through
    /// `core::panicking::unreachable_display`) hands over the
    /// raw aggregate instead: two wrapper fields holding the
    /// template pointer and the argument-array pointer. The
    /// aggregate comes in the two shapes `core::fmt` itself
    /// builds. In the `Arguments::new` shape the template field
    /// points at the encoded template bytes and the args field
    /// at the argument array. In the `Arguments::from_str`
    /// shape the args field is not a pointer at all but the
    /// tagged length word `(len << 1) | 1` in pointer clothing,
    /// while the template field points at the string's bytes —
    /// the encoding core's own `Arguments::as_str` decodes, and
    /// the one the `panic_nounwind_fmt` calls in core's UB
    /// checks hand over.
    pub(super) fn coerce_arguments(
        &self,
        frames: &[Frame],
        value: &Value,
        name: &str,
    ) -> InterpResult<FmtArguments> {
        let resolved = self.resolve_value(frames, value)?;
        match resolved {
            Value::FmtArguments(args) => Ok(args),
            Value::Enum { fields, .. } if fields.len() == 2 => {
                let mut fields = fields.into_iter();
                let template_field = fields.next().expect("two fields");
                let args_field = fields.next().expect("two fields");
                let template_ptr = unwrap_newtype(template_field);
                let args_ptr = unwrap_newtype(args_field);
                if let Value::Ref(RefTarget::Address(tagged)) = args_ptr {
                    if tagged & 1 == 1 {
                        let len = (tagged >> 1) as usize;
                        let bytes = self.resolve_bytes(frames, &template_ptr)?;
                        let text = bytes.get(..len).ok_or_else(|| {
                            InterpError::type_mismatch(
                                "fmt::Arguments string",
                                "byte data of the tagged length",
                                format!("{} bytes", bytes.len()),
                            )
                        })?;
                        let text = String::from_utf8(text.to_vec()).map_err(|_| {
                            InterpError::type_mismatch(
                                "fmt::Arguments string",
                                "UTF-8 string data",
                                "invalid UTF-8 bytes",
                            )
                        })?;
                        return Ok(FmtArguments::Plain(text));
                    }
                }
                let template = self.resolve_bytes(frames, &template_ptr)?;
                let args = self.resolve_arg_array(frames, &args_ptr)?;
                Ok(FmtArguments::Template { template, args })
            }
            other => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "fmt::Arguments value",
                other.kind_name(),
            )),
        }
    }

    /// The byte data a value denotes (an fmt pieces template):
    /// the bytes themselves, a string's bytes, or the data a
    /// pointer into string/byte data denotes from the pointed
    /// element on — the shape a string constant's `as_ptr` and
    /// a body-built `fmt::Arguments` template field take, where
    /// the pointer's target is the data value and its trailing
    /// index step selects the first byte. The denoted data is
    /// the target from that element, never the element itself
    /// (the single byte the plain resolution of such a pointer
    /// yields).
    pub(super) fn resolve_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        if let Value::Ref(_) = value {
            let target = self.deref_value(frames, value)?;
            if let Some(bytes) = self.byte_data_from(frames, &target)? {
                return Ok(bytes);
            }
        }
        match self.resolve_value(frames, value)? {
            Value::Bytes(b) => Ok(b),
            Value::Str(s) => Ok(s.into_bytes()),
            other => Err(InterpError::type_mismatch(
                "fmt pieces template",
                "byte data",
                other.kind_name(),
            )),
        }
    }

    /// The byte data a resolved target denotes from the element
    /// its trailing index step selects, when the value the steps
    /// before it reach is string or byte data: that value read
    /// without the final step, sliced from the selected element.
    /// Any other shape — no trailing index step, a target value
    /// that is not byte data, an element past the data's end —
    /// returns `None` and leaves the caller's ordinary
    /// resolution to judge the value.
    fn byte_data_from(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> InterpResult<Option<Vec<u8>>> {
        let (prefix, offset) = match target {
            Resolved::Place { projection, .. }
            | Resolved::Value(_, projection)
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. } => match projection.split_last() {
                Some((RefProj::Index(index), prefix)) => (prefix, *index),
                _ => return Ok(None),
            },
        };
        let trimmed = match target {
            Resolved::Place {
                frame_pos, local, ..
            } => Resolved::Place {
                frame_pos: *frame_pos,
                local: *local,
                projection: prefix.to_vec(),
            },
            Resolved::Value(base, _) => Resolved::Value(base.clone(), prefix.to_vec()),
            Resolved::Static { name, .. } => Resolved::Static {
                name: name.clone(),
                projection: prefix.to_vec(),
            },
            Resolved::Alloc { id, .. } => Resolved::Alloc {
                id: *id,
                projection: prefix.to_vec(),
            },
        };
        let base = match self.read_resolved(frames, &trimmed) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        let bytes = match base {
            Value::Bytes(bytes) => bytes,
            Value::Str(text) => text.into_bytes(),
            _ => return Ok(None),
        };
        Ok(bytes.get(offset..).map(|tail| tail.to_vec()))
    }

    /// Snapshot the elements of an fmt argument array.
    pub(super) fn resolve_arg_array(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Vec<Value>> {
        match self.resolve_value(frames, value)? {
            Value::Tuple(items) => Ok(items),
            other => Err(InterpError::type_mismatch(
                "fmt argument array",
                "array of fmt arguments",
                other.kind_name(),
            )),
        }
    }
}
