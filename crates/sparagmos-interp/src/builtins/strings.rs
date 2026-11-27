//! String and byte-slice resolution for the string builtins.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter, Resolved,
};
use crate::ir::{Body, Operand, Ty};
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
    /// The two `n`-byte ranges of a `compare_bytes` call,
    /// settled jointly: each side's element width is judged
    /// from its own evidence (see [`Self::byte_range_width`])
    /// and the two judgements combined (see
    /// `compare_byte_widths`) before either range is read, so
    /// a side whose storage records no element type (a `Vec`
    /// buffer block, a constant aggregate) is still read at
    /// the width the other side's type records.
    pub(super) fn read_compare_ranges(
        &self,
        frames: &[Frame],
        a_ptr: &Value,
        b_ptr: &Value,
        n: usize,
    ) -> InterpResult<(Vec<u8>, Vec<u8>)> {
        let a_evidence = self.byte_range_width(frames, a_ptr, n);
        let b_evidence = self.byte_range_width(frames, b_ptr, n);
        let (a_width, b_width) = compare_byte_widths(a_evidence, b_evidence);
        let a = self.read_byte_range(frames, a_ptr, n, a_width)?;
        let b = self.read_byte_range(frames, b_ptr, n, b_width)?;
        Ok((a, b))
    }

    /// The element width evidence for a `compare_bytes` range
    /// over `ptr`, as a `(recorded, derived)` pair (see
    /// [`Self::recorded_seq_elem_width`] and
    /// [`Self::derived_seq_elem_width`]). Either is `None`
    /// when its evidence is unavailable -- including when the
    /// pointer does not resolve at all, which the range read
    /// itself then reports.
    fn byte_range_width(
        &self,
        frames: &[Frame],
        ptr: &Value,
        n: usize,
    ) -> (Option<usize>, Option<usize>) {
        let Ok(target) = self.deref_value(frames, ptr) else {
            return (None, None);
        };
        let recorded = self.recorded_seq_elem_width(frames, &target);
        let derived = self.derived_seq_elem_width(frames, &target, n);
        (recorded, derived)
    }

    /// The element width of the sequence `target` denotes an
    /// element of, from the type recorded for the target's
    /// root (a place's local type, a static's declared type,
    /// an allocation's declared type), walked through the
    /// projection prefix that denotes the sequence itself --
    /// the whole projection, minus a trailing index or
    /// subslice step that denotes the element the target
    /// points at. `Some` only when the walk ends at an array
    /// or slice of scalar elements of known size. A constant
    /// target records no type, and ADT field steps are not
    /// followed (the limits `sequence_elem_width` also has).
    fn recorded_seq_elem_width(&self, frames: &[Frame], target: &Resolved) -> Option<usize> {
        let (root, projection): (Ty, &[RefProj]) = match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let frame = frames.get(*frame_pos)?;
                let body = self.program().functions.get(&frame.func)?;
                (body.local_tys.get(*local)?.clone(), projection)
            }
            Resolved::Static { name, projection } => {
                (self.program().static_tys.get(name)?.clone(), projection)
            }
            Resolved::Alloc { id, projection } => (self.alloc_ty(*id)?, projection),
            Resolved::Value(..) => return None,
        };
        let prefix = match projection.last() {
            Some(RefProj::Index(_)) | Some(RefProj::Subslice { .. }) => {
                &projection[..projection.len() - 1]
            }
            _ => projection,
        };
        let mut cur = root;
        for step in prefix {
            cur = match step {
                RefProj::Field(k) => match cur {
                    Ty::Tuple(ts) => ts.get(*k)?.clone(),
                    _ => return None,
                },
                RefProj::Index(_) => match cur {
                    Ty::Array(elem, _) | Ty::Slice(elem) => *elem,
                    _ => return None,
                },
                RefProj::Subslice { .. } => match cur {
                    Ty::Array(..) | Ty::Slice(..) => cur,
                    _ => return None,
                },
                _ => return None,
            };
        }
        match cur {
            Ty::Array(elem, _) | Ty::Slice(elem) => match *elem {
                Ty::Int(_) | Ty::Uint(_) | Ty::Float(_) | Ty::Bool => self
                    .ty_size(&elem)
                    .map(|size| size as usize)
                    .filter(|size| *size > 0),
                _ => None,
            },
            _ => None,
        }
    }

    /// The element width of the sequence `target` denotes,
    /// derived from the range itself: the sequence holds
    /// `count` elements from the denoted position, so a byte
    /// range of `n` covering exactly those elements has
    /// elements of `n / count` bytes. `Some` only when the
    /// count is positive, divides `n`, yields a plausible
    /// scalar width, and the first element actually reads as
    /// a scalar: a sequence of aggregates never derives a
    /// width, and neither does a range that stops partway
    /// through its sequence (a prefix range over a longer
    /// sequence keeps the plain byte read, and its error).
    fn derived_seq_elem_width(
        &self,
        frames: &[Frame],
        target: &Resolved,
        n: usize,
    ) -> Option<usize> {
        if n == 0 {
            return None;
        }
        let (seq_target, start, exact) = seq_head_target(target);
        let count = match exact {
            Some(len) => len,
            None => {
                let seq = self.read_resolved(frames, &seq_target).ok()?;
                let total = usize::try_from(seq.len()?).ok()?;
                total.checked_sub(start)?
            }
        };
        if count == 0 || n % count != 0 {
            return None;
        }
        let width = n / count;
        match self.read_seq_at(frames, target, 0) {
            // A width of 1 needs no derivation for integer
            // elements (the plain byte read already serves
            // them); it is derived only for boolean elements,
            // whose values the byte read cannot serialise.
            Ok(Value::Bool(_)) => {
                if width == 1 {
                    Some(width)
                } else {
                    None
                }
            }
            Ok(Value::Int(_) | Value::Uint(_) | Value::Float(_))
                if matches!(width, 2 | 4 | 8 | 16) =>
            {
                Some(width)
            }
            _ => None,
        }
    }

    /// The `n` bytes starting at a raw byte pointer, for
    /// `compare_bytes`: string and byte data read directly, any
    /// other aggregate element by element. The intrinsic's
    /// pointers are byte-denominated (`*const u8`) whatever
    /// storage they denote, so a range over a sequence of
    /// wider elements is read element by element at the
    /// settled `width`, each element serialised to its
    /// little-endian bytes: reading element `i` for byte `i`
    /// instead walks off the end of a short wider-element
    /// sequence (the uninitialised spare-capacity slots of a
    /// `Vec` buffer block, or past the end of a constant
    /// array). Without a settled width the elements are read
    /// as bytes, the behaviour byte-denominated storage has
    /// always had.
    pub(super) fn read_byte_range(
        &self,
        frames: &[Frame],
        ptr: &Value,
        n: usize,
        width: Option<usize>,
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
        if let Some(w) = width.filter(|w| *w > 0 && n % *w == 0) {
            let mut out = Vec::with_capacity(n);
            for e in 0..(n / w) {
                let elem = self.read_seq_at(frames, &target, e)?;
                out.extend_from_slice(&compare_element_bytes(&elem, w)?);
            }
            return Ok(out);
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
        // The same owned values in their real aggregate form
        // (a buffer the converted bodies filled through
        // raw-pointer writes, as `OsString::with_capacity`
        // plus `push` builds) read through the accessor that
        // follows the buffer pointer.
        if let Some(bytes) = self.os_string_bytes_any(frames, &resolved0)? {
            return Ok(bytes);
        }
        if let Some(bytes) = self.path_buf_bytes_any(frames, &resolved0)? {
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
    /// Mutate the text of the string stored at `target`, whatever
    /// form the stored value takes: the abstract string form
    /// mutates where it sits; a `String` in an aggregate form --
    /// the real `Vec` aggregate a converted body builds field by
    /// field, or the element form the return-boundary
    /// translation produces -- is read through the same
    /// coercions the string readers use (see
    /// `resolve_string`) and stored back in the abstract form.
    /// Any other shape is the type mismatch `name` reports.
    pub(super) fn modify_string(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        name: &str,
        f: impl FnOnce(&mut String) -> InterpResult<Value>,
    ) -> InterpResult<Value> {
        let stored = self.read_resolved(frames, target)?;
        if matches!(stored, Value::Str(_)) {
            return self.modify_resolved(frames, target, |slot| match slot {
                Value::Str(s) => f(s),
                other => Err(InterpError::type_mismatch(
                    name,
                    "string value",
                    other.kind_name(),
                )),
            });
        }
        let text = self.string_text_of_stored(frames, &stored)?;
        let mut text = text.ok_or_else(|| {
            InterpError::type_mismatch(name, "string value", stored.kind_name())
        })?;
        let result = f(&mut text)?;
        self.write_resolved(frames, target, Value::Str(text))?;
        Ok(result)
    }

    /// The text a stored (non-abstract) string value denotes,
    /// through the same reader composition `resolve_string`
    /// applies to operand values: the real aggregate form, the
    /// element form (the shape a `join` / `concat` result takes
    /// once it is stored in a place -- the vector as an element
    /// list whose elements past the first keep the
    /// `MaybeUninit` union wrapper the buffer's spare-capacity
    /// region stores), and the plain aggregate byte shapes.
    /// Any other shape returns `None` and leaves the caller's
    /// ordinary judgement to stand.
    pub(super) fn string_text_of_stored(
        &self,
        frames: &[Frame],
        stored: &Value,
    ) -> InterpResult<Option<String>> {
        if let Some(text) = self.string_from_real_aggregate(frames, stored)? {
            return Ok(Some(text));
        }
        if let Some(text) = self.string_from_element_form(stored) {
            return Ok(Some(text));
        }
        Ok(string_aggregate_bytes(stored)
            .and_then(|bytes| String::from_utf8(bytes).ok()))
    }

    /// The text of the string stored at `target`, read through
    /// the same coercions `modify_string` applies (the abstract
    /// form as-is, an aggregate form through the string
    /// readers). The store is not changed.
    pub(super) fn string_text_at(
        &self,
        frames: &[Frame],
        target: &Resolved,
        name: &str,
    ) -> InterpResult<String> {
        let stored = self.read_resolved(frames, target)?;
        if let Value::Str(text) = stored {
            return Ok(text);
        }
        let text = self.string_text_of_stored(frames, &stored)?;
        text.ok_or_else(|| {
            InterpError::type_mismatch(name, "string value", stored.kind_name())
        })
    }

    /// `String::retain(&mut String, F)`: keep the chars the
    /// predicate accepts, calling it once per char in order.
    /// The predicate operand is evaluated in place (never
    /// moved out of its place), exactly as the iterator
    /// consumers evaluate theirs, so an `FnMut` predicate's
    /// captures persist across the calls. The write-back
    /// mirrors std's two phases: until the first rejected
    /// char nothing is written (a predicate panic there
    /// leaves the string unchanged, as std's read-only scan
    /// does); from the first rejection on, the retained
    /// prefix is committed even when the predicate panics
    /// (std's panic guard truncates to exactly that prefix).
    pub(super) fn exec_string_retain(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        args: &[Operand],
    ) -> InterpResult<Value> {
        if args.len() != 2 {
            return Err(InterpError::type_mismatch(
                "builtin String::retain",
                "2 argument(s)",
                format!("{} argument(s)", args.len()),
            ));
        }
        let self_ref = self.eval_operand(frames, body, &args[0])?;
        let target = self.deref_value(frames, &self_ref)?;
        let text = self.string_text_at(frames, &target, "String::retain")?;
        let mut kept = String::new();
        let mut rejected = false;
        let mut failure: Option<InterpError> = None;
        for c in text.chars() {
            self.tick_step()?;
            match self.call_callable_operand_to_completion(
                frames,
                body,
                &args[1],
                vec![Value::Uint(u128::from(c as u32))],
            ) {
                Ok(Value::Bool(true)) => kept.push(c),
                Ok(Value::Bool(false)) => rejected = true,
                Ok(other) => {
                    return Err(InterpError::type_mismatch(
                        "String::retain predicate result",
                        "a boolean",
                        other.kind_name(),
                    ));
                }
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        match failure {
            None => {
                self.write_resolved(frames, &target, Value::Str(kept))?;
                Ok(Value::Unit)
            }
            Some(e) => {
                if rejected {
                    let _ = self.write_resolved(frames, &target, Value::Str(kept));
                }
                Err(e)
            }
        }
    }

    /// The effect of dropping a `std::string::Drain` value
    /// (the `StringDrainDrop` drop plan): remove the drain's
    /// byte range from the string its first field points at.
    /// std's destructor performs the removal with a
    /// `Vec::drain` over the range, guarded by
    /// `start <= end <= len` against the string's current
    /// length; here the removal is the same range deletion
    /// on the model's string, through the mutators' coercions.
    /// The remaining fields have no drop effect (the `Chars`
    /// state iterates a snapshot; the bounds are scalars).
    pub(crate) fn drop_string_drain(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
    ) -> InterpResult<()> {
        let fields = match value {
            Value::Enum { fields, .. } => fields,
            Value::Tuple(fields) => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    "string Drain drop glue",
                    "the drain's field layout",
                    other.kind_name(),
                ))
            }
        };
        if fields.len() != 4 {
            return Err(InterpError::type_mismatch(
                "string Drain drop glue",
                "[string, start, end, iter]",
                format!("{} field(s)", fields.len()),
            ));
        }
        let start = expect_index(&fields[1], "String::drain")?;
        let end = expect_index(&fields[2], "String::drain")?;
        let target = self.deref_value(frames, &fields[0])?;
        self.modify_string(frames, &target, "String::drain", |text| {
            if start <= end && end <= text.len() {
                text.replace_range(start..end, "");
            }
            Ok(Value::Unit)
        })?;
        Ok(())
    }

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
                // The vector may equally *be* the stored value,
                // with no `String` wrapper around it: the shape
                // `String::as_mut_vec` leaves in the receiver's
                // storage, whose returned reference denotes the
                // storage itself, so the vector operations
                // mutate the element form directly (the sparse
                // form a growth leaves included).
                if let Some(text) = self.string_from_element_form(other) {
                    return Ok(text);
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
        // A one-field wrapper around string data itself: the
        // aggregate a `String` value takes when it crosses a
        // raw-storage move inside converted code (the sort
        // machinery element rotation, above all), whose
        // only field holds the string in its data
        // form -- the shape the store for a place typed exactly
        // `String` unwraps again (see `normalize_for_ty`), but a
        // generic storage crossing leaves in place. The wrapper
        // denotes that string, as does the same wrapper around
        // the byte form of the data.
        if let Value::Enum {
            variant: 0, fields, ..
        } = cur
        {
            if fields.len() == 1 {
                match &fields[0] {
                    Value::Str(text) => return Some(text.clone()),
                    Value::Bytes(bytes) => {
                        return String::from_utf8(bytes.clone()).ok();
                    }
                    _ => {}
                }
            }
        }
        // The byte form itself: a `str` value read out of
        // byte storage (the tail field of a DST struct over a
        // byte array, above all) resolves to the storage's
        // bytes, which are the string's bytes under the
        // field's declared type.
        if let Value::Bytes(bytes) = cur {
            return String::from_utf8(bytes.clone()).ok();
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
            // An array value is a byte sequence in the same
            // sense as a vector's element list: the storage a
            // DST struct's `str` tail field overlays is a sized
            // byte array in the sized twin struct, and reading
            // the tail through the fat pointer yields that
            // array's value, whose elements are the string's
            // bytes.
            Value::Tuple(items) => (items, items.len()),
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

/// The value `std::str::pattern::StrSearcher::new` builds:
/// core's `StrSearcher { haystack, needle, searcher }` in its
/// real field shape. The constructor's MIR is not encoded in
/// core's rlib (it carries no `#[inline]`), so the model
/// fills the three `StrSearcherImpl` payloads exactly as
/// core does: an empty needle gets the `Empty` payload with
/// both match flags set, a one-byte needle the `Byte`
/// payload, and any longer needle the `TwoWay` payload whose
/// constant fields are the critical factorization core's own
/// constructor computes (see `two_way_searcher_fields`).
pub(super) fn str_searcher_value(haystack: &str, needle: &str) -> Value {
    let end = Value::Uint(haystack.len() as u128);
    let searcher = if needle.is_empty() {
        Value::Enum {
            variant: 0,
            fields: vec![Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Uint(0),
                    end,
                    Value::Bool(true),
                    Value::Bool(true),
                    Value::Bool(false),
                ],
            }],
        }
    } else if let &[b] = needle.as_bytes() {
        Value::Enum {
            variant: 1,
            fields: vec![Value::Enum {
                variant: 0,
                fields: vec![Value::Uint(u128::from(b)), Value::Uint(0), end],
            }],
        }
    } else {
        Value::Enum {
            variant: 2,
            fields: vec![Value::Enum {
                variant: 0,
                fields: two_way_searcher_fields(needle.as_bytes(), haystack.len()),
            }],
        }
    };
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Str(haystack.to_string()),
            Value::Str(needle.to_string()),
            searcher,
        ],
    }
}

/// The fields of a `TwoWaySearcher` for `needle` over a
/// haystack of length `end`, in core's field order, computed
/// by the critical-factorization preprocessing of core's
/// `TwoWaySearcher::new`: the maximal-suffix factorization,
/// the short-period case's separate reverse factorization,
/// and the byte set of the period -- or of the whole needle
/// in the long-period case, whose period is the lower bound
/// and whose memory fields carry the `usize::MAX` dummy
/// that marks it.
fn two_way_searcher_fields(needle: &[u8], end: usize) -> Vec<Value> {
    let (crit_pos_false, period_false) = maximal_suffix(needle, false);
    let (crit_pos_true, period_true) = maximal_suffix(needle, true);
    let (crit_pos, period) = if crit_pos_false > crit_pos_true {
        (crit_pos_false, period_false)
    } else {
        (crit_pos_true, period_true)
    };
    let (crit_pos_back, period, byteset, memory, memory_back) =
        if needle[..crit_pos] == needle[period..period + crit_pos] {
            // Short period case -- the period is exact.
            let crit_pos_back = needle.len()
                - Ord::max(
                    reverse_maximal_suffix(needle, period, false),
                    reverse_maximal_suffix(needle, period, true),
                );
            (
                crit_pos_back,
                period,
                byteset_create(&needle[..period]),
                0,
                needle.len(),
            )
        } else {
            // Long period case -- the period is approximated
            // and no memorization is used.
            (
                crit_pos,
                Ord::max(crit_pos, needle.len() - crit_pos) + 1,
                byteset_create(needle),
                usize::MAX,
                usize::MAX,
            )
        };
    vec![
        Value::Uint(crit_pos as u128),
        Value::Uint(crit_pos_back as u128),
        Value::Uint(period as u128),
        Value::Uint(u128::from(byteset)),
        Value::Uint(0),
        Value::Uint(end as u128),
        Value::Uint(memory as u128),
        Value::Uint(memory_back as u128),
    ]
}

/// Core `TwoWaySearcher::byteset_create`: bit `b & 63` for
/// each byte `b` of the slice.
fn byteset_create(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0u64, |a, &b| (1u64 << (b & 0x3f)) | a)
}

/// Core `TwoWaySearcher::maximal_suffix`: the starting
/// index and period of the maximal suffix of `arr` under
/// the byte order `order_greater` selects.
fn maximal_suffix(arr: &[u8], order_greater: bool) -> (usize, usize) {
    let mut left = 0;
    let mut right = 1;
    let mut offset = 0;
    let mut period = 1;
    while let Some(&a) = arr.get(right + offset) {
        let b = arr[left + offset];
        if (a < b && !order_greater) || (a > b && order_greater) {
            right += offset + 1;
            offset = 0;
            period = right - left;
        } else if a == b {
            if offset + 1 == period {
                right += offset + 1;
                offset = 0;
            } else {
                offset += 1;
            }
        } else {
            left = right;
            right += 1;
            offset = 0;
            period = 1;
        }
    }
    (left, period)
}

/// Core `TwoWaySearcher::reverse_maximal_suffix`: the
/// maximal suffix computation over the reverse of `arr`,
/// stopping early once a period of `known_period` is
/// reached; returns the factorization index counted from
/// the back.
fn reverse_maximal_suffix(arr: &[u8], known_period: usize, order_greater: bool) -> usize {
    let mut left = 0;
    let mut right = 1;
    let mut offset = 0;
    let mut period = 1;
    let n = arr.len();
    while right + offset < n {
        let a = arr[n - (1 + right + offset)];
        let b = arr[n - (1 + left + offset)];
        if (a < b && !order_greater) || (a > b && order_greater) {
            right += offset + 1;
            offset = 0;
            period = right - left;
        } else if a == b {
            if offset + 1 == period {
                right += offset + 1;
                offset = 0;
            } else {
                offset += 1;
            }
        } else {
            left = right;
            right += 1;
            offset = 0;
            period = 1;
        }
        if period == known_period {
            break;
        }
    }
    left
}

/// The target denoting the sequence `target` denotes an
/// element of, the index of that element in it, and the
/// element count from that index when a step records it
/// exactly (a subslice step carries its own length). A
/// target whose projection ends in neither an index nor a
/// subslice step denotes the sequence itself, at index 0.
fn seq_head_target(target: &Resolved) -> (Resolved, usize, Option<usize>) {
    fn stepped(projection: &[RefProj]) -> Option<(Vec<RefProj>, usize, Option<usize>)> {
        match projection.last() {
            Some(RefProj::Index(k)) => {
                Some((projection[..projection.len() - 1].to_vec(), *k, None))
            }
            Some(RefProj::Subslice { offset, len }) => Some((
                projection[..projection.len() - 1].to_vec(),
                *offset,
                Some(*len),
            )),
            _ => None,
        }
    }
    match target {
        Resolved::Place {
            frame_pos,
            local,
            projection,
        } => match stepped(projection) {
            Some((head, start, exact)) => (
                Resolved::Place {
                    frame_pos: *frame_pos,
                    local: *local,
                    projection: head,
                },
                start,
                exact,
            ),
            None => (target.clone(), 0, None),
        },
        Resolved::Static { name, projection } => match stepped(projection) {
            Some((head, start, exact)) => (
                Resolved::Static {
                    name: name.clone(),
                    projection: head,
                },
                start,
                exact,
            ),
            None => (target.clone(), 0, None),
        },
        Resolved::Alloc { id, projection } => match stepped(projection) {
            Some((head, start, exact)) => (
                Resolved::Alloc {
                    id: *id,
                    projection: head,
                },
                start,
                exact,
            ),
            None => (target.clone(), 0, None),
        },
        Resolved::Value(base, projection) => match stepped(projection) {
            Some((head, start, exact)) => (Resolved::Value(base.clone(), head), start, exact),
            None => (target.clone(), 0, None),
        },
    }
}

/// Settle the element widths of a `compare_bytes` call's two
/// ranges from each side's `(recorded, derived)` evidence: a
/// recorded width is the storage's own denomination and wins
/// for its side; a side with no recorded width takes the
/// other side's recorded width unless its own derivation
/// disagrees, because the call compares two ranges of one
/// element type (slice equality is the caller that produces
/// these ranges) while a derivation is only as strong as the
/// sequence it counted. With no recorded width on either
/// side, a derived width settles both sides when the two
/// derivations agree or only one side derives one; a
/// disagreement leaves both sides unsettled, on the plain
/// byte read.
fn compare_byte_widths(
    a: (Option<usize>, Option<usize>),
    b: (Option<usize>, Option<usize>),
) -> (Option<usize>, Option<usize>) {
    let (a_rec, a_der) = a;
    let (b_rec, b_der) = b;
    match (a_rec, b_rec) {
        (Some(x), Some(y)) => (Some(x), Some(y)),
        (Some(x), None) => (
            Some(x),
            match b_der {
                Some(d) if d != x => Some(d),
                _ => Some(x),
            },
        ),
        (None, Some(y)) => (
            match a_der {
                Some(d) if d != y => Some(d),
                _ => Some(y),
            },
            Some(y),
        ),
        (None, None) => match (a_der, b_der) {
            (Some(x), Some(y)) if x == y => (Some(x), Some(y)),
            (Some(x), None) => (Some(x), Some(x)),
            (None, Some(y)) => (Some(y), Some(y)),
            _ => (None, None),
        },
    }
}

/// The little-endian bytes of one scalar element of a
/// `compare_bytes` range, at the element width the range is
/// denominated in: an integer contributes the low `width`
/// bytes of its two's-complement word, a float its bit
/// pattern at its own width (the model holds floats widened
/// to `f64`, so a width-4 element narrows back), a boolean
/// its single byte. Anything else is not an element a byte
/// comparison can serialise.
fn compare_element_bytes(value: &Value, width: usize) -> InterpResult<Vec<u8>> {
    match value {
        Value::Int(v) => {
            if width > 16 {
                return Err(InterpError::type_mismatch(
                    "compare_bytes element",
                    "integer element of width at most 16",
                    format!("width {width}"),
                ));
            }
            Ok((*v as u128).to_le_bytes()[..width].to_vec())
        }
        Value::Uint(v) => {
            if width > 16 {
                return Err(InterpError::type_mismatch(
                    "compare_bytes element",
                    "integer element of width at most 16",
                    format!("width {width}"),
                ));
            }
            Ok(v.to_le_bytes()[..width].to_vec())
        }
        Value::Float(v) => match width {
            4 => Ok((*v as f32).to_bits().to_le_bytes().to_vec()),
            8 => Ok(v.to_bits().to_le_bytes().to_vec()),
            _ => Err(InterpError::type_mismatch(
                "compare_bytes element",
                "float element of width 4 or 8",
                format!("width {width}"),
            )),
        },
        Value::Bool(b) if width == 1 => Ok(vec![u8::from(*b)]),
        other => Err(InterpError::type_mismatch(
            "compare_bytes element",
            "scalar element",
            other.kind_name(),
        )),
    }
}

/// Registration fragment of the builtin name registry: the UTF-8 conversion and case-mapping surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn strings_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::str::from_utf8" | "core::str::from_utf8" => Builtin::FromUtf8,
        _ if canonical.contains("RangeInclusive") && canonical.ends_with("::new") => {
            Builtin::RangeInclusiveNew
        }
        _ if canonical.ends_with("::to_uppercase") => Builtin::ToUppercase,
        _ if canonical.ends_with("::to_lowercase") => Builtin::ToLowercase,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the string search surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn str_search_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::str::pattern::StrSearcher::new" => Builtin::StrSearcherNew,
        "core::slice::memchr::memchr" => Builtin::Memchr { reverse: false },
        "core::slice::memchr::memrchr" => Builtin::Memchr { reverse: true },
        "core::slice::index::RangeError::report" => Builtin::RangeErrorReport,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the `String` surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn string_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::string::ToString::to_string" | "alloc::string::ToString::to_string" => Builtin::ToStringToString,
        // The `String` / `str` surface the abstract string value
        // can answer directly. Registering a name here also keeps
        // the driver from collecting the std body (see
        // sparagmos-driver's reach pass), whose field projections
        // on the buffer struct the abstract value cannot serve.
        "std::string::String::new" | "alloc::string::String::new" => Builtin::StringNew,
        "std::string::String::len" | "alloc::string::String::len" => Builtin::StringLen,
        "std::string::String::is_empty" | "alloc::string::String::is_empty" => {
            Builtin::StringIsEmpty
        }
        "std::string::String::capacity" | "alloc::string::String::capacity" => {
            Builtin::StringCapacity
        }
        "std::string::String::with_capacity" | "alloc::string::String::with_capacity" => {
            Builtin::StringWithCapacity
        }
        "std::string::String::as_str" | "alloc::string::String::as_str" => Builtin::StringAsStr,
        "std::string::String::as_bytes" | "alloc::string::String::as_bytes" => {
            Builtin::StringAsBytes
        }
        "std::string::String::into_bytes" | "alloc::string::String::into_bytes" => {
            Builtin::StringIntoBytes
        }
        "std::string::String::as_mut_vec" | "alloc::string::String::as_mut_vec" => {
            Builtin::StringAsMutVec
        }
        "std::string::String::leak" | "alloc::string::String::leak" => {
            Builtin::StringLeak
        }
        "std::string::String::into_boxed_str" | "alloc::string::String::into_boxed_str" => {
            Builtin::StringIntoBoxedStr
        }
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the `String` mutators.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn string_mut_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::string::String::push_str" | "alloc::string::String::push_str" => {
            Builtin::StringPushStr
        }
        "std::string::String::push" | "alloc::string::String::push" => Builtin::StringPush,
        "std::string::String::insert" | "alloc::string::String::insert" => {
            Builtin::StringInsert
        }
        "std::string::String::insert_str" | "alloc::string::String::insert_str" => {
            Builtin::StringInsertStr
        }
        "std::string::String::remove" | "alloc::string::String::remove" => {
            Builtin::StringRemove
        }
        "std::string::String::truncate" | "alloc::string::String::truncate" => {
            Builtin::StringTruncate
        }
        "std::string::String::pop" | "alloc::string::String::pop" => Builtin::StringPop,
        "std::string::String::clear" | "alloc::string::String::clear" => {
            Builtin::StringClear
        }
        "std::string::String::retain" | "alloc::string::String::retain" => {
            Builtin::StringRetain
        }
        "std::string::String::replace_range" | "alloc::string::String::replace_range" => {
            Builtin::StringReplaceRange
        }
        "std::string::String::split_off" | "alloc::string::String::split_off" => {
            Builtin::StringSplitOff
        }
        "std::string::String::reserve" | "alloc::string::String::reserve"
        | "std::string::String::reserve_exact" | "alloc::string::String::reserve_exact" => {
            Builtin::StringReserve
        }
        "std::string::String::try_reserve" | "alloc::string::String::try_reserve"
        | "std::string::String::try_reserve_exact"
        | "alloc::string::String::try_reserve_exact" => Builtin::StringTryReserve,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the `str` surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn str_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "core::str::<impl str>::len" => Builtin::StrLen,
        "core::str::<impl str>::is_empty" => Builtin::StrIsEmpty,
        "core::str::<impl str>::as_bytes" => Builtin::StrAsBytes,
        "core::str::<impl str>::as_bytes_mut" => Builtin::StrAsBytesMut,
        "core::str::<impl str>::chars" => Builtin::StrChars,
        "core::str::<impl str>::bytes" => Builtin::StrBytes,
        "core::str::<impl str>::make_ascii_uppercase" => {
            Builtin::StrMakeAsciiCase(true)
        }
        "core::str::<impl str>::make_ascii_lowercase" => {
            Builtin::StrMakeAsciiCase(false)
        }
        "core::str::<impl str>::trim" => Builtin::StrTrim(TrimSide::Both),
        "core::str::<impl str>::trim_start" => Builtin::StrTrim(TrimSide::Start),
        "core::str::<impl str>::trim_end" => Builtin::StrTrim(TrimSide::End),
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the `str` predicates.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn str_predicate_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        _ if canonical.starts_with("core::str::<impl") && canonical.ends_with(">::contains") => {
            Builtin::StrContains
        }
        _ if canonical.starts_with("core::str::<impl")
            && canonical.ends_with(">::starts_with") =>
        {
            Builtin::StrStartsWith
        }
        _ if canonical.starts_with("core::str::<impl") && canonical.ends_with(">::ends_with") => {
            Builtin::StrEndsWith
        }
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the `FromStr` surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn from_str_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::str::FromStr::from_str" | "core::str::FromStr::from_str" => {
            Builtin::FromStrParse
        }
        _ => return None,
    })
}
