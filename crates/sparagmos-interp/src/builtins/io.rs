//! File I/O execution for the io builtins.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter,
};
use crate::value::Value;

use super::*;

impl<'a> Interpreter<'a> {
    /// The file-table id of the `std::fs::File` handle a
    /// receiver value denotes (a reference to the handle, or the
    /// handle value itself).
    pub(super) fn file_id_of(&self, frames: &[Frame], receiver: &Value) -> InterpResult<u64> {
        let value = self.resolve_value(frames, receiver)?;
        crate::interp::file_handle_id(&value).ok_or_else(|| {
            InterpError::type_mismatch(
                "file operation",
                "a std::fs::File handle",
                value.kind_name(),
            )
        })
    }

    /// Map the outcome of a host file operation to the guest
    /// `io::Result` value. A missing table entry (a handle the
    /// model never issued, or one already closed) is an explicit
    /// unsupported error, never a silent success.
    fn file_result<T>(
        &self,
        result: Option<std::io::Result<T>>,
        ok: impl FnOnce(T) -> Value,
    ) -> InterpResult<Value> {
        match result {
            Some(Ok(value)) => Ok(io_result_ok(ok(value))),
            Some(Err(error)) => Ok(io_result_err(host_io_error(&error))),
            None => Err(unsupported(
                "file operation on a handle that is not open".to_string(),
            )),
        }
    }

    /// The bytes of a byte-slice value (the buffer a `Write`
    /// call hands over).
    fn slice_bytes(
        &self,
        frames: &[Frame],
        value: &Value,
        name: &str,
    ) -> InterpResult<Vec<u8>> {
        let (items, _) = self.slice_elements(frames, value)?;
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            out.push(as_u128(&item, name)? as u8);
        }
        Ok(out)
    }

    /// Write `bytes` into the byte-slice value `fat` starting at
    /// element `offset` (the write a `Read::read` performs into
    /// its caller's buffer): the pointee's element sequence is
    /// rebuilt with the new bytes in place and stored back
    /// through the pointer's target, preserving the container's
    /// shape (array tuple, vector element list, sparse vector).
    fn write_bytes_into_slice(
        &self,
        frames: &mut Vec<Frame>,
        fat: &Value,
        offset: usize,
        bytes: &[u8],
    ) -> InterpResult<()> {
        let Value::FatRef { base, len } = fat else {
            return Err(InterpError::type_mismatch(
                "file read buffer",
                "a byte slice pointer",
                fat.kind_name(),
            ));
        };
        if offset + bytes.len() > *len as usize {
            return Err(unsupported(format!(
                "file read of {} bytes at offset {offset} exceeds the {len}-byte buffer",
                bytes.len()
            )));
        }
        let target = self.deref_value(frames, base)?;
        let current = self.read_resolved(frames, &target)?;
        let new = replace_slice_bytes(&current, offset, bytes).ok_or_else(|| {
            unsupported(format!(
                "file read into a {} buffer is not modelled",
                current.kind_name()
            ))
        })?;
        self.write_resolved(frames, &target, new)
    }

    /// Append `bytes` to the `Vec<u8>` value a `&mut Vec<u8>`
    /// argument denotes (the append `Read::read_to_end`
    /// performs).
    fn append_bytes_to_vec(
        &self,
        frames: &mut Vec<Frame>,
        arg: &Value,
        bytes: &[u8],
    ) -> InterpResult<()> {
        let target = self.deref_value(frames, arg)?;
        let current = self.read_resolved(frames, &target)?;
        let new = match current {
            Value::Enum {
                variant,
                mut fields,
            } => {
                fields.extend(bytes.iter().map(|b| Value::Uint(*b as u128)));
                Value::Enum { variant, fields }
            }
            Value::SparseVec { mut elems, len, cap } => {
                elems.extend(bytes.iter().map(|b| Value::Uint(*b as u128)));
                let len = len + bytes.len() as u128;
                let cap = cap.max(len).max(elems.len() as u128);
                Value::SparseVec { elems, len, cap }
            }
            other => {
                return Err(InterpError::type_mismatch(
                    "read_to_end destination",
                    "a Vec<u8> value",
                    other.kind_name(),
                ))
            }
        };
        self.write_resolved(frames, &target, new)
    }

    /// Execute one file-I/O builtin node (see
    /// `file_io_terminator` in `mir_convert`): a `Read` /
    /// `Write` / `Seek` method on `std::fs::File`, a `Write`
    /// method on a standard stream or a `Vec<u8>`, or one of
    /// `File`'s inherent handle methods. File operations
    /// delegate to the host file in the interpreter's file
    /// table; stream writes go to the captured output the
    /// stream denotes, and a `Vec<u8>` write appends in
    /// place. The guest-visible results are std's own shapes
    /// (the host errno in the `Os` error form, the bare-kind
    /// form for the errors std raises itself).
    pub(super) fn exec_file_io(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        generic_args: &[String],
        vals: Vec<Value>,
    ) -> InterpResult<Value> {
        let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
        let self_ty = self_ty
            .strip_prefix("&mut ")
            .or_else(|| self_ty.strip_prefix("&"))
            .unwrap_or(self_ty);
        let arity = |n: usize| -> InterpResult<()> {
            if vals.len() == n {
                Ok(())
            } else {
                Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    format!("{n} argument(s)"),
                    format!("{} argument(s)", vals.len()),
                ))
            }
        };
        let not_open = |what: &str| -> InterpError {
            unsupported(format!("{what} on a handle that is not open"))
        };
        match name {
            "std::io::Read::read" => {
                arity(2)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let Value::FatRef { len, .. } = &vals[1] else {
                    return Err(InterpError::type_mismatch(
                        "Read::read buffer",
                        "a byte slice pointer",
                        vals[1].kind_name(),
                    ));
                };
                // A single `read` may legally return short, so
                // the host buffer is capped: a sparse buffer
                // value claiming an astronomical length cannot
                // exhaust the host through one call.
                let want = (*len).min(1 << 23) as usize;
                let mut buf = vec![0u8; want];
                let read = self.with_file(id, |f| {
                    <std::fs::File as std::io::Read>::read(f, &mut buf)
                });
                match read {
                    Some(Ok(n)) => {
                        self.write_bytes_into_slice(frames, &vals[1], 0, &buf[..n])?;
                        Ok(io_result_ok(Value::Uint(n as u128)))
                    }
                    Some(Err(error)) => Ok(io_result_err(host_io_error(&error))),
                    None => Err(not_open("Read::read")),
                }
            }
            "std::io::Read::read_exact" => {
                arity(2)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let Value::FatRef { len, .. } = &vals[1] else {
                    return Err(InterpError::type_mismatch(
                        "Read::read_exact buffer",
                        "a byte slice pointer",
                        vals[1].kind_name(),
                    ));
                };
                let total = *len as usize;
                let mut buf: Vec<u8> = Vec::new();
                let mut filled = 0usize;
                // The host buffer grows only as far as the file
                // actually fills it, so a sparse buffer value
                // cannot force a huge allocation either.
                let outcome = self.with_file(id, |f| {
                    while filled < total {
                        if buf.len() < total {
                            let grow = (total - buf.len()).min(1 << 16);
                            buf.resize(buf.len() + grow, 0);
                        }
                        match <std::fs::File as std::io::Read>::read(
                            f,
                            &mut buf[filled..],
                        ) {
                            Ok(0) => break,
                            Ok(n) => filled += n,
                            Err(e)
                                if e.kind()
                                    == std::io::ErrorKind::Interrupted =>
                            {
                                continue
                            }
                            Err(e) => return Some(e),
                        }
                    }
                    None
                });
                match outcome {
                    None => Err(not_open("Read::read_exact")),
                    Some(Some(error)) => {
                        self.write_bytes_into_slice(
                            frames,
                            &vals[1],
                            0,
                            &buf[..filled],
                        )?;
                        Ok(io_result_err(host_io_error(&error)))
                    }
                    Some(None) => {
                        self.write_bytes_into_slice(
                            frames,
                            &vals[1],
                            0,
                            &buf[..filled],
                        )?;
                        if filled == total {
                            Ok(io_result_ok(Value::Unit))
                        } else {
                            Ok(io_result_err(io_simple_error_value(
                                std::io::ErrorKind::UnexpectedEof as u32,
                            )))
                        }
                    }
                }
            }
            "std::io::Read::read_to_end" => {
                arity(2)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let mut data: Vec<u8> = Vec::new();
                let read = self.with_file(id, |f| {
                    <std::fs::File as std::io::Read>::read_to_end(f, &mut data)
                });
                match read {
                    Some(Ok(_)) => {
                        let n = data.len();
                        self.append_bytes_to_vec(frames, &vals[1], &data)?;
                        Ok(io_result_ok(Value::Uint(n as u128)))
                    }
                    Some(Err(error)) => Ok(io_result_err(host_io_error(&error))),
                    None => Err(not_open("Read::read_to_end")),
                }
            }
            "std::io::Read::read_to_string" => {
                arity(2)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let mut data: Vec<u8> = Vec::new();
                let read = self.with_file(id, |f| {
                    <std::fs::File as std::io::Read>::read_to_end(f, &mut data)
                });
                match read {
                    Some(Ok(_)) => {
                        // std validates the appended bytes as
                        // UTF-8 and leaves the string untouched
                        // when they are not (its guard truncates
                        // back), with a bare `InvalidData` error.
                        let text = match std::str::from_utf8(&data) {
                            Ok(text) => text.to_string(),
                            Err(_) => {
                                return Ok(io_result_err(
                                    io_simple_error_value(
                                        std::io::ErrorKind::InvalidData
                                            as u32,
                                    ),
                                ))
                            }
                        };
                        let target = self.deref_value(frames, &vals[1])?;
                        let current = self.read_resolved(frames, &target)?;
                        let Value::Str(mut existing) = current else {
                            return Err(InterpError::type_mismatch(
                                "Read::read_to_string destination",
                                "a String value",
                                current.kind_name(),
                            ));
                        };
                        let n = text.len();
                        existing.push_str(&text);
                        self.write_resolved(
                            frames,
                            &target,
                            Value::Str(existing),
                        )?;
                        Ok(io_result_ok(Value::Uint(n as u128)))
                    }
                    Some(Err(error)) => Ok(io_result_err(host_io_error(&error))),
                    None => Err(not_open("Read::read_to_string")),
                }
            }
            "std::io::BufRead::read_line"
            | "std::io::Stdin::read_line" => {
                arity(2)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let mut bytes: Vec<u8> = Vec::new();
                let read = self.with_file(id, |f| {
                    read_until_byte(f, &mut bytes, b'\n')
                });
                match read {
                    Some(Ok(())) => {
                        // As `read_to_string` does, the line is
                        // validated as UTF-8 before it joins the
                        // string; invalid bytes are a bare
                        // `InvalidData` and the string is left
                        // untouched, as std's guard leaves it.
                        let text = match std::str::from_utf8(&bytes) {
                            Ok(text) => text.to_string(),
                            Err(_) => {
                                return Ok(io_result_err(
                                    io_simple_error_value(
                                        std::io::ErrorKind::InvalidData
                                            as u32,
                                    ),
                                ))
                            }
                        };
                        let target = self.deref_value(frames, &vals[1])?;
                        let current =
                            self.read_resolved(frames, &target)?;
                        let Value::Str(mut existing) = current else {
                            return Err(InterpError::type_mismatch(
                                "BufRead::read_line destination",
                                "a String value",
                                current.kind_name(),
                            ));
                        };
                        let n = text.len();
                        existing.push_str(&text);
                        self.write_resolved(
                            frames,
                            &target,
                            Value::Str(existing),
                        )?;
                        Ok(io_result_ok(Value::Uint(n as u128)))
                    }
                    Some(Err(error)) => {
                        Ok(io_result_err(host_io_error(&error)))
                    }
                    None => Err(not_open("BufRead::read_line")),
                }
            }
            "std::io::BufRead::read_until" => {
                arity(3)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let delim = match self.resolve_value(frames, &vals[1])? {
                    Value::Uint(n) => n as u8,
                    Value::Int(n) => n as u8,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "BufRead::read_until delimiter",
                            "a byte",
                            other.kind_name(),
                        ))
                    }
                };
                let mut bytes: Vec<u8> = Vec::new();
                let read = self.with_file(id, |f| {
                    read_until_byte(f, &mut bytes, delim)
                });
                match read {
                    Some(Ok(())) => {
                        let n = bytes.len();
                        self.append_bytes_to_vec(
                            frames, &vals[2], &bytes,
                        )?;
                        Ok(io_result_ok(Value::Uint(n as u128)))
                    }
                    Some(Err(error)) => {
                        Ok(io_result_err(host_io_error(&error)))
                    }
                    None => Err(not_open("BufRead::read_until")),
                }
            }
            "std::io::Read::read_buf" | "std::io::Read::read_buf_exact" => {
                arity(2)?;
                let id = self.file_id_of(frames, &vals[0])?;
                // The cursor value is `BorrowedCursor { buf }`:
                // a one-field struct holding the `&mut
                // BorrowedBuf`; the buf value is `BorrowedBuf {
                // buf, filled, init }` — the byte slice, the
                // filled count and the initialised count.
                let cursor_fields =
                    aggregate_fields(&vals[1]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "Read::read_buf cursor",
                            "a BorrowedCursor value",
                            vals[1].kind_name(),
                        )
                    })?;
                let buf_ref = cursor_fields
                    .first()
                    .ok_or_else(|| {
                        unsupported(
                            "Read::read_buf cursor has no buffer field"
                                .to_string(),
                        )
                    })?
                    .clone();
                let buf_target = self.deref_value(frames, &buf_ref)?;
                let buf_value = self.read_resolved(frames, &buf_target)?;
                let buf_fields =
                    aggregate_fields(&buf_value).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "Read::read_buf buffer",
                            "a BorrowedBuf value",
                            buf_value.kind_name(),
                        )
                    })?;
                if buf_fields.len() != 3 {
                    return Err(unsupported(format!(
                        "Read::read_buf on a BorrowedBuf of {} fields is not modelled",
                        buf_fields.len()
                    )));
                }
                let slice = buf_fields[0].clone();
                let filled = as_u128(&buf_fields[1], name)? as usize;
                let init = as_u128(&buf_fields[2], name)? as usize;
                let Value::FatRef {
                    len: capacity, ..
                } = &slice
                else {
                    return Err(InterpError::type_mismatch(
                        "Read::read_buf buffer slice",
                        "a byte slice pointer",
                        slice.kind_name(),
                    ));
                };
                let want = (*capacity as usize).saturating_sub(filled);
                let exact = name.ends_with("read_buf_exact");
                let mut tmp = vec![0u8; want];
                let mut got = 0usize;
                let outcome = self.with_file(id, |f| {
                    loop {
                        if got >= want {
                            return None;
                        }
                        match <std::fs::File as std::io::Read>::read(
                            f,
                            &mut tmp[got..],
                        ) {
                            Ok(0) => return None,
                            Ok(n) => {
                                got += n;
                                if !exact {
                                    return None;
                                }
                            }
                            Err(e)
                                if e.kind()
                                    == std::io::ErrorKind::Interrupted =>
                            {
                                continue
                            }
                            Err(e) => return Some(e),
                        }
                    }
                });
                match outcome {
                    None => Err(not_open("Read::read_buf")),
                    Some(Some(error)) => {
                        Ok(io_result_err(host_io_error(&error)))
                    }
                    Some(None) => {
                        self.write_bytes_into_slice(
                            frames,
                            &slice,
                            filled,
                            &tmp[..got],
                        )?;
                        let mut new_fields = buf_fields.to_vec();
                        new_fields[1] = Value::Uint((filled + got) as u128);
                        if filled + got > init {
                            new_fields[2] =
                                Value::Uint((filled + got) as u128);
                        }
                        let new_buf = match &buf_value {
                            Value::Enum { variant, .. } => Value::Enum {
                                variant: *variant,
                                fields: new_fields,
                            },
                            _ => Value::Tuple(new_fields),
                        };
                        self.write_resolved(frames, &buf_target, new_buf)?;
                        if exact && got < want {
                            Ok(io_result_err(io_simple_error_value(
                                std::io::ErrorKind::UnexpectedEof as u32,
                            )))
                        } else {
                            Ok(io_result_ok(Value::Unit))
                        }
                    }
                }
            }
            "std::io::Write::write" => {
                arity(2)?;
                let bytes = self.slice_bytes(frames, &vals[1], name)?;
                if self_ty.starts_with("std::vec::Vec<u8") {
                    self.append_bytes_to_vec(frames, &vals[0], &bytes)?;
                    return Ok(io_result_ok(Value::Uint(bytes.len() as u128)));
                }
                if stream_write_receiver(self_ty) {
                    if stream_is_stderr(self_ty) {
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        self.push_error_output(&text);
                        return Ok(io_result_ok(Value::Uint(bytes.len() as u128)));
                    }
                    return match self.write_stdout_partial(&bytes) {
                        Ok(n) => Ok(io_result_ok(Value::Uint(n as u128))),
                        Err(error) => Ok(io_result_err(host_io_error(&error))),
                    };
                }
                let id = self.file_id_of(frames, &vals[0])?;
                let written = self.with_file(id, |f| {
                    <std::fs::File as std::io::Write>::write(f, &bytes)
                });
                self.file_result(written, |n| Value::Uint(n as u128))
            }
            "std::io::Write::write_all" => {
                arity(2)?;
                let bytes = self.slice_bytes(frames, &vals[1], name)?;
                if self_ty.starts_with("std::vec::Vec<u8") {
                    self.append_bytes_to_vec(frames, &vals[0], &bytes)?;
                    return Ok(io_result_ok(Value::Unit));
                }
                if stream_write_receiver(self_ty) {
                    if stream_is_stderr(self_ty) {
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        self.push_error_output(&text);
                        return Ok(io_result_ok(Value::Unit));
                    }
                    return match self.write_stdout(&bytes) {
                        Ok(()) => Ok(io_result_ok(Value::Unit)),
                        Err(error) => Ok(io_result_err(host_io_error(&error))),
                    };
                }
                let id = self.file_id_of(frames, &vals[0])?;
                let written = self.with_file(id, |f| {
                    <std::fs::File as std::io::Write>::write_all(f, &bytes)
                });
                self.file_result(written, |()| Value::Unit)
            }
            "std::io::Write::flush" => {
                arity(1)?;
                if stream_write_receiver(self_ty) {
                    if stream_is_stderr(self_ty) {
                        // The captured error stream is
                        // unbuffered in this model; there is
                        // nothing to flush.
                        return Ok(io_result_ok(Value::Unit));
                    }
                    // The captured output stream is likewise
                    // unbuffered; a write-through sink (a
                    // real stream behind the run) is flushed
                    // for real, and its error is the guest's.
                    return match self.flush_stdout() {
                        Ok(()) => Ok(io_result_ok(Value::Unit)),
                        Err(error) => Ok(io_result_err(host_io_error(&error))),
                    };
                }
                let id = self.file_id_of(frames, &vals[0])?;
                let flushed = self.with_file(id, |f| {
                    <std::fs::File as std::io::Write>::flush(f)
                });
                self.file_result(flushed, |()| Value::Unit)
            }
            "std::io::Seek::seek" => {
                arity(2)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let pos = match &vals[1] {
                    Value::Enum { variant: 0, fields } if fields.len() == 1 => {
                        std::io::SeekFrom::Start(as_u128(&fields[0], name)? as u64)
                    }
                    Value::Enum { variant: 1, fields } if fields.len() == 1 => {
                        std::io::SeekFrom::End(as_i128(&fields[0], name)? as i64)
                    }
                    Value::Enum { variant: 2, fields } if fields.len() == 1 => {
                        std::io::SeekFrom::Current(as_i128(&fields[0], name)? as i64)
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "Seek::seek position",
                            "a SeekFrom value",
                            other.kind_name(),
                        ))
                    }
                };
                let sought = self.with_file(id, |f| {
                    <std::fs::File as std::io::Seek>::seek(f, pos)
                });
                self.file_result(sought, |n| Value::Uint(n as u128))
            }
            "std::io::Seek::rewind" => {
                arity(1)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let sought = self.with_file(id, |f| {
                    <std::fs::File as std::io::Seek>::seek(
                        f,
                        std::io::SeekFrom::Start(0),
                    )
                });
                self.file_result(sought, |_| Value::Unit)
            }
            "std::io::Seek::stream_position" => {
                arity(1)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let sought = self.with_file(id, |f| {
                    <std::fs::File as std::io::Seek>::seek(
                        f,
                        std::io::SeekFrom::Current(0),
                    )
                });
                self.file_result(sought, |n| Value::Uint(n as u128))
            }
            "std::fs::File::metadata" => {
                arity(1)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let md = self.with_file(id, |f| f.metadata());
                self.file_result(md, |md| Value::Enum {
                    variant: 0,
                    fields: vec![metadata_value(&md)],
                })
            }
            "std::fs::File::try_clone" => {
                arity(1)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let cloned = self.with_file(id, |f| f.try_clone());
                match cloned {
                    Some(Ok(new_file)) => {
                        let new_id = self.register_file(new_file);
                        Ok(io_result_ok(file_value(new_id)))
                    }
                    Some(Err(error)) => Ok(io_result_err(host_io_error(&error))),
                    None => Err(not_open("File::try_clone")),
                }
            }
            "std::fs::File::sync_all" => {
                arity(1)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let synced = self.with_file(id, |f| f.sync_all());
                self.file_result(synced, |()| Value::Unit)
            }
            "std::fs::File::sync_data" => {
                arity(1)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let synced = self.with_file(id, |f| f.sync_data());
                self.file_result(synced, |()| Value::Unit)
            }
            "std::fs::File::set_len" => {
                arity(2)?;
                let id = self.file_id_of(frames, &vals[0])?;
                let len = as_u128(&vals[1], name)? as u64;
                let set = self.with_file(id, |f| f.set_len(len));
                self.file_result(set, |()| Value::Unit)
            }
            _ => Err(unsupported(format!(
                "call to {name} is not runtime-supported"
            ))),
        }
    }
}
