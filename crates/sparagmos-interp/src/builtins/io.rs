//! File I/O execution for the io builtins.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter,
};
use crate::value::{RefProj, RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// The file-table id of the `std::fs::File` handle a
    /// receiver value denotes (a reference to the handle, or the
    /// handle value itself).
    pub(super) fn file_id_of(&self, frames: &[Frame], receiver: &Value) -> InterpResult<u64> {
        let mut value = self.resolve_value(frames, receiver)?;
        // The handle is an integer under a chain of one-field
        // ADT wrappers (see `file_value`); a receiver recovered
        // from a vtable dispatch hands over the outermost
        // wrapper whose field is a *reference* to the stored
        // handle value instead, so follow reference layers
        // while unwrapping. (A `Box<dyn Read>` over a `File`
        // arrived in exactly that shape and was refused as not
        // a file handle.)
        for _ in 0..8 {
            match &value {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    match self.resolve_value(frames, &fields[0]) {
                        Ok(resolved) => value = resolved,
                        Err(_) => break,
                    }
                }
                Value::Int(id) => return Ok(*id as u64),
                Value::Uint(id) => return Ok(*id as u64),
                _ => break,
            }
        }
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

    /// Write `bytes` through a `std::fs::File` handle, in
    /// the descriptor model's terms (see `desc_route` in
    /// `interp::state`): a handle over a file-table entry
    /// writes to that file, and so does a handle built by
    /// `File::from_raw_fd` over a descriptor the routing
    /// assigns to an entry (descriptor 0's stdin entry above
    /// all). A handle over descriptor 1 or 2 -- numbers the
    /// file counter never issues, so no table entry ever
    /// carries them -- writes to the guest's standard stream
    /// through the captured-stream model instead, so the
    /// bytes land where the guest's own stream writes land,
    /// as a native descriptor write lands on the process's
    /// stream. `None` when the handle names nothing the
    /// model holds (the caller's not-open error).
    fn file_write(
        &mut self,
        id: u64,
        bytes: &[u8],
    ) -> Option<std::io::Result<usize>> {
        if self.has_file(id) {
            return self.with_file(id, |f| {
                <std::fs::File as std::io::Write>::write(f, bytes)
            });
        }
        match self.desc_route(id) {
            crate::interp::DescRoute::Stdout => {
                Some(self.write_stdout_partial(bytes))
            }
            crate::interp::DescRoute::Stderr => {
                Some(self.write_stderr_bytes(bytes))
            }
            _ => self.desc_entry_id(id).and_then(|entry| {
                self.with_file(entry, |f| {
                    <std::fs::File as std::io::Write>::write(f, bytes)
                })
            }),
        }
    }

    /// The `write_all` form of [`Self::file_write`].
    fn file_write_all(
        &mut self,
        id: u64,
        bytes: &[u8],
    ) -> Option<std::io::Result<()>> {
        if self.has_file(id) {
            return self.with_file(id, |f| {
                <std::fs::File as std::io::Write>::write_all(f, bytes)
            });
        }
        match self.desc_route(id) {
            crate::interp::DescRoute::Stdout => {
                Some(self.write_stdout(bytes))
            }
            crate::interp::DescRoute::Stderr => Some(
                self.write_stderr_bytes(bytes).map(|_| ()),
            ),
            _ => self.desc_entry_id(id).and_then(|entry| {
                self.with_file(entry, |f| {
                    <std::fs::File as std::io::Write>::write_all(
                        f, bytes,
                    )
                })
            }),
        }
    }

    /// The `flush` form of [`Self::file_write`]: the captured
    /// streams are unbuffered, so flushing a standard-stream
    /// handle settles the stdout line buffer, as flushing
    /// the guest's own `Stdout` handle does.
    fn file_flush(
        &mut self,
        id: u64,
    ) -> Option<std::io::Result<()>> {
        if self.has_file(id) {
            return self.with_file(id, |f| {
                <std::fs::File as std::io::Write>::flush(f)
            });
        }
        match self.desc_route(id) {
            crate::interp::DescRoute::Stdout => {
                Some(self.flush_stdout())
            }
            crate::interp::DescRoute::Stderr => Some(Ok(())),
            _ => self.desc_entry_id(id).and_then(|entry| {
                self.with_file(entry, |f| {
                    <std::fs::File as std::io::Write>::flush(f)
                })
            }),
        }
    }

    /// The bytes of a byte-slice value (the buffer a `Write`
    /// call hands over).
    pub(super) fn slice_bytes(
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
    pub(super) fn write_bytes_into_slice(
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
    pub(crate) fn exec_file_io(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        generic_args: &[String],
        vals: Vec<Value>,
    ) -> InterpResult<Value> {
        let name_cow = crate::builtins::io_spelling(name);
        let name = &*name_cow;
        let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
        let self_ty = self_ty
            .strip_prefix("&mut ")
            .or_else(|| self_ty.strip_prefix("&"))
            .unwrap_or(self_ty);
        let self_ty = crate::builtins::self_ty_key(self_ty);
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
                let read = if self.has_file(id) {
                    self.with_file(id, |f| {
                        <std::fs::File as std::io::Read>::read(
                            f, &mut buf,
                        )
                    })
                } else if matches!(
                    self.desc_route(id),
                    crate::interp::DescRoute::Stdout
                        | crate::interp::DescRoute::Stderr
                ) {
                    Some(Err(std::io::Error::from_raw_os_error(9)))
                } else {
                    self.desc_entry_id(id).and_then(|entry| {
                        self.with_file(entry, |f| {
                            <std::fs::File as std::io::Read>::read(
                                f, &mut buf,
                            )
                        })
                    })
                };
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
                        return match self.write_stderr_bytes(&bytes) {
                            Ok(n) => Ok(io_result_ok(Value::Uint(n as u128))),
                            Err(error) => Ok(io_result_err(host_io_error(&error))),
                        };
                    }
                    return match self.write_stdout_partial(&bytes) {
                        Ok(n) => Ok(io_result_ok(Value::Uint(n as u128))),
                        Err(error) => Ok(io_result_err(host_io_error(&error))),
                    };
                }
                let id = self.file_id_of(frames, &vals[0])?;
                let written = self.file_write(id, &bytes);
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
                        return match self.write_stderr_bytes(&bytes) {
                            Ok(_) => Ok(io_result_ok(Value::Unit)),
                            Err(error) => Ok(io_result_err(host_io_error(&error))),
                        };
                    }
                    return match self.write_stdout(&bytes) {
                        Ok(()) => Ok(io_result_ok(Value::Unit)),
                        Err(error) => Ok(io_result_err(host_io_error(&error))),
                    };
                }
                let id = self.file_id_of(frames, &vals[0])?;
                let written = self.file_write_all(id, &bytes);
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
                let flushed = self.file_flush(id);
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
                match md {
                    Some(Ok(md)) => Ok(io_result_ok(Value::Enum {
                        variant: 0,
                        fields: vec![metadata_value(&md)?],
                    })),
                    Some(Err(error)) => {
                        Ok(io_result_err(host_io_error(&error)))
                    }
                    None => Err(unsupported(
                        "file operation on a handle that is not open"
                            .to_string(),
                    )),
                }
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

/// Registration fragment of the builtin name registry: the `std::io` print entry points.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn io_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::io::_print" => Builtin::Print,
        "std::io::BufWriter::<W>::flush_buf::BufGuard::remaining" => {
            Builtin::BufGuardRemaining
        }
        "std::io::_eprint" => Builtin::Eprint,
        "std::io::stdio::attempt_print_to_stdout" => Builtin::AttemptPrint,
        "std::io::stdio::attempt_print_to_stderr" => Builtin::AttemptEprint,
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the standard stream constructors.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn io_stream_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::io::stdout" => Builtin::Stdout,
        "std::io::stdin" => Builtin::Stdin,
        "std::io::Stdout::lock" | "std::io::Stderr::lock"
        | "std::io::Stdin::lock" => Builtin::StdStreamLock,
        "std::io::Write::write_fmt" => Builtin::WriteFmt,
        _ => return None,
    })
}

impl<'a> Interpreter<'a> {
    /// `BufGuard::remaining` (the guard `BufWriter::flush_buf`
    /// lends its buffer to): the unwritten part of the buffer,
    /// `&buffer[written..]`. The returned fat reference
    /// denotes the buffer allocation's element window from
    /// `written` on — the sub-slice projection over the
    /// allocation the vector's buffer pointer names, with
    /// the remaining length.
    pub(super) fn buf_guard_remaining(
        &self,
        frames: &mut Vec<Frame>,
        guard: &Value,
        name: &str,
    ) -> InterpResult<Value> {
        let value = self.resolve_value(frames, guard)?;
        let fields = match &value {
            Value::Enum { fields, .. } | Value::Tuple(fields) => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a BufGuard value",
                    other.kind_name(),
                ))
            }
        };
        if fields.len() != 2 {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a BufGuard value (buffer, written)",
                value.kind_name(),
            ));
        }
        let written = match &fields[1] {
            Value::Uint(n) => *n,
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a BufGuard written count",
                    other.kind_name(),
                ))
            }
        };
        let vec_value = self.resolve_value(frames, &fields[0])?;
        let total = crate::interp::vec_value_len(&vec_value).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "a BufGuard buffer vector",
                vec_value.kind_name(),
            )
        })?;
        let remaining = total.checked_sub(written).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "a BufGuard whose written count fits its buffer",
                value.kind_name(),
            )
        })?;
        // The buffer allocation: the innermost field-0 chain
        // of the vector aggregate ends at the buffer pointer,
        // a reference to the allocation holding the elements.
        let mut cur = &vec_value;
        let alloc_id = loop {
            match cur {
                Value::Ref(RefTarget::Alloc { id, .. }) => break *id,
                Value::Enum { fields, .. } | Value::Tuple(fields)
                    if !fields.is_empty() =>
                {
                    cur = &fields[0];
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a BufGuard buffer allocation pointer",
                        other.kind_name(),
                    ))
                }
            }
        };
        Ok(Value::FatRef {
            base: Box::new(Value::Ref(RefTarget::Alloc {
                id: alloc_id,
                projection: vec![RefProj::Subslice {
                    offset: written as usize,
                    len: remaining as usize,
                }],
            })),
            len: remaining,
        })
    }

}
