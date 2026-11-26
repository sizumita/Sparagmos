//! Descriptor-level libc I/O: `libc::read` and
//! `libc::write` over the interpreter's one descriptor
//! space, and the errno cell the location calls hand out.
//!
//! A guest descriptor number names an entry of the file
//! table or the socket table (the two share one counter,
//! so the space is one space), or descriptor 0, the file
//! model's stdin entry. The call runs on the host object
//! behind the entry and reports the outcome the C call
//! reports: the byte count on success, and -1 with the
//! host errno recorded in the errno cell on failure.
//! Descriptors 1 and 2 are the guest's own standard
//! streams: a `libc::write` there goes to the captured
//! stdout / stderr the stdio model writes, not to a host
//! descriptor of this process. A descriptor naming no entry fails with
//! `EBADF`: that is the shape of a number never issued, of
//! a descriptor already closed, and of a descriptor a
//! parent hands a re-executed child it was cloexec in --
//! the child's tables start empty, so every number the
//! parent passes names nothing, exactly as the host
//! child's empty descriptor table answers.
//!
//! The errno cell is the four-byte allocation
//! `errno_cell_ref` (in `interp::state`) creates on first
//! use: the location calls
//! (`std::sys::io::error::unix::errno_location`, the libc
//! crate's `__errno_location`) return its address, the
//! failing operations here record their errno in it, and
//! the guest reads it back through the pointer -- the
//! path `std::io::Error::last_os_error` takes.

use std::io::{Read as _, Write as _};

use crate::error::{InterpError, InterpResult};
use crate::interp::{Frame, Interpreter, SocketEntry};
use crate::value::Value;

use super::*;

/// The host errno for a descriptor that names no open
/// entry, and for one open in the wrong direction
/// (`EBADF`).
const EBADF: i32 = 9;
/// The host errno reported when a failing operation's
/// host error carries no OS code (`EIO`).
const EIO: i32 = 5;
/// The cap on one read's host buffer, the same cap the
/// `Read::read` model applies: a single `read` may legally
/// return short, so a buffer value claiming an
/// astronomical length cannot exhaust the host through
/// one call.
const READ_CAP: usize = 1 << 23;

impl<'a> Interpreter<'a> {
    /// `errno_location() -> *mut c_int`: the errno cell's
    /// address.
    pub(super) fn exec_errno_location(&mut self) -> InterpResult<Value> {
        Ok(self.errno_cell_ref())
    }

    /// Record `code` in the guest's errno cell: the write
    /// a failing descriptor operation leaves for
    /// `std::io::Error::last_os_error` to read back.
    fn set_errno(
        &mut self,
        frames: &mut Vec<Frame>,
        code: i32,
    ) -> InterpResult<()> {
        let cell = self.errno_cell_ref();
        let target = self.deref_value(frames, &cell)?;
        self.write_resolved(frames, &target, Value::Int(i128::from(code)))
    }

    /// `libc::read(fd, buf, count) -> ssize_t`: read up to
    /// `count` bytes from the guest descriptor `fd` into
    /// the buffer `buf` denotes, on the host object behind
    /// the descriptor's table entry (see the module
    /// comment). A would-block outcome on a socket waits
    /// the way the socket model waits (see
    /// `libc_socket_read`).
    pub(super) fn exec_libc_read(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let fd = match &vals[0] {
            Value::Int(v) => *v as i64,
            Value::Uint(v) => *v as i64,
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a descriptor integer",
                    other.kind_name(),
                ))
            }
        };
        let count = match &vals[2] {
            Value::Uint(v) => *v as usize,
            Value::Int(v) if *v >= 0 => *v as usize,
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a byte count",
                    other.kind_name(),
                ))
            }
        };
        if fd < 0 {
            self.set_errno(frames, EBADF)?;
            return Ok(Value::Int(-1));
        }
        let id = fd as u64;
        let mut buf = vec![0u8; count.min(READ_CAP)];
        // Descriptor 0 is the file model's stdin entry,
        // registered on first use; any other descriptor is
        // looked up in the file table first, then the
        // socket table (one descriptor space, two tables).
        let file_read = if id == 0 {
            let stdin_id = self.stdin_file_id();
            self.with_file(stdin_id, |f| f.read(&mut buf))
        } else {
            self.with_file(id, |f| f.read(&mut buf))
        };
        let result = match file_read {
            Some(result) => Some(result),
            None => self.libc_socket_read(frames, name, id, &mut buf)?,
        };
        match result {
            Some(Ok(n)) => {
                if n > 0 {
                    self.write_read_bytes(frames, &vals[1], &buf[..n])?;
                }
                Ok(Value::Int(n as i128))
            }
            Some(Err(error)) => {
                let code = error.raw_os_error().unwrap_or(EIO);
                self.set_errno(frames, code)?;
                Ok(Value::Int(-1))
            }
            None => {
                self.set_errno(frames, EBADF)?;
                Ok(Value::Int(-1))
            }
        }
    }

    /// `libc::write(fd, buf, count) -> ssize_t`: write
    /// the `count` bytes the buffer `buf` denotes to the
    /// guest descriptor `fd`, on the host object behind
    /// the descriptor's table entry (see the module
    /// comment), the twin of `exec_libc_read`. Descriptors
    /// 1 and 2 write the guest's captured stdout and
    /// stderr. A would-block outcome on a socket waits
    /// the way the socket model waits (see
    /// `libc_socket_write`).
    pub(super) fn exec_libc_write(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let fd = match &vals[0] {
            Value::Int(v) => *v as i64,
            Value::Uint(v) => *v as i64,
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a descriptor integer",
                    other.kind_name(),
                ))
            }
        };
        let count = match &vals[2] {
            Value::Uint(v) => *v as usize,
            Value::Int(v) if *v >= 0 => *v as usize,
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a byte count",
                    other.kind_name(),
                ))
            }
        };
        if fd < 0 {
            self.set_errno(frames, EBADF)?;
            return Ok(Value::Int(-1));
        }
        let bytes = self.read_write_bytes(frames, &vals[1], count)?;
        if fd == 1 {
            return match self.write_stdout_partial(&bytes) {
                Ok(n) => Ok(Value::Int(n as i128)),
                Err(error) => {
                    let code = error.raw_os_error().unwrap_or(EIO);
                    self.set_errno(frames, code)?;
                    Ok(Value::Int(-1))
                }
            };
        }
        if fd == 2 {
            self.push_error_output(&String::from_utf8_lossy(&bytes));
            return Ok(Value::Int(bytes.len() as i128));
        }
        let id = fd as u64;
        // Descriptor 0 is the file model's stdin entry,
        // registered on first use; any other descriptor is
        // looked up in the file table first, then the
        // socket table (one descriptor space, two tables).
        // The host object answers for the entry's own
        // direction: a write against a read-only file or
        // the stdin duplicate fails with the host's errno.
        let file_write = if id == 0 {
            let stdin_id = self.stdin_file_id();
            self.with_file(stdin_id, |f| f.write(&bytes))
        } else {
            self.with_file(id, |f| f.write(&bytes))
        };
        let result = match file_write {
            Some(result) => Some(result),
            None => self.libc_socket_write(frames, name, id, &bytes)?,
        };
        match result {
            Some(Ok(n)) => Ok(Value::Int(n as i128)),
            Some(Err(error)) => {
                let code = error.raw_os_error().unwrap_or(EIO);
                self.set_errno(frames, code)?;
                Ok(Value::Int(-1))
            }
            None => {
                self.set_errno(frames, EBADF)?;
                Ok(Value::Int(-1))
            }
        }
    }

    /// One `write(2)` against the socket behind
    /// descriptor `id`; `None` when the descriptor names
    /// no socket. The host socket is nonblocking: while
    /// an attempt cannot proceed and the guest descriptor
    /// is a blocking one, the other guest threads run in
    /// scheduling rounds, and where no round makes
    /// progress the call suspends (or reports the honest
    /// would-block-forever error), the socket model's own
    /// pattern (see `libc_socket_read`). A
    /// guest-nonblocking descriptor, and a stream with a
    /// write timeout, surface the host's `WouldBlock`
    /// error instead, as `write(2)` under `O_NONBLOCK` /
    /// `SO_SNDTIMEO` does.
    fn libc_socket_write(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        id: u64,
        bytes: &[u8],
    ) -> InterpResult<Option<std::io::Result<usize>>> {
        enum Step {
            Done(std::io::Result<usize>),
            WouldBlock,
        }
        loop {
            let step = self.with_socket(id, |entry| match entry {
                SocketEntry::Stream {
                    socket,
                    nonblocking,
                    write_timeout,
                    ..
                } => match socket.write(bytes) {
                    Ok(n) => Step::Done(Ok(n)),
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && !*nonblocking
                            && write_timeout.is_none() =>
                    {
                        Step::WouldBlock
                    }
                    Err(error) => Step::Done(Err(error)),
                },
                SocketEntry::Udp {
                    socket,
                    nonblocking,
                } => match socket.send(bytes) {
                    Ok(n) => Step::Done(Ok(n)),
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && !*nonblocking =>
                    {
                        Step::WouldBlock
                    }
                    Err(error) => Step::Done(Err(error)),
                },
                SocketEntry::Listener { socket, .. } => {
                    // A listening socket has no write of
                    // its own in std's model types; the
                    // kernel's own answer (`EPIPE` on
                    // Linux) comes from a duplicate of the
                    // host descriptor.
                    use std::os::fd::AsFd as _;
                    let result = socket
                        .as_fd()
                        .try_clone_to_owned()
                        .and_then(|owned| {
                            let mut file = std::fs::File::from(owned);
                            file.write(bytes)
                        });
                    Step::Done(result)
                }
            });
            match step {
                Some(Step::Done(result)) => return Ok(Some(result)),
                Some(Step::WouldBlock) => {
                    self.tick_step()?;
                    let (_, progressed) = self.schedule_round(frames)?;
                    if !progressed {
                        return Err(self.block_or_suspend(
                            name,
                            "a blocking libc::write on a socket",
                        ));
                    }
                }
                None => return Ok(None),
            }
        }
    }

    /// Gather the bytes a descriptor write sends, from
    /// the guest buffer `ptr` denotes (the read
    /// `Write::write` performs out of its caller's buffer,
    /// for a raw pointer): the mirror of
    /// `write_read_bytes`. A fat pointer goes through the
    /// slice elements; a thin pointer to a sequence value
    /// takes the value's own bytes; a thin pointer to an
    /// element position inside a sequence goes through
    /// the sequence access, byte by byte. Only the bytes
    /// the pointer denotes are ever gathered: a count
    /// running past a fat slice's length gathers the
    /// slice's bytes and the write comes back short, as
    /// a `write` legally may, rather than inventing
    /// bytes past the buffer.
    fn read_write_bytes(
        &mut self,
        frames: &mut Vec<Frame>,
        ptr: &Value,
        count: usize,
    ) -> InterpResult<Vec<u8>> {
        if matches!(ptr, Value::FatRef { .. }) {
            let (elems, _) = self.slice_elements(frames, ptr)?;
            let mut out = Vec::with_capacity(count.min(elems.len()));
            for item in elems.iter().take(count) {
                out.push(as_u128(item, "libc::write")? as u8);
            }
            return Ok(out);
        }
        let target = self.deref_value(frames, ptr)?;
        let current = self.read_resolved(frames, &target)?;
        if let Some(bytes) = sequence_bytes(&current) {
            if bytes.len() >= count {
                return Ok(bytes[..count].to_vec());
            }
        }
        // The pointer denotes an element position inside
        // a sequence rather than the sequence itself (an
        // array's first-element pointer, the shape
        // `as_ptr` on an array value produces): the
        // sequence access reads each byte at its own
        // position, continuing at the element's siblings.
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let byte = self.read_seq_at(frames, &target, index)?;
            out.push(as_u128(&byte, "libc::write")? as u8);
        }
        Ok(out)
    }

    /// One `read(2)` against the socket behind descriptor
    /// `id`; `None` when the descriptor names no socket.
    /// The host socket is nonblocking: while an attempt
    /// cannot proceed and the guest descriptor is a
    /// blocking one, the other guest threads run in
    /// scheduling rounds, and where no round makes
    /// progress the call suspends (or reports the honest
    /// would-block-forever error), the socket model's own
    /// pattern (see `net_block` in `builtins::net`). A
    /// guest-nonblocking descriptor, and a stream with a
    /// read timeout, surface the host's `WouldBlock`
    /// error instead, as `read(2)` under `O_NONBLOCK` /
    /// `SO_RCVTIMEO` does.
    fn libc_socket_read(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        id: u64,
        buf: &mut [u8],
    ) -> InterpResult<Option<std::io::Result<usize>>> {
        enum Step {
            Done(std::io::Result<usize>),
            WouldBlock,
        }
        loop {
            let step = self.with_socket(id, |entry| match entry {
                SocketEntry::Stream {
                    socket,
                    nonblocking,
                    read_timeout,
                    ..
                } => match socket.read(buf) {
                    Ok(n) => Step::Done(Ok(n)),
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && !*nonblocking
                            && read_timeout.is_none() =>
                    {
                        Step::WouldBlock
                    }
                    Err(error) => Step::Done(Err(error)),
                },
                SocketEntry::Udp {
                    socket,
                    nonblocking,
                } => match socket.recv(buf) {
                    Ok(n) => Step::Done(Ok(n)),
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && !*nonblocking =>
                    {
                        Step::WouldBlock
                    }
                    Err(error) => Step::Done(Err(error)),
                },
                SocketEntry::Listener { socket, .. } => {
                    // A listening socket has no read of its
                    // own in std's model types; the kernel's
                    // own answer (`ENOTCONN` on Linux) comes
                    // from a duplicate of the host
                    // descriptor.
                    use std::os::fd::AsFd as _;
                    let result = socket
                        .as_fd()
                        .try_clone_to_owned()
                        .and_then(|owned| {
                            let mut file = std::fs::File::from(owned);
                            file.read(buf)
                        });
                    Step::Done(result)
                }
            });
            match step {
                Some(Step::Done(result)) => return Ok(Some(result)),
                Some(Step::WouldBlock) => {
                    self.tick_step()?;
                    let (_, progressed) = self.schedule_round(frames)?;
                    if !progressed {
                        return Err(self.block_or_suspend(
                            name,
                            "a blocking libc::read on a socket",
                        ));
                    }
                }
                None => return Ok(None),
            }
        }
    }

    /// Store the bytes a descriptor read produced into
    /// the guest buffer `ptr` denotes (the write
    /// `Read::read` performs into its caller's buffer,
    /// for a raw pointer): a fat pointer goes through the
    /// slice write; a thin pointer to a sequence value
    /// rebuilds the value with the new bytes in place,
    /// preserving its shape; a thin pointer to an element
    /// position inside a sequence goes through the
    /// sequence access, byte by byte.
    fn write_read_bytes(
        &mut self,
        frames: &mut Vec<Frame>,
        ptr: &Value,
        bytes: &[u8],
    ) -> InterpResult<()> {
        if matches!(ptr, Value::FatRef { .. }) {
            return self.write_bytes_into_slice(frames, ptr, 0, bytes);
        }
        let target = self.deref_value(frames, ptr)?;
        let current = self.read_resolved(frames, &target)?;
        if let Some(new) = replace_slice_bytes(&current, 0, bytes) {
            return self.write_resolved(frames, &target, new);
        }
        // The pointer denotes an element position inside a
        // sequence rather than the sequence itself (an
        // array's first-element pointer, the shape
        // `as_mut_ptr` on an array value produces): the
        // sequence access writes each byte at its own
        // position, continuing at the element's siblings.
        for (index, byte) in bytes.iter().enumerate() {
            self.write_seq_at(
                frames,
                &target,
                index,
                Value::Uint(u128::from(*byte)),
            )?;
        }
        Ok(())
    }
}

/// The bytes of a byte-sequence value (a byte block, a
/// string's data, an element sequence of byte scalars),
/// the shapes `replace_slice_bytes` writes into read
/// back out; `None` for any other value, and for an
/// element sequence holding a non-byte element.
fn sequence_bytes(value: &Value) -> Option<Vec<u8>> {
    fn collect(elems: &[Value]) -> Option<Vec<u8>> {
        elems
            .iter()
            .map(|item| match item {
                Value::Uint(v) => Some(*v as u8),
                Value::Int(v) => Some(*v as u8),
                _ => None,
            })
            .collect()
    }
    match value {
        Value::Bytes(data) => Some(data.clone()),
        Value::Str(text) => Some(text.as_bytes().to_vec()),
        Value::Tuple(items) => collect(items),
        Value::Enum { fields, .. } => collect(fields),
        Value::SparseVec { elems, .. } => collect(elems),
        _ => None,
    }
}

/// Registration fragment of the builtin name registry: the libc I/O surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn libc_io_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        // `exit` ends the process with the given status,
        // running the registered `atexit` handlers: the
        // same observable behaviour as `std::process::exit`,
        // whose builtin serves both.
        "libc::exit" | "libc::unix::exit" => Builtin::ProcessExit,
        "libc::read" | "libc::unix::read" => Builtin::LibcRead,
        "libc::write" | "libc::unix::write" => Builtin::LibcWrite,
        "std::sys::io::error::unix::errno_location"
        | "libc::__errno_location"
        | "libc::unix::__errno_location" => Builtin::ErrnoLocation,
        _ => return None,
    })
}
