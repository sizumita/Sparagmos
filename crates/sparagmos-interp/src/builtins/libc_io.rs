//! Descriptor-level libc I/O: `libc::read` over the
//! interpreter's one descriptor space, and the errno cell
//! the location calls hand out.
//!
//! A guest descriptor number names an entry of the file
//! table or the socket table (the two share one counter,
//! so the space is one space), or descriptor 0, the file
//! model's stdin entry. `libc::read` runs the read on the
//! host object behind the entry and reports the outcome
//! the C call reports: the byte count on success, and -1
//! with the host errno recorded in the errno cell on
//! failure. A descriptor naming no entry fails with
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

use std::io::Read as _;

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
