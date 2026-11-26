//! Socket syscall execution: the `std::sys::net` leaf
//! operations, performed on the interpreter's host socket
//! table (see `SocketEntry` in `interp.rs`).

use crate::error::{InterpError, InterpResult};
use crate::interp::{Frame, Interpreter, SocketEntry};
use crate::value::Value;

use super::*;

/// The guest socket handle value for table id `id`: the
/// descriptor integer under the wrapper chain the socket
/// model builds (public `TcpListener` / `TcpStream` /
/// `UdpSocket` -> sys socket type -> `Socket` -> `FileDesc`
/// -> `OwnedFd` -> descriptor), each layer a one-field
/// aggregate. The sys operations are inlined into the public
/// bodies inside std itself, so the model's entry points are
/// the public functions and the values they exchange are the
/// public shapes.
pub(super) fn socket_value(id: u64) -> Value {
    let mut value = Value::Int(id as i128);
    for _ in 0..5 {
        value = Value::Enum {
            variant: 0,
            fields: vec![value],
        };
    }
    value
}

/// One blocking socket operation's single attempt: the
/// operation completed (with the host's own result), it
/// cannot proceed yet, or the handle names no socket of the
/// operation's kind.
enum BlockStep<T> {
    Done(std::io::Result<T>),
    WouldBlock,
    Missing,
}

impl<'a> Interpreter<'a> {
    /// The socket-table id of the `std::net` handle a receiver
    /// value denotes (a reference to the handle, or the handle
    /// value itself).
    fn socket_id_of(
        &self,
        frames: &[Frame],
        receiver: &Value,
    ) -> InterpResult<u64> {
        let value = self.resolve_value(frames, receiver)?;
        crate::interp::file_handle_id(&value).ok_or_else(|| {
            InterpError::type_mismatch(
                "socket operation",
                "a std::net handle",
                value.kind_name(),
            )
        })
    }

    /// Map the outcome of a host socket operation to the guest
    /// `io::Result` value. A missing table entry (a handle the
    /// model never issued, or one already closed) is an
    /// explicit unsupported error, never a silent success.
    fn net_result<T>(
        &self,
        result: Option<std::io::Result<T>>,
        name: &str,
        ok: impl FnOnce(T) -> Value,
    ) -> InterpResult<Value> {
        match result {
            Some(Ok(value)) => Ok(io_result_ok(ok(value))),
            Some(Err(error)) => Ok(io_result_err(host_io_error(&error))),
            None => Err(unsupported(format!(
                "{name} on a handle that is not open"
            ))),
        }
    }

    /// Run a blocking socket operation the way the mpsc waits
    /// run (see `mpsc_recv` in `builtins::sync`): attempt it
    /// on the nonblocking host socket; while it cannot
    /// proceed, run the other guest threads in scheduling
    /// rounds; when no round makes progress, suspend where
    /// the scheduler permits it (a later activation
    /// re-executes the operation) and report the honest
    /// would-block-forever error where it does not. The
    /// attempt itself decides the guest-visible would-block
    /// outcome (a guest-nonblocking handle, or a stream with
    /// a read/write timeout, surfaces the host's `WouldBlock`
    /// error instead of waiting), exactly as std reports it.
    fn net_block<T>(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        situation: &str,
        mut attempt: impl FnMut(&Self) -> BlockStep<T>,
    ) -> InterpResult<Option<std::io::Result<T>>> {
        loop {
            match attempt(self) {
                BlockStep::Done(result) => return Ok(Some(result)),
                BlockStep::Missing => return Ok(None),
                BlockStep::WouldBlock => {
                    self.tick_step()?;
                    let (_, progressed) = self.schedule_round(frames)?;
                    if !progressed {
                        return Err(self.block_or_suspend(name, situation));
                    }
                }
            }
        }
    }

    /// Resolve a `ToSocketAddrs` argument into host addresses:
    /// `Ok(addrs)` on success, `Err(())` for the outcomes std
    /// reports as an `io::Error`. The address forms are the
    /// ones the `ToSocketAddrs` model itself resolves (see
    /// `resolve_socket_addrs`).
    fn resolve_host_addrs(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        value: &Value,
    ) -> InterpResult<Result<Vec<std::net::SocketAddr>, ()>> {
        let value = self.resolve_value(frames, value)?;
        let addrs = match self.resolve_socket_addrs(frames, self_ty, &value)? {
            Ok(addrs) => addrs,
            Err(()) => return Ok(Err(())),
        };
        let mut out = Vec::with_capacity(addrs.len());
        for addr in &addrs {
            match host_socket_addr(addr) {
                Some(host) => out.push(host),
                None => {
                    return Err(InterpError::type_mismatch(
                        "socket address resolution",
                        "a socket address value",
                        addr.kind_name(),
                    ))
                }
            }
        }
        Ok(Ok(out))
    }

    /// The host address a single-address argument denotes (a
    /// `&SocketAddr` parameter of a sys operation).
    fn host_addr_arg(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<std::net::SocketAddr> {
        let value = self.resolve_value(frames, value)?;
        host_socket_addr(&value).ok_or_else(|| {
            InterpError::type_mismatch(
                "socket operation",
                "a socket address argument",
                value.kind_name(),
            )
        })
    }

    /// An `Option<Duration>` argument as a host duration.
    fn option_duration_arg(
        &self,
        value: &Value,
        name: &str,
    ) -> InterpResult<Option<std::time::Duration>> {
        match value {
            Value::Enum { variant: 0, .. } => Ok(None),
            Value::Enum {
                variant: 1, fields,
            } if fields.len() == 1 => {
                let (secs, nanos) =
                    duration_pair(&fields[0]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            name,
                            "a Duration value",
                            fields[0].kind_name(),
                        )
                    })?;
                Ok(Some(std::time::Duration::new(
                    secs as u64,
                    nanos as u32,
                )))
            }
            other => Err(InterpError::type_mismatch(
                name,
                "an Option<Duration> value",
                other.kind_name(),
            )),
        }
    }

    /// A guest `Option<Duration>` value for a host duration.
    fn option_duration_value(
        &self,
        value: Option<std::time::Duration>,
    ) -> InterpResult<Value> {
        match value {
            None => Ok(none_value()),
            Some(d) => Ok(some_value(self.duration_value(
                d.as_secs(),
                u64::from(d.subsec_nanos()),
            )?)),
        }
    }

    /// A guest `Option<io::Error>` value for a host
    /// `take_error` outcome.
    fn option_error_value(
        &self,
        value: Option<std::io::Error>,
    ) -> Value {
        match value {
            None => none_value(),
            Some(e) => some_value(host_io_error(&e)),
        }
    }

    /// Execute one socket builtin node (see the `NetSys`
    /// registry entries): a `std::sys::net::connection::socket`
    /// leaf operation, performed on the interpreter's host
    /// socket table. The host sockets are real sockets, so
    /// addressing, errors and connection behaviour are the
    /// host's own; the guest-visible results are std's own
    /// shapes (the host errno in the `Os` error form).
    pub(super) fn exec_net_sys(
        &mut self,
        frames: &mut Vec<Frame>,
        op: NetSysOp,
        generic_args: &[String],
        vals: Vec<Value>,
    ) -> InterpResult<Value> {
        use NetSysOp as O;
        let arity = |n: usize| -> InterpResult<()> {
            if vals.len() == n {
                Ok(())
            } else {
                Err(InterpError::type_mismatch(
                    "socket operation",
                    format!("{n} argument(s)"),
                    format!("{} argument(s)", vals.len()),
                ))
            }
        };
        let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
        match op {
            O::ListenerBind => {
                arity(1)?;
                let addrs =
                    match self.resolve_host_addrs(frames, self_ty, &vals[0])? {
                        Ok(addrs) => addrs,
                        Err(()) => {
                            return Ok(io_result_err(io_simple_error_value(
                                ERROR_KIND_INVALID_INPUT,
                            )))
                        }
                    };
                // std's `each_addr`: try each address in
                // order, keep the first success, report the
                // last error.
                let mut last_err = None;
                for addr in addrs {
                    match std::net::TcpListener::bind(addr) {
                        Ok(listener) => {
                            if let Err(e) = listener.set_nonblocking(true) {
                                return Ok(io_result_err(host_io_error(&e)));
                            }
                            let id = self.register_socket(
                                SocketEntry::Listener {
                                    socket: listener,
                                    nonblocking: false,
                                },
                            );
                            return Ok(io_result_ok(socket_value(id)));
                        }
                        Err(e) => last_err = Some(e),
                    }
                }
                match last_err {
                    Some(e) => Ok(io_result_err(host_io_error(&e))),
                    None => Ok(io_result_err(io_simple_error_value(
                        ERROR_KIND_INVALID_INPUT,
                    ))),
                }
            }
            O::ListenerAccept => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let outcome = self.net_block(
                    frames,
                    "TcpListener::accept",
                    "no connection is pending and no pending thread can connect",
                    |this| {
                        this.with_socket(id, |entry| {
                            let SocketEntry::Listener {
                                socket,
                                nonblocking,
                            } = entry
                            else {
                                return BlockStep::Missing;
                            };
                            match socket.accept() {
                                Ok(pair) => BlockStep::Done(Ok(pair)),
                                Err(e)
                                    if e.kind()
                                        == std::io::ErrorKind::WouldBlock =>
                                {
                                    if *nonblocking {
                                        BlockStep::Done(Err(e))
                                    } else {
                                        BlockStep::WouldBlock
                                    }
                                }
                                Err(e) => BlockStep::Done(Err(e)),
                            }
                        })
                        .unwrap_or(BlockStep::Missing)
                    },
                )?;
                match outcome {
                    Some(Ok((stream, addr))) => {
                        if let Err(e) = stream.set_nonblocking(true) {
                            return Ok(io_result_err(host_io_error(&e)));
                        }
                        let new_id =
                            self.register_socket(SocketEntry::Stream {
                                socket: stream,
                                nonblocking: false,
                                read_timeout: None,
                                write_timeout: None,
                            });
                        Ok(io_result_ok(Value::Tuple(vec![
                            socket_value(new_id),
                            socket_addr_from_host(addr),
                        ])))
                    }
                    Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
                    None => Err(unsupported(
                        "TcpListener::accept on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::ListenerSocketAddr => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Listener { socket, .. } => {
                        Some(socket.local_addr())
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpListener::local_addr", |addr| {
                    socket_addr_from_host(addr)
                })
            }
            O::ListenerDuplicate => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let cloned = self
                    .with_socket(id, |entry| match entry {
                        SocketEntry::Listener {
                            socket,
                            nonblocking,
                        } => Some((socket.try_clone(), *nonblocking)),
                        _ => None,
                    })
                    .flatten();
                match cloned {
                    Some((Ok(socket), nonblocking)) => {
                        let new_id =
                            self.register_socket(SocketEntry::Listener {
                                socket,
                                nonblocking,
                            });
                        Ok(io_result_ok(socket_value(new_id)))
                    }
                    Some((Err(e), _)) => Ok(io_result_err(host_io_error(&e))),
                    None => Err(unsupported(
                        "TcpListener::try_clone on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::ListenerSetNonblocking => {
                arity(2)?;
                self.exec_set_nonblocking(frames, &vals, "TcpListener")
            }
            O::ListenerSetTtl => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let ttl = as_u128(&vals[1], "TcpListener::set_ttl")? as u32;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Listener { socket, .. } => {
                        Some(socket.set_ttl(ttl))
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpListener::set_ttl", |()| {
                    Value::Unit
                })
            }
            O::ListenerTtl => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Listener { socket, .. } => {
                        Some(socket.ttl())
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpListener::ttl", |v| {
                    Value::Uint(u128::from(v))
                })
            }
            O::ListenerSetOnlyV6 => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let only = as_bool(&vals[1], "TcpListener::set_only_v6")?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Listener { socket, .. } => {
                        Some(socket.set_only_v6(only))
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpListener::set_only_v6", |()| {
                    Value::Unit
                })
            }
            O::ListenerOnlyV6 => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Listener { socket, .. } => {
                        Some(socket.only_v6())
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpListener::only_v6", |v| {
                    Value::Bool(v)
                })
            }
            O::ListenerTakeError => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Listener { socket, .. } => {
                        Some(socket.take_error())
                    }
                    _ => None,
                });
                let result = flatten(result);
                match result {
                    Some(Ok(v)) => {
                        Ok(io_result_ok(self.option_error_value(v)))
                    }
                    Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
                    None => Err(unsupported(
                        "TcpListener::take_error on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::StreamConnect => {
                arity(1)?;
                let addrs =
                    match self.resolve_host_addrs(frames, self_ty, &vals[0])? {
                        Ok(addrs) => addrs,
                        Err(()) => {
                            return Ok(io_result_err(io_simple_error_value(
                                ERROR_KIND_INVALID_INPUT,
                            )))
                        }
                    };
                // std's `each_addr`, as for the listener.
                let mut last_err = None;
                for addr in addrs {
                    match std::net::TcpStream::connect(addr) {
                        Ok(stream) => {
                            if let Err(e) = stream.set_nonblocking(true) {
                                return Ok(io_result_err(host_io_error(&e)));
                            }
                            let id =
                                self.register_socket(SocketEntry::Stream {
                                    socket: stream,
                                    nonblocking: false,
                                    read_timeout: None,
                                    write_timeout: None,
                                });
                            return Ok(io_result_ok(socket_value(id)));
                        }
                        Err(e) => last_err = Some(e),
                    }
                }
                match last_err {
                    Some(e) => Ok(io_result_err(host_io_error(&e))),
                    None => Ok(io_result_err(io_simple_error_value(
                        ERROR_KIND_INVALID_INPUT,
                    ))),
                }
            }
            O::StreamConnectTimeout => {
                arity(2)?;
                let addr = self.host_addr_arg(frames, &vals[0])?;
                let (secs, nanos) =
                    duration_pair(&vals[1]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "TcpStream::connect_timeout",
                            "a Duration argument",
                            vals[1].kind_name(),
                        )
                    })?;
                let timeout =
                    std::time::Duration::new(secs as u64, nanos as u32);
                match std::net::TcpStream::connect_timeout(&addr, timeout) {
                    Ok(stream) => {
                        if let Err(e) = stream.set_nonblocking(true) {
                            return Ok(io_result_err(host_io_error(&e)));
                        }
                        let id = self.register_socket(SocketEntry::Stream {
                            socket: stream,
                            nonblocking: false,
                            read_timeout: None,
                            write_timeout: None,
                        });
                        Ok(io_result_ok(socket_value(id)))
                    }
                    Err(e) => Ok(io_result_err(host_io_error(&e))),
                }
            }
            O::StreamSocketAddr => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Stream { socket, .. } => {
                        Some(socket.local_addr())
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpStream::local_addr", |addr| {
                    socket_addr_from_host(addr)
                })
            }
            O::StreamPeerAddr => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Stream { socket, .. } => {
                        Some(socket.peer_addr())
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpStream::peer_addr", |addr| {
                    socket_addr_from_host(addr)
                })
            }
            O::StreamRead | O::StreamPeek => {
                arity(2)?;
                let peek = op == O::StreamPeek;
                let id = self.socket_id_of(frames, &vals[0])?;
                self.stream_read_into(frames, id, &vals[1], peek)
            }
            O::StreamReadVectored => {
                arity(2)?;
                // A vectored read may stop after the first
                // buffer: read into the first `IoSliceMut`'s
                // slice, exactly the short read the plain
                // `read` performs.
                let Some(fat) = self.first_io_slice(frames, &vals[1])?
                else {
                    return Ok(io_result_ok(Value::Uint(0)));
                };
                let redirected = vec![vals[0].clone(), fat];
                self.exec_net_sys(
                    frames,
                    O::StreamRead,
                    generic_args,
                    redirected,
                )
            }
            O::StreamWrite => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let bytes =
                    self.slice_bytes(frames, &vals[1], "TcpStream::write")?;
                self.stream_write_bytes(frames, id, &bytes)
            }
            O::StreamWriteVectored => {
                arity(2)?;
                // A vectored write may stop after the first
                // buffer: write the first `IoSlice`'s bytes,
                // exactly the short write the plain `write`
                // performs.
                let Some(fat) = self.first_io_slice(frames, &vals[1])?
                else {
                    return Ok(io_result_ok(Value::Uint(0)));
                };
                let redirected = vec![vals[0].clone(), fat];
                self.exec_net_sys(
                    frames,
                    O::StreamWrite,
                    generic_args,
                    redirected,
                )
            }
            O::StreamDuplicate => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let cloned = self
                    .with_socket(id, |entry| match entry {
                        SocketEntry::Stream {
                            socket,
                            nonblocking,
                            read_timeout,
                            write_timeout,
                        } => Some((
                            socket.try_clone(),
                            *nonblocking,
                            *read_timeout,
                            *write_timeout,
                        )),
                        _ => None,
                    })
                    .flatten();
                match cloned {
                    Some((Ok(socket), nonblocking, rt, wt)) => {
                        let new_id =
                            self.register_socket(SocketEntry::Stream {
                                socket,
                                nonblocking,
                                read_timeout: rt,
                                write_timeout: wt,
                            });
                        Ok(io_result_ok(socket_value(new_id)))
                    }
                    Some((Err(e), _, _, _)) => {
                        Ok(io_result_err(host_io_error(&e)))
                    }
                    None => Err(unsupported(
                        "TcpStream::try_clone on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::StreamShutdown => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let how = match &vals[1] {
                    Value::Enum { variant: 0, .. } => {
                        std::net::Shutdown::Read
                    }
                    Value::Enum { variant: 1, .. } => {
                        std::net::Shutdown::Write
                    }
                    Value::Enum { variant: 2, .. } => {
                        std::net::Shutdown::Both
                    }
                    Value::Uint(0) => std::net::Shutdown::Read,
                    Value::Uint(1) => std::net::Shutdown::Write,
                    Value::Uint(2) => std::net::Shutdown::Both,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "TcpStream::shutdown",
                            "a Shutdown value",
                            other.kind_name(),
                        ))
                    }
                };
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Stream { socket, .. } => {
                        Some(socket.shutdown(how))
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpStream::shutdown", |()| {
                    Value::Unit
                })
            }
            O::StreamSetNodelay => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let on = as_bool(&vals[1], "TcpStream::set_nodelay")?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Stream { socket, .. } => {
                        Some(socket.set_nodelay(on))
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpStream::set_nodelay", |()| {
                    Value::Unit
                })
            }
            O::StreamNodelay => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Stream { socket, .. } => {
                        Some(socket.nodelay())
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpStream::nodelay", |v| {
                    Value::Bool(v)
                })
            }
            O::StreamSetTtl => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let ttl = as_u128(&vals[1], "TcpStream::set_ttl")? as u32;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Stream { socket, .. } => {
                        Some(socket.set_ttl(ttl))
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpStream::set_ttl", |()| {
                    Value::Unit
                })
            }
            O::StreamTtl => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Stream { socket, .. } => {
                        Some(socket.ttl())
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "TcpStream::ttl", |v| {
                    Value::Uint(u128::from(v))
                })
            }
            O::StreamTakeError => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Stream { socket, .. } => {
                        Some(socket.take_error())
                    }
                    _ => None,
                });
                let result = flatten(result);
                match result {
                    Some(Ok(v)) => {
                        Ok(io_result_ok(self.option_error_value(v)))
                    }
                    Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
                    None => Err(unsupported(
                        "TcpStream::take_error on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::StreamSetNonblocking => {
                arity(2)?;
                self.exec_set_nonblocking(frames, &vals, "TcpStream")
            }
            O::StreamSetReadTimeout | O::StreamSetWriteTimeout => {
                arity(2)?;
                let read = op == O::StreamSetReadTimeout;
                let name = if read {
                    "TcpStream::set_read_timeout"
                } else {
                    "TcpStream::set_write_timeout"
                };
                let id = self.socket_id_of(frames, &vals[0])?;
                let dur = self.option_duration_arg(&vals[1], name)?;
                // std rejects a zero timeout as `InvalidInput`.
                if dur == Some(std::time::Duration::ZERO) {
                    return Ok(io_result_err(io_simple_error_value(
                        ERROR_KIND_INVALID_INPUT,
                    )));
                }
                let applied = self.with_socket(id, |entry| match entry {
                    SocketEntry::Stream {
                        read_timeout,
                        write_timeout,
                        ..
                    } => {
                        if read {
                            *read_timeout = dur;
                        } else {
                            *write_timeout = dur;
                        }
                        Some(())
                    }
                    _ => None,
                });
                let applied = applied.flatten();
                match applied {
                    Some(()) => Ok(io_result_ok(Value::Unit)),
                    None => Err(unsupported(format!(
                        "{name} on a handle that is not open"
                    ))),
                }
            }
            O::StreamReadTimeout | O::StreamWriteTimeout => {
                arity(1)?;
                let read = op == O::StreamReadTimeout;
                let name = if read {
                    "TcpStream::read_timeout"
                } else {
                    "TcpStream::write_timeout"
                };
                let id = self.socket_id_of(frames, &vals[0])?;
                let stored = self.with_socket(id, |entry| match entry {
                    SocketEntry::Stream {
                        read_timeout,
                        write_timeout,
                        ..
                    } => Some(if read {
                        *read_timeout
                    } else {
                        *write_timeout
                    }),
                    _ => None,
                });
                match stored {
                    Some(Some(dur)) => Ok(io_result_ok(
                        self.option_duration_value(dur)?,
                    )),
                    _ => Err(unsupported(format!(
                        "{name} on a handle that is not open"
                    ))),
                }
            }
            O::UdpBind => {
                arity(1)?;
                let addrs =
                    match self.resolve_host_addrs(frames, self_ty, &vals[0])? {
                        Ok(addrs) => addrs,
                        Err(()) => {
                            return Ok(io_result_err(io_simple_error_value(
                                ERROR_KIND_INVALID_INPUT,
                            )))
                        }
                    };
                let mut last_err = None;
                for addr in addrs {
                    match std::net::UdpSocket::bind(addr) {
                        Ok(socket) => {
                            if let Err(e) = socket.set_nonblocking(true) {
                                return Ok(io_result_err(host_io_error(&e)));
                            }
                            let id = self.register_socket(SocketEntry::Udp {
                                socket,
                                nonblocking: false,
                            });
                            return Ok(io_result_ok(socket_value(id)));
                        }
                        Err(e) => last_err = Some(e),
                    }
                }
                match last_err {
                    Some(e) => Ok(io_result_err(host_io_error(&e))),
                    None => Ok(io_result_err(io_simple_error_value(
                        ERROR_KIND_INVALID_INPUT,
                    ))),
                }
            }
            O::UdpSocketAddr => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Udp { socket, .. } => {
                        Some(socket.local_addr())
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "UdpSocket::local_addr", |addr| {
                    socket_addr_from_host(addr)
                })
            }
            O::UdpDuplicate => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let cloned = self
                    .with_socket(id, |entry| match entry {
                        SocketEntry::Udp {
                            socket,
                            nonblocking,
                        } => Some((socket.try_clone(), *nonblocking)),
                        _ => None,
                    })
                    .flatten();
                match cloned {
                    Some((Ok(socket), nonblocking)) => {
                        let new_id = self.register_socket(SocketEntry::Udp {
                            socket,
                            nonblocking,
                        });
                        Ok(io_result_ok(socket_value(new_id)))
                    }
                    Some((Err(e), _)) => Ok(io_result_err(host_io_error(&e))),
                    None => Err(unsupported(
                        "UdpSocket::try_clone on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::UdpRecvFrom => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let Value::FatRef { len, .. } = &vals[1] else {
                    return Err(InterpError::type_mismatch(
                        "UdpSocket::recv_from buffer",
                        "a byte slice pointer",
                        vals[1].kind_name(),
                    ));
                };
                let want = (*len).min(1 << 23) as usize;
                let outcome = self.net_block(
                    frames,
                    "UdpSocket::recv_from",
                    "no datagram is available and no pending thread can send",
                    |this| {
                        this.with_socket(id, |entry| {
                            let SocketEntry::Udp {
                                socket,
                                nonblocking,
                            } = entry
                            else {
                                return BlockStep::Missing;
                            };
                            let mut buf = vec![0u8; want];
                            match socket.recv_from(&mut buf) {
                                Ok((n, addr)) => {
                                    buf.truncate(n);
                                    BlockStep::Done(Ok((buf, addr)))
                                }
                                Err(e)
                                    if e.kind()
                                        == std::io::ErrorKind::WouldBlock =>
                                {
                                    if *nonblocking {
                                        BlockStep::Done(Err(e))
                                    } else {
                                        BlockStep::WouldBlock
                                    }
                                }
                                Err(e) => BlockStep::Done(Err(e)),
                            }
                        })
                        .unwrap_or(BlockStep::Missing)
                    },
                )?;
                match outcome {
                    Some(Ok((bytes, addr))) => {
                        let n = bytes.len();
                        self.write_bytes_into_slice(
                            frames,
                            &vals[1],
                            0,
                            &bytes,
                        )?;
                        Ok(io_result_ok(Value::Tuple(vec![
                            Value::Uint(n as u128),
                            socket_addr_from_host(addr),
                        ])))
                    }
                    Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
                    None => Err(unsupported(
                        "UdpSocket::recv_from on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::UdpRecv => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let Value::FatRef { len, .. } = &vals[1] else {
                    return Err(InterpError::type_mismatch(
                        "UdpSocket::recv buffer",
                        "a byte slice pointer",
                        vals[1].kind_name(),
                    ));
                };
                let want = (*len).min(1 << 23) as usize;
                let outcome = self.net_block(
                    frames,
                    "UdpSocket::recv",
                    "no datagram is available and no pending thread can send",
                    |this| {
                        this.with_socket(id, |entry| {
                            let SocketEntry::Udp {
                                socket,
                                nonblocking,
                            } = entry
                            else {
                                return BlockStep::Missing;
                            };
                            let mut buf = vec![0u8; want];
                            match socket.recv(&mut buf) {
                                Ok(n) => {
                                    buf.truncate(n);
                                    BlockStep::Done(Ok(buf))
                                }
                                Err(e)
                                    if e.kind()
                                        == std::io::ErrorKind::WouldBlock =>
                                {
                                    if *nonblocking {
                                        BlockStep::Done(Err(e))
                                    } else {
                                        BlockStep::WouldBlock
                                    }
                                }
                                Err(e) => BlockStep::Done(Err(e)),
                            }
                        })
                        .unwrap_or(BlockStep::Missing)
                    },
                )?;
                match outcome {
                    Some(Ok(bytes)) => {
                        let n = bytes.len();
                        self.write_bytes_into_slice(
                            frames,
                            &vals[1],
                            0,
                            &bytes,
                        )?;
                        Ok(io_result_ok(Value::Uint(n as u128)))
                    }
                    Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
                    None => Err(unsupported(
                        "UdpSocket::recv on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::UdpSendTo => {
                arity(3)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let bytes =
                    self.slice_bytes(frames, &vals[1], "UdpSocket::send_to")?;
                let addr = self.host_addr_arg(frames, &vals[2])?;
                let outcome = self.net_block(
                    frames,
                    "UdpSocket::send_to",
                    "the send buffer is full and no pending thread can receive",
                    |this| {
                        this.with_socket(id, |entry| {
                            let SocketEntry::Udp {
                                socket,
                                nonblocking,
                            } = entry
                            else {
                                return BlockStep::Missing;
                            };
                            match socket.send_to(&bytes, addr) {
                                Ok(n) => BlockStep::Done(Ok(n)),
                                Err(e)
                                    if e.kind()
                                        == std::io::ErrorKind::WouldBlock =>
                                {
                                    if *nonblocking {
                                        BlockStep::Done(Err(e))
                                    } else {
                                        BlockStep::WouldBlock
                                    }
                                }
                                Err(e) => BlockStep::Done(Err(e)),
                            }
                        })
                        .unwrap_or(BlockStep::Missing)
                    },
                )?;
                match outcome {
                    Some(Ok(n)) => Ok(io_result_ok(Value::Uint(n as u128))),
                    Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
                    None => Err(unsupported(
                        "UdpSocket::send_to on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::UdpSend => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let bytes =
                    self.slice_bytes(frames, &vals[1], "UdpSocket::send")?;
                let outcome = self.net_block(
                    frames,
                    "UdpSocket::send",
                    "the send buffer is full and no pending thread can receive",
                    |this| {
                        this.with_socket(id, |entry| {
                            let SocketEntry::Udp {
                                socket,
                                nonblocking,
                            } = entry
                            else {
                                return BlockStep::Missing;
                            };
                            match socket.send(&bytes) {
                                Ok(n) => BlockStep::Done(Ok(n)),
                                Err(e)
                                    if e.kind()
                                        == std::io::ErrorKind::WouldBlock =>
                                {
                                    if *nonblocking {
                                        BlockStep::Done(Err(e))
                                    } else {
                                        BlockStep::WouldBlock
                                    }
                                }
                                Err(e) => BlockStep::Done(Err(e)),
                            }
                        })
                        .unwrap_or(BlockStep::Missing)
                    },
                )?;
                match outcome {
                    Some(Ok(n)) => Ok(io_result_ok(Value::Uint(n as u128))),
                    Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
                    None => Err(unsupported(
                        "UdpSocket::send on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::UdpConnect => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let addrs =
                    match self.resolve_host_addrs(frames, self_ty, &vals[1])? {
                        Ok(addrs) => addrs,
                        Err(()) => {
                            return Ok(io_result_err(io_simple_error_value(
                                ERROR_KIND_INVALID_INPUT,
                            )))
                        }
                    };
                let mut last: Option<std::io::Result<()>> = None;
                for addr in addrs {
                    let result = self.with_socket(id, |entry| match entry {
                        SocketEntry::Udp { socket, .. } => {
                            Some(socket.connect(addr))
                        }
                        _ => None,
                    });
                    let result = flatten(result);
                    match result {
                        Some(Ok(())) => {
                            return Ok(io_result_ok(Value::Unit))
                        }
                        Some(Err(e)) => last = Some(Err(e)),
                        None => {
                            return Err(unsupported(
                                "UdpSocket::connect on a handle that is not open"
                                    .to_string(),
                            ))
                        }
                    }
                }
                match last {
                    Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
                    _ => Ok(io_result_err(io_simple_error_value(
                        ERROR_KIND_INVALID_INPUT,
                    ))),
                }
            }
            O::UdpSetBroadcast => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let on = as_bool(&vals[1], "UdpSocket::set_broadcast")?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Udp { socket, .. } => {
                        Some(socket.set_broadcast(on))
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "UdpSocket::set_broadcast", |()| {
                    Value::Unit
                })
            }
            O::UdpBroadcast => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Udp { socket, .. } => {
                        Some(socket.broadcast())
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "UdpSocket::broadcast", |v| {
                    Value::Bool(v)
                })
            }
            O::UdpSetTtl => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let ttl = as_u128(&vals[1], "UdpSocket::set_ttl")? as u32;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Udp { socket, .. } => {
                        Some(socket.set_ttl(ttl))
                    }
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "UdpSocket::set_ttl", |()| {
                    Value::Unit
                })
            }
            O::UdpTtl => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Udp { socket, .. } => Some(socket.ttl()),
                    _ => None,
                });
                let result = flatten(result);
                self.net_result(result, "UdpSocket::ttl", |v| {
                    Value::Uint(u128::from(v))
                })
            }
            O::UdpTakeError => {
                arity(1)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let result = self.with_socket(id, |entry| match entry {
                    SocketEntry::Udp { socket, .. } => {
                        Some(socket.take_error())
                    }
                    _ => None,
                });
                let result = flatten(result);
                match result {
                    Some(Ok(v)) => {
                        Ok(io_result_ok(self.option_error_value(v)))
                    }
                    Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
                    None => Err(unsupported(
                        "UdpSocket::take_error on a handle that is not open"
                            .to_string(),
                    )),
                }
            }
            O::UdpSetNonblocking => {
                arity(2)?;
                self.exec_set_nonblocking(frames, &vals, "UdpSocket")
            }
        }
    }

    /// One `TcpStream` read (or peek) into the guest byte
    /// slice `buf`: the shared core of the sys `read` /
    /// `peek` operations and the `std::io::Read::read`
    /// diversion. Returns the guest `io::Result<usize>`.
    fn stream_read_into(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
        buf: &Value,
        peek: bool,
    ) -> InterpResult<Value> {
        let Value::FatRef { len, .. } = buf else {
            return Err(InterpError::type_mismatch(
                "TcpStream::read buffer",
                "a byte slice pointer",
                buf.kind_name(),
            ));
        };
        // As for the file model: one `read` may return
        // short, and the host buffer is capped so a
        // sparse buffer value cannot exhaust the host.
        let want = (*len).min(1 << 23) as usize;
        let outcome = self.net_block(
            frames,
            "TcpStream::read",
            "no data is available and no pending thread can send",
            |this| {
                this.with_socket(id, |entry| {
                    let SocketEntry::Stream {
                        socket,
                        nonblocking,
                        read_timeout,
                        ..
                    } = entry
                    else {
                        return BlockStep::Missing;
                    };
                    let mut buf = vec![0u8; want];
                    let read = if peek {
                        socket.peek(&mut buf)
                    } else {
                        use std::io::Read as _;
                        socket.read(&mut buf)
                    };
                    match read {
                        Ok(n) => {
                            buf.truncate(n);
                            BlockStep::Done(Ok(buf))
                        }
                        Err(e)
                            if e.kind()
                                == std::io::ErrorKind::WouldBlock =>
                        {
                            if *nonblocking || read_timeout.is_some() {
                                BlockStep::Done(Err(e))
                            } else {
                                BlockStep::WouldBlock
                            }
                        }
                        Err(e) => BlockStep::Done(Err(e)),
                    }
                })
                .unwrap_or(BlockStep::Missing)
            },
        )?;
        match outcome {
            Some(Ok(bytes)) => {
                let n = bytes.len();
                self.write_bytes_into_slice(frames, buf, 0, &bytes)?;
                Ok(io_result_ok(Value::Uint(n as u128)))
            }
            Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
            None => Err(unsupported(
                "TcpStream::read on a handle that is not open".to_string(),
            )),
        }
    }

    /// One `TcpStream` write of `bytes`: the shared core of
    /// the sys `write` operation and the `std::io::Write::write`
    /// diversion. Returns the guest `io::Result<usize>`.
    fn stream_write_bytes(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
        bytes: &[u8],
    ) -> InterpResult<Value> {
        let bytes = bytes.to_vec();
        let outcome = self.net_block(
            frames,
            "TcpStream::write",
            "the send buffer is full and no pending thread can receive",
            |this| {
                this.with_socket(id, |entry| {
                    let SocketEntry::Stream {
                        socket,
                        nonblocking,
                        write_timeout,
                        ..
                    } = entry
                    else {
                        return BlockStep::Missing;
                    };
                    use std::io::Write as _;
                    match socket.write(&bytes) {
                        Ok(n) => BlockStep::Done(Ok(n)),
                        Err(e)
                            if e.kind()
                                == std::io::ErrorKind::WouldBlock =>
                        {
                            if *nonblocking || write_timeout.is_some() {
                                BlockStep::Done(Err(e))
                            } else {
                                BlockStep::WouldBlock
                            }
                        }
                        Err(e) => BlockStep::Done(Err(e)),
                    }
                })
                .unwrap_or(BlockStep::Missing)
            },
        )?;
        match outcome {
            Some(Ok(n)) => Ok(io_result_ok(Value::Uint(n as u128))),
            Some(Err(e)) => Ok(io_result_err(host_io_error(&e))),
            None => Err(unsupported(
                "TcpStream::write on a handle that is not open".to_string(),
            )),
        }
    }

    /// Execute a `std::io::Read` / `Write` method call on a
    /// `std::net::TcpStream` receiver (see the diversion in
    /// `file_io_terminator` in `mir_convert` and
    /// `is_stream_io_call` in `builtins::lookup`): the trait
    /// impl bodies in std call the sys socket layer, which
    /// std's own compilation inlines away, so the leaf
    /// methods are served here directly. The default trait
    /// bodies above them (`read_exact`, `write_all`, ...)
    /// are ordinary converted code and call these leaves.
    pub(super) fn exec_stream_io(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: Vec<Value>,
    ) -> InterpResult<Value> {
        let arity = |n: usize| -> InterpResult<()> {
            if vals.len() == n {
                Ok(())
            } else {
                Err(InterpError::type_mismatch(
                    format!("stream I/O builtin {name}"),
                    format!("{n} argument(s)"),
                    format!("{} argument(s)", vals.len()),
                ))
            }
        };
        match name {
            "std::io::Read::read" => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                self.stream_read_into(frames, id, &vals[1], false)
            }
            "std::io::Read::read_vectored" => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                match self.first_io_slice(frames, &vals[1])? {
                    Some(fat) => self.stream_read_into(frames, id, &fat, false),
                    None => Ok(io_result_ok(Value::Uint(0))),
                }
            }
            "std::io::Write::write" => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                let bytes =
                    self.slice_bytes(frames, &vals[1], "TcpStream::write")?;
                self.stream_write_bytes(frames, id, &bytes)
            }
            "std::io::Write::write_vectored" => {
                arity(2)?;
                let id = self.socket_id_of(frames, &vals[0])?;
                match self.first_io_slice(frames, &vals[1])? {
                    Some(fat) => {
                        let bytes = self.slice_bytes(
                            frames,
                            &fat,
                            "TcpStream::write_vectored",
                        )?;
                        self.stream_write_bytes(frames, id, &bytes)
                    }
                    None => Ok(io_result_ok(Value::Uint(0))),
                }
            }
            "std::io::Write::flush" => {
                arity(1)?;
                // A `TcpStream` flush is std's no-op: the
                // stream is unbuffered.
                Ok(io_result_ok(Value::Unit))
            }
            _ => Err(unsupported(format!(
                "stream I/O call {name} is not modelled"
            ))),
        }
    }

    /// `set_nonblocking` for any socket kind: the host socket
    /// stays nonblocking (the scheduler's waits depend on
    /// it); the call only records the guest-visible mode,
    /// which decides whether a would-block outcome waits or
    /// surfaces as the `WouldBlock` error.
    fn exec_set_nonblocking(
        &mut self,
        frames: &mut Vec<Frame>,
        vals: &[Value],
        kind: &str,
    ) -> InterpResult<Value> {
        let id = self.socket_id_of(frames, &vals[0])?;
        let on = as_bool(&vals[1], "set_nonblocking")?;
        let applied = self.with_socket(id, |entry| {
            entry.set_nonblocking(on);
        });
        match applied {
            Some(()) => Ok(io_result_ok(Value::Unit)),
            None => Err(unsupported(format!(
                "{kind}::set_nonblocking on a handle that is not open"
            ))),
        }
    }

    /// The byte slice the first `IoSlice` / `IoSliceMut` of a
    /// vectored-I/O buffer list denotes: the element is an
    /// aggregate whose fields carry the slice's fat pointer
    /// (the raw slice pointer the type stores). `None` for an
    /// empty list.
    fn first_io_slice(
        &self,
        frames: &[Frame],
        bufs: &Value,
    ) -> InterpResult<Option<Value>> {
        fn find_fat(value: &Value) -> Option<Value> {
            match value {
                Value::FatRef { .. } => Some(value.clone()),
                Value::Enum { fields, .. } | Value::Tuple(fields) => {
                    fields.iter().find_map(find_fat)
                }
                _ => None,
            }
        }
        let (items, _) = self.slice_elements(frames, bufs)?;
        for item in &items {
            if let Some(fat) = find_fat(item) {
                return Ok(Some(fat));
            }
        }
        Ok(None)
    }
}

/// Flatten a `with_socket` outcome: the entry existed and
/// produced a host result, or it did not.
fn flatten<T>(
    result: Option<Option<std::io::Result<T>>>,
) -> Option<std::io::Result<T>> {
    result.flatten()
}

/// A boolean argument of a socket option call.
fn as_bool(value: &Value, name: &str) -> InterpResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Uint(v) => Ok(*v != 0),
        Value::Int(v) => Ok(*v != 0),
        other => Err(InterpError::type_mismatch(
            name,
            "a boolean argument",
            other.kind_name(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The handle value the model builds unwraps back to its
    /// table id through the shared descriptor-slot reader.
    #[test]
    fn socket_value_roundtrips_handle_id() {
        assert_eq!(crate::interp::file_handle_id(&socket_value(3)), Some(3));
        assert_eq!(
            crate::interp::file_handle_id(&socket_value(41)),
            Some(41)
        );
    }
}
