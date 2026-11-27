//! The host-platform seam for the interpreter's host-side
//! services: the places where serving a guest operation
//! reaches for a host facility whose shape differs between
//! Unix and Windows hosts.
//!
//! The interpreter's value model speaks byte strings: guest
//! paths, `OsString`s, environment keys and values are all
//! byte sequences. On a Unix host that representation is
//! exact, because a host `OsStr` *is* a byte string
//! (`std::os::unix::ffi::OsStrExt`). On a Windows host an
//! `OsStr` is WTF-8 and std's public API exposes no byte
//! view of it, so the conversions here go through UTF-8
//! and report `None` where the bytes have no host-string
//! form. Callers turn `None` into the model's named
//! unsupported error (or, for a pure lookup, the lookup's
//! own miss): nothing in this module ever converts
//! lossily, invents a value, or reports a success the host
//! did not give.
//!
//! The same treatment covers the other Unix-shaped host
//! facilities the model uses: descriptor duplication of
//! the standard input and of a listening socket, the pipe
//! ends' transfer into the file model, Unix mode
//! permissions, and symlink creation (one operation on
//! Unix; two on Windows, split by the target's kind). The
//! same treatment covers the libc calls that exist only
//! on some Unix hosts (`prctl`, `clearenv`): a host whose
//! libc has no such call reports `None`, and the caller
//! names the operation unsupported — an unconditional
//! extern declaration would leave the symbol undefined
//! at link time on those hosts (macOS libc exports
//! neither).

use std::borrow::Cow;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

/// The bytes of a host `OsStr`: always available on Unix;
/// on Windows, available exactly when the string is valid
/// UTF-8 (a WTF-8 string holding unpaired surrogates has
/// no byte form in std's public API).
#[cfg(unix)]
pub(crate) fn os_str_bytes(s: &OsStr) -> Option<Cow<'_, [u8]>> {
    use std::os::unix::ffi::OsStrExt as _;
    Some(Cow::Borrowed(s.as_bytes()))
}

#[cfg(windows)]
pub(crate) fn os_str_bytes(s: &OsStr) -> Option<Cow<'_, [u8]>> {
    s.to_str().map(|text| Cow::Borrowed(text.as_bytes()))
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn os_str_bytes(s: &OsStr) -> Option<Cow<'_, [u8]>> {
    let _ = s;
    None
}

/// The host `OsStr` a byte string denotes: on Unix, the
/// bytes themselves; on Windows, the string when the
/// bytes are valid UTF-8 (bytes that are not UTF-8 denote
/// no string this host can express).
#[cfg(unix)]
pub(crate) fn os_str_from_bytes(bytes: &[u8]) -> Option<&OsStr> {
    use std::os::unix::ffi::OsStrExt as _;
    Some(OsStr::from_bytes(bytes))
}

#[cfg(windows)]
pub(crate) fn os_str_from_bytes(bytes: &[u8]) -> Option<&OsStr> {
    std::str::from_utf8(bytes).ok().map(OsStr::new)
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn os_str_from_bytes(bytes: &[u8]) -> Option<&OsStr> {
    let _ = bytes;
    None
}

/// The owned host string a byte string denotes (see
/// [`os_str_from_bytes`]).
pub(crate) fn os_string_from_bytes(bytes: &[u8]) -> Option<OsString> {
    os_str_from_bytes(bytes).map(OsStr::to_os_string)
}

/// The host path a byte string denotes (see
/// [`os_str_from_bytes`]).
pub(crate) fn path_from_bytes(bytes: &[u8]) -> Option<PathBuf> {
    os_string_from_bytes(bytes).map(PathBuf::from)
}

/// The bytes of a host path (see [`os_str_bytes`]).
pub(crate) fn path_bytes(path: &Path) -> Option<Vec<u8>> {
    os_str_bytes(path.as_os_str()).map(Cow::into_owned)
}

/// Host permissions carrying a Unix mode: the mode itself
/// on Unix. Windows permissions have no mode, so there is
/// no equivalent and the caller reports the operation
/// unsupported.
#[cfg(unix)]
pub(crate) fn permissions_from_mode(mode: u32) -> Option<std::fs::Permissions> {
    use std::os::unix::fs::PermissionsExt as _;
    Some(std::fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
pub(crate) fn permissions_from_mode(mode: u32) -> Option<std::fs::Permissions> {
    let _ = mode;
    None
}

/// Create the symlink `link` to `original` on the host:
/// one operation on Unix; on Windows the file and
/// directory forms are distinct operations, chosen by the
/// target's current kind — a target that does not exist
/// has no honest kind to choose, so the operation is
/// reported unsupported rather than guessed.
#[cfg(unix)]
pub(crate) fn symlink(original: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(original, link)
}

#[cfg(windows)]
pub(crate) fn symlink(original: &Path, link: &Path) -> std::io::Result<()> {
    match std::fs::metadata(original) {
        Ok(md) if md.is_dir() => std::os::windows::fs::symlink_dir(original, link),
        Ok(_) => std::os::windows::fs::symlink_file(original, link),
        Err(_) => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "symlink: the target does not exist, so its Windows \
             symlink kind cannot be chosen",
        )),
    }
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn symlink(original: &Path, link: &Path) -> std::io::Result<()> {
    let _ = (original, link);
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "symlink is not modelled on this host platform",
    ))
}

/// A duplicate of this process's standard input as a host
/// `File`, when the descriptor/handle can be duplicated:
/// the guest's stdin reads go through the ordinary file
/// model, so the entry the model registers is a real file
/// over the same host stream.
#[cfg(unix)]
pub(crate) fn stdin_dup_file() -> Option<std::fs::File> {
    use std::os::fd::AsFd as _;
    std::io::stdin()
        .as_fd()
        .try_clone_to_owned()
        .ok()
        .map(std::fs::File::from)
}

#[cfg(windows)]
pub(crate) fn stdin_dup_file() -> Option<std::fs::File> {
    use std::os::windows::io::AsHandle as _;
    std::io::stdin()
        .as_handle()
        .try_clone_to_owned()
        .ok()
        .map(std::fs::File::from)
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn stdin_dup_file() -> Option<std::fs::File> {
    None
}

/// Write `bytes` to a listening socket, for the libc
/// descriptor-write model: a listening socket has no write
/// of its own in std's model types, so the kernel's own
/// answer comes from a duplicate of the host descriptor.
/// A Windows socket is not a file descriptor and cannot be
/// duplicated into a `File`, so there is no equivalent:
/// the write is reported unsupported.
#[cfg(unix)]
pub(crate) fn listener_socket_write(
    socket: &std::net::TcpListener,
    bytes: &[u8],
) -> std::io::Result<usize> {
    use std::io::Write as _;
    use std::os::fd::AsFd as _;
    socket.as_fd().try_clone_to_owned().and_then(|owned| {
        let mut file = std::fs::File::from(owned);
        file.write(bytes)
    })
}

#[cfg(not(unix))]
pub(crate) fn listener_socket_write(
    socket: &std::net::TcpListener,
    bytes: &[u8],
) -> std::io::Result<usize> {
    let _ = (socket, bytes);
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "libc::write on a listening socket is not modelled on this host platform",
    ))
}

/// Read from a listening socket, for the libc
/// descriptor-read model (see [`listener_socket_write`]).
#[cfg(unix)]
pub(crate) fn listener_socket_read(
    socket: &std::net::TcpListener,
    buf: &mut [u8],
) -> std::io::Result<usize> {
    use std::io::Read as _;
    use std::os::fd::AsFd as _;
    socket.as_fd().try_clone_to_owned().and_then(|owned| {
        let mut file = std::fs::File::from(owned);
        file.read(buf)
    })
}

#[cfg(not(unix))]
pub(crate) fn listener_socket_read(
    socket: &std::net::TcpListener,
    buf: &mut [u8],
) -> std::io::Result<usize> {
    let _ = (socket, buf);
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "libc::read on a listening socket is not modelled on this host platform",
    ))
}
/// The host C library's `prctl`, backing the libc
/// builtin of the same name: a Linux-only process
/// control call, so hosts without it report `None` and
/// the caller names the operation unsupported. Where it
/// exists, the C function is variadic; it is declared
/// here with its full five-word shape, and the caller
/// zero-pads the arguments the guest's call did not
/// supply, which is the value a variadic callee reads
/// for them.
#[cfg(target_os = "linux")]
pub(crate) fn host_prctl(
    option: std::ffi::c_int,
    arg2: std::ffi::c_ulong,
    arg3: std::ffi::c_ulong,
    arg4: std::ffi::c_ulong,
    arg5: std::ffi::c_ulong,
) -> Option<std::ffi::c_int> {
    extern "C" {
        #[link_name = "prctl"]
        fn prctl(
            option: std::ffi::c_int,
            arg2: std::ffi::c_ulong,
            arg3: std::ffi::c_ulong,
            arg4: std::ffi::c_ulong,
            arg5: std::ffi::c_ulong,
        ) -> std::ffi::c_int;
    }
    Some(unsafe { prctl(option, arg2, arg3, arg4, arg5) })
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn host_prctl(
    option: std::ffi::c_int,
    arg2: std::ffi::c_ulong,
    arg3: std::ffi::c_ulong,
    arg4: std::ffi::c_ulong,
    arg5: std::ffi::c_ulong,
) -> Option<std::ffi::c_int> {
    let _ = (option, arg2, arg3, arg4, arg5);
    None
}

/// The host C library's `clearenv`, backing the libc
/// builtin of the same name: a glibc/BSD libc extension
/// with no macOS or Windows equivalent, so hosts without
/// it report `None` and the caller names the operation
/// unsupported. The guest's environment model is the
/// host environment itself (the enumeration and
/// single-key builtins read and mutate it directly), so
/// where the call exists, clearing it here clears the
/// guest's environment.
#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
pub(crate) fn host_clearenv() -> Option<std::ffi::c_int> {
    extern "C" {
        #[link_name = "clearenv"]
        fn clearenv() -> std::ffi::c_int;
    }
    Some(unsafe { clearenv() })
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "dragonfly"
)))]
pub(crate) fn host_clearenv() -> Option<std::ffi::c_int> {
    None
}

/// A duplicate of a listening socket as a second
/// `TcpListener` over the same host socket, for the
/// descriptor-duplication model (`dup` of a listener):
/// the duplicate shares the socket itself -- its address,
/// its queue, its options. The duplication goes through
/// the descriptor on Unix and through the socket handle
/// on Windows.
#[cfg(unix)]
pub(crate) fn listener_dup(
    socket: &std::net::TcpListener,
) -> Option<std::net::TcpListener> {
    use std::os::fd::AsFd as _;
    socket.as_fd().try_clone_to_owned().ok().map(|owned| {
        use std::os::fd::FromRawFd as _;
        use std::os::fd::IntoRawFd as _;
        // SAFETY: `owned` is the duplicate this call just
        // made; `into_raw_fd` relinquishes it to the
        // listener constructed from it, which closes it
        // on drop.
        unsafe { std::net::TcpListener::from_raw_fd(owned.into_raw_fd()) }
    })
}

#[cfg(windows)]
pub(crate) fn listener_dup(
    socket: &std::net::TcpListener,
) -> Option<std::net::TcpListener> {
    use std::os::windows::io::AsSocket as _;
    socket
        .as_socket()
        .try_clone_to_owned()
        .ok()
        .map(std::net::TcpListener::from)
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn listener_dup(
    socket: &std::net::TcpListener,
) -> Option<std::net::TcpListener> {
    let _ = socket;
    None
}

/// A host `File` over the read end of a host pipe, for
/// the libc `pipe` model: the pipe end enters the file
/// model as a real file over the same pipe, through the
/// host's descriptor (Unix) or handle (Windows)
/// duplication, as with [`stdin_dup_file`].
#[cfg(unix)]
pub(crate) fn file_from_pipe_reader(
    reader: std::io::PipeReader,
) -> Option<std::fs::File> {
    use std::os::fd::AsFd as _;
    reader
        .as_fd()
        .try_clone_to_owned()
        .ok()
        .map(std::fs::File::from)
}

#[cfg(windows)]
pub(crate) fn file_from_pipe_reader(
    reader: std::io::PipeReader,
) -> Option<std::fs::File> {
    use std::os::windows::io::AsHandle as _;
    reader
        .as_handle()
        .try_clone_to_owned()
        .ok()
        .map(std::fs::File::from)
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn file_from_pipe_reader(
    reader: std::io::PipeReader,
) -> Option<std::fs::File> {
    let _ = reader;
    None
}

/// A host `File` over the write end of a host pipe (see
/// [`file_from_pipe_reader`]).
#[cfg(unix)]
pub(crate) fn file_from_pipe_writer(
    writer: std::io::PipeWriter,
) -> Option<std::fs::File> {
    use std::os::fd::AsFd as _;
    writer
        .as_fd()
        .try_clone_to_owned()
        .ok()
        .map(std::fs::File::from)
}

#[cfg(windows)]
pub(crate) fn file_from_pipe_writer(
    writer: std::io::PipeWriter,
) -> Option<std::fs::File> {
    use std::os::windows::io::AsHandle as _;
    writer
        .as_handle()
        .try_clone_to_owned()
        .ok()
        .map(std::fs::File::from)
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn file_from_pipe_writer(
    writer: std::io::PipeWriter,
) -> Option<std::fs::File> {
    let _ = writer;
    None
}
