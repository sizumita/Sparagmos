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
//! the standard input and of a listening socket, Unix mode
//! permissions, and symlink creation (one operation on
//! Unix; two on Windows, split by the target's kind).

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
