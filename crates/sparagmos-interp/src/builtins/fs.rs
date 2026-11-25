//! Filesystem syscall execution and filesystem value constructors.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    Frame, Interpreter,
};
use crate::value::Value;

use super::*;

impl<'a> Interpreter<'a> {
    /// Execute a `std::sys::fs` leaf operation on the host
    /// filesystem. Every operation first rejects a path
    /// containing a NUL byte with `InvalidInput`, exactly as
    /// std's own CString conversion does before any syscall;
    /// the rest are the host operations themselves, with the
    /// host's errno returned in std's `Os` error shape (the
    /// guest's own `Error::kind` body maps it from there).
    pub(super) fn exec_fs_sys(
        &self,
        frames: &[Frame],
        op: FsSysOp,
        vals: Vec<Value>,
    ) -> InterpResult<Value> {
        let mut vals = vals.into_iter();
        let mut next_path = |vals: &mut std::vec::IntoIter<Value>| -> InterpResult<Vec<u8>> {
            let value = vals.next().ok_or_else(|| {
                InterpError::type_mismatch("fs operation", "path argument", "missing")
            })?;
            self.path_bytes(frames, &value)
        };
        let first = next_path(&mut vals)?;
        if first.contains(&0) {
            return Ok(io_result_err(io_simple_error_value(ERROR_KIND_INVALID_INPUT)));
        }
        let second = if matches!(
            op,
            FsSysOp::Copy | FsSysOp::HardLink | FsSysOp::Rename | FsSysOp::Symlink
        ) {
            let bytes = next_path(&mut vals)?;
            if bytes.contains(&0) {
                return Ok(io_result_err(io_simple_error_value(ERROR_KIND_INVALID_INPUT)));
            }
            Some(bytes)
        } else {
            None
        };
        let first = host_path(&first);
        Ok(match op {
            FsSysOp::Metadata => {
                io_result_from_host(std::fs::metadata(&first).map(|md| metadata_value(&md)))
            }
            FsSysOp::SymlinkMetadata => io_result_from_host(
                std::fs::symlink_metadata(&first).map(|md| metadata_value(&md)),
            ),
            FsSysOp::Canonicalize => io_result_from_host(
                std::fs::canonicalize(&first)
                    .map(|p| pathbuf_value(&host_path_bytes(&p))),
            ),
            FsSysOp::Copy => {
                let to = host_path(second.as_ref().expect("two-path operation"));
                io_result_from_host(std::fs::copy(&first, &to).map(|n| Value::Uint(n as u128)))
            }
            FsSysOp::HardLink => {
                let to = host_path(second.as_ref().expect("two-path operation"));
                io_result_from_host(std::fs::hard_link(&first, &to).map(|()| Value::Unit))
            }
            FsSysOp::ReadDir => {
                // The entries are snapshotted eagerly: the
                // modelled `ReadDir` value is the entry list
                // plus the read index (the owned-iterator
                // state), each entry already its
                // `io::Result<DirEntry>` value, so a per-entry
                // host error surfaces at its own position
                // exactly as the native iterator yields it.
                match std::fs::read_dir(&first) {
                    Ok(entries) => {
                        let mut items = Vec::new();
                        for entry in entries {
                            items.push(match entry {
                                Ok(ent) => io_result_ok(dir_entry_value(&ent)),
                                Err(e) => io_result_err(host_io_error(&e)),
                            });
                        }
                        io_result_ok(read_dir_value(items))
                    }
                    Err(e) => io_result_err(host_io_error(&e)),
                }
            }
            FsSysOp::ReadLink => io_result_from_host(
                std::fs::read_link(&first).map(|p| pathbuf_value(&host_path_bytes(&p))),
            ),
            FsSysOp::RemoveDir => {
                io_result_from_host(std::fs::remove_dir(&first).map(|()| Value::Unit))
            }
            FsSysOp::RemoveDirAll => {
                io_result_from_host(std::fs::remove_dir_all(&first).map(|()| Value::Unit))
            }
            FsSysOp::RemoveFile => {
                io_result_from_host(std::fs::remove_file(&first).map(|()| Value::Unit))
            }
            FsSysOp::Rename => {
                let to = host_path(second.as_ref().expect("two-path operation"));
                io_result_from_host(std::fs::rename(&first, &to).map(|()| Value::Unit))
            }
            FsSysOp::SetPermissions => {
                // The second argument is the `Permissions`
                // value, not a path: { inner: { mode } } in
                // std's shape.
                let perms = vals.next().ok_or_else(|| {
                    InterpError::type_mismatch(
                        "set_permissions",
                        "permissions argument",
                        "missing",
                    )
                })?;
                let mode = permissions_mode(&perms).ok_or_else(|| {
                    unsupported("set_permissions: unrecognised Permissions shape".to_string())
                })?;
                io_result_from_host(
                    std::fs::set_permissions(
                        &first,
                        std::os::unix::fs::PermissionsExt::from_mode(mode),
                    )
                    .map(|()| Value::Unit),
                )
            }
            FsSysOp::Symlink => {
                let to = host_path(second.as_ref().expect("two-path operation"));
                io_result_from_host(
                    std::os::unix::fs::symlink(&first, &to).map(|()| Value::Unit),
                )
            }
        })
    }
}

/// A guest `std::fs::File` value denoting file-table id
/// `id`: the public wrapper chain, one single-field struct
/// per layer — `File { inner: sys File { inner: FileDesc {
/// fd: OwnedFd { fd } } }` — with the table id in the
/// descriptor slot.
/// Read from `file` one byte at a time until `delim` has
/// been read or the stream ends, appending the bytes read:
/// a line read over an unbuffered host file, where reading
/// past the delimiter in a chunk could not be handed back
/// to the stream for the next read.
pub(super) fn read_until_byte(
    file: &mut std::fs::File,
    bytes: &mut Vec<u8>,
    delim: u8,
) -> std::io::Result<()> {
    let mut byte = [0u8; 1];
    loop {
        match <std::fs::File as std::io::Read>::read(file, &mut byte) {
            Ok(0) => return Ok(()),
            Ok(_) => {
                bytes.push(byte[0]);
                if byte[0] == delim {
                    return Ok(());
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::Interrupted =>
            {
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

pub(super) fn file_value(id: u64) -> Value {
    let mut value = Value::Int(id as i128);
    for _ in 0..4 {
        value = Value::Enum {
            variant: 0,
            fields: vec![value],
        };
    }
    value
}

/// The flags and platform parameters of a guest `OpenOptions`
/// value: the public struct wraps the sys options in a single
/// field; the sys value's fields in order are `read`, `write`,
/// `append`, `truncate`, `create`, `create_new` (booleans),
/// then the unix `custom_flags` and `mode`. `None` for any
/// other shape.
pub(super) fn open_options_of(
    value: &Value,
) -> Option<(bool, bool, bool, bool, bool, bool, i32, u32)> {
    let inner = match value {
        Value::Enum { fields, .. } | Value::Tuple(fields)
            if fields.len() == 1 =>
        {
            &fields[0]
        }
        _ => value,
    };
    let fields = match inner {
        Value::Enum { fields, .. } | Value::Tuple(fields) => fields,
        _ => return None,
    };
    if fields.len() != 8 {
        return None;
    }
    let flag = |i: usize| match fields[i] {
        Value::Bool(v) => Some(v),
        _ => None,
    };
    let custom_flags = match fields[6] {
        Value::Int(v) => v as i32,
        Value::Uint(v) => v as i32,
        _ => return None,
    };
    let mode = match fields[7] {
        Value::Uint(v) => v as u32,
        Value::Int(v) => v as u32,
        _ => return None,
    };
    Some((
        flag(0)?,
        flag(1)?,
        flag(2)?,
        flag(3)?,
        flag(4)?,
        flag(5)?,
        custom_flags,
        mode,
    ))
}

/// `value` (an element sequence: an array tuple, a vector's
/// element list, a sparse vector, a byte block) with the
/// elements at `offset..offset + bytes.len()` replaced by
/// `bytes` as `u8` values, preserving the container's shape;
/// `None` when the range is out of bounds or the value is not
/// an element sequence. A sparse vector's stored prefix grows
/// to meet a write that ends exactly at its tail; a write past
/// a gap is refused (the gap's elements are not materialised).
pub(super) fn replace_slice_bytes(
    value: &Value,
    offset: usize,
    bytes: &[u8],
) -> Option<Value> {
    fn splice(elems: &[Value], offset: usize, bytes: &[u8]) -> Option<Vec<Value>> {
        if offset + bytes.len() > elems.len() {
            return None;
        }
        let mut out = elems.to_vec();
        for (i, b) in bytes.iter().enumerate() {
            out[offset + i] = Value::Uint(*b as u128);
        }
        Some(out)
    }
    match value {
        Value::Tuple(items) => Some(Value::Tuple(splice(items, offset, bytes)?)),
        Value::Enum { variant, fields } => Some(Value::Enum {
            variant: *variant,
            fields: splice(fields, offset, bytes)?,
        }),
        Value::SparseVec { elems, len, cap } => {
            if offset + bytes.len() > *len as usize {
                return None;
            }
            let mut out = elems.clone();
            for (i, b) in bytes.iter().enumerate() {
                let idx = offset + i;
                if idx < out.len() {
                    out[idx] = Value::Uint(*b as u128);
                } else if idx == out.len() {
                    out.push(Value::Uint(*b as u128));
                } else {
                    return None;
                }
            }
            let cap = (*cap).max(*len).max(out.len() as u128);
            Some(Value::SparseVec {
                elems: out,
                len: *len,
                cap,
            })
        }
        Value::Bytes(data) => {
            if offset + bytes.len() > data.len() {
                return None;
            }
            let mut out = data.clone();
            out[offset..offset + bytes.len()].copy_from_slice(bytes);
            Some(Value::Bytes(out))
        }
        _ => None,
    }
}

/// A `std::sys::fs::Metadata` value in std's shape:
/// `{ stat, statx_extra }` — the stat with its fields in
/// libc's order (the mode is the fourth field, which is what
/// `permissions()` reads), and the statx extra-fields option
/// std's statx-based constructor fills: `Some` with the
/// file's birth time and the full statx mask when the host
/// reports a birth time, `None` otherwise (the same fallback
/// std takes when the birth time is unavailable).
pub(super) fn metadata_value(md: &std::fs::Metadata) -> Value {
    use std::os::unix::fs::MetadataExt;
    let stat = Value::Enum {
        variant: 0,
        fields: vec![
            Value::Uint(md.dev() as u128),
            Value::Uint(md.ino() as u128),
            Value::Uint(md.nlink() as u128),
            Value::Uint(md.mode() as u128),
            Value::Uint(md.uid() as u128),
            Value::Uint(md.gid() as u128),
            Value::Uint(0),
            Value::Uint(md.rdev() as u128),
            Value::Int(md.size() as i128),
            Value::Int(md.blksize() as i128),
            Value::Int(md.blocks() as i128),
            Value::Int(md.atime() as i128),
            Value::Int(md.atime_nsec() as i128),
            Value::Int(md.mtime() as i128),
            Value::Int(md.mtime_nsec() as i128),
            Value::Int(md.ctime() as i128),
            Value::Int(md.ctime_nsec() as i128),
            Value::Tuple(vec![Value::Int(0), Value::Int(0), Value::Int(0)]),
        ],
    };
    let statx_extra = match md.created() {
        Ok(created) => {
            let since_epoch = created
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default();
            let timestamp = Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Int(since_epoch.as_secs() as i128),
                    Value::Int(since_epoch.subsec_nanos() as i128),
                ],
            };
            Value::Enum {
                variant: 1,
                fields: vec![Value::Enum {
                    variant: 0,
                    fields: vec![timestamp, Value::Uint(0x1fff)],
                }],
            }
        }
        Err(_) => Value::Enum {
            variant: 0,
            fields: vec![],
        },
    };
    Value::Enum {
        variant: 0,
        fields: vec![stat, statx_extra],
    }
}

/// The mode of a `std::fs::Permissions` value: `{ inner:
/// { mode } }` in std's shape.
fn permissions_mode(value: &Value) -> Option<u32> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    let Value::Enum { fields, .. } = fields.first()? else {
        return None;
    };
    match fields.first()? {
        Value::Uint(mode) => Some(*mode as u32),
        Value::Int(mode) => Some(*mode as u32),
        _ => None,
    }
}
