//! The libc process, descriptor and dynamic-loading
//! surface the other families do not cover: identity and
//! configuration calls (`sysconf`, `getuid`, `getgid`,
//! `getpid`, `prctl`), the environment clear (`clearenv`),
//! dynamic loading against the program's own exports
//! (`dlopen` / `dlsym`), the descriptor calls that run on
//! the interpreter's descriptor routing (`dup`, `dup2`,
//! `close`, `pipe`, `fcntl`; see `desc_route` in
//! `interp::state`), and the file-tree walk reached
//! only under versioned symbols (`nftw`). A `strlen`
//! the guest declares under a renamed symbol routes to
//! the existing string-length model through the same
//! link-symbol fallback in conversion.
//!
//! The identity and configuration calls execute the host
//! functions themselves: the guest runs in-process, so
//! the host process's credentials and configuration are
//! the values the guest observes — the same ground rule
//! as the environment model, which reads and mutates the
//! host environment directly. `getpid` reports the driver
//! process's own id, which is also the parent the
//! process model spawns the guest's children from, so a
//! guest reading its children's parent id observes this
//! value.
//!
//! The `nftw` walk fabricates a Unix `stat` block from
//! the host's Unix metadata, so it is modelled on Unix
//! hosts only; on another host the call is declined as
//! unsupported rather than answered with invented
//! fields.
//!
//! The `nftw` walk's flag validation differs by binding
//! version, the way glibc's does: the 2.2.5 binding
//! predates validation and accepts any flag word, the
//! 2.3.3 binding rejects unknown bits with `EINVAL`
//! before walking. The walk itself visits the host tree
//! the guest's path names (guest paths are host paths
//! throughout the file model) and invokes the guest
//! callback per entry with a fabricated `stat` block
//! carrying the entry's real metadata. `FTW_MOUNT` and
//! `FTW_CHDIR` are accepted but not honoured: the walk
//! crosses devices and always reports root-prefixed
//! paths.

#[cfg(unix)]
use std::os::unix::ffi::OsStrExt as _;
#[cfg(unix)]
use std::os::unix::ffi::OsStringExt as _;

use crate::error::{InterpError, InterpResult};
use crate::interp::Frame;
use crate::Interpreter;
use crate::value::{RefTarget, Value};

use super::*;

const EBADF: i32 = 9;
#[cfg(unix)]
const EINVAL: i32 = 22;
const EIO: i32 = 5;

/// The `fcntl` commands the model answers.
const F_GETFD: i64 = 1;

/// The `nftw` callback's entry-kind values (`tflag`).
#[cfg(unix)]
const FTW_F: i128 = 0;
#[cfg(unix)]
const FTW_D: i128 = 1;
#[cfg(unix)]
const FTW_DNR: i128 = 2;
#[cfg(unix)]
const FTW_NS: i128 = 3;
#[cfg(unix)]
const FTW_SL: i128 = 4;
#[cfg(unix)]
const FTW_DP: i128 = 5;
#[cfg(unix)]
const FTW_SLN: i128 = 6;

/// The `nftw` flag bits the current glibc validates
/// against (`FTW_PHYS | FTW_MOUNT | FTW_CHDIR |
/// FTW_DEPTH | FTW_ACTIONRETVAL`).
#[cfg(unix)]
const NFTW_KNOWN_FLAGS: i128 = 1 | 2 | 4 | 8 | 16;
#[cfg(unix)]
const FTW_PHYS: i128 = 1;
#[cfg(unix)]
const FTW_DEPTH: i128 = 8;

/// A numeric descriptor argument.
fn descriptor_arg(name: &str, value: &Value) -> InterpResult<i64> {
    match value {
        Value::Int(v) => Ok(*v as i64),
        Value::Uint(v) => Ok(*v as i64),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "a descriptor number",
            other.kind_name(),
        )),
    }
}

fn is_null_pointer(value: &Value) -> bool {
    matches!(value, Value::Ref(RefTarget::Null))
        || matches!(value, Value::Int(0) | Value::Uint(0))
}

impl<'a> Interpreter<'a> {
    /// `long sysconf(int)`: the host call itself — the
    /// guest observes the host process's configuration.
    /// A `-1` answer leaves the host `errno` in the
    /// guest's cell, the way the native call leaves it
    /// for the caller to inspect.
    pub(super) fn exec_libc_sysconf(
        &mut self,
        frames: &mut Vec<Frame>,
        _name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let which = as_i128(&vals[0], "sysconf")? as std::ffi::c_int;
        let result = unsafe { host_sysconf(which) };
        if result == -1 {
            let code = std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(EIO);
            if code != 0 {
                self.set_errno(frames, code)?;
            }
        }
        Ok(Value::Int(result as i128))
    }

    /// `uid_t getuid(void)`: the host process's real
    /// user id, which the guest shares.
    pub(super) fn exec_libc_getuid(&mut self) -> InterpResult<Value> {
        Ok(Value::Uint(unsafe { host_getuid() } as u128))
    }

    /// `gid_t getgid(void)`: the host process's real
    /// group id.
    pub(super) fn exec_libc_getgid(&mut self) -> InterpResult<Value> {
        Ok(Value::Uint(unsafe { host_getgid() } as u128))
    }

    /// `pid_t getpid(void)`: the driver process's own
    /// id — the parent of every child the process model
    /// spawns for the guest.
    pub(super) fn exec_libc_getpid(&mut self) -> InterpResult<Value> {
        Ok(Value::Int(std::process::id() as i128))
    }

    /// `int clearenv(void)`: the environment model is
    /// the host environment itself (reads and writes go
    /// to it directly), so on hosts whose libc provides
    /// the call the host clear is the faithful execution.
    /// Hosts without it (macOS, Windows) name the call
    /// unsupported through the platform seam.
    pub(super) fn exec_libc_clearenv(&mut self, name: &str) -> InterpResult<Value> {
        match crate::platform::host_clearenv() {
            Some(result) => Ok(Value::Int(result as i128)),
            None => Err(unsupported(format!(
                "call to {name}: clearenv is not modelled on this host platform"
            ))),
        }
    }

    /// `int prctl(int, ...)`: the host call with the
    /// four variadic words zero-padded where the guest's
    /// declaration passes fewer. The guest shares the
    /// host process, so process-control settings the
    /// call makes are the ones in force for it.
    pub(super) fn exec_libc_prctl(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let option = as_i128(&vals[0], name)? as std::ffi::c_int;
        let mut words = [0 as std::ffi::c_ulong; 4];
        for (slot, value) in words.iter_mut().zip(vals.iter().skip(1)) {
            *slot = as_u128(value, name)? as std::ffi::c_ulong;
        }
        let result = crate::platform::host_prctl(option, words[0], words[1], words[2], words[3])
            .ok_or_else(|| {
                unsupported(format!(
                    "call to {name}: prctl is not modelled on this host platform"
                ))
            })?;
        if result == -1 {
            let code = std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(EIO);
            self.set_errno(frames, code)?;
        }
        Ok(Value::Int(result as i128))
    }

    /// `void *dlopen(const char *, int)`: only the main
    /// program handle (`NULL` filename) is modelled —
    /// the one handle whose symbol table this run knows
    /// (the program's own exports). Loading a named
    /// library has no equivalent here and is declined.
    pub(super) fn exec_libc_dlopen(
        &mut self,
        _frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        if is_null_pointer(&vals[0]) {
            // The handle is the address of an opaque
            // allocation: a pointer-shaped value the
            // guest can store and hand back, which the
            // `dlsym` model recognises by identity.
            return Ok(Value::Ref(RefTarget::Alloc {
                id: self.dlopen_main_handle_id(),
                projection: Vec::new(),
            }));
        }
        Err(unsupported(format!(
            "call to {name}: dlopen of a named library is not modelled; \
             only the main program handle (a NULL filename) is"
        )))
    }

    /// `void *dlsym(void *, const char *)`: a lookup in
    /// the program's own link exports for the main
    /// program handle (and for `RTLD_DEFAULT`, a null
    /// handle). A found symbol answers the converted
    /// body's function pointer; a missing one answers a
    /// null pointer, the native not-found answer.
    pub(super) fn exec_libc_dlsym(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let main = is_null_pointer(&vals[0])
            || matches!(&vals[0], Value::Ref(RefTarget::Alloc { id, .. }) if *id == self.dlopen_main_handle_id());
        if !main {
            return Err(unsupported(format!(
                "call to {name}: dlsym on a handle this run did not issue"
            )));
        }
        let symbol = self.c_str_arg(frames, &vals[1])?;
        let symbol = symbol.to_string_lossy().into_owned();
        match self.program().link_exports.get(&symbol) {
            Some(body) => Ok(Value::FnPtr(body.clone())),
            None => Ok(Value::Ref(RefTarget::Null)),
        }
    }

    /// `int dup(int)`: a duplicate of what the descriptor
    /// denotes under the descriptor routing, at the
    /// lowest free number.
    pub(super) fn exec_libc_dup(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let fd = descriptor_arg(name, &vals[0])?;
        if fd < 0 {
            self.set_errno(frames, EBADF)?;
            return Ok(Value::Int(-1));
        }
        match self.desc_dup(fd as u64) {
            Some(new) => Ok(Value::Int(new as i128)),
            None => {
                self.set_errno(frames, EBADF)?;
                Ok(Value::Int(-1))
            }
        }
    }

    /// `int dup2(int, int)`: the descriptor `newfd` made
    /// to denote what `oldfd` denotes, closing `newfd`
    /// first when it was open.
    pub(super) fn exec_libc_dup2(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let oldfd = descriptor_arg(name, &vals[0])?;
        let newfd = descriptor_arg(name, &vals[1])?;
        if oldfd < 0 || newfd < 0 {
            self.set_errno(frames, EBADF)?;
            return Ok(Value::Int(-1));
        }
        match self.desc_dup2(oldfd as u64, newfd as u64) {
            Some(fd) => Ok(Value::Int(fd as i128)),
            None => {
                self.set_errno(frames, EBADF)?;
                Ok(Value::Int(-1))
            }
        }
    }

    /// `int close(int)`: the descriptor routing's close.
    /// Closing a number the run does not hold open
    /// answers `-1` with `EBADF`, including a second
    /// close of the same number.
    pub(super) fn exec_libc_close(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let fd = descriptor_arg(name, &vals[0])?;
        if fd < 0 || !self.desc_close(fd as u64) {
            self.set_errno(frames, EBADF)?;
            return Ok(Value::Int(-1));
        }
        Ok(Value::Int(0))
    }

    /// `int pipe(int[2])`: a host pipe whose ends enter
    /// the descriptor tables, the read end first — the
    /// same host object the process model's spawn pipes
    /// are.
    pub(super) fn exec_libc_pipe(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let (reader, writer) = match std::io::pipe() {
            Ok(pair) => pair,
            Err(error) => {
                let code = error.raw_os_error().unwrap_or(EIO);
                self.set_errno(frames, code)?;
                return Ok(Value::Int(-1));
            }
        };
        // Each pipe end enters the file model as a host
        // `File` over the same pipe (the platform seam
        // performs the host's transfer); a host that
        // cannot make the transfer declines the call
        // rather than inventing an answer.
        let (Some(read_file), Some(write_file)) = (
            crate::platform::file_from_pipe_reader(reader),
            crate::platform::file_from_pipe_writer(writer),
        ) else {
            return Err(unsupported(format!(
                "call to {name}: pipe is not modelled on this host platform"
            )));
        };
        let read_id = self.register_file(read_file);
        let write_id = self.register_file(write_file);
        let target = self.deref_value(frames, &vals[0])?;
        self.write_seq_at(frames, &target, 0, Value::Int(read_id as i128))?;
        self.write_seq_at(frames, &target, 1, Value::Int(write_id as i128))?;
        Ok(Value::Int(0))
    }

    /// `int fcntl(int, int, ...)`: `F_GETFD` is the
    /// command the model answers — descriptor flags are
    /// not tracked, so an open descriptor reports none
    /// set and a closed one fails with `EBADF`. Other
    /// commands are declined rather than answered with
    /// an invented value.
    pub(super) fn exec_libc_fcntl(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let fd = descriptor_arg(name, &vals[0])?;
        let cmd = as_i128(&vals[1], name)? as i64;
        if cmd != F_GETFD {
            return Err(unsupported(format!(
                "call to {name}: fcntl command {cmd} is not modelled"
            )));
        }
        if fd < 0
            || self.desc_route(fd as u64) == crate::interp::DescRoute::Closed
        {
            self.set_errno(frames, EBADF)?;
            return Ok(Value::Int(-1));
        }
        Ok(Value::Int(0))
    }

    /// `int nftw(const char *, int (*)(...), int, int)`:
    /// the file-tree walk. `legacy` selects the 2.2.5
    /// binding, which performs no flag validation; the
    /// current binding rejects unknown flag bits with
    /// `EINVAL` before walking, as current glibc does.
    #[cfg(unix)]
    pub(super) fn exec_libc_nftw(
        &mut self,
        frames: &mut Vec<Frame>,
        name: &str,
        vals: &[Value],
        legacy: bool,
    ) -> InterpResult<Value> {
        let flags = as_i128(&vals[3], name)?;
        if !legacy && flags & !NFTW_KNOWN_FLAGS != 0 {
            self.set_errno(frames, EINVAL)?;
            return Ok(Value::Int(-1));
        }
        let target = match &vals[1] {
            Value::FnPtr(target) => target.clone(),
            Value::Closure { def, env } if env.is_empty() => def.clone(),
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a function pointer callback",
                    other.kind_name(),
                ))
            }
        };
        let dir = self.c_str_arg(frames, &vals[0])?;
        let dir = std::path::PathBuf::from(
            std::ffi::OsString::from_vec(dir.into_bytes()),
        );
        if let Err(error) = std::fs::symlink_metadata(&dir) {
            let code = error.raw_os_error().unwrap_or(EIO);
            self.set_errno(frames, code)?;
            return Ok(Value::Int(-1));
        }
        match self.nftw_entry(frames, &target, &dir, 0, flags, name)? {
            Some(verdict) => Ok(Value::Int(verdict as i128)),
            None => Ok(Value::Int(0)),
        }
    }

    /// `nftw` on a non-Unix host: the walk's fabricated
    /// `stat` block carries the host's Unix metadata,
    /// which has no source here, so the call is declined
    /// rather than answered with invented fields.
    #[cfg(not(unix))]
    pub(super) fn exec_libc_nftw(
        &mut self,
        _frames: &mut Vec<Frame>,
        name: &str,
        _vals: &[Value],
        _legacy: bool,
    ) -> InterpResult<Value> {
        Err(unsupported(format!(
            "call to {name}: nftw is not modelled on this host platform"
        )))
    }

    /// One entry of the walk: reports the entry to the
    /// callback (descending first or last for a directory
    /// per `FTW_DEPTH`) and returns the callback's stop
    /// verdict, if any.
    #[cfg(unix)]
    fn nftw_entry(
        &mut self,
        frames: &mut Vec<Frame>,
        target: &str,
        path: &std::path::Path,
        level: i32,
        flags: i128,
        name: &str,
    ) -> InterpResult<Option<i32>> {
        let meta = match std::fs::symlink_metadata(path) {
            Ok(meta) => meta,
            Err(_) => {
                return self
                    .nftw_callback(frames, target, path, FTW_NS, level, name)
                    .map(|v| if v != 0 { Some(v) } else { None });
            }
        };
        let file_type = meta.file_type();
        if file_type.is_symlink() && flags & FTW_PHYS == 0 {
            return match std::fs::metadata(path) {
                Ok(followed) if followed.is_dir() => {
                    self.nftw_dir(frames, target, path, level, flags, name)
                }
                Ok(_) => self.nftw_report(
                    frames,
                    target,
                    path,
                    FTW_F,
                    level,
                    name,
                ),
                Err(_) => self.nftw_report(
                    frames,
                    target,
                    path,
                    FTW_SLN,
                    level,
                    name,
                ),
            };
        }
        if file_type.is_symlink() {
            return self.nftw_report(frames, target, path, FTW_SL, level, name);
        }
        if file_type.is_dir() {
            return self.nftw_dir(frames, target, path, level, flags, name);
        }
        self.nftw_report(frames, target, path, FTW_F, level, name)
    }

    /// A directory of the walk: the directory's own
    /// report and its children's, in name order (the
    /// native order is the directory's own, which no
    /// caller may rely on).
    #[cfg(unix)]
    fn nftw_dir(
        &mut self,
        frames: &mut Vec<Frame>,
        target: &str,
        path: &std::path::Path,
        level: i32,
        flags: i128,
        name: &str,
    ) -> InterpResult<Option<i32>> {
        let entries = match std::fs::read_dir(path) {
            Ok(entries) => entries,
            Err(_) => {
                return self.nftw_report(
                    frames,
                    target,
                    path,
                    FTW_DNR,
                    level,
                    name,
                );
            }
        };
        let depth_first = flags & FTW_DEPTH != 0;
        if !depth_first {
            if let Some(v) =
                self.nftw_report(frames, target, path, FTW_D, level, name)?
            {
                return Ok(Some(v));
            }
        }
        let mut children: Vec<std::path::PathBuf> =
            entries.flatten().map(|entry| entry.path()).collect();
        children.sort();
        for child in &children {
            if let Some(v) = self.nftw_entry(
                frames,
                target,
                child,
                level + 1,
                flags,
                name,
            )? {
                return Ok(Some(v));
            }
        }
        if depth_first {
            return self.nftw_report(frames, target, path, FTW_DP, level, name);
        }
        Ok(None)
    }

    /// Reports one entry and folds the callback's answer
    /// into the walk's stop verdict.
    #[cfg(unix)]
    fn nftw_report(
        &mut self,
        frames: &mut Vec<Frame>,
        target: &str,
        path: &std::path::Path,
        tflag: i128,
        level: i32,
        name: &str,
    ) -> InterpResult<Option<i32>> {
        let verdict =
            self.nftw_callback(frames, target, path, tflag, level, name)?;
        Ok(if verdict != 0 { Some(verdict) } else { None })
    }

    /// Invokes the guest callback for one entry with a
    /// fabricated path string, `stat` block and `FTW`
    /// struct in guest allocations.
    #[cfg(unix)]
    fn nftw_callback(
        &mut self,
        frames: &mut Vec<Frame>,
        target: &str,
        path: &std::path::Path,
        tflag: i128,
        level: i32,
        name: &str,
    ) -> InterpResult<i32> {
        use std::os::unix::fs::MetadataExt as _;
        let mut path_bytes = path.as_os_str().as_bytes().to_vec();
        let base = path_bytes
            .iter()
            .rposition(|b| *b == b'/')
            .map(|i| i + 1)
            .unwrap_or(0);
        path_bytes.push(0);
        let path_id = self.alloc_insert(Value::Bytes(path_bytes));
        let path_ptr = Value::Ref(RefTarget::Alloc {
            id: path_id,
            projection: Vec::new(),
        });
        // The `stat` block as the value model carries a
        // C struct: the field values in the host's x86_64
        // field order, carrying the entry's own metadata,
        // so the callback's field reads observe them.
        let stat_value = match std::fs::symlink_metadata(path) {
            Ok(meta) => Value::Tuple(vec![
                Value::Uint(meta.dev() as u128),
                Value::Uint(meta.ino() as u128),
                Value::Uint(meta.nlink() as u128),
                Value::Uint(meta.mode() as u128),
                Value::Uint(meta.uid() as u128),
                Value::Uint(meta.gid() as u128),
                Value::Uint(0),
                Value::Uint(meta.rdev() as u128),
                Value::Int(meta.size() as i128),
                Value::Int(meta.blksize() as i128),
                Value::Int(meta.blocks() as i128),
                Value::Int(meta.atime() as i128),
                Value::Int(meta.atime_nsec() as i128),
                Value::Int(meta.mtime() as i128),
                Value::Int(meta.mtime_nsec() as i128),
                Value::Int(meta.ctime() as i128),
                Value::Int(meta.ctime_nsec() as i128),
                Value::Uint(0),
                Value::Uint(0),
                Value::Uint(0),
            ]),
            Err(_) => Value::Tuple(vec![Value::Uint(0); 20]),
        };
        let stat_id = self.alloc_insert(stat_value);
        let stat_ptr = Value::Ref(RefTarget::Alloc {
            id: stat_id,
            projection: Vec::new(),
        });
        let ftw_id = self.alloc_insert(Value::Tuple(vec![
            Value::Int(base as i128),
            Value::Int(level as i128),
        ]));
        let ftw_ptr = Value::Ref(RefTarget::Alloc {
            id: ftw_id,
            projection: Vec::new(),
        });
        let result = self.call_sync(
            frames,
            target,
            vec![path_ptr, stat_ptr, Value::Int(tflag), ftw_ptr],
        )?;
        match result {
            Value::Int(v) => Ok(v as i32),
            Value::Uint(v) => Ok(v as i32),
            other => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "an integer callback result",
                other.kind_name(),
            )),
        }
    }
}

/// The registry fragment for this family: the libc
/// crate's process, descriptor and dynamic-loading
/// surface, under both the crate-root and the `unix`
/// module spellings.
pub(crate) fn libc_sys_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "libc::sysconf" | "libc::unix::sysconf" => Builtin::LibcSysconf,
        "libc::getuid" | "libc::unix::getuid" => Builtin::LibcGetuid,
        "libc::getgid" | "libc::unix::getgid" => Builtin::LibcGetgid,
        "libc::getpid" | "libc::unix::getpid" => Builtin::LibcGetpid,
        "libc::clearenv" | "libc::unix::clearenv"
        | "libc::unix::linux_like::clearenv" => Builtin::LibcClearenv,
        "libc::prctl" | "libc::unix::prctl"
        | "libc::unix::linux_like::linux_l4re_shared::prctl" => {
            Builtin::LibcPrctl
        }
        "libc::dlopen" | "libc::unix::dlopen" => Builtin::LibcDlopen,
        "libc::dlsym" | "libc::unix::dlsym" => Builtin::LibcDlsym,
        "libc::dup" | "libc::unix::dup" => Builtin::LibcDup,
        "libc::dup2" | "libc::unix::dup2" => Builtin::LibcDup2,
        "libc::close" | "libc::unix::close" => Builtin::LibcClose,
        "libc::pipe" | "libc::unix::pipe" => Builtin::LibcPipe,
        "libc::fcntl" | "libc::unix::fcntl" => Builtin::LibcFcntl,
        _ => return None,
    })
}
