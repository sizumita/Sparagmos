//! Callee-name classification: mapping canonical names to builtin kinds.



use crate::interp::FmtBuilderKind;
use crate::value::{FmtTrait, Value};

use super::*;

/// The index just past the first top-level `::` of a path, if
/// it has one: the end of its leading segment. A `<`-qualified
/// path (an impl path) has no leading segment to remove, and a
/// `::` inside a generic argument group does not separate
/// segments.
fn leading_segment_end(name: &str) -> Option<usize> {
    if name.starts_with('<') {
        return None;
    }
    let bytes = name.as_bytes();
    let mut depth = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => depth = depth.saturating_sub(1),
            b':' if depth == 0 && bytes.get(i + 1) == Some(&b':') => {
                return Some(i + 2);
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Whether a builtin consumes the caller-location stamp the
/// driver appends to `#[track_caller]` calls: the location
/// intrinsic (the stamp is its result) and the panic entry
/// points (the location belongs to the panic). Other
/// builtin-served track-caller functions (e.g. `Option::unwrap`)
/// are not stamped, so their argument lists stay as declared.
pub fn consumes_caller_location(kind: Builtin) -> bool {
    matches!(
        kind,
        Builtin::CallerLocation
            | Builtin::PanicStr
            | Builtin::PanicFmt
            | Builtin::PanicAny
            | Builtin::OptionUnwrap
            | Builtin::ResultUnwrap
            | Builtin::IndexIndex
            | Builtin::IndexMut
    )
}

/// Whether a body-less call that names this builtin only by its
/// plain trait path may be served by the builtin at run time.
/// Such a call reaches the interpreter as a name with no body
/// under it (a function value like `.map(Clone::clone)`, or a
/// call inside a body the converter kept in generic form): the
/// driver collects no body for a builtin-owned definition, and
/// conversion had no impl instance to resolve the name to, so
/// the direct-call conversion to `CallBuiltin` never saw it.
/// Only the value-semantics models qualify — they compute from
/// the argument values alone and need nothing conversion bakes
/// into a `CallBuiltin` (no drop glue, no caller-location stamp,
/// no generic tags: `Clone::clone`'s tag-keyed handle cases do
/// not arise for the plain-data values these calls carry, and
/// enum comparisons order structurally, as they do for a
/// `CallBuiltin` whose discriminant table conversion could not
/// bake either). Anything else keeps the missing-body error.
pub(crate) fn runtime_name_fallback(kind: Builtin) -> bool {
    matches!(
        kind,
        Builtin::PartialEqEq
            | Builtin::PartialEqNe
            | Builtin::PartialCmp
            | Builtin::OrdCmp
            | Builtin::PartialOrdRel(_)
            | Builtin::CloneClone
            | Builtin::WakerWake
            | Builtin::WakerWakeByRef
            | Builtin::VecNew
            | Builtin::VecWithCapacity
            // Only ever invoked with the Self type recovered
            // from an impl-qualified name (`<i32 as
            // std::default::Default>::default`); the plain trait
            // path carries no type for the model to default, and
            // the interpreter's fallback rejects it there.
            | Builtin::DefaultDefault
            // The OS error function table's entries: they are
            // only ever invoked indirectly, through the table
            // `get_os_functions` hands out (their bodies are
            // std-internal and never collected), so the name
            // fallback is the only dispatch that reaches them.
            | Builtin::IoDecodeErrorKind
            | Builtin::IoFormatOsError
            | Builtin::IoIsInterrupted
    )
}

/// The `core::fmt` trait methods themselves, named by their
/// plain trait path (`std::fmt::Display::fmt`): an explicit
/// UFCS call in a guest body names the trait item, and for a
/// foreign `Self` type the impl instance usually has no MIR
/// body at all (std's impls are not `#[inline]`, so no body
/// reaches the metadata) — the call can only be served by the
/// structural fmt engine, which formats the receiver into the
/// captured `Formatter` argument exactly as it formats an
/// `Argument` over the same value.
///
/// These names are deliberately absent from [`lookup`], for
/// the same reason as the foreign helpers below: the registry
/// gates driver collection, and a collected impl body (a user
/// type's `fmt`, a foreign `#[inline]` instance) is the
/// faithful execution and must win. This lookup is consulted
/// only from conversion's external-callee path, where the
/// callee is already known to have no converted body and the
/// call's generic arguments (the `Self` type first) are at
/// hand for the model (conversion's path, and the
/// `CallBuiltin` dispatch in `exec_builtin`, which carries
/// the same tags). It is not part of the interpreter's
/// run-time name fallback: that path holds no type tags, and
/// the structural engine keys several shapes (a `char`, an
/// array, an ADT) on the tag.
pub fn fmt_trait_method(canonical: &str) -> Option<Builtin> {
    use FmtTrait as F;
    Some(match canonical {
        "std::fmt::Display::fmt" | "core::fmt::Display::fmt" => Builtin::FmtMethod(F::Display),
        "std::fmt::Debug::fmt" | "core::fmt::Debug::fmt" => Builtin::FmtMethod(F::Debug),
        "std::fmt::Octal::fmt" | "core::fmt::Octal::fmt" => Builtin::FmtMethod(F::Octal),
        "std::fmt::LowerHex::fmt" | "core::fmt::LowerHex::fmt" => {
            Builtin::FmtMethod(F::LowerHex)
        }
        "std::fmt::UpperHex::fmt" | "core::fmt::UpperHex::fmt" => {
            Builtin::FmtMethod(F::UpperHex)
        }
        "std::fmt::Binary::fmt" | "core::fmt::Binary::fmt" => Builtin::FmtMethod(F::Binary),
        "std::fmt::LowerExp::fmt" | "core::fmt::LowerExp::fmt" => {
            Builtin::FmtMethod(F::LowerExp)
        }
        "std::fmt::UpperExp::fmt" | "core::fmt::UpperExp::fmt" => {
            Builtin::FmtMethod(F::UpperExp)
        }
        "std::fmt::Pointer::fmt" | "core::fmt::Pointer::fmt" => Builtin::FmtMethod(F::Pointer),
        _ => return None,
    })
}

/// The `std::net::ToSocketAddrs` trait method, named by its
/// plain trait path: name resolution is std-internal (the
/// impls for the address and string types ship in the
/// distributed std without MIR bodies), so a call whose
/// `Self` instance has no converted body can only be served
/// by the net model, which resolves the receiver into the
/// modelled `vec::IntoIter<SocketAddr>` the guest iterates.
///
/// Deliberately absent from [`lookup`], for the same reason
/// as [`fmt_trait_method`]: the registry gates driver
/// collection, and a collected impl body (a guest type's own
/// `ToSocketAddrs` impl) is the faithful execution and must
/// win. This lookup is consulted only where the callee is
/// already known to have no converted body — conversion's
/// external-callee path (which carries the call's generic
/// arguments, the `Self` type first) and the `CallBuiltin`
/// dispatch in `exec_builtin`, which carries the same tags.
/// It is not part of the interpreter's run-time name
/// fallback: that path holds no type tags, and the model
/// keys nothing on a bare name alone.
pub fn net_trait_method(canonical: &str) -> Option<Builtin> {
    match canonical {
        "std::net::ToSocketAddrs::to_socket_addrs" => Some(Builtin::ToSocketAddrs),
        _ => None,
    }
}

/// Models for body-less foreign callees whose semantics are
/// fixed outside the guest program: the C helpers of
/// `rust_test_helpers` (the static library the rustc UI tests
/// link for ABI checks; its source ships in the tree at
/// `tests/auxiliary/rust_test_helpers.c`) and
/// `proc_macro::is_available`.
///
/// These names are deliberately absent from [`lookup`]: the
/// registry gates driver collection (a registered name's body
/// is not collected), and a guest function that merely shares
/// a helper's last path segment must keep its own body. This
/// lookup is consulted only where a callee is already known
/// to have no converted body — conversion's external-callee
/// path and the interpreter's run-time name fallback — so a
/// collected body always wins. The C symbols match on the
/// last path segment because the declaration site (the guest
/// crate's own module path) is part of the callee's name;
/// names under `std::` / `core::` / `alloc::` never match, so
/// a library function can never be mistaken for a helper.
/// Each model validates the argument values it is handed and
/// fails with an explicit error on a shape it does not know,
/// never a silently different result.
pub fn foreign_helper_lookup(canonical: &str) -> Option<Builtin> {
    if canonical == "proc_macro::is_available"
        || canonical.ends_with("::proc_macro::is_available")
    {
        return Some(Builtin::ProcMacroIsAvailable);
    }
    if canonical.starts_with("std::")
        || canonical.starts_with("core::")
        || canonical.starts_with("alloc::")
    {
        return None;
    }
    let last = canonical.rsplit("::").next().unwrap_or(canonical);
    Some(match last {
        "rust_dbg_extern_identity_u8"
        | "rust_dbg_extern_identity_u32"
        | "rust_dbg_extern_identity_u64"
        | "rust_dbg_extern_identity_double"
        | "rust_dbg_extern_identity_TwoU8s"
        | "rust_dbg_extern_identity_TwoU16s"
        | "rust_dbg_extern_identity_TwoU32s"
        | "rust_dbg_extern_identity_TwoU64s"
        | "rust_dbg_extern_identity_TwoDoubles"
        | "rust_dbg_extern_identity_FiveU16s"
        | "rust_int8_to_int32"
        | "identity" => Builtin::FfiIdentity,
        "rust_dbg_extern_return_TwoU8s"
        | "rust_dbg_extern_return_TwoU16s"
        | "rust_dbg_extern_return_TwoU32s"
        | "rust_dbg_extern_return_TwoU64s" => Builtin::FfiReturnTwo,
        "rust_dbg_extern_return_FiveU16s" => Builtin::FfiReturnFive,
        "rust_dbg_call" => Builtin::FfiDbgCall,
        "rust_dbg_extern_empty_struct" => Builtin::FfiEmptyStruct,
        "rust_get_test_int" => Builtin::FfiGetTestInt,
        "rust_get_null_ptr" => Builtin::FfiGetNullPtr,
        "atol" => Builtin::LibcAtol,
        "atoll" => Builtin::LibcAtoll,
        "atoi" => Builtin::LibcAtoi,
        "strtol" => Builtin::LibcStrtol,
        "strtoll" => Builtin::LibcStrtoll,
        "strtoul" => Builtin::LibcStrtoul,
        "strtoull" => Builtin::LibcStrtoull,
        "isatty" => Builtin::LibcIsatty,
        "free" => Builtin::LibcFree,
        "lgamma_r" => Builtin::LgammaR,
        // The declaration site may name the item `lgamma` and
        // bind it to the reentrant symbol with
        // `#[link_name = "lgamma_r"]` (the shape of
        // `tests/ui/extern/lgamma-linkage.rs`): the signature
        // carries the sign pointer, which is `lgamma_r`'s.
        "lgamma" => Builtin::LgammaR,
        "rust_dbg_abi_1" => Builtin::FfiDbgAbi1,
        "rust_dbg_abi_2" => Builtin::FfiDbgAbi2,
        "rust_dbg_abi_3" => Builtin::FfiDbgAbi3,
        "rust_dbg_abi_4" => Builtin::FfiDbgAbi4,
        "get_x" => Builtin::FfiArgField { arg: 0, field: 0 },
        "get_y" => Builtin::FfiArgField { arg: 0, field: 1 },
        "get_z" => Builtin::FfiArgField { arg: 0, field: 2 },
        "get_c_many_params" => Builtin::FfiArgField { arg: 4, field: 2 },
        "get_c_exhaust_sysv64_ints" => Builtin::FfiArgField { arg: 7, field: 2 },
        "rust_dbg_new_some_u64" | "rust_dbg_new_some_u64u64" => Builtin::FfiNewSome,
        "rust_dbg_new_none_u64" | "rust_dbg_new_none_u64u64" => Builtin::FfiNewNone,
        "rust_dbg_unpack_option_u64" | "rust_dbg_unpack_option_u64u64" => {
            Builtin::FfiUnpackOption
        }
        "rust_interesting_average" => Builtin::FfiInterestingAverage,
        "rust_valist_interesting_average" => Builtin::FfiValistInterestingAverage,
        "rust_va_list_next_i32" => Builtin::FfiVaListNextI32,
        "square" => Builtin::FfiSquare128,
        "sub" => Builtin::FfiSub128,
        "increment_all_parts" => Builtin::FfiIncrementAllParts,
        "issue_97463_leak_uninit_data" => Builtin::FfiLeakUninitData,
        _ => return None,
    })
}

/// Models for compiler intrinsics the guest crate declares
/// itself with `#[rustc_intrinsic]` (the shape of the
/// `no_core` / minicore test crates, which redeclare the
/// intrinsics they exercise instead of linking core's). The
/// declaration site's own module path is part of the callee's
/// name (`simd_bitmask::simd_bitmask`), so the registry —
/// keyed by the `core::intrinsics` spelling — never matches
/// it directly.
///
/// A compiler intrinsic's identity is its item name: rustc
/// accepts a `#[rustc_intrinsic]` declaration only under the
/// name of a real intrinsic, so the last path segment of the
/// callee's name names the intrinsic, and the registry lookup
/// of the `core::intrinsics` path built from that segment is
/// the check that the interpreter models it. Like
/// [`foreign_helper_lookup`], this lookup is deliberately
/// absent from [`lookup`]: it is consulted only where a
/// callee is already known to have no converted body (the
/// driver's demand collection, which asks only about
/// instances Stable MIR itself classifies as intrinsics,
/// conversion's external-callee path — gated on the same
/// classification, which the driver records in the program —
/// and the interpreter's name-based dispatch of the builtin
/// nodes conversion emitted), so a collected body always
/// wins over the model, and an ordinary body-less function
/// that merely shares an intrinsic's name keeps its recorded
/// conversion failure instead of being served by the model.
pub fn intrinsic_lookup(canonical: &str) -> Option<Builtin> {
    let last = canonical.rsplit("::").next().unwrap_or(canonical);
    lookup(&format!("core::intrinsics::{last}"))
        .or_else(|| lookup(&format!("core::intrinsics::simd::{last}")))
}

/// A successful `core::fmt::Result` value (`Ok(())`).
pub(super) fn fmt_result_ok() -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Unit],
    }
}

/// Look a canonical callee name up in the registry, spelled
/// exactly as given; `None` when the interpreter has no model
/// for it.
/// Whether a `Write` method's `Self` type (reference markers
/// stripped) is one of the non-file receivers the interpreter
/// serves through its captured streams or an in-place append:
/// the public `std::io` stream handles and their locks, the
/// `std::sys` stdio handles the panic runtime writes through
/// directly, or a `Vec<u8>`.
pub(crate) fn stream_write_receiver(self_ty: &str) -> bool {
    matches!(
        self_ty,
        "std::io::Stdout"
            | "std::io::StdoutLock"
            | "std::io::Stderr"
            | "std::io::StderrLock"
            | "std::sys::stdio::unix::Stdout"
            | "std::sys::stdio::unix::Stderr"
    ) || self_ty.starts_with("std::vec::Vec<u8")
}

/// Whether a standard-stream receiver writes to the captured
/// error stream (rather than the captured output stream).
pub(super) fn stream_is_stderr(self_ty: &str) -> bool {
    matches!(
        self_ty,
        "std::io::Stderr"
            | "std::io::StderrLock"
            | "std::sys::stdio::unix::Stderr"
    )
}

/// Whether this call is a file-I/O builtin node: a `Read` /
/// `Write` / `Seek` method whose `Self` is `std::fs::File`
/// (or a child-process pipe handle, whose value is a
/// file-table id like `File`'s), or one of `File`'s inherent
/// handle methods. Conversion
/// emits builtin nodes for exactly these calls (see
/// `file_io_terminator` in mir_convert); the inherent names
/// are File-unique, and the trait names are gated on the
/// `Self` type so other receivers keep their collected
/// bodies. The `Write` methods additionally serve the
/// standard-stream and `Vec<u8>` receivers (see
/// `stream_write_receiver`), whose writes the interpreter
/// models through its captured streams / an in-place append.
pub(super) fn is_file_io_call(name: &str, generic_args: &[String]) -> bool {
    match name {
        "std::fs::File::metadata"
        | "std::fs::File::try_clone"
        | "std::fs::File::sync_all"
        | "std::fs::File::sync_data"
        | "std::fs::File::set_len" => true,
        "std::io::Read::read"
        | "std::io::Read::read_exact"
        | "std::io::Read::read_to_end"
        | "std::io::Read::read_to_string"
        | "std::io::Read::read_buf"
        | "std::io::Read::read_buf_exact"
        | "std::io::Seek::seek"
        | "std::io::Seek::rewind"
        | "std::io::Seek::stream_position" => {
            let self_ty = generic_args
                .first()
                .map(|tag| {
                    tag.trim_start_matches("&mut ")
                        .trim_start_matches('&')
                        .trim()
                })
                .unwrap_or("");
            // A child's piped output handles are files in the
            // interpreter's table (see `ProcessSpawned`), so
            // their reads go through the same arms as `File`'s;
            // the standard input handle is a file-table entry
            // too (see `Builtin::Stdin`).
            self_ty == "std::fs::File"
                || matches!(
                    self_ty,
                    "std::process::ChildStdout"
                        | "std::process::ChildStderr"
                        | "std::io::Stdin"
                        | "std::io::StdinLock"
                )
        }
        "std::io::Stdin::read_line" => true,
        "std::io::BufRead::read_line"
        | "std::io::BufRead::read_until" => {
            let self_ty = generic_args
                .first()
                .map(|tag| {
                    tag.trim_start_matches("&mut ")
                        .trim_start_matches('&')
                        .trim()
                })
                .unwrap_or("");
            matches!(
                self_ty,
                "std::io::Stdin" | "std::io::StdinLock"
            )
        }
        "std::io::Write::write"
        | "std::io::Write::write_all"
        | "std::io::Write::flush" => {
            let self_ty = generic_args
                .first()
                .map(|tag| {
                    tag.trim_start_matches("&mut ")
                        .trim_start_matches('&')
                        .trim()
                })
                .unwrap_or("");
            self_ty == "std::fs::File"
                || stream_write_receiver(self_ty)
                || self_ty == "std::process::ChildStdin"
        }
        _ => false,
    }
}

/// Whether this call name is a `std::io::Read` / `Write`
/// leaf method on a `std::net::TcpStream` receiver: the
/// stream I/O diversion (see `file_io_terminator` in
/// `mir_convert` and `exec_stream_io` in `builtins::net`).
/// Like the file I/O names, these are deliberately absent
/// from `lookup`: conversion emits a builtin node only for
/// the `TcpStream` Self type, and every other receiver's
/// bodies are collected as usual.
pub(super) fn is_stream_io_call(
    name: &str,
    generic_args: &[String],
) -> bool {
    if !matches!(
        name,
        "std::io::Read::read"
            | "std::io::Read::read_vectored"
            | "std::io::Write::write"
            | "std::io::Write::write_vectored"
            | "std::io::Write::flush"
    ) {
        return false;
    }
    let self_ty = generic_args
        .first()
        .map(|tag| {
            tag.trim_start_matches("&mut ")
                .trim_start_matches('&')
                .trim()
        })
        .unwrap_or("");
    self_ty == "std::net::TcpStream"
}

pub fn lookup(canonical: &str) -> Option<Builtin> {
    use FmtTrait as F;
    Some(match canonical {
        "std::io::_print" => Builtin::Print,
        "std::io::_eprint" => Builtin::Eprint,
        "std::env::args" => Builtin::EnvArgs,
        "std::env::args_os" => Builtin::EnvArgsOs,
        "std::env::current_exe" => Builtin::EnvCurrentExe,
        "std::env::var::inner" => Builtin::EnvVarInner,
        "std::env::split_paths" => Builtin::EnvSplitPaths,
        "std::env::join_paths" => Builtin::EnvJoinPaths,
        "std::sys::env::unix::getenv" => Builtin::EnvGetenv,
        "std::sys::env::unix::setenv" => Builtin::EnvSetenv,
        "std::sys::env::unix::unsetenv" => Builtin::EnvUnsetenv,
        "std::env::current_dir" => Builtin::EnvCurrentDir,
        "std::sys::paths::unix::getcwd" => Builtin::EnvCurrentDir,
        "std::env::temp_dir" => Builtin::EnvTempDir,
        "std::sys::paths::unix::temp_dir" => Builtin::EnvTempDir,
        "std::env::home_dir" => Builtin::EnvHomeDir,
        "std::sys::paths::unix::home_dir" => Builtin::EnvHomeDir,
        "std::sys::paths::unix::chdir" => Builtin::SysChdir,
        "std::fs::read::inner" => Builtin::FsReadInner,
        "std::fs::read_to_string::inner" => Builtin::FsReadToStringInner,
        "std::string::String::from_utf8" => Builtin::StringFromUtf8,
        "std::fs::write::inner" => Builtin::FsWriteInner,
        "std::path::Path::is_absolute" => Builtin::PathIsAbsolute,
        "std::path::Path::is_file" => Builtin::PathIsFile,
        "std::path::Path::is_dir" => Builtin::PathIsDir,
        "std::path::Path::to_path_buf" => Builtin::PathToPathBuf,
        "std::path::Path::_join" => Builtin::PathJoin,
        "std::path::Path::parent" => Builtin::PathParent,
        "std::path::Path::file_name" => Builtin::PathFileName,
        "std::path::Path::file_stem" => Builtin::PathFileStem,
        "std::path::Path::extension" => Builtin::PathExtension,
        "std::path::Path::_starts_with" => Builtin::PathStartsWith,
        "std::path::Path::_ends_with" => Builtin::PathEndsWith,
        "std::fs::OpenOptions::_open" => Builtin::FsOpen,
        "std::fs::DirBuilder::_create" => Builtin::FsDirCreate,
        "std::string::String::from_utf8_lossy" => Builtin::StringFromUtf8Lossy,
        "std::str::from_utf8" => Builtin::StrFromUtf8,
        "core::str::from_utf8" => Builtin::StrFromUtf8,
        "core::ffi::c_str::strlen::runtime::strlen" => Builtin::LibcStrlen,
        "std::ffi::CStr::from_bytes_with_nul" | "core::ffi::CStr::from_bytes_with_nul" => {
            Builtin::CStrFromBytesWithNul
        }
        "std::ffi::CStr::from_bytes_until_nul" | "core::ffi::CStr::from_bytes_until_nul" => {
            Builtin::CStrFromBytesUntilNul
        }
        "std::ffi::CStr::to_str" | "core::ffi::CStr::to_str" => Builtin::CStrToStr,
        "alloc::ffi::c_str::<impl std::ffi::CStr>::to_string_lossy" => {
            Builtin::CStrToStringLossy
        }
        "std::ffi::CString::new::SpecNewImpl::spec_new_impl" => {
            Builtin::CStringSpecNewImpl
        }
        "std::ffi::CString::_from_vec_unchecked"
        | "std::ffi::CString::from_vec_unchecked" => {
            Builtin::CStringFromVecUnchecked
        }
        "std::ffi::CString::from_raw" => Builtin::CStringFromRaw,
        "std::ffi::CString::into_string" => Builtin::CStringIntoString,
        "std::ffi::OsString::to_string_lossy" => Builtin::ToStringLossy,
        "std::ffi::OsStr::to_string_lossy" => Builtin::ToStringLossy,
        "std::path::Path::to_string_lossy" => Builtin::ToStringLossy,
        "std::fs::DirEntry::path" => Builtin::DirEntryPath,
        "std::fs::DirEntry::file_name" => Builtin::DirEntryFileName,
        "std::fs::DirEntry::metadata" => Builtin::DirEntryMetadata,
        "std::sys::fs::metadata" => Builtin::FsSys(FsSysOp::Metadata),
        "std::sys::fs::symlink_metadata" => Builtin::FsSys(FsSysOp::SymlinkMetadata),
        "std::sys::fs::canonicalize" => Builtin::FsSys(FsSysOp::Canonicalize),
        "std::sys::fs::copy" => Builtin::FsSys(FsSysOp::Copy),
        "std::sys::fs::hard_link" => Builtin::FsSys(FsSysOp::HardLink),
        "std::sys::fs::read_dir" => Builtin::FsSys(FsSysOp::ReadDir),
        "std::sys::fs::read_link" => Builtin::FsSys(FsSysOp::ReadLink),
        "std::sys::fs::remove_dir" => Builtin::FsSys(FsSysOp::RemoveDir),
        "std::sys::fs::remove_dir_all" => Builtin::FsSys(FsSysOp::RemoveDirAll),
        "std::sys::fs::remove_file" => Builtin::FsSys(FsSysOp::RemoveFile),
        "std::sys::fs::rename" => Builtin::FsSys(FsSysOp::Rename),
        "std::sys::fs::set_permissions" => Builtin::FsSys(FsSysOp::SetPermissions),
        "std::sys::fs::symlink" => Builtin::FsSys(FsSysOp::Symlink),
        "std::net::TcpListener::bind" => {
            Builtin::NetSys(NetSysOp::ListenerBind)
        }
        "std::net::TcpListener::accept" => {
            Builtin::NetSys(NetSysOp::ListenerAccept)
        }
        "std::net::TcpListener::local_addr" => {
            Builtin::NetSys(NetSysOp::ListenerSocketAddr)
        }
        "std::net::TcpListener::try_clone" => {
            Builtin::NetSys(NetSysOp::ListenerDuplicate)
        }
        "std::net::TcpListener::set_ttl" => {
            Builtin::NetSys(NetSysOp::ListenerSetTtl)
        }
        "std::net::TcpListener::ttl" => {
            Builtin::NetSys(NetSysOp::ListenerTtl)
        }
        "std::net::TcpListener::set_only_v6" => {
            Builtin::NetSys(NetSysOp::ListenerSetOnlyV6)
        }
        "std::net::TcpListener::only_v6" => {
            Builtin::NetSys(NetSysOp::ListenerOnlyV6)
        }
        "std::net::TcpListener::take_error" => {
            Builtin::NetSys(NetSysOp::ListenerTakeError)
        }
        "std::net::TcpListener::set_nonblocking" => {
            Builtin::NetSys(NetSysOp::ListenerSetNonblocking)
        }
        "std::net::TcpStream::connect" => {
            Builtin::NetSys(NetSysOp::StreamConnect)
        }
        "std::net::TcpStream::connect_timeout" => {
            Builtin::NetSys(NetSysOp::StreamConnectTimeout)
        }
        "std::net::TcpStream::local_addr" => {
            Builtin::NetSys(NetSysOp::StreamSocketAddr)
        }
        "std::net::TcpStream::peer_addr" => {
            Builtin::NetSys(NetSysOp::StreamPeerAddr)
        }
        "std::net::TcpStream::peek" => {
            Builtin::NetSys(NetSysOp::StreamPeek)
        }
        "std::net::TcpStream::try_clone" => {
            Builtin::NetSys(NetSysOp::StreamDuplicate)
        }
        "std::net::TcpStream::shutdown" => {
            Builtin::NetSys(NetSysOp::StreamShutdown)
        }
        "std::net::TcpStream::set_nodelay" => {
            Builtin::NetSys(NetSysOp::StreamSetNodelay)
        }
        "std::net::TcpStream::nodelay" => {
            Builtin::NetSys(NetSysOp::StreamNodelay)
        }
        "std::net::TcpStream::set_ttl" => {
            Builtin::NetSys(NetSysOp::StreamSetTtl)
        }
        "std::net::TcpStream::ttl" => {
            Builtin::NetSys(NetSysOp::StreamTtl)
        }
        "std::net::TcpStream::take_error" => {
            Builtin::NetSys(NetSysOp::StreamTakeError)
        }
        "std::net::TcpStream::set_nonblocking" => {
            Builtin::NetSys(NetSysOp::StreamSetNonblocking)
        }
        "std::net::TcpStream::set_read_timeout" => {
            Builtin::NetSys(NetSysOp::StreamSetReadTimeout)
        }
        "std::net::TcpStream::read_timeout" => {
            Builtin::NetSys(NetSysOp::StreamReadTimeout)
        }
        "std::net::TcpStream::set_write_timeout" => {
            Builtin::NetSys(NetSysOp::StreamSetWriteTimeout)
        }
        "std::net::TcpStream::write_timeout" => {
            Builtin::NetSys(NetSysOp::StreamWriteTimeout)
        }
        "std::sys::net::connection::socket::TcpStream::read" => {
            Builtin::NetSys(NetSysOp::StreamRead)
        }
        "std::sys::net::connection::socket::TcpStream::read_vectored" => {
            Builtin::NetSys(NetSysOp::StreamReadVectored)
        }
        "std::sys::net::connection::socket::TcpStream::write" => {
            Builtin::NetSys(NetSysOp::StreamWrite)
        }
        "std::sys::net::connection::socket::TcpStream::write_vectored" => {
            Builtin::NetSys(NetSysOp::StreamWriteVectored)
        }
        "std::net::UdpSocket::bind" => {
            Builtin::NetSys(NetSysOp::UdpBind)
        }
        "std::net::UdpSocket::local_addr" => {
            Builtin::NetSys(NetSysOp::UdpSocketAddr)
        }
        "std::net::UdpSocket::try_clone" => {
            Builtin::NetSys(NetSysOp::UdpDuplicate)
        }
        "std::net::UdpSocket::recv_from" => {
            Builtin::NetSys(NetSysOp::UdpRecvFrom)
        }
        "std::net::UdpSocket::send_to" => {
            Builtin::NetSys(NetSysOp::UdpSendTo)
        }
        "std::net::UdpSocket::connect" => {
            Builtin::NetSys(NetSysOp::UdpConnect)
        }
        "std::net::UdpSocket::recv" => Builtin::NetSys(NetSysOp::UdpRecv),
        "std::net::UdpSocket::send" => Builtin::NetSys(NetSysOp::UdpSend),
        "std::net::UdpSocket::set_broadcast" => {
            Builtin::NetSys(NetSysOp::UdpSetBroadcast)
        }
        "std::net::UdpSocket::broadcast" => {
            Builtin::NetSys(NetSysOp::UdpBroadcast)
        }
        "std::net::UdpSocket::set_ttl" => {
            Builtin::NetSys(NetSysOp::UdpSetTtl)
        }
        "std::net::UdpSocket::ttl" => Builtin::NetSys(NetSysOp::UdpTtl),
        "std::net::UdpSocket::take_error" => {
            Builtin::NetSys(NetSysOp::UdpTakeError)
        }
        "std::net::UdpSocket::set_nonblocking" => {
            Builtin::NetSys(NetSysOp::UdpSetNonblocking)
        }
        "std::process::exit" => Builtin::ProcessExit,
        "std::process::abort" => Builtin::IntrinsicsAbort,
        "std::intrinsics::abort" | "core::intrinsics::abort" => {
            Builtin::IntrinsicsAbort
        }
        "std::io::stdout" => Builtin::Stdout,
        "std::io::stdin" => Builtin::Stdin,
        "std::io::Stdout::lock" | "std::io::Stderr::lock"
        | "std::io::Stdin::lock" => Builtin::StdStreamLock,
        "std::io::Write::write_fmt" => Builtin::WriteFmt,
        "std::fmt::write" | "core::fmt::write" => Builtin::FmtWrite,
        "std::fmt::format" | "std::fmt::format::format_inner"
        | "alloc::fmt::format::format_inner" => Builtin::Format,
        "std::fmt::Arguments::from_str" | "std::fmt::Arguments::from_str_nonconst" => {
            Builtin::ArgumentsFromStr
        }
        "std::fmt::Arguments::new" => Builtin::ArgumentsNew,
        "core::fmt::rt::Argument::new_display" => Builtin::ArgumentNew(F::Display),
        "core::fmt::rt::Argument::new_debug" => Builtin::ArgumentNew(F::Debug),
        "core::fmt::rt::Argument::new_debug_noop" => Builtin::ArgumentNew(F::DebugNoop),
        "core::fmt::rt::Argument::new_octal" => Builtin::ArgumentNew(F::Octal),
        "core::fmt::rt::Argument::new_lower_hex" => Builtin::ArgumentNew(F::LowerHex),
        "core::fmt::rt::Argument::new_upper_hex" => Builtin::ArgumentNew(F::UpperHex),
        "core::fmt::rt::Argument::new_pointer" => Builtin::ArgumentNew(F::Pointer),
        "core::fmt::rt::Argument::new_binary" => Builtin::ArgumentNew(F::Binary),
        "core::fmt::rt::Argument::new_lower_exp" => Builtin::ArgumentNew(F::LowerExp),
        "core::fmt::rt::Argument::new_upper_exp" => Builtin::ArgumentNew(F::UpperExp),
        "core::fmt::rt::Argument::from_usize" => Builtin::ArgumentFromUsize,
        // The captured-`Formatter` surface (see the fmt engine):
        // guest `fmt` implementation bodies write through these
        // methods, and derive's fast-path finishers render
        // through them. The finishers and the builder
        // constructors/methods have no MIR bodies; `write_fmt`
        // has one, displaced by the builtin so a guest body's
        // `write!` lands in the same capture as its `write_str`
        // calls.
        "std::fmt::Formatter::write_str" | "core::fmt::Formatter::write_str" => {
            Builtin::FormatterWriteStr
        }
        "std::fmt::Formatter::write_char" | "core::fmt::Formatter::write_char" => {
            Builtin::FormatterWriteChar
        }
        "std::fmt::Formatter::pad" | "core::fmt::Formatter::pad" => Builtin::FormatterPad,
        "std::fmt::Formatter::write_fmt" | "core::fmt::Formatter::write_fmt" => {
            Builtin::FormatterWriteFmt
        }
        "std::fmt::Formatter::alternate" | "core::fmt::Formatter::alternate" => {
            Builtin::FormatterAlternate
        }
        "std::fmt::Formatter::width" | "core::fmt::Formatter::width" => {
            Builtin::FormatterWidth
        }
        "std::fmt::Formatter::precision" | "core::fmt::Formatter::precision" => {
            Builtin::FormatterPrecision
        }
        "std::fmt::Formatter::fill" | "core::fmt::Formatter::fill" => Builtin::FormatterFill,
        "std::fmt::Formatter::sign_plus" | "core::fmt::Formatter::sign_plus" => {
            Builtin::FormatterSignPlus
        }
        "std::fmt::Formatter::sign_aware_zero_pad"
        | "core::fmt::Formatter::sign_aware_zero_pad" => Builtin::FormatterSignAwareZeroPad,
        "std::fmt::Formatter::debug_struct_field1_finish"
        | "core::fmt::Formatter::debug_struct_field1_finish" => {
            Builtin::DebugStructFinish { fields: 1 }
        }
        "std::fmt::Formatter::debug_tuple_field1_finish"
        | "core::fmt::Formatter::debug_tuple_field1_finish" => {
            Builtin::DebugTupleFinish { fields: 1 }
        }
        "std::fmt::Formatter::debug_struct_field2_finish"
        | "core::fmt::Formatter::debug_struct_field2_finish" => {
            Builtin::DebugStructFinish { fields: 2 }
        }
        "std::fmt::Formatter::debug_tuple_field2_finish"
        | "core::fmt::Formatter::debug_tuple_field2_finish" => {
            Builtin::DebugTupleFinish { fields: 2 }
        }
        "std::fmt::Formatter::debug_struct_field3_finish"
        | "core::fmt::Formatter::debug_struct_field3_finish" => {
            Builtin::DebugStructFinish { fields: 3 }
        }
        "std::fmt::Formatter::debug_tuple_field3_finish"
        | "core::fmt::Formatter::debug_tuple_field3_finish" => {
            Builtin::DebugTupleFinish { fields: 3 }
        }
        "std::fmt::Formatter::debug_struct_field4_finish"
        | "core::fmt::Formatter::debug_struct_field4_finish" => {
            Builtin::DebugStructFinish { fields: 4 }
        }
        "std::fmt::Formatter::debug_tuple_field4_finish"
        | "core::fmt::Formatter::debug_tuple_field4_finish" => {
            Builtin::DebugTupleFinish { fields: 4 }
        }
        "std::fmt::Formatter::debug_struct_field5_finish"
        | "core::fmt::Formatter::debug_struct_field5_finish" => {
            Builtin::DebugStructFinish { fields: 5 }
        }
        "std::fmt::Formatter::debug_tuple_field5_finish"
        | "core::fmt::Formatter::debug_tuple_field5_finish" => {
            Builtin::DebugTupleFinish { fields: 5 }
        }
        "std::fmt::Formatter::debug_struct_fields_finish"
        | "core::fmt::Formatter::debug_struct_fields_finish" => {
            Builtin::DebugStructFieldsFinish
        }
        "std::fmt::Formatter::debug_tuple_fields_finish"
        | "core::fmt::Formatter::debug_tuple_fields_finish" => {
            Builtin::DebugTupleFieldsFinish
        }
        "std::fmt::Formatter::debug_list" | "core::fmt::Formatter::debug_list" => {
            Builtin::FmtBuilderNew(FmtBuilderKind::List)
        }
        "std::fmt::Formatter::debug_set" | "core::fmt::Formatter::debug_set" => {
            Builtin::FmtBuilderNew(FmtBuilderKind::Set)
        }
        "std::fmt::Formatter::debug_map" | "core::fmt::Formatter::debug_map" => {
            Builtin::FmtBuilderNew(FmtBuilderKind::Map)
        }
        "std::fmt::Formatter::debug_struct" | "core::fmt::Formatter::debug_struct" => {
            Builtin::FmtBuilderNew(FmtBuilderKind::Struct)
        }
        "std::fmt::Formatter::debug_tuple" | "core::fmt::Formatter::debug_tuple" => {
            Builtin::FmtBuilderNew(FmtBuilderKind::Tuple)
        }
        "std::fmt::DebugList::entry" | "core::fmt::DebugList::entry"
        | "std::fmt::DebugSet::entry" | "core::fmt::DebugSet::entry" => {
            Builtin::FmtBuilderEntry
        }
        "std::fmt::DebugMap::entry" | "core::fmt::DebugMap::entry" => {
            Builtin::FmtBuilderMapEntry
        }
        "std::fmt::DebugStruct::field" | "core::fmt::DebugStruct::field" => {
            Builtin::FmtBuilderStructField
        }
        "std::fmt::DebugTuple::field" | "core::fmt::DebugTuple::field" => {
            Builtin::FmtBuilderTupleField
        }
        "std::fmt::DebugList::finish" | "core::fmt::DebugList::finish"
        | "std::fmt::DebugSet::finish" | "core::fmt::DebugSet::finish"
        | "std::fmt::DebugMap::finish" | "core::fmt::DebugMap::finish"
        | "std::fmt::DebugStruct::finish" | "core::fmt::DebugStruct::finish"
        | "std::fmt::DebugTuple::finish" | "core::fmt::DebugTuple::finish" => {
            Builtin::FmtBuilderFinish { exhaustive: true }
        }
        "std::fmt::DebugStruct::finish_non_exhaustive"
        | "core::fmt::DebugStruct::finish_non_exhaustive" => {
            Builtin::FmtBuilderFinish { exhaustive: false }
        }
        "std::iter::IntoIterator::into_iter" => Builtin::IntoIter,
        "std::iter::Iterator::next" => Builtin::IteratorNext,
        "std::iter::DoubleEndedIterator::next_back"
        | "core::iter::DoubleEndedIterator::next_back" => Builtin::IteratorNextBack,
        "std::iter::Iterator::size_hint" => Builtin::IteratorSizeHint,
        "std::iter::ExactSizeIterator::len" | "core::iter::ExactSizeIterator::len" => {
            Builtin::IteratorLen
        }
        // The slice views are named with their generic group
        // in the middle (`std::slice::Iter::<'a, T>::as_slice`),
        // so they match by prefix and suffix like the `Vec`
        // methods below.
        _ if (canonical.starts_with("std::slice::Iter")
            || canonical.starts_with("core::slice::Iter"))
            && (canonical.ends_with(">::as_slice")
                || canonical.ends_with(">::into_slice")
                || canonical.ends_with(">::as_mut_slice")) =>
        {
            Builtin::SliceIterAsSlice
        }
        "std::iter::Iterator::count" => Builtin::IteratorCount,
        "std::iter::Iterator::nth" => Builtin::IteratorNth,
        "std::iter::Iterator::fold" => Builtin::IteratorFold,
        "std::iter::Iterator::try_fold" => Builtin::IteratorTryFold,
        "std::iter::Iterator::for_each" => Builtin::IteratorForEach,
        "std::iter::Iterator::collect" => Builtin::IteratorCollect,
        "std::iter::FromIterator::from_iter" => Builtin::FromIteratorFromIter,
        // The thread surface (cooperative model, see `interp.rs`).
        // Registering these names also keeps the driver from
        // collecting the std thread machinery, whose internals
        // (spawn hooks, OS threads, packet transmutes) the value
        // model does not serve; the builtins model the observable
        // semantics instead.
        "std::thread::spawn" | "std::thread::spawn_unchecked" => Builtin::ThreadSpawn,
        "std::sync::Barrier::new" => Builtin::BarrierNew,
        "std::sync::Barrier::wait" => Builtin::BarrierWait,
        // The futex-backed sys locks (std's `sys::sync`
        // layer): the fast paths are plain atomic operations
        // the converted bodies serve, but the contended paths
        // sleep on a futex and have no converted body; the
        // builtins model the waiting on the cooperative
        // scheduler instead (see `builtins::sync`).
        "std::sys::sync::mutex::futex::Mutex::lock_contended" => {
            Builtin::SysMutexLockContended
        }
        "std::sys::sync::mutex::futex::Mutex::wake" => Builtin::SysMutexWake,
        "std::sys::sync::rwlock::futex::RwLock::read_contended" => {
            Builtin::SysRwLockReadContended
        }
        "std::sys::sync::rwlock::futex::RwLock::write_contended" => {
            Builtin::SysRwLockWriteContended
        }
        "std::sys::sync::rwlock::futex::RwLock::wake_writer_or_readers" => {
            Builtin::SysRwLockWakeWriterOrReaders
        }
        // The futex primitives themselves: bodies that ARE
        // converted (the rwlock's `downgrade`, `Once`, the
        // condvar, the thread parker) call them directly.
        "std::sys::sync::futex::unix::futex_wait" => Builtin::FutexWait,
        "std::sys::sync::futex::unix::futex_wake" => Builtin::FutexWake,
        "std::sys::sync::futex::unix::futex_wake_all" => Builtin::FutexWakeAll,
        "std::backtrace::Backtrace::capture" => Builtin::BacktraceCapture,
        "std::backtrace::Backtrace::force_capture" => {
            Builtin::BacktraceForceCapture
        }
        "std::backtrace::Backtrace::status" => Builtin::BacktraceStatus,
        _ if canonical.starts_with("std::thread::Builder")
            && (canonical.ends_with("::spawn")
                || canonical.ends_with("::spawn_unchecked")) =>
        {
            Builtin::ThreadBuilderSpawn
        }
        _ if canonical.starts_with("std::thread::JoinHandle")
            && canonical.ends_with("::join") =>
        {
            Builtin::ThreadJoin
        }
        // Optimizing MIR inlines `JoinHandle::join` into a
        // direct call to the inner type's join; it has the same
        // contract (the thread's result or panic payload) and
        // the same modelled receiver (see `join_handle_value`).
        _ if canonical.starts_with("std::thread::lifecycle::JoinInner")
            && canonical.ends_with("::join") =>
        {
            Builtin::ThreadJoin
        }
        _ if canonical.starts_with("std::thread::ScopedJoinHandle")
            && canonical.ends_with("::join") =>
        {
            Builtin::ScopedThreadJoin
        }
        _ if canonical.contains("::Scope::") && canonical.ends_with("::spawn") => {
            Builtin::ThreadScopedSpawn
        }
        "std::sys::thread_local::destructors::linux_like::register" => {
            Builtin::TlsRegisterDtor
        }
        "std::thread::local::panic_access_error" => Builtin::PanicAccessError,
        "std::thread::Builder::new" => Builtin::ThreadBuilderNew,
        "std::thread::Builder::name" => Builtin::ThreadBuilderName,
        "std::thread::Builder::stack_size" => Builtin::ThreadBuilderStackSize,
        "std::time::Duration::new" => Builtin::DurationNew,
        "std::time::Duration::from_secs" => Builtin::DurationFromSecs,
        "std::time::Duration::from_millis" => Builtin::DurationFromMillis,
        "std::time::Duration::from_micros" => Builtin::DurationFromMicros,
        "std::time::Duration::from_nanos" => Builtin::DurationFromNanos,
        "std::intrinsics::caller_location" => Builtin::CallerLocation,
        "std::intrinsics::va_arg" | "core::intrinsics::va_arg" => Builtin::VaArg,
        // The harness calls either a crate-local wrapper
        // (`<crate>::test::test_main_env_args`) or, for an empty
        // test list, the `test` crate's function directly.
        _ if canonical == "test::test_main_env_args"
            || canonical.ends_with("::test::test_main_env_args") =>
        {
            Builtin::TestMainEnvArgs
        }
        "std::panic::set_hook" => Builtin::PanicSetHook,
        "std::panic::set_backtrace_style" => Builtin::PanicSetBacktraceStyle,
        "std::panic::always_abort" => Builtin::PanicAlwaysAbort,
        "std::panic::take_hook" => Builtin::PanicTakeHook,
        "std::time::Instant::now" => Builtin::InstantNow,
        "std::time::SystemTime::now" => Builtin::SystemTimeNow,
        "std::sys::random::linux::hashmap_random_keys" => Builtin::HashmapRandomKeys,
        "std::sys::process::unix::common::Command::new" => Builtin::ProcessCommandNew,
        "std::sys::process::unix::common::Command::arg" => Builtin::ProcessCommandArg,
        "std::sys::process::unix::common::Command::cwd" => Builtin::ProcessCommandCwd,
        "std::sys::process::unix::common::Command::env_mut" => Builtin::ProcessCommandEnvMut,
        "std::sys::process::env::CommandEnv::set" => Builtin::ProcessCommandEnvSet,
        "std::sys::process::env::CommandEnv::remove" => Builtin::ProcessCommandEnvRemove,
        "std::sys::process::env::CommandEnv::clear" => Builtin::ProcessCommandEnvClear,
        "std::sys::process::unix::common::Command::stdin" => Builtin::ProcessCommandStdio,
        "std::sys::process::unix::common::Command::stdout" => Builtin::ProcessCommandStdio,
        "std::sys::process::unix::common::Command::stderr" => Builtin::ProcessCommandStdio,
        "std::sys::process::unix::unix::Command::spawn" => Builtin::ProcessSpawn,
        "std::env::current_exe" => Builtin::CurrentExe,
        "std::process::id" => Builtin::ProcessId,
        "core::io::error::os_functions::get_os_functions" => Builtin::IoGetOsFunctions,
        "core::io::error::os_functions::set_functions" => Builtin::IoSetOsFunctions,
        "core::io::error::os_functions::set_functions::set_functions_inner" => {
            Builtin::IoSetOsFunctions
        }
        "std::sys::io::error::unix::decode_error_kind" => Builtin::IoDecodeErrorKind,
        "alloc::io::error::custom_owner_from_box" => Builtin::IoCustomOwnerFromBox,
        "core::io::error::repr::Repr::new_custom" => Builtin::IoReprNewCustom,
        "std::sys::io::error::unix::format_error" => Builtin::IoFormatOsError,
        "std::sys::io::error::unix::is_interrupted" => Builtin::IoIsInterrupted,
        "std::process::Command::new" => Builtin::ProcessCommandNewPublic,
        "std::process::Command::arg" => Builtin::ProcessCommandArgPublic,
        "std::process::Command::current_dir" => Builtin::ProcessCommandCwdPublic,
        "std::process::Command::env" => Builtin::ProcessCommandEnvPublic,
        "std::process::Command::env_remove" => Builtin::ProcessCommandEnvRemovePublic,
        "std::process::Command::env_clear" => Builtin::ProcessCommandEnvClearPublic,
        "std::process::Command::stdin" => Builtin::ProcessCommandStdioPublic,
        "std::process::Command::stdout" => Builtin::ProcessCommandStdioPublic,
        "std::process::Command::stderr" => Builtin::ProcessCommandStdioPublic,
        "std::process::Command::spawn" => Builtin::ProcessSpawn,
        "std::process::Command::output" => Builtin::ProcessOutput,
        "std::process::Command::status" => Builtin::ProcessSpawn,
        "std::process::Child::wait" => Builtin::ProcessChildWait,
        "std::process::Child::try_wait" => Builtin::ProcessChildTryWait,
        "std::process::Child::kill" => Builtin::ProcessChildKill,
        "std::process::Child::wait_with_output" => {
            Builtin::ProcessChildWaitWithOutput
        }
        "std::process::Child::id" => Builtin::ProcessChildId,
        "std::sys::process::output" => Builtin::ProcessOutput,
        "std::time::Instant::checked_add" => Builtin::TimeCheckedAdd(false),
        "std::time::SystemTime::checked_add" => Builtin::TimeCheckedAdd(true),
        "std::time::Instant::checked_sub" => Builtin::TimeCheckedSub(false),
        "std::time::SystemTime::checked_sub" => Builtin::TimeCheckedSub(true),
        "std::time::Instant::duration_since" => Builtin::InstantDurationSince,
        "std::time::Instant::checked_duration_since" => {
            Builtin::InstantCheckedDurationSince
        }
        "std::time::Instant::saturating_duration_since" => {
            Builtin::InstantSaturatingDurationSince
        }
        "std::time::Instant::elapsed" => Builtin::InstantElapsed,
        "std::time::SystemTime::duration_since" => Builtin::SystemTimeDurationSince,
        "std::time::SystemTime::elapsed" => Builtin::SystemTimeElapsed,
        "std::thread::scope" => Builtin::ThreadScope,
        "std::thread::current" => Builtin::ThreadCurrent,
        "std::thread::Thread::id" => Builtin::ThreadIdOf,
        "std::thread::Thread::name" => Builtin::ThreadName,
        "std::thread::sleep" | "std::thread::sleep_ms" => Builtin::ThreadSleep,
        "std::thread::yield_now" => Builtin::ThreadYield,
        // The mpsc surface: the channel state lives in the
        // interpreter (see `ChannelState` in `interp.rs`).
        "std::sync::mpsc::channel" => Builtin::MpscChannel,
        "std::sync::mpsc::sync_channel" => Builtin::MpscSyncChannel,
        _ if canonical.contains("mpsc::SyncSender") && canonical.ends_with("::send") => {
            Builtin::MpscSend
        }
        _ if canonical.contains("mpsc::Sender") && canonical.ends_with("::send") => {
            Builtin::MpscSend
        }
        _ if canonical.contains("mpsc::SyncSender")
            && canonical.ends_with("::try_send") =>
        {
            Builtin::MpscTrySend
        }
        _ if canonical.contains("mpsc::Receiver") && canonical.ends_with("::recv") => {
            Builtin::MpscRecv
        }
        _ if canonical.contains("mpsc::Receiver")
            && canonical.ends_with("::try_recv") =>
        {
            Builtin::MpscTryRecv
        }
        _ if canonical.contains("mpsc::Receiver")
            && canonical.ends_with("::recv_timeout") =>
        {
            Builtin::MpscRecvTimeout
        }
        _ if canonical.contains("mpsc::Receiver") && canonical.ends_with("::iter") => {
            Builtin::MpscIter
        }
        "std::convert::From::from" => Builtin::FromFrom,
        // The waker wake methods: std's bodies read the
        // `RawWaker`'s vtable reference out of the waker value,
        // which the modelled `From` waker (a one-field aggregate
        // carrying the handle) cannot serve, so the dispatch is
        // modelled on the value's shape, exactly as the `Clone`
        // model and the `WakerDrop` glue judge it. Registering
        // the names also keeps the driver from collecting the
        // std bodies (see sparagmos-driver's reach pass).
        "std::task::Waker::wake" | "core::task::Waker::wake"
        | "std::task::LocalWaker::wake" | "core::task::LocalWaker::wake" => {
            Builtin::WakerWake
        }
        "std::task::Waker::wake_by_ref" | "core::task::Waker::wake_by_ref"
        | "std::task::LocalWaker::wake_by_ref"
        | "core::task::LocalWaker::wake_by_ref" => Builtin::WakerWakeByRef,
        "std::hint::must_use" => Builtin::MustUse,
        "std::ops::Deref::deref" => Builtin::Deref,
        "std::path::Path::parent" => Builtin::PathPart(PathPart::Parent),
        "std::path::Path::file_name" => Builtin::PathPart(PathPart::FileName),
        "std::path::Path::file_stem" => Builtin::PathPart(PathPart::FileStem),
        "std::path::Path::extension" => Builtin::PathPart(PathPart::Extension),
        "std::path::Path::components" => Builtin::PathComponents,
        "std::path::Path::to_path_buf" => Builtin::PathToPathBuf,
        "std::path::PathBuf::_push" => Builtin::PathBufPush,
        "std::path::PathBuf::pop" => Builtin::PathBufPop,
        "std::path::PathBuf::_set_extension" => Builtin::PathBufSetExtension,
        "std::path::Path::_with_extension" => Builtin::PathWithExtension,
        "std::path::PathBuf::new" => Builtin::PathBufNew,
        "std::path::PathBuf::with_capacity" => Builtin::PathBufNew,
        "std::str::from_utf8" | "core::str::from_utf8" => Builtin::FromUtf8,
        _ if canonical.contains("RangeInclusive") && canonical.ends_with("::new") => {
            Builtin::RangeInclusiveNew
        }
        _ if canonical.ends_with("::to_uppercase") => Builtin::ToUppercase,
        _ if canonical.ends_with("::to_lowercase") => Builtin::ToLowercase,
        // The panic path: these callees never return; the builtin
        // raises the guest panic the call stands for.
        "core::panicking::panic" => Builtin::PanicStr,
        "std::rt::panic_fmt" | "core::panicking::panic_fmt" => Builtin::PanicFmt,
        "core::panicking::assert_failed" => Builtin::AssertFailed,
        "core::panicking::panic_nounwind_fmt" => Builtin::PanicNounwindFmt,
        // `panic_nounwind(&str)` and `Option::expect_failed(&str)`
        // panic with exactly the given message, as `panic` does.
        "core::panicking::panic_nounwind" => Builtin::PanicStr,
        "std::option::expect_failed" | "core::option::expect_failed" => Builtin::PanicStr,
        "std::option::unwrap_failed" | "core::option::unwrap_failed" => Builtin::OptionUnwrapFailed,
        "std::result::unwrap_failed" | "core::result::unwrap_failed" => Builtin::ResultUnwrapFailed,
        "core::num::from_ascii_bytes_radix_panic" => Builtin::FromAsciiRadixPanic,
        "core::slice::index::slice_index_fail" => Builtin::SliceIndexFail,
        "core::slice::copy_from_slice_impl::len_mismatch_fail" => Builtin::LenMismatchFail,
        "std::str::pattern::StrSearcher::new" => Builtin::StrSearcherNew,
        "core::slice::memchr::memchr" => Builtin::Memchr { reverse: false },
        "core::slice::memchr::memrchr" => Builtin::Memchr { reverse: true },
        "core::slice::index::RangeError::report" => Builtin::RangeErrorReport,
        // The panic-catching path: the `catch_unwind` intrinsic
        // itself (its std wrapper's body is collected and runs;
        // only the intrinsic is modelled here), the payload
        // plumbing around it, and the `dyn Any` downcast methods,
        // whose collected bodies dispatch through `Any::type_id`,
        // a virtual call this model cannot resolve.
        "std::intrinsics::catch_unwind" | "core::intrinsics::catch_unwind" => {
            Builtin::CatchUnwind
        }
        "std::panicking::catch_unwind::cleanup" => Builtin::PanicCleanup,
        "std::panicking::panicking" | "std::thread::panicking" => Builtin::Panicking,
        "std::rt::panic_count::count_is_zero" => Builtin::PanicCountIsZero,
        "std::rt::panic_count::is_zero_slow_path" => Builtin::PanicCountIsZero,
        "std::panicking::begin_panic" | "std::rt::begin_panic" => Builtin::PanicAny,
        "std::panic::resume_unwind" | "std::panicking::resume_unwind" => {
            Builtin::ResumeUnwind
        }
        "<(dyn std::any::Any + 'static)>::downcast_ref"
        | "<(dyn std::any::Any + std::marker::Send + 'static)>::downcast_ref" => {
            Builtin::DynAnyDowncastRef
        }
        "<(dyn std::any::Any + 'static)>::downcast"
        | "<(dyn std::any::Any + std::marker::Send + 'static)>::downcast" => {
            Builtin::DynAnyDowncast(0)
        }
        "<(dyn std::any::Any + 'static)>::is"
        | "<(dyn std::any::Any + std::marker::Send + 'static)>::is" => Builtin::DynAnyIs,
        // By-value downcast is an inherent method on the `Box`
        // itself (in alloc's `convert` module), not on the dyn
        // type: `Box<dyn Any>::downcast` is the call guest code
        // actually makes.
        "std::boxed::convert::<impl std::boxed::Box<(dyn std::any::Any + 'static), A>>::downcast"
        | "std::boxed::convert::<impl std::boxed::Box<(dyn std::any::Any + std::marker::Send + 'static), A>>::downcast" => {
            Builtin::DynAnyDowncast(1)
        }
        // Comparison traits: the impls for the modelled data types
        // are the derived ones, i.e. semantic value comparison.
        _ if canonical.ends_with("::PartialEq::eq") => Builtin::PartialEqEq,
        _ if canonical.ends_with("::PartialEq::ne") => Builtin::PartialEqNe,
        _ if canonical.ends_with("::PartialOrd::partial_cmp") => Builtin::PartialCmp,
        _ if canonical.ends_with("::PartialOrd::lt") => Builtin::PartialOrdRel(RelOp::Lt),
        _ if canonical.ends_with("::PartialOrd::le") => Builtin::PartialOrdRel(RelOp::Le),
        _ if canonical.ends_with("::PartialOrd::gt") => Builtin::PartialOrdRel(RelOp::Gt),
        _ if canonical.ends_with("::PartialOrd::ge") => Builtin::PartialOrdRel(RelOp::Ge),
        _ if canonical.ends_with("::Ord::cmp") => Builtin::OrdCmp,
        _ if canonical.ends_with("::Clone::clone") => Builtin::CloneClone,
        "std::default::Default::default" | "core::default::Default::default" => {
            Builtin::DefaultDefault
        }
        "std::string::ToString::to_string" => Builtin::ToStringToString,
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
        "alloc::alloc::__rust_no_alloc_shim_is_unstable_v2" => {
            Builtin::NoAllocShimMarker
        }
        "std::convert::AsRef::as_ref" | "core::convert::AsRef::as_ref" => Builtin::AsRef,
        "std::convert::AsMut::as_mut" | "core::convert::AsMut::as_mut" => Builtin::AsMut,
        _ if canonical.starts_with("std::vec::Vec")
            && canonical.ends_with(">::into_boxed_slice") =>
        {
            Builtin::VecIntoBoxedSlice
        }
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
        "std::convert::Into::into" => Builtin::IntoInto,
        "std::borrow::ToOwned::to_owned" => Builtin::ToOwned,
        _ if canonical.starts_with("std::boxed::Box") && canonical.ends_with(">::new") => {
            Builtin::BoxNew
        }
        _ if canonical.starts_with("std::boxed::Box") && canonical.ends_with(">::new_uninit") => {
            Builtin::BoxNewUninit
        }
        "std::boxed::box_assume_init_into_vec_unsafe" => Builtin::BoxAssumeInitIntoVec,
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::new") => {
            Builtin::VecNew
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::push") => {
            Builtin::VecPush
        }
        _ if canonical.starts_with("std::vec::Vec")
            && canonical.ends_with(">::extend_from_slice") =>
        {
            Builtin::VecExtendFromSlice
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::len") => {
            Builtin::VecLen
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::is_empty") => {
            Builtin::VecIsEmpty
        }
        _ if canonical.starts_with("std::vec::Vec")
            && (canonical.ends_with(">::with_capacity")
                || canonical.ends_with(">::with_capacity_in")) =>
        {
            Builtin::VecWithCapacity
        }
        "std::vec::from_elem" => Builtin::VecFromElem,
        _ if canonical.starts_with("std::vec::Vec")
            && (canonical.ends_with(">::reserve")
                || canonical.ends_with(">::reserve_exact")) =>
        {
            Builtin::VecReserve
        }
        _ if canonical.starts_with("std::vec::Vec")
            && (canonical.ends_with(">::try_reserve")
                || canonical.ends_with(">::try_reserve_exact")) =>
        {
            Builtin::VecTryReserve
        }
        _ if canonical.starts_with("std::vec::Vec")
            && (canonical.ends_with(">::shrink_to_fit")
                || canonical.ends_with(">::shrink_to")) =>
        {
            Builtin::VecShrink
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::set_len") => {
            Builtin::VecSetLen
        }
        _ if canonical.starts_with("std::vec::Vec")
            && (canonical.ends_with(">::as_ptr") || canonical.ends_with(">::as_mut_ptr")) =>
        {
            Builtin::VecAsPtr
        }
        _ if canonical.starts_with("std::vec::Vec")
            && canonical.ends_with(">::extend_trusted") =>
        {
            Builtin::VecExtendTrusted
        }
        "std::vec::spec_from_iter_nested::SpecFromIterNested::from_iter" => {
            Builtin::VecFromIterNested
        }
        // The in-place collect entry the `SpecFromIter`
        // specialization selects when the iterator's source is a
        // `Vec` allocation (`SourceIter` + `InPlaceIterable` with
        // an `AsVecIntoIter` source: `vec::IntoIter` itself and
        // every adapter forwarding those traits). Its collected
        // body drives the source allocation's raw buffer through
        // `SourceIter::as_inner` / `AsVecIntoIter::as_into_iter`,
        // which the value model does not serve; the observable
        // semantics are the nested collect's -- the yielded
        // items, in order, in a fresh vector -- so the same
        // model serves this entry.
        "std::vec::in_place_collect::from_iter_in_place"
        | "alloc::vec::in_place_collect::from_iter_in_place" => {
            Builtin::VecFromIterNested
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::capacity") => {
            Builtin::VecCapacity
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::remove::assert_failed") => Builtin::VecRemoveAssertFailed,
        _ if canonical.starts_with("core::slice::<impl") && canonical.ends_with(">::iter") => {
            Builtin::SliceIter
        }
        _ if canonical.starts_with("std::collections::BTreeSet")
            && canonical.ends_with(">::iter") =>
        {
            Builtin::BTreeIter
        }
        _ if canonical.starts_with("std::collections::BTreeMap")
            && canonical.ends_with(">::iter") =>
        {
            Builtin::BTreeIter
        }
        _ if canonical.starts_with("std::collections::BTreeMap")
            && canonical.ends_with(">::keys") =>
        {
            Builtin::BTreeKeys
        }
        _ if canonical.starts_with("std::collections::BTreeMap")
            && canonical.ends_with(">::values") =>
        {
            Builtin::BTreeValues
        }
        _ if (canonical.starts_with("std::slice::<impl")
            || canonical.starts_with("alloc::slice::<impl"))
            && (canonical.ends_with(">::to_vec") || canonical.ends_with(">::to_vec_in")) =>
        {
            Builtin::SliceToVec
        }
        _ if canonical.starts_with("core::slice::<impl")
            && canonical.ends_with(">::windows") =>
        {
            Builtin::SliceWindows
        }
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
        "std::iter::empty" => Builtin::IterEmpty,
        "std::ops::Index::index" => Builtin::IndexIndex,
        "std::ops::IndexMut::index_mut" => Builtin::IndexMut,
        "std::ops::DerefMut::deref_mut" => Builtin::DerefMut,
        _ if canonical.starts_with("std::cell::Cell") && canonical.ends_with(">::new") => {
            Builtin::CellNew
        }
        _ if canonical.starts_with("std::cell::Cell") && canonical.ends_with(">::get") => {
            Builtin::CellGet
        }
        _ if canonical.starts_with("std::cell::Cell") && canonical.ends_with(">::set") => {
            Builtin::CellSet
        }
        _ if canonical.starts_with("std::cell::Cell") && canonical.ends_with(">::replace") => {
            Builtin::CellReplace
        }
        _ if canonical.starts_with("std::cell::Cell") && canonical.ends_with(">::into_inner") => {
            Builtin::CellIntoInner
        }
        _ if canonical.starts_with("std::cell::RefCell") && canonical.ends_with(">::new") => {
            Builtin::RefCellNew
        }
        _ if canonical.starts_with("std::cell::RefCell") && canonical.ends_with(">::borrow") => {
            Builtin::RefCellBorrow
        }
        _ if canonical.starts_with("std::cell::RefCell")
            && canonical.ends_with(">::borrow_mut") =>
        {
            Builtin::RefCellBorrowMut
        }
        _ if canonical.starts_with("std::cell::RefCell")
            && canonical.ends_with(">::try_borrow") =>
        {
            Builtin::RefCellTryBorrow
        }
        _ if canonical.starts_with("std::cell::RefCell")
            && canonical.ends_with(">::try_borrow_mut") =>
        {
            Builtin::RefCellTryBorrowMut
        }
        _ if canonical.starts_with("std::cell::RefCell") && canonical.ends_with(">::get_mut") => {
            Builtin::RefCellGetMut
        }
        _ if canonical.starts_with("std::cell::RefCell") && canonical.ends_with(">::replace") => {
            Builtin::RefCellReplace
        }
        _ if canonical.starts_with("std::cell::RefCell") && canonical.ends_with(">::as_ptr") => {
            Builtin::RefCellAsPtr
        }
        _ if canonical.contains("option::Option") && canonical.ends_with(">::unwrap") => {
            Builtin::OptionUnwrap
        }
        _ if canonical.contains("result::Result") && canonical.ends_with(">::unwrap") => {
            Builtin::ResultUnwrap
        }
        _ if canonical.contains("option::Option") && canonical.ends_with(">::is_some") => {
            Builtin::OptionIsSome
        }
        _ if canonical.contains("option::Option") && canonical.ends_with(">::is_none") => {
            Builtin::OptionIsNone
        }
        _ if canonical.contains("result::Result") && canonical.ends_with(">::is_ok") => {
            Builtin::ResultIsOk
        }
        _ if canonical.contains("result::Result") && canonical.ends_with(">::is_err") => {
            Builtin::ResultIsErr
        }
        _ if canonical.contains("option::Option") && canonical.ends_with(">::take") => {
            Builtin::OptionTake
        }
        "std::mem::drop" => Builtin::MemDrop,
        "std::mem::swap" => Builtin::MemSwap,
        "std::ptr::swap_nonoverlapping" | "core::ptr::swap_nonoverlapping"
        | "std::intrinsics::typed_swap_nonoverlapping"
        | "core::intrinsics::typed_swap_nonoverlapping" => Builtin::SwapNonoverlapping,
        "std::mem::size_of" => Builtin::MemSizeOf,
        "std::mem::align_of" => Builtin::MemAlignOf,
        "std::mem::size_of_val" => Builtin::MemSizeOfVal,
        "std::mem::align_of_val" => Builtin::MemAlignOfVal,
        // The same two intrinsics in their direct call form
        // (`Layout::for_value` inside a converted std body calls
        // them as functions, not as `std::mem` wrappers).
        "std::intrinsics::size_of_val" | "core::intrinsics::size_of_val" => {
            Builtin::MemSizeOfVal
        }
        "std::intrinsics::align_of_val" | "core::intrinsics::align_of_val" => {
            Builtin::MemAlignOfVal
        }
        "std::intrinsics::fadd_fast" | "core::intrinsics::fadd_fast" => {
            Builtin::FastFloat(FastFloatOp::Add)
        }
        "std::intrinsics::fsub_fast" | "core::intrinsics::fsub_fast" => {
            Builtin::FastFloat(FastFloatOp::Sub)
        }
        "std::intrinsics::fmul_fast" | "core::intrinsics::fmul_fast" => {
            Builtin::FastFloat(FastFloatOp::Mul)
        }
        "std::intrinsics::fdiv_fast" | "core::intrinsics::fdiv_fast" => {
            Builtin::FastFloat(FastFloatOp::Div)
        }
        "std::intrinsics::frem_fast" | "core::intrinsics::frem_fast" => {
            Builtin::FastFloat(FastFloatOp::Rem)
        }
        "std::intrinsics::float_to_int_unchecked" | "core::intrinsics::float_to_int_unchecked" => {
            Builtin::FloatToIntUnchecked
        }
        "std::mem::needs_drop" => Builtin::MemNeedsDrop,
        _ if canonical.starts_with("std::mem::ManuallyDrop") && canonical.ends_with(">::new") => {
            Builtin::ManuallyDropNew
        }
        "std::intrinsics::black_box" => Builtin::BlackBox,
        "std::intrinsics::ctpop" | "core::intrinsics::ctpop" => Builtin::IntBitOp(BitOp::CtPop),
        "std::intrinsics::ctlz" | "core::intrinsics::ctlz" => Builtin::IntBitOp(BitOp::Ctlz),
        "std::intrinsics::cttz" | "core::intrinsics::cttz" => Builtin::IntBitOp(BitOp::Cttz),
        // The `_nonzero` variants promise a non-zero operand (a
        // zero operand is undefined behaviour in the guest); on
        // every input where they are defined they agree with the
        // plain operation, which is what the model computes.
        "std::intrinsics::ctlz_nonzero" | "core::intrinsics::ctlz_nonzero" => {
            Builtin::IntBitOp(BitOp::Ctlz)
        }
        "std::intrinsics::cttz_nonzero" | "core::intrinsics::cttz_nonzero" => {
            Builtin::IntBitOp(BitOp::Cttz)
        }
        "std::intrinsics::bitreverse" | "core::intrinsics::bitreverse" => {
            Builtin::IntBitOp(BitOp::BitReverse)
        }
        "std::intrinsics::bswap" | "core::intrinsics::bswap" => Builtin::IntBitOp(BitOp::Bswap),
        "std::intrinsics::saturating_add" | "core::intrinsics::saturating_add" => {
            Builtin::SaturatingArith(SatOp::Add)
        }
        "std::intrinsics::saturating_sub" | "core::intrinsics::saturating_sub" => {
            Builtin::SaturatingArith(SatOp::Sub)
        }
        "std::intrinsics::atomic_load" | "core::intrinsics::atomic_load" => Builtin::AtomicLoad,
        "std::intrinsics::atomic_store" | "core::intrinsics::atomic_store" => {
            Builtin::AtomicStore
        }
        "std::intrinsics::atomic_cxchg" | "core::intrinsics::atomic_cxchg" => {
            Builtin::AtomicCxchg
        }
        "std::intrinsics::atomic_cxchgweak" | "core::intrinsics::atomic_cxchgweak" => {
            Builtin::AtomicCxchg
        }
        "std::intrinsics::atomic_xchg" | "core::intrinsics::atomic_xchg" => Builtin::AtomicXchg,
        "std::intrinsics::atomic_xadd" | "core::intrinsics::atomic_xadd" => Builtin::AtomicXadd,
        "std::intrinsics::atomic_xsub" | "core::intrinsics::atomic_xsub" => Builtin::AtomicXsub,
        "std::intrinsics::atomic_and" | "core::intrinsics::atomic_and" => {
            Builtin::AtomicFetchOp(FetchOp::And)
        }
        "std::intrinsics::atomic_or" | "core::intrinsics::atomic_or" => {
            Builtin::AtomicFetchOp(FetchOp::Or)
        }
        "std::intrinsics::atomic_xor" | "core::intrinsics::atomic_xor" => {
            Builtin::AtomicFetchOp(FetchOp::Xor)
        }
        "std::intrinsics::atomic_nand" | "core::intrinsics::atomic_nand" => {
            Builtin::AtomicFetchOp(FetchOp::Nand)
        }
        "std::intrinsics::atomic_max" | "core::intrinsics::atomic_max" => {
            Builtin::AtomicFetchOp(FetchOp::Max)
        }
        "std::intrinsics::atomic_min" | "core::intrinsics::atomic_min" => {
            Builtin::AtomicFetchOp(FetchOp::Min)
        }
        "std::intrinsics::atomic_umax" | "core::intrinsics::atomic_umax" => {
            Builtin::AtomicFetchOp(FetchOp::UMax)
        }
        "std::intrinsics::atomic_umin" | "core::intrinsics::atomic_umin" => {
            Builtin::AtomicFetchOp(FetchOp::UMin)
        }
        "std::intrinsics::atomic_fence" | "core::intrinsics::atomic_fence" => {
            Builtin::AtomicFence
        }
        "std::intrinsics::assert_inhabited" | "core::intrinsics::assert_inhabited" => {
            Builtin::AssertValidity
        }
        "std::intrinsics::assert_zero_valid" | "core::intrinsics::assert_zero_valid" => {
            Builtin::AssertValidity
        }
        "std::intrinsics::assert_mem_uninitialized_valid"
        | "core::intrinsics::assert_mem_uninitialized_valid" => Builtin::AssertValidity,
        // A volatile access is a plain access in this
        // single-threaded interpreter: volatility constrains the
        // compiler's reordering and caching, neither of which the
        // interpreter performs, and no concurrent observer exists.
        "std::intrinsics::volatile_load" | "core::intrinsics::volatile_load" => {
            Builtin::VolatileLoad
        }
        "std::intrinsics::unaligned_volatile_load"
        | "core::intrinsics::unaligned_volatile_load" => Builtin::AtomicLoad,
        "std::intrinsics::volatile_store" | "core::intrinsics::volatile_store" => {
            Builtin::VolatileStore
        }
        // A nontemporal store is a plain store in this
        // interpreter: the nontemporal hint only relaxes the
        // cache behaviour of the access, which no observer of
        // the sequential execution can distinguish.
        "std::intrinsics::nontemporal_store" | "core::intrinsics::nontemporal_store" => {
            Builtin::VolatileStore
        }
        "std::intrinsics::unaligned_volatile_store"
        | "core::intrinsics::unaligned_volatile_store" => Builtin::AtomicStore,
        "std::intrinsics::volatile_copy_memory" | "core::intrinsics::volatile_copy_memory" => {
            Builtin::VolatileCopy
        }
        "std::intrinsics::volatile_copy_nonoverlapping_memory"
        | "core::intrinsics::volatile_copy_nonoverlapping_memory" => Builtin::VolatileCopy,
        // The `copy` intrinsics in their call form (the statement
        // form of `copy_nonoverlapping` is handled in the
        // interpreter proper): the same element copy as the
        // `std::ptr` wrappers, whose snapshot makes it a correct
        // memmove.
        "std::intrinsics::copy" | "core::intrinsics::copy" => Builtin::PtrCopy,
        "std::intrinsics::copy_nonoverlapping" | "core::intrinsics::copy_nonoverlapping" => {
            Builtin::PtrCopy
        }
        "std::intrinsics::write_bytes" | "core::intrinsics::write_bytes" => Builtin::WriteBytes,
        "std::intrinsics::volatile_set_memory" | "core::intrinsics::volatile_set_memory" => {
            Builtin::VolatileSet
        }
        "std::intrinsics::arith_offset" | "core::intrinsics::arith_offset" => {
            Builtin::ArithOffset { wrapping: true }
        }
        "std::intrinsics::offset" | "core::intrinsics::offset" => {
            Builtin::ArithOffset { wrapping: false }
        }
        "std::intrinsics::ptr_offset_from" | "core::intrinsics::ptr_offset_from" => {
            Builtin::PtrOffsetFrom { unsigned: false }
        }
        "std::intrinsics::ptr_offset_from_unsigned"
        | "core::intrinsics::ptr_offset_from_unsigned" => {
            Builtin::PtrOffsetFrom { unsigned: true }
        }
        "std::mem::zeroed" => Builtin::MemZeroed,
        "std::ptr::null" | "std::ptr::null_mut" => Builtin::PtrNull,
        "std::ptr::copy_nonoverlapping" | "std::ptr::copy" => Builtin::PtrCopy,
        "std::intrinsics::compare_bytes" | "core::intrinsics::compare_bytes" => {
            Builtin::CompareBytes
        }
        "std::intrinsics::simd::simd_add" | "core::intrinsics::simd::simd_add" => {
            Builtin::SimdOp(SimdOp::Add)
        }
        "std::intrinsics::simd::simd_sub" | "core::intrinsics::simd::simd_sub" => {
            Builtin::SimdOp(SimdOp::Sub)
        }
        "std::intrinsics::simd::simd_mul" | "core::intrinsics::simd::simd_mul" => {
            Builtin::SimdOp(SimdOp::Mul)
        }
        "std::intrinsics::simd::simd_div" | "core::intrinsics::simd::simd_div" => {
            Builtin::SimdOp(SimdOp::Div)
        }
        "std::intrinsics::simd::simd_rem" | "core::intrinsics::simd::simd_rem" => {
            Builtin::SimdOp(SimdOp::Rem)
        }
        "std::intrinsics::simd::simd_and" | "core::intrinsics::simd::simd_and" => {
            Builtin::SimdOp(SimdOp::And)
        }
        "std::intrinsics::simd::simd_or" | "core::intrinsics::simd::simd_or" => {
            Builtin::SimdOp(SimdOp::Or)
        }
        "std::intrinsics::simd::simd_xor" | "core::intrinsics::simd::simd_xor" => {
            Builtin::SimdOp(SimdOp::Xor)
        }
        "std::intrinsics::simd::simd_shl" | "core::intrinsics::simd::simd_shl" => {
            Builtin::SimdOp(SimdOp::Shl)
        }
        "std::intrinsics::simd::simd_shr" | "core::intrinsics::simd::simd_shr" => {
            Builtin::SimdOp(SimdOp::Shr)
        }
        "std::intrinsics::simd::simd_eq" | "core::intrinsics::simd::simd_eq" => {
            Builtin::SimdOp(SimdOp::Eq)
        }
        "std::intrinsics::simd::simd_ne" | "core::intrinsics::simd::simd_ne" => {
            Builtin::SimdOp(SimdOp::Ne)
        }
        "std::intrinsics::simd::simd_lt" | "core::intrinsics::simd::simd_lt" => {
            Builtin::SimdOp(SimdOp::Lt)
        }
        "std::intrinsics::simd::simd_le" | "core::intrinsics::simd::simd_le" => {
            Builtin::SimdOp(SimdOp::Le)
        }
        "std::intrinsics::simd::simd_gt" | "core::intrinsics::simd::simd_gt" => {
            Builtin::SimdOp(SimdOp::Gt)
        }
        "std::intrinsics::simd::simd_ge" | "core::intrinsics::simd::simd_ge" => {
            Builtin::SimdOp(SimdOp::Ge)
        }
        "std::intrinsics::simd::simd_neg" | "core::intrinsics::simd::simd_neg" => {
            Builtin::SimdOp(SimdOp::Neg)
        }
        "std::intrinsics::simd::simd_fabs" | "core::intrinsics::simd::simd_fabs" => {
            Builtin::SimdOp(SimdOp::Fabs)
        }
        "std::intrinsics::simd::simd_ceil" | "core::intrinsics::simd::simd_ceil" => {
            Builtin::SimdOp(SimdOp::Ceil)
        }
        "std::intrinsics::simd::simd_floor" | "core::intrinsics::simd::simd_floor" => {
            Builtin::SimdOp(SimdOp::Floor)
        }
        "std::intrinsics::simd::simd_trunc" | "core::intrinsics::simd::simd_trunc" => {
            Builtin::SimdOp(SimdOp::Trunc)
        }
        "std::intrinsics::simd::simd_round" | "core::intrinsics::simd::simd_round" => {
            Builtin::SimdOp(SimdOp::Round)
        }
        "std::intrinsics::simd::simd_round_ties_even"
        | "core::intrinsics::simd::simd_round_ties_even" => {
            Builtin::SimdOp(SimdOp::RoundTiesEven)
        }
        "std::intrinsics::simd::simd_fsqrt" | "core::intrinsics::simd::simd_fsqrt" => {
            Builtin::SimdOp(SimdOp::Fsqrt)
        }
        "std::intrinsics::simd::simd_fexp" | "core::intrinsics::simd::simd_fexp" => {
            Builtin::SimdOp(SimdOp::Fexp)
        }
        "std::intrinsics::simd::simd_fexp2" | "core::intrinsics::simd::simd_fexp2" => {
            Builtin::SimdOp(SimdOp::Fexp2)
        }
        "std::intrinsics::simd::simd_flog" | "core::intrinsics::simd::simd_flog" => {
            Builtin::SimdOp(SimdOp::Flog)
        }
        "std::intrinsics::simd::simd_flog2" | "core::intrinsics::simd::simd_flog2" => {
            Builtin::SimdOp(SimdOp::Flog2)
        }
        "std::intrinsics::simd::simd_flog10" | "core::intrinsics::simd::simd_flog10" => {
            Builtin::SimdOp(SimdOp::Flog10)
        }
        "std::intrinsics::simd::simd_fsin" | "core::intrinsics::simd::simd_fsin" => {
            Builtin::SimdOp(SimdOp::Fsin)
        }
        "std::intrinsics::simd::simd_fcos" | "core::intrinsics::simd::simd_fcos" => {
            Builtin::SimdOp(SimdOp::Fcos)
        }
        "std::intrinsics::simd::simd_bswap" | "core::intrinsics::simd::simd_bswap" => {
            Builtin::SimdOp(SimdOp::Bswap)
        }
        "std::intrinsics::simd::simd_bitreverse" | "core::intrinsics::simd::simd_bitreverse" => {
            Builtin::SimdOp(SimdOp::BitReverse)
        }
        "std::intrinsics::simd::simd_ctlz" | "core::intrinsics::simd::simd_ctlz" => {
            Builtin::SimdOp(SimdOp::Ctlz)
        }
        "std::intrinsics::simd::simd_ctpop" | "core::intrinsics::simd::simd_ctpop" => {
            Builtin::SimdOp(SimdOp::Ctpop)
        }
        "std::intrinsics::simd::simd_cttz" | "core::intrinsics::simd::simd_cttz" => {
            Builtin::SimdOp(SimdOp::Cttz)
        }
        "std::intrinsics::simd::simd_saturating_add"
        | "core::intrinsics::simd::simd_saturating_add" => {
            Builtin::SimdOp(SimdOp::SaturatingAdd)
        }
        "std::intrinsics::simd::simd_saturating_sub"
        | "core::intrinsics::simd::simd_saturating_sub" => {
            Builtin::SimdOp(SimdOp::SaturatingSub)
        }
        "std::intrinsics::simd::simd_carryless_mul"
        | "core::intrinsics::simd::simd_carryless_mul" => Builtin::SimdOp(SimdOp::CarrylessMul),
        "std::intrinsics::simd::simd_funnel_shl" | "core::intrinsics::simd::simd_funnel_shl" => {
            Builtin::SimdOp(SimdOp::FunnelShl)
        }
        "std::intrinsics::simd::simd_funnel_shr" | "core::intrinsics::simd::simd_funnel_shr" => {
            Builtin::SimdOp(SimdOp::FunnelShr)
        }
        "std::intrinsics::simd::simd_maximum_number_nsz"
        | "core::intrinsics::simd::simd_maximum_number_nsz" => {
            Builtin::SimdOp(SimdOp::MaximumNumber)
        }
        "std::intrinsics::simd::simd_minimum_number_nsz"
        | "core::intrinsics::simd::simd_minimum_number_nsz" => {
            Builtin::SimdOp(SimdOp::MinimumNumber)
        }
        "std::intrinsics::simd::simd_splat" | "core::intrinsics::simd::simd_splat" => {
            Builtin::SimdOp(SimdOp::Splat)
        }
        "std::intrinsics::simd::simd_extract" | "core::intrinsics::simd::simd_extract" => {
            Builtin::SimdOp(SimdOp::Extract)
        }
        "std::intrinsics::simd::simd_extract_dyn" | "core::intrinsics::simd::simd_extract_dyn" => {
            Builtin::SimdOp(SimdOp::Extract)
        }
        "std::intrinsics::simd::simd_insert" | "core::intrinsics::simd::simd_insert" => {
            Builtin::SimdOp(SimdOp::Insert)
        }
        "std::intrinsics::simd::simd_insert_dyn" | "core::intrinsics::simd::simd_insert_dyn" => {
            Builtin::SimdOp(SimdOp::Insert)
        }
        "std::intrinsics::simd::simd_shuffle" | "core::intrinsics::simd::simd_shuffle" => {
            Builtin::SimdOp(SimdOp::Shuffle)
        }
        "std::intrinsics::simd::simd_cast" | "core::intrinsics::simd::simd_cast" => {
            Builtin::SimdOp(SimdOp::Cast)
        }
        "std::intrinsics::simd::simd_as" | "core::intrinsics::simd::simd_as" => {
            Builtin::SimdOp(SimdOp::Cast)
        }
        "std::intrinsics::simd::simd_bitmask" | "core::intrinsics::simd::simd_bitmask" => {
            Builtin::SimdOp(SimdOp::Bitmask)
        }
        "std::intrinsics::simd::simd_select" | "core::intrinsics::simd::simd_select" => {
            Builtin::SimdOp(SimdOp::Select)
        }
        "std::intrinsics::simd::simd_select_bitmask"
        | "core::intrinsics::simd::simd_select_bitmask" => Builtin::SimdOp(SimdOp::SelectBitmask),
        "std::intrinsics::simd::simd_gather" | "core::intrinsics::simd::simd_gather" => {
            Builtin::SimdOp(SimdOp::Gather)
        }
        "std::intrinsics::simd::simd_scatter" | "core::intrinsics::simd::simd_scatter" => {
            Builtin::SimdOp(SimdOp::Scatter)
        }
        "std::intrinsics::simd::simd_masked_load" | "core::intrinsics::simd::simd_masked_load" => {
            Builtin::SimdOp(SimdOp::MaskedLoad)
        }
        "std::intrinsics::simd::simd_masked_store"
        | "core::intrinsics::simd::simd_masked_store" => Builtin::SimdOp(SimdOp::MaskedStore),
        "std::intrinsics::simd::simd_arith_offset"
        | "core::intrinsics::simd::simd_arith_offset" => Builtin::SimdOp(SimdOp::ArithOffset),
        "std::intrinsics::simd::simd_reduce_add_ordered"
        | "core::intrinsics::simd::simd_reduce_add_ordered" => {
            Builtin::SimdOp(SimdOp::ReduceAddOrdered)
        }
        "std::intrinsics::simd::simd_reduce_add_unordered"
        | "core::intrinsics::simd::simd_reduce_add_unordered" => {
            Builtin::SimdOp(SimdOp::ReduceAddUnordered)
        }
        "std::intrinsics::simd::simd_reduce_mul_ordered"
        | "core::intrinsics::simd::simd_reduce_mul_ordered" => {
            Builtin::SimdOp(SimdOp::ReduceMulOrdered)
        }
        "std::intrinsics::simd::simd_reduce_mul_unordered"
        | "core::intrinsics::simd::simd_reduce_mul_unordered" => {
            Builtin::SimdOp(SimdOp::ReduceMulUnordered)
        }
        "std::intrinsics::simd::simd_reduce_max" | "core::intrinsics::simd::simd_reduce_max" => {
            Builtin::SimdOp(SimdOp::ReduceMax)
        }
        "std::intrinsics::simd::simd_reduce_min" | "core::intrinsics::simd::simd_reduce_min" => {
            Builtin::SimdOp(SimdOp::ReduceMin)
        }
        "std::intrinsics::simd::simd_reduce_and" | "core::intrinsics::simd::simd_reduce_and" => {
            Builtin::SimdOp(SimdOp::ReduceAnd)
        }
        "std::intrinsics::simd::simd_reduce_or" | "core::intrinsics::simd::simd_reduce_or" => {
            Builtin::SimdOp(SimdOp::ReduceOr)
        }
        "std::intrinsics::simd::simd_reduce_xor" | "core::intrinsics::simd::simd_reduce_xor" => {
            Builtin::SimdOp(SimdOp::ReduceXor)
        }
        "std::intrinsics::simd::simd_reduce_all" | "core::intrinsics::simd::simd_reduce_all" => {
            Builtin::SimdOp(SimdOp::ReduceAll)
        }
        "std::intrinsics::simd::simd_reduce_any" | "core::intrinsics::simd::simd_reduce_any" => {
            Builtin::SimdOp(SimdOp::ReduceAny)
        }
        "std::intrinsics::simd::simd_fma" | "core::intrinsics::simd::simd_fma" => {
            Builtin::SimdOp(SimdOp::Fma)
        }
        "std::intrinsics::simd::simd_relaxed_fma" | "core::intrinsics::simd::simd_relaxed_fma" => {
            Builtin::SimdOp(SimdOp::Fma)
        }
        "std::intrinsics::simd::simd_cast_ptr" | "core::intrinsics::simd::simd_cast_ptr" => {
            Builtin::SimdOp(SimdOp::CastPtr)
        }
        "std::intrinsics::simd::simd_expose_provenance"
        | "core::intrinsics::simd::simd_expose_provenance" => {
            Builtin::SimdOp(SimdOp::ExposeProvenance)
        }
        "std::intrinsics::simd::simd_with_exposed_provenance"
        | "core::intrinsics::simd::simd_with_exposed_provenance" => {
            Builtin::SimdOp(SimdOp::WithExposedProvenance)
        }
        "core::num::imp::overflow_panic::pow" => Builtin::IntPow,
        "std::str::FromStr::from_str" | "core::str::FromStr::from_str" => {
            Builtin::FromStrParse
        }
        "alloc::alloc::__rust_alloc" => Builtin::RustAlloc { zeroed: false },
        "alloc::alloc::__rust_alloc_zeroed" => Builtin::RustAlloc { zeroed: true },
        "alloc::alloc::__rust_dealloc" => Builtin::RustDealloc,
        "alloc::alloc::__rust_realloc" => Builtin::RustRealloc,
        "libc::unix::malloc" => Builtin::LibcMalloc,
        "libc::unix::calloc" => Builtin::LibcCalloc,
        "libc::unix::free" => Builtin::LibcFree,
        "libc::unix::realloc" => Builtin::LibcRealloc,
        "libc::unix::posix_memalign" => Builtin::LibcPosixMemalign,
        "libc::sigaction" | "libc::unix::sigaction" => Builtin::LibcSigaction,
        "libc::raise" | "libc::unix::raise" => Builtin::LibcRaise,
        "libc::atexit" | "libc::unix::atexit" => Builtin::LibcAtexit,
        "libc::read" | "libc::unix::read" => Builtin::LibcRead,
        "libc::write" | "libc::unix::write" => Builtin::LibcWrite,
        "std::sys::io::error::unix::errno_location"
        | "libc::__errno_location"
        | "libc::unix::__errno_location" => Builtin::ErrnoLocation,
        _ if canonical.starts_with("std::rc::Rc") && canonical.ends_with(">::new") => {
            Builtin::RcNew
        }
        _ if canonical.starts_with("std::sync::Arc") && canonical.ends_with(">::new") => {
            Builtin::ArcNew
        }
        _ if canonical.starts_with("std::rc::Rc") && canonical.ends_with(">::from_raw") => {
            Builtin::RcFromRaw
        }
        _ if canonical.starts_with("std::sync::Arc") && canonical.ends_with(">::from_raw") => {
            Builtin::ArcFromRaw
        }
        _ if (canonical.starts_with("std::rc::Weak")
            || canonical.starts_with("std::sync::Weak"))
            && canonical.ends_with(">::from_raw") =>
        {
            Builtin::WeakFromRaw
        }
        _ => return unicode_lookup_slow(canonical).or_else(|| float_method(canonical)),
    })
}

/// The `(element tag, inline capacity)` of a
/// `smallvec::SmallVec<[T; N]>` self-type tag, or `None` when
/// the tag does not denote a SmallVec over an array (the
/// `Array` implementation in use).
/// The tag may carry the naming crate's prefix
/// (`<caller>::smallvec::SmallVec<[...]>`), as rendered tags do
/// when the type is named through an explicit `extern crate`
/// item; from inside the crate's ordinary consumers the bare
/// path is spelled. Both spellings denote the same ADT.
pub(super) fn smallvec_array_arg(self_ty: &str) -> Option<(String, usize)> {
    let args = if let Some(rest) = self_ty.strip_prefix("smallvec::SmallVec<") {
        rest
    } else {
        let (head, tail) = self_ty.split_once("::")?;
        if head.is_empty()
            || !head
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return None;
        }
        tail.strip_prefix("smallvec::SmallVec<")?
    };
    // `args` is the array argument plus the closing `>` of the
    // SmallVec argument list: `[T; N]>`.
    if !args.starts_with('[') || !args.ends_with('>') {
        return None;
    }
    let bytes = args.as_bytes();
    let mut depth = 0i64;
    let mut close = None;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'[' | b'<' => depth += 1,
            b']' | b'>' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    if close != args.len() - 2 {
        return None;
    }
    let inner = &args[1..close];
    let mut depth = 0i64;
    let mut semi = None;
    for (i, &b) in inner.as_bytes().iter().enumerate() {
        match b {
            b'[' | b'<' => depth += 1,
            b']' | b'>' => depth -= 1,
            b';' if depth == 0 => semi = Some(i),
            _ => {}
        }
    }
    let semi = semi?;
    let cap: usize = inner[semi + 1..].trim().parse().ok()?;
    Some((inner[..semi].trim().to_string(), cap))
}

/// The value of one scalar element of type `tag` whose
/// storage bytes are `bytes` (little-endian, the element's own
/// width). `None` for a tag that names no scalar element type.
pub(super) fn scalar_elem_value(tag: &str, bytes: &[u8]) -> Option<Value> {
    let mut buf = [0u8; 16];
    buf[..bytes.len()].copy_from_slice(bytes);
    let bits = u128::from_le_bytes(buf);
    Some(match tag {
        "u8" => Value::Uint(bits & 0xff),
        "u16" => Value::Uint(bits & 0xffff),
        "u32" => Value::Uint(bits & 0xffff_ffff),
        "u64" | "usize" => Value::Uint(bits & 0xffff_ffff_ffff_ffff),
        "u128" => Value::Uint(bits),
        "i8" => Value::Int(i128::from(bits as u8 as i8)),
        "i16" => Value::Int(i128::from(bits as u16 as i16)),
        "i32" => Value::Int(i128::from(bits as u32 as i32)),
        "i64" | "isize" => Value::Int(i128::from(bits as u64 as i64)),
        "i128" => Value::Int(bits as i128),
        "bool" => Value::Bool(bytes.first().copied().unwrap_or(0) != 0),
        "char" => Value::Uint(bits & 0xffff_ffff),
        "f16" => Value::Float(crate::mir_convert::f16_to_f64(bits as u16)),
        "f32" => {
            Value::Float(crate::f128::f32_bits_to_f64_exact(bits as u32))
        }
        "f64" => Value::Float(f64::from_bits(bits as u64)),
        "f128" => Value::Float128(bits),
        _ => return None,
    })
}

/// The `usize` a scalar value holds, when it holds one.
pub(super) fn value_as_usize(value: &Value) -> Option<usize> {
    match value {
        Value::Uint(n) => usize::try_from(*n).ok(),
        Value::Int(n) => usize::try_from(*n).ok(),
        _ => None,
    }
}

/// The `(pointer, length)` a spilled SmallVec's data word
/// holds: the enum form's `Heap` variant carries the pair as
/// its two fields; the union form's heap field (index 1) is
/// the pair itself, a two-element tuple. The pointer sits
/// inside a `NonNull` one-field wrapper.
pub(super) fn smallvec_heap_parts(data: &Value) -> Option<(Value, usize)> {
    let Value::Enum {
        variant, fields, ..
    } = data
    else {
        return None;
    };
    let (ptr_field, len_field) = if *variant == 1 && fields.len() == 2 {
        (fields[0].clone(), fields[1].clone())
    } else {
        match fields.get(1) {
            Some(Value::Tuple(pair)) if pair.len() == 2 => {
                (pair[0].clone(), pair[1].clone())
            }
            Some(Value::Enum { fields: pair, .. }) if pair.len() == 2 => {
                (pair[0].clone(), pair[1].clone())
            }
            _ => return None,
        }
    };
    let mut ptr = ptr_field;
    for _ in 0..2 {
        match &ptr {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                ptr = fields[0].clone();
            }
            _ => break,
        }
    }
    if !matches!(ptr, Value::Ref(_)) {
        return None;
    }
    let len = value_as_usize(&len_field)?;
    Some((ptr, len))
}

/// Look a canonical callee name up in the registry, tolerating
/// a crate-qualified spelling. A callee reached through an
/// extern crate the referencing crate links (and possibly
/// renames) itself carries that crate's name in front of the
/// defining path -- `vec_macro_no_std::alloc::alloc::__rust_alloc`
/// for a `#![no_std]` crate with `extern crate alloc`, or
/// `krate::other::alloc::handle_alloc_error` through
/// `extern crate std as other`. The item is the same one the
/// unprefixed spelling names, so when the full spelling matches
/// nothing, retry with leading path segments removed, one at a
/// time. Names that match as spelled are unaffected, and a
/// collected body still wins over a builtin wherever the driver
/// collects one. All builtin-name consumers go through this
/// entry point; [`lookup`] is the exact-spelling registry
/// behind it.
pub fn lookup_qualified(canonical: &str) -> Option<Builtin> {
    if let Some(kind) = lookup(canonical) {
        return Some(kind);
    }
    let mut rest = canonical;
    while let Some(pos) = leading_segment_end(rest) {
        rest = &rest[pos..];
        if let Some(kind) = lookup(rest) {
            return Some(kind);
        }
    }
    None
}

/// Resolve a `core::unicode::unicode_data` skiplist lookup
/// callee (`<module>::lookup_slow`) to its builtin. The
/// property modules are core's generated tables; only the
/// modules whose lookup has the skiplist shape are modelled
/// (see [`UnicodeProperty`]), and a name in any other shape
/// stays an explicit unsupported call.
fn unicode_lookup_slow(canonical: &str) -> Option<Builtin> {
    let module = canonical.strip_suffix("::lookup_slow")?;
    let property = match module {
        "core::unicode::unicode_data::alphabetic" => UnicodeProperty::Alphabetic,
        "core::unicode::unicode_data::case_ignorable" => UnicodeProperty::CaseIgnorable,
        "core::unicode::unicode_data::cf" => UnicodeProperty::Cf,
        "core::unicode::unicode_data::cn_planes_0_3" => UnicodeProperty::CnPlanes03,
        "core::unicode::unicode_data::default_ignorable_code_point" => {
            UnicodeProperty::DefaultIgnorableCodePoint
        }
        "core::unicode::unicode_data::grapheme_extend" => UnicodeProperty::GraphemeExtend,
        "core::unicode::unicode_data::lt" => UnicodeProperty::Lt,
        "core::unicode::unicode_data::n" => UnicodeProperty::N,
        _ => return None,
    };
    Some(Builtin::UnicodeLookupSlow(property))
}

/// Resolve a primitive float method callee
/// (`core::f64::<impl f64>::sqrt`, `std::f32::<impl f32>::floor`,
/// ...) to its builtin. The methods live partly in `core` and partly
/// in `std`; both prefixes name the same inherent impls, so either is
/// accepted for every modelled method. A float method this registry
/// does not model stays an explicit unsupported call.
fn float_method(canonical: &str) -> Option<Builtin> {
    if let Some(rest) = canonical
        .strip_prefix("core::f128::<impl f128>::")
        .or_else(|| canonical.strip_prefix("std::f128::<impl f128>::"))
    {
        return float128_method(rest);
    }
    if let Some(rest) = canonical
        .strip_prefix("core::f16::<impl f16>::")
        .or_else(|| canonical.strip_prefix("std::f16::<impl f16>::"))
    {
        return float_method_named(rest).map(|op| Builtin::Float16Method { op });
    }
    let (method, f32) = if let Some(rest) = canonical
        .strip_prefix("core::f32::<impl f32>::")
        .or_else(|| canonical.strip_prefix("std::f32::<impl f32>::"))
    {
        (rest, true)
    } else if let Some(rest) = canonical
        .strip_prefix("core::f64::<impl f64>::")
        .or_else(|| canonical.strip_prefix("std::f64::<impl f64>::"))
    {
        (rest, false)
    } else {
        return None;
    };
    let op = float_method_named(method)?;
    Some(Builtin::FloatMethod { f32, op })
}

/// The shared inherent-method name table for the float types.
fn float_method_named(method: &str) -> Option<FloatOp> {
    use FloatOp as O;
    let op = match method {
        "is_nan" => O::IsNan,
        "is_infinite" => O::IsInfinite,
        "is_finite" => O::IsFinite,
        "is_sign_positive" => O::IsSignPositive,
        "is_sign_negative" => O::IsSignNegative,
        "abs" => O::Abs,
        "sqrt" => O::Sqrt,
        "floor" => O::Floor,
        "ceil" => O::Ceil,
        "round" => O::Round,
        "round_ties_even" => O::RoundTiesEven,
        "trunc" => O::Trunc,
        "fract" => O::Fract,
        "recip" => O::Recip,
        "signum" => O::Signum,
        "to_degrees" => O::ToDegrees,
        "to_radians" => O::ToRadians,
        "powi" => O::Powi,
        "powf" => O::Powf,
        "min" => O::Min,
        "max" => O::Max,
        "exp" => O::Exp,
        "exp2" => O::Exp2,
        "exp_m1" => O::ExpM1,
        "ln" => O::Ln,
        "ln_1p" => O::Ln1p,
        "log" => O::Log,
        "log2" => O::Log2,
        "log10" => O::Log10,
        "sin" => O::Sin,
        "cos" => O::Cos,
        "tan" => O::Tan,
        "asin" => O::Asin,
        "acos" => O::Acos,
        "atan" => O::Atan,
        "atan2" => O::Atan2,
        "sin_cos" => O::SinCos,
        "sinh" => O::Sinh,
        "cosh" => O::Cosh,
        "tanh" => O::Tanh,
        "asinh" => O::Asinh,
        "acosh" => O::Acosh,
        "atanh" => O::Atanh,
        "cbrt" => O::Cbrt,
        "hypot" => O::Hypot,
        "mul_add" => O::MulAdd,
        "div_euclid" => O::DivEuclid,
        "rem_euclid" => O::RemEuclid,
        "copysign" => O::CopySign,
        "to_bits" => O::ToBits,
        "from_bits" => O::FromBits,
        _ => return None,
    };
    Some(op)
}

/// Resolve an `f128` inherent method name to its builtin. Only
/// the operations the binary128 module implements exactly are
/// registered; the rest (e.g. `powf`, the trigonometric family)
/// stay explicit unsupported calls.
fn float128_method(method: &str) -> Option<Builtin> {
    use FloatOp as O;
    let op = match method {
        "is_nan" => O::IsNan,
        "is_infinite" => O::IsInfinite,
        "is_finite" => O::IsFinite,
        "is_sign_positive" => O::IsSignPositive,
        "is_sign_negative" => O::IsSignNegative,
        "abs" => O::Abs,
        "sqrt" => O::Sqrt,
        "floor" => O::Floor,
        "ceil" => O::Ceil,
        "round" => O::Round,
        "round_ties_even" => O::RoundTiesEven,
        "trunc" => O::Trunc,
        "fract" => O::Fract,
        "recip" => O::Recip,
        "signum" => O::Signum,
        "powi" => O::Powi,
        "min" => O::Min,
        "max" => O::Max,
        "to_bits" => O::ToBits,
        "from_bits" => O::FromBits,
        _ => return None,
    };
    Some(Builtin::Float128Method { op })
}
