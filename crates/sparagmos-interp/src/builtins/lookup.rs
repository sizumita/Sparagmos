//! Callee-name classification: mapping canonical names to builtin kinds.



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
///
/// The driver uses the same predicate from the other side: a
/// builtin-owned definition demanded *as a value* whose kind is
/// not servable here gets its real body collected instead (see
/// `value_demand_needs_body` in the driver's reach walk), so a
/// call through the value runs the body the compiler would have
/// run — drop glue and all — under the plain definition name.
pub fn runtime_name_fallback(kind: Builtin) -> bool {
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
        // The C string length, reached under its symbol
        // name by crates that declare it themselves under
        // a `#[link_name = "strlen"]` (the libc crate's own
        // declaration resolves through the qualified
        // registry instead).
        "strlen" => Builtin::LibcStrlen,
        // The process exit, reached under its versioned
        // symbol (`exit@GLIBC_2.2.5`) by raw-dylib
        // declarations; the version-stripped retry in the
        // conversion fallback lands here.
        "exit" => Builtin::ProcessExit,
        // The file-tree walk under its versioned symbols:
        // the 2.2.5 binding predates flag validation, the
        // 2.3.3 binding validates the flag word the way
        // current glibc does. The unversioned name serves
        // a plain `#[link_name = "nftw"]` declaration with
        // the current behaviour.
        "nftw@GLIBC_2.2.5" => Builtin::LibcNftwLegacy,
        "nftw@GLIBC_2.3.3" | "nftw" => Builtin::LibcNftw,
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
pub(crate) fn is_file_io_call(name: &str, generic_args: &[String]) -> bool {
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

/// Look a canonical callee name up in the registry.
///
/// The registry is dispersed over the builtin family modules:
/// each family owns a registration fragment (its `<family>_lookup`
/// function) holding one contiguous slice of the name table, and
/// this function chains the fragments in registry order, so the
/// first fragment that serves a name wins, exactly as the first
/// matching arm of the historical single match did. A name no
/// fragment serves falls to the two helper resolvers the table's
/// fallback arm always delegated to.
pub fn lookup(canonical: &str) -> Option<Builtin> {
    None
        .or_else(|| io_lookup(canonical))
        .or_else(|| env_lookup(canonical))
        .or_else(|| fs_lookup(canonical))
        .or_else(|| ffi_lookup(canonical))
        .or_else(|| fs_sys_lookup(canonical))
        .or_else(|| net_lookup(canonical))
        .or_else(|| process_exit_lookup(canonical))
        .or_else(|| io_stream_lookup(canonical))
        .or_else(|| fmt_lookup(canonical))
        .or_else(|| iterators_lookup(canonical))
        .or_else(|| sync_lookup(canonical))
        .or_else(|| backtrace_lookup(canonical))
        .or_else(|| thread_lookup(canonical))
        .or_else(|| time_lookup(canonical))
        .or_else(|| direct_intrinsic_lookup(canonical))
        .or_else(|| harness_lookup(canonical))
        .or_else(|| panic_hook_lookup(canonical))
        .or_else(|| time_now_lookup(canonical))
        .or_else(|| hash_lookup(canonical))
        .or_else(|| command_lookup(canonical))
        .or_else(|| io_error_lookup(canonical))
        .or_else(|| command_api_lookup(canonical))
        .or_else(|| raw_fd_lookup(canonical))
        .or_else(|| time_arith_lookup(canonical))
        .or_else(|| thread_api_lookup(canonical))
        .or_else(|| spin_lookup(canonical))
        .or_else(|| mpsc_lookup(canonical))
        .or_else(|| from_lookup(canonical))
        .or_else(|| waker_lookup(canonical))
        .or_else(|| deref_lookup(canonical))
        .or_else(|| path_lookup(canonical))
        .or_else(|| strings_lookup(canonical))
        .or_else(|| panic_lookup(canonical))
        .or_else(|| str_search_lookup(canonical))
        .or_else(|| catch_unwind_lookup(canonical))
        .or_else(|| equality_lookup(canonical))
        .or_else(|| defaults_lookup(canonical))
        .or_else(|| string_lookup(canonical))
        .or_else(|| alloc_shim_lookup(canonical))
        .or_else(|| as_ref_lookup(canonical))
        .or_else(|| vec_boxed_lookup(canonical))
        .or_else(|| string_mut_lookup(canonical))
        .or_else(|| str_lookup(canonical))
        .or_else(|| into_lookup(canonical))
        .or_else(|| box_lookup(canonical))
        .or_else(|| vec_lookup(canonical))
        .or_else(|| iter_guard_lookup(canonical))
        .or_else(|| str_predicate_lookup(canonical))
        .or_else(|| iter_misc_lookup(canonical))
        .or_else(|| cell_lookup(canonical))
        .or_else(|| pointer_lookup(canonical))
        .or_else(|| primitives_lookup(canonical))
        .or_else(|| atomics_lookup(canonical))
        .or_else(|| assert_intrinsics_lookup(canonical))
        .or_else(|| overflow_checks_lookup(canonical))
        .or_else(|| volatile_lookup(canonical))
        .or_else(|| ptr_lookup(canonical))
        .or_else(|| simd_lookup(canonical))
        .or_else(|| num_pow_lookup(canonical))
        .or_else(|| slice_sort_lookup(canonical))
        .or_else(|| int_inherent_lookup(canonical))
        .or_else(|| from_str_lookup(canonical))
        .or_else(|| alloc_lookup(canonical))
        .or_else(|| libc_alloc_lookup(canonical))
        .or_else(|| signals_lookup(canonical))
        .or_else(|| libc_io_lookup(canonical))
        .or_else(|| libc_sys_lookup(canonical))
        .or_else(|| f16_lookup(canonical))
        .or_else(|| rc_arc_lookup(canonical))
        .or_else(|| unicode_lookup_slow(canonical))
        .or_else(|| float_method(canonical))
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

/// Registration fragment of the builtin name registry: the caller-location and `va_arg` intrinsics.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn direct_intrinsic_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::intrinsics::caller_location" => Builtin::CallerLocation,
        "std::intrinsics::va_arg" | "core::intrinsics::va_arg" => Builtin::VaArg,
        _ => return None,
    })
}
