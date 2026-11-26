//! The io::Error value representation and error-kind tables.



use super::*;
use crate::value::{RefTarget, Value};


/// The name the model gives the drop functions an `io::Error`
/// `Custom` payload stores beside its trait object
/// (`alloc::io::error::custom_owner_from_box`'s nested
/// `drop_box_raw`, in its two instantiations). No converted
/// body exists for them (the enclosing function has none), so
/// a call through one of these pointers is an explicit error
/// unless the drop machinery has already accounted for the
/// payload — the packed representation's own drop plan runs
/// the payload's glue directly (see `DropPlan::IoErrorRepr`).
pub(super) const IO_ERROR_DROP_BOX_RAW_NAME: &str =
    "alloc::io::error::custom_owner_from_box::drop_box_raw";

/// The `Repr`-level value of an `io::Error` in the bitpacked
/// layout: the two inner nestings of [`io_error_value`] (the
/// representation struct around the `NonNull` around the tagged
/// word).
pub(super) fn io_repr_value(tagged_word: u64) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![Value::Ref(RefTarget::Address(tagged_word))],
        }],
    }
}

/// The allocation a `Custom` owner value names: unwrap the
/// one-field aggregates (the owner around its `NonNull`, the
/// `NonNull` around its pointer) down to the allocation
/// pointer.
pub(super) fn io_owner_alloc_id(value: &Value) -> Option<u64> {
    match value {
        Value::Ref(RefTarget::Alloc { id, projection }) if projection.is_empty() => Some(*id),
        Value::Enum { fields, .. } if fields.len() == 1 => io_owner_alloc_id(&fields[0]),
        _ => None,
    }
}

/// The address an owner-shaped value carries when its pointer
/// is an address-only one (the form a guest-reconstructed owner
/// takes after the packed word was decoded back): unwrap the
/// one-field aggregates down to the address.
pub(super) fn io_pointer_address(value: &Value) -> Option<u64> {
    match value {
        Value::Ref(RefTarget::Address(addr)) => Some(*addr),
        Value::Enum { fields, .. } if fields.len() == 1 => io_pointer_address(&fields[0]),
        _ => None,
    }
}

/// An `std::io::Error` value in the bitpacked `Repr` layout the
/// platform's `core::io::error` uses: the error is a one-field
/// struct whose field is the tagged word (a `NonNull<()>`),
/// with the payload in the word's top 32 bits over a two-bit
/// tag (`0b10` for an OS error, `0b11` for a bare `ErrorKind`).
fn io_error_value(tagged_word: u128) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![Value::Enum {
                variant: 0,
                fields: vec![Value::Ref(RefTarget::Address(tagged_word as u64))],
            }],
        }],
    }
}

/// An `std::io::Error` holding OS error `code` (an errno).
pub(super) fn io_os_error_value(code: i32) -> Value {
    io_error_value(((code as u32 as u128) << 32) | 0b10)
}

/// An `std::io::Error` holding the bare `ErrorKind` with the
/// given discriminant (the shape of std's `const_error!`).
pub(super) fn io_simple_error_value(kind_discriminant: u32) -> Value {
    io_error_value(((kind_discriminant as u128) << 32) | 0b11)
}

/// The tagged word an `std::io::Error` value in the bitpacked
/// representation carries: the address-only reference at the
/// bottom of the `Error { repr: Repr(NonNull) }` nesting. The
/// nesting's other leaves are zero-sized markers (the
/// `PhantomData` field a guest-built value carries), so the
/// word is the value tree's unique address leaf; a tree with
/// zero or several address leaves is not this representation.
pub(crate) fn io_error_tagged_word(value: &Value) -> Option<u64> {
    match value {
        Value::Ref(RefTarget::Address(word)) => Some(*word),
        Value::Enum { fields, .. } | Value::Tuple(fields) => {
            let mut found = None;
            for field in fields {
                if let Some(word) = io_error_tagged_word(field) {
                    if found.is_some() {
                        return None;
                    }
                    found = Some(word);
                }
            }
            found
        }
        _ => None,
    }
}

/// The `(kind, message)` an `std::io::Error` value in the
/// `SimpleMessage` representation carries, when the value holds
/// it in the shape guest code builds: the `&'static
/// SimpleMessage` was stored through the repr's type-erased
/// `NonNull<()>` slot, so the value tree's unique erased-constant
/// leaf is the message struct's allocation, preserved exactly by
/// the driver (see `Value::ErasedBlock`). A tree with zero or
/// several erased leaves is not this representation.
pub(super) fn io_error_simple_message(value: &Value) -> Option<(u32, String)> {
    let (bytes, relocs) = erased_constant_leaf(value)?;
    decode_simple_message(bytes, relocs)
}

/// The unique type-erased constant block at the bottom of a
/// value tree (the same uniqueness discipline
/// `io_error_tagged_word` applies to address leaves): the block
/// a reference denotes when its declared pointee type was
/// zero-sized and the driver preserved the allocation verbatim.
fn erased_constant_leaf(value: &Value) -> Option<(&[Option<u8>], &[(u32, Value)])> {
    match value {
        Value::Ref(RefTarget::Value(inner, _)) => match inner.as_ref() {
            Value::ErasedBlock { bytes, relocs } => Some((bytes, relocs)),
            _ => None,
        },
        Value::Enum { fields, .. } | Value::Tuple(fields) => {
            let mut found = None;
            for field in fields {
                if let Some(leaf) = erased_constant_leaf(field) {
                    if found.is_some() {
                        return None;
                    }
                    found = Some(leaf);
                }
            }
            found
        }
        _ => None,
    }
}

/// Decode a `core::io::error::SimpleMessage` from its preserved
/// allocation bytes. The layout is the one the layout query
/// reports for the type on this target: the `message: &'static
/// str` fat pointer first (its data pointer is the relocation at
/// offset 0, its length the word at offset 8), then the `kind:
/// ErrorKind` discriminant byte at offset 16, in a 24-byte
/// block. Every field is cross-checked (the length against the
/// string block's byte count, the discriminant against the kind
/// table, the bytes as UTF-8), so a block that is not a
/// `SimpleMessage` in this layout does not decode.
fn decode_simple_message(bytes: &[Option<u8>], relocs: &[(u32, Value)]) -> Option<(u32, String)> {
    if bytes.len() != 24 {
        return None;
    }
    let mut len_word = [0u8; 8];
    for (i, b) in bytes[8..16].iter().enumerate() {
        len_word[i] = (*b)?;
    }
    let len = u64::from_le_bytes(len_word);
    let kind = bytes[16]? as u32;
    error_kind_name(kind)?;
    let (_, data_ref) = relocs.iter().find(|(offset, _)| *offset == 0)?;
    let data = erased_constant_bytes(data_ref)?;
    if data.len() as u64 != len {
        return None;
    }
    String::from_utf8(data).ok().map(|message| (kind, message))
}

/// The byte content a preserved-constant reference denotes: a
/// fully-initialised erased block (the shape of a string
/// literal's allocation), a byte string, or a string value.
fn erased_constant_bytes(value: &Value) -> Option<Vec<u8>> {
    match value {
        Value::Ref(RefTarget::Value(inner, _)) => erased_constant_bytes(inner),
        Value::ErasedBlock { bytes, relocs } if relocs.is_empty() => {
            bytes.iter().copied().collect()
        }
        Value::Bytes(bytes) => Some(bytes.clone()),
        Value::Str(text) => Some(text.clone().into_bytes()),
        _ => None,
    }
}

/// The platform's message text for an OS error code: the text
/// the C library's `strerror` gives for it, which is exactly
/// what std's native `format_os_error` writes. Read through the
/// host's own `io::Error` (whose `Display` is built from that
/// text), with the ` (os error {code})` suffix std's `Display`
/// adds removed.
pub(super) fn os_error_message(code: i32) -> String {
    let full = std::io::Error::from_raw_os_error(code).to_string();
    let suffix = format!(" (os error {code})");
    match full.strip_suffix(&suffix) {
        Some(message) => message.to_string(),
        None => full,
    }
}

/// The variant name of the `ErrorKind` with the given
/// discriminant (the declaration order `from_prim` decodes in
/// `core::io::error`); `None` for a discriminant no variant
/// carries.
pub(super) fn error_kind_name(discriminant: u32) -> Option<&'static str> {
    Some(match discriminant {
        0 => "NotFound",
        1 => "PermissionDenied",
        2 => "ConnectionRefused",
        3 => "ConnectionReset",
        4 => "HostUnreachable",
        5 => "NetworkUnreachable",
        6 => "ConnectionAborted",
        7 => "NotConnected",
        8 => "AddrInUse",
        9 => "AddrNotAvailable",
        10 => "NetworkDown",
        11 => "BrokenPipe",
        12 => "AlreadyExists",
        13 => "WouldBlock",
        14 => "NotADirectory",
        15 => "IsADirectory",
        16 => "DirectoryNotEmpty",
        17 => "ReadOnlyFilesystem",
        18 => "FilesystemLoop",
        19 => "StaleNetworkFileHandle",
        20 => "InvalidInput",
        21 => "InvalidData",
        22 => "TimedOut",
        23 => "WriteZero",
        24 => "StorageFull",
        25 => "NotSeekable",
        26 => "QuotaExceeded",
        27 => "FileTooLarge",
        28 => "ResourceBusy",
        29 => "ExecutableFileBusy",
        30 => "Deadlock",
        31 => "CrossesDevices",
        32 => "TooManyLinks",
        33 => "InvalidFilename",
        34 => "ArgumentListTooLong",
        35 => "Interrupted",
        36 => "Other",
        37 => "UnexpectedEof",
        38 => "Unsupported",
        39 => "OutOfMemory",
        40 => "InProgress",
        41 => "TooManyOpenFiles",
        42 => "InputOutputError",
        43 => "Uncategorized",
        _ => return None,
    })
}

/// The message text of the `ErrorKind` with the given
/// discriminant: core's `ErrorKind::as_str` table (the text its
/// `Display` writes), entry for entry.
pub(super) fn error_kind_message(discriminant: u32) -> Option<&'static str> {
    Some(match discriminant {
        0 => "entity not found",
        1 => "permission denied",
        2 => "connection refused",
        3 => "connection reset",
        4 => "host unreachable",
        5 => "network unreachable",
        6 => "connection aborted",
        7 => "not connected",
        8 => "address in use",
        9 => "address not available",
        10 => "network down",
        11 => "broken pipe",
        12 => "entity already exists",
        13 => "operation would block",
        14 => "not a directory",
        15 => "is a directory",
        16 => "directory not empty",
        17 => "read-only filesystem or storage medium",
        18 => "filesystem loop or indirection limit (e.g. symlink loop)",
        19 => "stale network file handle",
        20 => "invalid input parameter",
        21 => "invalid data",
        22 => "timed out",
        23 => "write zero",
        24 => "no storage space",
        25 => "seek on unseekable file",
        26 => "quota exceeded",
        27 => "file too large",
        28 => "resource busy",
        29 => "executable file busy",
        30 => "deadlock",
        31 => "cross-device link or rename",
        32 => "too many links",
        33 => "invalid filename",
        34 => "argument list too long",
        35 => "operation interrupted",
        36 => "other error",
        37 => "unexpected end of file",
        38 => "unsupported",
        39 => "out of memory",
        40 => "in progress",
        41 => "too many open files",
        42 => "input/output error",
        43 => "uncategorized error",
        _ => return None,
    })
}

/// One reference count of an `RcInner` / `ArcInner` aggregate
/// as the handle constructor builds it: a one-field wrapper
/// (the `Cell` / `AtomicUsize`) around a one-field wrapper
/// (the `UnsafeCell`) around the count itself.
pub(super) fn is_count_cell(value: &Value) -> bool {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                matches!(fields[0], Value::Uint(_) | Value::Int(_))
            }
            _ => false,
        },
        _ => false,
    }
}

/// A host `io::Error` as a guest `io::Error` value: the errno
/// form when the host error carries one (the faithful case for
/// filesystem operations), otherwise the kind-only form.
pub(super) fn host_io_error(error: &std::io::Error) -> Value {
    match error.raw_os_error() {
        Some(code) => io_os_error_value(code),
        None => io_simple_error_value(42),
    }
}

pub(super) fn io_result_ok(value: Value) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![value],
    }
}

pub(super) fn io_result_err(error: Value) -> Value {
    Value::Enum {
        variant: 1,
        fields: vec![error],
    }
}

pub(super) fn io_result_from_host(result: Result<Value, std::io::Error>) -> Value {
    match result {
        Ok(value) => io_result_ok(value),
        Err(error) => io_result_err(host_io_error(&error)),
    }
}

/// Registration fragment of the builtin name registry: the `io::Error` OS-function surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn io_error_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
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
        _ => return None,
    })
}
